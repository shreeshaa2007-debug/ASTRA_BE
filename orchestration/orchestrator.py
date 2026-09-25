"""The orchestrator — the hand-rolled state machine of architecture.md §3 /
agent-plan.md § Orchestration routing. No framework: an ordered list of steps
and one routing decision keyed on the compliance verdict.

    sense -> validate -> [event_sensed]
      -> inventory / sourcing / logistics agents -> [agents_assessed]
      -> build problem -> optimize -> [plan_optimized]
      -> compliance -> [compliance_checked]
           APPROVED   -> [plan_finalized]                            -> COMPLETED
           ESCALATED  -> [approval_requested]                        -> AWAITING_APPROVAL
                            approve() -> [plan_finalized]            -> COMPLETED
                            reject()  -> [plan_rejected_by_human]    -> REJECTED
           REJECTED   -> [replan_requested] -> back to the agents with the offenders
                         excluded, ONCE; a second rejection, or nothing to exclude,
                         -> [run_failed]

Every [bracketed] step is a world-state checkpoint and the only place state
changes; the store (not this class) enforces what each may do, so even a bug
here cannot finalize a plan compliance rejected or one that needed a human.

The plan's diagram runs the three agents concurrently. They can't be: Sourcing
is sized to Inventory's deficit and Logistics is asked about the ports
Sourcing's candidate suppliers ship from. Each takes milliseconds, so nothing
is lost by running them in dependency order.

`run()` blocks until the run finishes or reaches the human-approval wait; the
API layer (Phase 15) runs it in a background thread and reads progress from the
checkpoint history. It holds no per-run state on `self`, so runs on different
simulations may proceed concurrently.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Mapping, Optional

from pydantic import BaseModel, Field, ValidationError

from backend.agents.compliance import tools as compliance_tools
from backend.agents.sensing.agent import SensingAgent
from backend.monitoring.metrics import metrics
from backend.optimization import tools as optimization_tools
from backend.optimization.engine import OptimizationEngine
from backend.orchestration import adapters
from backend.schemas.optimization import OptimizationSolution
from backend.schemas.sensing import SensingResult
from backend.schemas.world_state import (
    ApprovalDecision,
    ApprovalStatus,
    ComplianceStatus,
    ForecastRecord,
    InventoryAssessment,
    SimulationStatus,
    WorldState,
)
from backend.services.world_state import (
    MAX_REPLANS,
    CheckpointViolationError,
    ConcurrentModificationError,
    InvalidTransitionError,
    WorldStateError,
    WorldStateStore,
    state_changes_for_event,
)

logger = logging.getLogger("resilientsc.orchestrator")

Outcome = Literal["COMPLETED", "AWAITING_APPROVAL", "FAILED", "NO_DISRUPTION", "SENSING_REJECTED", "SENSING_ERROR"]
_SENSING_OUTCOME: dict[str, Outcome] = {"NO_DISRUPTION": "NO_DISRUPTION", "REJECTED": "SENSING_REJECTED", "ERROR": "SENSING_ERROR"}


class StepRecord(BaseModel):
    name: str
    ok: bool
    duration_ms: float
    detail: str = ""


class RunOutcome(BaseModel):
    """How a run ended. FAILED is a real failure of the pipeline and leaves the
    simulation FAILED. The three SENSING_* / NO_DISRUPTION outcomes are not:
    nothing entered the state, so the simulation is still CREATED and can be run
    again with a different signal."""

    outcome: Outcome
    simulation_id: str
    status: SimulationStatus
    message: str
    error: Optional[str] = None
    sensing: Optional[SensingResult] = None
    steps: list[StepRecord] = Field(default_factory=list)
    state: WorldState


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Orchestrator:
    def __init__(
        self,
        store: WorldStateStore,
        sensing: SensingAgent,
        *,
        compliance_rules: dict | None = None,
        validate_plan: Callable[[dict], dict] | None = None,
        engine: OptimizationEngine | None = None,
        parameter_overrides: dict | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ):
        self.store = store
        self.sensing = sensing
        self._rules = compliance_rules
        # injectable so a test can force a verdict the real rules would not produce
        self._validate_plan = validate_plan or (lambda plan: compliance_tools.validate_plan(plan, self._rules))
        self._engine = engine
        self._overrides = dict(parameter_overrides or {})
        self._clock = clock

    # ------------------------------------------------------------------ run
    def run(
        self,
        simulation_id: str,
        signal: str | Mapping[str, Any],
        product_id: str,
        *,
        as_of_date: str | None = None,
        tariff_overrides: dict[str, float] | None = None,
    ) -> RunOutcome:
        """One disruption -> one plan for one product. `tariff_overrides` (ISO3 ->
        %) is how a scenario sets the tariffs in effect, since a tariff's amount
        is not part of a sensed event (brief §13). One product per run: the
        world state holds one plan (`current_plan`)."""
        steps: list[StepRecord] = []
        state = self.store.get(simulation_id)
        if state.status != SimulationStatus.CREATED:  # fail before spending an LLM call
            raise InvalidTransitionError(f"simulation {simulation_id!r} is {state.status.value}; a run needs a CREATED simulation (reset it to run again)")

        sensed: SensingResult = self._step(steps, "sense", lambda: self.sensing.sense(signal), lambda r: r.status)
        if sensed.status != "EVENT":
            return self._outcome(_SENSING_OUTCOME[sensed.status], simulation_id, "; ".join(sensed.errors) or sensed.rationale or sensed.status, steps, sensed)

        try:
            return self._pipeline(state, sensed, str(product_id), as_of_date, tariff_overrides, steps)
        except (WorldStateError, ConcurrentModificationError):
            raise  # someone else moved the simulation, or a rule was broken: not this run's failure to record
        except Exception as exc:  # noqa: BLE001 — a run must end in a recorded state, never hang in RUNNING
            logger.exception("run %s failed unexpectedly", simulation_id)
            return self._fail(simulation_id, f"unexpected error: {type(exc).__name__}: {exc}", steps, sensed)

    def _pipeline(
        self, state: WorldState, sensed: SensingResult, product_id: str, as_of_date: str | None, tariff_overrides: dict[str, float] | None,
        steps: list[StepRecord],
    ) -> RunOutcome:
        changes = state_changes_for_event(state, sensed.event)
        if tariff_overrides:
            changes["tariffs"] = {**state.tariffs, **tariff_overrides}
        state = self._commit(state, "event_sensed", changes)

        extra_suppliers: frozenset[str] = frozenset()  # what a replan must leave out, beyond what the state says is disrupted
        extra_routes: frozenset[str] = frozenset()
        replans = 0
        while True:
            replanning = replans > 0
            outputs = self._step(
                steps, "agents (replan)" if replanning else "agents",
                lambda: optimization_tools.gather_agent_outputs(
                    product_id, as_of_date, state.disrupted_route_ids() | extra_routes, state.disrupted_supplier_ids() | extra_suppliers,
                    state.tariffs, self._overrides),
                lambda o: f"deficit {o.gross_deficit}, {len(o.sourcing_result['candidates_considered'])} suppliers, {len(o.logistics_results)} lanes",
            )
            if not replanning:  # a replan changes suppliers and routes, not what the Inventory Agent found
                state = self._commit(state, "agents_assessed", self._assessment(outputs))

            solution: OptimizationSolution = self._step(
                steps, "optimize",
                lambda: optimization_tools.optimize_supply_chain(optimization_tools.build_optimization_problem(
                    product_id, outputs.inventory_results, outputs.sourcing_result, outputs.logistics_results,
                    outputs.safety_stock_by_warehouse, outputs.parameters), self._engine),
                lambda s: s.status,
            )
            metrics.inc("optimizations_total", {"status": solution.status})
            if isinstance(solution.solver.get("solve_time_ms"), (int, float)):
                metrics.observe("optimization_solve_ms", solution.solver["solve_time_ms"])
            state = self._commit(state, "plan_optimized", {"current_plan": solution})  # recorded even if infeasible, for its diagnosis
            if solution.status != "OPTIMAL":
                return self._fail(state.simulation_id, self._optimization_error(solution, replanning), steps, sensed)

            verdict = self._step(
                steps, "compliance",
                lambda: self._validate_plan(adapters.plan_to_compliance_input(
                    solution, product_id, state.disrupted_route_ids() | extra_routes, state.disrupted_supplier_ids() | extra_suppliers)),
                lambda v: v["status"],
            )
            metrics.inc("compliance_verdicts_total", {"status": verdict["status"]})
            state = self._commit(state, "compliance_checked", {"compliance_status": ComplianceStatus(**verdict)})

            # ---- the one routing decision, keyed on the compliance status ----
            if verdict["status"] == "APPROVED":
                state = self._commit(state, "plan_finalized", {"approval_status": ApprovalStatus.NOT_REQUIRED})
                return self._outcome("COMPLETED", state.simulation_id, f"plan finalized automatically ({verdict['reason']})", steps, sensed)

            if verdict["status"] == "ESCALATED":
                state = self._commit(state, "approval_requested", {"approval_status": ApprovalStatus.PENDING})
                return self._outcome("AWAITING_APPROVAL", state.simulation_id, f"waiting for a human: {verdict['reason']}", steps, sensed)

            # REJECTED
            if replans >= MAX_REPLANS:
                return self._fail(state.simulation_id, f"rejected_after_replan: {verdict['reason']}", steps, sensed)
            suppliers, routes = adapters.exclusions_from_verdict(verdict["checks"], solution.problem)
            if not (suppliers or routes):
                return self._fail(
                    state.simulation_id,
                    f"compliance_rejected: {verdict['reason']} (the verdict names no supplier or route to exclude, so a replan would reproduce the same plan)",
                    steps, sensed)
            logger.info("compliance rejected the plan (%s); replanning without suppliers=%s routes=%s", verdict["reason"], sorted(suppliers), sorted(routes))
            metrics.inc("replans_total")
            state = self._commit(state, "replan_requested", {"current_plan": None, "compliance_status": None, "replan_count": state.replan_count + 1})
            extra_suppliers, extra_routes, replans = extra_suppliers | suppliers, extra_routes | routes, replans + 1

    # ------------------------------------------------- the human's decision
    def approve(self, simulation_id: str, decided_by: str, note: str = "", *, expected_version: int | None = None) -> WorldState:
        """A named human approves the escalated plan -> COMPLETED. Pass the
        `expected_version` you were shown, so approving a plan that has since
        changed fails instead of approving something you never saw."""
        return self._decide(simulation_id, "plan_finalized", ApprovalStatus.APPROVED, decided_by, note, expected_version)

    def reject(self, simulation_id: str, decided_by: str, note: str = "", *, expected_version: int | None = None) -> WorldState:
        """A named human rejects the escalated plan -> REJECTED (terminal)."""
        return self._decide(simulation_id, "plan_rejected_by_human", ApprovalStatus.REJECTED, decided_by, note, expected_version)

    def _decide(self, simulation_id: str, checkpoint: str, status: ApprovalStatus, decided_by: str, note: str, expected_version: int | None) -> WorldState:
        try:
            decision = ApprovalDecision(decided_by=decided_by.strip(), decided_at=self._clock(), note=note)
        except ValidationError as exc:
            raise CheckpointViolationError("a decision needs a named decider (decided_by)") from exc
        state = self.store.commit(
            simulation_id, checkpoint, {"approval_status": status, "approval_decision": decision},
            actor=decision.decided_by, expected_version=expected_version)
        metrics.inc("approvals_total", {"decision": status.value.lower()})
        logger.info("simulation %s: %s by %s", simulation_id, status.value, decision.decided_by,
                    extra={"event": "human_decision", "decision": status.value, "decided_by": decision.decided_by, "simulation_id": simulation_id})
        return state

    # ---------------------------------------------------------------- pieces
    def _commit(self, state: WorldState, checkpoint: str, changes: dict) -> WorldState:
        return self.store.commit(state.simulation_id, checkpoint, changes, expected_version=state.version)

    @staticmethod
    def _assessment(outputs: optimization_tools.AgentOutputs) -> dict:
        return {
            "inventory_status": [InventoryAssessment(**r) for r in outputs.inventory_results],
            "demand_forecasts": [
                ForecastRecord(
                    product_id=str(r["product"]), warehouse_id=r["warehouse"], horizon_days=outputs.parameters.planning_horizon_days,
                    forecast_demand=r["forecast_demand"], model_version=outputs.forecast_model_version)
                for r in outputs.inventory_results
            ],
        }

    @staticmethod
    def _optimization_error(solution: OptimizationSolution, replanning: bool) -> str:
        if solution.status == "INFEASIBLE":
            prefix = "infeasible_after_replan" if replanning else "OPTIMIZATION_INFEASIBLE"
        else:
            prefix = "OPTIMIZATION_ERROR"
        recovery = solution.diagnostics.get("recovery")
        return f"{prefix}: {solution.message}" + (f" Recovery: {recovery}" if recovery else "")

    def _fail(self, simulation_id: str, error: str, steps: list[StepRecord], sensed: SensingResult | None) -> RunOutcome:
        logger.error("run %s failed: %s", simulation_id, error)
        try:
            self.store.commit(simulation_id, "run_failed", {"error": error})
        except WorldStateError:
            pass  # already terminal: report what is recorded rather than mask the original error
        return self._outcome("FAILED", simulation_id, error, steps, sensed, error=error)

    def _outcome(self, outcome: Outcome, simulation_id: str, message: str, steps: list[StepRecord], sensed: SensingResult | None, error: str | None = None) -> RunOutcome:
        state = self.store.get(simulation_id)
        metrics.inc("runs_total", {"outcome": outcome})
        logger.info("run %s -> %s (%s)", simulation_id, outcome, state.status.value,
                    extra={"event": "run_outcome", "outcome": outcome, "status": state.status.value, "simulation_id": simulation_id})
        return RunOutcome(outcome=outcome, simulation_id=simulation_id, status=state.status, message=message, error=error, sensing=sensed, steps=steps, state=state)

    @staticmethod
    def _step(steps: list[StepRecord], name: str, work: Callable[[], Any], describe: Callable[[Any], str] | None = None) -> Any:
        started = time.perf_counter()
        step = name.split(" ")[0]  # "agents (replan)" is still the agents step, for the metrics
        try:
            result = work()
        except Exception as exc:
            steps.append(StepRecord(name=name, ok=False, duration_ms=round((time.perf_counter() - started) * 1000, 1), detail=f"{type(exc).__name__}: {exc}"))
            metrics.inc("run_steps_failed_total", {"step": step})
            metrics.observe("run_step_duration_ms", steps[-1].duration_ms, {"step": step})
            logger.warning("step %s failed (%s)", name, steps[-1].detail, extra={"event": "step", "step": name, "ok": False, "duration_ms": steps[-1].duration_ms})
            raise
        steps.append(StepRecord(name=name, ok=True, duration_ms=round((time.perf_counter() - started) * 1000, 1), detail=describe(result) if describe else ""))
        metrics.observe("run_step_duration_ms", steps[-1].duration_ms, {"step": step})
        logger.info("step %s ok (%s)", name, steps[-1].detail, extra={"event": "step", "step": name, "ok": True, "duration_ms": steps[-1].duration_ms})
        return result
