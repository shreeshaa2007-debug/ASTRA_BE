"""LangGraph wrapper for the ResilientSC orchestration pipeline.

This module wraps the existing :class:`Orchestrator` (the hand-rolled state
machine in ``orchestrator.py``) as a LangGraph ``StateGraph`` so the full
pipeline is visible as a proper node/edge graph while every piece of real
business logic continues to live in the original classes.

Pipeline graph
--------------

    [START]
       │
       ▼
    sense_node          — calls SensingAgent; decides EVENT vs NO_DISRUPTION/error
       │
       ├─ NO_DISRUPTION / SENSING_REJECTED / SENSING_ERROR ──► [END]
       │
       ▼
    agents_node         — gathers Inventory / Sourcing / Logistics agent outputs
       │
       ▼
    optimize_node       — runs the OptimizationEngine
       │
       ├─ not OPTIMAL ──────────────────────────────────────────► fail_node ──► [END]
       │
       ▼
    compliance_node     — validates the plan against compliance rules
       │
       ├─ APPROVED  ──────────────────────────────────────────► finalize_node ──► [END]
       ├─ ESCALATED ──────────────────────────────────────────► escalate_node ──► [END]
       │
       └─ REJECTED (replan allowed) ─► replan_node ──► agents_node  (loop)
                   (max replans hit)  ─────────────────────────────► fail_node

Shared state
------------
``PipelineState`` is the single typed dict that flows through every node. Each
node reads what it needs and writes only its own fields.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from typing_extensions import TypedDict

# LangGraph
from langgraph.graph import END, START, StateGraph

# Existing orchestration helpers (unchanged)
from backend.agents.sensing.agent import SensingAgent
from backend.monitoring.metrics import metrics
from backend.optimization import tools as optimization_tools
from backend.optimization.engine import OptimizationEngine
from backend.orchestration import adapters
from backend.orchestration.orchestrator import Orchestrator, RunOutcome, StepRecord
from backend.schemas.optimization import OptimizationSolution
from backend.schemas.sensing import SensingResult
from backend.schemas.world_state import (
    ApprovalStatus,
    ComplianceStatus,
    ForecastRecord,
    InventoryAssessment,
    WorldState,
)
from backend.services.world_state import (
    MAX_REPLANS,
    WorldStateStore,
    state_changes_for_event,
)

logger = logging.getLogger("resilientsc.langgraph_orchestrator")


# ---------------------------------------------------------------------------
# Shared graph state
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    """Mutable state object passed between every LangGraph node."""

    # inputs
    simulation_id: str
    signal: Any
    product_id: str
    as_of_date: Optional[str]
    tariff_overrides: Optional[dict]

    # sensing
    sensed: Optional[SensingResult]

    # agents
    agent_outputs: Optional[optimization_tools.AgentOutputs]

    # optimize
    solution: Optional[OptimizationSolution]

    # compliance
    verdict: Optional[dict]

    # replan bookkeeping
    replan_count: int
    extra_suppliers: frozenset
    extra_routes: frozenset

    # outcome
    outcome: Optional[str]
    error: Optional[str]
    steps: list
    world_state: Optional[WorldState]


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_graph(
    store: WorldStateStore,
    sensing: SensingAgent,
    *,
    compliance_rules: dict | None = None,
    engine: OptimizationEngine | None = None,
    parameter_overrides: dict | None = None,
):
    """Build and return a compiled LangGraph StateGraph.

    All business-logic objects are captured by closure — nodes themselves are
    pure functions ``(PipelineState) -> dict`` so LangGraph manages state updates.
    """
    from backend.agents.compliance import tools as compliance_tools

    _overrides = dict(parameter_overrides or {})

    # Reuse Orchestrator only for its _step() timing/logging utility
    _orchestrator = Orchestrator(
        store=store,
        sensing=sensing,
        compliance_rules=compliance_rules,
        engine=engine,
        parameter_overrides=parameter_overrides,
    )
    _step = _orchestrator._step

    # ── helpers ──────────────────────────────────────────────────────────────

    def _commit(state_obj: WorldState, checkpoint: str, changes: dict) -> WorldState:
        return store.commit(state_obj.simulation_id, checkpoint, changes, expected_version=state_obj.version)

    def _get_world(sim_id: str) -> WorldState:
        return store.get(sim_id)

    # ── nodes ─────────────────────────────────────────────────────────────────

    def sense_node(state: PipelineState) -> dict:
        """Node 1 — run the Sensing Agent against the incoming signal."""
        steps: list = list(state.get("steps") or [])
        sensed: SensingResult = _step(
            steps, "sense",
            lambda: sensing.sense(state["signal"]),
            lambda r: r.status,
        )
        outcome = None
        if sensed.status != "EVENT":
            _map = {"NO_DISRUPTION": "NO_DISRUPTION", "REJECTED": "SENSING_REJECTED", "ERROR": "SENSING_ERROR"}
            outcome = _map.get(sensed.status, "SENSING_ERROR")
        return {"sensed": sensed, "outcome": outcome, "steps": steps}

    def agents_node(state: PipelineState) -> dict:
        """Node 2 — gather Inventory, Sourcing and Logistics agent outputs."""
        steps = list(state.get("steps") or [])
        world = _get_world(state["simulation_id"])
        replanning = state.get("replan_count", 0) > 0
        extra_suppliers: frozenset = state.get("extra_suppliers") or frozenset()
        extra_routes: frozenset = state.get("extra_routes") or frozenset()

        if not replanning:
            sensed: SensingResult = state["sensed"]
            changes = state_changes_for_event(world, sensed.event)
            if state.get("tariff_overrides"):
                changes["tariffs"] = {**world.tariffs, **state["tariff_overrides"]}
            world = _commit(world, "event_sensed", changes)

        outputs = _step(
            steps,
            "agents (replan)" if replanning else "agents",
            lambda: optimization_tools.gather_agent_outputs(
                state["product_id"],
                state.get("as_of_date"),
                world.disrupted_route_ids() | extra_routes,
                world.disrupted_supplier_ids() | extra_suppliers,
                world.tariffs,
                _overrides,
            ),
            lambda o: (
                f"deficit {o.gross_deficit}, "
                f"{len(o.sourcing_result['candidates_considered'])} suppliers, "
                f"{len(o.logistics_results)} lanes"
            ),
        )

        if not replanning:
            assessment = {
                "inventory_status": [InventoryAssessment(**r) for r in outputs.inventory_results],
                "demand_forecasts": [
                    ForecastRecord(
                        product_id=str(r["product"]),
                        warehouse_id=r["warehouse"],
                        horizon_days=outputs.parameters.planning_horizon_days,
                        forecast_demand=r["forecast_demand"],
                        model_version=outputs.forecast_model_version,
                    )
                    for r in outputs.inventory_results
                ],
            }
            world = _commit(world, "agents_assessed", assessment)

        return {"agent_outputs": outputs, "steps": steps, "world_state": world}

    def optimize_node(state: PipelineState) -> dict:
        """Node 3 — build the optimization problem and solve it."""
        steps = list(state.get("steps") or [])
        outputs = state["agent_outputs"]
        world: WorldState = state.get("world_state") or _get_world(state["simulation_id"])

        solution: OptimizationSolution = _step(
            steps, "optimize",
            lambda: optimization_tools.optimize_supply_chain(
                optimization_tools.build_optimization_problem(
                    state["product_id"],
                    outputs.inventory_results,
                    outputs.sourcing_result,
                    outputs.logistics_results,
                    outputs.safety_stock_by_warehouse,
                    outputs.parameters,
                ),
                engine,
            ),
            lambda s: s.status,
        )
        metrics.inc("optimizations_total", {"status": solution.status})
        if isinstance(solution.solver.get("solve_time_ms"), (int, float)):
            metrics.observe("optimization_solve_ms", solution.solver["solve_time_ms"])

        world = _commit(world, "plan_optimized", {"current_plan": solution})

        error = None
        if solution.status != "OPTIMAL":
            replanning = state.get("replan_count", 0) > 0
            prefix = (
                "infeasible_after_replan" if solution.status == "INFEASIBLE" and replanning
                else "OPTIMIZATION_INFEASIBLE" if solution.status == "INFEASIBLE"
                else "OPTIMIZATION_ERROR"
            )
            recovery = solution.diagnostics.get("recovery")
            error = f"{prefix}: {solution.message}" + (f" Recovery: {recovery}" if recovery else "")

        return {"solution": solution, "steps": steps, "world_state": world, "error": error}

    def compliance_node(state: PipelineState) -> dict:
        """Node 4 — run compliance validator against the optimized plan."""
        steps = list(state.get("steps") or [])
        solution: OptimizationSolution = state["solution"]
        world: WorldState = state.get("world_state") or _get_world(state["simulation_id"])
        extra_suppliers: frozenset = state.get("extra_suppliers") or frozenset()
        extra_routes: frozenset = state.get("extra_routes") or frozenset()

        verdict: dict = _step(
            steps, "compliance",
            lambda: compliance_tools.validate_plan(
                adapters.plan_to_compliance_input(
                    solution, state["product_id"],
                    world.disrupted_route_ids() | extra_routes,
                    world.disrupted_supplier_ids() | extra_suppliers,
                ),
                compliance_rules,
            ),
            lambda v: v["status"],
        )
        metrics.inc("compliance_verdicts_total", {"status": verdict["status"]})
        world = _commit(world, "compliance_checked", {"compliance_status": ComplianceStatus(**verdict)})
        return {"verdict": verdict, "steps": steps, "world_state": world}

    def finalize_node(state: PipelineState) -> dict:
        """Terminal node — plan APPROVED, commit finalization."""
        world: WorldState = state.get("world_state") or _get_world(state["simulation_id"])
        verdict = state.get("verdict", {})
        world = _commit(world, "plan_finalized", {"approval_status": ApprovalStatus.NOT_REQUIRED})
        metrics.inc("runs_total", {"outcome": "COMPLETED"})
        logger.info("simulation %s: COMPLETED (auto-approved: %s)", state["simulation_id"], verdict.get("reason"))
        return {"outcome": "COMPLETED", "world_state": world}

    def escalate_node(state: PipelineState) -> dict:
        """Terminal node — plan ESCALATED, waiting for human approval."""
        world: WorldState = state.get("world_state") or _get_world(state["simulation_id"])
        verdict = state.get("verdict", {})
        world = _commit(world, "approval_requested", {"approval_status": ApprovalStatus.PENDING})
        metrics.inc("runs_total", {"outcome": "AWAITING_APPROVAL"})
        logger.info("simulation %s: AWAITING_APPROVAL (%s)", state["simulation_id"], verdict.get("reason"))
        return {"outcome": "AWAITING_APPROVAL", "world_state": world}

    def replan_node(state: PipelineState) -> dict:
        """Intermediate node — compliance REJECTED; prepare a replan pass."""
        world: WorldState = state.get("world_state") or _get_world(state["simulation_id"])
        verdict = state.get("verdict", {})
        solution: OptimizationSolution = state["solution"]
        extra_suppliers: frozenset = state.get("extra_suppliers") or frozenset()
        extra_routes: frozenset = state.get("extra_routes") or frozenset()
        replan_count: int = state.get("replan_count", 0)

        new_suppliers, new_routes = adapters.exclusions_from_verdict(verdict.get("checks", []), solution.problem)
        world = _commit(world, "replan_requested", {
            "current_plan": None,
            "compliance_status": None,
            "replan_count": replan_count + 1,
        })
        metrics.inc("replans_total")
        logger.info(
            "compliance rejected (%s); replanning without suppliers=%s routes=%s",
            verdict.get("reason"), sorted(new_suppliers), sorted(new_routes),
        )
        return {
            "extra_suppliers": extra_suppliers | new_suppliers,
            "extra_routes": extra_routes | new_routes,
            "replan_count": replan_count + 1,
            "world_state": world,
            "solution": None,
            "verdict": None,
        }

    def set_rejection_error(state: PipelineState) -> dict:
        """Patch node — sets 'error' before routing to fail_node on exhausted replan."""
        verdict = state.get("verdict") or {}
        replan_count = state.get("replan_count", 0)
        reason = verdict.get("reason", "compliance rejected")
        solution = state.get("solution")
        extra_suppliers = state.get("extra_suppliers") or frozenset()
        extra_routes = state.get("extra_routes") or frozenset()
        new_s, new_r = adapters.exclusions_from_verdict(
            verdict.get("checks", []), solution.problem if solution else None
        ) if solution else (frozenset(), frozenset())

        if replan_count >= MAX_REPLANS:
            error = f"rejected_after_replan: {reason}"
        else:
            error = f"compliance_rejected: {reason} (no offenders to exclude — replan would reproduce the same plan)"
        return {"error": error}

    def fail_node(state: PipelineState) -> dict:
        """Terminal node — record failure in the store and surface the error."""
        error = state.get("error") or "pipeline failed"
        sim_id = state["simulation_id"]
        try:
            store.commit(sim_id, "run_failed", {"error": error})
        except Exception:  # noqa: BLE001
            pass
        world = _get_world(sim_id)
        metrics.inc("runs_total", {"outcome": "FAILED"})
        logger.error("simulation %s: FAILED — %s", sim_id, error)
        return {"outcome": "FAILED", "world_state": world}

    # ── routing functions (conditional edges) ─────────────────────────────────

    def route_after_sense(state: PipelineState) -> str:
        """After sense_node: 'agents' if EVENT detected, else terminal outcome."""
        outcome = state.get("outcome")
        return outcome if outcome else "agents"

    def route_after_optimize(state: PipelineState) -> str:
        """After optimize_node: 'compliance' if OPTIMAL, else 'fail'."""
        return "fail" if state.get("error") else "compliance"

    def route_after_compliance(state: PipelineState) -> str:
        """After compliance_node: route on APPROVED / ESCALATED / REJECTED."""
        verdict = state.get("verdict") or {}
        status = verdict.get("status", "REJECTED")
        if status == "APPROVED":
            return "finalize"
        if status == "ESCALATED":
            return "escalate"
        # REJECTED — can we replan?
        replan_count = state.get("replan_count", 0)
        solution = state.get("solution")
        new_s, new_r = adapters.exclusions_from_verdict(
            verdict.get("checks", []), solution.problem if solution else None
        ) if solution else (frozenset(), frozenset())
        if replan_count >= MAX_REPLANS or not (new_s or new_r):
            return "set_rejection_error"
        return "replan"

    # ── assemble graph ────────────────────────────────────────────────────────

    builder = StateGraph(PipelineState)

    builder.add_node("sense", sense_node)
    builder.add_node("agents", agents_node)
    builder.add_node("optimize", optimize_node)
    builder.add_node("compliance", compliance_node)
    builder.add_node("finalize", finalize_node)
    builder.add_node("escalate", escalate_node)
    builder.add_node("replan", replan_node)
    builder.add_node("set_rejection_error", set_rejection_error)
    builder.add_node("fail", fail_node)

    builder.add_edge(START, "sense")

    builder.add_conditional_edges(
        "sense",
        route_after_sense,
        {
            "agents": "agents",
            "NO_DISRUPTION": END,
            "SENSING_REJECTED": END,
            "SENSING_ERROR": END,
        },
    )
    builder.add_edge("agents", "optimize")
    builder.add_conditional_edges(
        "optimize",
        route_after_optimize,
        {"compliance": "compliance", "fail": "fail"},
    )
    builder.add_conditional_edges(
        "compliance",
        route_after_compliance,
        {
            "finalize": "finalize",
            "escalate": "escalate",
            "replan": "replan",
            "set_rejection_error": "set_rejection_error",
        },
    )
    builder.add_edge("replan", "agents")           # ← replan loop-back
    builder.add_edge("set_rejection_error", "fail")

    builder.add_edge("finalize", END)
    builder.add_edge("escalate", END)
    builder.add_edge("fail", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_pipeline(
    graph,
    simulation_id: str,
    signal: str | Mapping[str, Any],
    product_id: str,
    *,
    as_of_date: str | None = None,
    tariff_overrides: dict | None = None,
) -> RunOutcome:
    """Invoke the compiled LangGraph and convert the final PipelineState
    into a RunOutcome (same shape the existing API layer expects).
    """
    initial: PipelineState = {
        "simulation_id": simulation_id,
        "signal": signal,
        "product_id": str(product_id),
        "as_of_date": as_of_date,
        "tariff_overrides": tariff_overrides,
        "replan_count": 0,
        "extra_suppliers": frozenset(),
        "extra_routes": frozenset(),
        "steps": [],
        "outcome": None,
        "error": None,
        "sensed": None,
        "agent_outputs": None,
        "solution": None,
        "verdict": None,
        "world_state": None,
    }

    final: PipelineState = graph.invoke(initial)

    world_state = final.get("world_state")
    outcome = final.get("outcome") or "FAILED"

    return RunOutcome(
        outcome=outcome,
        simulation_id=simulation_id,
        status=world_state.status if world_state else None,
        message=final.get("error") or f"pipeline ended: {outcome}",
        error=final.get("error"),
        sensing=final.get("sensed"),
        steps=final.get("steps") or [],
        state=world_state,
    )
