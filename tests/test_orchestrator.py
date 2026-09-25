"""Phase 14 tests — Orchestrator, brief §21 "ORCHESTRATION"/"END-TO-END" categories.

The real Inventory, Sourcing, Logistics, Optimization and Compliance code runs
in every test; only the LLM is a fake. Compliance's verdict is steered with
real rules (a rejected supplier, a restricted country, a low approval
threshold) wherever that is possible, and with an injected `validate_plan`
only for cases the real rules cannot produce (a rejection that names nothing to
exclude). Gathering the agents' outputs is memoized per distinct input to keep
the module fast — the computation is real, it just isn't repeated.
"""
from __future__ import annotations

import copy
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from backend.agents.sensing.agent import SensingAgent
from backend.agents.sensing.llm import LLMResponse, LLMUnavailableError
from backend.database.world_state_repository import SqlAlchemyWorldStateRepository
from backend.optimization import tools as optimization_tools
from backend.optimization.engine import PrototypeOptimizationEngine
from backend.orchestration import Orchestrator, adapters
from backend.schemas.optimization import (
    OptimizationParameters,
    OptimizationProblem,
    RouteOption,
    SupplierOption,
    WarehouseState,
)
from backend.schemas.world_state import ApprovalStatus, SimulationStatus
from backend.services.world_state import (
    CheckpointViolationError,
    ConcurrentModificationError,
    InvalidTransitionError,
    WorldStateStore,
)

requires_built_data = pytest.mark.skipif(
    not all(Path(p).exists() for p in (
        "data/processed/suppliers.csv", "data/processed/routes.csv", "data/processed/tariffs.csv",
        "data/processed/inventory_multi_warehouse.csv", "data/processed/demand_modeling_panel.csv",
        "ml/artifacts/xgboost_demand/2026.09.1/model.json",
    )),
    reason="run the Phase 3-5 pipelines first",
)

PRODUCT, AS_OF = "22197", "2011-11-30"
SUEZ = ["SHA-ROT-SUEZ", "SIN-ROT-SUEZ", "MUM-ROT-SUEZ", "CHE-ROT-SUEZ"]
HAPPY_PATH = ["simulation_created", "event_sensed", "agents_assessed", "plan_optimized", "compliance_checked", "plan_finalized"]
RULES_OPEN = {"rejected_suppliers": [], "restricted_countries": [], "approval_threshold": 10_000_000}
RULES_LOW_THRESHOLD = {**RULES_OPEN, "approval_threshold": 100_000}  # the Suez plan costs ~465k


@pytest.fixture(scope="module", autouse=True)
def memoized_agent_outputs():
    real, cache = optimization_tools.gather_agent_outputs, {}

    def memo(product_id, as_of_date=None, disrupted_route_ids=frozenset(), excluded_supplier_ids=frozenset(), tariff_rates=None, parameter_overrides=None, config=None):
        key = (product_id, as_of_date, frozenset(disrupted_route_ids), frozenset(excluded_supplier_ids),
               tuple(sorted((tariff_rates or {}).items())), tuple(sorted((parameter_overrides or {}).items())))
        if key not in cache:
            cache[key] = real(product_id, as_of_date, disrupted_route_ids, excluded_supplier_ids, tariff_rates, parameter_overrides, config)
        return copy.deepcopy(cache[key])

    patch = pytest.MonkeyPatch()
    patch.setattr(optimization_tools, "gather_agent_outputs", memo)
    yield
    patch.undo()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class FakeLLM:
    model = "fake-llm"

    def __init__(self, payload=None, error=None):
        self.payload, self.error, self.calls = payload, error, 0

    def generate_json(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return LLMResponse(json.dumps(self.payload), self.model, 1.0, 1)


def candidate(**overrides) -> dict:
    c = {
        "is_disruption": True, "rationale": "A vessel is aground and blocks the canal.", "event_type": "canal_closure", "location": "Suez Canal",
        "severity": "CRITICAL", "start_date": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(), "estimated_duration": 10,
        "affected_routes": list(SUEZ), "affected_suppliers": [], "affected_products": [], "confidence": 0.95,
    }
    c.update(overrides)
    return c


def build(rules=RULES_OPEN, llm=None, **kwargs):
    store = WorldStateStore(SqlAlchemyWorldStateRepository("sqlite:///:memory:"))  # real baseline from the processed data
    llm = llm or FakeLLM(candidate())
    orch = Orchestrator(store, SensingAgent(llm=llm), compliance_rules=rules, **kwargs)
    return store, orch, llm


def run(orch: Orchestrator, store: WorldStateStore, *, product=PRODUCT, **kwargs):
    sim = store.create("TEST").simulation_id
    return orch.run(sim, "A vessel is aground in the Suez Canal.", product, as_of_date=AS_OF, **kwargs)


def history(store, sim) -> list[str]:
    return [h.checkpoint for h in store.history(sim)]


def used_suppliers(state) -> set[str]:
    return {a.supplier_id for a in state.current_plan.allocations}


def rejecting(offenders: list[str]):
    """A stand-in compliance that rejects, naming `offenders`; records what it was asked to check."""
    calls: list[dict] = []

    def validate(plan: dict) -> dict:
        calls.append(plan)
        checks = [{"name": "supplier_permitted", "passed": False, "detail": "test rejection", "offenders": offenders}] if offenders else \
                 [{"name": "supplier_permitted", "passed": False, "detail": "test rejection"}]
        return {"status": "REJECTED", "checks": checks, "reason": "test rejection", "requires_human": False}

    validate.calls = calls
    return validate


@pytest.fixture(scope="module")
def baseline_plan():
    """The Suez-closure plan under open rules — what other tests compare against."""
    store, orch, _ = build()
    return run(orch, store).state.current_plan


# --------------------------------------------------------------------------- #
# the happy path: an in-policy plan finalizes with no human
# --------------------------------------------------------------------------- #
@requires_built_data
def test_an_in_policy_plan_runs_end_to_end_and_finalizes_automatically():
    store, orch, _ = build()
    out = run(orch, store)
    assert (out.outcome, out.status) == ("COMPLETED", SimulationStatus.COMPLETED) and out.error is None
    assert history(store, out.simulation_id) == HAPPY_PATH

    s = out.state
    assert s.version == 5 and s.approval_status == ApprovalStatus.NOT_REQUIRED and s.approval_decision is None and s.replan_count == 0
    assert s.compliance_status.status == "APPROVED" and s.current_plan.status == "OPTIMAL" and s.error is None
    assert s.disrupted_route_ids() == frozenset(SUEZ) and [d.event_type for d in s.current_disruptions] == ["canal_closure"]
    assert not any(a.route_id in SUEZ for a in s.current_plan.allocations)  # the plan routes around the closure the state recorded


@requires_built_data
def test_what_the_agents_found_is_committed_to_the_state():
    store, orch, _ = build()
    s = run(orch, store).state
    assert {a.warehouse for a in s.inventory_status} == {"Mumbai", "Chennai", "Delhi"} and all(a.product == PRODUCT for a in s.inventory_status)
    assert {f.warehouse_id for f in s.demand_forecasts} == {"Mumbai", "Chennai", "Delhi"}
    assert all(f.horizon_days == 45 and f.model_version == "2026.09.1" and f.forecast_demand > 0 for f in s.demand_forecasts)
    assert s.route_status["SHA-ROT-SUEZ"].value == "DISRUPTED" and s.route_status["SHA-ROT-CAPE"].value == "ALTERNATIVE"


@requires_built_data
def test_the_run_reports_each_step_with_timing():
    store, orch, _ = build()
    out = run(orch, store)
    assert [s.name for s in out.steps] == ["sense", "agents", "optimize", "compliance"] and all(s.ok and s.duration_ms >= 0 for s in out.steps)
    assert out.steps[0].detail == "EVENT" and out.steps[2].detail == "OPTIMAL" and out.steps[3].detail == "APPROVED"
    assert out.sensing.status == "EVENT" and out.sensing.event.event_id == out.state.current_disruptions[0].event_id


@requires_built_data
def test_the_same_disruption_gives_the_same_plan_every_time(baseline_plan):
    store, orch, _ = build()
    again = run(orch, store).state.current_plan
    assert again.objective_value == baseline_plan.objective_value
    assert [(a.supplier_id, a.route_id, a.quantity) for a in again.allocations] == [(a.supplier_id, a.route_id, a.quantity) for a in baseline_plan.allocations]


# --------------------------------------------------------------------------- #
# the human-approval gate
# --------------------------------------------------------------------------- #
@requires_built_data
def test_a_high_impact_plan_stops_and_waits_for_a_human():
    store, orch, _ = build(RULES_LOW_THRESHOLD)
    out = run(orch, store)
    assert (out.outcome, out.status) == ("AWAITING_APPROVAL", SimulationStatus.AWAITING_APPROVAL) and "approval threshold" in out.message
    assert out.state.compliance_status.status == "ESCALATED" and out.state.compliance_status.requires_human
    assert out.state.approval_status == ApprovalStatus.PENDING and out.state.approval_decision is None
    assert history(store, out.simulation_id)[-1] == "approval_requested" and "plan_finalized" not in history(store, out.simulation_id)


@requires_built_data
def test_a_named_human_approves_and_the_plan_is_finalized():
    store, orch, _ = build(RULES_LOW_THRESHOLD)
    waiting = run(orch, store)
    done = orch.approve(waiting.simulation_id, "  logistics.director  ", "checked against the carrier quotes", expected_version=waiting.state.version)
    assert done.status == SimulationStatus.COMPLETED and done.approval_status == ApprovalStatus.APPROVED
    assert done.approval_decision.decided_by == "logistics.director" and done.approval_decision.note == "checked against the carrier quotes"
    last = store.history(waiting.simulation_id)[-1]
    assert (last.checkpoint, last.actor) == ("plan_finalized", "logistics.director")  # the audit trail names the approver
    assert done.current_plan.objective_value == waiting.state.current_plan.objective_value  # the plan approved is the plan finalized


@requires_built_data
def test_a_human_can_reject_and_the_simulation_ends():
    store, orch, _ = build(RULES_LOW_THRESHOLD)
    waiting = run(orch, store)
    done = orch.reject(waiting.simulation_id, "cfo", "over budget this quarter")
    assert done.status == SimulationStatus.REJECTED and done.approval_status == ApprovalStatus.REJECTED and done.approval_decision.decided_by == "cfo"
    with pytest.raises(InvalidTransitionError):
        orch.approve(waiting.simulation_id, "someone.else")  # a rejection is final


@requires_built_data
def test_an_approval_needs_a_named_approver():
    store, orch, _ = build(RULES_LOW_THRESHOLD)
    sim = run(orch, store).simulation_id
    for blank in ("", "   "):
        with pytest.raises(CheckpointViolationError, match="named decider"):
            orch.approve(sim, blank)
    assert store.get(sim).status == SimulationStatus.AWAITING_APPROVAL


@requires_built_data
def test_approving_a_plan_that_changed_since_you_saw_it_is_refused():
    store, orch, _ = build(RULES_LOW_THRESHOLD)
    waiting = run(orch, store)
    with pytest.raises(ConcurrentModificationError):
        orch.approve(waiting.simulation_id, "director", expected_version=waiting.state.version - 1)
    assert store.get(waiting.simulation_id).status == SimulationStatus.AWAITING_APPROVAL


@requires_built_data
def test_a_plan_that_needed_no_approval_cannot_be_approved_and_nothing_can_be_approved_twice():
    store, orch, _ = build()
    auto = run(orch, store)
    with pytest.raises(InvalidTransitionError):
        orch.approve(auto.simulation_id, "director")

    store2, orch2, _ = build(RULES_LOW_THRESHOLD)
    sim = run(orch2, store2).simulation_id
    orch2.approve(sim, "director")
    with pytest.raises(InvalidTransitionError):
        orch2.approve(sim, "director")


# --------------------------------------------------------------------------- #
# compliance REJECTED -> one bounded replan
# --------------------------------------------------------------------------- #
@requires_built_data
def test_a_rejected_supplier_triggers_one_replan_without_it(baseline_plan):
    assert "S007" in {a.supplier_id for a in baseline_plan.allocations}  # the cheapest plan does use S007, so rejecting it bites
    store, orch, _ = build({**RULES_OPEN, "rejected_suppliers": ["S007"]})
    out = run(orch, store)

    assert out.outcome == "COMPLETED"
    assert history(store, out.simulation_id) == [
        "simulation_created", "event_sensed", "agents_assessed", "plan_optimized", "compliance_checked",
        "replan_requested", "plan_optimized", "compliance_checked", "plan_finalized"]
    assert out.state.replan_count == 1 and "S007" not in used_suppliers(out.state) and out.state.compliance_status.status == "APPROVED"
    assert out.state.current_plan.objective_value > baseline_plan.objective_value  # excluding the cheapest supplier is never free
    assert [s.name for s in out.steps] == ["sense", "agents", "optimize", "compliance", "agents (replan)", "optimize", "compliance"]


@requires_built_data
def test_a_restricted_country_excludes_every_supplier_in_it_on_replan():
    store, orch, _ = build({**RULES_OPEN, "restricted_countries": ["Turkey"]})
    out = run(orch, store)
    assert out.outcome == "COMPLETED" and out.state.replan_count == 1
    assert not any(s.region == "Turkey" for s in out.state.current_plan.problem.suppliers)  # S007 was never offered again


@requires_built_data
def test_a_replanned_plan_still_goes_through_the_approval_gate():
    store, orch, _ = build({**RULES_LOW_THRESHOLD, "rejected_suppliers": ["S007"]})
    out = run(orch, store)
    assert out.outcome == "AWAITING_APPROVAL" and out.state.replan_count == 1 and "S007" not in used_suppliers(out.state)
    assert orch.approve(out.simulation_id, "director").status == SimulationStatus.COMPLETED


@requires_built_data
def test_a_second_rejection_ends_the_run_instead_of_replanning_again():
    store, orch, _ = build({**RULES_OPEN, "rejected_suppliers": ["S007", "S006", "S005"]})  # the replan's best alternative is S005
    out = run(orch, store)
    assert out.outcome == "FAILED" and out.status == SimulationStatus.FAILED and out.error.startswith("rejected_after_replan")
    assert history(store, out.simulation_id)[-4:] == ["replan_requested", "plan_optimized", "compliance_checked", "run_failed"]
    assert out.state.replan_count == 1 and out.state.error == out.error


@requires_built_data
def test_a_replan_that_leaves_no_feasible_plan_fails_with_the_diagnosis_kept():
    validate = rejecting(["S007", "S006"])  # the two suppliers the 30-day plan depends on
    store, orch, _ = build(validate_plan=validate, parameter_overrides={"max_delivery_days": 30})
    out = run(orch, store)
    assert out.outcome == "FAILED" and out.error.startswith("infeasible_after_replan") and "Recovery" in out.error
    assert history(store, out.simulation_id)[-3:] == ["replan_requested", "plan_optimized", "run_failed"]
    assert out.state.current_plan.status == "INFEASIBLE" and out.state.current_plan.diagnostics["total_shortfall_units"] > 0
    assert len(validate.calls) == 1  # compliance is never asked to check a plan that doesn't exist


@requires_built_data
def test_a_rejection_that_names_nothing_to_exclude_fails_at_once_rather_than_replanning_in_vain():
    validate = rejecting([])
    store, orch, _ = build(validate_plan=validate)
    out = run(orch, store)
    assert out.outcome == "FAILED" and out.error.startswith("compliance_rejected") and "same plan" in out.error
    assert "replan_requested" not in history(store, out.simulation_id) and len(validate.calls) == 1


# --------------------------------------------------------------------------- #
# failures are recorded, never hung, never faked
# --------------------------------------------------------------------------- #
@requires_built_data
def test_demand_beyond_all_supply_fails_with_the_optimizers_recovery_hint_and_skips_compliance():
    validate = rejecting(["S007"])
    store, orch, _ = build(validate_plan=validate)
    out = run(orch, store, product="23166")  # ~21k units needed vs ~16k of supplier capacity
    assert out.outcome == "FAILED" and out.error.startswith("OPTIMIZATION_INFEASIBLE") and "Recovery" in out.error
    assert history(store, out.simulation_id) == ["simulation_created", "event_sensed", "agents_assessed", "plan_optimized", "run_failed"]
    assert out.state.current_plan.status == "INFEASIBLE" and validate.calls == []


@requires_built_data
def test_an_unexpected_error_mid_run_is_recorded_as_a_failure_not_left_running(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("forecasting service down")

    monkeypatch.setattr(optimization_tools, "gather_agent_outputs", explode)
    store, orch, _ = build()
    out = run(orch, store)
    assert out.outcome == "FAILED" and out.status == SimulationStatus.FAILED and out.error == "unexpected error: RuntimeError: forecasting service down"
    assert history(store, out.simulation_id) == ["simulation_created", "event_sensed", "run_failed"]
    assert [(s.name, s.ok) for s in out.steps] == [("sense", True), ("agents", False)] and "forecasting service down" in out.steps[1].detail


@requires_built_data
@pytest.mark.parametrize("llm, outcome", [
    (FakeLLM({"is_disruption": False, "rationale": "Just weather chat."}), "NO_DISRUPTION"),
    (FakeLLM(candidate(severity="high", affected_routes=["NOPE"])), "SENSING_REJECTED"),
    (FakeLLM(error=LLMUnavailableError("Gemini unavailable after 3 attempts")), "SENSING_ERROR"),
])
def test_when_sensing_finds_no_event_nothing_enters_the_state_and_the_simulation_can_be_rerun(llm, outcome):
    store, orch, _ = build(llm=llm)
    out = run(orch, store)
    assert out.outcome == outcome and out.status == SimulationStatus.CREATED and out.error is None and out.message
    assert history(store, out.simulation_id) == ["simulation_created"] and out.state.version == 0 and out.state.current_disruptions == []
    assert [s.name for s in out.steps] == ["sense"]

    # nothing was consumed: a good signal on the same simulation now works
    orch.sensing._llm = FakeLLM(candidate())
    again = orch.run(out.simulation_id, "A vessel is aground in the Suez Canal.", PRODUCT, as_of_date=AS_OF)
    assert again.outcome == "COMPLETED"


@requires_built_data
def test_a_simulation_that_has_already_run_is_refused_before_any_llm_call():
    store, orch, llm = build()
    first = run(orch, store)
    calls = llm.calls
    with pytest.raises(InvalidTransitionError, match="COMPLETED"):
        orch.run(first.simulation_id, "again", PRODUCT, as_of_date=AS_OF)
    assert llm.calls == calls  # no LLM quota spent on a run that could not proceed

    store.reset(first.simulation_id)
    assert orch.run(first.simulation_id, "A vessel is aground in the Suez Canal.", PRODUCT, as_of_date=AS_OF).outcome == "COMPLETED"  # reset makes it runnable again


# --------------------------------------------------------------------------- #
# the world state — not the data files — is what the agents are told
# --------------------------------------------------------------------------- #
@requires_built_data
def test_a_sensed_supplier_failure_keeps_that_supplier_out_of_the_plan(baseline_plan):
    store, orch, _ = build(llm=FakeLLM(candidate(event_type="supplier_failure", location="Istanbul, Turkey", affected_routes=[], affected_suppliers=["S007"])))
    out = run(orch, store)
    assert out.outcome == "COMPLETED"
    assert out.state.disrupted_supplier_ids() == frozenset({"S001", "S007"})  # S001 is disrupted in the baseline data, S007 by the event
    assert "S007" not in used_suppliers(out.state) and out.state.disrupted_route_ids() == frozenset()  # no route was blocked, and none was assumed
    assert out.state.compliance_status.status == "APPROVED"


@requires_built_data
def test_a_scenario_tariff_reaches_the_price_the_optimizer_sees(baseline_plan):
    store, orch, _ = build()
    out = run(orch, store, tariff_overrides={"TUR": 60.0})
    assert out.state.tariffs["TUR"] == 60.0 and out.state.tariffs["CHN"] == 2.18  # merged over the baseline, not replacing it
    turkish = next(s for s in out.state.current_plan.problem.suppliers if s.supplier_id == "S007")
    assert turkish.tariff_rate_pct == 60.0 and turkish.landed_unit_cost == pytest.approx(turkish.base_unit_cost * 1.6, abs=1e-3)
    assert out.state.current_plan.objective_terms["tariff"] > baseline_plan.objective_terms["tariff"]
    assert out.state.current_plan.objective_value > baseline_plan.objective_value


@requires_built_data
def test_a_tariff_report_alone_changes_no_route_and_the_plan_is_the_undisturbed_one():
    store, orch, _ = build(llm=FakeLLM(candidate(event_type="tariff_change", location="China", affected_routes=SUEZ[:2], affected_suppliers=["S002"])))
    out = run(orch, store)
    assert out.outcome == "COMPLETED" and out.state.disrupted_route_ids() == frozenset()  # even though the LLM listed routes

    # the plan is exactly the one for a healthy network, computed here without going through the state at all
    healthy = optimization_tools.optimize_supply_chain(optimization_tools.build_problem_from_agents(PRODUCT, AS_OF))
    assert out.state.current_plan.objective_value == healthy.objective_value
    assert {a.route_id for a in out.state.current_plan.allocations} == {a.route_id for a in healthy.allocations}


# --------------------------------------------------------------------------- #
# simulated triggers, isolation, concurrency
# --------------------------------------------------------------------------- #
@requires_built_data
def test_a_prestructured_simulated_trigger_runs_end_to_end_with_no_llm_and_no_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    store = WorldStateStore(SqlAlchemyWorldStateRepository("sqlite:///:memory:"))
    orch = Orchestrator(store, SensingAgent(), compliance_rules=RULES_OPEN)  # no LLM injected, none in the environment
    sim = store.create("SUEZ_CLOSURE").simulation_id
    trigger = {"scenario_type": "SUEZ_CLOSURE", "description": "Vessel grounded in the canal", "candidate": {k: v for k, v in candidate().items() if k != "is_disruption"}}
    out = orch.run(sim, trigger, PRODUCT, as_of_date=AS_OF)
    assert out.outcome == "COMPLETED" and out.sensing.llm["model"].startswith("none") and out.state.disrupted_route_ids() == frozenset(SUEZ)


@requires_built_data
def test_simulations_running_at_the_same_time_stay_isolated():
    store = WorldStateStore(SqlAlchemyWorldStateRepository("sqlite:///:memory:"))
    suez = Orchestrator(store, SensingAgent(llm=FakeLLM(candidate())), compliance_rules=RULES_OPEN)
    supplier = Orchestrator(store, SensingAgent(llm=FakeLLM(candidate(event_type="supplier_failure", location="Istanbul", affected_routes=[], affected_suppliers=["S007"]))), compliance_rules=RULES_OPEN)
    sim_a, sim_b = store.create("SUEZ").simulation_id, store.create("SUPPLIER_FAILURE").simulation_id

    with ThreadPoolExecutor(2) as pool:
        a = pool.submit(suez.run, sim_a, "Suez blocked", PRODUCT, as_of_date=AS_OF)
        b = pool.submit(supplier.run, sim_b, "Istanbul fire", PRODUCT, as_of_date=AS_OF)
        out_a, out_b = a.result(), b.result()

    assert out_a.outcome == out_b.outcome == "COMPLETED"
    assert store.get(sim_a).disrupted_route_ids() == frozenset(SUEZ) and store.get(sim_a).disrupted_supplier_ids() == frozenset({"S001"})
    assert store.get(sim_b).disrupted_route_ids() == frozenset() and store.get(sim_b).disrupted_supplier_ids() == frozenset({"S001", "S007"})
    assert history(store, sim_a) == HAPPY_PATH and history(store, sim_b) == HAPPY_PATH


@requires_built_data
def test_two_runs_started_on_one_simulation_cannot_both_proceed():
    store, orch, _ = build()
    sim = store.create("RACE").simulation_id
    barrier, results = threading.Barrier(2), []

    def attempt(_):
        barrier.wait()
        try:
            results.append(orch.run(sim, "Suez blocked", PRODUCT, as_of_date=AS_OF))
        except (InvalidTransitionError, ConcurrentModificationError) as exc:
            results.append(exc)

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(attempt, range(2)))
    finished = [r for r in results if not isinstance(r, Exception)]
    assert len(finished) == 1 and finished[0].outcome == "COMPLETED"
    assert history(store, sim) == HAPPY_PATH  # one event, one plan — the loser wrote nothing


# --------------------------------------------------------------------------- #
# the adapters between the optimizer and compliance — no data needed
# --------------------------------------------------------------------------- #
def make_solution(delay_penalty: float = 0.0):
    """S1 and S2 both ship from P; R1 is cheap but small, R2 carries the rest; W is 900 short."""
    suppliers = [
        SupplierOption(supplier_id=s, supplier_name=s, region=region, origin_port="P", capacity=600, base_unit_cost=10, landed_unit_cost=10 + i * (1 if i < 2 else 40),
                       tariff_rate_pct=0, lead_time_days=5, reliability=0.9, risk_level="LOW")
        for i, (s, region) in enumerate([("S1", "China"), ("S2", "India"), ("S3", "India")])
    ]  # S3 is dear and never chosen — it is the supplier an exclusion must still catch
    routes = [
        RouteOption(route_id="R1", origin="P", destination="Hub", transport_mode="air", capacity=400, cost_per_unit=1, transit_time_days=2),
        RouteOption(route_id="R2", origin="P", destination="Hub", transport_mode="sea", capacity=5000, cost_per_unit=5, transit_time_days=20),
    ]
    parameters = OptimizationParameters(planning_horizon_days=45, internal_transfer_cost_per_unit=5.0, internal_transfer_days=2,
                                        target_delivery_days=10 if delay_penalty else None, delay_penalty_per_unit_day=delay_penalty)
    problem = OptimizationProblem(product_id="X", parameters=parameters, suppliers=suppliers, routes=routes,
                                  warehouses=[WarehouseState(warehouse_id="W", current_stock=0, safety_stock=0, forecast_demand=900, stockout_risk="HIGH")])
    return PrototypeOptimizationEngine().optimize_supply_chain(problem)


def test_compliance_input_sums_a_suppliers_volume_across_routes_and_lists_every_route_used():
    solution = make_solution()
    assert solution.status == "OPTIMAL" and len({a.route_id for a in solution.allocations}) == 2
    plan = adapters.plan_to_compliance_input(solution, "X", frozenset({"R9"}), frozenset({"S9"}))
    assert plan["route_ids"] == ["R1", "R2"] and plan["disrupted_route_ids"] == frozenset({"R9"}) and plan["disrupted_supplier_ids"] == frozenset({"S9"})
    per_supplier = {a["supplier_id"]: a["quantity"] for a in plan["supplier_allocations"]}
    assert sum(per_supplier.values()) == 900 and all(a["product_id"] == "X" for a in plan["supplier_allocations"])
    assert len(plan["supplier_allocations"]) == len(per_supplier)  # one entry per supplier, not one per lane


def test_plan_spend_is_money_committed_and_excludes_the_delay_penalty():
    solution = make_solution(delay_penalty=2.0)
    terms = solution.objective_terms
    assert terms["delay_penalty"] > 0
    assert adapters.plan_spend(solution) == pytest.approx(terms["procurement"] + terms["tariff"] + terms["freight"] + terms["transfer"], abs=0.01)
    assert adapters.plan_spend(solution) < solution.objective_value


def test_exclusions_come_from_the_failed_checks_offenders():
    problem = make_solution().problem
    checks = [
        {"name": "supplier_permitted", "passed": False, "offenders": ["S1"]},
        {"name": "route_permitted", "passed": False, "offenders": ["R2"]},
        {"name": "cost_within_threshold", "passed": False},
    ]
    assert adapters.exclusions_from_verdict(checks, problem) == (frozenset({"S1"}), frozenset({"R2"}))


def test_a_restricted_country_excludes_all_its_suppliers_and_passed_checks_are_ignored():
    problem = make_solution().problem
    checks = [{"name": "country_permitted", "passed": False, "offenders": ["India"]}, {"name": "supplier_permitted", "passed": True, "offenders": ["S1"]}]
    assert {a.supplier_id for a in make_solution().allocations} == {"S1", "S2"}  # S3 is not in the plan...
    assert adapters.exclusions_from_verdict(checks, problem) == (frozenset({"S2", "S3"}), frozenset())  # ...but is in the restricted country


def test_a_verdict_that_names_nothing_yields_no_exclusions():
    problem = make_solution().problem
    assert adapters.exclusions_from_verdict([{"name": "supplier_permitted", "passed": False}], problem) == (frozenset(), frozenset())
    assert adapters.exclusions_from_verdict([], problem) == (frozenset(), frozenset())


# --------------------------------------------------------------------------- #
# live — the whole pipeline with the real Gemini. Opt in: RUN_LIVE_LLM_TESTS=1 and LLM_API_KEY.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (os.environ.get("RUN_LIVE_LLM_TESTS") == "1" and os.environ.get("LLM_API_KEY")), reason="live LLM tests are opt-in")
@requires_built_data
def test_live_a_free_text_report_becomes_an_approved_plan_end_to_end():
    store = WorldStateStore(SqlAlchemyWorldStateRepository("sqlite:///:memory:"))
    orch = Orchestrator(store, SensingAgent(), compliance_rules=RULES_LOW_THRESHOLD)  # real Gemini from the environment
    sim = store.create("LIVE_SUEZ").simulation_id
    out = orch.run(sim, "The container vessel Ever Forward has run aground in the Suez Canal, blocking all traffic. Salvage teams expect about 10 days to refloat it.",
                   PRODUCT, as_of_date=AS_OF)
    assert out.outcome == "AWAITING_APPROVAL", (out.outcome, out.message)
    assert out.state.disrupted_route_ids() and all("SUEZ" in r for r in out.state.disrupted_route_ids())
    assert orch.approve(sim, "live.test").status == SimulationStatus.COMPLETED
