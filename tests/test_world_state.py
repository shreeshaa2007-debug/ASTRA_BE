"""Phase 12 tests — Shared World State, brief §21 "STATE" category and §14
(create / get / update / reset, isolated by simulation_id).

Most tests run against an in-memory database and a hand-made baseline, so they
need no built data. The last section drives the real agents through the state
machine and is skipped if their data isn't built.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from itertools import count
from pathlib import Path

import pytest

from backend.database.world_state_repository import SqlAlchemyWorldStateRepository
from backend.optimization.engine import PrototypeOptimizationEngine
from backend.schemas.entities import RouteStatus, SupplierStatus
from backend.schemas.optimization import OptimizationParameters, OptimizationProblem, OptimizationSolution, SupplierOption, WarehouseState
from backend.schemas.world_state import (
    ApprovalDecision,
    ApprovalStatus,
    ComplianceStatus,
    DisruptionEvent,
    InventoryAssessment,
    SimulationStatus,
)
from backend.services.world_state import (
    MAX_REPLANS,
    Baseline,
    CheckpointViolationError,
    ConcurrentModificationError,
    InvalidTransitionError,
    SimulationExistsError,
    SimulationNotFoundError,
    UnknownCheckpointError,
    WorldStateStore,
    state_changes_for_event,
)

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
SUEZ = ["SHA-ROT-SUEZ", "SIN-ROT-SUEZ"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_baseline(**overrides) -> Baseline:
    data = dict(
        route_status={"SHA-ROT-SUEZ": RouteStatus.NORMAL, "SIN-ROT-SUEZ": RouteStatus.NORMAL, "SHA-ROT-CAPE": RouteStatus.ALTERNATIVE},
        supplier_status={"S1": SupplierStatus.ACTIVE, "S2": SupplierStatus.REDUCED},
        tariffs={"CHN": 2.18, "IND": 4.59},
    )
    data.update(overrides)
    return Baseline(**data)


def make_store(repo=None, baseline: Baseline | None = None) -> WorldStateStore:
    ticks = count()  # every write gets a distinct, increasing timestamp
    return WorldStateStore(
        repo or SqlAlchemyWorldStateRepository("sqlite:///:memory:"),
        baseline_loader=lambda: baseline or make_baseline(),
        clock=lambda: T0 + timedelta(seconds=next(ticks)),
    )


@pytest.fixture
def store() -> WorldStateStore:
    return make_store()


def suez_event(event_id: str = "E1", routes=None, suppliers=None) -> DisruptionEvent:
    return DisruptionEvent(
        event_id=event_id, event_type="canal_closure", location="Suez Canal", severity="CRITICAL", start_date=T0,
        estimated_duration=14, affected_routes=SUEZ if routes is None else routes, affected_suppliers=suppliers or [], confidence=0.95,
    )


def solved_plan(capacity: int = 500) -> OptimizationSolution:
    """A real solution from the real engine: OPTIMAL when capacity covers the 100 units needed, INFEASIBLE otherwise."""
    problem = OptimizationProblem(
        product_id="X",
        parameters=OptimizationParameters(planning_horizon_days=45, internal_transfer_cost_per_unit=5.0, internal_transfer_days=2),
        warehouses=[WarehouseState(warehouse_id="W", current_stock=0, safety_stock=0, forecast_demand=100, stockout_risk="HIGH")],
        suppliers=[SupplierOption(supplier_id="S1", supplier_name="S1", region="Testland", capacity=capacity, base_unit_cost=10,
                                  landed_unit_cost=10, tariff_rate_pct=0, lead_time_days=5, reliability=0.9, risk_level="LOW")],
    )
    return PrototypeOptimizationEngine().optimize_supply_chain(problem)


def verdict(status: str) -> ComplianceStatus:
    return ComplianceStatus(status=status, reason=f"test {status}", requires_human=status == "ESCALATED",
                            checks=[{"name": "cost_within_threshold", "passed": status != "ESCALATED"}])


def human(who: str = "logistics.director") -> ApprovalDecision:
    return ApprovalDecision(decided_by=who, decided_at=T0 + timedelta(days=1), note="reviewed")


def run_to_optimized(store: WorldStateStore, sim_id: str, plan: OptimizationSolution | None = None):
    state = store.get(sim_id)
    store.commit(sim_id, "event_sensed", state_changes_for_event(state, suez_event()))
    store.commit(sim_id, "agents_assessed", {"inventory_status": [
        InventoryAssessment(warehouse="Mumbai", product="X", forecast_demand=120.5, current_stock=40, stockout_risk="HIGH")]})
    return store.commit(sim_id, "plan_optimized", {"current_plan": plan or solved_plan()})


def run_to_compliance(store: WorldStateStore, sim_id: str, status: str):
    run_to_optimized(store, sim_id)
    return store.commit(sim_id, "compliance_checked", {"compliance_status": verdict(status)})


# --------------------------------------------------------------------------- #
# CREATE / GET
# --------------------------------------------------------------------------- #
def test_create_starts_a_simulation_at_the_baseline(store):
    s = store.create("SUEZ_CLOSURE")
    assert s.simulation_id.startswith("sim-")
    assert (s.status, s.version, s.scenario_type) == (SimulationStatus.CREATED, 0, "SUEZ_CLOSURE")
    assert s.route_status == make_baseline().route_status and s.supplier_status == make_baseline().supplier_status
    assert s.tariffs == {"CHN": 2.18, "IND": 4.59}
    assert s.current_plan is None and s.approval_status == ApprovalStatus.NOT_EVALUATED and s.current_disruptions == []


def test_created_ids_are_unique(store):
    assert len({store.create("A").simulation_id for _ in range(20)}) == 20


def test_create_with_an_explicit_id_that_already_exists_is_refused(store):
    store.create("A", simulation_id="sim-fixed")
    with pytest.raises(SimulationExistsError):
        store.create("B", simulation_id="sim-fixed")
    assert store.get("sim-fixed").scenario_type == "A"  # the original is untouched


def test_get_unknown_simulation_raises_with_the_api_error_code(store):
    with pytest.raises(SimulationNotFoundError) as exc:
        store.get("sim-nope")
    assert exc.value.error_code == "SIMULATION_NOT_FOUND"


def test_create_rejects_a_blank_scenario_type(store):
    with pytest.raises(ValueError):
        store.create("   ")


def test_get_returns_a_copy_so_mutating_it_cannot_change_the_stored_state(store):
    sim = store.create("A").simulation_id
    copy = store.get(sim)
    copy.route_status["SHA-ROT-SUEZ"] = RouteStatus.DISRUPTED
    copy.status = SimulationStatus.COMPLETED
    assert store.get(sim).route_status["SHA-ROT-SUEZ"] == RouteStatus.NORMAL
    assert store.get(sim).status == SimulationStatus.CREATED


# --------------------------------------------------------------------------- #
# isolation between simulations
# --------------------------------------------------------------------------- #
def test_simulations_do_not_share_mutable_state(store):
    a, b = store.create("SUEZ").simulation_id, store.create("SUPPLIER_FAILURE").simulation_id
    store.commit(a, "event_sensed", state_changes_for_event(store.get(a), suez_event()))
    assert store.get(a).disrupted_route_ids() == frozenset(SUEZ)
    assert store.get(b).disrupted_route_ids() == frozenset() and store.get(b).status == SimulationStatus.CREATED
    assert store.get(b).current_disruptions == []


def test_one_baseline_object_is_not_shared_between_simulations(store):
    baseline = make_baseline()
    a = store.create("A", baseline=baseline).simulation_id
    store.commit(a, "event_sensed", state_changes_for_event(store.get(a), suez_event()))
    assert store.create("B", baseline=baseline).route_status["SHA-ROT-SUEZ"] == RouteStatus.NORMAL
    assert baseline.route_status["SHA-ROT-SUEZ"] == RouteStatus.NORMAL


def test_concurrent_simulations_on_separate_threads_stay_isolated():
    store = make_store()
    ids = [store.create(f"S{i}").simulation_id for i in range(8)]

    def drive(i_sim):
        i, sim = i_sim
        state = store.get(sim)
        store.commit(sim, "event_sensed", state_changes_for_event(state, suez_event(event_id=f"E{i}")))
        return store.commit(sim, "agents_assessed", {"demand_forecasts": [
            {"product_id": "X", "warehouse_id": "W", "horizon_days": 14 + i, "forecast_demand": float(i), "model_version": "v"}]}).version

    with ThreadPoolExecutor(8) as pool:
        assert list(pool.map(drive, enumerate(ids))) == [2] * 8
    for i, sim in enumerate(ids):
        s = store.get(sim)
        assert [d.event_id for d in s.current_disruptions] == [f"E{i}"]
        assert s.demand_forecasts[0].horizon_days == 14 + i


def test_two_writers_racing_from_the_same_version_exactly_one_wins():
    store = make_store()
    sim = store.create("A").simulation_id
    changes = state_changes_for_event(store.get(sim), suez_event())
    barrier, outcomes = threading.Barrier(6), []

    def attempt(_):
        barrier.wait()
        try:
            store.commit(sim, "event_sensed", changes, expected_version=0)
            outcomes.append("won")
        except (ConcurrentModificationError, InvalidTransitionError):
            outcomes.append("lost")  # stale version, or already RUNNING by the time it got in

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(attempt, range(6)))
    assert outcomes.count("won") == 1 and outcomes.count("lost") == 5
    assert store.get(sim).version == 1


# --------------------------------------------------------------------------- #
# checkpoint discipline — what a commit may and may not do
# --------------------------------------------------------------------------- #
def test_unknown_checkpoint_is_refused(store):
    sim = store.create("A").simulation_id
    with pytest.raises(UnknownCheckpointError):
        store.commit(sim, "just_let_me_write", {"error": "x"})


def test_a_checkpoint_cannot_change_fields_it_does_not_own_and_nothing_is_written(store):
    sim = store.create("A").simulation_id
    with pytest.raises(CheckpointViolationError, match="current_plan"):
        store.commit(sim, "event_sensed", {**state_changes_for_event(store.get(sim), suez_event()), "current_plan": solved_plan()})
    assert store.get(sim).version == 0 and store.get(sim).current_disruptions == []
    assert [h.checkpoint for h in store.history(sim)] == ["simulation_created"]


@pytest.mark.parametrize("field", ["status", "version", "simulation_id", "approval_status", "timestamp"])
def test_bookkeeping_fields_cannot_be_written_through_a_checkpoint(store, field):
    sim = store.create("A").simulation_id
    with pytest.raises(CheckpointViolationError):
        store.commit(sim, "event_sensed", {"current_disruptions": [suez_event()], field: "x"})


def test_a_commit_with_no_changes_is_refused(store):
    with pytest.raises(CheckpointViolationError, match="no changes"):
        store.commit(store.create("A").simulation_id, "event_sensed", {})


def test_a_checkpoint_out_of_order_is_an_invalid_transition(store):
    sim = store.create("A").simulation_id
    with pytest.raises(InvalidTransitionError) as exc:
        store.commit(sim, "plan_optimized", {"current_plan": solved_plan()})  # nothing sensed yet
    assert exc.value.error_code == "INVALID_STATE_TRANSITION"


def test_invalid_values_are_rejected_and_nothing_is_written(store):
    sim = store.create("A").simulation_id
    with pytest.raises(CheckpointViolationError):
        store.commit(sim, "event_sensed", {"current_disruptions": [suez_event()], "route_status": {"SHA-ROT-SUEZ": "ON_FIRE"}})
    assert store.get(sim).version == 0


def test_a_stale_expected_version_is_refused_rather_than_overwriting(store):
    sim = store.create("A").simulation_id
    store.commit(sim, "event_sensed", state_changes_for_event(store.get(sim), suez_event()))
    with pytest.raises(ConcurrentModificationError) as exc:
        store.commit(sim, "agents_assessed", {"shipment_status": {}}, expected_version=0)
    assert exc.value.error_code == "STATE_CONFLICT"
    assert store.get(sim).version == 1


def test_history_records_every_checkpoint_in_order_with_actor_and_fields(store):
    sim = store.create("A", actor="scenario-loader").simulation_id
    run_to_optimized(store, sim)
    store.commit(sim, "compliance_checked", {"compliance_status": verdict("APPROVED")}, actor="compliance-agent")
    history = store.history(sim)
    assert [(h.version, h.checkpoint) for h in history] == [
        (0, "simulation_created"), (1, "event_sensed"), (2, "agents_assessed"), (3, "plan_optimized"), (4, "compliance_checked")]
    assert history[0].actor == "scenario-loader" and history[4].actor == "compliance-agent"
    assert history[1].changed_fields == ["current_disruptions", "route_status", "supplier_status"]
    assert [h.at for h in history] == sorted(h.at for h in history)


# --------------------------------------------------------------------------- #
# the approval gate and the bounded replan
# --------------------------------------------------------------------------- #
def test_in_policy_plan_finalizes_without_a_human(store):
    sim = store.create("A").simulation_id
    run_to_compliance(store, sim, "APPROVED")
    final = store.commit(sim, "plan_finalized", {"approval_status": ApprovalStatus.NOT_REQUIRED})
    assert (final.status, final.version) == (SimulationStatus.COMPLETED, 5)
    assert final.current_plan.status == "OPTIMAL" and final.approval_decision is None


def test_escalated_plan_waits_for_a_named_human_then_finalizes(store):
    sim = store.create("A").simulation_id
    run_to_compliance(store, sim, "ESCALATED")
    waiting = store.commit(sim, "approval_requested", {"approval_status": ApprovalStatus.PENDING})
    assert (waiting.status, waiting.approval_status) == (SimulationStatus.AWAITING_APPROVAL, ApprovalStatus.PENDING)

    final = store.commit(sim, "plan_finalized", {"approval_status": ApprovalStatus.APPROVED, "approval_decision": human()}, actor="logistics.director")
    assert final.status == SimulationStatus.COMPLETED and final.approval_decision.decided_by == "logistics.director"


def test_an_escalated_plan_cannot_be_finalized_as_if_it_needed_no_approval(store):
    sim = store.create("A").simulation_id
    run_to_compliance(store, sim, "ESCALATED")
    with pytest.raises(CheckpointViolationError, match="never sent for approval"):
        store.commit(sim, "plan_finalized", {"approval_status": ApprovalStatus.NOT_REQUIRED})
    assert store.get(sim).status == SimulationStatus.RUNNING


def test_an_escalated_plan_cannot_skip_the_approval_step(store):
    sim = store.create("A").simulation_id
    run_to_compliance(store, sim, "ESCALATED")
    with pytest.raises(CheckpointViolationError, match="never sent for approval"):
        store.commit(sim, "plan_finalized", {"approval_status": ApprovalStatus.APPROVED, "approval_decision": human()})


def test_approval_needs_a_named_approver(store):
    sim = store.create("A").simulation_id
    run_to_compliance(store, sim, "ESCALATED")
    store.commit(sim, "approval_requested", {"approval_status": ApprovalStatus.PENDING})
    with pytest.raises(CheckpointViolationError, match="named approver"):
        store.commit(sim, "plan_finalized", {"approval_status": ApprovalStatus.APPROVED})
    with pytest.raises(CheckpointViolationError):  # an empty name is not a name
        store.commit(sim, "plan_finalized", {"approval_status": ApprovalStatus.APPROVED, "approval_decision": {"decided_by": "", "decided_at": T0}})
    assert store.get(sim).status == SimulationStatus.AWAITING_APPROVAL


def test_a_plan_cannot_be_sent_for_approval_unless_compliance_escalated_it(store):
    sim = store.create("A").simulation_id
    run_to_compliance(store, sim, "APPROVED")
    with pytest.raises(CheckpointViolationError, match="ESCALATED"):
        store.commit(sim, "approval_requested", {"approval_status": ApprovalStatus.PENDING})


def test_a_compliance_rejected_plan_can_never_be_finalized(store):
    sim = store.create("A").simulation_id
    run_to_compliance(store, sim, "REJECTED")
    with pytest.raises(CheckpointViolationError, match="REJECTED"):
        store.commit(sim, "plan_finalized", {"approval_status": ApprovalStatus.NOT_REQUIRED})
    with pytest.raises(CheckpointViolationError, match="REJECTED"):  # not even with a human's signature
        store.commit(sim, "plan_finalized", {"approval_status": ApprovalStatus.APPROVED, "approval_decision": human()})


def test_an_infeasible_plan_is_recorded_for_its_diagnosis_but_can_go_no_further(store):
    sim = store.create("A").simulation_id
    state = run_to_optimized(store, sim, plan=solved_plan(capacity=10))  # only 10 of 100 units coverable
    assert state.current_plan.status == "INFEASIBLE" and state.current_plan.diagnostics["total_shortfall_units"] == 90
    with pytest.raises(CheckpointViolationError, match="OPTIMAL"):
        store.commit(sim, "compliance_checked", {"compliance_status": verdict("APPROVED")})
    with pytest.raises(CheckpointViolationError, match="OPTIMAL"):
        store.commit(sim, "plan_finalized", {"approval_status": ApprovalStatus.NOT_REQUIRED})


def test_human_rejection_is_terminal(store):
    sim = store.create("A").simulation_id
    run_to_compliance(store, sim, "ESCALATED")
    store.commit(sim, "approval_requested", {"approval_status": ApprovalStatus.PENDING})
    done = store.commit(sim, "plan_rejected_by_human", {"approval_status": ApprovalStatus.REJECTED, "approval_decision": human("cfo")})
    assert done.status == SimulationStatus.REJECTED and done.approval_decision.decided_by == "cfo"
    with pytest.raises(InvalidTransitionError):
        store.commit(sim, "plan_finalized", {"approval_status": ApprovalStatus.APPROVED, "approval_decision": human()})


def test_a_rejection_must_be_recorded_as_a_rejection(store):
    sim = store.create("A").simulation_id
    run_to_compliance(store, sim, "ESCALATED")
    store.commit(sim, "approval_requested", {"approval_status": ApprovalStatus.PENDING})
    with pytest.raises(CheckpointViolationError, match="REJECTED"):
        store.commit(sim, "plan_rejected_by_human", {"approval_status": ApprovalStatus.APPROVED, "approval_decision": human()})


def test_a_rejected_plan_is_replanned_once_then_the_run_fails_instead_of_looping(store):
    assert MAX_REPLANS == 1
    sim = store.create("A").simulation_id
    run_to_compliance(store, sim, "REJECTED")

    replanned = store.commit(sim, "replan_requested", {"current_plan": None, "compliance_status": None, "replan_count": 1})
    assert (replanned.replan_count, replanned.current_plan, replanned.compliance_status) == (1, None, None)
    assert replanned.status == SimulationStatus.RUNNING

    store.commit(sim, "plan_optimized", {"current_plan": solved_plan()})
    store.commit(sim, "compliance_checked", {"compliance_status": verdict("REJECTED")})
    with pytest.raises(CheckpointViolationError, match="replan limit"):
        store.commit(sim, "replan_requested", {"current_plan": None, "compliance_status": None, "replan_count": 2})

    failed = store.commit(sim, "run_failed", {"error": "infeasible_after_replan"})
    assert (failed.status, failed.error) == (SimulationStatus.FAILED, "infeasible_after_replan")


def test_a_replan_is_only_for_a_compliance_rejection(store):
    sim = store.create("A").simulation_id
    run_to_compliance(store, sim, "ESCALATED")
    with pytest.raises(CheckpointViolationError, match="REJECTED"):
        store.commit(sim, "replan_requested", {"current_plan": None, "compliance_status": None, "replan_count": 1})


def test_a_replan_must_discard_the_rejected_plan_and_count_itself(store):
    sim = store.create("A").simulation_id
    run_to_compliance(store, sim, "REJECTED")
    with pytest.raises(CheckpointViolationError, match="clear"):
        store.commit(sim, "replan_requested", {"replan_count": 1})  # plan and verdict left in place
    with pytest.raises(CheckpointViolationError, match="exactly 1"):
        store.commit(sim, "replan_requested", {"current_plan": None, "compliance_status": None, "replan_count": 0})


def test_run_failed_needs_a_reason_and_is_terminal(store):
    sim = store.create("A").simulation_id
    with pytest.raises(CheckpointViolationError, match="non-empty error"):
        store.commit(sim, "run_failed", {"error": ""})
    assert store.commit(sim, "run_failed", {"error": "sensing output invalid"}).status == SimulationStatus.FAILED
    for checkpoint, changes in (("event_sensed", {"current_disruptions": [suez_event()]}), ("run_failed", {"error": "again"})):
        with pytest.raises(InvalidTransitionError):
            store.commit(sim, checkpoint, changes)


def test_a_finished_simulation_accepts_no_more_checkpoints(store):
    sim = store.create("A").simulation_id
    run_to_compliance(store, sim, "APPROVED")
    store.commit(sim, "plan_finalized", {"approval_status": ApprovalStatus.NOT_REQUIRED})
    with pytest.raises(InvalidTransitionError):
        store.commit(sim, "agents_assessed", {"shipment_status": {}})


# --------------------------------------------------------------------------- #
# RESET
# --------------------------------------------------------------------------- #
def test_reset_returns_to_the_creation_state_and_keeps_the_audit_trail(store):
    sim = store.create("SUEZ").simulation_id
    run_to_compliance(store, sim, "REJECTED")
    store.commit(sim, "replan_requested", {"current_plan": None, "compliance_status": None, "replan_count": 1})
    before = store.get(sim)

    after = store.reset(sim)
    assert after.status == SimulationStatus.CREATED
    assert after.current_disruptions == [] and after.current_plan is None and after.compliance_status is None
    assert after.replan_count == 0 and after.inventory_status == [] and after.approval_status == ApprovalStatus.NOT_EVALUATED
    assert after.route_status == make_baseline().route_status and after.disrupted_route_ids() == frozenset()
    assert after.version == before.version + 1 and after.timestamp > before.timestamp
    assert after.created_at == before.created_at and after.simulation_id == sim

    checkpoints = [h.checkpoint for h in store.history(sim)]
    assert checkpoints[0] == "simulation_created" and checkpoints[-1] == "simulation_reset" and "compliance_checked" in checkpoints


def test_a_finished_simulation_can_be_reset_and_run_again(store):
    sim = store.create("A").simulation_id
    run_to_compliance(store, sim, "APPROVED")
    store.commit(sim, "plan_finalized", {"approval_status": ApprovalStatus.NOT_REQUIRED})
    assert store.reset(sim).status == SimulationStatus.CREATED
    run_to_compliance(store, sim, "APPROVED")
    assert store.commit(sim, "plan_finalized", {"approval_status": ApprovalStatus.NOT_REQUIRED}).status == SimulationStatus.COMPLETED


def test_reset_restores_the_baseline_the_simulation_was_created_with_not_todays():
    live = {"baseline": make_baseline()}
    store = WorldStateStore(SqlAlchemyWorldStateRepository("sqlite:///:memory:"), baseline_loader=lambda: live["baseline"])
    sim = store.create("A").simulation_id
    live["baseline"] = make_baseline(tariffs={"CHN": 99.0}, route_status={"OTHER": RouteStatus.NORMAL})  # data files changed since
    assert store.reset(sim).tariffs == {"CHN": 2.18, "IND": 4.59}


def test_reset_of_an_unknown_simulation_raises(store):
    with pytest.raises(SimulationNotFoundError):
        store.reset("sim-nope")


# --------------------------------------------------------------------------- #
# applying a sensed event
# --------------------------------------------------------------------------- #
def test_event_marks_its_routes_and_suppliers_disrupted_and_feeds_the_agents_inputs(store):
    state = store.create("A")
    changes = state_changes_for_event(state, suez_event(suppliers=["S1"]))
    after = store.commit(state.simulation_id, "event_sensed", changes)
    assert after.disrupted_route_ids() == frozenset(SUEZ)
    assert after.route_status["SHA-ROT-CAPE"] == RouteStatus.ALTERNATIVE  # unaffected routes keep their status
    assert after.disrupted_supplier_ids() == frozenset({"S1"})
    assert after.supplier_status["S2"] == SupplierStatus.REDUCED


def test_a_second_version_of_the_same_event_replaces_it(store):
    state = store.create("A")
    changes = state_changes_for_event(state, suez_event(routes=["SHA-ROT-SUEZ"]))
    state = store.commit(state.simulation_id, "event_sensed", changes)
    updated = state_changes_for_event(state, suez_event(routes=SUEZ))
    assert [d.event_id for d in updated["current_disruptions"]] == ["E1"]
    assert len(updated["current_disruptions"]) == 1


def test_an_event_naming_an_unknown_route_or_supplier_is_an_error_not_a_skip(store):
    state = store.create("A")
    with pytest.raises(ValueError, match=r"unknown route ids \['TYPO-1'\]"):
        state_changes_for_event(state, suez_event(routes=["SHA-ROT-SUEZ", "TYPO-1"]))
    with pytest.raises(ValueError, match="unknown supplier ids"):
        state_changes_for_event(state, suez_event(suppliers=["S99"]))


def test_disruption_event_validates_its_fields():
    with pytest.raises(ValueError):
        DisruptionEvent(event_id="E", event_type="x", location="y", severity="CRITICAL", start_date=T0, estimated_duration=1, confidence=1.5)
    with pytest.raises(ValueError):
        DisruptionEvent(event_id="E", event_type="x", location="y", severity="APOCALYPSE", start_date=T0, estimated_duration=1, confidence=0.5)


# --------------------------------------------------------------------------- #
# persistence, listing, deletion
# --------------------------------------------------------------------------- #
def test_state_survives_a_new_repository_and_store_on_the_same_database_file(tmp_path):
    url = f"sqlite:///{tmp_path / 'ws.db'}"
    first = make_store(SqlAlchemyWorldStateRepository(url))
    sim = first.create("SUEZ").simulation_id
    run_to_compliance(first, sim, "ESCALATED")

    reopened = make_store(SqlAlchemyWorldStateRepository(url))
    state = reopened.get(sim)
    assert (state.status, state.version) == (SimulationStatus.RUNNING, 4)
    assert state.compliance_status.requires_human and [h.checkpoint for h in reopened.history(sim)][-1] == "compliance_checked"


def test_the_repository_reads_its_url_from_database_url(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'from_env.db'}")
    sim = make_store(SqlAlchemyWorldStateRepository()).create("A").simulation_id
    assert (tmp_path / "from_env.db").exists()
    assert make_store(SqlAlchemyWorldStateRepository()).get(sim).simulation_id == sim


def test_the_optimization_solution_round_trips_through_the_database_intact(store):
    plan = solved_plan()
    sim = store.create("A").simulation_id
    run_to_optimized(store, sim, plan=plan)
    stored = store.get(sim).current_plan
    assert stored.model_dump() == plan.model_dump()
    assert stored.objective_value == plan.objective_value == 1000 and stored.allocations[0].quantity == 100


def test_list_simulations_filters_by_status_newest_first_and_honors_limit(store):
    ids = [store.create(f"S{i}").simulation_id for i in range(4)]
    run_to_compliance(store, ids[1], "APPROVED")
    store.commit(ids[1], "plan_finalized", {"approval_status": ApprovalStatus.NOT_REQUIRED})
    store.commit(ids[3], "run_failed", {"error": "boom"})  # written last, so most recently updated

    everything = store.list_simulations()
    assert [s.simulation_id for s in everything] == [ids[3], ids[1], ids[2], ids[0]]
    assert [s.simulation_id for s in store.list_simulations(SimulationStatus.COMPLETED)] == [ids[1]]
    assert [s.simulation_id for s in store.list_simulations(SimulationStatus.CREATED)] == [ids[2], ids[0]]
    assert len(store.list_simulations(limit=2)) == 2


def test_delete_removes_the_simulation_and_its_history(store):
    sim = store.create("A").simulation_id
    other = store.create("B").simulation_id
    store.delete(sim)
    with pytest.raises(SimulationNotFoundError):
        store.get(sim)
    with pytest.raises(SimulationNotFoundError):
        store.history(sim)
    with pytest.raises(SimulationNotFoundError):
        store.delete(sim)
    assert store.get(other).scenario_type == "B"


# --------------------------------------------------------------------------- #
# real data and real agents
# --------------------------------------------------------------------------- #
requires_built_data = pytest.mark.skipif(
    not all(Path(p).exists() for p in (
        "data/processed/suppliers.csv", "data/processed/routes.csv", "data/processed/tariffs.csv",
        "data/processed/inventory_multi_warehouse.csv", "data/processed/demand_modeling_panel.csv",
        "ml/artifacts/xgboost_demand/2026.09.1/model.json",
    )),
    reason="run the Phase 3-5 pipelines first",
)


@requires_built_data
def test_real_baseline_reflects_the_processed_datasets():
    from backend.services.world_state import load_baseline

    b = load_baseline()
    assert len(b.route_status) == 8 and b.route_status["SHA-ROT-SUEZ"] == RouteStatus.NORMAL and b.route_status["SHA-ROT-CAPE"] == RouteStatus.ALTERNATIVE
    assert b.supplier_status["S001"] == SupplierStatus.DISRUPTED and b.supplier_status["S002"] == SupplierStatus.REDUCED
    assert b.supplier_status["S003"] == SupplierStatus.ACTIVE
    assert set(b.tariffs) == {"CHN", "IND", "VNM", "TUR", "NLD"} and b.tariffs["IND"] == pytest.approx(4.59)


@requires_built_data
def test_real_agents_drive_a_simulation_through_every_checkpoint_of_the_escalated_path():
    """Suez closure end to end: sensed event -> state -> real Inventory/Sourcing/Logistics agents ->
    real optimizer -> real Compliance rules -> human approval -> final. Every arrow goes through the store."""
    from backend.agents.compliance import tools as compliance_tools
    from backend.agents.inventory.agent import InventoryAgent
    from backend.optimization import tools as optimization_tools

    as_of = "2011-11-30"
    store = WorldStateStore(SqlAlchemyWorldStateRepository("sqlite:///:memory:"))  # real baseline, real clock
    state = store.create("SUEZ_CLOSURE")
    assert state.route_status["SHA-ROT-SUEZ"] == RouteStatus.NORMAL

    event = suez_event(routes=["SHA-ROT-SUEZ", "SIN-ROT-SUEZ", "MUM-ROT-SUEZ", "CHE-ROT-SUEZ"], suppliers=[])
    state = store.commit(state.simulation_id, "event_sensed", state_changes_for_event(state, event))
    assert state.disrupted_route_ids() == frozenset(event.affected_routes)

    inventory = InventoryAgent(horizon_days=45).analyze_product("22197", as_of)
    state = store.commit(state.simulation_id, "agents_assessed", {"inventory_status": [InventoryAssessment(**r) for r in inventory]})
    assert {a.warehouse for a in state.inventory_status} == {"Mumbai", "Chennai", "Delhi"}

    problem = optimization_tools.build_problem_from_agents("22197", as_of, disrupted_route_ids=state.disrupted_route_ids())
    plan = optimization_tools.optimize_supply_chain(problem)
    state = store.commit(state.simulation_id, "plan_optimized", {"current_plan": plan})
    assert state.current_plan.status == "OPTIMAL" and not any(a.route_id in state.disrupted_route_ids() for a in state.current_plan.allocations)

    rules = {**compliance_tools.load_rules(), "approval_threshold": 100_000}  # this plan costs ~465k: high-impact under this policy
    verdict_dict = compliance_tools.validate_plan(
        {"supplier_allocations": [{"supplier_id": a.supplier_id, "product_id": "22197", "quantity": a.quantity} for a in plan.allocations],
         "route_id": None, "total_cost": plan.objective_value}, rules)
    assert verdict_dict["status"] == "ESCALATED"
    state = store.commit(state.simulation_id, "compliance_checked", {"compliance_status": ComplianceStatus(**verdict_dict)})
    state = store.commit(state.simulation_id, "approval_requested", {"approval_status": ApprovalStatus.PENDING})
    assert state.status == SimulationStatus.AWAITING_APPROVAL

    state = store.commit(state.simulation_id, "plan_finalized", {"approval_status": ApprovalStatus.APPROVED, "approval_decision": human()})
    assert state.status == SimulationStatus.COMPLETED
    assert [h.checkpoint for h in store.history(state.simulation_id)] == [
        "simulation_created", "event_sensed", "agents_assessed", "plan_optimized", "compliance_checked", "approval_requested", "plan_finalized"]

    # the state the run left behind is exactly what a fresh reader gets
    assert store.get(state.simulation_id).current_plan.objective_value == plan.objective_value
    assert store.reset(state.simulation_id).route_status["SHA-ROT-SUEZ"] == RouteStatus.NORMAL
