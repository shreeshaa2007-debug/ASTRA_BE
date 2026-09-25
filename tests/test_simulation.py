"""Phase 17 tests — scenario simulation, brief §21 "SIMULATION" category.

Three layers:
  * the scenario definitions (backend/config/scenarios.yaml) against the real network;
  * the comparison arithmetic (`evaluate_inaction`), on a hand-made plan whose numbers can be checked by hand;
  * the comparison on the real data and through the real API, including the property that matters most:
    a scenario's "mitigated" case is exactly the plan the live pipeline produces for the same trigger.

The real Inventory, Sourcing, Logistics and Optimization code runs; the LLM is a fake that is never called
(scenario triggers are structured, so Sensing skips the LLM — but still validates them).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.agents.sensing.agent import SensingAgent
from backend.api.context import AppContext
from backend.api.main import create_app
from backend.api.runs import RunRegistry
from backend.database.world_state_repository import SqlAlchemyWorldStateRepository
from backend.orchestration import Orchestrator, adapters
from backend.schemas.entities import RouteStatus, SupplierStatus
from backend.schemas.optimization import (
    Allocation,
    OptimizationParameters,
    OptimizationProblem,
    OptimizationSolution,
    RouteOption,
    SupplierOption,
    WarehouseState,
)
from backend.schemas.world_state import SimulationStatus
from backend.services.world_state import WorldStateStore, load_baseline
from backend.simulation import (
    ScenarioNotModeledError,
    UnknownScenarioError,
    WorldSpec,
    baseline_world,
    compare_scenario,
    compare_state,
    evaluate_inaction,
    get_scenario,
    load_scenarios,
    world_after_event,
)
from backend.simulation.scenarios import ScenarioDefinition

requires_built_data = pytest.mark.skipif(
    not all(Path(p).exists() for p in (
        "data/processed/suppliers.csv", "data/processed/routes.csv", "data/processed/tariffs.csv", "data/processed/disruptions.csv",
        "data/processed/inventory_multi_warehouse.csv", "data/processed/demand_modeling_panel.csv",
        "ml/artifacts/xgboost_demand/2026.09.1/model.json",
    )),
    reason="run the Phase 3-5 pipelines first",
)

PRODUCT = "22197"
SUEZ = ["CHE-ROT-SUEZ", "MUM-ROT-SUEZ", "SHA-ROT-SUEZ", "SIN-ROT-SUEZ"]
MODELED = ["SUEZ_CLOSURE", "SUPPLIER_FAILURE", "SEVERE_WEATHER", "TARIFF_INCREASE"]
UNMODELED = ["PORT_CONGESTION", "DEMAND_SURGE"]
RULES_OPEN = {"rejected_suppliers": [], "restricted_countries": [], "approval_threshold": 10_000_000}


class NeverCalledLLM:
    model = "never-called"

    def generate_json(self, **kwargs):  # scenario triggers are structured: Sensing must not reach the LLM
        raise AssertionError("a structured scenario trigger must not call the LLM")


def make(rules=RULES_OPEN):
    store = WorldStateStore(SqlAlchemyWorldStateRepository("sqlite:///:memory:"))
    orchestrator = Orchestrator(store, SensingAgent(llm=NeverCalledLLM()), compliance_rules=rules)
    ctx = AppContext(store, orchestrator, RunRegistry(None))  # runs inline
    return TestClient(create_app(context=ctx), raise_server_exceptions=False), store, orchestrator


def data(response, status=200):
    assert response.status_code == status, response.text
    return response.json()["data"]


def error(response, status: int, code: str) -> dict:
    assert response.status_code == status, response.text
    err = response.json()["error"]
    assert err["error_code"] == code and err["message"], err
    return err


def run_scenario_through_pipeline(scenario_id: str, product: str = PRODUCT, rules=RULES_OPEN):
    """What POST /api/scenarios/{id}/run does, against a store we can inspect."""
    _, store, orchestrator = make(rules)
    scenario = get_scenario(scenario_id)
    state = store.create(scenario_id)
    overrides = scenario.tariff_overrides(load_baseline().tariffs) or None
    outcome = orchestrator.run(state.simulation_id, scenario.trigger(), product, tariff_overrides=overrides)
    return outcome


# =========================================================================== #
# the definitions
# =========================================================================== #
@requires_built_data
def test_the_scenario_file_defines_four_modeled_scenarios_and_two_that_say_why_they_are_not():
    scenarios = load_scenarios()
    assert [s for s, d in scenarios.items() if d.modeled] == MODELED
    assert [s for s, d in scenarios.items() if not d.modeled] == UNMODELED
    for sid in UNMODELED:
        assert len(scenarios[sid].not_modeled_reason) > 40 and scenarios[sid].event is None


@requires_built_data
@pytest.mark.parametrize("scenario_id", MODELED)
def test_every_modeled_scenario_is_valid_against_the_real_network(scenario_id):
    scenario = get_scenario(scenario_id)
    world_after_event(scenario.disruption_event(), scenario.tariff_overrides(load_baseline().tariffs))  # unknown route/supplier ids raise here
    assert scenario.default_product_id == PRODUCT


def test_a_modeled_scenario_needs_an_event_and_an_unmodeled_one_needs_a_reason():
    base = dict(scenario_id="X", label="x", description="x", default_product_id="1")
    with pytest.raises(ValueError, match="needs an event"):
        ScenarioDefinition(modeled=True, **base)
    with pytest.raises(ValueError, match="must say why"):
        ScenarioDefinition(modeled=False, **base)
    with pytest.raises(ValueError, match="no defined effect"):
        ScenarioDefinition(modeled=True, event=dict(event_type="alien_invasion", location="x", severity="LOW", estimated_duration=1), **base)


def test_an_unknown_scenario_is_reported_with_the_known_ones():
    with pytest.raises(UnknownScenarioError, match="SUEZ_CLOSURE"):
        get_scenario("NOPE")


@requires_built_data
def test_tariff_add_pct_is_added_to_the_baseline_rate_and_an_unknown_country_is_refused():
    scenario = get_scenario("TARIFF_INCREASE")
    base = load_baseline().tariffs
    assert scenario.tariff_overrides(base) == {"TUR": pytest.approx(base["TUR"] + 25.0)}
    with pytest.raises(ValueError, match="no baseline tariff"):
        scenario.tariff_overrides({"CHN": 1.0})


@requires_built_data
@pytest.mark.parametrize("scenario_id", UNMODELED)
def test_an_unmodeled_scenario_refuses_to_simulate_rather_than_invent_a_number(scenario_id):
    scenario = get_scenario(scenario_id)
    with pytest.raises(ScenarioNotModeledError, match="not modeled"):
        compare_scenario(scenario)
    with pytest.raises(ScenarioNotModeledError):
        scenario.trigger()


# =========================================================================== #
# a scenario changes the world the way the pipeline does
# =========================================================================== #
@requires_built_data
def test_a_scenario_is_applied_by_the_pipelines_own_rule():
    base = baseline_world()
    assert base.disrupted_routes == frozenset() and base.disrupted_suppliers == frozenset({"S001"})  # S001 is disrupted in the data

    def world(sid):
        s = get_scenario(sid)
        return world_after_event(s.disruption_event(), s.tariff_overrides(load_baseline().tariffs))

    suez = world("SUEZ_CLOSURE")
    assert suez.disrupted_routes == frozenset(SUEZ) and suez.disrupted_suppliers == base.disrupted_suppliers
    failure = world("SUPPLIER_FAILURE")
    assert failure.disrupted_routes == frozenset() and failure.disrupted_suppliers == frozenset({"S001", "S007"})
    weather = world("SEVERE_WEATHER")
    assert weather.disrupted_routes == frozenset({"MUM-ROT-SUEZ"})
    tariff = world("TARIFF_INCREASE")  # a tariff lists S007 as exposed, but EVENT_EFFECTS blocks nothing
    assert tariff.disrupted_routes == base.disrupted_routes and tariff.disrupted_suppliers == base.disrupted_suppliers
    assert tariff.tariff_rates["TUR"] == pytest.approx(base.tariff_rates["TUR"] + 25.0)


@requires_built_data
@pytest.mark.parametrize("scenario_id", MODELED)
def test_a_scenarios_mitigated_case_is_exactly_the_plan_the_live_pipeline_produces_for_the_same_trigger(scenario_id):
    """The whole point of a scenario framework: the simulation and the pipeline cannot disagree."""
    outcome = run_scenario_through_pipeline(scenario_id)
    assert outcome.outcome == "COMPLETED", outcome.message
    state = outcome.state
    comparison = compare_scenario(get_scenario(scenario_id))
    assert comparison.mitigated.status == "OPTIMAL"
    assert comparison.mitigated.spend == pytest.approx(adapters.plan_spend(state.current_plan))
    assert comparison.mitigated.units_delivered == sum(a.quantity for a in state.current_plan.allocations)
    # the world the comparison evaluated is the world the pipeline's state ended up in
    assert set(comparison.newly_disrupted_routes) == {r for r, s in state.route_status.items() if s == RouteStatus.DISRUPTED}
    assert set(comparison.newly_disrupted_suppliers) == {s for s, st in state.supplier_status.items() if st == SupplierStatus.DISRUPTED} - {"S001"}


# =========================================================================== #
# the arithmetic, on a plan small enough to check by hand
# =========================================================================== #
def toy_baseline() -> OptimizationSolution:
    """Warehouse A must import (100 in stock + 160 bought - 200 demand = 60 >= safety 50); B has plenty."""
    problem = OptimizationProblem(
        product_id="P",
        parameters=OptimizationParameters(planning_horizon_days=30, internal_transfer_cost_per_unit=5.0, internal_transfer_days=2),
        warehouses=[
            WarehouseState(warehouse_id="A", current_stock=100, safety_stock=50, forecast_demand=200, stockout_risk="HIGH"),
            WarehouseState(warehouse_id="B", current_stock=300, safety_stock=50, forecast_demand=100, stockout_risk="LOW"),
        ],
        suppliers=[
            SupplierOption(supplier_id="S1", supplier_name="One", region="China", origin_port="Shanghai", capacity=500, base_unit_cost=10.0,
                           landed_unit_cost=10.5, tariff_rate_pct=5.0, lead_time_days=5, reliability=0.9, risk_level="LOW"),
            SupplierOption(supplier_id="S2", supplier_name="Two", region="Vietnam", origin_port="Singapore", capacity=500, base_unit_cost=20.0,
                           landed_unit_cost=20.0, tariff_rate_pct=0.0, lead_time_days=10, reliability=0.9, risk_level="LOW"),
        ],
        routes=[
            RouteOption(route_id="R1", origin="Shanghai", destination="Rotterdam", transport_mode="sea", capacity=500, cost_per_unit=2.0, transit_time_days=5),
            RouteOption(route_id="R2", origin="Singapore", destination="Rotterdam", transport_mode="rail", capacity=500, cost_per_unit=3.0, transit_time_days=10),
        ],
    )
    return OptimizationSolution(
        status="OPTIMAL", problem=problem, objective_value=0.0, objective_terms={"procurement": 0.0, "tariff": 0.0, "freight": 0.0, "transfer": 7.0},
        allocations=[
            Allocation(supplier_id="S1", route_id="R1", transport_mode="sea", quantity=100, landed_unit_cost=10.5, freight_unit_cost=2.0, arrival_days=10),
            Allocation(supplier_id="S2", route_id="R2", transport_mode="rail", quantity=60, landed_unit_cost=20.0, freight_unit_cost=3.0, arrival_days=20),
        ],
        inbound_by_warehouse={"A": 160, "B": 0}, end_stock_by_warehouse={"A": 60, "B": 200},
    )


def test_a_disrupted_route_loses_its_volume_and_the_shortfall_lands_on_the_warehouses_that_needed_it():
    summary, exposure = evaluate_inaction(toy_baseline(), WorldSpec.of(["R1"], [], {}))
    assert (summary.units_delivered, summary.units_short, summary.shipments) == (60, 100, 1)
    assert summary.spend == pytest.approx(60 * (20.0 + 3.0) + 7.0)  # what still arrives (S2, no tariff on Vietnam here) + the untouched transfer term
    assert summary.avg_arrival_days == 20.0 and summary.cost_per_unit == pytest.approx(summary.spend / 60, abs=0.01)
    # 100 of 160 inbound units vanish, all of them A's: A ends at 60 - 100 = -40 (stocked out by 40; 90 short of its safety stock of 50)
    assert (summary.stockout_units, summary.below_safety_units, summary.warehouses_below_safety) == (40, 90, ["A"])
    assert (exposure.shipments_at_risk, exposure.units_at_risk) == (1, 100)
    assert exposure.value_at_risk == pytest.approx(100 * (10.5 + 2.0))  # the baseline plan's own price for what is lost
    assert exposure.shipments[0].reason == "route R1 is disrupted" and exposure.shipments[0].supplier_id == "S1"


def test_a_disrupted_supplier_loses_all_its_shipments_whatever_route_they_use():
    summary, exposure = evaluate_inaction(toy_baseline(), WorldSpec.of([], ["S2"], {}))
    assert (summary.units_delivered, summary.units_short) == (100, 60)
    assert exposure.shipments[0].reason == "supplier S2 is disrupted" and exposure.units_at_risk == 60


def test_a_disruption_that_touches_nothing_leaves_the_plan_as_it_was():
    summary, exposure = evaluate_inaction(toy_baseline(), WorldSpec.of(["SOME-OTHER-ROUTE"], ["S9"], {}))
    assert (summary.units_delivered, summary.units_short, summary.stockout_units, summary.below_safety_units) == (160, 0, 0, 0)
    assert summary.warehouses_below_safety == [] and exposure.shipments_at_risk == 0 and exposure.value_at_risk == 0.0
    assert summary.spend == pytest.approx(100 * (10.5 + 2.0) + 60 * (20.0 + 3.0) + 7.0)  # = the baseline's own spend


def test_a_tariff_change_reprices_what_arrives_and_blocks_nothing():
    """Landed cost = base x (1 + tariff/100), the Sourcing Agent's own rule: S1 (China) 10 x 1.25 = 12.5, S2 (Vietnam) 20 x 1.10 = 22."""
    summary, exposure = evaluate_inaction(toy_baseline(), WorldSpec.of([], [], {"CHN": 25.0, "VNM": 10.0}))
    assert summary.units_short == 0 and exposure.shipments_at_risk == 0
    assert summary.spend == pytest.approx(100 * (12.5 + 2.0) + 60 * (22.0 + 3.0) + 7.0)


def test_when_nothing_arrives_there_is_no_average_arrival_or_unit_cost():
    summary, _ = evaluate_inaction(toy_baseline(), WorldSpec.of(["R1", "R2"], [], {}))
    assert (summary.units_delivered, summary.units_short, summary.shipments) == (0, 160, 0)
    assert summary.avg_arrival_days is None and summary.cost_per_unit is None and summary.mode_split == {}
    assert (summary.stockout_units, summary.below_safety_units) == (100, 150)  # A ends at 60 - 160 = -100: short 100 of demand, 150 of its safety stock


# =========================================================================== #
# the comparison on real data
# =========================================================================== #
@requires_built_data
@pytest.mark.parametrize("scenario_id", MODELED)
def test_the_three_cases_are_consistent_with_each_other(scenario_id):
    c = compare_scenario(get_scenario(scenario_id))
    b, u, m = c.baseline, c.unmitigated, c.mitigated
    assert (b.status, u.status, m.status) == ("OPTIMAL", "EVALUATED", "OPTIMAL")
    assert u.units_delivered + u.units_short == b.units_delivered  # nothing appears from nowhere
    assert c.exposure.units_at_risk == u.units_short and c.exposure.shipments_at_risk == len(c.exposure.shipments)
    assert m.units_delivered == b.units_delivered and m.units_short == 0  # mitigation covers the same demand
    assert m.spend >= b.spend - 0.01  # a disruption can only narrow the optimizer's choices: normal operations is the cheapest plan
    assert c.deltas.mitigation_cost == pytest.approx(m.spend - b.spend, abs=0.01)
    assert c.deltas.units_protected == m.units_delivered - u.units_delivered
    if not c.tariff_changes:  # without a repricing, what survives + what was lost = the baseline's spend, to the cent
        assert u.spend + c.exposure.value_at_risk == pytest.approx(b.spend, abs=0.05)
    assert c.engine == "prototype" and c.engine_label == "Prototype Optimization" and c.model_version and len(c.assumptions) >= 5


@requires_built_data
def test_the_suez_closure_on_the_default_product_pins_what_it_does():
    c = compare_scenario(get_scenario("SUEZ_CLOSURE"))
    assert c.product_id == PRODUCT and c.newly_disrupted_routes == SUEZ and c.newly_disrupted_suppliers == []
    assert c.baseline.spend == pytest.approx(384398.68, abs=0.5) and c.mitigated.spend == pytest.approx(424508.97, abs=0.5)
    assert (c.exposure.shipments_at_risk, c.exposure.units_at_risk) == (1, 252)
    assert c.exposure.shipments[0].route_id == "MUM-ROT-SUEZ" and c.exposure.shipments[0].supplier_id == "S003"
    assert c.unmitigated.warehouses_below_safety == ["Chennai", "Delhi", "Mumbai"] and c.unmitigated.below_safety_units == 252
    assert c.deltas.units_protected == 252 and c.deltas.mitigation_cost_pct == pytest.approx(10.4, abs=0.1)


@requires_built_data
def test_losing_the_main_supplier_is_far_worse_than_losing_a_lane():
    failure, suez = compare_scenario(get_scenario("SUPPLIER_FAILURE")), compare_scenario(get_scenario("SUEZ_CLOSURE"))
    assert failure.exposure.units_at_risk > 10 * suez.exposure.units_at_risk  # S007 supplies 3,759 of the 4,011 units
    assert failure.unmitigated.stockout_units > 0 and suez.unmitigated.stockout_units == 0  # inaction actually stocks out only in the first
    assert failure.deltas.mitigation_cost > suez.deltas.mitigation_cost > 0


@requires_built_data
def test_a_tariff_costs_money_but_leaves_every_shipment_standing():
    c = compare_scenario(get_scenario("TARIFF_INCREASE"))
    assert [(t.iso3, round(t.scenario_pct - t.baseline_pct, 6)) for t in c.tariff_changes] == [("TUR", 25.0)]
    assert c.exposure.shipments_at_risk == 0 and c.unmitigated.units_short == 0
    assert c.unmitigated.spend > c.baseline.spend * 1.1  # S007 supplies most of the plan
    assert c.mitigated.spend <= c.unmitigated.spend + 0.01  # re-optimizing can never do worse than leaving the plan alone


@requires_built_data
def test_a_scenario_that_does_not_touch_the_plan_says_so_by_being_a_no_op():
    c = compare_scenario(get_scenario("SEVERE_WEATHER"), "84077")  # 84077's baseline buys only from S007, direct
    assert c.exposure.shipments_at_risk == 0 and c.deltas.mitigation_cost == 0 and c.deltas.units_protected == 0
    assert c.unmitigated.spend == pytest.approx(c.baseline.spend)


@requires_built_data
def test_a_product_with_no_feasible_baseline_gets_an_honest_partial_answer():
    c = compare_scenario(get_scenario("SUEZ_CLOSURE"), "23166")  # demand far above what its suppliers can supply, even normally
    assert c.baseline.status == "INFEASIBLE" and "short of forecast demand" in c.baseline.message
    assert c.unmitigated.status == "NOT_AVAILABLE" and c.exposure is None
    assert c.deltas.mitigation_cost is None and c.deltas.units_protected is None


@requires_built_data
def test_a_product_nobody_supplies_cannot_be_compared_and_says_so():
    c = compare_scenario(get_scenario("SUEZ_CLOSURE"), "15036")
    assert c.baseline.status == "INFEASIBLE" and c.mitigated.status == "INFEASIBLE"


# =========================================================================== #
# comparing a simulation
# =========================================================================== #
@requires_built_data
def test_a_simulations_comparison_uses_its_own_plan_as_the_mitigated_case():
    outcome = run_scenario_through_pipeline("SUEZ_CLOSURE")
    state = outcome.state
    c = compare_state(state)
    assert c.simulation_id == state.simulation_id and c.scenario_id is None and c.product_id == PRODUCT
    assert c.mitigated.spend == pytest.approx(adapters.plan_spend(state.current_plan))
    assert c.warnings == []
    same = compare_scenario(get_scenario("SUEZ_CLOSURE"))
    assert (c.baseline.spend, c.exposure.units_at_risk, c.deltas.mitigation_cost) == (same.baseline.spend, same.exposure.units_at_risk, same.deltas.mitigation_cost)


@requires_built_data
def test_a_replanned_simulation_is_compared_by_the_plan_it_actually_produced_not_a_fresh_solve():
    """Compliance rejects S007, so the pipeline replans without it: the simulation's plan is dearer than what a fresh
    solve of the same disrupted network gives. The comparison must report the plan that was really produced."""
    outcome = run_scenario_through_pipeline("SUEZ_CLOSURE", rules={**RULES_OPEN, "rejected_suppliers": ["S007"]})
    state = outcome.state
    assert outcome.outcome == "COMPLETED" and state.replan_count == 1 and "S007" not in {a.supplier_id for a in state.current_plan.allocations}
    actual = adapters.plan_spend(state.current_plan)
    fresh = compare_scenario(get_scenario("SUEZ_CLOSURE")).mitigated.spend
    assert actual > fresh + 1000  # the replan really is a different, dearer plan
    assert compare_state(state).mitigated.spend == pytest.approx(actual) and compare_state(state).mitigated.spend != pytest.approx(fresh)


@requires_built_data
def test_a_simulation_run_on_an_earlier_date_is_flagged_because_its_baseline_would_be_on_different_stock():
    _, store, orchestrator = make()
    scenario = get_scenario("SUEZ_CLOSURE")
    state = store.create("SUEZ_CLOSURE")
    out = orchestrator.run(state.simulation_id, scenario.trigger(), PRODUCT, as_of_date="2011-11-30")
    assert out.outcome == "COMPLETED"
    assert any("earlier as-of date" in w for w in compare_state(out.state).warnings)
    assert compare_state(out.state, as_of_date="2011-11-30").warnings == []  # told the date, the baseline is rebuilt on the same stock


@requires_built_data
def test_a_simulation_without_a_plan_has_no_mitigated_case():
    _, store, _ = make()
    state = store.create("EMPTY")
    with pytest.raises(ValueError, match="no plan yet"):
        compare_state(state)
    c = compare_state(state, PRODUCT)
    assert c.mitigated.status == "NOT_AVAILABLE" and "no plan" in c.mitigated.message and c.deltas.units_protected is None


# =========================================================================== #
# the API
# =========================================================================== #
@requires_built_data
def test_the_scenario_list_shows_every_definition_and_why_two_cannot_run():
    client, _, _ = make()
    rows = {r["scenario_id"]: r for r in data(client.get("/api/scenarios"))}
    assert list(rows) == MODELED + UNMODELED
    assert rows["SUEZ_CLOSURE"]["event"]["affected_routes"] == ["SHA-ROT-SUEZ", "SIN-ROT-SUEZ", "MUM-ROT-SUEZ", "CHE-ROT-SUEZ"] and rows["SUEZ_CLOSURE"]["modeled"]
    assert rows["TARIFF_INCREASE"]["tariff_changes"][0]["iso3"] == "TUR"
    assert rows["DEMAND_SURGE"]["modeled"] is False and rows["DEMAND_SURGE"]["event"] is None and "demand-multiplier" in rows["DEMAND_SURGE"]["not_modeled_reason"]


@requires_built_data
def test_the_comparison_endpoint_returns_the_three_cases():
    client, _, _ = make()
    d = data(client.get("/api/scenarios/SUEZ_CLOSURE/comparison"))
    assert d["scenario_id"] == "SUEZ_CLOSURE" and d["product_id"] == PRODUCT and d["exposure"]["units_at_risk"] == 252
    assert [d[k]["case"] for k in ("baseline", "unmitigated", "mitigated")] == ["baseline", "unmitigated", "mitigated"]
    assert d["engine_label"] == "Prototype Optimization" and d["deltas"]["mitigation_cost"] > 0
    assert data(client.get("/api/scenarios/SUPPLIER_FAILURE/comparison?product_id=84077"))["product_id"] == "84077"


@requires_built_data
def test_the_comparison_endpoint_refuses_what_it_cannot_do_with_the_envelope():
    client, _, _ = make()
    error(client.get("/api/scenarios/NOPE/comparison"), 404, "SCENARIO_NOT_FOUND")
    err = error(client.get("/api/scenarios/PORT_CONGESTION/comparison"), 422, "SCENARIO_NOT_MODELED")
    assert "transit-delay" in err["message"] and err["recovery"]
    error(client.get("/api/scenarios/SUEZ_CLOSURE/comparison?product_id=NOPE"), 422, "VALIDATION_ERROR")


@requires_built_data
def test_running_a_scenario_creates_a_simulation_and_produces_the_pipelines_plan_without_an_llm():
    client, store, _ = make()
    accepted = data(client.post("/api/scenarios/SUPPLIER_FAILURE/run", json={}), 202)
    sim = accepted["simulation_id"]
    assert accepted["scenario_id"] == "SUPPLIER_FAILURE" and accepted["product_id"] == PRODUCT and accepted["status_url"].endswith("/status")
    status = data(client.get(f"/api/simulations/{sim}/status"))
    assert status["status"] == "COMPLETED" and status["run"]["outcome"]["outcome"] == "COMPLETED"  # inline run; NeverCalledLLM proves no LLM call
    state = store.get(sim)
    assert state.scenario_type == "SUPPLIER_FAILURE" and state.supplier_status["S007"] == SupplierStatus.DISRUPTED
    comparison = data(client.get(f"/api/simulations/{sim}/comparison"))
    scenario_side = data(client.get("/api/scenarios/SUPPLIER_FAILURE/comparison"))
    assert comparison["simulation_id"] == sim and comparison["mitigated"]["spend"] == pytest.approx(scenario_side["mitigated"]["spend"])
    assert comparison["exposure"]["units_at_risk"] == scenario_side["exposure"]["units_at_risk"]


@requires_built_data
def test_running_a_scenario_applies_its_tariff_and_refuses_an_unmodeled_one():
    client, store, _ = make()
    sim = data(client.post("/api/scenarios/TARIFF_INCREASE/run", json={}), 202)["simulation_id"]
    state = store.get(sim)
    assert state.tariffs["TUR"] == pytest.approx(load_baseline().tariffs["TUR"] + 25.0)
    assert state.disrupted_route_ids() == frozenset() and state.status == SimulationStatus.COMPLETED
    error(client.post("/api/scenarios/DEMAND_SURGE/run", json={}), 422, "SCENARIO_NOT_MODELED")
    error(client.post("/api/scenarios/NOPE/run", json={}), 404, "SCENARIO_NOT_FOUND")
    error(client.post("/api/scenarios/SUEZ_CLOSURE/run", json={"product_id": "NOPE"}), 422, "VALIDATION_ERROR")
    assert len(data(client.get("/api/simulations"))) == 1  # the refusals created nothing


@requires_built_data
def test_a_simulation_with_no_plan_has_no_comparison_yet():
    client, _, _ = make()
    sim = data(client.post("/api/simulations", json={"scenario_type": "X"}), 201)["simulation_id"]
    error(client.get(f"/api/simulations/{sim}/comparison"), 409, "PLAN_NOT_READY")
    error(client.get("/api/simulations/nope/comparison"), 404, "SIMULATION_NOT_FOUND")


@requires_built_data
def test_the_dashboard_now_reports_shipments_at_risk_and_exposure_from_the_latest_finalized_simulation():
    client, _, _ = make()
    d = data(client.get("/api/dashboard"))  # nothing finalized: null, and it says why
    assert d["kpis"]["shipments_at_risk"] is None and "no simulation has been finalized" in d["unavailable"]["estimated_exposure"]

    data(client.post("/api/scenarios/SUEZ_CLOSURE/run", json={}), 202)
    d = data(client.get("/api/dashboard"))
    k = d["kpis"]
    assert (k["shipments_at_risk"], k["units_at_risk"]) == (1, 252) and k["estimated_exposure"] == pytest.approx(46878.07, abs=0.5)
    assert d["unavailable"] == {} and "never observed shipments" in d["exposure_note"]


@requires_built_data
def test_the_dashboard_leaves_the_kpi_null_when_the_simulation_ran_on_an_earlier_date():
    client, store, orchestrator = make()
    scenario = get_scenario("SUEZ_CLOSURE")
    sim = store.create("SUEZ_CLOSURE").simulation_id
    orchestrator.run(sim, scenario.trigger(), PRODUCT, as_of_date="2011-11-30")
    d = data(client.get("/api/dashboard"))
    assert d["kpis"]["estimated_exposure"] is None and "earlier as-of date" in d["unavailable"]["estimated_exposure"]


@requires_built_data
def test_the_product_list_says_which_products_can_be_planned_at_all():
    client, _, _ = make()
    rows = data(client.get("/api/products"))
    assert len(rows) == 40
    assert [r["product_id"] for r in rows if r["has_suppliers"]] == ["22197", "23166", "84077"]  # the synthetic roster covers three products
    assert rows[0]["product_id"] == "22197" and rows[0]["supplier_count"] == 6 and all(r["supplier_count"] == 0 for r in rows[3:])
