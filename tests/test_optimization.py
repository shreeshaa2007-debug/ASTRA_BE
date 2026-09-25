"""Phase 10 tests — Optimization Engine, brief §21 "OPTIMIZATION" category.

Unit tests use small hand-built problems whose optimum is worked out on paper
in each test's comment, so a wrong answer is caught by arithmetic, not by
comparing the engine to itself. The integration tests at the bottom run the
real Inventory/Sourcing/Logistics agents and are skipped if their data isn't built.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.optimization import tools
from backend.optimization.engine import OptimizationEngine, PrototypeOptimizationEngine
from backend.schemas.optimization import (
    AgentRecommendations,
    OptimizationParameters,
    OptimizationProblem,
    RouteOption,
    SupplierOption,
    WarehouseState,
)

ENGINE = PrototypeOptimizationEngine()


# --------------------------------------------------------------------------- #
# builders for small hand-made problems
# --------------------------------------------------------------------------- #
def params(**overrides) -> OptimizationParameters:
    base = dict(planning_horizon_days=45, internal_transfer_cost_per_unit=5.0, internal_transfer_days=2)
    base.update(overrides)
    return OptimizationParameters(**base)


def wh(warehouse_id: str, stock: int, demand: float, safety: int = 0) -> WarehouseState:
    return WarehouseState(warehouse_id=warehouse_id, current_stock=stock, safety_stock=safety, forecast_demand=demand, stockout_risk="LOW")


def sup(supplier_id: str, landed: float, capacity: int, port: str | None = None, lead: float = 5, base: float | None = None) -> SupplierOption:
    return SupplierOption(
        supplier_id=supplier_id, supplier_name=f"Supplier {supplier_id}", region="Testland", origin_port=port, capacity=capacity,
        base_unit_cost=base if base is not None else landed, landed_unit_cost=landed, tariff_rate_pct=0.0,
        lead_time_days=lead, reliability=0.9, risk_level="LOW",
    )


def rt(route_id: str, origin: str, cost: float, capacity: int, days: float, mode: str = "sea") -> RouteOption:
    return RouteOption(route_id=route_id, origin=origin, destination="Hub", transport_mode=mode, capacity=capacity, cost_per_unit=cost, transit_time_days=days)


def problem(warehouses, suppliers, routes=(), parameters=None, **extra) -> OptimizationProblem:
    return OptimizationProblem(product_id="X", parameters=parameters or params(), warehouses=list(warehouses),
                               suppliers=list(suppliers), routes=list(routes), **extra)


def solve(p: OptimizationProblem):
    return ENGINE.optimize_supply_chain(p)


def qty_by_lane(solution) -> dict[tuple[str, str | None], int]:
    return {(a.supplier_id, a.route_id): a.quantity for a in solution.allocations}


def lane_problem(need: int = 1000, **param_overrides) -> OptimizationProblem:
    """One warehouse short by `need`; one supplier at port P with three routes of
    rising capacity and falling speed: R1 (cost 1, cap 200, 2d), R2 (3, 500, 10d), R3 (9, big, 20d)."""
    return problem(
        [wh("W", stock=0, demand=need)],
        [sup("X1", landed=10, capacity=10_000, port="P", lead=5)],
        [rt("R1", "P", 1, 200, 2, "air"), rt("R2", "P", 3, 500, 10, "rail"), rt("R3", "P", 9, 10_000, 20, "sea")],
        params(**param_overrides),
    )


# --------------------------------------------------------------------------- #
# core optimization — answers worked out by hand
# --------------------------------------------------------------------------- #
def test_buys_from_cheapest_supplier_first_and_spills_to_the_next_at_capacity():
    # need 800: A (10/unit, cap 500) fully, then B (12/unit) for 300, C (20) untouched
    # cost = 500*10 + 300*12 = 8600
    p = problem([wh("W", 100, 900)], [sup("A", 10, 500), sup("B", 12, 500), sup("C", 20, 500)])
    s = solve(p)
    assert s.status == "OPTIMAL"
    assert qty_by_lane(s) == {("A", None): 500, ("B", None): 300}
    assert s.objective_value == pytest.approx(8600)


def test_route_capacity_forces_a_multimodal_split():
    # need 1000, unit cost 10 + freight: fill R1 (11/unit) to 200, R2 (13/unit) to 500, rest on R3 (19/unit)
    # cost = 200*11 + 500*13 + 300*19 = 14400
    s = solve(lane_problem(1000))
    assert qty_by_lane(s) == {("X1", "R1"): 200, ("X1", "R2"): 500, ("X1", "R3"): 300}
    assert s.objective_value == pytest.approx(14400)
    assert s.objective_terms["procurement"] == pytest.approx(10_000)
    assert s.objective_terms["freight"] == pytest.approx(200 * 1 + 500 * 3 + 300 * 9)


def test_two_suppliers_share_one_routes_capacity():
    # S1 (10/unit) and S2 (12/unit) both ship on R (cost 1, cap 600); D is direct at 40/unit. Need 900.
    # R can carry only 600 between them: S1 500 + S2 100, and D covers the last 300.
    # cost = 500*11 + 100*13 + 300*40 = 18800. (Per-lane bounds alone would let S1+S2 push 900 through R.)
    p = problem(
        [wh("W", 0, 900)],
        [sup("S1", 10, 500, port="P"), sup("S2", 12, 500, port="P"), sup("D", 40, 5000)],
        [rt("R", "P", 1, 600, 5)],
    )
    s = solve(p)
    assert s.status == "OPTIMAL"
    assert qty_by_lane(s) == {("S1", "R"): 500, ("S2", "R"): 100, ("D", None): 300}
    assert s.objective_value == pytest.approx(18800)


def test_one_suppliers_capacity_is_shared_across_all_its_lanes():
    # supplier cap 500 but two routes of 300 each: 500 is fine (R1 300 + R2 200), 600 is not,
    # even though 300 + 300 fits the routes
    lanes = [rt("R1", "P", 1, 300, 5), rt("R2", "P", 2, 300, 5)]
    fits = solve(problem([wh("W", 0, 500)], [sup("S", 10, 500, port="P")], lanes))
    assert qty_by_lane(fits) == {("S", "R1"): 300, ("S", "R2"): 200}

    over = solve(problem([wh("W", 0, 600)], [sup("S", 10, 500, port="P")], lanes))
    assert over.status == "INFEASIBLE"
    assert over.diagnostics["total_shortfall_units"] == 100
    assert "supplier S" in over.diagnostics["saturated_resources"]


def test_tariff_is_reported_separately_from_base_procurement():
    # landed 11 = base 10 + 1 tariff, need 100 -> procurement 1000, tariff 100
    s = solve(problem([wh("W", 0, 100)], [sup("A", landed=11, capacity=500, base=10)]))
    assert s.objective_terms["procurement"] == pytest.approx(1000)
    assert s.objective_terms["tariff"] == pytest.approx(100)


def test_hard_deadline_drops_lanes_that_arrive_too_late():
    # deadline day 15: R3 arrives day 25 and is not offered. R2 arrives on day 15 exactly (inclusive).
    # need 600 -> R1 200 + R2 400: cost = 200*11 + 400*13 = 7400
    s = solve(lane_problem(600, max_delivery_days=15))
    assert qty_by_lane(s) == {("X1", "R1"): 200, ("X1", "R2"): 400}
    assert s.objective_value == pytest.approx(7400)
    assert any(e.kind == "lane" and e.id == "X1:R3" and "deadline" in e.reason for e in s.excluded_options)


def test_infeasible_problem_is_reported_with_a_diagnosis_never_as_a_plan():
    # same deadline, need 1000 but only R1+R2 (700 units) can arrive in time -> 300 short
    s = solve(lane_problem(1000, max_delivery_days=15))
    assert s.status == "INFEASIBLE"
    assert s.objective_value is None and not s.allocations and not s.transfers
    assert s.diagnostics["total_shortfall_units"] == 300
    assert s.diagnostics["shortfall_by_warehouse"] == {"W": pytest.approx(300)}
    assert {"route R1", "route R2"} <= set(s.diagnostics["saturated_resources"])
    assert "deadline" in s.diagnostics["recovery"].lower()
    assert ENGINE.validate_solution(s).valid is False


def test_soft_delay_penalty_flips_the_choice_to_the_faster_supplier():
    # A: 10/unit but 30 days lead; B: 14/unit, 5 days. Target day 10, 0.5 per unit-day late.
    # Without the penalty A wins (10 < 14). With it A costs 10 + 0.5*20 = 20 > B's 14, so B wins.
    suppliers = [sup("A", 10, 1000, lead=30), sup("B", 14, 1000, lead=5)]
    off = solve(problem([wh("W", 0, 100)], suppliers))
    on = solve(problem([wh("W", 0, 100)], suppliers, parameters=params(target_delivery_days=10, delay_penalty_per_unit_day=0.5)))
    assert qty_by_lane(off) == {("A", None): 100}
    assert qty_by_lane(on) == {("B", None): 100}
    assert on.objective_terms["delay_penalty"] == 0
    assert on.objective_value == pytest.approx(1400)


def test_delay_penalty_is_charged_and_reported_when_lateness_is_unavoidable():
    # only supplier arrives day 30, target 10, penalty 0.5 -> 10 extra per unit on 100 units
    s = solve(problem([wh("W", 0, 100)], [sup("A", 10, 1000, lead=30)], parameters=params(target_delivery_days=10, delay_penalty_per_unit_day=0.5)))
    assert s.objective_terms["delay_penalty"] == pytest.approx(1000)
    assert s.objective_value == pytest.approx(2000)


# --------------------------------------------------------------------------- #
# inventory: transfers and safety stock
# --------------------------------------------------------------------------- #
def test_transfer_is_used_instead_of_buying_when_another_warehouse_has_real_surplus():
    # Mum is 200 short; Del holds 1000 against demand 200 + safety 100 (700 spare). Transfer 200 at 5/unit = 1000
    # beats buying at 10/unit.
    s = solve(problem([wh("Mum", 100, 300), wh("Del", 1000, 200, safety=100)], [sup("A", 10, 5000)]))
    assert not s.allocations
    assert [(t.from_warehouse, t.to_warehouse, t.quantity) for t in s.transfers] == [("Del", "Mum", 200)]
    assert s.objective_value == pytest.approx(1000)


def test_transfer_is_capped_by_the_sources_own_forecast_demand_and_the_rest_is_bought():
    # Del has 400 in stock but needs 200 of it plus 100 safety, so it can spare only 100.
    # Mum's 200 shortfall = 100 transferred (5/unit) + 100 bought (10/unit) = 500 + 1000
    s = solve(problem([wh("Mum", 100, 300), wh("Del", 400, 200, safety=100)], [sup("A", 10, 5000)]))
    assert [(t.from_warehouse, t.to_warehouse, t.quantity) for t in s.transfers] == [("Del", "Mum", 100)]
    assert qty_by_lane(s) == {("A", None): 100}
    assert s.objective_value == pytest.approx(1500)
    assert s.end_stock_by_warehouse == {"Mum": 0, "Del": 100}


def test_safety_stock_is_a_hard_floor():
    # demand 100, stock 100, safety 50 -> must end at >= 50, so buy 50
    s = solve(problem([wh("W", 100, 100, safety=50)], [sup("A", 10, 500)]))
    assert qty_by_lane(s) == {("A", None): 50}
    assert s.end_stock_by_warehouse == {"W": 50}


def test_fractional_forecast_demand_is_covered_by_rounding_up_to_whole_units():
    s = solve(problem([wh("W", 0, 99.2)], [sup("A", 10, 500)]))
    assert qty_by_lane(s) == {("A", None): 100}


def test_no_action_when_stock_already_covers_demand_plus_safety_stock():
    s = solve(problem([wh("W", 1000, 100, safety=50)], [sup("A", 10, 500)]))
    assert s.status == "OPTIMAL"
    assert not s.allocations and not s.transfers and s.objective_value == 0
    assert "no action needed" in s.message


# --------------------------------------------------------------------------- #
# disruptions and unusable suppliers
# --------------------------------------------------------------------------- #
def test_supplier_whose_only_route_is_gone_is_excluded_with_a_reason():
    # S ships from Mumbai but no Mumbai route is available; D is direct and dearer
    p = problem([wh("W", 0, 100)], [sup("S", 10, 5000, port="Mumbai"), sup("D", 50, 5000)],
                [rt("SH1", "Shanghai", 1, 5000, 10)], disrupted_route_ids=["MUM-1"])
    s = solve(p)
    assert qty_by_lane(s) == {("D", None): 100}
    stranded = next(e for e in s.excluded_options if e.kind == "supplier" and e.id == "S")
    assert "no available route from Mumbai" in stranded.reason
    assert any(f["factor"] == "Suppliers with no usable lane" for f in s.decision_factors)
    assert any(f["factor"] == "Disrupted routes excluded" for f in s.decision_factors)


def test_supplier_with_no_capacity_is_excluded():
    s = solve(problem([wh("W", 0, 100)], [sup("Dead", 1, 0), sup("Live", 10, 500)]))
    assert qty_by_lane(s) == {("Live", None): 100}
    assert any(e.id == "Dead" and e.reason == "no capacity" for e in s.excluded_options)


def test_a_route_is_only_usable_by_suppliers_at_its_origin_port():
    # cheap route starts at Q but the only supplier ships from P -> must use the dearer P route
    p = problem([wh("W", 0, 100)], [sup("A", 10, 5000, port="P")], [rt("FromQ", "Q", 0.1, 5000, 5), rt("FromP", "P", 7, 5000, 5)])
    assert qty_by_lane(solve(p)) == {("A", "FromP"): 100}


# --------------------------------------------------------------------------- #
# solution integrity
# --------------------------------------------------------------------------- #
def test_objective_terms_sum_to_objective_and_quantities_are_whole_units():
    s = solve(lane_problem(1000))
    assert sum(s.objective_terms.values()) == pytest.approx(s.objective_value)
    assert all(isinstance(a.quantity, int) and a.quantity > 0 for a in s.allocations)
    assert s.solver["integer_quantities"] is True and s.solver["scipy_status"] == 0


def test_engine_labels_itself_as_the_prototype():
    s = solve(lane_problem(100))
    assert (s.engine, s.label) == ("prototype", "Prototype Optimization")


def test_every_optimal_plan_passes_independent_validation():
    for p in (lane_problem(1000), lane_problem(600, max_delivery_days=15),
              problem([wh("Mum", 100, 300), wh("Del", 400, 200, safety=100)], [sup("A", 10, 5000)])):
        s = solve(p)
        result = ENGINE.validate_solution(s)
        assert result.valid, result.violations


def test_binding_constraints_are_the_capacities_that_shaped_the_plan():
    s = solve(lane_problem(1000))
    binding = {c.name for c in s.constraint_status if c.binding}
    assert {"route_capacity:R1", "route_capacity:R2", "safety_stock:W"} <= binding
    assert "route_capacity:R3" not in binding


# --------------------------------------------------------------------------- #
# validate_solution catches a plan that breaks the rules
# --------------------------------------------------------------------------- #
def tampered(mutate):
    s = solve(lane_problem(1000)).model_copy(deep=True)
    mutate(s)
    return ENGINE.validate_solution(s)


def test_validation_catches_route_capacity_overrun():
    result = tampered(lambda s: setattr(next(a for a in s.allocations if a.route_id == "R1"), "quantity", 5000))
    assert not result.valid
    assert any("route_capacity:R1" in v for v in result.violations)


def test_validation_catches_a_route_that_does_not_exist():
    result = tampered(lambda s: setattr(s.allocations[0], "route_id", "NOPE"))
    assert not result.valid
    assert any("lane_validity" in v for v in result.violations)


def test_validation_catches_a_misreported_objective():
    result = tampered(lambda s: setattr(s, "objective_value", s.objective_value + 1000))
    assert not result.valid
    assert any("objective_value" in v for v in result.violations)


def test_validation_catches_broken_safety_stock():
    def cut_delivery(s):
        s.allocations[0].quantity -= 50
        s.inbound_by_warehouse["W"] -= 50
        s.end_stock_by_warehouse["W"] -= 50

    result = tampered(cut_delivery)
    assert not result.valid
    assert any("safety_stock:W" in v for v in result.violations)


def test_validation_catches_shipping_out_more_stock_than_a_warehouse_holds():
    s = solve(problem([wh("Mum", 100, 300), wh("Del", 1000, 200, safety=100)], [sup("A", 10, 5000)])).model_copy(deep=True)
    s.transfers[0].quantity = 5000  # Del holds 1000
    result = ENGINE.validate_solution(s)
    assert not result.valid
    assert any("transfer_availability:Del" in v for v in result.violations)


def test_validation_catches_a_deadline_breach():
    def slow_down(s):
        s.problem.parameters.max_delivery_days = 10.0

    result = tampered(slow_down)
    assert any("delivery_deadline" in v for v in result.violations)


def test_get_objective_value_refuses_a_solution_that_is_not_a_plan():
    infeasible = solve(lane_problem(1000, max_delivery_days=15))
    with pytest.raises(ValueError, match="INFEASIBLE"):
        ENGINE.get_objective_value(infeasible)


# --------------------------------------------------------------------------- #
# explanation: deviations from what each agent recommended
# --------------------------------------------------------------------------- #
def test_deviation_names_the_source_warehouses_own_forecast_when_the_agents_transfer_was_too_big():
    p = problem([wh("Mum", 100, 300), wh("Del", 400, 200, safety=100)], [sup("A", 10, 5000)],
                agent_recommendations=AgentRecommendations(inventory_transfers=[{"from": "Del", "to": "Mum", "quantity": 250}]))
    s = solve(p)
    dev = next(d for d in s.deviations if d["agent"] == "inventory")
    assert dev["recommended"] == "250 units Del -> Mum" and dev["planned"] == "100 units"
    assert "can spare only 100" in dev["reason"]


def test_a_transfer_the_agent_did_not_recommend_is_reported_with_the_shortfall_that_justifies_it():
    s = solve(problem([wh("Mum", 100, 300), wh("Del", 1000, 200, safety=100)], [sup("A", 10, 5000)]))
    dev = next(d for d in s.deviations if d["agent"] == "inventory")
    assert dev["recommended"] == "no transfer" and dev["planned"] == "200 units Del -> Mum"
    assert "short 200" in dev["reason"]


def test_sourcing_deviation_explains_a_stranded_supplier_and_the_one_that_replaced_it():
    p = problem(
        [wh("W", 0, 100)], [sup("S", 10, 5000, port="Mumbai"), sup("D", 50, 5000)], [rt("SH1", "Shanghai", 1, 5000, 10)],
        agent_recommendations=AgentRecommendations(sourcing_allocations=[{"supplier_id": "S", "quantity": 100}]),
    )
    devs = {d["recommended"]: d for d in solve(p).deviations if d["agent"] == "sourcing"}
    assert "unusable in the joint plan" in devs["100 units from S"]["reason"]
    assert devs["0 units from D"]["planned"] == "100 units"


def test_no_deviation_is_reported_when_the_plan_matches_the_agents():
    p = problem([wh("W", 100, 900)], [sup("A", 10, 500), sup("B", 12, 500)],
                agent_recommendations=AgentRecommendations(sourcing_allocations=[{"supplier_id": "A", "quantity": 500}, {"supplier_id": "B", "quantity": 300}]))
    assert solve(p).deviations == []


# --------------------------------------------------------------------------- #
# schemas, protocol, metrics
# --------------------------------------------------------------------------- #
def test_prototype_engine_satisfies_the_engine_protocol():
    assert isinstance(ENGINE, OptimizationEngine)


def test_parameters_default_the_hard_deadline_to_the_planning_horizon():
    assert params().max_delivery_days == 45.0


def test_parameters_reject_a_delay_penalty_without_a_target_and_a_target_past_the_deadline():
    with pytest.raises(ValidationError, match="requires target_delivery_days"):
        params(delay_penalty_per_unit_day=1.0)
    with pytest.raises(ValidationError, match="cannot exceed"):
        params(target_delivery_days=50)


def test_problem_rejects_duplicate_ids():
    with pytest.raises(ValidationError, match="duplicate supplier_id"):
        problem([wh("W", 0, 10)], [sup("A", 10, 5), sup("A", 11, 5)])


def test_metrics_summarize_the_plan():
    m = tools.get_optimization_metrics(solve(lane_problem(1000)))
    assert m["units_procured"] == 1000 and m["units_transferred"] == 0
    assert m["mode_split"]["air"] == {"units": 200, "share": 0.2}
    assert sum(v["share"] for v in m["mode_split"].values()) == pytest.approx(1)
    assert m["avg_all_in_unit_cost"] == pytest.approx(14.4)
    assert m["constraints_satisfied"] == m["constraints_total"]
    assert "route_capacity:R1" in m["binding_constraints"]


def test_metrics_for_an_infeasible_solution_carry_the_diagnosis_not_numbers():
    m = tools.get_optimization_metrics(solve(lane_problem(1000, max_delivery_days=15)))
    assert m["status"] == "INFEASIBLE" and "objective_value" not in m
    assert m["diagnostics"]["total_shortfall_units"] == 300


# --------------------------------------------------------------------------- #
# build_optimization_problem — from the agents' own output shapes
# --------------------------------------------------------------------------- #
CONFIG = {"network": {"inbound_hub": "Hub", "supplier_origin_port": {"S1": "P", "S2": None}}, "defaults": {}}


def candidate(supplier_id: str, landed: float = 10.0, capacity: int = 1000) -> dict:
    return {"supplier_id": supplier_id, "supplier_name": supplier_id, "region": "Testland", "status": "ACTIVE", "risk_level": "LOW",
            "reliability": 0.9, "capacity": capacity, "capacity_sufficient": True, "lead_time_days": 5, "unit_cost": landed - 1,
            "tariff_rate_pct": 10.0, "landed_unit_cost": landed, "landed_cost_for_requested_quantity": landed * 100}


def agent_outputs(**overrides) -> dict:
    out = dict(
        product_id="X",
        inventory_results=[{"warehouse": "W", "product": "X", "forecast_demand": 300.0, "current_stock": 100, "stockout_risk": "HIGH", "recommended_transfer": None}],
        sourcing_result={"product_id": "X", "required_quantity": 200, "supplier_allocations": [{"supplier_id": "S1", "quantity": 200}],
                         "candidates_considered": [candidate("S1"), candidate("S2", landed=30)]},
        logistics_results=[{"origin": "P", "destination": "Hub", "disrupted_route_ids": ["P-OLD"], "recommended_route_id": "P-1",
                            "alternative_routes": [{"route_id": "P-1", "transport_mode": "sea", "capacity": 1000, "cost_per_unit": 2.0, "transit_time_days": 10.0}]}],
        safety_stock_by_warehouse={"W": 0},
    )
    out.update(overrides)
    return out


def build(**overrides) -> OptimizationProblem:
    return tools.build_optimization_problem(parameters=params(), config=CONFIG, **agent_outputs(**overrides))


def test_builder_assembles_the_problem_from_agent_output_shapes():
    p = build()
    assert [w.warehouse_id for w in p.warehouses] == ["W"]
    assert {s.supplier_id: s.origin_port for s in p.suppliers} == {"S1": "P", "S2": None}
    assert p.routes[0].origin == "P" and p.routes[0].destination == "Hub"
    assert p.disrupted_route_ids == ["P-OLD"]
    assert p.agent_recommendations.logistics_route_ids == ["P-1"]
    assert any("S2" in a and "flatters" in a for a in p.assumptions)  # the freight-free supplier is called out


def test_builder_output_solves_end_to_end():
    s = solve(build())
    # need 200: S1 lane costs 10 + 2 = 12/unit, S2 is 30/unit direct -> all 200 from S1 on P-1
    assert qty_by_lane(s) == {("S1", "P-1"): 200}
    assert s.objective_value == pytest.approx(2400)


def test_builder_excludes_a_supplier_missing_from_the_origin_port_map_rather_than_assuming_it_direct():
    sourcing = agent_outputs()["sourcing_result"]
    sourcing["candidates_considered"].append(candidate("S9"))
    p = build(sourcing_result=sourcing)
    assert "S9" not in {s.supplier_id for s in p.suppliers}
    assert any(e.id == "S9" and "origin-port mapping" in e.reason for e in p.excluded_options)


def test_builder_excludes_routes_that_do_not_end_at_the_inbound_hub():
    logistics = agent_outputs()["logistics_results"]
    logistics[0]["destination"] = "Elsewhere"
    p = build(logistics_results=logistics)
    assert p.routes == []
    assert any(e.kind == "route" and "inbound hub" in e.reason for e in p.excluded_options)


def test_builder_maps_the_inventory_agents_recommended_transfer_onto_the_receiving_warehouse():
    inventory = agent_outputs()["inventory_results"]
    inventory[0]["recommended_transfer"] = {"from": "Other", "quantity": 40}
    p = build(inventory_results=inventory)
    assert p.agent_recommendations.inventory_transfers == [{"from": "Other", "to": "W", "quantity": 40}]


@pytest.mark.parametrize("override, message", [
    (dict(product_id="Y"), "products other than"),
    (dict(safety_stock_by_warehouse={}), "no safety stock"),
    (dict(inventory_results=[]), "at least one warehouse"),
])
def test_builder_rejects_inconsistent_agent_inputs(override, message):
    with pytest.raises(ValueError, match=message):
        build(**override)


def test_builder_rejects_a_sourcing_result_for_a_different_product():
    sourcing = agent_outputs()["sourcing_result"]
    sourcing["product_id"] = "Y"
    with pytest.raises(ValueError, match="not 'X'"):
        build(sourcing_result=sourcing)


# --------------------------------------------------------------------------- #
# integration — real Inventory, Sourcing and Logistics agents
# --------------------------------------------------------------------------- #
requires_built_data = pytest.mark.skipif(
    not all(Path(p).exists() for p in (
        "data/processed/suppliers.csv", "data/processed/routes.csv", "data/processed/inventory_multi_warehouse.csv",
        "data/processed/demand_modeling_panel.csv", "ml/artifacts/xgboost_demand/2026.09.1/model.json",
    )),
    reason="run the Phase 3-5 pipelines first",
)

SUEZ = frozenset({"SHA-ROT-SUEZ", "SIN-ROT-SUEZ", "MUM-ROT-SUEZ", "CHE-ROT-SUEZ"})
AS_OF = "2011-11-30"


@pytest.fixture(scope="module")
def healthy():
    return solve(tools.build_problem_from_agents("22197", AS_OF))


@pytest.fixture(scope="module")
def suez_closed():
    return solve(tools.build_problem_from_agents("22197", AS_OF, disrupted_route_ids=SUEZ))


@requires_built_data
def test_real_agents_healthy_network_yields_a_valid_optimal_plan(healthy):
    assert healthy.status == "OPTIMAL"
    assert ENGINE.validate_solution(healthy).valid
    assert all(a.arrival_days <= 45 for a in healthy.allocations)
    assert set(healthy.inbound_by_warehouse) == {"Mumbai", "Chennai", "Delhi"}
    assert all(healthy.end_stock_by_warehouse[w.warehouse_id] >= w.safety_stock for w in healthy.problem.warehouses)


@requires_built_data
def test_suez_closure_never_routes_through_a_closed_lane_and_strands_the_indian_suppliers(suez_closed):
    assert suez_closed.status == "OPTIMAL"
    assert ENGINE.validate_solution(suez_closed).valid
    assert not any(a.route_id in SUEZ for a in suez_closed.allocations)
    stranded = {e.id for e in suez_closed.excluded_options if e.kind == "supplier" and "no available route" in e.reason}
    assert {"S003", "S004"} <= stranded  # India's only lanes (Mumbai/Chennai -> Rotterdam) run through Suez
    assert not any(a.supplier_id in {"S003", "S004"} for a in suez_closed.allocations)


@requires_built_data
def test_suez_closure_costs_more_than_the_healthy_network(healthy, suez_closed):
    # removing options can never make the optimum cheaper; here it strictly costs more (Cape freight replaces Suez)
    assert suez_closed.objective_value > healthy.objective_value


@requires_built_data
def test_real_plan_reports_where_it_departs_from_the_sourcing_agent(suez_closed):
    sourcing_devs = [d for d in suez_closed.deviations if d["agent"] == "sourcing"]
    assert any("unusable in the joint plan" in d["reason"] for d in sourcing_devs)


@requires_built_data
def test_real_plan_supplier_allocations_pass_the_compliance_agents_supplier_policy(suez_closed):
    from backend.agents.compliance import tools as compliance_tools

    allocations = [{"supplier_id": a.supplier_id, "product_id": "22197", "quantity": a.quantity} for a in suez_closed.allocations]
    assert compliance_tools.check_supplier_policy(allocations)["passed"]
    for route_id in {a.route_id for a in suez_closed.allocations if a.route_id}:
        assert compliance_tools.check_route_policy(route_id, SUEZ)["passed"]


@requires_built_data
def test_real_demand_beyond_total_supplier_capacity_is_infeasible_not_a_fake_plan():
    # product 23166 needs ~21k units over 45 days; its available suppliers hold ~16k
    s = solve(tools.build_problem_from_agents("23166", AS_OF))
    assert s.status == "INFEASIBLE"
    assert s.diagnostics["total_shortfall_units"] > 0
    assert not s.allocations
    assert "capacity" in s.diagnostics["recovery"].lower()


@requires_built_data
def test_a_tighter_real_deadline_drops_slow_lanes_and_cannot_lower_cost(suez_closed):
    # 30 days still admits Vietnam via Singapore-Cape (day 28.1) but not the 40-day China Cape lane
    tight = solve(tools.build_problem_from_agents("22197", AS_OF, disrupted_route_ids=SUEZ, parameter_overrides={"max_delivery_days": 30}))
    assert tight.status == "OPTIMAL"
    assert ENGINE.validate_solution(tight).valid
    assert all(a.arrival_days <= 30 for a in tight.allocations)
    assert any(e.kind == "lane" and "deadline" in e.reason for e in tight.excluded_options)
    assert tight.objective_value >= suez_closed.objective_value


@requires_built_data
def test_a_real_deadline_that_leaves_too_little_supply_is_infeasible_and_says_which_lanes_missed_it():
    # 20 days with Suez closed: only Turkey (direct) and Shanghai air (<=200 units) arrive in time
    s = solve(tools.build_problem_from_agents("22197", AS_OF, disrupted_route_ids=SUEZ, parameter_overrides={"max_delivery_days": 20}))
    assert s.status == "INFEASIBLE"
    assert "S006:SIN-ROT-CAPE" in s.diagnostics["deadline_blocked_lanes"]
    assert "deadline" in s.diagnostics["recovery"].lower()


# --------------------------------------------------------------------------- #
# HiGHS and threads (found in Phase 15, when the API began running solves on worker threads)
# --------------------------------------------------------------------------- #
def test_solving_from_short_lived_threads_neither_hangs_nor_trips_the_native_fault_handler():
    """scipy's bundled HiGHS, on Windows, misbehaves when a solve runs on a thread that then exits: access-violation
    reports, and a plain LP that froze the whole process (the main thread stuck in Thread.start()). The API solves on
    worker threads, so the engine confines every solve to one permanent thread. This is the exact failure signature —
    thread churn around solves — run in a subprocess so a hang is a timeout, not a hung test session."""
    import os
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    script = textwrap.dedent("""
        import faulthandler, threading, warnings
        warnings.filterwarnings("ignore")
        faulthandler.enable()
        faulthandler.dump_traceback_later(60, exit=True)          # a hang becomes a stack dump and a non-zero exit
        from backend.optimization.engine import PrototypeOptimizationEngine
        from backend.schemas.optimization import (OptimizationParameters, OptimizationProblem, RouteOption, SupplierOption, WarehouseState)

        def problem(need):
            return OptimizationProblem(
                product_id="X",
                parameters=OptimizationParameters(planning_horizon_days=45, internal_transfer_cost_per_unit=5.0, internal_transfer_days=2),
                warehouses=[WarehouseState(warehouse_id="W", current_stock=0, safety_stock=0, forecast_demand=need, stockout_risk="HIGH")],
                suppliers=[SupplierOption(supplier_id="S", supplier_name="S", region="R", origin_port="P", capacity=10000, base_unit_cost=10,
                                          landed_unit_cost=10, tariff_rate_pct=0, lead_time_days=5, reliability=0.9, risk_level="LOW")],
                routes=[RouteOption(route_id="R1", origin="P", destination="H", transport_mode="air", capacity=200, cost_per_unit=1, transit_time_days=2),
                        RouteOption(route_id="R2", origin="P", destination="H", transport_mode="sea", capacity=10000, cost_per_unit=9, transit_time_days=20)])

        engine, results = PrototypeOptimizationEngine(), []
        def work():
            results.append(engine.optimize_supply_chain(problem(1000)).objective_value)          # a MILP
            results.append(engine.optimize_supply_chain(problem(50000)).status)                  # infeasible: the relaxed LP as well

        for _ in range(40):                                        # one short-lived thread after another
            t = threading.Thread(target=work); t.start(); t.join()
        burst = [threading.Thread(target=work) for _ in range(20)]  # then a concurrent burst
        [t.start() for t in burst]; [t.join() for t in burst]
        assert len(results) == 120 and {r for r in results if not isinstance(r, str)} == {17400.0} and {r for r in results if isinstance(r, str)} == {"INFEASIBLE"}
        print("finished")
    """)
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", script], capture_output=True, text=True, timeout=120,
                         env={**os.environ, "PYTHONPATH": str(Path.cwd())})
    assert out.returncode == 0 and "finished" in out.stdout, (out.stdout, out.stderr[-1500:])
    assert "access violation" not in out.stderr.lower() and "fatal exception" not in out.stderr.lower(), out.stderr[-1500:]


def test_the_solver_thread_hands_exceptions_back_to_the_caller_and_survives_them():
    from backend.optimization import engine as engine_module

    def boom():
        raise ArithmeticError("solver blew up")

    with pytest.raises(ArithmeticError, match="solver blew up"):
        engine_module._SOLVER.call(boom)
    assert engine_module._SOLVER.call(lambda: 7 * 6) == 42  # the thread is still serving after a failed job


def test_all_solves_run_on_the_one_solver_thread_including_when_called_from_it():
    import threading

    from backend.optimization import engine as engine_module

    names = {engine_module._SOLVER.call(lambda: threading.current_thread().name) for _ in range(5)}
    assert names == {"highs-solver"} and engine_module._SOLVER._thread.daemon is True  # daemon: never joined at exit
    assert engine_module._SOLVER.call(lambda: engine_module._SOLVER.call(lambda: "nested")) == "nested"  # re-entrant, no self-deadlock
