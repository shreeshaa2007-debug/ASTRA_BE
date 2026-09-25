"""Phase 7 tests — Logistics Agent tools + agent, brief §21 "AGENTS" category."""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.agents.logistics import tools
from backend.agents.logistics.agent import LogisticsAgent

requires_built_data = pytest.mark.skipif(not Path("data/processed/routes.csv").exists(), reason="run Phase 3's pipeline first")


# --------------------------------------------------------------------------- #
# tools.py
# --------------------------------------------------------------------------- #
@requires_built_data
def test_get_routes_filters_by_lane():
    routes = tools.get_routes(origin="Shanghai", destination="Rotterdam")
    assert len(routes) == 4  # suez, cape, rail, air
    assert {r["transport_mode"] for r in routes} == {"sea", "rail", "air"}


@requires_built_data
def test_get_routes_disrupted_override_does_not_mutate_source_data():
    tools.get_routes(origin="Shanghai", destination="Rotterdam", disrupted_route_ids=frozenset({"SHA-ROT-SUEZ"}))
    fresh = tools.get_routes(origin="Shanghai", destination="Rotterdam")
    normal = next(r for r in fresh if r["route_id"] == "SHA-ROT-SUEZ")
    assert normal["status"] == "NORMAL"  # unaffected by the previous call's override


@requires_built_data
def test_check_route_capacity_sufficient_and_insufficient():
    ok = tools.check_route_capacity("SHA-ROT-SUEZ", 5000)
    assert ok["sufficient"] is True
    assert ok["shortfall"] == 0

    too_much = tools.check_route_capacity("SHA-ROT-AIR", 5000)  # air capacity is 200
    assert too_much["sufficient"] is False
    assert too_much["shortfall"] == 4800


@requires_built_data
def test_check_route_capacity_unknown_route_raises():
    with pytest.raises(ValueError):
        tools.check_route_capacity("NOT-A-ROUTE", 100)


@requires_built_data
def test_calculate_transport_cost_matches_cost_per_unit_times_quantity():
    routes = tools.get_routes(origin="Shanghai", destination="Rotterdam", transport_mode="sea")
    suez = next(r for r in routes if r["route_id"] == "SHA-ROT-SUEZ")
    cost = tools.calculate_transport_cost("SHA-ROT-SUEZ", 100)
    assert cost == pytest.approx(suez["cost_per_unit"] * 100, abs=0.01)


@requires_built_data
def test_calculate_eta_adds_transit_time():
    eta = tools.calculate_eta("SHA-ROT-SUEZ", "2024-01-01")
    assert eta == "2024-01-15"  # 2024-01-01 + 14.3 days -> Jan 15 (floors the partial day)


@requires_built_data
def test_generate_alternative_routes_excludes_disrupted_and_sorts_by_cost():
    alts = tools.generate_alternative_routes(
        "Shanghai", "Rotterdam", quantity=1000, disrupted_route_ids=frozenset({"SHA-ROT-SUEZ"})
    )
    assert all(a["route_id"] != "SHA-ROT-SUEZ" for a in alts)
    costs = [a["total_cost"] for a in alts]
    assert costs == sorted(costs)


@requires_built_data
def test_rail_and_air_costs_are_documented_multiples_of_the_real_sea_baseline():
    """Grounds the rail/air cost figures back to the cited real-world ratios
    (ports_routes.py: rail ~2x sea, air ~6x sea) rather than letting them
    silently drift to arbitrary numbers on a future edit."""
    routes = {r["route_id"]: r for r in tools.get_routes(origin="Shanghai", destination="Rotterdam")}
    sea_cost = routes["SHA-ROT-SUEZ"]["cost_per_unit"]
    assert routes["SHA-ROT-RAIL"]["cost_per_unit"] == pytest.approx(sea_cost * 2.0, rel=0.01)
    assert routes["SHA-ROT-AIR"]["cost_per_unit"] == pytest.approx(sea_cost * 6.0, rel=0.01)


# --------------------------------------------------------------------------- #
# agent.py
# --------------------------------------------------------------------------- #
@requires_built_data
def test_plan_shipment_shows_true_baseline_even_when_that_route_is_disrupted():
    """The exact bug caught during development: looking up 'the NORMAL
    route' AFTER applying the disruption override finds nothing (the
    disrupted route no longer has status NORMAL), silently dropping
    additional_cost/additional_delay_days from every alternative."""
    agent = LogisticsAgent()
    plan = agent.plan_shipment("Shanghai", "Rotterdam", quantity=1000, disrupted_route_ids=frozenset({"SHA-ROT-SUEZ"}))

    assert plan["original_route"] is not None
    assert plan["original_route"]["route_id"] == "SHA-ROT-SUEZ"
    assert plan["original_route"]["status"] == "DISRUPTED"
    for alt in plan["alternative_routes"]:
        assert "additional_cost" in alt
        assert "additional_delay_days" in alt


@requires_built_data
def test_plan_shipment_recommendation_respects_capacity_not_just_price():
    """The second bug caught during development: recommending the cheapest
    route regardless of whether it can actually carry the shipment. Air is
    fastest and would sort first on some lanes but only has 200 units of
    capacity — a 1500-unit shipment must not be routed onto it."""
    agent = LogisticsAgent()
    plan = agent.plan_shipment("Shanghai", "Rotterdam", quantity=1500, disrupted_route_ids=frozenset({"SHA-ROT-SUEZ"}))

    recommended = next(a for a in plan["alternative_routes"] if a["route_id"] == plan["recommended_route_id"])
    assert recommended["capacity_sufficient"] is True
    assert plan["recommended_route_id"] != "SHA-ROT-AIR"


@requires_built_data
def test_plan_shipment_no_disruption_recommends_the_normal_route():
    agent = LogisticsAgent()
    plan = agent.plan_shipment("Shanghai", "Rotterdam", quantity=1000)
    assert plan["recommended_route_id"] == "SHA-ROT-SUEZ"
    assert plan["original_route"]["status"] == "NORMAL"


@requires_built_data
def test_plan_shipment_unknown_lane_raises():
    agent = LogisticsAgent()
    with pytest.raises(ValueError):
        agent.plan_shipment("Nowhere", "Nowhere Else", quantity=100)
