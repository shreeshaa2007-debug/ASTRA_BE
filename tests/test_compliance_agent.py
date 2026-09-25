"""Phase 9 tests — Compliance Agent, brief §21 "AGENTS" category. Every test
that needs a non-default rule (a rejected supplier, a restricted country)
passes its own `rules` dict rather than editing the shared
compliance_rules.yaml — that file's empty defaults are deliberate (see its
own comment) and no test should leave it mutated for the next one.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.agents.compliance import tools
from backend.agents.compliance.agent import ComplianceAgent
from backend.agents.logistics.agent import LogisticsAgent
from backend.agents.sourcing.agent import SourcingAgent

requires_built_data = pytest.mark.skipif(
    not (Path("data/processed/suppliers.csv").exists() and Path("data/processed/routes.csv").exists()),
    reason="run Phase 3's pipeline first",
)


def base_rules(**overrides) -> dict:
    rules = {"rejected_suppliers": [], "restricted_countries": [], "approval_threshold": 500_000}
    rules.update(overrides)
    return rules


# --------------------------------------------------------------------------- #
# load_rules
# --------------------------------------------------------------------------- #
def test_load_rules_has_the_three_required_keys():
    rules = tools.load_rules()
    assert set(rules.keys()) == {"rejected_suppliers", "restricted_countries", "approval_threshold"}
    assert isinstance(rules["approval_threshold"], (int, float))


# --------------------------------------------------------------------------- #
# check_supplier_policy
# --------------------------------------------------------------------------- #
@requires_built_data
def test_check_supplier_policy_passes_a_valid_allocation():
    result = tools.check_supplier_policy([{"supplier_id": "S003", "product_id": "22197", "quantity": 100}], base_rules())
    assert result["passed"] is True


@requires_built_data
def test_check_supplier_policy_rejects_sanctioned_supplier():
    result = tools.check_supplier_policy(
        [{"supplier_id": "S003", "product_id": "22197", "quantity": 100}], base_rules(rejected_suppliers=["S003"])
    )
    assert result["passed"] is False
    assert "S003" in result["detail"]


@requires_built_data
def test_check_supplier_policy_rejects_disrupted_supplier():
    # S001 is DISRUPTED for product 23166 (Chengdu Precision Components — see suppliers.py roster)
    result = tools.check_supplier_policy([{"supplier_id": "S001", "product_id": "23166", "quantity": 10}], base_rules())
    assert result["passed"] is False
    assert "DISRUPTED" in result["detail"]


@requires_built_data
def test_check_supplier_policy_rejects_insufficient_capacity():
    huge_qty = 10_000_000
    result = tools.check_supplier_policy([{"supplier_id": "S003", "product_id": "22197", "quantity": huge_qty}], base_rules())
    assert result["passed"] is False
    assert "short of" in result["detail"]


# --------------------------------------------------------------------------- #
# check_country_policy
# --------------------------------------------------------------------------- #
def test_check_country_policy_passes_when_nothing_restricted():
    result = tools.check_country_policy(["China", "India"], base_rules())
    assert result["passed"] is True


def test_check_country_policy_rejects_restricted_region():
    result = tools.check_country_policy(["China", "India"], base_rules(restricted_countries=["China"]))
    assert result["passed"] is False
    assert "China" in result["detail"]


# --------------------------------------------------------------------------- #
# check_route_policy
# --------------------------------------------------------------------------- #
@requires_built_data
def test_check_route_policy_none_route_id_passes():
    result = tools.check_route_policy(None)
    assert result["passed"] is True


@requires_built_data
def test_check_route_policy_passes_available_route():
    result = tools.check_route_policy("SHA-ROT-SUEZ")
    assert result["passed"] is True


@requires_built_data
def test_check_route_policy_rejects_disrupted_route():
    result = tools.check_route_policy("SHA-ROT-SUEZ", disrupted_route_ids=frozenset({"SHA-ROT-SUEZ"}))
    assert result["passed"] is False


@requires_built_data
def test_check_route_policy_rejects_unknown_route():
    result = tools.check_route_policy("NOT-A-ROUTE")
    assert result["passed"] is False


# --------------------------------------------------------------------------- #
# check_transaction_threshold
# --------------------------------------------------------------------------- #
def test_check_transaction_threshold_at_and_over_boundary():
    rules = base_rules(approval_threshold=1000)
    assert tools.check_transaction_threshold(1000, rules)["passed"] is True   # at threshold: still within
    assert tools.check_transaction_threshold(1000.01, rules)["passed"] is False


# --------------------------------------------------------------------------- #
# validate_plan — status derivation
# --------------------------------------------------------------------------- #
@requires_built_data
def test_validate_plan_approves_a_clean_cheap_plan():
    plan = {"supplier_allocations": [{"supplier_id": "S003", "product_id": "22197", "quantity": 100}], "route_id": None, "total_cost": 9600}
    result = tools.validate_plan(plan, base_rules())
    assert result["status"] == "APPROVED"
    assert result["requires_human"] is False


@requires_built_data
def test_validate_plan_escalates_over_threshold_with_no_violations():
    plan = {"supplier_allocations": [{"supplier_id": "S003", "product_id": "22197", "quantity": 100}], "route_id": None, "total_cost": 600_000}
    result = tools.validate_plan(plan, base_rules())
    assert result["status"] == "ESCALATED"
    assert result["requires_human"] is True


@requires_built_data
def test_validate_plan_rejects_sanctioned_supplier_even_if_cheap():
    plan = {"supplier_allocations": [{"supplier_id": "S003", "product_id": "22197", "quantity": 100}], "route_id": None, "total_cost": 9600}
    result = tools.validate_plan(plan, base_rules(rejected_suppliers=["S003"]))
    assert result["status"] == "REJECTED"
    assert result["requires_human"] is False


@requires_built_data
def test_validate_plan_rejection_takes_priority_over_threshold_escalation():
    """A plan that's BOTH over threshold AND has a hard violation must come
    back REJECTED, not ESCALATED — a violation is disqualifying regardless
    of cost."""
    plan = {"supplier_allocations": [{"supplier_id": "S003", "product_id": "22197", "quantity": 100}], "route_id": None, "total_cost": 999_999}
    result = tools.validate_plan(plan, base_rules(rejected_suppliers=["S003"]))
    assert result["status"] == "REJECTED"


# --------------------------------------------------------------------------- #
# agent.py — real end-to-end against Sourcing + Logistics agent output
# --------------------------------------------------------------------------- #
@requires_built_data
def test_agent_validates_real_sourcing_and_logistics_output():
    sourcing = SourcingAgent().generate_sourcing_mix("22197", required_quantity=3000)
    logistics = LogisticsAgent().plan_shipment("Shanghai", "Rotterdam", quantity=3000, disrupted_route_ids=frozenset({"SHA-ROT-SUEZ"}))

    result = ComplianceAgent().validate_sourcing_and_logistics(sourcing, logistics)
    assert result["status"] in {"APPROVED", "ESCALATED", "REJECTED"}
    route_check = next(c for c in result["checks"] if c["name"] == "route_permitted")
    assert route_check["passed"] is True  # Cape is not disrupted


@requires_built_data
def test_agent_handles_missing_logistics_result():
    sourcing = SourcingAgent().generate_sourcing_mix("22197", required_quantity=100)
    result = ComplianceAgent().validate_sourcing_and_logistics(sourcing, logistics_result=None)
    route_check = next(c for c in result["checks"] if c["name"] == "route_permitted")
    assert route_check["detail"] == "no route in this plan"


@requires_built_data
def test_agent_total_cost_sums_actual_allocation_costs_not_full_quantity_estimates():
    """Regression guard: total_cost must be built from each allocation's own
    landed cost at its allocated quantity, not generate_supplier_options()'s
    'if you bought the whole required_quantity from just this supplier'
    ranking figure (a different, larger number for a split allocation)."""
    sourcing = SourcingAgent().generate_sourcing_mix("22197", required_quantity=10000)  # splits across >=2 suppliers
    assert len(sourcing["supplier_allocations"]) >= 2

    result = ComplianceAgent().validate_sourcing_and_logistics(sourcing, logistics_result=None)
    threshold_check = next(c for c in result["checks"] if c["name"] == "cost_within_threshold")

    from backend.agents.sourcing import tools as sourcing_tools

    expected = sum(
        sourcing_tools.calculate_landed_cost(a["supplier_id"], "22197", a["quantity"])["landed_cost"]
        for a in sourcing["supplier_allocations"]
    )
    reported_cost = float(threshold_check["detail"].split("cost ")[1].split(" ")[0].replace(",", ""))
    assert reported_cost == pytest.approx(expected, abs=0.01)


# --------------------------------------------------------------------------- #
# Phase 14: several routes, state-disrupted suppliers, structured offenders
# --------------------------------------------------------------------------- #
@requires_built_data
def test_check_routes_policy_passes_only_if_every_route_does():
    ok = tools.check_routes_policy(["SHA-ROT-CAPE", "SHA-ROT-RAIL"])
    assert ok["passed"] is True and ok["name"] == "route_permitted" and ok["offenders"] == []

    bad = tools.check_routes_policy(["SHA-ROT-CAPE", "SHA-ROT-SUEZ"], disrupted_route_ids=frozenset({"SHA-ROT-SUEZ"}))
    assert bad["passed"] is False and bad["offenders"] == ["SHA-ROT-SUEZ"] and "SHA-ROT-SUEZ" in bad["detail"] and "SHA-ROT-CAPE" not in bad["detail"]


@requires_built_data
def test_check_routes_policy_with_no_routes_is_the_no_route_case():
    assert tools.check_routes_policy([]) == tools.check_route_policy(None)


@requires_built_data
def test_validate_plan_checks_every_route_in_route_ids():
    plan = {"supplier_allocations": [{"supplier_id": "S003", "product_id": "22197", "quantity": 100}], "total_cost": 9600,
            "route_ids": ["SHA-ROT-CAPE", "SHA-ROT-RAIL"]}
    assert tools.validate_plan(plan, base_rules())["status"] == "APPROVED"
    blocked = tools.validate_plan({**plan, "disrupted_route_ids": frozenset({"SHA-ROT-RAIL"})}, base_rules())
    assert blocked["status"] == "REJECTED" and "SHA-ROT-RAIL" in blocked["reason"]


@requires_built_data
def test_a_single_route_id_still_works_as_before():
    plan = {"supplier_allocations": [{"supplier_id": "S003", "product_id": "22197", "quantity": 100}], "total_cost": 9600, "route_id": "SHA-ROT-SUEZ"}
    assert tools.validate_plan(plan, base_rules())["status"] == "APPROVED"
    assert tools.validate_plan({**plan, "disrupted_route_ids": frozenset({"SHA-ROT-SUEZ"})}, base_rules())["status"] == "REJECTED"


@requires_built_data
def test_a_supplier_disrupted_in_the_world_state_but_not_in_the_data_is_rejected():
    alloc = [{"supplier_id": "S003", "product_id": "22197", "quantity": 100}]  # ACTIVE in suppliers.csv
    assert tools.check_supplier_policy(alloc, base_rules())["passed"] is True
    result = tools.check_supplier_policy(alloc, base_rules(), disrupted_supplier_ids=frozenset({"S003"}))
    assert result["passed"] is False and result["offenders"] == ["S003"] and "DISRUPTED" in result["detail"]

    plan = {"supplier_allocations": alloc, "route_id": None, "total_cost": 9600, "disrupted_supplier_ids": frozenset({"S003"})}
    assert tools.validate_plan(plan, base_rules())["status"] == "REJECTED"


@requires_built_data
def test_failed_checks_name_their_offenders_as_data_not_only_as_prose():
    a = [{"supplier_id": "S003", "product_id": "22197", "quantity": 100}, {"supplier_id": "S007", "product_id": "22197", "quantity": 10_000_000}]
    supplier = tools.check_supplier_policy(a, base_rules(rejected_suppliers=["S003"]))
    assert supplier["offenders"] == ["S003", "S007"]  # sanctioned, and over capacity

    assert tools.check_supplier_policy(a[:1], base_rules())["offenders"] == []
    assert tools.check_country_policy(["China", "India"], base_rules(restricted_countries=["China"]))["offenders"] == ["China"]
    assert tools.check_country_policy(["India"], base_rules())["offenders"] == []
    assert tools.check_route_policy("NOT-A-ROUTE")["offenders"] == ["NOT-A-ROUTE"]
    assert tools.check_route_policy("SHA-ROT-SUEZ")["passed"] is True


@requires_built_data
def test_a_rejected_verdict_carries_offenders_a_replan_can_use():
    plan = {"supplier_allocations": [{"supplier_id": "S003", "product_id": "22197", "quantity": 100}], "route_id": None, "total_cost": 9600}
    verdict = tools.validate_plan(plan, base_rules(rejected_suppliers=["S003"]))
    failed = [c for c in verdict["checks"] if not c["passed"]]
    assert verdict["status"] == "REJECTED" and [c["offenders"] for c in failed] == [["S003"]]
