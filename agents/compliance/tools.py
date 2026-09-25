"""Compliance Agent tool functions — per agent-plan.md: check_supplier_policy(),
check_country_policy(), check_transaction_threshold(), validate_plan().

Deterministic only, by explicit brief §12 instruction — no LLM call anywhere
in this module. Rules are read from backend/config/compliance_rules.yaml
(load_rules()), never hardcoded, so changing policy is a config edit, not a
code change. Reuses the Sourcing/Logistics agents' own tool functions for
supplier/route facts rather than re-reading their CSVs independently — one
source of truth per fact, not two copies that can drift apart.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from backend.agents.logistics import tools as logistics_tools
from backend.agents.sourcing import tools as sourcing_tools

RULES_PATH = Path("backend/config/compliance_rules.yaml")


def load_rules(path: Path = RULES_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def check_supplier_policy(
    supplier_allocations: list[dict], rules: dict | None = None, disrupted_supplier_ids: frozenset[str] = frozenset()
) -> dict:
    """Per allocation {"supplier_id", "product_id", "quantity"}: rejects if
    the supplier is on the sanctioned list, is DISRUPTED (§12's "unavailable
    supplier"), or can't actually cover the allocated quantity (§12's
    "insufficient supplier capacity") — reusing the Sourcing Agent's own
    check_supplier_capacity() rather than re-deriving it.
    """
    rules = rules or load_rules()
    violations = []
    offenders: list[str] = []  # structured, so a replan can exclude exactly these
    for alloc in supplier_allocations:
        supplier_id, product_id, quantity = alloc["supplier_id"], alloc["product_id"], alloc["quantity"]

        if supplier_id in rules["rejected_suppliers"]:
            violations.append(f"{supplier_id} is on the rejected-suppliers list")
            offenders.append(supplier_id)
            continue

        if supplier_id in disrupted_supplier_ids:  # disrupted in the shared world state, not just in suppliers.csv
            violations.append(f"{supplier_id} is currently DISRUPTED (unavailable)")
            offenders.append(supplier_id)
            continue

        matches = [s for s in sourcing_tools.get_suppliers(product_id=product_id) if s["supplier_id"] == supplier_id]
        if not matches:
            violations.append(f"{supplier_id} does not supply product {product_id}")
            offenders.append(supplier_id)
            continue
        if matches[0]["status"] == "DISRUPTED":
            violations.append(f"{supplier_id} is currently DISRUPTED (unavailable)")
            offenders.append(supplier_id)
            continue

        cap = sourcing_tools.check_supplier_capacity(supplier_id, product_id, quantity)
        if not cap["sufficient"]:
            violations.append(f"{supplier_id} capacity ({cap['capacity']}) is short of the requested {quantity} by {cap['shortfall']}")
            offenders.append(supplier_id)

    return {
        "name": "supplier_permitted", "passed": len(violations) == 0,
        "detail": "; ".join(violations) or "all suppliers permitted and available", "offenders": sorted(set(offenders)),
    }


def check_country_policy(regions: list[str], rules: dict | None = None) -> dict:
    rules = rules or load_rules()
    restricted_hit = sorted(set(regions) & set(rules["restricted_countries"]))
    return {
        "name": "country_permitted",
        "passed": len(restricted_hit) == 0,
        "detail": f"restricted: {', '.join(restricted_hit)}" if restricted_hit else "no restricted countries involved",
        "offenders": restricted_hit,
    }


def check_route_policy(route_id: str | None, disrupted_route_ids: frozenset[str] = frozenset()) -> dict:
    if route_id is None:
        return {"name": "route_permitted", "passed": True, "detail": "no route in this plan"}
    routes = logistics_tools.get_routes(disrupted_route_ids=disrupted_route_ids)
    match = next((r for r in routes if r["route_id"] == route_id), None)
    if match is None:
        return {"name": "route_permitted", "passed": False, "detail": f"unknown route_id {route_id!r}", "offenders": [route_id]}
    if match["status"] == "DISRUPTED":
        return {"name": "route_permitted", "passed": False, "detail": f"{route_id} is currently DISRUPTED", "offenders": [route_id]}
    return {"name": "route_permitted", "passed": True, "detail": f"{route_id} is available"}


def check_routes_policy(route_ids: list[str], disrupted_route_ids: frozenset[str] = frozenset()) -> dict:
    """`check_route_policy` for a plan that ships on several routes: passes only if every one does.
    Same check name, so a caller reading `route_permitted` needs no change."""
    if not route_ids:
        return check_route_policy(None)
    results = [check_route_policy(r, disrupted_route_ids) for r in route_ids]
    failed = [r for r in results if not r["passed"]]
    return {
        "name": "route_permitted", "passed": not failed,
        "detail": "; ".join(r["detail"] for r in (failed or results)),
        "offenders": [o for r in failed for o in r["offenders"]],
    }


def check_transaction_threshold(total_cost: float, rules: dict | None = None) -> dict:
    rules = rules or load_rules()
    threshold = rules["approval_threshold"]
    within = total_cost <= threshold
    return {
        "name": "cost_within_threshold",
        "passed": within,
        "detail": f"cost {total_cost:,.2f} {'is within' if within else 'exceeds'} the {threshold:,.2f} approval threshold",
    }


def validate_plan(plan: dict, rules: dict | None = None) -> dict:
    """`plan`: {"supplier_allocations": [...], "route_id": str|None,
    "total_cost": float, "disrupted_route_ids": frozenset (optional)}. A plan that
    ships on several routes gives `route_ids` (a list) instead of `route_id`;
    `disrupted_supplier_ids` (optional) lists suppliers disrupted in the world state.

    Status rule (brief §12): any hard violation (supplier/country/route) ->
    REJECTED, regardless of cost. No hard violation but over threshold ->
    ESCALATED, human approval required. Everything passes and under
    threshold -> APPROVED, no human step needed — matches agent-plan.md's
    "low-impact, in-policy actions proceed automatically; the escalation
    path exists specifically for the high-impact case."
    """
    rules = rules or load_rules()
    disrupted_route_ids = plan.get("disrupted_route_ids", frozenset())
    disrupted_supplier_ids = plan.get("disrupted_supplier_ids", frozenset())
    route_ids = list(plan.get("route_ids") or ([plan["route_id"]] if plan.get("route_id") else []))

    supplier_regions = [
        s["region"]
        for alloc in plan.get("supplier_allocations", [])
        for s in sourcing_tools.get_suppliers(product_id=alloc["product_id"])
        if s["supplier_id"] == alloc["supplier_id"]
    ]

    checks = [
        check_supplier_policy(plan.get("supplier_allocations", []), rules, disrupted_supplier_ids),
        check_country_policy(supplier_regions, rules),
        check_routes_policy(route_ids, disrupted_route_ids),
        check_transaction_threshold(plan.get("total_cost", 0.0), rules),
    ]

    hard_violations = [c for c in checks if not c["passed"] and c["name"] != "cost_within_threshold"]
    threshold_check = next(c for c in checks if c["name"] == "cost_within_threshold")

    if hard_violations:
        status, requires_human = "REJECTED", False
        reason = "; ".join(c["detail"] for c in hard_violations)
    elif not threshold_check["passed"]:
        status, requires_human = "ESCALATED", True
        reason = threshold_check["detail"]
    else:
        status, requires_human = "APPROVED", False
        reason = "all checks passed"

    return {"status": status, "checks": checks, "reason": reason, "requires_human": requires_human}
