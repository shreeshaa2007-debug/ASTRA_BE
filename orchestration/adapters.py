"""Glue between what the optimizer produces and what compliance consumes.

Compliance (Phase 9) was written against one sourcing allocation list and one
route; an optimized plan buys from several suppliers and ships on several
routes. These functions translate, and turn a REJECTED verdict into the
constraints a replan must respect.
"""
from __future__ import annotations

from collections import defaultdict

from backend.schemas.optimization import OptimizationProblem, OptimizationSolution

# The terms of the objective that are money actually spent. The delay penalty is
# a modeling device (what lateness is "worth"), not a cost to approve.
SPEND_TERMS = ("procurement", "tariff", "freight", "transfer")


def plan_spend(solution: OptimizationSolution) -> float:
    """What the plan commits the business to paying — what the approval
    threshold in compliance_rules.yaml is compared against."""
    return round(sum(solution.objective_terms.get(term, 0.0) for term in SPEND_TERMS), 2)


def plan_to_compliance_input(
    solution: OptimizationSolution,
    product_id: str,
    disrupted_route_ids: frozenset[str] = frozenset(),
    disrupted_supplier_ids: frozenset[str] = frozenset(),
) -> dict:
    """The `plan` dict compliance's validate_plan() takes. Allocations are
    summed per supplier, because compliance checks a supplier's capacity against
    what it is asked to supply in total, not per route."""
    per_supplier: dict[str, int] = defaultdict(int)
    for a in solution.allocations:
        per_supplier[a.supplier_id] += a.quantity
    return {
        "supplier_allocations": [{"supplier_id": s, "product_id": str(product_id), "quantity": q} for s, q in sorted(per_supplier.items())],
        "route_ids": sorted({a.route_id for a in solution.allocations if a.route_id}),
        "total_cost": plan_spend(solution),
        "disrupted_route_ids": frozenset(disrupted_route_ids),
        "disrupted_supplier_ids": frozenset(disrupted_supplier_ids),
    }


def exclusions_from_verdict(checks: list[dict], problem: OptimizationProblem) -> tuple[frozenset[str], frozenset[str]]:
    """(supplier ids, route ids) a replan must leave out, read from the failed
    checks' structured `offenders`. A restricted country excludes every supplier
    in it, not only the ones the rejected plan happened to use. Both empty means
    the verdict names nothing to change — replanning would reproduce the same plan."""
    suppliers: set[str] = set()
    routes: set[str] = set()
    for check in checks:
        if check.get("passed"):
            continue
        offenders = check.get("offenders") or []
        if check["name"] == "supplier_permitted":
            suppliers.update(offenders)
        elif check["name"] == "route_permitted":
            routes.update(offenders)
        elif check["name"] == "country_permitted":
            suppliers.update(s.supplier_id for s in problem.suppliers if s.region in offenders)
    return frozenset(suppliers), frozenset(routes)
