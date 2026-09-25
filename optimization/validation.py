"""Independent re-check of a plan against the problem's constraints.

Deliberately shares no code with the LP matrix assembly in engine.py: if the
matrices are built wrong, the solver can return a "feasible" answer that breaks
a real constraint, and this is the layer that catches it (brief §11 / agent-plan
`validate_solution`). Everything here is recomputed from plain tables.
"""
from __future__ import annotations

from collections import defaultdict

from backend.optimization import costing
from backend.schemas.optimization import (
    Allocation,
    ConstraintStatus,
    OptimizationProblem,
    OptimizationSolution,
    Transfer,
    ValidationResult,
)

TOL = 1e-6


def _status(name: str, category: str, sense: str, value: float, bound: float, detail: str) -> ConstraintStatus:
    if sense == "<=":
        satisfied = value <= bound + TOL
    elif sense == ">=":
        satisfied = value >= bound - TOL
    else:
        satisfied = abs(value - bound) <= TOL
    return ConstraintStatus(
        name=name, category=category, sense=sense, value=round(value, 6), bound=round(bound, 6),
        satisfied=satisfied, binding=satisfied and abs(value - bound) <= TOL, detail=detail,
    )


def end_stock_by_warehouse(problem: OptimizationProblem, transfers: list[Transfer], inbound_by_warehouse: dict[str, int]) -> dict[str, int]:
    """Stock each warehouse ends the horizon with: what it holds, plus inbound
    and transfers in, minus transfers out and its own forecast demand."""
    received: dict[str, int] = defaultdict(int)
    shipped: dict[str, int] = defaultdict(int)
    for t in transfers:
        received[t.to_warehouse] += t.quantity
        shipped[t.from_warehouse] += t.quantity
    return {
        w.warehouse_id: w.current_stock + inbound_by_warehouse.get(w.warehouse_id, 0)
        + received[w.warehouse_id] - shipped[w.warehouse_id] - w.demand_units
        for w in problem.warehouses
    }


def evaluate_constraints(
    problem: OptimizationProblem,
    allocations: list[Allocation],
    transfers: list[Transfer],
    inbound_by_warehouse: dict[str, int],
) -> list[ConstraintStatus]:
    suppliers = {s.supplier_id: s for s in problem.suppliers}
    routes = {r.route_id: r for r in problem.routes}
    params = problem.parameters
    out: list[ConstraintStatus] = []

    used_by_supplier: dict[str, int] = defaultdict(int)
    used_by_route: dict[str, int] = defaultdict(int)
    for a in allocations:
        used_by_supplier[a.supplier_id] += a.quantity
        if a.route_id:
            used_by_route[a.route_id] += a.quantity

    for s in problem.suppliers:
        used = used_by_supplier[s.supplier_id]
        out.append(_status(f"supplier_capacity:{s.supplier_id}", "supplier_capacity", "<=", used, s.capacity,
                           f"{used} of {s.capacity} units of {s.supplier_id}'s capacity"))
    for r in problem.routes:
        used = used_by_route[r.route_id]
        out.append(_status(f"route_capacity:{r.route_id}", "route_capacity", "<=", used, r.capacity,
                           f"{used} of {r.capacity} units on {r.route_id} ({r.transport_mode})"))

    # every allocation must sit on a real supplier and a route that starts at that
    # supplier's port (or on no route iff no freight leg is modeled for it), and
    # never on a disrupted route
    disrupted = set(problem.disrupted_route_ids)
    bad_lanes = []
    for a in allocations:
        s = suppliers.get(a.supplier_id)
        r = routes.get(a.route_id) if a.route_id else None
        if s is None:
            bad_lanes.append(f"unknown supplier {a.supplier_id}")
        elif a.route_id and (r is None or a.route_id in disrupted):
            bad_lanes.append(f"{a.supplier_id} on unavailable route {a.route_id}")
        elif (r.origin if r else None) != s.origin_port:
            bad_lanes.append(f"{a.supplier_id} (port {s.origin_port}) cannot use route {a.route_id}")
    out.append(_status("lane_validity", "lane_validity", "==", len(bad_lanes), 0,
                       "; ".join(bad_lanes) or "every allocation uses an available lane from its supplier's port"))

    latest = max((costing.arrival_days(suppliers[a.supplier_id], routes.get(a.route_id) if a.route_id else None)
                  for a in allocations if a.supplier_id in suppliers), default=0.0)
    out.append(_status("delivery_deadline", "delivery_deadline", "<=", latest, params.max_delivery_days,
                       f"latest arrival on day {latest:.1f}, deadline day {params.max_delivery_days:.1f}"))

    procured = sum(a.quantity for a in allocations)
    distributed = sum(inbound_by_warehouse.values())
    out.append(_status("inbound_balance", "flow_balance", "==", procured - distributed, 0,
                       f"{procured} units procured, {distributed} distributed to warehouses"))

    transfer_out: dict[str, int] = defaultdict(int)
    for t in transfers:
        transfer_out[t.from_warehouse] += t.quantity
    end_stock_by_wh = end_stock_by_warehouse(problem, transfers, inbound_by_warehouse)

    for w in problem.warehouses:
        end_stock = end_stock_by_wh[w.warehouse_id]
        out.append(_status(f"safety_stock:{w.warehouse_id}", "safety_stock", ">=", end_stock, w.safety_stock,
                           f"{w.warehouse_id} ends the horizon with {end_stock} units vs {w.safety_stock} safety stock"))
        out.append(_status(f"transfer_availability:{w.warehouse_id}", "transfer_availability", "<=", transfer_out[w.warehouse_id],
                           w.current_stock, f"{w.warehouse_id} ships out {transfer_out[w.warehouse_id]} of {w.current_stock} units on hand"))

    non_int = [f"{a.supplier_id}/{a.route_id}" for a in allocations if a.quantity <= 0] + \
              [f"{t.from_warehouse}->{t.to_warehouse}" for t in transfers if t.quantity <= 0]
    non_int += [w for w, q in inbound_by_warehouse.items() if q < 0]
    out.append(_status("integral_nonnegative_quantities", "integrality", "==", len(non_int), 0,
                       "; ".join(non_int) or "all quantities are positive whole units"))
    return out


def validate(solution: OptimizationSolution) -> ValidationResult:
    if solution.status != "OPTIMAL":
        return ValidationResult(valid=False, violations=[f"solution status is {solution.status}, not a plan"], checks_run=1)

    problem = solution.problem
    constraints = evaluate_constraints(problem, solution.allocations, solution.transfers, solution.inbound_by_warehouse)
    violations = [f"{c.name}: {c.detail}" for c in constraints if not c.satisfied]
    checks = len(constraints)

    # the reported cost must be what this plan actually costs under the problem's own
    # tables — only checkable when every allocation names a real supplier/route
    checks += 1
    if next(c for c in constraints if c.name == "lane_validity").satisfied:
        terms = costing.compute_objective_terms(problem, solution.allocations, solution.transfers)
        if solution.objective_value is None or abs(sum(terms.values()) - solution.objective_value) > 1e-4 * max(1.0, abs(solution.objective_value or 0.0)):
            violations.append(f"objective_value {solution.objective_value} does not match the recomputed plan cost {sum(terms.values()):.6f}")
    else:
        violations.append("objective_value could not be re-derived because the plan uses invalid lanes")

    checks += 1
    if solution.end_stock_by_warehouse != end_stock_by_warehouse(problem, solution.transfers, solution.inbound_by_warehouse):
        violations.append("end_stock_by_warehouse does not match the plan's own stock movements")

    return ValidationResult(valid=not violations, violations=violations, checks_run=checks)
