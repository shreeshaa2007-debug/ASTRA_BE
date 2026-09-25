"""Why the plan looks the way it does — for the Decision Center (api-plan.md
`GET /api/decisions/{id}`) and for the "deviation is recorded" rule in
agent-plan.md: the engine may depart from an agent's recommendation only when
a constraint or a cheaper joint option forces it to, and says which.

Template-driven and fact-only: every reason is derived from numbers already in
the problem or solution. No LLM, and no claim the data can't back — when the
cause of a difference isn't determinable, the reason says only that the joint
cost-minimal plan differs.
"""
from __future__ import annotations

from collections import defaultdict

from backend.optimization import costing
from backend.schemas.optimization import (
    Allocation,
    ConstraintStatus,
    ExcludedOption,
    OptimizationProblem,
    Transfer,
)

_GENERIC = "the joint cost-minimal plan differs from this agent's independent recommendation"


def _best_lane_cost(problem: OptimizationProblem, supplier_id: str, lanes: costing.LaneSet) -> float | None:
    costs = [costing.lane_total_unit_cost(problem.parameters, l.supplier, l.route) for l in lanes.lanes if l.supplier.supplier_id == supplier_id]
    return min(costs) if costs else None


def build_deviations(
    problem: OptimizationProblem,
    allocations: list[Allocation],
    transfers: list[Transfer],
    excluded: list[ExcludedOption],
    lanes: costing.LaneSet,
) -> list[dict]:
    recs = problem.agent_recommendations
    warehouses = {w.warehouse_id: w for w in problem.warehouses}
    suppliers = {s.supplier_id: s for s in problem.suppliers}
    routes = {r.route_id: r for r in problem.routes}
    deviations: list[dict] = []

    # --- Inventory Agent: recommended transfers ---------------------------------
    planned_transfers = {(t.from_warehouse, t.to_warehouse): t.quantity for t in transfers}
    recommended_pairs = set()
    for rec in recs.inventory_transfers:
        pair = (rec["from"], rec["to"])
        recommended_pairs.add(pair)
        planned = planned_transfers.get(pair, 0)
        if planned == rec["quantity"]:
            continue
        source = warehouses.get(rec["from"])
        spare = max(0, source.projected_surplus) if source else 0
        if source and rec["quantity"] > spare:
            reason = (
                f"{rec['from']} can spare only {spare} units after its own forecast demand ({source.demand_units}) "
                f"and safety stock ({source.safety_stock}); the agent sized the transfer against safety stock alone"
            )
        else:
            reason = _GENERIC
        deviations.append({
            "agent": "inventory",
            "recommended": f"{rec['quantity']} units {rec['from']} -> {rec['to']}",
            "planned": f"{planned} units",
            "reason": reason,
        })
    for (src, dst), qty in sorted(planned_transfers.items()):
        if (src, dst) in recommended_pairs:
            continue
        target = warehouses[dst]
        deviations.append({
            "agent": "inventory",
            "recommended": "no transfer",
            "planned": f"{qty} units {src} -> {dst}",
            "reason": (
                f"{dst} is short {-target.projected_surplus} units against forecast demand plus safety stock over the "
                f"{problem.parameters.planning_horizon_days}-day horizon; {src} has {max(0, warehouses[src].projected_surplus)} to spare"
            ),
        })

    # --- Sourcing Agent: greedy landed-cost mix ---------------------------------
    agent_qty = {a["supplier_id"]: a["quantity"] for a in recs.sourcing_allocations}
    plan_qty: dict[str, int] = defaultdict(int)
    for a in allocations:
        plan_qty[a.supplier_id] += a.quantity
    unusable = {e.id: e.reason for e in excluded if e.kind == "supplier"}
    used_costs = [
        costing.lane_total_unit_cost(problem.parameters, suppliers[a.supplier_id], routes.get(a.route_id) if a.route_id else None)
        for a in allocations
    ]
    marginal = max(used_costs) if used_costs else None

    for sid in sorted(set(agent_qty) | set(plan_qty)):
        recommended, planned = agent_qty.get(sid, 0), plan_qty.get(sid, 0)
        if recommended == planned:
            continue
        best = _best_lane_cost(problem, sid, lanes)
        if sid in unusable:
            reason = f"unusable in the joint plan: {unusable[sid]}"
        elif planned > recommended and best is not None:
            reason = f"the agent ranks suppliers by landed cost alone; counting freight and lead time this one costs {best:.2f}/unit all-in"
        elif planned < recommended and best is not None and marginal is not None and best > marginal:
            reason = f"best all-in cost {best:.2f}/unit (landed + freight) exceeds the {marginal:.2f}/unit of the plan's most expensive lane in use"
        elif planned < recommended and best is not None:
            reason = f"all-in cost {best:.2f}/unit is competitive, but shared supplier or route capacity is used up by other lanes"
        else:
            reason = _GENERIC
        deviations.append({"agent": "sourcing", "recommended": f"{recommended} units from {sid}", "planned": f"{planned} units", "reason": reason})

    # --- Logistics Agent: recommended route per lane ----------------------------
    route_qty: dict[str, int] = defaultdict(int)
    origin_qty: dict[str, int] = defaultdict(int)
    for a in allocations:
        if a.route_id:
            route_qty[a.route_id] += a.quantity
            origin_qty[routes[a.route_id].origin] += a.quantity
    # only a routing deviation when the origin actually ships in the plan: a lane the
    # plan never uses (no supplier there is allocated) isn't the router being overruled
    for route_id in recs.logistics_route_ids:
        route = routes.get(route_id)
        if route is None or route_qty.get(route_id) or not origin_qty.get(route.origin):
            continue
        others = sorted(r for r, q in route_qty.items() if q and routes[r].origin == route.origin)
        deviations.append({
            "agent": "logistics",
            "recommended": route_id,
            "planned": f"0 units; {route.origin} volume goes on {', '.join(others)}",
            "reason": "the agent picks the cheapest route by freight alone; the joint plan also applies each supplier's lead time to the delivery deadline and delay penalty",
        })

    return deviations


def build_decision_factors(
    problem: OptimizationProblem,
    allocations: list[Allocation],
    transfers: list[Transfer],
    constraints: list[ConstraintStatus],
    excluded: list[ExcludedOption],
    lanes: costing.LaneSet,
) -> list[dict[str, str]]:
    factors: list[dict[str, str]] = []
    params = problem.parameters

    if problem.disrupted_route_ids:
        factors.append({
            "factor": "Disrupted routes excluded",
            "detail": f"{', '.join(sorted(problem.disrupted_route_ids))} unavailable; the solver was never offered them.",
        })
    stranded = [e for e in excluded if e.kind == "supplier"]
    if stranded:
        factors.append({"factor": "Suppliers with no usable lane", "detail": "; ".join(f"{e.id}: {e.reason}" for e in stranded)})

    if allocations:
        total = sum(a.quantity for a in allocations)
        by_supplier: dict[str, int] = defaultdict(int)
        by_mode: dict[str, int] = defaultdict(int)
        for a in allocations:
            by_supplier[a.supplier_id] += a.quantity
            by_mode[a.transport_mode or "direct"] += a.quantity
        factors.append({
            "factor": "Sourcing mix",
            "detail": f"{total} units bought: " + ", ".join(f"{s} {q} ({q / total:.0%})" for s, q in sorted(by_supplier.items())),
        })
        factors.append({
            "factor": "Freight split",
            "detail": ", ".join(f"{m} {q} ({q / total:.0%})" for m, q in sorted(by_mode.items())),
        })
        latest = max(a.arrival_days for a in allocations)
        factors.append({
            "factor": "Delivery deadline",
            "detail": f"last arrival on day {latest:.1f} against a day-{params.max_delivery_days:g} deadline.",
        })

    limits = [c for c in constraints if c.binding and c.category in ("supplier_capacity", "route_capacity")]
    if limits:
        factors.append({
            "factor": "Capacity limits reached",
            "detail": "; ".join(f"{c.name.split(':', 1)[1]} full at {c.bound:g}" for c in limits) + " — volume beyond these spills to the next-cheapest lane.",
        })

    if transfers:
        moved = sum(t.quantity for t in transfers)
        all_in = [costing.lane_total_unit_cost(params, l.supplier, l.route) for l in lanes.lanes]
        vs = f" instead of buying new stock at {min(all_in):.2f}/unit or more" if all_in else ""
        factors.append({
            "factor": "Internal transfers used",
            "detail": f"{moved} units moved between warehouses at {costing.transfer_unit_cost(params):.2f}/unit{vs}.",
        })
    return factors
