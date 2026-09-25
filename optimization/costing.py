"""The one cost and lane model the engine, the validator and the explanation
all share. Kept separate from the LP assembly so validate_solution() re-derives
a plan's cost from the problem tables rather than trusting the solver's number.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from backend.schemas.optimization import (
    Allocation,
    ExcludedOption,
    OptimizationParameters,
    OptimizationProblem,
    RouteOption,
    SupplierOption,
    Transfer,
)

OBJECTIVE_TERMS = ("procurement", "tariff", "freight", "transfer", "delay_penalty")


@dataclass(frozen=True)
class Lane:
    """One way to get a supplier's goods to the hub: the supplier plus the
    route it ships on (None = no freight leg modeled for that supplier)."""

    supplier: SupplierOption
    route: RouteOption | None

    @property
    def label(self) -> str:
        return f"{self.supplier.supplier_id}:{self.route.route_id if self.route else 'direct'}"


@dataclass
class LaneSet:
    lanes: list[Lane] = field(default_factory=list)
    excluded: list[ExcludedOption] = field(default_factory=list)
    deadline_blocked: list[str] = field(default_factory=list)  # lane labels dropped only for missing the deadline


def lateness_days(params: OptimizationParameters, arrival_days: float) -> float:
    if params.target_delivery_days is None:
        return 0.0
    return max(0.0, arrival_days - params.target_delivery_days)


def arrival_days(supplier: SupplierOption, route: RouteOption | None) -> float:
    return supplier.lead_time_days + (route.transit_time_days if route else 0.0)


def lane_unit_costs(params: OptimizationParameters, supplier: SupplierOption, route: RouteOption | None) -> dict[str, float]:
    """Per-unit cost of buying from `supplier` and shipping on `route`, split
    into the same terms the objective reports."""
    delay = params.delay_penalty_per_unit_day * lateness_days(params, arrival_days(supplier, route))
    return {
        "procurement": supplier.base_unit_cost,
        "tariff": supplier.landed_unit_cost - supplier.base_unit_cost,
        "freight": route.cost_per_unit if route else 0.0,
        "delay_penalty": delay,
    }


def lane_total_unit_cost(params: OptimizationParameters, supplier: SupplierOption, route: RouteOption | None) -> float:
    return sum(lane_unit_costs(params, supplier, route).values())


def transfer_unit_cost(params: OptimizationParameters) -> float:
    """Transfers also pay the lateness penalty (their transit is short, but
    the formula is the same one external lanes use)."""
    return params.internal_transfer_cost_per_unit + params.delay_penalty_per_unit_day * lateness_days(
        params, params.internal_transfer_days
    )


def compute_objective_terms(problem: OptimizationProblem, allocations: list[Allocation], transfers: list[Transfer]) -> dict[str, float]:
    """Objective terms for a plan, looked up from the problem's own supplier
    and route tables — an allocation's stored unit costs are not trusted."""
    suppliers = {s.supplier_id: s for s in problem.suppliers}
    routes = {r.route_id: r for r in problem.routes}
    terms = dict.fromkeys(OBJECTIVE_TERMS, 0.0)

    for a in allocations:
        unit = lane_unit_costs(problem.parameters, suppliers[a.supplier_id], routes.get(a.route_id) if a.route_id else None)
        for term in ("procurement", "tariff", "freight", "delay_penalty"):
            terms[term] += unit[term] * a.quantity

    params = problem.parameters
    transfer_delay = params.delay_penalty_per_unit_day * lateness_days(params, params.internal_transfer_days)
    for t in transfers:
        terms["transfer"] += params.internal_transfer_cost_per_unit * t.quantity
        terms["delay_penalty"] += transfer_delay * t.quantity

    return {k: round(v, 6) for k, v in terms.items()}


def enumerate_lanes(problem: OptimizationProblem) -> LaneSet:
    """Every supplier x route lane the solver may use, plus what it dropped and
    why. A supplier ships only from its own origin port; a lane that can't
    arrive within max_delivery_days is dropped, and a supplier left with no lane
    is reported as excluded — never silently omitted."""
    params = problem.parameters
    routes_by_origin: dict[str, list[RouteOption]] = defaultdict(list)
    for r in problem.routes:
        if r.capacity > 0:
            routes_by_origin[r.origin].append(r)

    result = LaneSet()
    for s in problem.suppliers:
        if s.capacity <= 0:
            result.excluded.append(ExcludedOption(kind="supplier", id=s.supplier_id, reason="no capacity"))
            continue
        options: list[RouteOption | None] = [None] if s.origin_port is None else routes_by_origin.get(s.origin_port, [])
        if not options:
            result.excluded.append(ExcludedOption(
                kind="supplier", id=s.supplier_id,
                reason=f"no available route from {s.origin_port} (its lanes are disrupted or none are on file)",
            ))
            continue

        kept = 0
        missed: list[str] = []
        for route in options:
            lane = Lane(s, route)
            days = arrival_days(s, route)
            if days > params.max_delivery_days:
                reason = f"arrives on day {days:.1f}, after the day-{params.max_delivery_days:g} deadline"
                result.excluded.append(ExcludedOption(kind="lane", id=lane.label, reason=reason))
                result.deadline_blocked.append(lane.label)
                missed.append(f"{lane.label} {reason}")
                continue
            result.lanes.append(lane)
            kept += 1
        if kept == 0:
            result.excluded.append(ExcludedOption(kind="supplier", id=s.supplier_id, reason="every lane misses the deadline: " + "; ".join(missed)))
    return result
