"""Baseline vs. unmitigated vs. mitigated — what a disruption costs and what the response buys.

Three cases for one product, all from the same agents and the same optimizer the
live pipeline uses:

  baseline     the plan the optimizer would choose in normal conditions.
  unmitigated  that same plan left untouched ("inaction") when the disruption hits:
               purchases from a disrupted supplier, and volume on a disrupted
               route, never arrive; what does arrive is re-priced at the tariffs
               in effect. Nothing is re-optimized.
  mitigated    the optimizer's plan for the disrupted world — a fresh solve for a
               scenario, or the simulation's own plan when comparing a simulation.

Rules this module holds to (the project's rule is that nothing is invented):

* There is no shipment dataset, so the baseline is a *modeled counterfactual*,
  never observed shipments. Every result says so.
* There is no stock-out penalty in the model, so a shortfall is reported in
  units and never converted to money. `value_at_risk` is the baseline plan's own
  price for the purchases the disruption invalidates — a fact of the plan, not a
  loss estimate.
* Costs are in the MVP cost unit (no currency peg).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Literal, Mapping, Optional

from pydantic import BaseModel, Field

from backend.agents.sourcing import tools as sourcing_tools
from backend.optimization import tools as optimization_tools
from backend.optimization.engine import OptimizationEngine
from backend.orchestration import adapters
from backend.schemas.entities import RouteStatus, SupplierStatus
from backend.schemas.optimization import Allocation, OptimizationSolution
from backend.schemas.world_state import DisruptionEvent, WorldState
from backend.services.world_state import load_baseline, state_changes_for_event
from backend.simulation.scenarios import ScenarioDefinition


# --------------------------------------------------------------------------- #
# the world a case is evaluated in
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WorldSpec:
    """What the agents are told is true: disrupted routes and suppliers, and the tariffs in effect.
    Hashable, so a solve can be cached on it."""

    disrupted_routes: frozenset[str] = frozenset()
    disrupted_suppliers: frozenset[str] = frozenset()
    tariffs: tuple[tuple[str, float], ...] = ()

    @classmethod
    def of(cls, routes, suppliers, tariffs: Mapping[str, float]) -> "WorldSpec":
        return cls(frozenset(routes), frozenset(suppliers), tuple(sorted((k, float(v)) for k, v in tariffs.items())))

    @property
    def tariff_rates(self) -> dict[str, float]:
        return dict(self.tariffs)


def baseline_world() -> WorldSpec:
    """Normal conditions: the network as the data files have it."""
    base = load_baseline()
    return WorldSpec.of(
        (r for r, s in base.route_status.items() if s == RouteStatus.DISRUPTED),
        (s for s, st in base.supplier_status.items() if st == SupplierStatus.DISRUPTED),
        base.tariffs,
    )


def world_after_event(event: DisruptionEvent, tariff_overrides: Mapping[str, float] | None = None) -> WorldSpec:
    """The baseline with `event` applied by the pipeline's own rule (EVENT_EFFECTS), plus tariff overrides."""
    base = load_baseline()
    now = datetime.now(timezone.utc)
    state = WorldState(
        simulation_id="scenario", scenario_type="SCENARIO", created_at=now, timestamp=now,
        route_status=base.route_status, supplier_status=base.supplier_status, tariffs=base.tariffs,
    )
    after = state.model_copy(update=state_changes_for_event(state, event))
    return WorldSpec.of(after.disrupted_route_ids(), after.disrupted_supplier_ids(), {**base.tariffs, **(tariff_overrides or {})})


def world_of_state(state: WorldState) -> WorldSpec:
    return WorldSpec.of(state.disrupted_route_ids(), state.disrupted_supplier_ids(), state.tariffs)


# --------------------------------------------------------------------------- #
# solving a world
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Solved:
    solution: OptimizationSolution
    model_version: str


def _solve(product_id: str, as_of_date: Optional[str], world: WorldSpec, engine: OptimizationEngine | None) -> Solved:
    out = optimization_tools.gather_agent_outputs(product_id, as_of_date, world.disrupted_routes, world.disrupted_suppliers, world.tariff_rates)
    problem = optimization_tools.build_optimization_problem(
        product_id, out.inventory_results, out.sourcing_result, out.logistics_results, out.safety_stock_by_warehouse, out.parameters)
    return Solved(optimization_tools.optimize_supply_chain(problem, engine), out.forecast_model_version)


@lru_cache(maxsize=128)
def _solve_cached(product_id: str, as_of_date: Optional[str], world: WorldSpec) -> Solved:
    return _solve(product_id, as_of_date, world, None)


def solve_world(product_id: str, as_of_date: Optional[str], world: WorldSpec, engine: OptimizationEngine | None = None) -> Solved:
    """Deterministic in its arguments, so the default-engine result is cached: the Scenarios page and the
    dashboard ask for the same baseline again and again. The returned solution is shared — treat it as read-only."""
    if engine is None:
        return _solve_cached(str(product_id), as_of_date, world)
    return _solve(str(product_id), as_of_date, world, engine)


# --------------------------------------------------------------------------- #
# result models
# --------------------------------------------------------------------------- #
class ModeShare(BaseModel):
    units: int
    share: float


class CaseSummary(BaseModel):
    case: Literal["baseline", "unmitigated", "mitigated"]
    label: str
    status: Literal["OPTIMAL", "INFEASIBLE", "ERROR", "EVALUATED", "NOT_AVAILABLE"]
    message: str = ""
    spend: Optional[float] = None                 # what the plan pays: procurement + tariff + freight + transfers
    units_delivered: Optional[int] = None         # units the plan's purchases bring in
    units_short: Optional[int] = None             # planned purchases that never arrive
    cost_per_unit: Optional[float] = None         # spend / units delivered — comparable across cases that deliver different volumes
    avg_arrival_days: Optional[float] = None      # over the units that do arrive
    shipments: Optional[int] = None               # supplier x route allocations that arrive
    stockout_units: Optional[int] = None          # end-of-horizon demand no warehouse can serve
    below_safety_units: Optional[int] = None      # end-of-horizon units missing against safety stock
    warehouses_below_safety: list[str] = Field(default_factory=list)
    mode_split: dict[str, ModeShare] = Field(default_factory=dict)


class AtRiskShipment(BaseModel):
    supplier_id: str
    route_id: Optional[str]
    transport_mode: Optional[str]
    quantity: int
    value: float
    reason: str


class Exposure(BaseModel):
    shipments_at_risk: int
    units_at_risk: int
    value_at_risk: float
    shipments: list[AtRiskShipment] = Field(default_factory=list)


class Deltas(BaseModel):
    mitigation_cost: Optional[float] = None        # mitigated spend - baseline spend
    mitigation_cost_pct: Optional[float] = None
    avg_arrival_delta_days: Optional[float] = None  # mitigated - baseline
    units_protected: Optional[int] = None          # what mitigation delivers that inaction would not


class TariffChange(BaseModel):
    iso3: str
    baseline_pct: float
    scenario_pct: float


class ScenarioComparison(BaseModel):
    scenario_id: Optional[str] = None
    scenario_label: str
    simulation_id: Optional[str] = None
    product_id: str
    as_of_date: Optional[str] = None
    newly_disrupted_routes: list[str] = Field(default_factory=list)
    newly_disrupted_suppliers: list[str] = Field(default_factory=list)
    tariff_changes: list[TariffChange] = Field(default_factory=list)
    baseline: CaseSummary
    unmitigated: CaseSummary
    mitigated: CaseSummary
    exposure: Optional[Exposure] = None
    deltas: Deltas
    engine: str
    engine_label: str
    model_version: str
    assumptions: list[str]
    warnings: list[str] = Field(default_factory=list)


ASSUMPTIONS = [
    "The baseline is the plan the optimizer would choose in normal conditions for this product — a modeled counterfactual, "
    "not observed shipments (no shipment dataset exists).",
    "'Unmitigated' keeps the baseline plan as it is: purchases from a disrupted supplier, and volume on a disrupted route, "
    "never arrive; what does arrive is re-priced at the tariffs in effect. The inbound volume lost is spread over the warehouses "
    "in proportion to what the baseline plan sends each.",
    "A shortfall is reported in units, not money: the model has no stock-out penalty and none is invented. 'Value at risk' is the "
    "baseline plan's own price for the purchases the disruption invalidates, not a loss estimate.",
    "Single-period model: each warehouse is judged at the end of the planning horizon; stock-outs inside the horizon are not modeled.",
    "Costs are in the MVP cost unit; there is no currency peg.",
]


# --------------------------------------------------------------------------- #
# summarizing
# --------------------------------------------------------------------------- #
def _per_unit(spend: float, units: int) -> Optional[float]:
    return round(spend / units, 2) if units else None


def _ceil(x: float) -> int:
    return int(math.ceil(round(x, 6)))


def _summarize_solved(case: Literal["baseline", "mitigated"], label: str, sol: OptimizationSolution) -> CaseSummary:
    if sol.status != "OPTIMAL":
        return CaseSummary(case=case, label=label, status=sol.status, message=sol.message)
    metrics = optimization_tools.get_optimization_metrics(sol)
    return CaseSummary(
        case=case, label=label, status="OPTIMAL", message=sol.message,
        spend=adapters.plan_spend(sol), units_delivered=sum(a.quantity for a in sol.allocations), units_short=0,
        cost_per_unit=_per_unit(adapters.plan_spend(sol), sum(a.quantity for a in sol.allocations)),
        avg_arrival_days=metrics.get("avg_arrival_days"), shipments=len(sol.allocations),
        stockout_units=0, below_safety_units=0,  # the model's constraints guarantee it: safety stock is a hard floor
        mode_split={m: ModeShare(**v) for m, v in (metrics.get("mode_split") or {}).items()},
    )


def _why_lost(a: Allocation, world: WorldSpec) -> Optional[str]:
    if a.supplier_id in world.disrupted_suppliers:
        return f"supplier {a.supplier_id} is disrupted"
    if a.route_id and a.route_id in world.disrupted_routes:
        return f"route {a.route_id} is disrupted"
    return None


def evaluate_inaction(baseline: OptimizationSolution, world: WorldSpec) -> tuple[CaseSummary, Exposure]:
    """The baseline plan, unchanged, in the disrupted world."""
    problem = baseline.problem
    suppliers = {s.supplier_id: s for s in problem.suppliers}
    rates = world.tariff_rates

    at_risk: list[AtRiskShipment] = []
    planned = delivered = arriving = 0
    spend = arrival_units_days = 0.0
    modes: dict[str, int] = defaultdict(int)
    for a in baseline.allocations:
        planned += a.quantity
        reason = _why_lost(a, world)
        if reason:
            at_risk.append(AtRiskShipment(
                supplier_id=a.supplier_id, route_id=a.route_id, transport_mode=a.transport_mode, quantity=a.quantity,
                value=round(a.quantity * (a.landed_unit_cost + a.freight_unit_cost), 2), reason=reason))
            continue
        supplier = suppliers[a.supplier_id]
        iso3 = sourcing_tools.REGION_TO_ISO3.get(supplier.region)
        rate = rates.get(iso3) if iso3 in rates else supplier.tariff_rate_pct  # a country the world says nothing about keeps the rate the plan was built on
        landed = round(supplier.base_unit_cost * (1 + rate / 100), 4)  # same rule as calculate_landed_cost
        spend += a.quantity * (landed + a.freight_unit_cost)
        delivered += a.quantity
        arriving += 1
        arrival_units_days += a.arrival_days * a.quantity
        modes[a.transport_mode or "direct"] += a.quantity
    spend += baseline.objective_terms.get("transfer", 0.0)  # internal transfers happen inside the network; the disruption does not touch them

    kept = delivered / planned if planned else 1.0
    stockout = below_units = 0.0
    below: list[str] = []
    for w in problem.warehouses:
        inbound = baseline.inbound_by_warehouse.get(w.warehouse_id, 0)
        end = baseline.end_stock_by_warehouse[w.warehouse_id] - inbound * (1 - kept)
        stockout += max(0.0, -end)
        short_of_safety = max(0.0, w.safety_stock - end)
        below_units += short_of_safety
        if short_of_safety > 1e-6:
            below.append(w.warehouse_id)

    summary = CaseSummary(
        case="unmitigated", label="Unmitigated: the baseline plan left as it is", status="EVALUATED",
        message="No re-planning: only what the disruption leaves standing arrives.",
        spend=round(spend, 2), units_delivered=delivered, units_short=planned - delivered, cost_per_unit=_per_unit(spend, delivered),
        avg_arrival_days=round(arrival_units_days / delivered, 2) if delivered else None, shipments=arriving,
        stockout_units=_ceil(stockout), below_safety_units=_ceil(below_units), warehouses_below_safety=sorted(below),
        mode_split={m: ModeShare(units=q, share=round(q / delivered, 4)) for m, q in sorted(modes.items())} if delivered else {},
    )
    exposure = Exposure(
        shipments_at_risk=len(at_risk), units_at_risk=sum(s.quantity for s in at_risk),
        value_at_risk=round(sum(s.value for s in at_risk), 2), shipments=at_risk)
    return summary, exposure


def _deltas(baseline: CaseSummary, unmitigated: CaseSummary, mitigated: CaseSummary) -> Deltas:
    d = Deltas()
    if baseline.spend is not None and mitigated.spend is not None:
        d.mitigation_cost = round(mitigated.spend - baseline.spend, 2)
        d.mitigation_cost_pct = round(100 * d.mitigation_cost / baseline.spend, 1) if baseline.spend else None
    if baseline.avg_arrival_days is not None and mitigated.avg_arrival_days is not None:
        d.avg_arrival_delta_days = round(mitigated.avg_arrival_days - baseline.avg_arrival_days, 2)
    if mitigated.units_delivered is not None and unmitigated.units_delivered is not None:
        d.units_protected = mitigated.units_delivered - unmitigated.units_delivered
    return d


def _not_available(case: Literal["unmitigated", "mitigated"], label: str, why: str) -> CaseSummary:
    return CaseSummary(case=case, label=label, status="NOT_AVAILABLE", message=why)


# --------------------------------------------------------------------------- #
# the two entry points
# --------------------------------------------------------------------------- #
def _compare(
    *, scenario_id: Optional[str], scenario_label: str, simulation_id: Optional[str], product_id: str, as_of_date: Optional[str],
    world: WorldSpec, mitigated: Solved | None, no_mitigation_reason: str | None, engine: OptimizationEngine | None,
) -> ScenarioComparison:
    """`mitigated` is the plan to compare with, or None to solve the disrupted world afresh;
    `no_mitigation_reason` says there is no mitigated plan at all (a simulation that never produced one)."""
    base_world = baseline_world()
    baseline = solve_world(product_id, as_of_date, base_world, engine)

    baseline_case = _summarize_solved("baseline", "Baseline: normal operations", baseline.solution)
    mitigated_label = "Mitigated: the optimizer's plan for the disrupted network"
    if no_mitigation_reason:
        mitigated_case = _not_available("mitigated", mitigated_label, no_mitigation_reason)
    else:
        mitigated_case = _summarize_solved("mitigated", mitigated_label, (mitigated or solve_world(product_id, as_of_date, world, engine)).solution)
    if baseline.solution.status == "OPTIMAL":
        unmitigated_case, exposure = evaluate_inaction(baseline.solution, world)
    else:
        unmitigated_case = _not_available("unmitigated", "Unmitigated: the baseline plan left as it is", "there is no feasible baseline plan to leave in place")
        exposure = None

    base_tariffs, new_tariffs = base_world.tariff_rates, world.tariff_rates
    changes = [TariffChange(iso3=k, baseline_pct=base_tariffs.get(k, 0.0), scenario_pct=v) for k, v in sorted(new_tariffs.items()) if abs(v - base_tariffs.get(k, 0.0)) > 1e-9]
    return ScenarioComparison(
        scenario_id=scenario_id, scenario_label=scenario_label, simulation_id=simulation_id, product_id=str(product_id), as_of_date=as_of_date,
        newly_disrupted_routes=sorted(world.disrupted_routes - base_world.disrupted_routes),
        newly_disrupted_suppliers=sorted(world.disrupted_suppliers - base_world.disrupted_suppliers),
        tariff_changes=changes, baseline=baseline_case, unmitigated=unmitigated_case, mitigated=mitigated_case, exposure=exposure,
        deltas=_deltas(baseline_case, unmitigated_case, mitigated_case),
        engine=baseline.solution.engine, engine_label=baseline.solution.label, model_version=baseline.model_version,
        assumptions=list(ASSUMPTIONS),
    )


def compare_scenario(scenario: ScenarioDefinition, product_id: Optional[str] = None, as_of_date: Optional[str] = None,
                     engine: OptimizationEngine | None = None) -> ScenarioComparison:
    """Baseline vs. unmitigated vs. a fresh mitigated solve, for a defined scenario. Refuses a scenario
    the pipeline cannot represent (ScenarioNotModeledError) rather than guess."""
    event = scenario.disruption_event()
    tariffs = scenario.tariff_overrides(load_baseline().tariffs)
    return _compare(
        scenario_id=scenario.scenario_id, scenario_label=scenario.label, simulation_id=None, product_id=str(product_id or scenario.default_product_id),
        as_of_date=as_of_date, world=world_after_event(event, tariffs), mitigated=None, no_mitigation_reason=None, engine=engine)


def compare_state(state: WorldState, product_id: Optional[str] = None, as_of_date: Optional[str] = None,
                  engine: OptimizationEngine | None = None) -> ScenarioComparison:
    """The same three-way comparison for a simulation's own disruption. The mitigated case is the
    simulation's actual plan (replans and all), not a re-solve. The product comes from the plan."""
    plan = state.current_plan
    product = str(product_id or (plan.problem.product_id if plan else ""))
    if not product:
        raise ValueError(f"simulation {state.simulation_id!r} has no plan yet, so there is no product to compare; pass product_id")
    world = world_of_state(state)
    label = state.current_disruptions[-1].event_type.replace("_", " ").title() if state.current_disruptions else state.scenario_type
    result = _compare(
        scenario_id=None, scenario_label=label, simulation_id=state.simulation_id, product_id=product, as_of_date=as_of_date, world=world,
        mitigated=Solved(plan, "") if plan is not None else None,
        no_mitigation_reason=None if plan is not None else "this simulation has produced no plan", engine=engine)
    if plan is not None:
        # the simulation may have been run on an earlier as-of date than the ledger's latest, which this comparison uses
        base_stock = {w.warehouse_id: w.current_stock for w in solve_world(product, as_of_date, baseline_world(), engine).solution.problem.warehouses}
        sim_stock = {w.warehouse_id: w.current_stock for w in plan.problem.warehouses}
        if base_stock != sim_stock:
            result.warnings.append(
                "The simulation's inventory differs from the ledger's latest snapshot (it was run on an earlier as-of date), so the baseline here "
                "is computed on different stock than the simulation's plan. Compare the two with care.")
    return result
