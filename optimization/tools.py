"""Optimization Engine tool functions — per agent-plan.md: build_optimization_problem(),
optimize_supply_chain(), validate_solution(), get_optimization_metrics().

build_optimization_problem() is pure: it turns the three agents' outputs (the
exact shapes their agent.py files return) into an OptimizationProblem, touching
no data files, so it is testable with hand-made dicts. build_problem_from_agents()
is a convenience that runs those agents in order; the orchestrator (Phase 14)
calls gather_agent_outputs() directly, because it must commit what the
Inventory Agent found to the world state before the optimizer runs.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from backend.optimization.engine import OptimizationEngine, PrototypeOptimizationEngine
from backend.schemas.optimization import (
    AgentRecommendations,
    ExcludedOption,
    OptimizationParameters,
    OptimizationProblem,
    OptimizationSolution,
    RouteOption,
    SupplierOption,
    ValidationResult,
    WarehouseState,
)

CONFIG_PATH = Path("backend/config/optimization_config.yaml")


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_optimization_problem(
    product_id: str,
    inventory_results: list[dict],
    sourcing_result: dict,
    logistics_results: list[dict],
    safety_stock_by_warehouse: dict[str, int],
    parameters: OptimizationParameters,
    config: dict | None = None,
) -> OptimizationProblem:
    """`inventory_results`: InventoryAgent.analyze_product() output (one record
    per warehouse). `sourcing_result`: SourcingAgent.generate_sourcing_mix()
    output — its `candidates_considered` become the supplier options and its
    greedy `supplier_allocations` are kept only to report deviations.
    `logistics_results`: one LogisticsAgent.plan_shipment() output per origin
    port -> hub lane. The Inventory Agent's records carry no safety stock, so
    the caller supplies it (from the Inventory Agent's get_inventory tool).
    """
    config = config or load_config()
    network = config["network"]
    hub, port_of = network["inbound_hub"], network["supplier_origin_port"]
    product_id = str(product_id)

    if not inventory_results:
        raise ValueError("inventory_results is empty: at least one warehouse is required")
    if any(str(r["product"]) != product_id for r in inventory_results):
        raise ValueError(f"inventory_results contain products other than {product_id!r}")
    if str(sourcing_result["product_id"]) != product_id:
        raise ValueError(f"sourcing_result is for product {sourcing_result['product_id']!r}, not {product_id!r}")
    missing = [r["warehouse"] for r in inventory_results if r["warehouse"] not in safety_stock_by_warehouse]
    if missing:
        raise ValueError(f"no safety stock supplied for warehouse(s): {', '.join(missing)}")

    warehouses = [
        WarehouseState(
            warehouse_id=r["warehouse"], current_stock=r["current_stock"], safety_stock=safety_stock_by_warehouse[r["warehouse"]],
            forecast_demand=r["forecast_demand"], stockout_risk=r["stockout_risk"],
        )
        for r in inventory_results
    ]

    excluded: list[ExcludedOption] = []
    suppliers: list[SupplierOption] = []
    for c in sourcing_result["candidates_considered"]:
        sid = c["supplier_id"]
        if sid not in port_of:
            excluded.append(ExcludedOption(kind="supplier", id=sid, reason="no origin-port mapping in optimization_config.yaml"))
            continue
        suppliers.append(SupplierOption(
            supplier_id=sid, supplier_name=c["supplier_name"], region=c["region"], origin_port=port_of[sid],
            capacity=c["capacity"], base_unit_cost=c["unit_cost"], landed_unit_cost=c["landed_unit_cost"],
            tariff_rate_pct=c["tariff_rate_pct"] or 0.0, lead_time_days=c["lead_time_days"],
            reliability=c["reliability"], risk_level=c["risk_level"],
        ))

    routes: dict[str, RouteOption] = {}
    disrupted: set[str] = set()
    for lane in logistics_results:
        disrupted.update(lane["disrupted_route_ids"])
        if lane["destination"] != hub:
            excluded.append(ExcludedOption(kind="route", id=f"{lane['origin']}->{lane['destination']}", reason=f"does not end at the inbound hub {hub}"))
            continue
        for alt in lane["alternative_routes"]:
            routes[alt["route_id"]] = RouteOption(
                route_id=alt["route_id"], origin=lane["origin"], destination=lane["destination"],
                transport_mode=alt["transport_mode"], capacity=alt["capacity"], cost_per_unit=alt["cost_per_unit"],
                transit_time_days=alt["transit_time_days"],
            )

    transfers = [
        {"from": rec["recommended_transfer"]["from"], "to": rec["warehouse"], "quantity": rec["recommended_transfer"]["quantity"]}
        for rec in inventory_results if rec.get("recommended_transfer")
    ]
    recommendations = AgentRecommendations(
        inventory_transfers=transfers,
        sourcing_allocations=list(sourcing_result["supplier_allocations"]),
        logistics_route_ids=[l["recommended_route_id"] for l in logistics_results if l.get("recommended_route_id")],
    )

    freight_free = sorted(s.supplier_id for s in suppliers if s.origin_port is None)
    assumptions = [
        f"Single-period model: each warehouse must end the {parameters.planning_horizon_days}-day horizon at or above safety stock after its "
        "forecast demand; stock-outs inside the horizon are not modeled.",
        f"Inbound freight lands at {hub} and is available to every warehouse; onward distribution is not modeled (no data).",
        f"Inter-warehouse transfers cost {parameters.internal_transfer_cost_per_unit}/unit and take {parameters.internal_transfer_days} days — placeholders, "
        "no inter-warehouse freight data exists.",
        "Supplier reliability and risk level are reported but do not enter the objective; a DISRUPTED supplier is simply unavailable.",
    ]
    if freight_free:
        assumptions.append(
            f"No freight leg is modeled for {', '.join(freight_free)} (freight cost 0, transit 0 — no route on file), which flatters them against suppliers that pay freight."
        )

    return OptimizationProblem(
        product_id=product_id, parameters=parameters, warehouses=warehouses, suppliers=suppliers,
        routes=list(routes.values()), disrupted_route_ids=sorted(disrupted), excluded_options=excluded,
        agent_recommendations=recommendations, assumptions=assumptions,
    )


@dataclass
class AgentOutputs:
    """What the Inventory, Sourcing and Logistics agents returned for one
    product — kept whole so the orchestrator can commit the inventory side to
    the world state and still hand the same outputs to build_optimization_problem()."""

    parameters: OptimizationParameters
    inventory_results: list[dict]
    safety_stock_by_warehouse: dict[str, int]
    sourcing_result: dict
    logistics_results: list[dict]
    gross_deficit: int
    forecast_model_version: str


def gather_agent_outputs(
    product_id: str,
    as_of_date: str | None = None,
    disrupted_route_ids: frozenset[str] = frozenset(),
    excluded_supplier_ids: frozenset[str] = frozenset(),
    tariff_rates: dict[str, float] | None = None,
    parameter_overrides: dict | None = None,
    config: dict | None = None,
) -> AgentOutputs:
    """Runs the Inventory, Sourcing and Logistics agents for one product, in the
    order their data dependencies force: Sourcing is sized to Inventory's gross
    deficit (before transfers — the engine, not the agents, decides how much
    transfers cover), and Logistics is asked about the origin ports Sourcing's
    candidate suppliers ship from.

    `disrupted_route_ids`, `excluded_supplier_ids` and `tariff_rates` are what the
    shared world state (or a replan) says is true; omitted, the agents fall back
    to the processed data files.
    """
    from backend.agents.inventory import tools as inventory_tools
    from backend.agents.inventory.agent import InventoryAgent
    from backend.agents.logistics.agent import LogisticsAgent
    from backend.agents.sourcing.agent import SourcingAgent

    config = config or load_config()
    params = OptimizationParameters(**{**config["defaults"], **(parameter_overrides or {})})

    inventory_agent = InventoryAgent(horizon_days=params.planning_horizon_days)
    inventory_results = inventory_agent.analyze_product(product_id, as_of_date)
    safety = {r["warehouse"]: inventory_tools.get_inventory(r["warehouse"], product_id, as_of_date)["safety_stock"] for r in inventory_results}
    gross_deficit = sum(max(0, math.ceil(r["forecast_demand"]) + safety[r["warehouse"]] - r["current_stock"]) for r in inventory_results)

    sourcing_result = SourcingAgent().generate_sourcing_mix(product_id, gross_deficit, frozenset(excluded_supplier_ids), tariff_rates)

    port_of = config["network"]["supplier_origin_port"]
    ports = sorted({port_of.get(c["supplier_id"]) for c in sourcing_result["candidates_considered"]} - {None})
    logistics_results = []
    for port in ports:
        try:
            logistics_results.append(LogisticsAgent().plan_shipment(
                port, config["network"]["inbound_hub"], max(gross_deficit, 1), frozenset(disrupted_route_ids), as_of_date))
        except ValueError:
            pass  # no routes on file for this lane: its suppliers surface as "no available route" in the solution

    return AgentOutputs(params, inventory_results, safety, sourcing_result, logistics_results, gross_deficit, inventory_agent.predictor_version)


def build_problem_from_agents(
    product_id: str,
    as_of_date: str | None = None,
    disrupted_route_ids: frozenset[str] = frozenset(),
    parameter_overrides: dict | None = None,
    config: dict | None = None,
    excluded_supplier_ids: frozenset[str] = frozenset(),
    tariff_rates: dict[str, float] | None = None,
) -> OptimizationProblem:
    """Runs the three agents (gather_agent_outputs) and builds the joint problem."""
    config = config or load_config()
    out = gather_agent_outputs(product_id, as_of_date, disrupted_route_ids, excluded_supplier_ids, tariff_rates, parameter_overrides, config)
    return build_optimization_problem(
        product_id, out.inventory_results, out.sourcing_result, out.logistics_results, out.safety_stock_by_warehouse, out.parameters, config)


def optimize_supply_chain(problem: OptimizationProblem, engine: OptimizationEngine | None = None) -> OptimizationSolution:
    return (engine or PrototypeOptimizationEngine()).optimize_supply_chain(problem)


def validate_solution(solution: OptimizationSolution, engine: OptimizationEngine | None = None) -> ValidationResult:
    return (engine or PrototypeOptimizationEngine()).validate_solution(solution)


def get_optimization_metrics(solution: OptimizationSolution) -> dict:
    """Headline numbers for the Decision Center. Only meaningful for an
    OPTIMAL solution; anything else returns its status and message alone."""
    base = {"status": solution.status, "engine": solution.engine, "label": solution.label, "message": solution.message}
    if solution.status != "OPTIMAL":
        return {**base, "diagnostics": solution.diagnostics}

    problem = solution.problem
    suppliers = {s.supplier_id: s for s in problem.suppliers}
    procured = sum(a.quantity for a in solution.allocations)
    moved = sum(t.quantity for t in solution.transfers)

    by_mode: dict[str, int] = defaultdict(int)
    by_supplier: dict[str, int] = defaultdict(int)
    for a in solution.allocations:
        by_mode[a.transport_mode or "direct"] += a.quantity
        by_supplier[a.supplier_id] += a.quantity

    def share(counts: dict[str, int]) -> dict[str, dict]:
        return {k: {"units": q, "share": round(q / procured, 4)} for k, q in sorted(counts.items())}

    procurement_cost = sum(v for k, v in solution.objective_terms.items() if k != "transfer")
    return {
        **base,
        "objective_value": solution.objective_value,
        "objective_terms": solution.objective_terms,
        "units_procured": procured,
        "units_transferred": moved,
        "avg_all_in_unit_cost": round(procurement_cost / procured, 4) if procured else None,
        "avg_arrival_days": round(sum(a.arrival_days * a.quantity for a in solution.allocations) / procured, 2) if procured else None,
        "weighted_reliability": round(sum(suppliers[a.supplier_id].reliability * a.quantity for a in solution.allocations) / procured, 4) if procured else None,
        "mode_split": share(by_mode),
        "supplier_split": share(by_supplier),
        "constraints_total": len(solution.constraint_status),
        "constraints_satisfied": sum(c.satisfied for c in solution.constraint_status),
        "binding_constraints": [c.name for c in solution.constraint_status if c.binding],
        "deviation_count": len(solution.deviations),
        "solver": solution.solver,
    }
