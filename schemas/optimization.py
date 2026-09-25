"""Optimization Engine schemas — the problem the engine solves and the solution
it returns (architecture.md §4, agent-plan.md § Optimization Engine).

An OptimizationProblem is assembled from the Inventory, Logistics and Sourcing
agents' recommendations (backend/optimization/tools.py); an OptimizationSolution
carries the problem it answers, so validate_solution() can re-check a plan
against the constraints it claims to satisfy with nothing else in hand.

Costs are in the MVP cost unit every processed table uses (suppliers.csv
unit_cost, routes.csv cost_per_unit) — no real currency peg.
"""
from __future__ import annotations

import math
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

ENGINE_NAME = "prototype"
ENGINE_LABEL = "Prototype Optimization"  # rendered verbatim by the frontend (api-plan.md)


class OptimizationParameters(BaseModel):
    planning_horizon_days: int = Field(gt=0)
    # Hard deadline: a supplier->route lane whose lead time + transit exceeds
    # this is not offered to the solver. Defaults to the planning horizon —
    # stock that arrives after the window the forecast covers can't help it.
    max_delivery_days: Optional[float] = Field(default=None, gt=0)
    # Optional soft deadline: arrivals later than this are charged
    # delay_penalty_per_unit_day per unit per day late. Off unless both are set.
    target_delivery_days: Optional[float] = Field(default=None, ge=0)
    delay_penalty_per_unit_day: float = Field(default=0.0, ge=0)
    internal_transfer_cost_per_unit: float = Field(ge=0)
    internal_transfer_days: float = Field(ge=0)

    @model_validator(mode="after")
    def _fill_and_check(self) -> "OptimizationParameters":
        if self.max_delivery_days is None:
            self.max_delivery_days = float(self.planning_horizon_days)
        if self.delay_penalty_per_unit_day > 0 and self.target_delivery_days is None:
            raise ValueError("delay_penalty_per_unit_day > 0 requires target_delivery_days")
        if self.target_delivery_days is not None and self.target_delivery_days > self.max_delivery_days:
            raise ValueError("target_delivery_days cannot exceed max_delivery_days")
        return self


class WarehouseState(BaseModel):
    warehouse_id: str
    current_stock: int = Field(ge=0)
    safety_stock: int = Field(ge=0)
    forecast_demand: float = Field(ge=0)  # over planning_horizon_days
    stockout_risk: str

    @property
    def demand_units(self) -> int:
        """Forecasts are fractional; the plan covers whole units, rounded up."""
        return math.ceil(self.forecast_demand)

    @property
    def projected_surplus(self) -> int:
        """Stock left above safety after this warehouse's own forecast demand.
        Negative = the deficit it needs inbound or transferred stock to cover."""
        return self.current_stock - self.demand_units - self.safety_stock


class SupplierOption(BaseModel):
    supplier_id: str
    supplier_name: str
    region: str
    # Port whose lanes carry this supplier's goods to the hub; None = no freight
    # leg is modeled (supplier at/near the hub) — see optimization_config.yaml.
    origin_port: Optional[str] = None
    capacity: int = Field(ge=0)
    base_unit_cost: float = Field(gt=0)
    landed_unit_cost: float = Field(gt=0)  # base cost + tariff (Sourcing Agent's calculate_landed_cost)
    tariff_rate_pct: float = Field(ge=0)
    lead_time_days: float = Field(ge=0)
    reliability: float = Field(ge=0, le=1)
    risk_level: str


class RouteOption(BaseModel):
    route_id: str
    origin: str
    destination: str
    transport_mode: str
    capacity: int = Field(ge=0)
    cost_per_unit: float = Field(ge=0)
    transit_time_days: float = Field(gt=0)


class ExcludedOption(BaseModel):
    """Something the solver was never offered, and why — so the explanation can
    say "supplier S003 unusable: no available route" instead of just omitting it."""

    kind: Literal["supplier", "lane", "route"]
    id: str
    reason: str


class AgentRecommendations(BaseModel):
    """What the three agents independently recommended, kept so the solution
    can report where the joint optimum deviates from them and why."""

    inventory_transfers: list[dict[str, Any]] = Field(default_factory=list)  # {"from","to","quantity"}
    sourcing_allocations: list[dict[str, Any]] = Field(default_factory=list)  # {"supplier_id","quantity"}
    logistics_route_ids: list[str] = Field(default_factory=list)


class OptimizationProblem(BaseModel):
    product_id: str
    parameters: OptimizationParameters
    warehouses: list[WarehouseState] = Field(min_length=1)
    suppliers: list[SupplierOption] = Field(default_factory=list)
    routes: list[RouteOption] = Field(default_factory=list)
    disrupted_route_ids: list[str] = Field(default_factory=list)
    excluded_options: list[ExcludedOption] = Field(default_factory=list)
    agent_recommendations: AgentRecommendations = Field(default_factory=AgentRecommendations)
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ids_unique(self) -> "OptimizationProblem":
        for label, ids in (
            ("warehouse_id", [w.warehouse_id for w in self.warehouses]),
            ("supplier_id", [s.supplier_id for s in self.suppliers]),
            ("route_id", [r.route_id for r in self.routes]),
        ):
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {label} in problem")
        return self


class Allocation(BaseModel):
    """`quantity` units bought from `supplier_id` and shipped on `route_id`
    (None = no freight leg modeled for this supplier)."""

    supplier_id: str
    route_id: Optional[str]
    transport_mode: Optional[str]
    quantity: int = Field(gt=0)
    landed_unit_cost: float
    freight_unit_cost: float
    arrival_days: float


class Transfer(BaseModel):
    from_warehouse: str
    to_warehouse: str
    quantity: int = Field(gt=0)
    unit_cost: float


class ConstraintStatus(BaseModel):
    name: str
    category: Literal[
        "supplier_capacity", "route_capacity", "safety_stock", "transfer_availability",
        "delivery_deadline", "flow_balance", "lane_validity", "integrality",
    ]
    sense: Literal["<=", ">=", "=="]
    value: float
    bound: float
    satisfied: bool
    binding: bool  # satisfied with no slack left — the constraints that shaped the plan
    detail: str


class ValidationResult(BaseModel):
    valid: bool
    violations: list[str] = Field(default_factory=list)
    checks_run: int = 0


class OptimizationSolution(BaseModel):
    status: Literal["OPTIMAL", "INFEASIBLE", "ERROR"]
    engine: str = ENGINE_NAME
    label: str = ENGINE_LABEL
    message: str = ""
    problem: OptimizationProblem
    objective_value: Optional[float] = None
    objective_terms: dict[str, float] = Field(default_factory=dict)
    allocations: list[Allocation] = Field(default_factory=list)
    transfers: list[Transfer] = Field(default_factory=list)
    inbound_by_warehouse: dict[str, int] = Field(default_factory=dict)
    end_stock_by_warehouse: dict[str, int] = Field(default_factory=dict)
    constraint_status: list[ConstraintStatus] = Field(default_factory=list)
    excluded_options: list[ExcludedOption] = Field(default_factory=list)
    deviations: list[dict[str, Any]] = Field(default_factory=list)
    decision_factors: list[dict[str, str]] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)  # populated when INFEASIBLE
    solver: dict[str, Any] = Field(default_factory=dict)
