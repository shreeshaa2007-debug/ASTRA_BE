"""Shared World State schemas — architecture.md §5, brief §14.

One WorldState per simulation_id. Every agent reads it; only the orchestrator
writes it, and only through named checkpoints (backend/services/world_state).
Field names follow §14 exactly; the extras (status, version, replan_count,
approval_decision, error) are what the checkpoint machinery needs to enforce
the human-approval gate and the bounded replan.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from backend.schemas.entities import RiskLevel, RouteStatus, ShipmentStatus, SupplierStatus
from backend.schemas.optimization import OptimizationSolution


class SimulationStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    COMPLETED = "COMPLETED"    # plan finalized (auto-approved, or approved by a human)
    REJECTED = "REJECTED"      # a human rejected the escalated plan
    FAILED = "FAILED"          # run could not produce a final plan


TERMINAL_STATUSES = frozenset({SimulationStatus.COMPLETED, SimulationStatus.REJECTED, SimulationStatus.FAILED})


class ApprovalStatus(str, Enum):
    NOT_EVALUATED = "NOT_EVALUATED"   # compliance has not run yet
    NOT_REQUIRED = "NOT_REQUIRED"     # low-impact, in-policy: proceeds automatically
    PENDING = "PENDING"               # escalated, waiting on a human
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class DisruptionEvent(BaseModel):
    """A validated disruption — the Sensing Agent's output (brief §13). The LLM
    only ever proposes one of these; it becomes part of the state only via the
    `event_sensed` checkpoint, after validate_event() has passed."""

    event_id: str
    event_type: str
    location: str
    severity: RiskLevel
    start_date: datetime
    estimated_duration: float = Field(ge=0)  # days
    affected_routes: list[str] = Field(default_factory=list)
    affected_suppliers: list[str] = Field(default_factory=list)
    affected_products: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    summary: Optional[str] = Field(default=None, max_length=300)  # one-line rationale from sensing, for the UI; not a decision input


class InventoryAssessment(BaseModel):
    """One Inventory Agent record (brief §8 shape) as committed to the state."""

    warehouse: str
    product: str
    forecast_demand: float
    current_stock: int
    stockout_risk: str
    recommended_transfer: Optional[dict[str, Any]] = None  # {"from": ..., "quantity": ...}


class ForecastRecord(BaseModel):
    product_id: str
    warehouse_id: str
    horizon_days: int = Field(gt=0)
    forecast_demand: float = Field(ge=0)
    model_version: str


class ComplianceStatus(BaseModel):
    """The Compliance Agent's validate_plan() output (brief §12 shape)."""

    status: Literal["APPROVED", "REJECTED", "ESCALATED"]
    checks: list[dict[str, Any]] = Field(default_factory=list)
    reason: str
    requires_human: bool


class ApprovalDecision(BaseModel):
    decided_by: str = Field(min_length=1)
    decided_at: datetime
    note: str = ""


class WorldState(BaseModel):
    simulation_id: str
    scenario_type: str
    status: SimulationStatus = SimulationStatus.CREATED
    version: int = 0
    created_at: datetime
    timestamp: datetime  # last write (brief §14's `timestamp`)

    current_disruptions: list[DisruptionEvent] = Field(default_factory=list)
    route_status: dict[str, RouteStatus] = Field(default_factory=dict)
    supplier_status: dict[str, SupplierStatus] = Field(default_factory=dict)
    inventory_status: list[InventoryAssessment] = Field(default_factory=list)
    demand_forecasts: list[ForecastRecord] = Field(default_factory=list)
    shipment_status: dict[str, ShipmentStatus] = Field(default_factory=dict)  # keyed by shipment_id; no shipment dataset yet
    tariffs: dict[str, float] = Field(default_factory=dict)  # ISO3 -> tariff rate %, in effect for this simulation
    current_plan: Optional[OptimizationSolution] = None
    compliance_status: Optional[ComplianceStatus] = None
    approval_status: ApprovalStatus = ApprovalStatus.NOT_EVALUATED
    approval_decision: Optional[ApprovalDecision] = None
    replan_count: int = Field(default=0, ge=0)
    error: Optional[str] = None

    def disrupted_route_ids(self) -> frozenset[str]:
        """What the Logistics Agent and the optimizer take as `disrupted_route_ids`."""
        return frozenset(r for r, s in self.route_status.items() if s == RouteStatus.DISRUPTED)

    def disrupted_supplier_ids(self) -> frozenset[str]:
        """What the Sourcing Agent takes as `excluded_supplier_ids`."""
        return frozenset(s for s, st in self.supplier_status.items() if st == SupplierStatus.DISRUPTED)


class SimulationSummary(BaseModel):
    """Index-column view of a simulation — listing never deserializes the state."""

    simulation_id: str
    scenario_type: str
    status: SimulationStatus
    version: int
    created_at: datetime
    updated_at: datetime


class CheckpointRecord(BaseModel):
    """One entry in a simulation's append-only audit trail; also what the
    Agent Monitor timeline is built from."""

    simulation_id: str
    version: int
    checkpoint: str
    actor: str
    at: datetime
    changed_fields: list[str] = Field(default_factory=list)
