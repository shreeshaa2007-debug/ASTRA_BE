"""Sensing Agent schemas — agent-plan.md § Sensing Agent, brief §13.

The flow is RawSignal -> Classification (the LLM's *candidate*, unvalidated)
-> ValidationOutcome (plain Python) -> SensingResult. Only a SensingResult
with status EVENT carries a DisruptionEvent, and only validate_event() can
produce one: nothing the LLM returns is ever an event on its own.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from backend.schemas.world_state import DisruptionEvent


class RawSignal(BaseModel):
    """What came in: free text, or a simulated trigger from the scenario
    framework. A simulated trigger may carry a pre-structured `candidate`, which
    skips the LLM call — never validate_event()."""

    source: Literal["free_text", "simulated"]
    text: str
    scenario_type: Optional[str] = None
    candidate: Optional[dict[str, Any]] = None
    received_at: datetime


class RouteInfo(BaseModel):
    route_id: str
    origin: str
    destination: str
    transport_mode: str
    corridor: Optional[str] = None  # last segment of the id (SUEZ/CAPE/RAIL/AIR) — routes.csv's naming convention


class SupplierInfo(BaseModel):
    supplier_id: str
    supplier_name: str
    region: str


class SensingCatalog(BaseModel):
    """The ids that exist in this network. Shown to the LLM so it can ground
    its answer, and used by validate_event() to reject anything outside it."""

    routes: list[RouteInfo]
    suppliers: list[SupplierInfo]
    product_ids: list[str]

    @property
    def route_ids(self) -> set[str]:
        return {r.route_id for r in self.routes}

    @property
    def supplier_ids(self) -> set[str]:
        return {s.supplier_id for s in self.suppliers}


class Classification(BaseModel):
    """classify_event()'s output: the parsed candidate object (or None if the
    LLM's output wasn't a JSON object) plus how it was obtained."""

    candidate: Optional[dict[str, Any]]
    raw_text: str = ""
    parse_error: Optional[str] = None
    model: str
    latency_ms: float = 0.0
    attempts: int = 0


class ValidationOutcome(BaseModel):
    valid: bool
    event: Optional[DisruptionEvent] = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SensingResult(BaseModel):
    status: Literal["EVENT", "NO_DISRUPTION", "REJECTED", "ERROR"]
    event: Optional[DisruptionEvent] = None
    error_code: Optional[str] = None  # api-plan.md: INVALID_SENSING_OUTPUT / LLM_UNAVAILABLE / VALIDATION_ERROR
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rationale: Optional[str] = None
    signal: Optional[RawSignal] = None
    candidate: Optional[dict[str, Any]] = None  # what the LLM proposed, kept for the audit trail
    llm: dict[str, Any] = Field(default_factory=dict)
