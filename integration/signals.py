"""Inbound: a disruption signal delivered by an integration flow (SAP Integration Suite, or anything that can POST JSON).

Integration platforms deliver at-least-once — a flow that times out retries — so accepting a signal has to be
idempotent. The simulation id is derived from what the source says identifies the signal, so a redelivery names the
same simulation and is recognised rather than run twice; no lookup table, and two racing deliveries of one signal
resolve in the database (one insert wins).
"""
from __future__ import annotations

import hashlib
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.api.models import checked_text

ID_PATTERN = r"^[A-Za-z0-9._:/-]{1,128}$"
SOURCE_PATTERN = r"^[A-Za-z0-9._-]{1,64}$"
SCENARIO_TYPE = "INTEGRATION"


class IntegrationSignal(BaseModel):
    """A disruption reported by another system.

    `report` is the human-readable description (a plant-maintenance notification's text, a logistics provider's
    alert). Without `candidate` it goes through the Sensing Agent, an LLM that only proposes. With a `candidate` the
    source has already structured the event and the LLM is skipped — the candidate is still validated against the
    network's real routes, suppliers and products, and can be refused."""

    model_config = ConfigDict(str_strip_whitespace=True)

    source_system: str = Field(pattern=SOURCE_PATTERN, description="who is reporting, e.g. s4hana-prod")
    external_id: str = Field(pattern=ID_PATTERN, description="the source's own id for this signal; the idempotency key")
    product_id: str = Field(min_length=1, max_length=32, description="the product to plan; one product per run")
    report: str
    candidate: Optional[dict[str, Any]] = Field(default=None, description="an already-structured event; skips the LLM, not validation")
    tariff_overrides: Optional[dict[str, float]] = Field(default=None, description="ISO3 -> tariff %, merged over the baseline")

    @field_validator("report")
    @classmethod
    def _report(cls, value: str) -> str:
        return checked_text(value)

    @property
    def simulation_id(self) -> str:
        return signal_simulation_id(self.source_system, self.external_id)

    def trigger(self) -> str | dict:
        """What the orchestrator is handed as its signal."""
        if self.candidate is None:
            return self.report
        return {"description": self.report, "candidate": self.candidate, "scenario_type": SCENARIO_TYPE}


def signal_simulation_id(source_system: str, external_id: str) -> str:
    """`sig-` + 16 hex digits of a hash of (source, id): stable, fits the 64-character id limit, and does not put an
    external system's identifier straight into a URL path."""
    digest = hashlib.sha256(f"{source_system}\x1f{external_id}".encode()).hexdigest()[:16]
    return f"sig-{digest}"
