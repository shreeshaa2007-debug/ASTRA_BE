"""Scenario definitions and how one becomes a change to the world.

A scenario is data (backend/config/scenarios.yaml). This module loads it,
checks it against the real network, and turns it into the same things the live
pipeline works with: a DisruptionEvent, a simulated trigger for the Sensing
Agent, and tariff overrides. Nothing here decides what a disruption *does* —
that is EVENT_EFFECTS in backend/services/world_state/events.py, so a scenario
and a sensed report of the same event cannot drift apart.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from backend.schemas.entities import RiskLevel
from backend.schemas.world_state import DisruptionEvent
from backend.services.world_state.events import EVENT_EFFECTS

SCENARIOS_PATH = Path("backend/config/scenarios.yaml")


class ScenarioEvent(BaseModel):
    event_type: str
    location: str
    severity: RiskLevel
    estimated_duration: float = Field(ge=0)
    affected_routes: list[str] = Field(default_factory=list)
    affected_suppliers: list[str] = Field(default_factory=list)
    affected_products: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0, le=1)
    summary: Optional[str] = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def _known_type(self) -> "ScenarioEvent":
        if self.event_type not in EVENT_EFFECTS:
            raise ValueError(f"event_type {self.event_type!r} has no defined effect; known: {sorted(EVENT_EFFECTS)}")
        return self


class ScenarioDefinition(BaseModel):
    scenario_id: str
    label: str
    description: str
    modeled: bool
    not_modeled_reason: Optional[str] = None
    default_product_id: str
    event: Optional[ScenarioEvent] = None
    tariff_add_pct: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> "ScenarioDefinition":
        if self.modeled and self.event is None:
            raise ValueError(f"{self.scenario_id}: a modeled scenario needs an event")
        if not self.modeled and not self.not_modeled_reason:
            raise ValueError(f"{self.scenario_id}: a scenario that is not modeled must say why (not_modeled_reason)")
        return self

    # ---- what the scenario becomes -------------------------------------------------
    def disruption_event(self, now: datetime | None = None) -> DisruptionEvent:
        self._require_modeled()
        return DisruptionEvent(
            event_id=f"scenario-{self.scenario_id.lower()}", start_date=now or datetime.now(timezone.utc), **self.event.model_dump(),
        )

    def tariff_overrides(self, baseline_tariffs: Mapping[str, float]) -> dict[str, float]:
        """Absolute ISO3 -> % rates: the baseline rate plus this scenario's added points."""
        unknown = sorted(set(self.tariff_add_pct) - set(baseline_tariffs))
        if unknown:
            raise ValueError(f"{self.scenario_id}: tariff_add_pct names countries with no baseline tariff: {unknown}")
        return {iso3: round(baseline_tariffs[iso3] + add, 4) for iso3, add in self.tariff_add_pct.items()}

    def trigger(self, now: datetime | None = None) -> dict:
        """A simulated trigger for POST /api/simulations/{id}/run: the Sensing Agent skips the LLM
        for a pre-structured candidate, but still validates it like anything else."""
        event = self.disruption_event(now)
        candidate = event.model_dump(mode="json", exclude={"event_id"})
        candidate["start_date"] = event.start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        return {"scenario_type": self.scenario_id, "description": self.description, "candidate": candidate}

    def _require_modeled(self) -> None:
        if not self.modeled or self.event is None:
            raise ScenarioNotModeledError(self.scenario_id, self.not_modeled_reason or "")


class ScenarioNotModeledError(Exception):
    error_code = "SCENARIO_NOT_MODELED"

    def __init__(self, scenario_id: str, reason: str):
        super().__init__(f"scenario {scenario_id!r} is not modeled: {reason}")
        self.scenario_id, self.reason = scenario_id, reason


class UnknownScenarioError(Exception):
    error_code = "SCENARIO_NOT_FOUND"


def load_scenarios(path: Path = SCENARIOS_PATH) -> dict[str, ScenarioDefinition]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)["scenarios"]
    return {sid: ScenarioDefinition(scenario_id=sid, **body) for sid, body in raw.items()}


def get_scenario(scenario_id: str, path: Path = SCENARIOS_PATH) -> ScenarioDefinition:
    scenarios = load_scenarios(path)
    if scenario_id not in scenarios:
        raise UnknownScenarioError(f"unknown scenario {scenario_id!r}; known: {sorted(scenarios)}")
    return scenarios[scenario_id]
