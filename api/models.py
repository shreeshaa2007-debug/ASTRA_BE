"""Request bodies. Bounds live here so a bad request is a 422 at the door, not a
failed run discovered by polling."""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.agents.sensing import tools as sensing_tools

@lru_cache(maxsize=1)
def max_signal_chars() -> int:
    """The limit the Sensing Agent enforces, read from its config (lazily, so importing this
    module does no file I/O) so the two can't drift."""
    return sensing_tools.load_config()["limits"]["max_input_chars"]


def checked_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        raise ValueError("must not be empty")
    if len(stripped) > max_signal_chars():
        raise ValueError(f"is {len(stripped)} characters; the limit is {max_signal_chars()}")
    return text


class CreateSimulationRequest(BaseModel):
    scenario_type: str = Field(min_length=1, max_length=64, description="e.g. SUEZ_CLOSURE")
    simulation_id: Optional[str] = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$", description="omit to have one generated")


class SimulatedTrigger(BaseModel):
    """A simulated trigger from the scenario framework. `candidate` may carry an
    already-structured event: the LLM is then skipped, validation is not."""

    scenario_type: Optional[str] = Field(default=None, max_length=64)
    description: str
    candidate: Optional[dict[str, Any]] = None

    @field_validator("description")
    @classmethod
    def _description(cls, v: str) -> str:
        return checked_text(v)


class RunRequest(BaseModel):
    signal: str | SimulatedTrigger = Field(description="free text, or a simulated trigger")
    product_id: str = Field(min_length=1, max_length=32, description="the product to plan; one product per run")
    as_of_date: Optional[date] = Field(default=None, description="inventory as-of date; default: latest in the ledger")
    tariff_overrides: Optional[dict[str, float]] = Field(default=None, description="ISO3 -> tariff %, merged over the baseline")

    @field_validator("signal")
    @classmethod
    def _signal(cls, v):
        return checked_text(v) if isinstance(v, str) else v

    def signal_for_orchestrator(self) -> str | dict:
        if isinstance(self.signal, str):
            return self.signal
        return self.signal.model_dump(exclude_none=True)


class DecisionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)  # so "   " is a 422 here, not an approval by nobody

    decided_by: str = Field(min_length=1, max_length=100, description="who is deciding; recorded in the audit trail")
    note: str = Field(default="", max_length=1000)
    expected_version: Optional[int] = Field(default=None, ge=0, description="the version you were shown; refuses if the plan has since changed")


class ForecastRequest(BaseModel):
    """model-plan.md §6."""

    product_id: str = Field(min_length=1, max_length=32)
    location_id: str = Field(default="United Kingdom", min_length=1, max_length=64)
    forecast_horizon: int = Field(default=14, ge=1, le=90)
    recent_demand: list[float] = Field(min_length=1, max_length=365, description="actual demand, oldest first; the last 28 are used")
    start_date: Optional[date] = Field(default=None, description="first day to forecast; default: tomorrow")
