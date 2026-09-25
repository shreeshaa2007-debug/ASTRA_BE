"""Sensing Agent tool functions — per agent-plan.md: detect_disruption(),
classify_event(), validate_event().

classify_event() is the only function that talks to an LLM, and all it produces
is a *candidate*: a dict. validate_event() is plain Python and is the only
thing that can turn a candidate into a DisruptionEvent — so nothing an LLM says
becomes part of the world state without passing through it. A candidate that
fails is rejected with every reason listed; it is never repaired or coerced
(severity "high" is not quietly upgraded to "HIGH"; an unknown route id is not
dropped and the rest kept).
"""
from __future__ import annotations

import json
import logging
import math
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import ValidationError

from backend.agents.sensing import prompt
from backend.agents.sensing.llm import LLMClient
from backend.schemas.entities import RiskLevel
from backend.schemas.sensing import (
    Classification,
    RawSignal,
    RouteInfo,
    SensingCatalog,
    SupplierInfo,
    ValidationOutcome,
)
from backend.schemas.world_state import DisruptionEvent

CONFIG_PATH = Path("backend/config/sensing_config.yaml")
logger = logging.getLogger("resilientsc.sensing")

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MAX_LOCATION_CHARS = 200
MAX_SUMMARY_CHARS = 300


class InvalidSignalError(ValueError):
    """The input itself is unusable (empty, oversized, wrong type) — rejected before any LLM call."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_catalog() -> SensingCatalog:
    """The ids in this network, read through the other agents' own tool
    functions (one source of truth per fact)."""
    from backend.agents.inventory import tools as inventory_tools
    from backend.agents.logistics import tools as logistics_tools
    from backend.agents.sourcing import tools as sourcing_tools

    routes = []
    for r in logistics_tools.get_routes():
        parts = r["route_id"].split("-")
        routes.append(RouteInfo(
            route_id=r["route_id"], origin=r["origin"], destination=r["destination"], transport_mode=r["transport_mode"],
            corridor=parts[-1] if len(parts) >= 3 else None,
        ))
    suppliers: dict[str, SupplierInfo] = {}
    for s in sourcing_tools.get_suppliers():
        suppliers.setdefault(s["supplier_id"], SupplierInfo(supplier_id=s["supplier_id"], supplier_name=s["supplier_name"], region=s["region"]))
    return SensingCatalog(routes=routes, suppliers=list(suppliers.values()), product_ids=inventory_tools.get_product_ids())


# --------------------------------------------------------------------------- #
# detect_disruption
# --------------------------------------------------------------------------- #
def detect_disruption(raw: str | Mapping[str, Any], config: dict | None = None, now: datetime | None = None) -> RawSignal:
    """Normalizes the two ways a disruption arrives into one RawSignal.

    A `str` is free text. A mapping is a simulated trigger from the scenario
    framework: `{"scenario_type": ..., "description": ..., "candidate": {...}}`,
    where `candidate` is optional — if present the event is already structured,
    the LLM is skipped, and validate_event() still runs.
    """
    limit = (config or load_config())["limits"]["max_input_chars"]
    received_at = now or _utc_now()

    if isinstance(raw, str):
        source, text, scenario, candidate = "free_text", raw, None, None
    elif isinstance(raw, Mapping):
        source, text, scenario, candidate = "simulated", raw.get("description", ""), raw.get("scenario_type"), raw.get("candidate")
        if candidate is not None and not isinstance(candidate, dict):
            raise InvalidSignalError("a simulated trigger's candidate must be an object")
    else:
        raise InvalidSignalError(f"a disruption signal must be text or a simulated-trigger mapping, not {type(raw).__name__}")

    if not isinstance(text, str):
        raise InvalidSignalError("the signal's description must be text")
    text = _CONTROL_CHARS.sub("", text).strip()
    if not text:
        raise InvalidSignalError("the signal is empty")
    if len(text) > limit:
        raise InvalidSignalError(f"the signal is {len(text)} characters; the limit is {limit}")
    return RawSignal(source=source, text=text, scenario_type=scenario, candidate=candidate, received_at=received_at)


# --------------------------------------------------------------------------- #
# classify_event — the only LLM call
# --------------------------------------------------------------------------- #
def classify_event(
    signal: RawSignal, catalog: SensingCatalog, llm: LLMClient | None, config: dict | None = None, now: datetime | None = None
) -> Classification:
    """Asks the LLM for a candidate event. Returns it parsed but unvalidated;
    output that isn't a JSON object comes back with `candidate=None` and the
    reason in `parse_error`. Raises LLMUnavailableError if the LLM can't be reached."""
    if signal.candidate is not None:  # pre-structured simulated trigger: nothing to classify
        return Classification(candidate=signal.candidate, model="none (pre-structured simulated trigger)")

    if llm is None:
        raise ValueError("classify_event needs an LLM client unless the signal is a pre-structured simulated trigger")
    config = config or load_config()
    response = llm.generate_json(
        system_instruction=prompt.build_system_instruction(catalog, config["event_types"]),
        user_text=prompt.build_user_message(signal, now or _utc_now()),
        response_schema=prompt.response_schema(config["event_types"]),
    )
    base = dict(raw_text=response.text, model=response.model, latency_ms=response.latency_ms, attempts=response.attempts)
    try:
        parsed = json.loads(response.text)
    except json.JSONDecodeError as exc:
        return Classification(candidate=None, parse_error=f"output is not valid JSON ({exc.msg})", **base)
    if not isinstance(parsed, dict):
        return Classification(candidate=None, parse_error=f"output is a JSON {type(parsed).__name__}, not an object", **base)
    return Classification(candidate=parsed, **base)


# --------------------------------------------------------------------------- #
# validate_event — plain Python, the gate
# --------------------------------------------------------------------------- #
def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _id_list(candidate: dict, key: str, known: set[str], errors: list[str], warnings: list[str]) -> list[str]:
    value = candidate.get(key)
    if value is None:  # an event may legitimately affect no suppliers, say
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        errors.append(f"{key} must be a list of id strings")
        return []
    unique = list(dict.fromkeys(value))
    if len(unique) != len(value):
        warnings.append(f"{key} listed an id more than once; kept one of each")
    unknown = [v for v in unique if v not in known]
    if unknown:
        errors.append(f"{key} names ids that are not in this network: {unknown}")
    return unique


def _start_date(value: Any, now: datetime, limits: dict, errors: list[str], warnings: list[str]) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        errors.append("start_date is missing or not a string")
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        errors.append(f"start_date {value!r} is not an ISO-8601 timestamp")
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
        warnings.append("start_date carried no timezone and was taken as UTC")
    parsed = parsed.astimezone(timezone.utc)
    if parsed < now - timedelta(days=limits["max_past_days"]):
        errors.append(f"start_date {parsed.date()} is more than {limits['max_past_days']} days in the past")
    elif parsed > now + timedelta(days=limits["max_future_days"]):
        errors.append(f"start_date {parsed.date()} is more than {limits['max_future_days']} days in the future")
    return parsed


def validate_event(candidate: Any, catalog: SensingCatalog, config: dict | None = None, now: datetime | None = None) -> ValidationOutcome:
    """Schema, range and membership checks on an LLM candidate. Every problem
    found is reported, not just the first. The event_id is minted here — the
    LLM does not get to choose it — and fields outside the schema are ignored."""
    config = config or load_config()
    limits = config["limits"]
    now = now or _utc_now()
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(candidate, dict):
        return ValidationOutcome(valid=False, errors=["candidate is not a JSON object"])

    event_type = candidate.get("event_type")
    if not isinstance(event_type, str) or event_type not in config["event_types"]:
        errors.append(f"event_type {event_type!r} is not one of {config['event_types']}")

    location = candidate.get("location")
    if not isinstance(location, str) or not location.strip():
        errors.append("location is missing or blank")
    elif len(location) > MAX_LOCATION_CHARS:
        errors.append(f"location is {len(location)} characters; the limit is {MAX_LOCATION_CHARS}")

    severity = candidate.get("severity")
    if not isinstance(severity, str) or severity not in {r.value for r in RiskLevel}:
        errors.append(f"severity {severity!r} is not one of {[r.value for r in RiskLevel]} (exact match required)")

    start_date = _start_date(candidate.get("start_date"), now, limits, errors, warnings)

    duration = candidate.get("estimated_duration")
    if not _is_number(duration):
        errors.append(f"estimated_duration {duration!r} is not a number of days")
    elif not 0 <= duration <= limits["max_duration_days"]:
        errors.append(f"estimated_duration {duration} is outside 0..{limits['max_duration_days']} days")

    confidence = candidate.get("confidence")
    if not _is_number(confidence) or not 0 <= confidence <= 1:
        errors.append(f"confidence {confidence!r} is not a number in [0, 1]")
    elif confidence < limits["min_confidence"]:
        errors.append(f"confidence {confidence} is below the {limits['min_confidence']} needed to act on an event")

    routes = _id_list(candidate, "affected_routes", catalog.route_ids, errors, warnings)
    suppliers = _id_list(candidate, "affected_suppliers", catalog.supplier_ids, errors, warnings)
    products = _id_list(candidate, "affected_products", set(catalog.product_ids), errors, warnings)
    if not (routes or suppliers or products) and not any(e.startswith("affected_") for e in errors):
        errors.append("the event affects no route, supplier or product in this network")
    if routes and set(routes) == catalog.route_ids and len(catalog.route_ids) > 1:
        warnings.append("the event disrupts every route in the network")

    rationale = candidate.get("rationale")
    summary = rationale.strip() if isinstance(rationale, str) and rationale.strip() else None
    if summary and len(summary) > MAX_SUMMARY_CHARS:
        summary = summary[: MAX_SUMMARY_CHARS - 1] + "…"
        warnings.append("rationale was longer than the summary limit and was truncated")

    if errors:
        return ValidationOutcome(valid=False, errors=errors, warnings=warnings)
    try:
        event = DisruptionEvent(
            event_id=f"evt-{uuid.uuid4().hex[:10]}", event_type=event_type, location=location.strip(), severity=severity,
            start_date=start_date, estimated_duration=float(duration), affected_routes=routes, affected_suppliers=suppliers,
            affected_products=products, confidence=float(confidence), summary=summary,
        )
    except ValidationError as exc:  # defense in depth: the checks above should already have caught it
        return ValidationOutcome(valid=False, errors=[f"schema validation failed: {exc}"], warnings=warnings)
    return ValidationOutcome(valid=True, event=event, warnings=warnings)
