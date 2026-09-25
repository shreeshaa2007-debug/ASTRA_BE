"""Turns a validated DisruptionEvent into the field updates the `event_sensed`
checkpoint commits, so every caller applies an event the same way.
"""
from __future__ import annotations

from typing import Optional

from backend.schemas.entities import RouteStatus, SupplierStatus
from backend.schemas.world_state import DisruptionEvent, WorldState

# What an event of each type does to the routes and suppliers it lists:
# (status given to affected routes, status given to affected suppliers).
# None = the event is recorded but changes no status. This is the state layer's
# own rule, not the LLM's: a tariff or a demand surge lists routes/suppliers as
# *exposed*, and must never mark a lane blocked just because the LLM named it
# (caught live: a China tariff report listed six routes). The types must match
# `event_types` in backend/config/sensing_config.yaml — a test enforces that.
#
# DELAYED / REDUCED are recorded but not yet acted on: the Logistics and
# Sourcing agents exclude only DISRUPTED lanes and suppliers, and neither
# models a transit-time or capacity change. A tariff's *amount* is not part of
# the event at all (brief §13); it arrives from the scenario definition.
EVENT_EFFECTS: dict[str, tuple[Optional[RouteStatus], Optional[SupplierStatus]]] = {
    "canal_closure": (RouteStatus.DISRUPTED, SupplierStatus.DISRUPTED),
    "severe_weather": (RouteStatus.DISRUPTED, SupplierStatus.DISRUPTED),
    "supplier_failure": (RouteStatus.DISRUPTED, SupplierStatus.DISRUPTED),
    "port_congestion": (RouteStatus.DELAYED, SupplierStatus.REDUCED),
    "tariff_change": (None, None),
    "demand_surge": (None, None),
    "other": (None, None),
}


def state_changes_for_event(state: WorldState, event: DisruptionEvent) -> dict:
    """Records `event` (replacing an earlier version with the same event_id)
    and applies its type's effect (EVENT_EFFECTS) to the routes and suppliers
    it lists. An id the simulation doesn't know is an error, not something to
    skip: a typo'd route id would otherwise leave a blocked route looking
    healthy. An event type with no entry in EVENT_EFFECTS is an error too."""
    if event.event_type not in EVENT_EFFECTS:
        raise ValueError(f"event {event.event_id!r} has type {event.event_type!r}, which has no defined effect on the state")
    route_effect, supplier_effect = EVENT_EFFECTS[event.event_type]

    unknown_routes = sorted(set(event.affected_routes) - set(state.route_status))
    unknown_suppliers = sorted(set(event.affected_suppliers) - set(state.supplier_status))
    if unknown_routes or unknown_suppliers:
        parts = []
        if unknown_routes:
            parts.append(f"unknown route ids {unknown_routes}")
        if unknown_suppliers:
            parts.append(f"unknown supplier ids {unknown_suppliers}")
        raise ValueError(f"event {event.event_id!r} references {' and '.join(parts)}")

    route_status = dict(state.route_status)
    if route_effect is not None:
        route_status.update({r: route_effect for r in event.affected_routes})
    supplier_status = dict(state.supplier_status)
    if supplier_effect is not None:
        supplier_status.update({s: supplier_effect for s in event.affected_suppliers})

    return {
        "current_disruptions": [d for d in state.current_disruptions if d.event_id != event.event_id] + [event],
        "route_status": route_status,
        "supplier_status": supplier_status,
    }
