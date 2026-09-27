"""From world-state changes to domain events.

The store already funnels every change through one named checkpoint (`backend/services/world_state`), so this is the
single place that decides what the outside world hears about. `StoreEventBridge` is a store listener: after a change is
saved it maps the checkpoint to an event type and publishes it.

What an event carries is deliberately small and stable: identifiers, statuses and the plan's headline lines — enough for
an integration flow to route on and to act on (create a purchase requisition per allocation, start an approval task for
an escalated plan), not the whole state. A receiver that wants more reads `GET /api/simulations/{subject}`. Free text a
person typed (an approval note) and the raw report the disruption was sensed from are not sent.
"""
from __future__ import annotations

from typing import Any, Iterable

from backend.integration.events import DomainEvent, EventPublisher
from backend.schemas.world_state import WorldState
from backend.services.world_state import StateChange

TYPE_PREFIX = "com.resilientsc."

# every checkpoint the store has, as an event type
EVENT_TYPES: dict[str, str] = {
    "simulation_created": TYPE_PREFIX + "simulation.created",
    "event_sensed": TYPE_PREFIX + "disruption.sensed",
    "agents_assessed": TYPE_PREFIX + "agents.assessed",
    "plan_optimized": TYPE_PREFIX + "plan.optimized",
    "compliance_checked": TYPE_PREFIX + "plan.compliance.checked",
    "approval_requested": TYPE_PREFIX + "plan.approval.requested",
    "plan_finalized": TYPE_PREFIX + "plan.finalized",
    "plan_rejected_by_human": TYPE_PREFIX + "plan.rejected",
    "replan_requested": TYPE_PREFIX + "plan.replan.requested",
    "run_failed": TYPE_PREFIX + "run.failed",
    "simulation_reset": TYPE_PREFIX + "simulation.reset",
}

# the ones a business process acts on; the rest are internal steps (EVENTS_CHECKPOINTS=all publishes them too)
DEFAULT_CHECKPOINTS = ("event_sensed", "plan_optimized", "approval_requested", "plan_finalized", "plan_rejected_by_human", "run_failed")

COST_UNIT = "cost units (no currency peg: the model's costs are relative)"


def event_data(state: WorldState, checkpoint: str, actor: str) -> dict[str, Any]:
    data: dict[str, Any] = {
        "simulation_id": state.simulation_id, "scenario_type": state.scenario_type, "status": state.status.value,
        "version": state.version, "checkpoint": checkpoint, "actor": actor,
        "disruptions": [
            {"event_id": d.event_id, "event_type": d.event_type, "location": d.location, "severity": d.severity.value,
             "affected_routes": d.affected_routes, "affected_suppliers": d.affected_suppliers, "affected_products": d.affected_products}
            for d in state.current_disruptions
        ],
        "plan": None, "compliance": None,
        "approval": {"status": state.approval_status.value, "decided_by": None, "decided_at": None},
    }
    plan = state.current_plan
    if plan is not None:
        data["plan"] = {
            "product_id": plan.problem.product_id, "status": plan.status, "engine": plan.engine, "label": plan.label,
            "objective_value": plan.objective_value, "cost_unit": COST_UNIT,
            "allocations": [{"supplier_id": a.supplier_id, "route_id": a.route_id, "transport_mode": a.transport_mode, "quantity": a.quantity,
                             "landed_unit_cost": a.landed_unit_cost, "freight_unit_cost": a.freight_unit_cost, "arrival_days": a.arrival_days}
                            for a in plan.allocations],
            "transfers": [{"from_warehouse": t.from_warehouse, "to_warehouse": t.to_warehouse, "quantity": t.quantity} for t in plan.transfers],
        }
    if state.compliance_status is not None:
        data["compliance"] = {"status": state.compliance_status.status, "reason": state.compliance_status.reason,
                              "requires_human": state.compliance_status.requires_human}
    if state.approval_decision is not None:
        data["approval"].update(decided_by=state.approval_decision.decided_by, decided_at=state.approval_decision.decided_at.isoformat())
    if state.error:
        data["error"] = state.error
    return data


def build_event(change: StateChange, source: str) -> DomainEvent:
    state = change.state
    return DomainEvent(
        id=f"{state.simulation_id}:{state.version}",  # unique per change, identical on a retry: the key a receiver de-duplicates on
        type=EVENT_TYPES.get(change.checkpoint, TYPE_PREFIX + "simulation." + change.checkpoint),
        source=source, subject=state.simulation_id, time=state.timestamp.isoformat(),
        data=event_data(state, change.checkpoint, change.actor),
    )


class StoreEventBridge:
    """A `WorldStateStore` listener: `store.add_listener(StoreEventBridge(publisher))`."""

    def __init__(self, publisher: EventPublisher, *, source: str = "urn:resilientsc", checkpoints: Iterable[str] = DEFAULT_CHECKPOINTS):
        self._publisher = publisher
        self._source = source
        self._checkpoints = frozenset(checkpoints)
        unknown = sorted(self._checkpoints - set(EVENT_TYPES))
        if unknown:
            raise ValueError(f"unknown checkpoint(s) {unknown}; known: {sorted(EVENT_TYPES)}")

    def __call__(self, change: StateChange) -> None:
        if change.checkpoint in self._checkpoints:
            self._publisher.publish(build_event(change, self._source))


def checkpoints_from_env(raw: str | None) -> tuple[str, ...]:
    """EVENTS_CHECKPOINTS: unset -> the business milestones; `all` -> every checkpoint; else a comma-separated list."""
    if not raw or not raw.strip():
        return DEFAULT_CHECKPOINTS
    if raw.strip().lower() == "all":
        return tuple(EVENT_TYPES)
    return tuple(name.strip() for name in raw.split(",") if name.strip())
