"""The named checkpoints — the only ways the shared world state can change
(architecture.md §1/§5, agent-plan.md ground rules).

Agents never write. The orchestrator commits at these checkpoints and nowhere
else, and each checkpoint owns a fixed set of fields, so a run can't, say,
overwrite the plan while recording a disruption. The registry also encodes the
two safety properties the brief cares most about, so they hold even if the
orchestrator (Phase 14) has a bug:

  * a plan is finalized only if it is optimal, compliance did not reject it,
    and — when compliance escalated it — a named human approved it;
  * a compliance-rejected plan gets at most MAX_REPLANS replans, never a loop.

Phase 14 adds a checkpoint by adding an entry here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from backend.schemas.world_state import ApprovalStatus, SimulationStatus, WorldState

MAX_REPLANS = 1  # agent-plan.md § Orchestration routing: "max 1 replan attempt, then END"

Guard = Callable[[WorldState, WorldState], Optional[str]]  # (before, after) -> problem description, or None


@dataclass(frozen=True)
class CheckpointSpec:
    description: str
    allowed_fields: frozenset[str]
    from_statuses: frozenset[SimulationStatus]
    to_status: SimulationStatus
    guard: Guard = lambda before, after: None


def _plan_is_optimal(state: WorldState) -> bool:
    return state.current_plan is not None and state.current_plan.status == "OPTIMAL"


def _event_sensed(before: WorldState, after: WorldState) -> Optional[str]:
    return None if after.current_disruptions else "event_sensed requires at least one disruption in current_disruptions"


def _plan_optimized(before: WorldState, after: WorldState) -> Optional[str]:
    return None if after.current_plan is not None else "plan_optimized requires current_plan"


def _compliance_checked(before: WorldState, after: WorldState) -> Optional[str]:
    if not _plan_is_optimal(after):
        return "compliance can only check an OPTIMAL plan (none is on the state)"
    if after.compliance_status is None:
        return "compliance_checked requires compliance_status"
    return None


def _approval_requested(before: WorldState, after: WorldState) -> Optional[str]:
    c = after.compliance_status
    if c is None or not c.requires_human or c.status != "ESCALATED":
        return "approval can only be requested for a plan compliance ESCALATED to a human"
    if after.approval_status != ApprovalStatus.PENDING:
        return "approval_requested must set approval_status to PENDING"
    return None


def _plan_finalized(before: WorldState, after: WorldState) -> Optional[str]:
    if not _plan_is_optimal(after):
        return "cannot finalize: there is no OPTIMAL plan"
    c = after.compliance_status
    if c is None:
        return "cannot finalize: the plan has not been through compliance"
    if c.status == "REJECTED":
        return "cannot finalize: compliance REJECTED this plan"
    if c.requires_human:
        if before.status != SimulationStatus.AWAITING_APPROVAL:
            return "cannot finalize an escalated plan that was never sent for approval"
        if after.approval_status != ApprovalStatus.APPROVED or after.approval_decision is None:
            return "cannot finalize: an escalated plan needs approval_status APPROVED with a named approver in approval_decision"
    elif after.approval_status != ApprovalStatus.NOT_REQUIRED:
        return "an in-policy plan finalizes with approval_status NOT_REQUIRED"
    return None


def _plan_rejected_by_human(before: WorldState, after: WorldState) -> Optional[str]:
    if after.approval_status != ApprovalStatus.REJECTED or after.approval_decision is None:
        return "plan_rejected_by_human requires approval_status REJECTED with a named decider in approval_decision"
    return None


def _replan_requested(before: WorldState, after: WorldState) -> Optional[str]:
    if before.compliance_status is None or before.compliance_status.status != "REJECTED":
        return "a replan is only for a plan compliance REJECTED"
    if after.replan_count != before.replan_count + 1:
        return "replan_requested must increase replan_count by exactly 1"
    if after.replan_count > MAX_REPLANS:
        return f"replan limit reached ({MAX_REPLANS}); record run_failed instead of retrying"
    if after.current_plan is not None or after.compliance_status is not None:
        return "replan_requested must clear current_plan and compliance_status"
    return None


def _run_failed(before: WorldState, after: WorldState) -> Optional[str]:
    return None if after.error else "run_failed requires a non-empty error"


_ACTIVE = frozenset({SimulationStatus.RUNNING})
_OPEN = frozenset({SimulationStatus.CREATED, SimulationStatus.RUNNING, SimulationStatus.AWAITING_APPROVAL})

CHECKPOINTS: dict[str, CheckpointSpec] = {
    "event_sensed": CheckpointSpec(
        "Sensing Agent's validated event applied: disruption recorded, affected routes/suppliers marked, tariffs adjusted",
        frozenset({"current_disruptions", "route_status", "supplier_status", "tariffs"}),
        frozenset({SimulationStatus.CREATED}), SimulationStatus.RUNNING, _event_sensed,
    ),
    "agents_assessed": CheckpointSpec(
        "Inventory/Logistics/Sourcing recommendations recorded",
        frozenset({"inventory_status", "demand_forecasts", "shipment_status"}), _ACTIVE, SimulationStatus.RUNNING,
    ),
    "plan_optimized": CheckpointSpec(
        "Optimization Engine's solution recorded (an INFEASIBLE solution is recorded too, for its diagnosis)",
        frozenset({"current_plan"}), _ACTIVE, SimulationStatus.RUNNING, _plan_optimized,
    ),
    "compliance_checked": CheckpointSpec(
        "Compliance Agent's verdict on the optimized plan recorded",
        frozenset({"compliance_status"}), _ACTIVE, SimulationStatus.RUNNING, _compliance_checked,
    ),
    "approval_requested": CheckpointSpec(
        "Escalated plan sent to a human; the run waits",
        frozenset({"approval_status"}), _ACTIVE, SimulationStatus.AWAITING_APPROVAL, _approval_requested,
    ),
    "plan_finalized": CheckpointSpec(
        "Plan is final: auto-approved in-policy, or approved by a named human",
        frozenset({"approval_status", "approval_decision"}),
        frozenset({SimulationStatus.RUNNING, SimulationStatus.AWAITING_APPROVAL}), SimulationStatus.COMPLETED, _plan_finalized,
    ),
    "plan_rejected_by_human": CheckpointSpec(
        "A human rejected the escalated plan; terminal",
        frozenset({"approval_status", "approval_decision"}),
        frozenset({SimulationStatus.AWAITING_APPROVAL}), SimulationStatus.REJECTED, _plan_rejected_by_human,
    ),
    "replan_requested": CheckpointSpec(
        "Compliance REJECTED the plan; discard it and re-optimize (bounded by MAX_REPLANS)",
        frozenset({"current_plan", "compliance_status", "replan_count"}), _ACTIVE, SimulationStatus.RUNNING, _replan_requested,
    ),
    "run_failed": CheckpointSpec(
        "The run cannot produce a final plan (e.g. infeasible after replan); terminal",
        frozenset({"error"}), _OPEN, SimulationStatus.FAILED, _run_failed,
    ),
}
