"""The plan, its compliance verdict, and the human's decision (api-plan.md).
`{id}` is a simulation id: one simulation holds one plan."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api import views
from backend.api.context import AppContext, get_ctx
from backend.api.models import DecisionRequest
from backend.api.security import APPROVE, VIEW, Principal, decided_by, require

router = APIRouter(prefix="/api", tags=["decisions"], dependencies=[Depends(require(VIEW))])


def envelope(data):
    return {"data": data}


@router.get("/decisions/{simulation_id}")
def get_decision(simulation_id: str, ctx: AppContext = Depends(get_ctx)):
    """Why this plan: factors, the split, objective terms, constraints, and where it departs from each agent's own recommendation."""
    return envelope(views.decision_view(ctx.store.get(simulation_id)))


@router.get("/compliance/{simulation_id}")
def get_compliance(simulation_id: str, ctx: AppContext = Depends(get_ctx)):
    return envelope(views.compliance_view(ctx.store.get(simulation_id)))


def _decision_response(state) -> dict:
    return {
        "simulation_id": state.simulation_id, "status": state.status.value, "version": state.version,
        "approval_status": state.approval_status.value,
        "approval_decision": state.approval_decision.model_dump(mode="json") if state.approval_decision else None,
    }


@router.post("/decisions/{simulation_id}/approve")
def approve(simulation_id: str, body: DecisionRequest, ctx: AppContext = Depends(get_ctx), principal: Principal = Depends(require(APPROVE))):
    """A named human approves the escalated plan, which finalizes it. Pass `expected_version` — the
    version you were shown — to be refused if the plan has since changed."""
    return envelope(_decision_response(ctx.orchestrator.approve(simulation_id, decided_by(principal, body.decided_by), body.note, expected_version=body.expected_version)))


@router.post("/decisions/{simulation_id}/reject")
def reject(simulation_id: str, body: DecisionRequest, ctx: AppContext = Depends(get_ctx), principal: Principal = Depends(require(APPROVE))):
    """A named human rejects the escalated plan; the simulation ends REJECTED."""
    return envelope(_decision_response(ctx.orchestrator.reject(simulation_id, decided_by(principal, body.decided_by), body.note, expected_version=body.expected_version)))
