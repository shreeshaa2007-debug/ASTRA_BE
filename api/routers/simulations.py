"""Simulation lifecycle: create, read, run, poll, reset (api-plan.md)."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query

from backend.agents.inventory import tools as inventory_tools
from backend.api import views
from backend.api.context import AppContext, get_ctx
from backend.api.errors import ApiError
from backend.api.models import CreateSimulationRequest, RunRequest
from backend.schemas.world_state import SimulationStatus
from backend.simulation import compare_state

router = APIRouter(prefix="/api/simulations", tags=["simulations"])


def envelope(data):
    return {"data": data}


@router.post("", status_code=201)
def create_simulation(body: CreateSimulationRequest, ctx: AppContext = Depends(get_ctx)):
    """A new isolated simulation at the baseline network: routes, suppliers and tariffs as the data files have them."""
    return envelope(views.state_view(ctx.store.create(body.scenario_type, simulation_id=body.simulation_id)))


@router.get("")
def list_simulations(status: SimulationStatus | None = None, limit: int = Query(50, ge=1, le=200), ctx: AppContext = Depends(get_ctx)):
    return envelope([s.model_dump(mode="json") for s in ctx.store.list_simulations(status, limit)])


@router.get("/{simulation_id}")
def get_simulation(simulation_id: str, ctx: AppContext = Depends(get_ctx)):
    """The full world state."""
    return envelope(views.state_view(ctx.store.get(simulation_id)))


@router.post("/{simulation_id}/run", status_code=202)
def run_simulation(simulation_id: str, body: RunRequest, ctx: AppContext = Depends(get_ctx)):
    """Starts the pipeline on a worker thread and returns at once; poll `/status`.
    Everything that can be checked cheaply is checked here, so a bad request is a
    422/409 now rather than a failed run found by polling."""
    state = ctx.store.get(simulation_id)  # 404 if unknown
    if state.status != SimulationStatus.CREATED:
        raise ApiError(409, "INVALID_STATE_TRANSITION",
                       f"simulation {simulation_id!r} is {state.status.value}; only a CREATED simulation can be run (reset it to run again)")
    if body.product_id not in inventory_tools.get_product_ids():
        raise ApiError(422, "VALIDATION_ERROR", f"unknown product_id {body.product_id!r}; GET /api/inventory lists the products")
    unknown = sorted(set(body.tariff_overrides or {}) - set(state.tariffs))
    if unknown:
        raise ApiError(422, "VALIDATION_ERROR", f"tariff_overrides names countries with no baseline tariff: {unknown}; known: {sorted(state.tariffs)}")

    signal = body.signal_for_orchestrator()
    as_of = body.as_of_date.isoformat() if body.as_of_date else None
    record = ctx.runs.start(simulation_id, lambda: ctx.orchestrator.run(
        simulation_id, signal, body.product_id, as_of_date=as_of, tariff_overrides=body.tariff_overrides))
    return envelope({"simulation_id": simulation_id, "run_id": record.run_id, "run_state": record.state, "status_url": f"/api/simulations/{simulation_id}/status"})


@router.get("/{simulation_id}/status")
def get_status(simulation_id: str, ctx: AppContext = Depends(get_ctx)):
    """Current step, per-agent status, whether the run is waiting for a human, and how the latest run ended."""
    return envelope(views.read_status(ctx.store, ctx.runs, simulation_id))


@router.post("/{simulation_id}/reset")
def reset_simulation(simulation_id: str, ctx: AppContext = Depends(get_ctx)):
    """Back to the state the simulation was created with (from any status); the audit trail is kept."""
    if ctx.runs.is_running(simulation_id):
        raise ApiError(409, "RUN_IN_PROGRESS", f"a run is in progress for simulation {simulation_id!r}; wait for it to finish before resetting")
    state = ctx.store.reset(simulation_id)
    ctx.runs.forget(simulation_id)
    return envelope(views.state_view(state))


@router.get("/{simulation_id}/comparison")
def get_comparison(simulation_id: str, product_id: Optional[str] = None, as_of_date: Optional[str] = None, ctx: AppContext = Depends(get_ctx)):
    """This simulation's disruption compared three ways: normal operations, doing nothing, and the plan it
    actually produced (docs/api-plan.md, Phase 17). The product comes from the plan. The state does not record the
    as-of date a run used, so pass `as_of_date` if the run had one; otherwise the ledger's latest is used, and a
    warning says so when that differs from the plan's own stock."""
    state = ctx.store.get(simulation_id)
    if state.current_plan is None and not product_id:
        raise ApiError(409, "PLAN_NOT_READY", f"simulation {simulation_id!r} is {state.status.value}; it has no plan, so there is no product to compare")
    if product_id and product_id not in inventory_tools.get_product_ids():
        raise ApiError(422, "VALIDATION_ERROR", f"unknown product_id {product_id!r}; GET /api/products lists the products")
    return envelope(compare_state(state, product_id, as_of_date).model_dump(mode="json"))
