"""Inbound integration: a disruption reported by another system (SAP Integration Suite, an S/4HANA extension, a logistics
provider's webhook). See backend/integration/signals.py for why this is idempotent."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from backend.agents.inventory import tools as inventory_tools
from backend.api.context import AppContext, get_ctx
from backend.api.errors import ApiError
from backend.api.security import OPERATE, require
from backend.integration.signals import SCENARIO_TYPE, IntegrationSignal
from backend.services.world_state import SimulationExistsError, load_baseline

router = APIRouter(prefix="/api/integration", tags=["integration"], dependencies=[Depends(require(OPERATE))])


@router.post("/signals", status_code=202)
def receive_signal(body: IntegrationSignal, response: Response, ctx: AppContext = Depends(get_ctx)):
    """Accepts a disruption signal and starts the pipeline on it, exactly as `POST /simulations/{id}/run` would.

    Idempotent per (`source_system`, `external_id`): the first delivery answers 202 and starts a run; a redelivery of the
    same signal answers 200 with `duplicate: true` and the simulation's current status, and starts nothing. Poll
    `status_url` (or listen for the `com.resilientsc.plan.*` events) to follow it."""
    if body.product_id not in inventory_tools.get_product_ids():
        raise ApiError(422, "VALIDATION_ERROR", f"unknown product_id {body.product_id!r}; GET /api/inventory lists the products")
    unknown = sorted(set(body.tariff_overrides or {}) - set(load_baseline().tariffs))
    if unknown:
        raise ApiError(422, "VALIDATION_ERROR", f"tariff_overrides names countries with no baseline tariff: {unknown}")

    simulation_id = body.simulation_id
    status_url = f"/api/simulations/{simulation_id}/status"
    try:
        ctx.store.create(SCENARIO_TYPE, simulation_id=simulation_id, actor=f"integration:{body.source_system}")
    except SimulationExistsError:  # a redelivery, or the other half of a race between two deliveries
        response.status_code = 200
        return {"data": {"simulation_id": simulation_id, "duplicate": True, "status": ctx.store.get(simulation_id).status.value, "status_url": status_url}}

    record = ctx.runs.start(simulation_id, lambda: ctx.orchestrator.run(
        simulation_id, body.trigger(), body.product_id, tariff_overrides=body.tariff_overrides))
    return {"data": {"simulation_id": simulation_id, "duplicate": False, "run_id": record.run_id, "run_state": record.state, "status_url": status_url}}
