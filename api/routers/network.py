"""Read-only views over the network and the latest results: dashboard,
disruptions, inventory, forecasts, suppliers, routes, shipments, agent status
(api-plan.md).

Scoping is deliberate and the same everywhere: the dashboard, disruptions and
shipments follow the *latest finalized* simulation (api-plan.md: "reflects the
latest finalized simulation, not an in-progress one") unless `simulation_id` says
otherwise; the routes, suppliers and inventory tables show the *baseline network*
unless `simulation_id` overlays a simulation's state on it.
"""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query

from backend.api import views
from backend.api.context import AppContext, get_ctx
from backend.api.models import ForecastRequest
from backend.schemas.world_state import SimulationStatus, WorldState

router = APIRouter(prefix="/api", tags=["network"])


def envelope(data):
    return {"data": data}


def _scoped(ctx: AppContext, simulation_id: str | None) -> WorldState | None:
    """The named simulation (404 if unknown), else the latest finalized one, else None."""
    if simulation_id:
        return ctx.store.get(simulation_id)
    finished = ctx.store.list_simulations(SimulationStatus.COMPLETED, limit=1)
    return ctx.store.get(finished[0].simulation_id) if finished else None


def _overlay(ctx: AppContext, simulation_id: str | None) -> WorldState | None:
    """Baseline unless a simulation is named."""
    return ctx.store.get(simulation_id) if simulation_id else None


@router.get("/dashboard")
def dashboard(ctx: AppContext = Depends(get_ctx)):
    return envelope(views.dashboard_view(_scoped(ctx, None)))


@router.get("/disruptions")
def disruptions(
    simulation_id: Optional[str] = None,
    source: Literal["all", "curated", "noaa"] = "all",
    severity: Optional[Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    ctx: AppContext = Depends(get_ctx),
):
    """`active`: the events sensed in the scoped simulation. `historical`: 69,805 real events — 4 curated
    (Suez 2021 and others) and NOAA storm events — newest first, paginated."""
    return envelope(views.disruptions_view(_scoped(ctx, simulation_id), source, severity, limit, offset))


@router.get("/inventory")
def inventory(simulation_id: Optional[str] = None, product_id: Optional[str] = None, ctx: AppContext = Depends(get_ctx)):
    """Warehouse-level stock. With `simulation_id` and once its agents have run: that simulation's forecast,
    risk and recommended transfer. Otherwise the current ledger snapshot."""
    return envelope(views.inventory_rows(_overlay(ctx, simulation_id), product_id))


@router.get("/inventory/forecast")
def inventory_forecast(
    product_id: str,
    warehouse_id: str = "Mumbai",
    horizon_days: int = Query(14, ge=1, le=90),
    as_of_date: Optional[date] = None,
):
    """Actual (last 28 days) and predicted demand for one product at one warehouse. A point forecast: no confidence band is drawn."""
    return envelope(views.forecast_for_warehouse(product_id, warehouse_id, horizon_days, as_of_date.isoformat() if as_of_date else None))


@router.post("/forecast")
def forecast(body: ForecastRequest):
    """model-plan.md §6: forecast from demand history you supply."""
    return envelope(views.forecast_from_history(body))


@router.get("/suppliers")
def suppliers(simulation_id: Optional[str] = None, product_id: Optional[str] = None, ctx: AppContext = Depends(get_ctx)):
    """Suppliers with their status and tariff; with `simulation_id`, that simulation's status overlay and its recommended sourcing mix."""
    return envelope(views.supplier_rows(_overlay(ctx, simulation_id), product_id))


@router.get("/routes")
def routes(simulation_id: Optional[str] = None, ctx: AppContext = Depends(get_ctx)):
    """The network with each route's status; with `simulation_id`, that simulation's disruptions and the volume its plan puts on each route."""
    return envelope(views.route_rows(_overlay(ctx, simulation_id)))


@router.get("/shipments")
def shipments(simulation_id: Optional[str] = None, route_id: Optional[str] = None, status: Optional[str] = None, ctx: AppContext = Depends(get_ctx)):
    """No shipment dataset exists, so there are no observed shipments: this lists what the scoped plan proposes."""
    return envelope(views.shipment_rows(_scoped(ctx, simulation_id), route_id, status))


@router.get("/agents/status")
def agents_status(simulation_id: Optional[str] = None, ctx: AppContext = Depends(get_ctx)):
    """Per-agent status and the execution timeline. Default: the most recently updated simulation."""
    if not simulation_id:
        recent = ctx.store.list_simulations(limit=1)
        if not recent:
            return envelope({"simulation_id": None, "agents": [{"id": s, "name": n, "status": "PENDING", "detail": "", "latency_ms": None}
                                                              for s, n in views._STAGES], "timeline": []})
        simulation_id = recent[0].simulation_id
    return envelope(views.read_status(ctx.store, ctx.runs, simulation_id))
