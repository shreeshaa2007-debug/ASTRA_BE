"""API router for real-time port telemetry, maritime chokepoints, and congestion monitoring."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.api.context import AppContext, get_ctx
from backend.api.security import VIEW, require
from backend.services.ports_realtime import get_port_realtime_service

router = APIRouter(prefix="/api/ports", tags=["ports-realtime"])


def envelope(data: Any) -> Dict[str, Any]:
    return {"data": data}


@router.get("/realtime", summary="Get real-time port telemetry")
def get_realtime_ports(
    status: Optional[str] = Query(None, description="Filter by status (NORMAL, SLOWDOWN, CONGESTED, BLOCKED)"),
    severity: Optional[str] = Query(None, description="Filter by congestion severity (LOW, MEDIUM, HIGH, CRITICAL)"),
):
    """Returns live port operations data including vessel queue, anchorage wait time,
    congestion index, turnaround hours, and marine weather."""
    svc = get_port_realtime_service()
    ports = svc.extract_realtime_ports(force_refresh=False)

    if status:
        ports = [p for p in ports if p["operational_status"].upper() == status.upper()]
    if severity:
        ports = [p for p in ports if p["congestion_severity"].upper() == severity.upper()]

    return envelope(ports)


@router.get("/chokepoints", summary="Get maritime strategic chokepoints")
def get_chokepoints():
    """Returns strategic maritime chokepoint transit numbers, risk levels, and detour alternatives."""
    svc = get_port_realtime_service()
    return envelope(svc.get_chokepoints())


@router.get("/summary", summary="Get global port network summary")
def get_ports_summary():
    """Returns aggregate network congestion KPIs and alert levels."""
    svc = get_port_realtime_service()
    return envelope(svc.get_network_congestion_summary())


@router.post("/refresh", summary="Trigger live telemetry extraction")
def refresh_ports_telemetry():
    """Forces an immediate re-extraction and calculation of port operations and chokepoint telemetry."""
    svc = get_port_realtime_service()
    ports = svc.extract_realtime_ports(force_refresh=True)
    summary = svc.get_network_congestion_summary()
    return envelope({"message": "Real-time port telemetry refreshed", "summary": summary, "ports_count": len(ports)})


@router.get("/model-metrics", summary="Get trained port delay ML model performance and features")
def get_port_model_metrics():
    """Returns the trained XGBoost model R2 score, MAE, accuracy, and feature importances."""
    svc = get_port_realtime_service()
    return envelope(svc.get_model_metrics())


@router.api_route("/predict-delay", methods=["GET", "POST"], summary="Run live ML inference on port delay")
def predict_port_delay(
    port_id: Optional[str] = Query(None, description="Port UNLOCODE or ID (e.g. CNSHA, EGPSD, NLRTM)"),
    anchorage_vessels: Optional[float] = Query(None, description="Custom vessel queue count"),
    yard_utilization_pct: Optional[float] = Query(None, description="Custom yard utilization %"),
    wind_speed_knots: Optional[float] = Query(None, description="Wind speed in knots"),
    wave_height_meters: Optional[float] = Query(None, description="Wave height in meters"),
):
    """Predicts continuous delay hours and congestion severity using the newly trained XGBoost model."""
    svc = get_port_realtime_service()
    prediction = svc.predict_port_delay(
        port_id=port_id,
        anchorage_vessels=anchorage_vessels,
        yard_utilization_pct=yard_utilization_pct,
        wind_speed_knots=wind_speed_knots,
        wave_height_meters=wave_height_meters,
    )
    return envelope(prediction)


@router.get("/{port_id}", summary="Get specific port real-time telemetry")
def get_port_detail(port_id: str):
    svc = get_port_realtime_service()
    port = svc.get_port_by_id(port_id)
    if not port:
        raise HTTPException(status_code=404, detail=f"Port {port_id!r} not found in monitored network")
    return envelope(port)

