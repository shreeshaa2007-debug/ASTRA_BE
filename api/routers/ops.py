"""What an operator asks of a running system: is it ready, what has it been doing, and is its model still
trustworthy (docs/architecture.md §5.4). Read-only, except that a backtest is computed on request."""
from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from backend.agents.inventory import tools as inventory_tools
from backend.api import readiness
from backend.api.context import AppContext, get_ctx
from backend.api.errors import ApiError
from backend.api.security import VIEW, Principal, current_principal, require
from backend.monitoring import metrics, monitor

router = APIRouter(prefix="/api", tags=["operations"])


@router.get("/me")
def me(principal: Principal = Depends(current_principal)):
    """Who the API thinks you are and what you may do. With authentication off, an anonymous caller holding every scope."""
    return {"data": {"name": principal.name, "authenticated": principal.authenticated, "is_user": principal.is_user, "scopes": sorted(principal.scopes)}}


@router.get("/ready")
def ready(ctx: AppContext = Depends(get_ctx)):
    """200 when every required dependency answers, 503 when one does not; `degraded` flags a failed optional one.
    The body is the same either way: the list of checks and what each found."""
    checks = readiness.run_checks(ctx)
    is_ready = all(c["ok"] for c in checks if c["required"])
    body = {"data": {"ready": is_ready, "degraded": any(not c["ok"] for c in checks if not c["required"]), "checks": checks}}
    return JSONResponse(body, status_code=200 if is_ready else 503)


@router.get("/metrics", dependencies=[Depends(require(VIEW))])
def get_metrics(format: Literal["json", "prometheus"] = "json", ctx: AppContext = Depends(get_ctx)):
    """Counters, gauges and latency summaries for this process (uptime, requests, runs and their steps, the optimizer,
    the language model, compliance, approvals, the demand model). `?format=prometheus` is the Prometheus text format."""
    metrics.set_gauge("runs_in_flight", ctx.runs.in_flight())
    if format == "prometheus":
        return PlainTextResponse(metrics.prometheus(), media_type="text/plain; version=0.0.4")
    return {"data": metrics.snapshot()}


@router.get("/monitoring/model", dependencies=[Depends(require(VIEW))])
def model_monitoring(backtest_product_id: Optional[str] = None, horizon_days: int = Query(14, ge=1, le=60)):
    """The demand model's health: inference count, latency, version, missing-feature rate, drift warnings, and prediction
    error. With `backtest_product_id` it first scores the model against the last `horizon_days` of that product's
    history (a forecast made from before them, compared with what happened) and records the result."""
    result = None
    if backtest_product_id is not None:
        if backtest_product_id not in inventory_tools.get_product_ids():
            raise ApiError(422, "VALIDATION_ERROR", f"unknown product_id {backtest_product_id!r}; GET /api/products lists the products")
        try:
            result = monitor.backtest(backtest_product_id, horizon_days)
        except ValueError as exc:
            raise ApiError(422, "VALIDATION_ERROR", str(exc)) from exc
    return {"data": {**monitor.snapshot(), "backtest": result}}
