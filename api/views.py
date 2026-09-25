"""Response builders — pure functions from world state (and the processed
datasets) to JSON-ready dicts. No HTTP in here, so each is testable on its own.

Two rules run through everything:
  * Nothing is made up to fill a field. Where the data does not exist (there is no
    shipment dataset; "estimated exposure" needs a baseline-cost comparison that
    belongs to scenario simulation) the field is null and `unavailable` says why.
  * Anything derived from the optimizer carries `engine` and `label`
    ("Prototype Optimization") verbatim, so the frontend renders the label
    instead of inferring which engine ran (api-plan.md).
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from backend.agents.inventory import tools as inventory_tools
from backend.agents.logistics import tools as logistics_tools
from backend.agents.sourcing import tools as sourcing_tools
from backend.api.errors import ApiError
from backend.api.models import ForecastRequest
from backend.models.forecasting.predictor import load_predictor
from backend.optimization import tools as optimization_tools
from backend.orchestration import adapters
from backend.schemas.entities import RouteStatus, SupplierStatus
from backend.schemas.world_state import ApprovalStatus, CheckpointRecord, SimulationStatus, WorldState
from backend.services.world_state import load_baseline
from backend.simulation import compare_state

logger = logging.getLogger("resilientsc.api")

DISRUPTIONS_PATH = Path("data/processed/disruptions.csv")


def clean(value: Any) -> Any:
    """JSON-safe: NaN -> None (Starlette refuses NaN), numpy scalars -> Python."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return clean(value.item())
    return value


def plan_labels(state: WorldState) -> dict:
    return {"engine": state.current_plan.engine, "label": state.current_plan.label} if state.current_plan else {}


# --------------------------------------------------------------------------- #
# simulations
# --------------------------------------------------------------------------- #
def state_view(state: WorldState) -> dict:
    return {**state.model_dump(mode="json"), **plan_labels(state)}


_STEP_AFTER = {  # the last checkpoint written -> what the run is doing now
    "simulation_created": "sensing", "simulation_reset": "sensing", "event_sensed": "agents", "agents_assessed": "optimization",
    "plan_optimized": "compliance", "compliance_checked": "decision", "replan_requested": "agents",
}
_STAGES = [
    ("sensing", "Sensing Agent"), ("inventory", "Inventory Agent"), ("logistics", "Logistics Agent"), ("sourcing", "Sourcing Agent"),
    ("optimization", "Optimization Engine"), ("compliance", "Compliance Agent"), ("human_approval", "Human Approval"),
]
_STEP_FOR_STAGE = {"sensing": "sense", "optimization": "optimize", "compliance": "compliance",
                   "inventory": "agents", "logistics": "agents", "sourcing": "agents"}


def current_step(state: WorldState, last_checkpoint: str | None, running: bool) -> str:
    if state.status == SimulationStatus.CREATED:
        return "sensing" if running else "idle"
    if state.status == SimulationStatus.RUNNING:
        return _STEP_AFTER.get(last_checkpoint or "", "unknown")
    return {"AWAITING_APPROVAL": "approval", "COMPLETED": "done", "REJECTED": "rejected", "FAILED": "failed"}[state.status.value]


def agent_statuses(state: WorldState, run: dict | None, active: bool) -> list[dict]:
    """Per-agent status — the Agent Monitor's data — derived from the world state and how
    the run ended. No agent reports its own status; this is what the record shows happened.

    Deliberately from the state alone, not the checkpoint history: a poll can land while a
    run is committing, and the state and the history are read separately, so mixing them
    can disagree (the history saying an event was sensed before the state has it). The
    state is one consistent snapshot; a reset also clears it, so no "since the last reset"
    bookkeeping is needed."""
    outcome = (run or {}).get("outcome") or {}
    steps = {s["name"]: s for s in outcome.get("steps", [])}
    failed = state.status == SimulationStatus.FAILED
    sensed, assessed = bool(state.current_disruptions), bool(state.inventory_status)
    plan, verdict = state.current_plan, state.compliance_status

    def status_of(stage: str) -> tuple[str, str]:
        if stage == "sensing":
            if sensed:
                event = state.current_disruptions[-1]
                return "COMPLETE", f"{event.event_type} at {event.location} (confidence {event.confidence})"
            if active:
                return "RUNNING", ""
            if outcome.get("outcome") == "NO_DISRUPTION":
                return "NO_EVENT", outcome.get("message", "")
            if outcome.get("outcome") in ("SENSING_REJECTED", "SENSING_ERROR"):
                return "FAILED", outcome.get("message", "")
            return "PENDING", ""
        if stage in ("inventory", "logistics", "sourcing"):
            if assessed:
                return "COMPLETE", ""
            return ("RUNNING", "") if sensed and active else (("FAILED", state.error or "") if sensed and failed else ("PENDING", ""))
        if stage == "optimization":
            if plan is not None:
                return ("COMPLETE" if plan.status == "OPTIMAL" else "FAILED"), plan.message
            if assessed and active:
                return "RUNNING", ""
            return ("FAILED", state.error or "") if assessed and failed else ("PENDING", "")
        if stage == "compliance":
            if verdict is not None:
                return "COMPLETE", f"{verdict.status}: {verdict.reason}"
            return ("RUNNING", "") if plan is not None and plan.status == "OPTIMAL" and active else ("PENDING", "")
        # human_approval
        return {
            ApprovalStatus.PENDING: ("ACTION_REQUIRED", "waiting for a human decision"),
            ApprovalStatus.APPROVED: ("COMPLETE", f"approved by {state.approval_decision.decided_by}" if state.approval_decision else "approved"),
            ApprovalStatus.REJECTED: ("REJECTED", f"rejected by {state.approval_decision.decided_by}" if state.approval_decision else "rejected"),
            ApprovalStatus.NOT_REQUIRED: ("NOT_REQUIRED", "in-policy: no human decision needed"),
        }.get(state.approval_status, ("PENDING", ""))

    result = []
    for stage, name in _STAGES:
        status, detail = status_of(stage)
        step = steps.get(_STEP_FOR_STAGE.get(stage, ""))
        result.append({"id": stage, "name": name, "status": status, "detail": detail, "latency_ms": step["duration_ms"] if step else None})
    return result


def read_status(store, runs, simulation_id: str) -> dict:
    """Reads what the status endpoints need, in the one order that keeps a response coherent while a
    run is committing on another thread:

      1. `running`, then the run record *snapshotted to a dict* — a run registers as active before it
         changes any state, and it is marked FINISHED only after its last commit;
      2. only then the state and the history.

    So the only skew possible is a run that looks *older* than the state ("RUNNING" beside a COMPLETED
    state) — the next poll fixes it. The reverse — a FINISHED run beside a state that is still RUNNING —
    cannot happen. (Both were real: snapshotting the run late produced exactly that, and a state
    read before `running` could report a run that had just finished as stalled.)"""
    running = runs.is_running(simulation_id)
    record = runs.latest(simulation_id)
    run = record.to_dict() if record else None
    state = store.get(simulation_id)
    return status_view(state, store.history(simulation_id), run, running)


def status_view(state: WorldState, history: list[CheckpointRecord], run: dict | None, running: bool) -> dict:
    """`run` is a snapshot (RunRecord.to_dict()), taken by read_status before the state was read."""
    stalled = state.status == SimulationStatus.RUNNING and not running  # e.g. the server restarted mid-run
    active = running or (state.status == SimulationStatus.RUNNING and not stalled)
    return {
        "simulation_id": state.simulation_id, "scenario_type": state.scenario_type, "status": state.status.value, "version": state.version,
        "current_step": current_step(state, history[-1].checkpoint if history else None, running),
        "awaiting_approval": state.status == SimulationStatus.AWAITING_APPROVAL,
        "stalled": stalled, "error": state.error, "replan_count": state.replan_count,
        "run": run,
        "agents": agent_statuses(state, run, active),
        "timeline": [h.model_dump(mode="json") for h in history],
    }


# --------------------------------------------------------------------------- #
# decisions and compliance
# --------------------------------------------------------------------------- #
def _approval(state: WorldState) -> dict:
    return {"status": state.approval_status.value, "decision": state.approval_decision.model_dump(mode="json") if state.approval_decision else None}


def decision_view(state: WorldState) -> dict:
    plan = state.current_plan
    if plan is None:
        raise ApiError(409, "PLAN_NOT_READY", f"simulation {state.simulation_id!r} is {state.status.value}; no plan has been produced yet")
    metrics = optimization_tools.get_optimization_metrics(plan)
    optimal = plan.status == "OPTIMAL"
    return clean({
        "simulation_id": state.simulation_id, "simulation_status": state.status.value, "version": state.version,
        "engine": plan.engine, "label": plan.label, "plan_status": plan.status, "message": plan.message, "product_id": plan.problem.product_id,
        "objective_value": plan.objective_value, "objective_terms": plan.objective_terms, "plan_spend": adapters.plan_spend(plan) if optimal else None,
        "allocations": [a.model_dump(mode="json") for a in plan.allocations], "transfers": [t.model_dump(mode="json") for t in plan.transfers],
        "inbound_by_warehouse": plan.inbound_by_warehouse, "end_stock_by_warehouse": plan.end_stock_by_warehouse,
        "mode_split": metrics.get("mode_split"), "supplier_split": metrics.get("supplier_split"), "avg_arrival_days": metrics.get("avg_arrival_days"),
        "avg_all_in_unit_cost": metrics.get("avg_all_in_unit_cost"), "weighted_reliability": metrics.get("weighted_reliability"),
        "constraints": [c.model_dump(mode="json") for c in plan.constraint_status], "binding_constraints": metrics.get("binding_constraints", []),
        "decision_factors": plan.decision_factors, "deviations": plan.deviations, "assumptions": plan.problem.assumptions,
        "excluded_options": [e.model_dump(mode="json") for e in plan.excluded_options], "diagnostics": plan.diagnostics, "solver": plan.solver,
        "disruptions": [{"event_id": d.event_id, "event_type": d.event_type, "location": d.location, "severity": d.severity.value, "summary": d.summary}
                        for d in state.current_disruptions],
        "compliance": state.compliance_status.model_dump(mode="json") if state.compliance_status else None,
        "approval": _approval(state), "replan_count": state.replan_count,
    })


def compliance_view(state: WorldState) -> dict:
    verdict = state.compliance_status
    if verdict is None:
        raise ApiError(409, "PLAN_NOT_READY", f"simulation {state.simulation_id!r} is {state.status.value}; the plan has not been through compliance yet")
    optimal = state.current_plan is not None and state.current_plan.status == "OPTIMAL"
    return clean({
        "simulation_id": state.simulation_id, "simulation_status": state.status.value, "version": state.version,
        "compliance": verdict.model_dump(mode="json"), "approval": _approval(state), "replan_count": state.replan_count,
        "plan_spend": adapters.plan_spend(state.current_plan) if optimal else None, **plan_labels(state),
    })


# --------------------------------------------------------------------------- #
# dashboard
# --------------------------------------------------------------------------- #
def _exposure(latest: WorldState | None) -> tuple[dict, dict]:
    """(kpis, reasons): what the latest finalized simulation's disruption puts at risk in the *modeled baseline plan*
    (docs/api-plan.md: there is no shipment dataset, so these are never observed shipments). Anything that
    cannot be computed stays null and says why."""
    null = {"shipments_at_risk": None, "units_at_risk": None, "estimated_exposure": None}
    if latest is None:
        why = "no simulation has been finalized yet"
        return null, {"shipments_at_risk": why, "estimated_exposure": why}
    if latest.current_plan is None or latest.current_plan.status != "OPTIMAL":
        why = "the latest simulation has no feasible plan to compare against normal operations"
        return null, {"shipments_at_risk": why, "estimated_exposure": why}
    try:
        comparison = compare_state(latest)
        exposure = comparison.exposure
    except Exception as exc:  # noqa: BLE001 — a KPI that cannot be computed is reported, it must not take the dashboard down
        logger.exception("could not compute exposure for %s", latest.simulation_id)
        why = f"could not be computed: {type(exc).__name__}: {exc}"
        return null, {"shipments_at_risk": why, "estimated_exposure": why}
    if comparison.warnings:  # the run used an earlier as-of date than the ledger's latest: the baseline would be on different stock
        why = "the simulation was run on an earlier as-of date than the ledger's latest, so its baseline cannot be reconstructed here; GET /api/simulations/{id}/comparison?as_of_date=... has the figures"
        return null, {"shipments_at_risk": why, "estimated_exposure": why}
    if exposure is None:
        why = "there is no feasible plan for normal operations for this product, so there is no baseline plan to put at risk"
        return null, {"shipments_at_risk": why, "estimated_exposure": why}
    return (
        {"shipments_at_risk": exposure.shipments_at_risk, "units_at_risk": exposure.units_at_risk, "estimated_exposure": exposure.value_at_risk},
        {},
    )


def dashboard_view(latest: WorldState | None) -> dict:
    if latest is None:
        base = load_baseline()
        routes, suppliers, events, inventory, plan = base.route_status, base.supplier_status, [], [], None
        reflects = "the baseline network: no simulation has been finalized yet"
    else:
        routes, suppliers, events, inventory, plan = latest.route_status, latest.supplier_status, latest.current_disruptions, latest.inventory_status, latest.current_plan
        reflects = "the latest finalized simulation"
    active = sum(1 for s in suppliers.values() if s == SupplierStatus.ACTIVE)
    exposure_kpis, unavailable = _exposure(latest)
    return clean({
        "simulation_id": latest.simulation_id if latest else None, "reflects": reflects,
        **({"engine": plan.engine, "label": plan.label} if plan else {}),
        "kpis": {
            "active_disruptions": len(events),
            "routes_total": len(routes), "routes_disrupted": sum(1 for s in routes.values() if s == RouteStatus.DISRUPTED),
            "suppliers_total": len(suppliers), "suppliers_disrupted": sum(1 for s in suppliers.values() if s == SupplierStatus.DISRUPTED),
            "suppliers_reduced": sum(1 for s in suppliers.values() if s == SupplierStatus.REDUCED),
            "supplier_health_pct": round(100 * active / len(suppliers), 1) if suppliers else None,  # the share that are fully ACTIVE
            "inventory_records": len(inventory), "inventory_at_risk": sum(1 for a in inventory if a.stockout_risk in ("HIGH", "MEDIUM")),
            "plan_status": plan.status if plan else None, "plan_objective_value": plan.objective_value if plan else None,
            "plan_spend": adapters.plan_spend(plan) if plan and plan.status == "OPTIMAL" else None,
            "approval_status": latest.approval_status.value if latest else None,
            **exposure_kpis,
        },
        "unavailable": unavailable,
        "exposure_note": (
            "shipments_at_risk / estimated_exposure are the purchases in the modeled baseline plan (normal operations, same product) that the "
            "disruption invalidates, and the baseline plan's own price for them in MVP cost units. There is no shipment dataset, so they are "
            "never observed shipments, and not a loss estimate."
        ),
    })


# --------------------------------------------------------------------------- #
# network tables: routes, suppliers, inventory, shipments
# --------------------------------------------------------------------------- #
def _plan_of(state: WorldState | None):
    return state.current_plan if state and state.current_plan and state.current_plan.status == "OPTIMAL" else None


def route_rows(state: WorldState | None) -> dict:
    plan = _plan_of(state)
    planned: dict[str, int] = defaultdict(int)
    for a in (plan.allocations if plan else []):
        if a.route_id:
            planned[a.route_id] += a.quantity
    rows = []
    for r in logistics_tools.get_routes():
        status = state.route_status[r["route_id"]].value if state and r["route_id"] in state.route_status else r["status"]
        rows.append({**r, "status": status, "planned_quantity": planned.get(r["route_id"], 0) if plan else None})
    return clean({"simulation_id": state.simulation_id if state else None, "routes": rows, **(plan_labels(state) if plan else {})})


def supplier_rows(state: WorldState | None, product_id: str | None) -> dict:
    plan = _plan_of(state)
    if product_id is None and plan:
        product_id = plan.problem.product_id
    tariffs = state.tariffs if state else sourcing_tools.get_latest_tariffs()
    by_supplier: dict[str, int] = defaultdict(int)
    for a in (plan.allocations if plan else []):
        by_supplier[a.supplier_id] += a.quantity
    plan_product = plan.problem.product_id if plan else None
    total = sum(by_supplier.values())

    rows = []
    for s in sourcing_tools.get_suppliers(product_id=product_id):
        status = state.supplier_status[s["supplier_id"]].value if state and s["supplier_id"] in state.supplier_status else s["status"]
        iso3 = sourcing_tools.REGION_TO_ISO3.get(s["region"])
        in_plan = plan is not None and s["product_id"] == plan_product
        rows.append({**s, "status": status, "tariff_rate_pct": tariffs.get(iso3) if iso3 else None,
                     "recommended_quantity": by_supplier.get(s["supplier_id"], 0) if in_plan else None})
    mix = [{"supplier_id": sid, "quantity": q, "share": round(q / total, 4)} for sid, q in sorted(by_supplier.items())] if plan and total else None
    return clean({"simulation_id": state.simulation_id if state else None, "product_id": product_id, "suppliers": rows,
                  "recommended_mix": mix, **(plan_labels(state) if plan else {})})


def inventory_rows(state: WorldState | None, product_id: str | None) -> dict:
    if state is None or not state.inventory_status:
        rows = [{**r, "source": "ledger", "forecast_demand": None, "horizon_days": None, "stockout_risk": None, "days_of_cover": None, "recommended_transfer": None}
                for r in inventory_tools.get_inventory_snapshot(product_id=product_id)]
        return clean({"simulation_id": None, "source": "ledger", "inventory": rows})

    horizon = {(f.warehouse_id, f.product_id): f.horizon_days for f in state.demand_forecasts}
    plan = state.current_plan
    safety = {w.warehouse_id: w.safety_stock for w in plan.problem.warehouses} if plan else {}
    rows = []
    for a in state.inventory_status:
        if product_id is not None and a.product != product_id:
            continue
        days = horizon.get((a.warehouse, a.product))
        cover = round(a.current_stock / (a.forecast_demand / days), 1) if days and a.forecast_demand > 0 else None
        rows.append({"warehouse_id": a.warehouse, "product_id": a.product, "current_stock": a.current_stock, "safety_stock": safety.get(a.warehouse),
                     "forecast_demand": a.forecast_demand, "horizon_days": days, "stockout_risk": a.stockout_risk, "days_of_cover": cover,
                     "recommended_transfer": a.recommended_transfer, "source": "simulation"})
    model_versions = sorted({f.model_version for f in state.demand_forecasts})
    return clean({"simulation_id": state.simulation_id, "source": "simulation", "inventory": rows, "model_version": model_versions[0] if model_versions else None})


def shipment_rows(state: WorldState | None, route_id: str | None, status: str | None) -> dict:
    """There is no shipment dataset, so there are no observed shipments. What can be
    listed honestly is what the optimizer's plan proposes: one entry per
    supplier-and-route allocation, marked PROPOSED until the plan is finalized."""
    plan = _plan_of(state)
    shipments = []
    if plan:
        shipped = "PLANNED" if state.status == SimulationStatus.COMPLETED else "PROPOSED"
        for i, a in enumerate(plan.allocations, start=1):
            shipments.append({
                "shipment_id": f"{state.simulation_id}-{i}", "product_id": plan.problem.product_id, "supplier_id": a.supplier_id,
                "route_id": a.route_id, "transport_mode": a.transport_mode, "quantity": a.quantity, "arrival_days": a.arrival_days,
                "freight_unit_cost": a.freight_unit_cost, "status": shipped, "provenance": "derived from the optimizer's plan, not observed",
            })
    if route_id is not None:
        shipments = [s for s in shipments if s["route_id"] == route_id]
    if status is not None:
        shipments = [s for s in shipments if s["status"] == status.upper()]
    return clean({
        "simulation_id": state.simulation_id if state else None, "shipments": shipments,
        "note": "no shipment dataset exists; these are the shipments the plan proposes", **(plan_labels(state) if plan else {}),
    })


# --------------------------------------------------------------------------- #
# disruptions
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _historical() -> pd.DataFrame:
    df = pd.read_csv(DISRUPTIONS_PATH, parse_dates=["start_date", "end_date"])
    df["source"] = df["event_id"].astype(str).str.startswith("CURATED").map({True: "curated", False: "noaa"})
    return df.sort_values("start_date", ascending=False).reset_index(drop=True)


def historical_disruptions(source: str, severity: str | None, limit: int, offset: int) -> dict:
    df = _historical()
    if source != "all":
        df = df[df["source"] == source]
    if severity:
        df = df[df["severity"] == severity.upper()]
    rows = []
    for r in df.iloc[offset: offset + limit].to_dict("records"):
        rows.append({**r, "start_date": r["start_date"].isoformat(), "end_date": r["end_date"].isoformat() if pd.notna(r["end_date"]) else None,
                     "provenance": "curated real-world event" if r["source"] == "curated" else "real: NOAA Storm Events (weather only)"})
    return {"total": int(len(df)), "limit": limit, "offset": offset, "events": clean(rows)}


def disruptions_view(state: WorldState | None, source: str, severity: str | None, limit: int, offset: int) -> dict:
    return {
        "simulation_id": state.simulation_id if state else None,
        "active": [d.model_dump(mode="json") for d in state.current_disruptions] if state else [],
        "historical": historical_disruptions(source, severity, limit, offset),
    }


# --------------------------------------------------------------------------- #
# forecasts
# --------------------------------------------------------------------------- #
def forecast_for_warehouse(product_id: str, warehouse_id: str, horizon_days: int, as_of_date: str | None) -> dict:
    from backend.services.preprocessing.inventory import WAREHOUSE_PROFILES  # local import avoids a cycle at module load

    if warehouse_id not in WAREHOUSE_PROFILES:
        raise ApiError(422, "VALIDATION_ERROR", f"unknown warehouse_id {warehouse_id!r}; known: {sorted(WAREHOUSE_PROFILES)}")
    share = WAREHOUSE_PROFILES[warehouse_id]["demand_share"]
    try:
        predicted = inventory_tools.forecast_series(product_id, as_of_date, horizon_days)
        actual = inventory_tools.get_recent_demand(product_id, as_of_date)
    except ValueError as exc:
        raise ApiError(422, "VALIDATION_ERROR", str(exc)) from exc
    return clean({
        "product_id": product_id, "warehouse_id": warehouse_id, "horizon_days": horizon_days, "as_of_date": actual[-1]["date"],
        "actual": [{"date": d["date"], "value": round(d["actual"] * share, 2)} for d in actual],
        "predicted": [{"date": d["date"], "value": round(d["predicted"] * share, 2)} for d in predicted],
        "model_version": load_predictor().version,
        "confidence": None,
        "note": (f"point forecast only — XGBoost gives no interval, so no band is drawn. The model is trained on UK-aggregate demand; the warehouse series is "
                 f"that series times the fixed demand share {share}, the same split the inventory ledger uses."),
    })


def forecast_from_history(req: ForecastRequest) -> dict:
    """model-plan.md §6: forecast from demand history the caller supplies."""
    predictor = load_predictor()
    start = req.start_date or (date.today() + timedelta(days=1))
    recent = list(req.recent_demand)[-28:]
    dates, predicted = [], []
    try:
        for i in range(req.forecast_horizon):
            day = start + timedelta(days=i)
            value = float(predictor.predict_one(req.product_id, req.location_id, recent, day.weekday(), day.month))
            dates.append(day.isoformat())
            predicted.append(round(value, 3))
            recent = (recent + [value])[-28:]  # the window slides on the model's own predictions
    except ValueError as exc:
        raise ApiError(422, "VALIDATION_ERROR", str(exc)) from exc
    return {
        "product_id": req.product_id, "location_id": req.location_id, "dates": dates, "predicted_demand": predicted,
        "model_version": predictor.version, "confidence": None, "note": "point forecast; recursive multi-step, so error grows with the horizon",
    }
