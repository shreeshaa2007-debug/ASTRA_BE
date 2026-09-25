"""Inventory Agent tool functions — per agent-plan.md: get_inventory(),
forecast_demand(), calculate_stockout_risk(), calculate_transfer_recommendation().
Each is a plain, typed, independently testable function; the agent (agent.py)
composes them. None of these write anything — read-only lookups and pure
calculations, per the brief's "the agent should NOT directly execute
inventory transfers, it produces a structured recommendation."
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from backend.models.forecasting.predictor import load_predictor
from backend.monitoring.model_monitor import monitor

INVENTORY_PATH = Path("data/processed/inventory_multi_warehouse.csv")
MODELING_PANEL_PATH = Path("data/processed/demand_modeling_panel.csv")

# location_id the forecasting model was actually trained on (model-plan.md /
# ml/training) — always "United Kingdom" in this dataset, distinct from the
# synthetic warehouse_id values (Mumbai/Chennai/Delhi) inventory is split
# across. See inventory.py's WAREHOUSE_PROFILES module note.
TRAINED_LOCATION_ID = "United Kingdom"

HIGH_RISK_DAYS = 7
MEDIUM_RISK_DAYS = 14
DEFAULT_FORECAST_HORIZON_DAYS = 14
TRANSFER_TARGET_SAFETY_MULTIPLE = 1.5  # bring a deficit warehouse up to 1.5x its own safety stock, not further


@lru_cache(maxsize=1)
def _load_inventory() -> pd.DataFrame:
    df = pd.read_csv(INVENTORY_PATH, parse_dates=["date"])
    df["product_id"] = df["product_id"].astype(str)
    return df


@lru_cache(maxsize=1)
def _load_demand_panel() -> pd.DataFrame:
    df = pd.read_csv(MODELING_PANEL_PATH, parse_dates=["date"], usecols=["date", "product_id", "location_id", "demand_quantity"])
    df["product_id"] = df["product_id"].astype(str)
    return df


def get_product_ids() -> list[str]:
    """Every product the inventory ledger tracks — the universe a sensed
    disruption's `affected_products` is validated against."""
    return sorted(_load_inventory()["product_id"].unique())


def _resolve_as_of_date(as_of_date: str | pd.Timestamp | None) -> pd.Timestamp:
    inv = _load_inventory()
    if as_of_date is None:
        return inv["date"].max()
    return pd.Timestamp(as_of_date)


def get_inventory(warehouse_id: str, product_id: str, as_of_date: str | pd.Timestamp | None = None) -> dict:
    """Current stock snapshot for one (warehouse, product) as of a date
    (default: the latest date in the ledger)."""
    as_of = _resolve_as_of_date(as_of_date)
    inv = _load_inventory()
    row = inv[(inv["warehouse_id"] == warehouse_id) & (inv["product_id"] == str(product_id)) & (inv["date"] == as_of)]
    if row.empty:
        raise ValueError(f"No inventory record for warehouse={warehouse_id!r}, product={product_id!r}, date={as_of.date()}")
    r = row.iloc[0]
    return {
        "warehouse_id": warehouse_id,
        "product_id": str(product_id),
        "as_of_date": str(as_of.date()),
        "current_stock": int(r["closing_stock"]),
        "safety_stock": int(r["safety_stock"]),
        "stockout_flag": bool(r["stockout_flag"]),
    }


def get_inventory_snapshot(as_of_date: str | pd.Timestamp | None = None, product_id: str | None = None) -> list[dict]:
    """`get_inventory()` for every (warehouse, product) on one date at once
    (default: the latest date in the ledger), optionally for one product."""
    as_of = _resolve_as_of_date(as_of_date)
    inv = _load_inventory()
    rows = inv[inv["date"] == as_of]
    if product_id is not None:
        rows = rows[rows["product_id"] == str(product_id)]
    return [
        {
            "warehouse_id": r.warehouse_id,
            "product_id": r.product_id,
            "as_of_date": str(as_of.date()),
            "current_stock": int(r.closing_stock),
            "safety_stock": int(r.safety_stock),
            "stockout_flag": bool(r.stockout_flag),
        }
        for r in rows.sort_values(["product_id", "warehouse_id"]).itertuples()
    ]


def get_recent_demand(product_id: str, as_of_date: str | pd.Timestamp | None = None, days: int = 28) -> list[dict]:
    """The last `days` days of actual demand for `product_id` up to `as_of_date`
    (UK-aggregate, the level the model was trained on): [{"date", "actual"}, ...]."""
    as_of = _resolve_as_of_date(as_of_date)
    panel = _load_demand_panel()
    history = panel[(panel["product_id"] == str(product_id)) & (panel["date"] <= as_of)].sort_values("date").tail(days)
    if history.empty:
        raise ValueError(f"No demand history for product_id={product_id!r} as of {as_of.date()}")
    return [{"date": str(r.date.date()), "actual": float(r.demand_quantity)} for r in history.itertuples()]


def forecast_series(
    product_id: str,
    as_of_date: str | pd.Timestamp | None = None,
    horizon_days: int = DEFAULT_FORECAST_HORIZON_DAYS,
    predictor_version: str = "2026.09.1",
) -> list[dict]:
    """The model's day-by-day forecast for `product_id` over the next
    `horizon_days`, at the level it was trained on (UK-aggregate demand):
    [{"date": "YYYY-MM-DD", "predicted": float}, ...].

    Multi-day horizons are produced recursively: predict day 1, append that
    prediction to the lag window, predict day 2, and so on. This is a
    standard way to extend a 1-step-ahead model, not a hidden shortcut — but
    it does mean error compounds with horizon length, which is exactly why
    horizon_days defaults to a modest 14, not 90.
    """
    predictor = load_predictor(predictor_version)
    as_of = _resolve_as_of_date(as_of_date)

    panel = _load_demand_panel()
    history = panel[(panel["product_id"] == str(product_id)) & (panel["date"] <= as_of)].sort_values("date")
    if history.empty:
        raise ValueError(f"No demand history for product_id={product_id!r} as of {as_of.date()}")

    recent = list(history["demand_quantity"].tail(28).values)
    monitor.check_input_drift(str(product_id), recent)  # the actual window this forecast starts from, before any predicted value enters it
    series = []
    cursor_date = as_of
    for _ in range(horizon_days):
        cursor_date = cursor_date + pd.Timedelta(days=1)
        pred = predictor.predict_one(
            product_id=str(product_id),
            location_id=TRAINED_LOCATION_ID,
            recent_demand=recent,
            day_of_week=cursor_date.dayofweek,
            month=cursor_date.month,
        )
        series.append({"date": str(cursor_date.date()), "predicted": float(pred)})
        recent = recent[1:] + [pred]  # slide the window forward with our own prediction
    return series


def forecast_demand(
    product_id: str,
    warehouse_id: str,
    as_of_date: str | pd.Timestamp | None = None,
    horizon_days: int = DEFAULT_FORECAST_HORIZON_DAYS,
    predictor_version: str = "2026.09.1",
) -> dict:
    """Forecasts total demand for `product_id` at `warehouse_id` over the next
    `horizon_days`. The underlying XGBoost model (backend/models/forecasting)
    was trained on real UK-aggregate demand — it has no concept of the
    synthetic warehouses — so this forecasts at the product level against
    TRAINED_LOCATION_ID (see forecast_series), then allocates to `warehouse_id`
    by the same fixed demand_share used to build the inventory ledger
    (inventory.py's WAREHOUSE_PROFILES), keeping the split consistent
    everywhere it's used.
    """
    from backend.services.preprocessing.inventory import WAREHOUSE_PROFILES  # local import avoids a cycle at module load

    if warehouse_id not in WAREHOUSE_PROFILES:
        raise ValueError(f"Unknown warehouse_id {warehouse_id!r}; known: {list(WAREHOUSE_PROFILES)}")

    total_uk_forecast = sum(day["predicted"] for day in forecast_series(product_id, as_of_date, horizon_days, predictor_version))
    warehouse_forecast = total_uk_forecast * WAREHOUSE_PROFILES[warehouse_id]["demand_share"]
    return {
        "product_id": str(product_id),
        "warehouse_id": warehouse_id,
        "horizon_days": horizon_days,
        "forecast_demand": round(warehouse_forecast, 1),
        "model_version": load_predictor(predictor_version).version,
    }


def calculate_stockout_risk(current_stock: float, forecast_demand: float, safety_stock: float, horizon_days: int = DEFAULT_FORECAST_HORIZON_DAYS) -> str:
    """Days-of-cover thresholding: <7 days HIGH, 7-14 MEDIUM, >14 LOW — a
    stockout-imminent warehouse reads as HIGH even if `current_stock` is
    technically above `safety_stock`, because safety_stock is a reorder
    trigger, not a promise nothing runs out before the next delivery.
    """
    avg_daily_demand = max(forecast_demand / horizon_days, 1e-6)
    days_of_cover = current_stock / avg_daily_demand
    if days_of_cover < HIGH_RISK_DAYS:
        return "HIGH"
    if days_of_cover < MEDIUM_RISK_DAYS:
        return "MEDIUM"
    return "LOW"


def calculate_transfer_recommendation(warehouse_snapshots: list[dict]) -> dict | None:
    """`warehouse_snapshots`: one dict per warehouse for the SAME product,
    each with warehouse_id, current_stock, safety_stock, stockout_risk (the
    output of get_inventory() + calculate_stockout_risk() combined — agent.py
    builds this list). Returns {"from": ..., "quantity": ...} or None if no
    warehouse is at HIGH risk, or no other warehouse has transferable
    surplus (stock above its own safety_stock) to cover it.

    Picks the single highest-risk (lowest days-of-cover proxy: current_stock
    - safety_stock, most negative first) deficit warehouse and the single
    largest-surplus source — not a multi-way optimization (that's the
    Optimization Engine's job in Phase 10, once logistics/sourcing
    constraints are also in play; this tool gives it one clean candidate).
    """
    high_risk = [w for w in warehouse_snapshots if w["stockout_risk"] == "HIGH"]
    if not high_risk:
        return None
    deficit = min(high_risk, key=lambda w: w["current_stock"] - w["safety_stock"])

    candidates = [w for w in warehouse_snapshots if w["warehouse_id"] != deficit["warehouse_id"]]
    surpluses = [(w, w["current_stock"] - w["safety_stock"]) for w in candidates]
    surpluses = [(w, s) for w, s in surpluses if s > 0]
    if not surpluses:
        return None
    source, surplus = max(surpluses, key=lambda ws: ws[1])

    target_stock = round(deficit["safety_stock"] * TRANSFER_TARGET_SAFETY_MULTIPLE)
    needed = max(0, target_stock - deficit["current_stock"])
    quantity = int(min(needed, surplus))
    if quantity <= 0:
        return None

    return {"from": source["warehouse_id"], "to": deficit["warehouse_id"], "quantity": quantity}
