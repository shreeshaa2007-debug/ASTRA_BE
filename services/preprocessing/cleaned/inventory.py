"""Inventory regenerated from the corrected demand: a SIMULATED ledger, one row per (warehouse, product, day).

No public warehouse ledger exists (docs/data-plan.md §C), so stock is a reorder-point simulation run over real, corrected net
demand. It is labelled SYNTHETIC: the demand shape is real, but the starting stock, the policy and the split across warehouses are
assumptions (A09, A10). What changed from Phase 3, and why:

* the key is unique BY CONSTRUCTION: the ledger is a dense grid of warehouse x product x calendar day. Phase 3 wrote one row per
  demand row (per country) under a constant warehouse id, so (warehouse, product, date) repeated 27,901 times;
* lost sales are recorded, not hidden: Phase 3 set outbound = demand and clipped stock at zero, so a stock-out silently
  vanished. Here `outbound_quantity` is what was actually shipped, `unfilled_quantity` is what was not, and
  closing = opening + inbound - outbound holds exactly with no clipping;
* integer demand is split across warehouses by largest remainder, so the warehouses' demand always sums to the product's demand
  (Phase 3 rounded each share separately and lost or invented units);
* the ledger covers the modeling panel's products (the ones forecasts and suppliers refer to), not all 3,800: most of the
  catalogue is intermittent and has no meaningful stock story.

The policy is Phase 3's (A09): seed stock, safety stock and the replenishment quantity are sized on the whole year's mean demand,
so this is a scenario generator, not a backtest of an inventory policy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.services.preprocessing.cleaned import SYNTHETIC
from backend.services.preprocessing.cleaned import assumptions as A

WAREHOUSES = {  # id -> (city, ISO-3, index of the nearest World Port Index port or None for an inland city)
    "Mumbai": ("Mumbai", "IND", 48840),
    "Chennai": ("Chennai", "IND", 49450),
    "Delhi": ("Delhi", "IND", None),
}


def allocate_demand(demand: np.ndarray, shares: list[float]) -> np.ndarray:
    """Splits integer daily demand across warehouses (rows) so each day's column sums to that day's demand exactly.

    Each warehouse gets the floor of its share; the units left over go, one each, to the warehouses with the largest fractional
    parts (ties go to the earlier warehouse in `WAREHOUSE_ORDER`, so the split is reproducible).
    """
    demand = np.asarray(demand, dtype="int64")
    exact = np.asarray(shares, dtype="float64")[:, None] * demand[None, :]
    base = np.floor(exact + 1e-9).astype("int64")
    left = demand - base.sum(axis=0)
    assert (left >= 0).all() and (left < len(shares)).all(), "demand shares must sum to 1"
    out = base.copy()
    for day in np.flatnonzero(left):
        order = np.argsort(-(exact[:, day] - base[:, day]), kind="stable")
        out[order[: left[day]], day] += 1
    return out


def simulate(demand: np.ndarray, cover_multiplier: float, safety_multiplier: float) -> dict[str, np.ndarray]:
    """One warehouse-product ledger over the days of `demand` (integer units per day)."""
    mean_daily = max(float(demand.mean()), 0.1)
    safety = round(mean_daily * A.SAFETY_DAYS * safety_multiplier)
    replenish = round(mean_daily * A.REPLENISH_DAYS)
    closing = round(mean_daily * A.TARGET_DAYS_OF_COVER * cover_multiplier)

    n = len(demand)
    opening_a, inbound_a, outbound_a, closing_a, unfilled_a = (np.zeros(n, dtype="int64") for _ in range(5))
    for t in range(n):
        opening = closing
        wanted = int(demand[t])
        inbound = replenish if opening - wanted < safety else 0  # replenishment arrives the day it is triggered (A09)
        available = opening + inbound
        shipped = min(wanted, available)
        closing = available - shipped
        opening_a[t], inbound_a[t], outbound_a[t], closing_a[t], unfilled_a[t] = opening, inbound, shipped, closing, wanted - shipped
    return {"opening": opening_a, "inbound": inbound_a, "outbound": outbound_a, "closing": closing_a, "unfilled": unfilled_a,
            "safety": np.full(n, safety, dtype="int64")}


def build_inventory(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """`panel` is the dense modeling panel: every selected product on every calendar day. Returns (ledger, report)."""
    order = list(A.WAREHOUSE_ORDER)
    shares = [A.WAREHOUSE_PROFILES[w]["demand_share"] for w in order]
    assert abs(sum(shares) - 1.0) < 1e-9, "warehouse demand shares must sum to 1"

    frames = []
    for product_id, series in panel.groupby("product_id", sort=True):
        series = series.sort_values("date")
        assert series["date"].is_unique, f"product {product_id}: the modeling panel must have one row per day"
        split = allocate_demand(series["demand_quantity"].to_numpy(), shares)
        for i, warehouse_id in enumerate(order):
            profile = A.WAREHOUSE_PROFILES[warehouse_id]
            sim = simulate(split[i], profile["cover_multiplier"], profile["safety_multiplier"])
            frames.append(
                pd.DataFrame(
                    {
                        "warehouse_id": warehouse_id, "product_id": product_id, "date": series["date"].to_numpy(),
                        "opening_stock": sim["opening"], "inbound_quantity": sim["inbound"], "outbound_quantity": sim["outbound"],
                        "closing_stock": sim["closing"], "safety_stock": sim["safety"], "reorder_point": sim["safety"],
                        "demand_quantity": split[i], "unfilled_quantity": sim["unfilled"],
                    }
                )
            )
    ledger = pd.concat(frames, ignore_index=True)
    ledger["stockout_flag"] = ledger["unfilled_quantity"] > 0
    ledger["provenance"] = SYNTHETIC
    ledger["warehouse_id"] = pd.Categorical(ledger["warehouse_id"], categories=order, ordered=True)
    ledger = ledger.sort_values(["warehouse_id", "product_id", "date"], kind="stable").reset_index(drop=True)
    ledger["warehouse_id"] = ledger["warehouse_id"].astype(str)

    total_demand = int(ledger["demand_quantity"].sum())
    report = {
        "rows": len(ledger),
        "warehouses": order,
        "products": int(ledger["product_id"].nunique()),
        "days": int(ledger["date"].nunique()),
        "total_demand_units": total_demand,
        "total_outbound_units": int(ledger["outbound_quantity"].sum()),
        "total_unfilled_units": int(ledger["unfilled_quantity"].sum()),
        "unfilled_share_pct": round(100 * ledger["unfilled_quantity"].sum() / max(total_demand, 1), 3),
        "stockout_rows": int(ledger["stockout_flag"].sum()),
        "stockout_rows_by_warehouse": ledger.groupby("warehouse_id")["stockout_flag"].sum().astype(int).to_dict(),
        "replenishment_events": int((ledger["inbound_quantity"] > 0).sum()),
        "products_never_replenished": int((ledger.groupby("product_id")["inbound_quantity"].sum() == 0).sum()),
    }
    return ledger, report


def build_warehouse_table() -> pd.DataFrame:
    rows = [
        {"warehouse_id": wid, "warehouse_name": f"{city} warehouse (SYNTHETIC)", "city": city, "country_iso3": iso3, "nearest_port_id": port, "provenance": SYNTHETIC}
        for wid, (city, iso3, port) in WAREHOUSES.items()
    ]
    return pd.DataFrame(rows).astype({"nearest_port_id": "Int64"})


def build_policy_table() -> pd.DataFrame:
    rows = [
        {
            "warehouse_id": wid, "demand_share": p["demand_share"], "cover_multiplier": p["cover_multiplier"], "safety_multiplier": p["safety_multiplier"],
            "target_days_of_cover": A.TARGET_DAYS_OF_COVER, "safety_days": A.SAFETY_DAYS, "replenish_days": A.REPLENISH_DAYS,
            "replenishment_lead_time_days": A.REPLENISHMENT_LEAD_TIME_DAYS, "provenance": SYNTHETIC,
        }
        for wid, p in A.WAREHOUSE_PROFILES.items()
    ]
    return pd.DataFrame(rows)
