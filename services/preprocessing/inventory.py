"""INVENTORY preprocessing — DERIVED, not real (see docs/data-plan.md § C.
Inventory: no public warehouse ledger dataset exists). Built from the real
demand panel via a documented reorder-point policy, one ledger per
(product_id, location_id) pair from the demand panel, labeled
`Provenance.DERIVED` everywhere it surfaces.

Policy (stated, not hidden):
- opening_stock on day 1 = `target_days_of_cover` x that series' mean daily demand
- reorder point = `safety_days` x mean daily demand (the safety_stock field)
- when closing_stock would drop below the reorder point, an inbound
  replenishment of `replenish_days` x mean daily demand arrives that same day
  (an instant-replenishment simplification — a real system would model lead
  time; this is a Phase 3 preprocessing artifact, not a claim about real
  supplier lead times, which live in suppliers.py once synthesized)
"""
from __future__ import annotations

import pandas as pd

TARGET_DAYS_OF_COVER = 30
SAFETY_DAYS = 10
REPLENISH_DAYS = 21


def derive_inventory_ledger(demand_panel: pd.DataFrame, warehouse_id: str = "DEFAULT") -> tuple[pd.DataFrame, dict]:
    """`demand_panel` must have columns date, product_id, location_id,
    demand_quantity (the output of demand.build_demand_panel). Returns one
    ledger row per (date, product_id) — location_id folded into a single
    `warehouse_id` for the MVP (multi-warehouse allocation is a Phase 6+
    Inventory Agent concern, not a preprocessing one).

    CAVEAT (found in Phase 6, recorded here rather than silently fixed and
    forgotten): `demand_panel` must be a DENSE daily panel (every calendar
    day present, zero-demand days included as demand_quantity=0) — see
    prepare_modeling_data.py's module docstring for why. Phase 3's pipeline
    (backend/services/preprocessing/pipeline.py) calls this with the SPARSE
    panel instead (only days with an actual sale), so the committed
    data/processed/inventory.csv silently skips gap days in its running
    balance — e.g. product 85123A's ledger jumps from 2010-12-03 straight to
    2010-12-05. That file is a known-limited Phase 3 artifact, left as-is
    (nothing consumed it before Phase 6, and re-deriving it for the full
    19,131-series catalog is a real cost for no current consumer); the
    corrected version for the series that matter — the ones the forecasting
    model and Inventory Agent actually use — is `derive_multi_warehouse_ledger`
    below, built on the dense panel from prepare_modeling_data.py.
    """
    panel = demand_panel.sort_values(["product_id", "date"]).copy()
    rows = []
    stockout_count = 0

    for product_id, group in panel.groupby("product_id", sort=False):
        group = group.sort_values("date")
        mean_daily = max(group["demand_quantity"].mean(), 0.1)
        safety_stock = round(mean_daily * SAFETY_DAYS)
        replenish_qty = round(mean_daily * REPLENISH_DAYS)
        closing = round(mean_daily * TARGET_DAYS_OF_COVER)

        for _, row in group.iterrows():
            opening = closing
            outbound = int(row["demand_quantity"])
            inbound = replenish_qty if (opening - outbound) < safety_stock else 0
            closing = max(opening + inbound - outbound, 0)
            stockout = (opening + inbound) < outbound
            if stockout:
                stockout_count += 1

            rows.append(
                {
                    "warehouse_id": warehouse_id,
                    "product_id": product_id,
                    "date": row["date"],
                    "opening_stock": int(opening),
                    "inbound_quantity": int(inbound),
                    "outbound_quantity": outbound,
                    "closing_stock": int(closing),
                    "safety_stock": int(safety_stock),
                    "stockout_flag": bool(stockout),
                    "provenance": "derived",
                }
            )

    out = pd.DataFrame(rows)
    report = {
        "output_rows": len(out),
        "products_covered": out["product_id"].nunique(),
        "stockout_days": stockout_count,
        "policy": {
            "target_days_of_cover": TARGET_DAYS_OF_COVER,
            "safety_days": SAFETY_DAYS,
            "replenish_days": REPLENISH_DAYS,
        },
    }
    return out, report


# --------------------------------------------------------------------------- #
# Multi-warehouse split — SYNTHETIC allocation rule over REAL demand.
#
# The acquired dataset (UCI Online Retail) is single-location (~99% UK) — there
# is no real multi-warehouse structure to derive from. The Inventory Agent's
# whole point (cross-warehouse transfer recommendations) needs more than one
# location to reason about, so this apportions each product's real total
# demand across three named illustrative warehouses by a fixed, documented
# share — the day-to-day demand SHAPE is still the real series (just scaled),
# only the split across warehouses is synthetic. Each warehouse also gets its
# own stock policy multiplier, deliberately varied so the demo produces a mix
# of risk levels rather than three identical copies of the same story.
# --------------------------------------------------------------------------- #
WAREHOUSE_PROFILES = {
    # name -> (demand_share, starting_days_of_cover_multiplier, safety_days_multiplier)
    # Multipliers apply on top of TARGET_DAYS_OF_COVER/SAFETY_DAYS above.
    # Mumbai is deliberately under-stocked relative to its demand share —
    # produces a genuine HIGH-risk / transfer-candidate warehouse in the demo,
    # not a cherry-picked example.
    "Mumbai": {"demand_share": 0.45, "cover_multiplier": 0.35, "safety_multiplier": 0.5},
    "Chennai": {"demand_share": 0.30, "cover_multiplier": 1.0, "safety_multiplier": 1.0},
    "Delhi": {"demand_share": 0.25, "cover_multiplier": 1.6, "safety_multiplier": 1.2},
}


def derive_multi_warehouse_ledger(dense_panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """`dense_panel` must be a dense daily panel (e.g.
    ml/training/prepare_modeling_data.py's output, or any frame with columns
    date, product_id, demand_quantity covering every calendar day). Splits
    each product's real demand across WAREHOUSE_PROFILES and runs the same
    reorder-point bookkeeping as derive_inventory_ledger, per warehouse.
    """
    assert abs(sum(p["demand_share"] for p in WAREHOUSE_PROFILES.values()) - 1.0) < 1e-9, "demand shares must sum to 1.0"

    base = dense_panel[["date", "product_id", "demand_quantity"]].drop_duplicates(subset=["date", "product_id"])
    rows = []
    stockout_count = 0

    for warehouse_id, profile in WAREHOUSE_PROFILES.items():
        for product_id, group in base.groupby("product_id", sort=False):
            group = group.sort_values("date").copy()
            group["demand_quantity"] = group["demand_quantity"] * profile["demand_share"]

            mean_daily = max(group["demand_quantity"].mean(), 0.1)
            safety_stock = round(mean_daily * SAFETY_DAYS * profile["safety_multiplier"])
            replenish_qty = round(mean_daily * REPLENISH_DAYS)
            closing = round(mean_daily * TARGET_DAYS_OF_COVER * profile["cover_multiplier"])

            for _, row in group.iterrows():
                opening = closing
                outbound = int(round(row["demand_quantity"]))
                inbound = replenish_qty if (opening - outbound) < safety_stock else 0
                closing = max(opening + inbound - outbound, 0)
                stockout = (opening + inbound) < outbound
                if stockout:
                    stockout_count += 1

                rows.append(
                    {
                        "warehouse_id": warehouse_id,
                        "product_id": product_id,
                        "date": row["date"],
                        "opening_stock": int(opening),
                        "inbound_quantity": int(inbound),
                        "outbound_quantity": outbound,
                        "closing_stock": int(closing),
                        "safety_stock": int(safety_stock),
                        "stockout_flag": bool(stockout),
                        "provenance": "synthetic",  # allocation rule is synthetic; see module note above
                    }
                )

    out = pd.DataFrame(rows)
    report = {
        "output_rows": len(out),
        "warehouses": list(WAREHOUSE_PROFILES.keys()),
        "products_covered": out["product_id"].nunique(),
        "stockout_days": stockout_count,
        "warehouse_profiles": WAREHOUSE_PROFILES,
    }
    return out, report
