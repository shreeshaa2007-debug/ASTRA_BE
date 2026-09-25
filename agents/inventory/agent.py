"""InventoryAgent — composes the tool functions in tools.py into the exact
recommendation shape from brief §8. Produces recommendations only; never
writes to inventory or executes a transfer (agent-plan.md's ground rule —
the Optimization Engine, Phase 10, decides what's actually in the final plan).
"""
from __future__ import annotations

import pandas as pd

from backend.agents.inventory import tools

WAREHOUSES = ["Mumbai", "Chennai", "Delhi"]


class InventoryAgent:
    def __init__(self, predictor_version: str = "2026.09.1", horizon_days: int = tools.DEFAULT_FORECAST_HORIZON_DAYS):
        self.predictor_version = predictor_version
        self.horizon_days = horizon_days

    def analyze_product(self, product_id: str, as_of_date: str | pd.Timestamp | None = None) -> list[dict]:
        """Returns one record per warehouse for `product_id`, shaped exactly
        like the brief's §8 example. Every record from this call shares the
        same `recommended_transfer` (or None) — a transfer is a decision
        about the product across warehouses, not a per-warehouse fact.
        """
        snapshots = []
        for warehouse_id in WAREHOUSES:
            inv = tools.get_inventory(warehouse_id, product_id, as_of_date)
            fc = tools.forecast_demand(product_id, warehouse_id, as_of_date, self.horizon_days, self.predictor_version)
            risk = tools.calculate_stockout_risk(inv["current_stock"], fc["forecast_demand"], inv["safety_stock"], self.horizon_days)
            snapshots.append({**inv, **fc, "stockout_risk": risk})

        transfer = tools.calculate_transfer_recommendation(snapshots)

        return [
            {
                "warehouse": s["warehouse_id"],
                "product": s["product_id"],
                "forecast_demand": s["forecast_demand"],
                "current_stock": s["current_stock"],
                "stockout_risk": s["stockout_risk"],
                "recommended_transfer": (
                    {"from": transfer["from"], "quantity": transfer["quantity"]}
                    if transfer and transfer["to"] == s["warehouse_id"]
                    else None
                ),
            }
            for s in snapshots
        ]

    def run(self, as_of_date: str | pd.Timestamp | None = None) -> list[dict]:
        """All products, all warehouses — the full agent report an
        orchestrator run (Phase 15/17) would consume."""
        panel = tools._load_demand_panel()
        product_ids = sorted(panel["product_id"].unique())
        results = []
        for product_id in product_ids:
            results.extend(self.analyze_product(product_id, as_of_date))
        return results
