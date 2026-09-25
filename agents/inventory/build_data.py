"""Builds data/processed/inventory_multi_warehouse.csv — the corrected, dense,
multi-warehouse inventory ledger the Inventory Agent reads (see
backend/services/preprocessing/inventory.py's derive_multi_warehouse_ledger
docstring for why this exists separately from Phase 3's original
inventory.csv). Depends on Phase 4's dense modeling panel, so it only covers
the 40 series the forecasting model actually supports — consistent scope, not
a coincidence: the Inventory Agent's forecast_demand() tool calls that same
model, so there'd be no point stocking a ledger for products it can't forecast.

Run with: python -m backend.agents.inventory.build_data
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backend.services.preprocessing.inventory import derive_multi_warehouse_ledger

MODELING_PANEL_PATH = Path("data/processed/demand_modeling_panel.csv")
OUTPUT_PATH = Path("data/processed/inventory_multi_warehouse.csv")


def build() -> dict:
    panel = pd.read_csv(MODELING_PANEL_PATH, usecols=["date", "product_id", "demand_quantity"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel["product_id"] = panel["product_id"].astype(str)

    ledger, report = derive_multi_warehouse_ledger(panel)
    ledger.to_csv(OUTPUT_PATH, index=False)
    report["output_path"] = str(OUTPUT_PATH)
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, default=str))
