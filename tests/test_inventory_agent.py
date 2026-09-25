"""Phase 6 tests — brief §21 "AGENTS" category (inventory) plus the tool-level
unit tests agent-plan.md implies for each controlled tool function.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backend.agents.inventory import tools
from backend.agents.inventory.agent import InventoryAgent
from backend.services.preprocessing.inventory import WAREHOUSE_PROFILES, derive_multi_warehouse_ledger

requires_built_data = pytest.mark.skipif(
    not Path("data/processed/inventory_multi_warehouse.csv").exists(),
    reason="run `python -m backend.agents.inventory.build_data` first",
)


# --------------------------------------------------------------------------- #
# inventory.py — multi-warehouse derivation
# --------------------------------------------------------------------------- #
def test_warehouse_demand_shares_sum_to_one():
    assert abs(sum(p["demand_share"] for p in WAREHOUSE_PROFILES.values()) - 1.0) < 1e-9


def test_derive_multi_warehouse_ledger_has_no_date_gaps():
    dense = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10),
            "product_id": ["P1"] * 10,
            "demand_quantity": [5] * 10,
        }
    )
    ledger, report = derive_multi_warehouse_ledger(dense)
    for warehouse_id in WAREHOUSE_PROFILES:
        sub = ledger[(ledger["warehouse_id"] == warehouse_id) & (ledger["product_id"] == "P1")].sort_values("date")
        assert len(sub) == 10
        assert sub["date"].diff().dt.days.dropna().max() == 1.0


def test_derive_multi_warehouse_ledger_splits_demand_by_declared_share():
    dense = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5), "product_id": ["P1"] * 5, "demand_quantity": [100] * 5})
    ledger, _ = derive_multi_warehouse_ledger(dense)
    mumbai_outbound = ledger[(ledger.warehouse_id == "Mumbai") & (ledger.product_id == "P1")]["outbound_quantity"].iloc[2]
    assert mumbai_outbound == pytest.approx(100 * WAREHOUSE_PROFILES["Mumbai"]["demand_share"], abs=1)


# --------------------------------------------------------------------------- #
# tools.py
# --------------------------------------------------------------------------- #
@requires_built_data
def test_get_inventory_raises_for_unknown_record():
    with pytest.raises(ValueError):
        tools.get_inventory("Mumbai", "NOT-A-REAL-PRODUCT", "2011-01-01")


@requires_built_data
def test_get_inventory_returns_expected_shape():
    inv = tools.get_inventory("Mumbai", "21915", "2010-12-02")
    assert inv["current_stock"] == 0  # the known real stockout used in the agent-level test below
    assert set(inv.keys()) == {"warehouse_id", "product_id", "as_of_date", "current_stock", "safety_stock", "stockout_flag"}


@requires_built_data
def test_forecast_demand_rejects_unknown_warehouse():
    with pytest.raises(ValueError):
        tools.forecast_demand("21915", "NotAWarehouse", "2010-12-02")


@requires_built_data
def test_forecast_demand_allocates_by_warehouse_share():
    """Same product/date, different warehouses — forecasts must scale by
    demand_share, since they come from the same underlying UK-aggregate
    prediction (see forecast_demand()'s docstring)."""
    mumbai = tools.forecast_demand("21915", "Mumbai", "2010-12-02", horizon_days=7)
    chennai = tools.forecast_demand("21915", "Chennai", "2010-12-02", horizon_days=7)
    ratio = mumbai["forecast_demand"] / chennai["forecast_demand"]
    expected_ratio = WAREHOUSE_PROFILES["Mumbai"]["demand_share"] / WAREHOUSE_PROFILES["Chennai"]["demand_share"]
    assert ratio == pytest.approx(expected_ratio, rel=0.01)


def test_calculate_stockout_risk_thresholds():
    # 100 stock, 10/day avg demand -> 10 days cover -> MEDIUM (between 7 and 14)
    assert tools.calculate_stockout_risk(current_stock=100, forecast_demand=140, safety_stock=50, horizon_days=14) == "MEDIUM"
    # 50 stock, 10/day -> 5 days -> HIGH
    assert tools.calculate_stockout_risk(current_stock=50, forecast_demand=140, safety_stock=50, horizon_days=14) == "HIGH"
    # 300 stock, 10/day -> 30 days -> LOW
    assert tools.calculate_stockout_risk(current_stock=300, forecast_demand=140, safety_stock=50, horizon_days=14) == "LOW"


def test_calculate_stockout_risk_handles_zero_forecast_without_dividing_by_zero():
    # no forecast demand at all -> effectively infinite cover -> LOW, not a crash
    assert tools.calculate_stockout_risk(current_stock=10, forecast_demand=0, safety_stock=5, horizon_days=14) == "LOW"


# --------------------------------------------------------------------------- #
# calculate_transfer_recommendation — pure function, synthetic snapshots
# --------------------------------------------------------------------------- #
def test_transfer_recommendation_none_when_no_high_risk():
    snapshots = [
        {"warehouse_id": "A", "current_stock": 100, "safety_stock": 50, "stockout_risk": "MEDIUM"},
        {"warehouse_id": "B", "current_stock": 200, "safety_stock": 50, "stockout_risk": "LOW"},
    ]
    assert tools.calculate_transfer_recommendation(snapshots) is None


def test_transfer_recommendation_none_when_no_surplus_available():
    snapshots = [
        {"warehouse_id": "A", "current_stock": 10, "safety_stock": 50, "stockout_risk": "HIGH"},
        {"warehouse_id": "B", "current_stock": 40, "safety_stock": 50, "stockout_risk": "MEDIUM"},  # also below its own safety stock
    ]
    assert tools.calculate_transfer_recommendation(snapshots) is None


def test_transfer_recommendation_picks_largest_surplus_source():
    snapshots = [
        {"warehouse_id": "Deficit", "current_stock": 10, "safety_stock": 100, "stockout_risk": "HIGH"},
        {"warehouse_id": "SmallSurplus", "current_stock": 120, "safety_stock": 100, "stockout_risk": "LOW"},
        {"warehouse_id": "BigSurplus", "current_stock": 500, "safety_stock": 100, "stockout_risk": "LOW"},
    ]
    rec = tools.calculate_transfer_recommendation(snapshots)
    assert rec["from"] == "BigSurplus"
    assert rec["to"] == "Deficit"
    assert rec["quantity"] > 0


def test_transfer_recommendation_never_exceeds_source_surplus():
    snapshots = [
        {"warehouse_id": "Deficit", "current_stock": 0, "safety_stock": 1000, "stockout_risk": "HIGH"},  # huge need
        {"warehouse_id": "Source", "current_stock": 105, "safety_stock": 100, "stockout_risk": "LOW"},  # tiny surplus: 5
    ]
    rec = tools.calculate_transfer_recommendation(snapshots)
    assert rec["quantity"] <= 5


# --------------------------------------------------------------------------- #
# agent.py — end-to-end, against the real trained model + real derived ledger
# --------------------------------------------------------------------------- #
@requires_built_data
def test_agent_produces_real_high_risk_and_transfer_on_a_known_stockout_date():
    """2010-12-02 is a REAL date where product 21915's Mumbai ledger hit
    closing_stock=0 (verified by inspecting data/processed/
    inventory_multi_warehouse.csv directly, not picked to make a test pass) —
    this is the brief §8 example shape produced end-to-end by real logic."""
    agent = InventoryAgent()
    result = agent.analyze_product("21915", as_of_date="2010-12-02")

    assert len(result) == 3
    mumbai = next(r for r in result if r["warehouse"] == "Mumbai")
    assert mumbai["current_stock"] == 0
    assert mumbai["stockout_risk"] == "HIGH"
    assert mumbai["recommended_transfer"] is not None
    assert mumbai["recommended_transfer"]["quantity"] > 0

    others = [r for r in result if r["warehouse"] != "Mumbai"]
    assert all(r["recommended_transfer"] is None for r in others)


@requires_built_data
def test_agent_output_matches_brief_schema_keys():
    agent = InventoryAgent()
    result = agent.analyze_product("21915", as_of_date="2010-12-02")
    expected_keys = {"warehouse", "product", "forecast_demand", "current_stock", "stockout_risk", "recommended_transfer"}
    for record in result:
        assert set(record.keys()) == expected_keys


@requires_built_data
def test_agent_never_writes_anything(tmp_path, monkeypatch):
    """The brief is explicit: this agent must not execute transfers. There's
    no write path to test against directly, so this asserts the source files
    on disk are untouched by a run — the closest thing to a behavioral
    guarantee without mocking the filesystem entirely."""
    ledger_path = Path("data/processed/inventory_multi_warehouse.csv")
    before = ledger_path.stat().st_mtime

    agent = InventoryAgent()
    agent.analyze_product("21915", as_of_date="2010-12-02")

    assert ledger_path.stat().st_mtime == before
