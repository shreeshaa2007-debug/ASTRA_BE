"""Phase 3 preprocessing tests — brief §21 "DATA" category: schema validation,
missing data, invalid records, preprocessing correctness. These run against
real acquired data (data/raw/), not fixtures, so they double as a regression
check on the actual pipeline output.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest
from pydantic import ValidationError

from backend.schemas.entities import DemandRecord, DisruptionRecord, InventoryRecord, SupplierRecord
from backend.services.preprocessing import demand as demand_mod
from backend.services.preprocessing import disruptions as disruptions_mod
from backend.services.preprocessing import inventory as inventory_mod
from backend.services.preprocessing import suppliers as suppliers_mod
from backend.services.preprocessing.features import add_lag_features, add_rolling_features, time_based_split
from backend.services.preprocessing.utils import haversine_km, parse_money_string


# --------------------------------------------------------------------------- #
# utils.py
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("25.00K", 25_000.0),
        ("1.50M", 1_500_000.0),
        ("0.00K", 0.0),
        ("", 0.0),
        (None, 0.0),
        ("10", 10.0),          # no suffix
        ("2.5B", 2_500_000_000.0),
        ("not-a-number", 0.0),  # malformed — falls back to 0, never raises
    ],
)
def test_money_string_parser(raw, expected):
    assert parse_money_string(raw) == expected


def test_haversine_zero_for_identical_points():
    assert haversine_km(31.2, 121.5, 31.2, 121.5) == 0.0


def test_haversine_known_real_distance():
    # Shanghai <-> Rotterdam, real WPI coordinates (ports_routes.py). The
    # direct great-circle path runs overland across Siberia (~8,900 km) —
    # much shorter than any real sea route, which is exactly why
    # ports_routes.py sums waypoint legs (via Port Said or Cape Town) instead
    # of calling this function point-to-point for its route distances.
    d = haversine_km(31.216667, 121.5, 51.9, 4.483333)
    assert 8_500 < d < 9_500


# --------------------------------------------------------------------------- #
# features.py — leakage discipline is the single most important property here
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def toy_panel():
    return pd.DataFrame(
        {
            "product_id": ["A"] * 5 + ["B"] * 5,
            "location_id": ["X"] * 10,
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"] * 2
            ),
            "demand_quantity": [10, 20, 30, 40, 50, 1, 2, 3, 4, 5],
        }
    )


def test_lag_1_equals_previous_row_value(toy_panel):
    out = add_lag_features(toy_panel, ["product_id", "location_id"])
    a = out[out.product_id == "A"].sort_values("date")
    assert a["lag_1"].tolist() == [None, 10, 20, 30, 40] or a["lag_1"].isna().sum() == 1
    assert a["lag_1"].iloc[1:].tolist() == [10.0, 20.0, 30.0, 40.0]


def test_rolling_mean_excludes_current_row(toy_panel):
    """The leakage bug this guards against: `rolling(window).mean()` on the
    unshifted series would include row t's own value in its own feature. The
    fix is shift(1) before rolling — see features.py's module docstring."""
    out = add_lag_features(toy_panel, ["product_id", "location_id"])
    out = add_rolling_features(out, ["product_id", "location_id"])
    a = out[out.product_id == "A"].sort_values("date")
    # day 4 (value=40): prior values are 10,20,30 -> mean 20.0, NOT
    # mean(10,20,30,40)=25.0 (which is what the leaky version would give)
    assert math.isclose(a["rolling_mean_7"].iloc[3], 20.0)
    assert pd.isna(a["rolling_mean_7"].iloc[0])  # first row has no prior data


def test_rolling_features_do_not_leak_across_groups(toy_panel):
    """A second, interleaved group (B) must not contaminate A's rolling mean
    — this is the exact index-misalignment bug caught during development
    (transform+reset_index(drop=True) silently mixed rows across groups)."""
    out = add_lag_features(toy_panel, ["product_id", "location_id"])
    out = add_rolling_features(out, ["product_id", "location_id"])
    b = out[out.product_id == "B"].sort_values("date")
    assert math.isclose(b["rolling_mean_7"].iloc[3], 2.0)  # mean(1,2,3), not A's values


def test_time_based_split_is_chronological_not_random():
    panel = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=100, freq="D")})
    out = time_based_split(panel, train_frac=0.7, val_frac=0.15)
    train_max = out.loc[out.split == "train", "date"].max()
    val_min, val_max = out.loc[out.split == "val", "date"].min(), out.loc[out.split == "val", "date"].max()
    test_min = out.loc[out.split == "test", "date"].min()
    assert train_max < val_min <= val_max < test_min
    assert out["split"].isna().sum() == 0


# --------------------------------------------------------------------------- #
# demand.py — against the real acquired dataset
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def demand_panel():
    panel, _report = demand_mod.build_demand_panel()
    return panel


def test_demand_panel_has_no_duplicate_grain(demand_panel):
    key = ["date", "product_id", "location_id"]
    assert not demand_panel.duplicated(subset=key).any()


def test_demand_panel_quantities_non_negative(demand_panel):
    # cancelled/return rows (negative Quantity) must have been excluded in
    # clean_transactions, not summed into demand_quantity
    assert (demand_panel["demand_quantity"] >= 0).all()


def test_demand_records_validate_against_schema(demand_panel):
    sample = demand_panel.sample(50, random_state=1)
    for _, row in sample.iterrows():
        DemandRecord(**row.to_dict())  # raises on failure


# --------------------------------------------------------------------------- #
# inventory.py — derived, must stay internally consistent
# --------------------------------------------------------------------------- #
def test_inventory_closing_stock_never_negative(demand_panel):
    small = demand_panel[demand_panel["product_id"].isin(demand_panel["product_id"].unique()[:20])]
    ledger, _report = inventory_mod.derive_inventory_ledger(small)
    assert (ledger["closing_stock"] >= 0).all()


def test_inventory_balance_equation_holds(demand_panel):
    """opening + inbound - outbound must equal closing (or 0 when it would
    go negative) for every row — the core bookkeeping invariant."""
    small = demand_panel[demand_panel["product_id"] == demand_panel["product_id"].iloc[0]]
    ledger, _report = inventory_mod.derive_inventory_ledger(small)
    expected = (ledger["opening_stock"] + ledger["inbound_quantity"] - ledger["outbound_quantity"]).clip(lower=0)
    assert (ledger["closing_stock"] == expected).all()


def test_inventory_records_validate_against_schema(demand_panel):
    small = demand_panel[demand_panel["product_id"].isin(demand_panel["product_id"].unique()[:5])]
    ledger, _report = inventory_mod.derive_inventory_ledger(small)
    for _, row in ledger.sample(min(30, len(ledger)), random_state=1).iterrows():
        InventoryRecord(**row.to_dict())


# --------------------------------------------------------------------------- #
# suppliers.py — synthetic, must be honestly labeled and reproducible
# --------------------------------------------------------------------------- #
def test_suppliers_are_labeled_synthetic():
    df, _report = suppliers_mod.synthesize_suppliers()
    assert (df["provenance"] == "synthetic").all()


def test_suppliers_generation_is_reproducible():
    df1, _ = suppliers_mod.synthesize_suppliers(seed=42)
    df2, _ = suppliers_mod.synthesize_suppliers(seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_supplier_records_validate_against_schema():
    df, _report = suppliers_mod.synthesize_suppliers()
    for _, row in df.iterrows():
        SupplierRecord(**row.to_dict())


# --------------------------------------------------------------------------- #
# disruptions.py — schema + the end-after-start invariant
# --------------------------------------------------------------------------- #
def test_disruption_end_date_before_start_date_is_rejected():
    with pytest.raises(ValidationError):
        DisruptionRecord(
            event_id="X1",
            event_type="Test",
            location="Nowhere",
            start_date="2024-01-10",
            end_date="2024-01-01",  # before start — must be rejected
            severity="LOW",
            status="HISTORICAL",
        )


def test_curated_events_are_real_and_well_formed():
    curated = disruptions_mod.build_curated_events()
    assert len(curated) >= 4
    assert (curated["start_date"] <= curated["end_date"].fillna(curated["start_date"])).all()
    assert curated["event_id"].str.startswith("CURATED-").all()
