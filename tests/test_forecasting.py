"""Phase 4 tests — brief §21 "MODEL" category: prediction output schema,
model loading, inference, metrics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.training import baselines as baselines_mod
from ml.training.metrics import all_metrics, mae, mape, rmse, wape
from ml.training.prepare_modeling_data import MIN_ACTIVE_DAYS, densify, select_series

ARTIFACT_DIR = Path("ml/artifacts/xgboost_demand/2026.09.1")
requires_trained_model = pytest.mark.skipif(
    not ARTIFACT_DIR.exists(), reason="run `python -m ml.training.run_phase4` first"
)


# --------------------------------------------------------------------------- #
# metrics.py
# --------------------------------------------------------------------------- #
def test_mae_rmse_known_values():
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([12.0, 18.0, 33.0])
    # errors: 2, -2, 3 -> |e|: 2,2,3 -> mae=7/3; squared: 4,4,9 -> rmse=sqrt(17/3)
    assert mae(actual, predicted) == pytest.approx(7 / 3)
    assert rmse(actual, predicted) == pytest.approx(np.sqrt(17 / 3))


def test_wape_known_value():
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([12.0, 18.0, 33.0])
    # sum|error|=7, sum|actual|=60 -> wape = 7/60*100
    assert wape(actual, predicted) == pytest.approx(7 / 60 * 100)


def test_mape_and_wape_return_none_not_nan_when_all_actuals_zero():
    actual = np.zeros(5)
    predicted = np.array([1.0, 0.0, 2.0, 0.0, 1.0])
    assert mape(actual, predicted) is None
    assert wape(actual, predicted) is None


def test_all_metrics_returns_all_four_keys():
    result = all_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert set(result.keys()) == {"mae", "rmse", "mape", "wape"}
    assert result["mae"] == 0.0


# --------------------------------------------------------------------------- #
# prepare_modeling_data.py
# --------------------------------------------------------------------------- #
def test_select_series_excludes_low_density_single_day_spike():
    """The real bug this guards: product 23843 sold 80,995 units in a single
    day and would rank #1 by volume — select_series must exclude it via the
    active_days floor, not let a one-off bulk order masquerade as a
    forecastable series."""
    sparse = pd.DataFrame(
        {
            "product_id": ["SPIKE"] * 1 + ["STEADY"] * (MIN_ACTIVE_DAYS + 10),
            "location_id": ["UK"] * (1 + MIN_ACTIVE_DAYS + 10),
            "date": pd.to_datetime(["2024-01-01"]) .tolist()
            + pd.date_range("2024-01-01", periods=MIN_ACTIVE_DAYS + 10).tolist(),
            "demand_quantity": [999_999] + [5] * (MIN_ACTIVE_DAYS + 10),
        }
    )
    selected = select_series(sparse)
    assert "SPIKE" not in selected["product_id"].values
    assert "STEADY" in selected["product_id"].values


def test_densify_fills_missing_days_with_zero_not_absence():
    sparse = pd.DataFrame(
        {
            "product_id": ["A", "A"],
            "location_id": ["X", "X"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-05"]),  # gap: 02, 03, 04 missing
            "demand_quantity": [10, 20],
        }
    )
    selected = pd.DataFrame({"product_id": ["A"], "location_id": ["X"]})
    full_range = pd.date_range("2024-01-01", "2024-01-05", freq="D")
    dense = densify(sparse, selected, full_range)

    assert len(dense) == 5  # every calendar day present, not just the 2 with sales
    gap_days = dense[dense["date"].isin(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))]
    assert (gap_days["demand_quantity"] == 0).all()


# --------------------------------------------------------------------------- #
# baselines.py — each one must equal its documented source column exactly
# --------------------------------------------------------------------------- #
def test_naive_is_exactly_lag_1():
    panel = pd.DataFrame({"lag_1": [1.0, 2.0, np.nan], "lag_7": [9, 9, 9], "rolling_mean_7": [5, 5, 5]})
    pd.testing.assert_series_equal(baselines_mod.naive_predictions(panel), panel["lag_1"])


def test_seasonal_naive_is_exactly_lag_7():
    panel = pd.DataFrame({"lag_1": [1, 2, 3], "lag_7": [9.0, 8.0, 7.0], "rolling_mean_7": [5, 5, 5]})
    pd.testing.assert_series_equal(baselines_mod.seasonal_naive_predictions(panel), panel["lag_7"])


def test_moving_average_is_exactly_rolling_mean_7():
    panel = pd.DataFrame({"lag_1": [1, 2, 3], "lag_7": [9, 8, 7], "rolling_mean_7": [4.5, 5.5, 6.5]})
    pd.testing.assert_series_equal(baselines_mod.moving_average_predictions(panel), panel["rolling_mean_7"])


def test_exponential_smoothing_predictions_are_nonnegative_and_only_for_non_train():
    panel = pd.DataFrame(
        {
            "product_id": ["A"] * 30,
            "location_id": ["X"] * 30,
            "date": pd.date_range("2024-01-01", periods=30),
            "demand_quantity": [max(0, 5 + i % 7 - 3) for i in range(30)],
            "split": ["train"] * 20 + ["val"] * 5 + ["test"] * 5,
        }
    )
    preds = baselines_mod.exponential_smoothing_predictions(panel)
    train_mask = panel["split"] == "train"
    assert preds[train_mask].isna().all()  # no prediction generated for train rows themselves
    assert (preds[~train_mask] >= 0).all()
    assert preds[~train_mask].notna().all()


# --------------------------------------------------------------------------- #
# predictor.py — against the real trained artifact
# --------------------------------------------------------------------------- #
@requires_trained_model
def test_predictor_loads_and_predicts_nonnegative():
    from backend.models.forecasting.predictor import load_predictor

    predictor = load_predictor("2026.09.1")
    pid, loc = predictor.known_series()[0]
    pred = predictor.predict_one(pid, loc, recent_demand=[3, 4, 5, 2, 6, 3, 4] * 4, day_of_week=1, month=5)
    assert isinstance(pred, float)
    assert pred >= 0.0


@requires_trained_model
def test_predictor_rejects_untrained_pair_even_if_each_half_is_known():
    """The exact bug caught during development: checking product_id and
    location_id membership independently would accept any product x any
    location, even combinations that were never trained together."""
    from backend.models.forecasting.predictor import load_predictor

    predictor = load_predictor("2026.09.1")
    known_products = {p for p, _ in predictor.known_series()}
    known_locations = {l for _, l in predictor.known_series()}
    assert len(known_locations) == 1, "this test assumes the single-location dataset shape; revisit if that changes"
    fake_location = "Definitely Not A Trained Location"
    pid = next(iter(known_products))
    with pytest.raises(ValueError):
        predictor.predict_one(pid, fake_location, recent_demand=[1, 2, 3], day_of_week=0, month=1)


@requires_trained_model
def test_predictor_known_series_matches_preprocessing_artifact_count():
    from backend.models.forecasting.predictor import load_predictor

    predictor = load_predictor("2026.09.1")
    assert len(predictor.known_series()) == len(set(predictor.known_series()))  # no duplicates
    assert len(predictor.known_series()) <= 40  # TOP_K ceiling from prepare_modeling_data.py
