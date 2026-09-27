"""Baseline forecasters — per model-plan.md §1 / brief §7, these must exist
and be beaten before XGBoost is allowed to win the argument. Naive, seasonal
naive, and moving average are read directly off the lag/rolling features
already computed by build_feature_panel (backend/services/preprocessing/
features.py) — each one *is* a specific lag/rolling column, evaluated as a
row-wise "predict using only information available before this row" forecast,
the same footing XGBoost is evaluated on.

Exponential smoothing is different in kind: it's fit once per series on the
train segment and produces a genuine multi-step-ahead forecast for the
val+test horizon, not a per-row lag lookup — documented here rather than
glossed over, because it means ETS's evaluation isn't perfectly apples-to-apples
with the row-wise baselines and XGBoost (which effectively get fresh 1-step-
ahead context every row). This is the standard, practical way to evaluate
exponential smoothing; the caveat is in metrics.json, not hidden.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def naive_predictions(panel: pd.DataFrame) -> pd.Series:
    """Predict today's demand as yesterday's — lag_1."""
    return panel["lag_1"]


def seasonal_naive_predictions(panel: pd.DataFrame) -> pd.Series:
    """Predict today's demand as the same weekday last week — lag_7."""
    return panel["lag_7"]


def moving_average_predictions(panel: pd.DataFrame) -> pd.Series:
    """Predict today's demand as the trailing 7-day mean (rolling_mean_7,
    already leakage-safe — computed on shift(1))."""
    return panel["rolling_mean_7"]


def exponential_smoothing_predictions(panel: pd.DataFrame) -> pd.Series:
    """Per (product_id, location_id): fit additive Holt-Winters (weekly
    seasonality, period=7) on that series' train rows, forecast the val+test
    horizon in one shot. Additive, not multiplicative, because a densified
    intermittent series has real zero-demand days — multiplicative seasonal/
    trend terms are undefined at zero.
    """
    preds = pd.Series(index=panel.index, dtype=float)
    for (pid, loc), group in panel.groupby(["product_id", "location_id"], sort=False):
        group = group.sort_values("date")
        train = group[group["split"] == "train"]
        rest = group[group["split"] != "train"]
        if len(rest) == 0:
            continue
        if len(train) < 2 * 7:  # need at least two full seasonal cycles
            preds.loc[rest.index] = train["demand_quantity"].mean() if len(train) else 0.0
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ExponentialSmoothing(
                    train["demand_quantity"].values,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=7,
                    initialization_method="estimated",
                ).fit()
            forecast = model.forecast(len(rest))
        except Exception:
            forecast = np.full(len(rest), train["demand_quantity"].mean())
        preds.loc[rest.index] = np.clip(forecast, 0, None)  # demand can't be negative
    return preds


def compute_all_baselines(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel[["date", "product_id", "location_id", "split", "demand_quantity"]].copy()
    out["pred_naive"] = naive_predictions(panel)
    out["pred_seasonal_naive"] = seasonal_naive_predictions(panel)
    out["pred_moving_average"] = moving_average_predictions(panel)
    out["pred_exp_smoothing"] = exponential_smoothing_predictions(panel)
    return out
