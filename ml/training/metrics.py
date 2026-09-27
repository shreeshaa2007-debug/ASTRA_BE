"""MAE / RMSE / MAPE / WAPE — per model-plan.md §4. WAPE is the headline
metric: MAPE explodes or divides by zero on the many true-zero-demand days a
densified intermittent series has (see prepare_modeling_data.py); WAPE
(sum |error| / sum |actual|) stays well-defined and doesn't let a handful of
near-zero-actual days dominate the score.
"""
from __future__ import annotations

import numpy as np


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    """None (not NaN/inf) when every actual is zero — undefined, not "0%
    error"; a caller must handle the None rather than silently plotting it.
    """
    mask = actual != 0
    if not mask.any():
        return None
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def wape(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return None
    return float(np.sum(np.abs(actual - predicted)) / denom * 100)


def all_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {"mae": mae(actual, predicted), "rmse": rmse(actual, predicted), "mape": mape(actual, predicted), "wape": wape(actual, predicted)}
