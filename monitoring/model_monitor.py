"""Watching the demand model (docs/model-plan.md §8, brief §26 — lightweight on purpose).

Per prediction (`record_inference`, called by DemandPredictor.predict_one): the count, the latency, the model
version, and the missing-feature rate (lag/rolling inputs that were NaN because the series is shorter than 28
days). Per forecast (`check_input_drift`, called by forecast_series with the *actual* demand window the forecast
starts from): a simple drift warning — how many standard deviations the window's mean sits from that product's
training-time distribution of 28-day means. And `backtest` supplies the fifth signal, prediction error "once actuals
are known": forecast the last `horizon_days` of history from before them and score it (WAPE, MAE).

What this is not: a drift-detection library. It is one threshold on one statistic, per product, and it says so.
A warning means "recent demand looks unlike anything this model was trained on, so treat its forecasts with more
suspicion" — not that the forecast is wrong. (On this data it fires for real: the model was trained on
Dec-2010..Aug-2011 and never saw a Q4 ramp, and the ledger runs into December.)

Everything is in process and also goes to the structured log and the metrics registry.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from backend.data import get_datasets
from backend.monitoring.metrics import metrics
from backend.monitoring.settings import load_settings

logger = logging.getLogger("resilientsc.monitoring.model")

NUMERIC_FEATURES = 8  # lag_1, lag_2, lag_7, lag_14, lag_28, rolling_mean_7, rolling_mean_14, rolling_mean_28


@dataclass(frozen=True)
class TrainingStats:
    mean: float
    std: float
    windows: int


def load_training_stats(path: Path | None = None) -> dict[str, TrainingStats]:
    """Per product: mean and standard deviation of the 28-day rolling mean over the training split, from the
    first full window on (the first 27 days of a series have a partial window, which the model saw but which is
    not the distribution a full window is compared with). From the `demand_panel` dataset, or from the CSV at
    `path` when one is given."""
    columns = ["date", "product_id", "rolling_mean_28", "split"]
    panel = get_datasets().load("demand_panel", columns) if path is None else pd.read_csv(path, usecols=columns, parse_dates=["date"])
    panel["product_id"] = panel["product_id"].astype(str)
    train = panel[panel["split"] == "train"].sort_values(["product_id", "date"])
    stats: dict[str, TrainingStats] = {}
    for product, rows in train.groupby("product_id"):
        windows = rows["rolling_mean_28"].iloc[27:].dropna()
        if len(windows):
            stats[str(product)] = TrainingStats(float(windows.mean()), float(windows.std(ddof=0)), int(len(windows)))
    return stats


class ModelMonitor:
    def __init__(self, settings: dict | None = None, stats_loader: Callable[[], dict[str, TrainingStats]] = load_training_stats):
        self._settings_override = settings
        self._stats_loader = stats_loader
        self._stats_cache: dict[str, TrainingStats] | None = None
        self._lock = threading.Lock()
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        with self._lock:
            self._versions: Counter[str] = Counter()
            self._latencies: deque[float] = deque(maxlen=512)
            self._latency_total = 0.0
            self._features_checked = 0
            self._features_missing = 0
            self._drift: dict[str, dict] = {}
            self._warnings: deque[dict] = deque(maxlen=int(self.settings["warnings_kept"]))
            self._backtests: dict[str, dict] = {}

    @property
    def settings(self) -> dict:
        return (self._settings_override or load_settings())["model"]

    def _stats(self) -> dict[str, TrainingStats]:
        if self._stats_cache is None:
            self._stats_cache = self._stats_loader()
        return self._stats_cache

    # ------------------------------------------------------------ per inference
    def record_inference(self, *, version: str, product_id: str, latency_ms: float, features_missing: int, features_total: int = NUMERIC_FEATURES) -> None:
        with self._lock:
            self._versions[version] += 1
            self._latencies.append(latency_ms)
            self._latency_total += latency_ms
            self._features_checked += features_total
            self._features_missing += features_missing
        metrics.inc("forecast_inferences_total", {"model_version": version})
        metrics.observe("forecast_inference_ms", latency_ms)

    # ---------------------------------------------------------------- per forecast
    def check_input_drift(self, product_id: str, window: Sequence[float]) -> Optional[dict]:
        """Compares the mean of the last `drift_window_days` actual values with the product's training distribution."""
        s = self.settings
        recent = [float(v) for v in window][-int(s["drift_window_days"]):]
        if not recent:
            return None
        product = str(product_id)
        train = self._stats().get(product)
        window_mean = float(np.mean(recent))
        result: dict = {"product_id": product, "window_mean": round(window_mean, 3), "window_days": len(recent),
                        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if train is None or train.windows < int(s["min_training_windows"]) or train.std <= 0:
            result.update(status="NO_BASELINE", z=None, train_mean=None, train_std=None)
        else:
            z = (window_mean - train.mean) / train.std
            result.update(status="DRIFT" if abs(z) > float(s["drift_z_threshold"]) else "OK", z=round(z, 2),
                          train_mean=round(train.mean, 3), train_std=round(train.std, 3))
        with self._lock:
            previous = self._drift.get(product, {}).get("status")
            self._drift[product] = result
            newly_drifting = result["status"] == "DRIFT" and previous != "DRIFT"
            if newly_drifting:
                self._warnings.append(result)
        if newly_drifting:  # once per transition, not once per forecast
            metrics.inc("forecast_drift_warnings_total", {"product_id": product})
            logger.warning(
                "demand drift: product %s's recent %d-day mean %.1f is %+.1f standard deviations from its training distribution (mean %.1f, std %.1f)",
                product, len(recent), window_mean, result["z"], result["train_mean"], result["train_std"],
                extra={"event": "drift_warning", "product_id": product, "z": result["z"]})
        return result

    # ------------------------------------------------------------ prediction error
    def backtest(self, product_id: str, horizon_days: int = 14) -> dict:
        """Forecasts the last `horizon_days` of known history from before it and scores the forecast against what
        happened: the model's prediction error, now that the actuals are known."""
        from backend.agents.inventory import tools as inventory_tools  # local: the predictor imports this module

        product = str(product_id)
        panel = inventory_tools._load_demand_panel()
        history = panel[panel["product_id"] == product].sort_values("date")
        if len(history) <= horizon_days + 28:
            raise ValueError(f"not enough history for product {product!r} to backtest {horizon_days} days")
        latest = history["date"].max()
        as_of = latest - pd.Timedelta(days=horizon_days)
        predicted = {d["date"]: d["predicted"] for d in inventory_tools.forecast_series(product, as_of, horizon_days)}
        actual = {str(r.date.date()): float(r.demand_quantity) for r in history[history["date"] > as_of].itertuples()}
        dates = sorted(set(predicted) & set(actual))
        errors = [abs(predicted[d] - actual[d]) for d in dates]
        total_actual = sum(actual[d] for d in dates)
        result = {
            "product_id": product, "horizon_days": horizon_days, "as_of": str(as_of.date()), "days_scored": len(dates),
            "mae": round(float(np.mean(errors)), 3) if errors else None,
            "wape": round(sum(errors) / total_actual, 4) if total_actual > 0 else None,  # None when nothing was demanded: a ratio of zero is not an error rate
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with self._lock:
            self._backtests[product] = result
        logger.info("backtest %s: WAPE %s over %d days", product, result["wape"], len(dates), extra={"event": "backtest", "product_id": product, "wape": result["wape"]})
        return result

    # ------------------------------------------------------------------- report
    def snapshot(self) -> dict:
        s = self.settings
        with self._lock:
            latencies = sorted(self._latencies)
            count = sum(self._versions.values())

            def q(p: float) -> Optional[float]:
                return round(latencies[min(len(latencies) - 1, max(0, math.ceil(p * len(latencies)) - 1))], 3) if latencies else None

            return {
                "inference_count": count,
                "model_versions": dict(self._versions),
                "latency_ms": {"count": count, "mean": round(self._latency_total / count, 3) if count else None, "p50": q(0.5), "p95": q(0.95),
                               "max": round(latencies[-1], 3) if latencies else None},
                "missing_feature_rate": round(self._features_missing / self._features_checked, 4) if self._features_checked else None,
                "features_checked": self._features_checked,
                "drift": {
                    "threshold_z": float(s["drift_z_threshold"]), "window_days": int(s["drift_window_days"]),
                    "products_checked": len(self._drift), "products_drifting": sum(d["status"] == "DRIFT" for d in self._drift.values()),
                    "by_product": {p: {k: d[k] for k in ("status", "z", "window_mean", "train_mean", "train_std", "checked_at")} for p, d in sorted(self._drift.items())},
                    "warnings": list(self._warnings),
                    "method": "mean of the recent demand window, in standard deviations of the product's training-time 28-day means (a threshold check, not a drift-detection library)",
                },
                "backtests": dict(sorted(self._backtests.items())),
            }


monitor = ModelMonitor()
