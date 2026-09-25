"""Loads a trained ml/artifacts/xgboost_demand/<version>/ artifact and serves
predictions matching the POST /api/forecast contract in model-plan.md §6 /
docs/api-plan.md. This is the function Phase 15's API layer calls — it is not
itself an API endpoint (backend/api/ is the other Phase 15/16 work stream's
area; this module only needs to exist and be correct, not wire into FastAPI).

No model is trained at import time or at call time (brief §25: never train at
backend startup) — `load_predictor()` only reads saved artifact files.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from backend.monitoring.model_monitor import monitor

ARTIFACTS_ROOT = Path("ml/artifacts/xgboost_demand")


class DemandPredictor:
    def __init__(self, version: str):
        artifact_dir = ARTIFACTS_ROOT / version
        if not artifact_dir.exists():
            raise FileNotFoundError(f"No forecasting model artifact at {artifact_dir}")

        self.version = version
        self.booster = xgb.Booster()
        self.booster.load_model(str(artifact_dir / "model.json"))

        preprocessing = joblib.load(artifact_dir / "preprocessing.pkl")
        self.categories: dict[str, list] = preprocessing["categories"]
        self.feature_columns: list[str] = preprocessing["feature_columns"]
        self._known_series: list[tuple[str, str]] = preprocessing["known_series"]

        with open(artifact_dir / "metadata.json") as f:
            self.metadata = json.load(f)

    def predict_one(
        self,
        product_id: str,
        location_id: str,
        recent_demand: list[float],
        day_of_week: int,
        month: int,
        disruption_active: bool = False,
    ) -> float:
        """`recent_demand` must be the last 28 days of actual demand, most
        recent last (recent_demand[-1] = yesterday) — the same convention
        lag_1..lag_28 use in training. Fewer than 28 values is fine; missing
        lags are left NaN, matching how early-series rows were trained (see
        train_xgboost.py's module docstring on why NaN, not zero-fill).
        """
        if (product_id, location_id) not in self._known_series:
            raise ValueError(
                f"(product_id={product_id!r}, location_id={location_id!r}) was not one of this "
                f"model's trained series — this model (version {self.version}) only covers the "
                "top-K series selected in ml/training/prepare_modeling_data.py (checked against "
                "the actual trained pairs, not each column independently — a product and a "
                "location can each be 'known' without that specific combination ever having been "
                "trained). See that module's TOP_K/MIN_ACTIVE_DAYS to extend coverage, then retrain."
            )

        started = time.perf_counter()

        def lag(n: int) -> float:
            return recent_demand[-n] if len(recent_demand) >= n else np.nan

        row = {
            "lag_1": lag(1), "lag_2": lag(2), "lag_7": lag(7), "lag_14": lag(14), "lag_28": lag(28),
            "rolling_mean_7": np.mean(recent_demand[-7:]) if recent_demand else np.nan,
            "rolling_mean_14": np.mean(recent_demand[-14:]) if recent_demand else np.nan,
            "rolling_mean_28": np.mean(recent_demand[-28:]) if recent_demand else np.nan,
            "day_of_week": day_of_week,
            "month": month,
            "disruption_active": int(disruption_active),
            "product_id": pd.Categorical([product_id], categories=self.categories["product_id"])[0],
            "location_id": pd.Categorical([location_id], categories=self.categories["location_id"])[0],
        }
        X = pd.DataFrame([row])[self.feature_columns]
        for col in ("product_id", "location_id"):
            X[col] = pd.Categorical(X[col], categories=self.categories[col])

        dmatrix = xgb.DMatrix(X, enable_categorical=True)
        pred = float(self.booster.predict(dmatrix)[0])
        numeric = ("lag_1", "lag_2", "lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_14", "rolling_mean_28")
        monitor.record_inference(  # model monitoring (docs/model-plan.md §8): count, latency, version, missing-feature rate
            version=self.version, product_id=product_id, latency_ms=(time.perf_counter() - started) * 1000,
            features_missing=sum(1 for name in numeric if np.isnan(row[name])))
        return max(pred, 0.0)

    def known_series(self) -> list[tuple[str, str]]:
        """The (product_id, location_id) pairs this model version was
        actually trained on — the API layer should use this to answer "what
        can I forecast?" rather than guessing or hardcoding a list. NOT the
        cartesian product of known products x known locations: a product and
        a location can each be individually "known" (appear somewhere in
        self.categories) without that specific pair having been trained —
        e.g. product P sold only in the UK, location DE only had other
        products. Returning the cartesian product would silently overclaim
        coverage for pairs the model never saw.
        """
        return list(self._known_series)


_cached: dict[str, DemandPredictor] = {}


def load_predictor(version: str = "2026.09.1") -> DemandPredictor:
    if version not in _cached:
        _cached[version] = DemandPredictor(version)
    return _cached[version]
