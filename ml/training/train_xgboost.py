"""Trains ONE global XGBoost regressor across all 40 selected series (not one
model per series) — with this little data per series (~260 train days each),
a global model that conditions on product_id/location_id as categorical
features pools signal across series and is far more robust than fitting 40
separate small models. This is standard practice for multi-series demand
forecasting (it's what the M5 competition's top solutions did), not a
shortcut.

Missing lag/rolling values (the first `max(lag periods)` rows of each series,
before enough history exists) are left as NaN and handled by XGBoost's native
missing-value split-finding — not dropped, not zero-filled (zero-filling a
missing lag would silently claim "no prior demand" for a row that actually
just doesn't have that much history yet, which is a different, false claim).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from ml.training.metrics import all_metrics

FEATURE_COLUMNS = [
    "lag_1", "lag_2", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "day_of_week", "month", "disruption_active",
    "product_id", "location_id",
]
TARGET_COLUMN = "demand_quantity"
CATEGORICAL_COLUMNS = ["product_id", "location_id"]


def _prepare_xy(panel: pd.DataFrame, categories: dict[str, list] | None = None) -> tuple[pd.DataFrame, pd.Series, dict]:
    df = panel.copy()
    df["disruption_active"] = df["disruption_active"].astype(int)
    if categories is None:
        categories = {col: sorted(df[col].astype(str).unique().tolist()) for col in CATEGORICAL_COLUMNS}
    for col in CATEGORICAL_COLUMNS:
        df[col] = pd.Categorical(df[col].astype(str), categories=categories[col])
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(float)
    return X, y, categories


def train(panel: pd.DataFrame) -> dict:
    train_df = panel[panel["split"] == "train"]
    val_df = panel[panel["split"] == "val"]
    test_df = panel[panel["split"] == "test"]

    X_train, y_train, categories = _prepare_xy(train_df)
    X_val, y_val, _ = _prepare_xy(val_df, categories)
    X_test, y_test, _ = _prepare_xy(test_df, categories)

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        enable_categorical=True,
        tree_method="hist",
        early_stopping_rounds=25,
        eval_metric="mae",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    def eval_split(X, y, df):
        preds = np.clip(model.predict(X), 0, None)
        return {
            "row_metrics": all_metrics(y.values, preds),
            "predictions": pd.DataFrame(
                {"date": df["date"].values, "product_id": df["product_id"].values, "location_id": df["location_id"].values,
                 "actual": y.values, "predicted": preds}
            ),
        }

    val_eval = eval_split(X_val, y_val, val_df)
    test_eval = eval_split(X_test, y_test, test_df)

    importance = model.get_booster().get_score(importance_type="gain")
    importance_sorted = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))

    return {
        "model": model,
        "categories": categories,
        "trained_pairs": train_df[["product_id", "location_id"]].drop_duplicates().itertuples(index=False, name=None),
        "val_metrics": val_eval["row_metrics"],
        "test_metrics": test_eval["row_metrics"],
        "val_predictions": val_eval["predictions"],
        "test_predictions": test_eval["predictions"],
        "feature_importance_gain": importance_sorted,
        "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else None,
    }


def save_artifacts(result: dict, panel_report: dict, comparison: dict, version: str) -> Path:
    out_dir = Path("ml/artifacts/xgboost_demand") / version
    out_dir.mkdir(parents=True, exist_ok=True)

    result["model"].get_booster().save_model(str(out_dir / "model.json"))
    known_pairs = sorted({(str(p), str(l)) for p, l in result["trained_pairs"]})
    joblib.dump(
        {"categories": result["categories"], "feature_columns": FEATURE_COLUMNS, "known_series": known_pairs},
        out_dir / "preprocessing.pkl",
    )

    with open(out_dir / "feature_list.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    metrics = {
        "val": result["val_metrics"],
        "test": result["test_metrics"],
        "disruption_period_slice": {
            "note": "0 disruption_active rows in this dataset's date range (2010-12 to "
            "2011-12) — none of the acquired/curated disruption events overlap it "
            "(earliest is 2019). The evaluation code path exists (see features.py's "
            "add_disruption_features) and will populate this slice the first time a "
            "training window actually overlaps a flagged disruption window — not "
            "fabricated here.",
            "rows": 0,
        },
        "baseline_comparison": comparison,
        "feature_importance_gain": result["feature_importance_gain"],
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    metadata = {
        "model_name": "xgboost_demand",
        "model_version": version,
        "training_dataset": "data/processed/demand_modeling_panel.csv",
        "training_dataset_rows": panel_report["dense_panel_rows"],
        "training_date": datetime.now(timezone.utc).isoformat(),
        "feature_schema": FEATURE_COLUMNS,
        "best_iteration": result["best_iteration"],
        "xgboost_version": xgb.__version__,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    result["val_predictions"].to_csv(out_dir / "val_predictions.csv", index=False)
    result["test_predictions"].to_csv(out_dir / "test_predictions.csv", index=False)

    return out_dir
