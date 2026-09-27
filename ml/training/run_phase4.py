"""Phase 4 entry point: prepare data -> baselines -> XGBoost -> per-series
comparison -> save artifacts. Run with: python -m ml.training.run_phase4

Per brief §7 / model-plan.md §1: XGBoost does not win by assertion. Every
baseline (naive, seasonal naive, moving average, exponential smoothing) and
XGBoost are scored per series on the validation set's WAPE, and only the
series where XGBoost actually has the lowest WAPE count as "XGBoost wins" —
the summary this prints is that count out of 40, not a blanket claim.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ml.training import baselines as baselines_mod
from ml.training.metrics import wape
from ml.training.prepare_modeling_data import build_modeling_panel
from ml.training.train_xgboost import save_artifacts, train

VERSION = "2026.09.1"


def per_series_wape(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for (pid, loc), group in df.groupby(["product_id", "location_id"]):
        valid = group.dropna(subset=[pred_col, "demand_quantity"])
        if valid.empty:
            continue
        w = wape(valid["demand_quantity"].values, valid[pred_col].values)
        rows.append({"product_id": pid, "location_id": loc, "model": pred_col, "wape": w})
    return pd.DataFrame(rows)


def compare_models_per_series(baseline_preds: pd.DataFrame, xgb_val_preds: pd.DataFrame) -> dict:
    xgb_for_join = xgb_val_preds.rename(columns={"predicted": "pred_xgboost", "actual": "demand_quantity"})
    val_baselines = baseline_preds[baseline_preds["split"] == "val"]
    merged = val_baselines.merge(
        xgb_for_join[["date", "product_id", "location_id", "pred_xgboost"]],
        on=["date", "product_id", "location_id"],
        how="inner",
    )

    model_cols = ["pred_naive", "pred_seasonal_naive", "pred_moving_average", "pred_exp_smoothing", "pred_xgboost"]
    all_wapes = []
    for col in model_cols:
        w = per_series_wape(merged, col)
        w["model"] = col
        all_wapes.append(w)
    all_wapes_df = pd.concat(all_wapes, ignore_index=True)

    pivot = all_wapes_df.pivot_table(index=["product_id", "location_id"], columns="model", values="wape")
    pivot = pivot.dropna(how="all")
    winner = pivot.idxmin(axis=1)
    win_counts = winner.value_counts().to_dict()

    return {
        "series_compared": len(pivot),
        "win_counts_by_model_lowest_val_wape": win_counts,
        "xgboost_series_win_count": win_counts.get("pred_xgboost", 0),
        "xgboost_win_rate": round(win_counts.get("pred_xgboost", 0) / len(pivot), 3) if len(pivot) else None,
        "mean_wape_by_model": {c: round(float(v), 2) for c, v in pivot.mean().items()},
    }


def run() -> dict:
    print("[1/4] preparing dense modeling panel...")
    panel, panel_report = build_modeling_panel()
    print(json.dumps(panel_report, indent=2))

    print("\n[2/4] computing baselines (naive, seasonal naive, moving average, exp smoothing)...")
    baseline_preds = baselines_mod.compute_all_baselines(panel)

    print("\n[3/4] training XGBoost...")
    result = train(panel)
    print(f"val:  {result['val_metrics']}")
    print(f"test: {result['test_metrics']}")

    print("\n[4/4] comparing XGBoost against baselines, per series (val set)...")
    comparison = compare_models_per_series(baseline_preds, result["val_predictions"])
    print(json.dumps(comparison, indent=2))

    out_dir = save_artifacts(result, panel_report, comparison, VERSION)
    baseline_preds.to_csv(out_dir / "baseline_predictions.csv", index=False)
    print(f"\nArtifacts written to {out_dir}")

    return {"panel_report": panel_report, "test_metrics": result["test_metrics"], "comparison": comparison, "artifact_dir": str(out_dir)}


if __name__ == "__main__":
    run()
