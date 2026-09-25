"""Feature engineering for the demand panel — shared between offline training
(ml/training, Phase 4) and, later, online inference (model-plan.md §6), so this
module has no notebook-only logic and no side effects beyond the dataframe it
returns.

Leakage discipline (brief §6 "data leakage", model-plan.md §3): every lag and
rolling feature for row *t* is built only from rows strictly before *t*. Lags
use `shift(n)`, which is leak-free by construction. Rolling means are computed
on the *already-shifted* series (`shift(1).rolling(window)`), not on the raw
series — the common bug is `rolling(window).mean()` on the unshifted series,
which includes the current row's own value in its own feature. See
backend/tests/test_preprocessing.py::test_rolling_mean_excludes_current_row.
"""
from __future__ import annotations

import pandas as pd

LAG_PERIODS = (1, 2, 7, 14, 28)
ROLLING_WINDOWS = (7, 14, 28)


def add_lag_features(panel: pd.DataFrame, group_cols: list[str], value_col: str = "demand_quantity") -> pd.DataFrame:
    panel = panel.sort_values(group_cols + ["date"]).copy()
    grouped = panel.groupby(group_cols, sort=False)[value_col]
    for period in LAG_PERIODS:
        panel[f"lag_{period}"] = grouped.shift(period)
    return panel


def add_rolling_features(panel: pd.DataFrame, group_cols: list[str], value_col: str = "demand_quantity") -> pd.DataFrame:
    panel = panel.sort_values(group_cols + ["date"]).copy()
    # shift(1) first (drop the current row from its own window), then roll
    # per group using pandas' native SeriesGroupBy.rolling — NOT
    # groupby(...).transform(lambda s: s.rolling(...)), which dispatches one
    # Python-level call per group and is unusably slow with tens of thousands
    # of small (product_id, location_id) groups (verified: >15 min and still
    # running on this dataset's ~155k possible groups before being killed).
    # .rolling() on a groupby object runs in pandas' compiled per-group path.
    shifted = panel.groupby(group_cols, sort=False)[value_col].shift(1)
    group_keys = [panel[c] for c in group_cols]
    n_group_levels = len(group_cols)
    for window in ROLLING_WINDOWS:
        rolled = shifted.groupby(group_keys, sort=False).rolling(window, min_periods=1).mean()
        # rolled's index is (group_key..., original_row_index); drop only the
        # group-key levels so the result realigns to panel's original index
        # regardless of internal processing order (this is what the earlier
        # transform+reset_index(drop=True) version got wrong).
        panel[f"rolling_mean_{window}"] = rolled.reset_index(level=list(range(n_group_levels)), drop=True)
    return panel


def add_calendar_features(panel: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    panel = panel.copy()
    dt = pd.to_datetime(panel[date_col])
    panel["day_of_week"] = dt.dt.dayofweek
    panel["month"] = dt.dt.month
    return panel


def add_disruption_features(
    panel: pd.DataFrame,
    disruptions: pd.DataFrame | None,
    date_col: str = "date",
    location_col: str = "location_id",
) -> pd.DataFrame:
    """Flags rows that fall inside an active disruption window for that
    location. With no `disruptions` table (or none overlapping the panel's
    date range — true for the 2010-2011 UCI training window against our 2021+
    curated/2024 NOAA events), every row gets `disruption_active=False`; the
    column and the join logic are still exercised so the same function works
    unchanged once live simulation dates overlap real disruption windows.
    """
    panel = panel.copy()
    panel["disruption_active"] = False
    if disruptions is None or disruptions.empty:
        return panel

    dt = pd.to_datetime(panel[date_col])
    panel_start, panel_end = dt.min(), dt.max()

    starts = pd.to_datetime(disruptions["start_date"])
    ends = pd.to_datetime(disruptions["end_date"]).fillna(starts)
    # Cheap vectorized pre-filter: keep only events whose window overlaps the
    # panel's date range at all, BEFORE the expensive per-event location scan.
    # This is what makes the function safe to call with a large real event
    # table (e.g. 69,805 NOAA rows) against a panel whose dates don't
    # overlap them at all — the original per-event str.contains() scan over
    # every panel row, for every one of 69,805 events, was measured to still
    # be running after several minutes; this cuts straight to "0 candidates".
    overlapping = disruptions[(ends >= panel_start) & (starts <= panel_end)]
    if overlapping.empty:
        return panel

    for _, event in overlapping.iterrows():
        start = pd.to_datetime(event["start_date"])
        end = pd.to_datetime(event["end_date"]) if pd.notna(event.get("end_date")) else start
        loc_match = panel[location_col].astype(str).str.contains(str(event["location"]), case=False, na=False) if location_col in panel else True
        window_match = (dt >= start) & (dt <= end)
        panel.loc[window_match & loc_match, "disruption_active"] = True
    return panel


def time_based_split(
    panel: pd.DataFrame, date_col: str = "date", train_frac: float = 0.70, val_frac: float = 0.15
) -> pd.DataFrame:
    """Chronological split — NOT random k-fold (explicitly disallowed, brief §6
    / model-plan.md §3). Cutoffs are computed on the sorted unique dates, so
    every row for a given date lands in the same split regardless of which
    product/location group it belongs to.
    """
    panel = panel.copy()
    dates = pd.to_datetime(panel[date_col])
    unique_dates = sorted(dates.unique())
    n = len(unique_dates)
    train_cutoff = unique_dates[int(n * train_frac) - 1]
    val_cutoff = unique_dates[int(n * (train_frac + val_frac)) - 1]

    conditions = [dates <= train_cutoff, (dates > train_cutoff) & (dates <= val_cutoff), dates > val_cutoff]
    panel["split"] = pd.Series(pd.NA, index=panel.index, dtype="object")
    for cond, label in zip(conditions, ["train", "val", "test"]):
        panel.loc[cond, "split"] = label
    return panel


def build_feature_panel(demand_daily: pd.DataFrame, disruptions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full pipeline: lag + rolling + calendar + disruption features, then the
    time-based split. `demand_daily` must already be aggregated to one row per
    (date, product_id, location_id) — see demand.py.
    """
    group_cols = ["product_id", "location_id"]
    panel = add_lag_features(demand_daily, group_cols)
    panel = add_rolling_features(panel, group_cols)
    panel = add_calendar_features(panel)
    panel = add_disruption_features(panel, disruptions)
    panel = time_based_split(panel)
    return panel
