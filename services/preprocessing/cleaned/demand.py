"""Corrected demand: sale lines (net of cancellations) -> daily demand, the modeling panel and the product master.

Three tables come out of one set of net sale lines, so they cannot disagree:

    demand.csv                   one row per (date, product, country) with net demand > 0, no features
    demand_modeling_panel.csv    the 40 forecastable series on a dense daily calendar, with lag/rolling features and a time split
    product.csv                  one row per product that was ever sold

`demand.csv` deliberately has no lag features. Phase 3's file computed them on the SPARSE series, so "lag_1" meant "the previous
day that had a sale" (see ml/training/prepare_modeling_data.py); features belong on the dense panel, where they mean what
their names say.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.services.preprocessing.cleaned import DERIVED, REAL
from backend.services.preprocessing.cleaned import assumptions as A
from backend.services.preprocessing.cleaned.countries import RETAIL_LABEL_TO_ISO3
from backend.services.preprocessing.features import build_feature_panel


def attach_country(lines: pd.DataFrame) -> pd.DataFrame:
    """Adds `country_iso3`. A label that is not mapped fails loudly: it would otherwise become an orphan key."""
    out = lines.copy()
    out["country_iso3"] = out["location_id"].map(RETAIL_LABEL_TO_ISO3)
    unmapped = sorted(out.loc[out["country_iso3"].isna(), "location_id"].unique())
    assert not unmapped, f"retail country labels with no ISO-3 mapping: {unmapped}"
    return out


def build_demand_daily(lines: pd.DataFrame) -> pd.DataFrame:
    """Net demand per (date, product, country). A line whose whole quantity was cancelled contributes nothing."""
    kept = lines[lines["quantity_net"] > 0].copy()
    kept["date"] = kept["order_ts"].dt.normalize()
    daily = (
        kept.groupby(["date", "product_id", "location_id", "country_iso3"], as_index=False)
        .agg(demand_quantity=("quantity_net", "sum"), n_order_lines=("quantity_net", "size"), n_outlier_lines=("is_quantity_outlier", "sum"))
        .sort_values(["date", "product_id", "location_id"], kind="stable")
        .reset_index(drop=True)
    )
    daily["n_outlier_lines"] = daily["n_outlier_lines"].astype("int64")
    daily["provenance"] = DERIVED
    return daily


def select_series(daily: pd.DataFrame) -> pd.DataFrame:
    """The forecastable series: enough active days to have a pattern, then the largest by net volume. Ties are broken by id so the
    choice is reproducible (Phase 3 left ties to the sort's whim)."""
    stats = (
        daily.groupby(["product_id", "location_id"])
        .agg(total_demand=("demand_quantity", "sum"), active_days=("date", "nunique"))
        .reset_index()
    )
    eligible = stats[stats["active_days"] >= A.MODELING_MIN_ACTIVE_DAYS]
    return eligible.sort_values(["total_demand", "product_id", "location_id"], ascending=[False, True, True]).head(A.MODELING_TOP_K).reset_index(drop=True)


def densify(daily: pd.DataFrame, selected: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Every selected series on every calendar day; a day without a sale is zero demand, not a missing row."""
    frames = []
    for key in selected.itertuples(index=False):
        series = daily[(daily["product_id"] == key.product_id) & (daily["location_id"] == key.location_id)]
        values = series.set_index("date")["demand_quantity"].reindex(calendar, fill_value=0)
        frames.append(pd.DataFrame({"date": calendar, "product_id": key.product_id, "location_id": key.location_id, "demand_quantity": values.to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def build_modeling_panel(daily: pd.DataFrame, events: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(panel, selected series, report). `events` needs start_date, end_date, location; it can be empty."""
    selected = select_series(daily)
    calendar = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    dense = densify(daily, selected, calendar)
    panel = build_feature_panel(dense, events)

    flags = daily[["date", "product_id", "location_id", "n_outlier_lines"]]
    panel = panel.merge(flags, on=["date", "product_id", "location_id"], how="left")
    panel["has_outlier_line"] = panel.pop("n_outlier_lines").fillna(0).astype(int) > 0
    panel["provenance"] = DERIVED
    panel = panel.sort_values(["product_id", "location_id", "date"], kind="stable").reset_index(drop=True)

    stats = daily.groupby(["product_id", "location_id"])["date"].nunique()
    report = {
        "series_selected": len(selected),
        "series_eligible_by_active_days": int((stats >= A.MODELING_MIN_ACTIVE_DAYS).sum()),
        "series_total": len(stats),
        "calendar_days": len(calendar),
        "calendar": [str(calendar.min().date()), str(calendar.max().date())],
        "panel_rows": len(panel),
        "split_counts": panel["split"].value_counts().to_dict(),
        "disruption_active_rows": int(panel["disruption_active"].sum()),
        "rows_with_outlier_line": int(panel["has_outlier_line"].sum()),
        "locations": sorted(panel["location_id"].unique().tolist()),
    }
    return panel, selected, report


def build_product_table(lines: pd.DataFrame, forecast_products: set[str]) -> pd.DataFrame:
    """One row per product that has any sale line, so every line's `product_id` has a parent. Name and id are REAL; price, volume and
    ABC class are DERIVED (recorded per column in the data dictionary)."""
    names = (
        lines.dropna(subset=["description"]).assign(description=lambda d: d["description"].astype(str).str.strip())
        .groupby(["product_id", "description"]).size().reset_index(name="n")
        .sort_values(["product_id", "n", "description"], ascending=[True, False, True])
        .drop_duplicates("product_id").set_index("product_id")["description"]
    )
    kept = lines[lines["quantity_net"] > 0].copy()
    kept["revenue"] = kept["quantity_net"] * kept["unit_price_gbp"]
    kept["date"] = kept["order_ts"].dt.normalize()
    agg = kept.groupby("product_id").agg(
        avg_unit_price_gbp=("unit_price_gbp", "median"), net_units=("quantity_net", "sum"), net_revenue_gbp=("revenue", "sum"),
        active_days=("date", "nunique"), first_sale_date=("date", "min"), last_sale_date=("date", "max"),
    )
    products = pd.DataFrame(index=pd.Index(sorted(lines["product_id"].unique()), name="product_id")).join(agg)
    products["product_name"] = names.reindex(products.index)
    products["net_units"] = products["net_units"].fillna(0).astype("int64")
    products["net_revenue_gbp"] = products["net_revenue_gbp"].fillna(0.0).round(2)
    products["active_days"] = products["active_days"].fillna(0).astype("int64")

    ranked = products["net_revenue_gbp"].sort_values(ascending=False)
    cumulative = ranked.cumsum() / ranked.sum()
    a_cut, b_cut = A.ABC_CUTOFFS
    # a product is A if the revenue ranked above it is still under the cutoff, so the product that crosses 80% is still an A
    before = cumulative - ranked / ranked.sum()
    abc = np.where(before < a_cut, "A", np.where(before < b_cut, "B", "C"))
    products["abc_class"] = pd.Series(abc, index=ranked.index).reindex(products.index)
    products.loc[products["net_revenue_gbp"] <= 0, "abc_class"] = "C"
    products["is_forecast_series"] = products.index.isin(forecast_products)
    products["provenance"] = REAL
    products = products.reset_index()
    return products[["product_id", "product_name", "avg_unit_price_gbp", "abc_class", "net_units", "net_revenue_gbp", "active_days",
                     "first_sale_date", "last_sale_date", "is_forecast_series", "provenance"]]


PRODUCT_COLUMN_PROVENANCE = {
    "avg_unit_price_gbp": DERIVED, "abc_class": DERIVED, "net_units": DERIVED, "net_revenue_gbp": DERIVED, "active_days": DERIVED,
    "first_sale_date": DERIVED, "last_sale_date": DERIVED, "is_forecast_series": DERIVED,
}
