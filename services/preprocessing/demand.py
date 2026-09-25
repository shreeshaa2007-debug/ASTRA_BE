"""DEMAND preprocessing — data/raw/demand_uci_online_retail -> internal DEMAND
schema. Handles the quality issues recorded in data/dataset_registry.yaml:
cancelled-order rows, missing Description, non-product StockCodes, and the
raw dataset's transaction grain (needs aggregating up to date x product x
location before it matches the internal schema).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.services.preprocessing.utils import iqr_outlier_bounds

RAW_PATH = Path("data/raw/demand_uci_online_retail/Online_Retail.xlsx")

# StockCodes that are fees/adjustments, not real products — inconsistent
# product identifiers per brief §6; excluded from demand, not silently kept.
_NON_PRODUCT_CODES = {"POST", "DOT", "M", "MANUAL", "BANK CHARGES", "PADS", "CRUK", "AMAZONFEE", "C2"}


def load_raw() -> pd.DataFrame:
    return pd.read_excel(RAW_PATH, engine="openpyxl")


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Row-level cleaning. Returns (clean_df, quality_report) — the report is
    written into the pipeline manifest rather than silently discarded, so the
    cleaning decisions are auditable.
    """
    df = raw.copy()
    report: dict = {"input_rows": len(df)}

    df = df.drop_duplicates()
    report["exact_duplicates_dropped"] = report["input_rows"] - len(df)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    invalid_dates = df["InvoiceDate"].isna().sum()
    df = df[df["InvoiceDate"].notna()]
    report["invalid_dates_dropped"] = int(invalid_dates)

    is_cancellation = df["InvoiceNo"].astype(str).str.startswith("C")
    is_non_positive_qty = df["Quantity"] <= 0
    cancelled_or_return = is_cancellation | is_non_positive_qty
    report["cancelled_or_return_rows_excluded"] = int(cancelled_or_return.sum())
    df = df[~cancelled_or_return]

    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    is_non_product = df["StockCode"].isin(_NON_PRODUCT_CODES)
    report["non_product_stockcodes_excluded"] = int(is_non_product.sum())
    df = df[~is_non_product]

    df["Country"] = df["Country"].astype(str).str.strip()
    unspecified = df["Country"].isin(["Unspecified", "European Community", "nan"])
    report["unspecified_location_rows_excluded"] = int(unspecified.sum())
    df = df[~unspecified]

    low, high = iqr_outlier_bounds(df["Quantity"], k=3.0)
    is_outlier = (df["Quantity"] < low) | (df["Quantity"] > high)
    report["quantity_outliers_flagged"] = int(is_outlier.sum())
    df["is_quantity_outlier"] = is_outlier  # flagged, not dropped — a large
    # wholesale order is real demand, not necessarily an error

    report["output_rows"] = len(df)
    return df, report


def aggregate_to_daily_panel(clean: pd.DataFrame) -> pd.DataFrame:
    """Transaction rows -> one row per (date, product_id, location_id), the
    internal DEMAND grain."""
    df = clean.copy()
    df["date"] = df["InvoiceDate"].dt.date
    panel = (
        df.groupby(["date", "StockCode", "Country"], as_index=False)["Quantity"]
        .sum()
        .rename(columns={"StockCode": "product_id", "Country": "location_id", "Quantity": "demand_quantity"})
    )
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


def build_demand_panel() -> tuple[pd.DataFrame, dict]:
    raw = load_raw()
    clean, report = clean_transactions(raw)
    panel = aggregate_to_daily_panel(clean)
    report["daily_panel_rows"] = len(panel)
    report["unique_products"] = panel["product_id"].nunique()
    report["unique_locations"] = panel["location_id"].nunique()
    report["date_range"] = [str(panel["date"].min().date()), str(panel["date"].max().date())]
    return panel, report
