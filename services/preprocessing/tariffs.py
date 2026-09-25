"""TARIFFS preprocessing — data/raw/tariffs_worldbank_wdi -> internal TARIFFS
schema.

Grain mismatch (recorded in dataset_registry.yaml and schemas/entities.py):
the World Bank indicator is country-year; the internal schema wants
country-pair x product. This module does not pretend otherwise — it reshapes
to long format, drops regional aggregates (via the country metadata file's
Region column, the standard WDI trick: aggregates have a blank Region),
and fills `destination_country="ANY"` / `product_category="ALL"` with
`is_country_level_proxy=True` so downstream code has one consistent schema to
read while still being able to tell it's a proxy.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/tariffs_worldbank_wdi/API_TM.TAX.MRCH.WM.AR.ZS_DS2_en_csv_v2_35363.csv")
METADATA_PATH = Path("data/raw/tariffs_worldbank_wdi/Metadata_Country_API_TM.TAX.MRCH.WM.AR.ZS_DS2_en_csv_v2_35363.csv")


def build_tariffs_table() -> tuple[pd.DataFrame, dict]:
    wide = pd.read_csv(DATA_PATH, skiprows=4)
    meta = pd.read_csv(METADATA_PATH)
    report: dict = {"input_countries_and_aggregates": len(wide)}

    real_country_codes = set(meta.loc[meta["Region"].notna(), "Country Code"])
    report["regional_aggregates_dropped"] = len(wide) - len(wide[wide["Country Code"].isin(real_country_codes)])
    wide = wide[wide["Country Code"].isin(real_country_codes)]

    year_cols = [c for c in wide.columns if c.isdigit()]
    long = wide.melt(
        id_vars=["Country Name", "Country Code"],
        value_vars=year_cols,
        var_name="effective_year",
        value_name="tariff_rate",
    )
    long["effective_year"] = long["effective_year"].astype(int)
    report["rows_before_missing_drop"] = len(long)
    long = long[long["tariff_rate"].notna()]
    report["rows_after_missing_drop"] = len(long)

    out = pd.DataFrame(
        {
            "origin_country": long["Country Code"],
            "destination_country": "ANY",
            "product_category": "ALL",
            "effective_year": long["effective_year"],
            "tariff_rate": long["tariff_rate"].round(4),
            "is_country_level_proxy": True,
        }
    )
    report["output_rows"] = len(out)
    report["countries_with_any_data"] = out["origin_country"].nunique()
    report["most_recent_year_with_broad_coverage"] = int(
        long.groupby("effective_year").size().loc[lambda s: s > 100].index.max()
    )
    return out.reset_index(drop=True), report


def latest_rate_by_country(tariffs: pd.DataFrame) -> pd.DataFrame:
    """One row per country: its most recent available tariff_rate. This is
    what the Sourcing Agent's calculate_landed_cost() tool actually needs
    (agent-plan.md) — a single current-ish rate per origin country, not the
    full 1960-2025 history.
    """
    idx = tariffs.groupby("origin_country")["effective_year"].idxmax()
    return tariffs.loc[idx].reset_index(drop=True)
