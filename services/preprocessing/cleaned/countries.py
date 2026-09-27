"""The country bridge table: one code that the retail labels, the World Bank tariffs, the port index and the suppliers all map onto.

The audit found three incompatible country encodings: English labels in the retail data ('EIRE', 'RSA', 'Channel Islands'),
2-letter codes in the World Port Index and ISO-3 in the World Bank file. Only 'USA' matched by coincidence. `country.iso3` is the
key everything else points at.

The 217 real countries and their names, regions and income groups come from the World Bank's own metadata file (REAL). The
ISO-2 codes and the retail labels are mapped by hand for the countries this project's data actually uses (AUTHORED, assumption
A19) because no ISO 3166 table is available offline in this workspace; every ISO-3 written below is checked against the World
Bank list when the table is built, so a typo fails the build instead of producing an orphan key.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.services.preprocessing.cleaned import RAW_DIR, REAL

WDI_DIR = RAW_DIR / "tariffs_worldbank_wdi"
WDI_METADATA = WDI_DIR / "Metadata_Country_API_TM.TAX.MRCH.WM.AR.ZS_DS2_en_csv_v2_35363.csv"

# the retail file's 'Country' -> ISO-3 (the World Bank uses CHI for the Channel Islands, which ISO 3166 does not)
RETAIL_LABEL_TO_ISO3 = {
    "Australia": "AUS", "Austria": "AUT", "Bahrain": "BHR", "Belgium": "BEL", "Brazil": "BRA", "Canada": "CAN", "Channel Islands": "CHI",
    "Cyprus": "CYP", "Czech Republic": "CZE", "Denmark": "DNK", "EIRE": "IRL", "Finland": "FIN", "France": "FRA", "Germany": "DEU",
    "Greece": "GRC", "Hong Kong": "HKG", "Iceland": "ISL", "Israel": "ISR", "Italy": "ITA", "Japan": "JPN", "Lebanon": "LBN",
    "Lithuania": "LTU", "Malta": "MLT", "Netherlands": "NLD", "Norway": "NOR", "Poland": "POL", "Portugal": "PRT", "RSA": "ZAF",
    "Saudi Arabia": "SAU", "Singapore": "SGP", "Spain": "ESP", "Sweden": "SWE", "Switzerland": "CHE", "USA": "USA",
    "United Arab Emirates": "ARE", "United Kingdom": "GBR",
}

# ISO-3 -> ISO-2 for the countries the retail data, the supplier roster and the route ports use
ISO3_TO_ISO2 = {
    "AUS": "AU", "AUT": "AT", "BHR": "BH", "BEL": "BE", "BRA": "BR", "CAN": "CA", "CYP": "CY", "CZE": "CZ", "DNK": "DK", "IRL": "IE",
    "FIN": "FI", "FRA": "FR", "DEU": "DE", "GRC": "GR", "HKG": "HK", "ISL": "IS", "ISR": "IL", "ITA": "IT", "JPN": "JP", "LBN": "LB",
    "LTU": "LT", "MLT": "MT", "NLD": "NL", "NOR": "NO", "POL": "PL", "PRT": "PT", "ZAF": "ZA", "SAU": "SA", "SGP": "SG", "ESP": "ES",
    "SWE": "SE", "CHE": "CH", "USA": "US", "ARE": "AE", "GBR": "GB", "CHN": "CN", "IND": "IN", "VNM": "VN", "TUR": "TR", "EGY": "EG",
}
ISO2_TO_ISO3 = {v: k for k, v in ISO3_TO_ISO2.items()}


def load_wdi_countries(metadata_path: Path = WDI_METADATA) -> pd.DataFrame:
    """The World Bank's country rows with a Region; rows without one are regional aggregates ('World', 'European Union')."""
    meta = pd.read_csv(metadata_path)
    real = meta[meta["Region"].notna()]
    return pd.DataFrame(
        {"iso3": real["Country Code"], "name": real["TableName"], "region": real["Region"], "income_group": real["IncomeGroup"]}
    ).reset_index(drop=True)


def build_country_table() -> pd.DataFrame:
    table = load_wdi_countries()
    known = set(table["iso3"])
    unknown = sorted(({*RETAIL_LABEL_TO_ISO3.values(), *ISO3_TO_ISO2} - known))
    assert not unknown, f"ISO-3 codes not in the World Bank country list (typo?): {unknown}"
    label_by_iso3 = {iso3: label for label, iso3 in RETAIL_LABEL_TO_ISO3.items()}
    table["iso2"] = table["iso3"].map(ISO3_TO_ISO2)
    table["retail_label"] = table["iso3"].map(label_by_iso3)
    table["provenance"] = REAL
    return table[["iso3", "iso2", "name", "region", "income_group", "retail_label", "provenance"]]


COUNTRY_COLUMN_PROVENANCE = {"iso2": "AUTHORED", "retail_label": "AUTHORED"}
