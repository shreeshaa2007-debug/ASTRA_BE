"""Tariffs and disruption events: REAL observations kept apart from AUTHORED reference events and from SYNTHETIC scenarios.

TARIFFS are the World Bank's weighted-mean applied tariff, country and year, exactly as published. Nothing is filled, interpolated
or rounded: a country with no value stays absent from `tariff` and appears in `tariff_latest` as NO_DATA; a country whose latest
value is older than the reference year is STALE, with the number of years it lags. Two flags mark values a reader should look
at (above 50%, or a move of 10 points or more from the previous observation) without changing them. The flags are informational: the
US jump from 1.59% to 13.78% in 2019 is what the World Bank publishes, and it is flagged, not corrected.

DISRUPTION EVENTS come in three separate kinds, and the tables never mix them:

    weather_event_noaa    69,801 US storm events of 2024, REAL, with the parsed and derived fields the Phase 3 file lacked
    disruption_event      the four historical events the project hand-wrote (AUTHORED), each with a verification status
    scenario*             hypothetical stress tests (SYNTHETIC) in suppliers.py; never mixed with either of the above

The NOAA table is a US weather record. It shares no place or year with the demand data or the shipping lanes, so it is a
statistical reference for the Sensing agent's vocabulary and for event durations, not a source of in-network disruptions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.services.preprocessing.cleaned import AUTHORED, RAW_DIR, REAL
from backend.services.preprocessing.cleaned import assumptions as A
from backend.services.preprocessing.utils import parse_money_string

WDI_DATA = RAW_DIR / "tariffs_worldbank_wdi" / "API_TM.TAX.MRCH.WM.AR.ZS_DS2_en_csv_v2_35363.csv"
NOAA_CSV = RAW_DIR / "weather_noaa_storm_events" / "StormEvents_details_2024.csv"
TARIFF_SOURCE = "World Bank WDI TM.TAX.MRCH.WM.AR.ZS (tariff rate, applied, weighted mean, all products)"
NOAA_SOURCE = "NOAA NCEI Storm Events Database, 2024"


# --------------------------------------------------------------------------- tariffs
def build_tariff_tables(countries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    wide = pd.read_csv(WDI_DATA, skiprows=4)
    real = set(countries["iso3"])
    wide = wide[wide["Country Code"].isin(real)]
    years = [c for c in wide.columns if c.isdigit()]
    long = wide.melt(id_vars=["Country Code"], value_vars=years, var_name="effective_year", value_name="tariff_rate_pct")
    long["effective_year"] = long["effective_year"].astype(int)
    long = long.dropna(subset=["tariff_rate_pct"]).rename(columns={"Country Code": "origin_iso3"})
    long = long.sort_values(["origin_iso3", "effective_year"]).reset_index(drop=True)

    previous = long.groupby("origin_iso3")["tariff_rate_pct"].shift(1)
    extreme = long["tariff_rate_pct"] > A.TARIFF_EXTREME_PCT
    jump = (long["tariff_rate_pct"] - previous).abs() >= A.TARIFF_JUMP_PP
    long["quality_flag"] = np.select([extreme & jump, extreme, jump], ["EXTREME_VALUE;JUMP_GE_10PP", "EXTREME_VALUE", "JUMP_GE_10PP"], default="OK")

    tariff = long.assign(dest_iso3="ANY", hs_section="ALL", is_country_level_proxy=True, source=TARIFF_SOURCE, provenance=REAL)[
        ["origin_iso3", "dest_iso3", "hs_section", "effective_year", "tariff_rate_pct", "is_country_level_proxy", "quality_flag", "source", "provenance"]
    ]

    reporting = long.groupby("effective_year").size()
    reference_year = int(reporting[reporting >= A.TARIFF_BROAD_COVERAGE_MIN_COUNTRIES].index.max())
    latest = long.sort_values("effective_year").groupby("origin_iso3").tail(1).set_index("origin_iso3")
    out = countries[["iso3", "name"]].rename(columns={"iso3": "origin_iso3", "name": "country_name"}).set_index("origin_iso3")
    out = out.join(latest[["effective_year", "tariff_rate_pct", "quality_flag"]].rename(columns={"effective_year": "latest_year"}))
    out["latest_year"] = out["latest_year"].astype("Int64")
    out["reference_year"] = reference_year
    out["staleness_years"] = (reference_year - out["latest_year"]).astype("Int64")
    no_data = out["latest_year"].isna().to_numpy(dtype=bool)
    stale = (out["latest_year"] < reference_year).fillna(False).to_numpy(dtype=bool)  # a comparison with <NA> is <NA>, not False
    out["data_status"] = np.select([no_data, stale], ["NO_DATA", "STALE"], default="CURRENT")
    out["source"] = TARIFF_SOURCE
    out["provenance"] = REAL
    tariff_latest = out.reset_index()[["origin_iso3", "country_name", "latest_year", "tariff_rate_pct", "reference_year", "staleness_years",
                                       "data_status", "quality_flag", "source", "provenance"]]

    report = {
        "reference_year": reference_year, "observations": len(tariff), "countries_with_data": int(tariff["origin_iso3"].nunique()),
        "countries_total": len(countries), "status_counts": tariff_latest["data_status"].value_counts().to_dict(),
        "flag_counts": tariff["quality_flag"].value_counts().to_dict(), "values_altered_or_filled": 0,
        "max_staleness_years": int(tariff_latest["staleness_years"].max()),
    }
    return tariff, tariff_latest, report


# --------------------------------------------------------------------------- NOAA weather
def _severity(row_damage: float, casualties: int) -> str:
    """The Phase 3 rule, unchanged: any casualty or >= $10M is CRITICAL, >= $1M HIGH, >= $10k MEDIUM, else LOW."""
    if casualties > 0 or row_damage >= 10_000_000:
        return "CRITICAL"
    if row_damage >= 1_000_000:
        return "HIGH"
    if row_damage >= 10_000:
        return "MEDIUM"
    return "LOW"


def build_weather_event_table() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(NOAA_CSV, low_memory=False)
    property_usd = raw["DAMAGE_PROPERTY"].apply(parse_money_string)
    crops_usd = raw["DAMAGE_CROPS"].apply(parse_money_string)
    casualties = raw[["INJURIES_DIRECT", "INJURIES_INDIRECT", "DEATHS_DIRECT", "DEATHS_INDIRECT"]].sum(axis=1).astype(int)
    total = property_usd + crops_usd
    begin = pd.to_datetime(raw["BEGIN_DATE_TIME"], format="%d-%b-%y %H:%M:%S", errors="coerce")
    end = pd.to_datetime(raw["END_DATE_TIME"], format="%d-%b-%y %H:%M:%S", errors="coerce")
    valid = begin.notna()
    table = pd.DataFrame(
        {
            "event_id": "NOAA-" + raw["EVENT_ID"].astype(str), "event_type": raw["EVENT_TYPE"], "state": raw["STATE"].str.title(),
            "location": raw["STATE"].str.title() + ", USA", "start_ts": begin, "end_ts": end,
            "severity": [_severity(t, c) for t, c in zip(total, casualties)], "damage_property_usd": property_usd, "damage_crops_usd": crops_usd,
            "casualties": casualties, "begin_lat": raw["BEGIN_LAT"], "begin_lon": raw["BEGIN_LON"], "magnitude": raw["MAGNITUDE"],
            "magnitude_type": raw["MAGNITUDE_TYPE"], "source": NOAA_SOURCE, "provenance": REAL,
        }
    )[valid].reset_index(drop=True)
    report = {
        "rows": len(table), "input_rows": len(raw), "dropped_unparseable_dates": int((~valid).sum()),
        "date_range": [str(table["start_ts"].min()), str(table["start_ts"].max())], "event_types": int(table["event_type"].nunique()),
        "severity_counts": table["severity"].value_counts().to_dict(), "rows_without_coordinates": int(table["begin_lat"].isna().sum()),
        "end_before_start": int((table["end_ts"] < table["start_ts"]).sum()),
    }
    return table, report


WEATHER_COLUMN_PROVENANCE = {
    "state": "DERIVED", "location": "DERIVED", "severity": "DERIVED", "damage_property_usd": "DERIVED", "damage_crops_usd": "DERIVED", "casualties": "DERIVED",
}


# --------------------------------------------------------------------------- authored reference events
# Four events the project wrote down by hand about publicly reported real-world events. They are AUTHORED, not REAL: they are the
# authors' record of the reporting, not a dataset in data/raw. Dates and delay estimates are the authors' and have not been checked
# against a source file here, except where a status below says the World Bank data corroborates them.
_AUTHORED_EVENTS = [
    {"event_id": "CURATED-SUEZ-2021", "event_type": "Port/Canal Closure", "location": "Suez Canal, Egypt", "start_ts": "2021-03-23",
     "end_ts": "2021-03-29", "severity": "CRITICAL", "estimated_delay_days": 6.0,
     "source_note": "Container ship aground in the Suez Canal; traffic resumed after about six days. Widely reported; the demo scenario "
                    "dramatises the duration to 10 days. Delay = the blockage length, an estimate."},
    {"event_id": "CURATED-LALB-CONGESTION-2021", "event_type": "Port Congestion", "location": "Los Angeles/Long Beach, USA", "start_ts": "2021-09-01",
     "end_ts": "2022-03-01", "severity": "HIGH", "estimated_delay_days": 14.0,
     "source_note": "Container-ship anchorage queues at the ports. Window and delay are the authors' round-number estimates."},
    {"event_id": "CURATED-US-TARIFF-2019", "event_type": "Tariff Increase", "location": "United States (imports from China)", "start_ts": "2019-05-10",
     "end_ts": None, "severity": "HIGH", "estimated_delay_days": None,
     "source_note": "US tariff increase on imports from China; open-ended."},
    {"event_id": "CURATED-TEXAS-WINTER-STORM-URI-2021", "event_type": "Severe Weather", "location": "Texas, USA", "start_ts": "2021-02-13",
     "end_ts": "2021-02-17", "severity": "CRITICAL", "estimated_delay_days": 5.0,
     "source_note": "Winter Storm Uri: grid, petrochemical and supplier shutdowns across Texas. Delay is an estimate."},
]


def build_authored_event_tables(tariff: pd.DataFrame, routes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(disruption_event, disruption_impact, report). The tariff event's status is computed from the World Bank series."""
    events = pd.DataFrame(_AUTHORED_EVENTS)
    events["start_ts"] = pd.to_datetime(events["start_ts"])
    events["end_ts"] = pd.to_datetime(events["end_ts"])
    events["status"] = "HISTORICAL"
    events["verification_status"] = "UNVERIFIED_IN_WORKSPACE"

    us = tariff[tariff["origin_iso3"] == "USA"].set_index("effective_year")["tariff_rate_pct"]
    jump = float(us.get(2019, np.nan) - us.get(2018, np.nan))
    corroborated = bool(jump >= A.TARIFF_JUMP_PP)
    is_tariff = events["event_id"] == "CURATED-US-TARIFF-2019"
    events.loc[is_tariff, "verification_status"] = "CORROBORATED_BY_WDI" if corroborated else "NOT_CORROBORATED_BY_WDI"
    events.loc[is_tariff, "source_note"] += f" World Bank series: US {us.get(2018):.2f}% in 2018, {us.get(2019):.2f}% in 2019."
    events["provenance"] = AUTHORED

    suez_lanes = routes.loc[routes["chokepoints"].str.split("|").apply(lambda passes: "Suez Canal" in passes), "route_id"].tolist()
    impact = pd.DataFrame(
        [
            {"event_id": "CURATED-SUEZ-2021", "entity_type": "ROUTE", "entity_id": rid, "delay_days": 6.0, "capacity_factor": 0.0,
             "basis": "every lane through the canal is blocked while the canal is (an inference); the six days are the authored estimate", "provenance": AUTHORED}
            for rid in suez_lanes
        ]
    )
    events = events[["event_id", "event_type", "location", "start_ts", "end_ts", "severity", "estimated_delay_days", "status", "source_note",
                     "verification_status", "provenance"]]
    report = {
        "events": len(events), "impact_rows": len(impact), "tariff_event_us_jump_pp_2018_to_2019": round(jump, 2), "tariff_event_corroborated": corroborated,
        "verification_status_counts": events["verification_status"].value_counts().to_dict(),
    }
    return events, impact, report
