"""DISRUPTIONS preprocessing — two real sources, combined:

1. NOAA Storm Events 2024 (data/raw/weather_noaa_storm_events) — real US
   weather events, grounds the "Severe Weather" scenario's shape. Handles the
   brief's "inconsistent units" case directly: DAMAGE_PROPERTY/DAMAGE_CROPS
   are magnitude-suffixed strings ("25.00K"), parsed via
   preprocessing.utils.parse_money_string.
2. A small curated log of real, publicly documented non-weather disruption
   events (Suez 2021, etc.) — per docs/data-plan.md, disruption events are
   inherently sparse, so this is a short authored list with cited real dates,
   not a scraped dataset. Every entry links a real source in its comment.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.services.preprocessing.utils import parse_money_string

NOAA_PATH = Path("data/raw/weather_noaa_storm_events/StormEvents_details_2024.csv")

# Real, publicly documented events — one line each, source cited in-line.
# These ground the non-weather scenario types (Suez closure, tariff spike,
# port congestion) in actual history; the *simulator's* scenario parameters
# (Phase 17) may dramatize duration/severity for the demo, but this log
# records what really happened, not the demo's tuned version.
_CURATED_EVENTS = [
    {
        "event_id": "CURATED-SUEZ-2021",
        "event_type": "Port/Canal Closure",
        "location": "Suez Canal, Egypt",
        "start_date": "2021-03-23",
        "end_date": "2021-03-29",
        "severity": "CRITICAL",
        "affected_route": "SHA-ROT-SUEZ",
        "estimated_delay_days": 6.0,
        # Source: the Ever Given ran aground 2021-03-23 and was refloated/traffic
        # resumed 2021-03-29 — widely reported (Suez Canal Authority statements,
        # contemporaneous Reuters/AP coverage). Real duration was ~6 days, not
        # the dramatized 14 days used as the demo scenario's severity dial.
    },
    {
        "event_id": "CURATED-LALB-CONGESTION-2021",
        "event_type": "Port Congestion",
        "location": "Los Angeles/Long Beach, USA",
        "start_date": "2021-09-01",
        "end_date": "2022-03-01",
        "severity": "HIGH",
        "affected_route": None,
        "estimated_delay_days": 14.0,
        # Source: widely reported container-ship anchorage queues (Marine
        # Exchange of Southern California queue data), peaking ~Sept 2021.
    },
    {
        "event_id": "CURATED-US-TARIFF-2019",
        "event_type": "Tariff Increase",
        "location": "United States (imports from China)",
        "start_date": "2019-05-10",
        "end_date": None,
        "severity": "HIGH",
        "affected_route": None,
        "estimated_delay_days": None,
        # Source: corroborated by our own tariffs_worldbank_wdi dataset — US
        # TM.TAX.MRCH.WM.AR.ZS jumps to 13.78% in 2019 vs 1.59% in 2018.
    },
    {
        "event_id": "CURATED-TEXAS-WINTER-STORM-URI-2021",
        "event_type": "Severe Weather",
        "location": "Texas, USA",
        "start_date": "2021-02-13",
        "end_date": "2021-02-17",
        "severity": "CRITICAL",
        "affected_route": None,
        "estimated_delay_days": 5.0,
        # Source: Winter Storm Uri — widely reported grid/petrochemical/supplier
        # shutdowns across Texas, Feb 2021.
    },
]


def build_weather_disruptions() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(NOAA_PATH, low_memory=False)
    report: dict = {"input_rows": len(raw)}

    raw["damage_property_usd"] = raw["DAMAGE_PROPERTY"].apply(parse_money_string)
    raw["damage_crops_usd"] = raw["DAMAGE_CROPS"].apply(parse_money_string)
    raw["total_damage_usd"] = raw["damage_property_usd"] + raw["damage_crops_usd"]
    raw["total_casualties"] = raw[["INJURIES_DIRECT", "INJURIES_INDIRECT", "DEATHS_DIRECT", "DEATHS_INDIRECT"]].sum(axis=1)

    def severity_tier(row) -> str:
        if row["total_casualties"] > 0 or row["total_damage_usd"] >= 10_000_000:
            return "CRITICAL"
        if row["total_damage_usd"] >= 1_000_000:
            return "HIGH"
        if row["total_damage_usd"] >= 10_000:
            return "MEDIUM"
        return "LOW"

    raw["severity"] = raw.apply(severity_tier, axis=1)

    begin = pd.to_datetime(raw["BEGIN_DATE_TIME"], format="%d-%b-%y %H:%M:%S", errors="coerce")
    end = pd.to_datetime(raw["END_DATE_TIME"], format="%d-%b-%y %H:%M:%S", errors="coerce")
    report["unparseable_begin_dates"] = int(begin.isna().sum())
    valid = begin.notna()
    report["dropped_unparseable_dates"] = int((~valid).sum())

    end_clean = end[valid].where(end[valid].notna(), None)

    out = pd.DataFrame(
        {
            "event_id": "NOAA-" + raw.loc[valid, "EVENT_ID"].astype(str),
            "event_type": raw.loc[valid, "EVENT_TYPE"],
            "location": raw.loc[valid, "STATE"].str.title() + ", USA",
            "start_date": begin[valid],
            "end_date": end_clean,
            "severity": raw.loc[valid, "severity"],
            "affected_route": None,
            "affected_supplier": None,
            "estimated_delay_days": None,
            "status": "HISTORICAL",
        }
    )
    report["output_rows"] = len(out)
    return out.reset_index(drop=True), report


def build_curated_events() -> pd.DataFrame:
    df = pd.DataFrame(_CURATED_EVENTS)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["affected_supplier"] = None
    df["status"] = "HISTORICAL"
    return df


def build_disruptions_table() -> tuple[pd.DataFrame, dict]:
    weather, report = build_weather_disruptions()
    curated = build_curated_events()
    combined = pd.concat([weather, curated], ignore_index=True, sort=False)
    report["curated_events_added"] = len(curated)
    report["combined_rows"] = len(combined)
    return combined, report
