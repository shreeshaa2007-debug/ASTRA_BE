"""Suppliers and scenarios.

SUPPLIERS are SYNTHETIC and say so in their names: no public dataset carries supplier capacity, cost, lead time or reliability, and
Open Supply Hub needed a token (data/dataset_registry.yaml). The Phase 3 roster used names that read like real companies
("Chengdu Precision Components"); these are "SYNTHETIC Supplier S001". Nothing here describes a real firm.

The master tables hold NOMINAL properties only:

    supplier            one row per supplier (supplier_id is the key)
    supplier_product    one row per (supplier, product) (that PAIR is the key: supplier_id alone repeats)

Phase 3 stored a supplier's disruption in the master (S001 DISRUPTED with capacity 0, S002 REDUCED, and reliability drawn low
BECAUSE they were disrupted), so the "normal" baseline was not normal. Here a disruption is scenario data:

    scenario             what a scenario is
    scenario_state       what it does to which route or supplier (only deviations from normal are written)
    scenario_param       its numbers (duration, tariff change)

The normal baseline has no scenario_state rows, which means every supplier is ACTIVE and every lane NORMAL. The old demo start
(S001 disrupted, S002 reduced) is kept as the scenario DEMO_LEGACY_START, so nothing that could be shown before is lost. The six
scenarios of backend/config/scenarios.yaml are read from that file, not retyped, and a scenario's effect on its routes and
suppliers comes from the state layer's own rule (EVENT_EFFECTS), so a scenario changes the world here exactly as it does live.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from backend.services.preprocessing.cleaned import SYNTHETIC
from backend.services.preprocessing.cleaned import assumptions as A
from backend.services.preprocessing.cleaned.network import ROUTE_PORT_IDS
from backend.services.world_state.events import EVENT_EFFECTS

ROOT = Path(__file__).resolve().parents[4]
SCENARIOS_YAML = ROOT / "backend" / "config" / "scenarios.yaml"
OPTIMIZATION_YAML = ROOT / "backend" / "config" / "optimization_config.yaml"

# (supplier_id, city, country, ISO-3, risk tier): the Phase 3 roster's geography and risk tiers, without its company-like names
ROSTER = [
    ("S001", "Chengdu", "China", "CHN", "HIGH"),
    ("S002", "Shenzhen", "China", "CHN", "HIGH"),
    ("S003", "Pune", "India", "IND", "LOW"),
    ("S004", "Chennai", "India", "IND", "LOW"),
    ("S005", "Hanoi", "Vietnam", "VNM", "MEDIUM"),
    ("S006", "Ho Chi Minh City", "Vietnam", "VNM", "MEDIUM"),
    ("S007", "Istanbul", "Turkey", "TUR", "MEDIUM"),
    ("S008", "Rotterdam", "Netherlands", "NLD", "LOW"),
]

# the scenario the Suez blockage of 2021 grounds; the others are hypothetical
BASED_ON_EVENT = {"SUEZ_CLOSURE": "CURATED-SUEZ-2021"}


def build_supplier_table() -> pd.DataFrame:
    """One row per supplier. `origin_port_id` (which port's lanes carry its goods) is read from the optimizer's configuration, so
    the mapping is data here and stays the same one the optimizer uses."""
    origin = yaml.safe_load(OPTIMIZATION_YAML.read_text(encoding="utf-8"))["network"]["supplier_origin_port"]
    assert set(origin) == {r[0] for r in ROSTER}, "the roster and optimization_config.yaml must list the same suppliers"
    rows = [
        {
            "supplier_id": sid, "supplier_name": f"SYNTHETIC Supplier {sid}", "city": city, "country_iso3": iso3, "region": country,
            "risk_level": risk, "origin_port_id": ROUTE_PORT_IDS.get(origin[sid]) if origin[sid] else None, "provenance": SYNTHETIC,
        }
        for sid, city, country, iso3, risk in ROSTER
    ]
    return pd.DataFrame(rows).astype({"origin_port_id": "Int64"})


def pick_supplier_products(panel_series: pd.DataFrame) -> list[str]:
    """The highest-volume products in the modeling panel (A15), so suppliers, forecasts and stock refer to the same products."""
    totals = panel_series.groupby("product_id")["demand_quantity"].sum().sort_values(ascending=False, kind="stable")
    return totals.head(A.SUPPLIER_PRODUCT_COUNT).index.tolist()


def build_supplier_product_table(products: list[str], seed: int = A.SUPPLIER_SEED) -> pd.DataFrame:
    """(supplier, product) pairs with NOMINAL capacity, cost, lead time and reliability. Each pair draws from its own random stream,
    so the numbers of one pair do not move when another pair or a product is added. A supplier always carries at least one product."""
    draws = []
    for sid, _city, country, _iso3, risk in ROSTER:
        for product_id in products:
            rng = np.random.default_rng([seed, zlib.crc32(f"{sid}|{product_id}".encode())])
            carry = rng.random()
            capacity = int(A.BASE_CAPACITY_BY_COUNTRY[country] * rng.uniform(*A.CAPACITY_RANGE))
            unit_cost = round(float(rng.uniform(*A.UNIT_COST_RANGE)), 2)
            lead_time = int(rng.integers(A.LEAD_TIME_RANGE_DAYS[0], A.LEAD_TIME_RANGE_DAYS[1] + 1))
            reliability = round(float(rng.uniform(*A.RELIABILITY_BAND_BY_RISK[risk])), 2)
            draws.append(
                {"supplier_id": sid, "product_id": product_id, "capacity": capacity, "unit_cost": unit_cost, "lead_time_days": lead_time,
                 "reliability": reliability, "carry": carry}
            )
    frame = pd.DataFrame(draws)
    include = frame["carry"] < A.PRODUCT_CARRY_PROBABILITY
    for sid in frame["supplier_id"].unique():
        mine = frame["supplier_id"] == sid
        if not include[mine].any():
            include[frame.loc[mine, "carry"].idxmin()] = True
    out = frame[include].drop(columns="carry").reset_index(drop=True)
    out["provenance"] = SYNTHETIC
    return out


# --------------------------------------------------------------------------- scenarios
def _factor(status: str) -> float | None:
    return {"DISRUPTED": A.DISRUPTED_CAPACITY_FACTOR, "REDUCED": A.REDUCED_CAPACITY_FACTOR}.get(status)


def _state_row(scenario_id: str, entity_type: str, entity_id: str, status: str, basis: str) -> dict:
    return {"scenario_id": scenario_id, "entity_type": entity_type, "entity_id": entity_id, "status": status,
            "capacity_factor": _factor(status), "basis": basis, "provenance": SYNTHETIC}


def build_scenarios(routes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(scenario, scenario_state, scenario_param). `routes` is the cleaned route table, needed to find the lanes a chokepoint or a port closes."""
    config = yaml.safe_load(SCENARIOS_YAML.read_text(encoding="utf-8"))["scenarios"]
    def lanes_through(chokepoint: str) -> list[str]:
        return routes.loc[routes["chokepoints"].str.split("|").apply(lambda passes: chokepoint in passes), "route_id"].tolist()

    def lanes_from(origin: str) -> list[str]:
        return routes.loc[routes["origin"] == origin, "route_id"].tolist()

    scenarios, states, params = [], [], []

    def add_scenario(sid, name, description, kind, modeled, source):
        scenarios.append({"scenario_id": sid, "scenario_name": name, "description": " ".join(str(description).split()), "scenario_type": kind,
                          "is_modeled": bool(modeled), "based_on_event_id": BASED_ON_EVENT.get(sid), "source": source, "provenance": SYNTHETIC})

    def add_param(sid, key, value, unit, basis, qualifier="ALL", text=None):
        params.append({"scenario_id": sid, "param_key": key, "qualifier": qualifier, "param_value": value, "param_text": text, "unit": unit,
                       "basis": basis, "provenance": SYNTHETIC})

    add_scenario("BASELINE_NORMAL", "Normal supply chain", "Every supplier ACTIVE and every lane NORMAL. It has no scenario_state rows because "
                 "it changes nothing.", "BASELINE", True, "cleaned layer")

    for sid, spec in config.items():
        add_scenario(sid, spec["label"], spec["description"], "DISRUPTION", spec["modeled"], "backend/config/scenarios.yaml")
        event = spec.get("event")
        if not event:
            continue  # a scenario the pipeline records but cannot simulate keeps its row and has no effect to write
        route_effect, supplier_effect = EVENT_EFFECTS[event["event_type"]]
        routes_hit = list(event["affected_routes"])
        note = "from backend/config/scenarios.yaml through the state layer's EVENT_EFFECTS"
        if event["event_type"] == "severe_weather":
            # a closed port closes every lane that leaves it, including the Cape lane this layer adds
            origins = {routes.loc[routes["route_id"] == r, "origin"].iat[0] for r in routes_hit}
            routes_hit = sorted({r for o in origins for r in lanes_from(o)})
            note += "; a port closure also closes the port's other lanes"
        if route_effect is not None:
            states += [_state_row(sid, "ROUTE", r, route_effect.value, note) for r in routes_hit]
        if supplier_effect is not None:
            states += [_state_row(sid, "SUPPLIER", s, supplier_effect.value, note) for s in event["affected_suppliers"]]
        add_param(sid, "duration_days", float(event["estimated_duration"]), "days", "backend/config/scenarios.yaml estimated_duration")
        add_param(sid, "severity", None, "", "backend/config/scenarios.yaml", text=event["severity"])
        for iso3, points in (spec.get("tariff_add_pct") or {}).items():
            add_param(sid, "tariff_add_pct", float(points), "percentage points", "added to the World Bank baseline rate", qualifier=iso3)

    add_scenario("RED_SEA_DIVERSION", "Red Sea diversion", "Shipping avoids the Red Sea: every lane through Bab-el-Mandeb is unavailable and the "
                 "Cape lanes are the way round. A prolonged stress test, not a claim about any real event.", "DISRUPTION", True, "cleaned layer")
    states += [_state_row("RED_SEA_DIVERSION", "ROUTE", r, "DISRUPTED", "every lane whose chokepoints include Bab-el-Mandeb (A17)")
               for r in lanes_through("Bab-el-Mandeb")]
    add_param("RED_SEA_DIVERSION", "duration_days", A.RED_SEA_DURATION_DAYS, "days", "illustrative duration (A17)")

    add_scenario("DEMO_LEGACY_START", "Legacy demo start (S001 disrupted, S002 reduced)", "The state the Phase 3 supplier file baked into master "
                 "data, kept as a scenario so it can still be shown.", "LEGACY_STATE", True, "data/processed/suppliers.csv (Phase 3)")
    states += [_state_row("DEMO_LEGACY_START", "SUPPLIER", "S001", "DISRUPTED", "as in Phase 3 suppliers.csv"),
               _state_row("DEMO_LEGACY_START", "SUPPLIER", "S002", "REDUCED", "as in Phase 3 suppliers.csv")]

    return pd.DataFrame(scenarios), pd.DataFrame(states), pd.DataFrame(params)
