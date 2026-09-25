"""ROUTES/PORTS preprocessing — data/raw/ports_world_port_index -> internal
PORT and ROUTE schemas.

Port identity/location is real (World Port Index, NGA, 2017 snapshot — see
dataset_registry.yaml for the vintage caveat). Route *distances* are computed
from those real coordinates via a two-leg great-circle sum through a named
waypoint (Port Said for the Suez corridor, Cape Town for the Cape of Good Hope
detour) — a documented, physically-grounded proxy, not a real shipping-lane
distance/cost dataset (no such dataset was acquired — see
dataset_registry.yaml's "D. Transportation/logistics" and "E. Ports/routes"
entries). `transit_time_days` extends that proxy with a stated container-vessel
service-speed assumption. `capacity` and `cost_per_unit` have no real source at
all and are illustrative placeholders. Because of that weakest link, every
RouteRecord produced here is tagged `Provenance.SYNTHETIC` even though its
distance is real-coordinate-derived — see the module-level `PROVENANCE_NOTE`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from dbfread import DBF

from backend.services.preprocessing.utils import haversine_km

DBF_PATH = Path("data/raw/ports_world_port_index/WPI.dbf")

PROVENANCE_NOTE = (
    "SEA distance_km/transit_time_days: derived from real WPI port coordinates + a "
    "stated 18-knot service-speed assumption. RAIL distance/transit_time: the real "
    "China-Europe Railway Express service's publicly reported range (~11,000 km, "
    "~15-18 days), applied to Shanghai only — the corridor's real origin region — "
    "with Rotterdam standing in for the service's real terminus, Duisburg (~225 km "
    "further inland). AIR distance: real great-circle (correct physical model for "
    "aircraft). capacity/cost_per_unit for all modes: no public source exists (see "
    "dataset_registry.yaml); rail/air costs are sea cost x a documented multiplier "
    "(~2x, ~6x) taken from publicly reported typical ranges, not independently "
    "invented. Record-level provenance is SYNTHETIC throughout to reflect the "
    "weakest link (capacity/cost), even where distance is real-coordinate-derived."
)

# Real port coordinates, looked up from the actual WPI.dbf (INDEX_NO verified
# against the loaded table, not guessed) — see the port lookup used to build
# this table in Phase 3's development notes.
_KEY_PORTS = {
    "SHANGHAI": {"index_no": 59970, "country": "CN", "lat": 31.216667, "lon": 121.500000},
    "SINGAPORE": {"index_no": 50000, "country": "SG", "lat": 1.283333, "lon": 103.850000},
    "ROTTERDAM": {"index_no": 31140, "country": "NL", "lat": 51.900000, "lon": 4.483333},
    "PORT_SAID": {"index_no": 45140, "country": "EG", "lat": 31.266667, "lon": 32.300000},  # Suez corridor waypoint
    "CAPE_TOWN": {"index_no": 46770, "country": "ZA", "lat": -33.916667, "lon": 18.416667},  # Cape of Good Hope waypoint
    "MUMBAI": {"index_no": 48840, "country": "IN", "lat": 18.966667, "lon": 72.866667},
    "CHENNAI": {"index_no": 49450, "country": "IN", "lat": 13.100000, "lon": 80.300000},
    "SOUTHAMPTON": {"index_no": 35580, "country": "GB", "lat": 50.900000, "lon": -1.400000},
}

# (origin, destination, waypoint or None, transport_mode, route_id, status)
_SCENARIO_LANES = [
    ("SHANGHAI", "ROTTERDAM", "PORT_SAID", "sea", "SHA-ROT-SUEZ", "NORMAL"),
    ("SHANGHAI", "ROTTERDAM", "CAPE_TOWN", "sea", "SHA-ROT-CAPE", "ALTERNATIVE"),
    ("SINGAPORE", "ROTTERDAM", "PORT_SAID", "sea", "SIN-ROT-SUEZ", "NORMAL"),
    ("SINGAPORE", "ROTTERDAM", "CAPE_TOWN", "sea", "SIN-ROT-CAPE", "ALTERNATIVE"),
    ("MUMBAI", "ROTTERDAM", "PORT_SAID", "sea", "MUM-ROT-SUEZ", "NORMAL"),
    ("CHENNAI", "ROTTERDAM", "PORT_SAID", "sea", "CHE-ROT-SUEZ", "NORMAL"),
]

_CONTAINER_VESSEL_KNOTS = 18.0  # stated assumption — typical container-ship service speed
_KM_PER_NAUTICAL_MILE = 1.852

# Rail: grounded in the real China-Europe Railway Express service, not a
# computed geometric proxy — overland rail distance/transit time don't come
# from great-circle math the way sea/air routes do. Figures are the midpoints
# of publicly reported ranges (e.g. Chongqing/Shilong-Duisburg corridors:
# ~11,000 km, ~15-18 days station-to-station; rail typically costs ~1.5-3x
# sea freight for the same lane). Only Shanghai has an entry — that's the
# corridor's real endpoint region; Singapore/Mumbai/Chennai have no
# equivalent real China-Europe-rail service, so no rail row is fabricated
# for them. Destination is Rotterdam here as a proxy for the corridor's real
# terminus, Duisburg (~225 km further inland) — see PROVENANCE_NOTE.
_RAIL_LANES = [
    # (origin, destination, route_id, distance_km, transit_days_low, transit_days_high)
    ("SHANGHAI", "ROTTERDAM", "SHA-ROT-RAIL", 11000.0, 15, 18),
]
_RAIL_COST_MULTIPLIER_OVER_SEA = 2.0  # midpoint of the ~1.5-3x reported range

# Air: direct great-circle distance IS the right physical model (aircraft
# fly great circles, unlike ships which follow canals/capes) — computed, not
# hardcoded. Speed and cost multiplier are midpoints of publicly reported
# ranges (cargo aircraft cruise ~724-901 km/h; air freight typically runs
# ~4-10x sea freight cost for the same lane).
_AIR_LANES = [("SHANGHAI", "ROTTERDAM", "SHA-ROT-AIR")]
_AIR_SPEED_KMH = 800.0
_AIR_COST_MULTIPLIER_OVER_SEA = 6.0
_AIR_HANDLING_BUFFER_DAYS = 1.5  # pure flight time is ~11 hours for this lane — real air-freight
# transit quotes are 1-3 days door-to-door because of customs/ground handling on each end, not
# because planes are slow; a bare flight-time figure would understate the honest transit time


def build_ports_table() -> tuple[pd.DataFrame, dict]:
    table = DBF(DBF_PATH, load=False, encoding="latin1")
    records = []
    for rec in table:
        lat, lon = rec.get("LATITUDE"), rec.get("LONGITUDE")
        if lat is None or lon is None or (lat == 0 and lon == 0):
            continue
        records.append(
            {
                "index_no": int(rec["INDEX_NO"]),
                "port_name": (rec.get("PORT_NAME") or "").strip(),
                "country": (rec.get("COUNTRY") or "").strip(),
                "latitude": float(lat),
                "longitude": float(lon),
                "harbor_size": (rec.get("HARBORSIZE") or None),
                "harbor_type": (rec.get("HARBORTYPE") or None),
                "provenance": "real",
            }
        )
    df = pd.DataFrame.from_records(records)
    report = {
        "input_dbf_rows": len(table),
        "output_rows": len(df),
        "dropped_missing_or_zero_coords": len(table) - len(df),
    }
    return df, report


def _leg_distance_km(a: dict, b: dict) -> float:
    return haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])


def build_routes_table() -> tuple[pd.DataFrame, dict]:
    rows = []
    sea_cost_by_lane: dict[tuple[str, str], float] = {}  # (origin, dest) -> the NORMAL (Suez) sea cost_per_unit, for the rail/air multipliers below

    for origin, dest, waypoint, mode, route_id, status in _SCENARIO_LANES:
        o, d = _KEY_PORTS[origin], _KEY_PORTS[dest]
        if waypoint:
            w = _KEY_PORTS[waypoint]
            distance_km = _leg_distance_km(o, w) + _leg_distance_km(w, d)
        else:
            distance_km = _leg_distance_km(o, d)

        speed_kmh = _CONTAINER_VESSEL_KNOTS * _KM_PER_NAUTICAL_MILE
        transit_time_days = round(distance_km / speed_kmh / 24, 1)
        cost_per_unit = round(distance_km * 0.012, 2)  # illustrative $/unit placeholder
        if status == "NORMAL":
            sea_cost_by_lane[(origin, dest)] = cost_per_unit

        rows.append(
            {
                "route_id": route_id,
                "origin": origin.title().replace("_", " "),
                "destination": dest.title().replace("_", " "),
                "transport_mode": mode,
                "distance_km": round(distance_km, 1),
                "capacity": 8000,       # illustrative TEU capacity placeholder — see PROVENANCE_NOTE
                "transit_time_days": transit_time_days,
                "cost_per_unit": cost_per_unit,
                "status": status,
                "provenance": "synthetic",
            }
        )

    for origin, dest, route_id, distance_km, days_low, days_high in _RAIL_LANES:
        base_sea_cost = sea_cost_by_lane[(origin, dest)]
        rows.append(
            {
                "route_id": route_id,
                "origin": origin.title().replace("_", " "),
                "destination": dest.title().replace("_", " "),
                "transport_mode": "rail",
                "distance_km": distance_km,
                "capacity": 2000,  # a rail block train carries far fewer TEU per departure than a containership
                "transit_time_days": round((days_low + days_high) / 2, 1),
                "cost_per_unit": round(base_sea_cost * _RAIL_COST_MULTIPLIER_OVER_SEA, 2),
                "status": "ALTERNATIVE",
                "provenance": "synthetic",
            }
        )

    for origin, dest, route_id in _AIR_LANES:
        o, d = _KEY_PORTS[origin], _KEY_PORTS[dest]
        distance_km = _leg_distance_km(o, d)  # direct great circle — correct model for air, unlike sea
        base_sea_cost = sea_cost_by_lane[(origin, dest)]
        rows.append(
            {
                "route_id": route_id,
                "origin": origin.title().replace("_", " "),
                "destination": dest.title().replace("_", " "),
                "transport_mode": "air",
                "distance_km": round(distance_km, 1),
                "capacity": 200,  # air cargo capacity per flight is far smaller than sea/rail
                "transit_time_days": round(distance_km / _AIR_SPEED_KMH / 24 + _AIR_HANDLING_BUFFER_DAYS, 2),
                "cost_per_unit": round(base_sea_cost * _AIR_COST_MULTIPLIER_OVER_SEA, 2),
                "status": "ALTERNATIVE",
                "provenance": "synthetic",
            }
        )

    df = pd.DataFrame(rows)
    report = {"output_rows": len(df), "provenance_note": PROVENANCE_NOTE, "modes": sorted(df["transport_mode"].unique().tolist())}
    return df, report
