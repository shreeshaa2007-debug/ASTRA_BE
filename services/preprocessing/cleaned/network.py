"""Ports and routes.

PORTS are the World Port Index (REAL, NGA, a 2017 snapshot), all of them, with a stable integer key.

ROUTES fix two audit findings:

1. Distances. Phase 3 computed a sea lane's length as two great-circle legs through Port Said or Cape Town. A great circle from
   Shanghai to Port Said runs over Asia, so the "Suez" lanes were straight lines across land: Shanghai came out 19 km from
   Singapore, and the Cape lane was not the length of a Cape lane. Here a sea lane's distance is the sum of great-circle legs
   between the waypoints of scripts/build_lanes.py, the lanes the map draws, which backend/tests/test_map_lanes.py checks against
   the Natural Earth coastline (assumption A11). The old figure is kept in `legacy_distance_km` so the change is visible.
2. Alternatives. The Suez scenarios disrupt four lanes, but only Shanghai and Singapore had a way round: Mumbai and Chennai had
   none. Each origin now has a Suez lane and a Cape lane. The two new Cape lanes use waypoints authored below and checked against
   the same coastline by the tests.

A route's `provenance` is SYNTHETIC by the weakest-link rule Phase 3 set: capacity and cost have no source at all. The distance
column is DERIVED (recorded in the data dictionary); `distance_basis` says how, per lane.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pandas as pd
from dbfread import DBF

from backend.services.preprocessing.cleaned import PROCESSED_DIR, RAW_DIR, REAL, SYNTHETIC
from backend.services.preprocessing.cleaned import assumptions as A
from backend.services.preprocessing.cleaned.countries import ISO2_TO_ISO3

ROOT = Path(__file__).resolve().parents[4]
WPI_DBF = RAW_DIR / "ports_world_port_index" / "WPI.dbf"

# World Port Index INDEX_NO for the ports the routes use (assumption A18)
ROUTE_PORT_IDS = {"Shanghai": 59970, "Singapore": 50000, "Mumbai": 48840, "Chennai": 49450, "Rotterdam": 31140}

# Two lanes the map does not draw. Both leave the Indian coast for open water and join the trunk the existing Cape lanes take
# (INDIAN_OCEAN_TO_CAPE, from south of Madagascar round Agulhas, then up the Atlantic side of Africa). Chennai reuses the map's
# own Chennai -> south of Sri Lanka segment, so only Mumbai needs new coordinates.
MUMBAI_TO_CAPE_ENTRY = [(13.0, 68.5), (5.0, 65.0), (-3.0, 62.5)]

Lane = list[tuple[float, float]]


def lane_builder():
    """scripts/build_lanes.py loaded as a module: the one place the map's waypoints are written down."""
    spec = importlib.util.spec_from_file_location("build_lanes_for_cleaned_layer", ROOT / "scripts" / "build_lanes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def waypoints_and_chokepoints() -> tuple[dict[str, Lane], dict[str, list[str]]]:
    """Every sea lane's waypoints and the chokepoints it passes: the map's, plus the two Cape lanes it lacks."""
    b = lane_builder()
    tail = [*b.INDIAN_OCEAN_TO_CAPE[2:], *b.CAPE_TO_IBERIA, *b.ATLANTIC_TO_ROTTERDAM]  # from (-10, 62) on: shared with the existing lanes
    waypoints = {rid: list(w) for rid, w in b.WAYPOINTS.items() if rid.endswith(("-SUEZ", "-CAPE"))}
    waypoints["MUM-ROT-CAPE"] = [b.PORTS["Mumbai"], *MUMBAI_TO_CAPE_ENTRY, *tail, b.PORTS["Rotterdam"]]
    waypoints["CHE-ROT-CAPE"] = [b.PORTS["Chennai"], *b.CHENNAI_TO_SRI_LANKA, *(p for segment in b.CAPE_TAIL for p in segment), b.PORTS["Rotterdam"]]
    chokepoints = {rid: list(b.VIA[rid]) for rid in b.WAYPOINTS if rid in waypoints}
    chokepoints["MUM-ROT-CAPE"] = ["Cape of Good Hope"]
    chokepoints["CHE-ROT-CAPE"] = ["Cape of Good Hope"]
    return waypoints, chokepoints


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    r = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def polyline_km(waypoints: Lane) -> float:
    return sum(haversine_km(p, q) for p, q in zip(waypoints, waypoints[1:]))


def build_port_table() -> tuple[pd.DataFrame, dict]:
    table = DBF(WPI_DBF, load=False, encoding="latin1")
    rows, skipped = [], 0
    for rec in table:
        lat, lon = rec.get("LATITUDE"), rec.get("LONGITUDE")
        if lat is None or lon is None or (lat == 0 and lon == 0):
            skipped += 1
            continue
        country = (rec.get("COUNTRY") or "").strip() or None
        rows.append(
            {
                "port_id": int(rec["INDEX_NO"]), "port_name": (rec.get("PORT_NAME") or "").strip(), "country_iso2": country,
                "country_iso3": ISO2_TO_ISO3.get(country), "latitude": float(lat), "longitude": float(lon),
                "harbor_size": (rec.get("HARBORSIZE") or "").strip() or None, "harbor_type": (rec.get("HARBORTYPE") or "").strip() or None,
                "source": "NGA World Port Index (HDX mirror, 2017-03-28)", "provenance": REAL,
            }
        )
    ports = pd.DataFrame(rows)
    report = {
        "input_rows": len(table), "rows": len(ports), "skipped_missing_or_zero_coordinates": skipped,
        "with_iso3": int(ports["country_iso3"].notna().sum()), "without_iso3": int(ports["country_iso3"].isna().sum()),
        "duplicate_name_and_country": int(ports.duplicated(["port_name", "country_iso2"]).sum()),
    }
    return ports, report


def build_route_table() -> tuple[pd.DataFrame, dict]:
    b = lane_builder()
    waypoints, chokepoints = waypoints_and_chokepoints()
    legacy = pd.read_csv(PROCESSED_DIR / "routes.csv").set_index("route_id")["distance_km"].to_dict()  # read for comparison only

    knots_to_kmh = A.VESSEL_SPEED_KNOTS * A.KM_PER_NAUTICAL_MILE
    order = [  # old lanes in their old order, each new Cape lane beside its Suez sibling
        "SHA-ROT-SUEZ", "SHA-ROT-CAPE", "SIN-ROT-SUEZ", "SIN-ROT-CAPE", "MUM-ROT-SUEZ", "MUM-ROT-CAPE", "CHE-ROT-SUEZ", "CHE-ROT-CAPE",
    ]
    origin_name = {"SHA": "Shanghai", "SIN": "Singapore", "MUM": "Mumbai", "CHE": "Chennai"}
    rows: list[dict] = []
    suez_cost: dict[str, float] = {}
    for rid in order:
        km = polyline_km(waypoints[rid])
        via = rid.split("-")[-1]
        cost = round(km * A.COST_PER_KM, 2)
        if via == "SUEZ":
            suez_cost[rid[:3]] = cost
        rows.append(
            {
                "route_id": rid, "origin": origin_name[rid[:3]], "destination": "Rotterdam", "transport_mode": "sea",
                "lane_role": "PRIMARY" if via == "SUEZ" else "ALTERNATIVE", "via": via, "chokepoints": "|".join(chokepoints[rid]),
                "distance_km": round(km, 1), "distance_basis": "WAYPOINT_POLYLINE", "legacy_distance_km": legacy.get(rid),
                "capacity": A.CAPACITY_BY_MODE["sea"], "transit_time_days": round(km / knots_to_kmh / 24, 1), "cost_per_unit": cost,
                "route_note": ("Added: Cape alternative for the Suez and Red Sea scenarios; waypoints authored in cleaned/network.py (A11)"
                               if rid in ("MUM-ROT-CAPE", "CHE-ROT-CAPE") else "Distance re-derived along the map's sea-lane waypoints (A11)"),
            }
        )

    rail_days = sum(A.RAIL_TRANSIT_DAYS) / 2
    rows.append(
        {
            "route_id": "SHA-ROT-RAIL", "origin": "Shanghai", "destination": "Rotterdam", "transport_mode": "rail", "lane_role": "ALTERNATIVE",
            "via": "RAIL", "chokepoints": "", "distance_km": A.RAIL_DISTANCE_KM, "distance_basis": "REPORTED_SERVICE_RANGE",
            "legacy_distance_km": legacy.get("SHA-ROT-RAIL"), "capacity": A.CAPACITY_BY_MODE["rail"], "transit_time_days": round(rail_days, 1),
            "cost_per_unit": round(suez_cost["SHA"] * A.RAIL_COST_MULTIPLIER, 2),
            "route_note": "China-Europe Railway Express range as reported publicly, applied to Shanghai; Rotterdam stands in for the real terminus Duisburg (A12)",
        }
    )
    air_km = haversine_km(b.PORTS["Shanghai"], b.PORTS["Rotterdam"])
    rows.append(
        {
            "route_id": "SHA-ROT-AIR", "origin": "Shanghai", "destination": "Rotterdam", "transport_mode": "air", "lane_role": "ALTERNATIVE",
            "via": "AIR", "chokepoints": "", "distance_km": round(air_km, 1), "distance_basis": "GREAT_CIRCLE",
            "legacy_distance_km": legacy.get("SHA-ROT-AIR"), "capacity": A.CAPACITY_BY_MODE["air"],
            "transit_time_days": round(air_km / A.AIR_SPEED_KMH / 24 + A.AIR_HANDLING_DAYS, 2),
            "cost_per_unit": round(suez_cost["SHA"] * A.AIR_COST_MULTIPLIER, 2),
            "route_note": "A great circle is the correct model for an aircraft, unlike a ship (A12)",
        }
    )

    routes = pd.DataFrame(rows)
    routes.insert(3, "origin_port_id", routes["origin"].map(ROUTE_PORT_IDS))
    routes.insert(4, "destination_port_id", routes["destination"].map(ROUTE_PORT_IDS))
    routes["provenance"] = SYNTHETIC
    sea = routes[routes["transport_mode"] == "sea"].set_index("route_id")
    report = {
        "rows": len(routes),
        "added_lanes": ["MUM-ROT-CAPE", "CHE-ROT-CAPE"],
        "distance_change_km": {rid: {"legacy": float(legacy[rid]), "now": float(sea.loc[rid, "distance_km"])} for rid in sea.index if rid in legacy},
        "cape_over_suez": {p: round(float(sea.loc[f"{p}-ROT-CAPE", "distance_km"] / sea.loc[f"{p}-ROT-SUEZ", "distance_km"]), 3) for p in origin_name},
    }
    return routes, report


ROUTE_COLUMN_PROVENANCE = {
    "distance_km": "DERIVED", "transit_time_days": "DERIVED", "legacy_distance_km": "DERIVED", "chokepoints": "DERIVED", "via": "DERIVED",
    "lane_role": "SYNTHETIC", "origin_port_id": "AUTHORED", "destination_port_id": "AUTHORED",
}
