"""The map's shipping lanes (frontend/src/data/lanes.json, from scripts/build_lanes.py).

A sea lane that crosses a continent is the kind of thing nobody notices until it is on a projector: the Cape route used to be an
arc that clipped West Africa. So the drawn curves are checked against the *same* land polygons the map draws (Natural Earth 110m,
public/land110m.json). Two things make that check honest rather than lenient:

* points are sampled along what the browser draws — straight lines in Web Mercator between the stored points — not just at the
  stored points;
* the only places allowed to touch "land" are the ones the 110m data cannot resolve (a 1 px strait, a canal, a port on a river),
  and each is listed by name below, so widening the exemption is a visible decision.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LANES_JSON = ROOT / "frontend" / "src" / "data" / "lanes.json"
LAND_JSON = ROOT / "frontend" / "public" / "land110m.json"
ROUTES_CSV = ROOT / "data" / "processed" / "routes.csv"
BORDERS_JSON = ROOT / "frontend" / "public" / "borders110m.json"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_lanes", ROOT / "scripts" / "build_lanes.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_lanes"] = module
    spec.loader.exec_module(module)
    return module


builder = _load_builder()
DOC = json.loads(LANES_JSON.read_text(encoding="utf-8"))
LANES = DOC["lanes"]

# (south, west, north, east): places where 110m land is too coarse to show the water a ship really uses
EXEMPT_BOXES = {
    "Suez Canal (a canal is not in the 110m coastline)": (29.8, 32.2, 31.4, 32.7),
    "Bab-el-Mandeb (30 km wide)": (12.2, 42.9, 13.1, 43.9),
    "Strait of Gibraltar (14 km wide)": (35.6, -6.4, 36.3, -5.0),
    "Dover strait (33 km wide)": (50.5, 0.6, 51.4, 2.3),
    "Singapore Strait and the south end of Malacca": (0.9, 102.0, 1.8, 104.6),
}
ENDPOINT_RADIUS_DEG = 1.4  # a port sits on the coast; at 110m the coast can be a little further out than the quay


# --------------------------------------------------------------------------- land
def _rings():
    rings = []
    for feature in json.loads(LAND_JSON.read_text(encoding="utf-8"))["features"]:
        geometry = feature["geometry"]
        polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        for polygon in polygons:
            rings.append((polygon[0], polygon[1:]))  # exterior, holes (lakes)
    return rings


RINGS = _rings()


def _inside(ring, lng, lat) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat) and lng < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def on_land(lat: float, lng: float) -> bool:
    for exterior, holes in RINGS:
        if _inside(exterior, lng, lat) and not any(_inside(h, lng, lat) for h in holes):
            return True
    return False


# --------------------------------------------------------------------------- what the browser draws
def _mercator_y(lat: float) -> float:
    return math.degrees(math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))


def _lat_from_mercator(y: float) -> float:
    return math.degrees(2 * math.atan(math.exp(math.radians(y))) - math.pi / 2)


def drawn_points(path, step_deg=0.15):
    """Every stored point, plus points along the straight (in Mercator) line the browser draws between neighbours."""
    for (a1, o1), (a2, o2) in zip(path, path[1:]):
        y1, y2 = _mercator_y(a1), _mercator_y(a2)
        n = max(1, int(max(abs(o2 - o1), abs(y2 - y1)) / step_deg))
        for k in range(n):
            f = k / n
            yield _lat_from_mercator(y1 + (y2 - y1) * f), o1 + (o2 - o1) * f
    yield tuple(path[-1])


def exempt(lat, lng, endpoints) -> bool:
    if any(s <= lat <= n and w <= lng <= e for s, w, n, e in EXEMPT_BOXES.values()):
        return True
    return any(math.hypot(lat - a, (lng - b) * math.cos(math.radians(a))) < ENDPOINT_RADIUS_DEG for a, b in endpoints)


# --------------------------------------------------------------------------- tests
def sea_lanes():
    return {route_id: lane for route_id, lane in LANES.items() if route_id.endswith(("-SUEZ", "-CAPE"))}


def test_every_route_the_network_has_is_drawn_and_nothing_else():
    with open(ROUTES_CSV, encoding="utf-8") as f:
        route_ids = {row["route_id"] for row in csv.DictReader(f)}
    assert set(LANES) == route_ids


def test_lanes_start_and_end_on_the_ports_the_markers_are_drawn_on():
    with open(ROUTES_CSV, encoding="utf-8") as f:
        routes = {row["route_id"]: row for row in csv.DictReader(f)}
    for route_id, lane in LANES.items():
        origin, destination = builder.PORTS[routes[route_id]["origin"]], builder.PORTS[routes[route_id]["destination"]]
        assert tuple(lane["path"][0]) == pytest.approx(origin, abs=1e-3), route_id
        assert tuple(lane["path"][-1]) == pytest.approx(destination, abs=1e-3), route_id


@pytest.mark.parametrize("route_id", sorted(sea_lanes()))
def test_a_sea_lane_never_crosses_land_outside_the_straits_the_coastline_cannot_resolve(route_id):
    lane = LANES[route_id]["path"]
    endpoints = [tuple(lane[0]), tuple(lane[-1])]
    on_land_at = [(round(lat, 2), round(lng, 2)) for lat, lng in drawn_points(lane) if not exempt(lat, lng, endpoints) and on_land(lat, lng)]
    assert not on_land_at, f"{route_id} is drawn over land at {on_land_at[:6]} ({len(on_land_at)} sampled points)"


def test_the_land_check_can_actually_fail():
    """A checker that passes everything proves nothing: the straight line Singapore -> Rotterdam crosses Asia."""
    straight = [builder.PORTS["Singapore"], builder.PORTS["Rotterdam"]]
    endpoints = [straight[0], straight[1]]
    assert any(not exempt(lat, lng, endpoints) and on_land(lat, lng) for lat, lng in drawn_points(straight))
    assert on_land(48.0, 2.0) and not on_land(0.0, -30.0)  # France; the mid-Atlantic


def test_the_exemptions_are_small():
    """If an exemption grows to swallow a real crossing, this is where it shows: each box is a strait or a canal, not a sea."""
    for name, (s, w, n, e) in EXEMPT_BOXES.items():
        assert (n - s) <= 2.0 and (e - w) <= 2.6, name


def test_suez_lanes_go_through_the_canal_and_the_cape_lanes_round_the_cape():
    suez = builder.SUEZ
    for route_id in ("SHA-ROT-SUEZ", "SIN-ROT-SUEZ", "MUM-ROT-SUEZ", "CHE-ROT-SUEZ"):
        lane = LANES[route_id]
        assert "Suez Canal" in lane["via"] and "Strait of Gibraltar" in lane["via"] and "Bab-el-Mandeb" in lane["via"]
        assert math.dist(lane["path"][lane["blocked_at"]], suez) < 0.3, f"{route_id}: the cut point is not the canal"
    for route_id in ("SHA-ROT-CAPE", "SIN-ROT-CAPE"):
        lane = LANES[route_id]
        assert "Cape of Good Hope" in lane["via"] and "blocked_at" not in lane
        closest = min(math.dist(p, builder.CAPE) for p in lane["path"])
        assert closest < 1.5, f"{route_id} passes {closest:.1f} degrees from the Cape"
        assert not any(math.dist(p, suez) < 8 for p in lane["path"][10:-10]), f"{route_id} goes near Suez"


def test_the_lanes_agree_with_the_models_own_ordering_the_cape_is_the_long_way_round():
    """routes.csv says the Cape route is longer than the Suez one for the same ports; the drawing must not say otherwise."""
    def km(path):
        return sum(math.hypot((b - d) * 111 * math.cos(math.radians((a + c) / 2)), (a - c) * 111) for (a, b), (c, d) in zip(path, path[1:]))

    with open(ROUTES_CSV, encoding="utf-8") as f:
        model = {row["route_id"]: float(row["distance_km"]) for row in csv.DictReader(f)}
    for port in ("SHA", "SIN"):
        assert km(LANES[f"{port}-ROT-CAPE"]["path"]) > km(LANES[f"{port}-ROT-SUEZ"]["path"])
        assert model[f"{port}-ROT-CAPE"] > model[f"{port}-ROT-SUEZ"]


def test_chokepoints_are_where_the_lanes_that_name_them_go():
    for route_id, lane in LANES.items():
        for name in lane["via"]:
            at = DOC["chokepoints"][name]["at"]
            assert min(math.dist(p, at) for p in lane["path"]) < 1.6, f"{route_id} says it passes {name} but never gets near it"


def test_the_checked_in_lanes_are_what_the_generator_produces():
    assert LANES_JSON.read_text(encoding="utf-8") == builder.render(), "regenerate: python scripts/build_lanes.py"


def test_the_bundled_basemap_layers_exist_and_are_small():
    """Offline by design (no tile service, no key): the land and the borders ship with the app."""
    assert LAND_JSON.stat().st_size < 200_000 and BORDERS_JSON.stat().st_size < 100_000
    borders = json.loads(BORDERS_JSON.read_text(encoding="utf-8"))
    assert borders["geometry"]["type"] == "MultiLineString" and len(borders["geometry"]["coordinates"]) > 100
