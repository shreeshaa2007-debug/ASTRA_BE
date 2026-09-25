"""Logistics Agent tool functions — per agent-plan.md: get_routes(),
check_route_capacity(), calculate_transport_cost(), calculate_eta(),
generate_alternative_routes(). Reads data/processed/routes.csv (Phase 3/7 —
real port coordinates, documented-heuristic cost/time, see
backend/services/preprocessing/ports_routes.py). No shared world state exists
yet (Phase 12), so "which routes are currently disrupted" is passed in as a
parameter here rather than read from live state — the same pattern the
Inventory Agent uses.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROUTES_PATH = Path("data/processed/routes.csv")


@lru_cache(maxsize=1)
def _load_routes() -> pd.DataFrame:
    return pd.read_csv(ROUTES_PATH)


def get_routes(
    origin: str | None = None,
    destination: str | None = None,
    transport_mode: str | None = None,
    disrupted_route_ids: frozenset[str] = frozenset(),
) -> list[dict]:
    """Lists routes matching the given filters. `disrupted_route_ids`
    overrides each matching route's static `status` to DISRUPTED for this
    call — it does not mutate routes.csv (this tool is read-only, per
    agent-plan.md's ground rules), it's the caller telling this snapshot
    what's currently blocked.
    """
    df = _load_routes().copy()
    if origin is not None:
        df = df[df["origin"] == origin]
    if destination is not None:
        df = df[df["destination"] == destination]
    if transport_mode is not None:
        df = df[df["transport_mode"] == transport_mode]

    records = df.to_dict("records")
    for r in records:
        if r["route_id"] in disrupted_route_ids:
            r["status"] = "DISRUPTED"
    return records


def check_route_capacity(route_id: str, requested_quantity: int) -> dict:
    df = _load_routes()
    row = df[df["route_id"] == route_id]
    if row.empty:
        raise ValueError(f"Unknown route_id {route_id!r}")
    capacity = int(row.iloc[0]["capacity"])
    sufficient = requested_quantity <= capacity
    return {
        "route_id": route_id,
        "capacity": capacity,
        "requested_quantity": requested_quantity,
        "sufficient": sufficient,
        "shortfall": max(0, requested_quantity - capacity),
    }


def calculate_transport_cost(route_id: str, quantity: int) -> float:
    df = _load_routes()
    row = df[df["route_id"] == route_id]
    if row.empty:
        raise ValueError(f"Unknown route_id {route_id!r}")
    return round(float(row.iloc[0]["cost_per_unit"]) * quantity, 2)


def calculate_eta(route_id: str, departure_date: str | pd.Timestamp) -> str:
    df = _load_routes()
    row = df[df["route_id"] == route_id]
    if row.empty:
        raise ValueError(f"Unknown route_id {route_id!r}")
    transit_days = float(row.iloc[0]["transit_time_days"])
    eta = pd.Timestamp(departure_date) + pd.Timedelta(days=transit_days)
    return str(eta.date())


def _risk_label(capacity_check: dict, status: str) -> str:
    if not capacity_check["sufficient"]:
        return "HIGH"
    if status == "NORMAL":
        return "LOW"
    return "MEDIUM"  # any working alternative (Cape/rail/air) carries more inherent
    # uncertainty than the normal lane simply for being a less-traveled substitute —
    # a stated, simple rule, not a modeled risk score


def generate_alternative_routes(
    origin: str,
    destination: str,
    quantity: int,
    disrupted_route_ids: frozenset[str] = frozenset(),
    departure_date: str | pd.Timestamp | None = None,
) -> list[dict]:
    """Every non-disrupted route for the (origin, destination) lane, each
    annotated with a capacity check, total cost, and ETA, sorted cheapest
    first. This agent surfaces costed candidates — it does not decide the
    final mix (that's the Optimization Engine, Phase 10, once inventory and
    sourcing constraints are also in play; see agent-plan.md).
    """
    if departure_date is None:
        departure_date = pd.Timestamp.today().normalize()

    routes = get_routes(origin=origin, destination=destination, disrupted_route_ids=disrupted_route_ids)
    viable = [r for r in routes if r["status"] != "DISRUPTED"]

    candidates = []
    for r in viable:
        cap_check = check_route_capacity(r["route_id"], quantity)
        candidates.append(
            {
                "route_id": r["route_id"],
                "transport_mode": r["transport_mode"],
                "status": r["status"],
                "distance_km": r["distance_km"],
                "transit_time_days": r["transit_time_days"],
                "capacity": cap_check["capacity"],
                "capacity_sufficient": cap_check["sufficient"],
                "cost_per_unit": r["cost_per_unit"],
                "total_cost": calculate_transport_cost(r["route_id"], quantity),
                "eta": calculate_eta(r["route_id"], departure_date),
                "risk": _risk_label(cap_check, r["status"]),
            }
        )

    return sorted(candidates, key=lambda c: c["total_cost"])
