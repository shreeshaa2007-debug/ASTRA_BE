"""LogisticsAgent — composes tools.py into a proposed logistics plan for one
shipment: the normal route (if any), and costed/ranked alternatives when it's
disrupted. Read-only, per agent-plan.md — never modifies shipment records.
"""
from __future__ import annotations

import pandas as pd

from backend.agents.logistics import tools


class LogisticsAgent:
    def plan_shipment(
        self,
        origin: str,
        destination: str,
        quantity: int,
        disrupted_route_ids: frozenset[str] = frozenset(),
        departure_date: str | pd.Timestamp | None = None,
    ) -> dict:
        if departure_date is None:
            departure_date = pd.Timestamp.today().normalize()

        # The lane's designated baseline route, looked up WITHOUT the
        # disruption override — this is "what Shanghai->Rotterdam normally
        # costs/takes," used as the comparison point for every alternative's
        # additional_cost/additional_delay_days, whether or not it's
        # currently disrupted. Looking this up WITH the override applied
        # (the original version of this code) meant a disrupted normal route
        # simply vanished from the "which route has status NORMAL" search,
        # silently dropping the additional_cost/additional_delay_days fields
        # from every alternative exactly when they matter most — caught by
        # testing the disrupted-Suez case, not by inspection.
        undisrupted_routes = tools.get_routes(origin=origin, destination=destination)
        if not undisrupted_routes:
            raise ValueError(f"No routes on file for {origin} -> {destination}")
        normal = next((r for r in undisrupted_routes if r["status"] == "NORMAL"), None)
        normal_disrupted = normal is not None and normal["route_id"] in disrupted_route_ids

        alternatives = tools.generate_alternative_routes(origin, destination, quantity, disrupted_route_ids, departure_date)

        original_route_summary = None
        if normal is not None:
            original_route_summary = {
                "route_id": normal["route_id"],
                "transport_mode": normal["transport_mode"],
                "status": "DISRUPTED" if normal_disrupted else "NORMAL",
                "total_cost": tools.calculate_transport_cost(normal["route_id"], quantity),
                "eta": tools.calculate_eta(normal["route_id"], departure_date),
                "transit_time_days": normal["transit_time_days"],
            }

        # "shipment-level impact" (brief §9) — each alternative's delta vs the
        # normal route's baseline cost/time, whether or not the normal route
        # is itself currently disrupted (if it's not, this just shows what
        # you'd give up by choosing that alternative anyway).
        if original_route_summary is not None:
            for alt in alternatives:
                alt["additional_cost"] = round(alt["total_cost"] - original_route_summary["total_cost"], 2)
                alt["additional_delay_days"] = round(alt["transit_time_days"] - original_route_summary["transit_time_days"], 2)

        # Cheapest route that can actually carry the full quantity — every
        # entry in `alternatives` is already non-disrupted by construction
        # (generate_alternative_routes filters those out), so the check that
        # actually matters here is capacity, not status.
        recommended = next((a for a in alternatives if a["capacity_sufficient"]), None)

        return {
            "origin": origin,
            "destination": destination,
            "quantity": quantity,
            "departure_date": str(pd.Timestamp(departure_date).date()),
            "disrupted_route_ids": sorted(disrupted_route_ids),
            "original_route": original_route_summary,
            "alternative_routes": alternatives,
            "recommended_route_id": recommended["route_id"] if recommended else None,
        }
