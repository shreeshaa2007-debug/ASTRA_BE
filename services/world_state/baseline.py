"""The starting point every simulation is created from: route and supplier
status and the tariffs in effect, read through the agents' own tool functions
(one source of truth per fact, same rule as the Compliance Agent).

Inventory, demand forecasts and shipments are deliberately not in the
baseline: the agents produce those and the orchestrator commits them at the
`agents_assessed` checkpoint.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from backend.schemas.entities import RouteStatus, SupplierStatus

_SEVERITY = {SupplierStatus.ACTIVE: 0, SupplierStatus.REDUCED: 1, SupplierStatus.DISRUPTED: 2}


class Baseline(BaseModel):
    route_status: dict[str, RouteStatus] = Field(default_factory=dict)
    supplier_status: dict[str, SupplierStatus] = Field(default_factory=dict)
    tariffs: dict[str, float] = Field(default_factory=dict)  # ISO3 -> tariff rate %


def load_baseline() -> Baseline:
    from backend.agents.logistics import tools as logistics_tools
    from backend.agents.sourcing import tools as sourcing_tools

    route_status = {r["route_id"]: RouteStatus(r["status"]) for r in logistics_tools.get_routes()}

    # suppliers.csv has one row per (supplier, product); a supplier's status is
    # the same on every row today, but take the worst one rather than assume that
    supplier_status: dict[str, SupplierStatus] = {}
    for row in sourcing_tools.get_suppliers():
        status = SupplierStatus(row["status"])
        current = supplier_status.get(row["supplier_id"])
        if current is None or _SEVERITY[status] > _SEVERITY[current]:
            supplier_status[row["supplier_id"]] = status

    return Baseline(route_status=route_status, supplier_status=supplier_status, tariffs=sourcing_tools.get_latest_tariffs())
