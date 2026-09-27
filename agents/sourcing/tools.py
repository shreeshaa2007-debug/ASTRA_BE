"""Sourcing Agent tool functions — per agent-plan.md: get_suppliers(),
check_supplier_capacity(), calculate_supplier_cost(), calculate_landed_cost(),
generate_supplier_options(). Reads the `suppliers` dataset (Phase 3/8 —
synthetic, real product_ids, see backend/services/preprocessing/suppliers.py)
and the `tariffs` dataset (Phase 3 — real World Bank data, country-year
grain) for the landed-cost tariff term, through backend/data: from the
processed CSVs, or from `ref_suppliers` / `ref_tariffs` tables.
"""
from __future__ import annotations

import pandas as pd

from backend.data import cached_dataset_loader, get_datasets

# suppliers.csv's `region` is a country NAME (matches the roster in
# suppliers.py); tariffs.csv's `origin_country` is an ISO3 code (World Bank's
# native format). This is the join key between them — kept as one explicit,
# documented table rather than a silent assumption the two would just line up.
REGION_TO_ISO3 = {
    "China": "CHN",
    "India": "IND",
    "Vietnam": "VNM",
    "Turkey": "TUR",
    "Netherlands": "NLD",
}


@cached_dataset_loader()
def _load_suppliers() -> pd.DataFrame:
    df = get_datasets().load("suppliers")
    df["product_id"] = df["product_id"].astype(str)
    return df


@cached_dataset_loader()
def _load_tariffs() -> pd.DataFrame:
    return get_datasets().load("tariffs")


@cached_dataset_loader()
def _latest_tariff_by_country() -> dict[str, float]:
    """Most recent available tariff_rate per ISO3 country — 2022 for the
    countries this roster uses (see dataset_registry.yaml: 2023+ isn't yet
    published by the World Bank)."""
    tariffs = _load_tariffs()
    idx = tariffs.groupby("origin_country")["effective_year"].idxmax()
    latest = tariffs.loc[idx].set_index("origin_country")["tariff_rate"]
    return latest.to_dict()


def get_latest_tariffs() -> dict[str, float]:
    """Most recent tariff_rate (%) per ISO3 country, for the countries in this
    project's supplier roster — the tariff baseline the shared world state
    starts a simulation from."""
    latest = _latest_tariff_by_country()
    return {iso3: float(latest[iso3]) for iso3 in REGION_TO_ISO3.values() if iso3 in latest}


def get_suppliers(product_id: str | None = None, status: str | None = None) -> list[dict]:
    df = _load_suppliers()
    if product_id is not None:
        df = df[df["product_id"] == str(product_id)]
    if status is not None:
        df = df[df["status"] == status]
    return df.to_dict("records")


def check_supplier_capacity(supplier_id: str, product_id: str, requested_quantity: int) -> dict:
    df = _load_suppliers()
    row = df[(df["supplier_id"] == supplier_id) & (df["product_id"] == str(product_id))]
    if row.empty:
        raise ValueError(f"No supplier record for supplier_id={supplier_id!r}, product_id={product_id!r}")
    capacity = int(row.iloc[0]["capacity"])
    sufficient = requested_quantity <= capacity
    return {
        "supplier_id": supplier_id,
        "product_id": str(product_id),
        "capacity": capacity,
        "requested_quantity": requested_quantity,
        "sufficient": sufficient,
        "shortfall": max(0, requested_quantity - capacity),
    }


def calculate_supplier_cost(supplier_id: str, product_id: str, quantity: int) -> float:
    df = _load_suppliers()
    row = df[(df["supplier_id"] == supplier_id) & (df["product_id"] == str(product_id))]
    if row.empty:
        raise ValueError(f"No supplier record for supplier_id={supplier_id!r}, product_id={product_id!r}")
    return round(float(row.iloc[0]["unit_cost"]) * quantity, 2)


def calculate_landed_cost(supplier_id: str, product_id: str, quantity: int, tariff_rates: dict[str, float] | None = None) -> dict:
    """Landed cost = base supplier cost x (1 + tariff_rate/100), using that
    supplier's region's most recent real World Bank tariff rate. A region
    with no tariff data on file (shouldn't happen for this roster — see
    REGION_TO_ISO3 — but checked rather than assumed) falls back to a 0%
    tariff with `tariff_data_available=False`, so a caller can tell the
    difference between "no tariff" and "no data."

    `tariff_rates` (ISO3 -> %) overrides the World Bank data: the shared world
    state carries the tariffs in effect for a simulation, so a scenario's tariff
    change reaches the price. Omitted, the real data is used.
    """
    df = _load_suppliers()
    row = df[(df["supplier_id"] == supplier_id) & (df["product_id"] == str(product_id))]
    if row.empty:
        raise ValueError(f"No supplier record for supplier_id={supplier_id!r}, product_id={product_id!r}")
    supplier = row.iloc[0]

    iso3 = REGION_TO_ISO3.get(supplier["region"])
    rates = tariff_rates if tariff_rates is not None else _latest_tariff_by_country()
    tariff_rate = rates.get(iso3) if iso3 else None
    tariff_data_available = tariff_rate is not None
    tariff_rate = tariff_rate or 0.0

    landed_unit_cost = round(float(supplier["unit_cost"]) * (1 + tariff_rate / 100), 4)
    base_cost = float(supplier["unit_cost"]) * quantity
    landed_cost = round(landed_unit_cost * quantity, 2)

    return {
        "supplier_id": supplier_id,
        "product_id": str(product_id),
        "quantity": quantity,
        "base_cost": round(base_cost, 2),
        "tariff_rate_pct": tariff_rate,
        "tariff_data_available": tariff_data_available,
        "landed_unit_cost": landed_unit_cost,
        "landed_cost": landed_cost,
        "lead_time_days": int(supplier["lead_time_days"]),
    }


def generate_supplier_options(
    product_id: str,
    required_quantity: int,
    excluded_supplier_ids: frozenset[str] = frozenset(),
    tariff_rates: dict[str, float] | None = None,
) -> list[dict]:
    """Every supplier that carries `product_id` and isn't excluded (e.g.
    DISRUPTED, or already ruled out upstream), each annotated with capacity
    check + landed cost, sorted by landed cost ascending. Mirrors
    generate_alternative_routes()'s shape in the Logistics Agent — candidates
    for the Optimization Engine to allocate across, not a final decision.
    """
    suppliers = [s for s in get_suppliers(product_id=product_id) if s["supplier_id"] not in excluded_supplier_ids and s["status"] != "DISRUPTED"]

    options = []
    for s in suppliers:
        cap_check = check_supplier_capacity(s["supplier_id"], product_id, required_quantity)
        landed = calculate_landed_cost(s["supplier_id"], product_id, required_quantity, tariff_rates)
        options.append(
            {
                "supplier_id": s["supplier_id"],
                "supplier_name": s["supplier_name"],
                "region": s["region"],
                "status": s["status"],
                "risk_level": s["risk_level"],
                "reliability": s["reliability"],
                "capacity": cap_check["capacity"],
                "capacity_sufficient": cap_check["sufficient"],
                "lead_time_days": landed["lead_time_days"],
                "unit_cost": s["unit_cost"],
                "tariff_rate_pct": landed["tariff_rate_pct"],
                "landed_unit_cost": landed["landed_unit_cost"],
                "landed_cost_for_requested_quantity": landed["landed_cost"],
            }
        )
    return sorted(options, key=lambda o: o["landed_unit_cost"])
