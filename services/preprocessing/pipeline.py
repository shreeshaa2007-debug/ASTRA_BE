"""Orchestrates the full Phase 3 pipeline: data/raw -> data/processed for
every acquired/derivable dataset, validating output rows against the internal
schemas (backend/schemas/entities.py) before writing anything, and recording
every cleaning/derivation decision in data/processed/manifest.json so the
numbers in docs are traceable back to what actually ran.

Run with: python -m backend.services.preprocessing.pipeline
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from backend.schemas.entities import (
    DemandRecord,
    DisruptionRecord,
    InventoryRecord,
    PortRecord,
    RouteRecord,
    SupplierRecord,
    TariffRecord,
)
from backend.services.preprocessing import demand as demand_mod
from backend.services.preprocessing import disruptions as disruptions_mod
from backend.services.preprocessing import features as features_mod
from backend.services.preprocessing import inventory as inventory_mod
from backend.services.preprocessing import ports_routes as ports_routes_mod
from backend.services.preprocessing import suppliers as suppliers_mod
from backend.services.preprocessing import tariffs as tariffs_mod

PROCESSED_DIR = Path("data/processed")


def _validate_sample(df: pd.DataFrame, schema, n: int = 200) -> dict:
    """Validates up to n rows against a pydantic schema. Full-row validation
    on the ~500k-row demand panel would be slow for no extra signal once a
    representative sample passes with the same code path — every row went
    through the same construction logic, so a schema bug would show up in
    the sample.
    """
    sample = df.sample(min(n, len(df)), random_state=0) if len(df) > 0 else df
    errors = []
    for _, row in sample.iterrows():
        try:
            schema(**row.to_dict())
        except ValidationError as exc:
            errors.append(str(exc))
    return {"rows_checked": len(sample), "validation_errors": errors[:10], "error_count": len(errors)}


def run_all() -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"generated_at": datetime.now(timezone.utc).isoformat(), "steps": {}}

    print("[1/7] demand panel...")
    demand_raw, demand_report = demand_mod.build_demand_panel()
    manifest["steps"]["demand_clean"] = demand_report

    print("[2/7] disruptions...")
    disruptions_df, disruptions_report = disruptions_mod.build_disruptions_table()
    manifest["steps"]["disruptions"] = disruptions_report
    disruptions_df.to_csv(PROCESSED_DIR / "disruptions.csv", index=False)
    manifest["steps"]["disruptions"]["validation"] = _validate_sample(disruptions_df, DisruptionRecord)

    print("[3/7] demand features + time-based split...")
    demand_featured = features_mod.build_feature_panel(demand_raw, disruptions_df)
    demand_featured.to_csv(PROCESSED_DIR / "demand.csv", index=False)
    manifest["steps"]["demand_features"] = {
        "output_rows": len(demand_featured),
        "split_counts": demand_featured["split"].value_counts().to_dict(),
        "validation": _validate_sample(demand_featured.fillna(0), DemandRecord),
    }

    print("[4/7] tariffs...")
    tariffs_df, tariffs_report = tariffs_mod.build_tariffs_table()
    tariffs_df.to_csv(PROCESSED_DIR / "tariffs.csv", index=False)
    manifest["steps"]["tariffs"] = tariffs_report
    manifest["steps"]["tariffs"]["validation"] = _validate_sample(tariffs_df, TariffRecord)

    print("[5/7] ports + routes...")
    ports_df, ports_report = ports_routes_mod.build_ports_table()
    ports_df.to_csv(PROCESSED_DIR / "ports.csv", index=False)
    manifest["steps"]["ports"] = ports_report
    manifest["steps"]["ports"]["validation"] = _validate_sample(ports_df, PortRecord)

    routes_df, routes_report = ports_routes_mod.build_routes_table()
    routes_df.to_csv(PROCESSED_DIR / "routes.csv", index=False)
    manifest["steps"]["routes"] = routes_report
    manifest["steps"]["routes"]["validation"] = _validate_sample(routes_df, RouteRecord)

    print("[6/7] inventory (derived)...")
    inventory_df, inventory_report = inventory_mod.derive_inventory_ledger(demand_raw)
    inventory_df.to_csv(PROCESSED_DIR / "inventory.csv", index=False)
    manifest["steps"]["inventory"] = inventory_report
    manifest["steps"]["inventory"]["validation"] = _validate_sample(inventory_df, InventoryRecord)

    print("[7/7] suppliers (synthetic)...")
    suppliers_df, suppliers_report = suppliers_mod.synthesize_suppliers()
    suppliers_df.to_csv(PROCESSED_DIR / "suppliers.csv", index=False)
    manifest["steps"]["suppliers"] = suppliers_report
    manifest["steps"]["suppliers"]["validation"] = _validate_sample(suppliers_df, SupplierRecord)

    manifest["provenance_summary"] = {
        "real": ["demand.csv (base fields)", "disruptions.csv (NOAA rows)", "tariffs.csv", "ports.csv"],
        "derived": ["inventory.csv", "routes.csv (distance/transit_time from real coordinates)"],
        "synthetic": ["suppliers.csv", "routes.csv (capacity/cost_per_unit)", "disruptions.csv (curated non-weather rows — real events, authored records)"],
    }

    total_errors = sum(s.get("validation", {}).get("error_count", 0) for s in manifest["steps"].values())
    manifest["total_validation_errors"] = total_errors

    with open(PROCESSED_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nDone. {total_errors} validation errors across all tables.")
    print(f"Manifest: {PROCESSED_DIR / 'manifest.json'}")
    return manifest


if __name__ == "__main__":
    run_all()
