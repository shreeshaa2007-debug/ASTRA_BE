"""Builds the cleaned data layer: data/raw -> data/cleaned.

    python -m backend.services.preprocessing.cleaned.build

Run from the repository root. It reads data/raw (never writes it), reads data/processed only to compare, writes data/cleaned and
the audit tables of the cancellation matching to data/interim/cleaning, and finishes by proving three things: every integrity
check passed, the raw files are byte-for-byte what they were, and data/processed is untouched. If any of them fails the run fails.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backend.services.preprocessing.cleaned import CLEANED_DIR, INTERIM_DIR, PROCESSED_DIR, RAW_DIR, REAL
from backend.services.preprocessing.cleaned import (
    assumptions as A,
    cancellations,
    checks,
    countries,
    demand,
    inventory,
    layer_io,
    network,
    reference,
    report,
    suppliers,
)
from backend.services.preprocessing.cleaned.tables import LOAD_ORDER, SPECS

RETAIL_XLSX = RAW_DIR / "demand_uci_online_retail" / "Online_Retail.xlsx"


def hash_tree(*roots: Path) -> dict[str, str]:
    """SHA-256 of every file under the roots (placeholder .gitkeep files aside): the proof that a directory was not touched."""
    out: dict[str, str] = {}
    for root in roots:
        for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != ".gitkeep"):
            out[path.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def build_tables() -> tuple[dict[str, pd.DataFrame], dict[str, dict], dict[str, pd.DataFrame]]:
    """(tables, step reports, audit tables). Nothing is written here."""
    raw = pd.read_excel(RETAIL_XLSX, engine="openpyxl", dtype={"InvoiceNo": str, "StockCode": str})
    lines, pairs, outcomes, cancel_report = cancellations.build_sale_lines(raw)
    lines = demand.attach_country(lines)

    country = countries.build_country_table()
    weather, weather_report = reference.build_weather_event_table()
    ports, port_report = network.build_port_table()
    routes, route_report = network.build_route_table()
    tariff, tariff_latest, tariff_report = reference.build_tariff_tables(country)
    events, impact, event_report = reference.build_authored_event_tables(tariff, routes)

    daily = demand.build_demand_daily(lines)
    flag_events = pd.concat(
        [
            weather.rename(columns={"start_ts": "start_date", "end_ts": "end_date"})[["start_date", "end_date", "location"]],
            events.rename(columns={"start_ts": "start_date", "end_ts": "end_date"})[["start_date", "end_date", "location"]],
        ],
        ignore_index=True,
    )
    panel, selected, panel_report = demand.build_modeling_panel(daily, flag_events)
    product = demand.build_product_table(lines, set(panel["product_id"]))

    ledger, inventory_report = inventory.build_inventory(panel)
    supplier_products = suppliers.pick_supplier_products(panel)
    supplier = suppliers.build_supplier_table()
    supplier_product = suppliers.build_supplier_product_table(supplier_products)
    scenario, scenario_state, scenario_param = suppliers.build_scenarios(routes)

    sale_lines = lines.rename(columns={"raw_row": "source_row"})[
        ["order_id", "line_no", "source_row", "product_id", "quantity_gross", "quantity_cancelled", "quantity_net", "unit_price_gbp", "order_ts",
         "customer_id", "location_id", "country_iso3", "is_quantity_outlier"]
    ].assign(provenance=REAL)

    tables = {
        "country": country, "port": ports, "product": product, "supplier": supplier, "supplier_product": supplier_product,
        "warehouse": inventory.build_warehouse_table(), "route": routes, "sales_order_line": sale_lines, "demand": daily,
        "demand_modeling_panel": panel, "inventory": ledger, "inventory_policy": inventory.build_policy_table(), "tariff": tariff,
        "tariff_latest": tariff_latest, "weather_event_noaa": weather, "disruption_event": events, "disruption_impact": impact,
        "scenario": scenario, "scenario_state": scenario_state, "scenario_param": scenario_param,
    }
    reports = {
        "cancellations": cancel_report, "demand_panel": {**panel_report, "selected_series": selected.to_dict("records")}, "inventory": inventory_report,
        "ports": port_report, "routes": route_report, "tariffs": tariff_report, "weather": weather_report, "authored_events": event_report,
        "supplier_products": {"products": supplier_products},
    }
    return tables, reports, {"cancellation_pairs": pairs, "cancellation_outcomes": outcomes}


def legacy_comparison(tables: dict[str, pd.DataFrame], reports: dict[str, dict]) -> dict:
    """What the Phase 3 files got wrong, measured again here, and what changed. Read from data/processed, never written."""
    old_inventory = pd.read_csv(PROCESSED_DIR / "inventory.csv", usecols=["warehouse_id", "product_id", "date"])
    old_suppliers = pd.read_csv(PROCESSED_DIR / "suppliers.csv", dtype={"product_id": str})
    old_panel = pd.read_csv(PROCESSED_DIR / "demand_modeling_panel.csv", usecols=["product_id"], dtype={"product_id": str})
    old_demand_units = int(pd.read_csv(PROCESSED_DIR / "demand.csv", usecols=["demand_quantity"])["demand_quantity"].sum())
    old_series, new_series = set(old_panel["product_id"]), set(tables["demand_modeling_panel"]["product_id"])
    return {
        "phase3_inventory_rows": len(old_inventory),
        "phase3_inventory_repeated_keys": int(old_inventory.duplicated(["warehouse_id", "product_id", "date"]).sum()),
        "phase3_inventory_warehouse_ids": sorted(old_inventory["warehouse_id"].unique().tolist()),
        "phase3_supplier_status_in_master": old_suppliers.groupby("supplier_id")["status"].first().to_dict(),
        "phase3_supplier_products": sorted(old_suppliers["product_id"].unique().tolist()),
        "cleaned_supplier_products": reports["supplier_products"]["products"],
        "phase3_demand_units": old_demand_units,
        "cleaned_demand_units": int(tables["demand"]["demand_quantity"].sum()),
        "phase3_panel_products": len(old_series),
        "cleaned_panel_products": len(new_series),
        "panel_products_kept": len(old_series & new_series),
        "panel_products_dropped": sorted(old_series - new_series),
        "panel_products_added": sorted(new_series - old_series),
    }


def column_dictionary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name in LOAD_ORDER:
        spec, frame = SPECS[name], tables[name]
        for column in frame.columns:
            s = frame[column]
            text_length = int(s.dropna().astype(str).str.len().max()) if s.dtype == object and s.notna().any() else None
            rows.append(
                {
                    "table": name, "column": column, "dtype": str(s.dtype), "is_key": column in spec.key, "null_count": int(s.isna().sum()),
                    "null_pct": round(100 * float(s.isna().mean()), 3), "max_text_length": text_length,
                    "provenance": spec.provenance if column == "provenance" else spec.column_provenance.get(column, spec.provenance),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    started = time.time()
    raw_before = hash_tree(RAW_DIR)
    processed_before = hash_tree(PROCESSED_DIR)

    print("[1/5] building tables ...")
    tables, reports, audit = build_tables()

    print("[2/5] integrity checks ...")
    results = checks.run_all(tables)
    failed = [r for r in results if not r[1]]
    for name, passed, detail in results:
        print(f"   {'ok  ' if passed else 'FAIL'} {name}" + ("" if passed else f"  -- {detail}"))
    if failed:
        print(f"\n{len(failed)} check(s) failed; nothing was written.")
        return 1

    print("[3/5] writing data/cleaned ...")
    file_info: dict[str, dict] = {}
    for name in LOAD_ORDER:
        path = layer_io.write_table(name, tables[name], CLEANED_DIR)
        spec = SPECS[name]
        file_info[name] = {
            "file": path.name, "rows": len(tables[name]), "columns": len(tables[name].columns), "key": list(spec.key), "provenance": spec.provenance,
            "hana_table": spec.hana_table, "hana_tier": spec.hana_tier, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    for name, frame in audit.items():
        layer_io.write_table(name, frame, INTERIM_DIR)
    dictionary = column_dictionary(tables)
    dictionary.to_csv(CLEANED_DIR / "data_dictionary.csv", index=False, lineterminator="\n")

    print("[4/5] proving the sources were not touched ...")
    raw_unchanged = hash_tree(RAW_DIR) == raw_before
    processed_unchanged = hash_tree(PROCESSED_DIR) == processed_before
    if not (raw_unchanged and processed_unchanged):
        print("data/raw or data/processed changed during the build: this must never happen.")
        return 1

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": "python -m backend.services.preprocessing.cleaned.build",
        "environment": {"python": platform.python_version(), "pandas": pd.__version__},
        "seconds": round(time.time() - started, 1),
        "raw_sources_unchanged": raw_unchanged,
        "raw_sha256": raw_before,
        "processed_layer_unchanged": processed_unchanged,
        "tables": file_info,
        "steps": reports,
        "legacy_comparison": legacy_comparison(tables, reports),
        "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in results],
        "checks_failed": 0,
    }
    (CLEANED_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print("[5/5] writing DATA_READINESS_REPORT.md ...")
    (CLEANED_DIR / "DATA_READINESS_REPORT.md").write_text(report.render(manifest, dictionary, tables), encoding="utf-8")
    print(f"\nDone in {manifest['seconds']}s: {len(tables)} tables, {len(results)} checks passed, raw unchanged, processed unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
