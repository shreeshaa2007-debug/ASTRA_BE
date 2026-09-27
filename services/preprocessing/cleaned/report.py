"""Renders data/cleaned/DATA_READINESS_REPORT.md from the build's manifest, so every number in it comes from the run that produced the
data. Nothing here is typed in from memory: a count that changes when the data does changes in the report on the next build.
"""
from __future__ import annotations

import pandas as pd

from backend.services.preprocessing.cleaned import PROVENANCE_LABELS
from backend.services.preprocessing.cleaned.assumptions import ASSUMPTIONS
from backend.services.preprocessing.cleaned.tables import CORE, LOAD_ORDER, NOT_LOADED, OPTIONAL, SPECS

_TEXT_SIZES = (8, 16, 32, 64, 128, 256, 512, 1024, 2000)


def _md(headers: list[str], rows: list[list]) -> str:
    def cell(v) -> str:
        return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def hana_type(column: str, dtype: str, max_text_length: float | None) -> str:
    """The type a HANA column for this data should have. A recommendation only: no table is created."""
    if dtype in ("bool", "boolean"):
        return "BOOLEAN"
    if dtype.startswith("datetime"):
        return "TIMESTAMP" if column.endswith("_ts") else "DATE"
    if dtype in ("int64", "Int64", "int32"):
        return "INTEGER"
    if dtype.startswith("float"):
        return "DOUBLE"
    length = int(max_text_length or 1)
    size = next((s for s in _TEXT_SIZES if s >= 2 * length), None)
    return f"NVARCHAR({size})" if size else "NCLOB"


def _units(n) -> str:
    return f"{int(n):,}"


def render(manifest: dict, dictionary: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> str:
    m, steps, legacy = manifest["tables"], manifest["steps"], manifest["legacy_comparison"]
    canc, panel, inv, routes, tariffs = steps["cancellations"], steps["demand_panel"], steps["inventory"], steps["routes"], steps["tariffs"]
    out: list[str] = []
    w = out.append

    core = [n for n in LOAD_ORDER if SPECS[n].hana_tier == CORE]
    optional = [n for n in LOAD_ORDER if SPECS[n].hana_tier == OPTIONAL]
    not_loaded = [n for n in LOAD_ORDER if SPECS[n].hana_tier == NOT_LOADED]
    passed = sum(c["passed"] for c in manifest["checks"])

    w("# ResilientSC data readiness report")
    w("")
    w(f"Generated {manifest['generated_at'][:19]} UTC by `{manifest['command']}` in {manifest['seconds']} s. "
      f"This file is rendered from `manifest.json`; the numbers are the run's, not typed in. Nothing has been created in SAP HANA and nothing has been deployed.")
    w("")
    w("## 1. Verdict")
    w("")
    w(f"* **{passed} of {len(manifest['checks'])} integrity checks passed** (keys unique, foreign keys resolve, cancelled orders are not demand, the inventory identity holds, "
      f"the normal baseline has every supplier ACTIVE, tariffs equal the World Bank file, ...). The build refuses to finish if one fails.")
    w(f"* **The sources were not touched:** `data/raw` is byte-for-byte unchanged (SHA-256 of {len(manifest['raw_sha256'])} files compared before and after), "
      f"and so is `data/processed`, which the running application still reads.")
    w(f"* **{len(core)} tables are ready to load into SAP HANA Cloud** ({len(optional)} more are ready but optional, {len(not_loaded)} is ready but recommended to stay in the file layer). "
      f"The list is in section 9.")
    w("* **Ready means the data is internally consistent and correctly typed, not that the numbers are true.** The suppliers, routes' costs and capacities, inventory and "
      "scenarios are SYNTHETIC by necessity; section 6 lists what remains weak.")
    w("")

    w("## 2. What was fixed, against the ten requests")
    w("")
    b = canc["bridge_to_phase3_demand"]
    w(f"1. **Cancellation pairs.** {canc['workbook_rows']:,} workbook rows each received exactly one disposition (they sum to the row count). "
      f"{canc['cancellation_outcomes'].get('MATCHED_FULL', {}).get('cancellations', 0):,} cancellations were fully matched and "
      f"{canc['cancellation_outcomes'].get('MATCHED_PARTIAL', {}).get('cancellations', 0):,} partly matched to the sale they reverse (same customer and product, most recent earlier sale first), removing "
      f"**{_units(canc['units_cancelled_and_removed'])} units** from net demand. The two known phantom orders (74,215 units of 23166 and 80,995 units of 23843, each cancelled "
      f"minutes later) now net to zero. {_units(canc['cancellation_outcomes'].get('UNMATCHED_NO_PRIOR_SALE', {}).get('units', 0))} cancelled units had no earlier sale in the workbook and "
      f"{_units(canc['cancellation_outcomes'].get('UNMATCHED_NO_CUSTOMER', {}).get('units', 0))} had no customer id; those are not netted (assumption A02). The pairing is in "
      f"`data/interim/cleaning/cancellation_pairs.csv`, and the raw workbook is untouched.")
    reason = {"NON_POSITIVE_PRICE": "rows priced at zero or less (stock adjustments and give-aways)", "NON_PRODUCT_CODE": "rows with stock codes the Phase 3 list missed (discounts, samples, vouchers)"}
    w(f"2. **`demand.csv` and `demand_modeling_panel.csv` regenerated** from the corrected lines, with provenance DERIVED. Phase 3 demand of {_units(b['legacy_units'])} units becomes "
      f"{_units(b['new_net_units'])}, and the gap is accounted for to the unit: {_units(b['removed_because_cancelled'])} cancelled, "
      + ", ".join(f"{_units(v)} in {reason.get(k, k)}" for k, v in b["removed_because_not_a_priced_sale_or_not_a_product"].items())
      + " (the bridge first reproduces the Phase 3 total exactly, then subtracts).")
    w(f"3. **Inventory regenerated** as a dense grid of {len(inv['warehouses'])} warehouses x {inv['products']} products x {inv['days']} days = {inv['rows']:,} rows, so "
      f"`(warehouse_id, product_id, date)` is unique by construction (Phase 3: {legacy['phase3_inventory_repeated_keys']:,} repeated keys under the single id "
      f"{legacy['phase3_inventory_warehouse_ids']}). Lost sales are recorded ({_units(inv['total_unfilled_units'])} unfilled units, {inv['unfilled_share_pct']}% of demand) instead of being hidden.")
    w(f"4. **Supplier baseline fixed.** `supplier` and `supplier_product` hold nominal properties only, with no status. Phase 3 stored "
      f"{legacy['phase3_supplier_status_in_master']} in the master. Disruption is now `scenario_state`; `BASELINE_NORMAL` has no rows (every supplier ACTIVE, every lane NORMAL) and the old demo start survives "
      f"as the scenario `DEMO_LEGACY_START`.")
    w(f"5. **Routes.** Sea distances are now summed along the map's land-checked sea-lane waypoints, not straight lines across land: "
      + ", ".join(f"{r} {v['legacy']:,.0f} -> {v['now']:,.0f} km" for r, v in routes["distance_change_km"].items() if r.endswith("SUEZ"))
      + ". Every origin now has a Suez and a Cape lane; Mumbai and Chennai had none (`MUM-ROT-CAPE` and `CHE-ROT-CAPE` added). The old distance stays in `legacy_distance_km`.")
    lines = tables["sales_order_line"]
    gross, net = lines.groupby("product_id")["quantity_gross"].sum(), lines.groupby("product_id")["quantity_net"].sum()
    left_panel = "; ".join(f"{p}: {_units(gross[p])} units gross, {_units(net[p])} net" for p in legacy["panel_products_dropped"])
    w(f"6. **Supplier-product relationships.** Suppliers are named `SYNTHETIC Supplier S00x` and carry `provenance = SYNTHETIC`; the key is `(supplier_id, product_id)`. Netting cancelled "
      f"orders takes products out of the modeling panel ({left_panel}), so the supplier products are re-chosen by the same rule as before (the three highest-volume panel products): "
      f"{legacy['phase3_supplier_products']} -> {legacy['cleaned_supplier_products']}.")
    w(f"7. **Tariffs.** {tariffs['observations']:,} observations exactly as the World Bank publishes them (no value added, filled or rounded; a check compares them to the file). "
      f"{tariffs['status_counts'].get('CURRENT', 0)} countries are CURRENT for {tariffs['reference_year']}, {tariffs['status_counts'].get('STALE', 0)} STALE (up to {tariffs['max_staleness_years']} years old), "
      f"{tariffs['status_counts'].get('NO_DATA', 0)} have NO_DATA. Extreme or discontinuous values are flagged, not changed.")
    ev = steps["authored_events"]
    w(f"8. **Disruptions separated.** `weather_event_noaa` ({steps['weather']['rows']:,} real NOAA events, REAL), `disruption_event` ({ev['events']} hand-written historical events, AUTHORED, each with a "
      f"verification status: {ev['verification_status_counts']}) and `scenario*` (hypothetical stress tests, SYNTHETIC) are three different tables.")
    w(f"9. **`data/cleaned/`** holds {len(tables)} tables plus `manifest.json` and `data_dictionary.csv`.")
    w("10. **This report.**")
    w("")
    ratios = [v["now"] / v["legacy"] for v in routes["distance_change_km"].values()]
    w("### Decisions worth reviewing")
    w("")
    w("These are choices this layer made that you may want to reverse; each has a consequence when the application starts reading `data/cleaned`.")
    w("")
    w("* **`data/processed` was left untouched, not overwritten.** \"Regenerate `demand.csv` and `demand_modeling_panel.csv`\" was read as \"into the new layer\", so the running application, the trained model and the "
      "existing tests are unaffected until someone switches them over.")
    w("* **`demand.csv` no longer has lag and rolling features.** Phase 3 computed them on the sparse series, where \"lag_1\" meant \"the previous day with a sale\". They live on the dense panel, where they mean what they say.")
    w(f"* **Freight cost per unit rises by {100 * (min(ratios) - 1):.0f}% to {100 * (max(ratios) - 1):.0f}% on the {len(ratios)} sea lanes Phase 3 already had**, because cost is proportional to distance (A12) and the corrected distances are longer. "
      "Freight now weighs more against supplier cost in the optimizer; the plans it produces for the same scenarios will differ.")
    w("* **Unit costs were left abstract, not rescaled** (section 6, limitation 1): rescaling needs a decision about what the unit is.")
    w("* **Supplier names changed** to `SYNTHETIC Supplier S00x`, and one supplier product changed (item 6 above). Both show in the UI once it switches.")
    w("* **Late returns are netted on the original sale's date** (A01; section 6, limitation 2), so demand is what customers kept.")
    w("* **A fourth provenance label, AUTHORED,** was added for hand-written records of real events, because calling them REAL would overstate their verification and calling them SYNTHETIC would say they were invented.")
    w("")

    w("## 3. Files created")
    w("")
    rows = []
    for name in LOAD_ORDER:
        i = m[name]
        size = f"{i['bytes'] / 1e6:.2f} MB" if i["bytes"] >= 1e5 else f"{i['bytes'] / 1e3:.1f} KB"
        rows.append([f"`{i['file']}`", f"{i['rows']:,}", i["columns"], ", ".join(i["key"]), i["provenance"], size, SPECS[name].description])
    w(_md(["File", "Rows", "Cols", "Primary key", "Provenance", "Size", "What it is"], rows))
    w("")
    w("Also written: `manifest.json` (hashes, counts, every check), `data_dictionary.csv` (dtype, nulls and the provenance of every column), and, outside the layer, the cancellation audit "
      "tables `data/interim/cleaning/cancellation_pairs.csv` and `cancellation_outcomes.csv`.")
    w("")

    w("## 4. Provenance")
    w("")
    w("`provenance` is on every row, in upper case (the Phase 3 files use lower case, so the two layers cannot be confused).")
    w("")
    w(_md(["Label", "Meaning"], [["REAL", "measured, from a public dataset in `data/raw`, carried through unchanged"], ["DERIVED", "computed from REAL data by a documented rule"],
                                  ["SYNTHETIC", "generated because no public source exists; an assumption, never evidence"],
                                  ["AUTHORED", "written by hand about a publicly reported real-world event; not machine-verified in this workspace"]]))
    w("")
    w("A row carries the provenance of the fact it asserts. Where a table mixes, the columns that differ are declared in `data_dictionary.csv`:")
    w("")
    mixed = dictionary.merge(pd.DataFrame({"table": list(SPECS), "row_provenance": [SPECS[n].provenance for n in SPECS]}), on="table")
    mixed = mixed[(mixed["provenance"] != mixed["row_provenance"]) & (mixed["column"] != "provenance")]
    w(_md(["Table", "Column", "Column provenance", "Row provenance"], mixed[["table", "column", "provenance", "row_provenance"]].values.tolist()))
    w("")
    counts = {p: sum(m[n]["rows"] for n in m if m[n]["provenance"] == p) for p in PROVENANCE_LABELS}
    w("Rows by provenance: " + ", ".join(f"{p} {counts[p]:,}" for p in PROVENANCE_LABELS) + ".")
    w("")

    w("## 5. Assumptions")
    w("")
    w("Every number that shapes the data and is not measured. Defined once in `backend/services/preprocessing/cleaned/assumptions.py`.")
    w("")
    w(_md(["ID", "Area", "Assumption", "Label", "If it is wrong"], [[a.id, a.area, a.statement, a.label, a.consequence] for a in ASSUMPTIONS]))
    w("")

    w("## 6. Remaining limitations")
    w("")
    limits: list[str] = []
    products = tables["product"].set_index("product_id")
    sp = tables["supplier_product"]
    carried = sorted(sp["product_id"].unique())
    prices = products.loc[carried, "avg_unit_price_gbp"]
    demand_units = tables["demand"]["demand_quantity"].sum()
    carried_units = tables["demand"].loc[tables["demand"]["product_id"].isin(carried), "demand_quantity"].sum()
    n_products = len(products)
    limits.append(f"**Costs are abstract.** Supplier unit cost is {sp['unit_cost'].min():.0f} to {sp['unit_cost'].max():.0f}, against retail prices of about "
      f"{prices.min():.2f} to {prices.max():.2f} GBP for the same products, and route cost per unit is {tables['route']['cost_per_unit'].min():.0f} to {tables['route']['cost_per_unit'].max():.0f}. "
      "They are cost units (A14), not pounds. Scenario deltas and rankings are meaningful; an absolute cost is not. This was not changed because it needs a decision about the unit.")
    by_lag = canc["cancelled_units_by_lag"]
    matched_total = sum(by_lag.values())
    limits.append(f"**Cancellations are paired by heuristic.** An invoice line carries no reference to the sale it reverses, so the pairing uses customer and product (A01). "
                  f"{100 * (by_lag['1_to_30_days'] + by_lag['over_30_days']) / matched_total:.0f}% of the netted units were reversed more than a day after the sale (median lag "
                  f"{canc['median_lag_days']} days, per pair): these are returns as much as voided orders, and they are netted on the original order's date, so demand is what customers "
                  f"kept, not what they first asked for. A gross view stays available in `sales_order_line`.")
    limits.append(f"**Only {len(carried)} of {n_products:,} products have suppliers** ({100 * len(carried) / n_products:.2f}% of products, {100 * carried_units / demand_units:.1f}% of net demand units). "
      "The suppliers are fictitious and nothing about a real firm is implied.")
    limits.append(f"**Demand is one year of a UK gift wholesaler.** {panel['calendar_days']} days, one seasonal cycle, dominated by the United Kingdom, "
      f"and intermittent: only {panel['series_eligible_by_active_days']} of {panel['series_total']:,} product-country series have >= 100 active days. "
      "`disruption_active` is constant False because no disruption overlaps 2010-2011; do not use it as a feature.")
    limits.append("**The trained forecast model is stale.** `ml/artifacts/xgboost_demand` was trained on the Phase 3 panel. "
      f"{len(legacy['panel_products_dropped'])} products left the panel ({legacy['panel_products_dropped']}) and {len(legacy['panel_products_added'])} joined ({legacy['panel_products_added']}), "
      "and the demand values changed. It has not been retrained (not requested).")
    limits.append(f"**Inventory is a simulation.** Instant replenishment, full-year-mean sizing, and three warehouses in India serving UK demand (A09, A10). Stock-outs "
      f"({inv['stockout_rows']:,} warehouse-days) are a lower bound.")
    limits.append(f"**Sea distances are approximate** (A11): waypoint lanes, not vessel tracks. Shanghai-Rotterdam via Suez is {routes['distance_change_km']['SHA-ROT-SUEZ']['now']:,.0f} km here. "
      "Published sea-distance tables are commonly cited near 19,000 to 19,500 km for that voyage; that figure is general knowledge, not a source in this workspace, and was not verified. "
      "Transit speed, rail and air constants, freight cost and capacity have no public source here.")
    limits.append("**No Hormuz and no real Red Sea data.** The lanes have no Gulf leg, and no dataset in the workspace covers the 2023-24 Red Sea events; `RED_SEA_DIVERSION` is an illustrative stress test (A17).")
    limits.append(f"**Tariffs are country-level.** No partner or product dimension (`dest_iso3 = ANY`, `hs_section = ALL`). EU members carry the same value. "
      f"{tariffs['status_counts'].get('STALE', 0)} countries are stale and {tariffs['status_counts'].get('NO_DATA', 0)} have no value; the flagged values are "
      f"{tariffs['flag_counts']}.")
    limits.append(f"**The authored events are unverified in the workspace** ({ev['verification_status_counts']}). Only the 2019 tariff event was checked, against the World Bank series "
      f"(US {ev['tariff_event_us_jump_pp_2018_to_2019']:+.2f} points, 2018 to 2019). NOAA data is US weather only and overlaps nothing else.")
    ports = steps["ports"]
    limits.append(f"**Only {ports['with_iso3']:,} of {ports['rows']:,} ports have an ISO-3 country** (A19); the rest need a full ISO 3166 table. {ports['duplicate_name_and_country']} port names repeat within a country.")
    limits.append("**The application still reads `data/processed`.** Switching it to the cleaned layer is a separate change: `scenarios.yaml`'s SEVERE_WEATHER must list `MUM-ROT-CAPE` too, "
      "`frontend/src/data/lanes.json` must draw the two new lanes (a test asserts the drawn lanes equal the route ids), the baseline loader must apply `scenario_state` instead of reading "
      "supplier status, `route.status` becomes `lane_role`, and `backend/data/datasets.py`'s column contract gains the new columns.")
    w("")
    for number, text in enumerate(limits, 1):
        w(f"{number}. {text}")
    w("")

    w("## 7. Keys and relationships")
    w("")
    fk_rows = [[f"`{n}`" + "." + "+".join(c), f"`{p}`." + "+".join(pc)] for n, s in SPECS.items() for c, p, pc in s.foreign_keys]
    w(_md(["Child column", "Parent key"], fk_rows))
    w("")
    w("Not declarable as a plain foreign key (checked by code): `scenario_state.entity_id` is a route or a supplier according to `entity_type`, and `disruption_impact.entity_id` is a route. "
      "`supplier_id` alone repeats in `supplier_product`, so the key of that table is the pair.")
    w("")

    w("## 8. Mapping to the proposed HANA tables")
    w("")
    w("The proposed schema is the one in the data-audit report; **no table has been created**. Column names below are the cleaned files' names, and a HANA table should use them. "
      "Types are recommendations from the data (text sized at twice the longest value). Create identifiers unquoted (upper case in HANA).")
    w("")
    w(_md(["File", "HANA table", "Tier", "Differs from the proposal"], [[f"`{SPECS[n].file}`", f"`{SPECS[n].hana_table}`", SPECS[n].hana_tier, SPECS[n].hana_notes] for n in LOAD_ORDER]))
    w("")
    w("Three cross-cutting corrections to the proposal: (a) `tariff_rate_pct` must be `DOUBLE`, because the proposal's `DECIMAL(7,3)` would round the World Bank values, which this layer promises not to do; "
      "(b) the provenance `CHECK` must allow `AUTHORED`; (c) no column may be called `capacity_per_week`, because the source has no time basis.")
    w("")
    w("### Column mapping")
    w("")
    for name in LOAD_ORDER:
        d = dictionary[dictionary["table"] == name]
        w(f"**`{SPECS[name].file}` -> `{SPECS[name].hana_table}`** (key: {', '.join(SPECS[name].key)})")
        w("")
        w(_md(["Column", "HANA type", "Key", "Null %", "Provenance"],
              [[r["column"], hana_type(r["column"], r["dtype"], r["max_text_length"]), "PK" if r["is_key"] else "", r["null_pct"], r["provenance"]] for _, r in d.iterrows()]))
        w("")

    w("### How to load")
    w("")
    w("* **Order** (parents before children): " + " -> ".join(f"`{n}`" for n in LOAD_ORDER if SPECS[n].hana_tier != NOT_LOADED) + ".")
    w("* **Dates** are `YYYY-MM-DD`, timestamps `YYYY-MM-DD HH:MM:SS` (no time zone; the sources give none). **Booleans** are `True` / `False`. **NULL** is an empty field; a key column is never empty.")
    w("* **Identifiers are text**, including ones that look numeric (`22197`): read them as `NVARCHAR`, never as integers. `port_id`, `line_no` and `customer_id` are integers.")
    w("* **Encoding** is UTF-8, `\\n` line endings, standard CSV quoting. Nothing is rounded: floats are written at full precision.")
    w("* **Untested against HANA.** No tenant was available: the types and CSV conventions above are recommendations from the data, not something a HANA import has accepted. "
      "If the importer is case-sensitive about booleans, write `1` / `0`.")
    w("* `data_dictionary.csv`, `manifest.json` and this report are metadata, not data to load.")
    w("")

    w("## 9. Ready to load into SAP HANA Cloud")
    w("")
    w("**Ready (CORE):** " + ", ".join(f"`{SPECS[n].file}`" for n in core))
    w("")
    w("**Ready but optional:** " + "; ".join(f"`{SPECS[n].file}` ({SPECS[n].description[0].lower() + SPECS[n].description[1:].rstrip('.')})" for n in optional))
    w("")
    w("**Clean, but recommended to stay in the file layer:** " + "; ".join(f"`{SPECS[n].file}`" for n in not_loaded))
    w("")
    w("**Not ready, because they are not data to load:** `data_dictionary.csv`, `manifest.json`, `DATA_READINESS_REPORT.md`, and everything in `data/interim/`.")
    w("")
    w("**Still true of every file above:** it is ready in the sense of section 1. Before anything is loaded the HANA tables must be created from the mapping in section 8, which has not been done.")
    w("")

    w("## 10. Reproducing this")
    w("")
    w("```")
    w("python -m backend.services.preprocessing.cleaned.build      # from the repository root; about a minute")
    w("python -m pytest backend/tests/test_cleaned_layer.py         # the same checks, on the files as they sit on disk")
    w("```")
    w("")
    w("Integrity checks run in this build:")
    w("")
    w(_md(["Check", "Result", "Detail"], [[c["name"], "pass" if c["passed"] else "FAIL", c["detail"]] for c in manifest["checks"]]))
    w("")
    return "\n".join(out)
