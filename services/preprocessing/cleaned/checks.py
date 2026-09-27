"""Integrity gates for the cleaned layer. The build refuses to finish if one fails, and the tests run the same checks on the files
as they sit on disk, so "ready" means these passed and not that the code looks right.

Two kinds: generic (every table's key is unique and not null, its provenance labels are valid, its foreign keys resolve) and the
specific promises the cleaning makes (cancelled orders are not demand, the inventory identity holds, the normal baseline has every
supplier ACTIVE, ...). Each returns (name, passed, detail).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from backend.services.preprocessing.cleaned import PROVENANCE_LABELS
from backend.services.preprocessing.cleaned import reference as reference_mod
from backend.services.preprocessing.cleaned.tables import SPECS

Check = tuple[str, bool, str]

PHANTOM_ORDERS = {"541431": 74215, "581483": 80995}  # order -> units; each was cancelled minutes later (see cancellations.py)
STATE_COLUMNS_THAT_BELONG_TO_SCENARIOS = {"status", "state", "disrupted", "is_disrupted"}


def _c(name: str, passed: bool, detail: str = "") -> Check:
    return (name, bool(passed), detail)


def generic_checks(tables: dict[str, pd.DataFrame]) -> list[Check]:
    out: list[Check] = []
    for name, spec in SPECS.items():
        frame = tables[name]
        missing = [c for c in spec.key if c not in frame.columns]
        if missing:
            out.append(_c(f"{name}: key columns exist", False, f"missing {missing}"))
            continue
        nulls = int(frame[list(spec.key)].isna().any(axis=1).sum())
        dupes = int(frame.duplicated(subset=list(spec.key)).sum())
        out.append(_c(f"{name}: key {list(spec.key)} has no nulls", nulls == 0, f"{nulls} null-key rows"))
        out.append(_c(f"{name}: key {list(spec.key)} is unique", dupes == 0, f"{dupes} repeated keys in {len(frame):,} rows"))
        bad = sorted(set(frame["provenance"].dropna().unique()) - set(PROVENANCE_LABELS)) if "provenance" in frame else ["no provenance column"]
        out.append(_c(f"{name}: every row has a valid provenance", not bad and frame["provenance"].notna().all(), f"invalid: {bad}"))
        for child_cols, parent, parent_cols in spec.foreign_keys:
            child = frame[list(child_cols)].dropna()
            parent_keys = tables[parent][list(parent_cols)].dropna().drop_duplicates()
            orphans = int(len(child.merge(parent_keys, left_on=list(child_cols), right_on=list(parent_cols), how="left", indicator=True).query("_merge == 'left_only'")))
            out.append(_c(f"{name}.{'+'.join(child_cols)} -> {parent}.{'+'.join(parent_cols)} resolves", orphans == 0, f"{orphans} orphan rows"))
    return out


def layer_checks(t: dict[str, pd.DataFrame]) -> list[Check]:
    out: list[Check] = []
    lines, demand, panel, inv = t["sales_order_line"], t["demand"], t["demand_modeling_panel"], t["inventory"]

    # ---- demand: cancelled orders are not demand
    out.append(_c("demand: gross - cancelled = net on every sale line, and net is never negative",
                  ((lines["quantity_gross"] - lines["quantity_cancelled"]) == lines["quantity_net"]).all() and (lines["quantity_net"] >= 0).all()))
    out.append(_c("demand: total net units on sale lines = total units in demand.csv",
                  int(lines["quantity_net"].sum()) == int(demand["demand_quantity"].sum()), f"{int(lines['quantity_net'].sum()):,} vs {int(demand['demand_quantity'].sum()):,}"))
    phantom = lines[lines["order_id"].isin(PHANTOM_ORDERS)]
    out.append(_c("demand: the two known cancelled orders (74,215 and 80,995 units) contribute nothing",
                  len(phantom) == 2 and (phantom["quantity_net"] == 0).all() and int(demand["demand_quantity"].max()) < min(PHANTOM_ORDERS.values()),
                  f"largest daily demand is {int(demand['demand_quantity'].max()):,}"))
    out.append(_c("demand: no non-positive demand rows", (demand["demand_quantity"] > 0).all()))
    out.append(_c("demand: every sale line is a priced sale", (lines["unit_price_gbp"] > 0).all() and (lines["quantity_gross"] > 0).all()))

    # ---- the modeling panel agrees with demand.csv
    p = panel.assign(date=pd.to_datetime(panel["date"]))
    d = demand.assign(date=pd.to_datetime(demand["date"]))
    series = p[["product_id", "location_id"]].drop_duplicates()
    sub = d.merge(series, on=["product_id", "location_id"])
    out.append(_c("panel: each series' total equals its total in demand.csv (densifying only adds zero days)",
                  int(sub["demand_quantity"].sum()) == int(p["demand_quantity"].sum()), f"{int(sub['demand_quantity'].sum()):,} vs {int(p['demand_quantity'].sum()):,}"))
    days = p.groupby(["product_id", "location_id"])["date"].nunique()
    out.append(_c("panel: dense calendar (every series has every day)", days.nunique() == 1 and int(days.iloc[0]) == p["date"].nunique(), f"{p['date'].nunique()} days"))
    out.append(_c("panel: 40 series", len(series) == 40, f"{len(series)} series"))

    # ---- inventory: unique key, the identity, no hidden lost sales
    i = inv.assign(date=pd.to_datetime(inv["date"]))
    grid = i["warehouse_id"].nunique() * i["product_id"].nunique() * i["date"].nunique()
    out.append(_c("inventory: (warehouse_id, product_id, date) is unique AND the grid is complete", len(i) == grid and not i.duplicated(["warehouse_id", "product_id", "date"]).any(),
                  f"{len(i):,} rows, grid {grid:,}"))
    out.append(_c("inventory: closing = opening + inbound - outbound exactly, with no clipping",
                  (i["opening_stock"] + i["inbound_quantity"] - i["outbound_quantity"] == i["closing_stock"]).all()))
    out.append(_c("inventory: stock and flows are never negative",
                  bool((i[["opening_stock", "inbound_quantity", "outbound_quantity", "closing_stock", "unfilled_quantity", "safety_stock"]] >= 0).all().all())))
    ordered = i.sort_values(["warehouse_id", "product_id", "date"])
    prev_closing = ordered.groupby(["warehouse_id", "product_id"])["closing_stock"].shift(1)
    out.append(_c("inventory: each day opens with the previous day's closing stock", bool((ordered["opening_stock"][prev_closing.notna()] == prev_closing.dropna()).all())))
    out.append(_c("inventory: outbound + unfilled = demand, and stockout_flag = (unfilled > 0)",
                  bool(((i["outbound_quantity"] + i["unfilled_quantity"] == i["demand_quantity"]) & (i["stockout_flag"] == (i["unfilled_quantity"] > 0))).all())))
    by_day = i.groupby(["product_id", "date"])["demand_quantity"].sum().reset_index().merge(p[["product_id", "date", "demand_quantity"]], on=["product_id", "date"], suffixes=("_inv", "_panel"))
    out.append(_c("inventory: the warehouses' demand sums to the panel's demand on every product-day (no unit lost or invented)",
                  len(by_day) == len(i) // i["warehouse_id"].nunique() and (by_day["demand_quantity_inv"] == by_day["demand_quantity_panel"]).all()))
    out.append(_c("inventory: only products of the modeling panel", set(i["product_id"]) <= set(p["product_id"])))

    # ---- suppliers: nominal master data, pair key, no state
    sup, sp = t["supplier"], t["supplier_product"]
    stateful = sorted((set(sup.columns) | set(sp.columns)) & STATE_COLUMNS_THAT_BELONG_TO_SCENARIOS)
    out.append(_c("suppliers: no disruption state in the master tables", not stateful, f"state columns found: {stateful}"))
    out.append(_c("suppliers: supplier_id alone repeats in supplier_product (so the pair is the key), and is unique in supplier",
                  sp["supplier_id"].duplicated().any() and sup["supplier_id"].is_unique))
    out.append(_c("suppliers: every supplier is named SYNTHETIC", sup["supplier_name"].str.startswith("SYNTHETIC").all()))
    out.append(_c("suppliers: every supplier carries a product and every product has at least two suppliers",
                  set(sup["supplier_id"]) <= set(sp["supplier_id"]) and (sp.groupby("product_id")["supplier_id"].nunique() >= 2).all()))
    out.append(_c("suppliers: nominal capacity is positive for every pair (a disruption is not baked in)", (sp["capacity"] > 0).all()))
    out.append(_c("suppliers: supplier products are products of the modeling panel and of the inventory ledger",
                  set(sp["product_id"]) <= set(p["product_id"]) and set(sp["product_id"]) <= set(i["product_id"])))

    # ---- scenarios: the baseline is normal, disruption is data
    sc, st, prm = t["scenario"], t["scenario_state"], t["scenario_param"]
    out.append(_c("scenarios: BASELINE_NORMAL exists and changes nothing (no state rows: every supplier ACTIVE, every lane NORMAL)",
                  "BASELINE_NORMAL" in set(sc["scenario_id"]) and not (st["scenario_id"] == "BASELINE_NORMAL").any()))
    legacy = st[st["scenario_id"] == "DEMO_LEGACY_START"].set_index("entity_id")["status"].to_dict()
    out.append(_c("scenarios: the old demo start (S001 DISRUPTED, S002 REDUCED) is kept as a scenario", legacy == {"S001": "DISRUPTED", "S002": "REDUCED"}, str(legacy)))
    wanted = set(yaml.safe_load(open("backend/config/scenarios.yaml", encoding="utf-8"))["scenarios"])
    out.append(_c("scenarios: every scenario of scenarios.yaml is present", wanted <= set(sc["scenario_id"]), f"missing {sorted(wanted - set(sc['scenario_id']))}"))
    route_ids, supplier_ids = set(t["route"]["route_id"]), set(sup["supplier_id"])
    bad_state = st[~(((st["entity_type"] == "ROUTE") & st["entity_id"].isin(route_ids)) | ((st["entity_type"] == "SUPPLIER") & st["entity_id"].isin(supplier_ids)))]
    out.append(_c("scenarios: every scenario_state entity is a real route or supplier", bad_state.empty, f"{len(bad_state)} unknown entities"))
    bad_impact = t["disruption_impact"][~t["disruption_impact"]["entity_id"].isin(route_ids)]
    out.append(_c("scenarios: every historical impact names a real route", bad_impact.empty))
    out.append(_c("scenarios: scenario_param keys are unique per scenario and qualifier", not prm.duplicated(["scenario_id", "param_key", "qualifier"]).any()))

    # ---- routes: every origin has a way round
    r = t["route"]
    sea = r[r["transport_mode"] == "sea"].assign(origin_code=lambda d: d["route_id"].str[:3])
    both = sea.groupby("origin_code")["via"].apply(lambda v: {"SUEZ", "CAPE"} <= set(v))
    out.append(_c("routes: each of the four origins has both a SUEZ and a CAPE lane", bool(both.all()) and len(both) == 4, str(both.to_dict())))
    wide = sea.pivot(index="origin_code", columns="via", values="distance_km")
    out.append(_c("routes: the Cape lane is longer than the Suez lane for every origin", bool((wide["CAPE"] > wide["SUEZ"]).all()), str((wide["CAPE"] / wide["SUEZ"]).round(2).to_dict())))
    out.append(_c("routes: every Phase 3 route id is kept", {"SHA-ROT-SUEZ", "SHA-ROT-CAPE", "SIN-ROT-SUEZ", "SIN-ROT-CAPE", "MUM-ROT-SUEZ", "CHE-ROT-SUEZ", "SHA-ROT-RAIL", "SHA-ROT-AIR"} <= route_ids))
    suez_lanes = set(r.loc[r["chokepoints"].fillna("").str.split("|").apply(lambda cs: "Suez Canal" in cs), "route_id"])
    config = yaml.safe_load(open("backend/config/scenarios.yaml", encoding="utf-8"))["scenarios"]["SUEZ_CLOSURE"]["event"]["affected_routes"]
    out.append(_c("routes: the lanes scenarios.yaml says a Suez closure blocks are exactly the lanes through the canal", suez_lanes == set(config), f"{sorted(suez_lanes)} vs {sorted(config)}"))

    # ---- tariffs: as published
    tariff = t["tariff"]
    wide_raw = pd.read_csv(reference_mod.WDI_DATA, skiprows=4)
    years = [c for c in wide_raw.columns if c.isdigit()]
    raw_long = wide_raw.melt(id_vars=["Country Code"], value_vars=years, var_name="y", value_name="v").dropna(subset=["v"])
    raw_long = raw_long[raw_long["Country Code"].isin(set(t["country"]["iso3"]))]
    merged = tariff.assign(y=tariff["effective_year"].astype(str)).merge(raw_long, left_on=["origin_iso3", "y"], right_on=["Country Code", "y"], how="outer", indicator=True)
    # 1e-9 absorbs the last-digit noise of reading a CSV back; rounding to even 4 decimals (Phase 3 did) would differ by up to 5e-5 and fail
    out.append(_c("tariffs: every value equals the World Bank file, none added, filled or rounded",
                  bool((merged["_merge"] == "both").all()) and bool(np.isclose(merged["tariff_rate_pct"], merged["v"], rtol=0, atol=1e-9).all()), f"{len(tariff):,} observations"))
    tl = t["tariff_latest"]
    out.append(_c("tariffs: every country is CURRENT, STALE or NO_DATA", set(tl["data_status"]) <= {"CURRENT", "STALE", "NO_DATA"} and len(tl) == len(t["country"])))

    # ---- provenance is kept apart
    out.append(_c("events: NOAA rows are all REAL and never mixed with AUTHORED events", set(t["weather_event_noaa"]["provenance"]) == {"REAL"} and set(t["disruption_event"]["provenance"]) == {"AUTHORED"}))
    synthetic = ("supplier", "supplier_product", "warehouse", "route", "inventory", "inventory_policy", "scenario", "scenario_state", "scenario_param")
    out.append(_c("provenance: everything invented is SYNTHETIC", all(set(t[n]["provenance"]) == {"SYNTHETIC"} for n in synthetic)))
    out.append(_c("provenance: the real sources are REAL", all(set(t[n]["provenance"]) == {"REAL"} for n in ("country", "port", "product", "tariff", "sales_order_line"))))
    return out


def run_all(tables: dict[str, pd.DataFrame]) -> list[Check]:
    return generic_checks(tables) + layer_checks(tables)
