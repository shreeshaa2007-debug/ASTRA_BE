"""The cleaned data layer (data/cleaned): the fixes, and the gates that keep them fixed.

Three kinds of test, deliberately:

* toy data, for the logic (cancellation pairing, the inventory ledger, the allocation of integer demand);
* the built files on disk, for the promises the data makes (run `python -m backend.services.preprocessing.cleaned.build` first;
  these skip if the layer has not been built);
* mutations: each corrupts one thing in the built layer and requires the integrity checks to notice. A gate that has never been
  seen failing proves nothing, and the first build of this layer passed 119 of 119 checks on its first run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.services.preprocessing import demand as legacy_demand
from backend.services.preprocessing.cleaned import CLEANED_DIR, PROCESSED_DIR, cancellations, checks, layer_io, network, suppliers
from backend.services.preprocessing.cleaned import assumptions as A
from backend.services.preprocessing.cleaned import inventory as inv_mod
from backend.services.preprocessing.cleaned.tables import CORE, LOAD_ORDER, NOT_LOADED, OPTIONAL, SPECS

BUILT = (CLEANED_DIR / "manifest.json").exists()
needs_build = pytest.mark.skipif(not BUILT, reason="build the layer first: python -m backend.services.preprocessing.cleaned.build")


# --------------------------------------------------------------------------- cancellations, on toy data
def sale(invoice, code, qty, when, customer=1.0, price=2.5, country="United Kingdom", desc="THING"):
    return (invoice, code, desc, qty, pd.Timestamp(when), price, customer, country)


def raw(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=cancellations.RAW_COLUMNS)


def lines_of(*rows):
    lines, pairs, outcomes, report = cancellations.build_sale_lines(raw(*rows))
    return lines.set_index("order_id"), pairs, outcomes, report


def test_a_cancellation_reverses_the_sale_it_names_and_nets_to_zero():
    """The audit's finding in miniature: a 74,215-unit order cancelled minutes later must not be demand."""
    lines, pairs, outcomes, _ = lines_of(sale("100", "X", 74215, "2011-01-18 10:01"), sale("C101", "X", -74215, "2011-01-18 10:17"))
    assert lines.loc["100", "quantity_net"] == 0 and lines.loc["100", "quantity_cancelled"] == 74215
    assert outcomes["outcome"].tolist() == [cancellations.MATCHED_FULL]
    assert pairs["matched_quantity"].tolist() == [74215]


def test_a_partial_cancellation_reduces_only_the_units_cancelled():
    lines, *_ = lines_of(sale("100", "X", 100, "2011-01-01"), sale("C101", "X", -30, "2011-01-05"))
    assert lines.loc["100", "quantity_net"] == 70


def test_a_cancellation_reverses_the_most_recent_sale_first_and_spills_into_older_ones():
    lines, pairs, *_ = lines_of(sale("100", "X", 10, "2011-01-01"), sale("101", "X", 20, "2011-01-02"), sale("C102", "X", -25, "2011-01-03"))
    assert lines.loc["101", "quantity_net"] == 0  # the newer sale goes first
    assert lines.loc["100", "quantity_net"] == 5  # and the remaining 5 units come off the older one
    assert pairs["matched_quantity"].sum() == 25


def test_a_cancellation_never_reverses_a_sale_made_after_it():
    lines, _, outcomes, _ = lines_of(sale("C100", "X", -5, "2011-01-01"), sale("101", "X", 20, "2011-01-02"))
    assert lines.loc["101", "quantity_net"] == 20
    assert outcomes["outcome"].tolist() == [cancellations.UNMATCHED_NO_PRIOR_SALE]


def test_a_cancellation_only_touches_the_same_customer_and_the_same_product():
    lines, *_ = lines_of(sale("100", "X", 10, "2011-01-01", customer=1.0), sale("101", "Y", 10, "2011-01-01", customer=2.0),
                         sale("102", "Y", 10, "2011-01-01", customer=1.0), sale("C103", "X", -10, "2011-01-02", customer=1.0))
    assert lines.loc["100", "quantity_net"] == 0
    assert lines.loc["101", "quantity_net"] == 10 and lines.loc["102", "quantity_net"] == 10


def test_a_cancellation_larger_than_what_was_sold_is_partial_and_net_never_goes_negative():
    lines, _, outcomes, _ = lines_of(sale("100", "X", 10, "2011-01-01"), sale("C101", "X", -25, "2011-01-02"))
    assert lines.loc["100", "quantity_net"] == 0
    row = outcomes.iloc[0]
    assert (row["outcome"], row["matched_quantity"], row["unmatched_quantity"]) == (cancellations.MATCHED_PARTIAL, 10, 15)


def test_a_cancellation_without_a_customer_is_recorded_as_unmatched_not_guessed():
    lines, _, outcomes, _ = lines_of(sale("100", "X", 10, "2011-01-01"), sale("C101", "X", -10, "2011-01-02", customer=np.nan))
    assert lines.loc["100", "quantity_net"] == 10
    assert outcomes["outcome"].tolist() == [cancellations.UNMATCHED_NO_CUSTOMER]


def test_every_row_gets_exactly_one_disposition_and_the_ones_that_are_not_sales_are_excluded():
    rows = [
        sale("100", "X", 5, "2011-01-01"),
        sale("100", "X", 5, "2011-01-01"),  # an exact duplicate of the row above
        sale("101", "POST", 1, "2011-01-01", price=18.0),  # postage is not a product
        sale("102", "GIFT_0001_20", 1, "2011-01-01"),  # nor is a voucher
        sale("103", "Y", 4, "2011-01-01", price=0.0),  # a give-away
        sale("104", "Y", -3, "2011-01-01", customer=np.nan, desc="damaged"),  # a write-off on a non-C invoice
        sale("105", "Z", 2, "2011-01-01", country="Unspecified"),
        sale("106", "Z", 2, "2011-01-01"),
    ]
    lines, _, _, report = lines_of(*rows)
    d = report["dispositions"]
    assert sum(d.values()) == len(rows)
    assert d == {"DUPLICATE_ROW": 1, "NON_PRODUCT_CODE": 2, "NON_POSITIVE_PRICE": 1, "STOCK_ADJUSTMENT": 1, "UNSPECIFIED_COUNTRY": 1, "SALE": 2}
    assert sorted(lines.index) == ["100", "106"]


def test_the_demand_bridge_accounts_for_every_unit_between_the_old_rules_and_the_new():
    """`build_sale_lines` asserts the bridge closes; this shows what it is made of on a case with all three reasons."""
    _, _, _, report = lines_of(sale("100", "X", 10, "2011-01-01"), sale("C101", "X", -4, "2011-01-02"), sale("102", "Y", 7, "2011-01-01", price=0.0),
                               sale("103", "D", 1, "2011-01-01"))
    bridge = report["bridge_to_phase3_demand"]
    assert bridge["legacy_units"] == 10 + 7 + 1  # what the Phase 3 rules kept: three positive rows (D was not on its list)
    assert bridge["removed_because_cancelled"] == 4
    assert bridge["removed_because_not_a_priced_sale_or_not_a_product"] == {"NON_POSITIVE_PRICE": 7, "NON_PRODUCT_CODE": 1}
    assert bridge["new_net_units"] == 6


# --------------------------------------------------------------------------- inventory, on toy data
def test_integer_demand_is_split_across_warehouses_without_losing_or_inventing_a_unit():
    shares = [0.45, 0.30, 0.25]
    demand = np.random.default_rng(0).integers(0, 300, size=2000)
    split = inv_mod.allocate_demand(demand, shares)
    assert (split.sum(axis=0) == demand).all()  # Phase 3 rounded each share separately and did not have this property
    assert (split >= 0).all()
    assert (np.abs(split - np.outer(shares, demand)) < 1).all()  # nobody is off by a whole unit from their share


def test_a_stockout_is_recorded_as_unfilled_demand_and_stock_never_goes_negative():
    # stock is sized on the series' mean (A09), so the spike must be an outlier against a long, quiet series to exceed it
    demand = np.ones(300, dtype="int64")
    spike = 150
    demand[spike] = 1_000
    ledger = inv_mod.simulate(demand, cover_multiplier=1.0, safety_multiplier=1.0)
    assert ledger["unfilled"][spike] > 0 and ledger["closing"][spike] == 0
    assert ((ledger["outbound"] + ledger["unfilled"]) == demand).all()
    assert (ledger["opening"] + ledger["inbound"] - ledger["outbound"] == ledger["closing"]).all()  # exact, no clipping
    assert (ledger["closing"] >= 0).all()


def test_the_ledger_is_a_complete_grid_with_a_unique_key():
    days = pd.date_range("2011-01-01", periods=30)
    panel = pd.concat([pd.DataFrame({"date": days, "product_id": p, "location_id": "United Kingdom", "demand_quantity": np.arange(30) % 7}) for p in ("A1", "B2")])
    ledger, report = inv_mod.build_inventory(panel)
    assert len(ledger) == 3 * 2 * 30 and not ledger.duplicated(["warehouse_id", "product_id", "date"]).any()
    assert report["total_demand_units"] == int(panel["demand_quantity"].sum())


# --------------------------------------------------------------------------- suppliers and scenarios
def test_adding_a_product_does_not_change_another_pairs_numbers():
    two = suppliers.build_supplier_product_table(["22197", "84077"]).set_index(["supplier_id", "product_id"])
    three = suppliers.build_supplier_product_table(["22197", "84077", "85099B"]).set_index(["supplier_id", "product_id"])
    shared = two.index.intersection(three.index)
    assert len(shared) > 5
    pd.testing.assert_frame_equal(two.loc[shared], three.loc[shared])


def test_the_supplier_product_key_is_the_pair_and_every_supplier_carries_something():
    sp = suppliers.build_supplier_product_table(["22197", "84077", "85099B"])
    assert not sp.duplicated(["supplier_id", "product_id"]).any()
    assert sp["supplier_id"].duplicated().any()  # so supplier_id alone is not a key
    assert set(sp["supplier_id"]) == {r[0] for r in suppliers.ROSTER}
    assert (sp["capacity"] > 0).all()  # nominal: nothing is disrupted in the master


def test_the_scenarios_take_their_lanes_from_the_network_and_the_state_layers_rule():
    routes, _ = network.build_route_table()
    scenario, state, params = suppliers.build_scenarios(routes)
    by = lambda sid: state[state["scenario_id"] == sid]  # noqa: E731
    assert by("BASELINE_NORMAL").empty
    assert set(by("SUEZ_CLOSURE")["entity_id"]) == {"SHA-ROT-SUEZ", "SIN-ROT-SUEZ", "MUM-ROT-SUEZ", "CHE-ROT-SUEZ"}
    assert set(by("RED_SEA_DIVERSION")["entity_id"]) == set(by("SUEZ_CLOSURE")["entity_id"])  # every SUEZ lane passes Bab-el-Mandeb
    assert set(by("SEVERE_WEATHER")["entity_id"]) == {"MUM-ROT-SUEZ", "MUM-ROT-CAPE"}  # a closed port closes both of its lanes
    assert by("SUPPLIER_FAILURE")[["entity_type", "entity_id", "status"]].values.tolist() == [["SUPPLIER", "S007", "DISRUPTED"]]
    assert params.query("scenario_id == 'TARIFF_INCREASE' and param_key == 'tariff_add_pct'")[["qualifier", "param_value"]].values.tolist() == [["TUR", 25.0]]


# --------------------------------------------------------------------------- routes
def test_sea_distances_are_no_longer_straight_lines_across_land():
    routes, report = network.build_route_table()
    sea = routes[routes["transport_mode"] == "sea"].set_index("route_id")["distance_km"]
    assert sea["SHA-ROT-SUEZ"] - sea["SIN-ROT-SUEZ"] > 3000  # Phase 3 had them 19 km apart
    for origin in ("SHA", "SIN", "MUM", "CHE"):
        assert sea[f"{origin}-ROT-CAPE"] > sea[f"{origin}-ROT-SUEZ"]
        assert 1.2 < report["cape_over_suez"][origin] < 2.0
    assert {"MUM-ROT-CAPE", "CHE-ROT-CAPE"} <= set(sea.index)


def test_no_sea_lane_leg_the_distance_is_summed_over_crosses_land():
    """The same coastline test the map's lanes pass, applied to the great-circle legs this layer actually measures."""
    from backend.tests import test_map_lanes as land  # noqa: PLC0415  (loads the map's own land polygons)

    builder = network.lane_builder()
    waypoints, _ = network.waypoints_and_chokepoints()
    assert len(waypoints) == 8
    for route_id, lane in waypoints.items():
        points = []
        for p, q in zip(lane, lane[1:]):
            points += builder.great_circle(p, q, n=max(2, int(network.haversine_km(p, q) / 15)))
        ends = [lane[0], lane[-1]]
        on_land = [(round(a, 2), round(o, 2)) for a, o in points if not land.exempt(a, o, ends) and land.on_land(a, o)]
        assert not on_land, f"{route_id} crosses land at {on_land[:5]}"


def test_the_hand_mapped_route_ports_are_the_ports_the_map_draws():
    """Guards assumption A18: each WPI id must sit within 30 km of the coordinates the lanes start from."""
    ports, _ = network.build_port_table()
    by_id = ports.set_index("port_id")
    builder = network.lane_builder()
    for name, port_id in network.ROUTE_PORT_IDS.items():
        row = by_id.loc[port_id]
        assert network.haversine_km((row["latitude"], row["longitude"]), builder.PORTS[name]) < 30, name


# --------------------------------------------------------------------------- consistency with the rest of the codebase
def test_constants_shared_with_earlier_modules_have_not_drifted():
    from ml.training import prepare_modeling_data as ml_panel  # noqa: PLC0415

    assert (A.MODELING_TOP_K, A.MODELING_MIN_ACTIVE_DAYS) == (ml_panel.TOP_K, ml_panel.MIN_ACTIVE_DAYS)
    assert legacy_demand._NON_PRODUCT_CODES <= A.NON_PRODUCT_CODES  # the new list only ever adds


def test_every_table_has_a_hana_mapping_a_tier_and_a_place_in_the_load_order():
    assert set(LOAD_ORDER) == set(SPECS) and len(LOAD_ORDER) == len(set(LOAD_ORDER))
    assert all(s.hana_table and s.hana_tier in (CORE, OPTIONAL, NOT_LOADED) for s in SPECS.values())
    position = {name: i for i, name in enumerate(LOAD_ORDER)}
    for name, spec in SPECS.items():
        for _cols, parent, _pcols in spec.foreign_keys:
            assert position[parent] < position[name], f"{name} would load before its parent {parent}"


# --------------------------------------------------------------------------- the built layer
@pytest.fixture(scope="module")
def layer() -> dict[str, pd.DataFrame]:
    if not BUILT:
        pytest.skip("layer not built")
    return layer_io.read_layer()


def failing(tables) -> list[str]:
    return [name for name, passed, _ in checks.run_all(tables) if not passed]


@needs_build
def test_every_integrity_check_passes_on_the_files_as_they_sit_on_disk(layer):
    assert failing(layer) == []


@needs_build
def test_the_sources_are_exactly_what_they_were_when_the_layer_was_built():
    """data/raw and data/processed are read-only inputs: their hashes must still match what the build recorded."""
    manifest = json.loads((CLEANED_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["raw_sources_unchanged"] and manifest["processed_layer_unchanged"]
    for path, digest in manifest["raw_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path


@needs_build
def test_the_manifest_matches_the_files_and_the_dictionary_covers_every_column(layer):
    manifest = json.loads((CLEANED_DIR / "manifest.json").read_text(encoding="utf-8"))
    dictionary = pd.read_csv(CLEANED_DIR / "data_dictionary.csv")
    for name, frame in layer.items():
        assert manifest["tables"][name]["rows"] == len(frame), name
        assert manifest["tables"][name]["columns"] == len(frame.columns), name
        assert set(dictionary.loc[dictionary["table"] == name, "column"]) == set(frame.columns), name
    assert manifest["checks_failed"] == 0


@needs_build
def test_the_phase3_demand_total_is_reproduced_exactly_before_anything_is_subtracted(layer):
    """If the reproduced Phase 3 rules did not give Phase 3's own number, the bridge would prove nothing."""
    manifest = json.loads((CLEANED_DIR / "manifest.json").read_text(encoding="utf-8"))
    old = pd.read_csv(PROCESSED_DIR / "demand.csv", usecols=["demand_quantity"])["demand_quantity"].sum()
    assert manifest["steps"]["cancellations"]["bridge_to_phase3_demand"]["legacy_units"] == old


@needs_build
def test_noaa_severity_follows_the_phase3_rule_row_for_row(layer):
    old = pd.read_csv(PROCESSED_DIR / "disruptions.csv", usecols=["event_id", "severity"])
    old = old[old["event_id"].str.startswith("NOAA-")].set_index("event_id")["severity"]
    new = layer["weather_event_noaa"].set_index("event_id")["severity"]
    assert len(new) == len(old) and (new.reindex(old.index) == old).all()


@needs_build
def test_the_normal_baseline_resolves_to_every_supplier_active_and_every_lane_normal(layer):
    state = layer["scenario_state"]
    overrides = state[state["scenario_id"] == "BASELINE_NORMAL"]
    suppliers_status = {s: "ACTIVE" for s in layer["supplier"]["supplier_id"]} | overrides.query("entity_type == 'SUPPLIER'").set_index("entity_id")["status"].to_dict()
    assert set(suppliers_status.values()) == {"ACTIVE"}
    assert set(layer["route"]["lane_role"]) == {"PRIMARY", "ALTERNATIVE"}  # a role, never a disruption


@needs_build
def test_the_supplier_products_are_the_three_highest_volume_products_of_the_panel(layer):
    panel = layer["demand_modeling_panel"]
    top = panel.groupby("product_id")["demand_quantity"].sum().sort_values(ascending=False).head(3).index
    assert set(layer["supplier_product"]["product_id"]) == set(top)
    assert "23166" not in set(panel["product_id"])  # its 74,215-unit order was cancelled; what was left is not top-40 demand


# --------------------------------------------------------------------------- mutations: the gates must fail when the data is wrong
def mutated(layer, **tables):
    return {**layer, **tables}


def assert_caught(layer, expected: str, **tables):
    names = failing(mutated(layer, **tables))
    assert any(expected in n for n in names), f"nothing caught it: expected a failing check containing {expected!r}, got {names}"


@needs_build
def test_mutation_a_cancelled_order_counted_as_demand_again(layer):
    lines = layer["sales_order_line"].copy()
    is_phantom = lines["order_id"] == "541431"
    lines.loc[is_phantom, "quantity_cancelled"] = 0
    lines.loc[is_phantom, "quantity_net"] = lines.loc[is_phantom, "quantity_gross"]
    assert_caught(layer, "the two known cancelled orders", sales_order_line=lines)


@needs_build
def test_mutation_a_repeated_inventory_key(layer):
    inv = pd.concat([layer["inventory"], layer["inventory"].iloc[[0]]], ignore_index=True)
    assert_caught(layer, "is unique", inventory=inv)
    assert_caught(layer, "grid is complete", inventory=inv)


@needs_build
def test_mutation_stock_that_does_not_add_up(layer):
    inv = layer["inventory"].copy()
    inv.loc[5, "closing_stock"] += 1
    assert_caught(layer, "closing = opening + inbound - outbound", inventory=inv)


@needs_build
def test_mutation_a_lost_sale_that_disappears(layer):
    inv = layer["inventory"].copy()
    inv.loc[7, "demand_quantity"] += 3  # demand that was neither shipped nor recorded as unfilled
    assert_caught(layer, "outbound + unfilled = demand", inventory=inv)


@needs_build
def test_mutation_disruption_state_leaking_back_into_the_supplier_master(layer):
    assert_caught(layer, "no disruption state", supplier=layer["supplier"].assign(status="ACTIVE"))
    sp = layer["supplier_product"].copy()
    sp.loc[0, "capacity"] = 0  # a supplier disrupted in the master
    assert_caught(layer, "nominal capacity is positive", supplier_product=sp)


@needs_build
def test_mutation_the_normal_baseline_with_a_disrupted_supplier(layer):
    state = pd.concat([layer["scenario_state"], pd.DataFrame([{"scenario_id": "BASELINE_NORMAL", "entity_type": "SUPPLIER", "entity_id": "S001", "status": "DISRUPTED",
                                                                "capacity_factor": 0.0, "basis": "x", "provenance": "SYNTHETIC"}])], ignore_index=True)
    assert_caught(layer, "BASELINE_NORMAL exists and changes nothing", scenario_state=state)


@needs_build
def test_mutation_a_tariff_that_was_rounded(layer):
    assert_caught(layer, "none added, filled or rounded", tariff=layer["tariff"].assign(tariff_rate_pct=layer["tariff"]["tariff_rate_pct"].round(1)))


@needs_build
def test_mutation_a_tariff_that_was_invented(layer):
    invented = pd.concat([layer["tariff"], layer["tariff"].iloc[[0]].assign(effective_year=2030, tariff_rate_pct=5.0)], ignore_index=True)
    assert_caught(layer, "none added, filled or rounded", tariff=invented)


@needs_build
def test_mutation_a_cape_lane_shorter_than_its_suez_lane(layer):
    route = layer["route"].copy()
    route.loc[route["route_id"] == "SHA-ROT-CAPE", "distance_km"] = 100.0
    assert_caught(layer, "Cape lane is longer", route=route)


@needs_build
def test_mutation_an_origin_with_no_way_round_the_cape(layer):
    route = layer["route"]
    assert_caught(layer, "both a SUEZ and a CAPE", route=route[route["route_id"] != "MUM-ROT-CAPE"].reset_index(drop=True))


@needs_build
def test_mutation_a_country_that_the_demand_refers_to_disappears(layer):
    country = layer["country"]
    assert_caught(layer, "resolves", country=country[country["iso3"] != "GBR"].reset_index(drop=True))


@needs_build
def test_mutation_real_and_authored_records_mixed(layer):
    assert_caught(layer, "NOAA rows are all REAL", weather_event_noaa=layer["weather_event_noaa"].assign(provenance="AUTHORED"))


@needs_build
def test_mutation_a_product_left_with_a_single_supplier(layer):
    sp = layer["supplier_product"]
    only_one = sp[(sp["product_id"] != "22197") | (sp["supplier_id"] == "S002")].reset_index(drop=True)
    assert_caught(layer, "at least two suppliers", supplier_product=only_one)


@needs_build
def test_mutation_a_panel_that_no_longer_matches_demand(layer):
    panel = layer["demand_modeling_panel"].copy()
    panel.loc[10, "demand_quantity"] += 1
    assert_caught(layer, "each series' total equals", demand_modeling_panel=panel)


@needs_build
def test_mutation_a_missing_or_repeated_key(layer):
    product = layer["product"].copy()
    product.loc[0, "product_id"] = None
    assert_caught(layer, "has no nulls", product=product)
    scenario = pd.concat([layer["scenario"], layer["scenario"].iloc[[0]]], ignore_index=True)
    assert_caught(layer, "is unique", scenario=scenario)


@needs_build
def test_mutation_a_row_with_an_invented_provenance_label(layer):
    assert_caught(layer, "valid provenance", supplier=layer["supplier"].assign(provenance="made-up"))
