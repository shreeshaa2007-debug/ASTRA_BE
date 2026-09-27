"""The registry of the cleaned layer: one entry per file in data/cleaned, declaring its key, its foreign keys, its provenance and
where it lands in SAP HANA Cloud.

The build, the integrity checks, the tests and the readiness report all read this, so a file's key or mapping is written down once.
`hana_table` names the table of the PROPOSED HANA schema (the one in the data-audit report); nothing is created in HANA. A `hana_notes`
entry says where the file differs from that proposal, because the proposal was drafted before the data was cleaned and a few of its
columns did not survive contact with the data.

Load tiers:
    CORE       load these; the application and the agents need them
    OPTIONAL   ready and correct, but large or redundant: load only if you want the extra lineage
    NOT_LOADED clean and usable, but recommended to stay in the file / AI layer
"""
from __future__ import annotations

from dataclasses import dataclass, field

from backend.services.preprocessing.cleaned import AUTHORED, DERIVED, REAL, SYNTHETIC
from backend.services.preprocessing.cleaned.countries import COUNTRY_COLUMN_PROVENANCE
from backend.services.preprocessing.cleaned.demand import PRODUCT_COLUMN_PROVENANCE
from backend.services.preprocessing.cleaned.network import ROUTE_COLUMN_PROVENANCE
from backend.services.preprocessing.cleaned.reference import WEATHER_COLUMN_PROVENANCE

CORE, OPTIONAL, NOT_LOADED = "CORE", "OPTIONAL", "NOT_LOADED"

FK = tuple[tuple[str, ...], str, tuple[str, ...]]  # (child columns, parent table, parent columns)


@dataclass(frozen=True)
class TableSpec:
    name: str
    key: tuple[str, ...]
    description: str
    provenance: str  # of the rows; a column may differ, see column_provenance
    hana_table: str
    hana_tier: str
    hana_notes: str = ""
    column_provenance: dict[str, str] = field(default_factory=dict)
    foreign_keys: tuple[FK, ...] = ()

    @property
    def file(self) -> str:
        return f"{self.name}.csv"


_SPECS = [
    TableSpec("country", ("iso3",), "217 World Bank countries with the ISO-2 code and retail label the data needs; the join key for tariffs, ports, demand and suppliers.",
              REAL, "country", CORE, "Adds region and income_group. iso2 is NULL for countries the data does not use (assumption A19).", COUNTRY_COLUMN_PROVENANCE),
    TableSpec("port", ("port_id",), "All 3,669 World Port Index ports (2017 snapshot), keyed by the WPI's own INDEX_NO.", REAL, "port", CORE,
              "Adds country_iso2 and source. country_iso3 is NULL for ports outside the mapped countries. The proposal's chan_depth and cargo_depth "
              "are omitted: no agent reads them and their unit could not be verified offline.",
              {"country_iso3": DERIVED},
              foreign_keys=((("country_iso3",), "country", ("iso3",)),)),
    TableSpec("product", ("product_id",), "Every product that has a sale line: name, median price, ABC class and volume from the corrected net demand.", REAL, "product", CORE,
              "The proposal's category column is not produced (no source), so leave it out or NULL. avg_unit_price is avg_unit_price_gbp; adds volume and "
              "date columns. product_id is text ('85123A').", PRODUCT_COLUMN_PROVENANCE),
    TableSpec("supplier", ("supplier_id",), "Eight fictitious suppliers, named SYNTHETIC. Nominal master data only: no status.", SYNTHETIC, "supplier", CORE,
              "home_port_id in the proposal is origin_port_id here (NULL for S007 and S008, as in the optimizer's configuration). Adds city, region, risk_level.",
              foreign_keys=((("country_iso3",), "country", ("iso3",)), (("origin_port_id",), "port", ("port_id",)))),
    TableSpec("supplier_product", ("supplier_id", "product_id"), "Which supplier carries which product, with nominal capacity, unit cost, lead time and reliability.",
              SYNTHETIC, "supplier_product", CORE,
              "The proposal's capacity_per_week is capacity here: the optimizer treats it as a cap over a planning run, with no time basis, so do not "
              "name it per-week. unit_cost is in ABSTRACT cost units (A14).",
              foreign_keys=((("supplier_id",), "supplier", ("supplier_id",)), (("product_id",), "product", ("product_id",)))),
    TableSpec("warehouse", ("warehouse_id",), "Three synthetic warehouses in India that the inventory ledger is split across.", SYNTHETIC, "warehouse", CORE,
              "name is warehouse_name; port_id is nearest_port_id (NULL for inland Delhi); adds city and provenance.",
              foreign_keys=((("country_iso3",), "country", ("iso3",)), (("nearest_port_id",), "port", ("port_id",)))),
    TableSpec("route", ("route_id",), "Ten lanes: a Suez and a Cape lane for each of four origins, plus one rail and one air lane.", SYNTHETIC, "route", CORE,
              "dest_port_id is destination_port_id; mode is transport_mode; base_transit_days is transit_time_days; capacity_per_week is capacity (no time "
              "basis). Adds lane_role, chokepoints, distance_basis, legacy_distance_km, route_note and the origin and destination labels.",
              ROUTE_COLUMN_PROVENANCE,
              foreign_keys=((("origin_port_id",), "port", ("port_id",)), (("destination_port_id",), "port", ("port_id",)))),
    TableSpec("sales_order_line", ("order_id", "line_no"), "Every priced sale line of the workbook with gross, cancelled and net quantity: the lineage behind demand.",
              REAL, "sales_order_line", OPTIONAL,
              "Not the proposal's quantity + is_cancellation: cancellation invoices are not rows here; each sale line carries quantity_gross, "
              "quantity_cancelled and quantity_net instead. quantity_cancelled, quantity_net and is_quantity_outlier are DERIVED. Large (about 520k rows): "
              "load it only if you want to recompute demand inside HANA.",
              {"quantity_cancelled": DERIVED, "quantity_net": DERIVED, "is_quantity_outlier": DERIVED, "country_iso3": DERIVED},
              foreign_keys=((("product_id",), "product", ("product_id",)), (("country_iso3",), "country", ("iso3",)))),
    TableSpec("demand", ("date", "product_id", "location_id"), "Net demand per day, product and country (cancellations netted out), no features.", DERIVED, "demand_daily", CORE,
              "The proposal made demand_daily a VIEW over sales_order_line. Because netting is done outside HANA, load this file as a TABLE instead and "
              "skip the view (or load both and keep the view for cross-checking).",
              foreign_keys=((("product_id",), "product", ("product_id",)), (("country_iso3",), "country", ("iso3",)))),
    TableSpec("demand_modeling_panel", ("date", "product_id", "location_id"), "The 40 forecastable series on a dense daily calendar, with lag and rolling features and a time split.",
              DERIVED, "ref_demand_panel", CORE,
              "Maps to the EXISTING ref_demand_panel contract, which reads 6 of the 17 columns (date, product_id, location_id, demand_quantity, "
              "rolling_mean_28, split). Load those 6; the lag and rolling features stay in the ML layer.",
              foreign_keys=((("product_id",), "product", ("product_id",)),)),
    TableSpec("inventory", ("warehouse_id", "product_id", "date"), "A simulated stock ledger, one row per warehouse, product and day, with lost sales recorded.", SYNTHETIC,
              "inventory_snapshot", CORE,
              "The proposal held on_hand, safety_stock and reorder_point only. This file also holds the flows (opening, inbound, outbound, closing), "
              "demand and unfilled quantity, so the table needs those columns; on_hand is closing_stock.",
              foreign_keys=((("warehouse_id",), "warehouse", ("warehouse_id",)), (("product_id",), "product", ("product_id",)))),
    TableSpec("inventory_policy", ("warehouse_id",), "The parameters the inventory simulation used, per warehouse.", SYNTHETIC, "inventory_policy", CORE,
              "A NEW table, not in the proposal: the policy is data, so the ledger can be reproduced from it.",
              foreign_keys=((("warehouse_id",), "warehouse", ("warehouse_id",)),)),
    TableSpec("tariff", ("origin_iso3", "dest_iso3", "hs_section", "effective_year"), "The World Bank tariff series as published: country and year, no value filled or changed.",
              REAL, "tariff", CORE, "Adds is_country_level_proxy, quality_flag and source. dest_iso3 is always 'ANY' and hs_section 'ALL' (the source has neither "
              "dimension); they stay in the key so a finer source can be loaded later.",
              foreign_keys=((("origin_iso3",), "country", ("iso3",)),)),
    TableSpec("tariff_latest", ("origin_iso3",), "One row per country: latest value, how stale it is, or NO_DATA.", REAL, "tariff_latest", OPTIONAL,
              "Derivable from tariff and country, so create it as a VIEW rather than loading the file, unless you want the staleness columns materialised.",
              {"reference_year": DERIVED, "staleness_years": DERIVED, "data_status": DERIVED, "country_name": REAL},
              foreign_keys=((("origin_iso3",), "country", ("iso3",)),)),
    TableSpec("weather_event_noaa", ("event_id",), "69,801 real US storm events of 2024 with parsed damage and a derived severity.", REAL, "weather_event", NOT_LOADED,
              "Not in the proposal on purpose: US weather shares no place or year with the demand data or the lanes. Keep it in the file layer as a "
              "reference for event durations and the Sensing agent's vocabulary. Ready to load if you want it.", WEATHER_COLUMN_PROVENANCE),
    TableSpec("disruption_event", ("event_id",), "Four hand-written historical events (AUTHORED), each with a verification status.", AUTHORED, "disruption_event", CORE,
              "region is location; adds estimated_delay_days, status, source_note and verification_status. The provenance CHECK in the proposal (REAL, "
              "DERIVED, SYNTHETIC) must also allow AUTHORED."),
    TableSpec("disruption_impact", ("event_id", "entity_type", "entity_id"), "Which lanes a historical event blocked (an inference, labelled).", AUTHORED, "disruption_impact", CORE,
              "The proposal's cost_multiplier is not produced (no source). Adds basis and provenance (AUTHORED).",
              foreign_keys=((("event_id",), "disruption_event", ("event_id",)),)),
    TableSpec("scenario", ("scenario_id",), "The scenarios: the normal baseline, the six of scenarios.yaml, a Red Sea diversion and the legacy demo start.", SYNTHETIC, "scenario", CORE,
              "name is scenario_name; adds scenario_type, is_modeled, based_on_event_id, source and provenance.",
              foreign_keys=((("based_on_event_id",), "disruption_event", ("event_id",)),)),
    TableSpec("scenario_state", ("scenario_id", "entity_type", "entity_id"), "What each scenario does to which lane or supplier. Only deviations from normal are written.", SYNTHETIC,
              "scenario_state", CORE, "A NEW table, not in the proposal: the proposal reused disruption_impact for this, which would have mixed hypothetical "
              "stress tests with historical events.", foreign_keys=((("scenario_id",), "scenario", ("scenario_id",)),)),
    TableSpec("scenario_param", ("scenario_id", "param_key", "qualifier"), "A scenario's numbers: duration, severity, tariff change per country.", SYNTHETIC, "scenario_param", CORE,
              "Adds qualifier (part of the key; 'ALL' when a parameter is not per-country), param_text, unit, basis and provenance.",
              foreign_keys=((("scenario_id",), "scenario", ("scenario_id",)),)),
]

SPECS: dict[str, TableSpec] = {s.name: s for s in _SPECS}

# parents before children, so a load never violates a foreign key
LOAD_ORDER = ("country", "port", "product", "warehouse", "supplier", "supplier_product", "route", "tariff", "disruption_event", "disruption_impact",
              "scenario", "scenario_state", "scenario_param", "inventory_policy", "inventory", "demand", "demand_modeling_panel",
              "sales_order_line", "tariff_latest", "weather_event_noaa")
assert set(LOAD_ORDER) == set(SPECS), "every table needs a place in the load order"
