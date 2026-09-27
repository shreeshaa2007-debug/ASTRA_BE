"""The SAP HANA Cloud core data model, as data: 17 tables, their columns, types, keys, foreign keys and checks.

This is the single source the DDL, the data dictionary, the validation SQL, the load plan and the loader are all generated from
(`python -m backend.database.hana_core`), so they cannot disagree. Its only input about the DATA is `data/cleaned/`, which nothing
here reads for modification: `hana_core.py` reads the CSVs to check that this specification fits them and to compute expected row
counts and control totals.

Decisions worth knowing before reading the tables (the ones that need approval are listed in HANA_LOAD_PLAN.md):

* Identifiers are upper case and QUOTED, so a column called "DATE" or "MONTH" cannot collide with a keyword. In HANA a quoted
  upper-case identifier is the same identifier as an unquoted one, so `SELECT date FROM demand` still works.
* A foreign key exists only where every non-null value in the cleaned data resolves to a parent key that is itself a primary key.
  Where it does not (a polymorphic entity id, a text label, the tariff sentinel 'ANY') there is NO foreign key, and the relationship is
  checked by the validation SQL instead. Referencing and referenced columns have IDENTICAL types.
* Measures with a fixed number of decimals are DECIMAL with headroom (never fewer decimals than the data has), derived statistics
  are DOUBLE, and `tariff_rate_pct` is DOUBLE as required. Nothing is rounded on the way in; the loader hands HANA exact `Decimal`s.
* `capacity` has no time unit anywhere in this model: the source states none and none is invented.
* CHECK constraints are limited to provenance, non-negativity, ranges the data's own meaning fixes, and vocabularies the codebase
  already defines (`backend/schemas/entities.py`, `EVENT_EFFECTS`). They are declared apart from the columns so that dropping one is a
  one-line change.
"""
from __future__ import annotations

from dataclasses import dataclass

PROVENANCE_VALUES = ("REAL", "DERIVED", "SYNTHETIC", "AUTHORED")

# --------------------------------------------------------------------------- domains: one definition per kind of column
ISO3 = "NVARCHAR(3)"
ISO2 = "NVARCHAR(2)"
PROVENANCE = "NVARCHAR(12)"
PRODUCT_ID = "NVARCHAR(32)"
SUPPLIER_ID = "NVARCHAR(16)"
WAREHOUSE_ID = "NVARCHAR(32)"
ROUTE_ID = "NVARCHAR(32)"
EVENT_ID = "NVARCHAR(64)"
SCENARIO_ID = "NVARCHAR(32)"
ENTITY_ID = "NVARCHAR(64)"  # a route or a supplier, by entity_type
PORT_ID = "INTEGER"
COORDINATE = "DECIMAL(9,6)"  # the data has 6 decimals
KILOMETERS = "DECIMAL(10,2)"  # the data has 1
DAYS = "DECIMAL(8,2)"  # the data has 2
MONEY = "DECIMAL(16,4)"  # abstract cost units or GBP; the data has 2 decimals
FRACTION = "DECIMAL(5,4)"  # 0..1
MULTIPLIER = "DECIMAL(6,4)"
PARAMETER = "DECIMAL(14,4)"
FREE_TEXT = "NVARCHAR(500)"
NOTE = "NVARCHAR(256)"

PROVENANCE_DESCRIPTION = "Where the row's facts come from: REAL, DERIVED, SYNTHETIC or AUTHORED (see DATA_READINESS_REPORT.md section 4)."

# vocabularies taken from the codebase, not invented here
RISK_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")  # backend/schemas/entities.py RiskLevel
EVENT_STATUSES = ("ACTIVE", "RESOLVED", "HISTORICAL")  # DisruptionStatus
DEVIATION_STATUSES = ("DISRUPTED", "REDUCED", "DELAYED")  # what EVENT_EFFECTS can set on a route or a supplier
ENTITY_TYPES = ("ROUTE", "SUPPLIER")
VERIFICATION_STATUSES = ("UNVERIFIED_IN_WORKSPACE", "CORROBORATED_BY_WDI", "NOT_CORROBORATED_BY_WDI")  # cleaned/reference.py
SCENARIO_TYPES = ("BASELINE", "DISRUPTION", "LEGACY_STATE")  # cleaned/suppliers.py
TRANSPORT_MODES = ("sea", "rail", "air")  # lower case, exactly as the cleaned routes have them
LANE_ROLES = ("PRIMARY", "ALTERNATIVE")


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    nullable: bool
    description: str


@dataclass(frozen=True)
class ForeignKey:
    name: str
    columns: tuple[str, ...]
    parent: str
    parent_columns: tuple[str, ...]


@dataclass(frozen=True)
class Check:
    name: str  # the part after CK_<TABLE>_
    expression: str
    description: str


@dataclass(frozen=True)
class Table:
    name: str
    csv: str
    description: str
    columns: tuple[Column, ...]
    primary_key: tuple[str, ...]
    foreign_keys: tuple[ForeignKey, ...] = ()
    checks: tuple[Check, ...] = ()

    def column(self, name: str) -> Column:
        return next(c for c in self.columns if c.name == name)


def col(name: str, type_: str, description: str) -> Column:
    return Column(name, type_, False, description)


def opt(name: str, type_: str, description: str) -> Column:
    return Column(name, type_, True, description)


def fk(table: str, columns: str | tuple[str, ...], parent: str, parent_columns: str | tuple[str, ...], suffix: str = "") -> ForeignKey:
    columns = (columns,) if isinstance(columns, str) else columns
    parent_columns = (parent_columns,) if isinstance(parent_columns, str) else parent_columns
    return ForeignKey(f"FK_{table}_{suffix or parent}", columns, parent, parent_columns)


def one_of(column: str, values: tuple[str, ...]) -> str:
    return f'"{column}" IN ({", ".join(repr(v) for v in values)})'


_PROV = col("PROVENANCE", PROVENANCE, PROVENANCE_DESCRIPTION)

TABLES: tuple[Table, ...] = (
    Table("COUNTRY", "country.csv", "The 217 World Bank countries, with the ISO-2 code and retail label the project's data needs. The join key for tariffs, ports, demand and suppliers.", (
        col("ISO3", ISO3, "ISO 3166-1 alpha-3 country code as the World Bank writes it (CHI for the Channel Islands, which ISO does not code). The key every country reference points at."),
        opt("ISO2", ISO2, "ISO 3166-1 alpha-2 code. Filled only for the countries the cleaned data uses; NULL for the rest, so it is not a complete ISO table."),
        col("NAME", "NVARCHAR(100)", "Country name as the World Bank writes it."),
        col("REGION", "NVARCHAR(64)", "World Bank region."),
        col("INCOME_GROUP", "NVARCHAR(32)", "World Bank income group."),
        opt("RETAIL_LABEL", "NVARCHAR(64)", "The label the retail dataset uses for this country ('EIRE', 'RSA', ...). NULL if the retail data never names it. Matches DEMAND.LOCATION_ID."),
        _PROV,
    ), ("ISO3",)),

    Table("PORT", "port.csv", "All 3,669 World Port Index ports (a 2017 snapshot), keyed by the index's own number.", (
        col("PORT_ID", PORT_ID, "World Port Index INDEX_NO."),
        col("PORT_NAME", "NVARCHAR(128)", "Port name as the World Port Index writes it, upper case. Not unique: some names repeat within a country, so the key is PORT_ID."),
        col("COUNTRY_ISO2", ISO2, "The World Port Index's 2-letter country code. Not a foreign key: COUNTRY.ISO2 is filled only for some countries."),
        opt("COUNTRY_ISO3", ISO3, "ISO-3 country, derived from COUNTRY_ISO2 where the country table can map it; NULL for the rest."),
        col("LATITUDE", COORDINATE, "Decimal degrees, north positive."),
        col("LONGITUDE", COORDINATE, "Decimal degrees, east positive."),
        opt("HARBOR_SIZE", "NVARCHAR(1)", "World Port Index harbour size code (L, M, S, V). NULL where the index does not report it."),
        opt("HARBOR_TYPE", "NVARCHAR(2)", "World Port Index harbour type code. NULL where not reported."),
        col("SOURCE", NOTE, "The dataset and vintage the row comes from."),
        _PROV,
    ), ("PORT_ID",), (fk("PORT", "COUNTRY_ISO3", "COUNTRY", "ISO3"),), (
        Check("LATITUDE", '"LATITUDE" BETWEEN -90 AND 90', "a latitude is within +-90 degrees"),
        Check("LONGITUDE", '"LONGITUDE" BETWEEN -180 AND 180', "a longitude is within +-180 degrees"),
    )),

    Table("PRODUCT", "product.csv", "Every product that has a sale line: name, median price, ABC class and volume from the corrected net demand.", (
        col("PRODUCT_ID", PRODUCT_ID, "The retail stock code, as text ('85123A', '22197'). Never a number."),
        col("PRODUCT_NAME", "NVARCHAR(200)", "The description most often used for the code."),
        opt("AVG_UNIT_PRICE_GBP", "DOUBLE", "Median unit price of the product's net-positive sale lines, in GBP. DOUBLE and not DECIMAL because a few medians of two prices carry floating-point noise in the cleaned file that must not be rounded away silently. NULL for a product whose every sale was cancelled."),
        col("ABC_CLASS", "NVARCHAR(1)", "A, B or C by cumulative net revenue (80% / 95%). A revenue ranking, not a measure of criticality."),
        col("NET_UNITS", "INTEGER", "Units sold net of cancellations."),
        col("NET_REVENUE_GBP", MONEY, "Net units times price, in GBP."),
        col("ACTIVE_DAYS", "INTEGER", "Days with a net sale."),
        opt("FIRST_SALE_DATE", "DATE", "First day with a net sale. NULL if none."),
        opt("LAST_SALE_DATE", "DATE", "Last day with a net sale. NULL if none."),
        col("IS_FORECAST_SERIES", "BOOLEAN", "True for the 40 products of the modeling panel."),
        _PROV,
    ), ("PRODUCT_ID",), (), (
        Check("ABC_CLASS", one_of("ABC_CLASS", ("A", "B", "C")), "ABC class vocabulary"),
        Check("NET_UNITS", '"NET_UNITS" >= 0', "net units cannot be negative"),
        Check("NET_REVENUE", '"NET_REVENUE_GBP" >= 0', "net revenue cannot be negative"),
        Check("ACTIVE_DAYS", '"ACTIVE_DAYS" >= 0', "a count cannot be negative"),
        Check("SALE_DATES", '"FIRST_SALE_DATE" IS NULL OR "LAST_SALE_DATE" IS NULL OR "FIRST_SALE_DATE" <= "LAST_SALE_DATE"', "the first sale is not after the last"),
    )),

    Table("WAREHOUSE", "warehouse.csv", "Three synthetic warehouses in India that the inventory ledger is split across.", (
        col("WAREHOUSE_ID", WAREHOUSE_ID, "Warehouse code (the city's name)."),
        col("WAREHOUSE_NAME", "NVARCHAR(128)", "Display name, marked SYNTHETIC."),
        col("CITY", "NVARCHAR(64)", "City."),
        col("COUNTRY_ISO3", ISO3, "Country of the warehouse."),
        opt("NEAREST_PORT_ID", PORT_ID, "Nearest World Port Index port; NULL for inland Delhi."),
        _PROV,
    ), ("WAREHOUSE_ID",), (fk("WAREHOUSE", "COUNTRY_ISO3", "COUNTRY", "ISO3"), fk("WAREHOUSE", "NEAREST_PORT_ID", "PORT", "PORT_ID", "NEAREST_PORT"))),

    Table("SUPPLIER", "supplier.csv", "Eight fictitious suppliers, named SYNTHETIC. Nominal master data only: no status, so the normal baseline has every supplier active.", (
        col("SUPPLIER_ID", SUPPLIER_ID, "Supplier code S001-S008. Unique here; it repeats in SUPPLIER_PRODUCT, whose key is the pair."),
        col("SUPPLIER_NAME", "NVARCHAR(128)", "'SYNTHETIC Supplier S00x': nothing implies a real company."),
        col("CITY", "NVARCHAR(64)", "The hypothetical location."),
        col("COUNTRY_ISO3", ISO3, "Country of the supplier."),
        col("REGION", "NVARCHAR(64)", "Country name, as the optimizer's configuration calls it."),
        col("RISK_LEVEL", "NVARCHAR(16)", "Static risk tier, assigned by country. Not derived from reliability."),
        opt("ORIGIN_PORT_ID", PORT_ID, "The port whose lanes carry this supplier's goods (from the optimizer's configuration). NULL where no freight leg is modeled."),
        _PROV,
    ), ("SUPPLIER_ID",), (fk("SUPPLIER", "COUNTRY_ISO3", "COUNTRY", "ISO3"), fk("SUPPLIER", "ORIGIN_PORT_ID", "PORT", "PORT_ID", "ORIGIN_PORT")), (
        Check("RISK_LEVEL", one_of("RISK_LEVEL", RISK_LEVELS), "risk vocabulary of the platform"),
    )),

    Table("SUPPLIER_PRODUCT", "supplier_product.csv", "Which supplier carries which product, with nominal capacity, unit cost, lead time and reliability. All SYNTHETIC.", (
        col("SUPPLIER_ID", SUPPLIER_ID, "The supplier."),
        col("PRODUCT_ID", PRODUCT_ID, "The product."),
        col("CAPACITY", "INTEGER", "Nominal maximum units of this product the supplier can provide in one planning run. The source states NO time unit and none is assumed here."),
        col("UNIT_COST", MONEY, "Cost per unit in ABSTRACT cost units: not pounds, and not commensurate with PRODUCT.AVG_UNIT_PRICE_GBP."),
        col("LEAD_TIME_DAYS", "INTEGER", "Days from order to shipment."),
        col("RELIABILITY", FRACTION, "Share of orders delivered as promised, 0 to 1; follows the supplier's risk tier."),
        _PROV,
    ), ("SUPPLIER_ID", "PRODUCT_ID"), (fk("SUPPLIER_PRODUCT", "SUPPLIER_ID", "SUPPLIER", "SUPPLIER_ID"), fk("SUPPLIER_PRODUCT", "PRODUCT_ID", "PRODUCT", "PRODUCT_ID")), (
        Check("CAPACITY", '"CAPACITY" > 0', "nominal capacity is positive: a disruption is scenario data, never the master"),
        Check("UNIT_COST", '"UNIT_COST" > 0', "a unit cost is positive"),
        Check("LEAD_TIME", '"LEAD_TIME_DAYS" >= 0', "a lead time cannot be negative"),
        Check("RELIABILITY", '"RELIABILITY" BETWEEN 0 AND 1', "a share"),
    )),

    Table("ROUTE", "route.csv", "Ten lanes to Rotterdam: a Suez lane and a Cape lane from each of four origins, plus one rail and one air lane.", (
        col("ROUTE_ID", ROUTE_ID, "Lane code, e.g. SHA-ROT-SUEZ."),
        col("ORIGIN", "NVARCHAR(64)", "Origin port name as the routes have always called it."),
        col("DESTINATION", "NVARCHAR(64)", "Destination port name."),
        col("ORIGIN_PORT_ID", PORT_ID, "Origin, as a World Port Index port."),
        col("DESTINATION_PORT_ID", PORT_ID, "Destination, as a World Port Index port."),
        col("TRANSPORT_MODE", "NVARCHAR(16)", "sea, rail or air (lower case, as delivered)."),
        col("LANE_ROLE", "NVARCHAR(16)", "PRIMARY or ALTERNATIVE. A role, never a disruption: disruption is scenario data."),
        col("VIA", "NVARCHAR(16)", "SUEZ, CAPE, RAIL or AIR."),
        opt("CHOKEPOINTS", NOTE, "Chokepoints the lane passes, separated by '|'. NULL for rail and air."),
        col("DISTANCE_KM", KILOMETERS, "Kilometres. Sea lanes: summed great-circle legs along land-checked waypoints (approximate; see DISTANCE_BASIS)."),
        col("DISTANCE_BASIS", "NVARCHAR(32)", "How DISTANCE_KM was obtained: WAYPOINT_POLYLINE, GREAT_CIRCLE or REPORTED_SERVICE_RANGE."),
        opt("LEGACY_DISTANCE_KM", KILOMETERS, "The Phase 3 distance, kept so the correction is visible. NULL for the two lanes Phase 3 did not have."),
        col("CAPACITY", "INTEGER", "Abstract capacity units compared with unit quantities. No time unit is stated or assumed."),
        col("TRANSIT_TIME_DAYS", DAYS, "Days door to door, from the distance and an assumed speed."),
        col("COST_PER_UNIT", MONEY, "Freight cost per unit in abstract cost units."),
        col("ROUTE_NOTE", FREE_TEXT, "How the lane was built and what to be careful of."),
        _PROV,
    ), ("ROUTE_ID",), (fk("ROUTE", "ORIGIN_PORT_ID", "PORT", "PORT_ID", "ORIGIN_PORT"), fk("ROUTE", "DESTINATION_PORT_ID", "PORT", "PORT_ID", "DESTINATION_PORT")), (
        Check("TRANSPORT_MODE", one_of("TRANSPORT_MODE", TRANSPORT_MODES), "transport modes of the network"),
        Check("LANE_ROLE", one_of("LANE_ROLE", LANE_ROLES), "a lane's role"),
        Check("DISTANCE", '"DISTANCE_KM" > 0', "a distance is positive"),
        Check("CAPACITY", '"CAPACITY" >= 0', "a capacity cannot be negative"),
        Check("TRANSIT_TIME", '"TRANSIT_TIME_DAYS" > 0', "a transit time is positive"),
        Check("COST", '"COST_PER_UNIT" >= 0', "a cost cannot be negative"),
    )),

    Table("TARIFF", "tariff.csv", "The World Bank weighted-mean applied tariff, by country and year, exactly as published: nothing filled, interpolated or rounded.", (
        col("ORIGIN_ISO3", ISO3, "Country whose tariff this is."),
        col("DEST_ISO3", ISO3, "Always 'ANY': the source has no partner dimension. A sentinel, not a country, so not a foreign key."),
        col("HS_SECTION", "NVARCHAR(8)", "Always 'ALL': the source has no product dimension."),
        col("EFFECTIVE_YEAR", "INTEGER", "Year of the observation."),
        col("TARIFF_RATE_PCT", "DOUBLE", "Weighted-mean applied tariff, percent, at the World Bank's own precision. DOUBLE so no value is rounded."),
        col("IS_COUNTRY_LEVEL_PROXY", "BOOLEAN", "True: the value is a country-level proxy, not a country-pair by product rate."),
        col("QUALITY_FLAG", "NVARCHAR(32)", "OK, EXTREME_VALUE (above 50%), JUMP_GE_10PP (10 points or more from the previous observation) or both. Informational: no value was changed."),
        col("SOURCE", NOTE, "Indicator and publisher."),
        _PROV,
    ), ("ORIGIN_ISO3", "DEST_ISO3", "HS_SECTION", "EFFECTIVE_YEAR"), (fk("TARIFF", "ORIGIN_ISO3", "COUNTRY", "ISO3"),), (
        Check("RATE", '"TARIFF_RATE_PCT" >= 0', "a tariff is not negative; there is no upper bound because the source publishes values above 100%"),
        Check("YEAR", '"EFFECTIVE_YEAR" BETWEEN 1960 AND 2100', "the range the platform's TariffRecord accepts"),
    )),

    Table("DISRUPTION_EVENT", "disruption_event.csv", "Four historical events written by hand about publicly reported events (AUTHORED), each with a verification status. Not weather data and not scenarios.", (
        col("EVENT_ID", EVENT_ID, "Event code."),
        col("EVENT_TYPE", "NVARCHAR(64)", "Kind of event."),
        col("LOCATION", "NVARCHAR(255)", "Where it happened."),
        col("START_TS", "TIMESTAMP", "Start, no time zone."),
        opt("END_TS", "TIMESTAMP", "End; NULL if open-ended."),
        col("SEVERITY", "NVARCHAR(16)", "LOW, MEDIUM, HIGH or CRITICAL."),
        opt("ESTIMATED_DELAY_DAYS", DAYS, "The authors' estimate of the delay in days; NULL where none is estimated."),
        col("STATUS", "NVARCHAR(16)", "HISTORICAL for these events."),
        col("SOURCE_NOTE", FREE_TEXT, "What the event was and how far to trust the dates."),
        col("VERIFICATION_STATUS", "NVARCHAR(32)", "UNVERIFIED_IN_WORKSPACE, or CORROBORATED_BY_WDI where the World Bank series confirms it."),
        _PROV,
    ), ("EVENT_ID",), (), (
        Check("SEVERITY", one_of("SEVERITY", RISK_LEVELS), "severity vocabulary of the platform"),
        Check("STATUS", one_of("STATUS", EVENT_STATUSES), "event status vocabulary of the platform"),
        Check("VERIFICATION", one_of("VERIFICATION_STATUS", VERIFICATION_STATUSES), "verification statuses the cleaned layer assigns"),
        Check("PERIOD", '"END_TS" IS NULL OR "END_TS" >= "START_TS"', "an event does not end before it starts"),
        Check("DELAY", '"ESTIMATED_DELAY_DAYS" IS NULL OR "ESTIMATED_DELAY_DAYS" >= 0', "a delay cannot be negative"),
    )),

    Table("DISRUPTION_IMPACT", "disruption_impact.csv", "Which lanes a historical event blocked. An inference, labelled AUTHORED.", (
        col("EVENT_ID", EVENT_ID, "The event."),
        col("ENTITY_TYPE", "NVARCHAR(16)", "ROUTE or SUPPLIER: what ENTITY_ID names."),
        col("ENTITY_ID", ENTITY_ID, "A route id or a supplier id according to ENTITY_TYPE, so it has no foreign key; the validation SQL checks it."),
        opt("DELAY_DAYS", DAYS, "Days of delay the event caused there."),
        opt("CAPACITY_FACTOR", FRACTION, "Share of nominal capacity left, 0 to 1."),
        col("BASIS", FREE_TEXT, "Why this entity is listed."),
        _PROV,
    ), ("EVENT_ID", "ENTITY_TYPE", "ENTITY_ID"), (fk("DISRUPTION_IMPACT", "EVENT_ID", "DISRUPTION_EVENT", "EVENT_ID", "EVENT"),), (
        Check("ENTITY_TYPE", one_of("ENTITY_TYPE", ENTITY_TYPES), "entity types the platform models"),
        Check("DELAY", '"DELAY_DAYS" IS NULL OR "DELAY_DAYS" >= 0', "a delay cannot be negative"),
        Check("CAPACITY_FACTOR", '"CAPACITY_FACTOR" IS NULL OR "CAPACITY_FACTOR" BETWEEN 0 AND 1', "a share"),
    )),

    Table("SCENARIO", "scenario.csv", "The scenarios: the normal baseline, the six of backend/config/scenarios.yaml, a Red Sea diversion and the legacy demo start. All SYNTHETIC.", (
        col("SCENARIO_ID", SCENARIO_ID, "Scenario code."),
        col("SCENARIO_NAME", "NVARCHAR(128)", "Display name."),
        col("DESCRIPTION", FREE_TEXT, "What the scenario simulates."),
        col("SCENARIO_TYPE", "NVARCHAR(16)", "BASELINE, DISRUPTION or LEGACY_STATE."),
        col("IS_MODELED", "BOOLEAN", "False if the pipeline records this kind of event but nothing acts on it yet."),
        opt("BASED_ON_EVENT_ID", EVENT_ID, "The historical event the scenario is grounded in, if any (only the Suez closure)."),
        col("SOURCE", NOTE, "Where the definition comes from."),
        _PROV,
    ), ("SCENARIO_ID",), (fk("SCENARIO", "BASED_ON_EVENT_ID", "DISRUPTION_EVENT", "EVENT_ID", "EVENT"),), (
        Check("SCENARIO_TYPE", one_of("SCENARIO_TYPE", SCENARIO_TYPES), "scenario kinds the cleaned layer defines"),
    )),

    Table("SCENARIO_STATE", "scenario_state.csv", "What each scenario does to which lane or supplier. Only deviations from normal are stored: absence of a row means ACTIVE / NORMAL.", (
        col("SCENARIO_ID", SCENARIO_ID, "The scenario."),
        col("ENTITY_TYPE", "NVARCHAR(16)", "ROUTE or SUPPLIER: what ENTITY_ID names."),
        col("ENTITY_ID", ENTITY_ID, "A route id or a supplier id according to ENTITY_TYPE, so it has no foreign key; the validation SQL checks it."),
        col("STATUS", "NVARCHAR(16)", "DISRUPTED, REDUCED or DELAYED."),
        opt("CAPACITY_FACTOR", FRACTION, "Share of nominal capacity left, 0 to 1. A placeholder for REDUCED."),
        col("BASIS", FREE_TEXT, "Why the entity is affected."),
        _PROV,
    ), ("SCENARIO_ID", "ENTITY_TYPE", "ENTITY_ID"), (fk("SCENARIO_STATE", "SCENARIO_ID", "SCENARIO", "SCENARIO_ID", "SCENARIO"),), (
        Check("ENTITY_TYPE", one_of("ENTITY_TYPE", ENTITY_TYPES), "entity types the platform models"),
        Check("STATUS", one_of("STATUS", DEVIATION_STATUSES), "the states an event can put a lane or supplier in"),
        Check("CAPACITY_FACTOR", '"CAPACITY_FACTOR" IS NULL OR "CAPACITY_FACTOR" BETWEEN 0 AND 1', "a share"),
    )),

    Table("SCENARIO_PARAM", "scenario_param.csv", "A scenario's numbers: duration, severity, tariff change per country.", (
        col("SCENARIO_ID", SCENARIO_ID, "The scenario."),
        col("PARAM_KEY", "NVARCHAR(64)", "duration_days, severity or tariff_add_pct."),
        col("QUALIFIER", "NVARCHAR(16)", "ISO-3 country for a per-country parameter, else 'ALL'. Part of the key, so never NULL."),
        opt("PARAM_VALUE", PARAMETER, "Numeric value, in UNIT."),
        opt("PARAM_TEXT", "NVARCHAR(64)", "Text value, for a parameter that is not a number."),
        opt("UNIT", "NVARCHAR(32)", "Unit of PARAM_VALUE."),
        col("BASIS", FREE_TEXT, "Where the number comes from."),
        _PROV,
    ), ("SCENARIO_ID", "PARAM_KEY", "QUALIFIER"), (fk("SCENARIO_PARAM", "SCENARIO_ID", "SCENARIO", "SCENARIO_ID", "SCENARIO"),), (
        Check("VALUE_OR_TEXT", '"PARAM_VALUE" IS NOT NULL OR "PARAM_TEXT" IS NOT NULL', "a parameter has a value"),
    )),

    Table("INVENTORY_POLICY", "inventory_policy.csv", "The parameters the inventory simulation used, per warehouse, so the ledger can be reproduced.", (
        col("WAREHOUSE_ID", WAREHOUSE_ID, "The warehouse."),
        col("DEMAND_SHARE", FRACTION, "Share of each product's demand this warehouse gets; the shares sum to 1."),
        col("COVER_MULTIPLIER", MULTIPLIER, "Multiplies the starting days of cover."),
        col("SAFETY_MULTIPLIER", MULTIPLIER, "Multiplies the safety days."),
        col("TARGET_DAYS_OF_COVER", "INTEGER", "Days of mean demand held at the start."),
        col("SAFETY_DAYS", "INTEGER", "Days of mean demand held as safety stock."),
        col("REPLENISH_DAYS", "INTEGER", "Days of mean demand ordered when stock falls below safety."),
        col("REPLENISHMENT_LEAD_TIME_DAYS", "INTEGER", "Days between ordering and arrival. 0 means a replenishment arrives the day it is triggered."),
        _PROV,
    ), ("WAREHOUSE_ID",), (fk("INVENTORY_POLICY", "WAREHOUSE_ID", "WAREHOUSE", "WAREHOUSE_ID", "WAREHOUSE"),), (
        Check("DEMAND_SHARE", '"DEMAND_SHARE" BETWEEN 0 AND 1', "a share"),
        Check("MULTIPLIERS", '"COVER_MULTIPLIER" > 0 AND "SAFETY_MULTIPLIER" > 0', "multipliers are positive"),
        Check("DAYS", '"TARGET_DAYS_OF_COVER" >= 0 AND "SAFETY_DAYS" >= 0 AND "REPLENISH_DAYS" >= 0 AND "REPLENISHMENT_LEAD_TIME_DAYS" >= 0', "day counts cannot be negative"),
    )),

    Table("INVENTORY", "inventory.csv", "A SIMULATED stock ledger, one row per warehouse, product and day, with lost sales recorded. Not an observation.", (
        col("WAREHOUSE_ID", WAREHOUSE_ID, "The warehouse."),
        col("PRODUCT_ID", PRODUCT_ID, "The product."),
        col("DATE", "DATE", "The day."),
        col("OPENING_STOCK", "INTEGER", "Units on hand at the start of the day."),
        col("INBOUND_QUANTITY", "INTEGER", "Units received."),
        col("OUTBOUND_QUANTITY", "INTEGER", "Units actually shipped."),
        col("CLOSING_STOCK", "INTEGER", "Units on hand at the end: opening + inbound - outbound, exactly."),
        col("SAFETY_STOCK", "INTEGER", "Safety stock for this warehouse and product."),
        col("REORDER_POINT", "INTEGER", "Reorder point; equal to safety stock under instant replenishment."),
        col("DEMAND_QUANTITY", "INTEGER", "Net demand allocated to this warehouse: outbound + unfilled."),
        col("UNFILLED_QUANTITY", "INTEGER", "Demand that could not be shipped: a lost sale, recorded not hidden."),
        col("STOCKOUT_FLAG", "BOOLEAN", "True when UNFILLED_QUANTITY is above zero."),
        _PROV,
    ), ("WAREHOUSE_ID", "PRODUCT_ID", "DATE"), (fk("INVENTORY", "WAREHOUSE_ID", "WAREHOUSE", "WAREHOUSE_ID", "WAREHOUSE"), fk("INVENTORY", "PRODUCT_ID", "PRODUCT", "PRODUCT_ID")), (
        Check("QUANTITIES", '"OPENING_STOCK" >= 0 AND "INBOUND_QUANTITY" >= 0 AND "OUTBOUND_QUANTITY" >= 0 AND "CLOSING_STOCK" >= 0 AND "SAFETY_STOCK" >= 0 '
                            'AND "REORDER_POINT" >= 0 AND "DEMAND_QUANTITY" >= 0 AND "UNFILLED_QUANTITY" >= 0', "stock and flows cannot be negative"),
    )),

    Table("DEMAND", "demand.csv", "Net demand per day, product and country, with cancellations netted out and no features. Derived from the corrected sale lines.", (
        col("DATE", "DATE", "The day."),
        col("PRODUCT_ID", PRODUCT_ID, "The product."),
        col("LOCATION_ID", "NVARCHAR(64)", "The retail country label, as text ('United Kingdom', 'EIRE'). Equivalent to COUNTRY_ISO3 (1:1) but no foreign key: the label column of COUNTRY is neither a key nor unique."),
        col("COUNTRY_ISO3", ISO3, "The customer's country."),
        col("DEMAND_QUANTITY", "INTEGER", "Net units demanded that day: sale lines less the cancellations that reversed them. Always above zero: a day with no net sale has no row."),
        col("N_ORDER_LINES", "INTEGER", "Sale lines behind the number."),
        col("N_OUTLIER_LINES", "INTEGER", "Of those, lines flagged as unusually large (kept, not removed)."),
        _PROV,
    ), ("DATE", "PRODUCT_ID", "LOCATION_ID"), (fk("DEMAND", "PRODUCT_ID", "PRODUCT", "PRODUCT_ID"), fk("DEMAND", "COUNTRY_ISO3", "COUNTRY", "ISO3", "COUNTRY")), (
        Check("QUANTITY", '"DEMAND_QUANTITY" > 0', "a demand row has positive net demand"),
        Check("LINES", '"N_ORDER_LINES" > 0 AND "N_OUTLIER_LINES" >= 0', "counts"),
    )),

    Table("DEMAND_MODELING_PANEL", "demand_modeling_panel.csv", "The 40 forecastable series on a dense daily calendar (a day without a sale is 0), with lag and rolling features and a time split.", (
        col("DATE", "DATE", "The day."),
        col("PRODUCT_ID", PRODUCT_ID, "The product."),
        col("LOCATION_ID", "NVARCHAR(64)", "The retail country label. Always 'United Kingdom' in this panel. No foreign key (see DEMAND.LOCATION_ID), and the panel has no ISO-3 column."),
        col("DEMAND_QUANTITY", "INTEGER", "Net units that day; 0 on a day without a sale."),
        opt("LAG_1", "DOUBLE", "Demand one day earlier. NULL at the start of a series. Whole numbers stored as the float the cleaned file delivers."),
        opt("LAG_2", "DOUBLE", "Demand two days earlier."),
        opt("LAG_7", "DOUBLE", "Demand seven days earlier."),
        opt("LAG_14", "DOUBLE", "Demand fourteen days earlier."),
        opt("LAG_28", "DOUBLE", "Demand twenty-eight days earlier."),
        opt("ROLLING_MEAN_7", "DOUBLE", "Mean of the previous 7 days, current day excluded (no leakage). DOUBLE keeps full precision."),
        opt("ROLLING_MEAN_14", "DOUBLE", "Mean of the previous 14 days."),
        opt("ROLLING_MEAN_28", "DOUBLE", "Mean of the previous 28 days."),
        col("DAY_OF_WEEK", "INTEGER", "0 = Monday to 6 = Sunday."),
        col("MONTH", "INTEGER", "1 to 12."),
        col("DISRUPTION_ACTIVE", "BOOLEAN", "Constant False: no disruption overlaps 2010-2011. Do not use as a feature."),
        col("SPLIT", "NVARCHAR(8)", "train, val or test: a chronological 70/15/15 split."),
        col("HAS_OUTLIER_LINE", "BOOLEAN", "True if an unusually large order line contributes to the day."),
        _PROV,
    ), ("DATE", "PRODUCT_ID", "LOCATION_ID"), (fk("DEMAND_MODELING_PANEL", "PRODUCT_ID", "PRODUCT", "PRODUCT_ID"),), (
        Check("QUANTITY", '"DEMAND_QUANTITY" >= 0', "demand cannot be negative"),
        Check("DAY_OF_WEEK", '"DAY_OF_WEEK" BETWEEN 0 AND 6', "Monday to Sunday"),
        Check("MONTH", '"MONTH" BETWEEN 1 AND 12', "a month"),
        Check("SPLIT", one_of("SPLIT", ("train", "val", "test")), "the split vocabulary"),
    )),
)

# parents before children, in the order the user asked for
LOAD_ORDER: tuple[str, ...] = tuple(t.name for t in TABLES)
BY_NAME: dict[str, Table] = {t.name: t for t in TABLES}
