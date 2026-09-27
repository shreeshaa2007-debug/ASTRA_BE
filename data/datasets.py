"""The reference datasets the agents read, and the table each one is when it lives in a database.

This module is the contract between the application and whatever holds its reference data. Today that is
`data/processed/*.csv`; on SAP it is tables — or views over S/4HANA — in SAP HANA Cloud. Either way the agents ask
for a dataset by *name* and get a DataFrame with exactly these columns. A customer's system is connected by making
`ref_suppliers`, `ref_routes`, ... return these columns, not by editing an agent.

`product_id` is a string everywhere: the CSVs hold it as an integer, the readers have always cast it to text, and
a database key is not an integer to arithmetic on.
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import TIMESTAMP, Boolean, Column, Date, Double, Integer, MetaData, String, Table
from sqlalchemy.types import TypeEngine

# Column types are ones every dialect renders as a real type: `TIMESTAMP`, not the generic `DateTime`, which the HANA
# dialect renders as DATETIME — a type HANA does not have.

# Separate from the world-state metadata on purpose: a deployment can create the state tables and leave these
# to whoever owns the master data (they may be views), and the loader can (re)create them without touching state.
metadata = MetaData()

TABLE_PREFIX = "ref_"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    csv: str  # file name under the data directory
    columns: tuple[tuple[str, TypeEngine], ...]
    dates: tuple[str, ...] = ()  # columns returned as datetime64 by every backend
    order_by: tuple[str, ...] = ()  # a database has no row order; this is the one it is asked for

    @property
    def table_name(self) -> str:
        return f"{TABLE_PREFIX}{self.name}"

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.columns)

    def table(self) -> Table:
        return TABLES[self.name]


def _spec(name: str, csv: str, columns: list[tuple[str, TypeEngine]], *, dates: tuple[str, ...] = (), order_by: tuple[str, ...] = ()) -> DatasetSpec:
    return DatasetSpec(name, csv, tuple(columns), dates, order_by)


DATASETS: dict[str, DatasetSpec] = {s.name: s for s in (
    _spec("suppliers", "suppliers.csv", [
        ("supplier_id", String(32)), ("supplier_name", String(128)), ("region", String(64)), ("product_id", String(32)),
        ("capacity", Integer()), ("unit_cost", Double()), ("lead_time_days", Integer()), ("reliability", Double()),
        ("risk_level", String(16)), ("status", String(16)), ("provenance", String(32)),
    ], order_by=("supplier_id", "product_id")),
    _spec("routes", "routes.csv", [
        ("route_id", String(64)), ("origin", String(64)), ("destination", String(64)), ("transport_mode", String(16)),
        ("distance_km", Double()), ("capacity", Integer()), ("transit_time_days", Double()), ("cost_per_unit", Double()),
        ("status", String(16)), ("provenance", String(32)),
    ], order_by=("route_id",)),
    _spec("tariffs", "tariffs.csv", [
        ("origin_country", String(8)), ("destination_country", String(8)), ("product_category", String(32)),
        ("effective_year", Integer()), ("tariff_rate", Double()), ("is_country_level_proxy", Boolean()),
    ], order_by=("origin_country", "destination_country", "product_category", "effective_year")),
    _spec("inventory", "inventory_multi_warehouse.csv", [
        ("warehouse_id", String(64)), ("product_id", String(32)), ("date", Date()), ("opening_stock", Integer()),
        ("inbound_quantity", Integer()), ("outbound_quantity", Integer()), ("closing_stock", Integer()),
        ("safety_stock", Integer()), ("stockout_flag", Boolean()), ("provenance", String(32)),
    ], dates=("date",), order_by=("warehouse_id", "product_id", "date")),
    # only the columns the application reads: the CSV also carries the model's lag/rolling features, which are
    # training data and stay in the ML pipeline
    _spec("demand_panel", "demand_modeling_panel.csv", [
        ("date", Date()), ("product_id", String(32)), ("location_id", String(64)), ("demand_quantity", Integer()),
        ("rolling_mean_28", Double()), ("split", String(16)),
    ], dates=("date",), order_by=("product_id", "location_id", "date")),
    _spec("disruptions", "disruptions.csv", [
        ("event_id", String(64)), ("event_type", String(64)), ("location", String(255)), ("start_date", TIMESTAMP()),
        ("end_date", TIMESTAMP()), ("severity", String(16)), ("affected_route", String(64)), ("affected_supplier", String(64)),
        ("estimated_delay_days", Double()), ("status", String(16)),
    ], dates=("start_date", "end_date"), order_by=("start_date", "event_id")),
)}

TABLES: dict[str, Table] = {
    spec.name: Table(spec.table_name, metadata, *(Column(name, type_) for name, type_ in spec.columns))
    for spec in DATASETS.values()
}
