"""Writing the cleaned tables and reading them back the way a loader should: identifiers as text, integer keys as integers.

An identifier column that happens to hold only digits (the three supplier products, say) is read back by pandas as a number while
the same column elsewhere is text ('85123A'), and a join between the two silently matches nothing. Naming the identifier columns
once here means every reader, including the tests, types them the same way.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.services.preprocessing.cleaned import CLEANED_DIR
from backend.services.preprocessing.cleaned.tables import SPECS

TEXT_COLUMNS = frozenset({
    "product_id", "order_id", "supplier_id", "warehouse_id", "location_id", "route_id", "event_id", "entity_id", "scenario_id", "based_on_event_id",
    "iso3", "iso2", "origin_iso3", "dest_iso3", "country_iso3", "hs_section", "param_key", "qualifier", "entity_type", "cancel_order_id", "sale_order_id",
})
INTEGER_COLUMNS = frozenset({"port_id", "origin_port_id", "destination_port_id", "nearest_port_id", "line_no", "customer_id", "source_row", "effective_year"})


def format_for_csv(frame: pd.DataFrame) -> pd.DataFrame:
    """Dates as YYYY-MM-DD, timestamps (columns ending _ts) as YYYY-MM-DD HH:MM:SS; everything else as it is. Nothing is rounded."""
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].dt.strftime("%Y-%m-%d %H:%M:%S" if column.endswith("_ts") else "%Y-%m-%d")
    return out


def write_table(name: str, frame: pd.DataFrame, directory: Path = CLEANED_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    format_for_csv(frame).to_csv(path, index=False, lineterminator="\n", encoding="utf-8")
    return path


def read_table(name: str, directory: Path = CLEANED_DIR) -> pd.DataFrame:
    path = directory / f"{name}.csv"
    header = pd.read_csv(path, nrows=0).columns
    dtype = {c: "string" for c in header if c in TEXT_COLUMNS} | {c: "Int64" for c in header if c in INTEGER_COLUMNS}
    frame = pd.read_csv(path, dtype=dtype, keep_default_na=True, na_values=[""], low_memory=False)
    for c in [c for c in header if c in TEXT_COLUMNS]:
        frame[c] = frame[c].astype(object).where(frame[c].notna(), None)  # plain str, so set()/merge behave like the built frames
    return frame


def read_layer(directory: Path = CLEANED_DIR) -> dict[str, pd.DataFrame]:
    return {name: read_table(name, directory) for name in SPECS}
