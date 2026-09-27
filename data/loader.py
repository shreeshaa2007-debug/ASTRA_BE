"""Load the processed CSV datasets into a database: the way SAP HANA Cloud (or any SQLAlchemy database) gets the
`ref_*` tables the `SqlDatasetRepository` reads.

Idempotent by design: each dataset is replaced in one transaction (delete, then insert), so a load that fails half
way leaves the previous contents, and running it twice leaves the same rows as running it once. It deletes rather than
drops, so it works on tables somebody else created — an HDI container deployed from a design-time artifact — and it
creates a table only when it does not exist (`create=True`), never alters one.
"""
from __future__ import annotations

import logging
from typing import Iterable, Iterator, Sequence

import pandas as pd
from sqlalchemy import DateTime, String
from sqlalchemy.engine import Engine

from backend.data.datasets import DATASETS, DatasetSpec
from backend.data.repository import CsvDatasetRepository

logger = logging.getLogger("resilientsc.data")

DEFAULT_CHUNK_ROWS = 5000


def _records(df: pd.DataFrame, spec: DatasetSpec) -> Iterator[dict]:
    """Rows as plain Python values: what a DB-API driver accepts, with NaN/NaT as NULL."""
    frame = df.copy()
    for name, type_ in spec.columns:
        if isinstance(type_, String):
            frame[name] = frame[name].map(lambda v: None if pd.isna(v) else str(v))
        elif name in spec.dates:
            as_dt = pd.to_datetime(frame[name])
            frame[name] = [None if pd.isna(v) else (v.to_pydatetime() if isinstance(type_, DateTime) else v.date()) for v in as_dt]
    frame = frame.astype(object).where(frame.notna(), None)
    for row in frame.itertuples(index=False, name=None):
        yield dict(zip(frame.columns, row))


def _check_lengths(df: pd.DataFrame, spec: DatasetSpec) -> None:
    """A too-long string is an error, not a silent truncation: databases differ in what they do with one."""
    for name, type_ in spec.columns:
        if isinstance(type_, String) and type_.length:
            lengths = df[name].dropna().astype(str).str.len()  # empty for a column that is entirely NULL
            longest = int(lengths.max()) if len(lengths) else 0
            if longest > type_.length:
                raise ValueError(f"dataset {spec.name!r}: column {name!r} holds a value of {longest} characters; the table allows {type_.length}")


def load_csvs_into(
    engine: Engine, data_dir: str = "data/processed", *, schema: str | None = None,
    names: Iterable[str] | None = None, create: bool = True, chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> dict[str, int]:
    """Replace each named dataset's table with the rows of its CSV; returns rows loaded per dataset. `engine` is used
    as given — pass one built with the schema (backend.database.engine.build_engine) or use `schema` here."""
    if schema:
        engine = engine.execution_options(schema_translate_map={None: schema})
    source = CsvDatasetRepository(data_dir)
    selected: Sequence[str] = list(names) if names is not None else list(DATASETS)
    unknown = [n for n in selected if n not in DATASETS]
    if unknown:
        raise KeyError(f"unknown dataset(s) {unknown}; known: {sorted(DATASETS)}")

    loaded: dict[str, int] = {}
    for name in selected:
        spec = DATASETS[name]
        df = source.load(name)  # exactly the dataset's columns; the CSV's extra feature columns stay behind
        _check_lengths(df, spec)
        table = spec.table()
        with engine.begin() as conn:
            if create:
                table.create(conn, checkfirst=True)
            conn.execute(table.delete())
            batch: list[dict] = []
            for record in _records(df, spec):
                batch.append(record)
                if len(batch) >= chunk_rows:
                    conn.execute(table.insert(), batch)
                    batch.clear()
            if batch:
                conn.execute(table.insert(), batch)
        loaded[name] = len(df)
        logger.info("loaded %d rows into %s", len(df), spec.table_name, extra={"event": "reference_data_loaded", "dataset": name, "rows": len(df)})
    return loaded
