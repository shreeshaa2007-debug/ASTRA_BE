"""Where the reference datasets come from.

`DatasetRepository` is the seam. `CsvDatasetRepository` reads the processed CSV files (the default, and what every
test and the demo use); `SqlDatasetRepository` reads the `ref_*` tables of any SQLAlchemy database — SAP HANA Cloud on
BTP, or SQLite in a test. Both return the same columns (see `datasets.py`) with the same dtypes, so an agent cannot
tell them apart; `test_datasets.py` proves the plans they lead to are identical.

Configuration (environment):

    DATA_BACKEND         csv (default) | sql
    DATA_DIR             csv: the directory holding the files             (default data/processed)
    DATA_DATABASE_URL    sql: which database                              (default: the world-state database)
    DATA_SCHEMA          sql: the schema holding the ref_* tables         (default: the world-state schema)
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence, TypeVar

import numpy as np
import pandas as pd
from sqlalchemy import String, inspect, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, ProgrammingError

from backend.data.datasets import DATASETS, DatasetSpec

DEFAULT_DATA_DIR = "data/processed"


class DatasetUnavailableError(FileNotFoundError):
    """The dataset is not there: a missing file, or a missing table. A FileNotFoundError so every place that already
    turns one into a 503 DATASET_UNAVAILABLE keeps doing so."""


class DatasetRepository(Protocol):
    def load(self, name: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
        """The dataset as a DataFrame — all of its columns, or just `columns`, in the order asked for."""

    def missing(self) -> list[str]:
        """What is absent, in words an operator can act on; empty when every dataset is present."""

    def describe(self) -> str:
        """Where the data comes from, safe to log."""


def _spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}") from None


def _columns(spec: DatasetSpec, columns: Sequence[str] | None) -> list[str]:
    if columns is None:
        return list(spec.column_names)
    unknown = [c for c in columns if c not in spec.column_names]
    if unknown:  # the contract is the same on every backend, so a column that only one of them happens to have is an error
        raise KeyError(f"dataset {spec.name!r} has no column(s) {unknown}; it has {list(spec.column_names)}")
    return list(columns)


class CsvDatasetRepository:
    def __init__(self, root: str | Path = DEFAULT_DATA_DIR):
        self._root = Path(root)

    def _path(self, spec: DatasetSpec) -> Path:
        return self._root / spec.csv

    def load(self, name: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
        spec = _spec(name)
        cols = _columns(spec, columns)
        path = self._path(spec)
        if not path.exists():
            raise DatasetUnavailableError(f"dataset {name!r}: {path} does not exist")
        # text columns are read as text: an identifier that looks like a number stays an identifier, and a column with no
        # values is an object column of missing values on every backend, not a float one on this
        text = {name: str for name, type_ in spec.columns if isinstance(type_, String) and name in cols}
        df = pd.read_csv(path, usecols=cols, dtype=text, parse_dates=[c for c in spec.dates if c in cols] or False)
        return df[cols]

    def missing(self) -> list[str]:
        return [str(self._path(spec)) for spec in DATASETS.values() if not self._path(spec).exists()]

    def describe(self) -> str:
        return f"csv:{self._root.as_posix()}"


class SqlDatasetRepository:
    def __init__(self, engine: Engine, *, schema: str | None = None, description: str = "sql"):
        """`engine` should already carry the schema (`build_engine` does that with a schema translate map); `schema`
        is only for asking the database whether a table exists, which bypasses that map."""
        self._engine = engine
        self._schema = schema
        self._description = description

    def load(self, name: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
        spec = _spec(name)
        cols = _columns(spec, columns)
        table = spec.table()
        statement = select(*(table.c[c] for c in cols)).order_by(*(table.c[c] for c in spec.order_by))
        try:
            with self._engine.connect() as conn:
                df = pd.read_sql(statement, conn)
        except (ProgrammingError, OperationalError) as exc:
            # "the table is not there" and "the database is not there" are different problems with different fixes
            if not self._has_table(spec):
                raise DatasetUnavailableError(f"dataset {name!r}: table {spec.table_name} does not exist"
                                              f"{f' in schema {self._schema}' if self._schema else ''}") from exc
            raise
        for name, type_ in spec.columns:
            if name in df.columns and isinstance(type_, String):
                df[name] = df[name].astype(object).where(df[name].notna(), np.nan)  # NULL is NaN, as a CSV's empty cell is
        for column in spec.dates:
            if column in df.columns:
                df[column] = pd.to_datetime(df[column])
        return df[cols]

    def _has_table(self, spec: DatasetSpec) -> bool:
        return inspect(self._engine).has_table(spec.table_name, schema=self._schema)

    def missing(self) -> list[str]:
        inspector = inspect(self._engine)
        return [f"table {spec.table_name}" for spec in DATASETS.values() if not inspector.has_table(spec.table_name, schema=self._schema)]

    def describe(self) -> str:
        return self._description


# --------------------------------------------------------------------------- #
# the process-wide repository
# --------------------------------------------------------------------------- #
_current: DatasetRepository | None = None
_lock = threading.Lock()
_cached: list = []  # every cache that holds something derived from a dataset


def cached_dataset_loader(maxsize: int | None = 1) -> Callable:
    """`lru_cache` for a function whose result comes from the datasets. Registered, so switching the repository
    (`set_datasets`) empties it — a cache that outlived a backend switch would answer from the old data."""
    def decorate(fn):
        wrapped = lru_cache(maxsize=maxsize)(fn)
        _cached.append(wrapped)
        return wrapped
    return decorate


def clear_dataset_caches() -> None:
    for wrapped in _cached:
        wrapped.cache_clear()


def datasets_from_env(env: Mapping[str, str] | None = None) -> DatasetRepository:
    env = os.environ if env is None else env
    backend = (env.get("DATA_BACKEND") or "csv").strip().lower()
    if backend == "csv":
        return CsvDatasetRepository(env.get("DATA_DIR") or DEFAULT_DATA_DIR)
    if backend == "sql":
        from backend.database.engine import DatabaseTarget, build_engine, resolve_database_target
        base = resolve_database_target(env=env)
        target = DatabaseTarget(env.get("DATA_DATABASE_URL") or base.url, env.get("DATA_SCHEMA") or base.schema,
                                "DATA_DATABASE_URL" if env.get("DATA_DATABASE_URL") else base.source)
        return SqlDatasetRepository(build_engine(target), schema=target.schema, description=f"sql:{target.describe()}")
    raise ValueError(f"DATA_BACKEND must be 'csv' or 'sql', not {backend!r}")


def get_datasets() -> DatasetRepository:
    """The repository the agents read from, built from the environment on first use."""
    global _current
    if _current is None:
        with _lock:
            if _current is None:
                _current = datasets_from_env()
    return _current


def set_datasets(repository: DatasetRepository | None) -> None:
    """Install a repository (None: rebuild from the environment on next use), and drop everything cached from the old one."""
    global _current
    with _lock:
        _current = repository
    clear_dataset_caches()
