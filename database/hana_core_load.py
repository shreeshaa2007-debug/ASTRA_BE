"""Reading data/cleaned into the HANA core model: exact typed values, a pre-flight that needs no database, and the loader.

Three things use this module:

* the package generator (`hana_core.py`) runs `preflight` and refuses to write the DDL unless every cleaned value fits the column
  it is declared for;
* `scripts/load_hana_core.py` uses `load_tables` to insert the CSVs into HANA Cloud, in the dependency order, and `run_validation`
  to execute the validation SQL afterwards;
* the tests run the same code against an in-memory SQLite database (`sqlite_connection`), which is a REHEARSAL of the load and of the
  validation SQL, not a substitute for HANA: the DDL and the SQL are written in the portable subset for exactly that reason, and the
  one HANA-only keyword (`COLUMN` in `CREATE COLUMN TABLE`) is translated.

Nothing here modifies a CSV, and nothing connects to a database unless a caller hands it a connection. The values are never coerced
silently: a text longer than its column, a decimal with more places than its scale, an integer out of range or an empty required
field is an ERROR that names the table, the row and the column.
"""
from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from backend.database.hana_core_spec import LOAD_ORDER, TABLES, Column, Table

DEFAULT_DATA_DIR = Path("data/cleaned")
_TYPE = re.compile(r"^(NVARCHAR|DECIMAL)\((\d+)(?:,(\d+))?\)$")
_INT_MIN, _INT_MAX = -(2**31), 2**31 - 1


def parse_type(sql_type: str) -> tuple[str, int, int]:
    """('NVARCHAR', 32, 0), ('DECIMAL', 16, 4), ('INTEGER', 0, 0) ..."""
    m = _TYPE.match(sql_type)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3) or 0)
    return sql_type, 0, 0


def converter(column: Column) -> Callable[[str], Any]:
    """CSV text -> the exact Python value for this column, or a ValueError saying why the value does not fit."""
    base, precision, scale = parse_type(column.type)

    def convert(text: str) -> Any:
        if text == "":
            if not column.nullable:
                raise ValueError("empty, but the column is NOT NULL")
            return None
        if base == "NVARCHAR":
            if len(text) > precision:
                raise ValueError(f"{len(text)} characters do not fit NVARCHAR({precision})")
            return text
        if base == "INTEGER":
            value = int(text)
            if not _INT_MIN <= value <= _INT_MAX:
                raise ValueError(f"{value} is outside INTEGER")
            return value
        if base == "DECIMAL":
            value = Decimal(text)
            exponent = value.as_tuple().exponent
            decimals = max(0, -exponent) if isinstance(exponent, int) else 0
            if decimals > scale:
                raise ValueError(f"{text} has {decimals} decimals; DECIMAL({precision},{scale}) would round it")
            if len(str(abs(int(value)))) > precision - scale and abs(value) >= 1:
                raise ValueError(f"{text} has too many integer digits for DECIMAL({precision},{scale})")
            return value
        if base == "DOUBLE":
            return float(text)
        if base == "BOOLEAN":
            if text not in ("True", "False"):
                raise ValueError(f"{text!r} is not True or False")
            return text == "True"
        if base == "DATE":
            return date.fromisoformat(text)
        if base == "TIMESTAMP":
            return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        raise ValueError(f"no converter for {column.type}")

    return convert


def expected_counts(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, int]:
    """Row counts the cleaned layer's own manifest recorded, keyed by HANA table name."""
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))["tables"]
    return {t.name: manifest[t.csv.removesuffix(".csv")]["rows"] for t in TABLES}


def iter_rows(table: Table, data_dir: Path = DEFAULT_DATA_DIR) -> Iterator[tuple[int, tuple[Any, ...]]]:
    """(csv line number, converted values in the table's column order). Raises ValueError, naming table, line and column, on a bad value."""
    convert = [(c.name, converter(c)) for c in table.columns]
    with open(data_dir / table.csv, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if [h.upper() for h in header] != [c.name for c in table.columns]:
            raise ValueError(f"{table.name}: {table.csv} has columns {header}, the table declares {[c.name for c in table.columns]}")
        for line, record in enumerate(reader, start=2):
            if len(record) != len(convert):
                raise ValueError(f"{table.name} line {line}: {len(record)} fields, expected {len(convert)}")
            values = []
            for (name, fn), text in zip(convert, record):
                try:
                    values.append(fn(text))
                except (ValueError, ArithmeticError) as exc:
                    raise ValueError(f"{table.name} line {line} column {name}: {exc}") from None
            yield line, tuple(values)


@dataclass
class Preflight:
    """What is true of the cleaned CSVs against the declared model, worked out with no database."""
    rows: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    totals: dict[str, int] = field(default_factory=dict)  # control totals, computed from the CSVs

    @property
    def ok(self) -> bool:
        return not self.errors


def preflight(data_dir: Path = DEFAULT_DATA_DIR, tables: tuple[Table, ...] = TABLES, max_errors: int = 25) -> Preflight:
    """Every value fits its column, every primary key is present and unique, every foreign key resolves, every row count matches the
    manifest. All in memory: parents' keys are collected as they are read, in load order."""
    result = Preflight()
    keys: dict[str, set[tuple]] = {}
    expected = expected_counts(data_dir)
    for table in tables:
        index = {c.name: i for i, c in enumerate(table.columns)}
        pk = [index[c] for c in table.primary_key]
        fks = [(f, [index[c] for c in f.columns]) for f in table.foreign_keys]
        seen: set[tuple] = set()
        count = 0
        try:
            for line, values in iter_rows(table, data_dir):
                count += 1
                key = tuple(values[i] for i in pk)
                if key in seen:
                    result.errors.append(f"{table.name} line {line}: duplicate primary key {key}")
                seen.add(key)
                for f, cols in fks:
                    ref = tuple(values[i] for i in cols)
                    if None not in ref and ref not in keys.get(f.parent, set()):
                        result.errors.append(f"{table.name} line {line}: {f.name} -> {f.parent} has no parent {ref}")
                if len(result.errors) >= max_errors:
                    break
        except ValueError as exc:
            result.errors.append(str(exc))
        keys[table.name] = {tuple(k) for k in seen}
        result.rows[table.name] = count
        if count != expected[table.name] and len(result.errors) < max_errors:
            result.errors.append(f"{table.name}: {count} rows read, the cleaned manifest says {expected[table.name]}")
    return result


# --------------------------------------------------------------------------- SQL text
def quote(identifier: str) -> str:
    return f'"{identifier}"'


def insert_sql(table: Table) -> str:
    columns = ", ".join(quote(c.name) for c in table.columns)
    return f"INSERT INTO {quote(table.name)} ({columns}) VALUES ({', '.join('?' for _ in table.columns)})"  # qmark: hdbcli and sqlite3 agree


def split_statements(sql: str) -> list[str]:
    """Statements of a SQL file: split on ';' outside quotes, with -- comments dropped. Enough for the files of this package."""
    statements, current, in_quote = [], [], False
    for raw in sql.splitlines():
        line = raw
        if not in_quote and line.lstrip().startswith("--"):
            continue
        for char in line:
            if char == "'":
                in_quote = not in_quote
            if char == ";" and not in_quote:
                statement = "".join(current).strip()
                if statement:
                    statements.append(statement)
                current = []
            else:
                current.append(char)
        current.append("\n")
    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


# --------------------------------------------------------------------------- loading
def load_tables(connection: Any, data_dir: Path = DEFAULT_DATA_DIR, tables: tuple[Table, ...] = TABLES, batch_size: int = 5000,
                echo: Callable[[str], None] = print) -> dict[str, int]:
    """Inserts each CSV, parents first, committing after each table. A failure rolls back the table in progress and stops: tables
    already committed stay, and the message says which. `connection` is any DB-API connection with qmark parameters."""
    counts: dict[str, int] = {}
    cursor = connection.cursor()
    for table in tables:
        sql, batch, n = insert_sql(table), [], 0
        try:
            for _line, values in iter_rows(table, data_dir):
                batch.append(values)
                if len(batch) >= batch_size:
                    cursor.executemany(sql, batch)
                    n += len(batch)
                    batch = []
                    if n % 50_000 == 0:  # DEMAND is 300k rows over a network: say it is still going
                        echo(f"    ... {n:,} rows of {table.name}")
            if batch:
                cursor.executemany(sql, batch)
                n += len(batch)
            connection.commit()
        except Exception as exc:
            connection.rollback()
            raise RuntimeError(f"loading {table.name} failed after {n:,} rows (rolled back; earlier tables stay loaded): {exc}") from exc
        counts[table.name] = n
        echo(f"  loaded {n:>8,} rows into {table.name}")
    return counts


def table_counts(connection: Any, tables: tuple[Table, ...] = TABLES) -> dict[str, int]:
    cursor = connection.cursor()
    out = {}
    for t in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {quote(t.name)}")
        out[t.name] = int(cursor.fetchone()[0])
    return out


def empty_tables(connection: Any) -> None:
    """DELETE every row of the 17 tables, children first. Destructive; only ever called on request."""
    cursor = connection.cursor()
    for name in reversed(LOAD_ORDER):
        cursor.execute(f"DELETE FROM {quote(name)}")
    connection.commit()


def run_validation(connection: Any, sql: str) -> list[dict[str, Any]]:
    """Runs a validation script (each statement is one scorecard) and returns every row as a dict."""
    cursor = connection.cursor()
    rows: list[dict[str, Any]] = []
    for statement in split_statements(sql):
        cursor.execute(statement)
        names = [d[0].upper() for d in cursor.description]
        rows += [dict(zip(names, record)) for record in cursor.fetchall()]
    return rows


# --------------------------------------------------------------------------- the target database: look before writing, verify after
def ensure_transactional(connection: Any) -> None:
    """hdbcli connections start in autocommit mode, where the rollback that keeps a failed table from being half-loaded would do nothing.
    sqlalchemy-hana already turns autocommit off on every connection it makes; this keeps the loader safe on a bare hdbcli one too."""
    if hasattr(connection, "getautocommit") and connection.getautocommit():
        connection.setautocommit(False)


@dataclass
class Target:
    """What the schema the load would write to holds. Obtained with SELECTs only."""
    user: str
    schema: str
    columns: dict[str, int]  # model table -> its number of columns, for the model's tables that exist there
    counts: dict[str, int]  # the same tables -> rows


def inspect_target(connection: Any, schema: str, dialect: str = "hana", tables: tuple[Table, ...] = TABLES) -> Target:
    """Which of the model's tables exist in `schema`, how many columns and rows each has. Read-only. The loader's statements are
    unqualified, so on HANA this refuses to go on unless the session is really in `schema`: counts and INSERTs would land elsewhere."""
    cursor = connection.cursor()
    wanted = {t.name for t in tables}
    if dialect == "hana":
        cursor.execute("SELECT CURRENT_USER, CURRENT_SCHEMA FROM DUMMY")
        user, current = cursor.fetchone()
        if current != schema:
            raise RuntimeError(f"the session schema is {current}, not {schema}: nothing was counted or written")
        cursor.execute("SELECT TABLE_NAME, COUNT(*) FROM SYS.TABLE_COLUMNS WHERE SCHEMA_NAME = ? GROUP BY TABLE_NAME", (schema,))
        columns = {name: int(n) for name, n in cursor.fetchall() if name in wanted}
    else:  # the SQLite rehearsal
        user, columns = "sqlite", {}
        for t in tables:
            cursor.execute(f"PRAGMA table_info({quote(t.name)})")
            if n := len(cursor.fetchall()):
                columns[t.name] = n
    counts = {}
    for t in tables:
        if t.name in columns:
            cursor.execute(f"SELECT COUNT(*) FROM {quote(t.name)}")
            counts[t.name] = int(cursor.fetchone()[0])
    return Target(user, schema, columns, counts)


@dataclass
class LoadPlan:
    to_load: list[str] = field(default_factory=list)  # empty tables, in load order
    skip: list[str] = field(default_factory=list)  # already complete: only with resume
    problems: list[str] = field(default_factory=list)  # anything here means: write nothing

    @property
    def ok(self) -> bool:
        return not self.problems


def plan_load(target: Target, expected: dict[str, int], resume: bool = False, tables: tuple[Table, ...] = TABLES) -> LoadPlan:
    """What the load may do, decided without writing. Every table must exist with the model's number of columns and be EMPTY. With `resume`,
    a table that already holds exactly its expected rows (an earlier run committed it) is left alone; any other non-empty table is a
    problem. Nothing here ever deletes: a table that is not as expected stops the load and is left exactly as it is."""
    plan = LoadPlan()
    for t in tables:
        if t.name not in target.columns:
            plan.problems.append(f"{t.name} does not exist in {target.schema}: create the 17 tables first (HANA_PREFLIGHT_{target.schema}.sql)")
        elif target.columns[t.name] != len(t.columns):
            plan.problems.append(f"{t.name} has {target.columns[t.name]} columns in {target.schema}, the model declares {len(t.columns)}: it is not this model's table")
        elif (rows := target.counts[t.name]) == 0:
            plan.to_load.append(t.name)
        elif resume and rows == expected[t.name]:
            plan.skip.append(t.name)
        elif resume:
            plan.problems.append(f"{t.name} holds {rows:,} rows, neither 0 nor the {expected[t.name]:,} of a complete load: someone has to look at it, it was not touched")
        else:
            plan.problems.append(f"{t.name} already holds {rows:,} rows: refusing to load into a table that is not empty (--resume continues after a partial load)")
    return plan


def verify_counts(connection: Any, expected: dict[str, int], tables: tuple[Table, ...] = TABLES) -> list[tuple[str, int, int, str]]:
    """(table, expected, actual, PASS|FAIL), the actual read back from the database with COUNT(*), not taken from what was inserted."""
    actual = table_counts(connection, tables)
    return [(t.name, expected[t.name], actual[t.name], "PASS" if actual[t.name] == expected[t.name] else "FAIL") for t in tables]


def run_load(connection: Any, data_dir: Path, expected: dict[str, int], schema: str, dialect: str = "hana", resume: bool = False,
             read_only: bool = False, echo: Callable[[str], None] = print, tables: tuple[Table, ...] = TABLES) -> int:
    """Look, decide, load, verify. Returns 0 (loaded and every count verified; or, read-only, the target is ready), 1 (a load or a count
    failed), 2 (refused before writing anything). The caller has already pointed the session at `schema` and creates nothing here."""
    ensure_transactional(connection)
    try:
        target = inspect_target(connection, schema, dialect, tables)
    except RuntimeError as exc:
        echo(str(exc))
        return 2
    echo(f"connected as {target.user}, schema {target.schema}: {len(target.columns)} of the {len(tables)} model tables exist, {sum(target.counts.values()):,} rows in them")
    plan = plan_load(target, expected, resume, tables)
    if not plan.ok:
        echo("refusing to write anything:\n  " + "\n  ".join(plan.problems))
        return 2
    if plan.skip:
        echo(f"--resume: {len(plan.skip)} tables already hold exactly their expected rows and are left untouched (their values are not re-read: run 03_validate.sql): {', '.join(plan.skip)}")
    echo(f"the load would write {sum(expected[n] for n in plan.to_load):,} rows into {len(plan.to_load)} empty tables, in this order: {', '.join(plan.to_load)}" if read_only
         else f"loading {sum(expected[n] for n in plan.to_load):,} rows into {len(plan.to_load)} empty tables, in this order: {', '.join(plan.to_load)}")
    if read_only:
        echo("read-only check: nothing was written.")
        return 0
    try:
        load_tables(connection, data_dir, tuple(t for t in tables if t.name in plan.to_load), echo=echo)
    except RuntimeError as exc:
        echo(str(exc))
        return 1
    rows = verify_counts(connection, expected, tables)
    echo(f"row counts read back from {schema}:\n  {'TABLE':24s} {'EXPECTED':>9s} {'ACTUAL':>9s}")
    for name, want, got, outcome in rows:
        echo(f"  {name:24s} {want:>9,} {got:>9,}  {outcome}")
    failed = [name for name, _, _, outcome in rows if outcome != "PASS"]
    echo("every row count matches." if not failed else f"row counts differ in: {', '.join(failed)}")
    return 1 if failed else 0


def explain_connection_error(exc: Exception, secret: str | None = None) -> str:
    """One line for the error plus what to look at. `secret` (the password) is scrubbed from the text, whatever the driver put in it."""
    first = str(exc).splitlines()[0][:300] if str(exc) else ""
    text = f"{type(exc).__name__}: {first}"
    if secret:
        text = text.replace(secret, "***")
    low = str(exc).lower()
    if isinstance(exc, ModuleNotFoundError) or "no module named" in low or "can't load plugin" in low:
        hint = "the SAP drivers are not installed in this Python: py -3.12 -m pip install -r requirements-sap.txt"
    elif "authentication" in low or "invalid user" in low or "[10]" in low:
        hint = "HANA rejected the user name or password"
    elif "certificate" in low or "ssl" in low or "tls" in low:
        hint = "the TLS handshake failed: HANA Cloud accepts only encrypted connections, so keep encrypt=true"
    else:
        hint = "check the host and port (HANA Cloud listens on 443), that the instance is running, and that this machine's IP is in the instance's allowed connections"
    return f"{text}\n  {hint}"


# --------------------------------------------------------------------------- the local rehearsal
def sqlite_ddl(hana_ddl: str) -> str:
    """The HANA DDL as SQLite will take it: the only HANA-specific word is COLUMN. Types, keys, foreign keys and CHECKs are portable."""
    return hana_ddl.replace("CREATE COLUMN TABLE", "CREATE TABLE")


def sqlite_connection(ddl: str | None = None) -> sqlite3.Connection:
    """An in-memory SQLite database with foreign keys enforced and explicit adapters for the typed values the loader passes."""
    sqlite3.register_adapter(Decimal, str)  # exact text; the NUMERIC column then stores it
    sqlite3.register_adapter(date, lambda d: d.isoformat())
    sqlite3.register_adapter(datetime, lambda d: d.strftime("%Y-%m-%d %H:%M:%S"))
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON")
    if ddl:
        for statement in split_statements(sqlite_ddl(ddl)):
            connection.execute(statement)
        connection.commit()
    return connection
