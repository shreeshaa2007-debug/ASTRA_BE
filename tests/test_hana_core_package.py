"""The SAP HANA implementation package (db/hana/core): the model, the DDL, the validation SQL, the dictionary and the loader.

Nothing here touches SAP HANA. What is tested is everything that can be tested without it:

* the specification against the cleaned data (every value fits its column; keys unique; foreign keys resolve), and against the
  requirements: tariff_rate_pct is DOUBLE, capacity is `capacity` with no time unit, provenance allows all four labels, no
  foreign key exists that the data does not support;
* the DDL and the validation SQL run against an in-memory SQLite database (`hana_core_load.sqlite_connection`). SQLite is a
  rehearsal of the logic, not HANA: the SQL is in the portable subset, `CREATE COLUMN TABLE` is translated and the three
  HANA-specific functions of the validation file have SQLite equivalents behind a switch;
* mutations. The constraints must REJECT bad rows, and, on a copy of the tables with no constraints at all, the validation SQL
  must FIND them. A check that has never been seen failing proves nothing.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

from backend.database import hana_core, hana_core_load as load
from backend.database.hana_core_spec import BY_NAME, LOAD_ORDER, PROVENANCE_VALUES, TABLES
from backend.services.preprocessing.cleaned.tables import SPECS as CLEANED_SPECS

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "cleaned"
PACKAGE_DIR = ROOT / "db" / "hana" / "core"
BUILT = (DATA_DIR / "manifest.json").exists()
needs_data = pytest.mark.skipif(not BUILT, reason="build the cleaned layer first: python -m backend.services.preprocessing.cleaned.build")

REQUESTED_ORDER = ["COUNTRY", "PORT", "PRODUCT", "WAREHOUSE", "SUPPLIER", "SUPPLIER_PRODUCT", "ROUTE", "TARIFF", "DISRUPTION_EVENT", "DISRUPTION_IMPACT",
                   "SCENARIO", "SCENARIO_STATE", "SCENARIO_PARAM", "INVENTORY_POLICY", "INVENTORY", "DEMAND", "DEMAND_MODELING_PANEL"]


# --------------------------------------------------------------------------- the model, against the requirements
def test_the_seventeen_tables_are_the_requested_ones_in_the_requested_order():
    assert list(LOAD_ORDER) == REQUESTED_ORDER


def test_every_requirement_on_types_and_names_holds():
    hana_core.assert_model_is_sound()
    tariff = BY_NAME["TARIFF"]
    assert tariff.column("TARIFF_RATE_PCT").type == "DOUBLE"  # required: no value may be rounded
    for t in TABLES:
        for c in t.columns:
            assert "PER_WEEK" not in c.name and "PER_DAY" not in c.name and "PER_MONTH" not in c.name, f"{t.name}.{c.name} invents a time unit"
    for name in ("SUPPLIER_PRODUCT", "ROUTE"):
        capacity = BY_NAME[name].column("CAPACITY")
        assert capacity.type == "INTEGER"
        assert not any(unit in capacity.description.lower() for unit in ("per week", "per day", "per month", "teu", "weekly", "daily")), name
    assert "no time" in BY_NAME["SUPPLIER_PRODUCT"].column("CAPACITY").description.lower()


def test_every_table_lets_provenance_be_all_four_labels_and_nothing_else():
    ddl = hana_core.render_create_sql()
    allowed = "IN ('REAL', 'DERIVED', 'SYNTHETIC', 'AUTHORED')"
    assert PROVENANCE_VALUES == ("REAL", "DERIVED", "SYNTHETIC", "AUTHORED")
    for t in TABLES:
        assert f'CONSTRAINT "CK_{t.name}_PROVENANCE" CHECK ("PROVENANCE" {allowed})' in ddl, t.name
        assert not t.column("PROVENANCE").nullable


def test_primary_and_foreign_keys_are_exactly_the_ones_the_cleaned_layer_declares():
    cleaned = {s.file.removesuffix(".csv"): s for s in CLEANED_SPECS.values()}
    declared_fks, cleaned_fks = set(), set()
    for t in TABLES:
        c = cleaned[t.csv.removesuffix(".csv")]
        assert tuple(k.upper() for k in c.key) == t.primary_key, t.name
        assert [x.upper() for x in _header(t)] == [col.name for col in t.columns], t.name
        declared_fks |= {(t.name, f.columns, f.parent.upper(), f.parent_columns) for f in t.foreign_keys}
        cleaned_fks |= {(t.name, tuple(x.upper() for x in cols), parent.upper(), tuple(x.upper() for x in pcols)) for cols, parent, pcols in c.foreign_keys}
    assert declared_fks == cleaned_fks and len(declared_fks) == 20


def test_relationships_the_data_does_not_support_have_no_foreign_key():
    """No fake foreign keys: a polymorphic id, a text label, a sentinel and a partly filled code are not keys."""
    fk_columns = {(t.name, c) for t in TABLES for f in t.foreign_keys for c in f.columns}
    for not_fk in [("DEMAND", "LOCATION_ID"), ("DEMAND_MODELING_PANEL", "LOCATION_ID"), ("SCENARIO_STATE", "ENTITY_ID"), ("DISRUPTION_IMPACT", "ENTITY_ID"),
                   ("TARIFF", "DEST_ISO3"), ("PORT", "COUNTRY_ISO2")]:
        assert not_fk not in fk_columns, not_fk
    for t in TABLES:  # and where there IS a foreign key, it references a primary key with the identical type
        for f in t.foreign_keys:
            parent = BY_NAME[f.parent]
            assert f.parent_columns == parent.primary_key
            assert all(t.column(a).type == parent.column(b).type for a, b in zip(f.columns, f.parent_columns))


def _header(table) -> list[str]:
    return (DATA_DIR / table.csv).open(encoding="utf-8").readline().strip().split(",") if BUILT else [c.name for c in table.columns]


# --------------------------------------------------------------------------- the package on disk
@needs_data
def test_the_checked_in_package_is_what_the_generator_writes():
    package = hana_core.build_package(DATA_DIR)
    assert set(package) == {"01_create_tables.sql", "02_create_tables.sql", "02_comments.sql", "HANA_PREFLIGHT_HACKFEST0119.sql", "03_validate.sql", "99_drop_tables.sql", "HANA_DATA_DICTIONARY.csv", "HANA_DATA_DICTIONARY.md", "HANA_LOAD_PLAN.md"}
    for name, text in package.items():
        assert (PACKAGE_DIR / name).read_text(encoding="utf-8") == text, f"{name} is stale: python -m backend.database.hana_core"


@needs_data
def test_the_cleaned_data_was_not_modified_by_any_of_this():
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))["tables"]
    for name, info in manifest.items():
        assert hashlib.sha256((DATA_DIR / info["file"]).read_bytes()).hexdigest() == info["sha256"], name


@needs_data
def test_every_cleaned_value_fits_the_column_it_is_declared_for():
    pre = load.preflight(DATA_DIR)
    assert pre.errors == [] and pre.rows == load.expected_counts(DATA_DIR)


@needs_data
def test_the_dictionary_describes_every_column_of_the_ddl_and_only_those():
    facts = hana_core.read_facts(DATA_DIR)
    rows = hana_core.dictionary_rows(facts)
    assert list(rows[0]) == ["TABLE", "COLUMN", "DATA TYPE", "NULLABLE", "PRIMARY KEY", "FOREIGN KEY", "PROVENANCE", "DESCRIPTION"]
    assert [(r["TABLE"], r["COLUMN"]) for r in rows] == [(t.name, c.name) for t in TABLES for c in t.columns]
    ddl = hana_core.render_create_sql()
    for r in rows:
        assert f'"{r["COLUMN"]}" {r["DATA TYPE"]}{" NOT NULL" if r["NULLABLE"] == "NO" else ""}' in ddl
        assert r["PROVENANCE"] in PROVENANCE_VALUES and r["DESCRIPTION"].strip()
    assert sum(r["PRIMARY KEY"] == "YES" for r in rows) == sum(len(t.primary_key) for t in TABLES)
    assert sum(bool(r["FOREIGN KEY"]) for r in rows) == sum(len(f.columns) for t in TABLES for f in t.foreign_keys)


def test_the_teardown_drops_children_before_parents_and_the_comments_cover_every_column():
    drop = hana_core.render_drop_sql()
    order = [line.split('"')[1] for line in drop.splitlines() if line.startswith("DROP TABLE")]
    assert order == list(reversed(REQUESTED_ORDER))
    comments = hana_core.render_comments_sql()
    assert comments.count("COMMENT ON COLUMN") == sum(len(t.columns) for t in TABLES) and comments.count("COMMENT ON TABLE") == 17


# --------------------------------------------------------------------------- the first-run pre-flight for HACKFEST0119
SCHEMA = "HACKFEST0119"


def _preflight_execs(sql: str) -> list[str]:
    """The statements STEP 2 hands to EXEC, un-escaped (the doubled quotes of a SQL string literal turned back into single ones)."""
    return [e.replace("''", "'") for e in re.findall(r"EXEC '((?:[^']|'')*)';", sql)]


def test_the_preflight_can_only_read_or_create_never_destroy_replace_or_load():
    sql = hana_core.render_preflight_sql()
    code = re.sub(r"--[^\n]*", "", sql).upper()
    for word in ("DROP", "DELETE", "TRUNCATE", "ALTER", "INSERT", "UPDATE", "MERGE", "UPSERT", "GRANT", "REVOKE", "RENAME", "IMPORT", "REPLACE", "COMMENT"):
        assert not re.search(rf"\b{word}\b", code), word
    assert code.count("CREATE COLUMN TABLE") == 17 and "CREATE OR REPLACE" not in code
    assert code.count("EXEC '") == 17  # nothing is created any other way


def test_the_preflight_creates_exactly_the_tables_of_01_and_only_writes_the_schema_out():
    sql = hana_core.render_preflight_sql()
    execs = _preflight_execs(sql)
    plain = [s.strip() for s in load.split_statements(hana_core.render_create_sql())]
    assert len(execs) == len(plain) == 17
    assert [e.replace(f'"{SCHEMA}".', "") for e in execs] == plain  # types, keys, foreign keys, CHECKs, quoting and order: identical
    for t, e in zip(TABLES, execs):
        assert e.startswith(f'CREATE COLUMN TABLE "{SCHEMA}"."{t.name}" (')
    assert sum(e.count("REFERENCES") for e in execs) == sum(e.count(f'REFERENCES "{SCHEMA}"."') for e in execs) == 20  # every parent is qualified too
    assert 'TARIFF_RATE_PCT" DOUBLE NOT NULL' in execs[7]
    assert all("IN ('REAL', 'DERIVED', 'SYNTHETIC', 'AUTHORED')" in e for e in execs)  # written ''REAL'' inside the EXEC string, REAL once un-escaped


def test_the_preflight_sets_the_schema_first_and_stops_at_a_gate_before_anything_is_created():
    sql = hana_core.render_preflight_sql()
    code = re.sub(r"--[^\n]*", "", sql)
    assert code.strip().startswith(f'SET SCHEMA "{SCHEMA}";')
    assert sql.index("SIGNAL SQL_ERROR_CODE 10001") < sql.index("SIGNAL SQL_ERROR_CODE 10002") < sql.index("EXEC '")
    gate = sql[sql.index("gate 2"):sql.index("SIGNAL SQL_ERROR_CODE 10002")]
    for name in LOAD_ORDER:  # the gate looks for every one of the 17 names, in the target schema only
        assert f"'{name}'" in gate, name
    assert f"SCHEMA_NAME = '{SCHEMA}'" in gate
    assert [e.split('"')[3] for e in _preflight_execs(sql)] == list(LOAD_ORDER)  # parents before children, the load order


def test_the_preflight_comments_hold_no_semicolon_or_apostrophe_a_client_could_misread():
    for line in hana_core.render_preflight_sql().splitlines():
        if line.lstrip().startswith("--"):
            assert ";" not in line and "'" not in line, line


def test_the_preflight_verifies_the_types_the_requirements_pin():
    sql = hana_core.render_preflight_sql()
    types = sql[sql.index("3c."):]
    assert "SELECT 'TARIFF', 'TARIFF_RATE_PCT', 'DOUBLE' FROM DUMMY" in types
    for table in ("SUPPLIER_PRODUCT", "ROUTE"):
        assert f"SELECT '{table}', 'CAPACITY', 'INTEGER' FROM DUMMY" in types
    wanted = [(t.name, c.name, c.type) for t in TABLES for c in t.columns if c.type.startswith(("DECIMAL", "DOUBLE")) or c.name == "CAPACITY"]
    lines = [line for line in types.splitlines() if line.strip().startswith(("SELECT", "UNION ALL SELECT")) and line.rstrip().endswith("FROM DUMMY")]
    assert len(lines) == len(wanted)
    for line, (table, column, sql_type) in zip(lines, wanted):  # one line per DECIMAL / DOUBLE / CAPACITY column, in table order
        assert all(f"'{v}'" in line for v in (table, column, sql_type)), line
    n_ck = sum(1 + len(t.checks) for t in TABLES)
    assert f"{n_ck} FROM DUMMY" in sql and "CHECK_CONDITION IS NOT NULL" in sql  # the CHECK constraints are counted in the catalog too


def test_the_preflight_instructions_exist_and_name_the_file_to_run():
    text = (PACKAGE_DIR / "HANA_PREFLIGHT_INSTRUCTIONS.md").read_text(encoding="utf-8")
    assert "HANA_PREFLIGHT_HACKFEST0119.sql" in text


# --------------------------------------------------------------------------- 02_create_tables.sql: the 17 empty tables, paste-and-run
def _parse_created_tables(sql: str) -> dict[str, dict]:
    """Reads SQL TEXT, not the Python model: {table: {cols: [(name, type, nullable)], pk: [names], fks: {column: 'PARENT.COLUMN'}}}."""
    code = re.sub(r"--[^\n]*", "", sql)
    tables = {}
    for statement in (s.strip() for s in code.split(";") if s.strip()):
        table, body = re.match(r'CREATE COLUMN TABLE "\w+"\."(\w+)" \((.*)\)$', statement, re.S).groups()
        cols, pk, fks = [], [], {}
        for line in (ln.strip().rstrip(",") for ln in body.split("\n")):
            if c := re.match(r'^"(\w+)" ([A-Z]+(?:\(\d+(?:,\d+)?\))?)( NOT NULL)?$', line):
                cols.append((c.group(1), c.group(2), c.group(3) is None))
            if p := re.search(r"PRIMARY KEY \(([^)]*)\)", line):
                pk = re.findall(r'"(\w+)"', p.group(1))
            if f := re.search(r'FOREIGN KEY \(([^)]*)\) REFERENCES "\w+"\."(\w+)" \(([^)]*)\)', line):
                fks.update(zip(re.findall(r'"(\w+)"', f.group(1)), (f"{f.group(2)}.{c}" for c in re.findall(r'"(\w+)"', f.group(3)))))
        tables[table] = {"cols": cols, "pk": pk, "fks": fks}
    return tables


def test_02_create_tables_is_exactly_17_create_statements_in_the_target_schema_and_nothing_else():
    sql = hana_core.render_create_tables_sql()
    code = re.sub(r"--[^\n]*", "", sql)
    statements = [s.strip() for s in code.split(";") if s.strip()]
    assert len(statements) == 17 == code.count("CREATE COLUMN TABLE")
    assert all(s.startswith(f'CREATE COLUMN TABLE "{SCHEMA}"."') for s in statements)
    assert [s.split('"')[3] for s in statements] == REQUESTED_ORDER  # the 17 names, exactly, in the load order
    for word in ("INSERT", "DROP", "TRUNCATE", "DELETE", "ALTER", "UPDATE", "MERGE", "UPSERT", "GRANT", "COMMENT", "SET SCHEMA", "EXEC", "DO BEGIN", "REPLACE"):
        assert not re.search(rf"\b{word}\b", code.upper()), word
    for line in sql.splitlines():  # a client must not mistake a comment for a statement end or a string
        if line.lstrip().startswith("--"):
            assert ";" not in line and "'" not in line, line


def test_02_create_tables_is_01_with_only_the_schema_written_out():
    with_schema = load.split_statements(hana_core.render_create_tables_sql())
    plain = load.split_statements(hana_core.render_create_sql())
    assert len(with_schema) == len(plain) == 17
    assert [s.replace(f'"{SCHEMA}".', "") for s in with_schema] == plain  # types, keys, foreign keys, CHECKs, quoting and order: identical
    assert sum(s.count(f'REFERENCES "{SCHEMA}"."') for s in with_schema) == sum(s.count("REFERENCES") for s in with_schema) == 20


@needs_data
def test_02_create_tables_on_disk_matches_the_dictionary_and_the_cleaned_csv_headers():
    """The FILE as it sits in db/hana/core, parsed as text, against the dictionary file and the cleaned CSVs: not against the model that wrote it."""
    parsed = _parse_created_tables((PACKAGE_DIR / "02_create_tables.sql").read_text(encoding="utf-8"))
    assert list(parsed) == REQUESTED_ORDER
    dictionary = list(csv.DictReader((PACKAGE_DIR / "HANA_DATA_DICTIONARY.csv").open(encoding="utf-8", newline="")))
    assert len(dictionary) == sum(len(t["cols"]) for t in parsed.values()) == 164
    for row in dictionary:
        t = parsed[row["TABLE"]]
        col = next(c for c in t["cols"] if c[0] == row["COLUMN"])
        assert col[1] == row["DATA TYPE"], row
        assert (row["NULLABLE"] == "YES") == col[2], row
        assert (row["PRIMARY KEY"] == "YES") == (row["COLUMN"] in t["pk"]), row
        assert row["FOREIGN KEY"] == t["fks"].get(row["COLUMN"], ""), row
    for table in TABLES:
        assert [c[0] for c in parsed[table.name]["cols"]] == [h.upper() for h in _header(table)], table.name  # same columns, same order as the cleaned CSV


# --------------------------------------------------------------------------- the loader's own logic
def test_a_value_is_converted_exactly_or_refused_never_rounded_or_truncated():
    money, text, flag = BY_NAME["SUPPLIER_PRODUCT"].column("UNIT_COST"), BY_NAME["SUPPLIER_PRODUCT"].column("PRODUCT_ID"), BY_NAME["TARIFF"].column("IS_COUNTRY_LEVEL_PROXY")
    assert str(load.converter(money)("113.52")) == "113.52"
    with pytest.raises(ValueError, match="would round it"):
        load.converter(money)("113.123456")  # DECIMAL(16,4) has 4 places
    with pytest.raises(ValueError, match="do not fit"):
        load.converter(text)("x" * 33)
    with pytest.raises(ValueError, match="NOT NULL"):
        load.converter(text)("")
    assert load.converter(BY_NAME["SUPPLIER"].column("ORIGIN_PORT_ID"))("") is None  # a nullable column takes an empty field as NULL
    assert load.converter(flag)("True") is True and load.converter(flag)("False") is False
    with pytest.raises(ValueError):
        load.converter(flag)("yes")
    tariff = load.converter(BY_NAME["TARIFF"].column("TARIFF_RATE_PCT"))
    assert tariff("8.379999999999999") == 8.379999999999999  # DOUBLE keeps exactly what the file holds


def test_sql_files_are_split_on_semicolons_outside_quotes_and_comments():
    sql = "-- a comment; with a semicolon\nSELECT 'a;b' AS X;\n\nSELECT 2 -- trailing\n;\n"
    assert load.split_statements(sql) == ["SELECT 'a;b' AS X", "SELECT 2 -- trailing"]
    assert all(s.count("'") % 2 == 0 for s in load.split_statements(hana_core.render_create_sql()))


def test_the_insert_uses_qmark_parameters_which_hdbcli_and_sqlite_share():
    sql = load.insert_sql(BY_NAME["SCENARIO_PARAM"])
    assert sql == ('INSERT INTO "SCENARIO_PARAM" ("SCENARIO_ID", "PARAM_KEY", "QUALIFIER", "PARAM_VALUE", "PARAM_TEXT", "UNIT", "BASIS", "PROVENANCE") '
                   "VALUES (?, ?, ?, ?, ?, ?, ?, ?)")


# --------------------------------------------------------------------------- pre-flight mutations: bad data is refused before any database
SMALL = tuple(BY_NAME[n] for n in ("COUNTRY", "PORT", "PRODUCT", "WAREHOUSE", "SUPPLIER", "SUPPLIER_PRODUCT"))


@pytest.fixture()
def copy_of_cleaned(tmp_path):
    if not BUILT:
        pytest.skip("cleaned layer not built")
    for t in SMALL:
        shutil.copy(DATA_DIR / t.csv, tmp_path / t.csv)
    for name in ("manifest.json", "data_dictionary.csv"):
        shutil.copy(DATA_DIR / name, tmp_path / name)
    return tmp_path


def edit(path: Path, fn) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(fn(lines)) + "\n", encoding="utf-8")


def errors_after(copy: Path, csv: str, fn) -> str:
    edit(copy / csv, fn)
    return "\n".join(load.preflight(copy, SMALL).errors)


def test_preflight_passes_on_an_untouched_copy(copy_of_cleaned):
    assert load.preflight(copy_of_cleaned, SMALL).ok


def test_preflight_refuses_a_text_too_long_for_its_column(copy_of_cleaned):
    assert "do not fit NVARCHAR(32)" in errors_after(copy_of_cleaned, "supplier_product.csv", lambda ls: [ls[0]] + [ls[1].replace("S001,85099B", "S001," + "9" * 40)] + ls[2:])


def test_preflight_refuses_a_decimal_that_would_be_rounded(copy_of_cleaned):
    assert "would round it" in errors_after(copy_of_cleaned, "supplier_product.csv", lambda ls: [ls[0]] + [ls[1].replace("113.52", "113.123456")] + ls[2:])


def test_preflight_refuses_an_empty_required_field(copy_of_cleaned):
    assert "NOT NULL" in errors_after(copy_of_cleaned, "supplier.csv", lambda ls: [ls[0], ls[1].replace("SYNTHETIC Supplier S001", "")] + ls[2:])


def test_preflight_refuses_a_duplicate_primary_key(copy_of_cleaned):
    assert "duplicate primary key" in errors_after(copy_of_cleaned, "supplier.csv", lambda ls: ls + [ls[1]])


def test_preflight_refuses_an_orphan_foreign_key(copy_of_cleaned):
    assert "has no parent" in errors_after(copy_of_cleaned, "supplier_product.csv", lambda ls: [ls[0]] + [ls[1].replace("85099B", "NOSUCHPRODUCT")] + ls[2:])


def test_preflight_refuses_a_renamed_column_and_a_changed_row_count(copy_of_cleaned):
    assert "has columns" in errors_after(copy_of_cleaned, "warehouse.csv", lambda ls: [ls[0].replace("warehouse_id", "warehouse_code")] + ls[1:])
    edit(copy_of_cleaned / "warehouse.csv", lambda ls: ["warehouse_id,warehouse_name,city,country_iso3,nearest_port_id,provenance"] + ls[1:-1])
    assert "rows read" in "\n".join(load.preflight(copy_of_cleaned, SMALL).errors)


# --------------------------------------------------------------------------- the DDL and validation SQL, rehearsed in SQLite
@pytest.fixture(scope="module")
def facts():
    if not BUILT:
        pytest.skip("cleaned layer not built")
    return hana_core.read_facts(DATA_DIR)


@pytest.fixture(scope="module")
def constrained(facts):
    """The DDL as written, loaded with every cleaned row: keys, foreign keys, CHECKs and NOT NULLs all enforced."""
    conn = load.sqlite_connection(hana_core.render_create_sql())
    load.load_tables(conn, DATA_DIR, echo=lambda _: None)
    return conn


@pytest.fixture(scope="module")
def unconstrained():
    """The same tables and rows with NO constraints, so bad data can get in and the validation SQL has to be the thing that notices."""
    conn = load.sqlite_connection()
    for t in TABLES:
        conn.execute(f'CREATE TABLE "{t.name}" ({", ".join(chr(34) + c.name + chr(34) + " " + c.type for c in t.columns)})')
    for t in TABLES:  # a plain index, NOT a unique one: duplicates must still be possible, but the orphan checks need a lookup
        conn.execute(f'CREATE INDEX "IX_{t.name}" ON "{t.name}" ({", ".join(chr(34) + k + chr(34) for k in t.primary_key)})')
    conn.commit()
    load.load_tables(conn, DATA_DIR, echo=lambda _: None)
    return conn


@pytest.fixture(scope="module")
def groups(facts):
    return hana_core.Validation(facts, "sqlite").build()


def scorecard(conn, groups, group=None) -> list[dict]:
    cursor = conn.cursor()
    rows = []
    for name, selects in groups.items():
        if group and name != group:
            continue
        cursor.execute("\nUNION ALL\n".join(selects))
        names = [d[0] for d in cursor.description]
        rows += [dict(zip(names, r)) for r in cursor.fetchall()]
    return rows


@contextmanager
def mutation(conn, *statements):
    conn.execute("SAVEPOINT mutation")
    try:
        for s in statements:
            conn.execute(s)
        yield
    finally:
        conn.execute("ROLLBACK TO mutation")
        conn.execute("RELEASE mutation")


@needs_data
def test_the_rehearsal_loads_every_row_and_every_validation_check_passes(constrained, groups, facts):
    assert load.table_counts(constrained) == facts.rows == load.expected_counts(DATA_DIR)
    rows = scorecard(constrained, groups)
    assert len(rows) > 150 and {r["OUTCOME"] for r in rows} == {"PASS"}, [r for r in rows if r["OUTCOME"] != "PASS"]
    assert {r["CHECK_GROUP"] for r in rows} >= {"ROW_COUNT", "NULL_PRIMARY_KEY", "DUPLICATE_PRIMARY_KEY", "ORPHAN_FOREIGN_KEY", "PROVENANCE", "DATES", "QUANTITIES_AND_RANGES",
                                                 "TARIFF_RANGE", "CONSISTENCY", "CONTROL_TOTALS"}


@needs_data
def test_the_validation_covers_everything_that_was_asked_for(groups):
    text = hana_core.render_validation_sql(hana_core.read_facts(DATA_DIR))
    for wanted in ("duplicate demand keys", "duplicate inventory keys", "supplier-product key uniqueness", "no NULL in", "-> COUNTRY.ISO3", "invalid", "rows"):
        assert wanted.lower() in text.lower() or wanted == "invalid", wanted
    assert {"ROW_COUNT", "NULL_PRIMARY_KEY", "DUPLICATE_PRIMARY_KEY", "ORPHAN_FOREIGN_KEY", "PROVENANCE", "DATES", "QUANTITIES_AND_RANGES", "TARIFF_RANGE"} <= set(groups)
    assert "TO_DATE" in text and "MONTH(" in text and "ADD_DAYS(" in text  # the HANA file uses HANA's functions...
    assert "strftime" not in text  # ...and none of SQLite's


# ---- the constraints reject bad rows
@needs_data
@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO \"COUNTRY\" VALUES ('GBR', 'GB', 'Duplicate', 'Europe', 'High income', NULL, 'REAL')",  # duplicate primary key
        "INSERT INTO \"COUNTRY\" VALUES ('ZZZ', NULL, 'X', 'R', 'I', NULL, 'made-up')",  # invalid provenance
        "INSERT INTO \"COUNTRY\" VALUES ('ZZZ', NULL, NULL, 'R', 'I', NULL, 'REAL')",  # NOT NULL name
        "INSERT INTO \"SUPPLIER_PRODUCT\" VALUES ('S001', 'NOSUCH', 100, 1, 1, 0.5, 'SYNTHETIC')",  # orphan foreign key
        "INSERT INTO \"SUPPLIER_PRODUCT\" VALUES ('S001', '22197', 0, 1, 1, 0.5, 'SYNTHETIC')",  # a disrupted supplier baked into the master
        "INSERT INTO \"SUPPLIER_PRODUCT\" VALUES ('S001', '22197', 5, 1, 1, 1.5, 'SYNTHETIC')",  # reliability above 1
        "UPDATE \"INVENTORY\" SET \"OUTBOUND_QUANTITY\" = -1 WHERE rowid = 1",  # negative quantity
        "UPDATE \"DEMAND\" SET \"DEMAND_QUANTITY\" = 0 WHERE rowid = 1",  # a demand row with no demand
        "UPDATE \"TARIFF\" SET \"TARIFF_RATE_PCT\" = -1 WHERE rowid = 1",  # negative tariff
        "UPDATE \"PORT\" SET \"LATITUDE\" = 91 WHERE \"PORT_ID\" = 31140",  # by key: SQLite aliases a lone INTEGER primary key to rowid
        "UPDATE \"DISRUPTION_EVENT\" SET \"END_TS\" = '2000-01-01 00:00:00' WHERE rowid = 1",  # ends before it starts
        "UPDATE \"ROUTE\" SET \"TRANSPORT_MODE\" = 'boat' WHERE rowid = 1",
        "UPDATE \"SCENARIO_STATE\" SET \"STATUS\" = 'ACTIVE' WHERE rowid = 1",  # a deviation, never 'ACTIVE'
        "UPDATE \"DEMAND_MODELING_PANEL\" SET \"MONTH\" = 13 WHERE rowid = 1",
    ],
)
def test_the_constraints_reject_a_bad_row(constrained, sql):
    with pytest.raises(sqlite3.IntegrityError):
        constrained.execute(sql)
    constrained.rollback()


# ---- the validation SQL finds bad rows even when there are no constraints
MUTATIONS = [
    ("a NULL primary key", "NULL_PRIMARY_KEY", "COUNTRY", ["UPDATE \"COUNTRY\" SET \"ISO3\" = NULL WHERE \"ISO3\" = 'GBR'"]),
    ("a duplicate demand key", "DUPLICATE_PRIMARY_KEY", "duplicate demand keys", ['INSERT INTO "DEMAND" SELECT * FROM "DEMAND" LIMIT 1']),
    ("a duplicate inventory key", "DUPLICATE_PRIMARY_KEY", "duplicate inventory keys", ['INSERT INTO "INVENTORY" SELECT * FROM "INVENTORY" LIMIT 1']),
    ("a repeated supplier-product pair", "DUPLICATE_PRIMARY_KEY", "supplier-product key uniqueness", ['INSERT INTO "SUPPLIER_PRODUCT" SELECT * FROM "SUPPLIER_PRODUCT" LIMIT 1']),
    ("an orphan foreign key", "ORPHAN_FOREIGN_KEY", "SUPPLIER_PRODUCT.PRODUCT_ID -> PRODUCT.PRODUCT_ID", ['UPDATE "SUPPLIER_PRODUCT" SET "PRODUCT_ID" = \'NOSUCH\' WHERE rowid = 1']),
    ("an orphan country on demand", "ORPHAN_FOREIGN_KEY", "DEMAND.COUNTRY_ISO3", ['UPDATE "DEMAND" SET "COUNTRY_ISO3" = \'ZZZ\' WHERE rowid = 1']),
    ("an entity that is neither a route nor a supplier", "ORPHAN_FOREIGN_KEY", "SCENARIO_STATE.ENTITY_ID", ['UPDATE "SCENARIO_STATE" SET "ENTITY_ID" = \'NOPE\' WHERE rowid = 1']),
    ("a demand location that is not the country's label", "ORPHAN_FOREIGN_KEY", "DEMAND.LOCATION_ID", ['UPDATE "DEMAND" SET "LOCATION_ID" = \'Atlantis\' WHERE rowid = 1']),
    ("an invalid provenance label", "PROVENANCE", "SUPPLIER: every row has one of", ['UPDATE "SUPPLIER" SET "PROVENANCE" = \'made-up\' WHERE rowid = 1']),
    ("a date outside the cleaned window", "DATES", "DEMAND: DATE", ['UPDATE "DEMAND" SET "DATE" = \'1999-01-01\' WHERE rowid = 1']),
    ("a NULL date", "DATES", "INVENTORY: DATE", ['UPDATE "INVENTORY" SET "DATE" = NULL WHERE rowid = 1']),
    ("an event that ends before it starts", "DATES", "an event does not end before it starts", ['UPDATE "DISRUPTION_EVENT" SET "END_TS" = \'2000-01-01 00:00:00\' WHERE "END_TS" IS NOT NULL']),
    ("a month that disagrees with its date", "DATES", "MONTH agrees with DATE", ['UPDATE "DEMAND_MODELING_PANEL" SET "MONTH" = 12 WHERE "MONTH" = 1']),
    ("a negative inventory quantity", "QUANTITIES_AND_RANGES", "stock and flows cannot be negative", ['UPDATE "INVENTORY" SET "UNFILLED_QUANTITY" = -1 WHERE rowid = 1']),
    ("a demand row with no demand", "QUANTITIES_AND_RANGES", "a demand row has positive net demand", ['UPDATE "DEMAND" SET "DEMAND_QUANTITY" = 0 WHERE rowid = 1']),
    ("a disrupted supplier baked into the master", "QUANTITIES_AND_RANGES", "nominal capacity is positive", ['UPDATE "SUPPLIER_PRODUCT" SET "CAPACITY" = 0 WHERE rowid = 1']),
    ("a negative tariff", "TARIFF_RANGE", "no negative rate", ['UPDATE "TARIFF" SET "TARIFF_RATE_PCT" = -5 WHERE rowid = 1']),
    ("an unflagged tariff above 50%", "TARIFF_RANGE", "not flagged EXTREME_VALUE", ['UPDATE "TARIFF" SET "TARIFF_RATE_PCT" = 75, "QUALITY_FLAG" = \'OK\' WHERE rowid = 1']),
    ("a tariff outside the year range", "DATES", "EFFECTIVE_YEAR", ['UPDATE "TARIFF" SET "EFFECTIVE_YEAR" = 1800 WHERE rowid = 1']),
    ("stock that does not add up", "CONSISTENCY", "closing = opening + inbound - outbound", ['UPDATE "INVENTORY" SET "CLOSING_STOCK" = "CLOSING_STOCK" + 1 WHERE rowid = 5']),
    ("a lost sale that disappears", "CONSISTENCY", "no lost sale is hidden", ['UPDATE "INVENTORY" SET "DEMAND_QUANTITY" = "DEMAND_QUANTITY" + 3 WHERE rowid = 7']),
    ("a stockout flag that disagrees", "CONSISTENCY", "STOCKOUT_FLAG is set exactly", ['UPDATE "INVENTORY" SET "STOCKOUT_FLAG" = 1 WHERE "UNFILLED_QUANTITY" = 0 AND rowid < 50']),
    ("a broken opening balance", "CONSISTENCY", "opens with the previous day's closing", ['UPDATE "INVENTORY" SET "OPENING_STOCK" = "OPENING_STOCK" + 1 WHERE rowid = 30']),
    ("a disrupted baseline", "CONSISTENCY", "BASELINE_NORMAL changes nothing", ["INSERT INTO \"SCENARIO_STATE\" VALUES ('BASELINE_NORMAL', 'SUPPLIER', 'S001', 'DISRUPTED', 0, 'x', 'SYNTHETIC')"]),
    ("a Suez lane with no Cape lane", "CONSISTENCY", "every SUEZ lane has a CAPE lane", ['DELETE FROM "ROUTE" WHERE "ROUTE_ID" = \'MUM-ROT-CAPE\'']),
    ("a Cape lane shorter than the Suez lane", "CONSISTENCY", "CAPE lane is longer", ['UPDATE "ROUTE" SET "DISTANCE_KM" = 1 WHERE "ROUTE_ID" = \'SHA-ROT-CAPE\'']),
    ("a product left with one supplier", "CONSISTENCY", "at least two", ['DELETE FROM "SUPPLIER_PRODUCT" WHERE "PRODUCT_ID" = \'22197\' AND "SUPPLIER_ID" <> \'S002\'']),
    ("a lost row", "ROW_COUNT", "DEMAND has", ['DELETE FROM "DEMAND" WHERE rowid = 1']),
    ("a changed quantity", "CONTROL_TOTALS", "SUM(DEMAND_QUANTITY)", ['UPDATE "DEMAND" SET "DEMAND_QUANTITY" = "DEMAND_QUANTITY" + 1 WHERE rowid = 1']),
]


@needs_data
def test_the_untouched_tables_pass_every_group(unconstrained, groups):
    assert {r["OUTCOME"] for r in scorecard(unconstrained, groups)} == {"PASS"}


@needs_data
@pytest.mark.parametrize("what,group,name,statements", MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_the_validation_sql_finds(unconstrained, groups, what, group, name, statements):
    with mutation(unconstrained, *statements):
        failed = [r for r in scorecard(unconstrained, groups, group) if r["OUTCOME"] == "FAIL"]
    assert any(name in r["CHECK_NAME"] or name in r["TABLE_NAME"] for r in failed), f"{what}: not caught by {group}; failing: {[r['CHECK_NAME'] for r in failed]}"


@needs_data
def test_a_warning_is_a_known_property_and_turns_into_a_fail_only_for_errors(unconstrained, groups):
    """The tariff flags and the per-table provenance are WARNs: a changed count shows as WARN, a real violation as FAIL."""
    with mutation(unconstrained, 'UPDATE "TARIFF" SET "QUALITY_FLAG" = \'OK\' WHERE "QUALITY_FLAG" LIKE \'EXTREME%\' AND "TARIFF_RATE_PCT" < 100'):
        outcomes = {(r["CHECK_NAME"], r["OUTCOME"]) for r in scorecard(unconstrained, groups, "TARIFF_RANGE")}
    assert any(o == "WARN" and "flagged EXTREME_VALUE" in n for n, o in outcomes)
    with mutation(unconstrained, 'UPDATE "COUNTRY" SET "PROVENANCE" = \'SYNTHETIC\' WHERE "ISO3" = \'GBR\''):
        outcomes = [(r["SEVERITY"], r["OUTCOME"]) for r in scorecard(unconstrained, groups, "PROVENANCE") if r["OUTCOME"] != "PASS"]
    assert outcomes == [("WARN", "WARN")]


# --------------------------------------------------------------------------- the script: it must not connect unless asked
def _script():
    spec = importlib.util.spec_from_file_location("load_hana_core_script", ROOT / "scripts" / "load_hana_core.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def quick_preflight(monkeypatch):
    """The pre-flight reads 370k rows; the script's decisions are what these tests are about."""
    if not BUILT:
        pytest.skip("cleaned layer not built")
    rows = {t.name: 1 for t in TABLES}
    monkeypatch.setattr(load, "preflight", lambda data_dir=None, *a, **k: load.Preflight(rows=rows))


def test_the_default_run_is_a_dry_run_that_connects_to_nothing(quick_preflight, capsys):
    script = _script()
    assert script.main([]) == 0
    assert "dry run: nothing was created, loaded or connected to" in capsys.readouterr().out


def test_execute_refuses_without_an_explicit_schema(quick_preflight, capsys):
    assert _script().main(["--execute"]) == 2
    assert "explicit --schema" in capsys.readouterr().out


def test_execute_refuses_a_sqlite_target_and_a_strange_schema_name(quick_preflight, monkeypatch, capsys):
    for name in ("DATABASE_URL", "DATABASE_SCHEMA", "VCAP_SERVICES"):
        monkeypatch.delenv(name, raising=False)
    script = _script()
    assert script.main(["--execute", "--schema", "X; DROP TABLE Y"]) == 2
    assert script.main(["--execute", "--schema", "HACKFEST0119"]) == 2  # nothing configured: the default target is SQLite
    assert "SQLite" in capsys.readouterr().out


# --------------------------------------------------------------------------- the HANA load path, decided and rehearsed without HANA
def _fresh_target():
    """An in-memory database with the 17 empty tables: what HACKFEST0119 holds after the pre-flight succeeds."""
    return load.sqlite_connection(hana_core.render_create_sql())


def _plan(conn, resume=False, expected=None):
    expected = expected or {t.name: 1 for t in TABLES}
    return load.plan_load(load.inspect_target(conn, "HACKFEST0119", dialect="sqlite"), expected, resume)


def test_nothing_may_be_loaded_into_a_schema_without_the_tables():
    plan = _plan(load.sqlite_connection())
    assert not plan.ok and len(plan.problems) == 17 and plan.to_load == []
    assert all("does not exist in HACKFEST0119" in p and "HANA_PREFLIGHT_HACKFEST0119.sql" in p for p in plan.problems)


def test_a_table_that_is_not_this_models_table_is_refused():
    conn = _fresh_target()
    conn.execute('DROP TABLE "DEMAND_MODELING_PANEL"')  # test-local SQLite only
    conn.execute('CREATE TABLE "DEMAND_MODELING_PANEL" ("DATE" DATE, "X" INTEGER)')
    plan = _plan(conn)
    assert plan.problems == [f"DEMAND_MODELING_PANEL has 2 columns in HACKFEST0119, the model declares {len(BY_NAME['DEMAND_MODELING_PANEL'].columns)}: it is not this model's table"]


def test_an_empty_schema_plan_loads_all_17_in_dependency_order():
    plan = _plan(_fresh_target())
    assert plan.ok and plan.to_load == list(LOAD_ORDER) and plan.skip == []


@needs_data
def test_a_table_that_holds_rows_stops_the_load_and_nothing_is_ever_deleted():
    conn = _fresh_target()
    load.load_tables(conn, DATA_DIR, SMALL, echo=lambda _: None)
    before = load.table_counts(conn)
    expected = load.expected_counts(DATA_DIR)
    plan = _plan(conn, expected=expected)
    assert not plan.ok and [p.split(" already holds")[0] for p in plan.problems] == [t.name for t in SMALL]
    # a full run refuses before writing a single row, and leaves what was there exactly as it was
    out = []
    assert load.run_load(conn, DATA_DIR, expected, "HACKFEST0119", dialect="sqlite", echo=out.append, tables=SMALL) == 2
    assert load.table_counts(conn) == before and "refusing to write anything" in "\n".join(out)


@needs_data
def test_resume_skips_complete_tables_loads_empty_ones_and_refuses_a_half_loaded_one():
    conn = _fresh_target()
    load.load_tables(conn, DATA_DIR, SMALL, echo=lambda _: None)
    expected = load.expected_counts(DATA_DIR)
    plan = _plan(conn, resume=True, expected=expected)
    assert plan.ok and plan.skip == [t.name for t in SMALL] and plan.to_load == [n for n in LOAD_ORDER if n not in {t.name for t in SMALL}]
    conn.execute('DELETE FROM "SUPPLIER_PRODUCT" WHERE "SUPPLIER_ID" = (SELECT MIN("SUPPLIER_ID") FROM "SUPPLIER_PRODUCT")')  # damage the fixture, not HANA
    problems = _plan(conn, resume=True, expected=expected).problems
    assert len(problems) == 1 and problems[0].startswith("SUPPLIER_PRODUCT holds") and "neither 0 nor the 14" in problems[0]


@needs_data
def test_a_read_only_check_reports_the_plan_and_writes_nothing():
    conn = _fresh_target()
    out = []
    assert load.run_load(conn, DATA_DIR, load.expected_counts(DATA_DIR), "HACKFEST0119", dialect="sqlite", read_only=True, echo=out.append) == 0
    assert sum(load.table_counts(conn).values()) == 0
    text = "\n".join(out)
    assert "the load would write 372,210 rows into 17 empty tables" in text and "read-only check: nothing was written" in text


@needs_data
def test_a_failed_load_stops_at_that_table_and_resume_finishes_it_without_deleting_anything(monkeypatch):
    first = TABLES[:8]  # COUNTRY .. TARIFF: the full 372k rows would only make this slow
    conn = _fresh_target()
    expected = load.expected_counts(DATA_DIR)
    real = load.iter_rows

    def failing(table, data_dir=DATA_DIR):
        for n, item in enumerate(real(table, data_dir)):
            if table.name == "TARIFF" and n == 100:
                raise ValueError("TARIFF line 102 column TARIFF_RATE_PCT: simulated")
            yield item

    monkeypatch.setattr(load, "iter_rows", failing)
    out = []
    assert load.run_load(conn, DATA_DIR, expected, "HACKFEST0119", dialect="sqlite", echo=out.append, tables=first) == 1
    counts = load.table_counts(conn, first)
    assert all(counts[t.name] == expected[t.name] for t in first[:7])  # every earlier table is committed
    assert counts["TARIFF"] == 0  # the failing table was rolled back
    assert "loading TARIFF failed after 0 rows (rolled back; earlier tables stay loaded)" in "\n".join(out)
    monkeypatch.setattr(load, "iter_rows", real)
    kept = dict(counts)
    assert load.run_load(conn, DATA_DIR, expected, "HACKFEST0119", dialect="sqlite", echo=lambda _: None, tables=first) == 2  # without --resume the same target is refused
    assert load.table_counts(conn, first) == kept
    out = []
    assert load.run_load(conn, DATA_DIR, expected, "HACKFEST0119", dialect="sqlite", resume=True, echo=out.append, tables=first) == 0
    assert load.table_counts(conn, first) == {t.name: expected[t.name] for t in first} and "every row count matches." in "\n".join(out)
    assert {r[3] for r in load.verify_counts(conn, expected, first)} == {"PASS"}


@needs_data
def test_the_counts_are_read_back_from_the_database_not_taken_from_what_was_inserted():
    conn = _fresh_target()
    expected = load.expected_counts(DATA_DIR)
    load.load_tables(conn, DATA_DIR, SMALL, echo=lambda _: None)
    assert {r[3] for r in load.verify_counts(conn, expected, SMALL)} == {"PASS"}
    conn.execute('DELETE FROM "SUPPLIER_PRODUCT" WHERE rowid = (SELECT MIN(rowid) FROM "SUPPLIER_PRODUCT")')  # a row that went missing after the insert
    rows = load.verify_counts(conn, expected, SMALL)
    assert [(n, want, got) for n, want, got, outcome in rows if outcome == "FAIL"] == [("SUPPLIER_PRODUCT", 14, 13)]


# --------------------------------------------------------------------------- the connection: a stand-in for hdbcli that records everything
class _FakeCursor:
    def __init__(self, conn):
        self.conn, self.rows = conn, []

    def execute(self, sql, params=None):
        self.conn.statements.append(sql)
        self.rows = [("SOMEUSER", self.conn.current_schema)] if "CURRENT_SCHEMA" in sql else []

    def fetchone(self):
        return self.rows[0]

    def fetchall(self):
        return self.rows


class _FakeHana:
    """Answers the two questions the loader asks first, records every statement, never touches a network."""
    def __init__(self, current_schema="HACKFEST0119", autocommit=True):
        self.statements, self.autocommit, self.closed, self.current_schema = [], autocommit, False, current_schema

    def getautocommit(self):
        return self.autocommit

    def setautocommit(self, value):
        self.autocommit = value

    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        self.closed = True


def test_a_bare_hdbcli_connection_is_taken_out_of_autocommit_before_anything_is_written():
    fake = _FakeHana(autocommit=True)  # hdbcli's own default
    load.ensure_transactional(fake)
    assert fake.autocommit is False
    load.ensure_transactional(load.sqlite_connection())  # a connection without the hdbcli calls is left alone


def test_the_loader_refuses_to_count_or_write_when_the_session_is_not_in_the_target_schema():
    fake = _FakeHana(current_schema="SOME_OTHER_SCHEMA")
    out = []
    assert load.run_load(fake, DATA_DIR, {t.name: 1 for t in TABLES}, "HACKFEST0119", echo=out.append) == 2
    assert "session schema is SOME_OTHER_SCHEMA, not HACKFEST0119" in out[0]
    assert all(s.lstrip().upper().startswith("SELECT") for s in fake.statements)  # it only ever looked


def test_the_first_questions_asked_of_hana_are_read_only_selects_filtered_on_the_target_schema():
    fake = _FakeHana()
    out = []
    assert load.run_load(fake, DATA_DIR, {t.name: 1 for t in TABLES}, "HACKFEST0119", echo=out.append) == 2  # the fake holds no tables
    assert fake.statements == ["SELECT CURRENT_USER, CURRENT_SCHEMA FROM DUMMY",
                               "SELECT TABLE_NAME, COUNT(*) FROM SYS.TABLE_COLUMNS WHERE SCHEMA_NAME = ? GROUP BY TABLE_NAME"]
    assert sum("does not exist in HACKFEST0119" in line for line in out[1].splitlines()) == 17


def test_a_connection_error_names_what_to_look_at_and_never_repeats_the_password():
    text = load.explain_connection_error(RuntimeError("Connection failed: user rejected password hunter2 (RTE:[89008])"), secret="hunter2")
    assert "hunter2" not in text and "***" in text and "allowed connections" in text
    assert "pip install -r requirements-sap.txt" in load.explain_connection_error(ModuleNotFoundError("No module named 'hdbcli'"))
    assert "rejected the user name or password" in load.explain_connection_error(RuntimeError("authentication failed"))


# --------------------------------------------------------------------------- the script's wiring, with the driver stubbed out
@pytest.fixture()
def hana_env(monkeypatch):
    for name in ("DATABASE_SCHEMA", "VCAP_SERVICES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATABASE_URL", "hana+hdbcli://SOMEUSER:hunter2@example.hana.prod-eu10.hanacloud.ondemand.com:443?encrypt=true&sslValidateCertificate=true")
    fake = _FakeHana()
    monkeypatch.setattr("backend.database.engine.build_engine", lambda target: type("E", (), {"raw_connection": staticmethod(lambda: fake)})())
    calls = {}
    monkeypatch.setattr(load, "run_load", lambda conn, data_dir, expected, schema, **kw: calls.update(kw, schema=schema) or 0)
    return fake, calls


def test_check_target_connects_and_sets_the_schema_but_asks_for_a_read_only_run(quick_preflight, hana_env, capsys):
    fake, calls = hana_env
    assert _script().main(["--check-target", "--schema", "HACKFEST0119"]) == 0
    assert calls == {"read_only": True, "resume": False, "schema": "HACKFEST0119"}
    assert fake.statements == ['SET SCHEMA "HACKFEST0119"'] and fake.closed  # no CREATE, no DELETE, and the connection is closed
    out = capsys.readouterr().out
    assert "hunter2" not in out and "SOMEUSER:***@" in out  # the target line masks the password


def test_execute_loads_and_passes_resume_on_only_when_asked_and_creates_or_deletes_nothing_by_itself(quick_preflight, hana_env):
    fake, calls = hana_env
    assert _script().main(["--execute", "--schema", "HACKFEST0119"]) == 0
    assert calls == {"read_only": False, "resume": False, "schema": "HACKFEST0119"}
    assert _script().main(["--execute", "--schema", "HACKFEST0119", "--resume"]) == 0
    assert calls["resume"] is True
    assert fake.statements == ['SET SCHEMA "HACKFEST0119"'] * 2  # the script itself issued no DDL and no DELETE


def test_check_target_cannot_be_combined_with_anything_that_writes(quick_preflight, hana_env):
    fake, _ = hana_env
    for extra in ("--execute", "--create", "--truncate"):
        assert _script().main(["--check-target", "--schema", "HACKFEST0119", extra]) == 2
    assert fake.statements == []  # it refused before connecting


def test_a_failed_connection_is_reported_with_a_hint_and_without_the_password(quick_preflight, hana_env, monkeypatch, capsys):
    def refuse(target):
        raise RuntimeError("Connection failed (RTE:[89008]) for user SOMEUSER with password hunter2")
    monkeypatch.setattr("backend.database.engine.build_engine", refuse)
    assert _script().main(["--execute", "--schema", "HACKFEST0119"]) == 2
    out = capsys.readouterr().out
    assert "could not connect" in out and "allowed connections" in out and "hunter2" not in out


def test_the_loader_reads_data_cleaned_only_wherever_it_is_run_from(quick_preflight, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)  # a working directory that has no data/ at all
    script = _script()
    assert script.main([]) == 0  # the defaults are anchored to the repository, not to the directory the command was typed in
    assert script.DATA_DIR == ROOT / "data" / "cleaned" and script.SQL_DIR == ROOT / "db" / "hana" / "core"
    assert script.main(["--data-dir", str(ROOT / "data" / "processed")]) == 2
    assert script.main(["--data-dir", str(ROOT / "data" / "raw")]) == 2
    assert capsys.readouterr().out.count("is not the cleaned layer") == 2
    assert script.main(["--data-dir", str(tmp_path)]) == 2  # a directory that is not a cleaned layer at all
    assert "has no manifest.json" in capsys.readouterr().out
