"""Phase 21 — the reference data has a seam: the same agents run on processed CSV files or on database tables.

What is proved here is not that a table can be read, but that *nothing downstream can tell the difference*: every
dataset comes back column-for-column and value-for-value the same, and the plan the optimizer builds from tables is the
plan it builds from files. A HANA deployment rests on exactly that.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from backend.agents.logistics import tools as logistics_tools
from backend.api import readiness
from backend.api.errors import install_error_handlers
from backend.data import (
    DATASETS,
    CsvDatasetRepository,
    DatasetUnavailableError,
    SqlDatasetRepository,
    datasets_from_env,
    get_datasets,
    set_datasets,
)
from backend.data.loader import load_csvs_into
from backend.database.engine import DatabaseTarget, build_engine
from backend.integration import NullPublisher
from backend.simulation.comparison import WorldSpec, solve_world

DATA_DIR = Path("data/processed")
pytestmark = pytest.mark.skipif(not (DATA_DIR / "suppliers.csv").exists(), reason="processed datasets are not built (Phases 3-5)")


def sqlite_engine(path: Path):
    return build_engine(DatabaseTarget(f"sqlite:///{path.as_posix()}", None, "test"))


@pytest.fixture(scope="module")
def loaded_engine(tmp_path_factory):
    engine = sqlite_engine(tmp_path_factory.mktemp("refdata") / "ref.db")
    counts = load_csvs_into(engine, str(DATA_DIR))
    assert set(counts) == set(DATASETS)
    return engine


@pytest.fixture(autouse=True)
def restore_default_datasets():
    yield
    set_datasets(None)  # whatever a test installed, the next test starts from the environment's choice — and empty caches


def sorted_like_sql(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """A database has no row order; the SQL repository asks for the dataset's `order_by`, so compare against that."""
    return df.sort_values(list(DATASETS[name].order_by), kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# the contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(DATASETS))
def test_every_dataset_column_the_contract_names_is_in_the_csv(name):
    header = pd.read_csv(DATA_DIR / DATASETS[name].csv, nrows=0).columns
    assert set(DATASETS[name].column_names) <= set(header)


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_a_database_table_returns_exactly_what_the_csv_does(loaded_engine, name):
    spec = DATASETS[name]
    from_csv = CsvDatasetRepository(DATA_DIR).load(name)
    from_sql = SqlDatasetRepository(loaded_engine).load(name)

    assert list(from_sql.columns) == list(from_csv.columns) == list(spec.column_names)
    assert len(from_sql) == len(from_csv)
    expected = sorted_like_sql(from_csv.assign(**({"product_id": from_csv["product_id"].astype(str)} if "product_id" in from_csv else {})), name)
    # dtypes included: a column that came back as text where the files give a number would break arithmetic downstream
    pd.testing.assert_frame_equal(from_sql, expected, check_dtype=True)


def test_a_subset_of_columns_comes_back_in_the_order_asked_for(loaded_engine):
    columns = ["split", "date", "product_id"]
    for repository in (CsvDatasetRepository(DATA_DIR), SqlDatasetRepository(loaded_engine)):
        assert list(repository.load("demand_panel", columns).columns) == columns


def test_an_unknown_dataset_or_column_is_an_error_on_every_backend(loaded_engine):
    for repository in (CsvDatasetRepository(DATA_DIR), SqlDatasetRepository(loaded_engine)):
        with pytest.raises(KeyError, match="unknown dataset"):
            repository.load("nope")
        with pytest.raises(KeyError, match="no column"):
            repository.load("routes", ["route_id", "colour"])  # a column only the CSV happens to have would be a trap


# --------------------------------------------------------------------------- #
# the point of the seam: the same decisions
# --------------------------------------------------------------------------- #
SUEZ = frozenset({"SHA-ROT-SUEZ", "MUM-ROT-SUEZ", "SIN-ROT-SUEZ", "CHE-ROT-SUEZ"})


@pytest.mark.parametrize("product,world", [
    ("22197", WorldSpec()),
    ("84077", WorldSpec()),
    ("22197", WorldSpec(disrupted_routes=SUEZ)),
    ("22197", WorldSpec(disrupted_suppliers=frozenset({"S007"}))),
])
def test_the_plan_built_from_database_tables_is_the_plan_built_from_files(loaded_engine, product, world):
    set_datasets(CsvDatasetRepository(DATA_DIR))
    from_files = solve_world(product, None, world).solution

    set_datasets(SqlDatasetRepository(loaded_engine))
    from_tables = solve_world(product, None, world).solution

    assert from_files.status == from_tables.status
    assert from_files.status == "OPTIMAL" or product == "23166"
    assert from_files.objective_value == from_tables.objective_value
    assert from_files.model_dump(mode="json", exclude={"solver"}) == from_tables.model_dump(mode="json", exclude={"solver"})  # solver: timings


def test_switching_the_repository_empties_what_was_cached_from_the_old_one():
    real = logistics_tools.get_routes()
    assert len(real) > 1

    class OneRoute(CsvDatasetRepository):
        def load(self, name, columns=None):
            df = super().load(name, columns)
            return df.head(1) if name == "routes" else df

    set_datasets(OneRoute(DATA_DIR))
    assert len(logistics_tools.get_routes()) == 1  # a stale cache would still say len(real)
    set_datasets(None)
    assert len(logistics_tools.get_routes()) == len(real)


# --------------------------------------------------------------------------- #
# the loader
# --------------------------------------------------------------------------- #
def test_loading_twice_leaves_the_same_rows_as_loading_once(tmp_path):
    engine = sqlite_engine(tmp_path / "twice.db")
    first = load_csvs_into(engine, str(DATA_DIR), names=["suppliers", "routes"])
    second = load_csvs_into(engine, str(DATA_DIR), names=["suppliers", "routes"])
    assert first == second
    repository = SqlDatasetRepository(engine)
    assert len(repository.load("suppliers")) == first["suppliers"]


def test_a_value_too_long_for_its_column_is_refused_not_truncated(tmp_path):
    (tmp_path / "suppliers.csv").write_text(
        ",".join(DATASETS["suppliers"].column_names) + "\n" + ",".join(["S1", "x" * 500, "China", "1", "1", "1.0", "1", "0.5", "LOW", "ACTIVE", "test"]) + "\n")
    engine = sqlite_engine(tmp_path / "long.db")
    with pytest.raises(ValueError, match=r"supplier_name.*500 characters.*allows 128"):
        load_csvs_into(engine, str(tmp_path), names=["suppliers"])
    assert SqlDatasetRepository(engine).missing()  # and nothing was created half-loaded


def test_a_failed_load_leaves_the_previous_contents(tmp_path):
    engine = sqlite_engine(tmp_path / "keep.db")
    load_csvs_into(engine, str(DATA_DIR), names=["routes"])
    before = SqlDatasetRepository(engine).load("routes")
    (tmp_path / "routes.csv").write_text(
        ",".join(DATASETS["routes"].column_names) + "\n" + ",".join(["R" * 200] + ["x"] * 9) + "\n")
    with pytest.raises(ValueError):
        load_csvs_into(engine, str(tmp_path), names=["routes"])
    pd.testing.assert_frame_equal(SqlDatasetRepository(engine).load("routes"), before)


def test_an_unknown_dataset_name_is_refused(tmp_path):
    with pytest.raises(KeyError, match="unknown dataset"):
        load_csvs_into(sqlite_engine(tmp_path / "x.db"), str(DATA_DIR), names=["suppliers", "wat"])


# --------------------------------------------------------------------------- #
# absent data, said plainly
# --------------------------------------------------------------------------- #
def test_a_missing_table_is_dataset_unavailable_not_a_database_outage(tmp_path):
    repository = SqlDatasetRepository(sqlite_engine(tmp_path / "empty.db"))
    with pytest.raises(DatasetUnavailableError, match=r"ref_suppliers"):
        repository.load("suppliers")
    assert repository.missing() == [f"table {spec.table_name}" for spec in DATASETS.values()]


def test_a_database_that_cannot_be_reached_is_still_an_operational_error(tmp_path):
    repository = SqlDatasetRepository(sqlite_engine(tmp_path / "no" / "such" / "dir" / "x.db"))
    with pytest.raises(OperationalError):
        repository.load("suppliers")


def test_a_missing_csv_is_dataset_unavailable(tmp_path):
    with pytest.raises(DatasetUnavailableError, match="suppliers"):
        CsvDatasetRepository(tmp_path).load("suppliers")
    assert len(CsvDatasetRepository(tmp_path).missing()) == len(DATASETS)


def test_the_api_turns_a_missing_dataset_into_a_503_that_names_it():
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/boom")
    def boom():
        raise DatasetUnavailableError("dataset 'suppliers': table ref_suppliers does not exist")

    response = TestClient(app, raise_server_exceptions=False).get("/boom")
    error = response.json()["error"]
    assert response.status_code == 503 and error["error_code"] == "DATASET_UNAVAILABLE"
    assert "ref_suppliers" in error["message"] and "load_reference_data" in error["recovery"]


def fake_ctx():
    sensing = SimpleNamespace(llm_status=lambda: {"configured": False, "circuit": "closed"})
    return SimpleNamespace(store=SimpleNamespace(list_simulations=lambda limit=1: []), orchestrator=SimpleNamespace(sensing=sensing), events=NullPublisher())


def datasets_check(ctx=None):
    return next(c for c in readiness.run_checks(ctx or fake_ctx()) if c["name"] == "datasets")


def test_readiness_reports_missing_tables_and_how_to_fix_it(tmp_path):
    set_datasets(SqlDatasetRepository(sqlite_engine(tmp_path / "ready.db"), description="sql:test"))
    check = datasets_check()
    assert not check["ok"] and check["required"]
    assert "ref_suppliers" in check["detail"] and "scripts/load_reference_data.py" in check["detail"]


def test_readiness_is_satisfied_once_the_tables_are_loaded(loaded_engine):
    set_datasets(SqlDatasetRepository(loaded_engine, description="sql:test"))
    check = datasets_check()
    assert check["ok"], check["detail"]
    assert "sql:test" in check["detail"]


def test_readiness_says_not_ready_rather_than_crashing_when_the_dataset_database_is_unreachable(tmp_path):
    set_datasets(SqlDatasetRepository(sqlite_engine(tmp_path / "no" / "such" / "dir" / "x.db"), description="sql:down"))
    check = datasets_check()
    assert not check["ok"] and "sql:down" in check["detail"]


def test_readiness_for_files_keeps_saying_what_it_always_said(tmp_path):
    set_datasets(CsvDatasetRepository(tmp_path))
    check = datasets_check()
    assert not check["ok"] and "run the Phase 3-5 pipelines" in check["detail"]


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def test_the_default_is_the_processed_csv_files():
    repository = datasets_from_env({})
    assert isinstance(repository, CsvDatasetRepository) and repository.describe() == "csv:data/processed"
    assert get_datasets().describe().startswith("csv:")


def test_data_dir_moves_the_csv_backend():
    assert datasets_from_env({"DATA_DIR": "elsewhere"}).describe() == "csv:elsewhere"


def test_the_sql_backend_reads_its_own_database_url_and_says_where_it_reads_from(tmp_path):
    url = f"sqlite:///{(tmp_path / 'own.db').as_posix()}"
    repository = datasets_from_env({"DATA_BACKEND": "sql", "DATA_DATABASE_URL": url, "DATABASE_URL": "sqlite:///should-not-be-used.db"})
    assert isinstance(repository, SqlDatasetRepository)
    assert "own.db" in repository.describe() and "DATA_DATABASE_URL" in repository.describe()


def test_the_sql_backend_defaults_to_the_world_state_database(tmp_path):
    url = f"sqlite:///{(tmp_path / 'shared.db').as_posix()}"
    engine = sqlite_engine(tmp_path / "shared.db")
    load_csvs_into(engine, str(DATA_DIR), names=["routes"])
    repository = datasets_from_env({"DATA_BACKEND": "sql", "DATABASE_URL": url})
    assert len(repository.load("routes")) == 8


def test_a_mistyped_backend_is_an_error_not_a_silent_default():
    with pytest.raises(ValueError, match="DATA_BACKEND"):
        datasets_from_env({"DATA_BACKEND": "hana"})
