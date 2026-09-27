"""Phase 21 — SAP BTP / SAP HANA Cloud readiness of the platform layer: service bindings, the database target, the schema.

Nothing here talks to SAP. What it proves is that the *configuration* SAP hands an application — `VCAP_SERVICES` — turns
into the right connection, that secrets stay out of every message, that the schema renders as valid HANA DDL, and that
choosing HANA changes nothing about how the application behaves on SQLite. A live HANA tenant is the one thing missing
(docs/sap-readiness.md says what that leaves unproven).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import event, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError

from backend.database import ddl
from backend.database.engine import DEFAULT_DATABASE_URL, DatabaseTarget, build_engine, env_flag, resolve_database_target
from backend.database.world_state_repository import SqlAlchemyWorldStateRepository, metadata as state_metadata
from backend.sap import btp, hana
from backend.services.world_state import WorldStateStore

PASSWORD = "p@ss:/w#rd?&=%"  # what a generated HANA password can look like: every character that breaks a URL
HANA_CREDENTIALS = {"host": "abc123.hana.trial-us10.hanacloud.ondemand.com", "port": "443", "user": "RSC_RT_USER", "password": PASSWORD,
                    "schema": "RSC_SCHEMA", "hdi_user": "RSC_DT_USER", "hdi_password": "design-time-secret", "certificate": "-----BEGIN CERTIFICATE-----"}


def vcap(**services) -> dict[str, str]:
    """An environment as Cloud Foundry gives one: {label: [instances]}."""
    body = {label: [{"label": label, "name": name, "tags": tags, "credentials": credentials} for name, tags, credentials in instances]
            for label, instances in services.items()}
    return {"VCAP_APPLICATION": "{}", "VCAP_SERVICES": json.dumps(body)}


HANA_ENV = vcap(hana=[("resilientsc-db", ["hana", "database"], HANA_CREDENTIALS)],
                xsuaa=[("resilientsc-uaa", ["xsuaa"], {"url": "https://uaa.example", "clientsecret": "uaa-secret"})])


# --------------------------------------------------------------------------- #
# service bindings
# --------------------------------------------------------------------------- #
def test_bindings_are_read_from_vcap_services():
    bindings = btp.load_bindings(HANA_ENV)
    assert {(b.label, b.name) for b in bindings} == {("hana", "resilientsc-db"), ("xsuaa", "resilientsc-uaa")}
    assert btp.find_binding(label="hana", env=HANA_ENV).credentials["host"].endswith("hanacloud.ondemand.com")
    assert btp.find_binding(tag="database", env=HANA_ENV).name == "resilientsc-db"
    assert btp.find_binding(name="resilientsc-uaa", env=HANA_ENV).label == "xsuaa"
    assert btp.find_binding(label="destination", env=HANA_ENV) is None


def test_no_vcap_means_no_bindings_and_not_on_btp():
    assert btp.load_bindings({}) == [] and btp.find_binding(label="hana", env={}) is None
    assert not btp.running_on_btp({}) and btp.running_on_btp(HANA_ENV)


def test_a_binding_never_prints_its_credentials():
    binding = btp.find_binding(label="hana", env=HANA_ENV)
    assert PASSWORD not in repr(binding) and PASSWORD not in str(binding) and "design-time-secret" not in repr(binding)


def test_two_instances_of_one_service_must_be_told_apart():
    env = vcap(hana=[("db-a", ["hana"], HANA_CREDENTIALS), ("db-b", ["hana"], HANA_CREDENTIALS)])
    with pytest.raises(btp.BTPConfigError, match=r"db-a.*db-b|db-b.*db-a") as caught:
        btp.find_binding(label="hana", env=env)
    assert PASSWORD not in str(caught.value)  # it names the instances, never what is in them
    assert btp.find_binding(label="hana", name="db-b", env=env).name == "db-b"  # naming one resolves it


@pytest.mark.parametrize("raw,message", [
    ('{"hana": [{"credentials": {"password": "SECRETVALUE"', "not valid JSON"),
    ('["not", "an", "object"]', "must be a JSON object"),
    ('{"hana": {"name": "x"}}', "must be a list"),
    ('{"hana": ["nope"]}', "not an object"),
    ('{"hana": [{"credentials": "SECRETVALUE"}]}', "credentials are not an object"),
])
def test_a_malformed_vcap_services_is_a_clear_error_that_leaks_nothing(raw, message):
    with pytest.raises(btp.BTPConfigError, match=message) as caught:
        btp.load_bindings({"VCAP_SERVICES": raw})
    assert "SECRETVALUE" not in str(caught.value) and "SECRETVALUE" not in repr(caught.value.__cause__ or "")


# --------------------------------------------------------------------------- #
# HANA connection
# --------------------------------------------------------------------------- #
def test_the_hana_url_carries_credentials_that_would_break_a_string_url():
    url = hana.hana_url(HANA_CREDENTIALS)
    assert url.drivername == "hana+hdbcli" and url.port == 443 and url.host == HANA_CREDENTIALS["host"]
    assert url.username == "RSC_RT_USER" and url.password == PASSWORD  # the runtime user, not the design-time hdi_user
    assert dict(url.query) == {"encrypt": "true", "sslValidateCertificate": "true"}  # HANA Cloud accepts only encrypted connections
    rendered = url.render_as_string(hide_password=False)
    assert make_url(rendered).password == PASSWORD  # survives a round trip through text, as a DATABASE_URL would
    assert PASSWORD not in url.render_as_string(hide_password=True)


def test_missing_hana_credentials_are_named_not_guessed():
    with pytest.raises(btp.BTPConfigError, match=r"missing credentials: host, password") as caught:
        hana.hana_url({"port": 443, "user": "u"})
    assert "hana binding" in str(caught.value)


# --------------------------------------------------------------------------- #
# which database
# --------------------------------------------------------------------------- #
def test_the_default_is_a_local_sqlite_file():
    target = resolve_database_target(env={})
    assert target.url == DEFAULT_DATABASE_URL and target.schema is None and target.source == "default" and target.is_sqlite


def test_an_explicit_url_beats_the_environment_which_beats_a_bound_hana_instance():
    env = {**HANA_ENV, "DATABASE_URL": "sqlite:///from-env.db"}
    assert resolve_database_target("sqlite:///argument.db", env=env).url == "sqlite:///argument.db"
    assert resolve_database_target(env=env).url == "sqlite:///from-env.db"
    assert resolve_database_target(env=HANA_ENV).source == "BTP binding 'resilientsc-db'"


def test_a_bound_hana_instance_is_used_when_nothing_says_otherwise():
    target = resolve_database_target(env=HANA_ENV)
    assert make_url(target.url).drivername == "hana+hdbcli" and not target.is_sqlite
    assert target.schema == "RSC_SCHEMA"  # the binding names its own schema, so tables are qualified with it
    described = target.describe()
    assert PASSWORD not in described and "***" in described and "RSC_SCHEMA" in described and "resilientsc-db" in described


def test_the_schema_can_be_overridden():
    assert resolve_database_target(env={**HANA_ENV, "DATABASE_SCHEMA": "OTHER"}).schema == "OTHER"
    assert resolve_database_target("sqlite://", "MINE", env=HANA_ENV).schema == "MINE"


@pytest.mark.parametrize("raw,expected", [("true", True), ("1", True), ("YES", True), ("false", False), ("0", False), ("off", False), ("", True)])
def test_boolean_flags(raw, expected):
    assert env_flag("X", True, {"X": raw}) is expected


def test_a_mistyped_flag_is_an_error_not_a_silent_default():
    with pytest.raises(ValueError, match="X must be"):
        env_flag("X", True, {"X": "maybe"})


# --------------------------------------------------------------------------- #
# the repository on that target
# --------------------------------------------------------------------------- #
def test_tables_are_created_by_default_and_not_when_the_database_user_may_not_run_ddl(tmp_path, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'a.db').as_posix()}"
    assert WorldStateStore(SqlAlchemyWorldStateRepository(url)).list_simulations() == []  # created, and works

    cold = SqlAlchemyWorldStateRepository(f"sqlite:///{(tmp_path / 'b.db').as_posix()}", auto_create=False)
    with pytest.raises(OperationalError, match="no such table"):
        cold.list_summaries()  # and it did not quietly create them behind the DBA's back

    monkeypatch.setenv("DATABASE_AUTO_CREATE", "false")
    with pytest.raises(OperationalError, match="no such table"):
        SqlAlchemyWorldStateRepository(f"sqlite:///{(tmp_path / 'c.db').as_posix()}").list_summaries()
    monkeypatch.setenv("DATABASE_AUTO_CREATE", "true")
    assert SqlAlchemyWorldStateRepository(f"sqlite:///{(tmp_path / 'c.db').as_posix()}").list_summaries() == []


def test_a_schema_is_applied_to_every_statement_without_the_tables_knowing(tmp_path):
    """SQLite's ATTACH is a schema: `rsc.world_states` lives in a second database file, which is how this is checked
    without a server. If the translation were dropped from any statement it would hit the main database and fail."""
    main, attached = tmp_path / "main.db", tmp_path / "rsc.db"
    repo = SqlAlchemyWorldStateRepository(f"sqlite:///{main.as_posix()}", schema="rsc", auto_create=False)
    event.listen(repo._engine, "connect", lambda dbapi_conn, _record: dbapi_conn.execute(f"ATTACH DATABASE '{attached.as_posix()}' AS rsc"))
    state_metadata.create_all(repo._engine)

    store = WorldStateStore(repo)
    sim = store.create("SCHEMA_TEST").simulation_id
    store.commit(sim, "event_sensed", {"current_disruptions": [_event()]})
    assert store.get(sim).version == 1 and [c.checkpoint for c in store.history(sim)] == ["simulation_created", "event_sensed"]

    with repo._engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM rsc.world_states")).scalar() == 1
        assert conn.execute(text("SELECT count(*) FROM rsc.world_state_checkpoints")).scalar() == 2
        assert conn.execute(text("SELECT count(*) FROM main.sqlite_master WHERE name LIKE 'world_state%'")).scalar() == 0


def _event():
    from datetime import datetime, timezone
    return {"event_id": "E1", "event_type": "PORT_CLOSURE", "location": "Suez", "severity": "HIGH", "start_date": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
            "estimated_duration": 5, "affected_routes": ["SHA-ROT-SUEZ"], "confidence": 0.9}


def test_a_networked_database_gets_connection_health_checks():
    pytest.importorskip("sqlalchemy_hana")
    engine = build_engine(DatabaseTarget(hana.hana_url(HANA_CREDENTIALS), None, "test"))
    assert engine.pool._pre_ping is True  # HANA Cloud closes idle connections; a pooled one may be dead


# --------------------------------------------------------------------------- #
# the schema as SQL
# --------------------------------------------------------------------------- #
EXPECTED_TABLES = {"world_states", "world_state_checkpoints", "ref_suppliers", "ref_routes", "ref_tariffs", "ref_inventory", "ref_demand_panel", "ref_disruptions"}


def created(statements):
    return [s.split("(", 1)[0].replace("CREATE TABLE", "").strip().strip('"').split(".")[-1].strip('"') for s in statements if s.startswith("CREATE TABLE")]


def test_ddl_covers_every_table_and_creates_a_table_before_one_that_references_it():
    tables = created(ddl.statements("sqlite"))
    assert set(tables) == EXPECTED_TABLES
    assert tables.index("world_states") < tables.index("world_state_checkpoints")


def test_ddl_can_be_limited_to_the_state_or_the_reference_tables():
    assert set(created(ddl.statements("sqlite", include=("state",)))) == {"world_states", "world_state_checkpoints"}
    assert all(t.startswith("ref_") for t in created(ddl.statements("sqlite", include=("reference",))))


def test_ddl_can_be_qualified_with_a_schema():
    statements = ddl.statements("sqlite", schema="RSC")
    assert all('"RSC".' in s or "RSC." in s for s in statements if s.startswith(("CREATE TABLE", "CREATE INDEX")))


def test_ddl_indexes_the_columns_the_repository_filters_on():
    joined = "\n".join(ddl.statements("sqlite", include=("state",)))
    assert "ix_world_states_status" in joined and "ix_world_states_updated_at" in joined and "ix_world_state_checkpoints_simulation_id" in joined


def test_the_ddl_it_generates_actually_runs(tmp_path):
    """Not just that it looks like SQL: a database accepts it, and the application then works on those tables."""
    import sqlite3
    path = tmp_path / "from_ddl.db"
    with sqlite3.connect(path) as conn:
        for statement in ddl.statements("sqlite"):
            conn.execute(statement)
    store = WorldStateStore(SqlAlchemyWorldStateRepository(f"sqlite:///{path.as_posix()}", auto_create=False))
    assert store.get(store.create("FROM_DDL").simulation_id).scenario_type == "FROM_DDL"


def test_an_unknown_dialect_is_refused():
    with pytest.raises(SystemExit, match="unknown dialect"):
        ddl.statements("oracle")


def test_the_command_line_prints_or_writes(tmp_path, capsys):
    assert ddl.main(["--dialect", "sqlite", "--include", "state"]) == 0
    assert "CREATE TABLE world_states" in capsys.readouterr().out
    target = tmp_path / "out" / "schema.sql"
    assert ddl.main(["--dialect", "sqlite", "--write", str(target)]) == 0
    assert "ref_suppliers" in target.read_text(encoding="utf-8")


# --- these need the HANA dialect (pip install -r requirements-sap.txt) ---------------------------------------- #
HANA_SCHEMA_FILE = Path("db/hana/schema.sql")


def test_the_hana_ddl_uses_only_types_hana_has():
    pytest.importorskip("sqlalchemy_hana")
    sql = ddl.render("hana")
    assert "DATETIME" not in sql  # the generic DateTime renders as this, and HANA has no such type
    for expected in ("NVARCHAR(", "NCLOB", "DOUBLE", "TIMESTAMP", "GENERATED BY DEFAULT AS IDENTITY"):
        assert expected in sql
    assert "VARCHAR(" not in sql.replace("NVARCHAR(", "")  # unicode-safe: names a person typed are stored as they were typed


def test_the_checked_in_hana_schema_is_what_the_table_definitions_generate():
    """The file a DBA runs must not drift from the tables the application reads and writes."""
    pytest.importorskip("sqlalchemy_hana")
    assert HANA_SCHEMA_FILE.read_text(encoding="utf-8") == ddl.render("hana"), "regenerate: python -m backend.database.ddl --dialect hana --write db/hana/schema.sql"


# --------------------------------------------------------------------------- #
# which language model
# --------------------------------------------------------------------------- #
class _Fake:
    model = "fake"

    def generate_json(self, **kwargs):  # pragma: no cover - never called here
        raise AssertionError


@pytest.fixture
def custom_provider():
    from backend.agents.sensing import llm
    llm.register_llm_provider("custom", lambda cfg: _Fake(), lambda env: env.get("CUSTOM_LLM_URL") is not None)
    yield
    llm._PROVIDERS.pop("custom")


def test_gemini_is_the_default_provider_and_is_configured_by_its_key(monkeypatch):
    from backend.agents.sensing.llm import GeminiClient, build_llm_from_env, llm_configured, llm_provider_name
    assert llm_provider_name({}) == "gemini" and llm_provider_name({"LLM_PROVIDER": " Gemini "}) == "gemini"
    assert llm_configured({"LLM_API_KEY": "k"}) and not llm_configured({}) and not llm_configured({"LLM_API_KEY": ""})
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "k")
    assert isinstance(build_llm_from_env({}), GeminiClient)


def test_an_unknown_provider_is_a_clear_service_error_that_lists_the_ones_there_are():
    from backend.agents.sensing.llm import LLMUnavailableError, build_llm_from_env, llm_configured
    with pytest.raises(LLMUnavailableError, match=r"'aicore' is not a known provider; known: \['gemini'\]"):
        build_llm_from_env({}, {"LLM_PROVIDER": "aicore"})
    assert not llm_configured({"LLM_PROVIDER": "aicore", "LLM_API_KEY": "k"})  # a key does not make an unknown provider configured


def test_a_registered_provider_is_what_the_sensing_agent_uses(custom_provider, monkeypatch):
    from backend.agents.sensing.agent import SensingAgent
    from backend.agents.sensing.llm import CircuitBreakerLLM, llm_configured
    monkeypatch.setenv("LLM_PROVIDER", "custom")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("CUSTOM_LLM_URL", raising=False)
    agent = SensingAgent()
    assert agent.llm_status()["configured"] is False and not llm_configured()
    monkeypatch.setenv("CUSTOM_LLM_URL", "https://example")
    assert agent.llm_status()["configured"] is True
    client = agent._llm_client()
    assert isinstance(client, CircuitBreakerLLM) and client.model == "fake"  # and it is still behind the circuit breaker


# --------------------------------------------------------------------------- #
# the deployment descriptors
# --------------------------------------------------------------------------- #
# They cannot be deployed here (no BTP subaccount), so what is checked is what can be: that they parse, that they agree with
# each other and with the code, and that the things a deployment must never do are absent.
def _yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_every_name_the_mta_requires_is_something_it_defines():
    mta = _yaml("mta.yaml")
    resources = {r["name"] for r in mta["resources"]}
    provided = {p["name"] for m in mta["modules"] for p in m.get("provides", [])}
    for module in mta["modules"]:
        for required in module.get("requires", []):
            assert required["name"] in resources | provided, f"{module['name']} requires {required['name']}, which nothing defines"


def test_the_approuter_routes_to_the_destination_the_mta_declares():
    mta, app = _yaml("mta.yaml"), _json("approuter/xs-app.json")
    approuter = next(m for m in mta["modules"] if m["type"] == "approuter.nodejs")
    declared = {r["properties"]["name"]: r["properties"] for r in approuter["requires"] if r.get("group") == "destinations"}
    routed = {r["destination"] for r in app["routes"] if "destination" in r}
    assert routed and routed <= set(declared), f"xs-app.json routes to {routed}, the MTA declares {set(declared)}"
    assert all(declared[d]["forwardAuthToken"] is True for d in routed)  # or the API would never see who is calling
    assert all(r["authenticationType"] == "xsuaa" for r in app["routes"])  # nothing is served without signing in
    assert app["routes"][-1]["source"] == "^/(.*)$" and app["routes"][0]["source"].startswith("^/api/")  # the API first: the catch-all would swallow it


def test_the_scopes_xsuaa_grants_are_exactly_the_scopes_the_api_enforces():
    from backend.api.security import SCOPES
    security = _json("xs-security.json")
    declared = {s["name"].removeprefix("$XSAPPNAME.").lower() for s in security["scopes"]}
    assert declared == set(SCOPES), "a scope the API checks but XSUAA cannot grant is a scope nobody can hold (or the reverse)"
    templates = {t["name"]: t for t in security["role-templates"]}
    assert all(ref in {s["name"] for s in security["scopes"]} for t in templates.values() for ref in t["scope-references"])
    assert all(ref.removeprefix("$XSAPPNAME.") in templates for c in security["role-collections"] for ref in c["role-template-references"])
    grants = {n: {r.removeprefix("$XSAPPNAME.").lower() for r in t["scope-references"]} for n, t in templates.items()}
    assert "approve" not in grants["Operator"] and "operate" not in grants["Approver"]  # starting a run and releasing a plan are different roles


def test_the_api_module_is_started_the_way_the_app_expects():
    mta = _yaml("mta.yaml")
    api = next(m for m in mta["modules"] if m["name"] == "resilientsc-api")
    params = api["parameters"]
    assert params["instances"] == 1  # run records and the in-flight guard are per process
    assert params["command"] == "uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT" and params["health-check-http-endpoint"] == "/api/health"
    from backend.api.main import app
    assert "/api/health" in {r.path for r in app.routes}
    ignored = api["build-parameters"]["ignore"]
    assert "backend/" not in ignored and "ml/" not in ignored, "the app and its model artifact must be uploaded"
    assert "backend/tests/" in ignored and ".env" in ignored and ".venv/" in ignored  # and the local secrets and environment must not be
    assert not any(k.upper().endswith(("KEY", "SECRET", "PASSWORD", "TOKEN")) for k in api["properties"]), "a secret does not belong in the descriptor"


def test_the_descriptor_configures_only_values_the_app_accepts():
    from backend.agents.sensing.llm import _PROVIDERS
    from backend.integration import build_publisher_from_env
    props = next(m for m in _yaml("mta.yaml")["modules"] if m["name"] == "resilientsc-api")["properties"]
    assert props["DATA_BACKEND"] in {"csv", "sql"} and props["LLM_PROVIDER"] in _PROVIDERS
    build_publisher_from_env({"EVENTS_BACKEND": props["EVENTS_BACKEND"]})  # raises on a value the app would refuse to start with
    assert props["LOG_FORMAT"] in {"text", "json"} and env_flag("X", True, {"X": props["DATABASE_AUTO_CREATE"]}) in {True, False}


def test_a_cloud_foundry_deployment_installs_the_sap_packages_from_the_one_file_its_buildpack_reads():
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    assert "-r requirements-sap.txt" in requirements
    sap = Path("requirements-sap.txt").read_text(encoding="utf-8")
    for package in ("hdbcli", "sqlalchemy-hana", "PyJWT"):
        assert package in sap


def test_generated_or_secret_files_stay_out_of_version_control():
    ignored = Path(".gitignore").read_text(encoding="utf-8")
    for pattern in (".env", "approuter/resources/", "approuter/node_modules/", "mta_archives/"):
        assert pattern in ignored


# --------------------------------------------------------------------------- #
# the connection check a person runs on a real tenant
# --------------------------------------------------------------------------- #
def _load_check_script():
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("check_sap_connection", Path("scripts/check_sap_connection.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_sap_connection"] = module  # a dataclass looks its own module up by name
    spec.loader.exec_module(module)
    return module


def test_the_connection_check_passes_on_a_working_setup_and_leaves_nothing_behind(tmp_path):
    check = _load_check_script()
    url = f"sqlite:///{(tmp_path / 'check.db').as_posix()}"
    lines: list[str] = []
    code = check.main(["--database-url", url], env={}, out=lines.append)
    text_out = "\n".join(lines)
    assert code == 0, text_out
    assert "PASS  database: connect" in text_out and "PASS  database: world-state tables" in text_out and "0 failed" in text_out
    assert SqlAlchemyWorldStateRepository(url).list_summaries() == []  # the simulation it made is gone


def test_the_connection_check_says_what_is_wrong_and_exits_non_zero(tmp_path):
    check = _load_check_script()
    lines: list[str] = []
    code = check.main(["--database-url", f"sqlite:///{(tmp_path / 'no' / 'such' / 'dir' / 'x.db').as_posix()}"], env={}, out=lines.append)
    assert code == 1 and any(line.startswith("FAIL  database: connect") for line in lines)


def test_the_connection_check_reports_broken_auth_and_events_configuration_by_name(tmp_path):
    check = _load_check_script()
    url = f"sqlite:///{(tmp_path / 'c.db').as_posix()}"
    results = {c.name: c for c in check.run_checks({"AUTH_MODE": "jwt", "EVENTS_BACKEND": "webhook"}, database_url=url)}
    assert results["authentication: config"].status == "FAIL" and "AUTH_JWKS_URL" in results["authentication: config"].detail
    assert results["events: config"].status == "FAIL" and "EVENTS_WEBHOOK_URL" in results["events: config"].detail


def test_the_connection_check_verifies_a_token_when_given_one(tmp_path):
    from backend.tests.test_sap_security import ISSUER, KEY, PREFIX, XSAPP, claims, token
    check = _load_check_script()
    env = {"AUTH_MODE": "jwt", "AUTH_JWKS_URL": "https://idp.example/keys", "AUTH_ISSUER": ISSUER, "AUTH_AUDIENCE": XSAPP, "AUTH_SCOPE_PREFIX": PREFIX}
    url = f"sqlite:///{(tmp_path / 'd.db').as_posix()}"
    without = {c.name: c for c in check.run_checks(env, database_url=url)}
    assert without["authentication: token"].status == "SKIP"
    bad = {c.name: c for c in check.run_checks(env, database_url=url, token="not.a.jwt")}
    assert bad["authentication: token"].status == "FAIL"


def test_the_btp_frontend_build_settings_exist_and_are_not_swallowed_by_the_env_ignore_rule():
    """`.env*` is git-ignored (secrets); the BTP build's two non-secret settings live in one of those files, so it is
    un-ignored by name. Without it a fresh clone builds a UI that calls localhost:8000 from inside BTP, with no error."""
    settings = dict(line.split("=", 1) for line in Path("frontend/.env.btp").read_text(encoding="utf-8").splitlines() if "=" in line and not line.startswith("#"))
    assert settings == {"VITE_API_BASE_URL": "", "VITE_CSRF_TOKEN": "true"}  # same origin, and the Approuter's CSRF token
    assert "!.env.btp" in Path("frontend/.gitignore").read_text(encoding="utf-8").splitlines()
    scripts = _json("frontend/package.json")["scripts"]
    assert "--mode btp" in scripts["build:btp"] and "../approuter/resources" in scripts["build:btp"]
    app = _json("approuter/xs-app.json")
    assert any(r.get("localDir") == "resources" for r in app["routes"])  # the directory build:btp writes into is the one the Approuter serves
