"""Which database, and how to talk to it.

`resolve_database_target` decides *which* database — in this order, first that applies:

    1. the argument the caller passed (tests, scripts)
    2. DATABASE_URL                       any SQLAlchemy URL: sqlite:///..., hana+hdbcli://...
    3. a bound SAP HANA Cloud instance    VCAP_SERVICES, label `hana` (see backend/sap/btp.py)
    4. ./resilientsc.db                   the local default

`build_engine` decides *how*: SQLite needs its threading options, every networked database needs connections
checked before use (HANA Cloud closes idle ones), and an optional schema is applied to every table without the table
definitions knowing about it. Nothing else in the code base contains an `if dialect == ...`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine, make_url
from sqlalchemy.pool import StaticPool

from backend.sap import btp, hana

DEFAULT_DATABASE_URL = "sqlite:///./resilientsc.db"
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class DatabaseTarget:
    url: str | URL
    schema: str | None
    source: str  # where the answer came from, for the log line an operator reads first

    @property
    def is_sqlite(self) -> bool:
        return make_url(self.url).get_backend_name() == "sqlite"

    def describe(self) -> str:
        """Safe to log: the password is masked."""
        text = make_url(self.url).render_as_string(hide_password=True)
        return f"{text}{f' schema={self.schema}' if self.schema else ''} (from {self.source})"


def env_flag(name: str, default: bool, env: Mapping[str, str] | None = None) -> bool:
    raw = (os.environ if env is None else env).get(name, "").strip().lower()
    if not raw:
        return default
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ValueError(f"{name} must be one of true/false, not {raw!r}")


def resolve_database_target(url: str | URL | None = None, schema: str | None = None, env: Mapping[str, str] | None = None) -> DatabaseTarget:
    env = os.environ if env is None else env
    schema = schema or env.get("DATABASE_SCHEMA") or None
    if url:
        return DatabaseTarget(url, schema, "argument")
    if env.get("DATABASE_URL"):
        return DatabaseTarget(env["DATABASE_URL"], schema, "DATABASE_URL")
    binding = btp.find_binding(label="hana", env=env)
    if binding is not None:
        # the binding names its own schema: qualifying every table with it is what makes this work for both a plain
        # schema and an HDI container, whatever the runtime user's default schema happens to be
        return DatabaseTarget(hana.hana_url(binding.credentials), schema or binding.credentials.get("schema") or None, f"BTP binding {binding.name!r}")
    return DatabaseTarget(DEFAULT_DATABASE_URL, schema, "default")


def build_engine(target: DatabaseTarget) -> Engine:
    options: dict = {}
    if target.is_sqlite:
        options["connect_args"] = {"check_same_thread": False, "timeout": 30}
        if str(target.url) in ("sqlite://", "sqlite:///:memory:"):
            options["poolclass"] = StaticPool  # one shared connection, or every checkout would see a fresh empty DB
    else:
        options["pool_pre_ping"] = True  # a networked database drops idle connections; find out before using one, not during
    engine = create_engine(target.url, **options)
    if target.schema:
        engine = engine.execution_options(schema_translate_map={None: target.schema})
    return engine
