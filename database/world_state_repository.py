"""World-state persistence adapter — architecture.md §2/§5.

`WorldStateRepository` is the interface the rest of the system codes against;
`SqlAlchemyWorldStateRepository` implements it on SQLAlchemy Core, for any
database SQLAlchemy has a dialect for. Which one is `backend/database/engine.py`'s
business: DATABASE_URL, or a bound SAP HANA Cloud instance on BTP, or SQLite.
The schema is one row per simulation with the full state as JSON plus indexed
columns, and it compiles to valid HANA DDL as-is (`python -m backend.database.ddl`).

The repository does no business logic: it stores what it is handed and
enforces exactly one thing, optimistic concurrency — `save` only succeeds if
the row is still at the version the caller read.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from typing import Protocol

from sqlalchemy import Column, ForeignKey, Integer, MetaData, String, Table, Text, delete, insert, select, update
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError

from backend.database.engine import DEFAULT_DATABASE_URL, build_engine, env_flag, resolve_database_target  # noqa: F401 — DEFAULT_DATABASE_URL was defined here
from backend.schemas.world_state import CheckpointRecord, SimulationSummary, WorldState

logger = logging.getLogger("resilientsc.database")


class SimulationNotFoundError(LookupError):
    error_code = "SIMULATION_NOT_FOUND"


class SimulationExistsError(ValueError):
    error_code = "VALIDATION_ERROR"


class ConcurrentModificationError(RuntimeError):
    """The simulation changed between the caller reading it and writing it."""

    error_code = "STATE_CONFLICT"


ACTOR_MAX_LENGTH = 256

metadata = MetaData()

world_states = Table(
    "world_states", metadata,
    Column("simulation_id", String(64), primary_key=True),
    Column("scenario_type", String(64), nullable=False),
    Column("status", String(32), nullable=False, index=True),
    Column("version", Integer, nullable=False),
    Column("created_at", String(40), nullable=False),
    Column("updated_at", String(40), nullable=False, index=True),
    Column("state_json", Text, nullable=False),
    Column("initial_state_json", Text, nullable=False),  # what reset() restores
)

checkpoints = Table(
    "world_state_checkpoints", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("simulation_id", String(64), ForeignKey("world_states.simulation_id"), nullable=False, index=True),
    Column("version", Integer, nullable=False),
    Column("checkpoint", String(64), nullable=False),
    Column("actor", String(ACTOR_MAX_LENGTH), nullable=False),  # an approver's name or e-mail address goes here; HANA enforces the length
    Column("at", String(40), nullable=False),
    Column("changed_fields", Text, nullable=False),
)


def _ts(value: datetime) -> str:
    """Fixed-width ISO-8601 so the string columns sort chronologically."""
    return value.isoformat(timespec="microseconds")


class WorldStateRepository(Protocol):
    def create(self, state: WorldState, *, actor: str) -> None: ...
    def get(self, simulation_id: str) -> WorldState | None: ...
    def get_initial_state(self, simulation_id: str) -> WorldState | None: ...
    def save(self, state: WorldState, *, expected_version: int, checkpoint: str, actor: str, changed_fields: list[str]) -> None: ...
    def list_summaries(self, status: str | None = None, limit: int = 50) -> list[SimulationSummary]: ...
    def history(self, simulation_id: str) -> list[CheckpointRecord]: ...
    def delete(self, simulation_id: str) -> bool: ...


class SqlAlchemyWorldStateRepository:
    def __init__(self, database_url: str | URL | None = None, *, schema: str | None = None, auto_create: bool | None = None):
        """`auto_create` (env DATABASE_AUTO_CREATE, default true) creates the two tables if they are missing. Turn it
        off where the application's database user may not run DDL — an HDI container on HANA Cloud — and create the
        tables from `python -m backend.database.ddl` instead."""
        target = resolve_database_target(database_url, schema)
        self._engine = build_engine(target)
        logger.info("world-state database: %s", target.describe(), extra={"event": "database_target"})
        # a single in-memory connection is not safe to use from two threads at once
        self._lock = threading.RLock()
        if auto_create if auto_create is not None else env_flag("DATABASE_AUTO_CREATE", True):
            metadata.create_all(self._engine)

    def create(self, state: WorldState, *, actor: str) -> None:
        payload = state.model_dump_json()
        with self._lock, self._engine.begin() as conn:
            try:
                conn.execute(insert(world_states).values(
                    simulation_id=state.simulation_id, scenario_type=state.scenario_type, status=state.status.value,
                    version=state.version, created_at=_ts(state.created_at), updated_at=_ts(state.timestamp),
                    state_json=payload, initial_state_json=payload,
                ))
            except IntegrityError as exc:
                raise SimulationExistsError(f"simulation {state.simulation_id!r} already exists") from exc
            self._log(conn, state, "simulation_created", actor, [])

    def get(self, simulation_id: str) -> WorldState | None:
        return self._load("state_json", simulation_id)

    def get_initial_state(self, simulation_id: str) -> WorldState | None:
        return self._load("initial_state_json", simulation_id)

    def _load(self, column: str, simulation_id: str) -> WorldState | None:
        with self._lock, self._engine.connect() as conn:
            row = conn.execute(select(world_states.c[column]).where(world_states.c.simulation_id == simulation_id)).first()
        return WorldState.model_validate_json(row[0]) if row else None

    def save(self, state: WorldState, *, expected_version: int, checkpoint: str, actor: str, changed_fields: list[str]) -> None:
        with self._lock, self._engine.begin() as conn:
            result = conn.execute(
                update(world_states)
                .where(world_states.c.simulation_id == state.simulation_id, world_states.c.version == expected_version)
                .values(status=state.status.value, version=state.version, updated_at=_ts(state.timestamp), state_json=state.model_dump_json())
            )
            if result.rowcount == 0:
                exists = conn.execute(select(world_states.c.version).where(world_states.c.simulation_id == state.simulation_id)).first()
                if exists is None:
                    raise SimulationNotFoundError(f"no simulation {state.simulation_id!r}")
                raise ConcurrentModificationError(
                    f"simulation {state.simulation_id!r} is at version {exists[0]}, not the expected {expected_version}"
                )
            self._log(conn, state, checkpoint, actor, changed_fields)

    @staticmethod
    def _log(conn, state: WorldState, checkpoint: str, actor: str, changed_fields: list[str]) -> None:
        conn.execute(insert(checkpoints).values(
            simulation_id=state.simulation_id, version=state.version, checkpoint=checkpoint, actor=actor,
            at=_ts(state.timestamp), changed_fields=json.dumps(changed_fields),
        ))

    def list_summaries(self, status: str | None = None, limit: int = 50) -> list[SimulationSummary]:
        query = select(
            world_states.c.simulation_id, world_states.c.scenario_type, world_states.c.status,
            world_states.c.version, world_states.c.created_at, world_states.c.updated_at,
        ).order_by(world_states.c.updated_at.desc(), world_states.c.simulation_id).limit(limit)
        if status is not None:
            query = query.where(world_states.c.status == status)
        with self._lock, self._engine.connect() as conn:
            rows = conn.execute(query).all()
        return [SimulationSummary(simulation_id=r[0], scenario_type=r[1], status=r[2], version=r[3], created_at=r[4], updated_at=r[5]) for r in rows]

    def history(self, simulation_id: str) -> list[CheckpointRecord]:
        with self._lock, self._engine.connect() as conn:
            rows = conn.execute(select(checkpoints).where(checkpoints.c.simulation_id == simulation_id).order_by(checkpoints.c.id)).all()
        return [
            CheckpointRecord(simulation_id=r.simulation_id, version=r.version, checkpoint=r.checkpoint, actor=r.actor,
                             at=r.at, changed_fields=json.loads(r.changed_fields))
            for r in rows
        ]

    def delete(self, simulation_id: str) -> bool:
        with self._lock, self._engine.begin() as conn:
            conn.execute(delete(checkpoints).where(checkpoints.c.simulation_id == simulation_id))
            return conn.execute(delete(world_states).where(world_states.c.simulation_id == simulation_id)).rowcount > 0
