"""The API's shared services, built once and handed to handlers through
dependency injection — which is what lets tests substitute a fake LLM, an
in-memory database and synchronous runs without touching a handler."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from fastapi import Request

from backend.agents.sensing.agent import SensingAgent
from backend.api.runs import RunRegistry
from backend.database.world_state_repository import SqlAlchemyWorldStateRepository
from backend.orchestration import Orchestrator
from backend.services.world_state import WorldStateStore


@dataclass
class AppContext:
    store: WorldStateStore
    orchestrator: Orchestrator
    runs: RunRegistry
    shutdown: Callable[[], None] = lambda: None


def build_default_context(*, compliance_rules: dict | None = None, run_inline: bool = False, max_workers: int = 4) -> AppContext:
    """The production wiring: the database from DATABASE_URL, the LLM from the
    environment (built lazily by the Sensing Agent, so a missing key surfaces as
    a clean LLM_UNAVAILABLE on the first run, not a crash at startup)."""
    store = WorldStateStore(SqlAlchemyWorldStateRepository())
    orchestrator = Orchestrator(store, SensingAgent(), compliance_rules=compliance_rules)
    executor = None if run_inline else ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="run")
    return AppContext(store, orchestrator, RunRegistry(executor), (lambda: executor.shutdown(wait=True, cancel_futures=True)) if executor else (lambda: None))


def get_ctx(request: Request) -> AppContext:
    app = request.app
    if app.state.ctx is None:
        with app.state.ctx_lock:
            if app.state.ctx is None:
                app.state.ctx = app.state.ctx_factory()
    return app.state.ctx


def make_state(app, ctx_factory: Callable[[], AppContext]) -> None:
    app.state.ctx, app.state.ctx_factory, app.state.ctx_lock = None, ctx_factory, threading.Lock()
