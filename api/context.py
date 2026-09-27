"""The API's shared services, built once and handed to handlers through
dependency injection — which is what lets tests substitute a fake LLM, an
in-memory database and synchronous runs without touching a handler."""
from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from fastapi import Request

logger = logging.getLogger("resilientsc.api")

from backend.agents.sensing.agent import SensingAgent
from backend.api.runs import RunRegistry
from backend.database.world_state_repository import SqlAlchemyWorldStateRepository
from backend.integration import EventPublisher, NullPublisher, StoreEventBridge, build_publisher_from_env, checkpoints_from_env
from backend.sap.sbpa_bridge import SBPAWebhookPublisher
from backend.sap.approval_portal import ApprovalPortalPublisher, get_store as get_approval_store
from backend.orchestration import Orchestrator
from backend.services.world_state import WorldStateStore


@dataclass
class AppContext:
    store: WorldStateStore
    orchestrator: Orchestrator
    runs: RunRegistry
    shutdown: Callable[[], None] = lambda: None
    events: EventPublisher = field(default_factory=NullPublisher)  # what the outside world is told; see backend/integration


def build_default_context(*, compliance_rules: dict | None = None, run_inline: bool = False, max_workers: int = 4) -> AppContext:
    """The production wiring: the database from DATABASE_URL, the LLM from the
    environment (built lazily by the Sensing Agent, so a missing key surfaces as
    a clean LLM_UNAVAILABLE on the first run, not a crash at startup)."""
    store = WorldStateStore(SqlAlchemyWorldStateRepository())
    events = build_publisher_from_env()  # EVENTS_BACKEND: none by default, so nothing leaves the process unless asked
    if not isinstance(events, NullPublisher):
        store.add_listener(StoreEventBridge(events, source=os.environ.get("EVENTS_SOURCE") or "urn:resilientsc",
                                            checkpoints=checkpoints_from_env(os.environ.get("EVENTS_CHECKPOINTS"))))

    # Human-approval workflow — prefer SBPA if configured, otherwise use the built-in Approval Portal
    sbpa = SBPAWebhookPublisher.from_env()
    if sbpa is not None:
        # SAP Build Process Automation (requires SBPA_WEBHOOK_URL in .env)
        store.add_listener(StoreEventBridge(sbpa, source=os.environ.get("EVENTS_SOURCE") or "urn:resilientsc",
                                            checkpoints=("approval_requested",)))
        logger.info("approval: using SAP Build Process Automation")
    else:
        # Built-in Approval Portal at /approval (no SAP BTP required)
        store.add_listener(StoreEventBridge(ApprovalPortalPublisher(), source=os.environ.get("EVENTS_SOURCE") or "urn:resilientsc",
                                            checkpoints=("approval_requested",)))
        logger.info("approval: using built-in Approval Portal at /approval (set SBPA_WEBHOOK_URL to use SAP BTP instead)")

    orchestrator = Orchestrator(store, SensingAgent(), compliance_rules=compliance_rules)
    executor = None if run_inline else ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="run")

    def shutdown() -> None:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)  # in-flight runs finish first, so their last events are queued...
        events.close()  # ...and then sent, before the process exits

    return AppContext(store, orchestrator, RunRegistry(executor), shutdown, events)


def get_ctx(request: Request) -> AppContext:
    app = request.app
    if app.state.ctx is None:
        with app.state.ctx_lock:
            if app.state.ctx is None:
                app.state.ctx = app.state.ctx_factory()
    return app.state.ctx


def make_state(app, ctx_factory: Callable[[], AppContext]) -> None:
    app.state.ctx, app.state.ctx_factory, app.state.ctx_lock = None, ctx_factory, threading.Lock()
