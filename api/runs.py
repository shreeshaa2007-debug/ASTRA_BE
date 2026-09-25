"""Background runs. `POST /api/simulations/{id}/run` returns at once (api-plan.md:
"async — kicks off the run; poll status"), so the orchestrator runs on a worker
thread and this registry remembers how each simulation's latest run went.

What it guards: one run per simulation at a time. The orchestrator only moves a
simulation out of CREATED after sensing (an LLM call, seconds), so without this a
second POST in that window would pass the state check and start a duplicate run.

What it does not do: survive a restart. Records are in memory; the *state* is
durable, so after a restart a simulation left RUNNING is reported as `stalled`
(status is RUNNING but no run is registered here), not silently as "in progress".
"""
from __future__ import annotations

import contextvars
import logging
import threading
import time
import uuid
from concurrent.futures import Executor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from backend.api.errors import ApiError
from backend.monitoring import bind
from backend.monitoring.metrics import metrics
from backend.monitoring.settings import load_settings

logger = logging.getLogger("resilientsc.api")


@dataclass
class RunRecord:
    run_id: str
    simulation_id: str
    started_at: datetime
    finished_at: datetime | None = None
    state: str = "RUNNING"  # RUNNING | FINISHED | CRASHED
    outcome: dict[str, Any] | None = None  # the orchestrator's RunOutcome, minus the state (fetch that separately)
    error: str | None = None  # set only if the run raised something the orchestrator does not record itself
    overdue_seconds: float = 120.0

    @property
    def elapsed_ms(self) -> float:
        end = self.finished_at or datetime.now(timezone.utc)
        return round((end - self.started_at).total_seconds() * 1000, 1)

    @property
    def overdue(self) -> bool:
        """Still running past the configured limit. Nothing kills it — a worker thread cannot be interrupted safely —
        but a person or an alert can see that it is stuck."""
        return self.state == "RUNNING" and self.elapsed_ms > self.overdue_seconds * 1000

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id, "simulation_id": self.simulation_id, "state": self.state,
            "started_at": self.started_at.isoformat(), "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "elapsed_ms": self.elapsed_ms, "overdue": self.overdue, "outcome": self.outcome, "error": self.error,
        }


class RunRegistry:
    def __init__(self, executor: Executor | None = None, overdue_seconds: float | None = None):
        """`executor=None` runs the work inline, on the calling thread — for
        tests that need a run to be finished when the POST returns."""
        self._executor = executor
        self._overdue_seconds = overdue_seconds if overdue_seconds is not None else float(load_settings()["runs"]["overdue_seconds"])
        self._lock = threading.Lock()
        self._latest: dict[str, RunRecord] = {}
        self._active: set[str] = set()

    def start(self, simulation_id: str, work: Callable[[], Any]) -> RunRecord:
        with self._lock:
            if simulation_id in self._active:
                raise ApiError(409, "RUN_IN_PROGRESS", f"a run is already in progress for simulation {simulation_id!r}")
            record = RunRecord(uuid.uuid4().hex[:12], simulation_id, datetime.now(timezone.utc), overdue_seconds=self._overdue_seconds)
            self._latest[simulation_id] = record
            self._active.add(simulation_id)

        def job() -> None:
            final = "FINISHED"
            started = time.perf_counter()
            with bind(simulation_id=simulation_id, run_id=record.run_id):  # every log line the run makes carries these
                metrics.set_gauge("runs_in_flight", self.in_flight())
                logger.info("run %s started", record.run_id, extra={"event": "run_started"})
                try:
                    outcome = work()
                    record.outcome = outcome.model_dump(mode="json", exclude={"state"})
                except Exception as exc:  # noqa: BLE001 — the orchestrator records its own failures; this is what escaped it
                    logger.exception("run %s for %s crashed", record.run_id, simulation_id, extra={"event": "run_crashed"})
                    record.error, final = f"{type(exc).__name__}: {exc}", "CRASHED"
                    metrics.inc("runs_total", {"outcome": "CRASHED"})
                finally:
                    record.finished_at = datetime.now(timezone.utc)
                    record.state = final  # last, so a reader that sees FINISHED also sees the outcome
                    with self._lock:
                        self._active.discard(simulation_id)
                    duration_ms = round((time.perf_counter() - started) * 1000, 1)
                    metrics.observe("run_duration_ms", duration_ms)
                    metrics.set_gauge("runs_in_flight", self.in_flight())
                    logger.info("run %s %s in %.0f ms", record.run_id, final.lower(), duration_ms,
                                extra={"event": "run_finished", "state": final, "outcome": (record.outcome or {}).get("outcome"), "duration_ms": duration_ms})

        # a worker thread does not inherit the request's context variables; hand it a copy
        ctx = contextvars.copy_context()
        if self._executor is None:
            ctx.run(job)
        else:
            try:
                self._executor.submit(lambda: ctx.run(job))
            except Exception:
                with self._lock:
                    self._active.discard(simulation_id)
                raise
        return record

    def latest(self, simulation_id: str) -> RunRecord | None:
        return self._latest.get(simulation_id)

    def forget(self, simulation_id: str) -> None:
        """Drop a finished run's record (a reset makes it about a state that no longer exists). A run still in flight is kept."""
        with self._lock:
            if simulation_id not in self._active:
                self._latest.pop(simulation_id, None)

    def in_flight(self) -> int:
        with self._lock:
            return len(self._active)

    def is_running(self, simulation_id: str) -> bool:
        with self._lock:
            return simulation_id in self._active
