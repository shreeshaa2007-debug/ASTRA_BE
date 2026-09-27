"""WorldStateStore — create / get / update / reset the shared world state
(brief §14), isolated per simulation_id.

Reads hand out copies: `get()` deserializes a fresh WorldState every call, so
nothing an agent does to the object it was given can reach the stored state.
The only write path is `commit()`, which requires a named checkpoint
(checkpoints.py) — that is the concrete mechanism behind "agents recommend,
only the orchestrator writes".
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from pydantic import ValidationError

from backend.database.world_state_repository import WorldStateRepository
from backend.monitoring.metrics import metrics
from backend.schemas.world_state import CheckpointRecord, SimulationStatus, SimulationSummary, WorldState
from backend.services.world_state.baseline import Baseline, load_baseline
from backend.services.world_state.checkpoints import CHECKPOINTS
from backend.services.world_state.errors import (
    CheckpointViolationError,
    ConcurrentModificationError,
    InvalidTransitionError,
    SimulationNotFoundError,
    UnknownCheckpointError,
)

DEFAULT_ACTOR = "orchestrator"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


logger = logging.getLogger("resilientsc.world_state")


@dataclass(frozen=True)
class StateChange:
    """What a listener is told after a change is durable: which checkpoint, the state it produced, and who did it."""

    checkpoint: str
    state: WorldState  # a private copy per listener: none can alter what the next one sees
    actor: str


StateListener = Callable[[StateChange], None]


class WorldStateStore:
    def __init__(
        self,
        repository: WorldStateRepository,
        *,
        baseline_loader: Callable[[], Baseline] = load_baseline,
        clock: Callable[[], datetime] = utc_now,
        listeners: Iterable[StateListener] = (),
    ):
        self._repo = repository
        self._baseline_loader = baseline_loader
        self._clock = clock
        self._listeners: list[StateListener] = list(listeners)

    def add_listener(self, listener: StateListener) -> None:
        """Be told about every change once it is saved (the hook the SAP Integration event bridge uses). A listener
        runs after the commit, so it can neither veto nor undo it; one that raises is logged and counted and does not
        affect the commit, the caller, or the other listeners."""
        self._listeners.append(listener)

    # ---- CREATE ---------------------------------------------------------- #
    def create(self, scenario_type: str, *, simulation_id: str | None = None, baseline: Baseline | None = None, actor: str = DEFAULT_ACTOR) -> WorldState:
        """New isolated simulation at the baseline. The starting state is kept
        so reset() returns to exactly this point, not to whatever the data
        files say later."""
        if not scenario_type.strip():
            raise ValueError("scenario_type must not be empty")
        baseline = baseline if baseline is not None else self._baseline_loader()
        now = self._clock()
        state = WorldState(
            simulation_id=simulation_id or f"sim-{uuid.uuid4().hex[:12]}",
            scenario_type=scenario_type,
            created_at=now,
            timestamp=now,
            route_status=dict(baseline.route_status),
            supplier_status=dict(baseline.supplier_status),
            tariffs=dict(baseline.tariffs),
        )
        self._repo.create(state, actor=actor)
        self._record("simulation_created", state, actor)
        return self.get(state.simulation_id)

    def _record(self, checkpoint: str, state: WorldState, actor: str) -> None:
        """Every change to a simulation is a checkpoint; each leaves a log line and a count, so the log is the audit trail's twin."""
        metrics.inc("checkpoints_total", {"checkpoint": checkpoint})
        logger.info("checkpoint %s -> %s (v%d, by %s)", checkpoint, state.status.value, state.version, actor,
                    extra={"event": "checkpoint", "simulation_id": state.simulation_id, "checkpoint": checkpoint, "status": state.status.value,
                           "version": state.version, "actor": actor})
        for listener in list(self._listeners):
            try:
                listener(StateChange(checkpoint, state.model_copy(deep=True), actor))
            except Exception:  # noqa: BLE001 — the change is already saved; nothing a listener does may un-do or fail it
                metrics.inc("state_listener_errors_total", {"checkpoint": checkpoint})
                logger.exception("a state listener failed on checkpoint %s", checkpoint,
                                 extra={"event": "state_listener_error", "simulation_id": state.simulation_id, "checkpoint": checkpoint})

    # ---- GET ------------------------------------------------------------- #
    def get(self, simulation_id: str) -> WorldState:
        state = self._repo.get(simulation_id)
        if state is None:
            raise SimulationNotFoundError(f"no simulation {simulation_id!r}")
        return state

    def list_simulations(self, status: SimulationStatus | None = None, limit: int = 50) -> list[SimulationSummary]:
        return self._repo.list_summaries(status.value if status else None, limit)

    def history(self, simulation_id: str) -> list[CheckpointRecord]:
        records = self._repo.history(simulation_id)
        if not records:  # every created simulation has at least its creation entry
            raise SimulationNotFoundError(f"no simulation {simulation_id!r}")
        return records

    # ---- UPDATE ---------------------------------------------------------- #
    def commit(
        self,
        simulation_id: str,
        checkpoint: str,
        changes: dict[str, Any],
        *,
        actor: str = DEFAULT_ACTOR,
        expected_version: int | None = None,
    ) -> WorldState:
        """Apply `changes` at a named checkpoint. Rejected — nothing written —
        if the checkpoint is unknown, touches a field it doesn't own, isn't
        legal from the simulation's current status, carries invalid values, or
        would leave the state breaking one of the checkpoint's invariants.
        Pass `expected_version` (the version you read) to fail rather than
        overwrite if another writer got there first.
        """
        spec = CHECKPOINTS.get(checkpoint)
        if spec is None:
            raise UnknownCheckpointError(f"unknown checkpoint {checkpoint!r}; known: {sorted(CHECKPOINTS)}")
        if not changes:
            raise CheckpointViolationError(f"checkpoint {checkpoint!r} was given no changes")
        illegal = sorted(set(changes) - spec.allowed_fields)
        if illegal:
            raise CheckpointViolationError(f"checkpoint {checkpoint!r} may only change {sorted(spec.allowed_fields)}, not {illegal}")

        before = self.get(simulation_id)
        if expected_version is not None and before.version != expected_version:
            raise ConcurrentModificationError(f"simulation {simulation_id!r} is at version {before.version}, not the expected {expected_version}")
        if before.status not in spec.from_statuses:
            raise InvalidTransitionError(
                f"checkpoint {checkpoint!r} is not allowed while the simulation is {before.status.value} "
                f"(allowed from: {sorted(s.value for s in spec.from_statuses)})"
            )

        merged = {name: getattr(before, name) for name in WorldState.model_fields}
        merged.update(changes)
        merged.update(status=spec.to_status, version=before.version + 1, timestamp=self._clock())
        try:
            # through JSON, so what is saved is exactly what a later get() will read back
            after = WorldState.model_validate_json(WorldState.model_validate(merged).model_dump_json())
        except ValidationError as exc:
            raise CheckpointViolationError(f"invalid values for {sorted(changes)} at checkpoint {checkpoint!r}: {exc}") from exc

        problem = spec.guard(before, after)
        if problem:
            raise CheckpointViolationError(problem)

        self._repo.save(after, expected_version=before.version, checkpoint=checkpoint, actor=actor, changed_fields=sorted(changes))
        self._record(checkpoint, after, actor)
        return after

    # ---- RESET ----------------------------------------------------------- #
    def reset(self, simulation_id: str, *, actor: str = DEFAULT_ACTOR) -> WorldState:
        """Back to the state the simulation was created with, from any status
        (including a terminal one). The audit trail is kept and gains a
        `simulation_reset` entry; the version keeps increasing."""
        before = self.get(simulation_id)
        initial = self._repo.get_initial_state(simulation_id)
        after = initial.model_copy(update={"version": before.version + 1, "timestamp": self._clock()})
        self._repo.save(after, expected_version=before.version, checkpoint="simulation_reset", actor=actor, changed_fields=["*"])
        self._record("simulation_reset", after, actor)
        return after

    def delete(self, simulation_id: str) -> None:
        if not self._repo.delete(simulation_id):
            raise SimulationNotFoundError(f"no simulation {simulation_id!r}")
