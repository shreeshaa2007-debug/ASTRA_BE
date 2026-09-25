"""World-state errors. Each carries an `error_code` from api-plan.md's error
list so Phase 15 can map it to the response envelope without a lookup table."""
from __future__ import annotations

from backend.database.world_state_repository import (  # noqa: F401  (re-exported: one import site for callers)
    ConcurrentModificationError,
    SimulationExistsError,
    SimulationNotFoundError,
)


class WorldStateError(Exception):
    error_code = "VALIDATION_ERROR"


class UnknownCheckpointError(WorldStateError):
    """Not one of the named checkpoints in checkpoints.CHECKPOINTS."""


class CheckpointViolationError(WorldStateError):
    """The write touches a field the checkpoint doesn't own, carries an invalid
    value, or would leave the state breaking an invariant (e.g. finalizing a
    high-impact plan nobody approved)."""

    error_code = "CHECKPOINT_VIOLATION"


class InvalidTransitionError(WorldStateError):
    """The simulation's current status doesn't allow this checkpoint."""

    error_code = "INVALID_STATE_TRANSITION"
