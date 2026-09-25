"""Who a log line is about. Three context variables — the HTTP request, the simulation, the run — that
the log formatter stamps on every record made while they are set, so one grep on a `simulation_id`
follows a simulation from the POST that created it, through every step of its run on a worker thread,
to the human's approval.

Context variables do not cross into a thread pool by themselves; `run_in_context` is how a job carries
them there (backend/api/runs.py)."""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Callable, Iterator

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
simulation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("simulation_id", default=None)
run_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("run_id", default=None)

_VARS = {"request_id": request_id_var, "simulation_id": simulation_id_var, "run_id": run_id_var}


def current() -> dict[str, str]:
    """The ids that are set right now."""
    return {name: value for name, var in _VARS.items() if (value := var.get()) is not None}


@contextmanager
def bind(**ids: str | None) -> Iterator[None]:
    """Sets the given ids for the duration of the block (a None leaves that id as it was)."""
    tokens = [(_VARS[name], _VARS[name].set(value)) for name, value in ids.items() if value is not None]
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def run_in_context(ctx: contextvars.Context, fn: Callable[[], Any]) -> Any:
    """Runs `fn` inside a copy of the context that was current when `ctx` was captured (see `contextvars.copy_context`)."""
    return ctx.run(fn)
