"""Error handling — every failure leaves the API as the envelope in
api-plan.md, never as a stack trace or a bare FastAPI `{"detail": ...}`:

    { "error": { "status": "error", "error_code": "...", "message": "...", "recovery": "..." } }

Error codes are the api-plan.md list plus the ones the build turned up
(STATE_CONFLICT, INVALID_STATE_TRANSITION, CHECKPOINT_VIOLATION, RUN_IN_PROGRESS,
PLAN_NOT_READY, INTERNAL_ERROR, LLM_UNAVAILABLE, NOT_FOUND) and the three of authentication
(UNAUTHENTICATED 401, FORBIDDEN 403, AUTH_UNAVAILABLE 503). The world-state
errors already carry their `error_code`, so mapping them is a status lookup.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.agents.sensing.llm import LLMUnavailableError
from backend.data import DatasetUnavailableError
from backend.monitoring import request_id_var
from backend.simulation.scenarios import ScenarioNotModeledError, UnknownScenarioError
from backend.services.world_state import (
    CheckpointViolationError,
    ConcurrentModificationError,
    InvalidTransitionError,
    SimulationExistsError,
    SimulationNotFoundError,
    UnknownCheckpointError,
)

logger = logging.getLogger("resilientsc.api")

RECOVERY = {
    "SIMULATION_NOT_FOUND": "List simulations with GET /api/simulations, or create one with POST /api/simulations.",
    "INVALID_STATE_TRANSITION": "Check GET /api/simulations/{id}/status. A finished simulation must be reset (POST /api/simulations/{id}/reset) before it can run again.",
    "STATE_CONFLICT": "Re-read the simulation (GET /api/simulations/{id}) and retry against its current version.",
    "CHECKPOINT_VIOLATION": "The simulation is not in a state where this is allowed; check GET /api/simulations/{id}/status.",
    "RUN_IN_PROGRESS": "Poll GET /api/simulations/{id}/status until the run finishes.",
    "PLAN_NOT_READY": "Run the simulation first: POST /api/simulations/{id}/run.",
    "DATASET_UNAVAILABLE": "Build data/processed/ by running the preprocessing pipeline (Phases 3-5) — or, with DATA_BACKEND=sql, load the ref_* tables with scripts/load_reference_data.py.",
    "MODEL_UNAVAILABLE": "Train and register the forecasting model (ml/training).",
    "LLM_UNAVAILABLE": "Check LLM_PROVIDER, LLM_API_KEY / LLM_MODEL and network access, then retry.",
    "VALIDATION_ERROR": "Fix the request and retry.",
    "DATABASE_UNAVAILABLE": "Check DATABASE_URL and that the database is reachable; GET /api/ready reports which dependency is down.",
    "SCENARIO_NOT_FOUND": "List the scenarios with GET /api/scenarios.",
    "SCENARIO_NOT_MODELED": "Pick a scenario whose \"modeled\" is true in GET /api/scenarios; the others say why they cannot be simulated yet.",
    "UNAUTHENTICATED": "Send a valid bearer token (Authorization: Bearer <token>). Behind the SAP Approuter this is added for you once you are signed in.",
    "FORBIDDEN": "Your token is valid but lacks the scope this needs; ask for the role collection that carries it. GET /api/me shows what you have.",
    "AUTH_UNAVAILABLE": "The identity provider's signing keys could not be fetched; retry shortly, and check AUTH_JWKS_URL / the XSUAA binding.",
}


class ApiError(Exception):
    def __init__(self, status_code: int, error_code: str, message: str, recovery: str | None = None, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.status_code, self.error_code, self.message = status_code, error_code, message
        self.recovery = recovery or RECOVERY.get(error_code)
        self.headers = headers or {}


def error_response(status_code: int, error_code: str, message: str, recovery: str | None = None, headers: dict[str, str] | None = None) -> JSONResponse:
    body: dict = {"status": "error", "error_code": error_code, "message": message}
    recovery = recovery or RECOVERY.get(error_code)
    if recovery:
        body["recovery"] = recovery
    request_id = request_id_var.get()
    if request_id:  # the id that is also in every log line for this request: what a person quotes when asking what went wrong
        body["request_id"] = request_id
    return JSONResponse(status_code=status_code, content={"error": body}, headers={**({"X-Request-ID": request_id} if request_id else {}), **(headers or {})} or None)


# world-state / persistence error class -> HTTP status; the error_code comes from the exception itself
_STATUS = {
    SimulationNotFoundError: 404,
    SimulationExistsError: 409,
    ConcurrentModificationError: 409,
    InvalidTransitionError: 409,
    CheckpointViolationError: 409,
    UnknownCheckpointError: 500,
    UnknownScenarioError: 404,
    ScenarioNotModeledError: 422,
}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return error_response(exc.status_code, exc.error_code, exc.message, exc.recovery, exc.headers)

    for exc_class, status in _STATUS.items():
        def make(status_code: int):
            async def handler(request: Request, exc: Exception):
                return error_response(status_code, getattr(exc, "error_code", "VALIDATION_ERROR"), str(exc))
            return handler
        app.add_exception_handler(exc_class, make(status))

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        parts = []
        for e in exc.errors():
            where = ".".join(str(p) for p in e["loc"] if p not in ("body", "query", "path"))
            parts.append(f"{where}: {e['msg']}" if where else e["msg"])
        return error_response(422, "VALIDATION_ERROR", "; ".join(parts) or "invalid request")

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
        return error_response(exc.status_code, code, str(exc.detail))

    @app.exception_handler(DatasetUnavailableError)
    async def _missing_dataset(request: Request, exc: DatasetUnavailableError):
        # more specific than FileNotFoundError, which it is a subclass of: the data may be a database table, not a file
        logger.error("dataset unavailable: %s", exc)
        return error_response(503, "DATASET_UNAVAILABLE", f"a required dataset is unavailable: {exc}")

    @app.exception_handler(FileNotFoundError)
    async def _missing_file(request: Request, exc: FileNotFoundError):
        # a processed dataset or model artifact that hasn't been built
        code = "MODEL_UNAVAILABLE" if "artifact" in str(exc).lower() else "DATASET_UNAVAILABLE"
        logger.error("missing file: %s", exc)
        return error_response(503, code, f"a required file is missing: {exc}")

    @app.exception_handler(OperationalError)
    async def _database(request: Request, exc: OperationalError):
        logger.error("database unavailable: %s", exc.orig or exc, extra={"event": "database_unavailable"})
        return error_response(503, "DATABASE_UNAVAILABLE", "the database is not reachable right now")

    @app.exception_handler(LLMUnavailableError)
    async def _llm(request: Request, exc: LLMUnavailableError):
        return error_response(503, exc.error_code, str(exc))

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return error_response(500, "INTERNAL_ERROR", "an unexpected error occurred; it has been logged")
