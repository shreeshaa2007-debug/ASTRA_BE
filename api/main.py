"""ResilientSC backend API — Phase 15.

The endpoints of docs/api-plan.md, backed by the real pipeline: the shared world
state (Phase 12), the orchestrator and its agents (Phases 6-14), the optimizer
(Phase 10). Run with:

    uvicorn backend.api.main:app --port 8000 --env-file .env

`create_app()` takes the services as arguments so tests can substitute a fake LLM,
an in-memory database and synchronous runs; `app` is the production wiring, built
lazily on first request so importing this module touches no database or network.

"""
from __future__ import annotations

import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.api.context import AppContext, build_default_context, make_state
from backend.api.errors import install_error_handlers
from backend.api.routers import decisions, network, ops, scenarios, simulations
from backend.monitoring import configure_logging, load_settings, metrics, request_id_var

DEFAULT_CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

logger = logging.getLogger("resilientsc.api")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
# polled or probed constantly: logged at DEBUG so the INFO log stays a record of what people did
_QUIET_PATHS = ("/api/health", "/api/ready", "/api/metrics")


def create_app(
    *,
    context: AppContext | None = None,
    compliance_rules: dict | None = None,
    run_inline: bool = False,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        if app.state.ctx is not None:
            app.state.ctx.shutdown()  # lets in-flight runs finish before the process exits

    app = FastAPI(
        title="ResilientSC API",
        version="1.0.0",
        description="AI-assisted supply-chain resilience: sense a disruption, plan a response, check compliance, ask a human. See docs/api-plan.md.",
        lifespan=lifespan,
    )
    settings = load_settings()
    configure_logging(settings["logging"]["level"], settings["logging"]["format"])  # idempotent; until now nothing configured a handler, so INFO was dropped
    make_state(app, (lambda: context) if context is not None else (lambda: build_default_context(compliance_rules=compliance_rules, run_inline=run_inline)))

    origins = cors_origins or [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()] or DEFAULT_CORS_ORIGINS
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])
    install_error_handlers(app)

    @app.middleware("http")
    async def observe(request: Request, call_next):
        """Gives the request an id (the caller's X-Request-ID if it is a sane one), logs one line for it, counts and times it.
        The id is on every log line made while it is handled, in the response header, and in an error envelope."""
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _REQUEST_ID.match(incoming) else uuid.uuid4().hex[:16]
        request_id_var.set(request_id)
        started = time.perf_counter()

        def record(status: int) -> None:
            duration_ms = round((time.perf_counter() - started) * 1000, 1)
            route = getattr(request.scope.get("route"), "path", None) or "unmatched"  # the template, so /simulations/sim-1 and /sim-2 are one series
            metrics.inc("http_requests_total", {"method": request.method, "route": route, "status": status})
            metrics.observe("http_request_duration_ms", duration_ms, {"route": route})
            quiet = request.url.path in _QUIET_PATHS or (request.method == "GET" and request.url.path.endswith("/status"))
            level = logging.ERROR if status >= 500 else logging.DEBUG if quiet else logging.INFO
            logger.log(level, "%s %s -> %d (%.0f ms)", request.method, request.url.path, status, duration_ms,
                       extra={"event": "request", "method": request.method, "path": request.url.path, "route": route, "status": status, "duration_ms": duration_ms,
                              **({"simulation_id": sid} if (sid := request.path_params.get("simulation_id")) else {})})

        try:
            response = await call_next(request)
        except Exception:
            record(500)  # the error handler turns it into the envelope; this is the request's own record
            raise
        response.headers["X-Request-ID"] = request_id
        record(response.status_code)
        return response

    @app.get("/api/health", tags=["meta"])
    def health():
        return {"status": "ok", "mode": "live", "llm_configured": bool(os.environ.get("LLM_API_KEY"))}

    for router in (simulations.router, decisions.router, network.router, scenarios.router, ops.router):
        app.include_router(router)
    return app


app = create_app()
