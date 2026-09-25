"""The production app for end-to-end tests: everything real — the FastAPI app, the SQLite world
state, the thread pool the runs execute on, every agent, the optimizer, the compliance rules
from backend/config/ — except the LLM, which is a deterministic script (a live model would make
a test depend on a network and on the model's mood). Started as a real process:

    uvicorn backend.tests.e2e.app:app --port ...

Environment (all optional):
    DATABASE_URL       the world-state database (the harness gives each server its own file)
    E2E_LLM_DELAY      seconds the "LLM" takes to answer — a window to act in while sensing runs
    E2E_AGENT_DELAY    seconds added to the agents step — a window in which the state is RUNNING
    E2E_LLM=real       use the production wiring instead (needs LLM_API_KEY): the opt-in live tests
    E2E_LLM_FAIL=1     every LLM call fails as if the provider were down (behind the same circuit breaker production uses)
    E2E_EXTRA_REQUIRED_FILE   a path readiness must find: point it at nothing to make the process "alive but not ready"
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from backend.agents.sensing.agent import SensingAgent
from backend.agents.sensing.llm import CircuitBreakerLLM, LLMResponse, LLMUnavailableError
from backend.api import readiness
from backend.api.context import AppContext, build_default_context
from backend.api.main import create_app
from backend.api.runs import RunRegistry
from backend.database.world_state_repository import SqlAlchemyWorldStateRepository
from backend.monitoring import load_settings
from backend.optimization import tools as optimization_tools
from backend.orchestration import Orchestrator
from backend.services.world_state import WorldStateStore

SUEZ = ["SHA-ROT-SUEZ", "SIN-ROT-SUEZ", "MUM-ROT-SUEZ", "CHE-ROT-SUEZ"]


def _event(event_type: str, location: str, severity: str, duration: float, *, routes=(), suppliers=(), rationale: str) -> dict:
    return {
        "is_disruption": True, "rationale": rationale, "event_type": event_type, "location": location, "severity": severity,
        "start_date": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(), "estimated_duration": duration,
        "affected_routes": list(routes), "affected_suppliers": list(suppliers), "affected_products": [], "confidence": 0.95,
    }


class ScriptedLLM:
    """Reads the report the way the real model would for the four demo reports, and calls anything
    else no disruption. Only the text between <report> tags counts: the prompt around it lists
    every route id, including the SUEZ ones."""

    model = "scripted-e2e-llm"

    def __init__(self, delay: float = 0.0, fail: bool = False):
        self.delay, self.fail = delay, fail

    def generate_json(self, *, system_instruction: str, user_text: str, response_schema: dict) -> LLMResponse:
        started = time.perf_counter()
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise LLMUnavailableError("scripted outage: the provider is down")
        match = re.search(r"<report>\n?(.*?)\n?</report>", user_text, flags=re.S)
        report = (match.group(1) if match else user_text).lower()
        if "istanbul" in report or "fire" in report:
            payload = _event("supplier_failure", "Istanbul, Turkey", "HIGH", 30, suppliers=["S007"], rationale="A plant fire takes the Istanbul supplier offline.")
        elif "suez" in report:
            payload = _event("canal_closure", "Suez Canal", "CRITICAL", 10, routes=SUEZ, rationale="A vessel is aground and blocks the canal.")
        elif "cyclone" in report:
            payload = _event("severe_weather", "Mumbai, India", "HIGH", 5, routes=["MUM-ROT-SUEZ"], rationale="A cyclone closes the port of Mumbai.")
        elif "tariff" in report:
            payload = _event("tariff_change", "China", "MEDIUM", 60, suppliers=["S002"], rationale="A tariff increase on Chinese imports.")
        else:
            payload = {"is_disruption": False, "rationale": "Nothing in this report concerns the supply chain."}
        return LLMResponse(json.dumps(payload), self.model, round((time.perf_counter() - started) * 1000, 1), 1)


def _slow_down_the_agents(seconds: float) -> None:
    real = optimization_tools.gather_agent_outputs

    def slow(*args, **kwargs):
        time.sleep(seconds)
        return real(*args, **kwargs)

    optimization_tools.gather_agent_outputs = slow


def build():
    if os.environ.get("E2E_LLM") == "real":
        return create_app(context=build_default_context())
    if os.environ.get("E2E_AGENT_DELAY"):
        _slow_down_the_agents(float(os.environ["E2E_AGENT_DELAY"]))
    store = WorldStateStore(SqlAlchemyWorldStateRepository())  # DATABASE_URL
    executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="run")
    if os.environ.get("E2E_EXTRA_REQUIRED_FILE"):
        readiness.REQUIRED_DATA = readiness.REQUIRED_DATA + (os.environ["E2E_EXTRA_REQUIRED_FILE"],)
    breaker = load_settings()["llm"]  # the production wiring wraps the real client in this breaker; so does the harness
    llm = CircuitBreakerLLM(ScriptedLLM(float(os.environ.get("E2E_LLM_DELAY", "0")), os.environ.get("E2E_LLM_FAIL") == "1"),
                            int(breaker["circuit_failure_threshold"]), float(breaker["circuit_cooldown_seconds"]))
    orchestrator = Orchestrator(store, SensingAgent(llm=llm))  # compliance rules: the real config file
    context = AppContext(store, orchestrator, RunRegistry(executor), lambda: executor.shutdown(wait=True, cancel_futures=True))
    return create_app(context=context)


app = build()
