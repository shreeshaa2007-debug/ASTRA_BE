"""Phase 15 tests — the backend API, brief §21 "API" category.

Every test drives the real FastAPI app through TestClient with the real
pipeline behind it (world state, orchestrator, all agents, optimizer,
compliance). Only the LLM is a fake and the database is in-memory. Runs are
synchronous unless a test is specifically about the background behaviour.
"""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.agents.sensing.agent import SensingAgent
from backend.agents.sensing.llm import LLMResponse, LLMUnavailableError
from backend.api import views
from backend.api.context import AppContext
from backend.api.main import create_app
from backend.api.runs import RunRegistry
from backend.database.world_state_repository import SqlAlchemyWorldStateRepository
from backend.optimization import tools as optimization_tools
from backend.orchestration import Orchestrator
from backend.services.world_state import WorldStateStore

requires_built_data = pytest.mark.skipif(
    not all(Path(p).exists() for p in (
        "data/processed/suppliers.csv", "data/processed/routes.csv", "data/processed/tariffs.csv", "data/processed/disruptions.csv",
        "data/processed/inventory_multi_warehouse.csv", "data/processed/demand_modeling_panel.csv",
        "ml/artifacts/xgboost_demand/2026.09.1/model.json",
    )),
    reason="run the Phase 3-5 pipelines first",
)
pytestmark = requires_built_data

PRODUCT, AS_OF = "22197", "2011-11-30"
SUEZ = ["SHA-ROT-SUEZ", "SIN-ROT-SUEZ", "MUM-ROT-SUEZ", "CHE-ROT-SUEZ"]
RULES_OPEN = {"rejected_suppliers": [], "restricted_countries": [], "approval_threshold": 10_000_000}
RULES_LOW = {**RULES_OPEN, "approval_threshold": 100_000}  # the Suez plan costs ~465k, so it needs a human


@pytest.fixture(scope="module", autouse=True)
def memoized_agent_outputs():
    """The agents' outputs are pure functions of their inputs; compute each distinct one once."""
    real, cache = optimization_tools.gather_agent_outputs, {}

    def memo(product_id, as_of_date=None, disrupted_route_ids=frozenset(), excluded_supplier_ids=frozenset(), tariff_rates=None, parameter_overrides=None, config=None):
        key = (product_id, as_of_date, frozenset(disrupted_route_ids), frozenset(excluded_supplier_ids),
               tuple(sorted((tariff_rates or {}).items())), tuple(sorted((parameter_overrides or {}).items())))
        if key not in cache:
            cache[key] = real(product_id, as_of_date, disrupted_route_ids, excluded_supplier_ids, tariff_rates, parameter_overrides, config)
        return copy.deepcopy(cache[key])

    patch = pytest.MonkeyPatch()
    patch.setattr(optimization_tools, "gather_agent_outputs", memo)
    yield
    patch.undo()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def candidate(**overrides) -> dict:
    c = {
        "is_disruption": True, "rationale": "A vessel is aground and blocks the canal.", "event_type": "canal_closure", "location": "Suez Canal",
        "severity": "CRITICAL", "start_date": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(), "estimated_duration": 10,
        "affected_routes": list(SUEZ), "affected_suppliers": [], "affected_products": [], "confidence": 0.95,
    }
    c.update(overrides)
    return c


SUPPLIER_FAILURE = candidate(event_type="supplier_failure", location="Istanbul, Turkey", affected_routes=[], affected_suppliers=["S007"])


class FakeLLM:
    model = "fake-llm"

    def __init__(self, payload=None, error=None):
        self.payload, self.error, self.calls = payload if payload is not None else candidate(), error, 0

    def generate_json(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return LLMResponse(json.dumps(self.payload), self.model, 1.0, 1)


class TextSwitchLLM(FakeLLM):
    """Answers with the supplier failure when the report mentions Istanbul, the Suez closure otherwise."""

    def generate_json(self, *, user_text, **kwargs):
        self.payload = SUPPLIER_FAILURE if "Istanbul" in user_text else candidate()
        return super().generate_json(user_text=user_text, **kwargs)


class SlowLLM(FakeLLM):
    """Takes long enough that a poll loop gets several turns while the run is still in flight."""

    def generate_json(self, **kwargs):
        time.sleep(0.25)
        return super().generate_json(**kwargs)


class BlockingLLM(FakeLLM):
    def __init__(self):
        super().__init__()
        self.started, self.release = threading.Event(), threading.Event()

    def generate_json(self, **kwargs):
        self.started.set()
        assert self.release.wait(30), "the test never released the LLM"
        return super().generate_json(**kwargs)


def make(rules=RULES_OPEN, llm=None, executor=None):
    store = WorldStateStore(SqlAlchemyWorldStateRepository("sqlite:///:memory:"))
    llm = llm or FakeLLM()
    ctx = AppContext(store, Orchestrator(store, SensingAgent(llm=llm), compliance_rules=rules), RunRegistry(executor),
                     (lambda: executor.shutdown(wait=True, cancel_futures=True)) if executor else (lambda: None))
    return TestClient(create_app(context=ctx), raise_server_exceptions=False), store, llm


def new_sim(client, scenario="TEST") -> str:
    r = client.post("/api/simulations", json={"scenario_type": scenario})
    assert r.status_code == 201, r.text
    return r.json()["data"]["simulation_id"]


def run_body(**overrides) -> dict:
    body = {"signal": "A vessel is aground in the Suez Canal.", "product_id": PRODUCT, "as_of_date": AS_OF}
    body.update(overrides)
    return body


def run(client, sim=None, **overrides) -> str:
    sim = sim or new_sim(client)
    r = client.post(f"/api/simulations/{sim}/run", json=run_body(**overrides))
    assert r.status_code == 202, r.text
    return sim


def data(response, status=200):
    assert response.status_code == status, response.text
    return response.json()["data"]


def error(response, status: int, code: str) -> dict:
    assert response.status_code == status, response.text
    err = response.json()["error"]
    assert err["status"] == "error" and err["error_code"] == code and err["message"], err
    return err


def wait_finished(client, sim, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = data(client.get(f"/api/simulations/{sim}/status"))
        if status["run"] and status["run"]["state"] != "RUNNING":
            return status
        time.sleep(0.05)
    raise AssertionError("the run did not finish in time")


# --------------------------------------------------------------------------- #
# the contract: every endpoint in docs/api-plan.md exists
# --------------------------------------------------------------------------- #
def test_every_endpoint_the_api_plan_lists_is_served():
    plan = Path("docs/api-plan.md").read_text(encoding="utf-8")
    documented = {(m, re.sub(r"\{[^}]+\}", "{}", path)) for m, path in re.findall(r"^\| `(GET|POST) (/api/[^`]+)`", plan, flags=re.M)}
    assert len(documented) >= 16, documented

    client, _, _ = make()
    served = {(m.upper(), re.sub(r"\{[^}]+\}", "{}", path)) for path, item in client.get("/openapi.json").json()["paths"].items() for m in item}
    assert documented <= served, f"documented but not served: {sorted(documented - served)}"


def test_the_frontend_calls_only_endpoints_that_exist():
    """If api.ts and the backend drift apart the UI breaks silently. Since Phase 17 every screen reads the real API:
    there are no fixture endpoints left for the frontend to call."""
    client, _, _ = make()
    source = Path("frontend/src/services/api.ts").read_text(encoding="utf-8")

    def normalize(path: str) -> str:
        return re.sub(r"\$\{[^}]+\}|\{[^}]+\}", "{}", re.sub(r"\$\{qs\(.*", "", path))

    called = {normalize(p) for p in re.findall(r"[`']((?:/api/)[^`'?]*)", source)}
    served = {normalize(p) for p in client.get("/openapi.json").json()["paths"]}

    assert len(called) >= 22, called  # the whole real API surface the UI uses, not a stub
    assert called <= served, f"the frontend calls endpoints that are not served: {sorted(called - served)}"
    assert not any("legacy" in p for p in served), "the fixture endpoints are gone"


def test_the_fixture_endpoints_are_gone_so_real_paths_cannot_be_confused_with_fixtures():
    client, _, _ = make()
    for old in ("/api/agents", "/api/warehouses", "/api/scenario-comparisons", "/api/events", "/api/legacy/metrics", "/api/legacy/scenario-comparisons"):
        error(client.get(old), 404, "NOT_FOUND")
    assert "counters" in data(client.get("/api/metrics"))  # /api/metrics is the real process-metrics endpoint (Phase 19), not the old fixture
    assert data(client.get("/api/routes"))["routes"][0]["route_id"]  # /api/routes is the real network table


# --------------------------------------------------------------------------- #
# envelope and errors
# --------------------------------------------------------------------------- #
def test_health_reports_live_mode_and_whether_an_llm_is_configured_never_the_key(monkeypatch):
    client, _, _ = make()
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert client.get("/api/health").json() == {"status": "ok", "mode": "live", "llm_configured": False}
    monkeypatch.setenv("LLM_API_KEY", "super-secret-key-value")
    body = client.get("/api/health")
    assert body.json()["llm_configured"] is True and "super-secret-key-value" not in body.text


def test_unknown_paths_and_wrong_methods_use_the_error_envelope_not_a_bare_detail():
    client, _, _ = make()
    error(client.get("/api/nope"), 404, "NOT_FOUND")
    error(client.delete("/api/simulations"), 405, "METHOD_NOT_ALLOWED")
    error(client.put("/api/dashboard"), 405, "METHOD_NOT_ALLOWED")


def test_request_validation_failures_are_422_with_the_field_named():
    client, _, _ = make()
    assert "scenario_type" in error(client.post("/api/simulations", json={}), 422, "VALIDATION_ERROR")["message"]
    assert "simulation_id" in error(client.post("/api/simulations", json={"scenario_type": "X", "simulation_id": "has spaces!"}), 422, "VALIDATION_ERROR")["message"]
    sim = new_sim(client)
    for bad, field in [({"product_id": None}, "product_id"), ({"signal": ""}, "signal"), ({"signal": "x" * 4001}, "signal"),
                       ({"as_of_date": "not-a-date"}, "as_of_date"), ({"tariff_overrides": {"TUR": "high"}}, "tariff_overrides")]:
        assert field in error(client.post(f"/api/simulations/{sim}/run", json=run_body(**bad)), 422, "VALIDATION_ERROR")["message"], bad
    assert data(client.get(f"/api/simulations/{sim}"))["status"] == "CREATED"  # none of them started anything


def test_a_malformed_body_is_a_422_not_a_500():
    client, _, _ = make()
    r = client.post("/api/simulations", content=b"{not json", headers={"content-type": "application/json"})
    error(r, 422, "VALIDATION_ERROR")


def test_an_unexpected_failure_is_a_generic_500_that_leaks_nothing(monkeypatch):
    client, _, _ = make()

    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail: /home/user/.aws/credentials")

    monkeypatch.setattr(views.logistics_tools, "get_routes", boom)
    err = error(client.get("/api/routes"), 500, "INTERNAL_ERROR")
    assert "secret" not in err["message"] and "credentials" not in json.dumps(err)


def test_a_missing_dataset_and_a_missing_model_are_503s_with_a_recovery_hint(monkeypatch):
    client, _, _ = make()

    def no_routes(*args, **kwargs):
        raise FileNotFoundError("data/processed/routes.csv")

    monkeypatch.setattr(views.logistics_tools, "get_routes", no_routes)
    err = error(client.get("/api/routes"), 503, "DATASET_UNAVAILABLE")
    assert "routes.csv" in err["message"] and "pipeline" in err["recovery"]

    def no_model(*args, **kwargs):
        raise FileNotFoundError("No forecasting model artifact at ml/artifacts/xgboost_demand/9.9")

    monkeypatch.setattr(views.inventory_tools, "forecast_series", no_model)
    error(client.get(f"/api/inventory/forecast?product_id={PRODUCT}"), 503, "MODEL_UNAVAILABLE")


def test_every_error_code_carries_a_recovery_hint():
    client, _, _ = make()
    assert "GET /api/simulations" in error(client.get("/api/simulations/nope"), 404, "SIMULATION_NOT_FOUND")["recovery"]
    assert error(client.get("/api/decisions/nope"), 404, "SIMULATION_NOT_FOUND")["recovery"]
    sim = new_sim(client)
    assert "run the simulation" in error(client.get(f"/api/decisions/{sim}"), 409, "PLAN_NOT_READY")["recovery"].lower()


# --------------------------------------------------------------------------- #
# simulations: create / get / list
# --------------------------------------------------------------------------- #
def test_creating_a_simulation_starts_it_at_the_baseline_network():
    client, _, _ = make()
    s = data(client.post("/api/simulations", json={"scenario_type": "SUEZ_CLOSURE"}), 201)
    assert s["simulation_id"].startswith("sim-") and (s["status"], s["version"], s["scenario_type"]) == ("CREATED", 0, "SUEZ_CLOSURE")
    assert len(s["route_status"]) == 8 and s["route_status"]["SHA-ROT-SUEZ"] == "NORMAL" and s["supplier_status"]["S001"] == "DISRUPTED"
    assert s["tariffs"]["IND"] == pytest.approx(4.59) and s["current_plan"] is None and "engine" not in s  # no plan, so no engine label yet


def test_simulation_ids_can_be_chosen_and_are_never_reused():
    client, _, _ = make()
    assert new_sim(client) != new_sim(client)
    assert data(client.post("/api/simulations", json={"scenario_type": "A", "simulation_id": "demo-1"}), 201)["simulation_id"] == "demo-1"
    error(client.post("/api/simulations", json={"scenario_type": "B", "simulation_id": "demo-1"}), 409, "VALIDATION_ERROR")
    assert data(client.get("/api/simulations/demo-1"))["scenario_type"] == "A"  # the original is untouched


def test_get_returns_the_full_state_with_the_optimizers_label_once_there_is_a_plan():
    client, _, _ = make()
    sim = run(client)
    s = data(client.get(f"/api/simulations/{sim}"))
    assert s["status"] == "COMPLETED" and s["current_plan"]["status"] == "OPTIMAL"
    assert (s["engine"], s["label"]) == ("prototype", "Prototype Optimization")  # rendered verbatim by the frontend
    assert s["demand_forecasts"][0]["model_version"] == "2026.09.1" and s["route_status"]["SHA-ROT-SUEZ"] == "DISRUPTED"
    error(client.get("/api/simulations/nope"), 404, "SIMULATION_NOT_FOUND")


def test_listing_filters_by_status_newest_first_and_honors_limit():
    client, _, _ = make()
    a, b = new_sim(client), new_sim(client)
    run(client, a)
    listed = data(client.get("/api/simulations"))
    assert [s["simulation_id"] for s in listed] == [a, b] and listed[0]["status"] == "COMPLETED"  # `a` was written last
    assert [s["simulation_id"] for s in data(client.get("/api/simulations?status=CREATED"))] == [b]
    assert len(data(client.get("/api/simulations?limit=1"))) == 1
    error(client.get("/api/simulations?status=BOGUS"), 422, "VALIDATION_ERROR")
    error(client.get("/api/simulations?limit=0"), 422, "VALIDATION_ERROR")


# --------------------------------------------------------------------------- #
# run and status
# --------------------------------------------------------------------------- #
def test_an_in_policy_run_completes_and_status_says_what_each_agent_did():
    client, _, _ = make()
    sim = new_sim(client)
    accepted = data(client.post(f"/api/simulations/{sim}/run", json=run_body()), 202)
    assert accepted["simulation_id"] == sim and accepted["run_id"] and accepted["status_url"] == f"/api/simulations/{sim}/status"

    s = data(client.get(f"/api/simulations/{sim}/status"))
    assert (s["status"], s["current_step"], s["awaiting_approval"], s["stalled"], s["error"]) == ("COMPLETED", "done", False, False, None)
    assert {a["id"]: a["status"] for a in s["agents"]} == {
        "sensing": "COMPLETE", "inventory": "COMPLETE", "logistics": "COMPLETE", "sourcing": "COMPLETE",
        "optimization": "COMPLETE", "compliance": "COMPLETE", "human_approval": "NOT_REQUIRED"}
    assert [t["checkpoint"] for t in s["timeline"]] == ["simulation_created", "event_sensed", "agents_assessed", "plan_optimized", "compliance_checked", "plan_finalized"]
    assert s["run"]["state"] == "FINISHED" and s["run"]["outcome"]["outcome"] == "COMPLETED" and s["run"]["error"] is None
    assert [step["name"] for step in s["run"]["outcome"]["steps"]] == ["sense", "agents", "optimize", "compliance"]
    sensing = next(a for a in s["agents"] if a["id"] == "sensing")
    assert "canal_closure" in sensing["detail"] and sensing["latency_ms"] is not None


def test_a_high_impact_run_stops_for_a_human_and_approval_finishes_it():
    client, _, _ = make(RULES_LOW)
    sim = run(client)
    s = data(client.get(f"/api/simulations/{sim}/status"))
    assert (s["status"], s["current_step"], s["awaiting_approval"]) == ("AWAITING_APPROVAL", "approval", True)
    human = next(a for a in s["agents"] if a["id"] == "human_approval")
    assert human["status"] == "ACTION_REQUIRED" and s["run"]["outcome"]["outcome"] == "AWAITING_APPROVAL"

    done = data(client.post(f"/api/decisions/{sim}/approve", json={"decided_by": "logistics.director", "note": "reviewed", "expected_version": s["version"]}))
    assert (done["status"], done["approval_status"]) == ("COMPLETED", "APPROVED") and done["approval_decision"]["decided_by"] == "logistics.director"
    after = data(client.get(f"/api/simulations/{sim}/status"))
    assert after["timeline"][-1]["checkpoint"] == "plan_finalized" and after["timeline"][-1]["actor"] == "logistics.director"  # the approver is on the record
    assert next(a for a in after["agents"] if a["id"] == "human_approval")["detail"] == "approved by logistics.director"


def test_a_run_request_is_checked_before_it_is_accepted():
    client, store, llm = make()
    sim = new_sim(client)
    assert "22197" not in error(client.post(f"/api/simulations/{sim}/run", json=run_body(product_id="NOPE")), 422, "VALIDATION_ERROR")["message"].split("known")[0]
    assert "TURR" in error(client.post(f"/api/simulations/{sim}/run", json=run_body(tariff_overrides={"TURR": 50.0})), 422, "VALIDATION_ERROR")["message"]
    error(client.post("/api/simulations/nope/run", json=run_body()), 404, "SIMULATION_NOT_FOUND")
    assert llm.calls == 0 and store.get(sim).version == 0  # nothing ran, no LLM quota spent


def test_a_simulation_that_has_already_run_cannot_be_run_again_until_reset():
    client, _, _ = make()
    sim = run(client)
    err = error(client.post(f"/api/simulations/{sim}/run", json=run_body()), 409, "INVALID_STATE_TRANSITION")
    assert "COMPLETED" in err["message"] and "reset" in err["recovery"]
    assert data(client.post(f"/api/simulations/{sim}/reset"))["status"] == "CREATED"
    run(client, sim)
    assert data(client.get(f"/api/simulations/{sim}"))["status"] == "COMPLETED"


def test_reset_restores_the_baseline_and_keeps_the_audit_trail():
    client, _, _ = make()
    sim = run(client)
    reset = data(client.post(f"/api/simulations/{sim}/reset"))
    assert reset["current_disruptions"] == [] and reset["current_plan"] is None and reset["route_status"]["SHA-ROT-SUEZ"] == "NORMAL" and reset["version"] > 5
    after = data(client.get(f"/api/simulations/{sim}/status"))
    timeline = [t["checkpoint"] for t in after["timeline"]]
    assert timeline[0] == "simulation_created" and timeline[-1] == "simulation_reset" and "plan_finalized" in timeline  # the audit trail is whole
    assert after["run"] is None and after["current_step"] == "idle"  # ...but the old run is not this simulation's run any more
    assert {a["id"]: a["status"] for a in after["agents"]} == {a["id"]: "PENDING" for a in after["agents"]}  # and no agent has run since the reset
    error(client.post("/api/simulations/nope/reset"), 404, "SIMULATION_NOT_FOUND")


@pytest.mark.parametrize("llm, outcome, agent_status", [
    (FakeLLM({"is_disruption": False, "rationale": "Weather chat."}), "NO_DISRUPTION", "NO_EVENT"),
    (FakeLLM(candidate(severity="high", affected_routes=["NOPE"])), "SENSING_REJECTED", "FAILED"),
    (FakeLLM(error=LLMUnavailableError("Gemini unavailable after 3 attempts")), "SENSING_ERROR", "FAILED"),
])
def test_when_sensing_finds_no_event_the_status_says_why_and_the_simulation_stays_runnable(llm, outcome, agent_status):
    client, _, _ = make(llm=llm)
    sim = run(client)
    s = data(client.get(f"/api/simulations/{sim}/status"))
    assert s["status"] == "CREATED" and s["current_step"] == "idle" and s["run"]["outcome"]["outcome"] == outcome
    assert {a["id"]: a["status"] for a in s["agents"]}["sensing"] == agent_status and all(a["status"] == "PENDING" for a in s["agents"][1:])
    assert s["run"]["outcome"]["message"]
    assert data(client.get(f"/api/simulations/{sim}"))["version"] == 0  # nothing entered the state

    llm.payload, llm.error = candidate(), None  # a good report on the same simulation now works
    run(client, sim)
    assert data(client.get(f"/api/simulations/{sim}"))["status"] == "COMPLETED"


def test_demand_beyond_all_supply_ends_failed_with_the_reason_and_a_diagnosis_on_the_decision():
    client, _, _ = make()
    sim = run(client, product_id="23166")  # ~21k units needed vs ~16k of capacity
    s = data(client.get(f"/api/simulations/{sim}/status"))
    assert (s["status"], s["current_step"]) == ("FAILED", "failed") and s["error"].startswith("OPTIMIZATION_INFEASIBLE") and "Recovery" in s["error"]
    statuses = {a["id"]: a["status"] for a in s["agents"]}
    assert statuses["optimization"] == "FAILED" and statuses["compliance"] == "PENDING" and statuses["inventory"] == "COMPLETE"
    d = data(client.get(f"/api/decisions/{sim}"))  # a decision explanation of "no feasible plan" is still an explanation
    assert d["plan_status"] == "INFEASIBLE" and d["allocations"] == [] and d["objective_value"] is None and d["diagnostics"]["total_shortfall_units"] > 0
    assert d["engine"] == "prototype" and d["label"] == "Prototype Optimization"


def test_a_replan_after_a_compliance_rejection_shows_in_the_timeline_and_the_decision():
    client, _, _ = make({**RULES_OPEN, "rejected_suppliers": ["S007"]})
    sim = run(client)
    s = data(client.get(f"/api/simulations/{sim}/status"))
    assert s["status"] == "COMPLETED" and s["replan_count"] == 1
    assert [t["checkpoint"] for t in s["timeline"]].count("compliance_checked") == 2 and "replan_requested" in [t["checkpoint"] for t in s["timeline"]]
    d = data(client.get(f"/api/decisions/{sim}"))
    assert d["replan_count"] == 1 and "S007" not in {a["supplier_id"] for a in d["allocations"]}


# --------------------------------------------------------------------------- #
# background execution
# --------------------------------------------------------------------------- #
def test_a_run_is_accepted_at_once_reports_progress_and_refuses_a_second_start_or_a_reset():
    llm = BlockingLLM()
    executor = ThreadPoolExecutor(2)
    client, _, _ = make(llm=llm, executor=executor)
    try:
        sim = new_sim(client)
        accepted = data(client.post(f"/api/simulations/{sim}/run", json=run_body()), 202)
        assert accepted["run_state"] == "RUNNING"  # returned while the LLM call is still blocked
        assert llm.started.wait(10)

        s = data(client.get(f"/api/simulations/{sim}/status"))
        assert (s["status"], s["current_step"], s["run"]["state"]) == ("CREATED", "sensing", "RUNNING")
        assert {a["id"]: a["status"] for a in s["agents"]}["sensing"] == "RUNNING"

        assert "in progress" in error(client.post(f"/api/simulations/{sim}/run", json=run_body()), 409, "RUN_IN_PROGRESS")["message"]
        error(client.post(f"/api/simulations/{sim}/reset"), 409, "RUN_IN_PROGRESS")
        assert llm.calls == 0  # the duplicate never reached the LLM

        llm.release.set()
        done = wait_finished(client, sim)
        assert done["status"] == "COMPLETED" and done["run"]["state"] == "FINISHED" and llm.calls == 1
    finally:
        llm.release.set()
        executor.shutdown(wait=True)


def test_simulations_run_at_the_same_time_on_worker_threads_stay_isolated():
    executor = ThreadPoolExecutor(4)
    client, _, _ = make(llm=TextSwitchLLM(), executor=executor)
    try:
        a, b = new_sim(client, "SUEZ"), new_sim(client, "SUPPLIER_FAILURE")
        client.post(f"/api/simulations/{a}/run", json=run_body(signal="Vessel aground in the Suez Canal"))
        client.post(f"/api/simulations/{b}/run", json=run_body(signal="Fire at the Istanbul plant"))
        assert wait_finished(client, a)["status"] == wait_finished(client, b)["status"] == "COMPLETED"
        sa, sb = data(client.get(f"/api/simulations/{a}")), data(client.get(f"/api/simulations/{b}"))
        assert {r for r, s in sa["route_status"].items() if s == "DISRUPTED"} == set(SUEZ) and sa["supplier_status"]["S007"] == "ACTIVE"
        assert {r for r, s in sb["route_status"].items() if s == "DISRUPTED"} == set() and sb["supplier_status"]["S007"] == "DISRUPTED"
    finally:
        executor.shutdown(wait=True)


def test_a_simulation_left_running_with_no_run_behind_it_is_reported_as_stalled():
    """What a restart mid-run looks like: durable state says RUNNING, but this process has no run for it."""
    client, store, _ = make()
    sim = new_sim(client)
    from backend.services.world_state import state_changes_for_event
    from backend.agents.sensing import tools as sensing_tools
    event = sensing_tools.validate_event(candidate(), sensing_tools.load_catalog(), sensing_tools.load_config()).event
    store.commit(sim, "event_sensed", state_changes_for_event(store.get(sim), event))

    s = data(client.get(f"/api/simulations/{sim}/status"))
    assert s["status"] == "RUNNING" and s["stalled"] is True and s["run"] is None
    assert {a["id"]: a["status"] for a in s["agents"]}["inventory"] == "PENDING"  # not "RUNNING": nothing is running


def test_a_crashed_worker_is_recorded_not_lost(monkeypatch):
    client, _, _ = make()
    sim = new_sim(client)

    def crash(*args, **kwargs):
        raise MemoryError("worker died")

    monkeypatch.setattr(views, "state_view", views.state_view)  # (no-op: the views are untouched)
    monkeypatch.setattr(Orchestrator, "run", crash)
    client.post(f"/api/simulations/{sim}/run", json=run_body())
    s = data(client.get(f"/api/simulations/{sim}/status"))
    assert s["run"]["state"] == "CRASHED" and "MemoryError: worker died" in s["run"]["error"]
    client.post(f"/api/simulations/{sim}/run", json=run_body())  # the in-flight marker was cleared, so it can be retried
    assert data(client.get(f"/api/simulations/{sim}/status"))["run"]["state"] == "CRASHED"


# --------------------------------------------------------------------------- #
# decisions, compliance, approval
# --------------------------------------------------------------------------- #
def test_the_decision_explains_the_plan():
    client, _, _ = make(RULES_LOW)
    sim = run(client)
    d = data(client.get(f"/api/decisions/{sim}"))
    assert (d["plan_status"], d["engine"], d["label"], d["product_id"]) == ("OPTIMAL", "prototype", "Prototype Optimization", PRODUCT)
    assert sum(d["objective_terms"].values()) == pytest.approx(d["objective_value"]) and d["plan_spend"] == pytest.approx(d["objective_value"])  # no delay penalty by default
    assert sum(a["quantity"] for a in d["allocations"]) == sum(d["inbound_by_warehouse"].values())
    assert sum(v["share"] for v in d["mode_split"].values()) == pytest.approx(1) and sum(v["share"] for v in d["supplier_split"].values()) == pytest.approx(1)
    assert d["constraints"] and all(c["satisfied"] for c in d["constraints"]) and d["binding_constraints"]
    assert d["decision_factors"] and d["assumptions"] and any(e["kind"] == "supplier" for e in d["excluded_options"])  # India's stranded suppliers
    assert d["disruptions"][0]["event_type"] == "canal_closure" and d["compliance"]["status"] == "ESCALATED" and d["approval"]["status"] == "PENDING"
    assert not any(a["route_id"] in SUEZ for a in d["allocations"])


def test_the_compliance_endpoint_reports_the_verdict_and_its_checks():
    client, _, _ = make(RULES_LOW)
    sim = run(client)
    c = data(client.get(f"/api/compliance/{sim}"))
    assert c["compliance"]["status"] == "ESCALATED" and c["compliance"]["requires_human"] is True
    checks = {x["name"]: x for x in c["compliance"]["checks"]}
    assert set(checks) == {"supplier_permitted", "country_permitted", "route_permitted", "cost_within_threshold"}
    assert checks["cost_within_threshold"]["passed"] is False and checks["supplier_permitted"]["passed"] is True
    assert c["plan_spend"] > 100_000 and c["approval"]["status"] == "PENDING" and c["engine"] == "prototype"


def test_decision_and_compliance_before_there_is_a_plan_are_409_plan_not_ready():
    client, _, _ = make()
    sim = new_sim(client)
    error(client.get(f"/api/decisions/{sim}"), 409, "PLAN_NOT_READY")
    error(client.get(f"/api/compliance/{sim}"), 409, "PLAN_NOT_READY")
    error(client.get("/api/decisions/nope"), 404, "SIMULATION_NOT_FOUND")
    error(client.get("/api/compliance/nope"), 404, "SIMULATION_NOT_FOUND")


def test_a_human_can_reject_an_escalated_plan():
    client, _, _ = make(RULES_LOW)
    sim = run(client)
    done = data(client.post(f"/api/decisions/{sim}/reject", json={"decided_by": "cfo", "note": "over budget"}))
    assert (done["status"], done["approval_status"]) == ("REJECTED", "REJECTED") and done["approval_decision"]["note"] == "over budget"
    error(client.post(f"/api/decisions/{sim}/approve", json={"decided_by": "someone"}), 409, "INVALID_STATE_TRANSITION")  # a rejection is final


def test_an_approval_needs_a_named_approver_and_the_current_version():
    client, _, _ = make(RULES_LOW)
    sim = run(client)
    for body in ({}, {"decided_by": ""}, {"decided_by": "   "}, {"decided_by": "x" * 101}, {"decided_by": "a", "expected_version": -1}):
        error(client.post(f"/api/decisions/{sim}/approve", json=body), 422, "VALIDATION_ERROR")
    version = data(client.get(f"/api/simulations/{sim}"))["version"]
    error(client.post(f"/api/decisions/{sim}/approve", json={"decided_by": "director", "expected_version": version - 1}), 409, "STATE_CONFLICT")
    assert data(client.get(f"/api/simulations/{sim}"))["status"] == "AWAITING_APPROVAL"  # none of that approved anything
    assert data(client.post(f"/api/decisions/{sim}/approve", json={"decided_by": "  director  ", "expected_version": version}))["approval_decision"]["decided_by"] == "director"


def test_nothing_can_be_approved_that_is_not_waiting_and_nothing_twice():
    client, _, _ = make()
    auto = run(client)
    error(client.post(f"/api/decisions/{auto}/approve", json={"decided_by": "director"}), 409, "INVALID_STATE_TRANSITION")  # in-policy: no approval to give
    error(client.post("/api/decisions/nope/approve", json={"decided_by": "d"}), 404, "SIMULATION_NOT_FOUND")
    new = new_sim(client)
    error(client.post(f"/api/decisions/{new}/approve", json={"decided_by": "d"}), 409, "INVALID_STATE_TRANSITION")

    client2, _, _ = make(RULES_LOW)
    sim = run(client2)
    data(client2.post(f"/api/decisions/{sim}/approve", json={"decided_by": "director"}))
    error(client2.post(f"/api/decisions/{sim}/approve", json={"decided_by": "director"}), 409, "INVALID_STATE_TRANSITION")


# --------------------------------------------------------------------------- #
# dashboard
# --------------------------------------------------------------------------- #
def test_the_dashboard_shows_the_baseline_network_before_anything_is_finalized():
    client, _, _ = make()
    d = data(client.get("/api/dashboard"))
    assert d["simulation_id"] is None and "baseline" in d["reflects"] and "engine" not in d
    k = d["kpis"]
    assert (k["routes_total"], k["routes_disrupted"], k["suppliers_total"], k["suppliers_disrupted"]) == (8, 0, 8, 1)  # S001 is disrupted in the data
    assert k["suppliers_reduced"] == 1 and k["supplier_health_pct"] == 75.0  # 6 of 8 fully ACTIVE: S001 disrupted, S002 reduced
    assert k["active_disruptions"] == 0 and k["plan_status"] is None


def test_a_kpi_that_cannot_be_computed_is_null_and_says_why_rather_than_being_invented():
    """With nothing finalized there is no plan to put at risk. (With one, Phase 17 computes them: test_simulation.py.)"""
    client, _, _ = make()
    d = data(client.get("/api/dashboard"))
    assert d["kpis"]["shipments_at_risk"] is None and d["kpis"]["estimated_exposure"] is None
    assert "no simulation has been finalized" in d["unavailable"]["shipments_at_risk"] and "no simulation has been finalized" in d["unavailable"]["estimated_exposure"]


def test_the_dashboard_follows_the_latest_finalized_simulation_not_one_in_progress():
    client, _, _ = make(RULES_LOW)
    waiting = run(client)  # AWAITING_APPROVAL: not finalized
    assert data(client.get("/api/dashboard"))["simulation_id"] is None

    data(client.post(f"/api/decisions/{waiting}/approve", json={"decided_by": "director"}))
    d = data(client.get("/api/dashboard"))
    assert d["simulation_id"] == waiting and d["engine"] == "prototype" and d["label"] == "Prototype Optimization"
    k = d["kpis"]
    assert (k["active_disruptions"], k["routes_disrupted"], k["plan_status"], k["approval_status"]) == (1, 4, "OPTIMAL", "APPROVED")
    assert k["plan_spend"] > 400_000 and k["inventory_records"] == 3

    second = run(client)
    data(client.post(f"/api/decisions/{second}/approve", json={"decided_by": "director"}))
    assert data(client.get("/api/dashboard"))["simulation_id"] == second  # the newer finalized one


# --------------------------------------------------------------------------- #
# disruptions
# --------------------------------------------------------------------------- #
def test_disruptions_lists_active_events_and_pages_through_69805_real_historical_ones():
    client, _, _ = make()
    d = data(client.get("/api/disruptions?limit=3"))
    assert d["active"] == [] and d["simulation_id"] is None
    h = d["historical"]
    assert h["total"] == 69805 and len(h["events"]) == 3 and (h["limit"], h["offset"]) == (3, 0)
    starts = [e["start_date"] for e in h["events"]]
    assert starts == sorted(starts, reverse=True)  # newest first
    assert data(client.get("/api/disruptions?limit=3&offset=3"))["historical"]["events"][0]["event_id"] != h["events"][0]["event_id"]
    assert len(data(client.get("/api/disruptions?limit=500"))["historical"]["events"]) == 500


def test_the_curated_real_world_events_are_separable_and_null_where_unknown_not_nan():
    client, _, _ = make()
    curated = data(client.get("/api/disruptions?source=curated"))["historical"]
    assert curated["total"] == 4
    suez = next(e for e in curated["events"] if e["event_id"] == "CURATED-SUEZ-2021")
    assert suez["affected_route"] == "SHA-ROT-SUEZ" and suez["severity"] == "CRITICAL" and suez["provenance"] == "curated real-world event"
    assert suez["affected_supplier"] is None  # NaN in the CSV; would have been invalid JSON
    noaa = data(client.get("/api/disruptions?source=noaa&severity=CRITICAL&limit=2"))["historical"]
    assert noaa["total"] == 1080 or noaa["total"] > 1000  # 1081 CRITICAL overall, minus the curated critical events
    assert all(e["provenance"].startswith("real: NOAA") and e["severity"] == "CRITICAL" for e in noaa["events"])


def test_disruptions_active_events_come_from_the_scoped_simulation():
    client, _, _ = make()
    sim = run(client)
    d = data(client.get("/api/disruptions?limit=1"))
    assert d["simulation_id"] == sim and d["active"][0]["event_type"] == "canal_closure" and d["active"][0]["affected_routes"] == SUEZ
    other = new_sim(client)
    assert data(client.get(f"/api/disruptions?simulation_id={other}&limit=1"))["active"] == []
    error(client.get("/api/disruptions?simulation_id=nope"), 404, "SIMULATION_NOT_FOUND")


@pytest.mark.parametrize("query", ["source=bogus", "severity=EXTREME", "limit=0", "limit=501", "offset=-1"])
def test_disruption_query_bounds_are_enforced(query):
    client, _, _ = make()
    error(client.get(f"/api/disruptions?{query}"), 422, "VALIDATION_ERROR")


# --------------------------------------------------------------------------- #
# network tables
# --------------------------------------------------------------------------- #
def test_routes_show_the_baseline_and_then_a_simulations_disruptions_and_planned_volume():
    client, _, _ = make()
    base = data(client.get("/api/routes"))
    assert base["simulation_id"] is None and len(base["routes"]) == 8 and all(r["planned_quantity"] is None for r in base["routes"])
    assert {r["route_id"]: r["status"] for r in base["routes"]}["SHA-ROT-SUEZ"] == "NORMAL"

    sim = run(client)
    over = data(client.get(f"/api/routes?simulation_id={sim}"))
    status = {r["route_id"]: r for r in over["routes"]}
    assert all(status[r]["status"] == "DISRUPTED" for r in SUEZ) and status["SHA-ROT-CAPE"]["status"] == "ALTERNATIVE"
    assert all(status[r]["planned_quantity"] == 0 for r in SUEZ) and sum(r["planned_quantity"] for r in over["routes"]) > 0
    assert over["engine"] == "prototype" and "engine" not in base  # only plan-derived data carries the label
    error(client.get("/api/routes?simulation_id=nope"), 404, "SIMULATION_NOT_FOUND")


def test_suppliers_show_status_tariff_and_the_recommended_mix():
    client, _, _ = make()
    base = data(client.get(f"/api/suppliers?product_id={PRODUCT}"))
    by_id = {s["supplier_id"]: s for s in base["suppliers"]}
    assert set(by_id) == {"S002", "S003", "S004", "S005", "S006", "S007"} and base["recommended_mix"] is None
    assert by_id["S003"]["tariff_rate_pct"] == pytest.approx(4.59) and by_id["S003"]["region"] == "India" and by_id["S003"]["provenance"] == "synthetic"

    sim = run(client, tariff_overrides={"TUR": 60.0})
    over = data(client.get(f"/api/suppliers?simulation_id={sim}"))  # product defaults to the plan's
    assert over["product_id"] == PRODUCT
    turkish = next(s for s in over["suppliers"] if s["supplier_id"] == "S007")
    assert turkish["tariff_rate_pct"] == 60.0  # the scenario's tariff, not the data file's
    assert over["recommended_mix"] and sum(m["share"] for m in over["recommended_mix"]) == pytest.approx(1)
    assert sum(s["recommended_quantity"] for s in over["suppliers"]) == sum(m["quantity"] for m in over["recommended_mix"])


def test_a_sensed_supplier_failure_shows_on_the_supplier_table():
    client, _, _ = make(llm=FakeLLM(SUPPLIER_FAILURE))
    sim = run(client)
    row = next(s for s in data(client.get(f"/api/suppliers?simulation_id={sim}"))["suppliers"] if s["supplier_id"] == "S007")
    assert row["status"] == "DISRUPTED" and row["recommended_quantity"] == 0


def test_inventory_is_the_ledger_until_a_simulation_has_assessed_it():
    client, _, _ = make()
    one = data(client.get(f"/api/inventory?product_id={PRODUCT}"))
    assert one["source"] == "ledger" and {r["warehouse_id"] for r in one["inventory"]} == {"Mumbai", "Chennai", "Delhi"}
    assert all(r["current_stock"] >= 0 and r["safety_stock"] > 0 and r["forecast_demand"] is None and r["source"] == "ledger" for r in one["inventory"])
    assert len(data(client.get("/api/inventory"))["inventory"]) == 120  # 40 products x 3 warehouses

    sim = run(client)
    sim_rows = data(client.get(f"/api/inventory?simulation_id={sim}"))
    assert sim_rows["source"] == "simulation" and sim_rows["model_version"] == "2026.09.1" and len(sim_rows["inventory"]) == 3
    for r in sim_rows["inventory"]:
        assert r["forecast_demand"] > 0 and r["horizon_days"] == 45 and r["safety_stock"] > 0 and r["stockout_risk"] in ("LOW", "MEDIUM", "HIGH")
        assert r["days_of_cover"] == pytest.approx(r["current_stock"] / (r["forecast_demand"] / 45), abs=0.06)
    assert data(client.get(f"/api/inventory?simulation_id={sim}&product_id=84077"))["inventory"] == []  # that simulation planned another product


def test_shipments_are_proposed_by_the_plan_never_pretended_to_be_observed():
    client, _, _ = make(RULES_LOW)
    empty = data(client.get("/api/shipments"))
    assert empty["shipments"] == [] and "no shipment dataset" in empty["note"]

    sim = run(client)  # waiting for approval
    proposed = data(client.get(f"/api/shipments?simulation_id={sim}"))["shipments"]
    assert proposed and all(s["status"] == "PROPOSED" and "not observed" in s["provenance"] and s["product_id"] == PRODUCT for s in proposed)
    assert len({s["shipment_id"] for s in proposed}) == len(proposed) and all(s["route_id"] not in SUEZ for s in proposed)

    data(client.post(f"/api/decisions/{sim}/approve", json={"decided_by": "director"}))
    planned = data(client.get("/api/shipments"))  # now the latest finalized
    assert planned["simulation_id"] == sim and {s["status"] for s in planned["shipments"]} == {"PLANNED"} and planned["engine"] == "prototype"

    routed = next(s["route_id"] for s in planned["shipments"] if s["route_id"])
    only = data(client.get(f"/api/shipments?route_id={routed}"))["shipments"]
    assert only and all(s["route_id"] == routed for s in only)
    assert data(client.get("/api/shipments?status=PROPOSED"))["shipments"] == []


def test_agent_status_defaults_to_the_most_recent_simulation_and_is_pending_before_any_exists():
    client, _, _ = make()
    none = data(client.get("/api/agents/status"))
    assert none["simulation_id"] is None and [a["status"] for a in none["agents"]] == ["PENDING"] * 7 and none["timeline"] == []
    first = run(client)
    second = new_sim(client)
    assert data(client.get("/api/agents/status"))["simulation_id"] == second  # most recently updated
    got = data(client.get(f"/api/agents/status?simulation_id={first}"))
    assert got["simulation_id"] == first and [a["id"] for a in got["agents"]] == ["sensing", "inventory", "logistics", "sourcing", "optimization", "compliance", "human_approval"]
    error(client.get("/api/agents/status?simulation_id=nope"), 404, "SIMULATION_NOT_FOUND")


# --------------------------------------------------------------------------- #
# forecasts
# --------------------------------------------------------------------------- #
def test_the_inventory_forecast_returns_actuals_and_a_point_forecast_with_no_invented_band():
    client, _, _ = make()
    f = data(client.get(f"/api/inventory/forecast?product_id={PRODUCT}&warehouse_id=Chennai&horizon_days=7&as_of_date={AS_OF}"))
    assert (f["product_id"], f["warehouse_id"], f["horizon_days"], f["as_of_date"], f["model_version"]) == (PRODUCT, "Chennai", 7, AS_OF, "2026.09.1")
    assert len(f["actual"]) == 28 and len(f["predicted"]) == 7 and f["confidence"] is None and "point forecast" in f["note"]
    dates = [p["date"] for p in f["predicted"]]
    assert dates[0] == "2011-12-01" and dates == sorted(dates) and all(p["value"] >= 0 for p in f["predicted"])
    assert f["actual"][-1]["date"] == AS_OF


def test_the_warehouse_forecast_is_the_model_forecast_times_the_warehouse_share():
    client, _, _ = make()
    total = 0.0
    for wh in ("Mumbai", "Chennai", "Delhi"):
        total += sum(p["value"] for p in data(client.get(f"/api/inventory/forecast?product_id={PRODUCT}&warehouse_id={wh}&horizon_days=5&as_of_date={AS_OF}"))["predicted"])
    from backend.agents.inventory import tools as inventory_tools
    uk = sum(d["predicted"] for d in inventory_tools.forecast_series(PRODUCT, AS_OF, 5))
    assert total == pytest.approx(uk, rel=0.01)  # the three shares partition the UK-level forecast


@pytest.mark.parametrize("query, fragment", [
    (f"product_id={PRODUCT}&warehouse_id=Atlantis", "warehouse_id"), ("product_id=NOPE", "No demand history"),
    (f"product_id={PRODUCT}&horizon_days=91", "horizon_days"), (f"product_id={PRODUCT}&horizon_days=0", "horizon_days"), ("", "product_id"),
])
def test_forecast_queries_are_validated(query, fragment):
    client, _, _ = make()
    assert fragment in error(client.get(f"/api/inventory/forecast?{query}"), 422, "VALIDATION_ERROR")["message"]


def test_post_forecast_predicts_from_the_history_the_caller_supplies():
    client, _, _ = make()
    history = [40.0 + (i % 7) for i in range(28)]
    f = data(client.post("/api/forecast", json={"product_id": PRODUCT, "forecast_horizon": 5, "recent_demand": history, "start_date": "2011-12-01"}))
    assert f["dates"] == ["2011-12-01", "2011-12-02", "2011-12-03", "2011-12-04", "2011-12-05"] and len(f["predicted_demand"]) == 5
    assert f["model_version"] == "2026.09.1" and f["confidence"] is None and all(v >= 0 for v in f["predicted_demand"])
    heavier = data(client.post("/api/forecast", json={"product_id": PRODUCT, "forecast_horizon": 5, "recent_demand": [h * 10 for h in history], "start_date": "2011-12-01"}))
    assert sum(heavier["predicted_demand"]) > sum(f["predicted_demand"])  # the history is actually used


@pytest.mark.parametrize("body", [
    {"product_id": PRODUCT, "recent_demand": []}, {"product_id": PRODUCT, "recent_demand": [1.0], "forecast_horizon": 0},
    {"product_id": PRODUCT, "recent_demand": [1.0], "forecast_horizon": 91}, {"product_id": PRODUCT, "recent_demand": ["a"]}, {"recent_demand": [1.0]},
])
def test_post_forecast_validates_its_body(body):
    client, _, _ = make()
    error(client.post("/api/forecast", json=body), 422, "VALIDATION_ERROR")


def test_a_series_the_model_was_not_trained_on_is_a_422_that_says_so():
    client, _, _ = make()
    err = error(client.post("/api/forecast", json={"product_id": PRODUCT, "location_id": "Atlantis", "recent_demand": [1.0, 2.0]}), 422, "VALIDATION_ERROR")
    assert "not one of this model's trained series" in err["message"]


# --------------------------------------------------------------------------- #
# wiring
# --------------------------------------------------------------------------- #
def test_cors_allows_the_frontend_origin_to_post_and_no_one_else():
    client, _, _ = make()
    ok = client.options("/api/simulations", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == "http://localhost:5173" and "POST" in ok.headers["access-control-allow-methods"]
    bad = client.options("/api/simulations", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers


def test_importing_the_app_touches_no_database_file_and_builds_nothing(tmp_path):
    code = ("import backend.api.main as m, pathlib; "
            "print('ctx', m.app.state.ctx); print('dbs', sorted(p.name for p in pathlib.Path('.').glob('*.db')))")
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=tmp_path, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(Path.cwd()), "DATABASE_URL": "sqlite:///./should_not_exist.db"})
    assert out.returncode == 0, out.stderr
    assert "ctx None" in out.stdout and "dbs []" in out.stdout


def test_the_default_app_persists_to_the_database_in_database_url(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'api.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    with TestClient(create_app(run_inline=True), raise_server_exceptions=False) as first:  # `with` exercises startup and shutdown
        sim = new_sim(first, "PERSISTED")
    assert (tmp_path / "api.db").exists()
    with TestClient(create_app(run_inline=True), raise_server_exceptions=False) as second:  # a new process, in effect
        assert data(second.get(f"/api/simulations/{sim}"))["scenario_type"] == "PERSISTED"


def test_the_default_app_reports_a_missing_llm_key_as_a_clean_run_outcome_not_a_startup_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'nokey.db'}")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with TestClient(create_app(run_inline=True), raise_server_exceptions=False) as client:
        sim = run(client)
        s = data(client.get(f"/api/simulations/{sim}/status"))
        assert s["run"]["outcome"]["outcome"] == "SENSING_ERROR" and "LLM_API_KEY" in s["run"]["outcome"]["message"] and s["status"] == "CREATED"


# --------------------------------------------------------------------------- #
# JSON safety
# --------------------------------------------------------------------------- #
def test_clean_makes_pandas_and_numpy_values_json_safe():
    dirty = {"a": float("nan"), "b": np.int64(3), "c": np.float64(2.5), "d": [np.nan, {"e": np.float64("nan")}], "f": (1, np.int32(2)), "g": "text"}
    cleaned = views.clean(dirty)
    assert cleaned == {"a": None, "b": 3, "c": 2.5, "d": [None, {"e": None}], "f": [1, 2], "g": "text"}
    json.dumps(cleaned, allow_nan=False)  # would raise on any NaN or numpy type left behind
    assert type(cleaned["b"]) is int and type(cleaned["c"]) is float


# --------------------------------------------------------------------------- #
# polling a run that is still writing (a race caught while testing this phase)
# --------------------------------------------------------------------------- #
def test_status_survives_a_state_older_than_the_history_it_is_paired_with():
    """The handler reads the state and the checkpoint history separately. A poll landing between a run's
    commits can pair a state from before `event_sensed` with a history from after it. The view must
    describe the state it was given, not crash on the disagreement (it did: IndexError -> 500)."""
    client, store, _ = make()
    sim = new_sim(client)
    early = store.get(sim)  # before anything happened: no event, no inventory, no plan
    from backend.agents.sensing import tools as sensing_tools
    from backend.services.world_state import state_changes_for_event

    event = sensing_tools.validate_event(candidate(), sensing_tools.load_catalog(), sensing_tools.load_config()).event
    store.commit(sim, "event_sensed", state_changes_for_event(early, event))
    late_history = store.history(sim)  # now includes event_sensed, which `early` does not know about
    assert "event_sensed" in [h.checkpoint for h in late_history] and early.current_disruptions == []

    view = views.status_view(early, late_history, run=None, running=True)  # must not raise
    statuses = {a["id"]: a["status"] for a in view["agents"]}
    assert statuses["sensing"] == "PENDING" or statuses["sensing"] == "RUNNING"  # describes the state it has: no event yet
    assert all(statuses[a] == "PENDING" for a in ("inventory", "logistics", "sourcing", "optimization", "compliance"))


def test_a_status_poll_landing_before_a_running_state_and_history_reads_never_errors():
    client, store, _ = make()
    sim = new_sim(client)
    early = store.get(sim)
    from backend.services.world_state import state_changes_for_event
    from backend.agents.sensing import tools as sensing_tools

    event = sensing_tools.validate_event(candidate(), sensing_tools.load_catalog(), sensing_tools.load_config()).event
    ahead = store.commit(sim, "event_sensed", state_changes_for_event(early, event))
    stale_history = store.history(sim)[:1]  # the other way round: history older than the state
    view = views.status_view(ahead, stale_history, run=None, running=True)
    assert {a["id"]: a["status"] for a in view["agents"]}["sensing"] == "COMPLETE" and view["current_step"] == "sensing"


def test_polling_status_flat_out_while_a_run_executes_on_a_worker_thread_never_returns_an_error():
    executor = ThreadPoolExecutor(2)
    client, _, _ = make(llm=SlowLLM(), executor=executor)
    try:
        sim = new_sim(client)
        client.post(f"/api/simulations/{sim}/run", json=run_body())
        polls, statuses = 0, set()
        deadline = time.time() + 30
        while time.time() < deadline:
            r = client.get(f"/api/simulations/{sim}/status")
            assert r.status_code == 200, r.text  # every poll, including the ones that land mid-commit
            body = r.json()["data"]
            polls, statuses = polls + 1, statuses | {body["status"]}
            assert not (body["stalled"]), "a run that is executing was reported stalled"
            if body["run"] and body["run"]["state"] != "RUNNING":
                break
        assert polls >= 5 and "CREATED" in statuses and "COMPLETED" in statuses  # polled through the whole run, start to finish
    finally:
        executor.shutdown(wait=True)


def test_no_response_ever_shows_a_finished_run_beside_a_running_state_or_a_healthy_run_as_stalled():
    """Smoke test: many runs in flight, several pollers going flat out, every response checked. It is NOT the guarantee —
    the race window is microseconds and this can pass on broken code; the deterministic test below is what pins the ordering."""
    executor = ThreadPoolExecutor(4)
    client, _, _ = make(executor=executor)
    stop, violations, polls = threading.Event(), [], [0]
    try:
        sims = [new_sim(client) for _ in range(24)]

        def poller():
            mine = TestClient(client.app, raise_server_exceptions=False)  # one client per thread
            while not stop.is_set():
                for sim in sims:
                    r = mine.get(f"/api/simulations/{sim}/status")
                    polls[0] += 1
                    if r.status_code != 200:
                        violations.append(("http", r.status_code, r.text[:120]))
                        continue
                    b = r.json()["data"]
                    if b["run"] and b["run"]["state"] != "RUNNING" and b["status"] == "RUNNING":
                        violations.append(("finished run, running state", sim))
                    if b["stalled"]:
                        violations.append(("stalled while a run exists", sim, b["run"] and b["run"]["state"]))

        pollers = [threading.Thread(target=poller) for _ in range(3)]
        for t in pollers:
            t.start()
        for sim in sims:
            client.post(f"/api/simulations/{sim}/run", json=run_body())
        for sim in sims:
            wait_finished(client, sim, timeout=60)
        stop.set()
        for t in pollers:
            t.join(timeout=30)
        assert polls[0] > 100, polls[0]  # it really was hammered
        assert violations == [], violations[:5]
    finally:
        stop.set()
        executor.shutdown(wait=True)


def test_read_status_stays_coherent_when_the_run_finishes_at_the_worst_possible_moment():
    """Deterministic version of the race: the run finishes in the instant after the state is read. Read in the right order
    (run flag, run snapshot, then state) the response is coherent — the run merely looks older than the state. Read in either
    wrong order it is not: 'run FINISHED beside a RUNNING state', or 'stalled' for a run that had just finished. Both happened."""
    from backend.agents.sensing import tools as sensing_tools
    from backend.services.world_state import state_changes_for_event

    class Done:  # what the orchestrator would have returned
        def model_dump(self, **kwargs):
            return {"outcome": "COMPLETED", "steps": []}

    class FinishesAfterFirstRead:
        """A store whose first get() returns the state, then lets the worker thread's run finish before anything else is read."""

        def __init__(self, real, finish):
            self.real, self.finish, self.fired = real, finish, False

        def get(self, simulation_id):
            state = self.real.get(simulation_id)
            if not self.fired:
                self.fired = True
                self.finish()
            return state

        def history(self, simulation_id):
            return self.real.history(simulation_id)

    class Deferred:  # an executor that holds the job until told to run it
        def __init__(self):
            self.jobs = []

        def submit(self, job):
            self.jobs.append(job)

    client, store, _ = make()
    sim = new_sim(client)
    event = sensing_tools.validate_event(candidate(), sensing_tools.load_catalog(), sensing_tools.load_config()).event
    store.commit(sim, "event_sensed", state_changes_for_event(store.get(sim), event))  # the run is mid-flight: state RUNNING

    executor = Deferred()
    runs = RunRegistry(executor)
    runs.start(sim, lambda: Done())  # registered as active, job not yet executed
    assert runs.is_running(sim)

    body = views.read_status(FinishesAfterFirstRead(store, lambda: executor.jobs.pop()()), runs, sim)
    assert not runs.is_running(sim)  # the run really did finish during the read

    assert body["status"] == "RUNNING"
    assert body["stalled"] is False, "a run that was in flight when the poll began was reported stalled"
    assert body["run"]["state"] == "RUNNING", "the run snapshot was taken after the state, so it shows a finish the state has not caught up to"

