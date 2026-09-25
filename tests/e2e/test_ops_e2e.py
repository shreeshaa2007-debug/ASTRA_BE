"""Phase 19 — observability, end to end: what a real process leaves behind and reports.

The in-process tests (backend/tests/test_monitoring.py) prove the machinery. These prove it in the deployed
shape: a real uvicorn process writing real JSON log lines to its real stderr, counting real requests, and
answering the operator's questions (is it ready? is it stuck? is the provider down?) over real HTTP.
"""
from __future__ import annotations

import json
import re
import time

import httpx
import pytest

from backend.tests.e2e.conftest import finish, free_text_run, requires_built_data, start_scenario, wait_for

pytestmark = [pytest.mark.e2e, requires_built_data]

SUEZ_TEXT = "The container vessel Ever Forward has run aground in the Suez Canal, blocking all traffic."


def data(response: httpx.Response, status: int = 200):
    assert response.status_code == status, response.text
    return response.json()["data"]


def server_log(server) -> list[dict]:
    """The JSON lines the server wrote to its log (uvicorn's own plain-text lines are skipped)."""
    entries = []
    for line in server.log.read_text(errors="replace").splitlines():
        if line.startswith("{"):
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def test_a_run_leaves_json_log_lines_in_the_servers_log_all_tied_to_its_simulation(server, api):
    started = api.post("/api/scenarios/SUEZ_CLOSURE/run", json={}, headers={"X-Request-ID": "e2e-trace-1"})
    sim = data(started, 202)["simulation_id"]
    assert started.headers["x-request-id"] == "e2e-trace-1"
    finish(api, sim)

    deadline = time.time() + 10  # the worker's last line may land just after the status flips
    while time.time() < deadline and not any(e.get("event") == "run_finished" and e.get("simulation_id") == sim for e in server_log(server)):
        time.sleep(0.1)
    mine = [e for e in server_log(server) if e.get("simulation_id") == sim]
    events = [(e.get("event"), e.get("step") or e.get("checkpoint") or e.get("outcome")) for e in mine]
    assert ("run_started", None) in events and ("run_finished", "COMPLETED") in events and ("run_outcome", "COMPLETED") in events
    assert [e[1] for e in events if e[0] == "step"] == ["sense", "agents", "optimize", "compliance"]
    assert [e[1] for e in events if e[0] == "checkpoint"] == ["simulation_created", "event_sensed", "agents_assessed", "plan_optimized", "compliance_checked", "plan_finalized"]
    run_lines = [e for e in mine if e.get("event") in ("run_started", "step", "run_outcome", "run_finished")]
    assert {e["request_id"] for e in run_lines} == {"e2e-trace-1"} and len({e["run_id"] for e in run_lines}) == 1  # the worker thread's lines name the request that started it
    request = next(e for e in server_log(server) if e.get("event") == "request" and e.get("request_id") == "e2e-trace-1")  # a scenario run's URL names no simulation, so find it by request id
    assert request["status"] == 202 and request["route"] == "/api/scenarios/{scenario_id}/run" and request["level"] == "INFO"
    assert all(set(e) >= {"ts", "level", "logger", "message"} for e in mine)
    assert not [e for e in mine if e["level"] in ("ERROR", "CRITICAL")]  # a healthy run logs no error


def test_an_error_names_its_request_and_the_same_id_is_in_the_servers_log(server, api):
    response = api.get("/api/simulations/sim-does-not-exist", headers={"X-Request-ID": "e2e-trace-404"})
    assert response.status_code == 404 and response.json()["error"]["request_id"] == "e2e-trace-404" and response.headers["x-request-id"] == "e2e-trace-404"
    deadline = time.time() + 5
    line = None
    while time.time() < deadline and line is None:
        line = next((e for e in server_log(server) if e.get("request_id") == "e2e-trace-404" and e.get("event") == "request"), None)
        time.sleep(0.1)
    assert line and line["status"] == 404 and line["simulation_id"] == "sim-does-not-exist" and line["route"] == "/api/simulations/{simulation_id}"


def test_metrics_and_readiness_over_real_http(server, api):
    def llm_calls() -> float:
        return sum(r["value"] for r in data(api.get("/api/metrics"))["counters"].get("llm_calls_total", []))

    calls_before = llm_calls()  # other tests share this server and may have called the (scripted) model
    finish(api, start_scenario(api, "SUEZ_CLOSURE"))
    snap = data(api.get("/api/metrics"))
    completed = next(r["value"] for r in snap["counters"]["runs_total"] if r["labels"] == {"outcome": "COMPLETED"})
    assert completed >= 1 and snap["uptime_seconds"] > 0
    routes = {r["labels"]["route"] for r in snap["counters"]["http_requests_total"]}
    assert "/api/scenarios/{scenario_id}/run" in routes and not any("sim-" in r for r in routes)  # templates, not one series per simulation
    assert {r["labels"]["step"] for r in snap["summaries"]["run_step_duration_ms"]} >= {"sense", "agents", "optimize", "compliance"}
    assert llm_calls() == calls_before  # a scenario's trigger is structured: running one never calls the model

    text = api.get("/api/metrics?format=prometheus")
    assert text.headers["content-type"].startswith("text/plain") and "# TYPE runs_total counter" in text.text and 'runs_total{outcome="COMPLETED"}' in text.text
    for line in text.text.splitlines():  # every sample line is `name{labels} number`
        assert line.startswith("#") or re.fullmatch(r"[a-z_]+(\{.*\})? -?[0-9.e+-]+", line), line

    ready = api.get("/api/ready")
    body = data(ready)
    assert ready.status_code == 200 and body["ready"] is True and {c["name"] for c in body["checks"]} >= {"database", "datasets", "forecast_model"}

    model = data(api.get("/api/monitoring/model?backtest_product_id=22197"))
    assert model["inference_count"] > 0 and model["backtest"]["product_id"] == "22197" and model["drift"]["by_product"]["22197"]["status"] in ("OK", "DRIFT")


def test_alive_is_not_ready_when_a_required_file_is_missing(make_server):
    server = make_server(env={"E2E_EXTRA_REQUIRED_FILE": "data/processed/this_file_does_not_exist.csv"}).start()
    with server.client() as api:
        assert api.get("/api/health").status_code == 200  # the process is up...
        ready = api.get("/api/ready")
        body = ready.json()["data"]
        assert ready.status_code == 503 and body["ready"] is False  # ...and cannot do its job
        failing = [c for c in body["checks"] if c["required"] and not c["ok"]]
        assert [c["name"] for c in failing] == ["datasets"] and "this_file_does_not_exist.csv" in failing[0]["detail"]


def test_a_provider_outage_trips_the_circuit_and_later_calls_fail_fast_without_calling_it(make_server):
    server = make_server(env={"E2E_LLM_FAIL": "1"}).start().warm()
    with server.client() as api:
        outcomes = []
        for _ in range(4):
            sim = free_text_run(api, SUEZ_TEXT)
            status = finish(api, sim)
            outcomes.append((status["run"]["outcome"]["outcome"], status["run"]["outcome"]["message"]))
        assert [o[0] for o in outcomes] == ["SENSING_ERROR"] * 4  # a run never hangs and never fakes success
        assert all("scripted outage" in m for _, m in outcomes[:3])
        assert "circuit open" in outcomes[3][1]  # the fourth was refused without a call

        snap = data(api.get("/api/metrics"))
        calls = {r["labels"]["result"]: r["value"] for r in snap["counters"]["llm_calls_total"]}
        assert calls == {"error": 4} and snap["counters"]["llm_circuit_open_total"][0]["value"] == 1
        assert {r["labels"]["outcome"] for r in snap["counters"]["runs_total"]} == {"SENSING_ERROR"}

        ready = data(api.get("/api/ready"))  # an optional dependency: degraded, not down
        circuit = next(c for c in ready["checks"] if c["name"] == "llm_circuit")
        assert ready["ready"] is True and ready["degraded"] is True and circuit["ok"] is False and "open" in circuit["detail"]
        assert api.post("/api/scenarios/SUEZ_CLOSURE/run", json={}).status_code == 202  # scenarios do not need the model
        assert any(e.get("event") == "llm_unavailable" and e["level"] == "ERROR" for e in server_log(server))


def test_a_run_past_its_limit_is_reported_overdue_while_it_runs_and_not_after(make_server):
    server = make_server(env={"E2E_AGENT_DELAY": "3", "RUN_OVERDUE_SECONDS": "1"}).start().warm()
    with server.client() as api:
        sim = start_scenario(api, "SUEZ_CLOSURE")
        late = wait_for(api, f"/api/simulations/{sim}/status", lambda s: s["run"] and s["run"]["overdue"], timeout=20)
        assert late["run"]["state"] == "RUNNING" and late["run"]["elapsed_ms"] > 1000
        done = finish(api, sim, timeout=60)
        assert done["run"]["overdue"] is False and done["status"] == "COMPLETED"  # it finished, so it is no longer late
