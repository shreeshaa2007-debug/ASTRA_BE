"""Phase 18 — end-to-end over real HTTP, brief §21 "END-TO-END".

Every test talks to a real uvicorn process over a real socket, backed by a real SQLite file, the real
thread pool, all five agents, the optimizer and the compliance rules from backend/config/. The one
substitution is the LLM (backend/tests/e2e/app.py: a deterministic script). What these tests add to the
in-process suites is what only a real process can show: that every endpoint tells the same story, that
concurrent runs stay apart, that the state outlives the process, and what a crash leaves behind.
"""
from __future__ import annotations

import os
import time

import httpx
import pytest

from backend.tests.e2e.conftest import (
    finish,
    free_text_run,
    requires_built_data,
    start_scenario,
    wait_for,
)

pytestmark = [pytest.mark.e2e, requires_built_data]

SUEZ = {"SHA-ROT-SUEZ", "SIN-ROT-SUEZ", "MUM-ROT-SUEZ", "CHE-ROT-SUEZ"}
SUEZ_TEXT = "The container vessel Ever Forward has run aground in the Suez Canal, blocking all traffic."
FIRE_TEXT = "Fire at the Istanbul plant"
WEATHER_TEXT = "The weather in Lisbon looks lovely this weekend, with sunshine and light winds."


def data(response: httpx.Response, status: int = 200):
    assert response.status_code == status, response.text
    return response.json()["data"]


def error(response: httpx.Response, status: int, code: str) -> dict:
    assert response.status_code == status, response.text
    err = response.json()["error"]
    assert err["status"] == "error" and err["error_code"] == code and err["message"], err
    assert "Traceback" not in response.text and "File \"" not in response.text  # no stack trace ever leaves the process
    return err


# =========================================================================== #
# one story, told by every endpoint
# =========================================================================== #
def test_a_scenario_runs_end_to_end_and_every_endpoint_tells_the_same_story(api):
    sim = start_scenario(api, "SUEZ_CLOSURE")
    status = finish(api, sim)
    assert (status["status"], status["run"]["outcome"]["outcome"], status["awaiting_approval"], status["stalled"]) == ("COMPLETED", "COMPLETED", False, False)
    assert [(a["id"], a["status"]) for a in status["agents"]] == [
        ("sensing", "COMPLETE"), ("inventory", "COMPLETE"), ("logistics", "COMPLETE"), ("sourcing", "COMPLETE"),
        ("optimization", "COMPLETE"), ("compliance", "COMPLETE"), ("human_approval", "NOT_REQUIRED")]
    assert [t["checkpoint"] for t in status["timeline"]] == [
        "simulation_created", "event_sensed", "agents_assessed", "plan_optimized", "compliance_checked", "plan_finalized"]
    assert [t["version"] for t in status["timeline"]] == list(range(6))

    decision, compliance = data(api.get(f"/api/decisions/{sim}")), data(api.get(f"/api/compliance/{sim}"))
    comparison, shipments = data(api.get(f"/api/simulations/{sim}/comparison")), data(api.get(f"/api/shipments?simulation_id={sim}"))
    routes = {r["route_id"]: r for r in data(api.get(f"/api/routes?simulation_id={sim}"))["routes"]}
    suppliers = data(api.get(f"/api/suppliers?simulation_id={sim}"))
    inventory = data(api.get(f"/api/inventory?simulation_id={sim}"))
    dashboard = data(api.get("/api/dashboard"))

    # the plan's price is the same number wherever it appears
    spend = decision["plan_spend"]
    assert spend == pytest.approx(compliance["plan_spend"]) == pytest.approx(comparison["mitigated"]["spend"], abs=0.01) == pytest.approx(dashboard["kpis"]["plan_spend"])
    assert decision["plan_status"] == "OPTIMAL" and decision["approval"]["status"] == "NOT_REQUIRED" and compliance["compliance"]["status"] == "APPROVED"

    # what is bought is what is shipped is what each route carries
    units = sum(a["quantity"] for a in decision["allocations"])
    assert units == sum(s["quantity"] for s in shipments["shipments"]) == comparison["mitigated"]["units_delivered"]
    assert {s["status"] for s in shipments["shipments"]} == {"PLANNED"}  # finalized
    for route_id, route in routes.items():
        assert route["planned_quantity"] == sum(a["quantity"] for a in decision["allocations"] if a["route_id"] == route_id)
    assert {m["supplier_id"]: m["quantity"] for m in suppliers["recommended_mix"]} == {s: v["units"] for s, v in decision["supplier_split"].items()}
    assert all(routes[r]["status"] == "DISRUPTED" for r in SUEZ) and sum(r["status"] == "DISRUPTED" for r in routes.values()) == 4
    assert not any(a["route_id"] in SUEZ for a in decision["allocations"])  # the plan never uses a closed lane

    # the dashboard follows the latest finalized simulation — this one — and agrees with the comparison
    assert dashboard["simulation_id"] == sim and dashboard["kpis"]["routes_disrupted"] == 4 and dashboard["kpis"]["active_disruptions"] == 1
    assert dashboard["kpis"]["shipments_at_risk"] == comparison["exposure"]["shipments_at_risk"] == 1
    assert dashboard["kpis"]["estimated_exposure"] == pytest.approx(comparison["exposure"]["value_at_risk"])
    assert (inventory["source"], len(inventory["inventory"])) == ("simulation", 3)
    assert comparison["mitigated"]["units_short"] == 0 and comparison["unmitigated"]["units_short"] == comparison["exposure"]["units_at_risk"]


# =========================================================================== #
# the human in the loop
# =========================================================================== #
def test_an_escalated_plan_waits_for_a_named_human_and_the_decision_is_final(api):
    sim = start_scenario(api, "SUPPLIER_FAILURE")
    status = finish(api, sim)
    assert (status["status"], status["awaiting_approval"]) == ("AWAITING_APPROVAL", True)
    compliance = data(api.get(f"/api/compliance/{sim}"))
    assert compliance["compliance"]["status"] == "ESCALATED" and "500,000.00 approval threshold" in compliance["compliance"]["reason"]
    assert {s["status"] for s in data(api.get(f"/api/shipments?simulation_id={sim}"))["shipments"]} == {"PROPOSED"}  # not yet a commitment
    assert data(api.get("/api/dashboard"))["simulation_id"] != sim  # the dashboard shows finalized work only

    version = compliance["version"]
    error(api.post(f"/api/decisions/{sim}/approve", json={"decided_by": "   "}), 422, "VALIDATION_ERROR")  # nobody is not a decider
    error(api.post(f"/api/decisions/{sim}/approve", json={"decided_by": "Dana", "expected_version": version - 1}), 409, "STATE_CONFLICT")
    assert data(api.get(f"/api/simulations/{sim}/status"))["status"] == "AWAITING_APPROVAL"  # neither refusal changed anything

    decided = data(api.post(f"/api/decisions/{sim}/approve", json={"decided_by": "Dana Approver", "note": "within contingency budget", "expected_version": version}))
    assert (decided["status"], decided["approval_status"], decided["approval_decision"]["decided_by"]) == ("COMPLETED", "APPROVED", "Dana Approver")
    assert {s["status"] for s in data(api.get(f"/api/shipments?simulation_id={sim}"))["shipments"]} == {"PLANNED"}
    assert data(api.get("/api/dashboard"))["simulation_id"] == sim
    timeline = data(api.get(f"/api/simulations/{sim}/status"))["timeline"]
    assert timeline[-1]["checkpoint"] == "plan_finalized" and timeline[-1]["actor"] == "Dana Approver"  # the audit trail names the human

    error(api.post(f"/api/decisions/{sim}/approve", json={"decided_by": "Dana Approver"}), 409, "INVALID_STATE_TRANSITION")  # a decision is made once
    error(api.post(f"/api/decisions/{sim}/reject", json={"decided_by": "Someone Else"}), 409, "INVALID_STATE_TRANSITION")


def test_a_rejected_plan_closes_the_simulation_and_a_reset_reopens_it(api):
    sim = start_scenario(api, "SUPPLIER_FAILURE")
    finish(api, sim)
    version = data(api.get(f"/api/compliance/{sim}"))["version"]
    rejected = data(api.post(f"/api/decisions/{sim}/reject", json={"decided_by": "Dana Approver", "note": "too dear", "expected_version": version}))
    assert (rejected["status"], rejected["approval_status"]) == ("REJECTED", "REJECTED")
    assert data(api.get(f"/api/decisions/{sim}"))["plan_status"] == "OPTIMAL"  # the rejected plan is still there to read
    error(api.post(f"/api/simulations/{sim}/run", json={"signal": SUEZ_TEXT, "product_id": "22197"}), 409, "INVALID_STATE_TRANSITION")

    reset = data(api.post(f"/api/simulations/{sim}/reset"))
    assert reset["status"] == "CREATED" and reset["current_plan"] is None and reset["approval_status"] == "NOT_EVALUATED"
    assert reset["route_status"] and all(s != "DISRUPTED" for s in reset["route_status"].values())  # back to the network it was created with
    assert api.post(f"/api/simulations/{sim}/run", json={"signal": SUEZ_TEXT, "product_id": "22197"}).status_code == 202
    assert finish(api, sim)["status"] == "COMPLETED"
    history = [t["checkpoint"] for t in data(api.get(f"/api/simulations/{sim}/status"))["timeline"]]
    assert "plan_rejected_by_human" in history and "simulation_reset" in history  # the audit trail keeps what was undone


# =========================================================================== #
# free text
# =========================================================================== #
def test_free_text_reports_are_sensed_validated_and_planned(api):
    suez = free_text_run(api, SUEZ_TEXT)
    status = finish(api, suez)
    assert status["status"] == "COMPLETED" and "canal_closure at Suez Canal" in status["agents"][0]["detail"]
    state = data(api.get(f"/api/simulations/{suez}"))
    assert {r for r, s in state["route_status"].items() if s == "DISRUPTED"} == SUEZ and state["current_disruptions"][0]["event_id"].startswith("evt-")

    fire = free_text_run(api, FIRE_TEXT)
    assert finish(api, fire)["status"] == "AWAITING_APPROVAL"
    assert data(api.get(f"/api/simulations/{fire}"))["supplier_status"]["S007"] == "DISRUPTED"


def test_a_report_about_nothing_leaves_the_simulation_free_to_run_again(api):
    sim = free_text_run(api, WEATHER_TEXT)
    status = finish(api, sim)
    assert (status["status"], status["run"]["outcome"]["outcome"], status["agents"][0]["status"]) == ("CREATED", "NO_DISRUPTION", "NO_EVENT")
    assert "Nothing in this report" in status["run"]["outcome"]["message"] and [t["checkpoint"] for t in status["timeline"]] == ["simulation_created"]
    assert api.post(f"/api/simulations/{sim}/run", json={"signal": SUEZ_TEXT, "product_id": "22197"}).status_code == 202  # same simulation, new report
    assert finish(api, sim)["status"] == "COMPLETED"


def test_a_bad_report_or_product_is_refused_at_the_door_not_discovered_by_polling(api):
    sim = data(api.post("/api/simulations", json={"scenario_type": "E2E"}), 201)["simulation_id"]
    for body, fragment in (
        ({"signal": "", "product_id": "22197"}, "signal"), ({"signal": "x" * 4001, "product_id": "22197"}, "4001"),
        ({"signal": SUEZ_TEXT, "product_id": "NOPE"}, "unknown product_id"), ({"product_id": "22197"}, "signal"),
    ):
        assert fragment in error(api.post(f"/api/simulations/{sim}/run", json=body), 422, "VALIDATION_ERROR")["message"]
    assert data(api.get(f"/api/simulations/{sim}/status"))["run"] is None  # nothing was started


# =========================================================================== #
# concurrency, durability, crashes
# =========================================================================== #
def test_simultaneous_runs_do_not_interfere_and_a_duplicate_run_is_refused(make_server):
    # the agents step is slowed so the runs genuinely overlap; the "LLM" takes a second so a free-text simulation stays CREATED while its run is active
    server = make_server(env={"E2E_AGENT_DELAY": "1.5", "E2E_LLM_DELAY": "1.0"}).start().warm()
    with server.client() as api:
        alone = time.time()
        finish(api, start_scenario(api, "SUEZ_CLOSURE"), timeout=120)
        one_run = time.time() - alone

        scenarios = ["SUEZ_CLOSURE", "SUPPLIER_FAILURE", "SEVERE_WEATHER", "TARIFF_INCREASE"]
        started = time.time()
        sims = {s: start_scenario(api, s) for s in scenarios}
        for sim in sims.values():
            finish(api, sim, timeout=120)
        assert time.time() - started < 3 * one_run  # four runs took less than three run-lengths: they overlapped rather than queued

        for scenario, sim in sims.items():
            state = data(api.get(f"/api/simulations/{sim}"))
            expected = data(api.get(f"/api/scenarios/{scenario}/comparison"))
            assert state["scenario_type"] == scenario and state["status"] in ("COMPLETED", "AWAITING_APPROVAL")
            assert data(api.get(f"/api/decisions/{sim}"))["plan_spend"] == pytest.approx(expected["mitigated"]["spend"], abs=0.01)
            assert {r for r, s in state["route_status"].items() if s == "DISRUPTED"} == set(expected["newly_disrupted_routes"])  # no state bled between them
            assert {s for s, st in state["supplier_status"].items() if st == "DISRUPTED"} - {"S001"} == set(expected["newly_disrupted_suppliers"])

        # a second run on a simulation whose first run is still in flight is refused, not queued
        sim = free_text_run(api, SUEZ_TEXT)
        error(api.post(f"/api/simulations/{sim}/run", json={"signal": SUEZ_TEXT, "product_id": "22197"}), 409, "RUN_IN_PROGRESS")
        assert finish(api, sim, timeout=90)["status"] == "COMPLETED"  # and the first run was not disturbed by the refusal


def test_the_world_state_outlives_the_process(make_server):
    server = make_server().start().warm()
    with server.client() as api:
        done, waiting = start_scenario(api, "SUEZ_CLOSURE"), start_scenario(api, "SUPPLIER_FAILURE")
        finish(api, done), finish(api, waiting)
        before = {s: (data(api.get(f"/api/simulations/{s}")), data(api.get(f"/api/decisions/{s}"))) for s in (done, waiting)}
        version = before[waiting][1]["version"]

    server.restart()  # a crash: nothing is flushed, nothing is shut down
    with server.client() as api:
        for sim, (state_before, decision_before) in before.items():
            state, decision = data(api.get(f"/api/simulations/{sim}")), data(api.get(f"/api/decisions/{sim}"))
            assert (state["status"], state["version"], state["approval_status"]) == (state_before["status"], state_before["version"], state_before["approval_status"])
            assert decision["plan_spend"] == decision_before["plan_spend"] and decision["allocations"] == decision_before["allocations"]
        assert {s["simulation_id"] for s in data(api.get("/api/simulations"))} == {done, waiting}
        status = data(api.get(f"/api/simulations/{done}/status"))
        assert status["status"] == "COMPLETED" and status["run"] is None and status["stalled"] is False  # the in-memory run record is gone; the state is not

        # the human's decision still works after the restart, because it needs only the state
        decided = data(api.post(f"/api/decisions/{waiting}/approve", json={"decided_by": "Dana Approver", "expected_version": version}))
        assert decided["status"] == "COMPLETED" and data(api.get("/api/dashboard"))["simulation_id"] == waiting


def test_a_crash_mid_run_is_reported_as_stalled_and_recoverable(make_server):
    server = make_server(env={"E2E_AGENT_DELAY": "2.5"}).start().warm()
    with server.client() as api:
        sim = free_text_run(api, SUEZ_TEXT)
        wait_for(api, f"/api/simulations/{sim}/status", lambda s: s["status"] == "RUNNING", timeout=30, every=0.02)  # sensed, now inside the agents step
        server.kill()

    server.start()
    with server.client() as api:
        status = data(api.get(f"/api/simulations/{sim}/status"))
        assert (status["status"], status["stalled"], status["run"], status["current_step"]) == ("RUNNING", True, None, "agents")  # honest: nothing is running it
        assert status["agents"][0]["status"] == "COMPLETE" and status["agents"][1]["status"] != "COMPLETE"  # sensing finished; the agents never did
        error(api.post(f"/api/simulations/{sim}/run", json={"signal": SUEZ_TEXT, "product_id": "22197"}), 409, "INVALID_STATE_TRANSITION")  # not CREATED: it needs a reset

        assert data(api.post(f"/api/simulations/{sim}/reset"))["status"] == "CREATED"
        assert api.post(f"/api/simulations/{sim}/run", json={"signal": SUEZ_TEXT, "product_id": "22197"}).status_code == 202
        recovered = finish(api, sim, timeout=90)
        assert recovered["status"] == "COMPLETED" and recovered["stalled"] is False


# =========================================================================== #
# the edges of the process
# =========================================================================== #
def test_bad_requests_get_the_error_envelope_and_never_a_stack_trace(api):
    error(api.get("/api/nope"), 404, "NOT_FOUND")
    error(api.delete("/api/simulations"), 405, "METHOD_NOT_ALLOWED")
    error(api.get("/api/simulations/sim-does-not-exist/status"), 404, "SIMULATION_NOT_FOUND")
    error(api.post("/api/simulations", content=b"{not json", headers={"content-type": "application/json"}), 422, "VALIDATION_ERROR")
    error(api.post("/api/simulations", json={"scenario_type": ""}), 422, "VALIDATION_ERROR")
    error(api.post("/api/simulations", json={"scenario_type": "X", "simulation_id": "has spaces!"}), 422, "VALIDATION_ERROR")
    error(api.get("/api/scenarios/NOPE/comparison"), 404, "SCENARIO_NOT_FOUND")
    error(api.get("/api/scenarios/DEMAND_SURGE/comparison"), 422, "SCENARIO_NOT_MODELED")
    error(api.get("/api/inventory/forecast?product_id=22197&warehouse_id=Atlantis"), 422, "VALIDATION_ERROR")
    error(api.get("/api/decisions/sim-does-not-exist"), 404, "SIMULATION_NOT_FOUND")
    fresh = data(api.post("/api/simulations", json={"scenario_type": "EMPTY"}), 201)["simulation_id"]
    error(api.get(f"/api/decisions/{fresh}"), 409, "PLAN_NOT_READY")
    error(api.post(f"/api/decisions/{fresh}/approve", json={"decided_by": "Dana"}), 409, "INVALID_STATE_TRANSITION")  # nothing is waiting for a decision


def test_cors_lets_the_configured_browser_origin_in_and_no_one_else(make_server):
    server = make_server(cors_origins="http://127.0.0.1:5999").start()
    with server.client() as api:
        ok = api.options("/api/simulations", headers={"Origin": "http://127.0.0.1:5999", "Access-Control-Request-Method": "POST"})
        assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == "http://127.0.0.1:5999"
        bad = api.options("/api/simulations", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
        assert "access-control-allow-origin" not in bad.headers


def test_the_llm_key_is_reported_as_a_boolean_and_appears_nowhere(make_server):
    secret = "sk-e2e-not-a-real-key-12345"
    server = make_server(env={"LLM_API_KEY": secret}).start()
    with server.client() as api:
        assert api.get("/api/health").json() == {"status": "ok", "mode": "live", "llm_configured": True}
        sim = free_text_run(api, SUEZ_TEXT)
        finish(api, sim)
        for path in ("/api/health", f"/api/simulations/{sim}", f"/api/simulations/{sim}/status", "/openapi.json", "/api/simulations"):
            assert secret not in api.get(path).text
    assert secret not in server.log.read_text(errors="replace")


# =========================================================================== #
# live — the whole stack with the real Gemini. Opt in: RUN_LIVE_LLM_TESTS=1 and LLM_API_KEY.
# =========================================================================== #
@pytest.mark.skipif(not (os.environ.get("RUN_LIVE_LLM_TESTS") == "1" and os.environ.get("LLM_API_KEY")), reason="live LLM tests are opt-in")
def test_live_the_real_model_senses_a_report_and_the_rest_of_the_stack_plans_it(make_server):
    server = make_server(env={"E2E_LLM": "real"}).start().warm()
    with server.client(timeout=120) as api:
        suez = free_text_run(api, "The container vessel Ever Forward has run aground in the Suez Canal, blocking all traffic. Salvage teams expect about 10 days to refloat it.")
        status = finish(api, suez, timeout=120)
        assert status["status"] in ("COMPLETED", "AWAITING_APPROVAL")
        assert {r for r, s in data(api.get(f"/api/simulations/{suez}"))["route_status"].items() if s == "DISRUPTED"} <= SUEZ | {"SIN-ROT-CAPE", "SHA-ROT-CAPE"}
        chat = free_text_run(api, WEATHER_TEXT)
        assert finish(api, chat, timeout=120)["run"]["outcome"]["outcome"] == "NO_DISRUPTION"
