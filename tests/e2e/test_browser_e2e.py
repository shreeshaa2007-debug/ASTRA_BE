"""Phase 18 — the user's journeys, in a real browser against real processes.

Headless Chrome (over the DevTools protocol, backend/tests/e2e/browser/) drives the Vite-served frontend,
which talks to a real uvicorn process with a real database and the scripted LLM. Each test is a journey
a person would make; the page's numbers are checked against the API rather than against constants, so
they fail when the UI and the backend disagree, not when a demo value changes.

Skipped without node, `npm install` in frontend/, or Chrome/Edge (set CHROME_PATH). Screenshots of every
journey are written under the pytest tmp directory (see the `artifacts` folder of a failing test).
"""
from __future__ import annotations

import re

import pytest

from backend.tests.e2e.conftest import finish, requires_browser, requires_built_data, start_scenario

pytestmark = [pytest.mark.e2e, pytest.mark.browser, requires_built_data, requires_browser]

SUEZ = {"SHA-ROT-SUEZ", "SIN-ROT-SUEZ", "MUM-ROT-SUEZ", "CHE-ROT-SUEZ"}
BAD_TEXT = ("undefined", "NaN", "[object Object]")


@pytest.fixture(scope="module")
def stack(make_stack):
    return make_stack()


@pytest.fixture()
def api(stack):
    with stack.client() as c:
        yield c


def data(response):
    assert response.status_code == 200, response.text
    return response.json()["data"]


def n(x: float) -> str:
    """How the UI formats a whole number (toLocaleString, en-US)."""
    return f"{round(x):,}"


def has(text: str, needle: str) -> bool:
    """Case-insensitive: several labels are CSS-uppercased, and innerText returns them uppercased."""
    return needle.lower() in text.lower()


def assert_clean(result: dict, *, allow_problems=()) -> None:
    assert "error" not in result, result.get("error")
    unexpected = [p for p in result["problems"] if not any(a in p for a in allow_problems)]
    assert unexpected == [], "the page logged errors:\n" + "\n".join(unexpected)


# =========================================================================== #
def test_every_screen_renders_without_errors_and_agrees_with_the_api(stack, api, browser):
    sim = start_scenario(api, "SUEZ_CLOSURE")
    finish(api, sim)
    decision, comparison = data(api.get(f"/api/decisions/{sim}")), data(api.get(f"/api/simulations/{sim}/comparison"))

    r = browser.run("screens", {"baseUrl": stack.web.url, "simId": sim})
    assert_clean(r)
    assert has(r["header"], "API Connected") and set(r["navResult"].values()) == {"ok"}

    for where in ("empty", "withSim"):  # nothing on any screen is a JavaScript accident
        for title, text in r[where].items():
            assert not [bad for bad in BAD_TEXT if bad in text], f"{where}/{title} shows a broken value:\n{text[:600]}"
            assert len(text) > 80, f"{where}/{title} rendered almost nothing"

    # the UI talks to itself and its backend, nothing else: no font, tile or script from anywhere on the internet, so it looks the same offline
    origins = {re.match(r"(https?://[^/]+)", u).group(1) for u in r["requests"] if u.startswith("http")}
    assert origins <= {stack.web.url, stack.server.url}, sorted(origins)
    assert any(u.endswith("/land110m.json") for u in r["requests"]) and any(u.endswith("material-symbols-outlined.woff2") for u in r["requests"])  # the bundled map and icon font

    empty = r["empty"]  # no simulation: honest empty states, not placeholder data
    assert has(empty["Overview"], "Global Supply Chain Command Center") and has(empty["Overview"], "ROUTES AVAILABLE")
    for title in ("AI Decisions", "Compliance & Approvals"):
        assert has(empty[title], "No simulation yet") and has(empty[title], "no placeholder data")
    assert has(empty["Disruption Simulator"], "NO RUN YET")

    shown = r["withSim"]
    assert sim in r["sidebar"]
    decisions = shown["AI Decisions"]
    assert sim in decisions and has(decisions, "OPTIMAL") and has(decisions, "Compared with normal operations and with doing nothing")
    assert abs(int(re.search(r"PLAN SPEND\s+([\d,]+) cost units", decisions, re.I).group(1).replace(",", "")) - decision["plan_spend"]) <= 1
    assert has(decisions, f"{comparison['deltas']['units_protected']:,} units secured")
    assert has(shown["Compliance & Approvals"], "IN POLICY") and sim in shown["Compliance & Approvals"]
    assert sim in shown["Inventory"] and "Mumbai" in shown["Inventory"] and "Chennai" in shown["Inventory"]
    assert has(shown["Sourcing"], "Recommended sourcing mix") and decision["allocations"][0]["supplier_id"] in shown["Sourcing"]
    logistics = shown["Logistics"]
    assert all(route in logistics for route in SUEZ) and logistics.count("DISRUPTED") >= 4 and "PLANNED" in logistics
    assert has(shown["Agent Monitor"], "plan_finalized") and has(shown["Agent Monitor"], "6 entries")
    assert has(shown["Disruption Simulator"], "PIPELINE COMPLETE")
    assert all(has(shown["Scenarios"], label) for label in ("Suez Canal Closure", "Supplier Failure", "Demand Surge"))


def test_the_scenarios_page_shows_the_apis_comparison_and_runs_a_scenario_through_the_pipeline(make_stack, browser):
    # a backend whose agents take a second, so the moment between choosing a scenario and its comparison arriving is long enough to look at
    stack = make_stack(env={"E2E_AGENT_DELAY": "1.0"})
    with stack.client(timeout=120) as api:
        return _scenarios_journey(stack, api, browser)


def _scenarios_journey(stack, api, browser):
    scenarios = data(api.get("/api/scenarios"))
    r = browser.run("scenarios", {
        "baseUrl": stack.web.url, "scenarios": [{"label": s["label"], "modeled": s["modeled"]} for s in scenarios], "pipelineLabel": "Supplier Failure"})
    assert_clean(r)

    for s in scenarios:
        page = r["pages"][s["label"]]
        assert page["clicked"] == "ok"
        if not s["modeled"]:
            assert has(re.sub(r"\s+", " ", page["text"]), s["not_modeled_reason"]) and has(page["text"], "Not simulated")
            continue
        c = data(api.get(f"/api/scenarios/{s['scenario_id']}/comparison?product_id={s['default_product_id']}"))
        text = page["text"]
        changes = [f"route {x}" for x in c["newly_disrupted_routes"]] + [f"supplier {x}" for x in c["newly_disrupted_suppliers"]] + [f"{t['iso3']} tariff" for t in c["tariff_changes"]]
        # never another scenario's numbers under this scenario's name: right after the choice the page is either still loading or already this scenario's
        assert has(page["immediate"], "Solving the baseline") or all(has(page["immediate"], x) for x in changes), (
            f"{s['label']}: the page showed a different scenario's comparison right after the switch: " + page["immediate"][:900].replace(chr(10), " | "))
        for case in ("baseline", "unmitigated", "mitigated"):  # every case's spend, as the API computed it
            assert has(text, f"{n(c[case]['spend'])} cost units"), f"{s['scenario_id']}: {case} spend {n(c[case]['spend'])} not on the page"
        if c["deltas"]["units_protected"]:
            assert has(text, f"{c['deltas']['units_protected']:,} units secured")
        else:
            assert has(text, "Nothing lost to protect")
        assert has(text, f"{'+' if c['deltas']['mitigation_cost'] >= 0 else ''}{n(c['deltas']['mitigation_cost'])} cost units")
        assert all(x in text for x in c["newly_disrupted_routes"] + c["newly_disrupted_suppliers"])  # what the scenario changes is on the page

    assert r["pipelineClick"] == "ok" and r["runClick"] == "ok" and re.search(r"AWAITING HUMAN APPROVAL", r["landed"] or "", re.I), r["landed"]
    state = data(api.get(f"/api/simulations/{r['simId']}"))
    assert state["scenario_type"] == "SUPPLIER_FAILURE" and state["status"] == "AWAITING_APPROVAL" and state["supplier_status"]["S007"] == "DISRUPTED"
    spend = data(api.get(f"/api/decisions/{r['simId']}"))["plan_spend"]
    assert n(spend) in r["simulatorText"] and has(r["simulatorText"], "Proceed to Approvals")  # the simulator shows the plan the API holds


def test_a_human_approves_and_rejects_in_the_ui_and_a_lost_race_shows_the_truth(stack, api, browser):
    sims = {k: start_scenario(api, "SUPPLIER_FAILURE") for k in ("approve", "reject", "conflict")}
    for sim in sims.values():
        assert finish(api, sim)["status"] == "AWAITING_APPROVAL"

    r = browser.run("approval", {"baseUrl": stack.web.url, "apiUrl": stack.server.url, "approveSim": sims["approve"], "rejectSim": sims["reject"],
                                 "conflictSim": sims["conflict"], "approver": "Dana Approver"})
    # the one console error allowed is the browser reporting the 409 it was meant to get in the lost race
    assert_clean(r, allow_problems=("409",))
    assert has(r["approveBefore"], "HUMAN APPROVAL REQUIRED")

    assert (r["approveDisabledWithoutName"], r["approveClick"]) == ("DISABLED", "ok")  # no name, no decision
    assert "PLAN APPROVED BY DANA APPROVER" in r["approveDone"].upper()
    approved = data(api.get(f"/api/simulations/{sims['approve']}/status"))
    assert approved["status"] == "COMPLETED" and approved["timeline"][-1]["actor"] == "Dana Approver"
    assert has(r["monitorAfterApprove"], "plan_finalized") and has(r["monitorAfterApprove"], "Dana Approver")
    assert sims["approve"] in r["dashboardAfterApprove"]  # the newly finalized plan is what the dashboard now shows

    assert (r["rejectDisabledWithoutName"], r["rejectClick"]) == ("DISABLED", "ok")
    assert "PLAN REJECTED BY DANA APPROVER" in r["rejectDone"].upper()
    assert data(api.get(f"/api/simulations/{sims['reject']}/status"))["status"] == "REJECTED"

    assert r["rivalStatus"] == 200 and r["conflictClick"] == "ok"
    assert r["conflictDone"] and "RIVAL APPROVER" in r["conflictDone"].upper()  # the refused click made the page re-read the truth
    decided = data(api.get(f"/api/simulations/{sims['conflict']}"))
    assert decided["approval_decision"]["decided_by"] == "Rival Approver" and decided["status"] == "COMPLETED"  # the loser did not overwrite the winner


def test_free_text_reports_through_the_simulator_and_what_the_page_remembers(stack, api, browser):
    r = browser.run("freetext", {"baseUrl": stack.web.url})
    assert_clean(r)
    assert has(r["empty"], "NO RUN YET")

    assert r["weather1"]["clicked"] == "ok" and re.search("NO DISRUPTION SENSED", r["weather1"]["done"] or "", re.I)
    assert has(r["weatherText"], "Nothing in this report concerns the supply chain")
    assert r["weatherSim1"] == r["weatherSim2"] and re.search("NO DISRUPTION SENSED", r["weather2"]["done"] or "", re.I)  # a still-CREATED simulation is re-run, not replaced
    assert data(api.get(f"/api/simulations/{r['weatherSim1']}/status"))["status"] == "CREATED"

    assert re.search("AWAITING HUMAN APPROVAL", r["fire"]["done"] or "", re.I) and r["fireSim"] != r["weatherSim1"]
    assert has(r["fireText"], "Proceed to Approvals") and data(api.get(f"/api/simulations/{r['fireSim']}"))["supplier_status"]["S007"] == "DISRUPTED"

    assert re.search("PIPELINE COMPLETE", r["suez"]["done"] or "", re.I) and r["suezSim"] not in (r["fireSim"], r["weatherSim1"])
    routes = data(api.get(f"/api/simulations/{r['suezSim']}"))["route_status"]
    assert {k for k, v in routes.items() if v == "DISRUPTED"} == SUEZ and has(r["suezText"], "Auto-approved")

    assert r["afterReloadSim"] == r["suezSim"] and r["suezSim"] in r["reloadSidebar"]  # a reload keeps the simulation...
    assert has(r["afterReload"], "PIPELINE COMPLETE")  # ...and shows where it got to, without running anything again

    assert r["typeCustom"] == "ok" and r["runWhenTooLong"] == "DISABLED" and has(r["counter"], "4001/4000")  # the backend's limit, enforced before the request
    assert r["runWhenEmpty"] == "DISABLED"


def test_the_ui_survives_the_backend_going_away_and_recovers_without_a_reload(make_stack, browser):
    stack = make_stack()
    with stack.client() as api:
        sim = start_scenario(api, "SUEZ_CLOSURE")
        finish(api, sim)

    r = browser.run("outage", {"baseUrl": stack.web.url, "simId": sim}, hooks={"stop": stack.server.kill, "start": stack.server.start})
    assert "error" not in r, r.get("error")
    assert has(r["onlineHeader"], "API Connected") and has(r["onlineDashboard"], "ROUTES AVAILABLE")

    assert (r["stop"], r["start"]) == ("ok", "ok")
    assert has(r["downHeader"], "API Unreachable")
    for key, heading in (("downScenarios", "Scenario Comparison"), ("downInventory", "Inventory Intelligence"), ("downDashboard", "Global Supply Chain Command Center")):
        text = r[key]
        assert has(text, heading), f"{key}: the screen stopped rendering"  # it degrades, it does not crash
        assert has(text, "API_UNREACHABLE") and has(text, "Could not reach the backend") and has(text, "uvicorn backend.api.main:app")  # and says how to fix it
    assert not [p for p in r["downProblems"] if p.startswith("EXCEPTION")], r["downProblems"]  # failed requests are logged; unhandled exceptions are not

    assert has(r["upHeader"], "API Connected") and r["retry"] == "ok"
    assert has(r["recoveredDashboard"], "ROUTES AVAILABLE") and not has(r["recoveredDashboard"], "API_UNREACHABLE")  # back, with no reload
    assert has(r["recoveredScenarios"], "Suez Canal Closure") and r["problems"] == []
