"""Phase 19 — what an operator sees in the UI: the Operations panel, readiness, and a run that is late."""
from __future__ import annotations

import re

import pytest

from backend.tests.e2e.conftest import finish, requires_browser, requires_built_data, start_scenario

pytestmark = [pytest.mark.e2e, pytest.mark.browser, requires_built_data, requires_browser]


def data(response):
    assert response.status_code == 200, response.text
    return response.json()["data"]


def has(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


def test_the_operations_panel_shows_what_the_backend_counts_and_can_backtest_the_model(make_stack, browser):
    stack = make_stack()
    with stack.client() as api:
        sim = start_scenario(api, "SUEZ_CLOSURE")
        finish(api, sim)

    r = browser.run("operations", {"baseUrl": stack.web.url, "simId": sim, "productId": "22197"})
    assert "error" not in r, r.get("error")
    assert r["problems"] == [], r["problems"]
    before, after = r["before"], r["after"]

    with stack.client() as api:
        snap = data(api.get("/api/metrics"))
        model = data(api.get("/api/monitoring/model"))
        ready = data(api.get("/api/ready"))

    # the pipeline card is the backend's own counters
    completed = next(x["value"] for x in snap["counters"]["runs_total"] if x["labels"] == {"outcome": "COMPLETED"})
    assert has(before, "Pipeline runs") and has(before, f"COMPLETED {int(completed)}")
    for step in snap["summaries"]["run_step_duration_ms"]:
        assert re.search(rf"{step['labels']['step']}\s+[\d.]+m?s / [\d.]+m?s \(×{step['count']}\)", before, re.I), step
    assert has(before, "Solves") and has(before, "OPTIMAL")
    assert has(before, "Language model") and has(before, "never calls the model")  # this run's trigger was structured: honest about it

    # the model card: real inferences, and drift measured in standard deviations
    assert re.search(rf"Predictions\s+{model['inference_count'] - 14}\b", before), before[before.find("Demand model"):][:400]
    assert has(before, "2026.09.1") and re.search(r"Drift \(threshold 3(\.0)?σ\)", before, re.I) and has(before, "Not measured yet")
    drift = model["drift"]["by_product"]["22197"]  # the state now: the backtest re-checked the product from an earlier date, so compare with the page after it
    shown = f"{'+' if drift['z'] > 0 else ''}{drift['z']:g}"
    assert re.search(rf"product 22197\s+{drift['status']} {re.escape(shown)}σ", after, re.I), (drift, shown)

    # the backtest button runs one and shows its error rate
    assert r["backtestClick"] == "ok" and r["wape"]
    api_wape = model["backtests"]["22197"]["wape"]
    assert f"WAPE {round(api_wape * 100)}%" in after.upper().replace("WAPE", "WAPE")
    assert has(after, "product 22197 · 14d to")

    # readiness lists every check the backend reported
    assert has(before, "READY") and all(has(before, c["name"]) for c in ready["checks"])


def test_a_backend_that_is_alive_but_not_ready_is_shown_as_such_with_the_reason(make_stack, browser):
    stack = make_stack(env={"E2E_EXTRA_REQUIRED_FILE": "data/processed/this_file_does_not_exist.csv"})
    r = browser.run("notready", {"baseUrl": stack.web.url})
    assert "error" not in r, r.get("error")
    assert has(r["header"], "API Not Ready") and not has(r["header"], "API Connected")  # alive is not ready
    assert r["chipTitle"] and "this_file_does_not_exist.csv" in r["chipTitle"] and "datasets" in r["chipTitle"]
    assert r["readiness"] and has(r["text"], "NOT READY") and has(r["text"], "run the Phase 3-5 pipelines")


def test_a_run_past_its_limit_is_marked_overdue_in_the_simulator_and_clears_when_it_finishes(make_stack, browser):
    stack = make_stack(env={"E2E_AGENT_DELAY": "3", "RUN_OVERDUE_SECONDS": "1"})
    r = browser.run("overdue", {"baseUrl": stack.web.url})
    assert "error" not in r, r.get("error")
    assert r["run"] == "ok" and r["overdue"] and has(r["overdue"], "OVERDUE")
    assert has(r["whileOverdue"], "longer than the backend's limit") and has(r["whileOverdue"], "check the backend log")
    assert r["finished"] and not has(r["after"], "OVERDUE")  # it finished, so the warning is gone
