"""Phase 19 tests — observability and error handling, brief §20 / §26.

Layers: the logging machinery (format, correlation ids across a thread pool, redaction), the metrics registry
(arithmetic, bounded memory, Prometheus output), the model monitor (drift, missing features, prediction error),
the LLM circuit breaker (with a fake clock), and then all of it through the real API and pipeline: a run leaves
the log lines, counters and timings it should, every failure carries a request id, and the readiness check names
what is missing.
"""
from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from backend.agents.sensing.agent import SensingAgent
from backend.agents.sensing.llm import CircuitBreakerLLM, LLMResponse, LLMUnavailableError
from backend.api import readiness
from backend.api.context import AppContext
from backend.api.main import create_app
from backend.api.runs import RunRecord, RunRegistry
from backend.database.world_state_repository import SqlAlchemyWorldStateRepository
from backend.monitoring import Metrics, ModelMonitor, bind, configure_logging, current, load_settings, metrics, monitor, redact
from backend.monitoring.model_monitor import TrainingStats, load_training_stats
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


@pytest.fixture(autouse=True)
def clean_observability():
    metrics.reset()
    monitor.reset()
    yield
    configure_logging("INFO", "text")  # never leave a test's capture buffer attached


def capture(level: str = "DEBUG") -> io.StringIO:
    buffer = io.StringIO()
    configure_logging(level, "json", stream=buffer)
    return buffer


def lines(buffer: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in buffer.getvalue().splitlines() if line.strip()]


# =========================================================================== #
# logging
# =========================================================================== #
def test_a_json_log_line_is_one_parseable_object_with_the_extras_as_keys():
    buffer = capture()
    logging.getLogger("resilientsc.test").info("step %s ok", "optimize", extra={"event": "step", "step": "optimize", "duration_ms": 12.5})
    (entry,) = lines(buffer)
    assert entry["level"] == "INFO" and entry["logger"] == "resilientsc.test" and entry["message"] == "step optimize ok"
    assert (entry["event"], entry["step"], entry["duration_ms"]) == ("step", "optimize", 12.5) and entry["ts"].endswith("+00:00")
    assert "request_id" not in entry  # no context, no ids


def test_an_exception_is_logged_with_its_traceback_in_one_line():
    buffer = capture()
    try:
        1 / 0
    except ZeroDivisionError:
        logging.getLogger("resilientsc.test").exception("boom")
    (entry,) = lines(buffer)
    assert entry["level"] == "ERROR" and "ZeroDivisionError" in entry["exc"] and "Traceback" in entry["exc"]


def test_bound_ids_are_stamped_on_every_line_made_while_they_are_set_and_only_then():
    buffer = capture()
    log = logging.getLogger("resilientsc.test")
    with bind(request_id="req-1", simulation_id="sim-1"):
        log.info("inside")
        with bind(run_id="run-1"):
            log.info("nested")
            assert current() == {"request_id": "req-1", "simulation_id": "sim-1", "run_id": "run-1"}
        log.info("after nested")
    log.info("outside")
    inside, nested, after_nested, outside = lines(buffer)
    assert (inside["request_id"], inside["simulation_id"], "run_id" in inside) == ("req-1", "sim-1", False)
    assert nested["run_id"] == "run-1" and "run_id" not in after_nested
    assert not {"request_id", "simulation_id", "run_id"} & set(outside)


def test_configure_logging_replaces_its_own_handler_instead_of_stacking_one():
    logger = logging.getLogger("resilientsc")
    for _ in range(3):
        configure_logging("INFO", "json")
    configure_logging("INFO", "text")
    assert len([h for h in logger.handlers if getattr(h, "_resilientsc_handler", False)]) == 1
    with pytest.raises(ValueError, match="'text' or 'json'"):
        configure_logging("INFO", "xml")


def test_text_format_is_human_readable_and_still_shows_the_structured_fields():
    buffer = io.StringIO()
    configure_logging("INFO", "text", stream=buffer)
    with bind(simulation_id="sim-9"):
        logging.getLogger("resilientsc.test").info("hello", extra={"event": "x", "n": 3})
    text = buffer.getvalue()
    assert "INFO" in text and "hello" in text and "event=x" in text and "n=3" in text and "simulation_id=sim-9" in text


def test_a_secret_in_the_environment_never_reaches_a_log_line(monkeypatch):
    monkeypatch.setenv("SOME_SERVICE_API_KEY", "sk-live-abcdef123456")
    monkeypatch.setenv("SHORT_TOKEN", "abc")  # too short to be worth redacting: it would mangle ordinary words
    buffer = capture()
    log = logging.getLogger("resilientsc.test")
    log.info("calling with key %s", "sk-live-abcdef123456")
    log.info("plain message about abc")
    try:
        raise RuntimeError("upstream said: bad key sk-live-abcdef123456")
    except RuntimeError:
        log.exception("failed")
    assert "sk-live-abcdef123456" not in buffer.getvalue() and "***" in buffer.getvalue()
    assert "plain message about abc" in buffer.getvalue()
    assert redact("x sk-live-abcdef123456 y") == "x *** y"


def test_a_worker_thread_carries_the_simulation_and_run_ids_of_the_job_it_runs():
    buffer = capture()
    log = logging.getLogger("resilientsc.test")
    executor = ThreadPoolExecutor(max_workers=2)
    registry = RunRegistry(executor, overdue_seconds=60)

    class Outcome:
        def model_dump(self, **kwargs):
            return {"outcome": "COMPLETED", "steps": []}

    def work():
        log.info("inside the job")
        return Outcome()

    with bind(request_id="req-77"):  # the POST that started it
        record = registry.start("sim-42", work)
    executor.shutdown(wait=True)
    entries = lines(buffer)
    events = [e.get("event") for e in entries]
    assert events == ["run_started", None, "run_finished"]
    assert {e["simulation_id"] for e in entries} == {"sim-42"} and {e["run_id"] for e in entries} == {record.run_id}
    assert {e["request_id"] for e in entries} == {"req-77"}  # the request that started the run is on the run's own log lines
    assert entries[-1]["state"] == "FINISHED" and entries[-1]["outcome"] == "COMPLETED" and entries[-1]["duration_ms"] >= 0


def test_settings_come_from_the_file_and_the_environment_wins(monkeypatch, tmp_path):
    assert load_settings()["model"]["drift_z_threshold"] == 3.0 and load_settings()["llm"]["circuit_failure_threshold"] == 3
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("LOG_FORMAT", "JSON")
    monkeypatch.setenv("RUN_OVERDUE_SECONDS", "7.5")
    s = load_settings()
    assert (s["logging"]["level"], s["logging"]["format"], s["runs"]["overdue_seconds"]) == ("DEBUG", "json", 7.5)
    assert load_settings(tmp_path / "missing.yaml")["metrics"]["reservoir_size"] == 512  # no file: the defaults


# =========================================================================== #
# metrics
# =========================================================================== #
def test_counters_gauges_and_summaries_do_their_arithmetic():
    m = Metrics()
    m.inc("things_total", {"kind": "a"}), m.inc("things_total", {"kind": "a"}, n=2), m.inc("things_total", {"kind": "b"})
    m.set_gauge("depth", 4), m.set_gauge("depth", 2)
    for v in range(1, 101):
        m.observe("latency_ms", v)
    snap = m.snapshot()
    assert {tuple(r["labels"].items()): r["value"] for r in snap["counters"]["things_total"]} == {(("kind", "a"),): 3, (("kind", "b"),): 1}
    assert snap["gauges"]["depth"][0]["value"] == 2
    s = snap["summaries"]["latency_ms"][0]
    assert (s["count"], s["sum"], s["min"], s["max"], s["mean"], s["p50"], s["p95"]) == (100, 5050, 1, 100, 50.5, 50, 95)
    assert m.counter_value("things_total", {"kind": "a"}) == 3 and m.counter_total("things_total") == 4 and m.counter_value("nope") == 0


def test_label_order_does_not_split_a_series():
    m = Metrics()
    m.inc("x_total", {"a": 1, "b": 2}), m.inc("x_total", {"b": 2, "a": 1})
    assert len(m.snapshot()["counters"]["x_total"]) == 1 and m.counter_value("x_total", {"b": 2, "a": 1}) == 2


def test_a_latency_series_keeps_memory_bounded_but_its_count_and_sum_exact():
    m = Metrics(reservoir_size=10)
    for v in range(1000):
        m.observe("l", v)
    s = m.snapshot()["summaries"]["l"][0]
    assert s["count"] == 1000 and s["sum"] == sum(range(1000)) and s["max"] == 999 and s["min"] == 0
    assert s["p50"] >= 990  # the quantiles describe the recent window, not all of history


def test_timed_records_the_duration_whether_the_block_returns_or_raises():
    m = Metrics()
    with m.timed("work_ms", {"step": "a"}):
        time.sleep(0.01)
    with pytest.raises(RuntimeError):
        with m.timed("work_ms", {"step": "b"}):
            raise RuntimeError("x")
    rows = {r["labels"]["step"]: r for r in m.snapshot()["summaries"]["work_ms"]}
    assert rows["a"]["count"] == 1 and rows["a"]["sum"] >= 10 and rows["b"]["count"] == 1


def test_many_threads_incrementing_lose_nothing():
    m = Metrics()

    def hammer():
        for _ in range(2000):
            m.inc("hits_total"), m.observe("l", 1.0)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    [t.start() for t in threads], [t.join() for t in threads]
    assert m.counter_value("hits_total") == 16000 and m.snapshot()["summaries"]["l"][0]["count"] == 16000


def test_prometheus_text_is_well_formed_with_help_type_labels_and_escaping():
    m = Metrics()
    m.describe("http_total", "Requests.")
    m.inc("http_total", {"route": '/a"b\\c', "status": 200}, 3)
    m.set_gauge("in_flight", 2)
    m.observe("dur_ms", 10), m.observe("dur_ms", 30)
    text = m.prometheus()
    assert "# HELP http_total Requests." in text and "# TYPE http_total counter" in text and "# TYPE in_flight gauge" in text and "# TYPE dur_ms summary" in text
    assert 'http_total{route="/a\\"b\\\\c",status="200"} 3' in text
    assert 'dur_ms{quantile="0.5"} 10' in text and "dur_ms_sum 40" in text and "dur_ms_count 2" in text and "process_uptime_seconds" in text
    assert text.endswith("\n")


# =========================================================================== #
# model monitoring
# =========================================================================== #
def stats(mean=100.0, std=10.0, windows=200):
    return lambda: {"P1": TrainingStats(mean, std, windows)}


def test_inference_stats_and_the_missing_feature_rate_are_what_was_recorded():
    mm = ModelMonitor(stats_loader=stats())
    mm.record_inference(version="v1", product_id="P1", latency_ms=2.0, features_missing=0)
    mm.record_inference(version="v1", product_id="P1", latency_ms=4.0, features_missing=4)  # a short series: 4 of 8 lags are NaN
    mm.record_inference(version="v2", product_id="P1", latency_ms=6.0, features_missing=0)
    s = mm.snapshot()
    assert s["inference_count"] == 3 and s["model_versions"] == {"v1": 2, "v2": 1}
    assert s["missing_feature_rate"] == round(4 / 24, 4) and s["features_checked"] == 24
    assert (s["latency_ms"]["mean"], s["latency_ms"]["p50"], s["latency_ms"]["max"]) == (4.0, 4.0, 6.0)
    assert metrics.counter_value("forecast_inferences_total", {"model_version": "v1"}) == 2


def test_a_recent_window_far_from_the_training_distribution_is_flagged_and_one_near_it_is_not():
    mm = ModelMonitor(stats_loader=stats(mean=100, std=10))
    ok = mm.check_input_drift("P1", [105.0] * 28)
    drift = mm.check_input_drift("P1", [140.0] * 28)
    assert (ok["status"], ok["z"]) == ("OK", 0.5) and (drift["status"], drift["z"]) == ("DRIFT", 4.0)
    low = mm.check_input_drift("P1", [50.0] * 28)
    assert (low["status"], low["z"]) == ("DRIFT", -5.0)  # drift is in either direction


def test_only_the_most_recent_window_counts_and_the_threshold_is_configurable():
    mm = ModelMonitor(stats_loader=stats(mean=100, std=10))
    assert mm.check_input_drift("P1", [1000.0] * 100 + [100.0] * 28)["status"] == "OK"  # the old spike is outside the 28-day window
    strict = ModelMonitor(settings={"model": {"drift_z_threshold": 1.0, "drift_window_days": 7, "min_training_windows": 30, "warnings_kept": 5}}, stats_loader=stats(100, 10))
    assert strict.check_input_drift("P1", [100.0] * 100 + [115.0] * 7)["status"] == "DRIFT"  # 1.5σ > 1σ, over a 7-day window


def test_a_product_with_no_usable_baseline_is_reported_as_such_not_as_ok():
    mm = ModelMonitor(stats_loader=lambda: {"P1": TrainingStats(100, 10, 5), "P2": TrainingStats(50, 0.0, 200)})
    assert mm.check_input_drift("P1", [100.0] * 28)["status"] == "NO_BASELINE"   # 5 training windows: too few to call a distribution
    assert mm.check_input_drift("P2", [50.0] * 28)["status"] == "NO_BASELINE"    # zero spread: a z-score would divide by nothing
    assert mm.check_input_drift("UNKNOWN", [1.0] * 28)["status"] == "NO_BASELINE"
    assert mm.check_input_drift("P1", []) is None


def test_a_drift_warning_is_raised_once_per_transition_not_once_per_forecast():
    mm = ModelMonitor(stats_loader=stats())
    buffer = capture("WARNING")
    for _ in range(5):
        mm.check_input_drift("P1", [200.0] * 28)   # still drifting: not news
    assert len(mm.snapshot()["drift"]["warnings"]) == 1 and metrics.counter_value("forecast_drift_warnings_total", {"product_id": "P1"}) == 1
    mm.check_input_drift("P1", [100.0] * 28)       # back to normal...
    mm.check_input_drift("P1", [200.0] * 28)       # ...and out again: news
    assert len(mm.snapshot()["drift"]["warnings"]) == 2
    warnings = [e for e in lines(buffer) if e.get("event") == "drift_warning"]
    assert len(warnings) == 2 and warnings[0]["product_id"] == "P1" and warnings[0]["z"] == 10.0 and warnings[0]["level"] == "WARNING"
    d = mm.snapshot()["drift"]
    assert (d["products_checked"], d["products_drifting"], d["by_product"]["P1"]["status"]) == (1, 1, "DRIFT")


def test_training_statistics_use_only_the_training_split_and_only_full_windows(tmp_path):
    rows = []
    for i in range(60):  # P1: train days 0..39, val days 40..59; rolling_mean_28 = day index (a ramp), so it is easy to reason about
        rows.append({"date": pd.Timestamp("2011-01-01") + pd.Timedelta(days=i), "product_id": "P1", "rolling_mean_28": float(i), "split": "train" if i < 40 else "val"})
    path = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    s = load_training_stats(path)["P1"]
    assert s.windows == 13 and s.mean == pytest.approx(sum(range(27, 40)) / 13)  # train days 27..39; the 27 partial windows and the whole val split are excluded


@requires_built_data
def test_a_real_forecast_is_monitored_and_the_q4_ramp_the_model_never_saw_is_flagged():
    from backend.agents.inventory import tools as inventory_tools

    series = inventory_tools.forecast_series("22197", None, 14)
    s = monitor.snapshot()
    assert len(series) == 14 and s["inference_count"] == 14 and s["model_versions"] == {"2026.09.1": 14}
    assert s["missing_feature_rate"] == 0.0 and s["latency_ms"]["p95"] > 0
    d = s["drift"]["by_product"]["22197"]  # trained on Dec-2010..Aug-2011, forecasting from late Nov-2011: demand has ramped far beyond anything it saw
    assert d["status"] == "DRIFT" and d["z"] > 3 and d["window_mean"] > 4 * d["train_mean"]


@requires_built_data
def test_a_backtest_scores_the_model_against_what_actually_happened():
    r = monitor.backtest("22197", 14)
    assert (r["horizon_days"], r["days_scored"], r["as_of"]) == (14, 14, "2011-11-25") and r["mae"] > 0
    assert 0 < r["wape"] < 5  # a ratio of total absolute error to total demand
    assert monitor.snapshot()["backtests"]["22197"]["wape"] == r["wape"]
    with pytest.raises(ValueError, match="not enough history"):
        monitor.backtest("22197", horizon_days=10_000)


# =========================================================================== #
# the LLM circuit breaker
# =========================================================================== #
class Flaky:
    model = "flaky"

    def __init__(self):
        self.calls, self.fail = 0, True

    def generate_json(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise LLMUnavailableError("Gemini unavailable after 3 attempts")
        return LLMResponse("{}", self.model, 1.0, 1)


def call(breaker):
    return breaker.generate_json(system_instruction="s", user_text="u", response_schema={})


def test_the_circuit_opens_after_repeated_failures_and_stops_calling_the_model():
    now = [0.0]
    inner = Flaky()
    breaker = CircuitBreakerLLM(inner, failure_threshold=3, cooldown_seconds=30, clock=lambda: now[0])
    for _ in range(3):
        with pytest.raises(LLMUnavailableError, match="after 3 attempts"):
            call(breaker)
    assert breaker.circuit_state() == "open" and inner.calls == 3
    with pytest.raises(LLMUnavailableError, match="circuit open"):
        call(breaker)
    assert inner.calls == 3  # refused at once, without a call
    assert metrics.counter_value("llm_circuit_open_total") == 1


def test_after_the_cooldown_one_trial_call_decides_whether_the_circuit_closes():
    now = [0.0]
    inner = Flaky()
    breaker = CircuitBreakerLLM(inner, failure_threshold=2, cooldown_seconds=30, clock=lambda: now[0])
    for _ in range(2):
        with pytest.raises(LLMUnavailableError):
            call(breaker)
    now[0] = 29.9
    assert breaker.circuit_state() == "open"
    now[0] = 30.0
    assert breaker.circuit_state() == "half_open"
    with pytest.raises(LLMUnavailableError, match="after 3 attempts"):  # the trial fails: open again, for a fresh cooldown
        call(breaker)
    assert inner.calls == 3 and breaker.circuit_state() == "open"
    now[0] = 60.0
    inner.fail = False
    assert call(breaker).model == "flaky" and breaker.circuit_state() == "closed"  # the trial succeeds
    inner.fail = True
    with pytest.raises(LLMUnavailableError):
        call(breaker)
    assert breaker.circuit_state() == "closed"  # one failure after recovery is not a streak


def test_a_success_in_between_resets_the_streak():
    inner = Flaky()
    breaker = CircuitBreakerLLM(inner, failure_threshold=3, cooldown_seconds=30, clock=lambda: 0.0)
    for outcome_fails in (True, True, False, True, True):
        inner.fail = outcome_fails
        try:
            call(breaker)
        except LLMUnavailableError:
            pass
    assert breaker.circuit_state() == "closed"  # never three in a row


def test_the_agent_reports_its_llm_and_circuit_state_without_calling_it(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert SensingAgent().llm_status() == {"configured": False, "circuit": "closed"}
    monkeypatch.setenv("LLM_API_KEY", "abcdefgh12345")
    assert SensingAgent().llm_status()["configured"] is True
    breaker = CircuitBreakerLLM(Flaky(), failure_threshold=1, cooldown_seconds=30)
    with pytest.raises(LLMUnavailableError):
        call(breaker)
    assert SensingAgent(llm=breaker).llm_status() == {"configured": True, "circuit": "open"}


# =========================================================================== #
# runs that take too long
# =========================================================================== #
def test_a_run_that_is_still_going_past_the_limit_is_reported_overdue_and_a_finished_one_is_not():
    from datetime import datetime, timedelta, timezone

    started = datetime.now(timezone.utc) - timedelta(seconds=5)
    running = RunRecord("r1", "sim", started, overdue_seconds=2.0)
    assert running.overdue and running.to_dict()["overdue"] is True and running.to_dict()["elapsed_ms"] >= 5000
    assert not RunRecord("r2", "sim", started, overdue_seconds=60.0).overdue
    done = RunRecord("r3", "sim", started, finished_at=started + timedelta(seconds=1), state="FINISHED", overdue_seconds=2.0)
    assert not done.overdue and done.to_dict()["elapsed_ms"] == pytest.approx(1000, abs=1)  # it took 1s of a 2s allowance, and stopped counting when it finished
    slow_but_finished = RunRecord("r4", "sim", started, finished_at=started + timedelta(seconds=10), state="FINISHED", overdue_seconds=2.0)
    assert not slow_but_finished.overdue  # it took 5x its allowance, but it is over: "overdue" means still running past the limit


# =========================================================================== #
# through the real API and pipeline
# =========================================================================== #
class NeverCalledLLM:
    model = "never"

    def generate_json(self, **kwargs):
        raise AssertionError("scenario triggers are structured: no LLM")


def make(engine=None, threshold=10_000_000):
    store = WorldStateStore(SqlAlchemyWorldStateRepository("sqlite:///:memory:"))
    orchestrator = Orchestrator(store, SensingAgent(llm=NeverCalledLLM()), compliance_rules={"rejected_suppliers": [], "restricted_countries": [], "approval_threshold": threshold}, engine=engine)
    ctx = AppContext(store, orchestrator, RunRegistry(None))
    return TestClient(create_app(context=ctx), raise_server_exceptions=False), store, ctx


def data(response, status=200):
    assert response.status_code == status, response.text
    return response.json()["data"]


def rows(snapshot, kind, name):
    return {tuple(sorted(r["labels"].items())): r for r in snapshot[kind].get(name, [])}


def test_every_response_carries_a_request_id_and_a_sane_caller_supplied_one_is_kept():
    client, _, _ = make()
    generated = client.get("/api/health").headers["x-request-id"]
    assert len(generated) == 16
    assert client.get("/api/health", headers={"X-Request-ID": "trace-abc.123"}).headers["x-request-id"] == "trace-abc.123"
    replaced = client.get("/api/health", headers={"X-Request-ID": "has spaces and <script>"}).headers["x-request-id"]
    assert replaced != "has spaces and <script>" and len(replaced) == 16  # never echo what could forge a log line


def test_an_error_names_its_request_and_the_same_id_is_on_the_log_line():
    client, _, _ = make()
    buffer = capture("INFO")
    response = client.get("/api/simulations/sim-nope", headers={"X-Request-ID": "trace-404"})
    err = response.json()["error"]
    assert response.status_code == 404 and err["request_id"] == "trace-404" and response.headers["x-request-id"] == "trace-404"
    line = next(e for e in lines(buffer) if e.get("event") == "request")
    assert (line["request_id"], line["status"], line["method"], line["route"], line["simulation_id"]) == ("trace-404", 404, "GET", "/api/simulations/{simulation_id}", "sim-nope")


@requires_built_data
def test_a_run_leaves_a_searchable_trail_of_log_lines_all_tied_to_its_simulation():
    client, _, _ = make()
    buffer = capture("INFO")
    sim = data(client.post("/api/scenarios/SUEZ_CLOSURE/run", json={}, headers={"X-Request-ID": "trace-run"}), 202)["simulation_id"]
    mine = [e for e in lines(buffer) if e.get("simulation_id") == sim]
    events = [(e.get("event"), e.get("step") or e.get("checkpoint") or e.get("outcome")) for e in mine]
    assert ("run_started", None) in events and ("run_finished", "COMPLETED") in events
    assert [e[1] for e in events if e[0] == "step"] == ["sense", "agents", "optimize", "compliance"]
    assert [e[1] for e in events if e[0] == "checkpoint"] == ["simulation_created", "event_sensed", "agents_assessed", "plan_optimized", "compliance_checked", "plan_finalized"]  # the log is the audit trail's twin
    assert ("run_outcome", "COMPLETED") in events
    run_lines = [e for e in mine if e.get("event") in ("run_started", "step", "run_outcome", "run_finished")]
    assert len({e["run_id"] for e in run_lines}) == 1 and {e["request_id"] for e in run_lines} == {"trace-run"}  # the worker's lines still name the request that began it
    assert all(e["duration_ms"] >= 0 for e in mine if e.get("event") == "step")
    request_line = next(e for e in lines(buffer) if e.get("event") == "request" and e["route"] == "/api/scenarios/{scenario_id}/run")
    assert request_line["status"] == 202 and request_line["request_id"] == "trace-run"


@requires_built_data
def test_a_run_moves_the_counters_and_timings_an_operator_would_watch():
    client, _, _ = make(threshold=100_000)  # the supplier-failure plan costs ~750k: over this, so it needs a human
    sim = data(client.post("/api/scenarios/SUPPLIER_FAILURE/run", json={}), 202)["simulation_id"]
    snap = data(client.get("/api/metrics"))
    counter = lambda name: {k: r["value"] for k, r in rows(snap, "counters", name).items()}  # noqa: E731
    assert counter("runs_total") == {(("outcome", "AWAITING_APPROVAL"),): 1}
    assert counter("optimizations_total") == {(("status", "OPTIMAL"),): 1} and counter("compliance_verdicts_total") == {(("status", "ESCALATED"),): 1}
    assert counter("sensing_outcomes_total") == {(("status", "EVENT"),): 1}
    assert counter("checkpoints_total")[(("checkpoint", "approval_requested"),)] == 1
    assert "llm_calls_total" not in snap["counters"]  # a structured trigger: the LLM was never called, and that is what the counter says
    steps = {dict(k)["step"]: r["count"] for k, r in rows(snap, "summaries", "run_step_duration_ms").items()}
    assert steps == {"sense": 1, "agents": 1, "optimize": 1, "compliance": 1}
    assert rows(snap, "summaries", "optimization_solve_ms") and rows(snap, "summaries", "run_duration_ms")

    version = data(client.get(f"/api/compliance/{sim}"))["version"]
    data(client.post(f"/api/decisions/{sim}/approve", json={"decided_by": "Dana", "expected_version": version}))
    assert data(client.get("/api/metrics"))["counters"]["approvals_total"][0] == {"labels": {"decision": "approved"}, "value": 1}


@requires_built_data
def test_http_metrics_use_the_route_template_so_a_thousand_simulations_are_one_series():
    client, _, _ = make()
    for i in range(3):
        client.get(f"/api/simulations/sim-{i}")  # three 404s
    client.get("/api/nope")
    snap = data(client.get("/api/metrics"))
    series = {(dict(k)["route"], dict(k)["status"]): r["value"] for k, r in rows(snap, "counters", "http_requests_total").items()}
    assert series[("/api/simulations/{simulation_id}", "404")] == 3 and series[("unmatched", "404")] == 1
    assert not any("sim-" in route for route, _ in series)
    assert rows(snap, "summaries", "http_request_duration_ms")


def test_metrics_can_be_scraped_as_prometheus_text():
    client, _, _ = make()
    client.get("/api/health")
    response = client.get("/api/metrics?format=prometheus")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/plain")
    assert '# TYPE http_requests_total counter' in response.text and 'http_requests_total{method="GET",route="/api/health",status="200"}' in response.text
    assert "# HELP http_requests_total" in response.text


@requires_built_data
def test_a_step_that_fails_is_counted_by_step_and_the_run_is_recorded_as_failed():
    class BrokenEngine:
        def optimize_supply_chain(self, problem):
            raise RuntimeError("the solver fell over")

    client, _, _ = make(engine=BrokenEngine())
    buffer = capture("INFO")
    sim = data(client.post("/api/scenarios/SUEZ_CLOSURE/run", json={}), 202)["simulation_id"]
    status = data(client.get(f"/api/simulations/{sim}/status"))
    assert status["status"] == "FAILED" and "solver fell over" in status["error"]
    snap = data(client.get("/api/metrics"))
    assert {dict(k)["step"]: r["value"] for k, r in rows(snap, "counters", "run_steps_failed_total").items()} == {"optimize": 1}
    assert {dict(k)["outcome"] for k in rows(snap, "counters", "runs_total")} == {"FAILED"}
    failed = [e for e in lines(buffer) if e.get("event") == "step" and e.get("ok") is False]
    assert len(failed) == 1 and failed[0]["step"] == "optimize" and failed[0]["simulation_id"] == sim and failed[0]["level"] == "WARNING"


@requires_built_data
def test_readiness_lists_what_it_checked_and_is_ready_when_the_core_is_there():
    client, _, _ = make()
    response = client.get("/api/ready")
    body = data(response)
    assert response.status_code == 200 and body["ready"] is True
    assert [c["name"] for c in body["checks"]] == ["database", "datasets", "forecast_model", "llm", "llm_circuit"]
    assert all(c["ok"] for c in body["checks"]) and body["degraded"] is False


@requires_built_data
def test_a_missing_dataset_makes_it_not_ready_and_names_the_file(monkeypatch):
    client, _, _ = make()
    monkeypatch.setattr(readiness, "REQUIRED_DATA", readiness.REQUIRED_DATA + ("data/processed/does_not_exist.csv",))
    response = client.get("/api/ready")
    body = response.json()["data"]
    assert response.status_code == 503 and body["ready"] is False
    datasets = next(c for c in body["checks"] if c["name"] == "datasets")
    assert datasets["ok"] is False and "does_not_exist.csv" in datasets["detail"] and "Phase 3-5" in datasets["detail"]


@requires_built_data
def test_a_missing_llm_key_is_degraded_not_down(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    store = WorldStateStore(SqlAlchemyWorldStateRepository("sqlite:///:memory:"))
    ctx = AppContext(store, Orchestrator(store, SensingAgent()), RunRegistry(None))  # the production wiring: the LLM is built from the environment
    client = TestClient(create_app(context=ctx), raise_server_exceptions=False)
    response = client.get("/api/ready")
    body = response.json()["data"]
    assert response.status_code == 200 and body["ready"] is True and body["degraded"] is True
    llm = next(c for c in body["checks"] if c["name"] == "llm")
    assert llm["ok"] is False and llm["required"] is False and "scenarios still run" in llm["detail"]


@requires_built_data
def test_an_unreachable_database_is_a_clean_503_that_leaks_no_sql_and_readiness_says_so():
    client, store, _ = make()

    def broken(*args, **kwargs):
        raise OperationalError("SELECT secret_column FROM world_state WHERE id = ?", {"id": 1}, Exception("unable to open database file"))

    store.get, store.list_simulations = broken, broken
    response = client.get("/api/simulations/sim-1", headers={"X-Request-ID": "trace-db"})
    err = response.json()["error"]
    assert response.status_code == 503 and err["error_code"] == "DATABASE_UNAVAILABLE" and err["request_id"] == "trace-db" and err["recovery"]
    assert "secret_column" not in response.text and "SELECT" not in response.text
    ready = client.get("/api/ready")
    assert ready.status_code == 503 and next(c for c in ready.json()["data"]["checks"] if c["name"] == "database")["ok"] is False
    assert client.get("/api/health").status_code == 200  # alive, though not ready: the two questions have different answers


@requires_built_data
def test_the_model_monitoring_endpoint_reports_the_model_and_can_backtest_it():
    client, _, _ = make()
    assert data(client.get("/api/monitoring/model"))["inference_count"] == 0  # nothing has forecast yet
    data(client.get("/api/inventory/forecast?product_id=22197&warehouse_id=Mumbai&horizon_days=7"))
    after = data(client.get("/api/monitoring/model?backtest_product_id=22197&horizon_days=14"))
    assert after["inference_count"] >= 7 and after["model_versions"] == {"2026.09.1": after["inference_count"]}
    assert after["backtest"]["product_id"] == "22197" and after["backtest"]["wape"] is not None and "22197" in after["backtests"]
    assert after["drift"]["by_product"]["22197"]["status"] in ("OK", "DRIFT", "NO_BASELINE") and after["drift"]["threshold_z"] == 3.0
    err = client.get("/api/monitoring/model?backtest_product_id=NOPE").json()["error"]
    assert err["error_code"] == "VALIDATION_ERROR" and "unknown product_id" in err["message"]


@requires_built_data
def test_run_status_reports_how_long_the_run_has_taken_and_whether_it_is_overdue():
    client, _, _ = make()
    sim = data(client.post("/api/scenarios/SUEZ_CLOSURE/run", json={}), 202)["simulation_id"]
    run = data(client.get(f"/api/simulations/{sim}/status"))["run"]
    assert run["state"] == "FINISHED" and run["overdue"] is False and run["elapsed_ms"] > 0
