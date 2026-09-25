"""In-process metrics: counters, gauges and latency summaries, as JSON and as Prometheus text.

Deliberately small and dependency-free. Everything is per process — like the run registry, so the
deployment note is the same: one worker, or scrape each. Latency series keep a bounded window of recent
observations (`reservoir_size`), so the p50/p95 describe *recent* behaviour and memory stays fixed, while
`count` and `sum` are exact over the process's lifetime.

Names follow Prometheus conventions (`*_total` for counters, the unit in the name). `describe()` gives a
metric its help text; an undescribed metric still works.
"""
from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Iterator, Mapping

Labels = tuple[tuple[str, str], ...]


def _labels(labels: Mapping[str, object] | None) -> Labels:
    return tuple(sorted((k, str(v)) for k, v in (labels or {}).items()))


def _quantile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank quantile of an already sorted, non-empty list."""
    return sorted_values[min(len(sorted_values) - 1, max(0, math.ceil(q * len(sorted_values)) - 1))]


class _Summary:
    def __init__(self, window: int):
        self.count, self.total, self.minimum, self.maximum = 0, 0.0, math.inf, -math.inf
        self.recent: deque[float] = deque(maxlen=window)

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.minimum, self.maximum = min(self.minimum, value), max(self.maximum, value)
        self.recent.append(value)

    def as_dict(self) -> dict:
        recent = sorted(self.recent)
        return {
            "count": self.count, "sum": round(self.total, 3), "min": round(self.minimum, 3), "max": round(self.maximum, 3),
            "mean": round(self.total / self.count, 3), "p50": round(_quantile(recent, 0.5), 3), "p95": round(_quantile(recent, 0.95), 3),
        }


class Metrics:
    def __init__(self, reservoir_size: int = 512):
        self._lock = threading.Lock()
        self._window = reservoir_size
        self._help: dict[str, str] = {}
        self._counters: dict[str, dict[Labels, float]] = defaultdict(dict)
        self._gauges: dict[str, dict[Labels, float]] = defaultdict(dict)
        self._summaries: dict[str, dict[Labels, _Summary]] = defaultdict(dict)
        self._started = time.time()

    def describe(self, name: str, help_text: str) -> None:
        self._help[name] = help_text

    def inc(self, name: str, labels: Mapping[str, object] | None = None, n: float = 1) -> None:
        key = _labels(labels)
        with self._lock:
            self._counters[name][key] = self._counters[name].get(key, 0) + n

    def set_gauge(self, name: str, value: float, labels: Mapping[str, object] | None = None) -> None:
        with self._lock:
            self._gauges[name][_labels(labels)] = value

    def observe(self, name: str, value: float, labels: Mapping[str, object] | None = None) -> None:
        key = _labels(labels)
        with self._lock:
            series = self._summaries[name].get(key)
            if series is None:
                series = self._summaries[name][key] = _Summary(self._window)
            series.add(float(value))

    @contextmanager
    def timed(self, name: str, labels: Mapping[str, object] | None = None) -> Iterator[None]:
        """Observes the wall-clock milliseconds of the block, whether it returns or raises."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, (time.perf_counter() - started) * 1000, labels)

    def counter_value(self, name: str, labels: Mapping[str, object] | None = None) -> float:
        with self._lock:
            return self._counters.get(name, {}).get(_labels(labels), 0)

    def counter_total(self, name: str) -> float:
        with self._lock:
            return sum(self._counters.get(name, {}).values())

    def reset(self) -> None:
        with self._lock:
            self._counters.clear(), self._gauges.clear(), self._summaries.clear()
            self._started = time.time()

    # ------------------------------------------------------------------ output
    def snapshot(self) -> dict:
        with self._lock:
            def rows(table, value):
                return {name: [{"labels": dict(k), **value(v)} for k, v in sorted(series.items())] for name, series in sorted(table.items())}

            return {
                "uptime_seconds": round(time.time() - self._started, 1),
                "counters": rows(self._counters, lambda v: {"value": v}),
                "gauges": rows(self._gauges, lambda v: {"value": v}),
                "summaries": rows(self._summaries, lambda s: s.as_dict()),
            }

    def prometheus(self) -> str:
        """The Prometheus text exposition format (counters and gauges as-is; a summary as _count, _sum and quantiles)."""
        def fmt(labels: Labels, extra: tuple[tuple[str, str], ...] = ()) -> str:
            pairs = labels + extra
            return "{" + ",".join(f'{k}="{v.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"' for k, v in pairs) + "}" if pairs else ""

        lines: list[str] = []
        with self._lock:
            def header(name: str, kind: str) -> None:
                if name in self._help:
                    lines.append(f"# HELP {name} {self._help[name]}")
                lines.append(f"# TYPE {name} {kind}")

            for name, series in sorted(self._counters.items()):
                header(name, "counter")
                lines += [f"{name}{fmt(k)} {v:g}" for k, v in sorted(series.items())]
            for name, series in sorted(self._gauges.items()):
                header(name, "gauge")
                lines += [f"{name}{fmt(k)} {v:g}" for k, v in sorted(series.items())]
            for name, series in sorted(self._summaries.items()):
                header(name, "summary")
                for k, s in sorted(series.items()):
                    recent = sorted(s.recent)
                    lines += [f"{name}{fmt(k, (('quantile', str(q)),))} {_quantile(recent, q):g}" for q in (0.5, 0.95)]
                    lines += [f"{name}_sum{fmt(k)} {s.total:g}", f"{name}_count{fmt(k)} {s.count}"]
            lines += ["# TYPE process_uptime_seconds gauge", f"process_uptime_seconds {time.time() - self._started:g}"]
        return "\n".join(lines) + "\n"


metrics = Metrics()

for _name, _help in {
    "http_requests_total": "HTTP requests, by method, route template and status code.",
    "http_request_duration_ms": "HTTP request duration in milliseconds, by route template.",
    "runs_total": "Finished pipeline runs, by outcome.",
    "run_duration_ms": "Wall-clock milliseconds of a whole pipeline run.",
    "run_step_duration_ms": "Milliseconds of one pipeline step (sense, agents, optimize, compliance).",
    "run_steps_failed_total": "Pipeline steps that raised, by step.",
    "runs_in_flight": "Runs executing right now.",
    "replans_total": "Plans re-optimized after compliance rejected them.",
    "compliance_verdicts_total": "Compliance verdicts, by status.",
    "approvals_total": "Human decisions on escalated plans, by decision.",
    "optimizations_total": "Optimizer solves, by resulting plan status.",
    "optimization_solve_ms": "Milliseconds the solver itself took.",
    "sensing_outcomes_total": "Sensing results, by status.",
    "llm_calls_total": "Calls made to the language model, by result.",
    "llm_latency_ms": "Milliseconds a successful language-model call took.",
    "llm_retries_total": "Retries the language-model client made after a transient failure.",
    "llm_circuit_open_total": "Calls refused because the language-model circuit breaker was open.",
    "forecast_inferences_total": "Demand-model predictions, by model version.",
    "forecast_inference_ms": "Milliseconds one demand-model prediction took.",
    "forecast_drift_warnings_total": "Times a product's recent demand drifted beyond the threshold from its training distribution.",
    "checkpoints_total": "World-state checkpoints committed, by name.",
}.items():
    metrics.describe(_name, _help)
