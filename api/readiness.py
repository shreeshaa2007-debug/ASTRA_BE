"""Readiness: can this process do its job right now, and if not, which dependency is missing?

`/api/health` answers "is the process alive" and does nothing else. This answers the question an operator
actually has after a deploy. Before it, a missing processed dataset or model artifact was found by the first
user's request failing with DATASET_UNAVAILABLE. Each check is cheap, makes no LLM call and reads no secret.

A *required* check that fails means the app cannot serve its core function (503). An *optional* one that fails
means degraded, not down: with no LLM key, scenarios still run (their trigger is structured) and only free-text
reports cannot be sensed.
"""
from __future__ import annotations

from pathlib import Path

from backend.api.context import AppContext
from backend.data import get_datasets
from backend.models.forecasting.predictor import ARTIFACTS_ROOT

# Files a deployment insists on beyond the datasets themselves (which the dataset repository reports on: files for the
# CSV backend, `ref_*` tables for the SQL one). Empty by default; tests and the e2e app add to it.
REQUIRED_DATA: tuple[str, ...] = ()


def _check(name: str, ok: bool, detail: str, required: bool = True) -> dict:
    return {"name": name, "ok": ok, "required": required, "detail": detail}


def run_checks(ctx: AppContext) -> list[dict]:
    checks: list[dict] = []
    try:
        ctx.store.list_simulations(limit=1)
        checks.append(_check("database", True, "the world-state database answers"))
    except Exception as exc:  # noqa: BLE001 — whatever it is, the database is not usable
        checks.append(_check("database", False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"))

    repository = get_datasets()
    try:
        missing = repository.missing() + [p for p in REQUIRED_DATA if not Path(p).exists()]
    except Exception as exc:  # noqa: BLE001 — a dataset database that cannot be asked is not ready either
        checks.append(_check("datasets", False, f"cannot check {repository.describe()}: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}"))
    else:
        fix = "run the Phase 3-5 pipelines from the repo root" if repository.describe().startswith("csv") else "load them with scripts/load_reference_data.py"
        checks.append(_check("datasets", not missing, f"all datasets are present ({repository.describe()})" if not missing else f"missing: {', '.join(missing)} ({fix})"))

    versions = sorted(p.name for p in ARTIFACTS_ROOT.iterdir() if (p / "model.json").exists()) if ARTIFACTS_ROOT.exists() else []
    checks.append(_check("forecast_model", bool(versions), f"artifact version(s): {', '.join(versions)}" if versions else f"no model artifact under {ARTIFACTS_ROOT}"))

    events = ctx.events.status()
    if events["backend"] != "none":  # optional: a dead integration endpoint degrades the app, it does not stop it planning
        detail = f"{events['backend']} -> {events.get('target', 'the log')}"
        if "queued" in events:
            detail += f" (published {events['published']}, failed {events['failed']}, dropped {events['dropped']}, queued {events['queued']}"
            detail += f"; last error: {events['last_error']})" if events["last_error"] else ")"
        checks.append(_check("events", events["healthy"], detail, required=False))

    llm = ctx.orchestrator.sensing.llm_status()
    checks.append(_check("llm", llm["configured"], "an LLM key is configured" if llm["configured"] else "no LLM_API_KEY: free-text reports cannot be sensed (scenarios still run)", required=False))
    if llm["configured"]:
        checks.append(_check("llm_circuit", llm["circuit"] != "open", f"circuit breaker is {llm['circuit']}" + (" — calls are being refused after repeated failures" if llm["circuit"] == "open" else ""), required=False))
    return checks
