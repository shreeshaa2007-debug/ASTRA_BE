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
from backend.models.forecasting.predictor import ARTIFACTS_ROOT

REQUIRED_DATA = (
    "data/processed/suppliers.csv", "data/processed/routes.csv", "data/processed/tariffs.csv", "data/processed/disruptions.csv",
    "data/processed/inventory_multi_warehouse.csv", "data/processed/demand_modeling_panel.csv",
)


def _check(name: str, ok: bool, detail: str, required: bool = True) -> dict:
    return {"name": name, "ok": ok, "required": required, "detail": detail}


def run_checks(ctx: AppContext) -> list[dict]:
    checks: list[dict] = []
    try:
        ctx.store.list_simulations(limit=1)
        checks.append(_check("database", True, "the world-state database answers"))
    except Exception as exc:  # noqa: BLE001 — whatever it is, the database is not usable
        checks.append(_check("database", False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"))

    missing = [p for p in REQUIRED_DATA if not Path(p).exists()]
    checks.append(_check("datasets", not missing, "all processed datasets are present" if not missing else f"missing: {', '.join(missing)} (run the Phase 3-5 pipelines from the repo root)"))

    versions = sorted(p.name for p in ARTIFACTS_ROOT.iterdir() if (p / "model.json").exists()) if ARTIFACTS_ROOT.exists() else []
    checks.append(_check("forecast_model", bool(versions), f"artifact version(s): {', '.join(versions)}" if versions else f"no model artifact under {ARTIFACTS_ROOT}"))

    llm = ctx.orchestrator.sensing.llm_status()
    checks.append(_check("llm", llm["configured"], "an LLM key is configured" if llm["configured"] else "no LLM_API_KEY: free-text reports cannot be sensed (scenarios still run)", required=False))
    if llm["configured"]:
        checks.append(_check("llm_circuit", llm["circuit"] != "open", f"circuit breaker is {llm['circuit']}" + (" — calls are being refused after repeated failures" if llm["circuit"] == "open" else ""), required=False))
    return checks
