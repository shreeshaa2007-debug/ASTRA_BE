"""Scenario simulation (docs/api-plan.md, Phase 17): the defined scenarios, what each does to the
network compared with normal operations and with doing nothing, and running one through the
full pipeline. `GET /api/simulations/{id}/comparison` (routers/simulations.py) is the same
comparison for a simulation's own disruption.

Comparisons are read-only, deterministic and cached, so they are GETs; running a scenario creates
a simulation, so that is a POST.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.agents.inventory import tools as inventory_tools
from backend.api.context import AppContext, get_ctx
from backend.api.errors import ApiError
from backend.services.world_state import load_baseline
from backend.simulation import compare_scenario, get_scenario, load_scenarios

router = APIRouter(prefix="/api", tags=["scenarios"])


def envelope(data):
    return {"data": data}


class RunScenarioRequest(BaseModel):
    product_id: Optional[str] = Field(default=None, min_length=1, max_length=32, description="default: the scenario's own default product")


def _checked_product(product_id: str) -> str:
    if product_id not in inventory_tools.get_product_ids():
        raise ApiError(422, "VALIDATION_ERROR", f"unknown product_id {product_id!r}; GET /api/products lists the products")
    return product_id


@router.get("/scenarios")
def list_scenarios():
    """The scenario definitions. A scenario with `modeled: false` says why it cannot be simulated."""
    baseline_tariffs = load_baseline().tariffs
    rows = []
    for s in load_scenarios().values():
        rows.append({
            "scenario_id": s.scenario_id, "label": s.label, "description": s.description, "modeled": s.modeled,
            "not_modeled_reason": s.not_modeled_reason, "default_product_id": s.default_product_id,
            "event": s.event.model_dump(mode="json") if s.event else None,
            "tariff_changes": [{"iso3": k, "baseline_pct": baseline_tariffs.get(k, 0.0), "scenario_pct": v} for k, v in s.tariff_overrides(baseline_tariffs).items()],
        })
    return envelope(rows)


@router.get("/scenarios/{scenario_id}/comparison")
def scenario_comparison(scenario_id: str, product_id: Optional[str] = None, as_of_date: Optional[str] = None):
    """Baseline (normal operations) vs. unmitigated (the baseline plan left in place) vs. mitigated
    (the optimizer's plan for the disrupted network), for one product. Costs are in MVP cost units."""
    scenario = get_scenario(scenario_id)  # 404 if unknown
    product = _checked_product(product_id or scenario.default_product_id)
    return envelope(compare_scenario(scenario, product, as_of_date).model_dump(mode="json"))


@router.post("/scenarios/{scenario_id}/run", status_code=202)
def run_scenario(scenario_id: str, body: RunScenarioRequest | None = None, ctx: AppContext = Depends(get_ctx)):
    """Creates a simulation for the scenario and runs the full pipeline on it — the Sensing Agent
    skips the LLM for a structured trigger but still validates it. Poll the returned status_url."""
    scenario = get_scenario(scenario_id)
    trigger = scenario.trigger()  # 422 SCENARIO_NOT_MODELED if the pipeline cannot represent it
    product = _checked_product((body.product_id if body else None) or scenario.default_product_id)
    overrides = scenario.tariff_overrides(load_baseline().tariffs) or None

    state = ctx.store.create(scenario.scenario_id)
    record = ctx.runs.start(state.simulation_id, lambda: ctx.orchestrator.run(state.simulation_id, trigger, product, tariff_overrides=overrides))
    return envelope({
        "simulation_id": state.simulation_id, "scenario_id": scenario.scenario_id, "product_id": product, "run_id": record.run_id,
        "run_state": record.state, "status_url": f"/api/simulations/{state.simulation_id}/status",
    })


@router.get("/products")
def products():
    """The products the inventory ledger tracks, and how many suppliers each has. A product with none
    cannot be planned: every plan for it is INFEASIBLE (the synthetic supplier roster covers three products)."""
    from backend.agents.sourcing import tools as sourcing_tools

    counts: dict[str, int] = {}
    for row in sourcing_tools.get_suppliers():
        counts[row["product_id"]] = counts.get(row["product_id"], 0) + 1
    rows = [{"product_id": p, "supplier_count": counts.get(p, 0), "has_suppliers": counts.get(p, 0) > 0} for p in inventory_tools.get_product_ids()]
    return envelope(sorted(rows, key=lambda r: (not r["has_suppliers"], r["product_id"])))
