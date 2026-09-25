"""Phase 8 tests — Sourcing Agent tools + agent, brief §21 "AGENTS" category."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backend.agents.sourcing import tools
from backend.agents.sourcing.agent import SourcingAgent

requires_built_data = pytest.mark.skipif(
    not (Path("data/processed/suppliers.csv").exists() and Path("data/processed/tariffs.csv").exists()),
    reason="run Phase 3's pipeline first",
)


@requires_built_data
def test_region_to_iso3_covers_every_region_in_the_roster():
    suppliers = tools._load_suppliers()
    for region in suppliers["region"].unique():
        assert region in tools.REGION_TO_ISO3, f"{region!r} has no ISO3 mapping — calculate_landed_cost would silently skip its tariff"


@requires_built_data
def test_supplier_and_other_agents_share_real_product_ids():
    """Regression guard for the exact bug found at the start of Phase 8:
    suppliers.csv originally used placeholder 'P001'-style ids with zero
    overlap with the real UCI product_ids every other agent uses."""
    suppliers = tools._load_suppliers()
    demand_panel = pd.read_csv("data/processed/demand_modeling_panel.csv", usecols=["product_id"])
    real_ids = set(demand_panel["product_id"].astype(str))
    assert set(suppliers["product_id"]).issubset(real_ids)


@requires_built_data
def test_get_suppliers_filters_by_product_and_status():
    active_only = tools.get_suppliers(product_id="22197", status="ACTIVE")
    assert all(s["status"] == "ACTIVE" and s["product_id"] == "22197" for s in active_only)


@requires_built_data
def test_check_supplier_capacity_unknown_pair_raises():
    with pytest.raises(ValueError):
        tools.check_supplier_capacity("S999", "22197", 100)


@requires_built_data
def test_calculate_supplier_cost_matches_unit_cost_times_quantity():
    suppliers = tools.get_suppliers(product_id="22197")
    s = suppliers[0]
    cost = tools.calculate_supplier_cost(s["supplier_id"], "22197", 10)
    assert cost == pytest.approx(s["unit_cost"] * 10, abs=0.01)


@requires_built_data
def test_calculate_landed_cost_applies_the_real_india_tariff_rate():
    """India's real 2022 weighted-mean applied tariff (World Bank WDI,
    dataset_registry.yaml) is 4.59% — checked against the live tariffs.csv
    rather than hardcoded, so this fails loudly if the join breaks."""
    india_suppliers = [s for s in tools.get_suppliers(product_id="22197") if s["region"] == "India"]
    assert india_suppliers, "test assumes at least one India supplier carries product 22197"
    s = india_suppliers[0]

    tariffs = pd.read_csv("data/processed/tariffs.csv")
    india_rate = tariffs[tariffs["origin_country"] == "IND"].sort_values("effective_year").iloc[-1]["tariff_rate"]

    landed = tools.calculate_landed_cost(s["supplier_id"], "22197", 1)
    assert landed["tariff_rate_pct"] == pytest.approx(india_rate)
    assert landed["tariff_data_available"] is True
    assert landed["landed_unit_cost"] == pytest.approx(s["unit_cost"] * (1 + india_rate / 100), rel=1e-4)


@requires_built_data
def test_calculate_landed_cost_unknown_region_reports_no_tariff_data_available(monkeypatch):
    """A region with no ISO3 mapping falls back to 0% with an explicit flag,
    not a silent (and misleadingly cheap) 0% that looks like real data."""
    fake_suppliers = pd.DataFrame(
        [{"supplier_id": "SX", "supplier_name": "Test", "region": "Atlantis", "product_id": "TEST",
          "capacity": 100, "unit_cost": 50.0, "lead_time_days": 5, "reliability": 0.9, "risk_level": "LOW", "status": "ACTIVE"}]
    )
    monkeypatch.setattr(tools, "_load_suppliers", lambda: fake_suppliers)

    landed = tools.calculate_landed_cost("SX", "TEST", 10)
    assert landed["tariff_data_available"] is False
    assert landed["tariff_rate_pct"] == 0.0


@requires_built_data
def test_generate_supplier_options_excludes_disrupted_and_sorts_by_landed_unit_cost():
    options = tools.generate_supplier_options("23166", 1000)
    assert all(o["status"] != "DISRUPTED" for o in options)
    costs = [o["landed_unit_cost"] for o in options]
    assert costs == sorted(costs)


# --------------------------------------------------------------------------- #
# agent.py
# --------------------------------------------------------------------------- #
@requires_built_data
def test_sourcing_mix_matches_brief_schema_and_fully_covers_when_capacity_allows():
    agent = SourcingAgent()
    result = agent.generate_sourcing_mix("22197", required_quantity=10000)

    assert set(result["supplier_allocations"][0].keys()) == {"supplier_id", "quantity"}
    assert result["fully_covered"] is True
    assert result["unmet_quantity"] == 0
    assert sum(a["quantity"] for a in result["supplier_allocations"]) == 10000


@requires_built_data
def test_sourcing_mix_allocates_cheapest_suppliers_first():
    agent = SourcingAgent()
    result = agent.generate_sourcing_mix("22197", required_quantity=10000)
    allocated_ids = [a["supplier_id"] for a in result["supplier_allocations"]]
    ranked_ids = [c["supplier_id"] for c in result["candidates_considered"]]
    assert allocated_ids == ranked_ids[: len(allocated_ids)]


@requires_built_data
def test_sourcing_mix_reports_shortfall_honestly_instead_of_overcommitting():
    agent = SourcingAgent()
    result = agent.generate_sourcing_mix("22197", required_quantity=999_999)
    assert result["fully_covered"] is False
    assert result["unmet_quantity"] > 0
    total_allocated = sum(a["quantity"] for a in result["supplier_allocations"])
    assert total_allocated + result["unmet_quantity"] == 999_999
    # never allocates more than a supplier's own capacity
    for alloc in result["supplier_allocations"]:
        cand = next(c for c in result["candidates_considered"] if c["supplier_id"] == alloc["supplier_id"])
        assert alloc["quantity"] <= cand["capacity"]


@requires_built_data
def test_sourcing_mix_excludes_disrupted_supplier_end_to_end():
    agent = SourcingAgent()
    result = agent.generate_sourcing_mix("23166", required_quantity=500)
    assert all(a["supplier_id"] != "S001" for a in result["supplier_allocations"])


# --------------------------------------------------------------------------- #
# Phase 14: tariffs in effect come from the world state, not only the data files
# --------------------------------------------------------------------------- #
@requires_built_data
def test_tariff_rates_override_the_data_files_for_that_call_only():
    real = tools.calculate_landed_cost("S003", "22197", 100)  # India, real 2022 rate 4.59%
    assert real["tariff_rate_pct"] == pytest.approx(4.59)

    raised = tools.calculate_landed_cost("S003", "22197", 100, tariff_rates={"IND": 25.0})
    assert raised["tariff_rate_pct"] == 25.0 and raised["tariff_data_available"] is True
    assert raised["landed_unit_cost"] == pytest.approx(tools._load_suppliers().query("supplier_id == 'S003' and product_id == '22197'").iloc[0]["unit_cost"] * 1.25, abs=1e-3)

    assert tools.calculate_landed_cost("S003", "22197", 100)["landed_unit_cost"] == real["landed_unit_cost"]  # nothing was mutated


@requires_built_data
def test_a_country_missing_from_the_override_reports_no_tariff_data_rather_than_falling_back_to_the_files():
    landed = tools.calculate_landed_cost("S003", "22197", 100, tariff_rates={"CHN": 2.0})  # India not given
    assert landed["tariff_data_available"] is False and landed["tariff_rate_pct"] == 0.0


@requires_built_data
def test_supplier_options_and_the_agents_mix_are_priced_with_the_given_tariffs():
    baseline = {o["supplier_id"]: o["landed_unit_cost"] for o in tools.generate_supplier_options("22197", 1000)}
    raised = {o["supplier_id"]: o["landed_unit_cost"] for o in tools.generate_supplier_options("22197", 1000, tariff_rates={**tools.get_latest_tariffs(), "TUR": 80.0})}
    assert raised["S007"] > baseline["S007"] and all(raised[s] == baseline[s] for s in baseline if s != "S007")  # only Turkey moved

    mix = SourcingAgent().generate_sourcing_mix("22197", 1000, tariff_rates={**tools.get_latest_tariffs(), "TUR": 80.0})
    assert next(c for c in mix["candidates_considered"] if c["supplier_id"] == "S007")["tariff_rate_pct"] == 80.0
    assert mix["supplier_allocations"][0]["supplier_id"] != "S007"  # at +80% the cheapest-landed order changes


@requires_built_data
def test_the_baseline_tariffs_price_suppliers_exactly_as_the_data_files_do():
    """The world state starts from get_latest_tariffs(); passing it back must change nothing."""
    with_none = tools.generate_supplier_options("22197", 1000)
    with_baseline = tools.generate_supplier_options("22197", 1000, tariff_rates=tools.get_latest_tariffs())
    assert with_none == with_baseline
