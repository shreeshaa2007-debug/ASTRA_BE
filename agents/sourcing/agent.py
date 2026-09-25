"""SourcingAgent — composes tools.py into a proposed supplier_allocations
list, per brief §10's exact output shape. Greedy cheapest-landed-unit-cost-
first bin-packing across supplier capacity — a reasonable, explainable
starting mix, not a claim of optimality (the Optimization Engine, Phase 10,
makes the final joint call alongside inventory/logistics constraints; see
agent-plan.md's "again, this should be a recommendation, not uncontrolled
execution").
"""
from __future__ import annotations

from backend.agents.sourcing import tools


class SourcingAgent:
    def generate_sourcing_mix(
        self,
        product_id: str,
        required_quantity: int,
        excluded_supplier_ids: frozenset[str] = frozenset(),
        tariff_rates: dict[str, float] | None = None,
    ) -> dict:
        options = tools.generate_supplier_options(product_id, required_quantity, excluded_supplier_ids, tariff_rates)

        allocations = []
        remaining = required_quantity
        for option in options:
            if remaining <= 0:
                break
            if not option["capacity"]:
                continue
            take = min(remaining, option["capacity"])
            allocations.append({"supplier_id": option["supplier_id"], "quantity": take})
            remaining -= take

        return {
            "product_id": str(product_id),
            "required_quantity": required_quantity,
            "supplier_allocations": allocations,
            "unmet_quantity": remaining,
            "fully_covered": remaining == 0,
            "candidates_considered": options,
        }
