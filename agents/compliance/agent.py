"""ComplianceAgent — thin wrapper around tools.validate_plan(), plus a helper
that assembles a `plan` dict from the Sourcing and Logistics agents' own
output shapes, since that's how the orchestrator (Phase 14+) will actually
call this: check the combined result of those two agents before anything is
marked final.
"""
from __future__ import annotations

from backend.agents.compliance import tools
from backend.agents.sourcing import tools as sourcing_tools


class ComplianceAgent:
    def validate_plan(self, plan: dict) -> dict:
        return tools.validate_plan(plan)

    def validate_sourcing_and_logistics(self, sourcing_result: dict, logistics_result: dict | None = None) -> dict:
        """`sourcing_result`: SourcingAgent.generate_sourcing_mix()'s output.
        `logistics_result`: LogisticsAgent.plan_shipment()'s output, or None
        if this plan has no shipment leg (e.g. a pure re-sourcing decision).

        total_cost sums each allocation's own landed cost at ITS allocated
        quantity (via the Sourcing Agent's own calculate_landed_cost tool —
        not generate_supplier_options()'s "landed cost at the full
        required_quantity" figure, which is only meaningful for ranking
        candidates, not for costing an actual partial allocation) plus the
        logistics leg's total_cost, if any.
        """
        total_cost = sum(
            sourcing_tools.calculate_landed_cost(alloc["supplier_id"], sourcing_result["product_id"], alloc["quantity"])["landed_cost"]
            for alloc in sourcing_result["supplier_allocations"]
        )

        plan = {
            "supplier_allocations": [
                {**alloc, "product_id": sourcing_result["product_id"]} for alloc in sourcing_result["supplier_allocations"]
            ],
            "route_id": logistics_result["recommended_route_id"] if logistics_result else None,
            "disrupted_route_ids": frozenset(logistics_result["disrupted_route_ids"]) if logistics_result else frozenset(),
        }

        if logistics_result and logistics_result["recommended_route_id"]:
            recommended = next(
                a for a in logistics_result["alternative_routes"] if a["route_id"] == logistics_result["recommended_route_id"]
            )
            total_cost += recommended["total_cost"]

        plan["total_cost"] = round(total_cost, 2)
        return tools.validate_plan(plan)
