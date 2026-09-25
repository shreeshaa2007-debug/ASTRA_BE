"""Prototype Optimization Engine — architecture.md §4.

Minimizes total cost of covering every warehouse's forecast demand plus safety
stock over the planning horizon, jointly across the three agents' options:

  variables (whole units, >= 0)
    x[supplier, route]   units bought from a supplier and shipped on a route
    t[a, b]              units moved from warehouse a to warehouse b
    inbound[w]           units of procured stock delivered to warehouse w

  minimize   sum landed_cost*x + freight*x + delay_penalty*x + transfer_cost*t
  subject to
    supplier capacity     sum_r x[s, r]              <= capacity[s]
    route capacity        sum_s x[s, r]              <= capacity[r]
    inbound balance       sum x                       = sum_w inbound[w]
    safety stock (each w) stock + inbound + t_in - t_out - demand >= safety
    transfer availability t_out[w]                   <= stock[w]
    delivery deadline     lanes arriving after max_delivery_days are not offered

Quantities are integers, so this is solved as a MILP with HiGHS (scipy's
`linprog(integrality=...)` — scipy is the deliberate stand-in for the
PuLP/OR-Tools family the brief rules out, see architecture.md §3). There are no
binary decisions: no fixed charges or minimum order sizes exist in the data.

An infeasible problem is reported as INFEASIBLE with a diagnosis, never as a
plan. Every OPTIMAL plan is re-checked by validation.py, an independent code
path, before it is returned.
"""
from __future__ import annotations

import math
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
from scipy.optimize import linprog

from backend.optimization import costing, explanation, validation
from backend.schemas.optimization import (
    Allocation,
    ConstraintStatus,
    OptimizationProblem,
    OptimizationSolution,
    Transfer,
    ValidationResult,
)

BIG_M = 1e6  # shortfall penalty in the infeasibility-diagnosis solve: dearer than any real lane
TIME_LIMIT_SECONDS = 30.0
INTEGRALITY_TOL = 1e-5


@runtime_checkable
class OptimizationEngine(Protocol):
    """What the rest of the system may rely on; a SAP IBP engine would implement
    the same four methods (architecture.md §2/§4)."""

    def optimize_supply_chain(self, problem: OptimizationProblem) -> OptimizationSolution: ...
    def validate_solution(self, solution: OptimizationSolution) -> ValidationResult: ...
    def get_objective_value(self, solution: OptimizationSolution) -> float: ...
    def get_constraint_status(self, solution: OptimizationSolution) -> list[ConstraintStatus]: ...


@dataclass
class _Model:
    c: np.ndarray
    A_ub: np.ndarray
    b_ub: np.ndarray
    A_eq: np.ndarray
    b_eq: np.ndarray
    bounds: list[tuple[float, float | None]]
    transfer_cols: dict[tuple[str, str], int]
    inbound_cols: dict[str, int]
    slack_cols: dict[str, int]

    @property
    def n_vars(self) -> int:
        return len(self.c)


def _assemble(problem: OptimizationProblem, lanes: list[costing.Lane], relax: bool = False) -> _Model:
    """Builds the constraint matrices. With `relax`, each warehouse gets a
    penalized shortfall variable so the problem is always feasible — used only
    to diagnose *why* the real problem isn't."""
    params = problem.parameters
    wids = [w.warehouse_id for w in problem.warehouses]
    n_lanes = len(lanes)

    col = n_lanes
    transfer_cols: dict[tuple[str, str], int] = {}
    for a in wids:
        for b in wids:
            if a != b:
                transfer_cols[(a, b)] = col
                col += 1
    inbound_cols = {}
    for w in wids:
        inbound_cols[w] = col
        col += 1
    slack_cols = {}
    if relax:
        for w in wids:
            slack_cols[w] = col
            col += 1
    n = col

    c = np.zeros(n)
    bounds: list[tuple[float, float | None]] = [(0, None)] * n
    for i, lane in enumerate(lanes):
        c[i] = costing.lane_total_unit_cost(params, lane.supplier, lane.route)
        cap = lane.supplier.capacity if lane.route is None else min(lane.supplier.capacity, lane.route.capacity)
        bounds[i] = (0, cap)
    for j in transfer_cols.values():
        c[j] = costing.transfer_unit_cost(params)
    for j in slack_cols.values():
        c[j] = BIG_M

    ub_rows: list[np.ndarray] = []
    ub_rhs: list[float] = []

    def add_ub(coefs: dict[int, float], rhs: float) -> None:
        row = np.zeros(n)
        for j, v in coefs.items():
            row[j] += v
        ub_rows.append(row)
        ub_rhs.append(rhs)

    by_supplier: dict[str, dict[int, float]] = {}
    by_route: dict[str, dict[int, float]] = {}
    for i, lane in enumerate(lanes):
        by_supplier.setdefault(lane.supplier.supplier_id, {})[i] = 1.0
        if lane.route is not None:
            by_route.setdefault(lane.route.route_id, {})[i] = 1.0
    suppliers = {s.supplier_id: s for s in problem.suppliers}
    routes = {r.route_id: r for r in problem.routes}
    for sid, coefs in by_supplier.items():
        add_ub(coefs, suppliers[sid].capacity)
    for rid, coefs in by_route.items():
        add_ub(coefs, routes[rid].capacity)

    for w in problem.warehouses:
        wid = w.warehouse_id
        coefs = {inbound_cols[wid]: -1.0}
        if relax:
            coefs[slack_cols[wid]] = -1.0
        for other in wids:
            if other != wid:
                coefs[transfer_cols[(other, wid)]] = -1.0  # arriving stock
                coefs[transfer_cols[(wid, other)]] = 1.0   # departing stock
        add_ub(coefs, w.current_stock - w.demand_units - w.safety_stock)
        if len(wids) > 1:
            add_ub({transfer_cols[(wid, other)]: 1.0 for other in wids if other != wid}, w.current_stock)

    eq = np.zeros((1, n))
    eq[0, :n_lanes] = 1.0
    for j in inbound_cols.values():
        eq[0, j] = -1.0

    return _Model(c, np.array(ub_rows), np.array(ub_rhs), eq, np.zeros(1), bounds, transfer_cols, inbound_cols, slack_cols)


class _SolverThread:
    """Every HiGHS call in the process runs on this one permanent thread.

    Found the hard way in Phase 15: scipy's bundled HiGHS misbehaves (Windows,
    scipy 1.13) when solves are run on threads that then exit. Solving from
    short-lived threads produced access-violation reports, and a plain LP
    froze the process outright — the main thread stuck inside Thread.start(),
    waiting for a thread that never got going. The API runs handlers and
    background runs on worker threads, so an optimizer that can freeze the
    server is not acceptable. Confining HiGHS to one thread that never exits
    removes the trigger (verified: 160 solves from ~140 short-lived threads,
    five times over, no faults, no hang).

    The cost is that solves are serialized. They take milliseconds, and each is
    bounded by TIME_LIMIT_SECONDS; a solver that needs to scale would take a
    process pool, not more threads.

    A daemon thread, deliberately: it must not be joined at interpreter exit,
    where its teardown is the very thing being avoided.
    """

    def __init__(self) -> None:
        self._jobs: "queue.Queue[tuple[Future, Callable[[], Any]]]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()

    def call(self, work: Callable[[], Any]) -> Any:
        if threading.current_thread() is self._thread:  # already there (a solve that triggers another): don't wait on ourselves
            return work()
        self._ensure_started()
        future: Future = Future()
        self._jobs.put((future, work))
        return future.result()  # re-raises whatever the solve raised

    def _ensure_started(self) -> None:
        if self._thread is None:
            with self._start_lock:
                if self._thread is None:
                    thread = threading.Thread(target=self._loop, name="highs-solver", daemon=True)
                    thread.start()
                    self._thread = thread

    def _loop(self) -> None:
        while True:
            future, work = self._jobs.get()
            try:
                future.set_result(work())
            except BaseException as exc:  # noqa: BLE001 — handed back to the caller, who decides
                future.set_exception(exc)


_SOLVER = _SolverThread()


def _solve(model: _Model, integer: bool):
    return _SOLVER.call(lambda: linprog(
        model.c, A_ub=model.A_ub, b_ub=model.b_ub, A_eq=model.A_eq, b_eq=model.b_eq, bounds=model.bounds,
        integrality=np.ones(model.n_vars) if integer else None, method="highs", options={"time_limit": TIME_LIMIT_SECONDS},
    ))


def _finite(value) -> float | None:
    return None if value is None or not math.isfinite(float(value)) else float(value)


class PrototypeOptimizationEngine:
    """scipy/HiGHS implementation of OptimizationEngine."""

    def optimize_supply_chain(self, problem: OptimizationProblem) -> OptimizationSolution:
        started = time.perf_counter()
        lane_set = costing.enumerate_lanes(problem)
        excluded = list(problem.excluded_options) + lane_set.excluded

        model = _assemble(problem, lane_set.lanes)
        result = _solve(model, integer=True)
        solver = {
            "name": "HiGHS", "interface": "scipy.optimize.linprog", "integer_quantities": True,
            "variables": model.n_vars, "constraints": len(model.b_ub) + len(model.b_eq),
            "candidate_lanes": len(lane_set.lanes), "scipy_status": int(result.status),
        }

        def finish(solution: OptimizationSolution) -> OptimizationSolution:
            solution.solver = {**solver, "solve_time_ms": round((time.perf_counter() - started) * 1000, 1)}
            return solution

        if result.status == 2:
            return finish(self._infeasible(problem, lane_set, excluded))
        if result.status != 0:
            return finish(OptimizationSolution(
                status="ERROR", problem=problem, excluded_options=excluded,
                message=f"solver did not reach an optimal solution (HiGHS status {result.status}): {result.message}",
            ))

        x = np.rint(result.x)
        if np.abs(result.x - x).max() > INTEGRALITY_TOL:
            return finish(OptimizationSolution(status="ERROR", problem=problem, excluded_options=excluded,
                                               message="solver returned fractional unit quantities"))
        solution = self._plan_from_solution(problem, lane_set, model, x, excluded)
        solver["mip_gap"] = _finite(getattr(result, "mip_gap", None))

        # the solver's objective and an independent recomputation must agree, and the plan
        # must pass the independent constraint check; if not, refuse to return it as a plan
        problems = validation.validate(solution).violations
        if abs(solution.objective_value - float(result.fun)) > 1e-4 * max(1.0, abs(float(result.fun))):
            problems.append(f"solver objective {float(result.fun):.6f} != recomputed plan cost {solution.objective_value:.6f}")
        if problems:
            return finish(OptimizationSolution(
                status="ERROR", problem=problem, excluded_options=excluded,
                message="solver returned a plan that failed independent validation: " + "; ".join(problems),
            ))
        return finish(solution)

    def _plan_from_solution(self, problem, lane_set, model, x, excluded) -> OptimizationSolution:
        params = problem.parameters
        allocations = [
            Allocation(
                supplier_id=lane.supplier.supplier_id,
                route_id=lane.route.route_id if lane.route else None,
                transport_mode=lane.route.transport_mode if lane.route else None,
                quantity=int(x[i]),
                landed_unit_cost=lane.supplier.landed_unit_cost,
                freight_unit_cost=lane.route.cost_per_unit if lane.route else 0.0,
                arrival_days=costing.arrival_days(lane.supplier, lane.route),
            )
            for i, lane in enumerate(lane_set.lanes) if int(x[i]) > 0
        ]
        transfers = [
            Transfer(from_warehouse=a, to_warehouse=b, quantity=int(x[j]), unit_cost=costing.transfer_unit_cost(params))
            for (a, b), j in model.transfer_cols.items() if int(x[j]) > 0
        ]
        inbound = {w: int(x[j]) for w, j in model.inbound_cols.items()}
        terms = costing.compute_objective_terms(problem, allocations, transfers)
        constraints = validation.evaluate_constraints(problem, allocations, transfers, inbound)

        return OptimizationSolution(
            status="OPTIMAL",
            problem=problem,
            message="optimal plan found" if allocations or transfers else "no action needed: current stock already covers forecast demand plus safety stock",
            objective_value=round(sum(terms.values()), 6),
            objective_terms=terms,
            allocations=allocations,
            transfers=transfers,
            inbound_by_warehouse=inbound,
            end_stock_by_warehouse=validation.end_stock_by_warehouse(problem, transfers, inbound),
            constraint_status=constraints,
            excluded_options=excluded,
            deviations=explanation.build_deviations(problem, allocations, transfers, excluded, lane_set),
            decision_factors=explanation.build_decision_factors(problem, allocations, transfers, constraints, excluded, lane_set),
        )

    def _infeasible(self, problem: OptimizationProblem, lane_set: costing.LaneSet, excluded) -> OptimizationSolution:
        """Re-solves with penalized shortfall slack so the message can say how
        far short the network falls and which limits are exhausted — the
        recovery hint api-plan.md's OPTIMIZATION_INFEASIBLE error carries."""
        model = _assemble(problem, lane_set.lanes, relax=True)
        relaxed = _solve(model, integer=False)
        diagnostics: dict = {"deadline_blocked_lanes": lane_set.deadline_blocked}
        message = "No feasible plan under current supplier, route and deadline limits."

        if relaxed.status == 0:
            shortfall = {w: round(float(relaxed.x[j]), 3) for w, j in model.slack_cols.items() if relaxed.x[j] > 1e-6}
            used_supplier: dict[str, float] = {}
            used_route: dict[str, float] = {}
            for i, lane in enumerate(lane_set.lanes):
                used_supplier[lane.supplier.supplier_id] = used_supplier.get(lane.supplier.supplier_id, 0.0) + relaxed.x[i]
                if lane.route is not None:
                    used_route[lane.route.route_id] = used_route.get(lane.route.route_id, 0.0) + relaxed.x[i]
            saturated = [f"supplier {s.supplier_id}" for s in problem.suppliers if used_supplier.get(s.supplier_id, 0.0) >= s.capacity - 1e-6]
            saturated += [f"route {r.route_id}" for r in problem.routes if used_route.get(r.route_id, 0.0) >= r.capacity - 1e-6]

            total = math.ceil(sum(shortfall.values()) - 1e-6)
            diagnostics.update(total_shortfall_units=total, shortfall_by_warehouse=shortfall, saturated_resources=saturated)
            message = f"No feasible plan: {total} units short of forecast demand plus safety stock even with every available supplier and route at capacity."

        hints = []
        if lane_set.deadline_blocked:
            hints.append("relax the delivery deadline")
        if diagnostics.get("saturated_resources"):
            hints.append("add supplier or route capacity")
        if problem.disrupted_route_ids or any(e.kind == "supplier" for e in excluded):
            hints.append("restore a disrupted route or add a supplier alternative")
        diagnostics["recovery"] = " or ".join(hints).capitalize() + "." if hints else "Reduce demand or add supply alternatives."

        return OptimizationSolution(status="INFEASIBLE", problem=problem, excluded_options=excluded, message=message, diagnostics=diagnostics)

    def validate_solution(self, solution: OptimizationSolution) -> ValidationResult:
        return validation.validate(solution)

    def get_objective_value(self, solution: OptimizationSolution) -> float:
        if solution.status != "OPTIMAL" or solution.objective_value is None:
            raise ValueError(f"no objective value: solution status is {solution.status}")
        return solution.objective_value

    def get_constraint_status(self, solution: OptimizationSolution) -> list[ConstraintStatus]:
        return validation.evaluate_constraints(solution.problem, solution.allocations, solution.transfers, solution.inbound_by_warehouse)
