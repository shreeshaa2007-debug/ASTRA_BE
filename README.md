# ASTRA_BE — ResilientSC backend

The backend for **ResilientSC**, an AI-assisted supply-chain disruption response system built
for an SAP Hackfest. It senses a disruption, forecasts its inventory impact, evaluates
logistics and sourcing alternatives, optimizes a response, checks it for compliance, and asks
a human to approve anything high-impact before it's final. It never lets an agent or an LLM
autonomously execute a change.

This repo is the **backend-only** counterpart to [ASTRA_FE](https://github.com/shreeshaa2007-debug/ASTRA_FE)
(the frontend) — both are split out of the same ResilientSC project for parallel development.

## Pipeline

```
Disruption report (free text or a structured signal)
  → Sensing Agent      LLM parses it into a structured, validated event
  → Shared World State updated
  → Inventory Agent    demand forecast → stockout risk → transfer recommendation
  → Logistics Agent     route status/capacity/cost → alternative routes
  → Sourcing Agent       supplier status/capacity/cost/tariffs → sourcing mix
  → Optimization Engine   MILP (scipy/HiGHS) → one feasible min-cost plan
  → Compliance Agent        deterministic rules → APPROVE / ESCALATE / REJECT
  → Human approval             required if escalated; REJECTED → one bounded replan
  → Final plan → Shared World State updated
```

## Layout

```
agents/         sensing/ inventory/ logistics/ sourcing/ compliance/ — one package per agent,
                each exposing plain tool functions plus an agent.py that composes them
api/            FastAPI app (api/main.py: create_app()) — routers/ (simulations, decisions,
                network, scenarios, ops, integration, ports), request/response models, auth
config/         env-driven settings + policy files (compliance_rules.yaml, scenarios.yaml,
                sensing_config.yaml, optimization_config.yaml, monitoring.yaml) — no hardcoded
                thresholds or secrets
data/           reference-data access (CSV files or ref_* SQL tables) + data/processed/*.csv,
                the processed datasets themselves
database/       world-state persistence (SQLite / SAP HANA Cloud via DATABASE_URL), engine
                selection, DDL export (database/hana_core.py — see "SAP readiness" below)
integration/    CloudEvents out (SAP Integration Suite), disruption signals in
ml/             training/evaluation/artifacts for the demand-forecasting model (see models/)
models/         forecasting/predictor.py — serves the trained model artifact
monitoring/     structured logging, metrics, model drift/backtest monitoring
optimization/   the OptimizationEngine Protocol + PrototypeOptimizationEngine (scipy/HiGHS MILP)
orchestration/  the pipeline's state machine — sense → agents → optimize → compliance → approval
sap/            everything that knows SAP formats: BTP/VCAP_SERVICES, HANA, XSUAA/OAuth, and
                the human-approval bridge (see below)
schemas/        pydantic models — the internal data contracts
services/       cross-cutting: world-state store/checkpoints, preprocessing pipelines,
                real-time port telemetry (see below)
simulation/     scenario definitions (backend/config/scenarios.yaml) + baseline/unmitigated/
                mitigated comparison
tests/          unit, integration, and end-to-end (tests/e2e/) — real processes over real HTTP
tools/          controlled tool functions the agents call — the only things an LLM can invoke
```

## Beyond the base pipeline

Two capabilities live only in this backend (not yet mirrored in the sibling ResilientSC
monorepo this project also exists in), both real and running, not mocked:

- **Real-time port telemetry + ML delay prediction** (`services/ports_realtime.py`,
  `api/routers/ports.py`) — extracts and serves port congestion, vessel queue, anchorage wait
  and maritime-chokepoint telemetry, and a trained delay-prediction model
  (`GET /api/ports/predict-delay`, `GET /api/ports/model-metrics`). `agents/logistics/tools.py`'s
  `calculate_eta` folds a live port delay into its ETA when available.
- **Human-approval workflow, two ways** (`sap/sbpa_bridge.py`, `sap/approval_portal.py`) — an
  escalated plan can be routed to **SAP Build Process Automation** (a CloudEvents webhook to an
  iFlow, which calls back `POST /api/decisions/{id}/approve|reject`) when `SBPA_WEBHOOK_URL` is
  set, or, with no SAP BTP environment at all, to a **built-in Approval Portal** — an HTML
  approval page at `/approval` with optional email notification (`APPROVAL_NOTIFY_EMAIL` +
  SMTP settings), so the human-approval gate works end to end with zero SAP configuration.

### This branch (`optimized`) specifically adds

`database/hana_core.py`'s `render_create_tables_sql()` / `02_create_tables.sql` — the same 17
`CREATE COLUMN TABLE` statements as `01_create_tables.sql`, with every table and `REFERENCES`
target pre-qualified by the target schema instead of relying on `SET SCHEMA`, for pasting into
a SQL console that may not already be in that schema. **Not yet run against a live HANA
tenant.**

## Running it

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# HANA Cloud / BTP / Integration Suite support (optional):
pip install -r requirements-sap.txt

cp .env.example .env    # fill in LLM_API_KEY at minimum (Gemini, aistudio.google.com, free tier)
uvicorn api.main:app --port 8000 --env-file .env
```

By default it runs on SQLite, the processed CSV files, no sign-in and no outbound integration —
every SAP-facing setting in `.env.example` is optional and off until configured (see
`AUTH_MODE`, `DATABASE_URL`, `EVENTS_BACKEND`, `SBPA_WEBHOOK_URL` there). CORS defaults to a
frontend dev server on `http://localhost:5173` — set `CORS_ORIGINS` (comma-separated) if the
frontend runs elsewhere.

Interactive API docs: `http://localhost:8000/docs`. Readiness: `GET /api/ready` (503 + which
dependency, if something required is missing).

```bash
pytest                     # everything: unit, integration, end-to-end
pytest -m "not e2e"        # fast path: no subprocesses, no browser
```

## Status

Datasets under `data/processed/` are being replaced with a corrected set by the data team
(same table names and columns, so the pipeline code above doesn't need to change) — the demand
model in `ml/` will be retrained once that lands. Treat anything derived from the current
`data/processed/*.csv` as provisional until that's done.
