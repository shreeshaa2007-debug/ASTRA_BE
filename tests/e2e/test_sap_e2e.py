"""Phase 21 — the SAP-readiness wiring, as a real process.

The unit tests inject their keys and their receivers. This starts the production wiring under a real uvicorn and lets the
*environment* configure everything a BTP deployment would: reference data read from database tables (`DATA_BACKEND=sql`),
callers authenticated by JWTs whose signing keys are fetched over HTTP from a JWKS endpoint (as XSUAA publishes them), and
every business milestone POSTed as a CloudEvent to a webhook (as an Integration Suite flow would receive it).

What is still not real: the identity provider (a local JWKS server signs nothing, it just serves a public key), the
receiver (a local HTTP server), the database (SQLite). See docs/sap-readiness.md for what that leaves unproven.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization

from backend.data.loader import load_csvs_into
from backend.database.engine import DatabaseTarget, build_engine
from backend.simulation.comparison import WorldSpec, solve_world
from backend.tests.e2e.conftest import requires_built_data
from backend.tests.test_sap_integration import Receiver
from backend.tests.test_sap_security import ISSUER, KEY, OTHER_KEY, PREFIX, PUBLIC, XSAPP, bearer, claims, token

pytestmark = [pytest.mark.e2e, requires_built_data]

EVENTS_TOKEN = "events-bearer-token-that-must-never-be-logged"
FIRE_TEXT = "Fire at the Istanbul plant"  # -> the scripted model reads a supplier failure (S007): a plan compliance sends to a human


class Jwks:
    """Serves the public half of the signing key at /keys, the way an identity provider publishes it."""

    def __init__(self):
        jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(PUBLIC))
        body = json.dumps({"keys": [{**jwk, "kid": "k1", "use": "sig", "alg": "RS256"}]}).encode()
        self.requests = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                outer.requests += 1
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/keys"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


OPERATOR = claims(("View", "Operate"), sub="op-1", email="olivia.operator@example.com", user_name="olivia")
APPROVER = claims(("View", "Approve"), sub="ap-1", email="alice.approver@example.com", user_name="alice")
READER = claims(("View",), sub="rd-1", email="rita.reader@example.com", user_name="rita")


@pytest.fixture(scope="module")
def stack(make_server, tmp_path_factory):
    work = tmp_path_factory.mktemp("sap-e2e")
    db = work / "sap.db"
    load_csvs_into(build_engine(DatabaseTarget(f"sqlite:///{db.as_posix()}", None, "e2e")), "data/processed")  # what scripts/load_reference_data.py does

    jwks, receiver = Jwks(), Receiver()
    server = make_server(db=db, env={
        "E2E_WIRING": "default",
        "DATA_BACKEND": "sql",
        "AUTH_MODE": "jwt", "AUTH_JWKS_URL": jwks.url, "AUTH_ISSUER": ISSUER, "AUTH_AUDIENCE": f"{XSAPP},sb-{XSAPP}", "AUTH_SCOPE_PREFIX": PREFIX,
        "EVENTS_BACKEND": "webhook", "EVENTS_WEBHOOK_URL": receiver.url + "/hook/{topic}", "EVENTS_AUTH": "bearer", "EVENTS_BEARER_TOKEN": EVENTS_TOKEN,
    }).start()
    # the first solve loads the forecasting model; do it before a test is timed
    httpx.get(f"{server.url}/api/scenarios/SUEZ_CLOSURE/comparison", headers=bearer(token(OPERATOR)), timeout=120).raise_for_status()
    yield server, receiver, jwks
    receiver.close()
    jwks.close()


def wait_events(receiver: Receiver, count: int, timeout: float = 20) -> list[dict]:
    deadline = time.time() + timeout
    while time.time() < deadline and len(receiver.requests) < count:
        time.sleep(0.1)
    assert len(receiver.requests) >= count, f"only {len(receiver.requests)} of {count} events arrived: {[r['path'] for r in receiver.requests]}"
    return receiver.requests


def test_a_real_process_configured_only_by_the_environment_is_protected_and_ready(stack):
    server, _, jwks = stack
    with server.client() as api:
        assert api.get("/api/health").status_code == 200  # open: the platform's health check cannot sign in

        refused = api.get("/api/simulations")
        assert refused.status_code == 401 and refused.headers["www-authenticate"] == "Bearer" and refused.json()["error"]["error_code"] == "UNAUTHENTICATED"
        tampered = api.get("/api/simulations", headers=bearer(token(OPERATOR)[:-5] + "AAAAA"))
        forged = api.get("/api/simulations", headers=bearer(token(OPERATOR, key=OTHER_KEY.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))))  # someone else's key, our key id
        assert tampered.status_code == forged.status_code == 401

        assert api.get("/api/simulations", headers=bearer(token(READER))).status_code == 200  # a real token verified against fetched keys
        assert jwks.requests >= 1, "the signing keys were never fetched: the test did not exercise the JWKS path"
        me = api.get("/api/me", headers=bearer(token(READER))).json()["data"]
        assert me == {"name": "rita.reader@example.com", "authenticated": True, "is_user": True, "scopes": ["view"]}

        ready = api.get("/api/ready").json()["data"]
        checks = {c["name"]: c for c in ready["checks"]}
        assert ready["ready"] and checks["datasets"]["ok"] and "sql:" in checks["datasets"]["detail"], checks["datasets"]  # reading tables, not files
        assert checks["events"]["ok"] and "webhook" in checks["events"]["detail"] and EVENTS_TOKEN not in json.dumps(ready)


def test_an_integration_signal_becomes_an_approved_plan_and_every_milestone_is_announced(stack):
    server, receiver, _ = stack
    with server.client() as api:
        operator, approver, reader = (bearer(token(c)) for c in (OPERATOR, APPROVER, READER))
        signal = {"source_system": "s4hana-prod", "external_id": "PM-NOTIF-4711", "product_id": "22197", "report": FIRE_TEXT}

        assert api.post("/api/integration/signals", json=signal, headers=reader).status_code == 403  # a reader cannot start a response
        accepted = api.post("/api/integration/signals", json=signal, headers=operator)
        assert accepted.status_code == 202, accepted.text
        sim = accepted.json()["data"]["simulation_id"]
        deadline = time.time() + 60
        while time.time() < deadline:
            status = api.get(f"/api/simulations/{sim}/status", headers=reader).json()["data"]
            if status["status"] == "AWAITING_APPROVAL":
                break
            time.sleep(0.2)
        assert status["status"] == "AWAITING_APPROVAL" and status["awaiting_approval"]

        # -- what the integration flow has been told so far
        got = wait_events(receiver, 3)
        assert [r["path"] for r in got] == ["/hook/com/resilientsc/disruption/sensed", "/hook/com/resilientsc/plan/optimized", "/hook/com/resilientsc/plan/approval/requested"]
        assert all(r["headers"]["authorization"] == f"Bearer {EVENTS_TOKEN}" and r["headers"]["content-type"].startswith("application/cloudevents+json") for r in got)
        events = [json.loads(r["body"]) for r in got]
        assert all(e["specversion"] == "1.0" and e["subject"] == sim for e in events)
        assert events[2]["data"]["compliance"]["status"] == "ESCALATED" and events[2]["data"]["approval"]["status"] == "PENDING"
        assert events[1]["data"]["plan"]["status"] == "OPTIMAL" and events[1]["data"]["plan"]["allocations"]

        # -- the same signal again: recognised, not run again, not announced again
        again = api.post("/api/integration/signals", json=signal, headers=operator)
        assert again.status_code == 200 and again.json()["data"]["duplicate"] is True and again.json()["data"]["simulation_id"] == sim
        time.sleep(1.0)
        assert len(receiver.requests) == 3

        # -- who may release the plan, and under whose name
        version = api.get(f"/api/simulations/{sim}", headers=reader).json()["data"]["version"]
        body = {"decided_by": "Mallory the CFO", "expected_version": version}
        assert api.post(f"/api/decisions/{sim}/approve", json=body, headers=operator).status_code == 403  # running things is not approving them
        assert api.post(f"/api/decisions/{sim}/approve", json=body, headers=reader).status_code == 403
        done = api.post(f"/api/decisions/{sim}/approve", json=body, headers=approver)
        assert done.status_code == 200, done.text
        assert done.json()["data"]["approval_decision"]["decided_by"] == "alice.approver@example.com"  # the token's identity, not the body's claim

        final = json.loads(wait_events(receiver, 4)[3]["body"])
        assert final["type"] == "com.resilientsc.plan.finalized" and final["data"]["status"] == "COMPLETED"
        assert final["data"]["approval"]["decided_by"] == "alice.approver@example.com" and "Mallory" not in json.dumps(final)
        ids = [json.loads(r["body"])["id"] for r in receiver.requests]
        assert len(ids) == len(set(ids)) == 4 and all(i.startswith(f"{sim}:") for i in ids)

        # -- the plan built from database tables, behind auth, is the plan the file-based pipeline builds
        state = api.get(f"/api/simulations/{sim}", headers=reader).json()["data"]
        expected = solve_world("22197", None, WorldSpec.of([], ["S007"], state["tariffs"])).solution
        assert state["current_plan"]["objective_value"] == expected.objective_value and state["current_plan"]["status"] == "OPTIMAL"


def test_no_secret_or_token_reached_the_servers_log(stack):
    server, *_ = stack
    log = server.log.read_text(encoding="utf-8", errors="replace")
    assert log.count("checkpoint") > 0, "the log has nothing in it, so this proves nothing"
    for secret in (EVENTS_TOKEN, token(OPERATOR), token(APPROVER)):
        assert secret not in log
    assert "Bearer " not in log
