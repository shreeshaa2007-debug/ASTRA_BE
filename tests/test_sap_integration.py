"""Phase 21 — SAP Integration Suite readiness: what the application tells other systems, and what it accepts from them.

Outbound is tested against a real HTTP server on localhost (so what is asserted is what actually went over a socket:
headers, body, retries, a redirect that must not be followed) and against fakes where only the logic is at stake.
Nothing here talks to SAP; the receiving end of an iFlow or an Advanced Event Mesh broker is what is not covered.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from backend.agents.sensing.agent import SensingAgent
from backend.api.context import AppContext
from backend.api.main import create_app
from backend.api.runs import RunRegistry
from backend.database.world_state_repository import SqlAlchemyWorldStateRepository
from backend.integration import (
    DEFAULT_CHECKPOINTS,
    EVENT_TYPES,
    BackgroundPublisher,
    DomainEvent,
    EventDeliveryError,
    LoggingPublisher,
    NullPublisher,
    StoreEventBridge,
    WebhookPublisher,
    build_event,
    build_publisher_from_env,
    checkpoints_from_env,
)
from backend.integration.events import basic_auth, bearer_auth
from backend.integration.signals import IntegrationSignal, signal_simulation_id
from backend.monitoring.metrics import metrics
from backend.orchestration import Orchestrator
from backend.sap.oauth import ClientCredentialsTokenProvider, TokenError, require_https
from backend.services.world_state import StateChange, WorldStateStore
from backend.tests.test_api import PRODUCT, RULES_LOW, RULES_OPEN, FakeLLM, candidate, memoized_agent_outputs, requires_built_data, run_body  # noqa: F401

pytestmark = requires_built_data


# --------------------------------------------------------------------------- #
# a real server on localhost
# --------------------------------------------------------------------------- #
class Receiver:
    """Records every request; answers with the next scripted status (default 200)."""

    def __init__(self, statuses=()):
        self.requests: list[dict] = []
        self.statuses = list(statuses)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                outer.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
                status = outer.statuses.pop(0) if outer.statuses else 200
                self.send_response(status)
                if status in (301, 302, 307):
                    self.send_header("Location", "http://127.0.0.1:1/sign-in")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def receiver():
    r = Receiver()
    yield r
    r.close()


def event(n=1, type_="com.resilientsc.plan.finalized") -> DomainEvent:
    return DomainEvent(id=f"sim-1:{n}", type=type_, source="urn:resilientsc", subject="sim-1", time="2026-09-25T10:00:00+00:00", data={"n": n})


class Recorder:
    """A publisher that keeps what it was given."""

    def __init__(self):
        self.events: list[DomainEvent] = []

    def publish(self, e):
        self.events.append(e)

    def status(self):
        return {"backend": "recorder", "healthy": True}

    def close(self, timeout=5.0):
        pass


# --------------------------------------------------------------------------- #
# the event
# --------------------------------------------------------------------------- #
def test_an_event_is_a_cloudevents_1_0_structured_document():
    doc = event().to_dict()
    assert doc == {"specversion": "1.0", "id": "sim-1:1", "type": "com.resilientsc.plan.finalized", "source": "urn:resilientsc", "subject": "sim-1",
                   "time": "2026-09-25T10:00:00+00:00", "datacontenttype": "application/json", "data": {"n": 1}}


def test_every_checkpoint_the_store_has_maps_to_an_event_type():
    from backend.services.world_state import CHECKPOINTS
    assert set(EVENT_TYPES) == set(CHECKPOINTS) | {"simulation_created", "simulation_reset"}, "a new checkpoint needs an event type"
    assert all(t.startswith("com.resilientsc.") for t in EVENT_TYPES.values()) and len(set(EVENT_TYPES.values())) == len(EVENT_TYPES)
    assert set(DEFAULT_CHECKPOINTS) <= set(EVENT_TYPES)


def test_the_checkpoints_setting():
    assert checkpoints_from_env(None) == checkpoints_from_env("  ") == DEFAULT_CHECKPOINTS
    assert set(checkpoints_from_env("ALL")) == set(EVENT_TYPES)
    assert checkpoints_from_env("plan_finalized, run_failed") == ("plan_finalized", "run_failed")
    with pytest.raises(ValueError, match="unknown checkpoint"):
        StoreEventBridge(Recorder(), checkpoints=["plan_finalised"])  # a typo is an error at startup, not events that silently never come


# --------------------------------------------------------------------------- #
# the store's listener hook
# --------------------------------------------------------------------------- #
def store_with(*listeners):
    return WorldStateStore(SqlAlchemyWorldStateRepository("sqlite:///:memory:"), listeners=listeners)


def test_a_listener_hears_every_change_after_it_is_saved():
    heard = []
    store = store_with(lambda change: heard.append((change.checkpoint, change.state.version, change.actor, store.get(change.state.simulation_id).version)))
    sim = store.create("T", actor="someone").simulation_id
    store.reset(sim, actor="operator")
    assert heard == [("simulation_created", 0, "someone", 0), ("simulation_reset", 1, "operator", 1)]  # by then the store already agrees


def test_a_listener_that_raises_changes_nothing_and_does_not_stop_the_others():
    calls, before = [], metrics.counter_total("state_listener_errors_total")

    def broken(change):
        raise RuntimeError("the integration is down")

    store = store_with(broken, lambda change: calls.append(change.checkpoint))
    sim = store.create("T").simulation_id  # does not raise
    assert store.get(sim).version == 0 and calls == ["simulation_created"]
    assert metrics.counter_total("state_listener_errors_total") == before + 1


def test_a_listener_is_handed_a_copy_so_one_listener_cannot_alter_what_the_next_one_sees():
    seen = []

    def vandal(change: StateChange):
        change.state.tariffs["CHN"] = 9999.0
        change.state.scenario_type = "HACKED"

    store = store_with(vandal, lambda change: seen.append((change.state.scenario_type, change.state.tariffs.get("CHN"))))
    sim = store.create("T").simulation_id
    assert seen == [("T", store.get(sim).tariffs.get("CHN"))] and seen[0][1] != 9999.0  # the second listener saw the state as it was saved
    assert store.get(sim).scenario_type == "T"  # and the stored state was never in reach


def test_listeners_can_be_added_later():
    store, heard = store_with(), []
    store.add_listener(lambda change: heard.append(change.checkpoint))
    store.create("T")
    assert heard == ["simulation_created"]


# --------------------------------------------------------------------------- #
# the events a real run produces
# --------------------------------------------------------------------------- #
def make_app(rules=RULES_OPEN, publisher=None, checkpoints=DEFAULT_CHECKPOINTS, llm=None):
    store = WorldStateStore(SqlAlchemyWorldStateRepository("sqlite:///:memory:"))
    publisher = publisher or Recorder()
    store.add_listener(StoreEventBridge(publisher, checkpoints=checkpoints))
    llm = llm or FakeLLM()
    ctx = AppContext(store, Orchestrator(store, SensingAgent(llm=llm), compliance_rules=rules), RunRegistry(None), events=publisher)
    return TestClient(create_app(context=ctx), raise_server_exceptions=False), store, publisher, llm


def start(client, **overrides):
    sim = client.post("/api/simulations", json={"scenario_type": "T"}).json()["data"]["simulation_id"]
    assert client.post(f"/api/simulations/{sim}/run", json=run_body(**overrides)).status_code == 202
    return sim


def test_an_in_policy_plan_tells_the_world_it_was_sensed_optimized_and_finalized_and_nothing_else():
    client, _, recorder, _ = make_app()
    sim = start(client)
    assert [e.type.removeprefix("com.resilientsc.") for e in recorder.events] == ["disruption.sensed", "plan.optimized", "plan.finalized"]
    assert all(e.subject == sim and e.source == "urn:resilientsc" for e in recorder.events)  # internal steps (agents assessed, compliance) stay internal


def test_an_escalated_plan_asks_for_approval_and_a_decision_is_announced_with_who_made_it():
    client, _, recorder, _ = make_app(RULES_LOW)
    sim = start(client)
    kinds = [e.type.removeprefix("com.resilientsc.") for e in recorder.events]
    assert kinds == ["disruption.sensed", "plan.optimized", "plan.approval.requested"]
    asked = recorder.events[-1].data
    assert asked["status"] == "AWAITING_APPROVAL" and asked["compliance"]["requires_human"] is True and asked["approval"]["status"] == "PENDING"

    client.post(f"/api/decisions/{sim}/approve", json={"decided_by": "Dana Director", "note": "private note"})
    final = recorder.events[-1]
    assert final.type.endswith("plan.finalized") and final.data["approval"]["decided_by"] == "Dana Director" and final.data["status"] == "COMPLETED"
    assert "private note" not in json.dumps(final.to_dict())  # what a person typed is not broadcast


def test_a_rejection_and_a_failure_are_announced_too():
    client, _, recorder, _ = make_app(RULES_LOW)
    sim = start(client)
    client.post(f"/api/decisions/{sim}/reject", json={"decided_by": "cfo"})
    assert recorder.events[-1].type.endswith("plan.rejected") and recorder.events[-1].data["approval"]["status"] == "REJECTED"


def test_the_plan_travels_as_data_an_integration_can_act_on():
    client, _, recorder, _ = make_app()
    start(client)
    plan = recorder.events[-1].data["plan"]
    assert plan["product_id"] == PRODUCT and plan["status"] == "OPTIMAL" and plan["engine"] == "prototype" and plan["objective_value"] > 0
    assert plan["allocations"] and {"supplier_id", "route_id", "transport_mode", "quantity", "landed_unit_cost", "arrival_days"} <= set(plan["allocations"][0])
    assert "no currency" in plan["cost_unit"]  # a receiver must not assume rupees or euros
    assert recorder.events[0].data["plan"] is None and recorder.events[0].data["disruptions"][0]["event_type"] == "canal_closure"


def test_an_event_id_is_unique_per_change_and_the_same_on_a_redelivery():
    client, store, recorder, _ = make_app(checkpoints=list(EVENT_TYPES))
    sim = start(client)
    ids = [e.id for e in recorder.events]
    assert len(ids) == len(set(ids)) and all(i.startswith(f"{sim}:") for i in ids)
    replay = [build_event(StateChange(c.checkpoint, store.get(sim), c.actor), "urn:resilientsc").id for c in store.history(sim)[-1:]]
    assert replay == [f"{sim}:{store.get(sim).version}"]  # derived from (simulation, version): a retry carries the same key


def test_all_checkpoints_can_be_published_when_asked():
    client, _, recorder, _ = make_app(checkpoints=list(EVENT_TYPES))
    start(client)
    kinds = {e.type.removeprefix("com.resilientsc.") for e in recorder.events}
    assert {"simulation.created", "agents.assessed", "plan.compliance.checked"} <= kinds


def test_a_publisher_that_fails_does_not_fail_the_run_or_the_request():
    class Exploding(Recorder):
        def publish(self, e):
            raise RuntimeError("the broker is on fire")

    client, store, _, _ = make_app(publisher=Exploding())
    sim = start(client)
    assert store.get(sim).status.value == "COMPLETED"  # the plan was made and finalized regardless


# --------------------------------------------------------------------------- #
# delivery over HTTP
# --------------------------------------------------------------------------- #
def test_an_event_arrives_as_the_body_of_a_post_with_the_cloudevents_content_type(receiver):
    WebhookPublisher(receiver.url + "/http/resilientsc", auth=bearer_auth("tok-123")).publish(event())
    (got,) = receiver.requests
    assert got["path"] == "/http/resilientsc" and got["headers"]["content-type"].startswith("application/cloudevents+json")
    assert got["headers"]["authorization"] == "Bearer tok-123" and json.loads(got["body"]) == event().to_dict()


def test_the_url_can_carry_the_topic_for_a_broker_that_routes_by_path(receiver):
    WebhookPublisher(receiver.url + "/topic/{topic}").publish(event(type_="com.resilientsc.plan.approval.requested"))
    WebhookPublisher(receiver.url + "/by-type/{type}").publish(event())
    assert [r["path"] for r in receiver.requests] == ["/topic/com/resilientsc/plan/approval/requested", "/by-type/com.resilientsc.plan.finalized"]


def test_basic_auth_is_sent_as_basic_auth(receiver):
    WebhookPublisher(receiver.url, auth=basic_auth("integ", "p:w")).publish(event())
    assert receiver.requests[0]["headers"]["authorization"] == "Basic aW50ZWc6cDp3"


def test_a_server_error_is_retried_and_a_success_ends_it(receiver):
    receiver.statuses = [503, 502, 200]
    sleeps = []
    WebhookPublisher(receiver.url, max_retries=3, backoff_seconds=1.0, sleep=sleeps.append).publish(event())
    assert len(receiver.requests) == 3 and sleeps == [1.0, 2.0]  # backing off, then done


def test_a_client_error_is_final_because_retrying_the_same_bad_event_cannot_help(receiver):
    receiver.statuses = [400, 200]
    with pytest.raises(EventDeliveryError, match="HTTP 400"):
        WebhookPublisher(receiver.url, sleep=lambda s: None).publish(event())
    assert len(receiver.requests) == 1


def test_giving_up_says_which_event_and_why_after_the_configured_attempts(receiver):
    receiver.statuses = [500] * 10
    with pytest.raises(EventDeliveryError, match=r"plan.finalized for sim-1 was not delivered: HTTP 500"):
        WebhookPublisher(receiver.url, max_retries=2, sleep=lambda s: None).publish(event())
    assert len(receiver.requests) == 3


def test_a_redirect_is_a_failure_not_a_delivery(receiver):
    """A POST sent to an endpoint that answers with a redirect to a sign-in page must not look delivered."""
    receiver.statuses = [302]
    with pytest.raises(EventDeliveryError, match="HTTP 302"):
        WebhookPublisher(receiver.url, sleep=lambda s: None).publish(event())
    assert len(receiver.requests) == 1


def test_an_unreachable_endpoint_is_retried_then_reported():
    dead = Receiver()
    url = dead.url
    dead.close()
    sleeps = []
    with pytest.raises(EventDeliveryError, match="not delivered"):
        WebhookPublisher(url, max_retries=1, timeout_seconds=1, sleep=sleeps.append).publish(event())
    assert len(sleeps) == 1


def test_business_data_is_not_sent_in_clear_text_to_a_remote_host():
    with pytest.raises(ValueError, match="https"):
        WebhookPublisher("http://integration.example.com/events")
    with pytest.raises(ValueError, match="https"):
        WebhookPublisher("ftp://integration.example.com/events")
    WebhookPublisher("https://integration.example.com/events")
    WebhookPublisher("http://localhost:9000/events")  # this machine is fine: a local mock, a sidecar


# --------------------------------------------------------------------------- #
# OAuth client credentials
# --------------------------------------------------------------------------- #
class FakeTokenEndpoint:
    def __init__(self, tokens=None, fail=None, lifetime=3600):
        self.calls, self.requests, self.fail, self.lifetime = 0, [], fail, lifetime
        self.tokens = list(tokens or ["t1", "t2", "t3"])

    def __call__(self, request, timeout=None):
        self.calls += 1
        self.requests.append(request)
        if self.fail:
            raise self.fail
        body = json.dumps({"access_token": self.tokens[self.calls - 1], "expires_in": self.lifetime, "token_type": "bearer"}).encode()
        return _Response(body)


class _Response:
    status = 200

    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def provider(endpoint, clock, **kw):
    return ClientCredentialsTokenProvider("https://uaa.example/oauth/token", "cid", "s3cr3t/+=", opener=endpoint, clock=clock, **kw)


def test_a_token_is_fetched_once_and_reused_until_shortly_before_it_expires():
    now = [1000.0]
    endpoint = FakeTokenEndpoint(lifetime=3600)
    p = provider(endpoint, lambda: now[0], early_expiry_seconds=60)
    assert p.token() == p.token() == "t1" and endpoint.calls == 1
    now[0] += 3539
    assert p.token() == "t1"
    now[0] += 2  # inside the last minute: refreshed before it can expire mid-request
    assert p.token() == "t2" and endpoint.calls == 2


def test_the_client_secret_travels_in_a_basic_header_and_never_in_the_url_or_form():
    endpoint = FakeTokenEndpoint()
    provider(endpoint, time.monotonic, scope="events.send").token()
    request = endpoint.requests[0]
    assert request.get_header("Authorization").startswith("Basic ") and "s3cr3t" not in request.full_url
    assert b"s3cr3t" not in request.data and b"grant_type=client_credentials" in request.data and b"scope=events.send" in request.data


def test_an_invalidated_token_is_replaced():
    endpoint = FakeTokenEndpoint()
    p = provider(endpoint, time.monotonic)
    assert p.token() == "t1"
    p.invalidate()
    assert p.token() == "t2"


@pytest.mark.parametrize("failure", [
    urllib.error.HTTPError("https://uaa.example", 401, "unauthorized", {}, None),
    urllib.error.URLError("no route to host"), TimeoutError("timed out"),
])
def test_a_token_that_cannot_be_had_is_an_error_that_does_not_contain_the_secret(failure):
    p = provider(FakeTokenEndpoint(fail=failure), time.monotonic)
    with pytest.raises(TokenError) as caught:
        p.token()
    assert "s3cr3t" not in str(caught.value) and "s3cr3t" not in repr(caught.value.__cause__)


def test_an_answer_that_is_not_a_token_is_an_error():
    class Garbage(FakeTokenEndpoint):
        def __call__(self, request, timeout=None):
            return _Response(b'{"error": "unauthorized_client"}')

    with pytest.raises(TokenError, match="no access_token"):
        provider(Garbage(), time.monotonic).token()


def test_the_token_endpoint_must_be_https_and_the_credentials_present():
    with pytest.raises(ValueError, match="https"):
        ClientCredentialsTokenProvider("http://uaa.example/oauth/token", "a", "b")
    with pytest.raises(ValueError, match="both required"):
        ClientCredentialsTokenProvider("https://uaa.example/oauth/token", "a", "")
    require_https("http://127.0.0.1:8080/token", "x")


def test_a_401_from_the_receiver_gets_one_fresh_token_and_one_more_try(receiver):
    endpoint = FakeTokenEndpoint()
    p = provider(endpoint, time.monotonic)
    receiver.statuses = [401, 200]
    WebhookPublisher(receiver.url, auth=p.authorization_header, on_unauthorized=p.invalidate).publish(event())
    assert [r["headers"]["authorization"] for r in receiver.requests] == ["Bearer t1", "Bearer t2"] and endpoint.calls == 2


def test_a_receiver_that_keeps_saying_401_is_not_hammered_with_new_tokens(receiver):
    endpoint = FakeTokenEndpoint(tokens=["t1", "t2", "t3", "t4"])
    p = provider(endpoint, time.monotonic)
    receiver.statuses = [401] * 10
    with pytest.raises(EventDeliveryError, match="HTTP 401"):
        WebhookPublisher(receiver.url, auth=p.authorization_header, on_unauthorized=p.invalidate, sleep=lambda s: None).publish(event())
    assert endpoint.calls == 2  # the original token, and the one refresh


# --------------------------------------------------------------------------- #
# the background queue
# --------------------------------------------------------------------------- #
class Gate(Recorder):
    """A publisher that waits for permission — a slow endpoint."""

    def __init__(self):
        super().__init__()
        self.open = threading.Event()

    def publish(self, e):
        assert self.open.wait(10), "the test never opened the gate"
        super().publish(e)


def test_publishing_never_waits_for_a_slow_endpoint_and_events_go_out_in_order():
    gate = Gate()
    background = BackgroundPublisher(gate)
    started = time.perf_counter()
    for n in range(20):
        background.publish(event(n))
    assert time.perf_counter() - started < 0.5  # the caller was not held up
    gate.open.set()
    assert background.flush(5) and [e.data["n"] for e in gate.events] == list(range(20))
    assert background.status()["published"] == 20
    background.close()


def test_a_full_queue_drops_and_says_so_instead_of_blocking_or_growing_without_bound():
    gate = Gate()
    background = BackgroundPublisher(gate, queue_size=2)
    before = metrics.counter_total("events_dropped_total")
    for n in range(10):
        background.publish(event(n))
    time.sleep(0.05)
    status = background.status()
    assert status["dropped"] >= 6 and not status["healthy"]
    assert metrics.counter_total("events_dropped_total") - before == status["dropped"]
    gate.open.set()
    background.close()


def test_a_failing_delivery_is_counted_and_the_worker_carries_on():
    class Flaky(Recorder):
        def publish(self, e):
            if e.data["n"] % 2 == 0:
                raise EventDeliveryError("nope")
            super().publish(e)

    background = BackgroundPublisher(Flaky())
    for n in range(6):
        background.publish(event(n))
    assert background.flush(5)
    status = background.status()
    assert (status["published"], status["failed"]) == (3, 3) and status["consecutive_failures"] == 0 and status["healthy"]
    assert status["last_error"] == "nope"
    background.close()


def test_health_turns_bad_after_consecutive_failures_and_recovers_on_a_success():
    class Switch(Recorder):
        ok = False

        def publish(self, e):
            if not self.ok:
                raise EventDeliveryError("down")
            super().publish(e)

    inner = Switch()
    background = BackgroundPublisher(inner, unhealthy_after=3)
    for n in range(3):
        background.publish(event(n))
    background.flush(5)
    assert not background.status()["healthy"]
    inner.ok = True
    background.publish(event(9))
    background.flush(5)
    assert background.status()["healthy"]
    background.close()


def test_closing_sends_what_is_queued_first():
    gate = Gate()
    background = BackgroundPublisher(gate)
    for n in range(5):
        background.publish(event(n))
    gate.open.set()
    background.close(timeout=5)
    assert len(gate.events) == 5


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def test_by_default_nothing_leaves_the_process():
    assert isinstance(build_publisher_from_env({}), NullPublisher) and build_publisher_from_env({}).status() == {"backend": "none", "healthy": True}
    assert isinstance(build_publisher_from_env({"EVENTS_BACKEND": "log"}), LoggingPublisher)


def test_a_webhook_is_built_from_the_environment_and_reports_where_it_sends_without_its_secrets():
    env = {"EVENTS_BACKEND": "webhook", "EVENTS_WEBHOOK_URL": "https://integration.example/http/rsc?apikey=SECRETKEY",
           "EVENTS_OAUTH_TOKEN_URL": "https://uaa.example/oauth/token", "EVENTS_OAUTH_CLIENT_ID": "cid", "EVENTS_OAUTH_CLIENT_SECRET": "s3cr3t"}
    publisher = build_publisher_from_env(env)
    try:
        status = publisher.status()
        assert status["backend"] == "webhook" and status["target"] == "https://integration.example/http/rsc"
        assert "SECRETKEY" not in json.dumps(status) and "s3cr3t" not in json.dumps(status)  # the query string can carry a key; it is not shown
    finally:
        publisher.close()


@pytest.mark.parametrize("env,message", [
    ({"EVENTS_BACKEND": "kafka"}, "EVENTS_BACKEND must be"),
    ({"EVENTS_BACKEND": "webhook"}, "needs EVENTS_WEBHOOK_URL"),
    ({"EVENTS_BACKEND": "webhook", "EVENTS_WEBHOOK_URL": "https://x", "EVENTS_AUTH": "oauth2"}, "EVENTS_OAUTH_TOKEN_URL, EVENTS_OAUTH_CLIENT_ID, EVENTS_OAUTH_CLIENT_SECRET"),
    ({"EVENTS_BACKEND": "webhook", "EVENTS_WEBHOOK_URL": "https://x", "EVENTS_AUTH": "basic"}, "EVENTS_BASIC_USER"),
    ({"EVENTS_BACKEND": "webhook", "EVENTS_WEBHOOK_URL": "https://x", "EVENTS_AUTH": "bearer"}, "EVENTS_BEARER_TOKEN"),
    ({"EVENTS_BACKEND": "webhook", "EVENTS_WEBHOOK_URL": "https://x", "EVENTS_AUTH": "kerberos"}, "EVENTS_AUTH must be"),
    ({"EVENTS_BACKEND": "webhook", "EVENTS_WEBHOOK_URL": "http://remote.example"}, "https"),
])
def test_a_half_configured_integration_stops_the_app_from_starting_and_says_what_is_missing(env, message):
    with pytest.raises(ValueError, match=message):
        build_publisher_from_env(env)


def test_readiness_reports_the_integration_as_an_optional_check():
    client, _, publisher, _ = make_app(publisher=BackgroundPublisher(WebhookPublisher("https://integration.example/e")))
    publisher.close(0.1)
    checks = {c["name"]: c for c in client.get("/api/ready").json()["data"]["checks"]}
    assert checks["events"]["required"] is False and checks["events"]["ok"] and "webhook -> https://integration.example/e" in checks["events"]["detail"]
    off = {c["name"] for c in make_app()[0].get("/api/ready").json()["data"]["checks"]}
    assert "events" in off  # a recorder counts as configured: only the `none` backend hides the check
    none_client = TestClient(create_app(context=AppContext(*(lambda s: (s, Orchestrator(s, SensingAgent(llm=FakeLLM())), RunRegistry(None)))(
        WorldStateStore(SqlAlchemyWorldStateRepository("sqlite:///:memory:"))))))
    assert "events" not in {c["name"] for c in none_client.get("/api/ready").json()["data"]["checks"]}


# --------------------------------------------------------------------------- #
# inbound signals
# --------------------------------------------------------------------------- #
def signal(external_id="NOTIF-1", source="s4hana-prod", **overrides):
    body = {"source_system": source, "external_id": external_id, "product_id": PRODUCT, "report": "A vessel is aground in the Suez Canal."}
    body.update(overrides)
    return body


def test_a_signal_starts_a_run_and_a_redelivery_of_it_does_not():
    client, store, recorder, llm = make_app()
    first = client.post("/api/integration/signals", json=signal())
    assert first.status_code == 202
    body = first.json()["data"]
    sim = body["simulation_id"]
    assert body["duplicate"] is False and sim == signal_simulation_id("s4hana-prod", "NOTIF-1") and body["status_url"] == f"/api/simulations/{sim}/status"
    assert store.get(sim).status.value == "COMPLETED" and llm.calls == 1
    assert [c.actor for c in store.history(sim)][0] == "integration:s4hana-prod"  # who asked for it, in the audit trail

    again = client.post("/api/integration/signals", json=signal())
    assert again.status_code == 200 and again.json()["data"] == {"simulation_id": sim, "duplicate": True, "status": "COMPLETED", "status_url": body["status_url"]}
    assert llm.calls == 1 and len(recorder.events) == 3  # not sensed again, not re-announced


def test_the_same_id_from_a_different_system_is_a_different_signal():
    assert signal_simulation_id("s4hana-prod", "N1") != signal_simulation_id("ariba", "N1") != signal_simulation_id("ariba", "N2")
    assert signal_simulation_id("a", "b") == signal_simulation_id("a", "b") and len(signal_simulation_id("a", "b" * 128)) <= 64
    assert signal_simulation_id("ab", "c") != signal_simulation_id("a", "bc")  # the pair is hashed with a separator, not glued


def test_two_deliveries_of_one_signal_racing_start_exactly_one_run():
    class Slow(FakeLLM):
        def generate_json(self, **kwargs):
            time.sleep(0.2)
            return super().generate_json(**kwargs)

    from concurrent.futures import ThreadPoolExecutor
    store = WorldStateStore(SqlAlchemyWorldStateRepository("sqlite:///:memory:"))
    executor = ThreadPoolExecutor(max_workers=4)
    llm = Slow()
    ctx = AppContext(store, Orchestrator(store, SensingAgent(llm=llm), compliance_rules=RULES_OPEN), RunRegistry(executor))
    client = TestClient(create_app(context=ctx), raise_server_exceptions=False)
    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = sorted(pool.map(lambda _: client.post("/api/integration/signals", json=signal("RACE")).status_code, range(8)))
    executor.shutdown(wait=True)
    assert statuses == [200] * 7 + [202] and llm.calls == 1


def test_a_structured_candidate_skips_the_llm_but_not_validation():
    client, store, _, llm = make_app()
    ok = client.post("/api/integration/signals", json=signal("STRUCT-1", candidate={"is_disruption": True, **{k: v for k, v in candidate().items() if k != "is_disruption"}}))
    assert ok.status_code == 202 and llm.calls == 0
    assert store.get(ok.json()["data"]["simulation_id"]).current_disruptions  # accepted on its own say-so about the shape, checked against the network

    bad = client.post("/api/integration/signals", json=signal("STRUCT-2", candidate=candidate(affected_routes=["NO-SUCH-ROUTE"])))
    assert bad.status_code == 202  # accepted for processing...
    assert store.get(bad.json()["data"]["simulation_id"]).status.value == "CREATED" and llm.calls == 0  # ...and refused by validation, with no event in the state


@pytest.mark.parametrize("override,field", [
    ({"source_system": "has space"}, "source_system"), ({"source_system": ""}, "source_system"), ({"external_id": "bad id"}, "external_id"),
    ({"external_id": "x" * 129}, "external_id"), ({"report": "   "}, "report"), ({"report": "x" * 5000}, "report"), ({"product_id": ""}, "product_id"),
])
def test_a_malformed_signal_is_refused_at_the_door_naming_the_field(override, field):
    client, store, _, llm = make_app()
    response = client.post("/api/integration/signals", json=signal(**override))
    assert response.status_code == 422 and field in response.json()["error"]["message"]
    assert store.list_simulations() == [] and llm.calls == 0


def test_an_unknown_product_or_tariff_country_creates_nothing():
    client, store, _, _ = make_app()
    assert client.post("/api/integration/signals", json=signal(product_id="NOPE")).status_code == 422
    assert client.post("/api/integration/signals", json=signal(tariff_overrides={"ATLANTIS": 10.0})).status_code == 422
    assert store.list_simulations() == []  # so the corrected signal is not mistaken for a duplicate of a half-made one


def test_a_signal_can_carry_a_tariff_change_the_plan_then_reflects():
    client, store, recorder, _ = make_app()
    base = client.post("/api/integration/signals", json=signal("T-1")).json()["data"]["simulation_id"]
    changed = client.post("/api/integration/signals", json=signal("T-2", tariff_overrides={"TUR": 60.0})).json()["data"]["simulation_id"]
    assert store.get(changed).tariffs["TUR"] == 60.0 and store.get(base).tariffs["TUR"] != 60.0


def test_the_integration_endpoint_is_documented_in_the_openapi_contract():
    client, *_ = make_app()
    spec = client.get("/openapi.json").json()
    assert "/api/integration/signals" in spec["paths"] and "IntegrationSignal" in spec["components"]["schemas"]
    assert IntegrationSignal.model_json_schema()["required"] == ["source_system", "external_id", "product_id", "report"]
