"""Phase 21 — who may call the API on SAP BTP.

Tokens here are real: signed with a freshly generated RSA key, verified by the real `JwtAuthenticator`, sent through the
real app. What is not real is the identity provider — its published keys are supplied directly rather than fetched from an
XSUAA tenant. The tests are about what the application does with a token, including the tokens an attacker would send.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from backend.agents.sensing.agent import SensingAgent
from backend.api.context import AppContext
from backend.api.main import create_app
from backend.api.runs import RunRegistry
from backend.api.security import (
    ANONYMOUS,
    JwtAuthenticator,
    NoAuthenticator,
    Principal,
    build_authenticator_from_env,
    decided_by,
)
from backend.database.world_state_repository import SqlAlchemyWorldStateRepository
from backend.orchestration import Orchestrator
from backend.sap import btp, xsuaa
from backend.services.world_state import WorldStateStore
from backend.tests.test_api import RULES_LOW, RULES_OPEN, FakeLLM, memoized_agent_outputs, requires_built_data, run_body  # noqa: F401 — the fixture is used by name

pytestmark = requires_built_data

ISSUER = "https://rsc.authentication.eu10.hana.ondemand.com/oauth/token"
XSAPP = "resilientsc!t4711"
PREFIX = f"{XSAPP}."


def new_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


KEY, OTHER_KEY = new_key(), new_key()
PUBLIC = KEY.public_key()
PRIVATE_PEM = KEY.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
PUBLIC_PEM = PUBLIC.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)


def claims(scopes=("View", "Operate", "Approve"), *, user=True, **override) -> dict:
    now = int(time.time())
    base = {"iss": ISSUER, "aud": [XSAPP, f"sb-{XSAPP}"], "iat": now, "exp": now + 3600, "sub": "user-1",
            "scope": [f"{PREFIX}{s}" for s in scopes] + ["openid"]}
    if user:
        base.update(user_name="alice", email="alice@example.com", grant_type="authorization_code")
    else:  # what XSUAA issues for a client-credentials grant: a technical client, no person
        base.update(sub="sb-workflow", client_id="sb-workflow", cid="sb-workflow", grant_type="client_credentials")
    base.update(override)
    return {k: v for k, v in base.items() if v is not None}


def token(payload=None, key=PRIVATE_PEM, alg="RS256", **kw) -> str:
    return jwt.encode(payload if payload is not None else claims(), key, algorithm=alg, headers={"kid": "k1"}, **kw)


def bearer(tok: str) -> dict:
    return {"Authorization": f"Bearer {tok}"}


def b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def forged_hs256(payload: dict, secret: bytes) -> str:
    """The classic algorithm-confusion token: HS256, keyed with the RSA *public* key (which everyone has)."""
    head, body = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()), b64(json.dumps(payload).encode())
    signature = hmac.new(secret, f"{head}.{body}".encode(), hashlib.sha256).digest()
    return f"{head}.{body}.{b64(signature)}"


def unsigned(payload: dict) -> str:
    return f"{b64(json.dumps({'alg': 'none', 'typ': 'JWT'}).encode())}.{b64(json.dumps(payload).encode())}."


def authenticator(**over) -> JwtAuthenticator:
    return JwtAuthenticator(**{"issuer": ISSUER, "audience": [XSAPP, f"sb-{XSAPP}"], "key_resolver": lambda _token: PUBLIC, "scope_prefix": PREFIX, **over})


def make(auth=None, rules=RULES_OPEN, llm=None):
    store = WorldStateStore(SqlAlchemyWorldStateRepository("sqlite:///:memory:"))
    orchestrator = Orchestrator(store, SensingAgent(llm=llm or FakeLLM()), compliance_rules=rules)
    ctx = AppContext(store, orchestrator, RunRegistry(None))
    return TestClient(create_app(context=ctx, authenticator=auth), raise_server_exceptions=False), store


def envelope(response, status, code):
    assert response.status_code == status, response.text
    error = response.json()["error"]
    assert error["error_code"] == code and error["message"], error
    return error


# --------------------------------------------------------------------------- #
# with authentication off, nothing changed
# --------------------------------------------------------------------------- #
def test_with_authentication_off_every_caller_can_do_everything_as_before():
    client, _ = make()  # the default: NoAuthenticator
    assert client.post("/api/simulations", json={"scenario_type": "T"}).status_code == 201
    me = client.get("/api/me").json()["data"]
    assert me == {"name": "anonymous", "authenticated": False, "is_user": False, "scopes": ["approve", "operate", "view"]}


def test_with_authentication_off_the_approver_is_the_name_in_the_request():
    client, _ = make(rules=RULES_LOW)
    sim = client.post("/api/simulations", json={"scenario_type": "T"}).json()["data"]["simulation_id"]
    client.post(f"/api/simulations/{sim}/run", json=run_body())
    approved = client.post(f"/api/decisions/{sim}/approve", json={"decided_by": "Dana Director"}).json()["data"]
    assert approved["approval_decision"]["decided_by"] == "Dana Director"


# --------------------------------------------------------------------------- #
# with a JWT authenticator: who is let in
# --------------------------------------------------------------------------- #
def test_a_request_without_a_token_is_401_with_the_challenge_and_the_usual_envelope():
    client, _ = make(authenticator())
    for method, path in (("get", "/api/simulations"), ("get", "/api/dashboard"), ("post", "/api/simulations"), ("get", "/api/metrics"), ("get", "/api/me")):
        response = getattr(client, method)(path)
        error = envelope(response, 401, "UNAUTHENTICATED")
        assert response.headers["www-authenticate"] == "Bearer" and error["recovery"] and error["request_id"], path


def test_probes_stay_open_so_the_platform_can_health_check_an_app_it_cannot_sign_in_to():
    client, _ = make(authenticator())
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/ready").status_code in (200, 503)  # answers; whether it is ready is another question
    assert client.get("/api/health").json()["status"] == "ok"


def test_a_valid_token_is_admitted_and_me_says_who_and_what():
    client, _ = make(authenticator())
    me = client.get("/api/me", headers=bearer(token())).json()["data"]
    assert me == {"name": "alice@example.com", "authenticated": True, "is_user": True, "scopes": ["approve", "operate", "view"]}


@pytest.mark.parametrize("make_token,why", [
    (lambda: token(claims(exp=int(time.time()) - 3600)), "expired"),
    (lambda: token(claims(iss="https://evil.example/oauth/token")), "issued by someone else"),
    (lambda: token(claims(aud=["some-other-app"])), "issued for another app"),
    (lambda: token(claims(aud=None)), "no audience at all"),
    (lambda: token(claims(exp=None)), "never expires"),
    (lambda: token(key=OTHER_KEY.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())), "signed with someone else's key"),
    (lambda: forged_hs256(claims(), PUBLIC_PEM), "algorithm confusion: HS256 keyed with the public key"),
    (lambda: unsigned(claims()), "alg none"),
    (lambda: token()[:-4] + "AAAA", "signature tampered with"),
    (lambda: "not.a.jwt", "garbage"),
])
def test_a_token_that_is_not_right_for_this_app_is_refused_without_saying_which_check_failed(make_token, why):
    client, _ = make(authenticator())
    tok = make_token()
    response = client.get("/api/simulations", headers=bearer(tok))
    error = envelope(response, 401, "UNAUTHENTICATED")
    assert error["message"] in ("the token is not valid", "the token has expired"), why  # no hint an attacker can adjust to
    assert tok not in response.text


@pytest.mark.parametrize("header", ["Basic dXNlcjpwYXNz", "Bearer", "Bearer   ", "token abc", ""])
def test_only_a_bearer_token_will_do(header):
    client, _ = make(authenticator())
    envelope(client.get("/api/simulations", headers={"Authorization": header}), 401, "UNAUTHENTICATED")


def test_the_signing_keys_being_unreachable_is_503_not_a_verdict_on_the_caller():
    def unreachable(_token):
        raise jwt.PyJWKClientConnectionError("cannot reach the identity provider")

    client, _ = make(authenticator(key_resolver=unreachable))
    response = client.get("/api/simulations", headers=bearer(token()))
    error = envelope(response, 503, "AUTH_UNAVAILABLE")
    assert "www-authenticate" not in response.headers and error["recovery"]  # a challenge would tell a client to fetch a new token; that is not the problem


# --------------------------------------------------------------------------- #
# scopes
# --------------------------------------------------------------------------- #
def test_a_reader_can_read_but_not_start_anything_and_not_approve():
    client, _ = make(authenticator())
    reader = bearer(token(claims(scopes=("View",))))
    assert client.get("/api/simulations", headers=reader).status_code == 200
    assert client.get("/api/scenarios", headers=reader).status_code == 200
    assert client.get("/api/metrics", headers=reader).status_code == 200
    for method, path, body in (("post", "/api/simulations", {"scenario_type": "T"}), ("post", "/api/scenarios/suez_closure/run", {}),
                               ("post", "/api/simulations/x/run", run_body()), ("post", "/api/simulations/x/reset", None),
                               ("post", "/api/decisions/x/approve", {"decided_by": "a"}), ("post", "/api/decisions/x/reject", {"decided_by": "a"}),
                               ("post", "/api/integration/signals", {})):
        response = getattr(client, method)(path, headers=reader, **({"json": body} if body is not None else {}))
        assert response.status_code == 403 and response.json()["error"]["error_code"] == "FORBIDDEN", path


def test_a_token_with_no_scopes_at_all_is_authenticated_but_can_do_nothing():
    client, _ = make(authenticator())
    nobody = bearer(token(claims(scopes=())))
    assert client.get("/api/me", headers=nobody).json()["data"]["scopes"] == []
    envelope(client.get("/api/simulations", headers=nobody), 403, "FORBIDDEN")


def test_scopes_belonging_to_another_application_grant_nothing_here():
    """A token for `other!t9` with an `Approve` scope must not become approval rights on this app: only this app's own
    prefix is stripped, so the foreign scope stays `other!t9.approve` and matches nothing."""
    client, _ = make(authenticator())
    foreign = bearer(token(claims(scope=["other!t9.Approve", "other!t9.Operate", "other!t9.View"])))
    assert client.get("/api/me", headers=foreign).json()["data"]["scopes"] == []
    envelope(client.get("/api/simulations", headers=foreign), 403, "FORBIDDEN")


def test_approving_is_its_own_scope_an_operator_cannot_release_an_escalated_plan():
    client, store = make(authenticator(), rules=RULES_LOW)
    operator = bearer(token(claims(scopes=("View", "Operate"))))
    sim = client.post("/api/simulations", json={"scenario_type": "T"}, headers=operator).json()["data"]["simulation_id"]
    assert client.post(f"/api/simulations/{sim}/run", json=run_body(), headers=operator).status_code == 202
    assert store.get(sim).status.value == "AWAITING_APPROVAL"
    envelope(client.post(f"/api/decisions/{sim}/approve", json={"decided_by": "operator"}, headers=operator), 403, "FORBIDDEN")
    assert store.get(sim).status.value == "AWAITING_APPROVAL"  # and nothing was approved by trying

    approver = bearer(token(claims(scopes=("View", "Approve"))))
    assert client.post(f"/api/decisions/{sim}/approve", json={"decided_by": "x"}, headers=approver).status_code == 200


# --------------------------------------------------------------------------- #
# who an approval is recorded against
# --------------------------------------------------------------------------- #
def approved_by(client, store, tok, claimed):
    sim = client.post("/api/simulations", json={"scenario_type": "T"}, headers=bearer(token(claims()))).json()["data"]["simulation_id"]
    client.post(f"/api/simulations/{sim}/run", json=run_body(), headers=bearer(token(claims())))
    response = client.post(f"/api/decisions/{sim}/approve", json={"decided_by": claimed}, headers=bearer(tok))
    assert response.status_code == 200, response.text
    return store.get(sim).approval_decision.decided_by, [c.actor for c in store.history(sim)][-1]


def test_a_persons_approval_is_recorded_under_their_own_identity_whatever_the_request_claims():
    client, store = make(authenticator(), rules=RULES_LOW)
    recorded, actor = approved_by(client, store, token(claims()), claimed="Mallory the CFO")
    assert recorded == actor == "alice@example.com"


def test_a_workflows_approval_names_the_human_it_reports_and_the_client_that_vouched():
    """SAP Build Process Automation calls back with its own (technical) token after a person decided in a task inbox."""
    client, store = make(authenticator(), rules=RULES_LOW)
    recorded, actor = approved_by(client, store, token(claims(scopes=("View", "Approve"), user=False)), claimed="Dana Director")
    assert recorded == actor == "Dana Director (via sb-workflow)"


def test_a_long_identity_is_cut_to_what_the_audit_trail_can_hold_rather_than_failing_the_approval():
    long_name = "x" * 400 + "@example.com"
    principal = Principal("s", long_name, frozenset({"approve"}), authenticated=True, is_user=True)
    assert 0 < len(decided_by(principal, "ignored")) <= 256


def test_decided_by_unit_rules():
    person = Principal("s", "alice@example.com", frozenset(), authenticated=True, is_user=True)
    client = Principal("c", "sb-workflow", frozenset(), authenticated=True, is_user=False)
    assert decided_by(ANONYMOUS, "typed name") == "typed name"
    assert decided_by(person, "typed name") == "alice@example.com"
    assert decided_by(client, "Dana") == "Dana (via sb-workflow)"


def test_a_technical_token_is_recognised_by_its_grant_and_by_having_no_person_in_it():
    auth = authenticator()
    person = auth.authenticate(f"Bearer {token(claims())}")
    workflow = auth.authenticate(f"Bearer {token(claims(user=False))}")
    assert person.is_user and person.name == "alice@example.com"
    assert not workflow.is_user and workflow.name == "sb-workflow"
    lookalike = auth.authenticate(f"Bearer {token(claims(user=False, grant_type='authorization_code'))}")  # a token with no user in it is not a person
    assert not lookalike.is_user


def test_scopes_may_arrive_as_a_space_separated_string_as_some_identity_providers_send_them():
    principal = authenticator().authenticate(f"Bearer {token(claims(scope=f'{PREFIX}View {PREFIX}Approve openid'))}")
    assert principal.scopes == {"view", "approve"}


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
XSUAA_CREDENTIALS = {"url": "https://rsc.authentication.eu10.hana.ondemand.com/", "xsappname": XSAPP, "clientid": f"sb-{XSAPP}", "clientsecret": "s3cret"}
XSUAA_ENV = {"VCAP_APPLICATION": "{}", "VCAP_SERVICES": json.dumps({"xsuaa": [{"name": "rsc-uaa", "label": "xsuaa", "tags": ["xsuaa"], "credentials": XSUAA_CREDENTIALS}]})}


def test_xsuaa_settings_come_from_the_binding():
    settings = xsuaa.jwt_settings(XSUAA_CREDENTIALS)
    assert settings == {"jwks_url": "https://rsc.authentication.eu10.hana.ondemand.com/token_keys", "issuer": ISSUER,
                        "audience": [XSAPP, f"sb-{XSAPP}"], "scope_prefix": PREFIX}
    with pytest.raises(btp.BTPConfigError, match="xsuaa binding is missing credentials: xsappname"):
        xsuaa.jwt_settings({"url": "https://x", "clientid": "c"})


def test_the_default_is_no_authentication_unless_an_xsuaa_instance_is_bound():
    assert isinstance(build_authenticator_from_env({}), NoAuthenticator)
    assert isinstance(build_authenticator_from_env({"VCAP_SERVICES": "{}"}), NoAuthenticator)


def test_an_app_bound_to_xsuaa_is_protected_unless_someone_says_otherwise_on_purpose():
    protected = build_authenticator_from_env(XSUAA_ENV)
    assert isinstance(protected, JwtAuthenticator) and "token_keys" in protected.describe()
    assert isinstance(build_authenticator_from_env({**XSUAA_ENV, "AUTH_MODE": "none"}), NoAuthenticator)


def test_an_incomplete_jwt_configuration_refuses_to_start_instead_of_falling_back_to_open():
    with pytest.raises(ValueError, match="missing: jwks_url, issuer, audience"):
        build_authenticator_from_env({"AUTH_MODE": "jwt"})
    with pytest.raises(ValueError, match="missing: issuer, audience"):
        build_authenticator_from_env({"AUTH_MODE": "jwt", "AUTH_JWKS_URL": "https://idp.example/keys"})
    with pytest.raises(ValueError, match="AUTH_MODE must be"):
        build_authenticator_from_env({"AUTH_MODE": "maybe"})


def test_the_app_itself_will_not_start_with_a_broken_identity_configuration(monkeypatch):
    """Built at startup, not on the first request: a misconfigured provider must stop the deployment, not fail open."""
    monkeypatch.setenv("AUTH_MODE", "jwt")
    for name in ("AUTH_JWKS_URL", "AUTH_ISSUER", "AUTH_AUDIENCE", "VCAP_SERVICES"):
        monkeypatch.delenv(name, raising=False)
    store = WorldStateStore(SqlAlchemyWorldStateRepository("sqlite:///:memory:"))
    ctx = AppContext(store, Orchestrator(store, SensingAgent(llm=FakeLLM())), RunRegistry(None))
    with pytest.raises(ValueError, match="AUTH_MODE=jwt needs"):
        create_app(context=ctx)


def test_explicit_settings_override_the_binding_and_work_without_one():
    env = {"AUTH_MODE": "jwt", "AUTH_JWKS_URL": "https://ias.example/oauth2/certs", "AUTH_ISSUER": "https://ias.example", "AUTH_AUDIENCE": "a, b", "AUTH_SCOPE_PREFIX": ""}
    described = build_authenticator_from_env(env).describe()
    assert "https://ias.example" in described and "['a', 'b']" in described and "ias.example/oauth2/certs" in described
    overridden = build_authenticator_from_env({**XSUAA_ENV, "AUTH_ISSUER": "https://ias.example"}).describe()
    assert "https://ias.example" in overridden and "token_keys" in overridden


@pytest.mark.parametrize("algorithms", [("HS256",), ("none",), ("RS256", "HS512")])
def test_a_shared_secret_or_unsigned_algorithm_cannot_even_be_configured(algorithms):
    with pytest.raises(ValueError, match="asymmetric"):
        authenticator(algorithms=algorithms)


def test_an_authenticator_needs_to_know_who_a_token_must_be_for():
    with pytest.raises(ValueError, match="issuer and an audience"):
        authenticator(audience=[])
    with pytest.raises(ValueError, match="AUTH_JWKS_URL"):
        JwtAuthenticator(issuer=ISSUER, audience=["a"])
