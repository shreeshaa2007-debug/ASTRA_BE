"""Who is calling, and may they.

Three scopes: `view` (read anything), `operate` (create, run and reset simulations, start a run from an integration
signal) and `approve` (approve or reject an escalated plan). They are separate on purpose: whoever can start runs is not
thereby someone who can release an escalated plan. Scopes are cumulative only by role assignment (an Operator role
carries view + operate), never by inference here.

Two modes, chosen by `AUTH_MODE`:

    none    (default; what tests and local development use) every caller is an anonymous principal holding every scope.
            The API behaves exactly as it always has, including taking the approver's name from the request body.
    jwt     every request must carry `Authorization: Bearer <JWT>`. The token is verified (signature against the
            identity provider's published keys, issuer, audience, expiry) and its scopes checked per endpoint. An approval
            is then recorded under the *token's* identity, whatever the request body says — the answer to "the approver
            is just a string".

With no `AUTH_MODE` set, a bound XSUAA instance (`VCAP_SERVICES`) turns `jwt` on by itself: an app deployed to SAP BTP is
protected unless someone says otherwise on purpose. A `jwt` configuration that is incomplete stops the application from
starting; it never falls back to `none`.

    AUTH_JWKS_URL / AUTH_ISSUER / AUTH_AUDIENCE (comma-separated) / AUTH_SCOPE_PREFIX     when not taken from XSUAA
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from fastapi import Depends, Request

from backend.api.errors import ApiError
from backend.database.world_state_repository import ACTOR_MAX_LENGTH
from backend.monitoring.metrics import metrics
from backend.sap import btp, xsuaa

logger = logging.getLogger("resilientsc.api.security")

VIEW, OPERATE, APPROVE = "view", "operate", "approve"
SCOPES = frozenset({VIEW, OPERATE, APPROVE})
_BEARER = {"WWW-Authenticate": "Bearer"}


@dataclass(frozen=True)
class Principal:
    subject: str  # stable identifier (the token's `sub`)
    name: str  # what the audit trail records
    scopes: frozenset[str]
    authenticated: bool
    is_user: bool = False  # a person, as opposed to a technical client such as a workflow calling back

    def can(self, scope: str) -> bool:
        return scope in self.scopes


ANONYMOUS = Principal("anonymous", "anonymous", SCOPES, authenticated=False)


class AuthenticationError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code = status, code


class Authenticator(Protocol):
    def authenticate(self, authorization: str | None) -> Principal: ...
    def describe(self) -> str: ...


class NoAuthenticator:
    def authenticate(self, authorization: str | None) -> Principal:
        return ANONYMOUS

    def describe(self) -> str:
        return "none (every caller is anonymous with every scope)"


class JwtAuthenticator:
    def __init__(
        self, *, issuer: str, audience: Sequence[str], jwks_url: str | None = None, key_resolver: Callable[[str], Any] | None = None,
        scope_prefix: str = "", algorithms: Sequence[str] = ("RS256",), leeway_seconds: int = 30, keys_cache_seconds: int = 3600,
    ):
        import jwt  # PyJWT[crypto]: imported here so the default (auth off) deployment does not need it

        self._jwt = jwt
        if not issuer or not audience:
            raise ValueError("JWT authentication needs an issuer and an audience: a token is only good for the app it was issued for")
        if jwks_url is None and key_resolver is None:
            raise ValueError("JWT authentication needs AUTH_JWKS_URL (or a key resolver)")
        if any(a.lower().startswith("hs") or a.lower() == "none" for a in algorithms):
            raise ValueError("only asymmetric algorithms are accepted: a shared-secret or unsigned token proves nothing")
        self._issuer, self._audience, self._prefix = issuer, list(audience), scope_prefix
        self._algorithms, self._leeway = list(algorithms), leeway_seconds
        self._jwks_url = jwks_url
        if key_resolver is not None:
            self._resolve = key_resolver
        else:
            client = jwt.PyJWKClient(jwks_url, cache_keys=True, lifespan=keys_cache_seconds, timeout=10)
            self._resolve = lambda token: client.get_signing_key_from_jwt(token).key

    def describe(self) -> str:
        return f"jwt (issuer {self._issuer}, audience {self._audience}, keys {self._jwks_url or 'injected'})"

    def authenticate(self, authorization: str | None) -> Principal:
        jwt = self._jwt
        token = self._bearer(authorization)
        try:
            key = self._resolve(token)
            claims = jwt.decode(token, key, algorithms=self._algorithms, audience=self._audience, issuer=self._issuer,
                                leeway=self._leeway, options={"require": ["exp", "iss", "aud"]})
        except jwt.ExpiredSignatureError:
            raise AuthenticationError(401, "UNAUTHENTICATED", "the token has expired") from None
        except jwt.PyJWKClientConnectionError:
            raise AuthenticationError(503, "AUTH_UNAVAILABLE", "the identity provider's signing keys could not be fetched") from None
        except jwt.PyJWTError as exc:
            # which check failed is for the log, not for the caller: it would tell an attacker what to adjust
            logger.warning("token rejected: %s", type(exc).__name__, extra={"event": "auth_rejected", "reason": type(exc).__name__})
            raise AuthenticationError(401, "UNAUTHENTICATED", "the token is not valid") from None
        return self._principal(claims)

    @staticmethod
    def _bearer(authorization: str | None) -> str:
        if not authorization:
            raise AuthenticationError(401, "UNAUTHENTICATED", "no bearer token was sent")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise AuthenticationError(401, "UNAUTHENTICATED", "the Authorization header must be 'Bearer <token>'")
        return token.strip()

    def _principal(self, claims: Mapping[str, Any]) -> Principal:
        raw = claims.get("scope", claims.get("scp", []))
        granted = raw.split() if isinstance(raw, str) else [str(s) for s in raw]
        scopes = frozenset(
            (s[len(self._prefix):] if self._prefix and s.startswith(self._prefix) else s).lower() for s in granted
        ) & SCOPES
        technical = claims.get("grant_type") == "client_credentials" or not any(claims.get(k) for k in ("user_name", "email", "user_id"))
        name = claims.get("email") or claims.get("user_name") or claims.get("client_id") or claims.get("cid") or claims.get("sub") or "unknown"
        return Principal(subject=str(claims.get("sub") or name), name=str(name), scopes=scopes, authenticated=True, is_user=not technical)


def build_authenticator_from_env(env: Mapping[str, str] | None = None) -> Authenticator:
    env = os.environ if env is None else env
    mode = (env.get("AUTH_MODE") or "").strip().lower()
    binding = btp.find_binding(label="xsuaa", env=env)
    if not mode:
        mode = "jwt" if binding is not None else "none"
    if mode == "none":
        return NoAuthenticator()
    if mode != "jwt":
        raise ValueError(f"AUTH_MODE must be none or jwt, not {mode!r}")

    settings: dict[str, Any] = xsuaa.jwt_settings(binding.credentials) if binding is not None else {}
    if env.get("AUTH_JWKS_URL"):
        settings["jwks_url"] = env["AUTH_JWKS_URL"]
    if env.get("AUTH_ISSUER"):
        settings["issuer"] = env["AUTH_ISSUER"]
    if env.get("AUTH_AUDIENCE"):
        settings["audience"] = [a.strip() for a in env["AUTH_AUDIENCE"].split(",") if a.strip()]
    if "AUTH_SCOPE_PREFIX" in env:
        settings["scope_prefix"] = env["AUTH_SCOPE_PREFIX"]
    missing = [k for k in ("jwks_url", "issuer", "audience") if not settings.get(k)]
    if missing:
        raise ValueError("AUTH_MODE=jwt needs an XSUAA binding or AUTH_JWKS_URL, AUTH_ISSUER and AUTH_AUDIENCE; missing: " + ", ".join(missing))
    return JwtAuthenticator(**settings)


# --------------------------------------------------------------------------- #
# FastAPI wiring
# --------------------------------------------------------------------------- #
def current_principal(request: Request) -> Principal:
    """The caller. Resolved once per request (FastAPI caches a dependency), however many endpoints' checks ask."""
    authenticator: Authenticator = request.app.state.authenticator
    try:
        return authenticator.authenticate(request.headers.get("authorization"))
    except AuthenticationError as exc:
        metrics.inc("auth_failures_total", {"code": exc.code})
        raise ApiError(exc.status, exc.code, str(exc), headers=_BEARER if exc.status == 401 else None) from None


def require(scope: str) -> Callable[..., Principal]:
    """A dependency that admits only a caller holding `scope`, and hands the endpoint the `Principal`."""
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}")

    def dependency(principal: Principal = Depends(current_principal)) -> Principal:
        if not principal.can(scope):
            metrics.inc("auth_failures_total", {"code": "FORBIDDEN"})
            logger.warning("%s lacks scope %s", principal.name, scope, extra={"event": "auth_forbidden", "scope": scope})
            raise ApiError(403, "FORBIDDEN", f"this needs the '{scope}' scope")
        return principal

    return dependency


def decided_by(principal: Principal, claimed: str) -> str:
    """The name an approval is recorded under.

    * a person's token: theirs, whatever the request says;
    * a technical client's token (a workflow calling back after a human decided in SAP Build Process Automation):
      the name it reports, tagged with the client that vouched for it — grant `approve` only to a client you trust to do so;
    * no authentication (mode `none`): the name in the request, as before."""
    if not principal.authenticated:
        return claimed
    name = principal.name if principal.is_user else f"{claimed} (via {principal.name})"
    return name[:ACTOR_MAX_LENGTH]  # the audit trail's actor column has a length; a database that enforces it would fail the approval itself
