"""OAuth 2.0 client-credentials tokens — how an application authenticates *itself* to a SAP BTP service.

XSUAA (and SAP Cloud Identity Services) issue a token for a service key's client id and secret; SAP Integration Suite,
Advanced Event Mesh, AI Core and the Destination service all accept it as a bearer token. The token is cached until
shortly before it expires, so a burst of calls costs one token request, and forgotten on `invalidate()` (a 401 says
the cached one is no longer good).

Standard library only, in the same style as the Gemini client: no new runtime dependency, the secret sent in a header
and never in a URL, and no error message that could contain it.
"""
from __future__ import annotations

import base64
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

LOOPBACK = {"localhost", "127.0.0.1", "::1"}


class TokenError(RuntimeError):
    """A token could not be obtained. The message never contains the client secret or a token."""


def require_https(url: str, what: str) -> None:
    """Credentials and business data go over TLS; plain http is allowed only to this machine (tests, a local mock)."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme == "https" or (parts.scheme == "http" and (parts.hostname or "") in LOOPBACK):
        return
    raise ValueError(f"{what} must be an https URL (http is allowed only for localhost), got scheme {parts.scheme!r}")


class ClientCredentialsTokenProvider:
    def __init__(
        self, token_url: str, client_id: str, client_secret: str, *, scope: str | None = None, timeout_seconds: float = 10.0,
        early_expiry_seconds: float = 60.0, opener: Callable[..., Any] = urllib.request.urlopen, clock: Callable[[], float] = time.monotonic,
    ):
        require_https(token_url, "the OAuth token URL")
        if not client_id or not client_secret:
            raise ValueError("an OAuth client id and secret are both required")
        self._url, self._client_id, self._secret, self._scope = token_url, client_id, client_secret, scope
        self._timeout, self._early = timeout_seconds, early_expiry_seconds
        self._opener, self._clock = opener, clock
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at = 0.0

    def token(self) -> str:
        with self._lock:
            if self._token is None or self._clock() >= self._expires_at:
                self._token, self._expires_at = self._fetch()
            return self._token

    def invalidate(self) -> None:
        with self._lock:
            self._token = None

    def authorization_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token()}"}

    def _fetch(self) -> tuple[str, float]:
        form = {"grant_type": "client_credentials"}
        if self._scope:
            form["scope"] = self._scope
        basic = base64.b64encode(f"{self._client_id}:{self._secret}".encode()).decode()
        request = urllib.request.Request(
            self._url, data=urllib.parse.urlencode(form).encode(), method="POST",
            headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        )
        try:
            with self._opener(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise TokenError(f"the token endpoint refused the client credentials (HTTP {exc.code})") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TokenError(f"the token endpoint could not be reached ({type(exc).__name__})") from None
        except (ValueError, UnicodeDecodeError):
            raise TokenError("the token endpoint answered with something that is not JSON") from None
        token, lifetime = payload.get("access_token"), payload.get("expires_in")
        if not isinstance(token, str) or not token:
            raise TokenError("the token endpoint's answer has no access_token")
        seconds = float(lifetime) if isinstance(lifetime, (int, float)) and lifetime > 0 else 300.0  # a missing lifetime: short, not forever
        return token, self._clock() + max(seconds - self._early, 0.0)
