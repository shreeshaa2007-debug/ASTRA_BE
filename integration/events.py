"""Domain events, and the publishers that carry them to SAP Integration Suite (or anything else that listens).

An event is a CloudEvents 1.0 structured-mode JSON document — the envelope SAP Integration Suite, Advanced Event Mesh
and SAP's own business events use, so an integration flow can route on `type` without knowing this application.

    EVENTS_BACKEND        none (default) | log | webhook
    EVENTS_WEBHOOK_URL    https endpoint; `{type}` and `{topic}` are filled in per event, e.g. an Advanced Event Mesh
                          REST delivery point .../topic/{topic}, or a Cloud Integration iFlow's HTTPS sender endpoint
    EVENTS_AUTH           none | basic | bearer | oauth2         (default: oauth2 when EVENTS_OAUTH_TOKEN_URL is set)
    EVENTS_BASIC_USER / EVENTS_BASIC_PASSWORD / EVENTS_BEARER_TOKEN
    EVENTS_OAUTH_TOKEN_URL / EVENTS_OAUTH_CLIENT_ID / EVENTS_OAUTH_CLIENT_SECRET / EVENTS_OAUTH_SCOPE
    EVENTS_TIMEOUT_SECONDS (10)   EVENTS_MAX_RETRIES (3)   EVENTS_QUEUE_SIZE (1000)

Delivery is at-least-once and never on the request path: `BackgroundPublisher` queues events and one worker sends them in
order, so a slow or dead endpoint cannot slow down or fail a commit. Every event's `id` is `<simulation_id>:<version>` —
unique per change and identical on a retry — so a receiver can de-duplicate. What this does not give: events survive a
process restart (the queue is in memory). The durable record is the checkpoint audit trail; see docs/sap-readiness.md.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from backend.monitoring.metrics import metrics
from backend.sap.oauth import ClientCredentialsTokenProvider, TokenError, require_https

logger = logging.getLogger("resilientsc.integration")

SPEC_VERSION = "1.0"
CONTENT_TYPE = "application/cloudevents+json; charset=utf-8"
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class DomainEvent:
    id: str
    type: str  # reverse-DNS, e.g. com.resilientsc.plan.finalized
    source: str  # who says so
    subject: str  # what it is about: the simulation id
    time: str  # ISO-8601 UTC
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"specversion": SPEC_VERSION, "id": self.id, "type": self.type, "source": self.source, "subject": self.subject,
                "time": self.time, "datacontenttype": "application/json", "data": self.data}


class EventDeliveryError(RuntimeError):
    """The event was not delivered. The message names the event and the reason, never a credential."""


class EventPublisher(Protocol):
    def publish(self, event: DomainEvent) -> None: ...
    def status(self) -> dict[str, Any]: ...
    def close(self, timeout: float = 5.0) -> None: ...


class NullPublisher:
    """The default: no integration configured, so nothing leaves the process."""

    def publish(self, event: DomainEvent) -> None:
        return None

    def status(self) -> dict[str, Any]:
        return {"backend": "none", "healthy": True}

    def close(self, timeout: float = 5.0) -> None:
        return None


class LoggingPublisher:
    """Writes each event to the log — for development, and as a way to see exactly what an integration flow would receive."""

    def publish(self, event: DomainEvent) -> None:
        logger.info("domain event %s (%s)", event.type, event.subject, extra={"event": "domain_event", "event_type": event.type, "event_id": event.id})

    def status(self) -> dict[str, Any]:
        return {"backend": "log", "healthy": True}

    def close(self, timeout: float = 5.0) -> None:
        return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A POST that is redirected to a sign-in page must not look delivered: 3xx is an error, not something to follow."""

    def redirect_request(self, *args, **kwargs):
        return None


def _default_opener(request, timeout):
    return urllib.request.build_opener(_NoRedirect).open(request, timeout=timeout)


def basic_auth(user: str, password: str) -> Callable[[], dict[str, str]]:
    header = {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}
    return lambda: header


def bearer_auth(token: str) -> Callable[[], dict[str, str]]:
    header = {"Authorization": f"Bearer {token}"}
    return lambda: header


class WebhookPublisher:
    def __init__(
        self, url: str, *, auth: Callable[[], dict[str, str]] | None = None, on_unauthorized: Callable[[], None] | None = None,
        timeout_seconds: float = 10.0, max_retries: int = 3, backoff_seconds: float = 1.0,
        opener: Callable[..., Any] = _default_opener, sleep: Callable[[float], None] = time.sleep,
    ):
        require_https(url, "the events webhook URL")
        self._url = url
        self._auth = auth or (lambda: {})
        self._on_unauthorized = on_unauthorized
        self._timeout, self._max_retries, self._backoff = timeout_seconds, max_retries, backoff_seconds
        self._opener, self._sleep = opener, sleep

    def _url_for(self, event: DomainEvent) -> str:
        topic = event.type.replace(".", "/")
        return self._url.replace("{type}", urllib.parse.quote(event.type, safe="")).replace("{topic}", urllib.parse.quote(topic, safe="/"))

    def publish(self, event: DomainEvent) -> None:
        body = json.dumps(event.to_dict(), separators=(",", ":")).encode("utf-8")
        url = self._url_for(event)
        reason, refreshed = "no attempt made", False
        for attempt in range(self._max_retries + 1):
            retry = True
            try:
                request = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": CONTENT_TYPE, "Accept": "*/*", **self._auth()})
                with self._opener(request, timeout=self._timeout) as response:
                    status = getattr(response, "status", 200)
                if 200 <= status < 300:
                    return
                reason, retry = f"HTTP {status}", status in _RETRYABLE_STATUS
            except urllib.error.HTTPError as exc:
                exc.close()
                if exc.code == 401 and self._on_unauthorized is not None and not refreshed:
                    refreshed = True  # the cached token may simply have expired early: get a new one once, straight away
                    self._on_unauthorized()
                    reason = "HTTP 401"
                    continue
                reason, retry = f"HTTP {exc.code}", exc.code in _RETRYABLE_STATUS
            except TokenError as exc:
                reason = str(exc)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                reason = f"{type(exc).__name__}"
            if not retry:
                break
            if attempt < self._max_retries:
                self._sleep(self._backoff * (2 ** attempt))
        raise EventDeliveryError(f"event {event.type} for {event.subject} was not delivered: {reason}")

    def status(self) -> dict[str, Any]:
        parts = urllib.parse.urlsplit(self._url)
        return {"backend": "webhook", "target": f"{parts.scheme}://{parts.hostname}{parts.path}", "healthy": True}

    def close(self, timeout: float = 5.0) -> None:
        return None


class BackgroundPublisher:
    """Queue in front of another publisher, drained in order by one daemon thread. `publish` never blocks and never
    raises: a full queue drops the event (counted, logged), and a delivery that fails is logged and counted."""

    _STOP = object()

    def __init__(self, inner: EventPublisher, *, queue_size: int = 1000, unhealthy_after: int = 3):
        self._inner = inner
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._unhealthy_after = unhealthy_after
        self._lock = threading.Lock()
        self._published = self._failed = self._dropped = self._consecutive_failures = 0
        self._last_error: str | None = None
        self._worker = threading.Thread(target=self._run, name="event-publisher", daemon=True)
        self._worker.start()

    def publish(self, event: DomainEvent) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            with self._lock:
                self._dropped += 1
            metrics.inc("events_dropped_total", {"type": event.type})
            logger.error("event queue is full; dropped %s for %s", event.type, event.subject, extra={"event": "event_dropped", "event_type": event.type})

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                self._deliver(item)
            finally:
                self._queue.task_done()

    def _deliver(self, event: DomainEvent) -> None:
        started = time.perf_counter()
        try:
            self._inner.publish(event)
        except Exception as exc:  # noqa: BLE001 — a delivery problem is recorded, never allowed to stop the worker
            with self._lock:
                self._failed += 1
                self._consecutive_failures += 1
                self._last_error = str(exc)[:300]
            metrics.inc("events_failed_total", {"type": event.type})
            logger.error("%s", exc, extra={"event": "event_delivery_failed", "event_type": event.type, "event_id": event.id})
            return
        with self._lock:
            self._published += 1
            self._consecutive_failures = 0
        metrics.inc("events_published_total", {"type": event.type})
        metrics.observe("event_delivery_ms", round((time.perf_counter() - started) * 1000, 1), {"type": event.type})

    def flush(self, timeout: float = 5.0) -> bool:
        """Waits until everything queued so far has been attempted; False if it took longer than `timeout`."""
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.005)
        return not self._queue.unfinished_tasks

    def status(self) -> dict[str, Any]:
        with self._lock:
            healthy = self._consecutive_failures < self._unhealthy_after and self._dropped == 0
            return {**self._inner.status(), "queued": self._queue.qsize(), "published": self._published, "failed": self._failed,
                    "dropped": self._dropped, "consecutive_failures": self._consecutive_failures, "last_error": self._last_error, "healthy": healthy}

    def close(self, timeout: float = 5.0) -> None:
        """Tries to send what is queued (for up to `timeout`), then stops the worker."""
        self.flush(timeout)
        self._queue.put(self._STOP)
        self._worker.join(timeout)
        self._inner.close(timeout)


def build_publisher_from_env(env: Mapping[str, str] | None = None) -> EventPublisher:
    env = os.environ if env is None else env
    backend = (env.get("EVENTS_BACKEND") or "none").strip().lower()
    if backend == "none":
        return NullPublisher()
    if backend == "log":
        return LoggingPublisher()
    if backend != "webhook":
        raise ValueError(f"EVENTS_BACKEND must be none, log or webhook, not {backend!r}")

    url = env.get("EVENTS_WEBHOOK_URL", "").strip()
    if not url:
        raise ValueError("EVENTS_BACKEND=webhook needs EVENTS_WEBHOOK_URL")
    auth_mode = (env.get("EVENTS_AUTH") or ("oauth2" if env.get("EVENTS_OAUTH_TOKEN_URL") else "none")).strip().lower()
    auth: Callable[[], dict[str, str]] | None = None
    on_unauthorized: Callable[[], None] | None = None
    if auth_mode == "basic":
        if not env.get("EVENTS_BASIC_USER") or not env.get("EVENTS_BASIC_PASSWORD"):
            raise ValueError("EVENTS_AUTH=basic needs EVENTS_BASIC_USER and EVENTS_BASIC_PASSWORD")
        auth = basic_auth(env["EVENTS_BASIC_USER"], env["EVENTS_BASIC_PASSWORD"])
    elif auth_mode == "bearer":
        if not env.get("EVENTS_BEARER_TOKEN"):
            raise ValueError("EVENTS_AUTH=bearer needs EVENTS_BEARER_TOKEN")
        auth = bearer_auth(env["EVENTS_BEARER_TOKEN"])
    elif auth_mode == "oauth2":
        missing = [k for k in ("EVENTS_OAUTH_TOKEN_URL", "EVENTS_OAUTH_CLIENT_ID", "EVENTS_OAUTH_CLIENT_SECRET") if not env.get(k)]
        if missing:
            raise ValueError(f"EVENTS_AUTH=oauth2 needs {', '.join(missing)}")
        tokens = ClientCredentialsTokenProvider(env["EVENTS_OAUTH_TOKEN_URL"], env["EVENTS_OAUTH_CLIENT_ID"], env["EVENTS_OAUTH_CLIENT_SECRET"],
                                                scope=env.get("EVENTS_OAUTH_SCOPE") or None)
        auth, on_unauthorized = tokens.authorization_header, tokens.invalidate
    elif auth_mode != "none":
        raise ValueError(f"EVENTS_AUTH must be none, basic, bearer or oauth2, not {auth_mode!r}")

    webhook = WebhookPublisher(
        url, auth=auth, on_unauthorized=on_unauthorized,
        timeout_seconds=float(env.get("EVENTS_TIMEOUT_SECONDS") or 10), max_retries=int(env.get("EVENTS_MAX_RETRIES") or 3),
    )
    return BackgroundPublisher(webhook, queue_size=int(env.get("EVENTS_QUEUE_SIZE") or 1000))
