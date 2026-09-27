"""SAP Build Process Automation (SBPA) Bridge for ResilientSC.

This module provides the integration layer between ResilientSC's FastAPI backend
and SAP Build Process Automation.

Integration Architecture:
--------------------------

  ResilientSC Backend                    SAP BTP
  ─────────────────                      ───────
  StoreEventBridge                       Integration Suite
       │  CloudEvents                         │
       ▼                                      │
  SBPAWebhookPublisher ──── HTTP POST ──►  iFlow (trigger)
                                             │
                                             ▼
                                      SBPA Process Instance
                                       ├─ Automated Steps
                                       │   (call /api/simulations)
                                       └─ Human Task (Inbox)
                                           (approve/reject)
                                             │
                                             ▼
                               SBPA calls back:
                               POST /api/decisions/{id}/approve
                               POST /api/decisions/{id}/reject

What the SBPA process looks like:
----------------------------------

  [Start Event: plan.approval.requested]
        │
        ▼
  [Fetch Plan Details]     ← GET /api/decisions/{simulation_id}
  [Automated Task]         ← enrich context for approver
        │
        ▼
  [Human Task / Inbox]     ← approver sees: disruption, plan, cost, compliance
        │
        ├─ Approved ──► POST /api/decisions/{id}/approve
        │
        └─ Rejected ──► POST /api/decisions/{id}/reject
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import BaseModel

from backend.integration.events import DomainEvent, EventPublisher

logger = logging.getLogger("resilientsc.sbpa")


class SBPACallbackRequest(BaseModel):
    simulation_id: str
    decision: str          # "APPROVE" or "REJECT"
    decided_by: str
    note: str = ""
    expected_version: int | None = None


# ---------------------------------------------------------------------------
# SBPA Webhook Publisher
# ---------------------------------------------------------------------------

class SBPAWebhookPublisher(EventPublisher):
    """Publishes CloudEvents to a SAP Integration Suite iFlow webhook that
    triggers an SBPA process.

    Configure via environment variables:
        SBPA_WEBHOOK_URL        Full URL of the SAP Integration Suite iFlow endpoint
        SBPA_CLIENT_ID          OAuth2 client_id (from SBPA service binding)
        SBPA_CLIENT_SECRET      OAuth2 client_secret
        SBPA_TOKEN_URL          OAuth2 token URL  (e.g. https://<subaccount>.authentication.sap.hana.ondemand.com/oauth/token)
        SBPA_TRIGGER_EVENTS     Comma-separated list of event types to forward
                                Default: com.resilientsc.plan.approval.requested
    """

    DEFAULT_TRIGGER_EVENTS = {"com.resilientsc.plan.approval.requested"}

    def __init__(
        self,
        webhook_url: str,
        *,
        client_id: str | None = None,
        client_secret: str | None = None,
        token_url: str | None = None,
        trigger_events: set[str] | None = None,
        timeout_seconds: float = 10.0,
    ):
        self._webhook_url = webhook_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_url = token_url
        self._trigger_events = trigger_events or self.DEFAULT_TRIGGER_EVENTS
        self._timeout = timeout_seconds
        self._token_cache: dict[str, Any] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "SBPAWebhookPublisher | None":
        """Build from environment variables. Returns None if SBPA_WEBHOOK_URL is not set."""
        url = os.environ.get("SBPA_WEBHOOK_URL", "").strip()
        if not url:
            logger.info("SBPA_WEBHOOK_URL not set — SBPA integration disabled")
            return None
        raw_events = os.environ.get("SBPA_TRIGGER_EVENTS", "")
        trigger_events = (
            {e.strip() for e in raw_events.split(",") if e.strip()}
            if raw_events.strip()
            else None
        )
        return cls(
            webhook_url=url,
            client_id=os.environ.get("SBPA_CLIENT_ID"),
            client_secret=os.environ.get("SBPA_CLIENT_SECRET"),
            token_url=os.environ.get("SBPA_TOKEN_URL"),
            trigger_events=trigger_events,
        )

    def publish(self, event: DomainEvent) -> None:
        """Publish a CloudEvent to SBPA — only if its type is in trigger_events."""
        if event.type not in self._trigger_events:
            logger.debug("SBPA: skipping event type %s", event.type)
            return
        try:
            self._post_event(event)
            logger.info(
                "SBPA: published %s for simulation %s",
                event.type, event.subject,
                extra={"event": "sbpa_published", "event_type": event.type, "simulation_id": event.subject},
            )
        except Exception as exc:  # noqa: BLE001
            # Non-fatal: SBPA integration must not break the pipeline
            logger.error(
                "SBPA: failed to publish %s — %s",
                event.type, exc,
                extra={"event": "sbpa_publish_failed", "event_type": event.type, "error": str(exc)},
            )

    def _post_event(self, event: DomainEvent) -> None:
        payload = json.dumps({
            "specversion": "1.0",
            "id": event.id,
            "type": event.type,
            "source": event.source,
            "subject": event.subject,
            "time": event.time,
            "datacontenttype": "application/json",
            "data": event.data,
        }).encode()

        headers = {
            "Content-Type": "application/cloudevents+json",
            "ce-specversion": "1.0",
            "ce-id": event.id,
            "ce-type": event.type,
            "ce-source": event.source,
            "ce-subject": event.subject,
        }

        # OAuth2 client credentials — only if configured
        token = self._get_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        req = urllib.request.Request(
            self._webhook_url,
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"SBPA webhook returned HTTP {resp.status}")

    def _get_token(self) -> str | None:
        """Fetch an OAuth2 token using client credentials. Caches until 60s before expiry."""
        if not (self._token_url and self._client_id and self._client_secret):
            return None
        with self._lock:
            cached = self._token_cache
            if cached.get("expires_at", 0) > time.monotonic() + 60:
                return cached["access_token"]
            token_data = self._fetch_token()
            self._token_cache = {
                "access_token": token_data["access_token"],
                "expires_at": time.monotonic() + int(token_data.get("expires_in", 3600)),
            }
            return self._token_cache["access_token"]

    def _fetch_token(self) -> dict:
        import base64
        credentials = base64.b64encode(f"{self._client_id}:{self._client_secret}".encode()).decode()
        payload = b"grant_type=client_credentials"
        req = urllib.request.Request(
            self._token_url,
            data=payload,
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read())


# ---------------------------------------------------------------------------
# SBPA Callback Router (to add to FastAPI)
# ---------------------------------------------------------------------------

def make_sbpa_callback_router():
    """Returns a FastAPI router with a /api/sbpa/callback endpoint that SBPA
    calls after a human task completes in the SBPA Inbox.

    SBPA sends a POST with:
    {
        "simulation_id": "sim-123",
        "decision":      "APPROVE" | "REJECT",
        "decided_by":    "user@example.com",
        "note":          "optional reason",
        "expected_version": 3
    }

    Mount this router in main.py:
        from backend.sap.sbpa_bridge import make_sbpa_callback_router
        app.include_router(make_sbpa_callback_router())
    """
    from fastapi import APIRouter, Depends, HTTPException
    from backend.api.context import AppContext, get_ctx

    router = APIRouter(prefix="/api/sbpa", tags=["sbpa"])

    @router.post("/callback", summary="SBPA human-task callback")
    def sbpa_callback(body: SBPACallbackRequest, ctx: AppContext = Depends(get_ctx)):
        """Called by SAP Build Process Automation after the human approver
        completes the Inbox task. Forwards the decision to the orchestrator."""
        sim_id = body.simulation_id
        try:
            if body.decision.upper() == "APPROVE":
                state = ctx.orchestrator.approve(
                    sim_id, body.decided_by, body.note,
                    expected_version=body.expected_version,
                )
            elif body.decision.upper() == "REJECT":
                state = ctx.orchestrator.reject(
                    sim_id, body.decided_by, body.note,
                    expected_version=body.expected_version,
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unknown decision: {body.decision!r}")
        except Exception as exc:
            logger.error("SBPA callback failed for %s: %s", sim_id, exc)
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        logger.info(
            "SBPA callback: simulation %s %s by %s",
            sim_id, body.decision, body.decided_by,
            extra={"event": "sbpa_callback", "simulation_id": sim_id, "decision": body.decision},
        )
        return {
            "simulation_id": state.simulation_id,
            "status": state.status.value,
            "approval_status": state.approval_status.value,
            "decided_by": body.decided_by,
        }

    @router.get("/health", summary="SBPA integration health check")
    def sbpa_health():
        sbpa_url = os.environ.get("SBPA_WEBHOOK_URL", "")
        return {
            "sbpa_configured": bool(sbpa_url),
            "webhook_url": sbpa_url[:50] + "..." if len(sbpa_url) > 50 else sbpa_url or "(not set)",
            "trigger_events": list(SBPAWebhookPublisher.DEFAULT_TRIGGER_EVENTS),
        }

    return router
