"""Built-in Approval Portal — replaces SAP BTP (SBPA + Integration Suite) entirely.

What SAP BTP did → What this module does instead
─────────────────────────────────────────────────
Integration Suite iFlow  → Receives the escalation event internally (no HTTP hop needed)
SBPA Process Instance    → Stores a pending task in an in-memory queue + SQLite
SBPA Human Task / Inbox  → Serves a beautiful HTML approval page at /approval
SBPA Callback            → FastAPI route POST /approval/{id}/decide
Email notification       → Sends via Python smtplib (Gmail/Outlook) or logs if unconfigured

Configuration (all optional — works with zero env vars):
─────────────────────────────────────────────────────────
  APPROVAL_NOTIFY_EMAIL      Recipient email for "plan needs approval" notifications
  APPROVAL_SMTP_HOST         SMTP server host  (default: smtp.gmail.com)
  APPROVAL_SMTP_PORT         SMTP port         (default: 587)
  APPROVAL_SMTP_USER         SMTP username / Gmail address
  APPROVAL_SMTP_PASSWORD     SMTP password / Gmail App Password
  APPROVAL_BASE_URL          Public base URL of this backend (for links in emails)
                             (default: http://localhost:8000)
"""
from __future__ import annotations

import logging
import os
import smtplib
import threading
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

logger = logging.getLogger("resilientsc.approval_portal")


# ---------------------------------------------------------------------------
# In-process task store (replaces SBPA process instances)
# ---------------------------------------------------------------------------

class ApprovalTask:
    """One pending human-approval task, mirroring an SBPA process instance."""

    def __init__(self, simulation_id: str, data: dict):
        self.simulation_id = simulation_id
        self.data = data                        # full event payload
        self.created_at = datetime.now(timezone.utc)
        self.status = "PENDING"                 # PENDING | APPROVED | REJECTED
        self.decided_by: str | None = None
        self.decided_at: datetime | None = None
        self.note: str = ""


class ApprovalTaskStore:
    """Thread-safe in-memory store for pending approval tasks."""

    def __init__(self):
        self._tasks: dict[str, ApprovalTask] = {}
        self._lock = threading.Lock()

    def create(self, simulation_id: str, data: dict) -> ApprovalTask:
        task = ApprovalTask(simulation_id, data)
        with self._lock:
            self._tasks[simulation_id] = task
        return task

    def get(self, simulation_id: str) -> ApprovalTask | None:
        with self._lock:
            return self._tasks.get(simulation_id)

    def all_pending(self) -> list[ApprovalTask]:
        with self._lock:
            return [t for t in self._tasks.values() if t.status == "PENDING"]

    def all_tasks(self) -> list[ApprovalTask]:
        with self._lock:
            return list(self._tasks.values())

    def decide(self, simulation_id: str, decision: str, decided_by: str, note: str = "") -> ApprovalTask | None:
        with self._lock:
            task = self._tasks.get(simulation_id)
            if task and task.status == "PENDING":
                task.status = decision.upper()
                task.decided_by = decided_by
                task.decided_at = datetime.now(timezone.utc)
                task.note = note
            return task


# Global singleton store (used by the FastAPI router)
_store = ApprovalTaskStore()


def get_store() -> ApprovalTaskStore:
    return _store


# ---------------------------------------------------------------------------
# Email notifier (replaces SBPA sending the task to the user's Inbox)
# ---------------------------------------------------------------------------

def _send_email_notification(task: ApprovalTask) -> None:
    """Send an HTML email with Approve / Reject links."""
    recipient = os.environ.get("APPROVAL_NOTIFY_EMAIL", "").strip()
    if not recipient:
        logger.info(
            "approval_portal: no APPROVAL_NOTIFY_EMAIL set — skipping email for simulation %s. "
            "Open http://localhost:8000/approval to review.",
            task.simulation_id,
        )
        return

    smtp_host = os.environ.get("APPROVAL_SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("APPROVAL_SMTP_PORT", "587"))
    smtp_user = os.environ.get("APPROVAL_SMTP_USER", "").strip()
    smtp_pass = os.environ.get("APPROVAL_SMTP_PASSWORD", "").strip()
    base_url = os.environ.get("APPROVAL_BASE_URL", "http://localhost:8000").rstrip("/")

    if not smtp_user or not smtp_pass:
        logger.warning(
            "approval_portal: APPROVAL_SMTP_USER/PASSWORD not set — cannot send email for %s. "
            "Visit %s/approval to approve manually.",
            task.simulation_id, base_url,
        )
        return

    data = task.data
    plan = data.get("plan") or {}
    compliance = data.get("compliance") or {}
    disruptions = data.get("disruptions") or []
    disruption = disruptions[0] if disruptions else {}

    approve_url = f"{base_url}/approval/{task.simulation_id}/decide?decision=APPROVE&decided_by={smtp_user}"
    reject_url  = f"{base_url}/approval/{task.simulation_id}/decide?decision=REJECT&decided_by={smtp_user}"
    portal_url  = f"{base_url}/approval"

    html = f"""
    <html><body style="font-family:Arial,sans-serif;max-width:600px;margin:auto;padding:20px">
    <h2 style="color:#1a56db">⚠️ Supply Chain Plan Needs Your Approval</h2>
    <table style="width:100%;border-collapse:collapse;margin:16px 0">
      <tr style="background:#f3f4f6"><td style="padding:8px;font-weight:bold">Simulation ID</td>
          <td style="padding:8px">{task.simulation_id}</td></tr>
      <tr><td style="padding:8px;font-weight:bold">Product</td>
          <td style="padding:8px">{plan.get("product_id","—")}</td></tr>
      <tr style="background:#f3f4f6"><td style="padding:8px;font-weight:bold">Total Cost</td>
          <td style="padding:8px"><strong>{plan.get("objective_value","—")}</strong> cost units</td></tr>
      <tr><td style="padding:8px;font-weight:bold">Compliance Reason</td>
          <td style="padding:8px;color:#dc2626">{compliance.get("reason","—")}</td></tr>
      <tr style="background:#f3f4f6"><td style="padding:8px;font-weight:bold">Disruption</td>
          <td style="padding:8px">{disruption.get("event_type","—")} @ {disruption.get("location","—")}
          (severity: {disruption.get("severity","—")})</td></tr>
      <tr><td style="padding:8px;font-weight:bold">Affected Routes</td>
          <td style="padding:8px">{", ".join(disruption.get("affected_routes",[]) or ["none"])}</td></tr>
      <tr style="background:#f3f4f6"><td style="padding:8px;font-weight:bold">Affected Suppliers</td>
          <td style="padding:8px">{", ".join(disruption.get("affected_suppliers",[]) or ["none"])}</td></tr>
    </table>
    <div style="margin:24px 0;display:flex;gap:12px">
      <a href="{approve_url}" style="background:#16a34a;color:white;padding:12px 28px;text-decoration:none;border-radius:6px;font-weight:bold">
        ✅ APPROVE PLAN</a>
      &nbsp;&nbsp;
      <a href="{reject_url}" style="background:#dc2626;color:white;padding:12px 28px;text-decoration:none;border-radius:6px;font-weight:bold">
        ❌ REJECT PLAN</a>
    </div>
    <p style="color:#6b7280;font-size:13px">Or review all pending approvals at:
      <a href="{portal_url}">{portal_url}</a></p>
    </body></html>
    """

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[ResilientSC] Approve Plan — {task.simulation_id}"
        msg["From"] = smtp_user
        msg["To"] = recipient
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, [recipient], msg.as_string())

        logger.info("approval_portal: email sent to %s for simulation %s", recipient, task.simulation_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("approval_portal: email failed — %s. Visit %s/approval", exc, base_url)


# ---------------------------------------------------------------------------
# EventPublisher implementation (drop-in for SBPA webhook publisher)
# ---------------------------------------------------------------------------

class ApprovalPortalPublisher:
    """Replaces SBPAWebhookPublisher. Receives the CloudEvent internally,
    creates a task in the store and fires the email notification."""

    TRIGGER_EVENT = "com.resilientsc.plan.approval.requested"

    def publish(self, event) -> None:
        if event.type != self.TRIGGER_EVENT:
            return
        task = _store.create(event.subject, event.data)
        logger.info(
            "approval_portal: task created for simulation %s — visit /approval",
            event.subject,
        )
        # Send email in a daemon thread so it never blocks the pipeline
        t = threading.Thread(target=_send_email_notification, args=(task,), daemon=True)
        t.start()

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# FastAPI router — the Approval Portal UI and API
# ---------------------------------------------------------------------------

def make_approval_router():
    """Returns a FastAPI router that:
      GET  /approval                  → HTML portal listing all pending tasks
      GET  /approval/{id}             → HTML detail page for one simulation
      GET  /approval/{id}/decide      → Quick-decide via URL (from email links)
      POST /approval/{id}/decide      → JSON API (for programmatic use)
    """
    from fastapi import APIRouter, Depends, Form, Query, Request
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
    from backend.api.context import AppContext, get_ctx

    router = APIRouter(prefix="/approval", tags=["approval-portal"])

    # ── shared HTML shell ──────────────────────────────────────────────────

    def _html(title: str, body: str) -> HTMLResponse:
        return HTMLResponse(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title} — ResilientSC</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:'Segoe UI',Arial,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}}
    header{{background:linear-gradient(135deg,#1e3a8a,#1a56db);padding:20px 40px;display:flex;align-items:center;gap:16px}}
    header h1{{font-size:1.5rem;font-weight:700;color:white}}
    header span{{background:rgba(255,255,255,.15);padding:4px 12px;border-radius:20px;font-size:.8rem;color:#bfdbfe}}
    .container{{max-width:1000px;margin:32px auto;padding:0 24px}}
    .card{{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:24px;margin-bottom:20px}}
    .card h2{{font-size:1.1rem;font-weight:600;margin-bottom:16px;color:#93c5fd}}
    table{{width:100%;border-collapse:collapse}}
    th{{background:#0f172a;padding:10px 14px;text-align:left;font-size:.8rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}}
    td{{padding:10px 14px;border-top:1px solid #1e293b;font-size:.9rem}}
    tr:hover td{{background:#0f172a}}
    .badge{{display:inline-block;padding:3px 10px;border-radius:20px;font-size:.75rem;font-weight:600}}
    .PENDING{{background:#92400e;color:#fde68a}}
    .APPROVED{{background:#14532d;color:#86efac}}
    .REJECTED{{background:#7f1d1d;color:#fca5a5}}
    .btn{{display:inline-block;padding:10px 22px;border-radius:7px;font-weight:600;font-size:.9rem;text-decoration:none;cursor:pointer;border:none}}
    .btn-approve{{background:#16a34a;color:white}}
    .btn-reject{{background:#dc2626;color:white}}
    .btn-back{{background:#334155;color:#cbd5e1}}
    .btn:hover{{opacity:.88}}
    .info-row{{display:flex;gap:8px;margin-bottom:8px;font-size:.9rem}}
    .info-label{{color:#94a3b8;min-width:160px;flex-shrink:0}}
    .info-value{{color:#e2e8f0;font-weight:500}}
    .empty{{text-align:center;padding:48px;color:#64748b}}
    form.decide{{display:inline}}
    input.note{{background:#0f172a;border:1px solid #334155;color:#e2e8f0;padding:8px 12px;border-radius:6px;width:100%;margin:12px 0;font-size:.9rem}}
    .actions{{display:flex;gap:12px;margin-top:8px;flex-wrap:wrap;align-items:center}}
  </style>
</head>
<body>
  <header>
    <h1>🔗 ResilientSC Approval Portal</h1>
    <span>Replaces SAP Build Process Automation</span>
  </header>
  <div class="container">{body}</div>
</body>
</html>""")

    # ── GET /approval — list all tasks ────────────────────────────────────

    @router.get("", response_class=HTMLResponse, summary="Approval portal inbox")
    def approval_inbox():
        tasks = _store.all_tasks()
        pending = [t for t in tasks if t.status == "PENDING"]
        done = [t for t in tasks if t.status != "PENDING"]

        def task_row(t: ApprovalTask) -> str:
            plan = t.data.get("plan") or {}
            dis = (t.data.get("disruptions") or [{}])[0]
            return f"""<tr>
              <td><a href="/approval/{t.simulation_id}" style="color:#60a5fa">{t.simulation_id}</a></td>
              <td>{plan.get("product_id","—")}</td>
              <td>{dis.get("event_type","—")} @ {dis.get("location","—")}</td>
              <td>{plan.get("objective_value","—")}</td>
              <td><span class="badge {t.status}">{t.status}</span></td>
              <td>{t.created_at.strftime("%Y-%m-%d %H:%M")}</td>
              <td>{t.decided_by or "—"}</td>
            </tr>"""

        pending_rows = "".join(task_row(t) for t in pending) if pending else \
            '<tr><td colspan="7" class="empty">No pending approvals 🎉</td></tr>'
        done_rows = "".join(task_row(t) for t in done) if done else \
            '<tr><td colspan="7" class="empty">No completed approvals yet</td></tr>'

        cols = "<th>Simulation</th><th>Product</th><th>Disruption</th><th>Cost</th><th>Status</th><th>Created</th><th>Decided By</th>"
        body = f"""
          <div class="card">
            <h2>⏳ Pending Approvals ({len(pending)})</h2>
            <table><thead><tr>{cols}</tr></thead><tbody>{pending_rows}</tbody></table>
          </div>
          <div class="card">
            <h2>✅ Completed ({len(done)})</h2>
            <table><thead><tr>{cols}</tr></thead><tbody>{done_rows}</tbody></table>
          </div>"""
        return _html("Approval Inbox", body)

    # ── GET /approval/{id} — detail page ──────────────────────────────────

    @router.get("/{simulation_id}", response_class=HTMLResponse, summary="Approval detail")
    def approval_detail(simulation_id: str):
        task = _store.get(simulation_id)
        if not task:
            return _html("Not Found", f'<div class="card"><p class="empty">No approval task found for <code>{simulation_id}</code></p></div>')

        data = task.data
        plan = data.get("plan") or {}
        compliance = data.get("compliance") or {}
        disruptions = data.get("disruptions") or []
        dis = disruptions[0] if disruptions else {}
        allocs = plan.get("allocations") or []

        alloc_rows = "".join(
            f"<tr><td>{a.get('supplier_id','—')}</td><td>{a.get('quantity','—')}</td>"
            f"<td>{a.get('route_id','—')}</td><td>{a.get('transport_mode','—')}</td>"
            f"<td>{a.get('landed_unit_cost','—')}</td><td>{a.get('arrival_days','—')}</td></tr>"
            for a in allocs
        ) or '<tr><td colspan="6" class="empty">No allocations</td></tr>'

        decide_form = "" if task.status != "PENDING" else f"""
          <div class="card">
            <h2>🗳️ Make Your Decision</h2>
            <input class="note" id="note" placeholder="Optional note / reason...">
            <div class="actions">
              <button class="btn btn-approve" onclick="decide('APPROVE')">✅ Approve Plan</button>
              <button class="btn btn-reject" onclick="decide('REJECT')">❌ Reject Plan</button>
              <a href="/approval" class="btn btn-back">← Back to Inbox</a>
            </div>
          </div>
          <script>
          async function decide(decision) {{
            const note = document.getElementById('note').value;
            const decided_by = prompt('Your name / email:') || 'approver';
            const r = await fetch('/approval/{simulation_id}/decide', {{
              method: 'POST',
              headers: {{'Content-Type': 'application/json'}},
              body: JSON.stringify({{decision, decided_by, note}})
            }});
            const d = await r.json();
            if (r.ok) {{ alert('Decision recorded: ' + d.status); location.reload(); }}
            else {{ alert('Error: ' + (d.detail || JSON.stringify(d))); }}
          }}
          </script>"""

        status_badge = f'<span class="badge {task.status}">{task.status}</span>'
        decided_info = f" by <strong>{task.decided_by}</strong> at {task.decided_at.strftime('%Y-%m-%d %H:%M')}" \
            if task.decided_by else ""

        body = f"""
          <a href="/approval" class="btn btn-back" style="margin-bottom:16px;display:inline-block">← Back to Inbox</a>
          <div class="card">
            <h2>📋 Plan Details — {simulation_id}</h2>
            <div class="info-row"><span class="info-label">Status</span><span class="info-value">{status_badge}{decided_info}</span></div>
            <div class="info-row"><span class="info-label">Product</span><span class="info-value">{plan.get("product_id","—")}</span></div>
            <div class="info-row"><span class="info-label">Total Cost</span><span class="info-value">{plan.get("objective_value","—")} cost units</span></div>
            <div class="info-row"><span class="info-label">Compliance Reason</span><span class="info-value" style="color:#fca5a5">{compliance.get("reason","—")}</span></div>
            {'<div class="info-row"><span class="info-label">Note</span><span class="info-value">' + task.note + "</span></div>" if task.note else ""}
          </div>
          <div class="card">
            <h2>🌍 Disruption</h2>
            <div class="info-row"><span class="info-label">Type</span><span class="info-value">{dis.get("event_type","—")}</span></div>
            <div class="info-row"><span class="info-label">Location</span><span class="info-value">{dis.get("location","—")}</span></div>
            <div class="info-row"><span class="info-label">Severity</span><span class="info-value">{dis.get("severity","—")}</span></div>
            <div class="info-row"><span class="info-label">Affected Routes</span><span class="info-value">{", ".join(dis.get("affected_routes",[]) or ["none"])}</span></div>
            <div class="info-row"><span class="info-label">Affected Suppliers</span><span class="info-value">{", ".join(dis.get("affected_suppliers",[]) or ["none"])}</span></div>
          </div>
          <div class="card">
            <h2>📦 Proposed Allocations</h2>
            <table>
              <thead><tr><th>Supplier</th><th>Quantity</th><th>Route</th><th>Mode</th><th>Unit Cost</th><th>Arrival (days)</th></tr></thead>
              <tbody>{alloc_rows}</tbody>
            </table>
          </div>
          {decide_form}"""
        return _html(f"Approve — {simulation_id}", body)

    # ── GET /approval/{id}/decide — email link handler ────────────────────

    @router.get("/{simulation_id}/decide", response_class=HTMLResponse, summary="Quick decide via email link")
    def quick_decide(
        simulation_id: str,
        decision: str = Query(...),
        decided_by: str = Query("approver"),
        ctx: AppContext = Depends(get_ctx),
    ):
        task = _store.get(simulation_id)
        if not task or task.status != "PENDING":
            return _html("Already decided", f'<div class="card"><p class="empty">Simulation <code>{simulation_id}</code> has already been decided or does not exist.</p></div>')
        return _do_decide(simulation_id, decision.upper(), decided_by, "", ctx)

    # ── POST /approval/{id}/decide — JSON API ─────────────────────────────

    @router.post("/{simulation_id}/decide", summary="Submit approval decision (JSON)")
    def post_decide(simulation_id: str, body: dict, ctx: AppContext = Depends(get_ctx)):
        decision = (body.get("decision") or "").upper()
        decided_by = body.get("decided_by") or "approver"
        note = body.get("note") or ""
        return _do_decide(simulation_id, decision, decided_by, note, ctx)

    # ── shared decision handler ───────────────────────────────────────────

    def _do_decide(simulation_id: str, decision: str, decided_by: str, note: str, ctx: AppContext):
        from fastapi import HTTPException
        if decision not in ("APPROVE", "REJECT"):
            raise HTTPException(status_code=400, detail=f"decision must be APPROVE or REJECT, got {decision!r}")

        # Update local task store
        task = _store.decide(simulation_id, decision, decided_by, note)
        if not task:
            raise HTTPException(status_code=404, detail=f"No pending approval task for {simulation_id!r}")

        # Forward decision to the orchestrator (same as /api/decisions/{id}/approve)
        try:
            if decision == "APPROVE":
                state = ctx.orchestrator.approve(simulation_id, decided_by, note)
            else:
                state = ctx.orchestrator.reject(simulation_id, decided_by, note)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        logger.info(
            "approval_portal: simulation %s %s by %s",
            simulation_id, decision, decided_by,
            extra={"event": "portal_decision", "simulation_id": simulation_id, "decision": decision},
        )

        result = {
            "simulation_id": state.simulation_id,
            "status": state.status.value,
            "approval_status": state.approval_status.value,
            "decided_by": decided_by,
            "decision": decision,
        }
        # If called from email link → show a nice HTML page
        if isinstance(result, dict) and "status" in result:
            icon = "✅" if decision == "APPROVE" else "❌"
            color = "#16a34a" if decision == "APPROVE" else "#dc2626"
            from fastapi.responses import HTMLResponse as HR
            return HR(_html("Decision Recorded", f"""
              <div class="card" style="text-align:center;padding:48px">
                <div style="font-size:3rem">{icon}</div>
                <h2 style="margin:16px 0;color:{color}">{decision}D</h2>
                <p>Simulation <strong>{simulation_id}</strong> is now <strong>{state.status.value}</strong>.</p>
                <br>
                <a href="/approval" class="btn btn-back">← Back to Inbox</a>
              </div>""").content)
        return result

    return router
