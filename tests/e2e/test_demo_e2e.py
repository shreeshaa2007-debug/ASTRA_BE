"""Phase 20 — the demo itself, so it cannot rot.

`scripts/demo.py` is the one command a presenter runs. This launches it exactly that way (on scratch ports and a
scratch database), checks what it prints and what it started, walks the whole golden path of docs/demo-script.md in
a real browser (scripts/capture_demo.mjs raises the moment any step of the script no longer works), then stops it and
proves that nothing is left running. If a change breaks the demo, this is where you find out — not on stage.
"""
from __future__ import annotations

import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time

import httpx
import pytest

from backend.tests.e2e.conftest import REPO, find_chrome, free_port, requires_browser, requires_built_data

pytestmark = [pytest.mark.e2e, pytest.mark.browser, requires_built_data, requires_browser]

DEMO = REPO / "scripts" / "demo.py"
CAPTURE = REPO / "scripts" / "capture_demo.mjs"


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def run_demo(*args: str, **kwargs) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "NO_COLOR": "1"}
    return subprocess.run([sys.executable, str(DEMO), *args], cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180, **kwargs)


def test_the_preflight_passes_on_a_ready_checkout_and_says_what_it_checked():
    done = run_demo("--check", "--api-port", str(free_port()), "--web-port", str(free_port()))
    assert done.returncode == 0, done.stdout + done.stderr
    for name in ("python packages", "datasets", "forecast model", "node", "frontend deps", "api port", "web port", "llm key"):
        assert name in done.stdout, name
    assert "Ready to run." in done.stdout and "FAIL" not in done.stdout


def test_the_preflight_refuses_to_start_when_a_port_is_taken_and_names_it():
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        done = run_demo("--check", "--api-port", str(port), "--web-port", str(free_port()))
    assert done.returncode == 1 and "FAIL" in done.stdout and f"{port} is in use" in done.stdout and "--api-port" in done.stdout


def test_one_command_starts_the_demo_the_golden_path_works_and_ctrl_c_leaves_nothing_running(tmp_path):
    api_port, web_port = free_port(), free_port()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "NO_COLOR": "1"}
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen(
        [sys.executable, str(DEMO), "--fresh", "--no-open", "--api-port", str(api_port), "--web-port", str(web_port), "--db", str(tmp_path / "demo.db")],
        cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", creationflags=flags)
    lines: queue.Queue[str] = queue.Queue()
    threading.Thread(target=lambda: [lines.put(ln.rstrip("\n")) for ln in proc.stdout], daemon=True).start()
    output: list[str] = []
    ready = None
    try:
        deadline = time.time() + 240
        while time.time() < deadline and ready is None:
            try:
                line = lines.get(timeout=1)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue
            output.append(line)
            if line.startswith("DEMO_READY "):
                ready = json.loads(line[len("DEMO_READY "):])
        assert ready, "the demo did not report ready:\n" + "\n".join(output[-30:])

        # what it printed and what it started
        text = "\n".join(output)
        assert "The demo is running." in text and f"http://127.0.0.1:{web_port}" in text and "seeded" in text
        assert ready["api"] == f"http://127.0.0.1:{api_port}" and ready["seeded"]["status"] == "COMPLETED"
        assert httpx.get(f"{ready['api']}/api/ready", timeout=10).status_code == 200
        assert httpx.get(ready["web"], timeout=10).status_code == 200
        sims = httpx.get(f"{ready['api']}/api/simulations", timeout=10).json()["data"]
        assert [s["simulation_id"] for s in sims] == [ready["seeded"]["simulation_id"]]  # --fresh: only the seeded run, nothing from earlier

        # every step of docs/demo-script.md, in a real browser; the script raises the moment one no longer works
        out = tmp_path / "shots"
        env_node = {**os.environ, "CHROME_PATH": find_chrome() or ""}
        captured = subprocess.run(["node", str(CAPTURE), "--web", ready["web"], "--api", ready["api"], "--out", str(out)], cwd=REPO, env=env_node,
                                  capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
        assert captured.returncode == 0, captured.stdout[-2000:] + captured.stderr[-2000:]
        shots = sorted(p.name for p in out.glob("*.png"))
        assert len(shots) == 13 and shots[0] == "01-dashboard.png" and "13-agent-monitor.png" in shots and "FAILED.png" not in shots
        assert "console problems" not in captured.stdout, captured.stdout[-1500:]  # a clean console all the way through

        # the human approval in the script really happened, under the name the script types
        approved = [s for s in httpx.get(f"{ready['api']}/api/simulations", timeout=10).json()["data"] if s["scenario_type"] == "SUPPLIER_FAILURE"]
        assert len(approved) == 1 and approved[0]["status"] == "COMPLETED"
        timeline = httpx.get(f"{ready['api']}/api/simulations/{approved[0]['simulation_id']}/status", timeout=10).json()["data"]["timeline"]
        assert timeline[-1]["checkpoint"] == "plan_finalized" and timeline[-1]["actor"] == "Demo Presenter"
    finally:
        # Ctrl+C: what a presenter does when the demo is over
        if proc.poll() is None:
            proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
            try:
                proc.wait(45)
            except subprocess.TimeoutExpired:
                proc.kill()
                pytest.fail("the launcher did not exit within 45s of Ctrl+C:\n" + "\n".join(output[-15:]))

    time.sleep(1)
    assert not port_open(api_port) and not port_open(web_port), "the backend or the UI kept running after Ctrl+C"
    assert proc.returncode is not None
