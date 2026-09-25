"""End-to-end fixtures: real processes on real sockets.

  Server     the E2E app (backend/tests/e2e/app.py) under uvicorn, on its own SQLite file — can be
             stopped, killed (no graceful shutdown) and restarted on the same file and port.
  WebApp     the frontend's Vite dev server, pointed at a Server.
  run_browser  runs a script in headless Chrome over the DevTools protocol (browser/runner.mjs).

Everything skips cleanly when its prerequisite is missing: the built data, node + `npm install`, Chrome.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

import httpx
import pytest

REPO = Path(__file__).resolve().parents[3]
E2E_DIR = Path(__file__).resolve().parent
FRONTEND = REPO / "frontend"

BUILT_DATA = (
    "data/processed/suppliers.csv", "data/processed/routes.csv", "data/processed/tariffs.csv", "data/processed/disruptions.csv",
    "data/processed/inventory_multi_warehouse.csv", "data/processed/demand_modeling_panel.csv",
    "ml/artifacts/xgboost_demand/2026.09.1/model.json",
)
requires_built_data = pytest.mark.skipif(not all((REPO / p).exists() for p in BUILT_DATA), reason="run the Phase 3-5 pipelines first")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def find_chrome() -> str | None:
    candidates = [
        os.environ.get("CHROME_PATH"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chromium-browser"), shutil.which("chrome"),
    ]
    return next((c for c in candidates if c and Path(c).exists()), None)


class Server:
    """A real uvicorn process. `kill()` is a crash (no shutdown hooks run); `stop()` is polite."""

    def __init__(self, workdir: Path, *, port: int | None = None, db: Path | None = None, env: dict[str, str] | None = None, cors_origins: str = ""):
        self.workdir, self.port = workdir, port or free_port()
        self.db = db or workdir / "world_state.db"
        self.env = {**os.environ, "DATABASE_URL": f"sqlite:///{self.db.as_posix()}", "CORS_ORIGINS": cors_origins, "PYTHONWARNINGS": "ignore",
                    "LOG_FORMAT": "json", "LOG_LEVEL": "INFO", **(env or {})}  # JSON on stderr: the server's log file is parseable
        self.proc: subprocess.Popen | None = None
        self.log = workdir / f"server-{self.port}.log"

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout: float = 120) -> "Server":
        assert self.proc is None or self.proc.poll() is not None, "already running"
        with open(self.log, "ab") as log:
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "backend.tests.e2e.app:app", "--host", "127.0.0.1", "--port", str(self.port), "--log-level", "warning"],
                cwd=REPO, env=self.env, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"the server exited during startup:\n{self.log.read_text(errors='replace')[-3000:]}")
            try:
                if httpx.get(f"{self.url}/api/health", timeout=2).status_code == 200:
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        self.kill()
        raise RuntimeError(f"the server did not come up in {timeout}s:\n{self.log.read_text(errors='replace')[-3000:]}")

    def warm(self) -> "Server":
        """The first solve loads the forecasting model; do it before a test is timed or interrupted."""
        httpx.get(f"{self.url}/api/scenarios/SUEZ_CLOSURE/comparison", timeout=120).raise_for_status()
        return self

    def kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(30)

    stop = kill  # uvicorn on Windows has no graceful terminate from outside; either way the state is what the database holds

    def restart(self) -> "Server":
        self.kill()
        return self.start()

    def client(self, timeout: float = 60) -> httpx.Client:
        return httpx.Client(base_url=self.url, timeout=timeout)


@pytest.fixture(scope="session")
def make_server(tmp_path_factory):
    servers: list[Server] = []

    def make(**kwargs) -> Server:
        server = Server(tmp_path_factory.mktemp("e2e-server"), **kwargs)
        servers.append(server)
        return server

    yield make
    for s in servers:
        s.kill()


@pytest.fixture(scope="session")
def server(make_server) -> Server:
    """One shared server (fast scripted LLM) for the tests that do not need to hurt it."""
    return make_server().start().warm()


@pytest.fixture()
def api(server):
    with server.client() as c:
        yield c


# ---------------------------------------------------------------- browser
class WebApp:
    def __init__(self, workdir: Path, api_url: str, cors_port: int):
        self.port, self.api_url = cors_port, api_url
        self.log = workdir / f"vite-{cors_port}.log"
        self.proc: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout: float = 60) -> "WebApp":
        node = shutil.which("node")
        env = {**os.environ, "VITE_API_BASE_URL": self.api_url}
        with open(self.log, "ab") as log:  # node runs vite directly: an `npx` wrapper would leave the real process behind when killed
            self.proc = subprocess.Popen([node, "node_modules/vite/bin/vite.js", "--port", str(self.port), "--host", "127.0.0.1", "--strictPort"],
                                         cwd=FRONTEND, env=env, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vite exited during startup:\n{self.log.read_text(errors='replace')[-2000:]}")
            try:
                if httpx.get(self.url, timeout=2).status_code == 200:
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        self.stop()
        raise RuntimeError("vite did not come up")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(30)


def _browser_prerequisites() -> str | None:
    if not shutil.which("node"):
        return "node is not installed"
    if not (FRONTEND / "node_modules" / "vite").exists():
        return "run `npm install` in frontend/ first"
    if not find_chrome():
        return "Chrome/Edge not found (set CHROME_PATH)"
    return None


requires_browser = pytest.mark.skipif(_browser_prerequisites() is not None, reason=_browser_prerequisites() or "")


class Stack:
    """A backend and the frontend served against it."""

    def __init__(self, server: Server, web: WebApp):
        self.server, self.web = server, web

    def client(self, timeout: float = 60) -> httpx.Client:
        return self.server.client(timeout)


@pytest.fixture(scope="session")
def make_stack(make_server, tmp_path_factory):
    webs: list[WebApp] = []

    def make(**server_kwargs) -> Stack:
        web_port = free_port()
        # the API only lets a browser in from the origin the page is served from, so the backend learns it before it starts
        server = make_server(cors_origins=f"http://127.0.0.1:{web_port}", **server_kwargs).start().warm()
        web = WebApp(tmp_path_factory.mktemp("e2e-web"), server.url, web_port).start()
        webs.append(web)
        return Stack(server, web)

    yield make
    for w in webs:
        w.stop()


class Browser:
    """Runs one script (backend/tests/e2e/browser/<name>.mjs) in headless Chrome and returns what it returned.
    Runs that share `profile` share localStorage, so a test can stop the backend between two of them."""

    def __init__(self, workdir: Path, name: str):
        self.workdir, self.name, self.artifacts = workdir, name, workdir / "artifacts"
        self.artifacts.mkdir(parents=True, exist_ok=True)

    def run(self, script: str, args: dict, *, profile: str = "default", timeout: float = 240, hooks: dict[str, Callable[[], None]] | None = None) -> dict:
        """`hooks` are functions the script can call over HTTP (GET {hookUrl}/<name>) — the test's remote
        control over things a script cannot do from inside a browser, like stopping the backend."""
        with _HookServer(hooks or {}) as hook_url:
            return self._run(script, {**args, "hookUrl": hook_url}, profile, timeout)

    def _run(self, script: str, args: dict, profile: str, timeout: float) -> dict:
        env = {**os.environ, "E2E_ARGS": json.dumps(args), "E2E_CHROME": find_chrome() or "", "E2E_ARTIFACTS": str(self.artifacts),
               "E2E_PROFILE": str(self.workdir / f"profile-{profile}"), "E2E_DEBUG_PORT": str(free_port()), "E2E_SCRIPT": str(E2E_DIR / "browser" / f"{script}.mjs")}
        done = subprocess.run([shutil.which("node"), str(E2E_DIR / "browser" / "runner.mjs")], cwd=E2E_DIR / "browser", env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)  # node writes UTF-8; the locale codec would mangle × and σ
        lines = [ln for ln in done.stdout.splitlines() if ln.startswith("E2E_RESULT ")]
        if not lines:
            raise AssertionError(f"browser script {script!r} produced no result (exit {done.returncode}):\n{done.stdout[-3000:]}\n{done.stderr[-3000:]}")
        result = json.loads(lines[-1][len("E2E_RESULT "):])
        result.setdefault("problems", [])
        return result


class _HookServer:
    def __init__(self, hooks: dict[str, Callable[[], None]]):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                fn = hooks.get(self.path.strip("/"))
                body = b"unknown hook"
                if fn is not None:
                    fn()
                    body = b"ok"
                self.send_response(200 if fn is not None else 404)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # keep the test output clean
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)

    def __enter__(self) -> str:
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture()
def browser(tmp_path, request) -> Browser:
    return Browser(tmp_path, request.node.name)


# ---------------------------------------------------------------- helpers shared by the tests
def wait_for(client: httpx.Client, path: str, done, timeout: float = 90, every: float = 0.15) -> dict:
    """Polls GET `path` until done(data) is truthy; returns that data (or fails with the last one)."""
    deadline, last = time.time() + timeout, None
    while time.time() < deadline:
        last = client.get(path).json()["data"]
        if done(last):
            return last
        time.sleep(every)
    raise AssertionError(f"timed out waiting on {path}; last: {json.dumps(last)[:800]}")


def run_finished(status: dict) -> bool:
    return bool(status["run"]) and status["run"]["state"] != "RUNNING"


def start_scenario(client: httpx.Client, scenario_id: str, product_id: str | None = None) -> str:
    r = client.post(f"/api/scenarios/{scenario_id}/run", json={"product_id": product_id} if product_id else {})
    assert r.status_code == 202, r.text
    return r.json()["data"]["simulation_id"]


def finish(client: httpx.Client, sim: str, timeout: float = 90) -> dict:
    return wait_for(client, f"/api/simulations/{sim}/status", run_finished, timeout)


def free_text_run(client: httpx.Client, report: str, product_id: str = "22197", scenario_type: str = "E2E") -> str:
    sim = client.post("/api/simulations", json={"scenario_type": scenario_type}).json()["data"]["simulation_id"]
    r = client.post(f"/api/simulations/{sim}/run", json={"signal": report, "product_id": product_id})
    assert r.status_code == 202, r.text
    return sim
