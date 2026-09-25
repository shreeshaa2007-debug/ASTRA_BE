"""The LLM transport — the one place the Sensing Agent touches an external
service. Google Gemini's REST endpoint over the standard library: no SDK
dependency to drift, and the opener is injectable so tests never hit the network.

The client sends text and gets text back. It is given no tools, no state and no
write access, which is what makes "the LLM proposes, Python validates" true by
construction rather than by convention.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from backend.monitoring.metrics import metrics

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-2.5-flash"
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMUnavailableError(RuntimeError):
    """The LLM could not be reached or returned nothing usable. Distinct from
    an LLM that answered badly — that is a rejected candidate, not an outage."""

    error_code = "LLM_UNAVAILABLE"


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    latency_ms: float
    attempts: int


class CircuitBreakerLLM:
    """Stops calling a model that keeps failing. After `failure_threshold` consecutive LLMUnavailableErrors the
    circuit opens for `cooldown_seconds`: calls fail at once instead of each holding a worker for retries x timeout
    (three attempts at 30 s is a minute and a half of a thread that a queue of runs is waiting for). After the cooldown
    one trial call is let through; success closes the circuit, failure re-opens it. Wraps any LLMClient."""

    def __init__(self, inner: "LLMClient", failure_threshold: int = 3, cooldown_seconds: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self._inner, self._threshold, self._cooldown, self._clock = inner, failure_threshold, cooldown_seconds, clock
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def model(self) -> str:
        return self._inner.model

    def circuit_state(self) -> str:
        """'closed' (normal), 'open' (refusing calls) or 'half_open' (cooldown over: the next call is a trial)."""
        with self._lock:
            if self._opened_at is None:
                return "closed"
            return "half_open" if self._clock() - self._opened_at >= self._cooldown else "open"

    def generate_json(self, *, system_instruction: str, user_text: str, response_schema: dict) -> LLMResponse:
        with self._lock:
            if self._opened_at is not None and self._clock() - self._opened_at < self._cooldown:
                remaining = self._cooldown - (self._clock() - self._opened_at)
                metrics.inc("llm_circuit_open_total")
                raise LLMUnavailableError(f"the language model has failed {self._failures} times in a row; not calling it again for {remaining:.0f}s (circuit open)")
        try:
            response = self._inner.generate_json(system_instruction=system_instruction, user_text=user_text, response_schema=response_schema)
        except LLMUnavailableError:
            with self._lock:
                self._failures += 1
                if self._failures >= self._threshold:
                    self._opened_at = self._clock()
            raise
        with self._lock:
            self._failures, self._opened_at = 0, None
        return response


class LLMClient(Protocol):
    model: str

    def generate_json(self, *, system_instruction: str, user_text: str, response_schema: dict) -> LLMResponse: ...


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        *,
        timeout_seconds: float = 30,
        max_retries: int = 2,
        backoff_seconds: float = 2.0,
        thinking_budget: int | None = 0,
        base_url: str = GEMINI_BASE_URL,
        opener: Callable[..., Any] = urllib.request.urlopen,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not api_key:
            raise LLMUnavailableError("LLM_API_KEY is not set")
        self._api_key = api_key
        self.model = model
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._thinking_budget = thinking_budget
        self._base_url = base_url
        self._opener = opener
        self._sleep = sleep

    @classmethod
    def from_env(cls, llm_config: dict | None = None, **overrides) -> "GeminiClient":
        """LLM_API_KEY (required) and LLM_MODEL from the environment;
        transport settings from sensing_config.yaml's `llm:` section."""
        cfg = llm_config or {}
        return cls(
            os.environ.get("LLM_API_KEY", ""),
            os.environ.get("LLM_MODEL") or DEFAULT_MODEL,
            timeout_seconds=cfg.get("timeout_seconds", 30),
            max_retries=cfg.get("max_retries", 2),
            backoff_seconds=cfg.get("backoff_seconds", 2.0),
            thinking_budget=cfg.get("thinking_budget", 0),
            **overrides,
        )

    def generate_json(self, *, system_instruction: str, user_text: str, response_schema: dict) -> LLMResponse:
        generation_config: dict[str, Any] = {"temperature": 0, "responseMimeType": "application/json", "responseSchema": response_schema}
        if self._thinking_budget is not None:
            generation_config["thinkingConfig"] = {"thinkingBudget": self._thinking_budget}
        body = json.dumps({
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": generation_config,
        }).encode("utf-8")
        # the key travels in a header, never in the URL, so it can't end up in a log line or traceback
        request = urllib.request.Request(
            f"{self._base_url}/models/{self.model}:generateContent", data=body, method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": self._api_key},
        )

        started = time.perf_counter()
        last_error = "no attempt made"
        for attempt in range(1, self._max_retries + 2):
            try:
                with self._opener(request, timeout=self._timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return LLMResponse(self._extract_text(payload), self.model, round((time.perf_counter() - started) * 1000, 1), attempt)
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP {exc.code}: {self._error_message(exc)}"
                if exc.code not in _RETRYABLE_STATUS:
                    raise LLMUnavailableError(f"Gemini rejected the request ({last_error})") from None
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
                last_error = f"network error: {self._scrub(str(getattr(exc, 'reason', exc)))}"
            except json.JSONDecodeError:
                last_error = "Gemini returned a non-JSON response body"
            if attempt <= self._max_retries:
                self._sleep(self._backoff * attempt)
        raise LLMUnavailableError(f"Gemini unavailable after {self._max_retries + 1} attempts (last: {last_error})")

    @staticmethod
    def _extract_text(payload: dict) -> str:
        candidates = payload.get("candidates") or []
        parts = (candidates[0].get("content") or {}).get("parts") or [] if candidates else []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        if not text.strip():
            block = (payload.get("promptFeedback") or {}).get("blockReason")
            finish = candidates[0].get("finishReason") if candidates else None
            raise LLMUnavailableError(f"Gemini returned no output (blockReason={block}, finishReason={finish})")
        return text

    def _error_message(self, exc: urllib.error.HTTPError) -> str:
        try:
            message = json.loads(exc.read().decode("utf-8"))["error"]["message"]
        except Exception:
            message = exc.reason or "no message"
        return self._scrub(str(message))[:200]

    def _scrub(self, text: str) -> str:
        return text.replace(self._api_key, "***")
