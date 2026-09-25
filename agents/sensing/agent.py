"""SensingAgent — composes detect_disruption -> classify_event -> validate_event
into one call, and returns a SensingResult. It never touches the world state:
the orchestrator (Phase 14) commits `result.event` at the `event_sensed`
checkpoint, and only when the status is EVENT. That is the concrete
mechanism behind brief §13's "the LLM must not directly modify supply-chain
data" — the LLM's output is an argument to a validator, not a write.

Four outcomes, none of them a silent fake success:
  EVENT          a candidate passed validate_event()
  NO_DISRUPTION  the LLM read the text and found no supply-chain disruption in it
  REJECTED       the input, or the LLM's candidate, failed validation — logged, with every reason
  ERROR          the LLM could not be reached
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from backend.agents.sensing import tools
from backend.agents.sensing.llm import CircuitBreakerLLM, GeminiClient, LLMClient, LLMUnavailableError
from backend.monitoring.metrics import metrics
from backend.monitoring.settings import load_settings
from backend.schemas.sensing import SensingCatalog, SensingResult

logger = logging.getLogger("resilientsc.sensing")
_LOG_CANDIDATE_CHARS = 500


class SensingAgent:
    def __init__(
        self,
        llm: LLMClient | None = None,
        catalog: SensingCatalog | None = None,
        config: dict | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self.config = config or tools.load_config()
        self._llm = llm
        self._catalog = catalog
        self._clock = clock

    @property
    def catalog(self) -> SensingCatalog:
        if self._catalog is None:
            self._catalog = tools.load_catalog()
        return self._catalog

    def _llm_client(self) -> LLMClient:
        if self._llm is None:  # built on first use, so a pre-structured simulated trigger works with no API key
            breaker = load_settings()["llm"]
            self._llm = CircuitBreakerLLM(
                GeminiClient.from_env(self.config["llm"]), int(breaker["circuit_failure_threshold"]), float(breaker["circuit_cooldown_seconds"]))
        return self._llm

    def llm_status(self) -> dict:
        """For the readiness check: whether a model can be called at all, and whether its circuit breaker is holding it off.
        Reads no key and makes no call."""
        state = getattr(self._llm, "circuit_state", None)
        return {"configured": self._llm is not None or bool(os.environ.get("LLM_API_KEY")), "circuit": state() if callable(state) else "closed"}

    def sense(self, raw: str | Mapping[str, Any]) -> SensingResult:
        result = self._sense(raw)
        metrics.inc("sensing_outcomes_total", {"status": result.status})
        return result

    def _sense(self, raw: str | Mapping[str, Any]) -> SensingResult:
        now = self._clock()

        try:
            signal = tools.detect_disruption(raw, self.config, now)
        except tools.InvalidSignalError as exc:
            logger.warning("sensing input rejected: %s", exc)
            return SensingResult(status="REJECTED", error_code="VALIDATION_ERROR", errors=[str(exc)])

        try:
            # a pre-structured simulated trigger has nothing to classify, so it must not need an LLM (or an API key) at all
            llm = None if signal.candidate is not None else self._llm_client()
            classification = tools.classify_event(signal, self.catalog, llm, self.config, now)
        except LLMUnavailableError as exc:
            metrics.inc("llm_calls_total", {"result": "error"})
            logger.error("sensing could not reach the LLM: %s", exc, extra={"event": "llm_unavailable"})
            return SensingResult(status="ERROR", error_code=exc.error_code, errors=[str(exc)], signal=signal)
        if llm is not None:  # a call was really made
            metrics.inc("llm_calls_total", {"result": "ok"})
            metrics.observe("llm_latency_ms", classification.latency_ms)
            if classification.attempts > 1:
                metrics.inc("llm_retries_total", n=classification.attempts - 1)

        llm_info = {"model": classification.model, "latency_ms": classification.latency_ms, "attempts": classification.attempts}
        common = dict(signal=signal, candidate=classification.candidate, llm=llm_info)

        if classification.candidate is None:
            return self._rejected([classification.parse_error or "no candidate"], **common)
        candidate = classification.candidate
        rationale = candidate.get("rationale") if isinstance(candidate.get("rationale"), str) else None

        # a pre-structured simulated trigger is an event by definition; the LLM has to say so
        flag = candidate.get("is_disruption", True if signal.candidate is not None else None)
        if not isinstance(flag, bool):
            return self._rejected(["is_disruption is missing or not a boolean"], **common)
        if flag is False:
            logger.info("sensing found no disruption in the signal (%s)", rationale)
            return SensingResult(status="NO_DISRUPTION", rationale=rationale, **common)

        outcome = tools.validate_event(candidate, self.catalog, self.config, now)
        if not outcome.valid:
            return self._rejected(outcome.errors, warnings=outcome.warnings, **common)

        logger.info("sensed %s (%s) affecting routes=%s suppliers=%s products=%s",
                    outcome.event.event_id, outcome.event.event_type, outcome.event.affected_routes,
                    outcome.event.affected_suppliers, outcome.event.affected_products)
        return SensingResult(status="EVENT", event=outcome.event, warnings=outcome.warnings, rationale=rationale, **common)

    @staticmethod
    def _rejected(errors: list[str], warnings: list[str] | None = None, **common) -> SensingResult:
        shown = json.dumps(common.get("candidate"), default=str)[:_LOG_CANDIDATE_CHARS]
        logger.warning("sensing candidate rejected: %s | candidate=%s", "; ".join(errors), shown)
        return SensingResult(status="REJECTED", error_code="INVALID_SENSING_OUTPUT", errors=errors, warnings=warnings or [], **common)
