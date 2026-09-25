"""Logging that can be searched: one JSON object per line (or readable text), every line stamped with the
request / simulation / run it is about, and no secret in any of it.

Until now the code logged with `logging.getLogger("resilientsc...")` but nothing configured a handler, so
under uvicorn every INFO line — including the per-step run log — was silently dropped. `configure_logging`
is the missing half. It is idempotent (calling it again replaces its own handler, never stacks one) and it
attaches to the "resilientsc" logger only, leaving the root logger — and pytest's caplog — alone.

Structured fields: pass `extra={"event": "step", "step": "optimize", "duration_ms": 12.3}` and they become
keys of the JSON object (and `k=v` pairs in text mode).
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import IO

from backend.monitoring import context

LOGGER_NAME = "resilientsc"
_HANDLER_MARK = "_resilientsc_handler"

# what a LogRecord always has; anything else on a record came in through `extra`
_STANDARD = set(vars(logging.LogRecord("x", 0, "x", 0, "", (), None))) | {"message", "asctime", "taskName"}
_SECRET_NAME = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL)", re.I)
_MIN_SECRET_LENGTH = 8  # a shorter "secret" would redact ordinary words


def _secrets() -> list[str]:
    """Values of secret-looking environment variables, longest first (so a value containing another is fully redacted)."""
    values = {v for k, v in os.environ.items() if _SECRET_NAME.search(k) and len(v) >= _MIN_SECRET_LENGTH}
    return sorted(values, key=len, reverse=True)


def redact(text: str) -> str:
    for secret in _secrets():
        text = text.replace(secret, "***")
    return text


class RedactionFilter(logging.Filter):
    """Belt and braces: the code already avoids logging secrets (the Gemini key travels in a header and is
    scrubbed from its own errors), and this makes sure a future slip does not reach a log."""

    def filter(self, record: logging.LogRecord) -> bool:
        if _secrets():
            record.msg = redact(record.getMessage())
            record.args = ()
            if record.exc_info:
                record.exc_text = redact("".join(logging.Formatter().formatException(record.exc_info)))
        return True


class ContextFilter(logging.Filter):
    """Stamps the current request / simulation / run onto the record."""

    def filter(self, record: logging.LogRecord) -> bool:
        for name, value in context.current().items():
            if not hasattr(record, name):
                setattr(record, name, value)
        return True


def _extras(record: logging.LogRecord) -> dict:
    return {k: v for k, v in vars(record).items() if k not in _STANDARD and not k.startswith("_")}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname, "logger": record.name, "message": record.getMessage(),
            **_extras(record),
        }
        if record.exc_info:
            entry["exc"] = record.exc_text or self.formatException(record.exc_info)
        return json.dumps(entry, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        fields = " ".join(f"{k}={v}" for k, v in _extras(record).items())
        line = f"{ts} {record.levelname:<7} {record.name} {record.getMessage()}" + (f"  [{fields}]" if fields else "")
        if record.exc_info:
            line += "\n" + (record.exc_text or self.formatException(record.exc_info))
        return line


def configure_logging(level: str = "INFO", fmt: str = "text", stream: IO[str] | None = None) -> logging.Logger:
    """Attaches (or replaces) this app's handler on the "resilientsc" logger. Returns that logger."""
    if fmt not in ("text", "json"):
        raise ValueError(f"log format must be 'text' or 'json', not {fmt!r}")
    logger = logging.getLogger(LOGGER_NAME)
    for handler in [h for h in logger.handlers if getattr(h, _HANDLER_MARK, False)]:
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    setattr(handler, _HANDLER_MARK, True)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    handler.addFilter(ContextFilter())
    handler.addFilter(RedactionFilter())
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    return logger
