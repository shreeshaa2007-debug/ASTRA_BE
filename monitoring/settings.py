"""Observability settings: backend/config/monitoring.yaml, with the environment overriding the few
values an operator changes per process. Read fresh each call so a test can point it elsewhere."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

MONITORING_CONFIG_PATH = Path("backend/config/monitoring.yaml")

_DEFAULTS = {
    "logging": {"level": "INFO", "format": "text"},
    "metrics": {"reservoir_size": 512},
    "runs": {"overdue_seconds": 120},
    "llm": {"circuit_failure_threshold": 3, "circuit_cooldown_seconds": 30},
    "model": {"drift_z_threshold": 3.0, "drift_window_days": 28, "min_training_windows": 30, "warnings_kept": 50},
}


def load_settings(path: Path = MONITORING_CONFIG_PATH) -> dict:
    """The file's values over the defaults (a missing file is fine: the defaults are the file's contents),
    then LOG_LEVEL / LOG_FORMAT / RUN_OVERDUE_SECONDS from the environment."""
    settings = {section: dict(values) for section, values in _DEFAULTS.items()}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for section, values in (yaml.safe_load(f) or {}).items():
                settings.setdefault(section, {}).update(values or {})
    if os.environ.get("LOG_LEVEL"):
        settings["logging"]["level"] = os.environ["LOG_LEVEL"].upper()
    if os.environ.get("LOG_FORMAT"):
        settings["logging"]["format"] = os.environ["LOG_FORMAT"].lower()
    if os.environ.get("RUN_OVERDUE_SECONDS"):
        settings["runs"]["overdue_seconds"] = float(os.environ["RUN_OVERDUE_SECONDS"])
    return settings
