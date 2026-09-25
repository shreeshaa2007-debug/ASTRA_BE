"""Observability (Phase 19): structured logging with correlation ids, in-process metrics, and model
monitoring. Nothing here decides anything — it watches. See docs/architecture.md §5.4."""
from backend.monitoring.context import bind, current, request_id_var, run_id_var, simulation_id_var
from backend.monitoring.metrics import Metrics, metrics
from backend.monitoring.model_monitor import ModelMonitor, monitor
from backend.monitoring.settings import load_settings
from backend.monitoring.structured_logging import configure_logging, redact

__all__ = [
    "bind", "current", "request_id_var", "run_id_var", "simulation_id_var", "Metrics", "metrics", "ModelMonitor", "monitor",
    "load_settings", "configure_logging", "redact",
]
