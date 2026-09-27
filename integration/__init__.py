"""SAP Integration Suite (and friends): what the application says to other systems, and what it accepts from them.

Outbound: `DomainEvent`s (CloudEvents 1.0) published by an `EventPublisher` — `StoreEventBridge` turns each world-state
checkpoint into one. Inbound: `IntegrationSignal`s, accepted idempotently. See docs/sap-readiness.md.
"""
from backend.integration.bridge import DEFAULT_CHECKPOINTS, EVENT_TYPES, StoreEventBridge, build_event, checkpoints_from_env
from backend.integration.events import (
    BackgroundPublisher,
    DomainEvent,
    EventDeliveryError,
    EventPublisher,
    LoggingPublisher,
    NullPublisher,
    WebhookPublisher,
    build_publisher_from_env,
)

__all__ = [
    "DEFAULT_CHECKPOINTS", "EVENT_TYPES", "StoreEventBridge", "build_event", "checkpoints_from_env", "BackgroundPublisher", "DomainEvent",
    "EventDeliveryError", "EventPublisher", "LoggingPublisher", "NullPublisher", "WebhookPublisher", "build_publisher_from_env",
]
