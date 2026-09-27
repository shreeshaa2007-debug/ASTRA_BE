"""SAP BTP runtime environment: the services an application is bound to.

On Cloud Foundry, BTP passes an app its bound service instances (SAP HANA Cloud, XSUAA, Destination, Integration
Suite, AI Core, ...) as JSON in the `VCAP_SERVICES` environment variable:

    {"hana": [{"name": "resilientsc-db", "label": "hana", "tags": ["hana", "database"], "credentials": {...}}],
     "xsuaa": [{"name": "resilientsc-uaa", "label": "xsuaa", "tags": ["xsuaa"], "credentials": {...}}]}

This module is the only reader of that format. Not covered: Kyma/Kubernetes, where the SAP BTP service operator
mounts each binding as a directory of files — a different reader with the same `ServiceBinding` result would slot in
here (see docs/sap-readiness.md).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping


class BTPConfigError(RuntimeError):
    """The platform-provided configuration is unusable. Messages name the setting, never its value."""


@dataclass(frozen=True)
class ServiceBinding:
    label: str  # the service offering, e.g. "hana", "xsuaa"
    name: str  # the instance name chosen in the deployment descriptor
    tags: tuple[str, ...]
    credentials: dict[str, Any] = field(repr=False)  # repr=False: a stray print or log of a binding must not leak them


def running_on_btp(env: Mapping[str, str] | None = None) -> bool:
    """True on Cloud Foundry (which sets VCAP_APPLICATION for every app instance)."""
    return "VCAP_APPLICATION" in (os.environ if env is None else env)


def load_bindings(env: Mapping[str, str] | None = None) -> list[ServiceBinding]:
    env = os.environ if env is None else env
    raw = env.get("VCAP_SERVICES", "").strip()
    if not raw:
        return []
    try:
        services = json.loads(raw)
    except json.JSONDecodeError as exc:
        # deliberately not chained with the text: it is JSON that contains credentials
        raise BTPConfigError(f"VCAP_SERVICES is not valid JSON ({exc.msg} at position {exc.pos})") from None
    if not isinstance(services, dict):
        raise BTPConfigError("VCAP_SERVICES must be a JSON object keyed by service label")

    bindings: list[ServiceBinding] = []
    for label, instances in services.items():
        if not isinstance(instances, list):
            raise BTPConfigError(f"VCAP_SERVICES[{label!r}] must be a list of service instances")
        for instance in instances:
            if not isinstance(instance, dict):
                raise BTPConfigError(f"VCAP_SERVICES[{label!r}] contains an entry that is not an object")
            credentials = instance.get("credentials") or {}
            if not isinstance(credentials, dict):
                raise BTPConfigError(f"VCAP_SERVICES[{label!r}] has an instance whose credentials are not an object")
            bindings.append(ServiceBinding(
                label=str(instance.get("label") or label), name=str(instance.get("name") or ""),
                tags=tuple(str(t) for t in instance.get("tags") or ()), credentials=credentials,
            ))
    return bindings


def find_binding(*, label: str | None = None, tag: str | None = None, name: str | None = None,
                 env: Mapping[str, str] | None = None) -> ServiceBinding | None:
    """The first bound instance matching every criterion given, or None. An app bound to two instances of one
    service must say which by `name`; guessing between two databases would be worse than failing."""
    matches = [
        b for b in load_bindings(env)
        if (label is None or b.label == label) and (tag is None or tag in b.tags) and (name is None or b.name == name)
    ]
    if len(matches) > 1 and name is None:
        raise BTPConfigError(
            f"{len(matches)} service instances match (label={label!r}, tag={tag!r}): {[b.name for b in matches]}; say which by name"
        )
    return matches[0] if matches else None


def require(credentials: Mapping[str, Any], *keys: str, service: str) -> None:
    missing = [k for k in keys if not credentials.get(k)]
    if missing:
        raise BTPConfigError(f"the {service} binding is missing credentials: {', '.join(missing)}")
