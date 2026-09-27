"""SAP HANA Cloud connection settings from a service binding.

A bound `hana` instance carries `host`, `port`, `user`, `password` and `schema` (plus a JDBC `url`, a `certificate`
and, for HDI containers, a design-time `hdi_user`). SQLAlchemy reaches HANA through `sqlalchemy-hana` on top of SAP's
`hdbcli` driver: `pip install -r requirements-sap.txt`. Both are optional — the default deployment runs on SQLite
and never imports either.
"""
from __future__ import annotations

from typing import Any, Mapping

from sqlalchemy.engine import URL

from backend.sap.btp import require

DRIVERNAME = "hana+hdbcli"


def hana_url(credentials: Mapping[str, Any]) -> URL:
    """The runtime connection: TLS on and the server certificate validated against the system trust store (HANA
    Cloud only accepts encrypted connections). Built with `URL.create`, not string formatting, because generated
    HANA passwords contain characters that break a URL."""
    require(credentials, "host", "port", "user", "password", service="hana")
    return URL.create(
        DRIVERNAME,
        username=str(credentials["user"]), password=str(credentials["password"]),
        host=str(credentials["host"]), port=int(credentials["port"]),
        query={"encrypt": "true", "sslValidateCertificate": "true"},
    )
