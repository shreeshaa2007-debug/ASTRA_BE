"""SAP BTP XSUAA (the Authorization & Trust Management service): settings for validating the tokens it issues.

Behind the SAP Approuter a signed-in user's requests reach the backend carrying their XSUAA JWT. The bound `xsuaa`
instance says where its public keys are and who the token is for:

    url           https://<subdomain>.authentication.<region>.hana.ondemand.com
    xsappname     resilientsc!t12345           (scopes in a token are prefixed with this: `resilientsc!t12345.Approve`)
    clientid      sb-resilientsc!t12345

Tokens are verified against `<url>/token_keys` and must have been issued by `<url>/oauth/token`. Tokens issued through
SAP Cloud Identity Services (IAS) have a different issuer and key URL: set AUTH_ISSUER / AUTH_JWKS_URL explicitly.
"""
from __future__ import annotations

from typing import Any, Mapping

from backend.sap.btp import require


def jwt_settings(credentials: Mapping[str, Any]) -> dict[str, Any]:
    require(credentials, "url", "xsappname", "clientid", service="xsuaa")
    base = str(credentials["url"]).rstrip("/")
    return {
        "jwks_url": f"{base}/token_keys",
        "issuer": f"{base}/oauth/token",
        "audience": [str(credentials["xsappname"]), str(credentials["clientid"])],  # a token for either is a token for this app
        "scope_prefix": f"{credentials['xsappname']}.",
    }
