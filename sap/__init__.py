"""SAP-specific configuration adapters — the only place that knows SAP's formats.

The rest of the code base talks to vendor-neutral seams (a `WorldStateRepository`, a `DatasetRepository`, an
`EventPublisher`, an `Authenticator`, an `LLMClient`). This package turns what SAP BTP hands an application — service
bindings in `VCAP_SERVICES`, HANA Cloud credentials, XSUAA settings, OAuth client-credentials — into the arguments
those seams take. Nothing here opens a connection at import time, and nothing here logs a credential.

See docs/sap-readiness.md for what each module is for and what has and has not been verified against a real tenant.
"""
