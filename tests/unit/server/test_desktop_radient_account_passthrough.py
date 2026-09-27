"""The account op's response filter must pass ``verification`` through.

``public_data`` is the ONLY filter on the desktop proxy's single-op path (the
op ``account`` is that path), and it strips a fixed set of secret-shaped keys.
The Radient quota-recovery contract adds a ``verification`` object sibling to
``account``/``identity``; if the filter ever grew a positive allowlist — or a
new stripped key that collides — the UI's verify-to-claim callout would go
blank with no server-side error. These tests pin the pass-through and the
redaction together, so neither can change silently. The end-to-end shape over
real loopback HTTP is covered by ``tests/e2e/test_desktop_radient.py``.
"""

from __future__ import annotations

import json
from typing import Any

from local_operator.server.routes import desktop_radient


def _me_body() -> dict[str, Any]:
    return {
        "status": 200,
        "result": {
            "account": {"tenant_id": "t-1"},
            "identity": {"email": "operator@example.com"},
            "verification": {
                "email_verified": False,
                "signup_grant": "pending",
                "grant_amount": 5.0,
                "claim_url": "https://console.radienthq.com/dashboard/verification",
            },
            "access_token": "secret-token",
        },
    }


def test_public_data_keeps_verification_and_strips_secrets() -> None:
    out = desktop_radient.public_data(_me_body(), ["secret-token"])

    result = out["result"]
    assert result["verification"]["signup_grant"] == "pending"
    assert result["verification"]["grant_amount"] == 5.0
    assert result["verification"]["claim_url"].endswith("/dashboard/verification")
    # The redaction half is still the other side of the same coin: the bearer
    # that made the request must not travel back to the client, in a key or in
    # a string.
    assert "access_token" not in result
    assert "secret-token" not in json.dumps(out)


def test_public_data_tolerates_a_missing_verification_object() -> None:
    """An older backend must not break the caller: absence is simply absence."""
    body = _me_body()
    del body["result"]["verification"]

    out = desktop_radient.public_data(body, ["secret-token"])

    assert "verification" not in out["result"]
    assert out["result"]["account"]["tenant_id"] == "t-1"
