"""Owner-pinned Radient API calls. Never use the model-routing key cascade."""

from __future__ import annotations

from contextlib import closing
from typing import Any

import httpx

from local_operator.providers.auth_store import AuthStore
from local_operator.tunnels.config import DEFAULT_API_URL


def credential_id(selected: int | None = None) -> int:
    with closing(AuthStore()) as store:
        rows = [r for r in store.list_credentials("radient") if r.credential_type == "oauth"]
        if selected is not None:
            rows = [row for row in rows if row.id == selected]
        if len(rows) != 1:
            raise ValueError(
                "Log in with /login radient first. If multiple accounts are signed in, "
                "select one with --credential-id from lop login-status."
            )
        return rows[0].id


class RadientTunnels:
    """A connector stays bound to the credential chosen at enrollment.

    The inference resolver deliberately rotates credentials and may fall back
    to environment API keys. That policy is unsuitable for owning a tunnel:
    neither a quota event nor an expired login may switch the owning account.
    """

    def __init__(self, selected: int, client: httpx.AsyncClient) -> None:
        self.selected = selected
        self.client = client

    async def request(
        self,
        method: str,
        path: str = "",
        *,
        body: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> Any:
        with closing(AuthStore()) as store:
            row = store.get_credential(self.selected)
            if row is None or row.provider != "radient" or row.credential_type != "oauth":
                raise ValueError("The tunnel's Radient login is unavailable; log in again.")
            credentials = await store.ensure_oauth_fresh(self.selected)
        token = credentials.get("access") if credentials else None
        if not isinstance(token, str) or not token:
            raise ValueError("The tunnel's Radient login expired; log in again.")
        headers = {"Authorization": f"Bearer {token}"}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        response = await self.client.request(
            method,
            DEFAULT_API_URL + "/v1/tunnels" + path,
            json=body,
            headers=headers,
            timeout=30,
            follow_redirects=False,
        )
        # Never surface provider error bodies: they may echo a bearer, client
        # secret, or connector token. The status is enough for a useful retry.
        if response.status_code >= 400:
            raise ValueError(f"Radient tunnel request failed (HTTP {response.status_code}).")
        envelope = response.json()
        if not isinstance(envelope, dict) or "result" not in envelope:
            raise ValueError("Radient returned an invalid tunnel response.")
        return envelope["result"]
