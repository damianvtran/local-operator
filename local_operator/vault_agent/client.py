"""agent-server's Vault routes 1, 15, 27 and 29, spoken with the PERSON's bearer.

Every call here authenticates as the signed-in person (the stored Radient OAuth
access token, resolved by ``hub_org.eligibility.person_client``), never an
application key: Vault decisions are per person (VAULT.md §6.12 refuses
application keys with ``401``), and the agent proof on route 15 rides ON TOP of
that bearer rather than replacing it.

TRANSPORT RULES, each one a contract item:

* ``allow_redirects=False`` and a 3xx is a failure, as ``RadientClient``'s org
  calls do: a followed redirect re-sends the bearer wherever it points.
* No body is ever logged, and a route-15 answer is handed back to exactly one
  caller (``resolve.py``), which puts the value into the session credential
  store and nowhere else (VAULT.md §4: never in a message, log, summary or
  memory).
* No retry anywhere. A route-27 challenge is single-use, so a retried route-15
  call would re-present a burned challenge; a retry is a new mint, decided by
  the caller.

Refusals surface as ``clients._http.APIError`` (status, code, scrubbed details)
built by the shared ``api_error_from_response``, so the vault path classifies
the same error shape as every other Radient client.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping

#: Route-27 answer shapes for a ``resolve`` mint (VAULT.md §1a step 2). The local
#: id rides in ``X-Vault-Challenge``; the authority id goes inside the commitment.
LOCAL_CHALLENGE_RE = re.compile(r"^vch_[0-9a-f]{64}$")
AUTHORITY_CHALLENGE_RE = re.compile(r"^vch_[0-9a-f]{32}$")
#: Ids interpolated into a URL path. Restricted so a model-supplied id cannot
#: smuggle a path segment or a query into the request.
VAULT_ID_RE = re.compile(r"^vlt_[0-9A-Za-z_-]{1,128}$")
ITEM_ID_RE = re.compile(r"^vit_[0-9A-Za-z_-]{1,128}$")

#: Route 29 pagination guard. The authority pages by a byte budget (VAULT.md
#: §6.20: 16 live grants can take four pages at any limit), so "every page" is
#: the rule; this cap only stops a server bug from looping a client forever.
MAX_DELEGATION_PAGES = 32

_TIMEOUT_SECONDS = 30.0

Request = Callable[..., Any]


class VaultClient:
    """One person's view of the Vault routes this harness uses.

    ``request`` is ``requests.request``-shaped and injectable so tests drive the
    real request construction through a stub transport.
    """

    def __init__(
        self,
        *,
        base_url: str,
        bearer_headers: Mapping[str, str],
        secrets: list[str] | None = None,
        request: Request | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._headers = dict(bearer_headers)
        self._secrets = list(secrets or [])
        if request is None:
            import requests

            request = requests.request
        self._request = request

    @classmethod
    def for_person(cls, radient_client: Any) -> "VaultClient":
        """Build from ``hub_org.eligibility.person_client``'s ``RadientClient``."""

        return cls(
            base_url=radient_client.base_url,
            bearer_headers=radient_client.bearer_headers(),
            secrets=radient_client._credential_values(),
        )

    # -- transport ---------------------------------------------------------

    def _call(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        action: str,
    ) -> dict[str, Any]:
        import requests

        from local_operator.clients._http import APIError, api_error_from_response

        merged = dict(self._headers)
        if headers:
            merged.update(headers)
        try:
            response = self._request(
                method,
                f"{self.base_url}{path}",
                params=dict(params) if params else None,
                json=dict(json_body) if json_body is not None else None,
                headers=merged,
                allow_redirects=False,
                timeout=_TIMEOUT_SECONDS,
            )
        except requests.exceptions.RequestException as exc:
            # No response at all: status None, which every caller treats as
            # "outcome unknown", never as a refusal.
            raise APIError(f"Could not {action}", status_code=None) from exc
        status = int(getattr(response, "status_code", 0) or 0)
        if 300 <= status < 400:
            raise APIError(
                f"Radient returned an unexpected redirect while trying to {action}",
                status_code=status,
                code="unexpected_redirect",
            )
        if status >= 400 or status < 200:
            raise api_error_from_response(
                response, fallback_message=f"Could not {action}", secrets=self._secrets
            )
        try:
            body = response.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            raise APIError(
                f"Radient returned an unrecognised response while trying to {action}",
                status_code=status,
            )
        return body

    # -- route 1 -------------------------------------------------------------

    def status(self, tenant_id: str) -> dict[str, Any]:
        """``GET /v1/vaults/status?tenant_id=`` (route 1)."""

        return self._call(
            "GET", "/vaults/status", params={"tenant_id": tenant_id}, action="read Vault status"
        )

    # -- route 29 ------------------------------------------------------------

    def list_delegations(self, tenant_id: str) -> list[dict[str, Any]]:
        """Every row of ``GET /v1/vault-delegations`` (route 29), all pages.

        The cursor is opaque and contains a ``.``; it is passed as a query
        parameter (``requests`` encodes it) and never validated or rewritten
        here, which is the bug the console BFF had (design §1.2 item 1).
        """

        rows: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(MAX_DELEGATION_PAGES):
            params = {"tenant_id": tenant_id}
            if cursor:
                params["cursor"] = cursor
            body = self._call(
                "GET", "/vault-delegations", params=params, action="list Vault agent access"
            )
            page = body.get("delegations")
            if isinstance(page, list):
                rows.extend(row for row in page if isinstance(row, dict))
            nxt = body.get("next_cursor")
            if not isinstance(nxt, str) or not nxt:
                return rows
            cursor = nxt
        from local_operator.clients._http import APIError

        raise APIError("Vault agent access list did not terminate", status_code=None)

    # -- routes 27 + 15 ------------------------------------------------------

    def mint_resolve(self, *, vault_id: str, item_id: str) -> dict[str, Any]:
        """``POST /v1/vaults/challenges {op:"resolve", vault_id, item_id}`` (route 27)."""

        return self._call(
            "POST",
            "/vaults/challenges",
            json_body={"op": "resolve", "vault_id": vault_id, "item_id": item_id},
            action="start a Vault resolve",
        )

    def resolve_as_agent(
        self,
        *,
        vault_id: str,
        item_id: str,
        endpoint_id: str,
        challenge_id: str,
        delegation_id: str,
        timestamp: int,
        proof: str,
    ) -> dict[str, Any]:
        """``POST /v1/vault-items/:id/resolve?as=agent&endpoint_id=&vault_id=`` (route 15).

        ``endpoint_id`` goes on the wire EXACTLY as the proof signed it.
        """

        if not ITEM_ID_RE.fullmatch(item_id) or not VAULT_ID_RE.fullmatch(vault_id):
            raise ValueError("vault_id/item_id are not Vault ids")
        return self._call(
            "POST",
            f"/vault-items/{item_id}/resolve",
            params={"as": "agent", "endpoint_id": endpoint_id, "vault_id": vault_id},
            headers={
                "X-Vault-Challenge": challenge_id,
                "X-Vault-Delegation": delegation_id,
                "X-Vault-Agent-Ts": str(int(timestamp)),
                "X-Vault-Agent-Proof": proof,
            },
            action="resolve a Vault item",
        )
