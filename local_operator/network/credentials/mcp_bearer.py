"""The borrowed-bearer ``httpx.Auth`` an MCP server is connected with.

WHY THIS LIVES HERE AND NOT IN ``mcp/manager.py``, where it is used: that module
states (and relies on) a LAZY-SDK PROPERTY — it imports neither ``httpx`` nor
``anyio`` nor the ``mcp`` package at module level, so its transport-failure
classification can match exception classes by name without pulling the SDK in. An
``httpx.Auth`` subclass needs ``httpx`` at class-definition time, so putting the
class there would have broken that property for every import of the manager. This
module is imported ONLY from the brokered branch of ``_build_oauth_auth``, which is
reached only on a device that borrows, so the dependency arrives exactly when MCP
authentication arrives.

WHY AN ``httpx.Auth`` AT ALL, AND NOT A BRANCH IN THE REFRESH (build plan §0
finding 6): the MCP SDK reaches the wire through its own ``OAuthClientProvider``, so
the only seam where a DIFFERENT token can be presented is the client's ``auth=``.
``ensure_mcp_oauth_fresh`` returns endpoints, not tokens, so the design's original
hook point could not have returned one.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

#: The status that means "this bearer was refused". One name, because the repair
#: inside the flow and the challenge path outside it must agree on what a 401 is.
HTTP_UNAUTHORIZED = 401


class BrokeredBearerAuth(httpx.Auth):
    """An ``httpx.Auth`` that sets a BORROWED bearer, and repairs it once on a 401.

    WHY AN ``httpx.Auth`` AND NOT A BRANCH IN THE REFRESH (build plan §0 finding 6):
    the MCP SDK reaches the wire through its own ``OAuthClientProvider``, so the only
    seam where a different token can be presented is the client's ``auth=``. This
    class uses none of that machinery — it builds no provider, touches no
    ``McpTokenStorage``, binds no loopback port — it simply puts a header on the
    request the SDK already built.

    THE REPAIR IS DELIBERATELY ONE ROUND: on a 401 it reports the failure to the
    owner (which is the only party that may act on it) and asks for a fresh bearer
    once. A second 401 is passed through, because a loop here would be this device
    repeatedly asking a device that has already answered — the retry storm the
    refusal cache exists to prevent, one layer up.
    """

    def __init__(self, *, url: str, key: str, client: Any) -> None:
        self.url = url
        self.key = key
        self._client = client
        self._bearer = ""
        self._repaired = False

    def _blocking_bearer(self) -> str:
        """Borrow, or ``""``. Never raises: a failed borrow must not fail a connect.

        The connect that follows then behaves exactly as it did before brokering
        existed — an unauthenticated request against a server that wants a bearer,
        which the challenge path already turns into an actionable message.
        """
        from local_operator.network.credentials.types import Grant

        try:
            grant = self._client.request_grant_sync(self.key, provider="mcp-oauth", session_id="")
        except Exception:  # noqa: BLE001 — see the docstring
            return ""
        return grant.access_token if isinstance(grant, Grant) else ""

    async def async_auth_flow(self, request: httpx.Request) -> Any:
        if not self._bearer:
            self._bearer = await asyncio.to_thread(self._blocking_bearer)
        if not self._bearer:
            yield request
            return
        request.headers["Authorization"] = f"Bearer {self._bearer}"
        response = yield request
        if response.status_code != HTTP_UNAUTHORIZED or self._repaired:
            return
        self._repaired = True
        # Report FIRST and re-borrow second: the owner's own refresh (if any) is what
        # makes the second bearer different, and asking before reporting would hand
        # back the token that just failed.
        self._client.grants.drop(self.key)
        await asyncio.to_thread(self._client.report_sync, self.key, kind="invalid")
        self._bearer = await asyncio.to_thread(self._blocking_bearer)
        if not self._bearer:
            return
        request.headers["Authorization"] = f"Bearer {self._bearer}"
        yield request


#: The status that means "this bearer was refused" — one name, because the repair
#: above and any future caller must agree on it.
_HTTP_UNAUTHORIZED = 401


#: The name ``mcp/manager.py`` imports. Exported rather than renamed at the call
#: site so the seam has one spelling.
BROKERED_BEARER_AUTH = BrokeredBearerAuth
