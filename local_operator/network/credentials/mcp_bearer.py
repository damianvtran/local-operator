"""The borrowed-bearer ``httpx2.Auth`` an MCP server is connected with.

WHY THIS LIVES HERE AND NOT IN ``mcp/manager.py``, where it is used: that module
states (and relies on) a LAZY-SDK PROPERTY — it imports neither ``httpx`` nor
``anyio`` nor the ``mcp`` package at module level, so its transport-failure
classification can match exception classes by name without pulling the SDK in. An
``httpx2.Auth`` subclass needs ``httpx2`` at class-definition time, so putting the
class there would have broken that property for every import of the manager. This
module is imported ONLY from the brokered branch of ``_build_oauth_auth``, which is
reached only on a device that borrows, so the dependency arrives exactly when MCP
authentication arrives.

WHY AN ``httpx2.Auth`` AND NOT A BRANCH IN THE REFRESH (build plan §0 finding 6): the
MCP SDK reaches the wire through its own ``OAuthClientProvider``, so the only seam
where a DIFFERENT token can be presented is the client's ``auth=``.
``ensure_mcp_oauth_fresh`` returns endpoints, not tokens, so the design's original hook
point could not have returned one.

AND IT MUST BE **``httpx2``**, NOT ``httpx``. That is not a typo and it is the one
detail in this file a reader would otherwise "fix": the MCP SDK's transports are
built on a separate ``httpx2`` distribution
(``mcp.client.streamable_http.create_mcp_http_client`` types its ``auth`` parameter as
``httpx2.Auth``), while the rest of this repository imports ``httpx``. The two classes
are unrelated, so a subclass of one is REJECTED by the other — and rejected only at the
moment a borrower connects to an MCP server, on a path no unit test with no mesh
reaches. The type checker is what found it; keep the two names distinct here.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

# ``httpx2``, deliberately — see this module's docstring. The MCP SDK's transports
# type their ``auth`` parameter as ``httpx2.Auth``.
import httpx2

#: The status that means "this bearer was refused". ONE name: the first version had a
#: second, private copy of the same number at the bottom of the file (review round 1,
#: N1), and two constants for one fact are two places for it to drift.
HTTP_UNAUTHORIZED = 401


class BrokeredBearerAuth(httpx2.Auth):
    """An ``httpx2.Auth`` that sets a BORROWED bearer, and repairs it once on a 401.

    WHY AN ``httpx2.Auth`` AND NOT A BRANCH IN THE REFRESH: see the module docstring.
    This class uses none of the SDK's OAuth machinery — it builds no provider, touches
    no ``McpTokenStorage``, binds no loopback port — it puts a header on the request
    the SDK already built.

    THE BEARER IS RE-READ FOR EVERY REQUEST, and is never held here. The grant lives
    in the client's :class:`~local_operator.network.credentials.client.GrantCache`,
    which applies the grant's own expiry (``min(token_exp, now + grant_ttl_s)``); the
    first version kept the bearer on this object for the life of the connection and
    re-borrowed only after a 401, so a revoked borrower kept presenting it past the
    grant until the server happened to refuse it (review round 1, F4). The per-request
    read is a dict lookup while the grant is live, and a re-borrow (which the owner
    re-authorises against its CURRENT sharing list) once it is not.

    THE REPAIR IS DELIBERATELY ONE ROUND PER REQUEST: on a 401 it reports the failure
    to the owner (which is the only party that may act on it) and asks for a fresh
    bearer once. A second 401 is passed through, because a loop here would be this
    device repeatedly asking a device that has already answered — the retry storm the
    refusal cache exists to prevent, one layer up.
    """

    def __init__(self, *, url: str, key: str, client: Any) -> None:
        self.url = url
        self.key = key
        self._client = client

    def _live_bearer(self) -> str:
        """The cached grant's bearer while the grant is live, else ``""``. No I/O."""
        from local_operator.network.credentials.types import Grant

        grant = self._client.grants.get(self.key, "")
        return grant.access_token if isinstance(grant, Grant) else ""

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

    async def _bearer(self) -> str:
        return self._live_bearer() or await asyncio.to_thread(self._blocking_bearer)

    # ANNOTATED AS THE BASE DOES, not as ``Any``: this function contains ``yield``, so it
    # is an async GENERATOR, and the base class's own annotation is what says so.
    # Declaring ``Any`` made a checker read the override as a plain coroutine and
    # reject it against the protocol — which is the annotation being wrong, not the
    # code, and it is exactly the kind of wrongness a reader would copy.
    async def async_auth_flow(
        self, request: httpx2.Request
    ) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        bearer = await self._bearer()
        if not bearer:
            yield request
            return
        request.headers["Authorization"] = f"Bearer {bearer}"
        response = yield request
        if response.status_code != HTTP_UNAUTHORIZED:
            return
        # Report FIRST and re-borrow second: the owner's own refresh (if any) is what
        # makes the second bearer different, and asking before reporting would hand
        # back the token that just failed.
        self._client.grants.drop(self.key)
        await asyncio.to_thread(self._client.report_sync, self.key, kind="unauthorized")
        bearer = await asyncio.to_thread(self._blocking_bearer)
        if not bearer:
            return
        request.headers["Authorization"] = f"Bearer {bearer}"
        yield request
