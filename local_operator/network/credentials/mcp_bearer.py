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

        A REFUSAL IS EVIDENCE, NOT SILENCE. ``""`` alone left the connect to fail as
        a bare 401 with no durable record of WHY no bearer was offered, so every
        operator-facing surface could only guess between "the owner signed out",
        "the share was withdrawn" and "the owner is unreachable" — the three states
        that need three different repairs. Recording it here (through the same
        broker-code table the client uses) is what makes the cause PROVABLE on a
        later read; the value never reaches this file, and ``record`` never raises.
        """
        from local_operator.network.credentials.types import Grant

        try:
            grant = self._client.request_grant_sync(self.key, provider="mcp-oauth", session_id="")
        except Exception:  # noqa: BLE001 — see the docstring
            return ""
        if isinstance(grant, Grant):
            return grant.access_token
        self._record_refusal(grant)
        return ""

    def _record_refusal(self, error: Any) -> None:
        """Persist the broker's refusal CODE as this key's evidence. Never raises.

        EVERY read is inside the guard, including the owner labels: they are client
        lookups and can raise on a half-torn client, which is exactly the state a
        connect-time failure is often met in (review round 1, NIT).
        """
        code = str(getattr(error, "code", "") or "")
        from local_operator.network.credentials import mcp_state

        state = mcp_state.state_for_code(code)
        if not state:
            # A code the evidence vocabulary does not classify (a quota or rate
            # answer) records nothing: overwriting a precise older answer with a
            # vague new one is worse than leaving the reader to ask again.
            return
        if state == "revoked":
            # A COLD ``not_a_holder`` IS NOT A REVOCATION — the same rule the
            # client's own evidence writer applies, asked through the same method so
            # the two writers cannot disagree about it. A device that never held a
            # claim has lost nothing.
            try:
                if not self._client._has_prior_claim(self.key):  # noqa: SLF001
                    return
            except Exception:  # noqa: BLE001 — unknown is not a claim
                return
        try:
            owner_device = self._client.owner_of(self.key)
            owner_name = self._client.owner_label(self.key)
            network_id = getattr(self._client, "network_id", "")
            root = getattr(self._client, "root", None)
        except Exception:  # noqa: BLE001 — bookkeeping must never fail a borrow
            return
        mcp_state.record(
            network_id,
            self.key,
            state=state,
            code=code,
            owner_device=owner_device,
            owner_name=owner_name,
            root=root,
        )

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
        # back the token that just failed. The DROP comes after the report for the
        # same reason it does in ``store._report_failure`` (review round 1, Q1):
        # ``report_sync`` attributes the failure by reading the cached grant's row
        # id, and dropping first erased that read. ``report_sync`` drops the cache
        # itself before dialling, so nothing can re-serve the failed bearer.
        await asyncio.to_thread(self._client.report_sync, self.key, kind="unauthorized")
        # THE DURABLE HALF OF THE 401. Without it the only place this failure
        # exists is a bare ``auth-required`` on the remote: the broker refusal that
        # caused the empty bearer (``interactive_required``, ``refresh_failed``,
        # ``not_a_holder``) was swallowed by ``_blocking_bearer`` and nothing said
        # why. ``unauthorized`` is the one state only this layer can observe, so it
        # records it here; a later successful borrow overwrites it with ``synced``.
        from local_operator.network.credentials import mcp_state

        mcp_state.record(
            getattr(self._client, "network_id", ""),
            self.key,
            state="unauthorized",
            code="http_401",
            owner_device=self._client.owner_of(self.key),
            owner_name=self._client.owner_label(self.key),
            root=getattr(self._client, "root", None),
        )
        self._client.grants.drop(self.key)
        # The 401 is also the one signal that this device's cached "no" may be
        # stale: the OWNER may have re-signed in since it refused us, and a cached
        # ``interactive_required`` would otherwise answer this re-borrow WITHOUT
        # dialling — the repair below could never reach the owner it is about. Same
        # rule as ``store.rotate_sibling``'s healing re-ask, and the same absence of
        # assumption: the owner still decides on the dial.
        forget = getattr(self._client, "forget_refusal", None)
        if callable(forget):
            try:
                forget(self.key)
            except Exception:  # noqa: BLE001 — a cache repair must not fail a connect
                pass
        bearer = await asyncio.to_thread(self._blocking_bearer)
        if not bearer:
            return
        request.headers["Authorization"] = f"Bearer {bearer}"
        yield request
