"""One long-lived HTTP connection pool for Team Bridge hub calls, per event loop.

WHY THIS EXISTS (bridge-instant-load, 2026-11). ``TeamBridgeClient._dispatch``
used to open a fresh ``httpx.AsyncClient`` for every hub request. Every desktop
bridge read (status, directory, conversations, messages) therefore paid a new
TCP + TLS handshake to the hub: measured 41-124 ms per request against
``api.radienthq.com``, and a kept-alive messages read was 70 ms against 194 ms
cold. Opening Team Bridge fires several of those reads at once, so the
handshakes were a large share of the time-to-first-paint the operator felt.

WHO OWNS A POOL. A pool exists only on an event loop that an OWNER bound one to
(:func:`bind_hub_pool`) — the ``lop serve`` lifespan binds one for the server
loop and closes it at shutdown. Any other loop (a CLI command's
``asyncio.run``, a tool call, a test) has no owner, so :func:`hub_http_client`
falls back to the old per-request client, closed when the request ends. That is
deliberate rather than a missing feature: an un-owned loop has nobody to call
``aclose()``, and a client whose loop has already exited cannot close its
sockets at all — exactly the leak the per-request shape never had. A client is
NEVER shared across loops: httpx/anyio connections are bound to the loop that
opened them, so the registry is keyed by the loop object itself (weakly, so a
dead loop's entry cannot pin it).

INVARIANTS (each pinned by ``tests/team_bridge/test_http_pool.py``):

* The bearer travels PER REQUEST only — ``_dispatch`` sets ``Authorization`` on
  each request's own headers and the pooled client is built with NO default
  headers. That is also what makes a pool safe across a re-sign-in or an
  account change: the client holds no credential, so the next request simply
  carries the new row's token over the same sockets. Nothing to reset.
* No cookies, ever. httpx keeps a client-wide cookie jar and would replay a
  hub ``Set-Cookie`` on the next request — harmless per request, a cross-call
  (and potentially cross-account) state channel once the client lives for the
  whole server. The jar's policy refuses to store or return any cookie, which
  holds even for two concurrent requests (clearing after each response would
  race).
* ``follow_redirects=False`` exactly as before: a redirect is a refusal the
  caller maps, never a hop that could carry the bearer elsewhere.
* Timeouts stay per request (``_dispatch`` passes its own ``timeout`` to every
  call), so sharing a client never changes one operation's deadline.
* Response-size caps and the forced identity ``Accept-Encoding`` stay in
  ``_dispatch``: they are request-level, untouched by pooling.
* An injected test transport keys its own client, so a ``MockTransport`` test
  is never served by the production transport or another test's mock.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import weakref
from collections.abc import AsyncIterator
from http.cookiejar import CookieJar, DefaultCookiePolicy
from typing import Any

import httpx

logger = logging.getLogger(__name__)

#: How long an idle hub connection stays in the pool. The desktop re-reads
#: status/conversations on focus and on its refetch cadence; httpx's 5 s default
#: would drop the socket between most of those reads and re-pay the handshake.
#: 30 s stays well under the common 60 s load-balancer idle timeout, and httpcore
#: discards a socket the server already closed before reusing it.
KEEPALIVE_EXPIRY_S = 30.0


class _RefuseAllCookies(DefaultCookiePolicy):
    """A cookie policy that neither stores nor returns any cookie.

    The hub authenticates by bearer alone; any ``Set-Cookie`` it (or a proxy in
    front of it) sends is state we must not replay on a later request.
    """

    def set_ok(self, cookie: Any, request: Any) -> bool:  # noqa: ARG002
        return False

    def return_ok(self, cookie: Any, request: Any) -> bool:  # noqa: ARG002
        return False


def _cookieless_jar() -> CookieJar:
    return CookieJar(policy=_RefuseAllCookies())


def build_hub_client(transport: httpx.AsyncBaseTransport | None) -> httpx.AsyncClient:
    """The one client shape every hub request uses, pooled or per request.

    No ``headers``, no ``base_url``, no ``auth``: everything credentialed or
    addressed is supplied by ``_dispatch`` per request (see module invariants).
    """

    kwargs: dict[str, Any] = {
        "follow_redirects": False,
        "cookies": _cookieless_jar(),
    }
    if transport is not None:
        kwargs["transport"] = transport
    else:
        kwargs["limits"] = httpx.Limits(keepalive_expiry=KEEPALIVE_EXPIRY_S)
    return httpx.AsyncClient(**kwargs)


class HubHttpPool:
    """The long-lived clients of one event loop, one per (base URL, transport).

    Base URL is part of the key so a ``RADIENT_API_BASE_URL`` override (or a
    test hub) never shares sockets with another deployment; the transport is
    part of it so an injected mock gets its own client.
    """

    def __init__(self) -> None:
        self._clients: dict[tuple[str, httpx.AsyncBaseTransport | None], httpx.AsyncClient] = {}
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def client(
        self, base_url: str, transport: httpx.AsyncBaseTransport | None
    ) -> httpx.AsyncClient:
        # Synchronous on purpose: no await between the lookup and the insert,
        # so two concurrent first requests on the loop cannot build two clients.
        key = (base_url, transport)
        existing = self._clients.get(key)
        if existing is not None and not existing.is_closed:
            return existing
        created = build_hub_client(transport)
        self._clients[key] = created
        return created

    async def aclose(self) -> None:
        self._closed = True
        clients, self._clients = list(self._clients.values()), {}
        for http in clients:
            try:
                await http.aclose()
            except Exception:  # noqa: BLE001 — shutdown must never fail on a socket close
                logger.debug("hub http client close failed", exc_info=True)


_POOLS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, HubHttpPool] = (
    weakref.WeakKeyDictionary()
)


def bind_hub_pool() -> HubHttpPool:
    """Give the RUNNING loop a hub pool; the caller owns ``await pool.aclose()``.

    Idempotent per loop: a second bind returns the live pool. Call
    :func:`unbind_hub_pool` (or use :func:`owned_hub_pool`) at shutdown.
    """

    loop = asyncio.get_running_loop()
    pool = _POOLS.get(loop)
    if pool is None or pool.closed:
        pool = HubHttpPool()
        _POOLS[loop] = pool
    return pool


async def unbind_hub_pool() -> None:
    """Close and forget the running loop's pool; later calls go per request."""

    loop = asyncio.get_running_loop()
    pool = _POOLS.pop(loop, None)
    if pool is not None:
        await pool.aclose()


@contextlib.asynccontextmanager
async def owned_hub_pool() -> AsyncIterator[HubHttpPool]:
    """Bind a pool for the running loop for the duration of the block."""

    pool = bind_hub_pool()
    try:
        yield pool
    finally:
        await unbind_hub_pool()


@contextlib.asynccontextmanager
async def hub_http_client(
    base_url: str, transport: httpx.AsyncBaseTransport | None
) -> AsyncIterator[httpx.AsyncClient]:
    """The client one hub request should use on the running loop.

    Pooled (and left open) when this loop has an owner; otherwise a fresh
    client closed when the block exits — the pre-pool behaviour, kept for
    loops nobody will ever ``aclose()``.
    """

    try:
        loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    pool = _POOLS.get(loop) if loop is not None else None
    if pool is not None and not pool.closed:
        yield pool.client(base_url, transport)
        return
    async with build_hub_client(transport) as http:
        yield http
