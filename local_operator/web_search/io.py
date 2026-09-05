"""Session-owned HTTP pooling and cancellation-safe duplicate read collapse.

No process globals: sessions, credentials and event loops have independent
lifetimes. Owners are closed by Session.dispose. A temporary owner gives CLI
and embedding callers identical semantics without leaving connections alive.
Fetch keys include the *original* origin, not just the vetted destination IP:
virtual hosts sharing an IP must never share a TLS/SNI connection accidentally.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from http.cookiejar import Cookie, CookieJar, DefaultCookiePolicy
from typing import Any, TypeVar

import httpx

T = TypeVar("T")


class _NoCookies(DefaultCookiePolicy):
    """Pooling is a transport optimization, not a new browser login session."""

    def set_ok(self, cookie: Cookie, request: Any) -> bool:
        return False


@dataclass
class _Client:
    value: httpx.AsyncClient
    users: int = 0


@dataclass
class _Flight:
    task: asyncio.Task[Any]
    users: int = 0


class WebReadIO:
    """Bounded connection ownership; successful reads may be shared in flight.

    Results are not retained here: fetch already has a TTL/refresh-aware spill
    cache, while searches should reflect fresh indexes on sequential requests.
    One cancelled subscriber leaves other subscribers running. Cancelling the
    final subscriber cancels the upstream request too, avoiding orphaned work.
    """

    def __init__(self) -> None:
        self._clients: OrderedDict[tuple[Any, ...], _Client] = OrderedDict()
        self._flights: dict[tuple[Any, ...], _Flight] = {}
        self._closed = False

    @asynccontextmanager
    async def client(self, key: tuple[Any, ...], **kwargs: Any) -> AsyncIterator[httpx.AsyncClient]:
        if self._closed:
            raise RuntimeError("web I/O owner is closed")
        entry = self._clients.get(key)
        if entry is None:
            entry = _Client(httpx.AsyncClient(cookies=CookieJar(policy=_NoCookies()), **kwargs))
            self._clients[key] = entry
        self._clients.move_to_end(key)
        entry.users += 1
        try:
            yield entry.value
        finally:
            entry.users -= 1
            # Only idle clients may be evicted. Active request concurrency is
            # bounded by the tool scheduler; an LRU eviction must not break it.
            for stale_key, stale in list(self._clients.items()):
                if len(self._clients) <= 32:
                    break
                if not stale.users:
                    self._clients.pop(stale_key)
                    await stale.value.aclose()

    async def singleflight(self, key: tuple[Any, ...], run: Callable[[], Awaitable[T]]) -> T:
        if self._closed:
            raise RuntimeError("web I/O owner is closed")
        flight = self._flights.get(key)
        if flight is None:

            async def invoke() -> T:
                return await run()

            flight = _Flight(asyncio.create_task(invoke()))
            self._flights[key] = flight
        flight.users += 1
        try:
            return await asyncio.shield(flight.task)
        finally:
            flight.users -= 1
            if not flight.users:
                if self._flights.get(key) is flight:
                    self._flights.pop(key)
                if not flight.task.done():
                    flight.task.cancel()
                await asyncio.gather(flight.task, return_exceptions=True)

    async def aclose(self) -> None:
        self._closed = True
        tasks = [flight.task for flight in self._flights.values()]
        self._flights.clear()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        clients, self._clients = self._clients, OrderedDict()
        await asyncio.gather(
            *(entry.value.aclose() for entry in clients.values()), return_exceptions=True
        )
