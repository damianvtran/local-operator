"""What a previewed URL said, remembered PER SCOPE and never across accounts.

WHY THE KEY CARRIES THE ACCOUNT AND THE TENANT. A card is not a public fact: it
was resolved because one correspondent sent one link to one reader, and the
bytes behind it were fetched by a session that holds credentials. Sharing that
record between accounts would mean account B's renderer receiving an artifact id
that account A's message produced — the id would then be a capability one
account could replay into another session. So the key is the whole
``(account_id, tenant_id, canonical URL)`` triple, the artifact ids are derived
from the same triple, and a lookup for another scope is a MISS that fetches for
itself rather than a hit that leaks.

WHY THE CANONICAL FORM IS ONLY A KEY. Two links that differ in case or in the
order of their query parameters name one page, so they share one entry. What is
fetched and what is returned stay the URL the reader was shown: a cache that
rewrote the request (or the card's link) to its own normalised spelling would be
answering a question nobody asked. See :func:`canonical_url`.

WHAT IS CACHED, AND FOR HOW LONG. A ``ready`` card lives for
:data:`POSITIVE_TTL_S`; an ``absent`` answer — "this page has nothing to draw" —
lives for :data:`NEGATIVE_TTL_S`, which is much shorter because that judgement is
made from markup that changes far more often than a page's existence does. An
``error`` is never stored: it is a statement about the network at one instant
(a 503 mid-rollout, a dropped lookup), and caching it would pin one bad moment
onto every later reader of the same link.

BOUNDED IN TWO DIMENSIONS. Entries are LRU with a count cap and a resident-byte
cap, because both are real: a thousand small favicons and one 2 MB hero image
fail the same process in different ways. Eviction drops the entry AND its
artifact mapping, which is what makes the artifact route stop serving those
bytes — the blob may still be on disk until the store's own sweep, but nothing
in this process can name it any more.

ONE FLIGHT PER KEY. Concurrent resolves of one URL perform ONE fetch: the
shielded singleflight is the pattern
:meth:`~local_operator.web_search.io.WebReadIO.singleflight` uses, and for the
same reason — a renderer that navigates away mid-request must not cancel a fetch
other waiters are still holding.

NOTHING HERE CAN REVEAL A PAGE. :meth:`PreviewCache.stats` reports counts and
bytes and nothing else: no account id, no URL, no title. That is the
resident-size report the surface needs, and it is served read-only by
``GET /v1/desktop/team-bridge/link-preview/cache`` — a route that takes no
account receipt because the answer names no account.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable, Coroutine, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from local_operator.team_bridge.link_preview.models import LinkPreviewResolution

#: How long a resolved card is reused. Long enough that scrolling a channel back
#: does not re-fetch every card, short enough that a page redesigned this morning
#: shows its new title this evening.
POSITIVE_TTL_S: float = 6 * 60 * 60

#: How long an ``absent`` answer is reused. Shorter than the positive TTL on
#: purpose: "no metadata" is judged from markup, and markup is the part of a site
#: that changes weekly.
NEGATIVE_TTL_S: float = 30 * 60

#: Entry bound. A preview is at most a title, a description and two ids; 256 of
#: them is a channel's worth of scrollback without being an unbounded map.
MAX_ENTRIES: Final = 256

#: Resident-byte bound covering the artifact bytes the cache retains. Capped well
#: under the on-disk store's own budget so the in-memory accounting is the tighter
#: of the two: a cap the cache could never enforce would be decoration.
MAX_RESIDENT_BYTES: Final = 32 * 1024 * 1024

#: Characters of hex an artifact id is. The renderer's contract calls the id
#: opaque, 32 lowercase hex, sha256-derived and never the URL it came from; this
#: constant is the one place that length is decided.
ARTIFACT_ID_HEX_CHARS: Final = 32

#: The ports a URL may leave implicit. Dropped from the canonical form so that
#: ``https://a.test:443/x`` and ``https://a.test/x`` are one entry.
DEFAULT_PORTS: Final = {"http": 80, "https": 443}

#: Between the fields of an artifact id's digest. Without it ``("ab", "c")`` and
#: ``("a", "bc")`` would hash identically; a NUL cannot appear in an account id,
#: a tenant id, a URL or a kind, so it can never be the ambiguity back.
_FIELD_SEPARATOR: Final = b"\x00"


@dataclass(frozen=True)
class PreviewScope:
    """Whose preview this is: one account, inside one tenant.

    ``tenant_id`` is empty for an install with no committed Team Bridge owner.
    The account alone still isolates in that case, which is the property that
    matters: a missing tenant must narrow nothing.
    """

    account_id: str
    tenant_id: str = ""


@dataclass(frozen=True)
class PreviewKey:
    """One cached answer's identity: the scope and the canonical URL."""

    account_id: str
    tenant_id: str
    canonical_url: str


@dataclass(frozen=True)
class PreviewCacheStats:
    """What the cache is holding, in numbers that cannot name anything.

    ``resolutions`` counts passes through the resolver — one per cache miss that
    reached the origin — not HTTP requests: a card's own favicon and image are
    further requests inside one pass.
    """

    entries: int
    resident_bytes: int
    hits: int
    misses: int
    evictions: int
    resolutions: int


@dataclass
class _Flight:
    """One in-progress resolution and the number of callers waiting on it.

    The same shape :class:`~local_operator.web_search.io.WebReadIO` uses: the
    count decides whether the last waiter leaving should cancel the work.
    """

    task: asyncio.Task[LinkPreviewResolution]
    users: int = 0


@dataclass(frozen=True)
class _Entry:
    """One cached preview, its artifacts, and when they stop being reused."""

    resolution: LinkPreviewResolution
    artifacts: Mapping[str, int] = field(default_factory=dict)
    stored_at: float = 0.0
    ttl_s: float = 0.0

    @property
    def resident_bytes(self) -> int:
        return sum(self.artifacts.values())

    def fresh(self, now: float) -> bool:
        return now - self.stored_at < self.ttl_s


def canonical_url(url: str) -> str:
    """The identity of ``url`` for caching: what makes two links one link.

    Case in the scheme and host is not case-sensitive, a fragment never reaches
    the origin, and an explicit default port names the same socket as an implicit
    one, so all three are normalised away. The query is PRESERVED — it routinely
    selects the page, and dropping it would answer for the wrong document — and
    its parameters are ordered so ``?b=2&a=1`` and ``?a=1&b=2`` share one entry.

    USERINFO IS DROPPED. ``https://user:pass@host/x`` is the same resource as
    ``https://host/x``; keeping credentials in a cache key would put them in a
    structure that outlives the request they arrived with, which is the one thing
    this package must never do.

    THAT DROP IS NOT ITSELF A GUARD, AND IT IS REACHABLE — which is why the refusal
    has to come first (QA Q2-1). Dropping the pair is what makes the two spellings
    ONE KEY here, so a credentialed URL that reached this function would read the
    plain URL's entry, and on a miss would write its own ``blocked-host`` back
    under that shared key: one message's malformed link then denying the card the
    plain URL is entitled to for the whole negative TTL. The address guard in the
    fetcher prevents neither half, because a cache HIT never reaches the fetcher at
    all. So the service refuses credentials on the RAW url before it keys anything,
    and the engine-level refusal in ``validate_public_url`` stays as the depth
    defence that also covers every redirect hop.

    The normalisation itself stays regardless, because a key function must be
    total: it runs on whatever the renderer sent, and a key that carried a
    credential into a long-lived structure would be a leak even on a path nothing
    reaches yet.

    This value is a KEY and nothing else: it is never fetched and never returned.
    """

    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    try:
        port = parts.port
    except ValueError:
        # A malformed port is the fetcher's refusal to make, not ours; the entry
        # it produces is one the fetch never fills.
        port = None
    if port is not None and DEFAULT_PORTS.get(scheme) == port:
        port = None
    # An IPv6 literal must keep its brackets or the netloc is ambiguous.
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc = f"{netloc}:{port}"
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, parts.path or "/", query, ""))


def key_for(scope: PreviewScope, url: str) -> PreviewKey:
    """The cache key for ``url`` seen through ``scope``."""

    return PreviewKey(
        account_id=scope.account_id,
        tenant_id=scope.tenant_id,
        canonical_url=canonical_url(url),
    )


def artifact_id(scope: PreviewScope, canonical: str, kind: str) -> str:
    """The id a renderer receives for one artifact of one scoped preview.

    Derived from the scope and the canonical URL rather than from the bytes, so
    the id IS the ownership claim: account B composing its own digest for account
    A's link lands on B's own id, and there is no id it can construct that names
    A's blob. ``kind`` separates a favicon from an image of the same page.

    Truncated to :data:`ARTIFACT_ID_HEX_CHARS` per the renderer's contract. 128
    bits of sha256 is far beyond what a guess could reach, and the value is never
    used as a security boundary of its own: the artifact route asks this cache
    whether the caller's scope owns the id, and the store will not read a blob it
    does not hold.
    """

    digest = hashlib.sha256()
    for value in (scope.account_id, scope.tenant_id, canonical, kind):
        digest.update(value.encode("utf-8"))
        digest.update(_FIELD_SEPARATOR)
    return digest.hexdigest()[:ARTIFACT_ID_HEX_CHARS]


class PreviewCache:
    """Bounded, scoped, TTL'd memory of what previews said, with one flight per key.

    Process-wide in production (one ``lop serve`` serves one config root), but
    constructed rather than imported so a test can inject its own clock and
    bounds instead of sleeping for six hours.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_entries: int = MAX_ENTRIES,
        max_bytes: int = MAX_RESIDENT_BYTES,
        positive_ttl_s: float = POSITIVE_TTL_S,
        negative_ttl_s: float = NEGATIVE_TTL_S,
    ) -> None:
        self._clock = clock
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._positive_ttl_s = positive_ttl_s
        self._negative_ttl_s = negative_ttl_s
        self._entries: OrderedDict[PreviewKey, _Entry] = OrderedDict()
        self._flights: dict[PreviewKey, _Flight] = {}
        self._resident_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._resolutions = 0

    def get(self, key: PreviewKey) -> LinkPreviewResolution | None:
        """The cached answer for ``key`` if it is still fresh, else ``None``.

        A stale entry is left in place rather than dropped: the refetch that
        follows replaces it, and if that refetch errors (which is not cached) the
        old card keeps rendering instead of the surface losing the image it had.
        """

        entry = self._entries.get(key)
        if entry is None or not entry.fresh(self._clock()):
            self._misses += 1
            return None
        self._entries.move_to_end(key)
        self._hits += 1
        return entry.resolution

    def put(
        self,
        key: PreviewKey,
        resolution: LinkPreviewResolution,
        artifacts: Mapping[str, int] | None = None,
    ) -> None:
        """Remember ``resolution`` under ``key``, evicting whatever no longer fits.

        ``artifacts`` maps each artifact id the card carries to the bytes the
        cache is holding for it, which is what the resident-byte cap is measured
        against. An ``error`` is stored nowhere: see the module docstring.
        """

        if resolution.kind == "error":
            return
        ttl = self._positive_ttl_s if resolution.kind == "ready" else self._negative_ttl_s
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._resident_bytes -= previous.resident_bytes
        entry = _Entry(
            resolution=resolution,
            artifacts=dict(artifacts or {}),
            stored_at=self._clock(),
            ttl_s=ttl,
        )
        self._entries[key] = entry
        self._resident_bytes += entry.resident_bytes
        self._trim()

    def owns(self, scope: PreviewScope, artifact_id: str) -> bool:
        """Whether ``scope`` has a cached card carrying ``artifact_id``.

        This is the artifact route's whole authorization check, and it is an
        approval rather than a lookup: an id is served only while some entry of
        the CALLER's scope names it. Serving a hit refreshes that entry's LRU
        position, because a renderer loading a card's images is exactly the use
        the ordering is meant to track.

        Freshness is deliberately not part of it: the TTL decides when a card is
        refetched, not when the bytes a card already points at are revoked.
        """

        for key, entry in self._entries.items():
            if (
                key.account_id == scope.account_id
                and key.tenant_id == scope.tenant_id
                and artifact_id in entry.artifacts
            ):
                self._entries.move_to_end(key)
                return True
        return False

    def stats(self) -> PreviewCacheStats:
        """Counts and resident bytes, never an account id or a URL."""

        return PreviewCacheStats(
            entries=len(self._entries),
            resident_bytes=self._resident_bytes,
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            resolutions=self._resolutions,
        )

    async def singleflight(
        self, key: PreviewKey, run: Callable[[], Coroutine[object, object, LinkPreviewResolution]]
    ) -> LinkPreviewResolution:
        """Run ``run`` once for every caller holding ``key`` at the same time.

        THE SHIELD IS LOAD-BEARING. Each caller awaits the flight through
        :func:`asyncio.shield`, so one renderer navigating away cancels only its
        own await; the fetch continues for the others and its result is stored
        for them. When the LAST waiter leaves the flight is cancelled, because at
        that point nobody is left to want the answer and an abandoned request is
        just work nobody reads.

        Mirrors :meth:`local_operator.web_search.io.WebReadIO.singleflight`; the
        two are separate because they own different lifetimes (that one is
        session-owned, this one process-wide), not because the rule differs.
        """

        flight = self._flights.get(key)
        if flight is None:
            flight = _Flight(asyncio.create_task(run()))
            self._flights[key] = flight
            # One flight is one pass through the resolver, i.e. one round of
            # origin fetches. Counted here because this is where the decision to
            # reach the origin is actually taken.
            self._resolutions += 1
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

    def _trim(self) -> None:
        """Evict least-recently-used entries until both bounds hold again.

        Both caps are checked in one loop so a burst of writes pays for one pass.
        Eviction removes the entry AND its artifact mapping: an id no entry names
        is one this process cannot serve, which is the point of dropping it.
        """

        while self._entries and (
            len(self._entries) > self._max_entries or self._resident_bytes > self._max_bytes
        ):
            _key, entry = self._entries.popitem(last=False)
            self._resident_bytes -= entry.resident_bytes
            self._evictions += 1

    def keys(self) -> Iterator[PreviewKey]:
        """The keys currently held, oldest first. For tests and diagnostics only."""

        return iter(self._entries)


#: The process-wide cache, built on first use.
#:
#: A ``lop serve`` process serves exactly one config root, so "one cache per
#: process" and "one cache per install" are the same statement — and a restart
#: starting empty is correct: the entries would otherwise name artifact blobs
#: from a store whose process is gone.
_PROCESS_CACHE: PreviewCache | None = None


def get_preview_cache() -> PreviewCache:
    """The one cache this process has.

    A function rather than a module-level instance so tests can inject a cache
    with their own clock through the same seam the route reads.
    """

    global _PROCESS_CACHE
    if _PROCESS_CACHE is None:
        _PROCESS_CACHE = PreviewCache()
    return _PROCESS_CACHE
