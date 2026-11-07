"""The preview cache on its own: scope, TTLs, one flight per key, and its bounds.

These are unit tests of :mod:`local_operator.team_bridge.link_preview.cache` — no
server, no origin, no filesystem. The end-to-end proofs that a second account
cannot read the first account's artifact, that a repeated resolve does not fetch
and that concurrent resolves share one origin pass live in
``tests/unit/server/test_desktop_link_preview.py``, because those are properties of
the surface and only the route can show them.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from local_operator.team_bridge.link_preview.cache import (
    NEGATIVE_TTL_S,
    POSITIVE_TTL_S,
    PreviewCache,
    PreviewScope,
    artifact_id,
    canonical_url,
    key_for,
)
from local_operator.team_bridge.link_preview.models import LinkPreviewResolution

SCOPE = PreviewScope(account_id="acc_0123456789abcdef", tenant_id="ten_0123456789abcdef")


def _ready(url: str = "https://example.com/page") -> LinkPreviewResolution:
    # The cache never inspects the card beyond its ``kind``, so a url-only preview
    # is enough to pin the cache's own behaviour.
    return LinkPreviewResolution(kind="ready", preview={"url": url})  # type: ignore[arg-type]


def _absent() -> LinkPreviewResolution:
    return LinkPreviewResolution(kind="absent", reason="no-metadata")


def _error() -> LinkPreviewResolution:
    return LinkPreviewResolution(kind="error", retryable=True)


# ---------------------------------------------------------------------------
# What makes two links one link, and what makes an id one scope's id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        # Scheme and host are case-insensitive; the path is not.
        ("HTTPS://Example.COM/Path", "https://example.com/Path"),
        # A fragment never reaches the origin.
        ("https://example.com/x#section", "https://example.com/x"),
        # An explicit default port is the same socket as an implicit one.
        ("https://example.com:443/x", "https://example.com/x"),
        ("http://example.com:80/x", "http://example.com/x"),
        # A non-default port is part of the identity.
        ("https://example.com:8443/x", "https://example.com:8443/x"),
        # An empty path names the same resource as "/".
        ("https://example.com", "https://example.com/"),
        # The query SELECTS the page, so it is preserved EXACTLY as it is sent:
        # two keys may only meet when their bytes on the wire do (security S-8).
        ("https://example.com/p?b=2&a=1", "https://example.com/p?b=2&a=1"),
        ("https://example.com/p?a=1&b=2", "https://example.com/p?a=1&b=2"),
        ("https://example.com/r?to=\\\\evil.test/x", "https://example.com/r?to=\\\\evil.test/x"),
        (
            "https://example.com/r?to=%5C%5Cevil.test%2Fx",
            "https://example.com/r?to=%5C%5Cevil.test%2Fx",
        ),
        ("https://example.com/p?flag", "https://example.com/p?flag"),
        # httpx percent-encodes what it must, so these ARE the same request.
        ("https://example.com/p?a=\u00e9", "https://example.com/p?a=%C3%A9"),
        ("https://example.com/p?a b", "https://example.com/p?a%20b"),
        # Credentials are not part of what the page is, and must not become key
        # material that outlives the request they arrived with.
        ("https://user:sekret@example.com/x", "https://example.com/x"),
        # An IPv6 literal keeps its brackets.
        ("http://[::1]:8080/x", "http://[::1]:8080/x"),
    ],
)
def test_the_canonical_form_is_the_identity_of_a_link(raw: str, canonical: str) -> None:
    assert canonical_url(raw) == canonical


def test_the_key_carries_the_whole_scope() -> None:
    url = "https://example.com/p"
    base = key_for(SCOPE, url)
    assert key_for(PreviewScope(SCOPE.account_id), url) != base
    assert key_for(PreviewScope(SCOPE.account_id, "ten_other"), url) != base
    # The canonical form, not the raw spelling, is what the key holds.
    assert base.canonical_url == canonical_url(url)


def test_an_artifact_id_is_opaque_32_hex_and_separates_every_scope() -> None:
    url = "https://example.com/page"
    favicon = artifact_id(SCOPE, url, "favicon")

    assert re.fullmatch(r"[0-9a-f]{32}", favicon), favicon
    # Deterministic: the same scope, url and kind always name the same slot.
    assert favicon == artifact_id(SCOPE, url, "favicon")
    # Never the URL it was derived from.
    assert "example" not in favicon and "page" not in favicon
    # A different account, tenant, url or kind is a different slot — never a
    # collision that would let one scope's id resolve another scope's blob.
    assert favicon != artifact_id(PreviewScope("acc_other", SCOPE.tenant_id), url, "favicon")
    assert favicon != artifact_id(PreviewScope(SCOPE.account_id, "ten_other"), url, "favicon")
    assert favicon != artifact_id(SCOPE, "https://example.com/other", "favicon")
    assert favicon != artifact_id(SCOPE, url, "image")


# ---------------------------------------------------------------------------
# TTLs: what is reused, for how long, and what is never stored at all
# ---------------------------------------------------------------------------


def test_a_card_is_reused_inside_its_ttl_and_expires_after_it() -> None:
    now = [0.0]
    cache = PreviewCache(clock=lambda: now[0])
    key = key_for(SCOPE, "https://example.com/page")
    cache.put(key, _ready(), {"a" * 32: 10})

    assert cache.get(key) is not None
    now[0] = POSITIVE_TTL_S - 1
    assert cache.get(key) is not None, "a card inside its TTL was refetched"

    now[0] = POSITIVE_TTL_S + 1
    assert cache.get(key) is None, "an expired card was served"
    assert cache.stats().hits == 2
    assert cache.stats().misses == 1


def test_an_absent_answer_expires_sooner_than_a_card() -> None:
    now = [0.0]
    cache = PreviewCache(clock=lambda: now[0])
    key = key_for(SCOPE, "https://example.com/empty")
    cache.put(key, _absent(), {})

    now[0] = NEGATIVE_TTL_S - 1
    assert cache.get(key) is not None, "a negative answer inside its TTL was refetched"

    now[0] = NEGATIVE_TTL_S + 1
    assert cache.get(key) is None, "the negative TTL outlived the positive one's bound"
    assert NEGATIVE_TTL_S < POSITIVE_TTL_S


def test_an_error_is_never_cached() -> None:
    cache = PreviewCache()
    key = key_for(SCOPE, "https://example.com/broken")
    cache.put(key, _error(), {})
    assert cache.get(key) is None
    assert cache.stats().entries == 0


# ---------------------------------------------------------------------------
# One flight per key
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_callers_run_one_flight() -> None:
    cache = PreviewCache()
    key = key_for(SCOPE, "https://example.com/page")
    calls = 0

    async def run() -> LinkPreviewResolution:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02)
        return _ready()

    results = await asyncio.gather(*(cache.singleflight(key, run) for _ in range(5)))
    assert [result.kind for result in results] == ["ready"] * 5
    assert calls == 1, f"the flight ran {calls} times"
    assert cache.stats().resolutions == 1


@pytest.mark.asyncio
async def test_a_cancelled_waiter_does_not_cancel_the_flight() -> None:
    """The shield, proven: one renderer leaving must not take the answer with it."""

    cache = PreviewCache()
    key = key_for(SCOPE, "https://example.com/page")
    gate = asyncio.Event()
    calls = 0

    async def run() -> LinkPreviewResolution:
        nonlocal calls
        calls += 1
        await gate.wait()
        return _ready()

    survivor = asyncio.create_task(cache.singleflight(key, run))
    doomed = asyncio.create_task(cache.singleflight(key, run))
    await asyncio.sleep(0.02)

    doomed.cancel()
    with pytest.raises(asyncio.CancelledError):
        await doomed

    gate.set()
    result = await asyncio.wait_for(survivor, timeout=2.0)
    assert result.kind == "ready"
    assert calls == 1


@pytest.mark.asyncio
async def test_the_last_waiter_leaving_cancels_the_flight() -> None:
    """...and when nobody is left, the upstream work is not orphaned."""

    cache = PreviewCache()
    key = key_for(SCOPE, "https://example.com/page")
    cancelled = asyncio.Event()

    async def run() -> LinkPreviewResolution:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return _ready()

    waiter = asyncio.create_task(cache.singleflight(key, run))
    await asyncio.sleep(0.02)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await asyncio.wait_for(cancelled.wait(), timeout=2.0)


# ---------------------------------------------------------------------------
# Ownership, resident bytes and the two bounds
# ---------------------------------------------------------------------------


def test_an_artifact_belongs_to_exactly_one_scope() -> None:
    cache = PreviewCache()
    other_account = PreviewScope("acc_other", SCOPE.tenant_id)
    other_tenant = PreviewScope(SCOPE.account_id, "ten_other")
    key = key_for(SCOPE, "https://example.com/page")
    cache.put(key, _ready(), {"a" * 32: 10})

    assert cache.owns(SCOPE, "a" * 32) is True
    assert cache.owns(other_account, "a" * 32) is False
    assert cache.owns(other_tenant, "a" * 32) is False
    assert cache.owns(SCOPE, "b" * 32) is False


def test_stats_report_resident_bytes_and_nothing_identifiable() -> None:
    cache = PreviewCache()
    cache.put(
        key_for(SCOPE, "https://example.com/page?sender=secret"),
        _ready(),
        {"a" * 32: 400, "b" * 32: 600},
    )
    stats = cache.stats()
    assert stats.entries == 1
    assert stats.resident_bytes == 1000

    rendered = f"{stats}"
    assert "example" not in rendered
    assert SCOPE.account_id not in rendered
    assert SCOPE.tenant_id not in rendered
    # A negative answer holds no artifacts, so it costs nothing resident.
    cache.put(key_for(SCOPE, "https://example.com/empty"), _absent(), {})
    assert cache.stats().resident_bytes == 1000


def test_the_least_recently_used_entry_is_the_one_evicted() -> None:
    cache = PreviewCache(max_entries=2)
    first = key_for(SCOPE, "https://example.com/1")
    second = key_for(SCOPE, "https://example.com/2")
    third = key_for(SCOPE, "https://example.com/3")
    cache.put(first, _ready(), {"1" * 32: 10})
    cache.put(second, _ready(), {"2" * 32: 10})

    # Read the oldest so it is no longer the coldest.
    assert cache.get(first) is not None
    cache.put(third, _ready(), {"3" * 32: 10})

    assert cache.get(second) is None, "the coldest entry survived"
    assert cache.get(first) is not None
    # Eviction drops the artifact mapping with the entry: an id nothing names is
    # one this process cannot serve, which is the point of dropping it.
    assert cache.owns(SCOPE, "2" * 32) is False
    assert cache.owns(SCOPE, "1" * 32) is True
    assert cache.stats().entries == 2
    assert cache.stats().evictions == 1


def test_the_resident_byte_cap_evicts_until_the_cache_fits() -> None:
    cache = PreviewCache(max_bytes=100)
    kept = key_for(SCOPE, "https://example.com/kept")
    dropped = key_for(SCOPE, "https://example.com/dropped")
    cache.put(dropped, _ready(), {"d" * 32: 60})
    assert cache.stats().resident_bytes == 60

    cache.put(kept, _ready(), {"k" * 32: 60})
    stats = cache.stats()
    assert stats.resident_bytes == 60, "the byte cap did not evict"
    assert stats.evictions == 1
    assert cache.owns(SCOPE, "k" * 32) is True
    assert cache.owns(SCOPE, "d" * 32) is False


def test_replacing_an_entry_does_not_double_count_its_bytes() -> None:
    cache = PreviewCache()
    key = key_for(SCOPE, "https://example.com/page")
    cache.put(key, _ready(), {"a" * 32: 500})
    cache.put(key, _ready(), {"a" * 32: 700})
    assert cache.stats().entries == 1
    assert cache.stats().resident_bytes == 700
