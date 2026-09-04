"""The shared usage cache — TTL, stale serving, leases, and the wire format.

The cache is what makes `/usage` instant across every lop session on the
machine, so its contract is tested directly against a temp file: a fresh row
answers, an expired row does not (but stays servable), a lease serializes
refreshes, and a failure keeps the last good value readable.
"""

from __future__ import annotations

import time

import pytest

from local_operator.providers.usage import UsageAmount, UsageLimit, UsageReport
from local_operator.providers.usage_cache import (
    USAGE_ACCOUNT_BACKOFF_CAP_MS,
    USAGE_FAILURE_BACKOFF_MS,
    USAGE_LAST_GOOD_RETENTION_MS,
    UsageCacheStore,
    account_backoff_ms,
    account_preflight_key,
    fingerprint_accounts,
    fingerprint_secret,
    leased_account_usage,
    provider_cache_key,
    report_from_dict,
    report_to_dict,
)


def _report(provider: str = "anthropic", percent: float = 42.0) -> UsageReport:
    return UsageReport(
        provider=provider,
        fetched_at=int(time.time() * 1000),
        limits=[
            UsageLimit(
                id=f"{provider}:5h",
                label="5 hour",
                amount=UsageAmount(
                    used=percent, limit=100.0, used_fraction=percent / 100, unit="percent"
                ),
                window="5 hour",
                shared=True,
            )
        ],
        identity="me@example.com",
    )


@pytest.fixture
def cache(tmp_path):
    store = UsageCacheStore(tmp_path / "usage_cache.db")
    yield store
    store.close()


# -- wire format --------------------------------------------------------------
def test_report_round_trips_through_the_wire_format() -> None:
    report = _report()
    restored = report_from_dict(report_to_dict(report))
    assert restored is not None
    assert restored.provider == report.provider
    assert restored.fetched_at == report.fetched_at
    assert restored.identity == report.identity
    assert len(restored.limits) == 1
    limit = restored.limits[0]
    assert limit.id == "anthropic:5h"
    assert limit.amount.used_fraction == pytest.approx(0.42)
    assert limit.shared is True


def test_report_from_dict_rejects_garbage_as_a_miss() -> None:
    # A schema change or a corrupt row must read as a cache MISS, never an
    # exception on the /usage path.
    assert report_from_dict(None) is None
    assert report_from_dict({"no": "provider"}) is None
    assert report_from_dict({"provider": "x", "limits": [{"id": "a"}]}) is None


# -- keys ---------------------------------------------------------------------
def test_cache_key_names_provider_and_account_set() -> None:
    fp = fingerprint_accounts(["a@x.com", "b@x.com"])
    assert provider_cache_key("anthropic", fp) == f"anthropic:{fp}"


def test_fingerprint_is_order_independent() -> None:
    # Enumeration order (row id, round-robin) must not change the key.
    assert fingerprint_accounts(["a", "b"]) == fingerprint_accounts(["b", "a"])
    assert fingerprint_accounts([]) == "none"


def test_account_backoff_doubles_then_caps() -> None:
    assert account_backoff_ms(0) == 0
    assert account_backoff_ms(1) == 10_000
    assert account_backoff_ms(2) == 20_000
    assert account_backoff_ms(3) == 40_000
    assert account_backoff_ms(4) == 80_000
    assert account_backoff_ms(5) == 160_000
    assert account_backoff_ms(8) == USAGE_ACCOUNT_BACKOFF_CAP_MS


def test_report_round_trip_keeps_per_account_failure_state() -> None:
    report = _report()
    report.consecutive_failures = 3
    report.usage_unavailable = True
    report.next_probe_at_ms = 1_700_000_000_000
    restored = report_from_dict(report_to_dict(report))
    assert restored is not None
    assert restored.consecutive_failures == 3
    assert restored.usage_unavailable is True
    assert restored.next_probe_at_ms == 1_700_000_000_000


def test_report_round_trip_keeps_the_dead_grant_verdict() -> None:
    """The panel reads cached rows, so a flag that did not survive the round
    trip would show the re-login note only in the session that first saw the
    refusal — and `usage unavailable` in every session after it."""
    report = _report()
    report.credential_invalid = True
    restored = report_from_dict(report_to_dict(report))
    assert restored is not None
    assert restored.credential_invalid is True


def test_a_row_written_before_the_flag_existed_reads_as_valid() -> None:
    """The cache outlives the version that wrote it: an older row must be a
    plain report, never a spurious "sign in again"."""
    data = report_to_dict(_report())
    data.pop("credential_invalid", None)
    restored = report_from_dict(data)
    assert restored is not None
    assert restored.credential_invalid is False


def test_secret_fingerprint_never_contains_the_secret() -> None:
    fp = fingerprint_secret("sk-or-very-secret")
    assert "sk-or-very-secret" not in fp
    assert fingerprint_secret("sk-or-very-secret") == fp  # stable


# -- TTL and stale serving ----------------------------------------------------
def test_a_fresh_row_is_served_and_an_expired_one_is_not(cache) -> None:
    key = "anthropic:test"
    cache.set(key, "anthropic", [_report()], expires_at_ms=int(time.time() * 1000) + 60_000)
    assert cache.get(key) is not None

    cache.set(key, "anthropic", [_report()], expires_at_ms=int(time.time() * 1000) - 1)
    assert cache.get(key) is None
    # ...but stays servable as the stale fallback.
    assert cache.get(key, include_expired=True) is not None


def test_latest_account_report_slices_the_warmer_row_per_account(cache) -> None:
    """The first-resolve pick reads one account out of the per-provider-set
    payload: newest ``fetched_at`` among matching identities wins, rows of a
    different provider and ``:pf:`` preflight rows are ignored, and expiry
    does not matter (the caller applies its own age policy)."""
    now = int(time.time() * 1000)
    old_a = _report(percent=10.0)
    old_a.fetched_at = now - 60_000
    new_a = _report(percent=70.0)
    new_a.fetched_at = now - 1_000
    b = _report(percent=50.0)
    b.identity = "other@example.com"
    # Two fingerprints of the same provider (the account set changed once);
    # the older row is expired, which must not hide it.
    cache.set(
        provider_cache_key("anthropic", "fp-old"), "anthropic", [old_a, b], expires_at_ms=now - 1
    )
    cache.set(
        provider_cache_key("anthropic", "fp-new"), "anthropic", [new_a], expires_at_ms=now + 60_000
    )
    # Same identity under another provider must not bleed across.
    stray = _report(provider="openai", percent=99.0)
    stray.fetched_at = now
    cache.set(provider_cache_key("openai", "fp"), "openai", [stray], expires_at_ms=now + 60_000)
    # A preflight row under the anthropic prefix carrying the identity.
    pf = _report(percent=99.0)
    pf.fetched_at = now
    cache.set(
        account_preflight_key("anthropic", "me@example.com"),
        "anthropic",
        [pf],
        expires_at_ms=now + 60_000,
    )

    found = cache.latest_account_report("anthropic", {"me@example.com"})
    assert found is not None
    assert found.fetched_at == new_a.fetched_at
    assert found.limits[0].amount.used_fraction == 0.7
    other = cache.latest_account_report("anthropic", {"other@example.com", "cred:9"})
    assert other is not None and other.identity == "other@example.com"
    assert cache.latest_account_report("anthropic", {"nobody@example.com"}) is None
    assert cache.latest_account_report("anthropic", set()) is None


def test_expiry_ms_reports_the_rows_ttl(cache) -> None:
    key = "anthropic:test"
    assert cache.expiry_ms(key) is None
    expires = int(time.time() * 1000) + 60_000
    cache.set(key, "anthropic", [_report()], expires_at_ms=expires)
    assert cache.expiry_ms(key) == expires


def test_multiple_reports_round_trip_as_one_row(cache) -> None:
    # A provider with two accounts stores BOTH reports under one key.
    key = "anthropic:two"
    reports = [_report(percent=10.0), _report(percent=90.0)]
    cache.set(key, "anthropic", reports, expires_at_ms=int(time.time() * 1000) + 60_000)
    restored = cache.get(key)
    assert restored is not None
    assert len(restored) == 2
    assert restored[0].limits[0].amount.used == 10.0
    assert restored[1].limits[0].amount.used == 90.0


# -- failure fallback ----------------------------------------------------------
def test_write_failure_keeps_the_last_good_value_readable(cache) -> None:
    key = "anthropic:test"
    cache.set(key, "anthropic", [_report()], expires_at_ms=int(time.time() * 1000) + 60_000)
    # Expire it, then record a failure: the stale value must come back with a
    # short cool-down so a blip does not blank the report.
    cache.set(key, "anthropic", [_report()], expires_at_ms=int(time.time() * 1000) - 1)
    last_good = cache.write_failure(key, "anthropic")
    assert last_good is not None
    # Fresh again (within the backoff window).
    assert cache.get(key) is not None
    assert cache.expiry_ms(key) is not None
    assert cache.expiry_ms(key) <= int(time.time() * 1000) + USAGE_FAILURE_BACKOFF_MS + 5


def test_write_failure_with_no_history_returns_none(cache) -> None:
    assert cache.write_failure("anthropic:never", "anthropic") is None


# -- leases ---------------------------------------------------------------------
def test_the_fetch_lease_serializes_concurrent_refreshers(cache) -> None:
    key = "anthropic:test"
    assert cache.try_lease(key) is True
    # A second refresher in the same window loses the lease.
    other = UsageCacheStore(cache._db_path)
    try:
        assert other.try_lease(key) is False
    finally:
        other.close()
    # Releasing hands it back.
    cache.release_lease(key)
    assert other.try_lease(key) is True


def test_an_expired_lease_can_be_retaken(cache) -> None:
    key = "anthropic:test"
    assert cache.try_lease(key, ttl_ms=1) is True
    time.sleep(0.01)
    other = UsageCacheStore(cache._db_path)
    try:
        assert other.try_lease(key) is True
    finally:
        other.close()


def test_only_the_holder_releases_a_lease(cache, tmp_path) -> None:
    key = "anthropic:test"
    assert cache.try_lease(key) is True
    other = UsageCacheStore(cache._db_path)
    try:
        # The non-holder's release must not free the holder's lease.
        other.release_lease(key)
        assert other.try_lease(key) is False
    finally:
        other.close()


# -- degradation ----------------------------------------------------------------
def test_an_unopenable_cache_is_a_miss_not_an_error(tmp_path) -> None:
    # Point at a path that cannot be a database (a directory). Every operation
    # degrades to the uncached behaviour instead of raising.
    store = UsageCacheStore(tmp_path / "a_directory")
    (tmp_path / "a_directory").mkdir(exist_ok=True)
    assert store.get("k") is None
    store.set("k", "anthropic", [_report()], expires_at_ms=int(time.time() * 1000) + 1000)
    assert store.try_lease("k") is True  # no coordination, fetch anyway
    store.close()


def test_retention_prunes_only_rows_past_the_last_good_window(cache) -> None:
    key = "anthropic:old"
    # A row expired longer than the retention bound is dropped on the next write.
    ancient = int(time.time() * 1000) - USAGE_LAST_GOOD_RETENTION_MS - 60_000
    cache.set(key, "anthropic", [_report()], expires_at_ms=ancient)
    cache.set(
        "anthropic:other", "anthropic", [_report()], expires_at_ms=int(time.time() * 1000) + 60_000
    )
    assert cache.get(key, include_expired=True) is None
    assert cache.get("anthropic:other") is not None


def test_a_write_leaves_no_open_transaction_behind(cache) -> None:
    """`set()`'s pruning DELETEs must commit. sqlite3 opens an implicit
    transaction on the first write and leaves it OPEN until something commits;
    the bare cleanup DELETEs held the WAL write lock from one refresh to the
    next, so every other session's cache call blocked for the full 5s busy
    timeout and then failed — leases collapsed and writes were silently lost,
    each stall freezing that session's whole TUI."""
    cache.set("k", "p", [_report()], expires_at_ms=int(time.time() * 1000) + 60_000)
    assert cache._connect() is not None
    assert not cache._conn.in_transaction, "cleanup left the write lock held"

    # And the practical consequence: a SECOND connection's lease and write are
    # immediate, not a 5-second busy-timeout stall.
    other = UsageCacheStore(cache._db_path)
    try:
        start = time.perf_counter()
        assert other.try_lease("k2") is True
        other.set("k3", "p", [_report()], expires_at_ms=int(time.time() * 1000) + 60_000)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"peer session blocked {elapsed:.1f}s on the cache"
        assert other.get("k3") is not None, "peer write was silently lost"
    finally:
        other.close()


# -- preflight read-through (leased_account_usage) ----------------------------
# The routing helper is deliberately NOT the display read-through: it fetches
# live on every fast path (so a boundary can notice recovery/depletion) and only
# uses the lease to divert a concurrent PEER process that would otherwise
# duplicate the same account's fetch and earn the endpoint a per-source-IP 429.
# These tests pin exactly that contract. See docs/specs/preflight-usage-cache.md.


@pytest.mark.asyncio
async def test_preflight_lease_loser_serves_stale_without_a_network_fetch(tmp_path) -> None:
    """Fan-out collapse (the headline proof): 2 processes, 2 preflights, 1 fetch.

    Process A holds the account's fetch lease (mid-fetch). Process B's preflight
    must serve A's last-good row instead of crossing the network for the identical
    answer — the whole point of the ``:pf:`` lease.
    """
    db = tmp_path / "usage_cache.db"
    store_a = UsageCacheStore(db)
    store_b = UsageCacheStore(db)
    key = account_preflight_key("anthropic", "account-a")
    report_a = _report(percent=10.0)
    report_b = _report(percent=90.0)
    calls: list[str] = []

    async def fetch():
        calls.append("net")
        return report_b

    try:
        # A stale row A already wrote, and A holding the lease (mid-fetch).
        store_a.set(key, "anthropic", [report_a], expires_at_ms=store_a._now_ms() - 1)
        assert store_a.try_lease(key) is True

        served = await leased_account_usage(store_b, key, "anthropic", fetch)

        assert served is not None
        assert served.limits[0].amount.used == report_a.limits[0].amount.used  # stale served
        assert calls == []  # network NOT crossed by the lease-loser
    finally:
        store_a.close()
        store_b.close()


@pytest.mark.asyncio
async def test_preflight_free_lease_fetches_writes_and_releases(tmp_path) -> None:
    """The common (single-process) case: a free lease always fetches live, caches
    the result, and releases the lease so the next boundary can re-probe."""
    store = UsageCacheStore(tmp_path / "usage_cache.db")
    key = account_preflight_key("anthropic", "account-a")
    report = _report(percent=42.0)
    calls: list[str] = []

    async def fetch():
        calls.append("net")
        return report

    try:
        served = await leased_account_usage(store, key, "anthropic", fetch)

        assert served is report
        assert calls == ["net"]
        cached = store.get(key)
        assert cached is not None and cached[0].limits[0].amount.used == 42.0
        # Lease was released, not left to expire — the next boundary can fetch.
        assert store.try_lease(key) is True
    finally:
        store.close()


@pytest.mark.asyncio
async def test_preflight_keys_isolate_accounts_and_avoid_the_warmer_namespace(
    tmp_path,
) -> None:
    """Per-account isolation: two accounts get disjoint ``:pf:`` keys, each
    returning its OWN report — the namespace, not ``UsageReport.identity`` (which
    the raw preflight fetch never backfills), is what separates accounts. And the
    keys never collide with the warmer's per-provider-set rows."""
    db = tmp_path / "usage_cache.db"
    holder = UsageCacheStore(db)  # holds each account's lease → forces stale serving
    reader = UsageCacheStore(db)
    key_a = account_preflight_key("anthropic", "account-a")
    key_b = account_preflight_key("anthropic", "account-b")
    report_a = _report(percent=11.0)
    report_b = _report(percent=22.0)

    async def _fail():  # must never run: both are lease-losers with stale on hand
        raise AssertionError("lease-loser crossed the network")

    try:
        # Distinct namespace: contains ":pf:" and differs from the warmer's key.
        assert ":pf:" in key_a and ":pf:" in key_b
        assert key_a != key_b
        warmer_key = provider_cache_key("anthropic", fingerprint_accounts(["account-a"]))
        assert key_a != warmer_key

        reader.set(key_a, "anthropic", [report_a], expires_at_ms=reader._now_ms() - 1)
        reader.set(key_b, "anthropic", [report_b], expires_at_ms=reader._now_ms() - 1)
        assert holder.try_lease(key_a) is True
        assert holder.try_lease(key_b) is True

        served_a = await leased_account_usage(reader, key_a, "anthropic", _fail)
        served_b = await leased_account_usage(reader, key_b, "anthropic", _fail)

        assert served_a is not None and served_a.limits[0].amount.used == 11.0
        assert served_b is not None and served_b.limits[0].amount.used == 22.0
    finally:
        holder.close()
        reader.close()


@pytest.mark.asyncio
async def test_preflight_lease_loser_with_no_stale_fetches_live(tmp_path) -> None:
    """A cold start: the lease is held by a peer but no row exists. The loser
    cannot serve nothing, so it fetches live — matching the pre-cache behaviour."""
    db = tmp_path / "usage_cache.db"
    holder = UsageCacheStore(db)
    loser = UsageCacheStore(db)
    key = account_preflight_key("anthropic", "account-a")
    report = _report(percent=7.0)
    calls: list[str] = []

    async def fetch():
        calls.append("net")
        return report

    try:
        assert holder.try_lease(key) is True  # peer mid-fetch, nothing cached yet
        served = await leased_account_usage(loser, key, "anthropic", fetch)

        assert served is report
        assert calls == ["net"]
    finally:
        holder.close()
        loser.close()


@pytest.mark.asyncio
async def test_preflight_fails_open_when_the_cache_is_unavailable() -> None:
    """No store means the pre-cache behaviour: fetch live and pass the result
    through unchanged, including a ``None`` the routing caller fails open on."""
    calls: list[str] = []

    async def fetch_ok():
        calls.append("net")
        return _report(percent=3.0)

    async def fetch_none():
        calls.append("net")
        return None

    served = await leased_account_usage(None, "k", "anthropic", fetch_ok)
    assert served is not None and served.limits[0].amount.used == 3.0

    served_none = await leased_account_usage(None, "k", "anthropic", fetch_none)
    assert served_none is None
    assert calls == ["net", "net"]


@pytest.mark.asyncio
async def test_preflight_none_result_preserves_last_good_and_fails_open(tmp_path) -> None:
    """A ``None`` fetch (transport failure OR genuinely-empty) returns ``None`` so
    the routing caller fails open, while the last-good row stays servable to a
    concurrent lease-loser under the write_failure cool-down."""
    store = UsageCacheStore(tmp_path / "usage_cache.db")
    key = account_preflight_key("anthropic", "account-a")
    good = _report(percent=50.0)

    async def fetch_none():
        return None

    try:
        store.set(key, "anthropic", [good], expires_at_ms=store._now_ms() + 60_000)
        served = await leased_account_usage(store, key, "anthropic", fetch_none)

        assert served is None  # caller fails open
        # Last-good survives (write_failure re-wrote it under the short cool-down).
        retained = store.get(key, include_expired=True)
        assert retained is not None and retained[0].limits[0].amount.used == 50.0
    finally:
        store.close()
