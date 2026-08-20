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
    USAGE_FAILURE_BACKOFF_MS,
    USAGE_LAST_GOOD_RETENTION_MS,
    UsageCacheStore,
    fingerprint_accounts,
    fingerprint_secret,
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
