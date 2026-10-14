"""Read-only cached-usage seam for the paired-phone route.

The phone route reads cached quota over a connection whose promise is "no
writes". Both stores in the ordinary path violate that promise on CONSTRUCTION
(``mkdir`` + ``O_CREAT`` + WAL PRAGMA + schema DDL + ``commit`` + ``chmod``),
so the seam has its own constructors and these tests pin the promise at the
level a caller cannot forget it: build the controller, read, and assert the
filesystem is untouched.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from local_operator.providers.auth_store import AuthStore
from local_operator.providers.controller import ProviderController
from local_operator.providers.usage import UsageAmount, UsageLimit, UsageReport
from local_operator.providers.usage_cache import (
    UsageCacheStore,
    fingerprint_accounts,
    provider_cache_key,
)

pytestmark = pytest.mark.asyncio


def _snapshot(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


async def test_construction_and_empty_read_create_nothing(tmp_path) -> None:
    """No dirs, db, WAL or schema DDL on construct OR read of an absent cache."""
    root = tmp_path / "cfg"
    root.mkdir()
    before = _snapshot(root)
    controller = ProviderController.read_only_for_persisted_usage(
        root / "auth.db", root / "usage.db"
    )
    assert controller.cached_persisted_usage_reports_read_only(None) == []
    assert controller.cached_persisted_usage_reports_read_only("openai") == []
    assert _snapshot(root) == before, "a read-only route wrote to disk"


async def test_read_only_open_refuses_writes(tmp_path) -> None:
    """The handle cannot mutate even if a caller reaches for it directly."""
    db = tmp_path / "usage.db"
    cache = UsageCacheStore(db)
    cache.set("k", "openai", [UsageReport(provider="openai")], expires_at_ms=10**13)
    cache.close()

    ro = UsageCacheStore(db, read_only=True)
    conn = ro._connect()
    assert conn is not None
    with pytest.raises(Exception):
        conn.execute("DELETE FROM usage_reports")
    ro.close()


async def test_auth_store_read_only_constructor_creates_nothing(tmp_path) -> None:
    """``AuthStore.open_read_only`` writes nothing; an absent db is empty."""
    root = tmp_path / "cfg"
    root.mkdir()
    store = AuthStore.open_read_only(root / "auth.db")
    assert store.list_credentials() == []
    assert store.get_credential(1) is None
    assert _snapshot(root) == set(), "read-only store constructor wrote to disk"


async def test_populated_report_is_served_without_identity(tmp_path) -> None:
    """A cached report comes back for the persisted account, identity stripped."""
    auth_db = tmp_path / "auth.db"
    cache_db = tmp_path / "usage.db"
    seeder = AuthStore(auth_db)
    seeder.upsert_credential(
        "openai",
        {"access": "a", "refresh": "r", "expires": 10**15, "account_id": "acct"},
    )
    seeder.close()
    cache = UsageCacheStore(cache_db)
    report = UsageReport(
        provider="openai",
        identity="someone@example.test",
        notes="plan: pro",
        limits=[UsageLimit("openai:credits", "Credits", UsageAmount(remaining=10, limit=20))],
    )
    cache.set(
        provider_cache_key("openai", fingerprint_accounts(["acct"])),
        "openai",
        [report],
        expires_at_ms=int(time.time() * 1000) + 60_000,
    )
    cache.close()

    controller = ProviderController.read_only_for_persisted_usage(auth_db, cache_db)
    reports = controller.cached_persisted_usage_reports_read_only("openai")
    assert [r.provider for r in reports] == ["openai"]
    assert reports[0].limits[0].amount.remaining == 10
    # Identity and vendor notes never reach the phone wire.
    assert all(r.identity is None and r.notes is None for r in reports)
    assert [r.provider for r in controller.cached_persisted_usage_reports_read_only(None)] == [
        "openai"
    ]


async def test_corrupt_cache_is_unavailable_not_empty(tmp_path, monkeypatch) -> None:
    """A read FAILURE is UNAVAILABLE (``None``), distinct from empty (``[]``).

    A phone must not render "0 credits" for a cache it could not read: spent and
    unreadable are different claims. Empty is ``[]``; unreadable is ``None``.
    """
    auth_db = tmp_path / "auth.db"
    cache_db = tmp_path / "usage.db"
    seeder = AuthStore(auth_db)
    seeder.upsert_credential("openai", {"access": "a", "refresh": "r", "expires": 10**15})
    seeder.close()
    cache_db.write_bytes(b"not a database")
    controller = ProviderController.read_only_for_persisted_usage(auth_db, cache_db)

    def boom(*_args, **_kwargs):
        raise sqlite3.DatabaseError("unreadable")

    monkeypatch.setattr(UsageCacheStore, "get", boom)
    assert controller.cached_persisted_usage_reports_read_only("openai") is None


async def test_fingerprint_ignores_env_override_keys(tmp_path, monkeypatch) -> None:
    """The persisted-account fingerprint never folds in an ambient env key."""
    import os

    auth_db = tmp_path / "auth.db"
    seeder = AuthStore(auth_db)
    seeder.upsert_credential(
        "anthropic", {"access": "a", "refresh": "r", "expires": 10**15, "account_id": "persisted"}
    )
    seeder.close()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ambient-not-this-login")

    store = AuthStore.open_read_only(auth_db)
    fingerprint = ProviderController._persisted_account_fingerprint(store, "anthropic")
    expected = fingerprint_accounts(["persisted"])
    assert fingerprint == expected
