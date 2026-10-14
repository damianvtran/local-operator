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
    snap = controller.cached_persisted_usage_reports_read_only(None)
    assert snap.reports == [] and snap.unavailable is False
    assert controller.cached_persisted_usage_reports_read_only("openai").reports == []
    assert _snapshot(root) == before, "a read-only route wrote to disk"


async def test_no_wal_or_shm_sidecars_and_unchanged_mtime_inode(tmp_path) -> None:
    """A mode=ro path must not create -wal/-shm nor touch existing files.

    This is the BLOCKER: ``mode=ro`` alone still creates SQLite sidecars beside
    the database, which is a write. The seam opens ``immutable=1`` instead. The
    inode+mtime comparison catches an in-place rewrite the name check would miss.
    """
    auth_db = tmp_path / "auth.db"
    cache_db = tmp_path / "usage.db"
    seeder = AuthStore(auth_db)
    seeder.upsert_credential("openai", {"access": "a", "refresh": "r", "expires": 10**15})
    seeder.close()
    cache = UsageCacheStore(cache_db)
    cache.set("k", "openai", [UsageReport(provider="openai")], expires_at_ms=10**13)
    cache.close()

    names_before = sorted(p.name for p in tmp_path.iterdir())
    stat_before = {p.name: (p.stat().st_ino, p.stat().st_mtime_ns) for p in tmp_path.iterdir()}

    controller = ProviderController.read_only_for_persisted_usage(auth_db, cache_db)
    controller.cached_persisted_usage_reports_read_only(None)
    controller.cached_persisted_usage_reports_read_only("openai")

    assert sorted(p.name for p in tmp_path.iterdir()) == names_before, "sidecar created"
    for name, (ino, mtime) in stat_before.items():
        now = (tmp_path / name).stat()
        assert (now.st_ino, now.st_mtime_ns) == (ino, mtime), f"{name} was touched"


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
    snap = controller.cached_persisted_usage_reports_read_only("openai")
    assert not snap.unavailable and [r.provider for r in snap.reports] == ["openai"]
    assert snap.reports[0].limits[0].amount.remaining == 10
    # Identity and vendor notes never reach the phone wire.
    assert all(r.identity is None and r.notes is None for r in snap.reports)
    inv = controller.cached_persisted_usage_reports_read_only(None)
    assert [r.provider for r in inv.reports] == ["openai"]


async def test_real_corrupt_auth_store_is_unavailable(tmp_path) -> None:
    """A corrupt AUTH store is UNAVAILABLE (``None``), not empty — real file.

    No monkeypatch of the swallow site: the file itself is a non-database, so the
    real read path (not a fake exception) has to draw the distinction.
    """
    auth_db = tmp_path / "auth.db"
    cache_db = tmp_path / "usage.db"
    auth_db.write_bytes(b"not a database")
    cache = UsageCacheStore(cache_db)
    cache.set("k", "openai", [UsageReport(provider="openai")], expires_at_ms=10**13)
    cache.close()
    controller = ProviderController.read_only_for_persisted_usage(auth_db, cache_db)
    assert controller.cached_persisted_usage_reports_read_only("openai").unavailable is True


async def test_real_missing_usage_table_is_unavailable(tmp_path) -> None:
    """A database with no ``usage_reports`` table is UNAVAILABLE, not empty."""

    auth_db = tmp_path / "auth.db"
    cache_db = tmp_path / "usage.db"
    seeder = AuthStore(auth_db)
    seeder.upsert_credential("openai", {"access": "a", "refresh": "r", "expires": 10**15})
    seeder.close()
    conn = sqlite3.connect(str(cache_db))
    conn.execute("CREATE TABLE unrelated (x)")
    conn.commit()
    conn.close()

    controller = ProviderController.read_only_for_persisted_usage(auth_db, cache_db)
    assert controller.cached_persisted_usage_reports_read_only("openai").unavailable is True


async def test_real_corrupt_usage_payload_is_unavailable(tmp_path) -> None:
    """A row whose payload will not decode is UNAVAILABLE, not empty."""

    auth_db = tmp_path / "auth.db"
    cache_db = tmp_path / "usage.db"
    seeder = AuthStore(auth_db)
    # The fingerprint the seam derives is the persisted identity ("acct"), so the
    # seeded row must be keyed by the SAME fingerprint or the seam sees an absent
    # key and ``[]`` is the correct answer.
    seeder.upsert_credential(
        "openai", {"access": "a", "refresh": "r", "expires": 10**15, "account_id": "acct"}
    )
    seeder.close()
    cache = UsageCacheStore(cache_db)
    # Corrupt the row under the key the seam ACTUALLY reads (the persisted
    # account's fingerprint), not an unrelated key.
    cache.set(
        provider_cache_key("openai", fingerprint_accounts(["acct"])),
        "openai",
        [UsageReport(provider="openai")],
        expires_at_ms=10**13,
    )
    cache.close()
    conn = sqlite3.connect(str(cache_db))
    assert conn.execute("SELECT count(*) FROM usage_reports").fetchone()[0] >= 1
    conn.execute("UPDATE usage_reports SET payload = ?", ("{not json",))
    conn.commit()
    conn.close()

    controller = ProviderController.read_only_for_persisted_usage(auth_db, cache_db)
    assert controller.cached_persisted_usage_reports_read_only("openai").unavailable is True


async def test_absent_store_close_is_a_noop(tmp_path) -> None:
    """Closing a read-only store with no database must not raise."""
    store = AuthStore.open_read_only(tmp_path / "absent.db")
    store.close()  # AttributeError before the fix


async def test_injected_writable_cache_is_not_used(tmp_path) -> None:
    """An injected WRITABLE cache must not be reached by the seam."""

    auth_db = tmp_path / "auth.db"
    seeder = AuthStore(auth_db)
    seeder.upsert_credential("openai", {"access": "a", "refresh": "r", "expires": 10**15})
    seeder.close()
    writable_path = tmp_path / "writable-usage.db"
    controller = ProviderController(
        AuthStore.open_read_only(auth_db),
        None,
        usage_cache=UsageCacheStore(writable_path),
    )
    assert controller.cached_persisted_usage_reports_read_only("openai").reports == []
    assert not writable_path.exists(), "seam used the injected writable cache"


async def test_fingerprint_ignores_env_override_keys(tmp_path, monkeypatch) -> None:
    """The persisted-account fingerprint never folds in an ambient env key."""

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


async def test_foreign_auth_schema_is_unavailable(tmp_path) -> None:
    """A valid SQLite auth file WITHOUT the credential table is UNAVAILABLE.

    ``sqlite_master`` looks fine (there is a table), so a name-only probe would
    call this empty. The seam must fail closed instead: it is not a store it can
    read, on BOTH the inventory and explicit-provider paths.
    """
    import sqlite3

    auth_db = tmp_path / "auth.db"
    cache_db = tmp_path / "usage.db"
    conn = sqlite3.connect(str(auth_db))
    conn.execute("CREATE TABLE unrelated (x)")
    conn.commit()
    conn.close()
    cache = UsageCacheStore(cache_db)
    cache.set("k", "openai", [UsageReport(provider="openai")], expires_at_ms=10**13)
    cache.close()

    controller = ProviderController.read_only_for_persisted_usage(auth_db, cache_db)
    assert controller.cached_persisted_usage_reports_read_only(None).unavailable is True
    assert controller.cached_persisted_usage_reports_read_only("openai").unavailable is True


async def test_auth_table_missing_required_column_is_unavailable(tmp_path) -> None:
    """A ``auth_credentials`` table missing columns the read selects is UNAVAILABLE."""
    import sqlite3

    auth_db = tmp_path / "auth.db"
    cache_db = tmp_path / "usage.db"
    conn = sqlite3.connect(str(auth_db))
    conn.execute("CREATE TABLE auth_credentials (id INTEGER PRIMARY KEY, provider TEXT)")
    conn.commit()
    conn.close()
    cache = UsageCacheStore(cache_db)
    cache.set("k", "openai", [UsageReport(provider="openai")], expires_at_ms=10**13)
    cache.close()

    controller = ProviderController.read_only_for_persisted_usage(auth_db, cache_db)
    assert controller.cached_persisted_usage_reports_read_only("openai").unavailable is True


async def test_wal_deleted_account_is_returned_flagged_not_current(tmp_path) -> None:
    """A committed DELETE in ``-wal`` is flagged, not silently served as current.

    ``immutable=1`` cannot see the WAL, so the reader still sees the removed row
    (an ordinary connection sees count=0). The seam DISCLOSES staleness and
    returns the snapshot flagged, so the surface can label it or exclude the
    removed account — it must not be presented as current, and it must not be
    withheld (a blank phone /usage on a busy desktop is the worse failure).
    """
    import sqlite3

    auth_db = tmp_path / "auth.db"
    cache_db = tmp_path / "usage.db"
    seeder = AuthStore(auth_db)
    seeder.upsert_credential(
        "openai", {"access": "a", "refresh": "r", "expires": 10**15, "account_id": "acct"}
    )
    seeder.close()
    cache = UsageCacheStore(cache_db)
    cache.set(
        provider_cache_key("openai", fingerprint_accounts(["acct"])),
        "openai",
        [UsageReport(provider="openai", identity="acct")],
        expires_at_ms=int(time.time() * 1000) + 60_000,
    )
    cache.close()

    writer = sqlite3.connect(str(auth_db))
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("DELETE FROM auth_credentials")
    writer.commit()  # lives in -wal, uncheckpointed
    assert (tmp_path / "auth.db-wal").stat().st_size > 0
    assert sqlite3.connect(str(auth_db)).execute(
        "SELECT count(*) FROM auth_credentials"
    ).fetchone() == (0,)

    controller = ProviderController.read_only_for_persisted_usage(auth_db, cache_db)
    snap = controller.cached_persisted_usage_reports_read_only(None)
    assert snap.unavailable is False
    assert snap.may_be_stale is True, "stale snapshot not flagged"
    assert snap.cached_at is not None
    # The value is still returned (not withheld) so the surface can choose.
    assert [r.provider for r in snap.reports] == ["openai"]
    explicit = controller.cached_persisted_usage_reports_read_only("openai")
    assert explicit.may_be_stale is True
    assert [r.provider for r in explicit.reports] == ["openai"]
    writer.close()


async def test_wal_present_with_live_account_returns_value_flagged(tmp_path) -> None:
    """A non-empty WAL beside a LIVE account must not blank the inventory.

    The common busy-desktop case: the account still exists, there is just an
    uncheckpointed WAL. The inventory must come back with its value, flagged.
    """
    import sqlite3

    auth_db = tmp_path / "auth.db"
    cache_db = tmp_path / "usage.db"
    seeder = AuthStore(auth_db)
    seeder.upsert_credential(
        "openai", {"access": "a", "refresh": "r", "expires": 10**15, "account_id": "acct"}
    )
    seeder.close()
    cache = UsageCacheStore(cache_db)
    cache.set(
        provider_cache_key("openai", fingerprint_accounts(["acct"])),
        "openai",
        [UsageReport(provider="openai", identity="acct")],
        expires_at_ms=int(time.time() * 1000) + 60_000,
    )
    cache.close()

    # A writer that commits a REAL unrelated change keeps a non-empty WAL
    # without removing the account (a no-op UPDATE writes no frames).
    writer = sqlite3.connect(str(auth_db))
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("UPDATE auth_credentials SET updated_at = updated_at + 1")
    writer.commit()
    assert (tmp_path / "auth.db-wal").stat().st_size > 0

    controller = ProviderController.read_only_for_persisted_usage(auth_db, cache_db)
    snap = controller.cached_persisted_usage_reports_read_only(None)
    assert snap.may_be_stale is True
    assert snap.unavailable is False
    assert [r.provider for r in snap.reports] == ["openai"], "inventory was blanked"
    writer.close()


async def test_clean_database_is_not_flagged_stale(tmp_path) -> None:
    """No non-empty ``-wal`` means the read is current, not stale."""
    auth_db = tmp_path / "auth.db"
    cache_db = tmp_path / "usage.db"
    seeder = AuthStore(auth_db)
    seeder.upsert_credential(
        "openai", {"access": "a", "refresh": "r", "expires": 10**15, "account_id": "acct"}
    )
    seeder.close()
    controller = ProviderController.read_only_for_persisted_usage(auth_db, cache_db)
    snap = controller.cached_persisted_usage_reports_read_only(None)
    assert snap.may_be_stale is False
    assert snap.unavailable is False
