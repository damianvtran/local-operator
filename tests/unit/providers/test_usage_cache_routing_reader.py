"""The routing opt-in is committed-SQL read-only, not immutable phone display."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from local_operator.providers.usage import UsageAmount, UsageLimit, UsageReport
from local_operator.providers.usage_cache import (
    UsageCacheStore,
    account_preflight_key,
    report_to_dict,
)


def report(fraction: float) -> UsageReport:
    now = int(time.time() * 1000)
    return UsageReport(
        "anthropic",
        fetched_at=now,
        identity="synthetic@example.invalid",
        limits=[
            UsageLimit(
                id="shared",
                label="5 hour",
                shared=True,
                window="5 hour",
                amount=UsageAmount(used_fraction=fraction),
                resets_at_ms=now + 3_600_000,
            )
        ],
    )


def held_writer(path: Path) -> tuple[UsageCacheStore, str, sqlite3.Connection]:
    writer = UsageCacheStore(path)
    key = account_preflight_key("anthropic", "synthetic@example.invalid")
    expiry = int(time.time() * 1000) + 600_000
    writer.set(key, "anthropic", [report(0.1)], expires_at_ms=expiry)
    connection = writer._connect()
    assert connection is not None
    connection.execute("PRAGMA wal_autocheckpoint=0")
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    writer.set(key, "anthropic", [report(1.0)], expires_at_ms=expiry)
    return writer, key, connection


def fraction(store: UsageCacheStore, key: str) -> float:
    rows = store.get(key)
    assert rows is not None and len(rows) == 1
    value = rows[0].limits[0].amount.fraction()
    assert value is not None
    return value


def test_opt_in_observes_commits_not_uncommitted_changes_and_preserves_display(tmp_path):
    writer, key, connection = held_writer(tmp_path / "usage_cache.db")
    reader = UsageCacheStore(writer.db_path, read_only=True, wal_aware=True)
    display = UsageCacheStore(writer.db_path, read_only=True)
    reopened = None
    try:
        assert fraction(reader, key) == 1.0
        assert fraction(display, key) == 0.1
        assert display.read_only_sidecar_present()
        assert display.read_only_cached_at() is not None
        assert not reader.read_only_sidecar_present()
        assert reader.read_only_cached_at() is None
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "UPDATE usage_reports SET payload = ? WHERE key = ?",
            (json.dumps([report_to_dict(report(0.2))]), key),
        )
        assert fraction(reader, key) == 1.0
        connection.rollback()
        writer.set(key, "anthropic", [report(0.7)], expires_at_ms=int(time.time() * 1000) + 600_000)
        assert fraction(reader, key) == 0.7
        reader.close()
        reopened = UsageCacheStore(writer.db_path, read_only=True, wal_aware=True)
        assert fraction(reopened, key) == 0.7
        assert fraction(display, key) == 0.1
        sql = reopened._connect()
        assert sql is not None and sql.execute("PRAGMA query_only").fetchone() == (1,)
        with pytest.raises(sqlite3.OperationalError, match="readonly|read-only"):
            sql.execute("UPDATE usage_reports SET expires_at_ms = 0")
        assert fraction(writer, key) == 0.7
    finally:
        connection.rollback()
        if reopened is not None:
            reopened.close()
        reader.close()
        display.close()
        writer.close()


@pytest.mark.parametrize("directory", ["with spaces", "question?root", "fragment#root", "?# space"])
def test_routing_uri_escapes_explicit_root_names(tmp_path, directory):
    writer, key, _ = held_writer(tmp_path / directory / "usage_cache.db")
    reader = UsageCacheStore(writer.db_path, read_only=True, wal_aware=True)
    try:
        assert fraction(reader, key) == 1.0
    finally:
        reader.close()
        writer.close()


def test_missing_parent_is_not_created_and_mode_requires_read_only(tmp_path):
    path = tmp_path / "absent-parent" / "usage_cache.db"
    reader = UsageCacheStore(path, read_only=True, wal_aware=True)
    try:
        assert reader.get("synthetic") is None
        assert reader.latest_account_report("anthropic", {"synthetic"}) is None
        assert not path.parent.exists()
        with pytest.raises(ValueError, match="requires read_only"):
            UsageCacheStore(path, wal_aware=True)
        assert not path.parent.exists()
    finally:
        reader.close()


def test_coordination_failure_never_falls_back_to_available_stale_immutable(tmp_path, monkeypatch):
    writer, key, _ = held_writer(tmp_path / "usage_cache.db")
    reader = UsageCacheStore(writer.db_path, read_only=True, wal_aware=True)
    original = sqlite3.connect
    attempts = []

    def unavailable(database, *args, **kwargs):
        attempts.append((database, kwargs))
        # Make an illicit stale fallback succeed, so the assertion detects it.
        if "immutable=1" in str(database):
            return original(database, *args, **kwargs)
        raise sqlite3.OperationalError("synthetic unusable WAL coordination")

    monkeypatch.setattr(sqlite3, "connect", unavailable)
    try:
        assert reader.get(key) is None
        assert reader.latest_account_report("anthropic", {"synthetic@example.invalid"}) is None
        assert reader.get(key) is None
        assert len(attempts) == 1  # abandon this ranking, not one retry per account
        uri, options = attempts[0]
        assert "mode=ro" in uri and "immutable" not in uri and "nolock" not in uri
        assert options["timeout"] == 0.05
        reader.close()
        assert reader.get(key) is None
        assert len(attempts) == 2  # a later ranking can retry availability
    finally:
        reader.close()
        writer.close()


def test_corrupt_or_foreign_schema_is_unknown_without_repeated_queries(tmp_path):
    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"not a sqlite database")
    reader = UsageCacheStore(corrupt, read_only=True, wal_aware=True)
    try:
        assert reader.get("synthetic") is None
        assert reader._routing_read_unavailable
        assert corrupt.read_bytes() == b"not a sqlite database"
    finally:
        reader.close()
    foreign = tmp_path / "foreign.db"
    connection = sqlite3.connect(foreign)
    connection.execute("CREATE TABLE unrelated (n INTEGER)")
    connection.commit()
    connection.close()
    reader = UsageCacheStore(foreign, read_only=True, wal_aware=True)
    try:
        assert reader.get("synthetic") is None
        assert reader._routing_read_unavailable
        assert reader._conn is None
        assert reader.latest_account_report("anthropic", {"synthetic"}) is None
    finally:
        reader.close()


def test_conflicting_sql_lock_is_a_short_unknown_not_a_writer_timeout(tmp_path):
    writer, key, connection = held_writer(tmp_path / "usage_cache.db")
    # A rollback-journal exclusive lock genuinely blocks readers (a normal WAL
    # writer transaction does not). This changes only our synthetic fixture.
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("BEGIN EXCLUSIVE")
    reader = UsageCacheStore(writer.db_path, read_only=True, wal_aware=True)
    try:
        start = time.monotonic()
        assert reader.get(key) is None
        assert reader.get(key) is None
        assert time.monotonic() - start < 2.0  # generous fleet ceiling, not exact50ms
        assert reader._routing_read_unavailable
    finally:
        reader.close()
        connection.rollback()
        writer.close()


@pytest.mark.parametrize("protect", ["main-file", "directory"])
def test_permissions_yield_committed_truth_or_unknown_never_stale_main(tmp_path, protect):
    root = tmp_path / "protected"
    writer, key, _ = held_writer(root / "usage_cache.db")
    reader = UsageCacheStore(writer.db_path, read_only=True, wal_aware=True)
    target = writer.db_path if protect == "main-file" else root
    original_mode = target.stat().st_mode & 0o777
    target.chmod(0 if protect == "main-file" else 0o500)
    try:
        rows = reader.get(key)
        # Root privileges and usable existing WAL/SHM may still permit a read.
        # Refusing all read-only directories would encode a false SQLite claim.
        if rows is not None:
            assert rows[0].limits[0].amount.fraction() == 1.0
        else:
            assert reader._routing_read_unavailable
    finally:
        reader.close()
        target.chmod(original_mode)
        writer.close()
