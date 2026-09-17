"""`lop qwencloud-ticket migrate`: the value moves, and the plaintext GOES.

This is the only irreversible verb in the feature, so the assertions here are
deliberately made against the ARTIFACT rather than against the code's own
report: every "the plaintext is gone" claim is a byte-scan of the database and
BOTH its WAL sidecars, and :func:`test_the_scan_discriminates` proves that scan
could have seen the value in the first place.

Every test uses ``tmp_path`` and points ``LOCAL_OPERATOR_CONFIG_DIR`` at it, so
the secret store resolves inside the sandbox. Nothing here may touch
``~/.local-operator``, and the only ticket-shaped literal is the obvious fake.
"""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from local_operator import cli
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.qwencloud_console import (
    QWENCLOUD_CONSOLE_PROJECT_ID,
    QWENCLOUD_CONSOLE_PROVIDER,
    QWENCLOUD_TICKET_SECRET_NAME,
    QWENCLOUD_TICKET_STALE_MS,
)

#: Never a real cookie. A test needing a "different" value uses the -2 form.
FAKE_TICKET = "fake-console-ticket"
FAKE_TICKET_2 = "fake-console-ticket-2"


def _plaintext_on_disk(db_path: Path, needle: bytes) -> bool:
    """True if ``needle`` is in the database OR either WAL sidecar.

    Scanning ``auth.db`` alone is VACUOUS. ``AuthStore._connect`` sets
    ``PRAGMA journal_mode=WAL``, so a freshly written row lives in
    ``auth.db-wal`` and the main file reports False BEFORE any migration has
    happened -- a check built on it passes against a completely broken
    migration. Measured on this branch, and pinned by
    :func:`test_the_scan_discriminates`.
    """
    for path in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
        if path.exists() and needle in path.read_bytes():
            return True
    return False


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[AuthStore]:
    """A real ``AuthStore`` in a sandboxed config dir.

    A REAL store, not a fake: these tests need ``_conn``, a real WAL file and a
    real byte-scan. ``LOCAL_OPERATOR_CONFIG_DIR`` is redirected so the secret
    store the CLI resolves with ``base=None`` also lands in ``tmp_path`` -- it
    is the config dir, not ``LOCAL_OPERATOR_HOME``, that moves ``auth.db`` and
    ``secrets/``.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    opened = AuthStore(db_path=tmp_path / "auth.db")
    try:
        yield opened
    finally:
        opened.close()


@pytest.fixture()
def secret_base(tmp_path: Path) -> Path:
    """Where ``base=None`` resolves the secret store to, given the fixture above."""
    return tmp_path / "config"


@pytest.fixture(autouse=True)
def _no_usage_invalidation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the cache drop everywhere except the test that asserts on it.

    Patched at ``auth_cli``, which is where ``_qwencloud_ticket_migrate``
    imports it from at point of use; patching ``cli`` would not intercept a
    function-local import. The real call builds a ProviderController over the
    store, which is out of scope for every test here but E15.
    """
    monkeypatch.setattr(
        "local_operator.providers.auth_cli._invalidate_cached_usage",
        lambda *a, **k: None,
    )


def _seed_legacy_row(
    store: AuthStore, ticket: str = FAKE_TICKET, *, captured_at: int | None = None, **extra: Any
) -> None:
    """Write a PRE-MIGRATION row: the value in `auth.db`, as PR 1 stored it.

    Spelled out rather than calling `store_ticket`, because slice A's
    `store_ticket` no longer writes a `ticket` key -- this is the legacy shape
    the migration exists to find, and it has to be constructed deliberately.
    """
    payload: dict[str, Any] = {
        "ticket": ticket,
        "project_id": QWENCLOUD_CONSOLE_PROJECT_ID,
        "captured_at": int(time.time() * 1000) if captured_at is None else captured_at,
    }
    payload.update(extra)
    store.upsert_credential(QWENCLOUD_CONSOLE_PROVIDER, payload)


def _row_data(store: AuthStore) -> dict[str, Any]:
    rows = store.list_credentials(QWENCLOUD_CONSOLE_PROVIDER, include_disabled=True)
    assert len(rows) == 1, f"expected one row, got identity keys {[r.identity_key for r in rows]}"
    return rows[0].data


def _secret_value(base: Path) -> bytes:
    """The stored VALUE, for round-trip assertions only. Never printed."""
    from local_operator.secrets import access

    return access.open_store(base).get(QWENCLOUD_TICKET_SECRET_NAME)


def _patch_qwen(monkeypatch: pytest.MonkeyPatch, **fns: Any) -> None:
    """Patch the qwencloud_console seam the CLI imports at point of use.

    The import inside `_qwencloud_ticket_migrate` is `from ... import name`, so
    the binding resolved at call time is the module attribute.
    """
    from local_operator.providers import qwencloud_console

    for name, fn in fns.items():
        monkeypatch.setattr(qwencloud_console, name, fn)


# --- the scan itself ---------------------------------------------------------


def test_the_scan_discriminates(
    store: AuthStore, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """E2: the check could have SEEN the plaintext. Not a duplicate of E3.

    E3 asserts the plaintext is gone; this asserts the assertion is capable of
    failing. A security check that cannot fail proves nothing -- and a scan of
    `auth.db` alone returns clean here, BEFORE anything has been migrated.
    """
    _seed_legacy_row(store)
    db_path = tmp_path / "auth.db"

    assert _plaintext_on_disk(db_path, FAKE_TICKET.encode()) is True, (
        "the scan cannot see a freshly written plaintext ticket, so every "
        "'it is gone' assertion built on it is vacuous"
    )

    assert cli._qwencloud_ticket_migrate(store) == 0
    capsys.readouterr()

    assert _plaintext_on_disk(db_path, FAKE_TICKET.encode()) is False


def test_the_plaintext_is_gone_from_every_file(
    store: AuthStore, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """E3: db + both sidecars. `VACUUM` without the checkpoint leaves it behind."""
    _seed_legacy_row(store)

    assert cli._qwencloud_ticket_migrate(store) == 0
    capsys.readouterr()

    db_path = tmp_path / "auth.db"
    for path in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
        if path.exists():
            assert (
                FAKE_TICKET.encode() not in path.read_bytes()
            ), f"plaintext survives in {path.name}"


# --- the value moves ---------------------------------------------------------


def test_the_value_round_trips_into_the_secret_store(
    store: AuthStore, secret_base: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """E4: byte-identical, not truncated and not re-encoded."""
    _seed_legacy_row(store)

    assert cli._qwencloud_ticket_migrate(store) == 0
    capsys.readouterr()

    assert _secret_value(secret_base) == FAKE_TICKET.encode()
    data = _row_data(store)
    assert "ticket" not in data
    assert data["secret_name"] == QWENCLOUD_TICKET_SECRET_NAME
    assert data["length"] == len(FAKE_TICKET)


def test_migrate_is_idempotent(
    store: AuthStore, secret_base: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """E5: the second run is a no-op that changes nothing it should not."""
    _seed_legacy_row(store, captured_at=1_700_000_000_000)

    assert cli._qwencloud_ticket_migrate(store) == 0
    capsys.readouterr()
    first = _row_data(store)
    first_value = _secret_value(secret_base)

    assert cli._qwencloud_ticket_migrate(store) == 0
    out = capsys.readouterr()

    assert "already in the encrypted secret store" in out.out
    assert _row_data(store) == first, "the no-op rewrote the row"
    assert _row_data(store)["captured_at"] == 1_700_000_000_000
    assert _secret_value(secret_base) == first_value
    assert len(store.list_credentials(QWENCLOUD_CONSOLE_PROVIDER, include_disabled=True)) == 1


def test_migrate_reports_a_length_on_success(
    store: AuthStore, capsys: pytest.CaptureFixture[str]
) -> None:
    """E6: a length and a store name, never the value and never a zero."""
    _seed_legacy_row(store)

    assert cli._qwencloud_ticket_migrate(store) == 0
    out = capsys.readouterr()

    assert f"({len(FAKE_TICKET)} characters)" in out.out
    assert QWENCLOUD_TICKET_SECRET_NAME in out.out
    assert "(0 characters)" not in out.out
    assert FAKE_TICKET not in out.out
    assert FAKE_TICKET not in out.err


def test_captured_at_survives_the_migration(
    store: AuthStore, capsys: pytest.CaptureFixture[str]
) -> None:
    """E7: capture time is the staleness clock; re-stamping it hides a stale cookie."""
    ten_days_ago = int(time.time() * 1000) - (10 * 86_400_000)
    assert ten_days_ago < int(time.time() * 1000) - QWENCLOUD_TICKET_STALE_MS
    _seed_legacy_row(store, captured_at=ten_days_ago)

    assert cli._qwencloud_ticket_migrate(store) == 0
    capsys.readouterr()

    assert _row_data(store)["captured_at"] == ten_days_ago

    assert cli._qwencloud_ticket_action("status", store) == 0
    status = capsys.readouterr().out
    assert (
        "older than a console session usually lasts" in status
    ), "the staleness warning stopped firing, so the migration reset the clock"


def test_project_id_survives_so_a_later_set_upserts_in_place(
    store: AuthStore, capsys: pytest.CaptureFixture[str]
) -> None:
    """E8: without `project_id`, `_identity_key_for` returns None and `set` duplicates."""
    from local_operator.providers.qwencloud_console import store_ticket

    _seed_legacy_row(store)
    assert cli._qwencloud_ticket_migrate(store) == 0
    capsys.readouterr()

    assert _row_data(store)["project_id"] == QWENCLOUD_CONSOLE_PROJECT_ID

    store_ticket(store, FAKE_TICKET_2)
    rows = store.list_credentials(QWENCLOUD_CONSOLE_PROVIDER, include_disabled=True)
    assert len(rows) == 1, f"a later `set` INSERTED instead of upserting: {len(rows)} rows"


# --- the failure paths leave the plaintext alone -----------------------------


def test_a_failed_secret_write_leaves_the_plaintext(
    store: AuthStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """E9: the secret is written FIRST, so a failure there removes nothing."""
    from local_operator.providers.qwencloud_console import TicketStoreError

    _seed_legacy_row(store)

    def _boom(ticket: str, base: Any) -> None:
        raise TicketStoreError("the ticket could not be encrypted (SecretStoreError)")

    _patch_qwen(monkeypatch, _store_secret_value=_boom)

    assert cli._qwencloud_ticket_migrate(store) == 1
    out = capsys.readouterr()

    assert "NOTHING WAS CHANGED" in out.err
    assert _plaintext_on_disk(tmp_path / "auth.db", FAKE_TICKET.encode()) is True
    assert _row_data(store)["ticket"] == FAKE_TICKET
    assert FAKE_TICKET not in out.out
    assert FAKE_TICKET not in out.err


def test_a_failed_confirmation_leaves_the_plaintext(
    store: AuthStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """E10: "it did not raise" is not evidence the value is readable."""
    _seed_legacy_row(store)
    _patch_qwen(
        monkeypatch,
        _store_secret_value=lambda ticket, base: None,
        _secret_is_present=lambda base: False,
    )

    assert cli._qwencloud_ticket_migrate(store) == 1
    out = capsys.readouterr()

    assert "NOTHING WAS REMOVED" in out.err
    assert _plaintext_on_disk(tmp_path / "auth.db", FAKE_TICKET.encode()) is True
    assert _row_data(store)["ticket"] == FAKE_TICKET


def test_a_locked_store_says_unlock_and_changes_nothing(
    store: AuthStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """E11: the remedy survives the clause order, and no raw wire text reaches the user.

    Asserting on the MESSAGE and not merely the exit code is the point:
    catching `TicketStoreError` above `TicketStoreLocked` still exits 1, and
    the user silently loses the one action they can take.
    """
    from local_operator.providers.qwencloud_console import TicketStoreLocked

    _seed_legacy_row(store)

    def _locked(ticket: str, base: Any) -> None:
        raise TicketStoreLocked(
            "the secret store is hardened and locked, so the ticket cannot be "
            "stored. Run `lop secret unlock`, then retry"
        )

    _patch_qwen(monkeypatch, _store_secret_value=_locked)

    assert cli._qwencloud_ticket_migrate(store) == 1
    out = capsys.readouterr()

    assert "lop secret unlock" in out.err
    assert "no lop session is registered with the broker" not in out.err

    # THE ASSERTIONS THAT ACTUALLY PIN THE CLAUSE ORDER.
    #
    # `lop secret unlock` above does NOT discriminate, and that is worth
    # stating: the remedy lives in the EXCEPTION MESSAGE slice A raises, and
    # both clauses interpolate `{exc}` -- so swapping the clause order leaves
    # that string in stderr and the exit code at 1. Verified by mutation: with
    # the generic clause moved above the locked one, the assertions above all
    # still pass.
    #
    # What differs is the ADVICE THIS COMMAND ADDS. The locked path says
    # re-run migrate, because the plaintext is still in auth.db and unlocking
    # is the whole fix. The generic path instead offers the fresh-cookie
    # fallback, which for a merely-locked store is wrong and destructive
    # advice -- it tells the user to go re-authenticate when they only needed
    # to unlock.
    assert "re-run migrate" in out.err
    assert "capture a fresh cookie" not in out.err, (
        "the generic clause handled a locked store: the user is being told to "
        "re-authenticate when unlocking is the actual remedy"
    )

    assert _plaintext_on_disk(tmp_path / "auth.db", FAKE_TICKET.encode()) is True
    assert _row_data(store)["ticket"] == FAKE_TICKET


# --- the checkpoint, which is where this command lies if it is going to ------


def test_a_blocked_checkpoint_is_reported_not_claimed_as_success(
    store: AuthStore, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """E12a: a REAL blocking reader, because the busy case raises nothing.

    `PRAGMA wal_checkpoint(TRUNCATE)` reports a blocked checkpoint in the FIRST
    COLUMN of its return row -- measured `(1, 18, 18)` with the plaintext still
    readable in `auth.db-wal`. An implementation that discards that row and
    catches only `sqlite3.Error` claims success over a cookie still on disk, so
    monkeypatching an exception here would NOT discriminate: code blind to
    `busy=1` passes that test. Hence a second live connection holding a read
    snapshot.
    """
    _seed_legacy_row(store)
    db_path = tmp_path / "auth.db"

    reader = sqlite3.connect(str(db_path))
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT * FROM auth_credentials").fetchall()

        assert cli._qwencloud_ticket_migrate(store) == 0
        out = capsys.readouterr()
    finally:
        reader.close()

    assert "could NOT be cleared" in out.out
    assert "another process is reading the database" in out.out
    assert (
        "replaced with metadata only" not in out.out
    ), "the receipt claimed the plaintext was replaced while it is still readable"
    assert _plaintext_on_disk(db_path, FAKE_TICKET.encode()) is True
    # The VALUE still moved: this is a cleanup failure, not a migration failure.
    assert "ticket" not in _row_data(store)


def test_a_rerun_clears_a_previously_blocked_checkpoint(
    store: AuthStore, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """E12b: the advertised remedy is REAL — the no-op still compacts.

    If the already-migrated branch returned before the VACUUM, "re-run migrate;
    re-running clears it" would be a lie and the plaintext would stay readable
    permanently, with `status` reporting success because there is no `ticket`
    key. Unrepairable by the tool.
    """
    _seed_legacy_row(store)
    db_path = tmp_path / "auth.db"

    reader = sqlite3.connect(str(db_path))
    reader.execute("BEGIN")
    reader.execute("SELECT * FROM auth_credentials").fetchall()
    first = cli._qwencloud_ticket_migrate(store)
    capsys.readouterr()
    reader.close()

    assert first == 0
    assert (
        _plaintext_on_disk(db_path, FAKE_TICKET.encode()) is True
    ), "the blocked run did not leave the state this test exists to repair"

    second = cli._qwencloud_ticket_migrate(store)
    out = capsys.readouterr()

    assert second == 0
    assert "already in the encrypted secret store" in out.out
    assert (
        _plaintext_on_disk(db_path, FAKE_TICKET.encode()) is False
    ), "re-running did not clear the plaintext, so the warning's advice is a lie"


def test_a_busy_database_is_reported_but_not_fatal(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """E12c: the RAISED arm. A failed cleanup is not a failed migration."""
    _seed_legacy_row(store)

    real_execute = store._conn.execute

    def _fail_vacuum(sql: str, *args: Any, **kwargs: Any) -> Any:
        if "VACUUM" in sql:
            raise sqlite3.OperationalError("database is locked")
        return real_execute(sql, *args, **kwargs)

    monkeypatch.setattr(store._conn, "execute", _fail_vacuum)

    assert cli._qwencloud_ticket_migrate(store) == 0
    out = capsys.readouterr()

    assert "could NOT be cleared" in out.out
    assert "replaced with metadata only" not in out.out
    # The migration itself SUCCEEDED: the row moved and the value is readable.
    assert "ticket" not in _row_data(store)


def test_a_soft_deleted_row_is_migrated_too(
    store: AuthStore, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """E13: a disabled row is still ON DISK.

    Read without `include_disabled=True` it is invisible, migrate reports
    "nothing to migrate" with exit 0, and the plaintext survives -- the false
    success this whole feature exists to prevent.
    """
    _seed_legacy_row(store)
    row_id = store.list_credentials(QWENCLOUD_CONSOLE_PROVIDER)[0].id
    store._conn.execute(
        "UPDATE auth_credentials SET disabled_cause = ? WHERE id = ?", ("revoked", row_id)
    )
    store._conn.commit()
    assert store.list_credentials(QWENCLOUD_CONSOLE_PROVIDER) == []

    assert cli._qwencloud_ticket_migrate(store) == 0
    out = capsys.readouterr()

    assert "nothing to migrate" not in out.out
    assert _plaintext_on_disk(tmp_path / "auth.db", FAKE_TICKET.encode()) is False


# --- the value never escapes -------------------------------------------------


def test_the_value_never_reaches_stdout_stderr_or_an_exception(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """E14: every branch, including the ones that fail.

    The guarantee this command makes in place of a redaction registration (a
    bare CLI process has no session to register against) is that the value
    never leaves the process at all.
    """
    from local_operator.providers.qwencloud_console import (
        TicketStoreError,
        TicketStoreLocked,
    )

    raised: list[BaseException] = []

    def _record(exc: BaseException) -> BaseException:
        raised.append(exc)
        return exc

    # success
    _seed_legacy_row(store)
    assert cli._qwencloud_ticket_migrate(store) == 0
    # the already-migrated no-op
    assert cli._qwencloud_ticket_migrate(store) == 0

    outs: list[str] = []
    errs: list[str] = []
    captured = capsys.readouterr()
    outs.append(captured.out)
    errs.append(captured.err)

    # the no-row no-op
    store.delete_credentials_for_provider(QWENCLOUD_CONSOLE_PROVIDER)
    assert cli._qwencloud_ticket_migrate(store) == 0
    captured = capsys.readouterr()
    outs.append(captured.out)
    errs.append(captured.err)

    # locked, write failure, confirm failure, vacuum failure
    for patch, expected in (
        (
            {
                "_store_secret_value": lambda t, b: (_ for _ in ()).throw(
                    _record(TicketStoreLocked("hardened and locked. Run `lop secret unlock`"))
                )
            },
            1,
        ),
        (
            {
                "_store_secret_value": lambda t, b: (_ for _ in ()).throw(
                    _record(TicketStoreError("the ticket could not be encrypted"))
                )
            },
            1,
        ),
        (
            {
                "_store_secret_value": lambda t, b: None,
                "_secret_is_present": lambda b: False,
            },
            1,
        ),
    ):
        with monkeypatch.context() as patched:
            _seed_legacy_row(store)
            _patch_qwen(patched, **patch)
            assert cli._qwencloud_ticket_migrate(store) == expected
        captured = capsys.readouterr()
        outs.append(captured.out)
        errs.append(captured.err)

    # vacuum failure
    with monkeypatch.context() as patched:
        real_execute = store._conn.execute

        def _fail_vacuum(sql: str, *args: Any, **kwargs: Any) -> Any:
            if "VACUUM" in sql:
                raise _record(sqlite3.OperationalError("database is locked"))
            return real_execute(sql, *args, **kwargs)

        patched.setattr(store._conn, "execute", _fail_vacuum)
        assert cli._qwencloud_ticket_migrate(store) == 0
    captured = capsys.readouterr()
    outs.append(captured.out)
    errs.append(captured.err)

    assert len(outs) == 6, "a branch was not exercised"
    for stream in (*outs, *errs):
        assert FAKE_TICKET not in stream
    assert raised, "no exception branch was driven, so the str(exc) check is vacuous"
    for exc in raised:
        assert FAKE_TICKET not in str(exc)
        assert FAKE_TICKET not in repr(exc)


def test_migrate_invalidates_the_cached_usage_row(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """E15: C proved a stale note outlives the ~5 min TTL.

    Patched at `auth_cli`, where `_qwencloud_ticket_migrate` imports it from at
    point of use: patching `local_operator.cli._invalidate_cached_usage` would
    NOT intercept a function-local import, and the spy would never fire.
    """
    seen: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        "local_operator.providers.auth_cli._invalidate_cached_usage",
        lambda *a, **k: seen.append(a),
    )
    _seed_legacy_row(store)

    assert cli._qwencloud_ticket_migrate(store) == 0
    capsys.readouterr()

    assert seen == [(cli._QWENCLOUD_TICKET_AUGMENTS, store)]


def test_no_temp_file_holds_the_value(
    store: AuthStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """E16: nothing is spooled through a temp file on the way to the store.

    The load-bearing assertion is the SECOND one -- no file under the
    redirected tempdir contains the value -- because that is what an
    implementation spooling the ticket through `tempfile` would fail.

    The file-count assertion allows exactly one thing, and the exception is not
    a concession to make the test pass. `secrets/protocol.py:108 socket_path`
    moves the broker's RENDEZVOUS POINT to
    `gettempdir()/lop-secrets-<uid>-<digest>` when the natural path would
    exceed `MAX_SOCKET_PATH = 103` -- `sockaddr_un.sun_path` is 104 bytes on
    macOS -- and `lock_path` follows it there. pytest's own `tmp_path` is what
    forces that: measured 155 bytes for this test's config dir versus 80 for a
    shallower one, so an unconditional "no new file" assertion would pass or
    fail on the DEPTH OF THE TEST'S PATH rather than on anything migrate does.
    Only the socket and its lock move; the key, the database and the audit log
    stay in the secrets directory, and neither artifact carries the value
    (verified by the scan below, which covers them like any other file).

    A stray file under any OTHER name still fails this test. The allowlist is
    spelled as exact names rather than a `lop-*` glob for that reason: a broad
    pattern would launder a future scratch file into an exemption.
    """
    import re
    import tempfile

    spool = tmp_path / "spool"
    spool.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(spool))
    assert tempfile.gettempdir() == str(spool)

    before = set(spool.rglob("*"))
    _seed_legacy_row(store)

    assert cli._qwencloud_ticket_migrate(store) == 0
    capsys.readouterr()

    # Exactly `lop-secrets-<uid>-<12 hex>/`, and inside it only the socket and
    # its lock. `protocol.py` names all three: SOCKET_FILENAME, LOCK_FILENAME,
    # and the digest[:12] in `_runtime_fallback_dir`.
    rendezvous = re.compile(rf"^lop-secrets-{os.getuid()}-[0-9a-f]{{12}}$")

    def _permitted(relative: Path) -> bool:
        parts = relative.parts
        if not parts or not rendezvous.match(parts[0]):
            return False
        if len(parts) == 1:
            return True
        return len(parts) == 2 and parts[1] in ("broker.sock", "broker.lock")

    after = set(spool.rglob("*"))
    unexpected = sorted(
        str(path.relative_to(spool))
        for path in after - before
        if not _permitted(path.relative_to(spool))
    )
    assert not unexpected, f"migrate left files in the temp dir: {unexpected}"
    for path in after:
        if path.is_file():
            assert FAKE_TICKET.encode() not in path.read_bytes()
