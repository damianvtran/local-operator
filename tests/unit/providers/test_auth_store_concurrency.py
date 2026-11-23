"""One shared AuthStore read and written from several threads at once.

The mesh credential-renewal owner path runs its background claim
(``renewal.owner_claim`` → ``_record_claim``, thread mesh-control-conn) and a
node's write-back (``renewal.owner_return``, thread mesh-slow_0) against the SAME
store, and ``test_credentials_renewal_lend.py::
test_an_owner_invalid_grant_on_a_lent_login_waits_for_the_write_back`` went flaky
on main from two store races these tests pin directly:

1. ``_SerializedConnection`` locked only the ``execute`` call, so a caller's
   ``.fetchone()`` stepped the shared statement machinery unlocked: concurrent
   reads returned ``None`` rows or raised (the owner read its lent row as gone).
2. ``_update_payload`` read, mutated and wrote the whole payload with nothing
   held across the three, so a claim that read before an adoption committed
   wrote its stale payload over it (the adoption's ``awaiting_writeback_at``
   clear was undone).
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path
from typing import Any

import pytest

from local_operator.network.credentials import renewal
from local_operator.providers import auth_store
from local_operator.providers.auth_store import (
    AWAITING_WRITEBACK_KEY,
    REFRESH_DELEGATE_KEY,
    AuthStore,
)


def _lent_store(tmp_path: Path) -> tuple[AuthStore, int]:
    store = AuthStore(db_path=tmp_path / "auth.db")
    row = store.upsert_credential(
        "anthropic",
        {
            "type": "oauth",
            "refresh": "synthetic-refresh-0",
            "access": "synthetic-access-0",
            "expires": 1,
            REFRESH_DELEGATE_KEY: {"state": renewal.LENT_STATE, "epoch": 1},
            AWAITING_WRITEBACK_KEY: 5,
        },
    )
    return store, row.id


def test_concurrent_reads_on_one_store_never_return_a_bad_row(tmp_path: Path) -> None:
    """Two threads reading one row: every read is the row, none raises.

    Before the rows were drained under the lock this measured roughly one bad
    read in seven (``None`` rows, TypeError/JSONDecodeError/InterfaceError);
    one thread alone measured zero, so the count is the race, not the reader.
    """
    store, row_id = _lent_store(tmp_path)
    reads_per_thread = 4000
    bad: list[str] = []
    start = threading.Barrier(2)

    def hammer() -> None:
        start.wait()
        for _ in range(reads_per_thread):
            try:
                row = store.get_credential(row_id)
            except Exception as exc:  # noqa: BLE001 — the race's symptom is the count
                bad.append(type(exc).__name__)
                continue
            if row is None or row.data.get("refresh") != "synthetic-refresh-0":
                bad.append("bad-row")

    threads = [threading.Thread(target=hammer) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    store.close()
    assert bad == [], f"{len(bad)} bad reads of {2 * reads_per_thread}: {sorted(set(bad))}"


def test_a_claim_that_read_before_an_adoption_cannot_write_over_it(tmp_path: Path) -> None:
    """Force the traced schedule: claim reads → adoption commits → claim writes.

    The claim is the production ``_record_claim`` with the epoch it asked about;
    the adoption is ``owner_return``'s effect on the row (bump the epoch, clear
    the write-back wait). The claim's mutation is held between its read and its
    write on an Event until the adoption has either committed (the race: its
    stale payload then lands on top) or demonstrably cannot (the fix: the
    adoption waits for the claim's whole read-modify-write, then runs on the
    claim's result, and the claim's epoch guard sees the row it writes over).
    """
    store, row_id = _lent_store(tmp_path)
    claim_has_read = threading.Event()
    adoption_done = threading.Event()
    original = store._update_payload  # noqa: SLF001
    claim_thread_name = "test-claim"

    def paused_update(credential_id: int, mutate: Any, **kwargs: Any) -> bool:
        if threading.current_thread().name != claim_thread_name:
            return original(credential_id, mutate, **kwargs)

        def held(data: dict[str, Any]) -> bool:
            # `data` is the claim's read; the adoption may race only from here.
            claim_has_read.set()
            # Bounded only for the FIXED path, where the adoption is blocked by
            # this very read-modify-write and can never set the Event: the wait
            # then times out and the claim writes. On the racy path the adoption
            # commits and sets it at once.
            adoption_done.wait(timeout=1.0)
            return mutate(data)

        return original(credential_id, held, **kwargs)

    store._update_payload = paused_update  # type: ignore[method-assign]  # noqa: SLF001

    def adopt(data: dict[str, Any]) -> bool:
        record = dict(data[REFRESH_DELEGATE_KEY])
        record["epoch"] = 2
        data[REFRESH_DELEGATE_KEY] = record
        data["refresh"] = "synthetic-refresh-adopted"
        data.pop(AWAITING_WRITEBACK_KEY, None)
        return True

    claim = threading.Thread(
        target=renewal._record_claim,  # noqa: SLF001
        args=(store, row_id, {"state": "renewing", "epoch": 1, "at_ms": 7}),
        name=claim_thread_name,
    )
    claim.start()
    assert claim_has_read.wait(timeout=10)
    adopter = threading.Thread(
        target=lambda: (store._update_payload(row_id, adopt), adoption_done.set())  # noqa: SLF001
    )
    adopter.start()
    claim.join(timeout=10)
    adopter.join(timeout=10)
    assert not claim.is_alive() and not adopter.is_alive()

    row = store.get_credential(row_id)
    store.close()
    assert row is not None
    assert AWAITING_WRITEBACK_KEY not in row.data, "the claim undid the adoption"
    assert row.data["refresh"] == "synthetic-refresh-adopted"
    assert row.data[REFRESH_DELEGATE_KEY]["epoch"] == 2


# ---------------------------------------------------------------------------
# `_update_payload` failure paths (PR #4269 review round 1, S1/S2, R1-m1)
# ---------------------------------------------------------------------------


def _raise_after(
    monkeypatch: pytest.MonkeyPatch, store: AuthStore, prefix: str, exc: BaseException
) -> None:
    """Make the store's wrapped ``execute`` raise AFTER a matching statement ran."""
    conn = store._conn  # noqa: SLF001
    real = conn.execute

    def execute(sql: str, *args: Any, **kwargs: Any) -> Any:
        result = real(sql, *args, **kwargs)
        if sql.startswith(prefix):
            raise exc
        return result

    monkeypatch.setattr(conn, "execute", execute)


def _payload(store: AuthStore, row_id: int) -> dict[str, Any]:
    row = store.get_credential(row_id)
    assert row is not None
    return dict(row.data)


def test_a_wrapper_raising_after_begin_still_releases_the_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S1: BEGIN ran, then the wrapper raised before returning.

    The old ``began`` flag was set only once ``execute`` returned, so this left
    the transaction open on the shared connection and every later statement ran
    inside it. The entry-time sample rolls it back. The error is not lock
    contention, so it propagates (security round 2) -- after the rollback.
    """
    store, row_id = _lent_store(tmp_path)
    before = _payload(store, row_id)
    _raise_after(monkeypatch, store, "BEGIN", sqlite3.OperationalError("after begin"))

    with pytest.raises(sqlite3.OperationalError, match="after begin"):
        store._update_payload(row_id, lambda d: d.update(x=1) or True)  # noqa: SLF001
    monkeypatch.undo()
    assert store._conn.in_transaction is False  # noqa: SLF001
    assert _payload(store, row_id) == before


def test_a_read_error_propagates_and_releases_the_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S2: a failed marker READ raises, as before the reservation existed.

    Turning it into ``False`` would let the refresh path proceed as though no
    marker were there when the row could not be read at all.
    """
    store, row_id = _lent_store(tmp_path)

    def broken_read(_credential_id: int) -> Any:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(store, "get_credential", broken_read)
    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        store._update_payload(row_id, lambda d: True)  # noqa: SLF001
    assert store._conn.in_transaction is False  # noqa: SLF001


def test_a_write_error_is_best_effort_false_with_no_partial_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S2: a failed marker WRITE stays best-effort, and nothing is half-written."""
    store, row_id = _lent_store(tmp_path)
    before = _payload(store, row_id)
    _raise_after(monkeypatch, store, "UPDATE", sqlite3.OperationalError("database is locked"))

    assert store._update_payload(row_id, lambda d: d.update(x=1) or True) is False  # noqa: SLF001
    monkeypatch.undo()
    assert store._conn.in_transaction is False  # noqa: SLF001
    assert _payload(store, row_id) == before


def test_rows_are_drained_while_the_statement_lock_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-m1: the deterministic form of the concurrent-read test above.

    Another thread must be unable to take the lock while the cursor drains.
    """
    store, row_id = _lent_store(tmp_path)
    lock = store._conn.lock  # noqa: SLF001
    held: list[bool] = []
    real_init = auth_store._FetchedCursor.__init__  # noqa: SLF001

    def init(self: Any, cursor: sqlite3.Cursor) -> None:
        probe: list[bool] = []

        def try_take() -> None:
            got = lock.acquire(blocking=False)
            probe.append(not got)
            if got:
                lock.release()

        t = threading.Thread(target=try_take)
        t.start()
        t.join()
        held.extend(probe)
        real_init(self, cursor)

    monkeypatch.setattr(auth_store._FetchedCursor, "__init__", init)  # noqa: SLF001
    assert store.get_credential(row_id) is not None
    assert held and all(held)


# ---------------------------------------------------------------------------
# Reservation errors (PR #4269 security round 2, S2 residual; review R2-m1/m2)
# ---------------------------------------------------------------------------


def test_busy_reservation_is_best_effort_false_with_nothing_held(tmp_path: Path) -> None:
    """Real contention: another connection holds the write reservation.

    This is the one reservation failure that may be a skipped write: the row was
    never in doubt, a peer is simply writing.
    """
    store, row_id = _lent_store(tmp_path)
    before = _payload(store, row_id)
    store._conn.execute("PRAGMA busy_timeout = 0")  # noqa: SLF001
    peer = sqlite3.connect(tmp_path / "auth.db", timeout=0, isolation_level=None)
    try:
        peer.execute("BEGIN IMMEDIATE")
        wrote = store._update_payload(row_id, lambda d: d.update(x=1) or True)  # noqa: SLF001
        assert wrote is False
        assert store._conn.in_transaction is False  # noqa: SLF001
        peer.execute("ROLLBACK")
    finally:
        peer.close()
    assert _payload(store, row_id) == before


@pytest.mark.parametrize(
    ("message", "code", "contention"),
    [
        ("database is locked", None, True),
        ("database table is locked: auth_credentials", None, True),
        ("database is locked", sqlite3.SQLITE_BUSY, True),
        ("database is locked", sqlite3.SQLITE_BUSY | (2 << 8), True),  # BUSY_SNAPSHOT
        ("database table is locked", sqlite3.SQLITE_LOCKED, True),
        ("disk I/O error", None, False),
        ("disk I/O error", sqlite3.SQLITE_IOERR, False),
        ("database is locked", sqlite3.SQLITE_IOERR, False),  # the code wins
        ("database disk image is malformed", sqlite3.SQLITE_CORRUPT, False),
    ],
)
def test_only_busy_or_locked_counts_as_contention(
    message: str, code: int | None, contention: bool
) -> None:
    exc = sqlite3.OperationalError(message)
    if code is not None:
        exc.sqlite_errorcode = code  # type: ignore[attr-defined]
    assert auth_store._is_lock_contention(exc) is contention  # noqa: SLF001
    # A non-Operational class is never contention, whatever its text says.
    assert auth_store._is_lock_contention(sqlite3.DatabaseError(message)) is False  # noqa: SLF001


def _fail_inside_marker_arm(
    monkeypatch: pytest.MonkeyPatch, store: AuthStore, prefix: str, exc: BaseException
) -> list[int]:
    """Raise ``exc`` from the store's ``execute`` for ``prefix`` statements, but
    only while ``_arm_send_marker`` runs, so the rest of the refresh path (lease,
    row reads) is untouched and the fault lands exactly on the marker's
    reservation or read. Returns a list counting the injected faults."""
    conn = store._conn  # noqa: SLF001
    real_execute = conn.execute
    real_arm = store._arm_send_marker  # noqa: SLF001
    armed = [False]
    fired: list[int] = []

    def execute(sql: str, *args: Any, **kwargs: Any) -> Any:
        if armed[0] and sql.lstrip().upper().startswith(prefix):
            fired.append(1)
            raise exc
        return real_execute(sql, *args, **kwargs)

    def arm(*args: Any, **kwargs: Any) -> Any:
        armed[0] = True
        try:
            return real_arm(*args, **kwargs)
        finally:
            armed[0] = False

    monkeypatch.setattr(conn, "execute", execute)
    monkeypatch.setattr(store, "_arm_send_marker", arm)
    return fired


@pytest.mark.parametrize(
    ("prefix", "exc"),
    [
        ("BEGIN", sqlite3.DatabaseError("disk I/O error")),
        ("BEGIN", sqlite3.OperationalError("database disk image is malformed")),
        ("SELECT", sqlite3.OperationalError("disk I/O error")),
    ],
    ids=["reservation-database-error", "reservation-operational-corrupt", "marker-select"],
)
def test_a_marker_reservation_or_read_error_never_reaches_the_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix: str, exc: BaseException
) -> None:
    """R2-m2 / security S2: the caller-level guarantee, through the real refresh.

    An expired row goes through ``_ensure_oauth_fresh``; the send marker cannot
    be reserved or read. Downgrading that to "marker write skipped" would let the
    refresh POST a token whose send marker it never saw. The provider must not
    be called, the stored token must not move, the original error must surface
    (R2-m1: the rollback's own failure cannot mask it), and nothing stays held.
    """
    store = AuthStore(db_path=tmp_path / "auth.db")
    row = store.upsert_credential(
        "anthropic",
        {
            "type": "oauth",
            "refresh": "synthetic-refresh-0",
            "access": "synthetic-access-0",
            "expires": 1,
        },
    )
    before = _payload(store, row.id)
    calls: list[str | None] = []

    async def counting_refresh(creds: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(creds.get("refresh"))
        return {"type": "oauth", "refresh": "synthetic-refresh-1", "access": "synthetic-access-1"}

    monkeypatch.setattr(store, "_refresh_fn", lambda provider: counting_refresh)
    fired = _fail_inside_marker_arm(monkeypatch, store, prefix, exc)

    stored = store.get_credential(row.id)
    assert stored is not None
    with pytest.raises(type(exc), match=str(exc)):
        asyncio.run(store._ensure_oauth_fresh(stored))  # noqa: SLF001
    monkeypatch.undo()

    assert fired, "the fault never reached the marker write"
    assert calls == [], "the provider was called past an unreadable send marker"
    assert _payload(store, row.id) == before
    assert store._conn.in_transaction is False  # noqa: SLF001


def test_a_failing_rollback_does_not_mask_the_reservation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-m1: the error the caller sees is the one that failed the call."""
    store, row_id = _lent_store(tmp_path)
    _raise_after(monkeypatch, store, "BEGIN", sqlite3.DatabaseError("disk I/O error"))
    conn = store._conn  # noqa: SLF001
    real_rollback = conn.rollback
    rollbacks: list[int] = []

    def failing_rollback() -> None:
        rollbacks.append(1)
        real_rollback()
        raise sqlite3.OperationalError("rollback failed too")

    monkeypatch.setattr(conn, "rollback", failing_rollback)
    with pytest.raises(sqlite3.DatabaseError, match="disk I/O error"):
        store._update_payload(row_id, lambda d: True)  # noqa: SLF001
    assert rollbacks == [1]
    monkeypatch.undo()
    assert store._conn.in_transaction is False  # noqa: SLF001


# -- the post-refresh persist (#4269 round 1, deferred lost update) -----------
#
# `_ensure_oauth_fresh` reads the row (`fresh`), POSTs, and used to persist
# `fresh + refreshed` as one whole-payload UPDATE. Anything another writer
# committed during the POST -- a renewal claim, a delegate-record or marker
# change -- was overwritten whenever the stored refresh token had not moved. The
# providers make it worse: they return `dict(creds)` plus the new tokens, so the
# returned dict itself carries the stale copy of every other field. These stubs
# return that same shape on purpose.

_CLAIM_KEY = "synthetic_concurrent_claim"


def _expired_store(tmp_path: Path) -> tuple[AuthStore, int]:
    store = AuthStore(db_path=tmp_path / "auth.db")
    row = store.upsert_credential(
        "anthropic",
        {
            "type": "oauth",
            "refresh": "synthetic-refresh-0",
            "access": "synthetic-access-0",
            "expires": 1,
            "org_id": "synthetic-org",
        },
    )
    return store, row.id


def _provider_shaped_refresh(during_post: Any, *, rotate: bool) -> Any:
    """A stub `_refresh_fn` result: runs ``during_post`` as the "POST", then
    answers like the real providers (``dict(creds)`` + new tokens)."""

    async def refresh(creds: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        during_post()
        merged = dict(creds)
        merged["access"] = "synthetic-access-1"
        merged["expires"] = 2**62
        if rotate:
            merged["refresh"] = "synthetic-refresh-1"
        # A misbehaving refresh fn may not rewrite identity (PR-12).
        merged["org_id"] = "synthetic-other-org"
        return merged

    return refresh


def _claim(data: dict[str, Any]) -> bool:
    data[_CLAIM_KEY] = {"epoch": 7}
    return True


@pytest.mark.parametrize("rotate", [False, True], ids=["same-refresh", "rotated-refresh"])
@pytest.mark.parametrize("writer", ["same-store", "peer-store"])
def test_a_payload_change_committed_during_the_refresh_post_survives_the_persist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rotate: bool, writer: str
) -> None:
    """The reproduction: a claim committed mid-POST is still on the row after.

    ``same-store`` is another thread/coroutine of this process (the mesh claim
    on the shared store); ``peer-store`` is another process's connection. On
    main both lost the claim; the access token, the rotation (when there is one)
    and the identity restore must land as before.
    """
    store, row_id = _expired_store(tmp_path)
    peer = store if writer == "same-store" else AuthStore(db_path=tmp_path / "auth.db")
    refresh = _provider_shaped_refresh(
        lambda: peer._update_payload(row_id, _claim), rotate=rotate  # noqa: SLF001
    )
    monkeypatch.setattr(store, "_refresh_fn", lambda provider: refresh)

    stored = store.get_credential(row_id)
    assert stored is not None
    returned = asyncio.run(store._ensure_oauth_fresh(stored))  # noqa: SLF001

    after = _payload(store, row_id)
    expected_refresh = "synthetic-refresh-1" if rotate else "synthetic-refresh-0"
    for view in (after, returned):
        assert view.get(_CLAIM_KEY) == {"epoch": 7}, "the concurrent claim was overwritten"
        assert view["access"] == "synthetic-access-1"
        assert view["refresh"] == expected_refresh
        assert view["org_id"] == "synthetic-org"
        assert auth_store.REFRESH_SEND_UNCONFIRMED_KEY not in view
    assert store._conn.in_transaction is False  # noqa: SLF001
    if peer is not store:
        peer.close()
    store.close()


def test_a_relogin_during_the_refresh_post_is_not_clobbered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rotation guard, now run on the row read inside the reservation: a
    sign-in that replaced the payload mid-POST wins, and the caller is handed
    the sign-in's credential rather than the stale refresh's."""
    store, row_id = _expired_store(tmp_path)
    peer = AuthStore(db_path=tmp_path / "auth.db")
    relogin = {
        "type": "oauth",
        "refresh": "synthetic-relogin-refresh",
        "access": "synthetic-relogin-access",
        "expires": 2**62,
        "org_id": "synthetic-org",
    }

    def sign_in() -> None:
        assert peer.upsert_credential("anthropic", relogin).id == row_id

    refresh = _provider_shaped_refresh(sign_in, rotate=True)
    monkeypatch.setattr(store, "_refresh_fn", lambda provider: refresh)

    stored = store.get_credential(row_id)
    assert stored is not None
    returned = asyncio.run(store._ensure_oauth_fresh(stored))  # noqa: SLF001

    after = _payload(store, row_id)
    for view in (after, returned):
        assert view["refresh"] == "synthetic-relogin-refresh"
        assert view["access"] == "synthetic-relogin-access"
    assert store._conn.in_transaction is False  # noqa: SLF001
    peer.close()
    store.close()


def test_a_tombstone_during_the_refresh_post_still_stands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The grant-dead guard, also on the in-reservation read: the rotation is
    dropped, the tombstone stays, and the caller is told the grant is dead."""
    store, row_id = _expired_store(tmp_path)
    peer = AuthStore(db_path=tmp_path / "auth.db")

    def tombstone(data: dict[str, Any]) -> bool:
        data[auth_store.GRANT_DEAD_AT_KEY] = 5
        return True

    refresh = _provider_shaped_refresh(
        lambda: peer._update_payload(row_id, tombstone), rotate=True  # noqa: SLF001
    )
    monkeypatch.setattr(store, "_refresh_fn", lambda provider: refresh)

    stored = store.get_credential(row_id)
    assert stored is not None
    with pytest.raises(auth_store.CredentialInvalidError):
        asyncio.run(store._ensure_oauth_fresh(stored))  # noqa: SLF001

    after = _payload(store, row_id)
    assert after[auth_store.GRANT_DEAD_AT_KEY] == 5
    assert after["refresh"] == "synthetic-refresh-0"
    assert store._conn.in_transaction is False  # noqa: SLF001
    peer.close()
    store.close()


def test_a_busy_persist_raises_instead_of_dropping_the_refreshed_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The persist is NOT best-effort like a marker write: the presented token
    is already spent, so a skipped write would strand the rotation in memory
    and report success. Contention surfaces as the error the former bare UPDATE
    raised, with nothing held afterwards and the row untouched."""
    store, row_id = _expired_store(tmp_path)
    store._conn.execute("PRAGMA busy_timeout = 0")  # noqa: SLF001
    blocker = sqlite3.connect(tmp_path / "auth.db", timeout=0, isolation_level=None)

    refresh = _provider_shaped_refresh(lambda: blocker.execute("BEGIN IMMEDIATE"), rotate=True)
    monkeypatch.setattr(store, "_refresh_fn", lambda provider: refresh)
    # Sampled as the lease release starts, i.e. right after the persist raised:
    # the persist's own reservation must be gone by then. (The lease DELETE that
    # follows fails on the same blocker and is not part of this change.)
    in_tx_at_lease_release: list[bool] = []
    real_release = store._release_refresh_lease  # noqa: SLF001

    def release(credential_id: int) -> None:
        in_tx_at_lease_release.append(store._conn.in_transaction)  # noqa: SLF001
        real_release(credential_id)

    monkeypatch.setattr(store, "_release_refresh_lease", release)

    stored = store.get_credential(row_id)
    assert stored is not None
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            asyncio.run(store._ensure_oauth_fresh(stored))  # noqa: SLF001
        assert in_tx_at_lease_release == [False]
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    assert _payload(store, row_id)["refresh"] == "synthetic-refresh-0"
    store.close()


def test_a_non_best_effort_write_error_propagates_after_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`best_effort=False` turns a write-phase sqlite error into a raise (the
    marker writes keep their `False`), with the reservation released."""
    store, row_id = _lent_store(tmp_path)
    before = _payload(store, row_id)
    _raise_after(monkeypatch, store, "UPDATE", sqlite3.OperationalError("disk I/O error"))
    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        store._update_payload(row_id, _claim, best_effort=False)  # noqa: SLF001
    monkeypatch.undo()
    assert store._conn.in_transaction is False  # noqa: SLF001
    assert _payload(store, row_id) == before
