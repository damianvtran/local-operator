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
    inside it. The entry-time sample rolls it back.
    """
    store, row_id = _lent_store(tmp_path)
    before = _payload(store, row_id)
    _raise_after(monkeypatch, store, "BEGIN", sqlite3.OperationalError("after begin"))

    assert store._update_payload(row_id, lambda d: d.update(x=1) or True) is False  # noqa: SLF001
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
