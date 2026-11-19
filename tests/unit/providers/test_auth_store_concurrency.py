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

import threading
from pathlib import Path
from typing import Any

from local_operator.network.credentials import renewal
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
