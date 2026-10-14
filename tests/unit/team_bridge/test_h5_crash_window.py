"""H5 S-10 — the crash window and the two-writer race on REAL queue stores.

The contract pinned here against the REAL native issuance ledger, the REAL
keyed queue door and its interprocess lock:

- a crash AFTER the ask is appended but BEFORE the issuance row is marked
  raised replays (LIVE, then COLD/reconcile contexts) to the SAME ask id, and
  the ask log still carries exactly ONE event;
- completing the mark does not perturb the identity either;
- a mismatched fingerprint refuses instead of minting a second ask;
- two writers racing the SAME key settle on ONE row — the lock, not luck.
"""

from __future__ import annotations

import threading

from local_operator.team_bridge.ask_issuance import AskIssuance
from local_operator.team_bridge.store import OwnerReceipt, Store
from local_operator.team_bridge.types import OwnerSnapshot
from tests.unit.team_bridge.test_h5_queue_idem import FP, MESSAGE_ID, _question, _queue

OWNER = OwnerSnapshot(
    credential_id=7,
    account_id="acc_1",
    tenant_id="ten_1",
    endpoint_id="ep_1",
    canonical_base_url="https://hub.example",
    enrollment_generation=7,
)


def _enrolled_store(tmp_path) -> Store:
    store = Store(tmp_path)
    store.commit_enrollment(OWNER, OwnerReceipt("bridge_consent_v1", "ref-1"), cos_session_id="s1")
    return store


def _keyed(queue, *, key: str = "iss-crash", fingerprint: str = FP) -> dict:
    return queue.enqueue_idempotent(
        issuance_key=key, fingerprint=fingerprint, questions=[_question()]
    )


def test_crash_after_append_before_mark_replays_to_the_same_ask(tmp_path) -> None:
    _enrolled_store(tmp_path)
    issuance = AskIssuance(tmp_path)
    issuance.claim(
        OWNER, MESSAGE_ID, {"v": 1}, delivery_ref="dref_crash", local_row_id="peer-" + "ab" * 16
    )
    queue = _queue(tmp_path)
    first = _keyed(queue)
    assert first["ok"] is True
    ask_id = str(first["details"]["ask_id"])
    # The intentional interruption: the ask row IS durable, the issuance row is
    # still 'claimed' — the crash window the repair must close.
    assert issuance.read(OWNER, MESSAGE_ID)["state"] == "claimed"
    again = _keyed(queue)  # LIVE replay
    assert again["ok"] is True
    assert str(again["details"]["ask_id"]) == ask_id
    assert again["details"]["status"] == "existing"
    issuance.mark_raised(OWNER, MESSAGE_ID, ask_id)  # the recovery completes
    cold = _keyed(queue)  # COLD / reconcile replay after the mark
    assert str(cold["details"]["ask_id"]) == ask_id
    assert len(queue.records()) == 1  # exactly ONE durable ask event


def test_a_conflicting_fingerprint_refuses_instead_of_minting(tmp_path) -> None:
    _enrolled_store(tmp_path)
    queue = _queue(tmp_path)
    assert _keyed(queue)["ok"] is True
    conflicting = _keyed(queue, fingerprint="bb" * 32)
    assert conflicting["ok"] is False
    assert len(queue.records()) == 1


def test_two_writers_on_the_same_key_settle_on_one_row(tmp_path) -> None:
    _enrolled_store(tmp_path)
    outcomes: list[dict] = []
    barrier = threading.Barrier(2)

    def writer() -> None:
        # A separate AskQueue instance per thread: the REAL interprocess-lock
        # shape, not one instance's in-process state.
        queue = _queue(tmp_path)
        barrier.wait()
        outcomes.append(_keyed(queue))

    threads = [threading.Thread(target=writer) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(outcomes) == 2
    ids = {str(outcome["details"]["ask_id"]) for outcome in outcomes if outcome["ok"]}
    assert len(ids) == 1
    fresh = _queue(tmp_path)
    assert len(fresh.records()) == 1


def test_two_writers_with_conflicting_fingerprints_mint_exactly_one(tmp_path) -> None:
    _enrolled_store(tmp_path)
    outcomes: list[dict] = []
    barrier = threading.Barrier(2)
    fingerprints = [FP, "cc" * 32]

    def writer(fingerprint: str) -> None:
        queue = _queue(tmp_path)
        barrier.wait()
        outcomes.append(_keyed(queue, fingerprint=fingerprint))

    threads = [threading.Thread(target=writer, args=(fp,)) for fp in fingerprints]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(1 for outcome in outcomes if outcome["ok"]) == 1
    assert len(_queue(tmp_path).records()) == 1
