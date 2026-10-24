"""The turn-side BODY consumer — the per-call invariant, against a real store.

Every case here drives the REAL ``Store`` (sqlite on disk) through the module
the Session calls, because the invariant is about durable state transitions:
a green double would not prove a claim was released or a row stayed pending.

The aside rule is asserted structurally at the end: the aside builder never
consults the pending turn, so a frame cannot enter a compaction request.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Any

from local_operator.team_bridge.pending_turn import FetchedBody, PendingTurn
from local_operator.team_bridge.request_body import BodyBudget
from local_operator.team_bridge.store import (
    OwnerReceipt,
    OwnerSnapshot,
    Store,
    StoreRefused,
    state_db_path,
)

CONVERSATION_ID = "bcv_abcdefgh"
MESSAGE_ID = "bcmsg_abcdef01"
DIGEST = "9f2a" * 16
FINGERPRINT = "aa11" * 16
SESSION = "sess_cos"
BODY_TEXT = "canary body from the channel"


def _owner() -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=7,
        account_id="acc_1",
        tenant_id="3f2a1111-2222-4333-8444-555566667777",
        endpoint_id="ep_1",
        canonical_base_url="https://hub.example",
        enrollment_generation=1,
    )


def _store(tmp_path: Path) -> tuple[Store, int]:
    store = Store(tmp_path)
    store.commit_enrollment(
        _owner(), OwnerReceipt("bridge_consent_v1", "ref-1"), cos_session_id=SESSION
    )
    # Use the store's OWN trusted publisher rather than raw SQL: the policy
    # row's shape is the store's business, and a hand-built row would let this
    # test pass against a schema the real path cannot write.
    epoch = store.publish_screening_policy(
        _owner(),
        version="v1",
        template_sha256="ab" * 32,
        default_model="qa-default",
        revision="rev-1",
    )
    return store, epoch


def _reserve(store: Store, epoch: int) -> bool:
    return store.record_body_reservation(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        conversation_seq=3,
        mention=True,
        body_sha256=DIGEST,
        screening_fingerprint=FINGERPRINT,
        policy_epoch=epoch,
        consent_version="bridge_consent_v1",
        designated_session=SESSION,
        now=time.time(),
    )


def _row(store: Store) -> dict[str, Any]:
    row = store.body_reservation(_owner(), CONVERSATION_ID, MESSAGE_ID)
    assert row is not None
    return row


async def _fetch(conversation_id: str, message_id: str) -> FetchedBody | None:
    return FetchedBody(
        body=BODY_TEXT,
        body_sha256=DIGEST,
        conversation_label="#qa",
        sender_label="Ada",
    )


def _budget() -> BodyBudget:
    return BodyBudget(available_bytes=100_000, measured_from="test budget")


def _turn(tmp_path: Path) -> PendingTurn:
    return PendingTurn(_owner(), SESSION, config_dir=tmp_path)


def test_a_pending_row_is_claimed_and_its_frame_cached(tmp_path) -> None:
    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)
    turn = _turn(tmp_path)

    assert asyncio.run(turn.open(budget=_budget(), fetch=_fetch)) is True
    assert _row(store)["state"] == "claimed"

    text = turn.for_call()
    assert text is not None and "END UNTRUSTED DATA OBJECT" in text
    assert BODY_TEXT in text
    # The SAME bytes on the next physical call of this turn (S-1): re-rendering
    # would be a second frame rather than a re-send of the admitted one.
    assert turn.for_call() == text
    assert turn.appended_calls == 2


def test_settlement_refuses_to_complete_a_turn_that_never_appended(tmp_path) -> None:
    """S-16a: the silent-loss guard. A successful verdict is not enough."""

    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)
    turn = _turn(tmp_path)
    assert asyncio.run(turn.open(budget=_budget(), fetch=_fetch)) is True

    outcome = turn.settle(entry_id="entry_1", error=None, stop_reason="stop", aborted=False)

    assert outcome == "never_appended"
    row = _row(store)
    assert row["state"] == "pending" and row["last_error"] == "never_appended"


def test_drifted_consent_omits_the_frame_and_leaves_the_row_pending(tmp_path) -> None:
    """S-21: a withdrawal after the first call must stop the very next one."""

    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)
    turn = _turn(tmp_path)
    assert asyncio.run(turn.open(budget=_budget(), fetch=_fetch)) is True
    assert turn.for_call() is not None

    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        conn.execute("UPDATE enrollments SET consent_version = 'bridge_consent_v2'")

    assert turn.for_call() is None, "a drifted consent must not carry the body"
    outcome = turn.settle(entry_id="entry_1", error=None, stop_reason="stop", aborted=False)
    assert outcome == "drifted"
    row = _row(store)
    assert row["state"] == "pending" and row["last_error"] == "admission_drift"


def test_a_lost_lease_stops_injection_and_reports_claim_lost(tmp_path) -> None:
    """S-12: a SECOND consumer that reclaims an expired row takes the lease.

    Expiry alone does not steal a live claim — only a competing claimant does,
    which is why the test drives a real second PendingTurn rather than editing
    the expiry column and hoping.
    """

    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)
    first = _turn(tmp_path)
    assert asyncio.run(first.open(budget=_budget(), fetch=_fetch)) is True

    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        conn.execute("UPDATE body_reservations SET claim_expires_at = ?", (time.time() - 1,))

    second = _turn(tmp_path)
    assert asyncio.run(second.open(budget=_budget(), fetch=_fetch)) is True

    assert first.for_call() is None, "a turn that lost its lease must send without"
    assert second.for_call() is not None
    assert (
        first.settle(entry_id="entry_1", error=None, stop_reason="stop", aborted=False)
        == "claim_lost"
    )


def test_a_successful_turn_completes_and_clears_the_claim_tokens(tmp_path) -> None:
    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)
    turn = _turn(tmp_path)
    assert asyncio.run(turn.open(budget=_budget(), fetch=_fetch)) is True
    assert turn.for_call() is not None

    outcome = turn.settle(entry_id="entry_1", error=None, stop_reason="stop", aborted=False)

    assert outcome == "completed"
    row = _row(store)
    assert row["state"] == "completed"
    assert row["completed_entry_id"] == "entry_1"
    assert row["claim_token"] is None and row["turn_token"] is None


def test_an_aborted_or_failed_turn_leaves_the_row_retryable(tmp_path) -> None:
    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)
    turn = _turn(tmp_path)
    assert asyncio.run(turn.open(budget=_budget(), fetch=_fetch)) is True
    assert turn.for_call() is not None

    assert (
        turn.settle(entry_id="entry_1", error="boom", stop_reason="error", aborted=False)
        == "unverified"
    )
    assert _row(store)["state"] == "pending"


def test_a_same_account_relogin_does_not_serve_the_old_generation(tmp_path) -> None:
    """B1: the binding is the ROW's credential + generation, not just the key.

    ``owner_key`` is identical across a re-login, so a reader matching only the
    key would hand the OLD generation's reservation to the new owner — claim,
    fetch and injection included (design S-3b).
    """

    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)

    relogged = OwnerSnapshot(
        credential_id=8,  # a new credential row
        account_id="acc_1",
        tenant_id="3f2a1111-2222-4333-8444-555566667777",
        endpoint_id="ep_1",
        canonical_base_url="https://hub.example",
        enrollment_generation=2,
    )
    assert store.pending_body_reservation_for(relogged, SESSION) is None

    turn = PendingTurn(relogged, SESSION, config_dir=tmp_path)
    assert asyncio.run(turn.open(budget=_budget(), fetch=_fetch)) is False
    assert _row(store)["state"] == "pending"


def test_a_store_refusal_on_renew_is_drift_not_a_stolen_lease(tmp_path, monkeypatch) -> None:
    """m2: a busy store must not be reported as somebody else's claim."""

    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)
    turn = _turn(tmp_path)
    assert asyncio.run(turn.open(budget=_budget(), fetch=_fetch)) is True

    def busy(*_args: Any, **_kwargs: Any) -> bool:
        raise StoreRefused("state_unreadable", "busy")

    monkeypatch.setattr(Store, "renew_body_claim", busy)
    assert turn.for_call() is None
    assert (
        turn.settle(entry_id="entry_1", error=None, stop_reason="stop", aborted=False) == "drifted"
    )
    assert _row(store)["last_error"] == "admission_drift"


def test_a_hydrated_frame_carries_the_closed_sender_kind(tmp_path, monkeypatch) -> None:
    """M1: the kind the hub reported reaches the frame's trusted section."""

    store, epoch = _store(tmp_path)
    assert _reserve(store, epoch)

    async def agent_fetch(conversation_id: str, message_id: str) -> FetchedBody | None:
        return FetchedBody(
            body=BODY_TEXT,
            body_sha256=DIGEST,
            conversation_label="#qa",
            sender_label="Ada",
            sender_kind="agent",
        )

    turn = _turn(tmp_path)
    assert asyncio.run(turn.open(budget=_budget(), fetch=agent_fetch)) is True
    text = turn.for_call()
    assert text is not None
    assert "sender_kind: agent" in text


def test_the_aside_builder_never_consults_the_pending_turn() -> None:
    """The privacy boundary is structural: the aside path has no frame input.

    ``_read_only_prompt`` is the only builder for asides and compaction, and it
    reads ``_wire_legal_snapshot`` — not the turn's claim. If a future change
    routes a frame into an aside, this assertion is what fails first.
    """

    import inspect

    from local_operator.session import session as session_mod

    source = inspect.getsource(session_mod.Session._read_only_prompt)
    assert "_pending_body_turn" not in source
    assert "transform_context" not in source
