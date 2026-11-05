"""A confirmed failed handoff must not consume an admitted conversation card.

Exercise the real Store, conversation handler and wake adapter. Only hub,
screening and the final peer transport are synthetic; no runtime is launched.
"""

from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace

import pytest

from local_operator.mobile import peer_send
from local_operator.team_bridge.connector import coalesce, conversation, wake
from local_operator.team_bridge.store import Store
from tests.unit.team_bridge import test_conversation_delivery as rig

_instant_wakes = rig._instant_wakes


def _setup(tmp_path, monkeypatch, outcomes):
    owner = rig._owner("a", 1)
    config_dir = tmp_path / "a"
    config_dir.mkdir()
    rig._enroll(config_dir, owner, "sess_a")
    server = rig.FakeServer(notify={"bep_a": "mute"}, active={"bep_a": True}, mentioned={"bep_a"})
    rig._wire_rig(monkeypatch, server, {"a": (owner, config_dir)})
    monkeypatch.setattr(conversation, "deliver_wake", wake.deliver_wake)
    monkeypatch.setattr(peer_send, "session_has_durable_history", lambda *_a, **_k: True)
    attempts, accepted = [], []

    async def transport(*args, **kwargs):
        attempts.append(kwargs)
        result = outcomes.pop(0)
        if isinstance(result, BaseException):
            raise result
        if result != peer_send.DELIVERY_FAILED:
            accepted.append(kwargs)
        return peer_send.DeliveryOutcome(
            result,
            "PRIVATE-TRANSPORT-DETAIL",
            f"peer-{len(attempts)}",
            "unconfirmed",
            1,
            "peer_refused" if result == peer_send.DELIVERY_FAILED else "",
            "engaged",
            "sess_a",
        )

    monkeypatch.setattr(peer_send, "deliver_peer_message_outcome", transport)

    def handle():
        return asyncio.run(
            conversation.handle_conversation_available(
                owner,
                rig._ref(server),
                config_dir=config_dir,
                auth_store=None,
                access=SimpleNamespace(credential_id=owner.credential_id),
            )
        )

    return owner, config_dir, attempts, accepted, handle


def test_unknown_outcome_shape_fails_closed_and_typed_failure_raises(tmp_path, monkeypatch):
    """Unknown is never delivered: only a typed result may settle the card.

    An outcome object carrying ``detail`` but no ``state`` is contract drift, not
    a success report. The strict read raises, the handler's existing hold arm
    keeps the card owed, and no posted marker is written — so a future refactor
    cannot silently consume the message. The canonical failed state still raises.
    """
    monkeypatch.setattr(peer_send, "session_has_durable_history", lambda *_a, **_k: True)
    sess_owner = SimpleNamespace(account_id="account")
    calls = []

    async def malformed(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(detail="delivered")  # no state: not a real outcome

    monkeypatch.setattr(peer_send, "deliver_peer_message_outcome", malformed)
    with pytest.raises(AttributeError):
        asyncio.run(
            wake.deliver_wake(
                sess_owner, "synthetic card", session_id="sess_a", config_dir=tmp_path
            )
        )
    assert len(calls) == 1, "the drift arises AFTER the primitive returned"

    # Through the real handler that same drift is a retryable hold, not a card.
    owner, config_dir, _attempts, accepted, handle = _setup(
        tmp_path, monkeypatch, [peer_send.DELIVERY_DELIVERED]
    )
    monkeypatch.setattr(peer_send, "deliver_peer_message_outcome", malformed)
    outcome = handle()
    assert outcome["action"] == "hold" and outcome["code"] == "conversation_failed"
    row = rig._reservation_rows(config_dir)[0]
    assert row["state"] == "pending" and row["card_posted_at"] is None
    assert len(calls) == 2 and not accepted

    async def typed_failure(*args, **kwargs):
        return peer_send.DeliveryOutcome(
            peer_send.DELIVERY_FAILED,
            "not delivered",
            "peer-1",
            peer_send.WAKE_NOT_REQUESTED,
            1,
            "peer_refused",
            "engaged",
            "sess_a",
        )

    monkeypatch.setattr(peer_send, "deliver_peer_message_outcome", typed_failure)
    with pytest.raises(peer_send.DeliveryFailed):
        asyncio.run(
            wake.deliver_wake(
                sess_owner, "synthetic card", session_id="sess_a", config_dir=tmp_path
            )
        )


@pytest.mark.parametrize("failure", ["typed", "exception", "cancel"])
def test_failed_handoff_remains_owed_and_retries_once(tmp_path, monkeypatch, caplog, failure):
    """The old adapter returned a failed outcome's prose as success and lost this card."""
    first = {
        "typed": peer_send.DELIVERY_FAILED,
        "exception": OSError("PRIVATE-TRANSPORT-DETAIL"),
        "cancel": asyncio.CancelledError(),
    }[failure]
    owner, config_dir, attempts, accepted, handle = _setup(
        tmp_path, monkeypatch, [first, peer_send.DELIVERY_DELIVERED]
    )
    with caplog.at_level(logging.WARNING):
        if failure == "cancel":
            with pytest.raises(asyncio.CancelledError):
                handle()
        else:
            assert handle()["action"] == "hold"
    row = rig._reservation_rows(config_dir)[0]
    assert row["state"] == "pending" and row["card_posted_at"] is None
    assert not accepted
    claim = Store(config_dir).claim_status(owner, rig.MESSAGE_ID)
    assert claim is not None and claim["state"] == "released"
    assert "PRIVATE-TRANSPORT-DETAIL" not in caplog.text
    # Every handle opens a new Store. No pending in-memory claim is needed for recovery.
    assert handle()["action"] == "delivered"
    assert handle()["action"] == "duplicate"
    assert len(attempts) == 2 and len(accepted) == 1
    assert attempts[0]["text"] == attempts[1]["text"]
    assert attempts[0]["bridge_meta"] == attempts[1]["bridge_meta"]
    assert rig.MESSAGE_ID in accepted[0]["text"]
    assert rig._reservation_rows(config_dir)[0]["card_posted_at"] is not None


@pytest.mark.parametrize(
    "state",
    [
        peer_send.DELIVERY_DELIVERED,
        peer_send.DELIVERY_MAILBOX,
        peer_send.DELIVERY_UNCONFIRMED,
    ],
)
def test_accepted_or_uncertain_handoff_is_not_resent(tmp_path, monkeypatch, state):
    """An unknown old posted marker is not evidence of non-delivery: never reopen it."""
    _owner, config_dir, attempts, accepted, handle = _setup(tmp_path, monkeypatch, [state])
    assert handle()["action"] == "delivered"  # handoff, not a claim of model execution
    assert rig._reservation_rows(config_dir)[0]["card_posted_at"] is not None
    assert handle()["action"] == "duplicate"
    assert handle()["action"] == "duplicate"
    assert len(attempts) == len(accepted) == 1


def test_coalesced_failed_handoff_recovers_after_connector_restart(tmp_path, monkeypatch):
    """The durable owed-card hold must survive losing the coalescer's memory."""
    owner, config_dir, attempts, accepted, _handle = _setup(
        tmp_path, monkeypatch, [peer_send.DELIVERY_FAILED, peer_send.DELIVERY_DELIVERED]
    )
    clock = rig._Clock()
    monkeypatch.setattr(time, "time", clock)

    def iteration():
        return asyncio.run(
            rig.loop_mod.run_iteration(
                owner,
                config_dir=config_dir,
                access=SimpleNamespace(credential_id=owner.credential_id),
                auth_store=None,
            )
        )

    result = iteration()
    assert result["coalesced_wakes"][0]["action"] == "hold"
    assert rig._reservation_rows(config_dir)[0]["card_posted_at"] is None
    assert not accepted
    assert rig._hold_rows(config_dir)
    monkeypatch.setattr(coalesce, "CONNECTOR_COALESCER", coalesce.WakeCoalescer(window_s=0))
    clock.now += rig._PAST_A_LEASE
    iteration()
    clock.now += rig._PAST_A_LEASE
    iteration()
    assert len(attempts) == 2 and len(accepted) == 1
    assert rig._reservation_rows(config_dir)[0]["card_posted_at"] is not None
    assert rig._hold_rows(config_dir) == []


def test_no_history_designated_neighbor_keeps_message_errand(tmp_path, monkeypatch):
    """The cold designated-session branch still carries one mailbox errand, not a bare wake."""
    from local_operator.session.runtime import launch

    monkeypatch.setattr(peer_send, "session_has_durable_history", lambda *_a, **_k: False)
    errands = []

    async def engage(session_id, cwd, errand, **kwargs):
        errands.append(errand)
        return SimpleNamespace(state="started")

    monkeypatch.setattr(launch, "engage_runtime", engage)
    assert "engaged" in asyncio.run(
        wake.deliver_wake(
            SimpleNamespace(account_id="account"),
            "synthetic card",
            session_id="sess_a",
            config_dir=tmp_path,
        )
    )
    assert len(errands) == 1
    assert errands[0].text == "synthetic card"
    assert errands[0].mode == "mailbox" and errands[0].wake is True
