"""Listener handoff and ordered transfer record: no kill, one leader.

Source-level proof only: the acceptors are real asyncio servers bound to a real
loopback socket, but nothing here establishes, serves or kills a live daemon.
"""

from __future__ import annotations

import asyncio
import os
import socket

import pytest

from local_operator import install_transaction as tx
from local_operator import service_handoff as handoff


def test_platform_without_fd_passing_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """A platform that cannot pass a descriptor must refuse, not improvise."""
    monkeypatch.setattr(handoff, "_fd_socket_path_supported", lambda: False)
    owner = handoff.ListenerOwner(object(), servers=[], listener=socket.socket())
    with pytest.raises(handoff.ListenerUnsupported):
        owner.pass_control(socket.socket())


def test_pause_leaves_the_master_socket_open() -> None:
    """pause_accept closes ACCEPTORS only; the transferable master survives."""

    async def scenario() -> None:
        master = socket.socket()
        master.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        master.bind(("127.0.0.1", 0))
        master.listen(8)
        owner = handoff.ListenerOwner(object(), servers=[], listener=master)
        owner._bind_acceptor_factory(
            handoff._acceptor_factory(owner.socket, asyncio.Protocol)
        )
        await owner.start_fenced()
        ino_before = owner.listener_ino
        owner.pause_accept()
        # The master must still be a usable fd, and its identity unchanged.
        assert owner.socket.fileno() >= 0
        assert owner.listener_ino == ino_before
        await owner.resume_accept()
        assert owner.accepting is True
        owner.pause_accept()
        owner.socket.close()
        master.close()

    asyncio.run(scenario())


def test_listener_travels_over_the_control_channel_with_its_record() -> None:
    """The master descriptor arrives with the frame that names it."""

    async def scenario() -> None:
        master = socket.socket()
        master.bind(("127.0.0.1", 0))
        master.listen(8)
        owner = handoff.ListenerOwner(object(), servers=[], listener=master)
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            owner.pass_control(parent, b'{"nonce":"n1"}')
            received, payload = handoff.receive_listener(child)
            try:
                assert payload == b'{"nonce":"n1"}'
                assert received.getsockname() == master.getsockname()
            finally:
                received.close()
        finally:
            parent.close()
            child.close()
            owner.socket.close()
            master.close()

    asyncio.run(scenario())


def test_suppression_never_acts_on_unreadable_or_terminal() -> None:
    # Unreadable/missing => suppress NOTHING and report the set as unreadable.
    suppress, unreadable = handoff.suppressed_occurrences(None)
    assert suppress == set() and unreadable == {"<unreadable>"}
    # Terminal => never suppressed.
    terminal = {"s1": {"state": "terminal", "terminal": "done", "recorded_at": 0.0}}
    assert handoff.suppressed_occurrences(terminal) == (set(), set())
    # Present and running => suppressed; empty map => inert.
    running = {"s1": {"state": "running", "recorded_at": 0.0, "birth": "x:y"}}
    assert handoff.suppressed_occurrences(running, same_birth=lambda _b: True) == ({"s1"}, set())
    assert handoff.suppressed_occurrences({}) == (set(), set())


def test_suppression_reclassifies_only_on_birth_mismatch_not_age() -> None:
    old = {"s1": {"state": "running", "recorded_at": 1.0, "birth": "x:y"}}
    now = handoff.OCCURRENCE_STALE_S + 2.0
    # Age alone (birth still live) keeps suppression.
    assert handoff.suppressed_occurrences(old, now=now, same_birth=lambda _b: True) == (
        {"s1"},
        set(),
    )
    # A proved birth mismatch lifts it.
    assert handoff.suppressed_occurrences(old, now=now, same_birth=lambda _b: False) == (
        set(),
        set(),
    )
    # Fresh running entry is never reclassified even with a dead birth.
    fresh = {"s1": {"state": "running", "recorded_at": now, "birth": "x:y"}}
    assert handoff.suppressed_occurrences(fresh, now=now, same_birth=lambda _b: False) == (
        {"s1"},
        set(),
    )


def test_handoff_transitions_are_ordered_and_survive_reread(tmp_path) -> None:
    operation = tx.start_operation(tmp_path, baseline="A", target="B", route="latest")
    # ack without a commit is out of order
    with pytest.raises(tx.InstallTransactionError, match="out of order"):
        tx.advance_handoff(tmp_path, operation.identifier, "ack")
    tx.advance_handoff(tmp_path, operation.identifier, "ready", to_pid=os.getpid(), nonce="n1")
    # ack is still out of order: ready does not stand in for commit
    with pytest.raises(tx.InstallTransactionError, match="out of order"):
        tx.advance_handoff(tmp_path, operation.identifier, "ack")
    tx.advance_handoff(
        tmp_path, operation.identifier, "commit", from_pid=os.getpid(), inherited=["occ-1"]
    )
    tx.advance_handoff(tmp_path, operation.identifier, "ack", listener_ino="1:2")
    row = tx.handoff_status(tmp_path, operation.identifier)
    assert row["state"] == "ack" and row["inherited"] == ["occ-1"]
    assert row["listener_ino"] == "1:2" and row["nonce"] == "n1"


def test_handoff_unknown_or_absent_operation_is_refused_not_created(tmp_path) -> None:
    root = tmp_path / "absent"
    assert tx.handoff_status(root, "0" * 32) is None
    assert not root.exists()
    with pytest.raises(tx.InstallTransactionError):
        tx.advance_handoff(root, "0" * 32, "ready")
    # An unreadable/absent operation may leave a lock file behind, but it must
    # NOT fabricate an operation record for an identity it never issued.
    assert not (root / tx.OPERATION_DIR).is_dir()


def test_commit_without_ready_is_out_of_order(tmp_path) -> None:
    operation = tx.start_operation(tmp_path, baseline="A", target="B", route="latest")
    with pytest.raises(tx.InstallTransactionError, match="out of order"):
        tx.advance_handoff(tmp_path, operation.identifier, "commit")


def test_unknown_handoff_state_is_not_a_completion(tmp_path) -> None:
    operation = tx.start_operation(tmp_path, baseline="A", target="B", route="latest")
    with pytest.raises(tx.InstallTransactionError):
        tx.advance_handoff(tmp_path, operation.identifier, "complete")


def test_unit_inventory_classifies_every_owned_central_unit() -> None:
    """The §3 table is the inventory the wiring reads; drift is a defect."""
    names = {entry.name for entry in handoff.UNITS}
    assert names == {
        "serve", "mobile", "relay", "browser",
        "hub-scheduler", "wake-monitor", "tunnel", "usage",
    }
    assert {entry.name for entry in handoff.transferable_units()} == {
        "serve", "mobile", "relay", "browser",
    }
    assert {entry.name for entry in handoff.fenced_units()} == {
        "hub-scheduler", "wake-monitor",
    }
    # tunnel has no transferable state and MUST be reported pending, not moved.
    assert [entry.name for entry in handoff.pending_units()] == ["tunnel"]
    assert handoff.unit("tunnel").counts_as_moved is False
    assert handoff.unit("usage").kind == "service"
    with pytest.raises(KeyError):
        handoff.unit("nonexistent")


def test_checkpoint_record_is_well_formed_and_phase_checked() -> None:
    record = handoff.checkpoint_record(
        session_id="s",
        turn_id="t",
        committed_message_ids=["m1", "m2"],
        completed_effects=[{"tool": "read", "call_id": "c1", "result_digest": "d"}],
        pending_effect={"tool": "write", "call_id": "c2"},
        phase="resume",
    )
    assert record["phase"] == "resume"
    assert record["committed_message_ids"] == ["m1", "m2"]
    assert record["pending_effect"] == {"tool": "write", "call_id": "c2"}
    with pytest.raises(handoff.HandoffError):
        handoff.checkpoint_record(
            session_id="s", turn_id="t", committed_message_ids=[], phase="bogus"
        )


def test_exactly_once_skips_a_committed_call_and_refuses_without_a_call_id() -> None:
    record = handoff.checkpoint_record(
        session_id="s",
        turn_id="t",
        committed_message_ids=["m1"],
        completed_effects=[{"tool": "read", "call_id": "c1", "result_digest": "d"}],
    )
    assert handoff.effect_already_committed(record, "c1") is True
    assert handoff.effect_already_committed(record, "c2") is False
    # No stable call_id: cannot be proven committed => at-least-once (declared limit).
    assert handoff.effect_already_committed(record, "") is False


def test_resumable_requires_a_readable_record_with_phase_resume() -> None:
    good = handoff.checkpoint_record(session_id="s", turn_id="t", committed_message_ids=[])
    assert handoff.resumable(good) is True
    assert handoff.resumable(None) is False  # unreadable is not permission to continue
    deferred = handoff.checkpoint_record(
        session_id="s", turn_id="t", committed_message_ids=[], phase="deferred"
    )
    assert handoff.resumable(deferred) is False


def test_checkpoint_phase_vocabulary_matches_the_transcript_module() -> None:
    from local_operator.session import transcript as transcript_mod

    assert handoff.CHECKPOINT_PHASES == transcript_mod.CHECKPOINT_PHASES
