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


def test_inherited_occurrence_exclusions_read_from_the_record() -> None:
    assert handoff._scheduler_occurrence_exclusions(None) == []
    assert handoff._scheduler_occurrence_exclusions({"handoff": {"inherited": ["a", 3]}}) == ["a"]


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
