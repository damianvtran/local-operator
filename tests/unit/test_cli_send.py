"""The ``lop send`` self-send guard (U2).

``lop send`` is a child of the launching ``lop`` TUI, so ``os.getppid()`` IS the
sending session's pid. A target that resolves to that pid means the session is
messaging itself — which would paint a ``peer message from "<own name>"`` card
as though a DIFFERENT session sent it (and, in ``--wake``/``--now`` mode,
self-trigger a turn). The guard in ``send_command`` refuses before any network
call; these tests assert it fires on the self pid and does not fire on a
different pid.
"""

from __future__ import annotations

import argparse
import os
from unittest.mock import patch

from local_operator.cli import send_command


def _send_args(**overrides) -> argparse.Namespace:
    args = argparse.Namespace(
        target="peer",
        message="note",
        pid=None,
        session=None,
        steer=False,
        wake=False,
    )
    args.__dict__.update(overrides)
    return args


class _Record:
    """The minimal SessionRecord shape the guard and sender identity touch."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.session_id = "s1"
        self.conversation_name = "this session"
        self.model_label = "test/model"
        self.cwd = "/tmp"
        self.control_port = 1
        self.control_key = "k"


def test_self_send_is_refused_before_any_network_call(capsys) -> None:
    """A target whose pid is os.getppid() (the launching session) is rejected
    with a clear message and never reaches ``send_peer_message``."""
    own_pid = os.getppid()
    with (
        patch("local_operator.cli._resolve_peer_target", return_value=(_Record(own_pid), [], "")),
        patch("local_operator.cli._peer_red") as red,
        patch("local_operator.mobile.peer_client.send_peer_message") as send,
    ):
        rc = send_command(_send_args())
    assert rc == 1
    red.assert_called_once()
    assert "this session" in red.call_args[0][0]
    # No delivery was attempted: the guard is ahead of the asyncio.run dial.
    send.assert_not_called()


def test_send_to_a_different_pid_is_not_a_self_send(capsys) -> None:
    """A target pid that is NOT the launching session's pid passes the guard and
    proceeds to delivery (here forced to fail cheaply so no real socket opens)."""
    other_pid = os.getppid() + 9999
    with (
        patch("local_operator.cli._resolve_peer_target", return_value=(_Record(other_pid), [], "")),
        patch("local_operator.cli._peer_red") as red,
        patch(
            "local_operator.mobile.peer_client.send_peer_message",
            side_effect=ConnectionError("no real session"),
        ),
    ):
        rc = send_command(_send_args())
    # Delivery was attempted (the guard did not short-circuit) and failed
    # softly — the exact "could not deliver" path, not the self-send refusal.
    assert rc == 1
    assert red.called
    assert "this session" not in red.call_args[0][0]
