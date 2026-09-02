"""``lop send`` argument grammar and the self-send guard (U2).

The first half of this file pins the ARGUMENT GRAMMAR: which positional means
what once a ``--pid``/``--session`` selector is present. It is a regression
guard for a shipped defect where ``lop send --pid N "hello"`` bound "hello" to
``target``, left ``message`` empty and exited 1 with "no message given" — while
``lop send other "body" --pid N`` delivered to pid N and reported success, a
wrong-recipient hazard. See ``docs/design/peer-send.md`` §4.3.

The second half pins the self-send guard.

``lop send`` is usually a child of the launching ``lop`` TUI, but not always:
run from an agent's bash tool or through a shell wrapper it is a grandchild or
lower, so the sending session is found by walking the process ancestry rather
than by reading ``os.getppid()``. A target that resolves to THAT pid means the
session is messaging itself — which would paint a ``peer message from "<own name>"`` card
as though a DIFFERENT session sent it (and, in ``--wake``/``--now`` mode,
self-trigger a turn). The guard in ``send_command`` refuses before any network
call; these tests assert it fires on the self pid and does not fire on a
different pid.
"""

from __future__ import annotations

import argparse
import io
import os
from unittest.mock import patch

import pytest

from local_operator.cli import _bind_send_positionals, build_cli_parser, send_command


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


class _FakeTtyStdin:
    """A stdin that claims to be a terminal.

    The "no message given" path is only reachable when stdin is a TTY, and the
    test runner's stdin is not one — so the interactive case has to be modelled
    rather than inherited, otherwise the test silently exercises the pipe path.
    """

    def isatty(self) -> bool:
        return True

    def read(self) -> str:  # pragma: no cover - reaching this is the bug
        raise AssertionError("stdin must not be read when it is a tty")


def _parse_send(argv: "list[str]") -> argparse.Namespace:
    """Parse a real ``lop send`` command line through the production parser.

    The grammar cases go through ``build_cli_parser`` rather than a synthetic
    Namespace on purpose: the defect lived in how argparse SLOTTED the
    positionals, so a hand-built namespace would assert the fix while skipping
    the thing that was broken.
    """
    return build_cli_parser().parse_args(["send"] + argv)


def _bind(argv: "list[str]") -> "tuple[str | None, str | None, str]":
    return _bind_send_positionals(_parse_send(argv))


# --- Case 1-3: a selector present means a lone positional is the BODY ---


def test_pid_with_one_positional_binds_it_as_the_message() -> None:
    """THE DEFECT. ``lop send --pid N "body"`` used to bind "body" to target and
    then fail with "no message given" — an error that was factually false."""
    target, message, error = _bind(["--pid", "123", "hello there"])
    assert (target, message, error) == (None, "hello there", "")


def test_pid_with_wake_binds_the_message_and_keeps_the_flag() -> None:
    """The form the peer-messaging guide documented verbatim, which could not work."""
    args = _parse_send(["--pid", "123", "--wake", "the deploy finished; verify prod"])
    target, message, error = _bind_send_positionals(args)
    assert (target, message, error) == (None, "the deploy finished; verify prod", "")
    assert args.wake is True


def test_session_with_one_positional_binds_it_as_the_message() -> None:
    target, message, error = _bind(["--session", "sess-abc", "hello there"])
    assert (target, message, error) == (None, "hello there", "")


# --- Case 4: no selector means the historical grammar, unchanged ---


def test_two_positionals_without_a_selector_are_target_then_message() -> None:
    target, message, error = _bind(["peer", "hello there"])
    assert (target, message, error) == ("peer", "hello there", "")


# --- Case 5-6: a positional target AND a selector is refused, nothing delivered ---


def test_positional_target_plus_pid_is_refused_and_delivers_nothing() -> None:
    """The wrong-recipient hazard. The assertion that matters is the NON-CALL:
    the old behaviour delivered to --pid while naming another session and exited
    0, so an exit code alone would not prove the hazard is gone."""
    args = _parse_send(["some-other-session", "BODY", "--pid", "123"])
    with (
        patch("local_operator.cli._resolve_peer_target") as resolve,
        patch("local_operator.mobile.peer_client.send_peer_message") as send,
        patch("local_operator.cli._peer_red") as red,
    ):
        rc = send_command(args)
    assert rc == 1
    send.assert_not_called()
    # Nothing was even resolved: the refusal is ahead of the registry scan.
    resolve.assert_not_called()
    message = red.call_args[0][0]
    assert "ambiguous recipient" in message
    # The error names BOTH readings and prints two retypeable commands, which is
    # what makes the intentional break self-documenting (proposal §9 risk 1).
    assert "some-other-session" in message
    assert "--pid 123" in message
    assert "to address by pid" in message
    assert "to address by name" in message


def test_positional_target_plus_session_is_refused_and_delivers_nothing() -> None:
    args = _parse_send(["some-other-session", "BODY", "--session", "sess-abc"])
    with (
        patch("local_operator.cli._resolve_peer_target") as resolve,
        patch("local_operator.mobile.peer_client.send_peer_message") as send,
        patch("local_operator.cli._peer_red") as red,
    ):
        rc = send_command(args)
    assert rc == 1
    send.assert_not_called()
    resolve.assert_not_called()
    message = red.call_args[0][0]
    assert "ambiguous recipient" in message
    assert "--session sess-abc" in message
    assert "to address by session id" in message


# --- Case 7: the two selectors are mutually exclusive at the parser ---


def test_pid_and_session_together_is_a_parser_error() -> None:
    """argparse's own mutually-exclusive group rejects the pair, which is a
    better error than a hand-written check and keeps the branch out of the
    binder."""
    with pytest.raises(SystemExit) as exc:
        _parse_send(["--pid", "123", "--session", "sess-abc", "body"])
    assert exc.value.code == 2


# --- Case 8-9: the stdin paths are preserved exactly ---


def test_name_alone_leaves_the_message_unbound_for_stdin() -> None:
    """``cmd | lop send NAME`` — the documented pipe path."""
    target, message, error = _bind(["peer"])
    assert (target, message, error) == ("peer", None, "")


def test_selector_alone_leaves_the_message_unbound_for_stdin() -> None:
    """``cmd | lop send --pid N`` — the workaround the old guide taught. It is
    pinned explicitly so a future refactor of the binder cannot drop it."""
    target, message, error = _bind(["--pid", "123"])
    assert (target, message, error) == (None, None, "")


# --- Case 10-11: body precedence and the now-truthful "no message" error ---


def test_typed_body_wins_over_piped_stdin(monkeypatch, capsys) -> None:
    """Today ``echo X | lop send --pid N "Y"`` sent X, because "Y" was eaten as
    the target and never looked at. After the fix the typed argument wins, which
    matches every other CLI and rule 4 of the grammar."""
    monkeypatch.setattr("sys.stdin", io.StringIO("PIPED BODY\n"))
    args = _parse_send(["--pid", "123", "TYPED BODY"])
    record = _Record(os.getppid() + 9999)
    with (
        patch("local_operator.cli._resolve_peer_target", return_value=(record, [], "")),
        patch("local_operator.mobile.peer_client.send_peer_message") as send,
    ):
        send_command(args)
    assert send.call_args.kwargs["text"] == "TYPED BODY"


def test_selector_with_no_body_on_a_tty_still_reports_no_message(monkeypatch) -> None:
    """The "no message given" error survives — but is now reachable only when it
    is TRUE, i.e. when no body was typed and stdin is a terminal."""

    monkeypatch.setattr("sys.stdin", _FakeTtyStdin())
    args = _parse_send(["--pid", "123"])
    record = _Record(os.getppid() + 9999)
    with (
        patch("local_operator.cli._resolve_peer_target", return_value=(record, [], "")),
        patch("local_operator.mobile.peer_client.send_peer_message") as send,
        patch("local_operator.cli._peer_red") as red,
    ):
        rc = send_command(args)
    assert rc == 1
    send.assert_not_called()
    assert "no message given" in red.call_args[0][0]


# --- Case 12: the regression guard against "simplifying" the grammar ---


def test_flag_then_body_still_parses_with_a_positional_target() -> None:
    """``lop send NAME --wake "body"`` and ``--now "body"`` must keep working.

    This is the guard against two "obvious" simplifications that are both
    broken, and it is a PARSER-level test because that is where they break:

    * ``message`` as ``nargs="*"`` — a single ``nargs="*"`` positional cannot
      absorb words that follow an optional flag. This exact argv raises
      SystemExit(2), "unrecognized arguments: act now".
    * ``message`` as ``argparse.REMAINDER`` — binds ``message=['--wake', 'act
      now']``, swallowing the FLAG into the body: the command exits 0, delivers,
      and silently degrades wake to mailbox with "--wake" pasted into the text.

    Both were measured, not assumed. See ``docs/design/peer-send.md`` §4.3.
    """
    args = _parse_send(["peer", "--wake", "act now"])
    assert (args.target, args.message, args.wake) == ("peer", "act now", True)

    args = _parse_send(["peer", "--now", "stop, do X instead"])
    assert (args.target, args.message, args.steer) == ("peer", "stop, do X instead", True)


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


def test_self_send_is_refused_through_a_multi_hop_ancestry(capsys) -> None:
    """The guard and the sender identity must agree about who "this session" is.

    Reproduces the real shape of `lop send` from an agent's bash tool: session
    -> sh -> CLI. The guard used to compare the bare os.getppid() (the
    intermediate shell) while identity walked the ancestry to the session, so a
    self-send slipped through the guard and was then delivered carrying the
    session's OWN name — the mislabelled card the guard exists to prevent.
    """
    import local_operator.mobile.peer_send as peer_send_mod

    # send_command's identity walk starts at os.getppid() (the CLI's parent),
    # so the modelled chain starts there: sh -> session.
    shell_pid = os.getppid()
    session_pid = shell_pid + 2
    chain = {shell_pid: session_pid, session_pid: 1}

    record = _Record(session_pid)

    with (
        patch.object(peer_send_mod, "_parent_pid", lambda pid: chain.get(pid)),
        patch.object(
            peer_send_mod, "_record_for_pid", lambda pid: record if pid == session_pid else None
        ),
        patch("local_operator.cli._resolve_peer_target", return_value=(record, [], "")),
    ):
        code = send_command(_send_args())

    assert code == 1
    err = capsys.readouterr().err
    assert "that target is this session" in err


def test_a_grandparent_session_does_not_block_a_send_to_a_third_party(capsys) -> None:
    """The widened guard must not start refusing legitimate sends: only the
    resolved sending session is off limits, not every ancestor."""
    import local_operator.mobile.peer_send as peer_send_mod

    shell_pid = os.getppid()
    session_pid = shell_pid + 2
    other = _Record(session_pid + 500)
    chain = {shell_pid: session_pid, session_pid: 1}

    with (
        patch.object(peer_send_mod, "_parent_pid", lambda pid: chain.get(pid)),
        patch.object(
            peer_send_mod,
            "_record_for_pid",
            lambda pid: _Record(session_pid) if pid == session_pid else None,
        ),
        patch("local_operator.cli._resolve_peer_target", return_value=(other, [], "")),
    ):
        code = send_command(_send_args())

    err = capsys.readouterr().err
    assert "that target is this session" not in err
    # It got past the guard and failed on the dial instead (no live peer here).
    assert code == 1
