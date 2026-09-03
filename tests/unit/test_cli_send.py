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


def test_typed_body_beside_a_pipe_is_refused_not_silently_chosen(monkeypatch) -> None:
    """A typed body AND a piped body with a selector is REFUSED.

    This deliberately supersedes proposal §8 case 10, which specified that the
    typed argument wins. Review round 1 (BLOCKER-1) showed the "pick a winner"
    rule cannot be implemented safely: the binder has to choose the positional's
    meaning before it can know whether a pipe carried anything, so whichever
    body loses is discarded SILENTLY at exit 0. Two candidate bodies is the same
    shape as two candidate recipients and gets the same answer — refuse. Nothing
    is delivered, so the user cannot lose a payload without being told.
    """
    monkeypatch.setattr("sys.stdin", io.StringIO("PIPED BODY\n"))
    args = _parse_send(["--pid", "123", "TYPED BODY"])
    with (
        patch("local_operator.cli._resolve_peer_target") as resolve,
        patch("local_operator.mobile.peer_client.send_peer_message") as send,
        patch("local_operator.cli._peer_red") as red,
    ):
        rc = send_command(args)
    assert rc == 1
    send.assert_not_called()
    resolve.assert_not_called()
    assert "ambiguous body" in red.call_args[0][0]


def test_piped_body_with_a_target_and_selector_never_delivers_the_target(monkeypatch) -> None:
    """BLOCKER-1 regression guard: the silent payload-loss case.

    ``git log --stat | lop send alpha --pid 10`` — the exact form the CLI's own
    ambiguity guidance leads a user to type — delivered the literal string
    ``'alpha'`` with exit 0, discarding the commit log. The assertion that
    matters is that the TARGET NAME is never delivered as a body: a wrong body
    sent successfully is undetectable from the output, which is what made this
    worse than the loud bug the PR set out to fix.
    """
    monkeypatch.setattr("sys.stdin", io.StringIO("commit abc123\n 5 files changed\n"))
    args = _parse_send(["alpha", "--pid", "10"])
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
    assert "ambiguous body" in message
    # Both recoveries are shown, and neither is the refused target+selector pair.
    assert "`lop send --pid 10`" in message
    assert "to send the piped input" in message


def test_a_selector_alone_with_a_pipe_still_delivers_the_piped_body(monkeypatch) -> None:
    """The documented workaround must survive the BLOCKER-1 fix: with no
    positional at all, the piped bytes are the body and they are delivered."""
    monkeypatch.setattr("sys.stdin", io.StringIO("commit abc123\n 5 files changed\n"))
    args = _parse_send(["--pid", "123"])
    record = _Record(os.getppid() + 9999)
    with (
        patch("local_operator.cli._resolve_peer_target", return_value=(record, [], "")),
        patch("local_operator.mobile.peer_client.send_peer_message") as send,
    ):
        send_command(args)
    assert send.call_args.kwargs["text"] == "commit abc123\n 5 files changed\n"


def test_an_empty_redirect_is_not_a_piped_body(monkeypatch) -> None:
    """``lop send --pid N "hi" </dev/null`` must still deliver ``hi``.

    ``isatty()`` is False for an empty redirect exactly as it is for a pipe
    carrying data, so keying the rule off "stdin is not a tty" would turn every
    ``</dev/null`` invocation into an ambiguity error. Only real bytes count.
    """
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    args = _parse_send(["--pid", "123", "hi"])
    record = _Record(os.getppid() + 9999)
    with (
        patch("local_operator.cli._resolve_peer_target", return_value=(record, [], "")),
        patch("local_operator.mobile.peer_client.send_peer_message") as send,
    ):
        send_command(args)
    assert send.call_args.kwargs["text"] == "hi"


def test_no_selector_keeps_the_historical_stdin_precedence(monkeypatch) -> None:
    """Without a selector the grammar is unchanged: both positionals are filled,
    so the typed body wins over the pipe with no ambiguity to resolve."""
    monkeypatch.setattr("sys.stdin", io.StringIO("PIPED BODY\n"))
    args = _parse_send(["peer", "TYPED BODY"])
    record = _Record(os.getppid() + 9999)
    with (
        patch("local_operator.cli._resolve_peer_target", return_value=(record, [], "")),
        patch("local_operator.mobile.peer_client.send_peer_message") as send,
    ):
        send_command(args)
    assert send.call_args.kwargs["text"] == "TYPED BODY"


def test_candidate_guidance_tells_the_user_to_replace_the_target(monkeypatch, capsys) -> None:
    """BLOCKER-2 regression guard for the CLI surface.

    The old wording was ``disambiguate with --pid:``, and the invocation a user
    naturally forms from it — re-run the command with the flag appended — is the
    ``NAME BODY --pid N`` form the ambiguity guard refuses. The guidance must
    name the REPLACEMENT, and must not tell the user to add a flag.
    """
    monkeypatch.setattr("sys.stdin", _FakeTtyStdin())
    args = _parse_send(["alpha", "BODY"])
    candidates = [_Record(10), _Record(20)]
    with patch("local_operator.cli._resolve_peer_target", return_value=(None, candidates, "")):
        rc = send_command(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert "replace the target" in err
    assert "disambiguate with --pid" not in err
    # The worked example is the form that actually succeeds: selector + body,
    # with the target gone.
    assert "`lop send --pid 10 BODY`" in err


def test_shell_metacharacters_in_the_retype_suggestion_are_quoted() -> None:
    """Architect MINOR-1: the suggestions are meant to be retypeable, so a body
    containing backticks or ``$`` must not be mangled or command-substituted
    when pasted into a shell."""
    args = _parse_send(["alpha", "run `whoami` for $HOME", "--pid", "10"])
    _t, _m, error = _bind_send_positionals(args, None)
    # shlex.quote wraps it in single quotes, which bash does not expand.
    assert "'run `whoami` for $HOME'" in error
    assert '"run `whoami` for $HOME"' not in error


def test_a_blank_target_beside_a_selector_is_absence_not_a_conflict() -> None:
    """MAJOR-2: the CLI and the shared core must agree that an empty or
    whitespace target is ABSENCE. The core strips before comparing
    (``peer_send.py``), so the binder does too — otherwise the two layers
    disagree about the same input, which is the drift this PR exists to end."""
    target, message, error = _bind(["", "BODY", "--pid", "123"])
    assert (target, message, error) == (None, "BODY", "")
    target, message, error = _bind(["   ", "BODY", "--pid", "123"])
    assert (target, message, error) == (None, "BODY", "")


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_session_selector_is_an_error_not_a_grammar_switch(blank: str) -> None:
    """MAJOR-1: ``--session ''`` and ``--session '   '`` used to bind oppositely
    — the first fell through to the no-selector grammar (silently making the
    body a TARGET), the second was honoured and pasted whitespace into the error
    text. A selector the user explicitly passed but left blank is an error."""
    _t, _m, error = _bind(["--session", blank, "hello"])
    assert "empty --session" in error


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


class _BinaryStdin:
    """A non-tty stdin carrying bytes that are not valid UTF-8.

    Models a real `some-binary-producer | lop send …` (a gzip or an image piped
    by mistake). The real `sys.stdin` is a strict-UTF-8 text wrapper over a
    `.buffer`, so the fake has to expose the same shape for the decode path to
    be exercised rather than bypassed.
    """

    def __init__(self, raw: bytes) -> None:
        self.buffer = io.BytesIO(raw)

    def isatty(self) -> bool:
        return False

    def read(self) -> str:  # pragma: no cover - the buffer path is the real one
        raise AssertionError("the binary path must read sys.stdin.buffer")


def test_a_non_utf8_pipe_is_a_clean_error_not_a_traceback(monkeypatch) -> None:
    """Review round 2, MAJOR-1.

    Reading stdin ahead of resolution widened the blast radius of a decode
    failure: invocations that never consume the bytes (an unresolvable pid here)
    used to print a clean red line because base never read at all, and began
    raising UnicodeDecodeError out of ``send_command`` as a traceback. An
    uncaught traceback is never an acceptable user-visible failure — the same U1
    rule the delivery path already follows — so the read decodes leniently.
    """
    monkeypatch.setattr("sys.stdin", _BinaryStdin(b"\x1f\x8b\x08\x00\xa8\x8d\xff\xfe"))
    args = _parse_send(["--pid", "777777"])
    with (
        patch(
            "local_operator.cli._resolve_peer_target",
            return_value=(None, [], "no session found with pid 777777"),
        ),
        patch("local_operator.mobile.peer_client.send_peer_message") as send,
        patch("local_operator.cli._peer_red") as red,
    ):
        rc = send_command(args)
    # The pre-existing error survives verbatim; the decode never surfaces.
    assert rc == 1
    send.assert_not_called()
    assert red.call_args[0][0] == "no session found with pid 777777"


def test_a_non_utf8_pipe_is_delivered_as_replacement_characters(monkeypatch) -> None:
    """A binary body that reaches delivery is SENT as U+FFFD mojibake, not
    refused (QA round 3, Q6).

    ``validate_peer_body`` rejects only an empty or over-cap body, and
    replacement characters are neither, so this path ends at rc=0 with degraded
    text. Pinned explicitly because it is easy to assume the lenient decode
    "falls through to a refusal" — it does not, and the earlier name and comment
    here claimed exactly that. The prior behaviour was an uncaught traceback, so
    delivering degraded text is the improvement; refusing would require
    inventing a threshold for how much mojibake is too much.
    """
    monkeypatch.setattr("sys.stdin", _BinaryStdin(b"\xa8\x8d\xff\xfe"))
    args = _parse_send(["--pid", "123"])
    record = _Record(os.getppid() + 9999)
    with (
        patch("local_operator.cli._resolve_peer_target", return_value=(record, [], "")),
        patch("local_operator.mobile.peer_client.send_peer_message") as send,
    ):
        rc = send_command(args)
    assert rc == 0
    # Delivered as replacement characters, never as a crash.
    assert "\ufffd" in send.call_args.kwargs["text"]


def test_the_worked_example_never_echoes_a_piped_payload(monkeypatch, capsys) -> None:
    """Architect round 2, MINOR-1.

    The disambiguation example interpolated the body, so a piped payload was
    dumped back to stderr in full (measured: a 10 KB pipe produced 10,147 bytes
    of stderr) with an invitation to retype it — when the correct recovery for a
    piped body is to RE-PIPE. The example now shows the shape of the working
    command instead.
    """
    payload = "x" * 10000
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))
    args = _parse_send(["alpha"])
    candidates = [_Record(10), _Record(20)]
    with patch("local_operator.cli._resolve_peer_target", return_value=(None, candidates, "")):
        rc = send_command(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert payload not in err
    assert "<your piped input> | lop send --pid 10" in err
    # The candidate list above it stays correct and complete.
    assert "--pid 10" in err and "--pid 20" in err
    assert len(err) < 500


def test_a_long_typed_body_becomes_a_placeholder_not_a_truncation(monkeypatch, capsys) -> None:
    """QA round 3, Q7.

    A long body is replaced by ``'<your message>'``, not truncated. The line
    exists to show the SHAPE of the command and the user still has their own
    body one line up — but a truncation would PASTE AND RUN, silently
    delivering a 60-character stub ending in "...". A placeholder cannot be
    mistaken for a runnable command, which is the same reason the piped branch
    prints ``<your piped input>``.
    """
    monkeypatch.setattr("sys.stdin", _FakeTtyStdin())
    args = _parse_send(["alpha", "y" * 400])
    with patch("local_operator.cli._resolve_peer_target", return_value=(None, [_Record(10)], "")):
        send_command(args)
    err = capsys.readouterr().err
    assert "y" * 400 not in err
    assert "'<your message>'" in err
    # No truncated stub that could be pasted and delivered verbatim.
    assert "yyy" not in err


def test_a_short_typed_body_is_still_shown_in_full(monkeypatch, capsys) -> None:
    """The common case must stay genuinely helpful: a short typed body is
    reproduced verbatim so the suggestion is copy-pasteable."""
    monkeypatch.setattr("sys.stdin", _FakeTtyStdin())
    args = _parse_send(["alpha", "gates are green"])
    with patch("local_operator.cli._resolve_peer_target", return_value=(None, [_Record(10)], "")):
        send_command(args)
    assert "e.g. `lop send --pid 10 'gates are green'`" in capsys.readouterr().err


def test_the_guide_quotes_the_ambiguity_error_verbatim() -> None:
    """The §6 doc lockstep, asserted rather than claimed (review round 1,
    MINOR-1).

    GUIDE.md prints a transcript of the ambiguity refusal. Every other test here
    asserts SUBSTRINGS of that error, so a rewording would drift the guide
    silently — precisely the failure mode the lockstep exists to prevent. This
    reconstructs the guide's quoted block (unwrapping its hard line breaks) and
    compares it to what the binder actually produces.
    """
    from pathlib import Path

    guide = (
        Path(__file__).resolve().parents[2]
        / "local_operator"
        / "guides"
        / "peer-messaging"
        / "GUIDE.md"
    ).read_text()

    marker = '$ lop send "release cutter" "gates are green" --pid 12345'
    assert marker in guide, "the guide no longer shows the refusal transcript"
    block = guide[guide.index(marker) : guide.index("```", guide.index(marker))]
    # The guide hard-wraps for readability; the error itself is one line.
    documented = " ".join(block.strip().split("\n")[1:])

    args = _parse_send(["release cutter", "gates are green", "--pid", "12345"])
    _t, _m, actual = _bind_send_positionals(args, None)
    assert documented == actual


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
