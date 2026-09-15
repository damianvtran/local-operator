"""``lop refresh`` — the CLI front end of the rotation path.

What is pinned here is the CLI's OWN contract, not the ladder or the op (those
are ``tests/unit/session/runtime/test_control.py``): the exit-code triple
(0 every target answered / 1 no match / 2 partial — somebody's socket was
silent), the ``--all`` behaviour (no confirmation gate: ending nothing needs no
consent, but the listing still prints), the ``--json`` shape, and that a BUSY
target is a success rather than a failure, because the runtime that owns the
turn is the one that will do the moving.

The stubs mirror the real signatures keyword for keyword, and record what they
were handed, for the reason ``tests/unit/test_cli_stop.py`` spells out: a stub
narrower than the function it replaces passes until a caller uses a keyword for
real.
"""

from __future__ import annotations

import argparse
import json
from typing import Any
from unittest.mock import patch

from local_operator.cli import refresh_command
from local_operator.session.runtime import control
from local_operator.session.runtime.control import RefreshOutcome

_SEEN: dict[str, Any] = {}


def _args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "target": None,
        "pid": None,
        "session": None,
        "refresh_all": False,
        "json": False,
        "timeout": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class _Record:
    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.session_id = "s1"
        self.conversation_name = "the agent"
        self.model_label = "test/model"
        self.cwd = "/tmp"
        self.control_port = 1
        self.control_key = "k"
        self.version = "0.54.48"


def _outcome(method: str, pid: int = 4242, line: str | None = None) -> RefreshOutcome:
    return RefreshOutcome(
        pid=pid,
        session_id="s1",
        name="the agent",
        method=method,
        line=line or f'{method} "the agent"',
    )


async def _fake_refresh(record, *, timeout_s):  # noqa: ANN001, ANN202
    _SEEN["timeout_s"] = timeout_s
    _SEEN["pid"] = record.pid
    return _outcome("moved")


async def _fake_refresh_all(  # noqa: ANN001, ANN202
    *, timeout_s, own_pid=None, only_pids=None, _root=None
):
    _SEEN["timeout_s"] = timeout_s
    _SEEN["own_pid"] = own_pid
    return [
        _outcome("moved", pid=1, line='"one" (pid 1, running 0.54.46) is retiring now'),
        _outcome("busy", pid=2, line='"two" (pid 2, running 0.54.46) has a turn in flight'),
        _outcome("current", pid=3, line='"three" (pid 3, running 0.54.48) already runs it'),
    ]


def _single_target(record: _Record | None = None) -> "tuple[Any, list[Any], str]":
    return record or _Record(), [], ""


def test_a_moved_target_exits_0_and_prints_the_receipt(capsys) -> None:
    with (
        patch("local_operator.cli._resolve_stop_target", lambda _a: _single_target()),
        patch.object(control, "refresh_session", _fake_refresh),
    ):
        assert refresh_command(_args(target="the agent")) == 0
    out = capsys.readouterr().out
    assert "moved" in out
    assert _SEEN["timeout_s"] == control.DEFAULT_REFRESH_TIMEOUT_S


def test_a_busy_target_is_a_queued_move_not_a_failure(capsys) -> None:
    """The whole point of the command: a busy session is left working.

    Its exit code must say "nothing is wrong here" — the runtime retires by
    itself when the turn ends — so a script that rotates a fleet does not treat
    a healthy working session as an error.
    """

    async def _busy(record, *, timeout_s):  # noqa: ANN001, ANN202
        return _outcome("busy", line='"the agent" has a turn in flight')

    with (
        patch("local_operator.cli._resolve_stop_target", lambda _a: _single_target()),
        patch.object(control, "refresh_session", _busy),
    ):
        assert refresh_command(_args(target="the agent")) == 0
    assert "turn in flight" in capsys.readouterr().out


def test_an_unreachable_target_is_partial(capsys) -> None:
    """A silent socket is the one outcome whose move is NOT coming on its own."""

    async def _silent(record, *, timeout_s):  # noqa: ANN001, ANN202
        return _outcome("unreachable", line='"the agent" did not answer its control socket')

    with (
        patch("local_operator.cli._resolve_stop_target", lambda _a: _single_target()),
        patch.object(control, "refresh_session", _silent),
    ):
        assert refresh_command(_args(target="the agent")) == 2
    assert "did not answer" in capsys.readouterr().out


def test_no_match_exits_1(capsys) -> None:
    with patch(
        "local_operator.cli._resolve_stop_target", lambda _a: (None, [], "no session matched")
    ):
        assert refresh_command(_args(target="nope")) == 1
    assert "no session matched" in capsys.readouterr().err


def test_ambiguous_target_lists_candidates_and_exits_1(capsys) -> None:
    with patch(
        "local_operator.cli._resolve_stop_target",
        lambda _a: (None, [_Record(1), _Record(2)], ""),
    ):
        assert refresh_command(_args(target="agent")) == 1
    err = capsys.readouterr().err
    assert "2 sessions match" in err
    assert "--pid" in err


def test_all_lists_every_outcome_and_summarises(capsys) -> None:
    with (
        patch.object(control, "_rotation_targets", lambda root, own_pid=None: [_Record()] * 3),
        patch.object(control, "refresh_all", _fake_refresh_all),
    ):
        assert refresh_command(_args(refresh_all=True)) == 0
    out = capsys.readouterr().out
    assert "is retiring now" in out
    assert "has a turn in flight" in out
    assert "3 sessions: 1 retiring now, 1 will move when their turn ends" in out
    assert _SEEN["own_pid"] is None, "a CLI caller has no session of its own to exclude"


def test_all_with_nothing_running_says_so(capsys) -> None:
    with patch.object(control, "_rotation_targets", lambda root, own_pid=None: []):
        assert refresh_command(_args(refresh_all=True)) == 0
    assert "no live sessions to refresh" in capsys.readouterr().out


def test_all_needs_no_confirmation_even_in_a_pipe(monkeypatch, capsys) -> None:
    """Unlike ``stop --all``, this ends nothing, so there is nothing to consent to.

    Asserted on the thing that would be wrong for a pipe: reading stdin. A
    prompt here would hang a script that merely wants to ship a new build.
    """

    def _no_stdin(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("lop refresh must never read stdin")

    monkeypatch.setattr("builtins.input", _no_stdin)
    with (
        patch.object(control, "_rotation_targets", lambda root, own_pid=None: [_Record()]),
        patch.object(control, "refresh_all", _fake_refresh_all),
    ):
        assert refresh_command(_args(refresh_all=True)) == 0
    assert capsys.readouterr().out


def test_json_shape(capsys) -> None:
    with (
        patch("local_operator.cli._resolve_stop_target", lambda _a: _single_target()),
        patch.object(control, "refresh_session", _fake_refresh),
    ):
        assert refresh_command(_args(target="the agent", json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {
            "pid": 4242,
            "session_id": "s1",
            "name": "the agent",
            "method": "moved",
            "line": 'moved "the agent"',
        }
    ]


def test_a_positive_timeout_is_passed_through(capsys) -> None:
    with (
        patch("local_operator.cli._resolve_stop_target", lambda _a: _single_target()),
        patch.object(control, "refresh_session", _fake_refresh),
    ):
        refresh_command(_args(target="the agent", timeout=2.5))
    assert _SEEN["timeout_s"] == 2.5
    capsys.readouterr()


def test_a_non_positive_timeout_falls_back_to_the_default(capsys) -> None:
    """``--timeout 0`` means "the default", exactly as ``lop stop`` reads it."""
    with (
        patch("local_operator.cli._resolve_stop_target", lambda _a: _single_target()),
        patch.object(control, "refresh_session", _fake_refresh),
    ):
        refresh_command(_args(target="the agent", timeout=0))
    assert _SEEN["timeout_s"] == control.DEFAULT_REFRESH_TIMEOUT_S
    capsys.readouterr()


def test_the_parser_exposes_the_command_and_its_target_flags() -> None:
    """The wire between argparse and the command, asserted once.

    A subcommand whose dest names drift from what the command reads fails only
    when a user runs it, which is the failure mode ``refresh_parser`` avoids
    with ``dest="refresh_all"`` (``all`` is a builtin and a reserved word in
    most surfaces).
    """
    from local_operator.cli import build_cli_parser

    parsed = build_cli_parser().parse_args(["refresh", "--all", "--json", "--timeout", "3"])
    assert parsed.subcommand == "refresh"
    assert parsed.refresh_all is True
    assert parsed.json is True
    assert parsed.timeout == 3.0
    assert parsed.target is None and parsed.pid is None and parsed.session is None
