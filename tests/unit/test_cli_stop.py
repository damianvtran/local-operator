"""``lop stop`` — the CLI front end of the kill switch.

What is pinned here is the CLI's OWN contract, not the ladder (that is
``tests/unit/session/runtime/test_control.py``): the exit-code triple
(0 clean / 1 no match / 2 partial), the ``--all`` confirmation rules (a pipe
refuses without ``--yes``; ``--yes`` proceeds), the ``--json`` shape, and
that the resolver is the same one ``lop send`` uses. The ladder is stubbed at
``control.stop_session`` / ``control.stop_all`` so no socket is dialled and
no process is ever signalled from a test.
"""

from __future__ import annotations

import argparse
import io
import json
from typing import Any
from unittest.mock import patch

import pytest

from local_operator.cli import stop_command
from local_operator.session.runtime.control import StopOutcome


def _args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "target": None,
        "pid": None,
        "session": None,
        "stop_all": False,
        "yes": False,
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


def _outcome(method: str, pid: int = 4242, line: str | None = None) -> StopOutcome:
    return StopOutcome(
        pid=pid,
        session_id="s1",
        name="the agent",
        method=method,
        line=line or 'stopped "the agent"',
    )


async def _fake_stop(record, *, timeout_s, _root):  # noqa: ANN001, ANN202
    return _outcome("socket", pid=record.pid)


def test_no_match_exits_1(capsys) -> None:
    with (
        patch(
            "local_operator.cli._resolve_stop_target",
            return_value=(None, [], "no live session matches 'x'"),
        ),
        patch("local_operator.cli._peer_red") as red,
    ):
        rc = stop_command(_args(target="x"))
    assert rc == 1
    assert "no live session matches" in red.call_args[0][0]


def test_ambiguous_target_lists_candidates_and_exits_1(capsys) -> None:
    with patch(
        "local_operator.cli._resolve_stop_target",
        return_value=(None, [_Record(1), _Record(2)], ""),
    ):
        rc = stop_command(_args(target="the"))
    assert rc == 1
    err = capsys.readouterr().err
    assert "2 sessions match" in err
    assert "--pid 1" in err and "--pid 2" in err


def test_one_target_stopped_exits_0_and_prints_the_receipt(capsys) -> None:
    with (
        patch("local_operator.cli._resolve_stop_target", return_value=(_Record(), [], "")),
        patch("local_operator.session.runtime.control.stop_session", _fake_stop),
    ):
        rc = stop_command(_args(target="the agent"))
    assert rc == 0
    assert 'stopped "the agent"' in capsys.readouterr().out


def test_refused_identity_exits_2(capsys) -> None:
    """A refusal is a PARTIAL result, not a no-match: the target existed and
    was not stopped, which a script must be able to tell from 'wrong name'."""

    async def refuse(record, *, timeout_s, _root):  # noqa: ANN001, ANN202
        return _outcome("refused", line="refused to signal pid 4242 — identity mismatch")

    with (
        patch("local_operator.cli._resolve_stop_target", return_value=(_Record(), [], "")),
        patch("local_operator.session.runtime.control.stop_session", refuse),
    ):
        rc = stop_command(_args(target="the agent"))
    assert rc == 2


def test_already_exited_is_clean(capsys) -> None:
    """The dead-pid resolution is its own method (``gone``): nothing is
    left for a human to do, so it exits 0 — decided from the method, never
    from the receipt text (R1-7)."""

    async def gone(record, *, timeout_s, _root):  # noqa: ANN001, ANN202
        return _outcome("gone", line='"the agent" already exited')

    with (
        patch("local_operator.cli._resolve_stop_target", return_value=(_Record(), [], "")),
        patch("local_operator.session.runtime.control.stop_session", gone),
    ):
        rc = stop_command(_args(target="the agent"))
    assert rc == 0


def test_json_shape(capsys) -> None:
    with (
        patch("local_operator.cli._resolve_stop_target", return_value=(_Record(), [], "")),
        patch("local_operator.session.runtime.control.stop_session", _fake_stop),
    ):
        rc = stop_command(_args(target="the agent", json=True))
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows == [
        {
            "pid": 4242,
            "session_id": "s1",
            "name": "the agent",
            "method": "socket",
            "line": 'stopped "the agent"',
            "wakes_dormant": 0,
        }
    ]


def test_all_in_a_pipe_refuses_without_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pipe has no one to answer y/N, so --all must not proceed on its own —
    and must not hang on stdin either."""
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    with (
        patch("local_operator.cli._peer_red") as red,
        patch("local_operator.session.runtime.control.stop_all") as stop_all,
    ):
        rc = stop_command(_args(stop_all=True))
    assert rc == 1
    assert "--yes" in red.call_args[0][0]
    stop_all.assert_not_called()


def test_all_with_yes_runs_and_reports_partial(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    seen: dict[str, Any] = {}

    async def fake_all(*, own_pid, _root, only_pids=None, timeout_s=10.0):  # noqa: ANN001, ANN202
        seen.update(own_pid=own_pid, only_pids=only_pids, timeout_s=timeout_s)
        return [
            _outcome("socket", pid=1),
            _outcome("refused", pid=2, line='refused "the agent" (pid 2) — did not answer'),
        ]

    with (
        patch("local_operator.session.runtime.control.stop_all", fake_all),
        patch(
            "local_operator.session.runtime.control._stop_targets",
            return_value=[_Record(1), _Record(2)],
        ),
    ):
        rc = stop_command(_args(stop_all=True, yes=True, timeout=4.0))
    assert rc == 2
    out = capsys.readouterr().out
    assert "2 sessions: 1 stopped, 1 refused" in out
    # --timeout reaches the ladder (R1-5); the run is scoped to the scan.
    assert seen == {"own_pid": None, "only_pids": {1, 2}, "timeout_s": 4.0}


def test_all_with_nothing_running_says_so(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """An empty ``--all --yes`` still prints a line (D5)."""
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    async def fake_all(**kwargs):  # noqa: ANN003, ANN202
        return []

    with (
        patch("local_operator.session.runtime.control.stop_all", fake_all),
        patch("local_operator.session.runtime.control._stop_targets", return_value=[]),
    ):
        rc = stop_command(_args(stop_all=True, yes=True))
    assert rc == 0
    assert capsys.readouterr().out.strip() == "no sessions to stop"


def test_all_on_a_tty_prompts_and_n_aborts(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("sys.stdin", Tty("n\n"))
    prompts: list[str] = []

    def fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        return "n"

    monkeypatch.setattr("builtins.input", fake_input)
    with (
        patch("local_operator.session.runtime.control.stop_all") as stop_all,
        patch(
            "local_operator.session.runtime.control._stop_targets",
            return_value=[_Record(1), _Record(2)],
        ),
    ):
        rc = stop_command(_args(stop_all=True))
    assert rc == 1
    stop_all.assert_not_called()
    out = capsys.readouterr().out
    # The listing precedes the prompt, and the prompt carries the count (U7/D6).
    assert "will stop 2 sessions:" in out and "pid 1  the agent" in out
    assert prompts == ["stop all 2 lop sessions on this machine? [y/N] "]
    assert "aborted" in out


def test_resolver_is_the_send_resolver() -> None:
    """One target vocabulary: `lop stop` resolves through the same function
    `lop send` does, with the stop parser's own flag names as hints."""
    from local_operator.cli import _resolve_stop_target

    with patch("local_operator.mobile.peer_send.resolve_peer_target") as resolve:
        resolve.return_value = (None, [], "x")
        _resolve_stop_target(_args(target="foo", pid=7, session="s"))
    resolve.assert_called_once_with(
        target="foo",
        pid=7,
        session="s",
        pid_hint="--pid",
        session_hint="--session",
        include_wedged=True,
    )
