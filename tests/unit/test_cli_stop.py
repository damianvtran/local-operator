"""``lop stop`` — the CLI front end of the kill switch.

What is pinned here is the CLI's OWN contract, not the ladder (that is
``tests/unit/session/runtime/test_control.py``): the exit-code triple
(0 clean / 1 no match / 2 partial), the ``--all`` confirmation rules (a pipe
refuses without ``--yes``; ``--yes`` proceeds), the ``--json`` shape, and
that the resolver is the same one ``lop send`` uses. The ladder is stubbed at
``control.stop_session`` / ``control.stop_all`` so no socket is dialled and
no process is ever signalled from a test.

THE STUBS MIRROR THE REAL SIGNATURES, keyword for keyword (``timeout_s``,
``_root``, ``force``, and now ``_command``). A stub narrower than the function
it replaces is a test that passes until a caller uses a keyword for real — which
is exactly how this file failed when the CLI started naming ``_command`` (the
front end threaded into every stop marker): the CLI was right and the stub was
out of date, so the stub moved, not the call site. Each stub also records what it
was handed, so the CLI's own contract for the front-end name is asserted here
rather than inferred from the ladder's tests.
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
        "force": False,
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


#: What the stubs were handed, for the assertions that pin the CLI's own
#: contract (today: the front-end name it puts in every stop marker).
_SEEN: dict[str, Any] = {}


async def _fake_stop(  # noqa: ANN001, ANN202
    record, *, timeout_s, _root, force=False, _command=None, on_wait=None
):
    _SEEN["stop_command"] = _command
    _SEEN["on_wait"] = on_wait
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
    # The marker's whole value is naming WHO stopped the runtime, so the CLI
    # says what the user typed rather than the function it reached (MINOR-2).
    assert _SEEN["stop_command"] == "lop stop"
    # And the single stop leans on the same progress painter the sweep does, so
    # a wedged target's ~150 s silence is not the user's first sign of trouble
    # (U5).
    assert _SEEN["on_wait"].__name__ == "_stop_progress"


def test_refused_identity_exits_2(capsys) -> None:
    """A refusal is a PARTIAL result, not a no-match: the target existed and
    was not stopped, which a script must be able to tell from 'wrong name'."""

    async def refuse(  # noqa: ANN001, ANN202
        record, *, timeout_s, _root, force=False, _command=None, on_wait=None
    ):
        return _outcome("refused", line="refused to signal pid 4242 — identity mismatch")

    with (
        patch("local_operator.cli._resolve_stop_target", return_value=(_Record(), [], "")),
        patch("local_operator.session.runtime.control.stop_session", refuse),
    ):
        rc = stop_command(_args(target="the agent"))
    assert rc == 2


def test_a_skipped_busy_target_is_partial_not_clean(capsys) -> None:
    """A target left alone because a turn is in flight is a PARTIAL result.

    The user asked for these sessions to be stopped and one of them is still
    running, so a script must not read exit 0 and conclude the machine is
    quiet. Which is also why the skip gets its own method token (``busy``)
    rather than reusing the refusal's: the receipt says something different to
    a human ("stop it again once the turn ends") and the front end derives
    partial-vs-clean from the token, never from the prose.
    """

    async def busy(  # noqa: ANN001, ANN202
        record, *, timeout_s, _root, force=False, _command=None, on_wait=None
    ):
        return _outcome(
            "busy",
            line='skipped "the agent" (pid 4242) — a turn is in flight; '
            "stop it again once the turn ends, or --force to signal it now",
        )

    with (
        patch("local_operator.cli._resolve_stop_target", return_value=(_Record(), [], "")),
        patch("local_operator.session.runtime.control.stop_session", busy),
    ):
        rc = stop_command(_args(target="the agent"))
    assert rc == 2
    assert "turn is in flight" in capsys.readouterr().out


def test_the_all_summary_counts_a_skipped_target_separately(capsys) -> None:
    """The grouped receipt the ``--all`` path paints names the skip as its own group.

    A kill switch that reported a skipped session under "stopped" would be
    claiming work it did not do — the one thing the summary exists to prevent.
    """
    from local_operator.session.runtime.control import summarize

    outcomes = [
        _outcome("socket", pid=1, line='stopped "one"'),
        _outcome("busy", pid=2, line='skipped "two" (pid 2) — a turn is in flight'),
        _outcome("refused", pid=3, line='refused "three" (pid 3) — identity'),
    ]
    line = summarize(outcomes)
    assert "stopped" in line
    assert "1 left alone (a turn is in flight)" in line
    assert "1 refused" in line
    assert "3 sessions" in line
    capsys.readouterr()


def test_already_exited_is_clean(capsys) -> None:
    """The dead-pid resolution is its own method (``gone``): nothing is
    left for a human to do, so it exits 0 — decided from the method, never
    from the receipt text (R1-7)."""

    async def gone(  # noqa: ANN001, ANN202
        record, *, timeout_s, _root, force=False, _command=None, on_wait=None
    ):
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
        # One live target: an empty machine is a clean no-op in every mode
        # (D2-3), so the refusal is only reachable when there IS something
        # the confirmation would be about.
        patch(
            "local_operator.session.runtime.control._stop_targets",
            return_value=[_Record(pid=4242)],
        ),
    ):
        rc = stop_command(_args(stop_all=True))
    assert rc == 1
    assert "--yes" in red.call_args[0][0]
    stop_all.assert_not_called()


def test_all_with_yes_runs_and_reports_partial(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    seen: dict[str, Any] = {}

    async def fake_all(  # noqa: ANN001, ANN202
        *, own_pid, _root, only_pids=None, timeout_s=10.0, force=False, _command=None, on_wait=None
    ):  # noqa: ANN001, ANN202
        seen.update(
            own_pid=own_pid,
            only_pids=only_pids,
            timeout_s=timeout_s,
            command=_command,
            on_wait=on_wait,
        )
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
    # --timeout reaches the ladder (R1-5); the run is scoped to the scan, and a
    # sweep names itself rather than a single stop (MINOR-2). ``on_wait`` is the
    # CLI's progress painter and must reach the sweep too — a wedged runtime in
    # front of a dozen healthy ones is where its silence costs the most (U5).
    assert seen["on_wait"] is not None, "the sweep forwards the progress painter (U5)"
    assert seen["on_wait"].__name__ == "_stop_progress"
    seen.pop("on_wait")
    assert seen == {
        "own_pid": None,
        "only_pids": {1, 2},
        "timeout_s": 4.0,
        "command": "lop stop --all",
    }


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
