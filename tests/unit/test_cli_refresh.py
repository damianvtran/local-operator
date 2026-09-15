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

import pytest

from local_operator.cli import refresh_command
from local_operator.session.runtime import control
from local_operator.session.runtime.control import RefreshOutcome
from local_operator.session.runtime.types import SessionRecord

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
        self.source_ref = ""


def _session_record(version: str = "0.54.48", source_ref: str = "") -> SessionRecord:
    """A real record, for the cells that call the builders with one.

    The CLI stub above is deliberately narrower (it stands in for the resolver's
    input), but ``_refresh_line``/``_build_label`` are typed against
    ``SessionRecord`` and read the record's own published stamps — so the cells
    that exercise them build the real thing rather than a lookalike that would
    drift from it.
    """
    return SessionRecord(
        pid=4242,
        kind="tui",
        session_id="s1",
        conversation_name="the agent",
        cwd="/tmp",
        model_label="test/model",
        control_port=1,
        control_key="k",
        version=version,
        source_ref=source_ref,
    )


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
    # A singular count takes a singular possessive: the shipped line read
    # "1 will move when their turn ends" (D3, PR #1141).
    assert "3 sessions: 1 retiring now, 1 will move when its turn ends" in out
    assert _SEEN["own_pid"] is None, "a CLI caller has no session of its own to exclude"


def test_a_draining_target_is_settled_and_says_the_bound(capsys) -> None:
    """A target already leaving a signal is NOT a failure, and it says why.

    Nothing is owed by the caller — the exit is already scheduled — so the exit
    code stays 0. What the receipt must carry is the fact itself and the
    EXISTENCE of a bound: a signalled runtime works on, looking ordinary, for up
    to ``SIGNAL_DRAIN_S``, and two minutes of unexplained waiting reads as a
    hang (U2, PR #1141). ``draining`` is in ``REFRESH_SETTLED_METHODS``, and
    this cell is what pins that it stayed there.
    """
    from local_operator.session.runtime.types import SIGNAL_DRAIN_S

    async def _draining(record, *, timeout_s):  # noqa: ANN001, ANN202
        return _outcome("draining", line='"the agent" was signalled — it is leaving')

    with (
        patch("local_operator.cli._resolve_stop_target", lambda _a: _single_target()),
        patch.object(control, "refresh_session", _draining),
    ):
        assert refresh_command(_args(target="the agent")) == 0
    out = capsys.readouterr().out
    assert "leaving" in out
    # The claim is about the METHOD, not the stub's own prose: the real line is
    # built by ``control._refresh_line``, which is where the bound is named.
    line = control._refresh_line(
        _session_record(),
        "0.54.48",
        "draining",
        "",
    )
    assert "was signalled" in line and "its next boundary" in line
    assert f"up to {SIGNAL_DRAIN_S / 60:.0f} min" in line


def test_an_unsettled_install_is_partial_not_clean(capsys) -> None:
    """The install has moved but nobody has judged it yet — NOT "already current".

    This is the D1/M2 defect at the exit-code level: the runtime used to fold
    "the stamp matches" and "the marker has not settled" into one answer, the
    CLI mapped it to ``current`` (a settled method), and `lop refresh` — whose
    own documentation says its first run is `lop-update` — therefore exited 0
    and printed "already runs the build on disk" about a whole fleet that was
    about to rotate. The unsettled answer must stay OUT of
    ``REFRESH_SETTLED_METHODS``, so a script cannot read the rotation as done.
    """

    async def _unsettled(record, *, timeout_s):  # noqa: ANN001, ANN202
        return _outcome("unsettled", line='"the agent" has not judged the build on disk yet')

    with (
        patch("local_operator.cli._resolve_stop_target", lambda _a: _single_target()),
        patch.object(control, "refresh_session", _unsettled),
    ):
        assert refresh_command(_args(target="the agent")) == 2
    out = capsys.readouterr().out
    assert "has not judged" in out
    assert "unsettled" not in control.REFRESH_SETTLED_METHODS
    line = control._refresh_line(_session_record(), "0.54.48", "unsettled", "")
    assert "ask again in a few seconds" in line


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


def test_the_summary_agrees_with_its_count(capsys) -> None:
    """Both counts, because the bug was the SINGULAR one reading as plural.

    ``1 will move when their turn ends`` shipped (D3, PR #1141): the count noun
    inflected and the label did not. The plural form is unchanged, so this cell
    pins both halves of the fix rather than the new one alone.
    """
    from local_operator.session.runtime.control import summarize_refresh

    one = summarize_refresh([_outcome("busy")])
    assert one == "1 session: 1 will move when its turn ends", one
    three = summarize_refresh([_outcome("busy", pid=p) for p in (1, 2, 3)])
    assert three == "3 sessions: 3 will move when their turn ends", three


@pytest.mark.parametrize(
    ("version", "ref", "expected"),
    [
        ("0.55.0", "aaaaaaa1111111", "0.55.0@aaaaaaa"),
        ("0.55.0", "", "0.55.0"),
        ("", "bbbbbbb2222222", "bbbbbbb"),
        ("", "", "an unrecorded build"),
    ],
)
def test_the_build_label_names_the_ref_that_makes_two_builds_different(
    version: str, ref: str, expected: str
) -> None:
    """``version@ref[:7]``, and the fallbacks (D2, PR #1141).

    The dominant handover on this host is a same-version rebuild — `lop-update`
    builds from `main` while `pyproject.toml` still names the last release — so
    a version-only label printed identically on the row LEAVING the old build
    and the row already on the new one, which is the one distinction the
    receipts exist to draw. The TUI's build-skew notice already prints this form.
    """
    from local_operator.session.runtime.control import _build_label

    assert _build_label(_session_record(version, ref)) == expected


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
