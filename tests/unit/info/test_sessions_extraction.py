"""``lop sessions`` is behaviour-preserved by the extraction into ``info``.

``cli.sessions_command`` used to build its rows inline; ``/info`` needs the same
answer plus ``is_self`` and roll-up counters, so the builder moved to
``info.collect.session_rows`` and the CLI calls it. ``--json`` is a published
surface, so this file compares against the PRE-CHANGE literal rather than
against whatever the code now produces — a test that asserted "the output equals
the output" would pass through any drift at all.

The expectation below was captured by running the ORIGINAL ``sessions_command``
at ``4311eb653`` against this exact fixture with the clock frozen. A byte-level
before/after of the real command is on the PR; this is the in-suite guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from local_operator.info.collect import session_rows
from local_operator.info.model import SessionLine

#: Frozen clock, so every derived duration is a constant rather than a function
#: of when the suite ran.
NOW = 1_788_602_400.0


@dataclass
class _Record:
    pid: int
    kind: str
    session_id: str
    conversation_name: str
    cwd: str
    model_label: str
    started_at: float
    heartbeat_at: float
    pending: str | None = None
    busy: bool = False
    detached: bool = False
    subagents_running: int | None = None
    subagents_queued: int | None = None
    version: str = ""
    source_ref: str = ""


@dataclass
class _OldRecord:
    """A record written by a runtime that predates the newer fields."""

    pid: int
    kind: str
    session_id: str
    conversation_name: str
    cwd: str
    model_label: str
    started_at: float
    heartbeat_at: float


@dataclass
class _Usage:
    rss_bytes: int | None = None
    footprint_bytes: int | None = None


FIXTURE: list[tuple[Any, str]] = [
    (
        _Record(
            pid=4243,
            kind="tui",
            session_id="a3f9c21b7e40",
            conversation_name="Investigate request latency",
            cwd="/tmp/probe/workspace",
            model_label="anthropic/claude-sonnet-4-6",
            started_at=NOW - 1840.0,
            heartbeat_at=NOW - 3.0,
            pending="approval",
            busy=True,
            version="0.51.6",
            source_ref="4311eb653aa9",
            subagents_running=2,
            subagents_queued=1,
        ),
        "live",
    ),
    (
        _Record(
            pid=4244,
            kind="daemon",
            session_id="beef1234cafe",
            conversation_name="Mobile relay",
            cwd="/tmp/probe/relay",
            model_label="openai/gpt-5.2",
            started_at=NOW - 86400.0,
            heartbeat_at=NOW - 9.0,
            detached=True,
            version="0.51.5",
        ),
        "live",
    ),
    (
        _OldRecord(
            pid=999999,
            kind="exec",
            session_id="0badc0de0bad",
            conversation_name="Dead runtime",
            cwd="/tmp/probe/dead",
            model_label="anthropic/claude-opus-4-1",
            started_at=NOW - 600.0,
            heartbeat_at=NOW - 600.0,
        ),
        "stale",
    ),
]

#: EXACTLY what ``sessions_command`` produced before the extraction: same keys,
#: same order, same values — PLUS ``subagents_running``/``subagents_queued``,
#: added deliberately rather than discovered as a red test. ``--json`` is a
#: published surface and a fleet consumer counting agent trajectories across a
#: host wants them; the pin exists to make an addition a DECISION, which this
#: is, not to forbid one. Everything above them is byte-for-byte unchanged.
EXPECTED = [
    {
        "state": "live",
        "pid": 4243,
        "kind": "tui",
        "conversation_name": "Investigate request latency",
        "session_id": "a3f9c21b7e40",
        "model_label": "anthropic/claude-sonnet-4-6",
        "cwd": "/tmp/probe/workspace",
        "rss_bytes": 190_004_243,
        "footprint_bytes": None,
        "uptime_s": 1840.0,
        "heartbeat_age_s": 3.0,
        "pending": "approval",
        "busy": True,
        "detached": False,
        "version": "0.51.6",
        "source_ref": "4311eb653aa9",
        "subagents_running": 2,
        "subagents_queued": 1,
        # The stored row's clock (transcript activity); None on a live row,
        # which reports uptime_s/heartbeat_age_s instead. Present on every row
        # so the published shape is stable — a consumer never branches on key
        # existence.
        "last_activity_s": None,
        # THE LAST OUTCOME, from the attention store: the kind and the reason
        # the harness recorded. Empty here because this fixture names no root,
        # and the read is deliberately root-scoped (see
        # ``collect._with_stored_outcomes``) rather than falling back to
        # whatever store the machine happens to have.
        "completion_kind": "",
        "completion_reason": "",
    },
    {
        "state": "live",
        "pid": 4244,
        "kind": "daemon",
        "conversation_name": "Mobile relay",
        "session_id": "beef1234cafe",
        "model_label": "openai/gpt-5.2",
        "cwd": "/tmp/probe/relay",
        "rss_bytes": 190_004_244,
        "footprint_bytes": None,
        "uptime_s": 86400.0,
        "heartbeat_age_s": 9.0,
        "pending": None,
        "busy": False,
        "detached": True,
        "version": "0.51.5",
        "source_ref": "",
        "subagents_running": None,
        "subagents_queued": None,
        "last_activity_s": None,
        # THE LAST OUTCOME, from the attention store: the kind and the reason
        # the harness recorded. Empty here because this fixture names no root,
        # and the read is deliberately root-scoped (see
        # ``collect._with_stored_outcomes``) rather than falling back to
        # whatever store the machine happens to have.
        "completion_kind": "",
        "completion_reason": "",
    },
    {
        "state": "stale",
        "pid": 999999,
        "kind": "exec",
        "conversation_name": "Dead runtime",
        "session_id": "0badc0de0bad",
        "model_label": "anthropic/claude-opus-4-1",
        "cwd": "/tmp/probe/dead",
        "rss_bytes": None,
        "footprint_bytes": None,
        "uptime_s": 600.0,
        "heartbeat_age_s": 600.0,
        "pending": None,
        "busy": False,
        "detached": False,
        "version": "",
        "source_ref": "",
        # ``None``, not 0: ``_OldRecord`` has no such attribute at all, which
        # is exactly a runtime predating the fields. The distinction is the
        # whole point of the pair — see ``SessionsInfo.subagents_unreported``.
        "subagents_running": None,
        "subagents_queued": None,
        "last_activity_s": None,
        # THE LAST OUTCOME, from the attention store: the kind and the reason
        # the harness recorded. Empty here because this fixture names no root,
        # and the read is deliberately root-scoped (see
        # ``collect._with_stored_outcomes``) rather than falling back to
        # whatever store the machine happens to have.
        "completion_kind": "",
        "completion_reason": "",
    },
]


def _install_fixture(monkeypatch: Any) -> None:
    from local_operator.info import collect as collect_mod
    from local_operator.mobile import resources
    from local_operator.session.runtime import registry

    monkeypatch.setattr(registry, "scan", lambda root=None: FIXTURE)
    monkeypatch.setattr(
        resources,
        "session_resource_usage",
        lambda pids, **kwargs: {pid: _Usage(rss_bytes=190_000_000 + pid) for pid in pids},
    )
    monkeypatch.setattr(collect_mod.time, "time", lambda: NOW)


def test_session_rows_match_the_pre_extraction_output(monkeypatch: Any) -> None:
    _install_fixture(monkeypatch)
    assert session_rows() == EXPECTED


def test_session_rows_key_order_is_the_published_contract(monkeypatch: Any) -> None:
    """``--json`` consumers read this shape; the ORDER is part of it.

    Pinned explicitly rather than derived from ``dataclasses.asdict``, which
    would also have leaked ``is_self`` — a field the CLI never had — into a
    published surface.
    """
    _install_fixture(monkeypatch)
    for row, expected in zip(session_rows(), EXPECTED):
        assert list(row) == list(expected)


def test_is_self_is_not_published_by_the_cli_shape(monkeypatch: Any) -> None:
    _install_fixture(monkeypatch)
    assert all("is_self" not in row for row in session_rows())
    # It exists on the dataclass /info reads, which is the point of extracting
    # rather than copying.
    assert "is_self" in SessionLine.__dataclass_fields__


def test_cli_sessions_command_uses_the_shared_builder(monkeypatch: Any, capsys: Any) -> None:
    """Drive the REAL command, not the helper, so the wiring itself is pinned."""
    import argparse

    from local_operator import cli

    _install_fixture(monkeypatch)
    code = cli.sessions_command(
        argparse.Namespace(json=True, sessions_command=None, all=False, limit=None)
    )
    assert code == 0

    import json

    assert json.loads(capsys.readouterr().out) == EXPECTED


def test_cli_table_still_renders_every_row(monkeypatch: Any, capsys: Any) -> None:
    """The non-JSON path reads the same objects and keeps its eight columns."""
    import argparse

    from local_operator import cli

    _install_fixture(monkeypatch)
    code = cli.sessions_command(
        argparse.Namespace(json=False, sessions_command=None, all=False, limit=None)
    )
    assert code == 0

    out = capsys.readouterr().out
    assert "STATE" in out and "FOOTPRINT" in out and "HB_AGE" in out
    # The CONVERSATION column is 24 cells and the table has always truncated to
    # it; asserting the full name here would assert a change, not a preservation.
    assert "Investigate request late" in out
    assert "Mobile relay" in out
    assert "Dead runtime" in out
    assert out.count("\n") == 4  # header + one row per fixture record
    # And the key never reaches a terminal either.
    assert "control_key" not in out


def test_sessions_all_without_limit_passes_the_advertised_default(
    monkeypatch: Any, tmp_path: Any, capsys: Any
) -> None:
    """QA Q1/MAJOR-2 at the CLI wiring: ``None`` must reach collect as 50.

    The default cap is applied inside ``_stored_lines``; this pins that the
    CLI's argparse default (``None``) is what reaches it and becomes
    ``STORED_SESSIONS_DEFAULT_LIMIT`` there — the exact wiring whose absence
    made round 1 list the entire store. The scan is recorded rather than
    faked wholesale so the argument itself is the assertion.
    """
    import argparse

    from local_operator import cli
    from local_operator.info import collect as collect_mod

    _install_fixture(monkeypatch)
    seen: list[Any] = []

    class _Row:
        id = "dead00000001"
        name = "old work"
        mtime = 5.0

    def _record(directory: Any, limit: Any = None) -> list[Any]:
        seen.append(limit)
        return [_Row()]

    monkeypatch.setattr("local_operator.resume.recent_session_rows", _record)
    monkeypatch.setattr(cli, "config_dir", lambda: tmp_path)
    code = cli.sessions_command(
        argparse.Namespace(json=True, sessions_command=None, all=True, limit=None)
    )
    assert code == 0
    capsys.readouterr()
    assert seen == [collect_mod.STORED_SESSIONS_DEFAULT_LIMIT]


def test_sessions_limit_rejects_zero_and_negative(capsys: Any) -> None:
    """QA Q2: ``--limit 0`` parsed, listed nothing, and read as "no sessions".

    Zero and negative caps are typos, not requests — "no stored rows" is what
    plain `lop sessions` already means — so argparse refuses them up front
    rather than printing an empty listing that lies about the store.
    """
    import pytest as _pytest

    from local_operator import cli

    for bad in ("0", "-1"):
        with _pytest.raises(SystemExit) as raised:
            cli.build_cli_parser().parse_args(["sessions", "--all", "--limit", bad])
        assert raised.value.code == 2
        assert "positive integer" in capsys.readouterr().err


def test_sessions_empty_copy_names_the_search_that_was_asked_for(
    monkeypatch: Any, tmp_path: Any, capsys: Any
) -> None:
    """MINOR-5: `--all` on an empty store must not answer "no ACTIVE sessions".

    The live-only line re-teaches the old vocabulary on the very flag that
    opts into the store; the default listing keeps its established copy.
    """
    import argparse

    from local_operator import cli

    monkeypatch.setattr(cli, "config_dir", lambda: tmp_path)
    from local_operator.session.runtime import registry

    monkeypatch.setattr(registry, "scan", lambda root=None: [])

    code = cli.sessions_command(
        argparse.Namespace(json=False, sessions_command=None, all=True, limit=None)
    )
    assert code == 0
    assert "live or stored" in capsys.readouterr().out

    code = cli.sessions_command(
        argparse.Namespace(json=False, sessions_command=None, all=False, limit=None)
    )
    assert code == 0
    assert capsys.readouterr().out.strip() == "no active lop sessions"


def _seed_outcome(root: Any, session_id: str, *, kind: str, reason: str, cause: str) -> None:
    """One real attention-store row, written by the product's own publisher.

    Through ``AttentionStore.publish`` rather than by hand, so the fixture is
    the shape the runtime writes (identity, token, anchor and all) and cannot
    drift from the reader it is here to exercise.
    """
    import uuid

    from local_operator.session.attention import AttentionStore

    token = str(uuid.uuid4())
    AttentionStore(root / "attention.db").publish(
        f"session/{session_id}", token, f"completion-{token}", kind, reason=reason, cause=cause
    )


def _deliberate_stop_reason() -> str:
    """The sentence a rung-3 deliberate stop's outcome carries, from the code."""
    from local_operator.incidents import (
        DELIBERATE_CUT_OFF_CAUSE,
        render_cut_off_reason,
        render_stop_attribution,
    )

    return render_cut_off_reason(
        DELIBERATE_CUT_OFF_CAUSE,
        detail=render_stop_attribution(rung="sigkill", command="/stop --all", killer_pid=40609),
    )


def test_a_stored_outcome_reaches_the_rows(monkeypatch: Any, tmp_path: Any) -> None:
    """Design round 1, D1: `lop sessions` must be able to answer "why did this die".

    The reason is the one fact that OUTLIVES a killed runtime — a SIGKILLed
    process publishes nothing, its record is reaped, and the attention store is
    all that is left — so the CLI's rows carry it under ``completion_kind`` /
    ``completion_reason``, and a session with no recorded outcome carries empty
    strings rather than missing keys.
    """
    _install_fixture(monkeypatch)
    _seed_outcome(
        tmp_path,
        "a3f9c21b7e40",
        kind="interrupted",
        reason=_deliberate_stop_reason(),
        cause="user-stop",
    )

    rows = {row["session_id"]: row for row in session_rows(tmp_path)}
    stopped = rows["a3f9c21b7e40"]
    assert stopped["completion_kind"] == "interrupted"
    assert "killed by /stop --all" in stopped["completion_reason"]
    assert "killer pid 40609" in stopped["completion_reason"]
    quiet = rows["beef1234cafe"]
    assert (quiet["completion_kind"], quiet["completion_reason"]) == ("", "")


def test_the_table_explains_a_session_only_when_it_has_something_to_explain(
    monkeypatch: Any, tmp_path: Any, capsys: Any
) -> None:
    """The WHY column, and the reason it is CONDITIONAL.

    Appended only when at least one row has an outcome, following the
    LAST_ACTIVE column's precedent: a healthy listing has to parse exactly as it
    did before, or every consumer of `lop sessions` pays a re-flow for a column
    of blanks. The cell is the attributed phrase — the rung and the actor, which
    is what a reader cannot get anywhere else on this surface.
    """
    import argparse
    import json as _json

    from local_operator import cli

    _install_fixture(monkeypatch)
    monkeypatch.setattr(cli, "config_dir", lambda: tmp_path)

    code = cli.sessions_command(
        argparse.Namespace(json=False, sessions_command=None, all=False, limit=None)
    )
    assert code == 0
    healthy = capsys.readouterr().out
    assert "WHY" not in healthy, healthy

    _seed_outcome(
        tmp_path,
        "a3f9c21b7e40",
        kind="interrupted",
        reason=_deliberate_stop_reason(),
        cause="user-stop",
    )
    code = cli.sessions_command(
        argparse.Namespace(json=False, sessions_command=None, all=False, limit=None)
    )
    assert code == 0
    explained = capsys.readouterr().out
    assert "WHY" in explained.splitlines()[0], explained
    # The table names a row by its conversation, not by its id (that is the
    # CONVERSATION column's own truncation), so the row is found by the name.
    stopped_line = next(line for line in explained.splitlines() if "Investigate" in line)
    assert "killed by /stop --all" in stopped_line, stopped_line

    # `--json` carries the whole stored sentence, not the column's slice.
    code = cli.sessions_command(
        argparse.Namespace(json=True, sessions_command=None, all=False, limit=None)
    )
    assert code == 0
    payload = _json.loads(capsys.readouterr().out)
    row = next(item for item in payload if item["session_id"] == "a3f9c21b7e40")
    assert "killer pid 40609" in row["completion_reason"]
