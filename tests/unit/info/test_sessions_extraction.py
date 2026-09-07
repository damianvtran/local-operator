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
#: same order, same values.
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
    code = cli.sessions_command(argparse.Namespace(json=True, sessions_command=None))
    assert code == 0

    import json

    assert json.loads(capsys.readouterr().out) == EXPECTED


def test_cli_table_still_renders_every_row(monkeypatch: Any, capsys: Any) -> None:
    """The non-JSON path reads the same objects and keeps its eight columns."""
    import argparse

    from local_operator import cli

    _install_fixture(monkeypatch)
    code = cli.sessions_command(argparse.Namespace(json=False, sessions_command=None))
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
