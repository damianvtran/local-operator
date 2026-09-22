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

import re
from dataclasses import dataclass, replace
from typing import Any

import pytest

from local_operator import buildwatch
from local_operator.info.collect import session_rows
from local_operator.info.model import SessionLine
from local_operator.session.runtime.types import LEAVING_ON_SIGNAL
from local_operator.update import BuildStamp

#: The window pair the fixture publishes, built the way a runtime builds one
#: (``BuildStamp.label()``), so the cell below pins the string a real record carries.
UPDATE_PAIR = buildwatch.update_pair_text(
    BuildStamp(version="0.59.9", source_ref=""),
    BuildStamp(version="0.59.11", source_ref="ead71b673a9a"),
)

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
    #: The drain marker (``SessionRecord.leaving``). Absent from ``_OldRecord``
    #: below on purpose: an older runtime's record has no such field, and the
    #: listing must render it as "not leaving" rather than raising.
    leaving: str = ""
    #: The update window's pair (``SessionRecord.updating``), same additive contract
    #: as the field above.
    updating: str = ""
    #: The FAILED half of the same window (``SessionRecord.update_failed``). Absent
    #: from ``_OldRecord`` below for the same reason ``updating`` is: a record
    #: written by an older runtime has no such field, and listing must render the
    #: row as an ordinary one rather than raising (design review round 1, D1).
    update_failed: str = ""


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
        # WHETHER THIS RUNTIME IS FINISHING A TURN BEFORE LEAVING. Appended at
        # the end of the published key order, like the two above it: the field
        # is what makes a drain visible to a fleet reader at all (U1/U2, PR
        # #1141), and ``lop sessions`` prints it in its own LEAVING column.
        # Empty in this fixture because CONTAMINATING it would change every
        # BYTE of every row for every other cell in this file — the drain has
        # its own cells below.
        "leaving": "",
        "updating": "",
        "update_failed": "",
        # WHAT THE LAST BEAT MEASURED (the 2026-09-20 freeze): no-progress seconds
        # and this process's own CPU time over the same gap. Appended at the END
        # like every extension above it, and ``None`` on rows whose runtime does
        # not report them — the same unreported-vs-zero distinction
        # ``subagents_queued`` makes. Both are absent from the fake record below,
        # which is what a runtime predating the fields looks like.
        "beat_lag_s": None,
        "cpu_since_beat_s": None,
        # WHERE THIS SESSION'S STALL DUMP IS when its own bound fired, appended
        # after the pair above for the same append-only reason: ``None`` here
        # because no runtime in this fixture tripped it (the path is
        # ``stall_watchdog``'s to compose, and it is the one artifact a reader
        # needs after a freeze — see that module). ``stall_held`` is the third
        # state's reader and a bool on every row, so a fixture with no fire at all
        # is ``False`` rather than null.
        "stall_dump": None,
        "stall_held": False,
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
        # NOT LEAVING — present on every row so the published shape is stable.
        "leaving": "",
        "updating": "",
        "update_failed": "",
        # WHAT THE LAST BEAT MEASURED (the 2026-09-20 freeze): no-progress seconds
        # and this process's own CPU time over the same gap. Appended at the END
        # like every extension above it, and ``None`` on rows whose runtime does
        # not report them — the same unreported-vs-zero distinction
        # ``subagents_queued`` makes. Both are absent from the fake record below,
        # which is what a runtime predating the fields looks like.
        "beat_lag_s": None,
        "cpu_since_beat_s": None,
        # WHERE THIS SESSION'S STALL DUMP IS when its own bound fired, appended
        # after the pair above for the same append-only reason: ``None`` here
        # because no runtime in this fixture tripped it (the path is
        # ``stall_watchdog``'s to compose, and it is the one artifact a reader
        # needs after a freeze — see that module). ``stall_held`` is the third
        # state's reader and a bool on every row, so a fixture with no fire at all
        # is ``False`` rather than null.
        "stall_dump": None,
        "stall_held": False,
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
        # ``_OldRecord`` predates the field entirely, so this is the getattr
        # default: a record written by an older runtime lists as "not leaving"
        # rather than raising.
        "leaving": "",
        "updating": "",
        "update_failed": "",
        # WHAT THE LAST BEAT MEASURED (the 2026-09-20 freeze): no-progress seconds
        # and this process's own CPU time over the same gap. Appended at the END
        # like every extension above it, and ``None`` on rows whose runtime does
        # not report them — the same unreported-vs-zero distinction
        # ``subagents_queued`` makes. Both are absent from the fake record below,
        # which is what a runtime predating the fields looks like.
        "beat_lag_s": None,
        "cpu_since_beat_s": None,
        # WHERE THIS SESSION'S STALL DUMP IS when its own bound fired, appended
        # after the pair above for the same append-only reason: ``None`` here
        # because no runtime in this fixture tripped it (the path is
        # ``stall_watchdog``'s to compose, and it is the one artifact a reader
        # needs after a freeze — see that module). ``stall_held`` is the third
        # state's reader and a bool on every row, so a fixture with no fire at all
        # is ``False`` rather than null.
        "stall_dump": None,
        "stall_held": False,
    },
]


#: ``FIXTURE`` with one session inside its drain — the shape U2 is about.
#:
#: A separate set rather than a field on the shared one, because a non-empty
#: value there re-flows EVERY row's tail and so every other cell in this file
#: (the wide-glyph and WHY-clamp cells read the end of the row). The drain has
#: its own rows, and the rest of the file keeps its assertions.
DRAINING = [
    (
        replace(record, leaving=LEAVING_ON_SIGNAL) if index == 0 else record,
        state,
    )
    for index, (record, state) in enumerate(FIXTURE)
]


#: ``FIXTURE`` with one session MID-UPDATE — the idle handover's window, the state
#: the 2026-09-19 incident is about (``types.UPDATING``).
#:
#: A separate set for the reason ``DRAINING`` above is one: a non-empty value
#: re-flows that row's tail and so every cell that reads the end of the row. And
#: the two windows must not be confusable — a window is a runtime that HAS the
#: message and is coming back, a drain is one that refuses it — so this fixture
#: deliberately carries no ``leaving``, and the cell below asserts both halves.
UPDATING_WINDOW = [
    (replace(record, updating=UPDATE_PAIR) if index == 0 else record, state)
    for index, (record, state) in enumerate(FIXTURE)
]


#: The three SEQUENCE classes the clamp has to measure as units rather than as
#: characters: a VS16 selection (``\u2764\ufe0f`` is one 2-cell glyph from two code
#: points), a keycap sequence (``1\ufe0f\u20e3``), and a ZWJ family cluster (five
#: code points, one 2-cell glyph).
VS16 = "\u2764\ufe0f"
KEYCAP = "1\ufe0f\u20e3"
FAMILY = "\U0001f468\u200d\U0001f469\u200d\U0001f467"


#: Rows whose text columns and WHY reason carry those sequences.
#:
#: Deliberately NOT folded into ``FIXTURE``: that set is the ASCII equivalence
#: witness's input, and "the output did not move" is only a meaningful claim over
#: ASCII — the cell rule is SUPPOSED to differ from the character rule here. They
#: are a separate set because the round-2 review found the per-character measure
#: by running exactly these values, and the guard could not see it: no fixture row
#: carried a sequence, so the failing case lived only in the reviewer's hands
#: (review round 2, M1). A class of input with no fixture row is a class the suite
#: cannot defend.
GLYPH_FIXTURE: list[tuple[Any, str]] = [
    (
        _Record(
            pid=5150,
            kind="tui",
            session_id="facedeadbeef",
            conversation_name=VS16 * 20,  # 40 cells into a 24-cell column
            cwd="/tmp/probe/glyphs",
            model_label=KEYCAP * 14,  # 28 cells into a 24-cell column
            started_at=NOW - 1800.0,
            heartbeat_at=NOW - 3.6,
            pending=VS16 * 5,  # 10 cells into an 8-cell column
        ),
        "live",
    ),
    (
        _Record(
            pid=5151,
            kind="exec",
            session_id="facedeadcafe",
            conversation_name=FAMILY * 13,  # 26 cells into a 24-cell column
            cwd="/tmp/probe/family",
            model_label=FAMILY * 15,  # 30 cells into a 24-cell column
            started_at=NOW - 60.0,
            heartbeat_at=NOW - 1.0,
            pending=FAMILY * 3,  # 6 cells, inside the 8-cell column
        ),
        "live",
    ),
]


def _install_fixture(monkeypatch: Any, rows: Any = None) -> None:
    from local_operator.info import collect as collect_mod
    from local_operator.mobile import resources
    from local_operator.session.runtime import registry

    monkeypatch.setattr(registry, "scan", lambda root=None: FIXTURE if rows is None else rows)
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
    # A fleet where nobody is leaving renders exactly as it always did — no new
    # column, no re-flow — which is what the drain's own cell below asserts from
    # the other side.
    assert "LEAVING" not in out
    # And the key never reaches a terminal either.
    assert "control_key" not in out


def test_the_sessions_table_leads_with_words_while_json_keeps_the_token(
    monkeypatch: Any, capsys: Any
) -> None:
    """D5/QA Q1: the STATE column is read by a person; ``--json`` is not.

    ``lop wake status`` ends its wedge line with "'lop sessions' shows it", so
    this table is where an operator is sent — and it was the one person-facing
    surface where the raw state token stood with no sentence to qualify it,
    beside an ``HB_AGE`` that measures the same fact. The token itself stays
    exactly where a machine reads it: ``--json``'s ``state`` is the wire value
    the ~15 call sites and the desktop catalogue's ``status.code`` branch on.
    """
    import argparse
    import json

    from local_operator import cli
    from local_operator.info import collect as collect_mod
    from local_operator.mobile import resources
    from local_operator.session.runtime import registry

    quiet = _Record(
        pid=6001,
        kind="daemon",
        session_id="cafebabecafe",
        conversation_name="Quiet owner",
        cwd="/tmp/probe/quiet",
        model_label="anthropic/claude-opus-5",
        started_at=NOW - 900.0,
        heartbeat_at=NOW - 300.0,
        busy=True,
    )
    monkeypatch.setattr(registry, "scan", lambda root=None: [(quiet, "wedged")])
    # Usage sampling is for LIVE pids only, so this one has no measurement at
    # all — which is itself part of what the /info row now declines to print.
    monkeypatch.setattr(resources, "session_resource_usage", lambda pids, **kwargs: {})
    monkeypatch.setattr(collect_mod.time, "time", lambda: NOW)

    args = argparse.Namespace(json=False, sessions_command=None, all=False, limit=None)
    assert cli.sessions_command(args) == 0
    out = capsys.readouterr().out
    assert "STATE" in out
    row = next(line for line in out.split("\n") if "Quiet owner" in line)
    assert row.startswith("not answering"), row
    assert "wedged" not in row, row

    args.json = True
    assert cli.sessions_command(args) == 0
    assert json.loads(capsys.readouterr().out)[0]["state"] == "wedged"


def test_a_drain_is_published_in_the_rows_and_named_in_the_table(
    monkeypatch: Any, capsys: Any
) -> None:
    """The drain reaches every surface an operator reads (U1/U2, PR #1141).

    Before this, a signalled runtime was INDISTINGUISHABLE from an ordinary busy
    one for up to ``SIGNAL_DRAIN_S``: `lop sessions` reported ``live`` with
    nothing else to see, and the natural next move — a plain ``lop stop`` — cut
    the turn the drain was finishing. Both readings are asserted here, from the
    same input: the published row (which ``/info`` and the JSON contract carry)
    and the human table.
    """
    import argparse

    from local_operator import cli

    _install_fixture(monkeypatch, DRAINING)
    rows = session_rows()
    assert rows[0]["leaving"] == LEAVING_ON_SIGNAL
    assert [row["leaving"] for row in rows[1:]] == ["", ""]
    # THE TRAILING KEY IS THE CONTRACT, so this asserts the rule rather than one
    # column: a field is APPENDED to the published row and never inserted, so the
    # key order every existing consumer reads is unchanged by it. The measured
    # pair (``beat_lag_s``/``cpu_since_beat_s``) is the newest extension and
    # therefore the last two; ``leaving`` and ``updating`` must still be present
    # and before them, in that order.
    assert "leaving" in rows[0] and list(rows[0]).index("leaving") < list(rows[0]).index("updating")
    assert list(rows[0]).index("updating") < list(rows[0]).index("update_failed")
    assert list(rows[0]).index("update_failed") < list(rows[0]).index("beat_lag_s")
    assert list(rows[0]).index("beat_lag_s") < list(rows[0]).index("cpu_since_beat_s")
    assert list(rows[0]).index("cpu_since_beat_s") < list(rows[0]).index("stall_dump")
    # ``stall_held`` is the third state and the newest key, appended after the dump
    # path it qualifies: a reader that has ``stall_dump`` and not this one cannot
    # tell a runtime that survived its bound from one the bound ended.
    assert list(rows[0]).index("stall_dump") < list(rows[0]).index("stall_held")
    assert list(rows[0])[-1] == "stall_held"

    assert (
        cli.sessions_command(
            argparse.Namespace(json=False, sessions_command=None, all=False, limit=None)
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "LEAVING" in out
    assert LEAVING_ON_SIGNAL in out, out
    # ...AND THE BOUND IS INSIDE IT. The row is where an operator decides whether
    # to touch a draining runtime, and the phrase alone promises a boundary the
    # 120 s bound can take away (U9): the qualified sentence is the shipped one,
    # and the column has to be wide enough to print it whole.
    assert "(up to 2 min)" in out, out
    # The drain does NOT reclassify the row: STATE is liveness (the process IS
    # alive and heartbeating), and the drain is what it is doing — the same
    # division the NEEDS column makes for a parked gate.
    assert out.startswith("STATE")
    # Matched on the ROW rather than on a literal run of spaces: the table's
    # column widths belong to the table (main added an RSS FOOTPRINT column
    # while this branch was open), and a cell that pins the padding fails for a
    # change that has nothing to do with the LEAVING column.
    row = next(line for line in out.splitlines() if line.startswith("live"))
    assert re.match(r"live\s+4243\b", row), row


def test_an_update_window_is_published_in_the_rows_and_named_in_the_table(
    monkeypatch: Any, capsys: Any
) -> None:
    """The window reaches the same two surfaces the drain does, on its own terms.

    The operator's requirement is that a session mid-update is VISIBLE ("/info"
    and ``lop sessions`` are where a fleet is inspected), and the separation from
    ``leaving`` is the part that has to hold: the two fields say opposite things
    about the operator's message, so a row that carried the window in the drain's
    column would send them to re-send a message that is already queued.
    """
    import argparse

    from local_operator import cli

    _install_fixture(monkeypatch, UPDATING_WINDOW)
    rows = session_rows()
    assert rows[0]["updating"] == UPDATE_PAIR
    assert [row["updating"] for row in rows[1:]] == ["", ""]
    assert rows[0]["leaving"] == "", "a window is not a drain: nothing is being refused"

    assert (
        cli.sessions_command(
            argparse.Namespace(json=False, sessions_command=None, all=False, limit=None)
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "UPDATING" in out
    # The CELL, rendered from the pair through the one vocabulary: the new build is
    # what a rotation script reads, so a raw pair or a bare phase would both be wrong.
    assert "updating → 0.59.11@ead71b6" in out, out
    assert "LEAVING" not in out, "the column appears only when a row carries one"


#: ``FIXTURE`` with one session whose last update FAILED — the third phase, which
#: rendered nowhere on this surface until design review round 1 (D1).
FAILED_WINDOW = [
    (replace(record, update_failed=UPDATE_PAIR) if index == 0 else record, state)
    for index, (record, state) in enumerate(FIXTURE)
]


#: ``FIXTURE`` whose first session SURVIVED its own stall bound — the third
#: attribution state, and the one that reached no rendered surface before design review
#: round 1 (D1): with the exit held, such a runtime stays stalled for the life of the
#: process, so the listing is where a person learns it needs them.
HELD_SESSION = [
    (replace(record, leaving="") if index == 0 else record, state)
    for index, (record, state) in enumerate(FIXTURE)
]


def test_a_held_runtime_is_named_in_the_fleet_table(monkeypatch: Any, capsys: Any) -> None:
    """D1: the bound fired, dumped, and did NOT end it — the row must say so.

    Before this, ``stall_held`` reached the JSON row and no screen: a held session listed
    exactly as an idle one, under a state word ("not answering") that reads as "still
    settling" — while the safety net that used to resolve it has already fired and the
    only way out is a person running ``lop stop``. Both halves are asserted: the column
    and its cell when a row is held, and no column at all when none is.
    """
    import argparse

    from local_operator import cli
    from local_operator.session.runtime import stall_watchdog

    _install_fixture(monkeypatch, HELD_SESSION)
    held_pid = HELD_SESSION[0][0].pid
    monkeypatch.setattr(stall_watchdog, "held_pids", lambda *a, **k: {held_pid})

    rows = session_rows()
    assert rows[0]["stall_held"] is True
    assert [row["stall_held"] for row in rows[1:]] == [False, False]

    assert (
        cli.sessions_command(
            argparse.Namespace(json=False, sessions_command=None, all=False, limit=None)
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "STALLED" in out, f"the held state rendered nowhere: {out}"
    assert cli.HELD_CELL in out, out
    assert "lop stop" in out, "the cell must name the way out, not only the state"


def test_a_stale_pid_with_a_leftover_dump_is_not_held(monkeypatch: Any) -> None:
    """D8 (round 2): a held dump outlives the runtime, and the phrase must not.

    ``held_pids`` is a scan of the dump files and knows nothing about the process, so a
    held runtime that a person later stopped kept ``bound held`` — the phrase this round
    exists to make mean "still running, needs you" — beside a state word saying its pid
    is gone. The fence is here rather than in the panel, because this is the layer that
    has the state, and both surfaces read what this publishes.
    """
    from local_operator.session.runtime import stall_watchdog

    stale = [
        (record, "stale" if index == 0 else state) for index, (record, state) in enumerate(FIXTURE)
    ]
    _install_fixture(monkeypatch, stale)
    monkeypatch.setattr(stall_watchdog, "held_pids", lambda *a, **k: {stale[0][0].pid})
    monkeypatch.setattr(stall_watchdog, "fired_pids", lambda *a, **k: {stale[0][0].pid})

    rows = session_rows()
    assert rows[0]["stall_dump"], "the artifact is still published: the dump is the evidence"
    assert (
        rows[0]["stall_held"] is False
    ), "a leftover dump on a dead pid was rendered as a runtime that survived its bound"


def test_the_stalled_and_updating_cells_sit_under_their_own_headers(
    monkeypatch: Any, capsys: Any
) -> None:
    """Both gates on at once — the case the round-1 fix got backwards.

    A runtime can be held AND carry a failed update, and the first version of the column
    appended the header one way round and the cell the other: the held value rendered
    under ``UPDATING`` and the update value under ``STALLED`` (agent review round 2,
    MAJOR-3, from the rendered frame). One row is enough to show it, and this is that
    row with both facts set.
    """
    import argparse

    from local_operator import cli

    both = [
        (replace(record, leaving="", update_failed=UPDATE_PAIR) if index == 0 else record, state)
        for index, (record, state) in enumerate(FIXTURE)
    ]
    _install_fixture(monkeypatch, both)
    from local_operator.session.runtime import stall_watchdog

    monkeypatch.setattr(stall_watchdog, "held_pids", lambda *a, **k: {both[0][0].pid})
    rows = session_rows()
    assert rows[0]["stall_held"] is True and rows[0]["update_failed"] == UPDATE_PAIR

    assert (
        cli.sessions_command(
            argparse.Namespace(json=False, sessions_command=None, all=False, limit=None)
        )
        == 0
    )
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    header = lines[0]
    assert header.index("UPDATING") < header.index("STALLED"), header
    row = next(ln for ln in lines[1:] if "4243" in ln)
    # The VALUES must be in the same order as the HEADERS, or each is under the other's.
    assert row.index("update failed") < row.index(cli.HELD_CELL), row


def test_the_stalled_column_is_absent_when_no_row_is_held(monkeypatch: Any, capsys: Any) -> None:
    """The gate, in the other direction: no held row anywhere means no column.

    Same rule as ``LEAVING``/``UPDATING`` — a column that prints for a state nothing is
    in teaches a reader to ignore it — and the assertion is here because the cell is
    wide enough that a stray one would push every other column along with it.
    """
    import argparse

    from local_operator import cli

    _install_fixture(monkeypatch)
    assert (
        cli.sessions_command(
            argparse.Namespace(json=False, sessions_command=None, all=False, limit=None)
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "STALLED" not in out, out


def test_a_failed_window_is_named_in_the_fleet_table(monkeypatch: Any, capsys: Any) -> None:
    """D1: a fleet whose only news is an abandoned update must say so.

    The design reviewer seeded exactly this record into an isolated root and ran the
    real CLI: the UPDATING cell was BLANK, and with no open window anywhere in the
    fleet the whole column was dropped — so the session listed byte-identically to an
    ordinary idle one. Both halves are asserted here, against the same renderer.
    """
    import argparse

    from local_operator import cli

    _install_fixture(monkeypatch, FAILED_WINDOW)
    rows = session_rows()
    assert rows[0]["update_failed"] == UPDATE_PAIR
    assert rows[0]["updating"] == "", "a failed window is not an open one"

    assert (
        cli.sessions_command(
            argparse.Namespace(json=False, sessions_command=None, all=False, limit=None)
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "UPDATING" in out, "the column must print when the only news is a failure"
    assert "update failed" in out, out
    # AND NOTHING CLAIMS THE MOVE HAPPENED: the failed cell carries no build label,
    # because naming one there would read as "went to it".
    failed_row = next(line for line in out.splitlines() if line.startswith("live"))
    assert UPDATE_PAIR not in failed_row, failed_row


def test_the_leaving_column_fits_the_shipped_phrase() -> None:
    """The column width IS the phrase's width, so a reword cannot silently cut it.

    ``cli`` deliberately keeps session internals out of its module scope, so the
    two cannot be tied together by an import; the pin lives here instead, at the
    seam that would actually break. ``_fit_cell`` cuts an over-wide cell with a
    marker rather than wrapping it, which is right for an unforeseeable value and
    wrong for this one — the phrase is a constant this project authors, so any
    excess means the two drifted and the new clause is being sliced off the row
    an operator reads (UX round 2, U9).

    WALKING EVERY PUBLISHED PHRASE, not one of them (design round 1, D3). The pin
    named ``LEAVING_ON_SIGNAL``, so a phrase added 3 cells wider than it went
    through: ``leaving for the build on disk; work stalled for 15 min`` rendered as
    ``…work stalled for 15 `` — the unit cut off the end of the one clause the
    phrase exists to state, on the last column of the row, with no marker to say
    the cell had been trimmed.
    """
    from rich.cells import cell_len

    from local_operator import cli
    from local_operator.session.runtime.types import PUBLISHED_LEAVING_PHRASES

    widest = max(PUBLISHED_LEAVING_PHRASES, key=cell_len)
    assert cli.LEAVING_COLUMN_WIDTH == cell_len(widest), (
        f"LEAVING_COLUMN_WIDTH={cli.LEAVING_COLUMN_WIDTH} but the widest published "
        f"phrase is {cell_len(widest)} cells: {widest!r}"
    )
    for phrase in PUBLISHED_LEAVING_PHRASES:
        assert cell_len(phrase) <= cli.LEAVING_COLUMN_WIDTH, (
            f"{phrase!r} is {cell_len(phrase)} cells against a column of "
            f"{cli.LEAVING_COLUMN_WIDTH}: the cell is cut without a marker"
        )


def test_the_leaving_column_is_absent_when_nobody_is_leaving(monkeypatch: Any, capsys: Any) -> None:
    """A trailing column of blanks is not an improvement to a healthy listing.

    The rule the WHY and LAST_ACTIVE columns already follow, and the reason this
    is asserted rather than assumed: every operator reading `lop sessions` on an
    ordinary day pays for this change, and they must pay nothing — same header,
    same width, same rows.
    """
    import argparse

    from local_operator import cli

    _install_fixture(monkeypatch)
    assert (
        cli.sessions_command(
            argparse.Namespace(json=False, sessions_command=None, all=False, limit=None)
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "LEAVING" not in out
    assert "Dead runtime" in out


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


def _why_cell(line: str) -> str:
    """The trailing WHY cell of a rendered row, at its own published width.

    Measured in CELLS, because that is what the column is bounded by: a row
    whose cell carries a wide glyph occupies more cells than it has characters,
    so a character slice would hand back the padding of the column before it.
    The largest suffix of exactly the published width is the cell, which for an
    all-narrow row is the same characters the old slice returned. The column's
    own right-padding is stripped: those blanks are the width contract, not
    content, and the marker guarantees no cell content ends in whitespace.
    """
    from rich.cells import cell_len

    from local_operator.cli import WHY_COLUMN_WIDTH

    for start in range(len(line), -1, -1):
        if cell_len(line[start:]) == WHY_COLUMN_WIDTH:
            return line[start:].rstrip()
    raise AssertionError(f"no {WHY_COLUMN_WIDTH}-cell suffix in {line!r}")


def _rendered_why_row(monkeypatch: Any, tmp_path: Any, capsys: Any) -> tuple[str, str]:
    """The rendered header and the live row, for the table's width invariant."""
    import argparse

    from local_operator import cli

    code = cli.sessions_command(
        argparse.Namespace(json=False, sessions_command=None, all=False, limit=None)
    )
    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    return lines[0], next(row for row in lines if "Investigate" in row)


def _rendered_why_cell(monkeypatch: Any, tmp_path: Any, capsys: Any) -> str:
    """Render the table and return the live row's WHY cell."""
    return _why_cell(_rendered_why_row(monkeypatch, tmp_path, capsys)[1])


def _before_rendered_why_cell(monkeypatch: Any, tmp_path: Any, capsys: Any, reason: str) -> str:
    """The live cell, reached through the PRE-CHANGE rule's own precondition.

    It asserts the arithmetic identity ``reason[:WHY_COLUMN_WIDTH] == reason``
    rather than rendering that expression, because at or under the width the old
    slice IS the identity — there is no second string to compare against, so the
    honest measurement is that the fitted input survives the change untouched.
    The old rule's output and the input differ only above the width, which the
    overflow cases pin directly.
    """
    from local_operator.cli import WHY_COLUMN_WIDTH

    assert reason[:WHY_COLUMN_WIDTH] == reason, "the fitting case must not be cut"
    return _rendered_why_cell(monkeypatch, tmp_path, capsys)


def test_a_reason_wider_than_the_column_is_marked_not_silently_sliced(
    monkeypatch: Any, tmp_path: Any, capsys: Any
) -> None:
    """Design round 2, D8 — now TRIGGERED by this branch's own longer sentence.

    A slice at exactly the column width is invisible: it drops the tail and the
    row still reads as a finished sentence. That was survivable only while
    every reason happened to fit — the involuntary reason filled the 48-cell
    column exactly, which is why design round 2 filed the missing marker as a
    non-blocking follow-up instead of a finding. ``CUT_OFF_UNKNOWN`` is 58
    cells, so it is the first reason this surface has that EXCEEDS the budget
    and loses a word (``determined``) with nothing to say so.

    The sentence itself is untouched — round 1 approved it for the transcript
    and notice surfaces (D3/U3) and it is asserted below to reach ``--json`` in
    full — so this is the column's own summary marker, not a copy change.
    """
    import argparse
    import json as _json

    from local_operator import cli
    from local_operator.incidents import CUT_OFF_UNKNOWN

    _install_fixture(monkeypatch)
    monkeypatch.setattr(cli, "config_dir", lambda: tmp_path)
    assert len(CUT_OFF_UNKNOWN) > cli.WHY_COLUMN_WIDTH, "this test needs a wider reason"

    # A cause token this build does not know, which is the real shape that
    # paints this sentence: a newer runtime's token reaching an older viewer.
    _seed_outcome(
        tmp_path, "a3f9c21b7e40", kind="error", reason=CUT_OFF_UNKNOWN, cause="future-cause"
    )

    code = cli.sessions_command(
        argparse.Namespace(json=False, sessions_command=None, all=False, limit=None)
    )
    assert code == 0
    out = capsys.readouterr().out
    line = next(row for row in out.splitlines() if "Investigate" in row)
    cell = _why_cell(line)

    assert len(cell) == cli.WHY_COLUMN_WIDTH, repr(cell)
    assert cell.endswith("…"), cell
    # The cut really happened, and the marker is the ONLY thing that says so.
    assert "determined" not in line, line
    assert cell == CUT_OFF_UNKNOWN[: cli.WHY_COLUMN_WIDTH - 1] + "…", cell

    code = cli.sessions_command(
        argparse.Namespace(json=True, sessions_command=None, all=False, limit=None)
    )
    assert code == 0
    payload = _json.loads(capsys.readouterr().out)
    full = next(item for item in payload if item["session_id"] == "a3f9c21b7e40")
    assert full["completion_reason"] == CUT_OFF_UNKNOWN, full["completion_reason"]


@pytest.mark.parametrize("width", [47, 48])
def test_a_reason_that_fits_the_column_is_untouched(
    width: int, monkeypatch: Any, tmp_path: Any, capsys: Any
) -> None:
    """Both fit boundaries are byte-identical to the pre-change slice.

    Pinned at the widths rather than on sample sentences because the property is
    arithmetic: the old rule's slice is the identity at or under the budget, so a
    fitting cell may not change — otherwise the marker re-flows every row that
    was already fine. 48 is the exact fit; 47 is pinned beside it because it is
    the width the OLD rule was already silently producing for the 104-cell
    ``runtime-killed`` sentence, so the marker must not start marking reasons the
    column never had to cut.
    """
    from local_operator import cli

    # The parametrisation IS the published width's own boundary pair, so moving
    # the constant fails here loudly instead of drifting out of the pin.
    assert width in (cli.WHY_COLUMN_WIDTH - 1, cli.WHY_COLUMN_WIDTH), width

    _install_fixture(monkeypatch)
    monkeypatch.setattr(cli, "config_dir", lambda: tmp_path)
    fitting = "x" * width
    _seed_outcome(tmp_path, "a3f9c21b7e40", kind="error", reason=fitting, cause="future-cause")

    cell = _before_rendered_why_cell(monkeypatch, tmp_path, capsys, fitting)
    assert cell == fitting, repr(cell)
    assert not cell.endswith("…"), repr(cell)


@pytest.mark.parametrize("width", [49, 58])
def test_a_reason_over_the_column_gains_the_marker(
    width: int, monkeypatch: Any, tmp_path: Any, capsys: Any
) -> None:
    """The two clamp boundaries: the first cell that pays, and the 58-cell one.

    49 is the cheapest possible overflow — one cell, for the ellipsis. 58 is
    ``CUT_OFF_UNKNOWN``'s measured width, the shape the column was actually
    clipping silently before the marker (see the wide-glyph case below, where the
    same 58 cells are only 29 characters).
    """
    from local_operator import cli

    assert width == cli.WHY_COLUMN_WIDTH + 1 or width == 58, width
    assert 58 > cli.WHY_COLUMN_WIDTH, "the second case overflows only while it does"

    _install_fixture(monkeypatch)
    monkeypatch.setattr(cli, "config_dir", lambda: tmp_path)
    overflowing = "y" * width
    _seed_outcome(tmp_path, "a3f9c21b7e40", kind="error", reason=overflowing, cause="future-cause")

    cell = _rendered_why_cell(monkeypatch, tmp_path, capsys)
    assert cell == overflowing[: cli.WHY_COLUMN_WIDTH - 1] + "…", repr(cell)
    assert len(cell) == cli.WHY_COLUMN_WIDTH, repr(cell)


def _widest_prefix_of_cells(reason: str, budget: int) -> str:
    """The longest prefix of ``reason`` inside ``budget`` CELLS — the spec.

    Written here rather than imported from ``cli`` so the assertion states the
    invariant (a cell bound, wide glyphs included) instead of checking the
    implementation against itself; it is the same loop any correct cell-bound
    implementation must run.
    """
    from rich.cells import cell_len

    used = 0
    kept = ""
    for char in reason:
        width = cell_len(char)
        if used + width > budget:
            break
        kept += char
        used += width
    return kept


def test_a_wide_glyph_reason_is_clamped_by_cells_not_characters(
    monkeypatch: Any, tmp_path: Any, capsys: Any
) -> None:
    """Design round 3, D9 — the input is the PROVIDER's text, so it can be wide.

    The sentences in ``CUT_OFF_CAUSES`` are harness-authored English, but they
    are not this cell's only input: a FAILED turn's reason is ``outcome.error``
    (``session.py``'s turn-end writer), the attention store replays it verbatim,
    and ``completion_reason`` is fed to the clamp. A localised provider error is
    therefore genuine input, and under a ``len()`` comparison it was returned
    UNCUT — 29 characters against 58 cells — so the row rendered 208 cells
    against a 179-cell header.

    Pinned as the invariant rather than as one string: the cell is never wider
    than ``WHY_COLUMN_WIDTH`` cells, and the row it sits in is exactly as wide as
    the header, which is the property the defect broke.
    """
    import argparse
    import json as _json

    from rich.cells import cell_len

    from local_operator import cli

    _install_fixture(monkeypatch)
    monkeypatch.setattr(cli, "config_dir", lambda: tmp_path)
    # The defect's exact shape: the OLD rule let this through unclamped.
    reason = "模型调用失败：上游返回了无效的响应，请稍后重试或者更换模型"
    assert len(reason) <= cli.WHY_COLUMN_WIDTH < cell_len(reason), (
        len(reason),
        cell_len(reason),
    )
    _seed_outcome(tmp_path, "a3f9c21b7e40", kind="error", reason=reason, cause="future-cause")

    header, row = _rendered_why_row(monkeypatch, tmp_path, capsys)
    cell = _why_cell(row)

    assert cell_len(cell) <= cli.WHY_COLUMN_WIDTH, (cell_len(cell), repr(cell))
    # Q4 / D3, recorded rather than "fixed": an all-wide reason cannot use the
    # 47th cell, because a two-cell glyph cannot occupy an odd cell. The cell
    # therefore measures 47 and its last cell stays UNUSED — raggedness in one
    # cell of trailing space, not a correctness gap, and deliberately not padded
    # to 48: filling it would report width the text does not have.
    assert cell_len(cell) == cli.WHY_COLUMN_WIDTH - 1, cell_len(cell)
    assert cell == _widest_prefix_of_cells(reason, cli.WHY_COLUMN_WIDTH - 1) + "…", repr(cell)
    # The reflow, measured: a row that widened past the header is the bug, so the
    # table's own width is the assertion, not the string's length.
    # (The exactly-48 variant of this cell, which an all-wide reason cannot
    # reach, is pinned in the mixed-glyph test below.)
    assert cell_len(row) == cell_len(header), (cell_len(row), cell_len(header), row)

    # And the cut is the column's summary only: the full sentence stays one flag
    # away, which is what makes a marked cell safe to publish.
    code = cli.sessions_command(
        argparse.Namespace(json=True, sessions_command=None, all=False, limit=None)
    )
    assert code == 0
    payload = _json.loads(capsys.readouterr().out)
    full = next(item for item in payload if item["session_id"] == "a3f9c21b7e40")
    assert full["completion_reason"] == reason, full["completion_reason"]


def test_a_wide_glyph_reason_that_fits_the_column_still_pads_by_cells(
    monkeypatch: Any, tmp_path: Any, capsys: Any
) -> None:
    """The padding is the same char-vs-cell arithmetic as the clamp.

    24 double-width characters are 48 cells and 24 characters, so a cell that
    fills the column exactly must be returned untouched AND padded by zero — the
    format spec's own ``:<48`` would have added 24 blanks the cell did not need
    and left the row 24 cells wider than the header. Nothing follows this column
    today, so the excess is invisible trailing space; the table's width contract
    is still what a reader of the frame measures.
    """
    from rich.cells import cell_len

    from local_operator import cli

    _install_fixture(monkeypatch)
    monkeypatch.setattr(cli, "config_dir", lambda: tmp_path)
    fitting = "模" * (cli.WHY_COLUMN_WIDTH // 2)
    assert cell_len(fitting) == cli.WHY_COLUMN_WIDTH
    _seed_outcome(tmp_path, "a3f9c21b7e40", kind="error", reason=fitting, cause="future-cause")

    header, row = _rendered_why_row(monkeypatch, tmp_path, capsys)
    assert _why_cell(row) == fitting, repr(_why_cell(row))
    assert cell_len(row) == cell_len(header), (cell_len(row), cell_len(header), row)


def test_a_wide_glyph_reason_lands_on_the_full_budget_when_its_glyphs_allow_it(
    monkeypatch: Any, tmp_path: Any, capsys: Any
) -> None:
    """A wide reason clamped to EXACTLY 48 cells, not merely inside them.

    An all-wide reason cannot land on the budget: 47 is odd and every glyph costs
    two cells, so the widest prefix that fits is 46 cells and the marker leaves
    the cell at 47. One narrow glyph in the mix makes 47 reachable, and that is
    the case that proves the cut is bounded BY CELLS rather than stopping early
    because a cell bound looked safe — the difference between a correct cut and
    an over-conservative one is invisible in the all-wide case, which is what
    ``test_a_wide_glyph_reason_is_clamped_by_cells_not_characters`` pins alone.
    """
    from rich.cells import cell_len

    from local_operator import cli

    _install_fixture(monkeypatch)
    monkeypatch.setattr(cli, "config_dir", lambda: tmp_path)
    reason = "模" * 23 + "x" + "模" * 10
    assert cell_len(reason) > cli.WHY_COLUMN_WIDTH, cell_len(reason)
    assert (
        cell_len(_widest_prefix_of_cells(reason, cli.WHY_COLUMN_WIDTH - 1))
        == cli.WHY_COLUMN_WIDTH - 1
    ), "this reason was built to reach the budget exactly"
    _seed_outcome(tmp_path, "a3f9c21b7e40", kind="error", reason=reason, cause="future-cause")

    header, row = _rendered_why_row(monkeypatch, tmp_path, capsys)
    cell = _why_cell(row)

    assert cell.endswith("…"), repr(cell)
    assert cell_len(cell) == cli.WHY_COLUMN_WIDTH, (cell_len(cell), repr(cell))
    assert cell == _widest_prefix_of_cells(reason, cli.WHY_COLUMN_WIDTH - 1) + "…", repr(cell)
    # The reflow itself: a 48-cell cell pads by ZERO, so the row is the header's
    # own width even though the cell has fewer characters than cells.
    assert cell_len(row) == cell_len(header), (cell_len(row), cell_len(header), row)


def _cells_span(line: str, start: int, width: int) -> str:
    """The characters covering cells ``start .. start+width`` of ``line``.

    Written the way the terminal addresses the table rather than by slicing the
    string: a fixed-width column is a span of CELLS, so character offsets cannot
    name it once any cell is wide. This is what lets an assertion say "this column
    holds these 24 cells" for a CJK row, which is the only way to check that a
    clamped cell did not shift its neighbours.

    It walks GRAPHEMES, with rich's own splitter — the first version walked
    characters and measured each with ``cell_len(char)``, which is the same
    per-character mistake the clamp itself had (review round 2, M1): it started
    8 cells late on a VS16 row, because a selection sequence measures 1 cell per
    character and 2 as a unit. A cell walk has to use the unit the table does.
    """
    from rich.cells import split_graphemes

    spans, _total_cells = split_graphemes(line)
    used = 0
    kept = ""
    for span_start, span_end, span_width in spans:
        if used >= start + width:
            break
        if used + span_width > start:
            kept += line[span_start:span_end]
        used += span_width
    return kept


def test_ascii_rows_still_render_by_the_character_rule(
    monkeypatch: Any, tmp_path: Any, capsys: Any
) -> None:
    """The equivalence witness for clamping the three text columns (D2 / Q2).

    ``cell_len`` equals ``len`` for ASCII, so the new bound must be
    indistinguishable from the old one — ``value[:24]`` and the ``:<24`` padding
    it fed. Re-derived here from the row dicts as an expression of the OLD rule
    rather than asserted against a stored table, so this is a check and not a
    restatement of the implementation: if the clamp or its cell padding moved one
    ASCII character, it fails.

    The fixture is a real witness rather than a trivial one: "Investigate request
    latency" and "anthropic/claude-sonnet-4-6" are 27 characters against a
    24-character column, and NEEDS holds exactly its 8.
    """
    import argparse

    from rich.cells import cell_len

    from local_operator import cli

    _install_fixture(monkeypatch)
    monkeypatch.setattr(cli, "config_dir", lambda: tmp_path)
    rows = session_rows(tmp_path)

    code = cli.sessions_command(
        argparse.Namespace(json=False, sessions_command=None, all=False, limit=None)
    )
    assert code == 0
    header, *lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(rows), (len(lines), len(rows))

    # An ASCII header puts every label at the same character and cell offset, so
    # the columns below it can be addressed by the header's own offsets.
    needs_at = header.index("NEEDS")
    conversation_at = header.index("CONVERSATION")
    model_at = header.index("MODEL")

    for row, line in zip(rows, lines):
        assert cell_len(line) == cell_len(header), line
        old_needs = f"{(row.get('pending') or '')[:8]:<8}"
        old_name = f"{(row['conversation_name'] or row['session_id'] or '')[:24]:<24}"
        old_model = f"{(row['model_label'] or '')[:24]:<24}"
        assert line[needs_at : needs_at + 8] == old_needs, line
        assert line[conversation_at : conversation_at + 24] == old_name, line
        assert line[model_at : model_at + 24] == old_model, line


def test_a_wide_title_model_and_needs_are_clamped_by_cells(
    monkeypatch: Any, tmp_path: Any, capsys: Any
) -> None:
    """Design round 1, D2 / QA round 1, Q2 — the same defect, one column over.

    CONVERSATION, MODEL and NEEDS were cut by characters, so a 14-glyph CJK title
    (28 cells) went into a 24-cell column and pushed its neighbours along: design
    measured 181 cells against a 167-cell header. These values come from OUTSIDE
    this process — the title is whatever named the conversation, the label is the
    provider catalogue's own display name — so they are the same class of input
    the WHY cell had to stop trusting.

    No marker is added, and that is the point of the ASCII witness above: these
    columns have never carried one, and a marker would move every existing ASCII
    listing that overflows.
    """
    import argparse
    from dataclasses import replace

    from rich.cells import cell_len

    from local_operator import cli
    from local_operator.session.runtime import registry

    _install_fixture(monkeypatch)
    monkeypatch.setattr(cli, "config_dir", lambda: tmp_path)
    title = "模" * 14  # 28 cells into a 24-cell column
    model = "模型" * 13 + "/模"  # wider still, and mixed
    needs = "模" * 6  # 12 cells into an 8-cell column
    wide = [
        (
            replace(
                record,
                conversation_name=title,
                model_label=model,
                # `_OldRecord` predates the field, exactly as a running runtime
                # of that vintage does; the title and label still apply to it.
                **({"pending": needs} if hasattr(record, "pending") else {}),
            ),
            state,
        )
        for record, state in FIXTURE
    ]
    monkeypatch.setattr(registry, "scan", lambda root=None: wide)

    code = cli.sessions_command(
        argparse.Namespace(json=False, sessions_command=None, all=False, limit=None)
    )
    assert code == 0
    header, *lines = capsys.readouterr().out.splitlines()
    assert lines, "the fixture must render rows"
    rows = session_rows(tmp_path)
    assert len(lines) == len(rows), (len(lines), len(rows))

    conversation_at = header.index("CONVERSATION")
    model_at = header.index("MODEL")
    needs_at = header.index("NEEDS")

    for row, line in zip(rows, lines):
        # The defect's shape: the row was wider than its own header.
        assert cell_len(line) == cell_len(header), (cell_len(line), cell_len(header), line)
        assert "…" not in line, line
        # `_OldRecord` predates NEEDS, so that row's cell is blank — kept in the
        # set on purpose, because a row with nothing to clamp must still come out
        # header-width.
        expected_needs = "模" * 4 if row.get("pending") else " " * cli.NEEDS_COLUMN_WIDTH
        assert _cells_span(line, needs_at, cli.NEEDS_COLUMN_WIDTH) == expected_needs, line
        assert _cells_span(line, conversation_at, cli.CONVERSATION_COLUMN_WIDTH) == "模" * 12, line
        assert _cells_span(line, model_at, cli.MODEL_COLUMN_WIDTH) == "模型" * 6, line


def test_the_clamp_never_ends_a_reason_on_a_dangling_joiner() -> None:
    """Review round 1, Q1 — bounded cells, but a stray control character.

    ``U+200D`` means "join with the glyph AFTER me", so a cut landing right after
    one emitted a joiner with nothing to join, sitting immediately before the
    marker: a replacement box on some terminals, inside a cell that is otherwise
    the right width. Exercised through the clamp's own entry point because the
    case needs a glyph mix no fixture row carries.
    """
    from rich.cells import cell_len

    from local_operator import cli

    cluster = "\U0001f469\u200d\U0001f469"  # woman ZWJ woman, one 2-cell cluster

    def old_cut(text: str, budget: int) -> str:
        """The pre-fix loop, so the witness below cannot rot into a tautology.

        It is the same walk minus the joiner back-off, which is the only thing
        this test exists to establish.
        """
        used = 0
        for index, char in enumerate(text):
            width = cell_len(char)
            if used + width > budget:
                return text[:index]
            used += width
        return text

    # Straddles the cut: 49 cluster-aware cells, and the OLD loop stopped right
    # after the first cluster's joiner.
    straddling = "x" * 45 + cluster * 2
    assert cell_len(straddling) == 49, cell_len(straddling)
    pre_fix = old_cut(straddling, cli.WHY_COLUMN_WIDTH - 1)
    assert pre_fix.endswith("\u200d"), "the straddle this test needs is gone"
    cell = cli._clamp_reason_cell(straddling)
    assert cell.endswith("…"), repr(cell)
    assert not cell[:-1].endswith("\u200d"), repr(cell)
    assert cell_len(cell) <= cli.WHY_COLUMN_WIDTH, cell_len(cell)

    # A joiner INSIDE the kept prefix is doing its job and must survive: the
    # back-off is for a trailing joiner only.
    interior = "x" * 43 + cluster * 3
    assert cell_len(interior) > cli.WHY_COLUMN_WIDTH, cell_len(interior)
    kept = cli._clamp_reason_cell(interior)
    assert "\u200d" in kept, repr(kept)
    assert not kept[:-1].endswith("\u200d"), repr(kept)


def test_fit_and_pad_are_the_character_rule_for_ascii() -> None:
    """The smallest form of the D2 / Q2 proof, on the two primitives.

    ``_fit_cell`` is ``value[:width]`` and ``_pad_cell`` is ``f"{value:<{width}}"``
    whenever the text is ASCII, which is the whole of the byte-identity guarantee:
    the change is a bound, not a rendering. The long value is included because the
    fixture's own title overflows its column.
    """
    from rich.cells import cell_len

    from local_operator import cli

    for value in ("", "x", "Investigate request latency", "x" * 24, "x" * 25, "x" * 100):
        assert cell_len(value) == len(value)
        for width in (cli.NEEDS_COLUMN_WIDTH, cli.CONVERSATION_COLUMN_WIDTH):
            fitted = cli._fit_cell(value, width)
            assert fitted == value[:width], (value, width, fitted)
            assert cli._pad_cell(fitted, width) == f"{fitted:<{width}}", (value, width)


def test_sequence_glyphs_are_measured_as_units_not_characters() -> None:
    """Review round 2, M1 and M2 — the per-character measure, on its own cases.

    These are the exact inputs the reviewer ran: rich applies the VS16 upgrade and
    the ZWJ collapse only when it measures a STRING, so a loop advancing by
    ``cell_len(char)`` mis-measured both in opposite directions — a 40-cell
    selection sequence came back UNCUT against a 24-cell budget (worse than the
    character rule it replaced, which clipped it), and a family cluster was
    charged about three times its width, leaving a third of the column used.

    Pinned as a bound over every class and width rather than on one string,
    because the defect was a MEASURE and any of them could drift back.
    """
    from rich.cells import cell_len

    from local_operator import cli

    measured = {
        "selection": cell_len(VS16),
        "keycap": cell_len(KEYCAP),
        "family": cell_len(FAMILY),
    }
    assert measured == {"selection": 2, "keycap": 2, "family": 2}, measured
    assert len(VS16) == 2 and len(FAMILY) == 5, "and each is more than one character"

    # M1: over budget in both directions of the old error.
    assert cell_len(cli._fit_cell(VS16 * 20, 24)) == 24, "was returned UNCUT at 40 cells"
    assert cell_len(cli._fit_cell(KEYCAP * 20, 8)) == 8, "was returned at 16 cells"
    # M2: the collapse is no longer charged per code point — 47 cells of family
    # used to come back with 16 of them filled.
    assert cell_len(cli._cut_to_cells(FAMILY * 20, 47)) == 40, cell_len(
        cli._cut_to_cells(FAMILY * 20, 47)
    )
    assert cell_len(cli._fit_cell(FAMILY * 20, 24)) == 24, "the column is not a third used"

    for value in (VS16 * 20, KEYCAP * 20, FAMILY * 20):
        for width in (cli.NEEDS_COLUMN_WIDTH, cli.CONVERSATION_COLUMN_WIDTH, 47):
            fitted = cli._fit_cell(value, width)
            assert cell_len(fitted) <= width, (value[:4], width, cell_len(fitted))
            assert cell_len(fitted) > 0 or width == 0

    # The marked cell too: a reason over the budget is clamped AND marked, with the
    # marker inside the budget and no joiner left dangling by the cut.
    for reason in (VS16 * 30, KEYCAP * 30, FAMILY * 30):
        cell = cli._clamp_reason_cell(reason)
        assert cell.endswith("…"), repr(cell)
        assert cell_len(cell) <= cli.WHY_COLUMN_WIDTH, (cell_len(cell), repr(cell))
        assert not cell[:-1].endswith("\u200d"), repr(cell)


def test_sequence_glyph_rows_keep_the_table_header_width(
    monkeypatch: Any, tmp_path: Any, capsys: Any
) -> None:
    """The property a reader sees, on the classes that had no fixture row.

    The helper-level pins above are what let M1 through: a helper can be measured
    correctly and the assembled row still be wrong, which is exactly what the
    review measured (header 118 with rows 142/142/138 at the previous head). So
    this drives the real command over rows whose text columns carry a VS16
    selection, a keycap sequence and a ZWJ family — plus a WHY reason built from
    the same classes — and pins the row-level invariant: every row is exactly as
    wide as its header, and each clamped cell is inside its own column.
    """
    import argparse

    from rich.cells import cell_len

    from local_operator import cli
    from local_operator.session.runtime import registry

    _install_fixture(monkeypatch)
    monkeypatch.setattr(cli, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(registry, "scan", lambda root=None: GLYPH_FIXTURE)
    _seed_outcome(tmp_path, "facedeadbeef", kind="error", reason=VS16 * 30, cause="future-cause")
    _seed_outcome(tmp_path, "facedeadcafe", kind="error", reason=FAMILY * 30, cause="future-cause")

    code = cli.sessions_command(
        argparse.Namespace(json=False, sessions_command=None, all=False, limit=None)
    )
    assert code == 0
    header, *lines = capsys.readouterr().out.splitlines()
    assert "WHY" in header, header
    assert len(lines) == len(GLYPH_FIXTURE), (len(lines), len(GLYPH_FIXTURE))

    needs_at = header.index("NEEDS")
    conversation_at = header.index("CONVERSATION")
    model_at = header.index("MODEL")
    why_at = header.index("WHY")

    for line in lines:
        assert cell_len(line) == cell_len(header), (cell_len(line), cell_len(header), line)

    # Each clamped cell is inside its column, and the sequences are whole ones —
    # a clamp may not split a VS16 selection from its base or a family cluster.
    selection, family = lines
    assert _cells_span(selection, conversation_at, cli.CONVERSATION_COLUMN_WIDTH) == VS16 * 12
    assert _cells_span(selection, model_at, cli.MODEL_COLUMN_WIDTH) == KEYCAP * 12
    assert _cells_span(selection, needs_at, cli.NEEDS_COLUMN_WIDTH) == VS16 * 4
    assert _cells_span(family, conversation_at, cli.CONVERSATION_COLUMN_WIDTH) == FAMILY * 12
    assert _cells_span(family, model_at, cli.MODEL_COLUMN_WIDTH) == FAMILY * 12
    # A value already inside its column is returned untouched, sequence intact,
    # and padded by the column's remaining CELLS: 6 cells of family plus 2.
    assert _cells_span(family, needs_at, cli.NEEDS_COLUMN_WIDTH) == FAMILY * 3 + "  "

    for line in lines:
        # The span covers the padded column, so the padding comes off before the
        # cell's own shape is asserted.
        reason_cell = _cells_span(line, why_at, cli.WHY_COLUMN_WIDTH).rstrip()
        assert reason_cell.endswith("…"), repr(reason_cell)
        assert cell_len(reason_cell) <= cli.WHY_COLUMN_WIDTH, (cell_len(reason_cell), reason_cell)
        assert not reason_cell[:-1].endswith("\u200d"), repr(reason_cell)
