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

import pytest

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
