"""Current-session diagnostics: real slash dispatch, read-only and stale-safe."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from dataclasses import replace
from datetime import datetime

import pytest
from rich.style import Style

from local_operator.analytics.model import (
    COMPONENT_KEYS,
    SessionReport,
    SessionRequest,
    TimingSummary,
    UsageAggregate,
)
from local_operator.analytics.store import AnalyticsStore
from local_operator.session.frontend_state import FrontendSessionState
from local_operator.tui import theme as theme_mod
from local_operator.tui.app import OperatorApp, slash_command_for
from local_operator.tui.widgets.analytics_panel import METRIC_COST, METRIC_TOKENS
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.session_panel import (
    SessionDiagnostics,
    SessionScreen,
    build_session_report,
)
from tests.unit.analytics.test_store import _snap
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_slash_echo import _submit


def flowed(text: str) -> str:
    """The report's text with wrapping collapsed, for asserting on PROSE.

    ``_Body.note`` wraps a footnote at build time and indents every line, so a
    sentence spans several lines with a two-space indent on each (design D2 —
    folding instead put the continuation at column 0, where it read as a stray
    table row). A test about WORDING should not therefore depend on where the
    line broke; a test about LAYOUT asserts on the raw lines instead.
    """
    return " ".join(text.split())


def runtime():
    return SessionDiagnostics(
        "sess",
        "Investigate request latency",
        "selected/model",
        "effective/model",
        False,
        context_tokens=1200,
        context_window=200000,
        context_is_estimate=True,
        generation=3,
    )


def test_report_empty_unavailable_and_cost_knowledge():
    empty = build_session_report(SessionReport("sess"), runtime()).plain
    unavailable = build_session_report(SessionReport("sess", available=False), runtime()).plain
    assert "No recorded requests" in empty and "Ledger unavailable" not in empty
    assert "Ledger unavailable" in unavailable and "No recorded requests" not in unavailable
    assert "selected/model" in empty and "effective/model" in empty
    # An empty ledger still gets the live gauge — it is runtime data, not a
    # ledger read — with the `~` estimate mark and the window as the note.
    assert "Context window" in empty and "In context now" in empty
    assert "~1.2k" in empty and "of 200k" in empty
    # A failed read draws NO chart section, gauge included: nothing on that
    # frame should invite the reader to trust a number.
    assert "Context window" not in unavailable and "In context now" not in unavailable
    for cost, known, expected in [
        (0, 0, "$—"),
        (0, 1, "$0.0000+"),
        (0, 2, "$0.0000"),
        (1000, 1, "$0.0010+"),
    ]:
        report = SessionReport(
            "sess", aggregate=UsageAggregate(calls=2, cost_micro=cost, cost_known_calls=known)
        )
        text = flowed(build_session_report(report, runtime()).plain)
        assert expected in text
        assert "input, output and tool dollars are not recorded separately" in text
        assert "list-price estimate" in text
        assert "pending recorder writes" in text
        assert "not provider compute" in text
        # The tool caveat is unconditional, so its wording is pinned here: it is
        # the only thing keeping the Tool surface section from being read as
        # per-tool billing, and shortening it away is a §3.7 blocker.
        assert "Per-tool-name tokens and dollars are not recorded" in text
        assert "not the cost of any one tool call" in text


def test_runtime_captures_only_supported_scalars():
    class SessionWithState(FakeSession):
        frontend_state: FrontendSessionState

    session = SessionWithState()
    session.frontend_state = FrontendSessionState(
        session_id="sess", epoch="test", generation=8, context_tokens=90, context_window=100
    )
    captured = SessionDiagnostics.capture(session)
    session.frontend_state = session.frontend_state.model_copy(update={"generation": 9})
    assert captured.generation == 8 and captured.context_tokens == 90
    assert SessionDiagnostics.capture(FakeSession()).context_tokens is None


def test_slash_discovery_and_locality():
    from local_operator.session.frontend_state import CommandScope, _slash_capabilities

    command = slash_command_for("/session")
    assert command is not None and not command.echo and not command.consumes_prompt
    capabilities = _slash_capabilities()
    assert (
        next(c for c in capabilities if c.command == "session").scope == CommandScope.FRONTEND_LOCAL
    )


@pytest.mark.parametrize("size", [(80, 24), (50, 24)])
@pytest.mark.parametrize("close_key", ["escape", "q"])
@pytest.mark.asyncio
async def test_assembled_slash_snapshot_scroll_and_focus(tmp_path, monkeypatch, size, close_key):
    path = tmp_path / "ledger.db"
    monkeypatch.setattr("local_operator.analytics.store.default_db_path", lambda: path)
    store = AnalyticsStore(path)
    store.record_batch([replace(_snap(session_id="sess"), request_id="request-" + "a" * 80)])
    store.close()
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        history = list(session.history())
        await _submit(pilot, app, "/session")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(app.screen, SessionScreen)
        screen = app.screen
        assert screen.report is not None
        assert screen.report.aggregate.calls == 1
        assert screen._scroll.max_scroll_y > 0
        assert screen._scroll.max_scroll_x == 0
        await pilot.press("end")
        await pilot.pause()
        assert screen._scroll.scroll_y > 0
        await pilot.press("home")
        await pilot.pause()
        assert screen._scroll.scroll_y == 0
        store.record_batch([_snap(session_id="sess")])
        store.close()
        assert screen.report.aggregate.calls == 1  # pinned until close/reopen
        await pilot.press(close_key)
        await pilot.pause()
        assert not isinstance(app.screen, SessionScreen)
        assert app.focused is app.query_one(Editor)
        assert session.history() == history
        assert not session.prompts
        await _submit(pilot, app, "/session")
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, SessionScreen)
        assert app.screen.report is not None
        assert app.screen.report.aggregate.calls == 2


@pytest.mark.parametrize("replacement_kind", ["new", "resume", "epoch"])
@pytest.mark.asyncio
async def test_disk_worker_is_off_loop_and_drops_stale_session(monkeypatch, replacement_kind):
    started = threading.Event()
    finish = threading.Event()
    ui_thread = threading.get_ident()
    observed = []

    def delayed(self, session_id):
        observed.append(threading.get_ident())
        started.set()
        assert finish.wait(5)
        return SessionReport(session_id)

    monkeypatch.setattr(AnalyticsStore, "session_report", delayed)

    class EpochSession(FakeSession):
        frontend_state: FrontendSessionState

    session = EpochSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test() as pilot:
        await pilot.pause()
        session.frontend_state = FrontendSessionState(session_id="sess", epoch="before")
        app._cmd_session("", lambda body, kind="info": None)
        try:
            assert await asyncio.to_thread(started.wait, 5)

            # This loop-side mutation is possible while SQLite is blocked. A
            # replacement for the same ID models /resume as well as /new.
            class NewSession(FakeSession):
                @property
                def session_id(self) -> str:
                    return "new"

            if replacement_kind == "epoch":
                session.frontend_state = session.frontend_state.model_copy(
                    update={"epoch": "after"}
                )
            else:
                app._session = FakeSession() if replacement_kind == "resume" else NewSession()
            finish.set()
            await app.workers.wait_for_complete()
            assert len(observed) == 1 and observed[0] != ui_thread
            # The result cannot be published, even if the pending modal is
            # still finishing its queued mount before it can safely be popped.
            if isinstance(app.screen, SessionScreen):
                assert app.screen.presentation_cancelled and app.screen.report is None
            await pilot.pause()
            assert not isinstance(app.screen, SessionScreen)
        finally:
            finish.set()
            app._session = session


@pytest.mark.asyncio
async def test_invalid_arguments_never_request_model():
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test() as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/session another-id")
        await app.workers.wait_for_complete()
        assert not isinstance(app.screen, SessionScreen)
        assert not session.prompts


@pytest.mark.asyncio
async def test_scroll_hint_tracks_actual_overflow_after_resize():
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        screen = SessionScreen(SessionReport("sess"), runtime())
        app.push_screen(screen)
        await pilot.pause()
        assert screen._scroll.max_scroll_y == 0
        assert "scroll" not in str(screen._hint.render())
        await pilot.resize_terminal(50, 24)
        await pilot.pause()
        assert screen._scroll.max_scroll_y > 0
        assert "↑↓ scroll" in str(screen._hint.render())
        assert screen._scroll.max_scroll_x == 0


@pytest.mark.parametrize("cancel_key", ["escape", "q"])
@pytest.mark.asyncio
async def test_real_locked_read_can_be_cancelled_without_interrupting_draft(
    tmp_path, monkeypatch, cancel_key
):
    path = tmp_path / "ledger.db"
    monkeypatch.setattr("local_operator.analytics.store.default_db_path", lambda: path)
    store = AnalyticsStore(path)
    store.record_batch([_snap(session_id="sess")])
    store.close()
    # WAL readers intentionally bypass a writer lock. A real legacy DELETE
    # journal reproduces the reported slow-read path without sleeping a mock.
    locker = sqlite3.connect(path)
    locker.execute("PRAGMA journal_mode=DELETE")
    locker.execute("BEGIN EXCLUSIVE")
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    try:
        async with app.run_test(size=(50, 24)) as pilot:
            await pilot.pause()
            await _submit(pilot, app, "/session")
            try:
                assert isinstance(app.screen, SessionScreen)
                pending = app.screen
                assert pending.report is None
                assert "Loading usage records" in str(pending._body.render())
                assert "esc / q cancel" in str(pending._hint.render())
                await pilot.press(cancel_key)
                await pilot.press(*list("next task"))
                await pilot.pause()
                assert app.focused is app.query_one(Editor)
                assert app.query_one(Editor).text == "next task"
            finally:
                locker.rollback()
                await app.workers.wait_for_complete()
            await pilot.pause()
            assert not isinstance(app.screen, SessionScreen)
            assert pending.presentation_cancelled and pending.report is None
            assert app.focused is app.query_one(Editor)
            assert app.query_one(Editor).text == "next task"
            assert session.prompts == [] and session.aborts == []
    finally:
        locker.close()


@pytest.mark.asyncio
async def test_fast_result_during_mount_replaces_loading_text():
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = SessionScreen(None, runtime())
        app.push_screen(screen)
        # No pause/await: the SQLite worker can settle before the mounted body
        # exists. Losing this update would strand a fast read in loading state.
        screen.set_report(SessionReport("sess"))
        await pilot.pause()
        assert "No recorded requests" in str(screen._body.render())
        assert "Loading usage records" not in str(screen._body.render())
        assert "cancel" not in str(screen._hint.render())


@pytest.mark.asyncio
async def test_invalidating_buried_pending_view_never_pops_newer_modal():
    from local_operator.tui.widgets.analytics_panel import AnalyticsScreen

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test() as pilot:
        await pilot.pause()
        pending = SessionScreen(None, runtime())
        app.push_screen(pending)
        await pilot.pause()
        other = AnalyticsScreen(UsageAggregate())
        app.push_screen(other)
        await pilot.pause()
        pending.invalidate()
        pending.set_report(SessionReport("sess"))
        assert app.screen is other
        assert pending.report is None
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, (SessionScreen, AnalyticsScreen))
        assert app.focused is app.query_one(Editor)


# -- chart rendering ---------------------------------------------------------
#
# The screen is a set of bar tables whose whole value is that a bar means a
# known fraction of a known denominator. These tests pin the arithmetic, the
# column discipline that lets a reader compare rows across sections, and the
# degraded frames where the honest answer is "we cannot draw that".


def _agg(calls=1, ok=None, ctx=0, out=0, cost=0, known=0, components=None, reasoning=0, cached=0):
    aggregate = UsageAggregate(
        calls=calls,
        ok_calls=calls if ok is None else ok,
        context_tokens=ctx,
        output_tokens=out,
        cost_micro=cost,
        cost_known_calls=known,
        reasoning_tokens=reasoning,
        cache_read_tokens=cached,
    )
    if components:
        aggregate.components = {key: components.get(key, 0) for key in COMPONENT_KEYS}
    return aggregate


def _request(index, purpose="turn", outcome="ok", ctx=40000, out=1000, duration=2000.0):
    return SessionRequest(
        request_id=f"r{index}",
        ts_ms=1788602400000 + index * 47000,
        provider="anthropic",
        model_id="claude-sonnet-4-6",
        purpose=purpose,
        outcome=outcome,
        usage_reported=True,
        context_tokens=ctx,
        output_tokens=out,
        duration_ms=duration,
        ttft_ms=400.0,
        preparation_ms=25.0,
    )


def _populated():
    """A session with every section populated, priced, and partly failed."""
    components = {
        "conversation": 312200,
        "tool_results": 202600,
        "system_prompt": 172700,
        "tool_schemas": 103000,
        "custom_instructions": 46500,
        "tool_inventory": 29900,
    }
    return SessionReport(
        "sess",
        aggregate=_agg(25, 23, 900000, 37800, 49000, 25, components, 9500, 630000),
        by_model={
            ("anthropic", "claude-sonnet-4-6"): _agg(14, 14, 560000, 33600, 24990, 14),
            ("anthropic", "claude-haiku-4-5"): _agg(5, 5, 270000, 2700, 3870, 5),
            ("openai", "gpt-5.2-mini"): _agg(6, 4, 52000, 1500, 20140, 6),
        },
        by_purpose={
            "turn": _agg(14, 12, 560000, 33600, 24990, 14),
            "compaction": _agg(3, 2, 270000, 2700, 3870, 3),
            "aside": _agg(4, 4, 48000, 1200, 5340, 4),
        },
        by_purpose_outcome={
            ("turn", "ok"): 12,
            ("turn", "error"): 2,
            ("compaction", "ok"): 2,
            ("compaction", "error"): 1,
            ("aside", "ok"): 4,
        },
        timings={
            "duration_ms": TimingSummary(25, 2223, 400, 3770),
            "ttft_ms": TimingSummary(25, 410, 320, 606),
            "preparation_ms": TimingSummary(25, 26, 18, 44),
        },
        recent=tuple(
            _request(i, ctx=40000 + i * 4000, duration=2000.0 + i * 70)
            for i in range(11, -1, -1)  # the store returns newest-first
        ),
        first_ts_ms=1788602400000,
        last_ts_ms=1788602400000 + 700000,
    )


def _section(text, section):
    """Every line of one section: from its header to the next blank line."""
    lines = text.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(f"▌ {section}"))
    out = []
    for line in lines[start + 1 :]:
        if not line.strip():
            break
        out.append(line)
    return out


def _rows(text, section):
    """Just the table rows of one section, without its prose footnotes.

    A bar row is identified by its TRACK (`···`), not by any single glyph: the
    purpose legend and the tool caveat are prose containing `·` as a separator,
    and matching on that swept them in as rows.
    """
    return [line for line in _section(text, section) if "███" in line or "···" in line]


def _bar_column(line):
    """The x position where a row's bar starts."""
    return min(line.index(ch) for ch in "█·" if ch in line)


def test_sections_share_one_measured_column_set():
    """One left edge, one bar column, one number column, down the whole screen.

    This is the finding against /analytics that this screen exists not to
    repeat: measuring each table on its own rows starts the component bars and
    the per-provider numbers at two different x positions on one panel.
    """
    text = build_session_report(_populated(), runtime(), 88).plain
    sections = ["Context window", "By model", "By purpose", "Where input went", "Tool surface"]
    bar_starts = {_bar_column(row) for section in sections for row in _rows(text, section)}
    assert len(bar_starts) == 1, f"bars start at {sorted(bar_starts)}, expected one column"


def test_by_model_and_purpose_partition_the_session_total():
    """Share-of-total, so the rows of one section sum to ~100%."""
    text = build_session_report(_populated(), runtime(), 88).plain
    for section in ("By model", "By purpose"):
        percents = [
            int(p.rstrip("%"))
            for row in _rows(text, section)
            for p in row.split()
            if p.endswith("%")
        ]
        assert len(percents) == 3
        assert 99 <= sum(percents) <= 101, f"{section} percentages sum to {sum(percents)}"
        assert percents == sorted(percents, reverse=True), f"{section} is not sorted descending"


def test_tool_surface_percentages_match_where_input_went():
    """The same three components, the same denominator, so the same numbers.

    A different denominator here would print Tool results as 60% in one section
    and 23% in the other, on one screen, for the same tokens.
    """
    text = build_session_report(_populated(), runtime(), 88).plain
    split = {
        row.split("█")[0].strip(): row.split()[-1]
        for row in _rows(text, "Where input went")
        if "Tool" in row
    }
    tools = {
        row.split("█")[0].strip().removeprefix("└ "): row.split()[-1]
        for row in _rows(text, "Tool surface")
        if row.strip().startswith("└")
    }
    assert tools and tools == split
    # The parent is drawn in that same denominator, so parent ≈ Σ children.
    parent = next(r for r in _rows(text, "Tool surface") if "All tool context" in r)
    # Within one point: each row rounds its own percentage independently, so
    # the subtotal need not equal the sum of the rounded children exactly.
    assert (
        abs(
            int(parent.split()[-1].lstrip("~").rstrip("%"))
            - sum(int(v.lstrip("~").rstrip("%")) for v in tools.values())
        )
        <= 1
    )


def test_request_sequence_uses_window_max_not_share_of_total():
    """A tail does not partition the session; share-of-total would say nothing."""
    report = _populated()
    text = build_session_report(report, runtime(), 88).plain
    rows = _rows(text, "Last 12 requests")
    assert len(rows) == 12
    # Oldest first, so the chart reads left-to-right in time like the
    # transcript. The expected label is DERIVED in the ambient zone rather than
    # written out: the renderer formats local time, so a hardcoded stamp pins
    # the developer's timezone and fails on a UTC CI runner.
    oldest = min(request.ts_ms for request in report.recent)
    expected = datetime.fromtimestamp(oldest / 1000).astimezone().strftime("%H:%M:%S")
    assert rows[0].strip().startswith(expected)
    # The largest request fills the bar completely — the window-max rule.
    assert "·" not in rows[-1].split()[1]
    assert "·" in rows[0].split()[1]
    assert "%" not in text.split("▌ Last 12 requests")[1].split("▌")[0]


def _span_colours(text):
    """Every styled run of a report as ``(text, colour name)`` pairs.

    A span's style may be a Style OR a style-name string, so it is normalised
    before the colour is read.
    """
    out = []
    for span in text.spans:
        style = span.style if isinstance(span.style, Style) else Style.parse(str(span.style))
        out.append((text.plain[span.start : span.end], style.color.name if style.color else ""))
    return out


def test_partial_cost_plus_is_dimmed_as_a_flag_not_a_digit():
    """A lower-bound `+` at digit strength reads as part of the figure.

    The row engine routes money cells through `append_cost` precisely so this
    cannot regress into a second, hand-rolled cost cell.
    """
    aggregate = _agg(4, 4, 900, 100, cost=1000, known=2)
    report = SessionReport("sess", aggregate=aggregate, by_model={("p", "m"): aggregate})
    text = build_session_report(report, runtime(), 88, metric=METRIC_COST)
    colours = dict(_span_colours(text))
    assert colours["$0.0010"] == theme_mod.semantic_color("fg")
    assert colours["+"] == theme_mod.semantic_color("dim")


def test_failed_requests_are_warning_not_dim():
    """A failed request is the one thing on this screen a reader must not miss."""
    report = _populated()
    text = build_session_report(report, runtime(), 88)
    warning = theme_mod.semantic_color("warning")
    dim = theme_mod.semantic_color("dim")
    assert warning != dim
    styled = _span_colours(text)
    assert ("2 failed", warning) in styled
    assert ("14 req", dim) in styled
    # A healthy purpose carries no failure annotation at all.
    assert "aside" in text.plain and "0 failed" not in text.plain


def test_metric_toggle_switches_bars_and_says_so():
    report = _populated()
    tokens = build_session_report(report, runtime(), 88, metric=METRIC_TOKENS).plain
    cost = build_session_report(report, runtime(), 88, metric=METRIC_COST).plain
    assert "share of session tokens · t → cost" in tokens
    assert "share of session cost · t → tokens" in cost
    # Cost mode re-sorts by spend: the priciest model leads even though it is
    # not the biggest by tokens.
    assert _rows(tokens, "By model")[0].strip().startswith("anthropic/claude-sonnet-4-6")
    assert "$" in _rows(cost, "By model")[0]
    # SessionRequest holds no cost, so the sequence chart plots duration instead
    # of dividing the session total by the request count.
    assert "bars: tokens" in tokens and "bars: duration" in cost
    assert "ms" in _rows(cost, "Last 12 requests")[0]


def test_old_ledger_without_purpose_column_reads_as_one_unknown_row():
    """`col()` folds a missing column into `unknown`; `unknown` is not a failure."""
    aggregate = _agg(3, 3, 900, 100)
    report = SessionReport(
        "sess",
        aggregate=aggregate,
        by_model={("p", "m"): aggregate},
        by_purpose={"unknown": aggregate},
        by_purpose_outcome={("unknown", "unknown"): 3},
    )
    text = build_session_report(report, runtime(), 88).plain
    rows = _rows(text, "By purpose")
    assert len(rows) == 1 and rows[0].strip().startswith("unknown")
    # An outcome column that does not exist must not paint the session red.
    assert "failed" not in text
    # The legend contrasts `turn` with the harness's purposes; there is no
    # `turn` row here for it to contrast with.
    assert "turn = your requests" not in text


def test_fully_unpriced_cost_mode_explains_itself_instead_of_drawing_floor_bars():
    aggregate = _agg(3, 3, 900, 100, cost=0, known=0)
    report = SessionReport(
        "sess",
        aggregate=aggregate,
        by_model={("ollama", "llama3"): aggregate},
        by_purpose={"turn": aggregate},
    )
    text = build_session_report(report, runtime(), 88, metric=METRIC_COST).plain
    assert "no published price for these models · t → tokens" in text
    assert "$—" in text and "$0.00" not in text
    # No bar at all rather than a wall of identical one-cell floor marks.
    assert not any("█" in row for row in _rows(text, "By model"))
    assert "+ lower bound (some calls unpriced)" in text


def test_partial_pricing_marks_a_lower_bound_and_earns_the_legend():
    aggregate = _agg(4, 4, 900, 100, cost=1000, known=2)
    report = SessionReport("sess", aggregate=aggregate, by_model={("p", "m"): aggregate})
    text = build_session_report(report, runtime(), 88).plain
    assert "$0.0010+" in text
    assert "+ lower bound (some calls unpriced)" in text
    # A fully priced report needs no footnote for a mark nobody can see.
    priced = _agg(4, 4, 900, 100, cost=1000, known=4)
    clean = build_session_report(
        SessionReport("sess", aggregate=priced, by_model={("p", "m"): priced}), runtime(), 88
    ).plain
    assert "+ lower bound" not in clean


def _timing_rows(text):
    """The three fixed timing rows, whether or not they drew a track."""
    return _section(text, "Timings")[:3]


def test_zero_sample_timings_draw_no_bar():
    """A zero-width bar beside a real one implies a measured zero."""
    aggregate = _agg(3, 3, 900, 100)
    empty = {name: TimingSummary(0, None, None, None) for name in ("duration_ms", "ttft_ms")}
    report = SessionReport("sess", aggregate=aggregate, timings=empty)
    rows = _timing_rows(build_session_report(report, runtime(), 88).plain)
    assert all("unknown (0 samples)" in row for row in rows)
    assert not any("█" in row or "─" in row for row in rows)


def test_timings_share_one_millisecond_scale():
    """Independently scaled bars would draw prep and duration the same length."""
    text = build_session_report(_populated(), runtime(), 88).plain
    rows = _timing_rows(text)
    # Labels contain spaces, so locate the mean marker in the row itself.
    filled = [row.index("█") for row in rows]
    # Duration's mean sits far right of first-output's, which sits right of prep.
    assert filled[0] > filled[1] >= filled[2]
    # And the reader is told the axis is shared; an invisible shared scale is a trap.
    assert "shared ms scale" in text


def test_empty_components_say_so_but_keep_the_tool_caveat():
    aggregate = _agg(3, 3, 900, 100)
    text = build_session_report(SessionReport("sess", aggregate=aggregate), runtime(), 88).plain
    assert "no component data yet" in text
    assert "no tool context recorded yet" in text
    assert "Per-tool-name tokens and dollars are not recorded" in text


def test_zero_value_rows_and_healthy_path_noise_are_suppressed():
    text = build_session_report(_populated(), runtime(), 88).plain
    # A component with no tokens is a row that says nothing.
    assert "Environment" not in text and "Images (est.)" not in text
    # "0 requests missing usage · 0 unknown" is the absence of a problem.
    assert "missing usage" not in text
    assert "unavailable" not in text


def test_unattributed_residual_gets_no_bar():
    """It is a leftover, not a component; a bar would double-count it."""
    components = {"conversation": 400, "system_prompt": 100}
    aggregate = _agg(3, 3, 900, 100, components=components)
    text = build_session_report(SessionReport("sess", aggregate=aggregate), runtime(), 88).plain
    row = next(r for r in _section(text, "Where input went") if "Unattributed" in r)
    assert "█" not in row and "·" not in row
    assert "400" in text  # its peers still carry theirs
    assert row.split()[-1] == "400"  # 900 context − 500 attributed


def test_unattributed_residual_is_measured_with_its_peers():
    """It was appended AFTER measurement, so it was the one row off the grid.

    Review M2: with short component labels its own 28-char label ellipsised
    while the frame had ~30 cells free, and a residual wider than any measured
    value pushed its number off the shared column the screen exists to create.
    """
    # Short peer labels, so nothing else widens the label column for it.
    aggregate = _agg(6, 6, 280000, 5000, components={"conversation": 140000, "tool_results": 60000})
    text = build_session_report(SessionReport("sess", aggregate=aggregate), runtime(), 88).plain
    rows = _section(text, "Where input went")
    residual = next(r for r in rows if "Unattributed" in r)
    assert "…" not in residual, "label ellipsised while the frame had room"
    assert residual.rstrip().endswith("80k")

    # A residual WIDER than every measured value must still land in the shared
    # number column rather than pushing its own.
    wide = _agg(6, 6, 500900, 5000, components={"conversation": 400, "tool_results": 100})
    text = build_session_report(SessionReport("sess", aggregate=wide), runtime(), 88).plain
    rows = _section(text, "Where input went")
    residual = next(r for r in rows if "Unattributed" in r)
    peer = next(r for r in rows if "Conversation" in r)
    # Both values end in the same column: one number column, whole section.
    assert residual.index("500.4k") + len("500.4k") == peer.index("400") + len("400")


def test_full_share_row_does_not_crop_its_failure_count():
    """`format_percent(1.0)` is 4 chars; a 3-char cell cropped `2 failed`.

    Design D1: the overflowing cell pushed every 100% row one cell past the
    frame and `truncate(crop)` ate the rightmost column — the warning-coloured
    failure tally, rendering `2 faile`. Any single-model or single-purpose
    session is 100% by construction, so this was not an edge case.
    """
    one = _agg(18, 16, 460000, 14100, 49000, 18)
    report = SessionReport(
        "sess",
        aggregate=one,
        by_model={("anthropic", "claude-sonnet-4-6"): one},
        by_purpose={"turn": one},
        by_purpose_outcome={("turn", "ok"): 16, ("turn", "error"): 2},
    )
    # 84 is the real card width of a 100-column terminal, where this reproduced.
    for width in (84, 88, 100):
        text = build_session_report(report, runtime(), width).plain
        row = next(r for r in _rows(text, "By purpose") if "turn" in r)
        assert "100%" in row
        assert row.rstrip().endswith("2 failed"), f"cropped at {width}: {row!r}"
        assert len(row) <= width


@pytest.mark.parametrize("width,note,pct", [(88, True, True), (60, True, True), (50, False, False)])
def test_responsive_ladder_sheds_qualifiers_but_never_the_value(width, note, pct):
    """Below the ladder's steps the note and percentage shed; the number stays."""
    text = build_session_report(_populated(), runtime(), width).plain
    rows = _rows(text, "By model")
    assert len(rows) == 3
    for row in rows:
        assert len(row) <= width, f"row overflows {width}: {row!r}"
        # The bar never falls below the floor at which it stops meaning anything.
        assert sum(row.count(c) for c in "█·") >= 8
        assert row.split()[-1] if not note else True
    assert ("req" in "".join(rows)) is note
    assert ("%" in "".join(rows)) is pct
    # The value column never sheds: a bar without its number is a picture.
    assert all(any(ch.isdigit() for ch in row) for row in rows)


def test_narrow_labels_ellipsise_rather_than_starving_the_bar():
    text = build_session_report(_populated(), runtime(), 40).plain
    assert "…" in text
    for section in ("By model", "Where input went"):
        for row in _rows(text, section):
            assert len(row) <= 40
            assert sum(row.count(c) for c in "█·") >= 8


@pytest.mark.asyncio
async def test_metric_toggle_and_resize_repaint_through_the_real_screen():
    """`t` flips the bars in place and a resize re-does the column arithmetic."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = SessionScreen(_populated(), runtime())
        app.push_screen(screen)
        await pilot.pause()
        wide = str(screen._body.render())
        assert "share of session tokens · t → cost" in wide
        assert "t cost/tokens" in str(screen._hint.render())

        await pilot.press("t")
        await pilot.pause()
        flipped = str(screen._body.render())
        assert "share of session cost · t → tokens" in flipped
        assert "bars: duration" in flipped and "bars: tokens" not in flipped
        await pilot.press("t")
        await pilot.pause()
        assert str(screen._body.render()) == wide

        # The body is a function of the card width, so a resize must re-render
        # it, not merely refresh the scroll hint.
        await pilot.resize_terminal(50, 24)
        await pilot.pause()
        narrow = str(screen._body.render())
        assert narrow != wide
        # Every TABLE row fits the frame, so one row stays one record. The prose
        # footnotes are longer and are meant to wrap — they are prose.
        assert all(len(row) <= 50 for row in _rows(narrow, "By model"))
        assert all(len(row) <= 50 for row in _rows(narrow, "Where input went"))
        assert screen._scroll.max_scroll_x == 0


@pytest.mark.asyncio
async def test_toggle_is_not_advertised_when_there_is_nothing_to_plot():
    """Promising a key that does nothing is a dead control."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = SessionScreen(SessionReport("sess"), runtime())
        app.push_screen(screen)
        await pilot.pause()
        assert "t cost/tokens" not in str(screen._hint.render())
        before = str(screen._body.render())
        await pilot.press("t")
        await pilot.pause()
        assert str(screen._body.render()) == before


def test_unpriced_group_in_cost_mode_shows_no_share_not_a_zero_share():
    """QA Q1 / design D2: `cost_micro == 0` because the price is UNKNOWN.

    Dividing it by the session total yields a 0% that reads as a measured zero
    share, beside another row's real 100%, in the same column. QA's case: a
    local model taking 9 of 10 requests and 99% of the tokens rendered an empty
    track at 0% while a single priced call read as the entire spend.
    """
    priced = _agg(1, 1, 8000, 200, 3000, 1)
    unpriced = _agg(9, 9, 810000, 9000, 0, 0)
    total = _agg(10, 10, 818000, 9200, 3000, 1)
    report = SessionReport(
        "sess",
        aggregate=total,
        by_model={
            ("anthropic", "claude-sonnet-4-6"): priced,
            ("meta", "llama-4-scout"): unpriced,
        },
    )
    cost = build_session_report(report, runtime(), 88, metric=METRIC_COST).plain
    row = next(r for r in _section(cost, "By model") if "llama-4-scout" in r)
    assert "$—" in row, "the honest figure stays"
    assert "0%" not in row, "an unknown price is not a measured zero share"
    assert "█" not in row and "·" not in row, "no track for an uncomputed share"
    # The priced row is unaffected and still carries its real measurement.
    priced_row = next(r for r in _section(cost, "By model") if "claude" in r)
    assert "100%" in priced_row and "█" in priced_row
    # In TOKEN mode both rows are measured, so both keep their bars: the
    # suppression is about unknown COST, not about the model.
    tokens = build_session_report(report, runtime(), 88).plain
    assert all("█" in r for r in _rows(tokens, "By model"))
    assert "99%" in "".join(_rows(tokens, "By model"))


def test_sequence_chart_never_renders_an_absent_duration_as_zero():
    """Review M1: the store keeps `None` distinct from a value; so must the row.

    A legacy ledger has no `duration_ms`, and `or 0.0` printed a full-strength
    `0 ms` beside an empty track — `$0.00` for time. Same rule as `_timing_rows`
    ("an absent sample is not a fast one") and `_gauge_row`.
    """
    recent = tuple(
        SessionRequest(
            f"r{i}",
            1788602400000 + i * 10000,
            "p",
            "m",
            "unknown",
            "unknown",
            None,
            1000,
            100,
            None,
            None,
            None,
        )
        for i in range(3, -1, -1)
    )
    report = SessionReport("sess", aggregate=_agg(4, 4, 4000, 400), recent=recent)
    cost = build_session_report(report, runtime(), 88, metric=METRIC_COST).plain
    rows = _section(cost, "Last 4 requests")
    assert rows and all("unknown" in r for r in rows)
    assert not any("0 ms" in r for r in rows), "an absent sample rendered as measured zero"
    assert not any("█" in r for r in rows), "no bar for an unmeasured request"
    # A MIXED tail keeps the measured rows drawn and scales them among
    # themselves: an unmeasured request must not define the window max either.
    mixed = (
        SessionRequest(
            "b", 1788602400000 + 10000, "p", "m", "turn", "ok", True, 1000, 100, 4000.0, None, None
        ),
        SessionRequest(
            "a", 1788602400000, "p", "m", "turn", "ok", True, 1000, 100, None, None, None
        ),
    )
    report = SessionReport("sess", aggregate=_agg(2, 2, 2000, 200), recent=mixed)
    rows = _section(
        build_session_report(report, runtime(), 88, metric=METRIC_COST).plain, "Last 2 requests"
    )
    assert sum("█" in r for r in rows) == 1
    assert any("4,000 ms" in r for r in rows) and any("unknown" in r for r in rows)


def _tree_report(**kw):
    """A session that spent $31.28 itself and $71.06 through 20 subagents —
    the operator's real numbers, which is what makes the assertions readable."""
    return SessionReport(
        "sess",
        aggregate=UsageAggregate(
            calls=248,
            ok_calls=248,
            context_tokens=47_000_000,
            cost_micro=31_276_032,
            cost_known_calls=248,
        ),
        descendants_aggregate=UsageAggregate(
            calls=896,
            ok_calls=896,
            context_tokens=200_000_000,
            cost_micro=71_059_322,
            cost_known_calls=896,
        ),
        descendant_ids=tuple(f"kid{n:02d}" for n in range(20)),
        **kw,
    )


def test_est_cost_headline_is_the_tree_with_the_split_beneath_it():
    """The defect: /session showed $31.28 while the composer showed $102.60 for
    the same session. The headline becomes the tree; the split explains it."""
    text = build_session_report(_tree_report(), runtime(), width=120).plain
    assert "$102.34" in text  # the headline: what this session cost
    assert "$31.28 own · $71.06 subagents" in text  # the split, in the note slot
    # The asymmetry with every other section is LABELLED, not left to be found.
    assert "cost incl. subagents" in text
    assert "Includes 20 subagent sessions" in text
    assert "other sections are this session only" in text


def test_scope_is_never_silently_dropped_at_any_width():
    """Design D1 / QA Q1: the scope is load-bearing, not sheddable chrome.

    Before this, the split note needed a 66-cell card but ``_NOTE_MIN`` admitted
    it from 60, so 75-80 column terminals (including the canonical 80) painted
    "$71.06 subagent" mid-word, and below 75 the note AND the header's scope
    label both vanished — leaving a bare $102.34 where main showed $31.28 with
    nothing on screen saying the scope had changed. That is the reported defect's
    own shape (a right number that looks wrong) relocated to narrow terminals.

    So the requirement is not "the note fits" but "no width shows a tree figure
    without saying it is one", checked across the ladder rather than at one width.
    """
    report = _tree_report()
    for width in range(50, 161):
        lines = build_session_report(report, runtime(), width=width).plain.split("\n")
        cost_row = next(line for line in lines if "Est. cost" in line)
        flat = flowed("\n".join(lines))

        # 1. The headline is always the tree figure.
        assert "$102.34" in cost_row, width
        # 2. The scope is stated SOMEWHERE on screen at every width.
        assert ("subagent" in flat) or ("subs" in flat), width
        # 3. Nothing is cropped mid-word. Every rung of the ladder ends in a
        #    COMPLETE word, so a truncated "subagent"/"sub" tail can only mean a
        #    crop — which is the Q1 bisection, generalised to the whole ladder.
        tail = cost_row.rstrip()
        if "subagent" in tail:
            assert tail.endswith("subagents"), (width, cost_row)
        # 4. The row never overflows the frame it was built for.
        assert len(cost_row) <= width, (width, len(cost_row))


def test_the_scope_note_does_not_orphan_wrap_into_the_block():
    """Design D2: a wrapped footnote must stay indented under its block.

    The 103-character prose note wrapped at every width from 70 to ~128 and,
    because ``fold`` applies the body indent to the first line only, dropped
    "this session alone." at column 0 between the Est. cost figure and the next
    section header — where it read as an orphan row of the table. ``_Body.kv``'s
    own comment already names this failure mode.
    """
    report = _tree_report()
    for width in (70, 76, 80, 100, 128, 160):
        lines = build_session_report(report, runtime(), width=width).plain.split("\n")
        # Every non-blank body line is indented. Section headers legitimately
        # start at column 0 (the ▌ accent bar IS the left edge), so they are
        # the one exception; anything else at column 0 is an orphaned wrap.
        for line in lines[2:]:
            if line.strip() and not line.startswith("▌"):
                assert line.startswith("  "), (width, repr(line))
        # And the sentence survives the wrap intact, wherever it broke.
        assert "other sections are this session only" in flowed("\n".join(lines)), width


def test_own_only_sections_say_they_are_own_only():
    """by_model/by_purpose stay this session's calls — they answer "where did MY
    context go" — so their meta must say so once the headline is a tree."""
    report = _tree_report(
        by_model={("anthropic", "claude"): UsageAggregate(calls=248, context_tokens=47_000_000)},
        by_purpose={"turn": UsageAggregate(calls=248, context_tokens=47_000_000)},
    )
    text = build_session_report(report, runtime(), width=120).plain
    assert "share of this session only" in text

    # A childless session has one scope, so the longer wording would be noise.
    solo = SessionReport(
        "sess",
        aggregate=UsageAggregate(calls=2, context_tokens=100, cost_micro=5, cost_known_calls=2),
        descendants_aggregate=UsageAggregate(),
        by_model={("anthropic", "claude"): UsageAggregate(calls=2, context_tokens=100)},
    )
    solo_text = build_session_report(solo, runtime(), width=120).plain
    assert "share of session" in solo_text
    assert "share of this session only" not in solo_text
    assert "own ·" not in solo_text


def test_tree_total_carries_the_lower_bound_mark_from_a_child():
    """Sum the counters, not the booleans: a fully-priced parent whose child
    used an unpriced model has a tree total that IS a lower bound — and the
    legend explaining the ``+`` must not be suppressed."""
    report = SessionReport(
        "sess",
        aggregate=UsageAggregate(
            calls=2, context_tokens=10, cost_micro=1_000_000, cost_known_calls=2
        ),
        descendants_aggregate=UsageAggregate(
            calls=4, context_tokens=10, cost_micro=500_000, cost_known_calls=1
        ),
        descendant_ids=("kid",),
    )
    text = build_session_report(report, runtime(), width=120).plain
    assert "$1.50+" in text
    assert "lower bound" in text  # COST_LEGEND, drawn from the subtree scope


def test_restored_floor_is_reconciled_in_prose_not_copied():
    """``≥`` (restored transcript floor) and ``+`` (unpriced calls) are different
    deficits. /session says which figure is which instead of merging them."""
    text = flowed(
        build_session_report(
            _tree_report(), replace(runtime(), spend_is_floor=True), width=120
        ).plain
    )
    assert "restored floor" in text
    assert "what the ledger actually retained" in text
    # The mark itself is NOT copied onto the ledger figure.
    assert "≥$" not in text

    plain = build_session_report(_tree_report(), runtime(), width=120).plain
    assert "restored floor" not in plain
