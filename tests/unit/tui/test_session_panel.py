"""Current-session diagnostics: real slash dispatch, read-only and stale-safe."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from dataclasses import replace

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
        text = build_session_report(report, runtime()).plain
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
    text = build_session_report(_populated(), runtime(), 88).plain
    rows = _rows(text, "Last 12 requests")
    assert len(rows) == 12
    # Oldest first, so the chart reads left-to-right in time like the transcript.
    assert rows[0].strip().startswith("06:00:00")
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
