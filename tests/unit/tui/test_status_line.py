"""Status band — segment formats, the one-row guarantee, and overflow order.

The band carries nine things the owner asked to see on a terminal that fits
maybe six of them, so two properties are load-bearing and both are asserted
here rather than eyeballed:

- **It is exactly one row at every width.** A band that wraps pushes the
  editor off screen.
- **It sheds segments in a decided order.** Truncation would clip whichever
  segment happened to be last; the ladder drops whole segments, cheapest
  loss first, and these tests pin WHICH one goes at each step.
"""

from __future__ import annotations

import math
import re
from typing import Any, NotRequired, TypedDict, cast

from rich.cells import cell_len
from rich.style import Style
from textual.widgets import Static

from local_operator.compaction.thresholds import CompactionSettings, should_compact
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.status_line import (
    _DROP_LADDER,
    _DROP_LADDER_ESTIMATE,
    _DROP_LADDER_QUIET,
    _DROP_LADDER_QUIET_ESTIMATE,
    _MIN_GROUP_GAP,
    _SEP_RIGHT,
    _SPINNER_FRAMES,
    _UNBOUNDED_RUNGS,
    AGENT_PROFILE_CELLS,
    ICON_AGENT_PROFILE,
    ICON_AGENTS,
    ICON_APPROVALS,
    ICON_CONTEXT,
    ICON_COST,
    ICON_CWD,
    ICON_DURATION,
    ICON_JOBS,
    ICON_MCP,
    ICON_MODEL,
    ICON_TEAM,
    NAME_CELLS,
    NAME_CELLS_FLOOR,
    McpStatus,
    StatusLine,
    context_semantic_color,
    drop_ladder,
    format_agents,
    format_context_usage,
    format_cost,
    format_cwd,
    format_duration,
    format_mcp,
    format_model_label,
    format_window,
    mcp_semantic,
    truncate_name,
)


class FakeDock:
    """The three things StatusLine asks of its widget (width, paint, timer).

    A real ``Static`` needs a running app before ``set_interval`` works, and
    these tests drive the streaming transitions that arm the spinner.
    """

    def __init__(self, width: int = 80) -> None:
        self.width = width
        self.painted = None
        #: The ``layout`` flag of the last paint. The band must never ask for
        #: a layout pass — its box is fixed by the sheet and the default would
        #: reflow the whole screen on every spinner frame.
        self.layout: bool = True
        self.intervals: list[tuple[float, object]] = []

    @property
    def size(self):  # noqa: ANN201 - mirrors textual's geometry duck type
        return type("Size", (), {"width": self.width})()

    def update(self, content=None, *, layout: bool = True) -> None:  # noqa: ANN001
        """Mirrors ``Static.update`` parameter for parameter.

        Not ``**kwargs``: a double that swallows anything stops standing for
        the thing it is named after, which is how this one missed ``layout``.
        """
        self.painted = content
        self.layout = layout

    def set_interval(self, interval: float, callback):  # noqa: ANN001, ANN201
        self.intervals.append((interval, callback))
        return type("Timer", (), {"stop": lambda self: None})()


def _dock(width: int = 80) -> Static:
    """FakeDock where StatusLine's declared ``Static`` is asked for."""
    return cast(Static, FakeDock(width))


class FakeClock:
    """A monotonic clock the test advances by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _full_band(width: int = 200) -> tuple[StatusLine, FakeClock]:
    """A band with EVERY segment populated — the worst case for one-row."""
    clock = FakeClock()
    status = StatusLine(_dock(width), clock=clock)
    status.update(
        model_label="openrouter/moonshotai/kimi-k2-thinking",
        effort="high",
        cwd="/Users/tester/work/local-operator",
        context_tokens=496_000,
        context_window=1_000_000,
        subagents=3,
        cost="$12.40",
        conversation_name="Status band enrichment",
    )
    # U2 note: the active-agent/team segments are DELIBERATELY not populated in
    # this shared fixture. They widen the worst case, and the many geometric
    # sweep tests below (name-flex, eviction floors) are calibrated to the
    # pre-U2 segment set; recalibrating all of them to carry two always-on
    # segments would be a broad, error-prone change to tests that are not about
    # U2. The identity segments are instead driven explicitly where they are
    # under test: the ladder-ORDER test sets them so their shed position is
    # pinned, and the dedicated U2 tests cover their rendering, cap, and live
    # attach/detach.
    status._active_seconds = 2461.0  # 41m1s
    return status, clock


# -- segment formats ---------------------------------------------------------


def test_context_usage_is_percent_over_abbreviated_window() -> None:
    assert format_context_usage(496_000, 1_000_000) == "49.6%/1M"
    assert format_context_usage(100_000, 200_000) == "50.0%/200k"
    assert format_context_usage(1_000, 128_000) == "0.8%/128k"


def test_context_usage_without_a_window_reports_raw_tokens() -> None:
    """No denominator means no percentage.

    The unknown reads as ``—``, the same glyph unknown cost (``$—``) and a
    credential-less provider already use, so one row never spells "unknown"
    two different ways.
    """
    assert format_context_usage(12_400, 0) == "12.4k/—"
    assert format_context_usage(900, -1) == "900/—"


def test_context_usage_is_empty_before_any_tokens_are_spent() -> None:
    assert format_context_usage(0, 1_000_000) == ""


def test_window_abbreviation_drops_a_trailing_zero_decimal() -> None:
    assert format_window(1_000_000) == "1M"
    assert format_window(1_500_000) == "1.5M"
    assert format_window(200_000) == "200k"
    assert format_window(128_000) == "128k"
    assert format_window(512) == "512"


def test_duration_drops_units_that_stop_carrying_information() -> None:
    assert format_duration(0) == "0s"
    assert format_duration(9) == "9s"
    assert format_duration(59.9) == "59s"
    assert format_duration(60) == "1m"
    assert format_duration(2461) == "41m1s"
    assert format_duration(3600) == "1h"
    assert format_duration(3725) == "1h2m"


def test_duration_switches_to_days_and_stays_six_cells_wide() -> None:
    """Callers RESERVE cells for this string, so its width is part of its
    contract — ``WorkingBlock._CLOCK_COL`` sizes a row against it rather than
    measuring what comes back.

    It used to end at ``{h}h{m}m`` with an unbounded hours field, so ``100h30m``
    was seven cells and pushed that row over the terminal (review round 15).
    A days branch is also simply the better reading at that magnitude, and the
    ``100d+`` cap is what makes the bound hold by construction instead of holding
    until someone finds a bigger number.
    """
    assert format_duration(86_400) == "1d"
    assert format_duration(86_400 + 3600 * 5) == "1d5h"
    assert format_duration(362_400) == "4d4h"
    assert format_duration(86_400 * 99 + 3600 * 23) == "99d23h"
    assert format_duration(86_400 * 100) == "100d+"
    assert format_duration(86_400 * 100_000) == "100d+"

    # The whole contract in one line: no input produces a seventh cell. Stepped
    # by a prime so the sweep lands off every round boundary, and carried past
    # 100 days so it spans EVERY branch — an earlier version stopped at 400_000
    # seconds, which is 4.6 days, and so never reached the day format it was
    # written to defend, let alone the cap.
    assert max(cell_len(format_duration(s)) for s in range(0, 8_700_000, 97)) == 6
    # Negative and fractional inputs are not special-cased anywhere; they must
    # not be able to slip a wider string past the bound either.
    assert cell_len(format_duration(-1)) <= 6
    assert cell_len(format_duration(59.9)) <= 6


def test_agents_segment_is_empty_at_zero_and_pluralises() -> None:
    assert format_agents(0) == ""
    assert format_agents(-1) == ""
    assert format_agents(1) == "1 agent"
    assert format_agents(4) == "4 agents"


def test_cwd_is_home_relative_then_basename() -> None:
    from pathlib import Path

    inside = str(Path.home() / "work" / "local-operator")
    assert format_cwd(inside, short=False) == "~/work/local-operator"
    assert format_cwd(inside, short=True) == "local-operator"
    # Outside the home tree the absolute path is already the short form.
    assert format_cwd("/opt/thing", short=False) == "/opt/thing"
    assert format_cwd("/opt/thing", short=True) == "thing"


def test_an_unnamed_model_sheds_only_its_provider_prefixes() -> None:
    """The behaviour every model had before display names, and the one an
    aggregator id nobody has curated still gets: no name exists, so the selector
    is the honest rendering and ``short`` keeps its last path segment."""
    assert format_model_label("openrouter/moonshotai/kimi-k2", short=False) == (
        "openrouter/moonshotai/kimi-k2"
    )
    assert format_model_label("openrouter/moonshotai/kimi-k2", short=True) == "kimi-k2"
    assert format_model_label("ollama", short=True) == "ollama"


def test_a_named_model_renders_its_name_instead_of_its_selector() -> None:
    """The headline of the change, measured: 23 cells of selector become 13 of
    name, which is what makes room for the effort segment beside it."""
    assert format_model_label("anthropic/claude-opus-5", short=False) == "Claude Opus 5"
    assert cell_len("Claude Opus 5") < cell_len("anthropic/claude-opus-5")


def test_a_listing_name_reaches_the_band_for_a_model_no_registry_row_covers() -> None:
    """``name`` is the route by which a model the registry has not been taught
    about is named at all — the registry provably lags a direct provider's
    releases, and that is the case ``ModelSpec.display_name`` exists for."""
    assert (
        format_model_label("anthropic/claude-opus-6", short=False, name="Claude Opus 6")
        == "Claude Opus 6"
    )


def test_the_unknown_placeholder_does_not_reach_the_band() -> None:
    """The word the shared fallback is named, not a model. A nameless listing
    used to keep it and paint it on the band for every unshipped id."""
    assert format_model_label("xai/grok-4.20", short=False, name="Unknown") == "xai/grok-4.20"
    assert format_model_label("xai/grok-4.20", short=True, name="Unknown") == "grok-4.20"


def test_a_resold_model_keeps_its_selector_however_the_reseller_names_it() -> None:
    """Both shipped aggregators list the same models under the same names — 398 of
    ~400 in their real cached catalogues — so a reseller's name cannot say which
    route is answering, and the route is what differs in price and quota."""
    for provider in ("openrouter", "radient"):
        selector = f"{provider}/moonshotai/kimi-k2"
        assert (
            format_model_label(selector, short=False, name="MoonshotAI: Kimi K2 0711") == selector
        )


def test_the_band_refuses_a_name_another_model_already_answers_to() -> None:
    """A band printing a shared name could not say which model was replying, so
    the segment stays wide instead. Constructed rather than taken from the
    registry: the one shipped duplicate was a data defect and has been fixed, and
    a live listing renaming a model onto a sibling's name is how this reaches a
    user now."""
    borrowed = "Claude Opus 5"  # owned by anthropic/claude-opus-5
    assert format_model_label("openai/some-proxy-id", short=False, name=borrowed) == (
        "openai/some-proxy-id"
    )


def test_the_bands_name_is_the_string_the_picker_offered() -> None:
    """One model, one name.

    Crosses the real boundary rather than comparing the band to itself: the
    picker paints ``CatalogueEntry.label``, so the assertion reads that label off
    ``ProviderController.static_catalogue()`` and compares it to what this segment
    renders. Computing both sides from ``naming.model_label`` would have stayed
    green if either controller call site were changed to ``.compact``, which is
    precisely the disagreement the test is named for.
    """
    from local_operator.providers.controller import ProviderController

    # `static_catalogue` is the one method that reads no credentials, so the
    # store is genuinely unused here; cast rather than build a stub whose
    # only job would be to never be called.
    entries = ProviderController(auth_store=cast(Any, None)).static_catalogue()
    checked = 0
    for entry in entries:
        if entry.provider != "anthropic":
            continue
        band = format_model_label(entry.selector, short=False, name=entry.label)
        assert band == entry.label, f"{entry.selector}: band {band!r} != picker {entry.label!r}"
        checked += 1
    assert checked, "no anthropic rows in the static catalogue"


def test_cost_keeps_its_precision_ladder() -> None:
    assert format_cost(0.0021) == "$0.0021"
    assert format_cost(0.125) == "$0.125"
    assert format_cost(12.4) == "$12.40"


# -- the one-row guarantee ---------------------------------------------------


def test_band_is_one_row_at_every_width_with_all_segments_populated() -> None:
    """The regression this whole slice risks: nine segments wrapping to two
    rows on a narrow terminal, which pushes the editor off screen."""
    status, _clock = _full_band()
    for width in (20, 40, 60, 80, 120, 200):
        row = status.render_text(width)
        assert "\n" not in row.plain, width
        assert cell_len(row.plain) <= width, (width, row.plain)


def test_band_is_one_row_while_streaming_too() -> None:
    """The spinner is appended after the ladder has already fitted the row."""
    status, _clock = _full_band()
    status._streaming = True
    for width in (20, 40, 60, 80, 120, 200):
        row = status.render_text(width)
        assert "\n" not in row.plain, width
        assert cell_len(row.plain) <= width, (width, row.plain)


def test_the_model_segment_survives_a_terminal_far_too_narrow() -> None:
    """At any width the band still names the model, led by its icon.

    There is no brand glyph any more: `π` is omp's mark, not local-operator's,
    and a logo is the one thing on a status band that never tells the operator
    anything. The model icon leads instead, which is information.
    """
    status, _clock = _full_band()
    row = status.render_text(10)
    assert row.plain.startswith(ICON_MODEL)
    assert cell_len(row.plain) <= 10


# -- overflow priority -------------------------------------------------------


def test_the_session_name_outlives_the_counters_it_used_to_precede(monkeypatch) -> None:
    """The name is now the band's TRAILING segment, and it earns its place.

    It used to be the first rung on the ladder, on the reasoning that "the name
    is a label the user typed and already knows". Both halves of that stopped
    being true: the name is generated (or excerpted from the opening prompt),
    and it is the field an operator reads this corner for when four sessions are
    tiled — which is exactly when the terminal is narrow. So it now outlives the
    re-derivable duration and the two counters, and sheds only once nothing but
    live figures is left.
    """
    monkeypatch.setenv("HOME", "/Users/tester")
    status, _clock = _full_band()
    assert "Status band enrichment" in status.render_text(200).plain

    # Walk down to the first width that sheds it rather than computing it from
    # the 200-cell row. That row is PADDED to the full width (the right group is
    # edge-aligned), so its length is 200 whatever the content measures and
    # `len(row) - 1` says nothing about when the band overflows. The stub is
    # what is probed for: the ladder shortens the name before it drops it.
    for width in range(200, 4, -1):
        tight = status.render_text(width).plain
        if "Status band enric" not in tight:
            break
    else:  # pragma: no cover - the band always sheds something by width 5
        raise AssertionError("no width shed the conversation name")

    # The counters and the duration are already gone by the time the name is:
    # they are cheap to lose, and this band has stopped shedding the one field
    # that says which conversation it belongs to in order to keep them.
    assert "3 agents" not in tight
    assert "41m1s" not in tight
    # The live figures an operator acts on do outlive it, which is the half of
    # the old contract that was right.
    assert "49.6%/1M" in tight
    assert "$12.40" in tight

    # And there is a real band of widths where the name survives and the
    # counters do not — the whole point of the reorder, not just its rung order.
    kept = [
        width
        for width in range(60, 201)
        if "Status band enric" in status.render_text(width).plain
        and "3 agents" not in status.render_text(width).plain
    ]
    assert kept, "the name never outlived the subagent counter"


def test_segments_disappear_in_the_declared_ladder_order(monkeypatch) -> None:
    """Walk the width down and record the order segments leave the band.

    Asserted as a SEQUENCE rather than per-width thresholds: the thresholds
    move whenever a label's length changes, but the order is the contract.

    HOME is pinned because the cwd segment abbreviates through
    ``Path.home()``: under the suite's HOME isolation the fixture's
    ``/Users/tester/...`` path is outside the home tree and renders absolute,
    so the ``~/work/local-operator`` probe below would never match and the
    segment would look like it had been dropped at the very first width.
    """
    monkeypatch.setenv("HOME", "/Users/tester")
    status, _clock = _full_band()
    # U2: drive the two static-identity segments explicitly here (they are kept
    # out of the shared fixture — see `_full_band`) so their shed order is
    # pinned against the rest of the ladder.
    status.update(team="release-team", agent_profile="auditor")
    probes = {
        "subagents": "3 agents",
        "duration": "41m1s",
        "name-full": "Status band enrichment",
        "name-short": "Status band enric",
        "cost": "$12.40",
        "context": "49.6%/1M",
        "effort": "high",
        # U2: the two static-identity segments. Their names are chosen not to
        # collide with any other probe's needle ("auditor"/"release-team" appear
        # nowhere else on the band).
        "team": "release-team",
        "agent": "auditor",
        "cwd-full": "~/work/local-operator",
        "cwd-short": "local-operator",
        "model-full": "openrouter/moonshotai/kimi-k2-thinking",
        "model-short": "kimi-k2-thinking",
    }
    # Nothing may be missing before the walk starts, or its absence would be
    # recorded as a drop at width 200 and silently head the observed order.
    widest = status.render_text(200).plain
    assert [key for key, needle in probes.items() if needle not in widest] == []

    present = {key: True for key in probes}
    order: list[str] = []
    for width in range(200, 4, -1):
        plain = status.render_text(width).plain
        for key, needle in probes.items():
            if present[key] and needle not in plain:
                present[key] = False
                order.append(key)

    assert order == [
        "duration",  # re-derivable from the transcript
        "subagents",  # a counter, but not re-derivable without scrolling
        # Shortened to a stub, not dropped: the widest rung, so the cut recovers
        # more than any drop below it, and a stub still names the session.
        "name-full",
        "cwd-full",  # shortened to its basename, not dropped
        "model-full",  # shortened to the bare model id, not dropped
        "effort",  # a static setting: it does not change while they watch
        # U2: the two static-identity settings shed AFTER effort (kept longer)
        # because each names a persona/roster the user just deliberately
        # attached — higher-information than the effort word. team before agent:
        # the roster is the broader context and the profile is "who is replying
        # here", so the profile is the last static setting to go.
        "team",
        "agent",
        # Only now does the name go — one rung later than it used to sit in
        # ENTIRETY, and after everything above it has already been spent.
        "name-short",
        "cost",
        # The shortened cwd goes BEFORE the context number. By this rung it is a
        # basename — ~7 cells of "where am I" against ~9 cells of "how close is
        # compaction" — and the second is what predicts the operator's next move.
        "cwd-short",
        "context",  # the last number standing beside the model
        # The model label is never DROPPED; it survives to width 17, where the
        # irreducible-row path truncates it to `kimi-k2-t…` rather than leaving
        # a bare glyph on an empty strip.
        "model-short",
    ]


def test_shortening_keeps_the_segment_rather_than_dropping_it() -> None:
    """The two shorten steps are why cost and context survive as long as they
    do — assert the shortened forms actually appear, not just that the long
    forms vanished."""
    status, _clock = _full_band()
    row = status.render_text(60).plain
    assert "local-operator" in row
    assert "~/work/local-operator" not in row
    assert "kimi-k2-thinking" in row
    assert "openrouter/moonshotai/" not in row


# -- U2: active agent / team segments ----------------------------------------


def test_the_active_agent_and_team_segments_paint_with_their_glyphs() -> None:
    """Both static-identity segments render `icon name` on a wide band."""
    status, _clock = _full_band()
    status.update(team="release-team", agent_profile="auditor")
    row = status.render_text(200).plain
    assert f"{ICON_TEAM} release-team" in row
    assert f"{ICON_AGENT_PROFILE} auditor" in row


def test_the_identity_segments_are_absent_until_attached() -> None:
    """An unattached session shows neither segment, so its fit is unchanged.

    The two rungs are inert when their setting is unset: they occupy no cells,
    which is what keeps a session that never ran /agent or /team rendering
    exactly as it did before U2.
    """
    clock = FakeClock()
    status = StatusLine(_dock(200), clock=clock)
    status.update(model_label="test/model", cwd="/tmp", conversation_name="x")
    row = status.render_text(200).plain
    assert ICON_TEAM not in row
    assert ICON_AGENT_PROFILE not in row


def test_the_identity_segments_appear_and_clear_live() -> None:
    """attach -> re-attach -> detach flows through update() the way the app
    drives it: "" clears the segment, a new name replaces it."""
    clock = FakeClock()
    status = StatusLine(_dock(200), clock=clock)
    status.update(model_label="test/model", cwd="/tmp", conversation_name="x")

    status.update(agent_profile="auditor", team="release-team")
    row = status.render_text(200).plain
    assert f"{ICON_AGENT_PROFILE} auditor" in row
    assert f"{ICON_TEAM} release-team" in row

    # Re-attach a different profile: the segment changes, it does not stack.
    status.update(agent_profile="reviewer")
    row = status.render_text(200).plain
    assert "reviewer" in row
    assert "auditor" not in row

    # Detach both with the "" the app pushes on /agent clear and team detach.
    status.update(agent_profile="", team="")
    row = status.render_text(200).plain
    assert ICON_AGENT_PROFILE not in row
    assert ICON_TEAM not in row


def test_a_long_identity_name_is_capped_not_unbounded() -> None:
    """The name is a BOUNDED rung: a pathological profile name is truncated to
    :data:`AGENT_PROFILE_CELLS` rather than pushing the model label off the row.
    """
    clock = FakeClock()
    status = StatusLine(_dock(200), clock=clock)
    long_name = "x" * 80
    status.update(
        model_label="test/model",
        cwd="/tmp",
        conversation_name="x",
        agent_profile=long_name,
    )
    row = status.render_text(200).plain
    # The full 80-char name never reaches the band; the cap bounds the ink.
    assert long_name not in row
    assert "x" * AGENT_PROFILE_CELLS not in row  # truncation eats an ellipsis cell
    assert ICON_AGENT_PROFILE in row


def test_the_identity_rungs_are_bounded_and_present_in_every_ladder() -> None:
    """team/agent are on every ladder variant and neither is an unbounded rung.

    Being bounded is what lets them sit ahead of the numbers without risking the
    self-defeating tail cwd is barred from: a capped name always fits its floor.
    """
    for ladder in (
        _DROP_LADDER,
        _DROP_LADDER_QUIET,
        _DROP_LADDER_ESTIMATE,
        _DROP_LADDER_QUIET_ESTIMATE,
    ):
        assert "team" in ladder
        assert "agent" in ladder
        assert "team" not in _UNBOUNDED_RUNGS
        assert "agent" not in _UNBOUNDED_RUNGS
        # team sheds before agent (broader context goes first), and both shed
        # before the numbers an operator acts on.
        assert ladder.index("team") < ladder.index("agent")
        assert ladder.index("agent") < ladder.index("context")
        assert ladder.index("effort") < ladder.index("team")


# -- update() contract -------------------------------------------------------


def test_none_leaves_a_segment_unchanged() -> None:
    status, _clock = _full_band()
    status.update(cost=None, conversation_name=None, subagents=None)
    row = status.render_text(200).plain
    assert "$12.40" in row
    assert "Status band enrichment" in row
    assert "3 agents" in row


def test_zero_subagents_hides_the_segment_rather_than_showing_a_zero() -> None:
    status, _clock = _full_band()
    status.update(subagents=0)
    assert "agent" not in status.render_text(200).plain


def test_an_unknown_cost_renders_as_the_explicit_dash() -> None:
    """D20: the turn billed tokens but pricing is unknown. Absence would read
    as "free", which is the wrong lie."""
    status, _clock = _full_band()
    status.update(cost="$—")
    assert "$—" in status.render_text(200).plain


def test_an_empty_name_clears_the_segment() -> None:
    status, _clock = _full_band()
    status.update(conversation_name="")
    assert "Status band enrichment" not in status.render_text(200).plain


# -- duration is ACTIVE time -------------------------------------------------


def test_duration_accumulates_only_while_streaming() -> None:
    clock = FakeClock()
    status = StatusLine(_dock(200), clock=clock)
    status.update(model_label="p/m")

    status.update(streaming=True)
    clock.advance(30)
    status.update(streaming=False)
    clock.advance(3600)  # idle: a session left open has not been working
    status.update(streaming=True)
    clock.advance(45)
    status.update(streaming=False)

    assert "1m15s" in status.render_text(200).plain


def test_duration_ticks_live_during_a_turn() -> None:
    clock = FakeClock()
    status = StatusLine(_dock(200), clock=clock)
    status.update(streaming=True)
    clock.advance(9)
    assert "9s" in status.render_text(200).plain


def test_a_redundant_streaming_false_cannot_bank_the_same_interval_twice() -> None:
    """The prompt worker's ``finally`` repeats ``streaming=False`` after
    agent_end already sent it; double-banking would inflate every duration."""
    clock = FakeClock()
    status = StatusLine(_dock(200), clock=clock)
    status.update(streaming=True)
    clock.advance(10)
    status.update(streaming=False)
    clock.advance(10)
    status.update(streaming=False)
    assert "10s" in status.render_text(200).plain


def test_no_duration_segment_before_anything_has_run() -> None:
    clock = FakeClock()
    status = StatusLine(_dock(200), clock=clock)
    status.update(model_label="p/m", cwd="/tmp")
    assert "0s" not in status.render_text(200).plain


# -- accent budget and group seam --------------------------------------------


def _fills(row) -> dict[str, str]:
    """``{segment text: hex fill}`` for every styled span in a rendered row."""
    out: dict[str, str] = {}
    for span in row.spans:
        color = span.style.color if hasattr(span.style, "color") else None
        if color is not None and color.triplet is not None:
            out[row.plain[span.start : span.end]] = color.triplet.hex.lower()
    return out


def _to_lab(hex_color: str) -> tuple[float, float, float]:
    """sRGB hex → CIE L*a*b* (D65), the input CIEDE2000 needs."""
    value = hex_color.lstrip("#")
    rgb = [int(value[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    r, g, b = linear
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 216 / 24389 else (841 / 108) * t + 4 / 29

    fx, fy, fz = f(x), f(y), f(z)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def _delta_e_2000(first: str, second: str) -> float:
    """CIEDE2000 between two hex colours.

    The metric this codebase states its colour decisions in ("dE 3.06 is
    imperceptible", "the healthy lamp was 5.08 from the accent"), computed here
    so a colour assertion can defend the PERCEPTUAL gap rather than a palette
    name — a third green added to the ramp would slip past a name check.
    """
    l1, a1, b1 = _to_lab(first)
    l2, a2, b2 = _to_lab(second)
    c1 = math.hypot(a1, b1)
    c2 = math.hypot(a2, b2)
    c_bar = (c1 + c2) / 2
    g = 0.5 * (1 - math.sqrt(c_bar**7 / (c_bar**7 + 25**7))) if c_bar else 0.0
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if (a1p or b1) else 0.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if (a2p or b2) else 0.0

    d_lp = l2 - l1
    d_cp = c2p - c1p
    if c1p * c2p == 0:
        d_hp = 0.0
    elif abs(h2p - h1p) <= 180:
        d_hp = h2p - h1p
    else:
        d_hp = h2p - h1p - 360 if h2p > h1p else h2p - h1p + 360
    d_hp = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(d_hp) / 2)

    l_bar = (l1 + l2) / 2
    c_barp = (c1p + c2p) / 2
    if c1p * c2p == 0:
        h_barp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        h_barp = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        h_barp = (h1p + h2p + 360) / 2
    else:
        h_barp = (h1p + h2p - 360) / 2

    t = (
        1
        - 0.17 * math.cos(math.radians(h_barp - 30))
        + 0.24 * math.cos(math.radians(2 * h_barp))
        + 0.32 * math.cos(math.radians(3 * h_barp + 6))
        - 0.20 * math.cos(math.radians(4 * h_barp - 63))
    )
    s_l = 1 + (0.015 * (l_bar - 50) ** 2) / math.sqrt(20 + (l_bar - 50) ** 2)
    s_c = 1 + 0.045 * c_barp
    s_h = 1 + 0.015 * c_barp * t
    r_t = (
        -2
        * math.sqrt(c_barp**7 / (c_barp**7 + 25**7))
        * math.sin(math.radians(60 * math.exp(-(((h_barp - 275) / 25) ** 2))))
    )
    return math.sqrt(
        (d_lp / s_l) ** 2
        + (d_cp / s_c) ** 2
        + (d_hp / s_h) ** 2
        + r_t * (d_cp / s_c) * (d_hp / s_h)
    )


def test_the_accent_marks_a_live_turn_not_the_brand_glyph() -> None:
    """The accent budget's whole point is that seeing green MEANS something.

    Painting the always-on brand glyph accent and the streaming spinner dim made
    the band render identically whether the agent was working or idle — the one
    row an operator glances at for liveness — while the tool card and the working
    line both put their running signal in accent.
    """
    accent = theme_mod.semantic_color("accent").lower()

    status, _clock = _full_band()
    idle = _fills(status.render_text(200))
    assert accent not in idle.values(), f"idle band must spend no accent: {idle}"

    status.update(streaming=True)
    live = _fills(status.render_text(200))
    greens = [text for text, hex_ in live.items() if hex_ == accent]
    assert greens, "a streaming band must show its running indicator in accent"
    # …and that green is the spinner, not the brand glyph.
    assert all(ICON_MODEL not in text for text in greens), greens


def test_the_two_groups_never_crowd_closer_than_their_own_separator() -> None:
    """`_compose` pads with `max(_MIN_GROUP_GAP, …)`, and the FIT TEST reserves
    the same gap. Testing only the composed length let a row 'fit' with the
    groups one cell apart — tighter than the 3-cell ` · ` inside each group — so
    a filesystem path abutted a percentage and the left/right architecture the
    band is built on dissolved into one run. Reachable by dragging a window one
    cell at ordinary widths.
    """
    status, _clock = _full_band()
    for width in range(200, 20, -1):
        plain = status.render_text(width).plain.rstrip()
        # The widest internal run of spaces IS the group seam.
        gaps = [len(m.group(0)) for m in re.finditer(r" {2,}", plain)]
        if not gaps:
            # Everything shed down to a single group; nothing to separate.
            continue
        assert max(gaps) >= _MIN_GROUP_GAP, f"width {width}: seam {max(gaps)} < {_MIN_GROUP_GAP}"


def test_a_terminal_too_narrow_for_anything_still_names_the_model() -> None:
    """The ladder's old final rung DROPPED the model, leaving a bare glyph on an
    empty tinted strip — which reads as broken rather than compressed, and
    discards the answer to the only question the band still had room for. The
    label is truncated instead: `kimi-k2-t…` still says who is replying.
    """
    status, _clock = _full_band()
    for width in (30, 20, 17, 12, 8):
        plain = status.render_text(width).plain.rstrip()
        assert cell_len(plain) <= width
        body = plain.replace(ICON_MODEL, "").strip()
        assert body, f"width {width} rendered an empty band: {plain!r}"
        # Some recognisable part of the model id survives, not just the ellipsis.
        assert body.strip("…").strip(), f"width {width} kept only an ellipsis: {plain!r}"


# -- MCP segment -------------------------------------------------------------


def test_mcp_segment_counts_connected_servers_and_never_pluralises() -> None:
    """The count is SERVERS, and ``MCP`` is an initialism, not a noun."""
    assert format_mcp(McpStatus(configured=3, connected=3)) == "3 MCP"
    assert format_mcp(McpStatus(configured=1, connected=1)) == "1 MCP"
    # Configured but nothing up yet — a real state, and it renders.
    assert format_mcp(McpStatus(configured=2, connected=0)) == "0 MCP"


def test_mcp_segment_disappears_when_no_servers_are_configured() -> None:
    """``⊙ 0 MCP`` on a machine with no ``.mcp.json`` is seven cells asserting
    the absence of a feature the user never asked for. The segment appearing at
    all is part of the signal."""
    assert format_mcp(McpStatus()) == ""
    status = StatusLine(_dock(200))
    status.update(model_label="test/model", mcp=McpStatus())
    assert ICON_MCP not in status.render_text(200).plain


def test_mcp_glyph_is_a_single_cell_like_every_other_segment_icon() -> None:
    """The band's layout is measured arithmetic. A two-cell glyph would drift
    the right group's edge by a column on terminals that render it wide, which
    is why the reference's ``⊙`` was measured rather than assumed."""
    assert cell_len(ICON_MCP) == 1


def test_the_mcp_lamp_is_an_alarm_or_nothing() -> None:
    """A PARTIAL failure is the dangerous outcome: a healthy-looking count of 2
    on a machine where the third server died reads as "all good", and the user
    then spends the turn wondering why the agent cannot reach those tools.

    Healthy takes the NEUTRAL ramp, not a second green. `success` #57c785 is
    5.08 dE2000 from the accent #38c96a — this file's own comments reject 3.06
    as imperceptible — and the accent is the band's "a turn is live". `muted`
    against `dim` is 16.66 dE2000 apart, so the three states still read.
    """
    assert mcp_semantic(McpStatus(configured=3, connected=2, failed=True)) == "danger"
    assert mcp_semantic(McpStatus(configured=3, connected=3)) == "muted"
    # Configured, nothing up, nothing failed: the startup gate's normal state on
    # every launch, so it takes the dimmest step of the ramp.
    assert mcp_semantic(McpStatus(configured=3, connected=0)) == "dim"
    # Discovery itself failed: an alarm, and the only state with no count.
    assert mcp_semantic(McpStatus(discovery_failed=True)) == "danger"


def test_a_healthy_mcp_count_puts_no_second_green_on_a_streaming_band() -> None:
    """The accent means ONE thing on this row: a turn is live. Measured on the
    rendered row rather than on the rule, because the defect was two single-cell
    green glyphs ten cells apart — `⊙` in `success` at column 47 and the spinner
    in `accent` at 57 — which no assertion about the rule alone would catch.

    dE2000 between the two was 5.08, so this asserts the accent family appears
    on exactly ONE span and no other span carries a near neighbour of it.
    """
    status, _clock = _full_band()
    status.update(mcp=McpStatus(configured=2, connected=2), streaming=True)
    row = status.render_text(200)
    fills = _fills(row)
    accent = theme_mod.semantic_color("accent").lower()
    accented = [text for text, fill in fills.items() if fill == accent]
    assert len(accented) == 1, f"accent spent on {accented!r}"
    assert accented[0] in _SPINNER_FRAMES
    # No near neighbour of the accent anywhere else on the row (dE2000 < 10 is
    # the same-colour band at one cell; the old `success` lamp measured 5.08).
    for text, fill in fills.items():
        if text == accented[0]:
            continue
        assert _delta_e_2000(fill, accent) >= 10.0, f"{text!r} {fill} is a second green"


def test_a_discovery_failure_keeps_the_alarm_without_inventing_a_count() -> None:
    """`configured == 0` means two different things: a machine with no
    ``.mcp.json`` (stay away) and one whose config could not be read at all. The
    second renders the bare initialism — every count would be a fiction, and
    ``0 MCP`` in particular would claim the machine asked for nothing."""
    assert format_mcp(McpStatus(discovery_failed=True)) == "MCP"
    status = StatusLine(_dock(200))
    status.update(model_label="test/model", mcp=McpStatus(discovery_failed=True))
    row = status.render_text(200)
    assert f"{ICON_MCP} MCP" in row.plain
    assert _fills(row)[f"{ICON_MCP} "] == theme_mod.semantic_color("danger").lower()


def test_only_the_mcp_glyph_carries_the_state_colour() -> None:
    """Tinting ``2 MCP`` danger reads as "the number 2 is wrong". The glyph is
    the status lamp; the count beside it stays plain foreground."""
    status = StatusLine(_dock(200))
    status.update(model_label="test/model", mcp=McpStatus(configured=3, connected=2, failed=True))
    fills = _fills(status.render_text(200))
    danger = theme_mod.semantic_color("danger").lower()
    fg = theme_mod.semantic_color("fg").lower()
    assert fills[f"{ICON_MCP} "] == danger
    assert fills["2 MCP"] == fg


def test_an_ALARMING_mcp_segment_outlives_every_other_droppable_segment() -> None:
    """It is the reference's ``flexShrink={0}`` indicator, and the ladder's last
    rung WHEN IT IS AN ALARM. Two reasons, both asserted here by outcome: it is
    the cheapest segment in the band to keep, and its failure branch is the only
    alarm the band can raise — a cramped terminal is exactly where hiding it
    would make a user conclude the tools were never configured.
    """
    status, _clock = _full_band()
    status.update(mcp=McpStatus(configured=3, connected=2, failed=True))
    # Walk down until the MCP count goes, and record what is still standing.
    last_seen = None
    for width in range(200, 4, -1):
        plain = status.render_text(width).plain
        if "2 MCP" not in plain:
            break
        last_seen = plain
    assert last_seen is not None
    # At its final width every other droppable segment is already gone.
    for gone in ("49.6%/1M", "$12.40", "high", "3 agents", "41m1s"):
        assert gone not in last_seen, f"{gone} outlived the MCP segment: {last_seen!r}"


def test_a_healthy_mcp_count_sheds_before_the_cwd_and_the_model_label() -> None:
    """The rung's place is earned by the ALARM, not by the segment.

    Unconditional last place meant a count nobody has to act on outranked "where
    am I": at 40 cells the band read `◆ model › ⊙ 2 MCP` where the same terminal
    with no MCP configured showed `◆ test/model › ⌂ local-operator`. A courtesy
    sheds like one — just ahead of the cwd, and the two SHORTEN steps still come
    before it because they keep a segment instead of dropping one.
    """
    status = StatusLine(_dock(40))
    status.update(model_label="test/model", cwd="/Users/tester/work/local-operator")

    # Swept rather than pinned to one width: the invariant is an ORDER, and a
    # single width only ever samples one rung of it.
    status.update(mcp=McpStatus(configured=2, connected=2))
    for width in range(60, 9, -1):
        row = status.render_text(width).plain
        if ICON_MCP in row:
            # A healthy count never outlives the working directory.
            assert ICON_CWD in row, f"width {width}: mcp outlived the cwd: {row!r}"

    # The alarm does outlive both, which is the rung's whole justification.
    status.update(mcp=McpStatus(configured=2, connected=1, failed=True))
    alarm_alone = [
        width
        for width in range(60, 9, -1)
        if f"{ICON_MCP} 1 MCP" in status.render_text(width).plain
        and ICON_CWD not in status.render_text(width).plain
    ]
    assert alarm_alone, "a danger count must survive a width the cwd cannot"


def test_the_quiet_ladder_moves_mcp_and_the_alarm_is_last_either_way() -> None:
    """One ordering, two positions for one rung — not two hand-maintained
    ladders that can drift apart on the next reordering.

    Last place goes to the narrowest BOUNDED rung, and an alarm outranks a
    reading. `approvals` is now both: a bare `!` at one cell against `⊙ 3 MCP`'s
    seven. So it is authored last and every promotion leaves it there, which is
    what deleted the repair helper this test used to be about — that helper
    existed only because the segment then spelled out `! auto-approve always`
    and was the WIDEST rung in the band.
    """
    assert drop_ladder(McpStatus(configured=2, connected=1, failed=True)) is _DROP_LADDER
    assert drop_ladder(McpStatus(configured=2, connected=2)) is _DROP_LADDER_QUIET
    assert drop_ladder(McpStatus()) is _DROP_LADDER_QUIET
    assert _DROP_LADDER[-1] == "approvals"
    assert _DROP_LADDER[-2] == "mcp", "the danger count still outlives every reading"
    # A pending fork is a transient STATE, not a figure, so it outlives every
    # reading and sheds only ahead of the two remaining alarms. Bounded, so it
    # is a legal neighbour of the tail under `_UNBOUNDED_RUNGS`.
    assert _DROP_LADDER.index("fork") == _DROP_LADDER.index("mcp") - 1
    assert "fork" not in _UNBOUNDED_RUNGS
    assert _DROP_LADDER_QUIET.index("mcp") == _DROP_LADDER_QUIET.index("cwd") - 1
    # Promoting mcp cannot dislodge the alarm, because the alarm was never
    # sitting behind it.
    assert _DROP_LADDER_QUIET[-1] == "approvals"
    assert _DROP_LADDER_QUIET.index("context") < _DROP_LADDER_QUIET.index("approvals")
    # Same rungs in both ladders: the reorder moves things, it never drops one.
    assert sorted(_DROP_LADDER) == sorted(_DROP_LADDER_QUIET)


def test_an_estimated_context_sheds_before_the_working_directory() -> None:
    """The ladder ranks by what a segment is worth NOW, not by its slot.

    ``cwd`` sits ahead of ``context`` because the context number "predicts the
    operator's next action". That is true of a number the model reported and
    false of the boot estimate: before any turn nothing has been spent and
    nothing is near compaction, while "where am I" is if anything MORE useful
    in a session that has just opened.
    """
    for ladder in (
        drop_ladder(McpStatus(configured=2, connected=1, failed=True), context_estimated=True),
        drop_ladder(McpStatus(), context_estimated=True),
    ):
        assert ladder.index("context") < ladder.index("cwd")
    # Exact readings keep the documented order.
    for ladder in (
        drop_ladder(McpStatus(configured=2, connected=1, failed=True)),
        drop_ladder(McpStatus()),
    ):
        assert ladder.index("cwd") < ladder.index("context")
    # A reorder moves rungs; it never adds or drops one.
    assert sorted(drop_ladder(McpStatus(), context_estimated=True)) == sorted(_DROP_LADDER_QUIET)
    assert sorted(
        drop_ladder(McpStatus(configured=1, failed=True), context_estimated=True)
    ) == sorted(_DROP_LADDER)


def test_a_narrow_band_keeps_the_cwd_over_a_pre_turn_estimate(monkeypatch) -> None:
    """The rendered consequence, not just the rung order.

    Measured before the fix: between 40 and 48 cells an estimated reading
    evicted the working directory entirely, so a session that had just opened
    in the wrong directory rendered `◆ kimi-k2-thinking     ▦ 49.6%/1M` and
    never said where it was — trading the fact a user checks at boot for a
    number that cannot yet mean anything.
    """
    monkeypatch.setenv("HOME", "/Users/tester")
    status, _clock = _full_band()

    def narrowest_showing_cwd() -> int:
        widths = [w for w in range(30, 70) if ICON_CWD in status.render_text(w).plain]
        assert widths, "the cwd must survive somewhere in this range"
        return min(widths)

    status.update(context_is_estimate=False)
    exact_floor = narrowest_showing_cwd()
    status.update(context_is_estimate=True)
    estimate_floor = narrowest_showing_cwd()

    # The path survives materially narrower once the reading is only a guess:
    # 37 vs 51 cells as measured, i.e. the whole 40-48 band the review flagged.
    assert estimate_floor < exact_floor
    for width in range(estimate_floor, exact_floor):
        rendered = status.render_text(width).plain
        assert ICON_CWD in rendered, f"width {width} lost the cwd"
        assert "49.6%" not in rendered, f"width {width} kept the estimate over the cwd"


def test_the_last_surviving_rung_is_always_bounded() -> None:
    """What the tail rule actually protects, checked on all four variants.

    The render walk is monotone: it sheds down the ladder until the row fits
    and can never put a segment back. So whatever sits last has to be something
    that RELIABLY fits — and ``cwd`` does not, because it is as wide as the
    user's path.

    Ending on ``cwd`` is not merely suboptimal, it is self-defeating: the band
    sheds the armed ``! auto-approve`` alarm to make room for a path that then
    does not fit either, and paints neither. Measured on the quiet estimate
    ladder with a 24-character basename, that cost the alarm across 34-46 cells
    and paid for it with up to 29 blank ones.

    ``approvals`` takes last place in every variant now: it is a bare ``!``, so
    it is both the narrowest bounded rung and an alarm, which is the whole rule.
    It used to be the WIDEST rung (``! auto-approve always``, up to 20 cells) and
    was authored first for that reason, which is what made a repair helper
    necessary to keep promotions from stranding it at the end.
    """
    variants = {
        "full": _DROP_LADDER,
        "quiet": _DROP_LADDER_QUIET,
        "full+estimate": _DROP_LADDER_ESTIMATE,
        "quiet+estimate": _DROP_LADDER_QUIET_ESTIMATE,
    }
    for name, ladder in variants.items():
        assert ladder[-1] not in _UNBOUNDED_RUNGS, f"{name} ends on an unbounded rung"
        # A promotion reorders; it never adds or loses a rung.
        assert sorted(ladder) == sorted(_DROP_LADDER), f"{name} changed the rung set"

    # One position, all four variants: nothing needs repairing any more.
    assert _DROP_LADDER[-1] == "approvals"
    assert _DROP_LADDER_QUIET[-1] == "approvals"
    assert _DROP_LADDER_ESTIMATE[-1] == "approvals"
    assert _DROP_LADDER_QUIET_ESTIMATE[-1] == "approvals"
    assert _DROP_LADDER_QUIET_ESTIMATE.index("context") < _DROP_LADDER_QUIET_ESTIMATE.index(
        "cwd"
    ), "D21: a pre-turn estimate still sheds before the working directory"


def test_an_armed_alarm_outlives_a_path_that_would_not_have_fitted() -> None:
    """The rendered consequence of the rule above, at the widths it cost.

    Before: the walk shed ``approvals`` to keep an unbounded ``cwd`` that then
    failed to fit, so a session with the approval gate DISARMED painted no
    indication of it — a regression against main, where the same terminal keeps
    the alarm because a band with no estimate uses the quiet ladder.

    The alarm is a bare ``!`` now rather than ``! auto-approve``, which is why
    this probes the glyph as the band's last cell: the word is gone, the warning
    is not, and it is the ONE thing in the UI that says the gate is disarmed for
    longer than a scroll (a saved ``tool_approval_mode: auto`` is adopted at boot
    with no notice at all).
    """
    status, _clock = _full_band()
    status.update(
        cwd="/Users/tester/work/customer-identity-service",
        context_is_estimate=True,
        approvals_auto=True,
        mcp=McpStatus(configured=2, connected=2),
    )
    armed = [w for w in range(36, 52) if status.render_text(w).plain.rstrip().endswith("!")]
    assert armed, "the disarmed-gate alarm vanished at every narrow width"
    for width in armed:
        assert ICON_CWD not in status.render_text(width).plain

    # And it costs nothing in the common case: with the gate armed the rung
    # paints nothing, so the path still gets the width.
    status.update(approvals_auto=False)
    assert ICON_CWD in status.render_text(50).plain


# -- the trailing segment: the session name ----------------------------------


def test_the_session_name_is_the_bands_last_segment() -> None:
    """The slot the approval mode used to own.

    Asked for directly: "instead of auto approve indicator at the bottom right,
    let's change that to show the session name which is more valuable than
    showing the approval status". The right group is edge-aligned, so "last
    segment" means the last non-blank run in the row.
    """
    status, _clock = _full_band()
    status.update(conversation_name="Add todo guardrails", approvals_auto=False)
    assert status.render_text(200).plain.rstrip().endswith("Add todo guardrails")


def test_the_alarm_sits_beside_the_name_rather_than_replacing_it() -> None:
    """Both, in a fixed order — the name last, the one-cell alarm before it.

    The alarm is not removed, because nothing else in the UI says the gate is
    disarmed for longer than a scroll: `/approvals` answers on demand, the
    notice that latched the mode scrolls away, and a saved
    ``tool_approval_mode: auto`` is adopted at boot with no notice at all. What
    it loses is the prose — `auto` and `always` are one glyph now.
    """
    status, _clock = _full_band()
    status.update(
        conversation_name="Add todo guardrails", approvals_auto=True, approvals_always=True
    )
    row = status.render_text(200).plain.rstrip()
    assert row.endswith("Add todo guardrails")
    assert "auto-approve" not in row, "the alarm's prose was crowding out the name"
    assert "always" not in row
    # The glyph, immediately before the name and after its seam.
    assert re.search(r"! +‹ Add todo guardrails$", row)


def test_an_unnamed_session_leaves_the_trailing_slot_to_whatever_is_left() -> None:
    """What the segment shows BEFORE a name exists — the decision, pinned.

    The band never invents a label here. It carries no name until the first
    substantive message, at which point the opener names it (see
    ``local_operator.session.naming.provisional_title``), so this state lasts
    seconds rather than the minutes it used to. Meanwhile the trailing slot is
    simply the next segment along, and NO segment changes meaning: `!` always
    means the gate is disarmed, and the duration is always the duration. The
    working directory is not substituted, because the band already carries it in
    the identity group on the left — the terminal TITLE is the surface with
    nothing else to fall back on, and that one does substitute the cwd.
    """
    status, _clock = _full_band()
    status.update(conversation_name="", approvals_auto=True)
    assert status.render_text(200).plain.rstrip().endswith("!")
    status.update(approvals_auto=False)
    assert status.render_text(200).plain.rstrip().endswith("41m1s")


def test_a_long_name_is_truncated_rather_than_crowding_the_left_group() -> None:
    """The name is model-generated: it has no natural bound, and the band is a
    fixed two-row box, so an uncapped segment would evict the identity group
    instead of wrapping."""
    status, _clock = _full_band()
    status.update(conversation_name="x" * 200, approvals_auto=False)
    row = status.render_text(200).plain
    # `truncate_cells` spends one of the cells on the ellipsis, so the segment
    # measures NAME_CELLS in total rather than NAME_CELLS plus a marker.
    assert "x" * (NAME_CELLS - 1) + "…" in row
    assert "x" * NAME_CELLS not in row
    # The left group is intact at a width where an uncapped name would have
    # pushed the model label off the row entirely.
    assert ICON_MODEL in row


def test_a_late_arriving_name_repaints_the_segment() -> None:
    """Auto-naming lands asynchronously and a rename can land at any time; both
    reach the band through the same ``update`` call, so neither needs a
    restart."""
    status, _clock = _full_band()
    status.update(conversation_name="", approvals_auto=False)
    assert "Fix the login flow" not in status.render_text(200).plain
    status.update(conversation_name="Fix the login flow")
    assert status.render_text(200).plain.rstrip().endswith("Fix the login flow")
    status.update(conversation_name="Something the user typed")
    assert status.render_text(200).plain.rstrip().endswith("Something the user typed")


def test_the_name_never_takes_the_accent() -> None:
    """The accent budget is spent on "what the turn is on" (``local_operator.tcss``
    enumerates every site). A session's name is metadata, so it takes the same
    secondary ink as the numbers beside it."""
    status, _clock = _full_band()
    status.update(conversation_name="Add todo guardrails", approvals_auto=False)
    rendered = status.render_text(200)
    accent = theme_mod.semantic_color("accent")
    muted = theme_mod.semantic_color("muted")
    spans = [
        span
        for span in rendered.spans
        if "Add todo guardrails" in rendered.plain[span.start : span.end]
    ]
    assert spans, "the name was not painted as its own span"
    for span in spans:
        style = span.style
        colour = getattr(style, "color", None)
        assert colour is not None
        assert colour.name != accent, "the session name took the accent"
        assert colour.name == muted


class _LadderState(TypedDict):
    """One ladder variant's inputs. Typed because ``mcp`` is the only key whose
    value is not a bool, and a bare ``dict`` widens both to ``object``.
    """

    mcp: McpStatus
    estimated: NotRequired[bool]


#: Every ladder variant, by the state that selects it. The eviction defect below
#: existed in all four, so a sweep that only drove the default one would have
#: missed three quarters of it.
_LADDER_STATES: dict[str, _LadderState] = {
    "healthy-mcp/exact": {"mcp": McpStatus(configured=3, connected=3)},
    "failed-mcp/exact": {"mcp": McpStatus(configured=3, connected=1, failed=True)},
    "healthy-mcp/estimated": {"mcp": McpStatus(configured=3, connected=3), "estimated": True},
    "failed-mcp/estimated": {
        "mcp": McpStatus(configured=3, connected=1, failed=True),
        "estimated": True,
    },
}

#: Names of the three lengths a real session wears in one conversation: the
#: opener excerpt (40+ cells), the generated title, and a re-title. Two of the
#: three arrive with no user input at all.
_NAMES = (
    "The /resume picker shows (unnamed session) for image openers",
    "Fix unnamed sessions from image openers",
    "Bulk export the invoice columns",
    "Fix boot",
)


def _swept_band(state: _LadderState, *, alarm: bool) -> StatusLine:
    """A fully populated band in one ladder variant, ready to render at width."""
    status, _clock = _full_band()
    status.update(
        jobs=1,
        mcp=state["mcp"],
        approvals_auto=alarm,
        context_tokens=496_000,
        context_window=1_000_000,
    )
    status._context_is_estimate = bool(state.get("estimated"))
    return status


def _name_ink(row: str, width: int) -> int:
    """Cells the trailing name segment actually INKS on the row.

    The segment is unpadded — the name spends only the cells its truncated text
    measures and the inter-group filler absorbs whatever a short title frees —
    so the ink runs from the cell after the title's own seam to the band's
    right edge. Measured to ``width`` rather than to ``len(row)`` because the
    row is laid out to exactly ``width``: the title is the group's last segment
    and its final cell is the band's edge, so the edge is where the ink ends.
    """
    seam = f" {_SEP_RIGHT} "
    return width - row.rindex(seam) - len(seam)


def test_naming_a_session_never_costs_a_segment_for_nothing(monkeypatch) -> None:
    """Every concession the band makes for a name buys a name. All widths.

    The defect this pins was the opposite: between 83 and 94 columns (79-90 with
    the alarm disarmed, and in all four ladder variants) naming a session removed
    `▴ high` AND showed no name in exchange, so the band got strictly worse the
    moment the session earned a title — 7 cells of a setting the user chose by
    keystroke spent, 9 cells of hole added, nothing bought.

    The cause was the walk, not the rung order: it sheds monotonically, so the
    duration, the counters and the effort segment it gave up on the way to
    keeping the name stayed given up when the name was dropped two rungs later.
    ``_fit`` re-walks with the name off the table instead.

    So the contract has two halves, and both are asserted at every width in every
    variant. Where the name is DROPPED the band is exactly the band it would have
    been unnamed — no segment is missing. Where the name is SHOWN a segment may
    have been traded for it (that trade is the ladder's, and its order is
    asserted by ``test_segments_disappear_in_the_declared_ladder_order``), but at
    least a floor's worth of name has to be on the row in return.
    """
    monkeypatch.setenv("HOME", "/Users/tester")
    for label, state in _LADDER_STATES.items():
        for alarm in (True, False):
            status = _swept_band(state, alarm=alarm)
            for width in range(30, 181):
                status.update(conversation_name="")
                status.render_text(width)
                bare = set(status._dropped)
                for name in _NAMES:
                    status.update(conversation_name=name)
                    row = status.render_text(width).plain
                    where = f"{label} alarm={alarm} width={width}"
                    lost = set(status._dropped) - bare - {"name"}
                    if status.is_showing("name"):
                        # The GRANT is what the concessions bought, and the ink
                        # can legitimately run under it twice over: `Fix boot`
                        # inks 8 cells inside an 18-cell grant because that is
                        # the whole title, and a word-boundary cut spends up to
                        # a third of the cap on reading well (`truncate_name`'s
                        # bound). So a shown name is the complete title, or at
                        # least two thirds of a floor's worth of it.
                        ink = _name_ink(row, width)
                        assert ink >= NAME_CELLS_FLOOR * 2 // 3 or row.endswith(
                            name
                        ), f"{where}: {sorted(lost)} spent for a {ink}-cell name"
                    else:
                        assert (
                            not lost
                        ), f"{where}: the name was dropped and {sorted(lost)} went with it"


def test_the_layout_is_a_function_of_the_titles_ink_not_its_content(monkeypatch) -> None:
    """Two titles of equal measure get byte-identical layout — only the text differs.

    The name is inked unpadded now (see :data:`NAME_CELLS` on the reversal), so
    the row's columns legitimately move with the title's LENGTH — the fixed
    columns were the old reserved box's purchase, paid for with a permanent
    blank run this change removes. What must survive the reversal is that the
    length is all a title can move: the walk measures cells and never reads
    the string, so a re-title to a same-measure name (the equality this test
    constructs: same word boundaries, same cell count, so the same cut at
    every cap) changes not one column, and any two titles differ in layout
    only as far as their measures differ. A violation would mean the band's
    layout depends on the title's CONTENT, which is the door back to a row
    that reshuffles for reasons a reader cannot see.

    The pair share word-boundary structure on purpose — `truncate_name` cuts
    on words, so equal structure is what makes their ink equal at every cap.
    """
    monkeypatch.setenv("HOME", "/Users/tester")
    pairs = (
        ("Fix the login flow", "Add the audit sync"),
        ("Bulk export the invoice columns", "Auto import the payment receipts"[:31]),
        ("a", "z"),
    )
    for label, state in _LADDER_STATES.items():
        status = _swept_band(state, alarm=True)
        for width in range(30, 181):
            for first, second in pairs:
                status.update(conversation_name=first)
                row_a = status.render_text(width).plain
                shown_a = status.is_showing("name")
                status.update(conversation_name=second)
                row_b = status.render_text(width).plain
                shown_b = status.is_showing("name")
                where = f"{label} width={width} pair=({first!r}, {second!r})"
                assert shown_a == shown_b, f"{where}: one title survived, the other did not"
                if not shown_a:
                    assert row_a == row_b, f"{where}: unnamed rows differ"
                    continue
                ink = _name_ink(row_a, width)
                assert ink == _name_ink(row_b, width), f"{where}: the ink moved"
                assert (
                    row_a[: len(row_a) - ink] == row_b[: len(row_b) - ink]
                ), f"{where}: the layout depends on the title's content"


def test_the_name_takes_every_cell_the_row_can_spare(monkeypatch) -> None:
    """The segment is elastic between its floor and its cap, not one or the other.

    Two fixed widths meant up to 20 cells sat idle beside an 18-cell stub — a
    130-column terminal showed `Add todo guardrai…` next to an 18-cell hole with
    32 cells available, and the excerpt-to-title upgrade this feature exists for
    was invisible below 138 columns because both strings were cut to the same
    stub. So the test is not "the name is 18 or 40": it is that the gap closes to
    the seam and the box grows with the terminal.
    """
    monkeypatch.setenv("HOME", "/Users/tester")
    name = "Add todo guardrails to the operator loop"
    status = _swept_band(_LADDER_STATES["healthy-mcp/exact"], alarm=True)
    status.update(conversation_name=name)
    inks = []
    for width in range(100, 181):
        row = status.render_text(width).plain
        ink = _name_ink(row, width)
        inks.append(ink)
        # While the title is still hungry (cut short), the layout may not idle:
        # every run wider than the group gap has to be cells the title could
        # not have used. The name is unpadded and cut on a word boundary, so
        # the one legitimate excess is the word-cut remainder — the cells
        # between the cut title's ink and the cap the walk granted it, which
        # `truncate_name` bounds at a third of the cap. Anything past that is a
        # hole the old fixed-widths defect would have left.
        head = row[: len(row) - ink]
        longest_gap = max((len(run) for run in re.findall(r" {2,}", head)), default=0)
        assert (
            row.endswith(name) or longest_gap <= _MIN_GROUP_GAP + NAME_CELLS // 3
        ), f"width={width}: {longest_gap} cells idle beside a {ink}-cell name"
    # The segment is elastic: the ink grows with the terminal, reaching the
    # whole title (this one measures exactly NAME_CELLS) at the wide end and
    # never dropping under the floor's word-cut worst case at the narrow end.
    assert max(inks) == cell_len(name) == NAME_CELLS
    assert min(inks) >= NAME_CELLS_FLOOR * 2 // 3
    # More than the two widths the old constant pair allowed.
    assert len(set(inks)) > 5, f"the name still has only {sorted(set(inks))} widths"


def test_the_row_is_flush_to_the_bands_right_edge_at_every_title_length(monkeypatch) -> None:
    """The title's last inked cell IS the band's right edge — no dead tail.

    The name is emitted unpadded (see :data:`NAME_CELLS` on the reversal from
    the reserved box), so this holds by construction rather than by a slack
    payout: the row's trailing segment is the title's own ink, cut to the cap
    the walk granted it. Both halves are asserted because either alone passes
    on a shape this exists to reject — `row == row.rstrip()` is satisfied by a
    row that stops short of the edge, and `cell_len(row) == width` is satisfied
    by a row whose last cells are padding blanks.
    """
    monkeypatch.setenv("HOME", "/Users/tester")
    for label, state in _LADDER_STATES.items():
        status = _swept_band(state, alarm=True)
        for width in (100, 120, 160):
            for name in ("a", "short", "Bulk export the invoice columns"):
                status.update(conversation_name=name)
                row = status.render_text(width).plain
                if not status.is_showing("name"):
                    continue
                where = f"{label} width={width} name={name!r}"
                assert (
                    row == row.rstrip()
                ), f"{where}: {len(row) - len(row.rstrip())} dead cells at the band's edge"
                assert (
                    cell_len(row) == width
                ), f"{where}: the row stops {width - cell_len(row)} cells short of the edge"
                # The ink really is the title's, not a coincidence of
                # truncation. `truncate_name` is not a fixed point — a title
                # word-cut at a 20-cell cap inks 16 cells, and re-cutting the
                # original at 16 lands on an earlier word — so the assertion is
                # on the tail's shape: the whole title, or a prefix of it
                # ending in the ellipsis.
                tail = row[len(row) - _name_ink(row, width) :]
                assert tail == name or (
                    tail.endswith("…") and name.startswith(tail[:-1].rstrip())
                ), where


def test_the_titles_seam_stays_tight_against_the_title(monkeypatch) -> None:
    """The `‹` introduces the name and has to touch it (design review D1).

    Under the unpadded design this holds by construction — the title is the
    seam's immediate neighbour because nothing is painted between them — but it
    is pinned anyway, because both historical arrangements of the old reserve
    broke one half of it: `ljust` + `rstrip` kept the seam tight and ended the
    row early, and `rjust` reached the edge while orphaning the chevron behind
    a run of blanks that read as a segment that failed to render. Asserting
    the pair pins the one arrangement that does both.

    The third assertion is the one that falsifies the RESERVED design itself
    (review round 1, M1): paying the box's slack out on the seam's LEFT kept
    the seam tight and the row flush — both checks above passed against the
    very code this PR removes — while parking the dead run between the alarm
    and the chevron. So the cells left of the title's seam must end on ink:
    the seam's own leading space is the only blank allowed to touch them.
    """
    monkeypatch.setenv("HOME", "/Users/tester")
    for label, state in _LADDER_STATES.items():
        status = _swept_band(state, alarm=True)
        for width in (100, 120, 160):
            for name in ("a", "short", "Bulk export the invoice columns"):
                status.update(conversation_name=name)
                row = status.render_text(width).plain
                if not status.is_showing("name"):
                    continue
                where = f"{label} width={width} name={name!r}"
                seam = f" {_SEP_RIGHT} "
                tail = row[row.rindex(seam) + len(seam) :]
                assert tail and not tail.startswith(" "), f"{where}: the seam was orphaned"
                # Both halves together, because either alone is satisfied by a
                # shape this test exists to reject: `ljust` + `rstrip` also
                # leaves the seam tight (it just ends the row early), and
                # `rjust` also reaches the edge (it just orphans the seam). Only
                # asserting the pair pins the one arrangement that does both,
                # and without this the test passed against the code it replaced
                # (F2).
                assert (
                    cell_len(row) == width
                ), f"{where}: the seam is tight but the row stops short of the edge"
                # The discriminating half (M1): everything left of the title's
                # seam ends on ink. The old reserved box painted its unused
                # cells exactly here — a blank run between the alarm and the
                # `‹` — and both assertions above were satisfied by it. The
                # inter-group filler lives further left and is bounded by the
                # groups' own separators, so a trailing blank run here can only
                # be a reintroduced name reserve.
                head = row[: row.rindex(seam)]
                assert head == head.rstrip(), (
                    f"{where}: {len(head) - len(head.rstrip())} dead cells "
                    f"between the right group and the title's seam"
                )


def test_a_double_width_title_never_overflows_the_band(monkeypatch) -> None:
    """The band is ONE row, in cells — including for CJK and Hangul titles.

    Every cut and every fit test on this row must count CELLS (``cell_len``),
    never characters. Counting characters — as a ``str.rjust`` padding of the
    name once did — emits a row wider than the band for a title of
    double-width glyphs: 592 rows over their width across the ladder variants,
    worst case 11 cells. Rich drops the over-wide segment rather than clipping
    it, so the effect was a CJK-named conversation resuming with no name on
    the band at all — this feature's own failure, reintroduced for non-Latin
    titles (review round 1, F1).
    """
    monkeypatch.setenv("HOME", "/Users/tester")
    titles = (
        "修复恢复时的会话标题",
        "セッションのタイトルを復元する",
        "세션 제목 복원하기",
        "修复登录重定向循环的问题并添加测试用例以防止回归",
    )
    for label, state in _LADDER_STATES.items():
        for alarm in (True, False):
            status = _swept_band(state, alarm=alarm)
            for name in titles:
                status.update(conversation_name=name)
                for width in range(60, 181):
                    row = status.render_text(width).plain
                    assert cell_len(row) <= width, (
                        f"{label} alarm={alarm} width={width} name={name!r}: "
                        f"the row is {cell_len(row)} cells wide"
                    )
                    if not status.is_showing("name"):
                        # No name means no reserved box to fill, so the row is
                        # whatever the surviving segments are — only the
                        # overflow half of the contract applies here.
                        continue
                    # EQUAL, not merely `<=`. Both halves of the cell/character
                    # mismatch are failures and only one of them is an overflow:
                    # padding a double-width title by CHARACTERS overshoots the
                    # box, and a row that then strips the excess lands SHORT of
                    # the edge instead of over it (268 short rows measured that
                    # way). Asserting the row is exactly its width catches the
                    # mismatch in whichever direction it manifests — `<=` alone
                    # passed against the very code this pins (F2).
                    assert cell_len(row) == width, (
                        f"{label} alarm={alarm} width={width} name={name!r}: "
                        f"the row is {cell_len(row)} cells wide, not {width}"
                    )


def test_the_row_is_flush_even_when_the_name_is_the_only_segment(monkeypatch) -> None:
    """The name is not always preceded by a sibling, and the row must still be flush.

    A freshly-opened session has no counters, no context reading, no cost and no
    duration — ``format_jobs(0)``, ``format_agents(0)`` and
    ``format_context_usage(0, …)`` all return "" and the duration is suppressed
    at zero — so the title is the WHOLE right group and lands at index 0 of the
    join loop. The inter-group filler in ``_compose`` is what carries the row
    to the band's edge here (the unpadded title cannot), so this pins the
    sparse shape where a filler bug would surface first — the very first band
    a user sees (review round 1, F1).

    This is the sparse band on purpose. Every other band test populates the row.
    """
    monkeypatch.setenv("HOME", "/Users/tester")
    for width in (80, 100, 120, 160):
        for name in ("a", "Short", "Reduce agent RAM usage", "A much longer title than that"):
            dock = _dock(width)
            status = StatusLine(dock)
            status.update(
                model_label="anthropic/claude-opus-4",
                cwd="/Users/tester/work/local-operator",
                conversation_name=name,
            )
            row = status.render_text(width).plain
            if not status.is_showing("name"):
                continue
            assert (
                cell_len(row) == width
            ), f"width={width} name={name!r}: the row stops {width - cell_len(row)} cells short"


def test_the_bands_cut_lands_on_a_word_like_the_excerpt_it_was_handed() -> None:
    """``provisional_title`` cuts on a word boundary on purpose — "so it reads as
    a quotation rather than as a string that ran out of buffer" — and the band
    used to re-cut the same string mid-word one call later, throwing that away.

    The floor case is the exception the rule needs: at an 18-cell cap a word cut
    would leave `Add todo…`, which distinguishes two tiled sessions less well
    than the mid-word cut does. So the boundary is taken only while it costs
    less than a third of the cap.
    """
    excerpt = "The /resume picker shows (unnamed session) for image openers"
    assert truncate_name(excerpt, 40) == "The /resume picker shows (unnamed…"
    assert truncate_name(excerpt, 24) == "The /resume picker…"
    assert truncate_name("Add todo guardrails to the operator loop", 22) == "Add todo guardrails…"
    # Below the tolerance the cut stays where the cells are.
    assert truncate_name("Add todo guardrails to the operator loop", 18) == "Add todo guardrai…"
    # A name that fits is untouched, and a single unbroken token cannot be
    # word-cut at all.
    assert truncate_name("Fix boot", 40) == "Fix boot"
    assert truncate_name("x" * 60, 40) == "x" * 39 + "…"


def test_the_alarm_is_not_the_same_ink_as_the_cost_figure() -> None:
    """One cell of glyph means the INK is the whole signal, and in `warning` it
    was the identical hue to `◈ $73.92` three cells away (both `#e0b04b`), so the
    band's only alarm read as another figure. `danger` is the band's alarm ink —
    the `⊙` lamp already takes it when MCP discovery fails — and it stays legible
    on both ramps: 6.62:1 dark, 4.94:1 on the paper ramp.
    """
    status, _clock = _full_band()
    status.update(conversation_name="Add todo guardrails", approvals_auto=True)
    for ramp in ("dark", "light"):
        theme_mod.set_theme(ramp)
        try:
            rendered = status.render_text(200)
            inks = {
                rendered.plain[span.start : span.end].strip(): getattr(span.style, "color", None)
                for span in rendered.spans
            }
            alarm, cost = inks[ICON_APPROVALS], inks["$12.40"]
            assert alarm is not None and cost is not None
            assert alarm.name == theme_mod.semantic_color("danger"), ramp
            assert cost.name == theme_mod.semantic_color("warning"), ramp
            assert alarm.name != cost.name, f"{ramp}: the alarm and the cost are one ink"
        finally:
            theme_mod.set_theme("dark")


# -- repaint vs reflow -------------------------------------------------------


def test_the_band_never_asks_for_a_layout_pass() -> None:
    """The band's box is fixed by the sheet (``#status-band { height: 2 }``)
    and docked, so nothing it paints can move anything.

    Textual's ``Static.update`` reflows the WHOLE screen by default, and this
    runs on the 12.5 Hz spinner for the length of every turn. Measured A/B on a
    161-block transcript: 9.6% of a core with the default against 7.2% without
    it — about a quarter of the turn's idle burn, spent to animate one glyph.
    """
    dock = FakeDock(80)
    status = StatusLine(cast(Static, dock))

    status.update(model_label="openrouter/moonshotai/kimi-k2-thinking", cwd="/tmp")
    assert dock.layout is False

    # And on every later repaint, including the spinner's own.
    dock.layout = True
    status.update(streaming=True)
    assert dock.layout is False
    dock.layout = True
    status._advance_spinner()
    assert dock.layout is False


def test_the_band_still_paints_what_it_was_told() -> None:
    """The guard above is only safe while the content still lands: a repaint
    that skips the layout must not also skip the paint."""
    dock = FakeDock(200)
    status = StatusLine(cast(Static, dock))

    status.update(cost="$12.40")

    assert dock.painted is not None
    assert "$12.40" in dock.painted.plain


def test_every_segment_icon_stays_in_the_block_the_terminal_font_covers() -> None:
    """Segment icons live in Geometric Shapes (U+25xx), and that is a fix.

    `ICON_JOBS` was `⊞` (U+229E SQUARED PLUS, Mathematical Operators) and
    rendered as TOFU — an empty replacement box — in a real terminal, while
    every U+25xx glyph on the same row painted correctly. That is what makes
    this a property of the BLOCK rather than of one unlucky codepoint: fonts
    that cover Geometric Shapes routinely stop short of Mathematical Operators.

    This is asserted here because **no other check in the repo can see it**.
    `cell_len` returns 1 for tofu exactly as it does for a real glyph, so the
    band's width arithmetic stays correct and every layout test keeps passing
    while the icon is invisible to the user. An SVG export embeds the character
    rather than the rendered shape, so it cannot see it either. The only
    instruments are a human looking at a terminal, and this rule.

    `ICON_MCP` is deliberately EXCLUDED and left in U+22xx: it has not been
    observed broken, so it is recorded as a known risk at its definition rather
    than swapped on suspicion. If it is ever reported as an empty box, the fix
    is to move it into U+25xx and add it to this test.

    `ICON_APPROVALS` is excluded because it is plain ASCII `!`, and `ICON_CWD`
    because `⌂` (U+2302) predates the band and has been on screen in every
    release since without a report.
    """
    icons = {
        "ICON_MODEL": ICON_MODEL,
        "ICON_AGENTS": ICON_AGENTS,
        "ICON_JOBS": ICON_JOBS,
        "ICON_CONTEXT": ICON_CONTEXT,
        "ICON_COST": ICON_COST,
        "ICON_DURATION": ICON_DURATION,
    }
    for name, glyph in icons.items():
        assert len(glyph) == 1, f"{name} is not a single codepoint: {glyph!r}"
        point = ord(glyph)
        assert 0x2500 <= point <= 0x25FF, (
            f"{name} is U+{point:04X}, outside Geometric Shapes (U+25xx). "
            "cell_len cannot detect tofu, so a glyph outside this block has to "
            "be confirmed in a real terminal before it ships."
        )
        # The width rule the band's arithmetic depends on, checked on the same
        # pass: a two-cell glyph would drift the right group's edge by a column.
        assert cell_len(glyph) == 1, f"{name} is not one cell wide"

    # The jobs and context icons sit side by side in the right group, so they
    # have to be tellable apart at a glance and not merely unequal as strings.
    assert ICON_JOBS != ICON_CONTEXT


# ---------------------------------------------------------------------------
# The context reading's colour ramp
# ---------------------------------------------------------------------------


def test_the_context_colour_warms_as_the_context_fills() -> None:
    """Blue below 200k, purple above it, red above 500k.

    The reading is the one number in the band whose colour carries
    information: a session heading for compaction should be visible without
    being read. This is the ABSOLUTE half of the ramp, which is what keeps a
    very large window legible; an unknown window (0) isolates it, since the
    proportional half needs a denominator.
    """
    assert context_semantic_color(0, 0) == "signal"
    assert context_semantic_color(199_999, 0) == "signal"
    assert context_semantic_color(200_001, 0) == "label"
    assert context_semantic_color(499_999, 0) == "label"
    assert context_semantic_color(500_001, 0) == "danger"


def test_a_reading_exactly_on_a_boundary_keeps_the_calmer_colour() -> None:
    """Strictly greater-than on both ladders, so a context parked on a
    boundary does not flicker between two hues as the estimate wobbles by a
    token."""
    assert context_semantic_color(200_000, 0) == "signal"
    assert context_semantic_color(500_000, 0) == "label"
    # The proportional ladder on its own boundaries, isolated by a 200k window
    # where the absolute rungs are unreachable and cannot mask the result:
    # 55% is 110,000 and 80% is 160,000.
    assert context_semantic_color(110_000, 200_000) == "signal"
    assert context_semantic_color(110_001, 200_000) == "label"
    assert context_semantic_color(160_000, 200_000) == "label"
    assert context_semantic_color(160_001, 200_000) == "danger"


def test_a_small_window_still_reaches_the_warm_rungs() -> None:
    """D1: the absolute rungs are unreachable on most models.

    70% of the registry's windowed models are 200k or smaller, so an
    absolute-only ramp left them calm blue at 100% full with compaction
    already overdue. The proportional half is what fires there.
    """
    assert context_semantic_color(100_000, 200_000) == "signal"
    assert context_semantic_color(150_000, 200_000) == "label"
    assert context_semantic_color(199_000, 200_000) == "danger"
    assert context_semantic_color(200_000, 200_000) == "danger"


def test_the_reading_is_never_calm_while_compaction_is_due_at_the_default_trigger() -> None:
    """The property the ramp exists for, asserted against the REAL trigger.

    Colour meaning "how full am I" is only honest if a calm reading implies
    no pass is due. Checked against ``should_compact`` itself rather than
    against the ramp's own thresholds, so the two cannot drift apart.

    Scoped to the DEFAULT ``CompactionSettings`` on purpose, and named for it,
    because the ramp's fractions are constants while the trigger is user
    config: someone who sets ``threshold_percent`` below 0.55, or a
    ``threshold_tokens``/explicit ``reserve_tokens`` that resolves lower than
    the proportional rung, moves the trigger under the ramp and gets a calm
    reading while a pass is due. That degrades to the pre-change appearance
    rather than to a wrong one, and banding on the resolved trigger would
    mean plumbing ``resolve_threshold_tokens`` into the band. The claim is
    narrowed to what is actually verified rather than left sounding absolute.
    """
    settings = CompactionSettings()
    for window in (131_072, 200_000, 1_000_000):
        for numerator in range(1, 21):
            tokens = int(window * numerator / 20)
            if should_compact(tokens, window, settings):
                assert context_semantic_color(tokens, window) != "signal", (
                    f"{tokens:,}/{window:,} is past its compaction trigger "
                    "but still paints the calm base colour"
                )


def test_the_band_paints_the_context_segment_in_its_band_colour() -> None:
    """The ramp reaches the rendered row, not just the helper.

    Asserted on a 1M window at each band, because the window is exactly what
    the colour must NOT depend on — 300k of 1M is only 30% full and still
    warrants the warmer hue, since re-sending 300k tokens is expensive
    whatever the window.
    """
    for tokens, semantic in ((120_000, "signal"), (300_000, "label"), (700_000, "danger")):
        status = StatusLine(_dock(200))
        status.update(model_label="test/model", context_tokens=tokens, context_window=1_000_000)
        row = status.render_text(200)
        reading = format_context_usage(tokens, 1_000_000)
        assert reading in row.plain
        fills = _fills(row)
        painted = {text: fill for text, fill in fills.items() if reading in text}
        assert painted, f"the context segment was not painted for {tokens}"
        expected = theme_mod.semantic_color(semantic).lower()
        assert set(painted.values()) == {
            expected
        }, f"{tokens} tokens should paint {semantic} ({expected}), got {painted}"


def test_the_three_context_bands_are_visually_distinct() -> None:
    """A ramp nobody can tell apart is not a ramp. The three hues must be
    separated in both themes, since the palette differs between them."""
    for theme in ("dark", "light"):
        hexes = [theme_mod.semantic_color(name, theme) for name in ("signal", "label", "danger")]
        assert len(set(hexes)) == 3, f"{theme} reuses a hue across the context bands"


def test_the_warm_rungs_carry_weight_as_well_as_hue() -> None:
    """D3: hue alone cannot carry this signal.

    `signal` and `label` are ~35 dE apart in normal vision and 1.7 under
    deuteranopia, so for the commonest colour-vision deficiency the middle
    rung would not exist at all. Weight is orthogonal to hue and costs no
    cells. The base rung stays regular, so warm is the marked state.
    """
    weights: dict[int, bool] = {}
    for tokens in (120_000, 300_000, 700_000):
        status = StatusLine(_dock(200))
        status.update(model_label="test/model", context_tokens=tokens, context_window=1_000_000)
        row = status.render_text(200)
        reading = format_context_usage(tokens, 1_000_000)
        # ``Span.style`` is ``str | Style``; only the Style spans carry the
        # paint, and narrowing keeps this honest rather than casting.
        bolds = {
            span.style.bold
            for span in row.spans
            if isinstance(span.style, Style)
            and span.style.color is not None
            and reading in row.plain[span.start : span.end]
        }
        assert len(bolds) == 1, f"the reading was painted inconsistently at {tokens}"
        weights[tokens] = bool(bolds.pop())
    assert weights[120_000] is False, "the calm rung must not be bold"
    assert weights[300_000] is True, "the purple rung needs a non-colour carrier"
    assert weights[700_000] is True, "the red rung needs a non-colour carrier"


def test_weight_tracks_attention_across_every_red_in_the_band() -> None:
    """D6: three segments may take red, and weight must not invert them.

    The context reading is bold on its warm rungs as a colour-vision carrier.
    Left alone, that made the one red the band calls self-correcting heavier
    than the `⊙` MCP lamp, which is the state where the agent is genuinely
    missing tools — the salience the alarm colour exists to protect. Worst on
    a narrow terminal, where the drop ladder sheds what sits between them.
    """
    status = StatusLine(_dock(200))
    status.update(
        model_label="test/model",
        context_tokens=700_000,
        context_window=1_000_000,
        mcp=McpStatus(configured=3, connected=2, failed=True),
    )
    row = status.render_text(200)
    danger = theme_mod.semantic_color("danger").lower()

    reds = {
        row.plain[span.start : span.end]: span.style.bold
        for span in row.spans
        if isinstance(span.style, Style)
        and span.style.color is not None
        and span.style.color.triplet is not None
        and span.style.color.triplet.hex.lower() == danger
    }
    assert reds, "expected the row to carry red marks"
    assert all(reds.values()), (
        f"a red mark is unweighted while another is bold, which inverts the "
        f"band's attention order: {reds}"
    )
