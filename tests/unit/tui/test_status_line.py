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

import re

from rich.cells import cell_len

from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.status_line import (
    ICON_MCP,
    ICON_MODEL,
    McpStatus,
    StatusLine,
    _MIN_GROUP_GAP,
    format_agents,
    format_context_usage,
    format_cost,
    format_cwd,
    format_duration,
    format_mcp,
    format_model_label,
    format_window,
    mcp_semantic,
)


class FakeDock:
    """The three things StatusLine asks of its widget (width, paint, timer).

    A real ``Static`` needs a running app before ``set_interval`` works, and
    these tests drive the streaming transitions that arm the spinner.
    """

    def __init__(self, width: int = 80) -> None:
        self.width = width
        self.painted = None
        self.intervals: list[tuple[float, object]] = []

    @property
    def size(self):  # noqa: ANN201 - mirrors textual's geometry duck type
        return type("Size", (), {"width": self.width})()

    def update(self, renderable) -> None:  # noqa: ANN001
        self.painted = renderable

    def set_interval(self, interval: float, callback):  # noqa: ANN001, ANN201
        self.intervals.append((interval, callback))
        return type("Timer", (), {"stop": lambda self: None})()


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
    status = StatusLine(FakeDock(width), clock=clock)
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


def test_model_label_sheds_only_its_provider_prefixes() -> None:
    assert format_model_label("openrouter/moonshotai/kimi-k2", short=False) == (
        "openrouter/moonshotai/kimi-k2"
    )
    assert format_model_label("openrouter/moonshotai/kimi-k2", short=True) == "kimi-k2"
    assert format_model_label("ollama", short=True) == "ollama"


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


def test_the_label_the_user_typed_is_the_first_segment_to_go(monkeypatch) -> None:
    """The conversation name goes first, and every NUMBER outlives it.

    The name is a label the user typed and already knows; the counters and live
    figures beside it are not re-derivable at a glance. An earlier order shed the
    subagent count first, which meant a three-agent fan-out went invisible while
    a title the user had chosen was still on screen.
    """
    monkeypatch.setenv("HOME", "/Users/tester")
    status, _clock = _full_band()
    assert "Status band enrichment" in status.render_text(200).plain

    # Walk down to the first width that sheds it rather than computing it from
    # the 200-cell row. That row is PADDED to the full width (the right group is
    # edge-aligned), so its length is 200 whatever the content measures and
    # `len(row) - 1` says nothing about when the band overflows.
    for width in range(200, 4, -1):
        tight = status.render_text(width).plain
        if "Status band enrichment" not in tight:
            break
    else:  # pragma: no cover - the band always sheds something by width 5
        raise AssertionError("no width shed the conversation name")

    # At the very width that drops the name, everything numeric survives.
    assert "3 agents" in tight
    assert "41m1s" in tight
    assert "49.6%/1M" in tight
    assert "$12.40" in tight


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
    probes = {
        "subagents": "3 agents",
        "duration": "41m1s",
        "name": "Status band enrichment",
        "cost": "$12.40",
        "context": "49.6%/1M",
        "effort": "high",
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
        "name",  # a label the user typed and already knows
        "duration",  # re-derivable from the transcript
        "subagents",  # a counter, but not re-derivable without scrolling
        "cwd-full",  # shortened to its basename, not dropped
        "model-full",  # shortened to the bare model id, not dropped
        "effort",  # a static setting: it does not change while they watch
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
    status = StatusLine(FakeDock(200), clock=clock)
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
    status = StatusLine(FakeDock(200), clock=clock)
    status.update(streaming=True)
    clock.advance(9)
    assert "9s" in status.render_text(200).plain


def test_a_redundant_streaming_false_cannot_bank_the_same_interval_twice() -> None:
    """The prompt worker's ``finally`` repeats ``streaming=False`` after
    agent_end already sent it; double-banking would inflate every duration."""
    clock = FakeClock()
    status = StatusLine(FakeDock(200), clock=clock)
    status.update(streaming=True)
    clock.advance(10)
    status.update(streaming=False)
    clock.advance(10)
    status.update(streaming=False)
    assert "10s" in status.render_text(200).plain


def test_no_duration_segment_before_anything_has_run() -> None:
    clock = FakeClock()
    status = StatusLine(FakeDock(200), clock=clock)
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
    status = StatusLine(FakeDock(200))
    status.update(model_label="test/model", mcp=McpStatus())
    assert ICON_MCP not in status.render_text(200).plain


def test_mcp_glyph_is_a_single_cell_like_every_other_segment_icon() -> None:
    """The band's layout is measured arithmetic. A two-cell glyph would drift
    the right group's edge by a column on terminals that render it wide, which
    is why the reference's ``⊙`` was measured rather than assumed."""
    assert cell_len(ICON_MCP) == 1


def test_a_failed_server_wins_over_the_ones_that_did_connect() -> None:
    """A PARTIAL failure is the dangerous outcome: green beside a count of 2 on
    a machine where the third server died reads as "all good", and the user then
    spends the turn wondering why the agent cannot reach those tools."""
    assert mcp_semantic(McpStatus(configured=3, connected=2, failed=True)) == "danger"
    assert mcp_semantic(McpStatus(configured=3, connected=3)) == "success"
    # Configured, nothing up, nothing failed: the startup gate's normal state on
    # every launch, so it gets no colour at all.
    assert mcp_semantic(McpStatus(configured=3, connected=0)) == "dim"


def test_only_the_mcp_glyph_carries_the_state_colour() -> None:
    """Tinting ``2 MCP`` danger reads as "the number 2 is wrong". The glyph is
    the status lamp; the count beside it stays plain foreground."""
    status = StatusLine(FakeDock(200))
    status.update(model_label="test/model", mcp=McpStatus(configured=3, connected=2, failed=True))
    fills = _fills(status.render_text(200))
    danger = theme_mod.semantic_color("danger").lower()
    fg = theme_mod.semantic_color("fg").lower()
    assert fills[f"{ICON_MCP} "] == danger
    assert fills["2 MCP"] == fg


def test_the_mcp_segment_outlives_every_other_droppable_segment() -> None:
    """It is the reference's ``flexShrink={0}`` indicator, and the ladder's last
    rung. Two reasons, both asserted here by outcome: it is the cheapest segment
    in the band to keep, and its failure branch is the only alarm the band can
    raise — a cramped terminal is exactly where hiding it would make a user
    conclude the tools were never configured.
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
