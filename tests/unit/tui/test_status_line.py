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
from typing import cast

from rich.cells import cell_len
from textual.widgets import Static

from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.status_line import (
    _DROP_LADDER,
    _DROP_LADDER_ESTIMATE,
    _DROP_LADDER_QUIET,
    _DROP_LADDER_QUIET_ESTIMATE,
    _MIN_GROUP_GAP,
    _SPINNER_FRAMES,
    ICON_CWD,
    ICON_MCP,
    ICON_MODEL,
    McpStatus,
    StatusLine,
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


def test_the_quiet_ladder_moves_mcp_without_promoting_the_widest_alarm() -> None:
    """One ordering, two positions for one rung — not two hand-maintained
    ladders that can drift apart on the next reordering.

    With one caveat the first version got wrong: lifting `mcp` out of last place
    leaves whatever followed it at the end, and that is `approvals` — the 14-cell
    segment the full ladder sheds FIRST precisely because dropping it buys the
    most width. Left last, it outlived the context number in the quiet band,
    inverting the ladder's whole argument. The narrowest survivor goes last.
    """
    assert drop_ladder(McpStatus(configured=2, connected=1, failed=True)) is _DROP_LADDER
    assert drop_ladder(McpStatus(configured=2, connected=2)) is _DROP_LADDER_QUIET
    assert drop_ladder(McpStatus()) is _DROP_LADDER_QUIET
    assert _DROP_LADDER[-1] == "mcp"
    assert _DROP_LADDER_QUIET.index("mcp") == _DROP_LADDER_QUIET.index("cwd") - 1
    # The quiet band's last survivor is the context number, not the widest alarm.
    assert _DROP_LADDER_QUIET[-1] == "context"
    assert _DROP_LADDER_QUIET.index("approvals") < _DROP_LADDER_QUIET.index("context")
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


def test_no_ladder_variant_strands_the_widest_alarm_last() -> None:
    """The invariant every rung promotion has to preserve, checked on all four.

    ``approvals`` is the widest segment in the band (14 cells against the
    context number's ~9 and a basename's ~7), which is exactly why the authored
    order sheds it FIRST. Promoting any rung out of last place leaves whatever
    followed it at the end, and in this ladder that is reliably ``approvals`` —
    stranded there it outlives the narrow segments the promotion was meant to
    protect, inverting the ladder's whole argument.

    It was written inline for the mcp promotion and then silently not
    re-applied when the context promotion was added, which shipped a quiet
    estimate ladder ending ``… -> cwd -> approvals``. Asserted across every
    variant so the next promotion cannot reintroduce it either.
    """
    variants = {
        "full": _DROP_LADDER,
        "quiet": _DROP_LADDER_QUIET,
        "full+estimate": _DROP_LADDER_ESTIMATE,
        "quiet+estimate": _DROP_LADDER_QUIET_ESTIMATE,
    }
    for name, ladder in variants.items():
        assert ladder[-1] != "approvals", f"{name} strands the widest alarm last"
        # A promotion reorders; it never adds or loses a rung.
        assert sorted(ladder) == sorted(_DROP_LADDER), f"{name} changed the rung set"

    # And the specific tails, so a reorder that happens to dodge the assertion
    # above still has to be argued for.
    assert _DROP_LADDER[-1] == "mcp"
    assert _DROP_LADDER_QUIET[-1] == "context"
    assert _DROP_LADDER_ESTIMATE[-1] == "mcp"
    # The quiet band with a pre-turn estimate: the working directory is the last
    # thing standing, which is D21's argument carried to its conclusion.
    assert _DROP_LADDER_QUIET_ESTIMATE[-1] == "cwd"
