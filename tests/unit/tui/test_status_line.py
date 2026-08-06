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

from rich.cells import cell_len

from local_operator.tui.widgets.status_line import (
    StatusLine,
    format_agents,
    format_context_usage,
    format_cost,
    format_cwd,
    format_duration,
    format_model_label,
    format_window,
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


def test_the_brand_glyph_survives_a_terminal_far_too_narrow() -> None:
    status, _clock = _full_band()
    row = status.render_text(10)
    assert row.plain.startswith("π")
    assert cell_len(row.plain) <= 10


# -- overflow priority -------------------------------------------------------


def test_the_jobs_counter_is_the_first_segment_to_go(monkeypatch) -> None:
    """Transient counters are cheapest to lose; the owner's numbers are not."""
    monkeypatch.setenv("HOME", "/Users/tester")
    status, _clock = _full_band()
    assert "3 agents" in status.render_text(200).plain

    # Walk down to the first width that sheds a segment rather than computing
    # it from the 200-cell row. That row is PADDED to the full width (the
    # right group is edge-aligned), so its length is 200 whatever the content
    # measures and `len(row) - 1` says nothing about when the band overflows.
    for width in range(200, 4, -1):
        tight = status.render_text(width).plain
        if "3 agents" not in tight:
            break
    else:  # pragma: no cover - the band always sheds something by width 5
        raise AssertionError("no width shed the subagent counter")

    # The counter goes FIRST: at the very width that drops it, everything the
    # operator actually steers by is still on screen.
    assert "41m1s" in tight
    assert "49.6%/1M" in tight
    assert "$12.40" in tight
    assert "Status band enrichment" in tight


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
        "subagents",
        "duration",
        "name",
        "cwd-full",  # shortened to its basename, not dropped
        "model-full",  # shortened to the bare model id, not dropped
        "cost",
        "context",
        "effort",
        "cwd-short",
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
