"""The By-model rate table: its three inputs, and the two rates it must not confuse.

The section is the ONE part of the report whose data arrives LATER than the
screen (``model_rates()`` groups the raw ledger, seconds on a large one), so the
state machine is the thing worth pinning: no section when the caller does no
read, a stated wait while it is in flight, an explicit empty sentence when the
answer has no rows, and the table itself once it lands. "No answer yet" and "no
rows" are different facts and a reader must be able to tell them apart.

The second thing pinned here is the pair of rates. ``decode`` is a measured
generation window and ``wall`` is the whole call including the first-token wait;
a model can legitimately have one and not the other, and the cells must render
that asymmetry rather than one number twice.
"""

from __future__ import annotations

from local_operator.analytics.model import ModelRateRow, UsageAggregate
from local_operator.tui.widgets.analytics_panel import (
    MODEL_RATES_FAILED,
    MODEL_RATES_PENDING,
    build_report,
)


def _report_text(aggregate: UsageAggregate, *, model_rates, width: int = 120) -> str:
    return "\n".join(line.plain for line in build_report(aggregate, width, model_rates=model_rates))


def _row(*, decode_us: int, decode_tokens: int, decode_calls: int) -> ModelRateRow:
    """A row with a known decode window and a wall window four times as long."""
    return ModelRateRow(
        provider="deepseek",
        model_id="deepseek-flash",
        calls=10,
        output_tokens=1000,
        decode_us=decode_us,
        decode_tokens=decode_tokens,
        decode_calls=decode_calls,
        wall_us=4_000_000,
        wall_tokens=1000,
        wall_calls=10,
    )


def test_no_section_when_the_caller_does_not_do_the_read() -> None:
    """``None`` is "this caller has no such table", not "the table is empty".

    Every caller that composed a report before this feature passes nothing, and
    their output must be unchanged — a section appearing with a wait or an empty
    sentence in an unrelated frame would be a regression in those frames.
    """
    text = _report_text(UsageAggregate(calls=1), model_rates=None)
    assert "By model" not in text


def test_a_pending_read_says_so_rather_than_drawing_an_empty_table() -> None:
    text = _report_text(UsageAggregate(calls=1), model_rates=MODEL_RATES_PENDING)
    assert "By model" in text
    assert "reading the ledger" in text


def test_an_empty_answer_has_its_own_sentence() -> None:
    text = _report_text(UsageAggregate(calls=1), model_rates=[])
    assert "By model" in text
    assert "no per-model rows" in text
    assert "reading the ledger" not in text


def test_a_failed_read_is_not_rendered_as_an_empty_ledger() -> None:
    """A broken query must not claim the reader's ledger has no rows in it.

    The two are different facts and only one of them is about the ledger: "no
    per-model rows in this window" is a statement about what the operator ran,
    while a failed read knows nothing about it (review round 1, minor 3).
    """
    text = _report_text(UsageAggregate(calls=1), model_rates=MODEL_RATES_FAILED)
    assert "By model" in text
    assert "per-model read failed" in text
    assert "the rest of this report is unaffected" in text
    assert "no per-model rows" not in text
    assert "reading the ledger" not in text


def test_the_section_names_a_scope_the_headline_shares() -> None:
    """The meta must not leave the reader guessing which window they are on.

    The Totals above this table come from an unbounded aggregate, so the table is
    all-time too, and the meta says so — a 30-day table beside an all-time
    headline invites a comparison that is quietly wrong (review round 1, MAJOR 2).
    """
    text = _report_text(UsageAggregate(calls=1), model_rates=[])
    header = next(line for line in text.splitlines() if "By model" in line)
    assert "all time" in header
    # And the legend is NOT in the meta: the composed line is cropped at the
    # widths this panel is used at, and the half that got cut was the explanation
    # of the unknown marks (review round 1, minor 2).
    assert "tok/s decode =" not in header


def _model_row_line(text: str) -> str:
    """The By-model table's own row line, so an assertion cannot read a dash that
    belongs to another section.

    The rest of a report legitimately prints ``—`` for things this fixture leaves
    empty (an unpriced cost, a zero-denominator cache rate), so a whole-report
    substring check would pass for the wrong reason.
    """
    return next(line for line in text.splitlines() if "deepseek/deepseek-flash" in line)


def test_both_rates_are_printed_and_an_absent_one_is_a_dash() -> None:
    """A row can have a wall rate and no decode rate — the two are different.

    The decode window here is 1 s for 1 000 tokens (1 000 tok/s) against a 4 s
    wall window for the same tokens (250 tok/s), so a renderer that used one
    number twice fails on the values as well as on the shape.
    """
    covered = _row(decode_us=1_000_000, decode_tokens=1_000, decode_calls=10)
    row = _model_row_line(_report_text(UsageAggregate(calls=10), model_rates=[covered]))
    assert "1k tok/s" in row  # decode: 1000 tokens in 1 s, scaled by format_tokens
    assert "250 tok/s" in row  # wall: the same tokens over 4 s
    assert "10/10 calls" in row
    assert "—" not in row

    # A ledger that predates the metric: the SAME row shape with no window at
    # all. The decode cell is a dash while the wall cell keeps its number.
    unwindowed = _row(decode_us=0, decode_tokens=0, decode_calls=0)
    row = _model_row_line(_report_text(UsageAggregate(calls=10), model_rates=[unwindowed]))
    assert "— tok/s" in row
    assert "250 tok/s" in row
    assert "0/10 calls" in row


def test_a_partial_ledger_states_coverage_per_row() -> None:
    """Coverage is a fraction of that row's calls, not a global percentage."""
    partial = _row(decode_us=1_000_000, decode_tokens=1_000, decode_calls=4)
    text = _report_text(UsageAggregate(calls=10), model_rates=[partial])
    assert "4/10 calls" in text
