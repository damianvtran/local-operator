"""T4 (``tui.analytics``): the analytics panel's en copy is byte-identical to extraction time.

The fixture next to this file is a SNAPSHOT of the panel's copy as it was
before the extraction (``origin/main`` @ the slice's base): the page chrome
(title, scope, metric suffix), the empty state, every totals row label and
note, the search block's notes ladders and free/paid summaries, the table
tails and their column budgets, the per-model section's four states and its
legend, the chart titles/metas and the 13 hint-ladder tiers.

The tests below re-derive the live values and require every rendered string to
match the golden EXACTLY — the committed form of the slice's byte-identical
proof, re-runnable by review and QA, and the trip-wire for any later edit that
changes panel copy without regenerating the catalogue. They also pin what a
string comparison cannot: the COLUMN BUDGETS measure the catalogued tails
(the widths a later locale wave will change), and the two former module
constants (``COST_LEGEND`` aside) resolve through the catalogue rather than a
frozen English copy.
"""

from __future__ import annotations

import json
from pathlib import Path

from local_operator.analytics.model import (
    COMPONENT_KEYS,
    ModelRateRow,
    UsageAggregate,
    UsagePeriod,
)
from local_operator.i18n import catalogues, runtime
from local_operator.tui.costs import SearchSpendSnapshot
from local_operator.tui.widgets import analytics_panel as ap
from local_operator.tui.widgets.analytics_panel import (
    _model_rate_section,
    _row_overhead,
    build_report,
    rate_legend,
    search_component_text,
)

FIXTURE = Path(__file__).parent / "fixtures" / "tui_analytics_en_golden.json"
GOLDEN: dict[str, str] = json.loads(FIXTURE.read_text(encoding="utf-8"))


def _en(key: str, **params: object) -> str:
    """The message rendered from the FIXTURE source, never from the live file."""
    return runtime.render_message(GOLDEN[key], params, "en")


def _agg(*, calls: int = 100, ok_calls: int = 98) -> UsageAggregate:
    agg = UsageAggregate(
        calls=calls,
        ok_calls=ok_calls,
        input_tokens=500_000,
        output_tokens=120_000,
        cache_read_tokens=3_000_000,
        cache_write_tokens=80_000,
        reasoning_tokens=40_000,
        context_tokens=3_500_000,
        cost_micro=4_200_000,
        cost_known_calls=calls,
    )
    agg.components = {k: 0 for k in COMPONENT_KEYS}
    agg.components["conversation"] = 1_800_000
    agg.components["tool_results"] = 900_000
    return agg


def _text(
    agg: UsageAggregate,
    width: int = 90,
    *,
    daily: list[UsagePeriod] | None = None,
    monthly: list[UsagePeriod] | None = None,
) -> str:
    return "\n".join(line.plain for line in build_report(agg, width, daily=daily, monthly=monthly))


def test_catalogue_matches_the_pre_slice_golden() -> None:
    assert catalogues.load_catalogue("en", "tui.analytics") == GOLDEN
    assert len(GOLDEN) >= 90, "the golden is the whole panel's copy, not a sample"


def test_the_rate_legend_is_the_golden_sentence() -> None:
    scope = UsageAggregate(calls=3456, decode_calls=3)
    assert rate_legend(scope) == _en("tui.analytics.rate.legend", shown=3, total=3456)
    # Full coverage draws NO legend — the empty string is the contract the
    # footnote rides on (one line for the whole report, or none at all).
    assert rate_legend(UsageAggregate(calls=98, decode_calls=98)) == ""

    # Round-2 MAJOR: the word never pluralizes — base said "calls" for every
    # total — and the digits group exactly as base's f"{n:,}" did. At 1 the
    # one/other form said "1 call" and broke byte-identity; the 28-case battery
    # missed it because nothing sampled a total of 1.
    for total in (1, 2, 999, 1000, 1234, 12345):
        shown = 0 if total == 1 else 1
        assert (
            _en("tui.analytics.rate.legend", shown=shown, total=total)
            == f"decode rate over {shown} of {total:,} calls"
        )


def test_the_search_component_text_is_the_golden_rendering() -> None:
    # The unpriced split names the kind it actually counted.
    assert search_component_text(SearchSpendSnapshot(searches=5, unpriced_searches=5)) == _en(
        "tui.analytics.search.incl_unpriced_search", count=5, shown="5"
    )
    # A read-only session says read, never search (the kinds stay apart).
    assert search_component_text(SearchSpendSnapshot(reads=2)) == _en(
        "tui.analytics.search.incl_free_read", count=2, shown="2"
    )
    # A zero-count snapshot has nothing to say.
    assert search_component_text(SearchSpendSnapshot()) == ""


def test_the_model_section_states_are_the_golden_sentences() -> None:
    failed = _model_rate_section(ap.MODEL_RATES_FAILED, 90)
    assert failed is not None and _en("tui.analytics.model.failed") in failed.plain
    pending = _model_rate_section(ap.MODEL_RATES_PENDING, 90)
    assert pending is not None and _en("tui.analytics.model.pending") in pending.plain
    empty = _model_rate_section([], 90)
    assert empty is not None and _en("tui.analytics.model.empty") in empty.plain
    # The header carries title + meta from the catalogue.
    assert _en("tui.analytics.model.title") in failed.plain
    assert _en("tui.analytics.model.meta") in failed.plain


def test_the_model_row_tails_and_legend_are_the_golden_rendering() -> None:
    row = ModelRateRow(
        provider="anthropic",
        model_id="claude",
        calls=10,
        output_tokens=1000,
        decode_us=1_000_000,
        decode_tokens=500,
        decode_calls=8,
        wall_us=2_000_000,
        wall_tokens=1000,
        wall_calls=10,
    )
    block = _model_rate_section([row], 120)
    assert block is not None
    text = block.plain
    coverage = _en(
        "tui.analytics.table.coverage", covered=str(row.decode_calls), total=str(row.calls)
    )
    assert coverage in text
    # The two rate columns each carry a catalogued tail word.
    assert text.count(_en("tui.analytics.table.tok_s", value="")) == 2
    assert _en("tui.analytics.model.legend") in text


def test_the_totals_block_is_the_golden_rendering() -> None:
    text = _text(_agg())
    totals_line = next(
        line for line in text.splitlines() if _en("tui.analytics.totals.title") in line
    )
    assert totals_line.startswith(ap._SECTION_MARK + " " + _en("tui.analytics.totals.title"))
    # The meta carries the plural calls count and the measured suffix.
    assert _en("tui.analytics.totals.calls_meta", count="100") in totals_line
    assert _en("tui.analytics.totals.calls_measured") in totals_line
    # Failed-call count appears only because ok_calls != calls.
    assert _en("tui.analytics.totals.calls_failed", count="2") in totals_line
    for key in (
        "tui.analytics.totals.row_billed",
        "tui.analytics.totals.context_read",
        "tui.analytics.totals.fresh",
        "tui.analytics.totals.cache_read",
        "tui.analytics.totals.cache_write",
        "tui.analytics.totals.output",
        "tui.analytics.totals.cache_hit",
        "tui.analytics.totals.decode",
        "tui.analytics.totals.fresh_note",
        "tui.analytics.totals.cache_read_note",
        "tui.analytics.totals.cache_write_note",
        "tui.analytics.totals.cache_hit_note",
        "tui.analytics.totals.decode_note",
    ):
        assert _en(key) in text, key
    # The billed row composes its heads from the catalogue.
    assert _en("tui.analytics.totals.billed_note", context="3.5M", output="120k") in text
    # The input attribution section is keyed too.
    assert _en("tui.analytics.input.title") in text
    assert _en("tui.analytics.input.meta") in text


def test_the_empty_state_is_the_golden_rendering() -> None:
    text = _text(_agg(calls=0, ok_calls=0))
    assert _en("tui.analytics.empty.title") in text
    assert _en("tui.analytics.empty.hint") in text


def test_the_day_title_is_the_golden_plural() -> None:
    period = UsagePeriod(period="2026-10-10", model="m1", output_tokens=10, cost_micro=1)
    one = _text(_agg(), daily=[period])
    assert _en("tui.analytics.totals.days_with_usage", count=1, shown="1") in one
    assert _en("tui.analytics.chart.daily_empty") not in one
    # The rows-only meta states the metric clause from the catalogue.
    assert _en("tui.analytics.chart.metric_clause", active="cost", other="tokens") in one


def test_the_row_overhead_measures_the_catalogued_tails() -> None:
    groups = [("anthropic", _agg())]
    assert _row_overhead(groups, 200) > 0
    # The tails' widths come from the catalogue: the panel measures the SAME
    # renderings it appends, so a locale wave that widens a tail moves the
    # column budget with it rather than overflowing it.
    assert ap.cell_len(_en("tui.analytics.table.tokens", value="")) == len(" tokens")
    assert ap.cell_len(_en("tui.analytics.table.tok_s", value="")) == len(" tok/s")
    assert ap.cell_len(_en("tui.analytics.table.calls", value="")) == len(" calls")
    assert ap.cell_len(_en("tui.analytics.table.cache", value="")) == len(" cache")


def test_the_subagent_suffix_is_the_golden_plural() -> None:
    assert _en("tui.analytics.table.subagent_suffix", count=1, shown="1") == "+1 subagent"
    assert _en("tui.analytics.table.subagent_suffix", count=12, shown="12") == "+12 subagents"


def test_counts_render_raw_digits_at_and_above_one_thousand() -> None:
    """Round-1 MAJOR-1: counts are RAW digits, never ICU-grouped.

    The pre-slice f-strings printed raw counts (``f"{n} calls"``); ICU ``#``
    and numeric params format locale-grouped (``1,234``), which broke the
    byte-identity invariant for counts >= 1000. Every count-bearing message now
    interpolates a STRINGIFIED count through a plain ``{shown}`` substitution,
    keeping the numeric ``count`` only for plural SELECTION.

    This pins the class at the 999/1000/1234/12345 boundary for every affected
    message — the boundary the first capture battery (all counts <= 100) could
    not see, which is how the drift slipped round 1.
    """

    def msg(key: str, n: int, **extra: object) -> str:
        params: dict[str, object] = {"count": n, "shown": str(n)}
        params.update(extra)
        return _en(key, **params)

    for n in (1, 2, 999, 1000, 1234, 12345):
        word_s = "search" if n == 1 else "searches"
        word_r = "read" if n == 1 else "reads"
        word_sa = "subagent" if n == 1 else "subagents"
        word_d = "day" if n == 1 else "days"
        assert msg("tui.analytics.search.counted_search", n) == f"{n} {word_s}"
        assert msg("tui.analytics.search.counted_read", n) == f"{n} {word_r}"
        assert msg("tui.analytics.search.incl_free_search", n) == f"incl. {n} free {word_s}"
        assert msg("tui.analytics.search.incl_free_read", n) == f"incl. {n} free {word_r}"
        assert msg("tui.analytics.search.incl_unpriced_search", n) == f"incl. {n} unpriced {word_s}"
        assert msg("tui.analytics.search.incl_unpriced_read", n) == f"incl. {n} unpriced {word_r}"
        assert (
            msg("tui.analytics.search.note_search_full", n) == f"{n} {word_s} · no published price"
        )
        assert msg("tui.analytics.search.note_search_unpriced", n) == f"{n} {word_s} · unpriced"
        assert (
            msg("tui.analytics.search.note_search_partial", n, unpriced="3")
            == f"{n} {word_s} · 3 unpriced"
        )
        assert msg("tui.analytics.search.note_read_full", n) == f"{n} {word_r} · no published price"
        assert msg("tui.analytics.search.note_read_unpriced", n) == f"{n} {word_r} · unpriced"
        assert (
            msg("tui.analytics.search.note_read_partial", n, unpriced="3")
            == f"{n} {word_r} · 3 unpriced"
        )
        assert msg("tui.analytics.table.subagent_suffix", n) == f"+{n} {word_sa}"
        assert msg("tui.analytics.totals.days_with_usage", n) == f"{n} {word_d} with usage"
        assert _en("tui.analytics.totals.calls_meta", count=str(n)) == f"{n} calls"

    # Live-path wiring: the totals meta renders the raw count (1234, not
    # 1,234), and the search component keeps its digits at 1234 searches.
    text = _text(_agg(calls=1234, ok_calls=1200), 90)
    assert _en("tui.analytics.totals.calls_meta", count="1234") in text
    # The totals meta keeps RAW digits; the rate legend above it is the one
    # count that legitimately groups (the pre-slice code grouped it too).
    assert "1234 calls" in text
    component = search_component_text(SearchSpendSnapshot(searches=1234))
    assert component == "incl. 1234 free searches"
    assert "1,234" not in component
    # The unpriced branch carries its own count, raw, at the same magnitude.
    unpriced_component = search_component_text(
        SearchSpendSnapshot(searches=1234, unpriced_searches=1234)
    )
    assert unpriced_component == "incl. 1234 unpriced searches"
    # And the counted tally (the notes ladder's first rung source) at 1234.
    assert ap._counted_kinds(SearchSpendSnapshot(searches=1234)) == "1234 searches"
