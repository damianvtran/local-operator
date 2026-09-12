"""Ranking as a module of its own: stdlib-only, importable without a terminal.

``tests/unit/tui/test_model_picker.py`` still exercises this behaviour through
the widget and is deliberately left importing it from there — that file passing
unchanged is what proves the re-export. What is asserted HERE is what the move
made true and the widget cannot check about itself: that the order is available
to a non-TUI caller, at no textual/rich cost, and that the measured regressions
each docstring records are still held at the new address.
"""

from __future__ import annotations

import subprocess
import sys

from local_operator.model.ranking import (
    _MINOR_VERSION_PATTERN,
    _VERSION_PATTERN,
    ModelRow,
    _score,
    _version_key,
    rank_rows,
)


def _row(selector: str, **kwargs) -> ModelRow:
    provider, _, model_id = selector.partition("/")
    return ModelRow(provider=provider, model_id=model_id, **kwargs)


def test_importing_ranking_pulls_in_neither_textual_nor_rich():
    """The reason the move happened, asserted rather than assumed.

    The mobile daemon ranks its catalogue through this module. Importing the
    widget instead costs ~0.48 s and puts a whole terminal toolkit on the import
    path of a process that renders no terminal — so an import of textual or rich
    from here is the regression, not a style preference. A subprocess because
    the modules are already resident in this suite's own process.
    """
    probe = (
        "import sys; import local_operator.model.ranking; "
        "print(sorted(m for m in sys.modules if m in {'rich', 'textual'}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_widget_reexports_the_very_same_objects():
    """Not merely equal names — the SAME objects, so isinstance holds across both.

    ``tui/app.py`` builds a ``ModelRow`` imported from the widget and hands it to
    a ranker imported from here; two distinct dataclasses with one name would
    typecheck, pass most tests, and fail on the one path that compares them.
    """
    from local_operator.tui.widgets import model_picker

    assert model_picker.ModelRow is ModelRow
    assert model_picker.rank_rows is rank_rows
    assert model_picker._score is _score
    assert model_picker._version_key is _version_key
    assert model_picker._VERSION_PATTERN is _VERSION_PATTERN
    assert model_picker._MINOR_VERSION_PATTERN is _MINOR_VERSION_PATTERN


def test_direct_route_outranks_the_aggregated_one_for_the_same_model():
    """The phone's bug in one assertion: same model, two routes, direct first.

    The direct provider here sorts AFTER the aggregator alphabetically (``xai``
    > ``openrouter``), which is deliberate: with ``anthropic`` as the direct
    route this passes even with the aggregator tier deleted, because the
    provider-name rung alone produces the expected order. A fixture that cannot
    distinguish the rule from a coincidence does not test the rule.
    """
    rows = [
        _row("openrouter/x-ai/grok-4", aggregated=True),
        _row("xai/grok-4"),
    ]
    assert [row.selector for row in rank_rows(rows, "grok")] == [
        "xai/grok-4",
        "openrouter/x-ai/grok-4",
    ]
    # And with no query at all, which is the state the sheet opens in.
    assert [row.selector for row in rank_rows(rows, "")] == [
        "xai/grok-4",
        "openrouter/x-ai/grok-4",
    ]


def test_connected_rows_outrank_unconnected_ones():
    """A model that runs now beats one that needs a login, always."""
    rows = [_row("anthropic/claude-opus-5", connected=False), _row("openai/gpt-5")]
    assert [row.selector for row in rank_rows(rows, "")] == [
        "openai/gpt-5",
        "anthropic/claude-opus-5",
    ]


def test_substring_matches_beat_subsequence_ones():
    """``opus`` is a SUBSEQUENCE of ``claude-sonnet-4`` (o-p-u-s in anthropic/
    claude/sonnet), so ranking the fallback first led the list with a different
    model than the one whose name the user typed."""
    rows = [_row("anthropic/claude-sonnet-4"), _row("anthropic/claude-opus-5")]
    assert [row.selector for row in rank_rows(rows, "opus")] == ["anthropic/claude-opus-5"]


def test_subsequence_fallback_still_resolves_typos_and_elisions():
    """Keeping the fallback is what makes ``anthopus`` and ``sonnet4`` land."""
    rows = [_row("anthropic/claude-opus-5"), _row("anthropic/claude-sonnet-4")]
    # The fallback RANKS rather than filters to one, so the assertion is on the
    # row it leads with — which is the whole point of the tier.
    assert rank_rows(rows, "anthopus")[0].selector == "anthropic/claude-opus-5"
    assert rank_rows(rows, "sonnet4")[0].selector == "anthropic/claude-sonnet-4"


def test_kimi_k2_version_comes_from_the_version_not_the_serial():
    """The lookbehind excludes digits and dots ONLY.

    Excluding word characters too looks tidier and silently broke every id that
    glues its version to a letter: ``kimi-k2`` then matched nothing, took its
    version from the ``0905`` serial, and outranked ``kimi-k3``.
    """
    rows = [_row("moonshot/kimi-k2-0905"), _row("moonshot/kimi-k3")]
    assert [row.selector for row in rank_rows(rows, "kimi")] == [
        "moonshot/kimi-k3",
        "moonshot/kimi-k2-0905",
    ]
    # The key's version rung is NEGATED so a plain ascending sort puts the
    # newest first: k3 must therefore sort LOWER than k2, not higher.
    assert (
        _version_key(_row("moonshot/kimi-k3"))[0] < _version_key(_row("moonshot/kimi-k2-0905"))[0]
    )


def test_a_dated_snapshot_is_not_read_as_a_minor_version():
    """``opus-4-1`` is 4.1; ``sonnet-4-20250514`` is a DATE, not version 4.20250514.

    The minor-version rewrite is capped at two digits for exactly this reason —
    uncapped, every dated snapshot outranks every real version.
    """
    assert _MINOR_VERSION_PATTERN.sub(r"\1.\2", "claude-opus-4-1") == "claude-opus-4.1"
    assert (
        _MINOR_VERSION_PATTERN.sub(r"\1.\2", "claude-sonnet-4-20250514")
        == "claude-sonnet-4-20250514"
    )
    rows = [_row("anthropic/claude-sonnet-4-20250514"), _row("anthropic/claude-opus-4-1")]
    assert [row.selector for row in rank_rows(rows, "claude")][0] == "anthropic/claude-opus-4-1"


def test_version_is_the_first_number_not_the_largest():
    """Position is the reliable signal here, magnitude is not.

    Taking the largest number looks equivalent and is not: ``kimi-k2-0905``
    carries 2 and 905, so it scored 905 and led a list where ``kimi-k3`` came
    ninth. Every id in this catalogue puts the family version first and its
    serials and dates after.
    """
    assert _VERSION_PATTERN.findall("kimi-k2-0905")[0] == "2"
    assert _VERSION_PATTERN.findall("gpt-4.1-mini") == ["4.1"]
    rows = [_row("openai/gpt-4"), _row("openai/gpt-4.1")]
    assert [row.selector for row in rank_rows(rows, "gpt")][0] == "openai/gpt-4.1"


def test_a_query_matching_nothing_returns_nothing():
    """An empty result is a real answer; the picker says "no matching models"."""
    assert rank_rows([_row("openai/gpt-5")], "zzzz") == []


def test_score_reports_none_for_a_non_subsequence():
    assert _score("anthropic/claude-opus-5", "zzz") is None
    assert _score("anthropic/claude-opus-5", "opus") is not None
