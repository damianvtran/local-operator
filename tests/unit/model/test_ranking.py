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
    model than the one whose name the user typed.

    SUBSTRING LEADS; the subsequence row is RETAINED below it. The old assertion
    here was `== ["anthropic/claude-opus-5"]`, which passed because the base code
    chose `pool = exact or fuzzy` and EVICTED every fuzzy row whenever one exact
    match existed. That eviction is the R1-1 defect — it drops rows the user could
    see from the answer entirely — so membership is now "matched anything at all"
    and this test pins the ORDER (the substring match first), not the absence of
    the subsequence row.
    """
    rows = [_row("anthropic/claude-sonnet-4"), _row("anthropic/claude-opus-5")]
    assert [row.selector for row in rank_rows(rows, "opus")][0] == "anthropic/claude-opus-5"


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


def test_the_radient_auto_router_leads_the_auto_query_and_the_empty_one():
    """The measured ordering the operator asked for, in both sort branches.

    BEFORE this rule the picker led with the two OpenRouter rows on the query
    ``auto``, because the ``-score`` rung put them there: ``openrouter/auto``
    scores 7 to ``radient/auto``'s 6 (the OpenRouter id carries the query twice as
    a contiguous run) and the version keys collide at ``(-0.0, -0.0, …)``. The
    OpenRouter rows thus SCORE HIGHER, which is exactly why the preference rung
    sits above ``-score``: a rung below it changes nothing here. Asserted with no
    query as well, because ``/model`` opens on the empty sort and the two surfaces
    must agree.
    """
    rows = [
        _row("openrouter/openrouter/auto", aggregated=True),
        _row("openrouter/openrouter/auto-beta", aggregated=True),
        _row("radient/auto", aggregated=True),
        _row("radient/openrouter/auto-beta", aggregated=True),
    ]
    for query in ("auto", ""):
        assert rank_rows(rows, query)[0].selector == "radient/auto", query


def test_the_preference_is_query_independent_not_scoped_to_the_literal_auto():
    """The rung lifts the route for any query it matches, and that is the intent.

    A partial query is the only place the width shows: ``aut`` scores
    ``openrouter/openrouter/auto`` 5 to ``radient/auto``'s 4, so a rung that only
    fired on the literal ``auto`` would rank the OpenRouter row first here. The
    operator's reading ("Radient Auto comes first at the top") is that it lead
    whenever it is offered, so this pins the wider behaviour rather than leaving it
    as an undocumented consequence of the insertion point. See
    ``_preferred_router_rank``.
    """
    rows = [
        _row("openrouter/openrouter/auto", aggregated=True),
        _row("radient/auto", aggregated=True),
    ]
    assert _score("openrouter/openrouter/auto", "aut") == 5
    assert _score("radient/auto", "aut") == 4
    assert [row.selector for row in rank_rows(rows, "aut")][0] == "radient/auto"


def test_the_preference_lifts_only_the_radient_auto_route():
    """``radient/openrouter/auto`` is a router id reached through Radient's namespace
    and is deliberately NOT elevated: it stays below the OpenRouter rows.

    This is why the rung names the ROUTE rather than reusing ``is_meta_route_id``,
    which is true of this id too (see ``_preferred_router_rank``) — reusing it
    would lift this row as well, a second behavioural change the preference did not
    ask for.
    """
    rows = [
        _row("openrouter/openrouter/auto", aggregated=True),
        _row("radient/openrouter/auto", aggregated=True),
        _row("radient/auto", aggregated=True),
    ]
    assert [row.selector for row in rank_rows(rows, "auto")] == [
        "radient/auto",
        "openrouter/openrouter/auto",
        "radient/openrouter/auto",
    ]


def test_the_preference_is_provider_scoped_not_a_bare_id_test():
    """``ollama/auto`` is a model a user can simply have and must not be lifted.

    The same gate ``is_meta_route_id`` carries: the router id means something
    only inside the Radient namespace, so a bare ``model_id == "auto"`` test would
    silently reorder a local model the user named.
    """
    rows = [_row("ollama/auto"), _row("openai/gpt-5")]
    assert [row.selector for row in rank_rows(rows, "auto")] == [
        "ollama/auto",
    ]
    # And, with a same-tier neighbour that sorts before it alphabetically, it
    # keeps the ordinary order rather than being raised.
    rows = [_row("anthropic/auto"), _row("ollama/auto")]
    assert [row.selector for row in rank_rows(rows, "auto")] == [
        "anthropic/auto",
        "ollama/auto",
    ]


def test_the_preference_does_not_resurrect_a_dropped_or_tiered_row():
    """The preference is a sort rung, not a filter bypass.

    It cannot lift a row the caller filtered out (ranking never sees it), and it
    cannot lift an UNCONNECTED ``radient/auto`` above a connected row: the
    connected tier stays first, so a login-required router does not lead a list
    of models that run.
    """
    rows = [
        _row("radient/auto", aggregated=True, connected=False),
        _row("openai/auto"),
    ]
    assert [row.selector for row in rank_rows(rows, "auto")] == [
        "openai/auto",
        "radient/auto",
    ]
    # A decision-only row is dropped before any ordering, preference included —
    # so even a query that matches nothing else leaves ``[]``, not the router.
    assert [row.selector for row in rank_rows([_row("typesafe/auto")], "auto")] == []


def test_a_query_matching_nothing_returns_nothing():
    """An empty result is a real answer; the picker says "no matching models"."""
    assert rank_rows([_row("openai/gpt-5")], "zzzz") == []


def test_score_reports_none_for_a_non_subsequence():
    assert _score("anthropic/claude-opus-5", "zzz") is None
    assert _score("anthropic/claude-opus-5", "opus") is not None


def test_the_listings_human_name_is_a_match_target():
    """REPRODUCTION (D2): `grok 4.7` must resolve the row it names.

    The operator typed the model's HUMAN name, which the listing publishes
    (`SpaceXAI: Grok 4.7`) while the selector glues the same words differently
    (`x-ai/grok-4.7`). Scoring the selector alone returned an EMPTY list for a
    model that was demonstrably in the catalogue, and the same for
    `GPT 6 Luna` / `openai/gpt-6-luna`.
    """
    rows = [
        ModelRow(
            provider="openrouter",
            model_id="x-ai/grok-4.7",
            listing_name="SpaceXAI: Grok 4.7",
            aggregated=True,
        ),
        ModelRow(
            provider="openrouter",
            model_id="openai/gpt-6-luna",
            listing_name="OpenAI: GPT-6 Luna",
            aggregated=True,
        ),
    ]
    assert [row.selector for row in rank_rows(rows, "grok 4.7")] == ["openrouter/x-ai/grok-4.7"]
    assert [row.selector for row in rank_rows(rows, "gpt 6 luna")] == [
        "openrouter/openai/gpt-6-luna"
    ]
    # Case and separator spelling of the HUMAN name both resolve, because both
    # sides normalise the same way.
    for query in ("Grok 4.7", "spacexai grok", "GPT-6 Luna", "openai gpt 6"):
        assert rank_rows(rows, query), query


def test_a_name_match_never_outranks_an_identical_selector_match():
    """The new target joins the MATCH only; it must not disturb the ORDER.

    Two rows whose selectors both contain the query keep the same relative
    order they had before `listing_name` was consulted, so adding a name cannot
    silently promote an aggregator's row over a direct provider's.
    """
    rows = [
        _row("openrouter/x-ai/grok-4", aggregated=True),
        _row("xai/grok-4"),
    ]
    assert [row.selector for row in rank_rows(rows, "grok")] == [
        "xai/grok-4",
        "openrouter/x-ai/grok-4",
    ]


def test_the_bare_selector_still_resolves_unchanged():
    """The pre-existing spelling must not regress: `x-ai/grok-4.7` still lands."""
    rows = [
        ModelRow(
            provider="openrouter",
            model_id="x-ai/grok-4.7",
            listing_name="SpaceXAI: Grok 4.7",
            aggregated=True,
        )
    ]
    assert rank_rows(rows, "x-ai/grok-4.7")
    assert rank_rows(rows, "openrouter/x-ai/grok-4.7")


def test_match_key_normalises_punctuation_and_case_to_words():
    """One rule for both sides of the comparison, asserted directly.

    A per-spelling special case is what this test exists to prevent: the query
    and the row must go through the SAME function, or `grok 4.7` matches one
    spelling and not the other.
    """
    from local_operator.model.ranking import _match_key

    assert _match_key("SpaceXAI: Grok 4.7") == "spacexai grok 4 7"
    assert _match_key("x-ai/grok-4.7") == "x ai grok 4 7"
    assert _match_key("  grok   4.7 ") == "grok 4 7"
    assert _match_key("GPT-6 Luna") == "gpt 6 luna"


def test_adding_a_name_target_cannot_evict_a_row_the_selector_matched():
    """R1-1/B1: widening a match target must never REMOVE a row.

    The old code chose `pool = exact or fuzzy`, and `exact` was decided by "the
    needle is a substring of ANY target". Adding ``listing_name`` widened
    `exact`, so a row that matched only under the selector-only test was dropped
    from the answer rather than demoted — measured on the live listing: query
    `banana` lost five direct-provider rows, `older` lost eight of nine.

    The minimal shape: a direct row that matches the needle only as a
    SUBSEQUENCE of its selector, against an aggregator whose listing_name
    contains the needle literally. The direct row must still be present.
    """
    rows = [
        ModelRow("direct-prov", "banana-flash", "Banana Flash", 1000, 0.0, 0.0, True),
        # `banana-flash` carries the needle only as a subsequence (b-a-n-a-n-a… no:
        # as an actual substring) — use a real eviction shape: the needle is a
        # substring of the NAME but only a subsequence of the selector.
        ModelRow(
            "openrouter",
            "google/gemini-3.1-flash-image",
            "google/gemini-3.1-flash-image",
            1000,
            0.0,
            0.0,
            True,
            aggregated=True,
            listing_name="Google: Nano Banana 2 (Gemini 3.1 Flash Image)",
        ),
    ]
    # `banana` IS a substring of the direct row's selector here, so this is the
    # TRUE eviction shape: a row that matches the selector must never vanish.
    got = [row.selector for row in rank_rows(rows, "banana")]
    assert "direct-prov/banana-flash" in got, got


def test_a_subsequence_only_row_is_not_evicted_by_a_name_literal():
    """R1-1's exact `older`/`banana` mechanism, reduced.

    The direct row's selector matches `banana` only through the subsequence pass;
    another row's NAME contains `banana` literally. Both belong in the answer —
    membership is "matched anything at all", and QUALITY (not membership) decides
    which leads.
    """
    rows = [
        ModelRow("alibaba-token-plan", "deepseek-v4-flash-0731", "", 1000, 0.0, 0.0, True),
        ModelRow(
            "openrouter",
            "google/gemini-3.1-flash-image",
            "google/gemini-3.1-flash-image",
            1000,
            0.0,
            0.0,
            True,
            aggregated=True,
            listing_name="Google: Nano Banana 2 (Gemini 3.1 Flash Image)",
        ),
    ]
    got = [row.selector for row in rank_rows(rows, "banana")]
    assert "alibaba-token-plan/deepseek-v4-flash-0731" in got, got


def test_a_punctuation_only_query_does_not_return_the_whole_catalogue():
    """R1-2/B2: `.`/`...` must match nothing, not everything.

    `_match_key` collapses non-alphanumerics to '', so a punctuation-only query
    used to fall into the EMPTY-query branch — which lists the whole catalogue.
    Measured at 574 rows: `'.'` went 257 -> 574, `'...'` 0 -> 1816. The picker
    filters per keystroke, so that is one keystroke away.
    """
    rows = [_row("openai/gpt-5"), _row("anthropic/claude-opus-5")]
    for query in (".", "!", "?", "*", "-", "...", "~", "@", ":"):
        assert rank_rows(rows, query) == [], query
    # The genuinely-empty cases still list the catalogue.
    assert rank_rows(rows, "") != []
    assert rank_rows(rows, "   ") != []


def test_a_selector_match_outranks_a_name_only_sibling():
    """R1-3/B4: `grok-4.7-fast` must not lead `grok-4.7` for `grok 4.7`.

    The shorter sibling scores DENSER on the human name, so a score-only
    comparison promoted it and pushed the operator's exact model to second. The
    QUALITY rung fixes it: a hit in the SELECTOR leads a hit only in the name.
    """
    rows = [
        ModelRow("xai", "grok-4.7", "Grok 4.7", 1000, 0.0, 0.0, True),
        ModelRow("xai", "grok-4.7-fast", "Grok 4.7 Fast", 1000, 0.0, 0.0, True),
    ]
    assert [row.selector for row in rank_rows(rows, "grok 4.7")][0] == "xai/grok-4.7"


def test_grok_47_resolves_through_the_subsequence_pass():
    """R1-6: the docstring's own claim, pinned to what the code does.

    `grok 47` is NOT an exact-substring match (`47` is a different token from
    `4` then `7`) but the subsequence pass finds a `4` then a `7` in order, so it
    DOES resolve. The docstring used to say the opposite.
    """
    rows = [
        ModelRow(
            "openrouter",
            "x-ai/grok-4.7",
            "openrouter/x-ai/grok-4.7",
            1000,
            0.0,
            0.0,
            True,
            aggregated=True,
            listing_name="SpaceXAI: Grok 4.7",
        )
    ]
    assert [row.selector for row in rank_rows(rows, "grok 47")] == ["openrouter/x-ai/grok-4.7"]


def test_match_key_is_memoised_so_repeated_keystrokes_do_not_renormalise():
    """R1-7: the widget re-ranks per keystroke; `_match_key` is pure.

    Measured ~9-10x the base cost uncached (0.9 -> 8.7 ms at 1,816 rows). The
    cache is the fix; this pins that the identical call is a cache hit.
    """
    from local_operator.model.ranking import _match_key

    _match_key.cache_clear()
    _match_key("SpaceXAI: Grok 4.7")
    before = _match_key.cache_info()
    _match_key("SpaceXAI: Grok 4.7")
    after = _match_key.cache_info()
    assert after.hits == before.hits + 1
