"""``picker_rows``: the shaping the TUI picker and the phone's sheet now share.

Every assertion here is a behaviour that used to live inside ``tui/app.py``'s
``_catalogue_rows`` and was only reachable by standing up a Textual app. It is
tested at this level because the mobile daemon is now a caller too, and a rule
that holds for one surface and not the other is the exact defect the extraction
was made to prevent.
"""

from __future__ import annotations

from local_operator.providers.catalogue import picker_rows
from local_operator.providers.controller import CatalogueEntry


def _entry(selector: str, **kwargs) -> CatalogueEntry:
    provider, _, model_id = selector.partition("/")
    kwargs.setdefault("label", "")
    kwargs.setdefault("context_window", 0)
    kwargs.setdefault("input_price", 0.0)
    kwargs.setdefault("output_price", 0.0)
    kwargs.setdefault("connected", True)
    return CatalogueEntry(provider=provider, model_id=model_id, **kwargs)


def test_rows_outside_the_usable_set_are_hidden_and_counted():
    """HIDDEN, not demoted: a row that cannot be chosen is not a choice.

    The count is what lets the caller say "N hidden — /login <provider>", which
    is how discoverability survives the filter.
    """
    entries = [_entry("openai/gpt-5"), _entry("anthropic/claude-opus-5")]
    rows, hidden = picker_rows(entries, usable={"openai"})
    assert [row.selector for row in rows] == ["openai/gpt-5"]
    assert hidden == 1


def test_an_unreadable_store_shows_everything_rather_than_nothing():
    """``usable=None`` is "cannot tell", never "none of them".

    An empty picker claims the user owns no models — precisely the claim the app
    just failed to establish when the credential store would not open.
    """
    entries = [_entry("openai/gpt-5"), _entry("anthropic/claude-opus-5")]
    rows, hidden = picker_rows(entries, usable=None)
    assert {row.selector for row in rows} == {"openai/gpt-5", "anthropic/claude-opus-5"}
    assert hidden == 0


def test_the_current_model_survives_a_filter_that_excludes_its_provider():
    """Its marker is what answers "what am I on".

    Dropping it makes a broken configuration invisible instead of obvious — the
    session is still running that model whether or not the credential is gone.
    """
    entries = [_entry("openai/gpt-5"), _entry("anthropic/claude-opus-5")]
    rows, hidden = picker_rows(entries, usable={"openai"}, current="anthropic/claude-opus-5")
    assert {row.selector for row in rows} == {"openai/gpt-5", "anthropic/claude-opus-5"}
    assert hidden == 0


def test_hidden_counts_the_filter_only_not_the_current_row_exemption():
    """One real hidden entry must still be reported while a row is exempted."""
    entries = [
        _entry("openai/gpt-5"),
        _entry("anthropic/claude-opus-5"),
        _entry("mistral/mistral-large"),
    ]
    rows, hidden = picker_rows(entries, usable={"openai"}, current="anthropic/claude-opus-5")
    assert len(rows) == 2
    assert hidden == 1


def test_openai_rows_advertise_the_default_window_without_the_opt_in():
    """The provider publishes a maximum the Responses API only reaches with the
    opt-in, so a caller that has not opted in must not promise that capacity."""
    entries = [_entry("openai/gpt-5", context_window=400_000, default_context_window=272_000)]
    rows, _ = picker_rows(entries, usable={"openai"}, use_max_context=False)
    assert rows[0].context_window == 272_000
    rows, _ = picker_rows(entries, usable={"openai"}, use_max_context=True)
    assert rows[0].context_window == 400_000


def test_the_window_swap_applies_to_openai_alone():
    """Another provider's ``default_context_window`` is not an opt-out signal."""
    entries = [
        _entry("anthropic/claude-opus-5", context_window=1_000_000, default_context_window=200_000)
    ]
    rows, _ = picker_rows(entries, usable={"anthropic"}, use_max_context=False)
    assert rows[0].context_window == 1_000_000


def test_rows_come_back_ranked_not_in_catalogue_order():
    """The bug that forced this module out of the widget.

    The aggregated row is FIRST in the input, exactly as the registry ordered it
    for the phone, and must come back last. ``xai`` sorts after ``openrouter``
    alphabetically, so passing this needs the aggregator tier rather than the
    provider-name rung.
    """
    entries = [
        _entry("openrouter/x-ai/grok-4", aggregated=True),
        _entry("xai/grok-4"),
    ]
    rows, _ = picker_rows(entries, usable={"openrouter", "xai"})
    assert [row.selector for row in rows] == ["xai/grok-4", "openrouter/x-ai/grok-4"]


def test_the_query_filters_and_ranks_in_one_pass():
    entries = [_entry("openai/gpt-5"), _entry("anthropic/claude-opus-5")]
    rows, hidden = picker_rows(entries, usable={"openai", "anthropic"}, query="opus")
    assert [row.selector for row in rows] == ["anthropic/claude-opus-5"]
    # ``hidden`` reports the CREDENTIAL filter, not the query: a row the user
    # narrowed away themselves is not something to offer them a login for.
    assert hidden == 0


def test_every_presentable_field_survives_the_conversion():
    """The row is what both pickers render, so a dropped field is a blank cell."""
    entries = [
        _entry(
            "openrouter/vendor/model",
            label="Vendor Model",
            context_window=128_000,
            input_price=3.0,
            output_price=15.0,
            connected=False,
            aggregated=True,
            routed=True,
            max_context_window=256_000,
        )
    ]
    (row,), _ = picker_rows(entries, usable=None)
    assert (row.label, row.context_window, row.input_price, row.output_price) == (
        "Vendor Model",
        128_000,
        3.0,
        15.0,
    )
    assert (row.connected, row.aggregated, row.routed, row.max_context_window) == (
        False,
        True,
        True,
        256_000,
    )


def test_an_empty_catalogue_is_an_empty_result_not_an_error():
    assert picker_rows([], usable=set()) == ([], 0)
