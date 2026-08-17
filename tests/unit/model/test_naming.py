"""Model display names — precedence, the ambiguity refusal, and the fallback.

The band trades width for readability by printing ``Claude Opus 5`` where it used
to print ``anthropic/claude-opus-5``. That trade is only sound while the short
form still says which model is replying, so the interesting assertions here are
the REFUSALS: the cases where a name exists, is shorter, and is nevertheless not
used because more than one model answers to it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

from rich.cells import cell_len

from local_operator.model.configure import build_model_spec
from local_operator.model.naming import _ID_MARGIN, model_label
from local_operator.model.registry import (
    STATIC_MODEL_HOSTINGS,
    ModelInfo,
    ollama_default_model_info,
    static_models,
)


def _curated_rows() -> list[tuple[str, str, str]]:
    """``(hosting, model_id, curated name)`` for every row the registry ships."""
    return [
        (hosting, model_id, info.name)
        for hosting in STATIC_MODEL_HOSTINGS
        for model_id, info in static_models(hosting).items()
    ]


def test_no_two_shipped_rows_share_a_curated_name() -> None:
    """A shared name is a registry DEFECT, not a case to render around.

    Two google rows used to ship the identical ``Gemini 2.0 Flash Thinking Exp``
    while differing by an 8x output limit, so no surface that renders the name
    could say which was answering and both fell back to a 42-cell selector. They
    now carry their release tokens. This test is what stops the next duplicate
    reaching a user, and it is why the refusal machinery below has to be exercised
    with constructed names rather than with shipped rows.
    """
    owners: dict[str, str] = {}
    for hosting, model_id, name in _curated_rows():
        assert name not in owners, f"{hosting}/{model_id} and {owners.get(name)} share {name!r}"
        owners[name] = f"{hosting}/{model_id}"


# ---------------------------------------------------------------------------
# precedence
# ---------------------------------------------------------------------------


def test_a_caller_supplied_name_wins_over_the_curated_one() -> None:
    """The caller's name is ``ModelInfo.name`` as resolution produced it, and
    resolution already layered the provider's live listing over the registry. A
    module that re-preferred the curated string would undo that merge and pin a
    renamed model to its old label forever."""
    assert model_label("anthropic", "claude-opus-4-5-20251101").full == (
        "Claude Opus 4.5 (2025-11-01)"
    )
    assert model_label("anthropic", "claude-opus-4-5-20251101", "Claude Opus 4.5").full == (
        "Claude Opus 4.5"
    )


def test_the_curated_name_is_used_when_the_caller_has_none() -> None:
    """The band paints before any listing resolves and repaints eight times a
    second, so its default path has to answer from memory."""
    assert model_label("anthropic", "claude-opus-5").full == "Claude Opus 5"


def test_a_direct_providers_listing_names_a_model_no_registry_row_covers() -> None:
    """The reason a caller may pass a name at all, and the case
    ``ModelSpec.display_name`` exists for: the registry provably lags a direct
    provider's releases, and until someone curates the row the provider's own
    listing is the only thing that knows what it is called."""
    assert model_label("anthropic", "claude-opus-6", "Claude Opus 6").full == "Claude Opus 6"


def test_a_name_that_merely_echoes_the_id_is_not_a_name() -> None:
    """Endpoints with no display metadata answer with the key they were asked
    about. Promoting that spends the whole honesty budget to render the string it
    started from — and for a vendor-scoped id the echo is the SCOPED form while
    the band would have shown the bare tail, so both shapes are refused."""
    for echo in ("vendor/some-model", "some-model"):
        assert model_label("anthropic", "vendor/some-model", echo).full == (
            "anthropic/vendor/some-model"
        )


def test_a_rejected_supplied_name_still_lets_the_curated_one_through() -> None:
    """Rejection says "another model already answers to this", which is not a
    statement about whether THIS model has a good name of its own."""
    borrowed = static_models("google")["gemini-2.0-flash-001"].name
    assert model_label("anthropic", "claude-opus-5", borrowed).full == "Claude Opus 5"


# ---------------------------------------------------------------------------
# the ambiguity refusal
# ---------------------------------------------------------------------------


def test_a_name_another_model_of_the_same_provider_owns_is_refused() -> None:
    """The band would otherwise print one string for two different models. The
    selector comes back instead: wider, and unique by construction.

    Constructed rather than drawn from shipped data on purpose — see
    ``test_no_two_shipped_rows_share_a_curated_name``. A live listing renaming
    ``claude-opus-4-8`` to a name ``claude-opus-5`` already answers to is exactly
    how this reaches a user.
    """
    sibling = static_models("anthropic")["claude-opus-5"].name
    # NOT the borrowed name. It falls through to this row's own curated name,
    # which is the documented behaviour: "another model already answers to this"
    # says nothing about whether THIS model has a good name of its own.
    assert model_label("anthropic", "claude-opus-4-8", sibling).full == "Claude Opus 4.8"
    # And with no name of its own to fall back on, the selector is what is left.
    assert model_label("anthropic", "claude-opus-9", sibling).full == "anthropic/claude-opus-9"


def test_a_direct_provider_may_not_borrow_another_providers_curated_name() -> None:
    """Cross-provider, and the reason the index spans every hosting at once
    rather than one provider's rows: a gateway fronting ``openai`` that returned
    ``Claude Opus 5`` would render identically to the real Anthropic route."""
    borrowed = static_models("anthropic")["claude-opus-5"].name
    assert model_label("openai", "some-proxy-id", borrowed).full == "openai/some-proxy-id"
    # The owner keeps it.
    assert model_label("anthropic", "claude-opus-5", borrowed).full == "Claude Opus 5"


def test_a_case_variant_of_a_curated_name_is_refused_too() -> None:
    """The index is compared case-folded. A display name is being used here to
    decide whether two models are the same thing, and ``claude opus 5`` is not a
    different model from ``Claude Opus 5``."""
    for variant in ("claude opus 5", "CLAUDE OPUS 5", "  Claude Opus 5  "):
        assert model_label("openai", "some-proxy-id", variant).full == "openai/some-proxy-id"


def test_a_resellers_listing_name_is_never_used() -> None:
    """Not conservatism, arithmetic. A reseller's name describes the MODEL and
    every reseller resells the same models — measured against the two shipped
    aggregators' real cached catalogues, 398 of ~400 names are carried by a row in
    BOTH, while zero collide inside either one. So the two routes for one model
    would have rendered a byte-identical band label at the widest rung, on routes
    whose price and quota differ, where before this module they rendered distinct
    selectors."""
    name = "MoonshotAI: Kimi K2 0711"
    assert model_label("openrouter", "moonshotai/kimi-k2", name).full == (
        "openrouter/moonshotai/kimi-k2"
    )
    assert model_label("radient", "moonshotai/kimi-k2", name).full == ("radient/moonshotai/kimi-k2")


def test_two_routes_for_one_model_never_render_the_same_label() -> None:
    """The property the reseller rule exists for, stated as the user's question:
    "which route is answering?" must have a different answer on screen."""
    name = "Anthropic: Claude Opus 5"
    rendered = {
        model_label(provider, "anthropic/claude-opus-5", name).full
        for provider in ("openrouter", "radient")
    } | {model_label("anthropic", "claude-opus-5", "Claude Opus 5").full}
    assert len(rendered) == 3, rendered


def test_no_two_shipped_models_render_the_same_label_in_either_form() -> None:
    """The whole shipped registry, both forms. ``compact`` matters as much as
    ``full``: it is what the band actually shows once the ladder reaches
    ``shorten-model``, and it is what ``_compact_index`` exists to protect."""
    full_seen: dict[str, str] = {}
    compact_seen: dict[str, str] = {}
    for hosting, model_id, _ in _curated_rows():
        selector = f"{hosting}/{model_id}"
        label = model_label(hosting, model_id)
        assert label.full not in full_seen, f"{selector} / {full_seen.get(label.full)}"
        assert label.compact not in compact_seen, f"{selector} / {compact_seen.get(label.compact)}"
        full_seen[label.full] = selector
        compact_seen[label.compact] = selector


# ---------------------------------------------------------------------------
# the fallback
# ---------------------------------------------------------------------------


def test_a_model_with_no_curated_name_and_no_listing_keeps_its_selector() -> None:
    """A local Ollama tag: nothing has ever described it, and the string the
    operator typed beats an invented abbreviation of it."""
    label = model_label("ollama", "qwen3:32b")
    assert label.full == "ollama/qwen3:32b"
    assert label.compact == "qwen3:32b"


def test_a_provider_with_no_model_id_at_all_is_returned_unchanged() -> None:
    """Reached from the band's own field before a session exists, where the
    "selector" is whatever the host had — often not a selector."""
    assert model_label("ollama", "").full == "ollama"
    assert model_label("ollama", "").compact == "ollama"


# ---------------------------------------------------------------------------
# the compact form
# ---------------------------------------------------------------------------


def test_a_trailing_qualifier_is_dropped_when_that_stays_unambiguous() -> None:
    """The only shortening this module performs, and the whole reason the band's
    ``shorten-model`` rung has anything to do for a named model."""
    assert model_label("anthropic", "claude-opus-4-5-20251101").compact == "Claude Opus 4.5"


def test_a_qualifier_that_is_the_only_difference_is_never_dropped() -> None:
    """``Claude 3.7 Sonnet (Latest)`` and ``(2025-02-19)`` differ ONLY inside the
    brackets, so dropping them is exactly what makes the two indistinguishable.

    ``Claude 3.7 Sonnet`` is 17 cells against the (Latest) id's 24, so it is the
    narrowest thing on offer and every width rule here would reach for it. The
    ambiguity guard is the only reason it does not appear.
    """
    for model_id in ("claude-3-7-sonnet-latest", "claude-3-7-sonnet-20250219"):
        label = model_label("anthropic", model_id)
        assert label.full.endswith(")")
        assert label.compact != "Claude 3.7 Sonnet"


def test_the_compact_form_gives_way_when_the_id_is_meaningfully_narrower() -> None:
    """The rung that asks for this form is the one the band reaches under width
    pressure, so it must not be the reason another segment is dropped. A display
    name is free to be WIDER than the id it replaces, and when it is by more than
    a couple of cells the id wins here even though the name wins in the full
    form."""
    label = model_label("openai", "o4-mini")
    assert label.full == "OpenAI o4 mini"
    assert label.compact == "o4-mini"

    for hosting, model_id, _ in _curated_rows():
        compact = model_label(hosting, model_id).compact
        bare = model_id.rpartition("/")[2] or model_id
        assert cell_len(compact) < cell_len(bare) + _ID_MARGIN, f"{hosting}/{model_id}"


def test_a_name_is_kept_when_the_id_would_only_save_a_cell_or_two() -> None:
    """``_ID_MARGIN``'s reason to exist. A one-cell saving used to turn
    ``Qwen 2.5 Coder 32B Instruct`` into ``qwen2.5-coder-32b-instruct``: not
    truncation, a change of identity — title case and spaces becoming lowercase
    and hyphens — and it flipped back and forth as a terminal was resized one
    column across the threshold."""
    label = model_label("alibaba", "qwen2.5-coder-32b-instruct")
    assert label.compact == "Qwen 2.5 Coder 32B Instruct"
    assert cell_len(label.compact) > cell_len("qwen2.5-coder-32b-instruct")


def test_a_tie_on_width_goes_to_the_name() -> None:
    """``Claude Opus 5`` and ``claude-opus-5`` are both 13 cells; spending the
    same width on the readable one is free."""
    assert model_label("anthropic", "claude-opus-5").compact == "Claude Opus 5"


# ---------------------------------------------------------------------------
# the spec boundary — where the band actually gets its name from
# ---------------------------------------------------------------------------


def test_the_spec_carries_the_resolved_name() -> None:
    """``ModelSpec.display_name`` is the only route by which a name resolved
    through the disk catalogue reaches a repaint, so it has to survive the trip.

    ``info`` is passed explicitly: resolution otherwise reads whatever listing
    this machine has cached, which would make the assertion depend on the
    developer's ``~/.local-operator/cache``.
    """
    info = ModelInfo(id="claude-opus-5", name="Claude Opus 5", description="")
    assert build_model_spec("anthropic", "claude-opus-5", info).display_name == "Claude Opus 5"


def test_a_placeholder_row_supplies_no_name() -> None:
    """Resolution degrades to a row describing the PROVIDER when nothing describes
    the id — ``ollama_default_model_info`` is ``name="Ollama"`` — and that name is
    the same for every local model. Two Ollama tags both labelled ``Ollama`` is the
    exact ambiguity this module exists to prevent."""
    assert build_model_spec("ollama", "qwen3:32b", ollama_default_model_info).display_name == ""
    assert model_label("ollama", "qwen3:32b").full == "ollama/qwen3:32b"


def test_the_unknown_placeholder_name_is_not_a_display_name() -> None:
    """``Unknown`` is the shipped fallback's identity, not a model. Promoting it
    is how a nameless xAI listing painted the status band ``Unknown`` for a
    running Grok 4.6 session."""
    info = ModelInfo(id="grok-4.6", name="Unknown", description="Unknown model")
    assert build_model_spec("xai", "grok-4.6", info).display_name == ""
    assert model_label("xai", "grok-4.6", "Unknown").full == "xai/grok-4.6"
    assert model_label("xai", "grok-4.6", "Unknown").compact == "grok-4.6"


def test_a_normalised_id_still_supplies_its_name() -> None:
    """The placeholder guard compares ids under ``_normalised_id``, not by
    equality, and this is why. ``_info_from_discovery`` matches rows on the
    normalised id and then writes the LISTING's spelling into ``info.id``, so a
    user who types the id Google's own documentation shows
    (``models/gemini-2.5-pro``) resolves a row whose id is the bare
    ``gemini-2.5-pro``. Equality would call that "not about this model" and throw a
    perfectly good name away for every Gemini session started the documented way."""
    row = ModelInfo(
        id="gemini-2.5-pro",
        name="Gemini 2.5 Pro",
        description="",
        context_window=1_000_000,
        max_tokens=64_000,
    )
    for typed in ("gemini-2.5-pro", "models/gemini-2.5-pro"):
        assert build_model_spec("google", typed, row).display_name == "Gemini 2.5 Pro"


def test_a_name_shaped_object_that_is_not_a_string_is_no_name() -> None:
    """``info`` is duck-typed and ``name`` is the attribute most likely to hold
    something that is not a string on a stand-in — a ``MagicMock``'s ``.name`` is
    its own identity, so it arrives as a child mock. Feeding that to a ``str``
    pydantic field raises ``ValidationError``, and a display label must never be
    able to fail a session start.

    The stand-in carries real numbers so the mock reaches the name at all: the
    numeric fields are read first and a bare mock dies on the window comparison,
    which would make this test pass for the wrong reason.
    """
    stub = SimpleNamespace(
        id="claude-opus-5",
        name=MagicMock(),
        context_window=200_000,
        max_tokens=64_000,
        supports_images=True,
        supports_prompt_cache=True,
    )
    # `stub` is deliberately not a ModelInfo: the point is the defensive read
    # in build_model_spec, which must survive anything whose `.name` is not a
    # str (a MagicMock reaches it in the wild). Typed as Any to say so.
    assert build_model_spec("anthropic", "claude-opus-5", cast(Any, stub)).display_name == ""
