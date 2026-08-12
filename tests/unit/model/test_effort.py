"""Which reasoning-effort levels each model offers, and how they cycle.

The table these exercise is transcribed from the providers' own docs (see
``local_operator.model.effort``), so the tests below are about the two things
transcription cannot guarantee on its own: that the RIGHT ladder reaches the
right model — an over-generous one is an HTTP 400 on every turn, a stingy one
silently withholds a level the user is paying for — and that a level can never
outlive the model it was chosen for.
"""

from __future__ import annotations

import pytest

from local_operator.model.configure import build_model_spec
from local_operator.model.effort import (
    EFFORT_ORDER,
    default_effort,
    next_effort,
    resolve_effort,
    supported_efforts,
)


class TestTheLadderIsPerModel:
    """ "A supported effort level for the current model" — the set is not global."""

    @pytest.mark.parametrize(
        ("model_id", "levels"),
        [
            # Anthropic's three ladders, from the effort doc's per-level model
            # lists: 4.5 got neither extension, 4.6 got `max` only, and 4.7
            # onwards plus every generation-5 tier got both.
            ("claude-opus-4-5-20251101", ("low", "medium", "high")),
            ("claude-sonnet-4-6", ("low", "medium", "high", "max")),
            ("claude-opus-4-7", ("low", "medium", "high", "xhigh", "max")),
            ("claude-opus-5", ("low", "medium", "high", "xhigh", "max")),
            # A tier no list in this repo enumerates. The pattern reads the
            # GENERATION, which is what stopped `claude-fable-5` slipping past
            # the sampling rule the same way.
            ("claude-fable-5", ("low", "medium", "high", "xhigh", "max")),
            # OpenAI: gpt-5.4's model page states none/low/medium/high/xhigh.
            ("gpt-5.4", ("none", "low", "medium", "high", "xhigh")),
            # The o-series predates the extended vocabulary.
            ("o3", ("low", "medium", "high")),
            ("o4-mini", ("low", "medium", "high")),
        ],
    )
    def test_each_family_gets_its_own_documented_set(
        self, model_id: str, levels: tuple[str, ...]
    ) -> None:
        assert supported_efforts(model_id) == levels

    @pytest.mark.parametrize(
        "model_id",
        [
            "gpt-4.1",  # no reasoning at all
            "claude-3-5-sonnet-20241022",  # predates the effort parameter
            "gemini-2.5-pro",  # thinks, but not through a named tier we can send
            "deepseek-reasoner",  # reasons at a depth the API does not expose
        ],
    )
    def test_a_model_without_the_knob_offers_nothing(self, model_id: str) -> None:
        """Empty, not a default set. A level offered here would be a level the
        request cannot carry, which is a status band asserting a depth of
        thought that is not in force."""
        assert supported_efforts(model_id) == ()
        assert default_effort(model_id) is None

    def test_the_aggregator_prefix_resolves_to_the_same_model(self) -> None:
        """The ladder is a property of the MODEL, not of the route to it —
        the rule `supports_sampling_params` already follows."""
        assert supported_efforts("anthropic/claude-opus-5") == supported_efforts("claude-opus-5")
        assert supported_efforts("openrouter/openai/gpt-5.4") == supported_efforts("gpt-5.4")

    def test_every_level_named_is_a_level_the_ladder_knows(self) -> None:
        """The cycle order is the shared vocabulary; a family that invented a
        word outside it would cycle into a value nothing can sort."""
        for model_id in ("claude-opus-5", "gpt-5.4", "o3", "claude-sonnet-4-6"):
            assert set(supported_efforts(model_id)) <= set(EFFORT_ORDER)
            ranks = [EFFORT_ORDER.index(level) for level in supported_efforts(model_id)]
            assert ranks == sorted(ranks), model_id


class TestTheSpecCarriesIt:
    """``build_model_spec`` is the one place the ladder is derived."""

    def test_anthropic_boots_on_its_documented_default(self) -> None:
        """`high` is seeded rather than left blank because Anthropic documents
        sending it as identical to omitting the parameter — so the band states
        a real level from the first frame without changing the request."""
        spec = build_model_spec("anthropic", "claude-opus-5")
        assert spec.reasoning_effort == "high"
        assert spec.reasoning_efforts == ("low", "medium", "high", "xhigh", "max")

    def test_openai_boots_with_no_level_claimed(self) -> None:
        """OpenAI's default is per snapshot — `none` on gpt-5.4, `medium` on
        gpt-5.5 — so seeding either would put a level on the band that half the
        family is not running."""
        spec = build_model_spec("openai", "gpt-5.4")
        assert spec.reasoning_effort is None
        assert spec.reasoning_efforts == ("none", "low", "medium", "high", "xhigh")

    def test_a_ladder_makes_a_model_a_reasoning_model(self) -> None:
        """`claude-opus-5` matches none of the name markers — it says neither
        "thinking" nor "reasoner" — so before the ladder the band reported
        nothing at all for the deepest-reasoning model the app ships with."""
        assert build_model_spec("anthropic", "claude-opus-5").reasoning is True
        assert build_model_spec("openai", "gpt-4.1").reasoning is False

    def test_a_model_without_a_ladder_carries_neither_field(self) -> None:
        spec = build_model_spec("openai", "gpt-4.1")
        assert spec.reasoning_efforts == ()
        assert spec.reasoning_effort is None


class TestALevelCannotOutliveItsModel:
    """The guard behind `/model`, `/reload` and every fallback hop."""

    def test_a_shared_level_carries_across(self) -> None:
        assert resolve_effort("gpt-5.4", "medium") == "medium"

    def test_a_level_above_the_targets_ceiling_clamps_to_the_ceiling(self) -> None:
        """`xhigh` on a model whose ladder stops at `high` is a 400 on the very
        request a fallback was supposed to rescue."""
        assert resolve_effort("claude-opus-4-5-20251101", "xhigh") == "high"
        assert resolve_effort("claude-opus-4-5-20251101", "max") == "high"

    def test_none_clamps_to_the_floor_rather_than_escalating_to_the_default(self) -> None:
        """`none` is not a level like the others — it is the user saying do not
        reason. Resolving it to the target's default sent someone who had turned
        reasoning OFF on an OpenAI model to Anthropic's `high` on a failover hop:
        a bill they had explicitly opted out of, with no keystroke and no notice.
        The floor of the target's ladder is the nearest thing it can express."""
        assert resolve_effort("claude-opus-5", "none") == "low"
        assert resolve_effort("o3", "none") == "low"

    def test_an_unrecognised_value_still_falls_back_to_the_default(self) -> None:
        """Stale state or a hand-edited config: there is no rung to clamp
        toward, so the model's own default is the only honest answer."""
        assert resolve_effort("claude-opus-5", "turbo") == "high"

    def test_a_model_without_a_ladder_drops_it_entirely(self) -> None:
        assert resolve_effort("gpt-4.1", "high") is None

    def test_nothing_requested_means_the_models_own_default(self) -> None:
        assert resolve_effort("claude-opus-5", None) == "high"
        assert resolve_effort("gpt-5.4", None) is None


class TestTheCycleOrder:
    """``shift+tab`` walks the ladder upward and wraps."""

    def test_it_steps_up_one_rung_at_a_time(self) -> None:
        levels = supported_efforts("claude-opus-5")
        assert next_effort(levels, "low") == "medium"
        assert next_effort(levels, "medium") == "high"
        assert next_effort(levels, "high") == "xhigh"
        assert next_effort(levels, "xhigh") == "max"

    def test_the_top_wraps_to_the_bottom(self) -> None:
        """A cycle with a dead end is a control the user has to reason about."""
        levels = supported_efforts("claude-opus-5")
        assert next_effort(levels, "max") == "low"

    def test_a_full_cycle_visits_every_level_once(self) -> None:
        levels = supported_efforts("gpt-5.4")
        seen = []
        current = levels[0]
        for _ in range(len(levels)):
            seen.append(current)
            current = next_effort(levels, current)
        assert seen == list(levels)
        assert current == levels[0]

    def test_from_unset_it_starts_in_the_middle_not_at_index_zero(self) -> None:
        """On OpenAI index 0 is `none`: a user pressing the key to FIND the
        control would have turned reasoning off with their first press."""
        assert next_effort(supported_efforts("gpt-5.4"), None) == "medium"
        assert next_effort(supported_efforts("claude-opus-5"), None) == "medium"

    def test_a_level_the_model_no_longer_supports_restarts_the_cycle(self) -> None:
        """Stale state cannot wedge the key: an unrecognised current value is
        treated as unset rather than raising or sticking."""
        assert next_effort(supported_efforts("o3"), "xhigh") == "medium"

    def test_an_empty_ladder_cycles_to_nothing(self) -> None:
        assert next_effort((), "high") is None


#: Every Anthropic model the effort doc lists as supporting the parameter, read
#: from https://platform.claude.com/docs/en/build-with-claude/effort on
#: 2026-08-11. Transcribed here rather than derived from the table under test,
#: which is the whole point: the table is an interpretation of this list, and a
#: test that re-derived it could only ever agree with itself.
_DOC_SUPPORTED = {
    "claude-fable-5",
    "claude-mythos-5",
    "claude-mythos-preview",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-opus-4-5-20251101",
}


def test_no_shipped_anthropic_row_gets_a_ladder_the_doc_does_not_grant_it() -> None:
    """Closure over the REGISTRY, not over ids the table is known to route.

    The parametrized cases above pin the author's reading; this pins the
    contract, and it is the assertion that was missing when `\\d{2,}` in the
    4.7+ arm swallowed the 8-digit snapshot date in `claude-opus-4-20250514`
    and handed two shipped rows — one of them `recommended` — a five-rung
    ladder plus an `output_config` key on every single request.

    It also fails forward: a registry row added ahead of the table shows up
    here rather than in a provider's 400.
    """
    from local_operator.model.registry import anthropic_models

    over_granted = sorted(
        model_id
        for model_id in anthropic_models
        if supported_efforts(model_id) and model_id not in _DOC_SUPPORTED
    )
    assert over_granted == [], over_granted


def test_a_snapshot_date_is_not_read_as_a_generation_number() -> None:
    """The specific shape that broke it, kept as its own case because the
    registry closure above would go quiet the day those rows are retired."""
    assert supported_efforts("claude-opus-4-20250514") == ()
    assert supported_efforts("claude-sonnet-4-20250514") == ()
    # …while the forward-reading intent survives: a real generation 4.10 would
    # still be read as 4.7-or-later.
    assert supported_efforts("claude-opus-4-10") == ("low", "medium", "high", "xhigh", "max")
