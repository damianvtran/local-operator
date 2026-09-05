"""The fast-mode support table.

Every expectation here is pinned against a MEASURED provider response (recorded
in ``local_operator.model.speed``'s module docstring and probed live on
2026-09-04), not against prose in a vendor doc, because on this axis the two
disagree in ways that are silent until they are expensive.
"""

from __future__ import annotations

import pytest

from local_operator.model.speed import (
    ANTHROPIC_FAST_BETA,
    DIALECT_ANTHROPIC_SPEED,
    DIALECT_SERVICE_TIER,
    SERVICE_TIER_FAST,
    fast_mode_support,
    supports_fast_mode,
)


class TestAnthropic:
    """Anthropic ships fast mode on an explicit two-model list."""

    @pytest.mark.parametrize("model", ["claude-opus-5", "claude-opus-4-8"])
    def test_supported_opus_models_take_the_speed_key(self, model: str) -> None:
        support = fast_mode_support("anthropic", model)
        assert support is not None
        assert support.dialect == DIALECT_ANTHROPIC_SPEED
        assert support.value == "fast"

    @pytest.mark.parametrize("model", ["claude-opus-5", "claude-opus-4-8"])
    def test_the_speed_key_always_carries_its_beta_header(self, model: str) -> None:
        """Measured: ``speed`` without the header is HTTP 400 "Extra inputs are
        not permitted", so the table must never offer one without the other."""
        support = fast_mode_support("anthropic", model)
        assert support is not None
        assert support.beta_header == ANTHROPIC_FAST_BETA

    def test_opus_4_7_is_excluded_because_it_errors(self) -> None:
        """Anthropic documents ``speed: "fast"`` on 4.7 as a hard error."""
        assert fast_mode_support("anthropic", "claude-opus-4-7") is None

    def test_opus_4_6_is_excluded_because_it_lies(self) -> None:
        """The DANGEROUS exclusion, and the reason this is a list not a range.

        4.6 accepts the key, serves the request at standard speed anyway and
        bills it at standard rates. Offering the dial there would put a "fast"
        segment on the band over a request that is not fast — a status band
        asserting something untrue, which is worse than an error nobody can
        misread.
        """
        assert fast_mode_support("anthropic", "claude-opus-4-6") is None

    @pytest.mark.parametrize(
        "model", ["claude-sonnet-5", "claude-haiku-4-5-20251001", "claude-opus-4-20250514"]
    )
    def test_unlisted_models_get_nothing(self, model: str) -> None:
        """Only the models the vendor NAMES are offered the dial.

        ``claude-opus-4-20250514`` is the snapshot-date case ``model.effort``'s
        table had to defend against: an 8-digit date must not read as a
        generation number and win a capability the model does not have.
        """
        assert fast_mode_support("anthropic", model) is None

    @pytest.mark.parametrize("model", ["claude-opus-5-1", "claude-opus-5.1", "claude-opus-4-8-1"])
    def test_a_point_release_is_not_admitted_by_inference(self, model: str) -> None:
        """ "Explicit list" has to mean it (review F3): siblings within one
        generation diverge, so a 5.x point release is added on the vendor's
        word, never inherited from the generation."""
        assert fast_mode_support("anthropic", model) is None

    @pytest.mark.parametrize("model", ["claude-opus-5-20260101", "claude-opus-4-8-20260101"])
    def test_a_dated_snapshot_of_a_listed_model_is_still_that_model(self, model: str) -> None:
        assert fast_mode_support("anthropic", model) is not None

    def test_dotted_aggregator_spelling_resolves_the_same(self) -> None:
        """Anthropic hyphenates its snapshots; aggregators use a dot.

        The same repair ``model.effort`` documents: a punctuation difference
        must not silently cost a model its dial on one route.
        """
        assert fast_mode_support("anthropic", "claude-opus-4.8") is not None


class TestServiceTierRoutes:
    """OpenAI, xAI and OpenRouter share one key and one accepted value."""

    @pytest.mark.parametrize(
        ("provider", "model"),
        [
            ("openai", "gpt-5.4"),
            ("openai", "gpt-6"),
            ("xai", "grok-4.6"),
            ("openrouter", "openai/gpt-5.4"),
            ("openrouter", "anthropic/claude-opus-5"),
            ("openrouter", "x-ai/grok-4.6"),
        ],
    )
    def test_service_tier_routes_send_priority(self, provider: str, model: str) -> None:
        """NEVER ``"fast"``.

        Measured 2026-09-04: the ChatGPT/Codex backend answers HTTP 400
        ``Unsupported service_tier: fast`` and xAI answers HTTP 422 ``unknown
        variant `fast`, expected one of `auto`, `default`, `flex`, `standard`,
        `priority```. ``priority`` is the one word every service-tier route
        accepts, and OpenAI documents it as equivalent to ``fast``.
        """
        support = fast_mode_support(provider, model)
        assert support is not None
        assert support.dialect == DIALECT_SERVICE_TIER
        assert support.value == SERVICE_TIER_FAST == "priority"
        assert support.beta_header is None

    @pytest.mark.parametrize("model", ["o4-mini", "gpt-4o", "gpt-4.1"])
    def test_pre_gpt5_openai_models_get_nothing(self, model: str) -> None:
        assert fast_mode_support("openai", model) is None

    @pytest.mark.parametrize(
        "model", ["deepseek/deepseek-chat", "moonshotai/kimi-k2", "mistralai/mistral-small"]
    )
    def test_openrouter_is_gated_on_the_underlying_model(self, model: str) -> None:
        """The aggregator is not opened wide: only upstreams that sell a
        priority tier through it may be asked for one."""
        assert fast_mode_support("openrouter", model) is None


class TestRouteOwnsTheDialect:
    """The dialect belongs to the ROUTE, which is why the key is a pair."""

    def test_one_model_two_routes_two_dialects(self) -> None:
        """``claude-opus-5`` direct takes ``speed``; via OpenRouter it takes
        ``service_tier``. A model-only key would send Anthropic's spelling down
        an OpenAI-shaped pipe."""
        direct = fast_mode_support("anthropic", "claude-opus-5")
        aggregated = fast_mode_support("openrouter", "anthropic/claude-opus-5")
        assert direct is not None and aggregated is not None
        assert direct.dialect == DIALECT_ANTHROPIC_SPEED
        assert aggregated.dialect == DIALECT_SERVICE_TIER


class TestRoutesThatSellNoFastTier:
    @pytest.mark.parametrize(
        ("provider", "model"),
        [
            # `generateContent` has no service-tier field at all, so a dial here
            # would be a switch wired to nothing.
            ("google", "gemini-3-pro"),
            ("deepseek", "deepseek-reasoner"),
            ("kimi", "kimi-k2"),
            ("ollama", "llama3"),
            ("zai", "glm-4.6"),
        ],
    )
    def test_unsupported_routes_send_no_key(self, provider: str, model: str) -> None:
        assert fast_mode_support(provider, model) is None
        assert supports_fast_mode(provider, model) is False


class TestDegradation:
    """A lookup runs on a keystroke path and must never raise."""

    @pytest.mark.parametrize(
        ("provider", "model"),
        [("", "claude-opus-5"), ("anthropic", ""), ("", ""), ("  ", "  ")],
    )
    def test_blank_inputs_answer_none(self, provider: str, model: str) -> None:
        assert fast_mode_support(provider, model) is None

    def test_lookup_is_case_insensitive(self) -> None:
        assert fast_mode_support("Anthropic", "Claude-Opus-5") is not None
        assert fast_mode_support("OPENAI", "GPT-5.4") is not None
