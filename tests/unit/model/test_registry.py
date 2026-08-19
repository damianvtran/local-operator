import pytest

from local_operator.model.configure import build_model_spec
from local_operator.model.registry import (
    ModelInfo,
    _anthropic_family,
    anthropic_family_model_info,
    anthropic_models,
    get_model_info,
    qwencloud_token_plan_models,
    static_models,
)


@pytest.mark.parametrize(
    "provider, model_id, supported",
    [
        ("openai", "gpt-5", True),
        ("openai", "gpt-5.4", True),
        ("openai", "gpt-5.3-codex", True),
        ("openai", "gpt-4.1", False),
        ("openai", "gpt-4o", False),
        ("openrouter", "openai/gpt-5.4", False),
    ],
)
def test_responses_api_capability_is_pinned_to_direct_openai_gpt5(
    provider: str, model_id: str, supported: bool
) -> None:
    spec = build_model_spec(provider, model_id)
    assert spec.supports_responses_api is supported
    if supported:
        assert spec.supports_prompt_cache is True


def test_model_info_price_must_be_non_negative() -> None:
    """Test that the price_must_be_non_negative validator works correctly."""
    with pytest.raises(ValueError, match="Price must be non-negative."):
        ModelInfo(
            id="test-model",
            name="test-model",
            description="Mock model",
            input_price=-1,
            output_price=1,
            recommended=True,
        )
    with pytest.raises(ValueError, match="Price must be non-negative."):
        ModelInfo(
            id="test-model",
            name="test-model",
            description="Mock model",
            input_price=1,
            output_price=-1,
            recommended=True,
        )
    # Should not raise an error
    ModelInfo(
        id="test-model",
        name="test-model",
        description="Mock model",
        input_price=0,
        output_price=0,
        recommended=True,
    )
    ModelInfo(
        id="test-model",
        name="test-model",
        description="Mock model",
        input_price=1,
        output_price=1,
        recommended=False,
    )


def test_get_model_info() -> None:
    """Test that the get_model_info function works correctly."""

    # Test Anthropic
    model_info = get_model_info("anthropic", "claude-3-5-sonnet-20241022")
    assert model_info.max_tokens == 8192

    # Test Google
    model_info = get_model_info("google", "gemini-2.0-flash-001")
    assert model_info.context_window == 1_048_576

    # Test OpenAI
    model_info = get_model_info("openai", "gpt-4o")
    assert model_info.max_tokens == 128_000

    # Test OpenRouter
    model_info = get_model_info("openrouter", "any")
    assert model_info.context_window == -1

    # Test Alibaba
    model_info = get_model_info("alibaba", "qwen2.5-coder-32b-instruct")
    assert model_info.context_window == 131_072

    # Test Mistral
    model_info = get_model_info("mistral", "mistral-large-2411")
    assert model_info.max_tokens == 131_000

    # Test Kimi
    model_info = get_model_info("kimi", "moonshot-v1-8k")
    assert model_info.context_window == 8192

    # Test Deepseek
    model_info = get_model_info("deepseek", "deepseek-chat")
    assert model_info.context_window == 64_000

    # Test unknown model
    model_info = get_model_info("anthropic", "unknown_model")
    assert model_info.max_tokens == -1
    assert model_info.context_window == -1

    # Test Unsupported hosting provider
    with pytest.raises(ValueError, match="Unsupported hosting provider: unknown"):
        get_model_info("unknown", "any")


# -- QwenCloud Token Plan -----------------------------------------------------
#
# The Token Plan gateway's `/models` listing carries ONLY ids (checked live,
# 2026-08-18): no context_window, no max_tokens, no prices. Discovery therefore
# cannot correct these rows, and the registry is the sole source of the numbers
# a session runs on. Before these rows existed, `build_model_spec` fell through
# to the 128k unknown default and a 1M-window model compacted at 128k — the
# status band read `113.9%/128k` mid-conversation.


def test_token_plan_models_ship_their_real_windows() -> None:
    """Transcribed from Alibaba's published specs and cross-checked against
    OpenRouter's listing for the same upstream models (both said 1M/131k for
    qwen3.8-max on 2026-08-18). A drift here is a wrong compaction threshold,
    not a cosmetic gap."""
    flagship = qwencloud_token_plan_models["qwen3.8-max"]
    assert (flagship.context_window, flagship.max_tokens) == (1_000_000, 131_072)
    assert flagship.supports_images is True

    # Both the exact-id chain and the enumerable map must answer, because
    # `build_model_spec` reaches the former and discovery merges over the
    # latter — a row present in only one leaves the other path at 128k.
    assert get_model_info("alibaba-token-plan", "qwen3.8-max").context_window == 1_000_000
    assert get_model_info("alibaba-token-plan-oauth", "qwen3.8-max").context_window == 1_000_000
    assert static_models("alibaba-token-plan")["qwen3.8-max"].context_window == 1_000_000

    # Every chat row in the map carries a usable window: a zero or missing one
    # would silently disable compaction for that model (see build_model_spec).
    for model_id, info in qwencloud_token_plan_models.items():
        assert info.context_window and info.context_window > 0, model_id
        assert info.max_tokens and info.max_tokens > 0, model_id


def test_token_plan_spec_carries_the_window_to_the_session() -> None:
    """The spec IS what the session runs on — compaction thresholds derive from
    `context_window` — so the assertion is on the end of the pipe, not the map."""
    spec = build_model_spec("alibaba-token-plan", "qwen3.8-max")
    assert spec.context_window == 1_000_000
    assert spec.max_output_tokens == 131_072
    # The oauth login flavour serves the same catalogue and must not regress to
    # the 128k default just because the session config spelled the provider id
    # the way `/login` did.
    assert build_model_spec("alibaba-token-plan-oauth", "qwen3.8-max").context_window == 1_000_000


# -- Anthropic family inheritance ---------------------------------------------
#
# `get_model_info` above is an exact-id lookup and stays one: an id it does not
# ship is `unknown_model_info`, and callers that want better ask for it. What
# follows is that better answer, used by `configure._registry_fallback` for an id
# the registry has never seen. It exists because a status band reported
# `1.8%/200k` on a 1M-context Opus 5: a single per-vendor floor cannot be right for
# a vendor whose tiers no longer agree (Opus 5 serves 1M, Opus 4.5 serves 200k).


def test_the_five_series_ships_the_window_the_provider_reports() -> None:
    """Read from `GET /v1/models` on 2026-08-07. The offline path is the whole
    reason these rows exist, so a drift here is a silently wrong compaction
    threshold rather than a cosmetic one."""
    opus5 = anthropic_models["claude-opus-5"]
    assert (opus5.context_window, opus5.max_tokens) == (1_000_000, 128_000)
    assert opus5.supports_images is True
    assert opus5.supports_prompt_cache is True
    # Anthropic's own published rate, per MILLION tokens, from the "Model
    # pricing" table read 2026-08-10. These were 0.0 placeholders — `/v1/models`
    # quotes no prices, so nothing ever filled them in — and the status band read
    # "cost unavailable" for the whole generation as a result.
    assert (opus5.input_price, opus5.output_price) == (5.0, 25.0)
    # The 5m cache write (1.25x base) and the cache hit (0.1x base): this agent
    # runs with prompt caching on, so a cached turn billed at the full input rate
    # is wrong by an order of magnitude on the priciest model in the catalogue.
    assert (opus5.cache_writes_price, opus5.cache_reads_price) == (6.25, 0.50)
    # The generation where the tiers stopped agreeing, so neither may be inferred
    # from the other.
    assert anthropic_models["claude-sonnet-4-5-20250929"].context_window == 1_000_000
    assert anthropic_models["claude-opus-4-5-20251101"].context_window == 200_000


@pytest.mark.parametrize(
    "model_id, expected",
    [
        # Both id shapes Anthropic has shipped, tier before or after the version.
        ("claude-opus-5", ("opus", (5,))),
        ("claude-opus-4-5-20251101", ("opus", (4, 5))),
        ("claude-3-5-sonnet-20241022", ("sonnet", (3, 5))),
        ("claude-3-7-sonnet-latest", ("sonnet", (3, 7))),
        # A tier no hardcoded list would have contained.
        ("claude-fable-5", ("fable", (5,))),
        # Nothing to inherit from: no version, or no tier at all.
        ("claude-opus-latest", None),
        ("gpt-4o", None),
    ],
)
def test_the_family_parser_reads_both_id_shapes(model_id: str, expected) -> None:
    """The 8-digit snapshot date must not read as a version component: as
    (4, 5, 20251101) every dated snapshot becomes its own family, sorts above every
    real generation, and inherits nothing — the opposite of the point."""
    assert _anthropic_family(model_id) == expected


def test_a_dated_snapshot_inherits_the_model_it_is_a_snapshot_of() -> None:
    """The reported case. `claude-opus-5-20260112` is Opus 5, so it gets Opus 5's
    1M window rather than the vendor-wide 200k floor — an 84% loss of usable
    context, with compaction firing at 160k instead of 600k."""
    info = anthropic_family_model_info("claude-opus-5-20260112")
    assert info is not None
    assert (info.context_window, info.max_tokens) == (1_000_000, 128_000)
    assert info.id == "claude-opus-5-20260112"
    # Same model, so its real name is not a guess.
    assert info.name == "Claude Opus 5"


def test_an_undated_alias_inherits_the_snapshot_it_names() -> None:
    """`claude-sonnet-4-5` is how the docs spell the id whose registry row is
    dated, and Sonnet 4.5 is the 1M member of a 200k generation — exactly the pair
    a per-vendor floor gets wrong."""
    info = anthropic_family_model_info("claude-sonnet-4-5")
    assert info is not None
    assert info.context_window == 1_000_000
    assert info.max_tokens == 64_000


def test_a_newer_generation_inherits_limits_but_never_prices() -> None:
    """A generation released after this registry was last edited takes the newest
    known limits of its tier, because windows have only grown. Its PRICE is the one
    thing a new generation reliably changes, so it drops to the unknown zero rather
    than quoting the previous generation's."""
    info = anthropic_family_model_info("claude-opus-6")
    assert info is not None
    assert info.context_window == 1_000_000
    assert (info.input_price, info.output_price) == (0.0, 0.0)
    assert (info.cache_writes_price, info.cache_reads_price) == (None, None)
    # Not "Claude Opus 5": the band names the model that is answering.
    assert info.name == "claude-opus-6"


def test_inheritance_never_runs_backwards_to_an_older_generation() -> None:
    """The asymmetry that makes forward inheritance safe does not reverse. The
    default threshold is `min(0.8 * window, 600k)`, so a 200k-era model handed a 1M
    window triggers at 600k — past its real limit — and 400s every turn instead of
    compacting."""
    assert anthropic_family_model_info("claude-opus-2") is None
    assert anthropic_family_model_info("claude-sonnet-1-5") is None


def test_a_family_answer_is_never_the_registrys_own_object() -> None:
    """Sessions write to their `ModelInfo`, and these rows are module-level
    singletons shared by every session in the process."""
    first = anthropic_family_model_info("claude-opus-5-20260112")
    assert first is not None
    first.context_window = 1
    assert anthropic_models["claude-opus-5"].context_window == 1_000_000
