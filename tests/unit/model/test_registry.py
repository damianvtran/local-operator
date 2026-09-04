import pytest

from local_operator.model.configure import build_model_spec
from local_operator.model.registry import (
    ModelInfo,
    RecommendedOpenRouterModelIds,
    RecommendedRadientModelIds,
    _anthropic_family,
    anthropic_family_model_info,
    anthropic_models,
    deepseek_models,
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


def test_current_fallback_chain_models_have_first_class_prices() -> None:
    """The operator's default fallback chain must price without discovery.

    ``price_snapshot`` returns ``(0, False)`` — rendered ``$—`` — when both
    input and output prices are missing. After the serving-model fix, an xAI
    row that still had no registry price would appear in By provider with
    tokens and no dollars. These are the ids on the live default chain
    (plus grok-4.6, the one the operator is on right now).
    """
    grok = get_model_info("xai", "grok-4.6")
    assert grok.input_price == 2.00
    assert grok.output_price == 6.00
    assert grok.cache_reads_price == 0.50
    assert grok.context_window == 500_000

    k3 = get_model_info("kimi", "k3")
    assert k3.input_price == 3.00
    assert k3.output_price == 15.00
    assert k3.cache_reads_price == 0.30

    sol = get_model_info("openai", "gpt-5.6-sol")
    assert sol.input_price == 4.0
    assert sol.output_price == 20.0
    assert sol.cache_reads_price == 0.40

    glm = get_model_info("zai", "glm-5.3")
    assert glm.input_price == 1.4
    assert glm.output_price == 4.4


# -- QwenCloud Token Plan -----------------------------------------------------
#
# The Token Plan gateway's `/models` listing carries ONLY ids (checked live,
# 2026-08-18): no context_window, no max_tokens, no prices. Discovery therefore
# cannot correct these rows, and the registry is the sole source of the numbers
# a session runs on. Before these rows existed, `build_model_spec` fell through
# to the 128k unknown default and a 1M-window model compacted at 128k — the
# status band read `113.9%/128k` mid-conversation.


def test_token_plan_models_ship_their_real_windows() -> None:
    """Every row pinned to its exact numbers, not merely to "something positive".

    Output caps are what the endpoint's own `max_tokens` validator reports;
    windows are Alibaba's published figure, since the window cannot be probed
    (a boundary-sized prompt is refused for body size first). Where OpenRouter
    differs it is because it quotes the largest window across its whole routing
    pool, which is not a claim about this gateway \u2014 the reasoning is recorded
    beside the map.

    Pinning only the corroborated row would leave the deliberate deviations free
    to drift silently, so every row is asserted: a change to any of them is a
    changed compaction threshold and has to be a conscious edit.
    """
    assert {
        model_id: (info.context_window, info.max_tokens)
        for model_id, info in qwencloud_token_plan_models.items()
    } == {
        # Output caps as the endpoint's own `max_tokens` validator reports them
        # ("Range of max_tokens should be [1, N]"); see the note beside the map.
        "qwen3.8-max": (1_000_000, 131_072),
        "qwen3.7-max": (1_000_000, 131_072),
        "qwen3.7-plus": (1_000_000, 131_072),
        "qwen3.6-flash": (1_000_000, 65_536),
        "glm-5.2": (1_000_000, 131_072),
        # The two the endpoint does not validate; OpenRouter's figure stands.
        "deepseek-v4-pro": (1_000_000, 384_000),
        "deepseek-v4-flash-0731": (1_000_000, 393_216),
    }
    assert qwencloud_token_plan_models["qwen3.8-max"].supports_images is True

    # Both the exact-id chain and the enumerable map must answer, because
    # `build_model_spec` reaches the former and discovery merges over the
    # latter — a row present in only one leaves the other path at 128k.
    assert get_model_info("alibaba-token-plan", "qwen3.8-max").context_window == 1_000_000
    assert get_model_info("alibaba-token-plan-oauth", "qwen3.8-max").context_window == 1_000_000
    assert static_models("alibaba-token-plan")["qwen3.8-max"].context_window == 1_000_000

    # Every chat row in the map carries a usable window: a zero or missing one
    # would silently disable compaction for that model (see build_model_spec).
    # The exact-value assertion above already covers today's rows; this is the
    # guard for whatever is added next.
    for model_id, info in qwencloud_token_plan_models.items():
        assert info.context_window and info.context_window > 0, model_id
        assert info.max_tokens and info.max_tokens > 0, model_id


def test_token_plan_ships_a_row_for_every_chat_model_the_gateway_lists() -> None:
    """The SET, not just the values — a missing row is silent.

    A model absent from this map does not fail; it resolves to the 128k unknown
    default and runs with a wrong compaction threshold, which is exactly the
    defect this map exists to fix. `deepseek-v4-flash-0731` shipped that way in
    the first cut of this PR precisely because its id sits among the image and
    audio entries in the listing, so nothing but an explicit set comparison
    would have caught it.

    The two lists below are the gateway's `/models` response, split by whether a
    chat completion against the id returns a chat payload (verified live,
    2026-08-19). Re-run that when the listing changes rather than guessing from
    the id: `wan2.7-image` reads like an image-only model and answers chat
    requests with an empty body, while `deepseek-v4-flash-0731` reads like one
    of a family and is a full reasoning chat model.

    WHEN THIS FAILS, suspect the expected side first. Both literals are a
    snapshot of a remote catalogue, dated below; a provider that adds or
    withdraws a model breaks this test without anything in the repo changing.
    That is the intended trade-off — a new chat model must not reach users at
    the 128k default just because nobody noticed it — but it means the fix is
    usually to re-derive the snapshot, not to edit the map.
    """
    # Snapshot of GET /compatible-mode/v1/models, 2026-08-19.
    gateway_chat_models = {
        "qwen3.8-max",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.6-flash",
        "glm-5.2",
        "deepseek-v4-pro",
        "deepseek-v4-flash-0731",
    }
    # Same snapshot, same date: listed by the gateway but NOT chat models — they
    # return no chat payload (the image/TTS entries) or reject the route
    # outright (the realtime one). Kept as a named set because it is the half of
    # the listing this map deliberately omits, and a reader checking the map
    # against the gateway needs to see that the omission was a decision.
    gateway_non_chat_models = {
        "wan2.7-image",
        "wan2.7-image-pro",
        "qwen-audio-3.0-tts-plus",
        "qwen-audio-3.0-realtime-plus",
    }
    assert set(qwencloud_token_plan_models) == gateway_chat_models
    # The two sets partition the listing. This is the only non-redundant claim
    # left once the map is pinned above: it says the snapshot itself is
    # coherent, so an id moved from one literal to the other without being
    # removed from the first fails here rather than quietly widening the map.
    assert gateway_chat_models.isdisjoint(gateway_non_chat_models)
    assert gateway_chat_models | gateway_non_chat_models == {
        "qwen3.8-max",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.6-flash",
        "glm-5.2",
        "deepseek-v4-pro",
        "deepseek-v4-flash-0731",
        "wan2.7-image",
        "wan2.7-image-pro",
        "qwen-audio-3.0-tts-plus",
        "qwen-audio-3.0-realtime-plus",
    }


def test_token_plan_ships_no_row_the_gateway_serves_under_another_id() -> None:
    """`qwen3.8-max-preview` was shipped and then removed, and the removal is
    the point: a completion requested against that id comes back stamped
    ``"model": "qwen3.8-max"``, so it is an ALIAS the gateway resolves rather
    than a distinct SKU. Carrying it as its own row put a second, different
    window (983,616) on the same underlying model and offered a duplicate in the
    picker — and, because the listing does not advertise it, one that only the
    registry believed in."""
    assert "qwen3.8-max-preview" not in qwencloud_token_plan_models


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


def test_deepseek_dated_flash_snapshot_is_not_recommended() -> None:
    """#383: the dated V4 Flash snapshot must not be steered toward.

    It measured 0/5 on the harness's most basic agentic task, emitting literal
    `<|DSML|>` markup as assistant text instead of tool calls, so a new user
    picking the recommended option gets an agent that narrates actions it never
    performs. Both surfaces are asserted because they are read by different
    callers: `RecommendedOpenRouterModelIds` drives the `recommended` flag the
    server computes for OpenRouter/Radient listings, while the catalogue row's
    own `recommended` field is what the direct-DeepSeek listing returns.
    """
    assert "deepseek/deepseek-v4-flash-0731" not in RecommendedOpenRouterModelIds
    # Radient derives from the OpenRouter list, so it inherits the withdrawal;
    # asserted rather than assumed, since a future edit could fork the lists.
    assert "deepseek/deepseek-v4-flash-0731" not in RecommendedRadientModelIds
    assert deepseek_models["deepseek-v4-flash-0731"].recommended is False


def test_withdrawn_deepseek_snapshot_still_resolves_for_existing_configs() -> None:
    """Withdrawing a recommendation must not break a user who already pinned it.

    Deleting the catalogue row would resolve these lookups to the 128k unknown
    default, silently mis-setting the compaction threshold for a 1M-window
    model — a worse outcome than the bad recommendation. The row therefore
    stays; only the flag changes.
    """
    info = get_model_info("deepseek", "deepseek-v4-flash-0731")
    assert info.context_window == 1_048_576
    assert info.max_tokens == 32_768
    # Pricing must survive too, or an existing session's cost ledger silently
    # falls back to the unknown-model zero.
    assert info.input_price == 0.09 and info.output_price == 0.18

    spec = build_model_spec("deepseek", "deepseek-v4-flash-0731")
    assert spec.context_window == 1_048_576

    # The undated alias is a DIFFERENT id and keeps its recommendation: the
    # measured failure is specific to the July snapshot.
    assert "deepseek/deepseek-v4-flash" in RecommendedOpenRouterModelIds


@pytest.mark.parametrize("provider", ["radient", "openrouter"])
def test_an_aggregator_router_resolves_to_a_real_window_and_accepts_images(
    provider: str,
) -> None:
    """The router templates describe a ROUTE, and both facts below are things a
    session acts on rather than cosmetics.

    ``context_window=-1`` was normalised by ``build_model_spec`` to the 128k
    unknown default, and the session derives its compaction threshold from that
    number — so a router that accepts 1M compacted at an eighth of its room,
    and the picker advertised ``128k`` for it. ``supports_images=False`` is a
    positive statement of incapacity in this registry's three-state scheme, not
    an "unknown": it made the session strip images and announce that the model
    does not accept them, which is false for every model these routers select.
    """
    info = get_model_info(provider, "auto")
    assert info.context_window == 1_048_576
    assert info.supports_images is True

    spec = build_model_spec(provider, "auto")
    assert spec.context_window == 1_048_576, "the sentinel no longer collapses to 128k"
    assert spec.supports_images is True, "the session must not strip images on a router"

    # ``max_tokens`` stays unknown ON PURPOSE: the output cap belongs to the
    # SELECTED model and varies by an order of magnitude across the routes, so
    # there is no honest router-wide number to state. Pinned so a later edit
    # that invents one has to argue with this comment first.
    assert info.max_tokens == -1


@pytest.mark.parametrize("provider", ["radient", "openrouter"])
def test_an_arbitrary_aggregator_model_keeps_the_unknown_sentinels(provider: str) -> None:
    """The router's numbers must NOT leak onto the aggregator's other ids.

    The provider template answers for "a model this aggregator serves that we
    know nothing else about", and for that question ``-1`` is the honest
    answer: an arbitrary unlisted model could have any window, and handing it
    the router's 1M would suppress compaction on a small model until the
    provider rejected the turn. Only the ONE known router id gets the real
    numbers, which is why the two rows are separate.
    """
    info = get_model_info(provider, "some-vendor/never-heard-of-it")
    assert info.context_window == -1
    assert info.supports_images is False
