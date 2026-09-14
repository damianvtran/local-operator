import pytest

from local_operator.model import configure
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
    unknown_model_info,
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
        "qwen3.8-flash": (1_000_000, 131_072),
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
    # Snapshot of GET /compatible-mode/v1/models, re-derived 2026-09-04 against
    # the plan's published allowlist. `qwen3.8-flash` is the delta from the
    # 2026-08-19 snapshot: it serves chat (verified live) and is on the
    # allowlist, but had no row, so it resolved to the 128k unknown default.
    gateway_chat_models = {
        "qwen3.8-max",
        "qwen3.8-flash",
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
        "qwen3.8-flash",
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


def test_token_plan_models_declare_prompt_caching() -> None:
    """The flag picks the cache MODE, and the evidence for it is per row.

    `OpenAICompatClient._message_cache_markers` is called only when
    `request.model.supports_prompt_cache` is set, and this provider speaks the
    OpenAI-compatible wire. These rows shipped `False`, so lop emitted no
    `cache_control` at all and the gateway's implicit cache — which cannot be
    disabled — was the only thing running.

    The expected value is keyed by evidence class rather than asserted map-wide,
    because explicit-cache support is per model AND per region on this provider.
    A map-wide `is True` would force the next correctly-added row to carry a
    flag nobody checked, and would read as authority for it. Adding a row here
    fails this test until its evidence class is stated, which is the point.

    The classes, and the note beside the map for the full derivation:

    * MEASURED_EXPLICIT — `cache_creation_input_tokens` observed on the first
      call, which is the only explicit-cache-only signal the wire carries.
    * MARKED_HIT_ONLY — markers sent, second call reported `cached_tokens`, but
      `cache_creation_input_tokens` was not captured. Weaker than it looks: the
      implicit cache reports `cached_tokens` too, so this does not by itself
      prove the marker did anything.
    * INFERRED_INERT — not driven, and/or not on this region's (ap-southeast-1,
      International) explicit-cache list. True because an unrecognised marker is
      ignored and the implicit cache still applies, so the flag is inert here
      rather than wrong.
    """
    measured_explicit = {"qwen3.8-max", "qwen3.8-flash"}
    marked_hit_only = {"qwen3.7-plus", "qwen3.6-flash", "glm-5.2", "deepseek-v4-pro"}
    inferred_inert = {"qwen3.7-max", "deepseek-v4-flash-0731"}

    # Every row is classified exactly once, so a new row cannot ride in on a
    # blanket assertion.
    assert measured_explicit.isdisjoint(marked_hit_only)
    assert inferred_inert.isdisjoint(measured_explicit | marked_hit_only)
    assert (
        measured_explicit | marked_hit_only | inferred_inert
    ) == qwencloud_token_plan_models.keys()

    for model_id, info in qwencloud_token_plan_models.items():
        assert info.supports_prompt_cache is True, model_id

    # The end of the pipe, not just the map: `build_model_spec` is what the
    # client actually reads the flag from.
    assert build_model_spec("alibaba-token-plan", "qwen3.8-max").supports_prompt_cache is True
    assert build_model_spec("alibaba-token-plan", "qwen3.8-flash").supports_prompt_cache is True


def test_token_plan_vision_rows_match_what_the_endpoint_accepts() -> None:
    """Vision is per-row here, and both directions were checked live.

    The plan's allowlist marks `glm-5.2` and the deepseek rows as text-only
    while the qwen rows carry "visual understanding". Verified rather than
    transcribed, because the failure is silent in an unusual way: every row
    ACCEPTS an `image_url` content block without erroring, but the text-only
    ones never receive it — asked the colour of a solid blue PNG,
    `qwen3.8-flash` answers "Blue" while `glm-5.2` and `deepseek-v4-pro` reason
    aloud that no image was provided.

    So an `image_url` block that does not raise is not evidence of vision, and
    marking these rows True on the strength of the request succeeding would
    route a designer or QA subagent to a model that cannot see its screenshot.
    """
    assert qwencloud_token_plan_models["qwen3.8-flash"].supports_images is True
    assert qwencloud_token_plan_models["qwen3.6-flash"].supports_images is True
    assert qwencloud_token_plan_models["qwen3.7-plus"].supports_images is True
    assert qwencloud_token_plan_models["glm-5.2"].supports_images is False
    assert qwencloud_token_plan_models["deepseek-v4-pro"].supports_images is False


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


def test_the_two_router_id_sets_stay_in_agreement() -> None:
    """R3: the same two literals live in two modules, deliberately.

    ``registry`` is the leaf ``discovery`` imports, so the registry cannot
    import the discovery set without a cycle — the duplication is the right
    call. What was missing is anything holding the copies together: adding a
    third router to one side only produces a row whose picker LABEL and whose
    resolved CONTEXT WINDOW disagree, which is silent and confusing rather than
    loud. This pins the invariant so that drift fails here instead.
    """
    from local_operator.model.discovery import _META_ROUTE_IDS
    from local_operator.model.registry import AGGREGATOR_ROUTER_MODEL_IDS

    assert AGGREGATOR_ROUTER_MODEL_IDS == _META_ROUTE_IDS


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


# -- provider-QUALIFIED ids resolve as their bare id -------------------------
#
# The defect these pin: a session whose model is spelled with its PROVIDER —
# ``deepseek/deepseek-flash``, which is the shape a user-supplied ``model_name``,
# a ``--model`` flag, a ``retry.fallbackChains`` hop and a session's own saved
# selection all hand to ``configure.build_model_spec`` — missed the shipped row
# exactly. ``get_model_info`` fell to ``unknown_model_info`` (-1), which
# ``build_model_spec`` normalises into the 128k unknown default, so a 1M-context
# model resumed as 128k with ``max_output_tokens`` dropping 393216 -> 8192 and a
# compaction threshold eight times too small. Observed on this machine: session
# ``de71e4dbcbff`` (titled ``ds-route-smoke``) journalled the selector
# ``deepseek/deepseek/deepseek-flash`` and DeepSeek answered ``400 ... The
# supported API model names are deepseek-flash, deepseek-v4-pro, but you passed
# deepseek/deepseek-flash.``


@pytest.mark.parametrize(
    "hosting, bare_id",
    [
        ("deepseek", "deepseek-flash"),
        ("deepseek", "deepseek-v4-pro"),
        ("deepseek", "deepseek-chat"),
        ("anthropic", "claude-opus-5"),
        ("openai", "gpt-4o"),
        ("google", "gemini-2.0-flash-001"),
        ("xai", "grok-4.6"),
        ("zai", "glm-5.3"),
        ("kimi", "moonshot-v1-8k"),
        ("mistral", "mistral-large-2411"),
        ("alibaba", "qwen2.5-coder-32b-instruct"),
    ],
)
def test_a_qualified_id_resolves_to_the_same_row_as_its_bare_id(hosting: str, bare_id: str) -> None:
    """``<hosting>/<id>`` and ``<id>`` must be one model, on every hosting.

    Field-by-field rather than by identity so the assertion still means
    "the same model" if the lookup ever starts copying rows instead of handing
    out the shipped singletons.
    """
    bare = get_model_info(hosting, bare_id)
    qualified = get_model_info(hosting, f"{hosting}/{bare_id}")

    assert bare is not unknown_model_info, "the bare id must be a real row for this test to bite"
    assert qualified is not unknown_model_info
    assert qualified.id == bare.id
    assert qualified.name == bare.name
    assert qualified.context_window == bare.context_window
    assert qualified.max_tokens == bare.max_tokens
    assert qualified.input_price == bare.input_price
    assert qualified.output_price == bare.output_price


def test_the_deepseek_qualified_id_carries_its_real_1m_window() -> None:
    """The reported symptom, with the operator's own numbers.

    Named separately from the parametrized agreement test because agreeing on
    ``-1`` would satisfy that one: these are the two numbers the operator saw
    wrong (128k and 8192) against the row's real ones.
    """
    qualified = get_model_info("deepseek", "deepseek/deepseek-flash")

    assert qualified.context_window == 1_000_000
    assert qualified.max_tokens == 393_216


@pytest.mark.parametrize("provider", ["openrouter", "radient"])
def test_an_aggregator_vendor_namespace_is_never_stripped(provider: str) -> None:
    """The trap: an aggregator's ``vendor/id`` is the API's id, not a prefix.

    ``deepseek/deepseek-v4.1-flash`` under ``openrouter`` names OpenRouter's
    route to a DeepSeek model, and the harness passes it through whole (the
    failover selector splits on the FIRST slash, so ``provider='openrouter'``
    and ``model_id='deepseek/deepseek-v4.1-flash'``). A retry that stripped any
    leading ``vendor/`` would answer with DeepSeek's DIRECT-route row: a 1M
    window and direct-route prices for a model served over a reseller, which is
    the silent-wrong-answer shape this retry must not introduce.
    """
    direct = get_model_info("deepseek", "deepseek-flash")
    vendor = configure._registry_fallback(provider, "deepseek/deepseek-v4.1-flash")

    assert direct.context_window == 1_000_000
    assert vendor.context_window == -1, "the aggregator placeholder, not DeepSeek's row"
    assert vendor.supports_images is False
    assert vendor is not direct

    # The route id also has to travel WHOLE to the wire: the reseller bills and
    # routes by it, so a rewrite here would be a request for a model this
    # provider's endpoint does not serve.
    spec = build_model_spec(provider, "deepseek/deepseek-v4.1-flash", vendor)
    assert spec.model_id == "deepseek/deepseek-v4.1-flash"


@pytest.mark.parametrize("hosting", ["xai", "zai", "kimi", "anthropic", "google"])
def test_only_the_hosting_that_names_itself_is_stripped(hosting: str) -> None:
    """A prefix naming ANOTHER provider must not resolve to that provider's row.

    ``deepseek/deepseek-flash`` asked of ``xai`` is not an xAI model, and the
    honest answer is the unknown sentinel rather than DeepSeek's 1M row.
    """
    info = get_model_info(hosting, "deepseek/deepseek-flash")

    assert info.context_window == -1


@pytest.mark.parametrize(
    "model",
    [
        "totally-unknown-xyz",
        "deepseek/totally-unknown-xyz",
        # A bare ``deepseek/`` is a malformed id, not a qualified one: the
        # trailing id is empty, so there is nothing to retry and the unknown
        # sentinel must stand rather than resolving to something by accident.
        "deepseek/",
        "deepseek/deepseek/",
    ],
)
def test_an_unknown_id_still_takes_the_unknown_sentinel(model: str) -> None:
    """The 128k fallback must stay reachable rather than weakened into a lie.

    The retry is bounded by the dispatch chain's own answer, so an id nothing
    knows — however it is spelled — keeps returning ``unknown_model_info`` and
    its -1 sentinels; ``build_model_spec``'s 128k normalisation is the honest
    answer there.
    """
    assert get_model_info("deepseek", model) is unknown_model_info


def test_the_openai_branchs_miss_shape_is_preserved() -> None:
    """``openai`` indexes its map directly, so its miss is a ``KeyError``.

    That shape is a contract: ``configure._registry_fallback`` catches it, and a
    bare unshipped openai id must keep failing exactly as it did before the
    qualified-id retry existed — while the QUALIFIED spelling of a shipped id
    must resolve like the bare one rather than raising.
    """
    assert get_model_info("openai", "gpt-4o").id == "gpt-4o"
    assert get_model_info("openai", "openai/gpt-4o").id == "gpt-4o"

    with pytest.raises(KeyError):
        get_model_info("openai", "no-such-openai-model")
    with pytest.raises(KeyError):
        get_model_info("openai", "openai/no-such-openai-model")


def test_the_oauth_spelling_of_the_token_plan_hosting_resolves_too() -> None:
    """R2: the retry must not be gated on a table that omits a supported alias.

    ``alibaba-token-plan-oauth`` is answered by the dispatch chain under the
    CANONICAL ``alibaba-token-plan`` rows, so the qualified spelling used to miss
    while the bare id resolved — the same defect one supported hosting over.
    Gating the retry on the chain's own answer (the unknown sentinel) rather than
    on ``_STATIC_MODEL_MAPS`` is what closes it, without adding the alias key to
    a map that is documented walk-safe for ``static_models``.
    """
    bare = get_model_info("alibaba-token-plan-oauth", "deepseek-v4-flash-0731")
    qualified = get_model_info(
        "alibaba-token-plan-oauth", "alibaba-token-plan-oauth/deepseek-v4-flash-0731"
    )

    assert bare.context_window == 1_000_000
    assert bare is not unknown_model_info
    assert qualified.context_window == bare.context_window
    assert qualified.id == bare.id


@pytest.mark.parametrize("provider", ["ollama", "lmstudio"])
def test_a_local_models_own_name_is_never_stripped(provider: str) -> None:
    """A local runtime's ids are the SERVER's names, not a provider namespace.

    Both Ollama and LM Studio will serve an owner-prefixed HuggingFace name, so
    stripping a leading ``<hosting>/`` there would ask the endpoint for a
    different model.
    """
    from local_operator.model.configure import build_model_spec

    for model_id in (f"{provider}/hf.co/owner/model:Q4", "hf.co/owner/model:Q4"):
        assert build_model_spec(provider, model_id).model_id == model_id


#: OpenRouter's own namespaced ids — real rows in its public catalogue, none of
#: which has a bare counterpart (there is no `auto`/`free`/`fusion` model).
#: F1: the spec-boundary strip treated the aggregator's own name as a PROVIDER
#: prefix, rewrote each of these to its tail, and put a name no provider serves
#: into the request body — a working model failing its first turn.
_OPENROUTER_NAMESPACED_IDS = (
    "openrouter/auto",
    "openrouter/auto-beta",
    "openrouter/fusion",
    "openrouter/pareto-code",
    "openrouter/free",
    "openrouter/bodybuilder",
)


@pytest.mark.parametrize("provider", ["openrouter", "radient"])
@pytest.mark.parametrize("model_id", _OPENROUTER_NAMESPACED_IDS)
def test_an_aggregators_own_namespace_survives_the_spec_boundary(
    provider: str, model_id: str
) -> None:
    """An aggregator's ids travel WHOLE, asserted on the RENDERED body.

    ``spec.model_id`` is not a label — ``providers/clients.py::_build_body`` sets
    ``"model": request.model.model_id`` — so an id-equality assertion alone is
    the reason F1 had no test that could fail. This renders the body through the
    real client, for each of the six published ids on both aggregator hostings.

    ``radient-key`` is deliberately not a case: it sits in
    ``AGGREGATOR_PROVIDERS`` but is not a hosting ``configure_model`` accepts at
    all (it raises ``Unsupported hosting provider``), so its membership only
    matters to the exclusion set, not to a spec build.
    """
    from local_operator.harness.types import ChatRequest, Message, TextContent
    from local_operator.model.configure import build_model_spec
    from local_operator.providers.clients import OpenAICompatClient

    info = get_model_info(provider, model_id)
    spec = build_model_spec(provider, model_id, info)
    body = OpenAICompatClient("https://example.invalid")._build_body(
        ChatRequest(model=spec, messages=[Message(role="user", content=[TextContent(text="hi")])])
    )

    assert spec.model_id == model_id
    assert body["model"] == model_id


def test_the_aggregator_exclusion_does_not_loosen_the_vendor_trap() -> None:
    """The exclusion is by HOSTING, so the vendor-namespace rule still holds.

    An id whose prefix names another provider under an aggregator hosting is
    still never rewritten, and a direct hosting whose model name carries its own
    prefix is still canonicalised (the deepseek case this PR exists for).
    """
    from local_operator.model.configure import build_model_spec

    assert (
        build_model_spec(
            "openrouter", "deepseek/deepseek-v4.1-flash", get_model_info("openrouter", "x")
        ).model_id
        == "deepseek/deepseek-v4.1-flash"
    )
    assert build_model_spec("deepseek", "deepseek/deepseek-flash").model_id == "deepseek-flash"
    assert build_model_spec("deepseek", "deepseek-flash").model_id == "deepseek-flash"


def test_the_anthropic_family_resolver_and_template_still_answer() -> None:
    """The two answers that exist for ids the registry does not ship.

    Neither may be reached differently now: the family resolver owns a dated
    snapshot's real window (Opus 5 serves 1M), and the template owns the floor
    for an id whose tier cannot be parsed at all — the global unknown sentinel
    would put 128k/8192 on a Claude, numbers no Claude generation has had.
    """
    family = anthropic_family_model_info("claude-opus-5-20260112")
    assert family is not None
    assert family.context_window == 1_000_000

    templated = configure._registry_fallback("anthropic", "claude-unheard-of-6")
    assert templated is not unknown_model_info
    assert templated.context_window == 200_000
    assert templated.id == "claude-unheard-of-6"


@pytest.fixture
def offline_resolution(monkeypatch, tmp_path):
    """Resolve metadata with neither a network nor a shared disk cache.

    The enrichment legs are held at the registry's own answer, so a window that
    moves here can only have moved in the registry — which is the layer this
    change touches.
    """
    configure.invalidate_model_info_cache()
    monkeypatch.setattr(configure, "_from_price_catalogue", lambda p, m, info, **kw: info)
    monkeypatch.setattr(configure, "_from_aggregator_catalogue", lambda p, m, info, **kw: info)
    from local_operator.model import discovery as discovery_mod

    monkeypatch.setattr(discovery_mod, "available_models", lambda provider_id, **kw: ([], "ok"))
    yield
    configure.invalidate_model_info_cache()


def test_the_spec_built_from_a_qualified_id_carries_the_real_window(
    offline_resolution,
) -> None:
    """The user-visible half: the SPEC the session runs on, not the registry row.

    Compaction thresholds derive from ``ModelSpec.context_window`` and the wire
    from ``max_output_tokens``, so this is the assertion that matches what the
    operator saw (128k) and what the fix must produce (1M / 393216).
    """
    bare = build_model_spec("deepseek", "deepseek-flash")
    qualified = build_model_spec("deepseek", "deepseek/deepseek-flash")

    assert bare.context_window == 1_000_000
    assert qualified.context_window == bare.context_window
    assert bare.max_output_tokens == 393_216
    assert qualified.max_output_tokens == bare.max_output_tokens


def test_the_spec_canonicalises_a_qualified_model_name_for_the_wire(
    offline_resolution,
) -> None:
    """R1: the wire's model field must be the id the provider actually serves.

    ``build_model_spec`` is the ONE boundary every path builds a spec through, so
    it is where the ``provider/model`` selector spelling is canonicalised. The
    body is rendered through the real client, because ``spec.model_id`` is not a
    label — it is literally what the request carries. Measured live against
    ``api.deepseek.com``: the bare spelling answers HTTP 200 and the qualified
    one HTTP 400 ("The supported API model names are deepseek-flash,
    deepseek-v4-pro, but you passed deepseek/deepseek-flash.").
    """
    from local_operator.harness.types import ChatRequest, Message, TextContent
    from local_operator.providers.clients import OpenAICompatClient

    client = OpenAICompatClient("https://api.deepseek.com")
    seen = []
    for model_id in ("deepseek-flash", "deepseek/deepseek-flash"):
        spec = build_model_spec("deepseek", model_id)
        body = client._build_body(
            ChatRequest(
                model=spec, messages=[Message(role="user", content=[TextContent(text="hi")])]
            )
        )
        seen.append(body["model"])
        assert spec.model_id == "deepseek-flash"

    assert seen == ["deepseek-flash", "deepseek-flash"]


@pytest.mark.parametrize("provider", ["radient", "openrouter"])
def test_aggregator_router_rows_advertise_prompt_caching(provider: str) -> None:
    """The router row must not claim its route cannot cache.

    `supports_prompt_cache` gates two emissions on the chat-completions wire, and
    both of them exist to keep a long conversation's prompt cache warm: the
    `prompt_cache_key` OpenRouter uses as its sticky-routing key, and the
    `cache_control` marker that asks a provider to cache the prefix. A False here
    therefore did not mean "unknown" — it meant every router request was sent with
    no routing key and no cache marker, so a session paid full input price on every
    turn after the first, and OpenRouter fell back to an affinity it only
    establishes after a hit it had no way to earn.

    True is the honest value because the router is a route and today every route
    caches; the asymmetry is what settles it, since a hint or a marker sent to a
    model that does not cache is inert, while withholding them from one that does
    costs the full prefix on every turn.
    """
    info = get_model_info(provider, "auto")

    assert info.supports_prompt_cache is True

    spec = build_model_spec(provider, "auto", info)
    assert spec.supports_prompt_cache is True, "the spec the request builder reads must carry it"
