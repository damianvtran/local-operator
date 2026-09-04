from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from local_operator.providers.local import LOCAL_PRESETS, LOCAL_PROVIDER_IDS


class ProviderDetail(BaseModel):
    """Model for provider details.

    Attributes:
        id: Unique identifier for the provider
        name: Display name for the provider
        description: Description of the provider
        url: URL to the provider's platform
        requiredCredentials: List of required credential keys
    """

    id: str = Field(..., description="Unique identifier for the provider")
    name: str = Field(..., description="Display name for the provider")
    description: str = Field(..., description="Description of the provider")
    url: str = Field(..., description="URL to the provider's platform")
    requiredCredentials: List[str] = Field(..., description="List of required credential keys")
    # `default=` must be spelled as a keyword: a positional default in
    # `Field()` is honoured at runtime but is not recognised as a default by
    # static analysis, which then treats the field as required and flags every
    # construction site that omits it.
    recommended: bool = Field(
        default=False,
        description="Whether the provider is recommended for use in Local Operator",
    )


SupportedHostingProviders = [
    ProviderDetail(
        id="radient",
        name="Radient",
        description=(
            "Your Radient Pass provides you unified access to a variety of high end AI "
            "models and tools.  Radient makes using agentic AI simple and easy with "
            "transparent pricing, and helps you pick the best model for your use case."
        ),
        url="https://radienthq.com/",
        requiredCredentials=["RADIENT_API_KEY"],
        recommended=True,
    ),
    ProviderDetail(
        id="openai",
        name="OpenAI",
        description="OpenAI's API provides access to GPT-4o and other models",
        url="https://platform.openai.com/",
        requiredCredentials=["OPENAI_API_KEY"],
        recommended=True,
    ),
    ProviderDetail(
        id="anthropic",
        name="Anthropic",
        description="Anthropic's Claude models for AI assistants",
        url="https://www.anthropic.com/",
        requiredCredentials=["ANTHROPIC_API_KEY"],
        recommended=True,
    ),
    ProviderDetail(
        id="google",
        name="Google",
        description="Google's Gemini models for multimodal AI capabilities",
        url="https://ai.google.dev/",
        requiredCredentials=["GOOGLE_AI_STUDIO_API_KEY"],
        recommended=True,
    ),
    ProviderDetail(
        id="mistral",
        name="Mistral AI",
        description="Mistral AI's open and proprietary language models",
        url="https://mistral.ai/",
        requiredCredentials=["MISTRAL_API_KEY"],
        recommended=True,
    ),
    *[
        ProviderDetail(
            id=provider,
            name=name,
            description="Connect a user-operated OpenAI-compatible model server",
            url=url,
            requiredCredentials=[],
        )
        for provider, (name, _endpoint, url) in LOCAL_PRESETS.items()
    ],
    ProviderDetail(
        id="openrouter",
        name="OpenRouter",
        description="Access to multiple AI models through a unified API",
        url="https://openrouter.ai/",
        requiredCredentials=["OPENROUTER_API_KEY"],
        recommended=True,
    ),
    ProviderDetail(
        id="deepseek",
        name="DeepSeek",
        description="DeepSeek's language models for various AI applications",
        url="https://deepseek.ai/",
        requiredCredentials=["DEEPSEEK_API_KEY"],
        recommended=True,
    ),
    ProviderDetail(
        id="kimi",
        name="Kimi",
        description="Moonshot AI's Kimi models for Chinese and English language tasks",
        url="https://moonshot.cn/",
        requiredCredentials=["KIMI_API_KEY"],
        recommended=False,
    ),
    ProviderDetail(
        id="alibaba",
        name="Alibaba Cloud",
        description="Alibaba's Qwen models for natural language processing",
        url="https://www.alibabacloud.com/",
        requiredCredentials=["ALIBABA_CLOUD_API_KEY"],
        recommended=False,
    ),
    ProviderDetail(
        id="xai",
        name="xAI",
        description="xAI's Grok models for natural language processing",
        url="https://x.ai/",
        requiredCredentials=["XAI_API_KEY"],
        recommended=True,
    ),
    ProviderDetail(
        id="zai",
        name="Z.AI (GLM)",
        description=(
            "Z.AI's GLM models for coding and reasoning, with large context windows "
            "and a subscription coding plan that reports live quota"
        ),
        url="https://z.ai/",
        requiredCredentials=["ZAI_API_KEY"],
        recommended=True,
    ),
]
"""List of supported model hosting providers.

This list contains the names of all supported AI model hosting providers that can be used
with the Local Operator API. Each provider has its own set of available models and pricing.

The supported providers are:
- radient: Radient Pass model hosting with automatic model selection and unified tool access
- anthropic: Anthropic's Claude models
- ollama: Local model hosting with Ollama
- deepseek: DeepSeek's language models
- google: Google's Gemini models
- openai: OpenAI's GPT models
- openrouter: OpenRouter model aggregator
- alibaba: Alibaba's Qwen models
- kimi: Kimi AI's models
- mistral: Mistral AI's models
"""

RecommendedOpenRouterModelIds = [
    "anthropic/claude-sonnet-4",
    "anthropic/claude-3.7-sonnet",
    "openai/gpt-4.1",
    "mistralai/mistral-large-2411",
    "mistralai/mistral-large-2407",
    "mistralai/mistral-large",
    "x-ai/grok-3-beta",
    "google/gemini-2.5-pro-preview",
    # `deepseek/deepseek-v4-flash-0731` is deliberately ABSENT despite still
    # shipping a catalogue row below. Measured 0/5 on the harness's most basic
    # agentic task ("create a file called test.txt"): instead of emitting tool
    # calls it returns literal `<|DSML|>tool_calls>` markup as assistant TEXT,
    # so no tool ever executes and the agent narrates work it never performed.
    # In a tool-calling harness that reads as the product being broken, which
    # is worse than recommending nothing. The undated `deepseek-v4-flash` alias
    # stays because the failure is a regression in that pinned snapshot; do not
    # re-add the dated id without re-measuring it.
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-chat-v3.1",
]
"""List of recommended model IDs from OpenRouter.

This list contains the model IDs of recommended models available through the OpenRouter
provider. These models are selected based on performance, reliability, and community
feedback. The IDs follow the format 'provider/model-name' as used by OpenRouter's API.

The list includes models from various providers:
- Google's Gemini models
- Anthropic's Claude models
- OpenAI's GPT models
- Qwen models
- Mistral AI models
"""

RecommendedRadientModelIds = RecommendedOpenRouterModelIds + ["auto"]
"""List of recommended model IDs from Radient.

This list contains the model IDs of recommended models available through the Radient
provider. These models are selected based on performance, reliability, and community
feedback. The IDs follow the format 'provider/model-name' as used by OpenRouter's API.

The list includes models from various providers:
- Google's Gemini models
- Anthropic's Claude models
- OpenAI's GPT models
- Qwen models
- Mistral AI models
"""


class ModelInfo(BaseModel):
    """
    Represents the pricing information for a given model.

    Attributes:
        input_price (float): Cost per million input tokens.
        output_price (float): Cost per million output tokens.
        max_tokens (Optional[int]): Maximum number of tokens supported by the model.
        context_window (Optional[int]): Context window size of the model.
        supports_images (Optional[bool]): Whether the model supports images.
        supports_prompt_cache (bool): Whether the model supports prompt caching.
        supports_responses_api (bool): Whether the model supports OpenAI's
            Responses API.
        cache_writes_price (Optional[float]): Cost per million tokens for cache writes.
        cache_reads_price (Optional[float]): Cost per million tokens for cache reads.
        description (Optional[str]): Description of the model.
        limits_from_listing (bool): The window and max_tokens on this row were
        transcribed from the provider's own live listing on a date, so a live
        listing supersedes them.
        recommended (Optional[bool]): Whether the model is recommended for use in Local
        Operator.  This is determined based on community usage and feedback.
    """

    input_price: float = 0.0
    output_price: float = 0.0
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None
    supports_images: Optional[bool] = None
    supports_prompt_cache: bool = False
    supports_responses_api: bool = False
    cache_writes_price: Optional[float] = None
    cache_reads_price: Optional[float] = None
    description: str = Field(..., description="Description of the model")
    id: str = Field(..., description="Unique identifier for the model")
    name: str = Field(..., description="Display name for the model")
    # Route-local provenance, never filled from a different provider's limits.
    # Existing context_window remains the active budget for legacy consumers.
    default_context_window: int | None = None
    max_context_window: int | None = None
    limits_from_listing: bool = Field(
        default=False,
        description=(
            "This row's context_window/max_tokens were TRANSCRIBED from the provider's "
            "own live listing on a date, rather than being independent knowledge. A live "
            "listing therefore supersedes them and resolution should go and ask."
        ),
    )
    recommended: bool = Field(
        default=False,
        description=(
            "Whether the model is recommended for use in Local Operator. "
            "This is determined based on community usage and feedback."
        ),
    )

    @field_validator("input_price", "output_price")
    def price_must_be_non_negative(cls, value: float) -> float:
        """Validates that the price is non-negative."""
        if value < 0:
            raise ValueError("Price must be non-negative.")
        return value


def get_model_info(hosting: str, model: str) -> ModelInfo:
    """
    Retrieves the model information based on the hosting provider and model name.

    This function checks a series of known hosting providers and their associated
    models to return a `ModelInfo` object containing relevant details such as
    pricing, context window, and image support. If the hosting provider is not
    supported, a ValueError is raised. If the model is not found for a supported
    hosting provider, a default `unknown_model_info` is returned.

    Args:
        hosting (str): The hosting provider name (e.g., "openai", "google").
        model (str): The model name (e.g., "gpt-3.5-turbo", "gemini-1.0-pro").

    Returns:
        ModelInfo: The model information for the specified hosting and model.
                   Returns `unknown_model_info` if the model is not found for a
                   supported hosting provider.

    Raises:
        ValueError: If the hosting provider is unsupported.
    """
    model_info = unknown_model_info

    if hosting == "radient":
        # The router is a KNOWN endpoint and gets its own row; every other id
        # an aggregator serves is genuinely undescribed and keeps the template's
        # unknown sentinels. See `aggregator_router_model_info` for why the two
        # cannot share one row.
        if model in AGGREGATOR_ROUTER_MODEL_IDS:
            return aggregator_router_model_info
        return radient_default_model_info
    elif hosting == "anthropic":
        if model in anthropic_models:
            model_info = anthropic_models[model]
    elif hosting in LOCAL_PROVIDER_IDS:
        return ollama_default_model_info.model_copy(
            update={"id": hosting, "name": LOCAL_PRESETS[hosting][0]}
        )
    elif hosting == "deepseek":
        if model in deepseek_models:
            return deepseek_models[model]
    elif hosting == "google":
        if model in google_models:
            return google_models[model]
    elif hosting == "openai":
        return openai_models[model]
    elif hosting == "openrouter":
        if model in AGGREGATOR_ROUTER_MODEL_IDS:
            return aggregator_router_model_info
        return openrouter_default_model_info
    elif hosting == "alibaba":
        if model in qwen_models:
            return qwen_models[model]
    elif hosting in ("alibaba-token-plan", "alibaba-token-plan-oauth"):
        # Both login flavours serve the same catalogue; `store_credentials_as`
        # already aliases the oauth id to `alibaba-token-plan` on the discovery
        # path, but this chain is reachable directly (build_model_spec on a
        # session config), so it must answer for both spellings itself.
        if model in qwencloud_token_plan_models:
            return qwencloud_token_plan_models[model]
    elif hosting == "kimi":
        if model in kimi_models:
            return kimi_models[model]
    elif hosting == "mistral":
        if model in mistral_models:
            return mistral_models[model]
    elif hosting == "xai":
        if model in xai_models:
            return xai_models[model]
    elif hosting == "zai":
        if model in glm_models:
            return glm_models[model]
    else:
        raise ValueError(f"Unsupported hosting provider: {hosting}")

    return model_info


def static_models(hosting: str) -> Dict[str, "ModelInfo"]:
    """Every model this module knows for ``hosting``, keyed by model id.

    :func:`get_model_info` answers "describe THIS model" and cannot answer "what
    models exist", because its dispatch is a chain of `if hosting ==` branches
    with the maps closed inside it. Live discovery needs the second question: it
    merges a provider's API listing over what we shipped, and the shipped side has
    to be enumerable for that merge to have a left-hand side.

    Returns a COPY, so a caller that annotates rows in place — which a merge does
    by construction — cannot mutate the process-wide catalogue.

    Aggregators (openrouter, radient) and local runtimes (ollama) return ``{}``
    rather than their one placeholder entry. Their placeholder describes the
    ROUTER, not a model: offering `openrouter/openrouter` as a choice would be
    offering a model that does not exist, and for exactly those providers the live
    listing is authoritative anyway.
    """
    return dict(_STATIC_MODEL_MAPS.get(hosting, {}))


unknown_model_info: ModelInfo = ModelInfo(
    id="unknown",
    name="Unknown",
    max_tokens=-1,
    context_window=-1,
    supports_images=False,
    supports_prompt_cache=False,
    input_price=0.0,
    output_price=0.0,
    description="Unknown model with default settings",
    recommended=False,
)
"""
Default ModelInfo when model is unknown.

This ModelInfo is returned by `get_model_info` when a specific model
is not found within a supported hosting provider's catalog. It provides
a fallback with negative max_tokens and context_window to indicate
the absence of specific model details.
"""

anthropic_models: Dict[str, ModelInfo] = {
    # Numbers below are the provider's OWN, read from
    # `GET https://api.anthropic.com/v1/models?limit=50` on 2026-08-07: every entry
    # carries `max_input_tokens`, `max_tokens` and a `capabilities` object. They are
    # transcribed rather than inferred, because the whole point of these rows is to
    # be right when the listing cannot be reached — the shipped 200k family floor is
    # what made a 1M-context Opus 5 session compact at 160k and report `1.8%/200k`.
    #
    # Prices are Anthropic's OWN, transcribed from the "Model pricing" table at
    # https://platform.claude.com/docs/en/about-claude/pricing read 2026-08-10, in
    # this registry's unit of dollars per MILLION tokens. `/v1/models` quotes no
    # prices at all, which is why they arrived here as 0.0 placeholders and stayed
    # that way: 0.0 is this registry's "unknown", so the status band rendered
    # "cost unavailable" for the entire current generation while the older
    # OpenRouter-priced rows costed fine. Add a price here only from that page.
    #
    # `cache_writes_price` is the FIVE-MINUTE write (1.25x base input), not the
    # 1h write (2x): the Anthropic client sends `cache_control: {"type":
    # "ephemeral"}` with no `ttl` (clients.py:426/755), which is the 5m cache.
    # `cache_reads_price` is the cache hit at 0.1x base input.
    #
    # `supports_prompt_cache=True` is a family property (every Claude from 3 on
    # accepts `cache_control`) rather than a listing field; `supports_images` IS a
    # listing field (`capabilities.image_input.supported`) and is True for all ten.
    "claude-opus-5": ModelInfo(
        id="claude-opus-5",
        name="Claude Opus 5",
        max_tokens=128_000,
        context_window=1_000_000,
        supports_images=True,
        supports_prompt_cache=True,
        limits_from_listing=True,
        input_price=5.0,  # $5 / MTok
        output_price=25.0,  # $25 / MTok
        cache_writes_price=6.25,  # $6.25 / MTok (5m write)
        cache_reads_price=0.50,  # $0.50 / MTok
        description=(
            "Anthropic's Claude Opus 5 flagship: 1M-token context window and 128k "
            "of output, with adaptive thinking and effort tiers up to max."
        ),
        recommended=False,
    ),
    "claude-sonnet-5": ModelInfo(
        id="claude-sonnet-5",
        name="Claude Sonnet 5",
        max_tokens=128_000,
        context_window=1_000_000,
        supports_images=True,
        supports_prompt_cache=True,
        limits_from_listing=True,
        # These are Sonnet 5's STANDARD rates, not a promotion, and they are not
        # dated. The history matters because it is the opposite of what an older
        # comment here claimed: the model launched 2026-06-30 at $2/$10 billed as
        # introductory pricing through 2026-08-31, with a rise to $3/$15/$3.75/$0.30
        # scheduled for 2026-09-01. On 2026-08-10 Anthropic cancelled that rise and
        # made these numbers permanent (launch-post changelog edit of 2026-08-10 on
        # https://www.anthropic.com/news/claude-sonnet-5, and the
        # `claude-sonnet-5-introductory-pricing` note on
        # https://platform.claude.com/docs/en/about-claude/pricing: "The previously
        # scheduled increase to $3/$15 ... will not occur").
        #
        # So do NOT "restore" $3/$15 here on the strength of a stale third-party
        # pricing table — several still carry the cancelled increase. Anything other
        # than these four values over-reports every Sonnet 5 call in the status band
        # and the analytics ledger, which is what `test_sonnet_5_carries_the_standard_price`
        # pins against.
        input_price=2.0,  # $2 / MTok
        output_price=10.0,  # $10 / MTok
        cache_writes_price=2.50,  # $2.50 / MTok (5m write)
        cache_reads_price=0.20,  # $0.20 / MTok
        description=(
            "Claude Sonnet 5: the balanced tier of the 5 generation, with the same "
            "1M-token context window and 128k output as Opus 5."
        ),
        recommended=False,
    ),
    "claude-fable-5": ModelInfo(
        id="claude-fable-5",
        name="Claude Fable 5",
        max_tokens=128_000,
        context_window=1_000_000,
        supports_images=True,
        supports_prompt_cache=True,
        limits_from_listing=True,
        input_price=10.0,  # $10 / MTok
        output_price=50.0,  # $50 / MTok
        cache_writes_price=12.50,  # $12.50 / MTok (5m write)
        cache_reads_price=1.0,  # $1 / MTok
        description="Claude Fable 5: 1M-token context window and 128k of output.",
        recommended=False,
    ),
    "claude-opus-4-8": ModelInfo(
        id="claude-opus-4-8",
        name="Claude Opus 4.8",
        max_tokens=128_000,
        context_window=1_000_000,
        supports_images=True,
        supports_prompt_cache=True,
        limits_from_listing=True,
        input_price=5.0,  # $5 / MTok
        output_price=25.0,  # $25 / MTok
        cache_writes_price=6.25,  # $6.25 / MTok (5m write)
        cache_reads_price=0.50,  # $0.50 / MTok
        description="Claude Opus 4.8: 1M-token context window and 128k of output.",
        recommended=False,
    ),
    "claude-opus-4-7": ModelInfo(
        id="claude-opus-4-7",
        name="Claude Opus 4.7",
        max_tokens=128_000,
        context_window=1_000_000,
        supports_images=True,
        supports_prompt_cache=True,
        limits_from_listing=True,
        input_price=5.0,  # $5 / MTok
        output_price=25.0,  # $25 / MTok
        cache_writes_price=6.25,  # $6.25 / MTok (5m write)
        cache_reads_price=0.50,  # $0.50 / MTok
        description="Claude Opus 4.7: 1M-token context window and 128k of output.",
        recommended=False,
    ),
    "claude-opus-4-6": ModelInfo(
        id="claude-opus-4-6",
        name="Claude Opus 4.6",
        max_tokens=128_000,
        context_window=1_000_000,
        supports_images=True,
        supports_prompt_cache=True,
        limits_from_listing=True,
        input_price=5.0,  # $5 / MTok
        output_price=25.0,  # $25 / MTok
        cache_writes_price=6.25,  # $6.25 / MTok (5m write)
        cache_reads_price=0.50,  # $0.50 / MTok
        description="Claude Opus 4.6: 1M-token context window and 128k of output.",
        recommended=False,
    ),
    "claude-sonnet-4-6": ModelInfo(
        id="claude-sonnet-4-6",
        name="Claude Sonnet 4.6",
        max_tokens=128_000,
        context_window=1_000_000,
        supports_images=True,
        supports_prompt_cache=True,
        limits_from_listing=True,
        input_price=3.0,  # $3 / MTok
        output_price=15.0,  # $15 / MTok
        cache_writes_price=3.75,  # $3.75 / MTok (5m write)
        cache_reads_price=0.30,  # $0.30 / MTok
        description="Claude Sonnet 4.6: 1M-token context window and 128k of output.",
        recommended=False,
    ),
    # The 4.5 snapshots are the generation where the tiers stopped agreeing on a
    # window: Sonnet 4.5 serves 1M while Opus 4.5 and Haiku 4.5 serve 200k. Nothing
    # may infer one from the other — that inference is the bug this block fixes.
    "claude-opus-4-5-20251101": ModelInfo(
        id="claude-opus-4-5-20251101",
        name="Claude Opus 4.5 (2025-11-01)",
        max_tokens=64_000,
        context_window=200_000,
        supports_images=True,
        supports_prompt_cache=True,
        limits_from_listing=True,
        input_price=5.0,  # $5 / MTok
        output_price=25.0,  # $25 / MTok
        cache_writes_price=6.25,  # $6.25 / MTok (5m write)
        cache_reads_price=0.50,  # $0.50 / MTok
        description="Claude Opus 4.5: 200k-token context window and 64k of output.",
        recommended=False,
    ),
    "claude-sonnet-4-5-20250929": ModelInfo(
        id="claude-sonnet-4-5-20250929",
        name="Claude Sonnet 4.5 (2025-09-29)",
        max_tokens=64_000,
        context_window=1_000_000,
        supports_images=True,
        supports_prompt_cache=True,
        limits_from_listing=True,
        input_price=3.0,  # $3 / MTok
        output_price=15.0,  # $15 / MTok
        cache_writes_price=3.75,  # $3.75 / MTok (5m write)
        cache_reads_price=0.30,  # $0.30 / MTok
        description="Claude Sonnet 4.5: 1M-token context window and 64k of output.",
        recommended=False,
    ),
    "claude-haiku-4-5-20251001": ModelInfo(
        id="claude-haiku-4-5-20251001",
        name="Claude Haiku 4.5 (2025-10-01)",
        max_tokens=64_000,
        context_window=200_000,
        supports_images=True,
        supports_prompt_cache=True,
        limits_from_listing=True,
        input_price=1.0,  # $1 / MTok
        output_price=5.0,  # $5 / MTok
        cache_writes_price=1.25,  # $1.25 / MTok (5m write)
        cache_reads_price=0.10,  # $0.10 / MTok
        description="Claude Haiku 4.5: 200k-token context window and 64k of output.",
        recommended=False,
    ),
    "claude-opus-4-20250514": ModelInfo(
        id="claude-opus-4-20250514",
        name="Claude Opus 4 (2025-05-14)",
        max_tokens=32_000,
        context_window=200_000,
        supports_images=True,
        supports_prompt_cache=True,
        input_price=15.0,  # $15 / MTok
        output_price=18.75,  # $18.75 / MTok
        cache_writes_price=30.0,  # $30 / MTok
        cache_reads_price=1.50,  # $1.50 / MTok
        description=(
            "Anthropic's most capable and intelligent model yet. Claude Opus 4 sets new "
            "standards in complex reasoning and advanced coding."
        ),
        recommended=False,
    ),
    "claude-sonnet-4-20250514": ModelInfo(
        id="claude-sonnet-4-20250514",
        name="Claude Sonnet 4 (2025-05-14)",
        max_tokens=64_000,
        context_window=200_000,
        supports_images=True,
        supports_prompt_cache=True,
        input_price=3.0,
        output_price=15.0,
        cache_writes_price=3.75,
        cache_reads_price=0.30,
        description=(
            "Anthropic's high-performance model with exceptional reasoning and efficiency."
        ),
        recommended=True,
    ),
    "claude-3-7-sonnet-latest": ModelInfo(
        id="claude-3-7-sonnet-latest",
        name="Claude 3.7 Sonnet (Latest)",
        max_tokens=8192,
        context_window=200_000,
        supports_images=True,
        supports_prompt_cache=True,
        input_price=3.0,
        output_price=15.0,
        cache_writes_price=3.75,
        cache_reads_price=0.30,  # $0.30 / MTok (0.1x base)
        description=(
            "Anthropic's latest and most powerful model for coding and agentic "
            "tasks.  Latest version."
        ),
        recommended=True,
    ),
    "claude-3-7-sonnet-20250219": ModelInfo(
        id="claude-3-7-sonnet-20250219",
        name="Claude 3.7 Sonnet (2025-02-19)",
        max_tokens=8192,
        context_window=200_000,
        supports_images=True,
        supports_prompt_cache=True,
        input_price=3.0,
        output_price=15.0,
        cache_writes_price=3.75,
        cache_reads_price=0.30,  # $0.30 / MTok (0.1x base)
        description=(
            "Anthropic's latest and most powerful model for coding and agentic "
            "tasks.  Snapshot from February 2025."
        ),
        recommended=True,
    ),
    "claude-3-5-sonnet-20241022": ModelInfo(
        id="claude-3-5-sonnet-20241022",
        name="Claude 3.5 Sonnet",
        max_tokens=8192,
        context_window=200_000,
        supports_images=True,
        supports_prompt_cache=True,
        input_price=3.0,
        output_price=15.0,
        cache_writes_price=3.75,
        cache_reads_price=0.30,  # $0.30 / MTok (0.1x base)
        description="Anthropic's latest balanced model with excellent performance",
        recommended=True,
    ),
    "claude-3-5-haiku-20241022": ModelInfo(
        id="claude-3-5-haiku-20241022",
        name="Claude 3.5 Haiku (2024-10-22)",
        max_tokens=8192,
        context_window=200_000,
        supports_images=False,
        supports_prompt_cache=True,
        input_price=0.8,
        output_price=4.0,
        cache_writes_price=1.0,
        cache_reads_price=0.08,  # $0.08 / MTok, published per-model
        description="Fast and efficient model for simpler tasks",
        recommended=False,
    ),
    "claude-3-opus-20240229": ModelInfo(
        id="claude-3-opus-20240229",
        name="Claude 3 Opus (2024-02-29)",
        max_tokens=4096,
        context_window=200_000,
        supports_images=True,
        supports_prompt_cache=True,
        input_price=15.0,
        output_price=75.0,
        cache_writes_price=18.75,
        cache_reads_price=1.5,
        description="Anthropic's most powerful model for complex tasks",
        recommended=False,
    ),
    "claude-3-haiku-20240307": ModelInfo(
        id="claude-3-haiku-20240307",
        name="Claude 3 Haiku (2024-03-07)",
        max_tokens=4096,
        context_window=200_000,
        supports_images=True,
        supports_prompt_cache=True,
        input_price=0.25,
        output_price=1.25,
        cache_writes_price=0.3,
        cache_reads_price=0.025,  # $0.025 / MTok (0.1x base)
        description="Fast and efficient model for simpler tasks",
        recommended=False,
    ),
}

# TODO: Add fetch for token, context window, image support
ollama_default_model_info: ModelInfo = ModelInfo(
    max_tokens=-1,
    context_window=-1,
    supports_images=False,
    supports_prompt_cache=False,
    input_price=0.0,
    output_price=0.0,
    description="Local model hosting with Ollama",
    id="ollama",
    name="Ollama",
    recommended=False,
)

openrouter_default_model_info: ModelInfo = ModelInfo(
    max_tokens=-1,
    context_window=-1,
    supports_images=False,
    supports_prompt_cache=False,
    input_price=0.0,
    output_price=0.0,
    cache_writes_price=0.0,
    cache_reads_price=0.0,
    description="Access to various AI models from different providers through a single API",
    id="openrouter",
    name="OpenRouter",
    recommended=False,
)

radient_default_model_info: ModelInfo = ModelInfo(
    max_tokens=-1,
    context_window=-1,
    supports_images=False,
    supports_prompt_cache=False,
    input_price=0.0,
    output_price=0.0,
    cache_writes_price=0.0,
    cache_reads_price=0.0,
    description="Access to Radient AI models through their API",
    id="radient",
    name="Radient",
    recommended=False,
)

#: The ROUTER endpoint of an aggregator — `radient/auto`, `openrouter/auto` —
#: as distinct from a model reached THROUGH that aggregator.
#:
#: Deliberately separate from the two templates above, and the distinction is
#: the whole point. Those describe "a model this aggregator serves that we know
#: nothing else about", so their ``-1`` is honest: an arbitrary unlisted model
#: could have any window, and claiming a large one would suppress compaction
#: for a small model — the failure the ``-1`` sentinel exists to prevent. The
#: router is the opposite case. It is ONE known endpoint whose product is
#: dispatching to a frontier model, so these are properties of the ROUTE:
#:
#: * ``context_window``: ``-1`` is normalised by ``configure.build_model_spec``
#:   to ``UNKNOWN_CONTEXT_WINDOW`` (128k), and that number is not inert — the
#:   session derives its compaction threshold from it, so a router that will
#:   happily accept 1M compacted at an eighth of its room, and the picker
#:   advertised ``128k`` for it. 1,048,576 is a window every model these
#:   routers select today actually carries, so it is a conservative floor
#:   rather than an optimistic claim, and a live listing that states its own
#:   number still wins (``discovery._merge_one``).
#: * ``supports_images``: ``False`` is a POSITIVE STATEMENT OF INCAPACITY in
#:   the three-state scheme ``DiscoveredModel.supports_images`` documents, not
#:   an "unknown" — and it was a false one. It made the session strip images
#:   and announce "the current model does not accept images" on a router that
#:   accepts them, which is the user-visible half of this bug.
#:
#: ``max_tokens`` deliberately stays ``-1``. Unlike the window it has no honest
#: router-wide floor: the output cap is the SELECTED model's, it varies by an
#: order of magnitude across the routes, and ``build_model_spec``'s 8,192
#: fallback merely truncates a long answer where a wrong window silently
#: mis-compacts an entire conversation. Stating a number nobody can stand
#: behind is the error these templates exist to avoid.
aggregator_router_model_info: ModelInfo = ModelInfo(
    max_tokens=-1,
    context_window=1_048_576,
    supports_images=True,
    supports_prompt_cache=False,
    input_price=0.0,
    output_price=0.0,
    cache_writes_price=0.0,
    cache_reads_price=0.0,
    description="Automatic model selection across the aggregator's catalogue",
    id="auto",
    name="Automatic",
    recommended=False,
)

#: Model ids that ARE the router rather than a model, by aggregator spelling.
#: Mirrors ``discovery._META_ROUTE_IDS``; kept here as well because this module
#: must not import the discovery layer (it is the leaf the registry is built
#: from), and the set is two literals that change only when an aggregator adds
#: a router.
AGGREGATOR_ROUTER_MODEL_IDS = frozenset({"auto", "openrouter/auto"})

anthropic_default_model_info: ModelInfo = ModelInfo(
    max_tokens=64_000,
    context_window=200_000,
    supports_images=True,
    supports_prompt_cache=True,
    # NOT a guess at the model's real numbers: 0 is this registry's "unknown"
    # for a price, and a wrong price is worse than an absent one because the
    # session band renders it as fact. Anthropic's models listing carries no
    # prices at all, so an unshipped id keeps zeros until someone adds a row.
    input_price=0.0,
    output_price=0.0,
    cache_writes_price=0.0,
    cache_reads_price=0.0,
    description="Anthropic Claude model not described by the shipped registry",
    id="anthropic",
    name="Anthropic Claude",
    recommended=False,
)
"""LAST-RESORT floor for a Claude id nothing else can describe.

Reached only after :func:`anthropic_family_model_info` declines, i.e. for an id
whose tier and version cannot be parsed at all (``claude-opus-latest``) or whose
tier has no shipped row. A live ``/v1/models`` answer beats this outright — it
carries ``max_input_tokens`` and ``max_tokens`` per model — so this is the
offline-and-unparseable corner.

Falling through to :data:`unknown_model_info` instead hands the session
128k/8192/no-cache, which is wrong for every Claude generation ever shipped:
prompt caching and images are universal from Claude 3 on, and 200k is the
smallest window any current Claude serves. The direction matters because the two
errors are not symmetric. Under-report and the loss is silent — an under-reported
window throws away room and compacts early, an under-reported ``max_tokens``
TRUNCATES a long answer with no error at all, and ``supports_prompt_cache=False``
drops ``cache_control`` on the most expensive models in the catalogue.
Over-report and the provider answers 400 naming the real limit, which at least
says what happened. 200k is therefore deliberately the floor rather than the 1M
the 5 generation serves: a wrong 1M window puts the compaction threshold
(``min(0.8 * window, 600k)``) at 600k, so a genuinely-200k model would 400 on
every turn past 200k instead of merely compacting sooner than it had to.

``max_tokens`` is the floor among the CURRENT generations (64k) rather than the
all-time floor (Claude 3 Haiku is 4k), because this template is only ever reached
for an id the registry does not have — and every older, smaller-output model is
already in :data:`anthropic_models`, so an unknown id is by construction a newer
one.
"""

#: Alphabetic id segments that are not a model TIER. ``claude`` is the vendor
#: prefix and ``latest`` is an alias suffix (``claude-3-7-sonnet-latest``);
#: reading either as a tier would file an alias under a family of its own and
#: lose the inheritance this parser exists to provide.
_ANTHROPIC_NON_TIER_SEGMENTS = frozenset({"claude", "latest"})

#: A snapshot date is exactly 8 digits (``20251101``). It has to be told apart
#: from a version component or ``claude-opus-4-5-20251101`` parses as version
#: (4, 5, 20251101), which sorts above every real generation and would make each
#: dated snapshot its own family — the opposite of inheriting one.
_ANTHROPIC_DATE_DIGITS = 8


def _anthropic_family(model_id: str) -> Optional[tuple[str, tuple[int, ...]]]:
    """``(tier, version)`` parsed out of a Claude id, or ``None``.

    Anthropic has used two id shapes and both are still served, so the parser
    reads segments by KIND rather than by position: ``claude-opus-4-5-20251101``
    puts the tier before the version and ``claude-3-5-sonnet-20241022`` puts it
    after. Both yield ``("opus", (4, 5))`` and ``("sonnet", (3, 5))``.

    The tier is not matched against a fixed set of names. ``claude-fable-5`` is a
    live tier that no such list would have contained, and a list is wrong in the
    one direction that matters: an unrecognised tier gets no family and falls back
    to the 200k floor, which is exactly the under-reporting being fixed.

    ``None`` means "not a Claude id shaped like a family member" — no tier, or no
    version at all — and the caller must then use the flat template rather than
    guess.
    """
    tier = ""
    version: list[int] = []
    for segment in model_id.strip().casefold().split("-"):
        if segment.isdigit():
            if len(segment) != _ANTHROPIC_DATE_DIGITS:
                version.append(int(segment))
            continue
        if segment and not tier and segment not in _ANTHROPIC_NON_TIER_SEGMENTS:
            tier = segment
    if not tier or not version:
        return None
    return tier, tuple(version)


def model_family(model_id: str) -> str:
    """The vendor's model FAMILY a model-scoped quota cap would name.

    Anthropic scopes some weekly caps to a family (``7 day (Fable)``), and
    the Anthropic usage fetcher keys those caps on a slug of the family's
    display name — for Anthropic ids the slug of this function's return
    value is exactly that key (``fable``), so a quota verdict can say WHICH
    family it bound on and a credential block can be scoped to that family
    instead of taking the whole account out of rotation for a model that
    never draws on the spent window. Other providers' tier rows key on
    vendor display names that need not parse out of the model id (xAI's
    ``Grok 4``), which is why block READS match by "slug in model id" (the
    usage layer's own gating rule) rather than through this function.

    Empty means "no family parseable from the id" — the caller then has no
    family dimension to scope anything by, and account-wide treatment is the
    honest default. ``_anthropic_family`` already reads tiers by KIND rather
    than by a fixed list, so a new family (as ``fable`` once was) parses
    without touching this function.
    """
    parsed = _anthropic_family(model_id)
    return parsed[0] if parsed is not None else ""


def anthropic_family_model_info(model_id: str) -> Optional[ModelInfo]:
    """The best shipped description for an Anthropic id :data:`anthropic_models` lacks.

    Exists because the flat template is a FAMILY-BLIND answer, and Anthropic's
    families no longer share a window: Opus 5 and Sonnet 5 serve 1M while Opus 4.5
    serves 200k. A user on ``claude-opus-5-20260112`` — a dated snapshot of a model
    whose undated id is right here in the registry — was handed the 200k floor and
    a compaction threshold of 160k on a model with 1M of room, which is the report
    this function answers.

    Two ways to match, in order:

    1. **Same tier and same version** — the same model under another spelling
       (``claude-opus-5`` vs a dated snapshot of it, or an undated alias of a
       snapshot we ship). Everything transfers, prices included, because it is not
       a different model. The newest id wins when several snapshots share a
       version; ids are date-suffixed, so lexical order is chronological.
    2. **Same tier, NEWER version than anything shipped** — a generation released
       after this registry was last edited. Limits and capabilities are inherited
       from the newest shipped row of that tier because context windows have only
       ever grown within a tier, but prices are dropped to the "unknown" zero:
       a new generation is exactly where a price changes, and the status band
       renders a price as fact.

    An OLDER unshipped version returns ``None`` rather than inheriting downward.
    Monotonicity is the only reason the newer case is safe, and it does not run
    backwards: a 200k-era id must not be handed a 1M window, because that puts the
    compaction threshold beyond the model's real limit and 400s every turn.

    Returns a deep copy, so a caller that writes to the result (sessions do) cannot
    rewrite the process-wide registry.
    """
    wanted = _anthropic_family(model_id)
    if wanted is None:
        return None
    tier, version = wanted

    family: list[tuple[tuple[int, ...], str, ModelInfo]] = []
    for row_id, row in anthropic_models.items():
        parsed = _anthropic_family(row_id)
        if parsed is not None and parsed[0] == tier:
            family.append((parsed[1], row_id, row))
    if not family:
        return None

    same_version = [entry for entry in family if entry[0] == version]
    if same_version:
        row = max(same_version, key=lambda entry: entry[1])[2]
        return row.model_copy(deep=True, update={"id": model_id})

    newest_version, _, newest = max(family, key=lambda entry: (entry[0], entry[1]))
    if version < newest_version:
        return None
    return newest.model_copy(
        deep=True,
        update={
            "id": model_id,
            # NOT the older row's name: this is a different model, and labelling a
            # `claude-opus-6` session "Claude Opus 5" in the status band would be a
            # lie about which model is answering.
            "name": model_id,
            "input_price": 0.0,
            "output_price": 0.0,
            "cache_writes_price": None,
            "cache_reads_price": None,
            "description": (
                "Claude model not described by the shipped registry; limits "
                f"inherited from {newest.id}"
            ),
            "recommended": False,
        },
    )


openai_models: Dict[str, ModelInfo] = {
    # GPT-5.6 Sol. Official list from
    # https://developers.openai.com/api/docs/models/gpt-5.6-sol (read
    # 2026-08-23): $4 / $0.40 cached / $20 per million, 1,050,000 context,
    # 128k max output. Promotional rate through at least 2026-11-21; the
    # page also notes a 2x/1.5x long-context surcharge above 272k input,
    # which we do not invent a blended rate for (same convention as grok-4.6).
    "gpt-5.6-sol": ModelInfo(
        id="gpt-5.6-sol",
        name="GPT-5.6 Sol",
        input_price=4.0,
        output_price=20.0,
        cache_writes_price=5.0,  # 1.25x uncached input, per the same page
        cache_reads_price=0.40,
        max_tokens=128_000,
        context_window=1_050_000,
        supports_images=True,
        supports_prompt_cache=True,
        supports_responses_api=True,
        description="OpenAI GPT-5.6 Sol: frontier model for complex professional work.",
        recommended=True,
    ),
    "gpt-4o": ModelInfo(
        id="gpt-4o",
        name="GPT-4o",
        input_price=2.5,
        output_price=10.0,
        max_tokens=128_000,
        context_window=128_000,
        supports_images=True,
        supports_prompt_cache=False,
        description=(
            "OpenAI's latest flagship model with multimodal capabilities, optimized for "
            "speed and intelligence."
        ),
        recommended=False,
    ),
    "gpt-4o-mini": ModelInfo(
        id="gpt-4o-mini",
        name="GPT-4o mini",
        input_price=0.60,
        output_price=2.40,
        max_tokens=128_000,
        context_window=128_000,
        supports_images=True,
        supports_prompt_cache=False,
        description="Smaller, faster, and more cost-efficient version of GPT-4o.",
        recommended=False,
    ),
    "gpt-4.1": ModelInfo(
        id="gpt-4.1",
        name="GPT-4.1",
        input_price=2.0,
        output_price=8.0,
        max_tokens=1_047_576,
        context_window=1_047_576,
        supports_images=False,
        supports_prompt_cache=True,
        description="Smartest model for complex tasks.",
        recommended=True,
    ),
    "gpt-4.1-mini": ModelInfo(
        id="gpt-4.1-mini",
        name="GPT-4.1 mini",
        input_price=0.4,
        output_price=1.6,
        max_tokens=1_047_576,
        context_window=1_047_576,
        supports_images=False,
        supports_prompt_cache=True,
        description="Affordable model balancing speed and intelligence.",
        recommended=False,
    ),
    "gpt-4.1-nano": ModelInfo(
        id="gpt-4.1-nano",
        name="GPT-4.1 nano",
        input_price=0.1,
        output_price=0.4,
        max_tokens=1_047_576,
        context_window=1_047_576,
        supports_images=False,
        supports_prompt_cache=True,
        description="Fastest, most cost-effective model for low-latency tasks.",
        recommended=False,
    ),
    "o3": ModelInfo(
        id="o3",
        name="OpenAI o3",
        input_price=10.0,
        output_price=40.0,
        max_tokens=128_000,
        context_window=128_000,
        supports_images=True,
        supports_prompt_cache=False,
        description=(
            "Our most powerful reasoning model with leading performance on coding, "
            "math, science, and vision."
        ),
        recommended=False,
    ),
    "o4-mini": ModelInfo(  # Note: official page shows o4-mini, not o3-mini
        id="o4-mini",
        name="OpenAI o4 mini",
        input_price=1.1,
        output_price=4.4,
        max_tokens=128_000,
        context_window=128_000,
        supports_images=True,
        supports_prompt_cache=False,
        description=(
            "Our faster, cost-efficient reasoning model delivering strong performance "
            "on math, coding and vision."
        ),
        recommended=False,
    ),
    "gpt-4.5-preview": ModelInfo(  # Renamed from gpt-4.5 to align with "Preview" status
        id="gpt-4.5-preview",
        name="GPT-4.5 Preview",
        input_price=75.0,
        output_price=150.0,
        max_tokens=128_000,
        context_window=128_000,
        supports_images=True,
        supports_prompt_cache=False,
        description="Most advanced preview model from OpenAI, offering cutting-edge capabilities.",
        recommended=False,
    ),
    "gpt-4": ModelInfo(
        id="gpt-4",
        name="GPT-4",
        input_price=30.0,
        output_price=60.0,
        max_tokens=8192,
        context_window=8192,
        supports_images=False,
        supports_prompt_cache=False,
        description="More capable than any GPT-3.5 model, able to do more complex tasks. (Legacy)",
        recommended=False,
    ),
    "gpt-3.5-turbo": ModelInfo(
        id="gpt-3.5-turbo",
        name="GPT-3.5 Turbo",
        input_price=0.5,
        output_price=1.5,
        max_tokens=16385,
        context_window=16385,
        supports_images=False,
        supports_prompt_cache=False,
        description=(
            "Most capable GPT-3.5 model, optimized for chat at 1/10th the cost of "
            "GPT-4. (Legacy)"
        ),
        recommended=False,
    ),
    "gpt-3.5-turbo-16k": ModelInfo(
        id="gpt-3.5-turbo-16k",
        name="GPT-3.5 Turbo 16K",
        input_price=1.0,  # Was 1.0 in old, Holori shows 0.5/1.5 for generic 3.5 turbo.
        output_price=2.0,  # Was 2.0 in old.
        max_tokens=16385,
        context_window=16385,
        supports_images=False,
        supports_prompt_cache=False,
        description=(
            "Same capabilities as standard GPT-3.5 Turbo but with longer context. " "(Legacy)"
        ),
        recommended=False,
    ),
}


google_models: Dict[str, ModelInfo] = {
    "gemini-2.5-flash-preview-05-20": ModelInfo(
        id="gemini-2.5-flash-preview-05-20",
        name="Gemini 2.5 Flash Preview",
        max_tokens=65535,
        context_window=1048576,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=0.15,
        output_price=0.60,
        description=(
            "Google's latest general purpose model, which is fast and more cost effective "
            "for complex reasoning, coding, and scientific tasks"
        ),
        recommended=True,
    ),
    "gemini-2.5-pro-preview-05-06": ModelInfo(
        id="gemini-2.5-pro-preview-05-06",
        name="Gemini 2.5 Pro Preview",
        max_tokens=65535,
        context_window=1048576,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=1.25,
        output_price=10.0,
        description=(
            "Google's state-of-the-art multipurpose model, which excels at coding and "
            "complex reasoning tasks"
        ),
        recommended=True,
    ),
    "gemini-2.0-flash-001": ModelInfo(
        max_tokens=8192,
        context_window=1_048_576,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=0.1,
        output_price=0.4,
        description="Google's latest multimodal model with excellent performance",
        id="gemini-2.0-flash-001",
        name="Gemini 2.0 Flash",
        recommended=False,
    ),
    "gemini-2.0-flash-lite-preview-02-05": ModelInfo(
        id="gemini-2.0-flash-lite-preview-02-05",
        name="Gemini 2.0 Flash Lite Preview",
        max_tokens=8192,
        context_window=1_048_576,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=0,
        output_price=0,
        description="Lighter version of Gemini 2.0 Flash",
        recommended=False,
    ),
    "gemini-2.0-pro-exp-02-05": ModelInfo(
        id="gemini-2.0-pro-exp-02-05",
        name="Gemini 2.0 Pro Exp",
        max_tokens=8192,
        context_window=2_097_152,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=0,
        output_price=0,
        description="Google's most powerful Gemini model",
        recommended=False,
    ),
    "gemini-2.0-flash-thinking-exp-01-21": ModelInfo(
        id="gemini-2.0-flash-thinking-exp-01-21",
        # The release token, in the same parenthesised shape every other dated
        # row here uses. It is not decoration: this row and
        # `gemini-2.0-flash-thinking-exp-1219` below shipped the IDENTICAL name,
        # so any surface that renders the name rather than the id — the status
        # band, the model picker's label column — could not say which of the two
        # was answering, and they differ by an 8x output limit (65,536 against
        # 8,192). `model/naming.py` refuses a shared name and falls back to the
        # 42-cell selector, so the duplicate cost the band its whole left group
        # at narrow widths as well.
        name="Gemini 2.0 Flash Thinking Exp (01-21)",
        max_tokens=65_536,
        context_window=1_048_576,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=0,
        output_price=0,
        description="Experimental Gemini model with thinking capabilities",
        recommended=False,
    ),
    "gemini-2.0-flash-thinking-exp-1219": ModelInfo(
        id="gemini-2.0-flash-thinking-exp-1219",
        name="Gemini 2.0 Flash Thinking Exp (1219)",
        max_tokens=8192,
        context_window=32_767,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=0,
        output_price=0,
        description="Experimental Gemini model with thinking capabilities",
        recommended=False,
    ),
    "gemini-2.0-flash-exp": ModelInfo(
        id="gemini-2.0-flash-exp",
        name="Gemini 2.0 Flash Exp",
        max_tokens=8192,
        context_window=1_048_576,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=0,
        output_price=0,
        description="Experimental version of Gemini 2.0 Flash",
        recommended=False,
    ),
    "gemini-1.5-flash-002": ModelInfo(
        id="gemini-1.5-flash-002",
        name="Gemini 1.5 Flash 002",
        max_tokens=8192,
        context_window=1_048_576,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=0,
        output_price=0,
        description="Fast and efficient multimodal model",
        recommended=False,
    ),
    "gemini-1.5-flash-exp-0827": ModelInfo(
        id="gemini-1.5-flash-exp-0827",
        name="Gemini 1.5 Flash Exp 0827",
        max_tokens=8192,
        context_window=1_048_576,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=0,
        output_price=0,
        description="Experimental version of Gemini 1.5 Flash",
        recommended=False,
    ),
}

deepseek_models: Dict[str, ModelInfo] = {
    # V4 family (2026): 1M context, implicit prompt caching, an order of
    # magnitude cheaper than the V3 line. `-latest` floats to the newest
    # snapshot; the dated ids pin a snapshot so a rollout cannot silently
    # change behaviour under a long-running agent.
    "deepseek-v4-flash": ModelInfo(
        id="deepseek-v4-flash",
        name="DeepSeek V4 Flash",
        max_tokens=32_768,
        context_window=1_048_576,
        supports_images=False,
        supports_prompt_cache=True,
        input_price=0.14,
        output_price=0.28,
        cache_writes_price=0.14,
        cache_reads_price=0.014,
        description="Fast, cheap agentic workhorse with a 1M context window",
        recommended=True,
    ),
    "deepseek-v4-flash-0731": ModelInfo(
        id="deepseek-v4-flash-0731",
        name="DeepSeek V4 Flash (2026-07-31)",
        max_tokens=32_768,
        context_window=1_048_576,
        supports_images=False,
        supports_prompt_cache=True,
        input_price=0.09,
        output_price=0.18,
        cache_writes_price=0.09,
        cache_reads_price=0.009,
        description="Pinned July 2026 V4 Flash snapshot",
        # The ROW stays so a user who already pinned this model keeps correct
        # pricing and a 1M context window (dropping it would resolve them to the
        # 128k unknown default and silently mis-set compaction). Only the
        # recommendation is withdrawn — see RecommendedOpenRouterModelIds above
        # for the 0/5 tool-calling measurement behind it.
        recommended=False,
    ),
    "deepseek-v4-pro": ModelInfo(
        id="deepseek-v4-pro",
        name="DeepSeek V4 Pro",
        max_tokens=65_536,
        context_window=1_048_576,
        supports_images=False,
        supports_prompt_cache=True,
        input_price=0.435,
        output_price=0.87,
        cache_writes_price=0.435,
        cache_reads_price=0.0435,
        description="Stronger V4 tier for harder reasoning and long refactors",
        recommended=False,
    ),
    "deepseek-chat": ModelInfo(
        id="deepseek-chat",
        name="Deepseek Chat",
        max_tokens=8_192,
        context_window=64_000,
        supports_images=False,
        supports_prompt_cache=True,
        input_price=0.27,
        output_price=1.1,
        cache_writes_price=0.14,
        cache_reads_price=0.014,
        description="General purpose chat model",
        recommended=True,
    ),
    "deepseek-reasoner": ModelInfo(
        id="deepseek-reasoner",
        name="Deepseek Reasoner",
        max_tokens=8_000,
        context_window=64_000,
        supports_images=False,
        supports_prompt_cache=True,
        input_price=0.55,
        output_price=2.19,
        cache_writes_price=0.55,
        cache_reads_price=0.14,
        description="Specialized for complex reasoning tasks",
        recommended=False,
    ),
}

qwen_models: Dict[str, ModelInfo] = {
    # No row here carries a cache price, and their absence is a correction rather
    # than an omission. All 14 priced rows used to set
    # `cache_writes_price = input_price` and `cache_reads_price = output_price` —
    # the four price fields filled in the order (input, output, input, output),
    # which is a transcription slip, not a rate. It made a cached read cost 2-4x a
    # fresh input token, the exact inverse of what caching is; `qwen-max` billed
    # $9.60/MTok to re-read a token it charges $2.40 to read the first time.
    #
    # It was inert until `cost_for_usage` began pricing cache buckets, so nothing
    # had ever charged it. `None` is this registry's "unknown", which is the honest
    # value: Alibaba publishes no cache rate for these, and every row here is
    # `supports_prompt_cache=False`, so no cache token should ever reach the
    # arithmetic in the first place. If one does, `calculate_cost` falls back to
    # the input rate — wrong by the unknown discount rather than by 4x the wrong way.
    "qwen2.5-coder-32b-instruct": ModelInfo(
        id="qwen2.5-coder-32b-instruct",
        name="Qwen 2.5 Coder 32B Instruct",
        max_tokens=8_192,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=2.0,
        output_price=6.0,
        description="Specialized for code generation and understanding",
        recommended=False,
    ),
    "qwen2.5-coder-14b-instruct": ModelInfo(
        id="qwen2.5-coder-14b-instruct",
        name="Qwen 2.5 Coder 14B Instruct",
        max_tokens=8_192,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=2.0,
        output_price=6.0,
        description="Medium-sized code-specialized model",
        recommended=False,
    ),
    "qwen2.5-coder-7b-instruct": ModelInfo(
        id="qwen2.5-coder-7b-instruct",
        name="Qwen 2.5 Coder 7B Instruct",
        max_tokens=8_192,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.5,
        output_price=1.0,
        description="Efficient code-specialized model",
        recommended=False,
    ),
    "qwen2.5-coder-3b-instruct": ModelInfo(
        id="qwen2.5-coder-3b-instruct",
        name="Qwen 2.5 Coder 3B Instruct",
        max_tokens=8_192,
        context_window=32_768,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.5,
        output_price=1.0,
        description="Compact code-specialized model",
        recommended=False,
    ),
    "qwen2.5-coder-1.5b-instruct": ModelInfo(
        id="qwen2.5-coder-1.5b-instruct",
        name="Qwen 2.5 Coder 1.5B Instruct",
        max_tokens=8_192,
        context_window=32_768,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.0,
        output_price=0.0,
        description="Very compact code-specialized model",
        recommended=False,
    ),
    "qwen2.5-coder-0.5b-instruct": ModelInfo(
        id="qwen2.5-coder-0.5b-instruct",
        name="Qwen 2.5 Coder 0.5B Instruct",
        max_tokens=8_192,
        context_window=32_768,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.0,
        output_price=0.0,
        description="Smallest code-specialized model",
        recommended=False,
    ),
    "qwen-coder-plus-latest": ModelInfo(
        id="qwen-coder-plus-latest",
        name="Qwen Coder Plus Latest",
        max_tokens=129_024,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=3.5,
        output_price=7,
        description="Advanced code generation model",
        recommended=False,
    ),
    "qwen-plus-latest": ModelInfo(
        id="qwen-plus-latest",
        name="Qwen Plus Latest",
        max_tokens=129_024,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.8,
        output_price=2,
        description="Balanced performance Qwen model",
        recommended=True,
    ),
    "qwen-turbo-latest": ModelInfo(
        id="qwen-turbo-latest",
        name="Qwen Turbo Latest",
        max_tokens=1_000_000,
        context_window=1_000_000,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.8,
        output_price=2,
        description="Fast and efficient Qwen model",
        recommended=False,
    ),
    "qwen-max-latest": ModelInfo(
        id="qwen-max-latest",
        name="Qwen Max Latest",
        max_tokens=30_720,
        context_window=32_768,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=2.4,
        output_price=9.6,
        description="Alibaba's most powerful Qwen model",
        recommended=False,
    ),
    "qwen-coder-plus": ModelInfo(
        id="qwen-coder-plus",
        name="Qwen Coder Plus",
        max_tokens=129_024,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=3.5,
        output_price=7,
        description="Advanced code generation model",
        recommended=False,
    ),
    "qwen-plus": ModelInfo(
        id="qwen-plus",
        name="Qwen Plus",
        max_tokens=129_024,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.8,
        output_price=2,
        description="Balanced performance Qwen model",
        recommended=True,
    ),
    "qwen-turbo": ModelInfo(
        id="qwen-turbo",
        name="Qwen Turbo",
        max_tokens=1_000_000,
        context_window=1_000_000,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.3,
        output_price=0.6,
        description="Fast and efficient Qwen model",
        recommended=False,
    ),
    "qwen-max": ModelInfo(
        id="qwen-max",
        name="Qwen Max",
        max_tokens=30_720,
        context_window=32_768,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=2.4,
        output_price=9.6,
        description="Alibaba's most powerful Qwen model",
        recommended=True,
    ),
    "qwen-vl-max": ModelInfo(
        id="qwen-vl-max",
        name="Qwen VL Max",
        max_tokens=30_720,
        context_window=32_768,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=3,
        output_price=9,
        description="Multimodal Qwen model with vision capabilities",
        recommended=False,
    ),
    "qwen-vl-max-latest": ModelInfo(
        id="qwen-vl-max-latest",
        name="Qwen VL Max Latest",
        max_tokens=129_024,
        context_window=131_072,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=3,
        output_price=9,
        description="Multimodal Qwen model with vision capabilities",
        recommended=False,
    ),
    "qwen-vl-plus": ModelInfo(
        id="qwen-vl-plus",
        name="Qwen VL Plus",
        max_tokens=6_000,
        context_window=8_000,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=1.5,
        output_price=4.5,
        description="Balanced multimodal Qwen model",
        recommended=False,
    ),
    "qwen-vl-plus-latest": ModelInfo(
        id="qwen-vl-plus-latest",
        name="Qwen VL Plus Latest",
        max_tokens=129_024,
        context_window=131_072,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=1.5,
        output_price=4.5,
        description="Balanced multimodal Qwen model",
        recommended=False,
    ),
}

qwencloud_token_plan_models: Dict[str, ModelInfo] = {
    # QwenCloud Token Plan (provider id `alibaba-token-plan`): the subscription
    # gateway at token-plan.ap-southeast-1.maas.aliyuncs.com. Its `/models`
    # listing was checked directly (2026-08-18) and returns ONLY ids — no
    # context_window, no max_tokens, no pricing — so live discovery cannot
    # correct these rows and this map is the sole source of the numbers the
    # session runs on. Compaction thresholds derive from context_window, which
    # is why these entries exist at all: without them the spec fell back to the
    # 128k unknown default and a 1M-window model compacted (or 400'd) at 128k.
    #
    # OUTPUT CAPS come from the endpoint itself, which is the one first-party
    # source available here. It validates `max_tokens` and names the range in
    # the rejection, so the number can be asked for rather than transcribed:
    #
    #   $ POST /chat/completions {"model": "qwen3.8-max", "max_tokens": 9999999}
    #   {"error": {"message": "Range of max_tokens should be [1, 131072]"}}
    #
    # Run that against a SKU to re-derive its cap. Two rows (deepseek-v4-pro,
    # deepseek-v4-flash-0731) accept any value without validating, so their caps
    # are OpenRouter's figure for the same upstream model — the endpoint
    # declines to state one and nothing is gained by inventing a smaller number.
    #
    # CONTEXT WINDOWS cannot be probed the same way: a prompt large enough to
    # test the boundary is refused by a body-size limit (`RequestTooLarge`)
    # before any window check runs. They are transcribed from Alibaba's
    # published specs at 1,000,000, which OpenRouter corroborates for the qwen
    # rows. It reports 1,048,576 for glm-5.2 and deepseek-v4-pro; those keep
    # 1,000,000 because OpenRouter quotes the largest window across its whole
    # routing pool (glm-5.2 spans 96,890-1,048,576 over 31 routes) rather than
    # this gateway's, and understating a window only compacts early where
    # overstating one overflows mid-turn.
    #
    # Prices are deliberately 0.0: the Token Plan bills subscription CREDITS,
    # not per-token dollars, so any USD rate written here would be an invention.
    # 0.0 renders as "cost unavailable" rather than "free", per the registry's
    # zero-means-unknown convention for keyed providers. The same reasoning
    # leaves `cache_writes_price`/`cache_reads_price` unset: the cache discount
    # is real, but it is a CREDIT multiplier rather than a USD rate.
    #
    # PROMPT CACHING is supported, and was MEASURED rather than inferred. Each
    # text row below was driven twice with an identical 6.5k-token system
    # prefix carrying `cache_control: {"type": "ephemeral"}`, over the
    # OpenAI-compatible wire this provider actually uses. The second call
    # reported the prefix as a hit every time:
    #
    #   req 1  prompt_tokens_details.cache_creation_input_tokens = 6547
    #   req 2  prompt_tokens_details.cached_tokens               = 6547
    #
    # These rows previously said `supports_prompt_cache=False`, and that flag
    # is exactly what gates `_message_cache_markers` in
    # `providers/clients.py`. With it off lop sent no markers at all, so every
    # turn re-billed the entire prefix as fresh input. That is the expensive
    # direction for agent traffic, where a growing transcript is resent on
    # each turn: a week of real subagent sessions measured 1.5B input tokens
    # at a 97% cache-read share, where the flag alone is a ~6x difference in
    # billed input.
    #
    # Alibaba documents two modes, mutually exclusive per request
    # (https://www.alibabacloud.com/help/en/model-studio/context-cache):
    # IMPLICIT applies automatically to every supported model and cannot be
    # disabled, billing hits at 20% of the input price; EXPLICIT is what a
    # `cache_control` marker selects, billing hits at 10% with a 5-minute TTL
    # renewed on each hit. So the flag is not the difference between caching
    # and no caching — the server caches either way — it is the difference
    # between the 20% path and the 10% one.
    "qwen3.8-max": ModelInfo(
        id="qwen3.8-max",
        name="Qwen3.8 Max",
        max_tokens=131_072,
        context_window=1_000_000,
        supports_images=True,
        supports_prompt_cache=True,
        description="Alibaba's flagship Qwen3.8 model via the Token Plan subscription",
        # The only row marked recommended: it is the plan's flagship, the model
        # the subscription is bought for, and the one both available sources
        # describe identically.
        recommended=True,
    ),
    "qwen3.7-max": ModelInfo(
        id="qwen3.7-max",
        name="Qwen3.7 Max",
        max_tokens=131_072,
        context_window=1_000_000,
        supports_images=False,
        supports_prompt_cache=True,
        description="Previous-generation flagship Qwen model on the Token Plan",
        recommended=False,
    ),
    "qwen3.7-plus": ModelInfo(
        id="qwen3.7-plus",
        name="Qwen3.7 Plus",
        max_tokens=131_072,
        context_window=1_000_000,
        supports_images=True,
        supports_prompt_cache=True,
        description="Balanced Qwen model on the Token Plan",
        recommended=False,
    ),
    "qwen3.6-flash": ModelInfo(
        id="qwen3.6-flash",
        name="Qwen3.6 Flash",
        max_tokens=65_536,
        context_window=1_000_000,
        supports_images=True,
        supports_prompt_cache=True,
        description="Fast, lightweight Qwen model on the Token Plan",
        recommended=False,
    ),
    "qwen3.8-flash": ModelInfo(
        id="qwen3.8-flash",
        name="Qwen3.8 Flash",
        # On the plan's published allowlist
        # (https://docs.qwencloud.com/token-plan/personal/token-plan-personal-overview)
        # and missing here, so it resolved to `unknown_model_info` — a -1
        # context window, which the spec reads as the 128k default. That is the
        # exact defect the surrounding rows exist to prevent, and it is worse
        # for this SKU than for most: it is the current-generation Flash, so a
        # 1M-window model compacted at 128k.
        #
        # Both numbers below are first-party. The cap is what the endpoint
        # itself reports for an over-large request ("Range of max_tokens should
        # be [1, 131072]") — twice qwen3.6-flash's 65,536, which is the
        # substantive reason to prefer this row over that one. Vision was
        # confirmed against a real 64x64 PNG rather than assumed from the
        # allowlist's "visual understanding" column: a 1x1 image is rejected on
        # dimensions by every row here, which is itself evidence the endpoint
        # decodes the image instead of ignoring the block.
        max_tokens=131_072,
        context_window=1_000_000,
        supports_images=True,
        supports_prompt_cache=True,
        description="Current-generation fast Qwen model on the Token Plan",
        recommended=False,
    ),
    "glm-5.2": ModelInfo(
        id="glm-5.2",
        # Suffixed because `zai/glm-5.2` already answers to the bare name and
        # shipped names must be globally unique (see
        # test_no_two_shipped_rows_share_a_curated_name) — without the suffix
        # BOTH rows lose their name as ambiguous and neither band can say which
        # endpoint is serving. This fixes the FULL label only: `_drop_qualifier`
        # strips the suffix for the compact form, `_names_one` then rejects the
        # stripped name against the ambiguous `glm-5.2` bucket, and the compact
        # rung falls back to the bare id. Under width pressure the two rows
        # still read `GLM-5.2` and `glm-5.2`, which is a naming-module limit,
        # not something a registry name can repair.
        name="GLM-5.2 (Token Plan)",
        max_tokens=131_072,
        context_window=1_000_000,
        supports_images=False,
        supports_prompt_cache=True,
        description="Zhipu's GLM-5.2 served through the Token Plan",
        recommended=False,
    ),
    "deepseek-v4-flash-0731": ModelInfo(
        id="deepseek-v4-flash-0731",
        # In the gateway's listing and a working reasoning chat model — verified
        # with a live completion that came back stamped with this id and
        # `reasoning_tokens` in its usage. It was missed on the first pass
        # because the id sits among the image/audio entries, and without a row
        # it resolved to the 128k default: the exact defect this map fixes, one
        # line away from a sibling that got it right.
        #
        # Deliberately NOT suffixed "(Token Plan)" like the two rows below, even
        # though `deepseek/deepseek-v4-flash-0731` also ships. The suffix exists
        # only to break a collision between two IDENTICAL curated names, and
        # there is none here — the direct row is "DeepSeek V4 Flash (2026-07-31)"
        # — so adding one would be ceremony rather than a fix. It would not be
        # harmful either: verified against `naming.py`, a suffixed spelling
        # strips to "DeepSeek V4 Flash 0731" where the direct row strips to
        # "DeepSeek V4 Flash", so the two stay distinct in both label forms.
        name="DeepSeek V4 Flash 0731",
        max_tokens=393_216,
        context_window=1_000_000,
        supports_images=False,
        supports_prompt_cache=True,
        description="DeepSeek V4 Flash (0731) served through the Token Plan",
        recommended=False,
    ),
    "deepseek-v4-pro": ModelInfo(
        id="deepseek-v4-pro",
        # Suffixed for the same reason as `glm-5.2` above, with the same
        # full-label-only caveat: `deepseek/deepseek-v4-pro` already answers to
        # the bare name, and shipped names must be globally unique (see
        # test_no_two_shipped_rows_share_a_curated_name).
        name="DeepSeek V4 Pro (Token Plan)",
        max_tokens=384_000,
        context_window=1_000_000,
        supports_images=False,
        supports_prompt_cache=True,
        description="DeepSeek V4 Pro served through the Token Plan",
        recommended=False,
    ),
}

mistral_models: Dict[str, ModelInfo] = {
    "mistral-large-latest": ModelInfo(
        id="mistral-large-latest",
        name="Mistral Large Latest",
        max_tokens=131_000,
        context_window=131_000,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=2.0,
        output_price=6.0,
        description="Mistral's most powerful model.  Latest version.",
        recommended=False,
    ),
    "mistral-large-2411": ModelInfo(
        id="mistral-large-2411",
        name="Mistral Large 2411",
        max_tokens=131_000,
        context_window=131_000,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=2.0,
        output_price=6.0,
        description="Mistral's most powerful model.  Snapshot from November 2024.",
        recommended=False,
    ),
    "pixtral-large-2411": ModelInfo(
        id="pixtral-large-2411",
        name="Pixtral Large 2411",
        max_tokens=131_000,
        context_window=131_000,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=2.0,
        output_price=6.0,
        description="Mistral's multimodal model with image capabilities",
        recommended=False,
    ),
    "ministral-3b-2410": ModelInfo(
        id="ministral-3b-2410",
        name="Ministral 3B 2410",
        max_tokens=131_000,
        context_window=131_000,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.04,
        output_price=0.04,
        description="Compact 3B parameter model for efficient inference",
        recommended=False,
    ),
    "ministral-8b-2410": ModelInfo(
        id="ministral-8b-2410",
        name="Ministral 8B 2410",
        max_tokens=131_000,
        context_window=131_000,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.1,
        output_price=0.1,
        description="Medium-sized 8B parameter model balancing performance and efficiency",
        recommended=False,
    ),
    "mistral-small-2501": ModelInfo(
        id="mistral-small-2501",
        name="Mistral Small 2501",
        max_tokens=32_000,
        context_window=32_000,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.1,
        output_price=0.3,
        description="Fast and efficient model for simpler tasks",
        recommended=False,
    ),
    "pixtral-12b-2409": ModelInfo(
        id="pixtral-12b-2409",
        name="Pixtral 12B 2409",
        max_tokens=131_000,
        context_window=131_000,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=0.15,
        output_price=0.15,
        description="12B parameter multimodal model with vision capabilities",
        recommended=False,
    ),
    "open-mistral-nemo-2407": ModelInfo(
        id="open-mistral-nemo-2407",
        name="Open Mistral Nemo 2407",
        max_tokens=131_000,
        context_window=131_000,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.15,
        output_price=0.15,
        description="Open-source version of Mistral optimized with NVIDIA NeMo",
        recommended=False,
    ),
    "open-codestral-mamba": ModelInfo(
        id="open-codestral-mamba",
        name="Open Codestral Mamba",
        max_tokens=256_000,
        context_window=256_000,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.15,
        output_price=0.15,
        description="Open-source code-specialized model using Mamba architecture",
        recommended=False,
    ),
    "codestral-2501": ModelInfo(
        id="codestral-2501",
        name="Codestral 2501",
        max_tokens=256_000,
        context_window=256_000,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.3,
        output_price=0.9,
        description="Specialized for code generation and understanding",
        recommended=False,
    ),
}


YUAN_TO_USD = 0.14

kimi_models: Dict[str, ModelInfo] = {
    "moonshot-v1-8k": ModelInfo(
        id="moonshot-v1-8k",
        name="Moonshot V1 8K",
        max_tokens=8192,
        context_window=8192,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=12.00 * YUAN_TO_USD,
        output_price=12.00 * YUAN_TO_USD,
        cache_writes_price=24.00 * YUAN_TO_USD,
        cache_reads_price=0.02 * YUAN_TO_USD,
        description="General purpose language model with 8K context",
        recommended=False,
    ),
    "moonshot-v1-32k": ModelInfo(
        id="moonshot-v1-32k",
        name="Moonshot V1 32K",
        max_tokens=8192,
        context_window=32_768,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=24.00 * YUAN_TO_USD,
        output_price=24.00 * YUAN_TO_USD,
        cache_writes_price=24.00 * YUAN_TO_USD,
        cache_reads_price=0.02 * YUAN_TO_USD,
        description="General purpose language model with 32K context",
        recommended=False,
    ),
    "moonshot-v1-128k": ModelInfo(
        id="moonshot-v1-128k",
        name="Moonshot V1 128K",
        max_tokens=8192,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=60.00 * YUAN_TO_USD,
        output_price=60.00 * YUAN_TO_USD,
        cache_writes_price=24.00 * YUAN_TO_USD,
        cache_reads_price=0.02 * YUAN_TO_USD,
        description="General purpose language model with 128K context",
        recommended=False,
    ),
    "moonshot-v1-8k-vision-preview": ModelInfo(
        id="moonshot-v1-8k-vision-preview",
        name="Moonshot V1 8K Vision Preview",
        max_tokens=8192,
        context_window=8192,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=12.00 * YUAN_TO_USD,
        output_price=12.00 * YUAN_TO_USD,
        cache_writes_price=24.00 * YUAN_TO_USD,
        cache_reads_price=0.02 * YUAN_TO_USD,
        description="Multimodal model with 8K context",
        recommended=False,
    ),
    "moonshot-v1-32k-vision-preview": ModelInfo(
        id="moonshot-v1-32k-vision-preview",
        name="Moonshot V1 32K Vision Preview",
        max_tokens=8192,
        context_window=32_768,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=24.00 * YUAN_TO_USD,
        output_price=24.00 * YUAN_TO_USD,
        cache_writes_price=24.00 * YUAN_TO_USD,
        cache_reads_price=0.02 * YUAN_TO_USD,
        description="Multimodal model with 32K context",
        recommended=False,
    ),
    "moonshot-v1-128k-vision-preview": ModelInfo(
        id="moonshot-v1-128k-vision-preview",
        name="Moonshot V1 128K Vision Preview",
        max_tokens=8192,
        context_window=131_072,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=60.00 * YUAN_TO_USD,
        output_price=60.00 * YUAN_TO_USD,
        cache_writes_price=24.00 * YUAN_TO_USD,
        cache_reads_price=0.02 * YUAN_TO_USD,
        description="Multimodal model with 128K context",
        recommended=False,
    ),
    # Kimi K3. Official list price from
    # https://platform.kimi.ai/docs/pricing/chat-k3 (read 2026-08-23):
    # $3.00 cache-miss / $0.30 cache-hit / $15.00 output per million tokens,
    # 1,048,576 context, flat (no long-context surcharge). The coding-plan
    # host serves this as the bare id ``k3`` (and ``k3-256k``); the mainland
    # listing uses ``kimi-k3``. Both spellings must resolve or a failover
    # onto ``kimi/k3`` prices as unknown.
    "k3": ModelInfo(
        id="k3",
        name="Kimi K3",
        max_tokens=131_072,
        context_window=1_048_576,
        supports_images=True,
        supports_prompt_cache=True,
        input_price=3.00,
        output_price=15.00,
        cache_writes_price=3.00,
        cache_reads_price=0.30,
        description="Moonshot Kimi K3 flagship: 1M context, flat per-token pricing.",
        recommended=True,
    ),
}

# X.AI Grok models and pricing
xai_models: Dict[str, ModelInfo] = {
    # Current-generation Grok 4 family. Prices and windows transcribed from
    # https://docs.x.ai/developers/models (read 2026-08-23). The page quotes
    # two tiers per id: the <200k-prompt rate and a 2x long-context surcharge
    # once the prompt reaches 200k. We carry the <200k rate because that is
    # the list price of a typical agent turn; a 200k+ prompt is still billed
    # (and still appears in By provider) but the estimate understates it
    # rather than inventing a blended rate the table cannot express.
    "grok-4.6": ModelInfo(
        id="grok-4.6",
        name="Grok 4.6",
        max_tokens=131_072,
        context_window=500_000,
        supports_images=True,
        supports_prompt_cache=True,
        input_price=2.00,  # $2 / MTok (<200k prompt); $4 ≥200k
        output_price=6.00,  # $6 / MTok (<200k prompt); $12 ≥200k
        cache_writes_price=2.00,
        cache_reads_price=0.50,  # $0.50 / MTok (<200k); $1.00 ≥200k
        description="xAI Grok 4.6: frontier coding/agent model, 500k context.",
        recommended=True,
    ),
    "grok-4.5": ModelInfo(
        id="grok-4.5",
        name="Grok 4.5",
        max_tokens=131_072,
        context_window=500_000,
        supports_images=True,
        supports_prompt_cache=True,
        input_price=2.00,
        output_price=6.00,
        cache_writes_price=2.00,
        cache_reads_price=0.30,
        description="xAI Grok 4.5: previous-generation frontier, 500k context.",
        recommended=False,
    ),
    "grok-4.3": ModelInfo(
        id="grok-4.3",
        name="Grok 4.3",
        max_tokens=131_072,
        context_window=1_000_000,
        supports_images=True,
        supports_prompt_cache=True,
        input_price=1.25,
        output_price=2.50,
        cache_writes_price=1.25,
        cache_reads_price=0.20,
        description="xAI Grok 4.3: 1M-context Grok 4 family model.",
        recommended=False,
    ),
    # grok-3-beta, grok-3, grok-3-latest
    "grok-3-beta": ModelInfo(
        id="grok-3-beta",
        name="Grok-3 Beta",
        max_tokens=131_072,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=3.00,
        output_price=15.00,
        description="X.AI Grok-3 Beta: Text input and completion, large context window.",
        recommended=True,
    ),
    "grok-3": ModelInfo(
        id="grok-3",
        name="Grok-3",
        max_tokens=131_072,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=3.00,
        output_price=15.00,
        description="X.AI Grok-3: Text input and completion, large context window.",
        recommended=True,
    ),
    "grok-3-latest": ModelInfo(
        id="grok-3-latest",
        name="Grok-3 Latest",
        max_tokens=131_072,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=3.00,
        output_price=15.00,
        description="X.AI Grok-3 Latest: Text input and completion, large context window.",
        recommended=True,
    ),
    # grok-3-fast-beta, grok-3-fast, grok-3-fast-latest
    "grok-3-fast-beta": ModelInfo(
        id="grok-3-fast-beta",
        name="Grok-3 Fast Beta",
        max_tokens=131_072,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=5.00,
        output_price=25.00,
        description="X.AI Grok-3 Fast Beta: Text input and completion, large context window.",
        recommended=False,
    ),
    "grok-3-fast": ModelInfo(
        id="grok-3-fast",
        name="Grok-3 Fast",
        max_tokens=131_072,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=5.00,
        output_price=25.00,
        description="X.AI Grok-3 Fast: Text input and completion, large context window.",
        recommended=False,
    ),
    "grok-3-fast-latest": ModelInfo(
        id="grok-3-fast-latest",
        name="Grok-3 Fast Latest",
        max_tokens=131_072,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=5.00,
        output_price=25.00,
        description="X.AI Grok-3 Fast Latest: Text input and completion, large context window.",
        recommended=False,
    ),
    # grok-3-mini-beta, grok-3-mini, grok-3-mini-latest
    "grok-3-mini-beta": ModelInfo(
        id="grok-3-mini-beta",
        name="Grok-3 Mini Beta",
        max_tokens=131_072,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.30,
        output_price=0.50,
        description="X.AI Grok-3 Mini Beta: Text input and completion, large context window.",
        recommended=False,
    ),
    "grok-3-mini": ModelInfo(
        id="grok-3-mini",
        name="Grok-3 Mini",
        max_tokens=131_072,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.30,
        output_price=0.50,
        description="X.AI Grok-3 Mini: Text input and completion, large context window.",
        recommended=False,
    ),
    "grok-3-mini-latest": ModelInfo(
        id="grok-3-mini-latest",
        name="Grok-3 Mini Latest",
        max_tokens=131_072,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.30,
        output_price=0.50,
        description="X.AI Grok-3 Mini Latest: Text input and completion, large context window.",
        recommended=False,
    ),
    # grok-3-mini-fast-beta, grok-3-mini-fast, grok-3-mini-fast-latest
    "grok-3-mini-fast-beta": ModelInfo(
        id="grok-3-mini-fast-beta",
        name="Grok-3 Mini Fast Beta",
        max_tokens=131_072,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.60,
        output_price=4.00,
        description="X.AI Grok-3 Mini Fast Beta: Text input and completion, large context window.",
        recommended=False,
    ),
    "grok-3-mini-fast": ModelInfo(
        id="grok-3-mini-fast",
        name="Grok-3 Mini Fast",
        max_tokens=131_072,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.60,
        output_price=4.00,
        description="X.AI Grok-3 Mini Fast: Text input and completion, large context window.",
        recommended=False,
    ),
    "grok-3-mini-fast-latest": ModelInfo(
        id="grok-3-mini-fast-latest",
        name="Grok-3 Mini Fast Latest",
        max_tokens=131_072,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.60,
        output_price=4.00,
        description=(
            "X.AI Grok-3 Mini Fast Latest: Text input and completion, large context window."
        ),
        recommended=False,
    ),
    # grok-2-vision-1212, grok-2-vision, grok-2-vision-latest
    "grok-2-vision-1212": ModelInfo(
        id="grok-2-vision-1212",
        name="Grok-2 Vision 1212",
        max_tokens=8192,
        context_window=8192,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=2.00,
        output_price=10.00,
        description="X.AI Grok-2 Vision 1212: Text and image input, text completion.",
        recommended=False,
    ),
    "grok-2-vision": ModelInfo(
        id="grok-2-vision",
        name="Grok-2 Vision",
        max_tokens=8192,
        context_window=8192,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=2.00,
        output_price=10.00,
        description="X.AI Grok-2 Vision: Text and image input, text completion.",
        recommended=False,
    ),
    "grok-2-vision-latest": ModelInfo(
        id="grok-2-vision-latest",
        name="Grok-2 Vision Latest",
        max_tokens=8192,
        context_window=8192,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=2.00,
        output_price=10.00,
        description="X.AI Grok-2 Vision Latest: Text and image input, text completion.",
        recommended=False,
    ),
    # grok-2-image-1212, grok-2-image, grok-2-image-latest
    "grok-2-image-1212": ModelInfo(
        id="grok-2-image-1212",
        name="Grok-2 Image 1212",
        max_tokens=131_072,
        context_window=131_072,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=0.07,
        output_price=0.0,
        description="X.AI Grok-2 Image 1212: Each generated image.",
        recommended=False,
    ),
    "grok-2-image": ModelInfo(
        id="grok-2-image",
        name="Grok-2 Image",
        max_tokens=131_072,
        context_window=131_072,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=0.07,
        output_price=0.0,
        description="X.AI Grok-2 Image: Each generated image.",
        recommended=False,
    ),
    "grok-2-image-latest": ModelInfo(
        id="grok-2-image-latest",
        name="Grok-2 Image Latest",
        max_tokens=131_072,
        context_window=131_072,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=0.07,
        output_price=0.0,
        description="X.AI Grok-2 Image Latest: Each generated image.",
        recommended=False,
    ),
}

#: Z.AI's GLM family.
#:
#: Prices are USD per MILLION tokens and are Z.AI's own DIRECT list rates,
#: transcribed from the bundled catalogue omp ships for this provider.
#:
#: They deliberately do NOT match OpenRouter's `z-ai/*` listings, which quote
#: less for most GLM ids (e.g. `glm-5.2` at $0.462/$1.452 against Z.AI's
#: $1.40/$4.40). That is a real price difference, not a transcription error:
#: OpenRouter resells through its own discounted arrangements, and a user
#: billed by Z.AI directly is charged the direct rate. Quoting the aggregator's
#: number on the direct route would under-report every session's cost, so the
#: direct rate is the one carried here and the models.dev price-catalogue leg
#: (`configure._from_price_catalogue`) only ever fills a model this table does
#: not price at all.
#:
#: They are carried STATICALLY because Z.AI's `/models` endpoint
#: returns bare `{id, object, created, owned_by}` rows with no pricing, context
#: window, or capability data at all — discovery alone would report every GLM as
#: free and unpriced, so a live listing is merged OVER these rows rather than
#: replacing them.
#:
#: `cache_writes_price` equals `input_price` because Z.AI does not charge a
#: separate cache-write premium; cache READS are discounted (~19% of input),
#: which is the number that actually moves a long agent session's bill.
glm_models: Dict[str, ModelInfo] = {
    "glm-5.3": ModelInfo(
        id="glm-5.3",
        name="GLM-5.3",
        max_tokens=131_072,
        context_window=1_000_000,
        supports_images=False,
        supports_prompt_cache=True,
        input_price=1.4,
        output_price=4.4,
        cache_writes_price=1.4,
        cache_reads_price=0.26,
        description="Z.AI's flagship GLM-5.3 coding model with a 1M context window",
        recommended=True,
    ),
    "glm-5.2": ModelInfo(
        id="glm-5.2",
        name="GLM-5.2",
        max_tokens=131_072,
        context_window=1_000_000,
        supports_images=False,
        supports_prompt_cache=True,
        input_price=1.4,
        output_price=4.4,
        cache_writes_price=1.4,
        cache_reads_price=0.26,
        description="GLM-5.2 reasoning and coding model with a 1M context window",
        recommended=True,
    ),
    "glm-5.1": ModelInfo(
        id="glm-5.1",
        name="GLM-5.1",
        max_tokens=131_072,
        context_window=200_000,
        supports_images=False,
        supports_prompt_cache=True,
        input_price=1.4,
        output_price=4.4,
        cache_writes_price=1.4,
        cache_reads_price=0.26,
        description="GLM-5.1 reasoning and coding model",
        recommended=False,
    ),
    "glm-5-turbo": ModelInfo(
        id="glm-5-turbo",
        name="GLM-5 Turbo",
        max_tokens=131_072,
        context_window=200_000,
        supports_images=False,
        supports_prompt_cache=True,
        input_price=1.2,
        output_price=4.0,
        cache_writes_price=1.2,
        cache_reads_price=0.24,
        description="Latency-optimised GLM-5 variant",
        recommended=False,
    ),
    "glm-5": ModelInfo(
        id="glm-5",
        name="GLM-5",
        max_tokens=131_072,
        context_window=204_800,
        supports_images=False,
        supports_prompt_cache=True,
        input_price=1.0,
        output_price=3.2,
        cache_writes_price=1.0,
        cache_reads_price=0.2,
        description="GLM-5 general reasoning and coding model",
        recommended=False,
    ),
    "glm-4.7": ModelInfo(
        id="glm-4.7",
        name="GLM-4.7",
        max_tokens=131_072,
        context_window=204_800,
        supports_images=False,
        supports_prompt_cache=True,
        input_price=0.6,
        output_price=2.2,
        cache_writes_price=0.6,
        cache_reads_price=0.11,
        description="GLM-4.7 coding model",
        recommended=False,
    ),
    "glm-4.6": ModelInfo(
        id="glm-4.6",
        name="GLM-4.6",
        max_tokens=131_072,
        context_window=204_800,
        supports_images=False,
        supports_prompt_cache=True,
        input_price=0.6,
        output_price=2.2,
        cache_writes_price=0.6,
        cache_reads_price=0.11,
        description="GLM-4.6 coding model",
        recommended=False,
    ),
    "glm-4.5": ModelInfo(
        id="glm-4.5",
        name="GLM-4.5",
        max_tokens=98_304,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=True,
        input_price=0.6,
        output_price=2.2,
        cache_writes_price=0.6,
        cache_reads_price=0.11,
        description="GLM-4.5 general purpose model",
        recommended=False,
    ),
    "glm-4.5-air": ModelInfo(
        id="glm-4.5-air",
        name="GLM-4.5 Air",
        max_tokens=98_304,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=True,
        input_price=0.2,
        output_price=1.1,
        cache_writes_price=0.2,
        cache_reads_price=0.03,
        description="Lightweight, cheaper GLM-4.5 variant",
        recommended=False,
    ),
}

#: Every hosting whose model rows this module SHIPS, keyed by hosting id.
#:
#: Hoisted out of :func:`static_models` because two questions need it and only
#: one of them names a hosting: a display-name index has to enumerate every
#: curated name at once to find out which ones are shared by two models (see
#: ``model/naming.py``), and rebuilding that from a hard-coded second list of
#: hostings would rot the moment a provider is added here and not there.
_STATIC_MODEL_MAPS: Dict[str, Dict[str, "ModelInfo"]] = {
    "anthropic": anthropic_models,
    "openai": openai_models,
    "google": google_models,
    "deepseek": deepseek_models,
    "alibaba": qwen_models,
    # Token Plan models are keyed under the canonical storage id only, so a walk
    # of this map cannot count them twice. DISCOVERY reaches them under the
    # `alibaba-token-plan-oauth` spelling through `store_credentials_as`
    # (`discovery._static_rows`), the same way `xai-oauth` reaches `xai` — but
    # `naming._unambiguous_name` calls `static_models(provider)` WITHOUT that
    # resolution, so the oauth spelling renders as its bare selector rather than
    # the curated name. That is the pre-existing `xai-oauth` behaviour, not
    # something these rows introduce; the window itself is unaffected because
    # `get_model_info` answers for both spellings directly.
    "alibaba-token-plan": qwencloud_token_plan_models,
    "mistral": mistral_models,
    "kimi": kimi_models,
    "xai": xai_models,
    "zai": glm_models,
}

#: The hostings :func:`static_models` can answer for. Ordered as declared above
#: so anything built by walking it is reproducible run to run.
STATIC_MODEL_HOSTINGS: tuple[str, ...] = tuple(_STATIC_MODEL_MAPS)
