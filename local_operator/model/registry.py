from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


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
    recommended: bool = Field(
        False,
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
    ProviderDetail(
        id="ollama",
        name="Ollama",
        description="Run open-source large language models locally",
        url="https://ollama.ai/",
        requiredCredentials=[],
        recommended=False,
    ),
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
    "deepseek/deepseek-chat-v3-0324",
    "deepseek/deepseek-r1-0528",
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
        cache_writes_price (Optional[float]): Cost per million tokens for cache writes.
        cache_reads_price (Optional[float]): Cost per million tokens for cache reads.
        description (Optional[str]): Description of the model.
        recommended (Optional[bool]): Whether the model is recommended for use in Local
        Operator.  This is determined based on community usage and feedback.
    """

    input_price: float = 0.0
    output_price: float = 0.0
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None
    supports_images: Optional[bool] = None
    supports_prompt_cache: bool = False
    cache_writes_price: Optional[float] = None
    cache_reads_price: Optional[float] = None
    description: str = Field(..., description="Description of the model")
    id: str = Field(..., description="Unique identifier for the model")
    name: str = Field(..., description="Display name for the model")
    recommended: bool = Field(
        False,
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
        return radient_default_model_info
    elif hosting == "anthropic":
        if model in anthropic_models:
            model_info = anthropic_models[model]
    elif hosting == "ollama":
        return ollama_default_model_info
    elif hosting == "deepseek":
        if model in deepseek_models:
            return deepseek_models[model]
    elif hosting == "google":
        if model in google_models:
            return google_models[model]
    elif hosting == "openai":
        return openai_models[model]
    elif hosting == "openrouter":
        return openrouter_default_model_info
    elif hosting == "alibaba":
        if model in qwen_models:
            return qwen_models[model]
    elif hosting == "kimi":
        if model in kimi_models:
            return kimi_models[model]
    elif hosting == "mistral":
        if model in mistral_models:
            return mistral_models[model]
    elif hosting == "xai":
        if model in xai_models:
            return xai_models[model]
    else:
        raise ValueError(f"Unsupported hosting provider: {hosting}")

    return model_info


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
        cache_reads_price=3.0,
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
        cache_reads_price=3.0,
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
        cache_reads_price=3.0,
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
        cache_reads_price=0.8,
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
        cache_reads_price=0.3,
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

openai_models: Dict[str, ModelInfo] = {
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
        name="Gemini 2.0 Flash Thinking Exp",
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
        name="Gemini 2.0 Flash Thinking Exp",
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
    "qwen2.5-coder-32b-instruct": ModelInfo(
        id="qwen2.5-coder-32b-instruct",
        name="Qwen 2.5 Coder 32B Instruct",
        max_tokens=8_192,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=2.0,
        output_price=6.0,
        cache_writes_price=2.0,
        cache_reads_price=6.0,
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
        cache_writes_price=2.0,
        cache_reads_price=6.0,
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
        cache_writes_price=0.5,
        cache_reads_price=1.0,
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
        cache_writes_price=0.5,
        cache_reads_price=1.0,
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
        cache_writes_price=0.0,
        cache_reads_price=0.0,
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
        cache_writes_price=0.0,
        cache_reads_price=0.0,
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
        cache_writes_price=3.5,
        cache_reads_price=7,
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
        cache_writes_price=0.8,
        cache_reads_price=0.2,
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
        cache_writes_price=0.8,
        cache_reads_price=2,
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
        cache_writes_price=2.4,
        cache_reads_price=9.6,
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
        cache_writes_price=3.5,
        cache_reads_price=7,
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
        cache_writes_price=0.8,
        cache_reads_price=0.2,
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
        cache_writes_price=0.3,
        cache_reads_price=0.6,
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
        cache_writes_price=2.4,
        cache_reads_price=9.6,
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
        cache_writes_price=3,
        cache_reads_price=9,
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
        cache_writes_price=3,
        cache_reads_price=9,
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
        cache_writes_price=1.5,
        cache_reads_price=4.5,
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
        cache_writes_price=1.5,
        cache_reads_price=4.5,
        description="Balanced multimodal Qwen model",
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
}

# X.AI Grok models and pricing
xai_models: Dict[str, ModelInfo] = {
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
