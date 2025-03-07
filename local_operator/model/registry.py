from typing import Dict, Optional

from pydantic import BaseModel, field_validator

SupportedHostingProviders = [
    "anthropic",
    "ollama",
    "deepseek",
    "google",
    "openai",
    "openrouter",
    "alibaba",
    "kimi",
    "mistral",
]
"""List of supported model hosting providers.

This list contains the names of all supported AI model hosting providers that can be used
with the Local Operator API. Each provider has its own set of available models and pricing.

The supported providers are:
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
    """

    input_price: float = 0.0
    output_price: float = 0.0
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None
    supports_images: Optional[bool] = None
    supports_prompt_cache: bool = False
    cache_writes_price: Optional[float] = None
    cache_reads_price: Optional[float] = None
    description: Optional[str] = None

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

    if hosting == "anthropic":
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
        return openai_model_info_sane_defaults
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
    else:
        raise ValueError(f"Unsupported hosting provider: {hosting}")

    return model_info


unknown_model_info: ModelInfo = ModelInfo(
    max_tokens=-1,
    context_window=-1,
    supports_images=False,
    supports_prompt_cache=False,
    input_price=0.0,
    output_price=0.0,
    description="Unknown model with default settings",
)
"""
Default ModelInfo when model is unknown.

This ModelInfo is returned by `get_model_info` when a specific model
is not found within a supported hosting provider's catalog. It provides
a fallback with negative max_tokens and context_window to indicate
the absence of specific model details.
"""

anthropic_models: Dict[str, ModelInfo] = {
    "claude-3-5-sonnet-20241022": ModelInfo(
        max_tokens=8192,
        context_window=200_000,
        supports_images=True,
        supports_prompt_cache=True,
        input_price=3.0,
        output_price=15.0,
        cache_writes_price=3.75,
        cache_reads_price=0.3,
        description="Anthropic's latest balanced model with excellent performance",
    ),
    "claude-3-5-haiku-20241022": ModelInfo(
        max_tokens=8192,
        context_window=200_000,
        supports_images=False,
        supports_prompt_cache=True,
        input_price=0.8,
        output_price=4.0,
        cache_writes_price=1.0,
        cache_reads_price=0.08,
        description="Fast and efficient model for simpler tasks",
    ),
    "claude-3-opus-20240229": ModelInfo(
        max_tokens=4096,
        context_window=200_000,
        supports_images=True,
        supports_prompt_cache=True,
        input_price=15.0,
        output_price=75.0,
        cache_writes_price=18.75,
        cache_reads_price=1.5,
        description="Anthropic's most powerful model for complex tasks",
    ),
    "claude-3-haiku-20240307": ModelInfo(
        max_tokens=4096,
        context_window=200_000,
        supports_images=True,
        supports_prompt_cache=True,
        input_price=0.25,
        output_price=1.25,
        cache_writes_price=0.3,
        cache_reads_price=0.03,
        description="Fast and efficient model for simpler tasks",
    ),
}

anthropic_models["claude-3-5-sonnet-latest"] = anthropic_models["claude-3-5-sonnet-20241022"]
anthropic_models["claude-3-opus-latest"] = anthropic_models["claude-3-opus-20240229"]
anthropic_models["claude-3-haiku-latest"] = anthropic_models["claude-3-haiku-20240307"]

# TODO: Add fetch for token, context window, image support
ollama_default_model_info: ModelInfo = ModelInfo(
    max_tokens=-1,
    context_window=-1,
    supports_images=False,
    supports_prompt_cache=False,
    input_price=0.0,
    output_price=0.0,
    description="Local model hosting with Ollama",
)

# TODO: Add fetch for token, context window, image support
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
)

openai_model_info_sane_defaults: ModelInfo = ModelInfo(
    max_tokens=-1,
    context_window=128_000,
    supports_images=True,
    supports_prompt_cache=False,
    input_price=0,
    output_price=0,
    description="OpenAI's API provides access to GPT-4o, o3-mini, and other models",
)

google_models: Dict[str, ModelInfo] = {
    "gemini-2.0-flash-001": ModelInfo(
        max_tokens=8192,
        context_window=1_048_576,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=0,
        output_price=0,
        description="Google's latest multimodal model with excellent performance",
    ),
    "gemini-2.0-flash-lite-preview-02-05": ModelInfo(
        max_tokens=8192,
        context_window=1_048_576,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=0,
        output_price=0,
        description="Lighter version of Gemini 2.0 Flash",
    ),
    "gemini-2.0-pro-exp-02-05": ModelInfo(
        max_tokens=8192,
        context_window=2_097_152,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=0,
        output_price=0,
        description="Google's most powerful Gemini model",
    ),
    "gemini-2.0-flash-thinking-exp-01-21": ModelInfo(
        max_tokens=65_536,
        context_window=1_048_576,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=0,
        output_price=0,
        description="Experimental Gemini model with thinking capabilities",
    ),
    "gemini-2.0-flash-thinking-exp-1219": ModelInfo(
        max_tokens=8192,
        context_window=32_767,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=0,
        output_price=0,
        description="Experimental Gemini model with thinking capabilities",
    ),
    "gemini-2.0-flash-exp": ModelInfo(
        max_tokens=8192,
        context_window=1_048_576,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=0,
        output_price=0,
        description="Experimental version of Gemini 2.0 Flash",
    ),
    "gemini-1.5-flash-002": ModelInfo(
        max_tokens=8192,
        context_window=1_048_576,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=0,
        output_price=0,
        description="Fast and efficient multimodal model",
    ),
    "gemini-1.5-flash-exp-0827": ModelInfo(
        max_tokens=8192,
        context_window=1_048_576,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=0,
        output_price=0,
        description="Experimental version of Gemini 1.5 Flash",
    ),
}

google_models["gemini-2.0-flash"] = google_models["gemini-2.0-flash-001"]

deepseek_models: Dict[str, ModelInfo] = {
    "deepseek-chat": ModelInfo(
        max_tokens=8_000,
        context_window=64_000,
        supports_images=False,
        supports_prompt_cache=True,
        input_price=0,
        output_price=0.28,
        cache_writes_price=0.14,
        cache_reads_price=0.014,
        description="General purpose chat model",
    ),
    "deepseek-reasoner": ModelInfo(
        max_tokens=8_000,
        context_window=64_000,
        supports_images=False,
        supports_prompt_cache=True,
        input_price=0,
        output_price=2.19,
        cache_writes_price=0.55,
        cache_reads_price=0.14,
        description="Specialized for complex reasoning tasks",
    ),
}

qwen_models: Dict[str, ModelInfo] = {
    "qwen2.5-coder-32b-instruct": ModelInfo(
        max_tokens=8_192,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.002,
        output_price=0.006,
        cache_writes_price=0.002,
        cache_reads_price=0.006,
        description="Specialized for code generation and understanding",
    ),
    "qwen2.5-coder-14b-instruct": ModelInfo(
        max_tokens=8_192,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.002,
        output_price=0.006,
        cache_writes_price=0.002,
        cache_reads_price=0.006,
        description="Medium-sized code-specialized model",
    ),
    "qwen2.5-coder-7b-instruct": ModelInfo(
        max_tokens=8_192,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.001,
        output_price=0.002,
        cache_writes_price=0.001,
        cache_reads_price=0.002,
        description="Efficient code-specialized model",
    ),
    "qwen2.5-coder-3b-instruct": ModelInfo(
        max_tokens=8_192,
        context_window=32_768,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.0,
        output_price=0.0,
        cache_writes_price=0.0,
        cache_reads_price=0.0,
        description="Compact code-specialized model",
    ),
    "qwen2.5-coder-1.5b-instruct": ModelInfo(
        max_tokens=8_192,
        context_window=32_768,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.0,
        output_price=0.0,
        cache_writes_price=0.0,
        cache_reads_price=0.0,
        description="Very compact code-specialized model",
    ),
    "qwen2.5-coder-0.5b-instruct": ModelInfo(
        max_tokens=8_192,
        context_window=32_768,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.0,
        output_price=0.0,
        cache_writes_price=0.0,
        cache_reads_price=0.0,
        description="Smallest code-specialized model",
    ),
    "qwen-coder-plus-latest": ModelInfo(
        max_tokens=129_024,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=3.5,
        output_price=7,
        cache_writes_price=3.5,
        cache_reads_price=7,
        description="Advanced code generation model",
    ),
    "qwen-plus-latest": ModelInfo(
        max_tokens=129_024,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.8,
        output_price=2,
        cache_writes_price=0.8,
        cache_reads_price=0.2,
        description="Balanced performance Qwen model",
    ),
    "qwen-turbo-latest": ModelInfo(
        max_tokens=1_000_000,
        context_window=1_000_000,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.8,
        output_price=2,
        cache_writes_price=0.8,
        cache_reads_price=2,
        description="Fast and efficient Qwen model",
    ),
    "qwen-max-latest": ModelInfo(
        max_tokens=30_720,
        context_window=32_768,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=2.4,
        output_price=9.6,
        cache_writes_price=2.4,
        cache_reads_price=9.6,
        description="Alibaba's most powerful Qwen model",
    ),
    "qwen-coder-plus": ModelInfo(
        max_tokens=129_024,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=3.5,
        output_price=7,
        cache_writes_price=3.5,
        cache_reads_price=7,
        description="Advanced code generation model",
    ),
    "qwen-plus": ModelInfo(
        max_tokens=129_024,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.8,
        output_price=2,
        cache_writes_price=0.8,
        cache_reads_price=0.2,
        description="Balanced performance Qwen model",
    ),
    "qwen-turbo": ModelInfo(
        max_tokens=1_000_000,
        context_window=1_000_000,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.3,
        output_price=0.6,
        cache_writes_price=0.3,
        cache_reads_price=0.6,
        description="Fast and efficient Qwen model",
    ),
    "qwen-max": ModelInfo(
        max_tokens=30_720,
        context_window=32_768,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=2.4,
        output_price=9.6,
        cache_writes_price=2.4,
        cache_reads_price=9.6,
        description="Alibaba's most powerful Qwen model",
    ),
    "deepseek-v3": ModelInfo(
        max_tokens=8_000,
        context_window=64_000,
        supports_images=False,
        supports_prompt_cache=True,
        input_price=0,
        output_price=0.28,
        cache_writes_price=0.14,
        cache_reads_price=0.014,
        description="General purpose chat model",
    ),
    "deepseek-r1": ModelInfo(
        max_tokens=8_000,
        context_window=64_000,
        supports_images=False,
        supports_prompt_cache=True,
        input_price=0,
        output_price=2.19,
        cache_writes_price=0.55,
        cache_reads_price=0.14,
        description="Specialized for complex reasoning tasks",
    ),
    "qwen-vl-max": ModelInfo(
        max_tokens=30_720,
        context_window=32_768,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=3,
        output_price=9,
        cache_writes_price=3,
        cache_reads_price=9,
        description="Multimodal Qwen model with vision capabilities",
    ),
    "qwen-vl-max-latest": ModelInfo(
        max_tokens=129_024,
        context_window=131_072,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=3,
        output_price=9,
        cache_writes_price=3,
        cache_reads_price=9,
        description="Multimodal Qwen model with vision capabilities",
    ),
    "qwen-vl-plus": ModelInfo(
        max_tokens=6_000,
        context_window=8_000,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=1.5,
        output_price=4.5,
        cache_writes_price=1.5,
        cache_reads_price=4.5,
        description="Balanced multimodal Qwen model",
    ),
    "qwen-vl-plus-latest": ModelInfo(
        max_tokens=129_024,
        context_window=131_072,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=1.5,
        output_price=4.5,
        cache_writes_price=1.5,
        cache_reads_price=4.5,
        description="Balanced multimodal Qwen model",
    ),
}

mistral_models: Dict[str, ModelInfo] = {
    "mistral-large-2411": ModelInfo(
        max_tokens=131_000,
        context_window=131_000,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=2.0,
        output_price=6.0,
        description="Mistral's most powerful model",
    ),
    "pixtral-large-2411": ModelInfo(
        max_tokens=131_000,
        context_window=131_000,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=2.0,
        output_price=6.0,
        description="Mistral's multimodal model with image capabilities",
    ),
    "ministral-3b-2410": ModelInfo(
        max_tokens=131_000,
        context_window=131_000,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.04,
        output_price=0.04,
        description="Compact 3B parameter model for efficient inference",
    ),
    "ministral-8b-2410": ModelInfo(
        max_tokens=131_000,
        context_window=131_000,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.1,
        output_price=0.1,
        description="Medium-sized 8B parameter model balancing performance and efficiency",
    ),
    "mistral-small-2501": ModelInfo(
        max_tokens=32_000,
        context_window=32_000,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.1,
        output_price=0.3,
        description="Fast and efficient model for simpler tasks",
    ),
    "pixtral-12b-2409": ModelInfo(
        max_tokens=131_000,
        context_window=131_000,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=0.15,
        output_price=0.15,
        description="12B parameter multimodal model with vision capabilities",
    ),
    "open-mistral-nemo-2407": ModelInfo(
        max_tokens=131_000,
        context_window=131_000,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.15,
        output_price=0.15,
        description="Open-source version of Mistral optimized with NVIDIA NeMo",
    ),
    "open-codestral-mamba": ModelInfo(
        max_tokens=256_000,
        context_window=256_000,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.15,
        output_price=0.15,
        description="Open-source code-specialized model using Mamba architecture",
    ),
    "codestral-2501": ModelInfo(
        max_tokens=256_000,
        context_window=256_000,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=0.3,
        output_price=0.9,
        description="Specialized for code generation and understanding",
    ),
}

mistral_models["mistral-large-latest"] = mistral_models["mistral-large-2411"]

litellm_model_info_sane_defaults: ModelInfo = ModelInfo(
    max_tokens=-1,
    context_window=128_000,
    supports_images=True,
    supports_prompt_cache=False,
    input_price=0,
    output_price=0,
    description="LiteLLM proxy for accessing various AI models",
)

YUAN_TO_USD = 0.14

kimi_models: Dict[str, ModelInfo] = {
    "moonshot-v1-8k": ModelInfo(
        max_tokens=8192,
        context_window=8192,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=12.00 * YUAN_TO_USD,
        output_price=12.00 * YUAN_TO_USD,
        cache_writes_price=24.00 * YUAN_TO_USD,
        cache_reads_price=0.02 * YUAN_TO_USD,
        description="General purpose language model with 8K context",
    ),
    "moonshot-v1-32k": ModelInfo(
        max_tokens=8192,
        context_window=32_768,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=24.00 * YUAN_TO_USD,
        output_price=24.00 * YUAN_TO_USD,
        cache_writes_price=24.00 * YUAN_TO_USD,
        cache_reads_price=0.02 * YUAN_TO_USD,
        description="General purpose language model with 32K context",
    ),
    "moonshot-v1-128k": ModelInfo(
        max_tokens=8192,
        context_window=131_072,
        supports_images=False,
        supports_prompt_cache=False,
        input_price=60.00 * YUAN_TO_USD,
        output_price=60.00 * YUAN_TO_USD,
        cache_writes_price=24.00 * YUAN_TO_USD,
        cache_reads_price=0.02 * YUAN_TO_USD,
        description="General purpose language model with 128K context",
    ),
    "moonshot-v1-8k-vision-preview": ModelInfo(
        max_tokens=8192,
        context_window=8192,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=12.00 * YUAN_TO_USD,
        output_price=12.00 * YUAN_TO_USD,
        cache_writes_price=24.00 * YUAN_TO_USD,
        cache_reads_price=0.02 * YUAN_TO_USD,
        description="Multimodal model with 8K context",
    ),
    "moonshot-v1-32k-vision-preview": ModelInfo(
        max_tokens=8192,
        context_window=32_768,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=24.00 * YUAN_TO_USD,
        output_price=24.00 * YUAN_TO_USD,
        cache_writes_price=24.00 * YUAN_TO_USD,
        cache_reads_price=0.02 * YUAN_TO_USD,
        description="Multimodal model with 32K context",
    ),
    "moonshot-v1-128k-vision-preview": ModelInfo(
        max_tokens=8192,
        context_window=131_072,
        supports_images=True,
        supports_prompt_cache=False,
        input_price=60.00 * YUAN_TO_USD,
        output_price=60.00 * YUAN_TO_USD,
        cache_writes_price=24.00 * YUAN_TO_USD,
        cache_reads_price=0.02 * YUAN_TO_USD,
        description="Multimodal model with 128K context",
    ),
}
