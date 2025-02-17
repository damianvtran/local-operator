from typing import Any, Dict, Optional, Union

import requests
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from local_operator.clients.openrouter import OpenRouterClient
from local_operator.credentials import CredentialManager
from local_operator.mocks import ChatMock, ChatNoop
from local_operator.model.registry import (
    ModelInfo,
    get_model_info,
    openrouter_default_model_info,
)

ModelType = Union[ChatOpenAI, ChatOllama, ChatAnthropic, ChatGoogleGenerativeAI, ChatMock, ChatNoop]


class ModelConfiguration:
    """
    Configuration class for language models.

    Attributes:
        hosting (str): The hosting provider name
        name (str): The model name
        instance (ModelType): An instance of the language model (e.g., ChatOpenAI,
        ChatOllama).
        info (ModelInfo): Information about the model, such as pricing and rate limits.
        api_key (Optional[SecretStr]): API key for the model.
    """

    hosting: str
    name: str
    instance: ModelType
    info: ModelInfo
    api_key: Optional[SecretStr] = None

    def __init__(
        self,
        hosting: str,
        name: str,
        instance: ModelType,
        info: ModelInfo,
        api_key: Optional[SecretStr] = None,
    ):
        self.hosting = hosting
        self.name = name
        self.instance = instance
        self.info = info
        self.api_key = api_key


def _check_model_exists_payload(hosting: str, model: str, response_data: Dict[str, Any]) -> bool:
    """Check if a model exists in the provider's response data.

    Args:
        hosting (str): The hosting provider name
        model (str): The model name to check
        response_data (dict): Raw response data from the provider's API

    Returns:
        bool: True if model exists in the response data, False otherwise
    """
    if hosting == "google":
        # Google uses "models" key and model name in format "models/model-name"
        models = response_data.get("models", [])
        return any(m.get("name", "").replace("models/", "") == model for m in models)

    if hosting == "ollama":
        # Ollama uses "models" key with "name" field
        models = response_data.get("models", [])
        return any(m.get("name", "") == model for m in models)

    # Other providers use "data" key
    models = response_data.get("data", [])
    if not models:
        return False

    # Handle special case for Anthropic "latest" models
    if hosting == "anthropic" and model.endswith("-latest"):
        base_model = model.replace("-latest", "")
        # Check if any model ID starts with the base model name
        return any(m.get("id", "").startswith(base_model) for m in models)

    # Different providers use different model ID fields
    for m in models:
        model_id = m.get("id") or m.get("name") or ""
        if model_id == model:
            return True
    return False


def validate_model(hosting: str, model: str, api_key: SecretStr) -> bool:
    """Validate if the model exists and API key is valid by calling provider's model list API.

    Args:
        hosting (str): The hosting provider name
        model (str): The model name to validate
        api_key (SecretStr): API key to use for validation

    Returns:
        bool: True if model exists and API key is valid, False otherwise

    Raises:
        requests.exceptions.RequestException: If API request fails
    """
    if hosting == "deepseek":
        response = requests.get(
            "https://api.deepseek.com/v1/models",
            headers={"Authorization": f"Bearer {api_key.get_secret_value()}"},
        )
    elif hosting == "openai":
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key.get_secret_value()}"},
        )
    elif hosting == "openrouter":
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key.get_secret_value()}"},
        )
    elif hosting == "anthropic":
        response = requests.get(
            "https://api.anthropic.com/v1/models",
            headers={"x-api-key": api_key.get_secret_value(), "anthropic-version": "2023-06-01"},
        )
    elif hosting == "kimi":
        response = requests.get(
            "https://api.moonshot.cn/v1/models",
            headers={"Authorization": f"Bearer {api_key.get_secret_value()}"},
        )
    elif hosting == "alibaba":
        response = requests.get(
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models",
            headers={"Authorization": f"Bearer {api_key.get_secret_value()}"},
        )
    elif hosting == "google":
        response = requests.get(
            "https://generativelanguage.googleapis.com/v1/models",
            headers={"x-goog-api-key": api_key.get_secret_value()},
        )
    elif hosting == "mistral":
        response = requests.get(
            "https://api.mistral.ai/v1/models",
            headers={"Authorization": f"Bearer {api_key.get_secret_value()}"},
        )
    elif hosting == "ollama":
        # Ollama is local, so just check if model exists
        response = requests.get("http://localhost:11434/api/tags")
    else:
        return True

    if response.status_code == 200:
        return _check_model_exists_payload(hosting, model, response.json())
    return False


def get_model_info_from_openrouter(client: OpenRouterClient, model_name: str) -> ModelInfo:
    """
    Retrieves model information from OpenRouter based on the model name.

    Args:
        client (OpenRouterClient): The OpenRouter client instance.
        model_name (str): The name of the model to retrieve information for.

    Returns:
        ModelInfo: The model information retrieved from OpenRouter.

    Raises:
        ValueError: If the model is not found on OpenRouter.
        RuntimeError: If there is an error retrieving the model information.
    """
    models = client.list_models()
    for model in models.data:
        if model.id == model_name:
            model_info = openrouter_default_model_info
            # Openrouter returns the price per million tokens, so we need to convert it to
            # the price per token.
            model_info.input_price = model.pricing.prompt * 1_000_000
            model_info.output_price = model.pricing.completion * 1_000_000
            model_info.description = model.description
            return model_info

    raise ValueError(f"Model not found from openrouter models API: {model_name}")


def configure_model(
    hosting: str,
    model_name: str,
    credential_manager: CredentialManager,
    model_info_client: Optional[OpenRouterClient] = None,
) -> ModelConfiguration:
    """Configure and return the appropriate model based on hosting platform.

    Args:
        hosting (str): Hosting platform (deepseek, openai, anthropic, ollama, or noop)
        model_name (str): Model name to use
        credential_manager: CredentialManager instance for API key management
        model_info_client: OpenRouterClient instance for model info

    Returns:
        ModelConfiguration: Config object containing the configured model instance and API
        key if applicable

    Raises:
        ValueError: If hosting is not provided or unsupported
    """
    if not hosting:
        raise ValueError("Hosting is required")

    # Early return for test and noop cases
    if hosting == "test":
        return ModelConfiguration(
            hosting=hosting,
            name=model_name,
            instance=ChatMock(),
            info=ModelInfo(),
        )
    if hosting == "noop":
        return ModelConfiguration(
            hosting=hosting,
            name=model_name,
            instance=ChatNoop(),
            info=ModelInfo(),
        )

    configured_model = None
    api_key: Optional[SecretStr] = None

    if hosting == "deepseek":
        base_url = "https://api.deepseek.com/v1"
        if not model_name:
            model_name = "deepseek-chat"
        api_key = credential_manager.get_credential("DEEPSEEK_API_KEY")
        if not api_key:
            api_key = credential_manager.prompt_for_credential("DEEPSEEK_API_KEY")
        configured_model = ChatOpenAI(
            api_key=api_key,
            temperature=0.3,
            base_url=base_url,
            model=model_name,
        )

    elif hosting == "openai":
        if not model_name:
            model_name = "gpt-4o"
        api_key = credential_manager.get_credential("OPENAI_API_KEY")
        if not api_key:
            api_key = credential_manager.prompt_for_credential("OPENAI_API_KEY")
        temperature = 1.0 if model_name.startswith(("o1", "o3")) else 0.3
        configured_model = ChatOpenAI(
            api_key=api_key,
            temperature=temperature,
            model=model_name,
        )

    elif hosting == "openrouter":
        if not model_name:
            model_name = "google/gemini-2.0-flash-001"
        api_key = credential_manager.get_credential("OPENROUTER_API_KEY")
        if not api_key:
            api_key = credential_manager.prompt_for_credential("OPENROUTER_API_KEY")
        configured_model = ChatOpenAI(
            api_key=api_key,
            temperature=0.3,
            model=model_name,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://local-operator.com",
                "X-Title": "Local Operator",
            },
        )

    elif hosting == "anthropic":
        if not model_name:
            model_name = "claude-3-5-sonnet-latest"
        api_key = credential_manager.get_credential("ANTHROPIC_API_KEY")
        if not api_key:
            api_key = credential_manager.prompt_for_credential("ANTHROPIC_API_KEY")

        if not api_key:
            raise ValueError("Anthropic API key is required")

        configured_model = ChatAnthropic(
            api_key=api_key,
            temperature=0.3,
            model_name=model_name,
            timeout=None,
            stop=None,
        )

    elif hosting == "kimi":
        if not model_name:
            model_name = "moonshot-v1-32k"
        api_key = credential_manager.get_credential("KIMI_API_KEY")
        if not api_key:
            api_key = credential_manager.prompt_for_credential("KIMI_API_KEY")
        configured_model = ChatOpenAI(
            api_key=api_key,
            temperature=0.3,
            model=model_name,
            base_url="https://api.moonshot.cn/v1",
        )

    elif hosting == "alibaba":
        if not model_name:
            model_name = "qwen-plus"
        api_key = credential_manager.get_credential("ALIBABA_CLOUD_API_KEY")
        if not api_key:
            api_key = credential_manager.prompt_for_credential("ALIBABA_CLOUD_API_KEY")
        configured_model = ChatOpenAI(
            api_key=api_key,
            temperature=0.3,
            model=model_name,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )

    elif hosting == "google":
        if not model_name:
            model_name = "gemini-2.0-flash-001"
        api_key = credential_manager.get_credential("GOOGLE_AI_STUDIO_API_KEY")
        if not api_key:
            api_key = credential_manager.prompt_for_credential("GOOGLE_AI_STUDIO_API_KEY")
        configured_model = ChatGoogleGenerativeAI(
            api_key=api_key,
            temperature=0.3,
            model=model_name,
        )

    elif hosting == "mistral":
        if not model_name:
            model_name = "mistral-large-latest"
        api_key = credential_manager.get_credential("MISTRAL_API_KEY")
        if not api_key:
            api_key = credential_manager.prompt_for_credential("MISTRAL_API_KEY")
        configured_model = ChatOpenAI(
            api_key=api_key,
            temperature=0.3,
            model=model_name,
            base_url="https://api.mistral.ai/v1",
        )

    elif hosting == "ollama":
        if not model_name:
            raise ValueError("Model is required for ollama hosting")
        configured_model = ChatOllama(
            model=model_name,
            temperature=0.3,
        )

    else:
        raise ValueError(f"Unsupported hosting platform: {hosting}")

    model_info: ModelInfo

    if model_info_client:
        if hosting == "openrouter":
            model_info = get_model_info_from_openrouter(model_info_client, model_name)
        else:
            raise ValueError(f"Model info client not supported for hosting: {hosting}")
    else:
        model_info = get_model_info(hosting, model_name)

    return ModelConfiguration(
        hosting=hosting,
        name=model_name,
        instance=configured_model,
        info=model_info,
        api_key=api_key,
    )


def calculate_cost(model_info: ModelInfo, input_tokens: int, output_tokens: int) -> float:
    """
    Calculates the cost of a request based on token usage and model pricing.

    Args:
        model_info (ModelInfo): The pricing information for the model.
        input_tokens (int): The number of input tokens used in the request.
        output_tokens (int): The number of output tokens generated by the request.

    Returns:
        float: The total cost of the request.

    Raises:
        ValueError: If there is an error during cost calculation.
    """
    try:
        input_cost = (float(input_tokens) / 1_000_000.0) * model_info.input_price
        output_cost = (float(output_tokens) / 1_000_000.0) * model_info.output_price
        total_cost = input_cost + output_cost
        return total_cost
    except Exception as e:
        raise ValueError(f"Error calculating cost: {e}") from e
