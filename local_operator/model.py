from typing import Union

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


def configure_model(
    hosting: str, model: str, credential_manager
) -> Union[ChatOpenAI, ChatOllama, ChatAnthropic, None]:
    """Configure and return the appropriate model based on hosting platform.

    Args:
        hosting (str): Hosting platform (deepseek, openai, anthropic, ollama, or noop)
        model (str): Model name to use
        credential_manager: CredentialManager instance for API key management

    Returns:
        Union[ChatOpenAI, ChatOllama, ChatAnthropic, None]: Configured model instance
    """
    if not hosting:
        raise ValueError("Hosting is required")

    if hosting == "deepseek":
        base_url = "https://api.deepseek.com/v1"

        if not model:
            model = "deepseek-chat"

        api_key = credential_manager.get_credential("DEEPSEEK_API_KEY")
        if not api_key:
            api_key = credential_manager.prompt_for_credential("DEEPSEEK_API_KEY")

        return ChatOpenAI(
            api_key=SecretStr(api_key),
            temperature=0.3,
            base_url=base_url,
            model=model,
        )
    elif hosting == "openai":
        if not model:
            model = "gpt-4o"

        api_key = credential_manager.get_credential("OPENAI_API_KEY")
        if not api_key:
            api_key = credential_manager.prompt_for_credential("OPENAI_API_KEY")
        return ChatOpenAI(
            api_key=SecretStr(api_key),
            temperature=0.3,
            model=model,
        )
    elif hosting == "anthropic":
        if not model:
            model = "claude-3-5-sonnet-20240620"

        api_key = credential_manager.get_credential("ANTHROPIC_API_KEY")
        if not api_key:
            api_key = credential_manager.prompt_for_credential("ANTHROPIC_API_KEY")
        return ChatAnthropic(
            api_key=SecretStr(api_key),
            temperature=0.3,
            model_name=model,
            timeout=None,
            stop=None,
        )
    elif hosting == "kimi":
        if not model:
            model = "moonshot-v1-32k"

        api_key = credential_manager.get_credential("KIMI_API_KEY")
        if not api_key:
            api_key = credential_manager.prompt_for_credential("KIMI_API_KEY")

        return ChatOpenAI(
            api_key=SecretStr(api_key),
            temperature=0.3,
            model=model,
            base_url="https://api.moonshot.cn/v1",
        )
    elif hosting == "alibaba":
        if not model:
            model = "qwen-plus"

        api_key = credential_manager.get_credential("ALIBABA_CLOUD_API_KEY")
        if not api_key:
            api_key = credential_manager.prompt_for_credential("ALIBABA_CLOUD_API_KEY")

        return ChatOpenAI(
            api_key=SecretStr(api_key),
            temperature=0.3,
            model=model,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
    elif hosting == "ollama":
        if not model:
            raise ValueError("Model is required for ollama hosting")

        return ChatOllama(
            model=model,
            temperature=0.3,
        )
    elif hosting == "noop":
        # Useful for testing, will create a dummy operator
        return None
    else:
        raise ValueError(f"Unsupported hosting platform: {hosting}")
