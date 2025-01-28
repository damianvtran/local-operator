from typing import Union

from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


def configure_model(
    hosting: str, model: str, credential_manager
) -> Union[ChatOpenAI, ChatOllama, None]:
    """Configure and return the appropriate model based on hosting platform.

    Args:
        hosting (str): Hosting platform (deepseek, openai, ollama, or noop)
        model (str): Model name to use
        credential_manager: CredentialManager instance for API key management

    Returns:
        Union[ChatOpenAI, ChatOllama, None]: Configured model instance
    """
    if not hosting:
        raise ValueError("Hosting is required")
    if not model:
        raise ValueError("Model is required")

    if hosting == "deepseek":
        base_url = "https://api.deepseek.com/v1"
        api_key = credential_manager.get_api_key("DEEPSEEK_API_KEY")
        if not api_key:
            api_key = credential_manager.prompt_for_api_key("DEEPSEEK_API_KEY")
        return ChatOpenAI(
            api_key=SecretStr(api_key),
            temperature=0.5,
            base_url=base_url,
            model=model,
        )
    elif hosting == "openai":
        base_url = "https://api.openai.com"
        api_key = credential_manager.get_api_key("OPENAI_API_KEY")
        if not api_key:
            api_key = credential_manager.prompt_for_api_key("OPENAI_API_KEY")
        return ChatOpenAI(
            api_key=SecretStr(api_key),
            temperature=0.5,
            base_url=base_url,
            model=model,
        )
    elif hosting == "ollama":
        return ChatOllama(
            model=model,
            temperature=0.5,
        )
    elif hosting == "noop":
        # Useful for testing, will create a dummy operator
        return None
    else:
        raise ValueError(f"Unsupported hosting platform: {hosting}")
