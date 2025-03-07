"""
Model endpoints for the Local Operator API.

This module contains the FastAPI route handlers for model-related endpoints.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from local_operator.clients.openrouter import OpenRouterClient
from local_operator.credentials import CredentialManager
from local_operator.model.registry import (
    SupportedHostingProviders,
    anthropic_models,
    deepseek_models,
    google_models,
    kimi_models,
    mistral_models,
    ollama_default_model_info,
    openai_model_info_sane_defaults,
    openrouter_default_model_info,
    qwen_models,
)
from local_operator.server.dependencies import get_credential_manager
from local_operator.server.models.schemas import (
    CRUDResponse,
    ModelInfo,
    ModelListResponse,
    ProviderListResponse,
)

router = APIRouter(tags=["Models"])
logger = logging.getLogger("local_operator.server.routes.models")


@router.get(
    "/v1/models/providers",
    response_model=CRUDResponse[ProviderListResponse],
    summary="List model providers",
    description="Returns a list of available model providers supported by the Local Operator API.",
    responses={
        200: {
            "description": "List of providers retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": 200,
                        "message": "Providers retrieved successfully",
                        "result": {
                            "providers": [
                                "anthropic",
                                "openai",
                                "google",
                                "mistral",
                                "ollama",
                                "openrouter",
                                "deepseek",
                                "kimi",
                                "alibaba",
                            ]
                        },
                    }
                }
            },
        },
        500: {
            "description": "Internal server error",
            "content": {"application/json": {"example": {"detail": "Internal Server Error"}}},
        },
    },
)
async def list_providers():
    """
    List all available model providers.

    Returns:
        CRUDResponse: A response containing the list of providers.
    """
    try:
        # Get the list of providers from the registry
        providers = SupportedHostingProviders

        return CRUDResponse(
            status=200,
            message="Providers retrieved successfully",
            result=ProviderListResponse(providers=providers),
        )
    except Exception:
        logger.exception("Unexpected error while retrieving providers")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get(
    "/v1/models",
    response_model=CRUDResponse[ModelListResponse],
    summary="List all available models",
    description=(
        "Returns a list of all available models from all providers, including OpenRouter "
        "models if API key is configured. Optionally filter by provider."
    ),
    responses={
        200: {
            "description": "List of models retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": 200,
                        "message": "Models retrieved successfully",
                        "result": {
                            "models": [
                                {
                                    "id": "claude-3-opus-20240229",
                                    "provider": "anthropic",
                                    "info": {
                                        "input_price": 15.0,
                                        "output_price": 75.0,
                                        "max_tokens": 200000,
                                        "context_window": 200000,
                                        "supports_images": True,
                                        "supports_prompt_cache": False,
                                        "description": "Most powerful Claude model for "
                                        "highly complex tasks",
                                    },
                                },
                                {
                                    "id": "gpt-4o",
                                    "name": "GPT-4o",
                                    "provider": "openrouter",
                                    "info": {
                                        "input_price": 5.0,
                                        "output_price": 15.0,
                                        "max_tokens": None,
                                        "context_window": None,
                                        "supports_images": None,
                                        "supports_prompt_cache": False,
                                        "description": "OpenAI's most advanced multimodal model",
                                    },
                                },
                            ]
                        },
                    }
                }
            },
        },
        404: {
            "description": "Provider not found",
            "content": {"application/json": {"example": {"detail": "Provider not found: invalid"}}},
        },
        500: {
            "description": "Internal server error",
            "content": {"application/json": {"example": {"detail": "Internal Server Error"}}},
        },
    },
)
async def list_models(
    credential_manager: CredentialManager = Depends(get_credential_manager),
    provider: Optional[str] = None,
):
    """
    List all available models from all providers.

    This endpoint returns models from the registry and also includes OpenRouter models
    if the API key is configured. Results can be filtered by provider.

    Args:
        credential_manager: Dependency for managing credentials
        provider: Optional provider name to filter results

    Returns:
        CRUDResponse: A response containing the list of models.

    Raises:
        HTTPException: If provider is invalid or on server error
    """
    try:
        models = []

        providers_to_check = [provider] if provider else SupportedHostingProviders

        if provider and provider not in SupportedHostingProviders:
            raise HTTPException(status_code=404, detail=f"Provider not found: {provider}")

        # Add models from each provider
        for provider in providers_to_check:
            if provider == "anthropic":
                for model_name, model_info in anthropic_models.items():
                    models.append(
                        {
                            "id": model_name,
                            "provider": provider,
                            "info": model_info.model_dump(),
                        }
                    )
            elif provider == "deepseek":
                for model_name, model_info in deepseek_models.items():
                    models.append(
                        {
                            "id": model_name,
                            "provider": provider,
                            "info": model_info.model_dump(),
                        }
                    )
            elif provider == "google":
                for model_name, model_info in google_models.items():
                    models.append(
                        {
                            "id": model_name,
                            "provider": provider,
                            "info": model_info.model_dump(),
                        }
                    )
            elif provider == "kimi":
                for model_name, model_info in kimi_models.items():
                    models.append(
                        {
                            "id": model_name,
                            "provider": provider,
                            "info": model_info.model_dump(),
                        }
                    )
            elif provider == "mistral":
                for model_name, model_info in mistral_models.items():
                    models.append(
                        {
                            "id": model_name,
                            "provider": provider,
                            "info": model_info.model_dump(),
                        }
                    )
            elif provider == "ollama":
                models.append(
                    {
                        "id": "ollama",
                        "provider": provider,
                        "info": ollama_default_model_info.model_dump(),
                    }
                )
            elif provider == "openai":
                models.append(
                    {
                        "id": "openai",
                        "provider": provider,
                        "info": openai_model_info_sane_defaults.model_dump(),
                    }
                )
            elif provider == "alibaba":
                for model_name, model_info in qwen_models.items():
                    models.append(
                        {
                            "id": model_name,
                            "provider": provider,
                            "info": model_info.model_dump(),
                        }
                    )
            elif provider == "openrouter":
                # First add the default model info
                models.append(
                    {
                        "id": "openrouter",
                        "provider": provider,
                        "info": openrouter_default_model_info.model_dump(),
                    }
                )

                # Then try to get OpenRouter models if API key is configured
                api_key = credential_manager.get_credential("OPENROUTER_API_KEY")
                if api_key:
                    try:
                        # Create the OpenRouter client
                        client = OpenRouterClient(api_key=api_key)

                        # Get the list of models
                        openrouter_models = client.list_models()

                        # Add OpenRouter models
                        for model in openrouter_models.data:
                            # Get model info
                            model_info = ModelInfo(
                                input_price=model.pricing.prompt * 1_000_000,
                                output_price=model.pricing.completion * 1_000_000,
                                max_tokens=None,
                                context_window=None,
                                supports_images=None,
                                supports_prompt_cache=False,
                                cache_writes_price=None,
                                cache_reads_price=None,
                                description=(
                                    model.description
                                    if hasattr(model, "description") and model.description
                                    else f"OpenRouter model: {model.name}"
                                ),
                            )

                            models.append(
                                {
                                    "id": model.id,
                                    "name": model.name,
                                    "provider": "openrouter",
                                    "info": model_info.model_dump(),
                                }
                            )
                    except Exception:
                        # Continue without OpenRouter models
                        pass

        return CRUDResponse(
            status=200,
            message="Models retrieved successfully",
            result=ModelListResponse(models=models),
        )
    except HTTPException:
        # Re-raise HTTP exceptions to preserve their status code and detail
        raise
    except Exception:
        logger.exception("Unexpected error while retrieving models")
        raise HTTPException(status_code=500, detail="Internal Server Error")
