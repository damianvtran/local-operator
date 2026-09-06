"""
Model endpoints for the Local Operator API.

This module contains the FastAPI route handlers for model-related endpoints.
"""

import asyncio
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from local_operator.clients.openrouter import OpenRouterClient
from local_operator.clients.radient import RadientClient
from local_operator.credentials import CredentialManager
from local_operator.env import EnvConfig
from local_operator.model.discovery import available_models
from local_operator.model.registry import (
    ProviderDetail,
    RecommendedOpenRouterModelIds,
    RecommendedRadientModelIds,
    SupportedHostingProviders,
    anthropic_models,
    deepseek_models,
    glm_models,
    google_models,
    kimi_models,
    mistral_models,
    openai_models,
    qwen_models,
    xai_models,
)
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.local import (
    LOCAL_PROVIDER_IDS,
    local_api_key,
    local_model_info,
    resolve_base_url,
)
from local_operator.server.dependencies import (
    get_credential_manager,
    get_env_config,
    get_provider_auth_store,
)
from local_operator.server.models.schemas import (
    CRUDResponse,
    ModelEntry,
    ModelInfo,
    ModelListQueryParams,
    ModelListQuerySort,
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
                                {
                                    "id": "openai",
                                    "name": "OpenAI",
                                    "description": "OpenAI's API provides access to GPT-4o",
                                    "url": "https://platform.openai.com/",
                                    "requiredCredentials": ["OPENAI_API_KEY"],
                                },
                                {
                                    "id": "anthropic",
                                    "name": "Anthropic",
                                    "description": "Anthropic's Claude models for AI assistants",
                                    "url": "https://www.anthropic.com/",
                                    "requiredCredentials": ["ANTHROPIC_API_KEY"],
                                },
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
async def list_providers() -> CRUDResponse[ProviderListResponse]:
    """
    List all available model providers with their details.

    Returns:
        CRUDResponse: A response containing the list of provider objects with their details.
    """
    try:
        # Discovery belongs to the model request. Hiding stopped servers here
        # made them impossible to select in the desktop application's setup UI.
        provider_details = SupportedHostingProviders

        return CRUDResponse(
            status=200,
            message="Providers retrieved successfully",
            result=ProviderListResponse(providers=provider_details),
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
        "models if API key is configured. Optionally filter by provider and sort by field."
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
                                        "recommended": False,
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
                                        "recommended": True,
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
    provider_auth_store: AuthStore = Depends(get_provider_auth_store),
    query_params: ModelListQueryParams = Depends(),
    env_config: EnvConfig = Depends(get_env_config),
) -> CRUDResponse[ModelListResponse]:
    """
    List all available models from all providers.

    This endpoint returns models from the registry and also includes OpenRouter models
    if the API key is configured. Results can be filtered by provider and sorted by field.

    Args:
        credential_manager: Dependency for managing credentials
        query_params: Query parameters for filtering and sorting models

    Returns:
        CRUDResponse: A response containing the list of models.

    Raises:
        HTTPException: If provider is invalid or on server error
    """
    try:
        models: List[ModelEntry] = []
        providers_to_check: List[ProviderDetail] = []

        if query_params.provider:
            if query_params.provider not in [p.id for p in SupportedHostingProviders]:
                raise HTTPException(
                    status_code=404, detail=f"Provider not found: {query_params.provider}"
                )

            providers_to_check = [
                p for p in SupportedHostingProviders if p.id == query_params.provider
            ]
        else:
            providers_to_check = SupportedHostingProviders

        # Add models from each provider
        for provider_detail in providers_to_check:
            if provider_detail.id == "anthropic":
                for model_name, model_info in anthropic_models.items():
                    models.append(
                        ModelEntry(
                            id=model_name,
                            name=model_info.name,
                            provider=provider_detail.id,
                            info=model_info,
                        )
                    )
            elif provider_detail.id == "deepseek":
                for model_name, model_info in deepseek_models.items():
                    models.append(
                        ModelEntry(
                            id=model_name,
                            name=model_info.name,
                            provider=provider_detail.id,
                            info=model_info,
                        )
                    )
            elif provider_detail.id == "google":
                for model_name, model_info in google_models.items():
                    models.append(
                        ModelEntry(
                            id=model_name,
                            name=model_info.name,
                            provider=provider_detail.id,
                            info=model_info,
                        )
                    )
            elif provider_detail.id == "kimi":
                for model_name, model_info in kimi_models.items():
                    models.append(
                        ModelEntry(
                            id=model_name,
                            name=model_info.name,
                            provider=provider_detail.id,
                            info=model_info,
                        )
                    )
            elif provider_detail.id == "mistral":
                for model_name, model_info in mistral_models.items():
                    models.append(
                        ModelEntry(
                            id=model_name,
                            name=model_info.name,
                            provider=provider_detail.id,
                            info=model_info,
                        )
                    )
            elif provider_detail.id in LOCAL_PROVIDER_IDS:

                def local_entries(provider: str) -> list[ModelEntry]:
                    try:
                        endpoint = resolve_base_url(provider)
                    except ValueError:
                        logger.warning(
                            "Invalid endpoint configuration for local provider %s", provider
                        )
                        return []
                    if not endpoint:
                        return []
                    rows, _ = available_models(
                        provider,
                        api_key=local_api_key(provider, endpoint=endpoint),
                        base_url=endpoint,
                    )
                    result = []
                    for row in rows:
                        info = local_model_info(provider, row.id, base_url=endpoint).model_copy(
                            update={"name": row.name or row.id}
                        )
                        result.append(
                            ModelEntry(
                                id=row.id, name=row.name or row.id, provider=provider, info=info
                            )
                        )
                    return result

                # Native metadata reads share discovery's deadline and cache;
                # no synchronous HTTP may block the FastAPI event loop.
                models.extend(await asyncio.to_thread(local_entries, provider_detail.id))
            elif provider_detail.id == "openai":
                for model_name, model_info in openai_models.items():
                    models.append(
                        ModelEntry(
                            id=model_name,
                            name=model_info.name,
                            provider=provider_detail.id,
                            info=model_info,
                        )
                    )
            elif provider_detail.id == "alibaba":
                for model_name, model_info in qwen_models.items():
                    models.append(
                        ModelEntry(
                            id=model_name,
                            name=model_info.name,
                            provider=provider_detail.id,
                            info=model_info,
                        )
                    )
            elif provider_detail.id == "xai":
                for model_name, model_info in xai_models.items():
                    models.append(
                        ModelEntry(
                            id=model_name,
                            name=model_info.name,
                            provider=provider_detail.id,
                            info=model_info,
                        )
                    )
            elif provider_detail.id == "zai":
                for model_name, model_info in glm_models.items():
                    models.append(
                        ModelEntry(
                            id=model_name,
                            name=model_info.name,
                            provider=provider_detail.id,
                            info=model_info,
                        )
                    )
            elif provider_detail.id == "openrouter":
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
                            model_pricing_prompt = model.pricing.prompt
                            model_pricing_completion = model.pricing.completion
                            if model_pricing_prompt is None or model_pricing_prompt < 0.0:
                                model_pricing_prompt = 0.0
                            if model_pricing_completion is None or model_pricing_completion < 0.0:
                                model_pricing_completion = 0.0

                            # Get model info
                            model_info = ModelInfo(
                                id=model.id,
                                name=model.name,
                                input_price=model_pricing_prompt * 1_000_000,
                                output_price=model_pricing_completion * 1_000_000,
                                max_tokens=None,
                                context_window=None,
                                supports_images=None,
                                supports_prompt_cache=False,
                                cache_writes_price=None,
                                cache_reads_price=None,
                                recommended=model.id in RecommendedOpenRouterModelIds,
                                description=(
                                    model.description
                                    if model.description
                                    else f"OpenRouter model: {model.name}"
                                ),
                            )

                            models.append(
                                ModelEntry(
                                    id=model.id,
                                    name=model.name,
                                    provider="openrouter",
                                    info=model_info,
                                )
                            )
                    except Exception as e:
                        # Continue without OpenRouter models
                        logger.exception(f"Error fetching OpenRouter models: {e}")
                        pass
            elif provider_detail.id == "radient":
                # Then try to get Radient models if API key is configured
                from local_operator.providers.radient_credentials import (
                    resolve_radient_credential,
                )

                api_key = await resolve_radient_credential(
                    credential_manager, env_config.radient_api_base_url, store=provider_auth_store
                )
                if api_key:
                    try:
                        # Create the Radient client
                        client = RadientClient(
                            api_key=api_key, base_url=env_config.radient_api_base_url
                        )

                        # Get the list of models
                        radient_models = client.list_models()

                        # Add Radient models
                        for model in radient_models.data:
                            model_pricing_prompt = model.pricing.prompt
                            model_pricing_completion = model.pricing.completion
                            if model_pricing_prompt is None or model_pricing_prompt < 0.0:
                                model_pricing_prompt = 0.0
                            if model_pricing_completion is None or model_pricing_completion < 0.0:
                                model_pricing_completion = 0.0

                            # Get model info
                            model_info = ModelInfo(
                                id=model.id,
                                name=model.name,
                                input_price=model_pricing_prompt * 1_000_000,
                                output_price=model_pricing_completion * 1_000_000,
                                max_tokens=None,
                                context_window=None,
                                supports_images=None,
                                supports_prompt_cache=False,
                                cache_writes_price=None,
                                cache_reads_price=None,
                                recommended=model.id in RecommendedRadientModelIds,
                                description=(
                                    model.description
                                    if model.description
                                    else f"Radient model: {model.name}"
                                ),
                            )

                            models.append(
                                ModelEntry(
                                    id=model.id,
                                    name=model.name,
                                    provider="radient",
                                    info=model_info,
                                )
                            )
                    except Exception as e:
                        # Continue without Radient models
                        logger.exception(f"Error fetching Radient models: {e}")
                        pass

        # Sort the models based on the sort parameter and direction
        if query_params.sort == ModelListQuerySort.ID:
            # Sort by id
            models.sort(
                key=lambda model: model.id, reverse=(query_params.direction == "descending")
            )
        elif query_params.sort == ModelListQuerySort.PROVIDER:
            # Sort by provider
            models.sort(
                key=lambda model: model.provider, reverse=(query_params.direction == "descending")
            )
        elif query_params.sort == ModelListQuerySort.NAME:
            # Sort by name, handling None values
            models.sort(
                key=lambda model: (model.name is None, model.name or ""),
                reverse=(query_params.direction == "descending"),
            )
        elif query_params.sort == ModelListQuerySort.RECOMMENDED:
            # Sort by recommended (primary) and id (secondary)
            models.sort(key=lambda model: model.id)  # First sort by id ascending
            models.sort(
                key=lambda model: model.info.recommended,
                reverse=(query_params.direction == "descending"),
            )

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
