"""
Credential management endpoints for the Local Operator API.

This module contains the FastAPI route handlers for credential-related endpoints.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from local_operator.credentials import CredentialManager
from local_operator.server.dependencies import get_credential_manager
from local_operator.server.models.schemas import (
    CredentialListResult,
    CredentialUpdate,
    CRUDResponse,
)

router = APIRouter(tags=["Credentials"])
logger = logging.getLogger("local_operator.server.routes.credentials")


@router.get(
    "/v1/credentials",
    response_model=CRUDResponse[CredentialListResult],
    summary="List credentials",
    description="Retrieve a list of credential keys (without their values).",
    openapi_extra={
        "responses": {
            "200": {
                "description": "Credentials list retrieved successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Credentials retrieved successfully",
                            "result": {
                                "keys": ["OPENAI_API_KEY", "SERPAPI_API_KEY", "TAVILY_API_KEY"],
                            },
                        }
                    }
                },
            }
        },
    },
)
async def list_credentials(
    credential_manager: CredentialManager = Depends(get_credential_manager),
):
    """
    Retrieve a list of credential keys (without their values).

    Lists the provider-class STORE rows (presented with the reserved
    ``LOP_PROVIDER_`` prefix stripped, so a caller sees the env-key spelling it
    configured) UNIONED with the store's plain agent-class rows. The legacy
    ``CredentialManager`` file keys this used to union are GONE (PR2a) — the
    plaintext file is no longer a credential source. The desktop Settings section
    that consumed this is being removed separately; the endpoint stays correct
    for any other client.
    """
    try:
        # Provider-class store rows first, keyed by env-key name.
        from local_operator.providers.registry import stored_provider_env_keys

        keys = set(stored_provider_env_keys(credential_manager.config_dir))
        # Namespace-scoped plain agent secrets too — `lop secret set
        # OPENAI_API_KEY` is a value this endpoint's PATCH would list.
        from local_operator.secrets.access import open_store
        from local_operator.secrets.errors import SecretStoreError
        from local_operator.secrets.keys import store_path
        from local_operator.secrets.store import PROVIDER_SECRET_PREFIX

        if store_path(credential_manager.config_dir).exists():
            try:
                # The provider-class rows are STRIPPED here exactly as the
                # docstring promises. Adding the raw name made every provider
                # credential appear twice — once as ``LOP_PROVIDER_<KEY>`` from
                # this loop and once as ``<KEY>`` from ``stored_provider_env_keys``
                # above — so a client keyed on this list (the Settings UI) showed
                # a phantom, un-configurable second row for every provider key
                # (QA Q-2). Agent-class rows are unprefixed and pass through
                # untouched, which is what keeps them listed.
                keys.update(
                    (
                        record.name[len(PROVIDER_SECRET_PREFIX) :]
                        if record.name.startswith(PROVIDER_SECRET_PREFIX)
                        else record.name
                    )
                    for record in open_store(credential_manager.config_dir).list()
                )
            except (SecretStoreError, OSError):
                logger.warning("credential store unavailable", exc_info=True)

        result = CredentialListResult(keys=sorted(keys))

        return CRUDResponse(
            status=200,
            message="Credentials retrieved successfully",
            result=result.model_dump(),
        )
    except Exception as e:
        logger.exception("Error retrieving credentials")
        raise HTTPException(status_code=500, detail=f"Error retrieving credentials: {e}")


@router.patch(
    "/v1/credentials",
    response_model=CRUDResponse,
    summary="Update a credential",
    description="Update an existing credential or create a new one with the provided key "
    "and value.",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "example": {
                            "summary": "Update Credential Example",
                            "value": {
                                "key": "OPENAI_API_KEY",
                                "value": "sk-abcdefghijklmnopqrstuvwxyz",
                            },
                        }
                    }
                }
            }
        },
        "responses": {
            "200": {
                "description": "Credential updated successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Credential updated successfully",
                            "result": {},
                        }
                    }
                },
            }
        },
    },
)
async def update_credential(
    credential_data: CredentialUpdate,
    credential_manager: CredentialManager = Depends(get_credential_manager),
) -> JSONResponse:
    """
    Update an existing credential or create a new one.

    Writes a provider-class STORE row (``LOP_PROVIDER_<KEY>``, ``role="provider"``)
    — the consolidated home for provider keys — rather than the plaintext
    ``credentials.env``. The store row is what the store-first readers resolve,
    so this is the change that makes a Settings-set key take effect.
    """
    try:
        # Validate the key
        if not credential_data.key:
            raise HTTPException(status_code=400, detail="Credential key cannot be empty")

        # Set the credential as a provider-class store row, under the manager's
        # own config root so a server with a custom root writes where it reads.
        from local_operator.providers.registry import store_provider_key

        store_provider_key(
            credential_data.key, credential_data.value, base=credential_manager.config_dir
        )

        # A key is exactly the reason model metadata resolves poorly: without one
        # a provider's listing 401s and every model it describes falls back to the
        # 128k unknown default. That answer is memoized per TTL bucket, so in a
        # server — which runs for days — fixing the credential would otherwise
        # change nothing until the bucket rolled. Imported lazily to keep the
        # model/harness stack off this route's import path.
        from local_operator.model.configure import invalidate_model_info_cache

        invalidate_model_info_cache()

        response = CRUDResponse(
            status=200,
            message="Credential updated successfully",
            result={},
        )
        return JSONResponse(status_code=200, content=jsonable_encoder(response))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error updating credential")
        raise HTTPException(status_code=500, detail=f"Error updating credential: {e}")
