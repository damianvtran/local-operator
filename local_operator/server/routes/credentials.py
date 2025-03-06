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
    """
    try:
        # Get credentials from the credential manager
        non_empty_credentials = credential_manager.list_credential_keys(non_empty=True)

        result = CredentialListResult(keys=non_empty_credentials)

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
):
    """
    Update an existing credential or create a new one.
    """
    try:
        # Validate the key
        if not credential_data.key:
            raise HTTPException(status_code=400, detail="Credential key cannot be empty")

        # Set the credential
        credential_manager.set_credential(credential_data.key, credential_data.value)

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
