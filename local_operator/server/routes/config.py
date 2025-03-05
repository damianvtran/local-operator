"""
Configuration management endpoints for the Local Operator API.

This module contains the FastAPI route handlers for configuration-related endpoints.
"""

import logging
from typing import Any, Dict, cast

from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from local_operator.config import ConfigManager
from local_operator.server.dependencies import get_config_manager
from local_operator.server.models.schemas import (
    ConfigResponse,
    ConfigUpdate,
    CRUDResponse,
)

router = APIRouter(tags=["Configuration"])
logger = logging.getLogger("local_operator.server.routes.config")


@router.get(
    "/v1/config",
    response_model=CRUDResponse[ConfigResponse],
    summary="Get configuration",
    description="Retrieve the current configuration settings.",
    openapi_extra={
        "responses": {
            "200": {
                "description": "Configuration retrieved successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Configuration retrieved successfully",
                            "result": {
                                "version": "0.2.16",
                                "metadata": {
                                    "created_at": "2024-01-01T00:00:00",
                                    "last_modified": "2024-01-01T12:00:00",
                                    "description": "Local Operator configuration file",
                                },
                                "values": {
                                    "conversation_length": 100,
                                    "detail_length": 35,
                                    "max_learnings_history": 50,
                                    "hosting": "openrouter",
                                    "model_name": "openai/gpt-4o-mini",
                                    "auto_save_conversation": False,
                                },
                            },
                        }
                    }
                },
            }
        },
    },
)
async def get_config(
    config_manager: ConfigManager = Depends(get_config_manager),
):
    """
    Retrieve the current configuration settings.
    """
    try:
        config = config_manager.get_config()
        config_dict = {
            "version": config.version,
            "metadata": config.metadata,
            "values": config.values,
        }

        return CRUDResponse(
            status=200,
            message="Configuration retrieved successfully",
            result=cast(Dict[str, Any], config_dict),
        )
    except Exception as e:
        logger.exception("Error retrieving configuration")
        raise HTTPException(status_code=500, detail=f"Error retrieving configuration: {e}")


@router.patch(
    "/v1/config",
    response_model=CRUDResponse[ConfigResponse],
    summary="Update configuration",
    description="Update the configuration settings with new values.",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "example": {
                            "summary": "Update Configuration Example",
                            "value": {
                                "conversation_length": 150,
                                "detail_length": 50,
                                "hosting": "openrouter",
                                "model_name": "openai/gpt-4o-mini",
                                "auto_save_conversation": True,
                            },
                        }
                    }
                }
            }
        },
        "responses": {
            "200": {
                "description": "Configuration updated successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Configuration updated successfully",
                            "result": {
                                "version": "0.2.16",
                                "metadata": {
                                    "created_at": "2024-01-01T00:00:00",
                                    "last_modified": "2024-01-01T12:00:00",
                                    "description": "Local Operator configuration file",
                                },
                                "values": {
                                    "conversation_length": 150,
                                    "detail_length": 50,
                                    "max_learnings_history": 50,
                                    "hosting": "openrouter",
                                    "model_name": "openai/gpt-4o-mini",
                                    "auto_save_conversation": True,
                                },
                            },
                        }
                    }
                },
            }
        },
    },
)
async def update_config(
    config_update: ConfigUpdate,
    config_manager: ConfigManager = Depends(get_config_manager),
):
    """
    Update the configuration settings with new values.
    """
    try:
        # Filter out None values to only update provided fields
        updates = {k: v for k, v in config_update.model_dump().items() if v is not None}

        if not updates:
            raise HTTPException(status_code=400, detail="No valid update fields provided")

        # Update the configuration
        config_manager.update_config(updates)

        # Get the updated configuration
        config = config_manager.get_config()
        config_dict = {
            "version": config.version,
            "metadata": config.metadata,
            "values": config.values,
        }

        response = CRUDResponse(
            status=200,
            message="Configuration updated successfully",
            result=cast(Dict[str, Any], config_dict),
        )
        return JSONResponse(status_code=200, content=jsonable_encoder(response))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error updating configuration")
        raise HTTPException(status_code=500, detail=f"Error updating configuration: {e}")
