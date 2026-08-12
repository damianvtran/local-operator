"""
Configuration management endpoints for the Local Operator API.

This module contains the FastAPI route handlers for configuration-related endpoints.
"""

import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from local_operator.config import ConfigManager
from local_operator.paths import config_dir
from local_operator.server.dependencies import get_config_manager
from local_operator.server.models.schemas import (
    ConfigResponse,
    ConfigUpdate,
    CRUDResponse,
    SystemPromptResponse,
    SystemPromptUpdate,
)

router = APIRouter(tags=["Configuration"])
logger = logging.getLogger("local_operator.server.routes.config")


def system_prompt_file() -> Path:
    """The operator's custom-instructions file, resolved per REQUEST.

    A function and not a module constant, for the reason :func:`config_dir`
    gives for its own lookup: a constant freezes whatever the first importer
    saw. Here that also broke the feature's central promise — this endpoint,
    the desktop Settings box and the session loader are supposed to be one
    file, and a hardcoded ``~/.local-operator`` diverges from the loader the
    moment ``LOCAL_OPERATOR_CONFIG_DIR`` is set (design round 27).
    """
    return config_dir() / "system_prompt.md"


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
                                    "detail_length": 15,
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
            result=config_dict,
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
) -> JSONResponse:
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
            result=config_dict,
        )
        return JSONResponse(status_code=200, content=jsonable_encoder(response))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error updating configuration")
        raise HTTPException(status_code=500, detail=f"Error updating configuration: {e}")


@router.get(
    "/v1/config/system-prompt",
    response_model=CRUDResponse[SystemPromptResponse],
    summary="Get system prompt",
    description="Retrieve the current system prompt content.",
    openapi_extra={
        "responses": {
            "200": {
                "description": "System prompt retrieved successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "System prompt retrieved successfully",
                            "result": {
                                "content": "You are Local Operator, an AI assistant...",
                                "last_modified": "2024-01-01T12:00:00",
                            },
                        }
                    }
                },
            },
            "204": {
                "description": "System prompt file does not exist",
                "content": {"application/json": {}},
            },
            "404": {
                "description": "System prompt file not found",
                "content": {
                    "application/json": {"example": {"detail": "System prompt file not found"}}
                },
            },
        },
    },
)
async def get_system_prompt() -> CRUDResponse[SystemPromptResponse] | Response:
    """
    Retrieve the current system prompt content.
    """
    try:
        if not system_prompt_file().exists():
            return Response(status_code=204)

        content = system_prompt_file().read_text(encoding="utf-8")
        last_modified = datetime.fromtimestamp(system_prompt_file().stat().st_mtime).isoformat()

        return CRUDResponse(
            status=200,
            message="System prompt retrieved successfully",
            result=SystemPromptResponse(content=content, last_modified=last_modified),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error retrieving system prompt")
        raise HTTPException(status_code=500, detail=f"Error retrieving system prompt: {e}")


@router.patch(
    "/v1/config/system-prompt",
    response_model=CRUDResponse[SystemPromptResponse],
    summary="Update system prompt",
    description="Update the system prompt content.",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "example": {
                            "summary": "Update System Prompt Example",
                            "value": {
                                "content": "You are Local Operator, an AI assistant...",
                            },
                        }
                    }
                }
            }
        },
        "responses": {
            "200": {
                "description": "System prompt updated successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "System prompt updated successfully",
                            "result": {
                                "content": "You are Local Operator, an AI assistant...",
                                "last_modified": "2024-01-01T12:00:00",
                            },
                        }
                    }
                },
            },
        },
    },
)
async def update_system_prompt(system_prompt_update: SystemPromptUpdate) -> JSONResponse:
    """
    Update the system prompt content.
    """
    try:
        # Create directory if it doesn't exist
        system_prompt_file().parent.mkdir(parents=True, exist_ok=True)

        # Write the updated content to the file
        system_prompt_file().write_text(system_prompt_update.content, encoding="utf-8")

        # Get the updated timestamp
        last_modified = datetime.fromtimestamp(system_prompt_file().stat().st_mtime).isoformat()

        response = CRUDResponse(
            status=200,
            message="System prompt updated successfully",
            result=SystemPromptResponse(
                content=system_prompt_update.content, last_modified=last_modified
            ),
        )

        return JSONResponse(status_code=200, content=jsonable_encoder(response))
    except Exception as e:
        logger.exception("Error updating system prompt")
        raise HTTPException(status_code=500, detail=f"Error updating system prompt: {e}")
