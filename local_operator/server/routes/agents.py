"""
Agent management endpoints for the Local Operator API.

This module contains the FastAPI route handlers for agent-related endpoints.
"""

import logging
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path as FilePath
from typing import Any, Dict, cast

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Path,
    Query,
    UploadFile,
)
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.clients.radient import RadientClient
from local_operator.credentials import CredentialManager
from local_operator.env import EnvConfig, get_env_config
from local_operator.server.dependencies import (
    get_agent_registry,
    get_credential_manager,
)
from local_operator.server.models.schemas import (
    Agent,
    AgentCreate,
    AgentExecutionHistoryResult,
    AgentGetConversationResult,
    AgentListResult,
    AgentUpdate,
    CRUDResponse,
    ExecutionVariable,
    ExecutionVariablesResponse,
    ExecutionVariableUpdate,
)
from local_operator.types import AgentState

router = APIRouter(tags=["Agents"])
logger = logging.getLogger("local_operator.server.routes.agents")


@router.get(
    "/v1/agents",
    response_model=CRUDResponse[AgentListResult],
    summary="List agents",
    description="Retrieve a paginated list of agents with their details. Optionally filter "
    "by agent name and sort by various fields.",
    openapi_extra={
        "responses": {
            "200": {
                "description": "Agents list retrieved successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Agents retrieved successfully",
                            "result": {
                                "total": 20,
                                "page": 1,
                                "per_page": 10,
                                "agents": [
                                    {
                                        "id": "agent123",
                                        "name": "Example Agent",
                                        "created_date": "2024-01-01T00:00:00",
                                        "version": "0.2.16",
                                        "security_prompt": "Example security prompt",
                                        "hosting": "openrouter",
                                        "model": "openai/gpt-4o-mini",
                                        "description": "An example agent",
                                        "last_message": "Hello, how can I help?",
                                        "last_message_datetime": "2024-01-01T12:00:00",
                                        "temperature": 0.7,
                                        "top_p": 1.0,
                                        "top_k": 20,
                                        "max_tokens": 2048,
                                        "stop": None,
                                        "frequency_penalty": 0.0,
                                        "presence_penalty": 0.0,
                                        "seed": None,
                                    }
                                ],
                            },
                        }
                    }
                },
            }
        },
    },
)
async def list_agents(
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, description="Number of agents per page"),
    name: str = Query(None, description="Filter agents by name (case-insensitive)"),
    sort: str = Query(
        "last_message_datetime",
        description="Sort field (name, created_date, last_message_datetime)",
    ),
    direction: str = Query("desc", description="Sort direction (asc, desc)"),
):
    """
    Retrieve a paginated list of agents.

    Optionally filter the list by agent name using the 'name' query parameter.
    The filter is case-insensitive and matches agents whose names contain the provided string.

    Supports sorting by name, created_date, or last_message_datetime in ascending or
    descending order.
    Default sort is by last_message_datetime in descending order.
    """
    try:
        agents_list = agent_registry.list_agents()

        # Filter by name if provided
        if name:
            agents_list = [agent for agent in agents_list if name.lower() in agent.name.lower()]

        # Validate sort field
        valid_sort_fields = ["name", "created_date", "last_message_datetime"]
        if sort not in valid_sort_fields:
            sort = "last_message_datetime"

        # Validate direction
        is_ascending = direction.lower() == "asc"

        # Sort the agents list
        if sort == "name":
            agents_list.sort(key=lambda agent: agent.name.lower(), reverse=not is_ascending)
        elif sort == "created_date":
            # Sort directly using the datetime object, fallback to min datetime
            agents_list.sort(
                key=lambda agent: (
                    agent.created_date
                    if isinstance(agent.created_date, datetime)
                    else datetime.min.replace(tzinfo=timezone.utc)
                ),
                reverse=not is_ascending,
            )
        else:  # last_message_datetime (default)
            # Sort directly using the datetime object, fallback to created_date, then min datetime
            def get_sort_key(agent):
                # Prefer last_message_datetime if it's a valid datetime
                last_msg_dt = getattr(agent, "last_message_datetime", None)
                if isinstance(last_msg_dt, datetime):
                    # Ensure timezone awareness for comparison
                    return (
                        last_msg_dt
                        if last_msg_dt.tzinfo
                        else last_msg_dt.replace(tzinfo=timezone.utc)
                    )

                # Fallback to created_date if it's a valid datetime
                created_dt = getattr(agent, "created_date", None)
                if isinstance(created_dt, datetime):
                    # Ensure timezone awareness for comparison
                    return (
                        created_dt if created_dt.tzinfo else created_dt.replace(tzinfo=timezone.utc)
                    )

                # Absolute fallback if neither date is valid
                return datetime.min.replace(tzinfo=timezone.utc)

            agents_list.sort(key=get_sort_key, reverse=not is_ascending)

    except Exception as e:
        logger.exception("Error retrieving agents")
        raise HTTPException(status_code=500, detail=f"Error retrieving agents: {e}")

    total = len(agents_list)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_agents = agents_list[start_idx:end_idx]

    # Explicitly construct Agent objects from AgentData fields for the response
    agents_for_response = [Agent.model_validate(agent.model_dump()) for agent in paginated_agents]

    result = AgentListResult(
        total=total,
        page=page,
        per_page=per_page,
        agents=agents_for_response,  # Pass the list of Agent objects
    )

    return CRUDResponse(
        status=200,
        message="Agents retrieved successfully",
        result=result.model_dump(),
    )


@router.post(
    "/v1/agents",
    response_model=CRUDResponse[Agent],
    summary="Create a new agent",
    description="Create a new agent with the provided details.",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "example": {
                            "summary": "Create Agent Example",
                            "value": {
                                "name": "New Agent",
                                "security_prompt": "Example security prompt",
                                "hosting": "openrouter",
                                "model": "openai/gpt-4o-mini",
                                "description": "A helpful assistant",
                            },
                        }
                    }
                }
            }
        },
        "responses": {
            "201": {
                "description": "Agent created successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 201,
                            "message": "Agent created successfully",
                            "result": {
                                "id": "agent123",
                                "name": "New Agent",
                                "created_date": "2024-01-01T00:00:00",
                                "version": "0.2.16",
                                "security_prompt": "Example security prompt",
                                "hosting": "openrouter",
                                "model": "openai/gpt-4o-mini",
                                "description": "A helpful assistant",
                                "last_message": "",
                                "last_message_datetime": "2024-01-01T00:00:00",
                            },
                        }
                    }
                },
            }
        },
    },
)
async def create_agent(
    agent: AgentCreate,
    agent_registry: AgentRegistry = Depends(get_agent_registry),
):
    """
    Create a new agent.
    """
    try:
        agent_edit_metadata = AgentEditFields.model_validate(agent.model_dump(exclude_unset=True))
        new_agent = agent_registry.create_agent(agent_edit_metadata)
    except ValidationError as e:
        logger.exception("Validation error creating agent")
        raise HTTPException(status_code=422, detail=f"Validation error: {e}")
    except Exception as e:
        logger.error(f"Error type: {type(e).__name__}")
        logger.exception("Error creating agent")
        raise HTTPException(status_code=400, detail=f"Failed to create agent: {e}")

    new_agent_serialized = new_agent.model_dump()

    response = CRUDResponse(
        status=201,
        message="Agent created successfully",
        result=cast(Dict[str, Any], new_agent_serialized),
    )
    return JSONResponse(status_code=201, content=jsonable_encoder(response))


@router.get(
    "/v1/agents/{agent_id}",
    response_model=CRUDResponse[Agent],
    summary="Retrieve an agent",
    description="Retrieve details for an agent by its ID.",
    openapi_extra={
        "responses": {
            "200": {
                "description": "Agent retrieved successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Agent retrieved successfully",
                            "result": {
                                "id": "agent123",
                                "name": "Example Agent",
                                "created_date": "2024-01-01T00:00:00",
                                "version": "0.2.16",
                                "security_prompt": "Example security prompt",
                                "hosting": "openrouter",
                                "model": "openai/gpt-4o-mini",
                                "description": "An example agent",
                                "last_message": "Hello, how can I help?",
                                "last_message_datetime": "2024-01-01T12:00:00",
                            },
                        }
                    }
                },
            }
        },
    },
)
async def get_agent(
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    agent_id: str = Path(..., description="ID of the agent to retrieve", examples=["agent123"]),
):
    """
    Retrieve an agent by ID.
    """
    try:
        agent_obj = agent_registry.get_agent(agent_id)
    except KeyError as e:
        logger.exception("Agent not found")
        raise HTTPException(status_code=404, detail=f"Agent not found: {e}")
    except Exception as e:
        logger.exception("Error retrieving agent")
        raise HTTPException(status_code=500, detail=f"Error retrieving agent: {e}")

    if not agent_obj:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent_serialized = agent_obj.model_dump()

    return CRUDResponse(
        status=200,
        message="Agent retrieved successfully",
        result=cast(Dict[str, Any], agent_serialized),
    )


@router.patch(
    "/v1/agents/{agent_id}",
    response_model=CRUDResponse[Agent],
    summary="Update an agent",
    description="Update an existing agent with new details. Only provided fields will be updated.",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "example": {
                            "summary": "Update Agent Example",
                            "value": {
                                "name": "Updated Agent Name",
                                "security_prompt": "Updated security prompt",
                                "hosting": "openrouter",
                                "model": "openai/gpt-4o-mini",
                                "description": "Updated description",
                            },
                        }
                    }
                }
            }
        },
        "responses": {
            "200": {
                "description": "Agent updated successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Agent updated successfully",
                            "result": {
                                "id": "agent123",
                                "name": "Updated Agent Name",
                                "created_date": "2024-01-01T00:00:00",
                                "version": "0.2.16",
                                "security_prompt": "Updated security prompt",
                                "hosting": "openrouter",
                                "model": "openai/gpt-4o-mini",
                                "description": "Updated description",
                                "last_message": "Hello, how can I help?",
                                "last_message_datetime": "2024-01-01T12:00:00",
                            },
                        }
                    }
                },
            }
        },
    },
)
async def update_agent(
    agent_data: AgentUpdate,
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    agent_id: str = Path(..., description="ID of the agent to update", examples=["agent123"]),
):
    """
    Update an existing agent.
    """
    try:
        agent_edit_data = AgentEditFields.model_validate(agent_data.model_dump(exclude_unset=True))
        updated_agent = agent_registry.update_agent(agent_id, agent_edit_data)
    except KeyError as e:
        logger.exception("Agent not found")
        raise HTTPException(status_code=404, detail=f"Agent not found: {e}")
    except Exception as e:
        logger.exception("Error updating agent")
        raise HTTPException(status_code=400, detail=f"Failed to update agent: {e}")

    if not updated_agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    updated_agent_serialized = updated_agent.model_dump()

    return CRUDResponse(
        status=200,
        message="Agent updated successfully",
        result=cast(Dict[str, Any], updated_agent_serialized),
    )


@router.post(
    "/v1/agents/{agent_id}/upload",
    response_model=CRUDResponse,
    summary="Upload (push) an agent to Radient Agent Hub",
    description=(
        "Upload (push) the agent with the given ID to the Radient agents marketplace. "
        "Requires RADIENT_API_KEY."
    ),
    openapi_extra={
        "responses": {
            "200": {
                "description": "Agent uploaded to Radient successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Agent uploaded to Radient successfully",
                            "result": {"agent_id": "radient-agent-id"},
                        }
                    }
                },
            },
            "400": {
                "description": "Bad request",
                "content": {
                    "application/json": {"example": {"detail": "Error uploading agent to Radient"}}
                },
            },
            "401": {
                "description": "Unauthorized",
                "content": {
                    "application/json": {"example": {"detail": "RADIENT_API_KEY is required"}}
                },
            },
        },
    },
)
async def upload_agent_to_radient(
    agent_id: str = Path(..., description="ID of the agent to upload", examples=["agent123"]),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    env_config: EnvConfig = Depends(get_env_config),
    credential_manager: CredentialManager = Depends(get_credential_manager),
):
    """
    Upload (push) the agent with the given ID to the Radient agents marketplace.
    Requires RADIENT_API_KEY.
    """
    try:
        # Get config and credentials
        api_key = credential_manager.get_credential("RADIENT_API_KEY")
        if not api_key:
            raise HTTPException(status_code=401, detail="RADIENT_API_KEY is required")
        base_url = env_config.radient_api_base_url
        radient_client = RadientClient(api_key=api_key, base_url=base_url)

        # Get agent and export as zip
        try:
            agent = agent_registry.get_agent(agent_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
        zip_path, _ = agent_registry.export_agent(agent.id)

        # Upload to Radient
        try:
            agent_registry.upload_agent_to_radient(radient_client, agent_id, zip_path)
        except Exception as e:
            logger.exception("Error uploading agent to Radient")
            raise HTTPException(status_code=400, detail=f"Error uploading agent to Radient: {e}")

        return CRUDResponse(
            status=200,
            message="Agent uploaded to Radient successfully",
            result={"agent_id": agent_id},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error uploading agent to Radient")
        # If the error is about missing API key, return a clear message
        if "RADIENT_API_KEY" in str(e) or "credential" in str(e):
            raise HTTPException(status_code=401, detail="RADIENT_API_KEY is required")
        raise HTTPException(status_code=400, detail=f"Error uploading agent to Radient: {e}")


@router.get(
    "/v1/agents/{agent_id}/download",
    response_model=CRUDResponse[Agent],
    summary="Download (pull) an agent from Radient Agent Hub",
    description="Download (pull) an agent from the Radient agents marketplace by agent ID.",
    openapi_extra={
        "responses": {
            "200": {
                "description": "Agent downloaded from Radient successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Agent downloaded from Radient successfully",
                            "result": {
                                "id": "imported-agent-123",
                                "name": "Imported Agent",
                                "created_date": "2024-01-01T00:00:00",
                                "version": "0.2.16",
                                "security_prompt": "Example security prompt",
                                "hosting": "openrouter",
                                "model": "openai/gpt-4o-mini",
                                "description": "An imported agent",
                                "last_message": "",
                                "last_message_datetime": "2024-01-01T00:00:00",
                            },
                        }
                    }
                },
            },
            "400": {
                "description": "Bad request",
                "content": {
                    "application/json": {
                        "example": {"detail": "Error downloading agent from Radient"}
                    }
                },
            },
        },
    },
)
async def download_agent_from_radient(
    agent_id: str = Path(
        ..., description="ID of the agent to download from Radient", examples=["radient-agent-id"]
    ),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    env_config: EnvConfig = Depends(get_env_config),
):
    """
    Download (pull) an agent from the Radient agents marketplace by agent ID.
    """
    try:
        base_url = env_config.radient_api_base_url
        # API key not required for download
        radient_client = RadientClient(api_key=None, base_url=base_url)

        # Download from Radient
        try:
            imported_agent = agent_registry.download_agent_from_radient(radient_client, agent_id)
        except Exception as e:
            logger.exception("Error downloading agent from Radient")
            raise HTTPException(
                status_code=400, detail=f"Error downloading agent from Radient: {e}"
            )

        agent_serialized = imported_agent.model_dump()
        return CRUDResponse(
            status=200,
            message="Agent downloaded from Radient successfully",
            result=cast(Dict[str, Any], agent_serialized),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error downloading agent from Radient")
        raise HTTPException(status_code=400, detail=f"Error downloading agent from Radient: {e}")


@router.delete(
    "/v1/agents/{agent_id}",
    response_model=CRUDResponse,
    summary="Delete an agent",
    description="Delete an existing agent by its ID.",
    openapi_extra={
        "responses": {
            "200": {
                "description": "Agent deleted successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Agent deleted successfully",
                            "result": {},
                        }
                    }
                },
            }
        },
    },
)
async def delete_agent(
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    agent_id: str = Path(..., description="ID of the agent to delete", examples=["agent123"]),
):
    """
    Delete an existing agent.
    """
    try:
        agent_registry.delete_agent(agent_id)
    except KeyError as e:
        logger.exception("Agent not found")
        raise HTTPException(status_code=404, detail=f"Agent not found: {e}")
    except Exception as e:
        logger.exception("Error deleting agent")
        raise HTTPException(status_code=500, detail=f"Error deleting agent: {e}")

    return CRUDResponse(
        status=200,
        message="Agent deleted successfully",
        result={},
    )


@router.get(
    "/v1/agents/{agent_id}/conversation",
    response_model=CRUDResponse[AgentGetConversationResult],
    summary="Get agent conversation history",
    description="Retrieve the conversation history for a specific agent.",
    openapi_extra={
        "responses": {
            "200": {
                "description": "Agent conversation retrieved successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Agent conversation retrieved successfully",
                            "result": {
                                "agent_id": "agent123",
                                "last_message_datetime": "2023-01-01T12:00:00",
                                "first_message_datetime": "2023-01-01T11:00:00",
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": "You are a helpful assistant",
                                        "should_summarize": False,
                                        "summarized": False,
                                        "timestamp": "2023-01-01T11:00:00",
                                    },
                                    {
                                        "role": "user",
                                        "content": "Hello, how are you?",
                                        "should_summarize": True,
                                        "summarized": False,
                                        "timestamp": "2023-01-01T11:00:00",
                                    },
                                ],
                                "page": 1,
                                "per_page": 10,
                                "total": 2,
                                "count": 2,
                            },
                        }
                    }
                },
            }
        }
    },
)
async def get_agent_conversation(
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    agent_id: str = Path(
        ..., description="ID of the agent to get conversation for", examples=["agent123"]
    ),
    page: int = Query(1, ge=1, description="Page number to retrieve"),
    per_page: int = Query(10, ge=1, le=100, description="Number of messages per page"),
):
    """
    Retrieve the conversation history for a specific agent.

    Args:
        agent_registry: The agent registry dependency
        agent_id: The unique identifier of the agent
        page: The page number to retrieve (starts at 1)
        per_page: The number of messages per page (between 1 and 100)

    Returns:
        AgentGetConversationResult: The conversation history for the agent

    Raises:
        HTTPException: If the agent registry is not initialized or the agent is not found
    """
    try:
        conversation_history = agent_registry.get_agent_conversation_history(agent_id)
        total_messages = len(conversation_history)

        # Set default datetime values in case the conversation is empty
        first_message_datetime = datetime.now()
        last_message_datetime = datetime.now()

        if conversation_history:
            # Find the first and last message timestamps
            # This assumes ConversationRecord has a timestamp attribute
            # If not, we'll use the current time as a fallback
            try:
                first_message_datetime = min(
                    msg.timestamp
                    for msg in conversation_history
                    if hasattr(msg, "timestamp") and msg.timestamp is not None
                )

                last_message_datetime = max(
                    msg.timestamp
                    for msg in conversation_history
                    if hasattr(msg, "timestamp") and msg.timestamp is not None
                )
            except (AttributeError, ValueError):
                # If timestamps aren't available, use current time
                pass

        # Apply pagination
        start_idx = (page - 1) * per_page
        end_idx = min(start_idx + per_page, total_messages)

        # Check if page is out of bounds
        if start_idx >= total_messages and total_messages > 0:
            raise HTTPException(
                status_code=400,
                detail=f"Page {page} is out of bounds. "
                f"Total pages: {(total_messages + per_page - 1) // per_page}",
            )

        # Pages move backward in history, so we start from the end of the array and
        # move backward while maintaining the same order of messages
        paginated_messages = (
            conversation_history[-end_idx : -start_idx or None] if conversation_history else []
        )

        result = AgentGetConversationResult(
            agent_id=agent_id,
            first_message_datetime=first_message_datetime,
            last_message_datetime=last_message_datetime,
            messages=paginated_messages,
            page=page,
            per_page=per_page,
            total=total_messages,
            count=len(paginated_messages),
        )

        return CRUDResponse(
            status=200,
            message="Agent conversation retrieved successfully",
            result=result.model_dump(),
        )
    except KeyError:
        logger.exception(f"Agent with ID {agent_id} not found")
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error retrieving agent conversation")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving agent conversation: {str(e)}"
        )


@router.delete(
    "/v1/agents/{agent_id}/conversation",
    response_model=CRUDResponse,
    summary="Clear agent conversation",
    description="Clear the conversation history for a specific agent.",
    openapi_extra={
        "responses": {
            "200": {
                "description": "Agent conversation cleared successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Agent conversation cleared successfully",
                            "result": {},
                        }
                    }
                },
            },
            "404": {
                "description": "Agent not found",
                "content": {
                    "application/json": {"example": {"detail": "Agent with ID agent123 not found"}}
                },
            },
            "500": {
                "description": "Internal server error",
                "content": {
                    "application/json": {"example": {"detail": "Error clearing agent conversation"}}
                },
            },
        },
    },
)
async def clear_agent_conversation(
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    agent_id: str = Path(
        ..., description="ID of the agent to clear conversation for", examples=["agent123"]
    ),
):
    """
    Clear the conversation history for a specific agent.

    Args:
        agent_registry: The agent registry dependency
        agent_id: The unique identifier of the agent

    Returns:
        CRUDResponse: A response indicating success or failure

    Raises:
        HTTPException: If the agent registry is not initialized or the agent is not found
    """
    try:
        # Get the agent to verify it exists
        agent = agent_registry.get_agent(agent_id)

        # Get the current agent state
        agent_state = agent_registry.load_agent_state(agent_id)

        # Clear the conversation by saving an empty list
        agent_registry.save_agent_state(
            agent_id=agent_id,
            agent_state=AgentState(
                version=agent.version,
                conversation=[],
                execution_history=[],
                learnings=agent_state.learnings,
                schedules=agent_state.schedules,
                current_plan="",
                instruction_details=agent_state.instruction_details,
                agent_system_prompt=agent_state.agent_system_prompt,
            ),
        )
        agent_registry.save_agent_context(agent_id=agent_id, context={})

        return CRUDResponse(
            status=200,
            message="Agent conversation cleared successfully",
            result={},
        )
    except KeyError:
        logger.exception(f"Agent with ID {agent_id} not found")
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    except Exception as e:
        logger.exception("Error clearing agent conversation")
        raise HTTPException(status_code=500, detail=f"Error clearing agent conversation: {str(e)}")


@router.post(
    "/v1/agents/import",
    response_model=CRUDResponse[Agent],
    summary="Import an agent",
    description=(
        "Import an agent from a ZIP file containing agent state files with an agent.yml file."
    ),
    responses={
        201: {
            "description": "Agent imported successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": 201,
                        "message": "Agent imported successfully",
                        "result": {
                            "id": "imported-agent-123",
                            "name": "Imported Agent",
                            "created_date": "2024-01-01T00:00:00",
                            "version": "0.2.16",
                            "security_prompt": "Example security prompt",
                            "hosting": "openrouter",
                            "model": "openai/gpt-4o-mini",
                            "description": "An imported agent",
                            "last_message": "",
                            "last_message_datetime": "2024-01-01T00:00:00",
                        },
                    }
                }
            },
        },
        400: {
            "description": "Bad request",
            "content": {
                "application/json": {"example": {"detail": "Invalid ZIP file or missing agent.yml"}}
            },
        },
        500: {
            "description": "Internal server error",
            "content": {"application/json": {"example": {"detail": "Error importing agent"}}},
        },
    },
)
async def import_agent(
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    file: UploadFile = File(..., description="ZIP file containing agent state files"),
):
    """
    Import an agent from a ZIP file.

    The ZIP file should contain agent state files with an agent.yml file.
    A new ID will be assigned to the imported agent, and the current working directory
    will be reset to local-operator-home.

    Args:
        agent_registry: The agent registry dependency
        file: The uploaded ZIP file containing agent state files

    Returns:
        CRUDResponse: A response containing the imported agent details

    Raises:
        HTTPException: If there is an error importing the agent
    """
    # Create a temporary directory to save the uploaded file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = FilePath(temp_dir)
        zip_path = temp_dir_path / "agent.zip"

        # Save the uploaded file to the temporary directory
        with open(zip_path, "wb") as f:
            f.write(await file.read())

        # Use the AgentRegistry's import_agent method
        try:
            agent_obj = agent_registry.import_agent(zip_path)
            agent_serialized = agent_obj.model_dump()

            response = CRUDResponse(
                status=201,
                message="Agent imported successfully",
                result=cast(Dict[str, Any], agent_serialized),
            )
            return JSONResponse(status_code=201, content=jsonable_encoder(response))
        except ValueError as e:
            # Handle ValueError directly as 400 Bad Request
            error_msg = str(e)
            logger.exception(f"Invalid agent import data: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        except zipfile.BadZipFile:
            # Handle BadZipFile directly as 400 Bad Request
            logger.exception("Invalid ZIP file")
            raise HTTPException(status_code=400, detail="Invalid ZIP file")
        except Exception as e:
            # For other exceptions, check if they contain known error messages
            error_msg = str(e)
            logger.exception(f"Error importing agent: {error_msg}")

            # Check for specific error messages that should be 400 errors
            if "Missing agent.yml" in error_msg or "Invalid ZIP file" in error_msg:
                raise HTTPException(status_code=400, detail=error_msg)

            # Otherwise, return 500 Internal Server Error
            raise HTTPException(status_code=500, detail=f"Error importing agent: {error_msg}")


@router.get(
    "/v1/agents/{agent_id}/export",
    summary="Export an agent",
    description="Export an agent's state files as a ZIP file.",
    responses={
        200: {
            "description": "Agent exported successfully",
            "content": {"application/octet-stream": {}},
        },
        404: {
            "description": "Agent not found",
            "content": {
                "application/json": {"example": {"detail": "Agent with ID agent123 not found"}}
            },
        },
        500: {
            "description": "Internal server error",
            "content": {"application/json": {"example": {"detail": "Error exporting agent"}}},
        },
    },
)
async def export_agent(
    background_tasks: BackgroundTasks,
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    agent_id: str = Path(..., description="ID of the agent to export", examples=["agent123"]),
):
    """
    Export an agent's state files as a ZIP file.

    Args:
        background_tasks: FastAPI background tasks for cleanup
        agent_registry: The agent registry dependency
        agent_id: The unique identifier of the agent to export

    Returns:
        StreamingResponse: A streaming response containing the ZIP file

    Raises:
        HTTPException: If the agent is not found or there is an error exporting the agent
    """
    try:
        # Use the AgentRegistry's export_agent method
        zip_path, filename = agent_registry.export_agent(agent_id)

        # Ensure the file exists before returning it
        if not zip_path.exists():
            raise FileNotFoundError(f"Failed to create ZIP file at {zip_path}")

        # Add cleanup task to remove the temporary directory after the response is sent
        background_tasks.add_task(shutil.rmtree, zip_path.parent, ignore_errors=True)

        # Return the ZIP file as a streaming response
        return FileResponse(
            path=zip_path,
            filename=filename,
            media_type="application/octet-stream",
        )

    except KeyError:
        logger.exception(f"Agent with ID {agent_id} not found")
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    except Exception as e:
        logger.exception("Error exporting agent")
        raise HTTPException(status_code=500, detail=f"Error exporting agent: {str(e)}")


@router.get(
    "/v1/agents/{agent_id}/history",
    response_model=CRUDResponse[AgentExecutionHistoryResult],
    summary="Get agent execution history",
    description="Retrieve the execution history for a specific agent.",
    responses={
        200: {
            "description": "Agent execution history retrieved successfully",
            "model": CRUDResponse[AgentExecutionHistoryResult],
            "content": {
                "application/json": {
                    "example": {
                        "status": 200,
                        "message": "Agent execution history retrieved successfully",
                        "result": {
                            "agent_id": "agent123",
                            "history": [
                                {
                                    "code": "print('Hello, world!')",
                                    "stdout": "Hello, world!",
                                    "stderr": "",
                                    "logging": "",
                                    "message": "Code executed successfully",
                                    "formatted_print": "Hello, world!",
                                    "role": "system",
                                    "status": "success",
                                    "timestamp": "2024-01-01T12:00:00Z",
                                    "execution_type": "action",
                                    "action": "CODE",
                                    "task_classification": "data_science",
                                }
                            ],
                            "first_execution_datetime": "2024-01-01T12:00:00Z",
                            "last_execution_datetime": "2024-01-01T12:00:00Z",
                            "page": 1,
                            "per_page": 10,
                            "total": 1,
                            "count": 1,
                        },
                    }
                }
            },
        },
        400: {
            "description": "Bad request",
            "content": {
                "application/json": {
                    "example": {"detail": "Page 2 is out of bounds. Total pages: 1"}
                }
            },
        },
        404: {
            "description": "Agent not found",
            "content": {
                "application/json": {"example": {"detail": "Agent with ID agent123 not found"}}
            },
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {"detail": "Error retrieving agent execution history"}
                }
            },
        },
    },
)
async def get_agent_execution_history(
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    agent_id: str = Path(
        ..., description="ID of the agent to get execution history for", examples=["agent123"]
    ),
    page: int = Query(1, ge=1, description="Page number to retrieve"),
    per_page: int = Query(10, ge=1, le=100, description="Number of executions per page"),
):
    """
    Get the execution history for a specific agent.
    """
    try:
        execution_history = agent_registry.get_agent_execution_history(agent_id)
        total_executions = len(execution_history)

        # Default timestamps if no executions
        first_execution_datetime = datetime.now(timezone.utc)
        last_execution_datetime = datetime.now(timezone.utc)

        # Get actual timestamps if executions exist
        if execution_history:
            try:
                timestamps = [
                    execution.timestamp
                    for execution in execution_history
                    if hasattr(execution, "timestamp") and execution.timestamp is not None
                ]
                if timestamps:
                    try:
                        first_execution_datetime = min(timestamps)
                        last_execution_datetime = max(timestamps)
                    except TypeError:
                        # Handle offset-naive and offset-aware datetime comparison
                        def to_aware(dt):
                            if dt.tzinfo is None:
                                return dt.replace(tzinfo=timezone.utc)
                            return dt

                        try:
                            aware_timestamps = [to_aware(dt) for dt in timestamps]
                            first_execution_datetime = min(aware_timestamps)
                            last_execution_datetime = max(aware_timestamps)
                        except Exception:
                            logger.exception(
                                "Failed to normalize datetimes in agent execution history"
                            )
                            # Fallback to current time if normalization fails
                            first_execution_datetime = datetime.now(timezone.utc)
                            last_execution_datetime = datetime.now(timezone.utc)
                else:
                    # No valid timestamps, use current time
                    first_execution_datetime = datetime.now(timezone.utc)
                    last_execution_datetime = datetime.now(timezone.utc)
            except (AttributeError, ValueError, TypeError):
                logger.exception("Error processing execution timestamps")
                # If timestamps aren't available or error occurs, use current time
                first_execution_datetime = datetime.now(timezone.utc)
                last_execution_datetime = datetime.now(timezone.utc)

        # Apply pagination
        start_idx = (page - 1) * per_page
        end_idx = min(start_idx + per_page, total_executions)

        # Check if page is out of bounds
        if start_idx >= total_executions and total_executions > 0:
            raise HTTPException(
                status_code=400,
                detail=f"Page {page} is out of bounds. "
                f"Total pages: {(total_executions + per_page - 1) // per_page}",
            )

        # Pages move backward in history, so we start from the end of the array and
        # move backward while maintaining the same order of executions
        paginated_history = (
            execution_history[-end_idx : -start_idx or None] if execution_history else []
        )

        result = AgentExecutionHistoryResult(
            agent_id=agent_id,
            first_execution_datetime=first_execution_datetime,
            last_execution_datetime=last_execution_datetime,
            history=paginated_history,
            page=page,
            per_page=per_page,
            total=total_executions,
            count=len(paginated_history),
        )

        return CRUDResponse(
            status=200,
            message="Agent execution history retrieved successfully",
            result=result.model_dump(),
        )
    except KeyError:
        logger.exception(f"Agent with ID {agent_id} not found")
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error retrieving agent execution history")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving agent execution history: {str(e)}"
        )


@router.get(
    "/v1/agents/{agent_id}/system-prompt",
    response_model=CRUDResponse,
    summary="Get agent system prompt",
    description="Retrieve the system prompt for a specific agent.",
    responses={
        200: {
            "description": "Agent system prompt retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": 200,
                        "message": "Agent system prompt retrieved successfully",
                        "result": {"system_prompt": "You are a helpful assistant..."},
                    }
                }
            },
        },
        404: {
            "description": "Agent not found",
            "content": {
                "application/json": {"example": {"detail": "Agent with ID agent123 not found"}}
            },
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {"example": {"detail": "Error retrieving agent system prompt"}}
            },
        },
    },
)
async def get_agent_system_prompt(
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    agent_id: str = Path(..., description="ID of the agent", examples=["agent123"]),
):
    """
    Retrieve the system prompt for a specific agent.

    Args:
        agent_registry: The agent registry dependency
        agent_id: The unique identifier of the agent

    Returns:
        CRUDResponse: A response containing the agent's system prompt

    Raises:
        HTTPException: If the agent is not found or there is an error retrieving the system prompt
    """
    try:
        system_prompt = agent_registry.get_agent_system_prompt(agent_id)
        return CRUDResponse(
            status=200,
            message="Agent system prompt retrieved successfully",
            result={"system_prompt": system_prompt},
        )
    except KeyError:
        logger.exception(f"Agent with ID {agent_id} not found")
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    except FileNotFoundError as e:
        logger.exception(f"System prompt file not found for agent {agent_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except IOError as e:
        logger.exception(f"Error reading system prompt for agent {agent_id}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Error retrieving agent system prompt")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving agent system prompt: {str(e)}"
        )


@router.put(
    "/v1/agents/{agent_id}/system-prompt",
    response_model=CRUDResponse,
    summary="Update agent system prompt",
    description="Update the system prompt for a specific agent.",
    responses={
        200: {
            "description": "Agent system prompt updated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": 200,
                        "message": "Agent system prompt updated successfully",
                        "result": {},
                    }
                }
            },
        },
        404: {
            "description": "Agent not found",
            "content": {
                "application/json": {"example": {"detail": "Agent with ID agent123 not found"}}
            },
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {"example": {"detail": "Error updating agent system prompt"}}
            },
        },
    },
)
async def update_agent_system_prompt(
    system_prompt: Dict[str, str],
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    agent_id: str = Path(..., description="ID of the agent", examples=["agent123"]),
):
    """
    Update the system prompt for a specific agent.

    Args:
        system_prompt: A dictionary containing the system prompt text
        agent_registry: The agent registry dependency
        agent_id: The unique identifier of the agent

    Returns:
        CRUDResponse: A response indicating success or failure

    Raises:
        HTTPException: If the agent is not found or there is an error updating the system prompt
    """
    try:
        if "system_prompt" not in system_prompt:
            raise HTTPException(
                status_code=422, detail="Request body must contain 'system_prompt' field"
            )

        agent_registry.set_agent_system_prompt(agent_id, system_prompt["system_prompt"])
        return CRUDResponse(
            status=200,
            message="Agent system prompt updated successfully",
            result={},
        )
    except KeyError:
        logger.exception(f"Agent with ID {agent_id} not found")
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    except IOError as e:
        logger.exception(f"Error writing system prompt for agent {agent_id}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        if isinstance(e, HTTPException):
            raise

        logger.exception("Error updating agent system prompt")
        raise HTTPException(status_code=500, detail=f"Error updating agent system prompt: {str(e)}")


# Agent Execution Variables CRUD Endpoints
@router.get(
    "/v1/agents/{agent_id}/execution-variables",
    response_model=CRUDResponse[ExecutionVariablesResponse],
    summary="List agent execution variables",
    description="Retrieve all execution variables for a specific agent.",
)
async def list_agent_execution_variables(
    agent_id: str = Path(..., description="ID of the agent"),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
):
    try:
        variables = agent_registry.load_agent_context(agent_id)

        string_variables = [
            ExecutionVariable(key=k, value=str(v), type=type(v).__name__)
            for k, v in variables.items()
        ]

        return CRUDResponse(
            status=200,
            message="Execution variables retrieved successfully",
            result=ExecutionVariablesResponse(execution_variables=string_variables),
        )
    except KeyError:
        logger.warning(f"Agent not found when listing execution variables: {agent_id}")
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    except Exception as e:
        logger.exception(f"Error listing execution variables for agent {agent_id}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving execution variables: {str(e)}"
        )


@router.post(
    "/v1/agents/{agent_id}/execution-variables",
    response_model=CRUDResponse[ExecutionVariable],
    summary="Create an agent execution variable",
    description="Create a new execution variable for a specific agent.",
)
async def create_agent_execution_variable(
    variable_data: ExecutionVariable,
    agent_id: str = Path(..., description="ID of the agent"),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
):
    try:
        agent_registry.create_context_variable(agent_id, variable_data.key, variable_data.value)
        response_content = CRUDResponse(
            status=201,
            message="Execution variable created successfully",
            result=ExecutionVariable(
                key=variable_data.key,
                value=str(variable_data.value),  # Ensure value is string for response
                type=type(variable_data.value).__name__,
            ),
        )
        return JSONResponse(status_code=201, content=jsonable_encoder(response_content))
    except KeyError:
        logger.warning(f"Agent not found when creating execution variable: {agent_id}")
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    except ValueError as e:  # Handles case where variable key already exists
        logger.warning(
            f"Attempt to create existing execution variable '{variable_data.key}' "
            f"for agent {agent_id}: {str(e)}"
        )
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.exception(
            f"Error creating execution variable '{variable_data.key}' for agent {agent_id}"
        )
        raise HTTPException(status_code=500, detail=f"Error creating execution variable: {str(e)}")


@router.get(
    "/v1/agents/{agent_id}/execution-variables/{variable_key}",
    response_model=CRUDResponse[ExecutionVariable],
    summary="Get an agent execution variable",
    description="Retrieve a specific execution variable for an agent by its key.",
)
async def get_agent_execution_variable(
    agent_id: str = Path(..., description="ID of the agent"),
    variable_key: str = Path(..., description="Key of the execution variable"),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
):
    try:
        value = agent_registry.get_context_variable(agent_id, variable_key)
        return CRUDResponse(
            status=200,
            message="Execution variable retrieved successfully",
            result=ExecutionVariable(key=variable_key, value=value, type=type(value).__name__),
        )
    except KeyError as e:
        logger.warning(
            f"Agent '{agent_id}' or variable '{variable_key}' not found "
            "when getting execution variable"
        )
        # Distinguish between agent not found and variable not found for clarity
        if f"Agent with id {agent_id} not found" in str(e):
            raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Execution variable '{variable_key}' not found for agent {agent_id}",
            )
    except Exception as e:
        logger.exception(
            f"Error retrieving execution variable '{variable_key}' for agent {agent_id}"
        )
        raise HTTPException(
            status_code=500, detail=f"Error retrieving execution variable: {str(e)}"
        )


@router.patch(
    "/v1/agents/{agent_id}/execution-variables/{variable_key}",
    response_model=CRUDResponse[ExecutionVariable],
    summary="Update an agent execution variable",
    description="Update an existing execution variable for a specific agent.",
)
async def update_agent_execution_variable(
    variable_data: ExecutionVariableUpdate,  # Use ExecutionVariableUpdate
    agent_id: str = Path(..., description="ID of the agent"),
    variable_key: str = Path(..., description="Key of the execution variable to update"),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
):
    try:
        updated_context = agent_registry.update_context_variable(
            agent_id, variable_key, variable_data.value
        )
        updated_value = updated_context.get(variable_key)

        return CRUDResponse(
            status=200,
            message="Execution variable updated successfully",
            result=ExecutionVariable(
                key=variable_key,
                value=str(updated_value),  # Ensure value is string for the response model
                type=type(updated_value).__name__,
            ),
        )
    except KeyError as e:
        logger.warning(
            f"Agent '{agent_id}' or variable '{variable_key}' not found "
            "when updating execution variable"
        )
        if f"Agent with id {agent_id} not found" in str(e):
            raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Execution variable '{variable_key}' not found for agent {agent_id}",
            )
    except Exception as e:
        logger.exception(f"Error updating execution variable '{variable_key}' for agent {agent_id}")
        raise HTTPException(status_code=500, detail=f"Error updating execution variable: {str(e)}")


@router.delete(
    "/v1/agents/{agent_id}/execution-variables/{variable_key}",
    summary="Delete an agent execution variable",
    description="Delete an execution variable for a specific agent by its key.",
)
async def delete_agent_execution_variable(
    agent_id: str = Path(..., description="ID of the agent"),
    variable_key: str = Path(..., description="Key of the execution variable to delete"),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
):
    try:
        agent_registry.delete_context_variable(agent_id, variable_key)
        return CRUDResponse(
            status=200,
            message="Execution variable deleted successfully",
        )
    except KeyError as e:
        logger.warning(
            f"Agent '{agent_id}' or variable '{variable_key}' not found "
            "when deleting execution variable"
        )
        if f"Agent with id {agent_id} not found" in str(e):
            raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Execution variable '{variable_key}' not found for agent {agent_id}",
            )
    except Exception as e:
        logger.exception(f"Error deleting execution variable '{variable_key}' for agent {agent_id}")
        raise HTTPException(status_code=500, detail=f"Error deleting execution variable: {str(e)}")
