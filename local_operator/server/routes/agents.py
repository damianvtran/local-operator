"""
Agent management endpoints for the Local Operator API.

This module contains the FastAPI route handlers for agent-related endpoints.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, cast

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.server.dependencies import get_agent_registry
from local_operator.server.models.schemas import (
    Agent,
    AgentCreate,
    AgentExecutionHistoryResult,
    AgentGetConversationResult,
    AgentListResult,
    AgentUpdate,
    CRUDResponse,
)

router = APIRouter(tags=["Agents"])
logger = logging.getLogger("local_operator.server.routes.agents")


@router.get(
    "/v1/agents",
    response_model=CRUDResponse[AgentListResult],
    summary="List agents",
    description="Retrieve a paginated list of agents with their details. Optionally filter "
    "by agent name.",
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
):
    """
    Retrieve a paginated list of agents.

    Optionally filter the list by agent name using the 'name' query parameter.
    The filter is case-insensitive and matches agents whose names contain the provided string.
    """
    try:
        agents_list = agent_registry.list_agents()

        # Filter by name if provided
        if name:
            agents_list = [agent for agent in agents_list if name.lower() in agent.name.lower()]
    except Exception as e:
        logger.exception("Error retrieving agents")
        raise HTTPException(status_code=500, detail=f"Error retrieving agents: {e}")

    total = len(agents_list)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated = agents_list[start_idx:end_idx]
    agents_serialized = [
        agent.model_dump() if hasattr(agent, "model_dump") else agent for agent in paginated
    ]

    result = AgentListResult(
        total=total,
        page=page,
        per_page=per_page,
        agents=[Agent.model_validate(agent) for agent in agents_serialized],
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
        agent_registry.get_agent(agent_id)

        # Clear the conversation by saving an empty list
        agent_registry.save_agent_conversation(agent_id, [], [])
        agent_registry.save_agent_context(agent_id, {})

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
                first_execution_datetime = min(
                    execution.timestamp
                    for execution in execution_history
                    if hasattr(execution, "timestamp") and execution.timestamp is not None
                )

                last_execution_datetime = max(
                    execution.timestamp
                    for execution in execution_history
                    if hasattr(execution, "timestamp") and execution.timestamp is not None
                )
            except (AttributeError, ValueError):
                # If timestamps aren't available, use current time
                pass

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
