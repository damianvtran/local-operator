"""
Agent management endpoints for the Local Operator API.

This module contains the FastAPI route handlers for agent-related endpoints.
"""

import logging
from typing import Any, Dict, cast

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.server.dependencies import get_agent_registry
from local_operator.server.models.schemas import AgentCreate, AgentUpdate, CRUDResponse

router = APIRouter(tags=["Agents"])
logger = logging.getLogger("local_operator.server.routes.agents")


@router.get(
    "/v1/agents",
    response_model=CRUDResponse,
    summary="List agents",
    description="Retrieve a paginated list of agents with their details.",
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
                                        "created_date": "2024-01-01T00:00:00Z",
                                        "version": "0.2.16",
                                        "security_prompt": "Example security prompt",
                                        "hosting": "openrouter",
                                        "model": "openai/gpt-4o-mini",
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
):
    """
    Retrieve a paginated list of agents.
    """
    if agent_registry is None:
        raise HTTPException(status_code=500, detail="Agent registry not initialized")
    agent_registry = cast(AgentRegistry, agent_registry)

    try:
        agents_list = agent_registry.list_agents()
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

    return CRUDResponse(
        status=200,
        message="Agents retrieved successfully",
        result={
            "total": total,
            "page": page,
            "per_page": per_page,
            "agents": cast(Dict[str, Any], {"agents": agents_serialized}),
        },
    )


@router.post(
    "/v1/agents",
    response_model=CRUDResponse,
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
                                "created_date": "2024-01-01T00:00:00Z",
                                "version": "0.2.16",
                                "security_prompt": "Example security prompt",
                                "hosting": "openrouter",
                                "model": "openai/gpt-4o-mini",
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
    if agent_registry is None:
        raise HTTPException(status_code=500, detail="Agent registry not initialized")
    agent_registry = cast(AgentRegistry, agent_registry)

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
    response_model=CRUDResponse,
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
                                "created_date": "2024-01-01T00:00:00Z",
                                "version": "0.2.16",
                                "security_prompt": "Example security prompt",
                                "hosting": "openrouter",
                                "model": "openai/gpt-4o-mini",
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
    if agent_registry is None:
        raise HTTPException(status_code=500, detail="Agent registry not initialized")
    agent_registry = cast(AgentRegistry, agent_registry)

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
    response_model=None,
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
                                "created_date": "2024-01-01T00:00:00Z",
                                "version": "0.2.16",
                                "security_prompt": "Updated security prompt",
                                "hosting": "openrouter",
                                "model": "openai/gpt-4o-mini",
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
    if agent_registry is None:
        raise HTTPException(status_code=500, detail="Agent registry not initialized")
    agent_registry = cast(AgentRegistry, agent_registry)

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
    response_model=None,
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
    if agent_registry is None:
        raise HTTPException(status_code=500, detail="Agent registry not initialized")
    agent_registry = cast(AgentRegistry, agent_registry)

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
