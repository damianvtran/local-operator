"""
FastAPI server implementation for Local Operator API.

Provides REST endpoints for interacting with the Local Operator agent
through HTTP requests instead of CLI.
"""

import logging
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from fastapi import Body, FastAPI, HTTPException
from fastapi import Path as FPath
from fastapi import Query
from pydantic import BaseModel, Field
from tiktoken import encoding_for_model

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.executor import LocalCodeExecutor
from local_operator.model import configure_model
from local_operator.operator import ConversationRole, Operator, OperatorType
from local_operator.prompts import create_system_prompt

logger = logging.getLogger("local_operator.server")


class HealthCheckResponse(BaseModel):
    """Response from health check endpoint.

    Attributes:
        status: HTTP status code
        message: Health check message
    """

    status: int
    message: str


class ChatOptions(BaseModel):
    """Options for controlling the chat generation.

    Attributes:
        temperature: Controls randomness in responses. Higher values like 0.8 make output more
            random, while lower values like 0.2 make it more focused and deterministic.
            Default: 0.8
        top_p: Controls cumulative probability of tokens to sample from. Higher values (0.95) keep
            more options, lower values (0.1) are more selective. Default: 0.9
        top_k: Limits tokens to sample from at each step. Lower values (10) are more selective,
            higher values (100) allow more variety. Default: 40
        max_tokens: Maximum tokens to generate. Model may generate fewer if response completes
            before reaching limit. Default: 4096
        stop: List of strings that will stop generation when encountered. Default: None
        frequency_penalty: Reduces repetition by lowering likelihood of repeated tokens.
            Range from -2.0 to 2.0. Default: 0.0
        presence_penalty: Increases diversity by lowering likelihood of prompt tokens.
            Range from -2.0 to 2.0. Default: 0.0
        seed: Random number seed for deterministic generation. Default: None
    """

    temperature: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None


class ChatMessage(BaseModel):
    """A single message in the chat conversation.

    Attributes:
        role: The role of who sent the message - "system", "user", or "assistant"
        content: The actual text content of the message
    """

    role: str
    content: str


class ChatRequest(BaseModel):
    """Request body for chat generation endpoint.

    Attributes:
        hosting: Name of the hosting service to use for generation
        model: Name of the model to use for generation
        prompt: The prompt to generate a response for
        stream: Whether to stream the response token by token. Default: False
        context: Optional list of previous messages for context
        options: Optional generation parameters to override defaults
    """

    hosting: str
    model: str
    prompt: str
    stream: bool = False
    context: Optional[List[ChatMessage]] = None
    options: Optional[ChatOptions] = None


class ChatStats(BaseModel):
    """Statistics about token usage for the chat request.

    Attributes:
        total_tokens: Total number of tokens used in prompt and completion
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
    """

    total_tokens: int
    prompt_tokens: int
    completion_tokens: int


class ChatResponse(BaseModel):
    """Response from chat generation endpoint.

    Attributes:
        response: The generated text response
        context: List of all messages including the new response
        stats: Token usage statistics
    """

    response: str
    context: List[ChatMessage]
    stats: ChatStats


class CRUDResponse(BaseModel):
    """
    Standard response schema for CRUD operations.
    Attributes:
        status: HTTP status code
        message: Outcome message of the operation
        result: The resulting data, which can be an object, paginated list, or empty.
    """

    status: int
    message: str
    result: Optional[Dict[str, Any]] = None


class Agent(BaseModel):
    """Representation of an Agent."""

    id: str = Field(..., description="Unique identifier for the agent")
    name: str = Field(..., description="Name of the agent")
    description: Optional[str] = Field(None, description="Description of the agent")


class AgentCreate(BaseModel):
    """Data required to create a new agent."""

    name: str = Field(..., description="Name of the agent")
    description: Optional[str] = Field(None, description="Description of the agent")


class AgentUpdate(BaseModel):
    """Data for updating an existing agent."""

    name: Optional[str] = Field(None, description="Updated name of the agent")
    description: Optional[str] = Field(None, description="Updated description of the agent")


class AgentListResult(BaseModel):
    """Paginated list result for agents."""

    total: int = Field(..., description="Total number of agents")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Number of agents per page")
    agents: List[Agent] = Field(..., description="List of agents")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize on startup by setting up the credential and config managers
    config_dir = Path.home() / ".local-operator"
    agents_dir = config_dir / "agents"
    app.state.credential_manager = CredentialManager(config_dir=config_dir)
    app.state.config_manager = ConfigManager(config_dir=config_dir)
    app.state.agent_registry = AgentRegistry(config_dir=agents_dir)
    yield
    # Clean up on shutdown
    app.state.credential_manager = None
    app.state.config_manager = None
    app.state.agent_registry = None


app = FastAPI(
    title="Local Operator API",
    description="REST API interface for Local Operator agent",
    version=version("local-operator"),
    lifespan=lifespan,
)


def create_operator(request_hosting: str, request_model: str) -> Operator:
    """Create a LocalCodeExecutor for a single chat request using the app state managers
    and the hosting/model provided in the request."""
    credential_manager = getattr(app.state, "credential_manager", None)
    config_manager = getattr(app.state, "config_manager", None)
    agent_registry = getattr(app.state, "agent_registry", None)
    if credential_manager is None or config_manager is None or agent_registry is None:
        raise HTTPException(status_code=500, detail="Server configuration not initialized")
    agent_registry = cast(AgentRegistry, agent_registry)

    if not request_hosting:
        raise ValueError("Hosting is not set")

    model_instance = configure_model(
        credential_manager=credential_manager,
        hosting=request_hosting,
        model=request_model,
    )

    if not model_instance:
        raise ValueError("No model instance configured")

    executor = LocalCodeExecutor(
        model=model_instance,
        max_conversation_history=100,
        detail_conversation_length=10,
        can_prompt_user=False,
    )

    return Operator(
        executor=executor,
        credential_manager=credential_manager,
        model_instance=executor.model,
        config_manager=config_manager,
        type=OperatorType.SERVER,
        agent_registry=agent_registry,
        current_agent=None,
        training_mode=False,
    )


@app.post(
    "/v1/chat",
    response_model=ChatResponse,
    summary="Process chat request",
    description="Accepts a prompt and optional context/configuration, returns the model response "
    "and conversation history.",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "example": {
                            "summary": "Example Request",
                            "value": {
                                "prompt": "Print 'Hello, world!'",
                                "hosting": "openai",
                                "model": "gpt-4o",
                                "context": [],
                                "options": {"temperature": 0.7, "top_p": 0.9},
                            },
                        }
                    }
                }
            }
        }
    },
)
async def chat_endpoint(request: ChatRequest):
    """
    Process a chat request and return the response with context.

    The endpoint accepts a JSON payload containing the prompt, hosting, model selection, and
    optional parameters.
    ---
    responses:
      200:
        description: Successful response containing the model output and conversation history.
      500:
        description: Internal Server Error
    """
    try:
        # Create a new executor for this request using the provided hosting and model
        operator = create_operator(request.hosting, request.model)

        if request.context and len(request.context) > 0:
            # Override the default system prompt with the provided context
            conversation_history = [
                {"role": msg.role, "content": msg.content} for msg in request.context
            ]
            operator.executor.initialize_conversation_history(conversation_history)
        else:
            operator.executor.initialize_conversation_history()

        # Configure model options if provided
        if request.options:
            temperature = request.options.temperature or operator.executor.model.temperature
            if temperature is not None:
                operator.model.temperature = temperature
            operator.model.top_p = request.options.top_p or operator.executor.model.top_p

        response_json = await operator.handle_user_input(request.prompt)
        if response_json is not None:
            response_content = response_json.response
        else:
            response_content = ""

        # Calculate token stats using tiktoken
        tokenizer = encoding_for_model(request.model)
        prompt_tokens = sum(
            len(tokenizer.encode(msg["content"])) for msg in operator.executor.conversation_history
        )
        completion_tokens = len(tokenizer.encode(response_content))
        total_tokens = prompt_tokens + completion_tokens

        return ChatResponse(
            response=response_content,
            context=[
                ChatMessage(role=msg["role"], content=msg["content"])
                for msg in operator.executor.conversation_history
            ],
            stats=ChatStats(
                total_tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
        )

    except Exception:
        logger.exception("Unexpected error while processing chat request")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post(
    "/v1/chat/agents/{agent_id}",
    response_model=ChatResponse,
    summary="Process chat request using a specific agent",
    description=(
        "Accepts a prompt and optional context/configuration, retrieves the specified "
        "agent from the registry, applies it to the operator and executor, and returns the "
        "model response and conversation history."
    ),
    openapi_extra={
        "parameters": [
            {
                "name": "agent_id",
                "in": "path",
                "description": "The ID of the agent to be used for the chat",
                "required": True,
                "schema": {"type": "string"},
                "example": "agent123",
            }
        ],
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "example": {
                            "summary": "Example Request with Agent",
                            "value": {
                                "prompt": "How do I implement a binary search in Python?",
                                "hosting": "openai",
                                "model": "gpt-4o",
                                "context": [],
                                "options": {"temperature": 0.7, "top_p": 0.9},
                            },
                        }
                    }
                }
            }
        },
    },
)
async def chat_with_agent(
    agent_id: str = FPath(
        ..., description="ID of the agent to use for the chat", example="agent123"
    ),
    request: ChatRequest = Body(...),
):
    """
    Process a chat request using a specific agent from the registry and return the response with
    context. The specified agent is applied to both the operator and executor.
    """
    try:
        # Retrieve the agent registry from app state
        agent_registry = getattr(app.state, "agent_registry", None)
        if agent_registry is None:
            raise HTTPException(status_code=500, detail="Agent registry not initialized")
        agent_registry = cast(AgentRegistry, agent_registry)

        # Retrieve the specific agent from the registry
        agent_obj = agent_registry.get_agent(agent_id)
        if not agent_obj:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Create a new executor for this request using the provided hosting and model
        operator = create_operator(request.hosting, request.model)

        # Apply the retrieved agent to the operator and executor
        operator.current_agent = agent_obj
        setattr(operator.executor, "current_agent", agent_obj)

        if request.context and len(request.context) > 0:
            # Override the default system prompt with the provided context
            conversation_history = [
                {"role": msg.role, "content": msg.content} for msg in request.context
            ]
            operator.executor.initialize_conversation_history(conversation_history)
        else:
            operator.executor.initialize_conversation_history()

        # Configure model options if provided
        if request.options:
            temperature = request.options.temperature or operator.executor.model.temperature
            if temperature is not None:
                operator.model.temperature = temperature
            operator.model.top_p = request.options.top_p or operator.executor.model.top_p

        response_json = await operator.handle_user_input(request.prompt)
        response_content = response_json.response if response_json is not None else ""

        # Calculate token stats using tiktoken
        tokenizer = encoding_for_model(request.model)
        prompt_tokens = sum(
            len(tokenizer.encode(msg["content"])) for msg in operator.executor.conversation_history
        )
        completion_tokens = len(tokenizer.encode(response_content))
        total_tokens = prompt_tokens + completion_tokens

        return ChatResponse(
            response=response_content,
            context=[
                ChatMessage(role=msg["role"], content=msg["content"])
                for msg in operator.executor.conversation_history
            ],
            stats=ChatStats(
                total_tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
        )

    except Exception:
        logger.exception("Unexpected error while processing chat request with agent")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get(
    "/v1/agents",
    response_model=CRUDResponse,
    summary="List agents",
    description="Retrieve a paginated list of agents with their details.",
    tags=["Agents"],
    openapi_extra={
        "parameters": [
            {
                "name": "page",
                "in": "query",
                "description": "Page number for pagination",
                "required": False,
                "schema": {"type": "integer", "minimum": 1, "default": 1},
                "example": 1,
            },
            {
                "name": "per_page",
                "in": "query",
                "description": "Number of agents per page",
                "required": False,
                "schema": {"type": "integer", "minimum": 1, "default": 10},
                "example": 10,
            },
        ],
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
                                        "id": "agent1",
                                        "name": "Agent One",
                                        "description": "First test agent",
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
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, description="Number of agents per page"),
):
    """
    Retrieve a paginated list of agents.
    """
    agent_registry = getattr(app.state, "agent_registry", None)
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


@app.post(
    "/v1/agents",
    response_model=CRUDResponse,
    summary="Create a new agent",
    description="Create a new agent with the provided details.",
    tags=["Agents"],
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "example": {
                            "summary": "Create Agent Example",
                            "value": {"name": "New Agent", "description": "A newly created agent"},
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
                                "description": "A newly created agent",
                            },
                        }
                    }
                },
            }
        },
    },
)
async def create_agent(agent: AgentCreate = Body(...)):
    """
    Create a new agent.
    """
    agent_registry = getattr(app.state, "agent_registry", None)
    if agent_registry is None:
        raise HTTPException(status_code=500, detail="Agent registry not initialized")
    agent_registry = cast(AgentRegistry, agent_registry)

    try:
        new_agent = agent_registry.create_agent(
            agent_edit_metadata=cast(AgentEditFields, agent.dict())
        )
    except Exception as e:
        logger.exception("Error creating agent")
        raise HTTPException(status_code=400, detail=f"Failed to create agent: {e}")

    new_agent_serialized = new_agent.dict() if hasattr(new_agent, "dict") else new_agent

    return CRUDResponse(
        status=201,
        message="Agent created successfully",
        result=cast(Dict[str, Any], new_agent_serialized),
    )


@app.get(
    "/v1/agents/{agent_id}",
    response_model=CRUDResponse,
    summary="Retrieve an agent",
    description="Retrieve details for an agent by its ID.",
    tags=["Agents"],
    openapi_extra={
        "parameters": [
            {
                "name": "agent_id",
                "in": "path",
                "description": "The ID of the agent to retrieve",
                "required": True,
                "schema": {"type": "string"},
                "example": "agent123",
            }
        ],
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
                                "name": "New Agent",
                                "description": "A newly created agent",
                            },
                        }
                    }
                },
            }
        },
    },
)
async def get_agent(
    agent_id: str = FPath(..., description="ID of the agent to retrieve", example="agent123")
):
    """
    Retrieve an agent by ID.
    """
    agent_registry = getattr(app.state, "agent_registry", None)
    if agent_registry is None:
        raise HTTPException(status_code=500, detail="Agent registry not initialized")
    agent_registry = cast(AgentRegistry, agent_registry)

    try:
        agent_obj = agent_registry.get_agent(agent_id)
    except Exception as e:
        logger.exception("Error retrieving agent")
        raise HTTPException(status_code=500, detail=f"Error retrieving agent: {e}")

    if not agent_obj:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent_serialized = agent_obj.dict() if hasattr(agent_obj, "dict") else agent_obj

    return CRUDResponse(
        status=200,
        message="Agent retrieved successfully",
        result=cast(Dict[str, Any], agent_serialized),
    )


@app.patch(
    "/v1/agents/{agent_id}",
    response_model=CRUDResponse,
    summary="Update an agent",
    description="Update an existing agent with new details. Only provided fields will be updated.",
    tags=["Agents"],
    openapi_extra={
        "parameters": [
            {
                "name": "agent_id",
                "in": "path",
                "description": "The ID of the agent to update",
                "required": True,
                "schema": {"type": "string"},
                "example": "agent123",
            }
        ],
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "example": {
                            "summary": "Update Agent Example",
                            "value": {
                                "name": "Updated Agent Name",
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
                                "description": "Updated description",
                            },
                        }
                    }
                },
            }
        },
    },
)
async def update_agent(
    agent_id: str = FPath(..., description="ID of the agent to update", example="agent123"),
    agent_data: AgentUpdate = Body(...),
):
    """
    Update an existing agent.
    """
    agent_registry = getattr(app.state, "agent_registry", None)
    if agent_registry is None:
        raise HTTPException(status_code=500, detail="Agent registry not initialized")
    agent_registry = cast(AgentRegistry, agent_registry)

    try:
        updated_agent = agent_registry.update_agent(
            agent_id, cast(AgentEditFields, agent_data.model_dump(exclude_unset=True))
        )
    except Exception as e:
        logger.exception("Error updating agent")
        raise HTTPException(status_code=400, detail=f"Failed to update agent: {e}")

    if not updated_agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    updated_agent_serialized = (
        updated_agent.dict() if hasattr(updated_agent, "dict") else updated_agent
    )

    return CRUDResponse(
        status=200,
        message="Agent updated successfully",
        result=cast(Dict[str, Any], updated_agent_serialized),
    )


@app.delete(
    "/v1/agents/{agent_id}",
    response_model=CRUDResponse,
    summary="Delete an agent",
    description="Delete an existing agent by its ID.",
    tags=["Agents"],
    openapi_extra={
        "parameters": [
            {
                "name": "agent_id",
                "in": "path",
                "description": "The ID of the agent to delete",
                "required": True,
                "schema": {"type": "string"},
                "example": "agent123",
            }
        ],
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
    agent_id: str = FPath(..., description="ID of the agent to delete", example="agent123")
):
    """
    Delete an existing agent.
    """
    agent_registry = getattr(app.state, "agent_registry", None)
    if agent_registry is None:
        raise HTTPException(status_code=500, detail="Agent registry not initialized")
    agent_registry = cast(AgentRegistry, agent_registry)

    try:
        success = agent_registry.delete_agent(agent_id)
    except Exception as e:
        logger.exception("Error deleting agent")
        raise HTTPException(status_code=500, detail=f"Error deleting agent: {e}")

    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")

    return CRUDResponse(
        status=200,
        message="Agent deleted successfully",
        result={},
    )


@app.get(
    "/health",
    summary="Health Check",
    description="Returns the health status of the API server.",
)
async def health_check():
    """
    Health check endpoint.

    Returns:
        A JSON object with a "status" key indicating operational status.
    """
    return HealthCheckResponse(status=200, message="ok")
