"""
FastAPI server implementation for Local Operator API.

Provides REST endpoints for interacting with the Local Operator agent
through HTTP requests instead of CLI.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from fastapi import Body, FastAPI, HTTPException
from fastapi import Path as FPath
from fastapi import Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from tiktoken import encoding_for_model

from local_operator.admin import add_admin_tools
from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.executor import LocalCodeExecutor
from local_operator.model.configure import configure_model
from local_operator.operator import Operator, OperatorType
from local_operator.tools import ToolRegistry
from local_operator.types import ConversationRecord

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
    context: Optional[List[ConversationRecord]] = None
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
    context: List[ConversationRecord]
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
    name: str = Field(..., description="Agent's name")
    created_date: datetime = Field(..., description="The date when the agent was created")
    version: str = Field(..., description="The version of the agent")
    security_prompt: str = Field(
        "",
        description="The security prompt for the agent. Allows a user to explicitly "
        "specify the security context for the agent's code security checks.",
    )
    hosting: str = Field(
        "",
        description="The hosting environment for the agent. Defaults to ''.",
    )
    model: str = Field(
        "",
        description="The model to use for the agent. Defaults to ''.",
    )


class AgentCreate(BaseModel):
    """Data required to create a new agent."""

    name: str = Field(..., description="Agent's name")
    security_prompt: str | None = Field(
        None,
        description="The security prompt for the agent. Allows a user to explicitly "
        "specify the security context for the agent's code security checks.",
    )
    hosting: str | None = Field(
        None,
        description="The hosting environment for the agent. Defaults to 'openrouter'.",
    )
    model: str | None = Field(
        None,
        description="The model to use for the agent. Defaults to 'openai/gpt-4o-mini'.",
    )


class AgentUpdate(BaseModel):
    """Data for updating an existing agent."""

    name: str | None = Field(None, description="Agent's name")
    security_prompt: str | None = Field(
        None,
        description="The security prompt for the agent. Allows a user to explicitly "
        "specify the security context for the agent's code security checks.",
    )
    hosting: str | None = Field(
        None,
        description="The hosting environment for the agent. Defaults to 'openrouter'.",
    )
    model: str | None = Field(
        None,
        description="The model to use for the agent. Defaults to 'openai/gpt-4o-mini'.",
    )


class AgentListResult(BaseModel):
    """Paginated list result for agents."""

    total: int = Field(..., description="Total number of agents")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Number of agents per page")
    agents: List[Agent] = Field(..., description="List of agents")


def build_tool_registry(
    executor: LocalCodeExecutor, agent_registry: AgentRegistry, config_manager: ConfigManager
) -> ToolRegistry:
    """Build and initialize the tool registry with agent management tools.

    This function creates a new ToolRegistry instance and registers the core agent management tools:
    - create_agent_from_conversation: Creates a new agent from the current conversation
    - edit_agent: Modifies an existing agent's properties
    - delete_agent: Removes an agent from the registry
    - get_agent_info: Retrieves information about agents

    Args:
        executor: The LocalCodeExecutor instance containing conversation history
        agent_registry: The AgentRegistry for managing agents
        config_manager: The ConfigManager for managing configuration

    Returns:
        ToolRegistry: The initialized tool registry with all agent management tools registered
    """
    tool_registry = ToolRegistry()
    tool_registry.init_tools()
    add_admin_tools(tool_registry, executor, agent_registry, config_manager)
    return tool_registry


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

    model_configuration = configure_model(
        hosting=request_hosting,
        model_name=request_model,
        credential_manager=credential_manager,
    )

    if not model_configuration.instance:
        raise ValueError("No model instance configured")

    executor = LocalCodeExecutor(
        model_configuration=model_configuration,
        max_conversation_history=100,
        detail_conversation_length=10,
        can_prompt_user=False,
    )

    operator = Operator(
        executor=executor,
        credential_manager=credential_manager,
        model_configuration=model_configuration,
        config_manager=config_manager,
        type=OperatorType.SERVER,
        agent_registry=agent_registry,
        current_agent=None,
        training_mode=False,
    )

    tool_registry = build_tool_registry(executor, agent_registry, config_manager)
    executor.set_tool_registry(tool_registry)

    return operator


@app.post(
    "/v1/chat",
    response_model=ChatResponse,
    summary="Process chat request",
    description="Accepts a prompt and optional context/configuration, returns the model response "
    "and conversation history.",
    tags=["Chat"],
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
        model_instance = operator.executor.model_configuration.instance

        if request.context and len(request.context) > 0:
            # Override the default system prompt with the provided context
            conversation_history = [
                ConversationRecord(role=msg.role, content=msg.content) for msg in request.context
            ]
            operator.executor.initialize_conversation_history(conversation_history)
        else:
            operator.executor.initialize_conversation_history()

        # Configure model options if provided
        if request.options:
            temperature = request.options.temperature or model_instance.temperature
            if temperature is not None:
                model_instance.temperature = temperature
            model_instance.top_p = request.options.top_p or model_instance.top_p

        response_json = await operator.handle_user_input(request.prompt)
        if response_json is not None:
            response_content = response_json.response
        else:
            response_content = ""

        # Calculate token stats using tiktoken
        tokenizer = None
        try:
            tokenizer = encoding_for_model(request.model)
        except Exception:
            tokenizer = encoding_for_model("gpt-4o")

        prompt_tokens = sum(
            len(tokenizer.encode(msg.content)) for msg in operator.executor.conversation_history
        )
        completion_tokens = len(tokenizer.encode(response_content))
        total_tokens = prompt_tokens + completion_tokens

        return ChatResponse(
            response=response_content,
            context=[
                ConversationRecord(role=msg.role, content=msg.content)
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
    tags=["Chat"],
    openapi_extra={
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
        ..., description="ID of the agent to use for the chat", examples=["agent123"]
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
        try:
            agent_obj = agent_registry.get_agent(agent_id)
        except KeyError as e:
            logger.exception("Error retrieving agent")
            raise HTTPException(status_code=404, detail=f"Agent not found: {e}")

        # Create a new executor for this request using the provided hosting and model
        operator = create_operator(request.hosting, request.model)
        model_instance = operator.executor.model_configuration.instance
        # Apply the retrieved agent to the operator and executor
        operator.current_agent = agent_obj
        setattr(operator.executor, "current_agent", agent_obj)

        if request.context and len(request.context) > 0:
            # Override the default system prompt with the provided context
            conversation_history = [
                ConversationRecord(role=msg.role, content=msg.content) for msg in request.context
            ]
            operator.executor.initialize_conversation_history(conversation_history)
        else:
            operator.executor.initialize_conversation_history()

        # Configure model options if provided
        if request.options:
            temperature = request.options.temperature or model_instance.temperature
            if temperature is not None:
                model_instance.temperature = temperature
            model_instance.top_p = request.options.top_p or model_instance.top_p

        response_json = await operator.handle_user_input(request.prompt)
        response_content = response_json.response if response_json is not None else ""

        # Calculate token stats using tiktoken
        tokenizer = None
        try:
            tokenizer = encoding_for_model(request.model)
        except Exception:
            tokenizer = encoding_for_model("gpt-4o")

        prompt_tokens = sum(
            len(tokenizer.encode(msg.content)) for msg in operator.executor.conversation_history
        )
        completion_tokens = len(tokenizer.encode(response_content))
        total_tokens = prompt_tokens + completion_tokens

        return ChatResponse(
            response=response_content,
            context=[
                ConversationRecord(role=msg.role, content=msg.content)
                for msg in operator.executor.conversation_history
            ],
            stats=ChatStats(
                total_tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
        )

    except HTTPException:
        # Re-raise HTTP exceptions to preserve their status code and detail
        raise
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
async def create_agent(agent: AgentCreate = Body(...)):
    """
    Create a new agent.
    """
    agent_registry = getattr(app.state, "agent_registry", None)
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


@app.get(
    "/v1/agents/{agent_id}",
    response_model=CRUDResponse,
    summary="Retrieve an agent",
    description="Retrieve details for an agent by its ID.",
    tags=["Agents"],
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
    agent_id: str = FPath(..., description="ID of the agent to retrieve", examples=["agent123"])
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


@app.patch(
    "/v1/agents/{agent_id}",
    response_model=CRUDResponse,
    summary="Update an agent",
    description="Update an existing agent with new details. Only provided fields will be updated.",
    tags=["Agents"],
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
    agent_id: str = FPath(..., description="ID of the agent to update", examples=["agent123"]),
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


@app.delete(
    "/v1/agents/{agent_id}",
    response_model=CRUDResponse,
    summary="Delete an agent",
    description="Delete an existing agent by its ID.",
    tags=["Agents"],
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
    agent_id: str = FPath(..., description="ID of the agent to delete", examples=["agent123"])
):
    """
    Delete an existing agent.
    """
    agent_registry = getattr(app.state, "agent_registry", None)
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


@app.get(
    "/health",
    summary="Health Check",
    description="Returns the health status of the API server.",
    tags=["Health"],
)
async def health_check():
    """
    Health check endpoint.

    Returns:
        A JSON object with a "status" key indicating operational status.
    """
    return HealthCheckResponse(status=200, message="ok")
