"""
Chat endpoints for the Local Operator API.

This module contains the FastAPI route handlers for chat-related endpoints.
"""

import logging
from typing import TYPE_CHECKING  # Added

from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from tiktoken import encoding_for_model

from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.env import EnvConfig
from local_operator.jobs import JobManager

# from local_operator.scheduler_service import SchedulerService # Moved to TYPE_CHECKING
from local_operator.server.dependencies import (
    get_agent_registry,
    get_config_manager,
    get_credential_manager,
    get_env_config,
    get_job_manager,
    get_scheduler_service,
    get_websocket_manager,
)
from local_operator.server.models.schemas import (
    AgentChatRequest,
    ChatRequest,
    ChatResponse,
    ChatStats,
    CRUDResponse,
    JobResultSchema,
)
from local_operator.server.utils.attachment_utils import process_attachments

# Import job processor utilities when needed
from local_operator.server.utils.job_processor_queue import (
    create_and_start_job_process_with_queue,
    run_agent_job_in_process_with_queue,
    run_job_in_process_with_queue,
)
from local_operator.server.utils.operator import create_operator
from local_operator.server.utils.websocket_manager import WebSocketManager
from local_operator.types import ConversationRecord

if TYPE_CHECKING:
    from local_operator.scheduler_service import SchedulerService


router = APIRouter(tags=["Chat"])
logger = logging.getLogger("local_operator.server.routes.chat")


@router.post(
    "/v1/chat",
    response_model=CRUDResponse[ChatResponse],
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
                                "hosting": "openrouter",
                                "model": "google/gemini-2.0-flash-001",
                                "context": [],
                                "options": {"temperature": 0.2, "top_p": 0.9},
                            },
                        }
                    }
                }
            }
        }
    },
)
async def chat_endpoint(
    request: ChatRequest,
    credential_manager: CredentialManager = Depends(get_credential_manager),
    config_manager: ConfigManager = Depends(get_config_manager),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    env_config=Depends(get_env_config),
):
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
        operator = create_operator(
            request.hosting,
            request.model,
            credential_manager,
            config_manager,
            agent_registry,
            env_config=env_config,
        )

        model_instance = operator.executor.model_configuration.instance

        if request.context and len(request.context) > 0:
            # Override the default system prompt with the provided context
            conversation_history = [
                ConversationRecord(role=msg.role, content=msg.content) for msg in request.context
            ]
            operator.executor.initialize_conversation_history(conversation_history, overwrite=True)
        else:
            try:
                operator.executor.initialize_conversation_history()
            except ValueError:
                # Conversation history already initialized
                pass

        # Configure model options if provided
        if request.options:
            temperature = request.options.temperature or model_instance.temperature
            if temperature is not None:
                model_instance.temperature = temperature
            model_instance.top_p = request.options.top_p or model_instance.top_p

        processed_attachments = await process_attachments(request.attachments)
        response_json, final_response = await operator.handle_user_input(
            request.prompt, attachments=processed_attachments
        )

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
            len(tokenizer.encode(msg.content)) for msg in operator.executor.agent_state.conversation
        )
        completion_tokens = len(tokenizer.encode(response_content))
        total_tokens = prompt_tokens + completion_tokens

        return CRUDResponse(
            status=200,
            message="Chat request processed successfully",
            result=ChatResponse(
                response=final_response or "",
                context=[
                    ConversationRecord(role=msg.role, content=msg.content, files=msg.files)
                    for msg in operator.executor.agent_state.conversation
                ],
                stats=ChatStats(
                    total_tokens=total_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                ),
            ),
        )

    except Exception:
        logger.exception("Unexpected error while processing chat request")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post(
    "/v1/chat/agents/{agent_id}",
    response_model=CRUDResponse[ChatResponse],
    summary="Process chat request using a specific agent",
    description=(
        "Accepts a prompt and optional context/configuration, retrieves the specified "
        "agent from the registry, applies it to the operator and executor, and returns the "
        "model response and conversation history."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "example": {
                            "summary": "Example Request with Agent",
                            "value": {
                                "prompt": "How do I implement a binary search in Python?",
                                "hosting": "openrouter",
                                "model": "google/gemini-2.0-flash-001",
                                "options": {"temperature": 0.2, "top_p": 0.9},
                                "persist_conversation": False,
                            },
                        }
                    }
                }
            }
        },
    },
)
async def chat_with_agent(
    request: AgentChatRequest,
    credential_manager: CredentialManager = Depends(get_credential_manager),
    config_manager: ConfigManager = Depends(get_config_manager),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    env_config=Depends(get_env_config),
    agent_id: str = Path(
        ..., description="ID of the agent to use for the chat", examples=["agent123"]
    ),
):
    """
    Process a chat request using a specific agent from the registry and return the response with
    context. The specified agent is applied to both the operator and executor.
    """
    try:
        # Retrieve the specific agent from the registry
        try:
            agent_obj = agent_registry.get_agent(agent_id)
        except KeyError as e:
            logger.exception("Error retrieving agent")
            raise HTTPException(status_code=404, detail=f"Agent not found: {e}")

        # Create a new executor for this request using the provided hosting and model
        operator = create_operator(
            request.hosting,
            request.model,
            credential_manager,
            config_manager,
            agent_registry,
            current_agent=agent_obj,
            persist_conversation=request.persist_conversation,
            env_config=env_config,
        )
        model_instance = operator.executor.model_configuration.instance

        # Configure model options if provided
        if request.options:
            temperature = request.options.temperature or model_instance.temperature
            if temperature is not None:
                model_instance.temperature = temperature
            model_instance.top_p = request.options.top_p or model_instance.top_p

        processed_attachments = await process_attachments(request.attachments)
        response_json, final_response = await operator.handle_user_input(
            request.prompt, attachments=processed_attachments
        )
        response_content = response_json.response if response_json is not None else ""

        # Calculate token stats using tiktoken
        tokenizer = None
        try:
            tokenizer = encoding_for_model(request.model)
        except Exception:
            tokenizer = encoding_for_model("gpt-4o")

        prompt_tokens = sum(
            len(tokenizer.encode(msg.content)) for msg in operator.executor.agent_state.conversation
        )
        completion_tokens = len(tokenizer.encode(response_content))
        total_tokens = prompt_tokens + completion_tokens

        return CRUDResponse(
            status=200,
            message="Chat request processed successfully",
            result=ChatResponse(
                response=final_response or "",
                context=[
                    ConversationRecord(role=msg.role, content=msg.content)
                    for msg in operator.executor.agent_state.conversation
                ],
                stats=ChatStats(
                    total_tokens=total_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                ),
            ),
        )

    except HTTPException:
        # Re-raise HTTP exceptions to preserve their status code and detail
        raise
    except Exception:
        logger.exception("Unexpected error while processing chat request with agent")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post(
    "/v1/chat/async",
    response_model=CRUDResponse[JobResultSchema],
    summary="Process chat request asynchronously",
    description="Accepts a prompt and optional context/configuration, starts an asynchronous job "
    "to process the request and returns a job ID.",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "example": {
                            "summary": "Example Async Request",
                            "value": {
                                "prompt": "Print 'Hello, world!'",
                                "hosting": "openrouter",
                                "model": "google/gemini-2.0-flash-001",
                                "context": [],
                                "options": {"temperature": 0.2, "top_p": 0.9},
                            },
                        }
                    }
                }
            }
        }
    },
)
async def chat_async_endpoint(
    request: ChatRequest,
    credential_manager: CredentialManager = Depends(get_credential_manager),
    config_manager: ConfigManager = Depends(get_config_manager),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    job_manager: JobManager = Depends(get_job_manager),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager),
    env_config: EnvConfig = Depends(get_env_config),
    scheduler_service: "SchedulerService" = Depends(
        get_scheduler_service
    ),  # Changed to string literal
):
    """
    Process a chat request asynchronously and return a job ID.

    The endpoint accepts a JSON payload containing the prompt, hosting, model selection, and
    optional parameters. Instead of waiting for the response, it creates a background job
    and returns immediately with a job ID that can be used to check the status later.

    Args:
        request: The chat request containing prompt and configuration
        credential_manager: Dependency for managing credentials
        config_manager: Dependency for managing configuration
        agent_registry: Dependency for accessing agent registry
        job_manager: Dependency for managing asynchronous jobs
        websocket_manager: Dependency for managing WebSocket connections

    Returns:
        A response containing the job ID and status

    Raises:
        HTTPException: If there's an error setting up the job
    """
    try:
        processed_attachments = await process_attachments(request.attachments)

        # Create a job in the job manager
        job = await job_manager.create_job(
            prompt=request.prompt,
            model=request.model,
            hosting=request.hosting,
            agent_id=None,
        )

        # Create and start a process for the job using the utility function
        create_and_start_job_process_with_queue(
            job_id=job.id,
            process_func=run_job_in_process_with_queue,
            args=(
                job.id,
                request.prompt,
                processed_attachments,
                request.model,
                request.hosting,
                credential_manager,
                config_manager,
                agent_registry,
                env_config,
                request.context if request.context else None,
                request.options.model_dump() if request.options else None,
            ),
            job_manager=job_manager,
            websocket_manager=websocket_manager,
            scheduler_service=scheduler_service,
        )

        # Return job information
        response = CRUDResponse(
            status=202,
            message="Chat request accepted",
            result=JobResultSchema(
                id=job.id,
                agent_id=None,
                status=job.status,
                prompt=request.prompt,
                model=request.model,
                hosting=request.hosting,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                result=None,
            ),
        )
        return JSONResponse(status_code=202, content=jsonable_encoder(response))

    except HTTPException:
        # Re-raise HTTP exceptions to preserve their status code and detail
        raise
    except Exception:
        logger.exception("Unexpected error while setting up async chat job")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post(
    "/v1/chat/agents/{agent_id}/async",
    response_model=CRUDResponse[JobResultSchema],
    summary="Process agent chat request asynchronously",
    description=(
        "Accepts a prompt and optional context/configuration, retrieves the specified "
        "agent from the registry, starts an asynchronous job to process the request and returns "
        "a job ID."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "example": {
                            "summary": "Example Async Agent Request",
                            "value": {
                                "prompt": "How do I implement a binary search in Python?",
                                "hosting": "openrouter",
                                "model": "google/gemini-2.0-flash-001",
                                "options": {"temperature": 0.2, "top_p": 0.9},
                                "persist_conversation": False,
                                "user_message_id": "",
                            },
                        }
                    }
                }
            }
        }
    },
)
async def chat_with_agent_async(
    request: AgentChatRequest,
    credential_manager: CredentialManager = Depends(get_credential_manager),
    config_manager: ConfigManager = Depends(get_config_manager),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    job_manager: JobManager = Depends(get_job_manager),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager),
    env_config: EnvConfig = Depends(get_env_config),
    scheduler_service: "SchedulerService" = Depends(
        get_scheduler_service
    ),  # Changed to string literal
    agent_id: str = Path(
        ..., description="ID of the agent to use for the chat", examples=["agent123"]
    ),
):
    """
    Process a chat request asynchronously using a specific agent and return a job ID.

    The endpoint accepts a JSON payload containing the prompt, hosting, model selection, and
    optional parameters. It retrieves the specified agent from the registry, applies it to the
    operator and executor, and creates a background job that returns immediately with a job ID
    that can be used to check the status later.

    Args:
        request: The chat request containing prompt and configuration
        credential_manager: Dependency for managing credentials
        config_manager: Dependency for managing configuration
        agent_registry: Dependency for accessing agent registry
        job_manager: Dependency for managing asynchronous jobs
        websocket_manager: Dependency for managing WebSocket connections
        agent_id: ID of the agent to use for the chat

    Returns:
        A response containing the job ID and status

    Raises:
        HTTPException: If there's an error retrieving the agent or setting up the job
    """
    try:
        # Retrieve the specific agent from the registry
        try:
            agent_registry.get_agent(agent_id)
        except KeyError as e:
            logger.exception("Error retrieving agent")
            raise HTTPException(status_code=404, detail=f"Agent not found: {e}")

        processed_attachments = await process_attachments(request.attachments)
        # Create a job in the job manager
        job = await job_manager.create_job(
            prompt=request.prompt,
            model=request.model,
            hosting=request.hosting,
            agent_id=agent_id,
        )

        # Create and start a process for the job
        create_and_start_job_process_with_queue(
            job_id=job.id,
            process_func=run_agent_job_in_process_with_queue,
            args=(
                job.id,
                request.prompt,
                processed_attachments,
                request.model,
                request.hosting,
                agent_id,
                credential_manager,
                config_manager,
                agent_registry,
                env_config,
                request.persist_conversation,
                request.user_message_id,
            ),
            job_manager=job_manager,
            websocket_manager=websocket_manager,
            scheduler_service=scheduler_service,
        )

        # Return job information
        response = CRUDResponse(
            status=202,
            message="Chat request accepted",
            result=JobResultSchema(
                id=job.id,
                agent_id=agent_id,
                status=job.status,
                prompt=request.prompt,
                model=request.model,
                hosting=request.hosting,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                result=None,
            ),
        )
        return JSONResponse(status_code=202, content=jsonable_encoder(response))

    except HTTPException:
        # Re-raise HTTP exceptions to preserve their status code and detail
        raise
    except Exception:
        logger.exception("Unexpected error while setting up async chat job")
        raise HTTPException(status_code=500, detail="Internal Server Error")
