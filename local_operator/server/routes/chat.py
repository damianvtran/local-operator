"""
Chat endpoints for the Local Operator API.

This module contains the FastAPI route handlers for chat-related endpoints.
"""

import asyncio
import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Path
from tiktoken import encoding_for_model

from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.jobs import JobManager, JobStatus
from local_operator.server.dependencies import (
    get_agent_registry,
    get_config_manager,
    get_credential_manager,
    get_job_manager,
)
from local_operator.server.models.schemas import ChatRequest, ChatResponse, ChatStats
from local_operator.server.utils.operator import create_operator
from local_operator.types import ConversationRecord

router = APIRouter(tags=["Chat"])
logger = logging.getLogger("local_operator.server.routes.chat")


@router.post(
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
async def chat_endpoint(
    request: ChatRequest,
    credential_manager: CredentialManager = Depends(get_credential_manager),
    config_manager: ConfigManager = Depends(get_config_manager),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
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
        )

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


@router.post(
    "/v1/chat/agents/{agent_id}/sync",
    response_model=ChatResponse,
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
    request: ChatRequest,
    credential_manager: CredentialManager = Depends(get_credential_manager),
    config_manager: ConfigManager = Depends(get_config_manager),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
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
        )
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


@router.post(
    "/v1/chat/async",
    response_model=Dict[str, Any],
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
async def chat_async_endpoint(
    request: ChatRequest,
    credential_manager: CredentialManager = Depends(get_credential_manager),
    config_manager: ConfigManager = Depends(get_config_manager),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    job_manager: JobManager = Depends(get_job_manager),
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

    Returns:
        A response containing the job ID and status

    Raises:
        HTTPException: If there's an error setting up the job
    """
    try:
        # Create the operator with the specified model
        operator = create_operator(
            request.hosting,
            request.model,
            credential_manager,
            config_manager,
            agent_registry,
        )

        model_instance = operator.executor.model_configuration.instance

        # Initialize conversation history
        if request.context:
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

        # Create a job in the job manager
        job = await job_manager.create_job(
            prompt=request.prompt,
            model=request.model,
            hosting=request.hosting,
            agent_id=getattr(operator, "agent_id", None),
        )

        # Define the background task
        async def process_chat_job():
            try:
                await job_manager.update_job_status(job.id, JobStatus.PROCESSING)
                response_json = await operator.handle_user_input(request.prompt)

                # Create result with response and context
                result = {
                    "response": response_json.response if response_json is not None else "",
                    "context": [
                        {"role": msg.role, "content": msg.content}
                        for msg in operator.executor.conversation_history
                    ],
                }

                await job_manager.update_job_status(job.id, JobStatus.COMPLETED, result)
            except Exception as e:
                logger.exception(f"Job {job.id} failed: {str(e)}")
                await job_manager.update_job_status(job.id, JobStatus.FAILED, {"error": str(e)})

        # Create and register the task
        task = asyncio.create_task(process_chat_job())
        await job_manager.register_task(job.id, task)

        # Return job information
        return {
            "status": 202,
            "message": "Chat request accepted",
            "result": {"job_id": job.id, "status": job.status},
        }

    except HTTPException:
        # Re-raise HTTP exceptions to preserve their status code and detail
        raise
    except Exception:
        logger.exception("Unexpected error while setting up async chat job")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get(
    "/v1/chat/jobs/{job_id}",
    summary="Get chat job status",
    description="Retrieves the status and result of an asynchronous chat job.",
)
async def get_chat_job_status(
    job_id: str = Path(..., description="The ID of the chat job to retrieve"),
    job_manager: JobManager = Depends(get_job_manager),
):
    """
    Get the status and result of an asynchronous chat job.

    Args:
        job_id: The ID of the job to check
        job_manager: The job manager instance

    Returns:
        The job status and result if available

    Raises:
        HTTPException: If the job is not found or there's an error retrieving it
    """
    try:
        job = await job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        job_summary = job_manager.get_job_summary(job)

        return {
            "status": 200,
            "message": "Job status retrieved",
            "result": job_summary,
        }

    except HTTPException:
        # Re-raise HTTP exceptions to preserve their status code and detail
        raise
    except Exception:
        logger.exception(f"Unexpected error while retrieving job {job_id}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
