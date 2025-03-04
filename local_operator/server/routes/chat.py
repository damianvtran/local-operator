"""
Chat endpoints for the Local Operator API.

This module contains the FastAPI route handlers for chat-related endpoints.
"""

import logging
from typing import cast

from fastapi import APIRouter, Depends, HTTPException, Path
from tiktoken import encoding_for_model

from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.server.dependencies import (
    get_agent_registry,
    get_config_manager,
    get_credential_manager,
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
    "/v1/chat/agents/{agent_id}",
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
        # Retrieve the agent registry from app state
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
