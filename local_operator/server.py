"""
FastAPI server implementation for Local Operator API.

Provides REST endpoints for interacting with the Local Operator agent
through HTTP requests instead of CLI.
"""

import logging
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tiktoken import encoding_for_model

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize on startup by setting up the credential and config managers
    config_dir = Path.home() / ".local-operator"
    app.state.credential_manager = CredentialManager(config_dir=config_dir)
    app.state.config_manager = ConfigManager(config_dir=config_dir)
    yield
    # Clean up on shutdown
    app.state.credential_manager = None
    app.state.config_manager = None


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
    if credential_manager is None or config_manager is None:
        raise HTTPException(status_code=500, detail="Server configuration not initialized")

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
    )


@app.post(
    "/chat",
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
            operator.executor.conversation_history = [
                {"role": msg.role, "content": msg.content} for msg in request.context
            ]

        if request.context and len(request.context) > 0:
            # Override the default system prompt with the provided context
            conversation_history = [
                {"role": msg.role, "content": msg.content} for msg in request.context
            ]
        else:
            system_prompt = create_system_prompt()
            conversation_history = [
                {
                    "role": ConversationRole.SYSTEM.value,
                    "content": system_prompt,
                }
            ]

        operator.executor.conversation_history = conversation_history

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
