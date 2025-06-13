"""
Chat endpoints for the Local Operator API.

This module contains the FastAPI route handlers for chat-related endpoints.
"""

import logging
from pathlib import Path as FilePath
from typing import TYPE_CHECKING  # Added

from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from tiktoken import encoding_for_model

from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.env import EnvConfig
from local_operator.helpers import parse_agent_action_xml, parse_replacements
from local_operator.jobs import JobManager
from local_operator.prompts import EditFileInstructionsPrompt

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
    AgentEditFileRequest,
    AgentEditFileResponse,
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
from local_operator.types import ConversationRecord, ConversationRole

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


@router.post(
    "/v1/chat/agents/{agent_id}/edit",
    response_model=CRUDResponse,
    summary="Edit a file using an agent",
    description="Edit a file using a specific agent with an edit prompt",
    responses={
        200: {
            "description": "Successful response containing the edit diffs",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/CRUDResponse"},
                    "example": {
                        "status": 200,
                        "message": "File edit completed successfully",
                        "result": {
                            "file_path": "/path/to/file.py",
                            "edit_prompt": "Add error handling to the function",
                            "edit_diffs": [
                                {
                                    "find": "def my_function():\n    return result",
                                    "replace": "def my_function():\n    try:\n        return result\n    except Exception as e:\n        logger.error(f'Error: {e}')\n        raise",  # noqa: E501
                                }
                            ],
                            "raw_response": "...",
                        },
                    },
                }
            },
        },
        404: {"description": "Agent not found"},
        500: {"description": "Internal Server Error"},
    },
)
async def edit_file_with_agent(
    request: AgentEditFileRequest,
    credential_manager: CredentialManager = Depends(get_credential_manager),
    config_manager: ConfigManager = Depends(get_config_manager),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    env_config: EnvConfig = Depends(get_env_config),
    agent_id: str = Path(
        ..., description="ID of the agent to use for editing", examples=["agent123"]
    ),
):
    """
    Edit a file using a specific agent with an edit prompt.

    The endpoint loads the specified file, applies the agent's conversation history,
    and uses the agent to generate edit diffs based on the provided edit prompt.

    Args:
        request: The edit request containing file path and edit prompt
        credential_manager: Dependency for managing credentials
        config_manager: Dependency for managing configuration
        agent_registry: Dependency for accessing agent registry
        env_config: Environment configuration
        agent_id: ID of the agent to use for editing

    Returns:
        A response containing the generated edit diffs

    Raises:
        HTTPException: If there's an error retrieving the agent or processing the edit
    """
    try:
        # Retrieve the specific agent from the registry
        try:
            agent_obj = agent_registry.get_agent(agent_id)
        except KeyError as e:
            logger.exception("Error retrieving agent")
            raise HTTPException(status_code=404, detail=f"Agent not found: {e}")

        resolved_file_path = FilePath(request.file_path).expanduser().resolve()

        # Load the file content
        try:
            with open(resolved_file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"File not found: {resolved_file_path}")
        except Exception as e:
            logger.exception(f"Error reading file {resolved_file_path}")
            raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

        # Create operator with the agent
        operator = create_operator(
            request.hosting,
            request.model,
            credential_manager,
            config_manager,
            agent_registry,
            current_agent=agent_obj,
            persist_conversation=False,
            env_config=env_config,
        )

        # Load agent conversation history
        try:
            operator.executor.initialize_conversation_history()
        except ValueError:
            # Conversation history already initialized
            pass

        # Construct the edit prompt with file content
        edit_instruction = EditFileInstructionsPrompt.format(
            file_path=resolved_file_path,
            edit_prompt=request.edit_prompt,
            file_content=file_content,
            selection=request.selection,
        )

        processed_attachments = await process_attachments(request.attachments)

        operator.executor.append_to_history(
            ConversationRecord(
                role=ConversationRole.USER,
                content=edit_instruction,
                files=processed_attachments,
            )
        )

        # Retry mechanism for edit suggestions
        max_retries = 3
        edit_diffs = []
        raw_response = ""

        for attempt in range(max_retries):
            # Invoke the model to get edit suggestions
            response = await operator.executor.invoke_model(
                operator.executor.get_conversation_history()
            )

            response_content = (
                response.content
                if response and response.content and isinstance(response.content, str)
                else ""
            )

            raw_response = response_content

            if response_content:
                if "<action_response>" in response_content:
                    action_response = parse_agent_action_xml(response_content)
                    if action_response:
                        edit_diffs = action_response.get("replacements", [])
                else:
                    edit_diffs = parse_replacements(response_content)

                # Validate that all find strings exist in the file content
                invalid_finds = []
                for diff in edit_diffs:
                    find_text = diff.get("find", "")
                    if find_text and find_text not in file_content:
                        invalid_finds.append(find_text)

                # If all finds are valid, break out of retry loop
                if not invalid_finds:
                    break

                # If this is not the last attempt, add error message and retry
                if attempt < max_retries - 1:
                    error_message = "The following SEARCH blocks were not "
                    f"found in the file '{resolved_file_path}':\n"
                    for i, invalid_find in enumerate(invalid_finds, 1):
                        error_message += (
                            f"\n{i}. SEARCH block not found:\n---\n{invalid_find}\n---\n"
                        )

                    error_message += (
                        "\nPlease review the file content carefully and provide exact text matches "
                        "for the SEARCH blocks. Make sure to include proper indentation, spacing, "
                        "and line breaks exactly as they appear in the file."
                    )

                    operator.executor.append_to_history(
                        ConversationRecord(
                            role=ConversationRole.USER,
                            content=error_message,
                        )
                    )
                else:
                    # Last attempt failed, return with error details
                    error_details = (
                        f"Failed to find valid search blocks after {max_retries} attempts. "
                    )
                    error_details += f"Invalid search blocks: {[find for find in invalid_finds]}"
                    raise HTTPException(status_code=500, detail=error_details)

        # Return the edit response
        response_data = CRUDResponse(
            status=200,
            message="File edit completed successfully",
            result=AgentEditFileResponse(
                file_path=str(resolved_file_path),
                edit_prompt=request.edit_prompt,
                edit_diffs=edit_diffs,
                raw_response=raw_response,
            ),
        )

        return JSONResponse(status_code=200, content=jsonable_encoder(response_data))

    except HTTPException:
        # Re-raise HTTP exceptions to preserve their status code and detail
        raise
    except Exception as e:
        logger.exception("Unexpected error while processing file edit")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
