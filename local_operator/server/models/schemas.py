"""
Pydantic models for the Local Operator API.

This module contains all the Pydantic models used for request and response validation
in the Local Operator API.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar
from uuid import UUID

from pydantic import BaseModel, Field

# AgentEditFields will be used in the routes module
from local_operator.jobs import JobResult, JobStatus
from local_operator.model.registry import ModelInfo, ProviderDetail
from local_operator.types import (  # Added ScheduleUnit
    CodeExecutionResult,
    ConversationRecord,
    ScheduleUnit,
)


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
        attachments: Optional list of file paths (local or remote) to be used in the analysis.
            These files are expected to be publicly accessible.
    """

    hosting: str
    model: str
    prompt: str
    stream: bool = False
    context: Optional[List[ConversationRecord]] = None
    options: Optional[ChatOptions] = None
    attachments: Optional[List[str]] = None


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


T = TypeVar("T")


class CRUDResponse(BaseModel, Generic[T]):
    """
    Standard response schema for CRUD operations.

    Attributes:
        status: HTTP status code
        message: Outcome message of the operation
        result: The resulting data, which can be an object, paginated list, or empty.
    """

    status: int
    message: str
    result: Optional[T] = None


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
    description: str = Field(
        "",
        description="A description of the agent. Defaults to ''.",
    )
    tags: Optional[List[str]] = Field(
        None,
        description="Tags for the agent. Defaults to an empty list.",
    )
    categories: Optional[List[str]] = Field(
        None,
        description="Categories for the agent. Defaults to an empty list.",
    )
    last_message: str = Field(
        "",
        description="The last message sent to the agent. Defaults to ''.",
    )
    last_message_datetime: datetime = Field(
        ...,
        description="The date and time of the last message sent to the agent.",
    )
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Controls randomness in responses. Higher values like 0.8 make output more "
        "random, while lower values like 0.2 make it more focused and deterministic.",
    )
    top_p: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Controls cumulative probability of tokens to sample from. Higher "
        "values (0.95) keep more options, lower values (0.1) are more selective.",
    )
    top_k: Optional[int] = Field(
        None,
        description="Limits tokens to sample from at each step. Lower values (10) are "
        "more selective, higher values (100) allow more variety.",
    )
    max_tokens: Optional[int] = Field(
        None,
        description="Maximum tokens to generate. Model may generate fewer if response completes "
        "before reaching limit.",
    )
    stop: Optional[List[str]] = Field(
        None, description="List of strings that will stop generation when encountered."
    )
    frequency_penalty: Optional[float] = Field(
        None,
        description="Reduces repetition by lowering likelihood of repeated tokens. "
        "Range from -2.0 to 2.0.",
    )
    presence_penalty: Optional[float] = Field(
        None,
        description="Increases diversity by lowering likelihood of prompt tokens. "
        "Range from -2.0 to 2.0.",
    )
    seed: Optional[int] = Field(
        None, description="Random number seed for deterministic generation."
    )
    current_working_directory: Optional[str] = Field(
        ".",
        description="The current working directory for the agent.  Updated whenever the "
        "agent changes its working directory through code execution.  Defaults to '.'",
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
    description: str | None = Field(
        None,
        description="A description of the agent. Defaults to ''.",
    )
    temperature: float | None = Field(
        None,
        description="Controls randomness in responses. Higher values like 0.8 make "
        "output more random, while lower values like 0.2 make it more focused and "
        "deterministic.",
    )
    top_p: float | None = Field(
        None,
        description="Controls cumulative probability of tokens to sample from. Higher "
        "values (0.95) keep more options, lower values (0.1) are more selective.",
    )
    top_k: int | None = Field(
        None,
        description="Limits tokens to sample from at each step. Lower values (10) are "
        "more selective, higher values (100) allow more variety.",
    )
    max_tokens: int | None = Field(
        None,
        description="Maximum tokens to generate. Model may generate fewer if response completes "
        "before reaching limit.",
    )
    stop: List[str] | None = Field(
        None,
        description="List of strings that will stop generation when encountered.",
    )
    frequency_penalty: float | None = Field(
        None,
        description="Reduces repetition by lowering likelihood of repeated tokens. "
        "Range from -2.0 to 2.0.",
    )
    presence_penalty: float | None = Field(
        None,
        description="Increases diversity by lowering likelihood of prompt tokens. "
        "Range from -2.0 to 2.0.",
    )
    seed: int | None = Field(
        None,
        description="Random number seed for deterministic generation.",
    )
    current_working_directory: str | None = Field(
        "~/local-operator-home",
        description="The current working directory for the agent.  Updated whenever the "
        "agent changes its working directory through code execution.  Defaults to "
        "'~/local-operator-home'.",
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
        description="The model to use for the agent. Defaults to 'google/gemini-2.0-flash-001'.",
    )
    description: str | None = Field(
        None,
        description="A description of the agent.  Defaults to ''.",
    )
    tags: List[str] | None = Field(
        None,
        description="Tags for the agent.  Defaults to an empty list.",
    )
    categories: List[str] | None = Field(
        None,
        description="Categories for the agent.  Defaults to an empty list.",
    )
    temperature: float | None = Field(
        None,
        description="Controls randomness in responses. Higher values like 0.8 make output more "
        "random, while lower values like 0.2 make it more focused and deterministic.",
    )
    top_p: float | None = Field(
        None,
        description="Controls cumulative probability of tokens to sample from. Higher "
        "values (0.95) keep more options, lower values (0.1) are more selective.",
    )
    top_k: int | None = Field(
        None,
        description="Limits tokens to sample from at each step. Lower values (10) are more "
        "selective, higher values (100) allow more variety.",
    )
    max_tokens: int | None = Field(
        None,
        description="Maximum tokens to generate. Model may generate fewer if response completes "
        "before reaching limit.",
    )
    stop: List[str] | None = Field(
        None,
        description="List of strings that will stop generation when encountered.",
    )
    frequency_penalty: float | None = Field(
        None,
        description="Reduces repetition by lowering likelihood of repeated tokens. "
        "Range from -2.0 to 2.0.",
    )
    presence_penalty: float | None = Field(
        None,
        description="Increases diversity by lowering likelihood of prompt tokens. "
        "Range from -2.0 to 2.0.",
    )
    seed: int | None = Field(
        None,
        description="Random number seed for deterministic generation.",
    )
    current_working_directory: str | None = Field(
        None,
        description="The current working directory for the agent.  Updated whenever the "
        "agent changes its working directory through code execution.",
    )


class AgentListResult(BaseModel):
    """Paginated list result for agents."""

    total: int = Field(..., description="Total number of agents")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Number of agents per page")
    agents: List[Agent] = Field(..., description="List of agents")


class AgentGetConversationResult(BaseModel):
    """Schema for getting an agent conversation."""

    agent_id: str = Field(..., description="ID of the agent involved in the conversation")
    last_message_datetime: datetime = Field(
        ..., description="Date of the last message in the conversation"
    )
    first_message_datetime: datetime = Field(
        ..., description="Date of the first message in the conversation"
    )
    messages: List[ConversationRecord] = Field(
        default_factory=list, description="List of messages in the conversation"
    )
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Number of messages per page")
    total: int = Field(..., description="Total number of messages in the conversation")
    count: int = Field(..., description="Number of messages in the current page")


class AgentExecutionHistoryResult(BaseModel):
    """Schema for getting an agent execution history."""

    agent_id: str = Field(..., description="ID of the agent involved in the execution history")
    history: List[CodeExecutionResult] = Field(..., description="List of code execution results")
    last_execution_datetime: datetime = Field(
        ..., description="Date of the last execution in the history"
    )
    first_execution_datetime: datetime = Field(
        ..., description="Date of the first execution in the history"
    )
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Number of messages per page")
    total: int = Field(..., description="Total number of messages in the execution history")
    count: int = Field(..., description="Number of messages in the current page")


class JobResultSchema(BaseModel):
    """Schema for job result data.

    Attributes:
        id: Unique identifier for the job
        agent_id: Optional ID of the agent associated with the job
        status: Current status of the job
        prompt: The prompt that was submitted for processing
        model: The model used for processing
        hosting: The hosting service used
        created_at: Timestamp when the job was created
        started_at: Optional timestamp when the job processing started
        completed_at: Optional timestamp when the job completed
        result: Optional result data containing response, context, and stats
    """

    id: str = Field(..., description="Unique identifier for the job")
    agent_id: Optional[str] = Field(None, description="ID of the agent associated with the job")
    status: JobStatus = Field(..., description="Current status of the job")
    prompt: str = Field(..., description="The prompt that was submitted for processing")
    model: str = Field(..., description="The model used for processing")
    hosting: str = Field(..., description="The hosting service used")
    created_at: float = Field(..., description="Timestamp when the job was created")
    started_at: Optional[float] = Field(None, description="Timestamp when job processing started")
    completed_at: Optional[float] = Field(None, description="Timestamp when job completed")
    result: Optional[JobResult] = Field(
        None, description="Result data containing response, context, and stats"
    )


class AgentChatRequest(BaseModel):
    """Request body for chat generation endpoint.

    Attributes:
        hosting: Name of the hosting service to use for generation
        model: Name of the model to use for generation
        prompt: The prompt to generate a response for
        stream: Whether to stream the response token by token. Default: False
        options: Optional generation parameters to override defaults
        persist_conversation: Whether to persist the conversation history by
        continuously updating the agent's conversation history with each new message.
        Default: False
        user_message_id: Optional ID of the user message to assign to the first user message
            in the conversation.  This is used by the UI to prevent duplicate user
            messages after the initial render.
        attachments: Optional list of file paths (local or remote) to be used in the analysis.
            These files are expected to be publicly accessible.
    """

    hosting: str
    model: str
    prompt: str
    stream: bool = False
    options: Optional[ChatOptions] = None
    persist_conversation: bool = False
    user_message_id: Optional[str] = None
    attachments: Optional[List[str]] = None


class ConfigUpdate(BaseModel):
    """Data for updating configuration settings.

    Attributes:
        conversation_length: Number of conversation messages to retain
        detail_length: Maximum length of detailed conversation history
        max_learnings_history: Maximum number of learning entries to retain
        hosting: AI model hosting provider
        model_name: Name of the AI model to use
        auto_save_conversation: Whether to automatically save the conversation
    """

    conversation_length: Optional[int] = Field(
        None, description="Number of conversation messages to retain", ge=1
    )
    detail_length: Optional[int] = Field(
        None, description="Maximum length of detailed conversation history", ge=1
    )
    max_learnings_history: Optional[int] = Field(
        None, description="Maximum number of learning entries to retain", ge=1
    )
    hosting: Optional[str] = Field(None, description="AI model hosting provider")
    model_name: Optional[str] = Field(None, description="Name of the AI model to use")
    auto_save_conversation: Optional[bool] = Field(
        None, description="Whether to automatically save the conversation"
    )


class ConfigResponse(BaseModel):
    """Response containing configuration settings.

    Attributes:
        version: Configuration schema version for compatibility
        metadata: Metadata about the configuration
        values: Configuration settings
    """

    version: str = Field(..., description="Configuration schema version for compatibility")
    metadata: Dict[str, Any] = Field(..., description="Metadata about the configuration")
    values: Dict[str, Any] = Field(..., description="Configuration settings")


class SystemPromptResponse(BaseModel):
    """Response containing the system prompt content.

    Attributes:
        content: The content of the system prompt
        last_modified: Timestamp when the system prompt was last modified
    """

    content: str = Field(..., description="The content of the system prompt")
    last_modified: str = Field(
        ..., description="Timestamp when the system prompt was last modified"
    )


class SystemPromptUpdate(BaseModel):
    """Data for updating the system prompt.

    Attributes:
        content: The new content for the system prompt
    """

    content: str = Field(..., description="The new content for the system prompt")


class CredentialUpdate(BaseModel):
    """Data for updating a credential.

    Attributes:
        key: The credential key to update
        value: The new value for the credential
    """

    key: str = Field(..., description="The credential key to update")
    value: str = Field(..., description="The new value for the credential")


class CredentialKey(BaseModel):
    """Representation of a credential key.

    Attributes:
        key: The credential key name
    """

    key: str = Field(..., description="The credential key name")


class CredentialListResult(BaseModel):
    """Result containing a list of credential keys.

    Attributes:
        keys: List of credential keys
    """

    keys: List[str] = Field(..., description="List of credential keys")


class ModelEntry(BaseModel):
    """A single model entry.

    Attributes:
        id: Unique identifier for the model
        name: Optional display name for the model
        provider: The provider of the model
        info: Detailed information about the model
    """

    id: str = Field(..., description="Unique identifier for the model")
    name: Optional[str] = Field(None, description="Display name for the model")
    provider: str = Field(..., description="The provider of the model")
    info: ModelInfo = Field(..., description="Detailed information about the model")

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "id": "claude-3-opus-20240229",
                "provider": "anthropic",
                "info": {
                    "input_price": 15000.0,
                    "output_price": 75000.0,
                    "max_tokens": 200000,
                    "context_window": 200000,
                    "supports_images": True,
                    "supports_prompt_cache": False,
                    "description": "Most powerful Claude model for highly complex tasks",
                },
            }
        }


class ModelListResponse(BaseModel):
    """Response for listing models.

    Attributes:
        models: List of model entries
    """

    models: List[ModelEntry] = Field(..., description="List of model entries")

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "models": [
                    {
                        "id": "claude-3-opus-20240229",
                        "provider": "anthropic",
                        "info": {
                            "input_price": 15000.0,
                            "output_price": 75000.0,
                            "max_tokens": 200000,
                            "context_window": 200000,
                            "supports_images": True,
                            "supports_prompt_cache": False,
                            "description": "Most powerful Claude model for highly complex tasks",
                        },
                    },
                    {
                        "id": "gpt-4o",
                        "name": "GPT-4o",
                        "provider": "openai",
                        "info": {
                            "input_price": 5000.0,
                            "output_price": 15000.0,
                            "max_tokens": 128000,
                            "context_window": 128000,
                            "supports_images": True,
                            "supports_prompt_cache": False,
                            "description": "OpenAI's most advanced multimodal model",
                        },
                    },
                ]
            }
        }


class ProviderListResponse(BaseModel):
    """Response for listing providers.

    Attributes:
        providers: List of provider details
    """

    providers: List[ProviderDetail] = Field(..., description="List of provider details")

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "providers": [
                    {
                        "id": "openai",
                        "name": "OpenAI",
                        "description": "OpenAI's API provides access to GPT-4o and other models",
                        "url": "https://platform.openai.com/",
                        "requiredCredentials": ["OPENAI_API_KEY"],
                    },
                    {
                        "id": "anthropic",
                        "name": "Anthropic",
                        "description": "Anthropic's Claude models for safe, helpful AI assistants",
                        "url": "https://www.anthropic.com/",
                        "requiredCredentials": ["ANTHROPIC_API_KEY"],
                    },
                ]
            }
        }


class ModelListQuerySort(str, Enum):
    """Sorting options for model listings."""

    ID = "id"
    NAME = "name"
    PROVIDER = "provider"
    RECOMMENDED = "recommended"


class ModelListQueryParams(BaseModel):
    """Query parameters for listing models.

    Attributes:
        provider: Optional provider to filter models by
        sort: Optional field to sort models by (default: 'id')
        direction: Optional sort direction ('ascending' or 'descending', default: 'ascending')
    """

    provider: Optional[str] = Field(None, description="Provider to filter models by")
    sort: Optional[ModelListQuerySort] = Field(
        ModelListQuerySort.RECOMMENDED, description="Field to sort models by"
    )
    direction: Optional[str] = Field(
        "ascending", description="Sort direction ('ascending' or 'descending')"
    )


class AgentImportResponse(BaseModel):
    """Response for agent import endpoint.

    Attributes:
        agent_id: ID of the imported agent
        name: Name of the imported agent
    """

    agent_id: str = Field(..., description="ID of the imported agent")
    name: str = Field(..., description="Name of the imported agent")


class HealthCheckResponse(BaseModel):
    """Response for health check endpoint.

    Attributes:
        version: Version of the Local Operator
    """

    version: str = Field(..., description="Version of the Local Operator")


class WebsocketConnectionType(str, Enum):
    """Types of websocket connections."""

    MESSAGE = "message"
    HEALTH = "health"


# Schedule Schemas
class ScheduleResponse(BaseModel):
    """Response model for a schedule."""

    id: UUID
    agent_id: UUID
    prompt: str
    interval: int
    unit: ScheduleUnit
    is_active: bool
    one_time: bool
    created_at: datetime
    last_run_at: Optional[datetime] = None
    start_time_utc: Optional[datetime] = None
    end_time_utc: Optional[datetime] = None
    name: Optional[str] = None
    description: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
                    "agent_id": "b2c3d4e5-f6a7-8901-2345-67890abcdef0",
                    "prompt": "Run daily report",
                    "interval": 1,
                    "unit": "DAYS",
                    "is_active": True,
                    "one_time": False,
                    "created_at": "2024-05-15T10:00:00Z",
                    "last_run_at": "2024-05-14T10:00:00Z",
                    "start_time_utc": "2024-01-01T00:00:00Z",
                    "end_time_utc": None,
                    "name": "Daily Report Generation",
                    "description": "Generates the daily sales report.",
                }
            ]
        }
    }


class ScheduleCreateRequest(BaseModel):
    """Request model for creating a new schedule."""

    prompt: str = Field(..., description="The prompt for the scheduled task.")
    interval: int = Field(
        ..., gt=0, description="The interval value for the schedule (e.g., 5 for every 5 minutes)."
    )
    unit: ScheduleUnit = Field(..., description="The unit for the interval (MINUTES, HOURS, DAYS).")
    is_active: bool = Field(True, description="Whether the schedule is active upon creation.")
    one_time: bool = Field(False, description="Whether this is a one-time schedule.")
    start_time_utc: Optional[datetime] = Field(
        None, description="Optional UTC start time for the schedule."
    )
    end_time_utc: Optional[datetime] = Field(
        None, description="Optional UTC end time for the schedule."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Send weekly newsletter",
                    "interval": 1,
                    "unit": "DAYS",  # Assuming ScheduleUnit.DAYS, will be "DAYS" as string
                    "is_active": True,
                    "one_time": False,
                    "start_time_utc": "2024-06-01T09:00:00Z",
                    "name": "Weekly Newsletter",
                }
            ]
        }
    }


class ScheduleUpdateRequest(BaseModel):
    """Request model for updating an existing schedule. All fields are optional."""

    prompt: Optional[str] = Field(None, description="The prompt for the scheduled task.")
    interval: Optional[int] = Field(None, gt=0, description="The interval value for the schedule.")
    unit: Optional[ScheduleUnit] = Field(None, description="The unit for the interval.")
    is_active: Optional[bool] = Field(None, description="Whether the schedule is active.")
    one_time: Optional[bool] = Field(None, description="Whether this is a one-time schedule.")
    start_time_utc: Optional[datetime] = Field(
        None, description="Optional UTC start time for the schedule."
    )
    end_time_utc: Optional[datetime] = Field(
        None, description="Optional UTC end time for the schedule."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Run nightly backup",
                    "is_active": False,
                    "interval": 8,
                    "unit": "HOURS",
                }
            ]
        }
    }


class ScheduleListResponse(BaseModel):
    """Response model for a paginated list of schedules."""

    total: int = Field(..., description="Total number of schedules.")
    page: int = Field(..., description="Current page number.")
    per_page: int = Field(..., description="Number of schedules per page.")
    schedules: List[ScheduleResponse] = Field(
        ..., description="List of schedules for the current page."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "total": 100,
                    "page": 1,
                    "per_page": 10,
                    "schedules": [
                        {
                            "id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
                            "agent_id": "b2c3d4e5-f6a7-8901-2345-67890abcdef0",
                            "prompt": "Run daily report",
                            "interval": 1,
                            "unit": "DAYS",
                            "is_active": True,
                            "one_time": False,
                            "created_at": "2024-05-15T10:00:00Z",
                        }
                    ],
                }
            ]
        }
    }


class ExecutionVariable(BaseModel):
    """A single execution variable."""

    key: str = Field(..., description="The key of the execution variable.")
    value: str = Field(..., description="The value of the execution variable.")
    type: str = Field(..., description="The type of the execution variable.")


class ExecutionVariableUpdate(BaseModel):
    """Data for updating an execution variable."""

    value: str = Field(..., description="The new value of the execution variable.")
    type: Optional[str] = Field(None, description="The new type of the execution variable.")


class ExecutionVariablesResponse(BaseModel):
    """Response model for a list of execution variables."""

    execution_variables: List[ExecutionVariable] = Field(
        ..., description="A list of execution variables."
    )
