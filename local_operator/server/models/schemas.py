"""
Pydantic models for the Local Operator API.

This module contains all the Pydantic models used for request and response validation
in the Local Operator API.
"""

from datetime import datetime
from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

# AgentEditFields will be used in the routes module
from local_operator.jobs import JobResult, JobStatus
from local_operator.types import CodeExecutionResult, ConversationRecord


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
    last_message: str = Field(
        "",
        description="The last message sent to the agent. Defaults to ''.",
    )
    last_message_datetime: datetime = Field(
        ...,
        description="The date and time of the last message sent to the agent.",
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
    description: str | None = Field(
        None,
        description="A description of the agent.  Defaults to ''.",
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
    """

    hosting: str
    model: str
    prompt: str
    stream: bool = False
    options: Optional[ChatOptions] = None
    persist_conversation: bool = False
    user_message_id: Optional[str] = None
