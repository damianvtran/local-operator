"""Types module containing enums and type definitions used throughout the local-operator package."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ConversationRole(str, Enum):
    """Enum representing the different roles in a conversation with an AI model.

    Used to track who sent each message in the conversation history.
    Maps to the standard roles used by LangChain message types.
    """

    SYSTEM = "system"  # System prompts that define the AI's behavior
    USER = "user"  # Messages from the human user
    ASSISTANT = "assistant"  # Responses from the AI assistant
    HUMAN = "human"  # Alias for USER, supported by some LangChain models
    AI = "ai"  # Alias for ASSISTANT, supported by some LangChain models
    FUNCTION = "function"  # Function call messages in LangChain
    TOOL = "tool"  # Tool/plugin response messages in LangChain
    CHAT = "chat"  # Generic chat messages in LangChain


class ActionType(str, Enum):
    """Enum representing the different types of actions that can be taken in a conversation.

    Used to track the type of action being taken in a conversation.
    """

    CODE = "CODE"
    WRITE = "WRITE"
    EDIT = "EDIT"
    DONE = "DONE"
    ASK = "ASK"
    BYE = "BYE"
    READ = "READ"

    def __str__(self) -> str:
        """Return the string representation of the ActionType enum.

        Returns:
            str: The value of the ActionType enum.
        """
        return self.value


class ExecutionType(str, Enum):
    """Enum representing the different types of execution in a conversation workflow.

    Used to track the execution phase within the agent's thought process:
    - PLAN: Initial planning phase where the agent outlines its approach
    - ACTION: Execution of specific actions like running code or accessing resources
    - REFLECTION: Analysis and evaluation of previous actions and their results
    - RESPONSE: Final response generation based on the execution results
    - SECURITY_CHECK: Security check phase where the agent checks the safety of the code
    - CLASSIFICATION: Classification phase where the agent classifies the user's request
    - SYSTEM: An automatic static response from the system, such as an action cancellation.
    """

    PLAN = "plan"
    ACTION = "action"
    REFLECTION = "reflection"
    RESPONSE = "response"
    SECURITY_CHECK = "security_check"
    CLASSIFICATION = "classification"
    SYSTEM = "system"
    USER_INPUT = "user_input"
    NONE = "none"


class ConversationRecord(BaseModel):
    """A record of a conversation with an AI model.

    Attributes:
        role (ConversationRole): The role of the sender of the message
        content (str): The content of the message
        should_summarize (bool): Whether this message should be summarized
        ephemeral (bool): Whether this message is temporary/ephemeral
        summarized (bool): Whether this message has been summarized
        is_system_prompt (bool): Whether this message is a system prompt
        timestamp (datetime): When this message was created
        files (List[str]): The files that were created or modified during the code execution

    Methods:
        to_dict(): Convert the record to a dictionary format
        from_dict(data): Create a ConversationRecord from a dictionary
    """

    content: str = Field(default="")
    role: ConversationRole = Field(default=ConversationRole.ASSISTANT)
    should_summarize: Optional[bool] = True
    ephemeral: Optional[bool] = False
    summarized: Optional[bool] = False
    is_system_prompt: Optional[bool] = False
    timestamp: Optional[datetime] = None
    files: Optional[List[str]] = None

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Convert the conversation record to a dictionary format compatible with LangChain.

        Returns:
            dict: Dictionary with role and content fields for LangChain
        """
        return {
            "role": self.role.value,
            "content": self.content,
            "should_summarize": str(self.should_summarize),
            "ephemeral": str(self.ephemeral),
            "summarized": str(self.summarized),
            "is_system_prompt": str(self.is_system_prompt),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "files": self.files,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert the conversation record to a dictionary.

        Returns:
            dict: Dictionary representation with string values for role and booleans
        """
        return {
            "role": self.role.value,
            "content": self.content,
            "should_summarize": str(self.should_summarize),
            "ephemeral": str(self.ephemeral),
            "summarized": str(self.summarized),
            "is_system_prompt": str(self.is_system_prompt),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "files": self.files,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationRecord":
        """Create a ConversationRecord from a dictionary.

        Args:
            data (dict): Dictionary containing conversation record data

        Returns:
            ConversationRecord: New instance created from dictionary data
        """
        return cls(
            role=ConversationRole(data["role"]),
            content=data["content"],
            should_summarize=data.get("should_summarize", "true").lower() == "true",
            ephemeral=data.get("ephemeral", "false").lower() == "true",
            summarized=data.get("summarized", "false").lower() == "true",
            is_system_prompt=data.get("is_system_prompt", "false").lower() == "true",
            timestamp=(
                datetime.fromisoformat(data.get("timestamp", None))
                if data.get("timestamp")
                else None
            ),
            files=data.get("files", None),
        )


class ResponseJsonSchema(BaseModel):
    """Schema for JSON responses from the language model.

    Attributes:
        response (str): Natural language response explaining the actions being taken
        code (str): Python code to be executed to achieve the current goal
        action (str): Action to take next - one of: CONTINUE, DONE, ASK, BYE
        learnings (str): Learnings from the current step
        content (str): Content to be written to a file
        file_path (str): Path to the file to be written to
        mentioned_files (List[str]): List of files mentioned in the response
        replacements (List[Dict[str, str]]): List of replacements to be made in the file
    """

    response: str
    code: str = Field(default="")
    content: str = Field(default="")
    file_path: str = Field(default="")
    mentioned_files: List[str] = Field(default_factory=list)
    replacements: List[Dict[str, str]] = Field(default_factory=list)
    action: ActionType
    learnings: str = Field(default="")


class ProcessResponseStatus(str, Enum):
    """Status codes for process_response results."""

    SUCCESS = "success"
    CANCELLED = "cancelled"
    ERROR = "error"
    INTERRUPTED = "interrupted"
    CONFIRMATION_REQUIRED = "confirmation_required"
    NONE = "none"


class ProcessResponseOutput:
    """Output structure for process_response results.

    Attributes:
        status (ProcessResponseStatus): Status of the response processing
        message (str): Descriptive message about the processing result
    """

    def __init__(self, status: ProcessResponseStatus, message: str):
        self.status = status
        self.message = message


class CodeExecutionResult(BaseModel):
    """Represents the result of a code execution.

    Attributes:
        id (str): The unique identifier for the code execution
        stdout (str): The standard output from the code execution.
        stderr (str): The standard error from the code execution.
        logging (str): Any logging output generated during the code execution.
        message (str): The message to display to the user about the code execution.
        code (str): The code that was executed.
        formatted_print (str): The formatted print output from the code execution.
        role (ConversationRole): The role of the message sender (user/assistant/system)
        status (ProcessResponseStatus): The status of the code execution
        timestamp (datetime): The timestamp of the code execution
        files (List[str]): The files that were created or modified during the code execution
        action (ActionType): The action that was taken during the code execution
        execution_type (ExecutionType): The type of execution that was performed
        task_classification (str): The classification of the task that was performed
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    stdout: str = Field(default="")
    stderr: str = Field(default="")
    logging: str = Field(default="")
    message: str = Field(default="")
    code: str = Field(default="")
    formatted_print: str = Field(default="")
    role: ConversationRole = Field(default=ConversationRole.ASSISTANT)
    status: ProcessResponseStatus = Field(default=ProcessResponseStatus.NONE)
    timestamp: Optional[datetime] = None
    files: List[str] = Field(default_factory=list)
    action: Optional[ActionType] = None
    execution_type: ExecutionType = Field(default=ExecutionType.NONE)
    task_classification: str = Field(default="")


class AgentExecutorState(BaseModel):
    """Represents the state of an agent executor.

    Attributes:
        conversation (List[ConversationRecord]): The conversation history
        execution_history (List[CodeExecutionResult]): The execution history
    """

    conversation: List[ConversationRecord]
    execution_history: List[CodeExecutionResult]


class RelativeEffortLevel(str, Enum):
    """Enum representing the relative effort level of a user request.

    Used to track the relative effort level of a user request.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RequestClassification(BaseModel):
    """Represents the classification of a user request.

    Attributes:
        type (str): The type of request
        planning_required (bool): Whether planning is required for the request
        relative_effort (str): The relative effort required for the request
    """

    type: str
    planning_required: bool = Field(default=False)
    relative_effort: RelativeEffortLevel = Field(default=RelativeEffortLevel.LOW)
