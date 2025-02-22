"""Types module containing enums and type definitions used throughout the local-operator package."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


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

    RESEARCH = "RESEARCH"
    CODE = "CODE"
    WRITE = "WRITE"
    EDIT = "EDIT"
    CHECK = "CHECK"
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


class ConversationRecord(BaseModel):
    """A record of a conversation with an AI model.

    Attributes:
        role (ConversationRole): The role of the sender of the message
        content (str): The content of the message
        should_summarize (bool): Whether this message should be summarized
        ephemeral (bool): Whether this message is temporary/ephemeral
        summarized (bool): Whether this message has been summarized

    Methods:
        to_dict(): Convert the record to a dictionary format
        from_dict(data): Create a ConversationRecord from a dictionary
    """

    content: str
    role: ConversationRole
    should_summarize: Optional[bool] = True
    ephemeral: Optional[bool] = False
    summarized: Optional[bool] = False
    is_system_prompt: Optional[bool] = False

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
        )


class ResponseJsonSchema(BaseModel):
    """Schema for JSON responses from the language model.

    Attributes:
        previous_step_success (bool): Whether the previous step was successful
        previous_goal (str): The goal that was attempted in the previous step
        current_goal (str): The goal being attempted in the current step
        next_goal (str): The planned goal for the next step
        response (str): Natural language response explaining the actions being taken
        code (str): Python code to be executed to achieve the current goal
        action (str): Action to take next - one of: CONTINUE, DONE, ASK, BYE
        learnings (str): Learnings from the current step
        plan (str): Plan for the next step
    """

    previous_step_success: bool
    previous_goal: str
    current_goal: str
    next_goal: str
    response: str
    code: str
    content: str
    file_path: str
    replacements: List[Dict[str, str]]
    action: ActionType
    learnings: str
    plan: str
