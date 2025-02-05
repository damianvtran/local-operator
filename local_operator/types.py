"""Types module containing enums and type definitions used throughout the local-operator package."""

from enum import Enum

from pydantic import BaseModel


class ConversationRole(Enum):
    """Enum representing the different roles in a conversation with an AI model.

    Used to track who sent each message in the conversation history.
    """

    SYSTEM = "system"  # System prompts that define the AI's behavior
    USER = "user"  # Messages from the human user
    ASSISTANT = "assistant"  # Responses from the AI assistant


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
    """

    previous_step_success: bool
    previous_goal: str
    current_goal: str
    next_goal: str
    response: str
    code: str
    action: str
    learnings: str
