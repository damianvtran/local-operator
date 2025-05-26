import asyncio
import json
from typing import AsyncGenerator

from langchain_core.messages import BaseMessage

from local_operator.types import ConversationRole

USER_MOCK_RESPONSES = {
    "hello": {
        "response": "Hello! I am the test model.",
        "code": "",
        "action": "DONE",
        "learnings": "",
        "content": "",
        "file_path": "",
        "mentioned_files": [],
        "replacements": [],
    },
    "please proceed according to your plan": """Ok, I will now print hello world.

<action_response>
<action>CODE</action>
<code>print("Hello World")</code>
<response>I will execute a simple Python script to print "Hello World".</response>
<learnings>I learned about the Python print function.</learnings>
</action_response>
    """,
    '<code>print("Hello World")</code>': {
        "response": 'I will execute a simple Python script to print "Hello World".',
        "code": 'print("Hello World")',
        "action": "CODE",
        "learnings": "",
        "content": "",
        "file_path": "",
        "mentioned_files": [],
        "replacements": [],
    },
    "print hello world": """
<type>conversation</type>
<planning_required>true</planning_required>
<relative_effort>low</relative_effort>
<subject_change>false</subject_change>
    """,
    "<type>conversation</type>\n<planning_required>true</planning_required>\n<relative_effort>low</relative_effort>\n<subject_change>false</subject_change>": {  # noqa: E501
        "type": "conversation",
        "planning_required": True,
        "relative_effort": "low",
        "subject_change": False,
    },
    "think aloud about what you will need to do": (
        "I will need to print 'Hello World' to the console."
    ),
    "think aloud about what you did and the outcome": ("I printed 'Hello World' to the console."),
    "hello world": "I have printed 'Hello World' to the console.",
    "please summarize": "[SUMMARY] this is a summary",
}


class ChatMock:
    """A test model that returns predefined responses for specific inputs."""

    temperature: float | None
    model: str | None
    model_name: str | None
    api_key: str | None
    base_url: str | None
    max_tokens: int | None
    top_p: float | None
    frequency_penalty: float | None
    presence_penalty: float | None

    def __init__(self):
        self.temperature = 0.3
        self.model = "test-model"
        self.model_name = "test-model"
        self.api_key = None
        self.base_url = None
        self.max_tokens = 4096
        self.top_p = 0.9
        self.frequency_penalty = 0.0
        self.presence_penalty = 0.0

    async def ainvoke(self, messages):
        """Mock ainvoke method that returns predefined responses.

        Args:
            messages: List of message dicts with role and content

        Returns:
            BaseMessage instance containing the response
        """
        if not messages:
            raise ValueError("No messages provided to ChatMock")

        # Get last user message
        user_message = ""
        for msg in reversed(list(messages)):
            if (
                msg.get("role") == ConversationRole.USER.value
                and "agent_heads_up_display"
                not in msg.get("content", [])[0].get("text", "").lower()
            ):
                user_message = msg.get("content", [])[0].get("text", "") or ""
                break

        # Find best matching response
        user_message_lower = user_message.lower()
        best_match = None
        max_match_length = 0

        for key in USER_MOCK_RESPONSES:
            key_lower = key.lower()
            if key_lower in user_message_lower and len(key_lower) > max_match_length:
                best_match = key
                max_match_length = len(key_lower)

        if not best_match:
            if len(user_message) > 300:
                truncated_user_message = user_message[:300] + "..."
            else:
                truncated_user_message = user_message

            print(f"No mock response for message: {truncated_user_message}")
            raise ValueError(f"No mock response for message: {truncated_user_message}")

        if isinstance(USER_MOCK_RESPONSES[best_match], dict):
            response_content = json.dumps(USER_MOCK_RESPONSES[best_match])
        else:
            response_content = USER_MOCK_RESPONSES[best_match]

        return BaseMessage(content=response_content, type=ConversationRole.ASSISTANT.value)

    def invoke(self, messages):
        """Synchronous version of ainvoke."""
        return asyncio.run(self.ainvoke(messages))

    def stream(self, messages):
        """Mock stream method that yields chunks of the response."""
        response = self.invoke(messages)
        yield response

    async def astream(self, messages) -> AsyncGenerator[BaseMessage, None]:
        """Mock astream method that asynchronously yields chunks of the response."""
        response = await self.ainvoke(messages)
        yield response


class ChatNoop:
    """A test model that returns an empty response."""

    temperature: float | None
    model: str | None
    model_name: str | None
    api_key: str | None
    base_url: str | None
    max_tokens: int | None
    top_p: float | None
    frequency_penalty: float | None
    presence_penalty: float | None

    def __init__(self):
        self.temperature = 0.3
        self.model = "noop-model"
        self.model_name = "noop-model"
        self.api_key = None
        self.base_url = None
        self.max_tokens = 4096
        self.top_p = 0.9
        self.frequency_penalty = 0.0
        self.presence_penalty = 0.0

    async def ainvoke(self, messages):
        """Async version that returns an empty response."""
        return BaseMessage(content="", type=ConversationRole.ASSISTANT.value)

    def invoke(self, messages):
        """Synchronous version that returns an empty response."""
        return asyncio.run(self.ainvoke(messages))

    def stream(self, messages):
        """Mock stream method that yields an empty response."""
        response = self.invoke(messages)
        yield response

    async def astream(self, messages):
        """Mock astream method that asynchronously yields an empty response."""
        response = await self.ainvoke(messages)
        yield response
