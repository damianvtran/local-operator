import asyncio

from langchain_core.messages import BaseMessage

from local_operator.types import ConversationRole, ResponseJsonSchema

USER_MOCK_RESPONSES = {
    "hello": ResponseJsonSchema(
        previous_step_success=True,
        previous_goal="",
        current_goal="Greet the user",
        next_goal="",
        response="Hello! I am the test model.",
        code="",
        action="DONE",
        learnings="",
        plan="",
    ),
    "print hello world": ResponseJsonSchema(
        previous_step_success=True,
        previous_goal="",
        current_goal="Print Hello World",
        next_goal="",
        response='Sure, I will execute a simple Python script to print "Hello World".',
        code='print("Hello World")',
        action="CONTINUE",
        learnings="",
        plan="",
    ),
}

SYSTEM_MOCK_RESPONSES = {
    "Hello World": ResponseJsonSchema(
        previous_step_success=True,
        previous_goal="Print Hello World",
        current_goal="Complete task",
        next_goal="",
        response="I have printed 'Hello World' to the console.",
        code="",
        action="DONE",
        learnings="",
        plan="",
    )
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

        # Only consider the last message coming from the user
        user_message = ""
        user_message_index = -1
        for index, msg in reversed(list(enumerate(messages))):
            if msg.get("role") == ConversationRole.USER.value:
                user_message = msg.get("content", "")
                user_message_index = index
                break

        user_message_lower = user_message.lower()

        code_execution_response = ""
        code_execution_response_index = -1
        for index, msg in reversed(list(enumerate(messages))):
            if msg.get(
                "role"
            ) == ConversationRole.SYSTEM.value and "Code execution output" in msg.get(
                "content", ""
            ):
                code_execution_response = msg.get("content", "")
                code_execution_response_index = index
                break

        if user_message_index > code_execution_response_index:
            if user_message_lower in USER_MOCK_RESPONSES:
                response = USER_MOCK_RESPONSES[user_message_lower]
                return BaseMessage(
                    content=response.model_dump_json(),
                    type=ConversationRole.ASSISTANT.value,
                )
        else:
            for response in SYSTEM_MOCK_RESPONSES:
                if response in code_execution_response:
                    return BaseMessage(
                        content=SYSTEM_MOCK_RESPONSES[response].model_dump_json(),
                        type=ConversationRole.ASSISTANT.value,
                    )

        # Pass through the last message if no match found
        return BaseMessage(
            content=messages[-1].get("content", ""),
            type=ConversationRole.ASSISTANT.value,
        )

    def invoke(self, messages):
        """Synchronous version of ainvoke."""
        return asyncio.run(self.ainvoke(messages))

    def stream(self, messages):
        """Mock stream method that yields chunks of the response."""
        response = self.invoke(messages)
        yield response

    async def astream(self, messages):
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
