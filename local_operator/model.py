from typing import Union

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from local_operator.types import ConversationRole

USER_MOCK_RESPONSES = {
    "hello": "Hello! I am the test model.",
    "print hello world": """Sure, I will execute a simple Python script to print "Hello World".
```python
print("Hello World")
```
""",
}

SYSTEM_MOCK_RESPONSES = {
    "Hello World": "I have printed 'Hello World' to the console. [DONE]",
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
                return BaseMessage(content=response, type=ConversationRole.ASSISTANT.value)
        else:
            for response in SYSTEM_MOCK_RESPONSES:
                if response in code_execution_response:
                    return BaseMessage(
                        content=SYSTEM_MOCK_RESPONSES[response],
                        type=ConversationRole.ASSISTANT.value,
                    )

        # Pass through the last message if no match found
        return BaseMessage(
            content=messages[-1].get("content", ""),
            type=ConversationRole.ASSISTANT.value,
        )

    def invoke(self, messages):
        """Synchronous version of ainvoke."""
        import asyncio

        return asyncio.run(self.ainvoke(messages))

    def stream(self, messages):
        """Mock stream method that yields chunks of the response."""
        response = self.invoke(messages)
        yield response

    async def astream(self, messages):
        """Mock astream method that asynchronously yields chunks of the response."""
        response = await self.ainvoke(messages)
        yield response


def configure_model(
    hosting: str, model: str, credential_manager
) -> Union[ChatOpenAI, ChatOllama, ChatAnthropic, ChatMock, None]:
    """Configure and return the appropriate model based on hosting platform.

    Args:
        hosting (str): Hosting platform (deepseek, openai, anthropic, ollama, or noop)
        model (str): Model name to use
        credential_manager: CredentialManager instance for API key management

    Returns:
        Union[ChatOpenAI, ChatOllama, ChatAnthropic, None]: Configured model instance
    """
    if not hosting:
        raise ValueError("Hosting is required")

    if hosting == "deepseek":
        base_url = "https://api.deepseek.com/v1"

        if not model:
            model = "deepseek-chat"

        api_key = credential_manager.get_credential("DEEPSEEK_API_KEY")
        if not api_key:
            api_key = credential_manager.prompt_for_credential("DEEPSEEK_API_KEY")

        return ChatOpenAI(
            api_key=SecretStr(api_key),
            temperature=0.3,
            base_url=base_url,
            model=model,
        )
    elif hosting == "openai":
        if not model:
            model = "gpt-4o"

        api_key = credential_manager.get_credential("OPENAI_API_KEY")
        if not api_key:
            api_key = credential_manager.prompt_for_credential("OPENAI_API_KEY")

        temperature = 0.3

        # The o models only support temperature 1.0
        if model.startswith("o1") or model.startswith("o3"):
            temperature = 1.0

        return ChatOpenAI(
            api_key=SecretStr(api_key),
            temperature=temperature,
            model=model,
        )
    elif hosting == "anthropic":
        if not model:
            model = "claude-3-5-sonnet-20240620"

        api_key = credential_manager.get_credential("ANTHROPIC_API_KEY")
        if not api_key:
            api_key = credential_manager.prompt_for_credential("ANTHROPIC_API_KEY")
        return ChatAnthropic(
            api_key=SecretStr(api_key),
            temperature=0.3,
            model_name=model,
            timeout=None,
            stop=None,
        )
    elif hosting == "kimi":
        if not model:
            model = "moonshot-v1-32k"

        api_key = credential_manager.get_credential("KIMI_API_KEY")
        if not api_key:
            api_key = credential_manager.prompt_for_credential("KIMI_API_KEY")

        return ChatOpenAI(
            api_key=SecretStr(api_key),
            temperature=0.3,
            model=model,
            base_url="https://api.moonshot.cn/v1",
        )
    elif hosting == "alibaba":
        if not model:
            model = "qwen-plus"

        api_key = credential_manager.get_credential("ALIBABA_CLOUD_API_KEY")
        if not api_key:
            api_key = credential_manager.prompt_for_credential("ALIBABA_CLOUD_API_KEY")

        return ChatOpenAI(
            api_key=SecretStr(api_key),
            temperature=0.3,
            model=model,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
    elif hosting == "ollama":
        if not model:
            raise ValueError("Model is required for ollama hosting")

        return ChatOllama(
            model=model,
            temperature=0.3,
        )
    elif hosting == "test":
        return ChatMock()
    elif hosting == "noop":
        # Useful for testing, will create a dummy operator
        return None
    else:
        raise ValueError(f"Unsupported hosting platform: {hosting}")
