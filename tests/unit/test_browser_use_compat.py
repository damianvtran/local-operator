import importlib.util
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from local_operator.tools.general import (
    _BrowserUseLangChainAdapter,
    _prepare_browser_use_llm,
)

HAS_BROWSER_USE_NATIVE_LLM_API = importlib.util.find_spec("browser_use.llm") is not None
pytestmark = pytest.mark.skipif(
    not HAS_BROWSER_USE_NATIVE_LLM_API,
    reason="browser-use native llm API is not available in this environment",
)


class StructuredResult(BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_browser_use_langchain_adapter_basic_completion():
    fake_chat = MagicMock()
    fake_chat.model_name = "fake-openai-model"
    fake_chat.ainvoke = AsyncMock(
        return_value=AIMessage(
            content="done",
            usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        )
    )

    adapter = _BrowserUseLangChainAdapter(fake_chat)
    completion = await adapter.ainvoke([{"role": "user", "content": "hello"}])

    assert completion.completion == "done"
    assert completion.usage is not None
    assert completion.usage.prompt_tokens == 7
    assert completion.usage.completion_tokens == 3
    assert completion.usage.total_tokens == 10
    assert adapter.provider in {"langchain", "openai"}
    assert adapter.name == "fake-openai-model"


@pytest.mark.asyncio
async def test_browser_use_langchain_adapter_structured_output_fallback():
    fake_chat = MagicMock()
    fake_chat.model_name = "fake-structured-model"
    fake_chat.with_structured_output.side_effect = AttributeError("not supported")
    fake_chat.ainvoke = AsyncMock(return_value=AIMessage(content='{"answer": "ok"}'))

    adapter = _BrowserUseLangChainAdapter(fake_chat)
    completion = await adapter.ainvoke(
        [{"role": "user", "content": "Return JSON"}],
        output_format=StructuredResult,
    )

    assert isinstance(completion.completion, StructuredResult)
    assert completion.completion.answer == "ok"


def test_prepare_browser_use_llm_passthrough_for_native_like_model():
    class NativeLikeBrowserUseModel:
        async def ainvoke(self, messages, output_format=None):
            return messages

    NativeLikeBrowserUseModel.__module__ = "browser_use.llm.openai.chat"

    native_model = NativeLikeBrowserUseModel()
    assert _prepare_browser_use_llm(native_model) is native_model
