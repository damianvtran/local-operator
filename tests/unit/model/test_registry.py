import pytest

from local_operator.model.registry import ModelInfo, get_model_info


def test_model_info_price_must_be_non_negative() -> None:
    """Test that the price_must_be_non_negative validator works correctly."""
    with pytest.raises(ValueError, match="Price must be non-negative."):
        ModelInfo(
            id="test-model",
            name="test-model",
            description="Mock model",
            input_price=-1,
            output_price=1,
            recommended=True,
        )
    with pytest.raises(ValueError, match="Price must be non-negative."):
        ModelInfo(
            id="test-model",
            name="test-model",
            description="Mock model",
            input_price=1,
            output_price=-1,
            recommended=True,
        )
    # Should not raise an error
    ModelInfo(
        id="test-model",
        name="test-model",
        description="Mock model",
        input_price=0,
        output_price=0,
        recommended=True,
    )
    ModelInfo(
        id="test-model",
        name="test-model",
        description="Mock model",
        input_price=1,
        output_price=1,
        recommended=False,
    )


def test_get_model_info() -> None:
    """Test that the get_model_info function works correctly."""

    # Test Anthropic
    model_info = get_model_info("anthropic", "claude-3-5-sonnet-20241022")
    assert model_info.max_tokens == 8192

    # Test Google
    model_info = get_model_info("google", "gemini-2.0-flash-001")
    assert model_info.context_window == 1_048_576

    # Test OpenAI
    model_info = get_model_info("openai", "gpt-4o")
    assert model_info.max_tokens == 128_000

    # Test OpenRouter
    model_info = get_model_info("openrouter", "any")
    assert model_info.context_window == -1

    # Test Alibaba
    model_info = get_model_info("alibaba", "qwen2.5-coder-32b-instruct")
    assert model_info.context_window == 131_072

    # Test Mistral
    model_info = get_model_info("mistral", "mistral-large-2411")
    assert model_info.max_tokens == 131_000

    # Test Kimi
    model_info = get_model_info("kimi", "moonshot-v1-8k")
    assert model_info.context_window == 8192

    # Test Deepseek
    model_info = get_model_info("deepseek", "deepseek-chat")
    assert model_info.context_window == 64_000

    # Test unknown model
    model_info = get_model_info("anthropic", "unknown_model")
    assert model_info.max_tokens == -1
    assert model_info.context_window == -1

    # Test Unsupported hosting provider
    with pytest.raises(ValueError, match="Unsupported hosting provider: unknown"):
        get_model_info("unknown", "any")
