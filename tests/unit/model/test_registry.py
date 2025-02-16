import pytest

from local_operator.model.registry import ModelInfo, calculate_cost, get_model_info


def test_model_info_price_must_be_non_negative() -> None:
    """Test that the price_must_be_non_negative validator works correctly."""
    with pytest.raises(ValueError, match="Price must be non-negative."):
        ModelInfo(input_price=-1, output_price=1)
    with pytest.raises(ValueError, match="Price must be non-negative."):
        ModelInfo(input_price=1, output_price=-1)
    # Should not raise an error
    ModelInfo(input_price=0, output_price=0)
    ModelInfo(input_price=1, output_price=1)


def test_calculate_cost() -> None:
    """Test that the calculate_cost function works correctly."""
    model_info = ModelInfo(input_price=1, output_price=2)
    input_tokens = 1000
    output_tokens = 2000
    expected_cost = (input_tokens / 1_000_000) * model_info.input_price + (
        output_tokens / 1_000_000
    ) * model_info.output_price
    assert calculate_cost(model_info, input_tokens, output_tokens) == pytest.approx(expected_cost)

    # Test with zero tokens
    assert calculate_cost(model_info, 0, 0) == 0.0


def test_get_model_info() -> None:
    """Test that the get_model_info function works correctly."""
    # Test Anthropic
    model_info = get_model_info("anthropic", "claude-3-5-sonnet-20241022")
    assert model_info.max_tokens == 8192
    with pytest.raises(ValueError, match="Model unknown not found for Anthropic hosting."):
        get_model_info("anthropic", "unknown")

    # Test Google
    model_info = get_model_info("google", "gemini-2.0-flash-001")
    assert model_info.max_tokens == 8192
    with pytest.raises(ValueError, match="Model unknown not found for google hosting."):
        get_model_info("google", "unknown")

    # Test OpenAI
    model_info = get_model_info("openai", "any")
    assert model_info.max_tokens == -1

    # Test OpenRouter
    model_info = get_model_info("openrouter", "any")
    assert model_info.max_tokens == 8192

    # Test Alibaba
    model_info = get_model_info("alibaba", "qwen2.5-coder-32b-instruct")
    assert model_info.max_tokens == 8192
    with pytest.raises(ValueError, match="Model unknown not found for Alibaba hosting."):
        get_model_info("alibaba", "unknown")

    # Test Mistral
    model_info = get_model_info("mistral", "mistral-large-2411")
    assert model_info.max_tokens == 131_000
    with pytest.raises(ValueError, match="Model unknown not found for Mistral hosting."):
        get_model_info("mistral", "unknown")

    # Test Unsupported hosting provider
    with pytest.raises(ValueError, match="Unsupported hosting provider: unknown"):
        get_model_info("unknown", "any")
