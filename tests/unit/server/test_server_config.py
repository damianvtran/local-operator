"""
Tests for the configuration endpoints of the FastAPI server.

This module contains tests for configuration-related functionality, including
retrieving and updating configuration settings and system prompt.
"""

from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from local_operator.server.app import app
from local_operator.server.models.schemas import ConfigUpdate, SystemPromptUpdate


@pytest.mark.asyncio
async def test_get_config_success(test_app_client, mock_config_manager):
    """Test retrieving configuration successfully."""
    response = await test_app_client.get("/v1/config")

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "Configuration retrieved successfully"
    result = data.get("result")
    assert "version" in result
    assert "metadata" in result
    assert "values" in result


@pytest.mark.asyncio
async def test_update_config_success(test_app_client, mock_config_manager):
    """Test updating configuration successfully."""
    update_payload = ConfigUpdate(
        conversation_length=150,
        detail_length=50,
        max_learnings_history=50,
        hosting="openrouter",
        model_name="openai/gpt-4o-mini",
        auto_save_conversation=True,
    )

    response = await test_app_client.patch("/v1/config", json=update_payload.model_dump())

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "Configuration updated successfully"
    result = data.get("result")
    assert "version" in result
    assert "metadata" in result
    assert "values" in result
    values = result.get("values")
    assert values.get("conversation_length") == 150
    assert values.get("detail_length") == 50
    assert values.get("hosting") == "openrouter"
    assert values.get("model_name") == "openai/gpt-4o-mini"
    assert values.get("auto_save_conversation") is True


@pytest.mark.asyncio
async def test_update_config_partial(test_app_client, mock_config_manager):
    """Test updating only some configuration fields."""
    # First get the current config
    original_config = {
        "conversation_length": 200,
        "detail_length": 50,
        "max_learnings_history": 50,
        "hosting": "openrouter",
        "model_name": "openai/gpt-4o-mini",
        "auto_save_conversation": True,
    }
    mock_config_manager.update_config(original_config)

    update_payload = ConfigUpdate(
        conversation_length=100,
        detail_length=None,
        max_learnings_history=None,
        hosting=None,
        model_name=None,
        auto_save_conversation=None,
    )
    response = await test_app_client.patch("/v1/config", json=update_payload.model_dump())

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    result = data.get("result")
    values = result.get("values")

    # Check that only the specified field was updated
    assert values.get("conversation_length") == 100
    assert values.get("detail_length") == 50
    assert values.get("hosting") == "openrouter"
    assert values.get("model_name") == "openai/gpt-4o-mini"


@pytest.mark.asyncio
async def test_update_config_empty(test_app_client, mock_config_manager):
    """Test updating configuration with no fields provided."""
    update_payload = ConfigUpdate(
        conversation_length=None,
        detail_length=None,
        max_learnings_history=None,
        hosting=None,
        model_name=None,
        auto_save_conversation=None,
    )
    response = await test_app_client.patch("/v1/config", json=update_payload.model_dump())

    assert response.status_code == 400
    data = response.json()
    assert "No valid update fields provided" in data.get("detail", "")


@pytest.mark.asyncio
async def test_update_config_invalid_values(test_app_client, mock_config_manager):
    """Test updating configuration with invalid values."""
    # Create a transport and client directly to test error handling
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Simulate an error by patching the config manager to raise an exception
        app.state.config_manager.update_config = lambda _: (_ for _ in ()).throw(
            ValueError("Invalid configuration value")
        )

        update_payload = ConfigUpdate(
            conversation_length=150,
            detail_length=50,
            max_learnings_history=50,
            hosting="openrouter",
            model_name="openai/gpt-4o-mini",
            auto_save_conversation=True,
        )
        response = await ac.patch("/v1/config", json=update_payload.model_dump())

        assert response.status_code == 500
        data = response.json()
        assert "Error updating configuration" in data.get("detail", "")


@pytest.mark.asyncio
async def test_get_system_prompt_success(test_app_client):
    """Test retrieving system prompt successfully."""
    test_content = "You are Local Operator, an AI assistant..."
    test_timestamp = 1609459200.0  # 2021-01-01 00:00:00

    # Mock the file operations
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.read_text", return_value=test_content),
        patch("pathlib.Path.stat") as mock_stat,
    ):
        # Mock the stat result to return a fixed timestamp
        mock_stat_result = MagicMock()
        mock_stat_result.st_mtime = test_timestamp
        mock_stat.return_value = mock_stat_result

        response = await test_app_client.get("/v1/config/system-prompt")

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "System prompt retrieved successfully"
    result = data.get("result")
    assert result.get("content") == test_content
    assert "last_modified" in result


@pytest.mark.asyncio
async def test_get_system_prompt_not_found(test_app_client):
    """Test retrieving system prompt when file doesn't exist."""
    with patch("pathlib.Path.exists", return_value=False):
        response = await test_app_client.get("/v1/config/system-prompt")

    assert response.status_code == 204
    assert not response.content  # No content for 204 response


@pytest.mark.asyncio
async def test_get_system_prompt_error(test_app_client):
    """Test error handling when retrieving system prompt."""
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.read_text", side_effect=Exception("Test error")),
    ):
        response = await test_app_client.get("/v1/config/system-prompt")

    assert response.status_code == 500
    data = response.json()
    assert "Error retrieving system prompt" in data.get("detail", "")


@pytest.mark.asyncio
async def test_update_system_prompt_success(test_app_client, tmp_path, monkeypatch):
    """A real file under a real config dir, not a mock.

    Mocking the module attribute pinned the implementation's SHAPE (a
    module-level constant) rather than its behaviour, and that shape was the
    defect: it froze the path at import and ignored
    ``LOCAL_OPERATOR_CONFIG_DIR``, so this endpoint and the session loader
    stopped being the same file the moment the override was set.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    test_content = "You are Local Operator, an AI assistant with new capabilities..."
    update_payload = SystemPromptUpdate(content=test_content)

    response = await test_app_client.patch(
        "/v1/config/system-prompt", json=update_payload.model_dump()
    )

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "System prompt updated successfully"
    assert data.get("result", {}).get("content") == test_content
    # Written where the override says, and readable back through the endpoint.
    assert (tmp_path / "system_prompt.md").read_text(encoding="utf-8") == test_content


@pytest.mark.asyncio
async def test_the_endpoint_and_the_session_loader_read_one_file(
    test_app_client, tmp_path, monkeypatch
):
    """The feature's central promise: the desktop Settings box, this endpoint
    and every session's system prompt are ONE file. A hardcoded home path made
    that false under an override, silently — the endpoint would report content
    no session ever loaded."""
    from local_operator.session_factory import load_user_instructions

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    content = "- Always use conventional commits."
    await test_app_client.patch(
        "/v1/config/system-prompt",
        json=SystemPromptUpdate(content=content).model_dump(),
    )

    assert load_user_instructions() == content

    got = await test_app_client.get("/v1/config/system-prompt")
    assert got.status_code == 200
    assert got.json()["result"]["content"] == content


@pytest.mark.asyncio
async def test_update_system_prompt_error(test_app_client, tmp_path, monkeypatch):
    """Test error handling when updating system prompt."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    update_payload = SystemPromptUpdate(content="You are Local Operator, an AI assistant...")

    with patch("local_operator.server.routes.config.system_prompt_file") as mock_path:
        mock_path.return_value.write_text.side_effect = Exception("Test error")

        response = await test_app_client.patch(
            "/v1/config/system-prompt", json=update_payload.model_dump()
        )

    assert response.status_code == 500
    data = response.json()
    assert "Error updating system prompt" in data.get("detail", "")
