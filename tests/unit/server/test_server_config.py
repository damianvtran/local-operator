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
    data = response.json()
    assert data.get("status") == 204
    assert data.get("message") == "System prompt retrieved, system prompt is empty"
    result = data.get("result")
    assert result.get("content") == ""
    assert result.get("last_modified") == ""


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
async def test_update_system_prompt_success(test_app_client):
    """Test updating system prompt successfully."""
    test_content = "You are Local Operator, an AI assistant with new capabilities..."
    test_timestamp = 1609459200.0  # 2021-01-01 00:00:00
    test_iso_timestamp = "2021-01-01T00:00:00"
    update_payload = SystemPromptUpdate(content=test_content)

    # Mock the file operations
    with (
        patch("local_operator.server.routes.config.SYSTEM_PROMPT_FILE") as mock_file,
        patch("local_operator.server.routes.config.datetime") as mock_datetime,
    ):
        # Configure the mock file
        mock_file.parent.mkdir.return_value = None
        mock_file.write_text.return_value = None
        mock_stat_result = MagicMock()
        mock_stat_result.st_mtime = test_timestamp
        mock_file.stat.return_value = mock_stat_result

        # Mock datetime to return a datetime object with isoformat method
        mock_dt = MagicMock()
        mock_dt.isoformat.return_value = test_iso_timestamp
        mock_datetime.fromtimestamp.return_value = mock_dt

        response = await test_app_client.patch(
            "/v1/config/system-prompt", json=update_payload.model_dump()
        )

        # Verify the directory was created if needed
        mock_file.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        # Verify the content was written to the file
        mock_file.write_text.assert_called_once_with(test_content, encoding="utf-8")

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "System prompt updated successfully"
    result = data.get("result")
    assert result.get("content") == test_content
    assert result.get("last_modified") == test_iso_timestamp


@pytest.mark.asyncio
async def test_update_system_prompt_error(test_app_client):
    """Test error handling when updating system prompt."""
    test_content = "You are Local Operator, an AI assistant..."
    update_payload = SystemPromptUpdate(content=test_content)

    with patch("local_operator.server.routes.config.SYSTEM_PROMPT_FILE") as mock_file:
        mock_file.write_text.side_effect = Exception("Test error")
        mock_file.parent.mkdir.return_value = None

        response = await test_app_client.patch(
            "/v1/config/system-prompt", json=update_payload.model_dump()
        )

    assert response.status_code == 500
    data = response.json()
    assert "Error updating system prompt" in data.get("detail", "")
