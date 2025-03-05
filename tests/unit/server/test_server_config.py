"""
Tests for the configuration endpoints of the FastAPI server.

This module contains tests for configuration-related functionality, including
retrieving and updating configuration settings.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from local_operator.server.app import app
from local_operator.server.models.schemas import ConfigUpdate


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
