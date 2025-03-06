"""
Tests for the credential endpoints of the FastAPI server.

This module contains tests for credential-related functionality, including
retrieving and updating credential settings.
"""

from unittest.mock import patch

import pytest

from local_operator.credentials import CredentialManager
from local_operator.server.models.schemas import CredentialUpdate


@pytest.mark.asyncio
async def test_list_credentials_success(test_app_client, mock_credential_manager):
    """Test retrieving credentials list successfully."""
    mock_credential_manager.set_credential("OPENAI_API_KEY", "test-key")
    mock_credential_manager.set_credential("SERPAPI_API_KEY", "test-key2")

    response = await test_app_client.get("/v1/credentials")

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "Credentials retrieved successfully"
    result = data.get("result")
    assert "keys" in result
    assert "OPENAI_API_KEY" in result["keys"]
    assert "SERPAPI_API_KEY" in result["keys"]


@pytest.mark.asyncio
async def test_list_credentials_non_empty_only(test_app_client, mock_credential_manager):
    """Test retrieving only non-empty credentials."""
    # Set some credentials with values and some with empty values
    mock_credential_manager.set_credential("OPENAI_API_KEY", "test-key")
    mock_credential_manager.set_credential("EMPTY_API_KEY", "")
    mock_credential_manager.set_credential("SERPAPI_API_KEY", "test-key2")
    mock_credential_manager.set_credential("ANOTHER_EMPTY_KEY", "")

    response = await test_app_client.get("/v1/credentials")

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "Credentials retrieved successfully"
    result = data.get("result")
    assert "keys" in result

    # Only non-empty credentials should be in the list
    assert "OPENAI_API_KEY" in result["keys"]
    assert "SERPAPI_API_KEY" in result["keys"]
    assert "EMPTY_API_KEY" not in result["keys"]
    assert "ANOTHER_EMPTY_KEY" not in result["keys"]
    assert len(result["keys"]) == 2


@pytest.mark.asyncio
async def test_list_credentials_empty(test_app_client, mock_credential_manager):
    """Test retrieving credentials list when empty."""
    response = await test_app_client.get("/v1/credentials")

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "Credentials retrieved successfully"
    result = data.get("result")
    assert "keys" in result
    assert len(result["keys"]) == 0


@pytest.mark.asyncio
async def test_list_credentials_error(test_app_client, mock_credential_manager):
    """Test error handling when retrieving credentials list."""
    # Mock the open function to raise an exception
    with patch.object(CredentialManager, "get_credentials", side_effect=Exception("Test error")):
        response = await test_app_client.get("/v1/credentials")

    assert response.status_code == 500
    data = response.json()
    assert "Error retrieving credentials" in data.get("detail", "")


@pytest.mark.asyncio
async def test_update_credential_success(test_app_client, mock_credential_manager):
    """Test updating a credential successfully."""
    update_payload = CredentialUpdate(
        key="TEST_API_KEY",
        value="test-value",
    )
    response = await test_app_client.patch("/v1/credentials", json=update_payload.model_dump())

    assert mock_credential_manager.get_credential("TEST_API_KEY").get_secret_value() == "test-value"

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "Credential updated successfully"


@pytest.mark.asyncio
async def test_update_credential_empty_key(test_app_client, mock_credential_manager):
    """Test updating a credential with an empty key."""
    update_payload = CredentialUpdate(
        key="",
        value="test-value",
    )
    response = await test_app_client.patch("/v1/credentials", json=update_payload.model_dump())

    assert response.status_code == 400
    data = response.json()
    assert "Credential key cannot be empty" in data.get("detail", "")


@pytest.mark.asyncio
async def test_update_credential_error(test_app_client, mock_credential_manager):
    """Test error handling when updating a credential."""
    # Mock the set_credential method to raise an exception
    with patch.object(
        mock_credential_manager, "set_credential", side_effect=Exception("Test error")
    ):
        update_payload = CredentialUpdate(
            key="TEST_API_KEY",
            value="test-value",
        )
        response = await test_app_client.patch("/v1/credentials", json=update_payload.model_dump())

    assert response.status_code == 500
    data = response.json()
    assert "Error updating credential" in data.get("detail", "")
