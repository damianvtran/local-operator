"""
Tests for the credential endpoints of the FastAPI server.

This module contains tests for credential-related functionality, including
retrieving and updating credential settings.
"""

from unittest.mock import patch

import pytest

from local_operator.server.models.schemas import CredentialUpdate


def _store_key(manager, env_key: str, value: str) -> None:
    """Arm a provider-class STORE ROW under the manager's config root (PR2a)."""
    from local_operator.providers.registry import store_provider_key

    store_provider_key(env_key, value, base=manager.config_dir)


@pytest.mark.asyncio
async def test_list_credentials_success(test_app_client, mock_credential_manager):
    """Test retrieving credentials list successfully.

    The keys are armed as provider-class STORE ROWS (PR2a): the endpoint lists the
    store, and the legacy ``CredentialManager`` file view it used to union is gone.
    """
    _store_key(mock_credential_manager, "OPENAI_API_KEY", "test-key")
    _store_key(mock_credential_manager, "SERPAPI_API_KEY", "test-key2")

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
    """Test retrieving only non-empty credentials.

    A key with a value is a store ROW (PR2a). The "empty value" case is expressed
    as an absent row: the store refuses a blank payload at the writer, so a key
    the operator cleared is one with no row at all — the same "not a credential"
    answer the legacy file's blank line used to give.
    """
    _store_key(mock_credential_manager, "OPENAI_API_KEY", "test-key")
    _store_key(mock_credential_manager, "SERPAPI_API_KEY", "test-key2")

    response = await test_app_client.get("/v1/credentials")

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "Credentials retrieved successfully"
    result = data.get("result")
    assert "keys" in result

    # Only rows with a value are listed. ``RADIENT_API_KEY`` is the row the
    # shared ``test_app_client`` fixture arms (PR2a, for the Radient upload
    # routes), so the two keys this test stored plus that one are exactly the
    # non-empty set; a key with no row is absent.
    assert "OPENAI_API_KEY" in result["keys"]
    assert "SERPAPI_API_KEY" in result["keys"]
    assert "EMPTY_API_KEY" not in result["keys"]
    assert "ANOTHER_EMPTY_KEY" not in result["keys"]
    assert len(result["keys"]) == 3


@pytest.mark.asyncio
async def test_list_credentials_empty(test_app_client, mock_credential_manager):
    """A host whose only key is the fixture's own arming lists exactly that.

    The shared ``test_app_client`` fixture arms a ``RADIENT_API_KEY`` store row
    (PR2a, for the Radient upload routes), so "empty" here means every OTHER key
    is absent — the placeholder names this test would otherwise store are gone.
    """
    response = await test_app_client.get("/v1/credentials")

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "Credentials retrieved successfully"
    result = data.get("result")
    assert "keys" in result
    assert "EMPTY_API_KEY" not in result["keys"]
    assert "ANOTHER_EMPTY_KEY" not in result["keys"]
    assert result["keys"] == ["RADIENT_API_KEY"]


@pytest.mark.asyncio
async def test_list_credentials_error(test_app_client, mock_credential_manager):
    """Test error handling when retrieving credentials list."""
    # Mock the open function to raise an exception
    with patch(
        "local_operator.secrets.access.open_store",
        side_effect=Exception("Test error"),
    ):
        response = await test_app_client.get("/v1/credentials")

    assert response.status_code == 500
    data = response.json()
    assert "Error retrieving credentials" in data.get("detail", "")


@pytest.mark.asyncio
async def test_update_credential_success(test_app_client, mock_credential_manager):
    """The PATCH writes a provider-class store row under the manager's root."""
    from local_operator.secrets.access import open_store
    from local_operator.secrets.store import provider_secret_name

    update_payload = CredentialUpdate(
        key="TEST_API_KEY",
        value="test-value",
    )
    response = await test_app_client.patch("/v1/credentials", json=update_payload.model_dump())

    # The consolidated writer files the key as a LOP_PROVIDER_* row in the store
    # rooted at the manager's config dir, NOT in the plaintext credentials.env.
    stored = open_store(mock_credential_manager.config_dir).get(
        provider_secret_name("TEST_API_KEY"), role="provider"
    )
    assert stored.decode() == "test-value"

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "Credential updated successfully"


@pytest.mark.asyncio
async def test_update_credential_drops_the_model_metadata_memo(
    test_app_client, mock_credential_manager
):
    """A new key must take effect without a restart.

    Model metadata is memoized per 24h TTL bucket, and a missing credential is one
    of the reasons it resolves badly: the provider's listing 401s and every model
    it would have described falls back to the 128k unknown default, which is what
    compaction then sizes itself against. The server runs for days, so without this
    the user who just fixed the cause keeps the degraded answer for the rest of the
    bucket.
    """
    from local_operator.model.configure import (
        _resolve_model_info_cached,
        resolve_model_info,
    )

    resolve_model_info("anthropic", "claude-sonnet-4-20250514")
    assert _resolve_model_info_cached.cache_info().currsize > 0

    payload = CredentialUpdate(key="ANTHROPIC_API_KEY", value="sk-ant-new")
    response = await test_app_client.patch("/v1/credentials", json=payload.model_dump())

    assert response.status_code == 200
    assert _resolve_model_info_cached.cache_info().currsize == 0


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
    # The route writes through the provider-namespace writer; a failure there
    # must surface as a 500, not a silent success.
    with patch(
        "local_operator.providers.registry.store_provider_key",
        side_effect=Exception("Test error"),
    ):
        update_payload = CredentialUpdate(
            key="TEST_API_KEY",
            value="test-value",
        )
        response = await test_app_client.patch("/v1/credentials", json=update_payload.model_dump())

    assert response.status_code == 500
    data = response.json()
    assert "Error updating credential" in data.get("detail", "")
