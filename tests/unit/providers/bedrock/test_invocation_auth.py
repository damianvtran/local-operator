"""Explicit routing opts into the AWS chain, never into global availability."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from local_operator.providers.auth_store import AuthStore
from local_operator.providers.bedrock import connection as bc
from local_operator.providers.bedrock.auth import (
    BedrockCredentialError,
    reset_session_cache,
    resolve,
)
from local_operator.providers.controller import ProviderController


@pytest.fixture(autouse=True)
def isolated_aws(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import os

    for key in tuple(os.environ):
        if key.startswith(("AWS_", "LOP_BEDROCK_")) or key in bc.COMPUTE_MARKERS:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    reset_session_cache()
    yield
    reset_session_cache()


def test_invocation_does_not_enable_global_census(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWS_PROFILE", "synthetic")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "synthetic-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "synthetic-secret")
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    other = AuthStore(tmp_path / "other.db", config_dir=tmp_path)
    try:
        bc.configure_invocation_auth(store, "bedrock")
        assert asyncio.run(store.get_api_key("bedrock")) == "bedrock://ambient"
        access = asyncio.run(store.get_oauth_access("bedrock"))
        assert access is not None and access.access_token == "bedrock://ambient"
        assert bc.bedrock_env_key() is None
        controller = ProviderController(store)
        assert not controller.is_usable("bedrock")
        assert "bedrock" not in (controller.usable_providers() or set())
        assert not store.list_credentials("bedrock")
        bc.configure_invocation_auth(other, "openai")
        assert asyncio.run(other.get_api_key("bedrock")) is None
    finally:
        store.close()
        other.close()


@pytest.mark.parametrize("source", ["login", "migration"])
def test_stored_connection_and_env_win(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    try:
        bc.configure_invocation_auth(store, "bedrock")
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "synthetic-bearer")
        env = bc.bedrock_env_key()
        assert asyncio.run(store.get_api_key("bedrock")) == env
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK")
        descriptor = "bedrock://profile?name=stored&region=eu-west-1"
        store.upsert_credential("bedrock", {"type": "api_key", "source": source, "key": descriptor})
        assert asyncio.run(store.get_api_key("bedrock")) == descriptor
        access = asyncio.run(store.get_oauth_access("bedrock"))
        assert access is not None and access.access_token == descriptor
    finally:
        store.close()


def test_no_credentials_is_classified_not_a_fake_key() -> None:
    with pytest.raises(BedrockCredentialError) as error:
        asyncio.run(resolve("bedrock://ambient"))
    assert error.value.provider_error.request_sent is False
    assert "credential" in str(error.value).lower()


def test_explicit_profile_region_wins_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "credentials").write_text(
        "[synthetic]\naws_access_key_id = synthetic-id\naws_secret_access_key = synthetic-secret\n"
    )
    (tmp_path / "config").write_text("[profile synthetic]\nregion = us-west-2\n")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    auth = asyncio.run(resolve("bedrock://profile?name=synthetic&region=eu-west-1"))
    assert auth.region == "eu-west-1"
    assert auth.credentials.access_key == "synthetic-id"
