"""Voicing S0: the ``openai-key`` login and the speech rows' registry facts.

A green suite cannot show the property that matters here -- the key lands in ITS
namespace and never beside the ChatGPT rows the chat provider routes -- so the
login flow runs through the real controller and a real store, no network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest

from local_operator.providers.auth_store import AuthStore
from local_operator.providers.controller import ProviderController
from local_operator.providers.oauth.callback_server import LoginCallbacks
from local_operator.providers.registry import (
    PROVIDER_REGISTRY,
    credential_provider_id,
    env_key_names,
    get_provider_definition,
    is_speech_only,
    speech_only_message,
)
from local_operator.providers.usage_cache import UsageCacheStore


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[AuthStore]:
    # A root with no provider-class rows, and no ambient provider vars: the
    # point of every test below is what the probe does WITHOUT them.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "empty-config"))
    for var in ("OPENAI_API_KEY", "ELEVENLABS_API_KEY", "RADIENT_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    auth = AuthStore(db_path=tmp_path / "auth.db", config_dir=tmp_path / "config")
    yield auth
    auth.close()


# ---------------------------------------------------------------------------
# The registry rows
# ---------------------------------------------------------------------------


def test_openai_key_owns_a_namespace_and_declares_no_env_key() -> None:
    definition = get_provider_definition("openai-key")
    assert definition is not None
    # NO env var, so an ambient OPENAI_API_KEY can never surface on any view.
    assert definition.env_keys is None
    assert env_key_names("openai-key") == ()
    # Its own storage id -- never ``openai``, whose cascade walks ChatGPT OAuth rows.
    assert credential_provider_id("openai-key") == "openai-key"
    assert definition.store_credentials_as is None, "a flavour alias would route it to chat"
    assert definition.capabilities == frozenset({"tts"})
    assert definition.login_kind == "api_key"
    assert definition.paste_prompt_required is True
    assert definition.paste_is_api_key is True
    assert "openai-key" in {p.id for p in PROVIDER_REGISTRY if p.login is not None}


def test_openai_key_stays_off_every_chat_surface() -> None:
    assert is_speech_only("openai-key") is True
    # The sentence names what the wire serves; a TTS-only row is not "speech-to-text".
    assert "text-to-speech" in speech_only_message("openai-key")
    assert "speech-to-text" in speech_only_message("elevenlabs")


def test_elevenlabs_gained_tts_and_kept_its_flag_and_env_key() -> None:
    definition = get_provider_definition("elevenlabs")
    assert definition is not None
    assert definition.capabilities == frozenset({"stt", "tts"})
    assert definition.speech_only is True
    assert definition.env_keys == "ELEVENLABS_API_KEY"


@pytest.mark.asyncio
async def test_the_standard_login_flow_stores_a_pasted_key_under_openai_key(
    store: AuthStore, tmp_path: Path
) -> None:
    """``/login openai-key`` is paste-a-key, and the row lands in ITS namespace."""
    pasted = {"called": False}

    def factory(_definition: Any) -> LoginCallbacks:
        def paste() -> str:
            pasted["called"] = True
            return "  sk-test-speech-key  "

        return LoginCallbacks(on_manual_code_input=paste)

    controller = ProviderController(
        store,
        login_callbacks=factory,
        usage_cache=UsageCacheStore(tmp_path / "usage_cache.db"),
    )
    message = await controller.login("openai-key")

    assert pasted["called"]
    assert "openai-key" in message
    rows = store.list_credentials("openai-key")
    assert len(rows) == 1
    assert rows[0].credential_type == "api_key"
    assert rows[0].data["source"] == "login"
    assert rows[0].data["key"] == "sk-test-speech-key", "trimmed, like every paste-a-key login"
    assert store.list_credentials("openai") == [], "never beside the ChatGPT rows"
