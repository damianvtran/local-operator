"""Sync credential probes: persisted-row rules, fallbacks, never a raise.

The properties under test are the ones the createIf gates depend on: Radient
is advertised from PERSISTED LOGIN ROWS ONLY (never an export), the BYOK
probes accept a stored row or an export, and every probe failure reads "not
available" instead of raising — a broken probe must never take a session's
tool surface down with it.
"""

from __future__ import annotations

from typing import Any

from local_operator.audiogen import availability


class _Row:
    def __init__(self, credential_type: str, data: dict[str, Any]) -> None:
        self.credential_type = credential_type
        self.data = data


class _Store:
    def __init__(self, rows: dict[str, list[_Row]]) -> None:
        self._rows = rows
        self.closed = False

    def list_credentials(self, provider: str) -> list[_Row]:
        return list(self._rows.get(provider, []))

    def close(self) -> None:
        self.closed = True


def _pin_store(monkeypatch, rows):
    store = _Store(rows)

    def open_store(config_dir=None):
        return store

    monkeypatch.setattr(availability, "_open_store", open_store)
    return store


def test_radient_reads_persisted_rows_only(monkeypatch):
    _pin_store(monkeypatch, {"radient": [_Row("oauth", {"access_token": "t"})]})
    assert availability.radient_available() is True
    _pin_store(monkeypatch, {})
    # An exported RADIENT_API_KEY must never light the rung (the STT/mobile rule).
    monkeypatch.setenv("RADIENT_API_KEY", "exported")
    assert availability.radient_available() is False


def test_elevenlabs_prefers_login_rows_then_store_row_then_env(monkeypatch):
    _pin_store(monkeypatch, {"elevenlabs": [_Row("api_key", {"key": "login-key"})]})
    monkeypatch.setenv(availability.ELEVENLABS_ENV_KEY, "env-key")
    assert availability.elevenlabs_key() == "login-key"

    _pin_store(monkeypatch, {})
    monkeypatch.setattr(
        "local_operator.providers.registry.provider_secret_value",
        lambda env_key, base=None: "store-key" if env_key == "ELEVENLABS_API_KEY" else None,
    )
    assert availability.elevenlabs_key() == "store-key"

    monkeypatch.setattr(
        "local_operator.providers.registry.provider_secret_value",
        lambda env_key, base=None: None,
    )
    assert availability.elevenlabs_key() == "env-key"


def test_openai_reads_api_key_rows_only_never_an_oauth_grant(monkeypatch):
    _pin_store(
        monkeypatch,
        {availability.OPENAI_AUDIO_NAMESPACE: [_Row("oauth", {"access_token": "t"})]},
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "local_operator.providers.registry.provider_secret_value",
        lambda env_key, base=None: None,
    )
    assert availability.openai_audio_key() is None

    _pin_store(
        monkeypatch,
        {availability.OPENAI_AUDIO_NAMESPACE: [_Row("api_key", {"key": "platform-key"})]},
    )
    assert availability.openai_audio_key() == "platform-key"


def test_every_probe_failure_reads_unavailable(monkeypatch):
    def boom(config_dir=None):
        raise RuntimeError("store exploded")

    monkeypatch.setattr(availability, "_open_store", boom)
    monkeypatch.delenv(availability.ELEVENLABS_ENV_KEY, raising=False)
    monkeypatch.delenv(availability.OPENAI_ENV_KEY, raising=False)
    monkeypatch.setattr(
        "local_operator.providers.registry.provider_secret_value",
        lambda env_key, base=None: None,
    )
    assert availability.radient_available() is False
    assert availability.elevenlabs_key() is None
    assert availability.openai_audio_key() is None
    assert availability.voiceover_reachable() is False
    assert availability.sound_effect_reachable() is False


def test_the_two_reachability_gates_differ_for_openai_only_boxes(monkeypatch):
    _pin_store(monkeypatch, {})
    monkeypatch.delenv(availability.ELEVENLABS_ENV_KEY, raising=False)
    monkeypatch.setenv(availability.OPENAI_ENV_KEY, "openai-only")
    monkeypatch.setattr(
        "local_operator.providers.registry.provider_secret_value",
        lambda env_key, base=None: None,
    )
    # OpenAI serves voiceovers only: the voiceover tool may exist, the
    # sound-effect tool must not.
    assert availability.voiceover_reachable() is True
    assert availability.sound_effect_reachable() is False
