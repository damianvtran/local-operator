"""``speech.provider``: the ONE provider the user chose to hear and speak with.

Ben's decision (frozen): the user picks ``radient`` (billed to Radient credit)
or ``elevenlabs`` (their own key). No ordering, no fallthrough, and the same
provider for hearing and speaking. Unset means no provider chosen yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.settings_io import BY_KEY, read_setting, write_setting
from local_operator.stt import AudioPath
from local_operator.tts import VoicePath
from local_operator.voice import provider as speech_provider

KEY = "speech.provider"


@pytest.fixture()
def manager(tmp_path: Path) -> ConfigManager:
    return ConfigManager(tmp_path)


def test_the_setting_is_registered_unset_by_default_with_exactly_two_choices() -> None:
    setting = BY_KEY[KEY]
    assert setting.section == "speech"
    assert setting.path == ("speech", "provider")
    assert setting.kind is settings_io.Kind.ENUM
    assert setting.default is None
    assert [choice.value for choice in setting.choices] == [None, "radient", "elevenlabs"]


@pytest.mark.parametrize("value", ["radient", "elevenlabs"])
def test_a_chosen_provider_round_trips(manager: ConfigManager, value: str) -> None:
    write_setting(manager, BY_KEY[KEY], value)
    assert read_setting(manager, BY_KEY[KEY]) == value
    assert speech_provider.configured_provider(manager.config_dir) == value


@pytest.mark.parametrize("value", ["openai", "groq", "qwen", "auto", "", "RADIENT", True, 3])
def test_any_other_value_is_rejected_and_nothing_is_stored(
    manager: ConfigManager, value: object
) -> None:
    with pytest.raises(ValueError):
        write_setting(manager, BY_KEY[KEY], value)
    assert speech_provider.configured_provider(manager.config_dir) is None


def test_clearing_the_choice_deletes_the_key(manager: ConfigManager) -> None:
    write_setting(manager, BY_KEY[KEY], "elevenlabs")
    write_setting(manager, BY_KEY[KEY], None)
    assert speech_provider.configured_provider(manager.config_dir) is None
    assert "provider" not in (manager.get_config().values.get("speech") or {})


def test_unset_reads_as_none(manager: ConfigManager) -> None:
    assert speech_provider.configured_provider(manager.config_dir) is None


def test_the_reader_sees_a_write_made_after_this_manager_loaded(tmp_path: Path) -> None:
    """The write usually comes from another process; the server's copy is stale."""
    server_copy = ConfigManager(tmp_path)
    write_setting(ConfigManager(tmp_path), BY_KEY[KEY], "radient")
    assert speech_provider.configured_provider(server_copy.config_dir) == "radient"


@pytest.mark.parametrize("junk", ["openai", "groq", "", "Radient", 7, ["radient"], {"a": 1}])
def test_a_hand_edited_value_outside_the_two_reads_as_unset(tmp_path: Path, junk: object) -> None:
    """A bad file must spend nothing: it reads as 'not chosen', never as a pin."""
    import yaml

    (tmp_path / "config.yml").write_text(yaml.safe_dump({"values": {"speech": {"provider": junk}}}))
    assert speech_provider.configured_provider(tmp_path) is None


def test_an_unreadable_directory_reads_as_unset() -> None:
    assert speech_provider.configured_provider(Path("/nonexistent/lop-speech-provider")) is None


def test_each_provider_pins_exactly_one_rung_of_each_cascade() -> None:
    assert speech_provider.PINNED_STT == {
        "radient": AudioPath.PROVIDER_STT_RADIENT,
        "elevenlabs": AudioPath.PROVIDER_STT_ELEVENLABS,
    }
    assert speech_provider.PINNED_TTS == {
        "radient": VoicePath.PROVIDER_TTS_RADIENT,
        "elevenlabs": VoicePath.PROVIDER_TTS_ELEVENLABS,
    }
    assert set(speech_provider.PROVIDERS) == {"radient", "elevenlabs"}


def test_the_unset_refusal_is_catalogued_and_names_both_choices() -> None:
    from local_operator.i18n.keys import wire_voice

    text = wire_voice.provider_unset().envelope(code="voice_provider_unset")
    assert text["code"] == "voice_provider_unset"
    assert "Radient" in text["text"] and "ElevenLabs" in text["text"]
