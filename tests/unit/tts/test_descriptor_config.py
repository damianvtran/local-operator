"""The ``speech.voice.*`` settings and the descriptor they build.

Two halves: that a write round-trips through the same registry the TUI and the
desktop settings route read, and that the DESCRIPTOR the speak-aloud path
builds is the one those keys describe. The second half is what makes the
settings page's promise real — a key that writes but is never read is a page
that lies.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.config import ConfigManager
from local_operator.settings_io import BY_KEY, read_setting, write_setting
from local_operator.tts import descriptor as voicing
from local_operator.tts.descriptor import (
    DEFAULT_SPEECH_INSTRUCTIONS,
    MAX_PACE,
    MIN_PACE,
    descriptor_from_config,
)

SPEECH_KEYS = (
    "speech.voice.gender",
    "speech.voice.tone",
    "speech.voice.expressiveness",
    "speech.voice.pace",
    "speech.voice.language",
    "speech.voice.accent",
    "speech.voice.instructions",
)


@pytest.fixture
def manager(tmp_path: Path) -> ConfigManager:
    return ConfigManager(config_dir=tmp_path / "config")


def test_every_key_is_registered_once_in_one_section() -> None:
    for key in SPEECH_KEYS:
        assert key in BY_KEY, f"{key} is not in the settings registry"
        assert BY_KEY[key].section == "speech"
        assert BY_KEY[key].path[:2] == ("speech", "voice")


def test_the_defaults_are_the_consumers_own_constants(manager: ConfigManager) -> None:
    """Absent keys resolve to the descriptor module's defaults, not a second table."""
    built = descriptor_from_config(manager)
    assert built.tone == voicing.DEFAULT_TONE
    assert built.expressiveness == voicing.DEFAULT_EXPRESSIVENESS
    assert built.pace == voicing.DEFAULT_PACE
    assert built.language == voicing.DEFAULT_LANGUAGE
    assert built.accent == voicing.DEFAULT_ACCENT
    assert built.instructions == DEFAULT_SPEECH_INSTRUCTIONS


def test_a_write_round_trips_into_the_descriptor(manager: ConfigManager) -> None:
    """The TUI's ``lop config edit speech.voice.tone calm`` path, exercised."""
    write_setting(manager, BY_KEY["speech.voice.tone"], "calm")
    write_setting(manager, BY_KEY["speech.voice.pace"], 1.2)
    write_setting(manager, BY_KEY["speech.voice.expressiveness"], "high")

    # The registry reads back what was written...
    assert read_setting(manager, BY_KEY["speech.voice.tone"]) == "calm"
    # ...and the descriptor the route builds agrees.
    built = descriptor_from_config(manager)
    assert built.tone == "calm"
    assert built.pace == pytest.approx(1.2)
    assert built.expressiveness == "high"


def test_the_gender_setting_is_auto_by_default_and_overridable(manager: ConfigManager) -> None:
    assert descriptor_from_config(manager).gender == "auto"
    write_setting(manager, BY_KEY["speech.voice.gender"], "male")
    assert descriptor_from_config(manager).gender == "male"


def test_a_resolved_gender_beats_the_setting(manager: ConfigManager) -> None:
    """The classifier's answer is what travels: the hub refuses ``auto``."""
    assert descriptor_from_config(manager, resolved_gender="female").gender == "female"
    assert descriptor_from_config(manager, resolved_gender="female").resolved_gender() == "female"


def test_an_unresolvable_auto_never_sends_auto(manager: ConfigManager) -> None:
    """A descriptor that still says ``auto`` maps to a real row rather than a 400."""
    built = descriptor_from_config(manager)
    assert built.gender == "auto"
    assert built.resolved_gender() in ("female", "male")


def test_a_hand_edited_pace_is_clamped_to_the_accepted_window(
    manager: ConfigManager, tmp_path: Path
) -> None:
    """The settings validator bounds it; a hand-edited file must not 400 the hub."""
    manager.config.values.setdefault("speech", {}).setdefault("voice", {})["pace"] = 9.0
    assert descriptor_from_config(manager).pace == MAX_PACE
    manager.config.values["speech"]["voice"]["pace"] = 0.1
    assert descriptor_from_config(manager).pace == MIN_PACE


def test_a_non_finite_pace_falls_back_to_the_default(
    manager: ConfigManager,
) -> None:
    """S-5: `.nan` survives a min/max clamp, and then puts bare `NaN` on the wire.

    A hand-edited ``config.yml`` can carry ``.nan`` (valid YAML). ``min(max(nan,
    …), …)`` is ``nan``, and ``json.dumps`` defaults to ``allow_nan=True``, so
    the hub's Go decoder would reject EVERY spoken message with a 400 while the
    settings API (which validates) was bypassed. ``.inf`` is the same class.
    """
    import math

    manager.config.values.setdefault("speech", {}).setdefault("voice", {})["pace"] = float("nan")
    built = descriptor_from_config(manager)
    assert built.pace is not None and math.isfinite(built.pace)
    assert built.pace == voicing.DEFAULT_PACE

    manager.config.values["speech"]["voice"]["pace"] = float("inf")
    assert descriptor_from_config(manager).pace == voicing.DEFAULT_PACE


def test_an_invalid_language_falls_back_to_auto(manager: ConfigManager) -> None:
    manager.config.values.setdefault("speech", {}).setdefault("voice", {})["language"] = "english"
    assert descriptor_from_config(manager).language == "auto"
    manager.config.values["speech"]["voice"]["language"] = "es"
    assert descriptor_from_config(manager).language == "es"


def test_empty_instructions_mean_send_none(manager: ConfigManager) -> None:
    """An explicit empty is a state of its own: no sentences of our own are sent."""
    write_setting(manager, BY_KEY["speech.voice.instructions"], "")
    built = descriptor_from_config(manager)
    assert built.instructions == ""
    assert built.to_wire()["instructions"] == ""


def test_the_wire_payload_omits_the_fields_the_hub_distinguishes() -> None:
    """Absent vs explicit differ, so an absent field must not be sent as null."""
    built = voicing.VoiceDescriptor(gender="female", pace=None, instructions=None)
    payload = built.to_wire()
    assert "pace" not in payload
    assert "instructions" not in payload
    assert payload["gender"] == "female"
