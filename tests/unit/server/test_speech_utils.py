"""Tests for the agent-speech voice-selection helpers.

The properties pinned here are the findings this module was fixed for: the
classifier's answer IS the voice alias (no provider voice names baked in), and
repeated presses for the same agent answer from an in-process cache instead of
running a fresh model call each time.
"""

from datetime import datetime
from typing import Any, Dict

import pytest

from local_operator.agents import AgentData
from local_operator.server.utils import speech_utils
from local_operator.server.utils.speech_utils import (
    _gender_cache_key,
    determine_voice,
    parse_gender_from_xml,
)


class _ClassifierResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _CountingExecutor:
    """A stand-in for ServerExecutor that counts the classifier calls it serves."""

    def __init__(self, gender: str = "female") -> None:
        self.gender = gender
        self.calls = 0

    async def invoke_model(self, messages):
        self.calls += 1
        return _ClassifierResponse(f"<gender>{self.gender}</gender>")


def _agent(name: str = "Aria", description: str = "A friendly assistant") -> AgentData:
    """Build an agent; the fields travel through a typed dict because the type
    checker treats a pydantic model's defaulted fields as required."""
    fields: Dict[str, Any] = {
        "id": "speech-utils-agent",
        "name": name,
        "created_date": datetime.now(),
        "version": "1.0.0",
        "description": description,
    }
    return AgentData(**fields)


def _executor(gender: str = "female") -> Any:
    """A counting classifier stub; ``Any`` because the production type is ServerExecutor."""
    return _CountingExecutor(gender)


@pytest.fixture(autouse=True)
def _fresh_gender_cache():
    """Each test starts and ends with an empty classification cache."""
    speech_utils._GENDER_CACHE.clear()
    yield
    speech_utils._GENDER_CACHE.clear()


@pytest.mark.asyncio
async def test_determine_voice_is_the_classifier_alias():
    """The alias travels to the hub; no provider voice name is baked in here."""
    female = _executor(gender="female")
    male = _executor(gender="male")

    assert await determine_voice(_agent(name="Aria"), female) == "female"
    assert await determine_voice(_agent(name="Bram"), male) == "male"


@pytest.mark.asyncio
async def test_determine_voice_caches_per_name_and_description():
    """A second press for the same agent answers without a second model call."""
    executor = _executor(gender="female")

    first = await determine_voice(_agent(), executor)
    second = await determine_voice(_agent(), executor)

    assert first == second == "female"
    assert executor.calls == 1


@pytest.mark.asyncio
async def test_determine_voice_reclassifies_when_the_description_changes():
    executor = _executor(gender="male")

    assert await determine_voice(_agent(description="A friendly assistant"), executor) == "male"
    assert await determine_voice(_agent(description="A fleet admiral"), executor) == "male"
    assert executor.calls == 2


@pytest.mark.asyncio
async def test_the_cache_evicts_the_least_recently_used_entry(monkeypatch):
    """A long-lived daemon never pins more than the bound's worth of agents.

    Eviction is LRU, not FIFO: a hit moves its key to the most-recent end, so
    the entry that goes is the least recently USED one.
    """
    monkeypatch.setattr(speech_utils, "_GENDER_CACHE_MAX", 2)
    executor = _executor(gender="female")

    await determine_voice(_agent(name="A"), executor)  # calls=1
    await determine_voice(_agent(name="B"), executor)  # calls=2
    await determine_voice(_agent(name="A"), executor)  # hit: A becomes most recent
    assert executor.calls == 2

    await determine_voice(_agent(name="C"), executor)  # calls=3; evicts B
    await determine_voice(_agent(name="A"), executor)  # still cached -> no call
    assert executor.calls == 3

    await determine_voice(_agent(name="B"), executor)  # evicted -> calls=4
    assert executor.calls == 4


def test_gender_cache_key_separates_shifted_names_and_descriptions():
    """The NUL separator keeps ("ab", "c") from colliding with ("a", "bc")."""
    assert _gender_cache_key("ab", "c") != _gender_cache_key("a", "bc")


def test_parse_gender_from_xml_defaults_to_male():
    assert parse_gender_from_xml("<gender>female</gender>") == "female"
    assert parse_gender_from_xml("<gender>male</gender>") == "male"
    assert parse_gender_from_xml("no tags here") == "male"
    assert parse_gender_from_xml("<gender>other</gender>") == "male"
