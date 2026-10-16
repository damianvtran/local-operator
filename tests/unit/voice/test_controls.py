"""R6: the local front end's controls grammar matches WHOLE utterances only."""

from __future__ import annotations

import pytest

from local_operator.voice import controls


@pytest.mark.parametrize(
    ("utterance", "action"),
    [
        ("stop", "stop_speaking"),
        ("Stop.", "stop_speaking"),
        ("  STOP!  ", "stop_speaking"),
        ("stop talking", "stop_speaking"),
        ("Say that again?", "repeat_last"),
        ("repeat", "repeat_last"),
        ("louder", "louder"),
        ("speak   up", "louder"),
        ("softer", "softer"),
        ("“quieter”", "softer"),
    ],
)
def test_a_whole_utterance_phrase_is_a_control(utterance: str, action: str) -> None:
    assert controls.match(utterance) == action


@pytest.mark.parametrize(
    "utterance",
    [
        "stop the build",
        "please stop",
        "stop, then run the tests",
        "don't stop",
        "repeat the last command",
        "make the logs louder",
        "softer edges on the button",
        "stopp",
        "",
        "   ",
        "list files",
    ],
)
def test_anything_else_relays(utterance: str) -> None:
    """A keyword inside a sentence is a sentence for the agent, not a control."""
    assert controls.match(utterance) is None


def test_every_phrase_maps_into_the_closed_action_set() -> None:
    assert set(controls.PHRASES.values()) <= set(controls.CONTROL_ACTIONS)
    assert set(controls.CONTROL_ACTIONS) == {"stop_speaking", "repeat_last", "louder", "softer"}
    # The table is stored normalised, so what a reviewer reads is what matches.
    assert all(controls.normalise(phrase) == phrase for phrase in controls.PHRASES)


def test_no_phrase_aborts_the_session_turn() -> None:
    """R6: voice never aborts a turn in v1; aborting stays on screen."""
    assert not any("abort" in action or "cancel" in action for action in controls.CONTROL_ACTIONS)
