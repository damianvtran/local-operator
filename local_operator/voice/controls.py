"""The local front end's controls grammar (session-voice.md R6).

A CONTROL take is about the voice channel itself (stop the playback, say it
again, louder, softer). It is handled by the client and never reaches the
session: nothing is admitted and no receipt is spent. Every other take is a
RELAY, and its words go to the session's own agent unchanged.

WHOLE-UTTERANCE EXACT PHRASES ONLY, and that is the whole safety argument. A
take matches only when the ENTIRE normalised utterance is one of the phrases
below: "stop" is a control, "stop the build" is a relay. A keyword or prefix
match would turn a sentence meant for the agent into a client action and drop
it on the floor, which is worse than a control that has to be said plainly.

Voice does NOT abort a session turn in v1 (R6): a misheard "stop" would kill
tool work, so no phrase here maps to an abort, and aborting stays on screen.

Normalisation is deliberately small: case, surrounding punctuation the speech
recogniser adds ("Stop." / "stop!"), and runs of whitespace. Nothing else is
folded, so the set below is exactly what matches.
"""

from __future__ import annotations

import re
from typing import Literal

#: The closed action set (R6). ``stop_speaking`` is also what any push-to-talk
#: press does on the client; it is here so a spoken "stop" works the same way.
ControlAction = Literal["stop_speaking", "repeat_last", "louder", "softer"]

CONTROL_ACTIONS: tuple[ControlAction, ...] = (
    "stop_speaking",
    "repeat_last",
    "louder",
    "softer",
)

#: Every phrase, already normalised, mapped to its action. One table, so the
#: grammar a reviewer reads is the grammar that runs.
PHRASES: dict[str, ControlAction] = {
    "stop": "stop_speaking",
    "stop talking": "stop_speaking",
    "stop speaking": "stop_speaking",
    "be quiet": "stop_speaking",
    "quiet": "stop_speaking",
    "shh": "stop_speaking",
    "repeat": "repeat_last",
    "repeat that": "repeat_last",
    "say that again": "repeat_last",
    "say it again": "repeat_last",
    "again": "repeat_last",
    "louder": "louder",
    "speak up": "louder",
    "volume up": "louder",
    "softer": "softer",
    "quieter": "softer",
    "volume down": "softer",
}

# Punctuation a recogniser puts around a short take. Apostrophes and hyphens
# inside a word are not touched, so "don't stop" stays two words and relays.
_EDGE_PUNCTUATION = re.compile(r"^[\s\"'“”‘’.,!?;:…-]+|[\s\"'“”‘’.,!?;:…-]+$")
_WHITESPACE = re.compile(r"\s+")


def normalise(utterance: str) -> str:
    """Case-fold, trim edge punctuation and collapse whitespace. Nothing more."""
    text = _EDGE_PUNCTUATION.sub("", utterance or "")
    return _WHITESPACE.sub(" ", text).strip().casefold()


def match(utterance: str) -> ControlAction | None:
    """The control ``utterance`` IS, or ``None`` when it is a relay take."""
    return PHRASES.get(normalise(utterance))


__all__ = ["CONTROL_ACTIONS", "PHRASES", "ControlAction", "match", "normalise"]
