"""The canonical voicing descriptor and its configuration.

The descriptor is the provider-neutral dial a user sets once and every leg
honours as far as it can. It is deliberately shaped after the canonical map
(``voicing_map/speech_voicing_map.v1.json``), which is the same contract the
hub validates, so a descriptor this module builds is one the hub accepts
without a translation step.

WHY THE DEFAULTS LIVE HERE
--------------------------
``settings_io`` must stay off every consumer's import path (it is loaded on
every CLI start), so its registry rows carry literals. This module is the
other half of that pair: it is what a consumer falls back to when a key is
absent, and ``tests/unit/test_settings_io.py`` imports these constants and
asserts the registry literals match them. A drifted registry default is
therefore a red test rather than a settings page that lies about what the
synthesizer will send.

WHY ``instructions`` HAS A DEFAULT AT ALL
-----------------------------------------
#1835 removed the persona/delivery instructions string when the speak-aloud
path moved to ElevenLabs, accepting "a small delivery loss" because ElevenLabs
has no instructions field. The descriptor restores it as a DEFAULT: it is a
no-op on ElevenLabs (which ignores the field with a note) and it puts the
delivery guidance back on OpenAI, which honours it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional

__all__ = [
    "DEFAULT_ACCENT",
    "DEFAULT_EXPRESSIVENESS",
    "DEFAULT_GENDER",
    "DEFAULT_LANGUAGE",
    "DEFAULT_PACE",
    "DEFAULT_SPEECH_INSTRUCTIONS",
    "DEFAULT_TONE",
    "DESCRIPTOR_FIELDS",
    "DESCRIPTOR_VERSION",
    "MAX_INSTRUCTIONS",
    "MAX_PACE",
    "MIN_PACE",
    "VOICE_TONES",
    "VoiceDescriptor",
    "descriptor_from_config",
]

#: The descriptor schema this daemon speaks. Matches the hub's
#: ``DescriptorVersion`` and the map's ``descriptor_version``.
DESCRIPTOR_VERSION = 1

#: The accepted pace window, in provider-neutral units (1.0 = normal). Wider
#: than any vendor's own speed range on purpose: it is the caller's INTENT, and
#: a leg that cannot go that far clamps and says so. Out of this window is a
#: refusal at the hub, so the daemon clamps to it before sending.
MIN_PACE = 0.5
MAX_PACE = 2.0

#: The ``instructions`` cap, matching ``requests/tool.go`` on the hub.
MAX_INSTRUCTIONS = 8000

#: The v1 tone vocabulary, in the settings page's reading order. The field is
#: an open token downstream (a newer client may know tones this daemon does
#: not), but a user picks from this list.
VOICE_TONES: tuple[str, ...] = ("warm", "neutral", "bright", "calm", "authoritative")

#: The descriptor's field list, in the hub's fixed note order.
DESCRIPTOR_FIELDS: tuple[str, ...] = (
    "gender",
    "tone",
    "expressiveness",
    "pace",
    "language",
    "accent",
    "instructions",
)

# --- the consumer defaults (see the module docstring) -----------------------

#: ``auto`` is the pre-descriptor behaviour: the per-agent classifier picks a
#: gender per request. It is resolved CLIENT-SIDE before the descriptor is
#: sent — the hub refuses ``auto`` because it has no agent context to classify.
DEFAULT_GENDER = "auto"
DEFAULT_TONE = "warm"
DEFAULT_EXPRESSIVENESS = "medium"
DEFAULT_PACE = 1.0
#: ``auto`` means "send no language_code"; each call's own ``language_code``
#: (when the caller set one) still travels as a legacy pinned field.
DEFAULT_LANGUAGE = "auto"
DEFAULT_ACCENT = ""

#: The pre-#1835 native-dialect delivery guidance (see the module docstring).
#: The persona half of that string ("You are <agent>...") is NOT here: it is
#: per-agent, and this is one global setting. The agent's own identity is
#: carried by the message being spoken, not by a voicing instruction.
DEFAULT_SPEECH_INSTRUCTIONS = (
    "Speak aloud and pay attention to potentially multilingual inputs and make sure to use "
    "native accents for all different parts of the text, especially those that are not english. "
    "Strive for a casual and native-sounding conversational tone. Don't over-enunciate, consider "
    'word combinations that should have silent and natural transitions, like "raha hoon" -> '
    '"rahoon" or "je m\'appelle" -> "jm\'appelle".'
)

#: The language field's accepted shape, mirrored from the hub's validator.
_LANGUAGE_CODE = re.compile(r"^[a-z]{2}$")


@dataclass(frozen=True)
class VoiceDescriptor:
    """One provider-neutral voicing request (descriptor v1).

    ``pace`` and ``instructions`` keep ``None`` meaning "absent" distinct from
    "explicitly 1.0" and "explicitly empty": an explicit empty ``instructions``
    clears the configured default, while an absent one means the default was
    already resolved by the caller (the hub adds nothing when the field is
    absent — the map's own note says so).
    """

    gender: str = DEFAULT_GENDER
    tone: str = DEFAULT_TONE
    expressiveness: str = DEFAULT_EXPRESSIVENESS
    pace: Optional[float] = None
    language: str = DEFAULT_LANGUAGE
    accent: str = DEFAULT_ACCENT
    instructions: Optional[str] = None
    version: int = DESCRIPTOR_VERSION

    def resolved_gender(self, fallback: str = "female") -> str:
        """``gender`` with ``auto`` (and anything unrecognised) replaced.

        The hub refuses ``auto``, so a descriptor-bearing call must resolve it
        first. The caller passes the per-agent classifier's answer; the
        fallback is the map's documented unknown-gender row so a classifier
        that could not answer never produces a 400.
        """
        if self.gender in ("female", "male"):
            return self.gender
        return fallback

    def to_wire(self, *, gender: str | None = None) -> dict[str, Any]:
        """The JSON body the hub's ``voice_descriptor`` field accepts.

        ``gender`` overrides this descriptor's own value for callers that
        resolved ``auto`` themselves (the speak-aloud route does, from the
        per-agent classifier). Absent fields are OMITTED rather than sent as
        null: the hub distinguishes "absent" (use your own behaviour) from an
        explicit value, and a null would erase that distinction.
        """
        payload: dict[str, Any] = {
            "version": self.version,
            "gender": gender or self.resolved_gender(),
            "tone": self.tone,
            "expressiveness": self.expressiveness,
            "language": self.language,
            "accent": self.accent,
        }
        if self.pace is not None:
            payload["pace"] = self.pace
        if self.instructions is not None:
            payload["instructions"] = self.instructions
        return payload

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> "VoiceDescriptor":
        """Build a descriptor from the hub's wire shape.

        The inverse of :meth:`to_wire`, and the direction the golden-vector
        conformance test needs: a vector's descriptor is exactly the JSON this
        daemon sends, so the test drives the adapters through the same shape a
        request would carry. Absent keys become ``None`` rather than a default,
        because "absent" and "explicitly the default" are different states
        (see the class docstring).
        """
        return cls(
            gender=str(payload.get("gender") or ""),
            tone=str(payload.get("tone") or ""),
            expressiveness=str(payload.get("expressiveness") or ""),
            pace=None if payload.get("pace") is None else float(payload["pace"]),
            language=str(payload.get("language") or ""),
            accent=str(payload.get("accent") or ""),
            instructions=(
                None if payload.get("instructions") is None else str(payload["instructions"])
            ),
            version=int(payload.get("version") or DESCRIPTOR_VERSION),
        )


def _clean_accent(value: Any) -> str:
    """``accent`` with an unset value read as the empty string.

    A BCP-47 tag is validated by the hub; this only normalises the "no accent"
    spelling, which the config surface reaches through ``empty_unsets``.
    """
    return "" if value is None else str(value).strip()


def descriptor_from_config(
    config_manager: Any,
    *,
    resolved_gender: str | None = None,
    language_code: str | None = None,
) -> VoiceDescriptor:
    """Build the descriptor from ``speech.voice.*``, defaults applied.

    Every key is read per call (the section is LIVE), so a settings edit is
    picked up by the next spoken message with no relaunch.

    ``resolved_gender`` is the classifier's answer for the current agent; when
    it is ``None`` the configured gender stands (and the hub resolves nothing —
    a configured ``auto`` with no classifier answer is clamped to the map's
    unknown-gender row by :meth:`VoiceDescriptor.resolved_gender`).

    ``language_code`` is the CALLER's per-call language. It rides the legacy
    ``language_code`` field, which the hub's per-field pin rule keeps in force,
    so a caller that set one still gets it; the descriptor's own ``language``
    is the persistent setting.
    """

    def read(key: str, default: Any) -> Any:
        value = config_manager.get_nested_value(("speech", "voice", key), None)
        return default if value is None else value

    gender = resolved_gender or str(read("gender", DEFAULT_GENDER))
    pace_value = read("pace", DEFAULT_PACE)
    try:
        pace = float(pace_value)
    except (TypeError, ValueError):
        pace = DEFAULT_PACE
    # NON-FINITE FIRST, then the range. A hand-edited ``config.yml`` can carry
    # ``.nan`` (valid YAML) and ``.inf``; both survive a min/max clamp — NaN
    # compares false against everything — and ``to_wire`` then serialises a bare
    # ``NaN``/``Infinity`` token, which is not JSON the hub's decoder accepts, so
    # every spoken message becomes a 400 while the settings API is bypassed
    # (voicing S2 security round 1, S-5). The settings validator rejects these;
    # this is the belt for the file that never went through it.
    if not math.isfinite(pace):
        pace = DEFAULT_PACE
    # Clamp here rather than relying on the hub: an out-of-window pace is a 400
    # there, and the settings validator should have caught it, but a hand-edited
    # config.yml must not turn a spoken message into a refusal.
    pace = min(max(pace, MIN_PACE), MAX_PACE)

    instructions = read("instructions", DEFAULT_SPEECH_INSTRUCTIONS)
    instructions = None if instructions is None else str(instructions)
    if instructions is not None and len(instructions) > MAX_INSTRUCTIONS:
        # The hub refuses over-cap instructions. Truncating the user's own text
        # is worse than sending none of it, so the cap is enforced by the
        # settings validator and this is the belt for a hand-edited file.
        instructions = instructions[:MAX_INSTRUCTIONS]

    language = str(read("language", DEFAULT_LANGUAGE) or DEFAULT_LANGUAGE)
    if language != "auto" and not _LANGUAGE_CODE.match(language):
        language = DEFAULT_LANGUAGE

    return VoiceDescriptor(
        gender=gender,
        tone=str(read("tone", DEFAULT_TONE) or DEFAULT_TONE),
        expressiveness=str(
            read("expressiveness", DEFAULT_EXPRESSIVENESS) or DEFAULT_EXPRESSIVENESS
        ),
        pace=pace,
        language=language,
        accent=_clean_accent(read("accent", DEFAULT_ACCENT)),
        instructions=instructions,
    )
