"""Speech-to-text cascade: the shared tokens and result types.

The package is deliberately **import-light**: the server routes, the session
runtime and tests all import these names, and ``__init__`` therefore carries
nothing heavier than the standard library. The machinery lives beside it —
:mod:`local_operator.stt.cascade` (resolver + executor),
:mod:`local_operator.stt.clients` (the BYO HTTP clients),
:mod:`local_operator.stt.errors` (the shared failure mapping),
:mod:`local_operator.stt.audio` (caps + format helpers) and
:mod:`local_operator.stt.sidecar` (the forked model-transcription record).

The cascade order below is FROZEN (manager decisions, 2026-09-28): Radient →
ElevenLabs → OpenAI → SuperWhisper (reserved, never emitted) → model-audio +
sidecar → honest error. The tokens are the closed vocabulary every surface
speaks; ``none`` appears only in availability/refusal payloads, never on a
message.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

__all__ = [
    "AttemptOutcome",
    "AudioPath",
    "AudioPathResolution",
    "RungAvailability",
    "SttAttempt",
    "SttOutcome",
]


class AudioPath(StrEnum):
    """The closed vocabulary of speech/input paths.

    Values are the wire spellings: they travel in the resolver's payloads and
    (phase 1) in a message's ``input_path`` annotation, so they are pinned with
    the carriage work (OQ-6) rather than re-derived here.
    """

    PROVIDER_STT_RADIENT = "provider_stt_radient"
    PROVIDER_STT_ELEVENLABS = "provider_stt_elevenlabs"
    PROVIDER_STT_OPENAI = "provider_stt_openai"
    #: RESERVED token, pinned so the vocabulary is closed. The resolver reports
    #: this rung as present-but-unavailable and NO code path returns it: the
    #: superwhisper.com CLI is a history browser, not a non-interactive
    #: transcription interface. ``test_cascade_resolver`` pins both properties.
    PROVIDER_STT_SUPERWHISPER = "provider_stt_superwhisper"
    #: The recorded audio is sent to an audio-capable model; a forked sidecar
    #: best-effort fills the transcript record (see ``stt/sidecar.py``).
    MODEL_AUDIO_SIDECAR = "model_audio_sidecar"
    #: No usable path. Appears in availability/refusal payloads only.
    NONE = "none"


#: One rung's outcome in an executor attempt. ``skipped`` is not "not
#: available" — unavailable rungs are never attempted at all — it is a rung the
#: walk reached but did not spend, because the overall budget was already gone.
AttemptOutcome = Literal["ok", "failed", "skipped"]


@dataclass(frozen=True)
class RungAvailability:
    """One rung's availability, with the reason a user would be shown.

    Availability answers "is there a credential/model for this rung", NOT "will
    the call succeed" — see :func:`local_operator.stt.cascade.resolve_audio_path`.
    """

    path: AudioPath
    available: bool
    reason: str


@dataclass(frozen=True)
class AudioPathResolution:
    """The resolver's report: the chosen path, why, and every rung's state."""

    path: AudioPath
    reason: str
    #: Fixed cascade order, SuperWhisper included (available=False).
    rungs: tuple[RungAvailability, ...]
    #: Whether the model the caller asked about accepts audio input. This is
    #: what the surface reads to offer the audio door when no STT rung exists.
    model_audio_capable: bool


@dataclass(frozen=True)
class SttAttempt:
    """One rung's attempt inside :func:`local_operator.stt.cascade.transcribe_audio`."""

    path: AudioPath
    outcome: AttemptOutcome
    detail: str = ""


@dataclass(frozen=True)
class SttOutcome:
    """A successful cascade run: the text, the rung that produced it, and the walk."""

    text: str
    path: AudioPath
    attempts: tuple[SttAttempt, ...]
