"""Text-to-speech cascade: the shared tokens and result types.

The mirror of :mod:`local_operator.stt`, deliberately a SEPARATE package with
its own enum rather than a branch inside the STT one. The two cascades look
alike because they answer the same question — "which credential-bearing path
does this machine take?" — but their rungs are different sets, their providers
differ in what they can express, and a shared enum would let an STT-only token
leak into a TTS payload (``model_audio_sidecar`` has no meaning for a
synthesis call). One token vocabulary per direction.

The order below is FROZEN (design note §5): Radient → ElevenLabs → OpenAI.
The tokens are the closed vocabulary every surface speaks; ``none`` appears
only in availability/refusal payloads, never on a spoken message.

Import-light on purpose: the server routes and the session runtime import
these names, so nothing heavier than the standard library lives here. The
machinery is beside it — :mod:`local_operator.tts.cascade` (resolver +
executor), :mod:`local_operator.tts.adapters` (the map executors),
:mod:`local_operator.tts.descriptor` (the canonical descriptor and its config
defaults) and :mod:`local_operator.tts.clients` (the BYO HTTP clients).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

__all__ = [
    "AttemptOutcome",
    "RungAvailability",
    "TtsAttempt",
    "TtsOutcome",
    "VoicePath",
    "VoicePathResolution",
]


class VoicePath(StrEnum):
    """The closed vocabulary of text-to-speech paths.

    Values are the wire spellings: they travel in the resolver's payloads and
    in the ``X-Radient-Speech-Path`` header a served call returns, so they are
    the contract a surface reads rather than an internal name.
    """

    PROVIDER_TTS_RADIENT = "provider_tts_radient"
    PROVIDER_TTS_ELEVENLABS = "provider_tts_elevenlabs"
    PROVIDER_TTS_OPENAI = "provider_tts_openai"
    #: No usable path. Appears in availability/refusal payloads only.
    NONE = "none"


#: One rung's outcome in an executor attempt. ``skipped`` is not "not
#: available" — unavailable rungs are never attempted at all — it is a rung the
#: walk reached but did not spend, because the overall budget was already gone.
AttemptOutcome = Literal["ok", "failed", "skipped"]


@dataclass(frozen=True)
class RungAvailability:
    """One rung's availability, with the reason a user would be shown.

    Availability answers "is there a credential for this rung", NOT "will the
    call succeed" — see :func:`local_operator.tts.cascade.resolve_voice_path`.
    """

    path: VoicePath
    available: bool
    reason: str


@dataclass(frozen=True)
class VoicePathResolution:
    """The resolver's report: the chosen path, why, and every rung's state."""

    path: VoicePath
    reason: str
    #: Fixed cascade order, ``none`` excluded.
    rungs: tuple[RungAvailability, ...]
    #: Whether THIS surface can synthesize at all — true when any rung is
    #: available. A surface reads it to enable or explain its speak control
    #: without walking ``rungs`` itself; it is derived, never stored, so it
    #: cannot disagree with ``path``.
    servable: bool


@dataclass(frozen=True)
class TtsAttempt:
    """One rung's attempt inside :func:`local_operator.tts.cascade.synthesize_speech`."""

    path: VoicePath
    outcome: AttemptOutcome
    detail: str = ""


@dataclass(frozen=True)
class TtsOutcome:
    """A successful synthesis: the audio, the rung that produced it, and the walk."""

    audio: bytes
    path: VoicePath
    attempts: tuple[TtsAttempt, ...]
    #: The provider that ACTUALLY served, which is what the echoed-actual-path
    #: rule reports — the hub names its own leg in ``X-Radient-Speech-Provider``
    #: and this field carries it back, so a descriptor-bearing call that fell
    #: forward to OpenAI says so instead of claiming ElevenLabs. Empty when the
    #: leg did not name one.
    provider: str = ""
    #: The ``X-Radient-Speech-*`` headers the route puts on its own response,
    #: in order. For a hub-served call these are the hub's, relayed verbatim —
    #: they are the only place the leg that actually served is recorded, because
    #: a descriptor-bearing request names none. For a BYO leg the daemon is the
    #: mapper, so it emits the same header family from its own mapping. Always
    #: carries ``X-Radient-Speech-Path``.
    speech_headers: tuple[tuple[str, str], ...] = ()
