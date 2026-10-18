"""Audio generation (voiceover + sound effects): the lane's tokens and types.

The mirror of :mod:`local_operator.imagegen` for the audio half of the media
surface, deliberately a SEPARATE package with its own enum rather than a branch
inside the image one: the rungs are a different set (Radient's audio tools plus
two BYOK direct calls), the request shapes differ per direction (a voiceover is
text plus a voice; a sound effect is a prompt plus a MANDATORY duration), and a
shared enum would let an image-only token leak into an audio payload. One token
vocabulary per capability.

The cascade order is FROZEN as the platform's own plane first, then the user's
keys — Radient → ElevenLabs → OpenAI, first match wins — mirroring the TTS
cascade's economics (``tts/cascade.py``): a Radient refusal falls forward to
the user's own key, because the user's credits being out is not the platform
being broken. The executor fails FORWARD on every rung failure except user
cancellation (stop; no failover) and local validation errors (raised before
dispatch).

Import discipline (mirrors ``imagegen/__init__``): THIS module is stdlib-only.
Session-build paths — the tool builder's createIf gate, the classification
roster — import these names and must not drag in HTTP clients or pydantic. The
heavier companions are imported by their own consumers:

- :mod:`local_operator.audiogen.availability` — the sync credential probes
  (the ``createIf`` gate's substrate; no sockets on any session-build path).
- :mod:`local_operator.audiogen.pricing` — the ONE central rate table for
  BYOK estimates (``price_version`` pinned there; unknown → ``None``).
- :mod:`local_operator.audiogen.peaks` — waveform peaks (optional miniaudio
  import; never raises).
- :mod:`local_operator.audiogen.models` — the frozen tool params models and
  the wire request builders.
- :mod:`local_operator.audiogen.rungs` — the httpx-async rung execution, the
  sha256 verification and the honest best-effort cancel.
- :mod:`local_operator.audiogen.cascade` — the resolver and executor.

**Availability semantics, loud on purpose** (mirrors ``stt``/``tts``): rungs
answer "a credential exists", NOT "the call will succeed". A refused key, an
empty balance or an unreachable model surfaces at call time and fails forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# Re-exported under their generic names (session-build paths import this
# module; the generic types are stdlib-only, so no heavy import rides in).
from local_operator.artifacts import (
    ArtifactKind,
    AttemptOutcome,
    JobAttempt,
    JobOutcome,
    MediaAsset,
    RungAvailability,
)

__all__ = [
    "ArtifactKind",
    "AttemptOutcome",
    "AudioAttempt",
    "AudioOutcome",
    "AudioRoute",
    "AudioRouteResolution",
    "MediaAsset",
    "RungAvailability",
]


class AudioRoute(StrEnum):
    """The closed vocabulary of audio-generation routes.

    Values are the wire spellings: they travel in attempt records, the tools'
    ``details`` and the cost ledger, so they are pinned here rather than
    re-derived per surface.
    """

    RADIENT = "radient"
    ELEVENLABS = "elevenlabs"
    OPENAI = "openai"
    #: No usable route. Appears in availability/refusal payloads only.
    NONE = "none"


#: ``AudioAttempt`` / ``AudioOutcome`` are ALIASES of the generic records — one
#: shape, no isinstance traps (the ``imagegen`` D2 rule): a monkeypatched
#: producer cannot hand a consumer a type its ``isinstance`` check misses.
#: ``AudioOutcome.kind`` defaults to ``image`` in the generic record, so every
#: walk-produced outcome sets it explicitly (``ArtifactKind.AUDIO``).
AudioAttempt = JobAttempt
AudioOutcome = JobOutcome


@dataclass(frozen=True)
class AudioRouteResolution:
    """The resolver's report: the chosen route, why, and every rung's state."""

    route: AudioRoute
    reason: str
    #: Fixed cascade order, every route present.
    rungs: tuple[RungAvailability, ...]
    #: Whether any rung can serve the REQUESTED direction — a voiceover runs
    #: on all three routes, a sound effect only on Radient and ElevenLabs (the
    #: frozen contract's "OpenAI BYOK is voiceover only"). Derived, never
    #: stored, so it cannot disagree with ``rungs``.
    servable: bool
