"""Image generation cascade: the lane's tokens, resolver types and route enum.

The package is deliberately **import-light** (stdlib only, like ``stt/``'s
``__init__``): the tool builder, the session factory's classification roster
and the tests all import these names on paths that must not drag in the HTTP
client or the credential stores. Under media wave-2 the generic bodies of the
kind-neutral tokens live in :mod:`local_operator.artifacts` (also stdlib-only)
and are re-exported or aliased here, so THIS module's pinned surface keeps
resolving for the lane's tests and consumers.

**The cascade order is FROZEN** (architect design, 2026-10-08, decisions
D4–D9): Radient → FAL → OpenAI → honest error, first match wins, and the
executor fails FORWARD on every rung failure except (a) user cancellation
(stop; no failover) and (b) local validation errors (raised before dispatch).
Media wave-2 appends provider rungs to the order (imagegen.cascade owns the
constant); the relative order of the first three does not change.

- :mod:`local_operator.imagegen.availability` — the sync credential probes
  (the ``createIf`` gate's substrate; no sockets on any session-build path).
- :mod:`local_operator.imagegen.cascade` — the resolver and executor
  (:func:`run_image_cascade`).
- :mod:`local_operator.imagegen.rungs` — the httpx-async rung execution and
  the best-effort provider cancel (no threads, no subprocesses, so nothing
  registers with ``group_reaper`` — see the module docstring there).
- :mod:`local_operator.imagegen.media` — bounded asset downloads.
- :mod:`local_operator.imagegen.errors` — the failure mapping, this lane's
  own copy (no ``stt`` import).

**Availability semantics, loud on purpose** (mirrors ``stt/cascade``): rungs
answer "a credential exists", NOT "the call will succeed". A refused key, an
empty balance or an unreachable model surfaces at call time and fails forward.
Availability is not cached across calls.
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
    "ImageAttempt",
    "ImageOutcome",
    "ImageRoute",
    "ImageRouteResolution",
    "MediaAsset",
    "RungAvailability",
]


class ImageRoute(StrEnum):
    """The closed vocabulary of image-generation routes.

    Values are the wire spellings: they travel in attempt records, the tool's
    ``details`` and the PR/QA vocabulary, so they are pinned here rather than
    re-derived per surface.
    """

    RADIENT = "radient"
    FAL = "fal"
    OPENAI = "openai"
    #: The ChatGPT subscription's Codex-backend image tool (media wave-2,
    #: append-only). Wire spelling is hyphenated, unlike the rest.
    OPENAI_SUB = "openai-sub"
    #: Google's Gemini API (Nano Banana family; media wave-2, append-only).
    GOOGLE = "google"
    #: xAI's Grok Imagine images API (media wave-2, append-only); key or the
    #: Grok OAuth token both ride this route.
    XAI = "xai"
    #: OpenRouter's dedicated Images API (media wave-2, append-only): one key,
    #: the aggregator's whole image catalog.
    OPENROUTER = "openrouter"
    #: No usable route. Appears in availability/refusal payloads only.
    NONE = "none"


#: ``ImageAttempt`` / ``ImageOutcome`` are ALIASES of the generic records — one
#: shape, no isinstance traps (design D2), so a monkeypatched producer cannot
#: hand a consumer a type its ``isinstance`` check misses. ``ImageOutcome``
#: therefore carries ``kind`` (defaulting to ``image`` so legacy keyword
#: constructions keep working; every walk-produced outcome sets it explicitly)
#: and ``cost_source`` beside the fields it always had.
ImageAttempt = JobAttempt
ImageOutcome = JobOutcome


@dataclass(frozen=True)
class ImageRouteResolution:
    """The resolver's report: the chosen route, why, and every rung's state."""

    route: ImageRoute
    reason: str
    #: Fixed cascade order, every route present.
    rungs: tuple[RungAvailability, ...]
