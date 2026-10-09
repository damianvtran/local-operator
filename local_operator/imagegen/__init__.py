"""Image generation cascade: the shared tokens and result types.

The package is deliberately **import-light** (stdlib only, like ``stt/``'s
``__init__``): the tool builder, the session factory's classification roster
and the tests all import these names on paths that must not drag in the HTTP
client or the credential stores.

**The cascade order is FROZEN** (architect design, 2026-10-08, decisions
D4–D9): Radient → FAL → OpenAI → honest error, first match wins, and the
executor fails FORWARD on every rung failure except (a) user cancellation
(stop; no failover) and (b) local validation errors (raised before dispatch).

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
from typing import Literal

__all__ = [
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
    #: No usable route. Appears in availability/refusal payloads only.
    NONE = "none"


#: One rung's outcome in an executor attempt. ``skipped`` is not "not
#: available" — unavailable rungs are never attempted at all — it is a rung the
#: walk reached but did not spend: the overall budget was gone, or (Radient)
#: the affordability probe said the account cannot fund this request.
AttemptOutcome = Literal["ok", "failed", "skipped"]


@dataclass(frozen=True)
class RungAvailability:
    """One rung's availability, with the reason a user would be shown.

    Availability answers "is there a credential for this rung", NOT "will the
    call succeed" — see :func:`local_operator.imagegen.cascade.resolve_image_route`.
    """

    route: ImageRoute
    available: bool
    reason: str


@dataclass(frozen=True)
class ImageRouteResolution:
    """The resolver's report: the chosen route, why, and every rung's state."""

    route: ImageRoute
    reason: str
    #: Fixed cascade order, every route present.
    rungs: tuple[RungAvailability, ...]


@dataclass(frozen=True)
class ImageAttempt:
    """One rung's attempt inside :func:`local_operator.imagegen.cascade.run_image_cascade`.

    ``reason_class`` is a small closed token (``insufficient_balance``,
    ``insufficient_credits``, ``unauthorized``, ``rate_limited``, ``timeout``,
    ``network``, ``upstream``, ``refused``, ``cancelled``,
    ``invalid_response``, ``unknown``) so consumers can group failures without
    parsing prose; ``message`` is the human-facing sentence.
    """

    route: ImageRoute
    outcome: AttemptOutcome
    reason_class: str = ""
    message: str = ""
    status_code: int | None = None


@dataclass(frozen=True)
class MediaAsset:
    """One downloaded asset: bytes plus the facts surfaces need to render it."""

    data: bytes
    content_type: str
    source_url: str
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None


@dataclass(frozen=True)
class ImageOutcome:
    """A successful cascade run: the assets, the rung that produced them, and the walk."""

    assets: tuple[MediaAsset, ...]
    route: ImageRoute
    attempts: tuple[ImageAttempt, ...]
    #: The model id the provider actually ran (the route's default when the
    #: caller pinned none).
    model: str = ""
    #: Echoed for the caption/receipt so a cancelled or failed re-issue can
    #: repeat or edit the prompt without the model having to remember it.
    prompt: str = ""
    seed: int | None = None
    #: The provider's own job id, for the receipt and for support queries.
    generation_id: str | None = None
    #: Radient reports per-generation cost; FAL/OpenAI bill their own key with
    #: no returned figure, so this stays ``None`` there.
    cost_usd: float | None = None
