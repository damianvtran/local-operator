"""The rung seam: what every provider adapter on the walk declares and returns.

A rung is a kind-built closure over its concrete params — ``async def call(route)
-> RungResult`` — and the contract binding for all transports (poll, single
request/response, SSE stream) is exactly:

1. Return assets, or raise :class:`RungSkipped` (reached but not spent —
   capability mismatch, affordability), or raise anything else (transport,
   refusal, timeout) and let the walk classify and fail forward.
2. Emit progress only through the kind's ``emit`` with the canonical payload;
   no vendor word escapes.
3. Fill the shared :class:`CancelHandle` the instant a provider job exists and
   clear it the instant the job reaches a terminal state.
4. Resolve its own credential at call time and never report a cost the
   provider did not report.

This module is deliberately NOT stdlib-only in its TYPE vocabulary (the cancel
handle carries a secret) but avoids the runtime import: ``SecretStr`` is
imported for typing only, so the package's import-light property holds for
callers that only need the dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Optional

from local_operator.artifacts import CostSource, MediaAsset

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pydantic import SecretStr

__all__ = [
    "CancelHandle",
    "CancelSupport",
    "RungResult",
    "RungSkipped",
    "RungSpec",
]


class CancelSupport(StrEnum):
    """How far a provider can be asked to stop a running job (RFC §5.1).

    The vocabulary is reused verbatim from the hub-side provider-seam RFC so
    the two lanes cannot drift. v1 DECLARES it per rung and tests pin it; no
    branch reads it yet — the named future consumers are an honest cancel
    receipt ("this provider cannot be stopped") and a surface affordance.
    """

    NONE = "none"
    QUEUED_ONLY = "queued_only"
    SIGNAL = "signal"


@dataclass(frozen=True)
class RungSpec:
    """One rung's declaration: identity, capability, cancel, cost posture.

    Rungs become DATA (design D10): one table of these per kind, so adding a
    provider touches one checklist instead of seven code sites, and the
    resolver, labels and cost posture cannot drift from the executor.
    """

    #: Wire spelling: "radient", "fal", "openai", ... — travels in attempt
    #: records and details.
    route: str
    #: Human label for progress lines and approval text ("Radient", "FAL").
    label: str
    #: Artifact kinds this rung can serve ({"image"} today).
    kinds: frozenset[str]
    #: RFC §5.2 capability vocabulary: t2i | i2i | edit | t2v | i2v.
    capabilities: frozenset[str]
    cancel_support: CancelSupport
    #: "reported" | "rate_table" | "subscription" | None — see
    #: :data:`local_operator.artifacts.CostSource`.
    cost: CostSource | None = None


@dataclass
class CancelHandle:
    """What a best-effort provider cancel needs, updated as the walk proceeds.

    The walk owns one per call and threads it through the rungs: a rung FILLS
    it the instant a provider job exists, and CLEARS it the instant the job
    reaches a terminal state — so a cancel attempt after completion reads
    "none" (nothing left to cancel) rather than firing an ALREADY_COMPLETED
    round-trip. ``credential`` is held only in process memory and never
    printed; it exists so the cancel path does not have to re-resolve (and
    potentially re-refresh) a credential under a 5 s budget.

    ``provider`` is a ROUTE STRING (the kind's wire spelling), not a kind enum
    member: the generic layer must not know any provider's name, and the
    cancel arm that consumes it lives in the image lane, where the string is
    compared against its own ``ImageRoute`` members (which ARE strings).
    """

    provider: str | None = None
    request_id: str | None = None
    model: str | None = None
    #: The hub base a poll-rung cancel posts to (paths hang off it; kept so
    #: the cancel path does not need a second config resolution under a 5 s
    #: budget). URL-carried cancels travel on the absolute ``cancel_url``.
    base_url: str | None = None
    #: A response-carried cancel URL (or the derived fallback), when the
    #: provider has one.
    cancel_url: str | None = None
    credential: Optional["SecretStr"] = field(default=None)

    def clear(self) -> None:
        self.provider = None
        self.request_id = None
        self.model = None
        self.base_url = None
        self.cancel_url = None
        self.credential = None


class RungSkipped(Exception):
    """A rung the walk REACHED but did not spend, with the honest reason.

    Not a failure — the walk records it as a ``skipped`` attempt and moves to
    the next rung. Producers: an affordability probe (the account cannot fund
    this request) and a capability mismatch (a rung with no route for the
    request shape). ``reason_class`` is closed vocabulary (see
    :data:`local_operator.artifacts.errors.REASON_CLASSES`).
    """

    def __init__(self, message: str, *, reason_class: str) -> None:
        super().__init__(message)
        self.reason_class = reason_class


@dataclass(frozen=True)
class RungResult:
    """One successful rung run: the downloaded assets and the provider facts."""

    assets: list[MediaAsset]
    model: str
    generation_id: str | None = None
    #: A figure the provider REPORTED for this call; ``None`` when it reports
    #: none (most providers bill silently).
    cost_usd: float | None = None
    cost_source: CostSource | None = None
