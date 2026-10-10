"""The standard artifact-generation interface: job tokens and result types.

This package is the kind-neutral home for the one artifact job interface the
harness runs (submit / progress / cancel / steer / restart). ``generate_image``
rides it today; video and later artifact kinds extend it without forking the
walk, the progress payload or the cancel path (design:
``docs/design/artifact-generation.md``). ``local_operator/imagegen/`` stays the
IMAGE lane's own package — its wire code, budgets and labels — and re-exports
the generic pieces under its pinned names.

**Import discipline** (mirrors ``stt/``/``tts/``/``imagegen/`` ``__init__``s):
THIS module is stdlib-only. Session-build paths (the tool builder's gate, the
classification roster) import these names and must not drag in HTTP clients,
pydantic or credential stores. The heavier companions are imported by their
own consumers:

- :mod:`local_operator.artifacts.rung` — the rung seam (``RungSpec``,
  ``CancelSupport``, ``CancelHandle``, ``RungResult``, ``RungSkipped``).
- :mod:`local_operator.artifacts.walk` — ``run_job_walk`` and ``make_pause``.
- :mod:`local_operator.artifacts.progress` — ``emit_progress`` and
  ``progress_details`` (the canonical live-progress payload).
- :mod:`local_operator.artifacts.errors` — ``REASON_CLASSES`` and
  ``failure_reason_class``.

Vocabulary notes that the types cannot carry:

- ``route`` fields are plain ``str`` on purpose: the generic layer must not
  know any provider's name. Rung routes travel as the wire spellings their
  kind pins (``"radient"``, ``"fal"``, ...); a kind's resolver may type them
  with its own ``StrEnum`` because its members ARE ``str``.
- ``cost_usd`` carries a figure a provider REPORTED, or - for a
  subscription-funded call only - the published API-equivalent price of what
  the plan funded. ``cost_source`` labels where a figure came from and
  ``billing_basis`` what it means in money terms (``billed`` /
  ``subscription-api-equivalent`` / ``estimated``), with ``cost_provenance``
  naming the doc or provider field, so an estimate or a quota-funded call can
  never masquerade as a charge (design D8; the API-equivalent clause is the
  wave-2 cost rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

__all__ = [
    "ArtifactKind",
    "AttemptOutcome",
    "BillingBasis",
    "CostSource",
    "JobAttempt",
    "JobCancelled",
    "JobOutcome",
    "JobSpec",
    "JobUnavailable",
    "MediaAsset",
    "RungAvailability",
]


class ArtifactKind(StrEnum):
    """The artifact kinds this interface serves, as wire spellings.

    ``video`` is DECLARED, not built (media wave-2): the kind is the
    designated additive field for a future payload and the seam-proof test
    exercises the walk against it today. Only kinds with a real rung set are
    ever EMITTED on a wire.
    """

    IMAGE = "image"
    VIDEO = "video"


#: One rung's outcome in an executor attempt. ``skipped`` is not "not
#: available" — unavailable rungs are never attempted at all — it is a rung the
#: walk reached but did not spend: the overall budget was gone, or the rung's
#: own probe said the request shape (or the account) cannot fund a call.
AttemptOutcome = Literal["ok", "failed", "skipped"]

#: Where a ``cost_usd`` figure (or its documented absence) comes from:
#: ``reported`` = the provider returned it per call; ``rate_table`` = a
#: documented vendor price exists but the call returns no figure, so ``cost_usd``
#: stays ``None``; ``subscription`` = quota-funded, no cash figure exists.
CostSource = Literal["reported", "rate_table", "subscription"]

#: What a ``cost_usd`` AMOUNT means in money terms (additive beside
#: :data:`CostSource`, whose semantics are unchanged; ``None`` whenever there is
#: no amount). ``billed`` = cash/credits the provider actually charged for this
#: call; ``subscription-api-equivalent`` = the published API price of what a
#: subscription (plan quota) funded, labelled so it is NEVER read as a charge —
#: the same convention inference cost uses ("list price x usage", not an
#: invoice); ``estimated`` = a modelled figure, neither reported nor billed.
#: The operator's rule: a subscription-funded generation must still carry an
#: API-equivalent cost so spend across funding classes is comparable.
BillingBasis = Literal["billed", "subscription-api-equivalent", "estimated"]


class JobCancelled(Exception):
    """The in-flight job stopped because the USER asked it to.

    Deliberately NOT ``asyncio.CancelledError`` — that class is the loop's own
    cancellation machinery and must propagate untouched. A kind subclasses
    this (``ImageGenerationCancelled``) and its pause helper raises the
    subclass; the walk treats every subclass as a STOP (no failover) because
    the user asked for the walk to end.
    """


class JobUnavailable(RuntimeError):
    """The walk could not produce an artifact; carries the walk's record.

    ``attempts`` may be empty (no rung was available at all — the caller then
    names the setup remedies); ``resolution`` is the kind's full rung report.
    """

    def __init__(
        self,
        message: str,
        *,
        resolution: object | None = None,
        attempts: tuple[JobAttempt, ...] = (),
    ) -> None:
        super().__init__(message)
        self.resolution = resolution
        self.attempts = attempts


@dataclass(frozen=True)
class RungAvailability:
    """One rung's availability, with the reason a user would be shown.

    Availability answers "is there a credential for this rung", NOT "will the
    call succeed"; a refused key, an empty balance or an unreachable model
    surfaces at call time and fails forward.
    """

    route: str
    available: bool
    reason: str


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
class JobSpec:
    """One job request, in kind-neutral form.

    ``count`` is emitted as the frozen ``num_images`` payload key (historical
    name: the wire vocabulary predates this abstraction and every surface
    reads it). Kind-specific params (image_size/strength/source_url; a future
    video's duration/aspect) stay in the kind's tool params model and reach
    the rungs through the kind's dispatch closure — the walk never parses
    them.
    """

    kind: ArtifactKind
    prompt: str
    count: int = 1
    seed: int | None = None
    model: str | None = None


@dataclass(frozen=True)
class JobAttempt:
    """One rung's attempt inside a walk.

    ``reason_class`` is a small closed token (see
    :data:`local_operator.artifacts.errors.REASON_CLASSES`) so consumers can
    group failures without parsing prose; ``message`` is the human-facing
    sentence.
    """

    route: str
    outcome: AttemptOutcome
    reason_class: str = ""
    message: str = ""
    status_code: int | None = None


@dataclass(frozen=True)
class JobOutcome:
    """A successful walk run: the assets, the rung that produced them, the walk.

    ``kind`` defaults to ``image`` because the v1 image lane constructs this
    type directly in tests and legacy call sites with keyword arguments that
    predate the field; every walk-produced outcome sets it explicitly from the
    job spec, so the default only ever serves direct construction.

    ``cost_usd`` only ever carries a figure a provider REPORTED (design D8);
    ``cost_source`` labels it. ``lineage`` is a named future additive field
    (restart provenance) deliberately not built in v1.
    """

    assets: tuple[MediaAsset, ...]
    route: str
    attempts: tuple[JobAttempt, ...]
    kind: ArtifactKind = ArtifactKind.IMAGE
    #: The model id the provider actually ran (the route's default when the
    #: caller pinned none).
    model: str = ""
    #: Echoed for the caption/receipt so a cancelled or failed re-issue can
    #: repeat or edit the prompt without the model having to remember it.
    prompt: str = ""
    seed: int | None = None
    #: The provider's own job id, for the receipt and for support queries.
    generation_id: str | None = None
    cost_usd: float | None = None
    cost_source: CostSource | None = None
    #: Money meaning of ``cost_usd`` (see :data:`BillingBasis`) and where the
    #: figure comes from (doc source + date, or the provider field that
    #: reported it). Both ``None`` when no amount exists.
    billing_basis: BillingBasis | None = None
    cost_provenance: str | None = None
