"""Typed value objects for the classification layer.

These are the wire-facing shapes of the decision vendors, kept deliberately
dumb: no behaviour, no I/O, no imports from the rest of the harness. The
interface contract is ``docs/design/classification-layer.md`` §4 — read it
before changing a field, because three implementation slices code against it
and, more importantly, because the fields mirror TypeSafe's native request and
answer shape rather than a harness-shaped paraphrase. Anything this module
"improves" is something the cascade's second and third legs then have to
un-improve.

WHY ``criteria`` IS A PER-KIND UNION AND NOT ALWAYS A MAPPING
============================================================

A ``choice``/``noul`` question carries ``{option_id: description}``; a
``score`` question carries an ARRAY of level descriptions, and the vendor
answers with a float that indexes that array. Sending a score level list as a
mapping is a schema failure, and sending a choice's options as a list is one
too — measured against the OpenRouter alpha route on 2026-09-18, which answers
a malformed criteria shape with a 400 naming ``questions.<id>.criteria``
(see :func:`local_operator.classification.vendors._schema_error`). The
dataclasses therefore *validate* the shape at construction
(:meth:`Question.__post_init__`): the layer's own bug becomes a local
``ValueError`` in a stack trace we own, instead of a vendor 400 whose body we
have to parse and whose resulting behaviour is "disable the question shape and
try again next message".

``score`` criteria are normalised to a ``tuple`` so the frozen dataclass stays
hashable and cannot be mutated by a caller that kept a reference to the list it
passed in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import SecretStr

if TYPE_CHECKING:
    # Import-graph hygiene, not tidiness: this module is reachable from the
    # session composition root, and `local_operator.credentials` pulls the
    # crypto/secret-broker stack with it. The annotation is a string anyway
    # (`from __future__ import annotations`), so the runtime import buys
    # nothing.
    from local_operator.credentials import CredentialManager

QuestionKind = Literal["choice", "noul", "score"]

#: What a ``Recommendation.skipped`` may say. Enumerated here rather than in
#: ``service.py`` because it is part of the public contract (§4) and callers
#: branch on these strings.
SkipReason = Literal["disabled", "no-vendor", "empty-roster", "timeout", "error", "circuit-open"]


class DecisionVendorError(RuntimeError):
    """Any leg failing to produce a usable answer; the cascade catches this and moves on.

    Everything here is "weather": a transport failure, ``401``/``403``,
    ``429``, ``529``, a 5xx, or an unusable 200 body. The cascade's job is to
    treat all of them identically and try the next leg, so this one class
    carries the distinction only in :attr:`kind` — for logs and for the circuit
    breaker's accounting — never in control flow.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        status: int | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        #: One of ``transport`` | ``auth`` | ``rate-limit`` | ``overloaded`` |
        #: ``server`` | ``http`` | ``response``. Free-form on purpose: a new
        #: upstream failure mode should not require a new exception class to be
        #: classified in a log line.
        self.kind = kind
        #: The upstream HTTP status when there was one, else ``None``.
        self.status = status
        #: How many attempts the cascade spent before it gave up across all legs.
        #: Carried here so the ONE warning the operator sees can say whether the
        #: failure was a single refusal or a retried-and-still-failing leg.
        self.attempts = attempts


class DecisionSchemaError(RuntimeError):
    """Our request shape was rejected — a bug in THIS layer, not vendor weather.

    Deliberately NOT a subclass of :class:`DecisionVendorError`. If it were,
    every ``except DecisionVendorError`` in the cascade and in future callers
    would silently swallow our own bug into "try the next vendor", which is
    exactly the failure §4 forbids: the same malformed question would be sent
    to leg two, then to leg three, and the mismatch would show up as "every
    decision vendor is down" in the logs while the real cause — one wrong field
    — never gets looked at. A sibling class makes falling through require a
    deliberate edit.

    ``shape`` names the question the vendor complained about, when the error
    body names one (the OpenRouter alpha route reports a Zod ``path`` such as
    ``["questions", "recommend_skill", "criteria"]``). The service disables that
    question id for the rest of the session; ``None`` means "we could not tell
    which question", which disables nothing but still refuses to fall through.
    """

    def __init__(
        self, message: str, *, shape: str | None = None, status: int | None = None
    ) -> None:
        super().__init__(message)
        self.shape = shape
        self.status = status


@dataclass(frozen=True)
class Question:
    """One typed question sent alongside the state.

    ``criteria`` is ``{option_id: description}`` for ``choice``/``noul`` and a
    tuple of level descriptions for ``score``. See the module docstring for why
    the shape is enforced here.
    """

    id: str
    kind: QuestionKind
    instructions: str
    criteria: dict[str, str] | tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind == "score":
            if isinstance(self.criteria, dict):
                raise ValueError(
                    f"score question {self.id!r} takes an ARRAY of level descriptions, "
                    "not a mapping — the vendor reads a score criterion's value by index"
                )
            # A tuple, not the caller's list: the dataclass is frozen and
            # hashable, and a caller that keeps the list it passed in must not
            # be able to mutate a question already handed to a vendor.
            object.__setattr__(self, "criteria", tuple(self.criteria))
            return
        if not isinstance(self.criteria, dict):
            raise ValueError(
                f"{self.kind} question {self.id!r} takes a mapping of option id to "
                f"description, got {type(self.criteria).__name__}"
            )


@dataclass(frozen=True)
class Answer:
    """One typed answer.

    ``value`` is the choice id, a ``noul`` probability, or the float a ``score``
    answer reported (the vendor's score is a float over the level array, not a
    level name — measured 2026-09-18: ``{"type": "score", "score": 1.11,
    "legend": {"0": "trivial", "1": "routine", "2": "complex"}}``). The legend
    stays in ``probabilities`` as the vendor sent it, keyed by level index, so a
    caller that wants the level name can resolve it without this layer guessing.
    """

    id: str
    kind: QuestionKind
    value: str | float
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None


@dataclass(frozen=True)
class DecisionRequest:
    """A state plus its questions. ``state`` is already bounded — see §5."""

    state: str | dict[str, Any]
    questions: tuple[Question, ...]


@dataclass(frozen=True)
class DecisionResponse:
    """What a leg answered, with its own accounting.

    ``cost_usd`` is ``None`` when the vendor reported no ``usage.cost``: the
    contract is explicit that a missing cost is reported as missing, never
    estimated. (No price row for a decision model exists in this repo yet, so
    the "compute it from the configured price row" path has nothing to read.)

    The token counts follow the SAME rule, and it has to be stated because ``0``
    is a legal figure here: ``usage`` is optional on the wire (the route answers
    200 with an ``answers`` block and no ``usage`` at all), and the Radient route
    is documented as billing with output tokens zero. So a count the vendor did
    not send is ``None``, never ``0`` — a caller that conflated the two would
    print a fabricated ``tokens=0/0`` beside a real cost, which is exactly the
    unreadable figure this accounting exists to remove.
    """

    vendor: str
    model: str
    answers: dict[str, Answer]
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_s: float = 0.0


@runtime_checkable
class DecisionVendor(Protocol):
    """One cascade leg.

    Credential resolution is per-vendor because Radient's is an OAuth session
    held in ``AuthStore`` while the others are plain credential rows — and it
    takes the manager as an argument so a caller can probe a leg against a
    manager it does not own, which is what :func:`resolve_vendor` does while
    deciding which leg is usable.
    """

    name: str

    async def credential(self, manager: "CredentialManager") -> SecretStr | None:
        """The bearer this leg would send, or ``None`` when it has none."""
        ...

    async def decide(self, request: DecisionRequest, *, timeout_s: float) -> DecisionResponse:
        """Answer ``request`` or raise :class:`DecisionVendorError`/:class:`DecisionSchemaError`."""
        ...
