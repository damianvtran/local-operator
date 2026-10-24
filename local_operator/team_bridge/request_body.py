"""Assemble ONE admitted conversation BODY into a primary request.

This is the assembly half of the conversation BODY contract (proposal v2 §R2,
capacity delta §R4-C1). It consumes the reviewed frame from
:mod:`local_operator.team_bridge.body_frame` — it never re-implements framing,
escaping or sizing.

WHY assembly is a separate module from the frame: the frame answers "what does
the bytes look like"; this answers "may these bytes enter THIS request, once,
under THIS budget". Keeping them apart means the frame stays a pure function
(other callers, tests and auditors can render it) while every authority and
consumption rule lives in one place that a request builder can call.

THE PRIVACY BOUNDARY IS STRUCTURAL, NOT A PROMISE. The assembled text is
returned to the caller and must never be persisted. The intended consumer is
the harness's existing ``LoopConfig.transform_context`` hook: the loop copies
``list(context.messages)``, renders it, and keeps nothing — so an injected
frame is request-local by construction. Nothing in this module writes to the
transcript, the spool, SQLite, logs, traces, asides or compaction; it has no
I/O at all, and its only failure channel is a typed refusal carrying counts,
never message text. The call sites checked for that claim are recorded on PR
#2867's slice report rather than restated here, because a comment claiming
"this is safe everywhere" is exactly the kind of promise that rots.

THE BUDGET IS AN INPUT. This module never guesses a model's window, a token
count or a provider's headroom: ``BodyBudget.available_bytes`` is the caller's
measured capacity for this one frame, after its own fixed prefix and output
reserve. A frame that does not fit is refused (``BodyNotFitted``) and the row
stays pending — no truncation, no chunking, no provider switch.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

from .body_frame import (
    BodyFrameRefused,
    ConversationBodyFrame,
    build_body_frame,
    frame_within_capacity,
    request_size,
)

__all__ = [
    "BodyBudget",
    "BodyNotFitted",
    "CustomFrameInjection",
    "InjectedBody",
    "PendingBody",
    "RequestAuthority",
    "ReservationAlreadyConsumed",
    "TurnFence",
    "admission_digest",
    "assemble_injection",
    "body_details",
    "inject_into_context",
]


class BodyNotFitted(RuntimeError):
    """The whole message does not fit this request. The row stays pending."""

    def __init__(self, needed_bytes: int, available_bytes: int) -> None:
        super().__init__(f"body frame needs {needed_bytes} bytes, budget is {available_bytes}")
        self.code = "body_not_fitted"
        self.needed_bytes = needed_bytes
        self.available_bytes = available_bytes


class ReservationAlreadyConsumed(RuntimeError):
    """A retry tried to re-consume a reservation this turn already used."""

    def __init__(self, reservation_id: str) -> None:
        super().__init__("reservation already consumed in this turn")
        self.code = "reservation_already_consumed"
        # The id is a local opacity handle, not message content.
        self.reservation_id = reservation_id


@dataclass(frozen=True)
class BodyBudget:
    """The caller's MEASURED capacity for one frame, in bytes.

    ``measured_from`` names where the number came from (the effective model
    spec, a provider probe, a recorded ledger). It is required because an
    unlabelled budget cannot be audited later — a number with no provenance is
    indistinguishable from a guess (the recorded size-ledger gap).
    """

    available_bytes: int
    measured_from: str

    def validated(self) -> "BodyBudget":
        if not isinstance(self.available_bytes, int) or isinstance(self.available_bytes, bool):
            raise BodyFrameRefused("budget_not_int")
        if self.available_bytes < 0:
            raise BodyFrameRefused("budget_negative")
        if not isinstance(self.measured_from, str) or not self.measured_from.strip():
            raise BodyFrameRefused("budget_provenance_missing")
        return self


@dataclass(frozen=True)
class RequestAuthority:
    """The non-serialized admission descriptor carried beside a frame.

    WHAT IT IS FOR: a physical retry must be able to re-prove that the bytes it
    is about to send were admitted for THIS owner under THIS policy tuple,
    without re-screening and without a clearance cache. It therefore carries
    only opaque identity and policy digests — never the body, never a name,
    never a credential, never a bearer token that could be lifted.

    WHY IT MUST NEVER BE SERIALIZED: the descriptor is an OBSERVED-LOCAL fence,
    not a remote capability. Writing it into provider JSON, a trace, a
    transcript row or a log would turn a local check into a portable claim that
    outlives the conditions it describes. Callers pass it as a Python object
    only; this module has no ``to_dict``/``json`` surface, deliberately.
    """

    #: Digest over (owner key, policy tuple, screened body hash). Digests, not
    #: values: a retry compares them, an observer cannot reconstruct anything.
    admission_digest: str
    #: The screening receipt this admission rests on (contentless id).
    receipt_id: str
    #: The exact policy/consent tuple the admission was taken under.
    policy_digest: str


@dataclass(frozen=True)
class PendingBody:
    """One admitted message waiting for a request that can carry it whole."""

    reservation_id: str
    source: ConversationBodyFrame
    conversation_label: str
    sender_label: str
    body: str
    authority: RequestAuthority
    #: The hub's closed sender kind. It belongs with the source frame's typed
    #: fields, but it arrives on the fetched body rather than on the row, so it
    #: travels here and overwrites the source's placeholder before rendering.
    sender_kind: str = "unknown"


@dataclass(frozen=True)
class InjectedBody:
    """The assembled frame plus the reservation it consumed.

    ``text`` is the exact host-owned frame. It is returned ONCE per turn per
    reservation; the fence (below) is what makes the second attempt refuse.
    """

    text: str
    reservation_id: str
    size_bytes: int
    authority: RequestAuthority


class TurnFence:
    """One request turn's rendered-byte cache and append counter.

    WHY a cache rather than a claim-that-refuses: the unit of authority is the
    PHYSICAL PROVIDER CALL, and a turn makes several (one per tool step). The
    frame must be rendered once — two renders would be two frames and two
    sets of bytes the model sees — but every call of that turn must carry the
    SAME bytes, because dropping the frame on call 2 would leave the model
    reasoning about a body it can no longer see while call 1's answer keeps
    that body in context (the S-1 defect).

    WHY the object lives OUTSIDE the request: a retry, a failover or a
    provider re-entry builds a NEW message list; a flag on a message would
    travel with a copy. A fresh turn builds a fresh fence, which is why this is
    per-turn and not per-session.

    ``appended_calls`` counts APPENDS, not dispatches (S-23): it proves the
    bytes entered a request this turn, never that a provider received them.
    """

    def __init__(self) -> None:
        self._reservation_id: str | None = None
        self._text: str | None = None
        self._appended_calls = 0

    @property
    def reservation_id(self) -> str | None:
        return self._reservation_id

    @property
    def appended_calls(self) -> int:
        return self._appended_calls

    @property
    def rendered(self) -> str | None:
        return self._text

    def cache(self, reservation_id: str, text: str) -> None:
        """Fix this turn's bytes to ONE reservation, once.

        A second reservation in the same turn is refused rather than queued:
        one request can honestly carry one whole message, and quietly
        preferring the newer one would strand the first with no record.
        """

        if not isinstance(reservation_id, str) or not reservation_id:
            raise BodyFrameRefused("reservation_id_invalid")
        if not isinstance(text, str) or not text:
            raise BodyFrameRefused("frame_text_invalid")
        if self._reservation_id is not None:
            if self._reservation_id != reservation_id:
                raise ReservationAlreadyConsumed(reservation_id)
            return
        self._reservation_id = reservation_id
        self._text = text

    def record_append(self) -> None:
        """Note that the cached bytes went into a request this turn."""

        if self._text is None:
            raise BodyFrameRefused("append_before_render")
        self._appended_calls += 1


class CustomFrameInjection:
    """Append a rendered frame to a request as ONE transient custom row.

    WHY a class and not a bare function: the call site needs the same
    ``custom_type`` the renderer already understands, and naming it once here
    keeps the transform thin. The input list is never mutated — a caller that
    kept a reference (the durable context) must not gain a frame it could later
    persist.
    """

    #: The family type ``harness/render.py`` already knows. An unlisted custom
    #: type is dropped as bookkeeping and the body would never reach the model.
    CUSTOM_TYPE = "team_bridge_message"

    @classmethod
    def append(cls, messages: Sequence[Any], text: str, *, message_id: str) -> list[Any]:
        from local_operator.harness.types import CustomMessage

        return list(messages) + [
            CustomMessage(
                custom_type=cls.CUSTOM_TYPE,
                attribution="user",
                details={"text": text},
                id=message_id,
            )
        ]


def assemble_injection(
    pending: PendingBody,
    budget: BodyBudget,
    fence: TurnFence,
) -> InjectedBody:
    """Render ``pending`` whole, or refuse it — never trim, never split.

    Order matters and is the contract:

    1. validate the budget (an unlabelled or negative budget is a bug, not a
       small budget);
    2. claim the turn BEFORE rendering, so a rendering failure cannot leave the
       turn looking unused while a retry duplicates work;
    3. build the frame (the frame module refuses malformed ids, an over-cap
       body and a non-text label);
    4. fit the ESCAPED frame against the measured budget.
    """

    budget.validated()
    if fence.reservation_id not in (None, pending.reservation_id):
        raise ReservationAlreadyConsumed(pending.reservation_id)
    if fence.rendered is not None:
        # A second call of the SAME turn re-uses the bytes; re-rendering would
        # be a second frame rather than a re-send of the admitted one.
        cached = fence.rendered
        size = request_size(cached)
        return InjectedBody(
            text=cached,
            reservation_id=pending.reservation_id,
            size_bytes=size,
            authority=pending.authority,
        )
    source = pending.source
    if pending.sender_kind != source.sender_kind:
        # The fetched classification wins over the row's placeholder — the row
        # is contentless, so the hub is the only honest source for this field.
        source = ConversationBodyFrame(
            tenant_id=source.tenant_id,
            conversation_id=source.conversation_id,
            message_id=source.message_id,
            sender_kind=pending.sender_kind,
            mention=source.mention,
            conversation_seq=source.conversation_seq,
        ).validated()
    frame = build_body_frame(
        source,
        conversation_label=pending.conversation_label,
        sender_label=pending.sender_label,
        body=pending.body,
    )
    size = request_size(frame)
    if not frame_within_capacity(frame, request_bytes=budget.available_bytes):
        raise BodyNotFitted(size, budget.available_bytes)
    fence.cache(pending.reservation_id, frame)
    return InjectedBody(
        text=frame,
        reservation_id=pending.reservation_id,
        size_bytes=size,
        authority=pending.authority,
    )


def body_details(injected: InjectedBody) -> dict[str, Any]:
    """The ``details`` mapping the existing renderer expects.

    WHY this exact shape: ``harness/render.py`` renders a
    ``team_bridge_message`` custom entry as a user message carrying
    ``details["text"]`` verbatim, and that branch is the ONE place inbound
    bridge content becomes model input. Reusing it means assembly adds no
    renderer branch — and a frame cannot reach the model through a path whose
    framing rules nobody reviewed.

    Only the frame text is copied. The authority descriptor is deliberately
    absent: ``details`` is a persisted shape on the receiving path, and this
    mapping must stay safe to look at even though assembly output never enters
    durable history.
    """

    return {"text": injected.text}


def inject_into_context(
    messages: Sequence[Any],
    injected: InjectedBody,
    *,
    custom_type: str,
    message_id: str,
) -> list[Any]:
    """Return a NEW message list with the frame appended as one transient row.

    WHY this composes with the harness instead of replacing it: the only
    supported insertion point for host-authored context is a ``CustomMessage``
    in ``context.messages`` — ``harness/loop.py`` copies that list, renders it
    through the one ``convert_to_llm`` and persists nothing, so a frame added
    here is request-local by construction. ``custom_type`` must be the family
    type the renderer already knows (``TEAM_BRIDGE_MESSAGE_TYPE``), because an
    unlisted custom type is dropped as bookkeeping and the body would silently
    never reach the model.

    The input list is never mutated: a caller that kept a reference (the
    durable context) must not gain a frame it can later persist.
    """

    from local_operator.harness.types import CustomMessage

    return list(messages) + [
        CustomMessage(
            custom_type=custom_type,
            attribution="user",
            details=body_details(injected),
            id=message_id,
        )
    ]


def admission_digest(owner_key: str, policy_digest: str, body_sha256: str, receipt_id: str) -> str:
    """One opaque digest binding owner, policy, exact bytes and receipt.

    Pure and side-effect free; it records nothing. Callers recompute it instead
    of trusting a stored copy, which is what makes the descriptor an observed
    local fact rather than a cached clearance.
    """

    if not all(
        isinstance(value, str) and value
        for value in (
            owner_key,
            policy_digest,
            body_sha256,
            receipt_id,
        )
    ):
        raise BodyFrameRefused("authority_input_invalid")
    material = "\x1f".join((owner_key, policy_digest, body_sha256, receipt_id))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
