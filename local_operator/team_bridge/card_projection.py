"""Fail-closed model projection of conversation wake envelopes.

The human journal may record a card before a turn can hydrate its body. A
journal flush is only tool-pair-safe, not admission-safe. Therefore every model
renderer substitutes one of two content-free status markers. Only the native
ephemeral frame may carry screened source metadata and body together; journal
envelopes have no byte-bound admission proof and NEVER render as model text.
Live presentation capabilities cannot survive transcript reconstruction.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .types import ConversationFeedRef

logger = logging.getLogger(__name__)
PENDING_CARD = "[team_bridge: delivery_pending]"
EARLIER_CARD = "[team_bridge: delivered earlier — open the conversation to re-read]"
_HISTORY_STATE = "_bridge_card_history_state"


def is_conversation_card(details: Mapping[str, Any]) -> bool:
    """Recognize current and pre-fix journal cards without trusting their ids."""
    body = details.get("body")
    return isinstance(body, str) and body.startswith(
        ("Direct message ·", "Channel message ·", "Thread reply ·")
    )


def _identity(details: Mapping[str, Any]) -> tuple[str, str, tuple[str, ...]] | None:
    body = details.get("body")
    if not isinstance(body, str):
        return None
    fields: dict[str, str] = {}
    for line in body.splitlines():
        key, separator, value = line.partition(": ")
        if key in ("tenant_id", "conversation_id", "message_id", "message_ids") and separator:
            if key in fields:
                return None
            fields[key] = value
    keys = ("tenant_id", "conversation_id", "message_id")
    if any(not fields.get(key) for key in keys):
        return None
    ids = tuple(fields.get("message_ids", fields["message_id"]).split(","))
    if not all(ids) or len(set(ids)) != len(ids) or ids[-1] != fields["message_id"]:
        return None
    return fields["tenant_id"], fields["conversation_id"], ids


def _completed_before(details: Mapping[str, Any], runtime: Any) -> bool:
    """Choose a content-free status, NEVER permission to render an envelope.

    There is no persisted binding of envelope bytes to an admission. Completion
    therefore supplies only a historical status hint; it cannot authorize text,
    and neither a mislinked assistant entry nor substituted same-id bytes can
    widen this projection. Every grouped id must independently be completed.
    """
    identity = _identity(details)
    if runtime is None or identity is None:
        return False
    from .store import Store, StoreRefused

    try:
        if identity[0] != runtime.tenant_id or runtime.tenant_id != runtime.owner.tenant_id:
            return False
        for message_id in identity[2]:
            row = Store(runtime.config_dir).body_reservation(runtime.owner, identity[1], message_id)
            if row is None or row.get("state") != "completed":
                return False
        return True
    except (StoreRefused, AttributeError, KeyError, TypeError, ValueError):
        return False


@dataclass(frozen=True)
class _HistoryState:
    completed: Callable[[], bool]


def project_history(messages: list[Any], *, runtime: Callable[[], Any]) -> list[Any]:
    """Attach only a status reader; history never authorizes envelope bytes.

    The native ephemeral frame already includes the screened source and body.
    Raw journal cards remain markers even beside that frame: their bytes cannot
    borrow its admission merely by naming the same server-generated ids.
    """
    projected = []
    for message in messages:
        details = getattr(message, "details", None)
        if (
            getattr(message, "custom_type", None) == "team_bridge_message"
            and isinstance(details, Mapping)
            and is_conversation_card(details)
        ):
            state = _HistoryState(lambda details=details: _completed_before(details, runtime()))
            message = message.model_copy(update={"details": {**details, _HISTORY_STATE: state}})
        projected.append(message)
    return projected


def model_card_text(message: Any, messages: list[Any]) -> str:
    """The common render gate, including cold replay and auxiliary requests."""
    details = message.details
    text = details.get("text", "")
    if not is_conversation_card(details):
        frame = getattr(message, "_bridge_frame_admission", None)
        if isinstance(frame, _FrameAdmission) and (frame.text != text or not frame.valid()):
            return PENDING_CARD
        return text
    state = details.get(_HISTORY_STATE)
    if isinstance(state, _HistoryState) and state.completed():
        return EARLIER_CARD
    logger.info("bridge: card assembly pending code=delivery_pending")
    return PENDING_CARD


@dataclass(frozen=True)
class _FrameAdmission:
    ref: ConversationFeedRef
    text: str
    valid: Callable[[], bool]


@dataclass(frozen=True)
class _PresentedBody:
    ref: ConversationFeedRef
    text: str


def mark_admitted_frame(frame: Any, ref: ConversationFeedRef, valid: Callable[[], bool]) -> None:
    """Attach a request-local capability, outside all serialized model fields."""
    object.__setattr__(
        frame, "_bridge_frame_admission", _FrameAdmission(ref, frame.details["text"], valid)
    )


def carry_presented_body(source: Any, rendered: Any) -> None:
    """Attest the live fence NOW; the dispatch-time reader remains pure.

    Private Python attributes are deliberately not model fields/provider_payload:
    model_dump and JSONL never serialize them. Only ephemeral request copies
    receive them. Historical cards, even legitimately completed ones, cannot
    create a body presentation token.
    """
    admission = getattr(source, "_bridge_frame_admission", None)
    if (
        isinstance(admission, _FrameAdmission)
        and admission.text == source.details.get("text")
        and admission.valid()
    ):
        text = _rendered_text(rendered)
        if text == admission.text:
            object.__setattr__(
                rendered, "_bridge_presented_body", _PresentedBody(admission.ref, text)
            )


def _rendered_text(message: Any) -> str | None:
    content = getattr(message, "content", None)
    if not isinstance(content, list) or len(content) != 1:
        return None
    text = getattr(content[0], "text", None)
    return text if isinstance(text, str) else None


def presented_bridge_ref(messages: Any) -> ConversationFeedRef | None:
    """Purely recognise the exact admitted body PRESENT in a rendered payload.

    Markers, raw envelopes, cards without admission proof, and unstamped body
    bytes return None. A ref is issued only for the exact frame whose live
    owner/generation/consent/policy and reservation checks passed at rendering.
    No reads, side effects, cache or global state: revocation takes effect at
    the next render, not retroactively inside an already rendered request.
    Reusing a payload after external revocation can still recognise its token.
    Tokens are non-serialized, in-memory metadata, not portable authority.
    """
    refs = []
    for message in messages:
        proof = getattr(message, "_bridge_presented_body", None)
        if isinstance(proof, _PresentedBody) and _rendered_text(message) == proof.text:
            if proof.ref not in refs:
                refs.append(proof.ref)
    return refs[0] if len(refs) == 1 else None
