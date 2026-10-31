"""Fail-closed model projection of conversation wake envelopes.

The human journal may record a card before a turn can hydrate its body. A
journal flush is only tool-pair-safe, not admission-safe. Therefore every model
renderer substitutes a structural pending state unless the turn that owns the
matching screened reservation has attached its frame to THIS request. Durable
metadata is a locator, never an admission capability; the latter exists only in
an ephemeral request copy and cannot survive transcript reconstruction.
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
_CAPABILITY = "_bridge_card_admission"


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


def _live_binding(runtime: Any) -> dict[str, Any] | None:
    from .authority import authority_fence
    from .store import Store, StoreRefused, _owner_key

    if runtime is None:
        return None
    try:
        if runtime.tenant_id != runtime.owner.tenant_id or authority_fence(runtime) is not None:
            return None
        facts = Store(runtime.config_dir).per_call_admission_state(
            runtime.owner, runtime.session_id
        )
        if not facts or not facts.get("designation_matches"):
            return None
        return {
            "owner": _owner_key(runtime.owner),
            "credential_id": runtime.owner.credential_id,
            "generation": runtime.owner.enrollment_generation,
            "policy_epoch": facts["policy_epoch"],
            "consent_version": facts["consent_version"],
        }
    except (StoreRefused, AttributeError, KeyError, TypeError, ValueError):
        return None


def _row_matches(row: Mapping[str, Any], binding: Mapping[str, Any], runtime: Any) -> bool:
    owner = runtime.owner
    return all(
        row.get(key) == expected
        for key, expected in {
            "owner_key": binding["owner"],
            "tenant_id": owner.tenant_id,
            "account_id": owner.account_id,
            "endpoint_id": owner.endpoint_id,
            "canonical_base_url": owner.canonical_base_url,
            "credential_id": owner.credential_id,
            "enrollment_generation": owner.enrollment_generation,
            "policy_epoch": binding["policy_epoch"],
            "consent_version": binding["consent_version"],
            "designated_session": runtime.session_id,
        }.items()
    )


def _historical_card_matches(details: Mapping[str, Any], runtime: Any, transcript: Any) -> bool:
    """Prove prior ORDERING, not content: completed admission checked the bytes.

    Bodies are never persisted. A render cannot re-prove their content digest;
    it relies on the existing successful settlement contract and server-unique
    message ids. Bare completed flags, foreign epochs and absent assistant
    entries are insufficient. Each grouped id must independently qualify.
    """
    identity = _identity(details)
    binding = _live_binding(runtime)
    if identity is None or binding is None or identity[0] != runtime.tenant_id:
        return False
    from .store import Store, StoreRefused

    try:
        assistant_entries = {
            entry.id: entry
            for entry in transcript.entries()
            if entry.type == "message" and entry.payload.get("role") == "assistant"
        }
        for message_id in identity[2]:
            row = Store(runtime.config_dir).body_reservation(runtime.owner, identity[1], message_id)
            if (
                row is None
                or not _row_matches(row, binding, runtime)
                or row.get("conversation_id") != identity[1]
                or row.get("message_id") != message_id
                or row.get("state") != "completed"
                or not row.get("completed_entry_id")
                or row["completed_entry_id"] not in assistant_entries
                or not (
                    float(row["created_at"])
                    <= assistant_entries[row["completed_entry_id"]].ts
                    <= float(row["updated_at"])
                )
            ):
                return False
        return True
    except (StoreRefused, AttributeError, KeyError, TypeError, ValueError):
        return False


def project_history(
    messages: list[Any], *, runtime: Callable[[], Any], transcript: Any
) -> list[Any]:
    """Attach only ephemeral eligibility; revalidate even reused projections."""
    projected = []
    for message in messages:
        details = getattr(message, "details", None)
        if (
            getattr(message, "custom_type", None) == "team_bridge_message"
            and isinstance(details, Mapping)
            and is_conversation_card(details)
        ):
            candidate = details.get(_CAPABILITY)
            # The live transform already supplied its frame capability. A
            # historical capability is never a cache: its predicate reads the
            # CURRENT runtime, store and journal on each physical rendering.
            if not isinstance(candidate, _Admission) or candidate.frame is None:
                text = str(details.get("text") or "")
                candidate = _Admission(
                    None,
                    text,
                    lambda details=details: _historical_card_matches(
                        details, runtime(), transcript
                    ),
                )
                message = message.model_copy(
                    update={"details": {**details, _CAPABILITY: candidate}}
                )
        projected.append(message)
    return projected


@dataclass(frozen=True)
class _Admission:
    frame: Any
    text: str
    valid: Callable[[], bool]


def admit_matching_cards(
    messages: list[Any],
    *,
    frame: Any,
    tenant_id: str,
    conversation_id: str,
    message_id: str,
    valid: Callable[[], bool],
) -> list[Any]:
    """Authorize request copies only AFTER PendingTurn's per-call live check."""
    projected = []
    for message in messages:
        details = getattr(message, "details", None)
        if (
            getattr(message, "custom_type", None) == "team_bridge_message"
            and isinstance(details, Mapping)
            and is_conversation_card(details)
            and _identity(details) == (tenant_id, conversation_id, (message_id,))
        ):
            text = str(details.get("text") or "")
            message = message.model_copy(
                update={"details": {**details, _CAPABILITY: _Admission(frame, text, valid)}}
            )
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
    capability = details.get(_CAPABILITY)
    if (
        isinstance(capability, _Admission)
        and capability.text == text
        and (capability.frame is None or any(frame is capability.frame for frame in messages))
        and capability.valid()
    ):
        return text
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
