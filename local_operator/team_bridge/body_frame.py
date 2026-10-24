"""The ONE host-owned frame for a conversation BODY in a primary request.

Contract: ``conversation-ephemeral-body-capacity-design-delta.md`` (fixed/closed
request framing, untrusted-label boundary, exact-bytes decode) and proposal v2
§R2/§R4 (detached primary-request only, whole message, no truncation). This
module implements the REPRESENTATION half of that contract and nothing else: it
does not fetch, screen, admit, wake, schedule or persist.

WHY this is not ``card.build_conversation_card``: that builder is the HUMAN
surface, and its labels are TRUSTED header fields because a human reads a
rendered line. Here every server-resolved string — both labels and the BODY —
is untrusted DATA and must appear only inside the delimited object, never in the
``VERIFIED SOURCE`` section or as a host instruction. Sharing one builder would
force one of the two surfaces to lie about which strings it trusts, so the two
stay separate by contract rather than by accident.

WHY the escaping is not just ``json.dumps``: a JSON string may legally contain
``<``/``>``/``&``/backticks/brackets, and a decoded BODY is free-form text, so a
value could otherwise emit the frame's own delimiter syntax (``<`` / ``>`` in
markup, ``[``/``{`` in a would-be structured block). Escaping those characters
INSIDE strings keeps the object unambiguous. This is representational
delimiting only: it is NOT a claim that a model cannot follow an injection, and
nothing here grants authority — mention is attention metadata, and an inbound
message never starts new work.

WHY decoding must be byte-exact: the screened hash is computed over the exact
canonical UTF-8 bytes. Trimming, tag stripping, Unicode normalisation or
truncation at render time would silently deliver different bytes than the ones
that passed screening, so :func:`decode_untrusted_object` is specified to
recover exactly what was encoded, and the capacity rule sizes the ESCAPED
representation (escaping can only grow it).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "BodyFrameRefused",
    "ConversationBodyFrame",
    "MAX_BODY_BYTES",
    "SENDER_KINDS",
    "build_body_frame",
    "decode_untrusted_object",
    "frame_within_capacity",
    "request_size",
]

#: The server's ACCEPTED maximum for a conversation note/question BODY, in
#: bytes of NFC-normalised UTF-8 (``internal/bridge/bridge.go``'s
#: ``NoteQuestionBodyMaxBytes`` = 16 × 1024). ``screening._TYPE_CAPS`` holds the
#: same number for the same reason; the two must not drift, because a body that
#: passes one and fails the other is undeliverable in both directions.
MAX_BODY_BYTES = 16 * 1024

#: Closed sender-kind vocabulary. ``unknown`` is a REAL state, not a
#: placeholder: an older backend omits the field and the frame must then say it
#: does not know, rather than infer a kind from a display label.
SENDER_KINDS = ("human", "agent", "unknown")

#: Escaped INSIDE JSON strings only (see the module docstring). ``\u`` spellings
#: are ordinary JSON escapes, so ``json.loads`` restores the original character.
_DELIMITER_ESCAPES = {
    "<": "\\u003c",
    ">": "\\u003e",
    "&": "\\u0026",
    "`": "\\u0060",
    "[": "\\u005b",
    "]": "\\u005d",
    "{": "\\u007b",
    "}": "\\u007d",
    # The Unicode LINE/paragraph separators. Not markup, but a line-oriented
    # reader (or a terminal, or a log line) treats them as breaks, so an
    # unescaped one lets a body print the host's own end marker as a line of
    # its own. JSON escapes keep the decoded value byte-identical.
    "\u0085": "\\u0085",
    "\u2028": "\\u2028",
    "\u2029": "\\u2029",
}

#: Display-label bound (design delta: labels keep the existing 200-character
#: display bound; that bound never applies to the BODY). A label is presentation
#: only, so bounding it cannot change the screened bytes or their hash.
_LABEL_MAX = 200

#: The trusted section's closing line. The object's keys and these markers are
#: HOST-owned text; no untrusted value may contribute to them.
_SOURCE_MARKER = "VERIFIED SOURCE"
_OBJECT_MARKER = "UNTRUSTED DATA OBJECT"
_OBJECT_END = "END UNTRUSTED DATA OBJECT"
_BODY_REPRESENTATION = "JSON string; decode to canonical screened BODY"

#: Object key order is fixed so the frame is deterministic and a size ledger can
#: be reproducible; a caller cannot add a key.
_OBJECT_KEYS = ("conversation_label", "sender_label", "body")


class BodyFrameRefused(ValueError):
    """A local refusal. ``code`` is stable and machine-readable.

    Refusals are structural (a non-string body, an unclosed sender kind, an
    over-capacity whole message). A refusal must never be turned into a truncated
    or partially escaped frame: half a message is worse than a hold.
    """

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


@dataclass(frozen=True)
class ConversationBodyFrame:
    """The validated, typed fields of one admitted conversation message.

    These are the TRUSTED section's inputs: closed IDs, a closed ``sender_kind``,
    the mention flag and the conversation sequence. Nothing here is a display
    label (labels are untrusted data) and nothing here is authority — the
    caller has already screened and admitted the body, and this type only
    carries that decision's identity into the frame.

    Distinct from ``connector.conversation.ConversationCardMetadata`` on
    purpose: that type's labels are trusted header text for a human card, while
    the same strings here must sit inside the untrusted object.
    """

    tenant_id: str
    conversation_id: str
    message_id: str
    sender_kind: str
    mention: bool
    conversation_seq: int

    def validated(self) -> "ConversationBodyFrame":
        """Refuse an incomplete or malformed source. No field is inferred."""

        # The locators are re-validated with the SAME closed patterns the rest
        # of the package uses (``types``/``read_context``) instead of a local
        # rule. A second spelling here would let a value one layer rejects
        # reach the other, and every one of these strings is rendered INTO the
        # trusted section: a newline in ``message_id`` could forge the very
        # sender_kind/mention lines the frame exists to keep host-owned.
        from .read_context import TENANT_ID_PATTERN
        from .types import CONVERSATION_ID_PATTERN, is_conversation_message_id

        if not isinstance(self.tenant_id, str) or re.fullmatch(
            TENANT_ID_PATTERN, self.tenant_id
        ) is None:
            raise BodyFrameRefused("source_tenant_id_invalid")
        if not isinstance(self.conversation_id, str) or re.fullmatch(
            CONVERSATION_ID_PATTERN, self.conversation_id
        ) is None:
            raise BodyFrameRefused("source_conversation_id_invalid")
        if not is_conversation_message_id(self.message_id):
            raise BodyFrameRefused("source_message_id_invalid")
        if self.sender_kind not in SENDER_KINDS:
            # A caller-chosen kind would let an agent message claim to be human
            # (or the reverse) in the one section a model is told to trust.
            raise BodyFrameRefused("source_sender_kind_invalid")
        if not isinstance(self.mention, bool):
            raise BodyFrameRefused("source_mention_invalid")
        if not isinstance(self.conversation_seq, int) or isinstance(
            self.conversation_seq, bool
        ):
            raise BodyFrameRefused("source_conversation_seq_invalid")
        if self.conversation_seq < 1:
            # 1-based, matching ``ConversationFeedRef``: a sequence is a
            # position, and 0 is the sentinel the hub uses for "no cursor".
            raise BodyFrameRefused("source_conversation_seq_invalid")
        return self


def _escape_untrusted(value: Any) -> str:
    """JSON-quote ``value`` and escape the delimiter family INSIDE the string.

    Non-strings are refused rather than coerced: a ``None`` label must render as
    an explicit absence chosen by the caller, never as ``"None"`` text that a
    model could read as a name.
    """

    if not isinstance(value, str):
        raise BodyFrameRefused("untrusted_value_not_text")
    # ``ensure_ascii=False`` keeps multibyte text readable in the frame, which
    # is also what makes the payload's escaped size match what a model sees.
    encoded = json.dumps(value, ensure_ascii=False)
    return "".join(_DELIMITER_ESCAPES.get(character, character) for character in encoded)


def _bounded_label(value: Any) -> str:
    """Collapse whitespace and bound a display label — never the BODY.

    Mirrors ``connector.conversation._one_line``'s behaviour (one line, 200
    chars) without importing it: this module must stay independent of the
    connector so the frame can be rendered by any future primary-request caller.
    """

    if not isinstance(value, str):
        raise BodyFrameRefused("untrusted_value_not_text")
    bounded = " ".join(value.split())[:_LABEL_MAX]
    try:
        bounded.encode("utf-8")
    except UnicodeEncodeError as failure:
        # A lone surrogate cannot be UTF-8 encoded. Refusing it HERE keeps the
        # failure a typed local refusal rather than a raw codec error raised
        # later from request sizing.
        raise BodyFrameRefused("untrusted_label_invalid_encoding") from failure
    return bounded


def _untrusted_object(values: Mapping[str, Any]) -> str:
    parts = [f'"{key}":{_escape_untrusted(values[key])}' for key in _OBJECT_KEYS]
    return "{" + ", ".join(parts) + "}"


def build_body_frame(
    source: ConversationBodyFrame,
    *,
    conversation_label: str,
    sender_label: str,
    body: str,
) -> str:
    """Render the WHOLE admitted message as one primary-request block.

    One message per call, one message per request: the caller must not
    concatenate two frames, and this function never splits one frame across
    calls. Truncation is deliberately unavailable — an over-capacity message is
    refused so the pending row can wait for a whole-message window (proposal
    §R4: bounds and omission must never strand or complete an accepted row).
    """

    source.validated()
    if not isinstance(body, str):
        raise BodyFrameRefused("body_not_text")
    try:
        body_bytes = body.encode("utf-8")
    except UnicodeEncodeError as failure:
        raise BodyFrameRefused("body_invalid_encoding") from failure
    if len(body_bytes) > MAX_BODY_BYTES:
        # The canonical screened bytes, not the escaped form, decide whether the
        # server would have accepted the message at all.
        raise BodyFrameRefused("body_too_large")

    mention_line = "mention: true" if source.mention else "mention: false"
    lines = [
        _SOURCE_MARKER,
        f"tenant_id: {source.tenant_id}",
        f"conversation_id: {source.conversation_id}",
        f"message_id: {source.message_id}",
        f"sender_kind: {source.sender_kind}",
        mention_line,
        f"conversation_seq: {source.conversation_seq}",
        f"body_representation: {_BODY_REPRESENTATION}",
        _OBJECT_MARKER,
        _untrusted_object(
            {
                "conversation_label": _bounded_label(conversation_label),
                "sender_label": _bounded_label(sender_label),
                # The BODY is passed through untouched: any normalisation here
                # would change the bytes whose hash screening verified.
                "body": body,
            }
        ),
        _OBJECT_END,
    ]
    return "\n".join(lines)


def frame_within_capacity(frame: str, *, request_bytes: int) -> bool:
    """Does the ESCAPED frame fit a measured request budget?

    ``request_bytes`` is the caller's measured effective request capacity after
    its own fixed prefix and output reserve — this module never guesses a model
    spec or a token count. Escaping can grow a payload past its raw size, which
    is exactly why the check is on the rendered frame.
    """

    if not isinstance(frame, str) or not isinstance(request_bytes, int):
        raise BodyFrameRefused("capacity_argument_not_int")
    if request_bytes < 0:
        raise BodyFrameRefused("capacity_budget_negative")
    return request_size(frame) <= request_bytes


def request_size(frame: str) -> int:
    """The frame's UTF-8 size — one scalar for the request-size ledger."""

    return len(frame.encode("utf-8"))


def decode_untrusted_object(frame: str) -> dict[str, str]:
    """Recover the object's strings EXACTLY as they were encoded.

    Used by tests and by any future auditor to prove the round trip the contract
    requires. It is not a receive path: nothing in the runtime trusts text a
    model echoes back.
    """

    # Match the marker AND its newline together: a frame with a stray space
    # before the object must be refused rather than guessed at.
    marker_line = _OBJECT_MARKER + "\n"
    start = frame.find(marker_line)
    if start < 0:
        raise BodyFrameRefused("frame_object_missing")
    start += len(marker_line)
    end = frame.find("\n" + _OBJECT_END, start)
    if end < 0:
        raise BodyFrameRefused("frame_object_unterminated")
    try:
        parsed = json.loads(frame[start:end])
    except ValueError as failure:
        raise BodyFrameRefused("frame_object_malformed") from failure
    if not isinstance(parsed, dict) or set(parsed) != set(_OBJECT_KEYS):
        raise BodyFrameRefused("frame_object_keys")
    for key in _OBJECT_KEYS:
        if not isinstance(parsed[key], str):
            raise BodyFrameRefused("frame_object_not_text")
    return {key: parsed[key] for key in _OBJECT_KEYS}
