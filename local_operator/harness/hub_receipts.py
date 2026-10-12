"""Human-facing hub facts, shared by the TUI and relay projections.

A hub custom is incoming agent traffic, not a human turn. Its display body must
remain distinct from the model instruction envelope; normalising here prevents
resume and live-history delivery from silently disagreeing about that fact.
This helper neither dispatches nor acknowledges a message: expects_reply records
requested intent only, and cannot prove a reply is still outstanding.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from local_operator.harness.comms import TO_CHILD_INSTRUCTIONS

_ENVELOPE = re.compile(r"^<(parent-message|subagent-message)\b[^>]*>\n([\s\S]*)\n</\1>$")
_LABEL = re.compile(r"\blabel=(['\"])(.*?)\1")
_JOB = re.compile(r"\bjob=(['\"])(.*?)\1")
_PREAMBLES = {text: kind for kind, text in TO_CHILD_INSTRUCTIONS.items()}

#: Namespace the producer writes into a persisted user-role row's
#: ``provider_payload`` when a hub delivery could not stay a CustomMessage.
#: Namespaced so unrelated provider bookkeeping on the row survives untouched,
#: and read only as POSITIVE provenance: a malformed or body-less marker is not
#: a receipt and the caller keeps its ordinary user handling.
HUB_RECEIPT_MARKER_KEY = "hub_receipt"


def hub_receipt_marker(message: object) -> dict[str, Any] | None:
    """The typed hub facts a producer stored on a user-role row, or ``None``.

    The shape mirrors the facts a ``hub_message`` CustomMessage carries
    (direction / expects_reply / steer / body, plus advisory label and job_id
    when a producer had them). Display surfaces classify on this marker rather
    than re-parsing the model envelope; the envelope-parsing path remains only
    for rows written before the marker existed.
    """
    if isinstance(message, Mapping):
        # The TUI child fold works on ENCODED transcript payloads (dicts), not
        # Message objects; the marker lives under the same key either way.
        payload = message.get("provider_payload")
    else:
        payload = getattr(message, "provider_payload", None)
    if not isinstance(payload, Mapping):
        return None
    marker = payload.get(HUB_RECEIPT_MARKER_KEY)
    if not isinstance(marker, Mapping):
        return None
    body = marker.get("body")
    if not isinstance(body, str) or not body.strip():
        return None
    return dict(marker)


def hub_receipt_fields(details: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return full display bytes and retained optional facts, without guessing IDs.

    Structured body wins, including its whitespace. Envelope-only older rows are
    parsed only as this producer's exact wrapper/preamble; unknown envelopes stay
    readable. Trimming is an emptiness test, never a transformation of the body.
    Advisory sender/label fields are sanitised by the existing renderers, not
    trusted here. Existing optional media metadata survives, but no upload or
    attachment endpoint is invented for the text-only hub producer.
    """
    facts = dict(details)
    envelope = details.get("text")
    envelope = envelope if isinstance(envelope, str) else ""
    match = _ENVELOPE.fullmatch(envelope)
    parsed = match.group(2) if match else envelope
    if match:
        header = envelope.split("\n", 1)[0]
        if match.group(1) == "subagent-message":
            facts.setdefault("direction", "to_parent")
            for key, pattern in (("label", _LABEL), ("job_id", _JOB)):
                label = pattern.search(header)
                if label:
                    facts.setdefault(key, label.group(2))
        elif "direction" not in details or details.get("direction") in ("", "to_child"):
            instruction, separator, tail = parsed.partition("\n\n")
            kind = _PREAMBLES.get(instruction)
            if separator and kind:
                parsed = tail
                facts.setdefault("direction", "to_child")
                facts.setdefault("expects_reply", kind == "ask")
                facts.setdefault("steer", kind == "steer")
    body = details.get("body")
    body = body if isinstance(body, str) else (parsed if parsed.strip() else envelope)
    # These new view fields are optional on old relays. A malformed advisory
    # value must not invalidate an otherwise readable native/phone transcript;
    # absence means unknown, never a guessed identity or a false reply verdict.
    # The persisted/model-facing details are not mutated by this projection.
    for key in ("direction", "label", "job_id"):
        if key in facts and not isinstance(facts[key], str):
            del facts[key]
    for key in ("expects_reply", "steer"):
        if key in facts and not isinstance(facts[key], bool):
            del facts[key]
    sender = facts.get("sender")
    if isinstance(sender, Mapping):
        clean = dict(sender)
        for key in ("conversation_name", "session_id", "cwd", "model_label"):
            if key in clean and not isinstance(clean[key], str):
                del clean[key]
        if "pid" in clean and (
            not isinstance(clean["pid"], (str, int)) or isinstance(clean["pid"], bool)
        ):
            del clean["pid"]
        facts["sender"] = clean
    elif "sender" in facts:
        del facts["sender"]
    facts["body"] = body
    facts["text"] = envelope
    return body, facts
