"""Public model output, not an extension of the adapter action protocol.

Screenshots leave the shared context before text summarization. Concise facts
from the model's *visible reply* therefore need the same ordinary assistant
text path as interactive sessions; private reasoning is never an input here.
"""

from __future__ import annotations

import json
from typing import Any, Mapping
from urllib.parse import unquote

from local_operator.evaluation.evidence.models import canonical_bytes, canonical_digest
from local_operator.evaluation.protocol import ActionBatch
from local_operator.evaluation.receipts import RedactionSet

REPLY_VERSION = "1.0"
MAX_PUBLIC_OBSERVATIONS_CHARS = 2_000
_ENVELOPE_KEYS = {"reply_version", "action_batch", "public_observations"}
REJECTED_PUBLIC_REPLY = "(model reply rejected; no public observations accepted)"


def is_public_reply(value: Any) -> bool:
    # Reserve every envelope key: a misspelled/missing version must not silently
    # downgrade a reply with notes to the legacy actions-only interpretation.
    return isinstance(value, Mapping) and bool(_ENVELOPE_KEYS.intersection(value))


def looks_like_public_reply(payload: str) -> bool:
    """Whether a REJECTED reply appears to attempt the reserved envelope.

    Used only to withhold unvalidated output from corrective history and
    rejection evidence; acceptance is unaffected and stays with the strict
    decoder. A literal substring test is bypassed by legal JSON Unicode escapes
    (``public_\\u006fbservations``), and full decoding fails on truncated
    framing — exactly the combination that leaked an encoded known note before
    review round 1 (F1). So scan the raw string-literal SPANS for reserved
    keys: fully decodable names are compared after unescaping, and any
    truncatable prefix (e.g. a malformed ``"public_`` remainder) fails closed.
    Non-string positions are untouched, so legacy prose and actions-only
    rejection replay keep their raw text.
    """
    in_string = False
    escaped = False
    start = 0
    for index, char in enumerate(payload):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            if in_string:
                escaped = True
            continue
        if char != '"':
            continue
        if not in_string:
            in_string = True
            start = index + 1
            continue
        in_string = False
        span = payload[start:index]
        try:
            decoded = json.loads(payload[start - 1 : index + 1])
        except ValueError:
            decoded = None
        if isinstance(decoded, str) and decoded in _ENVELOPE_KEYS:
            return True
        if decoded is None or "\\" in span:
            # An escaped name decodes above only when complete; an undecodable
            # or escaped literal is conservatively tested by prefix too.
            if any(key.startswith(span) for key in _ENVELOPE_KEYS):
                return True
    return False


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            # Do not echo arbitrary keys/values into diagnostics or retry text.
            raise ValueError("model reply contains duplicate JSON keys")
        result[key] = value
    return result


def decode_public_reply(payload: str) -> dict[str, Any]:
    """Require one exact envelope; the legacy decoder keeps its own tolerance."""
    try:
        value = json.loads(payload, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError) as error:
        raise ValueError(
            "model reply must be one duplicate-free JSON object, with no trailing text"
        ) from error
    if not isinstance(value, dict) or set(value) != _ENVELOPE_KEYS:
        raise ValueError(
            "model reply requires exactly reply_version, action_batch, public_observations"
        )
    if value["reply_version"] != REPLY_VERSION:
        raise ValueError("unsupported model reply version")
    notes = value["public_observations"]
    if not isinstance(notes, str) or len(notes) > MAX_PUBLIC_OBSERVATIONS_CHARS:
        raise ValueError(
            "public_observations must be a string of at most "
            f"{MAX_PUBLIC_OBSERVATIONS_CHARS} characters"
        )
    try:
        notes.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("public_observations must be valid Unicode text") from error
    batch = value["action_batch"]
    if not isinstance(batch, dict) or set(batch) != {"actions"}:
        raise ValueError("model reply action_batch requires exactly an actions array")
    return value


def redact_public_reply(payload: str, redactions: RedactionSet) -> str:
    """Drop an entire unsafe note before either evidence or accepted replay.

    Scan decoded text (including percent escapes) before serializing: scanning
    raw JSON alone would miss a credential spelt with JSON unicode escapes.
    This is the existing resolved-secret boundary, not a general secret detector.
    """
    value = decode_public_reply(payload)
    notes = value["public_observations"]
    try:
        redactions.assert_clear(notes)
        redactions.assert_clear(unquote(notes))
    except ValueError:
        value["public_observations"] = "[redacted public observations]"
    return canonical_bytes(value).decode("utf-8")


def public_reply_contract() -> dict[str, Any]:
    """Publish a reproducible reply identity separately from the tool surface.

    The action array schema is borrowed, not copied: its vocabulary and bounds
    remain owned by ActionBatch. Negotiated execution restrictions are declared
    by the existing action_surface metadata/tool digest and still gate parsing.
    """
    batch_schema = ActionBatch.model_json_schema()
    schema = {
        "$defs": batch_schema["$defs"],
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_ENVELOPE_KEYS),
        "properties": {
            "reply_version": {"const": REPLY_VERSION},
            "action_batch": {
                "type": "object",
                "additionalProperties": False,
                "required": ["actions"],
                "properties": {"actions": batch_schema["properties"]["actions"]},
            },
            "public_observations": {"type": "string", "maxLength": MAX_PUBLIC_OBSERVATIONS_CHARS},
        },
    }
    contract = {
        "schema": schema,
        "legacy_plain_action_batch": True,
        "envelope_framing": "single-json-object-no-duplicate-keys-no-trailing-text",
        "binding": (
            "action observation_ids validated against current observation "
            "and negotiated action_surface"
        ),
        "public_observations": (
            "concise new observed facts/progress only; no deliberation or credentials"
        ),
    }
    return {
        "model_reply_contract": canonical_bytes(contract).decode("utf-8"),
        "model_reply_contract_digest": canonical_digest("runner-model-reply-v1", contract),
    }
