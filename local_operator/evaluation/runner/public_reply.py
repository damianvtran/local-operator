"""Public model output, not an extension of the adapter action protocol.

Screenshots leave the shared context before text summarization. Concise facts
from the model's *visible reply* therefore need the same ordinary assistant
text path as interactive sessions; private reasoning is never an input here.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, get_args
from urllib.parse import unquote

from local_operator.evaluation.action_surface import ActionSurface
from local_operator.evaluation.evidence.models import canonical_bytes, canonical_digest
from local_operator.evaluation.protocol import ComputerAction
from local_operator.evaluation.receipts import RedactionSet

REPLY_VERSION = "1.0"
MAX_PUBLIC_OBSERVATIONS_CHARS = 2_000
_ENVELOPE_KEYS = {"reply_version", "action_batch", "public_observations"}

#: Bounds on how much of a rejected reply's OWN key names may appear in the
#: diagnostic sent back to the model. The reserved keys are safe to name (they
#: come from a fixed set), but any other key is model-supplied text: echoing it
#: whole turns a malformed reply into an unbounded retry prompt and re-opens
#: the replay channel the reserved-key suppression exists to close.
#:
#: A key is quoted WHOLE or not at all -- never truncated. Truncation was the
#: first attempt and it failed closed in neither direction (review round 3):
#: ``repr`` expands escapes after the cut, so the rendered length depended on
#: the input's alphabet; and, far worse, cutting a secret that appeared as a
#: key left a prefix that ``_assert_redacted``'s substring check no longer
#: matched, converting a loud redaction failure into a silent leak. Quoting
#: whole-or-nothing means nothing is ever reshaped on the way out.
_MAX_EXTRA_KEY_CHARS = 40
_MAX_EXTRA_KEYS_SHOWN = 5


def is_quotable_key(key: str) -> bool:
    """Whether an unexpected key may be named back to the model verbatim.

    Conservative by construction: the key must be short, and must render to
    exactly itself under ``repr`` minus the quotes. That second test is what
    makes the bound independent of the alphabet -- anything carrying escapes,
    control characters, or quote marks expands when rendered, so it is counted
    rather than named.
    """
    if len(key) > _MAX_EXTRA_KEY_CHARS:
        return False
    rendered = repr(key)
    return rendered[1:-1] == key


REJECTED_PUBLIC_REPLY = "(model reply rejected; no public observations accepted)"

#: Longest slice of a rejected reply that may be replayed into the model's
#: history or published as evidence. The reply has to be there -- the model
#: must see WHAT it said to fix it, and a reader must see it to diagnose the
#: class -- but a runaway reply (a provider's max-token wall of prose) must not
#: cost the whole window on the retry or the whole bundle on the artifact.
#:
#: One bound for both boundaries, declared here rather than beside the client
#: because ``episode.py`` applies it while publishing the artifact and must not
#: import the provider-backed client to reach it.
MAX_REJECTED_REPLY_CHARS = 4_000

#: Appended to a reply cut by :data:`MAX_REJECTED_REPLY_CHARS`.
REJECTED_REPLY_TRUNCATED = "\n[... reply truncated]"

#: Published in place of a rejected reply whose text cannot be cleared for
#: evidence. Whole-or-nothing: a reply is either published in full (bounded) or
#: replaced by this marker, never published in part, because a partial rendering
#: is exactly what makes a canary stop matching the alarm that exists to catch
#: it.
REJECTED_REPLY_WITHHELD = (
    "(rejected reply withheld: it carried text the episode's redaction set forbids in evidence)"
)

#: JSON's own escape for a non-ASCII or escaping-sensitive character. Decoded
#: before a redaction scan for the same reason percent escapes are: the model's
#: reply reaches evidence as WIRE bytes, and a canary spelt ``\\u0068unter2``
#: matches no substring check on the raw text. Deliberately a plain pass over
#: the string rather than a JSON parse: the interesting replies are exactly the
#: ones that do not parse (that is why they were rejected), so a scan that only
#: understood well-formed JSON would be blind on the shapes it exists for.
_JSON_UNICODE_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")

#: What the reply-channel function tells the model it is for. Kept beside the
#: envelope it describes so the two cannot drift, and deliberately short: it
#: rides in the request's cache prefix on every call of every episode.
PUBLIC_REPLY_TOOL_DESCRIPTION = (
    "Submit your decision for the current observation. This means exactly the "
    "same thing as replying with the JSON envelope described in your "
    "instructions -- use whichever is natural, and never both."
)


def is_public_reply(value: Any) -> bool:
    # Reserve every envelope key: a misspelled/missing version must not silently
    # downgrade a reply with notes to the legacy actions-only interpretation.
    return isinstance(value, Mapping) and bool(_ENVELOPE_KEYS.intersection(value))


def looks_like_public_reply(payload: str) -> bool:
    """Whether a REJECTED reply appears to attempt the reserved envelope.

    Used only to withhold unvalidated output from CORRECTIVE HISTORY; acceptance
    is unaffected and stays with the strict decoder. The rejection ARTIFACT is a
    separate boundary with its own escape-aware scan (see
    :func:`rejected_reply_evidence`) -- withholding the reply from the bundle as
    well made the rejection classes unreadable, and the two boundaries have
    different jobs. A literal substring test is bypassed by legal JSON Unicode
    escapes (``public_\\u006fbservations``), and full decoding fails on truncated
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
        # Name the DEFECT, not just the rule. Validation is unchanged -- a
        # partial envelope is still rejected, because silently downgrading one
        # to the legacy interpretation would drop whatever the model meant to
        # put in ``public_observations``. What changes is what the model is
        # told, and that decides whether the correction can land.
        #
        # Reserving every envelope key means touching ONE of them commits the
        # reply to strict decoding, so the common failure is a near-miss: a
        # model emits ``{"action_batch": {...}}`` and gets told the rule it
        # already half-followed, without being told which half it missed.
        # Measured on minimax/minimax-m3, which produces exactly that shape in
        # ~1 of 10 replies: re-prompting with the bare rule recovered 4/10,
        # while naming the keys present and missing recovered 9/10. The
        # difference is the whole gap between an episode that continues and one
        # that spends its retry bound and seals as a model failure -- which is
        # how a paid canary episode died at three calls.
        if isinstance(value, dict):
            present = sorted(_ENVELOPE_KEYS & set(value))
            missing = sorted(_ENVELOPE_KEYS - set(value))
            extra = sorted(set(value) - _ENVELOPE_KEYS)
            parts = []
            if present:
                parts.append("carried " + ", ".join(repr(key) for key in present))
            if missing:
                parts.append("omitted " + ", ".join(repr(key) for key in missing))
            if extra:
                # ``present``/``missing`` are intersections with a fixed set, so
                # they can only ever name the three reserved keys. ``extra`` is
                # arbitrary MODEL-SUPPLIED text and must never be echoed whole:
                # a 50,000-character key produced a 50,000-character retry
                # prompt, which neither ``MAX_REJECTED_REPLY_CHARS`` nor
                # ``_diagnostic``'s cap intercepts, and which re-opens the
                # replay channel the reserved-key suppression below closes.
                #
                # A key is quoted only if it is ENTIRELY safe, and is otherwise
                # counted but not named. Truncating instead was the first
                # attempt and it failed closed in neither direction (review
                # round 3):
                #   - ``repr`` expands escapes AFTER a cut, so 40 characters of
                #     ``\U000e0001`` still rendered ~700, making the bound
                #     depend on the input's alphabet.
                #   - worse, a cut CONVERTS A LOUD FAILURE INTO A SILENT ONE.
                #     ``_assert_redacted`` is substring-based, so a secret
                #     longer than the cut survived as a prefix that no longer
                #     matched the canary: the leak stopped tripping the alarm
                #     that exists to catch it.
                # Quoting whole-or-nothing removes both: nothing is ever
                # reshaped on the way out, so a redaction canary still matches
                # and the rendered length is bounded by the charset itself.
                safe = [key for key in extra if is_quotable_key(key)]
                summary = ", ".join(repr(key) for key in safe[:_MAX_EXTRA_KEYS_SHOWN])
                withheld = len(extra) - len(safe[:_MAX_EXTRA_KEYS_SHOWN])
                if withheld and summary:
                    summary += f" and {withheld} more"
                elif withheld:
                    # Every key was unsafe or over the cap: report only how many
                    # to drop. The count alone is enough to act on, and is the
                    # part that carries no model-supplied text at all.
                    summary = f"{withheld} not shown"
                parts.append(f"added {len(extra)} unexpected key(s): {summary}")
            raise ValueError(
                "model reply used the reserved envelope but "
                + "; ".join(parts)
                + '. Reply with EITHER the plain batch {"actions": [...]} and no other '
                "top-level keys, OR the full envelope with exactly reply_version, "
                "action_batch, public_observations"
            )
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


def rejected_reply_evidence(reply: str | None, redactions: RedactionSet | None) -> str:
    """A rejected reply as evidence may publish it: scanned, then bounded.

    The model's own words about a refused turn are the one thing that makes a
    rejection class diagnosable after the fact, and they were the thing the
    bundle threw away: 273 of the MiniMax campaign's 280 rejection artifacts
    replaced the reply with the placeholder (counted 2026-09-12). The reply is
    therefore published here, on a boundary of its own.

    ORDER IS THE SECURITY PROPERTY, exactly as in ``_diagnostic``: every
    rendering is scanned BEFORE the bound is applied. ``RedactionSet`` is a
    substring check, so a reply cut first and scanned afterwards returns clean
    over a canary that was severed by the cut -- and ``publish_artifact``'s own
    scan then agrees for the same reason, sealing the fragment into the bundle.

    Fails closed and WHOLE: any hit replaces the entire reply with
    :data:`REJECTED_REPLY_WITHHELD` rather than masking the matching span. A
    partial masking still narrows a secret and, worse, leaves a rendering whose
    length depends on the secret's own alphabet. The reply was refused anyway --
    nothing executable is lost -- and the artifact still carries the class key
    and the diagnostic, so a withheld reply is still a readable rejection.

    ``redactions`` is required-not-defaulted at the call site for the reason
    ``_diagnostic`` gives: the UNSAFE call must not be the shorter one to write.
    An explicit ``None`` states that this rendering never reaches evidence.
    """

    if not reply:
        return ""
    if redactions is not None and not _reply_is_clear(reply, redactions):
        return REJECTED_REPLY_WITHHELD
    if len(reply) > MAX_REJECTED_REPLY_CHARS:
        return reply[:MAX_REJECTED_REPLY_CHARS] + REJECTED_REPLY_TRUNCATED
    return reply


def _reply_is_clear(reply: str, redactions: RedactionSet) -> bool:
    """Whether every rendering of a reply is free of the episode's canaries.

    Four renderings, cheapest first, because the canary can be hidden in each
    of them and the reply is untrusted bytes:

    * the raw text (the plaintext, base64 and hex canaries ``assert_clear``
      checks directly);
    * percent escapes, decoded;
    * ``\\uXXXX`` escapes, decoded -- the F1 shape, which is a legal JSON
      spelling of any character and would otherwise walk straight past a
      substring test;
    * the DECODED JSON value when the reply happens to parse, which hands
      ``assert_clear`` the structure rather than the text (it walks mappings
      and lists, so a canary nested in an action is seen).

    A decoding failure is not a failure to scan: the raw and escaped
    renderings are always checked, which is what makes this safe on the
    truncated replies that are the common case here.
    """

    unquoted = unquote(reply)
    unescaped = _JSON_UNICODE_ESCAPE.sub(lambda match: chr(int(match.group(1), 16)), reply)
    renderings: list[Any] = [reply, unquoted, unescaped, unquote(unescaped)]
    try:
        renderings.append(json.loads(reply))
    except (ValueError, RecursionError):
        # Truncated or malformed is the EXPECTED case for a rejected reply, and
        # the raw/escaped renderings above still cover it.
        pass
    for rendering in renderings:
        try:
            redactions.assert_clear(rendering)
        except ValueError:
            return False
    return True


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


#: Every action kind the protocol defines, used when no surface is supplied.
#: The published contract describes the protocol, not one adapter's negotiated
#: subset, so it advertises them all.
_ALL_ACTION_MODELS = get_args(get_args(ComputerAction)[0])


def public_reply_schema(action_surface: ActionSurface | None = None) -> dict[str, Any]:
    """The envelope's JSON Schema, shared by the contract and the reply channel.

    One definition with two readers. It is published in the evidence bundle's
    reply contract (below) AND handed to the harness's structured reply channel
    as that function's parameters, so the tool the model may call and the prose
    envelope it may write are provably the same shape rather than two hand-kept
    copies that drift apart.

    ``action_surface`` filters the admitted action kinds to the ones the
    NEGOTIATED surface actually accepts. Passing it is what keeps the offered
    schema honest: the prose prompt is surface-aware, so an unfiltered schema
    would advertise ``paste_text`` — or ``ask_user`` on an adapter without it —
    and the model following the schema would then be rejected by
    ``validate_batch``. Omitting it keeps the full batch, which is what the
    published contract wants.

    Derived from the models on every call rather than cached: they are frozen
    metadata, so the result is deterministic — which is what lets it ride in the
    prompt-cache prefix unchanged across an episode's turns, the surface being
    fixed for the episode — and recomputing keeps a new action kind from
    drifting out.
    """
    models = action_surface.models if action_surface is not None else _ALL_ACTION_MODELS
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_ENVELOPE_KEYS),
        "properties": {
            "reply_version": {"const": REPLY_VERSION},
            "action_batch": {
                "type": "object",
                "additionalProperties": False,
                "required": ["actions"],
                "properties": {
                    "actions": {
                        "type": "array",
                        # ``anyOf`` over INLINED member schemas, not a ``$ref``
                        # into ``$defs`` with a ``oneOf``/``discriminator``.
                        #
                        # This schema is handed to a provider as a function's
                        # parameters, and Gemini's ``FunctionDeclaration``
                        # accepts only a narrow OpenAPI 3.0 subset: ``$ref``,
                        # ``$defs``, ``oneOf`` and ``discriminator`` are all
                        # rejected with 400 INVALID_ARGUMENT, which would fail
                        # EVERY request of a Gemini-routed episode rather than
                        # degrading. ``tools/builtin.py`` carries the same
                        # warning for the same reason. The discriminated union
                        # is therefore flattened here: ``anyOf`` over concrete
                        # member schemas is the same admitted set, expressed in
                        # a shape every provider accepts.
                        "items": {"anyOf": [_inlined_action_schema(m) for m in models]},
                    }
                },
            },
            "public_observations": {"type": "string", "maxLength": MAX_PUBLIC_OBSERVATIONS_CHARS},
        },
    }


def _inlined_action_schema(model: Any) -> dict[str, Any]:
    """One action model's schema with its ``$defs`` resolved into place.

    Pydantic emits ``$ref``/``$defs`` for nested models. A provider that rejects
    those constructs cannot read the result, so the references are substituted
    for the definitions they name and the ``$defs`` block is dropped.
    """

    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    # ``kind`` carries the discriminator, and pydantic does not mark it
    # ``required`` because each model defaults it. That is safe under a
    # discriminated union, where the tag selects the member before its fields
    # are checked; it is NOT safe under the bare ``anyOf`` this flattening
    # produces, because a batch omitting ``kind`` then satisfies whichever
    # member happens to match on its other fields. The old schema denied such
    # a batch and pydantic still rejects it, so leaving it out would admit at
    # the schema what the validator refuses -- inviting the model into exactly
    # the rejection this contract exists to prevent. Requiring the tag keeps
    # the flattened set identical to the discriminated one.
    required = schema.get("required")
    if "kind" in schema.get("properties", {}) and isinstance(required, list):
        if "kind" not in required:
            schema["required"] = sorted([*required, "kind"])

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = defs.get(ref.rsplit("/", 1)[-1], {})
                merged = {k: v for k, v in node.items() if k != "$ref"}
                return {**resolve(target), **merged}
            return {key: resolve(value) for key, value in node.items()}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)


def public_reply_contract() -> dict[str, Any]:
    """Publish a reproducible reply identity separately from the tool surface.

    The action array schema is borrowed, not copied: its vocabulary and bounds
    remain owned by ActionBatch. Negotiated execution restrictions are declared
    by the existing action_surface metadata/tool digest and still gate parsing.
    """
    schema = public_reply_schema()
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
