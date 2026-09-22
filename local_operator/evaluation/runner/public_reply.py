"""Public model output, not an extension of the adapter action protocol.

Screenshots leave the shared context before text summarization. Concise facts
from the model's *visible reply* therefore need the same ordinary assistant
text path as interactive sessions; private reasoning is never an input here.

One reply, one shape, however it was framed. The envelope a reply is validated
against is ``{"actions": [...], "public_observations": ""}``; the
``action_batch`` object around the array, a ``reply_version``, a generic
tool-call wrapper, trailing text, and an ``actions`` value the model sent as a
JSON-encoded STRING rather than as the array are all FRAMING, and framing is
normalised here rather than refused. Nothing in this module reads a model, a
provider or a benchmark: the accepted set is a statement about our own contract
and about the serializations any harness uses to wrap a function call.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Mapping, Sequence, get_args
from urllib.parse import unquote

from local_operator.evaluation.action_surface import ActionSurface
from local_operator.evaluation.evidence.models import canonical_bytes, canonical_digest
from local_operator.evaluation.protocol import ComputerAction
from local_operator.evaluation.receipts import RedactionSet

# stdlib logging rather than ``local_operator.logger``: that module imports
# ``local_operator.paths``, which resolves the operator's config directory, and
# the runner's import graph is held to the isolation rule in
# ``tests/unit/evaluation/runner/test_isolation.py``. The logger OBJECT is the
# same either way — ``get_logger`` is ``logging.getLogger`` — so records still
# reach whatever handlers the entry point configured.
logger = logging.getLogger(__name__)

MAX_PUBLIC_OBSERVATIONS_CHARS = 2_000

#: The keys that mark a reply as the PUBLIC-OBSERVATION envelope rather than a
#: bare action batch. Reserved for exactly one job now: a REJECTED reply
#: carrying any of them is withheld from corrective history instead of being
#: replayed as facts, because its notes are unvalidated text (see
#: :func:`looks_like_public_reply`).
#:
#: ``reply_version`` is tolerated rather than required, and is deliberately left
#: in this set: a reply that carries it is envelope-shaped, and withholding such
#: a reply is the conservative direction. It is no longer part of the accepted
#: contract -- it carried nothing the channel and the schema did not already
#: pin, and requiring the model to restate it was a refusal class of its own.
_ENVELOPE_KEYS = {"reply_version", "action_batch", "public_observations"}

#: The keys a bare or enveloped reply may carry without being framing noise.
#: Everything else in a reply is IGNORED rather than refused: task, episode and
#: observation ids are pinned by the harness, so no sibling key can change what
#: executes -- but extras are still reported, never silently dropped.
_REPLY_KEYS = {"actions", "action_batch", "public_observations", "reply_version"}

#: The rule ``action_batch`` is held to, worded as it has always been worded.
#: Kept as one literal because ``classify_rejection`` reads this sentence out of
#: SEALED artifacts too, where it is the only surviving description of the
#: defect -- a reworded opening would silently reclassify them.
_BATCH_SHAPE_RULE = "model reply action_batch requires exactly an actions array"

#: The same rule with the accepted shape stated, for the one branch that still
#: raises it: a reply whose ``action_batch`` carries no actions array at all.
#: The rule alone names nothing a model can act on -- the doctrine on
#: ``rejection_hint`` is that every refusal states the accepted shape, literal or
#: bound, and this sentence is the whole repair turn for a reply that put its
#: actions somewhere else. The classifier keys on the rule, which is the prefix.
_BATCH_SHAPE_ACCEPTED = _BATCH_SHAPE_RULE + ' -- the batch is exactly {"actions": [...]}'

#: The sentence ``reply_version`` used to be refused with when it was nested
#: inside ``action_batch`` or duplicated at both levels. This build never emits
#: it -- a key in the wrong place is tolerated now, so there is no defect left to
#: report -- but ``classify_rejection`` still keys ``env-version-misplaced`` on
#: this exact literal, and every artifact that class was ever measured from
#: carries it. Deleting or rewording it would silently reclassify all of them.
_MISPLACED_REPLY_VERSION = (
    "'reply_version' belongs at the top level of the envelope, beside "
    "'action_batch' and 'public_observations', not inside 'action_batch'"
)

#: One generic function-call serialization layer: a call is a NAME plus an
#: argument object, and the reply already arrives on a channel that names the
#: contract. These are the keys any harness puts those arguments under, so
#: unwrapping one of them normalises WHERE the envelope is framed, never what it
#: says. A per-model or per-vendor marker table would be a different thing
#: entirely, and is refused in writing by ``harness/reply_channel.py``.
_TOOL_CALL_ARGUMENT_KEYS = (
    "parameters",
    "arguments",
    "input",
    "payload",
    "action",
    "data",
    "tool_input",
)

#: How many serialization layers may be unwrapped before giving up. One covers
#: every shape observed in the field -- ``{"tool_name": ..., "parameters":
#: {...}}``, ``{"tool_call": ..., "input": "{...}"}``, ``{"input": {...}}``
#: -- and two leaves room for a wrapper inside a wrapper without turning the
#: decoder into a search over arbitrary nesting.
_MAX_UNWRAP_DEPTH = 2

#: How many candidate ``{`` positions the trailing-remainder scan may try before
#: giving up. Each failed decode rescans forward one character, so an unbounded
#: scan over a remainder full of bare braces goes quadratic -- the same bound,
#: and the same reason, as ``_iter_json_objects`` in the tool layer. Giving up
#: means "no competing batch found", which degrades to the tolerant path rather
#: than to an error.
_MAX_TRAILING_DECODE_ATTEMPTS = 256


#: Bounds on how much of a rejected reply's OWN key names may appear in a
#: diagnostic or a log line. The reserved keys are safe to name (they come from
#: a fixed set), but any other key is model-supplied text: echoing it whole turns
#: a malformed reply into an unbounded retry prompt, and now turns a bounded log
#: line into an unbounded one, re-opening the replay channel the reserved-key
#: suppression exists to close.
#:
#: A key is quoted WHOLE or not at all -- :func:`_unexpected_key_summary`
#: owns both the rendering and the guard that decides what may be quoted.
_MAX_EXTRA_KEY_CHARS = 40
_MAX_EXTRA_KEYS_SHOWN = 5


def _unexpected_key_summary(keys: Sequence[str]) -> str:
    """The bounded, quotation-guarded rendering of unexpected key names.

    Truncation was the first attempt and it failed closed in neither direction
    (review round 3): ``repr`` expands escapes after the cut, so the rendered
    length depended on the input's alphabet; and, far worse, cutting a secret
    that appeared as a key left a prefix that ``_assert_redacted``'s substring
    check no longer matched, converting a loud redaction failure into a silent
    leak. Quoting whole-or-nothing means nothing is ever reshaped on the way
    out.

    Shared by every branch that names model-supplied keys back to the reader --
    a tolerated-but-ignored extra key, or a near-miss key in a hint -- because
    the bound and the guard are the security property here, and a second copy
    of them is a second place to get it wrong.
    """

    safe = [key for key in keys if is_quotable_key(key)]
    shown = safe[:_MAX_EXTRA_KEYS_SHOWN]
    summary = ", ".join(repr(key) for key in shown)
    withheld = len(keys) - len(shown)
    if withheld and summary:
        summary += f" and {withheld} more"
    elif withheld:
        # Every key was unsafe or over the cap: report only how many to drop.
        # The count alone is enough to act on, and is the part that carries no
        # model-supplied text at all.
        summary = f"{withheld} not shown"
    return summary


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
#:
#: Two exceptions, both deliberate. An ENVELOPE reply's history rendering is not
#: a slice of the reply at all but :data:`REJECTED_PUBLIC_REPLY` (see
#: ``provider_client.EpisodeModelClient``): unvalidated notes are not factual
#: memory, so the model is corrected from the hint rather than from its own
#: text, and this bound reaches that path only through the legacy batch. And the
#: artifact's copy is bounded at its own boundary -- ``evidence_reply`` travels
#: raw so that :func:`rejected_reply_evidence` can scan the WHOLE reply before
#: cutting it, because a reply cut first and scanned afterwards returns clean
#: over a canary the cut severed.
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

#: An escaped backslash, i.e. one level of JSON escaping applied to a backslash
#: that is itself the start of an escape. Removing it is what makes escapes
#: COMPOSE: a canary that was escaped and then carried inside a JSON string
#: arrives as ``\\\\u0066``, and only a scan that drops a backslash level (and
#: then decodes the escape underneath) sees the canary. Lookahead-bound so a
#: trailing ``\\`` at the end of a truncated reply is left alone.
_DOUBLE_BACKSLASH = re.compile(r"\\\\(?=.)")

#: How many levels :func:`_escape_chain` decodes before giving up. One level is
#: the F1 shape; two cover an escaped canary inside a JSON string; three is one
#: more than any observed reply needs. Bounded because every level costs a pass
#: over the reply, and a reply made of backslashes would otherwise make the scan
#: quadratic.
_MAX_ESCAPE_LEVELS = 3

#: What the reply-channel function tells the model it is for. Kept beside the
#: envelope it describes so the two cannot drift, and deliberately short: it
#: rides in the request's cache prefix on every call of every episode.
PUBLIC_REPLY_TOOL_DESCRIPTION = (
    "Submit your decision for the current observation. This means exactly the "
    "same thing as replying with the JSON object described in your "
    "instructions -- use whichever is natural, and never both."
)


def is_public_reply(value: Any) -> bool:
    """Whether a PARSED reply carries one of the reserved envelope keys.

    Two jobs, both about PUBLIC OBSERVATIONS rather than about acceptance. It
    commits a REJECTED reply to the withheld-from-history rule (the notes it
    wanted to publish are unvalidated text), and it is how a competing-batch
    scan finds the batch inside an ``action_batch`` wrapper. Acceptance itself no
    longer turns on this: :func:`normalise_public_reply` decodes every accepted
    spelling, so a reply is never refused for missing a key it may omit.
    """

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


class DecisionParseError(ValueError):
    """The provider returned something that is not a usable action batch.

    Declared here, beside the decoders that raise it, because this module owns
    what counts as a reply; ``provider_client`` re-exports the name for callers
    that treat it as a client error.
    """


#: Distinguishes "the model wrote no ``public_observations`` key" from "the
#: model wrote an empty one". The first is a legacy actions-only batch and has
#: nothing to publish; the second is an envelope that recorded an empty note and
#: is replayed as the model's visible reply.
_ABSENT = object()


def _decode_leading_json(payload: str) -> tuple[Any, str]:
    """Decode the leading JSON value and return it with any trailing noise.

    ``json.loads`` demands that the WHOLE string be one value, so a model that
    emitted a complete, correct batch and then appended a stray token lost the
    entire turn to ``Extra data: line 1 column 318 (char 317)``. That is a real
    and repeated failure -- one sealed episode paid for it three times, each a
    billed call discarded over noise the harness had already finished reading
    past. ``raw_decode`` stops at the end of the first complete value and says
    where it stopped, which is exactly the question being asked here.

    The tolerance is deliberately ONE-SIDED, because the three shapes are not
    equally knowable:

    * **Trailing noise** (``{...}原始内容``, ``{...} Hope that helps!``) is
      tolerated. The decision is already complete and unambiguous at the point
      the junk starts; nothing after it can change which actions were chosen.
    * **Leading noise** (``Sure, here you go: {...}``) is NOT skipped. Hunting
      forward for the first ``{`` means guessing where the value begins, and a
      preamble that itself contains a brace makes that guess wrong silently --
      the failure mode is executing a DIFFERENT batch than the model sent,
      which is far worse than losing the turn. A leading-junk reply still gets
      the ordinary parse error and a corrective re-prompt.
    * **A second batch for the SAME observation**, anywhere in the remainder,
      is genuinely ambiguous -- which one did the model mean? -- so it is NOT
      tolerated. Taking the first would execute a decision the model may have
      superseded.

    What makes the third rule safe to apply ANYWHERE, rather than only to an
    immediately adjacent object, is that it keys on ``observation_id``. Two
    weaker probes were measured against the real bundle and both fail:

    * "Does the remainder start with ``{``?" only catches a directly adjacent
      object. One character of anything else -- a comma, a newline, prose,
      ``原始内容`` -- disables it, so a superseding batch behind a separator is
      dropped silently, which is precisely what this rule claims to prevent.
    * "Does anything in the remainder parse as JSON?" over-fires: it rejects
      all three real bundle turns, because a model that quotes the harness's
      own ``The rejected reply was: {...}`` feedback back at itself carries a
      well-formed batch in its prose. Those batches are HISTORY, not a
      competing decision -- in every one of the three they bind a DIFFERENT
      observation than the turn being decided.

    Binding on the observation id separates those two cases exactly: a batch
    naming this observation is a decision about the screen in front of the
    model and therefore competes; a batch naming any other observation is a
    quotation of an older turn and cannot.

    Shared by BOTH decode paths since the contract collapsed to one shape: the
    envelope decoder used to call ``json.loads`` over the whole string and so
    refused exactly the reply this tolerance exists for, which is an internal
    inconsistency rather than a contract decision.
    """

    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    # ``json.loads`` skips leading whitespace and ``raw_decode`` does not, so
    # stripping here keeps this helper's contract identical to the call it
    # replaced. Doing it inside rather than relying on the caller matters
    # because this is a general entry point: a second caller that forgot to
    # strip would lose a turn to a leading newline, which is exactly the class
    # of loss this function exists to prevent. Only leading WHITESPACE is
    # skipped -- leading junk still fails at offset 0, by design.
    payload = payload.lstrip()
    try:
        decoded, end = decoder.raw_decode(payload)
    except (ValueError, RecursionError) as error:
        # Includes the leading-junk case: raw_decode starts at offset 0, so a
        # preamble fails here rather than being skipped past. Duplicate keys are
        # refused by the hook, and their wording is kept as it has always been
        # worded: ``classify_rejection`` keys the class on the phrase, and a
        # re-spelled sentence would reclassify every sealed artifact that
        # carries it.
        if "duplicate JSON keys" in str(error):
            raise DecisionParseError(
                "model reply must be one duplicate-free JSON object"
            ) from error
        raise DecisionParseError(f"decision is not valid JSON: {error}") from error
    trailing = payload[end:].strip()
    if trailing and _competing_batch_offset(trailing, decoded, decoder) is not None:
        raise DecisionParseError(
            "decision carries a second action batch for the same observation; "
            "send exactly one action batch"
        )
    return decoded, trailing


def _competing_batch_offset(trailing: str, decoded: Any, decoder: json.JSONDecoder) -> int | None:
    """Offset of a second batch in ``trailing`` that competes with ``decoded``.

    "Competes" means it names the SAME ``observation_id``: only a decision
    about the screen currently in front of the model can supersede the one
    already parsed. See :func:`_decode_leading_json` for why that test, rather
    than adjacency or bare JSON-ness, is the one that separates a superseding
    batch from the harness feedback a model quotes back at itself.

    Returns ``None`` when the remainder is ordinary prose, which is the common
    case and the one that must stay cheap.
    """

    observation_ids = _batch_observation_ids(decoded)
    if not observation_ids:
        return None
    # A decision is always an object, so only "{" can start a competing batch;
    # the scan is bounded the same way ``_iter_json_objects`` is bounded, since
    # a remainder full of bare braces would otherwise cost a rescan each.
    index = 0
    attempts = 0
    while attempts < _MAX_TRAILING_DECODE_ATTEMPTS:
        start = trailing.find("{", index)
        if start < 0:
            return None
        attempts += 1
        try:
            candidate, end = decoder.raw_decode(trailing, start)
        except (ValueError, RecursionError):
            # RecursionError as well as ValueError: the C decoder recurses per
            # nesting level and raises it (NOT a ValueError subclass) on a
            # deeply nested payload. Untrusted model output must degrade to
            # "no competing batch found", never to an unexpected exception.
            index = start + 1
            continue
        index = max(end, start + 1)
        if not isinstance(candidate, Mapping):
            continue
        if _batch_observation_ids(candidate) & observation_ids:
            return start
    return None


def _actions_from_json_string(value: Any) -> tuple[Any, bool]:
    """The action array a JSON-encoded ``actions`` STRING evidently carries.

    A model that writes its actions as a string -- ``{"actions": "[{...}]"}``
    -- has stated a complete, executable decision in a spelling this decoder
    used to refuse, and the refusal cost a whole paid call to repair with a
    re-prompt that then repeated the mistake: 9 of one OSWorld episode's 120
    calls were exactly this shape (judge5-20260921-231232, 7.5% of that
    episode's calls), and 30 of the campaign runs' 74 sealed refusals are it
    (counted 2026-09-22 over ``~/worktrees/osworld/runs/*/evidence/ep-*``). It
    is a GENERAL tolerance about a spelling of our own contract, so it belongs
    here, on the boundary every interface's reply crosses -- not at one caller.

    DECODED through :func:`_decode_leading_json`, NOT ``json.loads``, because
    in every one of those 30 the string is more than the array: it is the tail
    of the model's OWN envelope -- the array followed by
    ``, "public_observations": "..."}`` -- because the model double-encoded the
    rest of the object it was writing. Reusing that decoder means the tolerance
    keeps the one-sided rule a reply already gets: leading junk is still
    refused, and a second batch for the same observation inside the string
    still refuses the reply, so this can never execute a decision the model
    superseded.

    Returns ``(value, False)`` unless the leading JSON value is a NON-EMPTY
    ARRAY OF OBJECTS -- which is what an action array is, and the entire claim
    being made here. A string that does not decode, or decodes to a scalar, to
    an array of scalars, or to an empty array, comes back UNCHANGED, so that
    reply keeps the refusal it always had: the tolerance may only accept a
    decision, never widen a refusal into one, and a string that is not an action
    structure must stay the malformation it is.

    The claim stops at the ARRAY on purpose: whether the objects inside it are
    valid actions is the action protocol's question, not the decoder's (this
    module is framing-only -- see its docstring). A string carrying an array
    whose objects are not actions is still refused, by ``parse_decision`` and
    with that class and hint (``unknown-action-kind``, a missing ``kind``, a
    stale binding) rather than with the generic batch-shape sentence, which is
    the same layering the identical objects get when the array is not a string.

    No logging here, deliberately. This runs over candidate objects while
    scanning untrusted trailing text (see :func:`_batch_observation_ids`) as well
    as over the reply's own decision, and a record that means "a reply was
    accepted this way" must not fire for text that was merely looked at. The
    caller that accepts the reply records it.

    The note inside that tail is NOT recovered on purpose: it is a JSON
    FRAGMENT, so locating where the model's object began is the same guess
    :func:`_decode_leading_json` refuses to make for leading noise. The decision
    is what must not cost the turn; the note is memory, and a batch carrying no
    note is the ordinary legacy shape this module already accepts.
    """

    if not isinstance(value, str):
        return value, False
    try:
        decoded, trailing = _decode_leading_json(value)
    except DecisionParseError:
        return value, False
    if not isinstance(decoded, list) or not decoded:
        return value, False
    if not all(isinstance(action, Mapping) for action in decoded):
        return value, False
    if trailing and (
        _competing_batch_offset(
            trailing,
            # A batch-shaped VIEW of what was decoded, so the scan reads the
            # observation ids the way it reads them off any other batch. Handed
            # the bare array it would find no id and stand down, which is the
            # one way this tolerance could execute a decision the model
            # superseded -- the rule this module refuses to trade away. The
            # decoder is built with the same hook as ``_decode_leading_json``'s
            # so a candidate carrying duplicate keys is refused identically.
            {"actions": decoded},
            json.JSONDecoder(object_pairs_hook=_unique_object),
        )
        is not None
    ):
        return value, False
    return decoded, True


def _batch_observation_ids(value: Any) -> set[str]:
    """The observation ids an action-batch-shaped object binds to.

    Read from the ACTIONS rather than from a top-level ``observation_id``: a
    model reply carries the id per action (the runner supplies the batch-level
    one itself), so a top-level lookup finds nothing on the very shape this
    needs to compare. Returns an empty set for anything that is not batch
    shaped, which the caller treats as "not a competing decision".

    The batch is located through the same normalisation an accepted reply gets,
    so a WRAPPED second batch competes exactly as a bare one does -- otherwise
    normalising the framing would have quietly disabled this rule for the very
    shapes the normalisation admits.
    """

    try:
        value = _unwrap_tool_call(value)
    except DecisionParseError:
        # A wrapper whose payload is not JSON at all cannot be a batch this
        # scan is looking for. Degrading here is required: this runs over
        # untrusted trailing text on the decode hot path.
        return set()
    if not isinstance(value, Mapping):
        return set()
    actions = value.get("actions")
    if not isinstance(actions, list):
        nested = value.get("action_batch")
        actions = nested.get("actions") if isinstance(nested, Mapping) else nested
    # The same coercion the decision itself gets, for the reason the paragraph
    # above gives: a batch the model wrote as a JSON-encoded string competes
    # exactly as a bare one does, and accepting that spelling without reading
    # its ids here would have quietly disabled this rule -- a competing decision
    # inside the string's tail, or in text after the string, would be taken as
    # ordinary noise while the earlier batch executed.
    actions, _coerced = _actions_from_json_string(actions)
    if not isinstance(actions, list) or not actions:
        return set()
    return {
        action["observation_id"]
        for action in actions
        if isinstance(action, Mapping) and isinstance(action.get("observation_id"), str)
    }


def _carries_decision(value: Mapping[str, Any]) -> bool:
    """Whether a parsed object already carries THIS reply's decision.

    The guard that keeps unwrapping from ever running past a reply's own
    envelope: a decision that happens to mention a wrapper key beside its
    actions stays the decision it is, because nothing is ever unwrapped out of
    an object that has already stated one.
    """

    return "actions" in value or "action_batch" in value


def _unwrap_tool_call(value: Any) -> Any:
    """Strip up to :data:`_MAX_UNWRAP_DEPTH` generic function-call layers.

    A tool call is a name plus an argument object, so a harness that carries a
    call as TEXT carries one of :data:`_TOOL_CALL_ARGUMENT_KEYS` around the
    envelope. The reply already arrives on a channel that names the contract, so
    the wrapper holds no decision information at all -- unwrapping it normalises
    where the envelope is framed and leaves what it says untouched.

    A layer whose value is a STRING is parsed as JSON (that is the
    ``input``-as-a-JSON-string shape) through :func:`_decode_leading_json`, not
    through a second parser, so the tolerance and the competing-batch rule a
    wrapped reply gets are the ones a prose reply gets.

    Bounded twice over: by the depth, and by refusing to look inside an object
    that already carries a decision (see :func:`_carries_decision`). Anything
    still wrapped after the bound is left alone to be refused as the malformed
    reply it then is -- this is a normaliser, not a search.
    """

    for _ in range(_MAX_UNWRAP_DEPTH):
        if not isinstance(value, Mapping) or _carries_decision(value):
            return value
        candidates = [key for key in _TOOL_CALL_ARGUMENT_KEYS if key in value]
        if len(candidates) != 1:
            return value
        key = candidates[0]
        inner = value[key]
        if isinstance(inner, str):
            inner, _trailing = _decode_leading_json(inner)
        value = inner
    return value


def _public_note(framed: Mapping[str, Any], batch: Any) -> str | None:
    """The model's public note, from either level, or ``None`` if it wrote none.

    The note is the one part of a reply the harness carries VERBATIM into the
    next turn's context, so it is read from wherever the model put it rather
    than dropped for landing beside the actions instead of above them.

    Validation itself is unchanged: a note that is not a string, is over
    :data:`MAX_PUBLIC_OBSERVATIONS_CHARS`, or is not encodable as Unicode text
    still refuses the whole reply. A note that cannot be read must not be
    silently discarded, because that would publish a reply the model did not
    send.
    """

    note = framed.get("public_observations", _ABSENT)
    if note is _ABSENT and isinstance(batch, Mapping):
        note = batch.get("public_observations", _ABSENT)
    if note is _ABSENT:
        return None
    if not isinstance(note, str) or len(note) > MAX_PUBLIC_OBSERVATIONS_CHARS:
        raise DecisionParseError(
            "public_observations must be a string of at most "
            f"{MAX_PUBLIC_OBSERVATIONS_CHARS} characters"
        )
    try:
        note.encode("utf-8")
    except UnicodeEncodeError as error:
        raise DecisionParseError("public_observations must be valid Unicode text") from error
    return note


def _report_ignored_keys(framed: Mapping[str, Any], batch: Any) -> None:
    """Report -- never refuse -- keys the reply contract has no use for.

    Tolerated is not the same as silent. An unexpected key is a signal worth
    seeing (it usually means the model is guessing at a shape it was not given),
    and a tolerance nobody can observe is indistinguishable from the harness
    quietly mangling a reply.

    Every name travels through :func:`_unexpected_key_summary`, so a
    50,000-character model-supplied key cannot turn a bounded log line into an
    unbounded one -- the same guard, and the same reason, that used to bound the
    diagnostic this class of reply was refused with.
    """

    stray = sorted(set(framed) - _REPLY_KEYS)
    if isinstance(batch, Mapping):
        stray += sorted(set(batch) - {"actions", "public_observations"})
    if stray:
        logger.warning(
            "model reply carried %d key(s) the reply contract does not use; ignored: %s",
            len(stray),
            _unexpected_key_summary(stray),
        )


def normalise_public_reply(value: Any) -> tuple[list[Any], str | None]:
    """The one accepted reply shape, however the reply was framed.

    Returns the action array and the model's public note -- ``None`` for the
    note when the reply carried no ``public_observations`` key at all, which is
    what still separates a legacy actions-only batch (nothing to publish) from
    an envelope that recorded an empty note.

    What is ACCEPTED here is deliberately framing-blind: a bare action array, a
    bare array under ``action_batch``, the full envelope, any of those inside
    one generic tool-call wrapper, any of those with a ``reply_version`` or with
    extra keys beside them, and an ``actions`` value that is a JSON-encoded
    STRING carrying the array (see :func:`_actions_from_json_string`). What is
    still REFUSED is what cannot be read as a decision: malformed or duplicated
    JSON, two action arrays that could each be the decision, a batch that is not
    an object carrying ``actions``, and a string that does not decode to a
    non-empty array of action objects. Those are the refusals that are doing
    real work, and they are the only ones left in this module.
    """

    framed = _unwrap_tool_call(value)
    if framed is not value:
        logger.warning(
            "model reply was wrapped in a generic tool-call serialization; the "
            "envelope inside it was decoded"
        )
    if not isinstance(framed, Mapping):
        raise DecisionParseError("decision must be a JSON object")
    batch = framed.get("action_batch")
    nested_actions, nested_from_string = _actions_from_json_string(
        batch.get("actions") if isinstance(batch, Mapping) else batch
    )
    top_actions, top_from_string = _actions_from_json_string(framed.get("actions"))
    if isinstance(top_actions, list) and isinstance(nested_actions, list):
        # Two action arrays in one reply is the SAME ambiguity as two batches in
        # one payload -- which one did the model mean? -- and it is a question
        # about MEANING, not about framing, so it is not tolerated. Taking the
        # outer array could drop an action the model meant, and taking the inner
        # one could execute a decision the model superseded.
        raise DecisionParseError(
            "model reply carries a second action batch; send exactly one action batch"
        )
    actions = top_actions if isinstance(top_actions, list) else nested_actions
    if not isinstance(actions, list) or not actions:
        if batch is not None:
            raise DecisionParseError(_BATCH_SHAPE_ACCEPTED)
        raise DecisionParseError("decision must carry a non-empty actions array")
    note = _public_note(framed, batch)
    _report_ignored_keys(framed, batch)
    if top_from_string or nested_from_string:
        # Tolerated, never silent -- the same rule the tool-call wrapper and the
        # trailing-text tolerance state above, and it is the one record a
        # campaign can count this against (a tolerance nobody can observe is
        # indistinguishable from the harness quietly mangling a reply). Emitted
        # only once the reply IS accepted, so the line means what it says; the
        # count and nothing else, because the string's tail is model text and
        # this module never renders that into a log or an artifact unscanned.
        logger.warning(
            "model reply carried its actions as a JSON-encoded string; the leading "
            "JSON value was decoded and accepted as the decision it states "
            "(%d action(s))",
            len(actions),
        )
    return actions, note


def decode_public_reply(payload: str) -> dict[str, Any]:
    """The canonical envelope of a reply payload, whatever its framing.

    The string entry point, for callers holding the model's own bytes. The reply
    may be a bare action array, an ``action_batch`` wrapper, the full envelope,
    any of those inside one generic tool-call serialization, followed by text.

    Returns ``{"actions": [...], "public_observations": str}`` so no caller has
    to know which accepted spelling the model used. Duplicate JSON keys are
    still refused, and so is a second batch for the same observation: one reply
    is one decision.
    """

    value, _trailing = _decode_leading_json(payload)
    actions, note = normalise_public_reply(value)
    return {"actions": actions, "public_observations": note or ""}


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

    Fails closed: the answer is a yes only after every rendering below has been
    scanned, and a reply that cannot be DECODED is not a failure to scan -- the
    text renderings are always checked, which is what makes this safe on the
    truncated replies that are the common case here.
    """

    for rendering in _reply_renderings(reply):
        try:
            redactions.assert_clear(rendering)
        except ValueError:
            return False
    return True


def _reply_renderings(reply: str) -> list[Any]:
    """Every rendering of a rejected reply that a canary could be hiding in.

    The reply reaches evidence as WIRE bytes and has already been refused, so it
    can be anything: malformed, truncated, a well-formed envelope whose notes
    were never validated, or a reply that is itself JSON carrying JSON. Each of
    those hides a canary behind a different decoding:

    * the raw text, where ``assert_clear`` catches its own plaintext, base64,
      percent and hex variants;
    * the text under one or more levels of decoding -- percent escapes, JSON
      ``\\uXXXX`` escapes, and the removal of an escaped backslash. The escape
      case is the F1 shape: a legal JSON spelling of any character, invisible to
      a substring test. Successive levels are needed because escaping composes --
      an escaped string carried as a JSON string arrives with ``\\\\u0066``,
      where the canary only appears once a backslash level AND the escape have
      both been decoded;
    * the DECODED JSON value when the reply parses, which hands ``assert_clear``
      the structure rather than the text (it walks mappings and lists, so a
      canary nested in an action is seen), and the same decodings applied to
      every string INSIDE that value, which is where an inner JSON document's
      own escapes live.

    Decoding stops at :data:`_MAX_ESCAPE_LEVELS`. Nesting deeper than that is
    NOT covered, and this is the sole barrier on the path that publishes raw
    model output, so the limit is stated rather than implied: a canary escaped
    more than three times over -- a shape no observed reply has produced, and
    one that would have to be constructed deliberately -- would be published.
    """

    renderings: list[Any] = []
    texts = [reply]
    try:
        value = json.loads(reply)
    except (ValueError, RecursionError):
        # Truncated or malformed is the EXPECTED case for a rejected reply; the
        # text renderings below cover it.
        value = None
    if value is not None:
        renderings.append(value)
        texts.extend(_string_leaves(value))
    for text in texts:
        renderings.extend(_escape_chain(text))
    return renderings


def _escape_chain(text: str, *, levels: int = _MAX_ESCAPE_LEVELS) -> list[str]:
    """``text`` and every decoding of it up to ``levels`` levels deep.

    Breadth-first over three decoders per level, so combinations are covered
    rather than one branch of them: a reply can be percent-escaped inside a
    JSON escape, and each decoder composes with the others. Bounded because
    each level costs a full pass and a reply built from backslashes would make
    an unbounded walk quadratic; the frontier also shrinks naturally, as every
    decoder only ever removes characters or leaves the text alone.
    """

    chain = [text]
    seen = {text}
    frontier = [text]
    for _ in range(levels):
        next_frontier: list[str] = []
        for current in frontier:
            for decode in _ESCAPE_DECODERS:
                decoded = decode(current)
                if decoded == current or decoded in seen:
                    continue
                seen.add(decoded)
                chain.append(decoded)
                next_frontier.append(decoded)
        if not next_frontier:
            break
        frontier = next_frontier
    return chain


def _decode_unicode_escape(match: re.Match[str]) -> str:
    return chr(int(match.group(1), 16))


def _collapse_backslash_escape(match: re.Match[str]) -> str:
    return match.group(0)[1:]


def _string_leaves(value: Any) -> list[str]:
    """Every string a decoded JSON value carries, keys included."""

    leaves: list[str] = []
    if isinstance(value, str):
        leaves.append(value)
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            leaves.append(str(key))
            leaves.extend(_string_leaves(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            leaves.extend(_string_leaves(nested))
    return leaves


#: One decode of one level of escaping. Applied breadth-first by
#: :func:`_escape_chain`, in this order, because the three compose: a payload can
#: percent-escape inside a JSON escape, and an escaped string carried as a JSON
#: string needs a backslash level removed before its escapes mean anything.
_ESCAPE_DECODERS = (
    lambda text: _JSON_UNICODE_ESCAPE.sub(_decode_unicode_escape, text),
    unquote,
    lambda text: _DOUBLE_BACKSLASH.sub(_collapse_backslash_escape, text),
)


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
    """The reply contract's JSON Schema, shared by the contract and the channel.

    One definition with two readers. It is published in the evidence bundle's
    reply contract (below) AND handed to the harness's structured reply channel
    as that function's parameters, so the tool the model may call and the prose
    envelope it may write are provably the same shape rather than two hand-kept
    copies that drift apart.

    ONE shape, and the simplest one the decoder accepts: a required ``actions``
    array with an optional sibling note. The ``action_batch`` wrapper, a
    ``reply_version`` and trailing text are all still ACCEPTED -- the decoder is
    deliberately more permissive than the offer, because a reply is judged on
    its actions and framing it differently is not a decision about the task. The
    offer stays the crisp shape so a model that follows it cannot produce a
    reply we then have to normalise.

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

    The framing and binding entries describe what the DECODER accepts, which is
    deliberately wider than the schema it publishes: a reader of a bundle should
    be able to tell, without reading this module, why a reply with extra keys or
    a stale ``observation_id`` was accepted rather than refused.
    """
    schema = public_reply_schema()
    contract = {
        "schema": schema,
        "accepted_framings": (
            "one leading JSON object, and any trailing text after it; duplicate keys and a "
            "second action batch for the same observation are refused"
        ),
        "accepted_shapes": (
            '{"actions": [...]}, or the same array under one "action_batch" wrapper, or '
            "either of those inside one generic tool-call serialization; keys the reply "
            'contract does not use, including a "reply_version", are ignored'
        ),
        "binding": (
            "every action is bound to the current observation and its frames by the "
            "harness before validation, so an observation_id or frame_id in the reply "
            "selects nothing; the negotiated action_surface still gates every action"
        ),
        "public_observations": (
            "concise new observed facts/progress only; no deliberation or credentials"
        ),
    }
    return {
        "model_reply_contract": canonical_bytes(contract).decode("utf-8"),
        "model_reply_contract_digest": canonical_digest("runner-model-reply-v1", contract),
    }
