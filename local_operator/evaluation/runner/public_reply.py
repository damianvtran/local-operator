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
from enum import Enum
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
LEGACY_ACTION_BINDING = "legacy"
COMPACT_ACTION_BINDING = "compact"
_ACTION_BINDING_MODES = frozenset({LEGACY_ACTION_BINDING, COMPACT_ACTION_BINDING})
_BINDING_KEY = "observation_id"


def _validate_action_binding(action_binding: str) -> str:
    if action_binding not in _ACTION_BINDING_MODES:
        raise ValueError("action_binding must be 'legacy' or 'compact'")
    return action_binding


def _compact_action_schema(schema: dict[str, Any], models: tuple[Any, ...]) -> dict[str, Any]:
    """Remove repeated action IDs and require one batch-level binding instead."""
    members = schema["properties"]["actions"]["items"]["anyOf"]
    # The top-level field borrows the exact field schema from the same action
    # models; maintaining a second regex or bound would let provider validation
    # and the canonical protocol accept different identifiers.
    binding_schema = _inlined_action_schema(models[0])["properties"][_BINDING_KEY]
    for member in members:
        properties = member.get("properties")
        if isinstance(properties, dict):
            properties.pop(_BINDING_KEY, None)
        required = member.get("required")
        if isinstance(required, list):
            member["required"] = [name for name in required if name != _BINDING_KEY]
    schema["properties"] = {
        _BINDING_KEY: binding_schema,
        **schema["properties"],
    }
    schema["required"] = [_BINDING_KEY, *schema["required"]]
    return schema


def bind_compact_actions(
    value: Mapping[str, Any], actions: list[Any], expected_observation_id: str
) -> list[Any]:
    """Validate every supplied binding, then fill canonical per-action IDs.

    The compact reply requires the ID at its outer object, regardless of whether
    its action array is nested under ``action_batch``. Legacy per-action IDs are
    accepted as input only after each is checked against that same pending ID.
    """
    framed = _unwrap_tool_call(value)
    if not isinstance(framed, Mapping):
        raise ValueError("decision must be a JSON object")
    binding = framed.get(_BINDING_KEY)
    if not isinstance(binding, str) or binding != expected_observation_id:
        raise ValueError("action batch does not bind to the current observation_id")
    batch = framed.get("action_batch")
    if isinstance(batch, Mapping) and _BINDING_KEY in batch:
        nested = batch[_BINDING_KEY]
        if not isinstance(nested, str) or nested != binding:
            raise ValueError("action batch does not bind to the current observation_id")
    bound: list[Any] = []
    for action in actions:
        if not isinstance(action, Mapping):
            bound.append(action)
            continue
        if _BINDING_KEY in action:
            supplied = action[_BINDING_KEY]
            if not isinstance(supplied, str) or supplied != binding:
                raise ValueError("action batch does not bind to the current observation_id")
        bound.append({**action, _BINDING_KEY: binding})
    return bound


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
#: and the same reason, as ``_iter_json_objects`` in the tool layer. Exhaustion is
#: distinct from finding no competing batch: when the rest was not checked, the
#: parser must refuse rather than treat an unchecked tail as harmless.
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

    ``no_value_at_all`` is the ONE discriminator over this family, and it is a
    typed FIELD rather than a sentence because a sentence cannot carry it:
    :func:`_decode_leading_json` composes ``decision is not valid JSON:
    {error}`` for BOTH classes of parse failure -- the reply that never started
    a value, and the reply that started one and then broke, whose class
    ``classify_rejection`` splits on the parser's reported POSITION embedded in
    that same sentence -- so no reader of the text can tell them apart without
    re-parsing an offset out of a message. Only the first means "this payload
    states no decision"; :func:`_states_a_decision` reads the marker, and it is
    why the marker exists. Its ONE setter is the leading-object tolerance, which
    reads the classification off the typed reason it declined for
    (:class:`_LeadingObjectDecline`) -- so a new decline reason cannot silently
    land on the widening side. Every other raiser, here and in
    ``provider_client``, leaves it ``False``, which is the conservative side.
    """

    def __init__(self, *args: object, no_value_at_all: bool = False) -> None:
        super().__init__(*args)
        self.no_value_at_all = no_value_at_all


#: Distinguishes "the model wrote no ``public_observations`` key" from "the
#: model wrote an empty one". The first is a legacy actions-only batch and has
#: nothing to publish; the second is an envelope that recorded an empty note and
#: is replayed as the model's visible reply.
_ABSENT = object()


def _decode_leading_json(payload: str) -> tuple[Any, str, int]:
    """Decode the leading JSON value and return it with any trailing noise.

    Returns the value, its trailing text, and how many UTF-8 BYTES of framing
    preceded the value itself (0 for the ordinary reply, which starts with its
    decision; the size of the framing the leading tolerance skipped otherwise).
    The count is carried rather than recomputed by the caller so that a bundle's
    record of "this reply was read through the tolerance" is this decoder's own
    answer -- and it is counted in BYTES, which is what the sealed field states
    and what a consumer slicing the reply's own prefix needs (the framing this
    tolerance reads is not always ASCII: ``思考中：``, a fullwidth-bar DSML wrapper).

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
    * **Leading noise** (``Sure, here you go: {...}``) is TOLERATED under the
      rule :func:`_locate_leading_object` states and measures: the reply's FIRST
      ``{`` must begin a complete, decodable, decision-shaped JSON value, and
      NOTHING but the markers of a closing code fence may follow it. The hazard
      the earlier build refused every leading byte for is real -- hunting
      forward for the first bare ``{`` guesses where the value begins, and a
      preamble that itself carries a brace makes that guess wrong SILENTLY,
      executing a DIFFERENT batch than the model sent -- and the rule is
      structural rather than a count of candidates because of the shape that
      counting let through: when this turn's decision is unreadable, a SUPERSEDED
      decision quoted in the framing region is the only readable one, and
      executing it is worse than the rejection the tolerance removed. See that
      function for the three conditions and the measurement behind them.
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

    Why the second rule was widened, measured rather than argued. Over the
    arm-0625 OSWorld bundles (``~/worktrees/osworld/runs``, the current-code
    runs of 2026-09-24; 157 sealed ``decision-rejected`` artifacts) 60 replies
    were refused as ``leading-delimiter``. They split three ways: 43 published
    NO reply text at all (the decision arrived as a tool call the harness did
    not read as the reply channel -- a channel-recognition defect, not this
    one, and nothing in this module can recover it), 14 published text carrying
    no decision-shaped object at all (correctly refused), and 3 published a
    complete, UNIQUE decision behind junk: a prose preamble, and DeepSeek's own
    native ``<｜DSML｜parameter name="input" ...>`` tail welded to the envelope.
    Each of those 3 cost a billed corrective round trip with the model's intent
    already correct. In all 3 the framing holds no ``{`` at all and the envelope
    ends the reply, which is what lets the rule stay structural instead of
    counting candidates.

    The reply that rule is written against, in full: ``My earlier reply was:\n
    {a complete envelope}\nand it was refused because the coordinates were
    wrong.`` A count of decision-shaped objects reads THAT reply as one decision
    -- the quoted one -- and executes a superseded decision, which is strictly
    worse than the rejection the widening removed: a wrong click in a benchmark
    is unrecoverable, silently scored, and, because it is an acceptance, leaves
    no rejection artifact to notice it in. A truncated live decision after a
    quoted one is the same hole (the quote is the only COMPLETE object), and it
    is the shape the ``incomplete-json`` class -- 12.7% of these refusals --
    produces.

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
    # of loss this function exists to prevent. Whitespace at the head is not
    # part of any decision, so skipping it changes nothing; every other byte
    # before the value is the leading-noise tolerance's question, below.
    payload = payload.lstrip()
    #: How many UTF-8 bytes of framing preceded the decision's own value, after
    #: leading whitespace: 0 for a reply that began with its decision, and the
    #: size of the framing the tolerance skipped otherwise. Carried out rather
    #: than recomputed by the caller, so a bundle's record of "this reply was
    #: read through the leading tolerance" is the decoder's own answer.
    framing_bytes = 0
    try:
        decoded, end = decoder.raw_decode(payload)
    except (ValueError, RecursionError) as error:
        # Duplicate keys are refused by the hook, and their wording is kept as
        # it has always been worded: ``classify_rejection`` keys the class on
        # the phrase, and a re-spelled sentence would reclassify every sealed
        # artifact that carries it.
        if "duplicate JSON keys" in str(error):
            raise DecisionParseError(
                "model reply must be one duplicate-free JSON object"
            ) from error
        # The leading-object tolerance, gated on the ONE failure shape it exists
        # for: offset 0 could not start a value at all. The gate is the classifier's
        # own discriminator (``_LEADING_DELIMITER_RULE`` in ``provider_client``
        # reads the same phrase), so the replies this tolerance can recover are
        # exactly the ones that would otherwise be bucketed ``leading-delimiter``
        # -- and no reply that already STARTED a value is re-read here. That is
        # what keeps an object that broke inside (``incomplete-json``) and a
        # duplicate key at the head on the refusals they already had.
        if isinstance(error, json.JSONDecodeError) and "line 1 column 1 (char 0)" in str(error):
            located = _locate_leading_object(payload, decoder)
            if isinstance(located, _LeadingObjectDecline):
                # Nothing could start a value at offset 0, and the leading-object
                # tolerance could not read one behind the framing either. That
                # tolerance declines for FOUR reasons and only ONE of them means
                # the payload states no decision, so the marker
                # ``_states_a_decision`` gates the reply-channel widening on is
                # read off the DECLINE's own classification rather than
                # re-derived here from the parser's reported OFFSET -- an offset
                # is a property of the reply's FRAMING, not of the decision, and
                # three revisions of this gate each keyed a downstream heuristic
                # on it, one framing boundary at a time. A decline that is not
                # ``NOTHING_TO_READ`` is a decision the model STARTED writing
                # (round 2's class) or wrote in full and then discarded (round
                # 3's), and the conservative direction for both is the
                # pre-widening one: refuse on the prose and re-prompt, rather
                # than execute the call's envelope and never tell the model its
                # object was cut off or passed over.
                raise DecisionParseError(
                    f"decision is not valid JSON: {error}",
                    no_value_at_all=located.states_no_decision,
                ) from error
            decoded, end, value_start = located
            # Bytes, not code points. The located start is a character index
            # because that is what ``raw_decode`` takes, so the conversion
            # happens HERE, at the boundary where a character offset becomes the
            # sealed count -- a code-point index under-reports every non-ASCII
            # framing (``思考中：`` counted 5 where the reply carries 13 bytes), and
            # the field and its consumers are stated in bytes.
            framing_bytes = len(payload[:value_start].encode("utf-8"))
        else:
            raise DecisionParseError(f"decision is not valid JSON: {error}") from error
    trailing = payload[end:].strip()
    if trailing:
        observation_ids, uses_string_actions = _batch_observation_ids(decoded)
        offset, exhausted = _competing_batch_offset(trailing, observation_ids, decoder)
        if offset is not None or (exhausted and uses_string_actions):
            raise DecisionParseError(
                "decision carries a second action batch for the same observation; "
                "send exactly one action batch"
            )
    return decoded, trailing, framing_bytes


def _states_a_decision(payload: str) -> bool:
    """Whether the payload states a decision, readable or refused.

    The question the reply-channel widening is gated on. Reading an
    arbitrarily-named tool call as the channel is a RECOVERY -- it exists for a
    turn whose decision arrived on the call channel and would otherwise be lost
    (measured: 178 of the arm's 204 ``leading-delimiter`` refusal artifacts,
    recounted 2026-09-25 over ``~/worktrees/osworld/runs``, published no reply
    text at all) -- and it must therefore never step over a decision the
    model wrote in PROSE: a turn that answered on both channels would have its
    prose discarded and the call's bytes judged in its place, turning an accepted
    turn into a refusal.

    Deliberately loose, and deliberately asked of the decoder rather than
    re-derived here: any payload ``_decode_leading_json`` reads a value out of,
    or refuses for a defect INSIDE a value it read, states one. Only "nothing
    could be read at all" -- the failure the decoder marks with
    :attr:`DecisionParseError.no_value_at_all`, which is also what an empty or
    non-JSON prose reply produces -- answers ``False``.

    That marker is set from the leading tolerance's own typed decline reason
    (:class:`_LeadingObjectDecline`) and from nothing else, so the four ways
    that tolerance can decline are classified WHERE THE REASON IS CHOSEN rather
    than reconstructed here from a parser offset. Only one of them -- no ``{``
    in the payload at all -- means the model stated no decision; a truncation, a
    non-decision object, and a decision the reply does not END at all each mean
    it started or finished one, which is exactly the case this gate must not
    step over.

    The answer comes from that TYPED MARKER and never from the message text,
    because the text cannot carry it: the decoder composes one sentence for both
    parse-failure classes, so a substring test over it puts a TRUNCATED decision
    -- text that started a value and was cut off, which is a decision the model
    was writing -- on the widening side, where the pre-widening client refused
    the turn and re-prompted (26 prose-bearing refusal artifacts in the arm's
    corpus on 2026-09-25 are that shape, 16 of them beside a call, so the
    widening fired on them and the model was never told its object was cut off).

    The default is the conservative direction for the caller: a payload this
    cannot classify as decisionless WITHHOLDS the widening, so the prose is
    judged exactly as it was before the widening existed. A refusal reason added
    to the decoder later is on that default side too, because a reason that does
    not set the marker reads as a decision present -- and a decline reason added
    to the tolerance later is on it as well, because a new
    :class:`_LeadingObjectDecline` member states ``states_no_decision`` False
    until that classification is deliberately declared.
    """

    if not payload.strip():
        # The widening's own majority case -- 178 of the arm's 204
        # ``leading-delimiter`` artifacts above published no reply text at all
        # -- and an answer the marker below agrees with, so the exception is
        # skipped rather than built and discarded on every prose-only turn.
        return False
    try:
        _decode_leading_json(payload)
    except DecisionParseError as error:
        return not error.no_value_at_all
    return True


#: One line of a Markdown code fence: at least three backticks and, optionally,
#: an info string (``json``, ``python``) -- and nothing else on the line. An
#: anchor like ``\A``/``\Z`` rather than ``fullmatch`` with ``$`` so a trailing
#: newline cannot count as marker-only slack for text on the same line.
_FENCE_MARKER_LINE = re.compile(r"\A`{3,}[A-Za-z0-9_+.\-]*\Z")


def _is_fence_marker_only(remainder: str) -> bool:
    """Whether the text after a located decision is code-fence markers and whitespace.

    The one exception to "nothing may follow the located object", and it is safe
    for the reason the exception exists: a fence marker is not a value. It cannot
    carry a decision, a competition, or the context a quoted decision arrives
    with, so a remainder made only of marker lines leaves the located object the
    reply's ONLY readable statement -- which is the property condition 3 is
    there to guarantee. Refusing it bought nothing and cost the shape a model
    that follows Markdown actually writes: the closed fence
    (```` ```json\\n{decision}\\n``` ````) has been read by the offset-0 path since
    before this tolerance existed, and was refused by the located path's first
    form, while the unclosed fence -- the half-written spelling -- is what the
    tests happened to pin.

    Deliberately narrow: a header, prose, or anything beside the marker on its
    own line is not a fence marker, and any brace anywhere in the remainder still
    refuses the reply. Only backtick fences are recognised; a ``~~~`` fence has
    never been observed in this arm's traffic, and widening on that guess is
    exactly the kind of enumeration the rest of this rule avoids.
    """

    return all(
        not stripped or _FENCE_MARKER_LINE.match(stripped)
        for stripped in (line.strip() for line in remainder.splitlines())
    )


class _LeadingObjectDecline(Enum):
    """Why the leading-object tolerance could not read the reply's decision.

    :func:`_locate_leading_object` declines for FOUR distinct reasons, and a
    single ``None`` cannot say which one fired -- which is how three successive
    revisions of the reply-channel gate each ended up keying a downstream
    heuristic on the parser's reported OFFSET, a property of the reply's
    FRAMING rather than of the decision:

    * a foreign-named call overriding a valid prose decision (round 1);
    * an UNPREFACED truncated decision read as decisionless (round 2);
    * the same truncation -- and a complete decision with trailing text --
      arriving behind a preamble, a fence or native call syntax (round 3).

    So the reason is typed HERE, where it is chosen, and the classification the
    gate needs rides on the reason itself (:attr:`states_no_decision`) instead
    of being reconstructed by its caller. Only ``NOTHING_TO_READ`` means "this
    payload states no decision"; the other three each mean the model started or
    finished one, where the conservative direction is the pre-widening
    refusal-and-re-prompt.
    """

    #: No ``{`` in the payload at all: the model stated no decision to read.
    NOTHING_TO_READ = "nothing-to-read"
    #: The first ``{`` began a value that did not decode -- a decision the model
    #: was WRITING (round 2's class, which framing hides from an offset test).
    BEGINS_NOTHING = "begins-nothing"
    #: The first ``{`` decoded but states no decision: an example, or an echo of
    #: the harness's own feedback.
    NOT_DECISION_SHAPED = "not-decision-shaped"
    #: It decoded decision-shaped, but the reply does not END at it -- a decision
    #: written in full and then discarded in favour of trailing text or another
    #: object (round 3's class).
    DOES_NOT_END_AT_IT = "does-not-end-at-it"

    @property
    def states_no_decision(self) -> bool:
        """Whether this decline means the payload states no decision at all.

        A property rather than a per-member flag so a reason added later
        defaults to the conservative side without anyone having to remember it:
        a new member reads as "a decision may be present" until this line is
        deliberately widened, which is the direction
        :func:`_states_a_decision` states it wants.
        """

        return self is _LeadingObjectDecline.NOTHING_TO_READ


def _locate_leading_object(
    payload: str, decoder: json.JSONDecoder
) -> tuple[Any, int, int] | _LeadingObjectDecline:
    """The reply's leading DECISION when framing precedes it, else the decline.

    Returns the located ``(value, end, value_start)`` when the three conditions
    below hold, and otherwise the :class:`_LeadingObjectDecline` naming WHICH one
    declined -- a typed reason rather than ``None``, because the caller that
    gates the reply-channel widening on "this payload states no decision" must
    not have to reconstruct that distinction from an offset. Only
    ``NOTHING_TO_READ`` states no decision; the other three state one that is
    unreadable, not decision-shaped, or not the reply's last word.

    The exposure this rule answers is CONDITIONAL, and that belongs here because
    a bundle cannot show it: whether a misclassified reply changes a turn's
    OUTCOME depends on the call's own ARGUMENTS, and a sealed artifact publishes
    the call's name and shape rather than its bytes -- so a corpus can show that
    the widening fired where base refused and re-prompted, but not what
    executing that call would have done. The footprint is measured anyway, on
    2026-09-25 over the arm's 204 ``leading-delimiter`` artifacts: 23 sit beside
    a call, 14 of those contain a ``{``, and 10 of the 14 took the widening under
    the offset-keyed marker -- mostly the FRAMED shape, which is the shape these
    replies actually arrive in. Classifying here keeps the widening on 185 of
    the 204 (the 10 lost are replies that DID start a decision, which is the
    trade this rule makes deliberately).

    The question the second rule of :func:`_decode_leading_json` is about: a
    reply that put a preamble, a code fence or a native tool-call syntax wrapper
    in front of an otherwise byte-perfect batch. The licence is the narrowest
    one that reads those replies, and it is stated as three structural
    conditions rather than as a count of candidates:

    * **The reply's FIRST ``{`` must begin a complete, decodable JSON value.**
      Not "the first decodable object anywhere": a brace that begins nothing
      readable is an object that did not survive, and behind a preamble that is
      what a truncated or half-written decision looks like.
    * **That value must be decision-shaped** (:func:`_is_decision_shaped`). An
      object that is not a decision is an EXAMPLE -- the model demonstrating the
      shape it is about to use, or echoing the harness's own feedback -- and
      reading it would execute a decision the model did not make.
    * **Nothing may follow it but the markers of a closing code fence.** No
      second brace, no prose, no header -- a remainder holding anything with a
      value in it refuses the reply (see :func:`_is_fence_marker_only` for why a
      bare marker is the one thing that cannot carry a competing decision).

    Why the third condition is not "and no OTHER decision follows". Counting
    decision-SHAPED objects reads a superseded decision as this turn's: when
    this turn's decision is unreadable -- a truncation, which is 12.7% of the
    rejections measured on this arm -- a quoted envelope from an earlier attempt
    is then the only readable decision in the reply, and it is executed.
    Executing a stale decision is strictly worse than the rejection the tolerance
    removed: a wrong click in a benchmark is unrecoverable, silently scored, and,
    because it is an ACCEPTANCE, leaves no rejection artifact behind to notice it
    in. The quoted shape fails the third condition by construction, because a
    quote arrives with its context around it.

    Which is also why the located path is narrower than the offset-0 path it
    feeds: a reply that BEGINS with its decision keeps the one-sided trailing
    tolerance (``{...} Hope that helps!`` is still read -- the decision's
    position is the model's own statement of what it is answering with), while a
    located one must END at its decision. Everything before it is framing, about
    which nothing can be assumed; everything after it is the context a quoted
    decision is accompanied by, and this module cannot tell the two apart.

    Deliberately NOT a validation: whether the located object is ADMISSIBLE --
    its binding, its coordinates, its kind's fields -- stays the validator's
    question, so a located-but-invalid batch is refused with the class its own
    defect earns rather than disappearing into the leading one. The located
    object is handed over untouched.

    Cost: a constant number of operations -- one ``find``, one ``raw_decode``,
    one test over the remainder (two when that remainder is not empty) -- so this
    needs no attempt bound and states none. There is no scan whose exhaustion
    could leave a candidate unexamined, which is a property the earlier
    candidate-counting form had to earn with a bound and did not have.
    """

    start = payload.find("{")
    if start < 0:
        # The ONE decline that means the model stated no decision: there is no
        # brace to read. Every return below this line is a decision ATTEMPT.
        return _LeadingObjectDecline.NOTHING_TO_READ
    try:
        candidate, end = decoder.raw_decode(payload, start)
    except (ValueError, RecursionError):
        # RecursionError as well as ValueError (see ``_competing_batch_offset``),
        # and a candidate whose own keys are duplicated, which the hook refuses:
        # a reply carrying one is not read at all, the direction the
        # duplicate-key rule already set.
        return _LeadingObjectDecline.BEGINS_NOTHING
    if not _is_decision_shaped(candidate):
        return _LeadingObjectDecline.NOT_DECISION_SHAPED
    if payload[end:].strip() and not _is_fence_marker_only(payload[end:]):
        # A second brace -- a competing decision, an example, or an object that
        # did not survive -- or the context a quoted decision arrives with.
        # Either way the reply does not END at the decision it was read from.
        return _LeadingObjectDecline.DOES_NOT_END_AT_IT
    return candidate, end, start


def _is_decision_shaped(value: Any) -> bool:
    """Whether an object states a decision, judged by the normaliser's own rule.

    Deliberately NOT a validation. Whether the actions inside are admissible, or
    even well-typed, is the action protocol's question and stays the validator's;
    a located object that fails it must be refused with THAT class rather than
    disappearing into the leading one. All this asks is whether the normaliser
    would read a decision out of the object at all, so the located candidate and
    the accepted reply are judged by one rule instead of two.

    Degrades rather than raising, for the reason ``_batch_observation_ids``
    states: this runs over every brace in untrusted model text, so a wrapper
    whose payload is not JSON must read as "no decision here", never as an
    unexpected exception on the decode path.
    """

    try:
        unwrapped = _unwrap_tool_call(value)
    except DecisionParseError:
        return False
    return isinstance(unwrapped, Mapping) and _carries_decision(unwrapped)


def _competing_batch_offset(
    trailing: str,
    observation_ids: set[str],
    decoder: json.JSONDecoder,
) -> tuple[int | None, bool]:
    """The competing batch offset and whether unexamined candidates remain.

    "Competes" means it names the SAME ``observation_id``: only a decision
    about the screen currently in front of the model can supersede the one
    already parsed. See :func:`_decode_leading_json` for why that test, rather
    than adjacency or bare JSON-ness, is the one that separates a superseding
    batch from the harness feedback a model quotes back at itself.

    Compare candidate object IDs with the supplied observation IDs. Returns
    the competing object's offset, if found, and whether the candidate
    budget was exhausted while more objects remained unchecked. Ordinary prose
    without candidate objects returns ``(None, False)`` and stays cheap. Exhaustion
    is reported separately so only the new string-coercion path needs to refuse;
    legacy array replies keep their established bounded best-effort behavior.
    """

    if not observation_ids:
        return None, False
    # A decision is always an object, so only "{" can start a competing batch;
    # the scan is bounded the same way ``_iter_json_objects`` is bounded, since
    # a remainder full of bare braces would otherwise cost a rescan each.
    index = 0
    attempts = 0
    while attempts < _MAX_TRAILING_DECODE_ATTEMPTS:
        start = trailing.find("{", index)
        if start < 0:
            return None, False
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
        candidate_ids, _ = _batch_observation_ids(candidate)
        if candidate_ids & observation_ids:
            return start, False
    # Exhaustion matters only when there is at least one candidate that the
    # bounded scan did not inspect; exactly 256 harmless objects followed by
    # ordinary prose is fully checked and remains accepted.
    return None, trailing.find("{", index) >= 0


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
    keeps the rule a reply already gets: the string's leading JSON value is read
    (including past a preamble, under :func:`_locate_leading_object`'s
    exactly-one-decision rule), and a second batch for the same observation
    inside the string still refuses the reply, so this can never execute a
    decision the model superseded.

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
        decoded, trailing, _framing = _decode_leading_json(value)
    except DecisionParseError:
        return value, False
    if not isinstance(decoded, list) or not decoded:
        return value, False
    if not all(isinstance(action, Mapping) for action in decoded):
        return value, False
    if trailing:
        observation_ids, _uses_string_actions = _batch_observation_ids({"actions": decoded})
        offset, exhausted = _competing_batch_offset(
            trailing,
            observation_ids,
            # The decoder is built with the same hook as ``_decode_leading_json``'s
            # so a candidate carrying duplicate keys is refused identically.
            json.JSONDecoder(object_pairs_hook=_unique_object),
        )
        if offset is not None or exhausted:
            # Unlike the legacy array spelling, this reply is accepted only by
            # coercing the string to actions; if the bounded scan left candidate
            # objects unchecked, the ambiguity guarantee cannot be established.
            return value, False
    return decoded, True


def _batch_observation_ids(value: Any) -> tuple[set[str], bool]:
    """The observation ids an action-batch-shaped object binds to.

    Read from the ACTIONS rather than from a top-level ``observation_id``: a
    model reply carries the id per action (the runner supplies the batch-level
    one itself), so a top-level lookup finds nothing on the very shape this
    needs to compare. Returns an empty set for anything that is not batch
    shaped, which the caller treats as "not a competing decision". The second
    result indicates whether an accepted action array came from string coercion;
    only that new spelling must fail closed when the bounded outer scan expires.

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
        return set(), False
    if not isinstance(value, Mapping):
        return set(), False
    # Match the normaliser's two locations independently. In particular, the
    # top-level ``actions`` value may be a JSON-encoded string; ignoring it here
    # would make the outer trailing-batch scan blind to the very spelling that
    # normalization accepts. Unioning both valid arrays is conservative for an
    # ambiguous object carrying both locations: either one may bind a competing
    # decision to this observation.
    nested = value.get("action_batch")
    nested_value = nested.get("actions") if isinstance(nested, Mapping) else nested
    observation_ids: set[str] = set()
    uses_string_actions = False
    for raw_actions in (value.get("actions"), nested_value):
        actions, coerced = _actions_from_json_string(raw_actions)
        uses_string_actions = uses_string_actions or coerced
        if not isinstance(actions, list):
            continue
        observation_ids.update(
            action["observation_id"]
            for action in actions
            if isinstance(action, Mapping) and isinstance(action.get("observation_id"), str)
        )
    return observation_ids, uses_string_actions


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
            # The framing offset is the INNER string's question and is not
            # carried out of the normaliser: a wrapper's tolerance is not this
            # reply's framing, and the record belongs to the reply that was read.
            inner, _trailing, _framing = _decode_leading_json(inner)
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


def _report_ignored_keys(
    framed: Mapping[str, Any], batch: Any, *, action_binding: str = LEGACY_ACTION_BINDING
) -> None:
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

    known_reply_keys = _REPLY_KEYS | (
        {_BINDING_KEY} if action_binding == COMPACT_ACTION_BINDING else set()
    )
    nested_keys = {"actions", "public_observations"}
    if action_binding == COMPACT_ACTION_BINDING:
        nested_keys.add(_BINDING_KEY)
    stray = sorted(set(framed) - known_reply_keys)
    if isinstance(batch, Mapping):
        stray += sorted(set(batch) - nested_keys)
    if stray:
        logger.warning(
            "model reply carried %d key(s) the reply contract does not use; ignored: %s",
            len(stray),
            _unexpected_key_summary(stray),
        )


def normalise_public_reply(
    value: Any, *, action_binding: str = LEGACY_ACTION_BINDING
) -> tuple[list[Any], str | None, int]:
    """The one accepted reply shape, however the reply was framed.

    Returns the action array, the model's public note -- ``None`` for the note
    when the reply carried no ``public_observations`` key at all, which is what
    still separates a legacy actions-only batch (nothing to publish) from an
    envelope that recorded an empty note -- and how many action fields the
    sibling-kind tolerance dropped (:func:`drop_sibling_action_fields`),
    carried out of here because an accepted reply is the only place those
    replies are still visible: the ones it recovers stop producing the rejection
    artifacts that used to make the class countable.

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

    action_binding = _validate_action_binding(action_binding)
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
    # An action's OWN framing, and therefore this module's business by the same
    # rule the envelope's is: which fields an action carries is declared by its
    # ``kind``, so a field of a sibling kind states nothing the kind did not
    # already state. Dropped and reported -- see
    # :func:`drop_sibling_action_fields` for why dropping is the disposition this
    # contract uses and what stays refused.
    actions, tolerated_action_fields = drop_sibling_action_fields(actions)
    note = _public_note(framed, batch)
    _report_ignored_keys(framed, batch, action_binding=action_binding)
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
    return actions, note, tolerated_action_fields


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

    value, _trailing, _framing = _decode_leading_json(payload)
    actions, note, _dropped = normalise_public_reply(value)
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

#: The fields each action KIND declares, keyed by its ``kind`` discriminator.
#: Derived from the models rather than transcribed: a field added to a kind is
#: known to :func:`drop_sibling_action_fields` on the same commit that adds it,
#: where a hand-kept copy would drift, and the drift would be silent -- the
#: tolerance would simply stop recognising the field it exists for.
_ACTION_KIND_FIELDS: dict[str, frozenset[str]] = {
    kind: frozenset(model.model_fields)
    for model in _ALL_ACTION_MODELS
    for kind in get_args(model.model_fields["kind"].annotation)
}

#: Every field name the action vocabulary declares, across all kinds. A name in
#: here that the DECLARED kind does not take belongs to a SIBLING kind, and is
#: the only extra key this module drops. A name absent from here is not part of
#: the vocabulary at all and keeps the validator's ``extra_forbidden`` refusal:
#: nothing can say what the model meant by it, so dropping it would guess.
_ACTION_FIELD_NAMES: frozenset[str] = frozenset().union(*_ACTION_KIND_FIELDS.values())


def drop_sibling_action_fields(actions: list[Any]) -> tuple[list[Any], int]:
    """Drop fields that belong to a SIBLING action kind, and say so.

    THE MEASURED DEFECT. A model that puts ``frame_id`` on a ``wait``, or
    ``duration_ms`` on a ``click``, states a complete and unambiguous decision:
    the ``kind`` tag is required and explicit, so the stray field cannot change
    WHICH action the model chose, and the fields the chosen kind takes are all
    still there to be validated. Refusing the whole batch for it cost a billed
    corrective round trip with nothing wrong with the decision -- 46 of the 145
    decision-rejections counted over the arm-0625 episodes (31.7%) were exactly a
    field of a sibling kind, across ``frame_id``/``duration_ms``/``delta_y``/
    ``text`` on the kinds that do not take them; replaying the sealed corpus
    through both decoders recovers 23 of that kind and no other reply changes
    verdict (283 published rejected replies, 2026-09-24).

    WHY DROP RATHER THAN REFUSE. It is the disposition this contract already uses
    for framing that carries no decision information: an extra key the reply
    contract does not use is ignored and reported (``_report_ignored_keys``), a
    ``reply_version`` in the wrong place is ignored, and a generic tool-call
    wrapper is unwrapped. A field of a sibling kind is the action-level case of
    exactly that, and the alternative dispositions are both worse. Rewriting it
    into the action it belongs to (``duration_ms`` becoming a ``wait``) invents
    an action the model did not ask for, in an ORDERED batch, which changes what
    executes. Keeping the refusal spends a paid round trip on a decision that is
    already complete, and no hint can repair it better than reading it.

    What is deliberately NOT dropped, and stays refused:

    * a key that is not a field of ANY kind (``extra_forbidden``): nothing says
      what the model meant by it, and a near-miss field name is a real mistake;
    * an action whose ``kind`` is missing or not a known kind: there is no
      declared vocabulary to measure its fields against, so the mapping is left
      byte-identical and the validator refuses it with its own class;
    * a required field the chosen kind takes being absent, or present with the
      wrong type: dropped here means SIBLING fields only, so those still fail
      validation exactly as before.

    Tolerated is not the same as silent, and the report is one bounded log line
    per reply -- the same rule, and the same shared renderer, the extra-key
    tolerance uses. Both reply channels call this: the prose envelope (through
    :func:`normalise_public_reply`) and the offered reply-channel call
    (``action_tool._build_batch``), so the tolerance cannot come to mean two
    different things depending on which channel the model answered on.

    Returns the normalised actions and HOW MANY fields were dropped, and the
    count is part of the contract rather than a courtesy: a tolerance nobody can
    count is indistinguishable from one that stopped firing, and this one's
    whole justification is a measured rate. The dropped names themselves are
    model-text-free (``kind.field`` out of a fixed vocabulary) but the signed
    bundle carries only the count, and it carries it as its own
    ``reply_tolerance`` event (``ReplyTolerancePayload``) rather than as a field
    of the response it belongs to: that event is what makes the rate countable
    from bundles written now WITHOUT re-baselining the ``model_response`` events
    every earlier bundle was sealed with. The log line keeps the roster.

    Precondition: ``actions`` is a list of the reply's own action values. Every
    caller has already established that -- a reply whose ``actions`` is not a
    list is refused above them, by the decoder rather than by this tolerance --
    so there is no pass-through branch here to describe, and a caller that broke
    the precondition would fail loudly instead of silently dropping the reply's
    fields.
    """

    dropped: list[str] = []
    normalised: list[Any] = []
    for action in actions:
        if not isinstance(action, Mapping):
            normalised.append(action)
            continue
        kind = action.get("kind")
        declared = _ACTION_KIND_FIELDS.get(kind) if isinstance(kind, str) else None
        if declared is None:
            normalised.append(action)
            continue
        foreign = sorted(
            key for key in action if key not in declared and key in _ACTION_FIELD_NAMES
        )
        if not foreign:
            normalised.append(action)
            continue
        # Named ``kind.field`` so the line says which action kind received the
        # field and which kind it belongs to, out of a fixed vocabulary rather
        # than out of model text -- the renderer is shared anyway, because the
        # bound and the guard are the security property and a second copy of
        # them would be a second place to get it wrong.
        dropped.extend(f"{kind}.{key}" for key in foreign)
        normalised.append({key: value for key, value in action.items() if key not in foreign})
    if dropped:
        logger.warning(
            "model reply put %d action field(s) on a kind that does not take them; "
            "dropped, and the action was judged on its declared kind: %s",
            len(dropped),
            _unexpected_key_summary(dropped),
        )
    return normalised, len(dropped)


def public_reply_schema(
    action_surface: ActionSurface | None = None, action_binding: str = LEGACY_ACTION_BINDING
) -> dict[str, Any]:
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
    action_binding = _validate_action_binding(action_binding)
    models = action_surface.models if action_surface is not None else _ALL_ACTION_MODELS
    schema = {
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
    return (
        _compact_action_schema(schema, models)
        if action_binding == COMPACT_ACTION_BINDING
        else schema
    )


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


def public_reply_contract(action_binding: str = LEGACY_ACTION_BINDING) -> dict[str, Any]:
    """Publish a reproducible reply identity separately from the tool surface.

    The action array schema is borrowed, not copied: its vocabulary and bounds
    remain owned by ActionBatch. Negotiated execution restrictions are declared
    by the existing action_surface metadata/tool digest and still gate parsing.

    The framing and binding entries describe what the DECODER accepts, which is
    deliberately wider than the schema it publishes: a reader of a bundle should
    be able to tell, without reading this module, why a reply with extra keys or
    a stale ``observation_id`` was accepted rather than refused.
    """
    action_binding = _validate_action_binding(action_binding)
    schema = public_reply_schema(action_binding=action_binding)
    contract = {
        "schema": schema,
        "accepted_framings": (
            "one leading JSON object, any trailing text after it, and any preamble, code "
            "fence or native tool-call wrapper in FRONT of it when the reply carries "
            "exactly one decision; duplicate keys and a second action batch for the same "
            "observation are refused"
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
        "action_fields": (
            "a field that belongs to a SIBLING action kind is dropped and reported, never "
            "refused, because the required kind tag already states the action chosen; a "
            "field no kind declares, a missing or unknown kind, and a missing or mistyped "
            "required field are still refused"
        ),
        "public_observations": (
            "concise new observed facts/progress only; no deliberation or credentials"
        ),
    }
    if action_binding == COMPACT_ACTION_BINDING:
        contract["action_binding"] = COMPACT_ACTION_BINDING
        contract["binding"] = (
            "the required top-level observation_id must equal the current observation; "
            "any supplied legacy per-action IDs must also match before canonical actions "
            "are materialized"
        )
    return {
        "model_reply_contract": canonical_bytes(contract).decode("utf-8"),
        "model_reply_contract_digest": canonical_digest("runner-model-reply-v1", contract),
    }
