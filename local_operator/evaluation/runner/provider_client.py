"""The only runner module allowed to reach into ``local_operator.model``.

``episode.py`` must stay free of provider, config, and session imports so an
episode cannot inherit the operator's live configuration. That constraint has
to break somewhere for a real run, and it breaks here, behind
:class:`~local_operator.evaluation.runner.model.EpisodeModelClient`.

Strict decision parsing also lives here rather than in the runner core. A
malformed decision is surfaced as :class:`DecisionRejected` -- a billed call
whose reply is unusable -- which the runner records and retries with the
rejection fed back, so the parsing rules and the feedback that corrects them
stay in the same module.

**Context management** also lives here, on the central compaction engine
(``local_operator.compaction``) rather than a second implementation. The
runner hands over protocol-typed ``EpisodeTurn``s; :class:`_ContextBuilder`
turns them into an APPEND-ONLY message history — user(text + frame) /
assistant(the batch it chose, verbatim) — and the only thing that ever rewrites
a sent message is a whole-prefix rebuild through ``run_compaction_pass``. The
cadence of those rebuilds is the one real design tension in this module and
is documented on :meth:`_ContextBuilder.build`.
"""

from __future__ import annotations

import base64
import json
import re
import textwrap
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence, get_args, get_origin

from local_operator.evaluation.action_surface import (
    LEGACY_ACTION_SURFACE,
    ActionSurface,
)
from local_operator.evaluation.adapters.supervisor import verify_artifact
from local_operator.evaluation.evidence.models import RouteIdentity
from local_operator.evaluation.protocol import ActionBatch, Observation
from local_operator.evaluation.runner.model import (
    CompactionRecord,
    DecisionRejected,
    EpisodeTurn,
    ModelDecision,
    ModelUsage,
)
from local_operator.evaluation.runner.public_reply import (
    MAX_PUBLIC_OBSERVATIONS_CHARS,
    REJECTED_PUBLIC_REPLY,
    REPLY_VERSION,
    decode_public_reply,
    is_public_reply,
    looks_like_public_reply,
    public_reply_contract,
)
from local_operator.logger import get_logger

if TYPE_CHECKING:
    from local_operator.compaction.thresholds import CompactionSettings
    from local_operator.harness.types import Message

logger = get_logger(__name__)

#: Frames kept verbatim in the history by default. A GUI agent reads the
#: screen as STATE: the current frame is what it acts on and the last couple
#: are what it compares against, while anything older is a view the surface
#: has since replaced. This is a behavioural constant, not a tuning knob for a
#: particular benchmark -- it is the same ``keep_recent_frames`` an interactive
#: screen-driving session would set on its ``CompactionSettings``.
DEFAULT_KEEP_RECENT_FRAMES = 3

#: How many frames beyond ``keep_recent_frames`` may accumulate before the
#: whole prefix is rebuilt. See ``_ContextBuilder.build`` for why pruning is
#: batched into rebuilds instead of done per turn.
DEFAULT_REBUILD_EVERY_FRAMES = 8

#: Longest slice of a rejected reply replayed into the context as the model's
#: own words before the correction. The reply has to be there -- the model
#: must see WHAT it said to fix it -- but a runaway reply (a provider's
#: max-token wall of prose) must not cost the whole window on the retry.
MAX_REJECTED_REPLY_CHARS = 4_000

#: Longest run of tolerated trailing noise quoted into the observable warning
#: when a decision parses as a leading JSON value followed by junk. The point
#: of the quote is to let a reader RECOGNISE what the model appended, not to
#: reproduce it: a model that ran away into a wall of prose must not be able to
#: push a bounded log line into an unbounded one.
MAX_TOLERATED_TRAILING_CHARS = 200

#: How many candidate ``{`` positions the trailing-remainder scan may try
#: before giving up. Each failed decode rescans forward one character, so an
#: unbounded scan over a remainder full of bare braces goes quadratic -- the
#: same bound, and the same reason, as ``_iter_json_objects`` in the tool
#: layer. Giving up means "no competing batch found", which degrades to the
#: tolerant path rather than to an error.
_MAX_TRAILING_DECODE_ATTEMPTS = 256

#: Rendered in place of an observation's text when it is byte-identical to the
#: previous turn's. The adapter owns observation text and the runner never
#: second-guesses it; this is the one append-only-safe dedup, because it only
#: changes the message being appended.
UNCHANGED_OBSERVATION = "(unchanged)"

#: Rendered when an observation carries NO text at all. Kept distinct from
#: :data:`UNCHANGED_OBSERVATION` because the two mean opposite things and a
#: model that conflates them loses its only textual progress signal: a
#: benchmark whose adapter publishes the task once and then screenshots only
#: (OSWorld) yields ``text=None`` on every step after the first, and folding
#: those into "(unchanged)" told the model the SCREEN had not moved on every
#: single turn of a real paid episode.
NO_TEXTUAL_STATE = "(no textual state)"

#: Appended to an observation whose frame bytes are byte-identical to the
#: previous observation's. This is the only reliable "nothing happened" signal
#: a screenshot-only benchmark has: without it a model that clicked a dead
#: pixel sees a new observation id, a new image block, and no statement that
#: the two images are the same, so re-deciding the same click is the rational
#: reading of the context rather than a lapse.
UNCHANGED_FRAMES_NOTE = (
    "The screenshot is byte-identical to the previous observation's: your last "
    "action changed nothing visible. Do not repeat it -- try a different "
    "target, a different action kind, or wait for the surface to settle."
)

# The batch wire version this harness speaks; pinned here rather than taken
# from a model reply.
PROTOCOL_VERSION = "1.0"

#: Function keys are collapsed to a range in the prompt rather than listed:
#: F1-F24 is 24 of the vocabulary's 43 entries and the pattern is obvious.
_FUNCTION_KEY = re.compile(r"F\d+")

# Mirrors receipts.StrictIdentifier, which every evidence identifier must match.
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")


def _action_schema_lines(surface: ActionSurface = LEGACY_ACTION_SURFACE) -> list[str]:
    """Describe every action kind by reading the protocol models themselves.

    A parse failure costs a billed corrective re-prompt at best and the
    episode at worst (``EpisodeConfig.max_decision_retries``), so a prompt
    that under-specifies the wire shape is a correctness defect, not prompt
    polish. Deriving the text from ``ComputerAction`` means a new action kind
    or a changed literal cannot silently drift out of the instructions the
    model is given.
    """

    lines: list[str] = []
    for action in surface.models:
        fields: list[str] = []
        for name, field in action.model_fields.items():
            if name in ("kind", "observation_id"):
                continue
            choices = get_args(field.annotation)
            if choices and all(isinstance(choice, str) for choice in choices):
                rendered = "|".join(str(choice) for choice in choices)
            else:
                rendered = _type_name(field.annotation)
            optional = "" if field.is_required() else " (optional)"
            # The name is QUOTED like the two literal keys beside it. Rendered
            # bare it reads as a label rather than a JSON key: a real episode
            # answered the bare ``keys: [str, ...]`` line with ``"key":
            # ["Alt", "F10"]`` and lost the turn to extra_forbidden/missing.
            fields.append(f'"{name}": {rendered}{optional}')
        kind = action.model_fields["kind"].default
        detail = ", ".join(fields) if fields else "no further fields"
        lines.append(f'  {{"kind": "{kind}", "observation_id": "<id>", {detail}}}')
    return lines


def _named_keys_line(surface: ActionSurface) -> str:
    """The key vocabulary, wrapped, derived from the set the validator enforces.

    Stated rather than implied. ``KeyAction`` accepts a closed set and rejects
    everything else, so a model reaching for the obvious synonym loses the
    turn: probed against the real parser, a reply of ``["control", "k"]`` is
    refused with ``unknown key: 'control'`` while ``["ctrl", "k"]`` parses.
    The function keys are collapsed to a range because listing F1-F24 in full
    spends prompt budget on a pattern one phrase conveys.
    """

    named = sorted(key.lower() for key in surface.named_keys if not _FUNCTION_KEY.fullmatch(key))
    wrapped = textwrap.wrap(", ".join(named) + ", and f1 through f24", width=68)
    return textwrap.indent("\n".join(wrapped), "  ")


def _type_name(annotation: Any) -> str:
    """Name a field's JSON shape well enough for a model to emit it correctly.

    Sequences must render as arrays rather than the generic placeholder: told
    only "value", a model emits ``"ctrl+c"`` for ``KeyAction.keys`` where the
    protocol requires ``["ctrl", "c"]``, and that is a hard parse failure with
    no retry -- the same terminal-failure class B2 addressed, just narrower.
    """

    name = getattr(annotation, "__name__", None)
    if name in ("int", "str", "bool", "float"):
        return name
    origin = get_origin(annotation)
    if origin in (tuple, list, set, frozenset):
        args = [arg for arg in get_args(annotation) if arg is not Ellipsis]
        inner = _type_name(args[0]) if args else "value"
        return f"[{inner}, ...]"
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(args) == 1:
        return _type_name(args[0])
    return "value"


def _paste_instructions(surface: ActionSurface) -> str:
    if not surface.paste_text:
        return ""
    return """* "paste_text" is the supported Unicode path: replace CLIPBOARD with text,
  then send exactly the REQUIRED "keys" chord. Choose the chord for the focused
  application (for example ["ctrl", "v"] or ["ctrl", "shift", "v"]); there is
  NO default chord, focus change, automatic Enter, retry, or keyboard fallback.
  REQUIRED "clipboard_policy" must be "overwrite". The new text remains on
  CLIPBOARD; the old clipboard is not restored and PRIMARY is untouched.
  Text must be valid Unicode, 1 to 100000 characters; whitespace-only text is
  allowed. Tabs, CR and newlines are data; other control characters are refused.
  The receiving application may normalize text or submit on a newline. A wrong
  chord can do something else or nothing: inspect the next observation; success
  of the action does NOT prove insertion or task completion.
"""


def build_system_prompt(surface: ActionSurface = LEGACY_ACTION_SURFACE) -> str:
    """Compose the episode system prompt around the live protocol schema."""

    native_text = (
        "Only ASCII text is supported by this adapter; "
        "non-ASCII is rejected before any batch action."
        if surface.type_text_mode == "ascii"
        else "This adapter supports Unicode native text input."
    )
    ask_text = (
        '* "ask_user" -- you need a human answer. '
        "THE EPISODE PAUSES until the answer is delivered. "
        "Do not ask a question you can resolve by acting."
        if surface.ask_user
        else ""
    )
    return f"""You are operating a computer to complete one task.

Each user message is one observation of the screen: its text, and a screenshot
when one is attached. Your own earlier replies contain the actions you already
took and any public factual observations you recorded.
"{UNCHANGED_OBSERVATION}" means the observation's TEXT repeats the
previous one's; "{NO_TEXTUAL_STATE}" means the adapter published no text for
this step, which is normal for a screenshot-only benchmark and means the
screenshot is the whole state; "[screenshot omitted ...]" marks an older
screenshot that a newer one has replaced. A message starting
<previous-context-summary> summarises turns that are no longer shown.

An observation that says the screenshot is byte-identical to the previous
one's is telling you your last action did nothing. Treat that as evidence, not
noise: repeating an action that already had no effect wastes the step budget.
Change something -- a different target, a different action kind, a "wait" if
the surface may still be painting, or a scroll to bring the target into view.

Every observation lists its frames on a "Frames:" line as "<frame_id>
(<width>x<height>)", one per attached screenshot, in order. Any action that
takes a "frame_id" MUST use one of the ids named on the CURRENT observation's
"Frames:" line, exactly as written. Coordinates are in that frame's own pixel
space: x/y are integers, the origin (0,0) is its TOP-LEFT corner, x grows
right, y grows down, and both must lie inside the width and height given on
that line (0-based, so the largest valid x is width-1). The screenshot you are
shown IS that pixel space at exactly that size -- do not rescale it, do not
normalize to 0-1, and do not assume a different resolution. Never invent a
frame id or number the frames yourself.

If a reply of yours is rejected, the next user message starts "Your previous
reply was rejected:" and names the defect. Reply again for the SAME
observation with a corrected batch; nothing was executed.

Reply with a single JSON object and nothing else, with no prose and no code
fence:

  {{"reply_version": "{REPLY_VERSION}", "action_batch": {{"actions": [ ... ]}},
   "public_observations": ""}}

This is a MODEL-REPLY envelope, not the adapter protocol. Use exactly these
three keys; action_batch contains only actions. public_observations is a string
of at most {MAX_PUBLIC_OBSERVATIONS_CHARS} characters; empty is valid.
Record only concise NEW factual data
or visible progress observed on the CURRENT screen that may be needed later,
because old screenshots are removed before text summarization. Do not repeat
prior notes, invent facts, record credentials/secrets, or provide deliberation,
plans, explanations of your decision, or private reasoning. Do not claim the
chosen actions succeeded until a later observation shows their result.
Legacy replies containing only {{"actions": [ ... ]}} are also accepted.

Every action is an object whose type is given by the key "kind" (NOT "type"),
and every action must carry the "observation_id" of the observation you are
looking at right now. These are the only permitted shapes:

{chr(10).join(_action_schema_lines(surface))}

Field names are JSON keys spelled exactly as quoted above -- "keys" is not
"key", "text" is not "value". Where a field lists alternatives separated by
"|", you must use exactly one of those literal values.

You drive a real keyboard as well as a mouse, and most tasks cannot be done by
clicking alone: entering a title, a search term, an address, a date, or a file
name all require typing.

* "type" enters literal text wherever the keyboard focus already is. It does
  NOT click first, so focus the field with a click (or Tab) in an earlier
  action, then type. It uses native keyboard input, not the clipboard.
  {native_text}
  It does not press Enter for you.
{_paste_instructions(surface)}
* "key" presses named keys, and presses the ones in a single action TOGETHER
  as one chord: ["ctrl", "s"] is Ctrl+S, ["enter"] commits a field or dialog,
  ["tab"] moves focus, ["esc"] dismisses, and ["ctrl", "a"] followed by a
  "type" replaces a field's contents. Two presses in sequence are two separate
  "key" actions, not one chord. A single printable character is also a key, so
  ["ctrl", "k"] is valid. These are the ONLY named keys accepted, and a
  synonym is rejected -- "ctrl" not "control", "esc" not "escape", "enter"
  not "return":

{_named_keys_line(surface)}

A batch may contain SEVERAL actions and they execute in order against the
screen you are looking at, with one new observation at the end. That is how a
click-then-type-then-Enter sequence is done in one step rather than three.
Batch only what you can predict without seeing the screen in between; when the
result of an action decides the next one, end the batch and look.

Terminal actions end your turn in a special way, and each must be the ONLY action in
its batch. Their fields are listed above; what the list cannot tell you is what
they MEAN:

* "finish" -- you believe the task is done. The episode is then scored.
{ask_text}
"""


_SYSTEM_PROMPT = build_system_prompt()


class DecisionParseError(ValueError):
    """The provider returned something that is not a usable action batch."""


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
    quotation of an older turn and cannot. ``ActionBatch`` would refuse the
    latter as stale anyway, so nothing executable is being discarded.
    """

    decoder = json.JSONDecoder()
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
    except json.JSONDecodeError as error:
        # Includes the leading-junk case: raw_decode starts at offset 0, so a
        # preamble fails here rather than being skipped past.
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


def _batch_observation_ids(value: Any) -> set[str]:
    """The observation ids an action-batch-shaped object binds to.

    Read from the ACTIONS rather than from a top-level ``observation_id``: a
    model reply carries the id per action (the runner supplies the batch-level
    one itself), so a top-level lookup finds nothing on the very shape this
    needs to compare. Returns an empty set for anything that is not batch
    shaped, which the caller treats as "not a competing decision".
    """

    if not isinstance(value, Mapping):
        return set()
    if is_public_reply(value):
        value = value.get("action_batch")
        if not isinstance(value, Mapping):
            return set()
    actions = value.get("actions")
    if not isinstance(actions, list) or not actions:
        return set()
    return {
        action["observation_id"]
        for action in actions
        if isinstance(action, Mapping) and isinstance(action.get("observation_id"), str)
    }


def parse_decision(
    payload: str,
    observation: Observation,
    *,
    route: RouteIdentity,
    usage: ModelUsage | None = None,
    cost_micros: int = 0,
    stop_reason: str = "stop",
    provider_request_id: str = "unknown",
    tool_call_count: int = 0,
    prompt_cache_key: str | None = None,
    context_tokens: int | None = None,
    compaction: CompactionRecord | None = None,
    action_surface: ActionSurface = LEGACY_ACTION_SURFACE,
) -> ModelDecision:
    """Parse one strict JSON decision and bind it to the current observation.

    Strictness is deliberate: a batch whose actions name a different
    observation is stale, and executing it would apply a decision made about a
    screen the environment has already moved past. ``ActionBatch.validate_for``
    rejects that at the adapter boundary, so the failure is surfaced here where
    it can be attributed to the model and fed back to it.

    Strict about the DECISION, not about the framing around it: text appended
    after a complete batch is tolerated and reported rather than costing the
    turn, while a second batch for the SAME observation anywhere in that text
    is still refused as ambiguous (see :func:`_decode_leading_json` for which
    shapes are and are not tolerated, and why the tolerance only ever runs
    forwards).
    """

    decoded, trailing = _decode_leading_json(payload)
    if not isinstance(decoded, Mapping):
        raise DecisionParseError("decision must be a JSON object")
    public_reply = None
    if is_public_reply(decoded):
        try:
            envelope = decode_public_reply(payload)
        except ValueError as error:
            raise DecisionParseError(str(error)) from error
        decoded = envelope["action_batch"]
        # Keep the visible response, not a reconstruction from its actions. It
        # is redacted at the runner's resolved-secret boundary before replay.
        public_reply = payload.strip()
    if trailing:
        # Tolerated, but never silent. A model that reliably appends junk is a
        # signal worth seeing -- it may point at a prompt or provider problem
        # -- and a tolerance nobody can observe is indistinguishable from the
        # harness quietly mangling a reply. Bounded because the remainder is
        # untrusted model output of arbitrary length.
        quoted = trailing[:MAX_TOLERATED_TRAILING_CHARS]
        if len(trailing) > MAX_TOLERATED_TRAILING_CHARS:
            quoted += "[...]"
        logger.warning(
            "decision carried %d character(s) of trailing text after a complete "
            "JSON value; the batch was accepted and the remainder ignored: %r",
            len(trailing),
            quoted,
        )
    actions = decoded.get("actions")
    if not isinstance(actions, list) or not actions:
        raise DecisionParseError("decision must carry a non-empty actions array")
    try:
        batch = ActionBatch.model_validate(
            {
                # The wire version is pinned by the harness, never by the model:
                # a reply cannot select which protocol it is validated against.
                "protocol_version": PROTOCOL_VERSION,
                "task_id": observation.task_id,
                "episode_id": observation.episode_id,
                "observation_id": observation.observation_id,
                "actions": actions,
            },
            strict=True,
        )
    except Exception as error:
        raise DecisionParseError(f"decision is not a valid action batch: {error}") from error
    try:
        batch.validate_for(observation)
        action_surface.validate_batch(batch)
    except Exception as error:
        raise DecisionParseError(f"decision does not match this observation: {error}") from error
    return ModelDecision(
        action_batch=batch,
        public_reply=public_reply,
        route=route,
        usage=usage or ModelUsage(),
        cost_micros=cost_micros,
        stop_reason=stop_reason,
        provider_request_id=provider_request_id,
        tool_call_count=tool_call_count,
        prompt_cache_key=prompt_cache_key,
        context_tokens=context_tokens,
        compaction=compaction,
    )


#: Stop reasons that end a stream having DELIVERED content (or having been cut
#: off mid-content). Everything else -- ``refusal``, ``error``, ``aborted``, and
#: any marker a future wire client normalizes to -- ended the turn without the
#: provider ever committing to an answer. Named as an allow-list rather than a
#: deny-list of the abnormal ones so a newly introduced abnormal marker is
#: classified as abnormal by default: mis-reading an outage as a normal stop is
#: the failure this constant exists to prevent, and mis-reading a normal stop as
#: an outage would be caught immediately by the parse tests.
_NORMAL_CONTENT_STOPS = frozenset({"stop", "length", "toolUse"})

#: Stands in for a terminal marker the stream never supplied -- either no ``end``
#: event arrived at all, or one arrived carrying an empty ``stop_reason``. It is
#: deliberately a value OUTSIDE ``_NORMAL_CONTENT_STOPS``: coercing an absent
#: marker to ``"stop"`` would punch a fail-OPEN hole through an allow-list whose
#: entire purpose is to fail closed, and a probe confirmed the hole was live
#: (``stop_reason=""`` with empty text was classified as a normal stop and fell
#: into the JSON-parse misdiagnosis this module exists to remove). It is also a
#: valid ``StrictIdentifier``, so it can be recorded verbatim in the bundle's
#: ``model_response`` rather than being laundered into a stop the provider never
#: sent.
_UNSPECIFIED_STOP = "unspecified"


class ProviderStreamAbortedError(RuntimeError):
    """The stream ended abnormally without producing any usable content.

    Deliberately NOT a :class:`DecisionRejected`. That type means the provider
    answered, was billed, and the MODEL's reply failed strict parsing -- which
    the runner corrects by re-prompting, because a model that emitted bad JSON
    can emit good JSON when told what was wrong. None of that holds here: a
    refusal (or a mid-stream provider error) produced no reply at all, so there
    is nothing for the model to correct and a correction prompt asks it to fix
    text it never wrote.

    Raising a plain exception puts this on the protocol's provider path
    (``EpisodeModelClient``: "raising anything else is the contract for an
    unrecoverable provider failure"), so the runner seals the episode unscored
    as ``category="provider"`` / ``reason="infrastructure_failure"`` instead of
    ``model_failure``. That attribution is the point: the agent under test never
    got the chance to act, so charging it a model failure would fold a provider
    outage into the agent's number.

    Observed, and the reason this exists: three consecutive ``refusal`` ends
    (one billing zero tokens both ways) had their ``error`` dropped by
    ``_stream``, so empty text reached ``parse_decision`` and the episode's
    evidence recorded "decision is not valid JSON: Unterminated string" against
    a model that had emitted no bytes. The diagnostic named the wrong cause, so
    the failure was not diagnosable from the bundle at all.

    The message quotes the provider's own words UNBOUNDED and UNSCANNED on
    purpose, and this class cannot promise what happens to them: it holds no
    ``RedactionSet``, so the scan-then-bound ordering that keeps the prose safe
    is enforced one layer away, by the raise site's handler in
    ``episode._decide_once``, which renders it through
    ``_diagnostic(error, self._redactions)`` -- the scan-before-truncate
    contract documented on ``_diagnostic`` itself. Cutting the prose HERE would
    invert that order, severing a canary and letting the surviving prefix pass
    the later scan. Anyone changing where this exception is caught owns keeping
    that call redaction-carrying: most ``_diagnostic`` sites in ``episode.py``
    pass an explicit ``None`` (in-process renderings that never reach
    evidence), so "a handler exists" is not by itself the guarantee.

    The billing fields mirror :class:`DecisionRejected`'s. A refused turn is
    still a BILLED turn -- the provider read the whole prompt before deciding
    to refuse, and the motivating case carried a 48k input -- so the runner
    writes this attempt's request/response/usage triple from these fields
    before sealing. Without them a refused episode reported zero spend as a
    MEASURED figure, which is a false claim inside evidence whose only purpose
    is honest accounting.
    """

    def __init__(
        self,
        message: str,
        *,
        route: RouteIdentity | None = None,
        usage: ModelUsage | None = None,
        cost_micros: int = 0,
        stop_reason: str = _UNSPECIFIED_STOP,
        provider_request_id: str = "unknown",
        tool_call_count: int = 0,
        prompt_cache_key: str | None = None,
        context_tokens: int | None = None,
        compaction: CompactionRecord | None = None,
    ) -> None:
        super().__init__(message)
        # ``route`` may be None for the same reason it may be on a rejection: a
        # stream that aborted need not have reported what it served, and the
        # runner falls back to the requested route rather than inventing one.
        self.route = route
        self.usage = usage or ModelUsage()
        self.cost_micros = cost_micros
        self.stop_reason = stop_reason
        self.provider_request_id = provider_request_id
        self.tool_call_count = tool_call_count
        self.prompt_cache_key = prompt_cache_key
        self.context_tokens = context_tokens
        # A compaction on the way to this request already happened and was
        # billed; it must still be declared before the triple or the verifier
        # sees a compaction with no request to attach it to.
        self.compaction = compaction


class ContextUnrecoverableError(ValueError):
    """The context cannot be made to fit the window, so the request must not
    be sent.

    Raised only when a threshold pass has pruned, summarised, and shed every
    stale observation it may and the rebuilt prefix still exceeds the engine's
    recovery band — or when there is nothing to shed (a frameless benchmark).
    The runner records it as a harness (adapter) error and seals the episode
    UNSCORED: a context the harness can no longer build is the harness's
    limit, not an agent outcome, and scoring the partial would fold a harness
    defect into the agent's number. A scored truncation is not representable
    here — the last executed step's event is already written and the
    verifier's one-step-per-batch rule forbids a corrective re-write.
    """


class _ContextBuilder:
    """The append-only message history one episode sends to the model.

    Holds ``_messages`` (every message already sent, in order) plus two
    cursors: how many turns have had their OBSERVATION rendered as a user
    message, and which turns have had their BATCH rendered as the assistant
    message that follows it. Each :meth:`append_new_turns` appends only what
    is new. Once a message is in ``_messages`` it is never edited -- the only
    operation that replaces the list wholesale is :meth:`replace`, which a
    compaction pass drives.
    """

    def __init__(
        self,
        *,
        artifact_root: Path,
        keep_recent_frames: int,
        rebuild_every_frames: int,
    ) -> None:
        if keep_recent_frames < 0:
            raise ValueError("keep_recent_frames must be non-negative")
        if rebuild_every_frames < 1:
            raise ValueError("rebuild_every_frames must be positive")
        self._artifact_root = artifact_root
        self._keep_recent_frames = keep_recent_frames
        self._rebuild_every_frames = rebuild_every_frames
        self._messages: list[Message] = []
        self._rendered_turns = 0
        self._closed_turns: set[int] = set()
        self._previous_text: str | None = None

    @property
    def messages(self) -> list[Message]:
        return self._messages

    def append_new_turns(self, history: Sequence[EpisodeTurn]) -> None:
        """Render every turn not yet in the history, in order.

        A turn is rendered in its FINAL form the moment it is appended and is
        not touched again. Turn ``i``'s batch is unknown when its observation
        is sent (the model has not answered yet), so it is appended as the
        assistant message when the NEXT call arrives -- which is exactly the
        model replaying its own prior decision, verbatim, and is what makes
        the prefix append-only rather than re-rendered.
        """
        from local_operator.harness.types import Message, TextContent

        for index, turn in enumerate(history):
            if index >= self._rendered_turns:
                previous = history[index - 1] if index > 0 else None
                self._messages.append(self._render_observation(turn, previous))
                self._rendered_turns = index + 1
            if turn.batch is not None and index not in self._closed_turns:
                self._messages.append(
                    Message(
                        role="assistant",
                        content=[
                            TextContent(
                                text=(
                                    turn.public_reply
                                    if turn.public_reply is not None
                                    else turn.batch.to_canonical_json().decode("utf-8")
                                )
                            )
                        ],
                    )
                )
                self._closed_turns.add(index)

    def _render_observation(self, turn: EpisodeTurn, previous: EpisodeTurn | None) -> Message:
        from local_operator.harness.types import ImageContent, Message, TextContent

        observation = turn.observation
        text = observation.text
        # The adapter owns observation text and the runner never rewrites it;
        # the one dedup that is append-only-safe is choosing how to render the
        # message being appended, so a byte-identical repeat is sent as a
        # marker rather than as the same paragraph again.
        #
        # ABSENT text is not UNCHANGED text. Only compare when the adapter
        # actually published text on both turns: an adapter that publishes the
        # goal once and screenshots thereafter (OSWorld) has ``text=None`` from
        # step 1 on, and treating that as "unchanged" asserted that the screen
        # had not moved on every turn of a real episode -- the exact signal the
        # model needs to detect a no-op click, inverted into noise.
        if text is None:
            rendered_text = NO_TEXTUAL_STATE
        elif self._previous_text is not None and text == self._previous_text:
            rendered_text = UNCHANGED_OBSERVATION
        else:
            rendered_text = text
        self._previous_text = text
        lines = [
            f"Step: {observation.sequence}",
            f"Observation ID: {observation.observation_id}",
        ]
        if previous is None:
            # Stated once, on the first observation, where it stays in the
            # cached prefix; repeating it per turn would be tokens spent on a
            # fact the model already has.
            lines.insert(0, f"Task: {observation.task_id}")
        if previous is not None and previous.ask_answer is not None:
            # The answer to the previous turn's ask_user reaches the model as
            # part of the observation that followed it -- "the state after
            # that answer was delivered", as the system prompt promises.
            lines.append(f"Answer from the user: {previous.ask_answer}")
        lines.append(f"Frames: {_frames_line(observation)}")
        if previous is not None and _frames_identical(previous.observation, observation):
            # Stated on the observation itself rather than left for the model
            # to infer by diffing two base64 image blocks, which no model does
            # reliably. Digests are compared, not pixels: the frames are
            # content-addressed already, so this costs nothing.
            lines.append(UNCHANGED_FRAMES_NOTE)
        lines.extend(["", rendered_text])
        content: list[Any] = [TextContent(text="\n".join(lines))]
        for frame in observation.frames:
            # Bytes come through the SAME reader the runner verifies frames
            # with (O_NOFOLLOW, size, digest): a frame the runner would refuse
            # to publish is a frame the model must not be shown, and a second
            # reader here would be a second place for that check to drift.
            data = verify_artifact(self._artifact_root, frame.artifact)
            content.append(
                ImageContent(
                    data=base64.b64encode(data).decode("ascii"),
                    mime_type=frame.artifact.media_type,
                )
            )
        return Message(role="user", content=content)

    def append_rejection(self, reply: str, diagnostic: str) -> None:
        """Fold a rejected reply into the history so the next call corrects it.

        Appended, never substituted: the bad reply goes in as the assistant's
        own words and the diagnostic as the user turn that follows, which is
        the only append-only way to tell the model what was wrong. The next
        ``decide`` for the same observation then sends
        ``user(observation) / assistant(bad) / user(rejection)`` and, once it
        answers well, the runner's close of the turn appends that good batch
        after the rejection -- a conversation that reads exactly as it
        happened. The rendered/closed cursors are indices into the TURN
        history, not into ``_messages``, so the extra pair does not disturb
        them.
        """
        from local_operator.harness.types import Message, TextContent

        shown = reply
        if len(shown) > MAX_REJECTED_REPLY_CHARS:
            shown = shown[:MAX_REJECTED_REPLY_CHARS] + "\n[... reply truncated]"
        # A provider refuses an empty assistant message; a reply that was all
        # whitespace is replayed as a marker so the pair stays well-formed.
        self._messages.append(
            Message(role="assistant", content=[TextContent(text=shown or "(empty reply)")])
        )
        self._messages.append(Message(role="user", content=[TextContent(text=diagnostic)]))

    def frames_in_history(self) -> int:
        from local_operator.compaction.pruning import count_frame_messages

        return count_frame_messages(self._messages)

    def rebuild_due(self) -> bool:
        """Whether the frame budget says it is time to rebuild the prefix.

        THE cache/frames tension, settled: the model should see only the
        newest ``keep_recent_frames`` frames, but a provider's prompt cache is
        keyed on the exact bytes of the sent prefix, and the old pilot's direct
        API probe measured that ANY rewrite of an already-sent message costs
        the whole entry (0% hits, $262 per run with per-turn pruning). Pruning
        the frame that just fell out of the window every turn is therefore the
        worst possible policy. The coherent form is: N appends, all cached,
        then ONE rebuild that prunes every stale frame at once and accepts one
        miss. ``rebuild_every_frames`` is that N; at the default 8 the
        steady-state hit rate is about 7 of every 8 turns.
        """
        return self.frames_in_history() > self._keep_recent_frames + self._rebuild_every_frames

    def replace(self, messages: Sequence[Message]) -> None:
        self._messages = list(messages)


class ProviderModelClient:
    """Drives a real provider through the session stream function.

    The stream function already owns credential resolution and retry, so a
    failure that reaches this class is terminal and is allowed to propagate:
    the runner records it as a provider error and finalizes the episode
    unscored on a still-live session.

    One client serves one episode: it owns that episode's append-only message
    history (``_ContextBuilder``) and the compaction settings that govern its
    rebuilds. ``artifact_root`` is where the adapter published frames; the
    runner verifies frames from the same root.
    """

    def __init__(
        self,
        stream_fn: Any,
        *,
        route: RouteIdentity,
        model_spec: Any,
        artifact_root: Path,
        system_prompt: str = _SYSTEM_PROMPT,
        compaction: "CompactionSettings | None" = None,
        keep_recent_frames: int = DEFAULT_KEEP_RECENT_FRAMES,
        rebuild_every_frames: int = DEFAULT_REBUILD_EVERY_FRAMES,
        prompt_cache_key: str | None = None,
    ) -> None:
        from local_operator.compaction.thresholds import CompactionSettings

        self._stream_fn = stream_fn
        self._route = route
        self._model_spec = model_spec
        self._system_prompt = system_prompt
        self._prompt_cache_key = prompt_cache_key
        base = compaction or CompactionSettings()
        # The episode's frame policy is authoritative over whatever the
        # settings object carried: this client IS a screen-driving surface,
        # so the opt-in is made here, once, for the whole episode.
        self._compaction = base.model_copy(update={"keep_recent_frames": keep_recent_frames})
        self._context = _ContextBuilder(
            artifact_root=artifact_root,
            keep_recent_frames=keep_recent_frames,
            rebuild_every_frames=rebuild_every_frames,
        )
        self._last_provider_context_tokens: int | None = None
        self._last_request_ms = _now_ms()

    @property
    def model_reply_metadata(self) -> dict[str, Any]:
        # Optional client capability: scripted/historic clients must not claim
        # a prompt contract they never used. The runner stays provider-free.
        return public_reply_contract()

    async def decide(
        self,
        observation: Observation,
        history: Sequence[EpisodeTurn],
        *,
        action_surface: ActionSurface = LEGACY_ACTION_SURFACE,
    ) -> ModelDecision:
        from local_operator.harness.types import ChatRequest

        self._context.append_new_turns(history)
        compaction, extra_usage, extra_cost = await self._maybe_compact()

        # The system prompt rides in ``system_blocks`` rather than as a message
        # so the provider can place a cache breakpoint after this stable block;
        # every episode step repeats it verbatim.
        messages = self._context.messages
        request = ChatRequest(
            model=self._model_spec,
            system_blocks=[
                (
                    build_system_prompt(action_surface)
                    if self._system_prompt == _SYSTEM_PROMPT
                    else self._system_prompt + "\n\n" + build_system_prompt(action_surface)
                )
            ],
            messages=list(messages),
            tool_choice="none",
            prompt_cache_key=self._prompt_cache_key,
        )
        (
            text,
            usage,
            cost_micros,
            stop_reason,
            provider_request_id,
            context_tokens,
            stream_error,
        ) = await self._stream(request)
        self._last_request_ms = _now_ms()
        if context_tokens is not None:
            # A figure reported FOR the request just sent is measured against
            # the prefix that is about to be reused verbatim, so it stays
            # exact through plain appends until a rebuild replaces the list.
            self._last_provider_context_tokens = context_tokens
        # The summary call was a billed provider call on the way to this
        # decision. It is folded into THIS decision's figures rather than
        # reported separately so the bundle's usage events remain the whole
        # bill and the counters a pure sum of them.
        if extra_usage is not None:
            usage = _add_usage(usage, extra_usage)
            cost_micros += extra_cost
        # Checked BEFORE parsing, because parsing an empty string produces a
        # confident and completely wrong diagnosis. ``parse_decision`` reports
        # "decision is not valid JSON", the runner turns that into a correction
        # prompt asking the model to fix its JSON, and the model -- which said
        # nothing -- is re-prompted until the retry bound is spent. The episode
        # then seals as a MODEL failure for a provider's refusal.
        #
        # Both halves of the condition are required. ``error`` alone misses a
        # wire client that ends abnormally without composing prose; an abnormal
        # ``stop_reason`` alone would swallow the recoverable case where the
        # provider refused only AFTER streaming a parseable batch -- that text
        # is real model output and stays on the correctable rejection path.
        if (stream_error or stop_reason not in _NORMAL_CONTENT_STOPS) and not text.strip():
            raise ProviderStreamAbortedError(
                stream_error or f"provider ended the stream as '{stop_reason}' with no content",
                # The same billing provenance a rejection carries, and for the
                # same reason: the request was sent and read, so its usage is
                # part of the episode's bill whether or not a reply came back.
                route=self._route,
                usage=usage,
                cost_micros=cost_micros,
                stop_reason=stop_reason,
                provider_request_id=provider_request_id,
                prompt_cache_key=self._prompt_cache_key,
                context_tokens=_estimate_context(messages),
                compaction=compaction,
            )
        try:
            return parse_decision(
                text.strip(),
                observation,
                route=self._route,
                usage=usage,
                cost_micros=cost_micros,
                stop_reason=stop_reason,
                provider_request_id=provider_request_id,
                prompt_cache_key=self._prompt_cache_key,
                context_tokens=_estimate_context(messages),
                compaction=compaction,
                action_surface=action_surface,
            )
        except DecisionParseError as error:
            # The call happened and was billed; only the reply is unusable.
            # Fold the rejection into the history NOW, before the runner
            # decides whether to re-call, so a re-call for the same observation
            # is corrective by construction. The billing provenance rides on
            # the exception so the runner can write this attempt's triple.
            diagnostic = _rejection_prompt(str(error), observation)
            # Invalid notes are not factual memory. For envelope failures do
            # not quote their unvalidated (possibly secret) text on retries or
            # in evidence. Classification cannot rely on successful decoding
            # (a truncated reply fails) or literal keys (JSON Unicode escapes
            # bypass the substring test), so the reserved-key scan is
            # escape-aware and fails closed (F1, review round 1). Raw legacy
            # rejection text is retained only where no reserved key appears.
            shown = text
            try:
                rejected_value, _ = json.JSONDecoder().raw_decode(text.lstrip())
            except (ValueError, RecursionError):
                rejected_value = None
            if is_public_reply(rejected_value) or looks_like_public_reply(text):
                shown = REJECTED_PUBLIC_REPLY
            self._context.append_rejection(shown, diagnostic)
            raise DecisionRejected(
                diagnostic,
                # Truncated with the same bound the context replay uses: a
                # runaway reply must not be able to inflate the bundle either.
                reply=shown[:MAX_REJECTED_REPLY_CHARS],
                route=self._route,
                usage=usage,
                cost_micros=cost_micros,
                stop_reason=stop_reason,
                provider_request_id=provider_request_id,
                prompt_cache_key=self._prompt_cache_key,
                context_tokens=_estimate_context(messages),
                compaction=compaction,
            ) from error

    async def _maybe_compact(self) -> tuple[CompactionRecord | None, ModelUsage | None, int]:
        """Run one compaction pass when the frame budget or the token threshold says so.

        Two triggers, one pass. The frame budget (``rebuild_due``) is the usual
        one for a screen-driving episode; the token threshold is the ordinary
        session's trigger and applies here too because a long text-heavy episode can
        fill the window without ever tripping the frame budget. Either way the pass
        rebuilds the whole prefix ONCE, which is the only rewrite this client
        ever makes to sent messages.

        The token trigger is a SAFETY BOUND and is never silenced (review round
        2, M2): the only way to avoid a rebuild every turn is to make the pass
        EFFECTIVE, not to ignore the trigger. After a threshold pass that did
        not clear the engine's recovery band, stale observation turns are shed
        oldest-first (``_shed_stale_frames``) until the rebuilt prefix fits the
        reserve; only when even that cannot get under the line does the client
        raise :class:`ContextUnrecoverableError`, which the runner records as a
        harness error and seals unscored instead of sending a request the
        provider will reject. The earlier stall latch is removed
        because it let the priced context grow past the window between
        rebuilds — a provider rejection, strictly worse than a cache miss.
        """
        from local_operator.compaction.pass_ import run_compaction_pass
        from local_operator.compaction.thresholds import (
            compaction_context_tokens,
            resolve_threshold_tokens,
            should_compact,
        )

        messages = self._context.messages
        if not messages:
            return None, None, 0
        frame_due = self._context.rebuild_due()
        # The trigger judges what the provider will BILL for this request, not
        # what the local ruler counts: the provider's last figure described the
        # previous prefix (one observation ago), and the ruler prices every
        # frame at a flat 1,200 while vision providers bill per visual token.
        # Both under-read the request about to be sent by exactly the frames
        # appended since, so the local estimate is corrected by the engine's
        # per-frame addend (the same correction the shed band uses) before the
        # single resolver judges it. Without this a 24k window on a 5,000/frame
        # model sent a 1.01x request one turn before the trigger fired.
        tokens_before = compaction_context_tokens(
            self._last_provider_context_tokens, self._priced_estimate(messages)
        )
        window = int(getattr(self._model_spec, "context_window", 0) or 0)
        threshold_due = should_compact(tokens_before, window, self._compaction)
        if not (frame_due or threshold_due):
            return None, None, 0
        threshold = resolve_threshold_tokens(window, self._compaction) if threshold_due else None

        summary_usage: ModelUsage | None = None
        summary_cost = 0

        async def summarize(prompt: str) -> str:
            nonlocal summary_usage, summary_cost
            # TODO(F3, review round 2): this path discards ``_err`` and
            # ``_stop``, so a compaction summary call that the provider REFUSED
            # returns empty text and is folded into the context as a legitimate
            # (if useless) summary, rather than being attributed to the
            # provider the way the decision call now is. Pre-existing and
            # deliberately out of scope here; fixing it needs a decision about
            # whether an unsummarizable context is a provider failure or a
            # harness one (``ContextUnrecoverableError``), which is a different
            # question from this change's.
            text, usage, cost, _stop, _rid, _ctx, _err = await self._stream(
                self._summary_request(prompt)
            )
            summary_usage = usage
            summary_cost = cost
            return text

        before = len(messages)
        result = await run_compaction_pass(
            messages,
            model=self._model_spec,
            settings=self._compaction,
            summarize=summarize,
            now_ms=_now_ms(),
            last_activity_ms=self._last_request_ms,
            provider_context_tokens=self._last_provider_context_tokens,
            # A frame-budget rebuild lets the pass judge its own gate (on a small
            # context it prunes its frames and refuses the summary as
            # below-threshold, the common and correct outcome). A
            # threshold-triggered pass was ALREADY judged over the line by the
            # same resolver, so the gate is not re-asked after the prune:
            # re-asking let the prune alone slip just under the line, refuse
            # the summary, and re-fire the pass on the very next turn (review
            # round 1, m2).
            respect_threshold=not threshold_due,
        )
        # The pass returns the pruned list even when it refused to summarize; either
        # way it is the prefix to send from now on.
        self._context.replace(result.messages)
        # The provider's last reported context size described the prefix that was just
        # replaced, and ``compaction_context_tokens`` is max(provider, local), so a
        # stale figure would keep re-firing the pass against the NEW prefix on every
        # turn. Judge the rebuilt prefix on the local estimate until the provider
        # reports a fresh figure for it (the next request's usage reports
        # ``context_tokens``), which is one request away at most.
        self._last_provider_context_tokens = None

        shed = 0
        if threshold is not None:
            # A threshold pass must FIT, not merely run. Re-asking the trigger
            # after a pass that left the prefix over the engine's reserve band
            # would fire every turn for no headroom; silencing it would let the
            # priced context grow past the window. The effective answer is to
            # shed stale observation turns (oldest first) until it fits, and to
            # refuse the request when even that cannot help — see the class
            # docstring for why a silent trigger is never the alternative.
            fits, shed = self._enforce_threshold_fit(threshold, band=int(0.8 * threshold))
            if not fits:
                from local_operator.compaction.pruning import count_stale_observations

                # Says how much was sheddable: "after shedding" with nothing
                # shed was how a dead shed went unnoticed (review round 3).
                raise ContextUnrecoverableError(
                    "context cannot fit the window: the rebuilt prefix exceeds the "
                    "compaction threshold "
                    f"({self._priced_estimate(self._context.messages)} priced tokens "
                    f"against a band of {int(0.8 * threshold)}) with "
                    f"{count_stale_observations(self._context.messages)} stale "
                    "observation turn(s) that shedding could not fit under it; the "
                    "episode ends as a harness error rather than send a rejected request"
                )
        # A shed-only rebuild is still a rebuild: when the pass itself refused
        # (``nothing-to-summarize`` — the kept window was the whole tail) but
        # the shed removed turns, previously sent messages vanish from the next
        # request, and a ``context_compaction`` event is the only honest trace
        # of that (review round 4, m8). Returning nothing here would hide it
        # behind a bare ``message_count`` drop.
        if shed == 0 and result.frames_dropped == 0 and not result.ran and not result.pruned:
            return None, None, 0
        record = CompactionRecord(
            strategy=result.strategy or "prune",
            # The pass measures its own "before" AFTER its prune step (that is
            # the figure its trigger judges); the bundle wants the size of the
            # context the model last saw, so the pre-pass estimate is recorded.
            tokens_before=tokens_before,
            # After a shed the pass's own figure describes a prefix that no
            # longer exists; measure what is actually being sent.
            tokens_after=(
                _estimate_context(self._context.messages) if shed else result.tokens_after
            ),
            frames_dropped=result.frames_dropped,
            messages_before=before,
            messages_after=len(self._context.messages),
            summary_text=result.summary_text,
        )
        return record, summary_usage, summary_cost

    def _priced_estimate(self, messages: Sequence[Any]) -> int:
        """The local estimate corrected to what the provider will bill for frames.

        The ruler counts every image at a flat ``IMAGE_TOKEN_ESTIMATE`` (1,200);
        vision providers bill per visual token (Anthropic ~5,000 for a 1932px
        frame). The engine ships the family formula (``frame_token_estimate_for``)
        and the session prices a compaction's residual with it (its archive frame
        correction), so the same ADDEND per frame is applied here — never a
        multiplier on the whole context (review round 2, m4: a multiplier learned
        before a rebuild misprices the residual after it).
        """
        from local_operator.compaction.snapcompact import frame_token_estimate_for
        from local_operator.compaction.tokens import (
            IMAGE_TOKEN_ESTIMATE,
            estimate_messages_tokens,
        )

        frames = sum(
            1 for message in messages for block in message.content if block.type == "image"
        )
        per_frame = frame_token_estimate_for(self._model_spec.provider, self._model_spec.model_id)
        addend = max(0, per_frame - IMAGE_TOKEN_ESTIMATE)
        return estimate_messages_tokens(messages) + frames * addend

    def _enforce_threshold_fit(self, threshold: int, *, band: int) -> tuple[bool, int]:
        """After a threshold pass, shed stale observations until the prefix fits.

        Returns ``(fits, turns_shed)``. The pass above has already pruned and
        summarised everything it could; what remains is how many of the OLD
        observation turns to keep. A shed removes a stale turn (its
        observation — framed, or pruned to a notice — plus its assistant
        reply) from the front of the kept tail, never the current
        observation, never past a compaction marker in the tail, and is inert
        on a prefix with no stale turns (a text-only benchmark has nothing to
        shed — its window problem is the summary's, and deletion would corrupt
        the episode). ``fits`` is False only when no shed could get under
        ``band``.

        The band is judged on the PROVIDER's price, not the local ruler: the
        ruler counts a frame at a flat 1,200 tokens while vision providers bill
        per visual token (Anthropic ~5,000 for a 1932px frame), so a residual
        that fits locally can still be rejected on the wire. The session prices
        a compaction's residual the same way (its archive frame correction), and
        the engine ships the formula (``frame_token_estimate_for``), so the
        correction is ``frames x (per_frame - IMAGE_TOKEN_ESTIMATE)`` added to
        the local estimate — an ADDEND per frame, never a multiplier on the
        whole context (review round 2, m4).
        """
        if self._priced_estimate(self._context.messages) <= band:
            return True, 0
        return self._shed_stale_turns(band)

    def _shed_stale_turns(self, band: int) -> tuple[bool, int]:
        """Shed the oldest stale turns, one at a time, until the priced prefix
        fits ``band``; ``(False, 0)`` when no amount of shedding can.

        Walks ``limit`` DOWN from one below the current stale count: asking
        the engine for the count it already has removes nothing and must not
        be read as "nothing left to shed" — that reading is exactly how this
        method was dead code in review round 3 (M3). A step that removes
        nothing after the first decrement is the genuine end: the tail is
        down to the current observation, or the next turn is not stale.
        """
        from local_operator.compaction.pruning import (
            count_stale_observations,
            shed_stale_frames,
        )

        messages = self._context.messages
        stale = count_stale_observations(messages)
        if stale == 0:
            return False, 0
        best: list[Any] | None = None
        shed = 0
        limit = stale - 1
        while limit >= 0:
            candidate, removed = shed_stale_frames(messages, limit=limit)
            if removed == 0:
                break
            if self._priced_estimate(candidate) <= band and candidate[-1] is messages[-1]:
                best = candidate
                shed = removed
                break
            limit -= 1
        if best is None:
            return False, 0
        self._context.replace(best)
        return True, shed

    def _summary_request(self, prompt: str) -> Any:
        from local_operator.compaction.api import SUMMARIZATION_SYSTEM_PROMPT
        from local_operator.harness.types import ChatRequest, Message

        # Mirrors ``Session._one_shot_complete``: a summary is collected whole
        # before anything reads it, so a stalled stream may be retried
        # (``replayable``). No cache key is set HERE; the stream fn stamps the
        # episode's ``lop-eval-<id>`` on any request that carries none
        # (``configure.py``, ``_cache_lineage_id``), which is the same
        # treatment the session's one-shot summary gets. The key is a routing
        # hint, and the summary prompt shares no prefix with the history, so
        # it simply misses -- harmless, and consistent with ordinary sessions.
        return ChatRequest(
            model=self._model_spec,
            system_blocks=[SUMMARIZATION_SYSTEM_PROMPT],
            messages=[Message.user(prompt)],
            tools=[],
            tool_choice="none",
            replayable=True,
        )

    async def _stream(
        self, request: Any
    ) -> tuple[str, ModelUsage, int, str, str, int | None, str | None]:
        text = ""
        usage = ModelUsage()
        cost_micros = 0
        # A stream that never emits an ``end`` event never told us how it
        # ended, so it starts abnormal and only a real marker makes it normal.
        stop_reason = _UNSPECIFIED_STOP
        provider_request_id = "unknown"
        context_tokens: int | None = None
        stream_error: str | None = None
        async for event in self._stream_fn(request, None):
            if event.type == "text_delta":
                text += event.delta
            elif event.type == "usage":
                usage, cost_micros = _usage_from(event.usage)
                context_tokens = _context_tokens_from(event.usage)
            elif event.type == "end":
                # NOT ``or "stop"``. An empty marker is an ABSENT marker, and
                # normalizing it to a normal content stop would let a stream
                # that said nothing about how it ended take the parse path --
                # the exact misdiagnosis the allow-list below exists to stop.
                stop_reason = event.stop_reason or _UNSPECIFIED_STOP
                if event.usage is not None:
                    usage, cost_micros = _usage_from(event.usage)
                    context_tokens = _context_tokens_from(event.usage)
                # The provider's own request id is the only handle that ties a
                # bundle's model_response back to the provider's records, so it
                # is worth carrying when the wire client reports one.
                provider_request_id = _provider_request_id(event.provider_payload)
                # The wire clients are the ONLY layer where the provider's
                # actual terminal marker (``content_filter``, ``refusal``,
                # ``SAFETY``…) is still visible; downstream sees only the
                # normalized stop. Dropping this field is what left a refused
                # episode's evidence blaming the model's JSON.
                stream_error = event.error
        return (
            text,
            usage,
            cost_micros,
            stop_reason,
            provider_request_id,
            context_tokens,
            stream_error,
        )


def _frames_line(observation: Observation) -> str:
    """``id (WxH)`` for every frame, or ``none`` when the observation has none.

    This line is the frame-id CONTRACT made visible. The protocol lets an
    adapter name its frames however it likes (OSWorld publishes ``screen``;
    the test adapters publish ``frame-<n>``), and ``ActionBatch.validate_for``
    refuses any id the observation does not carry -- so a model that was never
    told the ids can only guess. The first paid episode guessed ``"1"``. The
    geometry is the model-visible size because that is the coordinate space
    the model's x/y are validated against.
    """

    if not observation.frames:
        return "none"
    return ", ".join(
        f"{frame.frame_id} ({frame.geometry.model_visible.width}x"
        f"{frame.geometry.model_visible.height})"
        for frame in observation.frames
    )


def _frames_identical(previous: Observation, current: Observation) -> bool:
    """Whether two observations carry exactly the same frame bytes.

    Compared by the frames' content-addressed digests, in order, so this is a
    statement about the BYTES rather than a perceptual judgement the runner is
    not entitled to make. An observation with no frames is never "identical":
    a frameless benchmark has no screen to be unchanged, and claiming
    otherwise would put a false no-op note on every one of its turns.
    """

    if not current.frames or len(previous.frames) != len(current.frames):
        return False
    return all(
        before.artifact.sha256 == after.artifact.sha256
        for before, after in zip(previous.frames, current.frames)
    )


def _rejection_prompt(reason: str, observation: Observation) -> str:
    """The user turn that asks for a corrected reply.

    Restates the observation id and the frame ids because those are the two
    facts a rejected reply most often got wrong, and because the correction
    is the LAST message in the request: a model reads it more reliably than
    the same facts several messages up.
    """

    return (
        f"Your previous reply was rejected: {reason}\n"
        "Nothing was executed. Reply again for this same observation "
        f"(Observation ID: {observation.observation_id}; Frames: "
        f"{_frames_line(observation)}) with a corrected JSON batch and nothing else."
    )


def create_provider_model_client(
    *,
    auth_store: Any,
    settings: Mapping[str, Any] | None,
    route: RouteIdentity,
    model_spec: Any,
    artifact_root: Path,
    episode_id: str,
    fallback_policy: str = "forbid",
    keep_recent_frames: int = DEFAULT_KEEP_RECENT_FRAMES,
    rebuild_every_frames: int = DEFAULT_REBUILD_EVERY_FRAMES,
    compaction: "CompactionSettings | None" = None,
) -> ProviderModelClient:
    """Build a provider-backed client from the harness's own stream function.

    Importing ``configure`` here rather than at module scope keeps the cost off
    anyone who only imports the runner contracts, and keeps the evaluation
    package's import-inertness property intact.

    The session id handed to the stream function is ALSO the provider cache
    key (``SessionStreamFn`` stamps ``prompt_cache_key`` from it), so it is
    minted per episode: ``lop-eval-<episode_id>``. One key for every episode
    would make unrelated tasks route as one prefix, which is the bug the old
    pilot shipped once.

    ``fallback_policy`` is the manifest's declared policy and it is HONOURED
    here, not merely recorded: under ``forbid`` the stream function's model
    fallback is switched off through the same ``retry.modelFallback`` setting
    an operator would use, so a provider outage fails the episode as a
    provider error instead of silently serving a different model. A served
    route that differs from the pinned one still seals ``route_changed`` (the
    runner compares every served route), so this is belt and braces: the
    fallback is sealed off, and a fallback that somehow happened is labelled.
    """

    from local_operator.model.configure import create_stream_fn

    effective: dict[str, Any] = dict(settings or {})
    if fallback_policy == "forbid":
        retry = dict(effective.get("retry") or {})
        retry["modelFallback"] = False
        effective["retry"] = retry
    session_id = f"lop-eval-{episode_id}"
    return ProviderModelClient(
        create_stream_fn(auth_store, effective, session_id=session_id),
        route=route,
        model_spec=model_spec,
        artifact_root=artifact_root,
        compaction=compaction,
        keep_recent_frames=keep_recent_frames,
        rebuild_every_frames=rebuild_every_frames,
        prompt_cache_key=session_id,
    )


def _add_usage(first: ModelUsage, second: ModelUsage) -> ModelUsage:
    return ModelUsage(
        input_tokens=first.input_tokens + second.input_tokens,
        output_tokens=first.output_tokens + second.output_tokens,
        reasoning_tokens=first.reasoning_tokens + second.reasoning_tokens,
        cache_read_tokens=first.cache_read_tokens + second.cache_read_tokens,
        cache_write_tokens=first.cache_write_tokens + second.cache_write_tokens,
    )


def _estimate_context(messages: Sequence[Any]) -> int:
    from local_operator.compaction.tokens import estimate_messages_tokens

    return estimate_messages_tokens(messages)


def _context_tokens_from(usage: Any) -> int | None:
    value = getattr(usage, "context_tokens", None)
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _provider_request_id(provider_payload: Any) -> str:
    """Extract the provider's request id, falling back to a valid placeholder.

    ``StrictIdentifier`` forbids most punctuation, so a provider id that does
    not fit the pattern is dropped rather than allowed to fail validation on a
    path that is only recording provenance.
    """

    if not isinstance(provider_payload, Mapping):
        return "unknown"
    raw = provider_payload.get("id")
    if not isinstance(raw, str) or not raw:
        return "unknown"
    # An over-length id is NOT truncated: a shortened id is still a valid
    # StrictIdentifier, so it would be recorded as provenance while matching
    # nothing in the provider's records -- a silently wrong handle is worse than
    # an honestly absent one, which is this function's whole philosophy.
    if len(raw) > 128 or not _IDENTIFIER.fullmatch(raw):
        return "unknown"
    return raw


def _usage_from(usage: Any) -> tuple[ModelUsage, int]:
    """Copy provider counts and cost, clamped to the non-negative evidence range.

    ``reasoning_tokens`` is a SUBSET of ``output_tokens`` in the harness's
    accounting, and the evidence payloads treat it the same way, so it is
    carried across unchanged rather than added on top.

    ``usd_cost`` is the provider's OWN billing figure, which the harness treats
    as ground truth over any token-times-rate reconstruction. Dropping it made
    every reconciliation report a free episode. An unreported cost stays 0 here
    because the evidence payload has no "unknown" encoding -- the distinction
    upstream between ``None`` and a real ``0.0`` cannot be represented, and
    inventing a number would be worse than under-reporting one.
    """

    model_usage = ModelUsage(
        input_tokens=max(0, int(usage.input_tokens or 0)),
        output_tokens=max(0, int(usage.output_tokens or 0)),
        reasoning_tokens=max(0, int(usage.reasoning_tokens or 0)),
        cache_read_tokens=max(0, int(usage.cache_read_tokens or 0)),
        cache_write_tokens=max(0, int(usage.cache_write_tokens or 0)),
    )
    cost = getattr(usage, "usd_cost", None)
    cost_micros = 0 if cost is None else max(0, round(float(cost) * 1_000_000))
    return model_usage, cost_micros
