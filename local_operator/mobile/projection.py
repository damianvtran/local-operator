"""Fold a live session into the phone-facing projection.

One class, :class:`ProjectionFold`, fed two ways:

- **events** — the harness ``AgentEvent`` stream (``session.subscribe``),
  folded incrementally so a streaming assistant row updates in place instead
  of repainting history;
- **history** — ``session.history()`` on attach/resume, folded wholesale so a
  phone that opens mid-conversation sees the same transcript the TUI shows.

The fold owns the render semantics the TUI established and the web UI must
match exactly: one row per tool call (state glyph + one-line summary +
diff counts, details behind a tap), notices as quiet system rows, steering
receipts reconciled against the queued count. Putting those semantics here —
server-side, once — is what keeps the phone a pure renderer; the alternative
(ship raw events, fold in TypeScript) is two implementations of the TUI's
contract drifting apart.

The fold is deliberately free of asyncio: it is a plain state machine the
daemon drives from its event callback, so it is testable without a loop and
safe to call from any thread that serializes calls per session.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

from local_operator.compaction.marker import COMPACTION_REFUSED_TYPE
from local_operator.harness.approval import GATE_TIMEOUT_CUSTOM_TYPE
from local_operator.harness.comms import extract_parent_message
from local_operator.harness.message_types import (
    HUB_MESSAGE_TYPE,
    PEER_MESSAGE_MESSAGE_TYPE,
)

# The row DECISIONS both this fold and the TUI's must make identically. They
# live outside both hosts precisely so neither can own them: every divergence
# the convergence review found was a decision one surface made and the other
# did not (docs/design/history-fold-convergence.md §3).
from local_operator.harness.rows import (
    assistant_row_text,
    assistant_stop_notice,
    compaction_refused_notice,
    gate_timeout_notice,
    is_harness_chrome,
    is_harness_notice_row,
    output_limit_call_receipt,
    turn_cut_tool_call,
    user_row_text,
    wake_receipt_headline,
)
from local_operator.harness.types import (
    AgentEndEvent,
    AgentEvent,
    AgentMessage,
    AgentStartEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    CustomMessage,
    ImageContent,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ModelChangeEvent,
    NoticeEvent,
    ReasoningDeltaEvent,
    RetryEndEvent,
    RetryStartEvent,
    SteeringDeliveredEvent,
    SubagentEndEvent,
    SubagentProgressEvent,
    SubagentStartEvent,
    ToolCallComposeEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
)
from local_operator.harness.wake import WAKE_PROMPT_MESSAGE_TYPE
from local_operator.mobile.types import (
    PROJECTION_TRANSCRIPT_LIMIT,
    PendingRequest,
    SessionProjection,
    SubagentRow,
    TodoItem,
    TodoPhase,
    TranscriptEntry,
)

# The PHASE WORDS the working line is dated by, imported rather than restated:
# the fold below matches its own phase against the one the producer folded
# (``FrontendStateStore.activity_phase_clock``), so a rename in either place
# must move together or the match silently stops and every attach-time clock
# goes back to counting from the phone's arrival — the divergence
# ``session/frontend_state.py`` warns about beside its own copies.
from local_operator.session.frontend_state import (
    ACTIVITY_PHASE_COMPOSING,
    ACTIVITY_PHASE_QUEUED,
    ACTIVITY_PHASE_RESPONDING,
    ACTIVITY_PHASE_RUNNING,
    ACTIVITY_PHASE_THINKING,
)

logger = logging.getLogger(__name__)

#: The folded phases an ATTACH may adopt the producer's instant for, by NAME.
#:
#: A phase qualifies only when the producer HOLDS its instant across the events
#: the fold renders that phase for — one zero per phase, restamped only when the
#: phase is ENTERED — so a fold that reads the instant while the producer is
#: already inside the phase is reading the number the producer itself holds.
#:
#: ``responding`` and ``composing`` are those phases: the prose edge is the first
#: non-empty delta of a model call, the dictation edge is a batch's first
#: announcement (and deliberately ONE zero for the batch's whole dictation).
#: ``thinking`` does NOT qualify: the producer re-zeroes it at every provider
#: call (``message_start``), turn boundary and tool end, and every label this
#: fold derives for it comes from one of those very events — so adopting an
#: older instant there would date a phase that has just begun by a previous
#: phase's zero, the exact pairing ``FrontendStateStore.activity_phase_clock``
#: exists to prevent. ``running`` does not qualify either: its finer anchor is
#: the call (step 1) and its folded edge is the batch's FIRST call, so a
#: narrowed label would report a shed sibling's age (the TUI's D9). ``queued``
#: is never folded by the producer, so it can match nothing.
_PHASE_ANCHOR_ADOPTABLE = frozenset({ACTIVITY_PHASE_RESPONDING, ACTIVITY_PHASE_COMPOSING})

#: How much of a tool result's text the expand payload carries. The phone's
#: expanded row is a readable window, not a log file — beyond this the right
#: surface is the terminal.
TOOL_OUTPUT_TAIL_CHARS = 8_000

#: Same bound for the args side of an expanded tool row.
TOOL_ARGS_CHARS = 4_000

#: How much of a subagent's launch prompt the roster row carries on the wire.
#: The list projection is a full repaint pushed ~30x/s and every subagent row
#: rides in it, so an uncapped prompt is a per-repaint tax that scales with
#: roster depth: a power-user session at 80+ subagents put ~385 KB of prompt
#: text into a single frame, pushing it toward the daemon's 1 MB control-frame
#: cap (``daemon._dial`` ``limit=1 << 20``) and risking the same wedge that
#: keeping transcripts off the wire fixed. The row only needs a preview for the
#: sheet's "Parent request" line; the FULL prompt is reachable through the child
#: transcript's launch message (resolved by ``launch_message_id``), which the
#: phone fetches lazily from the /history endpoint. This bound is generous
#: enough to read as the task while staying bounded per row.
SUBAGENT_PROMPT_PREVIEW_CHARS = 1_000

#: Preview bound for a settled child's ``result_text`` on the wire, shared by
#: the live ``SubagentEndEvent`` path, the ``set_subagent_details`` lifecycle
#: merge, and the durable rebuild so no path carries unbounded result text. A
#: 200-char preview is safe here because ``result_text`` is the child's own last
#: assistant message: it appears verbatim in the child transcript, which the
#: phone now fetches in full lazily from
#: ``/api/sessions/{sid}/agents/{job_id}/history`` — so truncating it on the
#: wire loses nothing the reader cannot recover by opening the conversation.
SUBAGENT_OUTCOME_CHARS = 200

#: Failure-tail bound for a settled child's ``error_text`` on the wire. Unlike
#: ``result_text``, ``error_text`` is ``str(exc)`` raised in the PARENT runner
#: (``session.subagent``) and is NEVER appended to the child transcript, so the
#: lazy /history fetch cannot recover it — the wire value is the ONLY copy the
#: phone's Outcome panel ever renders. It is therefore carried generously,
#: matching ``session._ROSTER_ERROR_CAP``, so a provider error or stack-trace
#: tail survives to the surface an operator opens precisely to see "what went
#: wrong". This costs almost nothing on frame size: 2000 chars across the worst
#: ~81-subagent roster is ~150 KB, well under the 1 MB control-frame cap.
SUBAGENT_ERROR_CHARS = 2_000


#: Soft cap for one serialized projection frame, applied by
#: :func:`cap_projection_frame` BEFORE the frame is written to a socket or an
#: SSE stream. The daemon's control-socket StreamReader refuses a line past
#: 1 MB (``limit=1 << 20`` in ``daemon._dial``) and drains it, so an oversized
#: push is a silently dropped repaint; a FLOOD of them (one session measured
#: 48k drops at 14% idle CPU) starves the daemon loop and stalls EVERY other
#: session's load. 700 KB keeps a generous margin under the hard limit for the
#: frame envelope and JSON escaping inflation (a payload of quotes/backslashes
#: can nearly double when escaped).
PROJECTION_FRAME_SOFT_CAP_BYTES = 700_000

#: Minimal bounds the frame cap degrades subagent text to. The cap only fires
#: past the soft cap, so these are preview lengths, not the normal ones: the
#: full prompt/result stay reachable through the child transcript's lazy
#: /history fetch, and a degraded roster row still names the child and its
#: state — the wedge alternative is no repaint at all.
FRAME_CAP_PROMPT_CHARS = 120
FRAME_CAP_RESULT_CHARS = 200
FRAME_CAP_ERROR_CHARS = 400

#: Floor for the transcript tail under the frame cap. Tier 3 halves the tail
#: toward this bound; below it the phone keeps the opening user message plus
#: the newest few rows and pages the rest from /history, which is exactly the
#: scroll contract the 80-row cap already established.
FRAME_CAP_TRANSCRIPT_FLOOR = 16

#: Per-row text bound applied by the LAST tier, once dropping details and
#: trimming the tail have not been enough. A single row can exceed the whole
#: cap on its own — a pasted file or a long tool output — and the operator's
#: real store contains 33 transcript lines over the soft cap, the largest
#: 946,552 B, so this is a reachable case rather than a theoretical one.
#: Truncating the row's TEXT is the last thing to give up because it is the
#: content the reader came for, but a dropped frame shows them nothing at all;
#: the full text remains one /history fetch away.
FRAME_CAP_ENTRY_TEXT_CHARS = 4_000

#: Hard floor the text tier walks down to when even 4,000 chars/row does not
#: fit (many rows, each individually modest). Below this a row is a stub and
#: the frame is structurally as small as this function can make it.
FRAME_CAP_ENTRY_TEXT_FLOOR = 200

#: Bound for one roster todo's text under the frame cap. Roster todos ride the
#: wire by design and are agent-authored, so a deep roster can carry hundreds
#: of unbounded strings; a todo only has to be recognisable in a working line.
FRAME_CAP_TODO_TEXT_CHARS = 120

#: Bounds for a pending card under the frame cap. The card is never dropped —
#: it is the one thing on the phone that blocks a turn — so these are generous
#: enough to answer the question and bounded enough to fit.
FRAME_CAP_PENDING_TITLE_CHARS = 2_000
FRAME_CAP_PENDING_DETAIL_CHARS = 2_000

#: The roster fields that survive everything except the LAST shed tier: each is
#: a pure derivation of ``parent_job_id``, which is why they can be dropped at
#: all. ``peer_ids`` is the one that actually grows the frame — it is every
#: sibling's job id, so a wide fan-out (256 children under one parent) makes
#: every row list the other 255 and the field becomes O(n^2) in roster WIDTH:
#: measured at 1,044,480 B of a 1,254,249-byte frame, 83% of it.
FRAME_CAP_DERIVED_ROSTER_FIELDS = ("peer_ids", "child_ids", "ancestor_ids", "ancestors")

#: What a roster row keeps in the LAST tier, when even the derived graph is not
#: enough. Only identity and the parent edge stay: a viewer can still render one
#: row per child, its label, its lifecycle state, and the hierarchy it belongs
#: to, and the list is its own count. Per-child detail is fetched on demand —
#: except ``error_text``, which exists nowhere else and is lost here (see the
#: tier's own comment in ``cap_projection_frame``).
FRAME_CAP_ROSTER_IDENTITY_FIELDS = ("job_id", "label", "parent_job_id", "status")


def _stated_epoch(value: Any) -> float | None:
    """The instant a producer STATED, or ``None`` when it stated none.

    ``bool`` is excluded deliberately: it is an ``int`` in Python, so a stray
    ``True`` in an epoch field would be read as 1970-01-01 and a live call
    would report itself as 56 years old. Same guard the fold already applies
    to a persisted ``duration_s``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _phase_pair(value: Any) -> tuple[str, float | None]:
    """The ``(phase, instant)`` a probed accessor answered, or an empty pair.

    The read is PROBED, so what comes back is whatever a host chose to return:
    a reduced facade or a stand-in may answer ``None``, a shape of its own, or
    a pair it built wrongly, and unpacking that here raises at the ATTACH — off
    ``RuntimeServer._serve``, whose handler ends the runtime ("session runtime
    loop died"), or into the app's rebind, which swallows it and leaves the
    bridge silently unsubscribed. Both are worse than the answer this returns:
    an empty pair matches no phase, so an unusable accessor withholds the clock
    exactly like a session that has no fold at all.

    ``str(phase or "")`` is why a stand-in's repr cannot match: an object whose
    string is not a phase word is not equal to any phase the fold displays.
    """
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return "", None
    phase, started_at = value
    return str(phase or ""), started_at


def monotonic_from_epoch(epoch: float, *, clock: Callable[[], float] = time.monotonic) -> float:
    """A monotonic instant whose age is ``now - epoch``, for a threaded clock.

    The phone's conversion from the wall-clock stamps that travel on events and
    in the producer's folded state (``ToolExecutionStartEvent.started_at_epoch``,
    ``FrontendStateStore.activity_phase_clock``) into the monotonic clock the
    projection counts on. Conversion is AGE-ONLY and happens ONCE: everything
    that ticks afterwards counts on ``clock()``, so a system-clock adjustment or
    a DST jump after the seed cannot move a counter that is already running.
    The clamp is the same rule for an epoch in the future (a producer whose
    clock is ahead of ours means "age unknown, treat as new" rather than a
    negative elapsed time that would render as a nonsense duration).

    Deliberately a second copy of ``tui/widgets/tool_card.monotonic_from_epoch``
    rather than an import of it: that module is a Textual widget module (it
    imports ``textual.timer`` and the transcript widget) and this fold is loaded
    by the phone daemon and by an owned session's runtime, neither of which
    should pull the TUI's widget tree in to divide one number. The TUI's own
    reader takes the same route the other way — ``tui/session_presentation.py``
    carries the folded-anchor helpers and imports no phone module — so the
    duplication is the module boundary, not a missing shared home. A change to
    the arithmetic here has to be made there too; the rule is small enough that
    two spellings stay cheaper than the coupling.
    """
    return clock() - max(0.0, time.time() - epoch)


def _live_call_map(value: Any) -> dict[str, Any]:
    """The live-call map a probed accessor answered, or an empty one.

    The sibling of :func:`_phase_pair` for the other anchor this fold reads at
    attach: ``live_tool_start_epochs()`` is PROBED, so what comes back is
    whatever the host returned, and the shapes a real facade might answer instead
    of a mapping — ``None``, a bare float, a string — raise where the phase half
    now does not (``TypeError: 'x' object is not iterable``, ``ValueError:
    dictionary update sequence element #0 has length 1``). Both reads sit on the
    same unattended attach path (``RuntimeServer._serve`` ends the session on a
    raise), and "cannot say" is the answer that keeps today's behaviour.
    """
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _message_text(message: AgentMessage) -> str:
    if isinstance(message, Message):
        return message.text
    if isinstance(message, CustomMessage):
        # Custom entries render their details payload's text-ish field when
        # they have one (compaction summaries, handoffs); the rest are
        # bookkeeping the transcript never showed.
        for key in ("text", "summary", "content"):
            value = message.details.get(key)
            if isinstance(value, str):
                return value
    return ""


def _summarize_args(tool_name: str, args: dict[str, Any]) -> str:
    """The one-line summary the collapsed row shows.

    Mirrors the TUI's ``_summary_from_args`` contract: the most identifying
    argument, compacted — a path, a command line, a pattern — never a dump.
    The ordering below is the TUI's priority: what a reader scans for first
    is the file or command being touched, not the options around it.
    """
    for key in ("path", "file_path", "file", "command", "pattern", "query", "url"):
        value = args.get(key)
        if isinstance(value, str) and value:
            return _compact(value, 80)
    if args:
        first_key = next(iter(args))
        value = args[first_key]
        text = value if isinstance(value, str) else repr(value)
        return _compact(f"{first_key}={text}", 80)
    return tool_name


#: Characters of reasoning a phone transcript row holds. Bounded because the
#: row is part of the projection the phone re-renders on every repaint, and
#: reasoning is chatty (one fragment per token). Generous enough to read a
#: sentence of thinking on a phone screen, far short of a whole phase.
REASONING_PREVIEW_CHARS = 600


def _reasoning_tail(text: str, limit: int = REASONING_PREVIEW_CHARS) -> str:
    """Length-bound reasoning text, keeping the NEWEST end.

    ``_compact`` keeps the HEAD, which is right for a prompt preview and wrong
    here: reasoning streams, so what a reader wants is what the model is
    thinking NOW. The bound matters because this row rides the whole projection
    on every repaint — a phone session that reasoned for a minute would
    otherwise re-send its entire thought per frame.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return "…" + collapsed[-(limit - 1) :]


def _compact(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _compact_multiline(text: str, limit: int) -> str:
    """Length-bound ``text`` while preserving its line structure.

    ``_compact`` flattens ALL whitespace (``text.split()`` splits on newlines
    too), which is right for a one-line preview like a prompt or a tool summary
    but destroys a multi-line handoff or a stack trace \u2014 exactly the outcome
    fields (``result_text``/``error_text``) a reader most wants readable on the
    failure path. This variant collapses only intra-line runs of spaces/tabs and
    trims trailing blank lines, so a traceback stays legible line-by-line, then
    applies the same character cap.
    """
    # Collapse spaces/tabs within each line but keep the line breaks between them.
    lines = [" ".join(line.split()) for line in text.splitlines()]
    collapsed = "\n".join(lines).strip("\n")
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


def _image_refs(message: AgentMessage) -> list[dict[str, Any]]:
    """Lightweight references to a user message's image blocks — index + mime,
    never the bytes. The phone fetches the pixels lazily from the image
    endpoint (see daemon.api_session_image), which reads them back out of the
    on-disk transcript by the same index. Carrying only the reference keeps a
    per-token projection repaint from re-sending megabytes of base64.

    A block with an empty ``data`` (an attachment reference that no longer
    resolves) is still listed: the endpoint degrades it to a broken-image
    marker, which is more honest than silently dropping the attachment row.
    """
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return []
    refs: list[dict[str, Any]] = []
    # ``index`` counts IMAGE blocks only (a text caption does not shift it),
    # which is exactly what the image endpoint's _image_bytes indexes by.
    image_index = 0
    for block in content:
        if isinstance(block, ImageContent):
            refs.append({"index": image_index, "mime_type": block.mime_type or "image/png"})
            image_index += 1
    return refs


def _tool_row_details(
    args: dict[str, Any], output: str, result_details: dict[str, Any] | None
) -> dict[str, Any]:
    """The expand-on-tap payload for a settled tool row.

    Shared by the history fold and :meth:`ProjectionFold._tool_details` so a
    replayed row expands to exactly what the live one did. The caps are the
    wire budget's, not a display choice: this payload is re-sent on every
    projection repaint.
    """
    details: dict[str, Any] = {}
    if args:
        details["args"] = {k: _compact(str(v), TOOL_ARGS_CHARS) for k, v in args.items()}
    if output:
        details["output"] = output[-TOOL_OUTPUT_TAIL_CHARS:]
    if result_details:
        # Diff payloads ride through whole — the expanded row renders the
        # coloured unified diff from them.
        for key in ("diff", "added", "removed", "lines_added", "lines_removed"):
            if key in result_details:
                details[key] = result_details[key]
    return details


def _diff_counts(details: dict[str, Any] | None) -> tuple[int, int]:
    """Green/red counts from a tool result's details, when the tool reported
    them (write/edit do). Zeroes, not guesses, everywhere else."""
    if not details:
        return 0, 0
    added = details.get("added") or details.get("lines_added") or 0
    removed = details.get("removed") or details.get("lines_removed") or 0
    try:
        return int(added), int(removed)
    except (TypeError, ValueError):
        return 0, 0


def _frame_bytes(data: dict[str, Any]) -> int:
    return len(json.dumps(data).encode("utf-8"))


#: Worst-case wire bytes per NON-ASCII character. ``_frame_bytes`` measures
#: ``json.dumps(...).encode("utf-8")`` and ``json.dumps`` defaults to
#: ``ensure_ascii=True``, so the wire never sees UTF-8 for these — it sees
#: ASCII escapes. Measured, per character:
#:
#:     'a'              -> 1 byte
#:     'é' / 'д' / '中'  -> 6 bytes   (\uXXXX)
#:     astral emoji     -> 12 bytes  (surrogate pair, \uXXXX\uXXXX)
#:
#: This is why counting CHARACTERS — or even UTF-8 bytes, which stops at 8x for
#: an emoji — under-counts the wire and let an oversized frame past the gate as
#: "not degraded". 12 is the true ceiling, so charging it can only ever
#: OVER-estimate, which is the safe direction for a gate that decides whether
#: to skip the real measurement: over-estimating forces a measurement that is
#: always correct, under-estimating drops a repaint in silence.
_WIRE_BYTES_PER_NON_ASCII_CHAR = 12

#: ASCII characters the serializer does NOT pass through 1:1. Quote and
#: backslash become two bytes; the C0 controls become six (``\u0007``), except
#: the five with short forms (``\n``, ``\t``, ``\r``, ``\b``, ``\f``) which
#: become two. Charging the 6-byte ceiling for every one of them is exact
#: enough and costs one ``str.count`` per class — both are C-level scans.
#:
#: Counting them rather than multiplying the whole string by a ceiling matters:
#: a flat 6x charge stopped an ordinary 140 KB English chat frame from taking
#: the cheap path at all, which is the hot repaint A7 exists to keep cheap.
_WIRE_ESCAPE_TWO_BYTE = '"\\'
_WIRE_ESCAPE_SIX_BYTE_CEILING = 6

#: Deletion table for :meth:`str.translate` holding every ASCII codepoint the
#: serializer escapes. Counting them as ``len(s) - len(s.translate(table))`` is
#: one C-level pass; the obvious Python generator over the string measured
#: 1,840 us across an 80-row frame against 152 us to serialize the whole thing,
#: i.e. it made the "cheap" estimate 12x more expensive than the work it
#: exists to avoid.
_WIRE_ESCAPED_ASCII = {
    ord(char): None for char in ('"', "\\", *(chr(i) for i in range(0x20)), "\x7f")
}


def _wire_charge(text: str) -> int:
    """Upper bound on the bytes ``text`` will occupy on the wire.

    ``str.isascii()`` is a single C-level scan with no allocation, so the
    overwhelmingly common all-ASCII field costs one branch plus its own
    length and the cheap path stays cheap (measured: 1.9 us across an 80-row
    frame, against 1.4 us for a bare ``len()`` sum). Non-ASCII text is charged
    the ceiling rather than measured exactly, because an exact per-field
    ``json.dumps`` costs 130 us across that same frame against 167 us to
    measure the WHOLE frame properly — exactness here would buy nothing over
    just doing the real thing.
    """
    if not text.isascii():
        return len(text) * _WIRE_BYTES_PER_NON_ASCII_CHAR
    # All-ASCII: one byte each, plus the extra bytes the escaped ones cost.
    # Quotes and backslashes take one extra byte; any control character is
    # charged the 6-byte ceiling, so \n (really 2) is over-charged by 4 — the
    # safe direction, and cheap enough that exactness buys nothing.
    escaped = len(text) - len(text.translate(_WIRE_ESCAPED_ASCII))
    return len(text) + escaped * (_WIRE_ESCAPE_SIX_BYTE_CEILING - 1)


#: Safety margin on the cheap size estimate, covering the fixed frame envelope
#: and the per-field quoting the estimate does not model. The CHARACTER->byte
#: inflation is charged exactly by :func:`_wire_charge`, not smuggled in here:
#: a divisor cannot absorb a 12x term (see that function).
_FRAME_CHEAP_PROXY_DIVISOR = 3

#: Per-row charge for the JSON envelope a transcript row carries regardless of
#: its text (keys, numeric fields, booleans). Without it the estimate
#: under-counts a frame made of many SHORT rows — measured 3.5x under on a
#: 90-row roster. Derived from the empty serialized size of the dataclass.
_FRAME_ROW_ENVELOPE_BYTES = 400

#: Transcript rows the STRUCTURAL gate will vouch for without measuring. A
#: frame of this many rows cannot approach the cap once each row is charged
#: the envelope above and its own text.
_FRAME_CHEAP_PROXY_MAX_ROWS = 120


def _frame_skips_measurement(projection: SessionProjection) -> bool:
    """Whether this projection is STRUCTURALLY too simple to need measuring.

    The gate is deliberately structural rather than exhaustive, and that
    choice is the whole point. An earlier revision summed every text field it
    knew about and skipped the measurement whenever the sum cleared a margin —
    which silently reintroduced the oversized-frame defect this cap exists to
    stop, because it did not know about the fields that actually grow: a
    subagent's ``todos`` (kept on the wire BY DESIGN by
    ``set_subagent_hydrated_details``, agent-authored and unbounded), a
    subagent's ``transcript``, and ``projection.pending``. Measured on the real
    shape: 80 children x 25 todos x 600 chars estimated 40,860 bytes against a
    real 1,331,102 — returned as "not degraded", so neither the registrant's
    warning nor the tier-4 warning fired and the daemon's 1 MB reader dropped
    the frame in silence.

    An exhaustive estimate is only correct until someone adds a field, and its
    failure mode is silent over-cap frames. So the gate asks a question that
    stays true as the payload evolves: does this projection have any of the
    parts that can grow without bound? A roster, a pending card, or more than
    a screenful of rows all mean "measure it". Anything added to those
    structures in future is covered automatically, because their mere presence
    already forces the measurement — the gate fails CLOSED.

    What remains is still the overwhelmingly common repaint: an ordinary
    conversation with no children and nothing waiting on the user.
    """
    if projection.subagents or projection.pending is not None:
        return False
    if len(projection.transcript) > _FRAME_CHEAP_PROXY_MAX_ROWS:
        return False
    # Rows are bounded in COUNT above; bound them in SIZE here so one pasted
    # file cannot ride through. Only the fields a row can grow without bound
    # are summed, because the row count is already capped.
    #
    # Every text field goes through _wire_charge, never bare len(): the wire
    # charges BYTES and json.dumps escapes non-ASCII, so a character sum
    # under-counted a CJK or emoji frame by 6-12x and vouched for a payload
    # the socket then dropped in silence. Ordinary Chinese, Japanese, Russian
    # or accented French prose is on that curve — this was never exotic input.
    budget = PROJECTION_FRAME_SOFT_CAP_BYTES // _FRAME_CHEAP_PROXY_DIVISOR
    total = 0
    for entry in projection.transcript:
        total += _FRAME_ROW_ENVELOPE_BYTES
        total += _wire_charge(entry.text) + _wire_charge(entry.summary)
        total += _wire_charge(entry.error) + _wire_charge(entry.tool_name)
        total += _wire_charge(entry.tool_call_id) + _wire_charge(entry.intent)
        total += sum(_wire_charge(str(ref)) for ref in entry.images)
        for value in entry.details.values():
            total += _wire_charge(str(value))
        if total > budget:
            return False
    for phase in projection.todos:
        total += _wire_charge(phase.name) + _FRAME_ROW_ENVELOPE_BYTES
        for item in phase.items:
            total += _wire_charge(item.text) + _wire_charge(item.reason)
            total += _FRAME_ROW_ENVELOPE_BYTES
        if total > budget:
            return False
    return total <= budget


def cap_projection_frame(
    projection: SessionProjection, *, cap_bytes: int = PROJECTION_FRAME_SOFT_CAP_BYTES
) -> tuple[dict[str, Any], bool]:
    """Serialize ``projection``, degrading optional payload tiers until the
    frame fits ``cap_bytes``. Returns ``(frame_dict, degraded)``.

    The registrant broadcasts ~30 repaints/s and the daemon's control reader
    drops any line past 1 MB, so an oversized projection is not an error the
    peer reports — it is a silently lost repaint, and a flood of them starves
    the daemon loop for every OTHER session (the wedge this cap exists to
    stop at the source). The tiers drop in order of recoverability:

    1. Subagent text previews (prompt/result/error) shrink to minimal bounds,
       any embedded child transcript is dropped, and — if that is not enough —
       roster todo text is truncated and then the roster's todo lists are
       dropped entirely. The full text stays reachable through the child
       transcript's lazy /history fetch, so nothing is lost that a tap cannot
       recover. Roster todos are agent-authored and unbounded, and a deep
       roster puts many of them on the wire at once, so they are the tier's
       real work rather than a corner of it.
    1b. A pending card's option consequence lines shrink to a readable bound.
       The card itself is the loudest thing on the phone and is never dropped;
       only the prose under each option is trimmed.
    2. Transcript rows lose their ``details`` expand payload (args/output/
       diff). The collapsed row still renders; expanding an old row is the
       one gesture that can re-fetch from /history.
    3. The transcript tail halves toward ``FRAME_CAP_TRANSCRIPT_FLOOR``,
       keeping the pinned opening user message — the same scroll contract the
       80-row cap already establishes, just tighter.
    4. Row TEXT (plus ``summary``/``error``) is truncated, walking down from
       ``FRAME_CAP_ENTRY_TEXT_CHARS`` to ``FRAME_CAP_ENTRY_TEXT_FLOOR``. This
       is last because the text is what the reader came for — but one pasted
       file can exceed the whole cap by itself, and tiers 1-3 cannot touch it,
       so without this the function returned a frame the socket then dropped
       whole. The full text stays one /history fetch away.
    5. The DERIVED roster graph is shed (``peer_ids``, ``child_ids``,
       ``ancestor_ids``, ``ancestors``). Tiers 1-4 shrink TEXT, so none of them
       can touch the one field that scales with roster WIDTH: ``peer_ids`` is
       O(n^2) across a flat sibling group (see
       ``FRAME_CAP_DERIVED_ROSTER_FIELDS``). ``parent_job_id`` stays on every
       row, so the same graph is rebuildable by the reader.
    6. The roster drops to identity rows (``FRAME_CAP_ROSTER_IDENTITY_FIELDS``).
       A viewer renders one row per child with its label, lifecycle state and
       parent; per-child detail is fetched on demand, the same trade the
       transcript already makes.

    A frame still over ``cap_bytes`` after every tier is logged at WARNING
    rather than returned silently: the caller cannot fix it, but a dropped
    repaint that nobody can see is exactly the failure mode this cap exists to
    make impossible, so it must at least be visible in the log. That is now a
    genuine last resort rather than the shape of an ordinary wide-fan-out
    session: tiers 5-6 are what make a 256-sibling roster fit, and a frame that
    survives them carries something else that is not bounded at all (see the
    ``_send_to`` ceiling in ``session/runtime/server.py``, which refuses to put
    what is left on the wire).

    The projection itself is never mutated (the fold owns it and republishes
    it; the daemon retains it): degradation happens on the serialized dict.
    """
    data = projection.to_json()
    # The under-cap repaint is the hot path (~30/s per streaming session), and
    # measuring it by serializing the whole frame doubles the cost of every
    # push. Skip that only for projections whose STRUCTURE cannot approach the
    # cap (see _frame_skips_measurement for why the test is structural rather
    # than a sum of known fields). Everything else is measured for real.
    if cap_bytes >= PROJECTION_FRAME_SOFT_CAP_BYTES and _frame_skips_measurement(projection):
        return data, False
    if _frame_bytes(data) <= cap_bytes:
        return data, False

    # Tier 1: subagent text previews down to minimal bounds.
    for row in data.get("subagents") or []:
        row["prompt"] = _compact(str(row.get("prompt") or ""), FRAME_CAP_PROMPT_CHARS)
        row["result_text"] = _compact_multiline(
            str(row.get("result_text") or ""), FRAME_CAP_RESULT_CHARS
        )
        row["error_text"] = _compact_multiline(
            str(row.get("error_text") or ""), FRAME_CAP_ERROR_CHARS
        )
        # A hydrated child transcript on the wire predates the lazy /history
        # fetch; if one is still embedded it is pure frame weight.
        row["transcript"] = []
    if _frame_bytes(data) <= cap_bytes:
        return data, True

    # Tier 1b: the pending card's option prose. The card is the loudest thing
    # on the phone, so the card and its option LABELS always survive — only
    # the consequence lines under them are bounded.
    pending = data.get("pending")
    if isinstance(pending, dict):
        pending["title"] = _compact(str(pending.get("title") or ""), FRAME_CAP_PENDING_TITLE_CHARS)
        pending["detail"] = _compact_multiline(
            str(pending.get("detail") or ""), FRAME_CAP_PENDING_DETAIL_CHARS
        )
        for option in pending.get("options") or []:
            if isinstance(option, dict):
                option["description"] = _compact(
                    str(option.get("description") or ""), FRAME_CAP_PENDING_DETAIL_CHARS
                )
        if _frame_bytes(data) <= cap_bytes:
            return data, True

    # Tier 1c: roster todo text, then the todo lists themselves. Kept on the
    # wire by design (set_subagent_hydrated_details), so a deep roster carries
    # many unbounded agent-authored strings — the shape that measured
    # 1,331,102 bytes from 80 children x 25 todos. Truncate first so the
    # working line still reads, and only drop the lists if that is not enough.
    for row in data.get("subagents") or []:
        for phase in row.get("todos") or []:
            for item in phase.get("items") or []:
                item["text"] = _compact(str(item.get("text") or ""), FRAME_CAP_TODO_TEXT_CHARS)
                item["reason"] = _compact(str(item.get("reason") or ""), FRAME_CAP_TODO_TEXT_CHARS)
    if _frame_bytes(data) <= cap_bytes:
        return data, True
    for row in data.get("subagents") or []:
        row["todos"] = []
    if _frame_bytes(data) <= cap_bytes:
        return data, True

    # Tier 2: drop the expand payload of every transcript row.
    for entry in data.get("transcript") or []:
        entry["details"] = {}
    if _frame_bytes(data) <= cap_bytes:
        return data, True

    # Tier 3: halve the transcript tail toward the floor, pinning the opening
    # user message exactly like ``ProjectionFold._cap_tail`` does.
    entries = data.get("transcript") or []
    limit = max(FRAME_CAP_TRANSCRIPT_FLOOR, PROJECTION_TRANSCRIPT_LIMIT // 2)
    while len(entries) > FRAME_CAP_TRANSCRIPT_FLOOR:
        first_user = next((e for e in entries if e.get("kind") == "user"), None)
        entries = entries[-limit:]
        if first_user is not None and first_user not in entries:
            entries = [first_user, *entries[1:]]
        data["transcript"] = entries
        if _frame_bytes(data) <= cap_bytes:
            return data, True
        limit = max(FRAME_CAP_TRANSCRIPT_FLOOR, limit // 2)

    # Tier 4: truncate row text. Tiers 1-3 cannot shrink a single oversized
    # row, so this is the only tier that bounds the pasted-file case.
    text_limit = FRAME_CAP_ENTRY_TEXT_CHARS
    while True:
        for entry in data.get("transcript") or []:
            for field in ("text", "summary", "error"):
                value = entry.get(field)
                if isinstance(value, str) and len(value) > text_limit:
                    entry[field] = _compact_multiline(value, text_limit)
                    if field == "text":
                        # Keep this fact across runtime -> relay serialization.
                        # The retained ID names a prefix, not the final result end.
                        entry["text_complete"] = False
        if _frame_bytes(data) <= cap_bytes:
            return data, True
        if text_limit <= FRAME_CAP_ENTRY_TEXT_FLOOR:
            break
        text_limit = max(FRAME_CAP_ENTRY_TEXT_FLOOR, text_limit // 4)

    # Tier 5: shed the DERIVED roster graph. Nothing is lost that the reader
    # cannot rebuild: all four fields are derivations of ``parent_job_id``,
    # which every tier keeps — the canonical side already reasons this way
    # (``frontend_state`` rebuilds its parent/peer/child edges from its job
    # rows, and its ``peers()`` derives what this field precomputes), and the
    # phone's store treats a missing list as empty rather than unmounting the
    # session. Empty lists rather than deleted keys: the wire shape stays
    # uniform, and an absent list and an empty one normalise identically.
    for row in data.get("subagents") or []:
        for field in FRAME_CAP_DERIVED_ROSTER_FIELDS:
            row[field] = []
    if _frame_bytes(data) <= cap_bytes:
        return data, True

    # Tier 6: identity rows. What stays is what a reader can neither derive nor
    # fetch per child: the job id, the label, the parent edge and the lifecycle
    # state. What goes is recoverable through the per-child fetch the transcript
    # already relies on — with ONE exception, named rather than glossed:
    # ``error_text`` is ``str(exc)`` from the parent runner and is never written
    # to the child's transcript, so the lazy /history fetch cannot bring it back
    # (see ``test_live_fold_keeps_failed_child_error_text_generous``). Losing it
    # is the price of a frame this size; the alternative is a frame no viewer can
    # read at all.
    #
    # The rows themselves are the count, so no ``subagent_count`` key is added:
    # the client rebuild filters to ``SessionProjection``'s own fields and would
    # drop an unknown key on the floor, which would make it a number nobody
    # reads.
    rows = data.get("subagents")
    if rows:
        data["subagents"] = [
            {key: row[key] for key in FRAME_CAP_ROSTER_IDENTITY_FIELDS if key in row}
            for row in rows
        ]
    if _frame_bytes(data) <= cap_bytes:
        return data, True

    # Every tier is spent. The frame is as small as this function can make it;
    # say so loudly rather than handing the socket a line it will drop whole.
    final_size = _frame_bytes(data)
    if final_size > cap_bytes:
        logger.warning(
            "mobile projection frame for session %s is %d bytes after every "
            "degradation tier (cap %d) — the control socket will drop it",
            data.get("session_id", "?"),
            final_size,
            cap_bytes,
        )
    return data, True


def fold_messages_to_entries(history: list[AgentMessage]) -> list[TranscriptEntry]:
    """Fold a message history into transcript entries — THE phone's row fold.

    UNCAPPED and pure. Both phone surfaces come through here: the daemon's
    history endpoint calls it directly for the older pages a cap dropped, and
    :meth:`ProjectionFold.fold_history` calls it for the attach seed and then
    layers its correlation-map maintenance and ``_cap_tail`` on top.

    WHY ONE FUNCTION. This used to be two near-identical folds (this one and
    a copy inside ``fold_history``), each carrying a comment asserting the
    other could not disagree with it. They disagreed: a hub steer rendered as
    a clean ``parent_message`` card here and leaked the raw
    ``<parent-message>`` XML envelope as a ``notice`` on the attach seed, so
    the phone contradicted ITSELF one scroll gesture apart. A contract two
    functions promise to keep is a contract nothing checks — the only fix
    that holds is that there is one function to change.

    Row semantics that both the phone and the TUI must agree on live in
    ``harness/rows.py`` and are called from here, so a decision cannot be
    made on one surface and missed on the other.

    Tool-result diffs ride in ``provider_payload["details"]`` (where the
    harness stores them); rehydrated messages carry that payload, so the
    write/edit rows expand to their diff exactly like the live ones.
    """
    entries: list[TranscriptEntry] = []
    # tool_call_id -> its row, local to this fold (a fresh fold re-pairs).
    tool_rows: dict[str, TranscriptEntry] = {}
    tool_args: dict[str, dict[str, Any]] = {}
    # tool_call_id -> the result that settled it, indexed UP FRONT. The
    # turn-level notice is decided on the ASSISTANT message, which this linear
    # fold reaches before the results that answer it, and the notice needs the
    # limit's arm to avoid naming a cause the call's own row contradicts
    # (design round 1, D1). One extra pass over the history, no extra state.
    # ``isinstance(Message)``, not a bare attribute read: a history holds
    # ``CustomMessage`` rows too (hub steers, gate timeouts), and those carry no
    # ``role`` or ``tool_call_id`` at all.
    settled: dict[str, AgentMessage] = {
        message.tool_call_id: message
        for message in history
        if isinstance(message, Message) and message.role == "tool" and message.tool_call_id
    }
    # Message ids whose assistant turn opened a bang-mode (`! cmd`) command,
    # so the call it issues opens expanded exactly as the TUI's does.
    bang_pending = False
    for message in history:
        if isinstance(message, CustomMessage):
            if message.custom_type == HUB_MESSAGE_TYPE:
                # The parent's own words (``body``), never the model-facing
                # envelope in ``details["text"]``. Reading the envelope here
                # is what leaked raw XML onto the attach seed.
                body = str(message.details.get("body") or "").strip()
                direction = message.details.get("direction")
                if body:
                    entries.append(
                        TranscriptEntry(
                            id=message.id,
                            kind=(
                                "parent_message" if direction == "to_child" else "subagent_message"
                            ),
                            text=body,
                        )
                    )
                continue
            if message.custom_type == PEER_MESSAGE_MESSAGE_TYPE:
                # A cross-session `lop send` delivery. The phone renders the raw
                # body plus the sender identity (from details["sender"]); the
                # model-facing wrapped envelope in details["text"] never travels.
                body = str(message.details.get("body") or "").strip()
                if body:
                    entries.append(
                        TranscriptEntry(
                            id=message.id,
                            kind="peer_message",
                            text=body,
                            details={"sender": message.details.get("sender") or {}},
                        )
                    )
                continue
            if message.custom_type == WAKE_PROMPT_MESSAGE_TYPE:
                # A wake delivery is a receipt with its own identity, not a
                # generic notice: a resumed session that showed the agent
                # answering a wake with no sign the wake fired is the bug this
                # row exists to prevent. The CATCH-UP prompt is skipped for the
                # opposite reason — it is user-attributed, so replaying it
                # would put a raw '(alarm) The session resumed…' line in the
                # transcript as if the user had typed it.
                details = message.details or {}
                if not details.get("wake_catchup"):
                    # Strip the model-facing envelope with the SAME helper the
                    # TUI's WakeBlock uses. The raw payload is
                    # '(alarm) Scheduled wake w-9 (1, every 6h) — cancel with
                    # wake({op:"cancel",id:"w-9"})', which is markup addressed
                    # to the model; painting it verbatim on a human surface is
                    # the same defect as the leaked <parent-message> rows.
                    raw = str(details.get("text", ""))
                    headline = wake_receipt_headline(raw)
                    _, _, body = raw.partition("\n\n")
                    entries.append(
                        TranscriptEntry(
                            id=message.id,
                            kind="notice",
                            # Headline plus the delivered prompt: the phone has
                            # no expand affordance for a notice row, so the
                            # prompt rides the same row rather than being
                            # dropped (the TUI hides it behind an expansion).
                            text=_compact(
                                f"{headline} — {body.strip()}" if body.strip() else headline,
                                400,
                            ),
                            details={"notice_kind": "wake"},
                        )
                    )
                continue
            if message.custom_type == GATE_TIMEOUT_CUSTOM_TYPE:
                # A gate that timed out unattended is the most expensive event
                # in the detached feature — up to a day of held residency ends
                # here — and it rendered NOWHERE on the phone: the user
                # returned to a conversation that promised an action and
                # appeared to simply stop.
                entries.append(
                    TranscriptEntry(
                        id=message.id,
                        kind="notice",
                        text=_compact(gate_timeout_notice(message.details or {}), 400),
                        details={"severity": "warning"},
                    )
                )
                continue
            if message.custom_type == COMPACTION_REFUSED_TYPE:
                # A compaction that did NOT run, correcting the optimistic
                # "compacting context…" receipt the routed command showed.
                # Severity is derived by the shared helper so the phone cannot
                # flatten a FAILURE into the same ink as a decline.
                text, severity = compaction_refused_notice(message.details or {})
                entries.append(
                    TranscriptEntry(
                        id=message.id,
                        kind="notice",
                        text=_compact(text, 400),
                        details={"severity": severity},
                    )
                )
                continue
            text = _message_text(message)
            if text:
                entries.append(
                    TranscriptEntry(id=message.id, kind="notice", text=_compact(text, 400))
                )
            continue
        if message.role == "user":
            parent_message = extract_parent_message(message.text)
            if parent_message is not None:
                # A persisted hub steer: model-facing XML around the parent's
                # own words. The phone shows the body the parent authored,
                # never the envelope.
                entries.append(
                    TranscriptEntry(id=message.id, kind="parent_message", text=parent_message.body)
                )
                continue
            if is_harness_chrome(message.text):
                # Harness chrome, not the user's words. The loop persists these
                # so the TRANSCRIPT records why the conversation continued, but
                # no front end paints one as a user bubble. The list is shared
                # with the TUI (``harness/rows.py``) because this fold used to
                # carry a PARTIAL copy of it — suppressing the connectivity
                # prompt while rendering the goal-loop and auto-continuation
                # prompts as the user's own words.
                continue
            if is_harness_notice_row(message):
                # A row the harness wrote — a stamped render of a
                # ``CustomMessage`` (model-switch notice, incident, wake
                # delivery) or a notice a compaction block carried forward from
                # before the stamp existed — and the operator never typed it.
                # The live fold has its own receipt for the moments these
                # announce, so dropping the rendered copy is live/replay
                # parity, and it also hides the rows an older build left in
                # existing transcripts. One decision, in ``harness/rows.py``,
                # for the same reason the chrome list lives there.
                continue
            # A `$skill` invocation persists as its EXPANDED payload, because
            # that is what the model was sent. Rendering it verbatim showed the
            # whole SKILL.md body as the user's bubble and titled the session
            # after it; the typed line rides the payload's own opening tag.
            text = user_row_text(message.text)
            # Carry image attachments as references so an image-only prompt
            # (the composer allows "" text + images) renders its thumbnails on
            # replay instead of round-tripping as an empty bubble — the same
            # inline render the live fold produces. The bytes are fetched
            # lazily from the image endpoint; only the reference travels here.
            refs = _image_refs(message)
            entries.append(TranscriptEntry(id=message.id, kind="user", text=text, images=refs))
            # A bang-mode receipt replays as open as it lived: the user row is
            # `! <command>` and the assistant message that follows carries
            # exactly one bash call, whose card opens expanded. SET only,
            # never cleared here — matching the TUI's replay exactly, which
            # clears the flag on the next ASSISTANT message rather than on an
            # intervening user row.
            if text.startswith("! "):
                bang_pending = True
        elif message.role == "assistant":
            # Consume the pending bang marker on EVERY assistant message:
            # record_shell writes the call-bearing assistant immediately after
            # the `!` row, so a later unrelated turn must not inherit the flag.
            message_bang = bang_pending
            bang_pending = False
            # Through the shared helper, not `if message.text:` — the latter
            # is truthy for `"   "`, so a whitespace-only turn painted an
            # extra EMPTY row here that the TUI (which tests its stripped
            # text) never emits. Same rows plus one blank one is not the
            # same rows.
            assistant_text = assistant_row_text(message.text)
            if assistant_text:
                entries.append(
                    TranscriptEntry(id=message.id, kind="assistant", text=assistant_text)
                )
            for call in message.tool_calls:
                # Only the FIRST call of a bang assistant message is the
                # command's own card; the shape record_shell writes has exactly
                # one, so consuming here is exact in practice.
                user_run = bool(
                    message_bang and message.tool_calls[0] is call and call.name == "bash"
                )
                entry = TranscriptEntry(
                    id=f"{message.id}:{call.id}",
                    kind="tool",
                    tool_call_id=call.id,
                    tool_name=call.name,
                    # UNKNOWN until a result pairs, not "done". A call whose
                    # result never arrived is a call that never returned, and
                    # painting it ✓ asserts an outcome nobody observed — the
                    # same "a default that means success" class as the three
                    # shipped duration bugs. The tool-role branch below settles
                    # it; anything this fold never pairs stays interrupted.
                    tool_state="interrupted",
                    summary=_summarize_args(call.name, call.arguments or {}),
                    details={"user_run": True} if user_run else {},
                )
                entries.append(entry)
                tool_rows[call.id] = entry
                tool_args[call.id] = call.arguments or {}
            notice = assistant_stop_notice(
                text=message.text,
                has_tool_calls=bool(message.tool_calls),
                stop_reason=getattr(message, "stop_reason", None),
                provider_payload=message.provider_payload,
                # The arm, from this turn's OWN results (design round 1, D1):
                # the notice may say the limit cut a call only when a call in
                # this turn says so, or it names a cause the call's own card
                # contradicts — the card is folded BEFORE this notice
                # (``[user, tool row, notice]``, measured on this fold), so the
                # claim and the row that refutes it are read as one turn rather
                # than as two facts about it.
                cut_tool_call=turn_cut_tool_call(message.tool_calls, settled),
            )
            if notice is not None:
                # A refused, failed or interrupted turn. The phone had no
                # ``stop_reason`` branch at all, so a truncated answer looked
                # complete and a failed turn looked like the agent ignoring
                # the user.
                text, severity = notice
                entries.append(
                    TranscriptEntry(
                        id=f"{message.id}:stop",
                        kind="notice",
                        text=_compact(text, 400),
                        details={"severity": severity},
                    )
                )
        elif message.role == "tool":
            entry = tool_rows.get(message.tool_call_id or "")
            if entry is not None:
                entry.tool_state = "failed" if message.is_error else "done"
                payload = message.provider_payload or {}
                result_details = payload.get("details")
                # A call the OUTPUT LIMIT kept from running persists a SYNTHETIC
                # result whose text is addressed to the MODEL, so the failed row
                # used to carry an imperative meant for the agent ("Reply with
                # the call itself…") as the operator's own receipt, and carried
                # it twice: as the row's error line and, clipped, in the
                # expand-on-tap output (review round 1, F2). Both take the
                # harness's vocabulary for this condition instead, from the arm
                # marker on the result. EVERY other tool message keeps its text
                # untouched — this is keyed on the marker, never on the wording.
                receipt = output_limit_call_receipt(result_details) if message.is_error else None
                result_text = receipt or message.text
                if message.is_error:
                    entry.error = _compact(result_text, 200)
                duration = payload.get("duration_s")
                if isinstance(duration, (int, float)) and not isinstance(duration, bool):
                    entry.elapsed_s = float(duration)
                entry.diff_added, entry.diff_removed = _diff_counts(
                    result_details if isinstance(result_details, dict) else None
                )
                details = _tool_row_details(
                    tool_args.get(message.tool_call_id or "", {}),
                    # The receipt belongs to the row's error line and NOWHERE
                    # else on this surface: passing it here as well put the
                    # identical sentence in the red paragraph above the args and
                    # in the sunken output block below them, so one tap showed
                    # one sentence twice (design round 1, D5). Nothing ran for
                    # this call, so it has no output to expand — the arguments
                    # still do, and that is what an operator opening the row is
                    # reading for.
                    "" if receipt else result_text,
                    result_details if isinstance(result_details, dict) else None,
                )
                # The expansion flag is set on the CALL and must survive the
                # result settling the row.
                if entry.details.get("user_run"):
                    details["user_run"] = True
                entry.details = details
    return entries


class ProjectionFold:
    """Incremental fold of one session's events into a SessionProjection."""

    def __init__(self, projection: SessionProjection) -> None:
        self.projection = projection
        # tool_call_id -> transcript entry id, so start/update/end land on
        # the same row regardless of interleaving.
        self._tool_rows: dict[str, str] = {}
        self._tool_started_at: dict[str, float] = {}
        self._tool_args: dict[str, dict[str, Any]] = {}
        # The streaming assistant row, if one is open.
        self._open_message_id: str | None = None
        #: The open REASONING row's id, tracked explicitly for the same reason
        #: ``_open_compaction_id`` is: several model calls in one turn each
        #: reason, and a reverse-scan fallback could finalize a later phase's row
        #: once the tail cap starts dropping rows.
        self._open_reasoning_id: str | None = None
        # The open compaction row's id, tracked explicitly the same way: a
        # reverse-scan fallback could finalize a LATER compaction's row with
        # an EARLIER end event once the tail cap starts dropping rows.
        self._open_compaction_id: str | None = None
        # Subagent roster by job id; progress updates are throttled upstream
        # (SubagentProgressEvent is never per-delta by contract).
        self._subagents: dict[str, SubagentRow] = {}
        self._subagent_started_at: dict[str, float] = {}
        # The working line's label, the PHASE that label belongs to, and its
        # clock origin.
        #
        # ``_activity_started_at`` is the instant the current phase is dated
        # from, or ``None`` when this fold has none it can honestly use (a phase
        # joined mid-flight whose producer stated no instant) — and that
        # ``None`` is published as such, not as a zero. The phase is tracked
        # separately from the label because the clock belongs to the phase: a
        # batch's second announcement or an intent revised mid-call relabels the
        # line without restarting a number the producer never restarted either.
        self._activity_started_at: float | None = None
        self._activity_phase: str = ""
        # The producer's folded PHASE at attach, and its instant when one was
        # stated. The name alone answers a question the fold cannot answer from
        # its own events — "was this phase already running when I arrived?" —
        # which is what separates an edge the fold watched (its own zero, and a
        # true one) from one it joined late (the producer's instant, or no clock
        # at all). Applied only through ``_activity_anchor``.
        self._attach_phase: str = ""
        self._attach_phase_at: float | None = None
        # Whether the fold has folded a turn-terminal event (agent_end /
        # turn_end) that no later agent_start has superseded. This is what
        # makes ``reconcile_streaming`` safe on the abort/error path: there the
        # session emits AgentEndEvent INLINE while its ``is_streaming`` flag is
        # still True (the flag clears several awaits later, in the turn's
        # ``finally``), so a mobile command landing in that window must not be
        # allowed to raise ``streaming`` back to True over the fold's correct
        # False. Seeded False so a mid-turn attach can still seed streaming up.
        self._streaming_ended = False
        # Approval/ask requests waiting on the user, FIFO. Owned sessions can
        # have several at once (a parallel tool batch); the phone renders the
        # front one and a "1 of N" badge. See push_pending/pop_pending.
        self._pending_queue: list[PendingRequest] = []
        # Optimistic user rows this fold painted whose MessageStartEvent has
        # not arrived yet, keyed by the message id the handle handed the
        # session. `absorb_user_event` pops the entry instead of scanning the
        # transcript tail: a phone steer is announced at a LATER tool boundary,
        # so assistant and tool rows push its echo out of any fixed window and
        # the tail scan repainted it (issue #231, the mobile twin of the TUI's
        # #227). Holds the TranscriptEntry itself, because the event's job is
        # to upgrade that row in place (id and image refs) rather than to
        # suppress a second one.
        #
        # An entry is retired by its own event, and otherwise by the wholesale
        # `fold_history` rebuild — the rows it described are gone and the
        # rebuilt tail carries the persisted messages instead, so a surviving
        # entry would point at a row no longer in the transcript.
        self._pending_user_echoes: dict[str, TranscriptEntry] = {}

    # -- history -----------------------------------------------------------

    def fold_history(self, history: list[AgentMessage]) -> None:
        """Wholesale fold on attach: rebuild the transcript tail from the
        session's persisted history.

        Row production is delegated ENTIRELY to
        :func:`fold_messages_to_entries` — this method owns only what is
        stateful and therefore cannot live in a pure function: the
        ``_tool_rows``/``_tool_args`` correlation maps that let a LATER live
        event settle a row this seed painted, and the ``_cap_tail`` trim to
        the render window.

        WHY THE DELEGATION. This method used to carry its own copy of the
        fold, under a comment promising it could not disagree with the
        original. It did disagree: a hub steer leaked its raw
        ``<parent-message>`` XML envelope here while rendering as a clean card
        on a history page, so the phone contradicted itself one scroll gesture
        apart. Both comments asserted a contract that nothing enforced. Now
        there is one function to change, and the seed and the page cannot
        disagree because they are the same code.

        The cap is applied AFTER the fold, so the extra notice rows the fold
        now produces (refusals, failed turns, gate timeouts) compete for the
        window like any other row — ``_cap_tail`` still pins the opening user
        message so a tail full of notices cannot push the conversation's
        subject out of the seed.
        """
        entries = fold_messages_to_entries(history)
        # Rebuild the correlation maps from the rows just produced. Keyed off
        # the fold's output rather than maintained during it: the pure fold
        # cannot own instance state, and its row ids are stable
        # (``{message.id}:{call.id}``), so this is a total reconstruction
        # rather than a second pass over the messages.
        self._tool_rows = {
            entry.tool_call_id: entry.id
            for entry in entries
            if entry.kind == "tool" and entry.tool_call_id
        }
        self._tool_args = {}
        for message in history:
            if isinstance(message, CustomMessage) or message.role != "assistant":
                continue
            for call in message.tool_calls:
                self._tool_args[call.id] = call.arguments or {}
        # A resumed fold starts clean: no streaming row, no half-run tools, and
        # no pending echoes — the rows those entries pointed at have just been
        # replaced wholesale, and the persisted history already carries any
        # message that was really delivered.
        self._open_message_id = None
        self._pending_user_echoes.clear()
        self.projection.transcript = self._cap_tail(entries)
        # Prune the correlation maps to the surviving tail: a long history
        # pairs every historical tool call, and keeping ids whose rows were
        # cut is dead weight per rebind (and a stale hit if a call id were
        # ever reused).
        surviving = {entry.id for entry in self.projection.transcript}
        self._tool_rows = {
            call_id: entry_id
            for call_id, entry_id in self._tool_rows.items()
            if entry_id in surviving
        }
        live_call_ids = set(self._tool_rows)
        self._tool_args = {
            call_id: args for call_id, args in self._tool_args.items() if call_id in live_call_ids
        }
        self._bump()

    # -- events ------------------------------------------------------------

    def fold_event(self, event: AgentEvent) -> None:
        """Fold one live event. The dispatch is explicit if/elif rather than
        a registry so a new harness event type fails loudly here (AttributeError
        on construction import) instead of being silently dropped."""
        p = self.projection
        if isinstance(event, AgentStartEvent):
            p.streaming = True
            self._streaming_ended = False
        elif isinstance(event, AgentEndEvent):
            p.streaming = False
            self._streaming_ended = True
            p.queued_count = 0
            # A CUT-OFF TURN IS AN ABORT, not a completion. The taxonomy flips an
            # involuntary end to `aborted=False, error=<notice>` so every
            # existing surface paints it as a failure, and this fold naively read
            # that as "the turn finished" — which silently removed the phone's
            # only recovery affordance for exactly the sessions the operator
            # reports losing (`composer.tsx` gates `interrupted — tap to resume`
            # on `stop_reason === "aborted"`, review round 1 MAJOR-1). The field
            # means "why streaming last stopped", and a turn that was cut off
            # stopped without finishing; `cut_off`/`cut_off_cause` is how the
            # session states that, so it is read here rather than inferred from
            # `aborted` alone.
            cut_off = bool(event.cut_off or event.cut_off_cause)
            p.stop_reason = "aborted" if (event.aborted or cut_off) else "completed"
            # ...and the phone's BUTTON needs to know which of the two it was:
            # "aborted" is both a deliberate stop and a cut-off, and pairing a
            # `Stopped with an error — ...` notice with a button reading
            # `interrupted — tap to resume` names one act two ways (design round
            # 2, D7). Deliberately a separate flag rather than a third
            # `stop_reason` value: the affordance is gated on `=== "aborted"`,
            # so a new token would strip it from every bundle not yet updated.
            p.cut_off = cut_off
            # A turn cut off mid-think ends with a reasoning row still open and
            # no ``message_end`` coming; sealing it here keeps the row from
            # absorbing the NEXT call's fragments ("closes the row" is about
            # which phase owns it, and this one is over).
            self._close_reasoning_row()
            self._close_open_message()
            if event.error:
                self._append(
                    TranscriptEntry(
                        id=f"err-{time.time_ns()}", kind="notice", text=_compact(event.error, 400)
                    )
                )
        elif isinstance(event, TurnEndEvent):
            # NOT a streaming terminal. TurnEndEvent fires after EVERY model
            # turn within a run (harness.loop yields it whenever the assistant
            # produced tool calls and the run will continue — loop.py ~589), so
            # a multi-batch turn emits several before the run ends. The session
            # keeps ``is_streaming`` True across them and clears it only after
            # AgentEndEvent, and the TUI's working line stays up the same way,
            # so the phone must too. Flipping streaming False here (and, worse,
            # latching ``_streaming_ended``) blanked the working line mid-run
            # and pinned it off. This branch previously set streaming False and
            # was harmless ONLY because the removed per-event is_streaming
            # re-read overwrote it every time; with the fold now authoritative
            # it must leave streaming alone. AgentEndEvent is the sole terminal.
            pass
        elif isinstance(event, MessageStartEvent):
            if isinstance(event.message, Message) and event.message.role == "assistant":
                entry = TranscriptEntry(
                    id=event.message.id,
                    kind="assistant",
                    text=event.message.text,
                    final=False,
                )
                self._append(entry)
                self._open_message_id = event.message.id
            elif isinstance(event.message, Message) and event.message.role == "user":
                self.absorb_user_event(event.message)
        elif isinstance(event, MessageUpdateEvent):
            if self._open_message_id and event.message.id == self._open_message_id:
                row = self._find(self._open_message_id)
                if row is not None:
                    # Append the delta; never re-read the whole message — the
                    # delta contract is what makes 30 Hz streaming cheap.
                    row.text += event.delta
        elif isinstance(event, ReasoningDeltaEvent):
            # The model's reasoning, streamed onto its own row. Display-only in
            # the same sense the event is: the row is not the assistant message,
            # never becomes one, and disappears at the next sync because the
            # reasoning is not in the durable transcript the history path
            # projects from — which is the honest rendering of a phase that has
            # no durable form.
            if event.delta:
                self._reasoning_row(event.message_id, event.delta)
        elif isinstance(event, MessageEndEvent):
            row = self._find(event.message.id)
            if row is not None:
                row.text = _message_text(event.message)
                row.final = True
            self._close_reasoning_row()
            self._open_message_id = None
        elif isinstance(event, ToolCallComposeEvent):
            # Same rekey the TUI does: the call's real id has arrived for a row
            # opened under an index-derived placeholder, so move the existing
            # row's correlation onto the real id. Without it ``_tool_row``
            # misses and appends a SECOND row for one call, and the placeholder
            # row is left composing forever because every later start/end
            # carries the real id.
            #
            # The entry's own ``id`` is deliberately left alone: clients diff
            # the transcript by row id, so re-identifying a row mid-turn would
            # read as the row being replaced. Only the correlation key moves.
            # Plain attribute read is safe HERE, unlike in the TUI: this branch
            # sits behind ``isinstance(event, ToolCallComposeEvent)``, so the
            # field is guaranteed to exist with its ``None`` default. The TUI's
            # equivalent keeps ``getattr`` because its dispatch routes on
            # ``event.type`` without re-validating, so a legacy relayed frame
            # reaches it as a bare ``AgentEvent`` (see the comment there).
            # An older runtime that omits the field arrives as ``None`` and takes
            # the falsy no-op path below.
            superseded = event.supersedes_tool_call_id
            if superseded:
                entry_id = self._tool_rows.pop(str(superseded), None)
                if entry_id is not None:
                    self._tool_rows[event.tool_call_id] = entry_id
                    promoted = self._find(entry_id)
                    if promoted is not None:
                        promoted.tool_call_id = event.tool_call_id
            row = self._tool_row(event.tool_call_id, event.tool_name)
            # Three endings to one dictation, in the frames the producer sends:
            #
            # * an ordinary frame — the model is still writing the call;
            # * `dictation_complete` — it has stopped, and the call may still be
            #   waiting behind a sibling's execution group (or for a group the
            #   turn never reached), so the row says `queued` rather than going
            #   on claiming the model is dictating something it finished
            #   writing. The TUI landed the same state in the same commit; the
            #   phone was the surface that used to keep the lie after the row
            #   above it had stopped;
            # * `not_run_reason` — it will never run (planning failure,
            #   duplicate id, steering skip), and the reason is the harness's
            #   own. Settled here rather than left to the client, because a
            #   phone has no retirement pass to fall back on.
            #
            # Both new fields are read off the model rather than with
            # `getattr`, unlike the TUI: this branch is behind an
            # ``isinstance`` check, so an older runtime's frame carries the
            # defaults (`False`/`None`) and takes the composing path below —
            # today's behaviour, unchanged.
            #
            # A row that has already STARTED outgrows the whole announcement,
            # and the TUI returns from its handler for exactly this case (its
            # running registry). The phone needs the guard too: a replayed
            # terminal frame for a call whose twin's start already landed would
            # otherwise relabel a running row `failed` — or, because the
            # terminal frame also carries `dictation_complete`, walk it BACK to
            # `queued`. Note the arms are exclusive on purpose: falling through
            # to them is the bug, not the fallback.
            started = row.tool_state in ("running", "done", "failed")
            if event.not_run_reason:
                if not started:
                    row.tool_state = "failed"
                    row.error = _compact(event.not_run_reason, 200)
                    # The reason in the one-line summary too: the row is all a
                    # phone shows by default, and "this call never ran, here is
                    # why" is the whole content of that fact.
                    row.summary = row.error
            elif started:
                pass
            elif event.dictation_complete:
                row.tool_state = "queued"
                row.summary = event.intent or f"waiting to run {event.tool_name}"
            else:
                row.tool_state = "composing"
                row.summary = event.intent or f"dictating {event.tool_name}"
            row.intent = event.intent or ""
            row.details["argument_bytes"] = event.argument_bytes
        elif isinstance(event, ToolExecutionStartEvent):
            row = self._tool_row(event.tool_call_id, event.tool_name)
            row.tool_state = "running"
            # The failure TEXT goes with the failure STATE. This row may have
            # been settled by a never-run verdict before its call started — two
            # calls sharing an id, the loser settling the row and the winner
            # running it — and the renderer draws `error` as a red danger line
            # inside the expansion for ANY state, so the phone would otherwise
            # keep `Duplicate call id '…' skipped.` over a row that succeeded,
            # and `hasDetails` true because of it.
            row.error = ""
            row.summary = _summarize_args(event.tool_name, event.args)
            row.intent = event.intent or row.intent
            # The call's OWN start instant where the producer stated one. A
            # start event this fold sees LATE — the fold was built at attach, the
            # stream was relayed, the event was redelivered — would otherwise be
            # dated from this fold's arrival, which is the fabricated zero the
            # operator's report is about: both the row's elapsed reading and the
            # duration its end event measures are taken from this instant. No
            # stated epoch keeps today's behaviour, which for a genuinely live
            # start IS the call's start (the row appears with the event that
            # began it) and for a late one is the arrival instant the end event's
            # own ``duration_s`` corrects.
            stated = _stated_epoch(event.started_at_epoch)
            self._tool_started_at[event.tool_call_id] = (
                monotonic_from_epoch(stated) if stated is not None else time.monotonic()
            )
            self._tool_args[event.tool_call_id] = event.args
        elif isinstance(event, ToolExecutionUpdateEvent):
            row = self._tool_row(event.tool_call_id, event.tool_name)
            # Partial output means the call IS running. Normally the start
            # event already said so, but a relayed stream can deliver an
            # update whose start was dropped, and such a row would otherwise
            # keep the factory default — which must never be a state this
            # path can observe to be false.
            if row.tool_state not in ("done", "failed"):
                row.tool_state = "running"
            text = getattr(event.partial_result, "text", "") or ""
            if text:
                row.details["partial"] = text[-TOOL_OUTPUT_TAIL_CHARS:]
        elif isinstance(event, ToolExecutionEndEvent):
            row = self._tool_row(event.tool_call_id, event.tool_name)
            result = event.result
            row.tool_state = "failed" if result.is_error else "done"
            # From the instant this fold holds for the call: its own observation
            # of the start, or the producer's seeded epoch
            # (``reconcile_clocks``), so a call that began BEFORE this fold
            # existed reports the duration it really ran rather than the time
            # since the phone attached. The default is the honest answer for a
            # call with no start at all — nothing to measure, which is exactly
            # what today's code measured — and the producer's own
            # ``duration_s``, when it stated one, is the authoritative half.
            measured = time.monotonic() - self._tool_started_at.pop(
                event.tool_call_id, time.monotonic()
            )
            row.elapsed_s = round(
                event.duration_s if event.duration_s is not None else measured,
                1,
            )
            row.diff_added, row.diff_removed = _diff_counts(result.details)
            if result.is_error:
                row.error = _compact(result.text, 200)
            row.details = self._tool_details(
                self._tool_args.pop(event.tool_call_id, {}), result.text, result.details
            )
        elif isinstance(event, NoticeEvent):
            # ``kind`` rides into ``details`` as the phone's ``severity``. This
            # fold is the phone's ONLY view of a LIVE notice, `NoticeRow` reads
            # the glyph and the ink from that field alone, and dropping it drew
            # every live warning as the quiet ``·`` in ``text-ink-dim`` -- the
            # tier this surface's own test file calls "a receipt nobody has to
            # read" -- while the SAME event, replayed after a reconnect, took
            # the branch below and rendered amber ``!``. One event, two inks,
            # decided by whether the client attached before or after it
            # (design round 1, D1; measured, replay ``{'severity': 'warning'}``
            # against live ``{}``). The three kinds map 1:1 onto the phone's
            # ``info|warning|error``.
            self._append(
                TranscriptEntry(
                    id=f"nt-{time.time_ns()}",
                    kind="notice",
                    text=_compact(event.text, 400),
                    details={"severity": event.kind},
                )
            )
        elif isinstance(event, SteeringDeliveredEvent):
            p.queued_count = max(0, p.queued_count - event.count)
        elif isinstance(event, SubagentStartEvent):
            self._subagents[event.job_id] = SubagentRow(
                job_id=event.job_id, label=event.label, status="running"
            )
            self._subagent_started_at[event.job_id] = time.monotonic()
        elif isinstance(event, SubagentProgressEvent):
            row = self._subagents.get(event.job_id)
            if row is not None:
                row.progress = event.progress
                row.elapsed_s = round(
                    time.monotonic() - self._subagent_started_at.get(event.job_id, time.monotonic())
                )
        elif isinstance(event, SubagentEndEvent):
            row = self._subagents.get(event.job_id)
            if row is None:
                row = SubagentRow(job_id=event.job_id, label=event.label)
                self._subagents[event.job_id] = row
            row.status = event.status  # type: ignore[assignment] — Literal matches
            row.progress = ""
            # result_text is a preview (recoverable via the child transcript);
            # error_text is carried generously because it is NOT in the
            # transcript and the wire value is the only copy the phone renders.
            # Both preserve newlines so a multi-line handoff/trace stays legible.
            row.result_text = _compact_multiline(event.result_text or "", SUBAGENT_OUTCOME_CHARS)
            row.error_text = _compact_multiline(event.error_text or "", SUBAGENT_ERROR_CHARS)
            row.elapsed_s = round(
                time.monotonic() - self._subagent_started_at.pop(event.job_id, time.monotonic())
            )
        elif isinstance(event, CompactionStartEvent):
            entry = TranscriptEntry(
                id=f"cx-{time.time_ns()}",
                kind="compaction",
                text="compacting context…",
                final=False,
            )
            self._append(entry)
            self._open_compaction_id = entry.id
        elif isinstance(event, CompactionEndEvent):
            row = self._find(self._open_compaction_id)
            self._open_compaction_id = None
            if row is not None:
                row.final = True
                row.text = (
                    f"context compacted {event.tokens_before:,} → " f"{event.tokens_after:,} tokens"
                    if event.success
                    else "context compaction failed"
                )
        elif isinstance(event, RetryStartEvent):
            note = f"retrying ({event.attempt}): {_compact(event.error, 120)}"
            if event.fallback_model:
                note += f" — falling back to {event.fallback_model}"
            self._append(TranscriptEntry(id=f"rt-{time.time_ns()}", kind="notice", text=note))
        elif isinstance(event, RetryEndEvent):
            pass  # the retry row already reads; success is the next assistant row
        elif isinstance(event, ModelChangeEvent):
            # The composer chip and the session-list model must name the
            # model ACTUALLY answering. A display that kept the selected
            # primary after a quota fallback is the stale chip the phone
            # showed while Grok was serving.
            p.model_label = f"{event.provider}/{event.model_id}"
            if event.effort:
                p.effort = event.effort
        # Unknown events are dropped by design: the fold renders a SUBSET of
        # the harness taxonomy (the phone has no use for wake/loop internals),
        # and ``extra="allow"`` on AgentEvent means matching must stay
        # structural, not exhaustive-by-name.
        self._sync_subagents()
        self._derive_activity(event)
        self._bump()

    # -- user turns --------------------------------------------------------------

    def note_user_message(
        self, text: str, *, steer: bool = False, message_id: str | None = None
    ) -> None:
        """Append the user's own message to the transcript. Called by the
        handle's prompt/steer path, because the harness only emits
        MessageStartEvent for ASSISTANT messages — a live user prompt never
        reaches the fold as an event, so without this the phone showed the
        agent's reply with no sign of what the human asked (and, for a
        phone-sent prompt, no echo of the tap at all).

        ``message_id`` is the id the handle handed the session for this very
        message (the ``ContinuationCommand`` id, which becomes the ``Message``
        id). Recorded in the pending-echo registry so :meth:`absorb_user_event`
        recognises the eventual event as THIS row rather than guessing from the
        transcript tail — see that method for why the guess was wrong. Omitted
        by callers with no id to offer, which keeps the legacy tail match.
        """
        entry = TranscriptEntry(
            # Synthetic id when the handle has none: the row still needs a
            # unique key for the web client's list reconciliation, and the
            # registry below is what carries the correlation instead.
            id=message_id or f"user-{time.time_ns()}",
            kind="steer" if steer else "user",
            text=text,
            final=True,
        )
        self._append(entry)
        if message_id:
            self._pending_user_echoes[message_id] = entry
        self._bump()

    def note_peer_message(self, text: str, *, sender: dict[str, Any] | None = None) -> None:
        """Optimistic echo of an inbound cross-session (`lop send`) message.

        Same reason as ``note_user_message``: the handle calls this the instant
        it delivers a peer message so an attached phone paints the peer card
        immediately, rather than waiting for the next full projection repaint.
        The row carries the sender identity in ``details`` for the label."""
        self._append(
            TranscriptEntry(
                id=f"peer-{time.time_ns()}",
                kind="peer_message",
                text=text,
                details={"sender": sender or {}},
                final=True,
            )
        )
        self._bump()

    def absorb_user_event(self, message: Message) -> bool:
        """Fold a live user ``MessageStartEvent``. The session emits these for
        user turns now, so a prompt from ANY front end reaches the fold — the
        TUI→phone direction that was missing. Returns True when it added the
        row; False when the row was already there (the handle's optimistic
        ``note_user_message`` echo for a phone-sent prompt), so the same
        message never appears twice on the phone.

        De-duped against an explicit registry keyed by MESSAGE ID, not by a
        scan of the transcript tail (issue #231). A phone STEER is announced
        only when the drain takes it, at a later tool boundary, so assistant
        and tool rows land between the echo and its event and push the echo out
        of any fixed-size window — the three-entry scan this replaces repainted
        the steer the moment that happened. The id is exact and order-free,
        which also fixes the mirror-image failure the window had: a genuinely
        distinct message whose words collide with a recent row was swallowed.

        The in-place UPGRADE is preserved and load-bearing: a phone-sent prompt
        is echoed WITHOUT image refs (the handle holds only the wire images),
        so when the real event arrives carrying the attachments the echoed row
        takes the message's id and refs rather than being skipped — otherwise
        the thumbnails never appear for the sender.
        """
        if not isinstance(message, Message) or message.role != "user":
            return False
        text = message.text
        refs = _image_refs(message)
        # Echo reconciliation runs BEFORE the envelope branch, not after. A
        # message whose text matches the envelope shape can still be one the
        # phone sent (a user quoting the wrapper, or a phone-issued steer);
        # taking the envelope branch first left that echo unreconciled, so the
        # row double-rendered under one id — and the surviving echo displayed
        # the raw XML this whole path exists to suppress. Resolving the echo
        # once, here, also keeps the two arms from growing separate copies of
        # the registry/tail-scan rules.
        entry = self._pending_user_echoes.pop(message.id, None)
        if entry is None:
            # No registered echo: either this message came from ANOTHER surface
            # (the TUI→phone direction this method exists to paint), or the
            # handle had no id to register. Fall back to the historical tail
            # scan ONLY for rows the registry does not own, so a legacy handle
            # keeps its de-dup while a row with an exact event still coming is
            # never consumed by a colliding neighbour's words.
            owned = {held.id for held in self._pending_user_echoes.values()}
            for candidate in reversed(self.projection.transcript[-3:]):
                if (
                    candidate.kind in ("user", "steer")
                    and candidate.text == text
                    and candidate.id not in owned
                ):
                    entry = candidate
                    break
        parent_message = extract_parent_message(text)
        if entry is not None:
            # Adopt the persisted identity either way: the row was keyed by the
            # command id (or a synthetic one), and the message id is what later
            # history folds and the web client's list reconciliation agree on.
            entry.id = message.id
            if parent_message is not None:
                # The echo captured the model-facing envelope verbatim. Rewrite
                # it in place to the parent's own words so the reconciled row
                # matches what the durable folds will render for it later.
                entry.kind = "parent_message"
                entry.text = parent_message.body
            if refs and not entry.images:
                entry.images = refs
            return False
        if parent_message is not None:
            # A hub steer delivered mid-turn announces itself as a user
            # MessageStartEvent whose text is the model-facing envelope. The
            # phone renders the parent's own words, never the XML — the same
            # rule the durable folds apply, so a live delivery and its later
            # history page agree about the row.
            self._append(
                TranscriptEntry(
                    id=message.id, kind="parent_message", text=parent_message.body, final=True
                )
            )
            return True
        self._append(
            TranscriptEntry(id=message.id, kind="user", text=text, images=refs, final=True)
        )
        return True

    def note_prompt_rejected(self, reason: str) -> None:
        """A quiet notice that a phone prompt did NOT land (the session
        rejected it — busy or compacting). The user row is never echoed, so
        without this the tap would look like it vanished into nothing."""
        self._append(
            TranscriptEntry(
                id=f"rej-{time.time_ns()}",
                kind="notice",
                text=_compact(f"not sent: {reason}", 200),
            )
        )
        self._bump()

    # -- working line (TUI WorkingBlock's phone counterpart) -------------------

    def _set_activity(
        self,
        label: str,
        *,
        phase: str,
        epoch: float | None = None,
        edge: bool = True,
        restart_clock: bool = False,
    ) -> None:
        """Update the working line's label and date the phase it belongs to.

        The clock is (re)computed when the PHASE changes, when ``restart_clock``
        forces one, and on the fold's first label; a same-phase relabel keeps
        the instant it already holds. That is the TUI's rule — ``WorkingBlock``'s
        phase clock moves when the phase moves, so the elapsed number is the
        PHASE's age — and it is why a batch's second announcement or an intent
        revised mid-call no longer restarts a number the producer never
        restarted either.

        ``phase`` is the producer's own word for the kind of work this label
        names (``ACTIVITY_PHASE_*``, imported from ``session/frontend_state`` so
        both surfaces match the same strings). ``epoch`` is an instant a
        producer STATED for this very work — the ``tool_execution_start``
        event's ``started_at_epoch`` — and it outranks every inferred instant:
        for a running call the call's own start is the finer anchor the phase
        fold deliberately does not supply, since a batch's phase edge is its
        FIRST call's start and a narrowed label must not report a shed
        sibling's age (the TUI's D9).

        ``edge`` says whether THIS event is the phase's beginning as far as this
        fold can tell. When it is, the fold's own instant for it is a TRUE zero
        — the TUI's widget does the same with its phase clock, and the producer
        restamps at those events for the phases it re-zeroes — so the phone
        paints ``0s`` from the first frame and counts up. When the fold joined a
        phase already in flight, ``_activity_anchor`` decides between the
        producer's own instant and NO clock at all.

        ``restart_clock`` marks an edge this fold OBSERVED for itself, so its
        instant is now — a turn boundary, a tool finishing — and any attach-time
        reading for that phase is dropped rather than allowed to date it.
        """
        p = self.projection
        if restart_clock:
            self._drop_attach_anchor()
        if restart_clock or phase != self._activity_phase or self._activity_started_at is None:
            self._activity_started_at = self._activity_anchor(phase, epoch, edge=edge)
        self._activity_phase = phase
        p.activity = label
        if self._activity_started_at is None:
            # No instant to date this phase from: publish the ABSENCE rather than
            # a zero, which is the whole point of the nullable field.
            p.activity_started_s = None
        else:
            self.redate_from_phase()

    def redate_from_phase(self) -> None:
        """Re-date the published band age from THIS fold's phase instant.

        ``activity_started_s`` is otherwise a number written when the phase last
        moved, and the long phases are exactly the ones with no band event in
        them: prose streams deltas that do not re-enter the label's arm, and a
        running call has no arm at all. A frame built mid-phase therefore
        carried the age as of the last edge, and a viewer that attached on it
        painted that stale number — at a known zero, ``0s`` counting from its
        own mount, which is the operator's reported defect rendered as a
        fabricated zero on the one surface this change is about (review round 3,
        MAJOR 1).

        Called at every frame build (the runtime's push, the handle hand-off)
        rather than on a timer: those are the moments the number becomes an
        answer to "how long has this been going". The client needs no change — a
        fresher number simply re-seeds it.

        A fold that has never adopted a phase writes NOTHING here. Ownership is
        the rule, and it is load-bearing: the projection object can be SHARED
        with the fold that does own the age — the runtime builds its own
        ``ProjectionFold`` over the handle's seed, and only the handle's fold is
        ever fed events — so writing this fold's empty state over it would erase
        a live age on every frame. That is exactly what review round 4's blocker
        measured: a phone attached to the runtime was served ``null`` for every
        phase, watched edges included. An unknown instant stays unknown because
        ``_set_activity`` publishes that absence itself, not because a reader
        clears it.
        """
        if self._activity_started_at is None:
            return
        self.projection.activity_started_s = round(time.monotonic() - self._activity_started_at, 1)

    def _activity_anchor(self, phase: str, epoch: float | None, *, edge: bool) -> float | None:
        """The instant to date ``phase`` from, or ``None`` when there is none.

        Three sources, in order of authority:

        1. an epoch the producer STATED for this work — the call's real start,
           from the process that ran it. This is also what dates a start event
           the fold sees LATE (a relayed stream, a redelivered seed) from the
           call rather than from the phone's arrival.
        2. the folded PHASE instant, while it is pending, matching, and a phase
           the producer HOLDS (:data:`_PHASE_ANCHOR_ADOPTABLE`) — how a phone
           that attached mid-prose or mid-dictation learns how long that work
           has been going, which no per-call stamp can answer because there is
           no call behind it.
        3. the edge ``edge`` names: an event this fold watched BEGIN a phase is
           that phase's zero, exactly as the TUI's widget zeroes its own phase
           clock on entry. This is the source that makes the ordinary phone
           tick, and the one the previous head got wrong by spending it on
           phases this fold had joined late.

        ``None`` is the fourth answer, and a real one: a phase this fold joined
        mid-flight, whose producer stated no instant, and which no event of this
        fold's has dated. The label is then published WITHOUT a clock rather
        than with a number counted from the attach — carried on the wire as
        ``activity_started_s = None``, which the phone renders by withholding the
        digits and keeping the reserved cells.
        """
        stated = _stated_epoch(epoch)
        if stated is not None:
            return monotonic_from_epoch(stated)
        if (
            self._attach_phase_at is not None
            and phase in _PHASE_ANCHOR_ADOPTABLE
            and self._attach_phase == phase
        ):
            anchor = self._attach_phase_at
            # One-shot, exactly like the epoch path: the anchor is an AGE
            # converted once, and every tick after this counts on the monotonic
            # clock rather than re-reading a wall-clock stamp on each repaint.
            self._drop_attach_anchor()
            return anchor
        return time.monotonic() if edge else None

    def _drop_attach_anchor(self) -> None:
        """Retire the attach-time phase reading.

        Called when it is spent (adopted by a matching label) and when a phase
        edge has just been observed LIVE: the producer restamps its phase at
        exactly those events, so a reading folded before attach describes a
        phase that is already over. Retiring it is how one turn's attach cannot
        date the next turn's work; the clock itself is never moved here — a
        running counter is only ever moved by the code that owns its edge.
        """
        self._attach_phase = ""
        self._attach_phase_at = None

    def _derive_activity(self, event: AgentEvent) -> None:
        """The label the TUI's WorkingBlock would show for this event, from the
        SAME rule: a running tool's intent, else the stream phase, else
        "thinking". Empty once the turn settles.

        Each label carries the phase it belongs to, which is what lets
        ``reconcile_clocks``' attach-time anchor be adopted only where it is the
        same kind of work.
        """
        p = self.projection
        if isinstance(event, AgentStartEvent):
            # A new turn: every phase in it is observed live from here, so the
            # attach-time reading (the PREVIOUS turn's by definition) goes.
            self._set_activity(
                ACTIVITY_PHASE_THINKING, phase=ACTIVITY_PHASE_THINKING, restart_clock=True
            )
            return
        if isinstance(event, AgentEndEvent):
            # The one true terminal: the run is over, clear the working line.
            p.activity = ""
            p.activity_started_s = None
            self._activity_started_at = None
            self._activity_phase = ""
            # No turn is in flight any more, so there is no phase for an
            # attach-time reading to date — and none is ever inherited by the
            # next turn.
            self._drop_attach_anchor()
            return
        if isinstance(event, TurnEndEvent):
            # A per-model-turn boundary, NOT the end of the run (loop.py ~589
            # yields it after every assistant turn that made tool calls). The
            # run is now waiting on the next model call, exactly like the gap
            # after a tool finishes — show "thinking", restart the clock for the
            # wait. Clearing it here blanked the working line mid-run; only
            # AgentEndEvent settles the turn.
            self._set_activity(
                ACTIVITY_PHASE_THINKING, phase=ACTIVITY_PHASE_THINKING, restart_clock=True
            )
            return
        if not p.streaming:
            return
        if isinstance(event, ToolCallComposeEvent):
            # The same three endings as the row above, because the activity line
            # is the other place that can go on saying "dictating" for a call
            # the model finished writing: the terminal frame means the wait is
            # now on the harness (the call is queued), and a never-run verdict
            # means the wait is over and the call is not coming.
            #
            # The two terminal arms are dated ``queued``, a phase the producer
            # never folds and whose clock the TUI withholds outright — "there is
            # no instant a 'waiting to run' age could honestly count from"
            # (``frontend_state``'s own words beside the phase constants). An
            # ``edge`` of ``False`` is that refusal: the ``queued`` phase is also
            # outside ``_PHASE_ANCHOR_ADOPTABLE``, so neither an attach-time
            # reading nor this fold's own arrival can date these labels, and the
            # band shows no number until the call starts or the phase moves on.
            if event.not_run_reason:
                self._set_activity(event.not_run_reason, phase=ACTIVITY_PHASE_QUEUED, edge=False)
            elif event.dictation_complete:
                self._set_activity(
                    f"waiting to run {event.tool_name}",
                    phase=ACTIVITY_PHASE_QUEUED,
                    edge=False,
                )
            else:
                # A dictation's own edge is its batch's FIRST announcement: the
                # producer folds one zero per batch and does not restamp on the
                # later ones, so this fold dates the label from the edge it
                # watched and from the producer's folded instant when it joined
                # the batch late — never from a later announcement, which would
                # be the batch's age reset by a name change.
                self._set_activity(
                    event.intent or f"dictating {event.tool_name}",
                    phase=ACTIVITY_PHASE_COMPOSING,
                    edge=(
                        self._activity_phase not in ("", ACTIVITY_PHASE_COMPOSING)
                        or self._attach_phase not in ("", ACTIVITY_PHASE_COMPOSING)
                    ),
                )
        elif isinstance(event, ToolExecutionStartEvent):
            # The call's OWN start when the producer stated one: a start event
            # that reaches this fold late — the fold was built at attach, the
            # stream was relayed, the event was redelivered — dates the row and
            # the band from the call rather than from the phone's arrival.
            #
            # With NO epoch the start event is the phase's edge only if this
            # fold watched the phase begin. A producer that states no epoch (an
            # older runtime) plus a call already in flight at attach would
            # otherwise publish the fold's own arrival as a KNOWN zero, and the
            # band would count `0s` up from the phone's mount while the call's
            # own row withholds its duration in the same frame — design round
            # 3's D7, the D4 class inverted. The `running` phase is not an
            # adoptable one (the batch's phase edge is its FIRST call's start,
            # D9), so an unwatched start with no epoch publishes no clock at all.
            self._set_activity(
                event.intent or f"running {event.tool_name}",
                phase=ACTIVITY_PHASE_RUNNING,
                epoch=event.started_at_epoch,
                edge=(
                    self._activity_phase not in ("", ACTIVITY_PHASE_RUNNING)
                    or self._attach_phase not in ("", ACTIVITY_PHASE_RUNNING)
                ),
            )
        elif isinstance(event, ToolExecutionEndEvent):
            # Back to waiting on the model: restart the clock for the gap.
            self._set_activity(
                ACTIVITY_PHASE_THINKING, phase=ACTIVITY_PHASE_THINKING, restart_clock=True
            )
        elif isinstance(event, MessageStartEvent):
            # A model call is in flight with nothing streamed yet: "thinking",
            # not "responding". The loop yields this from a placeholder at the
            # top of EVERY provider call, before the first token, and a
            # tool-only turn never produces a text delta after it — so keying
            # "responding" here claimed prose for every model call. The TUI's
            # WorkingBlock says "responding" only once its streaming block is
            # mounted, which happens on the first non-empty delta below.
            #
            # No forced restart, and that is the producer's rule rather than an
            # omission: its fold restamps ``thinking`` at every ``message_start``
            # (``frontend_state._fold_activity_phase``), so the number this fold
            # publishes from here — its own arrival — is the same zero the
            # producer holds. An attach-time reading is never adopted for this
            # label and could not honestly be: ``thinking`` is re-zeroed by
            # every event that can carry it, so any folded instant predates the
            # phase the label names (``_PHASE_ANCHOR_ADOPTABLE``).
            if isinstance(event.message, Message) and event.message.role == "assistant":
                self._set_activity(ACTIVITY_PHASE_THINKING, phase=ACTIVITY_PHASE_THINKING)
        elif isinstance(event, MessageUpdateEvent):
            # The first text delta is the transition to prose. Only from the
            # model-wait: a running tool's intent outranks narration arriving
            # beside it, and once "responding" it stays until the phase ends.
            #
            # The prose edge is a delta only when the model-wait preceded it —
            # the producer's own rule is ``into(RESPONDING)`` from ``THINKING``
            # — so a fold built mid-prose either adopts the producer's instant
            # for the phase it is already in or publishes no clock. Its first
            # delta after such an attach is the middle of a phase, not its
            # beginning, and dating it from the attach is the fabricated zero
            # this whole path exists to remove.
            if event.delta and p.activity in (ACTIVITY_PHASE_THINKING, ""):
                self._set_activity(
                    ACTIVITY_PHASE_RESPONDING,
                    phase=ACTIVITY_PHASE_RESPONDING,
                    edge=(
                        self._activity_phase == ACTIVITY_PHASE_THINKING
                        or self._attach_phase == ACTIVITY_PHASE_THINKING
                    ),
                )
        elif isinstance(event, MessageEndEvent):
            # Ends the PROSE phase only. `message_end` closes the model call,
            # but for a tool-calling turn it arrives AFTER the compose events
            # and BEFORE `tool_execution_start` — with the approval gate's wait
            # in between for a write/exec-tier call. The composed intent set
            # above must survive that window: the TUI's `_composing_cards` are
            # only adopted on tool start or cleared at turn settle, so its
            # working line keeps saying `composing …` across `message_end`
            # while only the streaming block unmounts. Downgrading anything but
            # "responding" here said "thinking" for the whole approval wait.
            #
            # The producer restamps its phase at this edge (the fold's
            # ``message_end`` arm), so the clock is restarted here too rather
            # than left to the label change alone.
            if (
                isinstance(event.message, Message)
                and event.message.role == "assistant"
                and p.activity == ACTIVITY_PHASE_RESPONDING
            ):
                self._set_activity(
                    ACTIVITY_PHASE_THINKING, phase=ACTIVITY_PHASE_THINKING, restart_clock=True
                )

    # -- todos / pending / state -------------------------------------------

    def set_subagent_details(self, comms: Any) -> None:
        """Project descendant metadata without touching child transcripts.

        Only direct children emit lifecycle events through the root session;
        nested children still live in the shared comms registry. This method is
        called for every root event, so it is deliberately restricted to the
        in-memory registry. Child history and attachment hydration belongs to
        ``TuiSessionHandle``'s worker path.

        What it reads is ONE :meth:`SubagentComms.roster_pass`: the roster rows,
        the nodes and the job row behind each node all come off a single linear
        walk. Walking the registry three times here — and ``nodes()`` was itself
        quadratic, see ``RosterPass`` — was the per-event cost this removes; a
        fourth walk added below would put it straight back.
        """
        read = comms.roster_pass()
        roster = {item.job_id: item for item in read.roster()}
        nodes = read.nodes()
        by_id = {node.job_id: node for node in nodes}
        children: dict[str | None, list[Any]] = {}
        for node in nodes:
            children.setdefault(node.parent_job_id, []).append(node)
        ancestor_nodes: dict[str, list[Any]] = {}

        def ancestors(job_id: str) -> list[Any]:
            """Root-to-parent lineage from the local maps (O(children), no I/O).

            Cycle-safe with the same stop-on-repeat contract as
            ``SubagentComms.ancestors``: a legacy snapshot may hold self or
            multi-node parent cycles, so a plain recursive walk would recurse
            forever. Building it here from ``by_id`` keeps the metadata path off
            any per-child comms lineage recomputation.
            """
            cached = ancestor_nodes.get(job_id)
            if cached is not None:
                return cached
            lineage: list[Any] = []
            seen = {job_id}
            parent_id = by_id[job_id].parent_job_id
            while parent_id is not None and parent_id not in seen:
                seen.add(parent_id)
                parent = by_id.get(parent_id)
                if parent is None:
                    break
                lineage.append(parent)
                parent_id = parent.parent_job_id
            lineage.reverse()
            ancestor_nodes[job_id] = lineage
            return lineage

        for node in nodes:
            job = read.job(node.job_id)
            lifecycle = roster.get(node.job_id)
            row = self._subagents.get(node.job_id)
            if row is None:
                row = SubagentRow(job_id=node.job_id, label=node.label)
                self._subagents[node.job_id] = row
            else:
                row.label = node.label
            row.parent_job_id = node.parent_job_id
            row.session_id = node.session_id
            # Compacted preview only — see SUBAGENT_PROMPT_PREVIEW_CHARS. The
            # full launch prompt is recoverable from the child transcript (its
            # launch user message, resolved by launch_message_id), which the
            # phone fetches lazily; carrying it uncapped in every repaint scales
            # the frame with roster depth.
            row.prompt = _compact(node.prompt or "", SUBAGENT_PROMPT_PREVIEW_CHARS)
            row.launch_message_id = node.launch_message_id
            row.effort = node.effort
            # Preserve #298's ancestor_ids feature on the O(children) path.
            lineage = ancestors(node.job_id)
            row.ancestors = [ancestor.label for ancestor in lineage]
            row.ancestor_ids = [ancestor.job_id for ancestor in lineage]
            row.child_ids = [child.job_id for child in children.get(node.job_id, [])]
            row.peer_ids = [
                peer.job_id
                for peer in children.get(node.parent_job_id, [])
                if peer.job_id != node.job_id
            ]
            row.agent = str(getattr(job, "agent_role", None) or node.agent_role or "task")
            row.model_label = str(getattr(job, "model_label", None) or "")
            if lifecycle is not None:
                # SubagentComms owns the merge between the live manager row and
                # its durable record. Consuming that resolved view here prevents
                # a swept/reconnected child from falling back to SubagentRow's
                # running default, and prevents stale live rows from reopening a
                # terminal outcome during the runner/manager settle window.
                status = lifecycle.status
                if status in ("running", "queued", "starting"):
                    mobile_status = "running"
                elif status in ("paused", "pausing"):
                    mobile_status = "parked"
                elif status in ("interrupted", "gone"):
                    mobile_status = "cancelled"
                else:
                    mobile_status = status
                row.status = mobile_status  # type: ignore[assignment] -- normalized literals
                # Bounded for the wire (a roster preview pushed ~30x/s), but the
                # two outcome fields are recovered differently on the phone, so
                # their caps differ. ``result_text`` is a PREVIEW: it is the
                # child's own last assistant message and appears verbatim in the
                # child transcript, which the phone fetches in full lazily from
                # ``/api/sessions/{sid}/agents/{job_id}/history`` — so a short cap
                # loses nothing. ``error_text`` is ``str(exc)`` from the parent
                # runner and is NEVER in the transcript; the wire value is the
                # ONLY copy the Outcome panel renders, so it is carried
                # GENEROUSLY (see SUBAGENT_ERROR_CHARS) or the failure tail is
                # lost with no recovery path. Newlines preserved on both so a
                # multi-line handoff or stack trace stays legible.
                row.result_text = _compact_multiline(
                    str(lifecycle.result_text or ""), SUBAGENT_OUTCOME_CHARS
                )
                row.error_text = _compact_multiline(
                    str(lifecycle.error_text or ""), SUBAGENT_ERROR_CHARS
                )
                if lifecycle.age_s is not None:
                    row.elapsed_s = max(0.0, float(lifecycle.age_s))
            details = getattr(job, "latest_details", None)
            progress = str(details.get("progress") or "") if isinstance(details, Mapping) else ""
            row.progress = progress if row.status == "running" else ""
            row.activity = row.progress or ("thinking" if row.status == "running" else "")
            # NOTE: no ``Transcript`` construction here. #298's full-screen
            # subagent conversation still gets its history and todos, but they
            # are hydrated OFF the Textual loop by ``TuiSessionHandle`` and
            # published through ``set_subagent_hydrated_details`` below. Reading
            # every child transcript synchronously on each root event was the
            # freeze this change removes.
        self._sync_subagents()
        self._bump()

    def set_subagent_hydrated_details(
        self,
        job_id: str,
        transcript: list[TranscriptEntry],
        todos: list[dict[str, Any]],
    ) -> bool:
        """Publish one worker-hydrated child without rebuilding the roster."""
        row = self._subagents.get(job_id)
        if row is None:
            return False
        # A subagent transcript is NEVER placed on the wire. The projection is a
        # full repaint pushed to the daemon on every folded event (~30x/s during
        # a turn), and the daemon's control-socket reader caps a single frame at
        # 1 MB (``daemon._dial`` ``limit=1 << 20``). Embedding even a tail-capped
        # child transcript per subagent — multiplied across a deep roster — blew
        # past that cap, so every push was skipped as an oversized frame and the
        # phone silently fell back to the stale durable disk fold (the real-time
        # regression this fixes). Todos stay on the wire (small, and needed for
        # the live working line); the transcript is fetched lazily on demand from
        # ``/api/sessions/{sid}/agents/{job_id}/history``, which reads the child's
        # own on-disk transcript. ``transcript`` is intentionally left empty here;
        # the argument is kept for signature/call-site compatibility.
        _ = transcript  # deliberately not projected — see comment above
        row.todos = self._todo_phases(todos)
        self._sync_subagents()
        self._bump()
        return True

    @staticmethod
    def _todo_phases(phases: list[dict[str, Any]]) -> list[TodoPhase]:
        from local_operator.tools.builtin import _as_phases

        return [
            TodoPhase(
                name=str(phase.get("name", "")),
                items=[
                    TodoItem(
                        text=item.get("text", ""),
                        status=item.get("status", "pending"),  # type: ignore[arg-type]
                        reason=item.get("reason", ""),
                    )
                    for item in phase.get("items", [])
                ],
            )
            for phase in _as_phases(phases)
        ]

    def set_todos(self, phases: list[dict[str, Any]]) -> None:
        """Refresh the todo list from the tool store. Called by the runtime
        after every event batch: the store is the only writer, so re-reading
        it is the fold — there is no todo event to listen for.

        The store is PHASED (``builtin.TODO_STORE`` holds
        ``list[TodoPhase]``), so the argument is a list of phase dicts
        ``{"name", "items":[{"text","status"[,"reason"]}]}``. It is run through
        ``builtin._as_phases`` defensively: the same coercion every store
        reader uses, so a hand-attached legacy flat list still projects as one
        implicit ``"Todos"`` phase instead of rendering empty-text rows (the
        bug this fold had when it iterated phase dicts as items)."""
        # Imported at call time, not module top: keeps the mobile wire layer
        # free of a hard import-time dependency on the tools package, which the
        # registrant startup path deliberately avoids paying for.
        from local_operator.tools.builtin import _as_phases

        self.projection.todos = self._todo_phases(_as_phases(phases))
        self._bump()

    def set_pending(self, pending: PendingRequest | None) -> None:
        """Replace the whole pending queue with zero or one request.

        The TUI-mirror handle (:class:`~local_operator.mobile.tui_handle`)
        uses this: the terminal owns approval serialization, so the phone only
        ever mirrors the ONE card the TUI shows. Owned sessions use the
        request-identified :meth:`push_pending`/:meth:`pop_pending` instead,
        because their gates resolve concurrently and a bare ``None`` cannot say
        WHICH one settled.
        """
        self._pending_queue = [pending] if pending is not None else []
        self._sync_pending()

    def push_pending(self, pending: PendingRequest) -> None:
        """Enqueue a request behind any already waiting; show the front one.

        A tool batch can open two write/exec approvals at once (``shared``
        tools run in parallel — see harness.loop._execute_tool_calls). Each
        gate calls this from its own task, so without a queue the second card
        overwrote the first and the first tool hung forever with no way to
        answer it. FIFO: the phone answers the oldest wait first, and the next
        surfaces on the repaint that clears it.
        """
        self._pending_queue.append(pending)
        self._sync_pending()

    def pop_pending(self, request_id: str) -> None:
        """Remove a settled (or timed-out) request by id and re-front the rest.

        Identified by id, not position: concurrent gates settle in whatever
        order the user answers or a timeout fires, which is not the order they
        were enqueued."""
        self._pending_queue = [req for req in self._pending_queue if req.request_id != request_id]
        self._sync_pending()

    def _sync_pending(self) -> None:
        """Project the queue onto the wire fields the phone renders: the front
        request as ``pending`` plus the total ``pending_count`` for the "1 of
        N" badge."""
        self.projection.pending = self._pending_queue[0] if self._pending_queue else None
        self.projection.pending_count = len(self._pending_queue)
        self._bump()

    def reconcile_streaming(self, is_streaming: bool) -> None:
        """Align ``streaming`` with the session's own ``is_streaming`` flag at
        the moments the fold cannot derive it from events alone: initial attach
        (a phone subscribing mid-turn never witnessed the ``agent_start``) and
        command boundaries (a prompt/abort/new/resume just changed turn state).

        Once the fold has folded a turn-terminal event (``_streaming_ended``),
        the fold is authoritative and reconcile does nothing: the session's
        ``is_streaming`` is briefly, lyingly still True on the abort/error path
        (it emits ``agent_end`` INLINE and clears the flag several awaits later,
        in the turn's ``finally``), so honouring the flag in that window is the
        exact bug this guard prevents — a command landing there would re-stick
        the phone on "in progress" with no later event to correct it. Before any
        terminal — a fresh attach that missed ``agent_start`` mid-turn, or a
        turn whose end the fold has not observed — the flag is the only truth
        there is, so reconcile trusts it in both directions. A later
        ``agent_start`` clears the latch so the next turn reconciles normally.
        """
        if self._streaming_ended:
            return
        if self.projection.streaming != bool(is_streaming):
            self.projection.streaming = bool(is_streaming)
            self._bump()

    def reconcile_clocks(self, session: Any) -> None:
        """Adopt the producer's start instants ONCE, at attach.

        The fold dates every clock from the events it observes, which is
        exactly wrong for a projection BUILT at attach: it never witnessed the
        ``agent_start``, the ``tool_execution_start`` or the ``tool_call_compose``
        that began the work in flight, so the first event it does see dates that
        work from the phone's own arrival. That is the operator's report — a
        live band reading ``0s`` and counting up from the moment the phone
        attached — and it is the same defect the TUI's cards and working block
        were fixed on by seeding from these very anchors
        (``tui/widgets/tool_card.py`` ``monotonic_from_epoch``,
        ``OperatorApp._current_activity``).

        The two anchors are read from the session, not from a second fold: both
        real session shapes answer them (``Session`` and ``AttachedSession``,
        declared in ``session/protocol.py``) and the PRODUCER is the process
        running the work, so its instants are the executor's own rather than a
        reconstruction. ``live_tool_start_epochs()`` dates calls that are in
        flight; ``activity_phase_clock()`` dates the phases with no call behind
        them (``thinking``/``responding``), which is the half a per-call stamp
        cannot answer.

        Both reads are probed and every refusal here is load-bearing:

        * a source that cannot answer at all — a reduced facade, a legacy
          producer, an embedder with no fold — seeds nothing, and today's
          behaviour stands rather than an exception being raised on attach. Both
          reads are SHAPE-checked as well as probed (``_phase_pair`` for the
          phase pair, ``_live_call_map`` for the call map): a host that has the
          member but answers the wrong shape would otherwise raise here, and
          this call sits on the unattended attach path
          (``RuntimeServer._serve``) where a raise ends the session for every
          viewer;
        * a call present in the map with ``None`` states that it STARTED
          without stating when, so NO entry is seeded for it: its end event
          then measures what it measured before rather than inheriting an
          instant this fold invented;
        * the phase instant is stored PENDING, never applied. A label asks for
          it through ``_activity_anchor``, which hands it over only when the
          phase it is about to display EQUALS the folded one AND that phase is
          one the producer HOLDS rather than re-zeroes on every event of its
          kind (:data:`_PHASE_ANCHOR_ADOPTABLE` — ``responding`` and
          ``composing``). A mismatch — the phone about to say ``responding``
          while the producer is still mid-``thinking``, a compaction fallback, a
          facade answering ``("", None)`` — adopts nothing, because one phase's
          zero under another phase's label is a wrong number where a blank one
          would be honest. ``FrontendStateStore.activity_phase_clock`` exists,
          and takes its two fields in one call, for the same reason.

        Entries seeded for live calls live exactly as long as an observed one:
        the call's own end event pops it, and a later start under a reused id
        overwrites it, so a seeded instant can never date a different call.
        """
        phase_clock = getattr(session, "activity_phase_clock", None)
        if callable(phase_clock):
            phase, phase_started_at = _phase_pair(phase_clock())
            self._attach_phase = phase
            stated = _stated_epoch(phase_started_at)
            self._attach_phase_at = monotonic_from_epoch(stated) if stated is not None else None
        starts = getattr(session, "live_tool_start_epochs", None)
        if not callable(starts):
            return
        for tool_call_id, epoch in _live_call_map(starts()).items():
            stated = _stated_epoch(epoch)
            if stated is None:
                continue
            # ``setdefault``, not assignment: a call this fold already watched
            # start keeps the instant it observed, so adopting an anchor can
            # never move a counter that is already running.
            self._tool_started_at.setdefault(str(tool_call_id), monotonic_from_epoch(stated))

    def set_state(
        self,
        *,
        model_label: str | None = None,
        model_selector: str | None = None,
        effort: str | None = None,
        effort_ladder: list[str] | None = None,
        conversation_name: str | None = None,
        cwd: str | None = None,
        queued_count: int | None = None,
        streaming: bool | None = None,
    ) -> None:
        p = self.projection
        if model_label is not None:
            p.model_label = model_label
        if model_selector is not None:
            p.model_selector = model_selector
        if effort is not None:
            p.effort = effort
        if effort_ladder is not None:
            p.effort_ladder = effort_ladder
        if conversation_name is not None:
            p.conversation_name = conversation_name
        if cwd is not None:
            p.cwd = cwd
        if queued_count is not None:
            p.queued_count = queued_count
        if streaming is not None:
            p.streaming = streaming
        self._bump()

    # -- internals ----------------------------------------------------------

    def _reasoning_row(self, message_id: str, delta: str) -> None:
        """Fold one reasoning fragment onto this call's reasoning row.

        ONE row per model call, keyed by the message it belongs to, so a run of
        thousands of fragments cannot become thousands of transcript rows: the
        phone re-renders the whole projection on every repaint, and a row per
        token would be an unbounded wire cost for content that scrolls past.

        The row is inserted ABOVE the assistant row the same call already opened
        at ``message_start``. Appending would leave the empty assistant row on
        top and the thinking under it, so the answer would materialise above the
        reasoning that produced it — the TUI retires its block before the
        answer mounts for the same ordering reason (``app.py``).
        """
        entry_id = f"rz-{message_id}" if message_id else "rz"
        row = self._find(entry_id)
        if row is None:
            # ``final=False`` while it streams, exactly as the assistant row
            # does at ``message_start``: the phone's "is this row still moving"
            # question reads this flag. ``text_complete`` is deliberately left at
            # its default -- that flag means "this is a pageable PREFIX of a row
            # that exists in full elsewhere" (frame-cap truncation), and
            # reasoning is bounded here because there is nothing to page: it is
            # never persisted anywhere.
            row = TranscriptEntry(id=entry_id, kind="reasoning", final=False)
            self._append(row)
            self._open_reasoning_id = entry_id
            rows = self.projection.transcript
            anchor = next(
                (index for index, item in enumerate(rows) if item.id == self._open_message_id),
                None,
            )
            if anchor is not None:
                rows.insert(anchor, rows.pop(rows.index(row)))
        row.text = _reasoning_tail(row.text + delta)

    def _close_reasoning_row(self) -> None:
        """Seal the open reasoning row, so the next call's phase opens its own."""
        if self._open_reasoning_id:
            row = self._find(self._open_reasoning_id)
            if row is not None:
                row.final = True
            self._open_reasoning_id = None

    def _tool_row(self, tool_call_id: str, tool_name: str) -> TranscriptEntry:
        entry_id = self._tool_rows.get(tool_call_id)
        row = self._find(entry_id) if entry_id else None
        if row is None:
            row = TranscriptEntry(
                id=f"tc-{tool_call_id}",
                kind="tool",
                tool_call_id=tool_call_id,
                tool_name=tool_name,
            )
            self._append(row)
            self._tool_rows[tool_call_id] = row.id
        return row

    def _tool_details(
        self, args: dict[str, Any], output: str, result_details: dict[str, Any] | None
    ) -> dict[str, Any]:
        """The live path's expand payload — one implementation with replay's.

        Delegated rather than duplicated so a live row and the replayed row
        for the same call expand to the same thing; the caps here are the
        wire budget's and must not drift between the two paths.
        """
        return _tool_row_details(args, output, result_details)

    def _append(self, entry: TranscriptEntry) -> None:
        self.projection.transcript.append(entry)
        self.projection.transcript = self._cap_tail(self.projection.transcript)
        if self._pending_user_echoes:
            # Drop echo entries whose row the cap just trimmed. Two reasons,
            # both bounding: an entry pointing at a row no longer in the
            # projection can only upgrade something invisible, and a steer the
            # phone RECALLS is never announced, so without this its entry would
            # sit in the dict for the life of the session. A trimmed row's
            # event now paints at the tail, which is the honest rendering — the
            # row it would have upgraded is gone.
            live = {row.id for row in self.projection.transcript}
            self._pending_user_echoes = {
                message_id: row
                for message_id, row in self._pending_user_echoes.items()
                if row.id in live
            }

    @staticmethod
    def _cap_tail(entries: list[TranscriptEntry]) -> list[TranscriptEntry]:
        """Trim to the render tail WITHOUT losing the opening user message.

        A bare ``[-LIMIT:]`` drops the first user turn on any session longer
        than the cap — and the opening prompt is the one row that names what
        the whole conversation is about (and, per the field report, the row
        that was always missing). Keep the transcript's first user message
        pinned at the head, then fill the rest from the tail. The web client
        still pages older history on scroll; this is about the projection
        never omitting the conversation's own opening.
        """
        if len(entries) <= PROJECTION_TRANSCRIPT_LIMIT:
            return entries
        tail = entries[-PROJECTION_TRANSCRIPT_LIMIT:]
        first_user = next((e for e in entries if e.kind == "user"), None)
        if first_user is not None and first_user not in tail:
            # Pin the opener and make room by dropping the OLDEST tail row
            # (``tail[1:]``), never the newest (``tail[:-1]``). ``_cap_tail``
            # runs on EVERY append, so dropping ``tail[-1]`` here discarded the
            # row just appended — and did so on each subsequent append, which
            # froze the transcript: past the cap, no new tool call or message
            # ever reached the phone (the field report's "last several tool
            # calls I can't see"). Dropping the oldest keeps the bound (one
            # pinned + LIMIT-1 newest = LIMIT) while the newest row always
            # survives. Older rows page back in on scroll via the history API.
            return [first_user, *tail[1:]]
        return tail

    def _find(self, entry_id: str | None) -> TranscriptEntry | None:
        if not entry_id:
            return None
        for entry in reversed(self.projection.transcript):
            if entry.id == entry_id:
                return entry
        return None

    def _close_open_message(self) -> None:
        if self._open_message_id:
            row = self._find(self._open_message_id)
            if row is not None:
                row.final = True
            self._open_message_id = None

    def _sync_subagents(self) -> None:
        self.projection.subagents = sorted(
            self._subagents.values(),
            key=lambda row: (row.status != "running", row.job_id),
        )

    def _bump(self) -> None:
        self.projection.version += 1
