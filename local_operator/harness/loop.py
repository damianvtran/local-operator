"""The agent loop — a provider-agnostic engine with native tool calling.

Provider-agnostic engine with native tool calling (the ``runLoopBody`` core).
The loop knows NOTHING about sessions, persistence, or UI: everything host-side
arrives through :class:`~local_operator.harness.types.LoopConfig` callbacks,
and the only boundary outward is the :class:`AgentEvent` stream.

Structure is two nested while loops:

- **Outer** — re-enters when steering/asides/follow-ups arrive at the yield
  boundary (after the model has stopped asking for tools).
- **Inner** — runs while the last response carried tool calls or pending
  messages remain; drains pending messages, calls the model, executes tools.

Guards: tool errors go back to the model as ``is_error`` results (never
raise into the loop); dangling tool calls on error/abort/length get synthetic
placeholder results so tool_use/tool_result pairing stays legal.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import logging
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

from pydantic import TypeAdapter, ValidationError

from local_operator.ansi import sanitize_prompt_line
from local_operator.harness.approval import ask_approval
from local_operator.harness.intent import (
    INTENT_FIELD,
    INTENT_SCAN_LIMIT,
    intent_is_injected,
    sanitize_intent,
    scan_streaming_intent,
)
from local_operator.harness.types import (
    FAULT_INVALID_ARGUMENTS,
    FAULT_KEY,
    OUTPUT_LIMIT_ARGUMENTS,
    OUTPUT_LIMIT_KEY,
    OUTPUT_LIMIT_TURN,
    AbortSignal,
    AgentEndEvent,
    AgentEvent,
    AgentMessage,
    AgentStartEvent,
    AgentTool,
    AgentToolUpdate,
    ApprovalDescribeFn,
    Aside,
    ChatRequest,
    Content,
    CustomMessage,
    InvalidToolArgumentsError,
    LoopConfig,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ModelChangeEvent,
    ModelSpec,
    NoticeEvent,
    ProviderTurnStartEvent,
    RenderedStreamError,
    StaleAside,
    StreamEndEvent,
    StreamEvent,
    StreamModelEvent,
    StreamStartEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    StreamUsageEvent,
    TextContent,
    ToolCall,
    ToolCallComposeEvent,
    ToolContext,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolResult,
    TurnEndEvent,
    TurnStartEvent,
    Usage,
)
from local_operator.incidents import REASONING_ECHO_MARKERS

#: How often a still-composing tool call re-announces its size. Fast enough that
#: the byte counter visibly moves (so the row reads as progress rather than as a
#: frozen label), slow enough that a token-by-token argument stream cannot flood
#: the UI thread with repaints.
COMPOSE_NOTICE_INTERVAL_S = 0.2

logger = logging.getLogger(__name__)


class _ToolDone:
    """Sentinel pushed through the tool-event queue when one execution
    finishes. A dedicated class rather than a bare ``object()`` so the queue's
    element type stays exact and ``isinstance`` narrows the other branch to a
    real event."""

    __slots__ = ()


class _BatchDone:
    """Sentinel pushed ONCE, after every runner task in a batch has settled.

    It is what ends the drain, and it exists because ``_ToolDone`` cannot do
    that job under cancellation: a runner cancelled before its body ran emits
    no receipt at all, so counting receipts against the number of tasks can
    wait forever. This is posted by a closer that has already awaited the
    tasks, so its arrival is proof there is nothing left to come.
    """

    __slots__ = ()


_TOOL_DONE = _ToolDone()
_BATCH_DONE = _BatchDone()

# Type adapters used to validate tool arguments against JSON-schema scalars.
_TYPE_ADAPTERS: dict[str, TypeAdapter[Any]] = {
    "string": TypeAdapter(str),
    "integer": TypeAdapter(int),
    "number": TypeAdapter(float),
    "boolean": TypeAdapter(bool),
    "array": TypeAdapter(list),
    "object": TypeAdapter(dict),
}

ABORTED_RESULT_TEXT = "aborted"
SKIPPED_RESULT_TEXT = "Tool call skipped: interrupted by steering."
# What the model is told about a call the OUTPUT LIMIT kept from running, kept
# distinct from ``ABORTED_RESULT_TEXT`` on purpose rather than for style:
# "aborted" says a turn stopped and explains nothing about the state of the call
# the model now sees replayed in its own history. Measured on a real provider, a
# model handed that bare "aborted" reported that its call "came through empty
# and was aborted", declined to retry, and the large file it was asked to write
# was never written and nothing said so (QA round 1, Q2: base wrote 110,703
# chars, the truncated branch produced ``tool_executions: []``). The remedy is
# the model's to take, so the result has to name it.
#
# HOW it is named decides whether the model takes it. The first wording
# ("cut off at the output limit ... re-emit this call with a smaller payload")
# demanded the size reduction without the authority to make it: under an
# instruction to emit the whole content and not abbreviate, the model read
# "smaller payload" as a requirement the user had forbidden it to satisfy and
# answered in prose instead of re-issuing the call -- measured on deepseek-flash
# at `high` with the same 3,000-line write cell, 1/6 runs wrote any file against
# 6/6 for the bare ``ABORTED_RESULT_TEXT``, and the single-variable
# counterfactual on that tree flipped it to base's shape 2/2 (QA #1077, Q10;
# full sample on the PR). ``TRUNCATED_RESULT_TEXT`` therefore grants the
# reduction instead of demanding it: the limit bounds ONE CALL's arguments and
# is not a licence to shorten the ANSWER, so a payload that fits is what to send
# here. It grants it without naming an instruction it cannot see (the loop is
# not shown the conversation's own standing instructions, so it must not assert
# that the user prohibited anything: review F3).
#
# TWO ARMS, TWO TEXTS, and the difference is load-bearing. The length arm
# appends its placeholder to EVERY call in the turn, and not every one of them
# was cut: a turn can dictate a call to completion and then hit the cap while
# writing prose or a second call. Telling such a call its arguments "were larger
# than the output limit allows" and would "be cut again" asserts two things that
# are false there -- and measurably so, since re-issuing those identical 43-byte
# arguments on the next turn ran and wrote the file (review F1 == QA Q1). Only
# ``TRUNCATED_RESULT_TEXT`` may carry the size claim; a call whose arguments
# arrived complete takes ``LENGTH_ENDED_CALL_RESULT_TEXT``, which says the one
# thing that is true of it (the turn ended before it ran) and still tells the
# model to re-issue it. ``_limit_cut_arguments`` below draws that line.
#
# NEITHER text claims a file was written or not written. The Q2 clause was
# "no file was written", which is vacuous for a ``read``/``bash``/``grep`` call
# and would be a false statement of consequence for any tool that writes
# nothing (review F5); "the tool was never executed" is the same guarantee and
# holds for every tool.
#
# BOTH are MODEL-FACING, and that is now their only audience. The message they
# are carried on is the tool result the transcript persists, so a resumed card
# used to paint this prose as the operator's own receipt -- imperatives
# addressed to a model, about a file, on the user's screen (review F2). The row
# takes its words from ``harness/rows.output_limit_call_receipt`` instead, keyed
# off the ``OUTPUT_LIMIT_KEY`` marker riding in ``details``; edit these strings
# freely, and edit THAT vocabulary if what an operator reads should change.
TRUNCATED_RESULT_TEXT = (
    "this call did not run: the output limit cut it off before its arguments "
    "finished arriving, so the tool was never executed. The limit is a size "
    "bound on one call's arguments, not a licence to shorten the answer — a "
    "payload that fits within it is what to send here. Re-issue the call now; "
    "the identical oversize arguments will be cut again. Reply with the call "
    "itself, not with an explanation of why it cannot be sent."
)

# The other arm of the same limit: a call whose arguments arrived COMPLETE in a
# turn the limit ended before it could run. Nothing here was cut and nothing
# here is oversize, so this text carries no size claim and asks for no
# reduction -- shrinking the payload would be a change the model has no reason
# to make. What it does carry is the fact that the call did not run and will
# not, which is the Q2 guarantee, and the instruction to re-issue it unchanged
# (the identical arguments do execute on the next turn; review F1).
LENGTH_ENDED_CALL_RESULT_TEXT = (
    "this call did not run: the turn ended at the output limit before this call "
    "was run, and a call in a turn that ends that way is not executed. Its "
    "arguments arrived complete, so there is nothing here to shrink. Re-issue "
    "this call as it is. Reply with the call itself, not with an explanation of "
    "why it cannot be sent."
)

#: Cap on the reason carried by a never-run call's terminal compose frame.
#:
#: The frame rides the live relay and the reconnect seed, and the seed measures
#: every row it retains against ``LIVE_EVENT_TEXT_FRAME_BUDGET_CHARS``; the
#: synthetic result it is drawn from can hold an invalid-arguments dump of any
#: size. The unit is CODEPOINTS — ``len()`` on a ``str``, the clip below — and
#: it is the one bound on this wire NOT measured in terminal cells or serialized
#: bytes, so the worst case is worth stating rather than assuming: a CJK reason
#: is ~2 cells and ~3 UTF-8 bytes per code point, i.e. ~600 cells / ~800 bytes
#: at this cap. 200 code points is the first line of a diagnostic and nothing
#: more — enough for `Tool not found: <name>` or a JSON-validation complaint,
#: two orders of magnitude inside that 60 KB budget, and clipped rather than
#: dropped so the row can never say "something went wrong" where it could name
#: the thing.
NOT_RUN_REASON_MAX_CHARS = 200

# Why a tool call did not run cleanly, classified WHERE THE REASON IS KNOWN and
# carried on ``ToolResult.details["__fault"]`` to the one place that reports it
# (``park``). Deriving the class at ``park`` instead would mean text-matching
# result strings like "Invalid arguments: " — fragile, and wrong the first time
# a message is reworded. ``_synthetic_result`` already takes ``details`` for
# exactly this purpose, and ``__approval_gate_failed`` set the convention.
#
# The split that matters is whether the MODEL is at fault, because only that
# half is a measurement of the model. The first three are calls the harness
# refused to dispatch because what the model emitted was not usable; the rest
# are the user's decision, our own bug, or the world failing. Only the first
# three feed the tool-call validity figure — see ``analytics.model.MODEL_FAULTS``,
# which owns that classification for the read side.
#
# ``FAULT_KEY`` and ``FAULT_INVALID_ARGUMENTS`` are re-exported from
# ``harness.types`` rather than declared here: a tool BODY also needs to claim
# the invalid-arguments class (see ``InvalidToolArgumentsError``), and tool
# modules import ``harness.types`` while this module imports them indirectly.
# Importing them keeps one spelling for the value the ledger stores.
FAULT_UNKNOWN_TOOL = "unknown_tool"  # model named a tool that does not exist
FAULT_DUPLICATE_ID = "duplicate_id"  # model emitted one call id twice
FAULT_DENIED = "denied"  # the user declined the call
FAULT_GATE_FAILED = "gate_failed"  # our approval plumbing raised
FAULT_ABORTED = "aborted"  # the user stopped the turn
FAULT_SKIPPED = "skipped"  # steering redirected before this call ran
FAULT_EXECUTION = "execution"  # the tool ran and failed (HTTP 500, missing file)
# Backfill for empty tool results (coerceToolResult): Anthropic rejects an
# empty ``is_error`` tool_result content with a 400, and other providers
# serialize "" — one placeholder block keeps the wire legal for every client.
EMPTY_TOOL_RESULT_TEXT = "[tool returned no output]"
# Steering-interrupt poll interval for ``interruptible`` tools mid-run.
STEERING_INTERRUPT_POLL_S = 0.25

# Count identical all-error batches, not seconds or successful repeated reads.
# Polling and long tasks are legitimate; spending six model calls submitting
# unchanged failing arguments is not progress. One explicit recovery hint is
# supplied halfway through, so a model can change course before the guard ends
# the run with a diagnosable error. These are loop safety bounds, not task caps.
REPEATED_TOOL_ERROR_WARNING = 3
REPEATED_TOOL_ERROR_LIMIT = 6

#: How many times an EMPTY ``length`` truncation (no text, no tool calls — the
#: reasoning model that spent its whole output budget thinking) is retried one
#: effort rung lower before the turn ends the ordinary way. Two covers the
#: observed failure shape (a high rung, then the rung below it also silent)
#: without burning the user's quota on a model that cannot answer today.
MAX_EMPTY_TRUNCATION_RETRIES = 2

#: How many times ONE run will continue a turn that the network cut short —
#: the laptop closed at home and opened at work, which is minutes of no route
#: to anywhere.
#:
#: The provider layer already waits that reconnect out patiently
#: (``CONNECTIVITY_MAX_RETRIES`` = 15 attempts against the SAME target, a 60s
#: cap). This is the outer budget for the case that wait was not enough, and it
#: is deliberately small: each continuation buys another full patient wait.
#:
#: MEASURED, not estimated, by summing the real delay function over the jitter:
#: one patient wait averages ~7.9 min, so the initial attempt plus three
#: continuations tolerates ~32 min of outage (~29-34 min across the jitter
#: range) — past any plausible commute — while staying bounded, because a
#: genuinely dead network has to surface an error rather than pin the session in
#: a silent retry forever.
#:
#: Only a cut the provider layer certifies as RESUMABLE is continued, and there
#: are two families today. A CONNECTIVITY loss, where the machine is offline:
#: nothing was wrong with the request, the credential or the provider, so
#: re-asking is the whole fix. And an AGGREGATOR's in-band report that one of
#: its UPSTREAM hosts died mid-stream (``is_aggregator_upstream_stream_failure``):
#: a gateway is a router, so the next attempt is served by another of its hosts
#: — re-issuing the turn IS the failover, rather than a pass dying while it
#: holds a half-composed tool call.
#:
#: Everything else stays terminal exactly as before, and the distinction is the
#: one both predicates are built on: the provider DID answer about the request
#: it was given — a first-party 5xx, or a refusal, 4xx or in-band — so replaying
#: that turn would re-bill it while hiding a failure the user needs to see.
MAX_CONNECTIVITY_CONTINUATIONS = 3

#: What the loop tells the model after the network cut its answer short.
#:
#: Phrased as an instruction about what the USER can see, because that is the
#: constraint the model cannot infer: the partial text is in the transcript and
#: has already been read, so restarting the answer would show it twice. It is a
#: user-role message rather than a bare trailing assistant turn on purpose —
#: see the call site: prefilling is rejected by current Claude models.
CONNECTIVITY_CONTINUATION_PROMPT = (
    "[system] Your previous response was cut off mid-sentence by a network "
    "interruption. The partial text above has already been shown to the user. "
    "Continue it seamlessly from exactly where it stopped — do not repeat any "
    "of it, do not restart, and do not apologise or mention the interruption."
)

#: What the loop tells the model when the interruption landed inside a TOOL CALL.
#:
#: The prompt above assumes the cut fell in a SENTENCE, and its instruction is
#: to continue that sentence seamlessly. A call still being dictated when the
#: socket died is truncated JSON: the continuation branch drops it — it cannot
#: be run, and cannot be replayed faithfully — so the model's own record of
#: having chosen that action is erased. Leaving it to the prose prompt then asks
#: the model to continue a sentence it never finished while the action it had
#: already decided on is silently lost, which is exactly what happened to the
#: aborted ``write`` call in the incident. This instruction carries the one fact
#: the drop erased, names the tool so a multi-call turn is unambiguous, and is
#: written to stand alone — a cut that produced no prose at all gets it on its
#: own — as well as to follow the prose prompt above.
CONNECTIVITY_TOOL_CALL_CONTINUATION_PROMPT = (
    "[system] A tool call ({tools}) was aborted by the network interruption "
    "before it finished, so it never ran. If you still need that action, issue "
    "the call again from scratch."
)


def _continuation_instruction(*, resumable_text: bool, interrupted: list[ToolCall]) -> str:
    """The instruction appended to an interrupted turn, shaped to the cut.

    Both halves are independent, because the two cuts are: the interruption can
    land in prose, inside a tool call, or — the incident's shape — in prose with
    a call already being dictated. The prose half is added only when there is
    partial prose to continue (this is the same view of "text" the serializer
    takes; the whitespace-only guard in the continuation branch is what keeps
    the two from disagreeing), and the call half only when a call was cut off.

    A turn that has NEITHER is re-asked whole rather than continued, so it never
    reaches here: with nothing committed to history there is no partial answer
    to refer to and the retry is a clean re-ask.
    """
    parts: list[str] = []
    if resumable_text:
        parts.append(CONNECTIVITY_CONTINUATION_PROMPT)
    if interrupted:
        tools = ", ".join(sorted({call.name for call in interrupted}))
        parts.append(CONNECTIVITY_TOOL_CALL_CONTINUATION_PROMPT.format(tools=tools))
    return " ".join(parts)


#: The two fixed ends of `CONNECTIVITY_TOOL_CALL_CONTINUATION_PROMPT`, split
#: around its ``{tools}`` hole. Declared from the template itself so the
#: recogniser below cannot fall out of step with the string it recognises.
_TOOL_CALL_INSTRUCTION_HEAD, _, _TOOL_CALL_INSTRUCTION_TAIL = (
    CONNECTIVITY_TOOL_CALL_CONTINUATION_PROMPT.partition("{tools}")
)


def _is_continuation_tool_list(text: str) -> bool:
    """Whether ``text`` is a tools list ``_continuation_instruction`` would write.

    The producer writes ``", ".join(sorted({call.name for call in interrupted}))``,
    so the list is comma-and-space separated and every name is a single
    whitespace-free token. Checked rather than waved through because the
    predicate below is what keeps harness chrome out of the user's row: an
    operator message that merely QUOTES one of these instructions — a whole
    line, a multi-line paste, a wrapping that put a newline inside the tools
    list — must stay theirs. A tool name this grammar rejects costs only the
    pre-fix behaviour (the row paints), never a swallowed operator turn.
    """
    names = text.split(", ")
    return all(name and not any(char.isspace() for char in name) for name in names)


def is_connectivity_continuation_instruction(text: str) -> bool:
    """Whether ``text`` is EXACTLY one of the instructions this module mints.

    :func:`_continuation_instruction` returns three shapes, and the front ends
    have to recognise all three: the prose prompt alone, the tool-call prompt
    alone (a cut that produced no prose), and the two joined by a single space
    (the incident's shape — prose with a call still being dictated).

    Lives HERE, beside the strings it matches, rather than in the shared
    ``harness.rows`` decision that calls it, for two reasons. The composed shape
    cannot be enumerated: the tool-call half interpolates the aborted calls'
    names, so a table of exact strings would have to list every possible tools
    list. And the shape is a property of the producer, not of a row — the row
    decision is "does the harness mint this text", and this is how the harness
    answers that about its own words.

    Deliberately not a loose prefix test on ``[system] ``: that prefix is also
    minted by the approval and question gates (see
    ``rows._HARNESS_NOTICE_HEADS``), and an operator is free to type it. What is
    matched is the fixed sentence each half opens with, plus the exact tail the
    tool-call half closes with.
    """
    stripped = text.strip()
    if stripped == CONNECTIVITY_CONTINUATION_PROMPT:
        return True
    if stripped.startswith(CONNECTIVITY_CONTINUATION_PROMPT + " "):
        stripped = stripped[len(CONNECTIVITY_CONTINUATION_PROMPT) + 1 :]
    if not (
        stripped.startswith(_TOOL_CALL_INSTRUCTION_HEAD)
        and stripped.endswith(_TOOL_CALL_INSTRUCTION_TAIL)
    ):
        return False
    tools = stripped[
        len(_TOOL_CALL_INSTRUCTION_HEAD) : len(stripped) - len(_TOOL_CALL_INSTRUCTION_TAIL)
    ]
    return _is_continuation_tool_list(tools)


# How the loop recognises "this DeepSeek thinking-mode request never carried the
# reasoning back": matched on the provider's OWN words (it arrives as a plain 400
# beside every other malformed-request refusal, and is classified as one), using
# the markers defined in ``local_operator.incidents`` and imported rather than
# re-spelled here. The classifier that names the refusal for the USER and the
# gate below that decides whether to RETRY it are answering the same question,
# and while each held a private copy they combined the two halves differently
# (this one requiring both, the rule only one) -- so a 429 body quoting the field
# name and the legacy rows' "unsupported field" 400 were both reported to the
# user as our own recovery having failed.

#: Turns this RUN has re-sent with the reasoning echo FILLED after such a
#: refusal. One, not more: the fill is a property of the request shape, so a
#: second identical attempt would spend a call to be told the same thing.
MAX_REASONING_ECHO_FILL_RETRIES = 1

#: Turns this RUN has retried after such a refusal with thinking turned OFF.
#: One, not more, and SEPARATE from the fill budget above: a route can need the
#: echo and still refuse (or refuse for a second reason), and the retreat is
#: what recovers the ones where the echo alone was not the whole story. The
#: separate counters are what keep "one of each" true without letting either
#: alone loop.
MAX_REASONING_ECHO_RETRIES = 1


def _is_reasoning_echo_rejection(error: str | None) -> bool:
    """Did the provider refuse this request for a missing reasoning echo?"""
    if not error:
        return False
    text = error.lower()
    return all(marker in text for marker in REASONING_ECHO_MARKERS)


def _thinking_off_effort(model: "ModelSpec") -> str | None:
    """The ladder rung that switches thinking OFF for a model that supports it.

    ``None`` when the model's ladder has no such rung, or when it is already on
    it -- a retry that cannot change the body is a wasted provider call, so the
    caller ends the turn instead. Reading the rung off the model's own ladder
    rather than inventing a wire key keeps this loop free of provider knowledge:
    the effort levels are shared vocabulary and each client spells the off state
    its own way (the DeepSeek chat body renders ``thinking: {"type":
    "disabled"}`` from it).
    """
    ladder = list(model.reasoning_efforts)
    if "none" not in ladder or model.reasoning_effort == "none":
        return None
    return "none"


def _lower_effort(model: "ModelSpec") -> str | None:
    """The effort one rung below ``model.reasoning_effort`` on its own ladder.

    ``None`` when the model has no ladder or already runs at its bottom rung:
    there is no cheaper setting to retry at, and a retry at the SAME effort
    would reproduce the same silent truncation, so the caller ends the turn
    instead."""
    ladder = list(model.reasoning_efforts)
    current = model.reasoning_effort
    if not ladder or current is None or current not in ladder:
        return None
    index = ladder.index(current)
    return ladder[index - 1] if index > 0 else None


# How long the batch waits for tools to unwind after an ABORT before it stops
# waiting and settles the turn anyway. An abort is a user pressing Esc, so the
# turn must end on a human timescale; a tool whose cleanup is slower than this
# (a process group refusing to die) keeps unwinding in the background while the
# turn it belonged to is already over.
ABORT_DRAIN_TIMEOUT_S = 2.0

_QueueItemT = TypeVar("_QueueItemT")


async def _get_before_timeout(queue: asyncio.Queue[_QueueItemT], timeout: float) -> _QueueItemT:
    """Get one queue item before ``timeout`` without losing a boundary item.

    The timeout and a queue delivery can become ready in the same event-loop
    turn. ``wait_for(queue.get())`` may report the timeout after the getter has
    already dequeued an item, orphaning that event from both this drain and its
    final queue flush. Racing explicit tasks lets delivery win every tie; if the
    timer wins alone, the getter is cancelled and joined before control returns,
    so it cannot consume a later item in the background.

    THE INVARIANT, on every exit path including the caller being cancelled: an
    item this call took off the queue is either returned to the caller or put
    back. Nothing is consumed and dropped. The first version of this helper held
    only the timeout half of that promise, which made it a silent behavioural
    divergence from the ``wait_for`` it replaced — ``wait_for`` never touches an
    already-completed getter, so a caller cancelled in the delivery turn left
    the item in the queue, while this helper's cleanup discarded it (R1-1, agent
    review round 1). In a helper whose entire reason to exist is event loss at a
    cancellation boundary, only the full invariant is worth stating.

    A reclaimed item returns at the TAIL. The promise is that it is still in the
    queue for the next reader, not that FIFO order survives a cancelled read;
    the abort drain that calls this abandons the queue on that path anyway.
    """
    getter = asyncio.create_task(queue.get())
    timer = asyncio.create_task(asyncio.sleep(timeout))

    def reclaim() -> None:
        """Hand back an item the getter dequeued that no caller will receive.

        A completed, uncancelled getter is the ONLY shape that can strand an
        item: ``Queue.get`` removes nothing until its task has run to
        completion, and a getter cancelled while still suspended re-wakes the
        next waiter on its way out, so the item never left the queue.
        """
        if getter.done() and not getter.cancelled() and getter.exception() is None:
            queue.put_nowait(getter.result())

    # Decided INSIDE the try, because the cleanup below runs before any return
    # and has to know whether a dequeued item is still on its way to the caller
    # or has been stranded. Computing it after the fact would reclaim the item
    # on the success path too, handing the caller a copy that is also back in
    # the queue. ``not cancelled`` rather than a result check so a getter that
    # somehow failed re-raises its own exception below, as ``wait_for`` did.
    delivered = False
    try:
        await asyncio.wait({getter, timer}, return_when=asyncio.FIRST_COMPLETED)
        delivered = getter.done() and not getter.cancelled()
    finally:
        # Stop both contestants SYNCHRONOUSLY, before any await in this block.
        # A join is a suspension point, so a contestant still running while we
        # are parked in one could take a further item off the queue.
        for task in (getter, timer):
            if not task.done():
                task.cancel()
        # A caller cancelled during ``wait`` lands here with ``delivered``
        # still False and the getter possibly already holding an item nobody
        # will receive. Hand it back before the joins, since a join is also
        # where a pending cancellation gets delivered and the rest of this
        # block skipped. On the timeout path the getter is still suspended and
        # holds nothing, so ``reclaim`` is a no-op.
        if not delivered:
            reclaim()
        try:
            for task in (getter, timer):
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        except BaseException:
            # Cancelled DURING the join, after the success path below was
            # already committed to returning this item. The joins are the only
            # awaits between taking the item and the caller receiving it, so
            # this is the last window in which the item can be stranded.
            if delivered:
                reclaim()
            raise

    if delivered:
        return getter.result()
    raise TimeoutError


def _describe_call(
    describe: ApprovalDescribeFn,
    arguments: dict[str, Any],
    cwd: str,
    context: ToolContext | None,
) -> str:
    """Call a describer, handing it the turn's context only if it asks for one.

    Opt-in by PARAMETER NAME, the way ``harness/approval.py`` resolves a gate's
    ``job_id``: every describer that takes ``(args, cwd)`` — which is all of them
    but the path one — is called exactly as before, while a describer that also
    declares ``context`` receives it. Counting parameters would be wrong in both
    directions (a ``*args`` describer would be handed an argument it cannot
    name, and a keyword-only third parameter would be missed), and the reason a
    describer needs the context at all is that some targets have no path to name
    without the session's roots: ``scratchpad://perf.md`` resolves to a file
    under the session directory, and an approval prompt that cannot name it is
    asking a person to authorise a file it will not show them.
    """
    try:
        parameters = inspect.signature(describe).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "context" not in parameters:
        return describe(arguments, cwd)
    wide = cast("Callable[..., str]", describe)
    return wide(arguments, cwd, context=context)


def _consume_claim(claimed: Counter[str], call_id: str) -> bool:
    """Spend one suppression owed for ``call_id``; say whether there was one.

    Module level, and named, so a test can exercise THIS rule rather than a
    retyped copy of it. The branch it serves is unreachable today (the
    source-side guard in ``park`` removes the collision it defends against), so
    no behavioural test can reach it — which is exactly why the rule needs a
    handle a unit test can hold (R7-3, agent review round 7).

    COUNTING, not membership, is the whole point. Call ids are not unique within
    a batch: a duplicate id yields one slot that started and one that did not,
    and matching by id suppressed the started call's genuine end event along
    with its twin's parked one (R5-1). Spending one claim per event leaves
    exactly the right number, and the events are identical to a consumer, so
    which one survives does not matter.
    """
    if not claimed.get(call_id, 0):
        return False
    claimed[call_id] -= 1
    return True


@dataclass
class LoopContext:
    """Mutable host context the loop reads and extends.

    ``system_blocks`` is an ordered LIST (providers place cache breakpoints
    per block); ``messages`` is the live transcript; ``tools`` is the current
    inventory. ``tool_context`` is handed to every ``tool.execute`` call.
    """

    system_blocks: list[str] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)
    tools: list[AgentTool] = field(default_factory=list)
    tool_context: ToolContext | None = None


@dataclass
class _PlannedCall:
    """One resolved tool call: either ready to run or pre-failed.

    ``args`` is what the tool actually receives; ``intent`` is the model's
    narration lifted out of it. They are separate fields because they are
    separate claims — the card shows the command, the working line shows what
    the model said it was doing, and when those disagree the transcript has
    to show the disagreement rather than hide it behind one string.
    """

    call: ToolCall
    tool: AgentTool | None = None
    args: dict[str, Any] = field(default_factory=dict)
    failure: ToolResult | None = None  # resolution/validation/approval failure
    intent: str | None = None
    # None means the tool has not declared independently lockable resources.
    # An empty tuple is a valid declaration of no exclusive resources.
    resources: tuple[str, ...] | None = None


def _tool_start_event(
    *,
    tool_call_id: str,
    tool_name: str,
    args: dict[str, Any] | None,
    intent: str | None,
) -> ToolExecutionStartEvent:
    """Build a ``tool_execution_start`` stamped with WHEN the call began.

    One factory for the three sites that dispatch a tool — the parallel
    ``runner``, its ``interruptible_runner`` sibling and the eval bridge —
    because the stamp is the only thing a late-attaching viewer can seed a
    live row's clock from, and a site that forgot it would leave that row
    counting from the switch. Nothing else in this file would notice the
    drift, so the three sites deliberately do not hold three copies of the
    construction.

    Stamped with ``time.time()``, not ``time.monotonic()``, because the value
    crosses a process boundary: ``live_events`` is serialized onto the attach
    wire, where a monotonic reading means nothing. It deliberately does NOT
    cross the durable boundary — ``FrontendSessionState.checkpoint()`` strips
    the folded map — so this stamp reaches an ATTACHING viewer and never a
    resumed session; a reader must not look for it as durable state.
    Readers convert the AGE once and tick on their own monotonic clock (see
    ``ToolCard.restore``), so a system-clock adjustment after the seed cannot
    move a counter that is already running.
    """
    return ToolExecutionStartEvent(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        args=args or {},
        intent=intent,
        started_at_epoch=time.time(),
    )


def _batches_shared(item: _PlannedCall) -> bool:
    """Whether this call may run alongside its neighbours in one batch.

    An unresolved or pre-failed call counts as shared: it never executes, so
    it cannot conflict with anything. Only a resolved ``exclusive`` tool
    forces a batch of one.
    """
    return (
        item.failure is not None
        or item.tool is None
        or item.tool.concurrency == "shared"
        or item.resources is not None
    )


def _limit_cut_arguments(call: ToolCall) -> bool:
    """Whether the OUTPUT LIMIT cut this call's arguments mid-dictation.

    The length arm appends its placeholder to every call in the turn, but only
    some of those calls were cut. A turn can dictate a call to COMPLETION and
    then spend its remaining budget on prose or on a second call, and the loop
    already treats such a call as a request the model did make -- the
    connectivity arm draws the same line (``truncated = [...]`` below: "a call
    whose arguments finished arriving BEFORE the cut is a complete request").
    Handing it the size framing would assert a cause the loop has not
    established: measured, the identical 43-byte arguments re-issued on the next
    turn ran and wrote the file (review F1 == QA Q1).

    Deliberately STRICTER than that site's ``raw_arguments and not arguments``,
    which cannot tell "the raw text would not parse" from "it parsed to an empty
    mapping": a zero-argument call serialized as ``{}`` -- or any complete call
    whose JSON is not an object -- has truthy raw text and empty ``arguments``
    either way, so the shorthand would repeat the false cause here. Re-parsing
    the raw text answers the question the TEXT needs answered -- did the
    arguments finish arriving -- and this runs only in the length arm, once per
    call, on a string the assembler already parsed once this turn.

    The parse failing means the arguments did not arrive as a COMPLETE JSON
    document. On a length-stopped turn that is the truncation: a JSON object
    cannot balance and then continue, so a fragment that fails to parse is a
    fragment that never finished, and a value that parses is a value the model
    finished emitting (``_assemble_tool_call`` reads the accumulated deltas
    once and leaves BOTH ``arguments`` empty and the unparseable fragment in
    ``raw_arguments`` when it cannot parse).

    It is NOT only that, and this is the boundary the predicate really draws:
    a call whose arguments arrived COMPLETE but are not valid JSON -- the case
    ``validate_tool_arguments`` exists for, e.g. a dictation cut short by the
    model rather than by the limit, like ``{"a": }`` -- fails the same parse
    and takes the size claim, on a call that was neither cut nor oversize
    (review round 2, MINOR-3). The two are NOT distinguishable from the
    fragment alone, and the residual is accepted in this direction on purpose:
    "cut" tells the model the call as dictated cannot be sent and to send it
    again, which is actionable for both, whereas the other arm's text asserts
    the arguments arrived COMPLETE -- a claim nothing unparseable supports, and
    the false-cause class this family exists to remove. The malformed case also
    self-corrects one turn later, where the next call's validation reports the
    parse failure by name.

    Unreachable for whitespace-only ``raw_arguments`` in production, and the
    ``not call.raw_arguments`` arm below says why: ``_assemble_tool_call``
    stores ``raw or None``, so a whitespace-only fragment arrives as ``None``
    rather than as a string that fails to parse (review round 2, NIT-1).
    """
    if not call.raw_arguments:
        # No raw text at all: a zero-argument call, complete by definition.
        return False
    try:
        json.loads(call.raw_arguments)
    except json.JSONDecodeError:
        return True
    return False


def _error_batch_fingerprint(calls: list[ToolCall], results: list[ToolResult]) -> str | None:
    """Recognize exact repeated failure without retaining tool output bodies.

    Calls and results are paired by position, since duplicate IDs are legal
    input the executor diagnoses. Successful calls, skipped work, and changing
    errors all break the streak. Intent narration is excluded: rewriting the
    explanation while repeating the failed operation is still the same error.
    """
    if not calls or len(calls) != len(results) or any(not result.is_error for result in results):
        return None
    digest = hashlib.sha256()
    for call, result in zip(calls, results):
        # The two synthetic texts below are the loop's own statement that the
        # call never ran, so they break the streak rather than counting as
        # evidence of the model repeating itself. Both are minted by the
        # EXECUTOR for calls it parks or skips, which is why they arrive here
        # paired with the results they stand for.
        #
        # ``TRUNCATED_RESULT_TEXT`` is deliberately NOT in this tuple, although a
        # model re-emitting a call the limit cut is no more floundering than
        # those two. It cannot be exempted HERE: the length arm appends its
        # placeholders to the context and never records them as results, so this
        # function is handed that turn with calls and no results and has already
        # returned ``None`` one guard above (``len(calls) != len(results)``)
        # before the membership test is reachable. An entry here would therefore
        # protect nothing while telling the next reader it was load-bearing
        # (review round 1, F4). If the length arm ever begins pairing its
        # placeholders with results, that ordering must be revisited with this
        # line -- the exemption this tuple provides only works for texts the
        # executor itself put in ``results``.
        if result.text in (ABORTED_RESULT_TEXT, SKIPPED_RESULT_TEXT):
            return None
        args = {key: value for key, value in call.arguments.items() if key != INTENT_FIELD}
        digest.update(
            json.dumps(
                [call.name, args, call.raw_arguments if not args else None],
                sort_keys=True,
                default=str,
            ).encode()
        )
        digest.update(result.text.encode())
    return digest.hexdigest()


async def _abortable_stream(
    stream: AsyncIterator[StreamEvent], signal: AbortSignal | None
) -> AsyncIterator[StreamEvent]:
    """Yield from ``stream`` but stop as soon as ``signal`` aborts.

    A provider stream is an ``async for`` parked in ``await``: between two
    tokens the loop is inside the socket read, and nothing there consults the
    abort flag. Aborting mid-stream therefore did nothing until the model's
    NEXT event arrived - for a model that had gone quiet (a long reasoning
    block, a stalled connection, a slow first token) that is seconds of a UI
    still painting a turn the user has already stopped, and on a wedged
    connection it is the read timeout.

    The stream is drained by a PUMP TASK feeding a queue, and the abort cancels
    that task. The cancellation lands inside the socket read, which is what
    actually releases the provider connection; the consumer here is woken by
    the same event and simply stops. Ending quietly rather than raising is what
    keeps the caller simple - the loop sees the stream finish, and its existing
    ``signal.aborted`` check labels the turn ``aborted``, pairs every dangling
    tool call, and emits the events a stopped turn owes the UI.

    A pump rather than the obvious "race each pull against the signal": racing
    per event costs two tasks and an ``asyncio.wait`` per token. Over a
    4000-delta response (an ordinary long answer), measured on an M-series Mac:
    a bare ``async for`` takes single-digit milliseconds, this pump takes
    single-digit milliseconds too, and the per-event race takes roughly **1.5
    SECONDS** - three orders of magnitude worse, and a visible stutter in the
    very stream this function exists to make more responsive. The pump pays one
    task for the whole stream instead of one per token.

    The queue is unbounded, which is safe for a reason specific to this caller:
    ``_model_turn`` already accumulates every delta of the response in
    ``text_parts``, so a transient second reference to data the loop is holding
    anyway cannot change the memory profile. In practice the queue stays near
    empty - the producer is network-bound and the consumer is not.
    """
    if signal is None:
        async for event in stream:
            yield event
        return

    queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
    finished = asyncio.Event()

    async def pump() -> None:
        async for event in stream:
            queue.put_nowait(event)

    task = asyncio.ensure_future(pump())
    # A DONE CALLBACK, not the pump's own ``finally``. ``ensure_future`` only
    # SCHEDULES the coroutine, so a cancel landing before the body runs — which
    # is exactly the pre-aborted fast path below — never executes any statement
    # inside ``pump``, ``finally`` included. Waking the consumer from a
    # ``finally`` therefore deadlocked the turn permanently: the queue stayed
    # empty, the event was never set, and the drain parked forever on a wake-up
    # nobody was going to send. A done callback fires for a task cancelled
    # before it starts, which is the whole difference. (Same hazard the batch
    # drain documents and defends against; this is the second instance of it.)
    task.add_done_callback(lambda _task: finished.set())
    watcher = asyncio.ensure_future(_cancel_when_aborted(signal, task))
    # An abort that has ALREADY fired must not be missed: the watcher only gets
    # to run on the next loop pass, by which time the pump could have consumed
    # the whole stream and spent a request the user had stopped.
    if signal.aborted:
        task.cancel()

    try:
        while True:
            if not queue.empty():
                yield queue.get_nowait()
                continue
            if task.done():
                # Drained AND the producer is finished. Surface a provider
                # failure the way an unwrapped ``async for`` would, so the
                # caller's error handling is unchanged by this wrapper; a
                # cancellation is the abort and is deliberately not re-raised.
                with contextlib.suppress(asyncio.CancelledError):
                    task.result()
                return
            getter = asyncio.ensure_future(queue.get())
            ended = asyncio.ensure_future(finished.wait())
            try:
                done, _pending = await asyncio.wait(
                    {getter, ended}, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                for pending in (getter, ended):
                    if not pending.done():
                        pending.cancel()
            if getter in done:
                yield getter.result()
    finally:
        watcher.cancel()
        if not task.done():
            task.cancel()
        with contextlib.suppress(BaseException):
            await task


async def _cancel_when_aborted(signal: AbortSignal, task: asyncio.Task[None]) -> None:
    """Cancel ``task`` when ``signal`` fires. Split out so the watcher holds no
    reference to the generator frame it belongs to."""
    await signal.wait()
    task.cancel()


class AgentLoop:
    """Runs turns: model streaming, tool execution, steering re-entry.

    Stateless between ``run`` calls; all run state lives in the generator
    frame, so one instance may serve sequential turns.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        initial_messages: list[AgentMessage],
        context: LoopContext,
        config: LoopConfig,
        signal: AbortSignal | None = None,
        generation: int = 0,
    ) -> AsyncIterator[AgentEvent]:
        """Run one prompt to completion, yielding :class:`AgentEvent`s.

        Async generator: the caller drives it with ``async for``. Emits
        ``agent_start`` first and exactly one terminal ``agent_end`` whose
        ``messages`` are every message produced by this run. ``generation``
        stamps both boundary events so UIs can drop superseded ends.
        """
        return self._run(initial_messages, context, config, signal, generation)

    async def run_to_end(
        self,
        initial_messages: list[AgentMessage],
        context: LoopContext,
        config: LoopConfig,
        signal: AbortSignal | None = None,
        generation: int = 0,
    ) -> list[AgentMessage]:
        """Convenience wrapper: run and return the final new messages."""
        final: list[AgentMessage] = []
        async for event in self._run(initial_messages, context, config, signal, generation):
            if isinstance(event, AgentEndEvent):
                final = list(event.messages)
        return final

    # ------------------------------------------------------------------
    # The two nested while loops
    # ------------------------------------------------------------------

    async def _run(
        self,
        initial_messages: list[AgentMessage],
        context: LoopContext,
        config: LoopConfig,
        signal: AbortSignal | None,
        generation: int = 0,
    ) -> AsyncIterator[AgentEvent]:
        signal, deadline_task = self._wire_deadline(config, signal)
        new_messages: list[AgentMessage] = []
        context.messages.extend(initial_messages)
        pending: list[AgentMessage] = []
        has_more_tool_calls = True  # forces the first model call
        reentries = 0  # outer-loop re-entries; capped by config
        # A reasoning model can spend its ENTIRE output budget thinking and be
        # cut off at ``length`` with nothing visible to show for it — the user
        # then watches minutes of "thinking" end in silence (session f3c058d1:
        # four consecutive empty 8192-token truncations on the fallback model).
        # The loop retries such a turn a bounded number of times one effort rung
        # lower, where the budget goes to output instead of thought; a turn
        # that DID produce text or calls is truncated, not silent, and keeps
        # the old pair-and-stop behaviour.
        empty_truncation_retries = 0
        # Turns this run has retried with thinking turned OFF after DeepSeek
        # refused a request for a missing reasoning echo -- and, separately,
        # turns it has re-sent with the echo FILLED. Run-scoped and topped up
        # by nothing: the refusal is a property of what this run is sending, so
        # a fresh allowance per turn would re-buy the same diagnosis.
        reasoning_echo_retries = 0
        reasoning_echo_fill_retries = 0
        # The provider/model pair a reasoning-echo FILL is in force for, set
        # when the loop has evidence that this route's requests were reaching
        # the provider without their echo. Applied to the spec of each later
        # request in this run, because the host's ``get_model`` resolver hands
        # back its OWN spec (the same reason ``effort_ceiling`` above exists) --
        # without it the fill would reach one request and be undone on the
        # next, which is a slower version of the bug. Scoped to the pair rather
        # than a bare flag so a mid-run model switch re-derives its own
        # answer instead of inheriting a decision made about another model.
        echo_fill_route: tuple[str, str] | None = None
        # Turns this run has continued after the network cut them short. Run-
        # scoped, not per-turn: a laptop carried between networks can interrupt
        # the same run more than once, and the budget bounds the RUN's total
        # tolerance for that rather than handing each turn a fresh allowance.
        connectivity_continuations = 0
        # The effort ceiling an empty-truncation retreat imposed; rides on
        # every request under it so the session's frozen auto-effort cannot
        # raise the retry back to the rung that produced nothing.
        effort_ceiling: str | None = None
        # The provider-reported context size of the latest call in THIS run
        # (``Usage.context_tokens``), stamped onto the next request as its
        # prompt-cache TTL hint. Run-local on purpose: the host's
        # ``get_context_tokens_hint`` is the cross-turn seed (last turn's
        # final call, or a resumed transcript's), and a long tool loop can
        # grow past the TTL threshold dozens of calls before the host's figure
        # is next refreshed — a subagent is ONE turn for its whole life and
        # would otherwise never carry a hint at all. Once a call in this run
        # has reported, its count beats the seed (review F9).
        run_context_tokens: int | None = None
        last_error_batch: str | None = None
        repeated_error_batches = 0

        yield AgentStartEvent(generation=generation)

        try:
            while True:
                # ---- inner loop: model + tools until quiescent -----------
                first_inner = True
                while has_more_tool_calls or pending:
                    # Steering can land between a tool batch and the next
                    # model call; drain it at the top of every continuation
                    # iteration so it reaches the next request. Asides keep
                    # their boundary semantics (collected after batches and
                    # at the yield edge) so a queued aside still forces its
                    # own follow-up model call.
                    if not first_inner:
                        if config.get_steering_messages is not None:
                            pending.extend(await config.get_steering_messages())
                    first_inner = False
                    if pending:
                        last_error_batch = None
                        repeated_error_batches = 0
                    self._drain_pending(pending, context)
                    pending = []

                    gate = config.before_model_call
                    if gate is not None:
                        proceed = gate()
                        if inspect.isawaitable(proceed):
                            proceed = await proceed
                        if not proceed:
                            yield AgentEndEvent(
                                messages=new_messages,
                                aborted=True,
                                error="stopped by gate",
                                generation=generation,
                            )
                            return

                    assistant, stop_reason, stream_error = None, "stop", None
                    turn_connectivity_loss = False
                    # The spec this call was BUILT with, which is not
                    # ``config.model`` whenever the host resolves per call. The
                    # echo fill below is gated on what was actually sent.
                    turn_model: "ModelSpec | None" = None
                    async for event in self._model_turn(
                        context,
                        config,
                        signal,
                        effort_ceiling=effort_ceiling,
                        echo_fill_route=echo_fill_route,
                        context_tokens_hint=(
                            run_context_tokens
                            if run_context_tokens is not None
                            else self._host_context_tokens_hint(config)
                        ),
                    ):
                        if isinstance(event, _ModelTurnResult):
                            assistant, stop_reason, stream_error = (
                                event.message,
                                event.stop_reason,
                                event.error,
                            )
                            turn_model = event.model
                            turn_connectivity_loss = event.connectivity_loss
                        else:
                            yield event
                    if assistant is None:
                        raise RuntimeError("model turn produced no assistant message")

                    if turn_connectivity_loss:
                        # THE NETWORK WENT AWAY MID-TURN — the laptop was closed
                        # at home and opened at work. The provider layer has
                        # already spent its patient budget waiting for a route
                        # back (~9 min of connectivity backoff), and there is
                        # still none, so this is the outer, coarser retry.
                        #
                        # Only reachable once output was already forwarded: with
                        # nothing forwarded, the failover driver retries in place
                        # and the turn never ends this way. That is exactly why
                        # the driver CANNOT fix this case itself — those deltas
                        # are already in the user's transcript, so re-issuing the
                        # same request there would stream them a second time
                        # (the invariant `test_a_turn_does_NOT_replay_output_the_
                        # user_already_read` locks down). Continuing HERE is what
                        # makes a retry safe: the partial answer is committed to
                        # the context as an assistant message just below, so the
                        # next request carries it as history and the model writes
                        # only the REMAINDER. Nothing the user has read is
                        # re-rendered, because the continuation is a new message
                        # rather than a replay of the old one.
                        #
                        # The cost of that safety is honesty about the seam: the
                        # model resumes from the text, not from its own hidden
                        # state, so a sentence cut mid-word is continued rather
                        # than rewritten. That beats the alternative this
                        # replaces, which was ending the entire run.
                        if connectivity_continuations < MAX_CONNECTIVITY_CONTINUATIONS:
                            connectivity_continuations += 1
                            # TRUNCATED calls are dropped; COMPLETE ones are not.
                            # A call still being dictated when the socket died is
                            # truncated JSON, and executing it would run a tool
                            # the model never finished asking for. But a call
                            # whose arguments finished arriving BEFORE the cut is
                            # a complete request the model did make, and throwing
                            # it away loses real work twice over: the tool never
                            # runs, and the continuation prompt below then
                            # describes a turn that was "cut off mid-sentence"
                            # while the model's own record of having asked for
                            # the call has been deleted from history.
                            #
                            # `_assemble_tool_call` already draws the line: it
                            # parses the accumulated JSON and leaves `arguments`
                            # empty while keeping `raw_arguments` when the parse
                            # failed. So "parsed, or had nothing to parse" is
                            # complete, and "had raw text that would not parse"
                            # is truncated. A call with NO raw arguments at all
                            # is a zero-argument call, which is complete.
                            truncated = [
                                call
                                for call in assistant.tool_calls
                                if call.raw_arguments and not call.arguments
                            ]
                            assistant.tool_calls = [
                                call for call in assistant.tool_calls if call not in truncated
                            ]
                            # WHITESPACE-ONLY TEXT IS NOT TEXT. The wire builders
                            # drop an assistant turn whose text is whitespace-only
                            # (`_is_empty_assistant`, clients.py: "Whitespace-only
                            # text counts as empty"), so a model that emitted a
                            # single newline before the cut would have its turn
                            # vanish from the request while the continuation
                            # prompt went on referring to "the partial text
                            # above" — pointing at nothing. The guard is aligned
                            # with the serializer so that case routes to the
                            # drop-and-re-ask arm instead. Tool calls ARE content
                            # by that same predicate, so a turn carrying only a
                            # complete call is still committable.
                            resumable_text = bool(assistant.text.strip())
                            if resumable_text or assistant.tool_calls:
                                # The partial answer becomes history, which is
                                # what makes the continuation additive instead of
                                # a replay.
                                #
                                # The partial text is committed VERBATIM,
                                # trailing space and all, so the transcript is
                                # byte-for-byte what the user actually read. That
                                # is safe only because SOMETHING always follows it
                                # below — the continuation prompt, or a tool
                                # result for a surviving call: Anthropic rejects
                                # trailing whitespace on a FINAL assistant
                                # message ("final assistant content cannot end
                                # with trailing whitespace"), and an interrupted
                                # delta is exactly how one is produced. This
                                # message is never final, so the rule does not
                                # apply and nothing has to be trimmed away from
                                # what was displayed.
                                context.messages.append(assistant)
                                new_messages.append(assistant)
                                if assistant.tool_calls:
                                    # PAIR the survivors. An assistant turn
                                    # carrying a `tool_use` block with no matching
                                    # `tool_result` is a 400 on the Anthropic
                                    # wire, and this turn becomes history for the
                                    # retry rather than being executed — the
                                    # network died before the loop could run it.
                                    # The same synthetic `aborted` result every
                                    # other abandoned-turn path here uses, so the
                                    # model learns the call did not run and may
                                    # re-issue it, which is strictly more
                                    # information than the call having vanished.
                                    self._append_results(
                                        context,
                                        [
                                            self._synthetic_result(call, ABORTED_RESULT_TEXT)
                                            for call in assistant.tool_calls
                                        ],
                                        new_messages,
                                        redact=config.redact_tool_result,
                                    )
                                if resumable_text or truncated:
                                    # KEYED ON PROSE ALONE — and on an
                                    # interrupted CALL, which has no prose to key
                                    # on — independently of the pairing above.
                                    # An `elif` here silently lost the
                                    # instruction for the one shape, and only
                                    # the prose half was ever emitted, so a turn
                                    # whose call was cut off mid-arguments got an
                                    # instruction to "continue seamlessly" a
                                    # sentence it never finished while the action
                                    # it had chosen quietly vanished from its
                                    # history. A cut that produced BOTH partial
                                    # prose and a complete call took the pairing
                                    # arm and never reached this one at all, so
                                    # text the user had already read was
                                    # committed to history with nothing telling
                                    # the model not to repeat it, and the answer
                                    # could restart mid-sentence ("Paris is the
                                    # capital of Paris is the capital of
                                    # France."). Keeping the calls (above) is
                                    # what made that shape reachable: before it,
                                    # any turn with a call had its calls cleared
                                    # and fell into the text arm, which did append
                                    # this prompt.
                                    #
                                    # Ordering is load-bearing: the synthetic tool
                                    # results are appended FIRST, so the prompt
                                    # still lands after them and the `tool_use` →
                                    # `tool_result` adjacency the wire requires is
                                    # never broken by a user turn wedged between.
                                    # Verified on both real serializers for this
                                    # shape: Anthropic gets blocks
                                    # `[[text],[text,tool_use],[tool_result],[text]]`
                                    # with no unpaired id, OpenAI gets roles
                                    # `user,assistant,tool,user`. Anthropic renders
                                    # a `tool_result` as a USER turn, so the prompt
                                    # follows one — legal since Anthropic began
                                    # accepting consecutive same-role messages
                                    # (Oct 2024), and the alternation concern the
                                    # comment below raises is about a TRAILING
                                    # assistant turn, which this shape never has.
                                    #
                                    # A CONTINUATION INSTRUCTION, not a bare
                                    # trailing assistant turn. Leaving the partial
                                    # answer last is "prefilling", which current
                                    # Claude models refuse outright ("Prefilling
                                    # assistant messages is not supported for this
                                    # model", HTTP 400) — so the very models this
                                    # harness defaults to would turn a recoverable
                                    # network blip into a hard failure. A
                                    # user-role instruction is legal on every
                                    # wire, keeps the alternation Anthropic
                                    # documents, and states the constraint the
                                    # model must respect: continue, do not
                                    # restart, because the user has ALREADY READ
                                    # the text above.
                                    #
                                    # Appended to new_messages as well as to the
                                    # live context, because `new_messages` is what
                                    # `AgentEndEvent` hands to the host to
                                    # PERSIST. Left out, a resumed session reads
                                    # the partial answer glued directly to its
                                    # continuation with no record that a network
                                    # interruption sat between them — and a run
                                    # that continued more than once persisted a
                                    # run of consecutive assistant messages that
                                    # no longer explains itself, which compaction
                                    # then summarises. It is harness chrome, so
                                    # the front ends suppress it on replay the
                                    # same way they already suppress the
                                    # compaction continuation prompt, through the
                                    # ONE shared decision in `harness.rows`
                                    # (`is_harness_chrome`), which recognises EVERY
                                    # shape `_continuation_instruction` can return —
                                    # the composed form included. Equality with the
                                    # prose prompt alone was not enough: the
                                    # composed and tool-call-only shapes are not
                                    # members of `harness_chrome_prompts()`, and a
                                    # resumed session painted them as the
                                    # operator's own words. REPLAY is
                                    # the only surface that needs it: this row
                                    # reaches the transcript via
                                    # `_persist_new_messages`, which appends
                                    # without emitting a `MessageStartEvent`, so
                                    # there is no live announcement to suppress
                                    # and the announce loop is correctly untouched.
                                    prompt = Message.user(
                                        _continuation_instruction(
                                            resumable_text=resumable_text,
                                            interrupted=truncated,
                                        )
                                    )
                                    context.messages.append(prompt)
                                    new_messages.append(prompt)
                            # Neither text nor a surviving call? The message is
                            # DROPPED rather than committed: there is nothing to
                            # continue from, and an assistant message with no
                            # content blocks is rejected outright on the Anthropic
                            # wire ("text content blocks must be non-empty"),
                            # which would kill the very run this is rescuing. The
                            # retry simply re-asks the original question.
                            #
                            # The turn is CLOSED before continuing. `TurnEndEvent`
                            # is what drives a front end's per-turn reconciliation
                            # — in the TUI, `_retire_live_tool_cards`, the only
                            # routine path that settles rows for calls that never
                            # ran. Without it a call the model was still dictating
                            # when the socket died leaves a spinner animating
                            # forever on a turn that ended, and the working line
                            # keeps announcing "composing a call" while the
                            # continued turn streams prose. Every other continuing
                            # path in this loop reaches a TurnEndEvent; this one
                            # must too.
                            yield TurnEndEvent(message=assistant, tool_results=[])
                            # The notice states what HAS happened, not what is
                            # hoped for, and it is WRITTEN FOR BOTH FAMILIES the
                            # branch serves: our own connection dying, and a
                            # gateway reporting its upstream host dying in band.
                            # "network connection lost" was true of the first and
                            # false of the second — on an aggregator's upstream
                            # failure the machine's network is fine and the
                            # gateway's host is what died — so it named a cause
                            # the loop cannot know. What it CAN know is the one
                            # fact both share: the stream stopped mid-answer and
                            # the rest of the turn is being fetched on the next
                            # attempt. It is emitted BEFORE the retry, so it
                            # cannot honestly claim a reconnection either — on a
                            # genuinely dead network the old wording told the user
                            # three times that the machine had reconnected and
                            # then ended the run. Naming the budget also shows it
                            # being spent rather than repeating one identical line.
                            yield NoticeEvent(
                                text=(
                                    "response stream cut mid-answer — "
                                    f"resuming the turn ({connectivity_continuations}/"
                                    f"{MAX_CONNECTIVITY_CONTINUATIONS})"
                                ),
                                kind="warning",
                            )
                            has_more_tool_calls = True
                            continue
                        # Budget spent and still offline: fall through to the
                        # terminal branch below, which surfaces the provider's
                        # own diagnostic error. A dead network must end the run
                        # with a bounded, named failure rather than retry forever.

                    if assistant.usage is not None and assistant.usage.context_tokens:
                        # Only a REPORTED count advances the hint: a wire that
                        # omits it must not blank a figure the previous call
                        # (or the host's seed) supplied.
                        run_context_tokens = int(assistant.usage.context_tokens)
                    context.messages.append(assistant)
                    new_messages.append(assistant)

                    if stop_reason in ("error", "aborted", "refusal"):
                        # FIRST recovery for a refused reasoning echo: re-send the
                        # SAME request with the echo filled.
                        #
                        # This is the BENIGN half of the pair and it is tried
                        # first, before anything gives up a capability. Filling
                        # the echo adds a short placeholder sentence per blank
                        # assistant turn and changes nothing else -- same route,
                        # same effort, same thinking mode, same tools -- while
                        # the alternative below disables the model's reasoning
                        # for the rest of the run. The asymmetry is why they are
                        # ordered this way and why they are gated differently:
                        # a refusal whose OWN WORDS say the echo is missing is
                        # evidence the fill is what this request needs, so this
                        # branch needs no capability bit at all.
                        #
                        # Capability-INDEPENDENT on purpose, and that is not a
                        # hypothetical: the bit is a prediction about which
                        # routes run this validator, derived from the model's
                        # family (``model.configure.reasoning_echo_required``).
                        # A family rule cannot know about a model it has never
                        # seen -- the next generation id, the same weights behind
                        # a rebranded or relayed host -- and a spec that never
                        # went through the derivation can state the bit off
                        # outright. The provider's own wording is direct evidence
                        # about THIS request, so when the wording says the echo is
                        # missing and the spec we SENT was not carrying one, the
                        # fill is the measured answer (live: unfilled body 400,
                        # filled body 200 on the same window).
                        #
                        # That gate is on ``turn_model`` rather than on
                        # ``config.model`` because the two differ whenever a host
                        # resolves per call: what matters is whether the request
                        # the provider just refused carried an echo, and only the
                        # resolved spec can answer that.
                        #
                        # Bounded like every other recovery in this loop, and
                        # gated on nothing having been SHOWN: a 400 arrives
                        # before the first byte, and a turn the user has already
                        # read may not be replayed.
                        #
                        # ``not assistant.tool_calls`` is that gate, and it means
                        # a refusal that arrives AFTER streamed tool-call deltas
                        # gets no recovery at all -- inherited from the retreat
                        # below, not introduced here, and recorded on the PR as
                        # not addressed. A turn whose call is already on screen
                        # (or executing) is no more replayable than one with text,
                        # so lifting it is a design question about partial calls
                        # rather than a line to change.
                        if (
                            stop_reason == "error"
                            and reasoning_echo_fill_retries < MAX_REASONING_ECHO_FILL_RETRIES
                            and not assistant.text.strip()
                            and not assistant.tool_calls
                            and turn_model is not None
                            and not turn_model.requires_reasoning_echo
                            and _is_reasoning_echo_rejection(stream_error)
                        ):
                            reasoning_echo_fill_retries += 1
                            # The refused turn must not reach the retry's
                            # history: it carries nothing the user saw, and the
                            # next request has to re-send the same conversation
                            # that was just refused.
                            if context.messages and context.messages[-1] is assistant:
                                context.messages.pop()
                            if new_messages and new_messages[-1] is assistant:
                                new_messages.pop()
                            config.model = config.model.model_copy(
                                update={"requires_reasoning_echo": True}
                            )
                            echo_fill_route = (turn_model.provider, turn_model.model_id)
                            yield NoticeEvent(
                                text=(
                                    "the provider refused this request for a "
                                    "missing reasoning echo — retrying with the "
                                    "echo filled in at the same effort"
                                ),
                                kind="warning",
                            )
                            has_more_tool_calls = True
                            continue
                        # SECOND recovery, and the one that needs a gate: a
                        # route that never sends the echo must not have a rung
                        # of its own ladder disabled by a message that merely
                        # resembles this one. DeepSeek's thinking mode can refuse
                        # a request for a missing reasoning echo even though the
                        # body echoes one on every turn it has anything for (see
                        # ``ModelSpec.requires_reasoning_echo``). That refusal is
                        # recoverable rather than fatal: the same request with
                        # thinking disabled answers 200 (measured live), so spend
                        # one call on that instead of ending the turn.
                        #
                        # Gated on nothing having been SHOWN: the loop may only
                        # replay a turn whose output the user has not read, and
                        # a 400 arrives before the first byte. Gated on the
                        # provider's OWN WORDS and not on the capability bit:
                        # the bit is a prediction about which routes run this
                        # validator, and a prediction that is wrong must not turn
                        # a recoverable refusal into a dead turn -- which is
                        # exactly what it did, as an unclassified
                        # "unknown: invalid request (HTTP 400)" incident on a
                        # route whose spec never got the bit
                        # (``model.configure._served_model_family`` records how
                        # a route can be right and the bit wrong). The wording
                        # is direct evidence that THIS request lost the echo, so
                        # it is the wording that decides; the rung check below
                        # is the real precondition, because a model with no
                        # thinking-off rung is one the retry cannot help.
                        #
                        # That precondition excludes the route this was written
                        # for: `openrouter/deepseek/deepseek-v4.1-flash` ships
                        # the ladder ('low','high','max') with no `none` rung, so
                        # a refusal there still ends the turn after one request.
                        # On that route the echo DERIVATION is the whole
                        # protection and this is a backstop for the routes that
                        # do have a rung; it is not the recovery that saves the
                        # aggregator, and the test that pins it must derive its
                        # spec rather than hand one a rung production never
                        # builds.
                        #
                        # Deliberately still reads the RUN's model for the
                        # ladder, exactly as it shipped: the retreat is a
                        # statement about the model this run is on, while the
                        # fill above is a statement about the request that was
                        # sent -- and only ``turn_model`` can answer the second.
                        if (
                            stop_reason == "error"
                            and reasoning_echo_retries < MAX_REASONING_ECHO_RETRIES
                            and not assistant.text.strip()
                            and not assistant.tool_calls
                            and _is_reasoning_echo_rejection(stream_error)
                        ):
                            thinking_off = _thinking_off_effort(config.model)
                            if thinking_off is not None:
                                reasoning_echo_retries += 1
                                # The refused turn must not reach the retry's
                                # history: it carries nothing the user saw, and
                                # the next request has to re-send the same
                                # conversation that was just refused.
                                if context.messages and context.messages[-1] is assistant:
                                    context.messages.pop()
                                if new_messages and new_messages[-1] is assistant:
                                    new_messages.pop()
                                config.model = config.model.model_copy(
                                    update={"reasoning_effort": thinking_off}
                                )
                                # A ceiling as well as the snapshot: a host whose
                                # resolver returns its OWN model would otherwise
                                # put the retry straight back at the rung that was
                                # just refused (the empty-truncation retreat sets
                                # one for the same reason).
                                effort_ceiling = thinking_off
                                yield NoticeEvent(
                                    text=(
                                        "the provider refused this request for a "
                                        "missing reasoning echo — retrying with "
                                        "thinking disabled"
                                    ),
                                    kind="warning",
                                )
                                has_more_tool_calls = True
                                continue
                        # Pair every dangling tool call so the wire stays legal.
                        # "refusal" rides this branch because it is terminal the
                        # same way an error is: the model declined, so there is
                        # nothing to feed back for another call — and it must
                        # NOT fall through to the clean-stop path, where the
                        # turn ended with an empty frame and no explanation of
                        # what the provider refused (the silent-refusal bug).
                        # ``stream_error`` carries the provider's own refusal
                        # message, which is what lets the user decide whether
                        # to rephrase or switch models.
                        placeholders = [
                            self._synthetic_result(call, ABORTED_RESULT_TEXT)
                            for call in assistant.tool_calls
                        ]
                        self._append_results(
                            context,
                            placeholders,
                            new_messages,
                            redact=config.redact_tool_result,
                        )
                        yield TurnEndEvent(message=assistant, tool_results=[])
                        aborted = stop_reason == "aborted" or bool(signal and signal.aborted)
                        yield AgentEndEvent(
                            messages=new_messages,
                            aborted=aborted,
                            error=stream_error,
                            generation=generation,
                        )
                        self._discard_pending_custom(pending)
                        return

                    tool_results: list[ToolResult] = []
                    if stop_reason == "length":
                        # Whitespace-only prose counts as NOTHING, which is how
                        # ``rows.assistant_stop_notice`` reads the same turn:
                        # it strips before deciding, so a bare truthiness test
                        # on ``text`` here had the live notice announce a CUT
                        # ANSWER over a fold that reported no answer at all
                        # (review R2-n4). Both sides strip, so the two surfaces
                        # cannot describe one event in two voices.
                        has_text = bool(assistant.text and assistant.text.strip())
                        silent = not has_text and not assistant.tool_calls
                        # The limit's ARM, per call, computed ONCE — the notice
                        # below states the turn's arm from this list and the
                        # placeholder loop stamps each call's marker from it, so
                        # the turn-level sentence the operator reads and the
                        # per-call row underneath it cannot disagree about the
                        # same call. Empty (and therefore harmless) on the arms
                        # with no call: `silent`, and the prose arms.
                        cut_calls = [_limit_cut_arguments(call) for call in assistant.tool_calls]
                        if silent and empty_truncation_retries < MAX_EMPTY_TRUNCATION_RETRIES:
                            lower = _lower_effort(config.model)
                            if lower is not None:
                                empty_truncation_retries += 1
                                # The empty turn must not reach the retry's
                                # wire history: an assistant message with no
                                # content blocks is an illegal request on the
                                # Anthropic wire, and replaying "I said
                                # nothing" teaches the retry to say nothing.
                                if context.messages and context.messages[-1] is assistant:
                                    context.messages.pop()
                                if new_messages and new_messages[-1] is assistant:
                                    new_messages.pop()
                                config.model = config.model.model_copy(
                                    update={"reasoning_effort": lower}
                                )
                                effort_ceiling = lower
                                yield NoticeEvent(
                                    text=(
                                        f"model spent its whole output budget "
                                        f"thinking and produced nothing — "
                                        f"retrying at effort {lower}"
                                    ),
                                    kind="warning",
                                )
                                has_more_tool_calls = True
                                continue
                        if silent:
                            # No cheaper rung to retry at (or the retries are
                            # spent): the turn is about to end with NOTHING on
                            # screen — minutes of "thinking" and then silence,
                            # the exact failure shape reported from session
                            # f3c058d1. A notice is the minimum honest frame,
                            # and it names WHICH limit ended the retreat
                            # (review N1) so the user knows whether another
                            # manual retry could still step down.
                            cause = (
                                f"{empty_truncation_retries} lower-effort "
                                f"{'retries' if empty_truncation_retries != 1 else 'retry'} spent"
                                if empty_truncation_retries >= MAX_EMPTY_TRUNCATION_RETRIES
                                else "no lower effort setting to retry at"
                            )
                            yield NoticeEvent(
                                text=(
                                    "the model spent its whole output budget "
                                    f"thinking and produced no visible output "
                                    f"({cause}) — retry, or switch to a model "
                                    "with a larger output budget"
                                ),
                                kind="warning",
                            )
                        elif has_text:
                            # A partial ANSWER, which is the case the missing
                            # signal hid best: the prose that arrived reads as a
                            # complete short reply, and neither the TUI nor the
                            # phone folded the stop into a notice (review round 1,
                            # B1; reproduced by QA Q1 against a live provider).
                            #
                            # This arm also owns the turn that streamed prose AND a
                            # call in flight, which is why it is tested before
                            # ``tool_calls`` below: ``rows.assistant_stop_notice``
                            # reads that same turn text-first, so checking the call
                            # first had the live row announce "mid tool call" over
                            # a fold that said "answer cut off" -- one event in two
                            # voices, the exact failure this family exists to
                            # prevent, on a turn where there genuinely IS an answer
                            # to cut off (design round 2, D7).
                            #
                            # The remedy clause is not decoration: with no call in
                            # flight this is the one truncation the loop does NOT
                            # auto-continue (above), so the reader is the only actor
                            # left and every sibling row in the family names a move
                            # (design round 1, D4). A cut call IS re-asked, but only
                            # the call: the answer's remainder was never sent, and
                            # only the reader can ask for it.
                            yield NoticeEvent(
                                text=(
                                    "the model hit the output limit — this answer "
                                    "is cut off, and the rest was never sent — "
                                    "ask again to continue, or narrow the request"
                                ),
                                kind="warning",
                            )
                        elif assistant.tool_calls:
                            # Visible truncation with a call in flight and no prose
                            # to pronounce it: the call was NOT executed (the
                            # batch below pairs placeholders instead). Nothing
                            # else said so -- the loop's only length notice was
                            # the silent arm above, no surface had a length arm
                            # at all, and the result the model got back read just
                            # "aborted" -- so a model asked to write a large file
                            # reported that the call "came through empty",
                            # declined to retry, and the file was never written
                            # (QA round 1, Q2). Say which limit it was and what
                            # the loop is doing about it.
                            #
                            # WHICH ARM, though: `cut_calls` above already
                            # answers it, and this line used to answer "cut"
                            # unconditionally. On a turn whose calls all arrived
                            # COMPLETE that told the operator the model was being
                            # re-asked for a smaller call while the model's own
                            # result for it said the opposite ("Its arguments
                            # arrived complete, so there is nothing here to
                            # shrink. Re-issue this call as it is.") and the
                            # call's row said the opposite again -- the same
                            # false cause as review F1, one surface up (design
                            # round 1, D1; QA Q-R2-1; review round 2, MINOR-2).
                            # A MIXED turn takes the cut line: a call in it really
                            # was cut mid-dictation, which is what that clause
                            # claims, and the complete calls' rows stay precise
                            # about themselves.
                            #
                            # The reader also still learns the call never ran,
                            # from the placeholder result appended below:
                            # ``rows.output_limit_call_receipt`` says so on the
                            # call's own row, from the arm marker that result
                            # carries (not from this model-facing text, which a
                            # row must not paint: review F2).
                            yield NoticeEvent(
                                text=(
                                    "the model hit the output limit mid tool call "
                                    "— nothing was executed; re-asking it to "
                                    "re-emit the call in smaller pieces"
                                    if any(cut_calls)
                                    else "the model hit the output limit before the "
                                    "call ran — nothing was executed; re-asking it "
                                    "to re-issue the call as it is"
                                ),
                                kind="warning",
                            )
                        # Truncated: pair placeholders, do NOT execute.
                        #
                        # A synthetic result rather than the bare
                        # ``ABORTED_RESULT_TEXT``: the model sees this as the result
                        # of the call it watched itself emit, and the actionable
                        # fact is that the OUTPUT LIMIT ended the turn, not that
                        # some turn ended. Same distinction, and same measured
                        # cost, as the constants' own comment.
                        #
                        # The TEXT is chosen per call, and so is the marker in
                        # ``details``: the size framing is true only where the
                        # limit really did cut the arguments, and a call that
                        # arrived complete gets the text that says so. The marker
                        # is what a display surface reads to render the row in
                        # its own vocabulary instead of this prose (review F1,
                        # F2 -- see ``_limit_cut_arguments``).
                        placeholders: list[ToolResult] = []
                        for call, cut in zip(assistant.tool_calls, cut_calls, strict=True):
                            placeholders.append(
                                self._synthetic_result(
                                    call,
                                    (
                                        TRUNCATED_RESULT_TEXT
                                        if cut
                                        else LENGTH_ENDED_CALL_RESULT_TEXT
                                    ),
                                    details={
                                        OUTPUT_LIMIT_KEY: (
                                            OUTPUT_LIMIT_ARGUMENTS if cut else OUTPUT_LIMIT_TURN
                                        )
                                    },
                                )
                            )
                        self._append_results(
                            context,
                            placeholders,
                            new_messages,
                            redact=config.redact_tool_result,
                        )
                    elif assistant.tool_calls:
                        async for event in self._execute_tool_calls(
                            assistant.tool_calls, context, config, signal, tool_results
                        ):
                            yield event
                        self._append_results(
                            context,
                            tool_results,
                            new_messages,
                            redact=config.redact_tool_result,
                        )

                    graceful_cancel = False
                    if config.graceful_cancel_requested is not None:
                        # Asked HERE, at the boundary, precisely because the
                        # tools that just ran may have pushed a commit or opened
                        # a merge request. A supervisor's cancel stops the run
                        # without tearing one of those in half; the completed
                        # work stays in the transcript and the turn ends as
                        # aborted, exactly like a signalled stop would.
                        try:
                            graceful_cancel = bool(config.graceful_cancel_requested())
                        # A broken host hook must not strand the turn.
                        except Exception:  # noqa: BLE001
                            graceful_cancel = False
                    if (signal is not None and signal.aborted) or graceful_cancel:
                        # The abort landed while the batch was running. Its
                        # results are already appended above (every call paired,
                        # cancelled ones as synthetic ``aborted`` results), so
                        # the context is legal — but the loop must NOT feed them
                        # back for another model call. Continuing would spend a
                        # request, and the reply to it, on a turn the user
                        # stopped: the stop would read as "it kept going, and
                        # then answered".
                        yield TurnEndEvent(message=assistant, tool_results=tool_results)
                        yield AgentEndEvent(
                            messages=new_messages, aborted=True, generation=generation
                        )
                        self._discard_pending_custom(pending)
                        return

                    yield TurnEndEvent(message=assistant, tool_results=tool_results)
                    fingerprint = _error_batch_fingerprint(assistant.tool_calls, tool_results)
                    repeated_error_batches = (
                        repeated_error_batches + 1
                        if fingerprint is not None and fingerprint == last_error_batch
                        else int(fingerprint is not None)
                    )
                    last_error_batch = fingerprint
                    if repeated_error_batches >= REPEATED_TOOL_ERROR_LIMIT:
                        names = ", ".join(sorted({call.name for call in assistant.tool_calls}))
                        yield AgentEndEvent(
                            messages=new_messages,
                            error=(
                                f"No progress: {names} returned the same errors for "
                                f"{repeated_error_batches} unchanged tool batches. "
                                "Change the arguments or resolve the reported blocker "
                                "before retrying."
                            ),
                            generation=generation,
                        )
                        return
                    if repeated_error_batches == REPEATED_TOOL_ERROR_WARNING:
                        recovery = Message.user(
                            "Harness recovery notice: the last three tool batches repeated "
                            "the same arguments and errors. Inspect the error and change approach; "
                            "do not repeat the unchanged failing calls. If an external condition "
                            "must change, use a supported wait operation or explain the blocker."
                        )
                        context.messages.append(recovery)
                        new_messages.append(recovery)
                        yield NoticeEvent(
                            text="Repeated tool errors — requesting a different approach.",
                            kind="warning",
                        )
                    has_more_tool_calls = bool(assistant.tool_calls)
                    if has_more_tool_calls and config.on_turn_end is not None:
                        # The boundary hook fires only when the loop will
                        # CONTINUE — a terminal boundary is the post-turn
                        # pass's job (the host's own after-run gate), and
                        # firing there too would run every host hook twice
                        # for the price of one decision.
                        turn_end = config.on_turn_end
                        outcome = turn_end(list(context.messages))
                        if inspect.isawaitable(outcome):
                            outcome = await outcome
                        if isinstance(outcome, list):
                            # Mid-run context replacement (automatic mid-turn
                            # compaction). The replacement is authoritative
                            # for the context; the run accumulator keeps only
                            # what this run produced that SURVIVED it — ids
                            # the replacement dropped were summarized away and
                            # must never reach post-run persistence, where
                            # they would resurrect after the compaction entry
                            # that superseded them. Matching is by id because
                            # the renderer passes plain Messages through as
                            # the same objects but customs as fresh ones.
                            survivors = {
                                getattr(m, "id", None) for m in outcome if getattr(m, "id", None)
                            }
                            context.messages[:] = outcome
                            new_messages = [
                                m for m in new_messages if getattr(m, "id", None) in survivors
                            ]

                    pending = await self._collect_inflight_injections(config)

                # ---- outer loop tail: yield boundary ----------------------
                before_yield = config.on_before_yield
                if before_yield is not None:
                    outcome = before_yield()
                    if inspect.isawaitable(outcome):
                        await outcome

                late = await self._collect_yield_injections(config)
                if late:
                    reentries += 1
                    if reentries > config.max_paused_turn_continuations:
                        # MAX_PAUSED_TURN_CONTINUATIONS guard: a producer
                        # that never stops (follow-ups arriving faster than
                        # they are consumed) must not re-enter forever.
                        logger.warning(
                            "paused-turn continuation limit (%d) reached; ending run",
                            config.max_paused_turn_continuations,
                        )
                        yield NoticeEvent(
                            text=(
                                f"Continuation limit reached "
                                f"({config.max_paused_turn_continuations}); stopping."
                            ),
                            kind="warning",
                        )
                        self._discard_pending_custom(late)
                        yield AgentEndEvent(messages=new_messages, generation=generation)
                        return
                    pending = late
                    has_more_tool_calls = True
                    continue
                break

            yield AgentEndEvent(messages=new_messages, generation=generation)
        finally:
            self._unwire_deadline(deadline_task)
            if signal is not None:
                # Drop any any_of() watcher tasks (e.g. the deadline combo) so
                # they do not outlive the run.
                signal.cancel()

    # ------------------------------------------------------------------
    # Model streaming
    # ------------------------------------------------------------------

    @staticmethod
    def _current_model(config: LoopConfig) -> "ModelSpec":
        """The spec to call RIGHT NOW, re-read at every provider call.

        ``config.model`` is bound once when the host builds the config, so on
        its own it pins a whole run — model, tools, model, tools — to whichever
        model the run started on. A user switching model mid-turn is switching
        precisely because the running model is doing badly, and their switch
        used to reach nothing until the turn ended. ``get_model`` is the host's
        answer to "which model now", asked once per call.

        Falls back to the snapshot when the host supplies no resolver (every
        embedder and test double that builds a ``LoopConfig`` by hand), which
        is what keeps this backwards compatible.

        A resolver that RAISES falls back too, rather than killing the run. The
        host is reading its own state, so a failure here is a host bug, and the
        useful behaviour is to keep the turn alive on the model it already had
        instead of losing the work in flight to a bad accessor.
        """
        resolver = config.get_model
        if resolver is None:
            return config.model
        try:
            live = resolver()
        except Exception:  # host accessor bug — never fatal to a running turn
            logger.exception("get_model resolver failed; using the run's model")
            return config.model
        # A resolver returning None is the declared "host has nothing better to
        # say" case (see the field), handled the same as having no resolver.
        return live if live is not None else config.model

    @staticmethod
    async def _current_system_blocks(
        context: LoopContext, config: LoopConfig, model: "ModelSpec"
    ) -> list[str]:
        """Resolve session-scoped prompt blocks immediately before one call.

        ``context.system_blocks`` remains the compatible snapshot for hosts that
        do not expose a live provider. Resolver failures are host bugs, but losing
        a running turn over a status-tail refresh would be worse than sending the
        last valid snapshot, so this mirrors :meth:`_current_model`'s fallback.
        """
        resolver = config.get_system_blocks
        if resolver is None:
            return list(context.system_blocks)
        try:
            live = resolver(model)
            if inspect.isawaitable(live):
                live = await live
        except OSError:
            # A durable host-state publication failed. Sending the old prompt
            # could ignore a newly tightened constraint; this call must stop
            # before reaching the provider instead of silently using stale
            # authority. Generic accessor bugs retain the legacy fallback.
            raise
        except Exception:  # host accessor bug — never fatal to a running turn
            logger.exception("get_system_blocks resolver failed; using the run's snapshot")
            return list(context.system_blocks)
        return list(live) if live is not None else list(context.system_blocks)

    # (``_abortable_stream`` is a module-level helper; see below the class.)

    @staticmethod
    def _host_context_tokens_hint(config: LoopConfig) -> int | None:
        """The host's cross-turn context-size seed, or ``None`` when it has
        none — a zero is treated as "nothing reported" because a provider
        never reports an empty context, so 0 here can only be an unset
        default, not a deliberate suppression (that is ``ChatRequest``'s
        explicit ``0``, which only the session's own direct calls set)."""
        reader = config.get_context_tokens_hint
        if reader is None:
            return None
        hint = reader()
        return int(hint) if hint else None

    async def _model_turn(
        self,
        context: LoopContext,
        config: LoopConfig,
        signal: AbortSignal | None,
        effort_ceiling: str | None = None,
        context_tokens_hint: int | None = None,
        echo_fill_route: tuple[str, str] | None = None,
    ) -> AsyncIterator[AgentEvent | _ModelTurnResult]:
        """One provider call: build the request, stream it, assemble the
        assistant message, emitting message_start/update/end events.

        The model is resolved HERE, per call, not once per run — see
        :meth:`_current_model`. ``context_tokens_hint`` is stamped onto the
        request by THIS loop, which owns the conversation the call belongs
        to — never remembered on the shared stream fn, where a subagent's
        registration would overwrite the parent's (review F8).

        ``echo_fill_route`` is the run's standing decision that this route's
        requests must carry the reasoning echo, taken after a provider REFUSED
        one for missing it. It is applied to the RESOLVED spec for the same
        reason ``effort_ceiling`` is: the host's resolver returns its own
        model, so a capability set only on the run's snapshot would reach one
        request and be undone on the next.
        """
        assistant = Message(role="assistant")
        text_parts: list[str] = []
        tool_states: dict[int, dict[str, Any]] = {}
        usage: Usage | None = None
        provider_payload: dict[str, Any] | None = None
        stop_reason = "stop"
        error: str | None = None
        # Set only by the except arm below, from the exception's own flag: the
        # harness cannot import ``providers`` to classify this itself.
        connectivity_loss = False
        # The spec the request below is BUILT with, captured before the stream
        # starts. Declared here rather than read after the ``try`` for two
        # reasons: the resolved ``model`` is assigned inside it, so a reader
        # after the block cannot be sure it exists; and a failover that serves
        # the call reports ITSELF through ``StreamModelEvent``, which is not the
        # spec we built the request from. The run loop's echo fill acts on what
        # we SENT.
        request_model: "ModelSpec | None" = None

        yield TurnStartEvent()
        yield MessageStartEvent(message=assistant)

        try:
            # Resolve every live session setting at the LAST safe point after
            # step-start handlers have run. The model snapshot is passed INTO the
            # async block build so its env label and the request model cannot tear
            # across that await. If /model changes during the build, the next call
            # sees it; this call remains internally consistent, exactly like an
            # already-open provider stream does.
            preparing_at = time.monotonic()
            model = self._current_model(config)
            system_blocks = await self._current_system_blocks(context, config, model)
            # Resolving the prompt can publish a durable host-state update
            # (new goal, credentials, knowledge) into the append-only history.
            # Snapshot AFTER that boundary so the request that prompted the
            # update sees it immediately, while the historical prefix stays
            # byte-stable. Copying before resolution delayed authority by one
            # model call and could send a newly obsolete instruction instead.
            shaped = list(context.messages)
            if config.transform_context is not None:
                outcome = config.transform_context(shaped)
                if inspect.isawaitable(outcome):
                    outcome = await outcome
                shaped = list(outcome)
            converted = config.convert_to_llm(shaped)
            if inspect.isawaitable(converted):
                converted = await converted
            if effort_ceiling is not None:
                # A retreat is in force -- an empty-truncation step-down, or the
                # reasoning-echo recovery's switch to thinking off. The host's
                # resolver returns ITS model, so clamp the RESOLVED spec or the
                # retry goes back out at the rung that just produced silence (or
                # with thinking on, for a request the provider just refused for
                # exactly that). This belongs here rather than in the resolver
                # because the ceiling is loop state, and ``effort_ceiling`` on the
                # request only covers hosts that re-impose an override downstream.
                ladder = model.reasoning_efforts
                current = model.reasoning_effort
                if (
                    current is not None
                    and effort_ceiling in ladder
                    and current in ladder
                    and ladder.index(current) > ladder.index(effort_ceiling)
                ):
                    model = model.model_copy(update={"reasoning_effort": effort_ceiling})
            if echo_fill_route is not None and (model.provider, model.model_id) == echo_fill_route:
                # The same resolved-spec problem, for the other half of the
                # refusal. Nothing else is touched: same route, same effort,
                # same tools — the ONLY delta is that every blank assistant
                # turn now carries its reasoning echo, which is the field the
                # provider said was missing.
                model = model.model_copy(update={"requires_reasoning_echo": True})
            # ``max_tokens`` is deliberately NOT set here. The generation bound
            # is part of the request contract (``harness/types.py``,
            # ``DEFAULT_TURN_OUTPUT_TOKENS``) and every request is filled from
            # that one policy as it is built, so a host cannot go out with a bound
            # that no call site remembered to impose. The number this replaces was
            # not an absent ask but the opposite one: a request that named nothing
            # carried the model's ADVERTISED capability verbatim, which on a 1M
            # aggregate model is 943,718 and is what let a single decision run to
            # 97,189 output tokens (95,098 of them reasoning). A host that wants a
            # different bound names ``max_tokens`` explicitly.
            request_model = model
            request = ChatRequest(
                model=model,
                system_blocks=system_blocks,
                messages=list(converted),
                tools=list(context.tools),
                effort_ceiling=effort_ceiling,
                context_tokens_hint=context_tokens_hint,
                preparation_ms=(time.monotonic() - preparing_at) * 1000,
            )
            stream = _abortable_stream(config.stream_fn(request, signal), signal)
            async for event in stream:
                if isinstance(event, StreamModelEvent):
                    model = event.model
                    yield ModelChangeEvent(
                        provider=model.provider,
                        model_id=model.model_id,
                        effort=model.reasoning_effort,
                        context_window=model.context_window,
                        default_context_window=model.default_context_window,
                        max_context_window=model.max_context_window,
                        context_metadata=True,
                        context_metadata_resolved=model.context_metadata_resolved,
                    )
                elif isinstance(event, StreamStartEvent):
                    # The provider is generating for THIS request. Republish it
                    # as a harness event so supervisors get a boundary that
                    # means "on the wire" — ``MessageStartEvent`` above is
                    # yielded from a placeholder before the request is even
                    # built, so it cannot carry that meaning.
                    yield ProviderTurnStartEvent(
                        response_id=event.response_id,
                        provider=model.provider,
                        model_id=model.model_id,
                    )
                elif isinstance(event, StreamTextDelta):
                    text_parts.append(event.delta)
                    yield MessageUpdateEvent(message=assistant, delta=event.delta)
                elif isinstance(event, StreamToolCallDelta):
                    state = tool_states.setdefault(
                        event.index,
                        {
                            "id": "",
                            "name": "",
                            "arg_parts": [],
                            "bytes": 0,
                            "announced": 0.0,
                            "key": "",
                            # Whether ``key`` is the index-derived placeholder
                            # rather than the provider's id. Tracked as a flag
                            # instead of sniffing a ``compose:`` prefix off the
                            # key, because a provider id is opaque and may
                            # legally look like anything.
                            "placeholder": False,
                            # The placeholder this call was announced under,
                            # once its real id has replaced it. Retained for
                            # the rest of the stream — see the emission below.
                            "supersedes": None,
                            # Bounded copy of the head of the argument stream,
                            # kept only until the intent scrape resolves. `None`
                            # means scanning is over — see below.
                            "head": "",
                            "intent": None,
                        },
                    )
                    if event.id:
                        state["id"] = event.id
                    if event.name:
                        state["name"] += event.name
                    if event.argument_delta:
                        state["arg_parts"].append(event.argument_delta)
                        state["bytes"] += len(event.argument_delta)
                        if state["head"] is not None:
                            state["head"] += event.argument_delta
                            state["intent"] = scan_streaming_intent(state["head"])
                            # Scanning stops for good once the intent has
                            # closed or the window is spent. `i` is injected as
                            # the FIRST schema property, so a leading intent
                            # resolves within a few tokens; re-matching an
                            # ever-growing buffer on every delta of a 14 KB
                            # `write` would burn the stream's own budget for
                            # nothing. Dropping the buffer also caps what this
                            # holds per in-flight call at the scan window.
                            if (
                                state["intent"] is not None
                                or len(state["head"]) >= INTENT_SCAN_LIMIT
                            ):
                                state["head"] = None
                    # Tell the UI a call is being COMPOSED. Without this the
                    # screen holds still for as long as the model takes to
                    # dictate the arguments — minutes for a file — with no tool
                    # card, because the call does not exist until its last token
                    # arrives. Throttled so a token-by-token stream cannot turn
                    # into a repaint storm; the first announcement is immediate
                    # so the row appears the moment the tool's name is known.
                    if state["name"]:
                        # The key is latched on the FIRST announcement and never
                        # recomputed. Evaluated each time, a provider that sends
                        # the name before the id changed the key mid-stream, and
                        # the UI — which keys its rows by it — mounted a second
                        # row for the same call and then marked the abandoned one
                        # interrupted.
                        if not state["key"]:
                            state["key"] = state["id"] or f"compose:{event.index}"
                            state["placeholder"] = not state["id"]
                        elif state["placeholder"] and state["id"]:
                            # The real id has arrived for a row announced under
                            # a placeholder. The identity moves ONCE, here, and
                            # is ANNOUNCED — the placeholder key is otherwise
                            # kept for the rest of the stream while
                            # ``tool_execution_start``/``_end`` carry the real
                            # id, so every consumer that keys rows by
                            # ``tool_call_id`` (the in-flight seed, the TUI's
                            # composing cards, the mobile projection) ends up
                            # holding TWO records for one call. A viewer joining
                            # mid-turn then gets a composing row nothing can
                            # adopt, and turn-end retirement paints it
                            # ``⊘ interrupted`` on a call that SUCCEEDED.
                            #
                            # Announced rather than silently recomputed: silent
                            # recomputation is the bug the latch above exists to
                            # prevent (two rows for one call). Consumers rekey
                            # the row they already have because this frame names
                            # both ids.
                            #
                            # Announced immediately and UNTHROTTLED, then
                            # REPEATED on every later compose frame for this
                            # call. The throttle may swallow an ordinary compose
                            # frame with no loss — a later one carries the same
                            # cumulative size — but a frame carrying the
                            # identity change is not interchangeable, so the
                            # announcement must survive a lossy path. Anything
                            # between here and a viewer may legitimately drop
                            # frames: the per-connection queue compacts and
                            # overflows, and a viewer that was away sees only
                            # what the seed retained. Naming the placeholder on
                            # every subsequent frame makes the hand-off
                            # idempotent — a consumer that already rekeyed finds
                            # nothing to drop — and makes it impossible for the
                            # ONE frame that carried it to be the one lost,
                            # which would strand the row this exists to rescue.
                            #
                            # This is load-bearing against the sibling compose
                            # fold (PR #770), which keeps only the NEWEST frame
                            # per ``tool_call_id``: a single announcement is
                            # precisely what that fold would discard, so the
                            # repeat is what lets the two changes coexist.
                            #
                            # The immediate emission is also the ordering
                            # guarantee: the hand-off reaches the seed before
                            # the ``tool_execution_start`` that follows on the
                            # real id.
                            #
                            # No guard on ``announced`` here: the key is only
                            # ever latched inside ``if state["name"]``, and the
                            # block immediately below stamps ``announced`` on
                            # that same first pass. So a placeholder that exists
                            # at all has already been published, and a
                            # "nothing announced yet" case cannot occur —
                            # verified with a raising probe over the suite.
                            state["supersedes"] = state["key"]
                            state["key"] = state["id"]
                            state["placeholder"] = False
                            yield ToolCallComposeEvent(
                                tool_call_id=state["key"],
                                tool_name=state["name"],
                                argument_bytes=state["bytes"],
                                intent=state["intent"],
                                supersedes_tool_call_id=state["supersedes"],
                            )
                            # Stamp the throttle too: the promotion IS this
                            # window's frame. Without it the block below fires
                            # again in the same iteration and emits a second,
                            # identical frame — harmless, but wasted at exactly
                            # the moment the queue is under pressure.
                            state["announced"] = time.monotonic()
                        now = time.monotonic()
                        first = state["announced"] == 0.0
                        if first or now - state["announced"] >= COMPOSE_NOTICE_INTERVAL_S:
                            state["announced"] = now
                            yield ToolCallComposeEvent(
                                tool_call_id=state["key"],
                                tool_name=state["name"],
                                argument_bytes=state["bytes"],
                                intent=state["intent"],
                                supersedes_tool_call_id=state["supersedes"],
                            )
                elif isinstance(event, StreamUsageEvent):
                    usage = event.usage
                elif isinstance(event, StreamEndEvent):
                    # Flush EVERY latched call here, and make the flush TERMINAL.
                    #
                    # The gate this loop used to carry (`bytes != reported`)
                    # existed to flush what the throttle swallowed: arguments
                    # commonly land in one burst inside a single window, so
                    # without it a row could report a fraction of the call —
                    # or, when the whole payload arrived faster than one
                    # window, never display a size at all. That is still true
                    # and it is now the smaller half of the job.
                    #
                    # This frame is also the ONE the composing row has been
                    # waiting for since it was mounted. This is the instant the
                    # model stopped writing the call, and nothing later in the
                    # step says so: the batch has not run yet
                    # (`_execute_tool_calls` follows the `MessageEndEvent`
                    # yielded below), and the call may be queued behind a
                    # sibling's execution group for that sibling's whole
                    # duration — a `wait(wait_ms=1800000)` ahead of an
                    # `exclusive` tool is the reported half-hour — or never run
                    # at all. So it is emitted UNCONDITIONALLY, for every
                    # latched call, rather than only when the size moved: a
                    # consumer that reads it as "the dictation is over" must
                    # not be able to miss it because the model happened to stop
                    # mid-window with nothing new to report.
                    #
                    # `dictation_complete` is what says so, and it is additive
                    # by design (see `ToolCallComposeEvent`): a viewer that
                    # predates the field ignores it and keeps today's
                    # behaviour, and a viewer meeting a producer that never
                    # sets it sees no such frame and behaves exactly as before.
                    #
                    # `supersedes_tool_call_id` is REPEATED here for the reason
                    # the promotion above already repeats it on every frame:
                    # anything between this and a viewer may legitimately drop
                    # frames, so the one frame that must not be lost is not the
                    # only carrier of the identity hand-off. Applying it twice
                    # stays idempotent — a consumer that already rekeyed finds
                    # nothing to drop.
                    for state in tool_states.values():
                        if not state["name"]:
                            continue
                        if state["placeholder"] and not state["id"]:
                            # The call's id NEVER arrived — the stream is over
                            # and the row is still keyed by its placeholder.
                            # This is the shape the OpenAI Responses API
                            # actually produces: ``clients.py`` yields the id
                            # and name in ONE delta with ``call_id`` defaulting
                            # to the empty string, and the argument deltas that
                            # follow carry no id at all. So the id does not
                            # arrive LATE on that path, it never arrives, and
                            # the late-arrival hand-off above cannot fire.
                            #
                            # Left alone, ``_assemble_tool_call`` mints a fresh
                            # uuid for the ToolCall and execution proceeds under
                            # an id the composing row has never seen — stranding
                            # it exactly as a late id would.
                            #
                            # So mint the identity HERE, once, and announce it,
                            # rather than letting the ToolCall mint a different
                            # one silently. Same factory and same shape the
                            # ToolCall would have used, deliberately: this id
                            # goes back to the provider as Anthropic's
                            # ``tool_use.id`` and Responses' ``call_id``, so it
                            # must be an ordinary opaque token. Promoting the
                            # PLACEHOLDER onto the wire instead would put
                            # ``compose:{index}`` in that field — a value no
                            # provider has agreed to accept, for no gain, since
                            # nothing keys off the prefix.
                            state["id"] = uuid.uuid4().hex[:12]
                            # Unguarded for the same reason as the late-arrival
                            # hand-off above: reaching here means a placeholder
                            # was latched, which only happens on a pass that
                            # also publishes the row.
                            state["supersedes"] = state["key"]
                            state["key"] = state["id"]
                            state["placeholder"] = False
                        # No throttle bookkeeping to stamp here: the emission
                        # below is not gated on ``bytes != reported`` any more
                        # (a dictation that ends without another delta must still
                        # be told it ended), and this block is the call's last
                        # word in this stream, so nothing reads a stamp from it.
                        yield ToolCallComposeEvent(
                            tool_call_id=state["key"] or "compose:0",
                            tool_name=state["name"],
                            argument_bytes=state["bytes"],
                            intent=state["intent"],
                            supersedes_tool_call_id=state["supersedes"],
                            dictation_complete=True,
                        )
                    stop_reason = event.stop_reason
                    if event.usage is not None:
                        usage = event.usage
                    provider_payload = event.provider_payload
                    error = event.error
                    if stop_reason == "refusal" and not error:
                        # Belt-and-braces: every wire client composes a refusal
                        # message, but a bare "refusal" end from a client that
                        # forgot must still say SOMETHING — a refusal nobody can
                        # see is the exact bug this stop_reason exists to fix.
                        error = "model refused the request (no details from provider)"
        except asyncio.CancelledError:
            # Assemble before re-raising. A hard cancel skips the tail of this
            # function entirely, so without this the message a consumer still
            # holds from ``MessageStartEvent`` would keep the empty content it
            # was created with, where before the per-delta rebuild it held every
            # delta received up to the cut. Nothing downstream reaches
            # ``context.messages`` on this path, but the object is observable,
            # and losing the partial answer on cancel is a behaviour change this
            # optimization has no business making.
            #
            # Guarded for the same reason as the assembly at the end of this
            # function: a cancel before any text delta must leave ``content``
            # as the empty LIST it was constructed with, not a list holding an
            # empty text block.
            if text_parts:
                assistant.content = [TextContent(text="".join(text_parts))]
            raise
        except Exception as exc:
            # `error` below is handed straight to the UI, which prints it as a
            # single "× HTTP 400: ..." line. Re-emitting the same failure as a
            # traceback duplicated it across whatever was on screen for zero
            # extra information. Unexpected types keep the stack — for a defect
            # the frames are the only clue there is.
            logger.warning(
                "model stream failed: %s", exc, exc_info=not isinstance(exc, RenderedStreamError)
            )
            stop_reason = "aborted" if (signal is not None and signal.aborted) else "error"
            error = error or str(exc)
            # A stream the NETWORK cut, as opposed to one a provider failed. Read
            # off the exception rather than re-classified here: the single
            # definition lives in ``providers.failover.is_connectivity_loss`` and
            # rides out on ``RenderedStreamError.connectivity_loss``. ``getattr``
            # because this arm also catches defects and third-party exceptions,
            # which carry no such attribute.
            #
            # Not applied to an ABORT: the user pressed the key, and a turn they
            # stopped must stay stopped even if a connectivity error is what the
            # cancelled socket happened to raise on its way out.
            connectivity_loss = stop_reason == "error" and bool(
                getattr(exc, "connectivity_loss", False)
            )

        if signal is not None and signal.aborted:
            # A stream CUT by the abort ends without its ``StreamEndEvent``, so
            # the local default ("stop") would otherwise stand and the turn
            # would read as a clean finish — the loop would go on to make
            # another model call for a turn the user has already stopped.
            # Recording the truth here is what lets the single ``("error",
            # "aborted")`` branch upstream pair the dangling calls and end the
            # run, instead of the abort having to be re-detected in each place.
            stop_reason = "aborted"

        # Assemble the text ONCE, here, rather than on every delta.
        #
        # This assignment used to live inside the delta loop, rebuilding the
        # whole accumulated string per token. That is quadratic in the length
        # of the response: a 2000-delta answer allocates ~2.4 GB of
        # immediately-dead strings to hold 2.4 MB of live text, and the churn
        # outruns the allocator's ability to return pages, so RSS climbs into
        # the gigabytes. With several subagents streaming at once it exhausted
        # system memory on a 36 GB machine.
        #
        # Consumers of MessageUpdateEvent must therefore append ``event.delta``
        # rather than re-read the message: the TUI keeps ``_assistant_buffer``,
        # the mobile projection appends to ``row.text`` (and documents "never
        # re-read the whole message"), headless print writes the delta, and the
        # server bridge in ``server/utils/operator.py`` accumulates its own
        # ``record.message`` for exactly this reason. The message itself does
        # not reach ``context.messages`` until the turn completes, and
        # MessageEndEvent carries the authoritative final text.
        #
        # Placed after the abort/error handling so a cut or failed stream still
        # reports the text that did arrive — the frozen row on an aborted turn
        # is what the user is left reading. The hard-cancel path re-raises
        # above and assembles there for the same reason.
        #
        # ONLY when there is text. A turn that emits tool calls and no prose is
        # ordinary — it is what every "call the tool, say nothing" step looks
        # like — and this assignment used to be unreachable for it, because it
        # lived inside the delta loop and no delta ever arrived. Hoisting it out
        # made it run unconditionally, so such a turn started carrying
        # ``[TextContent(text="")]`` where it used to carry ``[]``. Anthropic
        # rejects that on the NEXT request of the turn with
        # ``HTTP 400: messages: text content blocks must be non-empty``, which
        # kills the run outright. The empty list is what the Message was
        # constructed with and what every provider already handles.
        if text_parts:
            assistant.content = [TextContent(text="".join(text_parts))]
        assistant.tool_calls = [
            self._assemble_tool_call(state) for _, state in sorted(tool_states.items())
        ]
        assistant.stop_reason = stop_reason
        assistant.usage = usage
        if stop_reason == "refusal" and error:
            # The refusal message otherwise lives only in the run's AgentEndEvent
            # and dies with it: a resumed session replaying this message could
            # say "the model refused" but not WHAT it said, which is the half of
            # the diagnosis that decides between rephrasing and switching models.
            # ``provider_payload`` is the established home for harness bookkeeping
            # that must survive into the transcript (see ``pruned``/``details``);
            # wire clients never replay these keys to providers.
            provider_payload = {**(provider_payload or {}), "refusal": error}
        assistant.provider_payload = provider_payload
        # Token-cache settle gate (review RC-20): the message is finalized
        # (usage/stop_reason set), so any provisional cached estimate must be
        # dropped before the message enters the context. Lazy — a missing
        # compaction package degrades to no caching at all.
        try:
            from local_operator.compaction import tokens as _compaction_tokens

            _compaction_tokens.invalidate_message_cache(assistant)
        except ImportError:
            pass

        yield MessageEndEvent(message=assistant)
        yield _ModelTurnResult(
            message=assistant,
            stop_reason=stop_reason,
            error=error,
            connectivity_loss=connectivity_loss,
            model=request_model,
        )

    @staticmethod
    def _assemble_tool_call(state: dict[str, Any]) -> ToolCall:
        raw = "".join(state["arg_parts"]).strip()
        arguments: dict[str, Any] = {}
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    arguments = parsed
            except json.JSONDecodeError:
                # Leave arguments empty; validation reports the bad JSON.
                pass
        call = ToolCall(name=state["name"], arguments=arguments, raw_arguments=raw or None)
        if state["id"]:
            call.id = state["id"]
        return call

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    @staticmethod
    def _not_run_compose_frame(call: ToolCall, reason: str) -> ToolCallComposeEvent:
        """The terminal compose frame for a call that will never START.

        A call parked at planning (an unknown tool, invalid arguments, a
        duplicate id whose twin won the slot) or skipped by steering gets no
        ``tool_execution_start`` and — deliberately — no ``tool_execution_end``
        either: the API server matches tool records by id, and an end with no
        start "either resurrects a record that was never opened or, when a
        duplicate id collides, closes the REAL call's record early" (see
        ``_execute_batch.park``). It also never ran, so a start would claim the
        tool executed, moving approval, reporting and the analytics chokepoint
        (``_report_tool_call``) with it.

        That suppression is right, and it left the COMPOSE surface with no
        ending at all: the row the model's dictation announced stayed
        ``composing…``, with a ticking clock, until the TURN ended — settled
        then as ``never sent · N composed`` under the word ``interrupted``,
        which describes a call that was never interrupted. This frame is the
        ending the compose surface was missing, and it says nothing to the API
        server: it rides the same wire the announcement did.

        ``reason`` is the synthetic result's own text, so the row states the
        failure in the harness's words rather than in a second vocabulary this
        path would have to keep in step. It is bounded to one clipped line,
        because it rides the live relay and the reconnect seed, both of which
        budget text (``LIVE_EVENT_TEXT_FRAME_BUDGET_CHARS``) — the whole
        synthetic result is the mistake this avoids.

        Only an ANNOUNCED call gets one: the compose surface exists solely for
        a call whose dictation reached a viewer, and a frame for a call nobody
        was shown would mount a row for something that was never on screen.
        Every call reaching here has a name (``_assemble_tool_call`` copies the
        latched one), and the announcement is minted on the first name
        fragment, so a truthy name is exactly "this call was announced".
        """
        text = " ".join((reason or "").split()) or "Tool call not run"
        if len(text) > NOT_RUN_REASON_MAX_CHARS:
            text = text[: NOT_RUN_REASON_MAX_CHARS - 1].rstrip() + "…"
        return ToolCallComposeEvent(
            tool_call_id=call.id or "compose:0",
            tool_name=call.name,
            # The assembled argument payload, which is the same measurement the
            # streaming frames reported: their byte count is the sum of the
            # argument deltas, and those are what ``raw_arguments`` joined.
            argument_bytes=len(call.raw_arguments or ""),
            dictation_complete=True,
            not_run_reason=text,
        )

    def _not_run_frames(self, calls: list[ToolCall], reason: str) -> list[ToolCallComposeEvent]:
        """Terminal frames for a set of calls that will never START, one reason.

        The steering skip's shape: every call it drops after the batch's first
        slot shares one verdict, so they share one reason. The planning-failure
        path above carries a per-call reason and calls
        :meth:`_not_run_compose_frame` directly for that reason.
        """
        return [self._not_run_compose_frame(call, reason) for call in calls if call.name]

    async def _execute_tool_calls(
        self,
        calls: list[ToolCall],
        context: LoopContext,
        config: LoopConfig,
        signal: AbortSignal | None,
        results: list[ToolResult],
    ) -> AsyncIterator[AgentEvent]:
        """Resolve, validate and schedule one batch of calls.

        ``shared`` tools run in parallel, ``exclusive`` tools run alone. When
        ``interrupt_mode == "immediate"`` and steering is queued, remaining
        calls are skipped with synthetic results. Duplicate call ids within
        the batch are deduplicated (first wins; later duplicates become
        skipped results so tool_use/tool_result pairing stays legal).
        Approval prompts happen per-call INSIDE the tool task (after
        ``tool_execution_start``), so the UI shows the call while waiting and
        skipped calls never prompt.
        """
        seen_ids: set[str] = set()
        plan: list[_PlannedCall] = []
        for call in calls:
            if call.id in seen_ids:
                # Duplicate call id within one batch: first wins; the duplicate
                # is paired with a skipped result and never executes.
                logger.warning("duplicate tool call id %s dropped from batch", call.id)
                plan.append(
                    _PlannedCall(
                        call=call,
                        failure=self._synthetic_result(
                            call,
                            f"Duplicate call id '{call.id}' skipped.",
                            details={FAULT_KEY: FAULT_DUPLICATE_ID},
                        ),
                    )
                )
                continue
            seen_ids.add(call.id)
            plan.append(await self._plan_call(call, context, config))
        # Announce the batch's never-run verdicts BEFORE anything executes.
        #
        # A call with a `failure` here has been judged and will never start:
        # the tool is unknown, the arguments did not validate, or a duplicate id
        # lost to its twin (which parks up front in `_execute_batch`, after the
        # group runs). The verdict exists NOW, and the row it belongs to is
        # already on screen claiming the model is still writing the call — so
        # the compose surface is told now rather than at turn end, when the
        # retirement pass would settle it under the word `interrupted` for a
        # call that was never interrupted.
        #
        # Emitted before the group runs, and that order is deliberate: these
        # calls take no part in the execution that follows, so a viewer should
        # be able to stop waiting for them at the moment the harness decided.
        for failure in plan:
            if failure.failure is not None:
                yield self._not_run_compose_frame(failure.call, failure.failure.text)
        index = 0
        first_slot = True
        while index < len(plan):
            # ``_peek_steering`` and NOT ``_peek_interrupt``, which is a
            # correctness rule rather than an oversight: skipping the batch's
            # remaining calls is right when the USER redirected the work, and
            # wrong for a pending fork, which has redirected nothing and must
            # leave the parent's turn intact. See ``LoopConfig.has_pending_fork``.
            if (
                not first_slot
                and config.interrupt_mode == "immediate"
                and self._peek_steering(config)
            ):
                for remaining in plan[index:]:
                    # Marked at the source like every other fault, though this
                    # site bypasses ``park`` and so is not recorded: a call
                    # steering skipped before it was ever scheduled is excluded
                    # from both rates anyway, so its absence moves no figure.
                    results.append(
                        self._synthetic_result(
                            remaining.call,
                            SKIPPED_RESULT_TEXT,
                            details={FAULT_KEY: FAULT_SKIPPED},
                        )
                    )
                # ...and the SAME ending the batch's other never-run calls get,
                # for the same reason and at the same instant: this site
                # bypasses `_execute_batch` entirely (it is outside the
                # per-call loop, and below it the remaining calls are never
                # even scheduled), so without this their rows would sit
                # `composing…` until the turn died and the user would have to
                # infer the skip from the steering notice.
                #
                # After the loop rather than inside it: these frames describe
                # the batch's whole remaining tail, and emitting one set per
                # member would hand a viewer the same verdict twice per call.
                #
                # Only the calls that had NOT already failed planning: those
                # were settled with their own verdict before the batch ran, and
                # a second terminal frame would relabel a `Tool not found` row
                # as a steering skip.
                for frame in self._not_run_frames(
                    [item.call for item in plan[index:] if item.failure is None],
                    SKIPPED_RESULT_TEXT,
                ):
                    yield frame
                break

            if not _batches_shared(plan[index]):
                async for event in self._execute_batch(
                    plan[index : index + 1], context, config, signal, results
                ):
                    yield event
                index += 1
            else:
                end = index
                resources: set[str] = set()
                keyed = plan[index].resources is not None
                while end < len(plan) and _batches_shared(plan[end]):
                    item = plan[end]
                    # Keep a barrier between ordinary shared reads and keyed
                    # writes. Only tools explicitly declaring independent
                    # resources gain concurrency; a read immediately following
                    # a write must still observe it without having to infer
                    # arbitrary read/grep footprints from their arguments.
                    if (item.resources is not None) != keyed:
                        break
                    keys = set(item.resources or ())
                    if resources & keys:
                        break
                    resources.update(keys)
                    end += 1
                async for event in self._execute_batch(
                    plan[index:end], context, config, signal, results
                ):
                    yield event
                index = end
            first_slot = False

    async def _plan_call(
        self, call: ToolCall, context: LoopContext, config: LoopConfig
    ) -> _PlannedCall:
        """Resolve + validate one call. Approval is deliberately NOT here: it
        happens inside the runner after ``tool_execution_start`` so skipped
        calls never prompt (see :meth:`_runner_result`)."""
        tool = next((t for t in context.tools if t.name == call.name), None)
        if tool is None and config.resolve_fallback_tool is not None:
            tool = config.resolve_fallback_tool(call.name)
        if tool is None:
            return _PlannedCall(
                call=call,
                failure=self._synthetic_result(
                    call,
                    f"Tool not found: {call.name}",
                    details={FAULT_KEY: FAULT_UNKNOWN_TOOL},
                ),
            )

        # Lift the intent off BEFORE validation, and before anything else sees
        # the arguments. Both halves of that order are load-bearing:
        #
        # * Validating first would let narration cancel work. A model that
        #   streamed `"i": 3` fails `validate_tool_arguments` (it type-checks
        #   every declared property), and a planning failure parks a synthetic
        #   result WITHOUT ever emitting `tool_execution_start` — so a
        #   cosmetic field would silently swallow the call the user asked for.
        #   A malformed intent costs the narration and nothing else.
        # * Leaving it in `args` would break the call at the other end: every
        #   builtin params model is pydantic with `extra="forbid"`.
        #
        # `intent_is_injected` is what keeps this from stealing a real
        # argument: an MCP server that declares its own `i` never had ours
        # injected, so its value is left in `args` and forwarded.
        args = dict(call.arguments)
        intent: str | None = None
        if INTENT_FIELD in args and intent_is_injected(tool.parameters):
            intent = sanitize_intent(args.pop(INTENT_FIELD))

        errors = validate_tool_arguments(tool, args, call.raw_arguments)
        if errors:
            return _PlannedCall(
                call=call,
                tool=tool,
                failure=self._synthetic_result(
                    call,
                    "Invalid arguments: " + "; ".join(errors),
                    details={FAULT_KEY: FAULT_INVALID_ARGUMENTS},
                ),
            )
        resources: tuple[str, ...] | None = None
        if tool.resource_keys is not None:
            try:
                # Canonical paths and inode identities may probe a slow mount.
                # Planning must not stall unrelated sessions on the loop thread.
                keys = await asyncio.to_thread(
                    tool.resource_keys, args, (context.tool_context or ToolContext()).cwd
                )
                if (
                    isinstance(keys, tuple)
                    and len(keys) <= 32
                    and all(isinstance(key, str) and 0 < len(key) <= 4096 for key in keys)
                ):
                    resources = keys
            except Exception:
                # Unknown resource identity keeps the legacy exclusive
                # barrier; a failed optimization must never create a race.
                logger.debug("tool resource identity failed for %s", call.name, exc_info=True)
        return _PlannedCall(call=call, tool=tool, args=args, intent=intent, resources=resources)

    async def _runner_result(
        self,
        item: _PlannedCall,
        context: LoopContext,
        config: LoopConfig,
        signal: AbortSignal | None,
        queue: asyncio.Queue[AgentEvent | _ToolDone | _BatchDone],
    ) -> ToolResult:
        """Execute one planned call and return its result.

        Approval (write/exec tiers with a configured callback) runs here —
        AFTER ``tool_execution_start`` has been emitted — so the UI shows the
        pending call while the user decides, and a denied call never executed.
        """
        call = item.call
        if item.failure is not None or item.tool is None:
            return item.failure or self._synthetic_result(call, "Tool not found.")
        tool = item.tool

        tool_context = context.tool_context
        # A per-call tier override wins over the tool's static tier: a tool
        # that is write-tier for its worst op (hub resume) still has read-only
        # ops (hub list/peek) that must not prompt.
        tier = (
            tool.call_approval_tier(call.arguments)
            if tool.call_approval_tier is not None
            else tool.approval_tier
        )
        if (
            tier in ("write", "exec")
            and tool_context is not None
            and tool_context.request_approval is not None
        ):
            summary = self._approval_summary(tool, call, tool_context.cwd, tool_context)
            try:
                approved = await ask_approval(
                    tool_context.request_approval,
                    sanitize_prompt_line(call.name, limit=120),
                    summary,
                    tool_context.job_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # A gate that CRASHED has not decided anything, and reporting it
                # as "User denied approval" blamed the user for our own bug: the
                # report that comes back is "it denied my command", so nobody
                # goes looking for the exception. Two bash calls really did read
                # `User denied approv…` in a session whose band said
                # `! auto-approve` — a combination no user can produce — after a
                # widget raised inside the TUI's gate.
                #
                # The call is still NOT run: a gate that cannot answer has
                # granted nothing, and that half was always right. What changes
                # is that the failure now says so in its own words, and at ERROR
                # with the stack, because a fault inside a SECURITY gate is not
                # a warning.
                #
                # Deliberately NOT re-raised out of the loop, though the argument
                # is close. Every tool call has to come back paired with a result
                # (``_execute_batch``), and the sibling handler below already
                # answers a raising TOOL with an error result rather than a dead
                # turn; trading a misleading result for an aborted turn is not
                # the improvement. Being unmistakable is — the conservative-
                # looking silent denial is exactly what hid this for however long
                # it has been here, so the fix is loudness, not severity.
                logger.error(
                    "approval gate raised for %s; the call was NOT run",
                    call.name,
                    exc_info=True,
                )
                detail = str(exc).strip()
                named = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
                # `call.name` is MODEL-controlled and lands on a card the same
                # way the exception text does, so it gets the same guard the
                # approval prompt above already gives it (line 687). Both
                # outcomes sanitize it: an escape sequence in a tool name is a
                # cleared terminal whichever branch prints it.
                safe_name = sanitize_prompt_line(call.name, limit=120)
                return self._synthetic_result(
                    call,
                    # FIRST LINE is the card's failure label (the TUI takes
                    # `_first_line(result_text)`), which is the row read as
                    # read as `User denied approv…`. It therefore carries the
                    # whole diagnosis on its own and the detail follows below,
                    # where the expansion and the model both get it.
                    f"Approval gate failed for '{safe_name}' — the call was not run.\n"
                    f"{sanitize_prompt_line(named, limit=200)}\n"
                    "This is a harness fault, not a refusal by the user; the stack is in "
                    "the log.",
                    details={"__approval_gate_failed": True, FAULT_KEY: FAULT_GATE_FAILED},
                )
            if not approved:
                return self._synthetic_result(
                    call,
                    f"User denied approval for '{sanitize_prompt_line(call.name, 120)}'.",
                    details={FAULT_KEY: FAULT_DENIED},
                )

        def on_update(update: AgentToolUpdate) -> None:
            queue.put_nowait(
                ToolExecutionUpdateEvent(
                    tool_call_id=call.id, tool_name=tool.name, partial_result=update
                )
            )

        try:
            execution_context = context.tool_context or ToolContext()
            if tool.name == "eval":
                # The worker receives a request-owned capability, never the
                # Session object or an MCP client. Nested calls reuse THIS
                # executor's resolver, schema validation, role filters, approval
                # tier and events; Python composition cannot bypass any gate.
                # Sequential nested calls inherit eval's outer exclusive slot,
                # so they cannot reorder neighbouring native tool calls.
                async def dispatch_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
                    nested = ToolCall(
                        id=f"{call.id}:{uuid.uuid4().hex}", name=name, arguments=arguments
                    )
                    if name == "eval":
                        # DELIBERATELY not recorded and not fault-marked. This
                        # is a structural refusal of RECURSION, not a judgement
                        # about a tool call: no tool was resolved, nothing was
                        # dispatched, and it fits none of the classes the two
                        # rates are defined over. Counting it would need a ninth
                        # fault class that neither rate consumes, which would
                        # move the denominator without informing either figure.
                        return self._synthetic_result(
                            nested, "Recursive eval tool calls are not supported."
                        ).model_dump(mode="json")
                    if signal is not None and signal.aborted:
                        return self._synthetic_result(
                            nested, ABORTED_RESULT_TEXT, details={FAULT_KEY: FAULT_ABORTED}
                        ).model_dump(mode="json")
                    planned = await self._plan_call(nested, context, config)
                    if planned.failure is not None or planned.tool is None:
                        failure = planned.failure or self._synthetic_result(
                            nested, "Tool not found.", details={FAULT_KEY: FAULT_UNKNOWN_TOOL}
                        )
                        # This bridge never reaches ``park`` — it emits its own
                        # start/end events — so it must report here or every
                        # eval-driven tool call goes uncounted, which would
                        # quietly understate exactly the composition-heavy runs
                        # the accuracy figure is most wanted for.
                        self._report_tool_call(config, context, planned, failure, origin="nested")
                        return failure.model_dump(mode="json")
                    started = time.monotonic()
                    queue.put_nowait(
                        _tool_start_event(
                            tool_call_id=nested.id,
                            tool_name=name,
                            args=planned.args,
                            intent=planned.intent,
                        )
                    )
                    try:
                        result = await self._runner_result(planned, context, config, signal, queue)
                    except asyncio.CancelledError:
                        result = self._synthetic_result(
                            nested, ABORTED_RESULT_TEXT, details={FAULT_KEY: FAULT_ABORTED}
                        )
                        result.duration_s = time.monotonic() - started
                        self._report_tool_call(config, context, planned, result, origin="nested")
                        queue.put_nowait(
                            ToolExecutionEndEvent(
                                tool_call_id=nested.id,
                                tool_name=name,
                                result=result,
                                duration_s=result.duration_s,
                                is_error=True,
                            )
                        )
                        raise
                    # Redact before the result crosses back into arbitrary
                    # Python, the same text policy used for native history.
                    if config.redact_tool_result is not None:
                        result = result.model_copy(
                            update={
                                "content": [
                                    (
                                        TextContent(text=config.redact_tool_result(block.text))
                                        if isinstance(block, TextContent)
                                        else block
                                    )
                                    for block in result.content
                                ]
                            }
                        )
                    result.duration_s = time.monotonic() - started
                    queue.put_nowait(
                        ToolExecutionEndEvent(
                            tool_call_id=nested.id,
                            tool_name=name,
                            result=result,
                            duration_s=result.duration_s,
                            is_error=result.is_error,
                        )
                    )
                    self._report_tool_call(config, context, planned, result, origin="nested")
                    return result.model_dump(mode="json")

                execution_context = execution_context.model_copy(
                    update={"dispatch_tool": dispatch_tool}
                )
            return await tool.execute(call.id, item.args, signal, on_update, execution_context)
        except asyncio.CancelledError:
            raise
        except InvalidToolArgumentsError as exc:
            # An argument-SHAPE rejection raised from inside a tool body. This
            # is the one exception class that carries its own fault
            # classification, and this is the only place it is translated —
            # the marker is set here, where the reason is known, rather than
            # inferred later from the message text (``_classify_fault``).
            #
            # Why it needs its own branch: schema validation
            # (``validate_tool_arguments``) can only check a JSON-Schema type,
            # and a tool's real argument grammar is often finer. ``read``'s
            # ``range`` is typed ``str | None``, so the literal-quoted
            # ``'"270-330"'`` the model actually emitted passes validation and
            # fails in the parser — landing in the generic handler below and
            # being recorded as ``execution``, i.e. laundered out of the
            # model-accuracy figure. Logged at DEBUG, not WARNING: unlike the
            # handler below this is a MODEL mistake, not a harness defect, and
            # it is already counted where it belongs.
            logger.debug("tool %s rejected malformed arguments: %s", tool.name, exc)
            return ToolResult(
                tool_call_id=call.id,
                tool_name=tool.name,
                is_error=True,
                content=[TextContent(text=f"invalid arguments: {exc}")],
                details={FAULT_KEY: FAULT_INVALID_ARGUMENTS},
            )
        except Exception as exc:
            logger.warning("tool %s raised", tool.name, exc_info=True)
            return ToolResult(
                tool_call_id=call.id,
                tool_name=tool.name,
                is_error=True,
                content=[TextContent(text=f"Tool raised: {exc}")],
            )

    async def _execute_batch(
        self,
        batch: list[_PlannedCall],
        context: LoopContext,
        config: LoopConfig,
        signal: AbortSignal | None,
        results: list[ToolResult],
    ) -> AsyncIterator[AgentEvent]:
        """Run one concurrency batch, streaming start/update/end events out as
        the tools produce them (order of completion, per-slot results kept).

        ``interruptible`` tools race against a steering poll (every
        ``STEERING_INTERRUPT_POLL_S`` while ``interrupt_mode == "immediate"``);
        on a steering signal the tool task is cancelled and paired with a
        synthetic skipped result so tool_use/tool_result pairing stays legal.
        The poll uses the URGENT peek: courtesy injections (a scheduled wake
        riding the busy path) share the steering queue but must wait for the
        next successful tool boundary rather than kill a running tool.
        On generator cancellation (GeneratorExit) every runner task is
        cancelled before this generator returns.

        An ABORT is different from steering and stronger than both: a watcher
        cancels EVERY runner in the batch the instant the signal fires, whether
        or not the tool declared itself ``interruptible``. Steering is a
        redirect and may only interrupt a tool that opted in; an abort is the
        user pressing Esc, and a stop that waits for the slowest call in the
        batch is not a stop. Before this, a batch of non-interruptible tools
        (a `read` of a huge tree, an `edit`, an MCP call) ignored the signal
        entirely and the turn ended only when the last one finished — measured
        at multiple seconds after the keypress, with the UI still painting the
        work as live.

        ``interruptible`` therefore keeps exactly one meaning — "steering may
        interrupt this" — instead of quietly doubling as "the user may stop
        this", which is not a property any tool should get to decline.

        Results are keyed by BATCH SLOT, not call id: a model can emit two
        calls with the same id in one batch, and keying by id made the two
        slots collide into one result (duplicate tool_result ids on the wire,
        which Anthropic rejects). A slot whose call failed planning never
        runs; its synthetic result is parked in its slot up front.
        """
        queue: asyncio.Queue[AgentEvent | _ToolDone | _BatchDone] = asyncio.Queue()
        results_by_slot: list[ToolResult | None] = [None] * len(batch)
        tasks: list[asyncio.Task[None]] = []
        # Slot -> the monotonic instant its runner announced the start. A SET of
        # started slots was enough while every end event came from `park`, which
        # closes over its runner's own `started_at`; the post-abort backfill has
        # no such closure and was therefore the one emitter that could not
        # measure what it was reporting (review round 2, MAJOR-3). Membership is
        # unchanged — `slot in started_at_by_slot` reads the same as before — so
        # this widens what the batch remembers rather than how it decides.
        started_at_by_slot: dict[int, float] = {}
        peek = (
            config.has_urgent_steering_messages
            if config.has_urgent_steering_messages is not None
            else config.has_steering_messages
        )
        # A pending fork is also a reason to poll: without it, a fork requested
        # during a long interruptible tool would wait out the whole tool (a
        # ten-minute ``wait``, a slow MCP call) before reaching its boundary.
        poll_interruptible = config.interrupt_mode == "immediate" and (
            peek is not None or config.has_pending_fork is not None
        )
        # Set by the abort watcher so the runners' cancellation handlers can
        # tell an abort apart from a steering interrupt and label their
        # synthetic results correctly.
        aborting = False
        # WHEN the abort landed, on the same monotonic clock the runners stamp
        # their spans from. The post-abort backfill needs an END instant for a
        # tool it will never see finish, and every other candidate is a property
        # of the harness rather than of the tool: see the stamp below.
        aborted_at: float | None = None

        def park(
            slot: int,
            item: _PlannedCall,
            result: ToolResult,
            *,
            duration_s: float | None = None,
        ) -> None:
            # This executor owns the real start/end boundary. Persisting that
            # fact here keeps every later presenter from timing its own paint.
            if duration_s is not None:
                result.duration_s = max(0.0, duration_s)
            results_by_slot[slot] = result
            # A call that never STARTED never gets an end. Planning failures —
            # an unknown tool, or a duplicate id whose twin won the slot — are
            # parked up front with no task and no `ToolExecutionStartEvent`, so
            # announcing their end describes a lifecycle no consumer ever saw
            # begin: the API server matches by id and either resurrects a record
            # that was never opened or, when a duplicate id collides, closes the
            # REAL call's record early and publishes two TOOL_ENDs for one
            # TOOL_START.
            #
            # Suppressed HERE, at the single source, rather than downstream.
            # The event has two readers — the drain loop while the batch is live
            # and the post-abort flush after it gives up — and a guard in either
            # one alone leaves the other emitting it (R4-1 fixed the flush, R5-1
            # was the drain doing the same thing a moment earlier). The result
            # still parks, so the WIRE stays paired; only the event is withheld.
            started = slot in started_at_by_slot
            if started:
                queue.put_nowait(
                    ToolExecutionEndEvent(
                        tool_call_id=item.call.id,
                        tool_name=item.tool.name if item.tool is not None else item.call.name,
                        result=result,
                        duration_s=result.duration_s,
                        is_error=result.is_error,
                    )
                )
            # Every tool call the model emitted passes through here exactly
            # once — dispatched, rejected at planning, denied, aborted or
            # skipped — which is what makes this the one honest chokepoint for
            # the accuracy figure. ``_report_tool_call`` is put_nowait-only and
            # swallows everything: this runs ON THE EVENT LOOP inside a live
            # turn, so analytics may cost neither latency nor a raised turn.
            self._report_tool_call(config, context, item, result, origin="model")
            queue.put_nowait(_TOOL_DONE)

        async def runner(slot: int, item: _PlannedCall) -> None:
            tool_name = item.tool.name if item.tool is not None else item.call.name
            started_at = time.monotonic()
            started_at_by_slot[slot] = started_at
            await queue.put(
                # `item.args`, not `item.call.arguments`: the event must show
                # what the tool is actually being run with, and those two now
                # differ by the lifted `i`. Leaking it here would caption the
                # tool row with the intent — the TUI's argument summary scans
                # values for a row identity — reinstating on the card the
                # duplication that splitting fact from claim removes.
                _tool_start_event(
                    tool_call_id=item.call.id,
                    tool_name=tool_name,
                    args=item.args,
                    intent=item.intent,
                )
            )
            try:
                result = await self._runner_result(item, context, config, signal, queue)
            except asyncio.CancelledError:
                # Cancelled (abort/GeneratorExit): pair the call with a
                # synthetic aborted result so tool_use/tool_result pairing
                # stays legal, then propagate the cancellation.
                park(
                    slot,
                    item,
                    self._synthetic_result(
                        item.call, ABORTED_RESULT_TEXT, details={FAULT_KEY: FAULT_ABORTED}
                    ),
                    duration_s=time.monotonic() - started_at,
                )
                raise
            park(slot, item, result, duration_s=time.monotonic() - started_at)

        async def interruptible_runner(slot: int, item: _PlannedCall) -> None:
            tool_name = item.tool.name if item.tool is not None else item.call.name
            started_at = time.monotonic()
            started_at_by_slot[slot] = started_at
            await queue.put(
                _tool_start_event(
                    tool_call_id=item.call.id,
                    tool_name=tool_name,
                    args=item.args,
                    intent=item.intent,
                )
            )
            tool_task = asyncio.ensure_future(
                self._runner_result(item, context, config, signal, queue)
            )
            try:
                try:
                    while True:
                        done, _pending = await asyncio.wait(
                            {tool_task}, timeout=STEERING_INTERRUPT_POLL_S
                        )
                        if tool_task in done:
                            break
                        if signal is not None and signal.aborted:
                            break
                        if self._peek_interrupt(config):
                            tool_task.cancel()
                            break
                finally:
                    if not tool_task.done():
                        tool_task.cancel()
                try:
                    result = await tool_task
                except asyncio.CancelledError:
                    # The INNER task was cancelled (steering, or the run
                    # aborting): synthesize a skipped/aborted result so the
                    # call stays paired.
                    aborted = signal is not None and signal.aborted
                    text = ABORTED_RESULT_TEXT if aborted else SKIPPED_RESULT_TEXT
                    result = self._synthetic_result(
                        item.call,
                        text,
                        details={FAULT_KEY: FAULT_ABORTED if aborted else FAULT_SKIPPED},
                    )
                park(slot, item, result, duration_s=time.monotonic() - started_at)
            except asyncio.CancelledError:
                # THIS coroutine was cancelled from outside — which is what the
                # batch-wide abort watcher does. Without this the cancellation
                # unwound straight out and ``park`` was never reached, so the
                # call got a start event and no END event. The backfill below
                # keeps the WIRE legal, but it does not emit events: every
                # consumer other than the TUI (which retires orphaned cards at
                # the turn boundary) was left with a tool that never finished —
                # the API server holds the execution record IN_PROGRESS forever
                # and never publishes a TOOL_END on its SSE stream.
                #
                # It matters far more here than for the plain ``runner``:
                # ``interruptible`` covers bash, eval, wait, hub, ask, web
                # search and EVERY MCP tool, i.e. most of a real batch.
                park(
                    slot,
                    item,
                    self._synthetic_result(
                        item.call, ABORTED_RESULT_TEXT, details={FAULT_KEY: FAULT_ABORTED}
                    ),
                    duration_s=time.monotonic() - started_at,
                )
                raise

        async def abort_watcher() -> None:
            """Cancel every runner the moment the abort signal fires.

            The runners' own ``CancelledError`` handlers park a synthetic
            ``aborted`` result for each call, so the batch still comes back
            fully paired — cancelling here changes WHEN the turn ends, never
            whether the wire stays legal.
            """
            nonlocal aborting, aborted_at
            assert signal is not None
            await signal.wait()
            # Stamped BEFORE the cancellations, so the instant recorded is when
            # the tools were told to stop rather than when the last of them
            # acknowledged it.
            aborted_at = time.monotonic()
            aborting = True
            for task in tasks:
                if not task.done():
                    task.cancel()
            # WAKE THE DRAIN. It is parked in ``queue.get()``, which the abort
            # does not disturb, so without this nudge it only re-evaluates
            # ``aborting`` when a runner happens to emit something — and a
            # batch whose tools are all stuck in a slow unwind emits nothing.
            # The deadline would then be armed only after the cleanup it is
            # supposed to bound had already finished. The sentinel is ignored
            # by the drain's own branches; its only job is to end the wait.
            queue.put_nowait(_TOOL_DONE)

        watcher: asyncio.Task[None] | None = None
        # Declared before the ``try`` because ``finally`` reads it: an
        # exception raised while scheduling the runners must not turn into a
        # NameError that hides the real failure.
        close_task: asyncio.Task[None] | None = None
        try:
            runnable: list[tuple[int, _PlannedCall]] = []
            for slot, item in enumerate(batch):
                if item.failure is not None or item.tool is None:
                    # Duplicate-id and resolution failures never execute: the
                    # synthetic result parks in the slot without a task, so
                    # two slots can never collide on one results entry.
                    parked = item.failure or self._synthetic_result(item.call, "Tool not found.")
                    park(slot, item, parked)
                    # SAY SO, since the end event no longer does. `park` withholds
                    # the end event for a call that never started, which is right
                    # — but the headless renderer printed `✗ <name> failed` off
                    # that event, so suppressing it alone would turn a visible
                    # diagnostic into silence and leave an operator watching a
                    # hallucinated tool name produce nothing at all (R6-3, agent
                    # review round 6). A notice is the honest carrier: it reports
                    # the failure without claiming a lifecycle that never began.
                    # The model is unaffected either way — it still gets the
                    # `tool_result` parked above.
                    #
                    # The parked result's own text is the whole message, with
                    # no tool-name prefix bolted on. Both failure kinds already
                    # name what they need to ("Tool not found: reed_file",
                    # "Duplicate call id 'c1' skipped."), so a prefix repeated
                    # the name for an unknown tool and, worse, named the tool
                    # that DID run for a duplicate id — reading as though the
                    # user's real call had been dropped (D12/D13, design round
                    # 3). The call id, not the name, is what distinguishes the
                    # twins, and it is already in the duplicate's own text.
                    reason = " ".join(
                        block.text
                        for block in parked.content
                        if isinstance(block, TextContent) and block.text
                    ).strip()
                    yield NoticeEvent(
                        text=reason or f"{item.call.name}: tool not found",
                        kind="error",
                    )
                    continue
                runnable.append((slot, item))

            pending = iter(enumerate(runnable))

            async def worker() -> None:
                # A fixed number of workers refill immediately after completion.
                # Batch waves would leave seven idle slots behind one slow read;
                # one task per queued call would instead make memory unbounded.
                # next() has no await, so each slot has exactly one owner.
                for position, (slot, item) in pending:
                    if signal is not None and signal.aborted:
                        break
                    if (
                        position >= config.max_parallel_tools
                        and config.interrupt_mode == "immediate"
                        and self._peek_steering(config)
                    ):
                        park(
                            slot,
                            item,
                            self._synthetic_result(
                                item.call,
                                SKIPPED_RESULT_TEXT,
                                details={FAULT_KEY: FAULT_SKIPPED},
                            ),
                        )
                        continue
                    if item.tool is not None and item.tool.interruptible and poll_interruptible:
                        await interruptible_runner(slot, item)
                    else:
                        await runner(slot, item)

            tasks.extend(
                asyncio.ensure_future(worker())
                for _ in range(min(config.max_parallel_tools, len(runnable)))
            )
            # Started AFTER the runner tasks exist, so it can see all of them,
            # and only when there is a signal to watch. An already-aborted
            # signal is handled by the same path: ``wait()`` returns at once.
            if signal is not None and tasks:
                watcher = asyncio.ensure_future(abort_watcher())

            # Termination is keyed on the TASKS settling, not on counting one
            # ``_TOOL_DONE`` per task, and that is what makes cancellation
            # safe. ``ensure_future`` only SCHEDULES a runner: a cancel landing
            # in the same event-loop turn (which is exactly what the abort
            # watcher does) means the body never runs, so it parks nothing and
            # a counting drain would wait forever for a receipt no one will
            # ever send. The closer posts one sentinel after every task has
            # settled; the queue is FIFO, so everything the runners emitted is
            # already ahead of it and still drains in order.
            async def closer() -> None:
                await asyncio.gather(*tasks, return_exceptions=True)
                queue.put_nowait(_BATCH_DONE)

            close_task = asyncio.ensure_future(closer()) if tasks else None
            if close_task is not None:
                # The deadline is armed by the ABORT, not at entry, and it
                # bounds THIS loop rather than the cleanup in ``finally``. The
                # first version bounded the wrong wait: the drain sat here
                # until every task had settled, so by the time ``finally`` ran
                # its ``wait_for`` there was nothing left to wait for and the
                # budget was a no-op — a tool with a six-second unwind still
                # held the turn open for six seconds. That is the very failure
                # this PR exists to remove, moved from the tool body into its
                # cleanup.
                deadline: float | None = None
                while True:
                    if aborting and deadline is None:
                        deadline = asyncio.get_running_loop().time() + ABORT_DRAIN_TIMEOUT_S
                    if deadline is None:
                        event = await queue.get()
                    else:
                        remaining = deadline - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            logger.warning(
                                "tool cleanup still running %ss after abort; "
                                "settling the turn without it",
                                ABORT_DRAIN_TIMEOUT_S,
                            )
                            break
                        try:
                            event = await _get_before_timeout(queue, remaining)
                        except TimeoutError:
                            # The tasks are left running: they own their own
                            # resources, log their own failures, and the
                            # backfill below pairs whatever they never parked.
                            logger.warning(
                                "tool cleanup still running %ss after abort; "
                                "settling the turn without it",
                                ABORT_DRAIN_TIMEOUT_S,
                            )
                            break
                    if isinstance(event, _BatchDone):
                        break
                    if isinstance(event, _ToolDone):
                        # A per-tool receipt, or the abort watcher's nudge.
                        # Neither is an event a consumer should see.
                        continue
                    yield event

            # Backfill any slot whose runner was cancelled before it could park
            # its own result. Every call MUST come back paired or the next
            # request carries a ``tool_use`` with no ``tool_result`` and the
            # provider rejects the whole conversation — so an abort that races
            # a runner's first line must not be able to break the wire.
            #
            # The backfill also EMITS the end event, which it previously did
            # not. Repairing `results_by_slot` alone keeps the wire legal but
            # tells no consumer anything: a tool whose cleanup outran
            # ``ABORT_DRAIN_TIMEOUT_S`` had its start event announced and no end
            # event ever, so the API server holds that execution record
            # IN_PROGRESS forever and never publishes a TOOL_END on its SSE
            # stream (R2, agent review round 2). That is the same consumer
            # damage `interruptible_runner`'s own cancellation handler exists to
            # prevent — it closes the fast path, and the slow-cleanup path here
            # reopened it. The TUI happens to survive either way because it
            # retires orphaned cards at the turn boundary; nothing else does.
            #
            # Yielded rather than queued: the drain loop above has already
            # broken out and nothing will read the queue again.
            #
            # DECIDED IN ONE PASS, EMITTED IN ANOTHER, and the split is load
            # bearing. A generator suspends at every `yield`, handing control to
            # a consumer that may await; a runner whose cleanup lands in one of
            # those windows calls `park()`, which writes its end event to the
            # queue nobody reads any more and fills the slot. A single
            # interleaved loop then saw the now-filled slot and emitted nothing,
            # so that call kept its start event and never got an end — the very
            # damage this backfill exists to prevent, reachable only when the
            # consumer is slow enough to suspend us (R3-1, agent review round 3).
            # Snapshotting first means the decision is made while no await can
            # intervene, so it cannot be invalidated by what happens mid-emit.
            pending_ends: list[ToolExecutionEndEvent] = []
            # HOW MANY queued end events to swallow per call id — a count, not a
            # set, because call ids are NOT unique within a batch. A model can
            # emit two calls with one id; the loop keeps the first and turns the
            # second into a planning failure, so one id can name both a slot
            # that started and one that did not. A set keyed by id cannot tell
            # them apart and suppresses BOTH, dropping the genuine end event of
            # the call that really ran (R5-1, agent review round 5). This is the
            # same collision that makes `results_by_slot` keyed by slot.
            #
            # Counting is exact even though the events are indistinguishable:
            # with N carrying one id and K owed suppression, swallowing any K
            # leaves the right number, and they are identical to a consumer.
            #
            # Not seeded from the batch: `park` no longer queues an end for a
            # call that never started, so the only entries here are the ones
            # this backfill is about to emit itself.
            claimed: Counter[str] = Counter()
            for slot, item in enumerate(batch):
                if results_by_slot[slot] is None:
                    result = self._synthetic_result(item.call, ABORTED_RESULT_TEXT)
                    results_by_slot[slot] = result
                    if slot in started_at_by_slot and item.tool is not None:
                        # Only for calls that actually STARTED. A planning
                        # failure parked its result up front and never emitted a
                        # start event, so an end event for it would be the
                        # mirror image of this bug.
                        claimed[item.call.id] += 1
                        # STAMPED like every other emitter. This was the one
                        # end event in the loop carrying no interval, because it
                        # is built here rather than inside the runner that owns
                        # `started_at` — so an aborted call reported an active
                        # span it had genuinely measured as blank, and the
                        # receipt a viewer saw depended on whether it had
                        # watched the start (review round 2, MAJOR-3). The
                        # executor owns the start/end boundary (see `park`), and
                        # that argument does not stop applying because the call
                        # ended by abort: the tool really did run for this long
                        # before it was cut off.
                        #
                        # MEASURED TO THE ABORT, NOT TO NOW, and the difference
                        # is the whole finding of review round 3 (MAJOR-4).
                        # `now` here is not a property of the tool: this code
                        # runs only after the drain loop has waited out
                        # ABORT_DRAIN_TIMEOUT_S, and a call reaches the backfill
                        # only BECAUSE its unwind outran that budget — so
                        # `now - started_at` always contains the whole drain
                        # wait. Held the tool's real work at 0.10s and swept the
                        # cleanup length, `now` reported 2.103 / 2.104 / 2.103s
                        # for cleanups of 2.5 / 4 / 8s: pinned to the budget,
                        # 21x the span, and constant regardless of the tool. It
                        # was measuring the deadline.
                        #
                        # The abort instant is the honest end because it is when
                        # the tool was cut off; what happens after is the
                        # harness waiting, not the tool working.
                        #
                        # It does NOT make this identical to `park`, and the
                        # difference is stated rather than papered over: a
                        # runner that unwinds INSIDE the budget stamps in its
                        # own `CancelledError` handler, after its cleanup has
                        # run, so it reports start->cleanup-end (measured 0.603s
                        # for the same 0.101s of work with a 0.50s cleanup).
                        # That number is defensible there because the task was
                        # genuinely unwinding for the whole of it and the
                        # executor watched it happen. It is not available here:
                        # this emitter fires precisely because the tool has NOT
                        # finished unwinding and never will be observed doing
                        # so, so the only instants in scope are the start, the
                        # abort, and the deadline. Of those the abort is the
                        # only one that is a property of the tool.
                        #
                        # `max(0.0, ...)` is a live guard, not decoration: the
                        # watcher cancels tasks after stamping, so a runner that
                        # reaches its first line in that window starts AFTER the
                        # abort and would otherwise report a negative span.
                        ended_at = aborted_at if aborted_at is not None else time.monotonic()
                        result.duration_s = max(0.0, ended_at - started_at_by_slot[slot])
                        pending_ends.append(
                            ToolExecutionEndEvent(
                                tool_call_id=item.call.id,
                                tool_name=item.tool.name,
                                result=result,
                                # Both halves, like the three sibling emitters.
                                # `duration_s` has no validator ORing it with
                                # `result.duration_s` the way `_sync_error_flag`
                                # does for the error bit, so an emitter that
                                # sets only one ships an event whose live
                                # consumer and whose replay disagree.
                                duration_s=result.duration_s,
                                is_error=result.is_error,
                            )
                        )

            # THEN drain whatever the runners queued that nobody read. A tool
            # that parked between the drain loop giving up and this point put a
            # real end event on the queue and filled its own slot, so the
            # backfill above correctly skipped it — and without this flush that
            # event is simply dropped, leaving a start with no end. That is the
            # half of R3-1 a snapshot alone does not fix: the loss happens
            # BEFORE the backfill runs, not during its emit.
            #
            # ONE BOUNDARY REMAINS, and it is accepted rather than closed: a
            # cleanup that outruns the whole TURN, not just the drain budget,
            # settles after this generator has closed. There is no longer a
            # stream to emit into, so that call keeps its start and gets no end.
            #
            # NOT confined to some extreme corner. It is intermittent wherever a
            # tool's unwind overshoots ABORT_DRAIN_TIMEOUT_S while a consumer is
            # slow enough to still owe this generator a resume — observed around
            # a 2.3s cleanup against a 0.3s-per-event consumer, and an earlier
            # comment here claiming it was unreachable below ~2.5s/~0.5s was
            # simply wrong (R9 MAJOR-2). The honest statement is: rare in
            # practice, reachable in principle, and not bounded by a threshold
            # anyone should rely on.
            #
            # Accepted anyway, because the alternative is worse: closing it
            # means holding the turn open until the cleanup finishes, which is
            # exactly what ABORT_DRAIN_TIMEOUT_S exists to refuse — the user
            # pressed Esc and is owed their prompt back, which is this whole
            # change's purpose. A consumer holding execution records by id must
            # therefore reconcile them at the TURN boundary rather than trusting
            # every start to be followed by an end; the TUI already does exactly
            # that when it retires orphaned cards.
            #
            # `claimed` is DEFENCE IN DEPTH, not a live guard. It was
            # load-bearing when the two halves could collide; the source-side
            # withholding in `park` removes that collision, and `park` is
            # synchronous, so a slot is filled and its end queued atomically
            # with respect to the event loop and no `await` separates this
            # backfill from the flush below. There is therefore no interleaving
            # in which the flush meets an end for a slot the backfill claimed —
            # measured: 0 suppressions across the whole harness suite (R6-1,
            # agent review round 6). It stays as a belt, and it counts rather
            # than matching because ids collide within a batch, so a future
            # change that reintroduces an interleaving cannot resurrect R5-1 by
            # suppressing a started call's end along with its twin's.
            while True:
                try:
                    queued = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if isinstance(queued, (_ToolDone, _BatchDone)):
                    continue
                if isinstance(queued, ToolExecutionEndEvent) and _consume_claim(
                    claimed, queued.tool_call_id
                ):
                    continue
                yield queued

            results.extend(result for result in results_by_slot if result is not None)
            for end_event in pending_ends:
                yield end_event
        finally:
            if watcher is not None and not watcher.done():
                watcher.cancel()
            # GeneratorExit / abort: never leave runner tasks behind.
            for task in tasks:
                if not task.done():
                    task.cancel()
            if aborting:
                # Deliberately NOT awaited. The drain above already gave the
                # cleanup its budget, and the whole point of that deadline is
                # that the turn ends on a human timescale; awaiting here would
                # hand the time straight back. The tasks are cancelled, own
                # their own resources (bash kills its process group, eval tears
                # down its worker) and cannot write to this batch any more —
                # every slot is paired by the backfill above. ``gather``
                # retrieves their exceptions so a raising cleanup cannot
                # surface as an unobserved-task warning.
                if tasks:
                    detached = asyncio.gather(*tasks, return_exceptions=True)
                    detached.add_done_callback(lambda task: task.exception())
                if close_task is not None and not close_task.done():
                    close_task.cancel()
            else:
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                if close_task is not None:
                    if not close_task.done():
                        close_task.cancel()
                    with contextlib.suppress(BaseException):
                        await close_task

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _approval_summary(
        tool: AgentTool, call: ToolCall, cwd: str, context: ToolContext | None = None
    ) -> str:
        """The sentence the approval prompt shows for ``call``.

        The tool's own ``describe_approval`` when it has one, because only the
        tool knows which argument IS the decision — `bash`'s command, `write`'s
        resolved path, `browser`'s URL. The JSON fallback is for third-party and
        MCP tools the harness cannot introspect; it is honest but unranked, so a
        narrow terminal shows whichever field the serialiser happened to put
        first rather than the one that matters.
        """
        describe = tool.describe_approval
        if describe is not None:
            try:
                described = _describe_call(describe, call.arguments, cwd, context)
            except Exception:
                # A description is never worth failing a call over: fall through
                # to the dump, which is always renderable.
                logger.warning("approval description failed for %s", call.name, exc_info=True)
            else:
                # `isinstance`, not truthiness: a describer that returns a dict or
                # a Path is a bug in that tool, and letting it through raised deep
                # in the renderer where the failure reads as "approval denied".
                if isinstance(described, str) and described.strip():
                    return sanitize_prompt_line(described)
        return sanitize_prompt_line(
            f"{call.name}({call.raw_arguments or json.dumps(call.arguments)})"
        )

    @staticmethod
    def _classify_fault(result: ToolResult) -> str:
        """The fault class for a finished call: ``""`` when it ran cleanly.

        Reads the marker the SOURCE set (see the ``FAULT_*`` constants), and
        falls back to ``execution`` for any other error — a tool that ran and
        returned ``is_error``, which is by definition not a planning failure and
        so not the model's fault. Deliberately NOT a text match on the result:
        the reason is known where the result is built and nowhere else, and a
        reworded message must not silently reclassify a model fault as an
        execution error (or the reverse, which would inflate the benchmark).

        A TOOL BODY is one such source. The fallback is right only for a call
        whose arguments were usable, so a tool that rejects a malformed
        argument its JSON-Schema type was too coarse to catch must say so —
        by raising ``InvalidToolArgumentsError`` or returning a result already
        carrying the marker. Without that, an argument-shape rejection is
        indistinguishable here from a genuine execution failure and is
        silently laundered into the execution bucket.
        """
        if not result.is_error:
            return ""
        details = result.details or {}
        marker = details.get(FAULT_KEY)
        return str(marker) if isinstance(marker, str) and marker else FAULT_EXECUTION

    def _report_tool_call(
        self,
        config: LoopConfig,
        context: LoopContext,
        item: "_PlannedCall",
        result: ToolResult,
        *,
        origin: str,
    ) -> None:
        """Hand one finished call's outcome to the host. Never raises, never blocks.

        Called from ``park`` and from the nested ``dispatch_tool`` bridge, both
        of which are on the EVENT LOOP inside a live turn. The whole body is
        inside one guard because the callback is host-supplied: the harness's
        contract is that a measurement can never break the thing it measures,
        and that has to hold even against a host hook that throws.

        The callback is invoked SYNCHRONOUSLY and is deliberately NOT wrapped in
        a timeout or handed to a thread: at ~0.0018 ms for the shipped hook,
        scheduling would cost more than the work and would reorder samples. The
        consequence is that the non-blocking half of the contract is the host's
        to keep and cannot be enforced here — a hook that blocks adds its full
        duration to the turn (measured 0.002 s → 0.754 s against a 0.75 s
        sleep). ``LoopConfig.record_tool_call`` states the budget.

        The tool name is taken from the resolved tool when there is one and from
        the CALL otherwise — a hallucinated name has no tool, and that name is
        exactly what a "which tool does this model get wrong" view needs.
        """
        callback = config.record_tool_call
        if callback is None:
            return
        try:
            session_id = ""
            tool_context = context.tool_context
            if tool_context is not None:
                session_id = getattr(tool_context, "session_id", "") or ""
            if not session_id:
                return
            name = item.tool.name if item.tool is not None else item.call.name
            duration = result.duration_s
            callback(
                name,
                origin,
                self._classify_fault(result),
                -1.0 if duration is None else float(duration) * 1000.0,
            )
        except Exception:  # noqa: BLE001 — analytics must never raise into a turn
            logger.debug("tool-call analytics hook failed", exc_info=True)

    @staticmethod
    def _synthetic_result(
        call: ToolCall, text: str, details: Mapping[str, Any] | None = None
    ) -> ToolResult:
        """A result the loop invented because the call never ran.

        ``details`` carries an extra machine-readable marker for the cases a host
        must tell apart. ``__synthetic`` alone cannot: "the user said no" and
        "the approval gate crashed" are the same shape and opposite meanings.
        """
        return ToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            is_error=True,
            content=[TextContent(text=text)],
            # `__synthetic` LAST: it is the invariant this factory exists to
            # assert, so a caller's extra markers cannot displace it.
            details={**(details or {}), "__synthetic": True},
        )

    def _append_results(
        self,
        context: LoopContext,
        results: list[ToolResult],
        new_messages: list[AgentMessage],
        redact: Callable[[str], str] | None = None,
    ) -> None:
        for result in results:
            content: list[Content] = list(result.content)
            # Redact BEFORE the empty-result backfill: a redacted secret is
            # still text, so an image-only result is untouched and a genuinely
            # empty one still gets the placeholder it needs to serialize.
            if redact is not None:
                content = [
                    TextContent(text=redact(item.text)) if isinstance(item, TextContent) else item
                    for item in content
                ]
            # coerceToolResult: an empty tool result serializes as "" on
            # most wires and Anthropic REJECTS an empty ``is_error`` content
            # with a 400 — backfill one placeholder block. Image-only results
            # keep their blocks untouched (never text-flatten).
            if not content:
                content = [TextContent(text=EMPTY_TOOL_RESULT_TEXT)]
            # The bookkeeping stamp lives in ``Message.tool_result`` rather
            # than here. Stamping it at this call site is what let the OTHER
            # callers of that constructor lose it: the viewer's live row
            # (``AttachedSession._remember_live``) went through the same
            # constructor and silently carried no duration, so a resumed card
            # painted a blank column. One definition, so every producer of a
            # tool row agrees by construction.
            #
            # ``content`` rides in on a copy of the result because the
            # redaction and empty-result backfill above have already rewritten
            # it; the ToolResult itself must not be mutated.
            message = Message.tool_result(result.model_copy(update={"content": content}))
            context.messages.append(message)
            new_messages.append(message)

    @staticmethod
    def _drain_pending(pending: list[AgentMessage], context: LoopContext) -> int:
        for message in pending:
            context.messages.append(message)
            if isinstance(message, CustomMessage) and message.on_commit is not None:
                try:
                    message.on_commit()
                except Exception:
                    logger.warning("aside on_commit failed", exc_info=True)
        return len(pending)

    @staticmethod
    def _discard_pending_custom(pending: list[AgentMessage]) -> None:
        for message in pending:
            if isinstance(message, CustomMessage) and message.on_discard is not None:
                try:
                    message.on_discard()
                except Exception:
                    logger.warning("aside on_discard failed", exc_info=True)

    @staticmethod
    async def _collect_inflight_injections(config: LoopConfig) -> list[AgentMessage]:
        """Steering (consuming) + asides after each tool batch."""
        pending: list[AgentMessage] = []
        if config.get_steering_messages is not None:
            pending.extend(await config.get_steering_messages())
        if config.get_aside_messages is not None:
            pending.extend(_materialize_asides(await config.get_aside_messages()))
        return pending

    @staticmethod
    async def _collect_yield_injections(config: LoopConfig) -> list[AgentMessage]:
        """Steering + asides + follow-ups at the yield boundary."""
        pending: list[AgentMessage] = []
        if config.get_steering_messages is not None:
            pending.extend(await config.get_steering_messages())
        if config.get_aside_messages is not None:
            pending.extend(_materialize_asides(await config.get_aside_messages()))
        if config.get_follow_up_messages is not None:
            pending.extend(await config.get_follow_up_messages())
        return pending

    @staticmethod
    def _peek_steering(config: LoopConfig) -> bool:
        """Whether queued STEERING may cancel a running tool.

        Steering alone, deliberately narrow: this is the predicate the
        batch-skip branch reads, and skipping a batch's remaining calls is only
        ever right when the USER redirected the work.
        """
        peek = (
            config.has_urgent_steering_messages
            if config.has_urgent_steering_messages is not None
            else config.has_steering_messages
        )
        if peek is None:
            return False
        try:
            return bool(peek())
        except Exception:
            return False

    @staticmethod
    def _peek_interrupt(config: LoopConfig) -> bool:
        """Whether ANYTHING wants the running tool to stop now.

        Two reasons today: queued steering, and a pending fork waiting for a
        safe boundary. Spelled as one concept because the tool-cancellation
        sites genuinely ask that one question — a reader who found the fork
        check bolted onto something named ``_peek_steering`` would reasonably
        conclude a fork IS a steer, which is the misreading that leads to a fork
        injecting a user turn into the parent.

        The distinction from :meth:`_peek_steering` is not cosmetic: this one
        may interrupt a tool, and ONLY :meth:`_peek_steering` may skip a batch's
        remaining calls. See ``LoopConfig.has_pending_fork``.
        """
        if AgentLoop._peek_steering(config):
            return True
        fork = config.has_pending_fork
        if fork is None:
            return False
        try:
            return bool(fork())
        except Exception:
            # Same posture as the steering peek: a host callback that raises
            # must cost an interrupt, never the running turn.
            return False

    # -- deadline wiring ----------------------------------------------------

    def _wire_deadline(
        self, config: LoopConfig, signal: AbortSignal | None
    ) -> tuple[AbortSignal | None, asyncio.Task[None] | None]:
        """Arm a timeout task when ``config.deadline`` is set; return the
        (possibly combined) signal and the task to cancel on exit."""
        if config.deadline is None:
            return signal, None
        deadline_signal = AbortSignal()
        delay_s = max(0.0, (config.deadline - time.time() * 1000.0) / 1000.0)

        async def _trip() -> None:
            await asyncio.sleep(delay_s)
            deadline_signal.abort("deadline exceeded")

        task = asyncio.ensure_future(_trip())
        combined = (
            AbortSignal.any_of(signal, deadline_signal) if signal is not None else deadline_signal
        )
        return combined, task

    @staticmethod
    def _unwire_deadline(task: asyncio.Task[None] | None) -> None:
        if task is not None and not task.done():
            task.cancel()


def _materialize_asides(asides: Sequence[Aside]) -> list[AgentMessage]:
    """Invoke aside thunks at injection time and keep the live messages.

    A ``None`` result is dropped silently; a :class:`StaleAside` result is
    dropped too, but its originating :class:`CustomMessage` gets its
    ``on_discard`` hook fired here. ``on_commit`` is NOT fired here — it
    fires in :meth:`AgentLoop._drain_pending` when the message actually
    enters context, so an aborted run never commits pending asides.
    """
    out: list[AgentMessage] = []
    for item in asides:
        message = item() if callable(item) else item
        if message is None:
            continue
        if isinstance(message, StaleAside):
            if message.message.on_discard is not None:
                try:
                    message.message.on_discard()
                except Exception:
                    logger.warning("aside on_discard failed", exc_info=True)
            continue
        out.append(message)
    return out


def validate_tool_arguments(
    tool: AgentTool, arguments: dict[str, Any], raw_arguments: str | None = None
) -> list[str]:
    """Validate ``arguments`` against the tool's JSON-schema ``parameters``.

    Returns a list of human-readable errors (empty = valid); the loop turns
    those into ``is_error`` results back to the model rather than raising.
    Scalar type checks run through pydantic :class:`TypeAdapter`s.
    """
    if raw_arguments:
        try:
            parsed = json.loads(raw_arguments)
            if not isinstance(parsed, dict):
                return ["arguments must be a JSON object"]
        except json.JSONDecodeError as exc:
            return [f"arguments are not valid JSON: {exc}"]

    schema = tool.parameters or {}
    if not schema:
        return []
    errors: list[str] = []
    properties = schema.get("properties", {}) or {}
    for name in schema.get("required", []) or []:
        if name not in arguments:
            errors.append(f"missing required argument '{name}'")
    for name, value in arguments.items():
        prop_schema = properties.get(name)
        if not isinstance(prop_schema, dict):
            continue
        expected = prop_schema.get("type")
        if expected is None:
            continue
        types = expected if isinstance(expected, list) else [expected]
        if "null" in types and value is None:
            continue
        adapters = [_TYPE_ADAPTERS[t] for t in types if t in _TYPE_ADAPTERS]
        if not adapters:
            continue
        for adapter in adapters:
            try:
                adapter.validate_python(value)
                break
            except ValidationError:
                continue
        else:
            errors.append(f"argument '{name}' does not match type {' | '.join(types)}")
    return errors


# ---------------------------------------------------------------------------
# Internal carriers
# ---------------------------------------------------------------------------
@dataclass
class _ModelTurnResult:
    """Internal carrier so ``_model_turn`` can return its assembled message
    through the event stream without a second channel."""

    message: Message
    stop_reason: str
    error: str | None = None
    #: The spec the provider call was actually built with, resolved per call.
    #: Carried out so the run loop can act on what it SENT rather than on the
    #: run's snapshot -- the two differ whenever a host resolver is in play, and
    #: a recovery gated on the wrong one of them is a recovery that does not
    #: fire (see the reasoning-echo fill in ``run``).
    model: "ModelSpec | None" = None
    #: The stream died because the MACHINE was offline, not because the provider
    #: answered badly. Carried out to the run loop, which continues such a turn
    #: instead of ending the run on it — see ``MAX_CONNECTIVITY_CONTINUATIONS``.
    connectivity_loss: bool = False
