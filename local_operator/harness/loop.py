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
import inspect
import json
import logging
import time
from collections import Counter
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

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
    AbortSignal,
    AgentEndEvent,
    AgentEvent,
    AgentMessage,
    AgentStartEvent,
    AgentTool,
    AgentToolUpdate,
    Aside,
    ChatRequest,
    Content,
    CustomMessage,
    LoopConfig,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ModelSpec,
    NoticeEvent,
    RenderedStreamError,
    StaleAside,
    StreamEndEvent,
    StreamEvent,
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
# Backfill for empty tool results (coerceToolResult): Anthropic rejects an
# empty ``is_error`` tool_result content with a 400, and other providers
# serialize "" — one placeholder block keeps the wire legal for every client.
EMPTY_TOOL_RESULT_TEXT = "[tool returned no output]"
# Steering-interrupt poll interval for ``interruptible`` tools mid-run.
STEERING_INTERRUPT_POLL_S = 0.25
# How long the batch waits for tools to unwind after an ABORT before it stops
# waiting and settles the turn anyway. An abort is a user pressing Esc, so the
# turn must end on a human timescale; a tool whose cleanup is slower than this
# (a process group refusing to die) keeps unwinding in the background while the
# turn it belonged to is already over.
ABORT_DRAIN_TIMEOUT_S = 2.0


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


def _batches_shared(item: _PlannedCall) -> bool:
    """Whether this call may run alongside its neighbours in one batch.

    An unresolved or pre-failed call counts as shared: it never executes, so
    it cannot conflict with anything. Only a resolved ``exclusive`` tool
    forces a batch of one.
    """
    return item.tool is None or item.tool.concurrency == "shared"


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
                    async for event in self._model_turn(context, config, signal):
                        if isinstance(event, _ModelTurnResult):
                            assistant, stop_reason, stream_error = (
                                event.message,
                                event.stop_reason,
                                event.error,
                            )
                        else:
                            yield event
                    if assistant is None:
                        raise RuntimeError("model turn produced no assistant message")
                    context.messages.append(assistant)
                    new_messages.append(assistant)

                    if stop_reason in ("error", "aborted"):
                        # Pair every dangling tool call so the wire stays legal.
                        placeholders = [
                            self._synthetic_result(call, ABORTED_RESULT_TEXT)
                            for call in assistant.tool_calls
                        ]
                        self._append_results(context, placeholders, new_messages)
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
                        # Truncated: pair placeholders, do NOT execute.
                        placeholders = [
                            self._synthetic_result(call, ABORTED_RESULT_TEXT)
                            for call in assistant.tool_calls
                        ]
                        self._append_results(context, placeholders, new_messages)
                    elif assistant.tool_calls:
                        async for event in self._execute_tool_calls(
                            assistant.tool_calls, context, config, signal, tool_results
                        ):
                            yield event
                        self._append_results(context, tool_results, new_messages)

                    if signal is not None and signal.aborted:
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
    # (``_abortable_stream`` is a module-level helper; see below the class.)

    async def _model_turn(
        self,
        context: LoopContext,
        config: LoopConfig,
        signal: AbortSignal | None,
    ) -> AsyncIterator[AgentEvent | _ModelTurnResult]:
        """One provider call: build the request, stream it, assemble the
        assistant message, emitting message_start/update/end events.

        The model is resolved HERE, per call, not once per run — see
        :meth:`_current_model`.
        """
        shaped = list(context.messages)
        if config.transform_context is not None:
            outcome = config.transform_context(shaped)
            if inspect.isawaitable(outcome):
                outcome = await outcome
            shaped = list(outcome)
        converted = config.convert_to_llm(shaped)
        if inspect.isawaitable(converted):
            converted = await converted
        request = ChatRequest(
            # Resolved after `transform_context`/`convert_to_llm`, which can
            # await: the spec is read as late as possible so a switch made
            # while this call was being prepared still catches it.
            model=self._current_model(config),
            system_blocks=list(context.system_blocks),
            messages=list(converted),
            tools=list(context.tools),
        )

        assistant = Message(role="assistant")
        text_parts: list[str] = []
        tool_states: dict[int, dict[str, Any]] = {}
        usage: Usage | None = None
        provider_payload: dict[str, Any] | None = None
        stop_reason = "stop"
        error: str | None = None

        yield TurnStartEvent()
        yield MessageStartEvent(message=assistant)

        try:
            stream = _abortable_stream(config.stream_fn(request, signal), signal)
            async for event in stream:
                if isinstance(event, StreamTextDelta):
                    text_parts.append(event.delta)
                    assistant.content = [TextContent(text="".join(text_parts))]
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
                            "reported": -1,
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
                        now = time.monotonic()
                        first = state["announced"] == 0.0
                        if first or now - state["announced"] >= COMPOSE_NOTICE_INTERVAL_S:
                            state["announced"] = now
                            state["reported"] = state["bytes"]
                            yield ToolCallComposeEvent(
                                tool_call_id=state["key"],
                                tool_name=state["name"],
                                argument_bytes=state["bytes"],
                                intent=state["intent"],
                            )
                elif isinstance(event, StreamUsageEvent):
                    usage = event.usage
                elif isinstance(event, StreamEndEvent):
                    # Flush what the throttle swallowed. Arguments commonly land
                    # in one burst inside a single window, so without this the
                    # row's size could report a fraction of the call — or, when
                    # the whole payload arrives faster than one window, never
                    # display a size at all. It matters most on an aborted turn,
                    # where the frozen row is what the user is left reading.
                    for state in tool_states.values():
                        if state["name"] and state["bytes"] != state["reported"]:
                            state["reported"] = state["bytes"]
                            yield ToolCallComposeEvent(
                                tool_call_id=state["key"] or "compose:0",
                                tool_name=state["name"],
                                argument_bytes=state["bytes"],
                                intent=state["intent"],
                            )
                    stop_reason = event.stop_reason
                    if event.usage is not None:
                        usage = event.usage
                    provider_payload = event.provider_payload
                    error = event.error
        except asyncio.CancelledError:
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

        if signal is not None and signal.aborted:
            # A stream CUT by the abort ends without its ``StreamEndEvent``, so
            # the local default ("stop") would otherwise stand and the turn
            # would read as a clean finish — the loop would go on to make
            # another model call for a turn the user has already stopped.
            # Recording the truth here is what lets the single ``("error",
            # "aborted")`` branch upstream pair the dangling calls and end the
            # run, instead of the abort having to be re-detected in each place.
            stop_reason = "aborted"

        assistant.tool_calls = [
            self._assemble_tool_call(state) for _, state in sorted(tool_states.items())
        ]
        assistant.stop_reason = stop_reason
        assistant.usage = usage
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
        yield _ModelTurnResult(message=assistant, stop_reason=stop_reason, error=error)

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
                            call, f"Duplicate call id '{call.id}' skipped."
                        ),
                    )
                )
                continue
            seen_ids.add(call.id)
            plan.append(await self._plan_call(call, context, config))
        index = 0
        first_slot = True
        while index < len(plan):
            if (
                not first_slot
                and config.interrupt_mode == "immediate"
                and self._peek_steering(config)
            ):
                for remaining in plan[index:]:
                    results.append(self._synthetic_result(remaining.call, SKIPPED_RESULT_TEXT))
                break

            if not _batches_shared(plan[index]):
                async for event in self._execute_batch(
                    plan[index : index + 1], context, config, signal, results
                ):
                    yield event
                index += 1
            else:
                end = index
                while end < len(plan) and _batches_shared(plan[end]):
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
                failure=self._synthetic_result(call, f"Tool not found: {call.name}"),
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
                failure=self._synthetic_result(call, "Invalid arguments: " + "; ".join(errors)),
            )
        return _PlannedCall(call=call, tool=tool, args=args, intent=intent)

    async def _runner_result(
        self,
        item: _PlannedCall,
        context: LoopContext,
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
        if (
            tool.approval_tier in ("write", "exec")
            and tool_context is not None
            and tool_context.request_approval is not None
        ):
            summary = self._approval_summary(tool, call, tool_context.cwd)
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
                    # `_first_line(result_text)`), which is the row the owner
                    # read as `User denied approv…`. It therefore carries the
                    # whole diagnosis on its own and the detail follows below,
                    # where the expansion and the model both get it.
                    f"Approval gate failed for '{safe_name}' — the call was not run.\n"
                    f"{sanitize_prompt_line(named, limit=200)}\n"
                    "This is a harness fault, not a refusal by the user; the stack is in "
                    "the log.",
                    details={"__approval_gate_failed": True},
                )
            if not approved:
                return self._synthetic_result(
                    call, f"User denied approval for '{sanitize_prompt_line(call.name, 120)}'."
                )

        def on_update(update: AgentToolUpdate) -> None:
            queue.put_nowait(
                ToolExecutionUpdateEvent(
                    tool_call_id=call.id, tool_name=tool.name, partial_result=update
                )
            )

        try:
            return await tool.execute(
                call.id, item.args, signal, on_update, context.tool_context or ToolContext()
            )
        except asyncio.CancelledError:
            raise
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
        poll_interruptible = (
            config.interrupt_mode == "immediate" and config.has_steering_messages is not None
        )
        # Set by the abort watcher so the runners' cancellation handlers can
        # tell an abort apart from a steering interrupt and label their
        # synthetic results correctly.
        aborting = False

        def park(slot: int, item: _PlannedCall, result: ToolResult) -> None:
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
            started = item.failure is None and item.tool is not None
            if started:
                queue.put_nowait(
                    ToolExecutionEndEvent(
                        tool_call_id=item.call.id,
                        tool_name=item.tool.name if item.tool is not None else item.call.name,
                        result=result,
                        is_error=result.is_error,
                    )
                )
            queue.put_nowait(_TOOL_DONE)

        async def runner(slot: int, item: _PlannedCall) -> None:
            tool_name = item.tool.name if item.tool is not None else item.call.name
            await queue.put(
                # `item.args`, not `item.call.arguments`: the event must show
                # what the tool is actually being run with, and those two now
                # differ by the lifted `i`. Leaking it here would caption the
                # tool row with the intent — the TUI's argument summary scans
                # values for a row identity — reinstating on the card the
                # duplication that splitting fact from claim removes.
                ToolExecutionStartEvent(
                    tool_call_id=item.call.id,
                    tool_name=tool_name,
                    args=item.args,
                    intent=item.intent,
                )
            )
            try:
                result = await self._runner_result(item, context, signal, queue)
            except asyncio.CancelledError:
                # Cancelled (abort/GeneratorExit): pair the call with a
                # synthetic aborted result so tool_use/tool_result pairing
                # stays legal, then propagate the cancellation.
                park(slot, item, self._synthetic_result(item.call, ABORTED_RESULT_TEXT))
                raise
            park(slot, item, result)

        async def interruptible_runner(slot: int, item: _PlannedCall) -> None:
            tool_name = item.tool.name if item.tool is not None else item.call.name
            await queue.put(
                ToolExecutionStartEvent(
                    tool_call_id=item.call.id,
                    tool_name=tool_name,
                    args=item.args,
                    intent=item.intent,
                )
            )
            tool_task = asyncio.ensure_future(self._runner_result(item, context, signal, queue))
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
                        if self._peek_steering(config):
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
                    text = (
                        SKIPPED_RESULT_TEXT
                        if not (signal is not None and signal.aborted)
                        else ABORTED_RESULT_TEXT
                    )
                    result = self._synthetic_result(item.call, text)
                park(slot, item, result)
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
                park(slot, item, self._synthetic_result(item.call, ABORTED_RESULT_TEXT))
                raise

        async def abort_watcher() -> None:
            """Cancel every runner the moment the abort signal fires.

            The runners' own ``CancelledError`` handlers park a synthetic
            ``aborted`` result for each call, so the batch still comes back
            fully paired — cancelling here changes WHEN the turn ends, never
            whether the wire stays legal.
            """
            nonlocal aborting
            assert signal is not None
            await signal.wait()
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
                interruptible = item.tool.interruptible
                tasks.append(
                    asyncio.ensure_future(
                        interruptible_runner(slot, item)
                        if interruptible and poll_interruptible
                        else runner(slot, item)
                    )
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
                        getter = asyncio.ensure_future(queue.get())
                        try:
                            event = await asyncio.wait_for(getter, timeout=remaining)
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
                    if item.failure is None and item.tool is not None:
                        # Only for calls that actually STARTED. A planning
                        # failure parked its result up front and never emitted a
                        # start event, so an end event for it would be the
                        # mirror image of this bug.
                        claimed[item.call.id] += 1
                        pending_ends.append(
                            ToolExecutionEndEvent(
                                tool_call_id=item.call.id,
                                tool_name=item.tool.name,
                                result=result,
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
    def _approval_summary(tool: AgentTool, call: ToolCall, cwd: str) -> str:
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
                described = describe(call.arguments, cwd)
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

    @staticmethod
    def _append_results(
        context: LoopContext, results: list[ToolResult], new_messages: list[AgentMessage]
    ) -> None:
        for result in results:
            content: list[Content] = list(result.content)
            # coerceToolResult: an empty tool result serializes as "" on
            # most wires and Anthropic REJECTS an empty ``is_error`` content
            # with a 400 — backfill one placeholder block. Image-only results
            # keep their blocks untouched (never text-flatten).
            if not content:
                content = [TextContent(text=EMPTY_TOOL_RESULT_TEXT)]
            message = Message(
                role="tool",
                content=content,
                tool_call_id=result.tool_call_id,
                tool_name=result.tool_name,
                is_error=result.is_error,
            )
            # Compaction reads tool details (paths, useless flag) from here and
            # writes {"pruned": True, ...} back; wire clients ignore this key.
            if result.details is not None or result.useless:
                message.provider_payload = {"details": result.details, "useless": result.useless}
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
        if config.has_steering_messages is None:
            return False
        try:
            return bool(config.has_steering_messages())
        except Exception:
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
