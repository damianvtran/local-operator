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
import inspect
import json
import logging
import time
from collections.abc import AsyncIterator, Sequence
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
    NoticeEvent,
    RenderedStreamError,
    StaleAside,
    StreamEndEvent,
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


_TOOL_DONE = _ToolDone()

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

                    yield TurnEndEvent(message=assistant, tool_results=tool_results)
                    turn_end = config.on_turn_end
                    if turn_end is not None:
                        outcome = turn_end(list(context.messages))
                        if inspect.isawaitable(outcome):
                            await outcome

                    has_more_tool_calls = bool(assistant.tool_calls)
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

    async def _model_turn(
        self,
        context: LoopContext,
        config: LoopConfig,
        signal: AbortSignal | None,
    ) -> AsyncIterator[AgentEvent | _ModelTurnResult]:
        """One provider call: build the request, stream it, assemble the
        assistant message, emitting message_start/update/end events."""
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
            model=config.model,
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
            stream = config.stream_fn(request, signal)
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
        queue: asyncio.Queue[AgentEvent | _ToolDone],
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
            except Exception:
                logger.warning("approval callback raised for %s", call.name, exc_info=True)
                approved = False
            if not approved:
                return self._synthetic_result(call, f"User denied approval for '{call.name}'.")

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

        Results are keyed by BATCH SLOT, not call id: a model can emit two
        calls with the same id in one batch, and keying by id made the two
        slots collide into one result (duplicate tool_result ids on the wire,
        which Anthropic rejects). A slot whose call failed planning never
        runs; its synthetic result is parked in its slot up front.
        """
        queue: asyncio.Queue[AgentEvent | _ToolDone] = asyncio.Queue()
        results_by_slot: list[ToolResult | None] = [None] * len(batch)
        tasks: list[asyncio.Task[None]] = []
        poll_interruptible = (
            config.interrupt_mode == "immediate" and config.has_steering_messages is not None
        )

        def park(slot: int, item: _PlannedCall, result: ToolResult) -> None:
            results_by_slot[slot] = result
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
                # Cancelled for steering (or by the run aborting): synthesize a
                # skipped/aborted result so the call stays paired.
                text = (
                    SKIPPED_RESULT_TEXT
                    if not (signal is not None and signal.aborted)
                    else ABORTED_RESULT_TEXT
                )
                result = self._synthetic_result(item.call, text)
            park(slot, item, result)

        try:
            for slot, item in enumerate(batch):
                if item.failure is not None or item.tool is None:
                    # Duplicate-id and resolution failures never execute: the
                    # synthetic result parks in the slot without a task, so
                    # two slots can never collide on one results entry.
                    park(
                        slot,
                        item,
                        item.failure or self._synthetic_result(item.call, "Tool not found."),
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
            finished = 0
            while finished < len(tasks):
                item = await queue.get()
                if isinstance(item, _ToolDone):
                    finished += 1
                    continue
                yield item
            await asyncio.gather(*tasks, return_exceptions=True)
            results.extend(result for result in results_by_slot if result is not None)
        finally:
            # GeneratorExit / abort: never leave runner tasks behind.
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

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
    def _synthetic_result(call: ToolCall, text: str) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            is_error=True,
            content=[TextContent(text=text)],
            details={"__synthetic": True},
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
