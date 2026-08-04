"""The agent loop — a provider-agnostic engine with native tool calling.

Port of omp ``packages/agent/src/agent-loop.ts`` (the ``runLoopBody`` core).
The loop knows NOTHING about sessions, persistence, or UI: everything host-side
arrives through :class:`~local_operator.harness.types.LoopConfig` callbacks,
and the only boundary outward is the :class:`AgentEvent` stream.

Structure is omp's two nested while loops:

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

from local_operator.harness.types import (
    AbortSignal,
    AgentEndEvent,
    AgentEvent,
    AgentMessage,
    AgentStartEvent,
    AgentTool,
    AgentToolUpdate,
    ChatRequest,
    CustomMessage,
    LoopConfig,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    StreamUsageEvent,
    TextContent,
    ToolCall,
    ToolContext,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolResult,
    TurnEndEvent,
    TurnStartEvent,
    Usage,
)

logger = logging.getLogger(__name__)

# Sentinel pushed through the tool-event queue when one execution finishes.
_TOOL_DONE = object()

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
    """One resolved tool call: either ready to run or pre-failed."""

    call: ToolCall
    tool: AgentTool | None = None
    args: dict[str, Any] = field(default_factory=dict)
    failure: ToolResult | None = None  # resolution/validation/approval failure


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

        yield AgentStartEvent(generation=generation)

        try:
            while True:
                # ---- inner loop: model + tools until quiescent -----------
                while has_more_tool_calls or pending:
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
                            assistant, stop_reason, stream_error = event.message, event.stop_reason, event.error
                        else:
                            yield event
                    assert assistant is not None
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
                    pending = late
                    has_more_tool_calls = True
                    continue
                break

            yield AgentEndEvent(messages=new_messages, generation=generation)
        finally:
            self._unwire_deadline(deadline_task)

    # ------------------------------------------------------------------
    # Model streaming
    # ------------------------------------------------------------------

    async def _model_turn(
        self,
        context: LoopContext,
        config: LoopConfig,
        signal: AbortSignal | None,
    ) -> AsyncIterator[AgentEvent | "_ModelTurnResult"]:
        """One provider call: build the request, stream it, assemble the
        assistant message, emitting message_start/update/end events."""
        converted = config.convert_to_llm(list(context.messages))
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
                        event.index, {"id": "", "name": "", "arg_parts": []}
                    )
                    if event.id:
                        state["id"] = event.id
                    if event.name:
                        state["name"] += event.name
                    if event.argument_delta:
                        state["arg_parts"].append(event.argument_delta)
                elif isinstance(event, StreamUsageEvent):
                    usage = event.usage
                elif isinstance(event, StreamEndEvent):
                    stop_reason = event.stop_reason
                    if event.usage is not None:
                        usage = event.usage
                    provider_payload = event.provider_payload
                    error = event.error
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("model stream failed", exc_info=True)
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
        """Resolve, validate, approve and schedule one batch of calls.

        ``shared`` tools run in parallel (asyncio.gather), ``exclusive`` tools
        run alone. When ``interrupt_mode == "immediate"`` and steering is
        queued, remaining calls are skipped with synthetic results.
        """
        plan = [await self._plan_call(call, context, config) for call in calls]
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

            if plan[index].tool is not None and plan[index].tool.concurrency == "exclusive":
                async for event in self._execute_batch(plan[index : index + 1], context, signal, results):
                    yield event
                index += 1
            else:
                end = index
                while (
                    end < len(plan)
                    and (plan[end].tool is None or plan[end].tool.concurrency == "shared")
                ):
                    end += 1
                async for event in self._execute_batch(plan[index:end], context, signal, results):
                    yield event
                index = end
            first_slot = False

    async def _plan_call(self, call: ToolCall, context: LoopContext, config: LoopConfig) -> _PlannedCall:
        tool = next((t for t in context.tools if t.name == call.name), None)
        if tool is None and config.resolve_fallback_tool is not None:
            tool = config.resolve_fallback_tool(call.name)
        if tool is None:
            return _PlannedCall(
                call=call,
                failure=self._synthetic_result(call, f"Tool not found: {call.name}"),
            )

        errors = validate_tool_arguments(tool, call.arguments, call.raw_arguments)
        if errors:
            return _PlannedCall(
                call=call,
                tool=tool,
                failure=self._synthetic_result(
                    call, "Invalid arguments: " + "; ".join(errors)
                ),
            )

        tool_context = context.tool_context
        if (
            tool.approval_tier in ("write", "exec")
            and tool_context is not None
            and tool_context.request_approval is not None
        ):
            summary = f"{call.name}({call.raw_arguments or json.dumps(call.arguments)})"
            approved = await tool_context.request_approval(call.name, summary[:500])
            if not approved:
                return _PlannedCall(
                    call=call,
                    tool=tool,
                    failure=self._synthetic_result(call, f"User denied approval for '{call.name}'."),
                )

        return _PlannedCall(call=call, tool=tool, args=dict(call.arguments))


    async def _execute_batch(
        self,
        batch: list[_PlannedCall],
        context: LoopContext,
        signal: AbortSignal | None,
        results: list[ToolResult],
    ) -> AsyncIterator[AgentEvent]:
        """Run one concurrency batch, streaming start/update/end events out as
        the tools produce them (order of completion, per-slot results kept)."""
        queue: asyncio.Queue[Any] = asyncio.Queue()
        results_by_id: dict[str, ToolResult] = {}

        async def runner(item: _PlannedCall) -> None:
            call = item.call
            tool_name = item.tool.name if item.tool is not None else call.name
            await queue.put(
                ToolExecutionStartEvent(tool_call_id=call.id, tool_name=tool_name, args=call.arguments)
            )
            if item.failure is not None or item.tool is None:
                result = item.failure or self._synthetic_result(call, "Tool not found.")
            else:
                tool = item.tool

                def on_update(update: AgentToolUpdate) -> None:
                    queue.put_nowait(
                        ToolExecutionUpdateEvent(
                            tool_call_id=call.id, tool_name=tool.name, partial_result=update
                        )
                    )

                try:
                    result = await tool.execute(
                        call.id, item.args, signal, on_update, context.tool_context or ToolContext()
                    )
                except asyncio.CancelledError:
                    result = self._synthetic_result(call, ABORTED_RESULT_TEXT)
                except Exception as exc:
                    logger.warning("tool %s raised", tool.name, exc_info=True)
                    result = ToolResult(
                        tool_call_id=call.id,
                        tool_name=tool.name,
                        is_error=True,
                        content=[TextContent(text=f"Tool raised: {exc}")],
                    )
            results_by_id[call.id] = result
            await queue.put(
                ToolExecutionEndEvent(
                    tool_call_id=call.id, tool_name=tool_name, result=result, is_error=result.is_error
                )
            )
            await queue.put(_TOOL_DONE)

        tasks = [asyncio.ensure_future(runner(item)) for item in batch]
        finished = 0
        while finished < len(tasks):
            item = await queue.get()
            if item is _TOOL_DONE:
                finished += 1
                continue
            yield item
        await asyncio.gather(*tasks, return_exceptions=True)
        results.extend(results_by_id[item.call.id] for item in batch)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
            message = Message(
                role="tool",
                content=list(result.content),
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
        combined = AbortSignal.any_of(signal, deadline_signal) if signal is not None else deadline_signal
        return combined, task

    @staticmethod
    def _unwire_deadline(task: asyncio.Task[None] | None) -> None:
        if task is not None and not task.done():
            task.cancel()


def _materialize_asides(asides: Sequence[Any]) -> list[AgentMessage]:
    """Invoke aside thunks at injection time; drop ``None`` results and call
    commit/discard hooks on CustomMessage payloads."""
    out: list[AgentMessage] = []
    for item in asides:
        message = item() if callable(item) else item
        if message is None:
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

