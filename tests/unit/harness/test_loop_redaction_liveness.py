"""Event-loop liveness while a tool result is REDACTED: the subagent-slowness fix.

The credential scan every tool result goes through — ``VariableStore.
redact_with_report`` → ``scrub_secrets_with_hits`` → ``scrub_shapes_with_hits``
— used to run SYNCHRONOUSLY ON THE EVENT LOOP, from ``AgentLoop._append_results``
and the nested-eval path. It is C-level work over an unbounded result, so the
one loop serving the parent, **every concurrent child** and the TUI was parked
for its duration. Measured on this host, same text and same pass:

    | workload (scrub_shapes_with_hits, origin/main) | inline on loop | to_thread |
    |---|---|---|
    | plain 1.00 MB, no anchor                     | 583 ms,  12 heartbeats | 228 ms, 16 |
    | 1 MB + ONE ``postgres://user:…@host`` line   | 4,972 ms,  9 heartbeats | 4,024 ms, 281 |

9-12 heartbeats against 157-281. The hop does not make the scan faster; it lets
the loop serve every other child and the frame while it runs, which is the
reported "subagents are slow because they block each other".

This file holds that property two ways at once, from one run of the real
workload, following ``tests/unit/tools/test_loop_liveness.py`` (PR #1422's rig
for the same pass on the bash stream):

* structurally — a spy ``redact`` records the thread it was called on. That is
  the contract stated exactly, and no machine load can perturb it. It is the
  half with teeth here, because the pass is pure CPU and the assertion is
  binary.
* temporally — a :class:`LoopCpuProbe` samples the loop thread's own CPU time
  per wake, so a synchronous stretch shows up as one sample the size of the
  stretch. Loop-thread CPU rather than wall-clock gaps between wakes, because
  the gap moves with load (a busy machine steals wall time from the loop thread
  without handing it CPU time) while this statistic does not.

:data:`MAX_LOOP_CPU_S` is set the way PR #1422 set its bound — clear of the
servicing cost, and far under the cheapest re-introduced stall (the 286 ms
measured for a plain 1 MB with no anchor at all, and seconds for one carrying
an anchor). ``test_the_rig_runs_the_real_scan...`` pins that the fixture is
genuinely over the bound so the temporal half cannot pass by measuring nothing.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from local_operator.harness.loop import AgentLoop, LoopContext
from local_operator.harness.redaction import current_tool_source
from local_operator.redaction_shapes import scrub_secrets_with_hits
from local_operator.harness.types import (
    AgentEndEvent,
    AgentTool,
    ChatRequest,
    LoopConfig,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolResult,
)

MODEL = ModelSpec(provider="test", model_id="m")

# Cadence of the liveness probe, matching the tools rig.
HEARTBEAT_S = 0.02
#: The most CPU time the loop thread may accumulate between two heartbeat
#: wakes. Servicing the hop plus the append costs it single-digit
#: milliseconds; running the anchor-bearing scan on the loop costs it seconds.
MAX_LOOP_CPU_S = 0.08

#: Enough anchor-bearing text that the pass is seconds of C-level work when it
#: runs inline, so the bound is clear by more than an order of magnitude on any
#: machine this suite runs on.
RESULT_MB = 2
ANCHOR = "postgres://user:[redacted]@db.example.com:5432/app\n"
MASKED_ANCHOR = "postgres://[redacted]\n"


class LoopCpuProbe:
    """Heartbeat recording the loop thread's CPU time between its wakes.

    ``time.thread_time`` is per-thread and excludes time asleep or waiting on
    the GIL, so a sample is large only when the loop thread genuinely ran
    without yielding — never merely because the machine was busy.
    """

    def __init__(self) -> None:
        self.samples: list[float] = []
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
        last = time.thread_time()
        while not self._stop.is_set():
            await asyncio.sleep(HEARTBEAT_S)
            now = time.thread_time()
            self.samples.append(now - last)
            last = now

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        assert self._task is not None
        await self._task

    @property
    def worst(self) -> float:
        return max(self.samples)


class RedactSpy:
    """The REAL session hook, spied.

    The pass is ``scrub_secrets_with_hits`` — the function
    ``VariableStore.redact_with_report`` calls — over the fixture's own
    anchor-bearing text, so both assertions are about the real workload: the
    thread it ran on, and the CPU of one uninterrupted loop-thread frame it
    cost. A stub would make the temporal half measure nothing.

    ``mask_anchor`` swaps the DSN for a marker so the behavioural half can see
    the pass ran; the cost of that swap is negligible beside the scan.
    """

    def __init__(self) -> None:
        self.threads: list[int] = []
        # Built inside the test coroutine, so this IS the loop's thread.
        self.loop_thread = threading.get_ident()
        self.sources: list[tuple[str, str]] = []
        self.texts: list[str] = []

    def __call__(self, text: str) -> str:
        self.threads.append(threading.get_ident())
        self.texts.append(text)
        # What the explicit publish inside the worker buys: the hook needs to
        # know which call the bytes belong to in order to report a masked result.
        self.sources.append(current_tool_source())
        masked, _hits = scrub_secrets_with_hits(text)
        return masked.replace(ANCHOR, MASKED_ANCHOR)

    @property
    def result_calls(self) -> list[int]:
        """Thread ids for the calls carrying the RESULT's own bytes.

        The host hook is also reached by ``_scrub_history_arguments`` — the
        assistant turn's arguments, a different pass over a different string —
        so that traffic is filtered out and the assertions below are about the
        tool result's redaction alone.
        """
        return [
            thread
            for thread, text in zip(self.threads, self.texts)
            if "x" * 27 in text or ANCHOR.strip() in text
        ]

    def assert_off_loop(self) -> None:
        calls = self.result_calls
        assert calls, "the redact hook never ran — the seam stopped calling it"
        on_loop = [i for i in calls if i == self.loop_thread]
        assert not on_loop, (
            f"the redact hook ran on the event-loop thread {len(on_loop)} of "
            f"{len(calls)} times — a credential scan over a tool result is "
            "back on the one loop serving every child and the frame"
        )


def _anchor_bearing_text(megabytes: int) -> str:
    """Many SHORT lines plus ONE DSN-shaped line.

    Line COUNT is what the pass costs here, and that is the amplification the
    defect rides on. ``has_shape_anchor`` gates the whole rule table per line —
    ``text.lower()`` plus a scan of ``_SHAPE_ANCHORS``, which carries bare
    substrings like ``key``/``token``/``://`` — and ONE matching anchor anywhere
    disables the gate for EVERY line, so the rule table then runs per line. A
    single 2 MB line would pay one gate; 65k short lines pay 65k of them, which
    is the shape a real tool result (a log, a diff, a listing) actually has.
    """
    line = "x" * 27 + "\n"
    count = (megabytes * 1024 * 1024) // len(line)
    return line * count + ANCHOR


class _ScriptedStream:
    """One tool call, then a plain stop — the shortest path into ``_append_results``."""

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: Any):
        self.requests.append(request)
        turn: list[StreamEvent] = (
            [
                StreamToolCallDelta(index=0, id="c1", name="collect", argument_delta='{"text": "hi"}'),
                StreamEndEvent(stop_reason="toolUse"),
            ]
            if len(self.requests) == 1
            else [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")]
        )

        async def gen():
            for event in turn:
                yield event

        return gen()


def _collect_tool(seen: list[str], content: list[Any], *, name: str = "collect") -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update, context):
        seen.append(tool_call_id)
        return ToolResult(tool_call_id=tool_call_id, tool_name=name, content=content)

    return AgentTool(
        name=name,
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        execute=execute,
    )


def _config(stream: Any, redact: Any) -> LoopConfig:
    return LoopConfig(
        model=MODEL,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=stream,
        redact_tool_result=redact,
    )


async def _run_loop(context: LoopContext, config: LoopConfig) -> list[Any]:
    events: list[Any] = []
    async for event in AgentLoop().run([Message.user("go")], context, config, None):
        events.append(event)
    return events


def _tool_rows(events: list[Any]) -> list[Message]:
    end = events[-1]
    assert isinstance(end, AgentEndEvent)
    return [m for m in end.messages if isinstance(m, Message) and m.role == "tool"]


@pytest.mark.asyncio
async def test_the_loop_serves_heartbeats_while_a_large_result_is_redacted():
    """A multi-MB anchor-bearing tool result must not park the event loop.

    Today the pass runs on the loop thread, so the loop cannot serve ANY other
    task while a credential-bearing result is masked. The host hook here IS the
    real session hook — ``scrub_secrets_with_hits``, C-level work over an
    unbounded result — so the number below is the real pass, not a stand-in.

    The sample is the loop thread's own CPU time INTERRUPTED BY A YIELD, i.e.
    exactly the work performed on the loop thread that was not run by an awaited
    coroutine. A whole-pass sample is charged as one loop-thread frame, and the
    same pass on a worker thread charges the loop thread only the hop — so the
    two worlds separate by two orders of magnitude. (Sampling the rate over many
    ``await asyncio.sleep`` ticks would not: those coroutines run ON the loop
    thread, so a pass that runs between ticks is indistinguishable from one the
    loop interleaves for free.)
    """
    seen: list[str] = []
    context = LoopContext(
        tools=[_collect_tool(seen, [TextContent(text=_anchor_bearing_text(RESULT_MB))])]
    )
    spy = RedactSpy()
    config = _config(_ScriptedStream(), spy)

    probe = LoopCpuProbe()
    probe.start()
    try:
        events = await _run_loop(context, config)
    finally:
        await probe.stop()

    assert seen, "the tool never ran; liveness proved nothing"
    assert _tool_rows(events), "no tool row landed; the redaction path never ran"
    spy.assert_off_loop()
    assert probe.samples, "heartbeat never woke"
    assert probe.worst < MAX_LOOP_CPU_S, (
        f"{probe.worst:.3f}s of the credential scan ran in one uninterrupted frame on "
        f"the loop thread while a {RESULT_MB} MB anchor-bearing result was redacted — "
        "the scan is back on the one loop serving every child and the frame"
    )


@pytest.mark.asyncio
async def test_the_redacted_result_is_byte_identical_and_identifies_its_call():
    """The behavioural half: the hop changes WHERE the pass runs, not WHAT it returns.

    Pins the two things a naive refactor loses. (1) The masked text is exactly
    what the inline pass produced. (2) The tool IDENTITY is published inside the
    worker, so a real host's shape-hit reporter names the call it masked — the
    explicit publish is what makes the seam correct for either spelling of the
    hop (``asyncio.to_thread`` copies the calling context, a bare
    ``run_in_executor`` does not).
    """
    text = "prefix " + ANCHOR + "suffix"
    seen: list[str] = []
    context = LoopContext(tools=[_collect_tool(seen, [TextContent(text=text)])])
    spy = RedactSpy()

    rows = _tool_rows(await _run_loop(context, _config(_ScriptedStream(), spy)))

    assert len(rows) == 1, rows
    assert rows[0].content[0].text == "prefix " + MASKED_ANCHOR + "suffix"
    assert spy.sources, "the hook never ran"
    name, summary = spy.sources[-1]
    assert name == "collect", spy.sources
    assert summary, "the arguments summary must ride with the name"


@pytest.mark.asyncio
async def test_content_with_no_text_block_pays_no_hop():
    """The negative arm of the seam: an empty/imagery result stays on the loop.

    A hop with nothing to mask would be pure scheduling cost, so the decision is
    taken ON the loop and the worker is never entered. Exercised against the
    seam itself, because ``_append_results`` also drives the restore-history
    argument scrub — a different pass this seam deliberately does not touch.
    """
    calls: list[str] = []

    def redact(value: str) -> str:
        calls.append(value)
        return value

    content: list[Any] = []
    out = await AgentLoop._redact_content(content, redact, "collect", {"text": "hi"})
    assert calls == [], "an all-empty result was still handed to the redactor"
    assert out is content, "the no-text path must return the caller's own list"


@pytest.mark.asyncio
async def test_every_text_block_of_a_multi_block_result_is_masked():
    """The hop is per-result, not per-first-block.

    A seam that masked only ``content[0]`` would pass a single-block test and
    leak every later block, so a multi-block result is pinned.
    """
    seen: list[str] = []
    context = LoopContext(
        tools=[
            _collect_tool(
                seen,
                [TextContent(text="a " + ANCHOR), TextContent(text="b " + ANCHOR)],
            )
        ]
    )
    spy = RedactSpy()
    rows = _tool_rows(await _run_loop(context, _config(_ScriptedStream(), spy)))

    assert [block.text for block in rows[0].content] == [
        "a " + MASKED_ANCHOR,
        "b " + MASKED_ANCHOR,
    ]
    assert len(spy.result_calls) == 2, spy.result_calls


def test_the_rig_runs_the_real_scan_so_the_temporal_half_can_bite() -> None:
    """Why the fixture is the real pass and not a sleep.

    The bound above only means something if the work under it is the expensive
    thing. This pins the amplification the fixture relies on: ONE anchor line
    turns the per-line gate on for EVERY line, so a multi-MB result pays the
    whole rule table.
    """
    from local_operator.redaction_shapes import has_shape_anchor, scrub_shapes_with_hits

    assert has_shape_anchor(ANCHOR.strip()), "the fixture must carry a real anchor"
    text = _anchor_bearing_text(RESULT_MB)
    started = time.perf_counter()
    scrub_shapes_with_hits(text)
    elapsed = time.perf_counter() - started
    assert elapsed > MAX_LOOP_CPU_S, (
        f"the fixture finished in {elapsed:.4f}s — too small to distinguish a stall "
        "from servicing; raise RESULT_MB"
    )
