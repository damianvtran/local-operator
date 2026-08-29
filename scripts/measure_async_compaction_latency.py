"""User-visible latency of a turn boundary that triggers a compaction pass.

The compaction ADVISOR was already off the critical path, but the PASS it
authorises was not: ``_run_compaction`` was awaited inline at the gates, and it
makes its own summarization provider call. That was invisible while a pass only
ever fired at the 600k ceiling (the turn had to stop there anyway) and becomes a
mid-conversation stall once the advisor starts firing passes early and often.

This script measures that stall on the REAL path — the real ``Session``, the
real ``_on_turn_end`` hook, the real gate, the real plan, the real commit — with
only the provider stream substituted, because a real summarization call is the
one part whose duration is the provider's and not ours. The substituted call
sleeps ``--summary-delay`` seconds (default 3.0, well under the 20-50 s a real
summarization of a large context takes) so the inline/async difference is read
off the clock rather than argued about.

Both arms run in ONE process against the SAME session shape:

  INLINE  — the pre-change behaviour, reproduced by calling the synchronous
            ``_run_compaction`` from the gate exactly as the old code did.
  ASYNC   — what ships: the pass is spawned and the boundary returns.

It also shows the property that makes the async arm honest: the pass is still
running when the boundary returns, the conversation CONTINUES, and the pass
applies at a later boundary with the work added in between still present.

Run:
    PYTHONPATH=. .venv/bin/python scripts/measure_async_compaction_latency.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from local_operator.compaction import api as compaction_api  # noqa: E402
from local_operator.compaction.advisor import CompactionHint  # noqa: E402
from local_operator.compaction.api import CompactionSettings  # noqa: E402
from local_operator.harness.types import (  # noqa: E402
    CompactionEndEvent,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    Usage,
)
from local_operator.session.protocol import CompactionOutcome  # noqa: E402
from local_operator.session.session import Session  # noqa: E402
from local_operator.session.transcript import Transcript  # noqa: E402

MODEL = ModelSpec(provider="test", model_id="opus-like", context_window=1_000_000)
KEEP_RECENT = 40
#: Below the 600k ceiling and inside the advisor's 300k-600k operating band:
#: the whole point is that this pass fires EARLY, when nothing forces relief.
CONTEXT_TOKENS = 400_000


class DelayedSummaryStream:
    """Ordinary turns answer instantly; the summarization call takes ``delay``.

    Recognised by the summarizer's own system prompt, which is what actually
    distinguishes the call on the wire.
    """

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.summary_calls = 0

    def __call__(self, request: Any, signal: Any):
        blocks = getattr(request, "system_blocks", None) or []
        summary = any("context compaction summarizer" in str(b) for b in blocks)
        if summary:
            self.summary_calls += 1
        delay = self.delay if summary else 0.0

        async def gen():
            if delay:
                await asyncio.sleep(delay)
            yield StreamTextDelta(delta="SUMMARY" if summary else "reply")
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def settings() -> CompactionSettings:
    return CompactionSettings(
        keep_recent_tokens=KEEP_RECENT,
        # context-full is the strategy that MAKES a summarization call;
        # snapcompact rasterizes locally and would have no latency to measure.
        strategy="context-full",
        advisor_enabled=True,
        advisor_floor_tokens=200_000,
        advisor_trigger_tokens=300_000,
        advisor_every_n_turns=1,
    )


async def build_session(tmp: Path, stream: DelayedSummaryStream) -> Session:
    session = Session(
        model=MODEL,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp),
        system_blocks_provider=lambda: ["stable system prompt"],
        compaction_settings=settings(),
    )
    for index in range(6):
        await session.prompt(f"question {index} " + "detail " * 30)
    return session


def pin_context(tokens: int) -> None:
    """Pin the rulers so both arms plan the identical cut."""
    compaction_api.messages_tokens_upper_bound = lambda messages: tokens
    compaction_api.estimate_messages_tokens = lambda messages: tokens


def seed_hint(session: Session) -> None:
    """Park a validated hint the way a completed advisor call would."""
    session._advisor_hint = CompactionHint(
        preserve_from_id=session._context.messages[-1].id,
        preserve_tokens=KEEP_RECENT,
        compact_now=True,
        confidence=0.9,
        reason="task boundary reached",
        turn_index=session._generation,
    )


async def boundary(session: Session, tokens: int) -> Any:
    """Drive the REAL mid-turn hook at a given provider-reported context size."""
    assistant = Message.assistant("mid-run reply")
    assistant.usage = Usage(input_tokens=tokens, output_tokens=10, context_tokens=tokens)
    return await session._on_turn_end([*session._context.messages, assistant])


async def measure_inline(delay: float) -> float:
    """The PRE-CHANGE behaviour: the gate awaits the pass in line."""
    with tempfile.TemporaryDirectory() as tmp:
        stream = DelayedSummaryStream(delay)
        session = await build_session(Path(tmp) / "s", stream)
        pin_context(CONTEXT_TOKENS)
        seed_hint(session)

        # Reproduce the old call site exactly: plan, then AWAIT the commit.
        started = time.monotonic()
        planned = await session._plan_compaction(respect_threshold=True)
        # A refusal comes back as a CompactionOutcome rather than a plan; both
        # arms must measure a pass that actually runs, so bail loudly.
        if isinstance(planned, CompactionOutcome):
            raise AssertionError(f"planning refused: {planned.reason} {planned.detail}")
        outcome = await session._run_compaction(planned, reason="mid-turn")
        elapsed = time.monotonic() - started
        assert outcome.ran, f"inline pass did not run: {outcome.reason}"
        await session.dispose()
        return elapsed


async def measure_async(delay: float) -> tuple[float, float, bool, bool]:
    """What ships. Returns (boundary latency, apply latency, landed, kept work)."""
    with tempfile.TemporaryDirectory() as tmp:
        stream = DelayedSummaryStream(delay)
        session = await build_session(Path(tmp) / "s", stream)
        events: list[Any] = []
        session.subscribe(events.append)
        pin_context(CONTEXT_TOKENS)
        seed_hint(session)

        started = time.monotonic()
        await boundary(session, CONTEXT_TOKENS)
        elapsed = time.monotonic() - started

        in_flight = session._compaction_pass_in_flight
        # The conversation CONTINUES while the pass runs — the concurrency the
        # feature exists for. This message is added AFTER the cut was planned.
        session._context.messages.append(Message.assistant("WORK-DURING-PASS"))

        while session._compaction_pass_in_flight:
            await asyncio.sleep(0.01)

        apply_started = time.monotonic()
        await boundary(session, CONTEXT_TOKENS)
        apply_elapsed = time.monotonic() - apply_started

        landed = any(isinstance(e, CompactionEndEvent) and e.success for e in events)
        kept = any(
            "WORK-DURING-PASS" in (getattr(m, "text", "") or "") for m in session._context.messages
        )
        await session.dispose()
        assert in_flight, "the pass was never spawned"
        return elapsed, apply_elapsed, landed, kept


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-delay", type=float, default=3.0)
    args = parser.parse_args()
    delay = args.summary_delay

    print("Async compaction pass: user-visible turn latency")
    print("=" * 64)
    print(f"Substituted summarization call: {delay:.1f}s")
    print(f"Context pinned at {CONTEXT_TOKENS:,} tokens (below the 600k ceiling).")
    print()

    inline = await measure_inline(delay)
    async_boundary, apply_elapsed, landed, kept = await measure_async(delay)

    print(f"  INLINE (before)  boundary blocked : {inline:6.3f}s")
    print(f"  ASYNC  (after)   boundary blocked : {async_boundary:6.3f}s")
    print(f"  ASYNC            apply boundary   : {apply_elapsed:6.3f}s")
    print()
    saved = inline - async_boundary
    print(f"  User-visible latency removed      : {saved:6.3f}s ({saved / inline:.1%})")
    print()
    print("Correctness of the async arm:")
    print(f"  pass applied at a LATER boundary  : {landed}")
    print(f"  work added mid-pass still present : {kept}")

    ok = landed and kept and async_boundary < inline / 2
    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
