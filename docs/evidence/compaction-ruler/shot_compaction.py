"""Render the compaction receipt notice through the REAL OperatorApp.

Drives a real Session through a real compact_now() with the provider's
context_tokens pinned high (as after a long session), then feeds the resulting
outcome into the app's CompactionEnded message so the notice on screen is the
one the shipped code computes. The frame therefore moves when session.py's
tokens_after arithmetic moves, which is the point of the before/after pair.

Usage: shot_compaction.py <out.svg> [repo_root]
"""

from __future__ import annotations

import asyncio
import os
import sys

REPO = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/local-operator")
sys.path.insert(0, REPO)

from local_operator.compaction.api import CompactionSettings  # noqa: E402
from local_operator.harness.types import (  # noqa: E402
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    Usage,
)
from local_operator.session.session import Session  # noqa: E402
from local_operator.session.transcript import Transcript  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.events import CompactionEnded  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

TEXT_MODEL = ModelSpec(
    provider="test", model_id="reads", context_window=1_000_000, supports_images=False
)


class ScriptedStream:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, request, signal):
        self.requests.append(request)
        index = len(self.requests) - 1
        reply = self.replies[index] if index < len(self.replies) else "ok"

        async def gen():
            yield StreamTextDelta(delta=reply)
            yield StreamEndEvent(stop_reason="stop")

        return gen()


async def real_outcome(tmp):
    stream = ScriptedStream(["assistant reply " * 400] * 8)
    session = Session(
        model=TEXT_MODEL,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp / "sess"),
        system_blocks_provider=lambda: ["stable"],
        compaction_settings=CompactionSettings(keep_recent_tokens=2_000),
    )
    for index in range(8):
        await session.prompt(f"question {index} " + "detail " * 40)
    # A provider reading as after a long real session: the receipt's "before"
    # is this figure, so the "after" is the number under test.
    session._last_usage = Usage(input_tokens=1, context_tokens=546_458)
    outcome = await session.compact_now()
    await session.dispose()
    return outcome


async def main() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        outcome = await real_outcome(Path(td))
    print(f"tokens_before={outcome.tokens_before:,} tokens_after={outcome.tokens_after:,}")

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 18)) as pilot:
        await pilot.pause()
        app.post_message(
            CompactionEnded(
                reason="threshold",
                success=True,
                strategy=outcome.strategy,
                tokens_before=outcome.tokens_before,
                tokens_after=outcome.tokens_after,
            )
        )
        await pilot.pause()
        await pilot.pause()
        app.save_screenshot(sys.argv[1])


asyncio.run(main())
