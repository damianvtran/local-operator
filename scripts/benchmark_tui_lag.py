"""Reproduce long-session TUI and canonical-state performance scenarios.

Run from any directory with the repository's interpreter:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/benchmark_tui_lag.py

The output is evidence, not a CI timing gate. Unit tests assert the structural
properties that make these paths bounded without depending on machine speed.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# An absolute script invocation otherwise puts only ``scripts/`` on sys.path;
# resolve the checkout from this file so the harness works from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from textual.app import App, ComposeResult  # noqa: E402

from local_operator.session.frontend_state import (  # noqa: E402
    FrontendSessionState,
    FrontendStateStore,
    JobState,
)
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.subagent_view import fold_trajectory  # noqa: E402
from local_operator.tui.widgets.tool_card import ToolCard  # noqa: E402
from local_operator.tui.widgets.transcript import (  # noqa: E402
    NoticeBlock,
    TranscriptView,
    UserBlock,
)


class _TranscriptHost(App[None]):
    def compose(self) -> ComposeResult:
        yield TranscriptView()


def _summary(samples: list[float]) -> str:
    ordered = sorted(samples)
    p95 = ordered[int(0.95 * (len(ordered) - 1))]
    return (
        f"median={statistics.median(ordered) * 1_000:.3f} ms "
        f"p95={p95 * 1_000:.3f} ms max={max(ordered) * 1_000:.3f} ms"
    )


def _measure(operation: Callable[[], Any], *, samples: int = 15) -> str:
    elapsed = []
    for _ in range(samples):
        started = time.perf_counter()
        operation()
        elapsed.append(time.perf_counter() - started)
    return _summary(elapsed)


def _events(count: int = 500, payload_chars: int = 240) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    index = 0
    while len(events) + 5 <= count:
        body = f"message {index} " + "x" * payload_chars
        events.extend(
            [
                {"type": "message_start", "message": {"role": "assistant", "id": f"m{index}"}},
                {
                    "type": "message_update",
                    "message": {"role": "assistant", "id": f"m{index}"},
                    "delta": body,
                },
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "id": f"m{index}",
                        "content": [{"type": "text", "text": body}],
                    },
                },
                {
                    "type": "tool_execution_start",
                    "tool_call_id": f"t{index}",
                    "tool_name": "read",
                    "args": {"path": f"src/module_{index}.py"},
                },
                {
                    "type": "tool_execution_end",
                    "tool_call_id": f"t{index}",
                    "tool_name": "read",
                    "result": {"content": [{"type": "text", "text": "ok"}]},
                },
            ]
        )
        index += 1
    return events[:count]


def _large_store(children: int = 100, events_per_child: int = 500) -> FrontendStateStore:
    jobs = [
        JobState(
            id=f"child-{child}",
            type="task",
            label=f"job {child}",
            trajectory=[
                {"type": "message_update", "index": event, "delta": "x" * 120}
                for event in range(events_per_child)
            ],
        )
        for child in range(children)
    ]
    return FrontendStateStore(
        FrontendSessionState(session_id="benchmark", epoch="local", jobs=jobs)
    )


def benchmark_state() -> None:
    store = _large_store()
    print("state read, 100 children x 500 events:", _measure(lambda: store.state))
    print("unchanged scalar mutation:", _measure(lambda: store.mutate(streaming=False)))

    generations = iter(range(1, 16))
    print(
        "changed scalar mutation:",
        _measure(lambda: store.mutate(generation=next(generations))),
    )

    jobs = store._state.jobs
    print("unchanged jobs mutation:", _measure(lambda: store.mutate(jobs=jobs)))

    append_index = 500

    def append_progress() -> None:
        nonlocal append_index
        current = store._state.jobs
        changed = current[-1].model_copy(
            update={
                "trajectory": [
                    *current[-1].trajectory,
                    {"type": "notice", "index": append_index},
                ]
            }
        )
        append_index += 1
        store.mutate(jobs=[*current[:-1], changed])

    print("one-job progress append:", _measure(append_progress))


def benchmark_fold() -> None:
    events = _events()
    print(
        "subagent fold, 500 retained events:",
        _measure(lambda: fold_trajectory(events), samples=200),
    )


async def benchmark_mounted() -> None:
    for blocks in (1_000,):
        app = _TranscriptHost()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            view = app.query_one(TranscriptView)
            started = time.perf_counter()
            with view.batch_append():
                for index in range(blocks):
                    view.append_block(UserBlock(f"prompt {index} " + "word " * 16))
            await pilot.pause()
            await pilot.pause()
            elapsed_ms = (time.perf_counter() - started) * 1_000
            print(
                f"mount {blocks} transcript blocks: {elapsed_ms:.3f} ms "
                f"virtual={view.virtual_size} viewport={view.size}"
            )

    app = _TranscriptHost()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        view = app.query_one(TranscriptView)
        cards = []
        with view.batch_append():
            for index in range(500):
                card = ToolCard(f"tool-{index}", "read", {"path": f"src/module_{index}.py"})
                card.mark_done("ok")
                cards.append(card)
                view.append_block(card)
        await pilot.pause()
        await pilot.pause()
        samples = []
        for _ in range(30):
            started = time.perf_counter()
            cards[-1].refresh_row()
            await pilot.pause()
            samples.append(time.perf_counter() - started)
        print(
            "refresh one of 500 tool cards:",
            _summary(samples),
            f"virtual={view.virtual_size} viewport={view.size}",
        )

    for target_chars in (100_000, 500_000):
        app = _TranscriptHost()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            view = app.query_one(TranscriptView)
            with view.batch_append():
                for index in range(1_000):
                    view.append_block(NoticeBlock(f"prior event {index}", "info"))
            block = AssistantBlock()
            view.append_block(block)
            await pilot.pause()
            await pilot.pause()
            text = "x" * (target_chars - 128)
            block.update_text(text)
            await pilot.pause()
            samples = []
            for _ in range(20):
                text += "streaming words " * 8
                started = time.perf_counter()
                block.update_text(text)
                await pilot.pause()
                samples.append(time.perf_counter() - started)
            print(f"stream near {target_chars:,} chars over 1,000 blocks:", _summary(samples))


def main() -> None:
    benchmark_state()
    benchmark_fold()
    asyncio.run(benchmark_mounted())


if __name__ == "__main__":
    main()
