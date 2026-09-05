"""Offline before/after session microbenchmarks. No credentials or provider calls."""

import asyncio
import hashlib
import json
import os
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))


async def main():
    from local_operator.harness.types import (
        Message,
        ModelSpec,
        StreamEndEvent,
        StreamTextDelta,
    )
    from local_operator.providers.clients import AnthropicClient
    from local_operator.session.goal import GoalState
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript
    from local_operator.session_factory import (
        _KnowledgeHooks,
        _make_system_blocks_provider,
    )

    source_paths = [
        "local_operator/session/session.py",
        "local_operator/session/transcript.py",
        "local_operator/session_factory.py",
        "local_operator/prompts_api.py",
        "local_operator/harness/loop.py",
        "local_operator/providers/clients.py",
    ]
    result = {
        "checkout": root.name,
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in source_paths
        },
    }
    times, counts, loop_writes = [], [], []
    real_fsync = os.fsync
    loop_thread = threading.get_ident()
    for repeat in range(7):
        calls = []

        def fsync(fd):
            calls.append(threading.get_ident())
            real_fsync(fd)

        os.fsync = fsync
        with tempfile.TemporaryDirectory() as directory:
            store = Transcript(directory)
            messages = [Message.user("message " + str(i)) for i in range(48)]
            start = time.perf_counter()
            if hasattr(store, "append_messages"):
                await store.append_messages(messages)
            else:
                for message in messages:
                    await store.append_message(message)
            times.append((time.perf_counter() - start) * 1000)
            counts.append(len(calls))
            loop_writes.append(sum(thread == loop_thread for thread in calls))
        os.fsync = real_fsync
    result["paired_flush_48_messages"] = {
        "median_ms": statistics.median(times),
        "samples_ms": times,
        "fsync_counts": counts,
        "event_loop_fsync_counts": loop_writes,
    }

    with tempfile.TemporaryDirectory() as directory:
        # Seed on disk in one fixture write to keep setup out of the benchmark.
        rows = []
        for i in range(5000):
            message = (
                Message.user("first task") if i == 0 else Message.assistant("historical " + str(i))
            )
            rows.append(
                json.dumps(
                    {
                        "id": message.id,
                        "ts": float(i),
                        "type": "message",
                        "payload": {"kind": "message", **message.model_dump(exclude={"id"})},
                    }
                )
            )
        Path(directory, "transcript.jsonl").write_text("\n".join(rows) + "\n")
        store = Transcript(directory)
        provider = _make_system_blocks_provider([], store, _KnowledgeHooks(), cwd=directory)
        await provider()
        times = []
        for repeat in range(7):
            start = time.thread_time()
            for call in range(100):
                await provider()
            times.append((time.thread_time() - start) * 1000)
        result["prompt_prepare_100_calls_5000_rows"] = {
            "median_thread_cpu_ms": statistics.median(times),
            "samples_thread_cpu_ms": times,
        }

    with tempfile.TemporaryDirectory() as directory:
        requests = []

        def stream(request, signal):
            requests.append(request)

            async def events():
                yield StreamTextDelta(delta="done")
                yield StreamEndEvent(stop_reason="stop")

            return events()

        store = Transcript(directory)
        goal = GoalState()
        provider = _make_system_blocks_provider(
            [], store, _KnowledgeHooks(), cwd=directory, goal_state=goal
        )
        session = Session(
            model=ModelSpec(provider="anthropic", model_id="audit", context_window=1000000),
            stream_fn=stream,
            tools=[],
            transcript=store,
            system_blocks_provider=provider,
            goal_state=goal,
        )
        await session.prompt("Investigate " + "representative history " * 10000)
        goal.set("Only inspect files under src")
        await session.prompt("Continue")
        client = AnthropicClient()
        bodies = [client._build_body(request) for request in requests]

        def hierarchy(body):
            return "".join(
                json.dumps(body.get(key, []), separators=(",", ":"))
                for key in ("tools", "system", "messages")
            )

        before, after = map(hierarchy, bodies)
        common = len(os.path.commonprefix((before, after)))
        result["goal_change_real_anthropic_wire"] = {
            "initial_chars": len(before),
            "matching_prefix_chars": common,
            "matching_prefix_ratio": common / len(before),
            "system_blocks_unchanged": requests[0].system_blocks == requests[1].system_blocks,
        }
        await client.aclose()
        await session.dispose()
    print(json.dumps(result, indent=2))


asyncio.run(main())
