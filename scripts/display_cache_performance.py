"""Measure cold/warm canonical display requests and real owner socket latency.

    .venv/bin/python scripts/display_cache_performance.py --output /tmp/owner-cache
    .venv/bin/python scripts/display_cache_performance.py --source-root BASE --output OUT

Both trees run the same harness, with synthetic data and isolated HOME/config.
The optional --fleet measures cache retention for 50 transcript owners, not the
entire memory footprint of 50 running agent processes. No timing is a CI limit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent.parent)
PARSER.add_argument("--output", type=Path, required=True)
PARSER.add_argument("--messages", type=int, default=20_000)
PARSER.add_argument("--samples", type=int, default=5)
PARSER.add_argument("--fleet", action="store_true")
ARGS = PARSER.parse_args()
if ARGS.messages < 140 or ARGS.samples < 1:
    PARSER.error("--messages must be at least 140; --samples must be positive")
ARGS.output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ARGS.source_root.resolve()))
for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        del os.environ[_key]

import scripts.probe_isolation as isolation  # noqa: E402

# isort: split
import local_operator  # noqa: E402
from local_operator.harness.types import Message  # noqa: E402
from local_operator.mobile.attach_client import AttachClient  # noqa: E402
from local_operator.session.history_window import display_window  # noqa: E402
from local_operator.session.runtime.server import RuntimeServer  # noqa: E402
from local_operator.session.runtime.serving import ServingSessionHandle  # noqa: E402
from local_operator.session.transcript import Transcript  # noqa: E402
from tests.e2e.harness import ScriptedStream, build_session  # noqa: E402

ROOT = isolation.SANDBOX / "config"
RESULTS: list[dict[str, Any]] = []


def record(case: str, **values: Any) -> None:
    row = {"case": case, "source": str(local_operator.__file__), **values}
    RESULTS.append(row)
    print(json.dumps(row), flush=True)


def measure(call: Callable[[], Any]) -> tuple[Any, dict[str, float]]:
    cpu, wall = [], []
    result = None
    for _ in range(ARGS.samples):
        c, w = time.thread_time(), time.monotonic()
        result = call()
        cpu.append((time.thread_time() - c) * 1000)
        wall.append((time.monotonic() - w) * 1000)
    return result, {
        "median_cpu_ms": statistics.median(cpu),
        "max_cpu_ms": max(cpu),
        "median_wall_ms": statistics.median(wall),
        "max_wall_ms": max(wall),
    }


def cache_bytes(transcript: Transcript) -> int:
    return int(getattr(getattr(transcript, "_display_window_cache", None), "retained_bytes", 0))


async def main() -> None:
    directory = ROOT / "sessions/000000000001"
    transcript = Transcript(directory)
    await transcript.append_messages(
        [
            (
                Message.user(f"synthetic content {i}" + " x" * 500)
                if i % 2 == 0
                else Message.assistant(f"synthetic answer {i}" + " x" * 500)
            )
            for i in range(ARGS.messages)
        ]
    )
    cut = transcript.entries()[-1].id

    def capture():
        return display_window(
            transcript,
            conversation_id=directory.name,
            owner_epoch="synthetic-epoch",
            through_id=cut,
        )

    def cold():
        # Baseline has no cache attribute. Resetting only the new private cache
        # measures admission separately from warm reuse, not filesystem startup.
        if hasattr(transcript, "_display_window_cache"):
            transcript._display_window_cache = None
        return capture()

    first, cold_cost = measure(cold)
    _, warm_cost = measure(capture)

    def previous():
        return display_window(
            transcript,
            conversation_id=directory.name,
            owner_epoch="synthetic-epoch",
            through_id=cut,
            before=first.before_token,
        )

    previous()
    _, older_cost = measure(previous)
    record(
        "display",
        messages=ARGS.messages,
        durable_bytes=transcript.path.stat().st_size,
        returned=len(first.messages),
        wire_bytes=len(first.model_dump_json()),
        cold=cold_cost,
        warm=warm_cost,
        warm_older=older_cost,
        retained_bytes=cache_bytes(transcript),
    )
    session = build_session(directory, ScriptedStream([]), cwd=isolation.SANDBOX)
    server = None
    established = None
    writers = []
    try:

        def subscribe():
            subscription = session.subscribe_frontend(lambda _: None, display_window=True)
            subscription.unsubscribe()
            return subscription

        subscribe()
        _, subscription_cost = measure(subscribe)
        record("actual_warm_subscribe", **subscription_cost)
        handle = ServingSessionHandle(
            session, asyncio.get_running_loop(), cwd=str(isolation.SANDBOX)
        )
        server = RuntimeServer(handle, kind="daemon")
        await server.start_in_process()
        established = AttachClient(lambda _: None, lambda _: None)
        await established.connect(server.record, directory.name)
        gaps = []
        done = False

        async def ticker() -> None:
            previous_time = time.monotonic()
            while not done:
                await asyncio.sleep(0.005)
                now = time.monotonic()
                gaps.append((now - previous_time) * 1000)
                previous_time = now

        tick = asyncio.create_task(ticker())
        await asyncio.sleep(0.03)
        readers = []
        for _ in range(3):
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", server.record.control_port, limit=2 * 1024 * 1024
            )
            writer.write(
                (
                    json.dumps(
                        {
                            "key": server.record.control_key,
                            "client": "attach",
                            "frontend_state": True,
                            "events": True,
                            "display_window": True,
                        }
                    )
                    + "\n"
                ).encode()
            )
            await writer.drain()
            readers.append(reader)
            writers.append(writer)
        start = time.monotonic()
        answer = await established._request("ping")
        rtt = (time.monotonic() - start) * 1000
        for reader in readers:
            assert json.loads(await reader.readline())["op"] == "projection"
            assert json.loads(await reader.readline())["op"] == "frontend_sync"
        done = True
        await tick
        record(
            "three_full_attaches", control_result=answer, ping_ms=rtt, max_5ms_tick_gap_ms=max(gaps)
        )
        for writer in writers:
            writer.close()
            await writer.wait_closed()
        writers.clear()
        await established.detach()
        established = None
        for _ in range(1000):
            if not server._clients:
                break
            await asyncio.sleep(0)
        assert not server._clients
        with patch.object(
            server, "_projection_payload", wraps=server._projection_payload
        ) as payload:
            start = time.thread_time()
            for _ in range(100):
                await server._push()
            record(
                "no_recipient_push",
                pushes=100,
                payload_calls=payload.call_count,
                cpu_ms=(time.thread_time() - start) * 1000,
            )
    finally:
        for writer in writers:
            writer.close()
            await writer.wait_closed()
        if established is not None:
            await established.detach()
        if server is not None:
            await server.aclose()
        await session.dispose()
    if ARGS.fleet:
        owners = []
        for index in range(50):
            owner = Transcript(ROOT / "sessions" / f"{index + 2:012x}")
            await owner.append_messages(
                [Message.user(f"row {i} " + "x" * 1000) for i in range(1000)]
            )
            display_window(
                owner,
                conversation_id=owner.directory.name,
                owner_epoch="fleet",
                through_id=owner.entries()[-1].id,
            )
            owners.append(owner)
        record(
            "fifty_owner_cache",
            owners=50,
            messages_each=1000,
            retained_bytes=sum(cache_bytes(owner) for owner in owners),
            max_owner_bytes=max(cache_bytes(owner) for owner in owners),
        )
    (ARGS.output / "results.json").write_text(json.dumps(RESULTS, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
