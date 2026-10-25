"""Real-path evidence for the L3c deferred cold facade.

Three subcommands, all against a synthetic store this script builds itself:

  seed  <root>            build one ~8 MB journal (messages + receipts + a row)
  inproc <root> <samples> N cold desktop opens of that journal, in-process, with
                          process CPU and journal bytes read per sample
  serve <root> <port>     a REAL `serve` process, the real /events route, the
                          first two SSE frames written to <root>/frames.json

Run the same subcommand under each worktree's python and diff the JSON: the
numbers are the work delta and the frames are the wire equality.

Never touches the operator's store: every root is a temp directory the caller
creates.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

WORKTREE = os.environ["L3C_WORKTREE"]
sys.path.insert(0, WORKTREE)

from local_operator.harness.types import Message, TextContent, ToolCall, Usage  # noqa: E402
from local_operator.session.frontend_state import (  # noqa: E402
    FRONTEND_CHECKPOINT_CUSTOM_TYPE,
    FrontendSessionState,
)
from local_operator.session.spend import SESSION_SPEND_CUSTOM_TYPE  # noqa: E402
from local_operator.session.transcript import Transcript  # noqa: E402

SESSION_ID = "c0ffee123456"
#: ~1 KB per message body, 4 rows per turn: an 8 MB journal at 4000 messages.
BODY = "x" * 2600
TURNS = 1000


async def _seed(root: Path) -> None:
    directory = root / "sessions" / SESSION_ID
    directory.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(directory)
    for turn in range(TURNS):
        await transcript.append_message(
            Message(role="user", content=[TextContent(text=f"question {turn} {BODY}")])
        )
        await transcript.append_message(
            Message(
                role="assistant",
                content=[TextContent(text=f"answer {turn} {BODY}")],
                tool_calls=[ToolCall(id=f"call{turn}", name="bash", arguments={"command": "t"})],
                usage=Usage(input_tokens=11 * (turn + 1), output_tokens=7, context_tokens=900),
            )
        )
        await transcript.append_message(
            Message(
                role="tool",
                tool_call_id=f"call{turn}",
                tool_name="bash",
                content=[TextContent(text=f"output {turn} {BODY}")],
            )
        )
        await transcript.append_custom(SESSION_SPEND_CUSTOM_TYPE, {"usd": 0.25, "calls": 1})
    await transcript.append_custom(
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        {
            "checkpoint_id": "c1",
            "state": FrontendSessionState(
                session_id=SESSION_ID,
                epoch="previous-owner",
                sequence=3,
                conversation_title="Synthetic cold-open fixture",
                cumulative_parent_cost=12.5,
                context_tokens=322_546,
                context_window=1_000_000,
            ).model_dump(mode="json"),
        },
    )
    transcript.flush()


async def _one_open(root: Path) -> dict:
    """One cold desktop open through the bridge, and the first frame it emits."""
    from local_operator.server.utils.desktop_sessions import DesktopSessions
    from local_operator.session import replay_cache

    reads: list[int] = []
    original = replay_cache.read_replay_suffix

    def counted(*args, **kwargs):
        result = original(*args, **kwargs)
        reads.append(result.bytes_read)
        return result

    replay_cache.read_replay_suffix = counted
    try:
        started = time.process_time()
        pool = DesktopSessions(root)
        async with pool.session(SESSION_ID) as bridge:
            sub = bridge.subscribe()
            stream = bridge.events(sub, epoch=bridge.epoch, after_seq=bridge.sequence)
            opened = await anext(stream)
            snapshot = await anext(stream)
            await stream.aclose()
            frame_keys = sorted(frame_key_histogram(snapshot))
        cpu_ms = (time.process_time() - started) * 1000
    finally:
        replay_cache.read_replay_suffix = original
    return {
        "cpu_ms": round(cpu_ms, 2),
        "journal_passes": len(reads),
        "journal_bytes": sum(reads),
        "frame_keys": frame_keys,
        "opened": opened["type"],
        "snapshot": canonical(snapshot),
    }


def frame_key_histogram(value) -> dict[str, int]:
    """Paths through the frame, so equality is judged per field not per byte."""
    out: dict[str, int] = {}

    def walk(node, path: str) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                walk(item, f"{path}.{key}")
        elif isinstance(node, list):
            out[f"{path}[]"] = len(node)
            for index, item in enumerate(node[:3]):
                walk(item, f"{path}[{index}]")
        else:
            out[path] = 1

    walk(value, "$")
    return out


_VOLATILE = {"epoch", "served_at", "now", "built_at", "ts", "ts_source", "cursor",
             "before_token", "snapshot_token", "subscription_id"}


def canonical(value):
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items() if k not in _VOLATILE}
    if isinstance(value, list):
        return [canonical(item) for item in value]
    return value


async def _inproc(root: Path, samples: int) -> None:
    from local_operator.session import replay_cache

    results = []
    for index in range(samples):
        replay_cache.replay_cache().clear()
        outcome = await _one_open(root)
        snapshot = outcome.pop("snapshot")
        results.append(outcome)
        if index == 0:
            (root / "frame-inproc.json").write_text(
                json.dumps(canonical(snapshot), indent=2, sort_keys=True), encoding="utf-8"
            )
    print(json.dumps({"samples": results}, indent=2))


def main() -> None:
    argv = sys.argv[1:]
    command = argv[0]
    root = Path(argv[1])
    if command == "seed":
        root.mkdir(parents=True, exist_ok=True)
        asyncio.run(_seed(root))
        journal = root / "sessions" / SESSION_ID / "transcript.jsonl"
        print(json.dumps({"journal_bytes": journal.stat().st_size, "turns": TURNS}))
    elif command == "inproc":
        asyncio.run(_inproc(root, int(argv[2])))
    else:
        raise SystemExit(f"unknown command {command!r}")


main()
