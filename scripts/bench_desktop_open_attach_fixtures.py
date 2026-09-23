"""Synthetic session fixtures shaped like this host's real store.

SHAPES, NOT CONTENT, measured 2026-09-22 over ~/.local-operator/sessions with
stat / wc -l / grep -c only (n = 9,359 transcripts):

  transcript bytes   p50 0.70 MB  p90 1.59 MB  p95 2.66 MB  p99 6.0 MB  max 269 MB
  rows (327 sampled) p50 251      p90 886      max 23,150
  bytes/row p50 ~2.7 KB; largest single row 40 KB - 1.37 MB (big tool outputs)
  the 269 MB session: 247 frontend checkpoints = 180 MB (67% of the file),
  54 compactions; typical sessions: 0-3 compactions, 0-1 checkpoints.

Every row is written through the real Transcript API (append_messages /
append_custom / append_compaction), so the bytes are what the product writes.

Run with the WORKTREE venv, cwd = the local-operator worktree, isolated HOME:
  python scripts/bench_desktop_open_attach_fixtures.py \
      --config-dir $ISO/.local-operator --profiles tiny,p50,p90,p99,m5000,xl \
      --manifest $ISO/manifest.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# name: (turns, tool_calls_per_turn, tool_output_bytes,
#        checkpoint_every, compaction_every, ckpt_jobs)
PROFILES = {
    "tiny": (5, 1, 400, 0, 0, 0),  # ~25 rows: the "empty-ish" conversation
    "p50": (40, 2, 6_000, 0, 0, 0),  # ~250 rows / ~0.7 MB
    "p90": (130, 2, 4_200, 0, 0, 0),  # ~900 rows / ~1.6 MB
    "p99": (300, 3, 5_500, 60, 150, 5),  # ~2.4k rows / ~6 MB, 1-2 compactions
    "m5000": (830, 2, 1_200, 100, 250, 5),  # ~5,000 rows
    # The operator's worst shape: thousands of turns and FAT checkpoints (job
    # rosters) that dominate the bytes. Expensive to build (~1-2 min).
    "xl": (3000, 2, 2_500, 12, 60, 60),  # ~18k rows, ~200 MB
    # Same bytes, but a PRE-v2 journal: only an unversioned `selected_model` row, so
    # read_model_selection falls through to _forward_payloads and JSON-decodes every
    # custom row (all the fat checkpoints). 1 of 17 sampled large real journals (31.8 MB).
    "xl_legacy": (3000, 2, 2_500, 12, 60, 60),
    "p99_legacy": (300, 3, 5_500, 60, 150, 5),
}


def _fat_state(session_id: str, jobs: int) -> dict[str, Any]:
    """A checkpoint whose weight is its job roster, as on the real large session."""
    return {
        "session_id": session_id,
        "epoch": uuid.uuid4().hex,
        "conversation_title": "bench",
        "todos": [],
        "jobs": [
            {
                "id": uuid.uuid4().hex[:12],
                "type": "task",
                "status": "completed",
                "label": f"job {i}",
                "trajectory": [],
                "prompt": "p" * 3_000,
                "result": "r" * 9_000,
            }
            for i in range(jobs)
        ],
    }


async def build(config_dir: Path, profile: str, seed: int = 0) -> str:
    from local_operator.harness.types import Message, TextContent, ToolCall
    from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE
    from local_operator.session.transcript import Transcript

    turns, calls, out_bytes, ck_every, cp_every, ck_jobs = PROFILES[profile]
    rnd = random.Random(seed)
    session_id = uuid.uuid4().hex[:12]
    directory = config_dir / "sessions" / session_id
    directory.mkdir(parents=True)
    (directory / "created_at.json").write_text(json.dumps(time.time()))
    (directory / "desktop.json").write_text(
        json.dumps({"version": 1, "cwd": str(config_dir.parent / "work")})
    )
    transcript = Transcript(directory)
    # The birth row every real journal carries at byte ~318 (measured: the newest
    # v2 `selected_model` row sits at offset 317-320 in 12 of 17 sampled large
    # journals — sessions that never switched model). Its DEPTH is what makes the
    # backward settle scan in model_selection.read_model_selection O(file).
    birth = {"version": 2, "selector": "test/mock", "effort": None, "boot": "test/mock"}
    if profile.endswith("_legacy"):
        birth = {"selector": "test/mock", "effort": None}
    await transcript.append_custom("selected_model", birth)
    block = "".join(rnd.choice("abcdefghij  \n") for _ in range(256))
    first_kept = None
    for turn in range(turns):
        user = Message(
            role="user", content=[TextContent(text=f"turn {turn}: " + "q" * rnd.randint(40, 600))]
        )
        batch: list[Any] = [user]
        for _ in range(calls):
            call = ToolCall(
                name="bash", arguments={"command": "echo " + "x" * rnd.randint(10, 200)}
            )
            batch.append(
                Message(role="assistant", content=[TextContent(text="running")], tool_calls=[call])
            )
            payload = block * max(1, out_bytes // 256)
            batch.append(
                Message(
                    role="tool",
                    tool_call_id=call.id,
                    tool_name="bash",
                    content=[TextContent(text=payload)],
                )
            )
        batch.append(
            Message(
                role="assistant",
                content=[
                    TextContent(
                        text="## answer\n"
                        + "a " * rnd.randint(100, 1500)
                        + "\n```py\nprint(1)\n```"
                    )
                ],
                stop_reason="stop",
            )
        )
        await transcript.append_messages(batch)
        if ck_every and turn % ck_every == ck_every - 1:
            await transcript.append_custom(
                FRONTEND_CHECKPOINT_CUSTOM_TYPE,
                {"checkpoint_id": uuid.uuid4().hex, "state": _fat_state(session_id, ck_jobs)},
            )
        if cp_every and turn % cp_every == cp_every - 1 and first_kept:
            await transcript.append_compaction("summary " * 400, first_kept, 100_000)
        if cp_every and turn % cp_every == cp_every - 10:
            first_kept = user.id
    return session_id


def main() -> None:
    sys.path.insert(0, str(Path.cwd()))
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", required=True, type=Path)
    ap.add_argument("--profiles", default="tiny,p50,p90,p99,m5000")
    ap.add_argument("--manifest", type=Path)
    args = ap.parse_args()
    out = json.loads(args.manifest.read_text()) if args.manifest and args.manifest.exists() else {}
    for name in args.profiles.split(","):
        started = time.perf_counter()
        sid = asyncio.run(build(args.config_dir, name))
        path = args.config_dir / "sessions" / sid / "transcript.jsonl"
        rows = sum(1 for _ in path.open("rb"))
        out[name] = {"session_id": sid, "bytes": path.stat().st_size, "rows": rows}
        print(
            f"{name}: {sid} {path.stat().st_size / 1e6:.2f} MB {rows} rows "
            f"({time.perf_counter() - started:.1f}s build)",
            flush=True,
        )
    if args.manifest:
        args.manifest.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
