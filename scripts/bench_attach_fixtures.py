"""Synthetic session fixtures shaped like the operator's real store.

WHY THESE SHAPES. Sized from metadata-only measurements of
~/.local-operator/sessions on 2026-09-22 (`ls -l`/`wc -l`, plus a shape-only
census that printed row kinds and byte counts, never content):

  transcripts n=9357  p50=0.70 MB  p90=1.59 MB  p99=6.0 MB  max=269 MB
  rows (sampled, last 7 days) p50=281  p90=678  p99=2360  max=23147
  bytes/row p50 ~2.6 KB, p90 ~3.7 KB
  of 5443 transcripts touched in the last 7 days:
      only 236 (4.3%) carry a frontend_state_checkpoint_v1 row
      only 337 (6.2%) carry a compaction row
  byte share (5 large sessions): compaction 73% (preserve_data +
      preserved_user_turns, ~0.6-1.5 MB PER compaction row), tool 11%,
      assistant 10%, session_state 3%.

So the COMMON case is: 0.7-2 MB, ~300-700 rows, NO compaction and NO
checkpoint. That matters because the cold read's backward scan
(`read_replay_suffix`) stops only at a compaction boundary AND the newest
checkpoint; without them it reads the whole file (transcript.py:1051-1055).

Every fixture is written through the product's own `Transcript` append API so
the row encoding (exclude_defaults, raw_arguments drop, attachment
externalisation) is exactly what production writes.

Presets (name -> rows, ~bytes):
  s50     :   50 rows  (~0.1 MB)     small chat
  p50     :  300 rows  (~0.7 MB)     median real session  (no compaction/checkpoint)
  p90     :  650 rows  (~1.6 MB)     p90 real session     (no compaction/checkpoint)
  s5000   : 5000 rows  (~12 MB)      long, no compaction  (worst "whole file" case)
  c5000   : 5000 rows + 3 compactions with ~0.8 MB preserve payloads + checkpoint (~15 MB)
  huge    : 16000 rows + 20 compactions (~60 MB)  mirrors the 53-66 MB tail

Usage (from the repo root; ``scripts/bench_attach_latency.py`` measures against it):

  .venv/bin/python scripts/bench_attach_fixtures.py "$ISO/.local-operator" p50 s5000

The root is the ISOLATED config dir (``LOCAL_OPERATOR_CONFIG_DIR``), never the
operator's live ``~/.local-operator``: the rig writes into the sessions it is given.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import uuid
from pathlib import Path
from typing import Any

PRESETS: dict[str, dict[str, Any]] = {
    "s50": dict(turns=8, tool_per_turn=2, out_bytes=600, compactions=0, checkpoint=False),
    "p50": dict(turns=45, tool_per_turn=2, out_bytes=3200, compactions=0, checkpoint=False),
    "p90": dict(turns=100, tool_per_turn=2, out_bytes=3200, compactions=0, checkpoint=False),
    "s5000": dict(turns=800, tool_per_turn=2, out_bytes=3200, compactions=0, checkpoint=False),
    "c5000": dict(turns=800, tool_per_turn=2, out_bytes=3200, compactions=3, checkpoint=True),
    "huge": dict(turns=2600, tool_per_turn=2, out_bytes=3200, compactions=20, checkpoint=True),
}


def _filler(rng: random.Random, n: int) -> str:
    words = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
        "def return import await async class self path json line file"
    ).split()
    out = []
    size = 0
    while size < n:
        w = rng.choice(words)
        out.append(w)
        size += len(w) + 1
    return " ".join(out)


async def build(directory: Path, preset: str, seed: int = 7) -> dict[str, Any]:
    from local_operator.harness.types import Message, TextContent, ToolCall
    from local_operator.session.transcript import Transcript

    spec = PRESETS[preset]
    rng = random.Random(seed)
    directory.mkdir(parents=True, exist_ok=True)
    t = Transcript(directory)
    turns = spec["turns"]
    comp_at = set()
    if spec["compactions"]:
        step = turns // (spec["compactions"] + 1)
        comp_at = {step * (i + 1) for i in range(spec["compactions"])}
    first_kept: str | None = None
    batch: list[Any] = []
    for i in range(turns):
        user = Message(role="user", content=[TextContent(text=f"turn {i}: " + _filler(rng, 180))])
        batch.append(user)
        if i and i % 5 == 0:
            first_kept = user.id
        for k in range(spec["tool_per_turn"]):
            cid = f"call_{i}_{k}_{uuid.uuid4().hex[:6]}"
            batch.append(
                Message(
                    role="assistant",
                    content=[TextContent(text=_filler(rng, spec.get("asst_bytes", 1500)))],
                    tool_calls=[
                        ToolCall(
                            id=cid,
                            name="bash",
                            arguments={"command": "rg -n x " + _filler(rng, 60)},
                        )
                    ],
                    stop_reason="toolUse",
                )
            )
            batch.append(
                Message(
                    role="tool",
                    content=[TextContent(text=_filler(rng, spec["out_bytes"]))],
                    tool_call_id=cid,
                    tool_name="bash",
                )
            )
        batch.append(
            Message(
                role="assistant",
                content=[TextContent(text=_filler(rng, spec.get("asst_bytes", 1500)))],
                stop_reason="stop",
            )
        )
        if len(batch) > 400:
            await t.append_messages(batch)
            batch = []
        # Bookkeeping rows real sessions carry (session_spend.v1 is ~1 per provider call).
        if i % 3 == 0:
            if batch:
                await t.append_messages(batch)
                batch = []
            await t.append_custom("session_spend.v1", {"calls": i, "micro": i * 13})
        if i in comp_at and first_kept is not None:
            if batch:
                await t.append_messages(batch)
                batch = []
            await t.append_compaction(
                "summary " + _filler(rng, 600),
                first_kept,
                150_000,
                preserve_data={"blob": _filler(rng, 600_000)},
                preserved_user_turns=[
                    {"role": "user", "text": _filler(rng, 400)} for _ in range(400)
                ],
            )
    if batch:
        await t.append_messages(batch)
    if spec["checkpoint"]:
        # A realistic-size checkpoint is written by the product's own store in the
        # runtime; the cold path only needs the ROW to exist to stop the backward scan.
        await t.append_custom(
            "frontend_state_checkpoint_v1",
            {
                "checkpoint_id": uuid.uuid4().hex,
                "state": {"cwd": str(directory), "pad": _filler(rng, 30_000)},
            },
        )
    t.flush()
    path = directory / "transcript.jsonl"
    rows = sum(1 for _ in path.open("rb"))
    return {"preset": preset, "rows": rows, "bytes": path.stat().st_size}


def main() -> None:
    root = Path(sys.argv[1])
    presets = sys.argv[2:] or list(PRESETS)
    out = {}
    for p in presets:
        sid = f"{p:0<12}"[:12] if len(p) < 12 else p[:12]
        sid = (p.encode().hex() + "0" * 12)[:12]
        out[p] = asyncio.run(build(root / "sessions" / sid, p)) | {"session_id": sid}
        print(json.dumps(out[p]), flush=True)


if __name__ == "__main__":
    main()
