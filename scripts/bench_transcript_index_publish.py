#!/usr/bin/env python3
"""Benchmark transcript-index PUBLICATION cost per one-row append.

Usage:

    PYTHONPATH=. .venv/bin/python scripts/bench_transcript_index_publish.py \
        [--turns 14000] [--appends 20] [--keep]

WHY THIS EXISTS. ``write_index`` used to re-serialise and fsync the WHOLE derived
document on every refresh, and the live trigger (the journal-order correction in
``session/runtime/serving.py``) refreshes on every durable commit. On a real
470 MB journal that meant 62.9 MB written and ~1.3 s of CPU per appended row.
The incremental store (``transcript_index.py``, "Incremental publication")
appends a checksummed delta frame and replaces a few-KB head instead. This
script measures both layouts on the SAME code by running the classic layout as
the control (``_STORE_MIN_BYTES`` raised past any document, so every write is
base's whole-document write) and the store as the treatment.

It builds a synthetic journal in a temporary directory (default 14k turns,
which derives a ~59 MB index like the operator's largest session), deletes it
afterwards unless ``--keep`` is given, and never touches a live config dir.

Reported per layout: median/max wall and CPU per append refresh, bytes written
per publication (new or rewritten files whole, the delta by its growth), and a
cold reader's ``read_index`` time (a fresh process has no parse memo), so the
read-side amplification of snapshot + delta is visible next to the write win.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_operator.session import transcript_index as ti  # noqa: E402

SID = "benchpublish01"


def _journal(root: Path, turns: int) -> Path:
    path = root / "sessions" / SID / "transcript.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    filler = "lorem ipsum dolor sit amet " * 30
    with path.open("w", encoding="utf-8") as handle:
        for turn in range(turns):
            for role, text in (("user", f"question {turn} "), ("assistant", f"answer {turn} ")):
                row = {
                    "id": uuid.uuid4().hex,
                    "ts": float(turn),
                    "type": "message",
                    "payload": {
                        "kind": "message",
                        "role": role,
                        "content": [{"text": text + filler}],
                    },
                }
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return path


def _files(root: Path) -> dict[str, int]:
    return {p.name: p.stat().st_size for p in ti._index_dir(root).glob(f"{SID}.*")}


def _written(before: dict[str, int], after: dict[str, int]) -> int:
    total = 0
    for name, size in after.items():
        if name.endswith(".log") and name in before:
            total += size - before[name]
        elif name.endswith((".json", ".log")) and (
            name not in before or before[name] != size or name == f"{SID}.json"
        ):
            total += size
    return total


def _cold_read(root: Path) -> float:
    code = (
        "import sys,time;"
        f"sys.path.insert(0,{str(Path(__file__).resolve().parents[1])!r});"
        "from local_operator.session import transcript_index as ti;"
        f"t=time.perf_counter();ix=ti.read_index({str(root)!r},{SID!r});"
        "assert ix is not None;print(time.perf_counter()-t)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def _run(root: Path, journal: Path, appends: int, store: bool) -> dict[str, float]:
    ti._reset_for_tests()
    ti._STORE_MIN_BYTES = (1 << 20) if store else (1 << 62)
    shutil.rmtree(ti._index_dir(root), ignore_errors=True)
    ti.refresh_index(root, SID)
    walls, cpus, written = [], [], []
    for _ in range(appends):
        row = {
            "id": uuid.uuid4().hex,
            "ts": time.time(),
            "type": "message",
            "payload": {"kind": "message", "role": "assistant", "content": [{"text": "more"}]},
        }
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        before = _files(root)
        wall, cpu = time.perf_counter(), time.process_time()
        ti.refresh_index(root, SID)
        walls.append(time.perf_counter() - wall)
        cpus.append(time.process_time() - cpu)
        written.append(_written(before, _files(root)))
    median = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
    return {
        "wall_ms_median": median(walls) * 1000,
        "wall_ms_max": max(walls) * 1000,
        "cpu_ms_median": median(cpus) * 1000,
        "bytes_per_publication_median": median(written),
        "bytes_total": sum(written),
        "cold_read_s": _cold_read(root),
        "footprint_bytes": sum(_files(root).values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--turns", type=int, default=14000)
    parser.add_argument("--appends", type=int, default=20)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="bench-ti-publish-"))
    try:
        journal = _journal(root, args.turns)
        print(f"journal {journal.stat().st_size / 1e6:.1f} MB, {args.turns * 2} rows")
        for label, store in (("classic (control)", False), ("store", True)):
            result = _run(root, journal, args.appends, store)
            print(
                f"{label:18} wall median {result['wall_ms_median']:7.1f} ms "
                f"(max {result['wall_ms_max']:7.1f}) cpu median {result['cpu_ms_median']:7.1f} ms "
                f"bytes/publication {result['bytes_per_publication_median']:>12,} "
                f"total {result['bytes_total'] / 1e6:7.1f} MB "
                f"cold read_index {result['cold_read_s'] * 1000:6.0f} ms "
                f"index on disk {result['footprint_bytes'] / 1e6:5.1f} MB"
            )
    finally:
        if args.keep:
            print(f"kept {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
