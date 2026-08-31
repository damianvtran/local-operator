"""Benchmark the mobile relay's session-load paths against the REAL store.

Paths measured (these are what a phone opening a session pays):

- ``list``    : SessionTable.summaries() — the durable listing scan the list
                routes read, measured cold (first call) and warm (TTL cache).
- ``durable`` : mobile.daemon._durable_projection(sid) — the SSE seed frame
                for a session with no live process: full transcript parse +
                fold + roster rebuild (cold), then the incremental fold cache
                (warm).
- ``history`` : mobile.daemon._history_page(sid, None, 80) — the transcript
                page the phone fetches on open and on every scroll-up,
                cold and repeated.
- ``parse``   : raw Transcript(dir) construction alone, so the split between
                JSONL parse and fold is visible.

Each durable path is measured cold (first call) and warm (median of repeats).
Run with ``PYTHONPATH=.`` so THIS checkout's ``local_operator`` is imported.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from local_operator.mobile.daemon import (  # noqa: E402
    SessionTable,
    _durable_projection,
    _history_page,
)
from local_operator.paths import config_dir  # noqa: E402
from local_operator.session.transcript import Transcript  # noqa: E402


def pick_sessions(n: int = 8) -> list[tuple[str, int]]:
    """The n largest user-session transcripts: worst case is the common case."""
    sessions = config_dir() / "sessions"
    rows: list[tuple[str, int]] = []
    for child in sessions.iterdir():
        t = child / "transcript.jsonl"
        if t.is_file():
            rows.append((child.name, t.stat().st_size))
    rows.sort(key=lambda r: -r[1])
    return rows[:n]


def time_call(fn, *args, repeats: int = 3) -> tuple[float, float]:
    """(first-call seconds, median-of-repeats seconds)."""
    t0 = time.perf_counter()
    fn(*args)
    first = time.perf_counter() - t0
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        samples.append(time.perf_counter() - t0)
    return first, statistics.median(samples)


def main() -> None:
    out: dict[str, object] = {"python": sys.version.split()[0]}

    # The list path: summaries() cold (durable scan) and warm (TTL cache).
    table = SessionTable()
    first_list, warm_list = time_call(lambda: asyncio.run(table.summaries()))
    out["list_summaries_cold_s"] = round(first_list, 4)
    out["list_summaries_warm_s"] = round(warm_list, 6)
    out["list_rows"] = len(asyncio.run(table.summaries()))

    targets = pick_sessions()
    out["sessions"] = []
    for sid, size in targets:
        directory = config_dir() / "sessions" / sid
        first_p, warm_p = time_call(lambda: _durable_projection(sid))
        first_h, warm_h = time_call(lambda: _history_page(sid, None, 80))

        def _parse() -> int:
            return len(Transcript(directory)._entries)

        first_t, warm_t = time_call(_parse)
        out["sessions"].append(
            {
                "session_id": sid,
                "transcript_mb": round(size / 1e6, 1),
                "durable_projection_first_s": round(first_p, 3),
                "durable_projection_warm_s": round(warm_p, 3),
                "history_page_first_s": round(first_h, 3),
                "history_page_warm_s": round(warm_h, 3),
                "transcript_parse_first_s": round(first_t, 3),
                "transcript_parse_warm_s": round(warm_t, 3),
                "entries": _parse(),  # cheap now, page cache warm
            }
        )

    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
