#!/usr/bin/env python3
"""Estimate what coalescing a child's hub note with its own settle saves.

Context: ``SubagentComms.reply_to_parent`` now defers the parent's WAKE for an
unprompted note by ``harness.comms.HUB_NOTE_WAKE_GRACE_S`` (the note itself is
still queued at once). A parent parked in ``wait`` is then woken once — by the
settle, or by the window expiring — instead of once for the note and again for
the result. This script measures, from real transcripts, how many parent model
calls that removes and what they cost in input context.

What it counts, per top-level session in the window (read-only):

* a NOTE is a ``hub_message`` row with ``direction == "to_parent"``;
* it COALESCES under window G when the same job's ``job_result`` row lands
  within G seconds of the note (absent row = never settled in this transcript);
* its potentially avoidable calls are the parent's ``analytics.db`` ``calls`` rows (purpose
  ``turn``, opened ``?mode=ro``) that START inside ``[note, settle)`` —
  attributed ONLY when the note interrupted a parked ``wait`` (the wait's tool row
  reporting "a hub message arrived" sits just before the note), because then
  the note is the only reason the parent was awake. A parent that was busy
  with other tools would have made those calls anyway, so they are excluded.
  ``calls.ts_ms`` is the call's START while a transcript assistant row is
  stamped at its END, so the transcript cannot count these: the call reading
  the note is usually still streaming when the settle row lands;
* their input cost is those calls' ``context_tokens``. Overlapping spans are
  deduplicated by session and call-row index. These are attribution estimates
  (an upper bound), not proof that every counted call disappears: another
  arrival may independently require a call, or early delivery may discharge
  the note before the grace expires.

Top-level = a session id that never appears with a ``parent_session_id`` in
``calls`` (children's own notes go to THEIR parent, which may itself be a
child; those are excluded to match the measured population, top-level parents
at ~244k context).

It also reports how often the note's body reappears verbatim inside the
job_result text (the coalesced boundary then shows it twice; this PR does not
dedup, see the PR body).

Run: .venv/bin/python scripts/bench_wake_coalesce.py [--days 7] [--out FILE]
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

WINDOWS = (5.0, 10.0, 20.0, 30.0, 60.0, 120.0)
#: The tool row a note interrupt produces (``builtin._ARRIVAL_PHRASES``).
_HUB_INTERRUPT = "hub message arrived"
#: Rows to look back from a note for the interrupted wait: the wait's tool
#: result is journaled just before the drained note, with at most a sibling
#: row (another aside) between them.
_LOOKBACK = 4
#: Rows that are not a parent decision and never end the note->settle span.
_PASSIVE = ("session_state", "todo_snapshot", "subagent_progress")


def _child_sessions(db: sqlite3.Connection) -> set[str]:
    rows = db.execute("select distinct session_id from calls where parent_session_id != ''")
    return {row[0] for row in rows}


def _turn_calls(db: sqlite3.Connection, since_ms: int) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for sid, ts, ctx in db.execute(
        "select session_id, ts_ms, context_tokens from calls "
        "where purpose = 'turn' and parent_session_id = '' and ts_ms >= ? order by ts_ms",
        (since_ms,),
    ):
        out[sid].append((ts, ctx))
    return out


def _rows(path: Path) -> list[dict[str, Any]]:
    """Only the rows this analysis reads; skipping the rest by substring keeps a
    multi-GB week of transcripts to seconds instead of a full JSON parse."""
    keep = ('"role":"assistant"', '"hub_message"', '"job_result"', '"tool_name":"wait"')
    out = []
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if any(marker in line for marker in keep):
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
    except OSError:
        return []
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--days", type=float, default=7.0)
    parser.add_argument("--home", default=os.path.expanduser("~/.local-operator"))
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--max-sessions", type=int, default=310, help="Bound the recent top-level sample"
    )
    args = parser.parse_args()

    home = Path(args.home)
    cut = time.time() - args.days * 86400
    db = sqlite3.connect(f"file:{home / 'analytics.db'}?mode=ro", uri=True)
    children = _child_sessions(db)
    calls = _turn_calls(db, int(cut * 1000))
    db.close()

    sessions = notes = interrupted = 0
    settle_gaps: list[float] = []
    saved_calls: Counter[float] = Counter()
    saved_ctx: Counter[float] = Counter()
    attributed: dict[float, set[tuple[str, int]]] = defaultdict(set)
    coalesced: Counter[float] = Counter()
    coalesced_interrupting: Counter[float] = Counter()
    body_in_result = body_checked = 0

    candidates = sorted(
        (home / "sessions").glob("*/transcript.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for transcript in candidates:
        if sessions >= args.max_sessions:
            break
        sid = transcript.parent.name
        if sid in children:
            continue
        try:
            if transcript.stat().st_mtime < cut:
                continue
        except OSError:
            continue
        rows = _rows(transcript)
        sessions += 1
        session_calls = calls.get(sid, [])
        call_ts = [ts for ts, _ in session_calls]
        for index, row in enumerate(rows):
            payload = row.get("payload") or {}
            details = payload.get("details") or {}
            if payload.get("custom_type") != "hub_message":
                continue
            if details.get("direction") != "to_parent":
                continue
            note_ts = float(row.get("ts") or 0)
            if note_ts < cut:
                continue
            notes += 1
            job_id = details.get("job_id")
            was_interrupt = any(
                (prior.get("payload") or {}).get("tool_name") == "wait"
                and _HUB_INTERRUPT
                in "".join(
                    c.get("text", "") for c in (prior.get("payload") or {}).get("content") or []
                )
                for prior in rows[max(0, index - _LOOKBACK) : index]
            )
            interrupted += was_interrupt
            settle = None
            for later in rows[index + 1 :]:
                lp = later.get("payload") or {}
                if (
                    lp.get("custom_type") == "job_result"
                    and (lp.get("details") or {}).get("job_id") == job_id
                ):
                    settle = later
                    break
                elif lp.get("tool_name") == "wait" and str(job_id) in json.dumps(
                    lp.get("details") or {}
                ):
                    # The parent's own wait consumed the result: that tool row is
                    # the settle (no job_result row follows a consumed job).
                    if "still running" not in "".join(
                        c.get("text", "") for c in lp.get("content") or []
                    ):
                        settle = later
                        break
            if settle is None:
                continue
            gap = float(settle.get("ts") or 0) - note_ts
            settle_gaps.append(gap)
            body = (details.get("body") or "").strip()
            sp = settle.get("payload") or {}
            result_text = (sp.get("details") or {}).get("text") or "".join(
                c.get("text", "") for c in sp.get("content") or []
            )
            if body and result_text and gap <= 120:
                body_checked += 1
                body_in_result += body[:200] in result_text
            # Calls STARTED in [note, settle): the ones the note's wake bought.
            lo = bisect.bisect_left(call_ts, int(note_ts * 1000))
            hi = bisect.bisect_left(call_ts, int(float(settle.get("ts") or 0) * 1000))
            for window in WINDOWS:
                if gap <= window:
                    coalesced[window] += 1
                    if was_interrupt:
                        coalesced_interrupting[window] += 1
                        # Overlapping note spans can attribute the same parent
                        # call twice. Its per-session row index is stable even
                        # when two calls share a millisecond timestamp.
                        for call_index in range(lo, hi):
                            identity = (sid, call_index)
                            if identity not in attributed[window]:
                                attributed[window].add(identity)
                                saved_calls[window] += 1
                                saved_ctx[window] += session_calls[call_index][1]

    per_week = 7.0 / args.days
    lines = [
        f"window: last {args.days:g} days; top-level sessions scanned: {sessions}",
        f"to_parent notes: {notes}; notes that interrupted a parked wait: {interrupted}",
        (
            f"notes whose job settled (any delay): {len(settle_gaps)}; "
            f"median note->settle {statistics.median(settle_gaps):.1f}s"
            if settle_gaps
            else "no settles found"
        ),
        f"note body (first 200 chars) verbatim inside the job_result: "
        f"{body_in_result}/{body_checked} of notes settling within 120 s",
        "",
        "Attribution estimates, NOT demonstrated eliminated calls; overlapping spans deduplicated.",
        "Assumes the note alone caused these calls; concurrent arrivals and early "
        "delivery may reduce savings.",
        "grace_s  coalesced  coalesced&interrupting  parent_calls_estimate  "
        "input_tokens_estimate  per_week_tokens_estimate",
    ]
    for window in WINDOWS:
        lines.append(
            f"{window:7g}  {coalesced[window]:9d}  {coalesced_interrupting[window]:22d}  "
            f"{saved_calls[window]:18d}  {saved_ctx[window]:18,d}  "
            f"{int(saved_ctx[window] * per_week):15,d}"
        )
    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
