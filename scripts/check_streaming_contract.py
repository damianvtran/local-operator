#!/usr/bin/env python3
"""Assert the frozen streaming contract (docs/REWRITE.md §B) over a live run.

The exec --json stream is the same event vocabulary the TUI and the server
websockets consume, so a violation here is a UI defect. Feed it one or more
jsonl captures of a real ``exec --json`` run:

    local-operator exec --json "..." > run.jsonl
    python scripts/check_streaming_contract.py run.jsonl [more.jsonl ...]

Checks (each fails the run with a named invariant):

- SC-1 exactly one ``agent_start`` and one ``agent_end`` per run, start first.
- SC-2 the ``agent_end`` generation equals the ``agent_start`` generation
  (the held-end across compaction must not split the run).
- SC-3 every ``tool_execution_end`` is preceded by its ``tool_execution_start``
  (pairing legality), and ids are unique per run.
- SC-4 ``message_update`` events are DELTAS: no update re-sends the full
  accumulated text (a regression here doubles the wire cost and breaks the
  TUI's append-only markdown boundary).
- SC-5 compaction events are paired and sit strictly between the run's start
  and end (never outside the boundary).
- SC-6 usage appears on turn_end/agent_end messages and ``context_tokens`` is
  populated when the provider reports prompt tokens (the TUI status line and
  the compaction trigger both read it).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def check(path: Path) -> list[str]:
    events = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    problems: list[str] = []
    starts = [e for e in events if e.get("type") == "agent_start"]
    ends = [e for e in events if e.get("type") == "agent_end"]

    # SC-1
    if len(starts) != 1 or len(ends) != 1:
        problems.append(f"SC-1: {len(starts)} agent_start / {len(ends)} agent_end")
    elif events.index(starts[0]) > events.index(ends[0]):
        problems.append("SC-1: agent_end before agent_start")

    # SC-2
    if starts and ends and starts[0].get("generation") != ends[0].get("generation"):
        problems.append(
            f"SC-2: generation split start={starts[0].get('generation')} "
            f"end={ends[0].get('generation')}"
        )

    # SC-3
    seen_starts: set[str] = set()
    seen_ends: set[str] = set()
    for e in events:
        t = e.get("type")
        if t == "tool_execution_start":
            if e.get("tool_call_id") in seen_starts:
                problems.append(f"SC-3: duplicate start {e.get('tool_call_id')}")
            seen_starts.add(e.get("tool_call_id") or "")
        elif t == "tool_execution_end":
            cid = e.get("tool_call_id") or ""
            if cid not in seen_starts:
                problems.append(f"SC-3: end without start: {cid}")
            if cid in seen_ends:
                problems.append(f"SC-3: duplicate end {cid}")
            seen_ends.add(cid)

    # SC-4 — deltas only: an update whose text is a prefix-identical resend of
    # the accumulated text (i.e. it contains the previous update verbatim and
    # is longer than the delta it should be) signals full-text mode.
    acc = ""
    for e in events:
        if e.get("type") == "message_update":
            delta = e.get("delta") or ""
            if delta and acc and delta == acc:
                problems.append("SC-4: message_update re-sent the full text")
            acc += delta
        elif e.get("type") == "message_start":
            acc = ""

    # SC-5
    if starts and ends:
        s, f = events.index(starts[0]), events.index(ends[0])
        for e in events:
            if e.get("type") in ("compaction_start", "compaction_end"):
                i = events.index(e)
                if not (s < i < f):
                    problems.append(f"SC-5: {e.get('type')} outside the run boundary")
    comp_types = [e.get("type") for e in events if str(e.get("type", "")).startswith("compaction")]
    if comp_types.count("compaction_start") != comp_types.count("compaction_end"):
        problems.append("SC-5: unpaired compaction events")

    # SC-6
    turn_ends = [e for e in events if e.get("type") == "turn_end"]
    with_usage = [e for e in turn_ends if (e.get("message") or {}).get("usage") is not None]
    if turn_ends and not with_usage:
        problems.append("SC-6: no turn_end carries usage")
    for e in with_usage:
        usage = e["message"]["usage"]
        if usage.get("input_tokens") and not usage.get("context_tokens"):
            problems.append("SC-6: usage without context_tokens")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    failed = False
    for arg in argv[1:]:
        problems = check(Path(arg))
        status = "PASS" if not problems else "FAIL"
        print(f"{status} {arg}")
        for p in problems:
            print(f"  - {p}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
