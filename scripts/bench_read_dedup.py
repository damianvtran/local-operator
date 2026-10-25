#!/usr/bin/env python3
"""Replay real transcripts through the REAL read-dedup ledger and report savings.

What it measures
----------------
For every ``read`` result in each transcript, would
``local_operator.tools.read_dedup.ReadLedger.find`` have answered it with a
pointer — and what would that have saved? The ledger, coverage rule, residency
check (unblanked, under the replay bound, same ``tool_call_id`` in the live
list) and pointer text are the production objects, not re-implementations.
The live list is rebuilt from the transcript the way a Session holds it:

* tool results are appended after their whole assistant batch, as the loop
  does, so a read cannot dedup against a twin in its own parallel batch;
* a journalled ``prune`` entry blanks its target (``pruning._blank``);
* a ``compaction`` entry replaces the live list with the rows from
  ``first_kept_entry_id`` onward, exactly what invalidates residency live.

The one thing a replay cannot know is file CONTENT at the time — so the
fingerprint is a PROXY: a per-path version counter bumped by any ``edit`` /
``write`` of that path, and bumped for EVERY path by a mutating shell or
``eval`` command (the same conservative regex as the measurement that motivated
this work). "No mutation in between" stands in for "same sha256". Real
external edits (an editor, a sibling session) are invisible to the replay, so
the figures are an upper bound on what the live hash check lets through.

Persistence-weighted savings multiply each pointer's saved characters by the
number of later provider calls (assistant messages) before the next compaction
or the end of the transcript: that is how often the duplicate would have been
re-billed.

Read-only on the store: transcripts are CLONED into ``--work`` (APFS
``cp -c``; a plain copy elsewhere) and only the copies are opened. Never
point ``--work`` inside the sessions directory.

Usage::

    .venv/bin/python scripts/bench_read_dedup.py --work ~/workspace/read-dedup-bench \
        --limit 400 > "$LOCAL_OPERATOR_SCRATCHPAD/bench_read_dedup.txt"
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from local_operator.compaction.pruning import _blank
from local_operator.harness.replay_bound import DEFAULT_REPLAY_BOUND_CHARS
from local_operator.harness.types import Message, TextContent
from local_operator.tools.read_dedup import (
    MAX_ENTRIES,
    FileFingerprint,
    LedgerEntry,
    ReadLedger,
    dedup_notice,
    visible_end,
)

#: Shell/eval commands treated as mutating every file (the motivating
#: measurement's regex, kept identical so the two numbers are comparable).
MUTATING = re.compile(
    r"git (checkout|reset|rebase|merge|pull|stash|switch|am|apply|cherry-pick)|sed -i|>\s*\S"
    r"|\bmv\b|\bcp\b|python3? -|patch|perl -pi|uv run|make |ruff (format|check --fix)"
)
CLAMPED = ("[truncated at ", " more lines in file. Use range to continue]")
RANGE = re.compile(r"^(\d+)\s*-\s*(\d+)?$")
#: Cheap pre-filter on raw bytes; the store writes compact JSON today but the
#: spacing is not a contract, so match either.
READ_ROW = re.compile(rb'"tool_name":\s*"read"')


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") for block in content or () if isinstance(block, dict)
    )


def recordable(args: dict, details: dict, body: str) -> tuple[int, int | None] | None:
    """The requested ``(start, end)`` when this read is one the live tool
    would record/look up (plain text file, not raw-shaped, not clamped)."""
    path = str(args.get("path") or "")
    if not path or "://" in path or path.lower().startswith("memory:"):
        return None
    if not isinstance(details.get("path"), str):
        return None
    if any(k in details for k in ("url", "summary", "mime_type", "useless", "spill_id")):
        return None
    if body.startswith("Directory listing") or body == "(empty file)":
        return None
    spec = str(args.get("range") or "").strip()
    if not spec:
        return 1, None
    match = RANGE.match(spec)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)) if match.group(2) else None


def replay(path: Path) -> Counter:
    stats: Counter = Counter()
    live: list[Message] = []
    by_entry: dict[str, Message] = {}
    order: list[tuple[str, Message]] = []
    ledger = ReadLedger(lambda: live, replay_bound=DEFAULT_REPLAY_BOUND_CHARS)
    versions: Counter = Counter()
    epoch = 0
    calls: dict[str, tuple[str, dict]] = {}
    pending: list[tuple[str, Message]] = []
    # (saved_chars, assistant_count_at_dedup, epoch) for persistence weighting.
    hits: list[tuple[int, int, int]] = []
    # The same weighting for EVERY read result: the denominator that makes the
    # weighted saving a share rather than a bare number.
    landed: list[tuple[int, int, int]] = []
    assistants_per_epoch: Counter = Counter()
    assistant_count = 0

    def flush() -> None:
        for entry_id, message in pending:
            live.append(message)
            by_entry[entry_id] = message
            order.append((entry_id, message))
        pending.clear()

    with path.open() as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            kind = record.get("type")
            payload = record.get("payload") or {}
            if kind == "compaction":
                flush()
                epoch += 1
                first = payload.get("first_kept_entry_id")
                ids = [eid for eid, _ in order]
                kept = order[ids.index(first) :] if first in ids else []
                live[:] = [m for _, m in kept]
                order[:] = kept
                stats["compactions"] += 1
                continue
            if kind == "prune":
                target = by_entry.get(payload.get("target"))
                if target is not None:
                    _blank(target, payload.get("notice") or "[pruned]")
                continue
            if kind != "message":
                continue
            role = payload.get("role")
            if role != "tool":
                flush()
            if role == "assistant":
                assistant_count += 1
                assistants_per_epoch[epoch] += 1
                for call in payload.get("tool_calls") or ():
                    calls[call.get("id")] = (call.get("name"), call.get("arguments") or {})
                placeholder = Message(role="assistant", content=[])
                order.append((record.get("id"), placeholder))
                live.append(placeholder)
                continue
            if role != "tool":
                message = Message(role="user", content=[])
                order.append((record.get("id"), message))
                live.append(message)
                continue
            call_id = payload.get("tool_call_id") or ""
            name, args = calls.get(call_id, (payload.get("tool_name"), {}))
            body = text_of(payload.get("content"))
            pp = payload.get("provider_payload") or {}
            details = pp.get("details") if isinstance(pp.get("details"), dict) else {}
            is_error = bool(payload.get("is_error"))
            content_out = body

            if name in ("edit", "write") and isinstance(args, dict):
                versions[str(details.get("path") or args.get("path") or "")] += 1
            elif name in ("bash", "eval") and MUTATING.search(
                str((args or {}).get("command") or (args or {}).get("code") or "")
            ):
                versions["*"] += 1

            if name == "read" and not is_error:
                stats["reads"] += 1
                stats["read_chars"] += len(body)
                landed.append((len(body), assistant_count, epoch))
                span = recordable(args or {}, details, body)
                if span is not None:
                    realpath = details["path"]
                    start, end = span
                    version = (versions[realpath], versions["*"])

                    def current(v: tuple[int, int] = version) -> FileFingerprint:
                        return FileFingerprint(size=0, mtime_ns=0, sha256=repr(v))

                    if (args or {}).get("raw"):
                        if ledger.find(realpath, start, end, current) is not None:
                            stats["raw_bypassed"] += 1
                            stats["raw_bypassed_chars"] += len(body)
                    else:
                        entry = ledger.find(realpath, start, end, current)
                        if entry is not None:
                            shown = entry.end if end is None else min(end, entry.end)
                            notice = dedup_notice(realpath, start, shown, entry.tool_call_id)
                            if len(notice) < entry.chars:
                                saved = len(body) - len(notice)
                                stats["deduped"] += 1
                                stats["saved_chars"] += saved
                                hits.append((saved, assistant_count, epoch))
                                content_out = notice
                    if content_out is body:
                        # The production rule: record what the model sees.
                        clamped = any(marker in body for marker in CLAMPED)
                        last = start + body.count("\n") if not clamped else None
                        eof = not clamped and (end is None or (last is not None and last < end))
                        shown = visible_end(body, start, DEFAULT_REPLAY_BOUND_CHARS)
                        if shown is not None:
                            if last is None or shown < last:
                                eof = False
                            ledger.record(
                                LedgerEntry(
                                    realpath, start, shown, eof,
                                    current(), call_id, chars=len(body),
                                )
                            )
            message = Message(
                role="tool",
                content=[TextContent(text=content_out)],
                tool_call_id=call_id,
                tool_name=name,
                is_error=is_error,
                provider_payload={"details": details} if content_out is body else {},
            )
            pending.append((record.get("id"), message))
    flush()

    # Calls in each epoch AFTER the hit = times the duplicate would be re-sent.
    ends: dict[int, int] = {}
    running = 0
    for e in sorted(assistants_per_epoch):
        running += assistants_per_epoch[e]
        ends[e] = running
    for saved, at, e in hits:
        stats["weighted_chars"] += saved * max(0, ends[e] - at)
    for chars, at, e in landed:
        stats["read_weighted_chars"] += chars * max(0, ends.get(e, at) - at)
    stats["provider_calls"] = assistant_count
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--sessions", type=Path, default=Path.home() / ".local-operator" / "sessions"
    )
    parser.add_argument("--work", type=Path, required=True, help="scratch dir for clones")
    parser.add_argument("--limit", type=int, default=400, help="newest N transcripts with reads")
    args = parser.parse_args()
    sessions, work = args.sessions.resolve(), args.work.resolve()
    # Strictly outside the store: clones are full logical copies, and the
    # session scratchpad's byte cap counts them at that size (a 400-transcript
    # run is ~0.4 GB), so a session scratchpad is not a usable target either.
    if work == sessions or sessions in work.parents:
        parser.error("--work must be outside the sessions store")
    work.mkdir(parents=True, exist_ok=True)

    candidates = sorted(
        sessions.glob("*/transcript.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    chosen: list[Path] = []
    for src in candidates:
        if len(chosen) >= args.limit:
            break
        with src.open("rb") as handle:
            if not READ_ROW.search(handle.read()):
                continue
        dst = work / f"{src.parent.name}.jsonl"
        if not dst.exists():
            if sys.platform == "darwin":
                subprocess.run(["cp", "-c", str(src), str(dst)], check=True)
            else:
                shutil.copyfile(src, dst)
        chosen.append(dst)

    total: Counter = Counter()
    for path in chosen:
        total += replay(path)
    total["transcripts"] = len(chosen)

    def mb(chars: int) -> str:
        return f"{chars / 1e6:.2f} M chars"

    reads, read_chars = total["reads"], total["read_chars"]
    print(f"transcripts replayed:      {total['transcripts']} (newest with >=1 read)")
    print(f"provider calls:            {total['provider_calls']}")
    print(f"compactions seen:          {total['compactions']}")
    print(f"successful read results:   {reads} ({mb(read_chars)})")
    print(
        f"deduped reads:             {total['deduped']} "
        f"({total['deduped'] / max(reads, 1):.1%} of reads)"
    )
    print(
        f"chars saved at landing:    {mb(total['saved_chars'])} "
        f"({total['saved_chars'] / max(read_chars, 1):.1%} of read result chars)"
    )
    print(
        f"persistence-weighted:      {mb(total['weighted_chars'])} of "
        f"{mb(total['read_weighted_chars'])} "
        f"({total['weighted_chars'] / max(total['read_weighted_chars'], 1):.2%}; "
        "chars x later provider calls before compaction/end)"
    )
    print(
        f"raw=true bypassed:         {total['raw_bypassed']} reads "
        f"({mb(total['raw_bypassed_chars'])}) that would otherwise dedup"
    )
    print(f"ledger cap per session:    {MAX_ENTRIES} entries")
    print("fingerprint proxy:         no edit/write of the path and no mutating bash/eval between")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
