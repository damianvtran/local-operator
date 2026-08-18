#!/usr/bin/env python3
"""Session footprint benchmark: bytes on disk and tokens on resume.

``bench_context_budget.py`` measures the context a session STARTS with and
``bench_task_cost.py`` measures what a task costs to run. Neither looks at
what the session leaves behind. This one does, because that is the axis that
actually filled a developer's volume: the harness this project is compared
against accumulated 5.9 GB of unbounded session transcripts with single files
reaching 233 MB, and no cap on what enters the context does anything about it.

Four sections, all self-verifying (non-zero exit on a failed check):

1. **Encoding** — bytes per turn and per session as written today against the
   legacy encoder, recomputed entry by entry from the same messages, so the
   comparison is exact rather than a rerun against a different conversation.
2. **Replay** — prompt tokens a resumed session pushes at the provider, with
   and without the prune journal. This is the "fewer tokens per turn" half:
   without the journal a resume replays tool output the live session had
   already blanked, so resuming costs MORE than the session it resumed.
3. **Retention** — generates more sessions than the ceiling allows and proves
   the directory stays under budget with the live session intact.
4. **Projection** — what a heavy user accumulates per week, before and after.

Run:
    .venv/bin/python scripts/bench_session_footprint.py
    .venv/bin/python scripts/bench_session_footprint.py --transcript PATH
    .venv/bin/python scripts/bench_session_footprint.py --sessions-dir ~/.local-operator/sessions
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, cast

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from local_operator.compaction.api import prune_tool_outputs  # noqa: E402
from local_operator.compaction.tokens import (  # noqa: E402
    estimate_tokens,
    invalidate_message_cache,
)
from local_operator.harness.types import Message  # noqa: E402
from local_operator.paths import config_dir  # noqa: E402
from local_operator.session.retention import (  # noqa: E402
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_SESSIONS,
    sweep_sessions,
)
from local_operator.session.transcript import (  # noqa: E402
    CUSTOM_KIND_MESSAGE,
    ENTRY_MESSAGE,
    ENTRY_PRUNE,
    Transcript,
    _entry_to_message,
    encode_message_payload,
)

#: A heavy user's day, for the weekly projection. Deliberately aggressive:
#: 12 working sessions of 25 turns each, 6 days a week.
HEAVY_SESSIONS_PER_DAY = 12
HEAVY_TURNS_PER_SESSION = 25
HEAVY_DAYS_PER_WEEK = 6


class CheckFailed(Exception):
    """A benchmark assertion that did not hold."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def kb(value: float) -> str:
    return f"{value / 1024:,.1f} KB"


def mb(value: float) -> str:
    return f"{value / (1024 * 1024):,.1f} MB"


def _spill():
    """The spill module, or ``None`` when it is not present.

    Imported lazily and optionally so this benchmark keeps working against a
    checkout without it, rather than turning a missing sibling module into an
    ImportError at startup.
    """
    try:
        from local_operator.tools import spill
    except ImportError:
        return None
    return spill


def spill_dirname() -> str:
    spill = _spill()
    return str(spill.spill_dir()) if spill is not None else "(not present)"


def spill_limit() -> int:
    spill = _spill()
    return spill.SPILL_TOTAL_LIMIT_BYTES if spill is not None else 0


# --------------------------------------------------------------------------
# 1. Encoding
# --------------------------------------------------------------------------


def _row_bytes(entry: Any, payload: dict[str, Any]) -> int:
    line = json.dumps(
        {"id": entry.id, "ts": entry.ts, "type": entry.type, "payload": payload},
        separators=(",", ":"),
    )
    return len(line.encode("utf-8")) + 1  # + the newline


def encoded_sizes(entry: Any) -> tuple[int, int]:
    """``(legacy bytes, slim bytes)`` for one entry, from the same message.

    Both sides are recomputed from the rehydrated message rather than one
    being read off disk. That matters in both directions: a transcript
    written before this change is stored in the legacy form, one written
    after is stored in the slim form, and reading either from ``stat()``
    would report a 0% delta against itself. Recomputing makes the comparison
    exact and independent of which build produced the file.
    """
    message = _entry_to_message(entry) if entry.type == ENTRY_MESSAGE else None
    if message is None:
        size = _row_bytes(entry, entry.payload)
        return size, size
    kind = CUSTOM_KIND_MESSAGE if isinstance(message, Message) else "custom"
    legacy = _row_bytes(entry, {"kind": kind, **message.model_dump()})
    slim = _row_bytes(entry, {"kind": kind, **encode_message_payload(message)})
    return legacy, slim


@dataclass
class EncodingReport:
    path: Path
    entries: int
    turns: int
    new_bytes: int
    legacy_bytes: int
    has_raw_arguments: bool = False

    @property
    def saved(self) -> int:
        return self.legacy_bytes - self.new_bytes

    @property
    def pct(self) -> float:
        return 100.0 * self.saved / self.legacy_bytes if self.legacy_bytes else 0.0


def measure_encoding(path: Path) -> EncodingReport:
    entries = Transcript(path.parent).entries()
    legacy_bytes = 0
    new_bytes = 0
    has_raw = False
    for entry in entries:
        legacy, slim = encoded_sizes(entry)
        legacy_bytes += legacy
        new_bytes += slim
        has_raw = has_raw or any(
            call.get("raw_arguments") for call in entry.payload.get("tool_calls") or ()
        )
    # A "turn" is one user prompt; assistant/tool rows belong to the turn that
    # provoked them. Per-turn bytes is the number that scales with usage.
    turns = sum(
        1
        for entry in entries
        if entry.type == ENTRY_MESSAGE and entry.payload.get("role") == "user"
    )
    return EncodingReport(
        path=path,
        entries=len(entries),
        turns=max(turns, 1),
        new_bytes=new_bytes,
        legacy_bytes=legacy_bytes,
        has_raw_arguments=has_raw,
    )


# --------------------------------------------------------------------------
# 2. Replay tokens
# --------------------------------------------------------------------------


@dataclass
class ReplayReport:
    messages: int
    tokens_naive: int
    tokens_journalled: int
    pruned: int
    reclaimable_bytes: int

    @property
    def saved(self) -> int:
        return self.tokens_naive - self.tokens_journalled

    @property
    def pct(self) -> float:
        return 100.0 * self.saved / self.tokens_naive if self.tokens_naive else 0.0


def _cold_tokens(messages: Iterable[Any]) -> int:
    """Prompt tokens with a COLD estimate cache.

    ``estimate_tokens`` memoizes on ``message.id`` and a replayed message
    keeps the id it was persisted under, so measuring the same conversation
    twice in one process would report the first measurement twice and this
    whole section would read 0.0%. A real resume is a fresh process.
    """
    total = 0
    for message in messages:
        invalidate_message_cache(message)
        total += estimate_tokens(message)
    return total


async def measure_replay(path: Path) -> ReplayReport:
    """Prompt tokens a resume pushes, with and without the prune journal.

    ``path`` must be a scratch copy: the journal rows are stripped from it
    first, which reconstructs exactly the file a pre-change build would have
    left behind, and the pass that re-derives them is the same one the
    running session applies after every turn. Measuring the file as found
    would compare the journalled transcript against itself and report zero.
    """
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and f'"type":"{ENTRY_PRUNE}"' not in line
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    transcript = Transcript(path.parent)
    history = transcript.build_llm_history()
    naive = _cold_tokens(history)

    # Force the idle branch so the pass sweeps the whole history rather than
    # only the region outside the warm cache suffix: a resumed session's cache
    # is cold by definition, so there is no warm suffix to protect.
    now_ms = int(time.time() * 1000)
    prune_tool_outputs(cast("list[Message]", history), now_ms, last_activity_ms=0)
    pruned = [
        message
        for message in history
        if isinstance(message, Message) and (message.provider_payload or {}).get("pruned")
    ]
    for message in pruned:
        await transcript.append_prune(message.id, message.text)

    replayed = Transcript(path.parent).build_llm_history()
    journalled = _cold_tokens(replayed)
    return ReplayReport(
        messages=len(history),
        tokens_naive=naive,
        tokens_journalled=journalled,
        pruned=len(pruned),
        reclaimable_bytes=transcript.reclaimable_bytes(),
    )


# --------------------------------------------------------------------------
# 3. Retention
# --------------------------------------------------------------------------


@dataclass
class RetentionReport:
    generated: int
    ceiling: int
    remaining: int
    peak_bytes: int
    live_intact: bool


def measure_retention(generated: int, ceiling: int, session_bytes: int) -> RetentionReport:
    """Exceed the ceiling on purpose and watch the directory stay bounded."""
    root = Path(tempfile.mkdtemp(prefix="lo-retention-"))
    try:
        sessions = root / "sessions"
        sessions.mkdir()
        live = sessions / "live-session"
        live.mkdir()
        (live / "transcript.jsonl").write_text("live" * (session_bytes // 4))

        peak = 0
        for i in range(generated):
            directory = sessions / f"s{i:04d}"
            directory.mkdir()
            (directory / "transcript.jsonl").write_text("x" * session_bytes)
            result = sweep_sessions(
                sessions,
                live_dir=live,
                max_sessions=ceiling,
                max_bytes=0,
                max_age_days=0,
            )
            # ``bytes_on_disk``, not ``bytes_remaining``: live sessions are
            # exempt from the ceilings and their bytes are reported separately,
            # so the narrower figure would under-read the store by exactly the
            # live session this benchmark deliberately keeps open — and the
            # whole point here is to measure the real footprint.
            peak = max(peak, result.bytes_on_disk)
            check(
                len(list(sessions.iterdir())) <= ceiling + 1,
                f"sessions/ held {len(list(sessions.iterdir()))} dirs over a ceiling of {ceiling}",
            )
        live_intact = live.exists() and (live / "transcript.jsonl").exists()
        return RetentionReport(
            generated=generated,
            ceiling=ceiling,
            remaining=len(list(sessions.iterdir())),
            peak_bytes=peak,
            live_intact=live_intact,
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def find_transcripts(roots: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if root.is_file():
            found.append(root)
        elif root.is_dir():
            found.extend(sorted(root.glob("*/transcript.jsonl")))
    return [path for path in found if path.stat().st_size > 0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transcript",
        type=Path,
        action="append",
        default=None,
        help="a transcript.jsonl to measure (repeatable); default: the largest under the "
        "config dir's sessions/ and agents/",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="measure every transcript found, not just the largest",
    )
    parser.add_argument("--sessions-dir", type=Path, default=None)
    parser.add_argument("--retention-sessions", type=int, default=400)
    parser.add_argument("--retention-ceiling", type=int, default=25)
    parser.add_argument("--retention-session-bytes", type=int, default=4096)
    args = parser.parse_args()

    if args.transcript:
        transcripts = find_transcripts(args.transcript)
    else:
        base = args.sessions_dir or config_dir()
        roots = [base] if args.sessions_dir else [base / "sessions", base / "agents"]
        candidates = find_transcripts(roots)
        candidates.sort(key=lambda path: path.stat().st_size, reverse=True)
        # The largest by default: the interesting question is what a LONG
        # session costs, and averaging a hundred three-turn transcripts in
        # with it answers a different one.
        transcripts = candidates if args.all else candidates[:1]

    if not transcripts:
        print("no non-empty transcripts found; run a session first or pass --transcript")
        return 0

    failures: list[str] = []

    print("=" * 78)
    print("1. ENCODING — bytes on disk per turn and per session")
    print("=" * 78)
    totals = EncodingReport(Path("."), 0, 0, 0, 0)
    for path in transcripts:
        report = measure_encoding(path)
        totals.entries += report.entries
        totals.turns += report.turns
        totals.new_bytes += report.new_bytes
        totals.legacy_bytes += report.legacy_bytes
        print(f"\n  {path}")
        print(f"    entries {report.entries}, user turns {report.turns}")
        print(
            f"    per session : {kb(report.legacy_bytes)} -> {kb(report.new_bytes)}"
            f"   ({report.pct:.1f}% smaller)"
        )
        print(
            f"    per turn    : {kb(report.legacy_bytes / report.turns)} -> "
            f"{kb(report.new_bytes / report.turns)}"
        )
        if report.entries and not report.has_raw_arguments:
            # ``raw_arguments`` is dropped at WRITE time and is not
            # recoverable from the row, so a transcript this build already
            # wrote cannot show what dropping it saved. Point the reader at a
            # file written by an older build rather than quietly under-report.
            print(
                "    note: no raw_arguments on this file — it was written by the slim"
                " encoder, so the figure above excludes that component"
            )
        try:
            check(
                report.new_bytes <= report.legacy_bytes,
                f"{path}: slim encoding is larger than the legacy one",
            )
        except CheckFailed as exc:
            failures.append(str(exc))
    per_turn_before = totals.legacy_bytes / totals.turns
    per_turn_after = totals.new_bytes / totals.turns
    print(
        f"\n  TOTAL: {kb(totals.legacy_bytes)} -> {kb(totals.new_bytes)} "
        f"({totals.pct:.1f}% smaller), {kb(per_turn_before)} -> {kb(per_turn_after)} per turn"
    )

    print()
    print("=" * 78)
    print("2. REPLAY — prompt tokens a resumed session pushes")
    print("=" * 78)
    replay_saved_pct = 0.0
    for path in transcripts:
        scratch = Path(tempfile.mkdtemp(prefix="lo-replay-"))
        try:
            copy = scratch / "transcript.jsonl"
            copy.write_bytes(path.read_bytes())
            report = asyncio.run(measure_replay(copy))
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        print(f"\n  {path}")
        print(f"    replayed messages     : {report.messages}")
        print(f"    tool results blanked  : {report.pruned}")
        print(
            f"    prompt tokens on resume: {report.tokens_naive:,} -> "
            f"{report.tokens_journalled:,}  ({report.pct:.1f}% smaller)"
        )
        print(f"    disk reclaimable      : {kb(report.reclaimable_bytes)}")
        replay_saved_pct = max(replay_saved_pct, report.pct)
        try:
            check(
                report.tokens_journalled <= report.tokens_naive,
                f"{path}: journalled replay is more expensive than the naive one",
            )
        except CheckFailed as exc:
            failures.append(str(exc))

    print()
    print("=" * 78)
    print("3. RETENTION — exceeding the ceiling on purpose")
    print("=" * 78)
    try:
        retention = measure_retention(
            args.retention_sessions, args.retention_ceiling, args.retention_session_bytes
        )
        print(
            f"\n  generated {retention.generated} sessions against a ceiling of "
            f"{retention.ceiling}"
        )
        print(f"    directories left : {retention.remaining} (ceiling + the live session)")
        print(f"    peak bytes held  : {kb(retention.peak_bytes)}")
        print(f"    live session     : {'intact' if retention.live_intact else 'EVICTED'}")
        check(retention.live_intact, "the live session was evicted")
    except CheckFailed as exc:
        failures.append(str(exc))

    # The spill store is the OTHER bounded store on disk (large tool outputs
    # the tools package writes out of context). It self-evicts LRU under its
    # own hard ceiling and protects the live session's entries inside a grace
    # window, so the session sweeper deliberately does not touch it — a
    # second sweeper could evict a handle whose footer is still in the live
    # transcript. Reported here so the two ceilings add up to a stated total.
    print()
    print(f"  spill store          : {spill_dirname()} (swept by its own LRU, not by this)")
    print(f"    ceiling            : {mb(spill_limit())}")
    print(f"    TOTAL disk ceiling : {mb(DEFAULT_MAX_BYTES + spill_limit())}")

    print()
    print("=" * 78)
    print("4. PROJECTION — a heavy user's week")
    print("=" * 78)
    turns_per_week = HEAVY_SESSIONS_PER_DAY * HEAVY_TURNS_PER_SESSION * HEAVY_DAYS_PER_WEEK
    weekly_before = per_turn_before * turns_per_week
    weekly_after = per_turn_after * turns_per_week
    sessions_per_week = HEAVY_SESSIONS_PER_DAY * HEAVY_DAYS_PER_WEEK
    print(
        f"\n  {HEAVY_SESSIONS_PER_DAY} sessions/day x {HEAVY_TURNS_PER_SESSION} turns "
        f"x {HEAVY_DAYS_PER_WEEK} days = {turns_per_week:,} turns/week"
    )
    print(f"    written per week      : {mb(weekly_before)} -> {mb(weekly_after)}")
    print(f"    sessions per week     : {sessions_per_week}")
    print(f"    RETAINED, before      : unbounded — {mb(weekly_before)}/week, forever")
    # The count ceiling binds first for a normal user; the byte ceiling is the
    # backstop for a session that dumps far more than the measured average.
    by_count = DEFAULT_MAX_SESSIONS * per_turn_after * HEAVY_TURNS_PER_SESSION
    print(
        f"    RETAINED, after       : min(30 days, {DEFAULT_MAX_SESSIONS} sessions, "
        f"{mb(DEFAULT_MAX_BYTES)}) = {mb(min(by_count, DEFAULT_MAX_BYTES))} steady state"
    )

    print()
    if failures:
        print(f"FAILED {len(failures)} check(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
