#!/usr/bin/env python3
"""Offline replay: what a compaction advisor would actually have done.

The compaction advisor (BETA, ``values.compaction.advisor_enabled``) pulls the
trigger EARLIER than the configured threshold when the model judges the
session to be at a task boundary. The obvious way to evaluate that is "how
many tokens did it save", and it is the WRONG headline, because the cheapest
way to save tokens is to compact constantly and the cost of doing so is
invisible in a token count.

The metric that matters is **severance**: the fraction of compaction passes
whose cut point lands INSIDE the task the agent is currently executing, so the
summary paraphrases away the first half of work still in progress. On the real
session replayed here, five of seven actual passes severed a live task,
because ``find_cut_point`` keeps ``keep_recent_tokens`` (20k) of RECENCY while
the active-task span at those passes was 0.3k / 46.9k / 48.8k / 30.0k / 19.8k
/ 123.4k / 49.1k tokens (p50 32k, p90 99k).

So this script replays a real transcript at several trigger sizes under two
preserve rules:

- ``recency``    — today's rule: keep ``keep_recent_tokens``.
- ``task-aware`` — the shipped STEP 0 change: keep
  ``max(keep_recent_tokens, task_boundary_floor(...))``.

and reports passes, replay cost, and severance rate for each. Task-awareness
costs a couple of points of raw saving and buys back most of the severance;
judge it against that trade, not against the token column alone.

A "genuine user turn" here is the transcript's ``Message(role="user")``, which
is what ``Session._run_compaction`` treats as genuine — injected wake/hub/todo
deliveries are ``custom`` records and never appear as user messages. Note that
23 of this session's 69 user turns are CONTINUATIONS ("Continue", "Quota is
back..."), which are genuine user turns and are NOT task boundaries; the
``--continuations`` column reports severance restricted to that subset, since
it is where a purely local rule is weakest and the advisor earns its place.

Parsing is deliberately defensive: single payload lines in a real transcript
reach several hundred kilobytes, records of four different types are
interleaved, and a benchmark that dies on one malformed line is a benchmark
nobody runs on real data.

Run:
    .venv/bin/python scripts/bench_advisor_replay.py \
        ~/.local-operator/sessions/<id>/transcript.jsonl
    .venv/bin/python scripts/bench_advisor_replay.py <path> --json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from local_operator.compaction.tokens import _encode_len  # noqa: E402

#: Trigger sizes replayed. 600k is the shipped default ceiling (and where four
#: of the seven real passes fired); 300k and 400k are the earlier triggers an
#: advisor would plausibly ask for.
DEFAULT_TRIGGERS = (300_000, 400_000, 600_000)

#: Matches the short resumptions that are genuine user turns but not task
#: boundaries. Deliberately narrow and anchored: a heuristic that swept in
#: real requests would understate the problem it exists to measure.
_CONTINUATION_RE = re.compile(
    r"^(continue|go on|keep going|carry on|proceed|resume|yes resume it|"
    r"i approve[, ]*please continue|quota is back)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Turn:
    """One replayed history entry, reduced to what the replay needs."""

    index: int
    role: str
    tokens: int
    is_user: bool
    is_continuation: bool


@dataclass(frozen=True)
class ArmResult:
    """Replay outcome for one (trigger, preserve rule) pair."""

    trigger_tokens: int
    rule: str
    passes: int
    severed: int
    severed_at_continuations: int
    continuation_passes: int
    cumulative_replay_tokens: int
    peak_context_tokens: int
    #: Active-task span (tokens since the last genuine user turn) measured AT
    #: each pass. This is the distribution ``keep_recent_tokens`` has to cover
    #: to avoid severing, and the reason 20k does not.
    task_spans: tuple[int, ...] = ()

    @property
    def severance_rate(self) -> float:
        return self.severed / self.passes if self.passes else 0.0

    @property
    def continuation_severance_rate(self) -> float:
        if not self.continuation_passes:
            return 0.0
        return self.severed_at_continuations / self.continuation_passes


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed JSONL records, skipping anything unreadable.

    Defensive by design: a single corrupt or truncated line in a 20 MB
    transcript must not cost the whole measurement.
    """
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(record, dict):
                yield record


def _payload_text(payload: dict[str, Any]) -> str:
    """Every billable string in one message payload, concatenated.

    Approximates what the wire carries: text blocks plus serialized tool-call
    arguments. Image blocks are ignored — this replay compares two cut rules
    against each other, and both see the same frames.
    """
    parts: list[str] = []
    content = payload.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
    elif isinstance(content, str):
        parts.append(content)
    calls = payload.get("tool_calls")
    if calls:
        try:
            parts.append(json.dumps(calls))
        except (TypeError, ValueError):
            pass
    return "\n".join(parts)


def load_turns(path: Path) -> list[Turn]:
    """The transcript reduced to a token-weighted turn sequence."""
    turns: list[Turn] = []
    for record in _iter_records(path):
        kind = record.get("type")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        if kind == "message":
            role = str(payload.get("role") or "?")
            text = _payload_text(payload)
            is_user = role == "user"
            turns.append(
                Turn(
                    index=len(turns),
                    role=role,
                    # Floor of 1: an empty message still occupies a wire slot,
                    # and a zero would let a run of them collapse the walk.
                    tokens=max(1, _encode_len(text)),
                    is_user=is_user,
                    is_continuation=bool(is_user and _CONTINUATION_RE.match(text.strip())),
                )
            )
        elif kind == "compaction":
            # A prior pass's marker replays as its summary text, which is what
            # the live session carries forward.
            turns.append(
                Turn(
                    index=len(turns),
                    role="compaction",
                    tokens=max(1, _encode_len(str(payload.get("summary") or ""))),
                    is_user=False,
                    is_continuation=False,
                )
            )
    return turns


def _last_user_before(turns: Sequence[Turn], end: int) -> int | None:
    """Index of the newest genuine user turn at or before ``end``."""
    for i in range(min(end, len(turns) - 1), -1, -1):
        if turns[i].is_user:
            return i
    return None


def _task_span(turns: Sequence[Turn], end: int, cap: int) -> int:
    """Tokens from the last genuine user turn through ``end``, capped.

    The offline twin of ``cutpoint.task_boundary_floor``; kept local so the
    benchmark's baseline cannot silently move when production code changes,
    the same discipline ``bench_compaction_replay.py`` applies to its own
    ``before`` arm.
    """
    start = _last_user_before(turns, end)
    if start is None:
        return 0
    return min(sum(t.tokens for t in turns[start : end + 1]), cap)


def _cut_index(turns: Sequence[Turn], end: int, keep_tokens: int) -> int:
    """First KEPT index for a preserve budget of ``keep_tokens``.

    A simplified ``find_cut_point``: walk back accumulating tokens until the
    budget is met. The validity snap is omitted deliberately — it moves the
    cut by at most a few entries and would import production behaviour into
    what is meant to be an independent measurement.
    """
    accumulated = 0
    for i in range(end, -1, -1):
        accumulated += turns[i].tokens
        if accumulated >= keep_tokens:
            return i
    return 0


def replay(
    turns: Sequence[Turn],
    *,
    trigger_tokens: int,
    keep_recent_tokens: int,
    task_aware: bool,
    summary_tokens: int,
) -> ArmResult:
    """Replay the session under one trigger and one preserve rule.

    Severance is judged the way a user would: a pass SEVERS when its cut lands
    strictly after the start of the task in flight, i.e. when history the
    current request already produced is summarized away while the request is
    still being worked on.
    """
    context = 0
    cumulative = 0
    peak = 0
    passes = 0
    severed = 0
    continuation_passes = 0
    severed_at_continuations = 0
    spans: list[int] = []
    # Preserve window may not exceed half the trigger, mirroring
    # ``Session._advisor_floor_cap``: past that the pass has nothing left to
    # summarize and "protect the task" silently becomes "never compact".
    cap = max(keep_recent_tokens, trigger_tokens // 2)

    for i, turn in enumerate(turns):
        context += turn.tokens
        cumulative += context  # every turn replays the whole context so far
        peak = max(peak, context)
        if context <= trigger_tokens:
            continue

        keep = keep_recent_tokens
        if task_aware:
            keep = max(keep, _task_span(turns, i, cap))
        cut = _cut_index(turns, i, keep)
        task_start = _last_user_before(turns, i)

        passes += 1
        spans.append(_task_span(turns, i, 10**9))
        at_continuation = task_start is not None and turns[task_start].is_continuation
        if at_continuation:
            continuation_passes += 1
        if task_start is not None and cut > task_start:
            severed += 1
            if at_continuation:
                severed_at_continuations += 1

        # The pass replaces everything before the cut with a summary.
        context = summary_tokens + sum(t.tokens for t in turns[cut : i + 1])

    return ArmResult(
        trigger_tokens=trigger_tokens,
        rule="task-aware" if task_aware else "recency",
        passes=passes,
        severed=severed,
        severed_at_continuations=severed_at_continuations,
        continuation_passes=continuation_passes,
        cumulative_replay_tokens=cumulative,
        peak_context_tokens=peak,
        task_spans=tuple(spans),
    )


def _format_table(results: Sequence[ArmResult], baseline: int) -> str:
    header = (
        f"{'trigger':>9}  {'rule':<11}{'passes':>7}{'severed':>9}"
        f"{'sever%':>8}{'cont-sever%':>13}{'replay Mtok':>13}{'vs 600k':>9}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        delta = (1 - r.cumulative_replay_tokens / baseline) * 100 if baseline else 0.0
        lines.append(
            f"{r.trigger_tokens:>9,}  {r.rule:<11}{r.passes:>7}{r.severed:>9}"
            f"{r.severance_rate * 100:>7.0f}%{r.continuation_severance_rate * 100:>12.0f}%"
            f"{r.cumulative_replay_tokens / 1e6:>13,.0f}{delta:>8.1f}%"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("transcript", type=Path, help="path to a transcript.jsonl")
    parser.add_argument("--keep-recent-tokens", type=int, default=20_000)
    parser.add_argument(
        "--summary-tokens",
        type=int,
        default=6_000,
        help="size a compaction summary collapses to; measured summaries here run 4-8k",
    )
    parser.add_argument("--triggers", type=int, nargs="+", default=list(DEFAULT_TRIGGERS))
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    args = parser.parse_args(argv)

    if not args.transcript.exists():
        print(f"no such transcript: {args.transcript}", file=sys.stderr)
        return 2

    turns = load_turns(args.transcript)
    if not turns:
        print("transcript contained no replayable messages", file=sys.stderr)
        return 2

    user_turns = [t for t in turns if t.is_user]
    continuations = [t for t in user_turns if t.is_continuation]

    results: list[ArmResult] = []
    for trigger in sorted(args.triggers):
        for task_aware in (False, True):
            results.append(
                replay(
                    turns,
                    trigger_tokens=trigger,
                    keep_recent_tokens=args.keep_recent_tokens,
                    task_aware=task_aware,
                    summary_tokens=args.summary_tokens,
                )
            )

    baseline = next(
        (r.cumulative_replay_tokens for r in results if r.trigger_tokens == max(args.triggers)),
        0,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "entries": len(turns),
                    "user_turns": len(user_turns),
                    "continuation_turns": len(continuations),
                    "results": [asdict(r) for r in results],
                },
                indent=2,
            )
        )
        return 0

    print(f"transcript: {args.transcript}")
    print(f"entries: {len(turns):,}   user turns: {len(user_turns)}", end="")
    print(f"   continuations: {len(continuations)}")
    # Spans at the SHIPPED trigger under today's rule: the distribution
    # keep_recent_tokens would have to cover to stop severing tasks.
    shipped = next(
        (r for r in results if r.rule == "recency" and r.trigger_tokens == max(args.triggers)),
        None,
    )
    if shipped is not None and shipped.task_spans:
        spans = sorted(shipped.task_spans)
        print(
            f"active-task span at each pass ({shipped.trigger_tokens:,} recency): "
            + ", ".join(f"{s / 1000:,.1f}k" for s in shipped.task_spans)
        )
        print(
            f"  p50 {statistics.median(spans) / 1000:,.1f}k   "
            f"max {max(spans) / 1000:,.1f}k   "
            f"vs keep_recent_tokens {args.keep_recent_tokens / 1000:,.0f}k"
        )
    print()
    print(_format_table(results, baseline))
    print()
    print(
        "severance = passes whose cut lands inside the task in flight.\n"
        "cont-sever% = the same, restricted to passes during a CONTINUATION turn,\n"
        "which is where a purely local rule is weakest."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
