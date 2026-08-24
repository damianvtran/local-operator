#!/usr/bin/env python3
"""Cumulative context-replay benchmark for mid-turn compaction.

A model request replays the conversation accumulated so far. Measuring only the
largest request therefore misses the bill users actually pay: a long tool run
replays earlier tool results again at every subsequent boundary. This benchmark
builds a deterministic synthetic run of independently paired tool calls and
measures the cumulative estimated tokens sent to the model.

The ``before`` arm preserves the pre-fix forward-only cut selection locally so
its baseline cannot drift as production code changes. The ``after`` arm calls
the real :func:`find_cut_point`. Both arms use the same messages, threshold,
summary size, and token estimator, and every accepted partition is checked for
tool call/result integrity.

Run:
    .venv/bin/python scripts/bench_compaction_replay.py
    .venv/bin/python scripts/bench_compaction_replay.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from local_operator.compaction.cutpoint import (  # noqa: E402
    _message_tokens,
    find_cut_point,
)
from local_operator.harness.types import (  # noqa: E402
    AgentMessage,
    CustomMessage,
    Message,
    ToolCall,
)

CutSelector = Callable[[Sequence[AgentMessage], int], int | None]


@dataclass(frozen=True)
class Result:
    """Replay footprint and compaction outcomes for one selector."""

    cumulative_replay_tokens: int
    peak_context_tokens: int
    final_context_tokens: int
    compactions: int
    starved_boundaries: int


def _tokens(messages: Sequence[AgentMessage]) -> int:
    """Use the production estimator for the exact history sent to a model."""
    return sum(_message_tokens(message) for message in messages)


def _legacy_find_cut_point(messages: Sequence[AgentMessage], keep_recent_tokens: int) -> int | None:
    """Pre-fix selector: forward-only snap and no tool-calling boundary.

    This intentionally contains only the behavior relevant to the incident.
    Keeping it in the benchmark makes ``before`` reproducible after the Git
    commit that originally exhibited the bug is no longer easy to check out.
    """
    if not messages or keep_recent_tokens <= 0:
        return None

    accumulated = 0
    index = len(messages)
    while index > 0 and accumulated < keep_recent_tokens:
        index -= 1
        accumulated += _tokens([messages[index]])
    if index == 0 and accumulated < keep_recent_tokens:
        return None

    while index < len(messages):
        message = messages[index]
        valid = (
            isinstance(message, CustomMessage) and message.custom_type == "compaction_summary"
        ) or (
            isinstance(message, Message)
            and (message.role == "user" or (message.role == "assistant" and not message.tool_calls))
        )
        if valid:
            break
        index += 1
    if index >= len(messages) or index <= 1:
        return None
    return index


def _assert_pair_integrity(messages: Sequence[AgentMessage], cut: int) -> None:
    """Reject a saving obtained by handing a provider an orphaned pair."""
    summarized_calls = {
        call.id
        for message in messages[:cut]
        if isinstance(message, Message)
        for call in message.tool_calls
    }
    kept_results = {
        message.tool_call_id
        for message in messages[cut:]
        if isinstance(message, Message) and message.role == "tool" and message.tool_call_id
    }
    kept_calls = {
        call.id
        for message in messages[cut:]
        if isinstance(message, Message)
        for call in message.tool_calls
    }
    kept_result_ids = {
        message.tool_call_id
        for message in messages[cut:]
        if isinstance(message, Message) and message.role == "tool" and message.tool_call_id
    }
    assert not summarized_calls & kept_results
    assert kept_calls <= kept_result_ids


def _run(
    selector: CutSelector,
    *,
    calls: int,
    output_chars: int,
    threshold_tokens: int,
    keep_recent_tokens: int,
    summary_repetitions: int,
) -> Result:
    """Replay one run, compacting at each completed tool boundary."""
    messages: list[AgentMessage] = [
        Message.user("Investigate the failure and keep working until it is resolved. " * 20),
        Message.assistant("I will inspect the system and follow the evidence."),
        Message.user("Preserve exact tool evidence while you work. " * 20),
    ]
    cumulative = _tokens(messages)
    peak = cumulative
    compactions = 0
    starved = 0

    # Repeated prose is deliberate: it approximates bounded command output and
    # tokenizes consistently across machines without reading user transcripts.
    payload = "worker output: operation completed with diagnostic detail\n" * 200
    payload = (payload * ((output_chars // len(payload)) + 1))[:output_chars]

    for number in range(calls):
        call_id = f"call-{number}"
        assistant = Message.assistant(f"Running diagnostic step {number}.")
        assistant.tool_calls = [
            ToolCall(id=call_id, name="bash", arguments={"command": f"step {number}"})
        ]
        messages.extend(
            [
                assistant,
                Message(
                    role="tool",
                    content=Message.user(payload).content,
                    tool_call_id=call_id,
                    tool_name="bash",
                ),
            ]
        )

        context_tokens = _tokens(messages)
        if context_tokens >= threshold_tokens:
            cut = selector(messages, keep_recent_tokens)
            if cut is None:
                starved += 1
            else:
                _assert_pair_integrity(messages, cut)
                messages = [
                    CustomMessage(
                        custom_type="compaction_summary",
                        details={
                            "summary": "Earlier diagnostics summarized. " * summary_repetitions
                        },
                    ),
                    *messages[cut:],
                ]
                compactions += 1
                context_tokens = _tokens(messages)

        # The next inference replays the post-boundary history. This is the
        # cumulative cost that one-shot context-size measurements omit.
        cumulative += context_tokens
        peak = max(peak, context_tokens)

    return Result(
        cumulative_replay_tokens=cumulative,
        peak_context_tokens=peak,
        final_context_tokens=_tokens(messages),
        compactions=compactions,
        starved_boundaries=starved,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--output-chars", type=int, default=8_000)
    parser.add_argument("--threshold-tokens", type=int, default=30_000)
    parser.add_argument("--keep-recent-tokens", type=int, default=8_000)
    parser.add_argument("--summary-repetitions", type=int, default=12)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    common = dict(
        calls=args.calls,
        output_chars=args.output_chars,
        threshold_tokens=args.threshold_tokens,
        keep_recent_tokens=args.keep_recent_tokens,
        summary_repetitions=args.summary_repetitions,
    )
    before = _run(_legacy_find_cut_point, **common)
    after = _run(find_cut_point, **common)
    reduction = 1 - (after.cumulative_replay_tokens / before.cumulative_replay_tokens)
    report = {
        "workload": common,
        "before": asdict(before),
        "after": asdict(after),
        "cumulative_replay_reduction_percent": round(reduction * 100, 2),
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"workload: {args.calls} tool calls x {args.output_chars:,} output chars")
        print(f"before cumulative replay: {before.cumulative_replay_tokens:,} tokens")
        print(f"after cumulative replay:  {after.cumulative_replay_tokens:,} tokens")
        print(f"reduction:                {reduction:.1%}")
        peaks = f"{before.peak_context_tokens:,} / {after.peak_context_tokens:,}"
        print(f"before/after peak:        {peaks}")
        print(f"before starved boundaries:{before.starved_boundaries:>10}")
        print(f"after compactions:        {after.compactions:>10}")

    passed = (
        args.calls > 0
        and args.output_chars > 0
        and args.threshold_tokens > 0
        and args.keep_recent_tokens > 0
        and args.summary_repetitions > 0
        and reduction > 0
        and before.starved_boundaries > 0
        and after.compactions > 0
        and after.starved_boundaries == 0
        and after.peak_context_tokens < before.peak_context_tokens
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
