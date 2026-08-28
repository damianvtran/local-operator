#!/usr/bin/env python3
"""Collect REAL compaction-advisor hints against a recorded session.

The advisor's whole justification is a discrimination a token counter cannot
make: a short continuation turn ("Continue", "Quota is back, keep going") is a
genuine user turn but NOT a task boundary, so a purely local "preserve
everything since the last user turn" rule anchors to the wrong place. On the
session that motivated the feature, 23 of 69 user turns were continuations —
a third of the sample — so that subset is both the reason the model call earns
its keep and the likeliest place for it to fail.

This script replays a real transcript, and at each point where compaction
would have been considered it builds the ACTUAL advisor prompt
(:func:`local_operator.compaction.advisor.build_advisor_prompt`), makes a real
provider call, and validates the answer through the SHIPPED validator. It then
reports:

- overall: calls, hints accepted, hints rejected and why;
- on CONTINUATION probes specifically: how often the advisor anchored BEFORE
  the continuation turn (correct: the task started at the request being
  resumed) versus AT it (the local rule's failure mode, reproduced).

Rejections are counted, not hidden. A validator that rejects most answers is a
finding about the prompt, and this script exists to surface it rather than to
produce a flattering acceptance rate.

Costs real money: each probe sends a slice of the conversation. ``--limit``
bounds it and defaults low.

Run:
    .venv/bin/python scripts/collect_advisor_hints.py \
        ~/.local-operator/sessions/<id>/transcript.jsonl --limit 6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from local_operator.compaction.advisor import (  # noqa: E402
    build_advisor_prompt,
    parse_hint,
    validate_hint,
)
from local_operator.compaction.tokens import estimate_tokens  # noqa: E402
from local_operator.harness.types import (  # noqa: E402
    ChatRequest,
    Message,
    StreamTextDelta,
    TextContent,
)
from local_operator.model.configure import build_model_spec  # noqa: E402
from local_operator.providers.auth_store import AuthStore, default_db_path  # noqa: E402
from local_operator.providers.clients import client_for_spec  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from bench_advisor_replay import (  # noqa: E402
    _CONTINUATION_RE,
    _iter_records,
    _payload_text,
)

#: Probe every Nth user turn so the sample spans the session rather than
#: clustering at its start.
DEFAULT_STRIDE = 4

#: How much history each probe carries. A real advisor call sends the whole
#: live context; a replay cannot afford that per probe, and the discrimination
#: under test (which recent turn starts the task) is local to the tail.
PROBE_HISTORY_MESSAGES = 60

#: ``keep_recent_tokens`` as a FRACTION of the probe's history, not the
#: production constant.
#:
#: This matters or every probe is rejected for the wrong reason. The validator
#: refuses any hint whose preserve window is narrower than
#: ``max(keep_recent_tokens, task_boundary_floor)`` — the widen-only guard.
#: In production that compares a 20k floor against a ~500k context (4%). A
#: probe slice is ~23k tokens TOTAL, so passing the literal 20k asks the
#: advisor to preserve ~85% of what it was shown, which no anchor can satisfy:
#: the first run of this script rejected 10 of 10 probes that way, and every
#: one of the underlying answers was well-formed and sensible. Scaling keeps
#: the guard's SHAPE (a floor proportional to the context) while making the
#: probe measure the advisor rather than the slice size.
DEFAULT_KEEP_RECENT_FRACTION = 0.04


def load_messages(path: Path) -> list[Message]:
    """The transcript as harness ``Message`` objects, defensively parsed.

    Only text is reconstructed: the advisor reads the conversation's shape and
    subject matter, and tool-call plumbing would blow the probe budget without
    changing which turn starts the task.
    """
    messages: list[Message] = []
    for record in _iter_records(path):
        if record.get("type") != "message":
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        role = str(payload.get("role") or "")
        if role not in ("user", "assistant"):
            continue
        text = _payload_text(payload).strip()
        if not text:
            continue
        message = Message(
            role="user" if role == "user" else "assistant",
            content=[TextContent(text=text[:4000])],
        )
        # Reuse the transcript's own id so the advisor's anchor can be checked
        # against the record it actually refers to.
        record_id = record.get("id")
        if isinstance(record_id, str) and record_id:
            message.id = record_id
        messages.append(message)
    return messages


async def _ask(client, oauth, spec, history: Sequence[Message], prompt: str) -> str:
    request = ChatRequest(
        model=spec,
        system_blocks=["You are a coding agent working in a long session."],
        messages=[*history, Message.user(prompt)],
        tools=[],
        tool_choice="none",
        replayable=True,
    )
    parts: list[str] = []
    async for event in client.stream(request, None, oauth_access=oauth):
        if isinstance(event, StreamTextDelta):
            parts.append(event.delta)
    return "".join(parts)


async def main_async(args: argparse.Namespace) -> int:
    messages = load_messages(args.transcript)
    if not messages:
        print("transcript contained no usable messages", file=sys.stderr)
        return 2

    user_positions = [i for i, m in enumerate(messages) if m.role == "user"]
    continuations = {
        i for i in user_positions if _CONTINUATION_RE.match((messages[i].text or "").strip())
    }
    print(
        f"messages: {len(messages):,}  user turns: {len(user_positions)}  "
        f"continuations: {len(continuations)}"
    )

    # Probe points: bias toward continuations, since they are the subset the
    # feature is justified by. Each probe sits a few turns AFTER the user turn,
    # so there is in-flight work for the advisor to reason about.
    probes: list[int] = []
    for offset, position in enumerate(user_positions):
        if position in continuations or offset % args.stride == 0:
            target = min(position + 8, len(messages) - 1)
            if target > position:
                probes.append(target)
    probes = probes[: args.limit]

    spec = build_model_spec("anthropic", args.model)
    store = AuthStore(default_db_path())
    oauth = await store.get_oauth_access("anthropic")
    client = client_for_spec(spec)

    accepted = 0
    rejected = 0
    reasons: Counter[str] = Counter()
    cont_probes = 0
    cont_correct = 0
    rows: list[dict[str, Any]] = []

    for end in probes:
        start = max(0, end - PROBE_HISTORY_MESSAGES)
        history = messages[start : end + 1]
        # The task's true start: the newest user turn at or before the probe.
        task_start = next(
            (i for i in range(end, start - 1, -1) if messages[i].role == "user"), None
        )
        is_continuation = task_start in continuations if task_start is not None else False

        prompt = build_advisor_prompt(history, context_tokens=480_000, threshold_tokens=600_000)
        try:
            raw = await _ask(client, oauth, spec, history, prompt)
        except Exception as exc:  # noqa: BLE001 — a probe failure is data, not a crash
            reasons[f"provider error: {type(exc).__name__}"] += 1
            rejected += 1
            continue

        payload = parse_hint(raw)
        genuine = {m.id for m in history if m.role == "user"}
        # Floor scaled to the probe slice — see DEFAULT_KEEP_RECENT_FRACTION.
        slice_tokens = sum(estimate_tokens(m) for m in history)
        keep_recent = (
            args.keep_recent_tokens
            if args.keep_recent_tokens is not None
            else int(slice_tokens * DEFAULT_KEEP_RECENT_FRACTION)
        )
        hint = validate_hint(
            payload,
            history,
            genuine_user_ids=genuine,
            min_confidence=args.min_confidence,
            keep_recent_tokens=keep_recent,
            floor_cap=max(keep_recent, slice_tokens // 2),
        )
        if hint is None:
            rejected += 1
            reasons["unparseable" if payload is None else "failed validation"] += 1
            continue

        accepted += 1
        anchor = next((i for i, m in enumerate(history) if m.id == hint.preserve_from_id), None)
        absolute = start + anchor if anchor is not None else None
        if is_continuation:
            cont_probes += 1
            # CORRECT means the advisor anchored at or BEFORE the continuation
            # turn — i.e. it recognised that the task began at the request the
            # continuation resumes, not at the continuation itself. That is
            # precisely the judgement the local rule gets wrong.
            if absolute is not None and task_start is not None and absolute <= task_start:
                cont_correct += 1
        rows.append(
            {
                "probe_index": end,
                "continuation": is_continuation,
                "anchor_index": absolute,
                "task_start": task_start,
                "compact_now": hint.compact_now,
                "confidence": hint.confidence,
                "preserve_tokens": hint.preserve_tokens,
                "reason": hint.reason,
            }
        )

    print(f"\nprobes: {len(probes)}  accepted: {accepted}  rejected: {rejected}")
    for reason, count in reasons.most_common():
        print(f"  rejected — {reason}: {count}")
    if cont_probes:
        print(
            f"\nCONTINUATION accuracy: {cont_correct}/{cont_probes} "
            f"({cont_correct / cont_probes * 100:.0f}%) anchored at or before "
            "the resumed request"
        )
    else:
        print("\nno continuation probes ran")

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            tag = "cont" if row["continuation"] else "    "
            print(
                f"  {tag} probe={row['probe_index']:<6} anchor={row['anchor_index']} "
                f"task_start={row['task_start']} now={row['compact_now']} "
                f"conf={row['confidence']:.2f} {row['reason'][:60]}"
            )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("transcript", type=Path)
    parser.add_argument("--model", default="claude-opus-4-8")
    parser.add_argument("--limit", type=int, default=6, help="max probes (each costs a call)")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument(
        "--keep-recent-tokens",
        type=int,
        default=None,
        help="literal floor; default scales with the probe slice (see module docstring)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if not args.transcript.exists():
        print(f"no such transcript: {args.transcript}", file=sys.stderr)
        return 2
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
