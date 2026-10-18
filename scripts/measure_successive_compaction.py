"""Successive-compaction retention harness (BEFORE/AFTER measurement).

Reproduces the "cliff" deterministically: N successive snapcompact passes, each
folding the previous archive text in as ``previous_text`` and adding a fresh
serialized chunk. Measures:

  * ``truncated_chars`` across passes (the smoking gun: it climbs today),
  * how much of each pass's DISTINCTIVE material is still recoverable from the
    final archive's *replayed payload* (text edges + middle text pages +
    ``carry`` when present) — i.e. what the model can still read,
  * the replayed token estimate (must stay under the operator's ~100k bar).

Material "recoverable" means: the exact distinctive marker string is present in
the replayed payload. That is the honest test of retention — a marker that is
only inside a dropped page or inside a frame the replay foveated away is NOT
recoverable.

Run: python scripts/measure_successive_compaction.py --passes 12
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

sys.path.insert(0, ".")

from local_operator.compaction.snapcompact import (  # noqa: E402
    Archive,
    compact_to_archive,
    estimate_archive_tokens,
)

PROVIDER = "anthropic"
MODEL = "claude-opus-4-8"
WINDOW = 400_000

#: Distinctive, never-colliding token per pass: if it is not in the replayed
#: payload the model cannot read that pass's goal/decision line any more.
MARKER_RE = re.compile(r"<<RET-\d+>>")
DECISION_RE = re.compile(r"DECISION-\d+")
ASK_RE = re.compile(r"<<ASK-\d+>>")


def _chunk(i: int, lines: int) -> str:
    """One pass's newly discarded history, in the shape the real serializer emits.

    Carries the three kinds the operator named as material: a goal/decision
    marker (an assistant decision), an operator ask (a user turn's substance),
    and bulky filler that is genuinely disposable.
    """
    out = [
        f"--- assistant ---\nGoal for step {i}: <<RET-{i}>> DECISION-{i}: keep the "
        f"existing helper, do not add a new one (pass {i}).",
        f"--- user ---\n<<ASK-{i}>> please make sure the manager posts concise "
        f"updates in thread {i}.",
    ]
    for j in range(lines):
        out.append(
            f"--- assistant ---\nstep {i}.{j}: ran `make check-changed`; " "output: " + ("ok " * 40)
        )
        out.append(f"--- tool ---\nresult {i}.{j}: " + ("stdout line with detail " * 30))
    return "\n".join(out)


def _replayed_payload(archive: Archive) -> str:
    """Everything a replay turns into MODEL-READABLE TEXT (edges + text pages + carry)."""
    from local_operator.compaction.snapcompact import _policy_pages

    parts: list[str] = []
    carry = getattr(archive, "carry", "") or ""
    if carry:
        parts.append(carry)
    pages = _policy_pages(archive)
    if pages is not None and len(pages) == len(archive.middle_kinds):
        parts.append("\n".join(p for p, k in zip(pages, archive.middle_kinds) if k == "t"))
    else:
        # Legacy/reluctant path: only the verbatim edges are readable text.
        pass
    parts.append(archive.text_head)
    parts.append(archive.text_tail)
    return "\n".join(parts)


def run(passes: int, lines: int, verbose: bool = True) -> dict[str, Any]:
    prev_text: str | None = None
    series: list[dict[str, Any]] = []
    for i in range(passes):
        serialized = _chunk(i, lines)
        archive = compact_to_archive(
            _messages(serialized),
            PROVIDER,
            MODEL,
            prev_text,
            context_window=WINDOW,
        )
        payload = _replayed_payload(archive)
        series.append(
            {
                "pass": i,
                "truncated_chars": archive.truncated_chars,
                "text_chars": len(archive.text),
                "payload_chars": len(payload),
                "payload_tokens": estimate_archive_tokens(archive),
                "carry_chars": len(getattr(archive, "carry", "") or ""),
                "material_recoverable": {
                    "goal": f"<<RET-{i}>>" in payload,
                    "decision": f"DECISION-{i}" in payload,
                    "ask": f"<<ASK-{i}>>" in payload,
                },
            }
        )
        prev_text = archive.text
    final = series[-1]
    retained = [p for p in series if all(p["material_recoverable"].values())]
    result = {
        "passes": passes,
        "truncated_chars_series": [s["truncated_chars"] for s in series],
        "truncated_monotonic_steps": sum(
            1
            for a, b in zip(
                [s["truncated_chars"] for s in series],
                [s["truncated_chars"] for s in series][1:],
            )
            if b > a
        ),
        "final_payload_tokens": final["payload_tokens"],
        "final_text_chars": final["text_chars"],
        "final_carry_chars": final["carry_chars"],
        "passes_fully_recoverable": [p["pass"] for p in retained],
        "count_fully_recoverable": len(retained),
        "series": series,
    }
    if verbose:
        print(json.dumps({k: v for k, v in result.items() if k != "series"}, indent=2))
        for s in series:
            m = s["material_recoverable"]
            print(
                f"  pass {s['pass']:>2}: trunc={s['truncated_chars']:>9} "
                f"text={s['text_chars']:>7} carry={s['carry_chars']:>6} "
                f"payload_tok={s['payload_tokens']:>7} "
                f"goal={'Y' if m['goal'] else 'n'} "
                f"decision={'Y' if m['decision'] else 'n'} "
                f"ask={'Y' if m['ask'] else 'n'}"
            )
    return result


def _messages(serialized: str):
    """Turn a serialized block into the minimal Message list ``compact_to_archive`` needs."""
    from local_operator.harness.types import Message, TextContent

    msgs = []
    for role_block in serialized.split("--- ")[1:]:
        header, _, body = role_block.partition(" ---\n")
        role = header.strip()
        if role not in ("user", "assistant", "tool"):
            role = "assistant"
        msgs.append(Message(role=role, content=[TextContent(text=body)]))
    return msgs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=12)
    ap.add_argument("--lines", type=int, default=400)
    args = ap.parse_args()
    run(args.passes, args.lines)
