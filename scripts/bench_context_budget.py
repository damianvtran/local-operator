#!/usr/bin/env python3
"""Start-of-session context budget guard.

Enforces the performance contract in ``docs/REWRITE.md``: a fresh conversation
MUST start under a bounded context. Exits non-zero when the budget is blown,
so CI fails loudly rather than letting the start context drift upward one
harmless-looking paragraph at a time.

What it measures, and why that matters
--------------------------------------
This script used to measure a FICTION, and the fiction was optimistic in three
separate ways — which is worse than no guard, because a guard that cannot go
red is believed:

1. It built tools from a bare ``ToolContext``, so the createIf gates returned
   ``None`` for nine of the twenty-four default tools. It measured a surface
   no user has, and undercounted the single largest payload after the system
   prompt. It now builds the full surface via ``scripts/real_tool_surface``.
2. It passed no ``user_instructions`` and no ``repo_guidance``. Both ride the
   HEAD block of a real session and both are commonly kilobytes. They are now
   included, at representative sizes.
3. It counted only the SEMANTIC skills block plus schemas, dropping the
   instruction and inventory blocks from the total. All four blocks plus the
   tools array are now counted, because all five are on the wire.

Units
-----
CHARACTERS are the ground truth here. The analytics ledger apportions a call's
billed prompt tokens across components in proportion to each component's
character count, and the measured rate on this prompt surface is ~2.78
chars/billed-token — NOT cl100k's ~4.22. A projection made with a generic
tokenizer understates a saving by roughly 45%, so the budget is expressed in
billed tokens derived from characters at the measured rate.

Run:
    .venv/bin/python scripts/bench_context_budget.py
    .venv/bin/python scripts/bench_context_budget.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from local_operator.harness.types import AgentTool  # noqa: E402
from local_operator.prompts_api import build_system_blocks  # noqa: E402
from scripts.real_tool_surface import build_real_tools  # noqa: E402

#: Measured chars-per-billed-token for this prompt surface. See the module
#: docstring — this is the ledger's own apportioning rate, not a tokenizer's.
CHARS_PER_BILLED_TOKEN = 2.78

#: The enforced ceiling, in billed tokens.
#:
#: A RATCHET set to where the tree honestly stands today, not to the 30,000
#: aspiration in docs/REWRITE.md. Two things are going on and they pull in
#: opposite directions:
#:
#: - The contract's 30,000 was written against a measurement that omitted the
#:   instruction and inventory blocks, built two-thirds of the tool surface,
#:   and supplied neither custom instructions nor repo guidance. Measured
#:   honestly, ``origin/main`` before this change was ~31,700 billed tokens —
#:   i.e. the contract was ALREADY breached and the guard reported nothing.
#: - This change brings that to ~26,165 (measured, same script both sides).
#:
#: So the ceiling is set just above the current real figure. It is green by
#: MEASUREMENT rather than green by fiction, and it is the number a future
#: reduction tightens. The repo-guidance lazy-loading work lands separately
#: and should pull this down again; each context-reduction change is expected
#: to lower this line, and one that cannot has not reduced anything.
#:
#: Do NOT raise it to make a red run green without stating in the PR what
#: regressed — that converts a guard into a rubber stamp.
#:
#: Headroom is deliberately small (~2%): enough that an incidental wording
#: edit does not trip it, tight enough that a new paragraph or a new tool has
#: to be a decision rather than a surprise.
BUDGET_BILLED_TOKENS = 26_500

#: Measured on ``origin/main`` with THIS script, for comparison: 31,100 billed
#: tokens. Kept as a comment rather than asserted, because it is a historical
#: datum and re-measuring it needs the old tree.

#: Stand-ins for the operator-supplied text that rides the head block of a
#: real session. Fixed sizes, because a guard whose threshold moves with the
#: developer's own ~/.local-operator contents is not a guard — it would pass
#: on a machine with no custom instructions and fail on one with a long file,
#: for reasons having nothing to do with the diff. These lengths are typical
#: of a configured install (measured on this machine: ~5.7k chars of custom
#: instructions, ~8k of repo guidance after truncation).
_SAMPLE_USER_INSTRUCTIONS = "- A standing operator preference line.\n" * 150
_SAMPLE_REPO_GUIDANCE = "Project convention paragraph explaining a rule.\n" * 170


def tool_schema_chars(tools: list[AgentTool]) -> int:
    """Characters the provider tools array costs on every request.

    Mirrors what ``Session.context_breakdown`` counts: name, description and
    the JSON-serialized parameter schema, per tool.
    """
    return sum(
        len(tool.name) + len(tool.description or "") + len(json.dumps(tool.parameters or {}))
        for tool in tools
    )


def measure_start_context(
    *,
    user_instructions: str = _SAMPLE_USER_INSTRUCTIONS,
    repo_guidance: str = _SAMPLE_REPO_GUIDANCE,
) -> dict[str, int]:
    """Character cost of everything a fresh session puts on the wire."""
    tools = build_real_tools(str(REPO))
    blocks = build_system_blocks(
        tools,
        skills_block="<skills/>",
        env_details="cwd: /Users/example/project\nplatform: Darwin 25.6.0 (arm64)",
        date_str="2026-01-01",
        user_instructions=user_instructions,
        repo_guidance=repo_guidance,
    )
    parts = {
        "instructions": len(blocks[0]),
        "tool_inventory": len(blocks[1]),
        "environment": len(blocks[2]),
        "knowledge": len(blocks[3]),
        "tool_schemas": tool_schema_chars(tools),
    }
    parts["TOTAL"] = sum(parts.values())
    parts["n_tools"] = len(tools)
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budget",
        type=int,
        default=BUDGET_BILLED_TOKENS,
        help="ceiling in billed tokens (default: the enforced ratchet)",
    )
    parser.add_argument(
        "--inflate",
        type=int,
        default=0,
        help=(
            "append N characters of filler to the sample custom instructions. "
            "Exists to PROVE the guard can go red — a guard that cannot fail "
            "is worse than none (AGENTS.md, 'Prove the test can still fail')."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    parts = measure_start_context(
        user_instructions=_SAMPLE_USER_INSTRUCTIONS + ("x" * args.inflate)
    )
    total_chars = parts["TOTAL"]
    billed = round(total_chars / CHARS_PER_BILLED_TOKEN)

    if args.verbose or args.inflate:
        print(f"tool surface: {parts['n_tools']} tools")
        for key in ("instructions", "tool_inventory", "environment", "knowledge", "tool_schemas"):
            chars = parts[key]
            print(f"  {key:16} {chars:>8,} chars  ({chars / CHARS_PER_BILLED_TOKEN:>7,.0f} billed)")

    print(
        f"start context: {total_chars:,} chars = ~{billed:,} billed tokens "
        f"(at {CHARS_PER_BILLED_TOKEN} chars/token) vs budget {args.budget:,}"
    )
    if billed > args.budget:
        print(
            f"FAIL: start context exceeds the budget by {billed - args.budget:,} tokens.\n"
            "      Reduce the system prompt, the tool inventory or the tool schemas — "
            "or state in the PR what regressed before raising the budget."
        )
        return 1
    print(f"PASS ({args.budget - billed:,} billed tokens of headroom)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
