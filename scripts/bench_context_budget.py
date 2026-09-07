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

1. It built tools from a bare ``ToolContext``, so the createIf gates dropped a
   third of the default tools. It measured a surface no user has, and
   undercounted the single largest payload after the system prompt. It now
   builds the full surface via ``scripts/real_tool_surface``, which owns the
   exact counts and the reason for each gate — one place, not two.
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

DO NOT PIPE IT. This script's entire value is its EXIT CODE — 0 pass, 1 budget
or surface violation, 2 provenance mismatch — and a pipeline reports the LAST
command's status, so ``bench_context_budget.py | tail`` prints a failure while
exiting 0. That is not hypothetical: a reviewer's piped run reported ``EXIT=0``
while the real code was 1, and only checking ``$?`` outside a pipeline caught
it. AGENTS.md documents the same rc-swallowing for the lint gates. Read the
status directly, or use ``${PIPESTATUS[0]}`` if you must page the output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import local_operator  # noqa: E402
from local_operator.harness.types import AgentTool  # noqa: E402
from local_operator.prompts_api import build_system_blocks  # noqa: E402
from local_operator.tools.registry import DEFAULT_TOOL_NAMES  # noqa: E402
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
#:
#: Measured on the full 24-tool surface (see ``real_tool_surface``); on
#: ``origin/main`` with this same script the figure was 31,100.
BUDGET_BILLED_TOKENS = 26_500

#: How much slack is allowed before the guard demands the ratchet be TIGHTENED.
#:
#: This is what stops the ceiling above from being aspirational. A comment
#: asking future agents to lower the budget enforces nothing; a failing build
#: that names the new number does. Set to roughly double the current headroom
#: so incidental wording edits stay quiet, while a real saving — the smallest
#: individual optimization in this PR was ~600 billed tokens — opens enough
#: slack to require the ratchet to follow it down.
#:
#: Suppressed under ``--inflate``, which deliberately measures a fiction.
TIGHTEN_WHEN_HEADROOM_EXCEEDS = 1_200

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
    inflate_schemas: int = 0,
) -> dict[str, Any]:
    """Character cost of everything a fresh session puts on the wire.

    ``inflate_schemas`` pads the TOOL SCHEMAS specifically. The fail-proof
    lever has to reach the largest component — the schemas are ~56% of the
    total and are what this PR is about — not only the operator text, or
    "prove it can fail" demonstrates the guard over the wrong half.
    """
    tools = build_real_tools(str(REPO))
    if inflate_schemas:
        # A new property on the first tool: the shape a real schema regression
        # takes (a tool grows an argument), rather than opaque filler.
        padded = dict(tools[0].parameters or {})
        props = dict(padded.get("properties") or {})
        props["_bench_filler"] = {"type": "string", "description": "x" * inflate_schemas}
        padded["properties"] = props
        tools = [tools[0].model_copy(update={"parameters": padded}), *tools[1:]]
    blocks = build_system_blocks(
        tools,
        skills_block="<skills/>",
        env_details="cwd: /Users/example/project\nplatform: Darwin 25.6.0 (arm64)",
        date_str="2026-01-01",
        user_instructions=user_instructions,
        repo_guidance=repo_guidance,
    )
    parts: dict[str, Any] = {
        "instructions": len(blocks[0]),
        "tool_inventory": len(blocks[1]),
        "environment": len(blocks[2]),
        "knowledge": len(blocks[3]),
        "tool_schemas": tool_schema_chars(tools),
    }
    parts["TOTAL"] = sum(parts.values())
    parts["n_tools"] = len(tools)
    # Names, so a short surface can say WHICH tool a host-dependent gate ate.
    parts["names"] = [t.name for t in tools]
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
    parser.add_argument(
        "--inflate-schemas",
        type=int,
        default=0,
        help=(
            "grow a TOOL SCHEMA by N characters. The schemas are the largest "
            "component, so the fail-proof lever must reach them too."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # A measurement tool that can silently measure the WRONG TREE is the same
    # defect class as one that measures the wrong surface. `sys.path.insert(0,
    # REPO)` above does not win against a cwd that already holds a
    # `local_operator/` package (under `python -c`, cwd precedes PYTHONPATH),
    # so a reviewer comparing two checkouts can get two numbers from one tree
    # and no error. Assert what we actually imported, and say it out loud.
    resolved = Path(local_operator.__file__).resolve().parent
    expected = (REPO / "local_operator").resolve()
    if resolved != expected:
        print(
            f"PROVENANCE MISMATCH: measuring {resolved}\n"
            f"                     expected  {expected}\n"
            "      Refusing to report a figure for a tree this script does not live in."
        )
        return 2
    print(f"measuring: {resolved}")

    parts = measure_start_context(
        user_instructions=_SAMPLE_USER_INSTRUCTIONS + ("x" * args.inflate),
        inflate_schemas=args.inflate_schemas,
    )
    total_chars = parts["TOTAL"]
    billed = round(total_chars / CHARS_PER_BILLED_TOKEN)

    # The surface is stated ALWAYS, not only under --verbose: a silent drop
    # from the full tool set is exactly how this guard would go back to
    # measuring a machine no user has. See real_tool_surface._forced_browser_backend.
    n_tools = int(parts["n_tools"])
    parts_names = list(parts["names"])
    expected_tools = len(DEFAULT_TOOL_NAMES)
    print(f"tool surface: {n_tools}/{expected_tools} tools")
    if n_tools != expected_tools:
        # Name the tools, not just the count: the whole point of this check is
        # that a host-dependent gate dropped something, and "which one" is the
        # first question anyone reading a CI log will ask.
        missing = sorted(set(DEFAULT_TOOL_NAMES) - set(parts_names))
        print(
            f"FAIL: measured {n_tools} of {expected_tools} default tools.\n"
            f"      missing: {', '.join(missing) or '(none — duplicate names?)'}\n"
            "      The benchmark must measure the FULL surface on every host, or it "
            "reports headroom a real session does not have.\n"
            "      If a tool was deliberately removed, update DEFAULT_TOOL_NAMES; "
            "otherwise fix the gate that dropped it."
        )
        return 1

    if args.verbose or args.inflate or args.inflate_schemas:
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

    # SELF-TIGHTENING. Without this the ratchet is aspirational: a comment
    # saying "each reduction should lower this line" enforces nothing, so the
    # likeliest outcome is that today's number becomes the permanent ceiling
    # and the docs/REWRITE.md target is never revisited. Slack beyond the band
    # is therefore also a failure — a loud, trivially-fixed one that hands the
    # next agent the exact number to write down.
    #
    # It is a BAND rather than a hard equality so ordinary wording edits do not
    # trip it; only a real reduction (or a deliberate tool removal) opens
    # enough slack to require the ratchet to follow it down.
    headroom = args.budget - billed
    inflating = bool(args.inflate or args.inflate_schemas)
    if not inflating and headroom > TIGHTEN_WHEN_HEADROOM_EXCEEDS:
        print(
            f"FAIL: {headroom:,} billed tokens of headroom exceeds the "
            f"{TIGHTEN_WHEN_HEADROOM_EXCEEDS:,}-token slack band.\n"
            "      The context got smaller — good. Tighten the ratchet so the saving "
            "is defended:\n"
            f"      set BUDGET_BILLED_TOKENS = {billed + TIGHTEN_WHEN_HEADROOM_EXCEEDS // 2:,} "
            "in scripts/bench_context_budget.py."
        )
        return 1
    print(f"PASS ({headroom:,} billed tokens of headroom)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
