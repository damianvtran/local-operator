#!/usr/bin/env python3
"""Context-budget benchmark for the harness rewrite.

Measures the token cost of a fresh conversation's system context against a
skill corpus (default: the user's omp skills at ~/.omp/agent/skills), per the
performance contract in docs/REWRITE.md: a new conversation MUST start at
<= 30,000 context tokens.

Two strategies are compared:
  - static  : omp-style full listing of every non-hidden skill description
  - semantic: this rewrite's per-turn vector selection (top-k above threshold)

The semantic number is the one the contract binds. Exit code 1 when the
semantic start context exceeds the budget so CI can fail loudly.

Run: .venv/bin/python scripts/bench_context_budget.py [--skills-dir PATH]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from local_operator.compaction.tokens import estimate_tokens
from local_operator.harness.types import Message, ToolContext
from local_operator.prompts_api import build_system_blocks
from local_operator.skills.api import (
    SkillIndex,
    default_backend_from_env,
    default_skill_roots,
    discover_skills,
    render_block,
)
from local_operator.tools.registry import create_tools

BUDGET_TOKENS = 30_000
SAMPLE_QUERIES = [
    "review my merge request and check the CI pipeline",
    "how do I deploy this service to qa",
    "look up the customer's search usage last month",
    "draft a slack reply to the support thread",
    "plan the rewrite of the auth module",
]


def _env_details() -> str:
    import os
    import platform

    return f"cwd: {os.getcwd()}\nplatform: {platform.system()} {platform.release()}"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skills-dir", type=Path, default=None)
    parser.add_argument("--budget", type=int, default=BUDGET_TOKENS)
    args = parser.parse_args()

    roots = [args.skills_dir] if args.skills_dir else default_skill_roots(Path.cwd())
    skills, warnings = discover_skills(roots)
    if not skills:
        print(f"no skills found under {roots}; nothing to benchmark")
        return 0
    print(f"discovered {len(skills)} skills from {[str(r) for r in roots]}")
    for w in warnings:
        print(f"  warning: {w}")

    backend = default_backend_from_env(lambda _k: None, None)
    index = SkillIndex(skills, backend)
    t0 = time.time()
    await index.build()
    print(f"index built in {time.time() - t0:.2f}s (backend={type(backend).__name__})")

    tools = create_tools(ToolContext(cwd=str(Path.cwd())))
    date_str = time.strftime("%Y-%m-%d")
    env = _env_details()

    def context_tokens(skills_block: str) -> int:
        blocks = build_system_blocks(tools, skills_block, env, date_str)
        return sum(estimate_tokens(Message.user(b)) for b in blocks)

    # Static listing (omp behavior) — the number semantic selection must beat.
    visible = [s for s in skills if not s.hide]
    static_block = render_block(visible)
    static_tokens = context_tokens(static_block)

    # Semantic selection per sample query; report worst case.
    worst = 0
    worst_q = ""
    selected_counts = []
    for q in SAMPLE_QUERIES:
        picked = await index.select(q)
        selected_counts.append(len(picked))
        tok = context_tokens(render_block(picked))
        if tok > worst:
            worst, worst_q = tok, q
    print(f"static listing context : {static_tokens:>6} tokens ({len(visible)} skills)")
    print(
        f"semantic worst case    : {worst:>6} tokens "
        f"(selected {max(selected_counts)} of {len(skills)} skills; query: {worst_q!r})"
    )

    # Tool schemas ride on every request too; count them once.
    schema_tokens = estimate_tokens(
        Message.user(json.dumps([{"name": t.name, "parameters": t.parameters} for t in tools]))
    )
    print(f"tool schema payload    : {schema_tokens:>6} tokens")

    start_context = worst + schema_tokens
    print(f"TOTAL start context    : {start_context:>6} tokens vs budget {args.budget}")
    if start_context > args.budget:
        print("FAIL: start context exceeds the budget — optimize the system prompt")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
