#!/usr/bin/env python3
"""Benchmark cost, tokens and base harness overhead on complex tasks.

Runs the real agent (exec mode) against several non-trivial multi-step tasks
and reports, per task: wall time, prompt/completion/cache tokens, estimated
cost (from the model registry pricing), and the peak RSS deltas. It also
measures the BASE overhead — process size right after constructing a session
but before any turn runs — so you can see that almost all compute is the
agent's workload, not the harness.

Usage:
    OPENROUTER_API_KEY=... python scripts/bench_complex_tasks.py

Captures land under /tmp/lo-bench/<slug>.jsonl (the same event vocabulary the
TUI consumes). Requires a live provider key; the model defaults to a cheap
long-context OpenRouter model.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Allow running from the repo root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path("/tmp/lo-bench")
MODEL = os.environ.get("LO_BENCH_MODEL", "deepseek/deepseek-v4-flash-0731")
HOSTING = "openrouter"

# Complex, multi-step tasks that force several tool calls + file writes and,
# ideally, a reasoning pass. The point is workload, not trivia.
TASKS: list[tuple[str, str]] = [
    (
        "kvstore",
        "Create a Python module src/kvstore.py implementing an in-memory "
        "key-value store with get/set/delete/atomic compare-and-set, plus "
        "tests/test_kvstore.py using unittest. Then run the tests with "
        "python -m unittest and report the result.",
    ),
    (
        "parser",
        "Write a recursive-descent arithmetic expression parser "
        "(src/parser.py) supporting + - * / and parentheses with proper "
        "precedence, plus tests. Run the tests and report pass/fail.",
    ),
    (
        "httpd",
        "Scaffold a tiny CRUD HTTP server using only the Python standard "
        "library (src/httpd.py) with a JSON record store and GET/POST/PUT/"
        "DELETE routes, then write a smoke test and run it.",
    ),
    (
        "refactor",
        "Given these files are missing, create src/shapes.py defining Circle "
        "and Rectangle with area(), and src/main.py importing both and "
        "printing computed areas. Then grep for 'area' and read the files "
        "back to confirm correctness.",
    ),
]


def run_task(workdir: Path, slug: str, prompt: str) -> dict[str, Any]:
    capture = OUT_DIR / f"{slug}.jsonl"
    start = time.monotonic()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "local_operator.cli",
            "--hosting",
            HOSTING,
            "--model",
            MODEL,
            "--run-in",
            str(workdir),
            "--yolo",
            "exec",
            "--json",
            prompt,
        ],
        capture_output=True,
        text=True,
        env={**os.environ},
        timeout=900,
    )
    wall = time.monotonic() - start
    # The exec --json stream goes to stdout.
    capture.write_text(proc.stdout)
    tokens, cost = tally_cost(capture)
    return {
        "slug": slug,
        "exit": proc.returncode,
        "wall_s": round(wall, 2),
        "tokens": tokens,
        "cost_usd": round(cost, 4) if cost is not None else None,
        "capture": str(capture),
    }


def tally_cost(capture: Path) -> tuple[dict[str, int], float | None]:
    """Count final usage once and price cache-aware provider buckets.

    Unknown pricing is null, never a fabricated free run. A live listing with
    only base input/output prices cannot honestly price cached work, so this
    report uses the shared registry/accounting path or provider-reported cost.
    """
    from local_operator.harness.types import Usage
    from local_operator.model.configure import cost_for_usage
    from local_operator.model.registry import get_model_info

    in_tok = out_tok = 0
    context_tok = 0
    events = []
    for line in capture.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        events.append(event)
    # Both message_end and turn_end carry the same final Message. Choose one
    # canonical event family before summing; older captures without message_end
    # still work. Replayed messages are deduplicated by their durable ID.
    kind = "message_end" if any(e.get("type") == "message_end" for e in events) else "turn_end"
    seen = set()
    usages = []
    for event in events:
        if event.get("type") != kind:
            continue
        message = event.get("message") or {}
        identity = message.get("id")
        if identity and identity in seen:
            continue
        if identity:
            seen.add(identity)
        usage = (event.get("message") or {}).get("usage")
        if not usage:
            continue
        usages.append(Usage.model_validate(usage))
        in_tok += int(usage.get("input_tokens") or 0)
        out_tok += int(usage.get("output_tokens") or 0)
        context_tok = max(context_tok, int(usage.get("context_tokens") or 0))
    try:
        info = get_model_info(HOSTING, MODEL)
        cost = sum(cost_for_usage(HOSTING, info, usage) for usage in usages)
        if cost:
            return {"input": in_tok, "output": out_tok, "max_context": context_tok}, cost
    except Exception:
        pass
    return {"input": in_tok, "output": out_tok, "max_context": context_tok}, None


def measure_base_overhead(workdir: Path) -> dict[str, float]:
    """Peak RSS after building a session but before any turn: the harness's
    fixed cost. Compute here is nothing but imports + session construction."""
    import resource

    # ru_maxrss is BYTES on macOS, KiB on Linux — normalise to MiB so the
    # report is stable across hosts. One throwaway session construction (the
    # harness's fixed cost: imports + skills/embedding + tool inventory)
    # then measure the maxrss delta it drives.
    before = _rss_mib(resource)
    build_a_session(workdir)
    after = _rss_mib(resource)
    return {
        "rss_mib_before": before,
        "rss_mib_after_session": after,
        "session_overhead_mib": round(after - before, 1),
    }


def _rss_mib(resource) -> float:
    """Peak RSS in MiB, handling macOS (bytes) vs Linux (KiB) units."""
    import sys

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1.0 / (1024 * 1024) if sys.platform == "darwin" else 1.0 / 1024
    return round(value * scale, 1)


def build_a_session(workdir: Path) -> None:
    import argparse

    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager
    from local_operator.session_factory import build_initial_blocks

    args = argparse.Namespace(
        hosting=HOSTING,
        model=MODEL,
        agent_name=None,
        agent_id=None,
        yolo=True,
        train=False,
    )
    cm = ConfigManager(workdir)
    cred = CredentialManager(workdir / ".local-operator")
    reg = AgentRegistry(workdir / ".local-operator")
    import asyncio

    asyncio.run(build_initial_blocks(args, cm, cred, reg))


def main() -> int:
    out = {"model": MODEL, "hosting": HOSTING, "tasks": [], "base": {}}
    base_dir = Path(os.environ.get("LO_BENCH_BASE", "/tmp"))
    print(f"# OpenAI-equivalent cost benchmark — {MODEL}\n")

    base = measure_base_overhead(base_dir)
    out["base"] = base
    print("## Base harness overhead")
    print(
        f"  RSS after session construction +{base['session_overhead_mib']} MiB "
        f"({base['rss_mib_before']} -> {base['rss_mib_after_session']} MiB maxrss)\n"
    )

    for slug, prompt in TASKS:
        workdir = OUT_DIR / slug
        workdir.mkdir(parents=True, exist_ok=True)
        print(f"## {slug}")
        result = run_task(workdir, slug, prompt)
        out["tasks"].append(result)
        tok = result["tokens"]
        print(
            f"  exit={result['exit']} wall={result['wall_s']}s "
            f"input={tok['input']} output={tok['output']} "
            f"max_context={tok['max_context']} cost=${result['cost_usd']}"
        )
        sys.stdout.flush()

    report = OUT_DIR / "report.json"
    report.write_text(json.dumps(out, indent=2))
    print(f"\nReport: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
