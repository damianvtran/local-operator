#!/usr/bin/env python3
"""Tool-output footprint benchmark: tokens into context, bytes onto disk.

``bench_context_budget.py`` measures the harness at rest and
``bench_task_cost.py`` measures a whole task. Neither sees the thing that
actually blew up here: what ONE tool call costs when the command it ran
produced a lot of output.

The incident this exists to prevent had two halves, and this script measures
both because fixing one alone is what caused it:

- CONTEXT: ``BASH_OUTPUT_LIMIT_CHARS`` used to be 50 KiB, so a single bash
  result carried ~13,000 tokens into the prompt -- over 40% of the
  30,000-token start-context budget in docs/REWRITE.md, spent on one call.
- DISK: the reference implementation caps the context and spills the rest to
  disk, but retains those spills forever. On this workstation that unbounded
  half reached 6.8 GB and filled the volume.

So the numbers reported are per-tool-call tokens, per-turn tokens, bytes
written to disk, and the store's STEADY-STATE size after a workload that
writes far more than the ceiling. A context saving paid for with unbounded
disk is not a saving.

Self-verifying: the run fails (exit 1) if the after-tokens do not beat the
before-tokens, if any single result exceeds the budget, or -- most
importantly -- if the store's steady state exceeds its own ceiling.

Run:
    .venv/bin/python scripts/bench_tool_output.py
    .venv/bin/python scripts/bench_tool_output.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from local_operator.compaction.tokens import count_text_tokens  # noqa: E402
from local_operator.harness.types import ToolContext  # noqa: E402
from local_operator.tools import builtin, spill  # noqa: E402
from local_operator.tools.registry import create_tools  # noqa: E402

#: The cap in force before this work, kept as a literal so the "before" column
#: stays meaningful after the constant itself moves again.
LEGACY_LIMIT_CHARS = 50 * 1024

#: Turns are modelled as this many tool calls. Measured from the repo's own
#: benchmark transcripts, where a working turn issues 3-6 calls; 4 is the
#: middle and the per-turn column is a multiplier, not a separate measurement.
CALLS_PER_TURN = 4


@dataclass
class Workload:
    """One realistic big-output command and what it produced."""

    name: str
    raw_chars: int
    raw_tokens: int
    raw_lines: int
    before_chars: int
    before_tokens: int
    after_chars: int
    after_tokens: int
    spill_bytes: int
    expandable: bool

    @property
    def token_ratio(self) -> float:
        return self.before_tokens / max(self.after_tokens, 1)


def _legacy_truncate(text: str, limit: int = LEGACY_LIMIT_CHARS) -> str:
    """The pre-change truncation, reproduced exactly.

    Kept here rather than imported: the point of a before/after is that the
    "before" does not move when the implementation does, and the old function
    no longer exists to import.
    """
    marker = "\n\n... [output truncated] ...\n\n"
    if len(text) <= limit:
        return text
    budget = limit - len(marker)
    head = budget // 2
    return text[:head] + marker + text[len(text) - (budget - head) :]


def _capture(command: str, cwd: Path) -> str:
    """Run a real command and return its combined transcript."""
    proc = subprocess.run(
        ["/bin/sh", "-c", command], cwd=cwd, capture_output=True, text=True, timeout=900
    )
    return f"--- stdout ---\n{proc.stdout or '(empty)'}\n--- stderr ---\n{proc.stderr or '(empty)'}"


def _dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


async def _measure(name: str, text: str, context: ToolContext) -> Workload:
    """Before/after for one captured output, through the REAL code path."""
    before = _legacy_truncate(text)
    store = spill.get_store()
    bytes_before = _dir_bytes(store.root)
    after, details = builtin.spill_truncate(text, "bash", context)
    bytes_after = _dir_bytes(store.root)

    expandable = False
    if details is not None:
        handle = details["spill"]["handle"]
        # Prove the handle actually resolves rather than trusting that it was
        # written: a benchmark that reports a saving from a broken expansion
        # path is reporting a regression as a win.
        tools = {t.name: t for t in create_tools(context)}
        probe = await tools["read"].execute(
            "bench", {"path": handle, "range": "1-5"}, None, None, context
        )
        expandable = not probe.is_error and "1|" in probe.text

    return Workload(
        name=name,
        raw_chars=len(text),
        raw_tokens=count_text_tokens(text),
        raw_lines=len(text.splitlines()),
        before_chars=len(before),
        before_tokens=count_text_tokens(before),
        after_chars=len(after),
        after_tokens=count_text_tokens(after),
        spill_bytes=bytes_after - bytes_before,
        expandable=expandable,
    )


async def _steady_state(target_bytes: int) -> tuple[int, int, int, int]:
    """Push ``target_bytes`` of distinct outputs through the store.

    Returns ``(written, resident_before, resident_after, entries)``.

    This is the half the reference implementation gets wrong, so it is
    measured rather than asserted in a comment: the store must stay under its
    ceiling no matter how much is pushed through it. The target deliberately
    EXCEEDS the ceiling — a steady-state number taken from a workload that fit
    would prove nothing about eviction, which is the mechanism under test.
    """
    store = spill.get_store()
    resident_before = store.total_bytes()
    written = 0
    chunk = "steady state line with realistic length for a build log\n" * 2000
    index = 0
    while written < target_bytes:
        payload = f"run {index}\n{chunk}"
        written += len(payload.encode())
        store.write(payload, tool_name="bash", session_id="bench")
        index += 1
    return written, resident_before, store.total_bytes(), store.entry_count()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--overrun",
        type=float,
        default=1.5,
        help="multiple of the store ceiling to push through it (must exceed 1.0)",
    )
    args = parser.parse_args()

    scratch = Path(tempfile.mkdtemp(prefix="lo-bench-output-"))
    config = scratch / "config"
    import os

    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(config)
    context = ToolContext(cwd=str(REPO), session_id="bench-tool-output")

    try:
        print("capturing real workloads (this runs the actual commands)...")
        captures = {
            "pytest -v (unit suite)": _capture(
                ".venv/bin/python -m pytest tests/unit -v --no-header", REPO
            ),
            "grep -rn 'def ' (large search)": _capture(
                "grep -rn 'def ' local_operator --include=*.py", REPO
            ),
            "git log -p (long build-log-like)": _capture("git log -p --stat -n 80", REPO),
        }

        results = [await _measure(name, text, context) for name, text in captures.items()]

        print()
        print("=" * 96)
        print("TOKENS ENTERING CONTEXT PER TOOL CALL")
        print("=" * 96)
        print(
            f"{'workload':34}{'raw':>10}{'before':>10}{'after':>9}"
            f"{'saved':>9}{'ratio':>8}{'expand':>8}"
        )
        for r in results:
            print(
                f"{r.name:34}{r.raw_tokens:>10,}{r.before_tokens:>10,}{r.after_tokens:>9,}"
                f"{r.before_tokens - r.after_tokens:>9,}{r.token_ratio:>7.1f}x"
                f"{('yes' if r.expandable else 'NO'):>8}"
            )

        before_turn = sum(r.before_tokens for r in results) / len(results) * CALLS_PER_TURN
        after_turn = sum(r.after_tokens for r in results) / len(results) * CALLS_PER_TURN
        print()
        print(f"per TURN ({CALLS_PER_TURN} tool calls, mean of the workloads above):")
        print(f"  before : {before_turn:>10,.0f} tokens")
        print(f"  after  : {after_turn:>10,.0f} tokens")
        saved_pct = (1 - after_turn / before_turn) * 100
        print(f"  saved  : {before_turn - after_turn:>10,.0f} tokens ({saved_pct:.1f}%)")

        print()
        print("=" * 96)
        print("BYTES ON DISK")
        print("=" * 96)
        for r in results:
            print(f"{r.name:34} spilled {r.spill_bytes:>12,} B  (raw output {r.raw_chars:>12,} B)")

        ceiling = spill.SPILL_TOTAL_LIMIT_BYTES
        target = int(ceiling * args.overrun)
        print()
        print(
            f"pushing {target / 1024 / 1024:.0f} MB through a "
            f"{ceiling / 1024 / 1024:.0f} MB store ({args.overrun}x its ceiling)..."
        )
        written, resident_before, resident, entries = await _steady_state(target)
        print(f"  written to store : {written:>14,} B")
        print(f"  resident on disk : {resident:>14,} B  ({entries} entries)")
        print(f"  ceiling          : {ceiling:>14,} B")
        print(
            f"  reclaimed        : {written + resident_before - resident:>14,} B " "by LRU eviction"
        )

        # --- self-verification ------------------------------------------
        failures: list[str] = []
        for r in results:
            if r.after_tokens >= r.before_tokens:
                failures.append(f"{r.name}: after ({r.after_tokens}) did not beat before")
            if r.after_chars > builtin.TOOL_OUTPUT_LIMIT_CHARS * 1.15:
                failures.append(f"{r.name}: result {r.after_chars} chars exceeds the budget")
            if not r.expandable:
                failures.append(f"{r.name}: truncated output was NOT expandable by handle")
        if resident > ceiling:
            failures.append(f"store steady state {resident} exceeds its ceiling {ceiling}")

        if args.json:
            print()
            print(
                json.dumps(
                    {
                        "workloads": [asdict(r) for r in results],
                        "per_turn_before": round(before_turn),
                        "per_turn_after": round(after_turn),
                        "store_written": written,
                        "store_resident": resident,
                        "store_ceiling": ceiling,
                        "failures": failures,
                    },
                    indent=2,
                )
            )

        print()
        if failures:
            for failure in failures:
                print(f"FAIL: {failure}")
            return 1
        print("PASS: context shrank, every result fits the budget, every handle resolved,")
        print("      and the store held its ceiling under a deliberate overrun.")
        return 0
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
