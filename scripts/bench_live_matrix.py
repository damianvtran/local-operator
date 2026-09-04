"""Run balanced real-provider task pairs against two committed installations.

Uses each checkout's own Python from outside both repositories; the worker
verifies its imported package and exact committed revision. No credentials are
copied. Provider sampling and remote load remain uncontrolled, so these bounded
samples measure outcomes and variability rather than establish causality.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def summarize(output: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(output.glob("*/result.json")):
        result = json.loads(path.read_text())
        calls = result["calls"]
        rows.append(
            {
                "arm": result["arm"],
                "task": result["task"],
                "repeat": result["repeat"],
                "accepted": result["verdict"]["accepted"] and not result["failure"],
                "elapsed_s": result["elapsed_s"],
                "model_calls": len(calls),
                "source_stable": result["source_hash_start"] == result["source_hash_end"],
                "source_head": result["source_head"],
                "usage": {
                    key: sum(call.get("usage", {}).get(key, 0) or 0 for call in calls)
                    for key in (
                        "input_tokens",
                        "output_tokens",
                        "reasoning_tokens",
                        "cache_read_tokens",
                        "cache_write_tokens",
                    )
                },
                "result": str(path.resolve()),
            }
        )
    groups = []
    for arm in ("baseline", "candidate"):
        for task in ("repair", "aggregate"):
            selected = [row for row in rows if row["arm"] == arm and row["task"] == task]
            valid = [row for row in selected if row["source_stable"]]
            if not valid:
                continue
            total_input = sum(row["usage"]["input_tokens"] for row in valid)
            groups.append(
                {
                    "arm": arm,
                    "task": task,
                    "runs": len(selected),
                    "stable_runs": len(valid),
                    "accepted": sum(row["accepted"] for row in valid),
                    "mean_elapsed_s": statistics.mean(row["elapsed_s"] for row in valid),
                    "median_elapsed_s": statistics.median(row["elapsed_s"] for row in valid),
                    "mean_model_calls": statistics.mean(row["model_calls"] for row in valid),
                    "mean_input_tokens": total_input / len(valid),
                    "mean_output_tokens": statistics.mean(
                        row["usage"]["output_tokens"] for row in valid
                    ),
                    "mean_reasoning_tokens": statistics.mean(
                        row["usage"]["reasoning_tokens"] for row in valid
                    ),
                    # OpenAI cached input is a subset of input_tokens, not additive.
                    "cache_read_fraction": (
                        sum(row["usage"]["cache_read_tokens"] for row in valid) / total_input
                        if total_input
                        else None
                    ),
                }
            )
    return {
        "rows": rows,
        "groups": groups,
        "caveat": "Bounded stochastic observations, not causal speed or cost estimates.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline-sha", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name("bench_live_tasks.py").resolve()
    failures = []
    for repeat in range(1, args.repeats + 1):
        for task_index, task in enumerate(("repair", "aggregate")):
            order = (
                ("baseline", "candidate")
                if (repeat + task_index) % 2
                else ("candidate", "baseline")
            )
            for arm in order:
                repo = getattr(args, arm).resolve()
                command = [
                    str(repo / ".venv/bin/python"),
                    str(worker),
                    str(repo),
                    arm,
                    task,
                    str(repeat),
                    "--output",
                    str(output),
                    "--expected-sha",
                    getattr(args, f"{arm}_sha"),
                ]
                try:
                    result = subprocess.run(
                        command,
                        cwd=tempfile.gettempdir(),
                        text=True,
                        capture_output=True,
                        timeout=285,
                    )
                    if result.returncode:
                        # Provider errors may carry request text; do not expose
                        # stderr. The worker persists sanitized task outcomes.
                        failures.append(
                            {
                                "arm": arm,
                                "task": task,
                                "repeat": repeat,
                                "exit_code": result.returncode,
                            }
                        )
                        print(json.dumps(failures[-1]), flush=True)
                    else:
                        print(result.stdout.strip(), flush=True)
                except subprocess.TimeoutExpired:
                    failures.append(
                        {"arm": arm, "task": task, "repeat": repeat, "failure": "process_timeout"}
                    )
                    print(json.dumps(failures[-1]), flush=True)
                report = summarize(output)
                report["process_failures"] = failures
                (output / "summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
