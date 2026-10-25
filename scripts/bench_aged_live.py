#!/usr/bin/env python3
"""Matched live task trials for aged clearing, not a recorded-trajectory replay.

Run this SAME script using each worktree's own interpreter; ``--repo`` checks
its import origin and ``--arm`` labels the exact source revision. Credentials
stay in the real HOME's native AuthStore; only config, transcripts and a
synthetic workspace are isolated. No shared defaults or role pins change.

Three workloads exercise cross-file aggregation, anomaly investigation and
catalogue reconciliation. Each source is a noisy ~2k-token service export;
all relevant facts are real generated records, not a scripted assistant.
The provider chooses its calls, its notes and whether to re-fetch. Objective
checks use the generator's held-out expected answer, never the model's own
claim. The read inventory requirement models an audit that cannot sample.

This is a controlled long-retrieval workload, not evidence about arbitrary
coding tasks. One trial per invocation; an external seeded schedule interleaves
BASE/HEAD and repeats identical fixture seeds. Results include source identity,
actual provider usage/cost, correctness and actual same-path reads AFTER a
result was cleared (not the offline historical-repeat proxy).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--arm", choices=("base", "head"), required=True)
    parser.add_argument("--task", choices=("aggregate", "audit", "reconcile"), required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=95)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    asyncio.run(trial(args))


async def trial(args) -> None:
    import local_operator

    repo = args.repo.resolve()
    assert Path(local_operator.__file__).resolve().is_relative_to(repo)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    # Capture the runtime delta as well as HEAD: a remediation trial can run
    # before the batched commit without pretending it tested the old bytes.
    runtime_diff = subprocess.check_output(
        ["git", "diff", "HEAD", "--", "local_operator"], cwd=repo
    )
    from local_operator.compaction.thresholds import CompactionSettings
    from local_operator.harness.types import AgentTool, Message, TextContent, ToolResult
    from local_operator.model.configure import build_model_spec, create_stream_fn
    from local_operator.providers.auth_store import AuthStore
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    logging.disable(logging.CRITICAL)
    auth_path = Path.home() / ".local-operator" / "auth.db"
    root = Path(tempfile.mkdtemp(prefix="aged-live-"))
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(root / "config")
    os.environ["LOCAL_OPERATOR_HOME"] = str(root / "agent-home")
    workspace = root / "workspace"
    workspace.mkdir()
    seed = 292200 + args.repeat
    rng = random.Random(seed)
    pages = 1 if args.preflight else args.pages
    expected: dict[str, Any] = {}
    for index in range(pages):
        source = f"source-{index:03d}.txt"
        if args.task == "aggregate":
            region = rng.choice(["north", "south", "west"])
            amount = rng.randrange(-1000, 4000)
            expected[region] = expected.get(region, 0) + amount
            fact = {"record": index, "region": region, "cents": amount}
        elif args.task == "audit":
            qty, price = rng.randrange(1, 10), rng.randrange(1, 100)
            billed = qty * price + (7 if index % 9 == 2 else 0)
            if billed != qty * price:
                expected[str(index)] = billed - qty * price
            fact = {"record": index, "quantity": qty, "unit_cents": price, "billed_cents": billed}
        else:
            raw = rng.choice(["  ALPHA ", "beta  ", " GAMMA", "delta "])
            expected[str(index)] = raw.strip().lower() + "-" + str(1000 + index)
            fact = {"record": index, "raw_group": raw, "serial": 1000 + index}
        # These service-export lines are source evidence, not more task facts.
        # All pages have identical structure but distinct content/paths. This
        # prevents same-path superseding from testing a different prune pass.
        noise = "\n".join(
            f"telemetry {index}:{line}: routine checks completed successfully " + "nominal " * 30
            for line in range(50)
        )
        (workspace / source).write_text("TASK_RECORD " + json.dumps(fact) + "\n" + noise)
    manifests = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in workspace.glob("source-*")
    }
    reads: dict[str, int] = {}
    tool_counts = {"read": 0, "write": 0}
    cleared_paths: set[str] = set()
    refetches: list[str] = []
    requests: list[dict[str, Any]] = []

    async def read(call_id, arguments, signal, on_update, context):
        tool_counts["read"] += 1
        name = arguments.get("path", "")
        target = workspace / name
        if target.parent != workspace or not target.is_file():
            return ToolResult(
                tool_call_id=call_id,
                tool_name="read",
                is_error=True,
                content=[TextContent(text="Unknown source or note path")],
            )
        if name in cleared_paths:
            refetches.append(name)
        reads[name] = reads.get(name, 0) + 1
        return ToolResult(
            tool_call_id=call_id,
            tool_name="read",
            content=[TextContent(text=target.read_text())],
            details={"path": name},
        )

    async def write(call_id, arguments, signal, on_update, context):
        tool_counts["write"] += 1
        name, content = arguments.get("path", ""), arguments.get("content", "")
        if name not in ("notes.txt", "answer.json"):
            return ToolResult(
                tool_call_id=call_id,
                tool_name="write",
                is_error=True,
                content=[TextContent(text="Only notes.txt and answer.json are writable")],
            )
        (workspace / name).write_text(content)
        return ToolResult(
            tool_call_id=call_id, tool_name="write", content=[TextContent(text="Saved " + name)]
        )

    tools = [
        AgentTool(
            name="read",
            description="Read one synthetic service-export source or your notes.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            execute=read,
        ),
        AgentTool(
            name="write",
            description="Write notes.txt or the final answer.json artifact.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
            execute=write,
        ),
    ]
    auth = AuthStore(auth_path)
    stream = create_stream_fn(
        auth,
        {
            "retry": {
                "enabled": False,
                "usageAwareFallback": False,
                "usageAwareAccountPick": False,
            },
            "effort": {"auto": False},
        },
        session_id=f"aged-bench-{args.arm}-{args.task}-{args.repeat}",
    )
    model = build_model_spec("openrouter", "deepseek/deepseek-v4-flash-0731").model_copy(
        update={"reasoning_effort": "low", "max_output_tokens": 4096}
    )
    session = None

    async def capture(request, signal):
        if len(requests) >= pages * 4 + 20:
            raise RuntimeError("benchmark request budget")
        if session is not None:
            for message in session._context.messages:
                if (
                    isinstance(message, Message)
                    and message.role == "tool"
                    and (message.provider_payload or {}).get("pruned")
                ):
                    if message.text.startswith("[Older result cleared"):
                        path = ((message.provider_payload or {}).get("details") or {}).get("path")
                        if path:
                            cleared_paths.add(path)
        row = {
            "index": len(requests),
            "model": request.model.model_id,
            "stub_count": sum(m.text.startswith("[Older result cleared") for m in request.messages),
        }
        requests.append(row)
        started = time.monotonic()
        async for event in stream(request, signal):
            if (usage := getattr(event, "usage", None)) is not None:
                row["usage"] = usage.model_dump(mode="json")
            yield event
        row["wall_s"] = time.monotonic() - started
        if len(requests) % 10 == 0:
            print(
                json.dumps(
                    {
                        "progress": len(requests),
                        "arm": args.arm,
                        "task": args.task,
                        "stubs": row["stub_count"],
                    }
                ),
                flush=True,
            )

    settings = CompactionSettings()
    session = Session(
        model=model,
        stream_fn=capture,
        tools=tools,
        transcript=Transcript(root / "transcript"),
        cwd=str(workspace),
        session_id=f"aged-bench-{args.arm}-{args.task}-{args.repeat}",
        system_blocks_provider=lambda: [
            "You solve synthetic operational data tasks. Read only the named sources. "
            "Use notes.txt as durable working memory when useful. Tool output can be large; "
            "do not quote telemetry in your response. Never inspect credentials or unrelated files."
        ],
        compaction_settings=settings,
    )
    instruction = {
        "aggregate": (
            "Sum TASK_RECORD cents by region. answer.json must be an object mapping region "
            "to integer total cents."
        ),
        "audit": (
            "Find every TASK_RECORD with billed_cents != quantity * unit_cents. answer.json must "
            "map the record number (string) to billed_cents minus quantity*unit_cents, "
            "excluding correct records."
        ),
        "reconcile": (
            "Normalize every TASK_RECORD: strip raw_group whitespace, lowercase it, append '-' "
            "and the decimal serial. answer.json must map each record number (string) to that "
            "normalized identifier."
        ),
    }[args.task]
    prompt = (
        f"Audit the {pages} source exports source-000.txt through source-{pages-1:03d}.txt. "
        "Read EVERY source in filename order: these are independent audit evidence, no sampling. "
        "Issue exactly ONE read per assistant response so each source observation has its own "
        "audit boundary. You may write/read notes.txt to retain facts. Ignore telemetry; only "
        "the TASK_RECORD line is task data. " + instruction + " Write answer.json and finish."
    )
    started = time.monotonic()
    failure = None
    try:
        await asyncio.wait_for(session.prompt(prompt), timeout=args.timeout)
    except Exception as exc:
        failure = type(exc).__name__
    elapsed = time.monotonic() - started
    try:
        actual = json.loads((workspace / "answer.json").read_text())
    except (OSError, ValueError):
        actual = None
    totals: dict[str, float] = {}
    for row in requests:
        for key, value in row.get("usage", {}).items():
            if isinstance(value, (int, float)) and key != "at_ms":
                totals[key] = totals.get(key, 0) + value
    result = {
        "arm": args.arm,
        "task": args.task,
        "repeat": args.repeat,
        "seed": seed,
        "revision": revision,
        "runtime_diff_sha256": hashlib.sha256(runtime_diff).hexdigest(),
        "model": model.model_id,
        "provider": model.provider,
        "settings": settings.model_dump(),
        "pages": pages,
        "fixture_sha256": hashlib.sha256(
            json.dumps(manifests, sort_keys=True).encode()
        ).hexdigest(),
        "correct": actual == expected,
        "all_sources_read": all(p in reads for p in manifests),
        "calls": len(requests),
        "tool_calls": sum(tool_counts.values()),
        "read_calls": sum(reads.values()),
        "tool_counts": tool_counts,
        "wall_s": elapsed,
        "failure": failure,
        "usage_totals": totals,
        "cleared_paths": len(cleared_paths),
        "after_clear_refetches": refetches,
        "repeated_reads": sum(max(0, n - 1) for p, n in reads.items() if p in manifests),
        "requests": requests,
        "artifact_dir": str(root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("requests", "settings")}),
        flush=True,
    )
    await session.dispose()
    auth.close()


if __name__ == "__main__":
    main()
