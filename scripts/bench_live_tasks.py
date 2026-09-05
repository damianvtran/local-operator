"""Matched real-provider Sessions; all task content is synthetic and local.

Each invocation runs one arm/task/repeat in a fresh scratch directory. Uses the
native auth database directly, without printing or copying credentials. Outputs
contain scalar usage, timings, hashes and verifier verdicts, never auth values.
"""

import argparse
import ast
import asyncio
import csv
import hashlib
import importlib.util
import json
import logging
import os
import random
import subprocess
import time
import uuid
from pathlib import Path

from bench_live_workspace import initialize_workspace

parser = argparse.ArgumentParser()
parser.add_argument("repo")
parser.add_argument("arm")
parser.add_argument("task", choices=["repair", "aggregate", "preflight"])
parser.add_argument("repeat", type=int)
parser.add_argument("--output", required=True)
parser.add_argument("--effort", default="low", choices=["low", "high"])
parser.add_argument("--expected-sha", required=True, help="Exact committed runtime revision")
args = parser.parse_args()
repo = Path(args.repo).resolve()
output = Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=True)
actual_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
if actual_sha != args.expected_sha:
    raise SystemExit("Source HEAD differs from --expected-sha")
if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "local_operator"], cwd=repo).returncode:
    raise SystemExit("Runtime source has uncommitted changes")
run = output / f"{args.arm}-{args.task}-{args.repeat}-{uuid.uuid4().hex[:8]}"
workspace = run / "workspace"
initialize_workspace(workspace, environment=os.environ)
(run / "config").mkdir()
auth_db = Path.home() / ".local-operator" / "auth.db"
os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(run / "config")
# Intentionally share the native model catalogue/cache under the real home.
# Only config/transcripts/task files are isolated; these are NOT cold-cache runs.
# Model resolution/setup is outside the session.prompt timing.
# Source import must resolve from this installation even though execution starts
# outside the checkout. Do not set PYTHONPATH or sys.path to conceal a bad venv.
import local_operator  # noqa: E402

assert Path(local_operator.__file__).resolve().is_relative_to(repo), local_operator.__file__
from local_operator.harness.types import ToolContext  # noqa: E402
from local_operator.model.configure import (  # noqa: E402
    build_model_spec,
    create_stream_fn,
)
from local_operator.prompts_api import build_system_blocks  # noqa: E402
from local_operator.providers import clients as provider_clients  # noqa: E402
from local_operator.providers.auth_store import AuthStore  # noqa: E402
from local_operator.providers.clients import _estimated_prompt_tokens  # noqa: E402
from local_operator.session.session import Session  # noqa: E402
from local_operator.session.transcript import Transcript  # noqa: E402
from local_operator.tools.registry import create_tools  # noqa: E402

logging.disable(logging.CRITICAL)


def digest_source():
    result = hashlib.sha256()
    for path in sorted((repo / "local_operator").rglob("*")):
        if "__pycache__" in path.parts or path.suffix not in (".py", ".md", ".tcss"):
            continue
        result.update(str(path.relative_to(repo)).encode())
        data = path.read_text()
        if path.suffix == ".py":
            data = ast.dump(ast.parse(data), include_attributes=False)
        result.update(data.encode())
    return result.hexdigest()


def seed_task():
    if args.task == "preflight":
        return "Respond with exactly READY.", None
    if args.task == "repair":
        (workspace / "ledger.py").write_text(
            '"""Invoice totals by currency."""\nfrom decimal import Decimal\n\ndef sum'
            "marize(rows):\n    totals = {}\n    for row in rows:\n        currency = "
            "row['currency']\n        totals[currency] = totals.get(currency, 0.0) +"
            " float(row['amount'])\n    return {key: str(value) for key, value in to"
            "tals.items()}\n"
        )
        (workspace / "test_ledger.py").write_text(
            "import unittest\nfrom ledger import summarize\n\nclass LedgerTests(unitte"
            "st.TestCase):\n    def test_paid_only(self):\n        self.assertEqual(s"
            "ummarize([{'id': 'a', 'currency': 'usd', 'amount': '0.10', 'status': '"
            "paid'}, {'id': 'b', 'currency': 'USD', 'amount': '0.20', 'status': 'pa"
            "id'}, {'id': 'c', 'currency': 'USD', 'amount': '9.00', 'status': 'pend"
            "ing'}]), {'USD': {'count': 2, 'total': '0.30'}})\n    def test_latest_r"
            "ecord_wins(self):\n        self.assertEqual(summarize([{'id': 'a', 'cur"
            "rency': 'USD', 'amount': '5.00', 'status': 'paid'}, {'id': 'a', 'curre"
            "ncy': 'USD', 'amount': '5.00', 'status': 'void'}]), {})\n    def test_e"
            "mpty(self):\n        self.assertEqual(summarize([]), {})\n\nif __name__ ="
            "= '__main__':\n    unittest.main()\n"
        )
        prompt = (
            "Fix ledger.py summarize(rows). rows may be a list or one-pass iterator. "
            'For duplicate invoice id, the last row wins, before filtering to status="paid". '
            "Group paid invoices by uppercase currency. Return each group as "
            '{"count": integer, "total": string rounded to exactly2 decimal places using '
            "Decimal ROUND_HALF_UP}. Accept decimal amount strings including negative amounts; "
            "do not mutate input rows. Run existing tests and add meaningful coverage for "
            "iterator input, negative amounts and rounding. Work only in this directory. "
            "The python command is python3. Finish once implemented and verified."
        )
        return prompt, None
    rng = random.Random(7711 + args.repeat)
    customers = [
        {
            "customer_id": f"C{i:03}",
            "active": str(i % 5 != 0).lower(),
            "region": ("CA", "US", "GB")[i % 3],
        }
        for i in range(72)
    ]
    rows = [
        {
            "event_id": f"E{i:04}",
            "customer_id": rng.choice(customers)["customer_id"],
            "amount_cents": rng.randint(-3000, 15000),
            "status": rng.choice(["success", "success", "failed", "pending"]),
        }
        for i in range(640)
    ]
    for name, data in [("customers.csv", customers), ("events.csv", rows)]:
        with (workspace / name).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    eligible = {
        r["customer_id"] for r in customers if r["active"] == "true" and r["region"] in ("CA", "US")
    }
    totals = {}
    for row in rows:
        if row["customer_id"] in eligible and row["status"] == "success":
            total = totals.setdefault(
                row["customer_id"],
                {
                    "customer_id": row["customer_id"],
                    "success_count": 0,
                    "successful_total_cents": 0,
                },
            )
            total["success_count"] += 1
            total["successful_total_cents"] += row["amount_cents"]
    expected = sorted(
        [
            row
            for row in totals.values()
            if row["success_count"] >= 3 and row["successful_total_cents"] >= 10000
        ],
        key=lambda row: (-row["successful_total_cents"], row["customer_id"]),
    )
    prompt = (
        "Read customers.csv and events.csv and create eligibility.json. Include only "
        "active=true customers in CA or US. Aggregate their status=success events, "
        "including negative amount_cents. Keep customers with at least3 success events "
        "and successful_total_cents>=10000. Output a JSON array with exactly customer_id, "
        "success_count, successful_total_cents fields per row; sort by total descending, "
        "then customer_id ascending. Preserve both CSV inputs. Verify your result with "
        "an independent check before finishing. Use Python for aggregation; the python "
        "command is python3. Work only in this directory."
    )
    return prompt, expected


def verify(expected, initial_hashes):
    if args.task == "preflight":
        return {"accepted": True}
    if args.task == "aggregate":
        try:
            actual = json.loads((workspace / "eligibility.json").read_text())
            unchanged = all(
                hashlib.sha256((workspace / name).read_bytes()).hexdigest() == digest
                for name, digest in initial_hashes.items()
            )
            return {
                "accepted": actual == expected and unchanged,
                "output_matches": actual == expected,
                "inputs_unchanged": unchanged,
                "expected_rows": len(expected),
                "actual_rows": len(actual) if isinstance(actual, list) else -1,
            }
        except Exception as exc:
            return {"accepted": False, "verifier_error": type(exc).__name__}
    # Independent hidden checks run here, not in the model's editable tests.
    try:
        from decimal import ROUND_HALF_UP, Decimal

        module_spec = importlib.util.spec_from_file_location(
            "verified_ledger", workspace / "ledger.py"
        )
        assert module_spec is not None and module_spec.loader is not None
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        rng = random.Random(9188 + args.repeat)
        rows = [
            {
                "id": f"I{rng.randrange(30)}",
                "currency": rng.choice(["usd", "USD", "cad", "CAD"]),
                "amount": str(Decimal(rng.randrange(-90000, 90000)) / 1000),
                "status": rng.choice(["paid", "paid", "pending", "void"]),
            }
            for _ in range(240)
        ]
        before = json.dumps(rows, sort_keys=True)
        latest = {r["id"]: r for r in rows}
        groups = {}
        for r in latest.values():
            if r["status"] == "paid":
                key = r["currency"].upper()
                item = groups.setdefault(key, {"count": 0, "total": Decimal(0)})
                item["count"] += 1
                item["total"] += Decimal(r["amount"])
        reference = {
            key: {
                "count": value["count"],
                "total": str(value["total"].quantize(Decimal(".01"), rounding=ROUND_HALF_UP)),
            }
            for key, value in groups.items()
        }
        checks = [
            module.summarize(rows) == reference,
            module.summarize(iter(rows)) == reference,
            json.dumps(rows, sort_keys=True) == before,
            module.summarize([]) == {},
        ]
        return {
            "accepted": all(checks),
            "hidden_checks_passed": sum(checks),
            "hidden_checks": len(checks),
        }
    except Exception as exc:
        return {"accepted": False, "verifier_error": type(exc).__name__}


async def main():
    source_start = digest_source()
    prompt, expected = seed_task()
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in workspace.glob("*.csv")
    }
    auth = AuthStore(auth_db)
    session_id = f"bench-{args.arm}-{args.task}-{args.repeat}-{uuid.uuid4().hex}"
    settings = {
        "retry": {"enabled": False, "usageAwareFallback": False, "usageAwareAccountPick": False},
        "providers": {"openai": {"api": "responses"}},
        "effort": {"auto": False},
    }
    stream = create_stream_fn(auth, settings, session_id=session_id)
    spec = build_model_spec("openai", "gpt-5.6-sol").model_copy(
        update={"reasoning_effort": args.effort, "max_output_tokens": 8192}
    )
    calls = []
    tool_events = []
    # The candidate finalizes native accounting inside its body builder after
    # account selection. Observe the shaped metadata, not the earlier hint.
    bind_native = getattr(provider_clients, "bind_native_context", None)
    if bind_native is not None:

        def capture_binding(request, *parameters, **kwargs):
            shaped = bind_native(request, *parameters, **kwargs)
            if calls:
                calls[-1].update(
                    {
                        "hint": shaped.context_tokens_hint,
                        "hint_owner": shaped.context_tokens_hint_model,
                        "hint_measured": shaped.context_tokens_hint_measured,
                        "native_context_tokens": shaped.native_context_tokens,
                        "admission_estimate": _estimated_prompt_tokens(shaped),
                    }
                )
            return shaped

        provider_clients.bind_native_context = capture_binding
    real_client_for = stream._client_for

    class CaptureClient:
        def __init__(self, client):
            self.client = client

        async def stream(self, request, api_key, oauth_access=None):
            if len(calls) >= 14:
                raise RuntimeError("benchmark_call_budget")
            record = {
                "index": len(calls) + 1,
                "model": request.model.model_id,
                "purpose": getattr(request, "purpose", "turn"),
                "hint": request.context_tokens_hint,
                "hint_owner": getattr(request, "context_tokens_hint_model", None),
                "hint_measured": getattr(request, "context_tokens_hint_measured", None),
                "admission_estimate": _estimated_prompt_tokens(request),
                "native_input_messages": sum(
                    bool((m.provider_payload or {}).get("native_replay")) for m in request.messages
                ),
            }
            calls.append(record)
            start = time.monotonic()
            usage = None
            reply = []
            try:
                async for event in self.client.stream(request, api_key, oauth_access):
                    if "first_event_s" not in record:
                        record["first_event_s"] = time.monotonic() - start
                    if event.type == "text_delta":
                        reply.append(event.delta)
                    if getattr(event, "usage", None) is not None:
                        usage = event.usage
                    if getattr(event, "stop_reason", None):
                        record["stop_reason"] = event.stop_reason
                    yield event
            except Exception as exc:
                record["error_type"] = type(exc).__name__
                record["http_status"] = getattr(exc, "status", None)
                raise
            finally:
                record["elapsed_s"] = time.monotonic() - start
                if args.task == "preflight":
                    record["reply_ready"] = "".join(reply).strip() == "READY"
                if usage is not None:
                    record["usage"] = {
                        key: getattr(usage, key, None)
                        for key in (
                            "input_tokens",
                            "output_tokens",
                            "cache_read_tokens",
                            "cache_write_tokens",
                            "reasoning_tokens",
                            "context_tokens",
                        )
                    }

    stream._client_for = lambda spec: CaptureClient(real_client_for(spec))
    tools = create_tools(
        ToolContext(cwd=str(workspace), session_id=session_id),
        enabled=["read", "write", "edit", "bash", "eval", "glob", "grep"],
    )
    blocks = build_system_blocks(
        tools,
        "",
        "A synthetic benchmark workspace. Operate only inside the current directory. "
        "Do not inspect credentials or unrelated files.",
        "2026-09-04",
        interactive=False,
    )
    session = Session(
        model=spec,
        stream_fn=stream,
        tools=tools,
        transcript=Transcript(run / "transcript"),
        session_id=session_id,
        cwd=str(workspace),
        yolo=True,
        system_blocks_provider=lambda: blocks,
    )

    def on_event(event):
        if event.type in ("tool_execution_start", "tool_execution_end"):
            tool_events.append(
                {
                    "type": event.type,
                    "tool": event.tool_name,
                    "is_error": bool(getattr(getattr(event, "result", None), "is_error", False)),
                }
            )

    session.subscribe(on_event)
    system_hash = hashlib.sha256(json.dumps(blocks).encode()).hexdigest()
    schemas_hash = hashlib.sha256(
        json.dumps(
            [(tool.name, tool.description, tool.parameters) for tool in tools], sort_keys=True
        ).encode()
    ).hexdigest()
    started = time.monotonic()
    failure = None
    try:
        await asyncio.wait_for(session.prompt(prompt), timeout=240)
    except Exception as exc:
        failure = type(exc).__name__
    elapsed = time.monotonic() - started
    # Session.prompt renders provider failures instead of necessarily raising.
    # Keep artifact correctness distinct from a cleanly completed model turn.
    if failure is None and (not calls or calls[-1].get("stop_reason") != "stop"):
        failure = "incomplete_provider_turn"
    try:
        await asyncio.wait_for(session.dispose(), timeout=15)
    finally:
        await stream.close()
        auth.close()
    verdict = verify(expected, hashes)
    if args.task == "preflight":
        verdict["accepted"] = bool(calls and calls[-1].get("reply_ready") and not failure)
    result = {
        "arm": args.arm,
        "task": args.task,
        "repeat": args.repeat,
        "model": "openai/gpt-5.6-sol",
        "effort": args.effort,
        "source_head": actual_sha,
        "cache_isolation": (
            "shared native home catalogue; fresh config, transcript and task workspace"
        ),
        "system_sha256": system_hash,
        "tool_schemas_sha256": schemas_hash,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hash_start": source_start,
        "source_hash_end": digest_source(),
        "elapsed_s": elapsed,
        "failure": failure,
        "verdict": verdict,
        "calls": calls,
        "tool_events": tool_events,
        "artifact_dir": str(run),
        "settings": settings,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }
    (run / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "arm": args.arm,
                "task": args.task,
                "repeat": args.repeat,
                "elapsed_s": round(elapsed, 3),
                "accepted": verdict["accepted"],
                "failure": failure,
                "calls": len(calls),
                "result": str(run / "result.json"),
            }
        ),
        flush=True,
    )


asyncio.run(main())
