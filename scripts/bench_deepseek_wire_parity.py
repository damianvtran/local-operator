#!/usr/bin/env python3
"""Compare DeepSeek's chat/completions wire against its Anthropic-compatible wire.

DeepSeek serves two request shapes for the same model: the native
``/v1/chat/completions`` route this harness uses, and an Anthropic-compatible
``/anthropic/v1/messages`` route that DeepSeek's own harness uses for its
DeepSeek models. This benchmark runs one small, verifiable tool task over both,
so the only variable is the wire shape: same task, same three tool schemas,
same tool executor, same workspace layout, same output budget.

What it answers, and what it does not: it measures turns to completion,
tool-call validity, prompt/output tokens and wall time per wire. It does NOT
exercise ``local_operator``'s adapters -- it deliberately speaks both wires
directly, because the adapter is the thing being varied. The chat arm still
sends the shape this harness sends today (four system blocks, ``thinking`` plus
``reasoning_effort``); the messages arm sends the shape DeepSeek's harness sends
(one joined ``system`` string, ``thinking`` plus ``output_config.effort``).

Token accounting differs BY WIRE and must not be compared raw: on the OpenAI
shape ``usage.input_tokens`` already includes cached tokens, while on the
Messages shape it excludes them and ``cache_read_input_tokens`` carries them.
This script reports ``prompt_tokens`` normalised to the full prompt on both
wires, which is the only figure the two can be compared on.

Credentials come from the harness auth store read-only (the ``deepseek`` API-key
row) or ``DEEPSEEK_API_KEY``. The key is never printed.

Budget: 2 wires x --repeat runs x at most --max-turns small calls.

Examples:
    .venv/bin/python scripts/bench_deepseek_wire_parity.py --dry-run
    .venv/bin/python scripts/bench_deepseek_wire_parity.py --live --repeat 5
    .venv/bin/python scripts/bench_deepseek_wire_parity.py --live --wire messages
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

CHAT_URL = "https://api.deepseek.com/v1/chat/completions"
MESSAGES_URL = "https://api.deepseek.com/anthropic/v1/messages"
MODEL = "deepseek-flash"

SYSTEM_BLOCKS = [
    "You are a coding assistant working in a single directory.",
    "## Available tools\n\n- read_file\n- write_file\n- list_dir",
    "Today is 2026-10-17. A benchmark workspace; operate only inside it.",
    "<interactivity>\nNo interface is attached.",
]

TOOLS = [
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file and return its contents.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "write_file",
        "description": "Write UTF-8 text to a file, creating parent directories.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_dir",
        "description": "List entries in a directory.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
]

TASK = (
    "In this directory there is data.csv and spec.md. Read spec.md, then compute what it "
    "asks from data.csv, and write the result to result.json. Then reply DONE."
)

SPEC = """# Totals spec

Produce result.json containing a JSON object with exactly two keys:

- `total`: the sum of `amount` over rows whose `status` is `ok`, as a number
- `rows`: the count of those rows

Rows whose status is not `ok` are ignored. Amounts may be negative.
"""

DATA = "id,amount,status\n1,10.5,ok\n2,-3.25,ok\n3,7,skip\n4,2.25,ok\n5,100,ok\n"
EXPECTED_TOTAL = 10.5 - 3.25 + 2.25 + 100
EXPECTED_ROWS = 4


def seed(workspace: Path) -> None:
    """Create the task's inputs in ``workspace``."""
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "spec.md").write_text(SPEC)
    (workspace / "data.csv").write_text(DATA)


def _inside(workspace: Path, raw: str) -> Path:
    """Resolve ``raw`` against ``workspace``, refusing to escape it."""
    target = (workspace / raw).resolve()
    root = workspace.resolve()
    if target != root and root not in target.parents:
        raise ValueError("path outside workspace")
    return target


def run_tool(name: str, arguments: Any, workspace: Path) -> str:
    """Execute one tool call and return the text the model sees."""
    if not isinstance(arguments, dict):
        return "error: arguments were not a JSON object"
    try:
        if name == "read_file":
            return _inside(workspace, str(arguments["path"])).read_text()[:20000]
        if name == "write_file":
            target = _inside(workspace, str(arguments["path"]))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(arguments["content"]))
            return f"wrote {len(str(arguments['content']))} chars"
        if name == "list_dir":
            target = _inside(workspace, str(arguments.get("path", ".")))
            return "\n".join(sorted(entry.name for entry in target.iterdir()))
        return f"error: unknown tool {name}"
    except Exception as exc:  # noqa: BLE001 - the model sees the failure as a result
        return f"error: {type(exc).__name__}: {exc}"


def schema_valid(name: str, arguments: Any) -> bool:
    """Whether ``arguments`` satisfies the declared required fields."""
    spec = next((tool for tool in TOOLS if tool["name"] == name), None)
    if spec is None or not isinstance(arguments, dict):
        return False
    schema = spec["input_schema"]
    if any(field not in arguments for field in schema["required"]):
        return False
    return all(
        isinstance(arguments[field], str)
        for field in schema["required"]
        if schema["properties"].get(field, {}).get("type") == "string"
    )


def openai_tools() -> list[dict[str, Any]]:
    """The same schemas in OpenAI function-calling form."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        for tool in TOOLS
    ]


def chat_body(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """The body this harness sends today, in its four-system-block shape."""
    return {
        "model": MODEL,
        "messages": [{"role": "system", "content": block} for block in SYSTEM_BLOCKS] + messages,
        "tools": openai_tools(),
        "tool_choice": "auto",
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
        "max_tokens": 8192,
        "stream": False,
    }


def messages_body(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """The body DeepSeek's own harness sends, in its one-system-string shape."""
    return {
        "model": MODEL,
        "system": "\n\n".join(SYSTEM_BLOCKS),
        "messages": messages,
        "tools": TOOLS,
        "thinking": {"type": "enabled"},
        "output_config": {"effort": "high"},
        "max_tokens": 8192,
        "stream": False,
    }


def usage_row(payload: dict[str, Any], wire: str) -> dict[str, Any]:
    """One call's tokens, normalised so both wires report the full prompt."""
    usage = payload.get("usage") or {}
    if wire == "chat":
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        cached = int(usage.get("prompt_cache_hit_tokens") or 0) or int(
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        )
    else:
        reported = int(usage.get("input_tokens") or 0)
        cached = int(usage.get("cache_read_input_tokens") or 0)
        prompt = reported + cached
    return {
        "prompt": prompt,
        "cached": cached,
        "output": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
    }


async def call(client: Any, wire: str, key: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """One model call, returning the raw JSON payload."""
    if wire == "chat":
        url, body = CHAT_URL, chat_body(messages)
        headers = {"Authorization": f"Bearer {key}"}
    else:
        url, body = MESSAGES_URL, messages_body(messages)
        headers = {
            "Authorization": f"Bearer {key}",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        }
    response = await client.post(url, headers=headers, json=body)
    if response.status_code != 200:
        raise RuntimeError(f"{wire} HTTP {response.status_code}: {response.text[:300]}")
    return response.json()


def parse_chat(payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]], str]:
    """Split a chat payload into (text, tool calls, finish reason)."""
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    calls = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = None
        calls.append({"id": raw.get("id"), "name": function.get("name"), "arguments": arguments})
    return message.get("content") or "", calls, str(choice.get("finish_reason") or "")


def parse_messages(payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]], str]:
    """Split a Messages payload into (text, tool calls, stop reason)."""
    text = "".join(
        block.get("text", "")
        for block in payload.get("content") or []
        if block.get("type") == "text"
    )
    calls = [
        {"id": block.get("id"), "name": block.get("name"), "arguments": block.get("input")}
        for block in payload.get("content") or []
        if block.get("type") == "tool_use"
    ]
    return text, calls, str(payload.get("stop_reason") or "")


async def one_run(key: str, wire: str, workspace: Path, max_turns: int) -> dict[str, Any]:
    """Drive one task to completion on one wire and record what it cost."""
    import httpx

    messages: list[dict[str, Any]] = [{"role": "user", "content": TASK}]
    usage: list[dict[str, Any]] = []
    turns = malformed = invalid = tool_calls = 0
    final_text = ""
    finish = ""
    failure = None
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=180) as client:
        while turns < max_turns:
            try:
                payload = await call(client, wire, key, messages)
            except Exception as exc:  # noqa: BLE001 - recorded as the run's outcome
                failure = f"{type(exc).__name__}: {exc}"[:300]
                break
            turns += 1
            usage.append(usage_row(payload, wire))
            if wire == "chat":
                text, calls, finish = parse_chat(payload)
            else:
                text, calls, finish = parse_messages(payload)
            if not calls:
                final_text = text
                break
            for entry in calls:
                tool_calls += 1
                if entry["arguments"] is None:
                    malformed += 1
                elif not schema_valid(str(entry["name"]), entry["arguments"]):
                    invalid += 1
            if wire == "chat":
                messages.append(
                    {
                        "role": "assistant",
                        "content": text,
                        "tool_calls": [
                            {
                                "id": entry["id"],
                                "type": "function",
                                "function": {
                                    "name": entry["name"],
                                    "arguments": json.dumps(entry["arguments"] or {}),
                                },
                            }
                            for entry in calls
                        ],
                    }
                )
                for entry in calls:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": entry["id"],
                            "content": run_tool(str(entry["name"]), entry["arguments"], workspace),
                        }
                    )
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": e["id"],
                                "name": e["name"],
                                "input": e["arguments"],
                            }
                            for e in calls
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": entry["id"],
                                "content": run_tool(
                                    str(entry["name"]), entry["arguments"], workspace
                                ),
                            }
                            for entry in calls
                        ],
                    }
                )

    correct = False
    try:
        produced = json.loads((workspace / "result.json").read_text())
        correct = (
            isinstance(produced, dict)
            and set(produced) == {"total", "rows"}
            and abs(float(produced["total"]) - EXPECTED_TOTAL) < 1e-9
            and produced["rows"] == EXPECTED_ROWS
        )
    except Exception:  # noqa: BLE001 - a missing or malformed artifact is a failure
        pass
    return {
        "wire": wire,
        "turns": turns,
        "tool_calls": tool_calls,
        "malformed": malformed,
        "invalid": invalid,
        "correct": correct,
        "finish": finish if turns else "",
        "failure": failure,
        "elapsed_s": round(time.monotonic() - started, 3),
        "prompt_tokens": sum(row["prompt"] for row in usage),
        "cached_tokens": sum(row["cached"] for row in usage),
        "output_tokens": sum(row["output"] for row in usage),
        "usage": usage,
        "final_text": final_text[:120],
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Medians plus the full per-run values; never a single point estimate."""
    prompts = [row["prompt_tokens"] for row in rows]
    return {
        "wire": rows[0]["wire"],
        "runs": len(rows),
        "turns": [row["turns"] for row in rows],
        "turns_median": statistics.median(row["turns"] for row in rows),
        "correct": sum(1 for row in rows if row["correct"]),
        "malformed": sum(row["malformed"] for row in rows),
        "invalid": sum(row["invalid"] for row in rows),
        "tool_calls": sum(row["tool_calls"] for row in rows),
        "prompt_tokens_median": statistics.median(prompts),
        "prompt_tokens_spread": [min(prompts), max(prompts)],
        "cached_share": round(
            100.0 * sum(row["cached_tokens"] for row in rows) / max(1, sum(prompts)), 1
        ),
        "elapsed_s_median": round(statistics.median(row["elapsed_s"] for row in rows), 2),
        "failures": [row["failure"] for row in rows if row["failure"]],
    }


async def resolve_key() -> str:
    """Read the DeepSeek key without ever printing it."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from local_operator.providers.auth_store import AuthStore

    store = AuthStore(Path.home() / ".local-operator" / "auth.db")
    try:
        resolved = await store.get_api_key("deepseek")
    finally:
        store.close()
    if not resolved:
        raise SystemExit("no DeepSeek credential: set DEEPSEEK_API_KEY or run `lop login`")
    return resolved


async def main(args: argparse.Namespace) -> int:
    """Run the requested arms and print one JSON summary per arm."""
    wires = ["chat", "messages"] if args.wire == "both" else [args.wire]
    if args.dry_run:
        for wire in wires:
            body = chat_body([]) if wire == "chat" else messages_body([])
            print(
                json.dumps(
                    {
                        "wire": wire,
                        "url": CHAT_URL if wire == "chat" else MESSAGES_URL,
                        "body_keys": sorted(body),
                        "system_blocks": len(SYSTEM_BLOCKS) if wire == "chat" else 1,
                        "tools": len(body["tools"]),
                        "runs": args.repeat,
                        "budget_calls": args.repeat * args.max_turns,
                    }
                )
            )
        return 0

    key = await resolve_key()
    root = Path(args.out or tempfile.mkdtemp(prefix="ds-wire-parity-"))
    root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for wire in wires:
        rows = []
        for repeat in range(args.repeat):
            workspace = root / f"ws-{wire}-{repeat}"
            shutil.rmtree(workspace, ignore_errors=True)
            seed(workspace)
            row = await one_run(key, wire, workspace, args.max_turns)
            row["repeat"] = repeat
            rows.append(row)
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "wire",
                            "repeat",
                            "turns",
                            "tool_calls",
                            "malformed",
                            "invalid",
                            "correct",
                            "elapsed_s",
                            "prompt_tokens",
                            "cached_tokens",
                            "output_tokens",
                            "failure",
                        )
                    }
                ),
                flush=True,
            )
        summary = summarise(rows)
        summaries.append(summary)
        print(json.dumps({"summary": summary}), flush=True)
    (root / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps({"results_dir": str(root)}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The command line this benchmark accepts."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--wire", choices=["chat", "messages", "both"], default="both")
    parser.add_argument("--repeat", type=int, default=5, help="runs per wire (default 5)")
    parser.add_argument("--max-turns", type=int, default=8, help="hard turn cap per run")
    parser.add_argument("--out", default="", help="results directory (default: a temp dir)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, send nothing")
    parser.add_argument(
        "--live",
        action="store_true",
        help="required to spend provider calls; mirrors the other bench scripts' safety",
    )
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if not parsed.live and not parsed.dry_run:
        print("refusing to spend real calls without --live (or use --dry-run)", file=sys.stderr)
        raise SystemExit(2)
    if re.search(r"[^\x20-\x7e\n]", TASK):
        raise SystemExit("unexpected non-ASCII in the task text")
    raise SystemExit(asyncio.run(main(parsed)))
