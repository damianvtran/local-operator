#!/usr/bin/env python
"""Digest of the DeepSeek chat body built from a session, for tree-to-tree comparison.

Run it twice -- once with ``PYTHONPATH`` pointing at a clean ``origin/main``
worktree and once at this branch -- to prove that the capability-off body is
what the unfixed builder produces, byte for byte, rather than a claim in a
comment:

    git worktree add --detach /tmp/lo-before origin/main
    PYTHONPATH=/tmp/lo-before <this-venv>/bin/python \
        docs/evidence/deepseek-reasoning-echo/body_digest.py \
        --session ~/.local-operator/sessions/9daa47ece7ad
    PYTHONPATH=<this-checkout> <this-venv>/bin/python \
        docs/evidence/deepseek-reasoning-echo/body_digest.py \
        --session ~/.local-operator/sessions/9daa47ece7ad

``PYTHONPATH`` precedes the editable install's generated finder, so the first
command really does run main's ``local_operator`` (the script prints the file it
imported, so the run cannot lie about which tree answered).

The body is built with ``requires_reasoning_echo`` turned OFF where the field
exists, which is the pre-fix render on every tree. It needs no credential and
makes no network call.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import local_operator
from local_operator.harness.types import (
    ChatRequest,
    Message,
    ModelSpec,
    TextContent,
    ToolCall,
)
from local_operator.model.configure import build_model_spec
from local_operator.providers.clients import OpenAICompatClient
from local_operator.tools import builtin

DEEPSEEK_URL = "https://api.deepseek.com/v1"


def text_of(content: object) -> str:
    if isinstance(content, str):
        try:
            content = ast.literal_eval(content)
        except (ValueError, SyntaxError):
            return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def load(session: Path) -> tuple[list[Message], list[str], str | None]:
    rows = [json.loads(line) for line in (session / "transcript.jsonl").open()]
    messages: list[Message] = []
    system: list[str] = []
    scope: str | None = None
    for row in rows:
        payload = row.get("payload") or {}
        if row.get("type") == "custom":
            if payload.get("custom_type") == "system_prefix":
                system = list((payload.get("details") or {}).get("blocks") or [])
            continue
        native = (payload.get("provider_payload") or {}).get("native_replay")
        if isinstance(native, dict) and native.get("credential_scope"):
            scope = scope or native["credential_scope"]
        role, text = payload.get("role"), text_of(payload.get("content"))
        if role == "assistant":
            if payload.get("stop_reason") == "error":
                break
            messages.append(
                Message(
                    role="assistant",
                    content=[TextContent(text=text)] if text else [],
                    tool_calls=[
                        ToolCall(id=c["id"], name=c["name"], arguments=c.get("arguments") or {})
                        for c in payload.get("tool_calls") or []
                    ],
                    stop_reason=payload.get("stop_reason"),
                    provider_payload=payload.get("provider_payload"),
                )
            )
        elif role == "tool":
            messages.append(
                Message(
                    role="tool",
                    content=[TextContent(text=text)],
                    tool_call_id=payload.get("tool_call_id"),
                    tool_name=payload.get("tool_name"),
                )
            )
        else:
            messages.append(Message(role=role, content=[TextContent(text=text)]))
    return messages, system, scope


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()

    messages, system, scope = load(args.session)
    spec: ModelSpec = build_model_spec("deepseek", "deepseek-flash")
    if "requires_reasoning_echo" in type(spec).model_fields:
        spec = spec.model_copy(update={"requires_reasoning_echo": False})
    tools = [
        builder()
        for builder in (
            builtin.build_bash_tool,
            builtin.build_read_tool,
            builtin.build_edit_tool,
            builtin.build_write_tool,
            builtin.build_todo_tool,
        )
    ]
    body = OpenAICompatClient(DEEPSEEK_URL)._build_body(
        ChatRequest(model=spec, system_blocks=list(system), messages=messages, tools=tools),
        scope=scope,
    )
    entries = [m for m in body["messages"] if m.get("role") == "assistant"]
    print("tree:", local_operator.__file__)
    print("sha256:", hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest())
    print(
        "messages:",
        len(body["messages"]),
        "assistant:",
        len(entries),
        "blank_reasoning:",
        sum(1 for m in entries if not str(m.get("reasoning_content") or "").strip()),
    )


if __name__ == "__main__":
    main()
