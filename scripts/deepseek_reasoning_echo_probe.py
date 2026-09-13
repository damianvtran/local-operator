#!/usr/bin/env python
"""Live check: DeepSeek's thinking-mode reasoning echo, through the harness.

Not a unit test and not part of CI. Run manually, from a checkout with the fix
applied -- the live half spends real tokens on ``api.deepseek.com`` and needs a
working DeepSeek credential in the local auth store (which this reads at runtime
and never prints).

    .venv/bin/python scripts/deepseek_reasoning_echo_probe.py --digest
    .venv/bin/python scripts/deepseek_reasoning_echo_probe.py
    .venv/bin/python scripts/deepseek_reasoning_echo_probe.py \
        --session ~/.local-operator/sessions/9daa47ece7ad

Two shapes are checked, each TWICE, through ``OpenAICompatClient._build_body``
so that tool schemas, system blocks, native replay and every body extra are the
harness's own rather than a hand-written approximation:

* a minimal synthetic tool loop, and
* (with ``--session``) a real session's failing window, replayed with the
  credential scope its own transcript recorded -- the scope matters, because a
  mismatched one invalidates every stored native payload and is itself one of
  the ways users meet this 400.

The pair is the echo capability turned OFF and ON, otherwise identical. OFF is
what the builder produced before this change; ON is this branch. Expected: the
OFF body refused, the ON body accepted with nothing left blank.

The shape that reproduces the refusal is a request built MID TOOL LOOP -- one
that ends on a tool result, which is where the app builds the request that dies.
Requests that omit the echo are accepted in other shapes (adding a trailing user
turn, or tool-call ids copied from a reply the endpoint itself generated,
answered 200 where the same body 400ed), so the server-side rule is not fully
characterised and this probe does not claim to characterise it: it checks the
shape that fails and that the fill is a measured-safe superset on both. See
``ModelSpec.requires_reasoning_echo`` for the measurements in full.

``--digest`` prints the two bodies' sha256 and blank-turn counts WITHOUT any
network call, which is the reproducible half of the byte-identity claim: run it
under ``PYTHONPATH`` against a clean ``origin/main`` worktree and against this
branch and compare, to show the capability-off body is what ``main`` builds.

Exit status is 0 when every expectation holds, 1 otherwise, so the run can be
quoted as evidence rather than eyeballed.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, cast

import httpx

from local_operator.harness.types import (
    AgentTool,
    ChatRequest,
    Message,
    MessageRole,
    TextContent,
    ToolCall,
)
from local_operator.model.configure import build_model_spec
from local_operator.providers.clients import OpenAICompatClient
from local_operator.providers.replay import REASONING_ECHO_PLACEHOLDER
from local_operator.tools import builtin

DEEPSEEK_URL = "https://api.deepseek.com/v1"
AUTH_DB = Path("~/.local-operator/auth.db").expanduser()


def real_key() -> str:
    """The live DeepSeek credential, from the store -- never echoed anywhere."""
    con = sqlite3.connect(AUTH_DB)
    for (data,) in con.execute(
        "select data from auth_credentials where provider='deepseek' order by id desc"
    ):
        key = json.loads(data)["key"]
        if not key.startswith("sk-fake"):
            return key
    raise SystemExit("no live deepseek credential in the auth store")


def text_of(content: object) -> str:
    if isinstance(content, str):
        try:
            content = ast.literal_eval(content)
        except (ValueError, SyntaxError):
            return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def harness_tools() -> list[AgentTool]:
    """The runtime's own tool schemas, not a stand-in.

    They are load-bearing rather than decorative: the refusal is not reproduced
    without a ``tools`` param on the real history (measured 2026-09-12), so a
    probe that dropped them would report a clean 200 for a body the runtime
    cannot send.
    """
    return [
        builder()
        for builder in (
            builtin.build_bash_tool,
            builtin.build_read_tool,
            builtin.build_edit_tool,
            builtin.build_write_tool,
            builtin.build_todo_tool,
        )
    ]


def synthetic_history() -> list[Message]:
    """The minimal shape that reproduces the refusal: a tool loop whose assistant
    turns carry no ``reasoning_content`` at all (measured 400, repeated)."""
    first = ToolCall(
        id="call_a", name="bash", arguments={"command": "echo one", "i": "Echoing one"}
    )
    second = ToolCall(
        id="call_b", name="bash", arguments={"command": "echo two", "i": "Echoing two"}
    )
    return [
        Message.user("Run `echo one`, then `echo two`, then summarise."),
        Message.assistant("", tool_calls=[first]),
        Message(
            role="tool", tool_call_id="call_a", tool_name="bash", content=[TextContent(text="one")]
        ),
        Message.assistant("One printed.", tool_calls=[second]),
        Message(
            role="tool", tool_call_id="call_b", tool_name="bash", content=[TextContent(text="two")]
        ),
    ]


def session_history(session: Path) -> tuple[list[Message], list[str], str | None]:
    """Rebuild a transcript's request, and the scope its native state was bound to."""
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
        role = payload.get("role")
        text = text_of(payload.get("content"))
        if role == "assistant":
            # Stop at the turn that died on the refusal this check is about: the
            # request that failed is the one built just before it.
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
            # Any other role a transcript carries. Cast rather than coerced: the
            # reconstruction must replay what the row says, not what this script
            # would have written.
            messages.append(Message(role=cast(MessageRole, role), content=[TextContent(text=text)]))
    return messages, system, scope


def report(label: str, status: int, detail: str) -> bool:
    print(f"{status}  {label}  {detail}")
    return status < 400


async def check(
    name: str,
    messages: list[Message],
    system: list[str],
    scope: str | None,
    key: str,
    tools: list[AgentTool],
) -> bool:
    spec = build_model_spec("deepseek", "deepseek-flash")
    client = OpenAICompatClient(DEEPSEEK_URL)
    spec_without_echo = spec.model_copy(update={"requires_reasoning_echo": False})
    ok = True

    print(f"--- {name} ---")
    # Both bodies are BUILT first and posted second, so the ON body's fill count
    # can be compared against the OFF body's blank count -- the whole claim is
    # that the second is exactly the first plus the echo.
    built: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    for label, model in (
        ("echo OFF (pre-fix body)", spec_without_echo),
        ("echo ON (this fix)", spec),
    ):
        body = client._build_body(
            ChatRequest(
                model=model,
                system_blocks=list(system),
                messages=list(messages),
                tools=tools,
            ),
            scope=scope,
        )
        entries = [m for m in body["messages"] if m.get("role") == "assistant"]
        built.append((label, body, entries))
    blanks_before = sum(1 for m in built[0][2] if not str(m.get("reasoning_content") or "").strip())

    for label, body, entries in built:
        blank = [m for m in entries if not str(m.get("reasoning_content") or "").strip()]
        filled = [m for m in entries if m.get("reasoning_content") == REASONING_ECHO_PLACEHOLDER]
        async with httpx.AsyncClient(timeout=300) as http:
            # Posted VERBATIM: the body carries ``stream: true`` and
            # ``stream_options.include_usage``, as the runtime sends it, and
            # forcing ``stream: false`` here is refused for exactly that reason.
            response = await http.post(
                f"{DEEPSEEK_URL}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=body,
            )
        detail = f"messages={len(body['messages'])} assistant={len(entries)} blank={len(blank)}"
        if "echo ON" in label:
            detail += f" placeholders={len(filled)}"
        stopped = report(label, response.status_code, detail)
        if response.status_code >= 400:
            message = response.json().get("error", {}).get("message", "")
            print(f"       {message[:120]}")
        if label.startswith("echo OFF"):
            # The pre-fix body is the one the provider REFUSES, and it must have
            # something for the fix to fill in.
            ok = ok and not stopped and bool(blank)
        else:
            # And the fix fills exactly those turns, leaving none blank.
            ok = ok and stopped and not blank and len(filled) == blanks_before
    return ok


def digest(name: str, messages: list[Message], system: list[str], scope: str | None) -> None:
    """The two bodies' identity and shape, with no network and no credential.

    This is the reproducible half of the byte-identity claim: the OFF body is
    what the pre-fix builder produces, so running this under ``PYTHONPATH``
    against a clean ``origin/main`` worktree and against this branch must print
    the same sha256 for the same session.
    """
    client = OpenAICompatClient(DEEPSEEK_URL)
    spec = build_model_spec("deepseek", "deepseek-flash")
    print(f"--- {name} ---")
    for label, model in (
        ("echo OFF", spec.model_copy(update={"requires_reasoning_echo": False})),
        ("echo ON", spec),
    ):
        body = client._build_body(
            ChatRequest(
                model=model,
                system_blocks=list(system),
                messages=list(messages),
                tools=harness_tools(),
            ),
            scope=scope,
        )
        entries = [m for m in body["messages"] if m.get("role") == "assistant"]
        blank = sum(1 for m in entries if not str(m.get("reasoning_content") or "").strip())
        blob = json.dumps(body, sort_keys=True).encode()
        print(
            f"{label}  sha256={hashlib.sha256(blob).hexdigest()}  "
            f"messages={len(body['messages'])} assistant={len(entries)} blank={blank}"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=None, help="a transcript directory")
    parser.add_argument(
        "--digest",
        action="store_true",
        help="print the two bodies' sha256 and stop, with no network call",
    )
    args = parser.parse_args()

    if args.digest:
        digest("minimal synthetic tool loop", synthetic_history(), ["You are terse."], None)
        if args.session is not None:
            messages, system, scope = session_history(args.session)
            digest(f"real session {args.session.name}", messages, system, scope)
        return

    key = real_key()

    results = [
        await check(
            "minimal synthetic tool loop",
            synthetic_history(),
            ["You are terse."],
            None,
            key,
            harness_tools(),
        )
    ]
    if args.session is not None:
        messages, system, scope = session_history(args.session)
        print(f"recorded credential scope present: {bool(scope)}")
        results.append(
            await check(
                f"real session {args.session.name}", messages, system, scope, key, harness_tools()
            )
        )
    print("PASS" if all(results) else "FAIL")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
