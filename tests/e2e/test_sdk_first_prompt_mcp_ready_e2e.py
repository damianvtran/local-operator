"""An SDK session's FIRST ``prompt()`` reaches an undeclared, still-connecting MCP server.

The regression this pins: ``sdk.open_session`` (in-process owner, no declared
tools) followed at once by ``prompt()`` raced MCP discovery. Measured against
the slow fixture in fresh roots: the first provider request left 0.36 s after
``prompt()`` with the server still ``connecting``, and the model's
``read mcp://slowfix/search`` came back "Unknown tool 'search' for MCP server
'slowfix'", then its call "Tool not found". The #4328 gate waited only for
DECLARED names, so an undeclared SDK session waited on nothing. The fix
(``SessionSpec.mcp_ready_timeout_s``, default 30 s, applied through
``Session.set_first_turn_mcp_wait``) holds the first turn, bounded, for every
configured server that is still connecting.

Only the remote model is scripted: a loopback OpenAI-compatible provider that
records each wire body and follows guide://mcp's lazy-tool flow (read the
tool's ``mcp://`` URL, then call it). The SDK facade, ``create_session``, the
MCP manager, the stdio transport and the fixture server process are
production code; HOME and the config dir are fresh per test, so no tool cache
exists.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator import sdk
from local_operator.config import ConfigManager
from local_operator.session.spec import ApprovalPolicy, SessionRoots, SessionSpec

pytestmark = pytest.mark.e2e

FIXTURE = Path(__file__).with_name("slow_mcp_fixture.py")
TOOL = "mcp__slowfix_search"
DEFAULT_STEPS: list[tuple[str, dict[str, Any]]] = [
    ("read", {"path": "mcp://slowfix/search"}),
    (TOOL, {"query": "judges"}),
]
BIG_TOOL = "mcp__slowfix_bigsearch"
BIG_STEPS: list[tuple[str, dict[str, Any]]] = [
    ("read", {"path": "mcp://slowfix/bigsearch"}),
    (BIG_TOOL, {"query": "judges"}),
]


@pytest.fixture
def sdk_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fresh roots, a recording loopback provider, and the slow fixture server.

    Yields ``open(delay_s, **spec) -> async context manager`` plus the request
    log. The fixture's delay rides the server's config ``env``: the stdio child
    environment is an allowlist (``mcp.client.stdio.get_default_environment``
    plus config ``env``), so a variable set on this process never reaches it.
    """
    from local_operator import agent_shell

    home = tmp_path / "home"
    root = home / ".local-operator"
    agent_home = home / "agent-home"
    work = home / "work"
    for directory in (root, agent_home, work):
        directory.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    # A parent ``lop`` exports these; the SDK reads neither from its caller.
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name)
    monkeypatch.delenv(agent_shell.AGENT_SHELL_ENV, raising=False)
    monkeypatch.delenv(agent_shell.MAY_DELEGATE_ENV, raising=False)
    monkeypatch.delenv(agent_shell.ALLOW_NESTED_SESSION_ENV, raising=False)
    requests: list[dict[str, Any]] = []
    #: The scripted model's tool calls, in order; ``open_with(steps=...)`` swaps
    #: it per test. Counted against the conversation's tool results, so a later
    #: prompt (every step answered) just ends with "done".
    script: list[tuple[str, dict[str, Any]]] = list(DEFAULT_STEPS)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002 — the stdlib signature
            pass

        def do_GET(self):
            body = json.dumps({"object": "list", "data": [{"id": "fx", "object": "model"}]})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            answered = sum(1 for m in body.get("messages", []) if m.get("role") == "tool")
            steps = script
            delta: dict[str, Any] = {"role": "assistant", "content": "done"}
            finish = "stop"
            if answered < len(steps):
                name, arguments = steps[answered]
                call = {
                    "index": 0,
                    "id": f"call_{answered}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
                delta, finish = {"role": "assistant", "tool_calls": [call]}, "tool_calls"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk_delta, chunk_finish in [(delta, None), ({}, finish)]:
                chunk = {
                    "id": "fx",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "fx",
                    "choices": [{"index": 0, "delta": chunk_delta, "finish_reason": chunk_finish}],
                }
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    ConfigManager(root).update_config(
        {
            "hosting": "openai-compatible",
            "model_name": "fx",
            "providers": {
                "openai-compatible": {
                    "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                    "models": {"fx": {"context_window": 100000, "supports_tools": True}},
                }
            },
        }
    )
    roots = SessionRoots(config_dir=root, agent_home=agent_home, cwd=work, allow_volatile=True)

    def open_with(
        delay_s: float,
        *,
        steps: list[tuple[str, dict[str, Any]]] | None = None,
        big_schema_chars: int = 0,
        **spec: Any,
    ):
        script[:] = DEFAULT_STEPS if steps is None else steps
        env = {"SLOW_MCP_DELAY_S": str(delay_s)}
        if big_schema_chars:
            env["SLOW_MCP_BIG_SCHEMA_CHARS"] = str(big_schema_chars)
        (root / "mcp.json").write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "slowfix": {
                            "command": sys.executable,
                            "args": [str(FIXTURE)],
                            "env": env,
                        }
                    }
                }
            )
        )
        return sdk.open_session(
            SessionSpec(
                hosting="openai-compatible",
                model="fx",
                approvals=ApprovalPolicy.auto(),
                **spec,
            ),
            roots=roots,
        )

    try:
        yield open_with, requests
    finally:
        server.shutdown()


def _tool_results(requests: list[dict[str, Any]]) -> list[str]:
    """Every tool result of the run, in order (the LAST request carries them all)."""
    if not requests:
        return []
    return [str(m.get("content")) for m in requests[-1]["messages"] if m.get("role") == "tool"]


@pytest.mark.asyncio
async def test_the_default_first_prompt_reaches_an_undeclared_slow_server(sdk_root) -> None:
    open_with, requests = sdk_root
    async with open_with(1.5) as opened:
        session = cast(Any, opened)
        manager = session.mcp_manager
        assert (
            manager.get_connection_status("slowfix") == "connecting"
        ), "precondition: the server missed the 250 ms startup gate"
        await session.prompt("search for judges")
        assert manager.get_connection_status("slowfix") == "connected"

    results = _tool_results(requests)
    assert len(results) == 2, results
    assert results[0].startswith("# MCP tool ready: mcp__slowfix_search"), results[0]
    assert results[1] == "fixture result for judges", results[1]


@pytest.mark.asyncio
async def test_a_server_still_connecting_at_the_bound_refuses_and_a_retry_meets_the_gate(
    sdk_root,
) -> None:
    """Honest failure, not a zero-tools first turn: the bound refuses BEFORE any
    request, discovery keeps running, ``wait_for_tools`` then sees the lazy
    server's tool (it used to read ``missing`` here: lazy tools never entered
    the enforced inventory), and the retried prompt meets the gate again."""
    from local_operator.session.declared_mcp_ready import McpServersNotReady

    open_with, requests = sdk_root
    # 4 s of fixture sleep against a 0.3 s bound: the server cannot arrive in time.
    async with open_with(4.0, mcp_ready_timeout_s=0.3) as opened:
        session = cast(Any, opened)
        with pytest.raises(McpServersNotReady) as caught:
            await session.prompt("search for judges")
        assert caught.value.servers == {"slowfix": "connecting"}
        assert requests == [], "nothing may be sent once the bound refused the first prompt"
        assert (
            session.mcp_manager.get_connection_status("slowfix") == "connecting"
        ), "the bound refuses the turn; it never cancels discovery"

        readiness = await session.wait_for_tools([TOOL], timeout_s=30.0)
        assert readiness.status == "ready", readiness

        await session.prompt("search for judges")

    results = _tool_results(requests)
    assert results[-1] == "fixture result for judges", results


def _duplicate_tool_names(requests: list[dict[str, Any]]) -> list[dict[str, int]]:
    """Per request, every tool name its ``tools`` array carries more than once.

    Anthropic refuses such a request outright (400 "Tool names must be
    unique"); the loopback provider here accepts it, so the test has to look.
    """
    out: list[dict[str, int]] = []
    for body in requests:
        counts = Counter(tool["function"]["name"] for tool in body.get("tools") or ())
        out.append({name: count for name, count in counts.items() if count > 1})
    return out


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["undeclared_wait_for_tools", "declared_with_read"])
async def test_an_over_cap_tool_granted_and_then_read_is_published_once(sdk_root, shape) -> None:
    """PR #4349 agent review round 1, R1: a GRANT of a lazy MCP tool followed by
    the model's ``mcp://`` read of the same over-cap tool put two copies of the
    name in the inventory, and every later request in the session carried both.

    - ``undeclared_wait_for_tools``: the SDK's default shape through the gate
      SDK.md recommends. The grant used to bypass the factory's activation
      bookkeeping, so the read's ``activate`` added a second copy (reviewer's
      repro: inventory count 1 → 2, prompt 2's ``tools`` listing it twice).
    - ``declared_with_read``: the pre-existing route. A declaration grants the
      tool directly; the model's read then publishes it again. Fixed by
      ``refresh_selected`` keeping one copy per name.
    """
    open_with, requests = sdk_root
    spec: dict[str, Any] = {}
    if shape == "declared_with_read":
        spec["tools"] = ("read", BIG_TOOL)
    # 20,000 chars of parameter description: past the 16,000-char inline cap, so
    # the read takes the publication path (``activate``), not the deferred one.
    async with open_with(0.0, steps=BIG_STEPS, big_schema_chars=20_000, **spec) as opened:
        session = cast(Any, opened)
        if shape == "undeclared_wait_for_tools":
            readiness = await session.wait_for_tools([BIG_TOOL], timeout_s=30.0)
            assert readiness.status == "ready", readiness
        await session.prompt("search for judges")
        assert session.tool_inventory.count(BIG_TOOL) == 1, session.tool_inventory
        await session.prompt("search again")
        assert session.tool_inventory.count(BIG_TOOL) == 1, session.tool_inventory

    results = _tool_results(requests)
    assert results[0].startswith(f"# Enabled MCP tool: {BIG_TOOL}"), (
        "precondition: the read took the over-cap publication path",
        results[0][:300],
    )
    assert "too large to show inline" in results[0], results[0][:300]
    assert results[1] == "big fixture result for judges", results
    # Prompt 1 published the grant; prompt 2 is the request the duplicate reached.
    assert len(requests) >= 3, len(requests)
    assert all(
        BIG_TOOL in [tool["function"]["name"] for tool in body.get("tools") or ()]
        for body in requests
    ), "the granted tool stays published on every request"
    assert _duplicate_tool_names(requests) == [{}] * len(requests)
