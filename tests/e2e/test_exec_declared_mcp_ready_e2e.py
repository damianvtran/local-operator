"""A cold ``lop exec --tools mcp__…`` run sends its declared tools on the FIRST request.

The regression this pins (head-to-head benchmark, 2026-11-20, caveat 1): in a
fresh config dir there is no cached tool list, a cold Python stdio server
misses the MCP manager's 250 ms startup gate, and the first provider request
went out with NO tools — the model wrote its tool calls as prose and the run
returned nulls. The fix (``session/declared_mcp_ready.py``) makes the first
turn wait, bounded, for the servers the declaration names.

Only the remote model is scripted (a loopback OpenAI-compatible provider that
records each wire body). The CLI, config, MCP manager, the stdio transport and
the fixture server process are production code; HOME and the config dir are
fresh per test, so no tool cache exists, and no ambient ``CMUX_*`` id leaks in.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from local_operator.config import ConfigManager
from tests.e2e.harness import NO_NOTIFY_ENV

pytestmark = pytest.mark.e2e

FIXTURE = Path(__file__).with_name("slow_mcp_fixture.py")
MARKER = "DECLARED_MCP_READY_FIXTURE"
FETCH_MARKER = "FETCH_THE_PAGE"
DECLARED = ("mcp__slowfix_search", "mcp__slowfix_fetch")


@pytest.fixture
def cold_exec(tmp_path, monkeypatch):
    home = tmp_path / "home"
    root = home / ".local-operator"
    root.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            body = json.dumps(
                {"object": "list", "data": [{"id": "exec-fixture", "object": "model"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            offered = {tool["function"]["name"] for tool in body.get("tools") or []}
            answered = any(m.get("role") == "tool" for m in body.get("messages", []))
            delta: dict[str, Any] = {"role": "assistant", "content": "done"}
            finish_reason = "stop"
            if FETCH_MARKER in json.dumps(body.get("messages")) and not answered:
                if "mcp__slowfix_fetch" in offered:
                    call = {
                        "index": 0,
                        "id": "call_fetch",
                        "type": "function",
                        "function": {
                            "name": "mcp__slowfix_fetch",
                            "arguments": json.dumps({"url": "https://example.test/judges"}),
                        },
                    }
                    delta = {"role": "assistant", "tool_calls": [call]}
                    finish_reason = "tool_calls"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for delta, finish in [(delta, None), ({}, finish_reason)]:
                chunk = {
                    "id": "fixture",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "exec-fixture",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    ConfigManager(root).update_config(
        {
            "hosting": "openai-compatible",
            "model_name": "exec-fixture",
            "providers": {
                "openai-compatible": {
                    "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                    "models": {"exec-fixture": {"context_window": 100000, "supports_tools": True}},
                }
            },
        }
    )
    # NO mcp_cache.db: a cold worker. The server is the slow fixture, launched
    # by the same interpreter so it imports the same ``mcp`` SDK.
    (root / "mcp.json").write_text(
        json.dumps({"mcpServers": {"slowfix": {"command": sys.executable, "args": [str(FIXTURE)]}}})
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("CMUX_") and k != "NO_COLOR"}
    env["TERM"] = "xterm-256color"
    env.update(NO_NOTIFY_ENV)
    cwd = tmp_path / "cwd"
    cwd.mkdir()

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [sys.executable, "-m", "local_operator.cli", *args],
            env=env,
            stdin=subprocess.DEVNULL,
            text=True,
            capture_output=True,
            timeout=90,
            cwd=cwd,
        )
        print("COMMAND", "lop", *args, "EXIT", result.returncode)
        print("STDOUT", result.stdout)
        print("STDERR", result.stderr)
        return result

    try:
        yield run, requests, root
    finally:
        server.shutdown()
        server.server_close()
        serving.join(timeout=5)


def _turn_requests(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The run's own turn requests, in order, without the side channels.

    Conversation naming also carries the prompt as a user row (measured: two
    ``Name this conversation…`` calls precede the turn), and it never offers
    tools by design, so it is excluded by its own system line rather than by
    position.
    """
    turns = []
    for body in requests:
        messages = body.get("messages", [])
        system = json.dumps(messages[0].get("content")) if messages else ""
        carries_prompt = any(
            message.get("role") == "user" and MARKER in json.dumps(message.get("content"))
            for message in messages
        )
        if carries_prompt and "Name this conversation" not in system:
            turns.append(body)
    return turns


def _tool_names(body: dict[str, Any]) -> list[str]:
    return sorted(tool["function"]["name"] for tool in body.get("tools") or [])


def test_cold_exec_first_request_carries_the_declared_mcp_tools(cold_exec):
    run, requests, root = cold_exec
    assert not (root / "mcp_cache.db").exists(), "precondition: no warmed tool cache"

    result = run("exec", "--tools", ",".join(DECLARED), "--yolo", f"{MARKER}: look it up")

    assert result.returncode == 0, result.stderr
    turns = _turn_requests(requests)
    assert turns, f"no turn request reached the provider: {requests!r}"
    # THE property: the FIRST request — not a later one after the server
    # settled — already offers both declared tools.
    print("EVIDENCE", f"turn_requests={len(turns)}", f"first_tools={_tool_names(turns[0])}")
    assert _tool_names(turns[0]) == sorted(DECLARED)
    # And nothing reported them unreachable at the end of the run.
    assert "mcp__slowfix" not in result.stderr


def _tool_result_text(requests: list[dict[str, Any]]) -> str:
    """The MCP fetch result as the model was sent it, on the follow-up request."""
    for body in requests:
        for message in body.get("messages", []):
            if message.get("role") == "tool" and message.get("tool_call_id") == "call_fetch":
                content = message.get("content")
                if isinstance(content, list):
                    return "".join(part.get("text", "") for part in content)
                return str(content)
    raise AssertionError(f"no fetch result reached the provider: {len(requests)} requests")


@pytest.mark.parametrize("with_read", [False, True])
def test_a_large_mcp_result_is_shaped_by_what_the_run_can_read(cold_exec, with_read):
    """No ``read`` -> a 12,000-char result (bd_tools' cap) arrives WHOLE with no
    ``spill://`` route; with ``read`` -> today's 8 KiB cut plus the footer that
    names the handle. The benchmark's no-read arms spent tool calls fetching
    unreachable handles and re-searching instead."""
    run, requests, _root = cold_exec
    declared = ["mcp__slowfix_fetch"] + (["read"] if with_read else [])

    result = run("exec", "--tools", ",".join(declared), "--yolo", f"{MARKER} {FETCH_MARKER}")

    assert result.returncode == 0, result.stderr
    text = _tool_result_text(requests)
    print("EVIDENCE", f"with_read={with_read}", f"chars={len(text)}", f"tail={text[-160:]!r}")
    if with_read:
        assert "spill://" in text and 'read(path="spill://' in text
        assert len(text) < 12_000, "cut at the 8 KiB cap behind its footer, as before"
    else:
        assert "spill://" not in text and "read(" not in text
        assert "BEGIN" in text and text.endswith("END-OF-PAGE")
        assert len(text) == 12_000


def _write_servers(root: Path, servers: dict[str, dict[str, Any]]) -> None:
    (root / "mcp.json").write_text(json.dumps({"mcpServers": servers}))


def _slow(delay_s: float) -> dict[str, Any]:
    return {
        "command": sys.executable,
        "args": [str(FIXTURE)],
        "env": {"SLOW_MCP_DELAY_S": str(delay_s)},
    }


def test_a_wedged_prefix_sibling_does_not_hold_or_fail_the_declared_tool(cold_exec):
    """Review R1 / QA Q4r, on the real path: ``slowfix`` (wedged) is an
    ambiguous owner of ``mcp__slowfix_x_search``, which ``slowfix_x`` serves
    after ~1 s. The status-keyed gate waited the whole 30 s on ``slowfix`` and
    then failed the run (``declared tools unreachable:`` with an EMPTY list)."""
    import time

    run, requests, root = cold_exec
    _write_servers(root, {"slowfix": _slow(120), "slowfix_x": _slow(1)})
    started = time.monotonic()
    result = run("exec", "--tools", "mcp__slowfix_x_search", "--yolo", f"{MARKER}: q4r")
    elapsed = time.monotonic() - started
    turns = _turn_requests(requests)
    print("EVIDENCE", f"elapsed={elapsed:.1f}s", f"exit={result.returncode}")
    print("EVIDENCE", f"turn_requests={len(turns)}", f"first_tools={_tool_names(turns[0])}")
    assert result.returncode == 0, result.stderr
    assert _tool_names(turns[0]) == ["mcp__slowfix_x_search"]
    assert elapsed < 25, "the wedged sibling held the turn toward the 30 s bound"


@pytest.mark.parametrize("shape", ["bad-command", "instant-crash"])
def test_a_declared_server_that_cannot_start_fails_loudly_before_any_turn(cold_exec, shape):
    """QA Q-F2 / Q2a, on the real path: a declared server whose command does
    not exist had ALREADY failed when the first turn checked, so the gate
    neither waited nor raised and exec sent a turn with NO tools, exit 0. It
    must fail non-zero at once, name the tool and the reason, send no turn,
    and print no ERROR traceback ahead of ``exec failed:`` (review N1, QA O3)."""
    run, requests, root = cold_exec
    server: dict[str, Any] = (
        {"command": str(root / "no-such-mcp-server")}  # QA Q2a
        if shape == "bad-command"
        else {"command": sys.executable, "args": ["-I", "-S", "-c", "raise SystemExit(3)"]}  # Q2c
    )
    _write_servers(root, {"ghost": server})
    result = run("exec", "--tools", "mcp__ghost_search", "--yolo", f"{MARKER}: {shape}")
    failure = [line for line in result.stderr.splitlines() if line.startswith("exec failed:")]
    print(
        "EVIDENCE",
        shape,
        f"exit={result.returncode}",
        f"turn_requests={len(_turn_requests(requests))}",
    )
    print("EVIDENCE", failure)
    assert result.returncode != 0
    assert _turn_requests(requests) == [], "nothing may be sent without the declared reach"
    assert len(failure) == 1 and "mcp__ghost_search" in failure[0]
    assert "ghost failed: disconnected" in failure[0]
    assert "Traceback" not in result.stderr
