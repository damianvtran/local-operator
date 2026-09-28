"""Forwarded Claude Code / Codex PostToolUse hooks (``local_operator.hook_forwarding``).

The claim is "a hook written for Claude Code behaves the same under lop", so the
end-to-end cases run a copy of a real operator hook (``fixtures/hooks/
pr-opened-fix-reviews.sh``) through the REAL ``AgentLoop`` and assert on what
the NEXT model request carries.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator import hook_forwarding as hf
from local_operator.harness.loop import AgentLoop, LoopContext
from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    LoopConfig,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamToolCallDelta,
    TextContent,
    ToolContext,
    ToolResult,
)

FIXTURE = Path(__file__).parent / "fixtures" / "hooks" / "pr-opened-fix-reviews.sh"
MODEL = ModelSpec(provider="test", model_id="m")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated home with both sources switched on."""
    monkeypatch.setattr(hf, "forwarding_enabled", lambda: (True, True))
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".codex").mkdir(parents=True)
    return home


def _settings(path: Path, hooks: dict[str, Any], **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"hooks": hooks, **extra}))


def _cmd(
    command: str, matcher: str | None = None, timeout: float | None = None
) -> dict[str, Any]:
    entry: dict[str, Any] = {"type": "command", "command": command}
    if timeout is not None:
        entry["timeout"] = timeout
    group: dict[str, Any] = {"hooks": [entry]}
    if matcher is not None:
        group["matcher"] = matcher
    return group


# ---------------------------------------------------------------------------
# Config merging
# ---------------------------------------------------------------------------


def test_sources_merge_user_project_local_plugin_and_codex(
    home: Path, tmp_path: Path
) -> None:
    project = tmp_path / "proj"
    (project / ".git").mkdir(parents=True)
    sub = project / "pkg"
    sub.mkdir()
    plugin = tmp_path / "plugin"
    _settings(
        plugin / "hooks" / "hooks.json",
        {"PostToolUse": [_cmd("${CLAUDE_PLUGIN_ROOT}/p.sh")]},
    )
    (home / ".claude" / "plugins").mkdir()
    (home / ".claude" / "plugins" / "installed_plugins.json").write_text(
        json.dumps(
            {
                "plugins": {
                    "p@m": [{"scope": "user", "installPath": str(plugin)}],
                    "off@m": [{"scope": "user", "installPath": str(tmp_path / "x")}],
                }
            }
        )
    )
    _settings(
        home / ".claude" / "settings.json",
        {"PostToolUse": [_cmd("user", "Bash")]},
        enabledPlugins={"p@m": True, "off@m": False},
    )
    _settings(project / ".claude" / "settings.json", {"PostToolUse": [_cmd("project")]})
    _settings(
        project / ".claude" / "settings.local.json", {"PostToolUse": [_cmd("local")]}
    )
    _settings(home / ".codex" / "hooks.json", {"PostToolUse": [_cmd("codex")]})

    hooks = hf.load_hook_commands(str(sub), claude=True, codex=True, home=home)
    commands = [h.command for h in hooks]
    assert commands == [
        "user",
        "project",
        "local",
        "${CLAUDE_PLUGIN_ROOT}/p.sh",
        "codex",
    ]
    assert hooks[3].plugin_root == str(plugin)

    only_codex = hf.load_hook_commands(str(sub), claude=False, codex=True, home=home)
    assert [h.command for h in only_codex] == ["codex"]


def test_disable_all_hooks_and_bad_files_are_tolerated(
    home: Path, tmp_path: Path
) -> None:
    (home / ".claude" / "settings.json").write_text("{not json")
    _settings(
        home / ".codex" / "hooks.json", {"PostToolUse": [{"hooks": [{"type": "http"}]}]}
    )
    assert (
        hf.load_hook_commands(str(tmp_path), claude=True, codex=True, home=home) == []
    )
    _settings(
        home / ".claude" / "settings.json",
        {"PostToolUse": [_cmd("x")]},
        disableAllHooks=True,
    )
    assert (
        hf.load_hook_commands(str(tmp_path), claude=True, codex=False, home=home) == []
    )


def test_timeout_defaults_to_claudes_600_seconds(home: Path, tmp_path: Path) -> None:
    _settings(
        home / ".claude" / "settings.json",
        {"PostToolUse": [_cmd("a"), _cmd("b", timeout=7)]},
    )
    hooks = hf.load_hook_commands(str(tmp_path), claude=True, codex=False, home=home)
    assert [h.timeout_s for h in hooks] == [600.0, 7.0]


# ---------------------------------------------------------------------------
# Matchers and tool-name mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("matcher", "value", "expected"),
    [
        (None, "Bash", True),
        ("*", "Bash", True),
        ("", "Bash", True),
        ("Bash", "Bash", True),
        ("Bash", "BashOutput", False),
        ("Edit|Write", "Write", True),
        ("Edit, Write", "Edit", True),
        ("apply_patch|Write|Edit|MultiEdit", "Edit", True),
        ("mcp__.*", "mcp__gh__pr", True),
        ("^Edit$", "NotebookEdit", False),
        ("[", "Bash", False),
    ],
)
def test_matcher_semantics_mirror_claude(
    matcher: str | None, value: str, expected: bool
) -> None:
    assert hf.matcher_matches(matcher, value) is expected


def test_lop_tools_map_to_claude_names_and_inputs(tmp_path: Path) -> None:
    # Claude's Bash input shape: timeout in ms, run_in_background always present.
    assert hf.claude_tool("bash", {"command": "ls", "timeout": 5}, str(tmp_path)) == (
        "Bash",
        {"command": "ls", "timeout": 5000, "run_in_background": False},
    )
    name, tool_input = hf.claude_tool(
        "write", {"path": "a.txt", "content": "x"}, str(tmp_path)
    )
    assert name == "Write"
    assert tool_input == {"file_path": str(tmp_path / "a.txt"), "content": "x"}
    one = hf.claude_tool(
        "edit", {"path": "a", "old_text": "o", "new_text": "n"}, str(tmp_path)
    )
    assert (
        one[0] == "Edit" and one[1]["old_string"] == "o" and one[1]["new_string"] == "n"
    )
    many = hf.claude_tool(
        "edit",
        {
            "path": "a",
            "edits": [
                {"old_text": "1", "new_text": "2"},
                {"old_text": "3", "new_text": "4"},
            ],
        },
        str(tmp_path),
    )
    assert many[0] == "MultiEdit" and len(many[1]["edits"]) == 2
    assert hf.claude_tool("mcp__x__y", {"a": 1}, str(tmp_path)) == (
        "mcp__x__y",
        {"a": 1},
    )


def test_payload_carries_agent_fields_only_for_subagents(tmp_path: Path) -> None:
    main = hf.HookIdentity(session_id="s", cwd=str(tmp_path))
    child = hf.HookIdentity(
        session_id="s", cwd=str(tmp_path), agent_id="job1", agent_type="coder"
    )

    def payload(event: str, who: hf.HookIdentity) -> dict[str, Any]:
        return hf.build_payload(
            event, who, "Bash", {"command": "x"}, tool_use_id="c1", output="out", duration_s=0.5
        )

    p_main = payload(hf.POST_TOOL_USE, main)
    p_child = payload(hf.POST_TOOL_USE, child)
    assert "agent_id" not in p_main and "agent_type" not in p_main
    assert p_child["agent_id"] == "job1" and p_child["agent_type"] == "coder"
    assert p_main["hook_event_name"] == "PostToolUse"
    assert (
        p_main["tool_response"] == {"stdout": "out", "stderr": "", "interrupted": False}
        and p_main["duration_ms"] == 500
    )
    failed = payload(hf.POST_TOOL_USE_FAILURE, main)
    assert failed["error"] == "out" and "tool_response" not in failed


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


def _run(stdout: str = "", stderr: str = "", code: int = 0) -> hf.HookRun:
    return hf.HookRun(exit_code=code, stdout=stdout, stderr=stderr)


def test_output_contract() -> None:
    ctx = json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": "A",
            }
        }
    )
    assert hf.interpret(_run(ctx), hf.POST_TOOL_USE) == ["A"]
    block = json.dumps({"decision": "block", "reason": "R"})
    assert hf.interpret(_run(block), hf.POST_TOOL_USE) == ["R"]
    # Plain stdout on a tool event is debug-log only in Claude Code.
    assert hf.interpret(_run("hello"), hf.POST_TOOL_USE) == []
    # Exit 2: stderr reaches the model; any other failure is silent.
    assert hf.interpret(_run(stderr="warn", code=2), hf.POST_TOOL_USE) == ["warn"]
    assert hf.interpret(_run(stderr="boom", code=1), hf.POST_TOOL_USE) == []
    assert hf.interpret(_run("{broken}"), hf.POST_TOOL_USE) == []
    assert (
        hf.interpret(hf.HookRun(None, "", "", timed_out=True), hf.POST_TOOL_USE) == []
    )
    long = json.dumps({"hookSpecificOutput": {"additionalContext": "x" * 20_000}})
    assert len(hf.interpret(_run(long), hf.POST_TOOL_USE)[0]) < 10_100


@pytest.mark.asyncio
async def test_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    marker = tmp_path / "child.pid"
    hook = hf.HookCommand(
        event=hf.POST_TOOL_USE,
        matcher=None,
        command=f"sleep 30 & echo $! > {marker}; wait",
        timeout_s=0.5,
        source="test",
    )
    started = time.monotonic()
    run = await hf.run_hook(hook, {}, str(tmp_path))
    assert run.timed_out and time.monotonic() - started < 10
    child = int(marker.read_text())
    for _ in range(50):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.05)
    else:
        pytest.fail("the hook's background child survived the timeout")


@pytest.mark.asyncio
async def test_hook_env_exports_claude_placeholders(tmp_path: Path) -> None:
    hook = hf.HookCommand(
        event=hf.POST_TOOL_USE,
        matcher=None,
        command='printf "%s|%s" "$CLAUDE_PROJECT_DIR" "${CLAUDE_PLUGIN_ROOT}"',
        timeout_s=10,
        source="test",
        plugin_root="/plug",
    )
    run = await hf.run_hook(hook, {}, str(tmp_path))
    assert run.exit_code == 0 and run.stdout.endswith("|/plug")


# ---------------------------------------------------------------------------
# End to end: the operator's real hook through the real loop
# ---------------------------------------------------------------------------


class _Scripted:
    def __init__(self, turns: list[list[Any]]) -> None:
        self.turns = turns
        self.requests: list[Any] = []

    def __call__(self, request: Any, signal: AbortSignal | None):
        self.requests.append(request)
        turn = self.turns[len(self.requests) - 1]

        async def gen():
            for event in turn:
                yield event

        return gen()


def _fake_bash() -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update, context):
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="bash",
            content=[TextContent(text="https://github.com/o/r/pull/1")],
        )

    return AgentTool(
        name="bash",
        parameters={"type": "object", "properties": {"command": {"type": "string"}}},
        execute=execute,
    )


async def _turn(home: Path, cwd: Path, command: str, identity: hf.HookIdentity) -> str:
    """Run one bash call through AgentLoop; return the tool result the NEXT request carried."""

    async def post_tool_hooks(name, args, call_id, result):
        return await hf.forward_post_tool(
            identity,
            tool_name=name,
            args=args,
            tool_use_id=call_id,
            output=result.text,
            is_error=result.is_error,
            duration_s=result.duration_s,
            home=home,
        )

    stream = _Scripted(
        [
            [
                StreamToolCallDelta(index=0, id="c1", name="bash", argument_delta=""),
                StreamToolCallDelta(
                    index=0,
                    id="c1",
                    name=None,
                    argument_delta=json.dumps({"command": command}),
                ),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamEndEvent(stop_reason="stop")],
        ]
    )
    config = LoopConfig(
        model=MODEL,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=stream,
        post_tool_hooks=post_tool_hooks,
    )
    context = LoopContext(
        system_blocks=["sys"],
        tools=[_fake_bash()],
        tool_context=ToolContext(cwd=str(cwd)),
    )
    async for _ in AgentLoop().run([], context, config):
        pass
    assert len(stream.requests) == 2
    second = stream.requests[1]
    return json.dumps(
        second.model_dump(mode="json") if hasattr(second, "model_dump") else str(second)
    )


@pytest.mark.skipif(shutil.which("jq") is None, reason="the operator's hook needs jq")
@pytest.mark.asyncio
async def test_real_pr_hook_reaches_the_next_model_request(
    home: Path, tmp_path: Path
) -> None:
    _settings(
        home / ".claude" / "settings.json",
        {"PostToolUse": [_cmd(str(FIXTURE), "Bash")]},
    )
    main = hf.HookIdentity(session_id="s", cwd=str(tmp_path))
    child = hf.HookIdentity(
        session_id="s", cwd=str(tmp_path), agent_id="job", agent_type="coder"
    )
    needle = "Invoke the fix-pr-reviews skill"

    opened = await _turn(home, tmp_path, "gh pr create --title t --body b", main)
    assert needle in opened and "hook-context" in opened

    # The script's own subagent guard (agent_id) still works under lop.
    assert needle not in await _turn(home, tmp_path, "gh pr create --title t", child)
    # Drafts and unrelated commands stay silent.
    assert needle not in await _turn(
        home, tmp_path, "gh pr create --draft --title t", main
    )
    assert needle not in await _turn(home, tmp_path, "git status", main)


@pytest.mark.asyncio
async def test_forwarding_off_runs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hf, "forwarding_enabled", lambda: (False, False))
    marker = tmp_path / "ran"

    def boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("loaded config while off")

    monkeypatch.setattr(hf, "load_hook_commands", boom)
    notes = await hf.forward_post_tool(
        hf.HookIdentity(session_id="s", cwd=str(tmp_path)),
        tool_name="bash",
        args={"command": "x"},
        tool_use_id="c",
        output="",
        is_error=False,
        duration_s=None,
    )
    assert notes == [] and not marker.exists()


@pytest.mark.asyncio
async def test_a_raising_hook_callback_leaves_the_result_unchanged() -> None:
    async def broken(*_a: Any) -> list[str]:
        raise RuntimeError("bad host")

    config = LoopConfig(
        model=MODEL,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=_Scripted([]),
        post_tool_hooks=broken,
    )
    original = ToolResult(
        tool_call_id="c", tool_name="bash", content=[TextContent(text="ok")]
    )
    assert (
        await AgentLoop._apply_post_tool_hooks(config, "bash", {}, "c", original)
        is original
    )
