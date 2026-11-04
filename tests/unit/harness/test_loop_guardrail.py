"""The guardrail checkpoint at the loop's single call site (``_runner_result``).

Pins the incident property: under auto approval and under a yolo pin, a CRITICAL
deletion is answered with a synthetic refusal and the tool's ``execute`` is never
awaited; a HIGH one runs only on the judge's ALLOW; a LOW one runs with zero
judge calls.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.loop import (
    DESTRUCTIVE_TIMEOUT_NOTE,
    AgentLoop,
    LoopContext,
    _PlannedCall,
)
from local_operator.harness.types import (
    AgentTool,
    LoopConfig,
    Message,
    TextContent,
    ToolCall,
    ToolContext,
    ToolResult,
)
from local_operator.model.configure import build_model_spec

MODEL = build_model_spec("anthropic", "claude-sonnet-4-5")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    state = home / ".local-operator"
    (state / "sessions" / "own" / "scratchpad").mkdir(parents=True)
    (state / "config.yml").write_text("version: 1\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))
    return home


class _Judge:
    def __init__(self, answer: str = "ALLOW") -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.answer


def _tool(executed: list[dict[str, Any]], text: str = "ok") -> AgentTool:
    async def execute(call_id: str, args: dict[str, Any], *_: Any) -> ToolResult:
        executed.append(args)
        return ToolResult(tool_call_id=call_id, tool_name="bash", content=[TextContent(text=text)])

    return AgentTool(
        name="bash",
        parameters={"type": "object", "properties": {"command": {"type": "string"}}},
        execute=execute,
        approval_tier="exec",
    )


async def _run(
    home: Path,
    command: str,
    *,
    judge: Any = None,
    approve: bool | None = True,
    authorised: bool = False,
    tool_text: str = "ok",
) -> tuple[ToolResult, list[dict[str, Any]]]:
    executed: list[dict[str, Any]] = []

    async def request_approval(*_: Any) -> bool:
        return True  # auto approval: approves everything, like the incident run

    state = home / ".local-operator"
    context = LoopContext(
        messages=[Message.user("please clean up the ds* scratch files")],
        tool_context=ToolContext(
            cwd=str(home),
            session_id="own",
            scratchpad_dir=str(state / "sessions" / "own" / "scratchpad"),
            request_approval=request_approval if approve else None,
        ),
    )
    config = LoopConfig(
        model=MODEL,
        convert_to_llm=lambda m: m,
        stream_fn=lambda *_: None,
        guard_judge=judge,
        guard_override_authorised=(lambda: authorised),
    )
    args = {"command": command}
    call = ToolCall(id="c1", name="bash", arguments=args, raw_arguments=json.dumps(args))
    result = await AgentLoop()._runner_result(
        _PlannedCall(call=call, tool=_tool(executed, tool_text), args=args),
        context,
        config,
        None,
        asyncio.Queue(),
    )
    return result, executed


INCIDENT = "rm -rf ~/.local-operator/$(echo)/ 2>/dev/null; rm -rf $LOCAL_OPERATOR_SCRATCHPAD/ds*"


@pytest.mark.asyncio
@pytest.mark.parametrize("approve", [True, None], ids=["auto-approval", "yolo-no-gate"])
async def test_critical_refused_and_never_executed(home: Path, approve: bool | None) -> None:
    judge = _Judge("ALLOW")
    result, executed = await _run(home, INCIDENT, judge=judge, approve=approve)
    assert executed == []
    assert result.is_error
    assert result.content[0].text.startswith("BLOCKED by guardrail dangerous-delete (critical)")
    assert judge.calls == []  # CRITICAL never asks a model


@pytest.mark.asyncio
async def test_override_from_unauthorised_write_is_ignored(home: Path) -> None:
    (home / ".local-operator" / "config.yml").write_text(
        "version: 1\nvalues:\n  guardrails:\n    critical_override: true\n"
    )
    judge = _Judge("ALLOW")
    result, executed = await _run(home, "rm -rf ~/.local-operator", judge=judge, authorised=False)
    assert executed == [] and "(critical)" in result.content[0].text


@pytest.mark.asyncio
async def test_authorised_override_downgrades_to_judge_only(home: Path) -> None:
    (home / ".local-operator" / "config.yml").write_text(
        "version: 1\nvalues:\n  guardrails:\n    critical_override: true\n"
    )
    blocker = _Judge("BLOCK: the user never asked")
    result, executed = await _run(home, "rm -rf ~/.local-operator", judge=blocker, authorised=True)
    assert executed == [] and len(blocker.calls) == 1
    assert "the user never asked" in result.content[0].text
    allower = _Judge("ALLOW")
    result, executed = await _run(home, "rm -rf ~/.local-operator", judge=allower, authorised=True)
    assert len(executed) == 1


@pytest.mark.asyncio
async def test_high_runs_only_on_allow(home: Path) -> None:
    blocker = _Judge("BLOCK: not requested")
    result, executed = await _run(home, "rm -rf ~/projects-x", judge=blocker)
    assert executed == []
    assert result.content[0].text.startswith("BLOCKED by guardrail dangerous-delete (review)")
    assert "not requested" in result.content[0].text
    assert "please clean up" in blocker.calls[0][1]  # the user's own words reach the judge
    allower = _Judge("ALLOW")
    result, executed = await _run(home, "rm -rf ~/projects-x", judge=allower)
    assert len(executed) == 1 and not result.is_error


@pytest.mark.asyncio
async def test_high_without_judge_fails_closed(home: Path) -> None:
    result, executed = await _run(home, "rm -rf ~/projects-x", judge=None)
    assert executed == [] and "judge unavailable" in result.content[0].text


@pytest.mark.asyncio
async def test_low_runs_with_zero_judge_calls(home: Path) -> None:
    judge = _Judge("BLOCK: should never be asked")
    for command in ('rm -rf "$LOCAL_OPERATOR_SCRATCHPAD"/realproof/ds*', "rm /tmp/x", "ls"):
        result, executed = await _run(home, command, judge=judge)
        assert len(executed) == 1, command
    assert judge.calls == []


@pytest.mark.asyncio
async def test_destructive_timeout_carries_verify_note(home: Path) -> None:
    result, _ = await _run(
        home,
        "rm -rf /tmp/big-tree",
        tool_text="TIMEOUT after 120.0s (process killed)\nexit code: -9",
    )
    assert result.content[-1].text == DESTRUCTIVE_TIMEOUT_NOTE
    result, _ = await _run(
        home, "sleep 999", tool_text="TIMEOUT after 120.0s (process killed)\nexit code: -9"
    )
    assert all(c.text != DESTRUCTIVE_TIMEOUT_NOTE for c in result.content)


@pytest.mark.asyncio
async def test_user_block_rule_applies(home: Path) -> None:
    rules = home / ".local-operator" / "rules"
    rules.mkdir()
    (rules / "no-force-push.md").write_text(
        "---\nname: no-force-push\nscope: tool:bash\naction: block\n"
        "condition: ['git\\s+push\\b.*--force']\n---\nOpen a PR instead.\n"
    )
    result, executed = await _run(home, "git push --force origin main")
    assert executed == []
    assert result.content[0].text.startswith("BLOCKED by guardrail no-force-push (block)")
    assert "Open a PR instead." in result.content[0].text
