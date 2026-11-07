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


def _text(result: ToolResult) -> str:
    """The result's first text block (a guardrail refusal is always one)."""
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


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
    user_text: str = "please clean up the ds* scratch files",
) -> tuple[ToolResult, list[dict[str, Any]]]:
    executed: list[dict[str, Any]] = []

    async def request_approval(*_: Any) -> bool:
        return True  # auto approval: approves everything, like the incident run

    state = home / ".local-operator"
    context = LoopContext(
        messages=[Message.user(user_text)],
        tool_context=ToolContext(
            cwd=str(home),
            session_id="own",
            scratchpad_dir=str(state / "sessions" / "own" / "scratchpad"),
            request_approval=request_approval if approve else None,
        ),
    )
    config = LoopConfig(
        model=MODEL,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=lambda *_: None,
        guard_judge=judge,
        guard_override_authorised=(
            lambda: frozenset({"guardrails.critical_override"}) if authorised else frozenset()
        ),
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
    assert _text(result).startswith("BLOCKED by guardrail dangerous-delete (critical)")
    assert judge.calls == []  # CRITICAL never asks a model


@pytest.mark.asyncio
async def test_override_from_unauthorised_write_is_ignored(home: Path) -> None:
    (home / ".local-operator" / "config.yml").write_text(
        "version: 1\nvalues:\n  guardrails:\n    critical_override: true\n"
    )
    judge = _Judge("ALLOW")
    result, executed = await _run(home, "rm -rf ~/.local-operator", judge=judge, authorised=False)
    assert executed == [] and "(critical)" in _text(result)


@pytest.mark.asyncio
async def test_authorised_override_downgrades_to_judge_only(home: Path) -> None:
    (home / ".local-operator" / "config.yml").write_text(
        "version: 1\nvalues:\n  guardrails:\n    critical_override: true\n"
    )
    blocker = _Judge("BLOCK: the user never asked")
    result, executed = await _run(home, "rm -rf ~/.local-operator", judge=blocker, authorised=True)
    assert executed == [] and len(blocker.calls) == 1
    assert "the user never asked" in _text(result)
    # An ALLOW must cite what it rests on, and for an overridden CRITICAL root
    # only the operator's own words naming it can verify.
    allower = _Judge("ALLOW")
    result, executed = await _run(home, "rm -rf ~/.local-operator", judge=allower, authorised=True)
    assert executed == [] and "could not be verified" in _text(result)
    allower = _Judge("ALLOW evidence=e1")
    result, executed = await _run(home, "rm -rf ~/.local-operator", judge=allower, authorised=True)
    assert executed == []
    allower = _Judge("ALLOW operator")
    result, executed = await _run(
        home,
        "rm -rf ~/.local-operator",
        judge=allower,
        authorised=True,
        user_text="yes, delete ~/.local-operator entirely",
    )
    assert len(executed) == 1


@pytest.mark.asyncio
async def test_high_runs_only_on_allow(home: Path) -> None:
    blocker = _Judge("BLOCK: not requested")
    result, executed = await _run(home, "rm -rf ~/projects-x", judge=blocker)
    assert executed == []
    assert _text(result).startswith("BLOCKED by guardrail dangerous-delete (review)")
    assert "not requested" in _text(result)
    assert "please clean up" in blocker.calls[0][1]  # the user's own words reach the judge
    # An uncited ALLOW is a claim the checkpoint cannot verify: refused.
    result, executed = await _run(home, "rm -rf ~/projects-x", judge=_Judge("ALLOW"))
    assert executed == [] and "could not be verified" in _text(result)
    # Citing the operator verifies only when a direct operator row names the path.
    result, executed = await _run(home, "rm -rf ~/projects-x", judge=_Judge("ALLOW operator"))
    assert executed == []
    result, executed = await _run(
        home,
        "rm -rf ~/projects-x",
        judge=_Judge("ALLOW operator"),
        user_text="please delete ~/projects-x",
    )
    assert len(executed) == 1 and not result.is_error


@pytest.mark.asyncio
async def test_high_without_judge_fails_closed(home: Path) -> None:
    result, executed = await _run(home, "rm -rf ~/projects-x", judge=None)
    assert executed == [] and "judge unavailable" in _text(result)


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
    last = result.content[-1]
    assert isinstance(last, TextContent) and last.text == DESTRUCTIVE_TIMEOUT_NOTE
    result, _ = await _run(
        home, "sleep 999", tool_text="TIMEOUT after 120.0s (process killed)\nexit code: -9"
    )
    assert all(
        not isinstance(c, TextContent) or c.text != DESTRUCTIVE_TIMEOUT_NOTE for c in result.content
    )


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
    assert _text(result).startswith("BLOCKED by guardrail no-force-push (block)")
    assert "Open a PR instead." in _text(result)


@pytest.mark.asyncio
async def test_unauthorised_rules_disable_is_ignored(home: Path) -> None:
    """S5: a config edit the agent could make cannot switch the user's rules off."""
    state = home / ".local-operator"
    rules = state / "rules"
    rules.mkdir()
    (rules / "no-force-push.md").write_text(
        "---\nname: no-force-push\nscope: tool:bash\naction: block\n"
        "condition: ['git\\s+push\\b.*--force']\n---\nOpen a PR instead.\n"
    )
    (state / "config.yml").write_text(
        "version: 1\nvalues:\n  guardrails:\n    enabled: false\n    rules_dir: /nonexistent\n"
    )
    result, executed = await _run(home, "git push --force origin main")
    assert executed == [] and "no-force-push (block)" in _text(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["make test 2>&1", "npm test 2>&1 | tee log.txt", "echo e >&2"])
async def test_q1_fd_duplication_runs_with_zero_judge_calls(home: Path, command: str) -> None:
    judge = _Judge("BLOCK: should never be asked")
    result, executed = await _run(home, command, judge=judge)
    assert len(executed) == 1 and judge.calls == []


@pytest.mark.asyncio
async def test_loop_records_what_a_bash_call_created(home: Path) -> None:
    """The loop snapshots creation candidates before the call and records after.

    The fake tool really creates the directory, so this exercises the same
    plan -> execute -> commit order the live bash tool goes through; the next
    delete of that directory then ranks LOW with no judge call.
    """
    from local_operator.guardrails import provenance

    made = home / "qa-rig-loop"
    executed: list[dict[str, Any]] = []
    base = _tool(executed)

    async def execute(call_id: str, args: dict[str, Any], *rest: Any) -> ToolResult:
        made.mkdir()
        return await base.execute(call_id, args, *rest)

    tool = AgentTool(
        name="bash",
        parameters=base.parameters,
        execute=execute,
        approval_tier="exec",
    )
    state = home / ".local-operator"
    context = LoopContext(
        messages=[Message.user("set up a rig")],
        tool_context=ToolContext(
            cwd=str(home),
            session_id="own",
            ancestor_session_ids=("parent1",),
            scratchpad_dir=str(state / "sessions" / "own" / "scratchpad"),
        ),
    )
    config = LoopConfig(
        model=MODEL,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=lambda *_: None,
    )
    args = {"command": f"mkdir {made}"}
    call = ToolCall(id="c1", name="bash", arguments=args, raw_arguments=json.dumps(args))
    await AgentLoop()._runner_result(
        _PlannedCall(call=call, tool=tool, args=args), context, config, None, asyncio.Queue()
    )
    for owner in ("own", "parent1"):
        record = provenance.owned_record(str(made), owner)
        assert record is not None and record.kind == "mkdir", owner
    judge = _Judge("BLOCK: should never be asked")
    _, executed = await _run(home, f"rm -rf {made}", judge=judge)
    assert len(executed) == 1 and judge.calls == []


@pytest.mark.asyncio
async def test_loop_refuses_a_write_tool_call_onto_a_transcript(home: Path) -> None:
    """The checkpoint runs for every tool; a ``write`` onto a session transcript
    is refused before the tool runs, with the CRITICAL refusal text."""
    executed: list[dict[str, Any]] = []

    async def execute(call_id: str, args: dict[str, Any], *_: Any) -> ToolResult:
        executed.append(args)
        return ToolResult(tool_call_id=call_id, tool_name="write", content=[TextContent(text="ok")])

    tool = AgentTool(
        name="write",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        execute=execute,
        approval_tier="write",
    )
    state = home / ".local-operator"
    context = LoopContext(
        messages=[Message.user("hi")],
        tool_context=ToolContext(cwd=str(home), session_id="own"),
    )
    config = LoopConfig(
        model=MODEL,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=lambda *_: None,
    )
    for path, refused in (
        (str(state / "sessions" / "own" / "transcript.jsonl"), True),
        (str(home / "notes.md"), False),
    ):
        args = {"path": path, "content": "{}"}
        call = ToolCall(id="w1", name="write", arguments=args, raw_arguments=json.dumps(args))
        result = await AgentLoop()._runner_result(
            _PlannedCall(call=call, tool=tool, args=args), context, config, None, asyncio.Queue()
        )
        if refused:
            assert _text(result).startswith("BLOCKED by guardrail dangerous-delete (critical)")
    assert [a["path"] for a in executed] == [str(home / "notes.md")]
