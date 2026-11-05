"""Synthetic-only regression for the pre-execution privacy boundary.

No payload is executed: the runner uses an execution spy, while the isolated
Session errand path talks only to a recording fake stream. Canaries are test
data, never credentials retrieved from this machine.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from local_operator.harness.types import StreamEndEvent, StreamTextDelta
from tests.unit.harness.test_loop_guardrail import _run, _text
from tests.unit.session.test_config_live import RebindableStream, make_session


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "home"
    state = root / ".local-operator"
    (state / "sessions/own/scratchpad").mkdir(parents=True)
    (state / "config.yml").write_text("version: 1\n")
    monkeypatch.setenv("HOME", str(root))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))
    return root


CANARY = "synthetic-guard-privacy-canary-47f2d9"
COMMAND = 'rm -rf "$HOME/$OPENAI_API_KEY"'


@pytest.mark.parametrize("name", ["OPENAI_API_KEY", "PROJECT_WIDGET_TOKEN"])
@pytest.mark.parametrize("value", [CANARY, ""])
def test_s1_private_expansion_is_opaque_not_empty(home, name, value):
    from local_operator.guardrails.classify import Risk, classify_bash

    verdict = classify_bash(
        f'rm -rf "/tmp/${{{name}}}/child"',
        cwd=str(home),
        env={"HOME": str(home), name: value},
        scratchpad=None,
    )
    assert verdict.risk >= Risk.HIGH
    assert any(row.target.credential_derived and row.target.opaque for row in verdict.targets)
    assert all(not row.target.empty for row in verdict.targets)
    assert CANARY not in repr(verdict)


@pytest.mark.asyncio
async def test_s1_runner_session_three_channels(home, tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", CANARY)
    seen = []

    class Recording(RebindableStream):
        def __call__(self, request, signal):
            seen.append(request)

            async def stream():
                yield StreamTextDelta(delta="BLOCK: synthetic review refusal")
                yield StreamEndEvent(stop_reason="stop")

            return stream()

    session = make_session(tmp_path / "isolated-session", Recording({}))
    try:
        with caplog.at_level(logging.INFO, logger="local_operator.guardrails.check"):
            result, executed = await _run(home, COMMAND, judge=session._guard_judge)
        observations = {
            "request_leak": any(CANARY in str(request) for request in seen),
            "journal_leak": CANARY in caplog.text,
            "refusal_leak": CANARY in _text(result),
        }
        assert result.is_error and not executed
        assert observations == dict.fromkeys(observations, False)
    finally:
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [CANARY, "/", ".local-operator"])
async def test_s1_hidden_delete_cannot_be_allowed(home, monkeypatch, value):
    monkeypatch.setenv("OPENAI_API_KEY", value)
    result, executed = await _run(home, COMMAND, judge=lambda *_: _allow())
    assert result.is_error and not executed


async def _allow():
    return "ALLOW"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        'A="$OPENAI_API_KEY"; rm -rf "$A"',
        'A="$OPENAI_API_KEY"; (rm -rf "$A")',
        'A="$OPENAI_API_KEY"; echo x | xargs rm -rf "$A"',
        'rm -rf "$(mktemp -d)/$OPENAI_API_KEY"',
        'for A in "$OPENAI_API_KEY"; do rm -rf "$A"; done',
        'bash -c "rm -rf $OPENAI_API_KEY"',
        'eval "rm -rf $OPENAI_API_KEY"',
        'truncate -s 0 "$OPENAI_API_KEY"',
        ': > "$OPENAI_API_KEY"',
    ],
)
async def test_s1_private_alias_never_autoexecutes(home, monkeypatch, command):
    monkeypatch.setenv("OPENAI_API_KEY", CANARY)
    result, executed = await _run(home, command, judge=lambda *_: _allow())
    assert result.is_error and not executed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        'echo "$OPENAI_API_KEY"; rm -f "$LOCAL_OPERATOR_SCRATCHPAD/file.tmp"',
        "git commit -m 'OPENAI_API_KEY is a literal name'",
        "cat <<'EOF'\nOPENAI_API_KEY is data\nEOF",
        '# "$OPENAI_API_KEY" is a comment\nrm -f "$LOCAL_OPERATOR_SCRATCHPAD/file.tmp"',
        '(cd "$OPENAI_API_KEY"); rm -f "$LOCAL_OPERATOR_SCRATCHPAD/file.tmp"',
        'cd "$OPENAI_API_KEY" | cat; rm -f "$LOCAL_OPERATOR_SCRATCHPAD/file.tmp"',
        'A="$OPENAI_API_KEY"; A=/tmp/guard-output; rm -f "$A"',
    ],
)
async def test_s1_no_whole_command_taint(home, monkeypatch, command):
    monkeypatch.setenv("OPENAI_API_KEY", CANARY)
    calls = []

    async def judge(*args):
        calls.append(args)
        return "ALLOW"

    result, executed = await _run(home, command, judge=judge)
    assert not result.is_error and len(executed) == 1 and not calls


@pytest.mark.asyncio
async def test_s1_safe_child_inherit_is_unchanged(home, monkeypatch):
    import asyncio
    import json

    from local_operator.harness.loop import AgentLoop, LoopContext, _PlannedCall
    from local_operator.harness.types import LoopConfig, Message, ToolCall, ToolContext
    from local_operator.tools.registry import create_tools
    from tests.unit.harness.test_loop_guardrail import MODEL

    monkeypatch.setenv("OPENAI_API_KEY", CANARY)
    context = ToolContext(
        cwd=str(home),
        session_id="own",
        scratchpad_dir=str(home / ".local-operator/sessions/own/scratchpad"),
    )
    tool = next(t for t in create_tools(context, enabled=("bash",)) if t.name == "bash")
    # This child reports only comparison status; never print even the synthetic value.
    command = (
        'test "$OPENAI_API_KEY" = "synthetic-guard-privacy-canary-47f2d9"'
        " && printf inherit-preserved"
    )
    args = {"command": command}
    call = ToolCall(id="inherit", name="bash", arguments=args, raw_arguments=json.dumps(args))
    config = LoopConfig(
        model=MODEL,
        stream_fn=lambda *_: None,
        convert_to_llm=lambda messages: list(messages),
    )
    result = await AgentLoop()._runner_result(
        _PlannedCall(call=call, tool=tool, args=args),
        LoopContext(messages=[Message.user("check synthetic inheritance")], tool_context=context),
        config,
        None,
        asyncio.Queue(),
    )
    assert not result.is_error
    assert "inherit-preserved" in _text(result)
    assert CANARY not in _text(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        'cd "$OPENAI_API_KEY"; rm -rf sessions',
        'cd "$OPENAI_API_KEY"; rm -rf ./nested/data',
    ],
)
async def test_s1_hidden_cwd_cannot_be_allowed(home, monkeypatch, command):
    monkeypatch.setenv("OPENAI_API_KEY", str(home / ".local-operator"))
    result, executed = await _run(home, command, judge=lambda *_: _allow())
    assert result.is_error and not executed


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["review", "warn", "block"])
async def test_s1_custom_rule_boundaries(home, tmp_path, monkeypatch, caplog, action):
    monkeypatch.setenv("OPENAI_API_KEY", CANARY)
    rules = home / ".local-operator" / "rules"
    rules.mkdir()
    (rules / "privacy.md").write_text(
        f"---\nname: privacy-{CANARY}\nscope: tool:bash\ncondition: ['echo']\n"
        f"action: {action}\nquestion: {CANARY}\n---\n{CANARY}\n"
    )
    seen = []

    class Recording(RebindableStream):
        def __call__(self, request, signal):
            seen.append(request)

            async def stream():
                yield StreamTextDelta(delta=f"BLOCK: {CANARY}")
                yield StreamEndEvent(stop_reason="stop")

            return stream()

    session = make_session(tmp_path / "rule-session", Recording({}))
    try:
        with caplog.at_level(logging.INFO, logger="local_operator.guardrails.check"):
            result, executed = await _run(home, "echo ok", judge=session._guard_judge)
        assert CANARY not in _text(result)
        assert CANARY not in caplog.text
        assert all(CANARY not in str(request) for request in seen)
        assert len(seen) == (1 if action == "review" else 0)
        assert all(request.purpose == "guardrail" and not request.tools for request in seen)
        assert bool(executed) == (action == "warn")
    finally:
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["raise", "invalid"])
async def test_s1_redactor_failure_never_executes_or_exports(home, monkeypatch, caplog, failure):
    from local_operator.secrets import runtime

    calls = []

    def broken(text):
        if failure == "raise":
            raise RuntimeError(CANARY)
        return None

    async def judge(*args):
        calls.append(args)
        return "ALLOW"

    monkeypatch.setattr(runtime, "scrub", broken)
    result, executed = await _run(home, 'rm -rf "$HOME/Documents"', judge=judge)
    assert result.is_error and not executed and not calls
    assert CANARY not in _text(result) and CANARY not in caplog.text


@pytest.mark.asyncio
async def test_s1_critical_refusal_uses_registry_redaction(home, monkeypatch, caplog):
    from local_operator.secrets import runtime

    ledger = runtime._RedactionLedger()
    monkeypatch.setattr(runtime, "_LEDGER", ledger)
    ledger.register(str(home))
    with caplog.at_level(logging.INFO, logger="local_operator.guardrails.check"):
        result, executed = await _run(home, 'rm -rf "$HOME"', judge=lambda *_: _allow())
    assert result.is_error and not executed
    assert "(critical)" in _text(result)
    assert str(home) not in _text(result) and str(home) not in caplog.text


@pytest.mark.asyncio
async def test_s1_ordinary_literal_assignment_remains_low(home, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", CANARY)
    result, executed = await _run(
        home,
        'CLEANUP_PATH=/tmp/guard-test-output; rm -f "$CLEANUP_PATH"',
        judge=lambda *_: _allow(),
    )
    assert not result.is_error and len(executed) == 1
