"""Tiny plain-text affordance, exact existing storage/approval/privacy contracts."""

import json

import pytest

from local_operator.harness.types import ToolContext
from local_operator.memory import MemoryIdentity, MemoryStore
from local_operator.tools.memory_tools import build_memory_tool, execute_fact_memory
from local_operator.tools.registry import create_tools


@pytest.mark.asyncio
async def test_store_recall_and_existing_url_correction_share_one_engine(tmp_path):
    root = tmp_path / "cfg"
    context = ToolContext(
        session_id="tiny", memory_config_dir=str(root), session_dir=str(root / "sessions" / "tiny")
    )
    created = await execute_fact_memory(
        "a",
        {
            "op": "store",
            "scope": "session",
            "text": "Kestrel rollout ships Friday; runbook docs/kestrel.md",
        },
        context=context,
    )
    assert not created.is_error
    item = json.loads(str(getattr(created.content[0], "text", "")))
    bank = MemoryStore(root, MemoryIdentity("tiny"))
    assert bank.show("session", item["id"])["fact"]["source"]["kind"] == "reported"
    assert bank.show("session", item["id"])["fact"]["verified_at"] is None
    recall = await execute_fact_memory(
        "r", {"op": "recall", "scope": "session", "text": "Kestrel rollout"}, context=context
    )
    assert not recall.is_error and item["id"] in str(recall.content)
    from local_operator.tools.memory_tools import execute_memory

    changed = await execute_memory(
        "e",
        "edit",
        {
            "path": "memory://session/" + item["id"],
            "old_text": item["version"],
            "new_text": json.dumps({"title": "Kestrel corrected", "body": "Ship Monday"}),
        },
        context,
    )
    assert not changed.is_error
    assert bank.show("session", item["id"])["fact"]["body"] == "Ship Monday"


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["disabled", "owner", "quota", "oversize"])
async def test_refusals_preserve_existing_policy_and_never_truncate(tmp_path, monkeypatch, case):
    import local_operator.memory as memory

    root = tmp_path / "cfg"
    root.mkdir()
    if case == "disabled":
        (root / "config.yml").write_text('{"values":{"memory":{"enabled":false}}}')
    if case == "quota":
        monkeypatch.setattr(memory, "MAX_OWNER_OPS", 0)
    args = {
        "op": "store",
        "scope": "agent" if case == "owner" else "global",
        "text": "word " * 400 if case == "oversize" else "Kestrel release goal",
    }
    response = await execute_fact_memory(
        "r", args, context=ToolContext(session_id="s", agent_id="main", memory_config_dir=str(root))
    )
    assert response.is_error
    assert MemoryStore(root, MemoryIdentity()).inventory()["items"] == []


def test_default_advertised_but_explicit_allowlist_not_widened_and_approval_tiers():
    tool = build_memory_tool()
    assert tool.approval_tier == "write"
    assert tool.call_approval_tier is not None
    assert tool.call_approval_tier({"op": "recall"}) == "read"
    assert tool.call_approval_tier({"op": "store"}) == "write"
    assert "memory" in [t.name for t in create_tools(ToolContext())]
    assert "memory" not in [t.name for t in create_tools(ToolContext(), enabled=["read"])]


def test_schema_published_with_intent_stays_under_250_tokens():
    from local_operator.compaction.tokens import count_text_tokens

    tool = next(t for t in create_tools(ToolContext()) if t.name == "memory")
    envelope = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }
    text = json.dumps(envelope)
    assert "i" in tool.parameters["properties"]
    assert count_text_tokens(text) <= 250
    assert len(text) / 2.78 <= 250  # billed-density control, not only a generic tokenizer


def test_malformed_dedicated_args_are_recognised_and_never_assume_a_path():
    from local_operator.memory_privacy import is_memory, retained_arguments

    for args in (
        None,
        ["SSN 123-45-6789"],
        {"text": "SSN 123-45-6789"},
        {"op": "invalid", "scope": "invalid", "text": "a3bf" * 10},
    ):
        assert is_memory("memory", args)
        safe, raw, masked = retained_arguments(
            "memory", args, json.dumps({"payload": args, "alternate": "SSN 123-45-6789"})
        )
        assert masked and safe == {"memory_arguments_redacted": True}
        assert raw is not None
        assert "6789" not in raw and "a3bf" not in raw


def test_url_scheme_is_not_a_relative_windows_drive_in_protected_cwd(tmp_path, monkeypatch):
    from local_operator.memory import MemoryError, guard
    from local_operator.memory_privacy import retained_arguments

    protected = tmp_path / "config"
    protected.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(protected))
    monkeypatch.chdir(protected)
    args = {
        "op": "store",
        "scope": "session",
        "text": "Kestrel rollout: https://example.invalid/threads/kestrel",
    }
    assert not retained_arguments("memory", args, json.dumps(args))[2]
    guard("memory://team?q=Kestrel")
    with pytest.raises(MemoryError, match="Sensitive path"):
        guard("Read ~/.ssh/id_rsa")


@pytest.mark.asyncio
@pytest.mark.parametrize("cwd_kind", ["repo", "scratch", "neutral"])
async def test_exact_captured_model_arguments_replay_in_each_cwd(tmp_path, monkeypatch, cwd_kind):
    from pathlib import Path

    from local_operator.memory_privacy import retained_arguments

    root = tmp_path / "config"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    cwd = (
        root
        if cwd_kind == "scratch"
        else Path("/") if cwd_kind == "neutral" else Path(__file__).resolve().parents[2]
    )
    monkeypatch.chdir(cwd)
    args = {
        "i": "Storing Kestrel rollout facts",
        "op": "store",
        "scope": "session",
        "text": (
            "rollout goal: ship Friday; runbook: docs/kestrel.md; Slack channel: #kestrel-ops "
            "thread https://example.invalid/threads/kestrel; rollout owner: "
            "Mira Vale at mira@example.invalid"
        ),
    }
    assert not retained_arguments("memory", args, json.dumps(args))[2]
    context = ToolContext(session_id="replay", memory_config_dir=str(root))
    result = await execute_fact_memory("captured-call", args, context=context)
    assert not result.is_error, result
    bank = MemoryStore(root, MemoryIdentity("replay"))
    rows = bank.search("rollout", "session")
    assert rows[0]["body"] == args["text"]
    assert rows[0]["source"]["kind"] == "reported"


@pytest.mark.asyncio
async def test_host_denial_prevents_the_named_tool_from_writing(tmp_path):
    from local_operator.harness.loop import AgentLoop, LoopContext
    from local_operator.harness.types import Message, StreamEndEvent, StreamTextDelta
    from tests.unit.harness.test_loop import (
        ScriptedStream,
        make_config,
        tool_call_delta,
    )

    denied = []

    async def deny(name, summary):
        denied.append(name)
        return False

    stream = ScriptedStream(
        [
            [
                tool_call_delta(
                    0,
                    id="store-denied",
                    name="memory",
                    args=json.dumps(
                        {"op": "store", "scope": "global", "text": "Kestrel goal ships Friday"}
                    ),
                ),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="Not stored"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(
        tools=[build_memory_tool()],
        tool_context=ToolContext(request_approval=deny, memory_config_dir=str(tmp_path)),
    )
    result = await AgentLoop().run_to_end(
        [Message.user("Remember the rollout goal")], context, make_config(stream), None
    )
    assert denied == ["memory"]
    assert any(m.is_error for m in result if isinstance(m, Message) and m.role == "tool")
    assert not (tmp_path / "memory").exists()


def test_scope_schema_defines_lifetimes_instead_of_bare_enum():
    docs = build_memory_tool().parameters["properties"]["scope"]["description"]
    assert "session=this conversation" in docs
    assert "global=future sessions" in docs
    assert "team/agent=shared work/profile" in docs


@pytest.mark.asyncio
async def test_explicit_global_store_is_recalled_by_a_fresh_identity_without_scope_override(
    tmp_path,
):
    first = ToolContext(session_id="first", memory_config_dir=str(tmp_path))
    second = ToolContext(session_id="fresh", memory_config_dir=str(tmp_path))
    stored = await execute_fact_memory(
        "s",
        {"op": "store", "scope": "global", "text": "Kestrel rollout ships Friday"},
        context=first,
    )
    assert not stored.is_error
    recall = await execute_fact_memory(
        "r", {"op": "recall", "scope": "global", "text": "Kestrel rollout"}, context=second
    )
    assert not recall.is_error and "ships Friday" in str(recall.content)
    local = await execute_fact_memory(
        "local", {"op": "store", "scope": "session", "text": "Private session note"}, context=first
    )
    assert not local.is_error
    hidden = await execute_fact_memory(
        "h", {"op": "recall", "scope": "session", "text": "Private session note"}, context=second
    )
    assert not hidden.is_error and "Private session note" not in str(hidden.content)


@pytest.mark.asyncio
@pytest.mark.parametrize("cwd_kind", ["repo", "scratch", "neutral"])
@pytest.mark.parametrize(
    "locator", ["docs/.env", "deploy/kubeconfig-prod", "config/token.txt", "C:/.ssh/id_rsa"]
)
async def test_relative_sensitive_paths_refused_by_actual_store_and_capture_in_each_cwd(
    tmp_path, monkeypatch, cwd_kind, locator
):
    from pathlib import Path

    from local_operator.memory_capture import extract

    root = tmp_path / "config"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    cwd = (
        root
        if cwd_kind == "scratch"
        else Path("/") if cwd_kind == "neutral" else Path(__file__).resolve().parents[2]
    )
    monkeypatch.chdir(cwd)
    context = ToolContext(session_id="paths", memory_config_dir=str(root))
    result = await execute_fact_memory(
        "unsafe", {"op": "store", "scope": "global", "text": "Read " + locator}, context=context
    )
    assert result.is_error
    assert not (root / "memory").exists(), "refusal precedes every DB/index/oplog write"
    assert extract("Read " + locator) == [], "automatic capture uses the same path guard"
    good = await execute_fact_memory(
        "url",
        {"op": "store", "scope": "global", "text": "Runbook https://example.invalid/runbook"},
        context=context,
    )
    assert not good.is_error
    assert any(
        row["body"] == "https://example.invalid/runbook"
        for row in extract("https://example.invalid/runbook")
    )
    # A URL carrying credential material still uses the credential-shape guard.
    bad_url = await execute_fact_memory(
        "credential-url",
        {
            "op": "store",
            "scope": "global",
            "text": "https://operator:synthetic-pass123@example.invalid/runbook",
        },
        context=context,
    )
    assert bad_url.is_error
