"""AppendOnlyContextManager tests: the three sync cases + prefix."""

from __future__ import annotations

import pytest

from local_operator.harness.context import (
    AppendOnlyContextManager,
    AppendOnlyLog,
    StablePrefix,
    message_digest,
)
from local_operator.harness.loop import LoopContext
from local_operator.harness.types import AgentTool, Message, ToolResult


def make_tool(name: str, description: str = "d") -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update, context):
        return ToolResult(tool_call_id=tool_call_id, content=[])

    return AgentTool(name=name, description=description, execute=execute)


def user(text: str) -> Message:
    return Message.user(text)


def assistant(text: str) -> Message:
    return Message.assistant(text)


class TestStablePrefix:
    def test_build_reports_change_once(self):
        prefix = StablePrefix()
        context = LoopContext(system_blocks=["s1"], tools=[make_tool("a")])
        assert prefix.build(context) is True
        assert prefix.build(context) is False
        assert prefix.version == 1

    def test_system_block_change_detected(self):
        prefix = StablePrefix()
        context = LoopContext(system_blocks=["s1"], tools=[])
        prefix.build(context)
        context.system_blocks = ["s1", "volatile"]
        assert prefix.build(context) is True
        assert prefix.version == 2

    def test_tool_inventory_change_detected(self):
        prefix = StablePrefix()
        context = LoopContext(system_blocks=["s"], tools=[make_tool("a")])
        prefix.build(context)
        context.tools = [make_tool("a"), make_tool("b")]
        assert prefix.build(context) is True

    def test_tool_description_change_detected(self):
        prefix = StablePrefix()
        context = LoopContext(system_blocks=["s"], tools=[make_tool("a", "old")])
        prefix.build(context)
        context.tools = [make_tool("a", "new")]
        assert prefix.build(context) is True

    def test_invalidate_forces_rebuild(self):
        prefix = StablePrefix()
        context = LoopContext(system_blocks=["s"], tools=[])
        prefix.build(context)
        prefix.invalidate()
        assert prefix.built is False
        assert prefix.build(context) is True

    def test_to_context_snapshot(self):
        prefix = StablePrefix()
        context = LoopContext(system_blocks=["s"], tools=[make_tool("a", "desc")])
        prefix.build(context)
        snapshot = prefix.to_context()
        assert snapshot["system_blocks"] == ["s"]
        assert snapshot["tools"] == [("a", "desc")]


class TestAppendOnlyLog:
    def test_append_extend_to_messages(self):
        log = AppendOnlyLog()
        m1 = user("1")
        log.append(m1)
        log.extend([user("2"), user("3")])
        assert [m.text for m in log.to_messages()] == ["1", "2", "3"]
        assert len(log) == 3

    def test_replace_tail_compaction_only(self):
        log = AppendOnlyLog()
        log.append(user("a"))
        replacement = Message.assistant("summary")
        log.replace_tail(replacement)
        assert log.to_messages() == [replacement]

    def test_replace_tail_empty_raises(self):
        log = AppendOnlyLog()
        with pytest.raises(ValueError):
            log.replace_tail(user("x"))

    def test_truncate_and_clear(self):
        log = AppendOnlyLog()
        log.extend([user("1"), user("2"), user("3")])
        log.truncate(1)
        assert [m.text for m in log.to_messages()] == ["1"]
        log.clear()
        assert len(log) == 0


class TestSyncMessages:
    def test_case_1_append(self):
        manager = AppendOnlyContextManager()
        a, b, c = user("a"), user("b"), user("c")
        manager.sync_messages([a, b])
        assert manager.log.to_messages() == [a, b]
        # Pure append keeps object identity and never re-adds the head.
        manager.sync_messages([a, b, c])
        assert manager.log.to_messages() == [a, b, c]
        assert manager.log.entries()[0] is a

    def test_case_2_shrink_clear(self):
        """Compaction: the array shrank -> clear and replay."""
        manager = AppendOnlyContextManager()
        a, b, c = user("a"), user("b"), user("c")
        manager.sync_messages([a, b, c])
        summary = Message.assistant("SUMMARY")
        manager.sync_messages([summary, c])
        assert manager.log.to_messages() == [summary, c]

    def test_case_3_longest_stable_prefix_rewrite(self):
        """In-place rewrite: same length, one message changed mid-list; the
        log keeps the stable head and replaces only the diverged tail."""
        manager = AppendOnlyContextManager()
        a, b, c = user("a"), user("b"), user("c")
        manager.sync_messages([a, b, c])
        b_rewritten = Message.user("b (rewritten)")
        manager.sync_messages([a, b_rewritten, c])
        entries = manager.log.to_messages()
        assert len(entries) == 3
        assert entries[0] is a  # head untouched
        assert entries[1] == b_rewritten
        assert entries[2] == c  # tail re-appended after divergence

    def test_rewrite_shorter_tail(self):
        manager = AppendOnlyContextManager()
        a, b, c = user("a"), user("b"), user("c")
        manager.sync_messages([a, b, c])
        # Divergence at index 1 AND shrink -> shrink case wins (clear+replay).
        manager.sync_messages([a, Message.user("x")])
        assert [m.text for m in manager.log.to_messages()] == ["a", "x"]

    def test_identical_list_is_noop(self):
        manager = AppendOnlyContextManager()
        msgs = [user("a"), user("b")]
        manager.sync_messages(msgs)
        entries_before = manager.log.entries()
        manager.sync_messages(list(msgs))
        assert manager.log.entries() == entries_before


class TestDigest:
    def test_covers_role_and_text(self):
        assert message_digest(user("x")) != message_digest(assistant("x"))
        assert message_digest(user("x")) != message_digest(user("y"))

    def test_covers_tool_call_fields(self):
        call_a = Message(role="assistant", tool_calls=[])
        call_b = Message(role="assistant", tool_calls=[])
        from local_operator.harness.types import ToolCall

        call_b.tool_calls = [ToolCall(id="t1", name="bash", raw_arguments='{"c":1}')]
        assert message_digest(call_a) != message_digest(call_b)

    def test_covers_provider_payload(self):
        m1 = Message.user("x")
        m2 = Message.user("x", provider_payload={"native": True})
        assert message_digest(m1) != message_digest(m2)

    def test_stable_across_id_and_usage_changes(self):
        """Digest must NOT depend on fields the provider does not serialize
        (id, usage), so unrelated mutations don't invalidate the cache."""
        m1 = Message.user("x")
        m2 = Message.user("x", id="different-id")
        assert message_digest(m1) == message_digest(m2)


class TestManagerBuild:
    def test_build_delegates_to_prefix(self):
        manager = AppendOnlyContextManager()
        context = LoopContext(system_blocks=["s"], tools=[])
        assert manager.build(context) is True
        assert manager.build(context) is False

    def test_reset(self):
        manager = AppendOnlyContextManager()
        context = LoopContext(system_blocks=["s"], tools=[])
        manager.sync_messages([user("a")])
        manager.reset(context)
        assert len(manager.log) == 0
        assert manager.prefix.built is True
