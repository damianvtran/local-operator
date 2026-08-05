"""Serialization, prompt assembly, template contract, and the summary cap."""

import importlib.resources
import re


from local_operator.compaction import api as compaction_api
from local_operator.compaction.api import (
    MAX_SUMMARY_TOKENS,
    SUMMARIZATION_SYSTEM_PROMPT,
    TOOL_RESULT_MAX_CHARS,
    build_compaction_prompt,
    extract_file_ops_from_messages,
    format_file_operations,
    serialize_conversation,
    summarize_messages,
    upsert_file_operations,
)
from local_operator.harness.types import Message, ToolCall


def _tool_result(call_id: str, text: str, useless: bool = False) -> Message:
    message = Message(role="tool", tool_call_id=call_id, tool_name="bash")
    message.content = Message.user(text).content
    if useless:
        message.provider_payload = {"useless": True}
    return message


def _assistant_with_calls(text: str, *call_ids: str) -> Message:
    message = Message.assistant(text)
    message.tool_calls = [
        ToolCall(id=cid, name="bash", arguments={"command": f"cmd-{cid}"}) for cid in call_ids
    ]
    return message


def test_tool_result_truncation_boundary():
    """Exactly TOOL_RESULT_MAX_CHARS stays whole; one char over truncates with
    an explicit marker and the dropped count."""
    at_limit = "x" * TOOL_RESULT_MAX_CHARS
    over = "y" * (TOOL_RESULT_MAX_CHARS + 1)
    in_limit = serialize_conversation([_tool_result("c1", at_limit)])
    in_over = serialize_conversation([_tool_result("c2", over)])
    assert "[..." not in in_limit
    assert in_limit.endswith(at_limit)
    assert "more characters truncated]" in in_over
    assert in_over.count("y") == TOOL_RESULT_MAX_CHARS
    assert "1 more characters truncated" in in_over


def test_useless_drops_pair_but_sibling_survives():
    """A useless result AND its paired call vanish; a sibling call on the same
    assistant survives with its (non-useless) result."""
    assistant = _assistant_with_calls("doing two things", "c1", "c2")
    messages = [
        assistant,
        _tool_result("c1", "useless output", useless=True),
        _tool_result("c2", "real output"),
    ]
    rendered = serialize_conversation(messages)
    # The useless pair is gone.
    assert "cmd-c1" not in rendered
    assert "useless output" not in rendered
    # The assistant text and the surviving sibling remain.
    assert "doing two things" in rendered
    assert "cmd-c2" in rendered
    assert "real output" in rendered


def test_previous_summary_present_and_absent_rendering():
    messages = [Message.user("hi"), Message.assistant("there")]
    with_prior = build_compaction_prompt(messages, previous_summary="earlier work")
    without_prior = build_compaction_prompt(messages)
    assert "<previous-summary>" in with_prior
    assert "earlier work" in with_prior
    # The live template folds the whole previous-summary section when absent.
    assert "Previous summary" not in without_prior
    assert "<previous-summary>" not in without_prior


def test_every_template_var_is_supplied():
    """Every ``{{var}}`` referenced by compaction_summary.md is provided by
    build_compaction_prompt — no variable may render empty by accident."""
    template = (
        importlib.resources.files("local_operator")
        .joinpath("prompts_md/compaction_summary.md")
        .read_text(encoding="utf-8")
    )
    names = set(re.findall(r"\{\{(\w+)\}\}", template))
    # Strip the #if block names; they are conditionals, not data slots.
    if_names = set(re.findall(r"\{\{#if (\w+)\}\}", template))
    data_names = names - if_names
    supplied = {"transcript", "files", "previous_summary"}
    assert data_names <= supplied, f"template vars not supplied: {data_names - supplied}"
    # And the rendered prompt leaves no unsubstituted {{...}} behind.
    prompt = build_compaction_prompt([Message.user("hi")], previous_summary="s", files="f")
    assert "{{" not in prompt
    assert "}}" not in prompt


def test_summarize_messages_uses_fake_complete_fn():
    captured: dict[str, str] = {}

    async def fake_complete(system: str, prompt: str) -> str:
        captured["system"] = system
        captured["prompt"] = prompt
        return "  the summary  "

    result = _run(summarize_messages([Message.user("hi")], fake_complete))
    assert result == "the summary"
    assert captured["system"] == SUMMARIZATION_SYSTEM_PROMPT
    assert "<conversation>" in captured["prompt"]


def test_files_block_fires_on_auto_extract():
    """RC-5: with ``files=None`` (the default), file ops are extracted from the
    summarized messages and the live template's ``{{#if files}}`` block fires;
    ``files=""`` suppresses it."""
    reader = Message.assistant("")
    reader.tool_calls = [ToolCall(id="r1", name="read", arguments={"path": "/repo/a.py:50-200"})]
    messages = [Message.user("look at the file"), reader]
    prompt = build_compaction_prompt(messages)
    assert "<files>" in prompt
    assert "a.py (Read)" in prompt
    # Selector stripped in the rendered tree.
    assert ":50-200" not in prompt.split("<files>")[1]
    # Explicit empty string suppresses the block even with file ops present.
    suppressed = build_compaction_prompt(messages, files="")
    assert "<files>" not in suppressed


def test_summary_capped_at_max_summary_tokens():
    """complete_fn has no max-tokens knob, so the cap is post-hoc: an oversized
    summary is truncated with the marker."""
    huge = "word " * (MAX_SUMMARY_TOKENS * 4)  # well over the token cap

    async def fake_complete(system: str, prompt: str) -> str:
        return huge

    result = _run(summarize_messages([Message.user("hi")], fake_complete))
    assert result.endswith(compaction_api.SUMMARY_TRUNCATED_MARKER)
    from local_operator.compaction.tokens import _encode_len

    assert (
        _encode_len(result)
        <= MAX_SUMMARY_TOKENS + _encode_len(compaction_api.SUMMARY_TRUNCATED_MARKER) + 5
    )


def _run(coro):
    import asyncio

    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# File operations (RC-5 extraction / rendering)
# ---------------------------------------------------------------------------


def _assistant_read(path: str) -> Message:
    message = Message.assistant("")
    message.tool_calls = [ToolCall(id="r1", name="read", arguments={"path": path})]
    return message


def test_extract_file_ops_strips_selectors_and_schemes():
    messages = [
        _assistant_read("/repo/a.py:50-200"),
        _assistant_read("/repo/a.py:raw"),
        _assistant_read("skill://deploy-guide"),
        _assistant_read("artifact://3"),
    ]
    write = Message.assistant("")
    write.tool_calls = [ToolCall(id="w1", name="write", arguments={"path": "/repo/b.py"})]
    edit = Message.assistant("")
    edit.tool_calls = [ToolCall(id="e1", name="edit", arguments={"path": "/repo/a.py"})]
    ops = extract_file_ops_from_messages(messages + [write, edit])
    # Selectors stripped; both reads dedupe to one path.
    assert "/repo/a.py" in ops["read"]
    # scheme:// paths are excluded.
    assert not any("://" in p for p in ops["read"])
    assert ops["written"] == {"/repo/b.py"}
    assert ops["edited"] == {"/repo/a.py"}


def test_format_file_operations_markers_and_cap():
    read_files = [f"/repo/read_{i}.py" for i in range(25)]
    rendered = format_file_operations(read_files, [], set(read_files))
    assert "(Read)" in rendered
    assert "files elided…]" in rendered
    # RW for a file both read and modified.
    rw = format_file_operations(["/repo/x.py"], ["/repo/x.py"], {"/repo/x.py"})
    assert "(RW)" in rw
    wo = format_file_operations([], ["/repo/y.py"], None)
    assert "(Write)" in wo


def test_upsert_strips_legacy_tags():
    old = "summary body\n<read-files>\n/old/a.py\n</read-files>\n"
    updated = upsert_file_operations(old, ["/new/b.py"], [], None)
    assert "<read-files>" not in updated
    assert "/old/a.py" not in updated
    assert "b.py (Read)" in updated
    assert "<files>" in updated
