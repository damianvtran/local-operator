"""Aged tool-result clearing: age and batch thresholds, every protection,
stub determinism and locator, idempotence.

The pass exists because a delegated subagent's single launch turn never
reaches the compaction threshold, so old file views and diffs were re-billed
on every later call (see the comment block above ``clear_aged_tool_results``).
"""

from __future__ import annotations

from typing import Any

from local_operator.compaction.pruning import (
    AGED_CLEAR_PROTECTED_TOOLS,
    USELESS_NOTICE,
    aged_clear_notice,
    clear_aged_tool_results,
)
from local_operator.compaction.tokens import estimate_tokens
from local_operator.harness.types import Message, TextContent, ToolCall

_FILLER = "data " * 1_000  # ~1k estimated tokens


def _call(call_id: str, name: str, arguments: dict[str, Any]) -> Message:
    return Message(
        role="assistant",
        content=[TextContent(text="calling")],
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
    )


def _result(
    call_id: str,
    name: str,
    *,
    text: str = _FILLER,
    details: dict[str, Any] | None = None,
    is_error: bool = False,
) -> Message:
    message = Message(
        role="tool",
        tool_call_id=call_id,
        tool_name=name,
        content=[TextContent(text=text)],
        is_error=is_error,
    )
    message.provider_payload = {"details": details or {}}
    return message


def _pair(
    index: int,
    name: str = "read",
    arguments: dict[str, Any] | None = None,
    **result_kwargs: Any,
) -> list[Message]:
    call_id = f"c{index}"
    arguments = arguments if arguments is not None else {"path": f"/repo/f{index}.py"}
    if name == "read" and "details" not in result_kwargs:
        result_kwargs["details"] = {"path": arguments.get("path"), "range": arguments.get("range")}
    return [_call(call_id, name, arguments), _result(call_id, name, **result_kwargs)]


def _aged(messages: list[Message], calls: int) -> list[Message]:
    """Append ``calls`` assistant messages so everything above ages by that."""
    return messages + [Message.assistant(f"step {i}") for i in range(calls)]


def _results(messages: list[Message]) -> list[Message]:
    return [m for m in messages if m.role == "tool"]


def test_age_threshold_is_counted_in_assistant_messages():
    history = _aged(_pair(0) + _pair(1), 2)
    # Result 0 is followed by 3 assistants (call 1 + 2 fillers), result 1 by 2.
    _, cleared = clear_aged_tool_results(history, after_calls=3, min_tokens=0, batch_tokens=0)
    assert cleared == 1
    first, second = _results(history)
    assert first.text.startswith("[Older result cleared")
    assert second.text == _FILLER


def test_nothing_fires_below_the_batch_threshold():
    history = _aged(_pair(0) + _pair(1), 40)
    total = sum(estimate_tokens(m) for m in _results(history))
    _, cleared = clear_aged_tool_results(history, batch_tokens=total + 1)
    assert cleared == 0
    assert all(m.text == _FILLER for m in _results(history))


def test_at_the_batch_threshold_every_eligible_result_clears_at_once():
    history = _aged(_pair(0) + _pair(1) + _pair(2), 40)
    total = sum(estimate_tokens(m) for m in _results(history))
    _, cleared = clear_aged_tool_results(history, batch_tokens=total)
    assert cleared == 3
    assert all(m.provider_payload["pruned"] is True for m in _results(history))


def test_results_below_min_tokens_are_neither_counted_nor_cleared():
    small = _pair(0, text="tiny output " * 5)
    history = _aged(small + _pair(1), 40)
    _, cleared = clear_aged_tool_results(history, min_tokens=500, batch_tokens=0)
    assert cleared == 1
    assert _results(history)[0].text.startswith("tiny output")


def test_errors_are_never_cleared():
    history = _aged(_pair(0, is_error=True), 40)
    assert clear_aged_tool_results(history, batch_tokens=0)[1] == 0


def test_skill_and_guide_reads_are_never_cleared():
    history = _aged(
        _pair(0, arguments={"path": "skill://minerva"}, details={"url": "skill://minerva"})
        + _pair(1, arguments={"path": "guide://delivery"}, details={"path": "guide://delivery"})
        + _pair(2, name="skill", arguments={"name": "x"}),
        40,
    )
    assert clear_aged_tool_results(history, batch_tokens=0)[1] == 0


def test_coordination_tools_are_never_cleared():
    history: list[Message] = []
    for index, name in enumerate(sorted(AGED_CLEAR_PROTECTED_TOOLS)):
        history += _pair(index, name=name, arguments={"job_id": f"j{index}"})
    history = _aged(history, 40)
    assert clear_aged_tool_results(history, batch_tokens=0)[1] == 0
    # The set is the one the brief names; a silent shrink would expose a
    # manager's handoffs to clearing.
    assert {"wait", "hub", "task", "jobs", "todo", "project", "ask", "sessions", "send"} <= (
        AGED_CLEAR_PROTECTED_TOOLS
    )


def test_already_pruned_results_are_left_alone():
    history = _aged(_pair(0), 40)
    result = _results(history)[0]
    result.content = [TextContent(text=USELESS_NOTICE + " " + _FILLER)]
    result.provider_payload = {"details": {}, "pruned": True}
    assert clear_aged_tool_results(history, batch_tokens=0)[1] == 0
    assert result.text.startswith(USELESS_NOTICE)


def test_second_pass_is_a_no_op():
    history = _aged(_pair(0) + _pair(1), 40)
    assert clear_aged_tool_results(history, batch_tokens=0)[1] == 2
    snapshot = [m.model_dump() for m in history]
    assert clear_aged_tool_results(history, batch_tokens=0)[1] == 0
    assert [m.model_dump() for m in history] == snapshot


def test_pairing_survives_and_prior_details_are_kept():
    history = _aged(_pair(0), 40)
    before = len(history)
    clear_aged_tool_results(history, batch_tokens=0)
    assert len(history) == before
    result = _results(history)[0]
    assert result.tool_call_id == "c0"
    assert result.provider_payload["details"]["path"] == "/repo/f0.py"


def test_stub_names_the_read_path_and_range():
    history = _aged(_pair(0, arguments={"path": "/repo/a.py", "range": "10-40"}), 40)
    clear_aged_tool_results(history, batch_tokens=0)
    assert _results(history)[0].text == (
        "[Older result cleared to save context \u2014 read /repo/a.py 10-40."
        " Re-run it if you still need it.]"
    )


def test_stub_names_a_full_read_without_a_range():
    history = _aged(_pair(0, arguments={"path": "/repo/a.py"}), 40)
    clear_aged_tool_results(history, batch_tokens=0)
    assert "\u2014 read /repo/a.py. Re-run" in _results(history)[0].text


def test_stub_quotes_the_bash_command_clipped_to_one_line():
    command = "git diff origin/main..HEAD -- local_operator/\n" + "x" * 300
    history = _aged(_pair(0, name="bash", arguments={"command": command}), 40)
    clear_aged_tool_results(history, batch_tokens=0)
    text = _results(history)[0].text
    assert "bash `git diff origin/main..HEAD -- local_operator/ xxx" in text
    assert "\n" not in text
    quoted = text.split("`")[1]
    assert len(quoted) == 100 and quoted.endswith("\u2026")


def test_stub_names_the_grep_pattern_and_scope():
    history = _aged(
        _pair(0, name="grep", arguments={"pattern": "def _blank", "path": "local_operator"}),
        40,
    )
    clear_aged_tool_results(history, batch_tokens=0)
    assert "\u2014 grep 'def _blank' in local_operator." in _results(history)[0].text


def test_stub_falls_back_to_the_tool_name():
    history = _aged(_pair(0, name="web_fetch", arguments={"url": "https://x.test/a"}), 40)
    history += _aged(_pair(1, name="eval", arguments={"code": "1"}), 40)
    clear_aged_tool_results(history, batch_tokens=0)
    first, second = _results(history)
    assert "\u2014 web_fetch https://x.test/a." in first.text
    assert "\u2014 the eval call." in second.text


def test_stub_is_deterministic():
    """Byte-stable: the same message always renders the same stub, or the
    stub itself would bust the prompt cache on every request it sits in."""
    one = _result("c0", "bash")
    two = _result("c0", "bash")
    args = {"command": "ls -la"}
    assert aged_clear_notice(one, args) == aged_clear_notice(two, args)
    assert aged_clear_notice(one, args) == aged_clear_notice(one, dict(args))


def test_non_positive_age_disables_the_pass():
    history = _aged(_pair(0), 40)
    assert clear_aged_tool_results(history, after_calls=0, batch_tokens=0)[1] == 0
