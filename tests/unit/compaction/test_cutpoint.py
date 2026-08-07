"""Cut-point selection: hard pairing rules, snap-forward, partition shapes."""

import pytest

from local_operator.compaction.cutpoint import (
    _message_tokens,
    find_cut_point,
    prepare_partitions,
)
from local_operator.compaction.tokens import _encode_len, estimate_tokens
from local_operator.harness.types import CustomMessage, Message, ToolCall


def _big_user(words: int = 200) -> Message:
    return Message.user("word " * words)


def _assistant_with_call(call_id: str) -> Message:
    message = Message.assistant("running the tool now")
    message.tool_calls = [ToolCall(id=call_id, name="bash", arguments={"command": "ls"})]
    return message


def _tool_result(call_id: str, words: int = 200) -> Message:
    return Message(
        role="tool",
        content=Message.user("out " * words).content,
        tool_call_id=call_id,
        tool_name="bash",
    )


def test_cut_never_on_tool_result_or_pending_call_assistant():
    """user -> assistant(tool_calls) -> tool -> assistant: the cut must land on
    a message that keeps the call/result pair on one side."""
    messages = [
        Message.user("start"),
        _assistant_with_call("c1"),
        _tool_result("c1"),
        _big_user(),  # heavy message forces the backwards walk to stop early
        Message.user("and then we continued"),
    ]
    # Keep budget that forces the walk to stop inside the tool-call cluster.
    keep = estimate_tokens(messages[4]) + estimate_tokens(messages[3]) + 10
    cut = find_cut_point(messages, keep)
    assert cut is not None
    candidate = messages[cut]
    assert candidate.role != "tool"
    if candidate.role == "assistant":
        assert not candidate.tool_calls

    to_summarize, kept = prepare_partitions(messages, cut)
    assert to_summarize + kept == messages
    assert kept[0] is messages[cut]


def test_snap_forward_walks_past_tool_cluster():
    """When the walk stops ON the tool result, snapping must move forward to
    the next legal message, never backwards into the cluster."""
    messages = [
        _big_user(),
        _big_user(),
        _assistant_with_call("c1"),
        _tool_result("c1"),
        Message.user("after"),
    ]
    keep = estimate_tokens(messages[4]) + estimate_tokens(messages[3])
    cut = find_cut_point(messages, keep)
    assert cut is not None
    # Walk stops at index 3 (tool) or 2 (assistant w/ calls); snap lands on 4.
    assert messages[cut] is messages[4]


def test_no_cut_when_everything_is_recent():
    """All tokens inside the keep budget -> nothing worth summarizing."""
    messages = [Message.user("a"), Message.user("b"), Message.user("c")]
    total = sum(estimate_tokens(m) for m in messages)
    assert find_cut_point(messages, total + 100) is None


def test_no_cut_when_cut_would_be_trivial():
    """cut <= 1 means zero or one message to summarize -> None."""
    messages2 = [_big_user(), _big_user(400), Message.user("tail")]
    keep2 = estimate_tokens(messages2[2]) + 10
    assert find_cut_point(messages2, keep2) is None


def test_none_when_snap_runs_past_end():
    """Trailing tool cluster with nothing valid after it -> None."""
    messages = [
        _big_user(),
        _big_user(),
        _assistant_with_call("c1"),
        _tool_result("c1"),
    ]
    keep = estimate_tokens(messages[3]) + 10
    assert find_cut_point(messages, keep) is None


def test_empty_and_zero_budget():
    assert find_cut_point([], 1000) is None
    assert find_cut_point([_big_user()], 0) is None


def test_compaction_summary_marker_is_valid_cut():
    """A prior compaction marker may serve as the cut boundary."""
    marker = CustomMessage(custom_type="compaction_summary", details={"summary": "s" * 50})
    big = _big_user()
    tail = Message.user("tail")
    messages = [
        Message.user("old1"),
        Message.user("old2"),
        marker,
        big,
        tail,
    ]
    # Budget that exhausts exactly at the marker: big + tail alone are short.
    keep = estimate_tokens(big) + estimate_tokens(tail) + 5
    cut = find_cut_point(messages, keep)
    assert cut is not None
    assert messages[cut] is marker


def test_partition_shapes_and_validation():
    messages = [Message.user("a"), Message.user("b"), Message.user("c")]
    to_summarize, kept = prepare_partitions(messages, 2)
    assert len(to_summarize) == 2 and len(kept) == 1
    assert to_summarize[0] is messages[0] and kept[0] is messages[2]
    with pytest.raises(ValueError):
        prepare_partitions(messages, 0)
    with pytest.raises(ValueError):
        prepare_partitions(messages, 4)


# ---------------------------------------------------------------------------
# RC-10: canonical sequence + partition pair-integrity sweep
# ---------------------------------------------------------------------------


def _canonical_sequence() -> list[Message]:
    """user → assistant(tool_calls) → tool → tool → assistant → user.

    The assistant issues two calls (c1, c2); the two tool results answer one
    each. Sizes vary so the backwards walk can stop at every index.
    """
    assistant = Message.assistant("let me inspect both files")
    assistant.tool_calls = [
        ToolCall(id="c1", name="read", arguments={"path": "/repo/a.py"}),
        ToolCall(id="c2", name="read", arguments={"path": "/repo/b.py"}),
    ]
    tool_a = Message(
        role="tool",
        content=Message.user("file a " + "data " * 400).content,
        tool_call_id="c1",
        tool_name="read",
    )
    tool_b = Message(
        role="tool",
        content=Message.user("b").content,
        tool_call_id="c2",
        tool_name="read",
    )
    return [
        Message.user("please refactor the module"),
        assistant,
        tool_a,
        tool_b,
        Message.assistant("done " * 60),
        Message.user("great, now run the tests"),
    ]


def assert_partition_pair_integrity(
    to_summarize: list[Message | CustomMessage], kept: list[Message | CustomMessage]
) -> None:
    """Every tool_call_id answered in ``kept`` has its issuing assistant in
    ``kept``; no call issued in ``to_summarize`` is answered in ``kept``."""
    kept_call_ids = {c.id for m in kept if isinstance(m, Message) for c in m.tool_calls}
    summarized_call_ids = {
        c.id for m in to_summarize if isinstance(m, Message) for c in m.tool_calls
    }
    for message in kept:
        if not isinstance(message, Message) or message.role != "tool":
            continue
        call_id = message.tool_call_id
        assert (
            call_id in kept_call_ids
        ), f"tool result {call_id} is kept but its issuing assistant is not"
        assert (
            call_id not in summarized_call_ids
        ), f"tool result {call_id} is kept but its call was summarized away"


_CANONICAL = _canonical_sequence()
_TOTAL_TOKENS = sum(estimate_tokens(m) for m in _CANONICAL)


def _keep_values(messages: list[Message]) -> list[int]:
    """keep_recent_tokens values that make the backwards walk stop at every
    index: each cumulative suffix sum, plus 0."""
    values = [0]
    suffix = 0
    for message in reversed(messages):
        suffix += estimate_tokens(message)
        values.append(suffix)
    return sorted(set(values))


@pytest.mark.parametrize("keep", _keep_values(_CANONICAL))
def test_canonical_partition_integrity_sweep(keep: int):
    """Parametrized over every stop index of the keep budget (0..total): each
    yield of ``find_cut_point`` must produce a legal partition."""
    cut = find_cut_point(_CANONICAL, keep)
    if cut is None:
        return
    to_summarize, kept = prepare_partitions(_CANONICAL, cut)
    assert to_summarize + kept == _CANONICAL
    candidate = _CANONICAL[cut]
    assert candidate.role != "tool"
    if candidate.role == "assistant":
        # Any pending calls of the boundary assistant are answered in kept.
        answered = {m.tool_call_id for m in kept if isinstance(m, Message) and m.role == "tool"}
        assert all(c.id in answered for c in candidate.tool_calls)
    assert_partition_pair_integrity(to_summarize, kept)


def test_canonical_sweep_covers_range():
    """The sweep spans the full 0..total range and finds at least one cut."""
    assert _keep_values(_CANONICAL)[-1] == _TOTAL_TOKENS
    cuts = [find_cut_point(_CANONICAL, keep) for keep in _keep_values(_CANONICAL)]
    assert any(c is not None for c in cuts)


def test_marker_tokens_use_raw_encoder():
    """RC-12: compaction markers count via the raw encoder, no throwaway
    Message wrapper."""
    summary = "summary text " * 50
    marker = CustomMessage(custom_type="compaction_summary", details={"summary": summary})
    assert _message_tokens(marker) == _encode_len(summary)
