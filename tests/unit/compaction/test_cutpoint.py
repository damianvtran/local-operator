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


def test_snap_never_lands_on_a_tool_result():
    """When the walk stops ON a tool result, the snap must leave it, and the
    resulting partition must keep every call with its result.

    The cut may land on the ISSUING assistant (index 2) rather than past the
    whole cluster: that keeps the call and its result together on the kept
    side, which is the actual pairing rule. Requiring the cut to clear the
    cluster entirely was stricter than the invariant, and inside a long tool
    run it left no legal cut at all — see ``_is_valid_cut``.
    """
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
    assert messages[cut].role != "tool"
    to_summarize, kept = prepare_partitions(messages, cut)
    assert_partition_pair_integrity(to_summarize, kept)
    # The pass has to be worth making: the summarized side is non-empty.
    assert to_summarize


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


def test_no_cut_when_only_a_previous_summary_and_one_message_precede_it():
    """A prior marker is not history to summarize — it is already a summary.

    The state an on-demand ``/compact`` pressed straight after a pass lands in:
    the context is ``[marker, older, recent…]``, so a raw index test sees two
    messages before the cut and re-summarizes the previous summary plus one
    message, spending a provider call and a full cache rewrite to compress what
    is already compressed. Counted over REAL messages, there is one, and one is
    not worth the pass.
    """
    marker = CustomMessage(custom_type="compaction_summary", details={"summary": "s" * 400})
    # Held by name, not read back out of the mixed list: the keep budget is a
    # fact about THIS message, and indexing the list hands back the
    # ``Message | CustomMessage`` union that ``estimate_tokens`` cannot take.
    tail = Message.user("tail")
    messages = [marker, _big_user(), tail]
    keep = estimate_tokens(tail) + 10
    assert find_cut_point(messages, keep) is None

    # Two real messages before the cut and the pass IS worth running — the rule
    # is "fewer than two summarizable", not "any marker present".
    with_two = [marker, _big_user(), _big_user(), _big_user(), Message.user("tail")]
    cut = find_cut_point(with_two, keep)
    assert cut == 3
    assert [m for m in with_two[:cut] if m is not marker] == with_two[1:3]


def test_one_summarizable_message_still_runs_when_it_is_the_reason_the_window_is_full():
    """The marker exclusion must not block a pass over real tokens.

    Counting messages answers in the wrong unit. ``[marker, X, …]`` cut at
    index 2 leaves ONE summarizable message, which the count rule refuses and
    the older ``index <= 1`` rule ran. When X is a huge tool result it is the
    whole reason the context is full: refusing hands the AUTOMATIC trigger
    nothing to do while the window keeps filling, which is a pass blocked
    forever by the arithmetic meant to skip trivial ones.

    The line is the recency budget, which is the only scale in scope: a lone
    message that outweighs everything the caller asked to protect is not a
    trivial rewrite.
    """
    marker = CustomMessage(custom_type="compaction_summary", details={"summary": "s" * 400})
    tail = Message.user("tail")
    huge = _big_user(20_000)
    recent = _big_user(600)
    # The budget is met by `recent` alone, so the backwards walk stops there and
    # `huge` lands on the SUMMARIZE side with the marker — the shape where the
    # count rule and the token question disagree.
    keep = estimate_tokens(recent)

    cut = find_cut_point([marker, huge, recent, tail], keep)
    assert cut == 2

    # And the case the exclusion exists for is untouched: one SMALL message
    # after a pass is still not worth a provider call and a cache rewrite.
    assert find_cut_point([marker, _big_user(40), recent, tail], keep) is None


def test_history_ending_mid_tool_run_still_compacts():
    """A history captured MID-RUN ends in a tool cluster with nothing after
    it, and it must STILL yield a legal cut.

    This is the mid-turn compaction bug in miniature. The old rule treated a
    trailing cluster as uncuttable and answered ``None``, which the session
    reports as "nothing to compact" — so a long tool run could not be
    compacted at any size, sailed past the configured threshold, and got
    relief only once the run ended. The cut lands on the issuing assistant,
    which keeps the call/result pair intact.
    """
    messages = [
        _big_user(),
        _big_user(),
        _assistant_with_call("c1"),
        _tool_result("c1"),
    ]
    keep = estimate_tokens(messages[3]) + 10
    cut = find_cut_point(messages, keep)
    assert cut is not None
    assert messages[cut].role != "tool"
    to_summarize, kept = prepare_partitions(messages, cut)
    assert to_summarize
    assert_partition_pair_integrity(to_summarize, kept)


def test_a_cut_never_keeps_an_assistant_whose_call_is_unanswered():
    """The SAFETY half of the loosened rule, pinned against a live mutant.

    `_is_valid_cut` admits an assistant whose own calls are answered at or
    after the cut. The inverse must stay refused: an assistant holding a call
    with NO result anywhere would, if kept, hand the provider a dangling tool
    call, which is the same class of corruption as orphaning a result.

    This is deliberately shaped so the *pairing* rule is what decides. An
    earlier version of this test used a two-message history, which returns
    ``None`` from the triviality rule (`index <= 1`) before the predicate is
    ever consulted — it passed whatever `_is_valid_cut` did, so it pinned
    nothing. Here there is ample summarizable history and a legal cut exists
    (the trailing user message), so a cut IS returned and the only question is
    where it may land.

    Mutating the predicate's default from `-1` to a large value (treating an
    unanswered call as answered) makes this fail, which is what a regression
    would do.
    """
    unanswered = _assistant_with_call("never-answered")
    messages = [
        _big_user(),
        _big_user(),
        _big_user(),
        unanswered,
        Message.user("after " + "word " * 200),
    ]
    cut = find_cut_point(messages, estimate_tokens(messages[4]) + 10)
    assert cut is not None, "this history has a legal cut and must produce one"
    assert messages[cut] is not unanswered, (
        "the cut kept an assistant whose tool call has no result anywhere, "
        "which hands the provider a dangling call"
    )

    # The PROPERTY, over every budget that makes the walk stop at a different
    # index, rather than the single mutant above — pinning one mutant leaves
    # siblings alive (`>= 0` and `any`-for-`all` both survived a single-budget
    # version of this test).
    #
    # Two sibling mutants (`>= 0` for `>= index`, and `any` for `all`) survive
    # this and are EQUIVALENT rather than uncaught: an assistant always
    # precedes its own results in a well-formed history, so `result_at[id]` is
    # either `-1` (no result at all, which every variant rejects) or greater
    # than the assistant's own index — the three forms cannot disagree on any
    # history the harness can produce. The `-1` default is the load-bearing
    # part, and the mutant that changes it IS caught.
    #
    # The property is about the CUT INDEX, which is all ``_is_valid_cut``
    # governs: an assistant holding an unanswered call may never BE the cut.
    # It deliberately does not assert that no such assistant is anywhere in the
    # kept window, because a cut landing EARLIER than one keeps it, and that is
    # true on `origin/main` too — an unanswered call is a property of the
    # history the caller supplied, not something the cut point introduces.
    # Asserting the wider claim fails on unmutated code, which is how this
    # comment came to be here.
    budgets = {estimate_tokens(m) for m in messages}
    budgets |= {sum(estimate_tokens(m) for m in messages[i:]) for i in range(len(messages))}
    answered_anywhere = {
        m.tool_call_id for m in messages if isinstance(m, Message) and m.role == "tool"
    }
    checked = 0
    for budget in sorted(b for b in budgets if b > 0):
        candidate_cut = find_cut_point(messages, budget)
        if candidate_cut is None:
            continue
        checked += 1
        boundary = messages[candidate_cut]
        assert boundary.role != "tool", f"budget {budget}: cut landed on a tool result"
        if isinstance(boundary, Message) and boundary.tool_calls:
            unresolved = [c.id for c in boundary.tool_calls if c.id not in answered_anywhere]
            assert not unresolved, (
                f"budget {budget}: the cut is an assistant whose calls {unresolved} "
                "are answered nowhere, so the kept window opens with a dangling call"
            )
    assert checked, "no budget produced a cut, so this test asserted nothing"


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
