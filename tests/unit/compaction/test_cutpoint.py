"""Cut-point selection: hard pairing rules, snap-forward, partition shapes."""

import pytest

from local_operator.compaction.cutpoint import (
    _message_tokens,
    advance_cut_to_wire_band,
    extract_preserved_user_turns,
    find_cut_point,
    prepare_partitions,
)
from local_operator.compaction.tokens import _encode_len, estimate_tokens
from local_operator.harness.types import (
    CustomMessage,
    ImageContent,
    Message,
    TextContent,
    ToolCall,
)


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
    # Two real messages have only one nonempty prefix: no legal forward
    # alternative can satisfy the historical two-message minimum.
    messages2 = [_big_user(), _big_user(400)]
    keep2 = estimate_tokens(messages2[1]) + 10
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
    # Both halves of the predicate are load-bearing, and each is pinned by a
    # sibling test below: `all` (not `any`) by the partially-answered case, and
    # `>= index` (not `>= 0`) by the re-issued-id case.
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


def test_a_partially_answered_assistant_is_not_a_valid_cut():
    """`all`, not `any`: EVERY call of the boundary assistant must be answered.

    An assistant issuing two calls where only one result has arrived is the
    case that separates the two forms. Keeping it strands the unanswered call
    on the kept side with no result, which is the dangling-call corruption in
    the other direction from an orphaned result.

    This exists because an earlier comment here claimed `any` was an
    equivalent mutant, reasoning that an assistant always precedes its own
    results. That premise is true and does not bear on this case: what
    separates `all` from `any` is PARTIAL answering, not ordering. The claim
    was false and the coverage it argued against is this test.
    """
    partial = _assistant_with_call("answered")
    partial.tool_calls.append(ToolCall(id="never", name="bash", arguments={"command": "ls"}))
    messages = [
        _big_user(),
        _big_user(),
        _big_user(),
        partial,
        _tool_result("answered"),
        Message.user("tail"),
    ]

    # Sized so the backwards walk STOPS on the partial assistant: it is the cut
    # candidate, and `all` vs `any` decides whether it is accepted there. A
    # budget that lets the walk stop past it never consults the predicate at
    # all — the first version of this test did exactly that and passed under
    # the `any` mutant, pinning nothing.
    budget = sum(estimate_tokens(m) for m in messages[3:])
    cut = find_cut_point(messages, budget)

    assert cut is not None
    assert messages[cut] is not partial, (
        "the cut kept an assistant whose second call has no result, stranding a "
        "dangling call at the head of the kept window"
    )
    to_summarize, kept = prepare_partitions(messages, cut)
    assert_partition_pair_integrity(to_summarize, kept)


def test_an_assistant_reissuing_an_already_answered_id_is_not_a_valid_cut():
    """`>= index`, not `>= 0`: the result must follow THIS assistant.

    ``_result_indices`` keys by ``tool_call_id`` and last occurrence wins, so a
    re-issued id can resolve to a result that sits BEFORE the assistant that
    re-issued it. Comparing against ``0`` accepts that (a result exists
    somewhere); comparing against ``index`` is what requires the result to be
    on the kept side of this cut.
    """
    first = _assistant_with_call("x")
    reissued = _assistant_with_call("x")
    messages = [
        _big_user(),
        first,
        _tool_result("x"),
        reissued,
        Message.user("after " + "word " * 200),
    ]

    cut = find_cut_point(messages, estimate_tokens(messages[4]) + 10)

    assert cut is not None
    assert messages[cut] is not reissued, (
        "the cut kept an assistant whose only matching result precedes it, so "
        "the kept window opens with a call nothing answers"
    )


# ---------------------------------------------------------------------------
# Verbatim user-turn preservation: the structural half of "never summarize a
# user turn". These pin the extraction; the session tests pin the round trip.
# ---------------------------------------------------------------------------


def test_extract_preserves_every_user_turn_verbatim():
    """Every ``Message(role="user")`` in the summarized block comes back with
    its id and its EXACT text — no paraphrase, no dropped turn. Discriminates
    against summarizing the user side: the pre-fix path carried none of these
    out, so this list would be empty."""
    to_summarize = [
        Message.user("NEVER touch billing.py"),
        _assistant_with_call("c1"),
        _tool_result("c1"),
        Message.user("use the existing helper, don't add a new one"),
        Message.assistant("working on it"),
    ]
    preserved = extract_preserved_user_turns(to_summarize)
    assert [t["text"] for t in preserved] == [
        "NEVER touch billing.py",
        "use the existing helper, don't add a new one",
    ]
    assert preserved[0]["id"] == to_summarize[0].id
    assert preserved[1]["id"] == to_summarize[3].id


def test_extract_skips_non_user_and_empty_turns():
    """Assistant/tool content is NOT preserved verbatim (it is what the summary
    is for), and an empty user turn (a bare pasted screenshot) carries no
    constraint, so the preserved block never accrues blanks."""
    to_summarize = [
        Message.assistant("assistant prose"),
        _tool_result("c9"),
        Message(role="user", content=[]),
        Message.user("real instruction"),
    ]
    preserved = extract_preserved_user_turns(to_summarize)
    assert [t["text"] for t in preserved] == ["real instruction"]


def test_extract_id_filter_excludes_injected_user_role_content():
    """The production discriminator: a rendered history has prior markers and
    injected deliveries as plain user Messages. Only ids known to be genuine
    prompts are lifted, so a previous summary is never carried forward
    verbatim. Discriminates the no-filter path (which returns BOTH)."""
    genuine = Message.user("NEVER touch billing.py")
    rendered_marker = Message.user("<previous-context-summary>paraphrase</...>")
    to_summarize = [rendered_marker, genuine, Message.assistant("ok")]
    preserved = extract_preserved_user_turns(to_summarize, {genuine.id})
    assert [t["text"] for t in preserved] == ["NEVER touch billing.py"]


def test_extract_refuses_a_legacy_harness_notice_as_a_user_turn():
    """An unstamped notice is not a user turn, however it is shaped.

    A notice written before the ``harness_injected`` stamp existed (a real
    session carries eight switch-notice rows with no ``provider_payload`` at
    all) is a plain ``role="user"`` message, and harvesting one re-seats the
    harness's words as the operator's on every replay of the marker. The stamp
    cannot see it, so the notice's own head decides — and the operator's real
    turns are still lifted verbatim beside it.
    """
    notice = Message.user(
        "[model switch] You are now running as zai/glm-5.3 (was anthropic/claude-opus-5).\n"
        "Reason: provider failure"
    )
    genuine = Message.user("NEVER touch billing.py")
    to_summarize = [notice, Message.assistant("ok"), genuine]

    assert [t["text"] for t in extract_preserved_user_turns(to_summarize)] == [
        "NEVER touch billing.py"
    ]


# ---------------------------------------------------------------------------
# advance_cut_to_wire_band: the byte-ruled correction to the token-ruled cut
# ---------------------------------------------------------------------------

FRAME = 100_000  # bytes per synthetic observation; the helper is unit-agnostic


def _observation_run(groups: int, *, first_user_image: bool = False) -> list[Message]:
    """``user, (assistant call, tool result + image) * groups`` -- the shape of a
    screenshot-driving run, one FRAME-byte image per tool result."""
    messages: list[Message] = [Message.user("goal")]
    if first_user_image:
        messages[0] = Message(
            role="user",
            content=[TextContent(text="goal"), ImageContent(data="A" * FRAME)],
        )
    for index in range(groups):
        call_id = f"c{index}"
        messages.append(_assistant_with_call(call_id))
        messages.append(
            Message(
                role="tool",
                tool_call_id=call_id,
                tool_name="computer",
                content=[TextContent(text="ok"), ImageContent(data="A" * FRAME)],
            )
        )
    return messages


def _kept_bytes_of(messages: list[Message]):
    """The caller-side ruler: image bytes only, no ``fixed`` prefix."""
    sizes = [sum(len(b.data) for b in m.content if isinstance(b, ImageContent)) for m in messages]
    return lambda index: sum(sizes[index:])


def test_advance_moves_the_cut_later_until_the_kept_side_fits_the_band():
    messages = _observation_run(12)
    # A token-ruled cut that kept 10 frames: over an 8-frame band.
    cut = 3  # index 3 = assistant of group 1; keeps groups 1..11 = 11 frames
    advanced = advance_cut_to_wire_band(
        messages, cut, kept_bytes=_kept_bytes_of(messages), band_bytes=8 * FRAME
    )
    assert advanced > cut, "the cut did not move although the kept side was over the band"
    assert _kept_bytes_of(messages)(advanced) <= 8 * FRAME
    # FIRST fitting cut, not an over-shoot: one legal cut earlier would not fit.
    assert _kept_bytes_of(messages)(advanced - 2) > 8 * FRAME


def test_advance_is_inert_when_the_kept_side_already_fits_or_the_band_is_off():
    messages = _observation_run(12)
    kept = _kept_bytes_of(messages)
    assert advance_cut_to_wire_band(messages, 9, kept_bytes=kept, band_bytes=8 * FRAME) == 9
    assert advance_cut_to_wire_band(messages, 3, kept_bytes=kept, band_bytes=0) == 3


def test_advance_never_lands_on_a_tool_result_or_orphans_a_call():
    messages = _observation_run(12)
    advanced = advance_cut_to_wire_band(
        messages, 3, kept_bytes=_kept_bytes_of(messages), band_bytes=5 * FRAME
    )
    landed = messages[advanced]
    assert landed.role != "tool"
    answered_after = {m.tool_call_id for m in messages[advanced:] if m.role == "tool"}
    for summarized in messages[:advanced]:
        assert not any(call.id in answered_after for call in summarized.tool_calls)


def test_advance_never_passes_the_newest_observation():
    messages = _observation_run(12)
    # A band no cut can satisfy: even keeping only the newest frame is over it.
    advanced = advance_cut_to_wire_band(
        messages, 3, kept_bytes=_kept_bytes_of(messages), band_bytes=FRAME // 2
    )
    newest_image = max(
        i for i, m in enumerate(messages) if any(isinstance(b, ImageContent) for b in m.content)
    )
    assert advanced <= newest_image
    # ... and it is the furthest legal cut: the assistant that produced it.
    assert messages[advanced] is messages[newest_image - 1]


def test_advance_never_passes_a_user_attached_image_in_the_kept_window():
    messages = _observation_run(12)
    attached = Message(
        role="user", content=[TextContent(text="look at this"), ImageContent(data="A" * FRAME)]
    )
    messages.insert(7, attached)  # a person pasted a screenshot mid-run
    advanced = advance_cut_to_wire_band(
        messages, 3, kept_bytes=_kept_bytes_of(messages), band_bytes=2 * FRAME
    )
    assert advanced <= 7, "the cut summarized away an image the user attached"


def test_advance_honours_the_replayability_guard():
    messages = _observation_run(12)
    replayable_from = 9  # only entries at/after index 9 are transcript entries
    advanced = advance_cut_to_wire_band(
        messages,
        3,
        kept_bytes=_kept_bytes_of(messages),
        band_bytes=5 * FRAME,
        replayable=lambda index: index >= replayable_from,
    )
    assert advanced >= replayable_from
    # Nothing replayable at all -> the original cut, never an unreplayable one.
    assert (
        advance_cut_to_wire_band(
            messages,
            3,
            kept_bytes=_kept_bytes_of(messages),
            band_bytes=5 * FRAME,
            replayable=lambda index: False,
        )
        == 3
    )


def test_interrupted_tool_pair_keeps_issuing_call():
    messages = [
        *[_big_user() for _ in range(7)],
        _assistant_with_call("mcp__episode_actions_apply_actions:12"),
        Message.user("interleaved delivery"),
        _tool_result("mcp__episode_actions_apply_actions:12"),
        Message.assistant("done"),
    ]
    keep = sum(estimate_tokens(m) for m in messages[8:])
    cut = find_cut_point(messages, keep)
    assert cut == 7
    assert_partition_pair_integrity(*prepare_partitions(messages, cut))


def test_interleaved_multi_call_turn_moves_all_pairs_together():
    first = _assistant_with_call("a")
    first.tool_calls.append(ToolCall(id="b", name="bash", arguments={}))
    messages = [
        _big_user(),
        _big_user(),
        first,
        Message.user("delivery one"),
        _tool_result("a"),
        Message.assistant("still waiting"),
        Message.user("delivery two"),
        _tool_result("b"),
        Message.assistant("done"),
    ]
    for index in (3, 4, 5, 6):
        keep = sum(estimate_tokens(m) for m in messages[index:])
        cut = find_cut_point(messages, keep)
        expected = 2 if sum(estimate_tokens(m) for m in messages[2:]) <= 2 * keep else 8
        assert cut == expected
        assert_partition_pair_integrity(*prepare_partitions(messages, cut))
    # Byte-driven advancement uses the SAME legality rule, not just token cuts.
    advanced = advance_cut_to_wire_band(
        messages, 2, band_bytes=1, kept_bytes=lambda index: len(messages) - index
    )
    assert advanced == 8
    assert_partition_pair_integrity(*prepare_partitions(messages, advanced))


def test_reverse_order_pair_is_not_split():
    messages = [
        _big_user(),
        _big_user(),
        Message.user("before reversed pair"),
        _tool_result("reverse"),
        Message.user("delivery"),
        _assistant_with_call("reverse"),
        Message.assistant("done"),
    ]
    keep = sum(estimate_tokens(m) for m in messages[4:])
    cut = find_cut_point(messages, keep)
    # Retaining the large result would exceed twice this small keep budget.
    assert cut == 6
    assert_partition_pair_integrity(*prepare_partitions(messages, cut))


def test_no_safe_nonempty_tail_still_raises_for_missing_issuer():
    # There is no issuing call anywhere and the last message is its result.
    # Every nonempty kept suffix is invalid; snapping cannot repair the input.
    messages = [_big_user(), _big_user(), Message.user("delivery"), _tool_result("missing")]
    keep = sum(estimate_tokens(m) for m in messages[2:])
    with pytest.raises(AssertionError, match="without issuing calls anywhere"):
        find_cut_point(messages, keep)


@pytest.mark.parametrize("reverse", [False, True])
def test_final_pair_guard_survives_a_broken_selector(monkeypatch, reverse):
    from local_operator.compaction import cutpoint

    pair = [_assistant_with_call("guard"), _tool_result("guard")]
    if reverse:
        pair.reverse()
    messages = [_big_user(), _big_user(), pair[0], Message.user("delivery"), pair[1]]
    monkeypatch.setattr(cutpoint, "_snap_to_valid_cut", lambda *args: 3)
    keep = sum(estimate_tokens(m) for m in messages[3:])
    with pytest.raises(AssertionError, match="cut point 3 orphans tool"):
        find_cut_point(messages, keep)


def test_normal_compaction_partitions_are_byte_identical():
    import runpy
    from pathlib import Path

    baseline = runpy.run_path(str(Path(__file__).with_name("pre_pair_cutpoint_control.py")))
    messages = []
    for i in range(8):
        messages.extend(
            [
                _big_user(100),
                _assistant_with_call(f"normal-{i}"),
                _tool_result(f"normal-{i}", 100),
                Message.assistant("answer " * 80),
            ]
        )
    checked = 0
    for keep in range(1, sum(estimate_tokens(m) for m in messages) + 1, 7):
        old_cut = baseline["find_cut_point"](messages, keep)
        if old_cut is None:
            # R1 intentionally changes non-progressing refusals. Compatibility
            # here means byte identity for passes the original actually ran.
            continue
        cut = find_cut_point(messages, keep)
        assert cut == old_cut, keep
        if cut is not None:
            old_parts = baseline["prepare_partitions"](messages, old_cut)
            parts = prepare_partitions(messages, cut)
            assert [[m.model_dump_json() for m in part] for part in parts] == [
                [m.model_dump_json() for m in part] for part in old_parts
            ]
        checked += 1
    assert checked > 300


@pytest.mark.parametrize("extra", [0, 61, 121])
def test_pair_near_start_must_make_progress(extra):
    call = _assistant_with_call("a")
    call.tool_calls.append(ToolCall(id="b", name="bash", arguments={}))
    messages = [
        _big_user(121),
        call,
        _tool_result("a", 200),
        Message.user("delivery"),
        _tool_result("b", 200),
        Message.assistant("answer " * 60),
        _big_user(60),
        Message.assistant("latest " * 60),
    ]
    keep = sum(estimate_tokens(m) for m in messages[3:]) + extra
    cut = find_cut_point(messages, keep)
    assert cut == 5
    assert sum(estimate_tokens(m) for m in messages[:cut]) > 500
    assert_partition_pair_integrity(*prepare_partitions(messages, cut))


def test_missing_interior_issuer_is_repaired_forwards():
    messages = [
        _big_user(),
        _big_user(),
        Message.user("delivery"),
        _tool_result("missing"),
        Message.assistant("done"),
    ]
    cut = find_cut_point(messages, 202)
    assert cut == 4
    assert_partition_pair_integrity(*prepare_partitions(messages, cut))


@pytest.mark.parametrize("prefix_count", [0, 1])
def test_oversized_pair_does_not_pin_cut_to_start(prefix_count):
    call = _assistant_with_call("a")
    call.tool_calls.append(ToolCall(id="b", name="bash", arguments={}))
    messages = [
        *[_big_user() for _ in range(prefix_count)],
        call,
        _tool_result("a", 15000),
        Message.user("delivery"),
        _tool_result("b", 15000),
        Message.assistant("done"),
        Message.user("next"),
    ]
    # A broad sweep pins progress even when the token walk starts on the call,
    # on either tool result, or on the interleaved delivery.
    for keep in range(1000, 29000, 2000):
        cut = find_cut_point(messages, keep)
        assert cut is not None, (prefix_count, keep)
        assert cut >= prefix_count + 4
        assert sum(estimate_tokens(m) for m in messages[:cut]) > 29000
        assert_partition_pair_integrity(*prepare_partitions(messages, cut))


def test_oversized_backward_repair_is_bounded_even_with_useful_prefix():
    call = _assistant_with_call("a")
    call.tool_calls.append(ToolCall(id="b", name="bash", arguments={}))
    messages = [
        _big_user(),
        _big_user(),
        call,
        _tool_result("a", 15000),
        Message.user("delivery"),
        _tool_result("b", 200),
        Message.assistant("done"),
    ]
    keep = sum(estimate_tokens(m) for m in messages[4:])
    assert sum(estimate_tokens(m) for m in messages[2:]) > 2 * keep
    cut = find_cut_point(messages, keep)
    assert cut == 6
    assert sum(estimate_tokens(m) for m in messages[:cut]) > 15000
    assert_partition_pair_integrity(*prepare_partitions(messages, cut))


def test_replayed_tail_does_not_spuriously_veto_backward_retention():
    from local_operator.compaction.cutpoint import PRESERVED_USER_TURN_KEY

    call = _assistant_with_call("a")
    call.tool_calls.append(ToolCall(id="b", name="bash", arguments={}))
    preserved = _big_user(10000)
    preserved.provider_payload = {PRESERVED_USER_TURN_KEY: True}
    messages = [
        _big_user(),
        _big_user(),
        call,
        _tool_result("a", 10),
        preserved,
        _tool_result("b", 200),
        Message.assistant("done"),
    ]
    assert find_cut_point(messages, 300) == 2
    assert_partition_pair_integrity(*prepare_partitions(messages, 2))
