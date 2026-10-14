"""The quiet-group fold's block layer: classification, plan, and the list fold.

The definition and its cross-client fixture live in ``tui/quiet_groups.py``
(replayed by ``test_quiet_groups.py``), and the mounted widget is pinned in
``test_peer_message.py`` beside the receipts it folds. This file covers the
seam between them without booting an app: which block becomes which record,
which spans a run refuses, and the off-screen list fold a backward page uses
(``fold_quiet_group_list``), which no mounted test reaches because it exists
for runs that are inserted above the viewport.
"""

from __future__ import annotations

from local_operator.tui.quiet_groups import group_splitter_of, quiet_family_of
from local_operator.tui.session_presentation import OlderHistoryNotice
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.image_block import ImageBlock
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import (
    AskResponseBlock,
    HubMessageBlock,
    PeerMessageBlock,
    QuietGroupBlock,
    UserBlock,
    WakeBlock,
    fold_quiet_group_list,
    quiet_group_plan,
    quiet_group_record_of,
)

SENDER = {"pid": 7, "conversation_name": "peer-a"}


def _peer(body: str = "note") -> PeerMessageBlock:
    return PeerMessageBlock(body, SENDER)


def _tool(call_id: str = "t1", name: str = "bash") -> ToolCard:
    return ToolCard(call_id, name, {"command": "ls"})


# -- classification ----------------------------------------------------------


def test_each_row_classifies_into_the_derivations_vocabulary() -> None:
    assert quiet_group_record_of(_peer()).kind == "peer"
    assert quiet_group_record_of(WakeBlock("(alarm) wake w1")).kind == "wake"
    assert quiet_group_record_of(_tool()).kind == "tool"
    # A row this surface cannot positively classify falls to ``other`` and
    # SPLITS — the safe direction, and the only fact the derivation reads it
    # for (a user prompt must never fold inside a bar).
    assert quiet_group_record_of(UserBlock("hi")).kind == "other"
    assert group_splitter_of(quiet_group_record_of(UserBlock("hi")))
    assistant = AssistantBlock()
    assistant.update_text("on it")
    assistant.finalize_text()
    assistant_record = quiet_group_record_of(assistant)
    assert assistant_record.kind == "assistant"
    assert group_splitter_of(assistant_record), "visible prose splits"
    assert quiet_family_of(quiet_group_record_of(_peer())) == "peer"
    # BELT (S5 remediation): an assistant row whose visible text has not
    # landed is a BOUNDARY, never a foldable interior row — it may still
    # become the answer a group would then hide.
    empty = AssistantBlock()
    assert group_splitter_of(quiet_group_record_of(empty))
    pending = AssistantBlock()
    pending.update_text("   ")  # whitespace paints nothing and is not prose
    assert group_splitter_of(quiet_group_record_of(pending))


def test_an_ask_receipt_splits_like_the_relay_port() -> None:
    """An ask card is a QUESTION to the reader — the row they came back to
    answer, so a boundary, never a member a collapse could hide (review
    round 1, MAJOR-2). ``AskResponseBlock`` is a ``WakeBlock`` by inheritance
    (the shared ledger contract), not by meaning; the relay port names
    ``ask_response``/``ask_timeout`` as splitters the same way."""
    ask = AskResponseBlock({"text": "asked about the deploy"}, kind="response")
    assert quiet_group_record_of(ask).kind == "ask"
    assert group_splitter_of(quiet_group_record_of(ask))
    timeout = AskResponseBlock({"text": "asked about the rollback"}, kind="timeout")
    assert quiet_group_plan([_peer("one"), ask]) == []
    assert quiet_group_plan([ask, timeout]) == []


def test_the_quiet_pair_maps_to_the_sentinel_tool_name() -> None:
    """The record keeps the call's own name, so the module's exclusion (and
    the fixture's) can see it — the pair must be excludable from action
    counts even if a mixed-build runtime ever paints its row."""
    record = quiet_group_record_of(_tool("q1", "no_reply"))
    assert record.kind == "tool"
    assert record.tool_name == "no_reply"


def test_a_failed_card_is_a_failed_call_and_interrupted_is_not() -> None:
    failed = _tool("t1")
    failed.mark_failed("boom")
    assert quiet_group_record_of(failed).tool_state == "failed"
    stopped = _tool("t2")
    stopped.mark_interrupted()
    record = quiet_group_record_of(stopped)
    assert record.tool_state == "interrupted"
    assert record.tool_state != "failed", "a stopped call is not a failure"


def test_an_image_folds_with_the_tool_row_it_follows_and_splits_alone() -> None:
    image = ImageBlock(None, "image/png")
    assert quiet_group_record_of(image, prev_kind="tool").kind == "inside"
    assert quiet_group_record_of(image, prev_kind="peer").kind == "image"
    assert quiet_group_record_of(image).kind == "image"


# -- the plan ----------------------------------------------------------------


def test_an_interior_run_plans_one_group_spanning_its_tool_rows() -> None:
    run = [_peer("one"), _tool(), _peer("two")]
    (planned,) = quiet_group_plan(run, head_cut=False)
    assert planned.span == (0, 2)
    assert planned.group.count == 2
    assert planned.group.actions == 1
    assert planned.span_head_loaded is True


def test_the_run_edge_states_a_minimum_when_the_caller_reports_a_cut() -> None:
    run = [_peer("one"), _peer("two")]
    (planned,) = quiet_group_plan(run, head_cut=True)
    assert planned.span_head_loaded is False, "the caller's head-cut verdict"
    assert planned.group.first_ts is None and planned.group.last_ts is None


def test_a_fragment_head_is_refused_and_the_rows_keep_their_cards() -> None:
    """Rows above the run are ON HAND and in-span (the run begins mid-run):
    a bar would state a count over part of something larger — the module
    refuses the same sub-span, and so does the plan."""
    run = [_peer("one"), _peer("two")]
    assert quiet_group_plan(run, prev_block=_tool("t0"), head_cut=True) == []


def test_a_fragment_tail_is_refused() -> None:
    run = [_peer("one"), _peer("two")]
    assert quiet_group_plan(run, next_block=_peer("three")) == []


def test_a_head_notice_above_the_run_states_the_same_minimum() -> None:
    """The windowed view's boundary row the reader can SEE: it says older rows
    exist without being loaded, so a span beginning below it is cut too."""
    notice = OlderHistoryNotice("older messages above — scroll up to load")
    run = [_peer("one"), _peer("two")]
    (planned,) = quiet_group_plan(run, prev_block=notice)
    assert planned.span_head_loaded is False


def test_a_lone_trigger_is_not_a_group() -> None:
    assert quiet_group_plan([_peer("one")]) == []
    assert quiet_group_plan([_tool(), _peer("one"), _tool("t2")]) == []


def test_a_splitter_bounds_the_span_within_the_run() -> None:
    run = [_peer("one"), _peer("two"), UserBlock("steer"), _peer("three"), _peer("four")]
    plan = quiet_group_plan(run)
    assert [planned.span for planned in plan] == [(0, 1), (3, 4)]
    assert plan[0].span_head_loaded is True
    assert plan[0].group.open is False
    assert plan[1].group.open is True


def test_the_plan_derives_twice_identically() -> None:
    run = [_peer("one"), _tool(), _peer("two")]
    first = quiet_group_plan(run, head_cut=True)
    second = quiet_group_plan(run, head_cut=True)
    assert first == second


# -- the list fold (the backward-page seam) ----------------------------------


def test_the_list_fold_splices_a_bar_and_folds_its_members() -> None:
    run = [_peer("one"), _tool(), _peer("two")]
    folded = fold_quiet_group_list(run, fold_width=100)
    assert isinstance(folded[0], QuietGroupBlock)
    assert folded[1:] == run, "members keep their order and identity"
    bar = folded[0]
    assert bar.group.count == 2
    assert bar.partial is False
    assert all(not member.display for member in run), "the mount paints them folded"


def test_the_list_fold_refuses_a_fragment_and_touches_nothing() -> None:
    run = [_peer("one"), _peer("two")]
    folded = fold_quiet_group_list(run, prev_block=_tool("t0"))
    assert folded == run
    assert all(member.display for member in run)


def test_the_list_fold_is_stable_across_two_derivations() -> None:
    run = [_peer("one"), _tool(), _peer("two")]
    first = fold_quiet_group_list(run, fold_width=100)
    second = fold_quiet_group_list(run, fold_width=100)
    assert [type(block) for block in first] == [type(block) for block in second]
    first_bar = first[0]
    second_bar = second[0]
    assert isinstance(first_bar, QuietGroupBlock)
    assert isinstance(second_bar, QuietGroupBlock)
    assert first_bar.group == second_bar.group


def test_a_span_touching_an_existing_bar_is_refused() -> None:
    """The drain seam (review round 1, MINOR-1): a bar stands for rows still
    on screen (its hidden members), so a span touching one is the
    continuation of a folded group — folding it would put two bars over one
    logical run. The whole-span refusal, applied across a seam."""
    bar = fold_quiet_group_list([_peer("a"), _peer("b")], fold_width=100)[0]
    assert isinstance(bar, QuietGroupBlock)
    run = [_peer("c"), _peer("d")]
    assert fold_quiet_group_list(run, prev_block=bar) == run
    assert fold_quiet_group_list(run, next_block=bar) == run
    assert all(member.display for member in run)


def test_a_not_yet_visible_assistant_row_refuses_a_fold() -> None:
    """The belt at the plan layer: an assistant row with no visible text yet
    is a boundary, so [peer, pending-assistant, peer] never folds — and the
    same run with its prose landed splits the same way it always did."""
    pending = [_peer("one"), AssistantBlock(), _peer("two")]
    assert quiet_group_plan(pending) == []
    visible = AssistantBlock()
    visible.update_text("on it")
    visible.finalize_text()
    assert quiet_group_plan([_peer("one"), visible, _peer("two")]) == []


def test_a_picture_continues_only_its_own_owners_contiguous_chain() -> None:
    """R6-1 / Q-R8-1: the inside arm reads the row's OWN source coordinate —
    the same owner's NEXT part continues the chain, and nothing else does."""
    from local_operator.tui.widgets.image_block import ImageBlock
    from local_operator.tui.widgets.transcript import _source_owner_of

    def picture(anchor: str | None, part: int | None) -> ImageBlock:
        block = ImageBlock(None, "image/png")
        if anchor is not None:
            block.navigation_anchor_id = anchor
        if part is not None:
            block.navigation_anchor_part = part
        return block

    second = picture("h1", 2)
    assert quiet_group_record_of(second, prev_kind="inside", prev_owner=("h1", 1)).kind == "inside"
    # A different owner's picture, an unstamped picture, and a gap in the
    # chain all keep the conservative splitter reading.
    assert quiet_group_record_of(second, prev_kind="inside", prev_owner=("h2", 1)).kind == "image"
    assert quiet_group_record_of(second, prev_kind="inside", prev_owner=None).kind == "image"
    assert (
        quiet_group_record_of(picture("h1", 3), prev_kind="inside", prev_owner=("h1", 1)).kind
        == "image"
    )
    # The legacy arm is untouched: a picture directly after a tool or hub row.
    assert quiet_group_record_of(picture("t", 1), prev_kind="tool").kind == "inside"
    assert quiet_group_record_of(picture("h", 1), prev_kind="hub").kind == "inside"
    assert _source_owner_of(picture(None, None)) is None


def _hub(owner: str, part: int) -> HubMessageBlock:
    block = HubMessageBlock(
        "picture body",
        {"direction": "to_child", "body": "picture body", "steer": True},
    )
    block.navigation_anchor_id = owner
    block.navigation_anchor_part = part
    return block


def _owned_image(owner: str | None, part: int | None) -> ImageBlock:
    block = ImageBlock(None, "image/png")
    if owner is not None:
        block.navigation_anchor_id = owner
    if part is not None:
        block.navigation_anchor_part = part
    return block


def test_the_image_decision_table_has_no_or_loophole() -> None:
    """R8-1 / Q-R9-A1: the explicit grid — five predecessor families by five
    ownership shapes. Interior only for an allowed family (tool/hub/inside)
    with the SAME owner's NEXT part, or the unknown-provenance tool/hub legacy
    arm; a KNOWN contradiction splits even behind a hub or tool row, and a
    human or assistant predecessor ends the chain however the parts line up."""
    shapes = {
        "same-next": ("h1", 1, True),
        "same-gap": ("h1", 3, True),
        "other-owner": ("h2", 1, True),
        "image-unknown": (None, None, True),
        "prev-unknown": ("h1", 1, False),
    }
    allowed = {"hub", "tool", "inside"}
    for family in ("hub", "tool", "inside", "other", "assistant"):
        for shape, (owner, part, prev_known) in shapes.items():
            image = _owned_image(owner, part)
            prev_owner = ("h1", 0) if prev_known else None
            got = quiet_group_record_of(image, prev_kind=family, prev_owner=prev_owner).kind
            if owner is not None and part is not None and prev_known:
                want = "inside" if family in allowed and shape == "same-next" else "image"
            else:
                want = "inside" if family in ("tool", "hub") else "image"
            assert got == want, (family, shape, got, want)


def test_the_planner_refuses_a_fragment_whose_next_row_is_the_same_owners_next_part() -> None:
    """R8-2 / Q-R9-A2: the shared planner's bottom edge reads the last KEPT
    row's owner, so an off-page picture continuing the same owner's chain is
    not a boundary — every loaded cut refuses while it is on hand, and only a
    real boundary (or a next row that is NOT the owner's next part) closes."""
    h0, h1 = _hub("h0", 0), _hub("h1", 0)
    img1, img2 = _owned_image("h1", 1), _owned_image("h1", 2)

    assert quiet_group_plan([h0, h1], next_block=img1) == []
    assert quiet_group_plan([h0, h1, img1], next_block=img2) == []

    (planned,) = quiet_group_plan([h0, h1, img1, img2])
    assert planned.span == (0, 3)

    # A genuine boundary closes the partial run; so do a different owner and a
    # part gap, which are known contradictions and stay splitters.
    for closing in (
        UserBlock("audit the ingest path"),
        _owned_image("h9", 1),
        _owned_image("h1", 3),
    ):
        (closed,) = quiet_group_plan([h0, h1, img1], next_block=closing)
        assert closed.span == (0, 2)


def test_the_planner_top_edge_needs_the_predecessor_family_not_only_its_owner() -> None:
    """R8-2 / Q-R10-1: the TOP edge gates on the row ABOVE the edge row's
    FAMILY as well as its coordinate. Owner-only context reads every owned
    picture edge as a splitter and freezes a page-local fragment; the derived
    family refuses it. Human/answer families, a part gap and a different
    owner are known bounds and still close."""
    page = [_hub("h2", 0), _hub("h3", 0)]

    # A prefix of a hub receipt plus 1..3 of its pictures: the family above
    # the page's edge row is the hub itself for one picture and the chain's
    # own interior beyond — exactly what the source walk derives.
    for parts, family in ((1, "hub"), (2, "inside"), (3, "inside")):
        pictures = [_owned_image("h1", part) for part in range(1, parts + 1)]
        edge, above_owner = pictures[-1], ("h1", parts - 1)
        assert (
            quiet_group_plan(page, prev_block=edge, prev_kind=family, prev_owner=above_owner) == []
        ), (parts, family)
        # Without the family the same edge reads as a splitter: the bug twin.
        assert quiet_group_plan(page, prev_block=edge, prev_owner=above_owner) != []

    # Known bounds close: a human's or an answer's own picture (their family
    # ends the chain), a part gap and a different owner.
    for edge, family, above_owner in (
        (_owned_image("u0", 1), "other", ("u0", 0)),
        (_owned_image("a0", 1), "assistant", ("a0", 0)),
        (_owned_image("h1", 3), "inside", ("h1", 1)),
        (_owned_image("h9", 1), "inside", ("h1", 1)),
    ):
        (planned,) = quiet_group_plan(
            page, prev_block=edge, prev_kind=family, prev_owner=above_owner
        )
        assert planned.span == (0, 1)

    # An unknown prefix (head cut) keeps the minimum-marker reading: a plan
    # is made, but never claimed complete.
    (cut,) = quiet_group_plan(page, head_cut=True)
    assert cut.span == (0, 1) and cut.span_head_loaded is False
