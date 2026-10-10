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
