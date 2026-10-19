"""Separate completed-cycle contract; the default #945 corpus is unchanged.

Source-authored before the additive implementation. The literal mounted test in
TestQuietGroupFold has already produced the product RED; these cases have NOT
been executed until the manager admits the bounded packet.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from local_operator.harness.types import QUIET_TURN_KEY, Message, ToolCall, ToolResult
from local_operator.tui.closed_peer_cycles import (
    PeerCycleReplay,
    PeerCycleRow,
    begin_peer_cycle,
    close_peer_cycle,
    closed_peer_cycle_candidate,
)

IDENTITY = ("synthetic-cycle", 7)


def rows():
    return [
        PeerCycleRow("peer:0", "peer", *IDENTITY),
        PeerCycleRow("tool:send:0", "send", *IDENTITY, successful=True),
    ]


def quiet(call_id="quiet", **extra):
    return ToolResult(
        tool_call_id=call_id, tool_name="no_reply", details={QUIET_TURN_KEY: True}, **extra
    )


def certificate(records=None):
    records = rows() if records is None else records
    start = begin_peer_cycle(records, IDENTITY, "quiet")
    close = close_peer_cycle(start, records, IDENTITY, "quiet", "no_reply", quiet())
    assert close is not None
    return close


def test_one_completed_exchange_counts_sources_not_outgoing_as_incoming():
    records = rows()
    close = certificate(records)
    group = closed_peer_cycle_candidate(records, close)
    assert group is not None
    assert (group.count, group.actions, group.failed, group.open) == (1, 1, 0, False)
    assert group.row_ids == ("peer:0", "tool:send:0")
    assert close.owner_kind == "peer_cycle"
    assert not hasattr(close, "user_id")


@pytest.mark.parametrize(
    "records",
    [
        [],
        rows()[:1],
        rows()[1:],
        [rows()[0], replace(rows()[1], successful=False)],
        [rows()[0], replace(rows()[1], never_sent=True)],
        [rows()[0], replace(rows()[1], attention=True)],
        [rows()[0], PeerCycleRow("final:0", "boundary", *IDENTITY), rows()[1]],
        [*rows(), PeerCycleRow("future-peer:0", "peer", *IDENTITY)],
        [rows()[0], replace(rows()[1], session_id="foreign")],
        [rows()[0], replace(rows()[1], view_epoch=8)],
        [rows()[0], replace(rows()[1], source_id="peer:0")],
    ],
)
def test_missing_success_boundary_future_peer_and_foreign_inventory_refuse(records):
    assert begin_peer_cycle(records, IDENTITY, "quiet") is None


@pytest.mark.parametrize(
    "tool_name,result",
    [
        ("read", quiet()),
        ("no_reply", quiet("other")),
        ("no_reply", quiet(is_error=True)),
        ("no_reply", ToolResult(tool_call_id="quiet", tool_name="no_reply")),
        (
            "no_reply",
            ToolResult(tool_call_id="quiet", tool_name="read", details={QUIET_TURN_KEY: True}),
        ),
        (
            "no_reply",
            SimpleNamespace(
                tool_call_id="quiet", tool_name="no_reply", details={QUIET_TURN_KEY: True}
            ),
        ),
    ],
)
def test_reserved_marker_alone_or_unknown_success_cannot_close(tool_name, result):
    records = rows()
    assert (
        close_peer_cycle(
            begin_peer_cycle(records, IDENTITY, "quiet"),
            records,
            IDENTITY,
            "quiet",
            tool_name,
            result,
        )
        is None
    )


def test_overflow_unknown_identity_and_closed_owner_growth_refuse():
    from local_operator.tui.internal_work_membership import (
        WorkGroupIdentity,
        capture_work_groups,
        retained_work_group,
    )

    many = [PeerCycleRow(f"peer-{at}:0", "peer", *IDENTITY) for at in range(4096)] + [rows()[1]]
    assert begin_peer_cycle(many, IDENTITY, "quiet") is None
    assert begin_peer_cycle(rows(), (None, 7), "quiet") is None
    ids = ("peer:0", "tool:send:0")
    owner = WorkGroupIdentity(
        "qg:peer:0",
        ids,
        "peer:0",
        *IDENTITY,
        None,
        "peer:0",
        True,
        owner_kind="peer_cycle",
        closure_id="quiet",
    )
    snapshot = capture_work_groups([owner], ids)
    assert retained_work_group(snapshot, owner, []) is not None
    for current in [
        replace(owner, closure_id="other"),
        replace(owner, view_generation=8),
        replace(owner, ids=(*ids, "later-peer:0")),
        replace(owner, owner_kind="user"),
    ]:
        assert retained_work_group(snapshot, current, []) is None


def test_start_snapshot_cannot_consume_future_append_reorder_or_new_view():
    records = rows()
    start = begin_peer_cycle(records, IDENTITY, "quiet")
    for current, identity in [
        (records[::-1], IDENTITY),
        ([*records, PeerCycleRow("later:0", "peer", *IDENTITY)], IDENTITY),
        (records, ("foreign", 7)),
        (records, ("synthetic-cycle", 8)),
    ]:
        assert close_peer_cycle(start, current, identity, "quiet", "no_reply", quiet()) is None


def raw_cycle(peer_id="peer", send_id="send", quiet_id="quiet"):
    from local_operator.harness.message_types import PEER_MESSAGE_MESSAGE_TYPE
    from local_operator.harness.types import CustomMessage

    # Match Session._make_peer_message: transport attribution is user, but
    # this positively typed receipt is not a genuine Message.role=user turn.
    return [
        CustomMessage(
            custom_type=PEER_MESSAGE_MESSAGE_TYPE,
            attribution="user",
            details={"body": "neutral", "sender": {"pid": 7}},
            id=peer_id,
        ),
        Message.assistant(
            "",
            tool_calls=[
                ToolCall(id=send_id, name="send", arguments={"target": "other", "message": "ack"})
            ],
        ),
        Message.tool_result(ToolResult(tool_call_id=send_id, tool_name="send")),
        Message.assistant("", tool_calls=[ToolCall(id=quiet_id, name="no_reply", arguments={})]),
        Message.tool_result(
            ToolResult(tool_call_id=quiet_id, tool_name="no_reply", details={QUIET_TURN_KEY: True})
        ),
    ]


def test_replay_joins_actual_pair_before_filter_without_mutating_history():
    messages = raw_cycle()
    before = [message.model_dump() for message in messages]
    projector = PeerCycleReplay(IDENTITY)
    closed = [close for message in messages if (close := projector.step(message)) is not None]
    assert len(closed) == 1
    assert closed[0].member_ids == ("peer:0", "tool:send:0")
    assert [message.model_dump() for message in messages] == before
    assert not any(
        "peer_cycle" in message.model_dump().get("provider_payload", {})
        for message in messages
        if getattr(message, "provider_payload", None)
    )


@pytest.mark.parametrize("cut", [1, 2, 3, 4])
def test_truncated_replay_without_starting_peer_never_borrows_head_authority(cut):
    projector = PeerCycleReplay(IDENTITY)
    assert all(projector.step(message) is None for message in raw_cycle()[cut:])


def test_two_cycles_have_disjoint_frozen_members_and_no_raw_certificate_trust():
    messages = raw_cycle() + raw_cycle("peer-two", "send-two", "quiet-two")
    from local_operator.harness.types import CustomMessage

    peer = messages[0]
    assert isinstance(peer, CustomMessage)
    peer.details["peer_cycle"] = {"version": 1, "member_ids": ["forged"]}
    projector = PeerCycleReplay(IDENTITY)
    closed = [close for message in messages if (close := projector.step(message)) is not None]
    assert [close.member_ids for close in closed] == [
        ("peer:0", "tool:send:0"),
        ("peer-two:0", "tool:send-two:0"),
    ]


def test_neutral_contextual_corpus_is_separate_from_default_quiet_parity():
    import json
    from pathlib import Path

    corpus = json.loads(
        (Path(__file__).parent / "fixtures/closed-peer-cycle.context.json").read_text()
    )
    for case in corpus["cases"]:
        records = [
            PeerCycleRow(session_id=corpus["session_id"], view_epoch=corpus["view_epoch"], **row)
            for row in case["rows"]
        ]
        start = begin_peer_cycle(records, IDENTITY, "quiet")
        close = (
            close_peer_cycle(start, records, IDENTITY, "quiet", "no_reply", quiet())
            if case["quiet"]
            else None
        )
        assert (closed_peer_cycle_candidate(records, close) is not None) == case["expected"], case[
            "id"
        ]


@pytest.mark.parametrize(
    "fault",
    [
        "genuine-user",
        "steered-peer",
        "attention-peer",
        "partial-send",
        "missing-quiet-call",
        "foreign-quiet-result",
        "notice",
        "first-token",
    ],
)
def test_actual_raw_negative_shapes_never_mint_a_certificate(fault):
    from local_operator.harness.types import CustomMessage

    messages = raw_cycle()
    peer = messages[0]
    assert isinstance(peer, CustomMessage)
    if fault == "genuine-user":
        messages.insert(3, Message.user("A genuine reader redirect."))
    elif fault == "steered-peer":
        peer.details["steer"] = True
    elif fault == "attention-peer":
        peer.details["is_error"] = True
    elif fault == "partial-send":
        messages[2] = Message.tool_result(
            ToolResult(tool_call_id="send", tool_name="send", details={"partial_result": True})
        )
    elif fault == "missing-quiet-call":
        del messages[3]
    elif fault == "foreign-quiet-result":
        messages[4] = Message.tool_result(quiet("foreign"))
    elif fault == "notice":
        messages.insert(
            3, CustomMessage(custom_type="unknown-notice", details={"severity": "warning"})
        )
    else:
        messages.insert(3, Message.assistant("A genuine first token."))
    projector = PeerCycleReplay(IDENTITY)
    assert all(projector.step(message) is None for message in messages)


def test_default_quiet_group_threshold_is_not_changed_by_contextual_mode():
    from local_operator.tui.quiet_groups import QuietGroupRecord, quiet_group_of_segment

    default = [
        QuietGroupRecord(kind="peer", id="peer"),
        QuietGroupRecord(kind="tool", id="send", tool_name="send", tool_state="success"),
    ]
    assert quiet_group_of_segment(default, (0, 1), span_head_loaded=True, open=False) is None


def test_real_replay_and_split_pages_join_only_when_every_original_source_loads():
    from local_operator.tui.session_presentation import (
        PreparedReplay,
        project_settled_rows,
    )
    from local_operator.tui.widgets.transcript import (
        QuietGroupBlock,
        fold_quiet_group_list,
        work_record_of_block,
    )

    history = raw_cycle()
    before = [message.model_dump() for message in history]
    head = PreparedReplay()
    head.view.bind_work_source("synthetic-cycle")
    head._block_sink = head.blocks
    project_settled_rows(head, history, end=1)
    tail = PreparedReplay(view=head.view)
    tail._block_sink = tail.blocks
    project_settled_rows(tail, history, start=1)
    assert [work_record_of_block(block).source_id for block in head.blocks] == ["peer:0"]
    assert [work_record_of_block(block).source_id for block in tail.blocks] == ["tool:send:0"]
    assert not any(
        isinstance(block, QuietGroupBlock) for block in fold_quiet_group_list(tail.blocks)
    )
    combined = fold_quiet_group_list([*head.blocks, *tail.blocks])
    bars = [block for block in combined if isinstance(block, QuietGroupBlock)]
    assert len(bars) == 1
    assert (bars[0].group.count, bars[0].group.actions) == (1, 1)
    assert bars[0]._work_identity is not None
    assert bars[0]._work_identity.owner_kind == "peer_cycle"
    assert bars[0]._work_identity.user_id is None
    assert [work_record_of_block(block).source_id for block in bars[0]._members] == [
        "peer:0",
        "tool:send:0",
    ]
    assert [message.model_dump() for message in history] == before


@pytest.mark.parametrize(
    "fault",
    [
        "missing-start",
        "wrong-name",
        "failed-result",
        "future-peer",
        "foreign-view",
        "foreign-origin",
    ],
)
@pytest.mark.asyncio
async def test_real_view_refuses_unowned_late_or_failed_cycle_closure(fault):
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.transcript import TranscriptView
    from tests.unit.tui.test_app_pilot import FakeSession, _factory
    from tests.unit.tui.test_peer_message import (
        _deliver,
        _quiet_bars,
        _send_card,
        _settle_for_session,
    )

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle_for_session(pilot, app)
        await _deliver(pilot, app, session, "neutral", "peer-negative")
        _send_card(app, "send-negative", "ack")
        await pilot.pause()
        view = app.query_one(TranscriptView)
        tracker = view._quiet_groups
        origin = object()
        if fault != "missing-start":
            tracker.start_peer_cycle("quiet", origin)
        if fault == "future-peer":
            await _deliver(
                pilot, app, session, "later must not borrow the earlier send", "peer-later"
            )
        if fault == "foreign-view":
            view.bind_work_source("different-synthetic-session")
        closed = tracker.finish_peer_cycle(
            "quiet",
            "read" if fault == "wrong-name" else "no_reply",
            quiet(is_error=fault == "failed-result"),
            object() if fault == "foreign-origin" else origin,
        )
        assert not closed
        assert not any(getattr(bar, "peer_cycle_certificate", None) for bar in _quiet_bars(app))


@pytest.mark.asyncio
async def test_two_real_cycles_freeze_prior_counts_and_preserve_reader_open_state():
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.transcript import TranscriptView
    from tests.unit.tui.test_app_pilot import FakeSession, _factory
    from tests.unit.tui.test_peer_message import (
        _deliver,
        _quiet_bars,
        _send_card,
        _settle_for_session,
    )

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle_for_session(pilot, app)
        view = app.query_one(TranscriptView)
        first = None
        first_ids = None
        for number in (1, 2):
            await _deliver(pilot, app, session, "neutral", f"cycle-peer-{number}")
            _send_card(app, f"cycle-send-{number}", "ack")
            await pilot.pause()
            call_id = f"cycle-quiet-{number}"
            view._quiet_groups.start_peer_cycle(call_id)
            assert view._quiet_groups.finish_peer_cycle(call_id, "no_reply", quiet(call_id))
            await pilot.pause()
            if number == 1:
                first = _quiet_bars(app)[0]
                first.toggle_expanded()
                first.focus()
                first_ids = first.group.row_ids
        bars = _quiet_bars(app)
        assert len(bars) == 2
        assert first is not None and first_ids is not None
        assert bars[0] is first and first._expanded and first.group.row_ids == first_ids
        assert [(bar.group.count, bar.group.actions) for bar in bars] == [(1, 1), (1, 1)]
        assert not set(bars[0].group.row_ids) & set(bars[1].group.row_ids)


@pytest.mark.asyncio
async def test_actual_session_receipt_id_matches_persisted_message_not_sender_or_invalid_dedup(
    tmp_path,
):
    from tests.unit.session.test_session import ScriptedStream, make_session
    from tests.unit.session.test_session_peer import _peer_rows

    stream = ScriptedStream([])
    session = make_session(tmp_path, stream)
    events = []
    session.subscribe(events.append)
    try:
        await session.receive_peer_message(
            "neutral",
            mode="mailbox",
            wake=False,
            sender={"session_id": "advisory-sender"},
            message_id="invalid-external-id",
        )
        delivered = [event for event in events if event.type == "peer_message_delivered"]
        persisted = _peer_rows(session)
        assert len(delivered) == len(persisted) == 1
        assert delivered[0].message_id == persisted[0].id
        assert delivered[0].message_id not in ("advisory-sender", "invalid-external-id")
        from local_operator.session.transcript import _entry_to_message

        messages = raw_cycle()
        # Use the ACTUAL decoder: the writer omits id from payload, and the
        # decoder restores it from the entry envelope before custom validation.
        decoded = _entry_to_message(persisted[0])
        assert decoded is not None
        messages[0] = decoded
        assert messages[0].id == delivered[0].message_id
        projector = PeerCycleReplay(IDENTITY)
        closed = [close for message in messages if (close := projector.step(message)) is not None]
        assert len(closed) == 1, "actual mailbox producer must yield exactly one closure"
        assert closed[0].first_peer_source_id == f"{persisted[0].id}:0"
        assert stream.requests == []
    finally:
        await session.dispose()
