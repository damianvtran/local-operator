"""Loaded ID budgets, not an arbitrary 64-cycle eviction horizon.

SOURCE intent only until the focused allocation executes these tests. The first
case goes through actual PreparedReplay and the existing bar/ownership helpers:
a tiny registry-only test would miss certificates evicted before rows attach.
"""
from __future__ import annotations

from local_operator.tui.closed_peer_cycles import (
    MAX_PEER_CYCLE_ROWS, PeerCycleReplay, PeerCycleRow, begin_peer_cycle,
)
from local_operator.tui.internal_work_membership import capture_work_groups, retained_work_group
from local_operator.tui.session_presentation import PreparedReplay, project_settled_rows
from local_operator.tui.widgets.transcript import (
    QuietGroupBlock, fold_quiet_group_list, identity_of_work_group,
)
from tests.unit.tui.test_closed_peer_cycles import IDENTITY, raw_cycle


def _project(history, view=None):
    prepared = PreparedReplay() if view is None else PreparedReplay(view=view)
    prepared.view.bind_work_source("synthetic-cycle")
    prepared._block_sink = prepared.blocks
    project_settled_rows(prepared, history)
    bars = [block for block in fold_quiet_group_list(prepared.blocks)
            if isinstance(block, QuietGroupBlock)]
    return prepared, bars


def test_65_cycles_in_one_actual_loaded_replay_keep_first_closure_and_open_preference():
    history = [message for number in range(65)
               for message in raw_cycle(f"peer-{number}", f"send-{number}", f"quiet-{number}")]
    raw_before = [message.model_dump() for message in history]
    prepared, bars = _project(history)
    assert len(bars) == 65, "prepass must not evict the first loaded closure before its peer attaches"
    expected = [(f"peer-{number}:0", f"tool:send-{number}:0") for number in range(65)]
    assert [bar.group.row_ids for bar in bars] == expected
    assert [(bar.group.count, bar.group.actions, bar.group.open) for bar in bars] == [(1, 1, False)] * 65
    owners = [identity_of_work_group(bar.group, bar._members, expanded=number == 0)
              for number, bar in enumerate(bars)]
    assert all(owner is not None for owner in owners)
    known_owners = [owner for owner in owners if owner is not None]
    assert len(known_owners) == len(owners)
    ids = [source_id for pair in expected for source_id in pair]
    snapshot = capture_work_groups(known_owners, ids)
    _, rebuilt = _project(history, prepared.view)
    assert len(rebuilt) == 65
    current = identity_of_work_group(rebuilt[0].group, rebuilt[0]._members)
    assert current is not None
    retained = retained_work_group(snapshot, current, ids[2:], current_ids_unique=True)
    assert retained is not None
    assert snapshot.groups[retained].expanded
    assert [message.model_dump() for message in history] == raw_before


def test_one_peer_65_completed_communication_calls_fit_the_existing_4096_member_proof_budget():
    history = [raw_cycle()[0]]
    for number in range(65):
        history.extend(raw_cycle(send_id=f"send-{number}")[1:3])
    history.extend(raw_cycle()[3:])
    _, bars = _project(history)
    assert len(bars) == 1
    group = bars[0].group
    assert (group.count, group.actions, group.open) == (1, 65, False)
    assert group.row_ids == ("peer:0", *(f"tool:send-{number}:0" for number in range(65)))
    assert len(bars[0]._members) == 66


def test_member_budget_overflow_refuses_instead_of_certifying_an_arbitrary_late_suffix():
    assert MAX_PEER_CYCLE_ROWS == 4096
    records = [PeerCycleRow(f"peer-{number}:0", "peer", *IDENTITY) for number in range(4096)]
    records.append(PeerCycleRow("tool:send:0", "send", *IDENTITY, successful=True))
    assert begin_peer_cycle(records, IDENTITY, "quiet") is None
    replay = PeerCycleReplay(IDENTITY)
    history = [raw_cycle()[0]]
    for number in range(4096):
        history.extend(raw_cycle(send_id=f"send-{number}")[1:3])
    # Still the SAME over-budget interval: a later peer cannot rebase it into
    # a deceptively small candidate just because an earlier append hit the cap.
    history.extend(raw_cycle("late-peer", "late-send", "overflow-quiet"))
    assert not any(replay.step(message) is not None for message in history)
    fresh = [close for message in raw_cycle("fresh-peer", "fresh-send", "fresh-quiet")
             if (close := replay.step(message)) is not None]
    assert len(fresh) == 1
    assert fresh[0].member_ids == ("fresh-peer:0", "tool:fresh-send:0")
