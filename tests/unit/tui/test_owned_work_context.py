"""NEW contextual corpus and real replay/widget adapters, not quiet-parity changes.

Source-only authoring. These require the isolated, bounded pytest allocation;
construction-only checks are not mounted TUI or native UI evidence.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import AgentMessage, CustomMessage, Message, TextContent, ToolCall
from local_operator.tui.internal_work_membership import WorkRecord, owned_work_clauses, project_internal_work
from local_operator.tui.quiet_groups import QuietGroupRecord, quiet_groups_of
from local_operator.tui.session_presentation import PreparedReplay, project_settled_rows
from local_operator.tui.widgets.transcript import NoticeBlock, QuietGroupBlock, fold_quiet_group_list, project_work_blocks


CORPUS = json.loads((Path(__file__).parent / "fixtures/owned-work.context.json").read_text())


def _mounted_geometry(app, view) -> dict[str, object]:
    """Read primitive source-owned state; never format a widget on failure."""
    def source(block):
        return getattr(block, "tool_call_id", None) or getattr(block, "navigation_anchor_id", None) or type(block).__name__

    def box(block):
        return None if block is None else {
            "kind": type(block).__name__, "id": block.id,
            "display": block.display, "height": block.region.height,
            "relative_y": block.region.y - view.scrollable_content_region.y,
        }

    return {
        "work_identity": list(view.work_identity),
        "view_size": [view.size.width, view.size.height],
        "virtual_size": [view.virtual_size.width, view.virtual_size.height],
        "scroll_y": view.scroll_y, "max_scroll_y": view.max_scroll_y,
        "scroll_target_y": view.scroll_target_y,
        "is_following_tail": view.is_following_tail,
        "insert_anchor": None if view._insert_anchor is None else {
            "source": source(view._insert_anchor[0]), "gap": view._insert_anchor[1],
        },
        # This property means the CURRENT source still owes a painted page;
        # release_gate clears it only after the second restoration callback.
        "page_gate": app._resume_paging,
        "resume_fill_active": app._resume_fill_active,
        "welcome_visible": app._welcome_visible,
        "welcome": box(app._welcome), "head_notice": box(app._resume_head_notice),
        "children": [box(child) for child in view.children],
        "source_relative_rows": [[source(block), block.region.y - view.scrollable_content_region.y, block.display] for block in view.blocks()],
        "focused_source": source(app.focused) if app.focused is not None else None,
        "focused_relative_y": app.focused.region.y - view.scrollable_content_region.y if app.focused is not None else None,
        "conversation_started": view.conversation_started(),
        "rows": [[type(block).__name__, block.display, block.ends_empty_state] for block in view.blocks()],
    }


def _capture_mounted_frame(app, view, name: str) -> None:
    """Export the REAL styled app only when the bounded lane requests evidence.

    Exporting directly avoids save_screenshot's notification changing the next
    frame. Callback snapshots may precede paint: geometry is not pixel signoff.
    """
    directory = os.environ.get("OWNED_WORK_EVIDENCE_DIR")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{name}.svg").write_text(app.export_screenshot())
        facts = {"phase": name, **_mounted_geometry(app, view)}
        (path / f"{name}.json").write_text(json.dumps(facts, indent=2))
        print("MOUNTED_GEOMETRY", json.dumps(facts))


def _record(raw: dict[str, Any]) -> WorkRecord:
    details = raw.get("details", {})
    return WorkRecord(
        source_id=raw["id"], kind=raw["kind"], tool_name=raw.get("toolName", ""),
        custom_type=raw.get("customType", ""), final=raw.get("final", False),
        launched_job_id=raw.get("launchedJobId"), level=raw.get("level", "info"),
        producer="job_result" if raw.get("customType") == "job_result" else raw.get("producer", ""),
        job_id=details.get("job_id", raw.get("jobId")), status=details.get("status", raw.get("status")),
        direction=details.get("direction", ""), expects_reply=details.get("expects_reply"),
        steer=details.get("steer"), held=details.get("held", False),
    )


@pytest.mark.parametrize("is_bridge", [False, True])
def test_bridge_subclass_keeps_its_boundary_in_contextual_projection(is_bridge) -> None:
    from local_operator.tui.quiet_groups import group_splitter_of
    from local_operator.tui.widgets.transcript import (
        PeerMessageBlock, TeamBridgeBlock, peer_cycle_rows_of_blocks,
        quiet_group_record_of, work_record_of_block,
    )

    block = (TeamBridgeBlock if is_bridge else PeerMessageBlock)("Original typed receipt")
    block.navigation_anchor_id = "typed-receipt"
    block.navigation_anchor_part = 0
    block.work_view_identity = ("bridge-kind-test", 7)
    assert isinstance(block, PeerMessageBlock)  # The actual upstream inheritance collision.
    assert quiet_group_record_of(block).kind == ("bridge" if is_bridge else "peer")
    record = work_record_of_block(block)
    members = project_internal_work([
        WorkRecord("human-opener", "user"),
        WorkRecord("real-tool", "tool", tool_name="bash"),
        record,
    ])
    block.internal_work_membership = members[record.source_id]
    contextual = quiet_group_record_of(block)
    assert contextual.kind == ("bridge" if is_bridge else "peer")
    assert block.internal_work_membership.role == ("boundary" if is_bridge else "receipt")
    assert group_splitter_of(contextual) is is_bridge
    assert peer_cycle_rows_of_blocks([block])[0].kind == ("boundary" if is_bridge else "peer")


@pytest.mark.parametrize("is_bridge", [False, True])
def test_tracker_peer_role_admission_excludes_the_typed_bridge_subclass(monkeypatch, is_bridge) -> None:
    from local_operator.tui.widgets.transcript import PeerMessageBlock, QuietGroupTracker, TeamBridgeBlock, TranscriptView

    view = TranscriptView()
    view.bind_work_source("bridge-admission-test")
    identity = view.work_identity
    first = (TeamBridgeBlock if is_bridge else PeerMessageBlock)("Original first receipt")
    second = PeerMessageBlock("Original second receipt")
    for block, key in ((first, "first"), (second, "second")):
        block.navigation_anchor_id, block.navigation_anchor_part = key, 0
        block.work_view_identity = identity
    view._blocks = [first, second]
    tracker = QuietGroupTracker(view)
    admitted = []

    def renderer_boundary(certificate, members):
        # This test owns admission only; mounted grouping/focus has a separate
        # real-app gate and is not proved by replacing this rendering boundary.
        admitted.append((certificate.member_ids, members))
        return True

    monkeypatch.setattr(tracker, "_apply_peer_cycle", renderer_boundary)
    ids = ["first:0", "second:0"]
    record = {"version": 1, "session_id": identity[0], "quiet_call_id": "typed-quiet",
              "member_ids": ids, "members": [{"source_id": key, "role": "peer"} for key in ids]}
    assert tracker.apply_display_record(record) is (not is_bridge)
    assert len(admitted) == (0 if is_bridge else 1)
    if admitted:
        assert list(admitted[0][0]) == ids
        assert admitted[0][1] == [first, second]


def test_context_corpus_is_one_ordered_group_and_receipts_are_not_actions() -> None:
    records = [_record(raw) for raw in CORPUS["records"]]
    members = project_internal_work(records)
    assert [members[r.source_id].role for r in records] == CORPUS["roles"]
    quiet = [QuietGroupRecord(kind=r.kind, id=r.source_id, tool_name=r.tool_name, tool_state="success", work=members[r.source_id] if members[r.source_id].scope_id is not None else None) for r in records]
    groups = quiet_groups_of(quiet)
    assert len(groups) == 1
    assert groups[0].row_ids == tuple(r.source_id for r in records[1:9])
    assert groups[0].actions == 4
    assert groups[0].count == 2
    assert groups[0].key == "qg:tool:launch"
    assert list(owned_work_clauses(records[1:9], members) or ()) == CORPUS["counts"]
    assert members["job-result-a"].outcome == "unknown"


def _history(*, held: bool = False, failed: bool = False, with_user: bool = True):
    rows: list[AgentMessage] = [Message(role="user", id="user-replay", content=[TextContent(text="Investigate the sanitised deployment.")])] if with_user else []
    for at in range(3):
        rows.extend([
            Message(role="assistant", id=f"decl-{at}", tool_calls=[ToolCall(id=f"call-{at}", name="bash", arguments={"command": "printf safe"})]),
            Message(role="tool", id=f"result-{at}", tool_call_id=f"call-{at}", tool_name="bash", content=[TextContent(text="safe")]),
        ])
    details: dict[str, object] = {"job_id": "job-replay", "text": "Sanitised ORIGINAL child report.\nEvery original line remains available."}
    if held:
        details["held"] = True
    if failed:
        details["status"] = "failed"
    rows.append(CustomMessage(id="job-source", custom_type="job_result", attribution="user", details=details))
    return rows, details


def test_real_replay_preserves_delivered_payload_inside_owned_scope_without_inventing_an_answer() -> None:
    history, details = _history()
    replay = PreparedReplay()
    project_settled_rows(replay, history, fold_width=100)
    reports = [b for b in replay.blocks if getattr(b, "internal_work_source", None) is not None]
    assert len(reports) == 1
    assert reports[0].internal_work_payload == details
    assert isinstance(reports[0], NoticeBlock)
    assert reports[0]._text == details["text"]
    folded = fold_quiet_group_list(replay.blocks)
    bars = [b for b in folded if isinstance(b, QuietGroupBlock)]
    assert len(bars) == 1
    assert reports[0] in bars[0]._members
    assert bars[0].group.actions == 3
    assert "job result" in bars[0]._summary_text()


def test_missing_opener_keeps_delivered_policy_and_held_failure_attention_stays_visible() -> None:
    history, _ = _history(with_user=False)
    replay = PreparedReplay()
    project_settled_rows(replay, history, fold_width=100)
    dormant = [b for b in replay.blocks if getattr(b, "internal_work_source", None) is not None]
    assert len(dormant) == 1
    assert dormant[0].work_dormant and not dormant[0].display
    assert dormant[0].ends_empty_state is False
    for kwargs in ({"held": True}, {"failed": True}):
        history, _ = _history(with_user=False, **kwargs)
        replay = PreparedReplay()
        project_settled_rows(replay, history, fold_width=100)
        reports = [b for b in replay.blocks if getattr(b, "internal_work_source", None) is not None]
        assert len(reports) == 1
        assert reports[0].display
        assert not any(isinstance(b, QuietGroupBlock) and reports[0] in b._members for b in fold_quiet_group_list(replay.blocks))


def test_typed_failed_result_and_unknown_generic_notice_split_context_not_payload() -> None:
    history, details = _history(failed=True)
    replay = PreparedReplay()
    project_settled_rows(replay, history, fold_width=100)
    reports = [b for b in replay.blocks if getattr(b, "internal_work_source", None) is not None]
    assert len(reports) == 1
    unknown = NoticeBlock("Sanitised provenance-sensitive notice", kind="warning")
    replay.blocks.append(unknown)
    project_work_blocks(replay.blocks)
    assert reports[0].internal_work_membership is not None
    assert reports[0].internal_work_membership.attention == "required"
    assert reports[0].internal_work_payload == details
    assert unknown.internal_work_membership is not None
    assert unknown.internal_work_membership.role == "boundary"


def test_page_adoption_uses_current_closed_owner_and_refuses_stale_epoch_or_user() -> None:
    from local_operator.tui.widgets.tool_card import ToolCard
    from local_operator.tui.widgets.transcript import TranscriptView, UserBlock, quiet_group_plan

    view = TranscriptView()
    view.bind_work_source("synthetic-adoption")
    user = UserBlock("Loaded genuine user")
    user.navigation_anchor_id = "user"
    tools = [ToolCard(f"t{at}", "bash", {"command": "printf safe"}) for at in range(3)]
    for tool in tools:
        tool.mark_done("safe")
    for block in [user, *tools]:
        block.work_view_identity = view.work_identity
    project_work_blocks([user, *tools], view.work_identity)
    group = quiet_group_plan([user, *tools])[0][1]
    retired = QuietGroupBlock(group)
    retired.adopt_members(tools, expanded=True)
    replacement = QuietGroupBlock(group)
    replacement.adopt_members(tools, expanded=False)
    view._blocks = [user, replacement, *tools]
    tracker = view._quiet_groups
    tracker._open = retired
    tracker._pending_owned_page = (view.work_identity, False)
    tracker.adopt_reconciled_page()
    assert bool(tracker._open is replacement)
    assert not replacement.expanded, "a retired open bit never overrides the actual closed owner"
    # A genuinely foreign source remains foreign; positive fixture stamping
    # must not erase this independent rejection case.
    tools[-1].work_view_identity = ("foreign-session", view.work_identity[1])
    tracker._pending_owned_page = (view.work_identity, False)
    tracker.adopt_reconciled_page()
    assert tracker._open is None, "a foreign member cannot certify the replacement"
    tools[-1].work_view_identity = view.work_identity
    other = UserBlock("A different loaded user")
    other.navigation_anchor_id = "other-user"
    other.work_view_identity = view.work_identity
    view._blocks = [other, replacement, *tools]
    tracker._pending_owned_page = (view.work_identity, False)
    tracker.adopt_reconciled_page()
    assert tracker._open is None, "a different USER cannot adopt the old certificate"
    stale = view.work_identity
    view.bind_work_source("different-session")
    tracker._pending_owned_page = (stale, False)
    tracker.adopt_reconciled_page()
    assert tracker._open is None, "a stale session/epoch transaction is never replayed"


def test_tool_adapter_keeps_never_sent_faults_distinct_from_executed_failures() -> None:
    from local_operator.tui.widgets.tool_card import ToolCard
    from local_operator.tui.widgets.transcript import UserBlock, project_work_blocks, work_record_of_block

    for sent in (False, True):
        user = UserBlock("Loaded authority")
        user.navigation_anchor_id = "user"
        tool = ToolCard("tool", "bash", {"command": "printf safe"})
        if sent:
            tool.mark_failed("An executed failure")
        else:
            tool.mark_not_run("Unknown tool")
        assert work_record_of_block(tool).never_sent is not sent
        report = NoticeBlock("Original report")
        report.internal_work_source = WorkRecord("report", "custom", custom_type="job_result", producer="job_result", job_id="job")
        report.owned_delivery_only = True
        identity = ("synthetic-unsent", 1)
        for block in (user, tool, report):
            block.work_view_identity = identity
        project_work_blocks([user, tool, report], identity)
        assert report.work_dormant is not sent
        assert report.internal_work_membership is not None
        assert report.internal_work_membership.role == ("receipt" if sent else "boundary")


def test_bounded_core_upgrade_refuses_owner_epoch_topology_and_pinned_changes() -> None:
    from dataclasses import replace
    from local_operator.tui.internal_work_membership import WorkGroupIdentity, capture_work_groups, retained_work_group
    old = WorkGroupIdentity("qg:t1", ("t1", "receipt", "t2"), "t1", "synthetic", 1, None, "t1", True)
    prior = capture_work_groups([old], [*old.ids, "pinned-final"])
    current = WorkGroupIdentity("qg:new-tool", ("new-tool", *old.ids), "new-tool", "synthetic", 1, "loaded-user", "new-tool")
    assert retained_work_group(prior, current, ["loaded-user", "pinned-final"]) == 0
    for changed in (replace(current, session_id=None), replace(current, session_id="other"), replace(current, view_generation=2), replace(current, ids=("new-tool", "receipt", "t1", "t2")), replace(current, ids=("new-tool", "t1", "receipt")), replace(current, ids=(*current.ids, "pinned-final"))):
        assert retained_work_group(prior, changed, ["loaded-user", "pinned-final"]) is None
    constrained = capture_work_groups([replace(old, known_user_id="different-user")], [*old.ids, "pinned-final"])
    assert retained_work_group(constrained, current, ["loaded-user", "pinned-final"]) is None
    closed = capture_work_groups([replace(old, expanded=False)], [*old.ids, "pinned-final"])
    owner = retained_work_group(closed, current, ["loaded-user", "pinned-final"])
    assert owner == 0 and closed.groups[owner].expanded is False
    other = replace(old, key="qg:other", ids=("other", "other2"), anchor_id="other")
    merged = capture_work_groups([old, other], [*old.ids, *other.ids])
    assert retained_work_group(merged, replace(current, ids=(*current.ids, *other.ids)), []) is None
    assert not capture_work_groups([old], [str(at) for at in range(4097)]).complete
    assert not capture_work_groups([old, replace(other, key=old.key)], [*old.ids, *other.ids]).complete


def test_prepend_reclassifies_dormant_original_report_before_mount_without_duplicate_payload() -> None:
    from local_operator.tui.widgets.transcript import UserBlock
    history, details = _history(with_user=False)
    replay = PreparedReplay()
    replay.view.bind_work_source("synthetic-owned-page")
    project_settled_rows(replay, history, fold_width=100)
    reports = [b for b in replay.blocks if b.internal_work_source is not None]
    assert len(reports) == 1 and reports[0].work_dormant
    # Raw page rows and staged replay share the actual declared fixture session.
    view = replay.view
    # Staged source inventory, not a native/mounted execution claim. The app's
    # paging caller mounts this transaction only after this adapter returns.
    view._blocks = list(replay.blocks)
    user = UserBlock("The actual loaded user opener")
    user.navigation_anchor_id = "loaded-user"
    user.work_view_identity = view.work_identity
    page = view.prepare_owned_head_page([user], 0, head_cut=False)
    bars = [b for b in page if isinstance(b, QuietGroupBlock)]
    assert len(bars) == 1
    assert reports[0] in bars[0]._members and not reports[0].work_dormant
    assert reports[0].internal_work_payload == details
    assert isinstance(reports[0], NoticeBlock)
    assert reports[0]._text == details["text"]
    count = len(reports)
    project_settled_rows(replay, history, fold_width=100)
    assert len([b for b in replay.blocks if b.internal_work_source is not None]) == count


def test_source_generation_changes_only_for_real_clear_or_session_change(monkeypatch) -> None:
    from local_operator.tui.widgets.transcript import TranscriptView
    view = TranscriptView()
    # Only the scroll animator needs an app; keep the real clear/generation
    # path here. The mounted case below exercises that final scroll too.
    monkeypatch.setattr(view, "scroll_to", lambda *args, **kwargs: None)
    view.bind_work_source("synthetic-a")
    first = view.work_identity
    view.bind_work_source("synthetic-a")
    assert view.work_identity == first
    view.clear_blocks()
    assert view.work_identity == (first[0], first[1] + 1)
    view.bind_work_source("synthetic-b")
    assert view.work_identity == ("synthetic-b", first[1] + 2)


@pytest.mark.asyncio
async def test_mounted_dormant_report_keeps_empty_state_extent_and_source_once() -> None:
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.transcript import TranscriptView
    from tests.unit.tui.conftest import await_session_boot
    from tests.unit.tui.test_app_pilot import FakeSession, _factory
    session = FakeSession()
    session._history = []
    app = OperatorApp(lambda: _factory(session))
    row = CustomMessage(id="dormant-only", custom_type="job_result", attribution="user", details={"job_id":"synthetic-job","text":"Original dormant report."})
    async with app.run_test(size=(100,30)) as pilot:
        await await_session_boot(app)
        await pilot.pause()
        view = app.query_one(TranscriptView)
        before = (view.virtual_size.height, view.scroll_y)
        _capture_mounted_frame(app, view, "dormant-before")
        session._history = [row]
        app._project_settled_rows([row])
        await pilot.pause()
        _capture_mounted_frame(app, view, "dormant-after")
        reports = [b for b in view.blocks() if b.internal_work_source is not None]
        assert len(reports) == 1 and reports[0].work_dormant
        assert not reports[0].display and not reports[0].ends_empty_state
        assert not view.conversation_started()
        assert app._welcome_visible
        assert (view.virtual_size.height, view.scroll_y) == before
        app._project_settled_rows([row])
        assert len([b for b in view.blocks() if b.internal_work_source is not None]) == 1
        assert app._collect_resume_page_blocks([row]) == []
        assert session.prompts == [], "presentation never starts an extra model turn"
        generation = view.work_identity
        app._clear_transcript()
        await pilot.pause()
        assert view.work_identity == (generation[0], generation[1] + 1)
        assert not view.conversation_started()


@pytest.mark.asyncio
@pytest.mark.parametrize("overlap_page", [False, True], ids=["settled-page", "overlapping-page"])
async def test_mounted_open_legacy_qg_upgrades_on_loaded_opener_before_mount_and_keeps_focused_child(overlap_page, monkeypatch) -> None:
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.tool_card import ToolCard
    from local_operator.tui.widgets.transcript import PeerMessageBlock, TranscriptView
    from tests.unit.tui.conftest import await_session_boot
    from tests.unit.tui.test_app_pilot import FakeSession, _factory
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    scenario = "overlapping-page" if overlap_page else "settled-page"
    evidence_root = os.environ.get("OWNED_WORK_EVIDENCE_DIR")
    if evidence_root:
        monkeypatch.setenv("OWNED_WORK_EVIDENCE_DIR", str(Path(evidence_root) / scenario))
    async with app.run_test(size=(100,30)) as pilot:
        await await_session_boot(app)
        view = app.query_one(TranscriptView)
        # Loaded-tail content goes through the REAL app append edge. It both
        # retires Welcome and binds this existing fake session's synthetic ID;
        # a second manually invented binding would make page stamps foreign.
        tools = [ToolCard(f"scope-tool-{at}","bash",{"command":"printf safe"}) for at in range(3)]
        for at, tool in enumerate(tools):
            # Real output inside an EXISTING member provides scrollroom below
            # the focused row. No fake spacer/extra USER changes ownership.
            tool.mark_done("\n".join(f"Sanitised output line {line}" for line in range(48)) if at == 2 else "safe")
            app._append_block(tool)
        peer1 = PeerMessageBlock("First sanitised incoming note", {"conversation_name":"peer"})
        peer1.navigation_anchor_id = "peer-1"
        app._append_block(peer1)
        report = NoticeBlock("Original report remains available.")
        report.navigation_anchor_id = "report"
        report.internal_work_source = WorkRecord("report","custom",custom_type="job_result",producer="job_result",job_id="synthetic-job")
        report.internal_work_payload = {"job_id":"synthetic-job","text":"Original report remains available."}
        report.owned_delivery_only = True
        app._append_block(report)
        peer2 = PeerMessageBlock("Second sanitised incoming note", {"conversation_name":"peer"})
        peer2.navigation_anchor_id = "peer-2"
        app._append_block(peer2)
        await pilot.pause()
        old = next(b for b in view.blocks() if isinstance(b,QuietGroupBlock))
        old.toggle_expanded()
        await pilot.pause()
        assert not report.display and report.work_dormant
        tools[2].toggle_expanded()
        await pilot.pause()
        tools[1].focus()
        await pilot.pause()
        assert tools[1].has_focus
        # Exercise the real app's Ctrl-Home action BEFORE supplying the older
        # page. It releases tail-follow without moving focus (app contract).
        await pilot.press("ctrl+home")
        await pilot.wait_for_scheduled_animations()
        await pilot.pause()
        # Normal app content has no boot splash. Page ONLY if actual geometry
        # needs it; blindly paging an already-visible source scrolls past it.
        relative_y = tools[1].region.y - view.scrollable_content_region.y
        if relative_y >= view.scrollable_content_region.height:
            await pilot.press("pagedown")
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()
        _capture_mounted_frame(app, view, "upgrade-reader-preconditions")
        assert view.virtual_size.height > view.scrollable_content_region.height, "reader fixture must actually overflow"
        assert view.max_scroll_y > 0, "reader anchoring needs real scrollroom"
        assert 0 <= view.scroll_y < view.max_scroll_y, "the detached reader must remain before the tail"
        assert not app._welcome_visible and app._welcome is not None and not app._welcome.display
        assert app._welcome.region.height == 0, "a loaded conversation cannot retain boot-layout height"
        assert view.conversation_started()
        assert view.work_identity[0] == session.session_id, "app/page/live must share the actual synthetic session identity"
        assert view._insert_anchor is None, "no previous insertion may supply the anchor precondition"
        assert not view.is_following_tail, "an actual reader gesture must detach the tail"
        assert view.scroll_y == view.scroll_target_y, "the reader gesture must settle before paging"
        assert tools[1].has_focus
        assert 0 <= tools[1].region.y - view.scrollable_content_region.y < view.scrollable_content_region.height, "the pinned focused row must be on screen"
        _capture_mounted_frame(app, view, "upgrade-before")
        page_gap = tools[1].region.y - view.scrollable_content_region.y
        generation = view.work_identity
        # Observe the ACTUAL scheduled source callbacks without replacing their
        # behavior. In particular, one pause is not the restore/release fence.
        callback_count = 0
        original_schedule = view.call_after_refresh

        def observed_schedule(callback, *args, **kwargs):
            nonlocal callback_count
            name = getattr(callback, "__name__", "")
            is_welcome_refresh = name == "refresh" and getattr(callback, "__self__", None) is app._welcome
            if name not in {"settle_then_restore", "restore_anchor", "_settle_gap"} and not is_welcome_refresh:
                return original_schedule(callback, *args, **kwargs)
            callback_count += 1
            label = f"callback-{callback_count:02d}-{name}"

            def observed(*values, **named):
                _capture_mounted_frame(app, view, f"{label}-before")
                try:
                    return callback(*values, **named)
                finally:
                    _capture_mounted_frame(app, view, f"{label}-after")

            return original_schedule(observed, *args, **kwargs)

        monkeypatch.setattr(view, "call_after_refresh", observed_schedule)
        original_remeasure = view._remeasure_empty_state
        remeasure_count = 0

        def observed_remeasure():
            nonlocal remeasure_count
            remeasure_count += 1
            label = f"remeasure-{remeasure_count:02d}"
            _capture_mounted_frame(app, view, f"{label}-before")
            original_remeasure()
            _capture_mounted_frame(app, view, f"{label}-after")

        monkeypatch.setattr(view, "_remeasure_empty_state", observed_remeasure)
        app._resume_pending_head = [Message(role="user", id="loaded-user", content=[TextContent(text="The real loaded user opener")])]
        app._resume_mounted_ids.discard("loaded-user")
        # The real load-older path owns its paging lease, projects the raw USER,
        # captures/restores the viewport anchor and inserts the prepared page.
        # Calling prepare + insert ourselves would miss those interactions.
        app._mount_older_resume_page()
        new = next(b for b in view.blocks() if isinstance(b,QuietGroupBlock))
        _capture_mounted_frame(app, view, "upgrade-page-immediate")
        assert new.expanded and report.display and not report.work_dormant
        assert tools[1].has_focus, "no closed destination briefly hides the focused child"
        if overlap_page:
            # No await: this scenario DELIBERATELY admits the next source while
            # the real insertion owns its anchor and current-source paging gate.
            assert view._insert_anchor is not None and app._resume_paging
            _capture_mounted_frame(app, view, "overlap-before-append")
        else:
            # Semantic readiness, not an arbitrary sleep: both ownership fences
            # must clear and geometry must agree over two actual refreshes.
            deadline = time.monotonic() + 2
            previous = None
            stable = 0
            for refresh in range(10):
                await pilot.pause(0.05)
                _capture_mounted_frame(app, view, f"page-readiness-{refresh:02d}")
                geometry = _mounted_geometry(app, view)
                signature = json.dumps({key: geometry[key] for key in ("view_size", "virtual_size", "scroll_y", "max_scroll_y", "focused_relative_y", "source_relative_rows", "children")}, sort_keys=True)
                ready = view._insert_anchor is None and not app._resume_paging
                stable = stable + 1 if ready and signature == previous else (1 if ready else 0)
                previous = signature
                if stable >= 2:
                    break
                if time.monotonic() >= deadline:
                    break
            assert stable >= 2, "BLOCKED: paging readiness did not converge within 2s/10 refreshes"
            _capture_mounted_frame(app, view, "page-fully-settled")
        _capture_mounted_frame(app, view, "upgrade-after")
        assert tools[1].region.y - view.scrollable_content_region.y == page_gap, "real paging must preserve the focused row's viewport anchor"
        assert view.work_identity == generation, "paging is not a new view epoch"
        assert new.expanded and tools[1].has_focus
        assert sum(b is report for b in view.blocks()) == 1
        assert report.internal_work_payload["text"] == "Original report remains available."
        # OW-R2: stopping at insertion missed the retired live-tracker pointer.
        # The very next append must extend THIS replacement with the same owner,
        # not introduce a second tail group. No new USER authorises that append.
        from local_operator.tui.widgets.assistant import AssistantBlock
        from local_operator.tui.widgets.transcript import work_record_of_block
        next_tool = ToolCard("scope-tool-next", "bash", {"command": "printf safe"})
        app._append_block(next_tool)
        immediate_gap = tools[1].region.y - view.scrollable_content_region.y
        _capture_mounted_frame(app, view, "upgrade-next-live-immediate")
        assert immediate_gap == page_gap, "a live append must not move the detached reader even immediately"
        assert tools[1].has_focus and tools[1].display
        await pilot.pause()
        expected = [*tools, peer1, report, peer2, next_tool]
        bars = [b for b in view.blocks() if isinstance(b, QuietGroupBlock)]
        assert len(bars) == 1
        assert [work_record_of_block(b).source_id for b in bars[0]._members] == [work_record_of_block(b).source_id for b in expected]
        assert new.expanded and tools[1].has_focus
        memberships = {}
        for block in expected:
            membership = block.internal_work_membership
            assert membership is not None
            memberships[work_record_of_block(block).source_id] = membership
        assert owned_work_clauses(
            [work_record_of_block(block) for block in expected], memberships
        ) == ("4 shell actions", "2 incoming messages", "1 job result")
        # Capture BEFORE the assertion, so any real jump has its full source
        # state and styled frame rather than only a pytest region value.
        _capture_mounted_frame(app, view, "upgrade-next-live")
        assert tools[1].region.y - view.scrollable_content_region.y == immediate_gap
        next_tool.mark_done("safe")
        view.restate_quiet_member(next_tool)
        answer = AssistantBlock()
        answer.update_text("The genuine final remains outside the work group.")
        answer.mark_settled()
        app._append_block(answer)
        _capture_mounted_frame(app, view, "upgrade-settle-immediate")
        await pilot.pause()
        _capture_mounted_frame(app, view, "upgrade-settled")
        assert answer.display and answer not in new._members
        assert len([b for b in view.blocks() if isinstance(b, QuietGroupBlock)]) == 1
        assert new.expanded and tools[1].has_focus
        assert tools[1].region.y - view.scrollable_content_region.y == immediate_gap
        # Tail control / surgical mutants are separate, deferred allocations;
        # neither may hide a failed detached-reader continuation.
