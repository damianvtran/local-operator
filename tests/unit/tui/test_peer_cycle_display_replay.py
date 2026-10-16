"""Cold wrapped exchanges use the existing list planner and disclosure rows."""

import copy

import pytest

from local_operator.harness.types import Message
from local_operator.tui.session_presentation import PreparedReplay
from local_operator.tui.widgets.transcript import (
    QuietGroupBlock,
    fold_quiet_group_list,
    work_record_of_block,
)
from tests.unit.session.test_peer_cycle_display_archive import make_cycle


def project(history, records):
    replay = PreparedReplay()
    replay.view.bind_work_source("sess")
    replay.prepare(history, bound=100, fold_width=90, display_records={"records": records})
    return fold_quiet_group_list(replay.blocks, fold_width=90)


@pytest.mark.parametrize(
    "wrapped_send,send_count,physical,actions",
    [(False, 1, 3, 1), (True, 1, 4, 1), (False, 0, 2, 0)],
)
def test_cold_overlay_has_exact_original_source_inventory_and_truthful_counts(
    wrapped_send, send_count, physical, actions
):
    _, history, record = make_cycle(wrapped_send=wrapped_send, send_count=send_count)
    final = Message.assistant("genuine public final")
    blocks = project([*history, final], [record])
    bars = [block for block in blocks if isinstance(block, QuietGroupBlock)]
    assert len(bars) == 1
    assert (bars[0].group.count, bars[0].group.actions) == (1, actions)
    assert bars[0].group.row_ids == tuple(record["member_ids"])
    members = [
        block for block in blocks if work_record_of_block(block).source_id in record["member_ids"]
    ]
    assert len(members) == physical
    assert [work_record_of_block(block).source_id for block in members] == record["member_ids"]
    finals = [block for block in blocks if block.navigation_anchor_id == final.id]
    assert len(finals) == 1 and finals[0].display
    assert work_record_of_block(finals[0]).source_id not in bars[0].group.row_ids
    if wrapped_send:
        from local_operator.tui.widgets.tool_card import ToolCard

        child = next(
            block for block in members if isinstance(block, ToolCard) and block.tool_name == "send"
        )
        assert child._args["message"] == "[redacted]"
        assert "SYNTHETIC_SECRET" not in str(child._args)


@pytest.mark.parametrize("mutation", ["hash", "cut", "foreign", "version"])
def test_invalid_cold_record_never_invents_missing_send_or_hides_final(mutation):
    _, history, record = make_cycle(wrapped_send=True)
    record = copy.deepcopy(record)
    if mutation == "hash":
        record["roots"][0]["outcome_digest"] = "wrong"
    elif mutation == "cut":
        history = history[:-1]
    elif mutation == "foreign":
        record["session_id"] = "foreign"
    else:
        record["version"] = 2
    final = Message.assistant("visible final")
    blocks = project([*history, final], [record])
    assert not any(getattr(block, "tool_name", None) == "send" for block in blocks)
    assert all(block.display for block in blocks if not isinstance(block, QuietGroupBlock))
    assert not any(isinstance(block, QuietGroupBlock) for block in blocks)


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped_send", [False, True])
async def test_mounted_cold_page_join_keeps_exact_payload_open_and_focused_source(
    tmp_path, monkeypatch, wrapped_send
):
    import json
    import os
    from pathlib import Path

    from local_operator.harness.peer_cycle_display import persist_record
    from local_operator.session.history_window import display_window
    from local_operator.session.transcript import Transcript
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.tool_card import ToolCard
    from local_operator.tui.widgets.transcript import TranscriptView
    from tests.unit.tui.conftest import await_session_boot
    from tests.unit.tui.test_app_pilot import FakeSession, _factory
    from tests.unit.tui.test_owned_work_context import _capture_mounted_frame

    _, history, record = make_cycle(wrapped_send=wrapped_send, root_output_lines=48)
    transcript = Transcript(tmp_path / "sessions" / "sess")
    await transcript.append_messages(history[:-1])
    assert await persist_record(transcript, record)
    await transcript.append_message(history[-1])
    genuine_final = Message.assistant("Original genuine public final", stop_reason="stop")
    await transcript.append_message(genuine_final)
    all_rows = transcript.build_llm_history()
    # A normal80-message snap intentionally includes this short episode's
    # peer. The REAL owner's byte-budget cut can exclude only that receipt
    # while keeping the complete parent/tool-result unit. Derive that actual
    # cut instead of inventing an unreachable normal-resume precondition.
    tail_cost = sum(
        len(json.dumps(row.model_dump(mode="json")).encode()) + 1 for row in all_rows[1:]
    )
    page = display_window(
        transcript,
        conversation_id="sess",
        owner_epoch="synthetic-owner",
        through_id=all_rows[-1].id,
        max_wire_bytes=8192 + tail_cost,
    )
    assert page.start == 1 and page.has_more
    assert [row.id for row in page.messages] == [row.id for row in all_rows[1:]]

    class PersistedSession(FakeSession):
        def __init__(self, source: Transcript) -> None:
            super().__init__()
            self._transcript = source

    session = PersistedSession(transcript)
    evidence = os.environ.get("OWNED_WORK_EVIDENCE_DIR")
    if evidence:
        monkeypatch.setenv(
            "OWNED_WORK_EVIDENCE_DIR",
            str(Path(evidence) / ("cold-eval" if wrapped_send else "cold-mixed")),
        )
    app = OperatorApp(lambda: _factory(session))
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await await_session_boot(app)
            # Existing synthetic owner, real persisted data and the production
            # cold renderer. No manually inserted widget or invented source ID.
            session._history = all_rows
            app._project_settled_rows(list(page.messages))
            view = app.query_one(TranscriptView)
            view.set_quiet_head_cut(page.has_more)
            await pilot.pause()
            await pilot.pause()
            view = app.query_one(TranscriptView)
            root = next(
                block
                for block in view.blocks()
                if isinstance(block, ToolCard) and block.tool_call_id == record["quiet_root"]
            )
            assert not app._resume_pending_head
            assert not any(
                work_record_of_block(block).source_id == record["member_ids"][0]
                for block in view.blocks()
            )
            for bar in [block for block in view.blocks() if isinstance(block, QuietGroupBlock)]:
                if not bar._expanded:
                    bar.toggle_expanded()
            if not root._expanded:
                root.toggle_expanded()
            await pilot.pause()
            root.focus()
            await pilot.pause()
            await pilot.press("ctrl+home")
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()
            _capture_mounted_frame(app, view, "cold-before-page")
            assert app._welcome is not None
            assert not app._welcome_visible and app._welcome.region.height == 0
            assert view.max_scroll_y > 0 and not view.is_following_tail
            assert (
                root.has_focus
                and 0
                <= root.region.y - view.scrollable_content_region.y
                < view.scrollable_content_region.height
            )
            assert view._insert_anchor is None and not app._resume_paging
            gap = root.region.y - view.scrollable_content_region.y
            identity = view.work_identity
            # The actual owner-selected older raw prefix is handed to the
            # existing page transaction only after reader setup, as a delivered
            # older-page response would be. No Home callback can fetch it early.
            app._resume_pending_head = list(all_rows[: page.start])
            app._mount_older_resume_page()
            _capture_mounted_frame(app, view, "cold-page-immediate")
            ready = False
            previous = None
            for frame in range(10):
                await pilot.pause()
                _capture_mounted_frame(app, view, f"cold-page-refresh-{frame}")
                geometry = (
                    root.region.y - view.scrollable_content_region.y,
                    view.scroll_y,
                    view.virtual_size.height,
                )
                if view._insert_anchor is None and not app._resume_paging and geometry == previous:
                    ready = True
                    break
                previous = geometry
            _capture_mounted_frame(app, view, "cold-page-settled")
            assert ready, "real paging callbacks must release before anchor assertions"
            bars = [block for block in view.blocks() if isinstance(block, QuietGroupBlock)]
            assert len(bars) == 1
            assert bars[0].group.row_ids == tuple(record["member_ids"])
            assert (bars[0].group.count, bars[0].group.actions) == (1, 1)
            assert bars[0]._expanded and root._expanded and root.has_focus
            assert view.work_identity == identity
            assert root.region.y - view.scrollable_content_region.y == gap
            ids = [work_record_of_block(block).source_id for block in view.blocks()]
            assert all(ids.count(key) == 1 for key in record["member_ids"])
            final = next(
                block for block in view.blocks() if block.navigation_anchor_id == genuine_final.id
            )
            assert (
                final.display and work_record_of_block(final).source_id not in bars[0].group.row_ids
            )
    finally:
        await session.dispose()
