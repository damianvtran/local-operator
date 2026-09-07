"""Source state survives navigation; a view change is never a user answer."""

import asyncio
import base64
import io
import os
from unittest.mock import patch

import pytest
from PIL import Image
from textual.widgets.text_area import Selection

from local_operator.harness.types import AskOption, AskQuestion, ImageContent
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_drafts import SessionDraftStore
from local_operator.tui.session_interaction import SessionDraft
from local_operator.tui.session_navigation import SessionNavigation
from local_operator.tui.session_presentation import DraftRecoveryNotice
from local_operator.tui.widgets.ask_picker import AskPickerScreen
from local_operator.tui.widgets.editor import Attachment, PastedText
from tests.unit.tui.test_app_pilot import FakeSession, _factory


@pytest.fixture(autouse=True)
def isolate_source_state(tmp_path, monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(OperatorApp, "_start_update_check", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_terminal_title", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_multiplexer_broadcast", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_herdr_reporter", lambda _self: None)


@pytest.mark.asyncio
async def test_navigation_stays_pending_until_committed_frame_is_usable():
    ready = asyncio.get_running_loop().create_future()
    pending = []

    async def prepare(session_id):
        return session_id

    async def release(_prepared):
        pass

    navigator = SessionNavigation(
        prepare=prepare,
        commit=lambda *_args: ready,
        release=release,
        pending=pending.append,
        failed=lambda *_args: pytest.fail("unexpected navigation failure"),
    )
    navigator.select("synthetic-b")
    await asyncio.sleep(0)
    assert navigator.requested_id == "synthetic-b"
    assert not navigator.committed_id
    ready.set_result(None)
    await asyncio.sleep(0)
    assert navigator.committed_id == "synthetic-b"
    assert pending[-1] == ""
    await navigator.close()


@pytest.mark.asyncio
async def test_temporary_allow_all_does_not_follow_another_session():
    first, second = FakeSession(), FakeSession()
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test() as pilot:
        await pilot.pause()
        app._set_approve_all(True)
        first_source = app._interaction
        app._adopt_session(second)
        assert not app._approve_all
        pending = asyncio.create_task(app.request_tool_approval("write", "synthetic target"))
        await pilot.pause()
        assert app._approval is not None
        assert not pending.done()
        app._approval.resolve(False)
        assert await pending is False
        app._adopt_session(first)
        assert app._interaction is first_source
        assert app._approve_all


@pytest.mark.asyncio
async def test_late_source_gate_cannot_mount_in_another_conversation():
    first, second = FakeSession(), FakeSession()
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test() as pilot:
        await pilot.pause()
        source = app._interaction
        app._adopt_session(second)
        with pytest.raises(asyncio.CancelledError):
            await app._request_user_choice_on_app_loop(
                [
                    AskQuestion(
                        id="source-a",
                        question="Only source A",
                        options=[AskOption(label="yes"), AskOption(label="no")],
                    )
                ],
                source=source,
            )
        assert app._ask_screen is None
        assert app._approval is None


@pytest.mark.asyncio
async def test_recovery_restores_exact_markers_without_overwriting_newer_input():
    first, second = FakeSession(), FakeSession()
    app = OperatorApp(lambda: _factory(first))
    image_bytes = io.BytesIO()
    Image.new("RGB", (2, 2)).save(image_bytes, format="PNG")
    image = ImageContent(
        data=base64.b64encode(image_bytes.getvalue()).decode(), mime_type="image/png"
    )
    raw = "[Image #1, 2x2]\n[Paste #2, 120 lines]\noriginal question"
    accepted = SessionDraft(
        text=raw,
        attachments={
            1: Attachment(image, "[Image #1, 2x2]"),
            2: PastedText("\n".join(["pasted source"] * 120), "[Paste #2, 120 lines]"),
        },
        selection=Selection((2, 0), (2, 8)),
    )
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        source = app._interaction
        source.draft.text = "newer source A draft"
        app._adopt_session(second)
        app._restore_unsent_for(source, "expanded question", [image], accepted=accepted)
        assert source.unsent == [accepted]
        app._adopt_session(first)
        app._load_editor_draft(source.draft)
        app._sync_draft_recoveries(source)
        await pilot.pause()
        await pilot.click(DraftRecoveryNotice)
        await pilot.pause()
        editor = app._editor()
        assert editor.text == raw
        assert editor.attachments() == accepted.attachments
        assert editor.selection == accepted.selection
        assert source.unsent[0].text == "newer source A draft"
        assert first.prompts == []


@pytest.mark.asyncio
async def test_draft_spill_retains_recoveries_notices_and_policy():
    store = SessionDraftStore()
    draft = SessionDraft(
        text="current",
        approve_all=False,
        recoveries=[SessionDraft(text="recover this exact draft", shell_mode=True)],
        notices=[("Source-specific failure", "warning")],
    )
    try:
        with patch("local_operator.tui.session_drafts.DRAFT_RESIDENT_BYTES", 0):
            await store.put("synthetic-a", draft)
        restored = await store.get("synthetic-a")
        assert restored.recoveries[0].text == "recover this exact draft"
        assert restored.recoveries[0].shell_mode
        assert restored.notices == draft.notices
        assert restored.approve_all is False
        path = store._path("synthetic-a")
        assert path.exists()
        await store.put("synthetic-a", SessionDraft())
        assert not path.exists()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_pending_choice_snapshot_restores_typed_and_checked_state():
    app = OperatorApp(lambda: _factory(FakeSession()))
    question = AskQuestion(
        id="synthetic-choice",
        question="Choose a synthetic result",
        options=[AskOption(label="first"), AskOption(label="second")],
        multi=True,
    )
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        card = AskPickerScreen([question], lambda _value: None)
        app._mount_prompt(card)
        await pilot.pause()
        card.action_move(1)
        card.action_toggle_row()
        card.action_jump_other()
        await pilot.press(*"unfinished")
        snapshot = card.snapshot_state()
        app._unmount_prompt(card)
        restored = AskPickerScreen([question], lambda _value: None)
        restored.restore_state(snapshot)
        app._mount_prompt(restored)
        await pilot.pause()
        assert restored.typed_text == "unfinished"
        assert restored.checked_indexes == [1]
        assert restored.selected_index == restored.other_row
        assert restored.has_focus == snapshot.focused
        app._unmount_prompt(restored)


@pytest.mark.asyncio
@pytest.mark.parametrize("gate_kind", ["approval", "ask"])
async def test_hidden_stopped_callback_cannot_answer_selected_sources_gate(gate_kind):
    class Watched(FakeSession):
        is_remote = True

        def set_stopped_callback(self, callback):
            self.stopped_callback = callback

    first, second = Watched(), FakeSession()
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test() as pilot:
        await pilot.pause()
        source_a = app._interaction
        source_a.turn.open = True
        stopped = first.stopped_callback
        app._adopt_session(second)
        source_b = app._interaction
        stopped_id = app._stopped_session_id
        if gate_kind == "approval":
            pending = asyncio.create_task(app.request_tool_approval("write", "only B"))
        else:
            pending = asyncio.create_task(
                app._request_user_choice_on_app_loop(
                    [
                        AskQuestion(
                            id="only-b",
                            question="Only B",
                            options=[AskOption(label="yes"), AskOption(label="no")],
                        )
                    ],
                    source=source_b,
                )
            )
        await pilot.pause()
        stopped()
        await pilot.pause()
        assert app._session is second
        assert app._stopped_session_id == stopped_id
        assert not pending.done()
        assert not source_a.turn.open and source_a.loop.cancelled
        assert not source_b.loop.cancelled
        if gate_kind == "approval":
            assert app._approval is not None
            app._approval.resolve(False)
        else:
            assert app._ask_screen is not None
            await pilot.press("escape")
        await pending


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["achieved", "cancelled", "error"])
async def test_goal_loop_releases_one_worker_lease_on_every_exit(outcome, monkeypatch):
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test() as pilot:
        await pilot.pause()
        source = app._interaction
        source.loop.running = True

        async def prompt(*_args, **_kwargs):
            assert source.active_workers == 1
            if outcome == "cancelled":
                source.loop.cancelled = True
                raise asyncio.CancelledError
            if outcome == "error":
                raise RuntimeError("synthetic source failure")

        async def judge(_session, _goal):
            return True, "achieved"

        monkeypatch.setattr(session, "prompt", prompt)
        monkeypatch.setattr(app, "_judge_goal", judge)
        if outcome == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await app._loop_goal_worker("goal", source)
        else:
            await app._loop_goal_worker("goal", source)
        await pilot.pause()
        assert source.active_workers == 0
        assert not source.loop.running
        assert not source.must_retain


@pytest.mark.asyncio
async def test_replay_invalidation_retries_without_clearing_pending_intent():
    from local_operator.tui.session_navigation import PreparationInvalidated

    prepared, released, pending, committed = [], [], [], []

    async def prepare(_session_id):
        prepared.append(len(prepared) + 1)
        return prepared[-1]

    async def release(value):
        released.append(value)

    def commit(_session_id, value, _generation):
        if value == 1:
            raise PreparationInvalidated("replay changed")
        committed.append(value)

    navigation = SessionNavigation(
        prepare=prepare,
        commit=commit,
        release=release,
        pending=pending.append,
        failed=lambda *_args: pytest.fail("must retry canonical replay"),
    )
    navigation.select("source-b")
    assert navigation._task is not None
    await navigation._task
    assert prepared == [1, 2] and released == [1] and committed == [2]
    assert pending == ["source-b", ""]
    await navigation.close()
