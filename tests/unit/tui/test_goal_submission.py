"""Goal submission shares ordinary prompt echo, steering and owner routing."""

from __future__ import annotations

import pytest

from local_operator.harness.types import ImageContent
from local_operator.session.frontend_state import (
    CommandScope,
    FrontendSessionState,
    SlashCapability,
    SlashResult,
)
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import CompactionEnded
from local_operator.tui.widgets.editor import Attachment, PastedText
from local_operator.tui.widgets.transcript import UserBlock
from tests.unit.tui.test_app_pilot import FakeSession, GoalSession, _factory


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True])
async def test_composer_goal_submits_or_steers_one_plain_user_message(busy):
    session = GoalSession()
    session.streaming = busy
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._editor().load_text("/goal ship the checklist")
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert session.goal == "ship the checklist"
        assert [row.text() for row in app.query(UserBlock)] == ["ship the checklist"]
        if busy:
            assert session.prompts == []
            assert [message.content[0].text for message in session.queued_steering()] == [
                "ship the checklist"
            ]
        else:
            assert session.prompts == ["ship the checklist"]


@pytest.mark.asyncio
@pytest.mark.parametrize("arg", ["", "clear", "none", "reset", "CLEAR"])
async def test_goal_metadata_forms_do_not_start_work(arg):
    session = GoalSession()
    session.set_goal("existing")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._run_slash_command(f"/goal {arg}".rstrip())
        await pilot.pause()
        assert session.goal == ("existing" if not arg else "")
        assert session.prompts == []
        assert not list(app.query(UserBlock))


@pytest.mark.asyncio
@pytest.mark.parametrize("consumers", [None, [], ["team_attached"], ["goal_set"]])
async def test_tui_owner_completes_only_unclaimed_goal_receipts(consumers):
    session = GoalSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        result = await app.run_slash_authoritative("goal", "ship it", None, consumers=consumers)
        await app.workers.wait_for_complete()
        assert session.goal == "ship it"
        assert result["data"]["type"] == "goal_set"
        assert session.prompts == ([] if consumers == ["goal_set"] else ["ship it"])


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_owner", [False, True])
async def test_current_viewer_consumes_goal_receipt_through_plain_submission(legacy_owner):
    session = GoalSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        result = app._goal_slash_result("ship it", SlashResult)
        assert session.prompts == []
        outcome = result.model_dump()
        if legacy_owner:
            # The pre-fix owner stores metadata only and returns precisely
            # this shape on success; no other legacy result proves admission.
            outcome["data"] = {"stored": "ship it"}
        app._render_authoritative_slash("/goal", "ship it", outcome)
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert session.prompts == ["ship it"]
        assert [row.text() for row in app.query(UserBlock)] == ["ship it"]


@pytest.mark.asyncio
async def test_goal_paste_is_held_during_compaction_and_submitted_once_afterwards():
    session = GoalSession()
    app = OperatorApp(lambda: _factory(session))
    marker = "[Paste #1, 40 lines]"
    payload = "\n".join(f"task {index}" for index in range(40))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._compacting = True
        app._run_slash_command(f"/goal {marker}", {1: PastedText(payload, marker)})
        await pilot.pause()
        assert session.goal == payload
        assert session.prompts == []
        assert app._prompt_held_for_compaction == payload
        assert app._typed_held_for_compaction == marker
        app._compacting = False
        app.on_compaction_ended(CompactionEnded(reason="manual", success=True))
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert session.prompts == [payload]
        assert [row.text() for row in app.query(UserBlock)] == [payload]


@pytest.mark.asyncio
async def test_routed_goal_stores_and_submits_paste_payload_not_private_chip():
    routed = []

    class RoutedSession(GoalSession):
        frontend_state: FrontendSessionState

        async def route_shared_slash(self, command, args, images=()):
            routed.append((command, args))
            self.set_goal(args)
            return {
                "kind": "notice",
                "text": "goal set",
                "style": "info",
                "data": {"type": "goal_set", "stored": args, "request": args},
            }

    session = RoutedSession()
    session.frontend_state = FrontendSessionState(
        session_id=session.session_id,
        epoch="goal-owner",
        slash_capabilities=[
            SlashCapability(command="goal", scope=CommandScope.AUTHORITATIVE_SESSION)
        ],
    )
    app = OperatorApp(lambda: _factory(session))
    marker = "[Paste #1, 40 lines]"
    payload = "\n".join(f"task {index}" for index in range(40))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._run_slash_command(f"/goal {marker}", {1: PastedText(payload, marker)})
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert routed == [("goal", payload)]
        assert session.goal == payload
        assert session.prompts == [payload]
        assert [row.text() for row in app.query(UserBlock)] == [payload]


@pytest.mark.asyncio
async def test_goal_submits_cited_image_and_keeps_pasted_lookalikes_out_of_resolution():
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    image = ImageContent(data="aGk=", mime_type="image/png")
    marker = "[Image #2, 1x1]"
    paste_marker = "[Paste #1, 40 lines]"
    # A quoted citation inside pasted prose must not reorder the real image.
    payload = "a previous log quoted [Image #3, 1x1]"
    attachments = {
        1: PastedText(payload, paste_marker),
        2: Attachment(image, marker),
        3: Attachment(ImageContent(data="b2xk", mime_type="image/png"), "[Image #3, 1x1]"),
    }
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._run_slash_command(f"/goal Check {paste_marker} against {marker}", attachments)
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert session.goal == f"Check {payload} against {marker}"
        assert session.prompts == [session.goal]
        assert session.prompt_images == [[image]]
