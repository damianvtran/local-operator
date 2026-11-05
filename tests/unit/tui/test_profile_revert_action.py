"""A genuine OperatorApp/Pilot click, not a direct action-function call."""

import os
from pathlib import Path

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.tool_card import ToolCard
from scripts.visual_capture import save_capture
from tests.unit.session.test_account_profile_consent import SEED_INSTRUCTIONS
from tests.unit.session.test_account_profile_consent import (  # noqa: F401
    stores as consent_stores,
)
from tests.unit.session.test_profile_actions import saved
from tests.unit.tui.test_app_pilot import Editor, _await_session, _factory

stores = consent_stores
_EVIDENCE_ROOT = os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")


@pytest.mark.asyncio
@pytest.mark.parametrize("columns", [110, 60])
@pytest.mark.parametrize("activation", ["mouse", "keyboard"])
async def test_real_operator_revert_click_and_composer_negative_control(
    stores, columns, tmp_path, activation
):
    session, action, _ = await saved(stores)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(columns, 30)) as pilot:
        await _await_session(app, pilot)
        card = ToolCard("trusted-call", "agent", {"op": "update", "name": "reviewer"})
        card.mark_done("Updated agent 'reviewer'.", details={"profile_action": action})
        # A saved verification flag is NOT a live acknowledgment.
        assert card._profile_action_state == "unavailable"
        app._append_block(card)
        for _ in range(12):
            await pilot.pause()
            if card._profile_action_state == "available":
                break
        assert card._profile_action_state == "available"
        row = stores["account"].get_agent_by_name("reviewer")
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.press("r", "R")
        await pilot.pause()
        assert stores["account"].get_agent_system_prompt(row.id) == "Demand tests."
        assert card._profile_action_state == "available"
        assert editor.text == "rR"  # genuine composer input, not a no-op
        # Clear the composer so keyboard input cannot later become a prompt.
        editor.text = ""
        directory = Path(_EVIDENCE_ROOT or tmp_path) / "profile-action-frames"
        directory.mkdir(parents=True, exist_ok=True)
        save_capture(app, directory / f"before-{columns}-{activation}.svg")
        expanded = card.expanded
        # Revert is its own second-row hit target. The click must not expand
        # the parent tool row, and its state latch defeats same-batch repeats.
        if activation == "mouse":
            await pilot.click(card, offset=(7, 1))
        else:
            card.focus()
            await pilot.pause()
            assert card.has_focus
            await pilot.press("r", "R")
        await pilot.pause()
        assert card.expanded == expanded
        assert card._profile_action_state == "succeeded"
        assert stores["account"].get_agent_system_prompt(row.id) == SEED_INSTRUCTIONS
        save_capture(app, directory / f"after-{columns}-{activation}.svg")
        # A duplicate focused key cannot reexecute the old inverse after a
        # later legitimate change, even in the same batch of keyboard events.
        stores["account"].set_agent_system_prompt(row.id, "After revert, new edit.")
        card.focus()
        await pilot.press("r", "R")
        await pilot.pause()
        assert stores["account"].get_agent_system_prompt(row.id) == "After revert, new edit."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["no-live-proof", "dropped-response", "dropped-response-and-status"]
)
async def test_live_proof_and_transport_uncertainty_never_enable_a_blind_retry(
    stores, mode, monkeypatch
):
    from local_operator.profile_actions import ProfileActions

    session, action, _ = await saved(stores)
    live = session._profile_actions
    if mode == "no-live-proof":
        # Model-writable transcript/backup flags survive, host memory does not.
        session._profile_actions = ProfileActions(session._session_id, stores["account"].config_dir)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(110, 30)) as pilot:
        await _await_session(app, pilot)
        card = ToolCard("trusted-call", "agent", {"op": "update", "name": "reviewer"})
        card.mark_done("Saved.", details={"profile_action": action, "revert_verified": True})
        app._append_block(card)
        await pilot.pause()
        row = stores["account"].get_agent_by_name("reviewer")
        if mode == "no-live-proof":
            assert card._profile_action_state == "unavailable"
            await pilot.click(card, offset=(7, 1))
            card.focus()
            await pilot.press("r", "R")
            await pilot.pause()
            assert stores["account"].get_agent_system_prompt(row.id) == "Demand tests."
            assert not card.expanded
        else:
            assert card._profile_action_state == "available"

            class LostReply:
                calls = 0
                status_lost = mode == "dropped-response-and-status"

                async def profile_revert(self, request_id, *, approved=True):
                    self.calls += 1
                    live.activate(request_id, approved=approved, remember=False)
                    raise TimeoutError("synthetic lost reply AFTER actual apply")

                async def profile_action_status(self, **query):
                    if self.status_lost:
                        raise TimeoutError("synthetic lost status response")
                    return live.status(**query)

            client = LostReply()
            with monkeypatch.context() as scoped:
                scoped.setattr(session, "_client", client, raising=False)
                await pilot.click(card, offset=(7, 1))
                await pilot.pause()
                if client.status_lost:
                    assert card._profile_action_state == "pending"
                    assert "Check status" in card._content.plain
                    assert client.calls == 1
                    client.status_lost = False
                    await pilot.click(card, offset=(7, 1))  # read, never reapply
                    await pilot.pause()
                assert card._profile_action_state == "succeeded"
                assert stores["account"].get_agent_system_prompt(row.id) == SEED_INSTRUCTIONS
                assert client.calls == 1
