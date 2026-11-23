"""Real styled consent surface: safe defaults and lossless compact review.

The transport is the integration tool's fixture, not an agent-server E2E. Queue
answers are synthetic keyboard interactions, never actual operator approval.
"""

from __future__ import annotations

import time

import pytest

from local_operator.harness.types import AskQuestion
from local_operator.team_bridge.integrations import AskCopy, ask_question
from local_operator.tools.integrations_tool import execute_integrations
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.ask_picker import AskPickerScreen
from local_operator.tui.widgets.ask_queue import ask_rows
from tests.unit.team_bridge.test_integrations_tool import APP, CHANNEL, Rig
from tests.unit.tui.test_app_pilot import FakeSession, _factory


class ConsentSession(FakeSession):
    def __init__(self, rig: Rig):
        super().__init__()
        self.rig = rig

    def ask_queue(self):
        return self.rig.queue

    def respond_ask(self, ask_id, answers, *, by="unknown"):
        result = self.rig.queue.respond(ask_id, answers, by="user")
        if result.get("ok"):
            self.rig.answered.add(ask_id)
        return result


@pytest.mark.asyncio
@pytest.mark.parametrize("approve", [False, True])
async def test_open_enter_is_safe_but_deliberate_approval_works(tmp_path, monkeypatch, approve):
    from local_operator.tui import ask_open_policy

    monkeypatch.setattr(ask_open_policy, "AUTO_OPEN", False)
    rig = Rig(tmp_path, monkeypatch)
    rig.now[0] = int(time.time() * 1000)
    draft = await execute_integrations(
        "synthetic-keyboard-draft",
        {
            "op": "draft_app",
            "name": "Alertmanager",
            "scopes": ["chat:write"],
            "channel_ids": [CHANNEL],
            "credential": "token",
            "secret_name": "SYNTHETIC_PILOT_TOKEN",
        },
        None,
        None,
        rig.context(),
    )
    queued = draft.details
    assert queued and queued["state"] == "queued"
    session = ConsentSession(rig)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        app._session = session
        await pilot.pause()
        app._sync_ask_surface(ask_rows(rig.queue.projection()))
        app._expand_asks(queued["ask_id"])
        await pilot.pause()
        picker = app.query_one(AskPickerScreen)
        assert picker.question.options[picker.selected_index].label == "Cancel"
        assert picker.question.recommended is None
        if approve:
            await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
        result = await execute_integrations(
            "synthetic-keyboard-answer",
            {"op": "publish_app", "app_id": APP},
            None,
            None,
            rig.context(),
        )
        assert result.details is not None
        assert result.details["state"] == ("done" if approve else "declined")
        if approve:
            assert result.details["credential"]["sealed"] is True
            assert rig.store.get("SYNTHETIC_PILOT_TOKEN") == rig.server.minted[0].encode()
        else:
            assert not any(body and body.get("ask_ref") for _, _, body in rig.server.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (60, 20)])
@pytest.mark.parametrize("locale", ["en", "ur"])
async def test_complete_frozen_question_reachable_by_keyboard(size, locale):
    from local_operator.i18n.text import active_locale, set_active_locale

    previous = active_locale()
    set_active_locale(locale)
    try:
        raw, _, _ = ask_question(
            AskCopy(
                op="publish_app",
                app_name="Production monitoring / Datadog + Alertmanager",
                org_name="Acme",
                channels=[
                    "alerts-prod-critical-primary-region",
                    "platform-payments-infrastructure-on-call",
                    "security-incident-response-production",
                    "customer-escalations-north-america",
                ],
                scopes=["alerts:ingest", "chat:write", "chat:write.thread"],
                secret_name="RADIENT_PRODUCTION_ALERTMANAGER_INTEGRATION_APP_TOKEN",
                **dict(credential="app_token"),
            )
        )
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            # Mount in the real app dock, which loads the actual stylesheet and
            # subtracts composer/status/transcript geometry from the card budget.
            picker = AskPickerScreen([AskQuestion.model_validate(raw)])
            from local_operator.tui.widgets.transcript import UserBlock

            app._append_block(UserBlock("Review the frozen integration grant."))
            app._mount_prompt(picker)
            picker.focus()
            await pilot.pause()
            await pilot.press("ctrl+e")
            await pilot.pause()
            seen = set()
            for _ in range(len(raw["question"]) + 1):
                layout = picker._layout()
                painted = "\n".join(picker.render_lines_for_test())
                assert all(line in painted for line in layout.question)
                seen.update(layout.question)
                old = tuple(layout.question)
                await pilot.press("pagedown")
                await pilot.pause()
                if tuple(picker._layout().question) == old:
                    break
            full = picker._question_lines(picker._layout().width, reveal=True)
            assert all(line in seen for line in full), (size, locale, full, seen)
            assert picker.region.bottom <= app.screen.size.height
            assert not app.screen.show_vertical_scrollbar
    finally:
        set_active_locale(previous)
