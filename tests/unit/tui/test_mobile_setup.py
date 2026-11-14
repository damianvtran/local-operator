"""Exercise the actual slash worker/busy guard with synthetic setup outcomes."""

import threading
from unittest.mock import AsyncMock

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import NoticeKind
from local_operator.tunnels import setup
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.waiting import settle


@pytest.mark.asyncio
async def test_mobile_status_selected_owner_is_read_only(monkeypatch):
    observe = AsyncMock(
        return_value={
            "step": "apply",
            "verified": False,
            "receipts": [],
            "remedy": "Run lop tunnel setup apply",
        }
    )
    apply = AsyncMock()
    monkeypatch.setattr(setup, "observe", observe)
    monkeypatch.setattr(setup, "apply", apply)
    app = OperatorApp(lambda: _factory(FakeSession()))
    notices: list[str] = []

    def notice(body: str, kind: NoticeKind = "info") -> None:
        notices.append(body)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "_system_notice", notice)
        app._cmd_mobile("status --credential-id 17", notice)
        await pilot.pause(0.2)
        observe.assert_awaited_once_with(17)
        apply.assert_not_awaited()
        assert not app._tunnel_action_busy
        assert any("/mobile enable" in line for line in notices)


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", ["/mobile", "/mobile "])
async def test_mobile_composer_does_not_offer_provider_arguments(monkeypatch, typed):
    from local_operator.tui.widgets.editor import Editor

    observe = AsyncMock(
        return_value={
            "step": "login",
            "verified": False,
            "receipts": [],
            "credential_choices": [],
            "remedy": "Run lop login radient",
        }
    )
    monkeypatch.setattr(setup, "observe", observe)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        for char in typed:
            await pilot.press("slash" if char == "/" else "space" if char == " " else char)
        await pilot.pause()
        assert editor.picker.highlighted_name() not in {"radient", "openai", "anthropic"}
        await pilot.press("enter")
        await pilot.pause(0.3)
        observe.assert_awaited_once_with(None)
        assert not app._tunnel_action_busy
        assert session.prompts == []


@pytest.mark.asyncio
async def test_mobile_enable_progress_and_busy_guard(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    seen = []

    async def apply(selected=None, accepted=None, *, progress=None, resume_stopped=False):
        seen.append((selected, accepted))
        assert progress is not None
        progress("Synthetic reservation saved")
        started.set()
        # A hang guard only: the test releases this in its ``finally``. It is
        # long so a descheduled test loop cannot outwait it and clear the busy
        # flag before the guard assertion runs (the old 3 s was a clock bet).
        release.wait(30)
        return {
            "step": "failed",
            "verified": False,
            "receipts": ["Synthetic reservation saved"],
            "remedy": "Retry /mobile status",
        }

    monkeypatch.setattr(setup, "apply", apply)
    app = OperatorApp(lambda: _factory(FakeSession()))
    notices: list[str] = []

    def notice(body: str, kind: NoticeKind = "info") -> None:
        notices.append(body)

    # Event-driven, not timed: the fake apply runs on a worker thread behind
    # run_worker -> asyncio.to_thread -> asyncio.run, and a fixed
    # ``pilot.pause(0.2)`` lost that race on loaded CI shards (PR #3881, test
    # (3.12, 5), twice: ``started`` still unset). ``settle`` waits on the
    # thread's own ``started`` edge under a deadline that only catches a hang.
    # The first wait must NOT join the "mobile-setup" worker: it is parked on
    # ``release`` and asserting its intermediate busy state is the point.
    async with app.run_test(size=(100, 30)) as pilot:
        try:
            await pilot.pause()
            monkeypatch.setattr(app, "_system_notice", notice)
            app._cmd_mobile("enable 0.05 --credential-id 17", notice)
            await settle(pilot, started.is_set, what="fake mobile apply started")
            assert started.is_set() and app._tunnel_action_busy
            app._cmd_mobile("enable 0.05", notice)
            assert seen == [(17, "0.05")]
            assert any("already running" in line for line in notices)
            assert "Synthetic reservation saved" in notices
            release.set()
            await settle(
                pilot,
                lambda: not app._tunnel_action_busy,
                what="mobile setup worker finished",
                workers=("mobile-setup",),
            )
            assert not app._tunnel_action_busy
            assert any("Phone access could not finish setup" in line for line in notices)
            assert sum("Synthetic reservation saved" in line for line in notices) == 1
        finally:
            # Unpark the fake thread before app teardown, even when an
            # assertion above fails, so it can neither leak nor stall exit.
            release.set()
