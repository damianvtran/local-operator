"""Exercise the actual slash worker/busy guard with synthetic setup outcomes."""

import threading
from unittest.mock import AsyncMock

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import NoticeKind
from local_operator.tunnels import setup
from tests.unit.tui.test_app_pilot import FakeSession, _factory


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
async def test_mobile_enable_progress_and_busy_guard(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    seen = []

    async def apply(selected=None, accepted=None, *, progress=None):
        seen.append((selected, accepted))
        assert progress is not None
        progress("Synthetic reservation saved")
        started.set()
        release.wait(3)
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

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "_system_notice", notice)
        app._cmd_mobile("enable 0.05 --credential-id 17", notice)
        await pilot.pause(0.2)
        assert started.is_set() and app._tunnel_action_busy
        app._cmd_mobile("enable 0.05", notice)
        assert seen == [(17, "0.05")]
        assert any("already running" in line for line in notices)
        assert "Synthetic reservation saved" in notices
        release.set()
        await pilot.pause(0.2)
        assert not app._tunnel_action_busy
        assert any(
            "Mobile setup: failed" in line and "Synthetic reservation saved" in line
            for line in notices
        )
