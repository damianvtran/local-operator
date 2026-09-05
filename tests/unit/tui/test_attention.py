"""Real styled TUI geometry with a controlled host-probe boundary.

These are render-policy tests, not evidence that the OS focused a real window.
Native focus evidence is captured separately using the read-only host protocol.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import pytest
from textual.events import AppBlur, AppFocus
from textual.screen import Screen

from local_operator.harness.types import Message, TextContent
from local_operator.session.attention import AttentionStore
from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_app_pilot import FakeSession, _factory


class ReceiptSession(FakeSession):
    def __init__(self, path: Path, *, long: bool = False) -> None:
        super().__init__()
        self.store = AttentionStore(path)
        self.token = str(uuid.uuid4())
        text = "Finished result\n\n" + (
            "A long result line\n\n" * 90 if long else "The final answer is visible."
        )
        self.result = Message(role="assistant", content=[TextContent(text=text)])
        self._history = [self.result]
        self.store.publish("session/sess", self.token, self.result.id, "complete")

    async def refresh_attention(self) -> dict[str, Any]:
        return await asyncio.to_thread(self.store.state, "session/sess")

    async def acknowledge_attention(self, token: str) -> dict[str, Any]:
        return await asyncio.to_thread(self.store.acknowledge, "session/sess", token)


@pytest.mark.asyncio
async def test_default_focus_is_not_proof_and_rendered_focus_acknowledges(
    tmp_path, monkeypatch
) -> None:
    session = ReceiptSession(tmp_path / "attention.db")
    probes: list[bool] = []
    monkeypatch.setattr(
        "local_operator.tui.attention.terminal_is_foreground", lambda: probes.append(True) or True
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        # Boot history is mounted by a worker, then laid out on a later frame.
        # Wait on the actual geometry, not a fixed wall-clock delay.
        for _ in range(50):
            await pilot.pause()
            if app._completion_anchor_visible(session.result.id):
                break
        assert app._completion_anchor_visible(session.result.id)
        await app._poll_completion_attention()
        assert session.store.state("session/sess")["unseen"]
        assert not probes
        app.on_app_focus(AppFocus())
        await app._poll_completion_attention()
        assert not session.store.state("session/sess")["unseen"]
        count = len(probes)
        await app._poll_completion_attention()
        assert len(probes) == count


@pytest.mark.asyncio
async def test_overlay_scrollback_and_blur_do_not_acknowledge(tmp_path, monkeypatch) -> None:
    session = ReceiptSession(tmp_path / "attention.db", long=True)
    monkeypatch.setattr("local_operator.tui.attention.terminal_is_foreground", lambda: True)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(50):
            await pilot.pause()
            if app._completion_anchor_visible(session.result.id):
                break
        assert app._completion_anchor_visible(session.result.id)
        app.on_app_focus(AppFocus())
        app._transcript_view().scroll_home(animate=False)
        await pilot.pause()
        await app._poll_completion_attention()
        assert session.store.state("session/sess")["unseen"]
        app._transcript_view().scroll_end(animate=False)
        app.push_screen(Screen())
        await pilot.pause()
        await app._poll_completion_attention()
        assert session.store.state("session/sess")["unseen"]
        app.on_app_blur(AppBlur())
        app.pop_screen()
        await pilot.pause()
        await app._poll_completion_attention()
        assert session.store.state("session/sess")["unseen"]
