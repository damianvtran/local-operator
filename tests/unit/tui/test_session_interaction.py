"""The terminal may leave a source without stealing or cancelling its work."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.events import CompactionEnded, TurnAbandoned
from tests.unit.tui.test_app_pilot import FakeSession, _factory


@pytest.fixture(autouse=True)
def isolate_sources(tmp_path, monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_TERMINAL_TITLE", "1")
    monkeypatch.setattr(OperatorApp, "_check_for_update", lambda self: None)


class HeldSession(FakeSession):
    @property
    def goal(self) -> str:
        return "finish the source work"

    def __init__(self, *, fail=False):
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.fail = fail
        self.recorded_shell = []

    async def prompt(self, text, images=None):
        self.prompts.append(text)
        self.entered.set()
        await self.release.wait()
        if self.fail:
            raise ConnectionError("owner socket unreachable")

    async def record_shell(self, command, result):
        self.recorded_shell.append((command, result))


@pytest.mark.asyncio
async def test_loop_continues_source_and_other_context_stop_does_not_cancel_it():
    first = HeldSession()
    second = FakeSession()
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        source = app._interaction
        task = asyncio.create_task(app._loop_worker(3, source))
        await asyncio.wait_for(first.entered.wait(), 10)
        app._adopt_session(second, replay_history=False)
        app._cmd_loop("stop", lambda body, kind="info": None)
        assert not source.loop.cancelled
        first.release.set()
        await asyncio.wait_for(task, 10)
        assert len(first.prompts) == 3
        assert not second.prompts
        assert not source.loop.running
        assert source.active_workers == 0


@pytest.mark.asyncio
async def test_failed_source_prompt_restores_only_its_own_input():
    first = HeldSession(fail=True)
    second = FakeSession()
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        source = app._interaction
        app._start_turn("source A message")
        await asyncio.wait_for(first.entered.wait(), 10)
        app._adopt_session(second, replay_history=False)
        app._editor().load_text("source B draft")
        first.release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert source.draft.text == "source A message"
        assert source.unsent == []
        assert app._editor().text == "source B draft"
        assert not second.prompts
        assert source.active_workers == 0


@pytest.mark.asyncio
async def test_hidden_compaction_delivers_held_input_exactly_once_to_source():
    first = HeldSession()
    first.release.set()
    second = FakeSession()
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        source = app._interaction
        source.compaction.active = True
        source.compaction.held_prompt = "accepted by A"
        source.compaction.held_typed = "accepted by A"
        app._adopt_session(second, replay_history=False)
        for _ in range(2):
            event = CompactionEnded("manual", True)
            event.origin = source.controller
            app.post_message(event)
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert first.prompts == ["accepted by A"]
        assert second.prompts == []
        assert not source.compaction.held_prompt
        assert not source.compaction.active


@pytest.mark.asyncio
async def test_late_source_fallback_cannot_retire_current_turn():
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        source = app._interaction
        source.turn.open = True
        app._adopt_session(FakeSession(), replay_history=False)
        app._turn_open = True
        event = TurnAbandoned(source.turn.epoch, aborted=False, error="source failed")
        event.origin = source.controller
        app.post_message(event)
        await pilot.pause()
        assert app._turn_open
        assert not source.turn.open


@pytest.mark.asyncio
async def test_source_usage_never_charges_visible_conversation():
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        source = app._interaction
        app._adopt_session(FakeSession(), replay_history=False)
        with patch("local_operator.tui.app.turn_cost", return_value=1.25):
            app._charge_aside_for(source, object())
        assert source.accounting.total == 1.25
        assert app._total_cost == 0.0


@pytest.mark.asyncio
async def test_bang_settles_and_records_on_hidden_source():
    first = HeldSession()
    second = FakeSession()
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        source = app._interaction
        app._run_shell_command("sleep 0.1; printf source-A")
        worker = source.shell.worker
        app._adopt_session(second, replay_history=False)
        app._shell_card = None
        await asyncio.wait_for(worker.wait(), 20)
        assert len(first.recorded_shell) == 1
        assert "source-A" in first.recorded_shell[0][1].text
        assert not second.prompts
        assert source.shell.worker is None
        assert source.active_workers == 0
        assert app._shell_card is None
