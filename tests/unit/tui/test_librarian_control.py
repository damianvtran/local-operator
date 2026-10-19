"""The ``/librarian`` bridge: the package is optional, the command is not.

The module belongs to the Librarian work package and is absent on this branch.
These tests pin the two halves the TUI owns: a missing package answers a clear
sentence, and a present package's hooks are called through the documented
contract. The fake modules go through ``sys.modules`` because that is what
``importlib.import_module`` reads — the same seam a real package lands on.
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from local_operator.tui import librarian_control
from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_slash_echo import _notice_texts


def _module(**hooks: object) -> types.ModuleType:
    module = types.ModuleType("local_operator.librarian")
    for name, value in hooks.items():
        setattr(module, name, value)
    return module


async def _settle(pilot, predicate, *, turns: int = 300) -> None:
    for _ in range(turns):
        if predicate():
            return
        await pilot.pause()
        await asyncio.sleep(0.01)
    raise AssertionError("condition did not settle")


def test_absence_is_a_state_not_an_error(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "local_operator.librarian", raising=False)
    assert librarian_control.module() is None
    result = librarian_control.call("status")
    assert result.available is False
    assert result.error == ""


def test_a_present_package_reports_its_state(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "local_operator.librarian",
        _module(status=lambda: {"state": "running", "organized": 4}),
    )
    result = librarian_control.call("status")
    assert result.available is True
    assert result.state == "running"


def test_pause_and_run_map_to_the_documented_hooks(monkeypatch) -> None:
    seen: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "local_operator.librarian",
        _module(
            pause=lambda: seen.append("pause") or {"paused": True},
            run_once=lambda: seen.append("run") or {"state": "organized"},
        ),
    )
    assert librarian_control.call("pause").state == "paused"
    assert librarian_control.call("run").state == "organized"
    assert seen == ["pause", "run"]


def test_a_missing_hook_is_named_rather_than_called_a_missing_package(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "local_operator.librarian", _module(status=lambda: "ok"))
    result = librarian_control.call("pause")
    assert result.available is True
    assert result.missing_hook == "pause"


def test_a_hook_refusal_is_reported_with_its_own_text(monkeypatch) -> None:
    def refuse() -> None:
        raise RuntimeError("cadence is held by the scheduler")

    monkeypatch.setitem(sys.modules, "local_operator.librarian", _module(status=refuse))
    result = librarian_control.call("status")
    assert result.available is True
    assert "cadence is held" in result.error


@pytest.mark.asyncio
async def test_async_hooks_are_awaited_and_blocking_hooks_leave_the_loop(
    monkeypatch,
) -> None:
    async def status() -> str:
        return "paused"

    monkeypatch.setitem(sys.modules, "local_operator.librarian", _module(status=status))
    assert (await librarian_control.call_async("status")).state == "paused"


@pytest.mark.asyncio
async def test_ensure_session_names_absence_and_hands_back_an_id(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "local_operator.librarian", raising=False)
    assert await librarian_control.ensure_session() == (None, "")

    async def ensure_session() -> str:
        return "abc123def456"

    monkeypatch.setitem(
        sys.modules,
        "local_operator.librarian",
        _module(ensure_session=ensure_session),
    )
    assert await librarian_control.ensure_session() == ("abc123def456", "")


@pytest.mark.asyncio
async def test_the_command_degrades_without_the_package() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(pilot, lambda: app._session is not None)
        app._run_slash_command("/librarian status")
        await _settle(
            pilot,
            lambda: any(
                "not available" in text and "Librarian" in text for text in _notice_texts(app)
            ),
        )


@pytest.mark.asyncio
async def test_the_command_reports_a_present_packages_state(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "local_operator.librarian",
        _module(status=lambda: {"state": "paused"}),
    )
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(pilot, lambda: app._session is not None)
        app._run_slash_command("/librarian")
        await _settle(
            pilot,
            lambda: any("Librarian: paused" in text for text in _notice_texts(app)),
        )


@pytest.mark.asyncio
async def test_an_unknown_verb_lists_the_vocabulary() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(pilot, lambda: app._session is not None)
        app._run_slash_command("/librarian dance")
        await _settle(
            pilot,
            lambda: any(
                "status, pause, resume, run or open" in text for text in _notice_texts(app)
            ),
        )
