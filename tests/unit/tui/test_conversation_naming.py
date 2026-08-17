"""Auto-naming scheduling: when the title call fires, and what a failure costs.

The naming errand is decoration with one hard rule — it must never make the
session it decorates worse. Two regressions motivated this suite:

* Launched concurrently with the opening turn, it was a second simultaneous
  request on the same OAuth account at the account's busiest moment, and the
  concurrency ceilings made BOTH calls fail: the turn surfaced early
  provider-failure notices and the title died. The errand now waits for the
  turn to settle.
* A failed attempt used to spend the conversation's one naming request, so a
  minute-zero failure left the session (and the terminal title, which names
  the surface) on the cwd fallback forever. The latch now releases on
  failure and the next substantive message retries.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_app_pilot import FakeSession, _factory


class _GatedSession(FakeSession):
    """A session whose ``prompt`` blocks until the test releases it."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()
        self.title = ""

    async def prompt(self, text: str, images: Any = None) -> None:  # noqa: ANN401
        self.prompts.append(text)
        await self.gate.wait()

    async def complete_once(self, system: str, prompt: str) -> str:
        self.completions.append((system, prompt))
        return self.title


async def _settle() -> None:
    """Let app-thread workers run; two pauses is what the suite's timing needs."""
    for _ in range(4):
        await asyncio.sleep(0.02)


async def _boot(title: str = "") -> tuple[OperatorApp, _GatedSession]:
    session = _GatedSession()
    session.title = title
    app = OperatorApp(lambda: _factory(session))
    # Held open by the caller's `async with app.run_test(...)`; returning both
    # handles from inside the context keeps every test one flat block.
    return app, session


@pytest.mark.asyncio
async def test_naming_waits_for_the_live_turn_to_settle() -> None:
    """No second provider request while the opening turn still streams."""
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._start_turn("fix the login flow")
        app._maybe_name_conversation("fix the login flow")
        await _settle()
        assert session.completions == [], "naming raced the live turn"

        session.gate.set()  # the turn settles
        await _settle()
        assert len(session.completions) == 1, "naming did not fire once the turn settled"


@pytest.mark.asyncio
async def test_failed_attempt_releases_the_latch_for_a_retry() -> None:
    """A dead naming call costs a delayed title, not a nameless session."""
    app, session = await _boot(title="")  # complete_once answers nothing usable
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._maybe_name_conversation("please review the export columns")
        await _settle()
        assert session.completions, "attempt never ran"
        assert app._name_requested is False, "failure kept the once-only latch"

        session.title = "<title>Bulk export columns</title>"
        app._maybe_name_conversation("also add the assignee email column")
        await _settle()
        assert len(session.completions) == 2, "retry did not fire on the next message"
        assert session.conversation_name == "Bulk export columns"


@pytest.mark.asyncio
async def test_landed_title_paints_the_band_and_keeps_the_latch() -> None:
    """A successful attempt stores the name, paints the band, and stays spent."""
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._maybe_name_conversation("fix the login flow")
        await _settle()
        assert session.conversation_name == "Fix the login flow"
        assert app._name_requested is True
        assert app._status is not None
        assert app._status._conversation_name == "Fix the login flow"

        # A second substantive message must NOT rename what the user is reading.
        session.title = "<title>A different title</title>"
        app._maybe_name_conversation("now something else entirely")
        await _settle()
        assert len(session.completions) == 1
        assert session.conversation_name == "Fix the login flow"
