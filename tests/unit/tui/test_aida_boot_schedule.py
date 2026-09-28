"""The TUI boot ensure is SCHEDULED, never awaited (review round 2, NIT-3).

The server-side pin (`tests/e2e/test_desktop_api.py`) holds the lifespan's
ensure and asserts readiness; this is its terminal-side sibling. It needs no
pty because the scheduling is a named function (`_schedule_aida_boot_ensure`):
the property under test is that the CALL returns with the ensure still held —
which an awaited version cannot do — and that the task lands on the app so
nothing can collect it mid-flight.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from local_operator.tui import _schedule_aida_boot_ensure


@pytest.mark.asyncio
async def test_the_tui_boot_ensure_is_scheduled_and_the_call_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def held(*_args, **_kwargs):
        entered.set()
        await release.wait()

    monkeypatch.setattr("local_operator.aida.ensure_session", held)
    app = SimpleNamespace(_aida_boot_task=None)

    # The call must RETURN while the ensure is parked; an awaited version
    # would block here forever (or hand back a coroutine, failing the attr
    # assertions below).
    task = _schedule_aida_boot_ensure(app)

    assert app._aida_boot_task is task, "the task must be kept on the app"
    for _ in range(1000):
        if entered.is_set():
            break
        await asyncio.sleep(0)
    assert entered.is_set(), "the ensure must still run, just off the paint path"
    assert not task.done(), "the held ensure must be in flight, not awaited"

    release.set()
    await task
