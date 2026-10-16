"""The shared boot wait must observe adoption, not an empty worker list.

These small controls pin the harness itself. Real app coverage comes from the
pilot tests using it, optionally with LOP_TEST_SLOW_BOOT_WARM_S enabled.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from tests.unit.tui.conftest import await_session_boot


class _Workers(list[Any]):
    def __init__(self, app: Any, *, adopt: bool) -> None:
        super().__init__([SimpleNamespace(group="session"), SimpleNamespace(group="unrelated")])
        self.app = app
        self.adopt = adopt
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def wait_for_complete(self, workers: list[Any]) -> None:
        # Unrelated long-lived workers must not become prerequisites for boot.
        assert [worker.group for worker in workers] == ["session"]
        self.entered.set()
        await self.release.wait()
        if self.adopt:
            self.app._session = object()


@pytest.mark.asyncio
async def test_the_wait_cannot_return_until_the_worker_finishes_and_adopts() -> None:
    app: Any = SimpleNamespace(_session=None)
    app.workers = workers = _Workers(app, adopt=True)
    waiter = asyncio.create_task(await_session_boot(app))
    try:
        await asyncio.wait_for(workers.entered.wait(), 5)
        assert not waiter.done()
        assert app._session is None
        workers.release.set()
        await asyncio.wait_for(waiter, 5)
        assert app._session is not None
    finally:
        workers.release.set()
        if not waiter.done():
            waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)


@pytest.mark.asyncio
async def test_a_completed_worker_without_adoption_does_not_pass() -> None:
    app: Any = SimpleNamespace(_session=None)
    app.workers = workers = _Workers(app, adopt=False)
    workers.release.set()
    with pytest.raises(AssertionError, match="no session was adopted"):
        await await_session_boot(app)


@pytest.mark.asyncio
async def test_no_worker_is_legal_only_when_adoption_already_happened() -> None:
    app: Any = SimpleNamespace(_session=None, workers=[])
    with pytest.raises(AssertionError, match="no session was adopted"):
        await await_session_boot(app)
    app._session = object()
    await await_session_boot(app)
