"""Owner-death recovery semantics for RemoteSession."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import local_operator.session.remote as remote_module
from local_operator.session.remote import RemoteSession


@pytest.mark.asyncio
async def test_owner_death_takes_over_silently_and_retains_submitted_input(
    tmp_path, monkeypatch
) -> None:
    """A prompt submitted during rotation reaches the lease-winning Session."""

    class LocalWinner:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def prompt(self, text, images=None):  # noqa: ANN001
            self.prompts.append(text)

        async def dispose(self) -> None:
            pass

    winner = LocalWinner()

    async def takeover():
        return winner

    monkeypatch.setattr(remote_module, "find_owner_record", lambda *args: (None, None))
    remote = RemoteSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=takeover,
    )
    adopted: list[Any] = []

    async def adopt(local):  # noqa: ANN001
        adopted.append(local)
        # Mirrors OperatorApp._adopt_takeover_session: disposal happens from
        # inside the recovery task and must not self-cancel that task.
        await remote.dispose()

    remote.set_takeover_callback(adopt)
    remote._owner_ready.set()
    remote._on_disconnected("owner exited")
    submitted = asyncio.create_task(remote.prompt("continue after death"))
    await asyncio.wait_for(submitted, timeout=2)
    assert adopted == [winner]
    assert winner.prompts == ["continue after death"]
