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


@pytest.mark.asyncio
async def test_takeover_swaps_subscription_and_updates_flow_from_the_winner(
    tmp_path, monkeypatch
) -> None:
    """M3/n4: after adoption the dead remote store is silent and the local
    winner's store is live, with no additive cost from the swap."""
    from local_operator.session.frontend_state import (
        FrontendSessionState,
        FrontendStateStore,
    )

    class LocalWinner:
        def __init__(self) -> None:
            self._store = FrontendStateStore(
                FrontendSessionState(
                    session_id="s1",
                    epoch="winner",
                    cumulative_parent_cost=12.34,
                )
            )

        @property
        def frontend_state(self):  # noqa: ANN202
            return self._store.state

        def subscribe_frontend(self, handler):  # noqa: ANN001, ANN202
            return self._store.subscribe(handler)

        async def dispose(self) -> None:
            pass

    winner = LocalWinner()

    async def takeover():
        return winner

    monkeypatch.setattr(remote_module, "find_owner_record", lambda *args: (None, None))
    remote = RemoteSession(config_dir=tmp_path, session_id="s1", takeover_factory=takeover)
    remote._install_frontend(
        FrontendSessionState(session_id="s1", epoch="dead-owner", cumulative_parent_cost=12.34)
    )

    adopted: list[Any] = []
    received: list[Any] = []

    async def adopt(local):  # noqa: ANN001
        # Mirrors _adopt_takeover_session: unsubscribe the dead remote store,
        # subscribe the winner, and apply its snapshot as replacement state.
        adopted.append(local)
        subscription = local.subscribe_frontend(received.append)
        # Checkpoint reconciliation is replacement, never addition (no 24.68).
        assert subscription.sync.snapshot.cumulative_parent_cost == 12.34
        await remote.dispose()

    remote.set_takeover_callback(adopt)
    remote._owner_ready.set()
    remote._on_disconnected("owner exited")
    for _ in range(200):
        if adopted:
            break
        await asyncio.sleep(0.01)
    assert adopted == [winner]

    # Updates published after adoption arrive from the WINNER's store.
    winner._store.mutate(cumulative_parent_cost=12.5)
    assert [u.changes.get("cumulative_parent_cost") for u in received] == [12.5]
