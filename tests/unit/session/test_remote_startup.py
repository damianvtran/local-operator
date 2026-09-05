"""Initial attachment is a canonical boundary, not merely a connected socket."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from local_operator.session.remote import RemoteSession
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer
from tests.unit.session.runtime.test_server import FakeHandle
from tests.unit.session.test_remote_cold import _never, _seed_transcript


@pytest.mark.asyncio
async def test_initial_sync_blocks_mutations_and_replays_new_epoch_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hold a real socket's sync while commands and newer owner state arrive.

    The explicit barriers expose the old connected-but-unsynchronized window
    deterministically. No speed threshold or sleep is used to create the race.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    _seed_transcript(tmp_path, "s1")
    handle = FakeHandle()
    server = RuntimeServer(handle, kind="tui")
    engagements = 0

    async def engage(*args, **kwargs):
        nonlocal engagements
        engagements += 1
        server.start()
        (tmp_path / "sessions/s1/.session.pid").write_text(str(os.getpid()))
        async with asyncio.timeout(30):
            while not any(status == "live" for _, status in registry.scan(tmp_path)):
                await asyncio.sleep(0.01)

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", engage)
    viewer = await RemoteSession.cold(
        "s1", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    reached = asyncio.Event()
    release = asyncio.Event()
    updated = asyncio.Event()
    original_sync = viewer._await_frontend
    original_update = viewer._on_frontend_update

    async def held_sync():
        sync = await original_sync()
        reached.set()
        await release.wait()
        return sync

    def observe_update(data):
        original_update(data)
        updated.set()

    monkeypatch.setattr(viewer, "_await_frontend", held_sync)
    monkeypatch.setattr(viewer, "_on_frontend_update", observe_update)
    tasks: list[asyncio.Task[None]] = []
    try:
        async with asyncio.timeout(30):
            binding = asyncio.create_task(viewer._ensure_bound())
            tasks.append(binding)
            await reached.wait()
            assert viewer._client is not None and viewer._client.connected
            assert viewer.is_cold, "a connected socket must not advertise canonical readiness"
            assert not viewer._owner_ready.is_set()

            entered = [asyncio.Event() for _ in range(3)]

            async def command(index):
                entered[index].set()
                if index == 0:
                    await viewer._ensure_bound()
                elif index == 1:
                    await viewer.prompt("first prompt")
                else:
                    await viewer.route_shared_slash("goal", "latest choice")

            waiting = [asyncio.create_task(command(index)) for index in range(3)]
            tasks.extend(waiting)
            await asyncio.gather(*(event.wait() for event in entered))
            assert all(not task.done() for task in waiting)
            assert handle.calls == [], "no mutation may outrun initial canonical installation"

            # The owner can change while history/sync is still loading. This
            # suffix belongs to fake-owner, not the synthesized cold epoch.
            handle._frontend.mutate(goal="new owner state")
            await updated.wait()
            assert viewer.frontend_state.epoch != handle.frontend_state_seed.epoch
            release.set()
            await asyncio.gather(*tasks)
            assert engagements == 1
            assert not viewer.is_cold
            assert viewer._owner_ready.is_set()
            assert viewer.frontend_state.epoch == handle.frontend_state_seed.epoch
            assert viewer.goal == "new owner state"
            assert [call[0] for call in handle.calls] == ["prompt", "run_slash_authoritative"]
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await viewer.dispose()
        server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["error", "cancel", "dispose"])
async def test_interrupted_initial_sync_closes_socket_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    _seed_transcript(tmp_path, "s1")
    server = RuntimeServer(FakeHandle(), kind="tui")

    async def engage(*args, **kwargs):
        server.start()
        (tmp_path / "sessions/s1/.session.pid").write_text(str(os.getpid()))
        async with asyncio.timeout(30):
            while not any(status == "live" for _, status in registry.scan(tmp_path)):
                await asyncio.sleep(0.01)

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", engage)
    viewer = await RemoteSession.cold(
        "s1", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    reached = asyncio.Event()
    release = asyncio.Event()
    original = viewer._load_history

    async def held_history(*args, **kwargs):
        reached.set()
        await release.wait()
        if outcome == "error":
            raise RuntimeError("history read failed")
        await original(*args, **kwargs)

    monkeypatch.setattr(viewer, "_load_history", held_history)
    binding = asyncio.create_task(viewer._ensure_bound())
    try:
        async with asyncio.timeout(30):
            await reached.wait()
            client = viewer._client
            assert client is not None and client.connected
            pump = client._reader_task
            if outcome == "cancel":
                binding.cancel()
            elif outcome == "dispose":
                await viewer.dispose()
            release.set()
            with pytest.raises((RuntimeError, asyncio.CancelledError, ConnectionError)):
                await binding
            assert not client.connected
            assert viewer._client is None
            assert viewer.is_cold
            # Let the cancelled old pump deliver its final callback before
            # retrying: it cannot turn a deliberate close into owner recovery.
            if pump is not None:
                await asyncio.gather(pump, return_exceptions=True)
            assert not viewer._recovering
            if outcome != "dispose":
                monkeypatch.setattr(viewer, "_load_history", original)
                await viewer._ensure_bound()
                assert not viewer.is_cold
                assert viewer._owner_ready.is_set()
    finally:
        release.set()
        if not binding.done():
            binding.cancel()
        await asyncio.gather(binding, return_exceptions=True)
        await viewer.dispose()
        server.close()
