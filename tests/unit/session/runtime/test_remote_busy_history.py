"""Remote history must be captured by the owner even when its loop is busy.

The local off-loop fallback is deliberately windowless: a local viewer can read
its journal, but a remote viewer cannot substitute the origin's disk. These
cells block only a disposable daemon's session loop, never a fleet runtime.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tests.unit.session.runtime import test_daemon_serving_plane as helpers
from tests.unit.session.runtime.test_daemon_serving_plane import (
    _boot,
    _GuestWire,
    _recording_stream,
)
from tests.unit.session.test_session import wait_for

isolated_config = helpers.isolated_config


class _Wire(_GuestWire):
    def __init__(self, record: Any, *, locality: str = "remote") -> None:
        self.timeout, self.buf = 3.0, b""
        self.sock = socket.create_connection(("127.0.0.1", record.control_port), timeout=3)
        self.sock.sendall(
            json.dumps(
                {
                    "key": record.control_key,
                    "client": "attach",
                    "locality": locality,
                    "events": True,
                    "frontend_state": True,
                    "display_window": True,
                }
            ).encode()
            + b"\n"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("locality", ["remote", "local"])
async def test_busy_remote_owner_keeps_negotiated_history(
    isolated_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, locality: str
) -> None:
    session, _handle, runtime = await _boot(tmp_path, _recording_stream([]))
    loop = asyncio.get_running_loop()
    store = session._frontend_state_store
    baseline = len(store._subscribers)
    entered, release, grace_elapsed = threading.Event(), threading.Event(), threading.Event()
    original_grace = runtime._bind_with_grace

    async def observe_grace(task):
        subscription = await original_grace(task)
        if subscription is None:
            grace_elapsed.set()
        return subscription

    monkeypatch.setattr(runtime, "_bind_with_grace", observe_grace)

    def block_owner() -> None:
        entered.set()
        assert release.wait(3), "disposable owner release was lost"

    def attach() -> dict[str, Any]:
        loop.call_soon_threadsafe(block_owner)
        assert entered.wait(2)

        # Observe the real 100ms grace, not a shortened substitute. Remote
        # capture may resume after it; local capture must finish before release.
        def release_remote() -> None:
            if grace_elapsed.wait(2):
                release.set()

        releaser = threading.Thread(target=release_remote) if locality == "remote" else None
        if releaser is not None:
            releaser.start()
        wire = _Wire(runtime.record, locality=locality)
        try:
            sync = wire.read_until("frontend_sync")
            blocked_at_sync = not release.is_set()
            release.set()
            # One subscription must produce one exact-+1 update, not both the
            # abandoned on-loop attempt and a second fallback subscription.
            loop.call_soon_threadsafe(lambda: store.mutate(goal="synthetic-history-sequence"))
            update = wire.read_until("frontend_update")
            return {"sync": sync, "update": update, "owner_blocked_at_sync": blocked_at_sync}
        finally:
            release.set()
            if releaser is not None:
                releaser.join()
            wire.close()

    try:
        seen = await asyncio.to_thread(attach)
        frame = seen["sync"]
        assert frame.get("op") == "frontend_sync", frame.get("op")
        window = frame["data"].get("display_history")
        if locality == "remote":
            assert (
                window is not None
            ), "current remote owner lost negotiated history in off-loop fallback"
            assert window["status"] == "ok", window["status"]
            assert runtime.frontend_off_loop_binds == 0
        else:
            assert window is None
            assert runtime.frontend_off_loop_binds == 1
            assert seen["owner_blocked_at_sync"], "local attach waited out owner work"
        assert seen["update"]["data"]["sequence"] == frame["data"]["sequence"] + 1
        await wait_for(lambda: len(store._subscribers) == baseline and not runtime._clients)
        assert not runtime._abandoned_binds
    finally:
        release.set()
        runtime.close()
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["timeout", "disconnect"])
async def test_remote_history_wait_cleans_late_subscription(
    isolated_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    from local_operator.session.runtime import server as server_module

    monkeypatch.setattr(server_module, "_REMOTE_HISTORY_BIND_WAIT_S", 0.15)
    session, _handle, runtime = await _boot(tmp_path, _recording_stream([]))
    loop = asyncio.get_running_loop()
    store = session._frontend_state_store
    baseline = len(store._subscribers)
    entered, release, abandoned = threading.Event(), threading.Event(), threading.Event()
    original_release = runtime._release_when_landed
    original_grace = runtime._bind_with_grace
    grace_elapsed = threading.Event()

    async def observe_grace(task):
        subscription = await original_grace(task)
        if subscription is None:
            grace_elapsed.set()
        return subscription

    monkeypatch.setattr(runtime, "_bind_with_grace", observe_grace)

    def observe_abandon(task):
        original_release(task)
        abandoned.set()

    monkeypatch.setattr(runtime, "_release_when_landed", observe_abandon)

    def block_owner() -> None:
        entered.set()
        assert release.wait(3)

    def attach() -> str:
        loop.call_soon_threadsafe(block_owner)
        assert entered.wait(2)
        wire = _Wire(runtime.record)
        try:
            if ending == "timeout":
                frame = wire.read_until("bind_failed")
                assert frame.get("op") == "bind_failed", frame.get("op")
                assert wire.read() == {}, "failed bind must close its socket"
                return "bind_failed"
            wire.read()  # Welcome is independent of the blocked session loop.
            assert grace_elapsed.wait(2)
            return "disconnect"
        finally:
            wire.close()  # _drop_client cancels the shielded sync task.
            # The owner really remains blocked AFTER the connection goes away;
            # its late subscription must be released when it finally registers.
            try:
                assert abandoned.wait(2), "late-bind cleanup was not armed"
            finally:
                release.set()

    try:
        result = await asyncio.to_thread(attach)
        assert result == ("bind_failed" if ending == "timeout" else "disconnect")
        await wait_for(
            lambda: not runtime._abandoned_binds
            and not runtime._clients
            and len(store._subscribers) == baseline
        )
        assert runtime.frontend_off_loop_binds == 0
    finally:
        release.set()
        runtime.close()
        await session.dispose()


@pytest.mark.asyncio
async def test_missing_history_capability_keeps_remote_legacy_refusal(
    isolated_config: Path, tmp_path: Path
) -> None:
    from local_operator.network.projection import RemoteOwner, RemoteSessionFacts
    from local_operator.session.attached import AttachedSession
    from local_operator.session.frontend_state import FrontendSync

    session, _handle, runtime = await _boot(tmp_path, _recording_stream([]))
    runtime._record.capabilities.remove("display-history-window-v1")

    def attach() -> dict[str, Any]:
        wire = _Wire(runtime.record)
        try:
            return wire.read_until("frontend_sync")
        finally:
            wire.close()

    try:
        frame = await asyncio.to_thread(attach)
        assert frame["data"].get("display_history") is None
        facade = object.__new__(AttachedSession)
        facade.hydrate_deferred_history = AsyncMock()
        facade._load_history = AsyncMock()
        facade._display_window_requested = True
        facade._display_revision = 0
        facade._display_history = None
        facade._cold_painted_ids = None
        facade._owner = RemoteOwner(
            config_dir=tmp_path,
            session_id=runtime.record.session_id,
            facts=RemoteSessionFacts(
                session_id=runtime.record.session_id, device_id="synthetic-peer"
            ),
            root=tmp_path,
        )
        with pytest.raises(ConnectionError, match="too old to serve history"):
            await facade._load_frontend_history(FrontendSync.model_validate(frame["data"]))
        facade._load_history.assert_not_called()
    finally:
        runtime.close()
        await session.dispose()
