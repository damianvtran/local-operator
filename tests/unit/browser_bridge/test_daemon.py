from __future__ import annotations

import time
from contextlib import suppress
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from local_operator.browser_bridge import state as state_store
from local_operator.browser_bridge.daemon import create_app, pairing_status
from local_operator.browser_bridge.protocol import PROTO_VERSION

EXTENSION_ID = "a" * 32
ORIGIN = f"chrome-extension://{EXTENSION_ID}"


def test_rpc_auth_invalid_and_disconnected(tmp_path: Path) -> None:
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        key = app.state.bridge.state.session_key
        assert (
            client.post("/rpc", json={"id": "r-1", "method": "ping", "params": {}}).status_code
            == 401
        )
        invalid = client.post("/rpc", headers={"X-Bridge-Key": key}, json={"method": "ping"})
        assert invalid.status_code == 422
        disconnected = client.post(
            "/rpc",
            headers={"X-Bridge-Key": key},
            json={"id": "r-2", "method": "open", "params": {"url": "https://example.com"}},
        )
        assert disconnected.json()["error"]["code"] == "extension_disconnected"


def test_pairing_issues_token_then_authenticates(tmp_path: Path) -> None:
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/extension", headers={"origin": ORIGIN}) as socket:
            socket.send_json(
                {
                    "event": "hello",
                    "proto": PROTO_VERSION,
                    "token": "",
                    "extension_version": "0.1.0",
                    "browser": "Chrome/126",
                }
            )
            assert socket.receive_json() == {
                "event": "hello_ack",
                "proto": PROTO_VERSION,
                "paired": False,
            }
            code = pairing_status(tmp_path)["pending_code"]
            socket.send_json({"event": "pair", "code": code})
            paired = socket.receive_json()
            assert paired["ok"] is True
            assert len(paired["token"]) >= 32
        with client.websocket_connect("/extension", headers={"origin": ORIGIN}) as socket:
            socket.send_json(
                {
                    "event": "hello",
                    "proto": PROTO_VERSION,
                    "token": paired["token"],
                    "extension_version": "0.1.0",
                    "browser": "Chrome/126",
                }
            )
            assert socket.receive_json()["paired"] is True


def test_web_page_origin_is_rejected(tmp_path: Path) -> None:
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        try:
            with client.websocket_connect(
                "/extension", headers={"origin": "https://evil.example"}
            ) as socket:
                socket.receive_json()
        except Exception as exc:  # Starlette surfaces the explicit 4004 close.
            assert "4004" in str(exc) or getattr(exc, "code", None) == 4004
        else:  # pragma: no cover - rejection is mandatory
            raise AssertionError("web origin was accepted")


def _until(condition, timeout: float = 5.0) -> None:
    """The websocket receive loop runs on TestClient's portal thread, so a
    main-thread assertion right after send_json races the frame's processing.
    Poll the condition instead of reading state once."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    assert condition(), "condition did not settle in time"


def test_awaiting_origin_cleared_stops_the_health_echo(tmp_path: Path) -> None:
    """The extension's queue entry can expire, be cancelled, or be decided
    without a response the daemon will see. The extension then announces
    ``awaiting_origin_cleared``; the daemon must drop its record and republish
    so /health stops echoing a prompt the popup can no longer resolve — the
    stale echo that looped the approval popup on "Request changed." """
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        service = app.state.bridge
        with client.websocket_connect("/extension", headers={"origin": ORIGIN}) as socket:
            socket.send_json(
                {
                    "event": "hello",
                    "proto": PROTO_VERSION,
                    "token": "",
                    "extension_version": "0.1.0",
                    "browser": "Chrome/151",
                }
            )
            socket.receive_json()
            socket.send_json(
                {"event": "awaiting_origin", "id": "r-1", "origin": "https://docs.example"}
            )
            _until(lambda: service.link.awaiting_origin == {"r-1": "https://docs.example"})
            assert client.get("/health").json()["pending_origin"] == "https://docs.example"
            # The extension-side entry is gone; the daemon must forget it.
            socket.send_json({"event": "awaiting_origin_cleared", "id": "r-1"})
            _until(lambda: service.link.awaiting_origin == {})
            assert client.get("/health").json()["pending_origin"] == ""
            # A clearance for an id the daemon never recorded is a no-op.
            socket.send_json({"event": "awaiting_origin_cleared", "id": "r-unknown"})
            assert service.link.awaiting_origin == {}


def test_awaiting_origin_cleared_round_trips_through_the_wire_model() -> None:
    from local_operator.browser_bridge.protocol import AwaitingOriginCleared

    frame = AwaitingOriginCleared(id="r-1")
    assert AwaitingOriginCleared.model_validate_json(frame.model_dump_json()) == frame
    assert frame.event == "awaiting_origin_cleared"


def test_await_access_lock_topology() -> None:
    """Round-2 M3, structural half: await_access serializes on a PRIVATE
    per-request key (nothing can queue behind a human wait), request_access on
    the short shared __access__ key, tab commands per token, no-tab commands
    on __global__."""
    from local_operator.browser_bridge.daemon import BridgeService
    from local_operator.browser_bridge.protocol import Request

    key_of = BridgeService.lock_key_for
    a = key_of(Request(id="r-1", method="await_access", params={"url": "https://x.example"}))
    b = key_of(Request(id="r-2", method="await_access", params={"url": "https://x.example"}))
    assert a != b, "two awaits must never share a lock"
    assert a.startswith("__await__:")
    assert (
        key_of(Request(id="r-3", method="request_access", params={"url": "https://x.example"}))
        == "__access__"
    )
    assert key_of(Request(id="r-4", method="open", params={})) == "__global__"
    assert key_of(Request(id="r-5", method="goto", params={"tab": "bridge:1:n"})) == "bridge:1:n"


@pytest.mark.asyncio
async def test_await_access_does_not_block_request_access(tmp_path: Path) -> None:
    """Round-2 M3, behavioural half: with a fake extension whose await_access
    parks for 2 s, a concurrent request_access completes in milliseconds. With
    the old shared __access__ key this test fails: the request queues behind
    the full await slice."""
    import asyncio
    import time as time_module

    from local_operator.browser_bridge.daemon import BridgeService
    from local_operator.browser_bridge.protocol import Request, Response

    service = BridgeService(root=tmp_path)

    async def serve(payload: dict[str, object]) -> None:
        request = Request.model_validate(payload)

        async def respond() -> None:
            if request.method == "await_access":
                await asyncio.sleep(2.0)
                result = {"origin": "https://a.example", "state": "pending"}
            else:
                result = {"origin": "https://b.example", "state": "pending"}
            future = service.link.pending.get(request.id)
            if future and not future.done():
                future.set_result(Response(id=request.id, ok=True, result=result))

        asyncio.get_running_loop().create_task(respond())

    service.link.send = serve  # type: ignore[method-assign]
    started = time_module.monotonic()

    async def dispatch(method: str, request_id: str) -> float:
        request = Request(id=request_id, method=method, params={"url": "https://x.example"})
        await service._dispatch_serialized(request)
        return time_module.monotonic() - started

    waiter = asyncio.create_task(dispatch("await_access", "r-wait"))
    await asyncio.sleep(0.1)  # the waiter is now parked inside its own lock
    request_elapsed = await dispatch("request_access", "r-req")
    await_elapsed = await waiter
    # The request completed while the await was still parked: no queueing.
    assert request_elapsed < 1.0, f"request_access waited {request_elapsed:.2f}s behind await"
    assert await_elapsed >= 2.0


@pytest.mark.asyncio
async def test_cancelled_await_evicts_its_lock_key(tmp_path: Path) -> None:
    """Round-3 M1: cancelling a parked await_access (client disconnect
    surfaces as CancelledError inside the HTTP handler) must still evict its
    per-request lock key. Pre-fix, cancellation propagated past the eviction
    line and retained `__await__:<id>` forever — the reviewer's reproduction
    read the leaked key straight out of `_tab_locks`."""
    import asyncio

    from local_operator.browser_bridge.daemon import BridgeService
    from local_operator.browser_bridge.protocol import Request

    service = BridgeService(root=tmp_path)

    async def park_forever(payload: dict[str, object]) -> None:
        await asyncio.sleep(3600)

    service.link.send = park_forever  # type: ignore[method-assign]
    request = Request(id="r-cancel", method="await_access", params={"url": "https://x.example"})
    task = asyncio.get_running_loop().create_task(service._dispatch_serialized(request))
    await asyncio.sleep(0.1)  # parked inside the await wait
    assert "__await__:r-cancel" in service._tab_locks
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # The finally-path eviction covered the cancellation.
    assert service._tab_locks == {}


@pytest.mark.asyncio
async def test_await_lock_key_evicted_on_normal_and_error_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-3 M1: the two non-cancellation exits must evict too — normal
    completion and the daemon timeout on a never-answering extension."""
    import asyncio

    from local_operator.browser_bridge import daemon as daemon_mod
    from local_operator.browser_bridge.daemon import BridgeService
    from local_operator.browser_bridge.protocol import Request, Response

    # Monkeypatch COMMAND_TIMEOUTS for await_access to a fraction of a second so
    # the test verifies timeout eviction without stalling the suite for 30 seconds.
    monkeypatch.setitem(daemon_mod.COMMAND_TIMEOUTS, "await_access", 0.05)

    service = BridgeService(root=tmp_path)

    # Normal exit: the fake extension answers immediately.
    async def answer(payload: dict[str, object]) -> None:
        request = Request.model_validate(payload)
        future = service.link.pending.get(request.id)
        if future and not future.done():
            future.set_result(
                Response(id=request.id, ok=True, result={"origin": "x", "state": "pending"})
            )

    service.link.send = answer  # type: ignore[method-assign]
    await service._dispatch_serialized(
        Request(id="r-ok", method="await_access", params={"url": "https://x.example"})
    )
    assert service._tab_locks == {}

    # Error exit (timeout): the extension never answers; the daemon's command
    # timeout fires inside _dispatch_locked.
    async def silent(payload: dict[str, object]) -> None:
        await asyncio.sleep(daemon_mod.COMMAND_TIMEOUTS["await_access"] + 0.5)

    service.link.send = silent  # type: ignore[method-assign]
    await service._dispatch_serialized(
        Request(id="r-timeout", method="await_access", params={"url": "https://x.example"})
    )
    assert service._tab_locks == {}


@pytest.mark.asyncio
async def test_heartbeat_loop_survives_publish_failure(tmp_path: Path, monkeypatch) -> None:
    """RC1: a transient publish error must not kill the heartbeat writer.

    The incident: ENOSPC escaped ``while True: publish(); sleep()``, the task
    died, and nothing restarted it — so ``state.available()`` was False for
    every session on the machine for the daemon's whole life while ``/health``
    kept answering 200. The loop must absorb the error, keep ticking, and
    republish once writes succeed again.
    """
    import asyncio

    from local_operator.browser_bridge import daemon as daemon_module

    monkeypatch.setattr(state_store, "HEARTBEAT_INTERVAL_S", 0.01)
    service = daemon_module.BridgeService(root=tmp_path)
    service.link.websocket = object()  # type: ignore[assignment]

    calls = {"n": 0}
    real_publish = state_store.publish

    def flaky(state_value, root=None):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] <= 3:
            raise OSError(28, "No space left on device")
        return real_publish(state_value, root)

    monkeypatch.setattr(state_store, "publish", flaky)
    task = asyncio.create_task(service._heartbeat())
    try:
        # Long enough for the failures AND several successful ticks after.
        for _ in range(200):
            await asyncio.sleep(0.01)
            if calls["n"] > 6:
                break
        assert not task.done(), "heartbeat task died on a transient publish error"
        assert state_store.available(tmp_path), "heartbeat never recovered after writes resumed"
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_supervised_loop_keeps_running_after_repeated_failures(tmp_path: Path) -> None:
    """The class of bug, not the one syscall: no supervisory loop may exit."""
    import asyncio

    from local_operator.browser_bridge import daemon as daemon_module

    service = daemon_module.BridgeService(root=tmp_path)
    seen = {"n": 0}

    async def always_raises() -> None:
        seen["n"] += 1
        raise RuntimeError("boom")

    # Backoff is capped per failure; shrink it so the test is fast.
    task = asyncio.create_task(service._supervise("test", always_raises))
    try:
        for _ in range(300):
            await asyncio.sleep(0.01)
            if seen["n"] >= 2:
                break
        assert seen["n"] >= 2, "supervisor stopped re-running a failing iteration"
        assert not task.done()
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_supervised_loop_still_honours_cancellation(tmp_path: Path) -> None:
    """Shutdown depends on cancellation winning over the catch-all."""
    import asyncio

    from local_operator.browser_bridge import daemon as daemon_module

    service = daemon_module.BridgeService(root=tmp_path)

    async def forever() -> None:
        await asyncio.sleep(3600)

    task = asyncio.create_task(service._supervise("test", forever))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_publish_safely_swallows_oserror_and_reports_failure(tmp_path: Path, monkeypatch) -> None:
    """A failed cache write is logged and absorbed, never propagated."""
    from local_operator.browser_bridge import daemon as daemon_module

    service = daemon_module.BridgeService(root=tmp_path)

    def boom(state_value, root=None):  # type: ignore[no-untyped-def]
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(state_store, "publish", boom)
    assert service.publish_safely() is False
    monkeypatch.undo()
    assert service.publish_safely() is True


def test_driven_tabs_are_tracked_per_tab_and_cleared_on_close() -> None:
    """RC3/RC4: closing one tab clears exactly that tab, not the world."""
    from local_operator.browser_bridge import daemon as daemon_module

    link = daemon_module.ExtensionLink()
    assert link.current_url == ""

    link.note_driven("bridge:1:aa", "https://one.example", "One")
    link.note_driven("bridge:2:bb", "https://two.example", "Two")
    assert len(link.driven) == 2
    # Most recent wins the single-slot compatibility view.
    assert link.current_url == "https://two.example"

    link.note_closed("bridge:2:bb")
    assert len(link.driven) == 1
    assert link.current_url == "https://one.example"

    # The user's exact complaint: the last tab closing must clear "driving".
    link.note_closed("bridge:1:aa")
    assert link.driven == {}
    assert link.current_url == ""
    assert link.current_title == ""


def test_handleless_update_refreshes_rather_than_forking_a_phantom() -> None:
    """A `goto` result carries no handle; it must not double-count the tab."""
    from local_operator.browser_bridge import daemon as daemon_module

    link = daemon_module.ExtensionLink()
    link.note_driven("bridge:1:aa", "https://one.example", "One")
    link.note_driven("", "https://one.example/next", "Next")
    assert len(link.driven) == 1
    assert link.current_url == "https://one.example/next"


def test_tab_closed_event_clears_driven_state_over_the_socket(tmp_path: Path) -> None:
    """End to end over the real websocket: tab_closed empties `driving`."""
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/extension", headers={"origin": ORIGIN}) as socket:
            socket.send_json(
                {
                    "event": "hello",
                    "proto": PROTO_VERSION,
                    "token": "",
                    "extension_version": "0.1.6",
                    "browser": "Chrome/152",
                }
            )
            socket.receive_json()
            socket.send_json(
                {
                    "event": "tab_update",
                    "tab": "bridge:7:cc",
                    "url": "https://example.com",
                    "title": "Example",
                }
            )
            for _ in range(50):
                if client.get("/health").json()["current_url"]:
                    break
                time.sleep(0.02)
            health = client.get("/health").json()
            assert health["current_url"] == "https://example.com"
            assert len(health["driven_tabs"]) == 1

            socket.send_json({"event": "tab_closed", "tab": "bridge:7:cc"})
            for _ in range(50):
                if not client.get("/health").json()["current_url"]:
                    break
                time.sleep(0.02)
            cleared = client.get("/health").json()
            assert cleared["current_url"] == ""
            assert cleared["driven_tabs"] == []
