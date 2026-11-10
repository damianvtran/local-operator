"""Switch ordering ends at its reply, and cold metadata never mutates late.

The socket cases gate only post-reply maintenance, not the authoritative
mutation: a healthy ping alone must not hide a blocked second model switch.
The executor cases separately distinguish queue starvation from a slow build.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest

from local_operator.harness.types import ModelSpec
from local_operator.model import configure
from local_operator.session.runtime import serving
from local_operator.session.runtime.server import RuntimeServer
from tests.unit.session.runtime.test_server import (
    FakeHandle,
    _dial,
    _until,
    _wait_record,
)
from tests.unit.session.runtime.test_serving import make_handle


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["slash_result", "set_model"])
async def test_next_switch_does_not_wait_for_previous_postreply_refresh(op: str) -> None:
    class Handle(FakeHandle):
        def __init__(self) -> None:
            super().__init__()
            self.armed = False
            self.entered = threading.Event()
            self.release = threading.Event()
            self.refreshes = 0

        async def run_slash_authoritative(self, command: str, args: str, images: Any) -> Any:
            result = await super().run_slash_authoritative(command, args, images)
            self.armed = True
            return result

        async def set_model(self, provider: str, model_id: str) -> str:
            result = await super().set_model(provider, model_id)
            self.armed = True
            return result

        async def refresh(self):
            if self.armed:
                self.refreshes += 1
                self.entered.set()
                await asyncio.to_thread(self.release.wait, 5)
            return await super().refresh()

    handle = Handle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial(record, client="daemon")
        for req, model in [(1, "A"), (2, "B")]:
            frame = {"op": op, "req": req}
            if op == "slash_result":
                frame.update(command="model", args=f"test/{model}", images=[])
            else:
                frame.update(provider="test", model_id=model)
            writer.write(json.dumps(frame).encode() + b"\n")
            await writer.drain()
            await asyncio.wait_for(
                _until(reader, "result" if op == "slash_result" else "ack", req), 2
            )
            if req == 1:
                assert await asyncio.to_thread(handle.entered.wait, 2)
        assert (
            not handle.release.is_set()
        ), "the second switch must reply before refresh is released"
        calls = [
            call
            for call in handle.calls
            if call[0] == ("run_slash_authoritative" if op == "slash_result" else "set_model")
        ]
        assert len(calls) == 2
        assert handle.refreshes == 1, "maintenance must coalesce, not overlap roster scans"
        task = runtime._switch_refresh_task
        assert task is not None and not task.done()
        # Teardown below must cancel/join the owned maintenance task even while
        # its off-loop wait is still gated; releasing the thread prevents leaks.
    finally:
        handle.release.set()
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        runtime.close()
    assert runtime._switch_refresh_task is None


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [False, True], ids=["building", "queued"])
async def test_cold_resolution_deadline_cannot_mutate_after_expiry(
    monkeypatch: pytest.MonkeyPatch, queued: bool
) -> None:
    handle, session = make_handle()
    entered = threading.Event()
    release = threading.Event()
    applied = []
    built = []
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(serving, "_switch_executor", lambda: pool)
    monkeypatch.setattr(serving, "_MODEL_SWITCH_RESOLVE_S", 0.05, raising=False)
    monkeypatch.setattr(configure, "warm_model_info", lambda *_: None)
    monkeypatch.setattr(session, "set_model", lambda *a, **kw: applied.append(a), raising=False)

    def build(provider: str, model_id: str) -> ModelSpec:
        built.append(model_id)
        entered.set()
        assert release.wait(5)
        return ModelSpec(provider=provider, model_id=model_id)

    monkeypatch.setattr(configure, "build_model_spec", build)
    if queued:
        pool.submit(lambda: (entered.set(), release.wait(5)))
        assert await asyncio.to_thread(entered.wait, 2)
    try:
        with pytest.raises(TimeoutError):
            await handle.set_model("openai", "synthetic-cold")
        assert applied == []
        if queued:
            assert built == [], "an expired queued request must never start its build"
        else:
            assert entered.is_set(), "the control must actually expire inside the build"
        release.set()
        # A sentinel behind the work proves the late result has actually landed,
        # rather than merely sampling the unchanged model before it returns.
        await asyncio.wrap_future(pool.submit(lambda: None))
        await asyncio.sleep(0)
        assert applied == [], "a late worker result must not mutate selection"
    finally:
        release.set()
        pool.shutdown(wait=True)


@pytest.mark.asyncio
async def test_a_projection_built_before_a_switch_is_not_published_after_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = RuntimeServer(FakeHandle(), kind="tui")
    entered = threading.Event()
    release = threading.Event()
    sent = []
    scheduled = []
    monkeypatch.setattr(runtime, "_republish_identity", lambda: None)
    monkeypatch.setattr(runtime, "_projection_recipients", lambda: [object()])
    monkeypatch.setattr(runtime, "_push_soon", lambda: scheduled.append(True))
    monkeypatch.setattr(runtime, "_projection_frame", lambda _, frame: frame)

    def old_payload():
        entered.set()
        assert release.wait(5)
        return {"model_label": "test/old"}

    async def send(_, frame):
        sent.append(frame)

    monkeypatch.setattr(runtime, "_projection_payload", old_payload)
    monkeypatch.setattr(runtime, "_send_to", send)
    task = asyncio.create_task(runtime._push())
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        runtime._switch_revision += 1
        release.set()
        await task
        assert sent == [], "the old worker frame must not roll selection back"
        assert scheduled == [True], "dropping stale bytes still owes a fresh repaint"
        monkeypatch.setattr(runtime, "_projection_payload", lambda: {"model_label": "test/new"})
        await runtime._push()
        assert sent == [{"model_label": "test/new"}]
    finally:
        release.set()
        await task


@pytest.mark.asyncio
async def test_cold_loopback_listing_cannot_commit_after_resolution_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator.model.discovery import fetch_models

    body = json.dumps({"data": [{"id": "synthetic-cold", "display_name": "Synthetic"}]}).encode()
    reached = threading.Event()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body[:1])
                self.wfile.flush()
                reached.set()
                release.wait(5)
                self.wfile.write(body[1:])
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    handle, session = make_handle()
    applied = []
    rows = []
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(serving, "_switch_executor", lambda: pool)
    monkeypatch.setattr(serving, "_MODEL_SWITCH_RESOLVE_S", 0.05, raising=False)
    monkeypatch.setattr(configure, "warm_model_info", lambda *_: None)
    monkeypatch.setattr(session, "set_model", lambda *a, **kw: applied.append(a), raising=False)
    # A real pending response is the cancellation gate; unlike a clock-only
    # probe this cannot pass merely because HTTP setup had not started yet.
    with httpx.Client() as client:

        def build(provider: str, model_id: str) -> ModelSpec:
            rows.append(
                fetch_models(
                    "anthropic",
                    api_key="synthetic",
                    client=client,
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    timeout=2,
                )
            )
            return ModelSpec(provider=provider, model_id=model_id)

        monkeypatch.setattr(configure, "build_model_spec", build)
        try:
            monkeypatch.setattr(serving, "_MODEL_SWITCH_RESOLVE_S", 30)
            task = asyncio.create_task(handle.set_model("openai", "synthetic-cold"))
            assert await asyncio.to_thread(reached.wait, 2), "the loopback request never arrived"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert applied == []
            release.set()
            await asyncio.wrap_future(pool.submit(lambda: None))
            await asyncio.sleep(0)
            assert rows and rows[0] is not None, "the late HTTP listing must actually complete"
            assert applied == [], "successful late metadata must not switch the session"
        finally:
            release.set()
            pool.shutdown(wait=True)
            server.shutdown()
            server.server_close()
            thread.join()


@pytest.mark.asyncio
async def test_a_deferred_effort_switch_acks_queued_while_the_session_loop_is_blocked() -> None:
    """The operator's acceptance item: settings answer instantly even mid-step.

    A REAL ``ServingSessionHandle`` whose session loop is held by a synchronous
    ``time.sleep`` (a tool step or an eval holding the loop). Without
    ``defer_ok`` the reply could only come after the sleep, because the switch
    body hops onto that loop. With it the runtime answers ``queued`` at once,
    before any session-loop hop; the body then lands FIFO behind the blocking
    step and BEFORE the next thing the loop runs \u2014 which is where the agent
    loop's next ``_current_model`` read would be \u2014 and the outcome is
    published as ``settings_result`` for the ``request_id`` sent.
    """
    import time

    from tests.unit.session.runtime.test_serving import _AccessSession

    session_loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=session_loop.run_forever, daemon=True)
    loop_thread.start()
    session = _AccessSession("test/model")
    session.model = ModelSpec(provider="test", model_id="model", reasoning_efforts=("low", "high"))
    handle = serving.ServingSessionHandle(session, session_loop, cwd="/tmp")

    class Host(FakeHandle):
        """The wire host: switch ops go to the real serving handle."""

        async def set_effort(self, effort, *, request_id=None):  # noqa: ANN001, ANN202
            return await handle.set_effort(effort, request_id=request_id)

    runtime = RuntimeServer(Host(), kind="tui")
    runtime.start()
    writer = None
    # Long enough that an ack which waited on the loop could not pass the
    # 0.5 s acceptance bound below; no inline window exists any more.
    block_s = 3.0
    try:
        record = await _wait_record()
        reader, writer = await _dial(record, client="daemon")
        blocking = threading.Event()

        def hold() -> None:
            blocking.set()
            time.sleep(block_s)

        session_loop.call_soon_threadsafe(hold)
        assert await asyncio.to_thread(blocking.wait, 2)
        sent = time.monotonic()
        writer.write(
            json.dumps(
                {
                    "op": "set_effort",
                    "req": 1,
                    "effort": "high",
                    "defer_ok": True,
                    "request_id": "rid-1",
                }
            ).encode()
            + b"\n"
        )
        await writer.drain()
        ack = await asyncio.wait_for(_until(reader, "ack", 1), timeout=block_s + 5)
        acked_after = time.monotonic() - sent
        assert ack["detail"] == "queued", ack
        assert acked_after < 0.5, acked_after
        assert session.model.reasoning_effort != "high", "acked before it could land"

        # The "next step": session-loop work scheduled AFTER the switch was
        # sent, the way the turn's next ``_current_model`` read is. Hops land
        # FIFO, so the queued switch has applied by the time it runs.
        async def next_step() -> str | None:
            assert session.model is not None
            return session.model.reasoning_effort

        seen = asyncio.run_coroutine_threadsafe(next_step(), session_loop)
        assert await asyncio.wrap_future(seen) == "high"
        result = session._frontend_state_store.state.settings_result
        assert result is not None
        assert (result.request_id, result.ok, result.code) == ("rid-1", True, "applied")
        # The per-kind copies a redialled viewer and the phone read (R1-m2/M2).
        per_kind = session._frontend_state_store.state.settings_results
        assert per_kind["effort"].request_id == "rid-1" and per_kind["effort"].ok
        assert handle._fold.projection.settings_results["effort"]["request_id"] == "rid-1"
    finally:
        if writer is not None:
            writer.close()
        runtime.close()
        session_loop.call_soon_threadsafe(session_loop.stop)
        loop_thread.join(5)


@pytest.mark.asyncio
async def test_a_refused_identified_switch_publishes_its_refusal() -> None:
    """A queued switch's refusal has no reply left to ride: it goes on state."""
    from tests.unit.session.runtime.test_serving import _AccessSession

    session = _AccessSession("test/model")
    handle = serving.ServingSessionHandle(session, asyncio.get_running_loop(), cwd="/tmp")
    with pytest.raises(ValueError):
        await handle.set_effort("turbo", request_id="rid-2")
    result = session._frontend_state_store.state.settings_result
    assert result is not None
    assert (result.request_id, result.ok, result.code) == ("rid-2", False, "invalid")
    assert result.message == "model has no reasoning-effort levels; drop --effort"
    # The phone's correlated copy rides the projection (R1-M2), keyed by kind.
    projected = handle._fold.projection.settings_results["effort"]
    assert projected == {
        "request_id": "rid-2",
        "ok": False,
        "code": "invalid",
        "message": "model has no reasoning-effort levels; drop --effort",
    }
    assert handle._fold.projection.to_json()["settings_results"]["effort"]["ok"] is False
    # Unidentified switches publish nothing (the pre-``defer_ok`` contract).
    session._frontend_state_store.mutate(settings_result=None)
    with pytest.raises(ValueError):
        await handle.set_effort("turbo")
    assert session._frontend_state_store.state.settings_result is None


def test_a_projection_without_switches_omits_settings_results() -> None:
    """Sparse on the wire: frames with no identified switch stay byte-identical."""
    from local_operator.mobile.types import SessionProjection

    assert "settings_results" not in SessionProjection(session_id="s", pid=1).to_json()


@pytest.mark.asyncio
async def test_a_model_refusal_survives_a_later_effort_success_per_kind() -> None:
    """R1-m2: the single ``settings_result`` slot is overwritten across kinds.

    A viewer that redials after a model refusal and an effort success reads the
    full snapshot; ``settings_results`` must still carry the model refusal.
    """
    from tests.unit.session.runtime.test_serving import _AccessSession

    session = _AccessSession("test/model")
    session.model = ModelSpec(provider="test", model_id="model", reasoning_efforts=("low", "high"))
    handle = serving.ServingSessionHandle(session, asyncio.get_running_loop(), cwd="/tmp")
    handle._publish_settings_result(
        "rid-model", kind="model", ok=False, code="model_unknown", message="not offered"
    )
    await handle.set_effort("high", request_id="rid-effort")
    state = session._frontend_state_store.state
    assert state.settings_result is not None and state.settings_result.request_id == "rid-effort"
    assert state.settings_results["model"].request_id == "rid-model"
    assert state.settings_results["model"].ok is False
    assert state.settings_results["effort"].ok is True


@pytest.mark.asyncio
async def test_a_bare_settings_slash_listing_is_not_published_as_applied() -> None:
    """R1-m3: ``/model`` with no arguments is a listing, not a switch."""
    from local_operator.session.frontend_state import SlashResult
    from tests.unit.session.runtime.test_serving import _AccessSession

    session = _AccessSession("test/model")
    handle = serving.ServingSessionHandle(session, asyncio.get_running_loop(), cwd="/tmp")

    async def listing(*_args: Any, **_kwargs: Any) -> SlashResult:
        return SlashResult(kind="notice", text="models: test/model")

    async def passthrough(result: Any, *_args: Any) -> Any:
        return result

    handle._slash_result = listing  # type: ignore[method-assign]
    handle._complete_unconsumed_action = passthrough  # type: ignore[method-assign]
    await handle.run_slash_authoritative("model", "", request_id="rid-list")
    assert session._frontend_state_store.state.settings_result is None
    assert handle._fold.projection.settings_results == {}
    await handle.run_slash_authoritative("model", "test/other", request_id="rid-switch")
    published = session._frontend_state_store.state.settings_results["model"]
    assert (published.request_id, published.ok) == ("rid-switch", True)
