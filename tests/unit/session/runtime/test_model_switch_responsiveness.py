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
