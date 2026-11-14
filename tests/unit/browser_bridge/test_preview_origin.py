"""Fixed preview origins never mint a second family approval/notification."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import AsyncMock

import pytest

from local_operator.browser_bridge.preview_origin import (
    pin_quick_tunnel,
    quick_tunnel_origin,
)
from local_operator.harness.types import BrowserSurface, ToolContext
from local_operator.tools import builtin


def session(root: Path, sid: str, parent: str = "") -> ToolContext:
    directory = root / sid
    directory.mkdir(parents=True, exist_ok=True)
    if parent:
        (directory / "origin.json").write_text(json.dumps({"origin": "subagent", "parent": parent}))
    return ToolContext(session_id=sid, session_dir=str(directory), browser=BrowserSurface())


@pytest.fixture
def backend(monkeypatch):
    # Discovery only is synthetic; execute_browser still parses and dispatches
    # the public action. No live host, profile or approval can be reached.
    monkeypatch.setattr(builtin, "bridge_browser_reachable", AsyncMock(return_value=True))
    monkeypatch.setattr(builtin, "ui_browser_reachable", AsyncMock(return_value=False))
    call = AsyncMock(
        return_value=({"state": "allowed", "origin": "https://first.trycloudflare.com"}, None)
    )
    monkeypatch.setattr(builtin, "_bridge_call", call)
    return call


async def request(ctx, url, action="request_access"):
    return await builtin.execute_browser(
        "test-preview", {"action": action, "url": url, "backend": "extension"}, None, None, ctx
    )


@pytest.mark.asyncio
async def test_public_family_repeat_siblings_nested_and_unrelated(tmp_path, backend):
    parent = session(tmp_path, "root")
    child = session(tmp_path, "child", "root")
    sibling = session(tmp_path, "sibling", "root")
    nested = session(tmp_path, "nested", "child")
    first = await request(child, "https://FIRST.trycloudflare.com./path")
    assert not first.is_error and "Fixed-origin rule" in first.text
    assert "https://first.trycloudflare.com" in first.text
    assert backend.await_count == 1
    repeat = await request(parent, "http://first.trycloudflare.com:8080/other")
    assert not repeat.is_error and backend.await_count == 2
    for member in (parent, sibling, nested):
        denied = await request(member, "https://second.trycloudflare.com")
        assert denied.is_error
        assert denied.details["code"] == "preview_origin_changed"
        assert denied.details["pinned_origin"] == "https://first.trycloudflare.com"
        assert "ONE fixed origin per task" in denied.text
        assert backend.await_count == 2  # ZERO second-host approval RPCs.
    assert not (
        await request(session(tmp_path, "unrelated"), "https://second.trycloudflare.com")
    ).is_error
    assert backend.await_count == 3


@pytest.mark.asyncio
async def test_cancel_await_and_resumed_runtime_do_not_reset_pin(tmp_path, backend):
    ctx = session(tmp_path, "root")
    await request(ctx, "https://first.trycloudflare.com")
    for action in ("await_access", "cancel_access"):
        result = await request(ctx, "https://other.trycloudflare.com", action)
        assert not result.is_error  # These operations retain existing semantics.
    assert backend.await_count == 3
    resumed = session(tmp_path, "root")
    assert (await request(resumed, "https://other.trycloudflare.com")).is_error
    assert backend.await_count == 3
    assert not (await request(resumed, "https://first.trycloudflare.com")).is_error


@pytest.mark.asyncio
async def test_rpc_failure_keeps_pin_and_rule(tmp_path, backend):
    backend.return_value = (None, builtin._error("test", "browser", "isolated transport failure"))
    ctx = session(tmp_path, "root")
    first = await request(ctx, "https://first.trycloudflare.com")
    assert first.is_error and "Fixed-origin rule" in first.text
    assert "isolated transport failure" in first.text
    second = await request(ctx, "https://second.trycloudflare.com")
    assert second.details["code"] == "preview_origin_changed"
    assert backend.await_count == 1


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://localhost:4321",
        "http://127.0.0.1:4321",
        "https://trycloudflare.com",
        "https://nottrycloudflare.com",
        "https://a.trycloudflare.com.evil.invalid",
        "https://example.com/a.trycloudflare.com",
        "https://a.trycloudflare.com@example.com",
        "not a URL",
        "https://[broken",
        "https://a.trycloudflare.com:invalid",
        "ftp://a.trycloudflare.com",
        "https://a..trycloudflare.com",
        "https://a.trycloudflare.com..",
    ],
)
@pytest.mark.asyncio
async def test_non_quick_and_invalid_urls_retain_backend_semantics(tmp_path, backend, url):
    assert quick_tunnel_origin(url) is None
    ctx = session(tmp_path, "root")
    result = await request(ctx, url)
    # Existing public validation refuses non-HTTP syntax before discovery;
    # other malformed HTTP values remain the backend's responsibility.
    if url in {"not a URL", "ftp://a.trycloudflare.com"}:
        assert result.is_error and "refusing" in result.text
        backend.assert_not_awaited()
    else:
        assert not result.is_error
        assert backend.await_count == 1
        assert backend.call_args.args[2]["url"] == url
    assert not list(tmp_path.rglob("*.sqlite3"))


@pytest.mark.parametrize(
    "url",
    [
        "https://A.trycloudflare.com",
        "https://a.trycloudflare.com./x",
        "https://user:pass@a.trycloudflare.com/path",
        "https://a.trycloudflare.com:443",
        "https://%61.trycloudflare.com",
        "https://a%2etrycloudflare.com",
        r"https://a.trycloudflare.com\@example.com",
    ],
)
def test_real_host_normalized_without_credentials(url):
    host, origin = quick_tunnel_origin(url)
    assert host == "a.trycloudflare.com"
    assert origin.startswith("https://a.trycloudflare.com")
    assert "user" not in origin and "pass" not in origin


@pytest.mark.parametrize(
    "marker",
    [
        "{",
        "[]",
        '{"origin":"subagent"}',
        '{"parent":"../escape"}',
        '{"parent":"root"}',
        '{"parent":"missing"}',
        '{"parent":123}',
        '{"parent":false}',
        '{"parent":[]}',
    ],
)
@pytest.mark.asyncio
async def test_damaged_ancestry_fails_closed_without_rpc(tmp_path, backend, marker):
    ctx = session(tmp_path, "root")
    (Path(ctx.session_dir) / "origin.json").write_text(marker)
    result = await request(ctx, "https://first.trycloudflare.com")
    assert result.is_error and result.details["code"] == "preview_origin_unavailable"
    backend.assert_not_awaited()


@pytest.mark.asyncio
async def test_corrupt_pin_fails_closed(tmp_path, backend):
    ctx = session(tmp_path, "root")
    (Path(ctx.session_dir) / "browser-preview-origin.sqlite3").write_bytes(b"broken database")
    result = await request(ctx, "https://first.trycloudflare.com")
    assert result.details["code"] == "preview_origin_unavailable"
    backend.assert_not_awaited()


def test_atomic_competing_threads_and_persisted_process_resume(tmp_path):
    ctx = session(tmp_path, "root")
    child = session(tmp_path, "child", "root")
    gate = Barrier(2)

    def compete(directory, host):
        gate.wait()
        return pin_quick_tunnel(Path(directory), host, f"https://{host}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(compete, ctx.session_dir, "a.trycloudflare.com")
        b = pool.submit(compete, child.session_dir, "b.trycloudflare.com")
        pins = [a.result(), b.result()]
    assert sum(pin.allowed for pin in pins) == 1
    assert pins[0].host == pins[1].host
    winner = pins[0].host
    code = (
        "from pathlib import Path; "
        "from local_operator.browser_bridge.preview_origin import pin_quick_tunnel; "
        "import sys; p=pin_quick_tunnel(Path(sys.argv[1]),sys.argv[2],'https://'+sys.argv[2]); "
        "print(p.host,p.allowed)"
    )
    # A fresh interpreter has no process-local family/pin cache to lean on.
    result = subprocess.run(
        [sys.executable, "-c", code, child.session_dir, winner],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == f"{winner} True"
    assert (
        os.stat(Path(ctx.session_dir) / "browser-preview-origin.sqlite3").st_mode & 0o777 == 0o600
    )


@pytest.mark.asyncio
async def test_concurrent_public_requests_dispatch_one_host(tmp_path, backend):
    contexts = [session(tmp_path, "root"), session(tmp_path, "child", "root")]
    results = await asyncio.gather(
        *[request(ctx, f"https://host{i}.trycloudflare.com") for i, ctx in enumerate(contexts)]
    )
    assert sum(not result.is_error for result in results) == 1
    assert backend.await_count == 1
