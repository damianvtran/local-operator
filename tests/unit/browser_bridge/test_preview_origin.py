"""Fixed preview origins never mint a second family approval/notification."""

from __future__ import annotations

import asyncio
import errno
import json
import os
import stat
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
    assert parent.session_dir is not None
    child = session(tmp_path, "child", "root")
    assert child.session_dir is not None
    sibling = session(tmp_path, "sibling", "root")
    assert sibling.session_dir is not None
    nested = session(tmp_path, "nested", "child")
    assert nested.session_dir is not None
    first = await request(child, "https://FIRST.trycloudflare.com./path")
    assert not first.is_error and "Fixed-origin rule" in first.text
    assert "https://first.trycloudflare.com" in first.text
    assert backend.await_count == 1
    repeat = await request(parent, "http://first.trycloudflare.com:8080/other")
    assert not repeat.is_error and backend.await_count == 2
    for member in (parent, sibling, nested):
        denied = await request(member, "https://second.trycloudflare.com")
        assert denied.is_error
        assert denied.details is not None
        assert denied.details["code"] == "preview_origin_changed"
        assert denied.details is not None
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
    assert ctx.session_dir is not None
    await request(ctx, "https://first.trycloudflare.com")
    for action in ("await_access", "cancel_access"):
        result = await request(ctx, "https://other.trycloudflare.com", action)
        assert not result.is_error  # These operations retain existing semantics.
    assert backend.await_count == 3
    resumed = session(tmp_path, "root")
    assert resumed.session_dir is not None
    assert (await request(resumed, "https://other.trycloudflare.com")).is_error
    assert backend.await_count == 3
    assert not (await request(resumed, "https://first.trycloudflare.com")).is_error


@pytest.mark.asyncio
async def test_rpc_failure_keeps_pin_and_rule(tmp_path, backend):
    backend.return_value = (None, builtin._error("test", "browser", "isolated transport failure"))
    ctx = session(tmp_path, "root")
    assert ctx.session_dir is not None
    first = await request(ctx, "https://first.trycloudflare.com")
    assert first.is_error and "Fixed-origin rule" in first.text
    assert "isolated transport failure" in first.text
    second = await request(ctx, "https://second.trycloudflare.com")
    assert second.details is not None
    assert second.details["code"] == "preview_origin_changed"
    assert backend.await_count == 1


@pytest.mark.parametrize("failed_kind", ["file", "directory"])
@pytest.mark.asyncio
async def test_repeated_sync_failure_never_authorizes_rpc(
    tmp_path, backend, monkeypatch, failed_kind
):
    from local_operator.browser_bridge import preview_origin as policy

    root = session(tmp_path, "root")
    assert root.session_dir is not None
    child = session(tmp_path, "child", "root")
    assert child.session_dir is not None
    real_fsync = os.fsync
    failed_syncs = []

    def fail_sync(fd):
        kind = "directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"
        if kind == failed_kind:
            failed_syncs.append(kind)
            raise OSError(errno.EIO, "injected persistent sync failure")
        return real_fsync(fd)

    monkeypatch.setattr(policy.os, "fsync", fail_sync)
    pin_path = Path(root.session_dir) / "browser-preview-origin.json"
    pin_bytes = None
    for ctx in (root, root, child):
        result = await request(ctx, "https://first.trycloudflare.com")
        assert result.details is not None
        assert result.details["code"] == "preview_origin_unavailable"
        assert result.details is not None
        assert result.details["approval_request_sent"] is False
        backend.assert_not_awaited()
        if pin_bytes is None:
            pin_bytes = pin_path.read_bytes()
        assert pin_path.read_bytes() == pin_bytes
    assert failed_syncs == [failed_kind] * 3
    assert not (Path(child.session_dir) / pin_path.name).exists()

    # Clearing the fault commits the same identity, not a replacement pin.
    monkeypatch.setattr(policy.os, "fsync", real_fsync)
    assert not (await request(child, "https://first.trycloudflare.com")).is_error
    assert backend.await_count == 1
    changed = await request(root, "https://second.trycloudflare.com")
    assert changed.details is not None
    assert changed.details["code"] == "preview_origin_changed"
    assert backend.await_count == 1
    assert pin_path.read_bytes() == pin_bytes


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
    assert ctx.session_dir is not None
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
    assert not list(tmp_path.rglob("browser-preview-origin.*"))


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
    parsed = quick_tunnel_origin(url)
    assert parsed is not None
    host, origin = parsed
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
    assert ctx.session_dir is not None
    (Path(ctx.session_dir) / "origin.json").write_text(marker)
    result = await request(ctx, "https://first.trycloudflare.com")
    assert result.details is not None
    assert result.is_error and result.details["code"] == "preview_origin_unavailable"
    backend.assert_not_awaited()


@pytest.mark.asyncio
async def test_corrupt_pin_fails_closed(tmp_path, backend):
    ctx = session(tmp_path, "root")
    assert ctx.session_dir is not None
    (Path(ctx.session_dir) / "browser-preview-origin.json").write_bytes(b"broken database")
    result = await request(ctx, "https://first.trycloudflare.com")
    assert result.details is not None
    assert result.details["code"] == "preview_origin_unavailable"
    backend.assert_not_awaited()


def test_atomic_competing_threads_and_persisted_process_resume(tmp_path):
    ctx = session(tmp_path, "root")
    assert ctx.session_dir is not None
    child = session(tmp_path, "child", "root")
    assert child.session_dir is not None
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
    assert winner is not None
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
    assert os.stat(Path(ctx.session_dir) / "browser-preview-origin.json").st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_concurrent_public_requests_dispatch_one_host(tmp_path, backend):
    contexts = [session(tmp_path, "root"), session(tmp_path, "child", "root")]
    results = await asyncio.gather(
        *[request(ctx, f"https://host{i}.trycloudflare.com") for i, ctx in enumerate(contexts)]
    )
    assert sum(not result.is_error for result in results) == 1
    assert backend.await_count == 1


ALIASES = [
    "https://ｓｅｃｏｎｄ.trycloudflare.com",
    "https://second。trycloudflare.com",
    "https://second．trycloudflare.com",
    "https://second｡trycloudflare.com",
    "https:////second.trycloudflare.com",
]


@pytest.mark.parametrize("url", ALIASES)
@pytest.mark.asyncio
async def test_browser_aliases_pin_and_refuse_without_dispatch(tmp_path, backend, url):
    ctx = session(tmp_path, "root")
    assert ctx.session_dir is not None
    first = await request(ctx, url)
    assert not first.is_error
    parsed = quick_tunnel_origin(url)
    assert parsed is not None
    assert parsed[0] == "second.trycloudflare.com"
    assert not (await request(ctx, "https://second.trycloudflare.com")).is_error
    assert backend.await_count == 2
    denied = await request(ctx, "https://third.trycloudflare.com")
    assert denied.is_error and backend.await_count == 2
    assert denied.text.startswith("Refused preview origin https://third.trycloudflare.com.")
    assert "No new approval request was sent" in denied.text
    assert "Do not await this attempt" in denied.text


@pytest.mark.parametrize("url", ALIASES)
@pytest.mark.asyncio
async def test_second_browser_alias_has_zero_rpcs(tmp_path, backend, url):
    ctx = session(tmp_path, "root")
    assert ctx.session_dir is not None
    await request(ctx, "https://first.trycloudflare.com")
    result = await request(ctx, url)
    assert result.details is not None
    assert result.details["code"] == "preview_origin_changed"
    assert backend.await_count == 1


def test_nontransitional_idna_and_nonquick_international_domains():
    for url, host in (
        ("https://ß.trycloudflare.com", "xn--zca.trycloudflare.com"),
        ("https://xn--zca.trycloudflare.com", "xn--zca.trycloudflare.com"),
        ("https://ss.trycloudflare.com", "ss.trycloudflare.com"),
    ):
        parsed = quick_tunnel_origin(url)
        assert parsed is not None
        assert parsed[0] == host
    for url in ("https://bücher.example", "https://ß.example", "https://example。com"):
        assert quick_tunnel_origin(url) is None


@pytest.mark.asyncio
async def test_registry_survives_missing_creator_marker(tmp_path, backend):
    from local_operator.guardrails.provenance import read_lineage
    from local_operator.resume import mark_session_origin

    store = tmp_path / "sessions"
    root = session(store, "familyroot")
    assert root.session_dir is not None
    child = session(store, "child")
    assert child.session_dir is not None
    mark_session_origin(Path(child.session_dir), "subagent", parent="familyroot")
    assert read_lineage(tmp_path)["child"]["parent"] == "familyroot"
    await request(root, "https://first.trycloudflare.com")
    (Path(child.session_dir) / "origin.json").unlink()  # Fixture owned by this test.
    denied = await request(child, "https://second.trycloudflare.com")
    assert denied.details is not None
    assert denied.details["code"] == "preview_origin_changed"
    assert backend.await_count == 1
    assert not (Path(child.session_dir) / "browser-preview-origin.json").exists()


@pytest.mark.asyncio
async def test_runtime_ancestry_and_conflict_fail_closed(tmp_path, backend):
    root = session(tmp_path, "root")
    assert root.session_dir is not None
    child = session(tmp_path, "child")
    assert child.session_dir is not None
    child.ancestor_session_ids = ("root",)
    await request(root, "https://first.trycloudflare.com")
    assert (await request(child, "https://second.trycloudflare.com")).is_error
    assert backend.await_count == 1
    (Path(child.session_dir) / "origin.json").write_text('{"parent":"other"}')
    result = await request(child, "https://first.trycloudflare.com")
    assert result.details is not None
    assert result.details["code"] == "preview_origin_unavailable"
    assert backend.await_count == 1


@pytest.mark.parametrize("target", ["directory", "marker", "pin", "store"])
@pytest.mark.asyncio
async def test_symlinks_never_write_outside_or_dispatch(tmp_path, backend, target):
    store = tmp_path / "sessions"
    ctx = session(store, "root")
    assert ctx.session_dir is not None
    outside = tmp_path / "outside"
    outside.mkdir()
    unrelated = outside / "unrelated.json"
    unrelated.write_text('{"unrelated":true}')
    if target == "directory":
        (store / "parent").symlink_to(outside, target_is_directory=True)
        (Path(ctx.session_dir) / "origin.json").write_text('{"parent":"parent"}')
    elif target == "marker":
        (Path(ctx.session_dir) / "origin.json").symlink_to(unrelated)
    elif target == "pin":
        (Path(ctx.session_dir) / "browser-preview-origin.json").symlink_to(unrelated)
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(store, target_is_directory=True)
        ctx.session_dir = str(alias / "root")
    result = await request(ctx, "https://first.trycloudflare.com")
    assert result.details is not None
    assert result.details["code"] == "preview_origin_unavailable"
    backend.assert_not_awaited()
    assert unrelated.read_text() == '{"unrelated":true}'
    assert sorted(p.name for p in outside.iterdir()) == ["unrelated.json"]


@pytest.mark.parametrize("target", ["directory", "pin"])
@pytest.mark.asyncio
async def test_swap_between_descriptor_open_and_write_cannot_escape(
    tmp_path, backend, monkeypatch, target
):
    from local_operator.browser_bridge import preview_origin as policy

    ctx = session(tmp_path / "sessions", "root")
    assert ctx.session_dir is not None
    root = Path(ctx.session_dir)
    outside = tmp_path / "outside"
    outside.mkdir()
    unrelated = outside / "unrelated.json"
    unrelated.write_text("unchanged")
    real_open = os.open
    swapped = False

    def swap_open(path, flags, *args, **kwargs):
        nonlocal swapped
        fd = real_open(path, flags, *args, **kwargs)
        if path == "browser-preview-origin.json" and not swapped:
            swapped = True
            if target == "directory":
                root.rename(root.with_name("moved"))
                root.symlink_to(outside, target_is_directory=True)
            else:
                pin = root / "browser-preview-origin.json"
                pin.rename(root / "old-pin.json")
                pin.symlink_to(unrelated)
        return fd

    monkeypatch.setattr(policy.os, "open", swap_open)
    result = await request(ctx, "https://first.trycloudflare.com")
    assert result.details is not None
    assert swapped and result.details["code"] == "preview_origin_unavailable"
    backend.assert_not_awaited()
    assert unrelated.read_text() == "unchanged"
    assert sorted(p.name for p in outside.iterdir()) == ["unrelated.json"]


@pytest.mark.asyncio
async def test_denial_guidance_and_transport_separator(tmp_path, backend):
    ctx = session(tmp_path, "root")
    assert ctx.session_dir is not None
    backend.return_value = ({"state": "denied", "origin": "https://first.trycloudflare.com"}, None)
    denied = await request(ctx, "https://first.trycloudflare.com")
    assert "do not re-request it" in denied.text
    assert "Coordinate reuse" not in denied.text
    backend.return_value = (None, builtin._error("test", "browser", "transport failure"))
    failed = await request(ctx, "https://first.trycloudflare.com")
    assert "transport failure\n\nFixed-origin rule" in failed.text


@pytest.mark.asyncio
async def test_sharp_s_is_not_merged_with_ss(tmp_path, backend):
    ctx = session(tmp_path, "root")
    assert ctx.session_dir is not None
    await request(ctx, "https://ß.trycloudflare.com")
    assert not (await request(ctx, "https://xn--zca.trycloudflare.com")).is_error
    assert (await request(ctx, "https://ss.trycloudflare.com")).is_error
    assert backend.await_count == 2


def test_competing_processes_share_first_durable_pin(tmp_path):
    ctx = session(tmp_path, "root")
    assert ctx.session_dir is not None
    child = session(tmp_path, "child", "root")
    assert child.session_dir is not None
    code = (
        "from pathlib import Path; import sys; "
        "from local_operator.browser_bridge.preview_origin import pin_quick_tunnel; "
        "sys.stdin.readline(); "
        "p=pin_quick_tunnel(Path(sys.argv[1]),sys.argv[2],'https://'+sys.argv[2]); "
        "print(p.host,p.allowed)"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code, directory, host],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for directory, host in (
            (ctx.session_dir, "a.trycloudflare.com"),
            (child.session_dir, "b.trycloudflare.com"),
        )
    ]
    try:
        for process in processes:
            assert process.stdin is not None
            process.stdin.write("go\n")
            process.stdin.flush()
        outputs = [process.communicate(timeout=30) for process in processes]
        assert all(process.returncode == 0 for process in processes), outputs
        lines = [out.strip().split() for out, _ in outputs]
        assert lines[0][0] == lines[1][0]
        assert sorted(line[1] for line in lines) == ["False", "True"]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()


@pytest.mark.parametrize("iteration", range(12))
def test_atomic_first_file_creation_is_race_safe(tmp_path, iteration):
    # Repeating fresh creation exercises Darwin's openat(O_CREAT) losing-race
    # ENOENT, which an already-existing pin cannot reproduce.
    root = session(tmp_path, "root")
    assert root.session_dir is not None
    gate = Barrier(2)

    def pin(host):
        gate.wait()
        assert root.session_dir is not None
        return pin_quick_tunnel(Path(root.session_dir), host, "https://" + host)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(pin, host) for host in ("a.trycloudflare.com", "b.trycloudflare.com")
        ]
        pins = [future.result() for future in futures]
    assert sum(pin.allowed for pin in pins) == 1
    assert pins[0].host == pins[1].host
