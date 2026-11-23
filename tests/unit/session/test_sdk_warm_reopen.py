"""Real SDK owner, native Bedrock bytes and event-stream frames; no network."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import signal
import socket
import sys
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from local_operator import sdk
from tests.unit.providers.bedrock import eventstream as es
from tests.unit.session.test_request_observation_sdk import scratch as shared_scratch

# Register the cohort's existing isolated-root fixture in this pytest module.
scratch = shared_scratch

MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def frames(
    *,
    tool: tuple[str, dict[str, Any]] | None = None,
    write: bool = False,
    read: bool = False,
    text: str = "done",
) -> list[dict[str, Any]]:
    blocks = None
    if tool is not None:
        name, args = tool
        blocks = [[
            {"type": "content_block_start", "content_block": {
                "type": "tool_use",
                "id": "toolu_" + hashlib.sha256(json.dumps(args).encode()).hexdigest()[:12],
                "name": name, "input": {},
            }},
            {"type": "content_block_delta", "delta": {
                "type": "input_json_delta", "partial_json": json.dumps(args),
            }},
            {"type": "content_block_stop"},
        ]]
    usage: dict[str, Any] = {"input_tokens": 35}
    if write:
        usage.update(
            cache_creation_input_tokens=2000, cache_creation={"ephemeral_5m_input_tokens": 2000}
        )
    if read:
        usage["cache_read_input_tokens"] = 2000
    return es.anthropic_events(
        text=text, usage=usage, extra_blocks=blocks,
        stop_reason="tool_use" if tool else "end_turn",
    )


class NativeWire:
    def __init__(self) -> None:
        self.bytes: list[bytes] = []
        self.responses: list[list[dict[str, Any]]] = []
        self.remote_denials: list[str] = []

    async def send(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "not-connected.invalid":
            self.remote_denials.append(request.method)
            return httpx.Response(404, request=request)
        assert request.url.host == "bedrock-runtime.us-west-2.amazonaws.com", request.url
        assert request.url.path.endswith("/invoke-with-response-stream"), request.url
        self.bytes.append(await request.aread())
        assert self.responses, "unplanned native request"
        return httpx.Response(
            200, headers={"content-type": "application/vnd.amazon.eventstream"},
            content=es.stream_bytes(self.responses.pop(0)), request=request,
        )


def patch_native(monkeypatch: pytest.MonkeyPatch) -> NativeWire:
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "offline-fixture")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    # Avoid even background fixture traffic; only AF_UNIX in-process/session
    # coordination is allowed. The captured wire is below the native encoder.
    connect = socket.socket.connect

    def local_connect(sock: socket.socket, address: Any) -> None:
        assert sock.family == socket.AF_UNIX, f"outbound socket: {address}"
        connect(sock, address)

    def no_dns(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"outbound DNS: {args}")

    monkeypatch.setattr(socket.socket, "connect", local_connect)
    monkeypatch.setattr(socket, "getaddrinfo", no_dns)
    wire = NativeWire()

    async def send(
        client: httpx.AsyncClient, request: httpx.Request, **kwargs: Any
    ) -> httpx.Response:
        return await wire.send(request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    return wire


@pytest.fixture
def native(monkeypatch: pytest.MonkeyPatch) -> NativeWire:
    return patch_native(monkeypatch)


def tools(body: bytes) -> list[dict[str, Any]]:
    return json.loads(body)["tools"]


def options(scratch: Any, **overrides: Any) -> sdk.SessionSpec:
    return sdk.SessionSpec(**dict(
        {"hosting": "bedrock", "model": MODEL, "world_root": scratch.cwd_path}, **overrides,
    ))


async def seed(scratch: Any, native: NativeWire, *, write: bool = True) -> tuple[str, Path]:
    # Activate through the shipped resolver, not an internal setter.
    native.responses.extend([
        frames(tool=("read", {"path": "tool://library"})),
        frames(),
        frames(write=write, read=not write),
    ])
    async with sdk.open_session(options(scratch), roots=scratch, warm_tool_prefix=True) as owner:
        await owner.prompt("read the tool documentation")
        await owner.prompt("settled")
        owner_id = owner.session_id
    assert len(native.bytes) == 3 and not native.responses
    assert "library" in [tool["name"] for tool in tools(native.bytes[-1])]
    receipt = scratch.config_path / "sessions" / owner_id / "sdk-warm-tool-prefix.json"
    assert receipt.is_file() is write, receipt
    return owner_id, receipt


def raw_tools(body: bytes) -> bytes:
    # Slice what the real encoder emitted; comparing parsed dicts alone loses
    # schema field order and would miss a provider cache-prefix regression.
    start = body.decode().index('"tools":') + len('"tools":')
    text = body.decode()[start:].lstrip()
    _, end = json.JSONDecoder().raw_decode(text)
    return text[:end].encode()


def blocks(body: bytes) -> list[tuple[str, dict[str, Any], Any]]:
    obj = json.loads(body)
    rows = [("tool", x) for x in obj["tools"]] + [("system", x) for x in obj["system"]]
    rows += [(msg["role"], x) for msg in obj["messages"] for x in msg["content"]]
    return [(kind, {k: v for k, v in row.items() if k != "cache_control"}, row.get("cache_control"))
            for kind, row in rows]


def prefix(left: bytes, right: bytes) -> tuple[int, int, int]:
    old = blocks(left)
    shared = marks = 0
    for a, b in zip(old, blocks(right)):
        if a[:2] != b[:2]:
            break
        shared += 1
        marks += a[2] is not None
    return shared, len(old), marks


async def reopen(scratch: Any, owner_id: str, native: NativeWire, *, enabled: bool) -> None:
    native.responses.extend([frames(read=True), frames(read=True)])
    async with sdk.open_session(
        options(scratch, resume=owner_id), roots=scratch, warm_tool_prefix=enabled
    ) as owner:
        assert owner.session_id == owner_id
        await owner.prompt("reopen")
        await owner.prompt("continuation")
    assert not native.responses


@pytest.mark.asyncio
async def test_owner_reopen_publication(scratch: Any, native: NativeWire) -> None:
    owner_id, receipt = await seed(scratch, native)
    previous, settled = native.bytes[0], native.bytes[2]
    original = receipt.read_bytes()
    assert set(json.loads(original)) == {
        "version", "scope", "publication", "wall_ms", "monotonic_ms",
    }
    assert len(original) < 512 and b"library" not in original and b"offline-fixture" not in original

    await reopen(scratch, owner_id, native, enabled=False)
    baseline = native.bytes[3]
    assert raw_tools(baseline) == raw_tools(previous)
    await reopen(scratch, owner_id, native, enabled=True)
    warm, next_turn = native.bytes[5:7]
    assert raw_tools(warm) == raw_tools(next_turn) == raw_tools(settled)
    base_prefix, warm_prefix = prefix(settled, baseline), prefix(settled, warm)
    print(f"native body bytes: settled={len(settled)}, baseline={len(baseline)}, warm={len(warm)}; "
          f"ordered shared blocks (count/old/marked): baseline={base_prefix}, warm={warm_prefix}; "
          f"receipt bytes={len(original)}")
    assert warm_prefix[0] > base_prefix[0]
    assert warm_prefix[2] > 0
    assert all(mark.get("ttl", "5m") == "5m" for _, _, mark in blocks(warm) if mark)
    assert receipt.read_bytes() == original  # no TTL renewal from reopen or two more turns


async def _child_reopen(args: list[str]) -> None:
    roots = sdk.SessionRoots(config_dir=Path(args[0]), agent_home=Path(args[1]),
                             cwd=Path(args[2]), allow_volatile=True)
    with pytest.MonkeyPatch.context() as patch:
        wire = patch_native(patch)
        await reopen(roots, args[3], wire, enabled=True)
        print(json.dumps([os.getpid(), [body.hex() for body in wire.bytes]]))


@pytest.mark.asyncio
async def test_reopen_across_processes(scratch: Any, native: NativeWire) -> None:
    owner_id, _ = await seed(scratch, native)
    repo = Path(__file__).resolve().parents[3]
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", (
            "import asyncio, sys; "
            "from tests.unit.session.test_sdk_warm_reopen import _child_reopen; "
            "asyncio.run(_child_reopen(sys.argv[1:]))"
        ), str(scratch.config_path), str(scratch.agent_home_path), str(scratch.cwd_path), owner_id,
        cwd=scratch.cwd_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
        env={"HOME": os.environ["HOME"], "PATH": os.environ["PATH"], "PYTHONPATH": str(repo)},
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), 30)
    finally:
        if proc.returncode is None:
            os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()
    assert proc.returncode == 0, stderr.decode()
    with pytest.raises(ProcessLookupError):
        os.killpg(proc.pid, 0)
    pid, bodies = json.loads(stdout.splitlines()[-1])
    assert pid != os.getpid()
    assert len(bodies) == 2
    for body in bodies:
        assert raw_tools(bytes.fromhex(body)) == raw_tools(native.bytes[2])
    print(f"separate seed/reopen processes: {os.getpid()} -> {pid}; both child arrays exact")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["no-write", "expiry", "no-live-catalogue", "generation", "schema", "array-order",
     "other-owner", "narrower", "mcp-origin", "world", "disconnected", "nonlocal"],
)
async def test_invalid_hint_is_all_or_none(
    scratch: Any, native: NativeWire, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    from local_operator.session.warm_tool_prefix import _tool_digest

    owner_id, receipt = await seed(scratch, native, write=case != "no-write")
    cold = raw_tools(native.bytes[0])
    spec = options(scratch, resume=owner_id)
    if case == "other-owner":
        native.responses.append(frames())
        async with sdk.open_session(options(scratch), roots=scratch) as other:
            await other.prompt("materialize a distinct owner")
            other_id = other.session_id
        assert other_id != owner_id
        (receipt.parent.parent / other_id / receipt.name).write_bytes(receipt.read_bytes())
        spec = options(scratch, resume=other_id)
    elif case == "narrower":
        spec = options(scratch, resume=owner_id, tools=("read",))
    elif case == "expiry":
        data = json.loads(receipt.read_bytes())
        data["monotonic_ms"] -= 300_000
        receipt.write_text(json.dumps(data))
    elif case == "nonlocal":
        (scratch.cwd_path / ".mcp.json").write_text(json.dumps({"mcpServers": {
            "remote": {"type": "http", "url": "https://not-connected.invalid/mcp"},
        }}))
        spec = options(scratch, resume=owner_id, mcp_ready_timeout_s=None)
    elif case == "mcp-origin":
        data = json.loads(receipt.read_bytes())
        data["origin"] = ["mcp", "different-world", "tool"]
        receipt.write_text(json.dumps(data))

    native.responses.extend([frames(), frames()])
    async with sdk.open_session(spec, roots=scratch, warm_tool_prefix=True) as session:
        owner = cast(Any, session)
        if case == "nonlocal":
            assert owner.mcp_manager.get_all_server_names() == ["remote"]
            assert owner.mcp_manager.get_server_config("remote").type == "http"
        # Change conditions *after* the SDK enrolled but before any request.
        with monkeypatch.context() as patch:
            if case == "no-live-catalogue":
                patch.setattr(owner, "mcp_manager", None)
            elif case == "disconnected":
                await owner.mcp_manager.disconnect_all()
            elif case == "schema":
                tool = next(tool for tool in owner._tools if tool.name == "library")
                patch.setattr(tool, "description", tool.description + " different schema")
            elif case == "array-order":
                data = json.loads(receipt.read_bytes())
                preview = owner._publishable(owner._tools, activated=frozenset({"library"}))
                data["publication"] = _tool_digest(list(reversed(preview)))
                receipt.write_text(json.dumps(data))
            elif case == "generation":
                original = owner.mcp_manager.empty_catalogue_epoch()
                revisions = iter((original, original + 1))
                patch.setattr(owner.mcp_manager, "empty_catalogue_epoch", lambda: next(revisions))
            elif case == "world":
                owner.set_tool_confinement(scratch.cwd_path / "different-world")
            await owner.prompt(f"reject {case}")
        assert "library" not in owner._activated_tools
        await owner.prompt("no second restore attempt")
    names = [[t["name"] for t in tools(body)] for body in native.bytes[-2:]]
    if case == "narrower":
        assert names == [["read"], ["read"]]
    else:
        assert all(raw_tools(body) == cold for body in native.bytes[-2:])
    assert not native.responses


@pytest.mark.asyncio
async def test_live_mcp_blocks_restore_and_current_gate_decides(
    scratch: Any, native: NativeWire
) -> None:
    owner_id, receipt = await seed(scratch, native)
    before = receipt.read_bytes()
    script = scratch.cwd_path / "fixture.py"
    calls = scratch.cwd_path / "stub-calls.log"
    script.write_text('''from pathlib import Path
from mcp.server.mcpserver import MCPServer
server = MCPServer(name="localstub", log_level="WARNING")
@server.tool()
def inspect_stub(label: str) -> str:
    with Path(__file__).with_name("stub-calls.log").open("a") as log:
        log.write(label + "\\n")
    return "LOCAL STUB " + label
server.run()
''')
    (scratch.cwd_path / ".mcp.json").write_text(json.dumps({"mcpServers": {
        "localstub": {"command": sys.executable, "args": [str(script)], "preloadTools": True},
    }}))
    tool = "mcp__localstub_inspect_stub"
    consent_count = 0

    async def consent(*_args: Any) -> bool:
        nonlocal consent_count
        consent_count += 1
        return True

    async def literal(label: str, policy: sdk.ApprovalPolicy) -> bytes:
        native.responses.extend([frames(tool=(tool, {"label": label})), frames()])
        async with sdk.open_session(options(scratch, resume=owner_id, approvals=policy),
                                    roots=scratch, warm_tool_prefix=True) as session:
            await session.prompt(f"invoke inert local fixture ({label})")
            manager = cast(Any, session).mcp_manager
            assert manager.get_connection_status("localstub") == "connected"
            assert any(t.name == tool for t in manager.get_server_tools("localstub"))
        names = [t["name"] for t in tools(native.bytes[-2])]
        assert tool in names and "library" not in names
        return native.bytes[-1]

    # An actually approved execution exists in transcript, not in the hint.
    assert b"LOCAL STUB approved-before-close" in await literal(
        "approved-before-close", sdk.ApprovalPolicy.callback(consent)
    )
    assert calls.read_text().splitlines() == ["approved-before-close"]
    denied = await literal("must-not-execute", sdk.ApprovalPolicy.refuse())
    assert b"LOCAL STUB must-not-execute" not in denied
    assert b'"is_error":true' in denied
    assert calls.read_text().splitlines() == ["approved-before-close"]
    assert consent_count == 1
    assert b"LOCAL STUB approved-anew" in await literal(
        "approved-anew", sdk.ApprovalPolicy.callback(consent)
    )
    assert consent_count == 2 and calls.read_text().splitlines() == [
        "approved-before-close", "approved-anew",
    ]
    assert receipt.read_bytes() == before
    print("live stdio rejected warm hint; executions: prior approval and fresh consent only")


@pytest.mark.asyncio
async def test_short_declared_request_bytes_unchanged(scratch: Any, native: NativeWire) -> None:
    # Same logical owner, transcript, prompt, tools and structured response;
    # vary only the option/receipt. No hand-built provider body on either arm.
    (scratch.cwd_path / "fixture.txt").write_text("needle fixture finding\n")
    short = dict(tools=("read", "grep"), output_format="json")
    native.responses.append(frames(text='{"seed":true}'))
    async with sdk.open_session(options(scratch, **short), roots=scratch) as owner:
        await owner.prompt("seed structured output")
        owner_id = owner.session_id
    session_dir = scratch.config_path / "sessions" / owner_id
    snapshot = scratch.config_path / "short-fixture-backup"
    shutil.copytree(session_dir, snapshot)

    async def run(*, stale: bool) -> list[bytes]:
        start = len(native.bytes)
        native.responses.extend([
            frames(tool=("grep", {"pattern": "needle", "path": "fixture.txt"})),
            frames(text='{"found":true}'),
        ])
        async with sdk.open_session(options(scratch, resume=owner_id, **short), roots=scratch,
                                    **({"warm_tool_prefix": True} if stale else {})) as owner:
            await owner.prompt("find needle in fixture.txt and respond with JSON")
        result = native.bytes[start:]
        assert len(result) == 2 and b"needle fixture finding" in result[1]
        assert all({tool["name"] for tool in tools(body)} == {"read", "grep"} for body in result)
        return result

    default = await run(stale=False)
    session_dir.rename(scratch.config_path / "short-fixture-default-result")
    shutil.copytree(snapshot, session_dir)
    (session_dir / "sdk-warm-tool-prefix.json").write_text(json.dumps({
        "version": 1, "scope": "0" * 64, "publication": "0" * 64,
        "wall_ms": 0, "monotonic_ms": 0,
    }))
    opted_in = await run(stale=True)
    assert opted_in == default
    sizes = [len(body) for body in default]
    print(f"short declared search + structured output native bytes unchanged: {sizes}")
