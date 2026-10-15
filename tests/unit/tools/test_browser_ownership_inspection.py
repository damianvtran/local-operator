"""Real browser tool -> resource -> authenticated loopback HTTP, synthetic state only."""

from __future__ import annotations

import asyncio
import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.browser_bridge import resources
from local_operator.browser_bridge.backend import BridgeClient
from local_operator.harness.types import BrowserSurface, ToolContext
from local_operator.tools import builtin


@pytest.fixture
def peer(monkeypatch: pytest.MonkeyPatch):
    state: dict[str, Any] = {
        "calls": [],
        "result": {
            "version": 1,
            "session_match": True,
            "scope_missing": False,
            "generation_current": False,
            "generation_previous": True,
            "generation_ancestry": True,
            "request_ancestry_count": 1,
        },
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            pass

        def do_POST(self) -> None:
            assert self.headers["X-Bridge-Key"] == "synthetic-http-key"
            assert self.path == "/rpc"
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["calls"].append(request)
            response = {"id": request["id"], "ok": "error" not in state}
            response["error" if "error" in state else "result"] = state.get(
                "error", state["result"]
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

    class Server(ThreadingHTTPServer):
        # Eight simultaneous clients must test snapshot concurrency, not the
        # stdlib fixture's default listen backlog of five. Admission failures
        # there would conflate transport saturation with ownership behavior.
        request_queue_size = 32

    server = Server(("127.0.0.1", 0), Handler)
    # The default half-second shutdown poll dominates these tiny RPC cases;
    # keep teardown event-responsive instead of burning the unit shard budget.
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    monkeypatch.setattr(
        BridgeClient,
        "_read",
        lambda _self: SimpleNamespace(port=server.server_port, session_key="synthetic-http-key"),
    )
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def setup_resource(tmp_path: Path) -> tuple[resources.BrowserResource, ToolContext]:
    resource = resources.BrowserResource(tmp_path / "synthetic-session", "synthetic-session")
    resource.initialize()
    resource.record["bridge_generations"] = ["synthetic-previous"]
    resource._save()
    ctx = ToolContext(cwd=str(tmp_path))
    ctx.browser = BrowserSurface(resource)
    return resource, ctx


def call(ctx: ToolContext) -> dict[str, Any]:
    result = asyncio.run(
        builtin.execute_browser("synthetic-call", {"action": "inspect_ownership"}, context=ctx)
    )
    assert result.details == json.loads(result.content[0].text)
    return result.details


def forbid(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("inspection entered mutation/recovery/approval/selection")


def test_actual_tool_http_path_preserves_disk_and_memory(
    tmp_path: Path, peer: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    resource, ctx = setup_resource(tmp_path)
    before = resource.path.read_bytes()
    record = copy.deepcopy(resource.record)
    attrs = (resource.generation, resource.previous_generation, resource.host, resource.ownership)
    for name in (
        "initialize",
        "params",
        "client",
        "select_host",
        "ensure_recovered",
        "recover",
        "_save",
        "allocate",
    ):
        monkeypatch.setattr(resource, name, forbid)
    for name in (
        "_execute_browser",
        "_ownership_lane_host",
        "_explicit_open_lane",
        "_check_approval",
        "_browser_update_note",
    ):
        monkeypatch.setattr(builtin, name, forbid)
    assert call(ctx) == peer["result"]
    assert len(peer["calls"]) == 1
    assert peer["calls"][0]["method"] == "owner_inspect"
    assert peer["calls"][0]["params"]["owner_proof"] == resource.record["proof"]
    assert peer["calls"][0]["params"]["requester"] == "session:synthetic-session"
    assert resource.path.read_bytes() == before
    assert resource.record == record
    assert (
        resource.generation,
        resource.previous_generation,
        resource.host,
        resource.ownership,
    ) == attrs
    assert ctx.browser.surface_id == ""
    assert not ctx.browser.extension_update_notified


@pytest.mark.parametrize("code", ["unsupported_method", "capability_unsupported", "internal"])
def test_old_peers_are_typed_unsupported_no_retry(
    tmp_path: Path, peer: dict[str, Any], code: str
) -> None:
    resource, ctx = setup_resource(tmp_path)
    before = resource.path.read_bytes()
    peer["error"] = {
        "code": code,
        "message": "secret-proof-generation-url",
        "data": {"ledger": "private"},
    }
    assert call(ctx) == {"version": 1, "error_code": "ownership_inspection_unsupported"}
    assert len(peer["calls"]) == 1
    assert resource.path.read_bytes() == before


def test_peer_response_is_sanitized_including_foreign_counts(
    tmp_path: Path, peer: dict[str, Any]
) -> None:
    _, ctx = setup_resource(tmp_path)
    peer["result"] = {
        "version": 1,
        "session_match": False,
        "allocations_total": 42,
        "generation_current": True,
        "request_ancestry_count": 16,
        "owner_proof": "secret",
    }
    assert call(ctx) == {"version": 1, "session_match": False}
    peer["result"] = {
        "version": 1,
        "session_match": True,
        "generation_current": "secret",
        "request_ancestry_count": 999999,
        "ledger": "private",
    }
    assert call(ctx) == {"version": 1, "session_match": True}


def test_missing_record_does_not_initialize_or_dial(tmp_path: Path, peer: dict[str, Any]) -> None:
    directory = tmp_path / "absent"
    resource = resources.BrowserResource(directory, "synthetic")
    ctx = ToolContext(cwd=str(tmp_path))
    ctx.browser = BrowserSurface(resource)
    assert call(ctx) == {"version": 1, "local_scope_missing": True}
    assert not directory.exists()
    assert peer["calls"] == []
    ctx.browser = None
    assert call(ctx) == {"version": 1, "local_scope_missing": True}
    assert ctx.browser is None


@pytest.mark.parametrize(
    "patch",
    [
        {"proof": None},
        {"generation": "x" * 129},
        {"bridge_generations": ["g"] * 17},
        {"bridge_generations": [None]},
        {"lane_generations": []},
        {"session_id": "foreign"},
    ],
)
def test_malformed_local_record_is_read_only_no_delivery(
    tmp_path: Path, peer: dict[str, Any], patch: dict[str, Any]
) -> None:
    resource, ctx = setup_resource(tmp_path)
    resource.generation = ""
    resource.record.update(patch)
    resource._save()
    before = resource.path.read_bytes()
    assert call(ctx)["error_code"] == "ownership_inspection_invalid"
    assert peer["calls"] == []
    assert resource.path.read_bytes() == before


def test_oversized_record_and_wrong_host_do_not_deliver(
    tmp_path: Path, peer: dict[str, Any]
) -> None:
    resource, ctx = setup_resource(tmp_path)
    resource.path.write_text("x" * 65537)
    before = resource.path.read_bytes()
    assert call(ctx)["error_code"] == "ownership_inspection_invalid"
    assert resource.path.read_bytes() == before
    resource.record["host"] = "ui"
    resource._save()
    assert call(ctx)["error_code"] == "ownership_inspection_unsupported"
    assert peer["calls"] == []


def test_parallel_calls_preserve_local_storage(tmp_path: Path, peer: dict[str, Any]) -> None:
    resource, ctx = setup_resource(tmp_path)
    before = resource.path.read_bytes()

    async def parallel() -> None:
        replies = await asyncio.gather(
            *(
                builtin.execute_browser(str(i), {"action": "inspect_ownership"}, context=ctx)
                for i in range(8)
            )
        )
        assert [reply.details for reply in replies] == [peer["result"]] * 8

    asyncio.run(parallel())
    assert len(peer["calls"]) == 8
    assert resource.path.read_bytes() == before


def test_inspection_is_classified_read_before_harness_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    tool = builtin.build_browser_tool(None)
    assert tool is not None
    assert tool.call_approval_tier is not None
    assert tool.call_approval_tier({"action": " INSPECT_OWNERSHIP "}) == "read"
    assert tool.call_approval_tier({"action": "upload"}) == "exec"
    assert tool.call_approval_tier({"action": "open"}) == "write"


def test_wire_budget_nests_storage_and_outer_deadline() -> None:
    from local_operator.browser_bridge.protocol import COMMAND_TIMEOUTS, METHODS

    assert "owner_inspect" in METHODS
    assert 5 < COMMAND_TIMEOUTS["owner_inspect"] < 10


@pytest.mark.parametrize(
    "result",
    [None, [], {"version": True, "session_match": True}, {"version": 1, "session_match": "secret"}],
)
def test_malformed_peer_results_are_typed(
    tmp_path: Path, peer: dict[str, Any], result: Any
) -> None:
    resource, ctx = setup_resource(tmp_path)
    before = resource.path.read_bytes()
    peer["result"] = result
    # Transport rejects non-dict wire results first; a dict with the wrong
    # inspection schema reaches our narrower unsupported discriminator.
    assert call(ctx)["error_code"] in {
        "ownership_inspection_unsupported",
        "ownership_inspection_unavailable",
    }
    assert len(peer["calls"]) == 1
    assert resource.path.read_bytes() == before
