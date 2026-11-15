"""Cold-import capability shims exercise public approvals over authenticated HTTP.

These are simulated missing primitives on the test host, not native Windows
coverage. Discovery and the remote approval decision are synthetic; the public
tool, bridge client, authentication header and HTTP transport remain real.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

PROBE = r"""
import asyncio
import builtins
import importlib.abc
import importlib.util
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from local_operator.harness.types import BrowserSurface, ToolContext
from local_operator.tools import builtin
from local_operator.browser_bridge.backend import BridgeClient

mode = sys.argv[1]
assert "local_operator.browser_bridge.preview_origin" not in sys.modules
# A source override permits an identical before/after control without changing
# the worktree, dependencies, backend, or public execution path.
if len(sys.argv) > 2:
    class Source(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "local_operator.browser_bridge.preview_origin":
                return importlib.util.spec_from_loader(fullname, self)
        def create_module(self, spec):
            return None
        def exec_module(self, module):
            exec(compile(Path(sys.argv[2]).read_text(), sys.argv[2], "exec"), module.__dict__)
    sys.meta_path.insert(0, Source())
if mode == "fcntl":
    original_import = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == "fcntl":
            raise ModuleNotFoundError("simulated unavailable fcntl")
        return original_import(name, *args, **kwargs)
    builtins.__import__ = missing
elif mode == "dir_fd":
    os.supports_dir_fd = set()
elif mode == "nofollow":
    del os.O_NOFOLLOW
elif mode == "directory":
    del os.O_DIRECTORY
elif mode == "follow_symlinks":
    os.supports_follow_symlinks = set()

key = "synthetic-test-key-not-a-user-credential"
calls = []
state = "pending"
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def do_POST(self):
        assert self.path == "/rpc"
        assert self.headers.get("X-Bridge-Key") == key
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert request["method"] == "request_access"
        calls.append(request["params"]["url"])
        body = json.dumps({"id": request["id"], "ok": True, "result": {
            "state": state, "origin": request["params"]["url"]
        }}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever)
thread.start()
client = BridgeClient()
client._read = lambda: SimpleNamespace(port=server.server_port, session_key=key)
builtin.bridge_browser_reachable = AsyncMock(return_value=True)
builtin.ui_browser_reachable = AsyncMock(return_value=False)
# Only discovery is replaced: _bridge_call and BridgeClient.call run unchanged.
import local_operator.browser_bridge.backend as backend
backend.BridgeClient = lambda: client
root = Path(os.environ["HOME"]) / "sessions" / "synthetic-root"
root.mkdir(parents=True)
context = ToolContext(session_id="synthetic-root", session_dir=str(root), browser=BrowserSurface())
rows = []
async def request(url):
    before = len(calls)
    result = await builtin.execute_browser("synthetic-call", {
        "action": "request_access", "url": url, "backend": "extension"
    }, None, None, context)
    rows.append({"url": url, "backend_state": state, "error": result.is_error,
                 "details": result.details, "rpc_delta": len(calls)-before,
                 "text": result.text})
async def run():
    global state
    for state in ("pending", "allowed", "denied"):
        for url in ("https://example.com", "http://localhost:43871", "https://preview.example.com"):
            await request(url)
    state = "pending"
    for host in ("first", "first", "second"):
        await request("https://" + host + ".trycloudflare.com")
try:
    asyncio.run(run())
    print(json.dumps({"mode": mode, "rows": rows, "rpc_count": len(calls),
                      "pin_files": [p.name for p in root.iterdir()]}))
finally:
    server.shutdown()
    server.server_close()
    thread.join()
"""


@pytest.mark.parametrize(
    "mode", ["fcntl", "dir_fd", "nofollow", "directory", "follow_symlinks", "supported"]
)
def test_cold_import_public_approval_capabilities(tmp_path, mode):
    env = {"HOME": str(tmp_path), "LOCAL_OPERATOR_CONFIG_DIR": str(tmp_path / "config")}
    env["PATH"] = os.environ["PATH"]
    result = subprocess.run(
        [sys.executable, "-c", PROBE, mode],
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
        check=True,
    )
    report = json.loads(result.stdout)
    for row in report["rows"][:9]:
        assert row["rpc_delta"] == 1, row
        assert row["details"] is not None, row
        assert row["details"]["state"] == row["backend_state"], row
        assert "unexpectedly" not in row["text"], row
    quick = report["rows"][9:]
    if mode == "supported":
        assert [row["rpc_delta"] for row in quick] == [1, 1, 0]
        assert quick[-1]["details"]["code"] == "preview_origin_changed"
        assert report["pin_files"] == ["browser-preview-origin.json"]
    else:
        for row in quick:
            assert row["error"] and row["rpc_delta"] == 0, row
            assert row["details"]["code"] == "preview_origin_unavailable", row
            assert row["details"]["approval_request_sent"] is False, row
        assert report["pin_files"] == []
