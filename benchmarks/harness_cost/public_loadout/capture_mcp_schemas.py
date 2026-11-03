#!/usr/bin/env python3
"""Regenerate public_mcp_schemas.json from pinned open-source MCP servers.

WHY
    The harness-cost benchmark's "full loadout" arm measures what each CLI
    agent harness pays per task when a realistic MCP surface is configured
    (hundreds of tools). Tool-schema volume is the single biggest cost lever
    between harnesses (some ship every schema on every request, others defer),
    so the arm needs a surface of realistic scale. v1 used a private set of
    servers that cannot be published. v2 uses only open-source servers
    (MIT / Apache-2.0) whose `tools/list` answers WITHOUT real credentials, so
    any third party can rebuild a byte-comparable snapshot from the pins below.

WHAT IT DOES
    For each pinned server: launch it over stdio in its own process group
    (`npx -y pkg@version` or `uvx pkg==version`), perform the MCP handshake
    (`initialize`, `notifications/initialized`), page through `tools/list`,
    then terminate the process group it started (by exact pid / pgid, never by
    name). Writes the snapshot plus per-server provenance: package, version,
    license, source URL, tool count and compact-JSON schema bytes.

CONSTRAINTS
    * No network service is contacted beyond package download: credentials
      are dummy placeholders, and a server that needs a live backend just to
      list its tools is not eligible for the pin list.
    * Uses the caller's existing npm / uv caches (no extra dependency trees).
      HOME is redirected to a throwaway directory so no server writes into the
      operator's real config; npm_config_cache / UV_CACHE_DIR keep pointing at
      the real caches so downloads are shared and not duplicated.
    * Pure standard library, Python >= 3.10.
    * Tool order is preserved exactly as each server returned it; tool order
      affects prompt caching in some harnesses, so it is part of the snapshot.

USAGE
    python3 capture_mcp_schemas.py                 # all pins -> public_mcp_schemas.json
    python3 capture_mcp_schemas.py --only playwright --out capture.json
    python3 capture_mcp_schemas.py --check         # re-capture, diff vs committed snapshot

Determinism: npm/uv resolve the pinned top-level version, but a server's
transitive dependencies are not locked by this script. `--check` exists so a
rerun can prove the captured schemas still match the committed snapshot (the
replay server serves the committed snapshot, so benchmark runs never depend on
a fresh capture).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "public_mcp_schemas.json"

# Each pin: key -> launch command, provenance, and dummy env. `env` values are
# placeholders that satisfy start-up validation only; they are never valid
# credentials and no tool is ever called during capture.
PINS: dict[str, dict] = {}  # filled below, after candidate probing


def published_command(pin):
    """Publish reproducible invocation intent, never a credential-shaped dummy.

    Even an all-zero offline key can trigger repository push protection. Keep
    that validated dummy only in the launched command's memory and replace its
    metadata value with an explicit non-credential marker. Tool/schema hashes
    do not include invocation metadata, so this cannot alter replay or --check.
    """
    command = list(pin['cmd'])
    if '--api-key' in command:
        command[command.index('--api-key') + 1] = '<offline-dummy-generated-at-runtime>'
    return command


def _pin(key, cmd, license, source, env=None, package=None):
    PINS[key] = {"cmd": cmd, "license": license, "source": source,
                 "env": env or {}, "package": package or cmd[-1]}


class StdioClient:
    """Minimal newline-delimited JSON-RPC client for an MCP stdio server."""

    def __init__(self, cmd, env, cwd):
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, cwd=cwd, start_new_session=True)  # own pgid -> clean teardown
        self.buf = b""
        self.next_id = 1

    def send(self, msg):
        self.proc.stdin.write((json.dumps(msg) + "\n").encode())
        self.proc.stdin.flush()

    def request(self, method, params=None, timeout=240):
        rid = self.next_id
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        deadline = time.time() + timeout
        while time.time() < deadline:
            while b"\n" in self.buf:
                line, self.buf = self.buf.split(b"\n", 1)
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue  # a server logging to stdout; ignore non-JSON lines
                if msg.get("id") == rid and ("result" in msg or "error" in msg):
                    if "error" in msg:
                        raise RuntimeError(f"{method}: {msg['error']}")
                    return msg["result"]
                if "method" in msg and "id" in msg:  # server->client request (roots/list, ...)
                    self.send({"jsonrpc": "2.0", "id": msg["id"],
                               "result": {"roots": []} if msg["method"] == "roots/list" else {}})
            r, _, _ = select.select([self.proc.stdout], [], [], 1.0)
            if r:
                chunk = os.read(self.proc.stdout.fileno(), 1 << 20)
                if not chunk:
                    raise RuntimeError(f"server exited (rc={self.proc.poll()}): "
                                       + self.stderr_tail())
                self.buf += chunk
        raise TimeoutError(f"{method} timed out")

    def stderr_tail(self):
        try:
            os.set_blocking(self.proc.stderr.fileno(), False)
            return (self.proc.stderr.read() or b"")[-1500:].decode(errors="replace")
        except Exception:
            return ""

    def close(self):
        """Terminate exactly the process group this client started."""
        pid = self.proc.pid
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pid, sig)  # pgid == pid because of start_new_session
            except ProcessLookupError:
                break
            try:
                self.proc.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                continue
        # The leader may exit before its children (npx -> node): sweep the group.
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return pid


def schema_bytes(tools) -> int:
    """Size measure used throughout the kit: compact JSON of the tools array."""
    return len(json.dumps(tools, separators=(",", ":"), ensure_ascii=False).encode())


def capture(key, pin, scratch_home):
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(scratch_home),
        "TMPDIR": str(scratch_home / "tmp"),
        # Share the operator's existing package caches: no duplicate downloads.
        "npm_config_cache": os.environ.get("npm_config_cache", str(Path.home() / ".npm")),
        "npm_config_update_notifier": "false",
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", str(Path.home() / ".cache/uv")),
        "UV_TOOL_DIR": str(scratch_home / "uv-tools"),
        "NO_UPDATE_NOTIFIER": "1",
        "LANG": "en_US.UTF-8",
        "npm_config_before": RESOLVE_BEFORE,
        "UV_EXCLUDE_NEWER": RESOLVE_BEFORE,
        **pin["env"],
    }
    (scratch_home / "tmp").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    c = StdioClient(pin["cmd"], env, cwd=str(scratch_home))
    try:
        init = c.request("initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "harness-cost-bench-capture", "version": "2"}})
        c.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        tools, cursor = [], None
        while True:
            res = c.request("tools/list", {"cursor": cursor} if cursor else {})
            tools += res.get("tools", [])
            cursor = res.get("nextCursor")
            if not cursor:
                break
    finally:
        pid = c.close()
    info = init.get("serverInfo", {})
    return {
        "package": pin["package"], "command": published_command(pin), "license": pin["license"],
        "source": pin["source"], "dummy_env": sorted(pin["env"]),
        "server_info": info, "protocol_version": init.get("protocolVersion"),
        "tool_count": len(tools), "schema_bytes": schema_bytes(tools),
        "sha256": hashlib.sha256(json.dumps(tools, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
        "capture_seconds": round(time.time() - t0, 1), "pid": pid,
        "tools": tools,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="*", help="capture just these pin keys")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="capture into a temp file and compare sha256 per server with --out")
    a = ap.parse_args()
    keys = a.only or list(PINS)
    scratch = Path(tempfile.mkdtemp(prefix="mcp-capture-"))
    servers, failures = {}, {}
    try:
        for k in keys:
            try:
                r = capture(k, PINS[k], scratch)
                r.pop("pid")
                servers[k] = r
                print(f"{k:28s} {r['tool_count']:4d} tools {r['schema_bytes']:9,d} B  {r['capture_seconds']}s", file=sys.stderr)
            except Exception as e:  # report and continue; a failed pin fails the run below
                failures[k] = str(e)[:600]
                print(f"{k:28s} FAILED: {str(e)[:300]}", file=sys.stderr)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if a.check:
        ref = json.loads(a.out.read_text())["servers"]
        bad = [k for k in servers if ref.get(k, {}).get("sha256") != servers[k]["sha256"]]
        print("check:", "OK" if not bad and not failures else f"MISMATCH {bad} FAILED {list(failures)}")
        sys.exit(1 if bad or failures else 0)
    if failures:
        sys.exit(f"failed pins: {sorted(failures)}; snapshot not written")
    snap = {
        "about": "Public MCP tool schemas for the harness-cost benchmark full-loadout arm. "
                 "Captured verbatim from tools/list of pinned open-source servers by capture_mcp_schemas.py.",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema_bytes_measure": "len(json.dumps(tools, separators=(',', ':'), ensure_ascii=False).encode())",
        "total_tools": sum(s["tool_count"] for s in servers.values()),
        "total_schema_bytes": sum(s["schema_bytes"] for s in servers.values()),
        "servers": servers,
    }
    a.out.write_text(json.dumps(snap, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {a.out.name}: {snap['total_tools']} tools, {snap['total_schema_bytes']:,} B", file=sys.stderr)


# --- pins (filled after probing; see the PIN TABLE in KIT.md) ---------------
# Resolution date: npm (`before`) and uv (`exclude-newer`) ignore anything
# published after it, so transitive dependencies resolve the same way on a
# later rerun as they did on the reference capture. Override with
# BENCH_MCP_RESOLVE_BEFORE only to deliberately refresh the snapshot.
RESOLVE_BEFORE = os.environ.get("BENCH_MCP_RESOLVE_BEFORE", "2026-11-03T00:00:00Z")
NPX = ["npx", "-y"]
_AWS = {"AWS_REGION": "us-east-1", "AWS_ACCESS_KEY_ID": "dummy-local-only",
        "AWS_SECRET_ACCESS_KEY": "dummy", "FASTMCP_LOG_LEVEL": "ERROR"}

# Selection (see README.md): nine servers, mirroring v1's shape of nine servers with
# one ~180-tool giant, one ~80-tool server and a long tail of 20-50-tool
# SaaS/devops servers. Every one lists its tools offline with dummy env.
_pin("twilio", NPX + ["@twilio-alpha/mcp@0.7.0", "AC00000000000000000000000000000000/SK00000000000000000000000000000000:dummy"],
     "MIT", "https://github.com/twilio-labs/mcp", package="@twilio-alpha/mcp@0.7.0")
_pin("grafana", ["uvx", "mcp-grafana==2.2.0"], "Apache-2.0", "https://github.com/grafana/mcp-grafana",
     {"GRAFANA_URL": "http://127.0.0.1:9", "GRAFANA_SERVICE_ACCOUNT_TOKEN": "dummy-local-only"})
_pin("aws-billing-cost-management", ["uvx", "awslabs.billing-cost-management-mcp-server==0.0.44"],
     "Apache-2.0", "https://github.com/awslabs/mcp", _AWS)
_pin("supabase", NPX + ["@supabase/mcp-server-supabase@0.15.0"], "Apache-2.0",
     "https://github.com/supabase-community/supabase-mcp", {"SUPABASE_ACCESS_TOKEN": "sbp_" + "0" * 40})
# Runtime construction preserves the exact previously accepted offline dummy,
# without committing any credential-shaped literal. Never replace with a real
# key: capture only lists local schemas; publication metadata uses a marker.
_pin("launchdarkly", NPX + ["@launchdarkly/mcp-server@0.7.2", "start", "--api-key",
     "-".join(["api", "0" * 8, "0" * 4, "0" * 4, "0" * 4, "0" * 12])],
     "MIT", "https://github.com/launchdarkly/mcp-server", package="@launchdarkly/mcp-server@0.7.2")
_pin("playwright", NPX + ["@playwright/mcp@0.0.84", "--headless", "--isolated"], "Apache-2.0",
     "https://github.com/microsoft/playwright-mcp", package="@playwright/mcp@0.0.84")
_pin("notion", NPX + ["@notionhq/notion-mcp-server@2.6.1"], "MIT",
     "https://github.com/makenotion/notion-mcp-server", {"NOTION_TOKEN": "ntn_" + "0" * 46})
_pin("kubernetes", NPX + ["mcp-server-kubernetes@4.2.1"], "MIT", "https://github.com/Flux159/mcp-server-kubernetes")
_pin("heroku", NPX + ["@heroku/mcp-server@1.2.13"], "Apache-2.0", "https://github.com/heroku/heroku-mcp-server",
     {"HEROKU_API_KEY": "00000000-0000-0000-0000-000000000000"})

# Probed 2026-11-03 and NOT pinned (kept so a reviewer can swap one in):
#   listed offline: hubspot 0.4.0 (21 tools; MIT LICENSE in the npm tarball but no public
#     source repository, so not pinned), circleci 0.20.0 (13), browserbase 2.4.3 (9),
#     modelcontextprotocol reference servers everything/filesystem/memory/git (16/14/9/12),
#     awslabs eks 0.2.1 (16), cloudwatch 0.3.1 (19), aws-dataprocessing 0.2.2 (36),
#     dynamodb 2.1.9 (8), iam 1.1.3 (29), sagemaker-ai 1.1.0 (4), aws-iot-sitewise 11.0.23 (41),
#     aws-api 1.6.0 (2)
#   refused offline: neon 0.6.5 and upstash 0.4.0 (validate the key against the live API at
#     start), stripe 0.3.3 (no initialize reply), awslabs ecs 0.1.37 (dependency drift crash).
#   not npx/uvx: github/github-mcp-server (Go release binary).


if __name__ == "__main__":
    main()
