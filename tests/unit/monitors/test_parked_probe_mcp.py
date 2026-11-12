"""The parked probe's MCP, web_fetch and glob checks (follow-up to session park).

Every MCP case runs a REAL stdio (or HTTP) MCP server — ``parked_mcp_stub.py``,
copied per test so ``pgrep -f`` can attribute its processes — through the
probe's throwaway ``McpManager``. The acceptance checks pinned here:

- quiet: counters advance, the server is invoked once, no process is left;
- change: engage with the counters byte-identical;
- golden parity: the session's ``_run_monitor_check`` and the probe hash the
  same stub output identically, under and over the 8 KiB spill threshold;
- hint dropped: engage, the tool is never invoked;
- unreachable / re-auth: the session's unavailable tick is committed, no engage,
  no browser, no listening socket; a stall notice owed (−31 min) engages;
- passphrase-tier ``${REF}`` server: session-only (probe engages, pressure park
  skips).
"""

from __future__ import annotations

import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.monitors import parked
from local_operator.monitors import state as monitor_state
from local_operator.monitors import store as monitor_store
from local_operator.monitors.diff import content_hash
from local_operator.monitors.scheduler import (
    UNAVAILABLE_GONE_MS,
    UNAVAILABLE_STALL_MS,
    apply_unavailable_tick,
    normalize_result,
)
from local_operator.monitors.settings import read_monitor_settings
from local_operator.monitors.spec import MonitorSpec

SID = "probemcp0001"
NOW = 1_800_000_000_000
TOOL = "mcp__stub_read_thread"
STUB = Path(__file__).with_name("parked_mcp_stub.py")


def _row(tool: str, arguments: dict[str, Any], cwd: str) -> dict[str, Any]:
    return {
        "id": "m1",
        "name": "watch",
        "tool": tool,
        "arguments": arguments,
        "every_ms": 60_000,
        "until_at": None,
        "cwd": cwd,
        "created_at": 1,
        "next_due_at": NOW - 1,
        "disabled": False,
    }


def _spec(row: dict[str, Any]) -> MonitorSpec:
    return MonitorSpec.model_validate({k: row[k] for k in MonitorSpec.model_fields if k in row})


def _arm(root: Path, row: dict[str, Any], cwd: Path) -> None:
    monitor_store.write_entry(
        root, SID, cwd=str(cwd), monitors=[row], preserve={"parked_at": NOW - 10}
    )


def _counters(content_hash_: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema": 1,
        "monitor_id": "m1",
        "next_due_at": NOW - 1,
        "content_hash": content_hash_,
        "checks": 4,
        "last_check_at": NOW - 60_000,
        **extra,
    }


def _calls(stub: Path) -> int:
    path = stub / "calls"
    return len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0


def _stub_processes(script: Path) -> list[str]:
    found = subprocess.run(["pgrep", "-f", str(script)], capture_output=True, text=True)
    return found.stdout.split()


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    root = tmp_path / "cfg"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    cwd = tmp_path / "proj"
    cwd.mkdir()
    stub = tmp_path / "stub"
    stub.mkdir()
    script = tmp_path / "stub_server.py"
    shutil.copyfile(STUB, script)
    (stub / "thread.txt").write_text("thread: 3 replies\n", encoding="utf-8")
    server = {"command": sys.executable, "args": [str(script)], "env": {"STUB_DIR": str(stub)}}
    (cwd / ".mcp.json").write_text(json.dumps({"mcpServers": {"stub": server}}), encoding="utf-8")
    return SimpleNamespace(root=root, cwd=cwd, stub=stub, script=script, server=server)


def _write_servers(rig: SimpleNamespace, servers: dict[str, Any]) -> None:
    (rig.cwd / ".mcp.json").write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")


async def _session_hash(rig: SimpleNamespace, row: dict[str, Any]) -> str:
    """The hash the IN-SESSION scheduler stores for this row: ``Session._run_monitor_check``
    over a live MCP tool, then the scheduler's ``normalize_result`` + ``content_hash``."""
    from local_operator.harness.types import StreamEndEvent
    from local_operator.mcp.manager import McpManager
    from tests.unit.session.test_session import ScriptedStream, make_session

    manager = McpManager(str(rig.cwd), None, secret_base=rig.root)
    try:
        await manager.connect_configured_server("stub", interactive=False)
        tool = next(t for t in manager.get_tools() if t.name == row["tool"])
        session = make_session(
            rig.cwd, ScriptedStream([[StreamEndEvent(stop_reason="stop")]]), tools=[tool]
        )
        try:
            outcome = await session._run_monitor_check(_spec(row))
        finally:
            await session.dispose()
    finally:
        await manager.disconnect_all(in_task=True)
    assert outcome.get("error") is None, outcome
    return content_hash(
        normalize_result(str(outcome.get("text")), _spec(row), read_monitor_settings())
    )


# --- the shared predicate ---------------------------------------------------


def test_probeable_is_one_predicate_for_probe_and_pressure_park() -> None:
    def row(tool: str) -> dict[str, Any]:
        return {"tool": tool}

    for tool in ("bash", "read", "grep", "glob", "web_fetch"):
        assert parked.probeable(row(tool), "passphrase"), tool
    assert parked.probeable(row("code_requests"), "keyfile")
    assert not parked.probeable(row("code_requests"), "passphrase")
    assert not parked.probeable(row("web_search"), "keyfile")
    assert not parked.probeable(row("eval"), "keyfile")
    # MCP: tier-only when the servers were not classified ...
    assert parked.probeable(row(TOOL), "keyfile")
    assert not parked.probeable(row(TOOL), "passphrase")
    # ... and by server once they were.
    assert parked.probeable(row(TOOL), "passphrase", session_only_servers=set())
    assert not parked.probeable(row(TOOL), "keyfile", session_only_servers={"stub"})


def test_apply_unavailable_tick_is_the_schedulers_ladder() -> None:
    spec = _spec(_row("read", {"path": "x"}, "."))
    counters: dict[str, Any] = {"content_hash": "h", "checks": 4}
    assert apply_unavailable_tick(counters, spec, "down", NOW) is None
    assert counters["unavailable_since"] == NOW and counters["unavailable_ticks"] == 1
    assert counters["next_due_at"] == NOW + 60_000 and counters["checks"] == 4
    assert apply_unavailable_tick(counters, spec, "down", NOW + 60_000) is None
    assert counters["next_due_at"] == NOW + 60_000 + 120_000
    assert apply_unavailable_tick(counters, spec, "down", NOW + UNAVAILABLE_STALL_MS) == "stalled"
    counters["unavailable_notified"] = True
    assert apply_unavailable_tick(counters, spec, "down", NOW + UNAVAILABLE_STALL_MS) is None
    assert apply_unavailable_tick(counters, spec, "down", NOW + UNAVAILABLE_GONE_MS) == "gone"


# --- glob / web_fetch -------------------------------------------------------


@pytest.mark.asyncio
async def test_a_glob_watch_is_probed_and_stays_parked(rig: SimpleNamespace) -> None:
    (rig.cwd / "a.txt").write_text("a", encoding="utf-8")
    row = _row("glob", {"pattern": "*.txt", "path": str(rig.cwd)}, str(rig.cwd))
    _arm(rig.root, row, rig.cwd)
    env = parked._ProbeEnv(rig.root, SID, str(rig.cwd), tier="keyfile")
    tool = (await env.resolve(_spec(row))).tool
    from local_operator.harness.types import AbortSignal
    from local_operator.session.session import _monitor_comparable_text

    result = await tool.execute(
        "t", dict(row["arguments"]), AbortSignal(), None, env.context(_spec(row))
    )
    text, _b, _h = _monitor_comparable_text(result.text, result.details)
    baseline = content_hash(normalize_result(str(text), _spec(row), read_monitor_settings()))
    monitor_state.write_counters(rig.root, SID, "m1", _counters(baseline))
    outcome = await parked.probe_session(rig.root, SID, now_ms=NOW, tier="keyfile")
    assert outcome == {"verdict": "quiet", "why": "unchanged", "checked": 1}
    (rig.cwd / "b.txt").write_text("b", encoding="utf-8")
    advanced = monitor_state.read_counters(rig.root, SID, "m1")
    assert advanced is not None
    later = advanced["next_due_at"] + 1
    changed = await parked.probe_session(rig.root, SID, now_ms=later, tier="keyfile")
    assert changed["verdict"] == "engage" and changed["why"] == "m1: changed"


@pytest.mark.asyncio
async def test_a_web_fetch_watch_runs_through_the_registry_factory_with_session_settings(
    rig: SimpleNamespace,
) -> None:
    from local_operator.config import ConfigManager
    from local_operator.web_fetch.service import set_allow_private

    body = {"text": "status: green\n"}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            payload = body["text"].encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/status"
        row = _row("web_fetch", {"url": url}, str(rig.cwd))
        _arm(rig.root, row, rig.cwd)
        env = parked._ProbeEnv(rig.root, SID, str(rig.cwd), tier="keyfile")
        # Negative control: the session's default settings refuse loopback, so the
        # probe's answer must come from the CONFIGURED settings, not a default.
        monitor_state.write_counters(rig.root, SID, "m1", _counters("seed"))
        refused = await parked._check_one(rig.root, SID, row, read_monitor_settings(), NOW, env)
        assert refused == ("engage", "check returned an error"), refused
        set_allow_private(ConfigManager(config_dir=rig.root), True)
        env = parked._ProbeEnv(rig.root, SID, str(rig.cwd), tier="keyfile")
        tool = (await env.resolve(_spec(row))).tool
        from local_operator.harness.types import AbortSignal
        from local_operator.session.session import _monitor_comparable_text

        # Twice: the result's header reads "cache miss" on the first fetch and
        # "cache hit" after, in-session too; the baseline is the steady state.
        result: Any = None
        for _ in range(2):
            result = await tool.execute(
                "t", {"url": url}, AbortSignal(), None, env.context(_spec(row))
            )
        assert not result.is_error, result.text
        assert "cache hit" in result.text
        text, _b, _h = _monitor_comparable_text(result.text, result.details)
        baseline = content_hash(normalize_result(str(text), _spec(row), read_monitor_settings()))
        monitor_state.write_counters(rig.root, SID, "m1", _counters(baseline))
        quiet = await parked.probe_session(rig.root, SID, now_ms=NOW, tier="keyfile")
        assert quiet["verdict"] == "quiet", quiet
    finally:
        httpd.shutdown()
        httpd.server_close()


# --- MCP over stdio ---------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ["small", "spilled"])
async def test_mcp_quiet_tick_matches_the_session_hash_and_a_change_engages(
    rig: SimpleNamespace, size: str
) -> None:
    if size == "spilled":
        lines = [f"reply {n}: " + "x" * 120 for n in range(150)]
        (rig.stub / "thread.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert (rig.stub / "thread.txt").stat().st_size > 8 * 1024
    row = _row(TOOL, {"thread": "C1/1"}, str(rig.cwd))
    _arm(rig.root, row, rig.cwd)
    baseline = await _session_hash(rig, row)
    monitor_state.write_counters(rig.root, SID, "m1", _counters(baseline))
    before = _calls(rig.stub)

    quiet = await parked.probe_session(rig.root, SID, now_ms=NOW, tier="keyfile")
    # GOLDEN PARITY: quiet is only possible when the probe's hash equals the
    # session's byte for byte.
    assert quiet == {"verdict": "quiet", "why": "unchanged", "checked": 1}
    assert _calls(rig.stub) == before + 1
    after = monitor_state.read_counters(rig.root, SID, "m1")
    assert after is not None and after["checks"] == 5 and after["last_check_at"] == NOW
    assert _stub_processes(rig.script) == []

    with (rig.stub / "thread.txt").open("a", encoding="utf-8") as handle:
        handle.write("a new reply\n")
    later = after["next_due_at"] + 1
    changed = await parked.probe_session(rig.root, SID, now_ms=later, tier="keyfile")
    assert changed["verdict"] == "engage" and changed["why"] == "m1: changed"
    assert monitor_state.read_counters(rig.root, SID, "m1") == after
    assert not monitor_state.snapshot_exists(rig.root, SID, "m1")
    assert _stub_processes(rig.script) == []


@pytest.mark.asyncio
async def test_a_dropped_read_only_hint_engages_without_calling_the_tool(
    rig: SimpleNamespace,
) -> None:
    row = _row(TOOL, {"thread": "C1/1"}, str(rig.cwd))
    _arm(rig.root, row, rig.cwd)
    monitor_state.write_counters(rig.root, SID, "m1", _counters(await _session_hash(rig, row)))
    (rig.stub / "no_hint").touch()  # the server's NEXT tools/list has no hint
    before = _calls(rig.stub)
    snapshot = monitor_state.read_counters(rig.root, SID, "m1")
    outcome = await parked.probe_session(rig.root, SID, now_ms=NOW, tier="keyfile")
    assert outcome["verdict"] == "engage" and "no longer read-only" in outcome["why"]
    assert _calls(rig.stub) == before
    assert monitor_state.read_counters(rig.root, SID, "m1") == snapshot


@pytest.mark.asyncio
async def test_an_unreachable_server_records_the_unavailable_tick_and_stays_parked(
    rig: SimpleNamespace,
) -> None:
    _write_servers(rig, {"stub": {"command": str(rig.stub / "no-such-server"), "args": []}})
    row = _row(TOOL, {"thread": "C1/1"}, str(rig.cwd))
    _arm(rig.root, row, rig.cwd)
    monitor_state.write_counters(rig.root, SID, "m1", _counters("baseline"))
    outcome = await parked.probe_session(rig.root, SID, now_ms=NOW, tier="keyfile")
    assert outcome["verdict"] == "quiet" and outcome["unavailable"] == 1, outcome
    after = monitor_state.read_counters(rig.root, SID, "m1")
    assert after is not None
    assert after["unavailable_since"] == NOW and after["unavailable_ticks"] == 1
    assert after["checks"] == 4 and after["content_hash"] == "baseline"  # no strike, no diff
    assert after["next_due_at"] == NOW + 60_000
    assert not after.get("consecutive_failures")

    # The episode is 31 minutes old: the stall notice is owed, so the session is
    # reopened to send it — and the counters are left for its scheduler.
    stale = {**after, "unavailable_since": NOW - 31 * 60_000, "next_due_at": NOW - 1}
    monitor_state.write_counters(rig.root, SID, "m1", stale)
    owed = await parked.probe_session(rig.root, SID, now_ms=NOW, tier="keyfile")
    assert owed["verdict"] == "engage" and "stalled notice owed" in owed["why"]
    assert monitor_state.read_counters(rig.root, SID, "m1") == stale


@pytest.mark.asyncio
async def test_a_reachable_server_inside_an_outage_engages_for_the_restored_notice(
    rig: SimpleNamespace,
) -> None:
    row = _row(TOOL, {"thread": "C1/1"}, str(rig.cwd))
    _arm(rig.root, row, rig.cwd)
    baseline = await _session_hash(rig, row)
    counters = _counters(baseline, unavailable_since=NOW - 120_000, unavailable_ticks=2)
    monitor_state.write_counters(rig.root, SID, "m1", counters)
    outcome = await parked.probe_session(rig.root, SID, now_ms=NOW, tier="keyfile")
    assert outcome["verdict"] == "engage" and "restored" in outcome["why"]
    assert monitor_state.read_counters(rig.root, SID, "m1") == counters


@pytest.mark.asyncio
async def test_a_401_is_an_unavailable_tick_with_no_browser_and_no_callback_listener(
    rig: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Deny(http.server.BaseHTTPRequestHandler):
        def _deny(self) -> None:
            self.send_response(401)
            self.send_header("WWW-Authenticate", "Bearer")
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_GET = do_POST = do_DELETE = _deny  # noqa: N815

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Deny)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    browsers: list[str] = []
    listens: list[Any] = []

    async def no_browser(url: str) -> bool:
        browsers.append(url)
        return False

    real_listen = socket.socket.listen

    def spy_listen(self: socket.socket, *args: Any) -> None:
        listens.append(self.getsockname())
        real_listen(self, *args)

    monkeypatch.setattr("local_operator.mcp.auth.open_browser_quietly", no_browser)
    monkeypatch.setattr(socket.socket, "listen", spy_listen)
    monkeypatch.setattr("webbrowser.open", lambda *a, **k: browsers.append(str(a)) or False)
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/mcp"
        _write_servers(rig, {"stub": {"type": "http", "url": url}})
        row = _row(TOOL, {"thread": "C1/1"}, str(rig.cwd))
        _arm(rig.root, row, rig.cwd)
        monitor_state.write_counters(rig.root, SID, "m1", _counters("baseline"))
        outcome = await parked.probe_session(rig.root, SID, now_ms=NOW, tier="keyfile")
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert outcome["verdict"] == "quiet" and outcome.get("unavailable") == 1, outcome
    after = monitor_state.read_counters(rig.root, SID, "m1")
    assert after is not None and after["unavailable_ticks"] == 1
    assert after["checks"] == 4
    # The session's own re-auth sentence, so a later stall notice names the remedy.
    assert "needs re-authentication" in after["last_error"], after["last_error"]
    assert browsers == [] and listens == []


# --- credential tiers -------------------------------------------------------


@pytest.mark.asyncio
async def test_a_passphrase_tier_ref_server_is_session_only(rig: SimpleNamespace) -> None:
    server = {**rig.server, "env": {**rig.server["env"], "STUB_KEY": "${STUB_TOKEN}"}}
    _write_servers(rig, {"stub": server, "plain": rig.server})
    assert parked.session_only_servers(rig.root, str(rig.cwd), "passphrase") == {"stub"}
    assert parked.session_only_servers(rig.root, str(rig.cwd), "keyfile") == set()
    row = _row(TOOL, {"thread": "C1/1"}, str(rig.cwd))
    _arm(rig.root, row, rig.cwd)
    monitor_state.write_counters(rig.root, SID, "m1", _counters("baseline"))
    outcome = await parked.probe_session(rig.root, SID, now_ms=NOW, tier="passphrase")
    assert outcome["verdict"] == "engage" and "session-only" in outcome["why"]
    assert _calls(rig.stub) == 0 and _stub_processes(rig.script) == []


def test_pressure_park_skips_exactly_what_the_probe_cannot_run(
    rig: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.session.runtime import park_pressure

    rows = [
        _row(TOOL, {"thread": "C1/1"}, str(rig.cwd)),
        {**_row("glob", {"pattern": "*"}, str(rig.cwd)), "id": "m2"},
        {**_row("web_search", {"query": "x"}, str(rig.cwd)), "id": "m3"},
    ]
    monitor_store.write_entry(rig.root, SID, cwd=str(rig.cwd), monitors=rows)
    monkeypatch.setattr("local_operator.secrets.keys.key_mode", lambda *_a: "keyfile")
    assert park_pressure._unprobeable_monitors(rig.root, SID) == ["m3:web_search"]
    server = {**rig.server, "env": {"STUB_KEY": "${STUB_TOKEN}"}}
    _write_servers(rig, {"stub": server})
    monkeypatch.setattr("local_operator.secrets.keys.key_mode", lambda *_a: "passphrase")
    assert park_pressure._unprobeable_monitors(rig.root, SID) == [
        f"m1:{TOOL}",
        "m3:web_search",
    ]


# --- process hygiene ---------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
async def test_detach_stdio_false_keeps_stdio_servers_in_the_owner_group(
    rig: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Linux's default setsid would carry a stdio server out of the probe's group.

    Forced here on any POSIX host by making the platform rule answer "detach":
    the default manager then puts the child in its own group, and the probe's
    ``detach_stdio=False`` keeps it in ours.
    """
    from local_operator.mcp import manager as manager_mod

    monkeypatch.setattr(manager_mod, "stdio_start_new_session", lambda: True)

    async def child_pgid(detach: bool | None) -> int:
        mgr = manager_mod.McpManager(str(rig.cwd), None, secret_base=rig.root, detach_stdio=detach)
        try:
            await mgr.connect_configured_server("stub", interactive=False)
            pids = _stub_processes(rig.script)
            assert len(pids) == 1, pids
            return os.getpgid(int(pids[0]))
        finally:
            await mgr.disconnect_all(in_task=True)

    assert await child_pgid(None) != os.getpgrp()
    assert await child_pgid(False) == os.getpgrp()
    assert _stub_processes(rig.script) == []
