"""End-to-end evidence for live settings on the watcher poll.

Two REAL runtime processes (``python -m local_operator.session.runtime.process``,
the exact child ``lop`` spawns) from the worktree venv share ONE isolated config
dir. This driver attaches to each over its authenticated loopback socket with
the production ``RemoteSession`` client (what every ``lop`` TUI is), and a
THIRD process edits config.yml with the worktree's ``local-operator config
edit`` CLI. Each cell reports the wall time from the CLI write to the observed
behaviour change, with no ``/new`` and no relaunch.

Run:
  env HOME=/tmp/iso-live LOCAL_OPERATOR_CONFIG_DIR=/tmp/iso-live/.local-operator \
      /tmp/lop-live-settings/.venv/bin/python /tmp/lop-live-settings-e2e.py
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, "/tmp/lop-live-settings")

CONFIG_DIR = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
LOP = "/tmp/lop-live-settings/.venv/bin/local-operator"
PY = "/tmp/lop-live-settings/.venv/bin/python"
REPORT: list[str] = []


def log(line: str) -> None:
    print(line, flush=True)
    REPORT.append(line)


def cli_edit(key: str, value: str) -> float:
    """The third shell: `lop config edit <key> <value>`. Returns the write time."""
    proc = subprocess.run(
        [LOP, "config", "edit", key, value], capture_output=True, text=True, timeout=60
    )
    stamp = time.monotonic()
    log(f"  $ lop config edit {key} {value}\n    -> rc={proc.returncode} {proc.stdout.strip()!r}")
    assert proc.returncode == 0, proc.stderr
    return stamp


async def wait_until(pred, timeout: float = 8.0, label: str = "") -> float:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return time.monotonic() - t0
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {label}")


async def spawn_runtime(session_id: str, cwd: Path) -> tuple[subprocess.Popen[bytes], Any]:
    from local_operator.session.runtime import registry

    env = dict(os.environ)
    env["LOP_MOBILE_CHILD_CWD"] = str(cwd)
    env["LOP_MOBILE_CHILD_RESUME"] = session_id
    proc = subprocess.Popen(
        [PY, "-m", "local_operator.session.runtime.process"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=open(f"/tmp/iso-live/{session_id}.stderr", "wb"),
        start_new_session=True,
    )
    record = None

    def _found() -> bool:
        nonlocal record
        for rec, state in registry.scan(CONFIG_DIR):
            if getattr(rec, "session_id", "") == session_id and state == "live":
                record = rec
                return True
        return False

    await wait_until(_found, timeout=30, label=f"runtime record for {session_id}")
    return proc, record


async def attach(record: Any, session_id: str):
    from local_operator.session.remote import RemoteSession

    async def never():
        raise AssertionError("viewer must not take over")

    viewer = await RemoteSession.connect(
        record, session_id, config_dir=CONFIG_DIR, takeover_factory=never
    )
    return viewer


class Notices:
    def __init__(self, viewer: Any, name: str) -> None:
        self.items: list[tuple[str, str]] = []
        self.name = name
        viewer.subscribe(self._on)

    def _on(self, event: Any) -> None:
        if getattr(event, "type", "") == "notice":
            self.items.append((getattr(event, "kind", ""), getattr(event, "text", "")))
            log(f"  [{self.name} notice/{event.kind}] {event.text}")

    def has(self, needle: str) -> bool:
        return any(needle in text for _kind, text in self.items)


async def slash(viewer: Any, command: str, args: str = "") -> str:
    result = await viewer.route_shared_slash(command, args)
    if isinstance(result, dict):
        text = result.get("text") or str(result.get("data", ""))
    else:
        text = getattr(result, "text", "") or str(getattr(result, "data", ""))
    log(f"  [{viewer._label}] /{command} {args} -> {text}")
    return text


async def main() -> None:
    from local_operator.config import ConfigManager

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    manager = ConfigManager(CONFIG_DIR)
    manager.set_config_value("hosting", "test")
    manager.set_config_value("model_name", "m-boot")
    manager.set_config_value("tool_approval_mode", "ask")
    ws = Path("/tmp/iso-live/ws")
    ws.mkdir(parents=True, exist_ok=True)

    log("== boot: two runtimes (A, B) from the worktree binary, one shared config dir ==")
    proc_a, rec_a = await spawn_runtime("livesessa0001", ws)
    proc_b, rec_b = await spawn_runtime("livesessb0001", ws)
    a = await attach(rec_a, "livesessa0001")
    b = await attach(rec_b, "livesessb0001")
    a._label, b._label = "A", "B"
    na, nb = Notices(a, "A"), Notices(b, "B")
    log(f"  A pid={rec_a.pid} model={a.model_label}   B pid={rec_b.pid} model={b.model_label}")
    try:
        # ---------------- approvals ----------------
        log("\n== cell 1: tool_approval_mode ask -> auto (both runtimes, no /new) ==")
        log(f"  before: A {await slash(a, 'approvals')!r}")
        t = cli_edit("tool_approval_mode", "auto")
        dt = await wait_until(
            lambda: na.has("tool approvals: auto") and nb.has("tool approvals: auto"),
            label="auto notice in both runtimes",
        )
        log(f"  both runtimes announced auto {dt:.2f}s after the write")
        log(f"  after:  A {await slash(a, 'approvals')!r}")
        log(f"  after:  B {await slash(b, 'approvals')!r}")

        log("\n== cell 2: tool_approval_mode auto -> ask, overriding a per-session /approvals ==")
        await slash(b, "approvals", "auto")  # B explicitly auto; the file will flip it back
        t = cli_edit("tool_approval_mode", "ask")
        dt = await wait_until(
            lambda: na.has("tool approvals: ask") and nb.has("tool approvals: ask"),
            label="ask notice in both runtimes",
        )
        log(f"  both runtimes announced ask {dt:.2f}s after the write")
        log(f"  after:  A {await slash(a, 'approvals')!r}")
        log(f"  after:  B {await slash(b, 'approvals')!r}  (was /approvals auto; disk wins)")

        # ---------------- model ----------------
        log("\n== cell 3: model_name edit alone -> config-sourced runtimes switch ==")
        log(f"  before: A={a.model_label} B={b.model_label}")
        t = cli_edit("model_name", "m-live")
        dt = await wait_until(
            lambda: a.model_label == "test/m-live" and b.model_label == "test/m-live",
            label="both bands on test/m-live",
        )
        log(f"  after:  A={a.model_label} B={b.model_label}  ({dt:.2f}s after the write)")
        await wait_until(lambda: na.has("test/m-boot → test/m-live"), label="A receipt")
        await wait_until(lambda: nb.has("test/m-boot → test/m-live"), label="B receipt")

        log("\n== cell 4: B makes an explicit /model choice; default edit keeps it ==")
        await slash(b, "model", "test/m-chosen")
        await wait_until(lambda: b.model_label == "test/m-chosen", label="B on m-chosen")
        log(f"  B now {b.model_label} (explicit)")
        t = cli_edit("model_name", "m-live-2")
        dt = await wait_until(lambda: a.model_label == "test/m-live-2", label="A switched")
        await wait_until(lambda: nb.has("keeping test/m-chosen"), label="B keep notice")
        log(f"  after:  A={a.model_label} ({dt:.2f}s)  B={b.model_label} (kept)")

        log("\n== cell 5: hosting -> unknown provider: warning, no switch ==")
        t = cli_edit("hosting", "no-such-provider")
        await wait_until(lambda: na.has("unknown provider"), label="A warning")
        log(f"  after:  A={a.model_label} (unchanged)")
        cli_edit("hosting", "test")

        # ---------------- web tools ----------------
        log("\n== cell 6: web_fetch.enabled false -> read <url> refused per call ==")
        # Exercise the REAL tool executor inside runtime A's process: the
        # mock provider has no way to emit a web_fetch call, so run the same
        # engine the tool calls, in a fresh process sharing the config dir.
        def run_probe(tag: str) -> str:
            out = subprocess.run([PY, "/tmp/lop-live-settings-probe.py"], capture_output=True, text=True, timeout=60)
            log(f"  probe ({tag}):\n    " + out.stdout.strip().replace("\n", "\n    "))
            return out.stdout

        cli_edit("web_fetch.enabled", "false")
        cli_edit("web_search.enabled", "false")
        out = run_probe("after disable")
        assert "web_fetch is disabled by config" in out and "web_search is disabled" in out
        cli_edit("web_fetch.enabled", "true")
        cli_edit("web_search.enabled", "true")
        out = run_probe("after enable")
        assert "disabled by config" not in out

        log("\n== cell 7: web tool INVENTORY follows at the next turn boundary (runtime A) ==")
        before = await slash(a, "context")
        cli_edit("web_search.enabled", "false")
        cli_edit("web_fetch.enabled", "false")
        await asyncio.sleep(2.5)  # one poll interval
        await a.prompt("hello")
        await wait_until(lambda: not a.frontend_state.streaming, label="turn end")
        after = await slash(a, "context")
        log(f"  tool schemas before: {before}\n  tool schemas after : {after}")
        cli_edit("web_search.enabled", "true")
        cli_edit("web_fetch.enabled", "true")
        await asyncio.sleep(2.5)
        await a.prompt("hello again")
        await wait_until(lambda: not a.frontend_state.streaming, label="turn end")
        back = await slash(a, "context")
        log(f"  tool schemas back  : {back}")
    finally:
        for v in (a, b):
            try:
                await v.dispose()
            except Exception:
                pass
        for p in (proc_a, proc_b):
            p.terminate()
        for p in (proc_a, proc_b):
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
    Path("/tmp/lop-live-settings-e2e.log").write_text("\n".join(REPORT) + "\n")


asyncio.run(main())
