"""Desktop open/attach latency against a REAL ``local-operator serve``, vs a 300 ms budget.

WHAT IT DRIVES (the requests the desktop renderer makes on a conversation click; see
local-operator-ui ``canonical-sessions-store.ts`` ``openSession``,
``use-canonical-session.ts`` and ``desktop-stream.ts``):

  list      GET  /v1/desktop/sessions?limit=500&include_archived=true      (sidebar)
  validate  GET  /v1/desktop/sessions/{id}                   (openSession, parallel)
  stream    GET  /v1/desktop/sessions/{id}/events?frontend_replace=1             (SSE)
              -> t_open     first frame ("open", carries subscription_id)
              -> t_snapshot "snapshot" frame (state + 100-row page) = FIRST PAINT
  watch     POST /v1/desktop/sessions/{id}/watch {visible:true}  (lease-driven warm)
              -> t_attached first frame whose cold == false = RUNTIME ATTACHED
  history   GET  /v1/desktop/sessions/{id}/history?limit=100 [&before_id]
  warm      POST /v1/desktop/sessions/{id}/warm       (first keystroke; CONTROL route)

SCENARIOS
  --scenario open    cold-facade open per fixture profile (every run a fresh APFS
                     clone, so the page cache is cold), plus a re-open (cache warm).
  --scenario attach  open + visible watch, until the runtime is attached.
  --scenario busy    the 20 s hunt: attach, then SIGSTOP the runtime (a live pid whose
                     loop does not answer: the wire shape of an owner stuck in a
                     synchronous step), close the stream, and time a re-open, a
                     ``warm`` and a ``send`` (control), and the reads issued WHILE each
                     control call is in flight. SIGCONT + kill afterwards.
  --scenario list    sidebar list with the store padded to --pad-sessions directories.

ISOLATION (AGENTS.md "Isolating a run"): fresh HOME + LOCAL_OPERATOR_CONFIG_DIR under a
session-unique tmp root, every LOP_* / CMUX_* var stripped, ``test`` provider, own port,
own desktop token (never printed). Only pids this script started are signalled. The root
is removed unless --keep. Run it with ``env -u XPC_FLAGS`` on macOS (a child inheriting
``XPC_FLAGS=0x2`` cannot resolve DNS).

USAGE (python = the worktree's .venv). ``--tree`` is the tree ``serve`` runs from, so one
copy of this script measures a base worktree and a branch worktree in turn (A/B at
comparable load):

  .venv/bin/python scripts/bench_desktop_open_attach.py --scenario open \
      --profiles tiny,p50,p90,p99,m5000,xl --runs 5 --json out-open.json
  # the PR #1472 busy-owner rows (the SIGSTOP owner-silent repro), for one tree:
  .venv/bin/python scripts/bench_desktop_open_attach.py --scenario busy \
      --profiles p50 --runs 3 --tree <worktree> --json busy.json

Wall ms are reported beside host load (``os.getloadavg``) per row. On a host running ~25
sessions at load ~100, treat p95 wall as weather and structural waits (a 2 s budget, a
15 s envelope) as the signal.

WHERE IT CAME FROM. Written for the 2026-09-22 desktop load diagnosis and committed with
PR #1472 so that PR's before/after table can be reproduced by someone other than its
author. Its only local dependency is the fixture builder beside it
(``bench_desktop_open_attach_fixtures.py``). Nothing imports it, and no test runs it.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

BUDGET_MS = 300.0
HERE = Path(__file__).resolve().parent


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _load() -> float:
    return round(os.getloadavg()[0], 1)


class Server:
    def __init__(self, root: Path, tree: Path) -> None:
        self.root, self.tree = root, tree
        self.config_dir = root / ".local-operator"
        self.token = secrets.token_hex(32)
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.proc: subprocess.Popen[bytes] | None = None

    def env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("LOP_", "CMUX_"))}
        env.pop("XPC_FLAGS", None)
        env.pop("LOCAL_OPERATOR_DESKTOP_ORIGINS", None)
        env.update(
            HOME=str(self.root),
            LOCAL_OPERATOR_CONFIG_DIR=str(self.config_dir),
            LOCAL_OPERATOR_DESKTOP_TOKEN=self.token,
            PYTHONPATH=str(self.tree),
            # The REAL kill switch (``local_operator.tui.notify.ENV_DISABLE``).
            # The diagnosis copy set ``LOCAL_OPERATOR_NO_NOTIFY``, a name nothing
            # reads, so a mock-provider runtime this bench spawned could have put
            # its reply on the operator's lock screen; the environment builder
            # guard (``tests/unit/test_notification_isolation.py``) caught it.
            LOCAL_OPERATOR_NO_NOTIFICATIONS="1",
        )
        return env

    def start(self) -> None:
        import httpx

        self.config_dir.mkdir(parents=True, exist_ok=True)
        (self.root / "work").mkdir(exist_ok=True)
        cfg = self.config_dir / "config.yml"
        if not cfg.exists():
            cfg.write_text("version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n")
        log = (self.root / "serve.log").open("ab")
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "local_operator.cli",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
            ],
            env=self.env(),
            cwd=str(self.root / "work"),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise SystemExit(f"serve exited {self.proc.returncode}; see {self.root}/serve.log")
            try:
                if httpx.get(self.base + "/health", timeout=2).status_code < 500:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        raise SystemExit("serve never answered /health")

    def client(self, timeout: float | None = 60.0):
        import httpx

        return httpx.Client(
            base_url=self.base, headers={"Authorization": "Bearer " + self.token}, timeout=timeout
        )

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)


class Stream:
    """One SSE subscription on a thread, recording when each milestone frame lands."""

    def __init__(self, server: Server, sid: str, t0: float) -> None:
        self.server, self.sid, self.t0 = server, sid, t0
        self.marks: dict[str, float] = {}
        self.sizes: dict[str, int] = {}
        self.sub_id: str | None = None
        self.snapshot_cold: bool | None = None
        self.error: str | None = None
        self._stop = threading.Event()
        self._opened = threading.Event()
        self._snap = threading.Event()
        self._attached = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _mark(self, name: str, size: int = 0) -> None:
        if name not in self.marks:
            self.marks[name] = (time.perf_counter() - self.t0) * 1000
            self.sizes[name] = size

    def _run(self) -> None:
        import httpx

        try:
            with (
                self.server.client(timeout=None) as c,
                c.stream(
                    "GET",
                    f"/v1/desktop/sessions/{self.sid}/events",
                    params={"frontend_replace": 1},
                    headers={"Accept": "text/event-stream"},
                ) as r,
            ):
                self._mark("headers")
                if r.status_code != 200:
                    self.error = f"HTTP {r.status_code}"
                    return
                for line in r.iter_lines():
                    if self._stop.is_set():
                        return
                    if not line.startswith("data: "):
                        continue
                    frame = json.loads(line[6:])
                    kind = frame.get("type")
                    if kind == "open":
                        self.sub_id = frame["payload"]["subscription_id"]
                        self._mark("open", len(line))
                        self._opened.set()
                    elif kind == "snapshot":
                        self._mark("snapshot", len(line))
                        self.snapshot_cold = frame["payload"].get("cold")
                        self.cold_reason = frame["payload"].get("cold_reason")
                        self.snapshot_rows = len(frame["payload"]["history"]["entries"])
                        self._snap.set()
                        if frame["payload"].get("cold") is False:
                            self._mark("attached", len(line))
                            self._attached.set()
                    elif kind in ("frontend.update", "frontend.replace"):
                        if (frame.get("payload") or {}).get("cold") is False:
                            self._mark("attached", len(line))
                            self._attached.set()
        except (httpx.HTTPError, OSError) as e:  # noqa: PERF203
            if not self._stop.is_set():
                self.error = repr(e)
        finally:
            self._opened.set()
            self._snap.set()

    def start(self) -> "Stream":
        self.thread.start()
        return self

    def close(self) -> None:
        self._stop.set()


def _timed(fn) -> tuple[float, Any]:
    t = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t) * 1000, out


def clone_session(config_dir: Path, sid: str, suffix: int) -> str:
    """An APFS clone of a fixture under a fresh id, so each run is page-cache cold."""
    new = (sid[:8] + f"{suffix:04x}")[:12]
    src, dst = config_dir / "sessions" / sid, config_dir / "sessions" / new
    if dst.exists():
        shutil.rmtree(dst)
    subprocess.run(["cp", "-Rc", str(src), str(dst)], check=True)
    return new


def open_once(
    server: Server, sid: str, *, watch: bool, attach_timeout: float = 30.0
) -> dict[str, Any]:
    load = _load()
    t0 = time.perf_counter()
    stream = Stream(server, sid, t0).start()
    validate: dict[str, Any] = {}

    def _validate() -> None:
        with server.client() as c:
            ms, r = _timed(lambda: c.get(f"/v1/desktop/sessions/{sid}"))
            validate.update(ms=ms, status=r.status_code, bytes=len(r.content))

    vt = threading.Thread(target=_validate)
    vt.start()
    stream._snap.wait(120)
    row: dict[str, Any] = {"sid": sid, "load": load, "error": stream.error}
    if watch and stream.sub_id:
        with server.client() as c:
            ms, r = _timed(
                lambda: c.post(
                    f"/v1/desktop/sessions/{sid}/watch",
                    json={"subscription_id": stream.sub_id, "visible": True, "can_notify": False},
                )
            )
            row["watch_ms"], row["watch_status"] = round(ms, 1), r.status_code
        stream._attached.wait(attach_timeout)
    vt.join(120)
    with server.client() as c:
        ms, r = _timed(lambda: c.get(f"/v1/desktop/sessions/{sid}/history", params={"limit": 100}))
        row["history_tail_ms"], row["history_tail_bytes"] = round(ms, 1), len(r.content)
        entries = r.json()["result"]["entries"] if r.status_code == 200 else []
        if entries:
            ms, r = _timed(
                lambda: c.get(
                    f"/v1/desktop/sessions/{sid}/history",
                    params={"limit": 100, "before_id": entries[0]["id"]},
                )
            )
            row["history_older_ms"] = round(ms, 1)
    for k in ("headers", "open", "snapshot", "attached"):
        row[f"t_{k}_ms"] = round(stream.marks[k], 1) if k in stream.marks else None
    row["snapshot_bytes"] = stream.sizes.get("snapshot")
    row["snapshot_rows"] = getattr(stream, "snapshot_rows", None)
    row["snapshot_cold"] = stream.snapshot_cold
    row["cold_reason"] = getattr(stream, "cold_reason", None)
    row["validate_ms"] = round(validate.get("ms", float("nan")), 1)
    row["validate_bytes"] = validate.get("bytes")
    row["stream"] = stream
    return row


def summarize(values: list[float]) -> dict[str, Any]:
    v = sorted(x for x in values if x is not None)
    if not v:
        return {}
    p = lambda q: v[min(len(v) - 1, int(round(q * (len(v) - 1))))]  # noqa: E731
    return {
        "n": len(v),
        "p50": round(statistics.median(v), 1),
        "p95": round(p(0.95), 1),
        "max": round(v[-1], 1),
        "over_budget": sum(1 for x in v if x > BUDGET_MS),
    }


def _runtime_pid(config_dir: Path, sid: str) -> int | None:
    from local_operator.resume import live_runtime_pid

    return live_runtime_pid(config_dir, sid)


def _state(pid: int) -> str:
    out = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
    ).stdout.strip()
    return out or "gone"


def _kill_runtime(config_dir: Path, sid: str, spawned: set[int]) -> None:
    pid = _runtime_pid(config_dir, sid)
    for p in {pid, *spawned} - {None}:
        for sig in (signal.SIGCONT, signal.SIGTERM):
            try:
                os.kill(p, sig)
            except OSError:
                pass


def scenario_open(
    server: Server,
    manifest: dict[str, Any],
    profiles: list[str],
    runs: int,
    out: list[Any],
) -> None:
    for name in profiles:
        base = manifest[name]["session_id"]
        for i in range(runs):
            sid = clone_session(server.config_dir, base, i + 1)
            cold = open_once(server, sid, watch=False)
            cold["stream"].close()
            time.sleep(0.3)
            warm = open_once(server, sid, watch=False)  # same id: page cache warm, facade rebuilt
            warm["stream"].close()
            for label, row in (("cold", cold), ("reopen", warm)):
                row.pop("stream")
                row.update(
                    profile=name,
                    pass_=label,
                    bytes=manifest[name]["bytes"],
                    rows=manifest[name]["rows"],
                )
                out.append(row)
                print(
                    f"{name:6} {label:6} load={row['load']:6} open={row['t_open_ms']} "
                    f"snap={row['t_snapshot_ms']} ({row['snapshot_bytes']} B, "
                    f"{row['snapshot_rows']} rows) validate={row['validate_ms']} "
                    f"hist={row['history_tail_ms']}/{row.get('history_older_ms')} "
                    f"err={row['error']}",
                    flush=True,
                )
            time.sleep(0.5)


def scenario_attach(
    server: Server,
    manifest: dict[str, Any],
    profiles: list[str],
    runs: int,
    out: list[Any],
) -> None:
    for name in profiles:
        base = manifest[name]["session_id"]
        for i in range(runs):
            sid = clone_session(server.config_dir, base, 0x100 + i)
            row = open_once(server, sid, watch=True)
            stream = row.pop("stream")
            pid = _runtime_pid(server.config_dir, sid)
            row.update(profile=name, pass_="attach", runtime_pid=pid)
            stream.close()
            _kill_runtime(server.config_dir, sid, {pid} if pid else set())
            out.append(row)
            print(
                f"{name:6} attach load={row['load']} snap={row['t_snapshot_ms']} "
                f"attached={row['t_attached_ms']} "
                f"watch={row.get('watch_ms')} err={row['error']}",
                flush=True,
            )
            time.sleep(1.5)


args_gap = 20.0


def scenario_busy(
    server: Server,
    manifest: dict[str, Any],
    profiles: list[str],
    runs: int,
    out: list[Any],
) -> None:
    """Live-but-unresponsive owner: what each route pays, and whether control blocks reads."""
    for name in profiles:
        base = manifest[name]["session_id"]
        for i in range(runs):
            sid = clone_session(server.config_dir, base, 0x200 + i)
            first = open_once(server, sid, watch=True)
            stream0 = first.pop("stream")
            pid = _runtime_pid(server.config_dir, sid)
            row: dict[str, Any] = {
                "profile": name,
                "pass_": "busy",
                "sid": sid,
                "runtime_pid": pid,
                "attached_first_ms": first["t_attached_ms"],
                "load": _load(),
            }
            if not pid:
                row["error"] = "no runtime attached"
                out.append(row)
                print(row)
                continue
            # Freeze the owner WHILE the viewer still holds it, so it cannot see the lease
            # drop and drain: a live pid, a live record, a socket nobody answers -- the wire
            # shape of a runtime whose loop is blocked in a long synchronous step.
            os.kill(pid, signal.SIGSTOP)
            stream0.close()
            # The SSE generator only notices a gone client at its next write (the 15 s
            # heartbeat, desktop_sessions.py:2334), so the bridge stays acquired -- still
            # "attached" to the frozen owner -- until then. Wait it out so the next open dials.
            time.sleep(args_gap)
            row["pid_state_before"] = _state(pid)
            try:
                r1 = open_once(server, sid, watch=False)
                r1.pop("stream").close()
                row.update(
                    reopen_snapshot_ms=r1["t_snapshot_ms"],
                    reopen_validate_ms=r1["validate_ms"],
                    reopen_cold=r1["snapshot_cold"],
                    reopen_cold_reason=r1["cold_reason"],
                    pid_alive_stopped=_state(pid),
                )
                time.sleep(4)
                # A control call (the renderer's first-keystroke warm) against the
                # silent owner, and a read issued 200 ms into it: does the read queue
                # behind the control attach?
                ctl: dict[str, Any] = {}

                def _warm() -> None:
                    with server.client(timeout=90) as c:
                        ms, r = _timed(lambda: c.post(f"/v1/desktop/sessions/{sid}/warm", json={}))
                        ctl.update(ms=ms, status=r.status_code, body=r.text[:200])

                wt = threading.Thread(target=_warm)
                row["pid_state_at_warm"] = _state(pid)
                wt.start()
                time.sleep(0.2)
                r2 = open_once(server, sid, watch=False)
                r2.pop("stream").close()
                wt.join(120)
                # The SEND (a control route, sessions.message) against the frozen owner, and a
                # read issued 300 ms into it: the renderer's 20 s control deadline
                # (desktop-contract.ts:2179) is what the user sees if this exceeds it.
                time.sleep(args_gap)
                snd: dict[str, Any] = {}

                def _send() -> None:
                    import uuid as _u

                    with server.client(timeout=120) as c:
                        ms, r = _timed(
                            lambda: c.post(
                                f"/v1/desktop/sessions/{sid}/messages",
                                json={"request_id": str(_u.uuid4()), "text": "hi"},
                            )
                        )
                        snd.update(ms=ms, status=r.status_code, body=r.text[:300])

                st = threading.Thread(target=_send)
                st.start()
                time.sleep(0.3)
                r3 = open_once(server, sid, watch=False)
                r3.pop("stream").close()
                st.join(150)
                row.update(
                    send_ms=round(snd.get("ms", float("nan")), 1),
                    send_status=snd.get("status"),
                    send_body=snd.get("body"),
                    during_send_snapshot_ms=r3["t_snapshot_ms"],
                    during_send_validate_ms=r3["validate_ms"],
                    during_send_hist_ms=r3["history_tail_ms"],
                    during_send_cold=r3["snapshot_cold"],
                )
                row.update(
                    warm_ms=round(ctl.get("ms", float("nan")), 1),
                    warm_status=ctl.get("status"),
                    warm_body=ctl.get("body"),
                    during_warm_snapshot_ms=r2["t_snapshot_ms"],
                    during_warm_validate_ms=r2["validate_ms"],
                    during_warm_error=r2["error"],
                    during_warm_started_after_ms=200,
                )
            finally:
                _kill_runtime(server.config_dir, sid, {pid})
            out.append(row)
            print(json.dumps(row), flush=True)
            time.sleep(1.5)


def scenario_list(server: Server, pad: int, runs: int, out: list[Any]) -> None:
    sessions = server.config_dir / "sessions"
    existing = len(list(sessions.iterdir()))
    for i in range(max(0, pad - existing)):
        d = sessions / f"{0xa00000000000 + i:012x}"
        d.mkdir(exist_ok=True)
        (d / "created_at.json").write_text(json.dumps(time.time() - i))
        (d / "transcript.jsonl").write_text(
            json.dumps(
                {
                    "id": f"m{i}",
                    "ts": time.time() - i,
                    "type": "message",
                    "payload": {
                        "kind": "message",
                        "role": "user",
                        "content": [{"type": "text", "text": f"hello {i}"}],
                    },
                }
            )
            + "\n"
        )
    with server.client(timeout=120) as c:
        for i in range(runs):
            load = _load()
            ms, r = _timed(
                lambda: c.get(
                    "/v1/desktop/sessions", params={"limit": 500, "include_archived": "true"}
                )
            )
            out.append(
                {
                    "pass_": "list",
                    "store_dirs": len(list(sessions.iterdir())),
                    "ms": round(ms, 1),
                    "bytes": len(r.content),
                    "status": r.status_code,
                    "load": load,
                    "run": i,
                }
            )
            print(out[-1], flush=True)


def report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for row in rows:
        key = f"{row.get('profile', '-')}/{row['pass_']}"
        summary.setdefault(key, []).append(row)
    table = {}
    for key, group in summary.items():
        cols = {}
        for col in (
            "t_open_ms",
            "t_snapshot_ms",
            "t_attached_ms",
            "validate_ms",
            "history_tail_ms",
            "history_older_ms",
            "watch_ms",
            "ms",
            "warm_ms",
            "reopen_snapshot_ms",
            "during_warm_snapshot_ms",
            "snapshot_bytes",
        ):
            vals = [g.get(col) for g in group if isinstance(g.get(col), (int, float))]
            if vals:
                cols[col] = summarize(vals)
        table[key] = cols
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="open", choices=["open", "attach", "busy", "list", "all"])
    ap.add_argument("--profiles", default="tiny,p50,p90,p99")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--pad-sessions", type=int, default=2000)
    ap.add_argument("--root", type=Path, help="reuse a fixture root (skips building)")
    ap.add_argument("--tree", type=Path, default=Path.cwd(), help="local-operator tree to serve")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    for k in [k for k in os.environ if k.startswith(("LOP_", "CMUX_"))]:
        del os.environ[k]
    root = args.root or Path(tempfile.mkdtemp(prefix="lop-arch-bench-"))
    server = Server(root, args.tree.resolve())
    os.environ.update(HOME=str(root), LOCAL_OPERATOR_CONFIG_DIR=str(server.config_dir))
    sys.path.insert(0, str(args.tree.resolve()))
    manifest_path = root / "manifest.json"
    profiles = args.profiles.split(",")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    missing = [p for p in profiles if p not in manifest]
    if missing:
        server.config_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(HERE / "bench_desktop_open_attach_fixtures.py"),
                "--config-dir",
                str(server.config_dir),
                "--profiles",
                ",".join(missing),
                "--manifest",
                str(manifest_path),
            ],
            check=True,
            cwd=str(args.tree),
            env=server.env(),
        )
        manifest = json.loads(manifest_path.read_text())
    rows: list[dict[str, Any]] = []
    try:
        server.start()
        assert server.proc is not None
        print(f"serve pid {server.proc.pid} on {server.base}; root {root}", flush=True)
        scen = ["open", "attach", "busy", "list"] if args.scenario == "all" else [args.scenario]
        for s in scen:
            if s == "open":
                scenario_open(server, manifest, profiles, args.runs, rows)
            elif s == "attach":
                scenario_attach(server, manifest, profiles, args.runs, rows)
            elif s == "busy":
                scenario_busy(server, manifest, profiles, args.runs, rows)
            elif s == "list":
                scenario_list(server, args.pad_sessions, args.runs, rows)
    finally:
        server.stop()
        table = report(rows)
        print(json.dumps(table, indent=1))
        if args.json:
            args.json.write_text(
                json.dumps(
                    {
                        "budget_ms": BUDGET_MS,
                        "manifest": manifest,
                        "rows": rows,
                        "summary": table,
                        "tree": str(args.tree),
                    },
                    indent=1,
                    default=str,
                )
            )
        if not args.keep and not args.root:
            shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
