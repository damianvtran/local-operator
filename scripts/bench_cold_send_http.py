"""Drive the real desktop HTTP API and time the first send of a new conversation.

WHAT THIS MEASURES, AND WHY IT IS THE NUMBER THAT MATTERS
=========================================================
``scripts/bench_cold_engage.py`` decomposes the cold engage into phases, which
is what a change needs in order to be shown to move the phase it claims to
move. This script measures the thing the user actually waits for, the way the
user causes it:

    POST /v1/desktop/sessions                      open a new conversation
    POST /v1/desktop/sessions/{id}/messages        press send

The second POST is the reported number: the desktop's send button does not
return until the message is ADMITTED, and admission is what binds the cold
runtime — spawn, construction, publication, dial, frontend sync, history load.
A change that shortens the child's pre-publication window without shortening
this number has not delivered anything the user can feel.

The conversation is NOT subscribed to first. That is deliberate: a live watch
lease triggers the warm path, and a warmed session would measure the wrong
thing — the complaint this script exists to reproduce is the first send of a
brand-new conversation, which pays the engage inline.

ISOLATION
=========
It drives a real ``local-operator serve`` SUBPROCESS, started with its own
fresh ``HOME`` and ``LOCAL_OPERATOR_CONFIG_DIR``, its own port, and its own
desktop token. ``LOP_*`` and ``CMUX_*`` are stripped before it starts, so it
cannot attach to the operator's live sessions or rename their cmux workspaces.
Every runtime it spawns is killed between runs, and the whole root is removed
at the end.

``LOCAL_OPERATOR_DESKTOP_TOKEN`` is a capability, not a preference: without it
the server refuses the desktop surface. The token is generated per run and used
only in-process — it is never written into the repository or into the tree.

USAGE
=====
    .venv/bin/python scripts/bench_cold_send_http.py --runs 3
    .venv/bin/python scripts/bench_cold_send_http.py --runs 3 --json out.json
    .venv/bin/python scripts/bench_cold_send_http.py --runs 3 --mcp-variant mcpx \\
        --measured-tree 2bf8cb890 --label "pre-fix arm"
    .venv/bin/python scripts/bench_cold_send_http.py --runs 5 \\
        --pre-engage keystroke --think 3 --label "draft-warm arm"

PRE-ENGAGE ARMS, AND THE BRIDGE THEY MUST HOLD. ``--pre-engage keystroke``
reproduces the O3 pane: mint a draft, open its ``/events`` stream, beat
``/watch`` with the subscription id the stream published, fire ``/warm``, wait
``--think`` seconds (the typing the warm overlaps), then ``create`` with the
draft id and send. ``--pre-engage open`` is the O4 variant — the real pane
fires the mint+warm at PANE OPEN, so ``--think`` there means the whole
pane-open-to-send span; the rig's sequence is otherwise identical.

THE RIG MUST HOLD THE STREAM OR IT MEASURES NOTHING. A warm whose only bridge
user is its own request is cancelled when that request returns
(``DesktopSessionBridge.warm``'s own docstring), so the draft arms keep the
``/events`` stream open in a background thread for the WHOLE run — the same
standing user the pane's subscription is — and fire ``/warm`` only after the
stream published its ``subscription_id`` (the ``/watch`` beat needs a live
subscription, exactly as the renderer's does). **Attribution per run**: the
rig records a runtime-child census immediately BEFORE and AFTER the send; a
draft run whose pre-send census shows NO runtime child timed a cancelled or
never-fired warm, is labelled INVALID in the summary, and exits 4 rather than
reporting it as a draft-warm number.

MEASURED TREE. ``--measured-tree`` names the commit whose ``local_operator/`` this
run measures and is VERIFIED against the subtree on disk (``bench_tree``), because
the before arm of a before/after pair is measured with the subtree checked out of
another commit while the worktree HEAD stays put. The run is refused when they
disagree, so the artefact's ``rev`` always names a tree this run actually measured
(review round 2, R2-1).

DECLARATION, READ BACK. ``--mcp-variant mcpx`` is supposed to declare one MCP
server and ``none`` is supposed to declare nothing, and the session reports what
it actually sees back — recorded per row as ``mcp_declared_servers`` and graded in
the summary. A run whose declaration does not match its own arm name exits 3
rather than reporting a treatment arm that measured the control (QA round 1, Q1).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import secrets
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_operator.tui.notify import suppress_notifications_for_process  # noqa: E402
from scripts import bench_tree  # noqa: E402

_STRIPPED_PREFIXES = ("LOP_", "CMUX_")

#: How long to wait for the isolated server to answer /health. The server is a
#: whole FastAPI app plus the canonical store, so this is generous on purpose —
#: it is a start-up budget, not a measurement.
BOOT_TIMEOUT_S = 90.0


def _strip_inherited() -> None:
    for key in list(os.environ):
        if key.startswith(_STRIPPED_PREFIXES):
            del os.environ[key]


def _seed_config(root: Path) -> Path:
    """The same minimal config the other benchmarks use (`test` provider)."""
    config_dir = root / ".local-operator"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n", encoding="utf-8"
    )
    return config_dir


def _write_mcp_config(cwd: Path, variant: str) -> None:
    """Declare one MCP server, the way a real machine does.

    `none` declares nothing (the control); `mcpx` points at a closed port, so
    nothing is spawned and the refused connection is immediate — what is left is
    the SDK import and the config parse, i.e. what ANY declared server costs.
    This is the operator's own condition (gitlab, google-workspace) reduced to
    its cheapest form, because the whole point of the measurement is the
    difference between a machine with a server declared and one without.

    WRITTEN INTO THE SESSION'S CWD, and the parameter is the cwd rather than the
    run root because that is where discovery looks: resolution reads
    ``<cwd>/.mcp.json`` (plus the config dir, ``~/.claude.json``, ...), never
    ``<HOME>/.mcp.json``. Writing it to ``HOME`` made the `mcpx` arm declare
    nothing, i.e. measured the control arm twice (QA round 1, Q1).
    """
    if variant == "none":
        return
    entry: dict[str, Any] = {"type": "http", "url": "http://127.0.0.1:1/mcp"}
    (cwd / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"bench-slow": entry}}), encoding="utf-8"
    )


def _free_port() -> int:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    return port


def _wait_for_health(client: Any, url: str, proc: subprocess.Popen[bytes]) -> None:
    import httpx

    deadline = time.monotonic() + BOOT_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"server exited early with {proc.returncode}")
        try:
            response = client.get(url + "/health")
            if response.status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise SystemExit("server never answered /health")


def _standby_children(daemon_pid: int) -> list[int]:
    """This daemon's live ``[standby]`` children, by exact pid.

    THE READINESS OF THE DAEMON'S STANDBY IS DELIBERATELY UNOBSERVABLE FROM
    OUTSIDE, so this arm reports EVERY run and labels the first one. The daemon's
    standby is reached over a socketpair the daemon forked: the previous revision
    announced readiness with a file in the rooted store, and a path a reader can
    inspect is a path a same-uid impostor can bind (review round 1, R1-1). What is
    observable is that the child EXISTS; whether its warm has finished decides
    whether the send that arrives now is adopted or appended to a cold spawn, and
    the per-run numbers show which happened.
    """
    from local_operator.session.runtime import standby

    # Same trap, same fix as ``bench_standby_engage._standby_child`` (round 8, R8-1):
    # procps truncates ``command`` at the terminal width, so on Linux this recorded
    # ``standby_children: 0`` as a MEASUREMENT and reaped nothing — a dead instrument
    # returning a plausible zero. ``-eww`` for the width, ``-m <module>`` for the shape.
    out = subprocess.run(
        ["ps", "-eww", "-o", "pid=,ppid=,command="], capture_output=True, text=True
    )
    found: list[int] = []
    for line in out.stdout.splitlines():
        fields = line.split(None, 2)
        if len(fields) < 3:
            continue
        pid, ppid, command = fields
        if int(ppid) == daemon_pid and f"-m {standby.STANDBY_MODULE}" in command:
            found.append(int(pid))
    return sorted(found)


def _runtime_children(daemon_pid: int) -> list[int]:
    """This daemon's live SESSION-runtime children, by exact pid.

    The census the draft arms are attributable with: a draft warm that was
    actually held must show its runtime child HERE, before the send. A run whose
    pre-send census is empty timed a cancelled or never-fired warm and is
    reported INVALID rather than slow.

    ``-m RUNTIME_MODULE`` names a session runtime specifically; the standby
    (``STANDBY_MODULE``) is DELIBERATELY not counted — it is not a draft's spawn,
    and folding it in would make the attribution vacuous. Same ``-eww`` width
    trap as ``_standby_children``: procps truncates ``command`` at the terminal
    width, and a truncated line reads as a measurement of nothing.
    """
    from local_operator.session.runtime.types import RUNTIME_MODULE

    out = subprocess.run(
        ["ps", "-eww", "-o", "pid=,ppid=,command="], capture_output=True, text=True
    )
    found: list[int] = []
    for line in out.stdout.splitlines():
        fields = line.split(None, 2)
        if len(fields) < 3:
            continue
        pid, ppid, command = fields
        if int(ppid) == daemon_pid and f"-m {RUNTIME_MODULE}" in command:
            found.append(int(pid))
    return sorted(found)


def _kill_runtime(config_dir: Path, session_id: str) -> None:
    """Kill the runtime this session spawned, so runs stay independent.

    A benchmark that leaves its runtimes resident measures the next run against
    a machine it degraded itself.
    """
    from local_operator.mobile.attach_client import find_runtime_record

    try:
        record, _owner = find_runtime_record(config_dir, session_id)
    except Exception:  # noqa: BLE001 — a missing record is nothing to clean up
        return
    pid = getattr(record, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        return
    try:
        os.kill(pid, 15)
    except OSError:
        return
    for _ in range(300):
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.01)
    try:
        os.kill(pid, 9)
    except OSError:
        pass


class _DraftStream:
    """One draft's ``/events`` subscription, held for the whole run.

    WHY THE RIG MUST HOLD IT: a warm whose only bridge user is its own request
    is cancelled when that request returns (``DesktopSessionBridge.warm``'s own
    docstring), so without a standing subscription the draft arm would measure
    the cancellation rather than the engage. The thread also reads the stream's
    ``open`` frame, which is where the renderer gets the ``subscription_id`` its
    ``/watch`` beat then carries — the rig cannot invent one: the beat answers
    404 for a subscription the bridge never had.

    A DAEMON THREAD WITH ITS OWN CLIENT: the sync httpx client cannot
    interleave requests on a busy streaming response, and the run's own requests
    must keep flowing. Lines are read and discarded after the id is learned;
    ``close()`` unblocks the reader by closing the response, and the thread is a
    daemon so a wedged read can never outlive the process.
    """

    def __init__(self, base: str, headers: dict[str, str], draft_id: str) -> None:
        self.base, self.headers, self.draft_id = base, headers, draft_id
        self.subscription_id: str | None = None
        self.status: int | None = None
        self.error: BaseException | None = None
        self._opened = threading.Event()
        self._stop = threading.Event()
        self._response: Any = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_DraftStream":
        self._thread = threading.Thread(target=self._read, name="draft-events", daemon=True)
        self._thread.start()
        if not self._opened.wait(20.0):
            raise SystemExit(f"the draft events stream never opened ({self.error!r})")
        if self.status != 200:
            raise SystemExit(f"the draft events stream answered {self.status}")
        return self

    def _read(self) -> None:
        import httpx

        try:
            timeout = httpx.Timeout(10.0, read=120.0)
            with httpx.Client(base_url=self.base, headers=self.headers, timeout=timeout) as client:
                with client.stream(
                    "GET", f"/v1/desktop/sessions/{self.draft_id}/events"
                ) as response:
                    self.status = response.status_code
                    self._response = response
                    self._opened.set()
                    for line in response.iter_lines():
                        if self._stop.is_set():
                            break
                        if not line.startswith("data: "):
                            continue
                        frame = json.loads(line[len("data: ") :])
                        # The id lives in the OPEN frame's PAYLOAD, not at the
                        # top level: ``bridge.events`` yields
                        # ``{"type": "open", "payload": {"subscription_id": ...}}``.
                        payload = frame.get("payload") or {}
                        announced = payload.get("subscription_id")
                        if self.subscription_id is None and announced:
                            self.subscription_id = str(announced)
        except BaseException as exc:  # noqa: BLE001 — surfaced through `error`/`status`
            self.error = exc
            self._opened.set()

    def wait_subscription(self, timeout: float = 20.0) -> str:
        """The id the stream's ``open`` frame published (the watch beat's argument)."""
        deadline = time.monotonic() + timeout
        while self.subscription_id is None:
            if self.error is not None:
                raise SystemExit(f"the draft events stream failed: {self.error!r}")
            if time.monotonic() > deadline:
                raise SystemExit("the draft events stream never published its subscription id")
            time.sleep(0.02)
        return self.subscription_id

    def close(self) -> None:
        self._stop.set()
        if self._response is not None:
            try:
                self._response.close()
            except Exception:  # noqa: BLE001 — closing a live stream may raise by design
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _declared_servers(client: Any, session_id: str) -> list[str] | None:
    """The MCP servers the RUNNING session sees declared, asked of the session.

    A benchmark whose "declared" arm declares nothing is measuring its own
    control, and on this host that produced plausible-looking numbers for a whole
    campaign (QA round 1, Q1: the config was written to ``HOME`` while discovery
    reads ``<cwd>/.mcp.json``). So the declaration is READ BACK from the running
    session and recorded in the row, and ``_summarize`` grades it: the arm's own
    output now carries the proof that it measured what it claims.

    Read the cold GET before the timed send: it resolves the session's actual
    cwd without binding a runtime. The live POST refuses while the deferred
    manager is starting (and forever for a no-server session), so that refusal
    cannot distinguish absent configuration from wiring that is merely pending.
    Require the cold marker as well: verification must not warm the measured path.
    An unreadable response is unknown, never a successful empty control.
    """
    try:
        listed = client.get(f"/v1/desktop/sessions/{session_id}/mcp")
        listed.raise_for_status()
        data = listed.json()["result"]["data"]
        if data.get("cold") is not True:
            return None
        return sorted(str(server["name"]) for server in data["servers"])
    except Exception:  # noqa: BLE001 — a record we could not read is not a result
        return None


def _one_run(
    client: Any,
    base: str,
    workspace: Path,
    config_dir: Path,
    index: int,
    variant: str,
    *,
    pre_engage: str,
    think: float,
    daemon_pid: int,
    headers: dict[str, str],
) -> dict[str, Any]:
    """One new conversation, one first send, both response bodies kept.

    The host load is sampled per run and travels WITH the row, because a first
    send measured at load 10 and one at load 200 are not the same measurement and
    the difference is invisible after the fact (review round 1, R5).

    THE DRAFT ARMS RUN INSIDE A ``_DraftStream`` CONTEXT, from just after the
    mint to just after the send: the stream is the standing bridge user that
    keeps the warm alive (``warm()``'s own precondition), so the create and the
    send must happen while it is open. The control arm takes no stream.
    """
    loadavg = os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan")
    result: dict[str, Any] = {
        "run": index,
        "pre_engage": pre_engage,
        "think_s": think,
        "loadavg": round(loadavg, 1),
        "variant": variant,
        "mint_ms": None,
        "warm_ms": None,
        "watch_ms": None,
    }
    draft_id: str | None = None

    with contextlib.ExitStack() as stack:
        if pre_engage != "off":
            started = time.perf_counter()
            minted = client.post(
                "/v1/desktop/sessions/draft",
                json={"request_id": str(uuid.uuid4()), "cwd": str(workspace)},
            )
            result["mint_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
            assert minted.status_code == 200, minted.text
            draft_id = minted.json()["result"]["draft_id"]
            result["draft_id"] = draft_id
            stream = stack.enter_context(_DraftStream(base, headers, draft_id))
            subscription_id = stream.wait_subscription()
            started = time.perf_counter()
            watched = client.post(
                f"/v1/desktop/sessions/{draft_id}/watch",
                json={
                    "subscription_id": subscription_id,
                    "visible": True,
                    "can_notify": False,
                },
            )
            result["watch_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
            assert watched.status_code == 200, watched.text
            # THE KEYSTROKE POINT: the warm fires once, and only now that the
            # subscription holds the bridge — the same ordering the renderer
            # enforces, and the reason the stream is held at all.
            started = time.perf_counter()
            warmed = client.post(f"/v1/desktop/sessions/{draft_id}/warm", json={})
            result["warm_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
            assert warmed.status_code == 200, warmed.text
            if think > 0:
                time.sleep(think)

        started = time.perf_counter()
        create_body: dict[str, Any] = {
            "request_id": str(uuid.uuid4()),
            "cwd": str(workspace),
        }
        if draft_id is not None:
            create_body["draft_id"] = draft_id
        created = client.post("/v1/desktop/sessions", json=create_body)
        create_ms = round((time.perf_counter() - started) * 1000.0, 1)
        created_body = created.json()
        session_id = created_body["result"]["session_id"]

        declared = _declared_servers(client, session_id)
        census_before = _runtime_children(daemon_pid)
        started = time.perf_counter()
        sent = client.post(
            f"/v1/desktop/sessions/{session_id}/messages",
            json={"request_id": str(uuid.uuid4()), "text": "bench: one word"},
        )
        first_send_ms = round((time.perf_counter() - started) * 1000.0, 1)
        census_after = _runtime_children(daemon_pid)

    result.update(
        {
            "session_id": session_id,
            "create_ms": create_ms,
            "first_send_ms": first_send_ms,
            "runtime_children_before_send": census_before,
            "runtime_children_after_send": census_after,
            # THE PER-RUN ATTRIBUTION (spec §3.1): a draft arm whose pre-send
            # census shows no runtime child did not warm at all — the warm was
            # cancelled (no standing bridge user) or never fired (a refused
            # beat) — so its send number is not a draft-warm reading. ``None``
            # on the control arm, which has no warm to attribute.
            "warm_spawned_before_send": (bool(census_before) if pre_engage != "off" else None),
            "mcp_declared_servers": declared,
            "mcp_declaration_correct": declared == (["bench-slow"] if variant == "mcpx" else []),
            "create_status": created.status_code,
            "create_body": created_body,
            "send_status": sent.status_code,
            "send_body": (
                sent.json()
                if sent.headers.get("content-type", "").startswith("application/json")
                else sent.text
            ),
        }
    )
    _kill_runtime(config_dir, session_id)
    if draft_id is not None and draft_id != session_id:
        # The create minted FRESH (the draft was lost or expired) — reap the
        # warm's runtime by its own id too, or the rig leaks one process per
        # such run.
        _kill_runtime(config_dir, draft_id)
    return result


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [row["first_send_ms"] for row in rows]
    if not values:
        return {}
    declared = sorted({name for row in rows for name in (row.get("mcp_declared_servers") or [])})
    correct = [row.get("mcp_declaration_correct") for row in rows]
    draft_rows = [row for row in rows if row.get("pre_engage", "off") != "off"]
    return {
        "median": round(statistics.median(values), 1),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
        "n": len(values),
        "pre_engage": rows[0].get("pre_engage", "off"),
        "think_s": rows[0].get("think_s", 0.0),
        #: Draft arms only: whether EVERY run's pre-send census showed a warm's
        #: runtime (``all([])`` is True, so the None guard is explicit), and the
        #: runs that did not — the rows that timed a cancelled or never-fired
        #: warm and must not be quoted as draft-warm numbers.
        "warm_spawn_before_send": (
            all(row.get("warm_spawned_before_send") for row in draft_rows) if draft_rows else None
        ),
        "invalid_runs": [
            row["run"] for row in draft_rows if not row.get("warm_spawned_before_send")
        ],
        #: What the running sessions ACTUALLY had declared, and whether that
        #: matches the arm's own name for itself. A `mcpx` arm that declares
        #: nothing is the control arm wearing the treatment's label, which is the
        #: failure Q1 found; this is the field that makes it visible instead of
        #: plausible.
        "mcp_servers_seen": declared,
        "declaration_correct": all(correct) if correct else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="new conversations to time")
    parser.add_argument(
        "--pre-engage",
        choices=("off", "keystroke", "open"),
        default="off",
        help=(
            "off: today's flow (create -> send). keystroke/open: mint a draft, hold its "
            "events stream, beat /watch with the id the stream published, fire /warm, "
            "wait --think, then create with the draft id and send — the pre-engaged "
            "pane. The rig HOLDS THE BRIDGE (the stream) for the whole run: a warm "
            "whose only user is its own request is cancelled on return. open is the O4 "
            "variant — the real pane warms at PANE OPEN, so --think there is the whole "
            "pane-open-to-send span; the rig sequence is otherwise identical. A draft "
            "run whose pre-send census shows no runtime child is INVALID and exits 4"
        ),
    )
    parser.add_argument(
        "--think",
        type=float,
        default=0.0,
        help=(
            "seconds between the draft's warm and its create+send (the typing time). "
            "0 is the paste+send worst case; 3 is 'typed for three seconds'"
        ),
    )
    parser.add_argument(
        "--mcp-variant",
        type=str,
        default="none",
        choices=("none", "mcpx"),
        help="declare an MCP server (mcpx: HTTP on a closed port) or none",
    )
    parser.add_argument("--json", type=str, default="", help="write raw results here")
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="free-form note recorded WITH the numbers (see bench_cold_engage.py)",
    )
    parser.add_argument(
        "--measured-tree",
        type=str,
        default="",
        help=(
            "commit whose local_operator/ this run measures (default: the "
            "worktree HEAD). VERIFIED against the subtree on disk — the run is "
            "refused when they disagree (review round 2, R2-1)"
        ),
    )
    parser.add_argument(
        "--standby",
        choices=("off", "on"),
        default="off",
        help=(
            "off: LOP_RUNTIME_STANDBY_DISABLED=1, the fork+import cold spawn. on: the "
            "daemon keeps its pre-imported standby (session/runtime/standby.py). "
            "EVERY run is reported and NOTHING waits for the warm: a send that "
            "arrives before the standby is warm takes the cold path, which is the "
            "run-1 the operator actually gets after a boot — read the runs after "
            "that as the steady state"
        ),
    )
    parser.add_argument(
        "--warm-wait",
        type=float,
        default=0.0,
        help=(
            "seconds to wait after the daemon starts, before the first timed send "
            "(default 0: no wait). The standby's readiness is not observable from "
            "another process by design, so this is the only honest way to ask for "
            "the steady state, and the number is only meaningful when it exceeds "
            "the warm measured on this host (51.4 s at load ~120, and "
            "bench_standby_engage.py reports it per pass). With 0 the timed sends "
            "race the warm and the first one is the cold send the operator gets "
            "after a boot"
        ),
    )
    args = parser.parse_args()
    # Refuse BEFORE measuring (and the fields are re-derived at the end, so a
    # subtree that moved under the run is caught too): a campaign that records a
    # commit it did not measure is worse than one that never ran.
    try:
        bench_tree.describe(args.measured_tree)
    except bench_tree.MeasuredTreeError as exc:
        print(f"REFUSING TO MEASURE: {exc}", flush=True)
        return 2

    import httpx

    root = Path(tempfile.mkdtemp(prefix="lop-cold-send-"))
    config_dir = _seed_config(root)
    workspace = root / "workspace"
    workspace.mkdir()
    # The workspace IS the session cwd, so the declaration goes here (Q1).
    _write_mcp_config(workspace, args.mcp_variant)
    token = secrets.token_hex(32)
    port = _free_port()
    base = f"http://127.0.0.1:{port}"

    saved = {
        k: os.environ.get(k)
        for k in ("HOME", "LOCAL_OPERATOR_CONFIG_DIR", "PYTHONPATH", "LOP_RUNTIME_STANDBY_DISABLED")
    }
    _strip_inherited()
    # The daemon below (``lop serve``) is the process whose machine-wide feed
    # raises desktop banners, and this rig drives a real send through it.
    suppress_notifications_for_process("cold-send benchmark driving the real CLI")
    os.environ["HOME"] = str(root)
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(config_dir)
    os.environ["LOCAL_OPERATOR_DESKTOP_TOKEN"] = token
    os.environ.pop("LOCAL_OPERATOR_DESKTOP_ORIGINS", None)
    if args.standby == "off":
        os.environ["LOP_RUNTIME_STANDBY_DISABLED"] = "1"
    os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])

    server_log = root / "serve.log"
    rows: list[dict[str, Any]] = []
    proc: subprocess.Popen[bytes] | None = None
    try:
        with server_log.open("wb") as log:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "local_operator.cli",
                    "serve",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                env=dict(os.environ),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            headers = {"Authorization": "Bearer " + token}
            with httpx.Client(base_url=base, headers=headers, timeout=300.0) as client:
                _wait_for_health(client, base, proc)
                print(f"  server up on {base} (pid {proc.pid})", flush=True)
                if args.warm_wait:
                    print(
                        f"  waiting {args.warm_wait:.0f} s for the daemon's standby "
                        f"to warm (children now: {_standby_children(proc.pid)})",
                        flush=True,
                    )
                    time.sleep(args.warm_wait)
                for index in range(args.runs):
                    row = _one_run(
                        client,
                        base,
                        workspace,
                        config_dir,
                        index,
                        args.mcp_variant,
                        pre_engage=args.pre_engage,
                        think=args.think,
                        daemon_pid=proc.pid,
                        headers=headers,
                    )
                    row["standby_children"] = _standby_children(proc.pid)
                    rows.append(row)
                    detail = ""
                    if row["pre_engage"] != "off":
                        detail = (
                            f", mint {row['mint_ms']}, warm {row['warm_ms']}, "
                            f"create {row['create_ms']} ms, spawn-before-send: "
                            f"{'yes' if row['warm_spawned_before_send'] else 'NO — INVALID'}"
                        )
                    print(
                        f"  run {index + 1}/{args.runs}: first send = "
                        f"{row['first_send_ms']} ms  (create {row['create_status']}, "
                        f"send {row['send_status']}, load {row['loadavg']}, "
                        f"mcp declared: {row['mcp_declared_servers'] or 'none'}{detail})",
                        flush=True,
                    )
    finally:
        # The daemon's standby is a detached process of its own (a standby must
        # outlive the host that warmed it), so it is ended here by exact pid: the
        # holder of THIS root's standby lock, which no other process can hold.
        # The daemon's standby is a detached child of the daemon, and it exits on
        # its own when the daemon's end of the private channel closes - so ending
        # the daemon is most of the reaping. The rest is by exact pid, taken from
        # the daemon's own child list while it is still alive.
        if proc is not None and proc.poll() is None:
            for pid in _standby_children(proc.pid):
                try:
                    os.kill(pid, 15)
                except OSError:
                    pass
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    stats = _summarize(rows)
    # Re-verified at the END as well as before the first run: a subtree that moved
    # under a long campaign is exactly the case the field exists to catch.
    tree = bench_tree.describe(args.measured_tree)
    stats["rev"] = tree["rev"]
    stats["worktree_head"] = tree["worktree_head"]
    stats["measured_tree_verified"] = tree["verified"]
    stats["label"] = args.label
    stats["standby"] = args.standby
    print("\n--- first POST /messages wall time (ms) ---")
    print(f"  mcp variant: {args.mcp_variant}  standby: {args.standby}")
    if args.pre_engage != "off":
        print(
            f"  pre-engage: {args.pre_engage}  think: {args.think:.1f} s  "
            f"warm-spawn-before-send: {stats.get('warm_spawn_before_send')}  "
            f"invalid runs: {stats.get('invalid_runs') or 'none'}"
        )
    if args.standby == "on":
        print(
            "  every run is reported: a send that arrives before the daemon's standby\n"
            "  finishes warming takes the cold path, and the per-run numbers are what\n"
            "  shows it (readiness is not externally observable by design)."
        )
    print(bench_tree.format_banner(tree))
    if args.label:
        print(f"  label: {args.label}")
    for key, value in stats.items():
        print(f"  {key:<8} {value}")
    for row in rows:
        print(f"\n  run {row['run']} POST /v1/desktop/sessions -> {json.dumps(row['create_body'])}")
        print(f"  run {row['run']} POST /messages -> {json.dumps(row['send_body'])}")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"summary": stats, "rows": rows}, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json}")

    if server_log.exists():
        print(f"\n(server log: {server_log} — removed with {root})")
    shutil.rmtree(root, ignore_errors=True)

    if stats.get("declaration_correct") is False:
        # Non-zero, because a driver that only reads the exit status must not
        # treat an arm that measured its own control as a treatment reading
        # (QA round 1, Q1). The numbers are still written above and in --json,
        # so nothing is hidden — this only says they are not what the arm's name
        # claims.
        print(
            "\n  DECLARATION NOT VERIFIED: --mcp-variant "
            f"{args.mcp_variant!r} asked for one declaration and the running "
            f"sessions reported {stats.get('mcp_servers_seen') or 'none'}. The arm "
            "measured the wrong thing; see _write_mcp_config for where the "
            "declaration has to live (the session's cwd).",
            flush=True,
        )
        return 3
    if stats.get("invalid_runs"):
        # Non-zero for the same reason the declaration exit above is: a draft
        # warm that was not held (or never fired) leaves an empty pre-send
        # census, and the row it produced timed NO warm — reporting it as a
        # draft-warm number would be measuring a cancellation (spec §3.1).
        print(
            "\n  DRAFT RUNS INVALID: the pre-send census shows no runtime child for "
            f"runs {stats['invalid_runs']}. A warm whose only bridge user is its own "
            "request is cancelled when that request returns, so those rows timed no "
            "warm at all; check the /events hold and the /watch beat before quoting "
            "them.",
            flush=True,
        )
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
