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
import json
import os
import secrets
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
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
    client: Any, base: str, workspace: Path, config_dir: Path, index: int, variant: str
) -> dict[str, Any]:
    """One new conversation, one first send, both response bodies kept.

    The host load is sampled per run and travels WITH the row, because a first
    send measured at load 10 and one at load 200 are not the same measurement and
    the difference is invisible after the fact (review round 1, R5).
    """
    loadavg = os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan")
    created = client.post(
        "/v1/desktop/sessions",
        json={"request_id": str(uuid.uuid4()), "cwd": str(workspace)},
    )
    created_body = created.json()
    session_id = created_body["result"]["session_id"]

    declared = _declared_servers(client, session_id)
    started = time.perf_counter()
    sent = client.post(
        f"/v1/desktop/sessions/{session_id}/messages",
        json={"request_id": str(uuid.uuid4()), "text": "bench: one word"},
    )
    first_send_ms = round((time.perf_counter() - started) * 1000.0, 1)

    result = {
        "run": index,
        "session_id": session_id,
        "first_send_ms": first_send_ms,
        "loadavg": round(loadavg, 1),
        "variant": variant,
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
    _kill_runtime(config_dir, session_id)
    return result


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [row["first_send_ms"] for row in rows]
    if not values:
        return {}
    declared = sorted({name for row in rows for name in (row.get("mcp_declared_servers") or [])})
    correct = [row.get("mcp_declaration_correct") for row in rows]
    return {
        "median": round(statistics.median(values), 1),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
        "n": len(values),
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

    saved = {k: os.environ.get(k) for k in ("HOME", "LOCAL_OPERATOR_CONFIG_DIR", "PYTHONPATH")}
    _strip_inherited()
    # The daemon below (``lop serve``) is the process whose machine-wide feed
    # raises desktop banners, and this rig drives a real send through it.
    suppress_notifications_for_process("cold-send benchmark driving the real CLI")
    os.environ["HOME"] = str(root)
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(config_dir)
    os.environ["LOCAL_OPERATOR_DESKTOP_TOKEN"] = token
    os.environ.pop("LOCAL_OPERATOR_DESKTOP_ORIGINS", None)
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
                for index in range(args.runs):
                    row = _one_run(client, base, workspace, config_dir, index, args.mcp_variant)
                    rows.append(row)
                    print(
                        f"  run {index + 1}/{args.runs}: first send = "
                        f"{row['first_send_ms']} ms  (create {row['create_status']}, "
                        f"send {row['send_status']}, load {row['loadavg']}, "
                        f"mcp declared: {row['mcp_declared_servers'] or 'none'})",
                        flush=True,
                    )
    finally:
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
    print("\n--- first POST /messages wall time (ms) ---")
    print(f"  mcp variant: {args.mcp_variant}")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
