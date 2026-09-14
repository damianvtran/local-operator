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


def _write_mcp_config(root: Path, variant: str) -> None:
    """Declare one MCP server, the way a real machine does.

    `none` declares nothing (the control); `mcpx` points at a closed port, so
    nothing is spawned and the refused connection is immediate — what is left is
    the SDK import and the config parse, i.e. what ANY declared server costs.
    This is the operator's own condition (gitlab, google-workspace) reduced to
    its cheapest form, because the whole point of the measurement is the
    difference between a machine with a server declared and one without.
    """
    if variant == "none":
        return
    entry: dict[str, Any] = {"type": "http", "url": "http://127.0.0.1:1/mcp"}
    (root / ".mcp.json").write_text(
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


def _one_run(
    client: Any, base: str, workspace: Path, config_dir: Path, index: int
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


def _git_rev() -> str:
    """The exact commit under test, recorded next to the numbers (R5).

    A first-send figure is only comparable to another campaign's when the tree
    that produced it is in the artefact; best-effort, so a missing ``git``
    reports ``unknown`` rather than failing a measurement.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [row["first_send_ms"] for row in rows]
    if not values:
        return {}
    return {
        "median": round(statistics.median(values), 1),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
        "n": len(values),
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
    args = parser.parse_args()

    import httpx

    root = Path(tempfile.mkdtemp(prefix="lop-cold-send-"))
    config_dir = _seed_config(root)
    _write_mcp_config(root, args.mcp_variant)
    workspace = root / "workspace"
    workspace.mkdir()
    token = secrets.token_hex(32)
    port = _free_port()
    base = f"http://127.0.0.1:{port}"

    saved = {k: os.environ.get(k) for k in ("HOME", "LOCAL_OPERATOR_CONFIG_DIR", "PYTHONPATH")}
    _strip_inherited()
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
                    row = _one_run(client, base, workspace, config_dir, index)
                    rows.append(row)
                    print(
                        f"  run {index + 1}/{args.runs}: first send = "
                        f"{row['first_send_ms']} ms  (create {row['create_status']}, "
                        f"send {row['send_status']}, load {row['loadavg']})",
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
    rev = _git_rev()
    stats["rev"] = rev
    stats["label"] = args.label
    print("\n--- first POST /messages wall time (ms) ---")
    print(f"  mcp variant: {args.mcp_variant}")
    print(f"  measured tree: {rev[:9]}")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
