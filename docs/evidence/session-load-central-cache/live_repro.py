"""Live-shape reproduction, in ISOLATION: one slow open must not block other sessions.

The operator's symptom is that opening one conversation makes the whole desktop
unusable — "the backend could not complete this request", and the other
conversations stop loading. This script stands a server up from a chosen TREE
against a COPY of two sessions in a throwaway config dir, and measures exactly
that interference: a tiny session's snapshot on its own, and the same request
while a large session's cold open is in flight.

ISOLATION IS THE POINT OF THE SCRIPT, so it is enforced rather than intended:

* the store is COPIED with ``shutil.copytree`` into a temp dir; the operator's
  ``~/.local-operator`` is opened for reading only, and never handed to a server;
* the server runs with ``HOME`` and ``LOCAL_OPERATOR_CONFIG_DIR`` redirected and
  every ``CMUX_*``/``LOP_*`` variable stripped, so it can neither read the
  operator's config nor register against their live store;
* it is started in its own process group and killed by that group, because
  attaching a session may spawn a child runtime;
* the desktop bearer is a token this script invents
  (``LOCAL_OPERATOR_DESKTOP_TOKEN``), so no request here depends on the
  operator's own desktop pairing.

    .venv/bin/python docs/evidence/session-load-central-cache/live_repro.py \\
        --tree ~/workspace/repos/lo-session-load --side after --port 11812
    .venv/bin/python docs/evidence/session-load-central-cache/live_repro.py \\
        --tree ~/workspace/repos/lo-session-load-base --side before --port 11811

Run one per tree and compare: the number of requests the tiny session serves
INSIDE the large session's open window is the measurement, and it is a count
rather than a duration for the reason AGENTS.md gives — a count survives the load
average this machine runs at.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

#: The desktop plane refuses every request without a bearer. The value is
#: arbitrary here because this script starts the server that reads it.
TOKEN = "lo-session-load-probe-token"

#: 261 MB, and it carries no ``desktop.json`` — which is what makes its open take
#: ``locate()``'s checkpoint fallback, the branch that parsed the whole journal.
BIG = "bda7b76d34e0"
#: 642 bytes. Its own read is trivial, so anything it pays for is interference.
SMALL = "f44cf722cd71"

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--tree", type=Path, required=True, help="the worktree to run")
PARSER.add_argument("--side", required=True, help="a label for the output")
PARSER.add_argument("--port", type=int, default=11811)
PARSER.add_argument("--store", type=Path, default=Path.home() / ".local-operator" / "sessions")
PARSER.add_argument("--big", default=BIG)
PARSER.add_argument("--small", default=SMALL)
PARSER.add_argument("--output", type=Path, default=None)


def request(base: str, path: str) -> tuple[int, float, str]:
    """``(status, milliseconds, body)`` for one authorized request."""
    start = time.perf_counter()
    opened = urllib.request.Request(f"{base}{path}", headers={"Authorization": f"Bearer {TOKEN}"})
    try:
        with urllib.request.urlopen(opened, timeout=300) as response:
            body = response.read().decode("utf-8", "replace")
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        status = exc.code
    return status, (time.perf_counter() - start) * 1000, body


def stage(store: Path, big: str, small: str, root: Path) -> tuple[Path, Path]:
    home, config = root / "home", root / "config"
    sessions = config / "sessions"
    sessions.mkdir(parents=True)
    home.mkdir(parents=True)
    for session_id in (big, small):
        shutil.copytree(store / session_id, sessions / session_id)
    return home, config


def server_env(home: Path, config: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("CMUX_", "LOP_")) and key not in {"NO_COLOR", "VIRTUAL_ENV"}
    }
    env.update(
        {
            "HOME": str(home),
            "LOCAL_OPERATOR_CONFIG_DIR": str(config),
            "TERM": "xterm-256color",
            "PYTHONUNBUFFERED": "1",
            "LOCAL_OPERATOR_DESKTOP_TOKEN": TOKEN,
        }
    )
    return env


def start(tree: Path, home: Path, config: Path, port: int, log: Path) -> subprocess.Popen[bytes]:
    handle = log.open("wb")
    proc = subprocess.Popen(
        [
            str(tree / ".venv/bin/python"),
            "-m",
            "local_operator.cli",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(tree),
        env=server_env(home, config),
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited ({proc.returncode}): {log.read_text()[-2000:]}")
        try:
            status, _ms, _body = request(base, "/health")
            if status == 200:
                return proc
        except Exception:  # noqa: BLE001 — not listening yet
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server never became healthy: {log.read_text()[-2000:]}")


def stop(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def _hammer(base: str, small: str, while_running: threading.Thread) -> dict[str, object]:
    """Small-session snapshots for as long as the background request runs.

    The COUNT is the measurement: on the pre-change code it is 1, because the
    other session's request cannot get past the pool-wide lock until the open
    finishes.
    """
    latencies: list[float] = []
    statuses: list[int] = []
    while while_running.is_alive():
        status, ms, _body = request(base, f"/v1/desktop/sessions/{small}")
        statuses.append(status)
        latencies.append(round(ms, 1))
    return {
        "count": len(latencies),
        "median_ms": round(statistics.median(latencies), 1) if latencies else None,
        "max_ms": max(latencies) if latencies else None,
        "statuses": sorted(set(statuses)),
        "samples": latencies[:12],
    }


def probe(args: argparse.Namespace) -> dict[str, object]:
    root = Path(tempfile.mkdtemp(prefix=f"lo-live-{args.side}-"))
    home, config = stage(args.store, args.big, args.small, root)
    log = root / "server.log"
    proc = start(args.tree, home, config, args.port, log)
    base = f"http://127.0.0.1:{args.port}"
    result: dict[str, object] = {
        "side": args.side,
        "tree": str(args.tree),
        "load_average": [round(value, 2) for value in os.getloadavg()],
        "copied_session": args.big,
        "trivial_session": args.small,
    }
    try:
        status, ms, body = request(base, "/health")
        result["health"] = {
            "status": status,
            "ms": round(ms, 1),
            "body": json.loads(body)["result"],
        }
        # Warm the SMALL session first: its own cold open is a different
        # measurement, and the complaint is about the warm one that stops
        # answering while a large session opens.
        status, ms, _body = request(base, f"/v1/desktop/sessions/{args.small}")
        result["small_first_open"] = {"status": status, "ms": round(ms, 1)}
        alone = []
        for _ in range(3):
            status, ms, _body = request(base, f"/v1/desktop/sessions/{args.small}")
            alone.append({"status": status, "ms": round(ms, 1)})
        result["small_alone"] = alone
        result["small_alone_median_ms"] = round(statistics.median(item["ms"] for item in alone), 1)

        for label, path in (
            ("big_cold_snapshot", f"/v1/desktop/sessions/{args.big}"),
            ("big_history", f"/v1/desktop/sessions/{args.big}/history?limit=100"),
        ):
            outcome: dict[str, object] = {}

            def background(target: dict[str, object] = outcome, target_path: str = path) -> None:
                status, ms, body = request(base, target_path)
                target.update(status=status, ms=round(ms, 1), body=body[:200])

            thread = threading.Thread(target=background)
            thread.start()
            result[f"small_during_{label}"] = _hammer(base, args.small, thread)
            thread.join()
            result[label] = outcome
    finally:
        stop(proc)
    return result


if __name__ == "__main__":
    ARGS = PARSER.parse_args()
    payload = probe(ARGS)
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if ARGS.output is not None:
        ARGS.output.write_text(rendered + "\n")
    sys.exit(0)
