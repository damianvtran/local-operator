#!/usr/bin/env python3
"""Drive and timestamp one real ``lop serve`` retirement, against a fake install.

Called by ``run.sh`` with a daemon already up. Everything here is deliberately a
plain observation of the running process — the record file, one HTTP request,
and the process's exit — because the claim under test ("a daemon announces its
handover, refuses new work, then leaves") is only meaningful against real
sockets and a real install marker.

Usage:
    drive.py <record-path> <fake-prefix> <new-sha> [options]

Options:
    --token TOKEN       POST a create-session request the moment the retirement
                        is visible, and print the raw status + body.
    --hold-sse SECONDS  Open an SSE job stream before flipping the marker, hold
                        it for SECONDS while sampling the record (the negative
                        case), then release it and watch the retirement.
    --cwd PATH          Working directory for the create-session request.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

POLL_S = 0.02
EXIT_TIMEOUT_S = 120.0


def now() -> float:
    return time.monotonic()


def read_record(path: Path) -> dict | None:
    """The record as a reader sees it, or ``None`` while it is absent."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def flip(prefix: Path, sha: str, label: str) -> None:
    """Rewrite ``.lop-source`` the way ``lop-update`` does, after its installer.

    ``printf '%s %s\\n' "$COMMIT" "$REF"`` is the shell updater's exact shape
    (``update.source_ref`` discriminates on token shape), so this is the same
    file a real update writes — the flip is the update, minus the install.
    """
    (prefix / ".lop-source").write_text(f"{sha} {label}\n")


def post_create(base: str, token: str, cwd: str) -> tuple[int, str]:
    request = urllib.request.Request(
        f"{base}/v1/desktop/sessions",
        data=json.dumps(
            {"request_id": "01234567-89ab-cdef-0123-456789abcdef", "cwd": cwd}
        ).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


class SseHold:
    """One SSE job stream, held open on a reader thread. Term 1 of the predicate."""

    def __init__(self, base: str, job_id: str) -> None:
        self.url = f"{base}/v1/sse/jobs/{job_id}"
        self.opened = threading.Event()
        self.frames: list[str] = []
        self._response = None
        self._thread = threading.Thread(target=self._read, daemon=True)

    def start(self) -> None:
        self._thread.start()
        assert self.opened.wait(10), f"SSE stream never opened: {self.url}"

    def _read(self) -> None:
        try:
            self._response = urllib.request.urlopen(self.url, timeout=None)
            for raw in self._response:
                self.frames.append(raw.decode().rstrip())
                if len(self.frames) == 1:
                    self.opened.set()
        except Exception:  # noqa: BLE001 — closing the stream is the normal end
            self.opened.set()

    def close(self) -> None:
        if self._response is not None:
            self._response.close()


def main() -> int:
    args = sys.argv[1:]
    record_path = Path(args[0])
    prefix = Path(args[1])
    new_sha = args[2]
    token = None
    hold_s = 0.0
    cwd = str(Path.cwd())
    for index, item in enumerate(args[3:]):
        if item == "--token":
            token = args[3 + index + 1]
        elif item == "--hold-sse":
            hold_s = float(args[3 + index + 1])
        elif item == "--cwd":
            cwd = args[3 + index + 1]

    for _ in range(600):
        record = read_record(record_path)
        if record is not None:
            break
        time.sleep(POLL_S)
    else:
        print("FAIL: no record was ever published")
        return 1

    pid = record["pid"]
    base = f"http://{record['host']}:{record['port']}"
    print(f"daemon: pid={pid} port={record['port']} record={record_path}")
    print(f"boot record: version={record['version']} source_ref={record['source_ref']}")
    print(
        "record retiring fields at boot: "
        f"retiring_from={record.get('retiring_from')!r} "
        f"retiring_to={record.get('retiring_to')!r}"
    )

    hold = None
    if hold_s:
        hold = SseHold(base, "00000000-0000-0000-0000-000000000001")
        hold.start()
        print(f"held SSE stream: {hold.url}")
        print(f"  first frames: {hold.frames[:1]}")

    print(f"flipping {prefix}/.lop-source to {new_sha} at t=0.000s")
    t0 = now()
    flip(prefix, new_sha, "new-build")

    if hold is not None:
        # THE NEGATIVE CASE: the update lands while a live stream is attached,
        # and a daemon that retired here would have cut that stream. Every
        # sample across the window must show the record intact and the process
        # alive; the release below is what then lets the retirement happen.
        deadline = t0 + hold_s
        checks = 0
        while now() < deadline:
            time.sleep(1.0)
            checks += 1
            live = read_record(record_path) or {}
            print(
                f"t={now() - t0:6.3f}s check {checks}: pid_alive={alive(pid)} "
                f"retiring_from={live.get('retiring_from')!r} "
                f"retiring_to={live.get('retiring_to')!r}"
            )
        held = read_record(record_path) or {}
        assert held.get("retiring_to") == "", "the daemon retired under a live SSE stream"
        assert alive(pid), "the daemon exited under a live SSE stream"
        print(f"  verdict: {checks} samples across a held stream, no retirement")
        print(f"releasing the SSE stream at t={now() - t0:.3f}s")
        hold.close()

    # Watch for the announcement, then for the exit.
    announce_at = None
    exit_at = None
    posted = None
    while now() - t0 < EXIT_TIMEOUT_S:
        live = read_record(record_path)
        if live is None:
            exit_at = now()
            break
        if announce_at is None and live.get("retiring_to"):
            announce_at = now()
            print(f"\n--- t={announce_at - t0:.3f}s: the record announces the handover ---")
            print(json.dumps(live, indent=2, sort_keys=True))
            if token:
                status, body = post_create(base, token, cwd)
                posted = (status, body)
                print(f"\n--- new-session request while retiring: HTTP {status} ---")
                print(body)
        time.sleep(POLL_S)

    if exit_at is None:
        print("\nFAIL: the daemon never exited within the timeout")
        return 1

    print(f"\n--- t={exit_at - t0:.3f}s: the record is removed (clean exit) ---")
    gone_at = None
    for _ in range(200):
        if not alive(pid):
            gone_at = now()
            break
        time.sleep(POLL_S)
    print(f"record present after exit: {record_path.exists()}")
    print(
        f"process alive after exit: {alive(pid)}"
        + (
            f" (gone at t={gone_at - t0:.3f}s, i.e. {gone_at - exit_at:.3f}s after the "
            "record was removed — the lifespan removes it LAST, one step before uvicorn "
            "returns from serve())"
            if gone_at
            else ""
        )
    )
    print(
        "timings: announce at "
        f"{announce_at - t0 if announce_at else float('nan'):.3f}s, "
        f"exit at {exit_at - t0:.3f}s, "
        f"notice window "
        f"{exit_at - announce_at if announce_at else float('nan'):.3f}s"
    )
    if posted is not None:
        print(f"create-while-retiring answered {posted[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
