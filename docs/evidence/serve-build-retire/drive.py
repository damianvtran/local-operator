#!/usr/bin/env python3
"""Drive and timestamp ONE real ``lop serve`` retirement, against a fake install.

Called by ``run.sh`` with a daemon already up. Everything here is deliberately a
plain observation of the running process — its record file, ordinary HTTP
requests, and the process's exit — because the claims under test ("a daemon
announces its handover early, keeps serving, refuses once nothing is attached,
then leaves") are only meaningful against real sockets and a real install marker.

The two request shapes that matter are the app's OWN:
``GET /v1/desktop/sessions/{id}/events`` (the relay ``DesktopStreamRelay`` holds
for the life of a conversation view) and ``POST /v1/desktop/sessions`` (the
create whose refusal is the typed ``503 daemon-retiring``). A job SSE stream and
a hand-built subscription both look like work from inside the daemon; only the
desktop plane's own shapes decide whether a daemon the app is attached to can
ever roll forward, which is why the evidence holds THIS one and not the job
stream the first version of it held.

Modes (combinable; all against a daemon ``run.sh`` started):

    drive.py <record> <prefix> <new-sha> [--token T] [--cwd P]
             [--hold-desktop S] [--refuse-matrix] [--expect-no-retire --window S]

    (default)            flip, watch the announcement, show the daemon keeps
                         serving, then the exit and the record removal.
    --token T            use the desktop plane (bearer T) for the requests.
    --hold-desktop S     hold the app's own relay open for S seconds across the
                         flip: the announcement must be readable while the
                         daemon keeps serving and must NOT latch or exit. The
                         relay is dropped afterwards, which is what lets the
                         drain empty and the daemon finish.
    --refuse-matrix      once the refusal is live, POST every path that can
                         admit or start work and print each answer.
    --leave-latched      stop once the matrix has been answered, leaving a
                         daemon that is still refusing; run.sh's SIGTERM then
                         owns the exit (the refusal window in that run is
                         deliberately long so the whole matrix fits inside it).
    --expect-no-retire   the negative control: flip and prove NOTHING happens
                         (used for the ``--reload`` child, which must not
                         self-retire).
    --readonly-record-dir S
                         make the record's own directory unwritable for S
                         seconds right after the flip, so the announcement
                         write FAILS: the daemon must keep serving, must not
                         latch and must say so, and must complete the sequence
                         once the directory is writable again. Uses the
                         user-immutable flag, NOT a mode — see the comment where
                         it is applied.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

POLL_S = 0.02
SAMPLE_S = 1.0
EXIT_TIMEOUT_S = 180.0


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


def request(
    method: str,
    url: str,
    *,
    token: str | None = None,
    payload: dict | None = None,
    timeout: float = 10.0,
) -> tuple[int, str]:
    """One HTTP call, returning the status and body whatever it is (4xx included)."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()
    except Exception as error:  # noqa: BLE001 — a dead listener is a measurement too
        return 0, f"{type(error).__name__}: {error}"


class DesktopRelay:
    """``DesktopStreamRelay``'s own request, held open on a reader thread.

    ``GET /v1/desktop/sessions/{id}/events`` is the standing subscription the app
    keeps for as long as a conversation view is mounted: no turn boundary, no
    TTL, and the bridge it acquires is released only when the stream tears down.
    That is term 1 of ``server/retire.py::in_flight``'s desktop group, and the
    reason the daemon must announce before it drains.
    """

    def __init__(self, base: str, token: str, session_id: str) -> None:
        self.url = f"{base}/v1/desktop/sessions/{session_id}/events"
        self.token = token
        self.opened = threading.Event()
        self.frames: list[str] = []
        self._response = None
        self._thread = threading.Thread(target=self._read, daemon=True)

    def start(self) -> None:
        self._thread.start()
        assert self.opened.wait(15), f"the relay never opened: {self.url}"
        assert self.frames, "the relay opened with no frame at all"

    def _read(self) -> None:
        request_ = urllib.request.Request(
            self.url, headers={"Authorization": f"Bearer {self.token}"}, method="GET"
        )
        try:
            self._response = urllib.request.urlopen(request_, timeout=None)
            for raw in self._response:
                self.frames.append(raw.decode().rstrip())
                if self.frames:
                    self.opened.set()
        except Exception:  # noqa: BLE001 — closing the stream is the normal end
            self.opened.set()

    def close(self) -> None:
        if self._response is not None:
            self._response.close()


def _args() -> tuple[Path, Path, str, dict]:
    argv = sys.argv[1:]
    record_path, prefix, new_sha = Path(argv[0]), Path(argv[1]), argv[2]
    options: dict = {
        "token": None,
        "hold_s": 0.0,
        "matrix": False,
        "leave_latched": False,
        "expect_no_retire": False,
        "readonly_s": 0.0,
        "window": 20.0,
        "cwd": str(Path.cwd()),
    }
    for index, item in enumerate(argv[3:]):
        value = argv[4 + index] if len(argv) > 4 + index else None
        if item == "--token":
            options["token"] = value
        elif item == "--hold-desktop":
            options["hold_s"] = float(value or 0)
        elif item == "--refuse-matrix":
            options["matrix"] = True
        elif item == "--leave-latched":
            options["leave_latched"] = True
        elif item == "--expect-no-retire":
            options["expect_no_retire"] = True
        elif item == "--window":
            options["window"] = float(value or 0)
        elif item == "--readonly-record-dir":
            options["readonly_s"] = float(value or 0)
        elif item == "--cwd":
            options["cwd"] = value or options["cwd"]
    return record_path, prefix, new_sha, options


def main() -> int:
    record_path, prefix, new_sha, options = _args()
    token = options["token"]

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

    session_id = None
    relay = None
    if token:
        status, body = request(
            "POST",
            f"{base}/v1/desktop/sessions",
            token=token,
            payload={"request_id": "11111111-1111-1111-1111-111111111111", "cwd": options["cwd"]},
        )
        print(f"session created BEFORE the flip: HTTP {status} {body[:160]}")
        if status != 200:
            return 1
        session_id = json.loads(body)["result"]["session_id"]
        if options["hold_s"]:
            relay = DesktopRelay(base, token, session_id)
            relay.start()
            print(f"held the app's own relay: {relay.url}")
            print(f"  first frames: {relay.frames[:1]}")

    def health() -> tuple[int, str]:
        status, body = request("GET", f"{base}/health", token=token)
        return status, body

    def create(request_id: str) -> tuple[int, str]:
        """The desktop create — the request whose refusal is the typed 503.

        A REFUSED create leaves no receipt (the route refuses before the receipt
        is claimed), so re-issuing the same id is a fresh attempt each time until
        one is admitted; after that it is the receipt's replay. That makes it the
        one request that answers "is this daemon still admitting?" without
        starting any work at all.
        """
        return request(
            "POST",
            f"{base}/v1/desktop/sessions",
            token=token,
            payload={"request_id": request_id, "cwd": options["cwd"]},
        )

    print(f"flipping {prefix}/.lop-source to {new_sha} at t=0.000s")
    t0 = now()
    flip(prefix, new_sha, "new-build")

    if options["readonly_s"]:
        # THE FAILURE INJECTION (review round 1, MINOR-3): a record write that
        # fails. Real rather than simulated — the directory is made unwritable,
        # which is what a full disk or a wrong-owner config root looks like to
        # ``publish`` — and the assertion is that the daemon KEEPS SERVING and
        # does not latch, because a latch on a failed write is a daemon that
        # refuses everything forever and never leaves.
        run_dir = record_path.parent
        # THE FLAG, NOT A MODE, and that is a finding rather than a preference:
        # `run_dir()` re-applies `chmod 0700` on EVERY publish
        # (`session/runtime/registry.py`), so a `chmod 500` on this directory is
        # silently undone by the very write the injection means to fail — the
        # first version of this run measured an announcement landing on schedule
        # under a "read-only" directory. The user-immutable flag cannot be
        # cleared by the process it constrains, so the publish raises
        # PermissionError, which is the fault class this branch exists for (a
        # full disk, a wrong-owner config root). macOS/BSD only; nothing else in
        # this evidence needs it.
        immutable = getattr(stat, "UF_IMMUTABLE", None)
        if immutable is None or not hasattr(os, "chflags"):
            raise SystemExit("--readonly-record-dir needs the BSD/macOS immutable flag")
        os.chflags(run_dir, immutable)
        print(
            f"\n--- made {run_dir} immutable (UF_IMMUTABLE): the announcement write "
            "must now fail ---"
        )
        deadline = now() + options["readonly_s"]
        checks = 0
        while now() < deadline:
            time.sleep(SAMPLE_S)
            checks += 1
            live = read_record(record_path) or {}
            status, _ = create("88888888-8888-8888-8888-888888888888")
            print(
                f"t={now() - t0:6.3f}s check {checks}: pid_alive={alive(pid)} "
                f"create={status} retiring_from={live.get('retiring_from')!r} "
                f"retiring_to={live.get('retiring_to')!r}"
            )
            assert status == 200, "a failed announcement write latched the daemon"
            assert live.get("retiring_to") == "", "the announcement was written after all"
        os.chflags(run_dir, 0)
        print(f"--- {run_dir} writable again at t={now() - t0:.3f}s ---")

    if options["expect_no_retire"]:
        # THE NEGATIVE CONTROL (the ``--reload`` child): a settled change must
        # produce NO announcement, NO exit and NO refusal, because that process
        # serves through a socket its supervisor owns.
        deadline = t0 + options["window"]
        checks = 0
        while now() < deadline:
            time.sleep(SAMPLE_S)
            checks += 1
            live = read_record(record_path) or {}
            status, _ = health()
            print(
                f"t={now() - t0:6.3f}s check {checks}: pid_alive={alive(pid)} "
                f"getting_health={status} "
                f"retiring_from={live.get('retiring_from')!r} "
                f"retiring_to={live.get('retiring_to')!r}"
            )
        live = read_record(record_path) or {}
        assert live.get("retiring_to") == "", "a --reload child announced a handover"
        assert alive(pid), "a --reload child exited mid-serve"
        status, _ = health()
        assert status == 200, f"/health answered {status} after the flip"
        print(f"  verdict: {checks} samples, no announcement, no exit, /health still 200")
        return 0

    announce_at = None
    latch_at = None
    relay_closed_at = None
    admitted_reported = False
    hold_deadline = t0 + options["hold_s"] if relay is not None else 0.0
    matrix_rows: list[tuple[str, int, str]] = []
    while now() - t0 < EXIT_TIMEOUT_S:
        live = read_record(record_path)
        if live is None:
            break
        if announce_at is None and live.get("retiring_to"):
            announce_at = now()
            print(f"\n--- t={announce_at - t0:.3f}s: the record announces the handover ---")
            print(json.dumps(live, indent=2, sort_keys=True))
            print("(the daemon is STILL SERVING: the announcement is a notice, not a refusal)")

        if announce_at is not None and relay is not None and now() < hold_deadline:
            # THE NEGATIVE CASE, and the one that decides this design: the update
            # lands while the app's own relay is attached. Every sample must show
            # the announcement readable, the process alive and a request still
            # being served — the daemon holds the retirement, it does not latch.
            time.sleep(SAMPLE_S)
            status, body = health()
            print(
                f"t={now() - t0:6.3f}s holding: pid_alive={alive(pid)} "
                f"getting_health={status} retiring_to="
                f"{(read_record(record_path) or {}).get('retiring_to')!r} "
                f"relay_frames={len(relay.frames)}"
            )
            continue

        if relay is not None and relay_closed_at is None and now() >= hold_deadline:
            relay_closed_at = now()
            print(f"\n--- t={relay_closed_at - t0:.3f}s: dropping the relay (the view closes) ---")
            relay.close()
            time.sleep(0.5)

        if announce_at is not None and latch_at is None and token:
            status, body = create("22222222-2222-2222-2222-222222222222")
            if status == 503:
                latch_at = now()
                print(f"\n--- t={latch_at - t0:.3f}s: the daemon LATCHES (refuses new work) ---")
                print(f"create while latched: HTTP {status} {body}")
                if options["matrix"]:
                    print("\n--- refusal matrix: every path that can admit or start work ---")
                    matrix_rows = _matrix(base, token, session_id, options["cwd"], create)
                    for label, row_status, row_body in matrix_rows:
                        print(f"  {label:<46} HTTP {row_status} {row_body.strip()[:150]}")
                    print(f"  (run/mobile records: {_runtime_records(record_path)})")
                    if options["leave_latched"]:
                        print(
                            "\n--- this run STOPS here on purpose: the refusal window is "
                            "overridden long so the whole matrix fits inside it; the "
                            "harness' SIGTERM owns the exit ---"
                        )
                        return 0
            elif status == 200 and not admitted_reported:
                # Recorded ONCE, because the point is the contrast: the same
                # request that answers 200 here answers 503 below, on one
                # daemon, with the announcement readable in both cases.
                admitted_reported = True
                print(
                    f"t={now() - t0:6.3f}s still admitting: create -> HTTP 200 "
                    "(announced, not refusing)"
                )
        time.sleep(POLL_S)
    else:
        print("\nFAIL: the daemon neither exited nor removed its record within the timeout")
        return 1

    time.sleep(0.2)  # let the shell reap the child before asking whether it is gone
    exit_at = now()
    print(f"\n--- t={exit_at - t0:.3f}s: the record is removed (clean exit) ---")
    print(f"record present after exit: {record_path.exists()}")
    print(f"process alive after exit: {alive(pid)}")

    def _delta(later: float | None) -> str:
        """A timing, or ``n/a`` where the stage is not observable on this run.

        The ungoverned run cannot see the latch at all (the typed refusal lives
        on the desktop plane, which needs the app's bearer), and printing a
        placeholder number there would read as a measurement.
        """
        return f"{later - t0:.3f}s" if later else "n/a"

    print(
        f"timings: flip -> announce {_delta(announce_at)}, "
        f"latch at {_delta(latch_at)}, record gone at {_delta(exit_at)}"
    )
    if relay is not None:
        print(f"relay frames received while held: {len(relay.frames)}")
    return 0


def _runtime_records(record_path: Path) -> str:
    """The runtime records under the same isolated config root, as a count.

    The claim a refused ``/messages`` or ``/commands`` has to defeat is "no
    runtime was started": every spawn publishes one of these, so an unchanged
    (empty) list is the daemon-side proof that the refusal held at the seam
    rather than after it.
    """
    run_mobile = record_path.parent.parent.parent / "run" / "mobile"
    try:
        names = sorted(p.name for p in run_mobile.iterdir())
    except OSError:
        return "no run/mobile directory at all"
    return f"{len(names)} {names}" if names else "0 (none)"


def _matrix(
    base: str,
    token: str,
    session_id: str,
    cwd: str,
    create: "callable[[str], tuple[int, str]]",
) -> list[tuple[str, int, str]]:
    """POST every path that can admit or start work, and record each answer.

    Each row is one of the surfaces review round 1 found ungated: ``/messages``
    was measured returning 200 and reaching ``admit_prompt`` (whose
    ``_ensure_bound`` is the one call that can START a runtime), and
    ``/commands`` reaches it directly through ``bind_runtime()``.
    """
    rows: list[tuple[str, int, str]] = []
    rows.append(
        ("POST /v1/desktop/sessions (create)", *create("33333333-3333-3333-3333-333333333333"))
    )
    for label, suffix, payload in (
        ("POST /v1/desktop/sessions/{id}/warm", "/warm", {}),
        (
            "POST /v1/desktop/sessions/{id}/messages",
            "/messages",
            {"request_id": "44444444-4444-4444-4444-444444444444", "text": "hello"},
        ),
        (
            "POST /v1/desktop/sessions/{id}/commands",
            "/commands",
            {
                "request_id": "55555555-5555-5555-5555-555555555555",
                "command": "compact",
                "args": "",
            },
        ),
        (
            "POST /v1/desktop/sessions/{id}/answers",
            "/answers",
            {
                "epoch": "epoch",
                "request_id": "66666666-6666-6666-6666-666666666666",
                "approved": True,
            },
        ),
    ):
        status, body = request(
            "POST", f"{base}/v1/desktop/sessions/{session_id}{suffix}", token=token, payload=payload
        )
        rows.append((label, status, body))
    return rows


if __name__ == "__main__":
    sys.exit(main())
