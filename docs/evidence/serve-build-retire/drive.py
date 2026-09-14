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
             [--hold-desktop S] [--refuse-matrix] [--matrix-announced]
             [--expect-no-retire --window S] [--readonly-record-dir S]
             [--log PATH] [--revert] [--unreadable-marker] [--third-sha SHA]

    (default)            flip, watch the announcement, show the daemon keeps
                         serving, then the exit and the record removal.
    --token T            use the desktop plane (bearer T) for the requests.
    --hold-desktop S     hold the app's own relay open for S seconds across the
                         flip: the announcement must be readable while the
                         daemon keeps serving and must NOT latch or exit. The
                         relay is dropped afterwards, which is what lets the
                         drain empty and the daemon finish.
    --refuse-matrix      once the refusal is live, request EVERY route that
                         reaches the door and print each answer, plus the
                         spawn-seam lines the daemon logged while it ran (the
                         instrument round 2 asked for: an isolated config root
                         cannot show "no runtime was started", because a refused
                         request and an attempted spawn both leave no runtime
                         record behind).
    --matrix-announced   the SAME matrix while the daemon is merely announced and
                         still admitting (with the relay held, so it cannot
                         latch): the routes answer their own way and the daemon
                         log DOES name the spawn seam, which is what proves the
                         latched run's zero is a measurement rather than a gap.
                         Combine with --refuse-matrix to get both phases from one
                         daemon, which is the contrast the design is about.
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
    --log PATH           the daemon's own log, for the spawn-seam instrument
                         (``--refuse-matrix``/``--matrix-announced``).
    --revert             the announcement is re-read: after it lands, put the
                         install BACK on the build this process loaded, then
                         move it on again, and then on to ``--third-sha``. The
                         daemon must withdraw the handover, keep serving, and
                         re-announce onto whatever is actually on disk. Needs
                         --hold-desktop: with an empty drain it latches 50 ms
                         after announcing, which leaves nothing to observe.
    --unreadable-marker  the fail-closed guard: a marker nobody can READ is not a
                         build to leave for. The new build is written, aged past
                         the settle and ``chmod 000``-d; the daemon must stay.
                         Restoring the mode must then produce the ordinary
                         announcement, latch and exit from the same process.
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


def age_marker(prefix: Path, seconds: float) -> None:
    """Backdate the marker so the SETTLE window cannot be what decides a run.

    ``build_marker_age_s`` reads the file's mtime; a marker written a second ago
    is inside ``BUILD_SETTLE_S`` and would make any run that reported "no
    action" ambiguous between "not settled" and "not a move".
    """
    when = time.time() - seconds
    os.utime(prefix / ".lop-source", (when, when))


def wait_for(predicate: "callable[[], bool]", what: str, timeout: float = 60.0) -> None:
    """Spin until ``predicate`` holds, or fail naming what never happened."""
    deadline = now() + timeout
    while now() < deadline:
        if predicate():
            return
        time.sleep(POLL_S)
    raise AssertionError(f"timed out after {timeout:.0f}s waiting for {what}")


#: The daemon's own words for "the spawn seam was entered". ``session.runtime.launch``
#: logs the engage attempt; the supervisor logs the failure beside it. Review
#: round 2's instrument, which replaced "run/mobile is empty" — that cell could not
#: distinguish a refused request from one that was never tried, because an isolated
#: config root cannot construct a runtime either way.
SEAM_MARKERS = ("engage:", "could not start a runtime", "session.runtime.launch")


def seam_lines(log_path: Path | None, since: int = 0) -> list[str]:
    """The spawn-seam lines the daemon has logged, after the first ``since``."""
    if log_path is None:
        raise SystemExit("--refuse-matrix/--matrix-announced need --log PATH")
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return []
    lines = [line for line in text.splitlines() if any(m in line for m in SEAM_MARKERS)]
    return lines[since:]


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
        "matrix_announced": False,
        "leave_latched": False,
        "expect_no_retire": False,
        "readonly_s": 0.0,
        "window": 20.0,
        "cwd": str(Path.cwd()),
        "log": None,
        "revert": False,
        "revert_to": None,
        "unreadable_marker": False,
        "third_sha": None,
    }
    for index, item in enumerate(argv[3:]):
        value = argv[4 + index] if len(argv) > 4 + index else None
        if item == "--token":
            options["token"] = value
        elif item == "--hold-desktop":
            options["hold_s"] = float(value or 0)
        elif item == "--refuse-matrix":
            options["matrix"] = True
        elif item == "--matrix-announced":
            options["matrix_announced"] = True
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
        elif item == "--log":
            options["log"] = Path(value or "")
        elif item == "--revert":
            options["revert"] = True
        elif item == "--revert-to":
            options["revert_to"] = value
        elif item == "--third-sha":
            options["third_sha"] = value
        elif item == "--unreadable-marker":
            options["unreadable_marker"] = True
    return record_path, prefix, new_sha, options


def main() -> int:
    record_path, prefix, new_sha, options = _args()
    token = options["token"]

    if (options["matrix"] or options["matrix_announced"]) and options["log"] is None:
        print(
            "FAIL: --refuse-matrix/--matrix-announced need --log PATH (the spawn-seam "
            "instrument reads the DAEMON's log; there is no other way to tell an "
            "attempted spawn from a refused request in an isolated config root)"
        )
        return 1

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

    if options["unreadable_marker"]:
        # THE FAIL-CLOSED GUARD (QA round 2, OBS-1). The installer has written the new
        # build and the marker is aged past the settle, so nothing here is a
        # "not settled yet" measurement: the READ permission is taken away, and the
        # daemon must not announce. `update.source_ref` answers "" for a marker it
        # cannot read, so the stamp on disk is version-only and differs from the boot
        # stamp by the ref ALONE — which is exactly the same-version-rebuild case the
        # ref exists to disambiguate. Announcing here is leaving for a build this
        # process could not read; restoring the mode must then produce the ordinary
        # announcement from the same process, which is what makes this a guard rather
        # than a disabled watch.
        age_marker(prefix, 600)
        os.chmod(prefix / ".lop-source", 0o000)
        print(
            "\n--- the new build is on disk, aged past the settle, and UNREADABLE "
            "(chmod 000) ---"
        )
        deadline = t0 + options["window"]
        checks = 0
        while now() < deadline:
            time.sleep(SAMPLE_S)
            checks += 1
            live = read_record(record_path) or {}
            status, _ = health()
            print(
                f"t={now() - t0:6.3f}s check {checks}: pid_alive={alive(pid)} "
                f"getting_health={status} retiring_from={live.get('retiring_from')!r} "
                f"retiring_to={live.get('retiring_to')!r}"
            )
            assert (
                live.get("retiring_to") == ""
            ), "the daemon announced a handover onto a build it could not read"
        os.chmod(prefix / ".lop-source", 0o644)
        print(
            f"--- t={now() - t0:.3f}s: chmod 644 on the SAME file (still the new build); "
            "the ordinary announcement is expected from here ---"
        )

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
    announced_matrix_done = False
    reverted = False
    seam_seen = 0
    hold_deadline = t0 + options["hold_s"] if relay is not None else 0.0
    while now() - t0 < EXIT_TIMEOUT_S:
        live = read_record(record_path)
        if live is None:
            break
        if announce_at is None and live.get("retiring_to"):
            announce_at = now()
            print(f"\n--- t={announce_at - t0:.3f}s: the record announces the handover ---")
            print(json.dumps(live, indent=2, sort_keys=True))
            print("(the daemon is STILL SERVING: the announcement is a notice, not a refusal)")

        if options["matrix_announced"] and announce_at is not None and not announced_matrix_done:
            # PHASE 1 OF THE CONTRAST, on the same daemon and the same routes as the
            # latched matrix below: announced, the daemon is the only place its client
            # can work, so these requests are answered on their merits — and the daemon
            # log NAMES the spawn seam while they run, which is what proves the latched
            # phase's zero is a measurement rather than a gap (MINOR-1).
            announced_matrix_done = True
            before = len(seam_lines(options["log"]))
            rows_announced = _matrix(base, token, session_id, options["cwd"], create, tag="a")
            _print_matrix(
                "refusal matrix, phase 1: merely ANNOUNCED (every route answered)", rows_announced
            )
            time.sleep(1.0)  # the engage attempt is logged by the runtime's own task
            after = len(seam_lines(options["log"]))
            _print_seam(before, after, "ANNOUNCED", options["log"])
            seam_seen = after
            assert after > before, (
                "the announced matrix never entered the spawn seam, so the latched "
                "phase's zero lines would prove nothing (this is the assertion "
                "MINOR-1 asked for, in the direction that has to be able to fail)"
            )
            if not options["revert"]:
                # The relay goes on the next tick, so the drain can empty and the
                # latched phase (phase 2) follows. When the reversal is what this run
                # is about, the deadline is left where it is: that sequence needs the
                # daemon held across it, and it drops the relay itself at the end.
                hold_deadline = now()

        if options["revert"] and announce_at is not None and not reverted:
            # THE ANNOUNCEMENT IS RE-READ (MINOR-2), on a real daemon and across a real
            # marker: back to the boot build (a rolled-back or superseded `lop-update`
            # — the process is the right one after all), then forward again, then on to
            # a third build. The relay is still held, so the daemon cannot latch out
            # from under the observation; with an empty drain it latches one check
            # after announcing and there would be nothing to see.
            reverted = True
            if not options["revert_to"]:
                raise SystemExit("--revert needs --revert-to SHA (the boot build's)")
            print(
                f"\n--- t={now() - t0:.3f}s: putting the install BACK on the boot build "
                f"({options['revert_to'][:7]}) ---"
            )
            flip(prefix, options["revert_to"], "rolled-back")
            wait_for(
                lambda: not (read_record(record_path) or {}).get("retiring_to"),
                "the withdrawal of the handover",
            )
            live_reverted = read_record(record_path) or {}
            print(
                f"record after the reversion: retiring_from="
                f"{live_reverted.get('retiring_from')!r} "
                f"retiring_to={live_reverted.get('retiring_to')!r}"
            )
            status, body = create("77777777-7777-7777-7777-777777777777")
            print(f"create after the withdrawal: HTTP {status} {body.strip()[:120]}")
            assert status == 200, "a withdrawn handover refused new work"
            print("--- moving the install ON, first to the build it announced, then further ---")
            flip(prefix, new_sha, "new-build")
            age_marker(prefix, 600)
            wait_for(
                lambda: (read_record(record_path) or {})
                .get("retiring_to", "")
                .endswith(new_sha[:7]),
                "the re-announcement onto the build on disk",
            )
            print(
                f"re-announced: retiring_to="
                f"{(read_record(record_path) or {}).get('retiring_to')!r}"
            )
            if options["third_sha"]:
                flip(prefix, options["third_sha"], "third-build")
                age_marker(prefix, 600)
                wait_for(
                    lambda: (read_record(record_path) or {})
                    .get("retiring_to", "")
                    .endswith(options["third_sha"][:7]),
                    "the record to name the third build",
                )
                print(
                    "moved on again: retiring_to="
                    f"{(read_record(record_path) or {}).get('retiring_to')!r}"
                )
            hold_deadline = now()  # the observation is done: let the drain empty
            print(f"\n--- t={now() - t0:.3f}s: the relay is dropped and the daemon may finish ---")

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
                    before = len(seam_lines(options["log"]))
                    _print_matrix(
                        "refusal matrix, phase 2: LATCHED (every route that reaches the door)",
                        _matrix(base, token, session_id, options["cwd"], create, tag="b"),
                    )
                    time.sleep(1.0)
                    after = len(seam_lines(options["log"]))
                    _print_seam(before, after, "LATCHED", options["log"])
                    assert after == before, (
                        "a latched daemon entered the spawn seam: "
                        f"{[line.strip() for line in seam_lines(options['log'])[before:after]]}"
                    )
                    if seam_seen:
                        print(
                            f"  (the same instrument named the seam "
                            f"{seam_seen} time(s) while the daemon was merely ANNOUNCED, "
                            "so this zero is a measurement)"
                        )
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


def _matrix(
    base: str,
    token: str,
    session_id: str,
    cwd: str,
    create: "callable[[str], tuple[int, str]]",
    tag: str = "a",
) -> list[tuple[str, int, str]]:
    """Request EVERY route that reaches the door, and record each answer.

    THE SET IS DERIVED, not remembered. ``DesktopSessions.session()`` is the one
    place a desktop route obtains a bridge and the one place a bridge is built, so
    the routes that can admit or start work are exactly the routes that go through
    it. ``tests/unit/server/test_serve_retire.py`` walks the three desktop routers
    for that set and fails if a route reaching the door has no row in its matrix;
    this is the same set as ordinary requests, so what is on the record here is the
    ANSWER of every one of them rather than an argument about the list.

    Round 2's MAJOR-1 was measured against a five-row version (``create``,
    ``/warm``, ``/messages``, ``/commands``, ``/answers``) while ``/mcp``,
    ``/credentials``, ``/fork``, ``/asides`` and ``/adopt`` reached
    ``bind_runtime()`` on the same latched daemon. Every one of those answers below,
    beside the reads that go through the door as well (``/history``, the snapshot,
    ``/events``, ``/failovers``, ``/command-entities``, ``/skills``, ``/mcp`` GET)
    and the attachment route (``/watch``) — reads included, because a
    session-scoped answer can only come from the build the daemon is leaving.

    ``tag`` is the last hex digit of every request id, so the two phases of one
    daemon use ids a real client would not have reused: a request id already
    claimed (the receipt journal and the aside store both key on it) answers 409
    whatever the daemon's state, and that is a fact about the id rather than about
    admission. The one row that checks it deliberately — the duplicate guard — is
    marked in its label.
    """
    aside_id = "abcdef01-2345-6789-abcd-ef0123456789"

    def rid(n: int) -> str:
        # 8-4-4-4-12 hex, the route's own `RequestID` pattern, with the last digit of
        # each group of ids carrying the phase tag: a request id already claimed
        # answers 409 whatever the daemon's state, so the two phases of one daemon
        # must not share ids (the unit matrix reuses one id because each of its rows
        # runs in its own process).
        return f"01234567-89ab-cdef-0123-456789abc{n}{tag}{tag}"

    rows: list[tuple[str, int, str]] = []
    session_path = f"/v1/desktop/sessions/{session_id}"

    def call(
        label: str, method: str, path: str, payload: dict | None = None, budget: float = 10.0
    ) -> None:
        status, body = request(
            method, f"{base}{path}", token=token, payload=payload, timeout=budget
        )
        rows.append((label, status, body))

    def on_session(
        label: str, method: str, suffix: str, payload: dict | None = None, budget: float = 10.0
    ) -> None:
        call(label, method, session_path + suffix, payload, budget)

    status, body = create(rid(0))
    rows.append(("POST /v1/desktop/sessions (create)", status, body))

    call(
        "GET  /v1/desktop/skills (session in the query)",
        "GET",
        f"/v1/desktop/skills?session_id={session_id}",
    )
    on_session("GET  .../{id}/mcp (a session route)", "GET", "/mcp")
    on_session("POST .../{id}/mcp", "POST", "/mcp", {"action": "list"})
    on_session(
        "POST .../{id}/credentials", "POST", "/credentials", {"action": "list", "key": "TEST_KEY"}
    )
    on_session("POST .../{id}/fork", "POST", "/fork", {"request_id": rid(1), "message": ""})
    on_session("POST .../{id}/asides", "POST", "/asides", {"request_id": rid(2), "text": "hello"})
    # The one row whose request id was ALREADY claimed (in phase 1, or by the row
    # above it): a duplicate is a client error whatever the daemon's state, so once
    # the id is claimed this answers 409 — the label says which, rather than letting
    # a 409 read as a hole in the refusal.
    on_session(
        "POST .../{id}/asides (a REPEATED id: 409 once claimed)",
        "POST",
        "/asides",
        {"request_id": "01234567-89ab-cdef-0123-456789abc2aa", "text": "hello"},
    )
    on_session(
        "POST .../{id}/asides/{aside}/adopt",
        "POST",
        f"/asides/{aside_id}/adopt",
        {"request_id": rid(3), "confirmed": True},
    )
    call(
        "POST /v1/desktop/stop (a session route)",
        "POST",
        "/v1/desktop/stop",
        {"request_id": rid(4), "targets": [session_id], "confirmed": True},
    )
    on_session("GET  .../{id} (snapshot)", "GET", "")
    on_session("GET  .../{id}/history", "GET", "/history")
    on_session("GET  .../{id}/failovers", "GET", "/failovers")
    on_session("GET  .../{id}/command-entities", "GET", "/command-entities?command=compact")
    # A standing stream, so it gets a read budget of its own: announced-and-admitted
    # this call never returns (the observation IS the timeout), and latched it returns
    # the typed 503 immediately.
    on_session("GET  .../{id}/events (the app relay)", "GET", "/events", budget=3.0)
    on_session(
        "POST .../{id}/messages",
        "POST",
        "/messages",
        {"request_id": rid(5), "text": "hello"},
        budget=6.0,
    )
    on_session(
        "POST .../{id}/commands",
        "POST",
        "/commands",
        {"request_id": rid(6), "command": "compact", "args": ""},
    )
    on_session(
        "POST .../{id}/answers",
        "POST",
        "/answers",
        {"epoch": "epoch", "request_id": rid(7), "approved": True},
    )
    on_session(
        "POST .../{id}/watch",
        "POST",
        "/watch",
        {"subscription_id": "0" * 32, "visible": False, "can_notify": False},
    )
    on_session("POST .../{id}/warm", "POST", "/warm", {})
    return rows


def _print_matrix(title: str, rows: list[tuple[str, int, str]]) -> None:
    print(f"\n--- {title} ---")
    for label, status, body in rows:
        print(f"  {label:<46} HTTP {status} {body.strip()[:120]}")


def _print_seam(before: int, after: int, phase: str, log_path: Path | None) -> None:
    """The spawn-seam instrument: what the DAEMON logged, not what a config root kept.

    Round 2's MINOR-1: the cell this replaces argued from ``run/mobile`` being empty,
    which an isolated config root produces whether a request was refused or never
    tried at all — the runtime cannot be constructed without a configured hosting
    platform, so neither answer leaves a record. These are the daemon's own words for
    entering the seam instead, counted across the phase that just ran.
    """
    fresh = seam_lines(log_path)[before:after]
    print(f"  spawn-seam lines the daemon logged during the {phase} matrix: {len(fresh)}")
    for line in fresh[:3]:
        print(f"    | {line.strip()[:150]}")


if __name__ == "__main__":
    sys.exit(main())
