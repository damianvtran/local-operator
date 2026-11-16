"""Test-only bounded stderr and post-exit witnesses for owned signal-rig children."""

from __future__ import annotations

import json
import os
import platform
import select
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

# --- macOS native crash report (CI only, test-proof only) -----------------------
#
# WHY: a main-5dd e2e child died with SIGSEGV (returncode -11) and the only
# evidence was CPython's faulthandler dump -- a Python stack that says WHERE the
# interpreter was, not WHY the kernel faulted. macOS writes its own
# ``<proc>-<date>.ips`` report (exception type/subtype, faulting thread's native
# frames) to ~/Library/Logs/DiagnosticReports. This reads ONLY that report for
# the OWN child that just died, and attaches a bounded, allow-listed summary to the
# process-proof JSON the CI lane already uploads. It changes no signal, timeout,
# workflow or product code, and a missing/unmatched report is reported as such
# (never a fabricated positive). It does NOT replace or re-capture faulthandler.
#
# Constraints: runs only on a Darwin GitHub-Actions runner (so a developer's
# real home is never read); the directory is resolved from the OS user database
# (a fixture ``HOME`` does not locate CrashReporter); every read is bounded and
# symlink-refusing; only whitelisted fields leave the function (no registers,
# environment, argv, memory, full image paths, or the serve record's token).
NATIVE_FATAL_CODES = (-11,)  # SIGSEGV only: the observed failure, nothing speculative.
NATIVE_MAX_SCANNED = 200  # raw directory entries examined
NATIVE_MAX_CANDIDATES = 50  # .ips regular files recent enough to open
NATIVE_MAX_FILE_BYTES = 256 * 1024
NATIVE_MAX_FRAMES = 30
NATIVE_MAX_OUTPUT_BYTES = 4 * 1024
NATIVE_POLL_BUDGET_S = 10.0  # CrashReporter writes after the exit is observed
_NATIVE_TEXT = 120


def native_report_enabled(
    returncode: int | None, *, system: str, environ: Mapping[str, str]
) -> bool:
    """Only a SIGSEGV on a Darwin GitHub-Actions runner may read DiagnosticReports."""
    return (
        returncode in NATIVE_FATAL_CODES
        and system == "Darwin"
        and environ.get("CI") == "true"
        and environ.get("GITHUB_ACTIONS") == "true"
    )


def os_user_reports_dir() -> Path:
    """The user's DiagnosticReports, from the OS user database, not ``$HOME``.

    Imported lazily so nothing but the guarded CI path ever consults the real user.
    """
    import pwd

    return Path(pwd.getpwuid(os.getuid()).pw_dir) / "Library" / "Logs" / "DiagnosticReports"


def _clip(value: Any) -> str:
    return str(value)[:_NATIVE_TEXT]


def parse_ips_report(text: str, *, pid: int) -> dict[str, Any]:
    """Allow-listed summary of one ``.ips`` text, or ``{"reason": ...}`` when not ours.

    The format is a one-line JSON header followed by a JSON body. Anything torn,
    of another bug_type, another pid, or another process name is a non-match: a
    wrong-pid report must never be returned.
    """
    head, _, rest = text.partition("\n")
    try:
        header = json.loads(head)
        body = json.loads(rest)
    except ValueError:
        return {"reason": "report is not parseable JSON (torn or unknown format)"}
    if not isinstance(header, dict) or not isinstance(body, dict):
        return {"reason": "report has an unexpected shape"}
    if str(header.get("bug_type")) != "309":
        return {"reason": "not a crash report (bug_type != 309)"}
    if body.get("pid") != pid:
        return {"reason": "report is for another pid"}
    if "python" not in str(body.get("procName", "")).lower():
        return {"reason": "report is for another process name"}
    exception = body.get("exception")
    exception = exception if isinstance(exception, dict) else {}
    images = body.get("usedImages")
    images = images if isinstance(images, list) else []
    threads = body.get("threads")
    threads = threads if isinstance(threads, list) else []
    index = body.get("faultingThread")
    thread = threads[index] if isinstance(index, int) and 0 <= index < len(threads) else {}
    raw_frames = thread.get("frames") if isinstance(thread, dict) else None
    raw_frames = raw_frames if isinstance(raw_frames, list) else []
    frames = []
    for frame in raw_frames[:NATIVE_MAX_FRAMES]:
        if not isinstance(frame, dict):
            continue
        image_index = frame.get("imageIndex")
        image = (
            images[image_index]
            if isinstance(image_index, int) and 0 <= image_index < len(images)
            else {}
        )
        offset = frame.get("imageOffset")
        frames.append(
            {
                # Name only: ``path`` would leak the runner's filesystem layout.
                "image": _clip(image.get("name", "") if isinstance(image, dict) else ""),
                "symbol": _clip(frame.get("symbol", "")),
                "offset": offset if isinstance(offset, int) else None,
            }
        )
    return {
        "exception_type": _clip(exception.get("type", "")),
        "signal": _clip(exception.get("signal", "")),
        "subtype": _clip(exception.get("subtype", "")),
        "incident_id": _clip(header.get("incident_id", "")),
        "timestamp": _clip(header.get("timestamp", "")),
        "faulting_thread_frames": frames,
        "frames_total": len(raw_frames),
    }


def bound_native_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Shrink the frame list until the ASCII-escaped JSON fits ``NATIVE_MAX_OUTPUT_BYTES``.

    The drop is recorded (``frames_kept``/``frames_total``) so truncation is visible.
    """
    kept = list(summary.get("faulting_thread_frames", []))
    while True:
        candidate = {**summary, "faulting_thread_frames": kept, "frames_kept": len(kept)}
        if len(json.dumps(candidate, ensure_ascii=True)) <= NATIVE_MAX_OUTPUT_BYTES or not kept:
            return candidate
        kept = kept[: max(0, len(kept) - 1)]


def find_native_report(
    reports_dir: Path,
    *,
    pid: int,
    started_wall: float,
    poll_budget_s: float = NATIVE_POLL_BUDGET_S,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Locate this child's ``.ips`` report; always returns a dict, never raises.

    Matches on the child's own pid, a file mtime at/after its start, bug_type 309
    and a python process name. The wait is a CONDITION poll bounded by
    ``poll_budget_s`` (the report lands shortly after the exit), not a fixed sleep.
    """
    deadline = clock() + poll_budget_s
    outcome = "no matching report"
    while True:
        try:
            outcome = _scan_once(reports_dir, pid=pid, started_wall=started_wall)
        except OSError as error:
            outcome = f"reports directory unavailable: {type(error).__name__}"
        if isinstance(outcome, dict):
            return bound_native_summary({"found": True, **outcome})
        if clock() >= deadline:
            return {"found": False, "reason": outcome}
        sleep(0.5)


def _scan_once(reports_dir: Path, *, pid: int, started_wall: float) -> dict[str, Any] | str:
    if reports_dir.is_symlink() or not reports_dir.is_dir():
        return "reports directory missing or a symlink"
    last = "no matching report"
    scanned = 0
    candidates = 0
    with os.scandir(reports_dir) as entries:
        for entry in entries:
            scanned += 1
            if scanned > NATIVE_MAX_SCANNED or candidates >= NATIVE_MAX_CANDIDATES:
                last = f"no matching report within the scan bound ({scanned - 1} scanned)"
                break
            if not entry.name.endswith(".ips") or entry.is_symlink():
                continue
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if not entry.is_file(follow_symlinks=False) or info.st_mtime < started_wall:
                continue
            candidates += 1
            try:
                fd = os.open(entry.path, os.O_RDONLY | os.O_NOFOLLOW)
            except OSError:
                continue
            with os.fdopen(fd, "rb") as handle:
                if os.fstat(handle.fileno()).st_size > NATIVE_MAX_FILE_BYTES:
                    last = "a candidate report exceeded the size cap"
                    continue
                text = handle.read(NATIVE_MAX_FILE_BYTES + 1).decode("utf-8", errors="replace")
            parsed = parse_ips_report(text, pid=pid)
            if "reason" not in parsed:
                return parsed
            last = parsed["reason"]
    return last


class CapturedChild(subprocess.Popen[bytes]):
    """Drain stderr without backpressure; retain only a bounded diagnostic tail.

    A regular unbounded stderr file can exhaust a runner during a noisy failure;
    PIPE without a reader can stall the shutdown being measured. The reader owns
    a bounded tail file instead. Nothing changes the child's signals or limits.
    """

    # Storage/transport budgets, not timing or pass/fail thresholds. The uploaded
    # JSON uses 4 KiB tails so even JSON's worst-case 6x escaping fits the existing
    # process-proof lane's 96 KiB per-file contract. The larger raw stderr tail
    # is separately allowlisted for upload; never upload arbitrary fixture scratch.
    STDERR_BYTES = 64 * 1024
    REPORT_BYTES = 4 * 1024

    def __init__(self, args: list[str], *, config: Path, **kwargs: Any) -> None:
        self.config = config
        # Wall clock BEFORE the spawn: the native report's mtime can only be later.
        self._started_wall = time.time()
        logs = config / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        super().__init__(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
            **kwargs,
        )
        self.stderr_path = logs / f"runtime-stderr-{self.pid}.stderr"
        self._stop_r, self._stop_w = os.pipe()
        self._capture_error = ""
        self._reader = threading.Thread(target=self._capture, daemon=True)
        self._reader.start()
        self._finished = False

    def _capture(self) -> None:
        assert self.stderr is not None
        tail = b""
        try:
            os.set_blocking(self.stderr.fileno(), False)
            with self.stderr_path.open("wb") as output:
                stopping = False
                while True:
                    if not stopping:
                        ready, _, _ = select.select([self.stderr, self._stop_r], [], [])
                        stopping = self._stop_r in ready
                    try:
                        chunk = os.read(self.stderr.fileno(), self.STDERR_BYTES)
                    except BlockingIOError:
                        if stopping:
                            break
                        continue
                    if not chunk:
                        break
                    tail = (tail + chunk)[-self.STDERR_BYTES :]
                    output.seek(0)
                    output.truncate()
                    output.write(tail)
                    output.flush()
                    if stopping:
                        # An inherited writer may keep the pipe readable after
                        # the owned child exits. Snapshot at most one file-sized
                        # chunk; evidence collection must not await that writer.
                        break
        except OSError as error:
            self._capture_error = f"stderr capture unavailable: {error!r}"
        finally:
            self.stderr.close()
            os.close(self._stop_r)

    def _finish_capture(self) -> None:
        if self._finished:
            return
        # Wake the reader after exit even if a grandchild inherited stderr. It
        # drains already available bytes, then stops without waiting for that
        # unrelated fd owner or adding a sleep to the measured shutdown path.
        try:
            os.write(self._stop_w, b"x")
        except BrokenPipeError:
            pass  # EOF already ended the reader.
        finally:
            os.close(self._stop_w)
        self._reader.join()
        self._finished = True

    def _tail(self, path: Path) -> str:
        try:
            with path.open("rb") as source:
                source.seek(0, os.SEEK_END)
                source.seek(max(0, source.tell() - self.REPORT_BYTES))
                return source.read(self.REPORT_BYTES).decode("utf-8", errors="replace")
        except OSError as error:
            return f"unavailable: {error!r}"

    def shutdown_evidence(self, artifacts: Path | None = None) -> str:
        """Preserve abnormal exits in both assertion text and CI's JSON upload.

        Call only after exit (and before temporary config teardown). Successes
        close the reader but create no failure artifact. Diagnostic I/O failure
        is reported as evidence, never substituted for the original assertion.
        """
        code = self.poll()
        if code is None:
            return ""
        self._finish_capture()
        if code == 0:
            return ""
        stall = self.config / "logs" / f"runtime-stall-{self.pid}.log"
        payload = {
            "pid": self.pid,
            "returncode": code,
            "stderr_path": str(self.stderr_path),
            "stall_path": str(stall),
            "stderr_tail": self._capture_error or self._tail(self.stderr_path),
            "stall_tail": self._tail(stall),
        }
        if native_report_enabled(code, system=platform.system(), environ=os.environ):
            payload["native_report"] = find_native_report(
                os_user_reports_dir(), pid=self.pid, started_wall=self._started_wall
            )
        evidence = json.dumps(payload, ensure_ascii=True, indent=2)
        # The existing always() full-suite upload includes this exact glob.
        # Inline the same payload too: JUnit retains it even if staging fails.
        target = artifacts if artifacts is not None else Path("test-artifacts/process-proof")
        attachments = (
            artifacts / "attachments"
            if artifacts is not None
            else Path("test-artifacts/signal-shutdown")
        ) / str(self.pid)
        try:
            target.mkdir(parents=True, exist_ok=True)
            attachments.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.stderr_path, attachments / self.stderr_path.name)
            if stall.is_file():
                # Keep the complete watchdog dump, not only its inline tail:
                # finalization crashes can put the decisive frame at the top.
                shutil.copyfile(stall, attachments / stall.name)
            path = target / f"signal-shutdown-{self.pid}.json"
            path.write_text(evidence, encoding="utf-8")
            return f"shutdown evidence: {path}; attachments: {attachments}\n{evidence}"
        except OSError as error:
            return f"shutdown evidence artifact unavailable: {error!r}\n{evidence}"
