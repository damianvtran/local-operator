"""Test-only bounded stderr and post-exit witnesses for owned signal-rig children."""

from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any


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
