"""Host prerequisites for verified evaluators that hide tool failures.

Pinned OSWorld V2.1 ``task_025`` runs host ``tesseract`` from ``_ocr_region``
and turns a missing or failing binary into empty OCR fields, which the judge
then scores as a valid grade. Here that fault is a refusal at ``reset_start``,
before a provider or guest exists. Upstream files, outputs and scores stay
untouched; the task is never imported or executed.

Entries are keyed by exact task bytes plus release manifest, so an edited task
or another release keeps today's behaviour. Future convergence: the pending
FreeCAD scorer preflight (#4327) should move into this module and reuse the same
trusted search path, rather than add a second hook.
"""

from __future__ import annotations

import base64
import hashlib
import os
import selectors
import shutil
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path

from lop_osworld_v2_adapter.provisioning import ProvisioningError
from lop_osworld_v2_adapter.vendor_bridge import effective_evaluator_path

# Release manifest and exact task bytes from the OSWorld V2.1 task-hash pin.
_V21_RELEASE_SHA256 = "978686683bdf9f946703c1d316cbbd29c0e3ea97396e48d940ee92c597806ecf"
_TASK_025_SOURCE_SHA256 = "57b62cfb8e574a946e50e73530297a07ccde9e4a3233d8cc2a90785fb420fe32"

# Pillow-generated 128x64 white RGB PNG, the same input format as _ocr_region.
# Blank pixels check engine/language/format, not recognition accuracy. Embed
# the bytes to keep Pillow out of pre-allocation and stdin below POSIX PIPE_BUF.
_OCR_SAMPLE = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAIAAAABACAIAAABdtOgoAAAApklEQVR4nO3TMREAMAwDsbT8OScwfpEIePjz292h"
    "88NtBOh5QEyAmAAxAWICxASICRATICZATICYADEBYgLEBIgJEBMgJkBMgJgAMQFiAsQEiAkQEyAmQEyAmAAxAWIC"
    "xASICRATICZATICYADEBYgLEBIgJEBMgJkBMgJgAMQFiAsQEiAkQEyAmQEyAmAAxAWICxASICRATICZATICYADEBY"
    "gJM6wAbzgN9G7i4LgAAAABJRU5ErkJggg=="
)
# TessTsvRenderer + GetTSVText emit the header and one page row for a blank
# image. Bound stdout to that contract, not to the size of a task's real OCR.
_TSV_SAMPLE = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
    "left\ttop\twidth\theight\tconf\ttext\n"
    "1\t1\t0\t0\t0\t0\t0\t0\t128\t64\t-1\t\n"
)
_OCR_OUTPUT_LIMIT = len(_TSV_SAMPLE.encode("utf-8"))
_OCR_TIMEOUT_SECONDS = 20  # The pinned scorer's deadline, not a new tuning knob.


class _OCRProbeOutputLimit(Exception):
    pass


def _run_ocr_probe(tool: str, env: dict[str, str]) -> bytes:
    argv = [tool, "stdin", "stdout", "-l", "eng", "--psm", "6", "tsv"]
    with subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
        cwd="/",
        start_new_session=True,
    ) as child:
        deadline = time.monotonic() + _OCR_TIMEOUT_SECONDS
        assert child.stdin is not None and child.stdout is not None
        output = bytearray()
        try:
            # The whole sample fits in the empty pipe even if a broken tool
            # never reads stdin. If it already exited, keep its exit status.
            with suppress(BrokenPipeError), child.stdin:
                child.stdin.write(_OCR_SAMPLE)
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise subprocess.TimeoutExpired(argv, _OCR_TIMEOUT_SECONDS)
                    chunk = os.read(child.stdout.fileno(), _OCR_OUTPUT_LIMIT + 1 - len(output))
                    if not chunk:
                        break
                    output.extend(chunk)
                    if len(output) > _OCR_OUTPUT_LIMIT:
                        raise _OCRProbeOutputLimit
            returncode = child.wait(timeout=max(0, deadline - time.monotonic()))
        except BaseException:
            # Like the guest command wrapper, kill the owned group before
            # reaping its leader; a helper might still hold the pipe open.
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, argv)
    return bytes(output)


def validate_evaluator_dependencies(
    *,
    task_id: str,
    source: bytes,
    workspace_root: Path,
    tool_directories: tuple[str, ...] | None,
    forbidden_roots: tuple[Path, ...],
) -> None:
    # OCR is a fallback for ink without extractable text, not optional for a
    # comparable score: an OCR failure supplies empty fields to the judge.
    # Match bytes, not a task name or imports guessed from untrusted sources.
    if task_id != "task_025" or hashlib.sha256(source).hexdigest() != _TASK_025_SOURCE_SHA256:
        return
    try:
        release = (workspace_root / "benchmark_release.json").read_bytes()
    except OSError:
        return
    if hashlib.sha256(release).hexdigest() == _V21_RELEASE_SHA256:
        _require_tesseract(tool_directories, forbidden_roots)


def _require_tesseract(
    tool_directories: tuple[str, ...] | None, forbidden_roots: tuple[Path, ...]
) -> None:
    def refuse(reason: str) -> ProvisioningError:
        return ProvisioningError(
            f"OSWorld V2.1 task_025 evaluator needs host tesseract with English OCR and TSV "
            f"support: {reason}. Install it on the scoring worker and, if outside the default "
            "search path, pass --infra EVALUATOR_TOOL_PATH=<dir> for its trusted host directory."
        )

    roots = {
        candidate
        for forbidden in forbidden_roots
        for candidate in (forbidden.absolute(), Path(os.path.realpath(forbidden)))
    }

    def inside_episode(path: Path) -> bool:
        resolved = Path(os.path.realpath(path))
        return any(path.is_relative_to(root) or resolved.is_relative_to(root) for root in roots)

    search_path = effective_evaluator_path(tool_directories)
    trusted = {*os.defpath.split(os.pathsep), *(tool_directories or ())}
    entries = search_path.split(os.pathsep)
    # A host wrapper also uses PATH to look up helpers. Validate the complete
    # path before handing it to the child, not just the winning binary.
    if any(not os.path.isabs(entry) for entry in entries):
        raise refuse("the evaluator search path is empty or contains a relative entry")
    if any(entry not in trusted or inside_episode(Path(entry)) for entry in entries):
        raise refuse(
            "the evaluator search path includes a directory outside the trusted host tool locations"
        )
    tool = shutil.which("tesseract", path=search_path)
    if tool is None:
        raise refuse("no runnable executable on the evaluator search path")
    if inside_episode(Path(tool)):
        raise refuse("the first matching executable is outside the trusted host tool locations")
    if not Path(tool).is_file():
        raise refuse("the first matching executable is not a regular file")

    # PATH is the only declared worker setting needed for host tool lookup.
    # Keep it absent if scoring uses exec's default; never forward ambient
    # settings (including TESSDATA_PREFIX), credentials, or child diagnostics.
    env = {"PATH": search_path} if tool_directories is not None or "PATH" in os.environ else {}
    try:
        result = _run_ocr_probe(tool, env).decode("utf-8")
    except subprocess.TimeoutExpired:
        raise refuse("the OCR readiness check timed out") from None
    except _OCRProbeOutputLimit:
        raise refuse("the OCR readiness check exceeded the blank-image TSV response size") from None
    except subprocess.CalledProcessError as exc:
        raise refuse(
            f"the OCR readiness check failed (status {exc.returncode}); "
            "check that tesseract --list-langs includes eng"
        ) from None
    except OSError:
        raise refuse("the OCR readiness check could not run the host executable") from None
    except UnicodeDecodeError:
        raise refuse("the OCR readiness check did not return UTF-8 TSV") from None
    if result != _TSV_SAMPLE:
        raise refuse("the OCR readiness check did not return TSV for the blank image")
