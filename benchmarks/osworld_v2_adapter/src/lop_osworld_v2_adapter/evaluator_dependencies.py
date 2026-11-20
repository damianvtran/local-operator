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

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

from lop_osworld_v2_adapter.provisioning import ProvisioningError
from lop_osworld_v2_adapter.vendor_bridge import effective_evaluator_path

# Release manifest and exact task bytes from the OSWorld V2.1 task-hash pin.
_V21_RELEASE_SHA256 = "978686683bdf9f946703c1d316cbbd29c0e3ea97396e48d940ee92c597806ecf"
_TASK_025_SOURCE_SHA256 = "57b62cfb8e574a946e50e73530297a07ccde9e4a3233d8cc2a90785fb420fe32"

# A blank image isolates engine/language/format readiness from OCR accuracy.
_OCR_SAMPLE = b"P5\n128 64\n255\n" + b"\xff" * (128 * 64)
_TSV_HEADER = (
    b"level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
    b"left\ttop\twidth\theight\tconf\ttext"
)


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
            "search path, set EVALUATOR_TOOL_PATH to its trusted host directory."
        )

    search_path = effective_evaluator_path(tool_directories)
    trusted = {*os.defpath.split(os.pathsep), *(tool_directories or ())}
    # A relative entry would resolve from a different cwd at score time. The
    # standard worker has no ambient PATH; trust only its defaults and the
    # explicit operator directories that provisioning has already validated.
    if any(not os.path.isabs(entry) for entry in search_path.split(os.pathsep)):
        raise refuse("the evaluator search path is empty or contains a relative entry")
    tool = shutil.which("tesseract", path=search_path)
    if tool is None:
        raise refuse("no runnable executable on the evaluator search path")
    executable = Path(os.path.realpath(tool))
    if str(Path(tool).parent) not in trusted or any(
        candidate.is_relative_to(root)
        for candidate in (Path(tool), executable)
        for forbidden in forbidden_roots
        for root in (forbidden.absolute(), Path(os.path.realpath(forbidden)))
    ):
        raise refuse("the first matching executable is outside the trusted host tool locations")
    if not executable.is_file():
        raise refuse("the first matching executable is not a regular file")

    # Exercise the pinned invocation (including its existing 20 s deadline),
    # not --version: an installed binary can still lack eng or TSV support.
    # Never pass task inputs, evaluator credentials, cwd, or child output on.
    try:
        result = subprocess.run(
            [str(executable), "stdin", "stdout", "-l", "eng", "--psm", "6", "tsv"],
            input=_OCR_SAMPLE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={},
            cwd="/",
            shell=False,
            check=True,
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        raise refuse("the OCR readiness check timed out") from None
    except (OSError, subprocess.CalledProcessError):
        raise refuse("the OCR readiness check could not run successfully") from None
    if not result.stdout.startswith(_TSV_HEADER + b"\n"):
        raise refuse("the OCR readiness check did not return TSV")
