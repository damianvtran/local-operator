"""Host capabilities required by the pinned OSWorld scorers, not by guest setup."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from lop_osworld_v2_adapter.provisioning import ProvisioningError

# The build's verified benchmark_release.json is stable across adapter wheels;
# adapter-release.json is not (its handshake digest includes the wheel digest).
# OSWorld v2.1 moved these evaluators to host FreeCAD. Earlier releases with
# the same task IDs do not use this resolver.
_FREECAD_RELEASE = "978686683bdf9f946703c1d316cbbd29c0e3ea97396e48d940ee92c597806ecf"
_FREECAD_TASKS = {
    "task_103": ("103", "026db41453d43dc248c577644eab1fa582dac26c608aff88fbe047a42f83e318"),
    "task_104": ("104", "933aca3d06be57f21117435601199a6e8b5b8b51969422f31bc17c56cf31ea49"),
}
# Match the pinned scorer's import deadline, as a total budget rather than
# spending another full scorer deadline on each fallback before allocation.
_IMPORT_BUDGET_S = 20


@dataclass(frozen=True)
class FreeCADRuntime:
    scorer_id: str
    python: Path
    libraries: tuple[Path, ...]

    @property
    def pythonpath(self) -> str:
        return os.pathsep.join(map(str, self.libraries))


def _standard_candidates(scorer_id: str) -> Iterator[FreeCADRuntime]:
    if sys.platform == "linux":
        yield FreeCADRuntime(scorer_id, Path("/usr/bin/python3"), (Path("/usr/lib/freecad/lib"),))
    elif sys.platform == "darwin":
        for apps in (Path("/Applications"), Path.home() / "Applications"):
            prefix = apps / "FreeCAD.app/Contents/Resources"
            python = prefix / "bin/python"
            lib = prefix / "lib"
            if not python.is_file():
                continue
            yield FreeCADRuntime(scorer_id, python, (lib,))
            # The upstream fallback loads the bundle's modules via a matching
            # Python when the signed bundle's launcher cannot run standalone.
            # Do not use the worker PATH: it can include extra evaluator tools.
            for site in sorted(lib.glob("python3.*/site-packages")):
                version = site.parent.name
                if re.fullmatch(r"python3\.\d+", version):
                    fallback = shutil.which(
                        version, path="/usr/bin:/usr/local/bin:/opt/homebrew/bin"
                    )
                    if fallback is not None:
                        yield FreeCADRuntime(scorer_id, Path(fallback), (lib, site))


def _outside(path: Path, forbidden_roots: tuple[Path, ...]) -> bool:
    # Both the lexical path and the resolved target must be outside the
    # workspace and artifacts, including a standard install symlink into them.
    return not any(
        candidate.is_relative_to(root)
        for candidate in (path.absolute(), path.resolve())
        for root in forbidden_roots
    )


def _probe(runtime: FreeCADRuntime, scratch: Path, *, timeout: float) -> bool:
    # Match the pinned scorer's platform import environment, but use a fresh
    # empty cwd/HOME. Neither task code nor task-controlled inputs are present.
    env = {
        "HOME": str(scratch),
        "PYTHONPATH": runtime.pythonpath,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C.UTF-8",
    }
    imports = "import FreeCAD, Part, numpy, scipy"
    if sys.platform == "darwin":
        env.update(
            PATH=f"{runtime.python.parent}:/usr/bin:/bin",
            TMPDIR=str(scratch),
            LC_ALL="en_US.UTF-8",
        )
        imports = "import FreeCAD, Part, Import, numpy, scipy"
    try:
        result = subprocess.run(
            [str(runtime.python), "-c", imports],
            cwd=scratch,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


async def require_runtime(
    workspace: Path, *, task_id: str, source: bytes, forbidden_roots: tuple[Path, ...]
) -> FreeCADRuntime | None:
    declared = _FREECAD_TASKS.get(task_id)
    if declared is None or sys.platform not in {"darwin", "linux"}:
        return None
    try:
        manifest = (workspace / "benchmark_release.json").read_bytes()
    except FileNotFoundError:
        return None
    if hashlib.sha256(manifest).hexdigest() != _FREECAD_RELEASE:
        return None
    scorer_id, expected_source = declared
    if hashlib.sha256(source).hexdigest() != expected_source:
        raise ProvisioningError("Host FreeCAD Python preflight has a non-pinned scorer source")
    # A slow import must not stop the worker from handling RPC cancellation.
    return await asyncio.to_thread(_require_standard_runtime, scorer_id, forbidden_roots)


def _require_standard_runtime(scorer_id: str, forbidden_roots: tuple[Path, ...]) -> FreeCADRuntime:
    roots = tuple(root.resolve() for root in forbidden_roots)
    temp_root = Path(tempfile.gettempdir()).resolve()
    if not _outside(temp_root, roots):
        raise ProvisioningError("Host FreeCAD preflight needs a temp directory outside the episode")
    deadline = time.monotonic() + _IMPORT_BUDGET_S
    with tempfile.TemporaryDirectory(prefix="osworld-freecad-", dir=temp_root) as directory:
        scratch = Path(directory)
        candidates = [
            runtime
            for runtime in _standard_candidates(scorer_id)
            if all(
                path.is_absolute() and os.pathsep not in str(path) and _outside(path, roots)
                for path in (runtime.python, runtime.python.parent, *runtime.libraries)
            )
            and runtime.python.is_file()
        ]
        for index, runtime in enumerate(candidates):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # A stalled launcher must leave time for the matching Python, and
            # candidates after a fast failure can use the time it didn't need.
            if _probe(runtime, scratch, timeout=remaining / (len(candidates) - index)):
                return runtime
    raise ProvisioningError(
        "Host FreeCAD Python is unavailable or incompatible for this OSWorld release; "
        "install a compatible operator-local FreeCAD runtime in a standard location on the "
        "adapter worker host before retrying"
    )


@contextmanager
def score_environment(runtime: FreeCADRuntime | None) -> Iterator[None]:
    if runtime is None:
        yield
        return
    # These two selectors are the pinned scorer's existing explicitly scoped
    # interface. Pin the tested candidate so scoring cannot fall back to a
    # different interpreter (or re-read a different PATH) after the guest ran.
    # The worker serves one RPC at a time; do not install these in prepare or
    # reset, where they would be ambient during guest setup and model turns.
    prefix = f"TASK_{runtime.scorer_id}_FREECAD_"
    overrides = {prefix + "PYTHON": str(runtime.python), prefix + "LIB": runtime.pythonpath}
    previous = {name: os.environ.get(name) for name in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
