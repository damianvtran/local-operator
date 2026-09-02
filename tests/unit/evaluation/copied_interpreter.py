"""ONE way to build the copied interpreter the real-spawn tests launch.

``AdapterSupervisor.launch`` pins the worker's executable by content hash, so
a test that spawns a real worker needs an interpreter whose bytes it can
pin per run. ``python -m venv --copies`` gives one. Three test hosts
(``adapters/test_launch.py``, ``runner/test_episode_subprocess.py``,
``adapters/osworld/spawn_helpers.py``) used to carry their own copy of this
recipe, and all three had the same latent defect, which is why it now lives
here once.

THE DEFECT. Each copy tried ``sys.executable`` first and then fell through to
``shutil.which("python3")``. On a uv-managed CPython the first candidate's
copy cannot start at all: uv's builds link ``@rpath/libpythonX.Y.dylib`` with
an rpath of ``@executable_path/../lib``, and ``venv --copies`` copies the
executable but not the library, so the copy aborts in dyld. The helper then
silently took the NEXT candidate -- a Homebrew ``python3`` of whatever version
happened to be on PATH -- while the ``.pth`` it wrote still pointed at THIS
interpreter's site-packages. With a 3.14 venv and a 3.14 Homebrew that was
an accident that worked; with a 3.12 venv and a 3.14 Homebrew the worker
imported 3.14's ``pydantic_core`` extension under a 3.12 interpreter and died
before it could answer a handshake, and 16 spawn tests failed with an error
that named none of this.

THE RULES HERE. (1) Only the RUNNING interpreter is ever copied: the ``.pth``
is derived from it (``sysconfig`` purelib plus wherever ``pydantic`` was
imported from), so a copy of any other interpreter is wrong by construction,
not merely risky. (2) A copy that cannot start because its shared libpython
was not carried across is REPAIRED by copying that one library next to it,
which is what uv's own layout expects; a framework or static build never
hits this branch. (3) Before the copy is handed out it must import
``pydantic_core`` -- the compiled extension that fails first on an ABI
mismatch -- through the very ``.pth`` the worker will use, under the same
``-I -s -E`` flags the supervisor spawns with. (4) Any failure raises an
``AssertionError`` that names the interpreter, the step, and the captured
stderr. Skipping is deliberately not an option: a host with no usable copy
gives no real-spawn coverage at all, and a silent skip hides that from CI.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pydantic

# Written into the copied venv's site-packages so the worker can import
# local_operator and pydantic; both test hosts and the OSWorld spawn helper
# rely on this exact filename existing after ``copied_interpreter`` returns.
REPO_PTH_NAME = "_local_operator_repo.pth"

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Mirrors the supervisor's spawn flags (``supervisor.py``: ``-I -s -E -B``) so
# the probe exercises the same import environment the worker will get.
_WORKER_FLAGS = ("-I", "-s", "-E", "-B")


def dependency_roots() -> list[str]:
    """This interpreter's real import roots, for the copied venv's ``.pth``.

    CI installs the project with ``pip install -e`` and never creates a repo
    ``.venv``, so a hardcoded venv path leaves the spawned worker without
    pydantic and it dies instead of skipping. Deriving the roots from the
    running interpreter works under both layouts. Following an actually
    imported third-party dependency stays correct for editable installs whose
    packages live outside purelib.
    """

    roots = [str(_REPO_ROOT)]
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        roots.append(purelib)
    roots.append(str(Path(pydantic.__file__).resolve().parent.parent))
    seen: list[str] = []
    for root in roots:
        if root not in seen and Path(root).is_dir():
            seen.append(root)
    return seen


def site_packages_of(executable: Path) -> Path:
    """The copied venv's site-packages, where adapters and the ``.pth`` go."""

    return next(executable.parent.parent.glob("lib/python*/site-packages"))


def copied_interpreter(venv: Path) -> Path:
    """Build ``venv`` as a ``--copies`` clone of THIS interpreter and prove it.

    Returns the copied ``python3.X`` executable. The venv's site-packages
    already carries ``REPO_PTH_NAME`` on return, and the executable has been
    shown to import ``pydantic_core`` and the adapter API through it.
    """

    base = Path(os.path.realpath(sys.executable))
    shutil.rmtree(venv, ignore_errors=True)
    try:
        subprocess.run(
            [str(base), "-m", "venv", "--without-pip", "--copies", str(venv)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise AssertionError(f"venv --copies of {base} failed: {detail[-800:]}") from error
    executable = next(
        (
            item
            for item in sorted((venv / "bin").glob("python3.*"))
            if item.is_file() and not item.is_symlink()
        ),
        None,
    )
    if executable is None:
        raise AssertionError(f"venv --copies of {base} produced no copied python3.X executable")

    started = _probe(executable, "print('ok')")
    if started.returncode != 0:
        # The one known reason a faithful copy of the running interpreter
        # cannot start: a shared libpython the copy resolves relative to
        # itself. Carry it across and try once more; anything else is a real
        # failure and is reported as one.
        library = _shared_libpython(base)
        if library is None:
            raise AssertionError(
                f"copied interpreter {executable} (from {base}) did not start and the "
                f"base reports no shared libpython to carry across: {started.stderr[-800:]}"
            )
        target = venv / "lib" / library.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(library, target)
        started = _probe(executable, "print('ok')")
        if started.returncode != 0:
            raise AssertionError(
                f"copied interpreter {executable} (from {base}) still did not start after "
                f"copying {library} to {target}: {started.stderr[-800:]}"
            )

    site = site_packages_of(executable)
    (site / REPO_PTH_NAME).write_text("\n".join(dependency_roots()) + "\n")

    # The load-bearing check. pydantic_core is a compiled extension, so it is
    # the first import to fail on an interpreter/site-packages version mismatch
    # -- exactly the failure that used to surface as a dead worker.
    imported = _probe(
        executable,
        "import pydantic_core, local_operator.evaluation.adapters.api; "
        "import sys; print(sys.version_info[:2])",
    )
    if imported.returncode != 0:
        raise AssertionError(
            f"copied interpreter {executable} (from {base}, {sys.version_info[:2]}) cannot "
            f"import pydantic_core through {site / REPO_PTH_NAME}: {imported.stderr[-800:]}"
        )
    return executable


def _probe(executable: Path, code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executable), *_WORKER_FLAGS, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )


def _shared_libpython(base: Path) -> Path | None:
    """The shared library ``base`` links, if it is built ``--enable-shared``.

    Asked of the BASE interpreter rather than this process because
    ``sysconfig`` here describes the venv's view; the config vars are the
    same, but going through the base keeps the answer tied to the file that
    was actually copied.
    """

    query = (
        "import sysconfig; "
        "print(sysconfig.get_config_var('Py_ENABLE_SHARED') or 0); "
        "print(sysconfig.get_config_var('LIBDIR') or ''); "
        "print(sysconfig.get_config_var('INSTSONAME') or '')"
    )
    answer = _probe(base, query)
    if answer.returncode != 0:
        return None
    lines = answer.stdout.strip().splitlines()
    if len(lines) != 3 or lines[0].strip() != "1" or not lines[1] or not lines[2]:
        return None
    library = Path(lines[1]) / lines[2]
    # A framework build's INSTSONAME is ``Python.framework/Versions/X/Python``;
    # its executable resolves it through the framework, not ``@rpath``, so a
    # copy of it started fine and never reaches here. A plain file is the
    # uv/manylinux shape this repair exists for.
    return library if library.is_file() and "/" not in lines[2] else None
