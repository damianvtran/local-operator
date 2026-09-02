"""Shared helpers for the OSWorld adapter's real-spawn and end-to-end tests.

These follow the established ``adapters/test_launch.py`` and
``runner/test_episode_subprocess.py`` pattern: build a genuine copied
interpreter (``tests.unit.evaluation.copied_interpreter``, shared by all
three), install the REAL adapter wheel into its site-packages (so
``distribution_digest`` verifies the real RECORD), write a real
``adapter-release.json`` and ``tasks/`` corpus into a workspace, and compute
the three digests the selector pins. Only the model is scripted; the worker,
the wheel, the RPC, and the evidence writer are all real.

The module lives in the harness's test tree because it drives the real
``EpisodeRunner`` and ``AdapterSupervisor``, which are harness code. The
adapter wheel itself is built from ``benchmarks/osworld_v2_adapter`` once per
test session (session-scoped) so the per-test cost is a venv, not a build.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from local_operator.evaluation.adapters.api import (
    ADAPTER_SCHEMA_VERSION,
    AdapterSelector,
)
from local_operator.evaluation.adapters.discovery import workspace_digest
from tests.unit.evaluation.copied_interpreter import (
    copied_interpreter,
    site_packages_of,
)

# tests/unit/evaluation/adapters/osworld/spawn_helpers.py -> repo root is 5
# parents up (osworld -> adapters -> evaluation -> unit -> tests -> root).
ADAPTER_SRC = Path(__file__).resolve().parents[5] / "benchmarks" / "osworld_v2_adapter"
WHEEL_NAME = "lop_osworld_v2_adapter-0.1.1-py3-none-any.whl"


def build_adapter_wheel(out_dir: Path) -> Path:
    """Build the real adapter wheel from benchmarks/osworld_v2_adapter.

    Built with THIS interpreter's ``pip``, deliberately not with ``uv``. The
    developer workflow uses ``uv`` (see the adapter README and
    ``make adapter-osworld``), but ``uv`` is not on a stock GitHub runner's
    PATH, and a test that silently depends on the developer's toolchain is a
    test that only runs where it was written. ``pip`` ships with every
    interpreter that can run the suite and produces the same PEP 517 artifact
    from the same ``pyproject.toml``, so the thing under test is unchanged.

    THE BUILD RUNS ON AN ISOLATED COPY, NOT ON THE SOURCE TREE. ``pip wheel
    <srcdir>`` hands the directory to setuptools, which writes ``build/`` and
    ``src/*.egg-info`` INSIDE it. The suite runs under xdist, and the two
    module-scoped fixtures that need a wheel (``test_spawn``,
    ``test_discovery``) land on different workers, so two concurrent builds
    clobber each other's intermediates mid-copy. That is a RACE, not a
    deterministic failure: measured here at 2 of 3 parallel runs failing with
    4 errors each while the serial run stayed green -- which is exactly how a
    single green CI run can be luck rather than proof.

    Copying into a per-build temp directory gives each worker its own tree, so
    no two builds ever write the same path. This is preferred over sharing one
    wheel behind a file lock: a lock adds cross-process failure modes (stale
    locks, timeout tuning) to buy a few seconds of build time, whereas isolation
    removes the shared resource altogether and keeps each build independent.
    ``build/`` and ``*.egg-info`` are excluded from the copy so a developer's
    existing in-tree artifacts cannot leak into it.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    wheel = out_dir / WHEEL_NAME
    if wheel.exists():
        return wheel
    with tempfile.TemporaryDirectory(prefix="osworld-build-") as scratch:
        source = Path(scratch) / "adapter"
        shutil.copytree(
            ADAPTER_SRC,
            source,
            ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__", ".venv"),
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(out_dir), str(source)],
            check=True,
            capture_output=True,
        )
    if not wheel.exists():
        built = sorted(item.name for item in out_dir.glob("*.whl"))
        raise AssertionError(f"pip wheel did not produce {WHEEL_NAME}; built {built}")
    return wheel


def install_adapter_into_site(site: Path, wheel: Path) -> str:
    """Install the real wheel into a venv's site-packages; return package_digest.

    A real installer writes the RECORD with hashed rows, which is exactly what
    ``distribution_digest`` verifies -- a hand-written RECORD would test a
    fixture rather than the shipped artifact.

    ``pip --target`` rather than ``uv pip --python``: the copied venv is built
    ``--without-pip`` (it only has to run the worker), so it cannot install
    into itself, and ``uv`` is unavailable on CI. Installing with this
    interpreter's pip into the target site-packages produces the same installed
    layout and the same hashed RECORD.

    ``--no-compile`` is REQUIRED, not a speed knob. Byte-compiling on install
    appends ``__pycache__/*.pyc`` rows to RECORD with NO hash, and
    ``discovery._record_rows`` refuses any unhashed row as an editable or
    tampered install -- so a compiled install fails the handshake with
    "editable or unhashed adapter distributions are forbidden". It also matches
    what discovery wants: it never loads bytecode, deliberately.
    """

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-compile",
            "--target",
            str(site),
            str(wheel),
        ],
        check=True,
        capture_output=True,
    )
    from importlib.metadata import PathDistribution

    from local_operator.evaluation.adapters.discovery import distribution_digest

    info = site / "lop_osworld_v2_adapter-0.1.1.dist-info"
    return distribution_digest(PathDistribution(info))


def write_workspace(
    workspace: Path,
    tasks: dict[str, str],
    release_digest: str,
    *,
    provider: dict[str, object] | None = None,
) -> str:
    """Materialise a workspace: adapter-release.json + tasks/, then digest it.

    The workspace must contain no symlinks/hardlinks and exactly the canonical
    manifest bytes. Tasks are written (never linked) so ``workspace_digest``
    pins their exact bytes.

    ``provider`` writes the adapter-owned ``adapter-provider.json`` that selects
    the backend. This is how a SPAWNED worker is told to run the fake: the
    supervisor builds the child environment from a closed allowlist, so an env
    var could not reach it, and the selection has to be part of the
    digest-pinned workspace anyway or it would be invisible to the attestation
    that says which code produced the run.
    """

    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "adapter-release.json").write_text(
        json.dumps({"release_digest": release_digest}, separators=(",", ":"), sort_keys=True)
    )
    if provider is not None:
        (workspace / "adapter-provider.json").write_text(
            json.dumps(provider, separators=(",", ":"), sort_keys=True)
        )
    tasks_dir = workspace / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    for task_id, source in tasks.items():
        (tasks_dir / f"{task_id}.py").write_text(source)
    return workspace_digest(str(workspace))


def build_spawnable_adapter(
    tmp_path: Path,
    wheel: Path,
    tasks: dict[str, str],
    *,
    provider: dict[str, object] | None = None,
) -> AdapterSelector:
    """Install the real wheel into a real interpreter and pin a real workspace.

    Returns the selector ``AdapterSupervisor.launch`` needs. Every digest is
    computed from the artifact actually on disk, so a selector built here fails
    the handshake if the wheel or the workspace differs by one byte — which is
    the property that makes the spawned tests evidence rather than decoration.
    """

    # The copied venv has neither local_operator nor pydantic, and the worker
    # imports both before it can answer a handshake; ``copied_interpreter``
    # writes the .pth pointing at this interpreter's real roots and proves
    # the copy can import through it before handing it back.
    executable = copied_interpreter(tmp_path / "venv")
    site = site_packages_of(executable)

    package_digest = install_adapter_into_site(site, wheel)
    release_digest = release_digest_for(package_digest, tasks)
    workspace = tmp_path / "workspace"
    workspace_digest_value = write_workspace(workspace, tasks, release_digest, provider=provider)

    return AdapterSelector(
        # Tracked from the harness constant so a protocol bump fails loudly
        # here rather than as an opaque selector mismatch at handshake.
        schema_version=ADAPTER_SCHEMA_VERSION,
        adapter_id="osworld-v2",
        distribution="lop-osworld-v2-adapter",
        version="0.1.1",
        entry_point="lop_osworld_v2_adapter:create",
        package_digest=package_digest,
        release_digest=release_digest,
        python_executable=str(executable.resolve()),
        workspace=str(workspace),
        workspace_digest=workspace_digest_value,
        route_capability="computer",
    )


def spawn_config(tmp_path: Path) -> Any:
    """Episode config with timeouts sized for a REAL interpreter spawn.

    The in-process default of 5s is ample for a shared-memory adapter but
    marginal here: a real worker must start CPython and import pydantic and
    local_operator before it can answer the handshake. Under the parallel load
    this machine actually runs, that can exceed 5s, and the episode then fails
    at prepare — turning a genuine assertion into a flake about machine speed
    rather than about the adapter. Same reasoning, same numbers as
    ``runner/test_episode_subprocess._subprocess_config``.
    """

    from tests.unit.evaluation.runner.conftest import build_config

    return build_config(
        tmp_path,
        handshake_timeout=60.0,
        prepare_timeout=60.0,
        reset_timeout=60.0,
        step_timeout=60.0,
        score_timeout=60.0,
        cleanup_timeout=60.0,
    )


def release_digest_for(package_digest: str, tasks: dict[str, str]) -> str:
    """Compute the release digest exactly as the build script does, so the
    workspace manifest and the selector agree on the same attestation."""

    manifest = hashlib.sha256("".join(tasks[k] for k in sorted(tasks)).encode()).hexdigest()
    payload = f"lop-osworld-v2-adapter|0.1.1|{package_digest}|osworld-v2-2026.08.08|{manifest}"
    return hashlib.sha256(payload.encode()).hexdigest()
