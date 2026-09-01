"""Shared helpers for the OSWorld adapter's real-spawn and end-to-end tests.

These follow the established ``adapters/test_launch.py`` and
``runner/test_episode_subprocess.py`` pattern: build a genuine copied
interpreter, install the REAL adapter wheel into its site-packages (so
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
import os
import shutil
import subprocess
import sys
from pathlib import Path

from local_operator.evaluation.adapters.discovery import workspace_digest

# tests/unit/evaluation/adapters/osworld/spawn_helpers.py -> repo root is 5
# parents up (osworld -> adapters -> evaluation -> unit -> tests -> root).
ADAPTER_SRC = Path(__file__).resolve().parents[5] / "benchmarks" / "osworld_v2_adapter"
WHEEL_NAME = "lop_osworld_v2_adapter-0.1.0-py3-none-any.whl"


def real_interpreter(venv: Path) -> Path:
    """Copy a working interpreter so its content can be pinned per test run.

    Follows ``test_episode_subprocess._real_interpreter``: candidates are
    tried in turn because not every interpreter can host a ``--copies`` venv.
    Failing loudly (not skipping) keeps a host with no usable interpreter
    from silently dropping the only real-spawn coverage.
    """

    candidates = [
        os.path.realpath(sys.executable),
        shutil.which("python3") or "",
        sys.base_prefix + "/bin/python3",
    ]
    failures: list[str] = []
    for base in candidates:
        if not base or not os.path.exists(base):
            continue
        shutil.rmtree(venv, ignore_errors=True)
        try:
            subprocess.run(
                [base, "-m", "venv", "--without-pip", "--copies", str(venv)],
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            failures.append(f"{base}: venv creation failed ({error})")
            continue
        executable = next(
            (
                item
                for item in sorted((venv / "bin").glob("python3.*"))
                if item.is_file() and not item.is_symlink()
            ),
            None,
        )
        if executable is None:
            failures.append(f"{base}: produced no copied executable")
            continue
        probe = subprocess.run(
            [str(executable), "-I", "-c", "print('ok')"], capture_output=True, text=True
        )
        if probe.returncode == 0:
            return executable
        failures.append(f"{base}: copied interpreter did not run ({probe.stderr[-200:]})")
    raise AssertionError("no usable copied interpreter on this host: " + "; ".join(failures))


def build_adapter_wheel(out_dir: Path) -> Path:
    """Build the real adapter wheel from benchmarks/osworld_v2_adapter.

    The wheel is built once and reused: the tests assert the SHIPPED artifact
    loads, not a recompiled one per test.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    wheel = out_dir / WHEEL_NAME
    if wheel.exists():
        return wheel
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out_dir)],
        cwd=ADAPTER_SRC,
        check=True,
        capture_output=True,
    )
    if not wheel.exists():
        raise AssertionError(f"uv build did not produce {WHEEL_NAME}")
    return wheel


def install_adapter_into_site(site: Path, wheel: Path) -> str:
    """Install the real wheel into a venv's site-packages; return package_digest.

    Uses uv pip install so the RECORD is written by a real installer (hashed
    rows), which is what ``distribution_digest`` verifies. A hand-written
    RECORD would test a fixture, not the real artifact.
    """

    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(_venv_python_from_site(site)),
            "--no-deps",
            str(wheel),
        ],
        check=True,
        capture_output=True,
    )
    from importlib.metadata import PathDistribution

    from local_operator.evaluation.adapters.discovery import distribution_digest

    info = site / "lop_osworld_v2_adapter-0.1.0.dist-info"
    return distribution_digest(PathDistribution(info))


def _venv_python_from_site(site: Path) -> Path:
    # site is <venv>/lib/python3.N/site-packages, so the venv root is three
    # parents up: site-packages -> python3.N -> lib -> <venv>.
    venv = site.parent.parent.parent
    return (
        venv / "bin" / next(p.name for p in (venv / "bin").glob("python3.*") if not p.is_symlink())
    )


def write_workspace(workspace: Path, tasks: dict[str, str], release_digest: str) -> str:
    """Materialise a workspace: adapter-release.json + tasks/, then digest it.

    The workspace must contain no symlinks/hardlinks and exactly the canonical
    manifest bytes. Tasks are written (never linked) so ``workspace_digest``
    pins their exact bytes.
    """

    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "adapter-release.json").write_text(
        json.dumps({"release_digest": release_digest}, separators=(",", ":"), sort_keys=True)
    )
    tasks_dir = workspace / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    for task_id, source in tasks.items():
        (tasks_dir / f"{task_id}.py").write_text(source)
    return workspace_digest(str(workspace))


def release_digest_for(package_digest: str, tasks: dict[str, str]) -> str:
    """Compute the release digest exactly as the build script does, so the
    workspace manifest and the selector agree on the same attestation."""

    manifest = hashlib.sha256("".join(tasks[k] for k in sorted(tasks)).encode()).hexdigest()
    payload = f"lop-osworld-v2-adapter|0.1.0|{package_digest}|osworld-v2-2026.08.08|{manifest}"
    return hashlib.sha256(payload.encode()).hexdigest()
