"""Real spawn coverage for the only path that can start an adapter worker.

Every other adapter test drives the protocol in-process. This module builds a
genuine interpreter and a genuine installed distribution, then launches through
``AdapterSupervisor`` exactly as production does, because a manufactured
single-link interpreter fixture previously hid a policy that rejected every real
CPython install.
"""

from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from local_operator.evaluation.adapters.api import AdapterSelector, Handshake
from local_operator.evaluation.adapters.discovery import (
    distribution_digest,
    resolve_launch,
    validate_resolved_launch,
    workspace_digest,
)
from local_operator.evaluation.adapters.supervisor import AdapterSupervisor

RELEASE_DIGEST = "b" * 64
_ADAPTER_SOURCE = """from importlib.metadata import distribution

from local_operator.evaluation.adapters.api import AdapterCapabilities, AdapterMetadata
from local_operator.evaluation.adapters.discovery import distribution_digest


class TinyAdapter:
    def __init__(self, metadata):
        self.metadata = metadata

    async def inspect_requirements(self, params): raise NotImplementedError

    async def prepare(self, params): raise NotImplementedError

    async def reset_start(self, params): raise NotImplementedError

    async def observe(self, params): raise NotImplementedError

    async def execute(self, params): raise NotImplementedError

    async def ask_user_exchange(self, params): raise NotImplementedError

    async def score(self, params): raise NotImplementedError

    async def cleanup(self, params): raise NotImplementedError

    async def close(self, params): raise NotImplementedError


def create():
    installed = distribution("tiny-e2e-adapter")
    return TinyAdapter(
        AdapterMetadata(
            adapter_id="tiny-e2e",
            distribution="tiny-e2e-adapter",
            version="1.0",
            entry_point="tiny_e2e_adapter:create",
            package_digest=distribution_digest(installed),
            release_digest="%s",
            schema_version="1.0",
            capabilities=AdapterCapabilities(
                routes=("computer",), ask_user=False, scoring=False
            ),
        )
    )
""" % RELEASE_DIGEST


def _real_interpreter() -> Path:
    """Copy a working interpreter so its content can be pinned per test run."""

    candidates = [os.path.realpath(sys.executable), shutil.which("python3") or ""]
    for base in candidates:
        if not base or not os.path.exists(base):
            continue
        venv = Path(os.environ["PYTEST_LAUNCH_VENV"])
        shutil.rmtree(venv, ignore_errors=True)
        try:
            subprocess.run(
                [base, "-m", "venv", "--without-pip", "--copies", str(venv)],
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError):
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
            continue
        probe = subprocess.run(
            [str(executable), "-I", "-c", "print('ok')"], capture_output=True, text=True
        )
        if probe.returncode == 0:
            return executable
    pytest.skip("no usable copied interpreter is available on this host")


def _install_adapter(site: Path) -> str:
    repo = Path(__file__).resolve().parents[4]
    site_packages = repo / ".venv" / f"lib/python{sys.version_info.major}.{sys.version_info.minor}"
    (site / "_local_operator_repo.pth").write_text(f"{repo}\n{site_packages / 'site-packages'}\n")
    module = site / "tiny_e2e_adapter.py"
    module.write_text(_ADAPTER_SOURCE)
    info = site / "tiny_e2e_adapter-1.0.dist-info"
    info.mkdir(exist_ok=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: tiny-e2e-adapter\nVersion: 1.0\n")
    (info / "entry_points.txt").write_text(
        "[local_operator.evaluation_adapters.v1]\ntiny-e2e = tiny_e2e_adapter:create\n"
    )
    rows: list[list[str]] = []
    for path in sorted([module, info / "METADATA", info / "entry_points.txt"]):
        data = path.read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append([str(path.relative_to(site)), f"sha256={digest}", str(len(data))])
    rows.append([str((info / "RECORD").relative_to(site)), "", ""])
    target = io.StringIO()
    csv.writer(target, lineterminator="\n").writerows(rows)
    (info / "RECORD").write_text(target.getvalue())
    from importlib.metadata import PathDistribution

    return distribution_digest(PathDistribution(info))


def test_real_running_interpreter_resolves_and_revalidates() -> None:
    """A stock CPython install ships hardlinked names; it must still launch."""

    executable = Path(sys.executable).resolve()
    assert os.lstat(executable).st_nlink >= 1
    workspace = Path(__file__).resolve().parent
    selector = AdapterSelector.model_construct(
        python_executable=str(executable), workspace=str(workspace)
    )
    resolved = resolve_launch(selector)
    assert resolved.executable == str(executable)
    assert resolved.executable_sha256 == hashlib.sha256(executable.read_bytes()).hexdigest()
    validate_resolved_launch(resolved)


@pytest.mark.asyncio
async def test_supervisor_launch_completes_real_handshake_and_reaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTEST_LAUNCH_VENV", str(tmp_path / "venv"))
    executable = _real_interpreter()
    site = next(executable.parent.parent.glob("lib/python*/site-packages"))
    package_digest = _install_adapter(site)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "adapter-release.json").write_text(
        json.dumps({"release_digest": RELEASE_DIGEST}, separators=(",", ":"), sort_keys=True)
    )
    selector = AdapterSelector(
        schema_version="1.0",
        adapter_id="tiny-e2e",
        distribution="tiny-e2e-adapter",
        version="1.0",
        entry_point="tiny_e2e_adapter:create",
        package_digest=package_digest,
        release_digest=RELEASE_DIGEST,
        python_executable=str(executable.resolve()),
        workspace=str(workspace),
        workspace_digest=workspace_digest(str(workspace)),
        route_capability="computer",
    )
    supervisor = AdapterSupervisor.launch(selector)
    try:
        handshake = await supervisor.handshake(timeout=60)
        assert isinstance(handshake, Handshake)
        assert handshake.selector == selector
        assert handshake.metadata.adapter_id == "tiny-e2e"
        assert handshake.workspace_digest == selector.workspace_digest
    finally:
        await supervisor.terminate()
    assert supervisor.process.returncode is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(supervisor.pgid, 0)
    assert await asyncio.to_thread(supervisor.process.poll) is not None
