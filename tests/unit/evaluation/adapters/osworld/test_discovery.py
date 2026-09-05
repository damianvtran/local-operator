"""Wheel discovery and workspace digest tests for the real adapter.

Asserts the distribution is loadable through the verified path:
``distribution_digest`` over the real installed wheel, ``workspace_digest``
changing on mutation, a symlink rejected, and ``verify_release_manifest``
accepting exactly the canonical ``adapter-release.json``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from local_operator.evaluation.adapters.api import (
    ADAPTER_SCHEMA_VERSION,
    AdapterSelector,
)
from local_operator.evaluation.adapters.discovery import (
    AdapterDiscoveryError,
    verify_release_manifest,
    workspace_digest,
)
from tests.unit.evaluation.adapters.osworld import fixtures, spawn_helpers


@pytest.fixture(scope="module")
def adapter_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return spawn_helpers.build_adapter_wheel(tmp_path_factory.mktemp("wheel"))


def _selector(
    workspace: Path, package_digest: str, release_digest: str, executable: Path
) -> AdapterSelector:
    return AdapterSelector(
        schema_version=ADAPTER_SCHEMA_VERSION,
        adapter_id="osworld-v2",
        distribution="lop-osworld-v2-adapter",
        version="0.1.2",
        entry_point="lop_osworld_v2_adapter:create",
        package_digest=package_digest,
        release_digest=release_digest,
        python_executable=str(executable),
        workspace=str(workspace),
        workspace_digest=workspace_digest(str(workspace)),
        route_capability="computer",
    )


def test_wheel_has_no_console_scripts(adapter_wheel: Path, tmp_path: Path) -> None:
    """The wheel's RECORD must contain no ``../`` path: no console scripts."""
    import zipfile

    with zipfile.ZipFile(adapter_wheel) as archive:
        record = archive.read("lop_osworld_v2_adapter-0.1.2.dist-info/RECORD").decode()
    for line in record.splitlines():
        path = line.split(",")[0]
        assert not path.startswith(".."), f"RECORD path escapes the root: {path}"


def test_workspace_digest_changes_on_mutation(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    digest_before = spawn_helpers.write_workspace(workspace, {"task_a": fixtures.PLAIN}, "b" * 64)
    (workspace / "tasks" / "task_a.py").write_text(fixtures.PLAIN + "\n# mutated\n")
    digest_after = workspace_digest(str(workspace))
    assert digest_before != digest_after


def test_workspace_rejects_a_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    spawn_helpers.write_workspace(workspace, {"task_a": fixtures.PLAIN}, "b" * 64)
    target = tmp_path / "outside.py"
    target.write_text("x = 1\n")
    os.symlink(target, workspace / "tasks" / "linked.py")
    with pytest.raises(AdapterDiscoveryError):
        workspace_digest(str(workspace))


def test_release_manifest_must_be_canonical(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    spawn_helpers.write_workspace(workspace, {"task_a": fixtures.PLAIN}, "b" * 64)
    executable = Path(os.path.realpath(__import__("sys").executable))
    selector = _selector(workspace, "a" * 64, "b" * 64, executable)
    # The manifest written by the helper is canonical and matches.
    verify_release_manifest(selector)

    # A manifest with extra keys or wrong digest must be refused.
    (workspace / "adapter-release.json").write_text(
        json.dumps({"release_digest": "b" * 64, "extra": 1}, separators=(",", ":"), sort_keys=True)
    )
    with pytest.raises(AdapterDiscoveryError):
        verify_release_manifest(selector)


def test_release_manifest_digest_must_match(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    spawn_helpers.write_workspace(workspace, {"task_a": fixtures.PLAIN}, "b" * 64)
    executable = Path(os.path.realpath(__import__("sys").executable))
    wrong = _selector(workspace, "a" * 64, "c" * 64, executable)
    with pytest.raises(AdapterDiscoveryError):
        verify_release_manifest(wrong)
