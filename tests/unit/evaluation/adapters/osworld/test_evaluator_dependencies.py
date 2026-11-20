"""Host OCR readiness for the pinned task_025 evaluator, refused before allocation.

The real release manifest and task bytes are gated inputs, so these tests pin
synthetic bytes by patching the module's two digests; the matching rule (task
id AND exact source bytes AND exact release manifest) is what is under test.
"""

from __future__ import annotations

import hashlib
import os
import stat
import sys
from pathlib import Path

import pytest
from lop_osworld_v2_adapter import evaluator_dependencies as deps
from lop_osworld_v2_adapter.provisioning import ProvisioningError

_SOURCE = b"# synthetic stand-in for the pinned task_025 bytes\n"
_RELEASE = b'{"release": "synthetic-v2.1"}'


@pytest.fixture
def pinned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "benchmark_release.json").write_bytes(_RELEASE)
    monkeypatch.setattr(deps, "_TASK_025_SOURCE_SHA256", hashlib.sha256(_SOURCE).hexdigest())
    monkeypatch.setattr(deps, "_V21_RELEASE_SHA256", hashlib.sha256(_RELEASE).hexdigest())
    monkeypatch.delenv("PATH", raising=False)
    return workspace


def _tool(directory: Path, body: str) -> Path:
    """A fake tesseract: a Python script so the test needs no host OCR install."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "tesseract"
    path.write_text(f"#!{sys.executable}\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


_HEALTHY = (
    "import sys\n"
    "assert sys.argv[1:] == ['stdin','stdout','-l','eng','--psm','6','tsv'], sys.argv\n"
    "sys.stdin.buffer.read()\n"
    "sys.stdout.write('level\\tpage_num\\tblock_num\\tpar_num\\tline_num\\tword_num\\t'"
    "'left\\ttop\\twidth\\theight\\tconf\\ttext\\n'"
    "'1\\t1\\t0\\t0\\t0\\t0\\t0\\t0\\t128\\t64\\t-1\\t\\n')"
)


def _check(
    workspace: Path,
    tools: tuple[str, ...] | None,
    *,
    task_id: str = "task_025",
    source: bytes = _SOURCE,
    forbidden: tuple[Path, ...] = (),
) -> None:
    deps.validate_evaluator_dependencies(
        task_id=task_id,
        source=source,
        workspace_root=workspace,
        tool_directories=tools,
        forbidden_roots=(workspace, *forbidden),
    )


def test_healthy_tool_on_declared_evaluator_path_passes(pinned: Path, tmp_path: Path) -> None:
    tools = tmp_path / "host-tools"
    _tool(tools, _HEALTHY)
    _check(pinned, (str(tools),))


def test_missing_tool_is_refused_with_the_remedy(pinned: Path, tmp_path: Path) -> None:
    empty = tmp_path / "empty-tools"
    empty.mkdir()
    with pytest.raises(ProvisioningError, match="needs host tesseract.*no runnable executable"):
        _check(pinned, (str(empty),))


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("import sys; sys.exit(1)", "could not run successfully"),
        ("import sys; sys.stdout.write('tesseract 5.5\\n')", "did not return TSV"),
    ],
)
def test_unusable_tool_is_refused(pinned: Path, tmp_path: Path, body: str, reason: str) -> None:
    # Discriminating negative control: same location and name as the healthy
    # case, only the behaviour differs (missing eng/tsv support, crash).
    tools = tmp_path / "host-tools"
    _tool(tools, body)
    with pytest.raises(ProvisioningError, match=reason):
        _check(pinned, (str(tools),))


def test_probe_receives_no_secrets_or_cwd(
    pinned: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OSWORLD_EVAL_MODEL_API_KEY", "must-not-leak")
    tools = tmp_path / "host-tools"
    _tool(
        tools,
        "import os, sys\n"
        "assert 'OSWORLD_EVAL_MODEL_API_KEY' not in os.environ, 'secret leaked'\n"
        "assert os.getcwd() == '/', os.getcwd()\n" + _HEALTHY,
    )
    _check(pinned, (str(tools),))


def test_tool_inside_episode_roots_is_never_run(pinned: Path, tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    artifacts = tmp_path / "artifacts"
    _tool(artifacts / "bin", f"open({str(marker)!r}, 'w').close()\n" + _HEALTHY)
    with pytest.raises(ProvisioningError, match="outside the trusted host tool locations"):
        _check(pinned, (str(artifacts / "bin"),), forbidden=(artifacts,))
    assert not marker.exists()


def test_ambient_path_entry_is_not_trusted(
    pinned: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Production workers carry no PATH; an inherited one is not an operator
    # declaration, so a tool found only there is refused, never executed.
    marker = tmp_path / "ran"
    ambient = tmp_path / "ambient"
    _tool(ambient, f"open({str(marker)!r}, 'w').close()\n" + _HEALTHY)
    monkeypatch.setenv("PATH", os.pathsep.join((str(ambient), *os.defpath.split(os.pathsep))))
    with pytest.raises(ProvisioningError, match="outside the trusted"):
        _check(pinned, None)
    assert not marker.exists()


@pytest.mark.parametrize(
    ("task_id", "source", "release"),
    [
        ("task_plain", _SOURCE, _RELEASE),  # another task
        ("task_025", b"# edited bytes\n", _RELEASE),  # unpinned task_025 source
        ("task_025", _SOURCE, b"{}"),  # another benchmark release
        ("task_025", _SOURCE, None),  # workspace without a release manifest
    ],
)
def test_unverified_or_unrelated_tasks_are_unchanged(
    pinned: Path, task_id: str, source: bytes, release: bytes | None
) -> None:
    manifest = pinned / "benchmark_release.json"
    if release is None:
        manifest.unlink()
    else:
        manifest.write_bytes(release)
    # No tool anywhere: a check that fired would refuse.
    _check(pinned, None, task_id=task_id, source=source)


@pytest.mark.asyncio
async def test_reset_start_refuses_missing_ocr_before_provider_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, episode_id: str
) -> None:
    """The real reset path: the refusal lands before any provider is built."""

    from lop_osworld_v2_adapter.adapter import OSWorldV2Adapter

    from tests.unit.evaluation.adapters.osworld import fixtures
    from tests.unit.evaluation.adapters.osworld.test_secrets_and_rescue import (
        _AWS_SECRETS,
        _prepared,
        _reset,
        _workspace,
    )

    workspace = _workspace(tmp_path, {"task_025": fixtures.PLAIN}, provider={"provider": "aws"})
    (workspace / "benchmark_release.json").write_bytes(_RELEASE)
    source = (workspace / "tasks" / "task_025.py").read_bytes()
    monkeypatch.setattr(deps, "_TASK_025_SOURCE_SHA256", hashlib.sha256(source).hexdigest())
    monkeypatch.setattr(deps, "_V21_RELEASE_SHA256", hashlib.sha256(_RELEASE).hexdigest())
    monkeypatch.delenv("PATH", raising=False)
    adapter = OSWorldV2Adapter(workspace_root=workspace)
    await _prepared(adapter, episode_id)

    def forbidden() -> None:
        pytest.fail("provider construction reached before the OCR readiness check")

    monkeypatch.setattr(adapter, "_build_provider", forbidden)
    monkeypatch.setattr(deps.shutil, "which", lambda *_args, **_kwargs: None)
    with pytest.raises(ProvisioningError, match="task_025 evaluator needs host tesseract"):
        await adapter.reset_start(_reset(episode_id, "task_025", tmp_path, _AWS_SECRETS))
