"""Host OCR readiness for the pinned task_025 evaluator, refused before allocation.

The real release manifest and task bytes are gated inputs, so these tests pin
synthetic bytes by patching the module's two digests; the matching rule (task
id AND exact source bytes AND exact release manifest) is what is under test.
"""

from __future__ import annotations

import hashlib
import os
import select
import signal
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

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
    # PATH absent still means os.defpath. Keep the host's installed tools
    # out of every synthetic test, and exercise default precedence explicitly.
    default_tools = tmp_path / "default-tools"
    default_tools.mkdir()
    monkeypatch.setattr(os, "defpath", str(default_tools))
    monkeypatch.delenv("PATH", raising=False)
    return workspace


def _tool(directory: Path, body: str, name: str = "tesseract") -> Path:
    """A fake host tool: a Python script so the test needs no OCR install."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(f"#!{sys.executable}\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


_HEADER = (
    b"level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
    b"left\ttop\twidth\theight\tconf\ttext\n"
)
_PAGE = b"1\t1\t0\t0\t0\t0\t0\t0\t128\t64\t-1\t\n"
_HEALTHY = (
    "import struct, sys\n"
    "assert sys.argv[1:] == ['stdin','stdout','-l','eng','--psm','6','tsv'], sys.argv\n"
    "image = sys.stdin.buffer.read()\n"
    "assert image[:16] == b'\\x89PNG\\r\\n\\x1a\\n\\x00\\x00\\x00\\x0dIHDR'\n"
    "assert struct.unpack('>IIBBBBB', image[16:29]) == (128, 64, 8, 2, 0, 0, 0)\n"
    f"sys.stdout.buffer.write({_HEADER + _PAGE!r})"
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
    assert len(deps._OCR_SAMPLE) <= select.PIPE_BUF  # Stdin cannot block on a non-reader.
    _check(pinned, (str(tools),))


def test_system_tool_precedes_declared_path(pinned: Path, tmp_path: Path) -> None:
    _tool(Path(os.defpath), "import os\nassert 'PATH' in os.environ\n" + _HEALTHY)
    declared = tmp_path / "declared-tools"
    _tool(declared, "raise SystemExit(99)")
    _check(pinned, (str(declared),))


def test_default_lookup_leaves_path_unset(pinned: Path) -> None:
    _tool(Path(os.defpath), "import os\nassert 'PATH' not in os.environ\n" + _HEALTHY)
    _check(pinned, None)


def test_missing_tool_is_refused_with_the_remedy(pinned: Path, tmp_path: Path) -> None:
    empty = tmp_path / "empty-tools"
    empty.mkdir()
    with pytest.raises(
        ProvisioningError,
        match="needs host tesseract.*no runnable executable.*--infra EVALUATOR_TOOL_PATH=<dir>",
    ):
        _check(pinned, (str(empty),))


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("raise SystemExit(1)", r"status 1\); check that tesseract --list-langs includes eng"),
        ("import sys; sys.stdout.write('tesseract 5.5\\n')", "did not return TSV"),
        (f"import sys; sys.stdout.buffer.write({_HEADER!r})", "did not return TSV"),
        (
            f"import sys; sys.stdout.buffer.write({_HEADER + b'malformed row'!r})",
            "did not return TSV",
        ),
        (
            f"import sys; sys.stdout.buffer.write({_HEADER + _PAGE[:-1] + bytes([255])!r})",
            "did not return UTF-8 TSV",
        ),
        # A tool that came back ready with the old PGM sample must not pass.
        (
            "import sys\nif not sys.stdin.buffer.read().startswith(b'P5\\n'): sys.exit(17)\n"
            f"sys.stdout.buffer.write({_HEADER + _PAGE!r})",
            "status 17",
        ),
    ],
)
def test_unusable_tool_is_refused(pinned: Path, tmp_path: Path, body: str, reason: str) -> None:
    # Discriminating negative control: same location and name as the healthy
    # case, only the behaviour differs (missing eng/tsv support, crash).
    tools = tmp_path / "host-tools"
    _tool(tools, body)
    with pytest.raises(ProvisioningError, match=reason):
        _check(pinned, (str(tools),))


def test_declared_wrapper_gets_only_validated_path(
    pinned: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OSWORLD_EVAL_MODEL_API_KEY", "must-not-leak")
    monkeypatch.setenv("TESSDATA_PREFIX", "ambient-not-approved")
    monkeypatch.setenv("PATH", str(tmp_path / "ambient"))
    tools = tmp_path / "host-tools"
    _tool(tools, "print('helper ready')", name="ocr-probe-helper")
    expected_path = os.pathsep.join((os.defpath, str(tools)))
    _tool(
        tools,
        "import os, subprocess\n"
        "assert 'OSWORLD_EVAL_MODEL_API_KEY' not in os.environ, 'secret leaked'\n"
        "assert 'TESSDATA_PREFIX' not in os.environ, 'ambient setting leaked'\n"
        f"assert os.environ['PATH'] == {expected_path!r}\n"
        "assert os.getcwd() == '/', os.getcwd()\n"
        "assert subprocess.check_output(['ocr-probe-helper']) == b'helper ready\\n'\n" + _HEALTHY,
    )
    _check(pinned, (str(tools),))


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("import os\nwhile True: os.write(1, b'x' * 4096)", "exceeded.*TSV response size"),
        ("import time; time.sleep(60)", "timed out"),
        ("import os, time; os.close(1); time.sleep(60)", "timed out"),
    ],
    ids=["flood", "silent", "closed-stdout"],
)
def test_probe_reaps_owned_process_on_limit(
    pinned: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str, reason: str
) -> None:
    tools = tmp_path / "host-tools"
    _tool(tools, body)
    if reason == "timed out":
        monkeypatch.setattr(deps, "_OCR_TIMEOUT_SECONDS", 1)
    calls: list[subprocess.Popen[bytes]] = []
    popen = subprocess.Popen

    def capture(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        assert kwargs.get("start_new_session") is True
        child = popen(*args, **kwargs)
        calls.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", capture)
    with pytest.raises(ProvisioningError, match=reason):
        _check(pinned, (str(tools),))
    (child,) = calls
    assert child.returncode == -signal.SIGKILL
    with pytest.raises(ChildProcessError):
        os.waitpid(child.pid, os.WNOHANG)


def test_child_diagnostics_are_never_reported(pinned: Path, tmp_path: Path) -> None:
    tools = tmp_path / "host-tools"
    _tool(
        tools,
        "import sys; print('private-stdout'); sys.stderr.write('private-stderr'); sys.exit(4)",
    )
    with pytest.raises(ProvisioningError, match="status 4") as refusal:
        _check(pinned, (str(tools),))
    assert "private-" not in str(refusal.value)


def test_tool_inside_episode_roots_is_never_run(pinned: Path, tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    artifacts = tmp_path / "artifacts"
    _tool(artifacts / "bin", f"open({str(marker)!r}, 'w').close()\n" + _HEALTHY)
    with pytest.raises(ProvisioningError, match="outside the trusted host tool locations"):
        _check(pinned, (str(artifacts / "bin"),), forbidden=(artifacts,))
    assert not marker.exists()


def test_symlinked_tool_inside_episode_roots_is_never_run(pinned: Path, tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    inputs = tmp_path / "inputs"
    binary = _tool(inputs / "bin", f"open({str(marker)!r}, 'w').close()\n" + _HEALTHY)
    host_tools = tmp_path / "host-tools"
    host_tools.mkdir()
    (host_tools / "tesseract").symlink_to(binary)
    with pytest.raises(
        ProvisioningError, match="first matching executable is outside the trusted"
    ) as refusal:
        _check(pinned, (str(host_tools),), forbidden=(inputs,))
    assert str(binary) not in str(refusal.value)
    assert not marker.exists()


def test_ambient_path_entry_is_not_trusted(
    pinned: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Even when a system executable wins, its helpers must not resolve from
    # an undeclared ambient directory appended to PATH.
    marker = tmp_path / "ran"
    _tool(Path(os.defpath), f"open({str(marker)!r}, 'w').close()\n" + _HEALTHY)
    ambient = tmp_path / "ambient"
    ambient.mkdir()
    monkeypatch.setenv("PATH", os.pathsep.join((os.defpath, str(ambient))))
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
