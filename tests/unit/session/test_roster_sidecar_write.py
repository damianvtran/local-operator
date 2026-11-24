"""Platform gates on the roster sidecar writer (``_write_roster_sidecar``).

Why: on a windows-latest runner every roster write raised PermissionError from
``os.open(<session dir>, O_RDONLY)`` — Windows cannot open a directory as a file
descriptor — so the parent-directory fsync must be POSIX-only. POSIX keeps its
strict semantics (a refused directory sync still fails the write), and the
publish step rides out a transient Windows sharing violation on ``os.replace``
for a bounded number of attempts. These tests simulate the non-POSIX branch by
swapping the session module's ``os`` binding for a forwarding stand-in that
reports ``name == "nt"`` (see ``_pretend_non_posix``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from local_operator.session import session as session_module
from local_operator.session.session import (
    SUBAGENT_ROSTER_SIDECAR,
    _write_roster_sidecar,
)

PAYLOAD = {"version": 1, "generation": 3, "jobs": [], "accounting": [], "records": []}


def _leftover_temps(directory: Path) -> list[str]:
    return [p.name for p in directory.iterdir() if p.name.endswith(".tmp")]


@pytest.fixture
def target(tmp_path: Path) -> Path:
    return tmp_path / "sess" / SUBAGENT_ROSTER_SIDECAR


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(session_module.time, "sleep", sleeps.append)
    return sleeps


class _NonPosixOs:
    """Forward every attribute to the real ``os`` except ``name``.

    Why not ``monkeypatch.setattr(os, "name", "nt")``: that flips the global
    ``os.name`` for the whole interpreter, so anything else building a path
    meanwhile — e.g. the root conftest's real-store guard during
    ``temp.unlink()`` — makes pathlib pick ``WindowsPath`` on Linux and raise
    ``NotImplementedError`` (PR #4541 Q1). Rebinding only the session module's
    ``os`` keeps the simulation local to the code under test. Lookups are
    forwarded lazily, so a test's later ``monkeypatch.setattr(os, "replace",
    ...)`` on the real module is still seen through the stand-in.
    """

    name = "nt"

    def __getattr__(self, attribute: str):
        return getattr(os, attribute)


def _pretend_non_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    # monkeypatch restores the binding at teardown, even when the write raises.
    monkeypatch.setattr(session_module, "os", _NonPosixOs())


def _refuse_directory_open(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make ``os.open`` behave like Windows for directories."""
    real_open = os.open
    refused: list[str] = []

    def spy(path, flags, *args, **kwargs):
        if os.path.isdir(path):
            refused.append(os.fspath(path))
            raise PermissionError(13, "Permission denied", os.fspath(path))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", spy)
    return refused


def test_non_posix_skips_directory_fsync(monkeypatch: pytest.MonkeyPatch, target: Path) -> None:
    refused = _refuse_directory_open(monkeypatch)
    _pretend_non_posix(monkeypatch)
    _write_roster_sidecar(target, PAYLOAD)
    assert os.name == "posix"
    assert json.loads(target.read_text(encoding="utf-8")) == PAYLOAD
    assert refused == []
    assert _leftover_temps(target.parent) == []


def test_posix_directory_fsync_failure_still_raises(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    assert os.name == "posix"
    refused = _refuse_directory_open(monkeypatch)
    with pytest.raises(PermissionError):
        _write_roster_sidecar(target, PAYLOAD)
    assert refused == [str(target.parent)]


def test_posix_directory_fsync_oserror_propagates(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    real_open, real_fsync = os.open, os.fsync
    directory_fds: set[int] = set()

    def open_spy(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if os.path.isdir(path):
            directory_fds.add(fd)
        return fd

    def fsync(fd: int) -> None:
        if fd in directory_fds:
            raise OSError(22, "Invalid argument")
        real_fsync(fd)

    monkeypatch.setattr(os, "open", open_spy)
    monkeypatch.setattr(os, "fsync", fsync)
    with pytest.raises(OSError, match="Invalid argument"):
        _write_roster_sidecar(target, PAYLOAD)
    assert len(directory_fds) == 1
    # The rename already happened before the directory sync, as before.
    assert json.loads(target.read_text(encoding="utf-8")) == PAYLOAD


def test_non_posix_replace_retries_transient_sharing_violation(
    monkeypatch: pytest.MonkeyPatch, target: Path, _no_sleep: list[float]
) -> None:
    real_replace = os.replace
    calls: list[int] = []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) <= 2:
            raise PermissionError(13, "sharing violation", os.fspath(dst))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    _pretend_non_posix(monkeypatch)
    _write_roster_sidecar(target, PAYLOAD)
    assert os.name == "posix"
    assert len(calls) == 3
    assert len(_no_sleep) == 2
    assert json.loads(target.read_text(encoding="utf-8")) == PAYLOAD
    assert _leftover_temps(target.parent) == []


def test_non_posix_replace_gives_up_after_bounded_attempts(
    monkeypatch: pytest.MonkeyPatch, target: Path, _no_sleep: list[float]
) -> None:
    calls: list[int] = []

    def refuse(src, dst):
        calls.append(1)
        raise PermissionError(13, "sharing violation", os.fspath(dst))

    monkeypatch.setattr(os, "replace", refuse)
    _pretend_non_posix(monkeypatch)
    with pytest.raises(PermissionError):
        _write_roster_sidecar(target, PAYLOAD)
    assert os.name == "posix"
    assert len(calls) == session_module._ROSTER_REPLACE_ATTEMPTS
    assert sum(_no_sleep) < 0.5
    assert not target.exists()
    assert _leftover_temps(target.parent) == []


def test_posix_replace_permission_error_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, target: Path, _no_sleep: list[float]
) -> None:
    calls: list[int] = []

    def refuse(src, dst):
        calls.append(1)
        raise PermissionError(13, "Permission denied", os.fspath(dst))

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(PermissionError):
        _write_roster_sidecar(target, PAYLOAD)
    assert len(calls) == 1
    assert _no_sleep == []
    assert _leftover_temps(target.parent) == []
