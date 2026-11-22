"""Holder liveness contracts independent of the host process table."""

import errno
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from local_operator.guardrails import provenance as prov

_PID = 12345


@pytest.fixture
def proc_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Mock]:
    root = tmp_path / "proc"
    (root / "self").mkdir(parents=True)
    lookup = Mock(return_value=None)
    # Patch the module's OS boundary, not the shared os module: synthetic PIDs
    # must never reach the host's process lookup.
    monkeypatch.setattr(prov, "_PROC_ROOT", str(root))
    monkeypatch.setattr(prov, "os", SimpleNamespace(path=os.path, kill=lookup))
    return root, lookup


def test_visible_pid_needs_no_signal_lookup(proc_root: tuple[Path, Mock]) -> None:
    root, lookup = proc_root
    (root / str(_PID)).mkdir()
    assert prov._still_exists(_PID) is True
    lookup.assert_not_called()


@pytest.mark.parametrize(
    ("lookup_error", "exists"),
    [
        pytest.param(None, True, id="hidden-live"),
        pytest.param(ProcessLookupError(errno.ESRCH, "exited"), False, id="exited"),
        pytest.param(PermissionError(errno.EPERM, "denied"), True, id="permission"),
        pytest.param(OSError(errno.EIO, "unknown"), True, id="ambiguous"),
    ],
)
def test_missing_proc_entry_is_confirmed_by_signal_lookup(
    proc_root: tuple[Path, Mock], lookup_error: OSError | None, exists: bool
) -> None:
    _, lookup = proc_root
    lookup.side_effect = lookup_error
    assert prov._still_exists(_PID) is exists
    lookup.assert_called_once_with(_PID, 0)
