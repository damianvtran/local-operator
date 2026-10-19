"""The suite's guard against a test reaching the operator's REAL runtime root.

``tests/conftest.py::isolate_environment`` points ``LOCAL_OPERATOR_RUNTIME_ROOT``
at a per-test scratch directory and then asks ``procname.runtime_root()``
whether it honoured that. These cells call the guard directly, so they hold on
every platform and do not depend on any plant succeeding: the guard must raise
when the variable is missing and when ``runtime_root`` answers the passwd-home
root, which is the reviewer mutation that let ``test_procname.py`` fixtures
delete and republish the real ``~/.local/share/lop/runtime/<key>``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from local_operator import procname
from tests.conftest import (
    RuntimeRootEscape,
    _passwd_runtime_root,
    assert_runtime_root_isolated,
)


def test_this_test_already_runs_under_an_isolated_root(tmp_path_factory) -> None:
    resolved = procname.runtime_root()
    assert resolved is not None
    real = _passwd_runtime_root()
    assert real is None or Path(os.path.realpath(resolved)) != Path(os.path.realpath(real))
    assert (
        Path(os.path.realpath(tmp_path_factory.getbasetemp()))
        in Path(os.path.realpath(resolved)).parents
    )


@pytest.mark.parametrize("value", [None, ""])
def test_a_missing_or_empty_override_fails(tmp_path, monkeypatch, value) -> None:
    if value is None:
        monkeypatch.delenv("LOCAL_OPERATOR_RUNTIME_ROOT", raising=False)
    else:
        monkeypatch.setenv("LOCAL_OPERATOR_RUNTIME_ROOT", value)

    with pytest.raises(RuntimeRootEscape, match="unset or empty"):
        assert_runtime_root_isolated(tmp_path)


def test_code_that_ignores_the_override_fails(tmp_path, monkeypatch) -> None:
    """The variable is set, and ``runtime_root`` answers the real root anyway."""
    monkeypatch.setenv("LOCAL_OPERATOR_RUNTIME_ROOT", str(tmp_path / "runtime"))
    real = _passwd_runtime_root()
    assert real is not None, "precondition: a passwd home to compare against"
    monkeypatch.setattr(procname, "runtime_root", lambda: real)

    with pytest.raises(RuntimeRootEscape, match="REAL"):
        assert_runtime_root_isolated(tmp_path / "runtime")


def test_a_root_outside_the_tests_scratch_tree_fails(tmp_path, monkeypatch) -> None:
    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setenv("LOCAL_OPERATOR_RUNTIME_ROOT", str(elsewhere))

    with pytest.raises(RuntimeRootEscape, match="outside"):
        assert_runtime_root_isolated(tmp_path / "runtime")


def test_an_honoured_override_passes(tmp_path, monkeypatch) -> None:
    root = tmp_path / "runtime"
    monkeypatch.setenv("LOCAL_OPERATOR_RUNTIME_ROOT", str(root))

    assert assert_runtime_root_isolated(root) == root


def test_the_suites_own_venv_is_never_a_plant_target() -> None:
    """The venv half: ``<sys.prefix>/bin/Local Operator`` of the suite itself.

    The runtime root above keeps the IMAGE out of the operator's tree, but a
    plant still re-pointed the worktree's real link at that per-test root, which
    pytest then deleted. ``refuse_real_venv_plant`` answers ``None`` for the
    suite's own prefix, so nothing reaching a spawn path can plant there.
    """
    assert procname.branded_link_path() is None
    assert procname.ensure_branded_interpreter() is None
