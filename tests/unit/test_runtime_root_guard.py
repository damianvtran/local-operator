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
import sys
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


# --- The process-wide half: ``refuse_real_runtime_root`` -------------------------
#
# These cells never aim at the operator's real root. They point the guard's notion
# of "real" (``_REAL_RUNTIME_ROOT``) at a scratch directory and drive production
# code at it, so a guard that failed open would write into ``tmp_path``, not into
# ``~/.local/share/lop/runtime``.


def _aim_guard_at(monkeypatch: pytest.MonkeyPatch, fake_real: Path) -> None:
    from tests import conftest as suite

    monkeypatch.setattr(suite, "_REAL_RUNTIME_ROOT", fake_real)


@pytest.fixture(scope="module")
def root_outside_a_test_body() -> str | None:
    """``LOCAL_OPERATOR_RUNTIME_ROOT`` as a MODULE-scoped fixture sees it.

    Set up before this module's first test applies its function-scoped
    override, so it reads what every collection-time import, hook and
    wider-scoped teardown reads: the import-time default, or nothing.
    """
    return os.environ.get("LOCAL_OPERATOR_RUNTIME_ROOT")


def test_the_suite_has_a_runtime_root_outside_any_test_body(root_outside_a_test_body) -> None:
    """The import-time default: what code sees when no test's override is active."""
    from tests import conftest as suite

    assert root_outside_a_test_body == suite._SUITE_RUNTIME_ROOT, (
        "outside a test body the runtime root must be the suite's scratch root, "
        f"not {root_outside_a_test_body!r} (unset falls back to the real root)"
    )
    assert os.path.isdir(suite._SUITE_RUNTIME_ROOT)
    real = _passwd_runtime_root()
    assert real is None or not suite._is_within(Path(suite._SUITE_RUNTIME_ROOT), real)


def test_runtime_root_is_wrapped_process_wide() -> None:
    from tests import conftest as suite

    assert procname.runtime_root is suite._GUARDED_RUNTIME_ROOT


def test_a_real_root_answer_raises_and_is_recorded(tmp_path, monkeypatch) -> None:
    """The unswallowed path: a direct caller sees the refusal at once."""
    from tests import conftest as suite

    fake_real = tmp_path / "operator" / "runtime"
    _aim_guard_at(monkeypatch, fake_real)
    monkeypatch.setenv("LOCAL_OPERATOR_RUNTIME_ROOT", str(fake_real))

    with pytest.raises(RuntimeRootEscape, match="REAL"):
        procname.runtime_root()
    assert len(suite.drain_runtime_root_escapes()) == 1


def test_a_swallowed_real_root_plant_writes_nothing_and_is_still_recorded(
    tmp_path, monkeypatch
) -> None:
    """The silent path the guard exists for: production swallows the raise.

    ``ensure_branded_interpreter`` catches every exception (a brand is
    decoration), so a test that reached the real root used to pass with the
    plant done. Under the guard nothing is written there, and the escape is on
    record — which is what fails the test in ``isolate_environment``.
    """
    from tests import conftest as suite

    fake_real = tmp_path / "operator" / "runtime"
    _aim_guard_at(monkeypatch, fake_real)
    monkeypatch.setenv("LOCAL_OPERATOR_RUNTIME_ROOT", str(fake_real))
    prefix = tmp_path / "venv"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "pyvenv.cfg").write_text(f"home = {Path(sys.executable).parent}\n")
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))

    assert procname.ensure_branded_interpreter() is None
    assert not fake_real.exists(), "nothing may be planted in the guarded root"
    escapes = suite.drain_runtime_root_escapes()
    assert escapes and all("REAL" in line for line in escapes)


def test_an_escape_fails_the_test_that_caused_it(tmp_path) -> None:
    """End to end: a swallowed escape turns an otherwise passing test red.

    A nested run that uses the suite's REAL ``isolate_environment`` (imported from
    ``tests.conftest``, not a copy of its logic) and one test whose only fault is
    that a swallowed call reached a guarded root. The outer assertion is on the
    nested run's outcome, so a guard that records but never fails shows up here as
    a pass.
    """
    import subprocess
    import textwrap

    repo = Path(__file__).resolve().parents[2]
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "conftest.py").write_text(
        "from tests.conftest import isolate_environment  # noqa: F401\n"
    )
    (nested / "test_escape.py").write_text(textwrap.dedent("""
            from local_operator import procname
            from tests import conftest as suite

            def test_swallows_the_refusal(tmp_path, monkeypatch):
                fake_real = tmp_path / "operator" / "runtime"
                monkeypatch.setattr(suite, "_REAL_RUNTIME_ROOT", fake_real)
                monkeypatch.setenv("LOCAL_OPERATOR_RUNTIME_ROOT", str(fake_real))
                try:
                    procname.runtime_root()
                except Exception:
                    pass  # the shape of every production caller
            """))
    env = dict(os.environ, PYTHONPATH=str(repo))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:xdist",
            "--rootdir",
            str(nested),
            "-c",
            os.devnull,
            str(nested),
        ],
        cwd=nested,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "1 passed, 1 error" in output, output
    assert "reached the operator's real runtime root" in output, output
