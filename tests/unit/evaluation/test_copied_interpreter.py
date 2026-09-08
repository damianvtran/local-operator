"""The shared copied-interpreter helper fails loudly, never silently.

The defect this guards: three copies of the helper used to fall through
from a copy of the running interpreter that could not start (uv builds link
a relative ``libpython`` that ``venv --copies`` does not carry) to whatever
``python3`` was on PATH -- a different major version whose worker then died
importing the running interpreter's ``pydantic_core``. The failure surfaced
as a bare ``EOFError`` from the RPC reader, sixteen tests deep, naming none
of the cause. See ``copied_interpreter.py`` for the full account.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.evaluation import copied_interpreter as helper

pytestmark = pytest.mark.slow


def test_copied_interpreter_is_the_running_interpreter_and_imports_pydantic_core(
    tmp_path: Path,
) -> None:
    executable = helper.copied_interpreter(tmp_path / "venv")
    assert executable.is_file() and not executable.is_symlink()
    site = helper.site_packages_of(executable)
    assert (site / helper.REPO_PTH_NAME).is_file()
    # The copy is a copy of THIS interpreter, not of whatever is on PATH: the
    # version it reports through the worker's own flags is ours.
    probe = helper._probe(executable, "import sys, pydantic_core; print(*sys.version_info[:2])")
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.split() == [str(part) for part in sys.version_info[:2]]


def test_unrepairable_start_failure_names_the_base_and_the_dyld_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the copy cannot start and no libpython can be carried, say so."""

    started: list[bool] = []
    real_probe = helper._probe

    def failing_probe(executable: Path, code: str):  # type: ignore[no-untyped-def]
        if code == "print('ok')" and not started:
            started.append(True)
            result = real_probe(executable, "import sys; sys.exit(134)")
            result.stderr = "dyld: Library not loaded: @rpath/libpython"
            return result
        return real_probe(executable, code)

    monkeypatch.setattr(helper, "_probe", failing_probe)
    monkeypatch.setattr(helper, "_shared_libpython", lambda base: None)
    with pytest.raises(AssertionError) as raised:
        helper.copied_interpreter(tmp_path / "venv")
    message = str(raised.value)
    assert "did not start" in message
    assert "no shared libpython" in message
    assert str(Path(sys.executable).resolve().name) in message or "python" in message
    assert "@rpath/libpython" in message


def test_import_mismatch_names_the_pth_rather_than_falling_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copy that cannot import pydantic_core through its .pth is refused."""

    monkeypatch.setattr(helper, "dependency_roots", lambda: [str(tmp_path / "nowhere")])
    with pytest.raises(AssertionError) as raised:
        helper.copied_interpreter(tmp_path / "venv")
    message = str(raised.value)
    assert "cannot import pydantic_core" in message
    assert helper.REPO_PTH_NAME in message
    assert "ModuleNotFoundError" in message


def test_failed_base_query_reports_its_exit_and_bounded_output(tmp_path, monkeypatch) -> None:
    answer = subprocess.CompletedProcess(
        [],
        73,
        stdout="x" * 2000 + "STDOUT_TAIL",
        stderr="y" * 2000 + "SYNTHETIC_BASE_QUERY_FAILURE",
    )
    monkeypatch.setattr(helper, "_probe", lambda *_: answer)
    with pytest.raises(AssertionError, match="metadata query failed") as raised:
        helper._shared_libpython(tmp_path / "python")
    message = str(raised.value)
    assert "rc=73" in message
    assert "STDOUT_TAIL" in message and "SYNTHETIC_BASE_QUERY_FAILURE" in message
    assert len(message) < 2500
    assert "no shared libpython" not in message


@pytest.mark.parametrize(
    "stdout",
    [
        "truncated\n",
        "1\nmissing-line\n",
        "invalid\n/lib\nlibpython.so\n",
        "1\n\nlibpython.so\n",
        "1\n/lib\n\n",
    ],
)
def test_malformed_shared_metadata_is_not_reported_as_no_library(
    tmp_path, monkeypatch, stdout
) -> None:
    monkeypatch.setattr(helper, "_probe", lambda *_: subprocess.CompletedProcess([], 0, stdout, ""))
    with pytest.raises(AssertionError, match="metadata") as raised:
        helper._shared_libpython(tmp_path / "python")
    assert "stdout=" in str(raised.value) and "rc=0" in str(raised.value)


@pytest.mark.parametrize("stdout", ["0\n\n\n", "1\n/lib\nPython.framework/Versions/3/Python\n"])
def test_only_positive_nonshared_or_unsupported_metadata_returns_none(
    tmp_path, monkeypatch, stdout
) -> None:
    monkeypatch.setattr(helper, "_probe", lambda *_: subprocess.CompletedProcess([], 0, stdout, ""))
    assert helper._shared_libpython(tmp_path / "python") is None


def test_advertised_missing_library_names_the_candidate(tmp_path, monkeypatch) -> None:
    candidate = tmp_path / "libpython-test.dylib"
    stdout = f"1\n{tmp_path}\n{candidate.name}\n"
    monkeypatch.setattr(
        helper, "_probe", lambda *_: subprocess.CompletedProcess([], 0, stdout, "query-note")
    )
    with pytest.raises(AssertionError, match="is missing") as raised:
        helper._shared_libpython(tmp_path / "python")
    message = str(raised.value)
    assert str(candidate) in message and "query-note" in message and "rc=0" in message


def test_unreadable_advertised_library_names_the_candidate_and_keeps_the_cause(
    tmp_path, monkeypatch
) -> None:
    """An unreadable path is not evidence that no library exists.

    The stat itself can fail (a permission-denied mount, a stale automount),
    and reporting that as "no shared libpython" is the exact
    misclassification this module exists to prevent.
    """
    stdout = f"1\n{tmp_path}\nlibpython-unreadable.dylib\n"
    monkeypatch.setattr(
        helper, "_probe", lambda *_: subprocess.CompletedProcess([], 0, stdout, "query-note")
    )
    failure = PermissionError(13, "Permission denied")

    def refuse(self):  # noqa: ANN001, ANN202
        raise failure

    monkeypatch.setattr(Path, "is_file", refuse)
    with pytest.raises(AssertionError, match="cannot inspect shared libpython") as raised:
        helper._shared_libpython(tmp_path / "python")
    message = str(raised.value)
    assert "libpython-unreadable.dylib" in message
    assert "rc=0" in message and "query-note" in message
    assert "no shared libpython" not in message
    assert raised.value.__cause__ is failure


def test_query_failure_preserves_the_copied_interpreters_start_context(
    tmp_path, monkeypatch
) -> None:
    real_probe = helper._probe
    queries = []

    def injected(executable, code):
        if code == "print('ok')":
            return subprocess.CompletedProcess([], 134, "FIRST_START_STDOUT", "ORIGINAL_DYLD_ERROR")
        if "Py_ENABLE_SHARED" in code:
            queries.append(executable)
            return subprocess.CompletedProcess(
                [], 73, "BASE_QUERY_STDOUT", "SYNTHETIC_BASE_QUERY_FAILURE"
            )
        return real_probe(executable, code)

    monkeypatch.setattr(helper, "_probe", injected)
    with pytest.raises(AssertionError, match="metadata query failed") as raised:
        helper.copied_interpreter(tmp_path / "venv")
    message = str(raised.value)
    assert "rc=73" in message and "SYNTHETIC_BASE_QUERY_FAILURE" in message
    assert "rc=134" in message and "ORIGINAL_DYLD_ERROR" in message
    assert "FIRST_START_STDOUT" in message and "BASE_QUERY_STDOUT" in message
    assert isinstance(raised.value.__cause__, AssertionError)
    assert len(queries) == 1  # Never retry or cache the failed probe.
