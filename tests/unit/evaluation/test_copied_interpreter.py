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
