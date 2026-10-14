"""Preflight must detect the child's tree, not pytest's cwd-assisted import."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests import tree_identity

ROOT = Path(__file__).resolve().parents[2]


def test_this_interpreter_imports_the_tested_tree() -> None:
    assert tree_identity.mismatch(ROOT) is None


def test_a_foreign_child_package_is_refused_with_both_paths(tmp_path: Path) -> None:
    # A decoy PYTHONPATH exercises the real import lookup without installing
    # another dependency tree merely to represent a foreign editable install.
    foreign = tmp_path / "foreign"
    package = foreign / "local_operator"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    refusal = tree_identity.mismatch(ROOT, env={"PYTHONPATH": str(foreign)})
    assert refusal is not None
    assert str(ROOT / "local_operator") in refusal
    assert str(package) in refusal
    assert "uv pip install -e" in refusal


@pytest.mark.parametrize("failure", ["missing", "timeout", "error"])
def test_an_unanswered_probe_is_not_reported_as_compatible(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        if failure == "timeout":
            raise subprocess.TimeoutExpired(sys.executable, 30)
        if failure == "error":
            raise OSError("unavailable interpreter")
        return subprocess.CompletedProcess([sys.executable], 0, "\n", "")

    monkeypatch.setattr(tree_identity.subprocess, "run", run)
    refusal = tree_identity.mismatch(ROOT)
    assert refusal is not None
    assert "probe could not resolve it" in refusal


def test_a_synthetic_pytest_root_has_no_product_to_compare(tmp_path: Path) -> None:
    assert tree_identity.mismatch(tmp_path) is None
