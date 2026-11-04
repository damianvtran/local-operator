"""Source snapshots must not inherit local pytest sandboxes.

Inspect git's index rather than walking the checkout: local scratch belongs to
its owner and symlinks there may point outside the repository. Intentional
fixtures elsewhere remain source and must still obey the extractor's guard.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from local_operator.update import UpdateError, resolve_snapshot

ROOT = Path(__file__).resolve().parents[2]


def test_generated_pytest_sandbox_is_not_tracked() -> None:
    tracked = subprocess.check_output(["git", "ls-files", "-z", "--", ".tmp"], cwd=ROOT)
    assert not tracked, f"generated pytest sandbox is tracked: {tracked!r}"


def test_generated_pytest_sandbox_is_ignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", ".tmp/pytest-of-user/pytest-0/homecurrent"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()


def test_ignore_does_not_hide_nested_fixture_sandboxes() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "tests/fixtures/.tmp/current"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1, result.stderr.decode()


def test_committed_source_snapshot_is_extractable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(ROOT)
    snapshot = resolve_snapshot("HEAD")
    try:
        assert snapshot.commit == _git(ROOT, "rev-parse", "HEAD").strip()
        assert not (snapshot.path / ".tmp").exists()
        assert (snapshot.path / "local_operator" / "update.py").is_file()
    finally:
        shutil.rmtree(snapshot.path)


@pytest.mark.parametrize("target", ["/outside/home", "../../outside/home"])
def test_snapshot_still_rejects_unsafe_fixture_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    _snapshot_repo(tmp_path, target)
    monkeypatch.chdir(tmp_path)
    # Refuse the archive, not just the offending member: silently filtering it
    # would turn incomplete source into an apparently successful installation.
    with pytest.raises(UpdateError, match="could not unpack"):
        resolve_snapshot("HEAD")


def test_snapshot_keeps_intentional_relative_fixture_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _snapshot_repo(tmp_path, "payload.txt")
    monkeypatch.chdir(tmp_path)
    snapshot = resolve_snapshot("HEAD")
    try:
        fixture = snapshot.path / "fixtures" / "current"
        assert fixture.is_symlink()
        assert fixture.read_text() == "intentional fixture\n"
        assert snapshot.commit == _git(tmp_path, "rev-parse", "HEAD").strip()
        assert snapshot.version == "0.0.1"
    finally:
        shutil.rmtree(snapshot.path)


def _git(repo: Path, *args: str) -> str:
    # A synthetic local repository needs neither credentials nor the user's
    # signing/hooks configuration; command-local settings keep it hermetic.
    return subprocess.check_output(
        ["git", "-c", "core.hooksPath=/dev/null", *args], cwd=repo, text=True
    )


def _snapshot_repo(repo: Path, target: str) -> None:
    _git(repo, "init", "-q")
    (repo / "pyproject.toml").write_text('[project]\nname = "fixture"\nversion = "0.0.1"\n')
    fixtures = repo / "fixtures"
    fixtures.mkdir()
    (fixtures / "payload.txt").write_text("intentional fixture\n")
    (fixtures / "current").symlink_to(target)
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Archive test",
        "-c",
        "user.email=archive@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "fixture",
    )
