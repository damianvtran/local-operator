"""``scripts/rebase_proof.py``: MECHANICAL only for an unchanged, non-overlapping
rebase; every other shape is a convergence review.

Each case builds a tiny scratch repository and drives the script as a
subprocess, exactly as its Run line documents. Git runs with
``GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_SYSTEM=/dev/null GIT_TERMINAL_PROMPT=0``
(AGENTS.md keychain rule: this machine's system gitconfig carries the
osxkeychain helper, and Apple Git's second system file carries it too, so
``GIT_CONFIG_SYSTEM`` on its own does not remove it) and a fixed identity, so no
credential prompt and no ambient config can reach the result.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "rebase_proof.py"


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "CMUX_"))}
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_SYSTEM="/dev/null",
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_TERMINAL_PROMPT="0",
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.invalid",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.invalid",
    )
    return env


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], env=_env(), check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(repo: Path, files: dict[str, str], message: str) -> str:
    for name, body in files.items():
        (repo / name).write_text(body)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _setup(tmp_path: Path) -> tuple[Path, str]:
    """main with two files, a PR branch editing ``pr.txt``; returns old head."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, {"pr.txt": "one\ntwo\n", "shared.txt": "a\n"}, "base")
    _git(repo, "checkout", "-q", "-b", "pr")
    old_head = _commit(repo, {"pr.txt": "one\ntwo\nthree\n"}, "feat: pr change")
    return repo, old_head


def _rebase_onto_main(repo: Path) -> str:
    _git(repo, "checkout", "-q", "pr")
    _git(repo, "rebase", "-q", "main")
    return _git(repo, "rev-parse", "HEAD")


def _run(repo: Path, old: str, new: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), old, new, "--base", "main", "--repo", str(repo)],
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def test_identical_rebase_without_overlap_is_mechanical(tmp_path: Path) -> None:
    repo, old = _setup(tmp_path)
    _git(repo, "checkout", "-q", "main")
    _commit(repo, {"unrelated.txt": "upstream\n"}, "chore: upstream")
    new = _rebase_onto_main(repo)
    assert new != old
    result = _run(repo, old, new)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"### Rebase proof — {old[:10]}..{new[:10]}" in result.stdout
    assert "all =" in result.stdout
    assert "byte-identical" in result.stdout
    assert "overlap: none" in result.stdout
    assert "MECHANICAL — no re-review needed" in result.stdout


def test_upstream_touching_a_pr_file_needs_review(tmp_path: Path) -> None:
    # Upstream edits a line far from the PR's hunk: the rebase applies cleanly
    # and range-diff still says '=', which is exactly why overlap must be
    # checked separately — it is the only signal of a possible semantic clash.
    repo, old = _setup(tmp_path)
    _git(repo, "checkout", "-q", "main")
    _commit(repo, {"pr.txt": "ONE\ntwo\n"}, "fix: upstream edits pr.txt")
    new = _rebase_onto_main(repo)
    result = _run(repo, old, new)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "CONVERGENCE REVIEW NEEDED — files: `pr.txt`" in result.stdout
    assert "MECHANICAL" not in result.stdout


def test_changed_patch_needs_review(tmp_path: Path) -> None:
    repo, old = _setup(tmp_path)
    _git(repo, "checkout", "-q", "main")
    _commit(repo, {"unrelated.txt": "upstream\n"}, "chore: upstream")
    _rebase_onto_main(repo)
    # Amend during the "rebase": same commit count, different content.
    new = _commit(repo, {"pr.txt": "one\ntwo\nthree!\n"}, "feat: pr change")
    _git(repo, "reset", "-q", "--soft", "HEAD~2")
    new = _commit(repo, {}, "feat: pr change")
    result = _run(repo, old, new)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "CHANGED" in result.stdout
    assert "range-diff not all '='" in result.stdout
    assert "patch lines differ" in result.stdout
    # R1-3: no overlap here, yet the verdict must name what to review.
    assert "CONVERGENCE REVIEW NEEDED — files: `pr.txt`" in result.stdout
    assert "none" not in result.stdout.split("Verdict:")[1]


def test_merged_pr_against_its_own_base_hints_at_base_flag(tmp_path: Path) -> None:
    # QA Q2: after the PR merged, main contains the new head; exit 2 with a
    # one-line hint, never git's usage dump.
    repo, old = _setup(tmp_path)
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--ff-only", "pr")
    result = _run(repo, old, old)
    assert result.returncode == 2
    assert "pass --base" in result.stderr
    assert "usage:" not in result.stderr


def test_unknown_revision_is_an_error_not_a_pass(tmp_path: Path) -> None:
    repo, old = _setup(tmp_path)
    result = _run(repo, old, "0" * 40)
    assert result.returncode == 2
    assert "could not be computed" in result.stderr
