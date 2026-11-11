"""Real-git tests for scripts/gate_merge_base.py.

WHY: the merge gate's `actions/checkout` on a `labeled` event checks out a
cached `refs/pull/<n>/merge` that GitHub does not recompute when main moves
(three gates on 2026-11-10 tested mains 22-38 commits stale). These tests build
exactly that shape in throwaway repos — an origin whose main moved after the
PR's merge ref was computed — and prove the script tests the CURRENT tip, and
that a conflict or a missing base fails loudly with both SHAs named rather
than falling back to the stale checkout.

Isolation: system/global git config are nulled (the system config carries a
keychain credential helper on macOS), and every repo lives under tmp_path.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import gate_merge_base as gmb


@pytest.fixture(autouse=True)
def _isolated_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    for key in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{key}_NAME", "t")
        monkeypatch.setenv(f"GIT_{key}_EMAIL", "t@invalid")


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def commit(repo: Path, name: str, text: str) -> str:
    (repo / name).write_text(text)
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", name)
    return git(repo, "rev-parse", "HEAD")


def build(tmp_path: Path, *, conflict: bool = False, shallow: bool = False):
    """Origin with main; a PR branch; a stale merge; then main moves on.

    Returns (checkout, fresh_main, stale_main, head). `checkout` is detached
    at the stale merge, as `actions/checkout` leaves a labeled-event gate.
    """
    work = tmp_path / "work"
    work.mkdir()
    git(work, "init", "-q", "-b", "main")
    commit(work, "shared.txt", "base\n")
    for i in range(3):  # some depth, so the shallow path has to deepen
        commit(work, f"old{i}.txt", "x\n")
    stale_main = git(work, "rev-parse", "HEAD")
    git(work, "checkout", "-q", "-b", "pr")
    head = commit(work, "pr.txt", "pr\n")
    if conflict:
        head = commit(work, "shared.txt", "pr side\n")
    git(work, "checkout", "-q", "--detach", stale_main)
    git(work, "merge", "-q", "--no-ff", "-m", "stale merge", head)
    git(work, "update-ref", "refs/pull/1/merge", "HEAD")
    git(work, "checkout", "-q", "main")
    for i in range(3):
        commit(work, f"fix{i}.txt", "fixed on main\n")
    if conflict:
        commit(work, "shared.txt", "main side\n")
    fresh_main = git(work, "rev-parse", "HEAD")
    git(work, "update-ref", "refs/heads/pr", head)

    origin = tmp_path / "origin.git"
    git(tmp_path, "clone", "-q", "--bare", str(work), str(origin))
    git(work, "push", "-q", str(origin), "refs/pull/1/merge:refs/pull/1/merge")
    # GitHub serves any reachable SHA; file:// needs this to match it.
    git(origin, "config", "uploadpack.allowAnySHA1InWant", "true")

    checkout = tmp_path / "checkout"
    # Depth 2 mirrors the `changes` job's `fetch-depth: 2`; the other jobs'
    # depth-1 checkout differs only in lacking HEAD^1, which apply tolerates.
    depth = ["--depth=2"] if shallow else []
    git(tmp_path, "clone", "-q", *depth, f"file://{origin}", str(checkout))
    git(checkout, "fetch", "-q", *depth, "origin", "refs/pull/1/merge")
    git(checkout, "checkout", "-q", "--detach", "FETCH_HEAD")
    return checkout, fresh_main, stale_main, head


@pytest.mark.parametrize("shallow", [False, True])
def test_a_stale_merge_ref_is_rebuilt_on_the_current_main_tip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    shallow: bool,
) -> None:
    checkout, fresh, stale, head = build(tmp_path, shallow=shallow)
    if shallow:
        # Force the deepen loop to iterate, as a PR far behind main would.
        monkeypatch.setattr(gmb, "INITIAL_DEPTH", 1)
        monkeypatch.setattr(gmb, "DEEPEN_STEP", 1)
        assert git(checkout, "rev-parse", "--is-shallow-repository") == "true"
    assert git(checkout, "rev-parse", "HEAD^1") == stale  # the bug's precondition
    monkeypatch.chdir(checkout)
    out_file = tmp_path / "gh_output"

    assert (
        gmb.main(
            [
                "resolve",
                "--event",
                "pull_request",
                "--base-ref",
                "main",
                "--github-output",
                str(out_file),
            ]
        )
        == 0
    )
    assert out_file.read_text() == f"gate_base={fresh}\n"
    assert (
        gmb.main(
            ["apply", "--event", "pull_request", "--base", fresh, "--head", head, "--job", "test"]
        )
        == 0
    )

    assert git(checkout, "rev-parse", "HEAD^1") == fresh
    assert git(checkout, "rev-parse", "HEAD^2") == head
    assert (checkout / "fix2.txt").is_file() and (checkout / "pr.txt").is_file()
    tree = git(checkout, "rev-parse", "HEAD^{tree}")
    lines = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith(gmb.LOG_PREFIX)
    ]
    assert len(lines) == 1, lines
    assert lines[0].startswith(f"full-suite tested-base={fresh} head={head} tree={tree} merge=")
    assert lines[0].endswith(" job=test event=pull_request")


def test_two_jobs_given_the_same_base_test_the_same_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shards of one run must agree on the tree even though merge SHAs differ."""
    checkout, fresh, _stale, head = build(tmp_path)
    second = tmp_path / "second"
    git(tmp_path, "clone", "-q", str(checkout), str(second))
    git(second, "remote", "set-url", "origin", f"file://{tmp_path / 'origin.git'}")
    trees = []
    for repo in (checkout, second):
        monkeypatch.chdir(repo)
        assert gmb.main(["apply", "--event", "pull_request", "--base", fresh, "--head", head]) == 0
        trees.append(git(repo, "rev-parse", "HEAD^{tree}"))
    assert trees[0] == trees[1]


def test_a_conflict_fails_loudly_naming_both_shas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    checkout, fresh, _stale, head = build(tmp_path, conflict=True)
    monkeypatch.chdir(checkout)
    assert gmb.main(["apply", "--event", "pull_request", "--base", fresh, "--head", head]) == 1
    out = capsys.readouterr().out
    assert "::error title=Merge gate base::" in out
    assert fresh in out and head in out and "shared.txt" in out
    assert gmb.LOG_PREFIX not in out, "a failed rebuild must not print a tested-base line"
    # Left on a clean base, not mid-merge, so nothing downstream runs on debris.
    assert git(checkout, "status", "--porcelain") == ""


def test_a_pr_event_without_a_resolved_base_refuses_the_stale_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    checkout, _fresh, _stale, head = build(tmp_path)
    before = git(checkout, "rev-parse", "HEAD")
    monkeypatch.chdir(checkout)
    assert gmb.main(["apply", "--event", "pull_request", "--base", "", "--head", head]) == 1
    assert "refusing to test the stale merge ref" in capsys.readouterr().out
    assert git(checkout, "rev-parse", "HEAD") == before


@pytest.mark.parametrize("event", ["push", "schedule", "workflow_dispatch"])
def test_non_pr_events_change_nothing_and_still_log_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], event: str
) -> None:
    checkout, fresh, _stale, head = build(tmp_path)
    before = git(checkout, "rev-parse", "HEAD")
    monkeypatch.chdir(checkout)
    out_file = tmp_path / "gh_output"
    assert (
        gmb.main(["resolve", "--event", event, "--base-ref", "", "--github-output", str(out_file)])
        == 0
    )
    assert not out_file.exists()
    assert (
        gmb.main(["apply", "--event", event, "--base", fresh, "--head", head, "--job", "test"]) == 0
    )
    assert git(checkout, "rev-parse", "HEAD") == before
    lines = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith(gmb.LOG_PREFIX)
    ]
    assert lines == [
        f"full-suite tested-base=none head={before} "
        f"tree={git(checkout, 'rev-parse', 'HEAD^{tree}')} merge={before} job=test event={event}"
    ]
