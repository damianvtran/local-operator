"""Ignored-symlink and ``next-env.d.ts`` witnesses for plain worktree cleanup.

Field case (guard 0.113.0, session 045831e8588c, 2026-11-24): the session's
own finished Next.js QA worktree had no creation-ledger row and was held for
"2 ignored entries the guard cannot prove regenerable": an ignored
``.env.local`` SYMLINK into the primary checkout and ``next-env.d.ts``.
Unlinking a symlink deletes no bytes, and Next.js rewrites ``next-env.d.ts`` on
every build, so both are now witnesses for any tree. Everything else holds.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_operator.guardrails import provenance as prov
from local_operator.guardrails.check import CallContext, GuardSettings, check_tool_call
from local_operator.guardrails.classify import Risk, classify_bash
from local_operator.guardrails.preview import BUILTIN_PASS, preview_tool_call

ME = "aaaaaaaaaaa1"


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@e", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    state = home / ".local-operator"
    (state / "sessions" / ME / "scratchpad").mkdir(parents=True)
    (state / "config.yml").write_text("version: 1\n")
    monkeypatch.setattr(prov, "_live_cwd_inside", lambda _, **_kw: False)
    monkeypatch.setattr(prov, "cwd_holders", lambda real, sid, **_kw: ())
    monkeypatch.setattr(prov, "_github_repo", lambda _: "example/repo")
    monkeypatch.setattr(prov, "_gh_prs", lambda repo, branch: None)
    monkeypatch.setattr(prov, "_gh_open_prs", lambda repo, branch: None)
    monkeypatch.setattr(prov, "_running_owners", lambda real, sid, strict=False, **_: ())
    monkeypatch.setattr(prov, "_running_session_ids", lambda: set(), raising=False)
    monkeypatch.setattr(prov, "_cwd_inventory_cache", None)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))
    for key in [k for k in os.environ if k.startswith(("CMUX_", "GIT_", "LOP_"))]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    return home


def _env(home: Path) -> dict[str, str]:
    return {"HOME": str(home), "LOCAL_OPERATOR_CONFIG_DIR": str(home / ".local-operator")}


@pytest.fixture
def repo(home: Path) -> Path:
    """A primary checkout with an origin and the field ignore rules."""
    origin = home / "origin.git"
    _git("init", "-q", "--bare", "-b", "main", str(origin), cwd=home)
    repo = home / "repo"
    _git("init", "-q", "-b", "main", str(repo), cwd=home)
    (repo / ".gitignore").write_text(".worktrees/\n.env*\nnext-env.d.ts\nfifo-*\n")
    (repo / "a.txt").write_text("a\n")
    _git("add", ".", cwd=repo)
    _git("commit", "-qm", "init", cwd=repo)
    _git("remote", "add", "origin", str(origin), cwd=repo)
    _git("push", "-q", "origin", "main", cwd=repo)
    _git("fetch", "-q", "origin", cwd=repo)
    # The real local secret lives in the primary checkout, outside the tree.
    (repo / ".env.local").write_text("SYNTHETIC=1\n")
    return repo


@pytest.fixture
def wt(home: Path, repo: Path) -> Path:
    """The field shape: a detached QA tree at origin/main, NO ledger row."""
    wt = repo / ".worktrees" / "qa-n1"
    _git("worktree", "add", "-q", "--detach", str(wt), "origin/main", cwd=repo)
    assert prov.recorded_creators(str(wt)) == []
    return wt


def _rank(home: Path, command: str) -> Any:
    return classify_bash(command, cwd=str(home), env=_env(home), scratchpad=None, session_id=ME)


class _Judge:
    def __init__(self, answer: str = "BLOCK: not verified") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    async def __call__(self, system: str, prompt: str) -> str:
        self.prompts.append(system + "\n" + prompt)
        return self.answer


async def _check(home: Path, command: str, judge: _Judge) -> Any:
    ctx = CallContext(cwd=str(home), env=_env(home), scratchpad=None, session_id=ME)
    return await check_tool_call(
        "bash", {"command": command}, ctx, judge=judge, settings=GuardSettings(), forked_judge=judge
    )


def _commands(repo: Path, wt: Path) -> list[str]:
    return [
        f"git -C {repo} worktree remove {wt}",
        f"git -C {repo} worktree remove {wt.relative_to(repo)}",
    ]


def _assert_low(home: Path, repo: Path, wt: Path) -> None:
    for command in _commands(repo, wt):
        ranked = _rank(home, command)
        assert ranked.risk is Risk.LOW, (command, ranked.reasons)


def _assert_held(home: Path, repo: Path, wt: Path) -> None:
    assert prov.plain_cleanup_ignored_inventory(str(wt)).unverified_ignored >= 1
    for command in _commands(repo, wt):
        ranked = _rank(home, command)
        assert ranked.risk is not Risk.LOW, (command, ranked.reasons)


@pytest.mark.asyncio
async def test_field_case_env_symlink_and_next_env_rank_low(
    home: Path, repo: Path, wt: Path
) -> None:
    (wt / ".env.local").symlink_to(repo / ".env.local")
    (wt / "next-env.d.ts").write_text('/// <reference types="next" />\n')
    assert prov.plain_cleanup_ignored_inventory(str(wt)) == prov.IgnoredInventory(0, 0)
    _assert_low(home, repo, wt)
    context = CallContext(cwd=str(home), env=_env(home), scratchpad=None, session_id=ME)
    for command in _commands(repo, wt):
        report = preview_tool_call("bash", {"command": command}, context, settings=GuardSettings())
        assert report.builtin_effect == BUILTIN_PASS, report.render()
        judge = _Judge()
        result = await _check(home, command, judge)
        assert result is None or not result.blocked, result and result.text
        assert judge.prompts == []
    # Classification only: nothing was removed, the link's target is intact.
    assert wt.is_dir() and os.path.islink(wt / ".env.local")
    assert (repo / ".env.local").read_text() == "SYNTHETIC=1\n"


@pytest.mark.parametrize("name", [".env.local", ".env"])
def test_regular_env_file_still_holds(home: Path, repo: Path, wt: Path, name: str) -> None:
    (wt / name).write_text("only copy of local data\n")
    _assert_held(home, repo, wt)
    assert prov._interrupted_cleanup_state(str(wt), None) == "dirty"


def test_env_symlink_sheds(home: Path, repo: Path, wt: Path) -> None:
    (wt / ".env").symlink_to(repo / ".env.local")
    _assert_low(home, repo, wt)
    # --force: an own tree sheds the link as plain removal does; a tree with
    # no ledger row keeps base's hold there (#4503 R1/Q1, stale-link pin).
    assert prov._interrupted_cleanup_state(str(wt), None, own_regenerable=True) == "clean"
    assert prov._interrupted_cleanup_state(str(wt), None) == "dirty"


def test_next_env_directory_holds(home: Path, repo: Path, wt: Path) -> None:
    (wt / "next-env.d.ts").mkdir()
    (wt / "next-env.d.ts" / "data.txt").write_text("not Next's\n")
    _assert_held(home, repo, wt)
    assert prov._interrupted_cleanup_state(str(wt), None) == "dirty"


def test_next_env_regular_file_sheds_plain_and_own_force(home: Path, repo: Path, wt: Path) -> None:
    (wt / "next-env.d.ts").write_text('/// <reference types="next" />\n')
    _assert_low(home, repo, wt)
    assert prov._interrupted_cleanup_state(str(wt), None, own_regenerable=True) == "clean"
    assert prov._interrupted_cleanup_state(str(wt), None) == "dirty"


def test_dangling_symlink_sheds(home: Path, repo: Path, wt: Path) -> None:
    (wt / ".env.local").symlink_to(repo / "does-not-exist")
    assert os.path.islink(wt / ".env.local") and not (wt / ".env.local").exists()
    _assert_low(home, repo, wt)


@pytest.mark.parametrize("name", ["next-env.d.ts", ".env.local"])
def test_fifo_with_ignored_name_is_no_witness(wt: Path, name: str) -> None:
    # Git status never lists a FIFO (special files are skipped), so the
    # inventory has no row to judge; the witnesses themselves must still
    # refuse a special file under a witnessed name (lstat, no follow).
    os.mkfifo(wt / name)
    assert not prov._ignored_symlink(str(wt), name)
    assert not prov._next_env_types_file(str(wt), name)
    assert not prov._ignored_symlink(str(wt), "missing")  # OSError fails closed


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["rm -rf ~", "rm -rf ~/.local-operator", 'rm -rf "$EMPTY"/'])
async def test_critical_controls_unchanged(home: Path, command: str) -> None:
    judge = _Judge("ALLOW ledger")
    outcome = await _check(home, command, judge)
    assert outcome is not None and outcome.blocked, command
    assert judge.prompts == [], command


@pytest.fixture
def branch_wt(home: Path, repo: Path) -> Path:
    """A NON-detached, pushed branch tree, so the finished-tree route applies.

    Security review R1-MAJOR-1 (PR #4561): the detached fixture never reaches
    ``_merged_tree_sheds``, which is why a name-only ``next-env.d.ts`` entry in
    ``_REGENERABLE`` slipped past the directory test.
    """
    wt = repo / ".worktrees" / "feat-wt"
    _git("worktree", "add", "-q", "-b", "feat/wt", str(wt), "origin/main", cwd=repo)
    _git("push", "-q", "origin", "feat/wt", cwd=repo)
    _git("fetch", "-q", "origin", cwd=repo)
    assert prov.recorded_creators(str(wt)) == []
    return wt


def test_next_env_directory_holds_on_finished_branch_tree(
    home: Path, repo: Path, branch_wt: Path
) -> None:
    (branch_wt / "next-env.d.ts").mkdir()
    (branch_wt / "next-env.d.ts" / "data.txt").write_text("not Next's\n")
    assert prov.worktree_state(str(branch_wt), own=False, session_id=ME) == "dirty"
    _assert_held(home, repo, branch_wt)


def test_next_env_regular_file_sheds_on_finished_branch_tree(
    home: Path, repo: Path, branch_wt: Path
) -> None:
    (branch_wt / "next-env.d.ts").write_text('/// <reference types="next" />\n')
    assert prov.worktree_state(str(branch_wt), own=False, session_id=ME) == "clean"
    _assert_low(home, repo, branch_wt)
