"""Plain removal of a SIBLING subagent's finished worktree holding build output.

Field case (guard 0.113.0, 2026-11-24 20:36Z): root session R is still
running. Its subagent A ran ``git worktree add .worktrees/deploy-preserve-
digest-pin`` on Nov 20 and has since ended; other children of R built Rust
``target/`` in the tree. Today child B (ancestors [R]) ran a plain ``git
worktree remove .worktrees/deploy-preserve-digest-pin`` (no ``--force``) from
the primary checkout. The tree was clean, its head merged into origin's
default branch by a true merge commit, the remote branch deleted, GitLab
remote, and the only ignored entry was ``target/``. The guard refused:
"linked worktree state: 1 ignored entry the guard cannot prove regenerable".

Root cause (read-only probe of the field tree): ``target/`` held 51,744
entries, so ``worktree_state``'s nested-repository walk (``_holds_git``:
depth 4, 20,000 entries) ran out, an exhausted walk counts as "holds a
repository", the tree read ``dirty`` and ``finished_foreign_worktree``
answered ``state-dirty``. Ledger attribution, the running root and the
GitLab forge were NOT the cause. The rig reproduces it with 20,001 files.

Operator directive (Damian via Aida, 2026-11-24): agents must be able to
clean up their own task's finished worktrees without asking; the critical
class, consent rules and untrusted-message rules do not change.

Fixtures are tiny real git repositories with a local bare ``origin``; the
ledger row is written through the real ``commit_creations``, lineage through
the real ``record_lineage``. Only the session registry scan, the forge and the
live-cwd holder scans are synthetic. Every session id is synthetic.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.guardrails import provenance as prov
from local_operator.guardrails.check import CallContext, GuardSettings, check_tool_call
from local_operator.guardrails.classify import Risk, classify_bash
from local_operator.guardrails.preview import BUILTIN_PASS, preview_tool_call
from local_operator.session.runtime import registry

R = "1bba7ac8ecd0"  # the root session, still running
A = "aaaaaaaaaaa1"  # R's subagent that made the tree; ended
B = "8fb21a632af0"  # R's subagent removing it (the caller)
F = "fffffffffff9"  # a session outside R's task tree
NAME = "deploy-preserve-digest-pin"


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@e", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.fixture
def cwds() -> dict[str, str]:
    """Recorded session cwds (default ``/``), read by the registry fixture."""
    return {}


@pytest.fixture
def running(monkeypatch: pytest.MonkeyPatch, cwds: dict[str, str]) -> dict[str, str]:
    """Session registry: ``{sid: state}``, read by the REAL ``_running_owners``."""
    sessions = {R: "live", A: "ended", B: "live", F: "live"}

    def scan(*_a: Any, **_kw: Any) -> list[tuple[Any, str]]:
        return [
            (SimpleNamespace(session_id=sid, cwd=cwds.get(sid, "/")), state)
            for sid, state in sessions.items()
        ]

    monkeypatch.setattr(registry, "scan", scan)
    return sessions


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, running: dict[str, str]) -> Path:
    home = tmp_path / "home"
    state = home / ".local-operator"
    for sid in (R, A, B, F):
        (state / "sessions" / sid / "scratchpad").mkdir(parents=True)
    (state / "config.yml").write_text("version: 1\n")
    monkeypatch.setattr(prov, "_live_cwd_inside", lambda _, **_kw: False)
    monkeypatch.setattr(prov, "_cwd_entries", lambda *, fresh=False: ())
    monkeypatch.setattr(prov, "cwd_holders", lambda real, sid, **_kw: ())
    monkeypatch.setattr(prov, "_cwd_inventory_cache", None)
    # GitLab remote, MR merged: "no open MR" from every forge lookup.
    monkeypatch.setattr(prov, "_github_repo", lambda _: "group/deploy")
    monkeypatch.setattr(prov, "_gh_prs", lambda repo, branch: [])
    monkeypatch.setattr(prov, "_gh_open_prs", lambda repo, branch: [])
    monkeypatch.setattr(prov, "_glab_mrs", lambda repo, branch: [])
    monkeypatch.setattr(prov, "_glab_open_mrs", lambda repo, branch: [])
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))
    for key in [k for k in os.environ if k.startswith(("CMUX_", "GIT_", "LOP_"))]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    prov.record_lineage(A, origin="subagent", parent=R, state_root=state)
    prov.record_lineage(B, origin="subagent", parent=R, state_root=state)
    return home


def _env(home: Path) -> dict[str, str]:
    return {"HOME": str(home), "LOCAL_OPERATOR_CONFIG_DIR": str(home / ".local-operator")}


@pytest.fixture
def repo(home: Path) -> Path:
    """Primary checkout plus A's merged, clean tree with an ignored ``target/``."""
    origin = home / "origin.git"
    _git("init", "-q", "--bare", "-b", "main", str(origin), cwd=home)
    repo = home / "deploy"
    _git("init", "-q", "-b", "main", str(repo), cwd=home)
    (repo / ".gitignore").write_text(".worktrees/\ntarget/\n.env\ndata/\n")
    (repo / "Cargo.toml").write_text("[package]\nname = 'd'\n")
    _git("add", ".", cwd=repo)
    _git("commit", "-qm", "init", cwd=repo)
    _git("remote", "add", "origin", str(origin), cwd=repo)
    _git("push", "-q", "origin", "main", cwd=repo)
    _git("fetch", "-q", "origin", cwd=repo)
    wt = repo / ".worktrees" / NAME
    command = f"git worktree add -b fix/{NAME} .worktrees/{NAME} origin/main"
    pending = prov.plan_creations(command, cwd=str(repo), env=_env(home))
    assert pending is not None
    subprocess.run(["bash", "-c", command], cwd=repo, check=True, capture_output=True)
    written = prov.commit_creations(
        pending, session_id=A, ancestors=[R], env=_env(home), scratchpad=None
    )
    assert written, "the real ledger write recorded nothing"
    (wt / "deploy.rs").write_text("fn main() {}\n")
    _git("add", ".", cwd=wt)
    _git("commit", "-qm", "fix: preserve digest pin", cwd=wt)
    _git("push", "-q", "origin", f"fix/{NAME}", cwd=wt)
    # A true merge commit (GitLab "merge commit" method), then the remote
    # branch deleted, as in the field.
    _git("merge", "-q", "--no-ff", "-m", "Merge branch 'fix'", f"fix/{NAME}", cwd=repo)
    _git("push", "-q", "origin", "main", cwd=repo)
    _git("push", "-q", "origin", "--delete", f"fix/{NAME}", cwd=repo)
    _git("fetch", "-q", "--prune", "origin", cwd=repo)
    _bulk(wt / "target" / "debug" / "deps")
    (wt / "target" / "debug" / "foo").write_text("bin\n")
    return repo


def _bulk(directory: Path, n: int = prov._WALK_ENTRIES + 1) -> None:
    """More entries than ``_holds_git`` walks: the field ``target/`` shape."""
    directory.mkdir(parents=True)
    for i in range(n):
        (directory / f"d{i}.rlib").touch()


def _wt(repo: Path) -> Path:
    return repo / ".worktrees" / NAME


def _rank(home: Path, repo: Path, command: str, sid: str = B) -> Any:
    return classify_bash(command, cwd=str(repo), env=_env(home), scratchpad=None, session_id=sid)


class _Judge:
    def __init__(self, answer: str = "BLOCK: not verified") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    async def __call__(self, system: str, prompt: str) -> str:
        self.prompts.append(system + "\n" + prompt)
        return self.answer


async def _check(home: Path, repo: Path, command: str, judge: _Judge, sid: str = B) -> Any:
    ctx = CallContext(cwd=str(repo), env=_env(home), scratchpad=None, session_id=sid)
    return await check_tool_call(
        "bash", {"command": command}, ctx, judge=judge, settings=GuardSettings(), forked_judge=judge
    )


def _spellings(repo: Path) -> list[str]:
    return [
        f"git worktree remove .worktrees/{NAME}",
        f"git worktree remove {_wt(repo)}",
    ]


@pytest.mark.asyncio
async def test_a_field_case_is_low_with_no_judge_call(home: Path, repo: Path) -> None:
    for command in _spellings(repo):
        v = _rank(home, repo, command)
        assert v.risk is Risk.LOW, v.targets[0].reason
        ctx = CallContext(cwd=str(repo), env=_env(home), scratchpad=None, session_id=B)
        preview = preview_tool_call("bash", {"command": command}, ctx)
        assert preview.builtin_effect == BUILTIN_PASS, preview.render()
        judge = _Judge()
        outcome = await _check(home, repo, command, judge)
        assert outcome is None or not outcome.blocked, outcome
        assert judge.prompts == []
    assert _wt(repo).exists()  # classification never removed anything


def test_b_ignored_env_beside_target_sheds_but_nested_repo_holds(home: Path, repo: Path) -> None:
    """#4565 (operator directive 2026-11-24): a merged, finished tree sheds
    ANY ignored entry, an ignored ``.env`` included; only a nested repository
    (or an unwalkable entry) still holds. Was "``.env`` is held" before."""
    (_wt(repo) / ".env").write_text("TOKEN=synthetic\n")
    for command in _spellings(repo):
        assert _rank(home, repo, command).risk is Risk.LOW
    _git("init", "-q", str(_wt(repo) / "vendored"), cwd=home)
    for command in _spellings(repo):
        assert _rank(home, repo, command).risk is not Risk.LOW


def test_c_running_creator_is_not_low(home: Path, repo: Path, running: dict[str, str]) -> None:
    running[A] = "live"
    for command in _spellings(repo):
        v = _rank(home, repo, command)
        assert v.risk is not Risk.LOW, v.targets[0].reason


def test_c2_common_ancestor_cwd_inside_still_holds(
    home: Path, repo: Path, cwds: dict[str, str]
) -> None:
    cwds[R] = str(_wt(repo))
    v = _rank(home, repo, _spellings(repo)[0])
    assert v.risk is not Risk.LOW, v.targets[0].reason


def test_d_foreign_caller_gets_main_finished_tree_policy(home: Path, repo: Path) -> None:
    """A caller outside R's tree gets what main already gives a finished
    foreign tree holding only regenerable-named output (the 2026-11-24
    ``merged_regenerable`` route): LOW via the finished-tree proof, never via
    ownership. The fix only stops the walk's SIZE from hiding that fact."""
    v = _rank(home, repo, _spellings(repo)[1], sid=F)
    assert v.risk is Risk.LOW, v.targets[0].reason
    assert "created by" not in v.targets[0].reason  # not an ownership grant
    # #4565: an ignored ``.env`` no longer holds a merged tree; a nested
    # repository still does.
    (_wt(repo) / ".env").write_text("TOKEN=local-only-fixture")
    assert _rank(home, repo, _spellings(repo)[1], sid=F).risk is Risk.LOW
    _git("init", "-q", str(_wt(repo) / "vendored"), cwd=home)
    assert _rank(home, repo, _spellings(repo)[1], sid=F).risk is not Risk.LOW


def test_g_exhausted_walk_under_other_name_is_held(home: Path, repo: Path) -> None:
    """Only a ``_REGENERABLE`` name gets the full-depth re-walk."""
    _bulk(_wt(repo) / "data" / "cache")
    v = _rank(home, repo, _spellings(repo)[0])
    assert v.risk is not Risk.LOW, v.targets[0].reason


def test_h_deep_nested_repository_in_target_is_held(home: Path, repo: Path) -> None:
    """A ``.git`` below ``_holds_git``'s depth, past its entry bound, still holds."""
    deep = _wt(repo) / "target" / "a" / "b" / "c" / "d" / "e" / "vendored"
    _git("init", "-q", str(deep), cwd=home)
    real = os.path.realpath(_wt(repo))
    assert prov.worktree_state(real, facts={}) == "dirty"
    v = _rank(home, repo, _spellings(repo)[0])
    assert v.risk is not Risk.LOW, v.targets[0].reason


def test_i_field_tree_read_dirty_before_the_fix(home: Path, repo: Path) -> None:
    """Pins the failing predicate: the old walk alone runs out on ``target/``."""
    target = os.path.join(os.path.realpath(_wt(repo)), "target")
    assert prov._git_inside(target) == prov.GitInside(target, exhausted=True)
    assert prov._ignored_entry_holds_git(target, "target") is False


def test_e_open_mr_is_refused(home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    change = prov.ForgeChange(7, "open", "")
    monkeypatch.setattr(prov, "_gh_prs", lambda repo, branch: [change])
    monkeypatch.setattr(prov, "_glab_mrs", lambda repo, branch: [change])
    v = _rank(home, repo, _spellings(repo)[0])
    assert v.risk is Risk.CRITICAL, v.targets[0].reason


@pytest.mark.parametrize(
    "command",
    [f"git worktree remove --force .worktrees/{NAME}", f"rm -rf .worktrees/{NAME}"],
)
def test_f_force_and_rm_rf_are_not_low(home: Path, repo: Path, command: str) -> None:
    v = _rank(home, repo, command)
    assert v.risk is not Risk.LOW, v.targets[0].reason
