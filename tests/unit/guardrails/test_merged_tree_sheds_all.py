"""A finished, merged worktree sheds ALL ignored payload except nested repositories.

Field cases 2026-11-24 (guard 0.113.0, session 00e6a1cd8f62, disk at 4 GiB
free): three merged ``local-operator-ui`` worktrees (``composer-typing-perf``,
``desktop-inset-workspace-preview``, ``i18n-state-literal-guard``) were refused
for ``unverified_ignored`` 3/3/1. Their ignored rows: ``.DS_Store`` files, an
untagged ``.cache/typecheck/``, a REAL ``node_modules`` directory (the
shared-link witness needs a symlink), ``out/`` and
``scripts/_shared-composer-*.mjs``. Operator directive: "A directory your git
worktree list proves is yours and merged should be removable whatever its
ignored build output contains."

With ``finished_foreign_worktree``'s proof (registered, merged on origin, no
open PR, no running owner, no live cwd) plain ``git worktree remove`` now
sheds every ignored entry; a nested repository (``.git`` file/dir, bare
layout, a symlink into one) or an unreadable walk still holds. The proof
itself, ``--force`` and the critical classes are unchanged. See
``provenance._merged_tree_sheds``.
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

ME = "aaaaaaaaaaa1"  # the cleaning session
DONE = "bbbbbbbbbbb2"  # the ended session that made the tree
LIVE = "ccccccccccc3"  # a session still running

EXCLUDES = "node_modules/\n.cache/\nout/\n_shared-*.mjs\n.DS_Store\nvendored/\nlinked/\n"


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
    for sid in (ME, DONE, LIVE):
        (state / "sessions" / sid / "scratchpad").mkdir(parents=True)
    (state / "config.yml").write_text("version: 1\n")
    monkeypatch.setattr(prov, "_live_cwd_inside", lambda _, **_kw: False)
    monkeypatch.setattr(prov, "cwd_holders", lambda real, sid, **_kw: ())
    monkeypatch.setattr(prov, "_github_repo", lambda _: "example/repo")
    monkeypatch.setattr(prov, "_gh_prs", lambda repo, branch: None)
    monkeypatch.setattr(prov, "_gh_open_prs", lambda repo, branch: None)
    monkeypatch.setattr(prov, "_running_owners", lambda real, sid, strict=False, **_: ())
    monkeypatch.setattr(prov, "_running_session_ids", lambda: {LIVE}, raising=False)
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


def _record(home: Path, path: Path, creator: str) -> None:
    prov.write_records(
        [(os.path.realpath(path), "worktree")],
        owners=[creator],
        creator=creator,
        command=f"git worktree add {path}",
        state_root=home / ".local-operator",
    )


@pytest.fixture
def repo(home: Path) -> Path:
    origin = home / "origin.git"
    _git("init", "-q", "--bare", "-b", "main", str(origin), cwd=home)
    repo = home / "repo"
    _git("init", "-q", "-b", "main", str(repo), cwd=home)
    (repo / ".gitignore").write_text(".worktrees/\n")
    (repo / "a.txt").write_text("a\n")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "build.mjs").write_text("export {}\n")
    _git("add", ".", cwd=repo)
    _git("commit", "-qm", "init", cwd=repo)
    _git("remote", "add", "origin", str(origin), cwd=repo)
    _git("push", "-q", "origin", "main", cwd=repo)
    (repo / ".git" / "info" / "exclude").write_text(EXCLUDES)
    return repo


def _tree(home: Path, repo: Path, name: str, *, merged: bool = True, creator: str = DONE) -> Path:
    wt = repo / ".worktrees" / name
    _git("worktree", "add", "-q", "-b", f"fix/{name}", str(wt), "origin/main", cwd=repo)
    (wt / f"{name}.txt").write_text("done\n")
    _git("add", ".", cwd=wt)
    _git("commit", "-qm", f"fix {name}", cwd=wt)
    if merged:
        _git("merge", "-q", "--ff-only", f"fix/{name}", cwd=repo)
        _git("push", "-q", "origin", "main", cwd=repo)
    _git("fetch", "-q", "origin", cwd=repo)
    _field_payload(wt)
    _record(home, wt, creator)
    return wt


def _field_payload(wt: Path) -> None:
    """The ignored rows read from the three field trees (synthetic bytes)."""
    pkg = wt / "node_modules" / "react"  # a real DIRECTORY, not the shared link
    pkg.mkdir(parents=True)
    (pkg / "index.js").write_text("module.exports = {}\n")
    (pkg / "package.json").write_text('{"name": "react"}\n')
    tc = wt / ".cache" / "typecheck"  # no CACHEDIR.TAG
    tc.mkdir(parents=True)
    (tc / "node.tsbuildinfo").write_text('{"version": "5.6.3", "root": []}\n')
    (wt / "out" / "main").mkdir(parents=True)
    (wt / "out" / "main" / "index.js").write_text("compiled\n")
    (wt / "scripts" / "_shared-composer-abc123.mjs").write_text("bundle\n")
    for at in (wt, wt / "scripts", wt / "node_modules"):
        (at / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1" + os.urandom(32))


def _rank(home: Path, command: str, cwd: Path | None = None) -> Any:
    return classify_bash(
        command, cwd=str(cwd or home), env=_env(home), scratchpad=None, session_id=ME
    )


class _Judge:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def __call__(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        return "BLOCK: synthetic judge"


@pytest.mark.asyncio
@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
async def test_merged_tree_with_field_payload_is_low_with_no_judge(
    home: Path, repo: Path, relative: bool
) -> None:
    """(a) The field payload: LOW, preview pass, zero judge calls."""
    wt = _tree(home, repo, "composer-typing-perf")
    ignored = [r for r in _git("status", "--porcelain", "--ignored", cwd=wt).splitlines()]
    assert all(r.startswith("!! ") for r in ignored) and len(ignored) >= 5, ignored
    if relative:
        command, cwd = "git worktree remove .worktrees/composer-typing-perf", repo
    else:
        command, cwd = f"git -C {repo} worktree remove -- {wt}", home
    ranked = _rank(home, command, cwd)
    assert ranked.risk is Risk.LOW, ranked.reasons
    ctx = CallContext(cwd=str(cwd), env=_env(home), scratchpad=None, session_id=ME)
    report = preview_tool_call("bash", {"command": command}, ctx, settings=GuardSettings())
    assert report.builtin_effect == BUILTIN_PASS, report.render()
    judge = _Judge()
    result = await check_tool_call(
        "bash",
        {"command": command},
        ctx,
        judge=judge,
        settings=GuardSettings(),
        forked_judge=judge,
    )
    assert (result is None or not result.blocked) and not judge.prompts, result


def _nested(wt: Path, shape: str, outside: Path) -> None:
    if shape == "git-dir":
        subprocess.run(["git", "init", "-q", str(wt / "node_modules" / "react" / "x")], check=True)
    elif shape == "git-file":
        (wt / "node_modules" / "react" / ".git").write_text("gitdir: /elsewhere\n")
    elif shape == "bare":
        subprocess.run(
            ["git", "init", "-q", "--bare", str(wt / "node_modules" / "cache" / "mirror")],
            check=True,
        )
    elif shape == "named-git":
        # An ignored entry literally named ``.git`` below an ignored dir.
        (wt / "vendored" / "lib" / ".git").mkdir(parents=True)
        (wt / "vendored" / "lib" / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    elif shape == "symlink-to-repo":
        subprocess.run(["git", "init", "-q", str(outside)], check=True)
        (wt / "linked").symlink_to(outside)
    elif shape == "symlink-to-git-dir":
        subprocess.run(["git", "init", "-q", str(outside)], check=True)
        (wt / "linked").symlink_to(outside / ".git")


@pytest.mark.parametrize(
    "shape",
    ["git-dir", "git-file", "bare", "named-git", "symlink-to-repo", "symlink-to-git-dir"],
)
def test_nested_repository_in_payload_holds(
    home: Path, repo: Path, tmp_path: Path, shape: str
) -> None:
    """(b) A nested repository anywhere in the ignored payload keeps the hold."""
    wt = _tree(home, repo, f"nested-{shape}")
    _nested(wt, shape, tmp_path / "outside")
    _record(home, wt, DONE)
    ranked = _rank(home, f"git -C {repo} worktree remove -- {wt}")
    assert ranked.risk is not Risk.LOW, ranked.reasons


def test_unreadable_payload_holds(home: Path, repo: Path, request: pytest.FixtureRequest) -> None:
    """(b) A directory the walk cannot list holds (fail closed)."""
    wt = _tree(home, repo, "unreadable")
    deep = wt / "out" / "main"
    deep.chmod(0)
    request.addfinalizer(lambda: deep.chmod(0o755))
    if os.access(deep, os.R_OK):
        pytest.skip("directory modes are not enforced for this user")
    ranked = _rank(home, f"git -C {repo} worktree remove -- {wt}")
    assert ranked.risk is not Risk.LOW, ranked.reasons


def test_unmerged_tree_is_unchanged(home: Path, repo: Path) -> None:
    """(c) Unmerged: the hold stands whatever the payload."""
    wt = _tree(home, repo, "unmerged", merged=False)
    ranked = _rank(home, f"git -C {repo} worktree remove -- {wt}")
    assert ranked.risk is Risk.HIGH and ranked.targets[0].cleanup_hold, ranked.reasons


def test_open_pr_stays_critical(home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """(d) An open PR/MR refuses outright, payload notwithstanding."""
    wt = _tree(home, repo, "open-pr")
    monkeypatch.setattr(
        prov,
        "worktree_activity",
        lambda *_: prov.WorktreeActivity("fix/open-pr", "open", pr_number=7),
    )
    ranked = _rank(home, f"git -C {repo} worktree remove -- {wt}")
    assert ranked.risk is Risk.CRITICAL, ranked.reasons


def _detached(home: Path, repo: Path, name: str, *, merged: bool = True) -> Path:
    """A ``git worktree add --detach`` review/QA tree (five of Aida's nine)."""
    wt = _tree(home, repo, f"{name}-src", merged=merged)
    head = _git("rev-parse", "HEAD", cwd=wt).strip()
    out = repo / ".worktrees" / name
    _git("worktree", "add", "-q", "--detach", str(out), head, cwd=repo)
    _field_payload(out)
    # Ignored, as in the field repos' .gitignore (else it reads untracked).
    with open(repo / ".git" / "info" / "exclude", "a") as fh:
        fh.write("__pycache__/\n")
    (out / "__pycache__").mkdir()
    (out / "__pycache__" / "m.cpython-314.pyc").write_bytes(b"\x00pyc")
    return out


def _force(repo: Path, wt: Path) -> str:
    # The operator's exact spelling from the 2026-11-25 matrix.
    return f"git -C {repo} worktree remove --force -- {wt}"


@pytest.mark.parametrize("shape", ["branch", "detached"])
def test_force_on_merged_tree_with_ignored_payload_is_low(
    home: Path, repo: Path, shape: str
) -> None:
    """(e) Aida's live matrix (2026-11-25): nine git-clean ``.worktrees/*``
    trees, HEAD an ancestor of origin/main, ranked HIGH ("preservation check
    failed") for ``--force`` only because of ignored ``__pycache__``,
    ``.DS_Store``, ``node_modules``. Branch and detached shapes both go LOW."""
    wt = _tree(home, repo, "forced") if shape == "branch" else _detached(home, repo, "forced-d")
    ranked = _rank(home, _force(repo, wt))
    assert ranked.risk is Risk.LOW, ranked.reasons
    assert "verified removal payload" in ranked.targets[0].reason


@pytest.mark.parametrize("shape", ["branch", "detached"])
@pytest.mark.parametrize("control", ["untracked", "tracked-edit", "nested-repo", "next-env-dir"])
def test_force_on_merged_tree_still_holds_real_data(
    home: Path, repo: Path, shape: str, control: str
) -> None:
    """(e) ``--force`` also deletes tracked/untracked work: those, a nested
    repository and #4561's non-regular ``next-env.d.ts`` keep the hold."""
    wt = _tree(home, repo, "held") if shape == "branch" else _detached(home, repo, "held-d")
    if control == "untracked":
        (wt / "untracked.txt").write_text("work\n")
    elif control == "tracked-edit":
        (wt / "scripts" / "build.mjs").write_text("edited\n")
    elif control == "nested-repo":
        subprocess.run(["git", "init", "-q", str(wt / "node_modules" / "react" / "x")], check=True)
    else:
        with open(repo / ".git" / "info" / "exclude", "a") as fh:
            fh.write("next-env.d.ts\n")
        (wt / "next-env.d.ts").mkdir()
        (wt / "next-env.d.ts" / "data.txt").write_text("not Next's\n")
    ranked = _rank(home, _force(repo, wt))
    assert ranked.risk is not Risk.LOW, ranked.reasons


def test_force_on_unmerged_trees_holds(home: Path, repo: Path) -> None:
    """(e) Unmerged branch and a detached HEAD origin lacks both hold."""
    for wt in (
        _tree(home, repo, "unmerged-f", merged=False),
        _detached(home, repo, "unmerged-d", merged=False),
    ):
        ranked = _rank(home, _force(repo, wt))
        assert ranked.risk is not Risk.LOW, (wt, ranked.reasons)


def test_force_on_merged_tree_with_live_owner_holds(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(e) A running owner keeps the hold on both shapes."""
    trees = (_tree(home, repo, "owned"), _detached(home, repo, "owned-d"))
    monkeypatch.setattr(prov, "_running_owners", lambda real, sid, strict=False, **_: (LIVE,))
    for wt in trees:
        ranked = _rank(home, _force(repo, wt))
        assert ranked.risk is not Risk.LOW, (wt, ranked.reasons)


def test_unrelated_dirty_tree_is_unchanged(home: Path, repo: Path) -> None:
    """Tracked/untracked work still holds: only the ignored allowance widened."""
    wt = _tree(home, repo, "dirty")
    (wt / "untracked.txt").write_text("work\n")
    ranked = _rank(home, f"git -C {repo} worktree remove -- {wt}")
    assert ranked.risk is not Risk.LOW, ranked.reasons


@pytest.mark.parametrize("command", ["rm -rf ~", "rm -rf ~/.local-operator", 'rm -rf "$EMPTY"/'])
def test_critical_deletes_are_unchanged(home: Path, command: str) -> None:
    """(g) Home, the state root and an empty expansion keep their rank."""
    ranked = _rank(home, command)
    assert ranked.risk is Risk.CRITICAL, ranked.reasons


@pytest.mark.parametrize("creator", [ME, None], ids=["own-ledger-row", "no-ledger-row"])
def test_deregistered_worktree_remnant_pin(home: Path, repo: Path, creator: str | None) -> None:
    """(f) Scope B pin, no code change (see the PR's design note).

    The field remnant's shape (``orphan-ask-cascade``): its ``.git`` pointer
    FILE survived, the admin directory is gone, ignored build output remains.
    With this session's ledger row, ``rm -rf`` is LOW on base already
    (``owned_record``). With NO row (the field case had none) it stays off
    LOW, so a ledger-keyed rule would not have lifted the field refusal.
    """
    import shutil

    wt = repo / ".worktrees" / "remnant"
    _git("worktree", "add", "-q", "-b", "fix/remnant", str(wt), "origin/main", cwd=repo)
    if creator is not None:
        _record(home, wt, creator)
    (wt / "out").mkdir()
    (wt / "out" / "a.js").write_text("compiled\n")
    shutil.rmtree(repo / ".git" / "worktrees" / "remnant")
    assert (wt / ".git").is_file()
    ranked = _rank(home, f"rm -rf {wt}", repo)
    if creator is None:
        assert ranked.risk is not Risk.LOW, ranked.reasons
    else:
        assert ranked.risk is Risk.LOW, ranked.reasons
