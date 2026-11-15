"""An own DETACHED worktree with uncommitted scratch is the creator's to remove.

Operator directive (2026-11-14, verbatim, relayed by Aida): "the safety guards
should never refuse being able to clean up uncommitted files that look like
scratch, safe to delete, etc. it needs to be tuned … this sort of ask should
never come up again."

Field case: mesh node lop-remote-node-2 (Linux), session 99ffc68d3d05 ran
``git worktree add --detach ~/local-operator/.worktrees/attr3880-ci``, then
``git merge-tree`` + ``git read-tree`` left an uncommitted index and working
tree changes; ``git worktree remove`` of its own tree was refused by the judge
("busy detached worktree with substantial uncommitted changes" — judge
wording; no deterministic branch writes it). Two causes, both pinned here:

1. ``classify`` sent every own DIRTY tree to the judge, even a detached one
   whose HEAD a ref keeps — nothing but the creator's scratch can be lost.
2. On Linux ``creation_stamp`` is the root's ctime, which moves on any root
   entry change, so the ledger row stopped matching and the own tree read as
   foreign. ``WORKTREE_GITFILE_KIND`` adds the ``.git`` file as a witness.

Every test builds real repositories in a tmp HOME; nothing touches a live
session or the operator's ledger.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from local_operator.guardrails import provenance as prov
from local_operator.guardrails.check import (
    SUPPORTED_DELETE_PATHS,
    CallContext,
    GuardSettings,
    check_tool_call,
)
from local_operator.guardrails.classify import Risk, classify_bash

S = "99ffc68d3d05"  # the field session (synthetic store here)
S2 = "222222222222"  # an unrelated session


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    state = home / ".local-operator"
    for sid in (S, S2):
        (state / "sessions" / sid / "scratchpad").mkdir(parents=True)
    (state / "config.yml").write_text("version: 1\n")
    # No live cwd, no forge, no running owner: the synthetic trees are idle.
    monkeypatch.setattr(prov, "_live_cwd_inside", lambda _: False)
    # The own-tree path scans fresh through ``cwd_holders`` -> ``_cwd_entries``
    # and never asks ``_live_cwd_inside``; idle must hold there too, or a
    # host with an unreadable same-uid ``/proc/<pid>/cwd`` (GitHub's Linux
    # runner, train t6) fails the inventory closed and reads every tree busy.
    # ``test_own_worktree_cwd_holders``'s ``live`` fixture restores the real one.
    monkeypatch.setattr(prov, "_cwd_entries", lambda *, fresh=False: ())
    monkeypatch.setattr(prov, "_github_repo", lambda _: "damianvtran/local-operator")
    monkeypatch.setattr(prov, "_gh_prs", lambda repo, branch: [])
    monkeypatch.setattr(prov, "_gh_open_prs", lambda repo, branch: [])
    monkeypatch.setattr(prov, "_glab_mrs", lambda repo, branch: [])
    monkeypatch.setattr(prov, "_glab_open_mrs", lambda repo, branch: [])
    monkeypatch.setattr(prov, "_running_owners", lambda real, sid: ())
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


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@e", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run_recorded(home: Path, command: str, *, sid: str = S, cwd: Path | None = None) -> None:
    """Run ``command`` the way the harness loop does: plan, execute, commit."""
    env = {**_env(home), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    where = str(cwd or home)
    pending = prov.plan_creations(command, cwd=where, env=env)
    subprocess.run(
        ["bash", "-c", command],
        cwd=where,
        env={**os.environ, **env},
        check=True,
        capture_output=True,
    )
    assert pending is not None, command
    prov.commit_creations(pending, session_id=sid, env=env, scratchpad=None)


@pytest.fixture
def repo(home: Path) -> Path:
    """``~/local-operator`` with ``main`` and a ``feature`` branch to merge."""
    repo = home / "local-operator"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    (repo / ".gitignore").write_text(".worktrees/\n")
    (repo / "a.py").write_text("a = 1\n")
    (repo / "b.py").write_text("b = 1\n")
    _git("add", ".", cwd=repo)
    _git("commit", "-qm", "init", cwd=repo)
    _git("checkout", "-qb", "feature", cwd=repo)
    (repo / "b.py").write_text("b = 2\n")
    _git("commit", "-qam", "feature", cwd=repo)
    _git("checkout", "-q", "main", cwd=repo)
    # Pre-existing, as in the field (~/local-operator/.worktrees holds many
    # sessions' trees): no session's ledger records the container itself, so
    # only the tree's own row can vouch for it.
    (repo / ".worktrees").mkdir()
    return repo


def _attr3880(home: Path, repo: Path, *, name: str = "attr3880-ci", sid: str = S) -> Path:
    """The field shape: detached add, merge-tree + read-tree index, a modified file."""
    _run_recorded(
        home, f"cd ~/local-operator && git worktree add --detach .worktrees/{name}", sid=sid
    )
    wt = repo / ".worktrees" / name
    tree = _git("merge-tree", "--write-tree", "main", "feature", cwd=wt).splitlines()[0]
    _git("read-tree", tree, cwd=wt)  # uncommitted index
    (wt / "a.py").write_text("a = 'scratch'\n")  # uncommitted working-tree change
    return wt


def _rank(home: Path, command: str, *, sid: str = S) -> tuple[Risk, str]:
    v = classify_bash(command, cwd=str(home), env=_env(home), scratchpad=None, session_id=sid)
    return v.risk, " | ".join(t.reason for t in v.targets)


FIELD_COMMANDS = [
    "cd ~/local-operator && git worktree remove --force .worktrees/attr3880-ci",
    "git worktree remove --force ~/local-operator/.worktrees/attr3880-ci",
    "git -C ~/local-operator worktree remove -f .worktrees/attr3880-ci",
    "git worktree remove ~/local-operator/.worktrees/attr3880-ci",
    "rm -rf ~/local-operator/.worktrees/attr3880-ci",
]


@pytest.mark.parametrize("command", FIELD_COMMANDS)
def test_field_shape_own_detached_dirty_is_low(home: Path, repo: Path, command: str) -> None:
    wt = _attr3880(home, repo)
    assert prov.worktree_state(str(wt), own=True) == "dirty"  # really dirty
    risk, reason = _rank(home, command)
    assert risk is Risk.LOW, reason


def test_field_shape_reason_names_detached_scratch(home: Path, repo: Path) -> None:
    _attr3880(home, repo)
    risk, reason = _rank(home, FIELD_COMMANDS[0])
    assert risk is Risk.LOW
    assert "own-created detached worktree (no branch" in reason
    assert "M a.py" in reason  # the scratch it drops is still named


@pytest.mark.asyncio
async def test_field_shape_settles_with_no_judge_call(home: Path, repo: Path) -> None:
    _attr3880(home, repo)
    prompts: list[str] = []

    async def judge(system: str, prompt: str) -> str:
        prompts.append(prompt)
        return "BLOCK: should not be asked"

    ctx = CallContext(
        cwd=str(home), env=_env(home), scratchpad=None, session_id=S, intent="clean up"
    )
    for command in FIELD_COMMANDS:
        outcome = await check_tool_call(
            "bash",
            {"command": command},
            ctx,
            judge=None,
            settings=GuardSettings(),
            forked_judge=judge,
        )
        assert outcome is None or not outcome.blocked, outcome and outcome.text
    assert prompts == []


@pytest.mark.parametrize(
    "add",
    [
        "cd ~/local-operator && git worktree add --detach .worktrees/x",
        "cd ~/local-operator && git worktree add -d .worktrees/x",
        "git -C ~/local-operator worktree add --detach .worktrees/x",
        "git -C ~/local-operator worktree add --detach .worktrees/x HEAD",
        "cd ~/local-operator && git worktree add .worktrees/x --detach main",
    ],
)
def test_ledger_records_every_detached_add_form(home: Path, repo: Path, add: str) -> None:
    """Item 3: the add forms record the tree itself (and its ``.git`` witness)."""
    _run_recorded(home, add)
    wt = repo / ".worktrees" / "x"
    rec = prov.owned_record(str(wt), S)
    assert rec is not None and rec.path == os.path.realpath(wt) and rec.kind == "worktree"


def test_linux_ctime_stamp_keeps_matching_through_the_git_file(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Item 3, Linux hosts: a root entry change no longer orphans the row."""
    monkeypatch.setattr(prov, "creation_stamp", lambda st: int(st.st_ctime_ns))
    wt = _attr3880(home, repo)
    (wt / "new-root-file.txt").write_text("scratch\n")
    os.unlink(wt / "b.py")
    (wt / "b.py").write_text("b = 3\n")  # unlink+create, git's own rewrite shape
    assert prov.owned_record(str(wt), S) is not None
    risk, reason = _rank(home, FIELD_COMMANDS[0])
    assert risk is Risk.LOW, reason


def test_linux_ctime_stamp_without_witness_fails_closed(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: the ``.git`` row is what holds it; a rewritten file does not."""
    monkeypatch.setattr(prov, "creation_stamp", lambda st: int(st.st_ctime_ns))
    wt = _attr3880(home, repo)
    (wt / "new-root-file.txt").write_text("scratch\n")
    gitfile = wt / ".git"
    body = gitfile.read_text()
    gitfile.unlink()
    gitfile.write_text(body)  # a NEW .git inode generation (as worktree repair would)
    assert prov.owned_record(str(wt), S) is None


# -- 2a–2e: what still refuses or goes to the judge ---------------------------


def test_2a_branch_worktree_with_unpushed_commit_and_changes_stays_high(
    home: Path, repo: Path
) -> None:
    _run_recorded(home, "cd ~/local-operator && git worktree add -b wip .worktrees/wip")
    wt = repo / ".worktrees" / "wip"
    (wt / "a.py").write_text("a = 'unpushed'\n")
    _git("commit", "-qam", "unpushed", cwd=wt)
    (wt / "b.py").write_text("b = 'dirty'\n")
    for command in (
        "git worktree remove --force ~/local-operator/.worktrees/wip",
        "rm -rf ~/local-operator/.worktrees/wip",
    ):
        risk, reason = _rank(home, command)
        assert risk is Risk.HIGH, (command, reason)
        assert "own-created detached" not in reason


def test_2b_unrecorded_or_other_sessions_detached_tree_stays_high(home: Path, repo: Path) -> None:
    _attr3880(home, repo, sid=S2)  # another session created it
    _git("worktree", "add", "-q", "--detach", ".worktrees/nobody", cwd=repo)  # no ledger row
    (repo / ".worktrees" / "nobody" / "a.py").write_text("a = 'x'\n")
    for path in ("attr3880-ci", "nobody"):
        for command in (
            f"git worktree remove --force ~/local-operator/.worktrees/{path}",
            f"rm -rf ~/local-operator/.worktrees/{path}",
        ):
            risk, reason = _rank(home, command)
            assert risk is Risk.HIGH, (command, reason)
            assert "own-created detached" not in reason


def test_2c_primary_checkout_stays_refused(home: Path, repo: Path) -> None:
    _attr3880(home, repo)
    (repo / "a.py").write_text("a = 'primary work'\n")
    risk, reason = _rank(home, "rm -rf ~/local-operator")
    assert risk >= Risk.HIGH, reason


def test_2d_detached_with_dangling_commit_stays_high(home: Path, repo: Path) -> None:
    wt = _attr3880(home, repo)
    _git("commit", "-qam", "made only here", cwd=wt)  # reachable from NO ref
    (wt / "b.py").write_text("b = 'still dirty'\n")
    assert prov.detached_head(str(wt)) is not None
    assert prov.detached_head(str(wt))[1] is None  # type: ignore[index]
    for command in (FIELD_COMMANDS[0], "rm -rf ~/local-operator/.worktrees/attr3880-ci"):
        risk, reason = _rank(home, command)
        assert risk is Risk.HIGH, (command, reason)


def test_2d_nested_repository_in_detached_tree_stays_high(home: Path, repo: Path) -> None:
    wt = _attr3880(home, repo)
    nested = wt / "vendor"
    nested.mkdir()
    _git("init", "-q", cwd=nested)
    assert prov.own_detached_scratch(str(wt)) is None


@pytest.mark.parametrize("command", ["rm -rf ~", "rm -rf ~/.local-operator", "rm -rf /"])
def test_2e_critical_items_unchanged(home: Path, repo: Path, command: str) -> None:
    _attr3880(home, repo)
    risk, _ = _rank(home, command)
    assert risk is Risk.CRITICAL


def test_refusal_names_the_detached_route() -> None:
    """Item 4: a refusal (ledger missing) names the supported path plainly."""
    assert "`git worktree add --detach`" in SUPPORTED_DELETE_PATHS
    assert "`git worktree remove --force <exact path>`" in SUPPORTED_DELETE_PATHS
    assert "no creation record" in SUPPORTED_DELETE_PATHS


@pytest.mark.asyncio
async def test_judge_facts_say_detached_when_the_ledger_row_is_missing(
    home: Path, repo: Path
) -> None:
    """Item 4: with no ledger row the judge still reads "detached, no branch"."""
    _git("worktree", "add", "-q", "--detach", ".worktrees/nobody", cwd=repo)
    (repo / ".worktrees" / "nobody" / "a.py").write_text("a = 'x'\n")
    prompts: list[str] = []

    async def judge(system: str, prompt: str) -> str:
        prompts.append(prompt)
        return "BLOCK: not shown to be yours"

    ctx = CallContext(
        cwd=str(home), env=_env(home), scratchpad=None, session_id=S, intent="clean up"
    )
    outcome = await check_tool_call(
        "bash",
        {"command": "git worktree remove --force ~/local-operator/.worktrees/nobody"},
        ctx,
        judge=None,
        settings=GuardSettings(),
        forked_judge=judge,
    )
    assert outcome is not None and outcome.blocked
    assert prompts and "HEAD: detached (no branch) at" in prompts[0]
    assert "own-created detached worktree" not in prompts[0]  # no ledger row, no claim
    assert "`git worktree remove --force <exact path>`" in outcome.text


# -- PR #3987 round-1 remediation ----------------------------------------------


def test_refusal_guidance_is_reason_specific() -> None:
    """Review m2: a fixable refusal must not steer the agent to ask about its own tree."""
    text = SUPPORTED_DELETE_PATHS
    assert "treat it as a path you did not create" not in text
    assert "HEAD reachable from no ref → keep the commit with `git tag` or `git branch`" in text
    assert "held by a foreign pid → wait for that process to exit" in text
    assert "only 'not in the ledger' / no creation record means a path you did not create" in text


@pytest.mark.parametrize("name", ["café", "a\tb", 'q"uote'])
def test_nested_repo_under_a_quoted_name_is_not_scratch(home: Path, repo: Path, name: str) -> None:
    """Review m3: git C-quotes these names without ``-z``; the nested-repo check still sees them."""
    wt = _attr3880(home, repo)
    nested = wt / name
    nested.mkdir()
    _git("init", "-q", cwd=nested)
    (nested / "only.txt").write_text("only copy\n")
    _git("add", ".", cwd=nested)
    _git("commit", "-qm", "only copy", cwd=nested)
    plain = _git("status", "--porcelain", cwd=wt)
    assert '"' in plain  # the shape the old parser mis-read
    assert prov.own_detached_scratch(str(wt)) is None


def test_quoted_untracked_file_alone_is_still_scratch(home: Path, repo: Path) -> None:
    """The ``-z`` parse keeps the waiver for a plain quoted-name file (no nested repo)."""
    wt = _attr3880(home, repo)
    (wt / "café.tmp").write_text("scratch\n")
    assert prov.own_detached_scratch(str(wt)) is not None
    assert _rank(home, FIELD_COMMANDS[0])[0] is Risk.LOW


def test_own_detached_tree_outside_worktrees_keeps_base_rank(home: Path, repo: Path) -> None:
    """QA Q2: the waiver holds only under ``<repo>/.worktrees/``."""
    _run_recorded(home, "cd ~/local-operator && git worktree add --detach ~/outside-wt")
    wt = home / "outside-wt"
    (wt / "a.py").write_text("a = 'scratch'\n")
    assert (
        prov.own_detached_scratch(str(wt)) is not None
    )  # detached, HEAD kept: only location fails
    for command in (
        "git -C ~/local-operator worktree remove --force ~/outside-wt",
        "rm -rf ~/outside-wt",
    ):
        risk, reason = _rank(home, command)
        assert risk is Risk.HIGH, (command, reason)
        assert "own-created detached worktree" not in reason


def test_own_detached_tree_in_a_scratchpad_is_unchanged(home: Path, repo: Path) -> None:
    """QA Q2: a scratchpad tree never reaches the waiver (state-root bar), as on base."""
    pad = f"~/.local-operator/sessions/{S}/scratchpad/wt"
    _run_recorded(home, f"cd ~/local-operator && git worktree add --detach {pad}")
    wt = home / ".local-operator" / "sessions" / S / "scratchpad" / "wt"
    (wt / "a.py").write_text("a = 'scratch'\n")
    assert prov.recorded_creators(str(wt)) == [S]  # the ledger has the row ...
    for command in (f"git -C ~/local-operator worktree remove --force {pad}", f"rm -rf {pad}"):
        risk, reason = _rank(home, command)
        assert risk is Risk.HIGH, (command, reason)  # ... but the state-root bar holds
        assert "own-created detached worktree" not in reason


def test_rm_rf_low_reason_names_detached_head(home: Path, repo: Path) -> None:
    """Review n2: the ``rm -rf`` LOW reason carries the detached/kept-by fact."""
    _attr3880(home, repo)
    risk, reason = _rank(home, "rm -rf ~/local-operator/.worktrees/attr3880-ci")
    assert risk is Risk.LOW, reason
    assert "own-created detached worktree (no branch; HEAD " in reason
    assert "is kept by main" in reason or "is kept by feature" in reason


def test_glob_of_only_own_detached_trees_is_low(home: Path, repo: Path) -> None:
    """Review n1 (accepted): a glob reaching only own scratch trees ranks like ledger LOW."""
    _attr3880(home, repo)
    risk, reason = _rank(home, "rm -rf ~/local-operator/.worktrees/attr*")
    assert risk is Risk.LOW, reason


# -- rebase onto #3835 (foreign-ledger rows): both rules hold together --------


@pytest.mark.parametrize("command", FIELD_COMMANDS)
def test_own_scratch_tree_holding_another_sessions_dir_is_high(
    home: Path, repo: Path, command: str
) -> None:
    """Stricter wins: #3835 ranks a target holding foreign live work HIGH.

    The scratch waiver must not lower it: another session's directory inside
    this session's own detached tree would be deleted with it. What keeps it
    HIGH is ``owned_record``'s ``_foreign_inside`` bar (any other owner's row
    at or under the target, live or not, related or not), which is stricter
    than ``ForeignCreations``' "holds" rule, so the waiver is never reached.
    """
    wt = _attr3880(home, repo)
    _run_recorded(home, f"mkdir {wt}/theirs", sid=S2)
    assert prov.owned_record(str(wt), S) is None
    risk, reason = _rank(home, command)
    assert risk is Risk.HIGH, (command, reason)
    assert "own-created detached worktree" not in reason, reason


def _bump_worktree_dir_stamp(wt: Path) -> None:
    """Simulate Linux: the worktree DIR's ctime stamp moved; its ``.git`` row did not."""
    import sqlite3

    conn = sqlite3.connect(prov.ledger_path())
    try:
        conn.execute(
            "UPDATE created SET stamp = stamp + 1 WHERE path = ? AND kind = 'worktree'",
            (os.path.realpath(wt),),
        )
        conn.commit()
    finally:
        conn.close()


def test_foreign_match_keeps_own_worktree_nearest_through_git_file(home: Path, repo: Path) -> None:
    """Nearest owner (#3835) uses the same ``.git`` witness as ``owned_record``.

    Another session made the pool directory; this session's detached tree
    sits inside it. Once the tree's dir stamp moves (Linux ctime), its own row
    must still be the nearest live owner, or the broader foreign pool row
    makes this session's own file read as another session's work.
    """
    _run_recorded(home, "cd ~/local-operator && mkdir .worktrees/pool", sid=S2)
    wt = _attr3880(home, repo, name="pool/wt")
    _bump_worktree_dir_stamp(wt)
    assert prov.owned_record(str(wt), S) is not None
    foreign = prov.ForeignCreations(S)
    try:
        assert foreign.match([str(wt / "a.py")]) is None
        hit = foreign.match([str(repo / ".worktrees" / "pool")])
        assert hit is not None and hit.owner == S2  # the pool itself stays foreign
    finally:
        foreign.close()


def test_foreign_match_without_git_file_witness_falls_to_the_foreign_pool(
    home: Path, repo: Path
) -> None:
    """Negative control: a new ``.git`` inode generation is no witness."""
    _run_recorded(home, "cd ~/local-operator && mkdir .worktrees/pool", sid=S2)
    wt = _attr3880(home, repo, name="pool/wt")
    _bump_worktree_dir_stamp(wt)
    gitfile = wt / ".git"
    body = gitfile.read_text()
    gitfile.unlink()
    gitfile.write_text(body)
    foreign = prov.ForeignCreations(S)
    try:
        hit = foreign.match([str(wt / "a.py")])
        assert hit is not None and hit.owner == S2 and hit.relation == "inside"
    finally:
        foreign.close()


def _stale_worktree_row(wt: Path, *, owner: str) -> None:
    """Simulate inode reuse: ``owner``'s dead worktree row at ``wt``.

    Ledger rows are never pruned and ext4/xfs hand a freed inode number to the
    next directory, so a removed worktree's row can carry the live dev+ino of
    another session's new worktree at the same path, under an older stamp.
    APFS never reuses inode numbers, so the row is written directly.
    """
    import sqlite3

    real = os.path.realpath(wt)
    st = os.stat(real)
    conn = sqlite3.connect(prov.ledger_path())
    try:
        conn.execute(
            "INSERT OR REPLACE INTO created VALUES (?,?,?,?,?,?,?,?,?)",
            (
                owner,
                real,
                st.st_dev,
                st.st_ino,
                1.0,
                owner,
                "worktree",
                "git worktree add (removed long ago)",
                prov.creation_stamp(st) + 1,
            ),
        )
        conn.commit()
    finally:
        conn.close()


@pytest.mark.parametrize("rel", ["/runs", "/runs/r.json"])
def test_foreign_worktree_git_file_never_revives_another_owners_stale_row(
    home: Path, repo: Path, rel: str
) -> None:
    """R-C1: a ``.git`` companion vouches only for its own owner's row.

    S2's live worktree reuses the inode of a worktree S removed. S's dead row
    must not borrow S2's ``.git`` row as its generation witness: that put S
    among the tree's owners and ranked ``rm -rf <S2's wt>/runs`` LOW.
    """
    wt = _attr3880(home, repo, name="theirs", sid=S2)
    (wt / "runs").mkdir()
    (wt / "runs" / "r.json").write_text("{}")
    _stale_worktree_row(wt, owner=S)
    foreign = prov.ForeignCreations(S)
    try:
        hit = foreign.match([str(wt) + rel])
        assert hit is not None and hit.owner == S2 and hit.relation == "inside"
    finally:
        foreign.close()
    risk, reason = _rank(home, f"rm -rf ~/local-operator/.worktrees/theirs{rel}")
    assert risk is Risk.HIGH, reason
    assert "created by another session" in reason, reason
