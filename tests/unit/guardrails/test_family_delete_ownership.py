"""One subagent tree shares delete ownership of what it created (2026-11-14).

Field case, relayed by the operator ("end own-worktree false denials"):
workstream ``2183114d32b1`` delegated two GitLab MRs to subagents. Subagent
``c40435c7fc86`` created a ``feat/tfc-acm-sbx-auth`` worktree in each repo with

    cd ~/minervaai/infrastructure; for r in services terraform-cloud-bootstrap
    aws/acm; do (cd $r; git worktree add -q -b feat/tfc-acm-sbx-auth
    .worktrees/tfc-acm-sbx-auth origin/main; …); done

and, after both MRs merged, its SIBLING cleanup subagent ``bbb5e6c44eb8`` ran
``for r in terraform-cloud-bootstrap services; do git -C …/$r worktree remove
…/$r/.worktrees/tfc-acm-sbx-auth; …; done``. The guard refused: "no ledger row
shows this session created either worktree … linked worktree state: dirty".
Three causes, each pinned below:

1. The ledger had NO row for either tree: in ``do (cd $r; …)`` the ``do``
   hid the subshell's ``(`` from the resolver, the ``cd`` was dropped, and the
   creation resolved to ``~/minervaai/infrastructure/.worktrees/…`` (never
   made, so nothing was recorded); and only the first loop value could ever
   have been listed. Fixed in ``resolve`` (``_peel_subshell_keyword``,
   ``_cwd_fan_for``).
2. Ownership was the caller's own rows plus its descendants'. Now an
   ancestor's creation is the caller's too, and a sibling's or cousin's counts
   for a non-forced ``git worktree remove`` of the exact tree only, when clean,
   no PR/MR is open, and neither the creator nor anyone else is running in it.
3. The only extras were IGNORED ``.terraform`` dirs, which made a non-own tree
   ``dirty``; git itself removes ignored files without ``--force``.

Everything runs in a synthetic HOME with real ``git init``/``git worktree
add``, ledger rows written by the provenance API and lineage by the real
``resume.mark_session_origin``. The real field trees are never touched.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.guardrails import provenance as prov
from local_operator.guardrails import resolve as resolve_module
from local_operator.guardrails.check import CallContext, GuardSettings, check_tool_call
from local_operator.guardrails.classify import Risk, classify_bash
from local_operator.guardrails.resolve import resolve_command
from local_operator.resume import mark_session_origin

P = "2183114d32b1"  # workstream (tree root)
K = "c40435c7fc86"  # subagent that created the worktrees
C = "bbb5e6c44eb8"  # sibling cleanup subagent
G = "aaaa0000000c"  # C's own subagent (grandchild of P)
Q = "aaaa0000000d"  # K's subagent: a cousin of G
U = "999999999999"  # unrelated root session
REPOS = ("terraform-cloud-bootstrap", "services")


class _Judge:
    def __init__(self, answer: str = "BLOCK: should not be asked") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    async def __call__(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    state = home / ".local-operator"
    (state / "sessions").mkdir(parents=True)
    (state / "config.yml").write_text("version: 1\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))
    for key in [k for k in os.environ if k.startswith(("CMUX_", "GIT_", "LOP_"))]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    # Synthetic forge/process facts: no live cwd inside any tree, a GitLab
    # remote whose MRs are merged (as in the field), nobody running unless a
    # test says so through ``running``.
    monkeypatch.setattr(prov, "_live_cwd_inside", lambda _: False)
    monkeypatch.setattr(prov, "_cwd_inventory_cache", None)
    monkeypatch.setattr(prov, "_github_repo", lambda _: None)
    monkeypatch.setattr(prov, "_gitlab_repo", lambda _: "minerva/infrastructure")
    monkeypatch.setattr(
        prov, "_glab_mrs", lambda repo, branch: [prov.ForgeChange(102, "merged", "2026-11-14")]
    )
    monkeypatch.setattr(prov, "_glab_open_mrs", lambda repo, branch: [])
    # Real lineage, written the way every agent spawn path writes it.
    sessions = state / "sessions"
    for sid in (P, U):
        (sessions / sid / "scratchpad").mkdir(parents=True)
    for child, parent in ((K, P), (C, P), (G, C), (Q, K)):
        mark_session_origin(sessions / child, "subagent", parent=parent)
        (sessions / child / "scratchpad").mkdir(exist_ok=True)
    _set_running(monkeypatch, {})
    return home


def _set_running(monkeypatch: pytest.MonkeyPatch, running: dict[str, str]) -> None:
    """Make ``registry.scan`` report ``{session id: cwd}`` as live sessions."""
    from local_operator.session.runtime import registry

    records = [(SimpleNamespace(session_id=s, cwd=cwd), "live") for s, cwd in running.items()]
    monkeypatch.setattr(registry, "scan", lambda *a, **k: list(records))


def _path_key(path: str | None) -> str:
    # ``Target.path`` is Optional, so a bare ``sorted()`` does not type-check.
    # Sort by key rather than filtering: a ``None`` stays in the list and still
    # fails the equality against the all-``str`` expectation, as before.
    return path or ""


def _env(home: Path) -> dict[str, str]:
    return {"HOME": str(home), "LOCAL_OPERATOR_CONFIG_DIR": str(home / ".local-operator")}


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@e", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _ancestors(sid: str) -> list[str]:
    return prov.subagent_ancestors(sid)


@pytest.fixture
def infra(home: Path) -> Path:
    """``~/minervaai/infrastructure/{terraform-cloud-bootstrap,services}`` repos."""
    root = home / "minervaai" / "infrastructure"
    for name in REPOS:
        repo = root / name
        repo.mkdir(parents=True)
        _git("init", "-q", "-b", "main", cwd=repo)
        (repo / ".gitignore").write_text(".terraform/\n.env\nnode_modules/\n.worktrees/\n")
        (repo / "main.tf").write_text("# tf\n")
        _git("add", ".", cwd=repo)
        _git("commit", "-qm", "init", cwd=repo)
        _git("branch", "origin-main", cwd=repo)
    return root


def _add_tree(home: Path, repo: Path, creator: str, name: str = "tfc-acm-sbx-auth") -> Path:
    """``git worktree add`` by ``creator``, recorded through the ledger planner."""
    wt = repo / ".worktrees" / name
    command = f"cd {repo} && git worktree add -q -b feat/{name} .worktrees/{name} origin-main"
    _run_recorded(home, command, creator)
    assert wt.is_dir()
    return wt


def _run_recorded(home: Path, command: str, creator: str) -> list[str]:
    """Run ``command`` in bash between ``plan_creations``/``commit_creations``,
    exactly as the loop does around a bash call."""
    env = _env(home)
    pending = prov.plan_creations(command, cwd=str(home), env=env)
    subprocess.run(
        ["bash", "-c", command],
        cwd=home,
        check=True,
        capture_output=True,
        env={**os.environ, **env},
    )
    assert pending is not None
    return prov.commit_creations(
        pending,
        session_id=creator,
        ancestors=_ancestors(creator),
        env=env,
        scratchpad=None,
    )


def _ignore_only(wt: Path) -> None:
    """The field trees' only extras: ignored ``.terraform`` (+ an ignored ``.env``)."""
    (wt / ".terraform" / "providers").mkdir(parents=True)
    (wt / ".terraform" / "providers" / "lock").write_text("x")
    (wt / ".env").write_text("LOCAL_ONLY=1\n")
    _restamp(wt)


def _restamp(wt: Path) -> None:
    """Refresh the tree's ledger rows after adding an entry at its ROOT.

    On Linux ``creation_stamp`` is the directory's ctime, which a new root
    entry moves, so the row would stop matching (deliberately fail-closed;
    see ``test_own_worktree_removal._record``). macOS stamps birth time.
    """
    import sqlite3

    real = os.path.realpath(wt)
    st = os.stat(real)
    conn = sqlite3.connect(str(prov.ledger_path()))
    with conn:
        conn.execute(
            "UPDATE created SET dev=?, ino=?, stamp=? WHERE path=?",
            (st.st_dev, st.st_ino, prov.creation_stamp(st), real),
        )
    conn.close()


def _rank(home: Path, command: str, sid: str) -> Any:
    return classify_bash(command, cwd=str(home), env=_env(home), scratchpad=None, session_id=sid)


async def _check(home: Path, command: str, sid: str, judge: _Judge) -> Any:
    ctx = CallContext(
        cwd=str(home),
        env=_env(home),
        scratchpad=None,
        session_id=sid,
        intent="removing finished worktrees",
        operator=prov.operator_brief([], session_id=sid, cited=(command,)),
        messages=[],
    )
    return await check_tool_call(
        "bash", {"command": command}, ctx, judge=None, settings=GuardSettings(), forked_judge=judge
    )


def _remove(wt: Path, *, force: bool = False) -> str:
    repo = wt.parent.parent
    return f"git -C {repo} worktree remove {'--force ' if force else ''}{wt}"


# -- cause 1: the loop-in-subshell creation is recorded, per loop value ---------


FIELD_CREATE = (
    "cd ~/minervaai/infrastructure; set -e; for r in services terraform-cloud-bootstrap; "
    "do (cd $r; test ! -e .worktrees/tfc-acm-sbx-auth; git worktree add -q -b "
    "feat/tfc-acm-sbx-auth .worktrees/tfc-acm-sbx-auth origin-main; "
    "git -C .worktrees/tfc-acm-sbx-auth log --oneline -1); done"
)
FIELD_REMOVE = (
    "for r in terraform-cloud-bootstrap services; do "
    "git -C ~/minervaai/infrastructure/$r worktree remove "
    "~/minervaai/infrastructure/$r/.worktrees/tfc-acm-sbx-auth; echo rc=$?; "
    "test -e ~/minervaai/infrastructure/$r/.worktrees/tfc-acm-sbx-auth && echo STILL || "
    "echo absent; git -C ~/minervaai/infrastructure/$r worktree list; "
    "git -C ~/minervaai/infrastructure/$r status --short | head; "
    "git -C ~/minervaai/infrastructure/$r branch --list feat/tfc-acm-sbx-auth; done"
)


def test_field_loop_creation_lists_every_repository(home: Path, infra: Path) -> None:
    created = resolve_command(FIELD_CREATE, cwd=str(home), env=_env(home), home=str(home)).created
    assert sorted((c.path for c in created), key=_path_key) == sorted(
        str(infra / r / ".worktrees" / "tfc-acm-sbx-auth") for r in REPOS
    )


def test_field_loop_creation_is_recorded_for_both_trees(home: Path, infra: Path) -> None:
    written = _run_recorded(home, FIELD_CREATE, K)
    for repo in REPOS:
        wt = os.path.realpath(infra / repo / ".worktrees" / "tfc-acm-sbx-auth")
        assert wt in written
        rec = prov.owned_record(wt, K)
        assert rec is not None and rec.kind == "worktree" and rec.relation == ""


@pytest.mark.parametrize(
    "command",
    [
        # A cd in the body proper carries into the next iteration: bash runs
        # ``cd terraform-cloud-bootstrap`` from inside ``services``.
        "cd ~/minervaai/infrastructure; for r in services terraform-cloud-bootstrap; "
        "do cd $r; mkdir made; done",
        # A cd after the subshell moves where the next iteration starts.
        "cd ~/minervaai/infrastructure; for r in services terraform-cloud-bootstrap; "
        "do (cd $r; mkdir made); cd services; done",
        # A loop value that is not an existing directory: nothing guessed.
        "cd ~/minervaai/infrastructure; for r in services nowhere; do (cd $r; mkdir made); done",
    ],
    ids=["cd-outside-subshell", "cd-after-subshell", "missing-dir"],
)
def test_unprovable_loop_cwds_record_only_the_first_iteration(
    home: Path, infra: Path, command: str
) -> None:
    """Never record a guess: a LATER iteration's cwd after such a ``cd`` is
    unknown, so its ``made`` is not listed. Iteration 1 runs in ``services``
    for certain (review R1-M1 keeps that cwd), so its creation is listed;
    ``commit_creations`` still records it only if it is a new directory."""
    created = resolve_command(command, cwd=str(home), env=_env(home), home=str(home)).created
    assert [c.path for c in created] == [str(infra / "services" / "made")]


@pytest.mark.parametrize(
    "command",
    [
        "cd /tmp; for r in /tmp ~; do (cd $r; rm -rf .local-operator); done",
        "cd /tmp; for r in /tmp ~; do cd $r; rm -rf .local-operator; done",
        "cd ~/minervaai; for r in infrastructure ..; do (cd $r; rm -rf .local-operator); done",
        "cd ~; for r in minervaai .; do (cd $r; rm -rf .local-operator); done",
    ],
    ids=["subshell-abs", "plain-abs", "subshell-dotdot", "subshell-dot"],
)
def test_relative_delete_is_ranked_at_every_loop_cwd(home: Path, infra: Path, command: str) -> None:
    """A later iteration's cwd is ranked too: here it lands on the state root.

    Before: the walk saw the body once, at the first value, so the second
    iteration's ``~/.local-operator`` went unranked (LOW); and with ``do (cd``
    the ``cd`` itself was dropped, resolving against the outer cwd.
    """
    assert _rank(home, command, C).risk is Risk.CRITICAL


def test_unprovable_loop_cwd_makes_a_relative_delete_unresolved(home: Path, infra: Path) -> None:
    command = (
        "cd ~/minervaai/infrastructure; for r in services terraform-cloud-bootstrap; "
        "do cd $r; rm -rf build; done"
    )
    targets = resolve_command(command, cwd=str(home), env=_env(home), home=str(home)).targets
    # Iteration 1's cwd is certain; the later one is not (review R1-M1).
    assert [t.path for t in targets] == [str(infra / "services" / "build"), None]


def test_absolute_loop_cwds_fan_without_a_subshell(home: Path, infra: Path) -> None:
    command = (
        f"for r in {infra}/services {infra}/terraform-cloud-bootstrap; do cd $r; mkdir m; done"
    )
    created = resolve_command(command, cwd=str(home), env=_env(home), home=str(home)).created
    assert sorted((c.path for c in created), key=_path_key) == sorted(
        str(infra / r / "m") for r in REPOS
    )


def test_field_removal_loop_resolves_both_literal_paths(home: Path, infra: Path) -> None:
    targets = resolve_command(FIELD_REMOVE, cwd=str(home), env=_env(home), home=str(home)).targets
    assert sorted((t.path for t in targets if t.op == "delete"), key=_path_key) == sorted(
        str(infra / r / ".worktrees" / "tfc-acm-sbx-auth") for r in REPOS
    )


@pytest.mark.asyncio
async def test_field_case_sibling_cleanup_settles_without_the_judge(
    home: Path, infra: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole field case: K creates by loop, C (K's sibling) removes by loop.

    The parent is live (it is waiting on C) and must not count as an owner.
    """
    _run_recorded(home, FIELD_CREATE, K)
    for repo in REPOS:
        _ignore_only(infra / repo / ".worktrees" / "tfc-acm-sbx-auth")
    _set_running(monkeypatch, {P: str(home), C: str(home)})
    verdict = _rank(home, FIELD_REMOVE, C)
    assert verdict.risk is Risk.LOW, [t.reason for t in verdict.targets]
    judge = _Judge()
    outcome = await _check(home, FIELD_REMOVE, C, judge)
    assert outcome is None or not outcome.blocked, outcome and outcome.text
    assert judge.prompts == []


# -- cause 2: ancestors and descendants share ownership ---------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("creator", "remover"),
    [(P, C), (P, G), (C, P), (G, P)],
    ids=[
        "child-removes-parent",
        "grandchild-removes-root",
        "parent-removes-child",
        "root-removes-grandchild",
    ],
)
async def test_family_removes_clean_tree_with_zero_judge_calls(
    home: Path, infra: Path, monkeypatch: pytest.MonkeyPatch, creator: str, remover: str
) -> None:
    wt = _add_tree(home, infra / "services", creator)
    # The creator (an ancestor here, or the remover's own child) is live.
    _set_running(monkeypatch, {P: str(home)} if creator == P else {})
    verdict = _rank(home, _remove(wt), remover)
    assert verdict.risk is Risk.LOW, verdict.targets[0].reason
    judge = _Judge()
    outcome = await _check(home, _remove(wt), remover, judge)
    assert outcome is None or not outcome.blocked, outcome and outcome.text
    assert judge.prompts == []


def test_child_removes_parents_tree_holding_only_ignored_files(home: Path, infra: Path) -> None:
    wt = _add_tree(home, infra / "services", P)
    _ignore_only(wt)
    assert prov.worktree_state(str(wt)) == "dirty"  # the field's foreign reading
    verdict = _rank(home, _remove(wt), C)
    assert verdict.risk is Risk.LOW, verdict.targets[0].reason
    assert f"ancestor session {P}" in verdict.targets[0].reason


def test_child_removes_parents_tree_with_tracked_change_is_high(home: Path, infra: Path) -> None:
    wt = _add_tree(home, infra / "services", P)
    (wt / "main.tf").write_text("# edited\n")
    assert _rank(home, _remove(wt), C).risk is Risk.HIGH


def test_child_removes_parents_tree_with_untracked_file_is_high(home: Path, infra: Path) -> None:
    wt = _add_tree(home, infra / "services", P)
    (wt / "notes.md").write_text("wip\n")
    _restamp(wt)
    assert _rank(home, _remove(wt), C).risk is Risk.HIGH


def test_child_removes_parents_tree_with_open_mr_is_critical(
    home: Path, infra: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wt = _add_tree(home, infra / "services", P)
    monkeypatch.setattr(prov, "_glab_mrs", lambda repo, branch: [prov.ForgeChange(103, "open", "")])
    verdict = _rank(home, _remove(wt), C)
    assert verdict.risk is Risk.CRITICAL, verdict.targets[0].reason


def test_rm_rf_of_parents_tree_with_ignored_env_is_judged(home: Path, infra: Path) -> None:
    """Decision: ``rm -rf`` keeps today's rule for ignored files the caller did
    not make — an ancestor's ignored ``.env`` is not known to be reproducible —
    so it is judged; ``git worktree remove`` of the same tree (git's own rule)
    is LOW, and the same ``rm -rf`` by the creator itself stays LOW."""
    wt = _add_tree(home, infra / "services", P)
    _ignore_only(wt)
    assert _rank(home, f"rm -rf {wt}", C).risk is Risk.HIGH
    assert _rank(home, _remove(wt), C).risk is Risk.LOW
    assert _rank(home, f"rm -rf {wt}", P).risk is Risk.LOW


def test_rm_rf_of_parents_tree_with_regenerable_ignored_output_is_low(
    home: Path, infra: Path
) -> None:
    wt = _add_tree(home, infra / "services", P)
    (wt / "node_modules" / "x").mkdir(parents=True)
    _restamp(wt)
    verdict = _rank(home, f"rm -rf {wt}", C)
    assert verdict.risk is Risk.LOW, verdict.targets[0].reason


def test_unrelated_session_stays_foreign(home: Path, infra: Path) -> None:
    wt = _add_tree(home, infra / "services", P)
    _ignore_only(wt)
    assert prov.owned_record(str(wt), U) is None
    assert prov.owned_record(str(wt), U, kin=True) is None
    assert _rank(home, _remove(wt), U).risk is Risk.HIGH
    assert _rank(home, f"rm -rf {wt}", U).risk is Risk.HIGH


def test_lineage_never_comes_from_the_environment(
    home: Path, infra: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session claiming a parent through env or text gains nothing."""
    wt = _add_tree(home, infra / "services", P)
    _ignore_only(wt)
    monkeypatch.setenv("LOCAL_OPERATOR_PARENT_SESSION_ID", P)
    monkeypatch.setenv("LOP_PARENT_SESSION", P)
    assert prov.owned_record(str(wt), U) is None
    assert _rank(home, _remove(wt), U).risk is Risk.HIGH


# -- siblings and cousins: non-forced ``git worktree remove`` only ---------------


@pytest.mark.parametrize("remover", [C, Q], ids=["sibling", "cousin-of-creator's-child"])
def test_kin_clean_non_forced_remove_is_low(home: Path, infra: Path, remover: str) -> None:
    creator = K if remover == C else G  # Q (under K) vs G (under C): cousins
    wt = _add_tree(home, infra / "services", creator)
    _ignore_only(wt)
    verdict = _rank(home, _remove(wt), remover)
    assert verdict.risk is Risk.LOW, verdict.targets[0].reason
    assert f"sibling session {creator}" in verdict.targets[0].reason


def test_kin_rm_rf_is_high(home: Path, infra: Path) -> None:
    wt = _add_tree(home, infra / "services", K)
    assert prov.owned_record(str(wt), C) is None  # default: kin is not ownership
    assert _rank(home, f"rm -rf {wt}", C).risk is Risk.HIGH


def test_kin_forced_remove_is_high(home: Path, infra: Path) -> None:
    wt = _add_tree(home, infra / "services", K)
    assert _rank(home, _remove(wt, force=True), C).risk is Risk.HIGH


def test_kin_remove_while_creator_runs_is_high(
    home: Path, infra: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wt = _add_tree(home, infra / "services", K)
    _set_running(monkeypatch, {K: str(home), P: str(home)})
    verdict = _rank(home, _remove(wt), C)
    assert verdict.risk is Risk.HIGH
    assert K in verdict.targets[0].reason


def test_kin_dirty_tree_is_high_and_names_the_creator(home: Path, infra: Path) -> None:
    wt = _add_tree(home, infra / "services", K)
    (wt / "notes.md").write_text("wip\n")
    _restamp(wt)
    verdict = _rank(home, _remove(wt), C)
    assert verdict.risk is Risk.HIGH
    assert K in verdict.targets[0].reason


def test_kin_open_mr_is_critical(home: Path, infra: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wt = _add_tree(home, infra / "services", K)
    monkeypatch.setattr(prov, "_glab_mrs", lambda repo, branch: [prov.ForgeChange(136, "open", "")])
    assert _rank(home, _remove(wt), C).risk is Risk.CRITICAL


def test_kin_unknown_mr_state_is_not_waived(
    home: Path, infra: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#3816 waives an unknown PR state for the creator's OWN clean tree only."""
    wt = _add_tree(home, infra / "services", K)
    monkeypatch.setattr(prov, "_glab_mrs", lambda repo, branch: None)
    assert _rank(home, _remove(wt), C).risk is Risk.HIGH
    # The same unknown state does not hold the creator itself (#3816).
    assert _rank(home, _remove(wt), K).risk is Risk.LOW


def test_kin_container_row_vouches_for_nothing(home: Path, infra: Path) -> None:
    """A sibling's ``mkdir`` of a container is not its ``git worktree add``."""
    repo = infra / "services"
    _run_recorded(home, f"mkdir -p {repo}/.worktrees/box", K)
    tree = repo / ".worktrees" / "box" / "t"
    _git("worktree", "add", "-q", "-b", "feat/box", str(tree), "origin-main", cwd=repo)
    # Ignored-only, so the foreign reading is ``dirty`` and only a (wrong)
    # kin credit could make it LOW.
    _ignore_only(tree)
    verdict = _rank(home, _remove(tree), C)
    assert verdict.risk is Risk.HIGH, verdict.targets[0].reason
    assert "sibling session" not in verdict.targets[0].reason


# -- round 1 remediation (PR #3964) ---------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "for r in ~ /nowhere; do cd $r; rm -rf .local-operator; done",
        "cd ~; for r in . a; do cd $r; rm -rf .local-operator; done",
        "cd ~; for r in . nowhere; do (cd $r; rm -rf .local-operator); done",
        "cd ~; for r in /tmp ~/a; do cd $r; cd -; rm -rf .local-operator; done",
        "cd ~; for r in . a; do (cd -P $r; rm -rf .local-operator); done",
        "cd ~; for r in . a; do cd $r; rm -f .local-operator/config.yml; done",
    ],
)
def test_unnameable_later_iteration_keeps_the_first_cwd(home: Path, command: str) -> None:
    """Review R1-M1: bash runs iteration 1 at the first value before any
    unknown iteration, so its ``~/.local-operator`` delete is certain. Before
    the fix the whole body's cwd was forgotten and these ranked a judge-able
    HIGH; base 3bb7023 refused them outright."""
    verdict = _rank(home, command, U)
    assert verdict.risk is Risk.CRITICAL, [t.reason for t in verdict.targets]
    targets = resolve_command(command, cwd=str(home), env=_env(home), home=str(home)).targets
    assert None in [t.path for t in targets]  # the later iterations stay unresolved


def test_relative_value_after_an_absolute_one_is_not_read_from_the_header(home: Path) -> None:
    """Review R1-m1: ``cd ..`` in iteration 2 runs from ``~/a`` (iteration 1's
    cwd), so bash deletes ``~/.local-operator``; the walk once resolved it from
    the header's ``/tmp`` (``/.local-operator``, LOW). That previous-iteration
    cwd is not modelled, so it must be unresolved, never the header's."""
    (home / "a").mkdir()
    command = "cd /tmp; for r in ~/a ..; do cd $r; rm -rf .local-operator; done"
    targets = resolve_command(command, cwd=str(home), env=_env(home), home=str(home)).targets
    paths = [t.path for t in targets]
    assert str(home / "a" / ".local-operator") in paths
    assert None in paths
    assert "/.local-operator" not in paths
    assert _rank(home, command, U).risk >= Risk.HIGH


@pytest.mark.parametrize("flag", ["--f", "--fo", "--for", "--forc", "--force"])
def test_abbreviated_force_is_forced(home: Path, infra: Path, flag: str) -> None:
    """Review R1-m2: git accepts any unique prefix of ``--force`` (measured,
    git 2.55: ``git worktree remove --f`` removed an untracked-dirty tree)."""
    wt = _add_tree(home, infra / "services", K)
    command = f"git -C {wt.parent.parent} worktree remove {flag} {wt}"
    targets = resolve_command(command, cwd=str(home), env=_env(home), home=str(home)).targets
    assert [t.forced for t in targets] == [True]
    # A sibling's tree is admitted only for a NON-forced remove.
    assert _rank(home, command, C).risk is Risk.HIGH


def test_direct_remove_of_a_foreign_tree_names_the_recorded_creator(
    home: Path, infra: Path
) -> None:
    """QA Q1: the refusal says whose tree it is."""
    wt = _add_tree(home, infra / "services", K)
    (wt / "notes.md").write_text("wip\n")
    _restamp(wt)
    verdict = _rank(home, _remove(wt), U)
    assert verdict.risk is Risk.HIGH
    assert f"recorded creator: {K}" in verdict.targets[0].reason


def test_grandchild_sees_an_ancestor_not_a_parent(home: Path, infra: Path) -> None:
    """QA Q2: G is P's grandchild, so P is an ancestor, not its parent."""
    wt = _add_tree(home, infra / "services", P)
    _ignore_only(wt)
    verdict = _rank(home, _remove(wt), G)
    assert verdict.risk is Risk.LOW, verdict.targets[0].reason
    assert f"ancestor session {P}" in verdict.targets[0].reason
    assert "parent session" not in verdict.targets[0].reason


# -- round 2 remediation (PR #3964) ---------------------------------------------

#: Review R2-M1: an inner loop's relative ``cd`` of a loop word, after an
#: outer fanned ``cd``, dropped the outer iteration's cwd (``~``). Base
#: 3bb7023 ranked every row CRITICAL; 6ea7729 ranked the first four LOW and
#: the last HIGH. Bash (``pwd`` trace) deletes ``~/a/.local-operator`` then
#: ``~/.local-operator`` in each.
R2_M1_ROWS = [
    "cd ~; for r in a .; do for s in .; do (cd $r; cd $s; rm -rf .local-operator); done; done",
    "cd ~; for s in .; do for r in a .; do (cd $r; cd $s; rm -rf .local-operator); done; done",
    "cd ~; for r in a .; do (cd $r; for s in .; do cd $s; done; rm -rf .local-operator); done",
    "cd ~/a; for r in sub .; do for s in ..; do (cd $r; cd $s; rm -rf .local-operator); done; done",
    "cd ~; for r in a .; do for s in . .; do (cd $r; cd $s; rm -rf .local-operator); done; done",
]


@pytest.mark.parametrize("command", R2_M1_ROWS)
def test_inner_loop_cd_keeps_the_outer_fan(home: Path, command: str) -> None:
    (home / "a" / "sub").mkdir(parents=True)
    verdict = _rank(home, command, U)
    assert verdict.risk is Risk.CRITICAL, [t.reason for t in verdict.targets]
    paths = [
        t.path
        for t in resolve_command(command, cwd=str(home), env=_env(home), home=str(home)).targets
    ]
    assert str(home / ".local-operator") in paths


@pytest.mark.parametrize(
    "command",
    [
        # Controls the reviewer measured CRITICAL on head already: a plain
        # variable or a literal moves the outer fan the same way.
        "cd ~/a; s=..; for r in sub .; do (cd $r; cd $s; rm -rf .local-operator); done",
        "cd ~; for r in a .; do (cd $r; cd .; rm -rf .local-operator); done",
    ],
)
def test_literal_and_plain_variable_controls_stay_critical(home: Path, command: str) -> None:
    (home / "a" / "sub").mkdir(parents=True)
    assert _rank(home, command, U).risk is Risk.CRITICAL


def test_inner_loop_cd_over_ordinary_dirs_moves_the_fan(home: Path) -> None:
    """The outer fan is moved, not turned into an unknown cwd: both real
    iterations' ``build`` dirs are named, with no unresolved target.

    Review R3 (PR #3964) made the loop-cd model ADD-only: the pre-PR model's
    targets are unioned in (``resolve_command``). Base never saw this ``cd``
    (``do (`` hid it) and names ``~/build``, a missing path ranked HIGH, so
    the command ranks HIGH as on base: the accepted cost of the invariant."""
    for d in ("a", "b"):
        (home / d / "build").mkdir(parents=True)
    command = "cd ~; for r in a b; do for s in .; do (cd $r; cd $s; rm -rf build); done; done"
    walked = resolve_module._walk_once(
        command, cwd=str(home), env=_env(home), home=str(home), model=resolve_module._CwdModel()
    )
    paths = sorted((t.path for t in walked.resolution.targets), key=_path_key)
    assert paths == [str(home / "a" / "build"), str(home / "b" / "build")]
    targets = resolve_command(command, cwd=str(home), env=_env(home), home=str(home)).targets
    assert {str(home / "a" / "build"), str(home / "b" / "build")} <= {t.path for t in targets}
    assert _rank(home, command, U).risk is Risk.HIGH


def test_cd_dash_after_a_clearing_cd_in_a_loop_is_unresolved(home: Path) -> None:
    """Review R2-m1: ``cd /tmp`` clears the later-iteration flag, but ``cd -``
    returns to iteration 2's ``~`` (bash deletes ``~/.local-operator``), not
    iteration 1's ``~/a``. Base and 6ea7729 both ranked it LOW."""
    (home / "a").mkdir()
    for command in (
        "cd ~/a; for r in . ..; do cd $r; cd /tmp; cd -; rm -rf .local-operator; done",
        'cd ~/a; for r in . ..; do cd $r; cd /tmp; cd "$OLDPWD"; rm -rf .local-operator; done',
    ):
        targets = resolve_command(command, cwd=str(home), env=_env(home), home=str(home)).targets
        assert None in [t.path for t in targets], command
        assert _rank(home, command, U).risk >= Risk.HIGH, command
