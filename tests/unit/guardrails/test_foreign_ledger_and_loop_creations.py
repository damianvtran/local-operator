"""Two creation-ledger gaps found in the field (2026-11).

Gap A — loop creations. Session 3ef6295a5545 ran
``for n in l1-tool-verbs l2-file-sizes; do git worktree add -b feat/i18n-$n
.worktrees/i18n-$n …; done``; only the FIRST loop value got a ledger row, so
the creator's own later removal of ``.worktrees/i18n-l2-file-sizes`` went to
the judge and was refused. Every loop value is a creation candidate now.

Gap B — another session's directory. ``rm -rf
~/worktrees/osworld/osworld-21/runs`` deleted another session's sealed
benchmark evidence: a deep non-git directory under HOME ranked "ordinary
path" (LOW). A ledger row from a session outside this one's task tree at,
under or above the target now ranks HIGH and names that session.

HOME and the state root are synthetic under ``tmp_path``; session ids are
synthetic; git runs with system/global config disabled (no keychain helper).
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from pathlib import Path

import pytest

from local_operator.guardrails import evidence as ev_mod
from local_operator.guardrails import provenance as prov
from local_operator.guardrails.check import CallContext, GuardSettings, check_tool_call
from local_operator.guardrails.classify import Risk, classify_bash
from local_operator.guardrails.resolve import resolve_command

SID = "aaaaaaaaaaaa"
CHILD = "bbbbbbbbbbbb"
OTHER = "cccccccccccc"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    state = home / ".local-operator"
    for sid in (SID, CHILD, OTHER):
        (state / "sessions" / sid / "scratchpad").mkdir(parents=True)
    (state / "config.yml").write_text("version: 1\n")
    # CHILD is SID's subagent; OTHER is unrelated.
    (state / "sessions" / CHILD / "origin.json").write_text(
        json.dumps({"origin": "subagent", "parent": SID})
    )
    monkeypatch.setattr(prov, "_live_cwd_inside", lambda _: False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))
    for key in [k for k in os.environ if k.startswith(("CMUX_", "GIT_", "LOP_"))]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    ev_mod._TREE_CACHE.clear()
    return home


def _env(home: Path) -> dict[str, str]:
    return {"HOME": str(home), "LOCAL_OPERATOR_CONFIG_DIR": str(home / ".local-operator")}


def _run(home: Path, command: str, sid: str = SID, cwd: Path | None = None) -> list[str]:
    """Run ``command`` for real between the loop's plan/commit hooks."""
    where = str(cwd or home)
    pending = prov.plan_creations(command, cwd=where, env=_env(home))
    subprocess.run(["bash", "-c", command], cwd=where, check=True, env={**os.environ})
    if pending is None:
        return []
    return prov.commit_creations(pending, session_id=sid, env=_env(home), scratchpad=None)


def _verdict(home: Path, command: str, sid: str = SID):
    return classify_bash(command, cwd=str(home), env=_env(home), scratchpad=None, session_id=sid)


# -- Gap A: loop creations ------------------------------------------------------------


def test_resolver_lists_every_loop_value_as_a_creation(tmp_path: Path) -> None:
    res = resolve_command(
        "for n in a b c; do mkdir x-$n; done", cwd=str(tmp_path), env={}, home=str(tmp_path)
    )
    assert [c.path for c in res.created] == [str(tmp_path / f"x-{n}") for n in "abc"]
    clone = resolve_command(
        "for n in a b; do git clone https://example.com/o/$n.git; done",
        cwd=str(tmp_path),
        env={},
        home=str(tmp_path),
    )
    assert [c.path for c in clone.created] == [str(tmp_path / "a"), str(tmp_path / "b")]


@pytest.mark.parametrize(
    "command",
    ["for n in $(ls); do mkdir x-$n; done", "for n in $LIST; do mkdir x-$n; done"],
)
def test_unknown_loop_list_records_nothing(tmp_path: Path, command: str) -> None:
    assert resolve_command(command, cwd=str(tmp_path), env={}, home=str(tmp_path)).created == []


def test_loop_worktree_add_records_every_iteration(home: Path) -> None:
    """The 3ef6295a5545 shape, run for real through plan → bash → commit."""
    repo = home / "src" / "repo"
    repo.mkdir(parents=True)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "init"], cwd=repo, check=True)
    (repo / ".worktrees").mkdir()
    names = ("l1-tool-verbs", "l2-file-sizes")
    written = _run(
        home,
        f"for n in {' '.join(names)}; do "
        "git worktree add -q -b feat/i18n-$n .worktrees/i18n-$n main; done",
        cwd=repo,
    )
    for n in names:
        wt = os.path.realpath(repo / ".worktrees" / f"i18n-{n}")
        assert wt in written
        assert prov.owned_record(wt, SID) is not None


def test_loop_mkdir_records_each_and_never_a_preexisting_dir(home: Path) -> None:
    (home / "deep").mkdir()
    (home / "deep" / "x-b").mkdir()  # exists before: someone else's, never recorded
    written = _run(home, "for n in a b c; do mkdir -p ~/deep/x-$n; done")
    assert sorted(written) == [os.path.realpath(home / "deep" / f"x-{n}") for n in "ac"]


# -- Gap B: another session's created directory -----------------------------------------


def _osworld(home: Path, sid: str = OTHER) -> Path:
    base = home / "worktrees" / "osworld"
    base.mkdir(parents=True)
    _run(home, "mkdir ~/worktrees/osworld/osworld-21", sid=sid)
    runs = base / "osworld-21" / "runs"
    runs.mkdir()
    (runs / "result.json").write_text("{}")
    return base / "osworld-21"


@pytest.mark.parametrize("rel", ["osworld-21", "osworld-21/runs", ""])
def test_another_sessions_dir_child_or_ancestor_ranks_high(home: Path, rel: str) -> None:
    _osworld(home)
    target = f"~/worktrees/osworld/{rel}".rstrip("/")
    verdict = _verdict(home, f"rm -rf {target}")
    assert verdict.risk is Risk.HIGH
    assert f"created by another session ({OTHER}, mkdir" in verdict.targets[0].reason


def test_unrecorded_deep_dir_and_own_dir_stay_low(home: Path) -> None:
    (home / "worktrees" / "plain" / "runs").mkdir(parents=True)
    assert _verdict(home, "rm -rf ~/worktrees/plain/runs").risk is Risk.LOW
    _osworld(home, sid=SID)
    own = _verdict(home, "rm -rf ~/worktrees/osworld/osworld-21/runs")
    assert own.risk is Risk.LOW


@pytest.mark.parametrize("creator,deleter", [(CHILD, SID), (SID, CHILD)])
def test_same_task_tree_creation_stays_low(home: Path, creator: str, deleter: str) -> None:
    assert prov.same_task_tree(SID, CHILD)  # precondition: one subagent tree
    _osworld(home, sid=creator)
    verdict = _verdict(home, "rm -rf ~/worktrees/osworld/osworld-21/runs", sid=deleter)
    assert verdict.risk is Risk.LOW


def test_replaced_inode_no_longer_counts(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _osworld(home)
    assert _verdict(home, "rm -rf ~/worktrees/osworld/osworld-21/runs").risk is Risk.HIGH
    # Same path, another generation (deleted and re-made by someone else).
    real = prov.creation_stamp
    monkeypatch.setattr(prov, "creation_stamp", lambda st: real(st) + 1)
    assert _verdict(home, "rm -rf ~/worktrees/osworld/osworld-21/runs").risk is Risk.LOW


def test_unreadable_ledger_leaves_rank_unchanged(home: Path) -> None:
    _osworld(home)
    db = prov.ledger_path(home / ".local-operator")
    db.unlink()
    db.write_bytes(b"not a sqlite database" * 64)
    with pytest.raises(sqlite3.DatabaseError):
        sqlite3.connect(db).execute("SELECT * FROM created").fetchall()
    verdict = _verdict(home, "rm -rf ~/worktrees/osworld/osworld-21/runs")
    assert verdict.risk is Risk.LOW and verdict.targets[0].reason == "ordinary path"


def test_one_ledger_read_per_classify_call(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _osworld(home)
    (home / "worktrees" / "plain").mkdir()
    calls: list[object] = []
    real = prov._connect
    monkeypatch.setattr(prov, "_connect", lambda *a, **k: calls.append(a) or real(*a, **k))
    _verdict(home, "rm -rf ~/worktrees/plain ~/worktrees/osworld/osworld-21/runs ~/worktrees/x")
    assert len(calls) == 1


class _CountingJudge:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    async def __call__(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


@pytest.mark.asyncio
async def test_check_tool_call_refuses_and_names_the_owner(home: Path) -> None:
    _osworld(home)
    judge = _CountingJudge("BLOCK another session's benchmark evidence")
    ctx = CallContext(
        cwd=str(home),
        env=_env(home),
        scratchpad=None,
        session_id=SID,
        intent="cleaning up run output",
        operator=prov.operator_brief([], session_id=SID),
    )
    outcome = await check_tool_call(
        "bash",
        {"command": "rm -rf ~/worktrees/osworld/osworld-21/runs"},
        ctx,
        judge=None,
        settings=GuardSettings(),
        forked_judge=judge,
    )
    assert outcome is not None and outcome.blocked
    assert len(judge.prompts) == 1 and OTHER in judge.prompts[0]
    assert OTHER in outcome.text


# -- Review round 1 (PR #3835): live identity, nearest owner, case, cost ---------------

FRESH = "dddddddddddd"


def _rows(home: Path) -> list[tuple[str, str]]:
    conn = sqlite3.connect(prov.ledger_path(home / ".local-operator"))
    try:
        rows = conn.execute("SELECT owner, path FROM created ORDER BY owner, path").fetchall()
    finally:
        conn.close()
    return [(o, p.replace(str(home), "~")) for o, p in rows]


def test_dead_generation_owner_neither_excuses_nor_names(home: Path) -> None:
    """M1: SID's row for a removed ``~/w/runs`` must not stand in for OTHER's live one.

    Rows used to be grouped by path alone, so the dead row made the path look
    like SID's own (LOW) and an unrelated session's reason named SID.
    """
    (home / "w").mkdir()
    (home / ".local-operator" / "sessions" / FRESH / "scratchpad").mkdir(parents=True)
    _run(home, "mkdir ~/w/runs", sid=SID)
    subprocess.run(["rm", "-rf", str(home / "w" / "runs")], check=True)
    _run(home, "mkdir ~/w/runs && echo x > ~/w/runs/r.json", sid=OTHER)
    assert _rows(home) == [(SID, "~/w/runs"), (OTHER, "~/w/runs")]
    for sid in (SID, CHILD, FRESH):
        verdict = _verdict(home, "rm -rf ~/w/runs", sid=sid)
        assert verdict.risk is Risk.HIGH, sid
        assert f"created by another session ({OTHER}, mkdir" in verdict.targets[0].reason
        assert SID not in verdict.targets[0].reason


def _shared(home: Path) -> None:
    # Under ~/w, so no "direct child of $HOME" HIGH masks the ledger rule.
    (home / "w").mkdir()
    # ``mkdir -p`` records the topmost NEW component: OTHER, first in, owns ~/w/shared.
    _run(home, "mkdir -p ~/w/shared/qa-other", sid=OTHER)
    _run(home, "mkdir -p ~/w/shared/mine/sub", sid=SID)
    assert _rows(home) == [(SID, "~/w/shared/mine"), (OTHER, "~/w/shared")]


@pytest.mark.parametrize("rel", ["mine", "mine/sub"])
def test_own_dir_inside_a_foreign_dir_stays_low(home: Path, rel: str) -> None:
    """M2: the DEEPEST live row covering the target decides, and here it is SID's."""
    _shared(home)
    verdict = _verdict(home, f"rm -rf ~/w/shared/{rel}")
    assert verdict.risk is Risk.LOW, verdict.targets[0].reason


@pytest.mark.parametrize("rel", ["", "qa-other"])
def test_foreign_dir_itself_or_its_other_children_stay_high(home: Path, rel: str) -> None:
    """M2, other direction: ~/w/shared and OTHER's own child of it are still OTHER's."""
    _shared(home)
    verdict = _verdict(home, f"rm -rf ~/w/shared/{rel}".rstrip("/"))
    assert verdict.risk is Risk.HIGH
    assert f"created by another session ({OTHER}, mkdir" in verdict.targets[0].reason


def test_own_dir_holding_a_foreign_dir_ranks_high(home: Path) -> None:
    """Removing your own directory destroys another session's live one inside it."""
    (home / "w").mkdir()
    _run(home, "mkdir ~/w/mine", sid=SID)
    _run(home, "mkdir ~/w/mine/theirs", sid=OTHER)
    verdict = _verdict(home, "rm -rf ~/w/mine")
    assert verdict.risk is Risk.HIGH
    assert f"holds {home}/w/mine/theirs, created by another session ({OTHER}" in (
        verdict.targets[0].reason
    )


def _case_insensitive(path: Path) -> bool:
    probe = path / "CaseProbe"
    probe.mkdir()
    try:
        return (path / "caseprobe").exists()
    finally:
        probe.rmdir()


def test_case_variant_spelling_still_matches(home: Path) -> None:
    """m2: on case-insensitive APFS ``~/Worktrees`` IS ``~/worktrees``."""
    if not _case_insensitive(home):
        pytest.skip("case-sensitive filesystem: the variant names another path")
    _osworld(home)
    verdict = _verdict(home, "rm -rf ~/WORKTREES/Osworld/osworld-21/runs")
    assert verdict.risk is Risk.HIGH
    assert f"created by another session ({OTHER}, mkdir" in verdict.targets[0].reason


def test_ledger_has_a_path_index_and_reads_only_related_rows(home: Path) -> None:
    """m3: the foreign read is an indexed lookup, not a full-table scan."""
    _osworld(home)
    conn = sqlite3.connect(prov.ledger_path(home / ".local-operator"))
    try:
        indexed = {r[1] for r in conn.execute("PRAGMA index_list(created)")}
        cols = [[c[2] for c in conn.execute(f"PRAGMA index_info({name})")] for name in indexed]
        assert ["path"] in cols
        sql, params = prov._foreign_query([str(home / "plain" / "x"), str(home / "w")])
        plan = " | ".join(r[-1] for r in conn.execute(f"EXPLAIN QUERY PLAN {sql}", params))
    finally:
        conn.close()
    assert "SCAN created" not in plan and "USING INDEX" in plan, plan
