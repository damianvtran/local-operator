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
import sys
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


def _rerecord(home: Path, path: Path, sid: str, command: str) -> None:
    """Refresh ``sid``'s ledger row for ``path`` after the test changed that directory.

    WHY: on Linux ``creation_stamp`` is the directory's ``st_ctime_ns`` (no
    birth time there), and adding an entry moves it, so a row recorded before
    the fixture filled the directory stops matching and the foreign rule fails
    OPEN to "ordinary path" — the documented Linux limitation in
    ``ForeignCreations``. macOS stamps birth time, which is why these tests
    passed locally and failed on CI's Ubuntu full-suite shard (run
    43369555175). Same remedy as ``test_own_worktree_removal._record``: the
    ranking under test is the ledger rule, not the stamp's Linux semantics
    (``test_ctime_stamp_fails_open_once_the_dir_changes`` pins those).
    """
    prov.write_records(
        [(os.path.realpath(path), "mkdir")],
        owners=[sid],
        creator=sid,
        command=command,
        state_root=home / ".local-operator",
    )


def _osworld(home: Path, sid: str = OTHER) -> Path:
    base = home / "worktrees" / "osworld"
    base.mkdir(parents=True)
    _run(home, "mkdir ~/worktrees/osworld/osworld-21", sid=sid)
    runs = base / "osworld-21" / "runs"
    runs.mkdir()
    (runs / "result.json").write_text("{}")
    _rerecord(home, base / "osworld-21", sid, "mkdir ~/worktrees/osworld/osworld-21")
    return base / "osworld-21"


@pytest.mark.parametrize("rel", ["osworld-21", "osworld-21/runs", ""])
def test_another_sessions_dir_child_or_ancestor_ranks_high(home: Path, rel: str) -> None:
    _osworld(home)
    target = f"~/worktrees/osworld/{rel}".rstrip("/")
    verdict = _verdict(home, f"rm -rf {target}")
    assert verdict.risk is Risk.HIGH
    assert f"created by another session ({OTHER}, mkdir" in verdict.targets[0].reason


def test_ctime_stamp_fails_open_once_the_dir_changes(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Linux stamp semantics, runnable on any host (CI run 43369555175).

    With ``creation_stamp`` forced to ``st_ctime_ns`` (what Linux uses), a
    foreign row ranks HIGH while its directory is untouched since recording,
    and stops matching once an entry is added — the documented fail-open, so
    the rank returns to main's "ordinary path", never something lower.
    """
    monkeypatch.setattr(prov, "creation_stamp", lambda st: int(st.st_ctime_ns))
    (home / "w").mkdir()
    _run(home, "mkdir -p ~/w/theirs/runs && echo x > ~/w/theirs/runs/r.json", sid=OTHER)
    verdict = _verdict(home, "rm -rf ~/w/theirs/runs")
    assert verdict.risk is Risk.HIGH
    assert f"created by another session ({OTHER}, mkdir" in verdict.targets[0].reason
    before = os.stat(home / "w" / "theirs").st_ctime_ns
    # Kernel ctime is tick-granular: repeat the entry change until it moves.
    for _ in range(100_000):
        (home / "w" / "theirs" / "later").mkdir()
        (home / "w" / "theirs" / "later").rmdir()
        if os.stat(home / "w" / "theirs").st_ctime_ns != before:
            break
    assert os.stat(home / "w" / "theirs").st_ctime_ns != before
    verdict = _verdict(home, "rm -rf ~/w/theirs/runs")
    assert verdict.risk is Risk.LOW and verdict.targets[0].reason == "ordinary path"


def test_landing_recursion_keeps_foreign_ledger_and_critical_ranks(home: Path) -> None:
    """Convergence: a relink landing must share the call's foreign-ledger view."""
    from local_operator.guardrails.classify import _protected, _rank
    from local_operator.guardrails.resolve import Target

    foreign_dir = _osworld(home)
    ordinary = home / "worktrees" / "plain" / "runs"
    ordinary.mkdir(parents=True)
    prot = _protected(_env(home), None, SID)
    foreign = prov.ForeignCreations(SID)
    # Isolate the recursion's contract: both spellings are ordinary paths
    # without ledger evidence, so dropping `foreign` silently returns LOW.
    target = Target(path=str(ordinary), op="delete", program="rm", landings=(str(foreign_dir),))
    ranked = _rank(target, prot, SID, foreign)
    assert ranked.risk is Risk.HIGH
    assert f"created by another session ({OTHER}" in ranked.reason
    assert _rank(target, prot, SID).risk is Risk.LOW

    target.landings = (str(home / ".local-operator" / "config.yml"),)
    assert _rank(target, prot, SID, foreign).risk is Risk.CRITICAL


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
    # SID's mkdir moved ~/w/shared's Linux ctime stamp; see ``_rerecord``.
    _rerecord(home, home / "w" / "shared", OTHER, "mkdir -p ~/w/shared/qa-other")
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


# -- Review round 2 (PR #3835): M3, m5, m6 -------------------------------------------


def test_foreign_query_memory_is_bounded_for_a_very_deep_path(home: Path) -> None:
    """M3: the ancestor list must not hold every prefix of a 20k-component path.

    Before the bound each prefix of ``real`` was kept as a bind parameter:
    Σ len(prefix) = O(L²), ~100 MB of strings at 10k components and an OOM
    kill at 20k. Only ancestors no longer than ``PATH_MAX`` can be stored rows.
    """
    import tracemalloc

    deep = str(home) + "/w/f" + "/b" * 10_000
    tracemalloc.start()
    try:
        sql, params = prov._foreign_query([deep])
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 5 * 1024 * 1024, peak
    assert all(len(p) <= prov._path_max() for p in params)
    assert str(home) + "/w/f" in params and str(home) in params  # shallow ones kept
    assert len(params) <= prov._MAX_ANCESTORS + 2


def test_foreign_query_caps_ancestor_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """M3: even under a huge PATH_MAX only the shallowest ancestors are bound."""
    monkeypatch.setattr(prov, "_path_max", lambda: 10**9)
    _sql, params = prov._foreign_query(["/a" * 5_000])
    assert len(params) == prov._MAX_ANCESTORS
    assert params[:3] == ["/", "/a", "/a/a"]  # shallowest first; the deep range is dropped


def test_path_max_falls_back_when_os_pathconf_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows has no ``os.pathconf``; importing and calling must still work.

    CI ``xplat-probe-windows`` failed with ``AttributeError: module 'os' has no
    attribute 'pathconf'`` from this module's import. The callable is looked up
    rather than called unguarded, so the module imports and the helper returns
    a ceiling: 32767 (an extended-length NT path) on Windows, 1024 elsewhere.
    """
    expected = 32767 if os.name == "nt" else 1024
    monkeypatch.delattr(os, "pathconf", raising=False)
    prov._path_max.cache_clear()
    try:
        assert prov._path_max() == expected
        # The fallback bound reaches the query: shallow ancestors still bind
        # and the ancestor list stays capped.
        _sql, params = prov._foreign_query(["/a" * 2_000])
        assert params[:3] == ["/", "/a", "/a/a"]
        assert len(params) <= prov._MAX_ANCESTORS + 2
    finally:
        prov._path_max.cache_clear()  # do not cache the fallback for other tests

    # Cold interpreter with no ``os.pathconf`` at all: the import itself was
    # the CI failure, so it is exercised through a real import, not a reload.
    root = Path(__file__).resolve().parents[3]
    code = (
        "import os\n"
        "if hasattr(os, 'pathconf'):\n"
        "    del os.pathconf\n"
        "from local_operator.guardrails import provenance as p\n"
        "print(p._path_max())\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(root),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(root)},
        check=True,
    )
    assert proc.stdout.strip() == str(expected), proc.stdout + proc.stderr


def test_very_deep_target_classifies_bounded_and_keeps_other_targets_high(home: Path) -> None:
    """M3 end to end: the reviewer's repro, under a modest memory bound."""
    import tracemalloc

    (home / "w").mkdir()
    _run(home, "mkdir ~/w/f", sid=OTHER)
    tracemalloc.start()
    try:
        verdict = _verdict(home, "rm -rf ~/w/f" + "/b" * 20_000 + " ~/w/f")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 100 * 1024 * 1024, peak
    assert verdict.risk is Risk.HIGH
    assert any(f"created by another session ({OTHER}" in t.reason for t in verdict.targets)


def _pre_index_ledger(home: Path) -> Path:
    """A ledger as the pre-index code left it: same rows, no ``created_path``."""
    db = prov.ledger_path(home / ".local-operator")
    conn = sqlite3.connect(db)
    try:
        conn.execute("DROP INDEX created_path")
        conn.commit()
    finally:
        conn.close()
    return db


def test_read_only_pre_index_ledger_still_serves_lookups(home: Path) -> None:
    """m5: a read-only ledger without the index must not break either rule."""
    (home / "w").mkdir()
    _run(home, "mkdir ~/w/theirs", sid=OTHER)
    # This session's own worktree: LOW only through its ledger row — the
    # own-provenance path the reviewer saw flip to HIGH on this ledger.
    repo = home / "w" / "repo"
    repo.mkdir()
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "init"], cwd=repo, check=True)
    _run(home, "git worktree add -q -b x .worktrees/x main", sid=SID, cwd=repo)
    db = _pre_index_ledger(home)
    os.chmod(db, 0o444)
    os.chmod(db.parent, 0o555)  # no journal can be created next to it either
    try:
        foreign = _verdict(home, "rm -rf ~/w/theirs")
        own = _verdict(home, "rm -rf ~/w/repo/.worktrees/x")
    finally:
        os.chmod(db.parent, 0o755)
        os.chmod(db, 0o644)
    assert foreign.risk is Risk.HIGH, foreign.targets[0].reason
    assert f"created by another session ({OTHER}" in foreign.targets[0].reason
    assert own.risk is Risk.LOW and "created by this session" in own.targets[0].reason
    conn = sqlite3.connect(db)
    try:  # the read path wrote nothing: still no index
        assert "created_path" not in {r[1] for r in conn.execute("PRAGMA index_list(created)")}
    finally:
        conn.close()


def test_writer_holding_reserved_does_not_stall_or_hide_foreign_rows(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """m5: readers never take a write lock, so RESERVED (BEGIN IMMEDIATE) costs nothing.

    WHY THIS ASSERTS THE MECHANISM AND NOT A WALL-CLOCK BOUND: ``elapsed < 1.0``
    was a bet on machine load (AGENTS.md "Timing, flakes, and how to assert
    that something is fast"). The fact that cannot flake: the lookup never
    waits on the lock at all. The spy hands every read connection a ZERO busy
    timeout, so any contention would surface as SQLITE_BUSY at once, the
    lookup would degrade to "no foreign info" and the rank would drop to
    main's. The foreign row still ranking HIGH under a held RESERVED lock
    proves the reader never needed the busy handler. The EXCLUSIVE control
    below shows the same zero-wait spy does hide the row when a lock really
    blocks readers, so the assertion discriminates.
    """
    (home / "w").mkdir()
    _run(home, "mkdir ~/w/theirs", sid=OTHER)
    db = _pre_index_ledger(home)

    real_connect = sqlite3.connect
    read_opens: list[float] = []

    def zero_wait(*args, **kwargs):  # read connections get no busy wait at all
        if kwargs.get("timeout") == prov._DB_READ_TIMEOUT_S:
            read_opens.append(kwargs["timeout"])
            kwargs["timeout"] = 0
        return real_connect(*args, **kwargs)

    def verdict_under(lock: str):
        writer = real_connect(db, isolation_level=None)
        try:
            writer.execute(lock)
            # Scoped: ``monkeypatch.undo()`` would also revert the fixture's
            # HOME/config isolation.
            with monkeypatch.context() as m:
                m.setattr(prov.sqlite3, "connect", zero_wait)
                return _verdict(home, "rm -rf ~/w/theirs")
        finally:
            writer.execute("ROLLBACK")
            writer.close()

    verdict = verdict_under("BEGIN IMMEDIATE")
    assert read_opens, "the lookup did not open a read connection"
    assert verdict.risk is Risk.HIGH, verdict.targets[0].reason
    assert f"created by another session ({OTHER}" in verdict.targets[0].reason

    # Control: with zero busy wait, a lock that does block readers hides the row.
    read_opens.clear()
    control = verdict_under("BEGIN EXCLUSIVE")
    assert read_opens
    assert control.risk is not Risk.HIGH, control.targets[0].reason


def test_exclusive_lock_bounds_foreign_lookup_and_keeps_main_rank(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """m5: under an EXCLUSIVE writer the foreign lookup waits briefly, then ranks as main.

    WHY THIS ASSERTS THE MECHANISM AND NOT A WALL-CLOCK BOUND: the elapsed time
    is dominated by SQLite's own lock-reporting latency, not by the busy
    timeout we pass. A bare stdlib ``sqlite3.connect(timeout=0.25)`` against an
    EXCLUSIVE-locked file took ~1.3 s to report "locked" on a loaded dev box;
    QA measured the lookup at 0.74-0.79 s. Still bounded, but an absolute bound
    (``elapsed < _DB_TIMEOUT_S / 2`` measured 1.25-1.43 s and failed) is a bet
    on machine load (AGENTS.md "Timing, flakes, and how to assert that
    something is fast"). The facts that cannot flake: the connection is opened
    with ``_DB_READ_TIMEOUT_S`` (shorter than the 2 s write-path wait), SQLite
    is handed that value as its busy timeout, the lookup degrades to "no
    foreign info", and the final verdict is main's rank.
    """
    (home / "w").mkdir()
    _run(home, "mkdir ~/w/theirs", sid=OTHER)
    db = prov.ledger_path(home / ".local-operator")
    target = str(home / "w" / "theirs")
    assert prov._DB_READ_TIMEOUT_S < prov._DB_TIMEOUT_S

    real_connect = sqlite3.connect
    timeouts: list[float] = []

    def spy(*args, **kwargs):  # records the busy wait handed to SQLite
        timeouts.append(kwargs.get("timeout", 5.0))
        return real_connect(*args, **kwargs)

    writer = real_connect(db, isolation_level=None)
    foreign = prov.ForeignCreations(SID, state_root=home / ".local-operator")
    try:
        writer.execute("BEGIN EXCLUSIVE")
        # A scoped patch: ``monkeypatch.undo()`` would also revert the
        # fixture's HOME/config isolation.
        with monkeypatch.context() as m:
            m.setattr(prov.sqlite3, "connect", spy)
            hit = foreign.match([target])
            lookup_timeouts = list(timeouts)
            verdict = _verdict(home, f"rm -rf {target}")
    finally:
        foreign.close()
        writer.execute("ROLLBACK")
        writer.close()
    assert hit is None  # the lock is held: no foreign info, not a stall
    assert lookup_timeouts == [prov._DB_READ_TIMEOUT_S]
    # classify_bash's own ForeignCreations used the short wait too.
    assert prov._DB_READ_TIMEOUT_S in timeouts[len(lookup_timeouts) :]
    # n1: with the lookup given up, the rank is exactly main's (no ledger rule).
    assert verdict.risk is Risk.LOW and verdict.targets[0].reason == "ordinary path"

    # Unlocked, the same lookup's connection carries the short busy timeout
    # into SQLite itself, and the foreign row is found again.
    foreign = prov.ForeignCreations(SID, state_root=home / ".local-operator")
    try:
        conn = foreign._connection()
        assert conn is not None
        busy_ms = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert busy_ms == int(prov._DB_READ_TIMEOUT_S * 1000)
        assert foreign.match([target]) is not None
    finally:
        foreign.close()


@pytest.mark.parametrize("rel", ["w/f/\udcff", "w/plain/\udcff"])
def test_lone_surrogate_target_does_not_raise(home: Path, rel: str) -> None:
    """m6: a surrogateescape'd name (``rm -rf build/*`` on Linux) cannot be bound in SQLite."""
    (home / "w" / "plain").mkdir(parents=True)
    _run(home, "mkdir ~/w/f", sid=OTHER)
    verdict = _verdict(home, f"rm -rf ~/{rel}")
    if rel.startswith("w/f/"):  # its encodable ancestor still matches the foreign row
        assert verdict.risk is Risk.HIGH, verdict.targets[0].reason
    else:
        assert verdict.risk is Risk.LOW and verdict.targets[0].reason == "ordinary path"
