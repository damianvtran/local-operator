"""Own finished worktrees go without asking (operator directive, 2026-11-24).

Operator directive (Damian, via Aida, 2026-11-24): agents keep asking
permission to clean up after themselves, and that feeds disk-full crashes.
Triage of 14 recent asks: 8 followed a real refusal, most of them "N ignored
entries the guard cannot prove regenerable" on the agent's OWN merged tree.

These tests pin the guard half of the fix: a tree the creation ledger proves
this session (or its delegating family) made sheds every ignored entry except
a nested repository, on both the plain and the ``--force`` route, without a
judge call; kin, foreign, untracked and critical cases stay as on base.
Synthetic repositories only, in the shared tmp-HOME fixture.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_operator.guardrails import provenance as prov
from local_operator.guardrails.check import CallContext
from local_operator.guardrails.classify import Risk, classify_bash
from local_operator.guardrails.preview import BUILTIN_PASS, preview_tool_call
from tests.unit.guardrails import test_own_worktree_removal as _base
from tests.unit.guardrails.test_own_worktree_removal import S, S2, _check, _env, _Judge, _record

# The synthetic tmp-HOME layout, shared rather than copied.
home = _base.home
mobile = _base.mobile

#: The ignored payload of the field asks: secrets, logs, a local database and
#: a custom output directory, none of them on ``_REGENERABLE``.
PAYLOAD = (".env", "logs/", "tmp-db/", "out-x/")


def _exclude(tree: Path, *patterns: str) -> None:
    # info/exclude, not .gitignore: editing a tracked file is a change Git refuses.
    path = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-path", "info/exclude"],
        cwd=tree,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write("".join(f"{p}\n" for p in patterns))


def _payload_tree(home: Path, mobile: Path, name: str = "keyboard-bound") -> Path:
    tree = mobile / ".worktrees" / name
    _exclude(tree, *PAYLOAD)
    (tree / ".env").write_text("LOCAL_ONLY=1\n")
    (tree / "logs").mkdir()
    (tree / "logs" / "run.log").write_text("ok\n")
    (tree / "tmp-db").mkdir()
    (tree / "tmp-db" / "data.sqlite").write_bytes(b"SQLite format 3\x00")
    (tree / "out-x").mkdir()
    (tree / "out-x" / "bundle.js").write_text("void 0\n")
    _record(home, tree)  # the root changed after recording (Linux ctime stamp)
    return tree


def _rank(home: Path, command: str, sid: str = S) -> Any:
    prov._cwd_inventory_cache = None
    return classify_bash(command, cwd=str(home), env=_env(home), scratchpad=None, session_id=sid)


def _preview(home: Path, command: str, sid: str = S) -> Any:
    ctx = CallContext(cwd=str(home), env=_env(home), scratchpad=None, session_id=sid)
    return preview_tool_call("bash", {"command": command}, ctx)


def _spellings(tree: Path) -> list[str]:
    return [
        f"git worktree remove {tree}",
        f"cd {tree.parent.parent} && git worktree remove .worktrees/{tree.name}",
    ]


@pytest.mark.asyncio
async def test_own_tree_with_hand_made_ignored_payload_goes_without_asking(
    home: Path, mobile: Path
) -> None:
    tree = _payload_tree(home, mobile)
    inventory = prov.plain_cleanup_ignored_inventory(str(tree), own_regenerable=True)
    assert inventory == prov.IgnoredInventory(unverified_ignored=0, changes=0)
    for command in _spellings(tree):
        v = _rank(home, command)
        assert v.risk is Risk.LOW, (command, [t.reason for t in v.targets])
        assert not any(t.cleanup_hold for t in v.targets), command
        report = _preview(home, command)
        assert report.builtin_effect == BUILTIN_PASS, report.render()
        judge = _Judge("BLOCK: should not be asked")
        outcome = await _check(home, command, judge)
        assert outcome is None or not outcome.blocked, outcome and outcome.text
        assert judge.prompts == [], command
    # ``--force``: checked at its payload boundary, as in
    # ``test_exact_force_route_sheds_own_regenerable_payload``; the route's
    # later origin proof needs a forge remote this fixture does not have.
    real = os.path.realpath(tree)
    assert prov._interrupted_cleanup_state(real, None, own_regenerable=True) == "clean"
    assert prov._interrupted_cleanup_state(real, None) == "dirty"


@pytest.mark.asyncio
async def test_own_tree_with_an_ignored_nested_repository_stays_held(
    home: Path, mobile: Path
) -> None:
    tree = _payload_tree(home, mobile)
    subprocess.run(["git", "init", "-q", str(tree / "tmp-db" / "checkout")], check=True)
    _record(home, tree)
    inventory = prov.plain_cleanup_ignored_inventory(str(tree), own_regenerable=True)
    assert inventory is not None and inventory.unverified_ignored == 1, inventory
    v = _rank(home, f"git worktree remove {tree}")
    assert v.risk is Risk.HIGH and v.targets[0].cleanup_hold, v.targets[0].reason
    real = os.path.realpath(tree)
    assert prov._interrupted_cleanup_state(real, None, own_regenerable=True) == "dirty"


def test_foreign_tree_still_holds_its_ignored_env(home: Path, mobile: Path) -> None:
    tree = _payload_tree(home, mobile)
    v = _rank(home, f"git worktree remove {tree}", sid=S2)
    assert v.risk is Risk.HIGH and v.targets[0].cleanup_hold, v.targets[0].reason
    assert "ignored entr" in v.targets[0].reason


def test_own_tree_with_untracked_work_is_not_low(home: Path, mobile: Path) -> None:
    tree = _payload_tree(home, mobile)
    (tree / "notes.md").write_text("untracked work\n")
    _record(home, tree)
    inventory = prov.plain_cleanup_ignored_inventory(str(tree), own_regenerable=True)
    assert inventory is not None and inventory.changes == 1
    v = _rank(home, f"git worktree remove --force {tree}")
    assert v.risk is not Risk.LOW


@pytest.mark.asyncio
async def test_home_primary_state_and_empty_expansion_stay_refused(
    home: Path, mobile: Path
) -> None:
    for command in (
        "rm -rf ~",
        'rm -rf "$EMPTY"/',
        "rm -rf ~/.local-operator",
        f"rm -rf {mobile}",
    ):
        judge = _Judge("ALLOW ledger")
        outcome = await _check(home, command, judge)
        assert outcome is not None and outcome.blocked, command


def test_sibling_kin_tree_still_holds_its_ignored_env(home: Path, mobile: Path) -> None:
    """A sibling's tree (``kin``) is not this session's work product."""
    state = home / ".local-operator"
    root, maker, remover = "aaaaaaaaaaa0", "aaaaaaaaaaa1", "aaaaaaaaaaa2"
    for sid in (root, maker, remover):
        (state / "sessions" / sid / "scratchpad").mkdir(parents=True, exist_ok=True)
    prov.record_lineage(maker, origin="subagent", parent=root, state_root=state)
    prov.record_lineage(remover, origin="subagent", parent=root, state_root=state)
    tree = _payload_tree(home, mobile)
    prov.write_records(
        [(os.path.realpath(tree), "worktree")],
        owners=[maker],
        creator=maker,
        command=f"git worktree add -q .worktrees/{tree.name}",
        state_root=state,
    )
    v = _rank(home, f"git worktree remove {tree}", sid=remover)
    assert v.risk is Risk.HIGH and v.targets[0].cleanup_hold, v.targets[0].reason
    assert "ignored entr" in v.targets[0].reason


def test_ignored_hold_text_sends_no_one_to_ask_about_their_own_tree() -> None:
    """Directive 2026-11-24: "whether the rest may be deleted is the operator's
    decision" read as a cue to ask, even for the agent's own trees."""
    from local_operator.guardrails.check import SUPPORTED_DELETE_PATHS, cleanup_hold_text
    from local_operator.guardrails.classify import HOLD_IGNORED_UNVERIFIED, RankedTarget, Verdict
    from local_operator.guardrails.resolve import Target

    target = RankedTarget(
        Target(path="/synthetic/wt", op="delete", program="git"),
        Risk.HIGH,
        "linked worktree state: 1 ignored entry the guard cannot prove regenerable",
        cleanup_hold=True,
        hold_conditions=(HOLD_IGNORED_UNVERIFIED,),
    )
    text = cleanup_hold_text(Verdict(Risk.HIGH, [target]))
    assert "operator's decision" not in text
    assert "no operator question" in text and "paths you did not create" in text
    assert "`preview_tool_call` the identical call before asking" in text
    assert "re-issue it once, not ask" in text
    assert "`preview_tool_call` the identical call first" in SUPPORTED_DELETE_PATHS
    assert "Team Bridge" in SUPPORTED_DELETE_PATHS and "cannot" in SUPPORTED_DELETE_PATHS
