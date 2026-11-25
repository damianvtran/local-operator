"""Several ``git worktree remove`` targets in one call are judged one by one.

Field case (guard 0.113.3): one bash loop ran ``git -C <repo> worktree remove
--force -- <repo>/.worktrees/<n>`` for nine trees and was refused as REVIEW
with an empty "scoped eligibility list", although each eligible tree alone
ranked LOW. ``_literal_worktree_remove`` admitted exactly one target, so every
forced target of a loop or chain lost the exact route. These tests pin the
per-target reading: eligible trees rank exactly as alone, only the ineligible
one is refused and named, and an unpinned group still refuses wholesale.

Synthetic repos only (fixtures from ``test_finished_work_cleanup``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from local_operator.guardrails import classify as classify_mod
from local_operator.guardrails.classify import Risk
from local_operator.guardrails.resolve import resolve_command
from tests.unit.guardrails.test_finished_work_cleanup import (  # noqa: F401
    ME,
    _branch_tree,
    _git,
    _Judge,
    _rank,
    _record,
    home,
    repo,
)
from tests.unit.harness.test_loop_guardrail import _run, _text

NAMES = ["a1", "b2", "c3", "d4", "e5", "f6", "g7", "h8"]


@pytest.fixture
def real_home(home: Path, repo: Path) -> Path:  # noqa: F811
    # The exact route demands raw == realpath; macOS tmp lives under /private.
    return Path(os.path.realpath(home))


def _finished(root: Path, git_repo: Path, name: str) -> Path:
    """An own tree whose HEAD is on origin and whose only change is a deletion:
    the interrupted-cleanup shape that ranks LOW only on the exact route."""
    wt = _branch_tree(root, git_repo, name)
    _git("merge", "-q", "--ff-only", f"fix/{name}", cwd=git_repo)
    _git("push", "-q", "origin", "main", f"fix/{name}", cwd=git_repo)
    _git("fetch", "-q", "origin", cwd=git_repo)
    _record(root, wt, creator=ME)
    (wt / "a.txt").unlink()
    return wt


def _loop(git_repo: Path, names: list[str]) -> str:
    return (
        f"for n in {' '.join(names)}; do git -C {git_repo} worktree remove --force -- "
        f'"{git_repo}/.worktrees/$n"; done'
    )


def _by_name(verdict: classify_mod.Verdict) -> dict[str, classify_mod.RankedTarget]:
    return {os.path.basename(r.path): r for r in verdict.targets}


@pytest.mark.asyncio
async def test_nine_path_loop_refuses_only_the_ineligible_tree(real_home: Path) -> None:
    repo_ = real_home / "repo"
    trees = [_finished(real_home, repo_, n) for n in NAMES]
    bad = _branch_tree(real_home, repo_, "full-diff-history")
    _git("worktree", "lock", "--reason", "initializing", str(bad), cwd=repo_)
    alone = {
        t.name: _rank(real_home, f"git -C {repo_} worktree remove --force -- {t}") for t in trees
    }
    assert {v.risk for v in alone.values()} == {Risk.LOW}
    verdict = _rank(real_home, _loop(repo_, [*NAMES, bad.name]))
    ranked = _by_name(verdict)
    for name in NAMES:
        # Same path, same rank and reason, alone or in the group.
        assert ranked[name].risk is Risk.LOW
        assert ranked[name].reason == alone[name].targets[0].reason
    assert ranked[bad.name].risk is not Risk.LOW
    assert [os.path.basename(r.path) for r in verdict.flagged_targets] == [bad.name]
    # The real runner has a fake bash callback: even under auto-approval the
    # mixed call never reaches execute, whereas the all-LOW control does.
    mixed, dispatched = await _run(
        real_home, _loop(repo_, [*NAMES, bad.name]), judge=_Judge(), session_id=ME
    )
    assert mixed.is_error and dispatched == []
    text = _text(mixed)
    assert f"{bad} (" in text
    assert "Nothing was run." in text
    assert "Diagnostic only: 8 other resolved worktree paths ranked LOW" in text
    assert f"{trees[0]}, " in text
    assert "without the refused" not in text
    clean_cmd = _loop(repo_, NAMES)
    clean, dispatched = await _run(real_home, clean_cmd, session_id=ME)
    assert not clean.is_error, _text(clean)
    assert dispatched == [{"command": clean_cmd}]


@pytest.mark.parametrize("sep", [" && ", "; ", "\n"])
def test_mixed_chain_with_one_untracked_tree(real_home: Path, sep: str) -> None:
    repo_ = real_home / "repo"
    good = [_finished(real_home, repo_, n) for n in NAMES[:3]]
    bad = _branch_tree(real_home, repo_, "wip")
    (bad / "untracked.txt").write_text("wip\n")
    cmd = sep.join(f"git -C {repo_} worktree remove --force -- {t}" for t in [*good, bad])
    ranked = _by_name(_rank(real_home, cmd))
    assert [ranked[t.name].risk for t in good] == [Risk.LOW] * 3
    assert ranked["wip"].risk is Risk.HIGH
    assert "dirty" in ranked["wip"].reason
    loop = _by_name(_rank(real_home, _loop(repo_, [*NAMES[:3], "wip"])))
    assert {n: r.risk for n, r in loop.items()} == {n: r.risk for n, r in ranked.items()}


@pytest.mark.parametrize(
    "words",
    ["a1 $X", "a1 *", "a1 ~", "a1 ..", "$(ls)"],
    ids=["var", "glob", "tilde", "dotdot", "sub"],
)
def test_unpinned_group_keeps_the_whole_call_off_the_exact_route(
    real_home: Path, words: str
) -> None:
    repo_ = real_home / "repo"
    _finished(real_home, repo_, "a1")
    cmd = f'for n in {words}; do git -C {repo_} worktree remove --force -- "{real_home}/$n"; done'
    env = {"HOME": str(real_home)}

    res = resolve_command(cmd, cwd=str(real_home), env=env, home=str(real_home))
    assert classify_mod._literal_worktree_remove(cmd, res.targets) == frozenset()
    verdict = _rank(real_home, cmd)
    assert verdict.risk is not Risk.LOW
    if words == "a1 $X":
        # An unset $X makes the operand $HOME itself: CRITICAL for the whole call.
        assert verdict.risk is Risk.CRITICAL


@pytest.mark.parametrize("operand", ['"{root}/$n"', '"{root}/${{n}}"'])
def test_only_quoted_loop_reference_proves_one_operand(real_home: Path, operand: str) -> None:
    body = "git worktree remove --force -- " + operand.format(root=real_home)
    cmd = f"for n in 'two words'\ndo\n  {body}\ndone"
    env = {"HOME": str(real_home)}
    res = resolve_command(cmd, cwd=str(real_home), env=env, home=str(real_home))
    assert classify_mod._literal_worktree_remove(cmd, res.targets) == frozenset(
        {str(real_home / "two words")}
    )
    # Classification only: neither command is passed to Bash or a real tool.
    assert classify_mod._literal_remove_paths(cmd.replace('"', "")) is None


@pytest.mark.asyncio
async def test_one_high_row_excludes_path_from_low_diagnostic(real_home: Path) -> None:
    repo_ = real_home / "repo"
    mixed = _finished(real_home, repo_, "mixed")
    low = _finished(real_home, repo_, "low")
    prefix = f"git -C {repo_} worktree remove"
    cmd = "; ".join(
        [f"{prefix} -- {mixed}", f"{prefix} --force -- {mixed}", f"{prefix} --force -- {low}"]
    )
    verdict = _rank(real_home, cmd)
    assert {r.risk for r in verdict.targets if r.path == str(mixed)} == {Risk.LOW, Risk.HIGH}
    assert any(r.cleanup_hold and r.path == str(mixed) for r in verdict.targets)
    result, dispatched = await _run(real_home, cmd, session_id=ME)
    assert result.is_error and dispatched == []
    text = _text(result)
    assert "Nothing was run." in text
    flagged, diagnostic = text.split("Diagnostic only:", 1)
    assert str(mixed) in flagged and str(mixed) not in diagnostic
    assert f"1 other resolved worktree path ranked LOW in every row ({low})" in diagnostic
    assert "without the refused" not in text


def test_single_target_control_is_unchanged(real_home: Path) -> None:
    repo_ = real_home / "repo"
    wt = _finished(real_home, repo_, "solo")
    cmd = f"git -C {repo_} worktree remove --force -- {wt}"
    env = {"HOME": str(real_home)}
    res = resolve_command(cmd, cwd=str(real_home), env=env, home=str(real_home))
    assert classify_mod._literal_worktree_remove(cmd, res.targets) == frozenset({str(wt)})
    verdict = _rank(real_home, cmd)
    assert verdict.risk is Risk.LOW
    assert "verified removal payload (deletions-only)" in verdict.targets[0].reason
    # Anything else in the call keeps the base's "one literal command" bar.
    for other in (f"cd {repo_} && {cmd}", f"{cmd} &", f"{cmd} | cat", f"{cmd} # x"):
        r2 = resolve_command(other, cwd=str(real_home), env=env, home=str(real_home))
        assert classify_mod._literal_worktree_remove(other, r2.targets) == frozenset(), other
