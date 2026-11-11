"""A ``..`` after a symlink is ranked where the KERNEL lands it.

WHY THESE TESTS. Path lookup follows a link BEFORE it applies the ``..`` after
it, but the resolver used to ``normpath`` every word first. With
``~/other/sl -> <repo>/subdir``, ``rm -rf ~/other/sl/../.git`` read as
``~/other/.git`` ("ordinary path", LOW) while ``rm`` deletes ``<repo>/.git`` —
and the same shape reached the state root's ``config.yml``, any repo root or
``$HOME`` child (security review on PR #3802, comment 6801527527). Every
CRITICAL/HIGH target in the dangerous-delete guard was reachable at LOW.

Fixtures are synthetic (tmp HOME, session id ``own``); nothing here touches a
real checkout.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from local_operator.guardrails.classify import Risk, classify_bash


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    state = home / ".local-operator"
    (state / "sessions" / "own" / "scratchpad" / "build").mkdir(parents=True)
    (state / "secrets").mkdir()
    (state / "config.yml").write_text("x: 1\n")
    repo = home / "repo"
    (repo / "subdir").mkdir(parents=True)
    env = {**os.environ, "HOME": str(home), "GIT_CONFIG_SYSTEM": os.devnull}
    env.pop("GIT_DIR", None)
    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=env)
    (home / "w" / "other").mkdir(parents=True)
    (home / "a" / "b").mkdir(parents=True)
    (home / "x").mkdir()
    # The bypass shape: a link from an ordinary place into the repository.
    (home / "w" / "other" / "sl").symlink_to(repo / "subdir")
    (home / "w" / "other" / "stl").symlink_to(state / "sessions")
    (home / "x" / "sl").symlink_to(home / "a" / "b")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))
    return home


def _pad(home: Path) -> Path:
    return home / ".local-operator" / "sessions" / "own" / "scratchpad"


def _classify(home: Path, command: str, *, cwd: Path | None = None):
    return classify_bash(
        command,
        cwd=str(cwd or home),
        env={"HOME": str(home), "LOCAL_OPERATOR_SCRATCHPAD": str(_pad(home))},
        scratchpad=str(_pad(home)),
        session_id="own",
    )


def _only(verdict):
    assert len(verdict.targets) == 1, verdict.targets
    return verdict.targets[0]


def _real(path: Path) -> str:
    return os.path.realpath(path)


def test_dotdot_through_link_reaches_the_repo_git_dir(home: Path) -> None:
    ranked = _only(_classify(home, "rm -rf ~/w/other/sl/../.git"))
    assert ranked.target.path == _real(home / "repo" / ".git")
    assert ranked.risk is Risk.HIGH
    # Same ranking as the direct spelling of what the kernel deletes.
    direct = _only(_classify(home, "rm -rf ~/repo/.git"))
    assert (ranked.risk, ranked.reason) == (direct.risk, direct.reason)


def test_dotdot_through_link_ranks_like_the_direct_worktrees_dir(home: Path) -> None:
    (home / "repo" / ".worktrees" / "foreign").mkdir(parents=True)
    ranked = _only(_classify(home, "rm -rf ~/w/other/sl/../.worktrees"))
    assert ranked.target.path == _real(home / "repo" / ".worktrees")
    direct = _only(_classify(home, "rm -rf ~/repo/.worktrees"))
    assert (ranked.risk, ranked.reason) == (direct.risk, direct.reason)


def test_dotdot_through_link_to_the_state_root_is_critical(home: Path) -> None:
    ranked = _only(_classify(home, "rm -rf ~/w/other/stl/../config.yml"))
    assert ranked.risk is Risk.CRITICAL
    assert "config" in ranked.reason


def test_trailing_dotdot_names_the_links_real_parent(home: Path) -> None:
    # ``~/x/sl`` -> ``~/a/b``: ``~/x/sl/..`` is ``~/a``, a top-level home dir.
    ranked = _only(_classify(home, "rm -rf ~/x/sl/.."))
    assert ranked.target.path == _real(home / "a")
    assert ranked.risk is Risk.HIGH


def test_trailing_dotdot_reaching_a_repo_root_is_high(home: Path) -> None:
    ranked = _only(_classify(home, "rm -rf ~/w/other/sl/.."))
    assert ranked.target.path == _real(home / "repo")
    assert ranked.risk is Risk.HIGH


def test_glob_prefix_dotdot_expands_through_the_link(home: Path) -> None:
    verdict = _classify(home, "rm -rf ~/w/other/sl/../.gi*")
    assert {t.target.path for t in verdict.targets} == {_real(home / "repo")}


def test_relative_dotdot_after_logical_cd_into_a_link(home: Path) -> None:
    # ``cd`` keeps the logical ``…/sl``; the kernel's cwd is ``repo/subdir``.
    ranked = _only(_classify(home, "cd ~/w/other/sl && rm -rf ../.git"))
    assert ranked.target.path == _real(home / "repo" / ".git")
    assert ranked.risk is Risk.HIGH


def test_physical_cd_dotdot_then_relative_rm_is_high(home: Path) -> None:
    ranked = _only(_classify(home, "cd -P ~/w/other/sl/.. && rm -rf .git"))
    assert ranked.target.path == _real(home / "repo" / ".git")
    assert ranked.risk is Risk.HIGH


def test_logical_cd_dotdot_collapses_lexically_like_bash(home: Path) -> None:
    # bash/zsh canonicalise a LOGICAL cd's ``..`` before chdir (measured
    # ``pwd -P``): ``cd ~/w/other/sl/..`` really is ``~/w/other``.
    ranked = _only(_classify(home, "cd ~/w/other/sl/.. && rm -rf .git"))
    assert ranked.target.path == str(home / "w" / "other" / ".git")
    assert ranked.risk is Risk.LOW


def test_session_cwd_with_dotdot_is_resolved_physically(home: Path) -> None:
    ranked = _only(_classify(home, "rm -rf .git", cwd=home / "w" / "other" / "sl" / ".."))
    assert ranked.target.path == _real(home / "repo" / ".git")
    assert ranked.risk is Risk.HIGH


def test_plain_dotdot_without_links_is_unchanged(home: Path) -> None:
    ranked = _only(_classify(home, "rm -rf ~/a/b/../c"))
    assert ranked.target.path == str(home / "a" / "c")
    assert ranked.risk is Risk.LOW


def test_own_scratch_dotdot_stays_low(home: Path) -> None:
    ranked = _only(_classify(home, f"rm -rf {_pad(home)}/build/../tmp"))
    assert ranked.target.path == str(_pad(home) / "tmp")
    assert ranked.risk is Risk.LOW


def test_dotdot_after_a_missing_component_is_never_low(home: Path) -> None:
    # ``n`` does not exist yet: the ``ln`` before the ``rm`` could make it a
    # link anywhere, so the lexical ``~/w/.git`` must not earn a LOW.
    verdict = _classify(home, "ln -s ~/repo/subdir ~/w/n; rm -rf ~/w/n/../.git")
    ranked = next(t for t in verdict.targets if t.target.op == "delete")
    assert ranked.risk is Risk.HIGH
    assert "unresolved" in ranked.reason


def test_dotdot_after_a_missing_component_in_own_pad_is_never_low(home: Path) -> None:
    pad = _pad(home)
    verdict = _classify(home, f"ln -s ~/repo/subdir {pad}/n; rm -rf {pad}/n/../.git")
    assert verdict.risk is not Risk.LOW
    assert all(t.risk is not Risk.LOW for t in verdict.targets if t.target.op == "delete")


def test_dotdot_after_a_missing_component_keeps_critical(home: Path) -> None:
    verdict = _classify(home, "rm -rf ~/.local-operator/nope/../config.yml")
    assert verdict.risk is Risk.CRITICAL


def test_trailing_link_itself_is_not_followed(home: Path) -> None:
    # ``rm -rf ~/w/other/sl`` removes the link, not ``repo/subdir``: its path
    # stays the link (the realpath is still judged as a second spelling).
    ranked = _only(_classify(home, "rm -rf ~/w/other/sl/../subdir/../../w/other/sl"))
    assert ranked.target.path == str(home / "w" / "other" / "sl")


# -- round-1 remediation (PR #3811) -------------------------------------------------
#
# R1-1: ``physical_path`` reads the filesystem as it is at CHECK time, but an
# earlier part of the same command can replace a component with a new link
# (``ln -sfn``, ``rm && ln -s``, ``rm && mv <link>``) before the ``..`` is
# looked up. Every such shape ranked LOW (``<pad>/config.yml``) while the
# kernel deleted the state root's ``config.yml``.


@pytest.fixture
def lnk(home: Path) -> Path:
    """``<pad>/lnk -> <pad>/build`` (an innocent link at check time) and an
    existing ``<pad>/tmpd``."""
    pad = _pad(home)
    (pad / "lnk").symlink_to(pad / "build")
    (pad / "tmpd").mkdir()
    return pad


@pytest.mark.parametrize(
    "command",
    [
        "ln -sfn ~/.local-operator/secrets {pad}/lnk && rm -f {pad}/lnk/../config.yml",
        "ln -sfn ~/.local-operator/secrets {pad}/lnk; echo x > {pad}/lnk/../config.yml",
        "ln -sfn ~/repo/subdir {pad}/lnk && rm -rf {pad}/lnk/../.git",
        "rm -rf {pad}/tmpd && ln -s ~/repo/subdir {pad}/tmpd && rm -rf {pad}/tmpd/../.git",
        "rm -rf {pad}/tmpd && mv ~/w/other/sl {pad}/tmpd && rm -rf {pad}/tmpd/../.git",
        "cp -a ~/w/other/sl {pad}/tmpd && rm -rf {pad}/tmpd/../.git",
        "ln -sfn ~/repo/subdir {pad}/lnk && cd {pad}/lnk && rm -rf ../.git",
        "for d in {pad}/lnk; do ln -sfn ~/repo/subdir $d; rm -rf $d/../.git; done",
    ],
)
def test_dotdot_after_a_component_relinked_earlier_is_never_low(
    home: Path, lnk: Path, command: str
) -> None:
    verdict = _classify(home, command.format(pad=lnk))
    # The ``..`` word's row (the plain ``rm -rf <pad>/tmpd`` stays LOW).
    dotdot = [t for t in verdict.targets if "unresolved" in t.reason]
    assert dotdot, verdict.targets
    assert all(t.risk is Risk.HIGH for t in dotdot), dotdot
    assert verdict.risk is Risk.HIGH


def test_relinked_component_keeps_a_lexical_critical(home: Path, lnk: Path) -> None:
    command = f"ln -sfn /tmp {lnk}/lnk && rm -f {lnk}/lnk/../../../../config.yml"
    assert _classify(home, command).risk is Risk.CRITICAL


def test_existing_link_not_relinked_still_resolves_low(home: Path, lnk: Path) -> None:
    # Control: the same ``..`` through the same check-time link, with no
    # earlier part of the command touching it, stays LOW in the own pad.
    ranked = _only(_classify(home, f"rm -f {lnk}/lnk/../config.yml"))
    assert ranked.risk is Risk.LOW
    # ...and a link made earlier elsewhere does not unpin an unrelated word.
    verdict = _classify(home, f"ln -sfn /tmp {lnk}/other && rm -f {lnk}/lnk/../x")
    assert verdict.risk is Risk.LOW


# R1-2: the ``.git`` rule judged one path, so a glob naming ``.git`` ranked by
# its literal prefix (``~/repo``) as an ordinary path.


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf ~/repo/.gi*",
        "rm -rf ~/repo/.g?t",
        "rm -rf ~/repo/[.]git",
        "rm -rf ~/repo/.[g]it",
        "cd ~/repo && rm -rf .g*",
    ],
)
def test_glob_naming_a_git_dir_is_high(home: Path, command: str) -> None:
    verdict = _classify(home, command)
    assert verdict.risk is Risk.HIGH, verdict.targets
    assert any(".git directory" in t.reason for t in verdict.targets)


def test_glob_that_cannot_match_a_leading_dot_skips_the_git_rule(home: Path) -> None:
    (home / "repo" / "f.txt").write_text("x")
    # bash's ``*`` never matches ``.git`` without ``dotglob``.
    assert _classify(home, "rm -rf ~/repo/*.txt").risk is Risk.LOW


def test_glob_over_own_clone_git_dir_stays_low(home: Path) -> None:
    from local_operator.guardrails import provenance

    clone = home / "clone"
    env = {**os.environ, "HOME": str(home), "GIT_CONFIG_SYSTEM": os.devnull}
    subprocess.run(["git", "init", "-q", str(clone)], check=True, env=env)
    provenance.write_records(
        [(os.path.realpath(clone), "clone")], owners=["own"], creator="own", command="git clone"
    )
    assert _classify(home, "rm -rf ~/clone/.git").risk is Risk.LOW
    assert _classify(home, "rm -rf ~/clone/.gi*").risk is Risk.LOW


# Pre-existing over-block (both round-1 reports): a ``..`` after a missing
# component whose lexical landing is the OWN pad ranked CRITICAL ("session
# state"), which no judge can lift, for the agent's own cleanup.


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf {pad}/nope/../x",
        "mkdir -p {pad}/n && rm -rf {pad}/n/../build",
        "rm -rf $LOCAL_OPERATOR_SCRATCHPAD/newdir/../tmp",
    ],
)
def test_own_pad_dotdot_after_missing_component_is_high_not_critical(
    home: Path, command: str
) -> None:
    verdict = _classify(home, command.format(pad=_pad(home)))
    assert verdict.risk is Risk.HIGH, verdict.targets
    assert all(t.risk is Risk.HIGH and "unresolved" in t.reason for t in verdict.targets)


def test_missing_component_landing_outside_own_pad_in_state_stays_critical(home: Path) -> None:
    pad = _pad(home)
    # Lexically lands in another session directory: still evidence.
    assert _classify(home, f"rm -rf {pad}/nope/../../../other/x").risk is Risk.CRITICAL
