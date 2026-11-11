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


# A ``..`` after a missing component whose lexical landing is the OWN pad
# keeps main's CRITICAL ("session state"). Rounds 1-3 relaxed it to HIGH for
# the agent's own cleanup, and every round found one more move/copy shape that
# used the relaxation to reach the state root (R2-1, R3-1, R4-1 on PR #3811),
# so it was removed in round 4: name the physical path instead. The over-block
# is main's behaviour and is deferred to a follow-up.


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf {pad}/nope/../x",
        "mkdir -p {pad}/n && rm -rf {pad}/n/../build",
        "rm -rf $LOCAL_OPERATOR_SCRATCHPAD/newdir/../tmp",
    ],
)
def test_own_pad_dotdot_after_missing_component_keeps_mains_critical(
    home: Path, command: str
) -> None:
    verdict = _classify(home, command.format(pad=_pad(home)))
    assert verdict.risk is Risk.CRITICAL, verdict.targets


def test_missing_component_landing_outside_own_pad_in_state_stays_critical(home: Path) -> None:
    pad = _pad(home)
    # Lexically lands in another session directory: still evidence.
    assert _classify(home, f"rm -rf {pad}/nope/../../../other/x").risk is Risk.CRITICAL


# Round 2 (review R2-1/R2-2, QA Q2-1, PR #3811): an unpinned ``..`` is ranked by
# the landing its missing and re-linked parts produce — every missing component
# a directory, every component an earlier ``ln -s``/``cp -s`` re-linked to a
# LITERAL target followed to that target. Relaxing an own-pad lexical landing to
# HIGH must not hide a ``..`` that crosses a link AFTER the missing part, nor a
# re-link into the state root or another session.


@pytest.fixture
def slk(lnk: Path) -> Path:
    """``<pad>/slk -> <state>/secrets``, planted before the command."""
    (lnk / "slk").symlink_to(lnk.parent.parent.parent / "secrets")
    return lnk


_STATE = "~/.local-operator"


@pytest.mark.parametrize(
    ("command", "lands"),
    [
        # The missing part becomes a directory; ``slk`` then leaves the pad.
        ("mkdir -p {pad}/n && rm -f {pad}/n/../slk/../config.yml", "config.yml"),
        ("rm -f {pad}/nope/../slk/../config.yml", "config.yml"),
        # A literal re-link: the kernel follows the NEW link.
        (f"ln -sfn {_STATE}/secrets {{pad}}/lnk && rm -f {{pad}}/lnk/../config.yml", "config.yml"),
        (f"ln -sfn {_STATE}/secrets {{pad}}/lnk; echo x > {{pad}}/lnk/../config.yml", "config.yml"),
        (f"ln -sfn {_STATE}/secrets {{pad}}/new && rm -f {{pad}}/new/../config.yml", "config.yml"),
        (f"cp -s {_STATE}/secrets {{pad}}/new && rm -f {{pad}}/new/../config.yml", "config.yml"),
        ("cd {pad} && ln -sfn ../../../secrets rel && rm -f rel/../config.yml", "config.yml"),
        # A re-link into ANOTHER session's directory forges its ask store.
        (
            f"ln -sfn {_STATE}/sessions/other/sub {{pad}}/l && echo x >> {{pad}}/l/../asks.jsonl",
            "sessions/other/asks.jsonl",
        ),
        (
            f"ln -sfn {_STATE}/sessions/other/sub {{pad}}/l && rm -f {{pad}}/l/../asks.jsonl",
            "sessions/other/asks.jsonl",
        ),
    ],
)
def test_unpinned_dotdot_is_ranked_by_its_missing_and_relinked_landing(
    home: Path, slk: Path, command: str, lands: str
) -> None:
    verdict = _classify(home, command.format(pad=slk))
    assert verdict.risk is Risk.CRITICAL, verdict.targets
    # The judge/refusal names the file the kernel acts on, not ``<pad>/...``.
    landing = _real(home / ".local-operator") + "/" + lands
    assert any(t.target.path == landing for t in verdict.targets), verdict.targets


@pytest.mark.parametrize(
    "command",
    [
        # A literal re-link that stays in the own pad: HIGH, never LOW.
        "ln -sfn {pad}/build/deep {pad}/lnk && rm -f {pad}/lnk/../x",
        # Target unknowable (a variable) through a link on disk now: HIGH.
        "L=$(cat f); ln -sfn $L {pad}/lnk && rm -f {pad}/lnk/../x",
        # A later unknowable re-link replaces an earlier literal one.
        "ln -sfn {pad}/slk {pad}/lnk && rm -rf {pad}/lnk && mv ~/w/other/sl {pad}/lnk"
        " && rm -f {pad}/lnk/../x",
    ],
)
def test_unpinned_dotdot_landing_in_own_pad_is_high(home: Path, slk: Path, command: str) -> None:
    verdict = _classify(home, command.format(pad=slk))
    assert verdict.risk is Risk.HIGH, verdict.targets


@pytest.mark.parametrize(
    "command",
    [
        f"ln -sfn {_STATE} {{pad}}/lnk && cd -P {{pad}}/lnk && rm -f config.yml",
        f"ln -sfn {_STATE}/secrets {{pad}}/lnk && cd -P {{pad}}/lnk && rm -f ../config.yml",
        f"ln -sfn {_STATE} {{pad}}/lnk && set -P && cd {{pad}}/lnk && rm -f config.yml",
        f"set -o physical; ln -sfn {_STATE} {{pad}}/lnk; cd {{pad}}/lnk; rm -f config.yml",
    ],
)
def test_physical_cd_into_a_component_relinked_earlier_is_never_low(
    home: Path, lnk: Path, command: str
) -> None:
    # Review round 2 (pre-existing): ``cd -P`` resolved the CHECK-time link.
    verdict = _classify(home, command.format(pad=lnk))
    assert verdict.risk is not Risk.LOW, verdict.targets


# Review round 3 (R3-1, PR #3811): a link this command made and then MOVED or
# COPIED as a link (``mv``, ``cp -a``/``-P``) keeps its target, so a ``..``
# through the destination lands where the source pointed. Dropping the target
# at the destination read the missing destination as a directory, and the
# own-pad relaxation ranked the lexical ``<pad>/config.yml`` HIGH — below main,
# which ranks every such shape CRITICAL.
@pytest.mark.parametrize(
    ("command", "lands"),
    [
        (
            f"ln -sfn {_STATE}/secrets {{pad}}/a && mv {{pad}}/a {{pad}}/b"
            " && rm -f {pad}/b/../config.yml",
            "config.yml",
        ),
        (
            f"ln -sfn {_STATE}/secrets {{pad}}/a && cp -P {{pad}}/a {{pad}}/b"
            " && rm -f {pad}/b/../config.yml",
            "config.yml",
        ),
        (
            f"ln -sfn {_STATE}/secrets {{pad}}/a && cp -a {{pad}}/a {{pad}}/b"
            " && rm -f {pad}/b/../config.yml",
            "config.yml",
        ),
        # Moving INTO the link's directory leaves ``a`` the link it was.
        (
            f"ln -sfn {_STATE}/secrets {{pad}}/a && mv {{pad}}/tmpd {{pad}}/a"
            " && rm -f {pad}/a/../config.yml",
            "config.yml",
        ),
        (
            f"ln -sfn {_STATE}/sessions/other/sub {{pad}}/a && mv {{pad}}/a {{pad}}/b"
            " && echo x >> {pad}/b/../asks.jsonl",
            "sessions/other/asks.jsonl",
        ),
        # ``-t DIR``: the link lands as ``DIR/<name>``.
        (
            f"mkdir -p {{pad}}/x && ln -sfn {_STATE}/secrets {{pad}}/x/a"
            " && mv -t {pad} {pad}/x/a && rm -f {pad}/a/../config.yml",
            "config.yml",
        ),
        # A link already on disk carries its ``readlink`` target.
        ("mv {pad}/slk {pad}/b && rm -f {pad}/b/../config.yml", "config.yml"),
    ],
)
def test_moved_or_copied_link_keeps_its_target(
    home: Path, slk: Path, command: str, lands: str
) -> None:
    verdict = _classify(home, command.format(pad=slk))
    assert verdict.risk is Risk.CRITICAL, verdict.targets
    landing = _real(home / ".local-operator") + "/" + lands
    assert any(t.target.path == landing for t in verdict.targets), verdict.targets


@pytest.mark.parametrize(
    "command",
    [
        'mv "$(cat f)" {pad}/b && rm -f {pad}/b/../config.yml',
        "cp -a {pad}/nope {pad}/b && rm -f {pad}/b/../config.yml",
        'ln -sfn "$(cat f)" {pad}/a && mv {pad}/a {pad}/b && rm -f {pad}/b/../config.yml',
        # Was HIGH (own-pad relaxation) until round 4; main ranks it CRITICAL.
        "mv ~/w/other/sl {pad}/fresh && rm -f {pad}/fresh/../x",
        # Round-2 out-of-scope shape: a program the resolver does not model
        # (``rsync -l``) re-links the component; main ranks it CRITICAL.
        f"ln -sfn {_STATE}/secrets {{pad}}/a && rsync -l {{pad}}/a {{pad}}/b"
        " && rm -f {pad}/b/../config.yml",
    ],
)
def test_move_of_an_unknowable_source_keeps_mains_critical(
    home: Path, lnk: Path, command: str
) -> None:
    # The destination may become any link, and its target cannot be known:
    # the lexical landing under session state keeps main's CRITICAL rule.
    verdict = _classify(home, command.format(pad=lnk))
    assert verdict.risk is Risk.CRITICAL, verdict.targets


# Review round 3 (R3-2, PR #3811): ``l_k -> l_{k+1}/../l_{k+1}`` made the
# re-link walk resolve each sibling again at every level — 2^N — and the walker
# ran it for every named path, before and outside the judge timeout. The kernel
# gives up after 32/40 links in total, so the walk now shares one hop budget
# and fails closed (unknown) when it runs out.
@pytest.mark.parametrize(
    "command",
    [
        "rm -f {pad}/c/l0/x",
        "rm -f {pad}/nope/../c/l0/x",
        "ln -sfn {pad}/build {pad}/q && rm -f {pad}/nope/../c/l0/x",
        "ln -sfn {pad}/c/l0 {pad}/q && rm -f {pad}/q/../x",
    ],
)
def test_nested_link_chain_resolves_in_bounded_time(home: Path, command: str) -> None:
    import time

    pad = _pad(home)
    chain = pad / "c"
    depth = 24
    (chain / f"l{depth}").mkdir(parents=True)
    for k in range(depth):
        (chain / f"l{k}").symlink_to(f"l{k + 1}/../l{k + 1}")
    start = time.monotonic()
    verdict = _classify(home, command.format(pad=pad))
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, elapsed
    if "/../" in command:
        # Unpinned and missing at check time, landing lexically in session
        # state: main's CRITICAL, whether or not the walk ran out of hops.
        assert verdict.risk is Risk.CRITICAL, verdict.targets


@pytest.mark.parametrize(
    "command",
    [
        f"ln -sfn {_STATE} {{pad}}/lnk && cd -P {{pad}} && cd -P lnk && rm -f config.yml",
        f"ln -sfn {_STATE}/secrets {{pad}}/lnk && cd -P {{pad}} && rm -f lnk/../config.yml",
    ],
)
def test_physical_cwd_still_sees_a_relinked_component(home: Path, lnk: Path, command: str) -> None:
    # Review R3-3 (PR #3811): after ``cd -P`` the cwd is the realpath
    # (``/private/var/...`` on macOS), which never prefix-matched the lexical
    # spelling the ``ln`` recorded, so the re-link was missed and this was LOW.
    # The pad is spelled through a link here (as ``/var`` is on macOS), since
    # pytest's ``tmp_path`` is already physical.
    alias = home.parent / "alias"
    alias.symlink_to(home)
    spelled = alias / lnk.relative_to(home)
    verdict = _classify(home, command.format(pad=spelled))
    assert verdict.risk is not Risk.LOW, verdict.targets


# Review round 4 (R4-1, PR #3811): moving or copying a DIRECTORY that contains
# a link (``<pad>/tree/sub -> <state>/secrets`` on disk) keeps that link, but
# its new path is missing at check time, so ``sub`` read as a fresh directory
# and the own-pad relaxation ranked ``<pad>/b/config.yml`` HIGH. With the
# relaxation removed, an unpinned ``..`` landing lexically in session state is
# main's CRITICAL again, own pad included.
@pytest.mark.parametrize(
    "command",
    [
        "mv {pad}/tree {pad}/b && rm -f {pad}/b/sub/../config.yml",
        "cp -a {pad}/tree {pad}/b && rm -f {pad}/b/sub/../config.yml",
        "mv {pad}/tree {pad}/tmpd && rm -f {pad}/tmpd/tree/sub/../config.yml",
    ],
)
def test_moved_tree_containing_a_link_keeps_mains_critical(
    home: Path, lnk: Path, command: str
) -> None:
    (lnk / "tree").mkdir()
    (lnk / "tree" / "sub").symlink_to(home / ".local-operator" / "secrets")
    verdict = _classify(home, command.format(pad=lnk))
    assert verdict.risk is Risk.CRITICAL, verdict.targets
