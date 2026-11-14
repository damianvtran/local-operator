"""Regression tests for two guardrail false positives the operator reported (2026-11-09).

1. ``git -C <repo> worktree remove <exact path>`` on CLEAN linked worktrees was
   refused as a "git repository root" delete. Root causes in
   ``provenance.worktree_state``: under disk pressure its git probes outran the
   10 s bound (measured up to 34 s) and returned ``unknown``; and ignored
   regenerable output the allowlist did not name (``.DS_Store``,
   ``*.egg-info``, ``.cache``, ``_*.bundle.mjs``) made a clean tree "dirty".
2. Refusal wording ("final; never reword or retry") was read as forbidding a
   NEW artifact from fresh work in the agent's own worktree. The texts now say a
   refusal is final for THAT action only.

Every git call runs against a throwaway repository under ``tmp_path`` with the
system/global git config blanked (see the ``home`` fixture) — never against a
real checkout.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_operator.guardrails import provenance as prov
from local_operator.guardrails.check import (
    REFUSAL_SCOPE,
    CallContext,
    GuardSettings,
    _critical_text,
    review_text,
    rule_block_text,
)
from local_operator.guardrails.classify import Risk, classify_bash
from local_operator.guardrails.preview import preview_tool_call
from local_operator.guardrails.rules import Rule

SID = "bbbbbbbbbbbb"
# The real PR-lookup helpers, kept before the ``home`` fixture stubs them.
_REAL_GITHUB_REPO = prov._github_repo
_REAL_GH_OPEN_PRS = prov._gh_open_prs
_REAL_GH_PRS = prov._gh_prs
_REAL_GLAB_MRS = prov._glab_mrs


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    state = home / ".local-operator"
    (state / "sessions" / SID / "scratchpad").mkdir(parents=True)
    (state / "config.yml").write_text("version: 1\n")
    # Synthetic repos have no live cwd; process inventory has dedicated controls
    # below and must not make git regressions depend on fleet lsof latency.
    monkeypatch.setattr(prov, "_live_cwd_inside", lambda _: False)
    # Synthetic repos have no GitHub remote either: by default every worktree is
    # on GitHub with no open PR. The PR lookup has its own tests below.
    monkeypatch.setattr(prov, "_github_repo", lambda _: "o/r")
    monkeypatch.setattr(prov, "_gh_prs", lambda repo, branch: [])
    monkeypatch.setattr(prov, "_cwd_inventory_cache", None)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))
    for key in [k for k in os.environ if k.startswith(("CMUX_", "GIT_"))]:
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
    ).stdout


def _verdict(home: Path, command: str):
    return classify_bash(command, cwd=str(home), env=_env(home), scratchpad=None, session_id=SID)


@pytest.fixture
def repo(home: Path) -> Path:
    repo = home / "proj"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    (repo / ".gitignore").write_text(
        ".DS_Store\n*.egg-info/\n.cache/\n._*\nscripts/_*.bundle.mjs\n.env\n"
    )
    (repo / "f").write_text("x")
    _git("add", ".", cwd=repo)
    _git("commit", "-qm", "init", cwd=repo)
    return repo


def _add(repo: Path, where: Path, branch: str) -> Path:
    _git("-C", str(repo), "worktree", "add", "-q", "-b", branch, str(where), cwd=repo)
    return where


def test_sibling_worktree_outside_repo_with_git_dash_c_ranks_low(home: Path, repo: Path) -> None:
    """``~/lo-i18n-es`` shape: a sibling at $HOME's top level, removed via ``git -C``."""
    sib = _add(repo, home / "lo-sibling", "feat/sib")
    assert prov.worktree_state(str(sib)) == "clean"
    ranked = _verdict(home, f"git -C {repo} worktree remove {sib}")
    assert ranked.risk is Risk.LOW
    assert ranked.targets[0].reason.startswith("a clean linked git worktree (branch feat/sib;")
    # Negative controls on the same tree: rm -rf is not ``git worktree remove``.
    assert _verdict(home, f"rm -rf {sib}").risk is Risk.HIGH
    assert _verdict(home, f"git -C {repo} worktree remove {repo}").risk is Risk.HIGH  # primary


def _pad(home: Path) -> Path:
    return home / ".local-operator" / "sessions" / SID / "scratchpad"


def _pad_verdict(home: Path, command: str):
    """Classify with the session's own scratchpad known.

    Inside its own pad ``_high_reason`` names no repo-root reason, so a linked
    worktree there reaches the final ordinary-path LOW — the Q3-OBS shape.
    """
    return classify_bash(
        command, cwd=str(home), env=_env(home), scratchpad=str(_pad(home)), session_id=SID
    )


@pytest.mark.parametrize(
    "flags", ["", "--force ", "-f -f "], ids=["plain", "force", "double-force"]
)
def test_dirty_sibling_worktree_without_pr_ranks_high(home: Path, repo: Path, flags: str) -> None:
    """QA round 3 (Q3-OBS): a dirty sibling with no open PR used to rank LOW.

    The dirty note was appended and then discarded by the ordinary-path LOW;
    with ``--force`` git deletes the uncommitted work.
    """
    sib = _add(repo, _pad(home) / "lo-dirty", "feat/dirty")
    (sib / "wip.txt").write_text("uncommitted")
    assert prov.worktree_state(str(sib)) == "dirty"
    ranked = _pad_verdict(home, f"git -C {repo} worktree remove {flags}{sib}")
    assert ranked.risk is Risk.HIGH
    assert ranked.targets[0].reason.startswith(
        "a git repository root; linked worktree state: dirty; branch feat/dirty;"
    )


def test_locked_sibling_worktree_without_pr_ranks_high(home: Path, repo: Path) -> None:
    """A non-initializing lock outside any repo root is HIGH with the state named."""
    sib = _add(repo, _pad(home) / "lo-locked", "feat/locked")
    _git("-C", str(repo), "worktree", "lock", "--reason", "on a usb disk", str(sib), cwd=repo)
    assert prov.worktree_state(str(sib)) == "locked"
    for flags in ("", "-f -f "):
        ranked = _pad_verdict(home, f"git -C {repo} worktree remove {flags}{sib}")
        assert ranked.risk is Risk.HIGH, flags
        assert "linked worktree state: locked" in ranked.targets[0].reason


def test_clean_sibling_worktree_without_pr_stays_low(home: Path, repo: Path) -> None:
    """Control for Q3-OBS: the fix must not lift a clean, PR-less sibling."""
    sib = _add(repo, _pad(home) / "lo-clean", "feat/clean")
    ranked = _pad_verdict(home, f"git -C {repo} worktree remove {sib}")
    assert ranked.risk is Risk.LOW
    assert ranked.targets[0].reason.startswith("a clean linked git worktree (branch feat/clean;")


def test_regenerable_ignored_output_keeps_a_worktree_clean(home: Path, repo: Path) -> None:
    """The entries that made the operator's real clean worktrees "dirty"."""
    wt = _add(repo, repo / ".worktrees" / "wt", "wt")
    (wt / ".DS_Store").write_text("finder")
    (wt / "pkg.egg-info").mkdir()
    (wt / "pkg.egg-info" / "PKG-INFO").write_text("meta")
    (wt / ".cache").mkdir()
    (wt / ".cache" / "x").write_text("c")
    (wt / "._f").write_text("appledouble")
    (wt / "scripts").mkdir()
    (wt / "scripts" / "_table-pager-range-22269.bundle.mjs").write_text("bundle")
    assert prov.worktree_state(str(wt)) == "clean"
    assert _verdict(home, f"git -C {repo} worktree remove {wt}").risk is Risk.LOW
    # Still dirty: a non-regenerable ignored file, and an untracked file.
    (wt / ".env").write_text("SECRET=1")
    assert prov.worktree_state(str(wt)) == "dirty"
    assert _verdict(home, f"git -C {repo} worktree remove {wt}").risk is Risk.HIGH
    (wt / ".env").unlink()
    (wt / "notes.txt").write_text("work")
    assert prov.worktree_state(str(wt)) == "dirty"
    assert _verdict(home, f"git worktree remove --force {wt}").risk is Risk.HIGH


def test_collapsed_ignored_directory_is_judged_by_its_content(home: Path, repo: Path) -> None:
    """``git status --ignored`` reports a dir holding only ignored files as the dir.

    ``.cache/typecheck/`` (local-operator-ui) and an untracked ``scripts/`` with
    only ``_*.bundle.mjs`` surface as ``!! .cache/`` / ``!! scripts/``; a
    collapsed dir with a non-regenerable file inside is still dirty.
    """
    wt = _add(repo, repo / ".worktrees" / "col", "col")
    (wt / "scripts").mkdir()
    (wt / "scripts" / "_a.bundle.mjs").write_text("b")
    (wt / "scripts" / ".cache").mkdir()
    (wt / "scripts" / ".cache" / "anything").write_text("c")
    # Review m1: a build-named dir INSIDE a collapsed dir is walked, not trusted.
    assert prov.worktree_state(str(wt)) == "dirty"
    (wt / "scripts" / ".cache" / "anything").unlink()
    (wt / "scripts" / ".cache" / "x.pyc").write_text("c")
    assert prov.worktree_state(str(wt)) == "clean"
    (wt / "scripts" / "._keep").write_text("appledouble")
    assert prov.worktree_state(str(wt)) == "clean"
    with (wt / ".gitignore").open("a") as fh:  # a committed ignore rule
        fh.write("private/\n")
    _git("commit", "-qam", "ignore private", cwd=wt)
    (wt / "private").mkdir()
    (wt / "private" / ".DS_Store").write_text("finder")
    assert prov.worktree_state(str(wt)) == "clean"
    (wt / "private" / "notes.md").write_text("only copy")
    assert prov.worktree_state(str(wt)) == "dirty"


def test_unpushed_or_squash_merged_branch_is_clean_and_keeps_its_commits(
    home: Path, repo: Path
) -> None:
    """``git worktree remove`` never deletes the branch: its commits survive.

    The branch here is on no remote and is not an ancestor of ``main`` (the
    squash-merged shape). Removing the worktree for real must leave the
    branch and its commit intact — the reason no reachability test is asked.
    """
    wt = _add(repo, repo / ".worktrees" / "sq", "feat/squashed")
    (wt / "g").write_text("y")
    _git("add", "g", cwd=wt)
    _git("commit", "-qm", "branch-only work", cwd=wt)
    tip = _git("rev-parse", "HEAD", cwd=wt).strip()
    assert prov.worktree_state(str(wt)) == "clean"
    _git("-C", str(repo), "worktree", "remove", str(wt), cwd=repo)
    assert not wt.exists()
    assert _git("rev-parse", "feat/squashed", cwd=repo).strip() == tip


def test_locked_worktree_is_not_clean(home: Path, repo: Path) -> None:
    """``git worktree add`` leaves ``locked: initializing`` when killed mid-checkout."""
    wt = _add(repo, repo / ".worktrees" / "lk", "lk")
    _git("-C", str(repo), "worktree", "lock", "--reason", "initializing", str(wt), cwd=repo)
    assert prov.worktree_state(str(wt)) == "locked"
    ranked = _verdict(home, f"git -C {repo} worktree remove {wt}")
    assert ranked.risk is Risk.HIGH
    assert "linked worktree state: locked" in ranked.targets[0].reason  # actionable
    _git("-C", str(repo), "worktree", "unlock", str(wt), cwd=repo)
    assert prov.worktree_state(str(wt)) == "clean"


def test_relative_registration_paths_are_resolved(home: Path, repo: Path) -> None:
    wt = repo / ".worktrees" / "rel"
    _git(
        "-c",
        "worktree.useRelativePaths=true",
        "-C",
        str(repo),
        "worktree",
        "add",
        "-q",
        "-b",
        "rel",
        str(wt),
        cwd=repo,
    )
    assert prov.worktree_state(str(wt)) == "clean"


def test_status_probe_gets_the_measured_bound_and_a_timeout_fails_closed(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under disk pressure the status probe measured up to 34 s (n=6)."""
    assert prov._WORKTREE_STATUS_TIMEOUT_S > 34.3 > prov._GIT_TIMEOUT_S
    wt = _add(repo, repo / ".worktrees" / "slow", "slow")
    real_run = subprocess.run
    seen: list[float] = []

    def fake_run(cmd, *a, **kw):
        if "status" in cmd:
            seen.append(kw["timeout"])
            raise subprocess.TimeoutExpired(cmd, kw["timeout"])
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(prov.subprocess, "run", fake_run)
    assert prov.worktree_state(str(wt)) == "unknown"
    assert seen == [prov._WORKTREE_STATUS_TIMEOUT_S]
    assert _verdict(home, f"git -C {repo} worktree remove {wt}").risk is Risk.HIGH


# -- refusal scope wording ----------------------------------------------------------


def _scope_phrases(text: str) -> None:
    flat = " ".join(text.split())
    assert "final for THIS action" in flat or "final for THAT action" in flat
    assert "same effect on the same targets" in flat
    assert "fresh work" in flat.lower() and "needs no approval" in flat


def test_every_refusal_text_scopes_finality_to_that_action() -> None:
    _scope_phrases(REFUSAL_SCOPE)
    rule = Rule(name="r", scope="tool:*", action="block", body="no")
    for text in (
        review_text("dangerous-delete", "a git repository root", "~/x"),
        review_text("my-rule", "why", "t"),
        rule_block_text(rule),
    ):
        assert text.startswith("BLOCKED by guardrail")
        assert text.endswith(REFUSAL_SCOPE)


def test_critical_refusal_scopes_finality(home: Path) -> None:
    verdict = _verdict(home, f"rm -rf {home}")
    assert verdict.risk is Risk.CRITICAL
    assert _critical_text(verdict).endswith(REFUSAL_SCOPE)


def test_system_prompt_and_guide_scope_finality_and_keep_import_a_reroute() -> None:
    """The prompt line stays terse (CI's context-budget sits within tokens of its
    ceiling); the guide and the refusal text carry the full scope."""
    from local_operator.prompts_api import render_template

    prompt = " ".join(render_template("system.md", {}).split())
    assert (
        "`BLOCKED by guardrail` is final for that effect on those targets, by any "
        "route: report it. Fresh work of your own is not a reroute."
    ) in prompt
    root = Path(prov.__file__).resolve().parents[1]
    guide = " ".join((root / "guides" / "guardrails" / "GUIDE.md").read_text().split())
    _scope_phrases(guide)
    assert "BLOCKED by guardrail" in guide
    assert "importing the blocked material" in guide  # still a reroute


def test_fresh_receipt_write_into_own_worktree_is_not_refused(home: Path, repo: Path) -> None:
    """Positive control: PR #1813's auditor case — no guard refuses or judges it."""
    wt = _add(repo, repo / ".worktrees" / "audit", "audit")
    receipts = wt / "i18n" / "receipts"
    receipts.mkdir(parents=True)
    ctx = CallContext(
        cwd=str(wt),
        env=_env(home),
        scratchpad=str(home / ".local-operator" / "sessions" / SID / "scratchpad"),
        session_id=SID,
    )
    settings = GuardSettings(enabled=True, rules_dirs=())
    target = receipts / "es-2026-11-09.json"
    calls = [
        ("write", {"path": str(target), "content": "{}"}),
        ("edit", {"path": str(target), "old_text": "{}", "new_text": "{ }"}),
        ("bash", {"command": f"printf '{{}}' > {target}"}),
        ("bash", {"command": "cat > i18n/receipts/es.json <<'EOF'\n{}\nEOF"}),
    ]
    for tool, args in calls:
        report = preview_tool_call(tool, args, ctx, settings=settings)
        assert report.action == "allow", (tool, args, report.action, report.text)
        assert report.builtin_effect != "judge", (tool, args)


@pytest.mark.parametrize("flags", ["-f -f", "-ff", "--force --force"])
def test_stale_initializing_double_force_only(home: Path, repo: Path, flags: str) -> None:
    wt = _add(repo, repo / ".worktrees" / "stale", "stale")
    _git("worktree", "lock", "--reason", "initializing", str(wt), cwd=repo)
    lock = repo / ".git" / "worktrees" / "stale" / "locked"
    import time

    old = time.time() - 7200
    os.utime(lock, (old, old))
    assert prov.worktree_state(str(wt)) == "stale-initializing"
    assert _verdict(home, f"git -C {repo} worktree remove {flags} {wt}").risk is Risk.LOW
    for denied in ("", "-f", "-fff"):
        assert _verdict(home, f"git -C {repo} worktree remove {denied} {wt}").risk is Risk.HIGH
    (wt / "untracked").write_text("only copy")
    assert _verdict(home, f"git -C {repo} worktree remove {flags} {wt}").risk is Risk.HIGH
    (wt / "untracked").unlink()
    lock.write_text("user says keep")
    assert _verdict(home, f"git -C {repo} worktree remove {flags} {wt}").risk is Risk.HIGH


@pytest.mark.parametrize("inventory", [True, None])
def test_live_or_unknown_process_cwd_blocks_a_clean_worktree(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch, inventory: bool | None
) -> None:
    wt = _add(repo, repo / ".worktrees" / "busy", "busy")
    monkeypatch.setattr(prov, "_live_cwd_inside", lambda _: inventory)
    assert prov.worktree_state(str(wt)) == "busy"
    assert _verdict(home, f"git -C {repo} worktree remove {wt}").risk is Risk.HIGH


def test_process_inventory_timeout_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prov, "_cwd_inventory_cache", None)

    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == prov._CWD_INVENTORY_TIMEOUT_S == 20.0
        raise subprocess.TimeoutExpired(args[0], 20)

    monkeypatch.setattr(prov.subprocess, "run", timeout)
    # Exercise macOS branch independently of the CI runner's /proc.
    real_isdir = os.path.isdir
    monkeypatch.setattr(
        prov.os.path, "isdir", lambda p: False if p == "/proc/self" else real_isdir(p)
    )
    assert prov._live_cwd_inside("/synthetic/tree") is None


# -- round-1 remediation (PR #3700) -------------------------------------------------


def test_untracked_files_show_despite_show_untracked_files_no(home: Path, repo: Path) -> None:
    """Review M1: ``status.showUntrackedFiles=no`` must not hide work from the probe."""
    wt = _add(repo, repo / ".worktrees" / "cfg", "cfg")
    _git("config", "status.showUntrackedFiles", "no", cwd=repo)
    assert prov.worktree_state(str(wt)) == "clean"  # control: nothing there yet
    (wt / "notes.txt").write_text("only copy")
    assert prov.worktree_state(str(wt)) == "dirty"
    assert _verdict(home, f"git -C {repo} worktree remove {wt}").risk is Risk.HIGH
    (wt / "notes.txt").unlink()
    (wt / ".env").write_text("SECRET=1")  # ignored, not regenerable
    assert prov.worktree_state(str(wt)) == "dirty"


def test_build_named_dir_under_arbitrary_ignored_parent_is_walked(home: Path, repo: Path) -> None:
    """Review m1: ``work/build/thesis.md`` under an ignored ``work/`` is real work."""
    wt = _add(repo, repo / ".worktrees" / "m1", "m1")
    with (wt / ".gitignore").open("a") as fh:
        fh.write("work/\n")
    _git("commit", "-qam", "ignore work", cwd=wt)
    (wt / "work" / "build").mkdir(parents=True)
    (wt / "work" / "build" / "thesis.md").write_text("only copy")
    assert prov.worktree_state(str(wt)) == "dirty"
    (wt / "work" / "build" / "thesis.md").unlink()
    (wt / "work" / "pkg.egg-info").mkdir()  # a glob-matched dir is accepted whole
    (wt / "work" / "pkg.egg-info" / "PKG-INFO").write_text("meta")
    assert prov.worktree_state(str(wt)) == "clean"


def test_cwd_inventory_is_bounded_and_cached_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """QA Q-1: one inventory serves 30 s; a failure is never cached."""
    monkeypatch.setattr(prov, "_cwd_inventory_cache", None)
    real_isdir = os.path.isdir
    monkeypatch.setattr(
        prov.os.path, "isdir", lambda p: False if p == "/proc/self" else real_isdir(p)
    )
    calls: list[float] = []
    outcome = {"fail": True}

    def run(*args, **kwargs):
        calls.append(kwargs["timeout"])
        if outcome["fail"]:
            raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
        return subprocess.CompletedProcess(args[0], 0, "p1\nn/synthetic/tree/sub\n", "")

    clock = {"t": 1000.0}
    monkeypatch.setattr(prov.subprocess, "run", run)
    monkeypatch.setattr(prov.time, "monotonic", lambda: clock["t"])
    assert prov._live_cwd_inside("/synthetic/tree") is None  # timeout: fail closed
    outcome["fail"] = False
    assert prov._live_cwd_inside("/synthetic/tree") is True  # failure was not cached
    assert prov._live_cwd_inside("/synthetic/other") is False
    clock["t"] += prov._CWD_INVENTORY_TTL_S - 1
    assert prov._live_cwd_inside("/synthetic/tree") is True
    assert calls == [20.0, 20.0]  # two probes for four questions
    clock["t"] += 2
    prov._live_cwd_inside("/synthetic/tree")
    assert len(calls) == 3  # expired: re-read


def test_open_pr_refuses_any_linked_worktree_outright(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operator incident: a clean worktree whose branch had an open PR was removed."""
    seen: list[tuple[str, str]] = []

    def lookup(repo_name: str, branch: str) -> list[prov.ForgeChange]:
        seen.append((repo_name, branch))
        return [prov.ForgeChange(1828, "open", "2026-11-09")]

    monkeypatch.setattr(prov, "_gh_prs", lookup)
    wt = _add(repo, repo / ".worktrees" / "pp", "feat/profile-proposals-agents")
    assert prov.worktree_state(str(wt)) == "clean"
    verdict = _verdict(home, f"git -C {repo} worktree remove {wt}")
    assert verdict.risk is Risk.CRITICAL
    assert seen == [("o/r", "feat/profile-proposals-agents")]
    text = _critical_text(verdict)
    assert (
        "branch feat/profile-proposals-agents has open PR #1828; finish or close it first" in text
    )
    assert "no other running session owns it" in text  # owner state shown too
    (wt / "notes.txt").write_text("dirty")  # still refused outright when dirty
    assert _verdict(home, f"git -C {repo} worktree remove --force {wt}").risk is Risk.CRITICAL


def test_open_pr_refusal_is_not_lifted_by_an_answered_ask(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CRITICAL never reaches the ask/judge route: an earlier approval cannot cover it."""
    from local_operator.guardrails.check import BUILTIN_REFUSE_CRITICAL, builtin_branch

    monkeypatch.setattr(prov, "_gh_prs", lambda r, b: [prov.ForgeChange(7, "open", "")])
    wt = _add(repo, repo / ".worktrees" / "asked", "asked")
    verdict = _verdict(home, f"git -C {repo} worktree remove {wt}")
    assert builtin_branch(verdict, critical_override=False) == BUILTIN_REFUSE_CRITICAL


@pytest.mark.parametrize(
    ("repo_name", "prs", "detail"),
    [
        (None, [], "no GitHub or GitLab remote"),
        ("o/r", None, "gh lookup on o/r failed or timed out"),
    ],
)
def test_unknown_pr_state_goes_to_the_judge(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch, repo_name, prs, detail
) -> None:
    monkeypatch.setattr(prov, "_github_repo", lambda _: repo_name)
    monkeypatch.setattr(prov, "_gh_prs", lambda r, b: prs)
    wt = _add(repo, repo / ".worktrees" / "unk", "unk")
    verdict = _verdict(home, f"git -C {repo} worktree remove {wt}")
    assert verdict.risk is Risk.HIGH
    assert f"PR state unknown ({detail})" in verdict.targets[0].reason


def test_running_owner_session_goes_to_the_judge(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Owner = ledger creator of the ``worktree add`` or a live session cwd inside."""
    from types import SimpleNamespace

    from local_operator.session.runtime import registry

    wt = _add(repo, repo / ".worktrees" / "own", "own")
    real = os.path.realpath(wt)
    other = "cccccccccccc"
    prov.write_records(
        [(real, "worktree")], owners=[other], creator=other, command="git worktree add"
    )
    live: list[tuple[SimpleNamespace, str]] = [
        (SimpleNamespace(session_id=other, cwd=str(home)), "live")
    ]
    monkeypatch.setattr(registry, "scan", lambda *a, **k: live)
    verdict = _verdict(home, f"git -C {repo} worktree remove {wt}")
    assert verdict.risk is Risk.HIGH
    assert f"owning session running: {other}" in verdict.targets[0].reason
    # Not running any more → idle again (other checks pass) → LOW.
    live[:] = [(SimpleNamespace(session_id=other, cwd=str(home)), "stale")]
    assert _verdict(home, f"git -C {repo} worktree remove {wt}").risk is Risk.LOW
    # A live session whose recorded cwd is inside the tree also owns it...
    live[:] = [(SimpleNamespace(session_id="dddddddddddd", cwd=str(wt / "src")), "wedged")]
    assert _verdict(home, f"git -C {repo} worktree remove {wt}").risk is Risk.HIGH
    # ...but the caller itself is never its own conflict.
    live[:] = [(SimpleNamespace(session_id=SID, cwd=real), "live")]
    assert _verdict(home, f"git -C {repo} worktree remove {wt}").risk is Risk.LOW


def test_pr_lookup_reads_the_github_remote_and_never_prompts(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _git("remote", "add", "fork", "git@github.com:someone/fork.git", cwd=repo)
    origin = "https://github.com/damianvtran/local-operator-ui.git"
    _git("remote", "add", "origin", origin, cwd=repo)
    assert _REAL_GITHUB_REPO(str(repo)) == "damianvtran/local-operator-ui"  # origin first
    _git("remote", "set-url", "origin", "git@gitlab.com:minerva/core.git", cwd=repo)
    assert _REAL_GITHUB_REPO(str(repo)) == "someone/fork"
    _git("remote", "remove", "fork", cwd=repo)
    assert _REAL_GITHUB_REPO(str(repo)) is None
    seen: dict[str, Any] = {}

    def run(argv, **kwargs):
        prompt = kwargs["env"]["GH_PROMPT_DISABLED"]
        seen.update(argv=argv, timeout=kwargs["timeout"], prompt=prompt)
        return subprocess.CompletedProcess(argv, 0, '[{"number": 1828}]', "")

    monkeypatch.setattr(prov.subprocess, "run", run)
    assert _REAL_GH_OPEN_PRS("o/r", "feat/x") == [1828]
    assert seen["argv"][:3] == ["gh", "pr", "list"] and "--head" in seen["argv"]
    assert seen["timeout"] == prov._GH_TIMEOUT_S == 8.0 and seen["prompt"] == "1"

    def timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(prov.subprocess, "run", timeout)
    assert _REAL_GH_OPEN_PRS("o/r", "feat/x") is None


def _session_dir(home: Path) -> str:
    return str(home / ".local-operator" / "sessions")


def test_readers_over_session_state_are_not_destructive(home: Path) -> None:
    """Operator case (A): the exact loop, cwd under the sessions directory."""
    (home / ".local-operator" / "sessions" / "cccccccccccc").mkdir()
    cwd = _session_dir(home)
    loop = 'for s in */; do [ -d "$s" ] && echo "$s"; done | sort | uniq'
    other = f"{cwd}/cccccccccccc/x"
    for command in (
        loop,
        f"cat {other} | sort | uniq -c | head -3",
        f"shasum {other}; md5 -q {other}; file {other}; stat {other}; wc -l {other}; ls {cwd}",
        f"sort -u {other} | tail -2 | grep a",
    ):
        verdict = classify_bash(command, cwd=cwd, env=_env(home), scratchpad=None, session_id=SID)
        assert verdict.risk is Risk.LOW, (command, [t.reason for t in verdict.targets])


def test_sort_output_into_session_state_is_still_refused(home: Path) -> None:
    (home / ".local-operator" / "sessions" / "cccccccccccc").mkdir()
    cwd = _session_dir(home)
    other = f"{cwd}/cccccccccccc/x"
    for command in (f"sort -o {other} {other}", f"sort --output={other} /dev/null"):
        verdict = classify_bash(
            command, cwd=str(home), env=_env(home), scratchpad=None, session_id=SID
        )
        assert verdict.risk is Risk.CRITICAL, command
    for command in (f"uniq {other} {other}.2", f"file -C -m {other}", f"sort -T {cwd} {other}"):
        verdict = classify_bash(
            command, cwd=str(home), env=_env(home), scratchpad=None, session_id=SID
        )
        assert verdict.risk is Risk.HIGH, command


@pytest.mark.parametrize(
    ("template", "risk"),
    [
        # Review round 2, M2: getopt_long accepts any unambiguous prefix, so the
        # abbreviated writing switches are the same write (measured: Apple
        # sort ``--out=`` wrote the file; ``file --comp`` wrote ``.mgc``).
        ("sort -o {other} {other}", Risk.CRITICAL),
        ("sort --output={other} /dev/null", Risk.CRITICAL),
        ("sort --out={other} /dev/null", Risk.CRITICAL),
        ("sort --outp {other} /dev/null", Risk.CRITICAL),
        ("sort --o={other} /dev/null", Risk.CRITICAL),
        ("sort --ou={other} /dev/null", Risk.CRITICAL),
        ("sort --compress-prog=sh {other}", Risk.HIGH),
        # m4: ``-T``/``--temp*`` only writes sort's own temp files — HIGH, not CRITICAL.
        ("sort --temp={cwd} {other}", Risk.HIGH),
        ("sort -T {cwd} {other}", Risk.HIGH),
        ("file --comp -m {other}", Risk.HIGH),
        # Review nit N2: shorter prefixes of ``--compile`` (``--c`` is also
        # ambiguous with ``--checking-printout``) and sort's ambiguous ``--c``
        # (``--check``/``--compress-program``) must not read as read-only.
        ("file --c -m {other}", Risk.HIGH),
        ("file --com -m {other}", Risk.HIGH),
        ("sort --c {other}", Risk.HIGH),
        # An unknown long option fails closed rather than reading as LOW.
        ("sort --no-such-option {other}", Risk.HIGH),
    ],
)
def test_abbreviated_writing_long_options_are_not_read_only(
    home: Path, template: str, risk: Risk
) -> None:
    (home / ".local-operator" / "sessions" / "cccccccccccc").mkdir()
    cwd = _session_dir(home)
    command = template.format(other=f"{cwd}/cccccccccccc/x", cwd=cwd)
    verdict = classify_bash(command, cwd=cwd, env=_env(home), scratchpad=None, session_id=SID)
    assert verdict.risk is risk, (command, [t.reason for t in verdict.targets])


def test_abbreviated_read_only_long_options_stay_low(home: Path) -> None:
    """The allowlist admits read-only abbreviations (``--rev``, ``--bri``)."""
    (home / ".local-operator" / "sessions" / "cccccccccccc").mkdir()
    cwd = _session_dir(home)
    other = f"{cwd}/cccccccccccc/x"
    for command in (
        f"sort --rev --numeric-sort --key=2 {other}",
        f"sort --uniq -- {other}",
        f"file --bri --mime-type {other}",
    ):
        verdict = classify_bash(command, cwd=cwd, env=_env(home), scratchpad=None, session_id=SID)
        assert verdict.risk is Risk.LOW, (command, [t.reason for t in verdict.targets])


def test_refusal_text_says_another_route_is_the_same_action() -> None:
    """Review m2: the anti-reroute clause is in the text read at refusal time."""
    assert "by another route is still the same action" in REFUSAL_SCOPE


# -- GitLab (Minerva) worktrees: the disk lane's positive controls --------------------


@pytest.mark.parametrize(
    ("mrs", "risk"),
    [
        ([prov.ForgeChange(42, "open", "")], Risk.CRITICAL),
        ([], Risk.LOW),
        (None, Risk.HIGH),
    ],
)
def test_gitlab_open_mr_refuses_and_no_mr_stays_low(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch, mrs, risk: Risk
) -> None:
    monkeypatch.setattr(prov, "_github_repo", _REAL_GITHUB_REPO)  # real remote parsing
    seen: list[tuple[str, str]] = []

    def lookup(project: str, branch: str):
        seen.append((project, branch))
        return mrs

    monkeypatch.setattr(prov, "_glab_mrs", lookup)
    _git("remote", "add", "origin", "git@gitlab.com:minervaai/sub/core.git", cwd=repo)
    wt = _add(repo, repo / ".worktrees" / "gl", "feat/gl")
    verdict = _verdict(home, f"git -C {repo} worktree remove {wt}")
    assert verdict.risk is risk
    assert seen == [("minervaai/sub/core", "feat/gl")]
    reason = verdict.targets[0].reason
    if mrs:
        assert "branch feat/gl has open MR !42; finish or close it first" in reason
    elif mrs == []:
        assert "no open MR" in reason
    else:
        assert "PR state unknown (glab lookup on minervaai/sub/core failed" in reason


def test_glab_lookup_parses_remotes_and_never_prompts(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _git("remote", "add", "origin", "https://gitlab.example.com/grp/proj.git", cwd=repo)
    assert prov._gitlab_repo(str(repo)) == "https://gitlab.example.com/grp/proj"
    _git("remote", "set-url", "origin", "https://gitlab.com/minervaai/go-protos", cwd=repo)
    assert prov._gitlab_repo(str(repo)) == "minervaai/go-protos"
    assert _REAL_GITHUB_REPO(str(repo)) is None
    seen: dict[str, Any] = {}

    def run(argv, **kwargs):
        seen.update(argv=argv, timeout=kwargs["timeout"], prompt=kwargs["env"]["GLAB_NO_PROMPT"])
        return subprocess.CompletedProcess(argv, 0, '[{"iid": 7, "state": "opened"}]', "")

    monkeypatch.setattr(prov.subprocess, "run", run)
    assert prov._glab_open_mrs("g/p", "feat/x") == [7]
    assert seen["argv"] == [
        "glab",
        "mr",
        "list",
        "--repo",
        "g/p",
        "--source-branch",
        "feat/x",
        "-F",
        "json",
    ]
    assert seen["timeout"] == prov._GH_TIMEOUT_S and seen["prompt"] == "1"

    def fail(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, "", "unknown flag")

    monkeypatch.setattr(prov.subprocess, "run", fail)
    assert prov._glab_open_mrs("g/p", "feat/x") is None


# -- PR/MR history in the worktree facts (operator request, 2026-11-11) -------------
#
# "No open PR" used to read the same for merged, closed-unmerged and never-proposed
# branches. The facts now name the newest PR/MR and its date; ranking still rests on
# the open state alone, so every rank below matches what it was before.

_FC = prov.ForgeChange


@pytest.mark.parametrize(
    ("changes", "fact", "risk"),
    [
        ([_FC(120, "merged", "2026-10-01"), _FC(7, "open", "")], "open PR #7", Risk.CRITICAL),
        (
            [_FC(99, "closed", "2026-11-01"), _FC(123, "merged", "2026-11-09")],
            "no open PR; newest PR #123 merged 2026-11-09",
            Risk.LOW,
        ),
        (
            [_FC(55, "closed", "2026-11-02")],
            "no open PR; newest PR #55 closed unmerged 2026-11-02",
            Risk.LOW,
        ),
        ([], "no open PR; no PR found for this branch", Risk.LOW),
        (None, "PR state unknown (gh lookup on o/r failed or timed out)", Risk.HIGH),
    ],
    ids=["open", "merged", "closed", "never", "unknown"],
)
def test_worktree_facts_name_pr_history_without_changing_the_rank(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch, changes, fact: str, risk: Risk
) -> None:
    monkeypatch.setattr(prov, "_gh_prs", lambda r, b: changes)
    wt = _add(repo, repo / ".worktrees" / "hist", "hist")
    activity = prov.worktree_activity(str(wt), SID)
    assert fact in activity.facts()
    assert activity.pr_state == {Risk.CRITICAL: "open", Risk.LOW: "none"}.get(risk, "unknown")
    verdict = _verdict(home, f"git -C {repo} worktree remove {wt}")
    assert verdict.risk is risk
    assert fact in verdict.targets[0].reason


def test_gitlab_history_names_newest_mr_and_open_only_fallback_says_unchecked() -> None:
    merged = prov.WorktreeActivity(
        "b", "none", pr_label="MR !", newest=_FC(12, "merged", "2026-11-03")
    )
    assert "no open MR; newest MR !12 merged 2026-11-03" in merged.facts()
    unchecked = prov.WorktreeActivity("b", "none", pr_label="MR !", history_checked=False)
    assert "no open MR; merged/closed MRs not checked" in unchecked.facts()


@pytest.mark.parametrize("fallback", [[], [3], None])
def test_full_history_page_falls_back_to_the_open_only_listing(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch, fallback
) -> None:
    """A reused branch (10+ PRs) may hide an older open PR past the page."""
    page = [_FC(100 + i, "merged", "2026-11-0%d" % (i % 9 + 1)) for i in range(10)]
    monkeypatch.setattr(prov, "_gh_prs", lambda r, b: page)
    monkeypatch.setattr(prov, "_gh_open_prs", lambda r, b: fallback)
    wt = _add(repo, repo / ".worktrees" / "reused", "reused")
    activity = prov.worktree_activity(str(wt), SID)
    if fallback is None:
        assert activity.pr_state == "unknown"
    elif fallback:
        assert activity.pr_state == "open" and activity.pr_number == 3
    else:
        assert activity.pr_state == "none" and "newest PR #109 merged" in activity.facts()


def test_gh_history_lookup_is_one_bounded_call_and_failure_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}
    rows = (
        '[{"number": 5, "state": "MERGED", "mergedAt": "2026-11-09T15:16:50Z",'
        ' "closedAt": "2026-11-09T15:16:50Z", "updatedAt": "2026-11-10T00:00:00Z"},'
        ' {"number": 4, "state": "CLOSED", "mergedAt": null,'
        ' "closedAt": "2026-11-01T01:00:00Z", "updatedAt": "2026-11-01T01:00:00Z"}]'
    )

    def run(argv, **kwargs):
        seen.update(
            argv=argv, timeout=kwargs["timeout"], prompt=kwargs["env"]["GH_PROMPT_DISABLED"]
        )
        return subprocess.CompletedProcess(argv, 0, rows, "")

    monkeypatch.setattr(prov.subprocess, "run", run)
    assert _REAL_GH_PRS("o/r", "feat/x") == [
        _FC(5, "merged", "2026-11-09"),
        _FC(4, "closed", "2026-11-01"),
    ]
    assert seen["argv"] == [
        "gh", "pr", "list", "--repo", "o/r", "--head", "feat/x", "--state", "all",
        "--json", "number,state,mergedAt,closedAt,updatedAt", "--limit", "10",
    ]  # fmt: skip
    assert seen["timeout"] == prov._GH_TIMEOUT_S and seen["prompt"] == "1"

    def timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    def fail(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, "", "HTTP 502")

    def garbage(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, '[{"state": "OPEN"}]', "")

    for broken in (timeout, fail, garbage):
        monkeypatch.setattr(prov.subprocess, "run", broken)
        assert _REAL_GH_PRS("o/r", "feat/x") is None, broken.__name__


def test_glab_history_lookup_lists_all_states(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    rows = (
        '[{"iid": 9, "state": "opened", "updated_at": "2026-11-10T00:00:00Z"},'
        ' {"iid": 8, "state": "merged", "merged_at": "2026-11-08T00:00:00Z"}]'
    )

    def run(argv, **kwargs):
        seen.update(argv=argv, timeout=kwargs["timeout"])
        return subprocess.CompletedProcess(argv, 0, rows, "")

    monkeypatch.setattr(prov.subprocess, "run", run)
    assert _REAL_GLAB_MRS("g/p", "feat/x") == [
        _FC(9, "open", "2026-11-10"),
        _FC(8, "merged", "2026-11-08"),
    ]
    assert seen["argv"] == [
        "glab", "mr", "list", "--repo", "g/p", "--source-branch", "feat/x",
        "--all", "--per-page", "10", "-F", "json",
    ]  # fmt: skip
    assert seen["timeout"] == prov._GH_TIMEOUT_S


def _old_review_text(rule: str, reason: str, targets: str) -> str:
    """``review_text`` as it was before the worktree clause (965c5113855e)."""
    from local_operator.guardrails.check import BLOCK_MARKER, SUPPORTED_DELETE_PATHS

    return (
        f"{BLOCK_MARKER} {rule} (review): {reason}. Targets: {targets}. Nothing was run. "
        f"{SUPPORTED_DELETE_PATHS} {REFUSAL_SCOPE}"
    )


def test_high_worktree_refusal_carries_facts_and_quote_clause_others_unchanged(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.guardrails.check import (
        DANGEROUS_DELETE_RULE,
        WORKTREE_ASK_CLAUSE,
        _targets_text,
        worktree_flagged,
    )

    monkeypatch.setattr(prov, "_gh_prs", lambda r, b: [_FC(123, "merged", "2026-11-09")])
    wt = _add(repo, repo / ".worktrees" / "busy", "busy")
    (wt / "notes.txt").write_text("dirty")  # dirty → HIGH with the facts named
    verdict = _verdict(home, f"git -C {repo} worktree remove --force {wt}")
    assert verdict.risk is Risk.HIGH and worktree_flagged(verdict)
    text = review_text(
        DANGEROUS_DELETE_RULE, "r", _targets_text(verdict), worktree=worktree_flagged(verdict)
    )
    assert "newest PR #123 merged 2026-11-09" in text
    assert text.endswith(f"{REFUSAL_SCOPE} {WORKTREE_ASK_CLAUSE}")
    # Every other delete: the clause never appears, the text is byte-identical.
    other = home / "data"
    other.mkdir()
    (other / "x").write_text("x")
    plain = _verdict(home, f"rm -rf {other}")
    assert plain.risk is Risk.HIGH and not worktree_flagged(plain)
    targets = _targets_text(plain)
    got = review_text(DANGEROUS_DELETE_RULE, "r", targets, worktree=worktree_flagged(plain))
    assert got == review_text(DANGEROUS_DELETE_RULE, "r", targets)
    assert got == _old_review_text(DANGEROUS_DELETE_RULE, "r", targets)
    assert WORKTREE_ASK_CLAUSE not in got


def test_preview_of_a_high_worktree_remove_shows_history_and_clause(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.guardrails.check import WORKTREE_ASK_CLAUSE

    monkeypatch.setattr(prov, "_gh_prs", lambda r, b: [_FC(123, "merged", "2026-11-09")])
    wt = _add(repo, repo / ".worktrees" / "pv", "pv")
    (wt / "notes.txt").write_text("dirty")
    ctx = CallContext(cwd=str(home), env=_env(home), scratchpad=None, session_id=SID)
    settings = GuardSettings(enabled=True, rules_dirs=())
    report = preview_tool_call(
        "bash", {"command": f"git -C {repo} worktree remove --force {wt}"}, ctx, settings=settings
    )
    assert report.builtin_effect == "judge"
    assert "newest PR #123 merged 2026-11-09" in report.builtin_text
    assert WORKTREE_ASK_CLAUSE in report.builtin_text
    assert WORKTREE_ASK_CLAUSE in report.render()
