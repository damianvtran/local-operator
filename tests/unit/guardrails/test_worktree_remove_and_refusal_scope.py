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


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    state = home / ".local-operator"
    (state / "sessions" / SID / "scratchpad").mkdir(parents=True)
    (state / "config.yml").write_text("version: 1\n")
    # Synthetic repos have no live cwd; process inventory has dedicated controls
    # below and must not make git regressions depend on fleet lsof latency.
    monkeypatch.setattr(prov, "_live_cwd_inside", lambda _: False)
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
    assert "a clean linked git worktree" in [t.reason for t in ranked.targets]
    # Negative controls on the same tree: rm -rf is not ``git worktree remove``.
    assert _verdict(home, f"rm -rf {sib}").risk is Risk.HIGH
    assert _verdict(home, f"git -C {repo} worktree remove {repo}").risk is Risk.HIGH  # primary


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
    """The prompt line stays terse (CI's context-budget had 33 tokens of headroom
    on the base); the guide and the refusal text carry the full scope."""
    from local_operator.prompts_api import render_template

    prompt = " ".join(render_template("system.md", {}).split())
    assert (
        "A `BLOCKED by guardrail` result is final for that action (same effect, same "
        "targets, any route): report it. Fresh work of your own is not a reroute."
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
    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 5.0
        raise subprocess.TimeoutExpired(args[0], 5)
    monkeypatch.setattr(prov.subprocess, "run", timeout)
    # Exercise macOS branch independently of the CI runner's /proc.
    real_isdir = os.path.isdir
    monkeypatch.setattr(prov.os.path, "isdir", lambda p: False if p == "/proc/self" else real_isdir(p))
    assert prov._live_cwd_inside("/synthetic/tree") is None
