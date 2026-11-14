"""Guardrail cleanup follow-on: finished foreign worktrees, credential paths, loop links.

Operator request (verbatim): "The same is the case for cleaning up folders and
worktrees, the guardrails are not allowing agents to clean up after themselves
on obviously safe directories and worktree folders, etc so the guidance needs
to be improved there". Operator decision (ask a-891c, "Yes, rank it LOW"): a
clean, merged-on-origin, idle worktree with no open PR ranks LOW even if
another, finished session created it.

Fixtures are tiny real git repositories: a local BARE repository is the
``origin`` (its advertised HEAD is what ``ls-remote`` reads; no network), a
clone holds the linked worktrees, and the forge and session-owner lookups are
synthetic stubs. HOME and the state root are under ``tmp_path``; ids are
synthetic, never a live session.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_operator.guardrails import provenance as prov
from local_operator.guardrails.check import CallContext, GuardSettings, check_tool_call
from local_operator.guardrails.classify import Risk, classify_bash
from local_operator.guardrails.preview import preview_tool_call
from local_operator.guardrails.resolve import credential_path

ME = "aaaaaaaaaaa1"  # the cleaning session
DONE = "bbbbbbbbbbb2"  # the finished session that created the worktree


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@e", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    state = home / ".local-operator"
    for sid in (ME, DONE):
        (state / "sessions" / sid / "scratchpad").mkdir(parents=True)
    (state / "config.yml").write_text("version: 1\n")
    monkeypatch.setattr(prov, "_live_cwd_inside", lambda _: False)
    monkeypatch.setattr(prov, "_github_repo", lambda _: "example/repo")
    monkeypatch.setattr(prov, "_gh_prs", lambda repo, branch: [])
    monkeypatch.setattr(prov, "_gh_open_prs", lambda repo, branch: [])
    monkeypatch.setattr(prov, "_running_owners", lambda real, sid, strict=False: ())
    monkeypatch.setattr(prov, "_cwd_inventory_cache", None)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))
    for key in [k for k in os.environ if k.startswith(("CMUX_", "GIT_", "LOP_"))]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    return home


def _env(home: Path, **extra: str) -> dict[str, str]:
    return {"HOME": str(home), "LOCAL_OPERATOR_CONFIG_DIR": str(home / ".local-operator"), **extra}


def _record(home: Path, wt: Path, creator: str = DONE) -> None:
    prov.write_records(
        [(os.path.realpath(wt), "worktree")],
        owners=[creator],
        creator=creator,
        command=f"git worktree add {wt}",
        state_root=home / ".local-operator",
    )


def _repo(home: Path, default: str = "main") -> Path:
    """A clone of a bare ``origin`` whose advertised default branch is ``default``."""
    origin = home / "origin.git"
    _git("init", "-q", "--bare", "-b", default, str(origin), cwd=home)
    repo = home / "repo"
    _git("init", "-q", "-b", default, str(repo), cwd=home)
    (repo / ".gitignore").write_text(".worktrees/\n.venv/\n")
    (repo / "a.txt").write_text("a\n")
    _git("add", ".", cwd=repo)
    _git("commit", "-qm", "init", cwd=repo)
    _git("remote", "add", "origin", str(origin), cwd=repo)
    _git("push", "-q", "origin", default, cwd=repo)
    _git("fetch", "-q", "origin", cwd=repo)
    return repo


def _finished_tree(home: Path, repo: Path, name: str, default: str = "main") -> Path:
    """A worktree whose branch commit was merged (fast-forward) into origin's default."""
    wt = repo / ".worktrees" / name
    # Cut from origin's current tip, so each tree fast-forwards it in turn.
    _git("worktree", "add", "-q", "-b", f"fix/{name}", str(wt), f"origin/{default}", cwd=repo)
    (wt / f"{name}.txt").write_text("done\n")
    _git("add", ".", cwd=wt)
    _git("commit", "-qm", f"fix {name}", cwd=wt)
    _git("push", "-q", "origin", f"HEAD:{default}", cwd=wt)
    _git("fetch", "-q", "origin", cwd=repo)
    _record(home, wt)
    return wt


@pytest.fixture
def repo(home: Path) -> Path:
    return _repo(home)


def _rank(home: Path, command: str, sid: str = ME, **env: str) -> Any:
    return classify_bash(
        command, cwd=str(home), env=_env(home, **env), scratchpad=None, session_id=sid
    )


class _Judge:
    def __init__(self, answer: str = "BLOCK: not verified") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    async def __call__(self, system: str, prompt: str) -> str:
        self.prompts.append(system + "\n" + prompt)
        return self.answer


async def _check(home: Path, command: str, judge: _Judge | None = None, **env: str) -> Any:
    ctx = CallContext(cwd=str(home), env=_env(home, **env), scratchpad=None, session_id=ME)
    return await check_tool_call(
        "bash",
        {"command": command},
        ctx,
        judge=judge,
        settings=GuardSettings(),
        forked_judge=judge,
    )


# --- F. loop-made links -------------------------------------------------------


def test_loop_link_to_the_state_root_is_critical(home: Path) -> None:
    """The exact fail-open: ``link-b`` (2nd iteration) points at the state root."""
    (home / "worktrees" / "probe").mkdir(parents=True)
    v = _rank(
        home,
        "for n in a b; do ln -sfn ~/.local-operator ~/worktrees/probe/link-$n; done; "
        "rm -f ~/worktrees/probe/link-b/config.yml",
    )
    assert v.risk is Risk.CRITICAL, [(t.path, t.reason) for t in v.targets]


def test_fanned_target_through_an_earlier_link_is_critical(home: Path) -> None:
    """The S7 test applies to every iteration's target, not just the first."""
    v = _rank(home, "ln -sfn ~/.local-operator ~/l2; for n in x l2; do rm -f ~/$n/config.yml; done")
    assert v.risk is Risk.CRITICAL, [(t.path, t.reason) for t in v.targets]


def test_loop_link_to_a_harmless_dir_still_ranks_low(home: Path) -> None:
    (home / "worktrees" / "probe").mkdir(parents=True)
    (home / "worktrees" / "own").mkdir()
    v = _rank(
        home,
        "for n in a b; do ln -sfn ~/worktrees/own ~/worktrees/probe/link-$n; done; "
        "rm -f ~/worktrees/probe/link-b/x.txt",
    )
    assert v.risk is Risk.LOW, [(t.path, t.reason) for t in v.targets]


def test_loop_link_with_an_unpaired_pointee_variable_is_opaque(home: Path) -> None:
    """The pointee varies with a loop the link does not: never ordinary."""
    (home / "w").mkdir()
    v = _rank(
        home,
        "for p in ~/.local-operator ~/w; do ln -sfn $p ~/w/l; done; rm -f ~/w/l/config.yml",
    )
    assert v.risk >= Risk.HIGH, [(t.path, t.reason) for t in v.targets]


# --- A. finished foreign worktrees --------------------------------------------


def test_predicate_proves_a_finished_foreign_tree_safe(home: Path, repo: Path) -> None:
    wt = _finished_tree(home, repo, "done")
    got = prov.finished_foreign_worktree(str(wt), ME)
    assert got.safe, got
    assert "merged-on-origin" in got.reasons and "owner-idle" in got.reasons


@pytest.mark.parametrize(
    "command",
    ["git worktree remove --force {wt}", "rm -rf {wt}", "rm -rf {wt}/"],
    ids=["worktree-force", "rm-rf", "rm-rf-slash"],
)
def test_finished_foreign_tree_ranks_low_with_a_warning(
    home: Path, repo: Path, command: str
) -> None:
    wt = _finished_tree(home, repo, "done")
    v = _rank(home, command.format(wt=wt))
    assert v.risk is Risk.LOW, [(t.path, t.reason) for t in v.targets]
    assert any("finished worktree" in w and "merged" in w for w in v.warnings), v.warnings


@pytest.mark.asyncio
async def test_checkpoint_passes_it_with_the_warning_and_real_removal_completes(
    home: Path, repo: Path
) -> None:
    wt = _finished_tree(home, repo, "done")
    judge = _Judge("ALLOW")
    outcome = await _check(home, f"git worktree remove --force {wt}", judge)
    assert outcome is not None and not outcome.blocked
    assert judge.prompts == []
    assert any("finished worktree" in w for w in outcome.warnings), outcome.warnings
    _git("worktree", "remove", "--force", str(wt), cwd=repo)
    assert not wt.exists()
    # The branch ref survives: the commits are still reachable.
    assert _git("rev-parse", "--verify", "refs/heads/fix/done", cwd=repo).strip()


def _later_commit(home: Path, repo: Path, wt: Path) -> None:
    (wt / "later.txt").write_text("x\n")
    _git("add", ".", cwd=wt)
    _git("commit", "-qm", "later", cwd=wt)


def _detach(home: Path, repo: Path, wt: Path) -> None:
    _git("checkout", "-q", "--detach", cwd=wt)


def _stale_origin(home: Path, repo: Path, wt: Path) -> None:
    other = home / "other"
    _git("clone", "-q", str(home / "origin.git"), str(other), cwd=home)
    (other / "b.txt").write_text("b\n")
    _git("add", ".", cwd=other)
    _git("commit", "-qm", "moved", cwd=other)
    _git("push", "-q", "origin", "HEAD:main", cwd=other)


def _no_origin(home: Path, repo: Path, wt: Path) -> None:
    _git("remote", "remove", "origin", cwd=repo)


def _unreachable_origin(home: Path, repo: Path, wt: Path) -> None:
    _git("remote", "set-url", "origin", str(home / "missing.git"), cwd=repo)


def _ignored_venv(home: Path, repo: Path, wt: Path) -> None:
    (wt / ".venv").mkdir()
    (wt / ".venv" / "private-backup.json").write_text("{}")


def _untracked(home: Path, repo: Path, wt: Path) -> None:
    (wt / "wip.txt").write_text("wip")


@pytest.mark.parametrize(
    ("mutate", "verdict", "code"),
    [
        (_later_commit, "not-safe", "not-merged"),
        (_detach, "not-safe", "detached"),
        (_stale_origin, "unknown", "origin-ref-stale"),
        (_no_origin, "unknown", "origin-unreachable"),
        (_unreachable_origin, "unknown", "origin-unreachable"),
        (_ignored_venv, "not-safe", "ignored-content"),
        (_untracked, "not-safe", "state-dirty"),
    ],
    ids=lambda p: getattr(p, "__name__", p),
)
def test_missing_facts_never_earn_the_route(
    home: Path, repo: Path, mutate: Any, verdict: str, code: str
) -> None:
    wt = _finished_tree(home, repo, "done")
    mutate(home, repo, wt)
    got = prov.finished_foreign_worktree(str(wt), ME)
    assert (got.verdict, got.reasons) == (verdict, (code,)), got
    v = _rank(home, f"rm -rf {wt}")
    assert v.risk >= Risk.HIGH, [(t.path, t.reason) for t in v.targets]
    assert not v.warnings


def test_squash_merge_without_ancestry_stays_with_the_judge(home: Path, repo: Path) -> None:
    wt = repo / ".worktrees" / "squashed"
    _git("worktree", "add", "-q", "-b", "fix/squashed", str(wt), cwd=repo)
    (wt / "s.txt").write_text("s\n")
    _git("add", ".", cwd=wt)
    _git("commit", "-qm", "branch commit", cwd=wt)
    # The same content lands on main as a DIFFERENT commit (a squash).
    (repo / "s.txt").write_text("s\n")
    _git("add", "s.txt", cwd=repo)
    _git("commit", "-qm", "squash", cwd=repo)
    _git("push", "-q", "origin", "main", cwd=repo)
    _git("fetch", "-q", "origin", cwd=repo)
    _record(home, wt)
    assert prov.finished_foreign_worktree(str(wt), ME).reasons == ("not-merged",)
    assert _rank(home, f"rm -rf {wt}").risk is Risk.HIGH


def test_non_main_default_branch_is_read_from_origin(home: Path) -> None:
    repo = _repo(home, default="trunk")
    wt = _finished_tree(home, repo, "done", default="trunk")
    assert prov.finished_foreign_worktree(str(wt), ME).safe


@pytest.mark.parametrize(
    ("stub", "verdict", "code"),
    [
        ({"_gh_prs": lambda r, b: [prov.ForgeChange(7, "open", "")]}, "not-safe", "open-pr"),
        ({"_gh_prs": lambda r, b: None}, "unknown", "pr-unknown"),
        (
            {"_running_owners": lambda real, sid, strict=False: ("cccccccccc3",)},
            "not-safe",
            "owner-running",
        ),
    ],
    ids=["open-pr", "pr-unknown", "owner-running"],
)
def test_activity_facts_gate_the_route(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch, stub: Any, verdict: str, code: str
) -> None:
    wt = _finished_tree(home, repo, "done")
    for name, fn in stub.items():
        monkeypatch.setattr(prov, name, fn)
    got = prov.finished_foreign_worktree(str(wt), ME)
    assert (got.verdict, got.reasons) == (verdict, (code,)), got
    assert _rank(home, f"rm -rf {wt}").risk >= Risk.HIGH


def test_unreadable_owner_inventory_is_unknown_not_idle(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wt = _finished_tree(home, repo, "done")

    def broken(real: str, sid: str, strict: bool = False) -> tuple[str, ...]:
        if strict:
            raise OSError("registry unreadable")
        return ()

    monkeypatch.setattr(prov, "_running_owners", broken)
    assert prov.finished_foreign_worktree(str(wt), ME).reasons == ("owner-unknown",)
    assert _rank(home, f"rm -rf {wt}").risk is Risk.HIGH


def test_real_strict_owner_scan_without_a_ledger_row_is_unknown(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The REAL ``_running_owners(strict=True)``: a tree with no creator row
    (pre-ledger) is unknown ownership, never idle."""
    wt = repo / ".worktrees" / "unrecorded"
    _git("worktree", "add", "-q", "-b", "fix/unrecorded", str(wt), cwd=repo)
    monkeypatch.undo()  # drop every stub, then restore only the isolation
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(home / ".local-operator"))
    monkeypatch.setattr(prov, "_live_cwd_inside", lambda _: False)
    assert prov.finished_foreign_worktree(str(wt), ME).reasons == ("owner-unknown",)


def test_glob_over_only_finished_trees_is_low(home: Path, repo: Path) -> None:
    _finished_tree(home, repo, "one")
    _finished_tree(home, repo, "two")
    v = _rank(home, f"rm -rf {repo}/.worktrees/*")
    assert v.risk is Risk.LOW, [(t.path, t.reason) for t in v.targets]
    assert len(v.warnings) == 2


def test_one_dirty_foreign_tree_keeps_the_glob_high(home: Path, repo: Path) -> None:
    _finished_tree(home, repo, "one")
    two = _finished_tree(home, repo, "two")
    (two / "wip.txt").write_text("wip")
    v = _rank(home, f"rm -rf {repo}/.worktrees/*")
    assert v.risk is Risk.HIGH
    assert "wip.txt" in v.targets[0].reason


def test_parent_of_finished_trees_is_ranked_on_its_own_merits(home: Path, repo: Path) -> None:
    """All-safe roots remove only the worktree objection: the parent's own
    ranking (and any other foreign row in it) still applies."""
    _finished_tree(home, repo, "one")
    parent = repo / ".worktrees"
    (parent / "notes").mkdir()
    prov.write_records(
        [(os.path.realpath(parent / "notes"), "dir")],
        owners=[DONE],
        creator=DONE,
        command="mkdir notes",
        state_root=home / ".local-operator",
    )
    v = _rank(home, f"rm -rf {parent}")
    assert v.risk is Risk.HIGH
    assert "created by another session" in v.targets[0].reason


def test_critical_precedence_is_unchanged(home: Path, repo: Path) -> None:
    wt = _finished_tree(home, repo, "done")
    assert _rank(home, f"rm -rf {wt} ~/.local-operator").risk is Risk.CRITICAL


# --- C. credential path variables ---------------------------------------------

CANARY = "canary-k3ys-q9"
#: Synthetic token-shaped value: not a path, so it keeps the credential mark.
_TOKEN = "synthetic-token-" + "0" * 24


@pytest.mark.parametrize(
    "value",
    [
        "{home}/" + CANARY + "/old.pem",
        "~/" + CANARY + "/old.pem",
        "{home}/" + CANARY + "/my key.pem",
    ],
    ids=["absolute", "tilde", "spaces"],
)
def test_path_valued_credential_ranks_like_the_literal(home: Path, value: str) -> None:
    (home / CANARY).mkdir()
    v = _rank(
        home,
        'rm -f "$GOOGLE_APPLICATION_CREDENTIALS"',
        GOOGLE_APPLICATION_CREDENTIALS=value.format(home=home),
    )
    literal = _rank(home, f'rm -f "{value.format(home=home).replace("~", str(home), 1)}"')
    assert v.risk is literal.risk is Risk.LOW
    assert not any(t.target.credential_derived for t in v.targets)


def test_credential_path_into_the_state_root_is_critical(home: Path) -> None:
    v = _rank(
        home,
        'rm -f "$GOOGLE_APPLICATION_CREDENTIALS"',
        GOOGLE_APPLICATION_CREDENTIALS=str(home / ".local-operator" / "config.yml"),
    )
    assert v.risk is Risk.CRITICAL


@pytest.mark.parametrize(
    "value",
    [
        "ghp_0123456789abcdef",
        "https://example.com/key",
        "/tmp/a\nb",
        "$(cat k)",
        "keys/old.pem",
        "~other/k",
        "",
        "/zz-no-such-top/k",
        "/tmp/k;rm",
    ],
)
def test_non_path_credentials_keep_the_opaque_mark(value: str) -> None:
    assert credential_path(value, "/home/x") is None


@pytest.mark.asyncio
async def test_token_credential_keeps_the_existing_refusal(home: Path) -> None:
    out = await _check(home, 'rm -rf "$GITHUB_TOKEN"', GITHUB_TOKEN=_TOKEN)
    assert out is not None and out.blocked
    assert _TOKEN not in out.text


@pytest.mark.asyncio
async def test_no_carrier_discloses_the_private_path(
    home: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Refusal, judge brief, journal and preview: canary never shown, label is."""
    (home / CANARY).mkdir()
    label = "$GOOGLE_APPLICATION_CREDENTIALS <credential path>"
    caplog.set_level(logging.INFO, logger="local_operator.guardrails.check")
    # CRITICAL refusal: a credential path under the state root.
    crit = str(home / ".local-operator" / CANARY)
    out = await _check(
        home, 'rm -rf "$GOOGLE_APPLICATION_CREDENTIALS"', GOOGLE_APPLICATION_CREDENTIALS=crit
    )
    assert out is not None and out.blocked and CANARY not in out.text
    assert label in out.text
    # HIGH, judged: a top-level home directory, with a suffix glued on.
    high = {"GOOGLE_APPLICATION_CREDENTIALS": str(home / CANARY)}
    judge = _Judge()
    out = await _check(home, 'rm -rf "$GOOGLE_APPLICATION_CREDENTIALS"', judge, **high)
    assert out is not None and out.blocked and CANARY not in out.text
    assert judge.prompts and all(CANARY not in p for p in judge.prompts)
    assert any(label in p for p in judge.prompts)
    assert CANARY not in caplog.text and "guardrail.verdict" in caplog.text
    ctx = CallContext(cwd=str(home), env=_env(home, **high), scratchpad=None, session_id=ME)
    report = preview_tool_call("bash", {"command": 'rm -rf "$GOOGLE_APPLICATION_CREDENTIALS"'}, ctx)
    shown = repr(report)
    assert CANARY not in shown and label in shown
