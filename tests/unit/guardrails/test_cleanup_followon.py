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

from local_operator.guardrails import classify as cls
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


def _finished_tree(
    home: Path, repo: Path, name: str, default: str = "main", at: Path | None = None
) -> Path:
    """A worktree whose branch commit was merged (fast-forward) into origin's default.

    ``at``: where the tree lives (default ``<repo>/.worktrees/<name>``).
    """
    wt = at if at is not None else repo / ".worktrees" / name
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


# --- Round-1 remediation (PR #3992) -------------------------------------------
#
# Each test pins one ledger ID from the round-1 review (R*) / QA (Q*) comments.


async def _carriers(
    home: Path, caplog: pytest.LogCaptureFixture, command: str, **env: str
) -> tuple[Any, str]:
    """Every outward carrier of one call joined: refusal text, warnings, judge
    brief, journal (the check logger) and preview. Returns ``(outcome, text)``."""
    caplog.clear()
    caplog.set_level(logging.INFO, logger="local_operator.guardrails.check")
    judge = _Judge()
    out = await _check(home, command, judge, **env)
    ctx = CallContext(cwd=str(home), env=_env(home, **env), scratchpad=None, session_id=ME)
    report = preview_tool_call("bash", {"command": command}, ctx)
    parts = [
        out.text if out is not None else "",
        *(out.warnings if out is not None else []),
        *judge.prompts,
        caplog.text,
        repr(report),
    ]
    return out, "\n".join(parts)


def _canary_value(home: Path) -> str:
    sub = home / CANARY / "sub"
    sub.mkdir(parents=True)
    (sub / "old.pem").write_text("k\n")
    return str(sub / "old.pem")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        'rm -rf "$GOOGLE_APPLICATION_CREDENTIALS/.."',
        'D="$GOOGLE_APPLICATION_CREDENTIALS"; rm -rf "$D/.."',
        'rm -rf "$GOOGLE_APPLICATION_CREDENTIALS"*',
        'rm -rf "/x$GOOGLE_APPLICATION_CREDENTIALS"',
        'cd "$GOOGLE_APPLICATION_CREDENTIALS/.."; rm -rf ../sub',
        'ln -sfn "$GOOGLE_APPLICATION_CREDENTIALS/.." ~/L; rm -rf ~/L/x',
        'rm -rf "${GOOGLE_APPLICATION_CREDENTIALS%/*}"',
        'export D="$GOOGLE_APPLICATION_CREDENTIALS"; rm -rf "$D/.."',
        'for f in "$GOOGLE_APPLICATION_CREDENTIALS/.."; do rm -rf "$f"; done',
        'cd "$(dirname "$GOOGLE_APPLICATION_CREDENTIALS")"; rm -rf ..',
    ],
    ids=[
        "dotdot",
        "copied-dotdot",
        "glued-glob",
        "prefixed",
        "cd-dotdot",
        "link-pointee",
        "suffix-strip",
        "exported-copy",
        "loop-copy",
        "cd-dirname",
    ],
)
async def test_r1_q2_credential_target_outside_its_value_stays_opaque(
    home: Path, caplog: pytest.LogCaptureFixture, command: str
) -> None:
    """R1 = Q2: a word that leaves the value's own subtree ranks opaque, as on
    base 33c331e3, and no carrier names the value's directories."""
    env = {"GOOGLE_APPLICATION_CREDENTIALS": _canary_value(home)}
    v = _rank(home, command, **env)
    assert v.risk is not Risk.LOW, [(t.path, t.reason) for t in v.targets]
    assert all(CANARY not in (t.target.path or "") for t in v.targets)
    out, text = await _carriers(home, caplog, command, **env)
    assert out is not None and out.blocked
    assert CANARY not in text


def test_r1_value_subtree_still_ranks_like_the_literal(home: Path) -> None:
    """Control: the value itself and paths UNDER it keep item C's literal ranking."""
    value = _canary_value(home)
    env = {"GOOGLE_APPLICATION_CREDENTIALS": str(Path(value).parent)}
    for command in (
        'rm -f "$GOOGLE_APPLICATION_CREDENTIALS/old.pem"',
        'rm -rf "$GOOGLE_APPLICATION_CREDENTIALS"/*',
    ):
        v = _rank(home, command, **env)
        assert v.risk is Risk.LOW, (command, [(t.path, t.reason) for t in v.targets])
        assert not any(t.target.credential_derived for t in v.targets)


@pytest.mark.asyncio
async def test_r2_1_cd_into_the_value_then_climb_stays_opaque(
    home: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """R2-1 neighbour: ``cd "$VAR"; rm -rf ..`` with a DIRECTORY value still
    hides the cwd once detection keys on the expansion's refs."""
    env = {"GOOGLE_APPLICATION_CREDENTIALS": str(Path(_canary_value(home)).parent)}
    command = 'cd "$GOOGLE_APPLICATION_CREDENTIALS"; rm -rf ..'
    v = _rank(home, command, **env)
    assert v.risk is not Risk.LOW, [(t.path, t.reason) for t in v.targets]
    assert all(CANARY not in (t.target.path or "") for t in v.targets)
    out, text = await _carriers(home, caplog, command, **env)
    assert out is not None and out.blocked and CANARY not in text


@pytest.mark.parametrize(
    "command",
    [
        "rm -f ~/keys/sa.json.bak",
        "rm -f {home}/keys/sa.json.bak",
        "rm -rf {home}/keys/sa.json.d/x",
        "cp {home}/keys/new.json {home}/keys/sa.json.tmp",
        'rm -f "$HOME/keys/sa.json.bak"',
        "cd {home}/keys/sa.json.d && rm -rf x",
        "ln -sfn {home}/keys/sa.json.d ~/L && rm -rf ~/L/x",
    ],
    ids=[
        "tilde-bak",
        "abs-bak",
        "abs-subdir",
        "cp-tmp",
        "home-var-bak",
        "cd-literal",
        "ln-literal",
    ],
)
def test_r2_1_literal_paths_extending_the_value_rank_as_on_base(home: Path, command: str) -> None:
    """R2-1 (PR #3992): a word that never references the credential variable
    but whose literal text EXTENDS its value (key-rotation backups beside the
    live key) ranks exactly as on base 33c331e3: LOW, not credential-derived.
    A substring test once made these a final CRITICAL refusal aimed at HOME."""
    keys = home / "keys"
    (keys / "sa.json.d" / "x").mkdir(parents=True)
    for name in ("sa.json", "sa.json.bak", "new.json"):
        (keys / name).write_text("{}\n")
    env = {"GOOGLE_APPLICATION_CREDENTIALS": str(keys / "sa.json")}
    v = _rank(home, command.format(home=home), **env)
    assert v.risk is Risk.LOW, (command, [(t.path, t.reason) for t in v.targets])
    assert not any(t.target.credential_derived for t in v.targets)
    assert "credential" not in " ".join(t.reason for t in v.targets)


def _foreign_dir(home: Path) -> Path:
    """``~/w/<canary>`` recorded by another session, holding ``old.pem``."""
    d = home / "w" / CANARY
    d.mkdir(parents=True)
    (d / "old.pem").write_text("k\n")
    prov.write_records(
        [(os.path.realpath(d), "mkdir")],
        owners=[DONE],
        creator=DONE,
        command=f"mkdir -p {d}",
        state_root=home / ".local-operator",
    )
    return d


@pytest.mark.asyncio
async def test_r1_q2_ancestor_reason_hides_the_ledger_directory(
    home: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Q2's repro: the value sits inside another session's created dir; the
    "inside <dir>, created by another session" reason must not name it."""
    d = _foreign_dir(home)
    env = {"GOOGLE_APPLICATION_CREDENTIALS": str(d / "old.pem")}
    command = 'rm -f "$GOOGLE_APPLICATION_CREDENTIALS"'
    v = _rank(home, command, **env)
    assert v.risk is Risk.HIGH
    reason = v.targets[0].reason
    assert "created by another session" in reason and CANARY not in reason
    assert "a directory holding $GOOGLE_APPLICATION_CREDENTIALS <credential path>" in reason
    out, text = await _carriers(home, caplog, command, **env)
    assert out is not None and out.blocked
    assert CANARY not in text and "<credential path>" in text


@pytest.mark.asyncio
async def test_r1_q2_value_reached_through_a_symlink_is_hidden(
    home: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The value's lexical spelling holds no canary; its physical one does."""
    d = _foreign_dir(home)
    (home / "w" / "lnk").symlink_to(d)
    env = {"GOOGLE_APPLICATION_CREDENTIALS": str(home / "w" / "lnk" / "old.pem")}
    for command in (
        'rm -f "$GOOGLE_APPLICATION_CREDENTIALS"',
        'rm -rf "$GOOGLE_APPLICATION_CREDENTIALS/.."',
    ):
        out, text = await _carriers(home, caplog, command, **env)
        assert out is not None and out.blocked, command
        assert CANARY not in text, command


# --- Q1 (MAJOR): protected places keep main's rank ---------------------------


@pytest.mark.parametrize(
    "command", ["git worktree remove --force {wt}", "rm -rf {wt}"], ids=["worktree-force", "rm-rf"]
)
@pytest.mark.parametrize(
    "where",
    [".local-operator/worktrees/b", f".local-operator/sessions/{DONE}/scratchpad/b"],
    ids=["state-root", "other-scratchpad"],
)
def test_q1_finished_tree_in_a_protected_place_stays_high(
    home: Path, repo: Path, command: str, where: str
) -> None:
    wt = _finished_tree(home, repo, "b", at=home / where)
    assert prov.finished_foreign_worktree(str(wt), ME).safe  # the facts alone pass
    v = _rank(home, command.format(wt=wt))
    assert v.risk is Risk.HIGH, [(t.path, t.reason) for t in v.targets]
    assert not v.warnings


@pytest.mark.parametrize(
    "command", ["git worktree remove --force {wt}", "rm -rf {wt}"], ids=["worktree-force", "rm-rf"]
)
def test_q1_control_plain_finished_tree_elsewhere_stays_low(
    home: Path, repo: Path, command: str
) -> None:
    wt = _finished_tree(home, repo, "b", at=home / "w" / "b")
    assert _rank(home, command.format(wt=wt)).risk is Risk.LOW


# --- R2 (minor): another session's ledger row inside the tree ----------------


def _row_inside(home: Path, repo: Path, creator: str) -> Path:
    wt = _finished_tree(home, repo, "done")
    inner = wt / "emptydir"
    inner.mkdir()
    prov.write_records(
        [(os.path.realpath(inner), "mkdir")],
        owners=[creator],
        creator=creator,
        command=f"mkdir {inner}",
        state_root=home / ".local-operator",
    )
    # The mkdir moved the tree's ctime (Linux ``creation_stamp``): re-record
    # its own row so the predicate's owner fact still reads it.
    _record(home, wt)
    return wt


def test_r2_third_session_row_inside_the_tree_bars_the_route(home: Path, repo: Path) -> None:
    wt = _row_inside(home, repo, "ccccccccccc3")
    assert prov.finished_foreign_worktree(str(wt), ME).safe
    rows = prov.ForeignCreations(ME)
    try:
        got = cls._no_ledger_inside(rows, str(wt), prov.finished_foreign_worktree(str(wt), ME))
    finally:
        rows.close()
    assert got.reasons == ("foreign-ledger-inside",) and not got.safe
    for command in (f"rm -rf {wt}", f"git worktree remove --force {wt}"):
        v = _rank(home, command)
        assert v.risk is Risk.HIGH, (command, [(t.path, t.reason) for t in v.targets])


def test_r2_control_creators_own_row_inside_does_not_bar(home: Path, repo: Path) -> None:
    wt = _row_inside(home, repo, DONE)
    assert _rank(home, f"rm -rf {wt}").risk is Risk.LOW


# --- R3 (minor): notices only on a LOW verdict --------------------------------


def test_r3_high_verdict_carries_no_removed_without_review_notice(home: Path, repo: Path) -> None:
    _finished_tree(home, repo, "done")
    v = _rank(home, f"rm -rf {repo}")
    assert v.risk is Risk.HIGH
    assert v.warnings == []


# --- R4 (minor): the race re-check re-reads clean and busy -------------------


@pytest.mark.parametrize("change", ["file", "busy"])
def test_r4_state_change_during_the_forge_lookup_is_caught(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    wt = _finished_tree(home, repo, "done")
    moved = {"now": False}

    def forge(repo_: str, branch: str) -> list[Any]:
        # The forge lookup is the slow step the re-check guards.
        moved["now"] = True
        if change == "file":
            (wt / "late.txt").write_text("x\n")
        return []

    monkeypatch.setattr(prov, "_gh_prs", forge)
    monkeypatch.setattr(prov, "_live_cwd_inside", lambda _: moved["now"] and change == "busy")
    got = prov.finished_foreign_worktree(str(wt), ME)
    assert not got.safe and got.reasons == ("state-moved",), got


# --- Q4 (minor): the branch-ref clause is conditional ------------------------


def test_q4_warning_does_not_promise_the_branch_ref(home: Path, repo: Path) -> None:
    wt = _finished_tree(home, repo, "done")
    v = _rank(home, f"rm -rf {wt}")
    assert v.risk is Risk.LOW and v.warnings
    assert all("Its branch ref is kept" not in w for w in v.warnings)
    assert any("unless that is deleted too" in w for w in v.warnings)


# --- Round-3 remediation (PR #3992) -------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        'eval rm -rf "$GOOGLE_APPLICATION_CREDENTIALS/.."',
        'eval "rm -rf $GOOGLE_APPLICATION_CREDENTIALS/.."',
        'bash -c "rm -rf $GOOGLE_APPLICATION_CREDENTIALS/.."',
        'sh -c "rm -rf $GOOGLE_APPLICATION_CREDENTIALS/.."',
        'zsh -c "rm -rf $GOOGLE_APPLICATION_CREDENTIALS/.."',
        'env X=1 bash -c "rm -rf $GOOGLE_APPLICATION_CREDENTIALS/.."',
        'bash <<< "rm -rf $GOOGLE_APPLICATION_CREDENTIALS/.."',
        'source /dev/stdin <<< "rm -rf $GOOGLE_APPLICATION_CREDENTIALS/.."',
        'source <(echo "rm -rf $GOOGLE_APPLICATION_CREDENTIALS/..")',
        "bash <<EOF\nrm -rf $GOOGLE_APPLICATION_CREDENTIALS/..\nEOF",
    ],
    ids=[
        "eval-bare",
        "eval-quoted",
        "bash-c",
        "sh-c",
        "zsh-c",
        "env-bash-c",
        "bash-herestring",
        "source-herestring",
        "source-procsub",
        "bash-heredoc",
    ],
)
async def test_r3_1_rewalked_credential_text_stays_opaque(
    home: Path, caplog: pytest.LogCaptureFixture, command: str
) -> None:
    """R3-1 (PR #3992): ``eval``/``sh -c``/a herestring re-parse EXPANDED text,
    which drops the ``refs`` spans that mark the credential value, so the inner
    walk once ranked and printed the value's parent directory. These must stay
    at least HIGH and opaque, as on base 33c331e3, with no carrier naming it."""
    env = {"GOOGLE_APPLICATION_CREDENTIALS": _canary_value(home)}
    v = _rank(home, command, **env)
    assert v.risk in (Risk.HIGH, Risk.CRITICAL), (v.risk, [(t.path, t.reason) for t in v.targets])
    assert all(CANARY not in (t.target.path or "") for t in v.targets)
    out, text = await _carriers(home, caplog, command, **env)
    assert out is not None and out.blocked
    assert CANARY not in text


def test_r3_1_literal_eval_ranks_as_on_base(home: Path) -> None:
    """Control: an ``eval`` of literal text that only EXTENDS the value (no
    variable) is still re-walked and ranks like the bare command (LOW)."""
    keys = home / "keys"
    keys.mkdir()
    for name in ("sa.json", "sa.json.bak"):
        (keys / name).write_text("{}\n")
    env = {"GOOGLE_APPLICATION_CREDENTIALS": str(keys / "sa.json")}
    for command in ("eval rm -f ~/keys/sa.json.bak", "rm -f ~/keys/sa.json.bak"):
        v = _rank(home, command, **env)
        assert v.risk is Risk.LOW, (command, [(t.path, t.reason) for t in v.targets])
        assert not any(t.target.credential_derived for t in v.targets)
