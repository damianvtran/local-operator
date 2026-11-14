"""Scoped consent must survive final verification, not merely a judge's ALLOW."""

import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.guardrails import provenance as prov
from local_operator.guardrails.check import (
    CallContext,
    DeleteCase,
    GuardSettings,
    check_tool_call,
)
from local_operator.guardrails.judge import ContextFreeAnswer, JudgeBrief, render_brief
from local_operator.guardrails.provenance import OperatorBrief, OperatorWords
from local_operator.harness.types import Message
from local_operator.session.runtime import registry
from local_operator.session.runtime.types import SessionRecord


def case(text="I give full approval to delete completed worktrees"):
    return DeleteCase(
        paths=["/fixture/repo/.worktrees/done"],
        operator_only=frozenset(),
        unverifiable=[],
        operator=OperatorBrief(words=[OperatorWords("operator, direct", text)]),
        evidence=SimpleNamespace(listed={}, calls=[]),
        operator_words=(),
        excluded="",
        facts=(),
        env={},
        stop=("/",),
        home="/fixture",
        cwd="/fixture/repo",
    )


def test_reproduce_broad_consent_rejected_but_exact_path_accepted():
    c = case()
    assert c.verify("ALLOW operator")[1] == c.paths[0]
    assert not case(f"Delete {c.paths[0]}").verify("ALLOW operator")[1]


def test_scope_is_verified_only_with_fork_and_all_fresh_eligible_paths():
    c = case()
    eligible = frozenset(c.paths)
    basis, missing = c.verify("ALLOW scope=o1", forked=True, scope_eligible=eligible)
    assert not missing and "independent judge verified completion and scope" in basis
    assert c.verify("ALLOW scope=o1")[1]
    assert c.verify("ALLOW scope=o1", scope_eligible=eligible)[1]
    assert c.verify("ALLOW scope=o1", forked=True)[1]
    assert c.verify("ALLOW scope=o1", alone=True, forked=True, scope_eligible=eligible)[1]
    assert not c.evidence_basis()


@pytest.mark.parametrize(
    "answer",
    [
        "ALLOW",
        "ALLOW scope=o999",
        "ALLOW scope=o0",
        "ALLOW scope=o1,o2",
        "ALLOW scope=o1 evidence=e1",
        "ALLOW scope=o1 operator",
    ],
)
def test_scope_citation_contract(answer):
    c = case()
    assert c.verify(answer, forked=True, scope_eligible=frozenset(c.paths))[1]


@pytest.mark.parametrize("barrier", ["operator_only", "unverifiable", "ask", "peer", "mixed"])
def test_final_scope_barriers(barrier):
    c = case()
    eligible = frozenset(c.paths)
    if barrier == "operator_only":
        c.operator_only = eligible
    elif barrier == "unverifiable":
        c.unverifiable = ["opaque"]
    elif barrier == "ask":
        c.operator.words = [
            prov.OperatorWords("operator, answer to ask", "approved", ask_id="a-ab")
        ]
    elif barrier == "peer":
        c.operator.words = []
    else:
        c.paths.append("/fixture/other")
    assert c.verify("ALLOW scope=o1", forked=True, scope_eligible=eligible)[1]


def test_row_ids_preserve_positions_and_trust_labels():
    text = render_brief(
        JudgeBrief(
            "bash",
            "rm -rf x",
            "x",
            operator_words=(
                ("operator, answer to ask a-ab", "yes"),
                ("operator, direct", "cleanup"),
            ),
        )
    )
    assert 'source="operator, direct" id="o2"' in text
    assert 'id="o1"' not in text


def git(cwd, *args):
    return subprocess.run(
        ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    state = home / ".local-operator"
    state.mkdir()
    (state / "sessions" / "aaaaaaaaaaaa").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    repo = home / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    (repo / "tracked").write_text("fixture\n")
    (repo / ".gitignore").write_text(".worktrees/\nprivate/\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "fixture")
    wt = repo / ".worktrees" / "done"
    git(repo, "worktree", "add", "-q", "-b", "completed", str(wt))
    # Known FOREIGN creator, not this session's evidence/LOW shortcut.
    prov.write_records(
        [(str(wt.resolve()), "worktree")],
        owners=["bbbbbbbbbbbb"],
        creator="bbbbbbbbbbbb",
        command=f"git worktree add {wt}",
        state_root=state,
    )
    registry.run_dir().mkdir(parents=True)
    # Deterministic external boundaries; git registration/status and strict
    # ledger/registry inventories remain real. No live provider or session store.
    monkeypatch.setattr(prov, "_live_cwd_inside", lambda _: False)
    monkeypatch.setattr(prov, "_github_repo", lambda _: "fixture/repo")
    monkeypatch.setattr(prov, "_gh_prs", lambda *_: [])
    return home, repo, wt


async def guard(
    fixture,
    answer="ALLOW scope=o1",
    *,
    forked=True,
    text=None,
    command=None,
    peer=False,
    mutate=None,
):
    home, repo, wt = fixture
    text = text or "I give full approval to delete completed worktrees"
    rows = [Message.user(text)] if not peer else []
    ctx = CallContext(
        cwd=str(repo),
        env={
            "HOME": str(home),
            "LOCAL_OPERATOR_CONFIG_DIR": str(home / ".local-operator"),
        },
        scratchpad=None,
        session_id="aaaaaaaaaaaa",
        operator=prov.operator_brief(rows, session_id="aaaaaaaaaaaa"),
    )
    prompts = []

    async def provider(system, prompt):
        prompts.append((system, prompt))
        if mutate:
            mutate(ctx)
        return answer

    outcome = await check_tool_call(
        "bash",
        {"command": command or f"git worktree remove --force {wt}"},
        ctx,
        settings=GuardSettings(),
        judge=None if forked else provider,
        forked_judge=provider if forked else None,
    )
    assert outcome is not None
    assert wt.is_dir(), "guard-only proof must never execute deletion"
    return outcome, prompts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    ["dirty", "live", "open", "head", "branch", "registration", "revoked", "held"],
)
async def test_scope_rechecks_changes_during_await(fixture, monkeypatch, change):
    home, repo, wt = fixture

    def mutate(ctx):
        if change == "dirty":
            (wt / "tracked").write_text("unfinished during judge")
        elif change == "live":
            record = SessionRecord(
                pid=os.getpid(),
                kind="exec",
                session_id="bbbbbbbbbbbb",
                conversation_name="synthetic owner",
                cwd=str(wt),
                model_label="fixture",
                control_port=0,
                control_key="synthetic-not-a-secret",
            )
            (registry.run_dir() / "synthetic.json").write_text(json.dumps(asdict(record)))
        elif change == "open":
            monkeypatch.setattr(
                prov, "_gh_prs", lambda *_: [prov.ForgeChange(1, "open", "2026-11-12")]
            )
        elif change == "head":
            git(
                wt,
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "--allow-empty",
                "-qm",
                "new unfinished work",
            )
            assert prov.worktree_state(str(wt)) == "clean"
        elif change == "branch":
            git(wt, "switch", "-qc", "unrelated")
        elif change == "registration":
            old = Path(git(wt, "rev-parse", "--absolute-git-dir"))
            new = old.with_name("replacement-registration")
            old.rename(new)
            (wt / ".git").write_text(f"gitdir: {new}\n")
            assert prov.worktree_state(str(wt)) == "clean"
        elif change == "revoked":
            ctx.operator.words.append(OperatorWords("operator, direct", "Stop; leave this tree"))
        elif change == "held":
            Path(git(wt, "rev-parse", "--absolute-git-dir"), "locked").write_text("held")

    outcome, prompts = await guard(fixture, mutate=mutate)
    assert str(wt) in prompts[0][1], "judge must see the initially eligible tree"
    assert outcome.blocked, (change, outcome.warnings)
    assert not any("Completed-worktree cleanup authorized" in w for w in outcome.warnings)
    assert wt.is_dir()
    if change == "dirty":
        assert (wt / "tracked").read_text() == "unfinished during judge"
    print(f"DURING-JUDGE {change}: blocked={outcome.blocked}; tree_present={wt.is_dir()}")


@pytest.mark.asyncio
async def test_actual_guard_before_after_and_warning_without_deletion(fixture):
    before, _ = await guard(fixture, "ALLOW operator")
    assert before.blocked and "could not be verified" in before.text
    assert prov.worktree_state(str(fixture[2])) == "clean"
    assert prov._running_owners(str(fixture[2]), "aaaaaaaaaaaa", strict=True) == ()
    after, prompts = await guard(fixture)
    assert not after.blocked, (after.text, prompts)
    assert any("Completed-worktree cleanup authorized" in w for w in after.warnings)
    assert "Scoped cleanup eligibility" in prompts[0][1]
    assert 'id="o1"' in prompts[0][1]
    print("BEFORE:", before.text)
    print(
        "AFTER: blocked=",
        after.blocked,
        "warnings=",
        after.warnings,
        "tree_present=",
        fixture[2].is_dir(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "barrier",
    [
        "dirty",
        "ignored",
        "open",
        "live",
        "unknown-owner",
        "missing-registry",
        "malformed-registry",
        "unknown-cwd",
        "locked",
        "peer",
        "primary",
        "parent",
        "glob",
        "mixed",
        "symlink",
        "protected",
        "fallback",
        "plain",
    ],
)
async def test_actual_guard_scope_negative_controls(fixture, monkeypatch, barrier):
    home, repo, wt = fixture
    command = None
    answer = "ALLOW scope=o1"
    if barrier == "dirty":
        (wt / "tracked").write_text("unfinished")
    elif barrier == "ignored":
        (wt / "private").mkdir()
        (wt / "private" / "notes").write_text("unknown data")
    elif barrier == "open":
        monkeypatch.setattr(prov, "_gh_prs", lambda *_: [prov.ForgeChange(1, "open", "2026-11-12")])
    elif barrier == "live":
        record = SessionRecord(
            pid=os.getpid(),
            kind="exec",
            session_id="bbbbbbbbbbbb",
            conversation_name="synthetic owner",
            cwd=str(wt),
            model_label="fixture",
            control_port=0,
            control_key="synthetic-not-a-secret",
        )
        (registry.run_dir() / "synthetic.json").write_text(json.dumps(asdict(record)))
    elif barrier == "unknown-owner":
        monkeypatch.setattr(prov, "_connect", lambda *a, **kw: None)
    elif barrier == "missing-registry":
        monkeypatch.setattr(registry, "run_dir", lambda *a, **kw: home / "absent")
    elif barrier == "malformed-registry":
        (registry.run_dir() / "malformed.json").write_text("{")
    elif barrier == "unknown-cwd":
        monkeypatch.setattr(prov, "_live_cwd_inside", lambda _: None)
    elif barrier == "locked":
        Path(git(wt, "rev-parse", "--absolute-git-dir"), "locked").write_text("held")
    elif barrier in ("primary", "parent", "glob", "mixed", "symlink", "protected"):
        target = {
            "primary": str(repo),
            "parent": str(wt.parent),
            "glob": str(wt.parent / "*"),
            "mixed": f"{wt} {home / 'other'}",
            "symlink": str(repo / "alias"),
            "protected": str(home / ".local-operator"),
        }[barrier]
        if barrier == "symlink":
            (repo / "alias").symlink_to(wt, target_is_directory=True)
        command = f"rm -rf {target}"
    elif barrier == "fallback":
        answer = ContextFreeAnswer(answer)
    result, _ = await guard(
        fixture, answer, command=command, peer=barrier == "peer", forked=barrier != "plain"
    )
    if barrier in ("parent", "glob") and not result.blocked:
        # Existing foreign LOW shortcuts are outside this consent-only patch.
        assert not any("scope" in warning for warning in result.warnings)
        assert not result.rule
    else:
        assert result.blocked, (barrier, result)
    print(f"NEGATIVE {barrier}: {result.text}; tree_present={wt.is_dir()}")


def test_strict_inventory_is_opt_in_and_reports_missing_malformed_unreadable(fixture, monkeypatch):
    home, _, _ = fixture
    assert registry.scan(reap=False, strict=True) == []  # positively read empty directory
    root = registry.run_dir()
    bad = root / "bad.json"
    bad.write_text("{")
    assert registry.scan(reap=False) == []
    with pytest.raises(ValueError):
        registry.scan(reap=False, strict=True)
    original = Path.read_text

    def unreadable(self, *a, **kw):
        if self == bad:
            raise PermissionError("fixture unreadable record")
        return original(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", unreadable)
    with pytest.raises(PermissionError):
        registry.scan(reap=False, strict=True)
    monkeypatch.setattr(registry, "run_dir", lambda *a, **kw: home / "missing")
    assert registry.scan(reap=False) == []
    with pytest.raises(FileNotFoundError):
        registry.scan(reap=False, strict=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [False, True])
async def test_internal_context_free_fallback_cannot_claim_scope(fixture, empty):
    from local_operator.session.session import Session

    async def broken(*a, **kw):
        raise RuntimeError("synthetic fork unavailable")

    async def plain(*a):
        return "ALLOW scope=o1"

    fake = SimpleNamespace(
        _context=SimpleNamespace(messages=[] if empty else [Message.user("hello")]),
        complete_aside=broken,
        _guard_judge=plain,
    )
    from typing import Any

    fork: Any = Session._guard_judge_forked
    answer = await fork(fake, "SYSTEM", "BRIEF")
    assert isinstance(answer, ContextFreeAnswer)
    outcome, _ = await guard(fixture, answer)
    assert outcome.blocked


@pytest.mark.asyncio
async def test_two_exact_completed_trees_share_one_scope(fixture):
    home, repo, wt = fixture
    other = repo / ".worktrees" / "also-done"
    git(repo, "worktree", "add", "-q", "-b", "also-completed", str(other))
    prov.write_records(
        [(str(other.resolve()), "worktree")],
        owners=["cccccccccccc"],
        creator="cccccccccccc",
        command=f"git worktree add {other}",
        state_root=home / ".local-operator",
    )
    outcome, _ = await guard(fixture, command=f"rm -rf {wt} {other}")
    assert not outcome.blocked
    assert any("scope o1" in w and str(other) in w for w in outcome.warnings)
    assert wt.is_dir() and other.is_dir()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "instruction,reason",
    [
        ("Look at completed worktrees", "inspection is not consent"),
        (
            'Someone wrote "I approve deleting completed worktrees"',
            "quotation is not consent",
        ),
        ("Delete completed worktrees in repo B only", "location mismatch"),
        ("Delete completed worktrees, but keep done", "exclusion"),
        ("Stop; leave this tree", "revoked"),
        ("Delete completed worktrees", "owner hold in prefix"),
        ("Delete completed worktrees", "completion unknown in prefix"),
    ],
)
async def test_semantic_block_is_honored(fixture, instruction, reason):
    # Scripted provider responses verify the boundary, NOT live-model reasoning.
    outcome, _ = await guard(fixture, answer=f"BLOCK: {reason}", text=instruction)
    assert outcome.blocked and reason in outcome.text


def test_strict_owner_requires_current_generation(fixture):
    _, _, wt = fixture
    assert prov._running_owners(str(wt), "aaaaaaaaaaaa", strict=True) == ()
    conn = prov._connect(prov.ledger_path(), create=False)
    assert conn is not None
    try:
        conn.execute("UPDATE created SET stamp = 0")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(OSError, match="attribution unavailable"):
        prov._running_owners(str(wt), "aaaaaaaaaaaa", strict=True)


def test_strict_registry_does_not_hide_unreadable_directory(fixture, monkeypatch):
    root = registry.run_dir()
    original = Path.iterdir

    def unreadable(self):
        if self == root:
            raise PermissionError("synthetic unreadable inventory")
        return original(self)

    monkeypatch.setattr(Path, "iterdir", unreadable)
    with pytest.raises(PermissionError):
        registry.scan(reap=False, strict=True)
