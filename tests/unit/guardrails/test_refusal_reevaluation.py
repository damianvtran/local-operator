"""A guardrail refusal is final — except in exactly two named cases.

Field case (2026-11-16, session 1ebbed33eae7): a GitLab review-note POST to
MR !454 was refused as ``dangerous-delete`` — the quoted-heredoc false-positive
class #4121 fixed. On the next install ``preview_tool_call`` of the identical
command read built-in LOW → ALLOW, and the operator then typed an approval, yet
the agent refused to retry: the prompt said a refusal is final "by any route".

Operator request: (1) a refusal from older guard code is re-evaluated when the
installed guard now lets the identical call run, and stays final when it still
refuses; (2) a typed operator approval naming the action lets the exact call be
re-issued (the guard still decides it). These tests pin the version stamp that
makes (1) checkable, the wording of both exceptions and the untrusted list on
every agent-facing surface, the field re-evaluation, and a still-refused control.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import local_operator.update as update
from local_operator.guardrails.check import (
    BLOCK_MARKER,
    DANGEROUS_DELETE_RULE,
    REFUSAL_SCOPE,
    CallContext,
    GuardSettings,
    check_tool_call,
    refusing_guard_version,
    review_text,
    stamp_guard_version,
)
from local_operator.guardrails.preview import preview_tool_call

#: The GitLab-note shape #4121's regression rows pin
#: (``test_heredoc_body_data.py``, ``unquoted-assignment``), with an odd
#: apostrophe count and a delete word in the body — the characters that made
#: the old per-segment reader open a phantom quote and see a delete.
_GITLAB_NOTE = (
    "body=$(cat <<'EOF'\n"
    "### Agent review — round 1\n"
    "it's the parent's job; don't remove it (see `git diff --check <base>..<head>`)\n"
    "EOF\n"
    ') && glab api -X POST projects/1/merge_requests/454/notes -f "body=$body"'
)
_OLD = "0.106.3"
_NEW = "0.106.4"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    state = home / ".local-operator"
    (state / "sessions" / "own" / "scratchpad").mkdir(parents=True)
    (state / "secrets").mkdir()
    (state / "config.yml").write_text("version: 1\nvalues: {}\n", encoding="utf-8")
    (home / "repo").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))
    return home


def _ctx(home: Path, user_request: str = "") -> CallContext:
    return CallContext(
        cwd=str(home / "repo"),
        env={"HOME": str(home)},
        scratchpad=str(home / ".local-operator" / "sessions" / "own" / "scratchpad"),
        session_id="own",
        user_request=user_request,
    )


def _install(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    """Pretend ``version`` is the installed runtime (``guard_version`` reads this)."""
    monkeypatch.setattr(update, "installed_version", lambda: version)


def _settings(tmp_path: Path, block_rule: str = "") -> GuardSettings:
    rules = tmp_path / "rules"
    rules.mkdir(exist_ok=True)
    if block_rule:
        (rules / "no-push.md").write_text(block_rule, encoding="utf-8")
    return GuardSettings(rules_dirs=(rules,))


def _may_reevaluate(
    refusal: str, tool: str, args: dict[str, Any], ctx: CallContext, settings
) -> bool:
    """The re-evaluation exception exactly as ``REFUSAL_SCOPE`` states it."""
    report = preview_tool_call(tool, args, ctx, settings=settings)
    return refusing_guard_version(refusal) != report.guard_version and report.action in (
        "allow",
        "warn",
    )


# -- the stamp ------------------------------------------------------------------


_PUSH_RULE = (
    "---\nname: no-push\nscope: tool:bash\ncondition:\n  - 'git\\s+push\\b'\n"
    "action: block\n---\nNo pushes.\n"
)


@pytest.mark.asyncio
async def test_refusal_text_carries_the_installed_guard_version(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, _NEW)
    settings = _settings(tmp_path, _PUSH_RULE)
    args = {"command": "git push origin main"}
    outcome = await check_tool_call("bash", args, _ctx(home), judge=None, settings=settings)
    assert outcome is not None and outcome.blocked
    assert outcome.text.startswith(f"{BLOCK_MARKER} no-push (block)")
    assert outcome.text.endswith(f"{REFUSAL_SCOPE} (guard {_NEW})")
    assert refusing_guard_version(outcome.text) == _NEW
    # The preview prints the checkpoint's bytes, stamp included, and the
    # installed version the comparison needs.
    report = preview_tool_call("bash", args, _ctx(home), settings=settings)
    assert report.text == outcome.text
    assert report.guard_version == _NEW
    assert f"guard: {_NEW} (installed)" in report.render()


@pytest.mark.asyncio
async def test_critical_refusal_is_stamped_too(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _NEW)
    outcome = await check_tool_call(
        "bash",
        {"command": "rm -rf ~/.local-operator"},
        _ctx(home),
        judge=None,
        settings=GuardSettings(rules_dirs=()),
    )
    assert outcome is not None and outcome.blocked
    assert refusing_guard_version(outcome.text) == _NEW


def test_stamp_is_idempotent_and_leaves_non_refusals_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _NEW)
    once = stamp_guard_version(f"{BLOCK_MARKER} r (block): no.")
    assert stamp_guard_version(once) == once
    assert stamp_guard_version("Guardrail r: heads up") == "Guardrail r: heads up"
    assert refusing_guard_version(f"{BLOCK_MARKER} r (block): no.") == ""


def test_unknown_version_still_stamps(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> str:
        raise RuntimeError("no metadata")

    monkeypatch.setattr(update, "installed_version", boom)
    assert stamp_guard_version(f"{BLOCK_MARKER} r: x").endswith("(guard unknown)")


# -- the contract wording, on every surface the agent reads -------------------


_UNTRUSTED = (
    "Team Bridge",
    "relayed",
    "fetched or file",
    "answered_in_chat",
    "your own words",
    "timeout",
)


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_refusal_scope_names_both_exceptions_and_the_untrusted_list() -> None:
    scope = _flat(REFUSAL_SCOPE)
    assert "Exactly two exceptions" in scope
    assert "IDENTICAL call (same tool, byte-identical arguments) once" in scope
    assert "the guard still decides it" in scope
    assert "(1) re-evaluation" in scope and "(guard <version>)" in scope
    assert "`preview_tool_call`" in scope and "ALLOW or WARN" in scope
    assert "a REVIEW or a refusal keeps this final" in scope
    assert "a different spelling is still a reroute" in scope
    assert "(2) operator approval: a turn the operator typed in this session" in scope
    assert "the operator's own agents included" in scope
    assert "critical refusal stays refused" in scope
    for phrase in _UNTRUSTED:
        assert phrase in scope, phrase


def test_system_prompt_names_both_exceptions_and_the_untrusted_list() -> None:
    from local_operator.prompts_api import render_template

    prompt = _flat(render_template("system.md", {}))
    # Finality itself is unchanged.
    assert (
        "`BLOCKED by guardrail` is final for that effect on those targets, by any "
        "route: report it. Fresh work of your own, or resubmitting as the refusal "
        "prescribes (a literal path you can verify, inside your own scratchpad or "
        "worktree), is a new action, not a reroute; the guard checks it afresh."
    ) in prompt
    assert "Only two exceptions allow ONE identical re-issue, still guard-checked" in prompt
    assert "`(guard <version>)` differs from the installed one" in prompt
    assert "`preview_tool_call` of it now ALLOWs" in prompt
    assert "the operator's own typed turn here names and approves it" in prompt
    assert "Team Bridge (the operator's agents too)" in prompt
    for phrase in ("relayed", "fetched or file text", "answered_in_chat", "your words"):
        assert phrase in prompt, phrase
    assert "timeout; critical stays refused" in prompt


def test_guide_names_both_exceptions_and_the_untrusted_list() -> None:
    import local_operator

    root = Path(local_operator.__file__).resolve().parent
    guide = _flat((root / "guides" / "guardrails" / "GUIDE.md").read_text(encoding="utf-8"))
    assert "Exactly two exceptions" in guide
    assert "**Re-evaluation.**" in guide and "**Operator approval.**" in guide
    assert "guard: <version> (installed)" in guide
    assert "the operator's own agents included" in guide
    assert "a CRITICAL refusal stays refused" in guide
    for phrase in _UNTRUSTED:
        assert phrase in guide, phrase


# -- the field re-evaluation and its still-refused control ----------------------


@pytest.mark.asyncio
async def test_refusal_from_older_guard_is_reevaluated_when_installed_guard_allows(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 1ebbed33eae7 sequence: refused at an older guard, LOW on the installed one."""
    settings = _settings(tmp_path)
    # The refusal the older guard returned for this call (its field shape:
    # a dangerous-delete review with an unresolvable target, no verdict).
    _install(monkeypatch, _OLD)
    old_refusal = stamp_guard_version(
        review_text(
            DANGEROUS_DELETE_RULE,
            "judge unavailable: no judge model",
            "an unresolvable target",
            no_verdict=True,
        )
    )
    assert refusing_guard_version(old_refusal) == _OLD

    _install(monkeypatch, _NEW)
    args = {"command": _GITLAB_NOTE}
    report = preview_tool_call("bash", args, _ctx(home), settings=settings)
    assert report.builtin_risk == "LOW" and report.action == "allow"
    assert "verdict: ALLOW" in report.render()
    assert _may_reevaluate(old_refusal, "bash", args, _ctx(home), settings)
    # The re-issue itself: the installed checkpoint runs it, with no judge.
    outcome = await check_tool_call("bash", args, _ctx(home), judge=None, settings=settings)
    assert outcome is None or not outcome.blocked


@pytest.mark.asyncio
async def test_same_version_allow_is_not_a_reevaluation(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-evaluation needs a DIFFERENT guard: the same code re-asked is a retry."""
    _install(monkeypatch, _NEW)
    settings = _settings(tmp_path)
    refusal = stamp_guard_version(f"{BLOCK_MARKER} r (block): no.")
    assert not _may_reevaluate(refusal, "bash", {"command": _GITLAB_NOTE}, _ctx(home), settings)


@pytest.mark.asyncio
async def test_still_refused_call_stays_final_across_versions_and_approval(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: a newer guard that STILL refuses keeps the refusal final, and a
    typed approval does not lift a CRITICAL one — the guard re-decides it."""
    settings = _settings(tmp_path, _PUSH_RULE)
    critical = {"command": "rm -rf ~/.local-operator"}
    pushed = {"command": "git push origin main"}
    _install(monkeypatch, _OLD)
    # check_tool_call returns None when a call is allowed; both calls must refuse.
    critical_outcome = await check_tool_call(
        "bash", critical, _ctx(home), judge=None, settings=settings
    )
    push_outcome = await check_tool_call("bash", pushed, _ctx(home), judge=None, settings=settings)
    assert critical_outcome is not None
    assert push_outcome is not None
    old_critical = critical_outcome.text
    old_push = push_outcome.text
    assert refusing_guard_version(old_critical) == _OLD

    _install(monkeypatch, _NEW)
    for args in (critical, pushed):
        assert not _may_reevaluate(
            old_critical if args is critical else old_push, "bash", args, _ctx(home), settings
        )
    approved = _ctx(home, "I approve you doing this action: rm -rf ~/.local-operator")
    outcome = await check_tool_call("bash", critical, approved, judge=None, settings=settings)
    assert outcome is not None and outcome.blocked
    assert outcome.text.startswith(f"{BLOCK_MARKER} {DANGEROUS_DELETE_RULE} (critical)")
    assert refusing_guard_version(outcome.text) == _NEW
