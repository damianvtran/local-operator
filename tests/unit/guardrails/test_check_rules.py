"""User rules at the real pre-call gate: ``check_tool_call`` and what it returns.

The loader's own tests (``test_rules.py``) prove parsing; this pins the OUTCOME
a user gets when a rule file is present, through the function ``harness/loop.py``
calls — not a re-implementation — so a change in the order between the built-in
check and the user rules, or in how a rule's verdict reaches the caller, shows
up here. A ``block`` must become the call's own refusal text; a ``warn`` must
leave the call running and ride along with its result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.guardrails.check import (
    BLOCK_MARKER,
    CallContext,
    GuardSettings,
    check_tool_call,
)


@pytest.fixture
def rules_dir(tmp_path: Path) -> Path:
    """The default rules location for this call, as the loop wires it."""
    directory = tmp_path / "state" / "rules"
    directory.mkdir(parents=True)
    return directory


def _rule(name: str, action: str, condition: str, body: str, question: str = "") -> str:
    extra = f"question: {question}\n" if question else ""
    return (
        f"---\nname: {name}\nscope: tool:bash\ncondition: ['{condition}']\n"
        f"{extra}action: {action}\n---\n{body}\n"
    )


def _context(tmp_path: Path) -> CallContext:
    return CallContext(
        cwd=str(tmp_path),
        env={},
        scratchpad=str(tmp_path / "scratchpad"),
        session_id="own",
        user_request="tidy up the feature branch",
    )


def _settings(rules_dir: Path) -> GuardSettings:
    return GuardSettings(rules_dirs=(rules_dir,))


@pytest.mark.asyncio
async def test_block_rule_refuses_the_call(rules_dir: Path, tmp_path: Path) -> None:
    """A block's verdict is the call's result: marker, rule name and body."""
    (rules_dir / "no-force-push.md").write_text(
        _rule("no-force-push", "block", r"git\s+push\b.*--force", "Open a PR instead."),
        encoding="utf-8",
    )

    outcome = await check_tool_call(
        "bash",
        {"command": "git push --force origin main"},
        _context(tmp_path),
        judge=None,
        settings=_settings(rules_dir),
    )

    assert outcome is not None and outcome.blocked is True
    assert outcome.text.startswith(f"{BLOCK_MARKER} no-force-push (block)")
    assert "Open a PR instead." in outcome.text
    assert outcome.rule == "no-force-push"


@pytest.mark.asyncio
async def test_warn_rule_lets_the_call_run_and_carries_the_warning(
    rules_dir: Path, tmp_path: Path
) -> None:
    """A warn does not refuse: the call proceeds and the note is attached."""
    (rules_dir / "pushing.md").write_text(
        _rule("pushing", "warn", r"git\s+push\b", "Say which branch and where it goes."),
        encoding="utf-8",
    )

    outcome = await check_tool_call(
        "bash",
        {"command": "git push origin feature"},
        _context(tmp_path),
        judge=None,
        settings=_settings(rules_dir),
    )

    assert outcome is not None and outcome.blocked is False
    assert outcome.warnings == ["Guardrail pushing: Say which branch and where it goes."]


@pytest.mark.asyncio
async def test_block_wins_over_warn_when_both_match(rules_dir: Path, tmp_path: Path) -> None:
    """First refusal wins: a warn must never mask a matching block on the same call."""
    (rules_dir / "a-warn.md").write_text(
        _rule("a-warn", "warn", r"git\s+push\b", "A warning."), encoding="utf-8"
    )
    (rules_dir / "b-block.md").write_text(
        _rule("b-block", "block", r"git\s+push\b", "Not this one."), encoding="utf-8"
    )

    outcome = await check_tool_call(
        "bash",
        {"command": "git push origin feature"},
        _context(tmp_path),
        judge=None,
        settings=_settings(rules_dir),
    )

    assert outcome is not None and outcome.blocked is True
    assert "b-block" in outcome.text
    assert "A warning." not in outcome.text
