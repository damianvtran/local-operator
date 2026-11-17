"""The delete guard walks a shell function's body where the function is CALLED.

Before this, ``f(){ rm -rf ~; }; f`` ranked LOW: ``f(){`` was read as a program
and ``rm -rf ~`` as its arguments, so any destructive command could hide behind
a function name. PR #4147 tried walking bodies at their DEFINITION inside the
main pass and was reverted after QA Q5 (a body's assignment, even a ``local``
or one in a never-called function, replaced the outer binding and dropped a
CRITICAL to LOW) and security R4 (a home-made body-end scan stopped at a ``}``
inside a comment or heredoc). The design under test (``resolve.py``, "Shell
functions") walks bodies at call sites in a separate pass whose ranked facts
are unioned into main's, so it can only raise.

Every row is ranked three ways that must agree: ``classify_bash``,
``preview_tool_call`` and the real ``check_tool_call`` checkpoint. Commands are
strings only; nothing here is executed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.guardrails.check import CallContext, GuardSettings, check_tool_call
from local_operator.guardrails.classify import Risk, classify_bash
from local_operator.guardrails.preview import preview_tool_call

SID = "aaaaaaaaaaaa"
SP = "$LOCAL_OPERATOR_SCRATCHPAD"

#: (id, command, expected risk). Unless noted "guard", each row ranked LOW on
#: main (64f79389), where function bodies were never walked.
ROWS: list[tuple[str, str, Risk]] = [
    # Called functions: CRITICAL stays CRITICAL, both definition forms.
    ("call", "f(){ rm -rf ~; }; f", Risk.CRITICAL),
    ("function-kw", 'function g { rm -rf "$HOME/.local-operator"; }; g', Risk.CRITICAL),
    ("function-kw-parens", "function g() { rm -rf ~; }; g", Risk.CRITICAL),
    ("nested-definition", "outer(){ inner(){ rm -rf ~; }; inner; }; outer", Risk.CRITICAL),
    ("in-if", "f(){ rm -rf ~; }; if true; then f; fi", Risk.CRITICAL),
    ("in-substitution", "f(){ rm -rf ~; }; x=$(f)", Risk.CRITICAL),
    ("shadows-rm", "rm(){ /bin/rm -rf ~; }; rm x", Risk.CRITICAL),
    # A body that only reads its caller's arguments: opaque until pinned.
    ("positional-opaque", 'f(){ rm -rf "$1"; }; f /tmp/x', Risk.HIGH),
    # Defined only, never called: runs nothing (guard: LOW on main too).
    ("defined-only", "f(){ rm -rf ~; }", Risk.LOW),
    ("call-before-definition", "f; f(){ rm -rf ~; }", Risk.LOW),
    # Redefinition: the last definition before the call wins.
    ("redefined-destructive", "f(){ echo hi; }; f(){ rm -rf ~; }; f", Risk.CRITICAL),
    ("redefined-harmless", "f(){ rm -rf ~; }; f(){ echo hi; }; f", Risk.LOW),
    # Pipelines and background jobs still run the body.
    ("pipeline", "f(){ rm -rf ~; }; f | cat", Risk.CRITICAL),
    ("background", "f(){ rm -rf ~; }; f &", Risk.CRITICAL),
    # Subshell bodies.
    ("subshell-body", "f() ( rm -rf ~ ); f", Risk.CRITICAL),
    ("subshell-body-tight", "f()(rm -rf ~); f", Risk.CRITICAL),
    # A ``}`` inside a comment, heredoc or quoted string does not end the body:
    # if it did, ``L=/`` would run at definition and the later contained
    # assignment would win (security R4's shapes, with a literal pad path).
    ("brace-in-comment", f'f(){{ # }}\nL=/; }}; L="{SP}/x"; f; rm -rf "$L"', Risk.CRITICAL),
    (
        "brace-in-heredoc",
        f'f(){{ cat <<X\n}}\nX\nL=/; }}; L="{SP}/x"; f; rm -rf "$L"',
        Risk.CRITICAL,
    ),
    ("brace-in-quotes", f'f(){{ echo "}}"; L=/; }}; L="{SP}/x"; f; rm -rf "$L"', Risk.CRITICAL),
    # ``local`` does not leak out of the call: ``g`` sees the outer ``/``.
    ("local-scope", f'f(){{ local L="{SP}/x"; }}; g(){{ rm -rf "$L"; }}; L=/; f; g', Risk.CRITICAL),
    ("subshell-scope", f'f() ( L="{SP}/x" ); g(){{ rm -rf "$L"; }}; L=/; f; g', Risk.CRITICAL),
    # Unknown call targets reach every registered body.
    ("indirect-unset", 'f(){ rm -rf ~; }; "$fn"', Risk.CRITICAL),
    ("indirect-pinned", "f(){ rm -rf ~; }; fn=f; $fn", Risk.CRITICAL),
    ("indirect-pad-body", f'f(){{ rm -rf "{SP}/x"; }}; $fn', Risk.HIGH),
    ("eval", 'f(){ rm -rf ~; }; eval "$x"', Risk.CRITICAL),
    ("source", "f(){ rm -rf ~; }; source ./x.sh", Risk.CRITICAL),
    ("exported-to-bash", "f(){ rm -rf ~; }; export -f f; bash -c f", Risk.CRITICAL),
    # A child shell does not inherit an unexported function (guard).
    ("unexported-bash", "f(){ rm -rf ~; }; bash -c f", Risk.LOW),
    # Recursion and runaway fan-out fail closed to HIGH, never LOW.
    ("recursion", "f(){ f; }; f", Risk.HIGH),
    ("recursion-destructive", "f(){ rm -rf ./build; f; }; f", Risk.HIGH),
    (
        "fan-out-budget",
        "a(){ :; }; b(){ a; a; }; c(){ b; b; }; d(){ c; c; }; e(){ d; d; }; "
        "g(){ e; e; }; h(){ g; g; }; i(){ h; h; }; i",
        Risk.HIGH,
    ),
    # The own-scratchpad delete keeps working through a function (guard).
    ("scratch-delete", f'f(){{ rm -rf "{SP}/x"; }}; f', Risk.LOW),
    ("harmless-call", "g(){ echo hi; }; g", Risk.LOW),
]


@pytest.fixture
def ctx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CallContext:
    home = (tmp_path / "home").resolve()
    state = home / ".local-operator"
    pad = state / "sessions" / SID / "scratchpad"
    pad.mkdir(parents=True)
    (state / "config.yml").write_text("version: 1\n")
    (home / "repo").mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))
    env = {
        "HOME": str(home),
        "LOCAL_OPERATOR_CONFIG_DIR": str(state),
        "LOCAL_OPERATOR_SCRATCHPAD": str(pad),
        "PATH": "/usr/bin:/bin",
    }
    return CallContext(cwd=str(home / "repo"), env=env, scratchpad=str(pad), session_id=SID)


def _judge(answer: str):
    async def ask(system_prompt: str, brief: str) -> str:
        return answer

    return ask


@pytest.mark.parametrize(("command", "expected"), [r[1:] for r in ROWS], ids=[r[0] for r in ROWS])
def test_function_body_ranks(ctx: CallContext, command: str, expected: Risk) -> None:
    assert ctx.scratchpad is not None
    verdict = classify_bash(
        command, cwd=ctx.cwd, env=ctx.env, scratchpad=ctx.scratchpad, session_id=SID
    )
    assert verdict.risk == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "expected"), [r[1:] for r in ROWS], ids=[r[0] for r in ROWS])
async def test_preview_agrees_with_checkpoint(
    ctx: CallContext, command: str, expected: Risk
) -> None:
    """The preview's rank and the real checkpoint's outcome agree, row by row."""
    report = preview_tool_call(
        "bash", {"command": command}, ctx, settings=GuardSettings(enabled=False)
    )
    assert report.builtin_risk == expected.name
    settings = GuardSettings(rules_dirs=())
    allowed = await check_tool_call(
        "bash", {"command": command}, ctx, judge=_judge("ALLOW"), settings=settings
    )
    denied = await check_tool_call(
        "bash", {"command": command}, ctx, judge=_judge("BLOCK: not asked for"), settings=settings
    )
    if expected == Risk.LOW:
        # A LOW call never reaches the judge: even a denying one cannot stop it.
        assert denied is None or denied.blocked is False
    elif expected == Risk.HIGH:
        # HIGH goes to the judge: its answer decides.
        assert allowed is None or allowed.blocked is False
        assert denied is not None and denied.blocked is True
        assert denied.rule == "dangerous-delete"
    else:
        # CRITICAL is refused before any judge answer.
        assert allowed is not None and allowed.blocked is True
        assert allowed.rule == "dangerous-delete"
