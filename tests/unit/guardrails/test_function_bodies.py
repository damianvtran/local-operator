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
    # Runaway fan-out of a body whose state changes each call: past the call
    # budget, an already-walked body is skipped with an opaque reason.
    (
        "fan-out-budget",
        'a(){ n="$n."; }; b(){ a; a; }; c(){ b; b; }; d(){ c; c; }; e(){ d; d; }; '
        "g(){ e; e; }; h(){ g; g; }; i(){ h; h; }; i",
        Risk.HIGH,
    ),
    # Review round 1 (security R1-R6, QA) on e3ed0b84. Every row below ranked
    # wrongly there; the comment names the finding.
    # R1: a definition that does not start its lex piece.
    ("r1-brace-group", "{ f(){ rm -rf ~; }; f; }", Risk.CRITICAL),
    ("r1-subshell", "(f(){ rm -rf ~; }; f)", Risk.CRITICAL),
    ("r1-substitution", "x=$(f(){ rm -rf ~; }; f)", Risk.CRITICAL),
    ("r1-then", "if true; then f(){ rm -rf ~; }; f; fi", Risk.CRITICAL),
    ("r1-else", "if x; then f(){ echo a; }; else f(){ rm -rf ~; }; fi; f", Risk.CRITICAL),
    # R2: a redefinition that may not take effect keeps the earlier body.
    ("r2-and", "f(){ rm -rf ~; }; false && f(){ :; }; f", Risk.CRITICAL),
    ("r2-or", "f(){ rm -rf ~; }; true || f(){ :; }; f", Risk.CRITICAL),
    ("r2-test-and", 'f(){ rm -rf ~; }; [ -n "$X" ] && f(){ :; }; f', Risk.CRITICAL),
    ("r2-background", "f(){ rm -rf ~; }; f(){ :; } & f", Risk.CRITICAL),
    ("r2-pipe", "f(){ rm -rf ~; }; f(){ :; } | cat; f", Risk.CRITICAL),
    ("r2-substitution", 'f(){ rm -rf ~; }; echo "$(f(){ :; })"; f', Risk.CRITICAL),
    (
        "r2-in-if",
        f'L=/; f(){{ L="{SP}/x"; }}; if true; then f(){{ :; }}; fi; f; rm -rf "$L"',
        Risk.CRITICAL,
    ),
    # Candidates that leave different state: what follows cannot know which ran.
    (
        "r2-state-differs",
        f'L="{SP}/x"; f(){{ L=/; }}; false && f(){{ :; }}; f; rm -rf "$L"',
        Risk.HIGH,
    ),
    # R3: repeated calls of a body that changes nothing spend no budget, and
    # padding cannot hide a destructive body past the budget.
    ("r3-log-65", 'log(){ echo "$@"; }; ' + "; ".join(["log hi"] * 65), Risk.LOW),
    (
        "r3-log-80-lines",
        'log(){ echo "$@"; }\n' + "\n".join(f"log step{i}" for i in range(80)),
        Risk.LOW,
    ),
    (
        "r3-fan-out-inert",
        "a(){ :; }; b(){ a; a; }; c(){ b; b; }; d(){ c; c; }; e(){ d; d; }; "
        "g(){ e; e; }; h(){ g; g; }; i(){ h; h; }; i",
        Risk.LOW,
    ),
    ("r3-padding", "; ".join(["g(){ :; }"] + ["g"] * 70) + "; f(){ rm -rf ~; }; f", Risk.CRITICAL),
    ("r3-repeated-delete", "f(){ rm -rf ./build; }; " + "; ".join(["f"] * 70), Risk.LOW),
    # R4: ``trap``/``alias`` naming one function call it, pinned.
    (
        "r4-trap-cleanup",
        'set -e\ntmp=$(mktemp -d)\ncleanup() { rm -rf "$tmp"; }\n'
        'trap cleanup EXIT\necho hi > "$tmp/a"',
        Risk.LOW,
    ),
    ("r4-trap-pad", f'f(){{ rm -rf "{SP}/x"; }}; trap f EXIT', Risk.LOW),
    ("r4-alias-pad", f'f(){{ rm -rf "{SP}/x"; }}; alias g=f; g', Risk.LOW),
    ("r4-trap-before-definition", "trap cleanup EXIT; cleanup(){ rm -rf ~; }", Risk.CRITICAL),
    ("r4-alias-before-definition", "alias g=f; f(){ rm -rf ~; }; g", Risk.CRITICAL),
    # Non-literal actions still reach every body (fail closed).
    ("r4-trap-variable", f'f(){{ rm -rf "{SP}/x"; }}; trap "$a" EXIT', Risk.HIGH),
    ("r4-trap-script", "f(){ rm -rf ~; }; trap 'f; g' EXIT", Risk.CRITICAL),
    ("r4-trap-reset", "f(){ rm -rf ~; }; trap - EXIT", Risk.LOW),
    # R6: a loop body can call a definition made later in it.
    ("r6-loop-carried", "for i in 1 2; do f; f(){ rm -rf ~; }; done", Risk.CRITICAL),
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


def test_repeated_large_body_stays_fast(ctx: CallContext) -> None:
    """R5 (review round 1): 64 calls of a 1,500-statement body took 18.8 s.

    Calls from a state already walked, whose walk changed nothing, are not
    walked again (``_call_memo_key``), so this costs about one body walk. The
    bound is loose (a slow CI runner), far under the 18.8 s it replaces.
    """
    import time

    assert ctx.scratchpad is not None
    body = "; ".join(f"echo line{i}" for i in range(1500))
    command = f"f(){{ {body}; }}; " + "; ".join(["f"] * 64) + "; rm -rf ~"
    started = time.monotonic()
    verdict = classify_bash(
        command, cwd=ctx.cwd, env=ctx.env, scratchpad=ctx.scratchpad, session_id=SID
    )
    assert time.monotonic() - started < 8.0
    assert verdict.risk == Risk.CRITICAL
