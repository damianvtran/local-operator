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

from local_operator.guardrails import resolve
from local_operator.guardrails.check import CallContext, GuardSettings, check_tool_call
from local_operator.guardrails.classify import Risk, classify_bash
from local_operator.guardrails.preview import preview_tool_call
from local_operator.guardrails.provenance import plan_creations
from local_operator.guardrails.resolve import Creation, Resolution, resolve_command
from local_operator.tools.search_guard import lex


@pytest.mark.parametrize(
    "command",
    [
        "case choice in choice) tee marker ;; esac",
        "case choice in\nchoice)\ntee marker\n;;\nesac",
        "case choice in other) : ;; choice) tee marker ;; esac",
        "case choice in other) tee marker ;; choice) : ;; esac",
        "case choice in choice) case nested in nested) tee marker ;; esac ;; esac",
        "case choice in (other|choice) tee marker ;; esac",
        """case choice in '(' | "a)b" | choice) tee marker ;; esac""",
        r"case choice in a\)b|choice) tee marker ;; esac",
        "emit(){ tee marker; }; case choice in choice) emit ;; esac",
        "emit(){ case choice in choice) tee marker ;; esac; }; emit",
        "emit(){ tee marker; }; case choice in other) emit(){ :; } ;; choice) emit ;; esac",
    ],
)
def test_case_arm_reaches_output_target(command: str, tmp_path: Path) -> None:
    resolution = resolve_command(command, cwd=str(tmp_path), home=str(tmp_path), env={})
    assert [target.path for target in resolution.targets] == [str(tmp_path / "marker")]


def test_case_standalone_terminators_reach_later_arms(tmp_path: Path) -> None:
    command = (
        "case choice in\nfirst) tee first-marker\n;;\n"
        "second) :\n;;\nthird) tee third-marker\n;;\nesac"
    )
    resolution = resolve_command(command, cwd=str(tmp_path), home=str(tmp_path), env={})
    assert [target.path for target in resolution.targets] == [
        str(tmp_path / "first-marker"),
        str(tmp_path / "third-marker"),
    ]


def test_case_standalone_terminator_is_opt_in() -> None:
    text = "printf ';;'\n;;\nprintf done"
    ordinary = lex(text)
    assert ordinary.segments == [("printf ';;'", False, "\n"), ("printf done", False, "")]
    structural = lex(text, keep_empty_case_terminators=True)
    assert structural.segments == [ordinary.segments[0], ("", False, ";;"), ordinary.segments[1]]
    assert structural.bodies == [[], [], []]


@pytest.mark.parametrize("in_called_function", [False, True], ids=["case-walk", "call-walk"])
def test_supplementary_creation_ranks_but_cannot_nominate_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_called_function: bool
) -> None:
    command = "emit(){ :; }; case choice in choice) emit ;; esac"
    primary = Creation(kind="mkdir", path=str(tmp_path / "primary"))
    extra = Creation(kind="mkdir", path=str(tmp_path / "state" / "sessions" / "extra"))
    env = {"HOME": str(tmp_path), "LOCAL_OPERATOR_CONFIG_DIR": str(tmp_path / "state")}

    def walk(
        *_args: object, case_arms: bool, inline_functions: bool = False, **_kwargs: object
    ) -> resolve._Pass:
        created = []
        if case_arms and inline_functions == in_called_function:
            created.append(extra)
        elif not case_arms and not inline_functions:
            created.append(primary)
        return resolve._Pass(Resolution(created=created), engaged=False)

    monkeypatch.setattr(resolve, "_walk_once", walk)

    pending = plan_creations(command, cwd=str(tmp_path), env=env)
    assert pending is not None
    assert pending.exact == [(primary.kind, primary.path, primary.path)]
    assert not pending.links and not pending.patterned
    verdict = classify_bash(command, cwd=str(tmp_path), env=env, scratchpad=None)
    assert verdict.risk == Risk.CRITICAL


SID = "aaaaaaaaaaaa"
SP = "$LOCAL_OPERATOR_SCRATCHPAD"
#: PR #4116 proves freshness for an inline scratch-template cleanup in a
#: straight-line errexit script, not through functions or traps.
_HDR = "set -euo pipefail"
_MK = f'tmp=$(mktemp -d "{SP}/t.XXXXXXXX")'
_CLEAN = 'cleanup(){ rm -rf "$tmp"; }'

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
    # Review round 2 (security N1/N2/N5/N6) on 2553e392. Unless noted
    # "guard", every row below ranked LOW there.
    # N1: a nesting counter missed ``then if``/``do while`` and read a case
    # pattern ``b)`` or ``a\)`` as a closer, so a conditional redefinition
    # REPLACED the destructive body. Bash runs ``rm -rf ~`` in each.
    (
        "n1-if-if",
        'f(){ rm -rf ~; }; if [ -n "$X" ]; then if true; then :; fi; f(){ :; }; fi; f',
        Risk.CRITICAL,
    ),
    (
        "n1-if-for",
        'f(){ rm -rf ~; }; if [ -n "$X" ]; then for i in 1; do :; done; f(){ :; }; fi; f',
        Risk.CRITICAL,
    ),
    (
        "n1-while-while",
        "f(){ rm -rf ~; }; while false; do while false; do :; done; f(){ :; }; done; f",
        Risk.CRITICAL,
    ),
    (
        "n1-case-in-if",
        'f(){ rm -rf ~; }\nif [ -n "$X" ]; then\n  case "$Y" in\n    b)\n      echo b\n'
        "      ;;\n  esac\n  f(){ :; }\nfi\nf",
        Risk.CRITICAL,
    ),
    ("n1-case-arm", 'f(){ rm -rf ~\n}\ncase "$X" in\na)\nf(){ :; }\n;;\nesac\nf', Risk.CRITICAL),
    (
        "n1-escaped-paren",
        'f(){ rm -rf ~; }; if [ -n "$X" ]; then echo a\\); f(){ :; }; fi; f',
        Risk.CRITICAL,
    ),
    # After a closed compound the redefinition is no longer "certain": the
    # earlier body stays a candidate (over-approximation, fail closed).
    ("n1-after-closed-if", "f(){ rm -rf ~; }; if true; then :; fi; f(){ :; }; f", Risk.CRITICAL),
    # N2: a definition in an ``if``/``while``/``until`` test list runs.
    ("n2-if", "if f(){ rm -rf ~; }; then f; fi", Risk.CRITICAL),
    ("n2-while", "while f(){ rm -rf ~; }; do f; break; done", Risk.CRITICAL),
    ("n2-until", "until f(){ rm -rf ~; }; do f; break; done", Risk.CRITICAL),
    ("n2-if-pad", f'if f(){{ rm -rf "{SP}/x"; }}; then f; fi', Risk.LOW),  # guard
    # N6: C-style and nested loops see a definition made later in the body.
    ("n6-c-style-for", "for ((i=0;i<2;i++)); do f; f(){ rm -rf ~; }; done", Risk.CRITICAL),
    (
        "n6-nested-loop",
        "for i in 1 2; do for j in 1; do f; done; f(){ rm -rf ~; }; done",
        Risk.CRITICAL,
    ),
    # N5: alias chains, an alias shadowing a function, a trailing-blank alias.
    ("n5-alias-chain", "f(){ rm -rf ~; }\nalias g=f\nalias h=g\nh", Risk.CRITICAL),
    ("n5-alias-chain-reversed", "f(){ rm -rf ~; }\nalias h=g\nalias g=f\nh", Risk.CRITICAL),
    ("n5-alias-chain-one-command", "f(){ rm -rf ~; }\nalias g=f h=g\nh", Risk.CRITICAL),
    ("n5-alias-shadows-function", "f(){ :; }; g(){ rm -rf ~; }; alias f=g; f", Risk.CRITICAL),
    ("n5-alias-trailing-blank", "f(){ rm -rf ~; }\nalias s='sudo ' g=f\ns g", Risk.CRITICAL),
    ("n5-alias-cycle", "f(){ rm -rf ~; }\nalias g=h h=g\ng", Risk.LOW),  # guard: runs no f
    ("n5-alias-chain-pad", f'f(){{ rm -rf "{SP}/x"; }}\nalias g=f\nalias h=g\nh', Risk.LOW),
    # N7/N8: no template freshness exception for function calls or traps.
    # The checkpoint must consult the judge for an unproved template target,
    # while #4116's original inline cleanup stays LOW.
    ("inline-template-cleanup", f'{_HDR}\n{_MK}\nrm -rf "$tmp"', Risk.LOW),
    ("q1-trap-cleanup", f"{_HDR}\n{_MK}\n{_CLEAN}\ntrap cleanup EXIT", Risk.HIGH),
    ("q1-direct-call", f"{_HDR}\n{_MK}\n{_CLEAN}\ncleanup", Risk.HIGH),
    ("q1-defined-before-trap", f"{_HDR}\n{_CLEAN}\n{_MK}\ntrap cleanup EXIT", Risk.HIGH),
    ("q1-defined-before-call", f"{_HDR}\n{_CLEAN}\n{_MK}\ncleanup", Risk.HIGH),
    # Guards: none of these may reach LOW.
    (
        "q1-body-reassigns",
        f'{_HDR}\n{_MK}\ncleanup(){{ tmp=$HOME; rm -rf "$tmp"; }}\ntrap cleanup EXIT',
        Risk.CRITICAL,
    ),
    # Without a template proof, the climb is unresolved, not a pinned target.
    (
        "q1-body-climbs",
        f'{_HDR}\n{_MK}\ncleanup(){{ rm -rf "$tmp/.."; }}\ntrap cleanup EXIT',
        Risk.HIGH,
    ),
    (
        "q1-body-printf-v",
        f'{_HDR}\n{_MK}\ncleanup(){{ printf -v tmp %s "$HOME"; rm -rf "$tmp"; }}\n'
        "trap cleanup EXIT",
        Risk.HIGH,
    ),
    # The string-form trap is ranked as on main (pre-existing, out of scope).
    ("q1-string-trap", f'{_HDR}\n{_MK}\ntrap "rm -rf $tmp" EXIT', Risk.LOW),
    (
        "q1-conditional-definition",
        f'{_HDR}\n{_MK}\n[ -n "$X" ] && cleanup(){{ rm -rf "$tmp"; }}\ntrap cleanup EXIT',
        Risk.HIGH,
    ),
    ("q1-call-before-mktemp", f"{_HDR}\n{_CLEAN}\ncleanup\n{_MK}", Risk.HIGH),
    # The EXIT trap fires wherever the script stops: a failing second mktemp
    # (or a trap set before the first) leaves ``tmp`` empty, ``"$tmp/x"`` = /x.
    (
        "q1-trap-before-mktemp",
        f'{_HDR}\ncleanup(){{ rm -rf "$tmp/x"; }}\ntrap cleanup EXIT\n{_MK}',
        Risk.HIGH,
    ),
    (
        "q1-second-mktemp-after-trap",
        f'{_HDR}\n{_MK}\ncleanup(){{ rm -rf "$tmp/x"; }}\ntrap cleanup EXIT\n{_MK}',
        Risk.HIGH,
    ),
    # A variable the body reads changes after the trap and back: the trap may
    # have fired in between (``false`` exits under errexit).
    (
        "q1-variable-flips-after-trap",
        f'{_HDR}\n{_MK}\nn=x\ncleanup(){{ rm -rf "$tmp/$n"; }}\ntrap cleanup EXIT\n'
        "n=../..\nfalse\nn=x",
        Risk.HIGH,
    ),
    # A function shadowing ``mktemp`` (or taking arguments) is no proof.
    ("q1-mktemp-shadowed", f"{_HDR}\nmktemp(){{ echo /; }}\n{_MK}\n{_CLEAN}\ncleanup", Risk.HIGH),
    (
        "q1-call-with-args",
        f'{_HDR}\n{_MK}\ncleanup(){{ rm -rf "$tmp/$1"; }}\ncleanup ../..',
        Risk.HIGH,
    ),
    (
        "q1-redefined-after-trap",
        f"{_HDR}\n{_MK}\n{_CLEAN}\ntrap cleanup EXIT\ncleanup(){{ rm -rf ~; }}",
        Risk.CRITICAL,
    ),
    ("q1-other-trap-signal", f"{_HDR}\n{_MK}\n{_CLEAN}\ntrap cleanup INT EXIT", Risk.HIGH),
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (
            'case choice in choice) tee "$LOCAL_OPERATOR_CONFIG_DIR/config.yml" ;; esac',
            Risk.CRITICAL,
        ),
        ('case choice in choice) tee "$LOCAL_OPERATOR_SCRATCHPAD/marker" ;; esac', Risk.LOW),
        ("""case choice in "tee"|"marker) ;; esac") printf '%s' 'tee marker' ;; esac""", Risk.LOW),
        ("""printf '%s' 'case choice in choice) tee marker ;; esac' """, Risk.LOW),
    ],
)
async def test_case_arm_checkpoint(ctx: CallContext, command: str, expected: Risk) -> None:
    report = preview_tool_call(
        "bash", {"command": command}, ctx, settings=GuardSettings(enabled=False)
    )
    assert report.builtin_risk == expected.name
    result = await check_tool_call(
        "bash",
        {"command": command},
        ctx,
        judge=_judge("ALLOW"),
        settings=GuardSettings(rules_dirs=()),
    )
    assert (result is not None and result.blocked) == (expected == Risk.CRITICAL)


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
    # A resolver crash also ranks HIGH, but is not the conservative path we test.
    assert not any(reason.startswith("resolver error:") for reason in verdict.reasons)


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


def test_cleanup_without_errexit_ranks_as_inline(ctx: CallContext) -> None:
    """Q1 guard: without ``set -euo pipefail`` a failed mktemp may leave
    ``tmp`` empty, so #4116 grants no proof. The function form must rank
    exactly as its inline equivalent, never lower."""
    assert ctx.scratchpad is not None

    def rank(command: str) -> Risk:
        return classify_bash(
            command, cwd=ctx.cwd, env=ctx.env, scratchpad=ctx.scratchpad, session_id=SID
        ).risk

    inline = rank(f'{_MK}; rm -rf "$tmp"')
    assert inline != Risk.LOW
    assert rank(f"{_MK}; {_CLEAN}; trap cleanup EXIT") == inline
    assert rank(f"{_MK}; {_CLEAN}; cleanup") == inline


# QUARANTINED, not fixed: the wall-clock bound below is the defect, and it
# blocked security PR #4235's gate. Remove this marker in the same PR that
# replaces the bound with a load-invariant one.
@pytest.mark.skip(
    reason=(
        "QUARANTINED 2026-11-18 flaky-sweep ledger lopdev-E01 'Time-assertion backlog': "
        "wall bound `elapsed < 8.0` crossed on loaded CI runners (#4235 gate jobs "
        "135253226003, 135207196441); owner: time-assertion batch 2; real fix: derive "
        "the bound from counted work (memoised body walks), not wall seconds"
    )
)
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


# Field case (PR #4475's agent, guard 0.112.5 → 0.113.17): a script that posts
# PR comments through a helper function was refused as dangerous-delete with
# "shell function post_and_verify recurses". The helper validated the posted
# URL with ``case "$url" in "https://github.com/$repo/pull/$pr#…"*) ;;``: the
# legacy segmentation read that pattern as a command word holding an
# expansion, so it "reached" every defined function — including the one being
# walked, reported as recursion. Bash cannot run a word ending in an unquoted
# ``)`` (a syntax error outside a case), so such a word is only a pattern.
_CASE_IN_FUNCTION = (
    'f() {\n  case "$1" in\n    "x/$y"*) ;;\n    *) return 1 ;;\n  esac\n}\nf a',
    'f() {\n  case "$1" in\n    x/$y*) ;;\n  esac\n}\nf a',
    'g() { echo ok; }\nf() {\n  g\n  case "$u" in\n    "https://h/$r/pull/$p#c-"*) ;;\n'
    '    *) printf "bad\\n" >&2; return 1 ;;\n  esac\n}\nf a\nf b',
)


@pytest.mark.parametrize("command", _CASE_IN_FUNCTION, ids=["dq-var", "bare-var", "helper"])
def test_case_pattern_with_expansion_is_not_a_dynamic_command_word(command: str) -> None:
    verdict = classify_bash(command, cwd="/tmp", env={"HOME": "/tmp"}, scratchpad=None)
    assert verdict.risk is Risk.LOW, verdict.reasons


# #4646 R1-B1: a SUBSHELL ``( WORD )`` runs WORD and must keep reaching the
# functions. The command name is NOT resolvable (``$(printf f)``), so the
# dynamic-word path is the one exercised; ``cmd=f`` resolves straight to ``f``.
_SUBSHELL_REACHES = (
    'f() { rm -rf ~/x; }\ncmd=$(printf f)\n( "$cmd" )',
    'f() { rm -rf ~/x; }\ncmd=$(printf f)\n( "$cmd")',
    "f() { rm -rf ~/x; }\ncmd=$(printf f)\n($cmd)",
    "f() { rm -rf ~/x; }\n( $(printf f) )",
    'f() { rm -rf ~/x; }\ncmd=$(printf f)\n( ( "$cmd" ) )',
    'f() { rm -rf ~/x; }\ng() { ( "$1" ); }\ng "$(printf f)"',
    'f() { rm -rf ~/x; }\ncmd=$(printf f)\ncase a in\n  a) ( "$cmd" ) ;;\nesac',
    # #4646 R2-B1: a dynamic word AFTER an operator inside a subshell is a
    # ``"$cmd" )`` piece, the same shape as an arm pattern, but not in
    # pattern position.
    'f() { rm -rf ~/x; }\ncmd=$(printf f)\n( true; "$cmd" )',
    'f() { rm -rf ~/x; }\ncmd=$(printf f)\n( echo | "$cmd" )',
    'f() { rm -rf ~/x; }\ncmd=$(printf f)\n( true && "$cmd" )',
    'f() { rm -rf ~/x; }\ncmd=$(printf f)\nx=$( true; "$cmd" )',
    'f() { rm -rf ~/x; }\ncmd=$(printf f)\ncat <( true; "$cmd" )',
    'f() { rm -rf ~/x; }\ng() { ( true; "$1" ); }\ng "$(printf f)"',
    'f() { rm -rf ~/x; }\ncmd=$(printf f)\ncase a in\n  a) ( true; "$cmd" ) ;;\nesac',
)


@pytest.mark.parametrize(
    "command",
    _SUBSHELL_REACHES,
    ids=[
        "spaced",
        "tight-close",
        "bare",
        "subst",
        "nested",
        "in-function",
        "in-arm",
        "after-semicolon",
        "after-pipe",
        "after-and",
        "in-cmd-subst",
        "in-proc-subst",
        "after-semicolon-in-function",
        "after-semicolon-in-arm",
    ],
)
def test_subshell_with_dynamic_word_still_reaches_functions(command: str) -> None:
    verdict = classify_bash(command, cwd="/tmp", env={"HOME": "/tmp"}, scratchpad=None)
    assert verdict.risk is not Risk.LOW, verdict


@pytest.mark.parametrize(
    "command",
    ['f() { rm -rf ~/x; }\ncmd=$(printf f)\n"$cmd"'],
    ids=["unresolvable-dynamic-word"],
)
def test_case_pattern_fix_keeps_unresolvable_reach(command: str) -> None:
    verdict = classify_bash(command, cwd="/tmp", env={"HOME": "/tmp"}, scratchpad=None)
    assert verdict.risk is not Risk.LOW, verdict


@pytest.mark.parametrize(
    "command",
    [
        # A dynamic command word still reaches the functions it may run.
        'f() { rm -rf ~/x; }\ncmd=f\n"$cmd"',
        # A delete inside the matching arm is still ranked.
        'f() {\n  case "$1" in\n    "x/$y"*) rm -rf ~ ;;\n  esac\n}\nf a',
    ],
    ids=["dynamic-word", "delete-in-arm"],
)
def test_case_pattern_fix_keeps_real_reaches(command: str) -> None:
    verdict = classify_bash(command, cwd="/tmp", env={"HOME": "/tmp"}, scratchpad=None)
    assert verdict.risk is not Risk.LOW, verdict
