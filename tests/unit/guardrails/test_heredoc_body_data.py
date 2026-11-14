"""Heredoc and quoted body arguments are data, not delete commands.

Field false positive (2026-11-13): a reviewer subagent posting its round on
core PR #3960 ran

    cd <worktree> && gh pr view 3960 --json headRefOid && \
      gh pr comment 3960 --body "$(cat <<'EOF'
    ### Agent review — round 1
    … prose with an odd number of apostrophes and the word "remove" …
    EOF
    )"

and was refused as ``dangerous-delete (review)`` with "an unresolvable target".
Cause: the per-segment ``$(`` reader in ``resolve.py`` (``_read_sub``) was not
heredoc-aware, unlike the shared ``lex``. An apostrophe in the quoted body
opened a phantom quote, the substitution never balanced, its "body" ran to the
end of the line, and the inner walk saw open text holding a delete word.

The fix keeps every shape where body text really runs at least as strict as
before; the KEEP rows below pin that. The unquoted-heredoc rows were HIGH
before only by the same accident (an apostrophe in the body); they are now
HIGH because the ``$(…)``/backtick inside the expanded body is walked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.guardrails import check as guardrails_check
from local_operator.guardrails.check import (
    BUILTIN_PASS,
    CallContext,
    GuardSettings,
    check_tool_call,
)
from local_operator.guardrails.classify import Risk, classify_bash
from local_operator.guardrails.preview import preview_tool_call

#: Abridged from the refused command's body: the characters that mattered are
#: kept verbatim (an ODD apostrophe count — 7 here; an even count balanced
#: by luck and hid the bug — "remove", backticks, ``'…' <url>``).
_FIELD_BODY = (
    "### Agent review — round 1\n"
    "\n"
    "Reviewer: independent reviewer (`preview-origin-code-r1`), OpenAI `gpt-6.1-sol`\n"
    "Scope: 9c2bbb3b6863ec4444d99fe0c3a3a55f95e266b0..ab3235040ff007b37821105211e0c88c95613580\n"
    "\n"
    "For each, `node -e 'console.log(new URL(process.argv[1]).hostname)' <url>` actually "
    "returned `second.trycloudflare.com`.\n"
    "Actual repro: call the real `mark_session_origin(child, 'subagent', parent='familyroot')`; "
    "remove only this scratch child's marker.\n"
    "- No server/browser processes started, no merge/release action.\n"
)
_FIELD_COMMAND = (
    "cd ~/repo && gh pr view 3960 --json headRefOid && gh pr comment 3960 --body "
    "\"$(cat <<'EOF'\n" + _FIELD_BODY + 'EOF\n)"'
)
#: A body with an odd apostrophe count and delete words, for the variants.
_BODY = "it's the parent's job; don't remove it, rm -rf ~ is bad (see `x`)\n"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    state = home / ".local-operator"
    (state / "sessions" / "own" / "scratchpad").mkdir(parents=True)
    (state / "secrets").mkdir()
    (state / "config.yml").write_text("version: 1\nvalues: {}\n", encoding="utf-8")
    (home / "repo").mkdir(parents=True)
    (home / "x").mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))
    return home


def _context(home: Path) -> CallContext:
    return CallContext(
        cwd=str(home / "repo"),
        env={"HOME": str(home)},
        scratchpad=str(home / ".local-operator" / "sessions" / "own" / "scratchpad"),
        session_id="own",
    )


def _risk(home: Path, command: str) -> Risk:
    ctx = _context(home)
    return classify_bash(
        command, cwd=ctx.cwd, env=ctx.env, scratchpad=ctx.scratchpad, session_id="own"
    ).risk


# -- the field command and its data-as-argument siblings: LOW, no judge -------


@pytest.mark.asyncio
async def test_field_gh_pr_comment_heredoc_body_is_low_and_never_judged(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _risk(home, _FIELD_COMMAND) is Risk.LOW

    def _no_judge(*args: object, **kwargs: object) -> None:
        raise AssertionError("a LOW call must not reach the judge")

    monkeypatch.setattr(guardrails_check, "run_judge", _no_judge)
    rules = tmp_path / "rules"
    rules.mkdir()
    settings = GuardSettings(rules_dirs=(rules,))
    ctx = _context(home)
    report = preview_tool_call("bash", {"command": _FIELD_COMMAND}, ctx, settings=settings)
    # Main reproduced the field refusal here: builtin_effect "judge", whose
    # text was "dangerous-delete (review) … Targets: an unresolvable target".
    assert report.builtin_risk == "LOW"
    assert report.builtin_effect == BUILTIN_PASS
    assert report.action == "allow"
    # judge=None: a HIGH verdict would fail closed ("judge unavailable") here.
    outcome = await check_tool_call(
        "bash", {"command": _FIELD_COMMAND}, ctx, judge=None, settings=settings
    )
    assert outcome is None or not outcome.blocked


@pytest.mark.parametrize(
    "command",
    [
        "gh pr comment 1 --body \"$(cat <<'EOF'\n" + _BODY + 'EOF\n)"',
        'gh issue comment 1 --body "$(cat <<"EOF"\n' + _BODY + 'EOF\n)"',
        "gh pr create --title t --body \"$(cat <<'EOF'\n" + _BODY + 'EOF\n)"',
        "gh pr edit 1 --body \"$(cat <<'EOF'\n" + _BODY + 'EOF\n)"',
        "glab mr note 1 --message \"$(cat <<'EOF'\n" + _BODY + 'EOF\n)"',
        "gh pr comment 1 --body-file - <<'EOF'\n" + _BODY + "EOF",
        "msg=\"$(cat <<'EOF'\n" + _BODY + 'EOF\n)" && gh pr comment 1 --body "$msg"',
    ],
    ids=["pr-comment", "issue-comment-dq", "pr-create", "pr-edit", "glab-note", "body-file", "var"],
)
def test_quoted_heredoc_body_argument_is_data(home: Path, command: str) -> None:
    assert _risk(home, command) is Risk.LOW


# -- KEEP: body text that really runs stays dangerous -------------------------


@pytest.mark.parametrize(
    ("command", "floor"),
    [
        ("bash <<'EOF'\nrm -rf ~/x\nEOF", Risk.HIGH),
        ("sh -s <<'EOF'\nrm -rf ~/x\nEOF", Risk.HIGH),
        ("source /dev/stdin <<'EOF'\nrm -rf ~/x\nEOF", Risk.HIGH),
        ("cat <<'EOF' | bash\nrm -rf ~/x\nEOF", Risk.HIGH),
        ("xargs rm -rf <<'EOF'\n~/x\nEOF", Risk.HIGH),
        ("eval \"$(cat <<'EOF'\nrm -rf ~/x\nEOF\n)\"", Risk.HIGH),
        # A $(…) substitution as the rm target is an unresolvable target,
        # apostrophe or not (HIGH from a repo cwd, CRITICAL beside the state root).
        ('rm -rf "$(cat <<EOF\n~\nEOF\n)"', Risk.HIGH),
        ("rm -rf \"$(cat <<'EOF'\n~\nEOF\n)\" # it's", Risk.HIGH),
        # A delete after a quoted-body substitution is still seen.
        ("echo \"$(cat <<'EOF'\nit's\nEOF\n)\"; rm -rf ~/x", Risk.HIGH),
    ],
    ids=[
        "bash",
        "sh-s",
        "source-stdin",
        "pipe-bash",
        "xargs",
        "eval",
        "rm-sub",
        "rm-sub-apos",
        "after",
    ],
)
def test_executed_heredoc_text_stays_dangerous(home: Path, command: str, floor: Risk) -> None:
    assert _risk(home, command) >= floor


@pytest.mark.parametrize(
    "command",
    ['rm -rf "$(cat <<EOF\n~\nEOF\n)"', "rm -rf \"$(cat <<'EOF'\n~\nEOF\n)\" # it's"],
    ids=["unquoted", "quoted-apos"],
)
def test_substitution_rm_target_beside_state_root_stays_critical(home: Path, command: str) -> None:
    ctx = _context(home)
    verdict = classify_bash(
        command, cwd=str(home), env=ctx.env, scratchpad=ctx.scratchpad, session_id="own"
    )
    assert verdict.risk is Risk.CRITICAL


@pytest.mark.parametrize(
    "command",
    [
        "cat <<EOF\nhi $(rm -rf ~/x)\nEOF",
        "cat <<EOF\nhi `rm -rf ~/x`\nEOF",
        'gh pr comment 1 --body "$(cat <<EOF\nhi $(rm -rf ~/x)\nEOF\n)"',
        'gh pr comment 1 --body "$(cat <<EOF\nhi `rm -rf ~/x`\nEOF\n)"',
        'gh pr comment 1 --body "$(cat <<EOF\nhi $(rm -rf ~/x) it\'s\nEOF\n)"',
        'x="$(cat <<EOF\nhi $(rm -rf ~/x)\nEOF\n)"',
    ],
    ids=["top-dollar", "top-backtick", "sub-dollar", "sub-backtick", "sub-apos", "assign"],
)
def test_unquoted_heredoc_substitution_runs_and_stays_dangerous(home: Path, command: str) -> None:
    """Bash expands an unquoted-delimiter body, so its ``$(rm …)`` runs."""
    assert _risk(home, command) >= Risk.HIGH


def test_quoted_heredoc_substitution_text_is_not_run(home: Path) -> None:
    """The same ``$(rm …)`` text under a quoted delimiter is literal data."""
    assert _risk(home, "cat <<'EOF'\nhi $(rm -rf ~/x)\nEOF") is Risk.LOW


#: Security review R1 (PR #3962): openers whose delimiter WORD continues past
#: its first identifier. (opener, bash's delimiter, the prefix the old regex
#: read). Each row hides ``rm -rf ~`` between bash's delimiter line and a later
#: line equal to the misread prefix; bash runs it (verified under bash 5).
_CONTINUED_WORDS = [
    ('E"O"F', "EOF", "E"),
    ("'EO'F", "EOF", "EO"),
    ("EOF-x", "EOF-x", "EOF"),
    ("'EOF'.x", "EOF.x", "EOF"),
]


@pytest.mark.parametrize("prefix", ["x=", "gh pr comment 1 --body "], ids=["assign", "gh"])
@pytest.mark.parametrize("body", ["it's", "its"], ids=["apos", "no-apos"])
@pytest.mark.parametrize(
    ("opener", "real", "misread"), _CONTINUED_WORDS, ids=["dq-mid", "sq-prefix", "dash", "dot"]
)
def test_command_after_bashs_real_delimiter_is_walked(
    home: Path, prefix: str, body: str, opener: str, real: str, misread: str
) -> None:
    """The delete after the real delimiter line runs, so it is seen.

    Main ranked the apostrophe rows HIGH only by the phantom-quote accident and
    the apostrophe-free rows LOW; the whole-word delimiter makes both CRITICAL.
    """
    command = f'{prefix}"$(cat <<{opener}\n{body}\n{real}\nrm -rf ~\n{misread}\n)"'
    assert _risk(home, command) is Risk.CRITICAL


@pytest.mark.parametrize(
    "command",
    [
        "gh pr comment 1 --body \"$(cat <<'EOF'.x\n" + _BODY + 'EOF.x\n)"',
        'gh pr comment 1 --body "$(cat <<E"O"F\n' + _BODY + 'EOF\n)"',
    ],
    ids=["dot", "dq-mid"],
)
def test_whole_word_quoted_delimiter_body_is_still_data(home: Path, command: str) -> None:
    """Reading the whole word keeps a correctly closed quoted body as data."""
    assert _risk(home, command) is Risk.LOW


@pytest.mark.parametrize(
    "command",
    [
        'x="$(cat <<E$X\nit\nE$X\nrm -rf ~/x\n)"',
        'x="$(cat <<\'EOF\nit\nEOF\nrm -rf ~/x\n)"',
        'x="$(cat <<E\\OF\nit\nEOF\nrm -rf ~/x\n)"',
    ],
    ids=["dollar", "unclosed-quote", "backslash"],
)
def test_unreadable_delimiter_word_fails_closed(home: Path, command: str) -> None:
    """A delimiter word this reader will not model is never skipped over."""
    assert _risk(home, command) >= Risk.HIGH


def test_arithmetic_shift_is_not_a_heredoc_opener(home: Path) -> None:
    """``$((1<<x))``: ``<<`` is a shift; no body is read past the newline."""
    assert _risk(home, "echo $((1<<x))\nrm -rf ~/x") >= Risk.HIGH
    assert _risk(home, "echo $((1<<x))\nls") is Risk.LOW
