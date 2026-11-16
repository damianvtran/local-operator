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
@pytest.mark.parametrize(
    "command",
    [
        _FIELD_COMMAND,
        "body=$(cat <<'EOF'\n`git diff --check <base>..<head>`\nEOF\n)"
        ' && glab api projects/1/notes -f "body=$body"',
    ],
    ids=["quoted-body-argument", "unquoted-assignment"],
)
async def test_field_gh_pr_comment_heredoc_body_is_low_and_never_judged(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    assert _risk(home, command) is Risk.LOW

    def _no_judge(*args: object, **kwargs: object) -> None:
        raise AssertionError("a LOW call must not reach the judge")

    monkeypatch.setattr(guardrails_check, "run_judge", _no_judge)
    rules = tmp_path / "rules"
    rules.mkdir()
    settings = GuardSettings(rules_dirs=(rules,))
    ctx = _context(home)
    report = preview_tool_call("bash", {"command": command}, ctx, settings=settings)
    # Main reproduced the field refusal here: builtin_effect "judge", whose
    # text was "dangerous-delete (review) … Targets: an unresolvable target".
    assert report.builtin_risk == "LOW"
    assert report.builtin_effect == BUILTIN_PASS
    assert report.action == "allow"
    # judge=None: a HIGH verdict would fail closed ("judge unavailable") here.
    outcome = await check_tool_call(
        "bash", {"command": command}, ctx, judge=None, settings=settings
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


# The lexer intentionally retains unquoted-substitution bodies for split R10
# segments. A complete assignment must not walk that retained duplicate as an
# expansion-active OUTER heredoc after walking the nested substitution itself.
_REVIEW_LITERAL = (
    "it's literal review data: `auth-svc-preview-<id>`\n"
    "`CONTRACT_REF:<path>` and `git diff --check <base>..<head>`\n"
    "```sh\nrm -rf ~/.local-operator\n```\n"
)


@pytest.mark.parametrize("delimiter", ["'EOF'", '"EOF"', "E'O'F", "'EO'F"])
@pytest.mark.parametrize("shape", ["single", "multiple-bodies", "nested", "two-assignments"])
def test_complete_assignment_owns_quoted_heredoc_once(
    home: Path, delimiter: str, shape: str
) -> None:
    from local_operator.guardrails.resolve import resolve_command

    reader = f"cat <<{delimiter}\n{_REVIEW_LITERAL}EOF\n"
    if shape == "multiple-bodies":
        reader = f"cat <<{delimiter} <<'SECOND'\n{_REVIEW_LITERAL}EOF\n{_REVIEW_LITERAL}SECOND\n"
    elif shape == "nested":
        reader = f'printf "%s" "$(cat <<{delimiter}\n{_REVIEW_LITERAL}EOF\n)"'
    assignment = f"body=$({reader})"
    if shape == "two-assignments":
        assignment += f" other=$({reader})"
    command = assignment + ' && glab api projects/1/notes -f "body=$body"'
    ctx = _context(home)
    result = resolve_command(command, cwd=ctx.cwd, env=ctx.env, home=str(home))
    assert not result.targets, result.targets
    assert all(target.program == "glab" and target.op == "touch" for target in result.touched)
    report = preview_tool_call(
        "bash", {"command": command}, ctx, settings=GuardSettings(rules_dirs=())
    )
    assert report.builtin_risk == "LOW"
    assert report.builtin_effect == BUILTIN_PASS


@pytest.mark.parametrize("nesting", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("kind", ["write", "delete", "literal", "mixed"])
def test_assignment_dedup_recursion_budget_never_certifies_unexamined_effects(
    home: Path, nesting: int, kind: str
) -> None:
    """R1: ownership must not turn exhausted semantic analysis into LOW.

    The active expansion reaches the existing depth-3 budget one level before
    a literal reader does. Literal readers stay LOW while analyzed; beyond the
    budget even their safety is unproven, so conservative scrutiny is intended.
    These are preview-only inputs: no protected write or delete is executed.
    """
    from local_operator.guardrails.resolve import resolve_command

    body = "$(printf changed > ~/.local-operator/config.yml)"
    delimiter = "EOF"
    if kind == "delete":
        body = "$(rm -rf ~/.local-operator)"
    elif kind == "literal":
        delimiter = "'EOF'"
        body += "\nrm -rf ~/.local-operator"
    reader = f"cat <<{delimiter}\n{body}\nEOF\n"
    if kind == "mixed":
        reader = "cat <<'LITERAL' <<EOF\n`rm -rf ~/.local-operator`\nLITERAL\n" + body + "\nEOF\n"
    command = f"body=$({reader})"
    for _ in range(nesting - 1):
        command = f'body="$({command})"'
    ctx = _context(home)
    result = resolve_command(command, cwd=ctx.cwd, env=ctx.env, home=str(home))
    report = preview_tool_call(
        "bash", {"command": command}, ctx, settings=GuardSettings(rules_dirs=())
    )
    if kind == "literal" and nesting <= 3:
        assert not result.targets
        assert not result.opaque_reasons
        assert report.builtin_risk == "LOW"
        assert report.builtin_effect == BUILTIN_PASS
    elif kind != "literal" and nesting <= 2:
        assert result.targets
        assert report.builtin_risk == "CRITICAL"
        assert report.builtin_effect == "refuse-critical"
    else:
        assert "nested shell too deep to resolve" in result.opaque_reasons
        assert report.builtin_risk == "HIGH"
        assert report.builtin_effect == "judge"


@pytest.mark.parametrize(
    ("command", "owned"),
    [
        ("body=$(cat <<'EOF'\nliteral\nEOF\n)", [[True]]),
        ("cat <<'EOF'\nliteral\nEOF", [[False]]),
        ("body=$(echo hi; cat <<'EOF'\nliteral\nEOF\n)", [[], [False]]),
        ("body=$(cat <<'EOF'\nliteral\nEOF\n", [[False]]),
    ],
)
def test_lexer_retains_body_and_records_structural_substitution_owner(
    command: str, owned: list[list[bool]]
) -> None:
    from local_operator.tools.search_guard import lex

    result = lex(command)
    assert result.nested_bodies == owned
    assert [body for bodies in result.bodies for body in bodies] == ["literal"]


@pytest.mark.parametrize("separator", ["; ", " && ", "\n"])
@pytest.mark.parametrize("quoted", [True, False])
def test_split_assignment_retains_shell_body_safety(
    home: Path, separator: str, quoted: bool
) -> None:
    delimiter = "'EOF'" if quoted else "EOF"
    command = f"body=$(echo harmless{separator}bash <<{delimiter}\nrm -rf ~/.local-operator\nEOF\n)"
    assert _risk(home, command) is Risk.CRITICAL


@pytest.mark.parametrize(
    "command",
    [
        "body=$(cat <<EOF\n$(rm -rf ~/.local-operator)\nEOF\n)",
        "body=$(cat <<EOF\n`rm -rf ~/.local-operator`\nEOF\n)",
        "body=$(bash <<'EOF'\nrm -rf ~/.local-operator\nEOF\n)",
        "body=$(sh <<'EOF'\nrm -rf ~/.local-operator\nEOF\n)",
        "body=$(rm -rf ~/.local-operator)",
        "body=$(printf x > ~/.local-operator/config.yml)",
        "body=$(cat <<'EOF'\nliteral\nEOF\n); rm -rf ~/.local-operator",
        # An unpaired opener/unclosed substitution never earns ownership proof.
        "body=$(cat <<'EOF'\n`rm -rf ~/.local-operator`",
        "body=$(cat <<'EOF'\n`rm -rf ~/.local-operator`\nEOF\n",
    ],
)
def test_assignment_dedup_never_trusts_executable_or_malformed_body(
    home: Path, command: str
) -> None:
    assert _risk(home, command) >= Risk.HIGH


@pytest.mark.parametrize(
    "suffix",
    [
        "glab api projects/1/uploads -F file=@~/.local-operator/sessions/foreign/note.json",
        "glab api projects/1 --output ~/.local-operator/sessions/foreign/note.json",
        "printf x > ~/.local-operator/config.yml",
    ],
)
def test_assignment_dedup_keeps_real_protected_output_backstop(home: Path, suffix: str) -> None:
    command = "body=$(cat <<'EOF'\nliteral\nEOF\n) && " + suffix
    assert _risk(home, command) >= Risk.HIGH


def test_bash_assignment_preserves_quoted_review_bytes(home: Path) -> None:
    """Execute only cat/printf: destructive-looking review text stays literal."""
    import subprocess

    # macOS /bin/bash 3.2 incorrectly quote-tracks odd apostrophes inside
    # nested heredocs. Keep this portable shell oracle focused on literal
    # backticks/placeholders; the parser matrix above also covers apostrophes.
    literal = _REVIEW_LITERAL.replace("it's", "it is")
    command = "body=$(cat <<'EOF'\n" + literal + 'EOF\n)\nprintf "%s" "$body"'
    result = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-c", command],
        cwd=home / "repo",
        env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == b""
    assert result.stdout == literal.rstrip("\n").encode()


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


#: Security review round 2 R3 (PR #3962): under ``shopt -s extglob`` bash
#: reads ``E@(x)`` as ONE delimiter word, so the body ends at ``E@(x)`` and
#: ``rm -rf ~`` after it runs (verified under bash 5.3). Reading the word as
#: ``E@`` would skip to the later ``E@`` line and hide the delete as data.
_EXTGLOB_OPS = ["?", "*", "+", "@", "!"]


@pytest.mark.parametrize("op", _EXTGLOB_OPS, ids=["qmark", "star", "plus", "at", "bang"])
@pytest.mark.parametrize(
    ("shape", "floor"),
    [
        # Main ranked the apostrophe form HIGH (phantom quote) and the others
        # CRITICAL; each row must stay at least there.
        ('x="$(cat <<{w}\nit\'s\n{w}\nrm -rf ~\n{p}\n)"', Risk.HIGH),
        ('x="$(cat <<{w}\nits\n{w}\nrm -rf ~\n{p}\n)"', Risk.CRITICAL),
        ("cat <<{w}\nhi\n{w}\nrm -rf ~\n{p}\n", Risk.CRITICAL),
    ],
    ids=["sub-apos", "sub", "top"],
)
def test_extglob_delimiter_word_fails_closed(home: Path, op: str, shape: str, floor: Risk) -> None:
    command = "shopt -s extglob\n" + shape.format(w=f"E{op}(x)", p=f"E{op}")
    assert _risk(home, command) >= floor


#: Round 2 M1: ``$[ … ]`` is legacy arithmetic, so its ``<<`` is a shift and
#: no body follows; the ``rm`` on the next line runs whichever line comes after.
@pytest.mark.parametrize("closing", ["E]", "E"], ids=["bracket", "bare"])
@pytest.mark.parametrize("wrap", ["{c}", 'x="$({c})"'], ids=["top", "sub"])
def test_legacy_arithmetic_shift_is_not_a_heredoc_opener(
    home: Path, closing: str, wrap: str
) -> None:
    command = wrap.format(c=f"echo $[1<<E]\nrm -rf ~\n{closing}\n")
    assert _risk(home, command) is Risk.CRITICAL
    assert _risk(home, "echo $[a[1]<<2]\nls") is Risk.LOW


#: Round 2 M2: a CRLF opener line makes bash's delimiter ``EOF\r``; its body
#: ends at the ``EOF\r`` line and the delete after it runs.
@pytest.mark.parametrize(
    "command",
    [
        'x="$(cat <<EOF\r\nit\'s\nEOF\r\nrm -rf ~\nEOF\n)"',
        'x="$(cat <<EOF\r\nits\nEOF\r\nrm -rf ~\nEOF\n)"',
        "cat <<EOF\r\nit's\nEOF\r\nrm -rf ~\nEOF\n",
    ],
    ids=["sub-apos", "sub", "top"],
)
def test_carriage_return_delimiter_ranks_like_main(home: Path, command: str) -> None:
    assert _risk(home, command) is Risk.CRITICAL
