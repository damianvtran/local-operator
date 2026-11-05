"""The dry-run preview, and the calibration loop the guardrails guide teaches.

WHY THESE TESTS. A rule you cannot test is a rule you tune by guesswork, and the
guide now tells agents to ``preview_tool_call`` a candidate call before saving a
rule and to re-preview after narrowing one. That is only honest if the preview
answers with the checkpoint's OWN strings and OWN order: a preview that
re-describes the refusal in its own words would let an agent "verify" a rule
against a promise the checkpoint does not keep. So the central assertions here are
byte-equality with ``check_tool_call`` on the same rule, and the built-in layer's
precedence over the rules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.guardrails import check as guardrails_check
from local_operator.guardrails.check import (
    BLOCK_MARKER,
    CallContext,
    GuardSettings,
    check_tool_call,
)
from local_operator.guardrails.preview import preview_tool_call


def _rule(name: str, action: str, conditions: list[str], body: str, question: str = "") -> str:
    lines = "".join(f"  - '{pattern}'\n" for pattern in conditions)
    extra = f"question: {question}\n" if question else ""
    return (
        f"---\nname: {name}\nscope: tool:bash\ncondition:\n{lines}"
        f"{extra}action: {action}\n---\n{body}\n"
    )


def _settings(rules_dir: Path) -> GuardSettings:
    return GuardSettings(rules_dirs=(rules_dir,))


def _judge(answer: str):
    """A deterministic judge: ``check.py`` parses a first line of ALLOW/BLOCK."""

    async def ask(system_prompt: str, brief: str) -> str:
        return answer

    return ask


def _context(tmp_path: Path, *, home: Path | None = None, state: Path | None = None) -> CallContext:
    """The call's own surroundings: a scratchpad it may delete, an empty env.

    ``env`` stays empty on purpose: unlike the real loop, the preview defaults to
    no environment, and these tests must exercise the preview as an agent calls
    it, not a shape that only holds when the session env is threaded through.
    """
    scratchpad = (state or tmp_path / "state") / "sessions" / "own" / "scratchpad"
    return CallContext(
        cwd=str((home or tmp_path) / "repo"),
        env={"HOME": str(home)} if home else {},
        scratchpad=str(scratchpad),
        session_id="own",
    )


@pytest.fixture
def rules_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "state" / "rules"
    directory.mkdir(parents=True)
    return directory


@pytest.fixture
def guarded_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A home whose state root exists, with the config dir pointing at it."""
    home = tmp_path / "home"
    state = home / ".local-operator"
    (state / "sessions" / "own" / "scratchpad").mkdir(parents=True)
    (state / "secrets").mkdir()
    (state / "config.yml").write_text("version: 1\nvalues: {}\n", encoding="utf-8")
    (home / "repo" / "pkg" / "build").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))
    return home


# -- what the agent would see ------------------------------------------------


@pytest.mark.asyncio
async def test_preview_block_reports_the_rule_and_the_exact_text(
    rules_dir: Path, tmp_path: Path
) -> None:
    """A block prints the string the call would actually return, byte for byte."""
    (rules_dir / "no-force-push.md").write_text(
        _rule("no-force-push", "block", [r"git\s+push\b.*--force"], "Open a PR instead."),
        encoding="utf-8",
    )
    command = "git push --force origin main"
    ctx = _context(tmp_path)

    report = preview_tool_call("bash", {"command": command}, ctx, settings=_settings(rules_dir))
    outcome = await check_tool_call(
        "bash", {"command": command}, ctx, judge=None, settings=_settings(rules_dir)
    )

    assert report.action == "block"
    assert report.rule == "no-force-push"
    assert report.text.startswith(f"{BLOCK_MARKER} no-force-push (block)")
    assert outcome is not None and outcome.blocked is True
    # The whole point: not "similar to", not "starts the same way".
    assert report.text == outcome.text


@pytest.mark.asyncio
async def test_preview_warn_reports_the_appended_line(rules_dir: Path, tmp_path: Path) -> None:
    """A warn is not a refusal: the preview says the call runs and quotes the line."""
    (rules_dir / "pushing.md").write_text(
        _rule("pushing", "warn", [r"git\s+push\b"], "Say which branch and where it goes."),
        encoding="utf-8",
    )
    command = "git push origin feature"
    ctx = _context(tmp_path)

    report = preview_tool_call("bash", {"command": command}, ctx, settings=_settings(rules_dir))
    outcome = await check_tool_call(
        "bash", {"command": command}, ctx, judge=None, settings=_settings(rules_dir)
    )

    assert report.action == "warn"
    assert report.rule == ""
    assert report.text == ""
    assert list(report.warnings) == ["Guardrail pushing: Say which branch and where it goes."]
    assert outcome is not None and outcome.blocked is False
    assert list(outcome.warnings) == list(report.warnings)


def test_preview_on_an_unmatched_call_reports_no_rule(rules_dir: Path, tmp_path: Path) -> None:
    """The negative control for matching: a rule that does not match must not fire."""
    (rules_dir / "no-force-push.md").write_text(
        _rule("no-force-push", "block", [r"git\s+push\b.*--force"], "Open a PR instead."),
        encoding="utf-8",
    )

    report = preview_tool_call(
        "bash",
        {"command": "git push origin main"},
        _context(tmp_path),
        settings=_settings(rules_dir),
    )

    assert report.action == "allow"
    assert report.rule == "" and report.text == "" and report.rules == ()
    assert "ALLOW" in report.render()


@pytest.mark.asyncio
async def test_preview_reports_review_without_asking_the_judge(
    rules_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A review is reported as the QUESTION, never as a verdict.

    ``check_tool_call`` with no judge fails closed and returns "judge unavailable:
    …". The preview must not borrow that string: it would read as a refusal the
    call has not received. So the preview names the rule, shows the question and
    policy the judge would be asked about, and produces no refusal text at all.
    """
    (rules_dir / "force-push-review.md").write_text(
        _rule(
            "force-push-review",
            "review",
            [r"git\s+push\b.*--force"],
            "Force-pushing needs the user's word.",
            question="Did the user explicitly ask to force-push this branch?",
        ),
        encoding="utf-8",
    )
    command = "git push --force origin feature"

    # The string the preview must NOT print: with no judge wired, the real
    # checkpoint fails closed and says so. Asked first, before the judge is
    # booby-trapped below, because that trap turns the checkpoint's own call into
    # the generic error refusal.
    outcome = await check_tool_call(
        "bash",
        {"command": command},
        _context(tmp_path),
        judge=None,
        settings=_settings(rules_dir),
    )
    assert outcome is not None and "judge unavailable" in outcome.text

    def _no_judge(*args: object, **kwargs: object) -> None:
        raise AssertionError("the preview must never run a model")

    monkeypatch.setattr(guardrails_check, "run_judge", _no_judge)

    report = preview_tool_call(
        "bash", {"command": command}, _context(tmp_path), settings=_settings(rules_dir)
    )

    assert report.action == "review"
    assert report.rule == "force-push-review"
    assert report.text == ""
    assert report.rules[0].question == "Did the user explicitly ask to force-push this branch?"
    assert report.rules[0].policy == "Force-pushing needs the user's word."
    rendered = report.render()
    assert "would go to the judge" in rendered
    assert "judge unavailable" not in rendered


def test_preview_reports_the_builtin_ranking_before_the_rules(
    guarded_home: Path, tmp_path: Path
) -> None:
    """A rule that matches a CRITICAL call protects nothing, so the preview says so.

    The built-in layer refuses before the rule loop is entered. Reporting the
    user's rule as the deciding action there would tell an agent its calibration
    had a say in a call the checkpoint never lets reach it.
    """
    rules_dir = tmp_path / "state" / "rules"
    rules_dir.mkdir(parents=True)
    (rules_dir / "peek.md").write_text(
        _rule("peek", "block", [r"rm -rf"], "Never delete the state root."), encoding="utf-8"
    )
    ctx = _context(tmp_path, home=guarded_home, state=guarded_home / ".local-operator")

    report = preview_tool_call(
        "bash", {"command": "rm -rf ~/.local-operator"}, ctx, settings=_settings(rules_dir)
    )

    assert report.builtin_risk == "CRITICAL"
    assert report.action == "refuse"
    assert report.rule == "dangerous-delete"
    assert report.rules == ()
    assert "refused outright" in report.render()
    assert f"{BLOCK_MARKER} peek" not in report.render()


# -- control flow: a judge in front of the call does not end evaluation ------


@pytest.mark.asyncio
async def test_a_review_rule_does_not_make_a_later_block_unreachable(
    rules_dir: Path, tmp_path: Path
) -> None:
    """A ``review`` is a STEP, not a decision: on an ALLOW the checkpoint reads on.

    ``check.py::_check_tool_call`` returns only when the judge DENIES; when it
    allows, the loop continues and the next matching rule is consulted. So a
    ``block`` behind a ``review`` is exactly the rule that refuses the call the
    ``review`` let through — reporting it as "never reached" told an agent the
    guard has no hard block on a call the guard refuses, and invited it to "clean
    up" a rule that is doing the work.
    """
    (rules_dir / "a-review.md").write_text(
        _rule(
            "a-review",
            "review",
            [r"git\s+push\b"],
            "A push needs the user's word.",
            question="Did the user ask for this push?",
        ),
        encoding="utf-8",
    )
    (rules_dir / "b-block.md").write_text(
        _rule("b-block", "block", [r"--force"], "Never force-push main."), encoding="utf-8"
    )
    command = "git push --force origin main"
    ctx = _context(tmp_path)

    report = preview_tool_call("bash", {"command": command}, ctx, settings=_settings(rules_dir))

    assert report.action == "review", "the judge is the first thing the call meets"
    assert report.rule == "a-review"
    assert report.not_evaluated == (), "a rule behind a review IS reached when the judge allows"
    assert [entry.name for entry in report.rules] == ["a-review", "b-block"]
    assert report.rules[1].action == "block"
    assert report.rules[1].gated_by == "a-review"
    rendered = report.render()
    assert "a-review" in rendered and "b-block" in rendered
    assert "never reached" not in rendered
    assert "if every review above allows, b-block still refuses it" in rendered

    # The premise, against the live checkpoint: an ALLOW does not clear the call.
    allowed = await check_tool_call(
        "bash", {"command": command}, ctx, judge=_judge("ALLOW"), settings=_settings(rules_dir)
    )
    assert allowed is not None and allowed.blocked is True
    assert allowed.rule == "b-block"
    assert allowed.text == report.rules[1].text, "the report quotes the refusal that fires"
    denied = await check_tool_call(
        "bash",
        {"command": command},
        ctx,
        judge=_judge("BLOCK: not asked for"),
        settings=_settings(rules_dir),
    )
    assert denied is not None and denied.rule == "a-review", "a denial refuses at the review"


@pytest.mark.asyncio
async def test_a_warn_behind_a_block_is_reported_discarded(rules_dir: Path, tmp_path: Path) -> None:
    """A refusal carries no warnings, so the preview must not promise the line.

    ``_check_tool_call`` appends to its local warning list and returns the block
    outcome WITHOUT it, so the agent never sees a line a later ``block``
    suppressed — not when the block fires, and not when a gate in front of it
    denies first. A preview that says "appended to the result" there answers the
    calibration question ("is this warn noise?") with the opposite of the truth.
    """
    (rules_dir / "a-warn.md").write_text(
        _rule("a-warn", "warn", [r"git\s+push\b"], "Heads up: a push."), encoding="utf-8"
    )
    (rules_dir / "b-block.md").write_text(
        _rule("b-block", "block", [r"--force"], "Never force-push main."), encoding="utf-8"
    )
    command = "git push --force origin main"
    ctx = _context(tmp_path)

    report = preview_tool_call("bash", {"command": command}, ctx, settings=_settings(rules_dir))
    outcome = await check_tool_call(
        "bash", {"command": command}, ctx, judge=None, settings=_settings(rules_dir)
    )

    assert report.action == "block"
    assert report.warnings == (), "the refusal carries no warning line, on either side"
    assert outcome is not None and list(outcome.warnings) == []
    rendered = report.render()
    assert "discarded — b-block refuses this call" in rendered
    assert "appended to the result" not in rendered
    # The rule still shows up: it matched, the agent just never reads the line.
    assert [entry.name for entry in report.rules] == ["a-warn", "b-block"]
    assert report.rules[0].text == "Guardrail a-warn: Heads up: a push."


@pytest.mark.asyncio
async def test_a_builtin_judge_call_is_not_reported_as_allowed(
    guarded_home: Path, tmp_path: Path
) -> None:
    """A judge-gated call is undecided here — ALLOW would assert the judge's answer.

    The built-in layer sends a HIGH call to the judge BEFORE the rules are read
    (``check.py::_dangerous_delete``), so ``allow``/``warn`` are what the preview
    can see of the rules, not what happens to the call. The live checkpoint with a
    denying judge refuses it, which the preview must not contradict in its verdict.
    """
    rules_dir = tmp_path / "state" / "rules"
    rules_dir.mkdir(parents=True)
    (rules_dir / "deploy-warn.md").write_text(
        _rule("deploy-warn", "warn", [r"shutil"], "Deletes are logged."), encoding="utf-8"
    )
    ctx = _context(tmp_path, home=guarded_home, state=guarded_home / ".local-operator")
    command = "python -c \"import shutil; shutil.rmtree('~/.local-operator')\""

    report = preview_tool_call("bash", {"command": command}, ctx, settings=_settings(rules_dir))

    assert report.builtin_effect == "judge", "a HIGH call goes to the judge first"
    assert report.action == "warn", "the rules alone would append the line"
    rendered = report.render()
    assert "verdict: ALLOW" not in rendered
    assert "verdict: WARN" not in rendered
    assert "undecided here" in rendered
    assert "the built-in layer sends this call to the judge first" in rendered

    # The same call with a denying judge: a refusal, which the verdict above must
    # not have claimed away.
    denied = await check_tool_call(
        "bash",
        {"command": command},
        ctx,
        judge=_judge("BLOCK: not asked for"),
        settings=_settings(rules_dir),
    )
    assert denied is not None and denied.blocked is True
    assert denied.rule == "dangerous-delete"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "branch"),
    (
        ("rm -rf {pad}/tmp", "pass"),
        ("rm -rf ~/.local-operator", "refuse-critical"),
        ("rm -rf {home}", "refuse-critical"),
        ('rm -rf "$AWS_SECRET_ACCESS_KEY"', "refuse-credential"),
    ),
)
async def test_the_builtin_branch_is_what_the_checkpoint_does(
    guarded_home: Path, tmp_path: Path, command: str, branch: str
) -> None:
    """Pin the branch label to the checkpoint's own behaviour, shape by shape.

    ``builtin_branch`` is shared by the preview and ``_dangerous_delete``, so the
    two cannot disagree — but the LABELS are still a contract: if a branch's
    behaviour moves, the preview's words stop being true. Each row therefore
    asserts the checkpoint's outcome, not the helper's return value twice.
    """
    state = guarded_home / ".local-operator"
    ctx = _context(tmp_path, home=guarded_home, state=state)
    ctx = CallContext(
        cwd=ctx.cwd,
        env={**ctx.env, "AWS_SECRET_ACCESS_KEY": str(guarded_home)},
        scratchpad=ctx.scratchpad,
        session_id=ctx.session_id,
    )
    rules_dir = tmp_path / "state" / "rules"
    rules_dir.mkdir(parents=True)
    resolved = command.format(pad=ctx.scratchpad, home=guarded_home)

    report = preview_tool_call("bash", {"command": resolved}, ctx, settings=_settings(rules_dir))
    assert report.builtin_effect == branch

    # An ALLOW judge is the strongest pass the checkpoint can grant; a refusing
    # branch must stand through it.
    allowed = await check_tool_call(
        "bash",
        {"command": resolved},
        ctx,
        judge=_judge("ALLOW"),
        settings=_settings(rules_dir),
    )
    if branch == "pass":
        assert allowed is None or allowed.blocked is False, "a LOW call never asks the judge"
    elif branch == "refuse-credential":
        assert allowed is not None and allowed.blocked is True
        assert "unresolvable target" in allowed.text, "a secret-derived target is refused as such"
    else:
        assert allowed is not None and allowed.blocked is True
        assert allowed.rule == "dangerous-delete"
    if branch == "refuse-credential":
        # Also with the override an operator can authorise: consent cannot be
        # shown for a secret, so this branch is not the one the override lifts.
        overridden = preview_tool_call(
            "bash",
            {"command": resolved},
            ctx,
            settings=GuardSettings(rules_dirs=(rules_dir,), critical_override=True),
        )
        assert overridden.builtin_effect == "refuse-credential"


def test_an_authorised_critical_override_is_only_visible_when_the_caller_passes_it(
    guarded_home: Path, tmp_path: Path
) -> None:
    """The preview reads the loosening gate the caller tells it about, no more.

    The harness passes ``authorised=`` to the checkpoint, and both loosening keys
    are honoured only for a key in that set. A preview that read the config bare
    would come out STRICTER than a running process an operator authorised — safe,
    but stated, and closable by passing the set the caller already holds.
    """
    (guarded_home / ".local-operator" / "rules").mkdir()
    (guarded_home / ".local-operator" / "config.yml").write_text(
        "version: 1\nvalues:\n  guardrails:\n    critical_override: true\n", encoding="utf-8"
    )
    ctx = _context(tmp_path, home=guarded_home, state=guarded_home / ".local-operator")
    command = "rm -rf ~/.local-operator"

    strict = preview_tool_call("bash", {"command": command}, ctx)
    assert strict.action == "refuse", "an unauthorised override must not be modelled as on"

    authorised = preview_tool_call(
        "bash", {"command": command}, ctx, authorised=frozenset({"guardrails.critical_override"})
    )
    assert authorised.builtin_effect == "judge", "an authorised override reaches the judge"
    assert authorised.action != "refuse"


def test_the_disabled_rule_layer_is_authorised_like_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``enabled: false`` reaches the preview only through the authorised set.

    The checkpoint reads that key with ``authorised=`` from the harness; a bare
    read would report rules that a running process with the operator's authority
    has switched off. Both directions are asserted, so the argument is not a
    no-op: unauthorised keeps the rules, authorised drops them.
    """
    state = tmp_path / "state"
    (state / "rules").mkdir(parents=True)
    (state / "rules" / "no-npm.md").write_text(
        _rule("no-npm", "block", [r"npm\s+install"], "Install through the lockfile."),
        encoding="utf-8",
    )
    (state / "config.yml").write_text(
        "version: 1\nvalues:\n  guardrails:\n    enabled: false\n", encoding="utf-8"
    )
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))

    strict = preview_tool_call("bash", {"command": "npm install"})
    assert strict.action == "block", "an unauthorised `enabled: false` must not switch rules off"

    loose = preview_tool_call(
        "bash", {"command": "npm install"}, authorised=frozenset({"guardrails.enabled"})
    )
    assert loose.action == "allow"
    assert loose.rules == ()
    assert loose.rule_dirs == (), "the report does not claim folders it did not load"


def test_the_report_names_the_rule_dirs_it_loaded(rules_dir: Path, tmp_path: Path) -> None:
    """`rules_dirs=` REPLACES the live folder, so the report cannot say CURRENT."""
    (rules_dir / "no-force-push.md").write_text(
        _rule("no-force-push", "block", [r"git\s+push\b.*--force"], "Open a PR instead."),
        encoding="utf-8",
    )

    report = preview_tool_call(
        "bash", {"command": "git push --dry-run"}, _context(tmp_path), settings=_settings(rules_dir)
    )

    assert report.rule_dirs == (str(rules_dir),)
    rendered = report.render()
    assert str(rules_dir) in rendered, "the reader can see which folder was scored"
    assert "CURRENT" not in rendered


# -- the calibration loop: 3 true positives, 3 false positives ---------------

#: One pattern per way of forcing a push, and one for the branch it lands on —
#: the shapes a real "never force-push main" rule has to catch.
TRUE_POSITIVES = (
    "git push --force origin main",
    "git push origin main --force",
    "git push -f main",
)
#: Ordinary pushes. A rule that blocks these is a rule the user will ask you to
#: switch off, which is how a guardrail ends up disabled entirely.
FALSE_POSITIVES = (
    "git push --dry-run",
    "git push --tags",
    "git push origin HEAD:refs/heads/feature",
)


@pytest.mark.asyncio
async def test_calibration_narrows_without_losing_a_true_positive(
    rules_dir: Path, tmp_path: Path
) -> None:
    """Broad rule → six wrong answers; narrowed rule → three right ones.

    This is the loop the guide teaches, run end to end: the broad rule is scored
    against both sets, the narrowed one replaces it, and the same six calls go
    through the REAL checkpoint again. All six scores come from
    ``check_tool_call``, not from ``Rule.matches``, so a rule that matches in
    isolation but loses to the built-in layer or to another rule cannot pass here.
    """
    path = rules_dir / "no-force-push.md"
    broad = r"git\s+push\b"
    narrowed = r"git\s+push\b[^\n]*\s(--force|-f)(\s|$)"
    ctx = _context(tmp_path)

    async def scores() -> tuple[list[bool], list[bool]]:
        blocked = []
        for command in TRUE_POSITIVES + FALSE_POSITIVES:
            outcome = await check_tool_call(
                "bash", {"command": command}, ctx, judge=None, settings=_settings(rules_dir)
            )
            blocked.append(bool(outcome and outcome.blocked))
            assert command  # keep the loop readable in a failure report
        return blocked[: len(TRUE_POSITIVES)], blocked[len(TRUE_POSITIVES) :]

    path.write_text(_rule("no-force-push", "block", [broad], "Open a PR instead."), "utf-8")
    before_tp, before_fp = await scores()
    # The recorded false positive: a dry run and a tag push do not reach main at
    # all, and the feature-branch push is exactly what the user asked for.
    assert before_tp == [True, True, True]
    assert before_fp == [True, True, True]

    path.write_text(_rule("no-force-push", "block", [narrowed], "Open a PR instead."), "utf-8")
    after_tp, after_fp = await scores()
    assert after_tp == [True, True, True], "a narrowing that loses a true positive is a regression"
    assert after_fp == [False, False, False]

    # And the preview agrees, because it shares the checkpoint's matching: an
    # agent re-previewing after the edit sees the same six answers without a call.
    for command in TRUE_POSITIVES:
        report = preview_tool_call("bash", {"command": command}, ctx, settings=_settings(rules_dir))
        assert report.action == "block", command
    for command in FALSE_POSITIVES:
        report = preview_tool_call("bash", {"command": command}, ctx, settings=_settings(rules_dir))
        assert report.action == "allow", command


# -- the negative control: calibration cannot be a bypass --------------------


@pytest.mark.asyncio
async def test_editing_the_config_cannot_switch_off_the_rule_that_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent that hit a rule cannot lift it by editing config.yml.

    ``guardrails.enabled`` is a loosening key: ``read_settings`` honours it only
    when the HOST proves an operator set it in this process. A file edit — by the
    model, by hand, or surviving a restart — reads as the default, so the rule
    that just blocked still blocks on the next call.
    """
    state = tmp_path / "state"
    (state / "rules").mkdir(parents=True)
    (state / "rules" / "no-force-push.md").write_text(
        _rule("no-force-push", "block", [r"git\s+push\b.*--force"], "Open a PR instead."),
        encoding="utf-8",
    )
    (state / "config.yml").write_text(
        "version: 1\nvalues:\n  guardrails:\n    enabled: false\n", encoding="utf-8"
    )
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))
    ctx = _context(tmp_path)

    settings = guardrails_check.read_settings()
    assert settings.enabled is True, "an unauthorised file edit must not disable user rules"
    outcome = await check_tool_call(
        "bash", {"command": "git push --force origin main"}, ctx, judge=None, settings=settings
    )
    assert outcome is not None and outcome.blocked is True
    assert outcome.rule == "no-force-push"


@pytest.mark.asyncio
async def test_critical_delete_still_refuses_with_rules_and_override_present(
    guarded_home: Path, tmp_path: Path
) -> None:
    """No rule file and no config key can weaken the built-in dangerous-delete.

    Two edits that look like they would help an agent that just hit it — a rule
    directory that only carries warnings, and ``critical_override: true`` written
    into the config — and the CRITICAL refusal still stands, because the override
    needs the operator's in-process authorisation and the built-in is not a user
    rule file at all.
    """
    state = guarded_home / ".local-operator"
    (state / "rules").mkdir()
    (state / "rules" / "benign.md").write_text(
        _rule("benign", "warn", [r"rm\b"], "Fine."), encoding="utf-8"
    )
    (state / "config.yml").write_text(
        "version: 1\nvalues:\n  guardrails:\n    critical_override: true\n"
        f"    rules_dir: {state / 'rules'}\n",
        encoding="utf-8",
    )
    ctx = _context(tmp_path, home=guarded_home, state=state)

    settings = guardrails_check.read_settings()
    assert settings.critical_override is False
    assert settings.rules_dirs == (state / "rules",)

    outcome = await check_tool_call(
        "bash", {"command": "rm -rf ~/.local-operator"}, ctx, judge=None, settings=settings
    )
    assert outcome is not None and outcome.blocked is True
    assert outcome.rule == "dangerous-delete"
    assert "(critical)" in outcome.text
