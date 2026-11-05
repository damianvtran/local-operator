"""Dry-run the guardrail checkpoint against ONE call: no execution, no judge.

WHY. A rule you cannot test is a rule you tune by guesswork. The first evidence
that an ``action: block`` pattern is too broad is a call the user wanted run, and
the first evidence that it is too narrow is nothing at all — a pattern that never
fires looks exactly like one that is quietly broken. So the calibration loop needs
a way to ask, before trusting a change: *which rule matches this call, which action
fires, and what would the agent see*.

WHY A MODULE AND NOT A ``lop guardrails`` SUBCOMMAND. A subcommand is user-facing
surface: help text and labels that must be translated into every locale and
audited. The consumer here is an agent inside a session, whose ``eval`` kernel
already imports ``local_operator`` — a documented function is the whole surface
that is needed, and it adds no copy.

PURITY — what this module deliberately does NOT do. It never executes anything,
never calls the judge, never reads the conversation and never writes a file. It
loads the CURRENT rule files through ``load_rules``, reuses the checkpoint's own
matching (``Rule.matches``/``call_text``) and the checkpoint's own message
builders, so the ``block`` text and the ``warn`` line printed here are the same
strings ``check_tool_call`` returns rather than a second description of them that
can drift.

WHAT IT CANNOT TELL YOU (the guide says this too). The preview evaluates ONE call
that you hand it: it does not prove the rule catches the forms you did not think
of, it does not run the tool, it does not ask the judge — a ``review`` rule is
reported as *what the judge would be asked*, never as a verdict — and it does not
model another session's config.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from local_operator.guardrails.check import (
    DANGEROUS_DELETE_RULE,
    CallContext,
    GuardSettings,
    _safe_text,
    _targets_text,
    read_settings,
    review_text,
    rule_block_text,
    rule_warn_text,
)
from local_operator.guardrails.classify import Risk, Verdict, classify_bash
from local_operator.guardrails.rules import call_text, load_rules

#: What the built-in dangerous-delete layer does with this call before any rule is
#: consulted. ``""`` means the layer does not apply (not bash, or no command text);
#: ``judge`` means its answer — not the preview — decides.
BUILTIN_PASS = "pass"
BUILTIN_JUDGE = "judge"
BUILTIN_REFUSE_CRITICAL = "refuse-critical"
BUILTIN_REFUSE_CREDENTIAL = "refuse-credential"

_REFUSING = frozenset({BUILTIN_REFUSE_CRITICAL, BUILTIN_REFUSE_CREDENTIAL})

_LIMITS = (
    "not proven: this is ONE call against the CURRENT rule files. It does not run "
    "the call, does not ask the judge, does not read the conversation, and says "
    "nothing about forms of the call you did not hand it."
)


@dataclass(frozen=True)
class RulePreview:
    """One matching rule, and what the agent sees because of it.

    ``text`` is the exact string the checkpoint returns (``block``) or appends to
    the result (``warn``). It is empty for ``review``, where the text carries the
    judge's reason — an answer the preview must not fabricate; ``question`` and
    ``policy`` are what that answer would be asked about.
    """

    name: str
    action: str
    text: str = ""
    question: str = ""
    policy: str = ""


@dataclass(frozen=True)
class PreviewReport:
    """What the checkpoint would do with one call, as far as is statically knowable.

    ``action`` is the deciding outcome: ``allow`` (nothing matched, the call runs),
    ``warn`` (runs, with ``warnings`` appended), ``block`` (a rule refuses it),
    ``review`` (it goes to the judge, whose answer decides), or ``refuse`` (the
    built-in dangerous-delete layer refuses it before the rules are read).
    """

    tool: str
    call_text: str
    action: str = "allow"
    rule: str = ""
    text: str = ""
    warnings: tuple[str, ...] = ()
    rules: tuple[RulePreview, ...] = ()
    #: Rules that match this call but would never be reached, because an earlier
    #: rule already refused it or sent it to the judge.
    not_evaluated: tuple[str, ...] = ()
    builtin_risk: str = ""
    builtin_targets: str = ""
    builtin_effect: str = ""

    def render(self) -> str:
        """The report as text an agent reads back from its ``eval`` output."""
        lines = [f"preview: {self.tool}  {_one_line(self.call_text)}"]
        if self.builtin_effect:
            lines.append(
                f"built-in {DANGEROUS_DELETE_RULE}: risk={self.builtin_risk or 'LOW'} "
                f"targets={self.builtin_targets} → {_EFFECT_WORDS[self.builtin_effect]}"
            )
        for entry in self.rules:
            lines.append(f"matches {entry.name} ({entry.action}):")
            if entry.action == "review":
                lines.append(f"  would go to the judge — question: {entry.question or '(none)'}")
                lines.append(f"  policy: {_one_line(entry.policy) or '(empty)'}")
                # The refusal text is built from the judge's reason, which this
                # module does not have. Quote its shape with the reason marked as
                # the placeholder so a reader knows which string is missing, not
                # invent an answer and present it as the verdict.
                lines.append(
                    "  on denial the agent sees: "
                    + review_text(entry.name, "<judge reason>", f"this {self.tool} call")
                )
                continue
            label = "agent sees" if entry.action == "block" else "appended to the result"
            lines.append(f"  {label}: {entry.text}")
        if self.not_evaluated:
            lines.append(
                "never reached (an earlier rule already decided): " + ", ".join(self.not_evaluated)
            )
        if self.action == "allow":
            lines.append("verdict: ALLOW — no rule matches, nothing is added to the result")
        elif self.action == "warn":
            lines.append("verdict: WARN — the call runs; the line above is appended to its result")
        elif self.action == "block":
            lines.append(f"verdict: BLOCK by {self.rule} — the text above is the call's result")
        elif self.action == "review":
            lines.append(
                f"verdict: REVIEW by {self.rule} — the judge decides; a denial refuses the call"
            )
        else:
            lines.append(
                f"decision made before your rules: the built-in layer refuses this call "
                f"({_EFFECT_WORDS[self.builtin_effect]}), so no user rule is consulted"
            )
        lines.append(_LIMITS)
        return "\n".join(lines)


_EFFECT_WORDS = {
    BUILTIN_PASS: "runs; your rules are consulted next",
    BUILTIN_JUDGE: (
        "goes to the judge BEFORE your rules; if it is allowed, your rules are consulted next"
    ),
    BUILTIN_REFUSE_CRITICAL: "refused outright, no model call, before your rules are read",
    BUILTIN_REFUSE_CREDENTIAL: (
        "refused outright — a target comes from a secret, so no consent is possible"
    ),
}


def _one_line(text: str, limit: int = 160) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + " …"


def _ranked_targets(verdict: Verdict | None) -> str:
    return _targets_text(verdict) if verdict is not None else ""


def _builtin_effect(verdict: Verdict, *, critical_override: bool) -> str:
    """The branch ``check.py::_dangerous_delete`` takes, in the same order.

    Mirrors its first-refusal-wins order exactly: CRITICAL without an authorised
    override refuses before the credential check, which itself refuses before the
    judge. Keeping the order here is what makes ``refuse`` truthful — a rule that
    matches a refused call protects nothing.
    """
    if verdict.risk is Risk.LOW:
        return BUILTIN_PASS
    if verdict.risk is Risk.CRITICAL and not critical_override:
        return BUILTIN_REFUSE_CRITICAL
    if any(t.target.credential_derived for t in verdict.targets):
        return BUILTIN_REFUSE_CREDENTIAL
    return BUILTIN_JUDGE


def preview_tool_call(
    tool_name: str,
    args: Mapping[str, Any],
    ctx: CallContext | None = None,
    *,
    settings: GuardSettings | None = None,
) -> PreviewReport:
    """Resolve one call against the current rules. Pure: nothing runs, no model.

    ``settings`` defaults to :func:`read_settings`, i.e. the same ``guardrails.*``
    values the next real call would read — including the loosening gate, so a
    ``critical_override: true`` written into a config file without operator proof
    is read as off here exactly as it is by the checkpoint. ``ctx`` supplies the
    cwd/env/scratchpad/session the built-in layer resolves paths against; its
    conversation fields (``user_request``, ``intent``) are ignored, because the
    preview never reads the conversation.

    Returns the deciding action plus, for ``block`` and ``warn``, the exact text
    ``check_tool_call`` would produce. ``review`` carries the judge's question and
    policy and no text: the refusal text would contain an answer the judge has not
    given yet.
    """
    context = ctx or CallContext(cwd=None, env={}, scratchpad=None)
    live = settings if settings is not None else read_settings()
    rules = load_rules(*live.rules_dirs) if live.enabled else load_rules(None)

    verdict: Verdict | None = None
    effect = ""
    if tool_name == "bash":
        command = args.get("command")
        if isinstance(command, str) and command.strip():
            verdict = classify_bash(
                command,
                cwd=context.cwd,
                env=context.env,
                scratchpad=context.scratchpad,
                session_id=context.session_id,
            )
            effect = _builtin_effect(verdict, critical_override=live.critical_override)

    base: dict[str, Any] = {
        "tool": tool_name,
        "call_text": call_text(tool_name, args),
        "builtin_risk": verdict.risk.name if verdict is not None else "",
        "builtin_targets": _ranked_targets(verdict),
        "builtin_effect": effect,
    }
    if effect in _REFUSING:
        return PreviewReport(**base, action="refuse", rule=DANGEROUS_DELETE_RULE)

    matched: list[RulePreview] = []
    warnings: list[str] = []
    decider: RulePreview | None = None
    not_evaluated: list[str] = []
    for rule in rules.user:
        if not rule.matches(tool_name, args):
            continue
        if decider is not None:
            not_evaluated.append(rule.name)
            continue
        if rule.action == "warn":
            entry = RulePreview(rule.name, "warn", text=_safe_text(rule_warn_text(rule)))
            matched.append(entry)
            warnings.append(entry.text)
            continue
        # block or review: this rule is what the call meets, in load order.
        decider = (
            RulePreview(rule.name, "block", text=_safe_text(rule_block_text(rule)))
            if rule.action == "block"
            else RulePreview(rule.name, "review", question=rule.question, policy=rule.body)
        )
        matched.append(decider)

    if decider is not None and decider.action == "review":
        # No text: the refusal would carry the judge's reason, and this module
        # does not ask the judge. Quote the shape once so a reader can tell which
        # string they are NOT seeing.
        return PreviewReport(
            **base,
            action="review",
            rule=decider.name,
            warnings=tuple(warnings),
            rules=tuple(matched),
            not_evaluated=tuple(not_evaluated),
        )
    if decider is not None:
        return PreviewReport(
            **base,
            action="block",
            rule=decider.name,
            text=decider.text,
            warnings=tuple(warnings),
            rules=tuple(matched),
            not_evaluated=tuple(not_evaluated),
        )
    return PreviewReport(
        **base,
        action="warn" if warnings else "allow",
        warnings=tuple(warnings),
        rules=tuple(matched),
    )


__all__ = [
    "BUILTIN_JUDGE",
    "BUILTIN_PASS",
    "BUILTIN_REFUSE_CREDENTIAL",
    "BUILTIN_REFUSE_CRITICAL",
    "PreviewReport",
    "RulePreview",
    "preview_tool_call",
]
