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
loads the rule files its settings name through ``load_rules``, reuses the
checkpoint's own matching (``Rule.matches``/``call_text``) and the checkpoint's
own message builders, so the ``block`` text and the ``warn`` line printed here
are the same strings ``check_tool_call`` returns rather than a second description
of them that can drift.

WHAT IT CANNOT TELL YOU (the guide says this too). The preview evaluates ONE call
that you hand it: it does not prove the rule catches the forms you did not think
of, it does not run the tool, it does not ask the judge — a ``review`` rule is
reported as *what the judge would be asked*, never as a verdict — and it does not
model another session's config. Where the judge's answer decides, the report says
the answer is not known here instead of guessing it, and it models the checkpoint's
control flow rather than a simplified one: a ``review`` does not end evaluation
(the loop reads on when the judge allows), so a ``block`` behind one is reported
as reachable — that block is what refuses the call on an allow.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from local_operator.guardrails.check import (
    BUILTIN_JUDGE,
    BUILTIN_PASS,
    BUILTIN_REFUSE_CREDENTIAL,
    BUILTIN_REFUSE_CRITICAL,
    DANGEROUS_DELETE_RULE,
    FILE_DESTINATION_TOOLS,
    CallContext,
    GuardSettings,
    _critical_text,
    _safe_text,
    _targets_text,
    build_delete_case,
    builtin_branch,
    evidence_write_verdict,
    read_settings,
    review_text,
    rule_block_text,
    rule_warn_text,
)
from local_operator.guardrails.classify import Verdict, classify_bash
from local_operator.guardrails.rules import call_text, load_rules

#: ``""`` means the built-in layer does not apply (not bash, or no command text)
#: and is the only value a *rule* decision can report; every other value is the
#: branch ``check.builtin_branch`` chose, imported rather than re-spelled so the
#: preview cannot hold a second copy of the checkpoint's precedence.
_REFUSING = frozenset({BUILTIN_REFUSE_CRITICAL, BUILTIN_REFUSE_CREDENTIAL})

_LIMITS = (
    "not proven: this is ONE call against the rule files this preview loaded "
    "({dirs}). It does not run the call, does not ask the judge, does not read the "
    "conversation, and says nothing about forms of the call you did not hand it."
)


@dataclass(frozen=True)
class RulePreview:
    """One matching rule, and what the agent sees because of it.

    ``text`` is the exact string the checkpoint returns (``block``) or appends to
    the result (``warn``). It is empty for ``review``, where the text carries the
    judge's reason — an answer the preview must not fabricate; ``question`` and
    ``policy`` are what that answer would be asked about.

    ``gated_by`` names the ``review`` rule(s) above this one, comma-separated. The
    checkpoint only reads the next matching rule when the judge ALLOWED the review
    it just asked, so a rule with a gate is reached conditionally and its text is
    conditional too. Empty means the call reaches this rule unconditionally.
    """

    name: str
    action: str
    text: str = ""
    question: str = ""
    policy: str = ""
    gated_by: str = ""


@dataclass(frozen=True)
class PreviewReport:
    """What the checkpoint would do with one call, as far as is statically knowable.

    ``action`` is the furthest the preview can get WITHOUT an answer from the
    judge: ``refuse`` (the built-in layer refuses it before the rules run),
    ``block`` (a rule refuses it with no judge in front of that rule), ``review``
    (a judge's answer decides what happens next, and a ``block`` further down can
    still refuse the call if it allows), ``warn`` (nothing refuses it, so it runs
    with the ``warnings`` lines appended) or ``allow`` (nothing matches, it runs).

    ``allow`` and ``warn`` are only the whole story when no judge stands in front
    of the call: with ``builtin_effect == BUILTIN_JUDGE`` the built-in layer sends
    it to a judge first, and ``render`` says the outcome is undecided rather than
    claiming the call runs.

    ``warnings`` holds the lines the call's result will carry. A refusal carries
    none, so a ``block`` — and any call a later ``block`` will refuse — reports
    none: the ``warn`` rules that matched are under ``rules``, marked discarded.
    """

    tool: str
    call_text: str
    action: str = "allow"
    rule: str = ""
    text: str = ""
    warnings: tuple[str, ...] = ()
    rules: tuple[RulePreview, ...] = ()
    #: Rules that match this call but are never reached: a ``block`` rule above
    #: refuses before them in every branch (directly, or after the judge it is
    #: gated by denies the call too). A rule merely behind a ``review`` is NOT in
    #: here — the checkpoint keeps reading when the judge allows.
    not_evaluated: tuple[str, ...] = ()
    builtin_risk: str = ""
    builtin_targets: str = ""
    builtin_effect: str = ""
    #: The built-in layer's own refusal text (CRITICAL / credential), byte for
    #: byte what the checkpoint returns; for a judge branch, its denial shape.
    builtin_text: str = ""
    #: For a judge branch: the per-target facts and the evidence list the
    #: judge would be shown (``check.build_delete_case``), so a preview names
    #: the creating call the checkpoint would accept, or says there is none.
    builtin_facts: tuple[str, ...] = ()
    #: The rule directories this preview actually loaded, so the report states its
    #: own basis instead of calling a candidate folder "the current rules".
    rule_dirs: tuple[str, ...] = ()

    def render(self) -> str:
        """The report as text an agent reads back from its ``eval`` output."""
        lines = [f"preview: {self.tool}  {_one_line(self.call_text)}"]
        if self.builtin_effect:
            lines.append(
                f"built-in {DANGEROUS_DELETE_RULE}: risk={self.builtin_risk or 'LOW'} "
                f"targets={self.builtin_targets} → {_EFFECT_WORDS[self.builtin_effect]}"
            )
            for fact in self.builtin_facts:
                lines.append(f"  {fact}")
            if self.builtin_text:
                prefix = (
                    "on denial the agent sees: "
                    if self.builtin_effect == BUILTIN_JUDGE
                    else "the agent sees: "
                )
                lines.append(f"  {prefix}{self.builtin_text}")
        for index, entry in enumerate(self.rules):
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
            lines.append(f"  {self._label(index)}: {entry.text}")
        if self.not_evaluated:
            lines.append(
                "never reached — a block rule above refuses this call before them: "
                + ", ".join(self.not_evaluated)
            )
        judge_first = self.builtin_effect == BUILTIN_JUDGE
        if self.action == "block":
            if judge_first:
                lines.append(
                    f"verdict: BLOCK by {self.rule} — if the judge allows this call, "
                    "the text above is its result"
                )
            else:
                lines.append(f"verdict: BLOCK by {self.rule} — the text above is the call's result")
        elif self.action == "review":
            later_block = next((entry for entry in self.rules if entry.action == "block"), None)
            if later_block is not None:
                # The checkpoint returns only when the judge DENIES; on an allow
                # it keeps reading and this block is what refuses the call. Say
                # both, and say the answer is unknown — calling the later rule
                # unreachable would report the one case that really blocks as the
                # one case that cannot happen.
                lines.append(
                    f"verdict: REVIEW by {self.rule} — the judge decides and its answer is "
                    f"not known here. A denial refuses the call at {self.rule}; if every "
                    f"review above allows, {later_block.name} still refuses it"
                )
            else:
                lines.append(
                    f"verdict: REVIEW by {self.rule} — the judge decides; a denial refuses "
                    "the call, and its answer is not known here"
                )
        elif judge_first:
            # Nothing here can be decided: the built-in layer sends the call to a
            # judge before the rules are read, so "the call runs" would assert an
            # answer this module cannot know.
            tail = (
                "it runs and the line above is appended to the result"
                if self.action == "warn"
                else "no rule matches and nothing is added to the result"
            )
            lines.append(
                "verdict: undecided here — the built-in layer sends this call to the judge "
                f"first. If it allows, {tail}; if it denies, the call is refused"
            )
        elif self.action == "allow":
            lines.append("verdict: ALLOW — no rule matches, nothing is added to the result")
        elif self.action == "warn":
            lines.append("verdict: WARN — the call runs; the line above is appended to its result")
        else:
            lines.append(
                f"decision made before your rules: the built-in layer refuses this call "
                f"({_EFFECT_WORDS[self.builtin_effect]}), so no user rule is consulted"
            )
        lines.append(_LIMITS.format(dirs=", ".join(self.rule_dirs) or "(none)"))
        return "\n".join(lines)

    def _label(self, index: int) -> str:
        """What happens to the ``index``-th match's text — stated, not assumed.

        A ``warn`` line is discarded by ANY later ``block``: the checkpoint returns
        the refusal without extending its warnings, whether the block fires or a
        gate in front of it denies the call first. So the line is reported as
        discarded rather than promised, because the agent never sees it.
        """
        entry = self.rules[index]
        later = self.rules[index + 1 :]
        if entry.action == "block":
            if entry.gated_by:
                return f"reached only if the judge allows {entry.gated_by}; then agent sees"
            return "agent sees"
        blocking = next((other.name for other in later if other.action == "block"), "")
        if blocking:
            return f"discarded — {blocking} refuses this call, and a refusal carries no warnings"
        gating = next((other.name for other in later if other.action == "review"), "")
        if gating:
            return f"appended only if the judge allows {gating}"
        return "appended to the result"


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


def preview_tool_call(
    tool_name: str,
    args: Mapping[str, Any],
    ctx: CallContext | None = None,
    *,
    settings: GuardSettings | None = None,
    authorised: frozenset[str] = frozenset(),
) -> PreviewReport:
    """Resolve one call against the rules this process would load. Nothing runs.

    ``settings`` defaults to :func:`read_settings` called with ``authorised=``,
    i.e. the same ``guardrails.*`` values the next real call would read. Pass the
    keys when the caller knows them: the harness passes
    ``config.guard_override_authorised()`` into the checkpoint
    (``harness/loop.py``), and both loosening keys are honoured only for a key in
    that set — so without this argument an operator-authorised
    ``critical_override: true`` or ``enabled: false`` is invisible here and the
    preview comes out STRICTER than the running process. Never looser: the gate
    only lifts defaults, it never adds one. Pass ``settings=`` to model a folder
    or configuration this process cannot read at all; it is used as given, with no
    re-reading, and the report names the directories it actually loaded.

    ``ctx`` supplies the cwd/env/scratchpad/session the built-in layer resolves
    paths against; its conversation fields (``user_request``, ``intent``) are
    ignored, because the preview never reads the conversation.

    Returns the action the preview can establish plus, for ``block`` and ``warn``,
    the exact text ``check_tool_call`` would produce. ``review`` carries the
    judge's question and policy and no text: the refusal text would contain an
    answer the judge has not given yet. A ``review`` is not where evaluation stops
    — the checkpoint returns only when the judge DENIES, and keeps reading its
    rules when it allows — so a ``block`` after a review is reported as reachable.
    """
    context = ctx or CallContext(cwd=None, env={}, scratchpad=None)
    live = settings if settings is not None else read_settings(authorised=authorised)
    rules = load_rules(*live.rules_dirs) if live.enabled else load_rules(None)

    verdict: Verdict | None = None
    effect = ""
    builtin_text = ""
    builtin_facts: tuple[str, ...] = ()
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
            effect = builtin_branch(verdict, critical_override=live.critical_override)
            if effect == BUILTIN_REFUSE_CRITICAL:
                builtin_text = _critical_text(verdict)
            elif effect == BUILTIN_REFUSE_CREDENTIAL:
                builtin_text = review_text(
                    DANGEROUS_DELETE_RULE, "an unresolvable target", _targets_text(verdict)
                )
            elif effect == BUILTIN_JUDGE:
                case = build_delete_case(verdict, context)
                builtin_facts = (*case.facts, *case.evidence.describe())
                if case.excluded:
                    builtin_facts += (f"excluded as untrusted: {case.excluded}",)
                builtin_text = review_text(
                    DANGEROUS_DELETE_RULE, "<judge reason>", _targets_text(verdict)
                )

    elif tool_name in FILE_DESTINATION_TOOLS:
        # The checkpoint refuses a write/edit of the guard's own evidence
        # before any rule; report that same branch and text.
        verdict = evidence_write_verdict(tool_name, args, context)
        if verdict is not None:
            effect = BUILTIN_REFUSE_CRITICAL
            builtin_text = _critical_text(verdict)

    base: dict[str, Any] = {
        "tool": tool_name,
        "call_text": call_text(tool_name, args),
        "builtin_risk": verdict.risk.name if verdict is not None else "",
        "builtin_targets": _ranked_targets(verdict),
        "builtin_effect": effect,
        "builtin_text": _safe_text(builtin_text),
        "builtin_facts": tuple(_safe_text(f) for f in builtin_facts),
        "rule_dirs": tuple(str(path) for path in live.rules_dirs) if live.enabled else (),
    }
    if effect in _REFUSING:
        return PreviewReport(**base, action="refuse", rule=DANGEROUS_DELETE_RULE)

    matched: list[RulePreview] = []
    collected: list[str] = []
    #: The ``review`` rules above the current position, in order. A rule behind
    #: one is reached only if the judge allowed every one of them.
    gate: list[str] = []
    blocked: RulePreview | None = None
    unreachable: list[str] = []
    for rule in rules.user:
        if not rule.matches(tool_name, args):
            continue
        if blocked is not None:
            # A block above refuses this call in EVERY branch — directly, or
            # after a gate in front of it denies first — so nothing below runs.
            unreachable.append(rule.name)
            continue
        gated_by = ", ".join(gate)
        if rule.action == "warn":
            text = _safe_text(rule_warn_text(rule))
            collected.append(text)
            matched.append(RulePreview(rule.name, "warn", text=text, gated_by=gated_by))
            continue
        if rule.action == "review":
            matched.append(
                RulePreview(
                    rule.name,
                    "review",
                    question=rule.question,
                    policy=rule.body,
                    gated_by=gated_by,
                )
            )
            gate.append(rule.name)
            continue
        entry = RulePreview(
            rule.name, "block", text=_safe_text(rule_block_text(rule)), gated_by=gated_by
        )
        matched.append(entry)
        blocked = entry

    first_review = next((entry for entry in matched if entry.action == "review"), None)
    if blocked is not None:
        # The checkpoint returns the refusal WITHOUT extending its warnings, so on
        # a call a block refuses no ``warn`` line reaches the agent — not even one
        # that matched before the block, and not when a gate denied first.
        action = "review" if first_review is not None else "block"
        return PreviewReport(
            **base,
            action=action,
            rule=first_review.name if first_review is not None else blocked.name,
            text="" if first_review is not None else blocked.text,
            warnings=(),
            rules=tuple(matched),
            not_evaluated=tuple(unreachable),
        )
    warnings = tuple(collected)
    return PreviewReport(
        **base,
        action="review" if first_review is not None else ("warn" if warnings else "allow"),
        rule=first_review.name if first_review is not None else "",
        warnings=warnings,
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
