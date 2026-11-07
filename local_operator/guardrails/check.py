"""The ONE pre-execution guardrail checkpoint the agent loop calls per tool call.

WHY. On 2026-11-04 an unattended, auto-approved session deleted the whole
local-operator state root with one shell call (an empty-substitution delete of
the state root). The approval gate is the only other pre-execution checkpoint,
and under ``tool_approval_mode: auto`` or ``--yolo`` it approves everything. This
module is the second checkpoint, placed by ``harness/loop.py::_runner_result``
AFTER approval and BEFORE ``tool.execute``, so:

* a human "approve" in ask mode still meets the CRITICAL hard stop;
* auto/yolo runs pass through it — there is no flag that skips it;
* in-process janitors (session cleanup, disk reclaim) never come here, so they
  keep working: only MODEL tool calls are checked.

Order, first refusal wins:

1. Built-in dangerous-delete (bash only in v1): CRITICAL refuses with no model
   call; HIGH goes to the forked judge; LOW passes with zero cost.
   ``guardrails.critical_override`` (operator-authorised only, see
   ``harness/approval.loosening_is_authorised``) turns CRITICAL into HIGH —
   never into a pass.
2. User rules (``<config_dir>/rules/*.md``): block / review / warn. They can
   only add restriction.

The refusal is returned as the tool RESULT (not stderr) so the model reads it
as the outcome of its call, and its first words are the stable marker
``BLOCKED by guardrail`` that the system prompt tells the agent is final.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_operator.guardrails.classify import Risk, Verdict, classify_bash
from local_operator.guardrails.judge import (
    GUARDRAILS_JUDGE_TIMEOUT_DEFAULT,
    JudgeBrief,
    JudgeCall,
    run_judge,
)
from local_operator.guardrails.rules import Rule, call_text, load_rules

logger = logging.getLogger(__name__)

# S1: a guard's resolved target is new data, not a copy already scrubbed by
# tool-result handling. Scope the same redactor to judge, journal and refusal
# boundaries; ContextVar keeps concurrent calls' credential sets isolated.
_OUTPUT_REDACTOR: ContextVar[Callable[[str], str] | None] = ContextVar(
    "guardrail_output_redactor", default=None
)


def _safe_text(text: str) -> str:
    from local_operator.secrets.runtime import scrub

    redact = _OUTPUT_REDACTOR.get() or scrub
    result = redact(text)
    if not isinstance(result, str):
        raise TypeError("invalid redactor result")
    return result


#: The stable head of every refusal. Model-facing (the system prompt names it),
#: so English by design and never localised.
BLOCK_MARKER = "BLOCKED by guardrail"
DANGEROUS_DELETE_RULE = "dangerous-delete"

# Config keys (registry rows in settings_io; defaults pinned by test_settings_io).
GUARDRAILS_ENABLED_PATH: tuple[str, ...] = ("guardrails", "enabled")
GUARDRAILS_ENABLED_DEFAULT = True
GUARDRAILS_CRITICAL_OVERRIDE_PATH: tuple[str, ...] = ("guardrails", "critical_override")
GUARDRAILS_CRITICAL_OVERRIDE_DEFAULT = False
GUARDRAILS_JUDGE_TIMEOUT_PATH: tuple[str, ...] = ("guardrails", "judge_timeout_s")
GUARDRAILS_RULES_DIR_PATH: tuple[str, ...] = ("guardrails", "rules_dir")
GUARDRAILS_RULES_DIR_DEFAULT = ""
_MIN_TIMEOUT_S = 1.0
_MAX_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class GuardSettings:
    """The guardrail knobs as they apply to ONE call.

    ``enabled`` governs user rules only: the built-in dangerous-delete check is
    not switchable, because a config file the model can write must not be a way
    to lift the guard that protects that same file. ``critical_override`` is
    honoured only when the host proved an operator authorised it.
    """

    enabled: bool = GUARDRAILS_ENABLED_DEFAULT
    critical_override: bool = GUARDRAILS_CRITICAL_OVERRIDE_DEFAULT
    judge_timeout_s: float = GUARDRAILS_JUDGE_TIMEOUT_DEFAULT
    #: ``<state>/rules`` ALWAYS, then the configured ``rules_dir`` if any (R2-2).
    rules_dirs: tuple[Path, ...] = ()

    @property
    def rules_dir(self) -> Path | None:
        """The configured extra folder, or the default when none (back-compat)."""
        return self.rules_dirs[-1] if self.rules_dirs else None


@dataclass
class GuardOutcome:
    """``blocked`` → the call must not run and ``text`` is its result; otherwise
    ``warnings`` are appended to the real result after it runs."""

    blocked: bool
    text: str = ""
    rule: str = ""
    risk: str = ""
    warnings: list[str] = field(default_factory=list)
    #: The bash classification, kept so the tool can annotate a timeout.
    destructive: bool = False


#: The keys whose non-default value LOOSENS the guard (S5/R10). Each is
#: honoured only when the host proves an operator — not the model, not a shell
#: edit — wrote it in this process (``harness.approval.loosening_is_authorised``);
#: otherwise the safe default stands. ``judge_timeout_s`` is not here: a
#: shorter timeout blocks MORE (fail closed), and the bound caps a longer one.
LOOSENING_KEYS = frozenset({"guardrails.critical_override", "guardrails.enabled"})


def read_settings(
    *, authorised: frozenset[str] = frozenset(), override_authorised: bool | None = None
) -> GuardSettings:
    """Read ``guardrails.*`` at call time; any failure returns the safe defaults.

    Fresh per call, like ``tools.builtin._search_interception_config``, so an
    edit lands on the next call. An unreadable config is never "moved aside"
    here (that would rename the operator's file mid-turn) — defaults stand.

    ``authorised`` names the :data:`LOOSENING_KEYS` an operator set in this
    process. A loosening key the file carries WITHOUT that proof is read as its
    default: an agent cannot switch off the user's rules, repoint them at an
    empty folder, or lift the CRITICAL refusal by editing config.yml.
    ``override_authorised`` is the older single-key spelling.
    """
    if override_authorised:
        authorised = authorised | {"guardrails.critical_override"}
    from local_operator.config import CONFIG_FILE_NAME, ConfigManager
    from local_operator.paths import config_dir
    from local_operator.tools.shell_env import _config_file_is_unreadable

    default_rules = config_dir() / "rules"
    try:
        if _config_file_is_unreadable(config_dir() / CONFIG_FILE_NAME):
            return GuardSettings(rules_dirs=(default_rules,))
        config = ConfigManager(config_dir())
        enabled = config.get_nested_value(GUARDRAILS_ENABLED_PATH, GUARDRAILS_ENABLED_DEFAULT)
        override = config.get_nested_value(
            GUARDRAILS_CRITICAL_OVERRIDE_PATH, GUARDRAILS_CRITICAL_OVERRIDE_DEFAULT
        )
        timeout_raw = config.get_nested_value(
            GUARDRAILS_JUDGE_TIMEOUT_PATH, GUARDRAILS_JUDGE_TIMEOUT_DEFAULT
        )
        rules_raw = config.get_nested_value(GUARDRAILS_RULES_DIR_PATH, GUARDRAILS_RULES_DIR_DEFAULT)
    except Exception:  # noqa: BLE001 — config trouble must never weaken the guard
        return GuardSettings(rules_dirs=(default_rules,))
    try:
        timeout = min(max(float(timeout_raw), _MIN_TIMEOUT_S), _MAX_TIMEOUT_S)
    except (TypeError, ValueError):
        timeout = GUARDRAILS_JUDGE_TIMEOUT_DEFAULT
    # R2-2: ``rules_dir`` ADDS a folder; ``<state>/rules`` is always loaded.
    # Repointing can then never hide the operator's own rules, so the key needs
    # no loosening gate (gating it made a configured folder vanish after every
    # restart, since boot-time contents are never attributed).
    rules_dirs: tuple[Path, ...] = (default_rules,)
    if isinstance(rules_raw, str) and rules_raw:
        extra = Path(os.path.expanduser(rules_raw))
        if extra != default_rules:
            rules_dirs = (default_rules, extra)
    return GuardSettings(
        # Only a literal ``false`` disables user rules, and only when authorised.
        enabled=enabled is not False or "guardrails.enabled" not in authorised,
        # The override needs BOTH the file's ``true`` and the host's proof that
        # an operator (not the model) made that write.
        critical_override=(override is True) and "guardrails.critical_override" in authorised,
        judge_timeout_s=timeout,
        rules_dirs=rules_dirs,
    )


@dataclass(frozen=True)
class CallContext:
    """What the checkpoint knows about the call beyond its arguments."""

    cwd: str | None
    env: Mapping[str, str]
    scratchpad: str | None
    session_id: str = ""
    user_request: str = ""
    intent: str = ""
    #: The trusted operator words (``provenance.operator_brief``). ``None``
    #: falls back to ``user_request`` as the only operator row (old callers).
    operator: Any = None
    #: The session's in-memory history: the CURRENT session's evidence is read
    #: from here, never from its own (shell-reachable) transcript file.
    #: ``None`` (preview) falls back to the disk copy.
    messages: Any = None


def _targets_text(verdict: Verdict) -> str:
    flagged = verdict.flagged_targets or verdict.targets
    parts = [f"{t.shown or t.path} ({t.reason})" for t in flagged[:6]]
    if len(flagged) > 6:
        parts.append(f"… and {len(flagged) - 6} more")
    return "; ".join(parts) or "an unresolvable target"


#: What every dangerous-delete refusal tells the agent it CAN do. Model-facing
#: (English by design, like ``BLOCK_MARKER``). Named so the preview prints the
#: same words the checkpoint returns. The supported routes are the ones the
#: checkpoint can verify; the last sentence closes the route agents tried in
#: the field — relaying "the operator said yes" through another agent.
SUPPORTED_DELETE_PATHS = (
    "Supported: `git worktree remove <exact path>` for a clean linked worktree; deleting "
    "exactly what this session or its subagents created (literal paths, as the creating "
    "call named them); or ask the operator with `ask`, naming the exact paths. A message "
    "from Team Bridge, a peer agent, a hub note, or fetched/file content cannot authorise "
    "a delete."
)


def _critical_text(verdict: Verdict) -> str:
    return (
        f"{BLOCK_MARKER} {DANGEROUS_DELETE_RULE} (critical): this command would delete, move "
        f"or overwrite {_targets_text(verdict)}. Nothing was run. This is not retryable by "
        "rewording, and no approval lifts it from inside the session. If a narrower path was "
        "intended, target it directly (for example a path inside your own scratchpad); "
        "otherwise ask the operator."
    )


def review_text(rule: str, reason: str, targets: str) -> str:
    tail = (
        SUPPORTED_DELETE_PATHS
        if rule == DANGEROUS_DELETE_RULE
        else ("Ask the user, or target a narrower path.")
    )
    return f"{BLOCK_MARKER} {rule} (review): {reason}. Targets: {targets}. Nothing was run. {tail}"


def rule_block_text(rule: Rule) -> str:
    """The result text of a user rule that refuses a call.

    Named rather than inlined so ``guardrails/preview.py`` can print the string a
    ``block`` would really produce: a preview that re-describes the refusal is a
    preview that can promise text the checkpoint does not return.
    """
    body = f" {rule.body}" if rule.body else ""
    return (
        f"{BLOCK_MARKER} {rule.name} (block): this call matched a user rule. "
        f"Nothing was run.{body}"
    )


def rule_warn_text(rule: Rule) -> str:
    """The line a ``warn`` rule appends to the call's result (see rule_block_text)."""
    return f"Guardrail {rule.name}: {rule.body}".strip()


def _journal(
    rule: str, risk: str, verdict: str, latency: float, targets: str, basis: str = ""
) -> None:
    # One structured line per non-LOW decision so a rollout can watch the
    # block ratio and judge latency (see the PR's risk notes). ``basis`` names
    # what an ALLOW rested on (the creating call, the ask, the operator row).
    logger.info(
        "guardrail.verdict rule=%s risk=%s verdict=%s latency_s=%.3f targets=%s basis=%s",
        _safe_text(rule),
        _safe_text(risk),
        _safe_text(verdict),
        latency,
        _safe_text(targets)[:500],
        _safe_text(basis)[:500],
    )


#: What the built-in dangerous-delete layer does with a ranked call BEFORE any
#: user rule is consulted. Named constants and one predicate, rather than three
#: order-sensitive ``if``s inline, because ``guardrails/preview.py`` has to report
#: the same branch for a call it never runs: a second copy of this order would let
#: the preview promise a rule protects a call the checkpoint refuses first. Keep
#: every branch decision here so a new rank or override lands in ONE place.
BUILTIN_PASS = "pass"
BUILTIN_JUDGE = "judge"
BUILTIN_REFUSE_CRITICAL = "refuse-critical"
BUILTIN_REFUSE_CREDENTIAL = "refuse-credential"


def builtin_branch(verdict: Verdict, *, critical_override: bool) -> str:
    """The branch :func:`_dangerous_delete` takes for this verdict, first match wins.

    Order is the guard's own precedence, not a ranking of severity: CRITICAL
    without an authorised override refuses before the credential check, which
    itself refuses before the judge — so a secret-derived target that also ranks
    CRITICAL is refused for the reason the checkpoint would give it.
    """
    if verdict.risk is Risk.LOW:
        return BUILTIN_PASS
    if verdict.risk is Risk.CRITICAL and not critical_override:
        return BUILTIN_REFUSE_CRITICAL
    if any(t.target.credential_derived for t in verdict.targets):
        return BUILTIN_REFUSE_CREDENTIAL
    return BUILTIN_JUDGE


@dataclass
class DeleteCase:
    """Everything the judge is told about a HIGH delete, and how its ALLOW is checked.

    Built ONCE per call so the brief the judge reads and the verification of
    its answer use the same evidence ids and the same operator rows.
    """

    #: Concrete on-disk paths an ALLOW must cover (globs expanded to matches).
    paths: list[str]
    #: Paths only the operator's own words can cover (overridden CRITICAL).
    operator_only: frozenset[str]
    #: Flagged targets no citation can cover (opaque, unresolved, a glob with
    #: no match). Judged as before this change: context-free, plain ALLOW.
    unverifiable: list[str]
    operator: Any
    evidence: Any
    operator_words: tuple[tuple[str, str], ...]
    excluded: str
    facts: tuple[str, ...]
    env: Mapping[str, str]
    stop: tuple[str, ...]
    home: str
    cwd: str | None

    def _named_by(self, path: str, words: Any) -> bool:
        from local_operator.guardrails.provenance import names_path

        return names_path(words.text, path, home=self.home, cwd=self.cwd, protected=self.stop)

    def verify(self, answer: str) -> tuple[str, str]:
        """``(basis, missing)``: what verified each path, and what did not verify.

        The answer's first line may cite ``evidence=e1,e2``, ``ask=a-1234`` and
        ``operator``. A path verifies when a cited evidence call re-verifies
        (``evidence.verify``), a cited ask that the store proved names it, or a
        cited direct operator row names it. ``missing`` non-empty → block.
        """
        import re

        from local_operator.guardrails.evidence import verify as verify_evidence

        first = (answer or "").strip().splitlines()[0] if (answer or "").strip() else ""
        evidence_ids = [
            e
            for m in re.finditer(r"evidence\s*=\s*([\w,\s]+?)(?=\s+\w+=|\s*$|\s+operator)", first)
            for e in re.split(r"[,\s]+", m.group(1))
            if e
        ]
        ask_ids = re.findall(r"ask\s*=\s*(a-[0-9a-f]+)", first)
        operator = re.search(r"(?<![\w-])operator(?![\w-])", first) is not None
        # ``unverifiable`` targets (opaque, unresolved) keep the pre-provenance
        # contract: the CONTEXT-FREE judge's plain ALLOW decides them, because
        # there is no path to check a citation against. ``_dangerous_delete``
        # never sends such a call to the forked judge.
        by_evidence = verify_evidence(
            evidence_ids, self.paths, self.evidence, env=self.env, stop=self.stop
        )
        basis: list[str] = []
        missing: list[str] = []
        for path in self.paths:
            if path in by_evidence and path not in self.operator_only:
                basis.append(f"{path} created by {by_evidence[path]}")
                continue
            words = next(
                (
                    w
                    for w in self.operator.words
                    if (
                        (w.ask_id and w.ask_id in ask_ids)
                        or (operator and w.label == "operator, direct")
                    )
                    and self._named_by(path, w)
                ),
                None,
            )
            if words is not None:
                where = f"ask {words.ask_id}" if words.ask_id else "the operator's message"
                basis.append(f"{path} named by {where}")
                continue
            missing.append(path)
        if missing:
            more = f" and {len(missing) - 3} more" if len(missing) > 3 else ""
            return "", ", ".join(missing[:3]) + more
        if len(basis) > 6:
            basis = [*basis[:6], f"… and {len(basis) - 6} more"]
        return "; ".join(basis), ""


#: Per-target fact lines in a judge brief. The field case was 21 rig
#: directories in one delete; the judge must see each one's evidence id.
_MAX_FACTS = 64


def build_delete_case(verdict: Verdict, ctx: CallContext) -> DeleteCase:
    """Extract operator words, evidence and per-target facts for a HIGH delete."""
    import glob as _glob

    from local_operator.guardrails import evidence as ev_mod
    from local_operator.guardrails import provenance as prov
    from local_operator.guardrails.classify import _protected

    home = ctx.env.get("HOME") or os.path.expanduser("~")
    prot = _protected(ctx.env, ctx.scratchpad, ctx.session_id)
    stop = tuple(sorted({*prot.homes, *prot.state_roots, *prot.install_roots, "/"}))
    paths: list[str] = []
    unverifiable: list[str] = []
    operator_only: list[str] = []
    for ranked in verdict.flagged_targets:
        t = ranked.target
        if ranked.risk is Risk.CRITICAL and t.path is not None and not t.opaque and not t.pattern:
            # Reached only under an operator-authorised ``critical_override``.
            # History can never vouch for a protected root: only the operator's
            # own words naming it can (``operator`` / ``ask=`` citations).
            operator_only.append(os.path.realpath(t.path))
            continue
        if (
            ranked.risk is Risk.CRITICAL
            or t.path is None
            or t.opaque
            or t.unknown_cwd
            or t.climbs
            or t.empty
            or t.credential_derived
        ):
            unverifiable.append(ranked.path)
            continue
        if t.pattern is not None:
            try:
                matches = sorted(_glob.glob(t.pattern))
            except (OSError, ValueError):
                matches = []
            if not matches:
                unverifiable.append(t.pattern)
            paths.extend(os.path.realpath(m) for m in matches)
        else:
            paths.append(os.path.realpath(t.path))
    paths = list(dict.fromkeys(paths))
    operator_only = [p for p in dict.fromkeys(operator_only) if p not in paths]
    operator = ctx.operator
    if operator is None:
        operator = prov.OperatorBrief()
        if ctx.user_request:
            operator.words.append(prov.OperatorWords("operator, direct", ctx.user_request))
    evidence = ev_mod.gather(
        paths, session_id=ctx.session_id, env=ctx.env, stop=stop, messages=ctx.messages
    )
    facts: list[str] = []
    for path in [*paths, *operator_only][:_MAX_FACTS]:
        ids = [] if path in operator_only else (evidence.covers.get(path) or [])
        named = next(
            (
                w.label + (f" {w.ask_id}" if w.ask_id else "")
                for w in operator.words
                if prov.names_path(w.text, path, home=home, cwd=ctx.cwd, protected=stop)
            ),
            "",
        )
        tree = ""
        if os.path.exists(os.path.join(path, ".git")):
            tree = f"; git: {prov.worktree_state(path)}"
            if prov.is_primary_checkout(path):
                tree += " (primary checkout)"
        facts.append(
            f"{path}: created by this session or its subagents: "
            + (", ".join(ids) if ids else "no evidence")
            + tree
            + f"; operator named it: {'yes (' + named + ')' if named else 'no'}"
        )
    if len(paths) > _MAX_FACTS:
        facts.append(f"… and {len(paths) - _MAX_FACTS} more paths")
    facts.extend(f"{u}: cannot be verified (unresolved, opaque or critical)" for u in unverifiable)
    excluded = ", ".join(f"{n} {kind}" for kind, n in sorted(operator.excluded.items()))
    return DeleteCase(
        paths=[*paths, *operator_only],
        operator_only=frozenset(operator_only),
        unverifiable=unverifiable,
        operator=operator,
        evidence=evidence,
        operator_words=tuple((w.label, w.text) for w in operator.words),
        excluded=excluded,
        facts=tuple(facts),
        env=ctx.env,
        stop=stop,
        home=home,
        cwd=ctx.cwd,
    )


#: Builtins that write a file at a path argument, and that argument's key. The
#: shell resolver ranks bash writes; these tools write without a shell, so the
#: guard's own evidence (transcripts, lineage, ask stores, the ledger — see
#: ``classify._evidence_reason``) would otherwise be forgeable with one
#: ``write`` call (round-1 follow-up to R1-MAJOR-1). Only those files are
#: refused here; every other write/edit keeps its existing behaviour.
FILE_DESTINATION_TOOLS: Mapping[str, str] = {"write": "path", "edit": "path", "browser": "path"}


def evidence_write_verdict(
    tool_name: str, args: Mapping[str, Any], ctx: CallContext
) -> Verdict | None:
    """A CRITICAL verdict when this file-writing call targets the guard's evidence."""
    from local_operator.guardrails.classify import (
        RankedTarget,
        _evidence_reason,
        _protected,
    )
    from local_operator.guardrails.classify import _spellings as spellings
    from local_operator.guardrails.resolve import Target

    key = FILE_DESTINATION_TOOLS.get(tool_name)
    raw = args.get(key) if key else None
    if not isinstance(raw, str) or not raw.strip() or "://" in raw:
        # Empty, or an internal URL (``scratchpad://`` lands in the session's
        # own pad; other schemes are refused by the tool itself).
        return None
    home = ctx.env.get("HOME") or os.path.expanduser("~")
    text = home + raw[1:] if raw == "~" or raw.startswith("~/") else raw
    if not os.path.isabs(text):
        text = os.path.join(ctx.cwd or os.getcwd(), text)
    target = Target(path=os.path.normpath(text), op="overwrite", program=tool_name)
    prot = _protected(ctx.env, ctx.scratchpad, ctx.session_id)
    for spelling in spellings(text):
        reason = _evidence_reason(spelling, target, prot)
        if reason is not None:
            return Verdict(
                risk=Risk.CRITICAL, targets=[RankedTarget(target, Risk.CRITICAL, reason)]
            )
    return None


async def _dangerous_delete(
    args: Mapping[str, Any],
    ctx: CallContext,
    settings: GuardSettings,
    judge: JudgeCall | None,
    rule: Rule | None,
    forked_judge: JudgeCall | None = None,
) -> GuardOutcome | None:
    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        return None
    verdict = classify_bash(
        command, cwd=ctx.cwd, env=ctx.env, scratchpad=ctx.scratchpad, session_id=ctx.session_id
    )
    branch = builtin_branch(verdict, critical_override=settings.critical_override)
    if branch == BUILTIN_PASS:
        # Ordinary delete: runs with zero model calls. Still marked destructive
        # so a timeout mid-delete tells the agent to check before retrying.
        return GuardOutcome(blocked=False, destructive=True) if verdict.targets else None
    targets = _targets_text(verdict)
    risk = verdict.risk
    if branch == BUILTIN_REFUSE_CRITICAL:
        _journal(DANGEROUS_DELETE_RULE, "critical", "block", 0.0, targets)
        return GuardOutcome(
            blocked=True,
            text=_critical_text(verdict),
            rule=DANGEROUS_DELETE_RULE,
            risk="critical",
            destructive=True,
        )
    if branch == BUILTIN_REFUSE_CREDENTIAL:
        # Privacy-hidden roots cannot receive meaningful model consent. This
        # refuses even an ALLOW/critical override, while leaving known protected
        # roots on the existing CRITICAL path above. Never expose the env value.
        _journal(DANGEROUS_DELETE_RULE, "high", "block", 0.0, targets)
        return GuardOutcome(
            blocked=True,
            text=review_text(DANGEROUS_DELETE_RULE, "an unresolvable target", targets),
            rule=DANGEROUS_DELETE_RULE,
            risk="high",
            destructive=True,
        )
    case = build_delete_case(verdict, ctx)
    result = await run_judge(
        JudgeBrief(
            tool="bash",
            call_text=command,
            flagged=targets,
            policy=rule.body if rule else "",
            question=rule.question if rule else "",
            user_request=ctx.user_request,
            intent=ctx.intent,
            operator_words=case.operator_words,
            excluded=case.excluded,
            target_facts=case.facts,
            evidence=case.evidence.describe(),
        ),
        # The fork reads the session's history, injected text included, so it
        # only judges calls whose every target can be VERIFIED against what it
        # cites. An opaque/unresolved target keeps the context-free judge.
        (forked_judge or judge) if not case.unverifiable else judge,
        timeout=settings.judge_timeout_s,
    )
    risk_name = "critical-overridden" if risk is Risk.CRITICAL else "high"
    basis = ""
    reason = result.reason
    if result.allowed:
        # The judge's ALLOW is a CLAIM; the checkpoint verifies what it cited
        # (see ``guardrails/evidence.py``). A forked judge read the whole
        # session, injected text included, so an uncited or unverifiable
        # ALLOW is a block.
        basis, missing = case.verify(result.answer)
        if missing:
            reason = f"the safety check's approval could not be verified for {missing}"
    allowed = result.allowed and not reason
    _journal(
        DANGEROUS_DELETE_RULE,
        risk_name,
        "allow" if allowed else "block",
        result.latency_s,
        targets,
        basis,
    )
    if allowed:
        outcome = GuardOutcome(
            blocked=False, rule=DANGEROUS_DELETE_RULE, risk=risk_name, destructive=True
        )
        if basis:
            outcome.warnings.append(f"Guardrail {DANGEROUS_DELETE_RULE}: allowed — {basis}.")
        return outcome
    return GuardOutcome(
        blocked=True,
        text=review_text(DANGEROUS_DELETE_RULE, reason, targets),
        rule=DANGEROUS_DELETE_RULE,
        risk=risk_name,
        destructive=True,
    )


async def check_tool_call(
    tool_name: str,
    args: Mapping[str, Any],
    ctx: CallContext,
    *,
    judge: JudgeCall | None,
    settings: GuardSettings,
    forked_judge: JudgeCall | None = None,
) -> GuardOutcome | None:
    """Keep analysis-derived data private before any outward boundary.

    ``forked_judge`` (optional) answers dangerous-delete reviews from a fork of
    the session's own prefix, so it can see the history it is judging; user
    rules keep the context-free ``judge``.

    Use the existing registry/shape scrubber, not a second masking algorithm.
    Credential values stay local to this call; they are never registered,
    journaled, or included in the isolated model request. Fail closed without
    logging an exception that might itself contain sensitive text.
    """
    from local_operator.redaction_shapes import scrub_values
    from local_operator.secrets.runtime import scrub
    from local_operator.tools.shell_env import is_credential_shaped

    values = [value for name, value in ctx.env.items() if is_credential_shaped(name)]

    def redact(text: str) -> str:
        return scrub_values(scrub(text), values)

    def private(call: JudgeCall) -> JudgeCall:
        async def private_judge(system: str, prompt: str) -> str:
            try:
                return _safe_text(await call(_safe_text(system), _safe_text(prompt)))
            except Exception:  # no raw exception/traceback may cross this boundary
                return "BLOCK"

        return private_judge

    token = _OUTPUT_REDACTOR.set(redact)
    try:
        outcome = await _check_tool_call(
            tool_name,
            args,
            ctx,
            judge=private(judge) if judge is not None else None,
            settings=settings,
            forked_judge=private(forked_judge) if forked_judge is not None else None,
        )
        if outcome is not None:
            outcome.text = _safe_text(outcome.text)
            outcome.rule = _safe_text(outcome.rule)
            outcome.warnings = [_safe_text(warning) for warning in outcome.warnings]
        return outcome
    except Exception:
        return GuardOutcome(
            blocked=True,
            text=f"{BLOCK_MARKER} (error): the safety check failed, so this "
            "command was not run. Ask the user how to proceed.",
            rule="guardrail-error",
        )
    finally:
        _OUTPUT_REDACTOR.reset(token)


async def _check_tool_call(
    tool_name: str,
    args: Mapping[str, Any],
    ctx: CallContext,
    *,
    judge: JudgeCall | None,
    settings: GuardSettings,
    forked_judge: JudgeCall | None = None,
) -> GuardOutcome | None:
    """``None`` → run as-is; a non-blocking outcome may carry warnings."""
    rules = load_rules(*settings.rules_dirs) if settings.enabled else load_rules(None)
    builtin = next((r for r in rules.builtin if r.name == DANGEROUS_DELETE_RULE), None)
    outcome: GuardOutcome | None = None
    if tool_name == "bash":
        outcome = await _dangerous_delete(args, ctx, settings, judge, builtin, forked_judge)
        if outcome is not None and outcome.blocked:
            return outcome
    elif tool_name in FILE_DESTINATION_TOOLS:
        verdict = evidence_write_verdict(tool_name, args, ctx)
        if verdict is not None:
            targets = _targets_text(verdict)
            _journal(DANGEROUS_DELETE_RULE, "critical", "block", 0.0, targets)
            return GuardOutcome(
                blocked=True,
                text=_critical_text(verdict),
                rule=DANGEROUS_DELETE_RULE,
                risk="critical",
            )
    warnings: list[str] = []
    for rule in rules.user:
        if not rule.matches(tool_name, args):
            continue
        if rule.action == "warn":
            warnings.append(rule_warn_text(rule))
            _journal(rule.name, "rule", "warn", 0.0, "")
            continue
        if rule.action == "block":
            _journal(rule.name, "rule", "block", 0.0, "")
            return GuardOutcome(
                blocked=True,
                text=rule_block_text(rule),
                rule=rule.name,
                risk="rule",
                destructive=bool(outcome and outcome.destructive),
            )
        result = await run_judge(
            JudgeBrief(
                tool=tool_name,
                call_text=call_text(tool_name, args),
                flagged=f"matched user rule {rule.name}",
                policy=rule.body,
                question=rule.question,
                user_request=ctx.user_request,
                intent=ctx.intent,
            ),
            judge,
            timeout=settings.judge_timeout_s,
        )
        _journal(rule.name, "rule", "allow" if result.allowed else "block", result.latency_s, "")
        if not result.allowed:
            return GuardOutcome(
                blocked=True,
                text=review_text(rule.name, result.reason, f"this {tool_name} call"),
                rule=rule.name,
                risk="rule",
                destructive=bool(outcome and outcome.destructive),
            )
    if warnings or outcome is not None:
        base = outcome or GuardOutcome(blocked=False)
        base.warnings.extend(warnings)
        return base
    return None


def last_user_request(messages: Any) -> str:
    """The last thing a PERSON wrote, for the judge's brief.

    Harness-minted rows (goal continuations, wake deliveries, notices) are
    skipped by their ``harness_injected`` stamp and by their known opening
    words: in the incident the goal loop's continuation read like a user
    instruction, and a judge that took it for one would approve on the
    harness's behalf.
    """
    from local_operator.harness.rows import is_harness_chrome, is_harness_injection

    try:
        for message in reversed(list(messages or ())):
            if getattr(message, "role", None) != "user":
                continue
            if is_harness_injection(message):
                continue
            text = getattr(message, "text", "")
            if not isinstance(text, str):
                continue
            if not text.strip() or is_harness_chrome(text):
                continue
            return text
    except Exception:  # noqa: BLE001 — a missing brief field degrades, never raises
        logger.debug("guardrail: could not read the last user request", exc_info=True)
    return ""
