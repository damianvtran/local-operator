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
from collections.abc import Mapping
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
    rules_dir: Path | None = None


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


def read_settings(*, override_authorised: bool) -> GuardSettings:
    """Read ``guardrails.*`` at call time; any failure returns the safe defaults.

    Fresh per call, like ``tools.builtin._search_interception_config``, so an
    edit lands on the next call. An unreadable config is never "moved aside"
    here (that would rename the operator's file mid-turn) — defaults stand.
    """
    from local_operator.config import CONFIG_FILE_NAME, ConfigManager
    from local_operator.paths import config_dir
    from local_operator.tools.shell_env import _config_file_is_unreadable

    default_rules = config_dir() / "rules"
    try:
        if _config_file_is_unreadable(config_dir() / CONFIG_FILE_NAME):
            return GuardSettings(rules_dir=default_rules)
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
        return GuardSettings(rules_dir=default_rules)
    try:
        timeout = min(max(float(timeout_raw), _MIN_TIMEOUT_S), _MAX_TIMEOUT_S)
    except (TypeError, ValueError):
        timeout = GUARDRAILS_JUDGE_TIMEOUT_DEFAULT
    rules_dir = (
        Path(os.path.expanduser(rules_raw))
        if isinstance(rules_raw, str) and rules_raw
        else default_rules
    )
    return GuardSettings(
        # Only a literal ``false`` disables user rules; junk reads as on.
        enabled=enabled is not False,
        # The override needs BOTH the file's ``true`` and the host's proof that
        # an operator (not the model) made that write.
        critical_override=(override is True) and override_authorised,
        judge_timeout_s=timeout,
        rules_dir=rules_dir,
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


def _targets_text(verdict: Verdict) -> str:
    flagged = verdict.flagged_targets or verdict.targets
    parts = [f"{t.path} ({t.reason})" for t in flagged[:6]]
    if len(flagged) > 6:
        parts.append(f"… and {len(flagged) - 6} more")
    return "; ".join(parts) or "an unresolvable target"


def _critical_text(verdict: Verdict) -> str:
    return (
        f"{BLOCK_MARKER} {DANGEROUS_DELETE_RULE} (critical): this command would delete, move "
        f"or overwrite {_targets_text(verdict)}. Nothing was run. This is not retryable by "
        "rewording. If a narrower path was intended, target it directly (for example a path "
        "inside your own scratchpad); otherwise ask the user."
    )


def _review_text(rule: str, reason: str, targets: str) -> str:
    return (
        f"{BLOCK_MARKER} {rule} (review): {reason}. Targets: {targets}. Nothing was run. "
        "Ask the user, or target a narrower path."
    )


def _journal(rule: str, risk: str, verdict: str, latency: float, targets: str) -> None:
    # One structured line per non-LOW decision so a rollout can watch the
    # block ratio and judge latency (see the PR's risk notes).
    logger.info(
        "guardrail.verdict rule=%s risk=%s verdict=%s latency_s=%.3f targets=%s",
        rule,
        risk,
        verdict,
        latency,
        targets[:500],
    )


async def _dangerous_delete(
    args: Mapping[str, Any],
    ctx: CallContext,
    settings: GuardSettings,
    judge: JudgeCall | None,
    rule: Rule | None,
) -> GuardOutcome | None:
    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        return None
    verdict = classify_bash(
        command, cwd=ctx.cwd, env=ctx.env, scratchpad=ctx.scratchpad, session_id=ctx.session_id
    )
    if verdict.risk is Risk.LOW:
        # Ordinary delete: runs with zero model calls. Still marked destructive
        # so a timeout mid-delete tells the agent to check before retrying.
        return GuardOutcome(blocked=False, destructive=True) if verdict.targets else None
    targets = _targets_text(verdict)
    risk = verdict.risk
    if risk is Risk.CRITICAL and not settings.critical_override:
        _journal(DANGEROUS_DELETE_RULE, "critical", "block", 0.0, targets)
        return GuardOutcome(
            blocked=True,
            text=_critical_text(verdict),
            rule=DANGEROUS_DELETE_RULE,
            risk="critical",
            destructive=True,
        )
    result = await run_judge(
        JudgeBrief(
            tool="bash",
            call_text=command,
            flagged=targets,
            policy=rule.body if rule else "",
            question=rule.question if rule else "",
            user_request=ctx.user_request,
            intent=ctx.intent,
        ),
        judge,
        timeout=settings.judge_timeout_s,
    )
    risk_name = "critical-overridden" if risk is Risk.CRITICAL else "high"
    _journal(
        DANGEROUS_DELETE_RULE,
        risk_name,
        "allow" if result.allowed else "block",
        result.latency_s,
        targets,
    )
    if result.allowed:
        return GuardOutcome(
            blocked=False, rule=DANGEROUS_DELETE_RULE, risk=risk_name, destructive=True
        )
    return GuardOutcome(
        blocked=True,
        text=_review_text(DANGEROUS_DELETE_RULE, result.reason, targets),
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
) -> GuardOutcome | None:
    """``None`` → run as-is; a non-blocking outcome may carry warnings."""
    rules = load_rules(settings.rules_dir if settings.enabled else None)
    builtin = next((r for r in rules.builtin if r.name == DANGEROUS_DELETE_RULE), None)
    outcome: GuardOutcome | None = None
    if tool_name == "bash":
        outcome = await _dangerous_delete(args, ctx, settings, judge, builtin)
        if outcome is not None and outcome.blocked:
            return outcome
    warnings: list[str] = []
    for rule in rules.user:
        if not rule.matches(tool_name, args):
            continue
        if rule.action == "warn":
            warnings.append(f"Guardrail {rule.name}: {rule.body}".strip())
            _journal(rule.name, "rule", "warn", 0.0, "")
            continue
        if rule.action == "block":
            _journal(rule.name, "rule", "block", 0.0, "")
            body = f" {rule.body}" if rule.body else ""
            return GuardOutcome(
                blocked=True,
                text=f"{BLOCK_MARKER} {rule.name} (block): this call matched a user rule. "
                f"Nothing was run.{body}",
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
                text=_review_text(rule.name, result.reason, f"this {tool_name} call"),
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
