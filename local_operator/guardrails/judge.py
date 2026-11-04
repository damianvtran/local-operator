"""The forked, context-free judge for HIGH-risk calls — pre-execution, fail closed.

WHY A FORK AND NOT THE LIVE CONTEXT. In the 2026-11-04 incident the deleting
command was composed after ~312k tokens of a garbled unattended run; a judge
primed by that transcript inherits the confusion that produced the command, and
shipping it per call is slow and costly. So the judge sees a small, bounded
brief: the command, its resolved targets and why they were flagged, the last
thing the USER actually wrote (never harness-injected rows such as goal
continuations), the active goal, and the call's stated intent.

WHY FAIL CLOSED. The judge only runs for calls already ranked dangerous. A
judge that times out, raises, is unwired, or answers anything but the exact
contract must block: "the safety check was unavailable" is never consent. The
block text tells the agent how to proceed (ask the user), so an offline host
degrades to "dangerous deletes need a human", which is the intended posture.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: ``guardrails.judge_timeout_s`` default. Long enough for a small fresh request
#: on a slow provider, short enough that a stuck one does not hold the turn.
GUARDRAILS_JUDGE_TIMEOUT_DEFAULT = 20.0

#: A host-injected single model call: ``(system, user) -> answer text``. No
#: tools, no conversation — see the module docstring.
JudgeCall = Callable[[str, str], Awaitable[str]]

_MAX_COMMAND_CHARS = 2000
_MAX_USER_CHARS = 1000
_MAX_INTENT_CHARS = 300

JUDGE_SYSTEM_PROMPT = (
    "You are a safety check for a coding agent. The agent is about to run a tool "
    "call that a guardrail flagged. You see the call, why it was flagged, and what "
    "the user last asked for. ALLOW only if the user's request clearly calls for "
    "exactly this action on exactly these targets. If in doubt, BLOCK. Answer with "
    'exactly one line: "ALLOW" or "BLOCK: <one-sentence reason>".'
)


@dataclass(frozen=True)
class JudgeBrief:
    tool: str
    call_text: str
    flagged: str
    policy: str = ""
    question: str = ""
    user_request: str = ""
    intent: str = ""


@dataclass(frozen=True)
class JudgeResult:
    allowed: bool
    reason: str
    latency_s: float


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


def render_brief(brief: JudgeBrief) -> str:
    lines = [
        f"Tool: {brief.tool}",
        f"Call:\n{_clip(brief.call_text, _MAX_COMMAND_CHARS)}",
        f"Flagged because: {brief.flagged}",
    ]
    if brief.policy:
        lines.append(f"Policy: {_clip(brief.policy, _MAX_USER_CHARS)}")
    if brief.question:
        lines.append(f"Question to decide: {_clip(brief.question, _MAX_INTENT_CHARS)}")
    lines.append(
        "User's latest request: "
        + (_clip(brief.user_request, _MAX_USER_CHARS) if brief.user_request else "(none found)")
    )
    if brief.intent:
        lines.append(f"Agent's stated intent: {_clip(brief.intent, _MAX_INTENT_CHARS)}")
    return "\n\n".join(lines)


def parse_answer(text: str) -> tuple[bool, str]:
    """``(allowed, reason)``; anything off-contract is a block."""
    first = (text or "").strip().splitlines()[0].strip() if (text or "").strip() else ""
    stripped = first.strip("*`\" ")
    if stripped.upper() == "ALLOW":
        return True, ""
    if stripped.upper().startswith("BLOCK"):
        reason = stripped[5:].lstrip(":- ").strip()
        return False, reason or "the safety check refused this call"
    return False, "judge unavailable: answer was not ALLOW or BLOCK"


async def run_judge(
    brief: JudgeBrief, judge: JudgeCall | None, *, timeout: float
) -> JudgeResult:
    """Ask the judge; every failure mode is a block with a named reason."""
    started = time.monotonic()
    if judge is None:
        return JudgeResult(False, "judge unavailable: this host has no safety-check model", 0.0)
    try:
        answer = await asyncio.wait_for(judge(JUDGE_SYSTEM_PROMPT, render_brief(brief)), timeout)
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        return JudgeResult(
            False, f"judge unavailable: timeout after {timeout:g}s", time.monotonic() - started
        )
    except Exception as exc:  # noqa: BLE001 — a failed check is a block, never a crash
        logger.warning("guardrail judge failed: %s", type(exc).__name__, exc_info=True)
        return JudgeResult(
            False, f"judge unavailable: {type(exc).__name__}", time.monotonic() - started
        )
    allowed, reason = parse_answer(answer if isinstance(answer, str) else "")
    return JudgeResult(allowed, reason, time.monotonic() - started)
