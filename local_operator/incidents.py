"""Session incidents: why a run or capability failed, made model-visible.

The failover cascade answers a provider error by rotating credentials and
models, and its notices reach the UI — but none of it reached the MODEL. A
run that died on a quota error ended with an ``agent_end`` the transcript
persisted as a bare error string, so the next prompt (or a resumed session)
resumed blind: the model had no idea the last turn was killed by rate
limiting rather than its own bug, and "continue" meant re-guessing.

This module classifies error text into the categories an agent can act on
and formats one incident record. The session journals it as a
``session_incident`` custom message — appended to the LIVE context (so the
very next prompt sees it) and persisted to the transcript (so a resumed
session replays it). Classification is deliberately conservative: plain
substring rules over the error text, ordered most-specific first, because
the texts come from every provider's error envelope and no taxonomy covers
them all. Unknown is a valid answer — the raw text always rides along.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Custom-message type journaled by the session; rendered to a user message
#: by ``Session._default_convert_to_llm`` so both a live next-turn and a
#: resumed replay see the same incident.
SESSION_INCIDENT_MESSAGE_TYPE = "session_incident"

#: Custom-message type journaled by the session when the running model changes
#: (a deliberate ``set_model``, or a failover fallback to another model).
#: Rendered to a user message the same way as an incident, so the model NOTICES
#: it is now answering as a different model rather than only seeing a changed
#: static "Model:" line in the system prompt. Persisted, so a resumed session
#: replays the switch history too.
SESSION_MODEL_SWITCH_MESSAGE_TYPE = "session_model_switch"

#: Ordered (category, patterns) rules. First category whose pattern matches
#: (case-insensitive) wins; order is specificity, not severity.
_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "context-length",
        (
            "context length",
            "context window",
            "maximum context",
            "too long for the model",
            "prompt is too long",
            "request too large",
            "request was too large",
            "input too large",
            "input was too large",
        ),
    ),
    (
        "rate-limit",
        (
            "rate limit",
            "rate_limit",
            "429",
            "too many requests",
            "quota",
            "usage limit",
            "usage_limit",
            "capacity",
            "overloaded",
        ),
    ),
    (
        "auth",
        (
            "401",
            "403",
            "unauthorized",
            "forbidden",
            "invalid api key",
            "invalid_api_key",
            "authentication",
            "permission denied",
            "expired token",
            "refresh token",
        ),
    ),
    (
        "billing",
        ("402", "payment required", "billing", "credit", "insufficient funds"),
    ),
    (
        "provider",
        (
            "500",
            "502",
            "503",
            "504",
            "internal server error",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
            "server error",
            "upstream",
            "provider error",
        ),
    ),
    (
        "network",
        (
            "timeout",
            "timed out",
            "connection",
            "econnreset",
            "econnrefused",
            "enotfound",
            "network",
            "dns",
            "ssl",
            "certificate",
            "stream disconnected",
            "unexpected eof",
        ),
    ),
    ("mcp", ("mcp", "model context protocol", "tool bridge", "circuit breaker")),
    ("content-filter", ("content policy", "content filter", "safety system", "flagged")),
]

#: Hints the model can act on without the user, per category. Empty string
#: when the honest answer is "report and ask".
_HINTS: dict[str, str] = {
    "context-limit": "",
    "context-length": "The context is over the model's window: ask the user to "
    "/compact, or compact at the next boundary if compaction is on.",
    "rate-limit": "Back off and retry later; if it persists, tell the user which "
    "provider hit the limit — they may need to switch model or top up quota.",
    "auth": "Credentials were rejected: tell the user which provider and suggest "
    "`local-operator login <provider>`. Do not retry the identical request.",
    "billing": "The provider account cannot pay for this request: report it and "
    "wait for the user.",
    "provider": "The provider is failing server-side: a retry may work; if it "
    "repeats, suggest switching model or provider.",
    "network": "The connection failed mid-stream: retrying is usually right; if "
    "it repeats, check connectivity.",
    "mcp": "An MCP server is unavailable: its tools are gone until it reconnects. "
    "Do not call its tools in a tight loop; say which server is down.",
    "content-filter": "The provider refused the content: change the approach "
    "rather than resending the same request.",
}


@dataclass(frozen=True)
class Incident:
    """One classified failure. ``raw`` always carries the original text."""

    category: str
    raw: str
    provider: str = ""
    model: str = ""

    @property
    def hint(self) -> str:
        return _HINTS.get(self.category, "")

    def render(self) -> str:
        source = f" ({self.provider}/{self.model})" if self.provider or self.model else ""
        head = f"[session incident{source}] {self.category}:"
        lines = [f"{head} {self.raw.strip()[:500]}"]
        if self.hint:
            lines.append(f"suggested action: {self.hint}")
        lines.append(
            "This is why the previous turn ended. Take it into account before "
            "repeating the same request."
        )
        return "\n".join(lines)


def classify_incident(raw: str, provider: str = "", model: str = "") -> Incident:
    """Classify an error string; never raises, never returns None."""
    text = (raw or "").lower()
    for category, patterns in _RULES:
        for pattern in patterns:
            if re.search(re.escape(pattern), text):
                return Incident(category, raw, provider, model)
    return Incident("unknown", raw, provider, model)


def format_incident_message(raw: str, provider: str = "", model: str = "") -> str:
    """One-call formatter for the rendered user-visible text."""
    return classify_incident(raw, provider, model).render()


def format_model_switch_message(
    new_label: str,
    previous_label: str = "",
    *,
    reason: str = "",
    transient: bool = False,
) -> str:
    """Render the model-switch text injected into the model's context.

    ``new_label`` / ``previous_label`` are ``provider/model_id`` strings (the
    same vocabulary the status band and the system-prompt ``Model:`` line use,
    so two names for one object never read as two models). ``transient`` marks
    a per-request failover fallback that may return to the primary at the next
    boundary, as opposed to a deliberate switch that persists; ``reason``
    carries the failover cause when there is one.

    The message is phrased as present-tense state ("You are now running as X")
    rather than a command, so the model treats it as context for the turns that
    follow rather than an instruction to acknowledge.
    """
    if previous_label and previous_label != new_label:
        head = f"[model switch] You are now running as {new_label} (was {previous_label})."
    else:
        head = f"[model switch] You are now running as {new_label}."
    lines = [head]
    if reason.strip():
        lines.append(f"Reason: {reason.strip()[:200]}")
    if transient:
        lines.append(
            "This is a temporary fallback for the current request; the session "
            "may return to its primary model at a later turn. Capabilities and "
            "context window may differ from the primary."
        )
    else:
        lines.append(
            "This applies from now on. Capabilities, context window, and tone "
            "may differ from the previous model; act as the model you now are."
        )
    return "\n".join(lines)
