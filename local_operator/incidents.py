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

#: Custom-message type journaled by the session when a session credential is
#: stored or forgotten mid-conversation. The ONLY other advertisement of a
#: stored credential is the ``<session-credentials>`` block in the volatile
#: system-prompt tail, which the model has no reason to re-read when it
#: changes — so an operator who runs ``/credential FOO_KEY`` and says "I just
#: added the key" left the model to guess names until it happened to notice
#: the tail. This message lands in the LIVE context only, naming the KEY
#: ONLY: the value must never ride a message the provider sees. It is
#: deliberately NOT persisted — credentials are process-memory-only, so a
#: replayed "$FOO_KEY is injected into every bash command" would assert an
#: env var a restarted session does not have (review round 1, R2). Resume-time
#: discovery is already served honestly by the ``<session-credentials>``
#: block, which the prompt tail rebuilds from the (empty) live store each
#: turn.
SESSION_CREDENTIAL_MESSAGE_TYPE = "session_credential"

#: Custom-message type journaled by the session when the running model changes
#: (a deliberate ``set_model``, or a failover fallback to another model).
#: Rendered to a user message the same way as an incident, so the model NOTICES
#: it is now answering as a different model rather than only seeing a changed
#: static "Model:" line in the system prompt. Persisted, so a resumed session
#: replays the switch history too.
SESSION_MODEL_SWITCH_MESSAGE_TYPE = "session_model_switch"

#: Provider wordings that mean "this request does not fit", in every phrasing
#: the vendors actually use (anthropic's "prompt is too long", google's token
#: counts, the openai family's "maximum context length").
#:
#: Exported rather than inlined into :data:`_RULES` because the provider layer
#: needs the same judgement: ``clients._relayed_upstream_failure`` has to know
#: that a RELAYED overflow complaint is deterministic — the request is too big
#: and will stay too big — so it must not be retried as upstream weather. Two
#: independently-maintained lists in one repo would silently drift into
#: disagreeing about what an overflow looks like, and the failure mode of that
#: drift (a turn retried for ~35s against a defect no wait can fix) is exactly
#: what this list exists to prevent.
CONTEXT_LENGTH_MARKERS: tuple[str, ...] = (
    "context length",
    "context window",
    "maximum context",
    "too long for the model",
    "prompt is too long",
    "request too large",
    "request was too large",
    "input too large",
    "input was too large",
    # The SIZE-of-body wordings, which none of the above matched.
    # "Request exceeds the maximum size" is Anthropic's literal 413
    # text, and a session that hit it was classified ``unknown`` with
    # an empty hint — so the model was told nothing actionable and
    # retried the identical 34 MB request, forever. The rest cover the
    # proxy edge and the provider's own error code.
    #
    # Matched on wording, NOT on a bare "413": that substring occurs
    # in ordinary token and byte counts ("used 413000 tokens" already
    # classifies correctly as rate-limit) and would misfire.
    "exceeds the maximum size",
    "request_too_large",
    "request entity too large",
    "payload too large",
)

#: Ordered (category, patterns) rules. First category whose pattern matches
#: (case-insensitive) wins; order is specificity, not severity.
_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("context-length", CONTEXT_LENGTH_MARKERS),
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
    # Widened for the byte case: the harness now sheds the oldest screenshots
    # from the RENDERED history on its own when a request is too large, so
    # "ask the user to /compact" is only half the advice — the next turn is
    # often already sendable, and the model needs to know retrying is
    # reasonable rather than assuming the session is finished.
    "context-length": "The request was too large for the model: the harness compacts "
    "and drops the oldest screenshots automatically, so retry once; if it repeats, "
    "ask the user to /compact or send fewer and smaller images.",
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


def format_credential_message(
    key: str,
    *,
    action: str = "stored",
    replaced: bool = False,
) -> str:
    """Render the credential-change text injected into the model's context.

    ``key`` is the NORMALIZED credential name (the env-var name bash injects).
    The text states where the value lives and what may be done with it, because
    the two failure modes this exists to prevent are a model that does not know
    the credential exists (and guesses wrong names) and a model that tries to
    READ it back (``read_variable``, ``echo``) and burns turns on a refusal.

    The value is deliberately absent — this text is journaled to the transcript
    and sent to the provider, so it carries the key and nothing else.
    """
    if action == "forgot":
        return (
            f"[session credential] {key} was removed. It is no longer "
            "available as an environment variable to bash commands in this "
            "session; do not reference it."
        )
    verb = "replaced" if replaced else "stored"
    return (
        f"[session credential] {key} was just {verb} by the operator. Its "
        "value is held in session memory and injected as the environment "
        f"variable ${key} into every bash command — use it there (a child "
        "process reads it), never echo, print, or write it. It is not "
        "readable through read_variable."
    )


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
