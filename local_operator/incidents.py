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

It also carries the formatters for the other model-visible session records
that are NOT classified failures — a credential change, a model switch, and
an MCP recovery. Each has its own custom type and its own formatter for the
same reason: running them through :func:`classify_incident` would attach a
failure category and a "this is why the previous turn ended" tail to a
message that is not about a failure at all.
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

#: Custom-message type journaled by the session when an MCP server that was
#: ANNOUNCED BROKEN to the model connects again. The failure half of that pair
#: has always been model-visible (``McpManager.on_incident`` ->
#: ``Session._on_mcp_incident`` -> a ``session_incident`` message); the recovery
#: half was not, so an operator who ran ``/mcp login <server>`` mid-session left
#: the model holding a death notice — and its ``mcp`` hint, "its tools are gone
#: ... Do not call its tools" — for a server that had been usable for the rest
#: of the session. Observed live against ``minerva-qa``.
#:
#: It is a DEDICATED type rather than another ``session_incident`` because
#: ``journal_incident`` runs :func:`classify_incident`, whose ``mcp`` rule
#: matches the substring "mcp" and would append ``_HINTS["mcp"]`` — precisely
#: the "its tools are gone" sentence — to a message saying the opposite.
#:
#: LIVE CONTEXT ONLY, deliberately not persisted: an MCP connection is
#: process-scoped (``McpManager._connections`` is instance state and
#: ``disconnect_all`` runs on dispose), so a replayed "its N tools are
#: available to you now" would assert a live capability a restarted session may
#: not have — the same class as the credential record above, and the more
#: likely case for exactly the servers this serves, whose grants expire.
SESSION_MCP_RECOVERY_MESSAGE_TYPE = "session_mcp_recovery"

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
    # Vendors that describe the same overflow by COUNTING tokens rather than by
    # naming the context. Added after an audit found the list recognised 6 of
    # 10 real vendor wordings: google/vertex ("input token count ... exceeds"),
    # mistral ("too many tokens in prompt"), and bedrock ("input is too long")
    # all fell through, which for the provider layer meant a deterministic
    # overflow was retried as though it were upstream weather.
    #
    # Qualified rather than bare, because this list is the FIRST rule in
    # `_RULES` and therefore outranks "rate-limit". A bare "token count"
    # swallows a TPM message that happens to quote one ("Limit 90000 token
    # count per min"), and a bare "too many tokens" is verbatim AWS Bedrock's
    # ThrottlingException -- both are rate limits, and miscategorising them
    # tells the user to /compact a request whose only problem is that it
    # arrived too soon. The qualifiers name the INPUT, which a throttle never
    # does.
    "input token count",
    "prompt token count",
    "too many tokens in prompt",
    "too many input tokens",
    "input is too long",
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
    # A cut-off is the harness's own verdict, not a provider's, and its advice is
    # the opposite of "retry": whatever the turn was mid-way through may have
    # half-happened, so the model must re-establish state before redoing work.
    # The sentence is deliberately general because the CAUSE varies (retire,
    # signal, owner death); the cause itself rides ``Incident.raw``.
    "cut-off": "The runtime was cut off before this turn produced a result. The transcript "
    "holds whatever was written before it stopped and nothing after. Check the "
    "state of anything it was mid-way through before repeating the work; do not "
    "assume the request completed.",
}

#: Why a turn was cut off. Harness-authored, so unlike :data:`_RULES` these are
#: exact tokens rather than substring guesses over vendor text; the rendering
#: borrows :class:`Incident`'s shape but not its classifier.
#:
#: ``runtime-retired`` and ``install-mid-update`` are OURS to explain (a
#: planned retirement that caught a live turn; a lazy import against a
#: half-replaced install). ``runtime-shutdown`` covers an ordinary termination
#: signal, ``runtime-killed`` a process that vanished without exiting cleanly,
#: and ``owner-lost`` the viewer-side verdict that the runtime it was bound to
#: disappeared. ``user-stop`` is the one DELIBERATE cause, and it is what keeps
#: a user's own cancel from being reported as an error.
CUT_OFF_CAUSES: dict[str, str] = {
    "user-stop": "the session was stopped by the user",
    "runtime-retired": "the runtime retired so the next engage would run a newer build",
    "runtime-shutdown": "the runtime was terminated while this turn was running",
    "runtime-killed": (
        "the runtime disappeared without exiting cleanly while this turn was running"
    ),
    "install-mid-update": (
        "a local-operator install was being replaced on disk while this turn was running"
    ),
    "owner-lost": "the session's runtime went away while this turn was running",
    "disposed": "the session was disposed while this turn was running",
}

#: The sentence used when a cause token is not one this build knows — a newer
#: runtime's token reaching an older viewer. Naming the gap is honest; guessing
#: a cause would not be, and a refusal to render would hide the cut-off.
CUT_OFF_UNKNOWN = "the turn was cut off and the cause could not be determined"


def render_cut_off_reason(cause: str, *, detail: str = "") -> str:
    """One operator-facing sentence naming why a turn was cut off.

    This is the string the durable outcome stores as ``reason`` and every
    surface prints after its own prefix (``Stopped with an error — …``), so it
    is deliberately a sentence and not a paragraph: ``detail`` carries an
    optional parenthetical (a build pair, a pid, a started-at) rather than
    more prose, because the sidebar tooltip has one line and truncates the
    rest rather than wrapping it.
    """
    sentence = CUT_OFF_CAUSES.get(cause) or CUT_OFF_UNKNOWN
    return f"{sentence}{detail}"


def cause_from_reason(reason: str) -> str:
    """The cause token a rendered reason came from, or ``""`` when unknown.

    The inverse of :func:`format_cut_off_notice` / :func:`render_cut_off_reason`,
    for the one place that has the SENTENCE and needs the classification: a
    locally synthesised turn end carries operator-facing prose (that is what
    every surface prints), and a machine token must be recoverable from it so a
    consumer can group or log the cause without parsing English twice.

    Longest-prefix match rather than an exact one, because the notice append
    parenthetical detail (``(0.54.11@b133eba → 0.54.12@402af7f)``). Returns
    ``""`` for prose this vocabulary did not write, so a provider's own error
    message is never misclassified as a harness cause.
    """
    for cause, sentence in CUT_OFF_CAUSES.items():
        if sentence and sentence in reason:
            return cause
    return ""


def format_cut_off_notice(cause: str, *, detail: str = "") -> str:
    """The LIVE transcript notice for a cut-off turn.

    Deliberately different from :func:`format_cut_off_message`: the live row is
    what a watching user reads the moment the turn dies, so it leads with the
    fact (``turn cut off``) and adds the one consequence they need (what the
    transcript does and does not contain). The full incident form is for the
    NEXT turn's model context, where the consequence and the advice both earn
    their space.
    """
    return (
        f"turn cut off — {render_cut_off_reason(cause, detail=detail)}. "
        "The transcript holds what it wrote before that and nothing after."
    )


def format_cut_off_raw(reason: str) -> str:
    """Render an ALREADY-COMPOSED cut-off reason into the incident text.

    Split out from :func:`format_cut_off_message` because the restore path holds
    a reason string (with its detail already appended) rather than a cause and a
    detail it can recombine: re-deriving one from the other there would re-render
    a different sentence than the one the store and the sidebar are showing.
    """
    return Incident(category="cut-off", raw=reason).render()


def format_cut_off_message(cause: str, *, detail: str = "") -> str:
    """Render a cut-off cause into the established incident text.

    Built through :class:`Incident` rather than a parallel prose system, so the
    next turn's card and the resume replay are the SAME surface as every other
    failure, and the ``cut-off`` hint above is applied by the one renderer.
    """
    return format_cut_off_raw(render_cut_off_reason(cause, detail=detail))


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


def format_mcp_recovery_message(server: str, tool_count: int) -> str:
    """Render the MCP-recovery text injected into the model's context.

    The counterpart to the ``mcp`` incident category, and the reason it is a
    dedicated formatter rather than another :func:`classify_incident` category:
    the classifier matches the substring "mcp" and would append
    ``_HINTS["mcp"]`` — "its tools are gone until it reconnects. Do not call
    its tools in a tight loop" — plus ``Incident.render``'s "This is why the
    previous turn ended", to a message announcing the exact opposite.

    Present-tense STATE, like :func:`format_model_switch_message`, so the model
    treats it as context for the turns that follow rather than an instruction
    to acknowledge.

    The SUPERSEDE sentence is load-bearing and must not be trimmed to a bare
    "reconnected": the model is holding an earlier ``session_incident`` for
    this server whose hint explicitly says "do not call its tools", and two
    live claims leave it free to defer to the older, more emphatic one. It must
    be told the earlier notice no longer applies.

    ``tool_count`` is the REGISTERED tool count
    (``len(McpManager.get_server_tools(server))``), not ``len(conn.tools)``:
    ``_register_tools`` filters by ``enabledTools``/``disabledTools``, so the
    raw list overstates what the model can actually call.

    ZERO registered tools takes a different sentence, not a count-free variant
    of the same one (review round 1, R2). It is a real state — a server can be
    connected with every tool filtered out by ``disabledTools`` — and telling
    the model its tools "are usable now, so call them normally" against an
    empty inventory is false in the one direction that costs a wasted turn:
    the model goes looking for tools that are not there. The connection is
    still worth announcing, because it is what supersedes the incident and
    stops the model reporting the server as down.
    """
    if not tool_count:
        # Honest about the CONNECTION and silent about callable tools: no
        # "available again", no "call them normally". The supersede clause is
        # still required — the model is holding an incident that says the
        # server is unreachable, which is no longer true.
        return (
            f"[mcp recovery] MCP server '{server}' is connected again, but it "
            "currently exposes no enabled tools. This supersedes the earlier "
            "session incident about this server: the server itself is no "
            "longer failing, so stop reporting it as unavailable — but do not "
            "expect callable tools from it until some are enabled."
        )
    # Verb agreement is spelled out rather than templated: the design's draft
    # formatter read "1 tool are available again", which is text the model
    # actually reads.
    tools = "1 tool is" if tool_count == 1 else f"{tool_count} tools are"
    return (
        f"[mcp recovery] MCP server '{server}' is connected again and {tools} "
        "available again. This supersedes the earlier session incident about "
        "this server: its tools are usable now, so call them normally and stop "
        "reporting it as unavailable."
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
