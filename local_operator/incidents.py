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
that are NOT classified failures — a credential change, a model switch, an MCP
recovery, and an MCP server becoming unavailable. Each has its own custom type
and its own formatter for the same reason: running them through
:func:`classify_incident` would attach a failure category and a "this is why the
previous turn ended" tail to a message that is not about a failure at all.

The MARKERS those five records carry are no longer defined here. They moved to
:mod:`local_operator.harness.message_types`, the one neutral home that a
surface barred from importing this module can reach: the shared renderer
(``harness/render.py``) has to recognise a ``session_incident`` to replay it,
but an evaluation episode may not import this module — and importing a string
is enough to pull it in. Nothing in this module reads the markers, so it does
not re-import them; the values and their per-record notes live in that module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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

#: The provider wording that means "this DeepSeek thinking-mode request never
#: carried the reasoning back": the message this pins is, verbatim, "The
#: `reasoning_content` in the thinking mode must be passed back to the API."
#:
#: BOTH halves are required, and this tuple is the ONE definition of them: the
#: loop's recovery gate (``harness/loop.py``, which imports this) and the
#: classifier below must agree about what the refusal is. They once carried
#: private copies and combined them differently -- the loop requiring both, the
#: rule only one -- so a 429 body quoting ``requested.reasoning_content``, a
#: relayed 502 naming the field, and the legacy rows' "unsupported field
#: 'reasoning_content'" 400 all rendered as this category with the "switch
#: model" hint. That last one is the error
#: :data:`model.configure._DEEPSEEK_THINKING_MODELS` exists to AVOID, so
#: reporting it as our own recovery having failed inverted its meaning.
REASONING_ECHO_MARKERS: tuple[str, ...] = ("reasoning_content", "must be passed back")

#: One marker in a rule below: a plain string matches on its own, and a TUPLE is
#: a CONJUNCTION whose every element must be present. Any-of is the right default
#: for vendor wording; the conjunction form exists for the one refusal whose two
#: halves are each individually common (see :data:`REASONING_ECHO_MARKERS`).
Marker = str | tuple[str, ...]

#: Ordered (category, markers) rules. First category with a matching marker wins
#: (case-insensitive); order is specificity, not severity.
_RULES: list[tuple[str, tuple[Marker, ...]]] = [
    ("context-length", CONTEXT_LENGTH_MARKERS),
    # The DeepSeek thinking-mode validator's own wording, named before the
    # generic rules because the refusal arrives RELAYED through an aggregator
    # ("upstream ...") as often as directly, and a relayed body must not be read
    # as a provider fault. The harness answers it with a retry that turns
    # thinking off (``harness/loop.py``), so a user seeing this category means
    # that recovery did not apply or was refused too -- their model cannot
    # continue this conversation with thinking on, which only they can resolve.
    #
    # The markers are a CONJUNCTION, not a choice: each half alone is ordinary,
    # and order alone cannot separate them. A 429 quoting the field name now
    # falls through to rate-limit and a relayed 502 to provider, which is what
    # they are; and the legacy rows' "unsupported field 'reasoning_content'" 400
    # matches no other rule at all, so under the ORed form it landed HERE --
    # reporting the error the capability exists to AVOID as our own recovery
    # having failed. Sharing :data:`REASONING_ECHO_MARKERS` with the loop's gate
    # is what keeps the two answers from drifting apart again.
    (
        "reasoning-echo",
        (REASONING_ECHO_MARKERS,),
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
            # "provider error" used to sit here and is DELIBERATELY GONE. It was
            # never a provider's wording: it is the KIND LABEL this harness puts
            # in front of every wrapped transport failure
            # (``ProviderError.__str__`` -> "transient provider error: ..."), an
            # always-present prefix that classified the harness's own sentence
            # instead of the failure's evidence, from a rules position ordered
            # AHEAD of `network` — which is how a pre-connect failure on
            # 2026-09-15 came to be reported to the operator as the provider
            # failing server-side.
            #
            # Removing it on its own left a real hole, caught in agent review
            # round 1 (R1-1): the label was the ONLY matcher for 5xx statuses
            # outside the four enumerated above — 501, 505-510, Cloudflare's
            # 52x, nginx's 597 — because those four DIGITS were doing work they
            # could not carry, and for a status-less transport class whose name
            # carries no connection-ish wording ("transient provider error:
            # ReadError"). All of those fell through to `unknown`, whose
            # `Incident.render` omits the `suggested action:` line ENTIRELY —
            # strictly less than the model was told before this PR, which is the
            # opposite of the change's purpose. Two token families close the
            # hole, and neither is the label:
            #   - "http 5", a STATUS-SHAPED token. The harness's own rendering
            #     makes it unambiguous — ``ProviderError.__str__`` writes
            #     "(HTTP <status>)" — so the enumerated digits above are now
            #     examples rather than the coverage;
            #   - the transport CLASS NAMES on the `network` rule below, which is
            #     where a status-less socket failure belongs.
            "http 5",
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
            # The PRE-CONNECT wordings, listed explicitly rather than left to
            # the generic "connection" above. That substring happens to appear
            # in anyio's aggregate ("All connection attempts failed"), so the
            # aggregate currently lands here by accident; naming the shapes
            # pins it, and covers the ones no other token reaches — a bare
            # "ConnectError" class name and the EADDRNOTAVAIL wording the
            # incident itself carried.
            "connecterror",
            "connection attempts failed",
            "can't assign requested address",
            "cannot assign requested address",
            "eaddrnotavail",
            # The httpx/httpcore transport CLASS NAMES, which
            # ``wrap_transport_error`` puts verbatim into the message
            # ("<ClassName>: <detail>") and which are routinely raised with an
            # EMPTY detail (``httpx.ReadError('')`` is what a TCP RST mid-body
            # surfaces as). With no message and no status there is no other
            # token to match, and agent review R1-1 measured exactly that: a
            # status-less transport failure used to be caught by the harness's
            # own "provider error" label and, once the label was removed,
            # classified `unknown` with no hint at all.
            "readerror",
            "writeerror",
            "closeerror",
            "protocolerror",  # covers RemoteProtocolError and LocalProtocolError
        ),
    ),
    # NO ``mcp`` RULE, deliberately, and its absence is the fix rather than an
    # omission. An MCP server going unavailable is not a FAILURE of the turn —
    # it is a missing capability — and the rule that used to sit here matched the
    # bare substring "mcp", so it caught every MCP-mentioning failure and gave
    # it ``_HINTS["mcp"]`` plus ``Incident.render``'s "This is why the previous
    # turn ended." That last line was false: an MCP server failing to connect
    # (or its OAuth grant expiring) never ends a turn. Measured live on
    # 2026-09-20 against ``minerva-qa``, where the operator's session was told a
    # turn had died that had not.
    #
    # Deleted rather than narrowed. A future rule for whatever MCP text really
    # does describe a dead turn is welcome; what must not come back is a rule
    # keyed on the substring shared by BOTH halves of the MCP story, which is
    # also why removing the hint matters as much as removing the rule — a rule
    # without a hint still stamps a failure category, and a hint without a rule
    # is unreachable text the next reader has to check twice.
    #
    # The MCP-unavailable text lives in
    # :func:`format_mcp_unavailable_message` instead, on the same dedicated-
    # formatter argument as :func:`format_mcp_recovery_message`.
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
    # Two honest readings of the same category, because the hint is a static
    # string and WHICH recovery ran is per-model: on a model with no thinking-off
    # rung -- which includes the live aggregator routes to these weights -- the
    # loop never re-asks with thinking disabled, so a hint claiming that retry
    # "did not clear" it would describe a call that was never made. Both
    # recoveries are therefore named with their own precondition stated rather
    # than asserted: the echo fill is spent only when the spec was not already
    # carrying an echo, and the thinking-off retreat only when the model has the
    # rung.
    "reasoning-echo": "The provider refused the request because the conversation's "
    "reasoning was not carried back, and the harness could not clear it by "
    "re-sending with that echo filled (spent wherever the request was not "
    "already carrying one) or by retrying with thinking disabled (or this model "
    "has no such rung to retry at). Do not resend the same request unchanged: "
    "tell the user the model's thinking mode cannot continue this conversation "
    "and suggest switching model.",
    "auth": "Credentials were rejected: tell the user which provider and suggest "
    "`local-operator login <provider>`. Do not retry the identical request.",
    "billing": "The provider account cannot pay for this request: report it and "
    "wait for the user.",
    "provider": "The provider is failing server-side: a retry may work; if it "
    "repeats, suggest switching model or provider.",
    # Rewritten twice. The first version named the machine, which was right for
    # the pre-connect case this PR is about but WRONG for the rest of the
    # category: a refused or reset connection is a TCP RST from the far end,
    # which proves this machine's network works, and the failover layer
    # deliberately treats it as an ordinary transient so a dead local provider
    # keeps its rotation and fallback walk (agent review R1-2). The category has
    # to speak for both halves, so the hint does.
    "network": "A connection failure. If the far end refused or reset it, it did "
    "answer — retrying or switching target is reasonable. If this machine could "
    "not reach the network at all, retrying is usually right, and a repeat means "
    "checking this machine's connectivity rather than switching provider.",
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
#: half-replaced install — the shape the PRE-generation layout produced, still
#: reachable for a pip/pipx tree and for a process still running out of the old
#: fixed uv-tool tree, and unreachable by construction once a process belongs to
#: a generation, whose files are written once and never rewritten).
#: ``runtime-shutdown`` covers an ordinary termination signal,
#: ``runtime-killed`` a process that vanished without exiting cleanly,
#: and ``owner-lost`` the viewer-side verdict that the runtime it was bound to
#: stopped answering. ``user-stop`` is the one DELIBERATE cause, and it is what
#: keeps a user's own cancel from being reported as an error.
#: The ONE deliberate cause in :data:`CUT_OFF_CAUSES`. Exported so the readers
#: that have to tell a recorded stop apart from a cut-off do not re-spell the
#: token: the restore seam's journaling guard and the durable-outcome writer are
#: both answering that question, and a hard-coded copy on either side is one
#: rename away from silently admitting a user's own ``/stop`` into the cut-off
#: vocabulary.
DELIBERATE_CUT_OFF_CAUSE = "user-stop"

#: The DELIBERATE causes, as a SET rather than a token to compare against.
#:
#: WHY A SET AND NOT ``cause != DELIBERATE_CUT_OFF_CAUSE``: every reader that
#: has to tell a recorded stop apart from a cut-off asks the same question, and
#: an identity comparison against ONE token answers it only for the deliberate
#: causes that existed when the comparison was written. A future deliberate
#: token (a user-cancelled ``/fork``, a scripted ``lop stop --all``) added to
#: :data:`CUT_OFF_CAUSES` with the old comparison in place would be narrated to
#: the model as an involuntary cut-off — the exact misclassification the guard
#: exists to prevent — from the moment it was coined. Here it is admitted by
#: ADDING it to this set, so the two halves of the taxonomy cannot drift
#: (review round 1, NIT-1).
DELIBERATE_CUT_OFF_CAUSES: frozenset[str] = frozenset({DELIBERATE_CUT_OFF_CAUSE})


#: The sentence for :data:`CUT_OFF_CAUSES`' ``runtime-overdue`` rung, and the one
#: place the bound's number is spelled for a reader — RENDERED from the constant
#: that enforces it, never typed here: this sentence is repeated by every surface
#: that repeats a cut-off (the live notice, the durable outcome, the sidebar), and a
#: second copy of "15 min" is a copy that drifts.
#:
#: The import is FUNCTION-LOCAL, not module scope: this table is a leaf every
#: runtime module may import, and the runtime's own vocabulary module is the wrong
#: thing for a leaf to depend on at import time (``incidents`` is imported by
#: ``session/runtime/journal.py``, which the child loads before it has a session).
#: A dict entry may call a function; a dict entry may not defer an import.
#: COORDINATE NOTE: PR #1297 also touches this module (its install-marker work);
#: this is one new key and one new helper, no existing entry reworded.
def _overdue_cause_sentence() -> str:
    from local_operator.session.runtime.types import BUILD_DRAIN_PROGRESS_S, bound_text

    return (
        "the runtime left for the newer build after "
        f"{bound_text(BUILD_DRAIN_PROGRESS_S)} of no movement reported from the work in flight"
    )


CUT_OFF_CAUSES: dict[str, str] = {
    DELIBERATE_CUT_OFF_CAUSE: "the session was stopped by the user",
    "runtime-retired": "the runtime retired so the next engage would run a newer build",
    "runtime-shutdown": "the runtime was terminated while this turn was running",
    # The BOUNDED handover: a build drain that stopped waiting for its own work
    # (``process._leave_overdue``). It is deliberately its own token rather than
    # ``runtime-retired``, which is what it used to be recorded as: the retirement
    # sentence is true of it, but it is ALSO the sentence every ordinary build
    # handover leaves, so a turn that had to be FORCED out was indistinguishable
    # from one that waited its turn out in the only durable account of it — and by
    # the time an operator looks, that account is all there is (``lop sessions``
    # loses the record ~97 ms after the escalation). QA round 1 (Q-2) measured
    # exactly that: a successor narrated the generic retirement sentence, never the
    # bound.
    "runtime-overdue": _overdue_cause_sentence(),
    "runtime-killed": (
        # The trailing clause is the POST-MARKER meaning of this token, and it
        # is decidable now in a way it was not before the durable marker
        # existed: "disappeared without exiting cleanly" describes what the
        # PROCESS did, and a reader one row under a deliberate stop needs the
        # other half — that nothing recorded anyone asking for it. Without the
        # clause the two rows differ only in the word above them, and "asked
        # for" versus "never asked" is exactly the distinction this taxonomy
        # was extended to draw (design round 1, D4).
        "the runtime disappeared without exiting cleanly while this turn was running, "
        "and nothing recorded a stop"
    ),
    "install-mid-update": (
        "a local-operator install was being replaced on disk while this turn was running"
    ),
    # Deliberately about what the VIEWER can verify, not about what happened to
    # the process. The arm that paints this is the recovery loop's give-up,
    # which fires both for an owner whose record is gone and for a live-but-
    # silent one (the record is there, its pid is alive, and nothing answers) —
    # "went away" asserted a death that arm cannot establish (review round 1,
    # MINOR-3). "stopped answering" is the fact both shapes share.
    "owner-lost": "the session's runtime stopped answering while this turn was running",
    "disposed": "the session was disposed while this turn was running",
}

#: The sentence used when a cause token is not one this build knows — a newer
#: runtime's token reaching an older viewer. Naming the gap is honest; guessing
#: a cause would not be, and a refusal to render would hide the cut-off.
CUT_OFF_UNKNOWN = "the turn was cut off and the cause could not be determined"


def is_cut_off_cause(cause: str) -> bool:
    """True iff ``cause`` names an INVOLUNTARY cut-off this build can render.

    Membership AND kind, not identity: the vocabulary decides that the token is
    one this build understands, and :data:`DELIBERATE_CUT_OFF_CAUSES` decides
    that understanding a token is not enough to call the turn a cut-off.
    """
    return cause in CUT_OFF_CAUSES and not is_deliberate_cause(cause)


def is_deliberate_cause(cause: str) -> bool:
    """True iff ``cause`` is a recorded DELIBERATE act rather than a cut-off.

    The single place the deliberate half of the taxonomy is read, so a surface
    that needs "was this the user's own act?" (the phone's frame fill, the
    journal guard) cannot answer it with a comparison that goes stale.
    """
    return cause in DELIBERATE_CUT_OFF_CAUSES


def _render_cut_off_detail(detail: str) -> str:
    """The detail a cut-off sentence carries, punctuated into place.

    NORMALISED rather than concatenated, because the callers do not agree on
    the separator. Two of them (``attention._record_detail``,
    ``update._install_mid_update_reason``) hand over a string that already
    opens with a space and a bracket, while ``process._drain_detail`` composes
    a phrase and its pair (``the runtime declined to hand over 3x (0.56.2 →
    0.56.6)``) with neither. Appending the raw detail made every retirement
    read ``…would run a newer builddeclined 3x (0.56.2 → 0.56.6)`` in the live
    row, in the durable reason AND in the next turn's incident card (measured
    on this host, 2026-09-17). The separator belongs to the one function that
    knows the sentence precedes it, so a fourth caller cannot reintroduce the
    run-together text.

    ONE PARENTHETICAL, NEVER NESTED — the rule ``journal.row_detail`` states
    and every other producer already honours, and the reason a bare detail does
    NOT get wrapped in a second pair here. Wrapping the drain's phrase gave
    ``…would run a newer build (the runtime declined to hand over 3x (0.56.2 →
    0.56.6))``: two bracket levels, ``)).`` at the end of the clause, and a
    reader matching two levels across a line break to read one aside (design
    round 1, D1; QA round 1, Q2). The em dash is this vocabulary's own way of
    appending a clause to a sentence — ``format_cut_off_notice`` and
    ``catalog._stop_label`` both use it — so a bare detail is joined with it:
    ``…would run a newer build — the runtime declined to hand over 3x (0.56.2 →
    0.56.6)``. A detail that is ALREADY a parenthetical keeps its single pair,
    which is what leaves those callers untouched.
    """
    text = (detail or "").strip()
    if not text:
        return ""
    if text.startswith("(") and text.endswith(")"):
        return f" {text}"
    return f" — {text}"


def render_cut_off_reason(cause: str, *, detail: str = "") -> str:
    """One operator-facing sentence naming why a turn was cut off.

    This is the string the durable outcome stores as ``reason`` and every
    surface prints after its own prefix (``Stopped with an error — …``), so it
    is deliberately a sentence and not a paragraph: ``detail`` carries the
    why-now clause (a build pair, a pid, a started-at) rather than more prose,
    because the sidebar tooltip has one line and truncates the rest rather than
    wrapping it. The caller supplies the text of that clause and nothing else —
    its punctuation is added here (see :func:`_render_cut_off_detail`).
    """
    sentence = CUT_OFF_CAUSES.get(cause) or CUT_OFF_UNKNOWN
    return f"{sentence}{_render_cut_off_detail(detail)}"


#: How one rung of the stop ladder reads inside a deliberate stop's detail.
#:
#: IN THE RECEIPT'S NOUNS, not the kernel's, and in ONE register. The operator
#: reads the stop receipt — ``killed "X"`` / ``stopped "X" (sigterm)`` /
#: ``stopped "X"`` (``control._stopped_line``) — seconds before this sentence
#: lands in the list beside it, so a durable row saying ``SIGKILL`` while the
#: receipt that caused it said ``killed`` splits one act into two vocabularies
#: on two surfaces a reader compares directly. This repo has already ruled on
#: that class twice (``catalog.py``'s sidebar-word D5, the phone's button-word
#: D7: the receipt's word is the established one, so the durable sentence
#: follows it).
#:
#: The signal name that survives is ``sigterm``'s, and it survives as the
#: receipt's own parenthetical — ``stopped with a signal`` is the phrase, and
#: the rung token is one keystroke away in the artifact. What the operator
#: needs from the phrase is how hard the stop had to push: the same word
#: "stopped" covers a runtime that exited on request and one carrying orphaned
#: state because nothing else was left.
STOP_RUNG_LABELS: dict[str, str] = {
    "socket": "stopped on request",
    "sigterm": "stopped with a signal",
    "sigkill": "killed",
}

#: What an rung this build does NOT know renders as.
#:
#: The token itself must never reach the operator: before this, the fallback
#: was ``STOP_RUNG_LABELS.get(rung, rung)``, so the day a fourth rung is coined
#: the sidebar reads ``(future-rung by /stop, pid 1234)`` — the writer's
#: spelling leaking onto a surface that has no way to explain it. What IS true
#: of any deliberate stop of an unknown rung is that the session was stopped on
#: request, so that is what an unknown rung says.
STOP_RUNG_UNKNOWN = "stopped"

#: The rungs that mean the ladder had to push PAST the plain request.
#:
#: Load-bearing for copy, not for control flow: the surfaces that already show
#: the deliberate WORD (the sidebar's ``Interrupted``, the returned-to-turn
#: notice) have already said "the user stopped it", so spending the attribution
#: on rung 1 adds nothing they do not know — it is exactly what ``/stop`` does.
#: An escalation is the fact worth the cells: the runtime did not exit on
#: request and had to be signalled. This is also the whole of design round 1's
#: D1 complaint (a rung-3 kill and a rung-1 request were byte-identical).
ESCALATED_STOP_RUNGS: frozenset[str] = frozenset({"sigterm", "sigkill"})

#: How the killer's pid is LABELLED in the durable sentence.
#:
#: A bare ``pid 40609`` is the SAME TOKEN with the OPPOSITE referent one row up:
#: an error row's parenthetical (``attention._record_detail``) names the runtime
#: that DIED, and this one names the process that killed it. A reader comparing
#: the two rows concludes the same relationship about both. Labelling is one
#: word and removes the ambiguity without spending the pid, which is the
#: forensic fact 2026-09-13 lacked (design round 1, D2).
KILLER_PID_LABEL = "killer pid"


def render_stop_attribution(*, rung: str = "", command: str = "", killer_pid: object = None) -> str:
    """The parenthetical a DELIBERATE stop's reason carries.

    SCALARS RATHER THAN THE MARKER DICT, deliberately: the marker's schema
    belongs to its one writer (``control._stop_marker_payload``) and this module
    only renders text, so a field renamed on one side cannot silently empty the
    sentence on the other. The rung answers "how hard" (in
    :data:`STOP_RUNG_LABELS`'s operator nouns, never as a raw signal name), the
    command and pid answer "who" — and "who" is not trivia here: the
    2026-09-13 investigation could reproduce every consequence of the kill wave
    and still not name a single process that sent a signal, because nothing
    recorded it.

    Returns ``""`` rather than a bare ``" ()"`` when there is nothing to name,
    so an older or partial marker degrades to exactly the shared sentence.
    """
    label = (STOP_RUNG_LABELS.get(rung) or STOP_RUNG_UNKNOWN) if rung else ""
    who = command or ""
    if killer_pid is not None and str(killer_pid).strip():
        who = f"{who}, {KILLER_PID_LABEL} {killer_pid}".lstrip(", ")
    if label and who:
        return f" ({label} by {who})"
    if label:
        return f" ({label})"
    if who:
        return f" (by {who})"
    return ""


def _stop_detail_from_reason(reason: str) -> str:
    """The parenthetical :func:`render_stop_attribution` wrote, or ``""``.

    The INVERSE of the renderer, and the same shape ``cause_from_reason``
    already uses to read a sentence back: the surfaces that print a completion
    reason are handed the STORED string and nothing else (``AttentionStore``
    keeps kind/cause/reason, and ``rows.completion_notice`` / the banner body
    take the reason as an argument), so a surface that wants the rung has to
    recover it from the sentence this module wrote. It is exact rather than
    clever: the shared deliberate sentence, one trailing parenthetical, nothing
    else. Anything else — an involuntary cause, a pre-attribution marker, a
    provider's prose — returns ``""`` and the surface prints what it printed
    before.
    """
    sentence = CUT_OFF_CAUSES.get(DELIBERATE_CUT_OFF_CAUSE, "")
    if not sentence or not reason.startswith(sentence):
        return ""
    rest = reason[len(sentence) :].strip()
    if not (rest.startswith("(") and rest.endswith(")")):
        return ""
    return rest[1:-1]


def stop_rung_phrase(reason: str) -> str:
    """The attribution a ROW-sized surface can print, or ``""``.

    For the surfaces whose row already carries the deliberate word (the
    sidebar's ``Interrupted``, the returned-to-turn notice, the background
    banner): the phrase is spent only when the ladder ESCALATED
    (:data:`ESCALATED_STOP_RUNGS`), because that is the fact the word does not
    already carry. So a rung-3 stop reads ``Interrupted — killed by /stop --all``
    and a rung-1 request stays exactly ``Interrupted``.

    The killer's pid is deliberately NOT in the phrase. A tooltip and a banner
    body are read at a glance on a surface that also lists the pid of the
    runtime that DIED (design round 1, D2), and the pid is one read away in
    ``lop sessions --json``'s reason and in the artifact itself — where the
    question "which process sent the signal" is actually being asked.
    """
    detail = _stop_detail_from_reason(reason)
    if not detail:
        return ""
    for rung, label in STOP_RUNG_LABELS.items():
        if rung not in ESCALATED_STOP_RUNGS or not detail.startswith(f"{label} "):
            continue
        # Drop the labelled pid clause, keeping the rung and the actor: the
        # separator is the one :func:`render_stop_attribution` writes.
        return detail.split(f", {KILLER_PID_LABEL} ", 1)[0]
    return ""


def outcome_summary(reason: str) -> str:
    """One line explaining how a session's last turn ended, for a LIST column.

    Distinct from :func:`stop_rung_phrase` in exactly one way: a list column
    has no deliberate word of its own (``lop sessions`` prints state, not
    ``Interrupted``), so a stop on the plain request rung must still say what
    happened rather than render an empty cell. Order of preference is the
    escalation phrase when there is one — the rung and the actor are the facts
    a reader cannot get anywhere else on that surface — and the sentence's own
    first clause otherwise (dropping the parenthetical, which for an
    involuntary cause is build/pid/started-at detail the JSON row carries in
    full).
    """
    phrase = stop_rung_phrase(reason)
    if phrase:
        return phrase
    sentence = reason.split(" (", 1)[0].strip()
    return sentence


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


def _matches(text: str, marker: Marker) -> bool:
    """Does ``text`` carry this marker? A tuple is a conjunction of all of it."""
    if isinstance(marker, tuple):
        return all(re.search(re.escape(part), text) for part in marker)
    return bool(re.search(re.escape(marker), text))


def classify_incident(raw: str, provider: str = "", model: str = "") -> Incident:
    """Classify an error string; never raises, never returns None."""
    text = (raw or "").lower()
    for category, patterns in _RULES:
        if any(_matches(text, marker) for marker in patterns):
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


def format_shape_incident_message(tool: str, labels: "list[str]", summary: str = "") -> str:
    """Render the credential-SHAPE notice injected into the model's context.

    The counterpart to the shape pass in :mod:`local_operator.redaction_shapes`,
    and the reason it is its own formatter rather than a
    :func:`classify_incident` category: nothing FAILED. A tool returned a
    credential in a shape the table recognises, the harness masked it before the
    model could read it, and the only jobs this text has are to say so (so the
    model does not reason about a value it cannot see, or re-run the command
    hoping for a different result) and to make the event visible to the operator.

    ``labels`` are shape NAMES, never values — a notice that carried the
    credential would be the leak it exists to report. The summary is the one the
    harness built, already scrubbed and bounded.

    The last sentence is the point of the whole path: the operator has to rotate
    the credential. Today a miss is discovered by accident, from a transcript,
    weeks later; this row is what turns it into a ticket.
    """
    shapes = ", ".join(labels) if labels else "credential-shaped content"
    tool_name = tool or "a tool"
    where = f" The call was: {summary}." if summary else ""
    # "was about to reach you ... and was masked" rather than "the result
    # carried": the same notice serves a credential in a tool's OUTPUT and one
    # TYPED INTO a call's arguments, and only the first of those is a result.
    # A notice that misnamed the surface would send an operator looking in the
    # wrong place.
    # The FIRST row carries the action. The row is six lines in the card and the
    # head is all most readers take: it used to open with the mechanism (``a
    # credential in a shape the harness recognises (dsn-password)``), which put
    # "rotate it" in the fourth line. The bracketed head stays — the harness's
    # notice-row rules key on it, and a row that paints as the user's own words
    # would be worse than a jargon-first one.
    return (
        f"[credential redaction] rotate it — a credential ({shapes}) reached "
        f"{tool_name} and was masked before you saw it.{where} Treat it as "
        "compromised: the operator has to rotate it. Do not re-run the command to "
        "read the value; it is contained for the rest of this session."
    )


def format_mcp_unavailable_message(server: str, reason: str) -> str:
    """Render the MCP-unavailability text injected into the model's context.

    The counterpart to :func:`format_mcp_recovery_message`, and a dedicated
    formatter for the same reason: neither row is about a FAILED turn, so
    neither may go through :func:`classify_incident`. The classifier used to
    catch this text on its ``mcp`` rule — the bare substring "mcp" — and hand
    it ``_HINTS["mcp"]`` plus ``Incident.render``'s "This is why the previous
    turn ended", the last of which was simply false: an MCP server going away
    never ends a turn. Measured live on 2026-09-20 against ``minerva-qa``, whose
    expired grant told the operator a turn had died that had not.

    Present-tense STATE, like :func:`format_model_switch_message`, so the model
    treats it as context for the turns that follow rather than an instruction
    to acknowledge. The three lines carry three different jobs:

    * the head states the CAPABILITY change — the server is unavailable and its
      tools are gone — because that is what stops the model calling them;
    * ``Reason:`` carries the operator's own remedy (``/mcp reauth <server>``,
      a breaker that suspended auto-reconnect), which is what lets the model
      tell the user what to do rather than only that something is wrong;
    * the last line states the manual-recovery fact and keeps the model's
      instruction not to hammer the tools.

    **"for now", never "until it reconnects".** The first draft promised a
    self-heal, and there is none for either family this row is written for: an
    expired grant never heals by retrying (auto-reconnect is non-interactive by
    design, so only ``/mcp reauth`` restores it) and a tripped breaker has
    auto-reconnect SUSPENDED at the moment this row is written. "Until it
    reconnects" also implied an action nobody had taken. Design review round 1
    (D2): the promise has to be true for both families, which is why the head is
    reason-agnostic and the last line says the user has to restore it.

    The last line is read by TWO audiences — it rides the model's context and
    the transcript row the operator sees — so it is phrased as a fact about the
    agent rather than as an imperative addressed to whoever is reading (design
    review round 1, D5). An imperative with no label in front of it reads, to
    the human, as an instruction to them; the incident shape carried the same
    instruction, but under a ``suggested action:`` label that marked whose it
    was.

    No failure category and no ``suggested action:`` line, both of which belong
    to the incident shape this record is deliberately not. The reason is bounded
    at 200 characters exactly as :func:`format_model_switch_message` bounds its
    own, and is OMITTED when blank rather than printed empty: a dangling
    ``Reason:`` reads as a truncation. It is shaped COMMAND-FIRST by its callers
    (``mcp/manager.py``), so the one part a reader must not lose lands at the
    front of the line rather than mid-line after a restatement (design review
    round 1, D3).
    """
    lines = [f"[session warning] MCP server '{server}' is unavailable: its tools are gone for now."]
    if reason.strip():
        lines.append(f"Reason: {reason.strip()[:200]}")
    lines.append(
        "Its tools are not callable until the user restores it, and the agent "
        "should not retry them in a loop."
    )
    return "\n".join(lines)


def format_mcp_recovery_message(server: str, tool_count: int) -> str:
    """Render the MCP-recovery text injected into the model's context.

    The counterpart to :func:`format_mcp_unavailable_message`, and the reason
    it is a dedicated formatter rather than a :func:`classify_incident`
    category: the recovery announces the exact opposite of a failure, so
    classifying it would have appended a failure category plus
    ``Incident.render``'s "This is why the previous turn ended". The ``mcp``
    rule that used to be the live example of that mismatch — it matched the bare
    substring "mcp", so it took both halves of this pair — is deleted; both
    halves now have their own formatter instead.

    Present-tense STATE, like :func:`format_model_switch_message`, so the model
    treats it as context for the turns that follow rather than an instruction
    to acknowledge.

    The SUPERSEDE sentence is load-bearing and must not be trimmed to a bare
    "reconnected": the model is holding an earlier
    :func:`format_mcp_unavailable_message` row for this server that says its
    tools are gone and not to call them, and two live claims leave it free to
    defer to the older, more emphatic one. It must be told the earlier notice
    no longer applies.

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
    still worth announcing, because it is what supersedes the warning and
    stops the model reporting the server as down.
    """
    if not tool_count:
        # Honest about the CONNECTION and silent about callable tools: no
        # "available again", no "call them normally". The supersede clause is
        # still required — the model is holding a warning that says the
        # server is unreachable, which is no longer true.
        return (
            f"[mcp recovery] MCP server '{server}' is connected again, but it "
            "currently exposes no enabled tools. This supersedes the earlier "
            "warning about this server: the server itself is no "
            "longer failing, so stop reporting it as unavailable — but do not "
            "expect callable tools from it until some are enabled."
        )
    # Verb agreement is spelled out rather than templated: the design's draft
    # formatter read "1 tool are available again", which is text the model
    # actually reads.
    tools = "1 tool is" if tool_count == 1 else f"{tool_count} tools are"
    return (
        f"[mcp recovery] MCP server '{server}' is connected again and {tools} "
        "available again. This supersedes the earlier warning about "
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
