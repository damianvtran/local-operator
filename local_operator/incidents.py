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
from collections.abc import Sequence
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
    #
    # AND SO IS WHOSE RUN IT WAS (UX round 1, U5). "do not assume the request
    # completed" named a request the reader does not always have: the run this
    # row describes may be a DELIVERY turn, a wake run or a resume catch-up —
    # harness-initiated, opened with no human waiting on a receipt — and in that
    # case the operator's own request HAD completed, which is precisely the
    # falsehood that cost a turn re-verifying finished work in the report this
    # change answers. The subject is the WORK the run was doing, spelled out
    # rather than left as a pronoun with two candidate antecedents ("it" could
    # read as the turn or as the thing the turn was mid-way through — UX round 2,
    # U11), and it is true whichever owner the run had.
    "cut-off": "The runtime was cut off before this turn produced a result. The transcript "
    "holds whatever was written before it stopped and nothing after. Check the "
    "state of anything it was mid-way through before repeating the work; do not "
    "assume that work finished.",
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
#: that enforces it, never typed here: the sentence is repeated by every surface
#: that repeats a cut-off (the live notice, the durable outcome, the sidebar), and a
#: second copy of "15 min" is a copy that drifts.
#:
#: WHICH RUNTIMES CAN STILL RECORD THIS TOKEN: only ones that ran a build BEFORE the
#: force-cut was removed. No arm records it any more — ``process._abandon_move``
#: keeps the build it loaded and publishes ``UPDATE_FAILED_CAUSE`` instead — and the
#: sentence STAYS for the reason the phrase does (see
#: ``types.LEAVING_FOR_BUILD_OVERDUE``): a row written by an older runtime is read by
#: this one, and ``death_verdict`` rung 2 narrates whatever token a row carries.
#: Deleting the key would quietly demote those rows to the unknown-cause sentence,
#: which is a worse report than a historical one.
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


def _stall_bound_cause_sentence() -> str:
    """The sentence for :data:`STALL_BOUND_CAUSE`, written in the same shape.

    THE BOUND IS RENDERED FROM THE CONSTANT for the reason
    ``_overdue_cause_sentence`` gives, with one qualification stated rather than
    hidden: the bound an operator may override
    (``LOP_RUNTIME_STALL_SECONDS``) is not knowable from here, so this names the
    DEFAULT and says so. The exact figure the runtime was armed with is in its
    own dump header, and the detail a reader adds names which leg fired — that is
    where an operator who moved the number looks, rather than in a taxonomy
    sentence shared by every runtime on every host.
    """
    from local_operator.session.runtime.stall_watchdog import DEFAULT_STALL_S
    from local_operator.session.runtime.types import bound_text

    return (
        "the runtime ended ITSELF: its own stall bound fired after "
        f"{bound_text(DEFAULT_STALL_S)} of no progress by default, and the dump it "
        "left beside its log names what every thread was doing"
    )


def _update_failed_cause_sentence() -> str:
    """The sentence for a bounded handover that gave up (``types.UPDATE_FAILED_CAUSE``).

    A DIFFERENT EVENT from the rung above, said in the same shape, and the difference
    is the one the operator acts on: the overdue handover LEFT (on the old build's
    work, force-cut), while a failed update STAYED — the runtime is still here, on the
    build it loaded, and it is the update that did not happen. Rendering the two with
    one sentence would tell a reader to go looking for a handover that never occurred.

    IT NAMES NO NUMBER, and that is a correction rather than an omission (design
    review round 1, D2). THREE arms publish this token with THREE different bounds —
    the bounded update WINDOW (``buildwatch.UPDATE_LOCK_S``, seconds), the build DRAIN
    whose work went SILENT (``types.BUILD_DRAIN_PROGRESS_S``, fifteen minutes) and the
    build drain that was HELD with work still in flight
    (``types.BUILD_DRAIN_DWELL_S``, thirty minutes; ``process._abandon_move``'s dwell
    arm, which is the second bound the drain can run out of) — so a sentence rendered
    from any one constant is wrong for the other two by construction: an update that
    spent half an hour in the dwell reported that it "did not finish within 5s". The
    bound is known at the only place that can name it, so it rides the incident's
    DETAIL (``RuntimeServer.note_update_failed`` renders it from the argument its
    caller passed), and this sentence says the thing that is true of every arm.
    """
    return (
        "the update to the build on disk did not finish within its bound; the runtime "
        "kept the build it loaded"
    )


def update_failed_detail(pair: str, bound: float) -> str:
    """The detail a failed handover's incident carries: the pair, and the bound spent.

    A HELPER RATHER THAN AN INLINE JOIN, because the bound is the fact that was
    silently wrong until design review round 1 (D2): the sentence names none (three
    arms publish this token with three different bounds), so this is the ONLY place a
    reader can learn which bound the runtime actually ran out of — 5 s for the bounded
    update window, fifteen minutes for a drain whose work went silent, thirty minutes
    for a drain held with work still in flight (``types.BUILD_DRAIN_DWELL_S``).
    ``bound`` of 0 means the caller did not say, and then no number is claimed rather
    than a default being invented.
    """
    from local_operator.session.runtime.types import bound_text

    parts = [part for part in (pair, bound_text(bound) if bound else "") if part]
    return "; ".join(parts)


#: The cause token for a runtime that vanished with its turn still in flight —
#: the verdict :func:`journal.death_verdict`'s unattributed arm RETURNS, and the
#: one key of :data:`CUT_OFF_CAUSES` that is a reader's conclusion rather than a
#: runtime's own last word.
#:
#: NAMED BECAUSE ONE READER HAS TO REFUSE IT (review round 4, MINOR 1).
#: ``death_verdict``'s rung 2 narrates whatever token a row recorded, with no lead,
#: and that is right for a token the runtime writes about ITSELF —
#: ``runtime-overdue`` names its own mechanism and its bound. It is wrong for this
#: one: its whole meaning is that the act was never recorded, so a row carrying it
#: must reach the arm that says so rather than be answered by itself. Neither side
#: may be re-spelt, because the two sides are a key of one dict and a return value a
#: few lines apart, and a rename that moved only one of them would put a
#: harness-caused death on the arm that cannot name its actor.
KILL_CAUSE = "runtime-killed"

#: The class a runtime's OWN stall bound leaves behind when it fires — the one
#: death in this taxonomy whose author is the victim itself.
#:
#: WHY IT NEEDED A NAME, and what its absence cost. ``stall_watchdog`` arms a C
#: timer that, past its bound, dumps every thread — and on the builds that shipped
#: before 2026-09-23 it then ``_exit(1)``ed from that same C thread, which is the
#: ending this class was coined for. Production expiry is DUMP-ONLY now, so a fire
#: ends nothing and only a LEGACY dump can still witness such a death
#: (:func:`stall_watchdog.fire_outcome`). What holds for both eras is that the dump
#: names the pid and the arming time and nothing else, and until this token
#: existed there was no class anywhere that said the bound had done it. Measured
#: 2026-09-21: a peer session (pid 4698) died with its dump present and its death
#: recorded as ``unattributed`` — not "we could not tell", which is what
#: :data:`KILL_UNATTRIBUTED` is for, but **"the instrument knew and did not say
#: so"**. A firing instrument that cannot name itself leaves a reader unable to
#: tell a bound that fired from a bound that never fired, which is the whole
#: value of having armed it.
#:
#: ONE TOKEN FOR BOTH LEGS, and the legs are told apart in the DETAIL rather than
#: in the class. They are the same event to everyone downstream — the runtime
#: ended itself, on purpose, and left a dump — and splitting them into two
#: taxonomy keys would make every reader that branches on ``CUT_OFF_CAUSES``
#: learn a distinction it does not act on. What a reader DOES act on ("was this
#: a wait or a spin?") is the leg, and ``stall_watchdog.fired_leg`` names it.
STALL_BOUND_CAUSE = "runtime-stall-bound"

#: The LOOP'S own bound (``harness/loop.py``, the outer-loop continuation guard).
#: Involuntary, and NOT an error the model made: the turn had more work queued
#: than its producer's budget allows. Named rather than left as the bare
#: ``AgentEndEvent`` it used to be, because a bare end reads as a COMPLETED
#: ANSWER on every surface — a child cut off mid-list arrived at its parent
#: saying it had finished, which is the silent half of the subagent-stall
#: report. It is a token in this table rather than a private string so
#: :func:`is_cut_off_cause` answers True and every existing renderer (the
#: attention outcome, the subagent panel's "cut off" row, the roster) names it
#: without being taught a new field.
CONTINUATION_LIMIT_CAUSE = "continuation-limit"

CUT_OFF_CAUSES: dict[str, str] = {
    DELIBERATE_CUT_OFF_CAUSE: "the session was stopped by the user",
    # THE RUNTIME'S OWN BOUND ARMED AGAINST ITSELF. The sentence says who acted
    # (the runtime, not the operator and not another process) because that is the
    # fact this arm exists to state: every other involuntary arm here names an
    # actor outside the victim, and a reader who found one of those would go
    # looking for a reaper or a sweep that never ran.
    STALL_BOUND_CAUSE: _stall_bound_cause_sentence(),
    # Bounded on PURPOSE, unlike the arms above: the bound is a budget the loop
    # chose to spend, so the sentence says what ran out rather than naming an
    # actor — there is no reaper and no process to go looking for.
    CONTINUATION_LIMIT_CAUSE: (
        "the turn kept being asked to continue, so it was stopped and the pending message dropped"
    ),
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
    # The BOUNDED UPDATE WINDOW (``process._abandon_update_window``): the idle
    # handover ran out of its heartbeat bound, so the runtime ABANDONED the move and
    # kept the build it loaded (``types.UPDATE_FAILED_CAUSE``). It is NOT
    # ``runtime-retired``, which is what a handover that SUCCEEDED records — a failed
    # update narrated as an ordinary retirement is invisible in exactly the durable
    # account an operator opens to ask why a session is still on yesterday's build.
    "runtime-update-failed": _update_failed_cause_sentence(),
    KILL_CAUSE: (
        # The trailing clause is the POST-MARKER meaning of this token, and it
        # is decidable now in a way it was not before the durable marker
        # existed: "disappeared without exiting cleanly" describes what the
        # PROCESS did, and a reader one row under a deliberate stop needs the
        # other half — that nobody ASKED for it. Without the clause the two rows
        # differ only in the word above them, and "asked for" versus "never
        # asked" is exactly the distinction this taxonomy was extended to draw
        # (design round 1, D4).
        #
        # THE CLAUSE SAYS "ASKED FOR", NOT "RECORDED A STOP", and the change is
        # the involuntary markers' doing: while the ladder was the only writer,
        # "nothing recorded a stop" was true of every death on this arm. It stops
        # being true the moment a party that merely ACTS on a runtime (a prune
        # removing the install generation under it, an in-place install rewriting
        # it) records what it is about to do — the artifact then says "nothing
        # recorded a stop (its install generation was pruned by lop install prune)",
        # which contradicts itself where the operator reads it. What is true of
        # BOTH is the discriminator the clause exists for: no stop of this runtime
        # was asked for. The attribution parenthetical says who acted instead.
        "the runtime disappeared without exiting cleanly while this turn was running, "
        "and no stop was asked for"
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

#: What a ``runtime-killed`` verdict says when NO actor was recorded for the act.
#:
#: WHY THE WORD EXISTS AT ALL, in the operator's own requirement: "nothing should
#: kill runtimes en masse, ever; and if it does happen, it must be attributable."
#: The 2026-09-18 event killed 25 runtimes in 13 seconds and the artifacts could
#: not name a single process, because the only markers existed for stops the USER
#: asked for. A death whose marker names its actor now renders that actor; a death
#: whose evidence proves only that a turn was in flight and nobody recorded an act
#: says ``unattributed`` — an affirmative statement about the GAP rather than the
#: old shrug, and the one word that turns "we lost 25 runtimes and cannot say why"
#: into "none of these 25 had a recorded actor", which is a fact an investigation
#: can act on.
KILL_UNATTRIBUTED = "unattributed"

#: What an INVOLUNTARY act on a runtime is called, in the operator's words.
#:
#: Keyed by the ``mechanism`` token the writer stamps into the marker
#: (``control.note_involuntary_stop``), which is why the labels are clauses rather
#: than agentless nouns: they are rendered as ``<label> by <actor>``, the same shape
#: :func:`render_stop_attribution` uses, so the two kinds of attribution read alike
#: on the one surface that shows both.
#:
#: A mechanism this build does not know renders as the actor alone (see
#: :func:`render_involuntary_attribution`) rather than leaking a raw token — the
#: rule :data:`STOP_RUNG_UNKNOWN` already states for a future rung, for the same
#: reason: the writer's spelling must not reach a surface that cannot explain it.
INVOLUNTARY_MECHANISM_LABELS: dict[str, str] = {
    "generation-prune": "its install generation was pruned",
    "in-place-install": "its install was being replaced in place",
}


def render_involuntary_attribution(
    *, mechanism: str = "", actor: str = "", killer_pid: object = None
) -> str:
    """The parenthetical an INVOLUNTARY kill's reason carries, or ``""``.

    Scalar arguments rather than the marker dict, for the reason
    :func:`render_stop_attribution` gives: the schema belongs to its writer, and a
    field renamed on one side must not silently empty the sentence on the other.

    Nothing named at all returns ``""`` — the caller then says
    :data:`KILL_UNATTRIBUTED`, because "no actor recorded" is a different fact
    from "an actor we cannot print" and only one of them is a gap.
    """
    label = INVOLUNTARY_MECHANISM_LABELS.get(mechanism, "") if mechanism else ""
    who = actor or ""
    if killer_pid is not None and str(killer_pid).strip():
        who = f"{who}, {KILLER_PID_LABEL} {killer_pid}".lstrip(", ")
    if label and who:
        return f" ({label} by {who})"
    if label:
        return f" ({label})"
    if who:
        return f" (by {who})"
    return ""


def involuntary_kill_detail(
    *, mechanism: str = "", actor: str = "", killer_pid: object = None
) -> str:
    """The detail a ``runtime-killed`` reason carries for an INVOLUNTARY act.

    ONE decision point rather than the same ``or`` at each arm that needs it: the
    reader's question is "who was recorded", and the answer is either an
    attribution or :data:`KILL_UNATTRIBUTED` — never a blank where a reader
    expects the sentence to say which of the two it is.

    The parenthetical always OPENS with a space so
    :func:`render_cut_off_reason` treats it as an aside (its separator rule) — the
    shape ``attention._record_detail`` already hands over.
    """
    return (
        render_involuntary_attribution(mechanism=mechanism, actor=actor, killer_pid=killer_pid)
        or f" ({KILL_UNATTRIBUTED})"
    )


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


def render_cut_off_reason(cause: str, *, detail: str = "", clause: str = "") -> str:
    """One operator-facing sentence naming why a turn was cut off.

    This is the string the durable outcome stores as ``reason`` and every
    surface prints after its own prefix (``Stopped with an error — …``), so it
    is deliberately a sentence and not a paragraph: ``detail`` carries the
    why-now clause (a build pair, a pid, a started-at) rather than more prose,
    because the sidebar tooltip has one line and truncates the rest rather than
    wrapping it. The caller supplies the text of that clause and nothing else —
    its punctuation is added here (see :func:`_render_cut_off_detail`).

    ``clause`` IS OUTSIDE THE PARENTHETICAL, and that placement is the whole point
    (design review round 1, D3): a fact that belongs IN the sentence must not be
    filed with the build/pid detail, because the LIST column that renders a reason
    keeps only the first clause — ``outcome_summary`` splits at ``" ("`` — so a lead
    put in there reaches no list a person reads. The held-fire lead is exactly that
    kind of fact: it changes what the row MEANS (the runtime survived a bound and is
    still stalled) rather than adding detail about a death it did not cause.
    """
    sentence = CUT_OFF_CAUSES.get(cause) or CUT_OFF_UNKNOWN
    lead = f"; {clause}" if clause else ""
    return f"{sentence}{lead}{_render_cut_off_detail(detail)}"


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

    WHAT THE SPLIT KEEPS IS DELIBERATE, and it is why a fact that changes what the
    row MEANS has to live outside the parenthetical: the held-fire lead rides in the
    first clause (``render_cut_off_reason``'s ``clause``) precisely so that this
    function does not drop the one sentence that tells a reader a runtime survived
    its bound and is still stalled (design review round 1, D3 — with the lead inside
    the brackets, a held-then-killed death and a no-dump death rendered
    byte-identical here).
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


def format_shape_incident_message(
    tool: str,
    labels: "list[str]",
    summary: str = "",
    *,
    reached_model: bool = True,
) -> str:
    """Render the credential-SHAPE notice injected into the model's context.

    The counterpart to the shape pass in :mod:`local_operator.redaction_shapes`,
    and the reason it is its own formatter rather than a
    :func:`classify_incident` category: nothing FAILED. A tool carried a
    credential in a shape the table recognises, the harness handled it, and the
    jobs this text has are to say what happened (so the model does not reason
    about a value it cannot see, or re-run the command hoping for a different
    result) and to make the event visible to the operator.

    **Two classifications, and only one of them is an emergency.**

    A credential is COMPROMISED when a value reaches the model's context window:
    that text is journaled in plaintext, replays into later requests, and may end
    up in training data, and none of that can be undone by anyone here — so the
    operator has to rotate the credential, and that is the escalated text. That
    case is the one where readable credential material survives in the text the
    model reads — which is NOT always a mask that fell short, because a rule can
    preserve a run by design (the DSN rule keeps its userinfo username, so
    ``amqp://guest:guest@…`` reads back the password as the username) — and it is
    carried in by ``reached_model``.

    Everything else is CONTAINED. A value that reached `bash` (in a command's
    ``argv``, in a child's environment), that lived in this process's memory, or
    that was written to a file in plaintext is NOT compromised: the model never
    saw it, so there is nothing to rotate. What the contained wording asks for is
    cleanup, not rotation: delete any plaintext copy, and do it WITHOUT reading
    it, because reading it is what would turn the contained case into the
    escalated one.

    **That wording is not raised in-tree any more, and this docstring is where the
    disposition is recorded rather than left implicit.** The operator asked for the
    contained case to file nothing, and ``Session._queue_shape_incident`` is the
    single gate that drops it, so today no caller reaches this branch (agent review
    R1, finding 2). It is kept — not deleted — because it IS this formatter's
    contract and because the obligation it carries has to exist somewhere:
    *delete any plaintext copy a tool call may have written* is the only cleanup
    instruction this system has ever stated, and with the contained case silenced
    it now has no operator-facing surface anywhere in-tree. A caller that
    deliberately has something to say about a contained hit (a write-side advisory,
    say) can render the true words instead of inventing them; a reader looking for
    where the cleanup obligation is surfaced will find this paragraph and know that
    it is not.

    The wording is deliberately SURFACE-NEUTRAL about where the
    masking happened: the same notice serves a credential in a tool's OUTPUT, one
    TYPED INTO a call's arguments — where the tool did run with the real value and
    the containment is in the copy this session stores and replays — and one found
    by the history-scrub path, and a claim that named the wrong surface would be
    false in two of the three (agent review R1, finding 2).

    Whether a plaintext copy exists at all is not knowable from here — the value can
    reach this notice from a command's own text, which the history-scrub path masks
    through the same formatter — so the cleanup obligation is stated conditionally
    rather than dropped: it is the one action this path has.

    ``labels`` are shape NAMES, never values — a notice that carried the
    credential would be the leak it exists to report. The summary is the one the
    harness built, already scrubbed and bounded.

    ``reached_model`` defaults to True — the escalated reading — because a caller
    that cannot classify must not make the quieter claim.

    An EMPTY ``labels`` on the escalated path is a real state, not a caller error:
    a hit whose mask cannot be claimed (``complete=False``) is dropped from
    ``ShapeReport.labels`` while it still escalates if any readable material
    survived (``exposed=True``), so ``labels=()`` with ``reached_model=True``
    reaches here — see the shape-clause note in the body.
    """
    shapes = ", ".join(labels) if labels else "credential-shaped content"
    tool_name = tool or "a tool"
    where = f" The call was: {summary}." if summary else ""
    # "reached you ... and was masked" rather than "the result carried": the same
    # notice serves a credential in a tool's OUTPUT and one TYPED INTO a call's
    # arguments, and only the first of those is a result. A notice that misnamed
    # the surface would send an operator looking in the wrong place.
    # The bracketed head stays in both texts — the harness's notice-row rules key
    # on "[credential redaction] " (``harness/rows.py``), and a row that painted as
    # the user's own words would be worse than a jargon-first one.
    if reached_model:
        # WHEN THE TABLE COULD NOT NAME THE SHAPE, SAY SO — do not assert a generic
        # one. The reader's first question is "what matched?", and the ``shapes``
        # fallback above ("credential-shaped content") reads as a shape the table
        # DID identify, leaving an operator unable to tell an un-named hit from a
        # named one. This state is reachable on the shipped path, measured this
        # session on the corpus's own escalating case (the ``amqp`` DSN whose
        # username is its password): its hit grades ``complete=False, exposed=True``,
        # so ``shape_report`` yields ``labels=()`` with ``reached_model=True`` — an
        # ESCALATION naming no shape. Only the provenance clause changes; what the
        # guard MEASURED, and the rotate instruction below, are untouched.
        if labels:
            credential = f"a credential ({shapes})"
        else:
            credential = "a credential the shape table could not name"
        # The CAUSE has to be true in both directions the escalation covers. The
        # `amqp` DSN case masks its password whole and still escalates, because the
        # DSN rule keeps the userinfo username by design and an operator who used
        # one string for both leaves the value readable there — so "could not be
        # fully masked" named a mechanism this notice cannot prove, and this
        # module's doctrine is that an unprovable claim is worse than silence
        # (agent review R2, finding 1). What the check measured is stated instead:
        # the value is readable in the text the model gets, by either route, and
        # the notice does not pick one it cannot distinguish. The rotate
        # instruction is untouched — it is the reason the wording exists.
        return (
            f"[credential redaction] rotate it — {credential} reached "
            f"{tool_name} and its value is readable in this session's context: "
            f"either the mask did not cover it fully, or it survives in the text "
            f"another way.{where} A value that reached the model may be in "
            "training data, so treat it as compromised: the operator has to rotate "
            "it. Do not re-run the command to read the value."
        )
    return (
        f"[credential redaction] a credential ({shapes}) reached {tool_name} and "
        f"was masked before you saw it.{where} Nothing entered your context — the "
        "value was masked before it reached you, so there is no exposure. If the "
        "call wrote it to a file in plaintext, delete that file without reading it "
        "(rm -f): reading it is not needed and is not to be done."
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
    ``Reason:`` reads as a truncation. It is shaped COMMAND-FIRST at the two
    auth call sites (``mcp/manager.py``), so the one part a reader must not lose
    lands at the front of the line rather than mid-line after a restatement
    (design review round 1, D3). The breaker site is the third caller and names
    no verb: its remedy is a reconnect rather than a typed command, so its reason
    states the condition (``auto-reconnect suspended after >N attempts``) and the
    row's last line defers the recovery to the user without naming one.
    """
    lines = [f"[session warning] MCP server '{server}' is unavailable: its tools are gone for now."]
    if reason.strip():
        lines.append(f"Reason: {reason.strip()[:200]}")
    lines.append(
        "Its tools are not callable until the user restores it, and the agent "
        "should not retry them in a loop."
    )
    return "\n".join(lines)


def format_held_delivery_message(jobs: Sequence[tuple[str, str]], *, reason: str = "") -> str:
    """Render the row for job results the harness FAILED to hold for the next turn.

    ``jobs`` is ``(job_id, label)`` per result. The ID is carried as well as the
    label because the row's own remedy is addressed by id, and a reader shown only
    a label has nothing to substitute into it (UX round 2, U9 — the rendered head
    used to name labels while the suggested action said ``<job id>``).

    A dedicated formatter, authored HERE rather than as a paragraph at the call
    site, for the reason :func:`format_mcp_unavailable_message` gives: every
    operator- and model-facing incident sentence in this codebase is built in
    one place, and a hand-written one at a call site carries no head, no
    ``suggested action:`` slot, and none of the structure a reader's eye uses to
    tell a system record from the agent's own prose (design review round 1, D2 —
    measured in a rendered frame, where this row sat directly under the MCP
    warning's labelled shape and read as the agent narrating).

    IT LEADS WITH WHAT FAILED, because this row has exactly one caller — the
    failure arm of ``Session._hold_job_results_for_next_turn`` — and an earlier
    revision spent its first clause on the hold it did NOT achieve ("so they were
    held for the next turn"), contradicting its own parenthetical in the one path
    where acting is time-boxed: a reader told the reports were held has no reason
    to go and read them (design review round 2, D7; UX round 2, U10).

    The incident HEAD, because this is a session-level fact the next turn has to
    know before it trusts the conversation; but NOT ``Incident.render``'s
    closing tail ("This is why the previous turn ended"), which is FALSE here —
    the same reason the MCP row does not go through the classifier. Nothing
    failed and no turn ended: the runtime left while results were still arriving,
    and the operator's own turn completed.

    ONE PARAGRAPH FOR THE BATCH, however many results it names (design review
    round 1, D4). The incident's own batch was nine children, and a per-result
    notice painted nine near-identical warning paragraphs — the shape this file's
    own delivery contract argues against ("N children that settle during one
    parent turn are one piece of news"). The count and the ids carry the
    multiplicity; the per-job detail belongs in the log.

    THE ROUTES IT NAMES ARE THE ONES THAT ACTUALLY WORK, and it says what does
    not (design round 1 D1/D3; UX round 2, U9). The first draft sent the reader to
    the job ledger, refuted twice over: ``retention_expired`` drops a settled row
    on any read after ``DEFAULT_RETENTION_MS`` — five minutes, while this row's
    reader is a turn that may be hours away — and a successor's row is
    ``restored``, exempt from that window but carrying NO result text (the roster
    sidecar persists none), so ``wait`` answers with a bare header there.
    ``hub op='peek'`` reads the child's own transcript, which the sweep never
    touches — but only a SUBAGENT child has one: ``_on_job_completed`` also
    accepts background ``bash``, whose output lives in the job ledger and nowhere
    else. So the row names the transcript route for the children that have one,
    names the ledger route with its bound for the ones that do not, and does not
    pretend the same instruction covers both. The bound is rendered from the
    constant that enforces it rather than typed, like every other bound in this
    file (design review round 1, D3).
    """
    from local_operator.harness.jobs import DEFAULT_RETENTION_MS
    from local_operator.session.runtime.types import bound_text

    named = [
        f"{label} ({job_id})" if label and label != job_id else job_id for job_id, label in jobs
    ]
    count = len(named) or 1
    noun = "result" if count == 1 else "results"
    subject = "it" if count == 1 else "they"
    them = "it" if count == 1 else "them"
    head = (
        f"[session incident] held delivery: {count} background job {noun} could not be held "
        f"for the next turn — {subject} arrived after this session's runtime had committed "
        f"to leaving, and the harness could not write {them} into this conversation"
    )
    if reason.strip():
        head += f" ({reason.strip()[:200]})"
    lines = [head + "."]
    # EVERY id, with no "+N more" (design round 3, D16 = UX U13 = review MINOR-4):
    # the route below is addressed by id, so a truncated list made it followable
    # for five of nine — the incident's own shape. The list is reference material
    # on its own line, not part of the sentence a reader skims.
    if named:
        lines.append(f"Jobs: {', '.join(named)}")
    lines.append(
        # No present-tense claim about who has read these (review round 3,
        # MINOR-3): the row is durable and re-rendered forever, so "nothing has
        # read it" would become false the moment a later turn did read it — the
        # same class D10/U7 fixed for the held notice. "never reached this
        # conversation" is a fact about the write that stays true.
        f"suggested action: {subject} never reached this conversation — a subagent child "
        "keeps its own transcript, readable by id with hub op='peek' <job id> (paging with "
        "range='a-b'), and a background bash command has no transcript at all: its output "
        "lives only in its job row, which the ledger drops once "
        f"{bound_text(DEFAULT_RETENTION_MS / 1000.0)} have passed since it settled — and it "
        "is dropped on the next read after that, or sooner when the sweep runs on a "
        "settle, a cancel or a delivery sink's exit, so by the time this row is read that "
        "row is usually gone."
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
