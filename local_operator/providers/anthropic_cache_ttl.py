"""Anthropic prompt-cache TTL policy: which requests carry the 1-hour TTL.

WHY THIS IS A POLICY AND NOT A THRESHOLD. A 1h cache write costs 2x base input
against 1.25x for the default 5m, and pays back only when the SAME prefix is
read again after a 5-minute gap. Whether that happens depends far more on WHO
is asking than on how big the context is, which is what this harness's own
7-day ledger showed (2026-10-26, 59,016 Anthropic calls, ``analytics.db``;
figures in base-input-token equivalents, 5m write 1.25x, 1h write 2x, read
0.1x):

* MAIN thread, 1h vs 5m above 150k: 1h saves ~75M net (391 expiry rewrites
  avoided). The main conversation is what waits on the user, on subagents and
  on scheduled wakes, so its prefix outlives five minutes routinely.
* MAIN thread below 150k on 1h: ~+0.8M net. Small but positive — and the
  resume-after-a-break latency is something the user feels.
* SUBAGENTS above 150k on 1h: ~15M net WORSE than 5m (345 expiries avoided,
  113M of extra 2x write cost). A subagent runs a dense tool loop and then
  ends; its prefix is rarely read again after a 5-minute gap.
* SUBAGENTS below 150k on 1h: clearly worse (-57M).

That is the same split Claude Code (v2.1.274, decoded from the shipped binary)
makes: on a Claude subscription within plan limits its main REPL thread
always uses 1h, subagents/compaction/titles use 5m, and once the account is
drawing on overage ("extra usage") credits everything drops to 5m because
overage is billed like the API. API-key traffic is 5m unless opted in; lop's
opt-in is the long-standing size threshold, kept for the API-key main thread.

ROLES. ``main`` is a conversation turn of a top-level session; ``subagent`` is
a turn of a delegated child (a forked ``SessionStreamFn``, the same edge the
analytics ledger records as ``parent_session_id``); ``auxiliary`` is every
other ``purpose`` — compaction, naming, asides, clearance, supplements, team
bridge screens. Auxiliary calls are one-shots: nothing reads their write back
after five minutes, so they are always 5m and have no knob.

OVERAGE. Detected the way Claude Code detects it, from the unified rate-limit
headers every subscription response carries: the account is "using overage"
when ``anthropic-ratelimit-unified-status`` is ``rejected`` (plan limits spent)
while ``anthropic-ratelimit-unified-overage-status`` is ``allowed`` or
``allowed_warning`` (extra-usage credits are serving the request). The state is
per account and process-wide, because wire clients are rebuilt per call and a
response is the only place the fact appears; every subscription response
re-states it, so a reset window self-corrects on the next call. API-key
responses do not carry these headers and never set it.

NO LATCH. A TTL change does not re-key the cached prefix: measured live
(PR #2985 review round 1, ``claude-sonnet-5-5``, ~30k unique prefix), a 1h
write is read in full by a following 5m-marked request and vice versa, and a
5m-marked read does not shorten a 1h entry's life. So each request takes its
own TTL with no per-conversation state; the only wire constraint is one TTL
per request (a 1h marker may not follow a 5m one). The same fact is why the
auxiliary 5m errands (asides, clearance) still read the main thread's 1h
prefix.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Literal

CacheRole = Literal["main", "subagent", "auxiliary"]

#: Response header carrying the subscription's plan-limit state. Values seen in
#: Claude Code's parser: ``allowed``, ``allowed_warning``, ``rejected``.
UNIFIED_STATUS_HEADER = "anthropic-ratelimit-unified-status"
#: Response header carrying whether extra-usage (overage) credits may serve.
UNIFIED_OVERAGE_STATUS_HEADER = "anthropic-ratelimit-unified-overage-status"
_OVERAGE_SERVING = frozenset({"allowed", "allowed_warning"})

#: A subagent conversation turn that ran at least this long, request start to
#: stream end, moves the REST of that subagent's turns to the 1h TTL.
#:
#: Why a duration and not a size: the 5m TTL is refreshed when a request STARTS
#: (the prefix is read/written then), so a call that streams for more than five
#: minutes (30-70k output tokens) lets its own prefix expire before the next
#: request goes out, and that next request rewrites the whole prefix.
#:
#: Reproducible replay (PR #3072 review round 1; read-only ``analytics.db``,
#: 2026-10-27 04:34 to 2026-10-28 03:18 local, 0.948 days; Anthropic
#: ``purpose='turn'`` subagent rows billed ``subscription_api_equivalent``:
#: 49,796 calls, 665 sessions). A BREAK is a row with ``cache_miss_type =
#: 'unchanged'`` (compared, no prefix divergence, so pure expiry) whose cache
#: read is under 50% of the previous call's read+write, with the previous
#: context >= 20k and this write >= 20k: 167 breaks, 21.67M rewrite tokens. 42
#: of them (5.70M) follow a call >= 240 s, 37 of those at start-to-start gaps of
#: 315-684 s; the other 125 are ordinary idle expiry this rule does not target.
#: Simulating "after a subagent session's first ok call >= 240 s, send every
#: later write at 1h", weighting 5m write 1.25x, 1h write 2.0x and read 0.1x
#: base: 46 breaks (6.49M) avoided against 6.56M premium writes, net +2.54M,
#: i.e. about +2.68M base-equivalent tokens/day. A 300 s threshold nets only
#: +2.15M/day (17 sessions armed vs 27), so 240 s is kept: it also leaves
#: headroom for the gap between one call's end and the next one's start.
#: Caveats: under one day of data, and an avoided break's incremental write is
#: approximated as 0.
#:
#: Why "after the first one" is a usable predictor: long calls cluster (scout
#: analysis on the ledger since v0.90.18: once a session has had one call
#: >= 240 s, 5.0% of its later calls are long, against 0.1% otherwise).
#:
#: A module constant, not a config key: it is a measured property of the
#: workload, not an operator preference, and the kill switch already exists
#: (``cache_ttl_1h_min_context_tokens: 0`` turns every subscription 1h off).
SUBAGENT_LONG_CALL_1H_THRESHOLD_S = 240.0

#: An AUTOMATION main thread (``lop exec``, an SDK session) whose consecutive
#: turn requests were ever this far apart, START to START, moves back to the
#: ordinary main-thread TTL for the rest of the run.
#:
#: Why automation is 5m at all: the main thread's 1h exists for the IDLE gap at
#: a turn's end (the user, a wake, a subagent). In the solo exec runs measured
#: below, the next model call followed the previous tool results within
#: seconds and the run ended at the turn's end, so the 2x write bought nothing.
#: Team-led and ``--workstream`` runs DO idle inside the run (a manager waits
#: on its subagents), so the entry points do not mark them.
#: Measured (PR #4551 QA, t3_disc paired run, Bedrock Sonnet, ``/Users/Shared/
#: lopqa-c4551-54ba2d/paid``): lop's r2 unit crossed 150k and wrote 59,231
#: tokens at 1h on its last two calls, $0.098 of a $0.737 run; across all 56
#: QA ledgers under ``/Users/Shared/lopqa-*`` (234 turn calls, 179
#: consecutive pairs) the start-to-start gap had median 3.0 s and max 28 s.
#:
#: Why the call that ENDS a turn cannot be picked out instead: the TTL is fixed
#: when the request is SENT and the write happens then; whether that call ends
#: the turn (no tool calls) is known only from its response.
#:
#: Why START to START and why 240 s: the 5m entry is refreshed when a request
#: starts, so a start-to-start gap over 300 s is exactly an expiry, whether the
#: time went into a long stream or a long tool (a minutes-long bash, a subagent
#: ``wait``). The first such gap pays one rewrite, as ``after_long_call`` does
#: for subagents; gaps cluster (a manager run that waited once waits again), so
#: the rest of the run returns to 1h. 240 s is the same headroom the subagent
#: rule keeps below 300 s, and sits 8.6x above the largest gap measured above.
AUTOMATION_LONG_GAP_1H_THRESHOLD_S = SUBAGENT_LONG_CALL_1H_THRESHOLD_S


def mark_automation_session(session: object) -> None:
    """Mark ``session``'s own stream as an automation run (exec, SDK); never raises.

    The one helper both entry points call, so "which sessions are automation"
    is decided at the two places that know it rather than inferred from a
    missing viewer (a viewer-less interactive runtime still idles for the
    user). ``getattr`` because test doubles and attached viewers carry no
    ``SessionStreamFn``; such a session simply keeps the ordinary policy.
    """
    mark = getattr(getattr(session, "_stream_fn", None), "mark_automation", None)
    if callable(mark):
        mark()


def cache_role(purpose: str, delegated: bool) -> CacheRole:
    """Classify a request for the TTL policy from facts the request carries."""
    if purpose != "turn":
        return "auxiliary"
    return "subagent" if delegated else "main"


def decide_cache_ttl(
    *,
    role: CacheRole,
    subscription: bool,
    overage: bool,
    main_min_context_tokens: int,
    subagent_min_context_tokens: int,
    context_size: Callable[[], int],
    after_long_call: bool = False,
    automation: bool = False,
    after_long_gap: bool = False,
) -> str | None:
    """``"1h"`` or None (the 5m default) for one request.

    ``main_min_context_tokens`` is ``providers.anthropic.cache_ttl_1h_min_context_tokens``:
    the API-key / overage main-thread size threshold. Its ``0`` is the global
    "no 1h" switch: it turns the subscription main thread's 1h off AND the
    subagent after-long-call rule below (an explicit subagent size opt-in
    still applies), so an operator who set it before either existed does not
    start paying 2x writes. ``subagent_min_context_tokens``
    is ``providers.anthropic.subagent_cache_ttl_1h_min_context_tokens``, the
    subagent SIZE gate only (0 = no size opt-in; it does not switch off the
    after-long-call rule). ``context_size`` is called only when a size gate
    applies.

    ``after_long_call`` is ``ChatRequest.cache_after_long_call``: this
    conversation already had a turn of at least
    ``SUBAGENT_LONG_CALL_1H_THRESHOLD_S``. It moves a SUBSCRIPTION subagent
    (not in overage) to 1h, the one subagent case where the expiry it prevents
    outweighs the 2x write (see the constant). Overage and API-key traffic keep
    the API-key rules below, so the overage fallback still lands on 5m under
    the size threshold; the main thread is already 1h on a subscription.

    ``automation`` is ``ChatRequest.cache_automation``: this main thread is a
    non-interactive run (``lop exec``, an SDK session) whose next call follows
    its tool results within seconds and which ends at the turn's end. Such a
    main thread uses 5m on every credential until ``after_long_gap``
    (``ChatRequest.cache_after_long_gap``: two of its turn requests were ever
    ``AUTOMATION_LONG_GAP_1H_THRESHOLD_S`` apart, start to start), then takes
    the ordinary main-thread rules below. Subagents and auxiliary calls ignore
    both flags. The result stays request-wide, so every marker in one body,
    system head included, still carries one TTL.
    """
    if role == "auxiliary":
        return None
    if role == "main" and automation and not after_long_gap:
        return None
    if role == "main" and subscription and not overage:
        return "1h" if main_min_context_tokens > 0 else None
    if (
        role == "subagent"
        and after_long_call
        and subscription
        and not overage
        # The main-thread ``0`` is the operator's global "no 1h" switch; it
        # must keep meaning that for this rule too. Falling through (rather
        # than returning None) keeps an explicit subagent size opt-in working.
        and main_min_context_tokens > 0
    ):
        return "1h"
    threshold = main_min_context_tokens if role == "main" else subagent_min_context_tokens
    if threshold <= 0:
        return None
    return "1h" if context_size() >= threshold else None


# -- overage state -----------------------------------------------------------

_overage_lock = threading.Lock()
_overage_by_account: dict[str, bool] = {}


def observe_unified_headers(account: str, headers: Mapping[str, str]) -> None:
    """Record whether ``account`` is drawing on overage, from one response.

    Only a response that carries the unified status header says anything; one
    without it (API key, a proxy that strips headers) leaves the state alone.
    """
    if not account:
        return
    status = headers.get(UNIFIED_STATUS_HEADER)
    if status is None:
        return
    overage = headers.get(UNIFIED_OVERAGE_STATUS_HEADER)
    using = status == "rejected" and overage in _OVERAGE_SERVING
    with _overage_lock:
        _overage_by_account[account] = using


def overage_active(account: str) -> bool:
    """Whether the last response seen for ``account`` was served from overage."""
    if not account:
        return False
    with _overage_lock:
        return _overage_by_account.get(account, False)


def reset_overage_state() -> None:
    """Forget every observation (tests)."""
    with _overage_lock:
        _overage_by_account.clear()
