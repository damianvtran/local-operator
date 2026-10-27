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
) -> str | None:
    """``"1h"`` or None (the 5m default) for one request.

    ``main_min_context_tokens`` is ``providers.anthropic.cache_ttl_1h_min_context_tokens``:
    the API-key / overage main-thread size threshold. Its ``0`` turns every
    MAIN-THREAD 1h TTL off, the subscription one included, so an operator who
    set it before this policy existed does not start paying 2x writes; the
    subagent key is an independent opt-in. ``subagent_min_context_tokens`` is
    ``providers.anthropic.subagent_cache_ttl_1h_min_context_tokens`` (0 =
    never). ``context_size`` is called only when a size gate applies.
    """
    if role == "auxiliary":
        return None
    if role == "main" and subscription and not overage:
        return "1h" if main_min_context_tokens > 0 else None
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
