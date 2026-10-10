"""Radient out-of-credits recovery: the account-aware text a 402 carries.

WHY THIS EXISTS. A Radient request that runs out of credit fails with a bare
``out of credits (HTTP 402): insufficient credits`` (rendered, before the 402
label split, as ``rate limit or quota exceeded (HTTP 402): ...``) — true, and
useless: the account's free signup grant may simply be UNCLAIMED (the grant is
withheld until the operator follows the verification link emailed at signup),
in which case the remedy is an email, not a top-up. Nothing in the provider's
answer names that, and both surfaces that render it (the TUI error line / the
desktop's ``[session incident …]`` row) showed the bare refusal. This module
answers the ONE question a display site cannot derive from the error: what does
the Radient ACCOUNT say about its signup state?

THE FROZEN INTERFACE. ``GET /v1/me`` carries a top-level ``verification``
object, sibling to ``account``/``identity``::

    "verification": {"email_verified": bool,
                     "signup_grant": "claimed" | "pending" | "expired" | "none",
                     "grant_amount": <number, optional — captured at issue>,
                     "claim_url": "https://console.radienthq.com/dashboard/verification",
                     "first_topup": {            # optional, newer backends only
                         "bonus_amount": <number>,       # the registration bonus
                         "minimum_purchase": <number>,   # card top-up that earns it
                         "bonus_received": bool,         # true once no longer a first purchase
                         "topup_url": "https://console.radienthq.com/dashboard/billing"}}

``email_verified`` is Radient's OWN Turnstile-gated claim state and is NEVER
derived from Google/Microsoft OAuth claims, so it is read from this endpoint
and nowhere else. The object is ABSENT on an older backend and every consumer
must tolerate that: absence degrades to the neutral out-of-credits text, never
an error and never a blank. Every field of ``first_topup`` is optional too: the
bonus line is an OFFER, so it is stated only when every figure it quotes was
read and ``bonus_received`` is exactly ``false``.

WHAT THE TEXT SAYS, by account state (the copy is shared with the desktop UI
and written for this surface):

* not verified (``pending`` / ``expired``, or ``none`` with ``email_verified``
  false) — free credits are WAITING behind verification, so that comes first;
* verified (``claimed`` or ``email_verified``) — the credits are spent, so the
  top-up link, plus the first-top-up bonus line while the bonus is unclaimed;
* state unreadable (signed out, offline, ``/me`` failed, older backend) — a
  neutral message naming BOTH links conditionally. It never claims a state it
  could not read.

THE TRIGGER IS THE 402, NOT THE QUOTA KIND. A 429 is a rate limit whose remedy
is waiting, and advice about balances would be wrong for it. The gate therefore
reads the out-of-credits label ``failover`` writes for HTTP 402 (and, for a
transcript or follower written by an older runtime, the legacy quota label
carrying ``HTTP 402``).

HOW THE PROBE IS BOUNDED, and why it is shaped this way. A short-TTL process
cache fronts one bounded GET (failures swallowed — a non-200, an unparseable
body and a missing object all collapse to ``None``). A miss costs at most one
bounded request, and only on a path that is ALREADY a failed turn — the user is
looking at an error, so the worst case is that sentence arriving a few hundred
milliseconds later; a hit costs nothing. The bound is an explicit envelope
(``_CONNECT_TIMEOUT_S`` / ``_READ_TIMEOUT_S``) rather than a single number,
because httpx applies its timeout PER PHASE: the single ``5.0`` this replaces
was ~10s in the pathological case (blackholed connect, then a stalled body)
and review round 1 measured 5.04s on the connect phase alone. The token is the
STORED one, deliberately not refreshed: a refresh is a network write (and a
rotation) triggered by a turn that just failed, while a stale token merely
fails the probe and degrades to the neutral text.

THREE ACCESS PATTERNS, and the caller each one serves. THE AWAITED ARM
(:func:`append_usage_limit_recovery_async`) probes on a cache miss while
yielding the event loop — every site that can await uses it (the session's
incident journal, and the TUI's async sites through the app's awaited twin).
THE CACHED ARM (:func:`append_usage_limit_recovery_cached`) NEVER touches the
network: it answers from the cache or from the network-free "no stored
credential" read, and returns the text unchanged when only a probe could
decide. The TUI's sync handler renders through it — a sync handler runs on the
app's event loop, where a cold probe would freeze input and repaint (review
round 1, R1) — and the app completes the notice from an off-loop worker, so
the sentence still lands. THE BOUNDED SYNC ARM
(:func:`append_usage_limit_recovery`) blocks its caller for up to the probe
envelope and exists for exactly one surface: the headless renderer, whose
one-shot process exits with the line, so a cache-only answer there would
render the neutral fallback forever. Loop-side callers must not use it.

NOTHING HERE RAISES. Every branch returns a string: a recovery sentence is an
ADDITION to an error the user is already being shown, so a store read, a parse
or a fetch that goes wrong must leave that error intact rather than replace it
with a traceback — the same contract ``append_auth_recovery`` states.

NO DUPLICATES. The text this module produces is deterministic per cached
state, and :func:`append_recovery_line_once` refuses an append when the text
already carries ANY Radient recovery line (see ``_FAMILY_MARKERS``), so a
retried render cannot stack two remedies. The module is deliberately disjoint
from the AUTH recovery: quota-labelled errors are never auth-classified and
vice versa, so ``append_auth_recovery`` and this module can sit at the same
display site without double-firing.

CACHING AND WHO INVALIDATES IT. The cache holds FETCH-BACKED facts only (a
probe ran); the "no stored credential" answer is a cheap local read and is
never cached, because the moment it changes — the operator signing in — is
exactly the moment the process must stop answering from memory. A
successful-login invalidation is NOT wired in: the worst a stale entry can do
is pick a different sentence for at most ``_TTL_S`` seconds on a path that is already
failing, and reaching into the login flow to invalidate it would couple this
hint to a surface that does not own it.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

import httpx

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, typing only
    from local_operator.providers.auth_store import AuthStore

logger = logging.getLogger(__name__)

#: The claim page the frozen contract names. Used whenever the payload does not
#: carry its own ``claim_url`` (a missing field is tolerated, never fatal).
CLAIM_URL = "https://console.radienthq.com/dashboard/verification"

#: Where a top-up happens when the payload does not name a page (an older
#: backend carries no ``first_topup``), and the page the neutral text points at.
TOPUP_URL = "https://console.radienthq.com/dashboard/billing"

#: TTL of the process cache. Inside the 2-5 minute band the frozen contract
#: asks for: long enough that a quota storm costs one probe, short enough that
#: a freshly claimed grant is reflected while the user is still retrying.
_TTL_S = 180.0

#: The probe's wall envelope, spelled PER PHASE because that is how httpx
#: applies a timeout: connect may burn its full share on a blackholed host and
#: read its own on a stalled body, so the pathological total is ~connect +
#: read (~5.5s). The single ``5.0`` this replaces was two full phases (~10s),
#: and review round 1 measured 5.04s on the connect phase alone. A healthy API
#: answers in tens of milliseconds and never notices the difference.
_CONNECT_TIMEOUT_S = 3.0
_READ_TIMEOUT_S = 2.5
_PROBE_TIMEOUT = httpx.Timeout(
    connect=_CONNECT_TIMEOUT_S,
    read=_READ_TIMEOUT_S,
    write=_READ_TIMEOUT_S,
    pool=_READ_TIMEOUT_S,
)

#: Stable phrases carried by every text this module appends, matched
#: case-insensitively by :func:`_carries_family_text`.
#:
#: DELIBERATELY NOT THE SENTENCE OPENERS (agent review round 1, R1-2). The
#: guard used to key on "You're out of credits" / "You haven't verified your
#: email yet", which are ordinary English a provider's OWN 402 body can carry:
#: ``str(ProviderError(402, "You're out of credits. Add funds to continue."))``
#: made the substring guard read the provider's words as this module's remedy
#: and suppress the append. Each phrase below appears in every branch text this
#: module can produce and in nothing a provider writes about a refusal: "top up
#: in the radient console" (the verified top-up line, and the neutral text's
#: own second line) and "start using local operator for free" (the verification
#: head that pending/expired/none share). The line the old runtime wrote
#: ("Radient: …") is not a marker either: it can only reach this module through
#: a text that is already ours, and "Radient: " alone is a prefix ordinary tool
#: errors carry. Deliberately not the URLs: the payload's may be any https page
#: the backend supplies, so a URL-based marker would stop matching a line built
#: from one of them (review round 1, R2 on the previous revision).
_FAMILY_MARKERS = (
    "top up in the radient console",
    "start using local operator for free",
)


def _carries_family_text(text: str) -> bool:
    """Whether ``text`` already carries a text this module appended.

    Case-insensitive, so the two spellings of the console sentence ("Top up"
    opening a line, "top up" mid-sentence in the neutral text) are one marker;
    the match lives here rather than at each call site so the guard, the
    scheduler and the append cannot drift. Never raises — nothing on this path
    may.
    """
    lowered = text.lower()
    return any(marker in lowered for marker in _FAMILY_MARKERS)


@dataclass(frozen=True)
class FirstTopupFacts:
    """The optional ``first_topup`` object, parsed tolerantly.

    ``bonus_received`` is None (not stated) rather than False when the payload
    omits or mangles it: only a definite ``false`` earns the bonus line.
    """

    bonus_amount: float | None = None
    minimum_purchase: float | None = None
    bonus_received: bool | None = None
    topup_url: str | None = None


@dataclass(frozen=True)
class VerificationFacts:
    """The ``verification`` object, parsed tolerantly.

    Every field is optional because the payload is external input: a value of
    the wrong shape is dropped to ``None`` rather than rejected, and the
    sentence builder treats ``None`` as "not stated" — never as its absence
    meaning the opposite claim.
    """

    email_verified: bool | None = None
    signup_grant: str | None = None
    grant_amount: float | None = None
    claim_url: str | None = None
    first_topup: FirstTopupFacts | None = None


@dataclass(frozen=True)
class RecoveryFacts:
    """What the recovery text is built from.

    ``signed_in`` is tri-state for what it says about the PROBE, not for two
    different texts: ``False`` (no stored credential) and ``None`` (the
    credential store could not be read) both render the neutral text, because
    a 402 proves a credential was spent somewhere this process may not see —
    an environment key or a runtime override — and "no Radient account is
    signed in" would be a claim the evidence cannot support (the frozen
    contract words the signed-out case as neutral too). Only ``True``
    continues to the verification branches.
    """

    signed_in: bool | None
    verification: VerificationFacts | None = None


#: The process cache: ``(expires_at_monotonic, facts)``. One slot, because the
#: question it answers — "does the account behind this process's Radient
#: credential have an unclaimed grant?" — has one answer per process in every
#: deployment this hint serves.
_cache_lock = threading.Lock()
_cache: tuple[float, RecoveryFacts] | None = None


def reset_recovery_cache() -> None:
    """Drop the process cache. A test seam, and the fix if config is redirected."""
    global _cache
    with _cache_lock:
        _cache = None


def _cached_facts(now: float) -> RecoveryFacts | None:
    with _cache_lock:
        entry = _cache
    if entry is None:
        return None
    expires_at, facts = entry
    return facts if now < expires_at else None


def _remember(facts: RecoveryFacts, now: float) -> None:
    global _cache
    with _cache_lock:
        _cache = (now + _TTL_S, facts)


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _https_url(value: Any) -> str | None:
    """``value`` when it is a plain https URL, else None.

    These URLs come off the wire and are PRINTED into a terminal, so the shape
    is checked rather than trusted (agent review round 1, R1-3). Rejected:
    anything but ``https`` with a host; whitespace; control characters — C0
    AND C1, the latter reaching a terminal as escape/CSI introducers; FORMAT
    characters (``Cf``: bidi overrides like U+202E and zero-width joiners that
    can reorder or hide what a reader sees); and any netloc carrying
    ``userinfo``, because ``https://console.radienthq.com@evil.example/p``
    reads as the console host while addressing ``evil.example``. A rejected
    value falls back to the known console page at the call site.
    """
    if not isinstance(value, str) or not value:
        return None
    for char in value:
        if char.isspace() or unicodedata.category(char) in ("Cc", "Cf"):
            return None
    try:
        parts = urlsplit(value)
    except ValueError:
        return None
    if parts.scheme != "https" or not parts.netloc:
        return None
    if "@" in parts.netloc:
        return None
    return value


def _number(value: Any) -> float | None:
    """A finite non-bool number, else None (a bool is a number to Python, not to us)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _money(value: float | None) -> str | None:
    """``5.0`` as ``$5`` and ``2.5`` as ``$2.50``; None for an unusable amount.

    Whole numbers drop the cents, matching the verification email's subject so
    the figure reads the same in the inbox and in the terminal.
    """
    if value is None or value <= 0:
        return None
    return f"${int(value)}" if value.is_integer() else f"${value:,.2f}"


def parse_first_topup(value: Any) -> FirstTopupFacts | None:
    """Parse the optional ``first_topup`` object; ``None`` when it is absent."""
    if not isinstance(value, Mapping):
        return None
    received = value.get("bonus_received")
    return FirstTopupFacts(
        bonus_amount=_number(value.get("bonus_amount")),
        minimum_purchase=_number(value.get("minimum_purchase")),
        bonus_received=received if isinstance(received, bool) else None,
        topup_url=_https_url(value.get("topup_url")),
    )


def parse_verification(value: Any) -> VerificationFacts | None:
    """Parse the ``verification`` object; ``None`` when it is absent.

    Absence (an older backend) is distinct from emptiness on purpose: the
    caller renders the neutral text for ``None`` and a branch-specific sentence
    for an object, and the two must stay distinguishable so "not stated" never
    reads as "no grant".
    """
    if not isinstance(value, Mapping):
        return None
    grant = value.get("signup_grant")
    if not isinstance(grant, str) or grant not in ("claimed", "pending", "expired", "none"):
        grant = None
    verified = value.get("email_verified")
    if not isinstance(verified, bool):
        verified = None
    return VerificationFacts(
        email_verified=verified,
        signup_grant=grant,
        grant_amount=_number(value.get("grant_amount")),
        claim_url=_https_url(value.get("claim_url")),
        first_topup=parse_first_topup(value.get("first_topup")),
    )


def _first_topup_line(first_topup: FirstTopupFacts | None) -> str | None:
    """The first-top-up bonus sentence, only while the bonus is still on offer.

    No block (older backend), ``bonus_received`` true or not stated, or a
    figure that did not parse, all produce NO line: the bonus is an offer, and
    an offer that cannot be quoted exactly must not be made.
    """
    if first_topup is None or first_topup.bonus_received is not False:
        return None
    bonus = _money(first_topup.bonus_amount)
    minimum = _money(first_topup.minimum_purchase)
    if bonus is None or minimum is None:
        return None
    return f"Get an extra {bonus} free on your first top-up of {minimum} or more."


def _neutral_text() -> str:
    """The out-of-credits text that is true of EVERY account state.

    For an account this process could not read (signed out, offline, ``/me``
    failed, an older backend with no ``verification``). Both remedies are named
    behind their own condition, so a verified user is not told to verify and an
    unverified one is not told their credits are already spent.
    """
    return (
        "You're out of credits. If you haven't verified your email yet, verify it "
        f"to claim your free credits: {CLAIM_URL}\n"
        f"Otherwise, top up in the Radient console: {TOPUP_URL}"
    )


AccountState = Literal["verified", "unverified", "unreadable"]


def account_state(facts: RecoveryFacts) -> AccountState:
    """The ONE classification of what ``facts`` proved about the account.

    WHY A SEPARATE FUNCTION: :func:`recovery_line` (the sentence) and the
    pre-emptive quota notice (``quota_notice``: its state and which buttons it
    offers) must agree on which of the three situations an account is in. Two
    copies of this ladder is how a notice would say "verify your email" over a
    sentence that says "top up". ``recovery_line`` calls this; the notice calls
    this; neither re-derives it.

    - ``verified`` — a ``claimed`` grant, or ``email_verified`` true: the free
      credits are spent (or were never owed), so the remedy is a top-up.
    - ``unverified`` — ``pending`` / ``expired``, or any grant with an explicit
      ``email_verified: false``: free credits are WAITING behind verification.
    - ``unreadable`` — everything else (no probe answer, an older backend with
      no ``verification``, an unrecognised shape): a claim about the account
      would be a guess, so callers render the neutral text.
    """
    verification = facts.verification
    if verification is None:
        return "unreadable"
    if verification.signup_grant == "claimed" or verification.email_verified is True:
        return "verified"
    if verification.signup_grant in ("pending", "expired") or verification.email_verified is False:
        return "unverified"
    return "unreadable"


def recovery_line(facts: RecoveryFacts) -> str:
    """The user-facing text for ``facts``. Never empty, never raises.

    Decided by :func:`account_state`, each branch stating only what the payload
    proved:

    1. **verified** — the free credits are spent: the top-up link, plus the
       first-top-up bonus line while the bonus is unclaimed.
    2. **unverified** — free credits are WAITING behind verification, so that
       comes before any top-up. ``pending`` points at the inbox; ``expired`` at
       requesting a new link, because the mail itself is dead; anything else
       just opens the page.
    3. **unreadable** — the neutral text, which makes no claim about the
       account. That covers signed out too: the contract words the signed-out
       case as neutral, and a 402 proves a credential WAS spent somewhere this
       process may not see (an env key, a runtime override), so "you are not
       signed in" would be a claim it cannot support.
    """
    verification = facts.verification
    state = account_state(facts)
    if verification is not None and state != "unreadable":
        claim = verification.claim_url or CLAIM_URL
        if state == "verified":
            topup = (
                verification.first_topup.topup_url if verification.first_topup else None
            ) or TOPUP_URL
            lines = [f"You're out of credits. Top up in the Radient console: {topup}"]
            bonus = _first_topup_line(verification.first_topup)
            if bonus:
                lines.append(bonus)
            return "\n".join(lines)
        amount = _money(verification.grant_amount)
        credits = f"{amount} in free credits" if amount else "your free credits"
        head = (
            "You haven't verified your email yet. "
            f"Verify to claim {credits} and start using Local Operator for free."
        )
        if verification.signup_grant == "pending":
            action = f"Check your inbox for the Radient verification email, or open {claim}"
        elif verification.signup_grant == "expired":
            action = f"Your verification link has expired. Request a new one at {claim}"
        else:
            action = f"Open {claim} to verify your email."
        return f"{head}\n{action}"
    return _neutral_text()


def _resolve_token(store: AuthStore) -> str | None:
    """The bearer to probe ``/me`` with: the stored Radient credential, else None.

    OAuth rows win because the endpoint is the account's own surface and the
    access token is what ``fetch_radient_balance`` is handed for it; the
    store-first ``RADIENT_API_KEY`` row and the environment are the same
    fallback legs the registry readers use, so an API-key operator is probed
    with the credential they actually hold rather than told to sign in.
    """
    for row in reversed(list(store.list_credentials("radient"))):
        data = row.data if isinstance(row.data, Mapping) else {}
        token = data.get("access") if row.credential_type == "oauth" else data.get("key")
        if isinstance(token, str) and token:
            return token
    from local_operator.providers.registry import provider_secret_value

    stored = provider_secret_value("RADIENT_API_KEY")
    if stored:
        return stored
    return os.environ.get("RADIENT_API_KEY") or None


async def fetch_me_verification(
    client: httpx.AsyncClient, access_token: str
) -> VerificationFacts | None:
    """The verification object read through a caller-supplied client.

    Mirrors ``fetch_radient_balance``'s read pattern (the same URL base, a
    bearer header and ``follow_redirects=False``); every failure — transport,
    non-200, unparseable body, absent object — answers ``None``. Never raises:
    a probe is a read, not a routing decision.
    """
    from local_operator.providers.usage import RADIENT_API_URL

    try:
        response = await client.get(
            RADIENT_API_URL + "/me",
            headers=_bearer(access_token),
            timeout=_PROBE_TIMEOUT,
            follow_redirects=False,
        )
        if response.status_code != 200:
            return None
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    result = payload.get("result") if isinstance(payload, Mapping) else None
    if not isinstance(result, Mapping):
        return None
    return parse_verification(result.get("verification"))


def fetch_me_verification_sync(client: httpx.Client, access_token: str) -> VerificationFacts | None:
    """The synchronous twin of :func:`fetch_me_verification` (headless path)."""
    from local_operator.providers.usage import RADIENT_API_URL

    try:
        response = client.get(
            RADIENT_API_URL + "/me",
            headers=_bearer(access_token),
            timeout=_PROBE_TIMEOUT,
            follow_redirects=False,
        )
        if response.status_code != 200:
            return None
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    result = payload.get("result") if isinstance(payload, Mapping) else None
    if not isinstance(result, Mapping):
        return None
    return parse_verification(result.get("verification"))


async def _probe_verification_async(token: str) -> VerificationFacts | None:
    """One bounded probe, over a client this module owns. The network seam tests arm."""
    async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT, follow_redirects=False) as client:
        return await fetch_me_verification(client, token)


def _probe_verification_sync(token: str) -> VerificationFacts | None:
    """One bounded probe, over a client this module owns. The network seam tests arm."""
    with httpx.Client(timeout=_PROBE_TIMEOUT, follow_redirects=False) as client:
        return fetch_me_verification_sync(client, token)


async def get_recovery_facts(
    *, store: AuthStore | None = None, force_refresh: bool = False
) -> RecoveryFacts:
    """The cached-or-probed facts for this process's Radient account.

    ``store`` is a test seam and a caller's chance to reuse a store it already
    holds; production callers pass nothing and the process's shared store is
    read. ``force_refresh`` bypasses the process cache and re-probes, because
    the cache is the WRONG answer for the one caller that asks right after the
    account's state changed: a user who just clicked "I verified" must not be
    re-served the pre-verification facts for up to the TTL (R1-X1). The fresh
    result is remembered under the normal TTL, so the bypass also refreshes
    the cache for every reader after it. Never raises — see the module
    docstring.
    """
    now = time.monotonic()
    if not force_refresh:
        cached = _cached_facts(now)
        if cached is not None:
            return cached
    try:
        if store is None:
            from local_operator.providers.auth_store import shared_auth_store

            store = shared_auth_store()
        token = _resolve_token(store)
    except Exception:  # noqa: BLE001 — a hint must never raise; signed_in=None below
        logger.debug("Radient recovery: credential read failed", exc_info=True)
        return RecoveryFacts(signed_in=None)
    if token is None:
        return RecoveryFacts(signed_in=False)
    try:
        verification = await _probe_verification_async(token)
    except Exception:  # noqa: BLE001 — the probe swallows, but the contract is absolute
        logger.debug("Radient recovery: probe failed", exc_info=True)
        verification = None
    facts = RecoveryFacts(signed_in=True, verification=verification)
    _remember(facts, now)
    return facts


def get_recovery_facts_sync(
    *, store: AuthStore | None = None, force_refresh: bool = False
) -> RecoveryFacts:
    """The bounded synchronous twin — the HEADLESS-ONLY arm.

    ``force_refresh`` mirrors the async arm (see there): bypass the cache and
    re-probe. The twins share one cache, so a divergence here would be a trap.

    See the module docstring's access patterns: this is the one entry point
    that blocks on a cold cache. It is kept for the one-shot headless
    renderer; the TUI must use :func:`get_recovery_facts` (awaited) or
    :func:`usage_limit_recovery_line_cached` (never blocks).
    """
    now = time.monotonic()
    if not force_refresh:
        cached = _cached_facts(now)
        if cached is not None:
            return cached
    try:
        if store is None:
            from local_operator.providers.auth_store import shared_auth_store

            store = shared_auth_store()
        token = _resolve_token(store)
    except Exception:  # noqa: BLE001 — a hint must never raise; signed_in=None below
        logger.debug("Radient recovery: credential read failed", exc_info=True)
        return RecoveryFacts(signed_in=None)
    if token is None:
        return RecoveryFacts(signed_in=False)
    try:
        verification = _probe_verification_sync(token)
    except Exception:  # noqa: BLE001 — the probe swallows, but the contract is absolute
        logger.debug("Radient recovery: probe failed", exc_info=True)
        verification = None
    facts = RecoveryFacts(signed_in=True, verification=verification)
    _remember(facts, now)
    return facts


async def usage_limit_recovery_line(*, store: AuthStore | None = None) -> str:
    """The recovery sentence to append for this process's Radient account."""
    return recovery_line(await get_recovery_facts(store=store))


def usage_limit_recovery_line_sync(*, store: AuthStore | None = None) -> str:
    """The BLOCKING twin of :func:`usage_limit_recovery_line` (see the docs)."""
    return recovery_line(get_recovery_facts_sync(store=store))


def recovery_facts_cached() -> RecoveryFacts | None:
    """The process cache's facts when a probe has already answered, else ``None``.

    The FACTS-level cached arm, for a caller that must not probe AND must not
    reduce the answer to one sentence — the pre-emptive quota notice
    (``tui/quota_notice``) needs the facts themselves, because the shared
    classifier (``account_state``) names ``unverified`` from them and PR1's
    lesson is that classifying from sentence text is how two spellings of one
    state drift apart. ``None`` means exactly one thing: only a probe could
    decide, and the caller renders its own neutral answer. The "no stored
    credential" read is NOT consulted here — unlike
    :func:`usage_limit_recovery_line_cached`, this returns only what a probe
    has actually established.
    """
    return _cached_facts(time.monotonic())


def usage_limit_recovery_line_cached(*, store: AuthStore | None = None) -> str | None:
    """The sentence when this process ALREADY knows it, else ``None``.

    Cache-only by construction: it never probes and never blocks. The answers
    knowable without the wire are still computed — a warm cache, and the
    network-free "no stored credential" read (a store that cannot be read
    degrades to the neutral text, the same answer :func:`get_recovery_facts`
    gives it). ``None`` means exactly one thing: only a probe could decide, so
    a caller that can kick one off the event loop should (see
    :func:`usage_limit_recovery_pending`), and a caller that cannot renders
    the text unextended.
    """
    now = time.monotonic()
    cached = _cached_facts(now)
    if cached is not None:
        return recovery_line(cached)
    try:
        if store is None:
            from local_operator.providers.auth_store import shared_auth_store

            store = shared_auth_store()
        token = _resolve_token(store)
    except Exception:  # noqa: BLE001 — a hint must never raise; see the module docstring
        logger.debug("Radient recovery: credential read failed", exc_info=True)
        return recovery_line(RecoveryFacts(signed_in=None))
    if token is None:
        return recovery_line(RecoveryFacts(signed_in=False))
    return None


def usage_limit_recovery_pending(
    rendered_error: str, provider: str | None, *, store: AuthStore | None = None
) -> bool:
    """True when a probe could still add a sentence to ``rendered_error``.

    The TUI's sync handler asks this to decide whether the off-loop worker is
    worth scheduling: the trigger must apply (quota-labelled text, Radient
    provider), no family line may already be present, and the process must not
    already know the answer — a warm cache or a missing credential is
    knowable locally and renders through the cached arm instead.
    """
    if not usage_limit_recovery_applies(rendered_error, provider):
        return False
    if _carries_family_text(rendered_error):
        return False
    return usage_limit_recovery_line_cached(store=store) is None


def usage_limit_recovery_applies(rendered_error: str, provider: str | None) -> bool:
    """The trigger: an out-of-credits (HTTP 402) refusal on the Radient provider.

    The function keeps the name it shipped under (``usage_limit``) so its three
    call sites, the TUI, the session journal and the headless renderer, did not
    change shape; what it gates on is narrower than the name says. A 429 is a
    rate limit whose remedy is waiting, and advice about verification or
    top-ups would be wrong for it, so only the 402 label qualifies.

    The provider is normalized through ``credential_provider_id`` — the same
    translation every credential lookup uses — so a request that ran as the
    ``radient-key`` login flavour (which stores under ``radient``) is covered,
    while an unknown or unrelated provider is not.
    """
    from local_operator.providers.failover import is_rendered_out_of_credits_error

    if not is_rendered_out_of_credits_error(rendered_error):
        return False
    try:
        from local_operator.providers.registry import credential_provider_id

        return credential_provider_id(provider or "") == "radient"
    except Exception:  # noqa: BLE001 — an unknown provider must not raise here
        return False


def append_recovery_line_once(text: str, line: str) -> str:
    """``text`` with ``line`` appended, at most once. Never raises.

    Two guards: the exact line (a retried render of the same cached facts) and
    the family markers (a DIFFERENT branch's line already present, e.g. the
    grant expired between two attempts), so the remedy never stacks.
    """
    if not line:
        return text
    if line in text or _carries_family_text(text):
        return text
    if not text:
        return line
    return f"{text}\n{line}"


async def append_usage_limit_recovery_async(
    rendered_error: str, provider: str | None, *, store: AuthStore | None = None
) -> str:
    """``rendered_error`` plus the Radient usage-limit recovery, or unchanged.

    The async entry point, for display sites that can await (the
    session-incident journal). The probe is awaited rather than blocking, so
    its bounded wait yields to the app instead of freezing it.
    """
    if not usage_limit_recovery_applies(rendered_error, provider):
        return rendered_error
    if _carries_family_text(rendered_error):
        return rendered_error
    line = await usage_limit_recovery_line(store=store)
    return append_recovery_line_once(rendered_error, line)


def append_usage_limit_recovery_cached(
    rendered_error: str, provider: str | None, *, store: AuthStore | None = None
) -> str:
    """``rendered_error`` plus the Radient remedy when it is ALREADY known.

    The non-blocking twin, for sync callers that run on an event loop and may
    not block it (the TUI's handler; review round 1, R1). Cold and
    probe-decided: the text is returned unchanged, and a caller that can
    schedule one runs :func:`usage_limit_recovery_line` off-loop to complete
    the render — see ``OperatorApp._schedule_recovery_notice``.
    """
    if not usage_limit_recovery_applies(rendered_error, provider):
        return rendered_error
    if _carries_family_text(rendered_error):
        return rendered_error
    line = usage_limit_recovery_line_cached(store=store)
    return append_recovery_line_once(rendered_error, line or "")


def append_usage_limit_recovery(
    rendered_error: str, provider: str | None, *, store: AuthStore | None = None
) -> str:
    """The BOUNDED blocking twin, for the one surface that cannot await.

    That surface is the headless renderer: its one-shot process exits with
    the line it prints, so a cache-only answer would render the neutral
    fallback forever, and it cannot await. Everything loop-side must use the
    awaited or cached variants instead (see the module docstring's access
    patterns). Bounded like every path here: the worst case is one probe
    within the ~5.5s envelope, and only when a Radient quota error is actually
    being rendered — the process cache keeps every repeat free of it.
    """
    if not usage_limit_recovery_applies(rendered_error, provider):
        return rendered_error
    if _carries_family_text(rendered_error):
        return rendered_error
    line = usage_limit_recovery_line_sync(store=store)
    return append_recovery_line_once(rendered_error, line)
