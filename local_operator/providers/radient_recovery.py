"""Radient usage-limit recovery: the sentence a quota failure carries.

WHY THIS EXISTS. A Radient request that runs out of credit fails with a bare
``rate limit or quota exceeded (HTTP 402): insufficient credits`` — true, and
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
                     "claim_url": "https://console.radienthq.com/dashboard/verification"}

``email_verified`` is Radient's OWN Turnstile-gated claim state and is NEVER
derived from Google/Microsoft OAuth claims, so it is read from this endpoint
and nowhere else. The object is ABSENT on an older backend and every consumer
must tolerate that: absence degrades to the generic console line, never an
error and never a blank.

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
fails the probe and degrades to the generic line.

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
render the generic fallback forever. Loop-side callers must not use it.

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
never cached, because the moment it changes — the operator logging in — is
exactly the moment a stale "not signed in" would hurt. A successful-login
invalidation is NOT wired in: the worst a stale entry can do is pick a
different sentence for at most ``_TTL_S`` seconds on a path that is already
failing, and reaching into the login flow to invalidate it would couple this
hint to a surface that does not own it.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, typing only
    from local_operator.providers.auth_store import AuthStore

logger = logging.getLogger(__name__)

#: The claim page the frozen contract names. Used whenever the payload does not
#: carry its own ``claim_url`` (a missing field is tolerated, never fatal).
CLAIM_URL = "https://console.radienthq.com/dashboard/verification"

#: The account console, for the generic fallback (fetch failed, older backend,
#: or a state this module has no branch for).
CONSOLE_URL = "https://console.radienthq.com"

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

#: Stable PREFIXES carried by every sentence this module appends: each quota
#: branch line opens with ``Radient: `` and the no-sign-in remedy opens with
#: its own words. Deliberately NOT the console URL: the payload's ``claim_url``
#: may be any URL the backend supplies, so a URL-based marker silently stopped
#: matching a line built from a non-console URL — and a branch flip between
#: retries then stacked a second remedy (review round 1, R2). A retried render
#: carrying a DIFFERENT branch's line must not stack under the first, which
#: exact-line matching alone would miss.
_FAMILY_MARKERS = ("Radient: ", "No Radient account is signed in")

_GENERIC_LINE = f"Radient: check your account and credit balance at {CONSOLE_URL}."

_NO_SIGN_IN_LINE = (
    "No Radient account is signed in — to fix: `/login radient` in the TUI, "
    "`local-operator login radient` from a shell, or Settings → Radient account "
    "in the desktop app."
)


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


@dataclass(frozen=True)
class RecoveryFacts:
    """What the recovery sentence is built from.

    ``signed_in`` is TRI-state because the two failure answers differ in tone:
    a definite ``False`` (no stored credential at all) earns the "no Radient
    account is signed in" remedy, ``True`` continues to the verification
    branches, and ``None`` means the probe could not tell (the store read
    failed) so the sentence must not claim sign-in state either way and falls
    to the generic console line.
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


def parse_verification(value: Any) -> VerificationFacts | None:
    """Parse the ``verification`` object; ``None`` when it is absent.

    Absence (an older backend) is distinct from emptiness on purpose: the
    caller renders the generic console line for ``None`` and a branch-specific
    sentence for an object, and the two must stay distinguishable so
    "not stated" never reads as "no grant".
    """
    if not isinstance(value, Mapping):
        return None
    grant = value.get("signup_grant")
    if not isinstance(grant, str) or grant not in ("claimed", "pending", "expired", "none"):
        grant = None
    amount = value.get("grant_amount")
    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        amount = None
    verified = value.get("email_verified")
    if not isinstance(verified, bool):
        verified = None
    claim_url = value.get("claim_url")
    if not isinstance(claim_url, str) or not claim_url.strip():
        claim_url = None
    return VerificationFacts(
        email_verified=verified,
        signup_grant=grant,
        grant_amount=float(amount) if amount is not None else None,
        claim_url=claim_url,
    )


def recovery_line(facts: RecoveryFacts) -> str:
    """The user-facing sentence for ``facts``. Never empty, never raises.

    The pending branch is the one this feature exists for: it says WHERE the
    remedy is (the verification email) and what it is worth when the account
    states an amount. ``expired``/``none`` deliberately do NOT claim an email
    is waiting — the link is not in flight for either. ``claimed`` and every
    unknown state fall to the generic console line, which is the only claim
    that is true without knowing more.
    """
    if facts.signed_in is False:
        return _NO_SIGN_IN_LINE
    verification = facts.verification
    if verification is not None:
        claim = verification.claim_url or CLAIM_URL
        if verification.signup_grant == "pending":
            if verification.grant_amount is not None:
                credits = f"claim ${verification.grant_amount:,.2f} in free credits"
            else:
                credits = "claim your free credits"
            return (
                "Radient: the free signup credits are unclaimed — check your email "
                f"for the Radient verification link and {credits}: {claim}"
            )
        if verification.signup_grant == "expired":
            return (
                "Radient: the signup grant's claim window has expired — open "
                f"{claim} to check the account or request a new link."
            )
        if verification.signup_grant == "none":
            return (
                "Radient: no signup grant is attached to this account — open "
                f"{claim} to check the account."
            )
    return _GENERIC_LINE


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


async def get_recovery_facts(*, store: AuthStore | None = None) -> RecoveryFacts:
    """The cached-or-probed facts for this process's Radient account.

    ``store`` is a test seam and a caller's chance to reuse a store it already
    holds; production callers pass nothing and the process's shared store is
    read. Never raises — see the module docstring.
    """
    now = time.monotonic()
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


def get_recovery_facts_sync(*, store: AuthStore | None = None) -> RecoveryFacts:
    """The bounded synchronous twin — the HEADLESS-ONLY arm.

    See the module docstring's access patterns: this is the one entry point
    that blocks on a cold cache. It is kept for the one-shot headless
    renderer; the TUI must use :func:`get_recovery_facts` (awaited) or
    :func:`usage_limit_recovery_line_cached` (never blocks).
    """
    now = time.monotonic()
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


def usage_limit_recovery_line_cached(*, store: AuthStore | None = None) -> str | None:
    """The sentence when this process ALREADY knows it, else ``None``.

    Cache-only by construction: it never probes and never blocks. The answers
    knowable without the wire are still computed — a warm cache, and the
    network-free "no stored credential" read (a store that cannot be read
    degrades to the generic line, the same answer :func:`get_recovery_facts`
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
    if any(marker in rendered_error for marker in _FAMILY_MARKERS):
        return False
    return usage_limit_recovery_line_cached(store=store) is None


def usage_limit_recovery_applies(rendered_error: str, provider: str | None) -> bool:
    """The trigger: a usage-limit-classified error on the Radient provider.

    The provider is normalized through ``credential_provider_id`` — the same
    translation every credential lookup uses — so a request that ran as the
    ``radient-key`` login flavour (which stores under ``radient``) is covered,
    while an unknown or unrelated provider is not.
    """
    from local_operator.providers.failover import is_rendered_usage_limit_error

    if not is_rendered_usage_limit_error(rendered_error):
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
    if line in text or any(marker in text for marker in _FAMILY_MARKERS):
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
    if any(marker in rendered_error for marker in _FAMILY_MARKERS):
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
    if any(marker in rendered_error for marker in _FAMILY_MARKERS):
        return rendered_error
    line = usage_limit_recovery_line_cached(store=store)
    return append_recovery_line_once(rendered_error, line or "")


def append_usage_limit_recovery(
    rendered_error: str, provider: str | None, *, store: AuthStore | None = None
) -> str:
    """The BOUNDED blocking twin, for the one surface that cannot await.

    That surface is the headless renderer: its one-shot process exits with
    the line it prints, so a cache-only answer would render the generic
    fallback forever, and it cannot await. Everything loop-side must use the
    awaited or cached variants instead (see the module docstring's access
    patterns). Bounded like every path here: the worst case is one probe
    within the ~5.5s envelope, and only when a Radient quota error is actually
    being rendered — the process cache keeps every repeat free of it.
    """
    if not usage_limit_recovery_applies(rendered_error, provider):
        return rendered_error
    if any(marker in rendered_error for marker in _FAMILY_MARKERS):
        return rendered_error
    line = usage_limit_recovery_line_sync(store=store)
    return append_recovery_line_once(rendered_error, line)
