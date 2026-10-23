"""ElevenLabs' output format: prefer 192 kbps, drop to 128 once on a plan gate.

WHY THIS IS A MODULE, NOT A CONSTANT IN ONE CALLER. The daemon reaches
ElevenLabs' speech endpoint from two places — the audio tools' BYOK rung
(:mod:`local_operator.audiogen.rungs`) and the speech daemon's own BYO leg
(:mod:`local_operator.tts.clients`) — and both must send the same format and
make the same decision when the vendor refuses it. Two copies of the predicate
would be two answers to "was that a plan refusal?", which is exactly the
surface-drift this codebase vendors the voicing map to prevent.

WHY 192 IS A PREFERENCE AND NOT A REQUIREMENT. ``mp3_44100_192`` is gated
behind ElevenLabs' Creator plan (PCM/WAV at 44.1 kHz behind Pro); 128 kbps is
served to every account. 128 brick-walls the top octave at roughly 16 kHz,
which is what makes an effect sound tinny, so it should not be the default for
accounts that can hear the difference — but a generation that FAILS because we
asked for better audio is strictly worse than the 128 kbps audio it replaced.
Hence: ask for 192, and when — and only when — the vendor refuses the format on
plan grounds, retry the same call once at 128.

THE RETRY IS NOT A RETRY LOOP. Only a 400/403 whose error body names the
``output_format`` parameter (or, with ``param`` absent, carries the vendor's
entitlement class and prose naming a plan) is treated as a plan gate. A 401,
429, 5xx or any other 4xx is a different fault; a second call would spend the
vendor's rate budget and still fail, hiding the real cause behind an identical
error.

THE MEMO IS A PERFORMANCE MEMORY, NOT A POLICY. Once a key has been measured
capped, later calls skip the doomed attempt. It is keyed by a SHA-256 of the
key plus the endpoint and model, so one surface's refusal cannot downgrade
another, and it expires after six hours because a plan upgrade is invisible to
us. It is written only when the retry SUCCEEDED and the refusal was
entitlement-shaped: a retry that failed proves nothing about the plan, and a
bare ``invalid_parameter`` speaks about the parameter, not the tier.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from local_operator.clients._http import APIError

logger = logging.getLogger(__name__)

#: The format worth asking for: audibly better than the vendor's default and
#: available to Creator-tier accounts.
PREFERRED_OUTPUT_FORMAT = "mp3_44100_192"

#: The format every account is served. This is also the vendor's own default;
#: it is named here because the fallback must not silently become "whatever the
#: vendor picks" if it ever changes.
FALLBACK_OUTPUT_FORMAT = "mp3_44100_128"

#: The query parameter the vendor names in a plan refusal.
OUTPUT_FORMAT_PARAM = "output_format"

#: How long a measured cap is remembered. Six hours is long enough to keep a
#: capped key from spending a doomed call per generation and short enough that
#: an upgrade bought today is honoured today.
CAP_TTL_S = 6 * 60 * 60

#: A bound on the memo: this process is long-lived and may see many keys (a
#: hosted or multi-tenant deployment), so an unbounded dict would be a slow
#: leak. Entries are dropped expired-first when the bound is reached.
CAP_MAX_KEYS = 64

#: The vendor's classes for "your plan does not include this".
_ENTITLEMENT_CLASSES = frozenset({"authorization_error", "feature_not_available"})

#: Words that name an entitlement. Word-bounded on purpose: "explanation" holds
#: no "plan", and "planned"/"planet" must not read as "plan". The plural is
#: allowed because the vendor writes "plans", "subscriptions", "entitlements".
_ENTITLEMENT_WORD = re.compile(r"\b(plan|tier|subscription|upgrade|entitlement)s?\b", re.IGNORECASE)

#: key fingerprint -> monotonic expiry. Module-level because the audio tools and
#: the TTS daemon are separate call sites inside one long-lived process.
_caps: dict[str, float] = {}

T = TypeVar("T")


def _error_detail(exc: APIError) -> dict[str, Any]:
    """The vendor's ``detail`` object from the carried body, or ``{}``.

    ElevenLabs reports a refusal as ``{"detail": {type, code, status, message,
    param}}``; :func:`local_operator.clients._http.error_payload` deliberately
    reads only the hub's envelope, so the structured half is read here, from
    the scrubbed body the :class:`APIError` already carries. A body that is not
    JSON — an intermediary's HTML page — is not an error on the error path.
    """
    if not exc.body:
        return {}
    try:
        parsed = json.loads(exc.body)
    except (TypeError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    detail = parsed.get("detail")
    return detail if isinstance(detail, dict) else {}


def _classes(detail: dict[str, Any]) -> set[str]:
    return {
        str(detail.get(key) or "").strip().lower()
        for key in ("type", "code", "status")
        if isinstance(detail.get(key), str)
    }


def is_plan_refusal(exc: APIError) -> bool:
    """Whether ``exc`` is the vendor refusing the REQUESTED format on plan grounds.

    The exact predicate, the same shape as the hub's client
    (``agent-server internal/clients/elevenlabs.go``) so both halves of the
    product retry the same plan refusals. One deliberate narrowing: this side
    does not accept a bare ``invalid_parameter`` on the format, because the
    formats it sends are fixed constants rather than caller input — a
    validation class here would mean a code defect, not a tier.

    * status is 400 or 403 — never 401 (the key), 429 or 5xx;
    * ``detail.param`` is ``output_format``; or, when ``param`` is absent, the
      class is ``authorization_error``/``feature_not_available`` AND the prose
      names an entitlement. The second arm is a safety net: the vendor
      documents ``param`` as the offending parameter but a plan gate can arrive
      without it, and without this arm a free-tier key would get no audio where
      the old code always got 128.

    A 403 naming ``enable_logging``, or a 400 naming ``text``, is not this
    refusal: retrying it at 128 kbps would change the format and fix nothing.
    """
    if exc.status_code not in (400, 403):
        return False
    detail = _error_detail(exc)
    if not detail:
        return False
    param = detail.get("param")
    if isinstance(param, str) and param.strip():
        return param.strip().lower() == OUTPUT_FORMAT_PARAM
    if not (_classes(detail) & _ENTITLEMENT_CLASSES):
        return False
    return bool(_ENTITLEMENT_WORD.search(str(detail.get("message") or "")))


def is_plan_cap_learned(exc: APIError) -> bool:
    """Whether a successful fallback proves the plan lacks the preferred format.

    The stricter reading, used before writing the six-hour memo: the class must
    itself be an entitlement one. A bare ``invalid_parameter`` earns its one
    retry (the cost of a false negative is no audio at all) but is a statement
    about the parameter, so it must not suppress 192 for every later request.
    """
    if exc.status_code not in (400, 403):
        return False
    return bool(_classes(_error_detail(exc)) & _ENTITLEMENT_CLASSES)


def cap_key(api_key: str, *, endpoint: str, model: str) -> str:
    """The memo key: a key fingerprint scoped to the capability that refused.

    The key itself never appears in the memo or in a log line; the digest keeps
    two accounts on one process apart, and ``endpoint``/``model`` keep a
    sound-effect refusal from downgrading speech for the same key.
    """
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return f"{digest}|{endpoint}|{model}"


def is_capped(key: str) -> bool:
    """Whether ``key`` was measured capped and the measurement is still fresh."""
    expires = _caps.get(key)
    if expires is None:
        return False
    if expires <= time.monotonic():
        _caps.pop(key, None)
        return False
    return True


def remember_cap(key: str) -> None:
    """Record a measured cap, pruning expired entries and bounding the memo."""
    now = time.monotonic()
    expired = [stale for stale, expires in _caps.items() if expires <= now]
    for stale in expired:
        del _caps[stale]
    while len(_caps) >= CAP_MAX_KEYS:
        # The TTL is uniform, so insertion order is expiry order; dropping the
        # first entry evicts the oldest measurement.
        del _caps[next(iter(_caps))]
    _caps[key] = now + CAP_TTL_S


def reset_caps() -> None:
    """Clear the memo — for tests, and for a caller that has replaced a key."""
    _caps.clear()


async def post_with_format_fallback(
    send: Callable[[dict[str, str]], Awaitable[T]],
    *,
    api_key: str,
    endpoint: str,
    model: str,
    params: dict[str, str],
) -> T:
    """Call ``send(params)`` at the preferred format, dropping to 128 once.

    ``send`` performs ONE vendor call with the query parameters it is given and
    raises :class:`APIError` on a refusal, so the retry here is the same
    request at a different format rather than a second request shape.

    If the fallback also fails the ORIGINAL failure is raised (with the
    fallback's error attached as its cause): the first refusal is the one that
    describes what the caller asked for, and a "did the retry help?" question is
    answered by the cause rather than by replacing the diagnosis.

    Concurrency note, accepted rather than fixed: the memo is written after the
    vendor answers, so two concurrent first-time calls on a capped key can each
    spend one doomed attempt before either records the cap. The cost is one
    extra call per in-flight job, not per later job.
    """
    key = cap_key(api_key, endpoint=endpoint, model=model)
    requested = params.get(OUTPUT_FORMAT_PARAM)
    effective = dict(params)
    if requested == PREFERRED_OUTPUT_FORMAT and is_capped(key):
        # This key already refused 192 on this endpoint+model; relearning it on
        # every call would add a doomed vendor round trip to every generation.
        effective[OUTPUT_FORMAT_PARAM] = FALLBACK_OUTPUT_FORMAT

    try:
        return await send(effective)
    except APIError as exc:
        if effective.get(OUTPUT_FORMAT_PARAM) != PREFERRED_OUTPUT_FORMAT:
            raise
        if not is_plan_refusal(exc):
            raise
        retry = {**effective, OUTPUT_FORMAT_PARAM: FALLBACK_OUTPUT_FORMAT}
        try:
            result = await send(retry)
        except APIError as fallback_exc:
            raise exc from fallback_exc

        # Past this point the fallback answered: 192 is refused and 128 works.
        # Only an entitlement-shaped refusal is a fact about the plan worth
        # remembering; see is_plan_cap_learned.
        recorded = is_plan_cap_learned(exc)
        if recorded:
            remember_cap(key)
        # One structured line per downgrade so "why is this key serving 128 for
        # six hours" has an answer short of re-reading this module.
        # Deliberately absent: the key, its fingerprint, and the vendor's prose.
        logger.warning(
            "elevenlabs output format downgraded endpoint=%s model=%s requested=%s "
            "used=%s cap_recorded=%s status=%s class=%s",
            endpoint,
            model,
            PREFERRED_OUTPUT_FORMAT,
            FALLBACK_OUTPUT_FORMAT,
            recorded,
            exc.status_code,
            ",".join(sorted(_classes(_error_detail(exc)))),
        )
        return result


__all__ = [
    "CAP_MAX_KEYS",
    "CAP_TTL_S",
    "FALLBACK_OUTPUT_FORMAT",
    "OUTPUT_FORMAT_PARAM",
    "PREFERRED_OUTPUT_FORMAT",
    "cap_key",
    "is_capped",
    "is_plan_cap_learned",
    "is_plan_refusal",
    "post_with_format_fallback",
    "remember_cap",
    "reset_caps",
]
