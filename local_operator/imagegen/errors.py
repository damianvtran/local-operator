"""The image lanes's failure mapping — its own copy, deliberately not an STT import.

Why a copy rather than ``stt/errors``'s mapper: that module classifies
TRANSCRIPTION failures (its rung vocabulary is ``AudioPath``, its details are
built for the speech surfaces) and it carries a body-scrubbing contract tuned
to those routes. Image rules diverge the moment either lane moves: this module
owns exactly one shared primitive — ``clients/_http``'s payload scrubbers —
and a small, closed reason-class vocabulary the attempt records speak.

The reason classes exist because the tool's error report and the PR/QA
evidence both need to GROUP failures ("every rung refused for credits") without
parsing prose, and prose is the one thing this codebase keeps rewording.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

from local_operator.clients._http import (
    APIError,
    error_payload,
    scrub_details,
    scrub_secrets,
)

#: The closed reason-class vocabulary. Keep in sync with :func:`failure_reason_class`
#: and the docstring on ``ImageAttempt`` in the package ``__init__``.
REASON_CLASSES = frozenset(
    {
        "insufficient_balance",  # the rung was SKIPPED: Radient affordability probe
        "insufficient_credits",  # 402 from the provider
        "unauthorized",  # 401/403
        "not_found",  # 404 (model or job)
        "rate_limited",  # 429
        "upstream",  # 5xx
        "network",  # transport failure, no response (timeouts: "timeout")
        "timeout",  # a wait or request exceeded its bound
        "refused",  # any other 4xx
        "cancelled",  # the PROVIDER reported the job CANCELLED
        "invalid_response",  # unparseable / missing fields in a 2xx payload
        "unsupported",  # the provider has no route for the request SHAPE (a
        # skip, never a failure of the provider)
        # The Radient hub's own structured codes (``error_type`` on a FAILED
        # status), carried verbatim so consumers switch on the same tokens the
        # wire uses instead of parsing prose.
        "media_rejected",
        "media_failed",
        "media_rate_limited",
        "media_unavailable",
        "unknown",
    }
)


def api_error_from_httpx_response(
    response: "Any",
    *,
    fallback_message: str,
    secrets: Iterable[Optional[str]] = (),
) -> APIError:
    """Build an :class:`APIError` from a failed ``httpx`` response.

    The httpx twin of the requests-side builder, and the same shape ``stt``
    uses: designed ``error`` string, status/code/details, mask-on-the-way-in —
    plus the SCRUBBED raw body on :attr:`APIError.body`, because an upstream's
    own sentence is the whole diagnostic for a refusal that carries no
    designed vocabulary, and dropping it would make those refusals read "with
    no body" while the data was in hand.
    """
    raw = response.content.decode(errors="replace")
    message, code, details = error_payload(raw)
    return APIError(
        scrub_secrets(message or f"{fallback_message} (HTTP {response.status_code})", secrets),
        status_code=response.status_code,
        code=code,
        details=scrub_details(details, secrets),
        body=scrub_secrets(raw, secrets) if raw.strip() else None,
    )


def failure_reason_class(exc: BaseException) -> str:
    """The closed token for HOW a rung failed. See :data:`REASON_CLASSES`.

    The precedence is: an explicit ``code`` the raising site stamped (the
    transport clients stamp ``timeout``/``network``/``cancelled``/
    ``invalid_response`` where only they can tell), then the status family,
    then ``unknown``. Never raises — a classification that cannot answer must
    not take down the failure path it is classifying.
    """
    if isinstance(exc, APIError):
        code = (exc.code or "").strip().lower()
        if code in REASON_CLASSES:
            return code
        status = exc.status_code
        if status == 402:
            return "insufficient_credits"
        if status in (401, 403):
            return "unauthorized"
        if status == 404:
            return "not_found"
        if status == 429:
            return "rate_limited"
        if status is not None and status >= 500:
            return "upstream"
        if status is None:
            return "network"
        return "refused"
    if isinstance(exc, TimeoutError):  # incl. asyncio.TimeoutError on 3.11+
        return "timeout"
    return "unknown"
