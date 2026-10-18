"""The artifact lanes' failure mapping — one copy, deliberately not an STT import.

Why a copy rather than ``stt/errors``'s mapper: that module classifies
TRANSCRIPTION failures (its rung vocabulary is ``AudioPath``, its details are
built for the speech surfaces) and it carries a body-scrubbing contract tuned
to those routes. The artifact rules diverge the moment either lane moves: this
module owns exactly one shared primitive — ``clients/_http``'s payload
scrubbers — and a small, closed reason-class vocabulary the attempt records
speak.

The reason classes exist because the tool's error report and the PR/QA
evidence both need to GROUP failures ("every rung refused for credits") without
parsing prose, and prose is the one thing this codebase keeps rewording.
"""

from __future__ import annotations

from collections.abc import Mapping

from local_operator.clients._http import APIError

#: The closed reason-class vocabulary. Keep in sync with :func:`failure_reason_class`
#: and the docstring on ``JobAttempt`` in the package ``__init__``.
REASON_CLASSES = frozenset(
    {
        "insufficient_balance",  # the rung was SKIPPED: an affordability probe
        "insufficient_credits",  # 402 from the provider
        "unauthorized",  # 401/403
        "not_found",  # 404 (model or job)
        "rate_limited",  # 429
        "upstream",  # 5xx
        "network",  # transport failure, no response (timeouts: "timeout")
        "timeout",  # a wait or request exceeded its bound
        # A RETRYABLE refusal that is not a credential or a balance problem —
        # the audio contract's submit-time admission bound (HTTP 429,
        # ``audio_rate_limited``) names itself ``transient`` so a caller knows
        # to retry with backoff while the SURFACE still says "busy", not
        # "quota". The wire code rides ``JobAttempt.error_type`` beside it.
        "transient",
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
        # The audio tools' own failure words (frozen server contract v1): a
        # COMPLETED status carrying ``error_type`` is the audio lane's failure
        # shape, branched on as a pair — never on the status alone. The
        # admission/refusal pair: ``audio_rate_limited`` is both the submit
        # 429 (retried with backoff, then surfaced) and the admitted-job
        # settlement (zero cost, no automatic retry).
        "audio_failed",
        "audio_rate_limited",
        "unknown",
    }
)


def error_type_of(exc: BaseException) -> str | None:
    """The WIRE's structured error code for a rung failure, when it sent one.

    ``reason_class`` answers "how do I retry this?" (a small closed set);
    ``error_type`` answers "what did the provider actually say?" and travels
    verbatim so a surface can branch on the server's own token — the audio
    contract's ``audio_rate_limited`` arriving on a ``transient`` failure is
    the case that made the two distinct. The details map wins when it carries
    one (a raising site that mapped a status onto a wire code), else the
    ``code`` attribute. Never raises.
    """
    details = getattr(exc, "details", None)
    if isinstance(details, Mapping):
        value = details.get("error_type")
        if isinstance(value, str) and value.strip():
            return value.strip()
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code.strip():
        return code.strip()
    return None


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
