"""Shared HTTP error helpers for the API clients in this package.

Every client in this package wraps its ``requests`` calls in the same error
handling shape: catch ``RequestException``, pull the response body out of the
exception, and interpolate it into a ``RuntimeError`` message. The helpers here
are that extraction, in one place, so the behaviour is defined once instead of
being re-derived at every call site.

Two shapes exist on purpose:

* :func:`response_body` is the legacy shape — prose for a human, with the body
  interpolated into it. Call sites that predate machine-readable error codes
  keep using it, and their text is unchanged apart from the falsy bug it used to
  carry (see below).
* :class:`APIError` is the structured one — the upstream's own ``error`` prose
  plus its stable ``code`` and ``details`` as attributes. A caller that has to
  CHOOSE A NEXT STEP from a refusal (the desktop app has to distinguish a taken
  name from a moderation hold) cannot switch on a sentence, and the sentence is
  not what the upstream promised to keep stable.
"""

import json
from typing import Any, Dict, Iterable, Optional, Tuple

import requests
from requests.exceptions import RequestException

NO_RESPONSE_BODY = "No response body"
"""Stand-in used in error messages when a failed request has no readable body."""


def response_body(exc: RequestException) -> str:
    """Extract the response body from a failed ``requests`` call.

    Args:
        exc: The exception raised by ``requests``. ``RequestException`` declares
            ``response``, which is ``None`` for failures that never got one
            (connection errors, timeouts, invalid URLs).

    Returns:
        The decoded response body, or :data:`NO_RESPONSE_BODY` when there is none.
    """
    response = exc.response
    # ``is None``, NOT a falsy test. requests.Response.__bool__ returns
    # response.ok, so every 4xx/5xx response is falsy: the falsy form this used
    # to have reported NO_RESPONSE_BODY for exactly the responses that DO carry
    # one, which is every response an error path is handed. The bug was
    # documented here and left in place to keep the message text of the call
    # sites identical; it is fixed now because the publish path has to READ the
    # body (the hub's refusal codes ride in it) and a helper that claims there is
    # none is the wrong foundation for that.
    if response is None:
        return NO_RESPONSE_BODY
    # errors="replace": a body that is not valid UTF-8 (a gzip/binary error page,
    # a latin-1 proxy response) must not turn "report the failure" into a
    # UnicodeDecodeError raised from inside the error handler.
    body = response.content.decode(errors="replace")
    return body if body.strip() else NO_RESPONSE_BODY


REDACTION_MARKER = "[redacted]"
"""What a credential is replaced with in anything about to be surfaced."""


def redact_secrets(text: str, secrets: Iterable[Optional[str]]) -> str:
    """Replace every occurrence of a credential in ``text`` with a marker.

    WHY THIS EXISTS: an upstream is free to reflect the request it received -- the
    ``Authorization`` header included -- into its error body, and an error body is
    exactly what a useful failure message quotes. Quoting it verbatim is how an
    operator's API key ends up in a message the desktop app renders and the log
    keeps, so the body a caller is about to surface goes through here first. The
    body is otherwise kept: for a failure the upstream did not design an error
    vocabulary for, it is the only thing that says what happened.

    Args:
        text: The text about to be surfaced.
        secrets: The credential values to remove. Empty and ``None`` entries are
            skipped, so a caller with no key configured changes nothing.

    Returns:
        The text with every non-empty secret replaced.
    """

    for secret in secrets:
        if secret:
            text = text.replace(secret, REDACTION_MARKER)
    return text


class APIError(RuntimeError):
    """An upstream refusal that keeps its machine-readable half.

    Carries the fields a caller switches on rather than folding them into the
    message:

    Attributes:
        status_code: The HTTP status the upstream answered with, or ``None``
            when the request never got a response (connection error, timeout).
        code: The upstream's stable error code, or ``None`` when it did not send
            one (an intermediary's HTML error page, a 500 with prose only).
        details: The upstream's structured details, always a dict (empty when it
            sent none). Never the raw body.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.details: Dict[str, Any] = dict(details) if details else {}


def error_payload(body: str) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """Pull ``(message, code, details)`` out of an upstream error body.

    The shape is the hub's: ``{"error": "...", "code": "...", "details": {...}}``
    with ``code``/``details`` additive, so a body carrying only ``error`` (every
    older build) still yields its message.

    Tolerant by design: an HTML error page, an empty body, or a ``code`` of the
    wrong type must not turn a refusal into a parse crash on the error path, so
    anything unrecognised yields ``(None, None, {})`` and the caller composes its
    own message from the status.

    Args:
        body: The decoded response body.

    Returns:
        The message, the code and the details the upstream sent.
    """
    try:
        parsed = json.loads(body)
    except (TypeError, ValueError):
        return None, None, {}
    if not isinstance(parsed, dict):
        return None, None, {}

    message = parsed.get("error")
    if not isinstance(message, str) or not message.strip():
        message = None

    code = parsed.get("code")
    if not isinstance(code, str) or not code.strip():
        code = None

    details = parsed.get("details")
    if not isinstance(details, dict):
        details = {}

    return message, code, details


def api_error_from_response(
    response: Optional[requests.Response],
    *,
    fallback_message: str,
    secrets: Iterable[Optional[str]] = (),
) -> APIError:
    """Build an :class:`APIError` from a failed response.

    WHY THE RAW BODY IS NOT SURFACED: the body of a failure the upstream did not
    describe is not written for the person who ends up seeing it — an
    intermediary's HTML error page, a stack trace, a dumped request. Interpolating
    it into a user-facing message (which is what the legacy call sites do) is how
    an upstream's internals, and anything a proxy decided to echo back, reach a
    desktop user and the log. Only the designed ``error`` string is used; the
    status is carried for the caller to report, and the body stays out of both.

    Args:
        response: The failed response, or ``None`` when the request never got one.
        fallback_message: The message to use when the upstream sent no usable one.
        secrets: Credential values to remove from the message. An upstream that
            reflects the request it received into its prose must not be able to
            put the caller's key into a message that gets rendered.

    Returns:
        An :class:`APIError` carrying the status, code and details.
    """
    if response is None:
        return APIError(fallback_message, status_code=None)

    message, code, details = error_payload(response.content.decode(errors="replace"))
    return APIError(
        redact_secrets(message or f"{fallback_message} (HTTP {response.status_code})", secrets),
        status_code=response.status_code,
        code=code,
        details=details,
    )
