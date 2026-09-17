"""Shared HTTP error helpers for the API clients in this package.

Every client in this package wraps its ``requests`` calls in the same error
handling shape: catch ``RequestException``, pull the response body out of the
exception, and interpolate it into a ``RuntimeError`` message. The helpers here
are that extraction, in one place, so the behaviour is defined once instead of
being re-derived at every call site.

Two shapes exist on purpose:

* :func:`response_body` is the legacy shape — prose for a human, with the body
  interpolated into it. Call sites that predate machine-readable error codes
  keep using it, and they are UNCHANGED by the fix that made the body readable
  again: every body this module hands out goes through :func:`scrub_secrets`
  first, so a caller that interpolates one cannot leak a credential it never
  knew it was carrying. That is why the scrubber sits on the body's way OUT of
  this module rather than at each of the ~14 sites that surface one.
* :class:`APIError` is the structured one — the upstream's own ``error`` prose
  plus its stable ``code`` and ``details`` as attributes. A caller that has to
  CHOOSE A NEXT STEP from a refusal (the desktop app has to distinguish a taken
  name from a moderation hold) cannot switch on a sentence, and the sentence is
  not what the upstream promised to keep stable.
"""

import json
import re
from typing import Any, Dict, Iterable, Optional, Pattern, Tuple

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
        The decoded, CREDENTIAL-SCRUBBED response body, or
        :data:`NO_RESPONSE_BODY` when there is none.
    """
    # ``is None``, NOT a falsy test. requests.Response.__bool__ returns
    # response.ok, so every 4xx/5xx response is falsy: the falsy form this used
    # to have reported NO_RESPONSE_BODY for exactly the responses that DO carry
    # one, which is every response an error path is handed. The bug was
    # documented here and left in place to keep the message text of the call
    # sites identical; it is fixed now because the publish path has to READ the
    # body (the hub's refusal codes ride in it) and a helper that claims there is
    # none is the wrong foundation for that.
    #
    # Readable-and-scrubbed is ONE step, deliberately: the body became readable
    # in the same change that made the legacy call sites surface bodies they
    # never saw before, and those sites interpolate what they are handed. A
    # helper that returns the raw body here would put every one of their
    # upstreams' echoes -- the ``Authorization`` header included -- into a
    # message the desktop app renders and the log keeps.
    #
    # THE POLICY THIS SETTLES, since the two halves of the same package had
    # opposite ones: a body is SHOWN (scrubbed) when the caller has no
    # machine-readable account of the failure -- the legacy prose sites, whose
    # whole value is the upstream's own sentence -- and is NOT quoted at all when
    # the upstream did send a designed vocabulary, which is why the structured
    # path reports only its ``error``/``code``/``details``. Neither half may show
    # a credential, and neither is a body-free half waiting to be made
    # consistent with the other.
    return scrubbed_response_body(exc.response)


def scrubbed_response_body(response: Optional[requests.Response]) -> str:
    """The body of a FAILED response, scrubbed, for interpolation into a message.

    The direct-site counterpart of :func:`response_body`: several clients raise
    their own ``RuntimeError`` on a non-2xx status they saw themselves rather
    than one ``requests`` raised for them, and read the body off the response
    directly. They come through here so that "the body a message quotes" is
    scrubbed in one place instead of in each of them, which is also why the
    scrubber below is a pattern rather than a list of the caller's own keys: the
    helper does not know which client called it.

    Args:
        response: The failed response, or ``None`` when there was none.

    Returns:
        The decoded, scrubbed body, or :data:`NO_RESPONSE_BODY` when there is
        nothing to show.
    """

    if response is None:
        return NO_RESPONSE_BODY
    # errors="replace": a body that is not valid UTF-8 (a gzip/binary error page,
    # a latin-1 proxy response) must not turn "report the failure" into a
    # UnicodeDecodeError raised from inside the error handler.
    body = response.content.decode(errors="replace")
    return scrub_secrets(body) if body.strip() else NO_RESPONSE_BODY


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


#: Credential SHAPES masked in anything about to be surfaced, as
#: ``(pattern, replacement)``. A pattern rather than the caller's own key values
#: because this module produces the body for every client in the package and
#: knows none of their keys: a scrubber that has to be handed the secret is one
#: each new call site can forget, which is exactly how the fix to ``response_body``
#: briefly widened the exposure to five clients that never redacted anything.
#:
#: Kept NARROW on purpose. Masking too much makes an upstream's refusal
#: unreadable (the body is often the only account of what happened), so each rule
#: needs a credential to be spelled in a way its issuer spells one: a scheme
#: keyword, a credential-name assignment, a query parameter, or an issuer prefix
#: on a bare token. A high-entropy fragment with none of those around it is left
#: alone -- see :func:`scrub_secrets` for what that costs.
_CREDENTIAL_SHAPES: Tuple[Tuple[Pattern[str], str], ...] = (
    # `Authorization: Bearer <token>`, in every casing and with every punctuation
    # an upstream might use. The scheme keyword IS the credential context here.
    (re.compile(r"(?i)\b(bearer)\s+([A-Za-z0-9._~+/=-]{4,})"), r"\1 " + REDACTION_MARKER),
    # A named credential: `"api_key": "..."`, `api_key=...`, `token: ...`. The
    # name is bounded by word characters, so `max_tokens` (an ordinary model
    # parameter) is not one of these and its value survives -- the underscore is
    # a word character and there is no boundary between it and `tokens`.
    #
    # The 8-character floor is what keeps ordinary prose out of it: `"api_key":
    # "missing"` stays readable, while a key of real length is masked.
    (
        re.compile(
            r"(?i)\b(authorization|proxy-authorization|api[-_]?key|apikey|x-api-key|"
            r"access[-_]?token|refresh[-_]?token|auth[-_]?token|id[-_]?token|"
            r"secret[-_]?key|client[-_]?secret|password|passwd|token|secret)"
            r"\b(\"?\s*[:=]\s*)(\"?)([A-Za-z0-9._~+/=-]{8,})"
        ),
        r"\1\2\3" + REDACTION_MARKER,
    ),
    # A credential in a query string, which is how several of these APIs accept
    # one and therefore how one comes back in a URL an upstream quotes.
    (
        re.compile(r"(?i)([?&](?:api[-_]?key|apikey|key|token|access[-_]?token)=)([^&\s\"']{4,})"),
        r"\1" + REDACTION_MARKER,
    ),
    # A BARE credential, by the prefix its issuer gives it -- the shape left over
    # when an upstream quotes the request's header value without the header. The
    # prefixes are the issuers this package actually talks to and the vendors
    # whose keys are widely used here, not a general "looks random" heuristic.
    (
        re.compile(r"\b(?:sk|pk|rk|hf|gsk|xai|tvly|fal|serp|glpat|ya29)[-_][A-Za-z0-9_.-]{6,}"),
        REDACTION_MARKER,
    ),
    (re.compile(r"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{20,}\b"), REDACTION_MARKER),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), REDACTION_MARKER),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), REDACTION_MARKER),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"), REDACTION_MARKER),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), REDACTION_MARKER),
)


def scrub_secrets(text: str, secrets: Iterable[Optional[str]] = ()) -> str:
    """Remove credential-shaped content from a body about to be surfaced.

    WHAT THIS GUARANTEES, AND WHAT IT DOES NOT: it guarantees that a credential
    spelled the way its issuer spells one -- a ``Bearer`` header, a named field, a
    query parameter, a vendor-prefixed token -- cannot reach a message, whoever
    the caller is and whether or not they remembered anything. It does NOT
    recognise an opaque value with no such context around it (a bare tenant id, a
    short legacy key); for that, a client that KNOWS its own credential should
    also pass it to :func:`redact_secrets`, which removes exact values and is what
    ``RadientClient`` does around this call.

    Every body this package surfaces goes through here, so the cheap catch is
    applied once and centrally. The cost of a miss is a leaked credential in a
    user-facing message and in the log; the cost of masking something that was
    not a credential is a slightly less specific error sentence.

    Args:
        text: The text about to be surfaced.
        secrets: Credential values to remove exactly, in addition to the shapes.

    Returns:
        The text with every recognised credential replaced by a marker.
    """

    text = redact_secrets(text, secrets)
    for pattern, replacement in _CREDENTIAL_SHAPES:
        text = pattern.sub(replacement, text)
    return text


def scrub_details(value: Any, secrets: Iterable[Optional[str]] = ()) -> Any:
    """The STRUCTURED counterpart of :func:`scrub_secrets`, for a refusal's details.

    WHY A SECOND ENTRY POINT: a refusal's ``details`` are a JSON object and not a
    sentence, so the masker has to walk them instead of being handed one string --
    but they are the SAME credential shapes, and re-deriving the patterns for them
    is how the two places drift. The values in there are what a client RENDERS
    (``details.rule``, ``details.field``) and what the desktop app builds a
    sentence from, so a credential an upstream put in one of them reaches the
    operator by exactly the route the message half was already fixed for.

    Recursive because the shape is the upstream's, not ours: ``categories`` is a
    list, and an arm that meets a nested object should walk it rather than skip it
    -- a field this module has no shape for is where an unscrubbed string hides.

    KEYS are deliberately left alone. A key is the part of the contract a renderer
    switches on (``field``, ``rule``, ``existing_agent_id``), so masking one would
    break a reader to protect against a value no upstream has been seen to put in a
    key. Non-string scalars pass through unchanged, so ``owned_by_caller`` and
    ``limit_bytes`` keep being the boolean and the number they were: the details
    are machine-readable by design, and a masker that stringified them would break
    the callers this exists to keep working.

    Args:
        value: The details value about to be surfaced, at any nesting depth.
        secrets: Credential values to remove exactly, in addition to the shapes.

    Returns:
        The same shape, with every recognised credential replaced by the marker.
    """

    if isinstance(value, str):
        return scrub_secrets(value, secrets)
    if isinstance(value, dict):
        return {key: scrub_details(item, secrets) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(scrub_details(item, secrets) for item in value)
    if isinstance(value, list):
        return [scrub_details(item, secrets) for item in value]
    return value


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
            sent none). Never the raw body -- that is :attr:`body`'s job, and only
            where the prose sites need it -- and never a credential: the string
            values are masked on the way in (see :func:`scrub_details`), so this
            attribute does not hold one for a caller to surface later.
        body: The upstream body, ALREADY SCRUBBED, for the callers that have no
            machine-readable account of the failure and whose value is the
            upstream's own sentence: the legacy prose sites, and the
            transcription/speech paths that classify an upstream refusal by
            reading its envelope. ``None`` wherever the upstream sent a designed
            vocabulary -- :func:`api_error_from_response` deliberately leaves it
            unset there, because such a refusal is fully described by
            :attr:`code` and :attr:`details` and not carrying the body is one
            fewer place a credential could reach a client. The value comes from
            :func:`scrubbed_response_body` (shape rules) and :func:`redact_secrets`
            (this caller's own credential), so it is never the unscrubbed body.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        body: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.details: Dict[str, Any] = dict(details) if details else {}
        self.body = body


def api_error_from_exception(
    exc: RequestException,
    *,
    prefix: str,
    secrets: Iterable[Optional[str]] = (),
) -> APIError:
    """Build an :class:`APIError` from the exception a ``requests`` call raised.

    The typed twin of the legacy prose shape
    ``RuntimeError(f"{prefix}: {exc}, Response Body: {body}")``: the message is
    byte-for-byte what that produced, and the status and the scrubbed body are
    ALSO carried as attributes so a caller can classify the failure instead of
    re-parsing its own message text. That difference is the whole point on the
    transcription path, where a transport failure, a provider quota refusal and
    the daemon's own internal fault have to reach a client as different statuses
    and only the failure itself knows which it was.

    The message keeps the ``Response Body`` clause only when there was a response
    to quote: "the status was 500 and it sent no body" and "we never reached it"
    are different failures, and a reader of the log cannot tell them apart from a
    shared message otherwise. The 2xx envelope legs are built by the clients
    directly -- there is no exception to pass here when the status was a success.

    Args:
        exc: The ``RequestException`` that was caught.
        prefix: What the caller was doing, e.g. ``"Failed to create transcription"``.
        secrets: Credential values to remove from the message and the body. An
            upstream that reflects the request it received must not be able to
            put this caller's key into either.

    Returns:
        An :class:`APIError` carrying the upstream status and the scrubbed body.
    """

    response = exc.response
    # The module's own shape rules have already run inside
    # :func:`scrubbed_response_body`; this adds the caller's credential, which no
    # shape rule can recognise as one. Identity, not equality, for the sentinel:
    # it is a module singleton, so `is` cannot be fooled by a server whose body
    # genuinely reads "No response body" -- that text is decoded fresh and is a
    # different string object, and a real body must be kept rather than dropped as
    # if it were the sentinel.
    raw = scrubbed_response_body(response)
    scrubbed = redact_secrets(raw, secrets)
    message = f"{prefix}: {exc}"
    if response is not None:
        message = f"{message}, Response Body: {scrubbed}"
    return APIError(
        message,
        status_code=response.status_code if response is not None else None,
        body=None if raw is NO_RESPONSE_BODY else scrubbed,
    )


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
    # ``details`` goes through the masker here, at the point the upstream's body
    # enters the package, for the same reason ``message`` does: what leaves this
    # module is a value object other code surfaces, and the fix that made the body
    # readable for every client is what put a credential-shaped string into a field
    # nothing was masking (a hub is free to put one in ``details``, and an arm that
    # forwards the dict verbatim publishes it). Masking the VALUE rather than the
    # response keeps the property for every consumer of the attribute, including
    # the ones that do not exist yet.
    return APIError(
        scrub_secrets(message or f"{fallback_message} (HTTP {response.status_code})", secrets),
        status_code=response.status_code,
        code=code,
        details=scrub_details(details, secrets),
    )
