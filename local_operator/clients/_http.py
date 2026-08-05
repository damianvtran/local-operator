"""Shared HTTP error helpers for the API clients in this package.

Every client in this package wraps its ``requests`` calls in the same error
handling shape: catch ``RequestException``, pull the response body out of the
exception, and interpolate it into a ``RuntimeError`` message. The helper here
is that extraction, in one place, so the behaviour is defined once instead of
being re-derived at every call site.
"""

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
    # The falsy test below is deliberate and is NOT the same as `response is None`:
    # requests.Response.__bool__ returns response.ok, so any 4xx/5xx response is
    # falsy and its body is reported as NO_RESPONSE_BODY even though the server
    # sent one. That is what every call site this helper replaced already did, and
    # it is preserved here to keep error text unchanged -- preserved, not endorsed.
    # Changing this one line to `if response is None:` fixes every caller at once.
    if not response:
        return NO_RESPONSE_BODY
    return response.content.decode()
