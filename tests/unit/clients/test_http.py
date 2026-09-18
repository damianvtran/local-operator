"""Tests for the typed half of the shared HTTP error helpers.

``local_operator.clients._http`` has two ways to describe a failed upstream call,
and they answer different questions. The legacy one is a prose message the caller
interpolates (``response_body`` plus a prefix, raised as a ``RuntimeError``); the
typed one is :func:`api_error_from_exception`, which builds the same message but
also carries the upstream status and body as attributes, so a layer above the
client can pick a truthful status for ITS caller instead of re-parsing its own
message text. That is what the transcription route needs: a transport failure, a
provider quota refusal and the daemon's own internal fault have to reach a client
as different statuses, and only the failure itself knows which it was.

The helpers' own reading and scrubbing behaviour is covered by
``tests/unit/clients/test_error_body_redaction.py`` (on ``main``); this file covers
what the typed path adds -- the attributes, the message that must stay
byte-identical to the legacy shape, and the fact that a caller's own credential is
scrubbed out of BOTH the message and the carried body.
"""

from typing import Callable

import requests

from local_operator.clients._http import (
    NO_RESPONSE_BODY,
    REDACTION_MARKER,
    APIError,
    api_error_from_exception,
)


def _http_error(response: requests.Response) -> requests.HTTPError:
    """Raise the way ``raise_for_status()`` does: an HTTPError carrying the response."""
    return requests.exceptions.HTTPError("500 Server Error", response=response)


def test_api_error_body_defaults_to_none_so_a_caller_reads_one_shape() -> None:
    """``body`` is always readable: absent means "nothing to quote", not "no attribute".

    A classifier tests ``exc.body`` directly, so the attribute has to exist on
    every instance -- including the ones a structured refusal builds, which
    deliberately never carry a body.
    """
    assert APIError("boom").body is None
    assert APIError("boom", body="the upstream said so").body == "the upstream said so"


def test_api_error_from_exception_carries_the_status_and_the_body(
    real_response: Callable[[int, bytes], requests.Response],
) -> None:
    """The typed error keeps the upstream's status and its own words."""
    body = b'{"error":"OpenAI API error (insufficient_quota): You have no credits remaining."}'
    error = _http_error(real_response(500, body))

    exc = api_error_from_exception(error, prefix="Failed to create transcription")

    assert isinstance(exc, APIError)
    assert isinstance(exc, RuntimeError)  # existing `except RuntimeError` sites still catch it
    assert exc.status_code == 500
    assert exc.body == body.decode()
    assert "Failed to create transcription" in str(exc)
    assert "500" in str(exc)
    assert "insufficient_quota" in str(exc)


def test_api_error_from_exception_keeps_the_legacy_message_verbatim(
    real_response: Callable[[int, bytes], requests.Response],
) -> None:
    """The message is what the pre-existing prose sites produced, character for character.

    The typed path replaced an inline ``f"{prefix}: {exc}, Response Body: {body}"``
    at the transcription call site. Callers and tests identify these failures by
    that text, so re-expressing the error as a type must not reword it.
    """
    body = b'{"detail":"Invalid or expired token"}'
    error = _http_error(real_response(401, body))

    exc = api_error_from_exception(error, prefix="Failed to create transcription")

    assert str(exc) == f"Failed to create transcription: {error}, Response Body: {body.decode()}"


def test_api_error_from_exception_omits_the_body_clause_without_a_response() -> None:
    """A transport failure keeps its own text and claims no status or body."""
    error = requests.exceptions.ConnectionError("Connection refused")

    exc = api_error_from_exception(error, prefix="Failed to create transcription")

    assert exc.status_code is None
    assert exc.body is None
    assert "Connection refused" in str(exc)
    assert "Response Body" not in str(exc)


def test_api_error_from_exception_has_no_body_for_an_empty_body(
    real_response: Callable[[int, bytes], requests.Response],
) -> None:
    """The sentinel is not passed on as if it were content.

    The message still says the body was empty -- "500 with nothing in it" and "we
    never reached it" are different failures a reader must be able to tell apart,
    which is exactly why the clause is kept -- but ``body`` stays ``None`` so a
    classifier reads one shape.
    """
    exc = api_error_from_exception(
        _http_error(real_response(500, b"")), prefix="Failed to create transcription"
    )

    assert exc.status_code == 500
    assert exc.body is None
    assert NO_RESPONSE_BODY in str(exc)


def test_api_error_from_exception_scrubs_the_callers_credential_from_both_halves(
    real_response: Callable[[int, bytes], requests.Response],
) -> None:
    """An upstream that echoes the request cannot put the key in the message OR the body.

    Both are read by whoever handles the failure, and the body is what a
    classification reads, so scrubbing only the message would leave the key in the
    attribute a client-facing sentence is rendered from.

    The echoed spelling is deliberately one the SHAPE rules cannot see -- a bare
    value with no scheme keyword, credential name, query parameter or vendor
    prefix around it -- because that is the case :func:`redact_secrets` exists for
    and the only one that proves the caller's own values are threaded through.
    """
    key = "rdnt" + "a1b2c3d4" * 8
    body = ('{"detail":"refused: the key ' + key + ' is not valid"}').encode()
    error = _http_error(real_response(401, body))

    exc = api_error_from_exception(error, prefix="Failed to create transcription", secrets=[key])

    assert key not in str(exc)
    assert exc.body is not None
    assert key not in exc.body
    assert REDACTION_MARKER in str(exc)
