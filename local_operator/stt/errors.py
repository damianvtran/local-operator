"""The shared transcription failure mapping.

Extracted VERBATIM from ``server/routes/transcription.py`` (its lines 28-266
at the time of the move) so the legacy Radient route and the cascade executor
classify one refusal the same way, in one place — two surfaces that both end
up telling the user why their speech failed must never disagree about it. The
legacy route imports :func:`classify_upstream_failure` and keeps raising the
result; the byte-for-byte sentences for every legacy path are pinned by
``tests/unit/stt/test_errors.py``.

Two vocabularies live here:

* the **Radient** classification (unchanged): the daemon authenticates to
  Radient and to nobody else, so a 4xx has to be attributed between Radient's
  own edge and the provider it relayed by reading the body, not the status
  (:func:`_refusing_side`, and the recorded residual in its docstring).
* the **direct BYO rungs** (ElevenLabs, OpenAI): the daemon calls those hosts
  with the user's own key, so there is no second hop to attribute — a 401 is
  that provider refusing that key, and the sentences say so by name.

Import-light on purpose: FastAPI is pulled in only by the HTTP wrapper, so the
package stays importable from the session runtime and tests that never touch
the HTTP plane.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterable, Optional, Tuple

from local_operator.clients._http import (
    APIError,
    error_payload,
    scrub_details,
    scrub_secrets,
)
from local_operator.stt import AudioPath

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import HTTPException
    from httpx import Response as HttpxResponse

# Upstream text that means "the provider (not Radient) refused for want of
# credit". Matched as a substring against the lowercased upstream body.
#
# The list is deliberately short, literal and quota-specific. A provider quota
# refusal is the one upstream failure a user can act on, so it is worth
# recognising; anything broader ("error", "failed") would catch an ordinary
# provider fault and mislabel it as a billing problem, which is worse than not
# classifying it. Two entries were exactly that and are deliberately gone: a
# bare "billing" matched a region restriction or any docs link containing
# "/account/billing", and a bare "credit balance" matched any sentence that
# happened to contain both words. "credit balance is too low" is the provider
# wording that carries the same meaning -- it is what Anthropic says when the
# account is empty -- without the false positives.
PROVIDER_CREDIT_MARKERS = (
    "insufficient_quota",
    "insufficient quota",
    "exceeded your current quota",
    "no credits remaining",
    "out of credits",
    "insufficient credits",
    "credit balance is too low",
)

# Statuses that mean the request itself was refused -- a bad or revoked
# credential, an unroutable path, a parameter the server will not accept.
#
# The status alone does NOT say who refused it. Radient proxies a provider
# rejection as a plain 500, so a 4xx reaching the daemon is at least as likely
# to be Radient's own edge as the provider's; which one it was comes from the
# body. See _refusing_side.
PROVIDER_REJECTION_STATUSES = frozenset({400, 401, 403, 404, 422})

# Statuses only Radient's own edge can produce on this path, used to attribute a
# 4xx whose body carries no envelope to read. The daemon authenticates to
# Radient and to nobody else, so 401/403 is Radient refusing this app's
# credential -- the incident's cause #1, and the exact shape that used to be
# reported as "the provider rejected it". A 404 means the Radient endpoint the
# daemon was configured to call does not exist, i.e. a base-URL mistake, because
# Radient's own routes are fixed.
RADIENT_EDGE_STATUSES = frozenset({401, 403, 404})

# Radient's API is FastAPI, and FastAPI renders every error its own edge
# generates as a JSON object whose first key is `detail`: a string for a refusal
# it raised deliberately ({"detail":"Invalid or expired token"}), a list of
# field errors for a request it refused to parse. That envelope is evidence that
# Radient is the one speaking.
#
# A provider's refusal arrives in the provider's own envelope instead: OpenAI
# answers {"error": {...}}, and Radient's own wrapper around a provider fault
# carries the provider's words under the same key. So a top-level `error` key is
# evidence that the provider is speaking. `[^}]*?` keeps it to a TOP-LEVEL key
# (it cannot cross a closing brace), because a provider error nested inside a
# Radient `detail` value is Radient relaying it, not the provider answering us.
#
# Both patterns are anchored at the start of the body on purpose. The envelope is
# read WITHOUT a JSON parse, deliberately: a body that is not JSON at all -- an
# intermediary's HTML page, a truncated or empty response -- would raise inside a
# parser, on the error path, and lose the very evidence this reads. An envelope
# starts at the head, and a body that does not open like JSON matches neither
# pattern and falls through to the statuses below, which is the intended answer
# rather than an error.
_RADIENT_ERROR_ENVELOPE = re.compile(r'\A\s*\{\s*"detail"\s*:')
_PROVIDER_ERROR_ENVELOPE = re.compile(r'\A\s*\{[^}]*?"error"\s*:', re.DOTALL)

#: Display names for the rungs the daemon calls DIRECTLY. A rung in this map
#: changes the sentence vocabulary: there is no relay hop to attribute, so the
#: provider is named plainly instead of "provider or Radient's edge".
_DIRECT_PROVIDER_LABELS = {
    AudioPath.PROVIDER_STT_ELEVENLABS: "ElevenLabs",
    AudioPath.PROVIDER_STT_OPENAI: "OpenAI",
}


def _upstream_clause(exc: APIError) -> str:
    """Render the upstream status and body as a trailing diagnostic clause.

    The sentence in front of it says what to do; this says what actually
    happened, in the upstream's own words. Both are kept because the two
    audiences differ: a user reads the sentence, whoever is on support reads
    the clause -- and today a provider quota refusal reached the client as no
    text at all, which is what made it untriageable.

    A 2xx is worded differently from an error status because it is not
    self-evidently a failure: Radient reports some provider failures in the body
    of a 200, and "Upstream responded 200" sitting next to "Transcription
    failed upstream" reads as a contradiction rather than as the diagnostic it
    is.
    """
    if exc.body:
        if exc.status_code is not None and 200 <= exc.status_code < 300:
            return f" Radient reported an error (HTTP {exc.status_code}): {exc.body}"
        return f" Upstream responded {exc.status_code}: {exc.body}"
    return f" Upstream responded {exc.status_code} with no body."


def _refusing_side(exc: APIError) -> str:
    """Name the hop that refused a 4xx request, from the body rather than the status.

    The two failures need different sentences because they need different
    actions from the reader, and the incident this route was fixed for was
    Radient refusing the daemon's own credential while the message pointed at
    the provider's configuration.

    Returns:
        str: ``"radient"`` or ``"provider"``.
    """
    body = exc.body or ""
    if _RADIENT_ERROR_ENVELOPE.match(body):
        return "radient"
    if _PROVIDER_ERROR_ENVELOPE.match(body):
        return "provider"
    # No envelope to read -- an HTML page from a wrong base URL, an empty body, a
    # JSON object shaped like neither. Fall back to the statuses only one hop can
    # answer on this path, and otherwise to the provider, which is where a 400/422
    # that names nobody has always been attributed.
    #
    # RECORDED RESIDUAL, not a hidden one: this fallback attributes a 401/403/404
    # to Radient whatever the body says, so a provider 4xx relayed verbatim in a
    # FastAPI `detail` envelope, or any 401/403/404 with no envelope at all, is
    # attributed to the wrong hop. Reaching that needs Radient to forward a
    # provider's 4xx with the provider's own status and without its own `error`
    # wrapper, and no artefact in this repository shows it (Radient's own edge is
    # FastAPI and emits `detail`; its relay of a provider fault is the incident's
    # 500 with the provider's `error` body). Left as it is on purpose: guessing a
    # third signal that no evidence supports is how the original misdirection
    # happened. If that shape is ever seen in the wild, the discriminator needs a
    # real signal (Radient's own relay key) rather than another status guess.
    return "radient" if exc.status_code in RADIENT_EDGE_STATUSES else "provider"


def _radient_rejection_detail(exc: APIError) -> str:
    """Say that Radient refused the request, and name the fix for the status.

    Generic wording is deliberately avoided: "the request was rejected" tells the
    reader nothing they can act on, and the two shapes that matter here -- a
    refused credential and a base URL pointing at a route that does not exist --
    have specific, checkable remedies.
    """
    if exc.status_code == 401:
        return (
            "Transcription is unavailable: Radient refused this app's credentials. "
            "Sign in again in the app; if it keeps failing, the daemon's Radient "
            "API key is invalid or has expired." + _upstream_clause(exc)
        )
    if exc.status_code == 403:
        # 403 is an AUTHORISATION answer, not an authentication one, and the two
        # have different remedies. "Refused this app's credentials, sign in
        # again" asserts a cause the status does not establish (the credential
        # can be perfectly valid and simply not be entitled to this endpoint -- a
        # plan that excludes transcription, or an edge rule), and it sends the
        # reader to re-authenticate on a guess. Worded for what a 403 says.
        return (
            "Transcription is unavailable: Radient refused the request (403). A 403 "
            "is a permission answer rather than a rejected credential, so signing in "
            "again is not the fix; check what the account is entitled to use, and "
            "the upstream's own words below." + _upstream_clause(exc)
        )
    if exc.status_code == 404:
        return (
            "Transcription is unavailable: the Radient endpoint the daemon is "
            "configured to call was not found. Check the daemon's Radient API base "
            "URL." + _upstream_clause(exc)
        )
    return "Transcription is unavailable: Radient rejected the request." + _upstream_clause(exc)


def _direct_rejection_detail(exc: APIError, label: str) -> str:
    """Name a direct-BYO rung refusing the request, per status.

    The direct rungs have no relay hop, so the sentences name the provider and
    give the remedy that matches the status: replace a refused key (401), read
    what the account is entitled to (403, whose remedy is NOT re-entering the
    key), check the endpoint (404), or else state the rejection.
    """
    if exc.status_code == 401:
        return (
            f"Transcription is unavailable: {label} refused the API key stored for "
            f"it (HTTP 401). Check the key for this provider, or replace it and try "
            f"again." + _upstream_clause(exc)
        )
    if exc.status_code == 403:
        return (
            f"Transcription is unavailable: {label} refused the request (403). A 403 "
            f"is a permission answer rather than a rejected key, so re-entering the "
            f"key is not the fix; check what the account is entitled to use, and the "
            f"upstream's own words below." + _upstream_clause(exc)
        )
    if exc.status_code == 404:
        return (
            f"Transcription is unavailable: the {label} transcription endpoint was "
            f"not found. Check the endpoint the daemon is configured to call."
            + _upstream_clause(exc)
        )
    return f"Transcription is unavailable: {label} rejected the request." + _upstream_clause(exc)


def failure_status_and_detail(
    exc: APIError, provider: str, *, rung: AudioPath | None = None
) -> Tuple[int, str]:
    """Map an upstream transcription failure onto (status, detail text).

    The status has to be truthful on its own because clients branch on it:

    * 402 -- out of credit, and the fix is to add some. Used for Radient's own
      refusal *and* for a provider quota refusal: the action is the same class
      of thing for the user (top up, or switch provider), the failure is not
      retryable so it must not look like a 429 that the client may retry, and
      402 is this codebase's own billing/quota code (``providers/clients.py``
      maps a billing failure to 402), so that convention is kept.
    * 502 -- the upstream call failed: a provider rejection, a Radient edge
      rejection, a provider fault, or a transport failure that never reached
      the upstream at all. Not our fault, so never a 500.
    * 500 -- left to the caller for genuine internal faults only.

    ``rung`` selects the vocabulary. ``None`` (and the Radient rung) keeps the
    original Radient sentences byte-for-byte; an ElevenLabs/OpenAI rung uses
    the direct-provider sentences, because there is no relay hop to attribute.
    ``provider`` names the provider Radient relayed to, for the Radient
    vocabulary only (the legacy route passes its configured ``provider`` form
    value, or ``"upstream"`` when unset).
    """
    label = _DIRECT_PROVIDER_LABELS.get(rung) if rung is not None else None

    if exc.status_code is None:
        # A transport failure never reached the upstream, so there is no
        # upstream status or body to add: the client's own text ("Connection
        # refused", "timed out") is the entire diagnostic and is passed through
        # verbatim.
        return 502, str(exc)

    if exc.status_code == 402:
        if label is not None:
            return (
                402,
                f"Transcription is unavailable: your {label} credit balance is too "
                f"low. Add credits to continue." + _upstream_clause(exc),
            )
        # Radient's own credit refusal. Checked before the body markers below so
        # that "insufficient credits" in a Radient 402 is attributed to Radient
        # rather than to whichever provider the request happened to name.
        return (
            402,
            "Transcription is unavailable: your Radient credit balance is too low. "
            "Add credits to continue." + _upstream_clause(exc),
        )

    body = (exc.body or "").lower()
    if any(marker in body for marker in PROVIDER_CREDIT_MARKERS):
        if label is not None:
            return (
                402,
                f"Transcription is unavailable: your {label} account has run out of "
                f"credits. Add credits to continue." + _upstream_clause(exc),
            )
        return (
            402,
            f"Transcription is unavailable: the {provider} provider has run out of "
            f"credits. Switch to another provider, or add credits to your Radient "
            f"account." + _upstream_clause(exc),
        )

    if exc.status_code in PROVIDER_REJECTION_STATUSES:
        if label is not None:
            return 502, _direct_rejection_detail(exc, label)
        if _refusing_side(exc) == "radient":
            return 502, _radient_rejection_detail(exc)
        if exc.body is None:
            # A 4xx that sent no body at all names nobody, and the sentence below
            # would assert the provider on the status alone -- the same class of
            # mistake as the incident this route was fixed for, in the other
            # direction. The daemon authenticates to Radient and to no one else,
            # so a bare status is equally consistent with Radient's own edge and
            # with the provider it relayed to. Say what is known, name both
            # candidates, and let the status carry the rest.
            return (
                502,
                "The transcription request was refused upstream (HTTP "
                f"{exc.status_code}) and the upstream sent no body saying by "
                "whom. The daemon only authenticates to Radient, so this is "
                f"either Radient's edge or the {provider} provider it called."
                + _upstream_clause(exc),
            )
        return (
            502,
            f"The {provider} provider rejected the transcription request." + _upstream_clause(exc),
        )

    return 502, "Transcription failed upstream." + _upstream_clause(exc)


def classify_upstream_failure(
    exc: APIError, provider: str, *, rung: AudioPath | None = None
) -> "HTTPException":
    """The HTTP face of :func:`failure_status_and_detail` (legacy route compat).

    The legacy Radient route imports this under its old name and raises the
    result unchanged; the FastAPI import is deferred so this module stays
    importable without the HTTP plane.
    """
    from fastapi import HTTPException

    status_code, detail = failure_status_and_detail(exc, provider, rung=rung)
    return HTTPException(status_code=status_code, detail=detail)


def api_error_from_httpx_response(
    response: "HttpxResponse",
    *,
    fallback_message: str,
    secrets: Iterable[Optional[str]] = (),
) -> APIError:
    """Build an :class:`APIError` from a failed ``httpx`` response.

    The httpx twin of the requests-side builder: it reuses the same primitives
    (designed ``error`` string, status/code/details, mask-on-the-way-in), with
    one deliberate difference — the SCRUBBED raw body is carried on
    :attr:`APIError.body`. These refusals end up in the transcription
    surfaces' detail clause ("Upstream responded 500: …"), whose whole value
    is the upstream's own sentence, and the legacy Radient path carries its
    bodies the same way (``RadientClient._surfaceable_body``); a client that
    dropped the body here would make every direct-rung refusal read "with no
    body" while the data was in hand. Scrub first, carry the scrubbed value.
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


__all__ = [
    "PROVIDER_CREDIT_MARKERS",
    "PROVIDER_REJECTION_STATUSES",
    "RADIENT_EDGE_STATUSES",
    "api_error_from_httpx_response",
    "classify_upstream_failure",
    "failure_status_and_detail",
]
