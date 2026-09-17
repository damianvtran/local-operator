import os
import re
import shutil
import tempfile
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.status import (
    HTTP_401_UNAUTHORIZED,
    HTTP_402_PAYMENT_REQUIRED,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_502_BAD_GATEWAY,
)

from local_operator.clients._http import APIError
from local_operator.clients.radient import (
    RadientClient,
    RadientTranscriptionResponseData,
)
from local_operator.server.dependencies import get_radient_client
from local_operator.server.models.schemas import CRUDResponse

router = APIRouter()


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
RADIENT_EDGE_STATUSES = frozenset({HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN, HTTP_404_NOT_FOUND})

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
    if exc.status_code == HTTP_401_UNAUTHORIZED:
        return (
            "Transcription is unavailable: Radient refused this app's credentials. "
            "Sign in again in the app; if it keeps failing, the daemon's Radient "
            "API key is invalid or has expired." + _upstream_clause(exc)
        )
    if exc.status_code == HTTP_403_FORBIDDEN:
        # 403 is an AUTHORISATION answer, not an authentication one, and the two
        # have different remedies. "Refused this app's credentials, sign in
        # again" asserts a cause the status does not establish (the credential
        # can be perfectly valid and simply not entitled to this endpoint -- a
        # plan that excludes transcription, or an edge rule), and it sends the
        # reader to re-authenticate on a guess. Worded for what a 403 says.
        return (
            "Transcription is unavailable: Radient refused the request (403). A 403 "
            "is a permission answer rather than a rejected credential, so signing in "
            "again is not the fix; check what the account is entitled to use, and "
            "the upstream's own words below." + _upstream_clause(exc)
        )
    if exc.status_code == HTTP_404_NOT_FOUND:
        return (
            "Transcription is unavailable: the Radient endpoint the daemon is "
            "configured to call was not found. Check the daemon's Radient API base "
            "URL." + _upstream_clause(exc)
        )
    return "Transcription is unavailable: Radient rejected the request." + _upstream_clause(exc)


def _classify_upstream_failure(exc: APIError, provider: str) -> HTTPException:
    """Map an upstream transcription failure onto a status the client can act on.

    The client only ever shows ``detail``, but it branches on the status, so the
    status has to be truthful on its own:

    * 402 -- out of credit, and the fix is to add some. Used for Radient's own
      refusal *and* for a provider quota refusal: the action is the same class
      of thing for the user (top up, or switch provider), the failure is not
      retryable so it must not look like a 429 that the client may retry, and
      402 is this codebase's own billing/quota code (``providers/clients.py``
      maps a billing failure to 402), so that convention is kept.
    * 502 -- the upstream call failed: a provider rejection, a Radient edge
      rejection, a provider fault, or a transport failure that never reached
      Radient at all. Not our fault, so never a 500.
    * 500 -- left to the caller for genuine internal faults only.
    """
    if exc.status_code is None:
        # A transport failure never reached Radient, so there is no upstream
        # status or body to add: the client's own text ("Connection refused",
        # "timed out") is the entire diagnostic and is passed through verbatim.
        return HTTPException(status_code=HTTP_502_BAD_GATEWAY, detail=str(exc))

    # Radient's own credit refusal. Checked before the body markers below so
    # that "insufficient credits" in a Radient 402 is attributed to Radient
    # rather than to whichever provider the request happened to name.
    if exc.status_code == HTTP_402_PAYMENT_REQUIRED:
        return HTTPException(
            status_code=HTTP_402_PAYMENT_REQUIRED,
            detail=(
                "Transcription is unavailable: your Radient credit balance is too low. "
                "Add credits to continue." + _upstream_clause(exc)
            ),
        )

    body = (exc.body or "").lower()
    if any(marker in body for marker in PROVIDER_CREDIT_MARKERS):
        return HTTPException(
            status_code=HTTP_402_PAYMENT_REQUIRED,
            detail=(
                f"Transcription is unavailable: the {provider} provider has run out of "
                f"credits. Switch to another provider, or add credits to your Radient "
                f"account." + _upstream_clause(exc)
            ),
        )

    if exc.status_code in PROVIDER_REJECTION_STATUSES:
        if _refusing_side(exc) == "radient":
            return HTTPException(
                status_code=HTTP_502_BAD_GATEWAY, detail=_radient_rejection_detail(exc)
            )
        if exc.body is None:
            # A 4xx that sent no body at all names nobody, and the sentence below
            # would assert the provider on the status alone -- the same class of
            # mistake as the incident this route was fixed for, in the other
            # direction. The daemon authenticates to Radient and to no one else,
            # so a bare status is equally consistent with Radient's own edge and
            # with the provider it relayed to. Say what is known, name both
            # candidates, and let the status carry the rest.
            return HTTPException(
                status_code=HTTP_502_BAD_GATEWAY,
                detail=(
                    "The transcription request was refused upstream (HTTP "
                    f"{exc.status_code}) and the upstream sent no body saying by "
                    "whom. The daemon only authenticates to Radient, so this is "
                    f"either Radient's edge or the {provider} provider it called."
                    + _upstream_clause(exc)
                ),
            )
        return HTTPException(
            status_code=HTTP_502_BAD_GATEWAY,
            detail=(
                f"The {provider} provider rejected the transcription request."
                + _upstream_clause(exc)
            ),
        )

    return HTTPException(
        status_code=HTTP_502_BAD_GATEWAY,
        detail="Transcription failed upstream." + _upstream_clause(exc),
    )


@router.post(
    "/v1/transcriptions",
    response_model=CRUDResponse[RadientTranscriptionResponseData],
    summary="Transcribe Audio File",
    tags=["Transcription"],
)
async def create_transcription_endpoint(
    radient_client: Annotated[RadientClient, Depends(get_radient_client)],
    file: UploadFile = File(...),
    # `model` and `provider` are deliberately unset by default: the daemon must not
    # choose a speech-to-text backend. Radient's agent-server owns that choice (its
    # configured default provider/model), and `RadientClient.create_transcription`
    # omits each field from the multipart body when it is None or empty, so the
    # server-side default governs. Pinning them here was not harmless redundancy: an
    # OpenAI model id sent alongside a server defaulted to a non-OpenAI provider
    # (e.g. ElevenLabs Scribe v2) fails outright, so the old defaults silently locked
    # the talk feature to OpenAI and to one model id. Callers that genuinely need a
    # specific backend still pass both explicitly, and are forwarded unchanged.
    #
    # Nothing here enforces that pairing, though: `provider` alone is forwarded as
    # given, because the daemon cannot know which model ids a provider serves.
    model: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
    language: Optional[str] = Form(None),
    provider: Optional[str] = Form(None),
) -> CRUDResponse[RadientTranscriptionResponseData]:
    """
    Transcribe an audio file using the specified model and parameters.

    The audio file is sent as `multipart/form-data`.

    `model`, `provider`, `language` and `prompt` are optional and default to
    unset: an omitted field is left out of the upstream request so Radient's
    own configured provider/model default applies. Pass `model` and `provider`
    together to pin a specific backend.
    """
    if not radient_client.api_key:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Radient API key is not configured on the server.",
        )

    # Save the uploaded file temporarily.
    #
    # The directory is created BEFORE the try so the cleanup below owns it
    # unconditionally. It used to be created inside, which meant a failure
    # while writing (a short read, a full disk) raised HTTPException out of
    # this block and never reached the `finally: rmtree` further down — that
    # sits on the *second* try, which the error path never enters. The result
    # was a temp directory leaked on every failed upload.
    temp_dir = tempfile.mkdtemp()
    try:
        temp_file_path = os.path.join(temp_dir, file.filename if file.filename else "audio.tmp")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded audio file: {str(e)}",
        )
    finally:
        await file.close()

    try:
        transcription_result = radient_client.create_transcription(
            file_path=temp_file_path,
            model=model,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            language=language,
            provider=provider,
        )

        return CRUDResponse(
            status=200,
            message="Transcription created successfully",
            result=transcription_result,
        )
    except FileNotFoundError:
        # This case should ideally be caught by the client if temp_file_path is wrong,
        # but good to have a catch here.
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Temporary audio file not found after saving.",
        )
    except ValueError as ve:  # For validation errors from the client
        raise HTTPException(status_code=400, detail=str(ve))
    except APIError as upstream:
        raise _classify_upstream_failure(upstream, provider or "upstream")
    except RuntimeError as re:  # For API errors or other runtime issues from the client
        # A plain RuntimeError from the client is ours: the client raises one
        # only for a server-side configuration fault (no Radient API key) or an
        # unforeseen internal error. Every failure that came from Radient or
        # from the provider it called arrives typed, handled above.
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(re))
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during transcription: {str(e)}",
        )
    finally:
        # Clean up the temporary directory and its contents
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
