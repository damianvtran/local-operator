import os
import shutil
import tempfile
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.status import (
    HTTP_402_PAYMENT_REQUIRED,
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
# The list is deliberately short and literal. A provider quota refusal is the
# one upstream failure a user can fix from the UI, so it is worth recognising;
# anything broader ("error", "failed") would catch an ordinary provider fault
# and mislabel it as a billing problem, which is worse than not classifying it.
PROVIDER_CREDIT_MARKERS = (
    "insufficient_quota",
    "insufficient quota",
    "exceeded your current quota",
    "no credits remaining",
    "out of credits",
    "insufficient credits",
    "credit balance",
    "billing",
)

# Statuses that mean the provider rejected the request itself -- a bad or
# revoked key, an unavailable model, a parameter it will not accept. Radient
# usually proxies these as a plain 500, which is why the body matters more than
# the status it arrives with.
PROVIDER_REJECTION_STATUSES = frozenset({400, 401, 403, 404, 422})


def _upstream_clause(exc: APIError) -> str:
    """Render the upstream status and body as a trailing diagnostic clause.

    The sentence in front of it says what to do; this says what actually
    happened, in the upstream's own words. Both are kept because the two
    audiences differ: a user reads the sentence, whoever is on support reads
    the clause -- and today a provider quota refusal reached the client as no
    text at all, which is what made it untriageable.
    """
    if exc.body:
        return f" Upstream responded {exc.status_code}: {exc.body}"
    return f" Upstream responded {exc.status_code} with no body."


def _classify_upstream_failure(exc: APIError, provider: str) -> HTTPException:
    """Map an upstream transcription failure onto a status the client can act on.

    The client only ever shows ``detail``, but it branches on the status, so the
    status has to be truthful on its own:

    * 402 -- out of credit, and the fix is to add some. Used for Radient's own
      refusal *and* for a provider quota refusal: the action is the same class
      of thing for the user (top up, or switch provider), the failure is not
      retryable so it must not look like a 429 that the client may retry, and
      402 is the status the desktop already understands for exhausted credit,
      so that client keeps working without an update.
    * 502 -- the upstream call failed: a provider rejection, a provider fault,
      or a transport failure that never reached Radient at all. Not our fault,
      so never a 500.
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
                f"credits. Add credits to that provider, or switch to another one."
                + _upstream_clause(exc)
            ),
        )

    if exc.status_code in PROVIDER_REJECTION_STATUSES:
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
