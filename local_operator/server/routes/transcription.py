import math
import os
import shutil
import tempfile
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from local_operator.clients._http import APIError
from local_operator.clients.radient import (
    RadientClient,
    RadientTranscriptionResponseData,
    new_correlation_id,
)
from local_operator.clients.radient_cost import cost_from_payload
from local_operator.config import ConfigManager
from local_operator.i18n.keys import wire_errors_media as _wm
from local_operator.server.dependencies import get_config_manager, get_radient_client
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.utils.voice_spend import emit_voice_spend
from local_operator.stt.cleanup import (
    InvalidCleanupMode,
    configured_cleanup,
    normalize_language_hint,
    resolve_cleanup,
)
from local_operator.stt.errors import (
    classify_upstream_failure as _classify_upstream_failure,
)
from local_operator.stt.radient_cleanup import (
    normalize_radient_cleanup,
    radient_cleanup_options,
)

router = APIRouter()


# The upstream failure mapping that used to live here (the credit markers, the
# rejection statuses, the envelope readers and ``_classify_upstream_failure``)
# moved to ``local_operator.stt.errors``: the cascade executor classifies the
# same refusals, and two surfaces that both explain a failed transcription must
# not be able to disagree about it. ``_classify_upstream_failure`` below is the
# shared function under this module's historical name, imported unchanged; the
# sentences it produces are byte-for-byte what this module produced.


@router.post(
    "/v1/transcriptions",
    response_model=CRUDResponse[RadientTranscriptionResponseData],
    summary="Transcribe Audio File",
    tags=["Transcription"],
)
async def create_transcription_endpoint(
    radient_client: Annotated[RadientClient, Depends(get_radient_client)],
    config_manager: Annotated[ConfigManager, Depends(get_config_manager)],
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
    #: The same closed vocabulary as the cascade routes. Absent = ``auto``;
    #: ``auto``/``client`` apply the shared rules, ``off``/``provider`` are
    #: verbatim. ``edit`` is 422 until its opt-in slice.
    cleanup: Optional[str] = Form(None),
    #: Optional client correlation id (PR-3): additive and tolerant — a caller
    #: that has it sends it, every other form value keeps working unchanged,
    #: and it is used for the spend record (analytics + the session's pending
    #: marker) and for the hub correlation header. Never for credential lookup.
    session_id: Optional[str] = Form(None),
) -> CRUDResponse[RadientTranscriptionResponseData]:
    """
    Transcribe an audio file using the specified model and parameters.

    The audio file is sent as `multipart/form-data`.

    `model`, `provider`, `language` and `prompt` are optional and default to
    unset: an omitted field is left out of the upstream request so Radient's
    own configured provider/model default applies. Pass `model` and `provider`
    together to pin a specific backend.
    """
    # A direct call (tests, in-process callers) can pass the FastAPI ``Form``
    # sentinel; only a real string is a mode.
    raw_cleanup = cleanup if isinstance(cleanup, str) and cleanup.strip() else None
    try:
        mode = resolve_cleanup(raw_cleanup or configured_cleanup(config_manager))
    except InvalidCleanupMode as exc:
        raise HTTPException(status_code=422, detail=_wm.media_refusal(error=str(exc)).envelope())
    if not radient_client.api_key:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_wm.radient_key_missing().envelope(),
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
            detail=_wm.save_audio_failed(error=str(e)).envelope(),
        )
    finally:
        await file.close()

    try:
        request_id = new_correlation_id()
        transcription_result = radient_client.create_transcription(
            file_path=temp_file_path,
            model=model,
            response_format=response_format,
            temperature=temperature,
            # Radient's field is ISO-639-1; a BCP-47 tag or an unmapped
            # 3-letter label is omitted rather than sent as a 400.
            language=normalize_language_hint(language),
            provider=provider,
            session_id=str(session_id or "") or None,
            client_request_id=request_id,
            **radient_cleanup_options(mode, prompt, language),
        )

        # The legacy route shares the cascade's pure rules, so two live
        # endpoints cannot return different text for the same audio. The
        # response is additive: ``cleanup``/``raw_text``/``removed`` sit beside
        # the existing fields (the model allows extras).
        normalized = normalize_radient_cleanup(transcription_result, mode, caller_language=language)
        transcription_result.text = normalized.text
        transcription_result.language = normalized.language
        transcription_result.language_probability = normalized.language_probability
        transcription_result.cleanup = normalized.cleanup
        transcription_result.raw_text = normalized.raw_text
        transcription_result.raw_unavailable = normalized.raw_unavailable
        transcription_result.removed = [
            {"start": span.start, "end": span.end, "kind": span.kind, "token": span.token}
            for span in normalized.removed
        ]

        # The charge (PR-3): the hub's transcription envelope carries the cost
        # object inside ``result``. The unit count is the metered seconds —
        # the server's own when stated, else the audio duration CEILED, which
        # is the meter the hub prices by ("rounded up to the whole second");
        # an absent amount stays unknown, never zero. ``isfinite`` is load-
        # bearing: pydantic accepts ``duration=inf`` and ``math.ceil`` RAISES
        # on it, which would 500 the route and discard a successful paid call
        # (review round 1, MINOR 2) — a non-finite duration is simply "no
        # count stated", the same reading the adapter gives non-finite money.
        duration = getattr(transcription_result, "duration", None)
        units = None
        if isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0:
            units = float(math.ceil(duration))
        emit_voice_spend(
            config_manager=config_manager,
            channel="stt",
            provider=str(getattr(transcription_result, "provider", "") or "radient"),
            units=units,
            unit="seconds",
            cost=cost_from_payload(transcription_result.model_dump()),
            session_id=str(session_id or ""),
            request_id=request_id,
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
            detail=_wm.temp_audio_missing().envelope(),
        )
    except ValueError as ve:  # For validation errors from the client
        raise HTTPException(status_code=400, detail=_wm.media_refusal(error=str(ve)).envelope())
    except APIError as upstream:
        raise _classify_upstream_failure(upstream, provider or "upstream")
    except RuntimeError as re:  # For API errors or other runtime issues from the client
        # A plain RuntimeError from the client is ours: the client raises one
        # only for a server-side configuration fault (no Radient API key) or an
        # unforeseen internal error. Every failure that came from Radient or
        # from the provider it called arrives typed, handled above.
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_wm.media_refusal(error=str(re)).envelope(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_wm.transcription_unexpected(error=str(e)).envelope(),
        )
    finally:
        # Clean up the temporary directory and its contents
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
