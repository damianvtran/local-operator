"""The speech-cascade server surface.

Two routes, both gated on the managed-mode boundary like the legacy
``/v1/transcriptions`` they extend (``_LEGACY_CONTROL_PATHS`` in
``server/app.py``):

* ``GET /v1/stt/paths`` — the resolver's report: every rung in cascade order,
  whether it is available, and why; plus whether the model the caller names
  accepts audio input (that flag is what a surface reads to offer the audio
  door when no STT rung exists).
* ``POST /v1/stt/transcriptions`` — the cascade executor over one uploaded
  audio part. 200 with the text, path and per-rung attempts; 409 with the
  structured ``stt_unavailable`` payload when no STT rung was available; 402 /
  502 exactly as the legacy route maps upstream failures (the executor
  classifies with the shared mapper in ``stt/errors.py``, so the two surfaces
  cannot disagree).

The legacy ``/v1/transcriptions`` route stays the Radient-only path (rung 1),
but since LOH-5 slice 3 it shares the same pure cleanup rules and additive
provenance fields; it is not a second behaviour. Surfaces migrate to this
family when they want the cascade.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_409_CONFLICT,
    HTTP_422_UNPROCESSABLE_CONTENT,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_502_BAD_GATEWAY,
)

from local_operator.clients.radient_cost import RadientCost
from local_operator.config import ConfigManager
from local_operator.env import EnvConfig
from local_operator.i18n.keys import wire_errors_media as _wm
from local_operator.providers.auth_store import AuthStore
from local_operator.server.dependencies import (
    get_config_manager,
    get_env_config,
    get_provider_auth_store,
)
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.utils.voice_spend import emit_voice_spend
from local_operator.stt import AudioPath, AudioPathResolution
from local_operator.stt import audio as stt_audio
from local_operator.stt.cascade import (
    SttUnavailable,
    resolve_audio_path,
    transcribe_audio,
)
from local_operator.stt.cleanup import InvalidCleanupMode, configured_cleanup

router = APIRouter()


class SttAttemptPayload(BaseModel):
    """One rung's attempt, as the wire reports it."""

    path: AudioPath
    outcome: str
    detail: str = ""


class SttRemovedSpanPayload(BaseModel):
    """One span our cleanup rules removed, in raw-transcript coordinates."""

    start: int
    end: int
    kind: str
    token: str


class SttTranscriptionPayload(BaseModel):
    """The 200 body: the text, the rung that produced it, and the walk."""

    text: str
    path: AudioPath
    attempts: list[SttAttemptPayload]
    duration_s: float
    #: Cleanup provenance for ``text`` (never empty).
    cleanup: str = "none"
    #: Text before OUR rules. ``None`` means no raw copy exists; the reason is
    #: in ``raw_unavailable`` and must not be read as "nothing was removed".
    raw_text: Optional[str] = None
    raw_unavailable: str = ""
    language: Optional[str] = None
    removed: list[SttRemovedSpanPayload] = Field(default_factory=list)


def _resolve_query_model(provider: Optional[str], model: Optional[str]) -> Any | None:
    """Build the spec for a caller-named model, or ``None`` when none was named.

    The resolver reads exactly one attribute off the model (whether it accepts
    audio input), so building the real spec — not a bespoke capability lookup —
    keeps this endpoint's answer identical to the one a session would compute
    for the same model. Both halves are required together: a provider without a
    model cannot answer the capability question, and guessing the session's
    default model here would answer about a model the caller did not name.
    """
    if not provider and not model:
        return None
    if not provider or not model:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_CONTENT,
            detail=_wm.provider_model_pair().envelope(),
        )
    # An unknown provider is refused by name rather than answered about:
    # ``build_model_spec`` happily returns a placeholder spec for any string,
    # and a placeholder would make this endpoint report "the selected model
    # does not accept audio" for a provider that does not exist.
    from local_operator.providers.registry import get_provider_definition

    if get_provider_definition(provider) is None:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_CONTENT,
            detail=_wm.unknown_provider(provider=repr(provider)).envelope(),
        )
    # Imported lazily: model/configure.py is heavy and pulls the session
    # machinery, which the health of a plain `/v1/stt/paths` call should not
    # depend on for callers that name no model.
    from local_operator.model.configure import build_model_spec

    try:
        return build_model_spec(provider, model)
    except Exception as exc:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_CONTENT,
            detail=_wm.unknown_model(provider=provider, model=model, error=str(exc)).envelope(),
        )


def _rung_json(rung: Any) -> dict[str, Any]:
    """The 409 payload's per-rung entry (str-typed path for wire stability)."""
    return {"path": str(rung.path), "available": rung.available, "reason": rung.reason}


@router.get(
    "/v1/stt/paths",
    response_model=CRUDResponse[AudioPathResolution],
    summary="Resolve the speech-to-text path",
    tags=["Transcription"],
)
async def stt_paths_endpoint(
    config_manager: Annotated[ConfigManager, Depends(get_config_manager)],
    env_config: Annotated[EnvConfig, Depends(get_env_config)],
    store: Annotated[AuthStore, Depends(get_provider_auth_store)],
    provider: Optional[str] = Query(None, description="Model provider to probe for audio input"),
    model: Optional[str] = Query(None, description="Model id to probe for audio input"),
) -> CRUDResponse[AudioPathResolution]:
    """Report the cascade's rungs for this daemon, and the chosen path.

    Availability answers "a credential exists" for rungs 1-3 — a probe, not a
    promise. The per-submit truth (a refused key, an empty balance) surfaces at
    transcription time and falls forward.
    """
    resolution = await resolve_audio_path(
        config_dir=config_manager.config_dir,
        base_url=env_config.radient_api_base_url,
        model=_resolve_query_model(provider, model),
        store=store,
    )
    return CRUDResponse(
        status=200,
        message="Speech paths resolved",
        result=resolution,
    )


@router.post(
    "/v1/stt/transcriptions",
    response_model=CRUDResponse[SttTranscriptionPayload],
    summary="Transcribe Audio File With The Cascade",
    tags=["Transcription"],
)
async def create_stt_transcription_endpoint(
    config_manager: Annotated[ConfigManager, Depends(get_config_manager)],
    env_config: Annotated[EnvConfig, Depends(get_env_config)],
    store: Annotated[AuthStore, Depends(get_provider_auth_store)],
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    cleanup: Optional[str] = Form(None),
    #: Optional client correlation id (PR-3): additive and tolerant — used for
    #: the spend record (analytics + the session's pending marker) and the hub
    #: correlation header on the Radient rung; never for credential lookup.
    session_id: Optional[str] = Form(None),
    provider: Optional[str] = Query(
        None,
        description=(
            "Optional; with model, decides the 409's model_audio flag for the "
            "caller's own model rather than leaving it false"
        ),
    ),
    model: Optional[str] = Query(None, description="Optional; see provider"),
) -> CRUDResponse[SttTranscriptionPayload]:
    """Transcribe one upload through the provider cascade (Radient → EL → OpenAI)."""
    model_spec = _resolve_query_model(provider, model)

    # Save the upload before transcribing; the directory is created before the
    # try so the cleanup below owns it unconditionally (the legacy route's
    # leak, whose comment explains the shape, was fixed there and is not
    # reintroduced here).
    started = time.monotonic()
    temp_dir = tempfile.mkdtemp()
    try:
        # The temp name keeps the upload's own suffix where there is one so the
        # executor's suffix fallback describes the same container the caller
        # named; the sniffer is the primary evidence, the suffix only the
        # fallback (content over extension). ``basename`` because the uploaded
        # filename is client-controlled and must not steer the write path.
        basename = os.path.basename(file.filename or "")
        suffix = os.path.splitext(basename)[1] or stt_audio.extension_for_mime(
            file.content_type or ""
        )
        temp_file_path = os.path.join(temp_dir, f"upload{suffix}")
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

    # The request field wins; absent, the /settings key supplies the mode.
    requested_cleanup = cleanup if isinstance(cleanup, str) and cleanup.strip() else None
    effective_cleanup = requested_cleanup or configured_cleanup(config_manager)

    try:
        outcome = await transcribe_audio(
            Path(temp_file_path),
            config_dir=config_manager.config_dir,
            base_url=env_config.radient_api_base_url,
            store=store,
            model=model_spec,
            language=language,
            prompt=prompt,
            cleanup=effective_cleanup,
            # The caller's conversation id for the hub correlation header and
            # the emitted record — attribution only; the cascade's credential
            # scope stays as it was (QA round 1, Q4).
            correlation_session_id=str(session_id or "") or None,
        )
    except SttUnavailable as exc:
        if not exc.attempts:
            # No STT rung existed at all: the structured refusal a surface
            # reads to offer the audio door (``model_audio``).
            raise HTTPException(
                status_code=HTTP_409_CONFLICT,
                detail=_wm.media_refusal(error=str(exc)).envelope(code="stt_unavailable")
                | {
                    "rungs": [_rung_json(rung) for rung in exc.resolution.rungs],
                    "model_audio": exc.resolution.model_audio_capable,
                },
            )
        # Rungs existed but every one failed: the executor already classified
        # the chosen failure with the shared mapper (402 for quota, 502 for
        # upstream/transport), exactly as the legacy route maps them.
        raise HTTPException(
            status_code=exc.status_code or HTTP_502_BAD_GATEWAY,
            detail=_wm.media_refusal(error=exc.detail or str(exc)).envelope(),
        )
    except InvalidCleanupMode as exc:
        # A cleanup mode this build cannot execute is a client error, not the
        # prompt-length 400 arm below. The typed class keeps the two apart.
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_CONTENT,
            detail=_wm.media_refusal(error=str(exc)).envelope(),
        )
    except ValueError as ve:  # A client-side validation error (e.g. prompt length)
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST, detail=_wm.media_refusal(error=str(ve)).envelope()
        )
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_wm.transcription_unexpected(error=str(e)).envelope(),
        )
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    # The charge (PR-3): only the Radient rung reports one — the hub's envelope
    # carries the cost object and the executor parsed it into ``outcome.cost``.
    # The unit count is the server's metered seconds when stated; a BYO rung
    # states neither count nor amount (no audio-duration reader exists here),
    # and both stay unknown rather than zero. The provider label names the
    # vendor that actually served when the cost object omits one.
    cost = outcome.cost if isinstance(outcome.cost, RadientCost) else None
    _rung_providers = {
        AudioPath.PROVIDER_STT_ELEVENLABS.value: "elevenlabs",
        AudioPath.PROVIDER_STT_OPENAI.value: "openai",
    }
    emit_voice_spend(
        config_manager=config_manager,
        channel="stt",
        # The envelope's own provider field outranks the path label (QA round
        # 1, Q3): an amount-less older hub still names the vendor it used, and
        # the cost object — when one was stated — wins inside the builder.
        provider=outcome.provider or _rung_providers.get(outcome.path.value, ""),
        units=None,
        unit="seconds",
        cost=cost,
        session_id=str(session_id or ""),
        request_id="",
    )

    return CRUDResponse(
        status=200,
        message="Transcription created successfully",
        result=SttTranscriptionPayload(
            text=outcome.text,
            path=outcome.path,
            attempts=[
                SttAttemptPayload(path=attempt.path, outcome=attempt.outcome, detail=attempt.detail)
                for attempt in outcome.attempts
            ],
            duration_s=round(time.monotonic() - started, 3),
            cleanup=outcome.cleanup,
            raw_text=outcome.raw_text,
            raw_unavailable=outcome.raw_unavailable,
            language=outcome.language,
            removed=[
                SttRemovedSpanPayload(
                    start=span.start, end=span.end, kind=span.kind, token=span.token
                )
                for span in outcome.removed
            ],
        ),
    )
