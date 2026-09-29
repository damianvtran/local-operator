"""The forked, bounded model-transcription sidecar.

OQ-1 (manager decisions, 2026-09-28): on the ``model_audio_sidecar`` path the
STT chain is empty *by construction*, so the sidecar does NOT re-run it — it
asks the audio-capable model itself for a transcript, as a bounded second wire
call through the session's own provider/model with a transcription instruction,
and NEVER through the resolver (no recursion into the cascade that dispatched
it). When even that is not possible for the model's wire — no OpenAI-compatible
chat endpoint, a captured format the wire refuses, no credential — the record
says ``unavailable`` with the reason, which is a status, not a send failure.

The call shape lives in exactly ONE function (:func:`_model_transcript`) on
purpose: the research session that proved it live (25/25 calls HTTP 200 across
OpenRouter and Radient, 2026-09-28; wav accepted everywhere, mp3 on three
models) may refine it, and a second copy would be the thing that drifts.

The fork never races a turn and never raises into one: the task is registered
through the session's own tracked-spawn seam (``Session._spawn_background``),
which :meth:`Session.dispose` cancels and awaits — so "cancelled on dispose" is
the session's existing guarantee rather than a second registry beside it. Every
outcome is written to the durable transcript as an ``stt_transcript_v1`` custom
entry; **nothing renders it**, anywhere.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import Any, Coroutine, Literal, Optional, Protocol

import httpx

from local_operator.clients._http import APIError
from local_operator.stt import AudioPath
from local_operator.stt import audio as stt_audio
from local_operator.stt.errors import api_error_from_httpx_response

logger = logging.getLogger(__name__)

#: The whole sidecar's model-work bound, in seconds.
SIDECAR_TIMEOUT_S = 120.0

#: One wire call's bound. The single call in v1 means this normally fires
#: first and the overall bound is the backstop around everything the task adds.
SIDECAR_ATTEMPT_TIMEOUT_S = 60.0

#: The durable entry type. Add-only custom entry; a message row cannot be
#: rewritten post-hoc and the journal is append-only, so the record of what the
#: sidecar heard arrives as its own row.
STT_TRANSCRIPT_CUSTOM_TYPE = "stt_transcript_v1"

#: The transcription instruction, verbatim from the research POC that proved
#: the call shape (so the accuracy findings attach to the exact prompt).
TRANSCRIBE_INSTRUCTION = (
    "Transcribe the audio recording exactly as spoken. Output only the transcription "
    "text, with no commentary, labels, or quotation marks."
)

SidecarStatus = Literal["ok", "failed", "unavailable", "timeout"]


class ModelTranscriptUnavailable(RuntimeError):
    """The model's wire cannot produce a transcript at all (status: unavailable).

    Raised for the states where TRYING is impossible or pointless — an unknown
    provider, a non-OpenAI-compatible wire (v1 sends audio only over that one),
    a captured format the wire refuses, no credential, no model on the session.
    A call that was made and failed is NOT this: it is a ``failed`` record.
    """


class SidecarHost(Protocol):
    """The slice of ``Session`` the sidecar uses.

    Declared rather than imported so this module stays importable without the
    session machinery (which is what lets the server, the session runtime and
    tests share it), and so the coupling is written down: the sidecar needs the
    session's model, its transcript, and its tracked-spawn seam — no more.
    """

    @property
    def model(self) -> Any: ...

    @property
    def transcript(self) -> Any: ...

    def _spawn_background(self, coro: Coroutine[Any, Any, Any]) -> "asyncio.Task[Any] | None": ...


def fork_audio_sidecar(
    session: SidecarHost,
    *,
    message_id: str,
    audio: Any,
    config_dir: Any,
    store: Any,
) -> None:
    """Start the sidecar for one durable user message. Fire-and-forget.

    Called immediately after the message is durable (phase 1 wires this into
    ``Session.prompt``), so the sidecar never races the admission ack. The task
    is tracked by the session and cancelled by its dispose; it is never awaited
    by the turn and catches every failure of its own.

    ``store`` may be ``None``: the task then builds (and closes) its own store
    from ``config_dir``, matching the resolver's ownership rule.
    """
    coro = _run_audio_sidecar(
        session,
        message_id=message_id,
        audio=audio,
        config_dir=config_dir,
        store=store,
    )
    try:
        session._spawn_background(coro)
    except Exception:
        # Starting is best-effort too: a host whose spawn refused (no running
        # loop, a torn-down session) must not fail the message that was already
        # durable. The coroutine is closed here so it is not reported as
        # "never awaited".
        coro.close()
        logger.warning("the audio sidecar could not be started", exc_info=True)


async def _run_audio_sidecar(
    session: SidecarHost,
    *,
    message_id: str,
    audio: Any,
    config_dir: Any,
    store: Any,
) -> None:
    """One sidecar run: bounded model call, then one durable record.

    Cancellation (dispose) propagates — the session is going away and a
    half-written record has no reader — while every other failure is turned
    into a status.
    """
    owned_store = store is None
    if owned_store:
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(
            (config_dir / "auth.db") if config_dir is not None else None,
            config_dir=config_dir,
        )

    status: SidecarStatus
    text: str | None = None
    error: str | None = None
    path: str | None = None
    try:
        try:
            transcript_text = await asyncio.wait_for(
                _model_transcript(
                    session,
                    audio,
                    store=store,
                    timeout_s=SIDECAR_ATTEMPT_TIMEOUT_S,
                ),
                timeout=SIDECAR_TIMEOUT_S,
            )
        except asyncio.TimeoutError as exc:
            status = "timeout"
            error = str(exc) or (
                f"The model did not return a transcript within " f"{SIDECAR_TIMEOUT_S:.0f} s."
            )
        except ModelTranscriptUnavailable as exc:
            status = "unavailable"
            error = str(exc)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Includes APIError (the call was made and the wire refused) and
            # anything unforeseen; an empty str() would lose the only clue, so
            # the type name backs it up.
            status = "failed"
            error = str(exc) or type(exc).__name__
        else:
            status = "ok"
            text = transcript_text
            path = str(AudioPath.MODEL_AUDIO_SIDECAR)
        await _append_record(
            session,
            {
                "message_id": message_id,
                "status": status,
                "path": path,
                "text": text,
                "error": error,
                "at": time.time(),
            },
        )
        if status in ("failed", "timeout"):
            # The record says why for whoever reads the transcript; the LOG
            # says it for the operator whose daemon produced it, which is the
            # half design §4c and both round-1 briefs promise ("written to the
            # transcript and logged at warning") — v1 renders the record
            # nowhere, so a failed sidecar that only whispers into a JSONL
            # file is invisible in the one place failures are watched (agent
            # review round 1, m1 / QA Q2: an unreachable sidecar wire wrote
            # 'failed' and left zero WARNING lines while the capture control
            # proved the logger was live in that process). "ok" and
            # "unavailable" stay silent on purpose: the first is the happy
            # path, and the second is the deterministic honest answer on a
            # machine with no route for the model call — warning on it would
            # train the operator to ignore the line that matters.
            logger.warning("audio sidecar %s for message %s: %s", status, message_id, error)
    finally:
        if owned_store:
            store.close()


async def _append_record(session: SidecarHost, record: dict[str, Any]) -> None:
    """Write the record, never raising into the caller.

    The transcript is the v1 durable vehicle (OQ-5). A reduced host without a
    transcript, or a store that refuses the write, must not fail the sidecar:
    the status exists, the log keeps the diagnostic, and the turn is long gone.
    """
    transcript = getattr(session, "transcript", None)
    if transcript is None:
        logger.warning("audio sidecar: session has no transcript; record dropped")
        return
    try:
        await transcript.append_custom(STT_TRANSCRIPT_CUSTOM_TYPE, record)
    except Exception:
        logger.warning("audio sidecar: transcript record could not be written", exc_info=True)


async def _model_transcript(
    session: SidecarHost,
    audio: Any,
    *,
    store: Any,
    timeout_s: float = SIDECAR_ATTEMPT_TIMEOUT_S,
    client: Optional[httpx.AsyncClient] = None,
) -> str:
    """The ONE model-transcription call shape (OQ-1).

    OpenAI-compatible chat completions, one ``input_audio`` part plus the
    transcription instruction, sent with the session provider's own credential
    through the provider's own base URL. Raises
    :class:`ModelTranscriptUnavailable` when the wire cannot carry the call,
    and :class:`APIError` when it carried it and the upstream refused.
    """
    model = getattr(session, "model", None)
    provider = str(getattr(model, "provider", "") or "")
    model_id = str(getattr(model, "model_id", "") or "")
    if not provider or not model_id:
        raise ModelTranscriptUnavailable(
            "The session has no selected model to ask for a transcript."
        )

    from local_operator.providers.registry import get_provider_definition

    definition = get_provider_definition(provider)
    if definition is None:
        raise ModelTranscriptUnavailable(
            f"The selected model's provider ({provider}) is unknown, so the audio "
            f"cannot be sent to it."
        )
    if definition.wire != "openai-compat":
        raise ModelTranscriptUnavailable(
            f"The {provider} wire cannot take audio input in v1 (only the "
            f"OpenAI-compatible chat wire can)."
        )
    base_url = _provider_base_url(definition)
    if not base_url:
        raise ModelTranscriptUnavailable(
            f"No base URL is configured for {provider}, so the audio cannot be sent."
        )

    mime = str(getattr(audio, "mime_type", "") or "audio/wav")
    wire_format = stt_audio.format_for_model_wire(mime)
    if wire_format is None:
        raise ModelTranscriptUnavailable(
            f"The captured format ({mime}) is not one the model wire takes " f"(wav or mp3)."
        )

    data_b64 = getattr(audio, "data", "")
    try:
        audio_bytes = base64.b64decode(str(data_b64), validate=True)
    except Exception as exc:
        raise ModelTranscriptUnavailable(
            f"The audio block's payload is not valid base64 ({exc})."
        ) from exc

    api_key = await store.get_api_key(
        provider, getattr(session, "session_id", None), read_only=True
    )
    if not api_key:
        raise ModelTranscriptUnavailable(
            f"No {provider} credential is stored, so the audio cannot be sent."
        )

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TRANSCRIBE_INSTRUCTION},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(audio_bytes).decode("ascii"),
                            "format": wire_format,
                        },
                    },
                ],
            }
        ],
    }
    url = f"{base_url.rstrip('/')}/chat/completions"

    async def _post() -> httpx.Response:
        if client is not None:
            return await client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=httpx.Timeout(timeout_s),
            )
        async with httpx.AsyncClient() as owned:
            return await owned.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=httpx.Timeout(timeout_s),
            )

    try:
        response = await asyncio.wait_for(_post(), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        # The attempt bound is named here, where its value is known; the outer
        # sidecar bound re-raises with its own message when IT is the one that
        # fired, so the record always names the timeout that actually hit.
        raise TimeoutError(
            f"The model did not return a transcript within {timeout_s:.0f} s."
        ) from exc
    except httpx.TimeoutException as exc:
        # A wire timeout IS the timeout status, not a failure: mapping it to
        # "failed" would hide the one outcome the record exists to distinguish.
        raise TimeoutError(f"The {provider} model call timed out: {exc}") from exc
    except httpx.HTTPError as exc:
        raise APIError(f"The {provider} model call failed: {exc}", status_code=None) from exc
    if response.status_code >= 400:
        raise api_error_from_httpx_response(
            response,
            fallback_message=f"The {provider} model refused the transcription request",
            secrets=[api_key],
        )
    try:
        parsed = response.json()
    except ValueError as exc:
        raise APIError(
            f"The {provider} model returned a response that is not JSON "
            f"(HTTP {response.status_code}).",
            status_code=response.status_code,
        ) from exc
    choices = parsed.get("choices") if isinstance(parsed, dict) else None
    content: Any = None
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise APIError(
            f"The {provider} model returned no transcript text.",
            status_code=response.status_code,
        )
    return content.strip()


def _provider_base_url(definition: Any) -> str:
    """The base URL for a chat call to ``definition``'s provider.

    Radient's host is the one configurable one (staging vs production), and it
    resolves through the same helper every other Radient surface asks — see
    ``providers/key_check._base_url``, whose rule this mirrors.
    """
    if definition.id == "radient":
        from local_operator.env import resolve_radient_api_base_url

        return resolve_radient_api_base_url()
    return definition.base_url or ""


__all__ = [
    "SIDECAR_ATTEMPT_TIMEOUT_S",
    "SIDECAR_TIMEOUT_S",
    "STT_TRANSCRIPT_CUSTOM_TYPE",
    "ModelTranscriptUnavailable",
    "SidecarHost",
    "fork_audio_sidecar",
]
