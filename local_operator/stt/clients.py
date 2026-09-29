"""HTTP clients for the BYO speech-to-text rungs (ElevenLabs, OpenAI).

Why hand-rolled and not the provider stack: the chat provider machinery speaks
*messages*; these endpoints take an uploaded audio part and answer with text.
They are small, sync-free and independent of chat routing/failover decisions —
the cascade (``stt/cascade.py``) owns the ordering and the fall-forward, and
these clients own exactly one ``POST`` each.

Shapes verified 2026-09-28: ElevenLabs against the live API (the research
session's POC transcribed four fixtures through ``POST /v1/speech-to-text``
with ``model_id=scribe_v2`` and header ``xi-api-key``; scribe_v1 also 200)
and the OpenAI reference docs (``POST /v1/audio/transcriptions``, multipart
``file`` + ``model``, ``gpt-4o-transcribe`` current with ``whisper-1`` as the
long-lived fallback). Both base URLs are constructor arguments so tests can
point them at a local fake — the repo's established pattern.

Every failure is an :class:`~local_operator.clients._http.APIError` carrying
the upstream status and the designed error string, so the shared mapper in
``stt/errors.py`` classifies a rung failure exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import httpx

from local_operator.clients._http import APIError
from local_operator.stt import audio as stt_audio
from local_operator.stt.errors import api_error_from_httpx_response

#: The hosts. Overridable per instance for tests; the OpenAI default carries
#: the ``/v1`` the API reference uses, the ElevenLabs default does not (its
#: paths are ``/v1/...`` themselves).
ELEVENLABS_STT_BASE_URL = "https://api.elevenlabs.io"
OPENAI_STT_BASE_URL = "https://api.openai.com/v1"

#: The models. Scribe v2 is the current transcription model (the research POC
#: measured both; the daemon must not choose silently per request, so the
#: constant is the default and the caller may pin another id).
ELEVENLABS_STT_MODEL = "scribe_v2"
OPENAI_STT_MODEL = "gpt-4o-transcribe"

#: The long-lived OpenAI transcription model, used ONLY when the primary id is
#: refused as unavailable to the account (404 / ``model_not_found``). Accounts
#: that predate `gpt-4o-transcribe` access still transcribe; a caller that pins
#: a model explicitly gets exactly that model, no substitution.
OPENAI_STT_FALLBACK_MODEL = "whisper-1"

#: Default bound for one wire call. The executor always passes its own
#: remaining-budget value; standalone users get this.
DEFAULT_TRANSCRIBE_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class SttClientResult:
    """One successful transcription from one provider."""

    text: str
    #: The model id the provider actually ran (the fallback id when it fired).
    model: str
    provider: str


async def _post(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    files: dict[str, Any],
    data: dict[str, Any],
    timeout_s: float,
    label: str,
    secrets: list[str],
) -> tuple[httpx.Response, dict[str, Any]]:
    """One multipart POST, with the package's error shape on any failure."""
    try:
        response = await client.post(
            url, headers=headers, files=files, data=data, timeout=httpx.Timeout(timeout_s)
        )
    except httpx.HTTPError as exc:
        # A transport failure never reached the upstream: no status, no body;
        # the client's own text is the whole diagnostic (legacy shape).
        raise APIError(f"{label} request failed: {exc}", status_code=None) from exc
    if response.status_code >= 400:
        raise api_error_from_httpx_response(
            response,
            fallback_message=f"{label} refused the transcription request",
            secrets=secrets,
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise APIError(
            f"{label} returned a response that is not JSON (HTTP {response.status_code}).",
            status_code=response.status_code,
        ) from exc
    if not isinstance(payload, dict):
        raise APIError(
            f"{label} returned a {type(payload).__name__} where a JSON object was expected.",
            status_code=response.status_code,
        )
    return response, payload


async def _post_owned(
    client: Optional[httpx.AsyncClient],
    url: str,
    *,
    headers: dict[str, str],
    files: dict[str, Any],
    data: dict[str, Any],
    timeout_s: float,
    label: str,
    secrets: list[str],
) -> tuple[httpx.Response, dict[str, Any]]:
    """:func:`_post` on an injected client, or on one this call owns.

    The injected-client seam is how tests attach ``httpx.MockTransport``
    without a network; production call sites leave it ``None`` and pay one
    client per call (these calls are rare and already bounded).
    """
    if client is not None:
        return await _post(
            client,
            url,
            headers=headers,
            files=files,
            data=data,
            timeout_s=timeout_s,
            label=label,
            secrets=secrets,
        )
    async with httpx.AsyncClient() as owned:
        return await _post(
            owned,
            url,
            headers=headers,
            files=files,
            data=data,
            timeout_s=timeout_s,
            label=label,
            secrets=secrets,
        )


class ElevenLabsSttClient:
    """``POST /v1/speech-to-text`` with the user's own ElevenLabs key."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = (base_url or ELEVENLABS_STT_BASE_URL).rstrip("/")
        self._client = client

    async def transcribe(
        self,
        audio: bytes,
        *,
        mime: str,
        model: str | None = None,
        language: str | None = None,
        prompt: str | None = None,
        timeout_s: float = DEFAULT_TRANSCRIBE_TIMEOUT_S,
    ) -> SttClientResult:
        """Transcribe ``audio``; raise :class:`APIError` on any upstream refusal.

        ``prompt`` is accepted for the shared client interface and deliberately
        NOT sent: Scribe has no prompt field (API reference, 2026-09-28); its
        closest knob, ``transcript_edit``, is a billed post-edit and out of
        scope for v1. Refusing the whole rung over an optional hint would
        disable the fallback chain for a nicety, so the hint is simply not
        forwarded.
        """
        model_id = model or ELEVENLABS_STT_MODEL
        data: dict[str, str] = {"model_id": model_id}
        if language:
            data["language_code"] = language
        files = {"file": (stt_audio.filename_for_mime(mime), audio, mime)}
        response, payload = await _post_owned(
            self._client,
            f"{self._base_url}/v1/speech-to-text",
            headers={"xi-api-key": self._api_key},
            files=files,
            data=data,
            timeout_s=timeout_s,
            label="ElevenLabs",
            secrets=[self._api_key],
        )
        text = payload.get("text")
        if not isinstance(text, str):
            raise APIError(
                "ElevenLabs returned a response without transcript text.",
                status_code=response.status_code,
            )
        return SttClientResult(text=text, model=model_id, provider="elevenlabs")


class OpenAiSttClient:
    """``POST /v1/audio/transcriptions`` with the user's own OpenAI key."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = (base_url or OPENAI_STT_BASE_URL).rstrip("/")
        self._client = client

    async def transcribe(
        self,
        audio: bytes,
        *,
        mime: str,
        model: str | None = None,
        language: str | None = None,
        prompt: str | None = None,
        timeout_s: float = DEFAULT_TRANSCRIBE_TIMEOUT_S,
    ) -> SttClientResult:
        """Transcribe ``audio``, with one bounded model fallback.

        When ``model`` is None the default id is tried first; if the account
        cannot use it (404 / ``model_not_found``) the fallback constant is
        tried once. A pinned ``model`` is used as given — the caller asked for
        that model by name, and silently running another would be worse than
        the refusal.
        """
        candidates = (model,) if model else (OPENAI_STT_MODEL, OPENAI_STT_FALLBACK_MODEL)
        last: APIError | None = None
        for position, model_id in enumerate(candidates):
            data: dict[str, str] = {"model": model_id, "response_format": "json"}
            if language:
                data["language"] = language
            if prompt:
                data["prompt"] = prompt
            files = {"file": (stt_audio.filename_for_mime(mime), audio, mime)}
            try:
                response, payload = await _post_owned(
                    self._client,
                    f"{self._base_url}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    files=files,
                    data=data,
                    timeout_s=timeout_s,
                    label="OpenAI",
                    secrets=[self._api_key],
                )
            except APIError as exc:
                if model is None and position < len(candidates) - 1 and _model_unavailable(exc):
                    last = exc
                    continue
                raise
            text = payload.get("text")
            if not isinstance(text, str):
                raise APIError(
                    "OpenAI returned a response without transcript text.",
                    status_code=response.status_code,
                )
            return SttClientResult(text=text, model=model_id, provider="openai")
        # Only reachable when every candidate was skipped as model-unavailable.
        raise last or APIError("OpenAI transcription failed on every model candidate.")


def _model_unavailable(exc: APIError) -> bool:
    """Whether a refusal says "this account cannot use this model".

    OpenAI answers 404 with ``model_not_found`` for an id the account cannot
    reach; anything else (401, 429, 500) is a failure the fallback id would
    inherit, and retrying it would only spend a second call to reach the same
    answer.
    """
    return exc.status_code == 404 or exc.code == "model_not_found"


__all__ = [
    "DEFAULT_TRANSCRIBE_TIMEOUT_S",
    "ELEVENLABS_STT_BASE_URL",
    "ELEVENLABS_STT_MODEL",
    "OPENAI_STT_BASE_URL",
    "OPENAI_STT_FALLBACK_MODEL",
    "OPENAI_STT_MODEL",
    "ElevenLabsSttClient",
    "OpenAiSttClient",
    "SttClientResult",
]
