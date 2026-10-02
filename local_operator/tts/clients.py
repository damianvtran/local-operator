"""HTTP clients for the BYO text-to-speech rungs (ElevenLabs, OpenAI).

Why hand-rolled and not the chat provider stack: the chat machinery speaks
*messages*; these endpoints take text and answer with audio bytes. Shape,
headers and body are copied from the hub's own vendor clients
(``internal/clients/elevenlabs.go`` and ``internal/clients/openai.go``) rather
than re-derived, so the daemon's BYO leg sends what the hub's leg sends — that
is what makes the vendored map's mapping valid on both.

Both base URLs are constructor arguments so tests can point them at a local
fake — the repo's established pattern (``stt/clients.py``).

Every failure is an :class:`~local_operator.clients._http.APIError` carrying
the upstream status and the designed error string. The mapper is the SHARED
one from :mod:`local_operator.stt.errors`: it is provider-agnostic, and a
refusal classified twice is a refusal two surfaces can describe differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import httpx

from local_operator.clients._http import APIError
from local_operator.stt.errors import api_error_from_httpx_response
from local_operator.tts.adapters import Params

#: The hosts. Overridable per instance for tests; the OpenAI default carries
#: the ``/v1`` the API reference uses, the ElevenLabs default does not (its
#: paths are ``/v1/...`` themselves).
ELEVENLABS_TTS_BASE_URL = "https://api.elevenlabs.io"
OPENAI_TTS_BASE_URL = "https://api.openai.com/v1"

#: The ElevenLabs model the BYO leg runs. ``eleven_multilingual_v2`` is the
#: hub's own default (docs/SPEECH.md: it pronounces foreign words inside mixed
#: text, which is the delivery property this feature exists for) and the map
#: states language_code is honoured only on ``eleven_flash_v2_5`` — so a caller
#: that pins a language gets the note rather than a silent drop.
ELEVENLABS_TTS_MODEL = "eleven_multilingual_v2"

#: Default bound for one wire call.
DEFAULT_SYNTHESIZE_TIMEOUT_S = 60.0

#: The audio container both vendors serve on this path.
TTS_MEDIA_TYPE = "audio/mpeg"


@dataclass(frozen=True)
class TtsClientResult:
    """One successful synthesis from one provider."""

    audio: bytes
    model: str
    provider: str
    media_type: str = TTS_MEDIA_TYPE


async def _post_audio(
    client: Optional[httpx.AsyncClient],
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    params: dict[str, str],
    timeout_s: float,
    label: str,
    secrets: list[str],
) -> bytes:
    """One JSON POST answering with audio bytes, with the shared error shape.

    The injected-client seam is how tests attach ``httpx.MockTransport``
    without a network; production call sites leave it ``None`` and pay one
    client per call (these calls are rare and already bounded).
    """
    owned: Optional[httpx.AsyncClient] = None
    if client is None:
        owned = httpx.AsyncClient()
        client = owned
    try:
        try:
            response = await client.post(
                url,
                headers=headers,
                json=payload,
                params=params,
                timeout=httpx.Timeout(timeout_s),
            )
        except httpx.HTTPError as exc:
            # A transport failure never reached the upstream: no status, no
            # body; the client's own text is the whole diagnostic.
            raise APIError(f"{label} request failed: {exc}", status_code=None) from exc
        if response.status_code >= 400:
            raise api_error_from_httpx_response(
                response, fallback_message=f"{label} refused the request", secrets=secrets
            )
        return response.content
    finally:
        if owned is not None:
            await owned.aclose()


class ElevenLabsTtsClient:
    """``POST /v1/text-to-speech/{voice_id}`` with the user's own ElevenLabs key."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = (base_url or ELEVENLABS_TTS_BASE_URL).rstrip("/")
        self._client = client

    async def synthesize(
        self,
        text: str,
        *,
        params: Params,
        model: str | None = None,
        output_format: str | None = None,
        timeout_s: float = DEFAULT_SYNTHESIZE_TIMEOUT_S,
    ) -> TtsClientResult:
        """Synthesize ``text`` onto the mapped ``params``.

        ``params`` comes from the vendored map's ElevenLabs executor, so this
        client owns transport and nothing else — it does not decide stability,
        clamp speed, or choose a voice. The platform constants it does need
        (``similarity_boost``, ``use_speaker_boost``, the output format) come
        from the map's ``fixed`` block, which is the one place the hub's own
        client reads them from too.
        """
        model_id = model or ELEVENLABS_TTS_MODEL
        voice_settings: dict[str, Any] = {
            "stability": params.stability if params.stability is not None else DEFAULT_STABILITY,
            "similarity_boost": SIMILARITY_BOOST,
            "use_speaker_boost": USE_SPEAKER_BOOST,
        }
        if params.speed is not None:
            voice_settings["speed"] = params.speed
        payload: dict[str, Any] = {
            "text": text,
            "model_id": model_id,
            "voice_settings": voice_settings,
        }
        if params.language_code:
            payload["language_code"] = params.language_code
        # The voice id is a path segment and the output format a query value,
        # so httpx escapes both: a value that reached here with a slash or a
        # dot-segment must not address another endpoint on the vendor's API
        # under our key (the hub validates the same shape from the map's
        # `fixed.elevenlabs.voice_id_pattern`).
        audio = await _post_audio(
            self._client,
            f"{self._base_url}/v1/text-to-speech/{params.voice_id}",
            headers={
                "xi-api-key": self._api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            payload=payload,
            params={"output_format": output_format or OUTPUT_FORMAT},
            timeout_s=timeout_s,
            label="ElevenLabs",
            secrets=[self._api_key],
        )
        return TtsClientResult(audio=audio, model=model_id, provider="elevenlabs")


class OpenAiTtsClient:
    """``POST /v1/audio/speech`` with the user's own OpenAI API key."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = (base_url or OPENAI_TTS_BASE_URL).rstrip("/")
        self._client = client

    async def synthesize(
        self,
        text: str,
        *,
        params: Params,
        model: str | None = None,
        response_format: str = "mp3",
        timeout_s: float = DEFAULT_SYNTHESIZE_TIMEOUT_S,
    ) -> TtsClientResult:
        """Synthesize ``text`` onto the mapped ``params``.

        ``params.instructions`` is the text with the map's emulation sentences
        already composed in (tone, expressiveness, language, accent) — the
        client forwards it verbatim, because composing it a second time here
        would put a sentence the caller never asked for on a paid call.
        """
        model_id = model or OPENAI_TTS_MODEL
        payload: dict[str, Any] = {
            "model": model_id,
            "input": text,
            "voice": params.voice_id,
            "response_format": response_format,
        }
        if params.instructions:
            payload["instructions"] = params.instructions
        if params.speed is not None:
            payload["speed"] = params.speed
        audio = await _post_audio(
            self._client,
            f"{self._base_url}/audio/speech",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
            params={},
            timeout_s=timeout_s,
            label="OpenAI",
            secrets=[self._api_key],
        )
        return TtsClientResult(audio=audio, model=model_id, provider="openai")


#: The platform constants, read from the artifact rather than restated: a map
#: bump moves the daemon and the hub's executors together, which is the whole
#: point of the map being data.
def _map_fixed(provider: str) -> dict[str, Any]:
    from local_operator.tts.adapters import load_map

    return load_map().fixed.get(provider) or {}


#: The OpenAI TTS model the map pins (``fixed.openai.model``).
OPENAI_TTS_MODEL = str(_map_fixed("openai").get("model") or "gpt-4o-mini-tts")

#: ElevenLabs' fixed request values (``fixed.elevenlabs``). The stability floor
#: is only the fallback for a descriptor that carried no expressiveness: the
#: map's own ``medium`` row is 0.5, which is also the pre-descriptor constant.
SIMILARITY_BOOST = float(_map_fixed("elevenlabs").get("similarity_boost", 0.75))
USE_SPEAKER_BOOST = bool(_map_fixed("elevenlabs").get("use_speaker_boost", True))
OUTPUT_FORMAT = str(_map_fixed("elevenlabs").get("output_format") or "mp3_44100_128")
DEFAULT_STABILITY = 0.5

__all__ = [
    "DEFAULT_STABILITY",
    "DEFAULT_SYNTHESIZE_TIMEOUT_S",
    "ELEVENLABS_TTS_BASE_URL",
    "ELEVENLABS_TTS_MODEL",
    "OPENAI_TTS_BASE_URL",
    "OPENAI_TTS_MODEL",
    "OUTPUT_FORMAT",
    "SIMILARITY_BOOST",
    "TTS_MEDIA_TYPE",
    "USE_SPEAKER_BOOST",
    "ElevenLabsTtsClient",
    "OpenAiTtsClient",
    "TtsClientResult",
]
