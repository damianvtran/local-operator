"""Wire shapes of the BYO STT clients, over httpx.MockTransport."""

from __future__ import annotations

import re

import httpx
import pytest

from local_operator.clients._http import APIError
from local_operator.stt import clients
from local_operator.stt.cleanup import RAW_UNAVAILABLE_PROVIDER_NO_VERBATIM


def _multipart(request: httpx.Request) -> bytes:
    return request.read()


@pytest.mark.asyncio
async def test_elevenlabs_success_shape() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["xi-api-key"] = request.headers.get("xi-api-key")
        seen["body"] = _multipart(request)
        return httpx.Response(200, json={"text": "hello world", "language_code": "en"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = clients.ElevenLabsSttClient("el-secret", client=http)
        result = await client.transcribe(b"RIFFdata", mime="audio/wav")

    assert result.text == "hello world"
    assert result.model == clients.ELEVENLABS_STT_MODEL
    assert result.provider == "elevenlabs"
    assert seen["method"] == "POST"
    assert str(seen["url"]).endswith("/v1/speech-to-text")
    assert seen["xi-api-key"] == "el-secret"
    body = seen["body"]
    assert isinstance(body, bytes)
    assert b'name="model_id"' in body and b"scribe_v2" in body
    assert b'name="file"' in body and b'filename="audio.wav"' in body
    assert b"RIFFdata" in body
    # Scribe has no prompt field; the hint must not be invented onto the wire.
    assert b"prompt" not in body


@pytest.mark.asyncio
async def test_elevenlabs_language_goes_as_language_code() -> None:
    seen: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = _multipart(request)
        return httpx.Response(200, json={"text": "x"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = clients.ElevenLabsSttClient("k", client=http)
        await client.transcribe(b"x", mime="audio/wav", language="en")

    assert b'name="language_code"' in seen["body"]
    assert b"\r\n\r\nen\r\n" in seen["body"]


@pytest.mark.asyncio
async def test_elevenlabs_refusal_is_an_apierror_with_the_key_scrubbed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key el-secret"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = clients.ElevenLabsSttClient("el-secret", client=http)
        with pytest.raises(APIError) as caught:
            await client.transcribe(b"x", mime="audio/wav")

    assert caught.value.status_code == 401
    assert "el-secret" not in str(caught.value)


@pytest.mark.asyncio
async def test_elevenlabs_cleanup_fields_and_provenance() -> None:
    """``auto``/``provider`` are two explicit wire fields, not a prompt hint."""
    seen: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = _multipart(request)
        return httpx.Response(
            200, json={"text": "So I think we should ship.", "language_code": "eng"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = clients.ElevenLabsSttClient("k", client=http)
        result = await client.transcribe(
            b"x", mime="audio/wav", no_verbatim=True, tag_audio_events=False
        )

    body = seen["body"]
    assert b'name="no_verbatim"' in body and b"\r\n\r\ntrue\r\n" in body
    assert b'name="tag_audio_events"' in body and b"\r\n\r\nfalse\r\n" in body
    assert result.cleanup == "elevenlabs.no_verbatim"
    assert result.raw_text is None
    assert result.raw_unavailable == RAW_UNAVAILABLE_PROVIDER_NO_VERBATIM
    assert result.language == "eng"


@pytest.mark.asyncio
async def test_elevenlabs_verbatim_sends_no_cleanup_fields_and_preserves_raw() -> None:
    """``cleanup=off`` must be byte-identical to the pre-cleanup request.

    The fields are optional in the API; the verbatim path proves we do not send
    them at all rather than sending ``false`` (which would still be a wire
    change and could inherit a provider-side default later).
    """
    seen: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = _multipart(request)
        return httpx.Response(200, json={"text": "Um, so I think.", "language_code": "eng"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = clients.ElevenLabsSttClient("k", client=http)
        result = await client.transcribe(b"x", mime="audio/wav", language="en")

    body = seen["body"]
    assert b"no_verbatim" not in body
    assert b"tag_audio_events" not in body
    # The complete field set is the pre-cleanup one: model, optional language,
    # file. No cleanup field, not even a ``false``.
    names = sorted(match.decode() for match in re.findall(rb';\s*name="([^"]+)"', body))
    assert names == ["file", "language_code", "model_id"]
    assert result.text == "Um, so I think."
    assert result.raw_text == "Um, so I think."
    assert result.raw_unavailable == ""
    assert result.cleanup == "none"


@pytest.mark.asyncio
async def test_elevenlabs_no_verbatim_is_not_sent_for_a_pinned_older_model() -> None:
    """The provider documents no-verbatim on scribe_v2 only; do not 422 a pin."""
    seen: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = _multipart(request)
        return httpx.Response(200, json={"text": "raw text"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = clients.ElevenLabsSttClient("k", client=http)
        result = await client.transcribe(
            b"x", mime="audio/wav", model="scribe_v1", no_verbatim=True
        )

    assert b"no_verbatim" not in seen["body"]
    assert result.raw_text == "raw text"
    assert result.cleanup == "none"


@pytest.mark.asyncio
async def test_openai_success_shape() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        seen["body"] = _multipart(request)
        return httpx.Response(200, json={"text": "openai text"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = clients.OpenAiSttClient("oai-secret", client=http)
        result = await client.transcribe(b"MP3DATA", mime="audio/mpeg")

    assert result.text == "openai text"
    assert result.model == clients.OPENAI_STT_MODEL
    assert result.provider == "openai"
    assert result.raw_text == "openai text"
    assert result.cleanup == "none"
    assert str(seen["url"]).endswith("/v1/audio/transcriptions")
    assert seen["authorization"] == "Bearer oai-secret"
    body = seen["body"]
    assert isinstance(body, bytes)
    assert b'name="model"' in body and b"gpt-4o-transcribe" in body
    assert b'name="response_format"' in body and b"json" in body
    assert b'filename="audio.mp3"' in body


@pytest.mark.asyncio
async def test_openai_falls_back_to_whisper_when_the_model_is_unavailable() -> None:
    models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = _multipart(request)
        marker = b'name="model"\r\n\r\n'
        start = body.index(marker) + len(marker)
        models.append(body[start : body.index(b"\r\n", start)].decode())
        if len(models) == 1:
            return httpx.Response(
                404,
                json={
                    "error": "The model `gpt-4o-transcribe` does not exist",
                    "code": "model_not_found",
                },
            )
        return httpx.Response(200, json={"text": "fallback text"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = clients.OpenAiSttClient("k", client=http)
        result = await client.transcribe(b"x", mime="audio/wav")

    assert models == [clients.OPENAI_STT_MODEL, clients.OPENAI_STT_FALLBACK_MODEL]
    assert result.model == clients.OPENAI_STT_FALLBACK_MODEL
    assert result.text == "fallback text"


@pytest.mark.asyncio
async def test_openai_pinned_model_is_never_substituted() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404, json={"error": "nope", "code": "model_not_found"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = clients.OpenAiSttClient("k", client=http)
        with pytest.raises(APIError) as caught:
            await client.transcribe(b"x", mime="audio/wav", model="custom-model")

    assert caught.value.status_code == 404
    assert len(calls) == 1  # the caller pinned a model; no fallback id is tried


@pytest.mark.asyncio
async def test_openai_language_and_prompt_are_forwarded() -> None:
    seen: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = _multipart(request)
        return httpx.Response(200, json={"text": "x"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = clients.OpenAiSttClient("k", client=http)
        await client.transcribe(b"x", mime="audio/wav", language="en", prompt="Names, please")

    assert b'name="language"' in seen["body"] and b"\r\n\r\nen\r\n" in seen["body"]
    assert b'name="prompt"' in seen["body"] and b"Names, please" in seen["body"]


@pytest.mark.asyncio
async def test_transport_failure_keeps_the_client_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = clients.OpenAiSttClient("k", client=http)
        with pytest.raises(APIError) as caught:
            await client.transcribe(b"x", mime="audio/wav")

    assert caught.value.status_code is None
    assert "connection refused" in str(caught.value)


@pytest.mark.asyncio
async def test_a_non_json_success_body_is_refused() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = clients.ElevenLabsSttClient("k", client=http)
        with pytest.raises(APIError) as caught:
            await client.transcribe(b"x", mime="audio/wav")

    assert "not JSON" in str(caught.value)


@pytest.mark.asyncio
async def test_a_success_body_without_text_is_refused() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"language_code": "en"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = clients.ElevenLabsSttClient("k", client=http)
        with pytest.raises(APIError) as caught:
            await client.transcribe(b"x", mime="audio/wav")

    assert "without transcript text" in str(caught.value)


@pytest.mark.asyncio
async def test_base_urls_are_overridable_for_fakes() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"text": "x"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = clients.OpenAiSttClient("k", base_url="http://127.0.0.1:9/v1", client=http)
        await client.transcribe(b"x", mime="audio/wav")

    assert seen[0].startswith("http://127.0.0.1:9/v1/audio/transcriptions")


def test_the_fallback_constant_is_whisper_one() -> None:
    """A pin with a reason: the fallback id is what accounts without the
    newer model's access still transcribe through, so renaming it silently
    would move those accounts to a 404."""
    assert clients.OPENAI_STT_FALLBACK_MODEL == "whisper-1"
    assert clients.OPENAI_STT_MODEL == "gpt-4o-transcribe"
    assert clients.ELEVENLABS_STT_MODEL == "scribe_v2"
