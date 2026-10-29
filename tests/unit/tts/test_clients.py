"""The daemon's BYO ElevenLabs leg: preferred format, one-shot tier fallback.

The tool's BYOK rung and this leg share :mod:`local_operator.tts.output_format`,
so a plan refusal gets one answer on both surfaces. These tests pin this leg's
side of that contract against ``httpx.MockTransport`` — no network, no key.
"""

from __future__ import annotations

import httpx
import pytest

from local_operator.clients._http import APIError
from local_operator.tts import output_format as audio_format
from local_operator.tts.adapters import Params
from local_operator.tts.clients import ElevenLabsTtsClient

PLAN_REFUSAL = {
    "detail": {
        "type": "authorization_error",
        "code": "feature_not_available",
        "param": "output_format",
        "message": "This output format is only available on the Creator plan.",
    }
}
PARAMS = Params(voice_id="OZ0L6eISlOejga3XjDFt")


@pytest.fixture(autouse=True)
def _fresh_format_memo():
    """The 192->128 memo is process-wide by design; no test may inherit it."""
    audio_format.reset_caps()
    yield
    audio_format.reset_caps()


def _formats(seen: list[httpx.Request]) -> list[str]:
    return [request.url.params["output_format"] for request in seen]


@pytest.mark.asyncio
async def test_the_byo_leg_asks_for_the_preferred_format() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"mp3", headers={"content-type": "audio/mpeg"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await ElevenLabsTtsClient("el-user", client=http).synthesize(
            "hello", params=PARAMS
        )
    finally:
        await http.aclose()

    assert _formats(seen) == ["mp3_44100_192"]
    assert result.audio == b"mp3"


@pytest.mark.asyncio
async def test_the_byo_leg_drops_to_128_once_and_remembers_it() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.params["output_format"] == "mp3_44100_192":
            return httpx.Response(403, json=PLAN_REFUSAL)
        return httpx.Response(200, content=b"mp3", headers={"content-type": "audio/mpeg"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ElevenLabsTtsClient("el-user", client=http)
    try:
        first = await client.synthesize("hello", params=PARAMS)
        seen.clear()
        second = await client.synthesize("again", params=PARAMS)
    finally:
        await http.aclose()

    assert first.audio == second.audio == b"mp3"
    # The first call measured the cap; the second does not repeat the refusal.
    assert _formats(seen) == ["mp3_44100_128"]


@pytest.mark.asyncio
async def test_the_byo_leg_never_retries_an_auth_refusal() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        # Even a body naming output_format is not a tier refusal on a 401.
        return httpx.Response(401, json=PLAN_REFUSAL)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(APIError) as caught:
            await ElevenLabsTtsClient("el-user", client=http).synthesize("hello", params=PARAMS)
    finally:
        await http.aclose()

    assert _formats(seen) == ["mp3_44100_192"]
    assert caught.value.status_code == 401
