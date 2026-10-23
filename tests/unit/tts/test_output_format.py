"""The shared output-format helper: preference, classification and memo invariants.

These live beside :mod:`local_operator.tts.output_format` rather than in one
caller's suite because both call sites — the audio tool's BYOK rung and the
speech daemon's BYO leg — depend on exactly these properties, and a second
copy of "what counts as a plan refusal" is the drift this module exists to
prevent.
"""

from __future__ import annotations

import json
import time

import pytest

from local_operator.clients._http import APIError
from local_operator.tts import output_format as audio_format


@pytest.fixture(autouse=True)
def _fresh_memo():
    """The 192->128 memo is process-wide by design; no test may inherit it."""
    audio_format.reset_caps()
    yield
    audio_format.reset_caps()


def _error(status: int, detail: dict) -> APIError:
    return APIError("refused", status_code=status, body=json.dumps({"detail": detail}))


def _plan_refusal() -> APIError:
    return _error(
        403,
        {
            "type": "authorization_error",
            "code": "feature_not_available",
            "param": "output_format",
            "message": "This output format is only available on the Creator plan.",
        },
    )


def _cap_key(api_key: str = "k") -> str:
    return audio_format.cap_key(api_key, endpoint="probe", model="probe")


@pytest.mark.asyncio
async def test_the_helper_names_the_preferred_format_when_the_caller_omits_it() -> None:
    """m1: a caller that forgets the parameter cannot inherit the vendor's 128."""
    seen: list[dict[str, str]] = []

    async def send(params: dict[str, str]) -> str:
        seen.append(dict(params))
        return "ok"

    answer = await audio_format.post_with_format_fallback(
        send, api_key="k", endpoint="probe", model="probe", params={}
    )

    assert seen == [{"output_format": "mp3_44100_192"}]
    assert answer.value == "ok"
    assert answer.output_format == "mp3_44100_192"


@pytest.mark.asyncio
async def test_the_helper_reports_the_format_actually_used_on_a_fallback() -> None:
    seen: list[str] = []

    async def send(params: dict[str, str]) -> str:
        seen.append(params["output_format"])
        if params["output_format"] == audio_format.PREFERRED_OUTPUT_FORMAT:
            raise _plan_refusal()
        return "audio"

    answer = await audio_format.post_with_format_fallback(
        send, api_key="k", endpoint="probe", model="probe", params={}
    )

    assert seen == ["mp3_44100_192", "mp3_44100_128"]
    assert answer.value == "audio"
    assert answer.output_format == "mp3_44100_128"
    assert audio_format.bitrate_kbps_for(answer.output_format) == 128


@pytest.mark.asyncio
async def test_an_explicit_other_format_is_sent_as_is_and_never_retried() -> None:
    """n2: the fallback belongs to our preference, not to a caller's token."""
    seen: list[str] = []

    async def send(params: dict[str, str]) -> str:
        seen.append(params["output_format"])
        raise _plan_refusal()

    with pytest.raises(APIError):
        await audio_format.post_with_format_fallback(
            send,
            api_key="k",
            endpoint="probe",
            model="probe",
            params={audio_format.OUTPUT_FORMAT_PARAM: "mp3_22050_32"},
        )

    assert seen == ["mp3_22050_32"]


def test_a_retried_parameter_refusal_is_not_a_learned_cap() -> None:
    """m2: retried, but not remembered — the class spoke about the parameter."""
    exc = _error(
        400,
        {
            "type": "invalid_request_error",
            "code": "invalid_parameter",
            "param": "output_format",
            "message": "This format is not available for this model.",
        },
    )

    assert audio_format.is_plan_refusal(exc) is True
    assert audio_format.is_plan_cap_learned(exc) is False


def test_an_entitlement_refusal_is_a_learned_cap() -> None:
    assert audio_format.is_plan_cap_learned(_plan_refusal()) is True


@pytest.mark.asyncio
async def test_a_retried_refusal_leaves_the_key_uncapped() -> None:
    """The next call still asks for 192 after an invalid_parameter fallback."""
    seen: list[str] = []

    async def send(params: dict[str, str]) -> str:
        seen.append(params["output_format"])
        if params["output_format"] == audio_format.PREFERRED_OUTPUT_FORMAT:
            raise _error(
                400,
                {
                    "type": "invalid_request_error",
                    "code": "invalid_parameter",
                    "param": "output_format",
                },
            )
        return "audio"

    kwargs = {"api_key": "k", "endpoint": "probe", "model": "probe"}
    await audio_format.post_with_format_fallback(send, params={}, **kwargs)
    assert audio_format.is_capped(_cap_key()) is False

    seen.clear()
    await audio_format.post_with_format_fallback(send, params={}, **kwargs)
    assert seen == ["mp3_44100_192", "mp3_44100_128"]


def test_a_re_measured_cap_moves_to_the_newest_position() -> None:
    """m3: the eviction must drop the true oldest, not the freshest re-measure."""
    for index in range(audio_format.CAP_MAX_KEYS):
        audio_format.remember_cap(f"key-{index}")
    audio_format.remember_cap("key-0")  # re-measure the OLDEST entry
    audio_format.remember_cap("key-new")  # one over the bound

    assert len(audio_format._caps) == audio_format.CAP_MAX_KEYS
    assert "key-0" in audio_format._caps
    assert "key-1" not in audio_format._caps
    assert "key-new" in audio_format._caps


def test_the_memo_constants_are_the_documented_six_hours_and_bound() -> None:
    assert audio_format.CAP_TTL_S == 6 * 60 * 60
    assert audio_format.CAP_MAX_KEYS == 64


def test_the_memo_is_bounded_and_drops_the_oldest() -> None:
    for index in range(audio_format.CAP_MAX_KEYS + 5):
        audio_format.remember_cap(f"key-{index}")

    assert len(audio_format._caps) == audio_format.CAP_MAX_KEYS
    assert "key-0" not in audio_format._caps
    assert f"key-{audio_format.CAP_MAX_KEYS + 4}" in audio_format._caps


def test_an_expired_entry_is_forgotten() -> None:
    key = _cap_key()
    audio_format.remember_cap(key)
    with audio_format._caps_lock:
        audio_format._caps[key] = time.monotonic() - 1

    assert audio_format.is_capped(key) is False
    assert key not in audio_format._caps


def test_the_memo_never_holds_key_material() -> None:
    secret = "sk-live-super-secret"
    key = audio_format.cap_key(secret, endpoint="voiceover", model="eleven_multilingual_v2")

    assert secret not in key
    audio_format.remember_cap(key)
    assert all(secret not in stored for stored in audio_format._caps)


def test_bitrate_is_read_from_the_format_token_and_never_guessed() -> None:
    assert audio_format.bitrate_kbps_for("mp3_44100_192") == 192
    assert audio_format.bitrate_kbps_for("mp3_44100_128") == 128
    assert audio_format.bitrate_kbps_for("pcm_44100") is None
