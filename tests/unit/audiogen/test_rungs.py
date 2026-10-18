"""The rung executors against ``httpx.MockTransport`` — the wire shapes.

No network and no paid calls anywhere in this module: the Radient endpoint is
a scripted local transport and the BYOK vendors are too. What is under test:
the frozen submit/status/result walk, the failure PAIR (COMPLETED +
``audio_failed``), sha256 verification failing closed, the cost mapping
(quote → estimated, billed → billed, BYOK → rate-table estimate), the BYOK
request bodies, and the honest cancel receipt.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from local_operator.audiogen import AudioRoute
from local_operator.audiogen import rungs as audio_rungs
from local_operator.audiogen.models import AudioAsset, AudioJobSpec
from local_operator.audiogen.rungs import CancelHandle
from local_operator.clients._http import APIError

HUB = "https://hub.example/v1"
CDN_AUDIO = "https://cdn.example/one.mp3"
AUDIO_BYTES = b"ID3\x04\x00\x00\x00fake-mp3-bytes"
AUDIO_SHA = hashlib.sha256(AUDIO_BYTES).hexdigest()


async def _fast_pause(seconds: float) -> None:  # noqa: ARG001 - the seam's shape
    return None


def _spec(direction: str = "voiceover", **overrides: Any) -> AudioJobSpec:
    base: dict[str, Any] = {
        "tool": "generate_voiceover" if direction == "voiceover" else "generate_sound_effect",
        "direction": direction,
        "text": "Hello there, this is a test." if direction == "voiceover" else "A thunderclap",
    }
    if direction == "sound_effect":
        base.update({"duration_s": 2.5, "prompt_influence": 0.3})
    base.update(overrides)
    return AudioJobSpec(**base)


def _audio_block(**overrides: Any) -> dict[str, Any]:
    block = {
        "url": CDN_AUDIO,
        "expires_at": "2026-10-18T00:00:00Z",
        "content_type": "audio/mpeg",
        "format": "mp3",
        "sample_rate_hz": 44100,
        "channels": 1,
        "bitrate_kbps": 128,
        "duration_seconds": 12.5,
        "size_bytes": len(AUDIO_BYTES),
        "sha256": AUDIO_SHA,
        "provider_model_id": "eleven_multilingual_v2",
    }
    block.update(overrides)
    return block


def _radient_transport(
    *,
    submit: dict[str, Any] | None = None,
    statuses: list[dict[str, Any]] | None = None,
    result: dict[str, Any] | None = None,
    audio_bytes: bytes = AUDIO_BYTES,
    audio_headers: dict[str, str] | None = None,
    calls: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    submit_payload = submit or {
        "request_id": "req-1",
        "status": "IN_QUEUE",
        "model": "radient/voiceover",
        "provider": "elevenlabs",
        "cost": {
            "v": 1,
            "amount_usd": 0.05,
            "basis": "quote",
            "units": 30,
            "unit": "chars",
            "provider": "elevenlabs",
            "model": "eleven_multilingual_v2",
            "tool_type": "voiceover",
            "usage_record_id": None,
            "price_version": "svc-2026.10",
        },
    }
    status_script = list(statuses if statuses is not None else [{"status": "COMPLETED"}])
    result_payload = result or {
        "status": "COMPLETED",
        "audio": _audio_block(),
        "cost": {
            "v": 1,
            "amount_usd": 0.042,
            "basis": "billed",
            "units": 30,
            "unit": "chars",
            "unit_price_usd": 0.0014,
            "provider": "elevenlabs",
            "model": "eleven_multilingual_v2",
            "tool_type": "voiceover",
            "usage_record_id": "ur-9",
            "price_version": "svc-2026.10",
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        path = request.url.path
        if path == "/v1/tools/audio/voiceover" or path == "/v1/tools/audio/sound-effects":
            return httpx.Response(200, json=submit_payload)
        if path == "/v1/tools/audio/status":
            payload = status_script.pop(0) if status_script else {"status": "COMPLETED"}
            return httpx.Response(200, json=payload)
        if path == "/v1/tools/audio/result":
            return httpx.Response(200, json=result_payload)
        if path == "/one.mp3":
            return httpx.Response(
                200, content=audio_bytes, headers=audio_headers or {"content-type": "audio/mpeg"}
            )
        return httpx.Response(404, json={"error": "no such route"})

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# Radient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_radient_voiceover_walk_settles_cost_and_retains_financial_identity():
    calls: list[httpx.Request] = []
    transport = _radient_transport(
        statuses=[{"status": "IN_QUEUE"}, {"status": "IN_PROGRESS"}, {"status": "COMPLETED"}],
        calls=calls,
    )
    handle = CancelHandle()
    events: list[tuple[str, dict[str, Any]]] = []
    async with httpx.AsyncClient(transport=transport) as client:
        outcome = await audio_rungs.run_radient(
            _spec(),
            base_url=HUB,
            credential="tok-1",
            handle=handle,
            emit=lambda text, details: events.append((text, details)),
            pause=_fast_pause,
            client=client,
        )

    # Terminal suppresses cancel writes, but never erases financial correlation.
    assert handle.terminal is True
    assert handle.provider == AudioRoute.RADIENT and handle.request_id == "req-1"
    assert outcome.model == "radient/voiceover"
    assert outcome.generation_id == "req-1"
    assert outcome.cost_usd == 0.042, "the SETTLED figure wins over the quote"
    assert outcome.cost_source == "reported"
    assert outcome.billing_basis == "billed"
    assert outcome.usage_record_id == "ur-9"
    asset = outcome.assets[0]
    assert isinstance(asset, AudioAsset)
    assert asset.data == AUDIO_BYTES
    assert asset.content_type == "audio/mpeg"
    assert asset.sha256 == AUDIO_SHA
    assert asset.duration_s == 12.5
    assert asset.provider_model_id == "eleven_multilingual_v2"

    # Every request authenticated with the bearer and carried no provider key.
    submit = next(c for c in calls if c.method == "POST" and c.url.path.endswith("/voiceover"))
    assert submit.headers["authorization"] == "Bearer tok-1"
    body = json.loads(submit.content)
    assert body["text"] == "Hello there, this is a test."
    assert "voice" not in body, "unset canonical fields must not ride the body"
    assert body["model"] == "radient/voiceover"

    # The frozen progress vocabulary: queued/in_progress stages, no fraction.
    stages = [details.get("stage") for _text, details in events]
    assert stages[0] == "queued"
    assert "in_progress" in stages
    assert all(details.get("progress_fraction") is None for _t, details in events)
    assert all(details.get("tool_name") == "generate_voiceover" for _t, details in events)


@pytest.mark.asyncio
async def test_a_completed_error_pair_is_an_error_not_a_success():
    # The frozen failure shape: COMPLETED + error_type + settled zero cost —
    # branch on the PAIR, never on the status word alone.
    transport = _radient_transport(
        statuses=[{"status": "IN_PROGRESS"}],
        result={
            "status": "COMPLETED",
            "error_type": "audio_failed",
            "error": "The vendor failed this request.",
            "cost": {"v": 1, "amount_usd": 0.0, "basis": "billed"},
        },
    )
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(APIError) as caught:
            await audio_rungs.run_radient(
                _spec(),
                base_url=HUB,
                credential="tok",
                handle=CancelHandle(),
                emit=None,
                pause=_fast_pause,
                client=client,
            )
    assert caught.value.code == "audio_failed"
    assert "vendor failed" in str(caught.value)


@pytest.mark.asyncio
async def test_a_sha_mismatch_fails_closed_and_discards_the_bytes():
    transport = _radient_transport(
        statuses=[],
        audio_bytes=b"other bytes",
        result={"status": "COMPLETED", "audio": _audio_block()},
    )
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(APIError) as caught:
            await audio_rungs.run_radient(
                _spec(),
                base_url=HUB,
                credential="tok",
                handle=CancelHandle(),
                emit=None,
                pause=_fast_pause,
                client=client,
            )
    assert caught.value.code == "invalid_response"
    assert "sha256" in str(caught.value)


def _cancel_handle() -> CancelHandle:
    handle = CancelHandle()
    handle.provider = AudioRoute.RADIENT
    handle.request_id = "req-1"
    handle.base_url = HUB
    handle.credential = SecretStr("tok")
    return handle


async def _cancel_against(handler):
    """Run ``best_effort_cancel`` against a scripted transport.

    ``best_effort_cancel`` owns its client, so the rung's scope is patched to
    inject the scripted one. Restored in a ``finally``: a leaked patch would
    make every later cancel in the module hit a dead transport.
    """
    import contextlib

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        original = audio_rungs._client_scope

        @contextlib.asynccontextmanager
        async def scope(_client):
            yield client

        audio_rungs._client_scope = scope
        try:
            return await audio_rungs.best_effort_cancel(_cancel_handle())
        finally:
            audio_rungs._client_scope = original


@pytest.mark.asyncio
async def test_a_queued_cancel_is_a_guaranteed_stop_and_says_no_charge():
    """`vendor_stopped: true` is the contract's ONE no-charge promise.

    A queued job was never handed to the provider, so the stop is certain and
    the receipt may say so. This is the only branch allowed to say "you will
    not be charged".
    """
    receipt = await _cancel_against(
        lambda request: httpx.Response(
            200,
            json={
                "status": "CANCELLED",
                "cancellation": {"vendor_stopped": True},
            },
        )
    )
    assert receipt.outcome == "cancelled"
    assert receipt.vendor_stopped is True
    text = audio_rungs.cancel_receipt_text(receipt)
    assert "stopped before the provider was called" in text
    assert "you will not be charged" in text


@pytest.mark.asyncio
async def test_a_running_cancel_never_claims_no_charge():
    """A running job keeps going: the receipt says "charged if it produces audio"."""
    receipt = await _cancel_against(
        lambda request: httpx.Response(
            200,
            json={
                "status": "CANCELLED",
                "cancellation": {
                    "vendor_stopped": False,
                    "reason": "provider_has_no_cancel_api",
                },
            },
        )
    )
    assert receipt.outcome == "cancelled"
    assert receipt.vendor_stopped is False
    assert receipt.reason == "provider_has_no_cancel_api"
    text = audio_rungs.cancel_receipt_text(receipt)
    assert "no cancellation endpoint" in text
    assert "charged if it produces audio" in text
    assert "will not be charged" not in text
    # The settlement sentence appears ONLY when the poll was actually started.
    watched = audio_rungs.cancel_receipt_text(receipt, settling=True)
    assert watched.endswith(audio_rungs.SETTLEMENT_WATCH_NOTE)


@pytest.mark.asyncio
async def test_a_cancel_response_keeps_the_servers_pending_settlement_body():
    """The contract's pending facts survive parsing for the tool to forward.

    Live capture (``aud_6ad466c8e73ab027182a54fd``): a CANCELLED response
    carries ``cost`` at ``quote`` basis plus ``cancellation`` and ``note``, and
    NO settled figure. The parser must keep that object instead of discarding
    everything but the status word — the tool result is the only channel a
    surface has to render "awaiting settlement".
    """
    body = {
        "request_id": "aud_6ad466c8e73ab027182a54fd",
        "status": "CANCELLED",
        "cancellation": {
            "requested": True,
            "vendor_stopped": False,
            "reason": "provider_has_no_cancel_api",
            "detail": (
                "The provider has no cancellation endpoint. The generation continues "
                "and will be charged if it produces audio."
            ),
        },
        "cost": {
            "v": 1,
            "amount_usd": 0.0216,
            "basis": "quote",
            "units": 540,
            "unit": "char",
            "unit_price_usd": 4e-05,
            "provider": "radient",
            "model": "radient/voiceover-flash",
            "tool_type": "speech_generation",
            "usage_record_id": "6ad466c8e73ab027182a54ff",
            "price_version": "audio-rates@2026-10-17",
        },
        "note": (
            "Cancellation is best-effort: the request settles at its terminal outcome. "
            "A completed generation is charged normally."
        ),
    }
    receipt = await _cancel_against(lambda request: httpx.Response(200, json=body))
    assert receipt.outcome == "cancelled"
    assert receipt.vendor_stopped is False
    assert receipt.reason == "provider_has_no_cancel_api"
    assert receipt.server_payload == body


@pytest.mark.asyncio
async def test_a_repeat_queued_cancel_is_idempotent():
    """Cancelling an already-queued-cancelled job answers 200 vendor_stopped:true."""
    receipt = await _cancel_against(
        lambda request: httpx.Response(
            200, json={"status": "CANCELLED", "cancellation": {"vendor_stopped": True}}
        )
    )
    assert receipt.outcome == "cancelled"
    assert receipt.vendor_stopped is True


@pytest.mark.asyncio
async def test_a_completed_job_refuses_the_cancel_with_409():
    """The settled conflict is a CODE on a 409, not a 200 status word."""
    receipt = await _cancel_against(
        lambda request: httpx.Response(
            409, json={"code": "audio_already_completed", "error": "job finished"}
        )
    )
    assert receipt.outcome == "already_completed"
    text = audio_rungs.cancel_receipt_text(receipt)
    assert "already completed" in text and "discarded" in text


@pytest.mark.asyncio
async def test_a_cancel_with_nothing_to_stop_reads_none_and_says_so():
    receipt = await audio_rungs.best_effort_cancel(CancelHandle())
    assert receipt.outcome == "none"
    assert "nothing was left to cancel" in audio_rungs.cancel_receipt_text(receipt)


# ---------------------------------------------------------------------------
# The admission bound (HTTP 429, ``audio_rate_limited``)
# ---------------------------------------------------------------------------


def _rate_limited_response() -> httpx.Response:
    """The contract's fixed busy answer — no ``Retry-After``, by design."""
    return httpx.Response(
        429,
        json={
            "code": "audio_rate_limited",
            "error": "Audio generation is busy right now. Wait a moment and try again.",
        },
    )


@pytest.mark.asyncio
async def test_a_busy_submit_retries_with_bounded_backoff_then_succeeds(monkeypatch):
    """A refused submit created no job row, so retrying it cannot double-bill.

    The sleeps are recorded rather than paid: the module's ``_submit_sleep``
    seam exists for exactly this test.
    """
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(audio_rungs, "_submit_sleep", fake_sleep)
    attempts = {"submit": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/tools/audio/voiceover":
            attempts["submit"] += 1
            if attempts["submit"] <= 2:
                return _rate_limited_response()
            return httpx.Response(
                200,
                json={
                    "request_id": "req-busy",
                    "status": "COMPLETED",
                    "model": "radient/voiceover",
                    "cost": {"v": 1, "amount_usd": 0.01, "basis": "billed"},
                },
            )
        if request.url.path == "/v1/tools/audio/result":
            return httpx.Response(
                200,
                json={
                    "status": "COMPLETED",
                    "audio": {
                        "url": f"{HUB}/media/one.mp3",
                        "content_type": "audio/mpeg",
                        "duration_seconds": 12.5,
                        "sha256": AUDIO_SHA,
                    },
                },
            )
        if request.url.path == "/v1/media/one.mp3":
            return httpx.Response(200, content=AUDIO_BYTES)
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await audio_rungs.run_radient(
            _spec(),
            base_url=HUB,
            credential="tok",
            handle=CancelHandle(),
            emit=None,
            pause=_fast_pause,
            client=client,
        )
    assert outcome.generation_id == "req-busy"
    assert attempts["submit"] == 3
    assert sleeps == [2.0, 4.0], "bounded backoff, in the contract's steps"


@pytest.mark.asyncio
async def test_an_exhausted_busy_submit_surfaces_as_transient_busy(monkeypatch):
    """After the cap the refusal surfaces: retryable class, the wire's own word."""
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(audio_rungs, "_submit_sleep", fake_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        return _rate_limited_response()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(APIError) as caught:
            await audio_rungs.run_radient(
                _spec(),
                base_url=HUB,
                credential="tok",
                handle=CancelHandle(),
                emit=None,
                pause=_fast_pause,
                client=client,
            )
    exc = caught.value
    assert exc.code == "transient", "a busy process is not a balance signal"
    assert exc.status_code == 429
    assert exc.details.get("error_type") == "audio_rate_limited"
    assert "busy right now" in str(exc)
    assert sleeps == [2.0, 4.0, 8.0], "four attempts, three waits"
    # No header dependence: the response carried no Retry-After at all.
    assert audio_rungs.SUBMIT_MAX_ATTEMPTS == 4


@pytest.mark.asyncio
async def test_an_admitted_job_that_hits_the_vendor_limit_is_not_retried(monkeypatch):
    """POST-admission 429 settles as COMPLETED+audio_rate_limited: no retry.

    The contract forbids automatic retry of an admitted job (a resubmit is the
    caller's decision), so the submit count stays 1 and the failure carries the
    wire's word with zero cost.
    """
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(audio_rungs, "_submit_sleep", fake_sleep)
    submits = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/tools/audio/sound-effects":
            submits["count"] += 1
            return httpx.Response(200, json={"request_id": "req-limited", "status": "IN_QUEUE"})
        if request.url.path == "/v1/tools/audio/status":
            return httpx.Response(
                200,
                json={
                    "status": "COMPLETED",
                    "error_type": "audio_rate_limited",
                    "error": "Audio generation is busy right now. Wait a moment and try again.",
                    "cost": {"v": 1, "amount_usd": 0.0, "basis": "billed"},
                },
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(APIError) as caught:
            await audio_rungs.run_radient(
                _spec("sound_effect"),
                base_url=HUB,
                credential="tok",
                handle=CancelHandle(),
                emit=None,
                pause=_fast_pause,
                client=client,
            )
    assert caught.value.code == "audio_rate_limited"
    assert submits["count"] == 1, "an admitted job is never automatically resubmitted"
    assert sleeps == [], "the submit backoff belongs to a REFUSED submit only"


# ---------------------------------------------------------------------------
# BYOK — ElevenLabs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_elevenlabs_voiceover_sends_the_map_resolved_body():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, content=b"ID3\x04\x00\x00\x00mp3", headers={"content-type": "audio/mpeg"}
        )

    spec = _spec(voice="female", tone="calm", expressiveness="high", pace=1.5)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await audio_rungs.run_elevenlabs(
            spec,
            api_key="el-key",
            handle=CancelHandle(),
            emit=None,
            pause=_fast_pause,
            client=client,
        )

    request = seen[0]
    assert request.headers["xi-api-key"] == "el-key"
    assert request.url.path.startswith("/v1/text-to-speech/")
    assert request.url.params["output_format"] == "mp3_44100_128"
    body = json.loads(request.content)
    assert body["model_id"] == "eleven_multilingual_v2"
    assert body["text"] == spec.text
    # expressiveness=high -> stability 0.35 (the vendored map's own number);
    # pace 1.5 clamps to the map's 1.2 ceiling; language is ignored on the
    # multilingual default and the receipt says so.
    assert body["voice_settings"]["stability"] == 0.35
    assert body["voice_settings"]["speed"] == 1.2
    assert "language_code" not in body
    assert outcome.cost_source == "rate_table"
    assert outcome.billing_basis == "estimated"
    assert outcome.cost_usd == pytest.approx(0.08 * len(spec.text) / 1000.0)
    asset = outcome.assets[0]
    assert isinstance(asset, AudioAsset)
    notes = asset.voicing_notes
    assert notes is not None and "pace:clamped" in " ".join(notes["degraded"])


@pytest.mark.asyncio
async def test_elevenlabs_flash_honours_language():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"x", headers={"content-type": "audio/mpeg"})

    spec = _spec(voice="male", language="de", model="eleven_flash_v2_5")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await audio_rungs.run_elevenlabs(
            spec, api_key="k", handle=CancelHandle(), emit=None, pause=_fast_pause, client=client
        )
    body = json.loads(seen[0].content)
    assert body["language_code"] == "de"
    assert body["model_id"] == "eleven_flash_v2_5"


@pytest.mark.asyncio
async def test_elevenlabs_sound_effect_body_carries_the_required_duration():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"x", headers={"content-type": "audio/mpeg"})

    spec = _spec("sound_effect", loop=True, duration_s=3.5, prompt_influence=0.7)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await audio_rungs.run_elevenlabs(
            spec, api_key="k", handle=CancelHandle(), emit=None, pause=_fast_pause, client=client
        )
    request = seen[0]
    assert request.url.path == "/v1/sound-generation"
    body = json.loads(request.content)
    assert body["duration_seconds"] == 3.5
    assert body["prompt_influence"] == 0.7
    assert body["loop"] is True
    assert body["model_id"] == "eleven_text_to_sound_v2"
    # 3.5 s is FOUR started seconds: the estimate prices whole units, the same
    # count the ledger row carries (a fractional 3.5 × rate under-quoted it).
    assert outcome.cost_usd == pytest.approx(0.002 * 4)
    assert outcome.cost_source == "rate_table"
    assert outcome.billing_basis == "estimated"


# ---------------------------------------------------------------------------
# BYOK — OpenAI
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_voiceover_maps_the_descriptor_into_the_body():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"x", headers={"content-type": "audio/mpeg"})

    spec = _spec(voice="female", tone="bright", accent="scottish", instructions="Speak slowly.")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await audio_rungs.run_openai(
            spec,
            api_key="oa-key",
            handle=CancelHandle(),
            emit=None,
            pause=_fast_pause,
            client=client,
        )
    request = seen[0]
    assert request.headers["authorization"] == "Bearer oa-key"
    assert request.url.path == "/v1/audio/speech"
    body = json.loads(request.content)
    assert body["model"] == "gpt-4o-mini-tts"
    # The map selects the voice ROW by (gender, tone): female + bright is the
    # map's ``f-bright`` row, whose OpenAI voice is ``shimmer`` — not the
    # female warm default (``nova``).
    assert body["voice"] == "shimmer"
    assert body["response_format"] == "mp3"
    # The emulation composed the tone/accent guidance into instructions.
    assert "bright, upbeat" in body["instructions"]
    assert "scottish accent" in body["instructions"]
    assert body["instructions"].startswith("Speak slowly.")
    assert outcome.cost_source == "rate_table"


@pytest.mark.asyncio
async def test_openai_refuses_a_sound_effect_honestly():
    transport = httpx.MockTransport(lambda r: httpx.Response(500))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(APIError) as caught:
            await audio_rungs.run_openai(
                _spec("sound_effect"),
                api_key="k",
                handle=CancelHandle(),
                emit=None,
                pause=_fast_pause,
                client=client,
            )
    assert caught.value.code == "unsupported"
    assert "voiceover" in str(caught.value)


def test_the_byok_cancel_sentence_never_promises_a_stop():
    text = audio_rungs.cancel_receipt_text(
        audio_rungs.AudioCancelReceipt(outcome="none"), byok=True
    )
    assert "may still bill" in text
    assert "no cancel API" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("sha", [None, "", "bad", "g" * 64, "0" * 64, AUDIO_SHA])
async def test_radient_sha_is_required_well_formed_and_matching(sha):
    audio = _audio_block()
    if sha is None:
        audio.pop("sha256", None)
    else:
        audio["sha256"] = sha
    async with httpx.AsyncClient(transport=_radient_transport()) as client:
        if sha == AUDIO_SHA:
            asset = await audio_rungs._download_audio_asset(audio, http=client)
            assert asset.data == AUDIO_BYTES
        else:
            with pytest.raises(APIError, match="sha256"):
                await audio_rungs._download_audio_asset(audio, http=client)
