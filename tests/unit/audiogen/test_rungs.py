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
from local_operator.tts import output_format as audio_format

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
    assert sleeps == [10.0, 20.0], "the ladder's first two steps, over the 45 s hold"


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
    # The receipt names the capacity window, not just the provider's sentence:
    # the previous incident produced "busy right now" per item and could not
    # say how long the platform had been held.
    assert "4 attempts over" in str(exc)
    assert exc.details["attempts"] == 4
    assert sleeps == [10.0, 20.0, 35.0], "four attempts, three waits"
    # No header dependence: the response carried no Retry-After at all.
    assert audio_rungs.SUBMIT_MAX_ATTEMPTS == 4
    # 0 + 10 + 30 + 35 = the 65 s that clears the server's 45 s vendor bound.
    assert sum(sleeps) == 65.0


@pytest.mark.asyncio
async def test_the_capacity_window_is_a_fraction_of_the_batch_budget(monkeypatch):
    """A fake clock proves the 65 s ladder cannot eat the whole batch budget.

    ``SOUND_BATCH_TIMEOUT_S`` is 225 s and is deliberately NOT raised: the
    ladder is what absorbs the hold. One exhausted window must leave room for
    at least two more, or a 3-item resume would spend its entire budget just
    discovering that capacity is full.
    """
    from local_operator.tools import audio_tool

    clock = {"t": 1_000.0}
    sleeps: list[float] = []

    async def advance(seconds: float) -> None:
        sleeps.append(seconds)
        clock["t"] += seconds

    monkeypatch.setattr(audio_rungs, "_submit_sleep", advance)
    monkeypatch.setattr(audio_rungs, "_submit_clock", lambda: clock["t"])

    def handler(request: httpx.Request) -> httpx.Response:
        return _rate_limited_response()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(APIError) as caught:
            await audio_rungs._submit_with_backoff(
                client,
                f"{HUB}/tools/audio/sound-effects",
                body={"prompt": "click", "duration_s": 0.5},
                credential="tok",
            )

    window = caught.value.details["capacity_window_s"]
    assert window == 65.0
    assert caught.value.details["attempts"] == 4
    assert "Audio platform at capacity: 4 attempts over 65 s." in str(caught.value)
    # The window is MEASURED from the injected clock, not copied from the
    # constant tuple: the clock advanced by exactly the ladder's waits, which is
    # what makes the receipt a duration. This fails if the seam regresses.
    assert clock["t"] - 1_000.0 == pytest.approx(sum(sleeps)) == pytest.approx(window)
    # The batch bound is deliberately unchanged; the ladder-vs-budget proof is
    # a scheduling test at the tool seam (test_audio_tool), not this equality.
    assert audio_tool.SOUND_BATCH_TIMEOUT_S == 225.0


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


@pytest.mark.asyncio
async def test_a_5xx_carrying_the_rate_limited_code_is_not_retried_on_the_hub(monkeypatch):
    """M1: the STATUS guards the ladder; a body can never widen it.

    A 503 whose body repeats ``audio_rate_limited`` is a request that may have
    been accepted. Re-POSTing it could double-bill, so only HTTP 429 is the
    admission bound and every other status fails straight through.
    """
    posts: list[httpx.Request] = []
    sleeps: list[float] = []

    async def pause(seconds):
        sleeps.append(seconds)

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append(request)
        return httpx.Response(503, json={"code": "audio_rate_limited", "error": "upstream"})

    monkeypatch.setattr(audio_rungs, "_submit_sleep", pause)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(APIError) as caught:
            await audio_rungs._submit_with_backoff(
                client,
                f"{HUB}/tools/audio/sound-effects",
                body={"prompt": "click", "duration_s": 0.5},
                credential="tok",
            )
    assert len(posts) == 1, "a 5xx is the upstream's answer, never a refusal to retry"
    assert sleeps == []
    assert caught.value.status_code == 503


# ---------------------------------------------------------------------------
# BYOK — ElevenLabs
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_format_memo():
    """The 192->128 memo is process-wide by design; no test may inherit it."""
    audio_format.reset_caps()
    yield
    audio_format.reset_caps()


#: The measured shape of a plan refusal for the 192 format: the vendor names the
#: offending query parameter in ``detail.param`` and classes it as an
#: entitlement refusal. Prose is included because the real response has it, not
#: because the predicate reads it when ``param`` is present.
_ELEVENLABS_PLAN_REFUSAL = {
    "detail": {
        "type": "authorization_error",
        "code": "feature_not_available",
        "param": "output_format",
        "message": (
            "The 'mp3_44100_192' output format is only available on the Creator "
            "plan and above. Please upgrade your subscription."
        ),
    }
}


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
    assert request.url.params["output_format"] == "mp3_44100_192"
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
    # The receipt states the bitrate of the format that went on the wire.
    assert asset.bitrate_kbps == 192
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
    assert request.url.params["output_format"] == "mp3_44100_192"
    assert body["duration_seconds"] == 3.5
    assert body["prompt_influence"] == 0.7
    assert body["loop"] is True
    assert body["model_id"] == "eleven_text_to_sound_v2"
    # 3.5 s is FOUR started seconds: the estimate prices whole units, the same
    # count the ledger row carries (a fractional 3.5 × rate under-quoted it).
    assert outcome.cost_usd == pytest.approx(0.002 * 4)
    assert outcome.cost_source == "rate_table"
    assert outcome.billing_basis == "estimated"


@pytest.mark.asyncio
async def test_elevenlabs_admission_refusal_retries_on_the_shared_ladder(monkeypatch):
    """BYOK gets the SAME ladder as the hub: a 429 is a refusal, not a verdict.

    A BYOK 429 produces no audio and charges nothing, so retrying is safe — and
    a resumed batch must not fail an item because a vendor (or a slot on a
    shared key) was briefly busy. Batch submissions take the jittered arm; the
    jitter floor is pinned to the hold, so three refusals still clear it.
    """
    posts: list[httpx.Request] = []
    sleeps: list[float] = []

    async def pause(seconds):
        sleeps.append(seconds)

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append(request)
        if len(posts) < 4:
            return _rate_limited_response()
        return httpx.Response(200, content=b"ID3ok", headers={"content-type": "audio/mpeg"})

    monkeypatch.setattr(audio_rungs, "_submit_sleep", pause)
    monkeypatch.setattr(audio_rungs.random, "uniform", lambda low, high: low)
    spec = _spec("sound_effect", duration_s=0.5, batch_submission=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await audio_rungs.run_elevenlabs(
            spec, api_key="k", handle=CancelHandle(), emit=None, pause=_fast_pause, client=client
        )
    assert len(posts) == 4, "three refusals, then the admitted submit"
    assert sleeps == [10.0, 20.0, 35.0]
    assert outcome.assets[0].data == b"ID3ok"


@pytest.mark.asyncio
async def test_elevenlabs_a_400_is_the_answer_and_is_never_retried(monkeypatch):
    """Only the admission refusal is retried: a rejected request is final.

    A 400 (``audio_rejected``) means the request itself was wrong. Re-posting it
    would burn the ladder's 65 s and then fail identically, so it must surface
    on the first post — the counterpart to the retry test above.
    """
    posts: list[httpx.Request] = []
    sleeps: list[float] = []

    async def pause(seconds):
        sleeps.append(seconds)

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append(request)
        return httpx.Response(400, json={"code": "audio_rejected", "error": "bad duration"})

    monkeypatch.setattr(audio_rungs, "_submit_sleep", pause)
    spec = _spec("sound_effect", duration_s=0.5)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(APIError) as caught:
            await audio_rungs.run_elevenlabs(
                spec,
                api_key="k",
                handle=CancelHandle(),
                emit=None,
                pause=_fast_pause,
                client=client,
            )
    assert len(posts) == 1
    assert sleeps == []
    assert caught.value.code == "audio_rejected"


@pytest.mark.asyncio
async def test_elevenlabs_5xx_carrying_the_rate_limited_code_is_never_retried(monkeypatch):
    """The BYOK path the ladder was newly pointed at: status first, body never.

    A 503 with ``code: audio_rate_limited`` is the exact request a re-POST could
    double-bill. It must surface on the first post even though the body repeats
    the admission code.
    """
    posts: list[httpx.Request] = []
    sleeps: list[float] = []

    async def pause(seconds):
        sleeps.append(seconds)

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append(request)
        return httpx.Response(503, json={"code": "audio_rate_limited", "error": "upstream"})

    monkeypatch.setattr(audio_rungs, "_submit_sleep", pause)
    spec = _spec("sound_effect", duration_s=0.5)
    key = "key"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(APIError) as caught:
            await audio_rungs.run_elevenlabs(
                spec,
                api_key=key,
                handle=CancelHandle(),
                emit=None,
                pause=_fast_pause,
                client=client,
            )
    assert len(posts) == 1
    assert sleeps == []
    assert caught.value.status_code == 503
async def _run_byok(spec: AudioJobSpec, *, api_key: str, handler):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await audio_rungs.run_elevenlabs(
            spec,
            api_key=api_key,
            handle=CancelHandle(),
            emit=None,
            pause=_fast_pause,
            client=client,
        )


def _formats(seen: list[httpx.Request]) -> list[str]:
    return [request.url.params["output_format"] for request in seen]


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["voiceover", "sound_effect"])
async def test_elevenlabs_drops_to_128_once_when_the_plan_refuses_192(direction):
    """A lower-tier key must still get audio, on BOTH vendor endpoints."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.params["output_format"] == "mp3_44100_192":
            return httpx.Response(403, json=_ELEVENLABS_PLAN_REFUSAL)
        return httpx.Response(200, content=AUDIO_BYTES, headers={"content-type": "audio/mpeg"})

    spec = _spec(direction, duration_s=2.0) if direction == "sound_effect" else _spec(direction)
    outcome = await _run_byok(spec, api_key="capped-key", handler=handler)

    assert _formats(seen) == ["mp3_44100_192", "mp3_44100_128"]
    assert outcome.assets[0].data == AUDIO_BYTES
    # M2: the fact follows the fallback rather than a constant. Only the
    # voiceover leg states a bitrate; a sound effect legitimately reports none.
    # ``assets`` is typed as the base MediaAsset, so narrow before reading the
    # audio-specific fact (pyright would not see it on the base class).
    asset = outcome.assets[0]
    assert isinstance(asset, AudioAsset)
    assert asset.bitrate_kbps == (128 if direction == "voiceover" else None)


@pytest.mark.asyncio
async def test_elevenlabs_skips_the_doomed_attempt_for_the_same_key():
    """The measured cap is remembered, so later calls go straight to 128."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.params["output_format"] == "mp3_44100_192":
            return httpx.Response(403, json=_ELEVENLABS_PLAN_REFUSAL)
        return httpx.Response(200, content=AUDIO_BYTES, headers={"content-type": "audio/mpeg"})

    await _run_byok(_spec("voiceover"), api_key="capped-key", handler=handler)
    seen.clear()
    await _run_byok(_spec("voiceover"), api_key="capped-key", handler=handler)

    assert _formats(seen) == ["mp3_44100_128"]


@pytest.mark.asyncio
async def test_elevenlabs_retries_a_parameter_refusal_but_does_not_remember_it():
    """A 400 naming the format is retried; it is not a statement about the tier."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.params["output_format"] == "mp3_44100_192":
            return httpx.Response(
                400,
                json={
                    "detail": {
                        "type": "invalid_request_error",
                        "code": "invalid_parameter",
                        "param": "output_format",
                        "message": "This format is not available for this model.",
                    }
                },
            )
        return httpx.Response(200, content=AUDIO_BYTES, headers={"content-type": "audio/mpeg"})

    await _run_byok(_spec("voiceover"), api_key="format-key", handler=handler)
    seen.clear()
    await _run_byok(_spec("voiceover"), api_key="format-key", handler=handler)

    # Retried once on BOTH calls: the second call still attempts 192, because a
    # parameter refusal must not suppress the preference for six hours.
    assert _formats(seen) == ["mp3_44100_192", "mp3_44100_128"]


@pytest.mark.asyncio
async def test_elevenlabs_memo_is_scoped_to_the_endpoint_that_refused():
    """A sound-effect refusal must not downgrade voiceover for the same key."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.params["output_format"] == "mp3_44100_192":
            return httpx.Response(403, json=_ELEVENLABS_PLAN_REFUSAL)
        return httpx.Response(200, content=AUDIO_BYTES, headers={"content-type": "audio/mpeg"})

    await _run_byok(_spec("sound_effect", duration_s=1.0), api_key="capped-key", handler=handler)
    seen.clear()
    await _run_byok(_spec("voiceover"), api_key="capped-key", handler=handler)

    # Voiceover is a different capability: it still asks for 192 first.
    assert _formats(seen)[0] == "mp3_44100_192"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body"),
    [
        (
            400,
            {
                "detail": {
                    "type": "invalid_request_error",
                    "code": "invalid_parameter",
                    "param": "text",
                    "message": "text is invalid",
                }
            },
        ),
        (
            403,
            {
                "detail": {
                    "type": "authorization_error",
                    "code": "feature_not_available",
                    "param": "enable_logging",
                    "message": "Zero retention mode is only available on the Enterprise plan.",
                }
            },
        ),
    ],
)
async def test_elevenlabs_does_not_retry_a_refusal_of_another_parameter(status, body):
    """Only output_format is a tier gate we can answer by changing the format."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=body)

    with pytest.raises(APIError) as caught:
        await _run_byok(_spec("voiceover"), api_key="k", handler=handler)

    assert _formats(seen) == ["mp3_44100_192"]
    assert caught.value.status_code == status


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 429, 500])
async def test_elevenlabs_never_retries_an_auth_rate_or_server_refusal(status):
    """Even a body naming output_format must not earn a retry on these statuses."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=_ELEVENLABS_PLAN_REFUSAL)

    with pytest.raises(APIError) as caught:
        await _run_byok(_spec("voiceover"), api_key="k", handler=handler)

    assert _formats(seen) == ["mp3_44100_192"]
    assert caught.value.status_code == status


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("detail", "retried"),
    [
        (
            # The safety arm: a plan gate with no `param`, recognised by class + prose.
            {
                "type": "feature_not_available",
                "code": "feature_not_available",
                "message": "This feature is not available on your current plan.",
            },
            True,
        ),
        (
            # Same class, prose that names no entitlement: not a tier statement.
            {
                "type": "feature_not_available",
                "code": "feature_not_available",
                "message": "This feature is not available for this voice.",
            },
            False,
        ),
    ],
)
async def test_elevenlabs_unnamed_refusal_needs_entitlement_prose(detail, retried):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.params["output_format"] == "mp3_44100_192":
            return httpx.Response(403, json={"detail": detail})
        return httpx.Response(200, content=AUDIO_BYTES, headers={"content-type": "audio/mpeg"})

    try:
        outcome = await _run_byok(_spec("voiceover"), api_key="k", handler=handler)
    except APIError as exc:
        assert not retried
        assert exc.status_code == 403
    else:
        assert retried
        assert outcome.assets[0].data == AUDIO_BYTES
    assert len(seen) == (2 if retried else 1)


@pytest.mark.asyncio
async def test_elevenlabs_surfaces_the_original_refusal_when_the_fallback_fails():
    """A 128 attempt that fails too must not replace the 192 diagnosis."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params["output_format"] == "mp3_44100_192":
            return httpx.Response(403, json=_ELEVENLABS_PLAN_REFUSAL)
        return httpx.Response(500, json={"detail": {"status": "internal_error"}})

    with pytest.raises(APIError) as caught:
        await _run_byok(_spec("voiceover"), api_key="k", handler=handler)

    assert caught.value.status_code == 403
    assert isinstance(caught.value.__cause__, APIError)
    assert caught.value.__cause__.status_code == 500
    # A failed retry proves nothing about the plan: the key is not capped.
    assert not audio_format.is_capped(
        audio_format.cap_key("k", endpoint="voiceover", model="eleven_multilingual_v2")
    )


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


@pytest.mark.asyncio
async def test_batch_admission_backoff_is_jittered_and_bounded(monkeypatch):
    """Jitter may only LENGTHEN: the floor is the documented 65 s window.

    The previous range (0.75-1.0) shortened the wait to as little as 49 s, which
    is inside a 45 s hold plus the terminal write it has to clear — so a batch
    worker could retry straight back into the window the ladder exists to
    outlast. Pinning ``uniform`` to its LOW end proves the floor; the companion
    test pins the high end.
    """
    sleeps = []

    async def pause(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(audio_rungs, "_submit_sleep", pause)
    monkeypatch.setattr(audio_rungs.random, "uniform", lambda low, high: low)
    requests = []

    def handler(request):
        requests.append(request)
        return _rate_limited_response()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(APIError) as caught:
            await audio_rungs._submit_with_backoff(
                client,
                f"{HUB}/tools/audio/sound-effects",
                body={"prompt": "click", "duration_s": 0.5},
                credential="tok",
                jitter=True,
            )
    assert caught.value.details["error_type"] == "audio_rate_limited"
    assert len(requests) == 4
    assert sleeps == [10.0, 20.0, 35.0], "the jittered floor is the hold, never below it"


@pytest.mark.asyncio
async def test_batch_jitter_can_only_move_a_wait_later(monkeypatch):
    """The other end of the range: jitter stretches, and the receipt says so."""
    swallows = []

    async def pause(seconds):
        swallows.append(seconds)

    monkeypatch.setattr(audio_rungs, "_submit_sleep", pause)
    monkeypatch.setattr(audio_rungs.random, "uniform", lambda low, high: high)

    def handler(request):
        return _rate_limited_response()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(APIError):
            await audio_rungs._submit_with_backoff(
                client,
                f"{HUB}/tools/audio/sound-effects",
                body={"prompt": "click", "duration_s": 0.5},
                credential="tok",
                jitter=True,
            )
    assert swallows == pytest.approx([11.5, 23.0, 40.25])


def _decodable_wav(seconds: float = 0.5, rate: int = 8000) -> bytes:
    """A real, decodable clip so the duration derivation has something to read."""
    import struct

    frames = b"\x00\x00" * int(rate * seconds)
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(frames))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(frames))
    )
    return header + frames


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["sound_effect", "voiceover"])
async def test_byok_legs_carry_a_duration_read_from_the_bytes(direction):
    """REGRESSION: a BYOK record rendered NO PLAYER because duration_s was None.

    ElevenLabs and OpenAI state no duration, so a leg that only reads a provider
    field leaves the artifact without a length - and the sound card, which needs
    one for the waveform and the clock, draws no player at all: a generated clip
    the user cannot play, with no error. The duration is now derived from the
    returned bytes, so assert it is present AND that it matches the bytes rather
    than merely being non-null.
    """
    pytest.importorskip("miniaudio")
    from local_operator.audiogen.peaks import compute_duration

    audio = _decodable_wav(0.5)
    expected = compute_duration(audio)
    assert expected == pytest.approx(0.5, abs=1e-6)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=audio, headers={"content-type": "audio/wav"})

    spec = _spec(direction, duration_s=0.5) if direction == "sound_effect" else _spec(direction)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await audio_rungs.run_elevenlabs(
            spec,
            api_key="el-key",
            handle=CancelHandle(),
            emit=None,
            pause=_fast_pause,
            client=client,
        )

    asset = outcome.assets[0]
    assert asset.duration_s is not None, "a BYOK asset without a duration renders no player"
    assert asset.duration_s == pytest.approx(expected, abs=1e-9), "the bytes are the source"
