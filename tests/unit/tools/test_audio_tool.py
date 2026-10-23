"""The audio tools: frozen schema, gates, results, failure, cancel, ledger.

No provider calls anywhere: the cascade is faked at the tool module's
``run_audio_cascade`` seam, and the attachment store is redirected into
``tmp_path``. What is under test is the tool CONTRACT — the schema the model
sees, which sessions get which tool, the lane-D result shape, the failure and
cancellation receipts, and the ``audio`` channel row each outcome hands the
ledger.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import SecretStr

from local_operator.artifacts import ArtifactKind, JobAttempt, JobOutcome
from local_operator.artifacts.naming import media_filename
from local_operator.audiogen import AudioRoute
from local_operator.audiogen import availability as audio_availability
from local_operator.audiogen import pricing
from local_operator.audiogen import rungs as audio_rungs
from local_operator.audiogen.cascade import (
    AudioGenerationCancelled,
    AudioGenerationUnavailable,
)
from local_operator.audiogen.models import AudioAsset
from local_operator.harness.types import AttachmentContent, TextContent, ToolContext
from local_operator.tools import audio_tool


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    store_root = tmp_path / "attachments"
    store_root.mkdir()
    monkeypatch.setattr("local_operator.session.attachments.attachments_dir", lambda: store_root)
    return store_root


def _text(result) -> str:
    """The first TextContent block's text, narrowed (the test oracle)."""
    for block in result.content:
        if isinstance(block, TextContent):
            return block.text
    raise AssertionError("no TextContent in the result")


def _details(result) -> dict[str, Any]:
    """The result's ``details`` mapping (never ``None`` in this lane)."""
    assert result.details is not None
    return result.details


def _context(
    tmp_path,
    spent: list[Any] | None = None,
    *,
    spawn_background=None,
    journal_audio_settle=None,
) -> ToolContext:
    """A context with the audio hooks a real session binds."""
    context = ToolContext(cwd=str(tmp_path))
    updates: dict[str, Any] = {}
    if spent is not None:
        updates["record_channel_spend"] = spent.append
    if spawn_background is not None:
        updates["spawn_background"] = spawn_background
    if journal_audio_settle is not None:
        updates["journal_audio_settle"] = journal_audio_settle
    return context.model_copy(update=updates) if updates else context


def _asset(duration_s: float | None = 12.5) -> AudioAsset:
    return AudioAsset(
        data=b"ID3\x04\x00\x00\x00fake-mp3-bytes",
        content_type="audio/mpeg",
        source_url="https://cdn.example/one.mp3",
        duration_s=duration_s,
        format="mp3",
        sample_rate_hz=44100,
        channels=1,
        bitrate_kbps=128,
        provider_model_id="eleven_multilingual_v2",
        sha256="a" * 64,
    )


def _outcome(**overrides: Any) -> JobOutcome:
    base: dict[str, Any] = {
        "kind": ArtifactKind.AUDIO,
        "assets": (_asset(),),
        "route": AudioRoute.RADIENT,
        "attempts": (JobAttempt(route=str(AudioRoute.RADIENT), outcome="ok"),),
        "model": "radient/voiceover",
        "prompt": "Hello there.",
        "generation_id": "req-1",
        "cost_usd": 0.042,
        "cost_source": "reported",
        "billing_basis": "billed",
        "cost_provenance": "Radient /tools/audio/result cost.amount_usd (basis: billed)",
        "usage_record_id": "ur-9",
    }
    base.update(overrides)
    return JobOutcome(**base)


# ---------------------------------------------------------------------------
# The frozen schema
# ---------------------------------------------------------------------------


def _string_arm(prop: dict[str, Any]) -> dict[str, Any]:
    """The non-null variant of an optional field (see the note below)."""
    for node in prop.get("anyOf", [prop]):
        if node.get("type") == "string" or "enum" in node or "maximum" in node or "minimum" in node:
            return node
    return prop


def test_the_voiceover_schema_is_the_frozen_one():
    schema = audio_tool.GenerateVoiceoverParams.model_json_schema()
    properties = schema["properties"]
    assert set(properties) == {
        "text",
        "voice",
        "tone",
        "expressiveness",
        "pace",
        "language",
        "accent",
        "instructions",
        "model",
        "filename",
    }
    assert schema["required"] == ["text"]
    assert properties["text"]["maxLength"] == 10_000
    assert properties["text"]["minLength"] == 1
    # Optional fields round-trip as anyOf[string, null] in the raw pydantic
    # schema; the wire collapses the null arm (``optional_null_unions``), so
    # every bound reads off the string arm here.
    assert _string_arm(properties["instructions"])["maxLength"] == 8_000
    pace = _string_arm(properties["pace"])
    assert pace["minimum"] == 0.5 and pace["maximum"] == 2.0
    assert _string_arm(properties["tone"])["enum"] == [
        "warm",
        "neutral",
        "bright",
        "calm",
        "authoritative",
    ]
    assert _string_arm(properties["expressiveness"])["enum"] == ["low", "medium", "high"]


def test_the_sound_effect_schema_is_the_frozen_one():
    schema = audio_tool.GenerateSoundEffectParams.model_json_schema()
    properties = schema["properties"]
    assert set(properties) == {
        "prompt",
        "duration_s",
        "sounds",
        "loop",
        "prompt_influence",
        "model",
        "filename",
    }
    assert _string_arm(properties["prompt"])["maxLength"] == 600
    assert _string_arm(properties["duration_s"])["minimum"] == 0.5
    assert _string_arm(properties["duration_s"])["maximum"] == 30.0
    assert properties["loop"]["default"] is False
    assert properties["prompt_influence"]["default"] == 0.3


def test_no_provider_specific_field_rides_either_schema():
    for model in (audio_tool.GenerateVoiceoverParams, audio_tool.GenerateSoundEffectParams):
        for name, node in model.model_json_schema()["properties"].items():
            assert "provider" not in name, name
            assert "elevenlabs" not in str(node).lower(), name
            assert "openai" not in str(node).lower(), name


# ---------------------------------------------------------------------------
# Gates: which sessions get which tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_gates_are_per_direction_and_refuse_honestly(monkeypatch, tmp_path):
    voice: dict[str, bool] = {"value": False}
    effect: dict[str, bool] = {"value": False}
    monkeypatch.setattr(
        audio_availability, "voiceover_reachable", lambda config_dir=None: voice["value"]
    )
    monkeypatch.setattr(
        audio_availability, "sound_effect_reachable", lambda config_dir=None: effect["value"]
    )
    context = _context(tmp_path)
    assert audio_tool.build_generate_voiceover_tool(context) is None
    assert audio_tool.build_generate_sound_effect_tool(context) is None
    voice["value"] = True
    built = audio_tool.build_generate_voiceover_tool(context)
    assert built is not None and built.name == "generate_voiceover"
    assert built.approval_tier == "write"
    assert built.interruptible is True and built.concurrency == "shared"
    # OpenAI-only: the voiceover tool exists, the sound-effect tool must not.
    assert audio_tool.build_generate_sound_effect_tool(context) is None
    effect["value"] = True
    assert audio_tool.build_generate_sound_effect_tool(context) is not None


# ---------------------------------------------------------------------------
# Approval text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_voiceover_card_names_the_route_and_the_estimate(monkeypatch):
    monkeypatch.setattr(audio_tool, "_preferred_route", lambda direction: AudioRoute.ELEVENLABS)
    text = audio_tool._describe_voiceover_approval({"text": "x" * 1000, "model": None}, ".")
    assert "1,000 chars" in text
    assert "ElevenLabs" in text
    assert "$0.0800" in text, "the rate table prices 1k chars at $0.08"
    assert "estimated" in text and "billed figure is the provider's" in text


@pytest.mark.asyncio
async def test_the_radient_card_says_the_server_quotes(monkeypatch):
    monkeypatch.setattr(audio_tool, "_preferred_route", lambda direction: AudioRoute.RADIENT)
    text = audio_tool._describe_voiceover_approval({"text": "hello"}, ".")
    assert "Radient" in text
    assert "quotes the price when it runs" in text
    assert "$" not in text


@pytest.mark.asyncio
async def test_the_sound_effect_card_prices_the_duration(monkeypatch):
    monkeypatch.setattr(audio_tool, "_preferred_route", lambda direction: AudioRoute.ELEVENLABS)
    text = audio_tool._describe_sound_effect_approval({"prompt": "a clap", "duration_s": 10}, ".")
    assert "10s sound effect" in text
    assert "$0.0200" in text, "10 s at $0.002/s"


@pytest.mark.asyncio
async def test_describers_never_fail_on_malformed_args(monkeypatch):
    monkeypatch.setattr(audio_tool, "_preferred_route", lambda direction: None)
    assert "[unresolvable]" in audio_tool._describe_voiceover_approval({}, ".")
    assert "[unresolvable]" in audio_tool._describe_sound_effect_approval({"prompt": "  "}, ".")


# ---------------------------------------------------------------------------
# Execution: the lane-D result, the ledger row, failure and cancellation
# ---------------------------------------------------------------------------


def _fake_cascade(monkeypatch, outcome_or_exc, capture: dict[str, Any] | None = None):
    async def fake_run(**kwargs):
        if capture is not None:
            capture.update(kwargs)
        if isinstance(outcome_or_exc, BaseException):
            raise outcome_or_exc
        return outcome_or_exc

    monkeypatch.setattr(audio_tool, "run_audio_cascade", fake_run)


@pytest.mark.asyncio
async def test_the_result_registers_once_and_shapes_the_frozen_caption(monkeypatch, tmp_path):
    spent: list[Any] = []
    _fake_cascade(monkeypatch, _outcome())
    result = await audio_tool.execute_generate_voiceover(
        "call-1",
        {"text": "Hello there."},
        None,
        None,
        _context(tmp_path, spent),
    )
    assert not result.is_error
    caption, block = result.content
    assert isinstance(caption, TextContent) and caption.text
    assert "voiceover (12.5s)" in caption.text
    assert "Radient" in caption.text and "digest" in caption.text
    assert "billed" in caption.text and "$0.042" in caption.text
    assert isinstance(block, AttachmentContent)
    assert block.kind == "audio"
    assert block.name == "hello-there.mp3", "the REQUEST names the asset, not the model"
    assert block.duration_s == 12.5
    assert block.attachment and len(block.attachment) == 32
    details = _details(result)
    assert details["provider"] == "radient"
    assert details["request_id"] == "req-1"
    assert details["cost_usd"] == 0.042
    assert details["billing_basis"] == "billed"
    assert details["usage_record_id"] == "ur-9"
    assert details["audio"]["sample_rate_hz"] == 44100
    assert details["audio"]["sha256"] == "a" * 64
    assert details["artifacts"][0]["digest"] == block.attachment
    assert details["artifacts"][0]["kind"] == "audio"
    assert details["attempts"] == [
        {"route": "radient", "outcome": "ok", "reason_class": "", "message": ""}
    ]

    assert len(spent) == 1
    record = spent[0]
    assert record.channel == "audio"
    assert record.provider == "radient"
    assert record.unit == "chars" and record.units == float(len("Hello there."))
    assert record.billing_basis == "billed"
    assert record.status == "ok"
    assert record.request_id == "req-1"


@pytest.mark.asyncio
async def test_a_byok_outcome_keeps_the_estimate_flag(monkeypatch, tmp_path):
    spent: list[Any] = []
    outcome = _outcome(
        route=AudioRoute.ELEVENLABS,
        model="eleven_multilingual_v2",
        cost_usd=0.0075,
        cost_source="rate_table",
        billing_basis="estimated",
        cost_provenance="elevenlabs eleven_multilingual_v2 at the central audio rate table",
        usage_record_id=None,
    )
    _fake_cascade(monkeypatch, outcome)
    result = await audio_tool.execute_generate_voiceover(
        "call-2", {"text": "Hello there."}, None, None, _context(tmp_path, spent)
    )
    assert not result.is_error
    assert "estimated from published rates" in _text(result)
    record = spent[0]
    assert record.billing_basis == "estimated"
    assert (
        record.cost_source == "catalogue"
    ), "a BYOK estimate is a catalogue figure, never reported"
    assert record.billing_basis != "billed"


@pytest.mark.asyncio
async def test_a_sound_effect_ledger_row_counts_seconds(monkeypatch, tmp_path):
    spent: list[Any] = []
    outcome = _outcome(route=AudioRoute.ELEVENLABS, model="eleven_text_to_sound_v2")
    _fake_cascade(monkeypatch, outcome)
    result = await audio_tool.execute_generate_sound_effect(
        "call-3",
        {
            "prompt": "a clap",
            "duration_s": 4,
            "filename": "Bird Chirping Sound Effect",
        },
        None,
        None,
        _context(tmp_path, spent),
    )
    assert not result.is_error
    assert "sound effect" in _text(result)
    assert [b.name for b in result.content if isinstance(b, AttachmentContent)] == [
        "bird-chirping-sound-effect.mp3"
    ], "an explicit filename wins and is display-safe"
    assert spent[0].unit == "seconds" and spent[0].units == 4.0


@pytest.mark.asyncio
async def test_a_sub_second_byok_effect_row_counts_one_started_second(monkeypatch, tmp_path):
    """0.5 s is ONE unit at 2000 micro, and the row stays an estimate.

    The merged lane recorded ``units: 0.5`` beside a half-cost figure for this
    exact request. The count comes from the same whole-second rule the
    estimate prices with, and the label must not drift towards a charge.
    """
    spent: list[Any] = []
    estimate = pricing.sound_effect_estimate("elevenlabs", None, 0.5)
    assert estimate is not None and estimate.amount_usd == pytest.approx(0.002)
    outcome = _outcome(
        route=AudioRoute.ELEVENLABS,
        model="eleven_text_to_sound_v2",
        cost_usd=estimate.amount_usd,
        cost_source="rate_table",
        billing_basis="estimated",
        cost_provenance=estimate.price_version,
        usage_record_id=None,
    )
    _fake_cascade(monkeypatch, outcome)
    result = await audio_tool.execute_generate_sound_effect(
        "call-sfx-half",
        {"prompt": "a single clap", "duration_s": 0.5},
        None,
        None,
        _context(tmp_path, spent),
    )
    assert not result.is_error
    assert "Cost $0.002 " in _text(result)
    assert [b.name for b in result.content if isinstance(b, AttachmentContent)] == [
        "single-clap.mp3"
    ], "omitted, the name derives from the prompt"
    row = spent[0]
    assert row.units == 1.0 and row.unit == "seconds"
    assert row.amount_micro == 2000
    assert row.billing_basis == "estimated"
    assert row.cost_source == "catalogue"
    assert row.price_version == pricing.PRICE_VERSION


@pytest.mark.asyncio
async def test_the_failure_shape_is_an_error_with_the_attempts(monkeypatch, tmp_path):
    spent: list[Any] = []
    exc = AudioGenerationUnavailable(
        "Audio generation failed on every available provider:\n- Radient: refused",
        attempts=(
            JobAttempt(
                route="radient", outcome="failed", reason_class="refused", message="refused"
            ),
        ),
    )
    _fake_cascade(monkeypatch, exc)
    result = await audio_tool.execute_generate_voiceover(
        "call-4", {"text": "Hello there."}, None, None, _context(tmp_path, spent)
    )
    assert result.is_error
    assert _text(result) == (
        "Audio generation failed on every available provider:\n- Radient: refused"
    )
    assert _details(result)["error_type"] == "refused"
    assert _details(result)["attempts"][0]["reason_class"] == "refused"
    assert spent[0].status == "failed" and spent[0].amount_micro is None


@pytest.mark.asyncio
async def test_a_timed_out_job_returns_the_request_id_and_no_artifact(monkeypatch, tmp_path):
    spent: list[Any] = []

    async def fake_run(**kwargs):
        kwargs["handle"].provider = AudioRoute.RADIENT
        kwargs["handle"].request_id = "req-hold"
        kwargs["handle"].model = "radient/voiceover"
        raise AudioGenerationUnavailable(
            "Radient exceeded its 180s generation budget.",
            attempts=(
                JobAttempt(
                    route="radient", outcome="failed", reason_class="timeout", message="budget"
                ),
            ),
        )

    monkeypatch.setattr(audio_tool, "run_audio_cascade", fake_run)
    result = await audio_tool.execute_generate_voiceover(
        "call-5", {"text": "Hello there."}, None, None, _context(tmp_path, spent)
    )
    assert result.is_error
    assert "request req-hold" in _text(result)
    assert _details(result)["request_id"] == "req-hold"
    assert all(not isinstance(block, AttachmentContent) for block in result.content)
    assert spent[0].status == "failed" and spent[0].request_id == "req-hold"


@pytest.mark.asyncio
async def test_a_cache_refusal_degrades_to_a_note_never_a_raise(monkeypatch, tmp_path):
    _fake_cascade(monkeypatch, _outcome())
    monkeypatch.setattr(
        "local_operator.session.attachments.cache_media",
        lambda *args, **kwargs: None,
    )
    result = await audio_tool.execute_generate_voiceover(
        "call-6", {"text": "Hello there."}, None, None, _context(tmp_path)
    )
    # EVERY asset refused: an honest error naming the source, not a caption
    # claiming an attachment that does not exist.
    assert result.is_error
    assert "could not be registered" in _text(result)


def _cancelled_run(*, request_id: str | None = "req-cancel"):
    """A cascade that fills the Radient handle and raises the user's stop."""

    async def fake_run(**kwargs):
        handle = kwargs["handle"]
        handle.provider = AudioRoute.RADIENT
        handle.request_id = request_id
        handle.model = "radient/voiceover"
        # A real Radient handle carries the destination and the resolved
        # bearer: the settle poll cannot reach the job without them.
        handle.base_url = "https://hub.example/v1"
        handle.credential = SecretStr("tok")
        raise AudioGenerationCancelled()

    return fake_run


def _pending_cancel_body() -> dict[str, Any]:
    """The live cancel response for a running Radient job (captured 2026-10-18).

    Pending settlement is spelled by ``cost.basis="quote"`` with no settled
    figure; the surface needs the explicit ``settled``/``cost_usd`` nulls too.
    """
    return {
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


@pytest.mark.asyncio
async def test_a_cancel_result_forwards_the_servers_pending_settlement_fields(
    monkeypatch, tmp_path
):
    """A surface renders "awaiting settlement" only from these explicit fields.

    The server's quote/cancellation/note facts ride through verbatim, and the
    two nulls the UI's predicate requires are present — never a fabricated
    amount, never a charge claim. The cancel receipt's own fields stay.
    """
    spent: list[Any] = []
    body = _pending_cancel_body()

    async def fake_cancel(handle):
        return audio_rungs.AudioCancelReceipt(
            outcome="cancelled",
            vendor_stopped=False,
            reason="provider_has_no_cancel_api",
            server_payload=body,
        )

    monkeypatch.setattr(audio_tool, "run_audio_cascade", _cancelled_run())
    monkeypatch.setattr(audio_rungs, "best_effort_cancel", fake_cancel)
    result = await audio_tool.execute_generate_voiceover(
        "call-cancel-fields", {"text": "Hello there."}, None, None, _context(tmp_path, spent)
    )
    assert result.is_error
    details = _details(result)
    assert details["settled"] is None
    assert details["cost_usd"] is None
    assert details["cost"]["basis"] == "quote"
    assert details["cost"]["amount_usd"] == pytest.approx(0.0216)
    assert details["cost"]["units"] == 540 and details["cost"]["unit"] == "char"
    assert details["cost"]["usage_record_id"] == "6ad466c8e73ab027182a54ff"
    assert details["cancellation"]["vendor_stopped"] is False
    assert details["cancellation"]["reason"] == "provider_has_no_cancel_api"
    assert "no cancellation endpoint" in details["cancellation"]["detail"]
    assert "best-effort" in details["note"]
    # The receipt's existing fields are untouched.
    assert details["stage"] == "cancelled"
    assert details["vendor_stopped"] is False
    assert details["cancel_handle"]["request_id"] == "req-cancel"
    assert details["cancel_reason"] == "provider_has_no_cancel_api"


@pytest.mark.asyncio
async def test_a_cancel_body_without_a_quote_fabricates_no_settlement_fields(monkeypatch, tmp_path):
    """No cost object (BYOK / a bare body) adds no settled or cost keys.

    Forwarding is verbatim plus the contract's pending nulls; anything else is
    the harness inventing a state the server never stated.
    """
    spent: list[Any] = []

    async def fake_cancel(handle):
        return audio_rungs.AudioCancelReceipt(
            outcome="cancelled",
            vendor_stopped=True,
            server_payload={"status": "CANCELLED", "cancellation": {"vendor_stopped": True}},
        )

    monkeypatch.setattr(audio_tool, "run_audio_cascade", _cancelled_run())
    monkeypatch.setattr(audio_rungs, "best_effort_cancel", fake_cancel)
    result = await audio_tool.execute_generate_voiceover(
        "call-cancel-bare", {"text": "Hello there."}, None, None, _context(tmp_path, spent)
    )
    details = _details(result)
    assert "settled" not in details
    assert "cost_usd" not in details
    assert "cost" not in details
    assert details["cancellation"] == {"vendor_stopped": True}


@pytest.mark.asyncio
async def test_a_queued_stop_says_no_charge_and_starts_no_watch(monkeypatch, tmp_path):
    """`vendor_stopped: true` is the ONE branch allowed to promise no charge."""
    spent: list[Any] = []
    spawned: list[Any] = []

    async def fake_cancel(handle):
        return audio_rungs.AudioCancelReceipt(outcome="cancelled", vendor_stopped=True)

    monkeypatch.setattr(audio_tool, "run_audio_cascade", _cancelled_run())
    monkeypatch.setattr(audio_rungs, "best_effort_cancel", fake_cancel)
    result = await audio_tool.execute_generate_voiceover(
        "call-7",
        {"text": "Hello there."},
        None,
        None,
        _context(tmp_path, spent, spawn_background=spawned.append),
    )
    assert result.is_error
    text = _text(result)
    assert "stopped before the provider was called" in text
    assert "you will not be charged" in text
    assert "Checking for settlement" not in text
    assert spawned == [], "a guaranteed stop has nothing to observe"
    assert _details(result)["stage"] == "cancelled"
    assert _details(result)["vendor_stopped"] is True
    assert _details(result)["cancel_handle"]["request_id"] == "req-cancel"
    assert spent[0].status == "cancelled"


@pytest.mark.asyncio
async def test_a_running_stop_starts_the_bounded_settlement_poll(monkeypatch, tmp_path):
    """The receipt says "checking", and a session-owned task is actually started."""
    spent: list[Any] = []
    spawned: list[Any] = []
    journaled: list[dict[str, Any]] = []

    async def fake_cancel(handle):
        return audio_rungs.AudioCancelReceipt(
            outcome="cancelled",
            vendor_stopped=False,
            reason="provider_has_no_cancel_api",
        )

    async def fake_settle(payload, *, record_spend, journal):
        journaled.append({"payload": payload, "record_spend": record_spend, "journal": journal})

    monkeypatch.setattr(audio_tool, "run_audio_cascade", _cancelled_run())
    monkeypatch.setattr(audio_rungs, "best_effort_cancel", fake_cancel)
    from local_operator.audiogen import settle as audio_settle

    monkeypatch.setattr(audio_settle, "settle_cancelled_audio", fake_settle)

    result = await audio_tool.execute_generate_voiceover(
        "call-7b",
        {"text": "Hello there."},
        None,
        None,
        _context(tmp_path, spent, spawn_background=spawned.append, journal_audio_settle=None),
    )
    assert result.is_error
    text = _text(result)
    assert "no cancellation endpoint" in text
    assert "charged if it produces audio" in text
    assert text.endswith("Checking for settlement.")
    assert _details(result)["settling"] is True
    assert _details(result)["cancel_reason"] == "provider_has_no_cancel_api"
    assert len(spawned) == 1
    # The coroutine handed to the session carries the same record id as the
    # cancel row, so a settled charge supersedes it rather than doubling up.
    await spawned[0]
    assert journaled and journaled[0]["payload"]["request_id"] == "req-cancel"
    assert spent[0].status == "cancelled"


@pytest.mark.asyncio
async def test_a_settle_payload_carries_the_same_ledger_key_as_the_cancel_row(
    monkeypatch, tmp_path
):
    spent: list[Any] = []
    payloads: list[dict[str, Any]] = []

    async def fake_cancel(handle):
        return audio_rungs.AudioCancelReceipt(outcome="cancelled", vendor_stopped=False)

    async def fake_settle(payload, *, record_spend, journal):
        payloads.append(payload)

    monkeypatch.setattr(audio_tool, "run_audio_cascade", _cancelled_run())
    monkeypatch.setattr(audio_rungs, "best_effort_cancel", fake_cancel)
    from local_operator.audiogen import settle as audio_settle

    monkeypatch.setattr(audio_settle, "settle_cancelled_audio", fake_settle)
    spawned: list[Any] = []
    await audio_tool.execute_generate_voiceover(
        "call-7c",
        {"text": "Hello there."},
        None,
        None,
        _context(tmp_path, spent, spawn_background=spawned.append),
    )
    assert len(spawned) == 1
    await spawned[0]
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["record_id"] == spent[0].record_id
    assert payload["units"] == float(len("Hello there.")) and payload["unit"] == "chars"
    assert payload["request_id"] == "req-cancel"
    assert payload["provider"] == "radient"
    assert payload["rev"] == 0


@pytest.mark.asyncio
async def test_a_byok_cancel_says_the_provider_may_still_bill(monkeypatch, tmp_path):
    spent: list[Any] = []
    updates: list[Any] = []

    async def fake_run(**kwargs):
        # A BYOK call fills NO server handle; the active route rides progress.
        kwargs["emit"](
            "Generating via OpenAI (gpt-4o-mini-tts)…",
            audio_rungs.progress_details(
                tool="generate_voiceover",
                stage="in_progress",
                provider=str(AudioRoute.OPENAI),
                model="gpt-4o-mini-tts",
            ),
        )
        raise AudioGenerationCancelled()

    monkeypatch.setattr(audio_tool, "run_audio_cascade", fake_run)
    result = await audio_tool.execute_generate_voiceover(
        "call-8",
        {"text": "Hello there."},
        None,
        updates.append,
        _context(tmp_path, spent),
    )
    assert result.is_error
    text = _text(result)
    assert "may still bill" in text and "no cancel API" in text
    assert spent[0].status == "cancelled"


@pytest.mark.asyncio
async def test_invalid_arguments_are_a_model_fault(monkeypatch, tmp_path):
    """Bad args come back as the guard's marked invalid-arguments result.

    The tool's own guard converts the deliberate refusal into a MODEL-fault
    result (never an execution error bucket, never a raise into the loop);
    the marker text is the contract the loop's classifier reads.
    """
    _fake_cascade(monkeypatch, _outcome())

    missing = await audio_tool.execute_generate_sound_effect(
        "call-9", {"prompt": "x"}, None, None, _context(tmp_path)  # duration missing
    )
    assert missing.is_error and "invalid arguments" in _text(missing)
    empty = await audio_tool.execute_generate_voiceover(
        "call-10", {"text": ""}, None, None, _context(tmp_path)
    )
    assert empty.is_error and "invalid arguments" in _text(empty)
    unknown_tone = await audio_tool.execute_generate_voiceover(
        "call-11", {"text": "ok", "tone": "shouty"}, None, None, _context(tmp_path)
    )
    assert unknown_tone.is_error and "invalid arguments" in _text(unknown_tone)


@pytest.mark.asyncio
async def test_a_completed_job_cancel_renders_the_already_completed_pair(monkeypatch, tmp_path):
    """A 409 refusal is "already finished" ink, never an error's ink."""
    spent: list[Any] = []
    spawned: list[Any] = []

    async def fake_cancel(handle):
        return audio_rungs.AudioCancelReceipt(outcome="already_completed")

    monkeypatch.setattr(audio_tool, "run_audio_cascade", _cancelled_run())
    monkeypatch.setattr(audio_rungs, "best_effort_cancel", fake_cancel)
    result = await audio_tool.execute_generate_voiceover(
        "call-7d",
        {"text": "Hello there."},
        None,
        None,
        _context(tmp_path, spent, spawn_background=spawned.append),
    )
    details = _details(result)
    assert details["error_type"] == "media_already_completed"
    assert "already completed" in details["error"]
    assert len(spawned) == 1, "409 must observe the already completed charge"
    from local_operator.audiogen import settle as audio_settle

    async def observed(**kwargs):
        return audio_settle.SettleOutcome(
            status="COMPLETED",
            terminal=True,
            cost_usd=0.042,
            billing_basis="billed",
            usage_record_id="usage-409",
        )

    monkeypatch.setattr(audio_settle, "observe_audio_settlement", observed)
    await spawned[0]
    assert spent[-1].amount_micro == 42000
    assert spent[-1].billing_basis == "billed"
    assert spent[-1].usage_record_id == "usage-409"
    assert spent[-1].record_id == spent[0].record_id
    assert spent[-1].rev == 1


@pytest.mark.asyncio
async def test_a_busy_failure_surfaces_the_wire_word_beside_the_retry_class(monkeypatch, tmp_path):
    """`reason_class` says transient; `error_type` says audio_rate_limited."""
    spent: list[Any] = []
    exc = AudioGenerationUnavailable(
        "Audio generation failed on every available provider:\n- Radient: busy",
        attempts=(
            JobAttempt(
                route="radient",
                outcome="failed",
                reason_class="transient",
                error_type="audio_rate_limited",
                message="Audio generation is busy right now. Wait a moment and try again.",
            ),
        ),
    )
    _fake_cascade(monkeypatch, exc)
    result = await audio_tool.execute_generate_voiceover(
        "call-12", {"text": "Hello there."}, None, None, _context(tmp_path, spent)
    )
    assert result.is_error
    details = _details(result)
    assert details["error_type"] == "audio_rate_limited"
    assert details["attempts"][0]["reason_class"] == "transient"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        None,
        "sha",
        "download",
        "store",
        "vendor_zero",
        "status_billed",
        "immediate_billed",
        "partial_receipt",
        "changed_receipt",
    ],
)
async def test_accepted_radient_charge_survives_artifact_delivery(monkeypatch, tmp_path, failure):
    """Exercise the real tool -> cascade -> HTTP rung, never a paid service."""
    import hashlib

    import httpx

    from local_operator.audiogen import cascade
    from local_operator.session.channel_spend import ChannelSpend, ChannelSpendRecord

    spent = []
    calls = []
    data = b"synthetic verified audio"
    terminal_cost = {
        "amount_usd": 0 if failure == "vendor_zero" else 0.042,
        "basis": "billed",
        "units": 30,
        "unit": "chars",
        "usage_record_id": "usage-known",
    }

    def respond(request):
        calls.append(request.url.path)
        if request.url.path.endswith("/voiceover"):
            assert spent == [], "the accepted row must originate at submit, not delivery"
            return httpx.Response(
                200,
                json={
                    "request_id": "accepted-42",
                    "status": "COMPLETED" if failure == "immediate_billed" else "IN_QUEUE",
                    "cost": (
                        terminal_cost
                        if failure == "immediate_billed"
                        else {"amount_usd": 0.05, "basis": "quote"}
                    ),
                },
            )
        if request.url.path.endswith("/status"):
            assert spent[0].amount_micro == 50000 and spent[0].rev == 0
            assert spent[0].billing_basis == "estimated"
            if failure == "vendor_zero":
                return httpx.Response(
                    200,
                    json={
                        "status": "COMPLETED",
                        "error_type": "audio_failed",
                        "cost": terminal_cost,
                    },
                )
            return httpx.Response(
                200,
                json={
                    "status": "COMPLETED",
                    **(
                        {"cost": terminal_cost}
                        if failure in {"status_billed", "partial_receipt", "changed_receipt"}
                        else {}
                    ),
                },
            )
        if request.url.path.endswith("/result"):
            return httpx.Response(
                200,
                json={
                    "status": "COMPLETED",
                    "cost": (
                        {k: v for k, v in terminal_cost.items() if k != "usage_record_id"}
                        if failure == "partial_receipt"
                        else (
                            {**terminal_cost, "usage_record_id": "usage-revised"}
                            if failure == "changed_receipt"
                            else terminal_cost
                        )
                    ),
                    "audio": {
                        "url": "https://fixture.invalid/audio",
                        "content_type": "audio/mpeg",
                        "sha256": (
                            "0" * 64 if failure == "sha" else hashlib.sha256(data).hexdigest()
                        ),
                    },
                },
            )
        assert spent[-1].amount_micro == 42000, "billed money exists before the CDN read"
        return httpx.Response(
            403 if failure in {"download", "partial_receipt", "changed_receipt"} else 200,
            content=data,
            headers={"content-type": "audio/mpeg"},
        )

    monkeypatch.setattr(
        audio_availability, "radient_available", lambda config_dir=None, **kwargs: True
    )
    monkeypatch.setattr(
        audio_availability,
        "elevenlabs_key",
        lambda config_dir=None, **kwargs: "available-but-unused",
    )
    monkeypatch.setattr(
        audio_availability, "openai_audio_key", lambda config_dir=None, **kwargs: None
    )

    async def pause(seconds):
        return None

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:

        async def dispatch(route, **kwargs):
            assert route == AudioRoute.RADIENT, "accepted jobs cannot buy another provider"
            return await audio_rungs.run_radient(
                kwargs["spec"],
                credential="synthetic",
                base_url="https://fixture.invalid/v1",
                handle=kwargs["handle"],
                emit=kwargs["emit"],
                pause=pause,
                client=client,
            )

        monkeypatch.setattr(cascade, "_run_route", dispatch)
        if failure == "store":
            monkeypatch.setattr(
                "local_operator.session.attachments.cache_media", lambda *a, **k: None
            )
        result = await audio_tool.execute_generate_voiceover(
            "accepted-test", {"text": "hello"}, None, None, _context(tmp_path, spent)
        )
    assert spent, (_text(result), calls)
    assert result.is_error == (failure not in {None, "status_billed", "immediate_billed"})
    if failure == "partial_receipt":
        final_fold = ChannelSpend()
        for row in spent:
            final_fold.apply(row)
        assert final_fold.rows()[0].amount_micro == 42000
        assert final_fold.rows()[0].request_id == "accepted-42"
        assert final_fold.rows()[0].usage_record_id == "usage-known"
    assert [row.rev for row in spent] == ([0, 1, 2] if failure == "changed_receipt" else [0, 1])
    assert len({row.record_id for row in spent}) == 1
    assert all(row.request_id == "accepted-42" for row in spent)
    terminal = spent[-1]
    assert terminal.amount_micro == (0 if failure == "vendor_zero" else 42000)
    assert terminal.billing_basis == "billed" and terminal.cost_source == "server_reported"
    assert terminal.usage_record_id == (
        "usage-revised" if failure == "changed_receipt" else "usage-known"
    )
    assert terminal.units == 30
    assert ChannelSpendRecord.from_details(terminal.to_details()) == terminal
    folded = ChannelSpend()
    for row in spent:
        folded.apply(row)
    assert folded.rows() == [terminal]
    assert folded.total_known_micro() == terminal.amount_micro
    if failure in {"sha", "download", "store", "partial_receipt", "changed_receipt"}:
        assert "billed but the artifact could not be stored" in _text(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cancel_outcome,vendor_stopped",
    [
        ("cancelled", True),
        ("cancelled", False),
        ("already_completed", None),
    ],
)
async def test_accepted_cancel_revises_the_same_charge(
    monkeypatch, tmp_path, cancel_outcome, vendor_stopped
):
    import hashlib

    import httpx

    from local_operator.audiogen import settle

    spent, spawned = [], []
    data = b"settled audio"

    async def accepted_then_cancelled(**kwargs):
        handle = kwargs["handle"]
        handle.provider = AudioRoute.RADIENT
        handle.request_id = "accepted-cancel"
        handle.base_url = "https://fixture.invalid/v1"
        handle.credential = SecretStr("synthetic")
        audio_rungs._observe_cost(
            handle, {"cost": {"amount_usd": 0.05, "basis": "quote"}}, terminal=False
        )
        assert spent[0].amount_micro == 50000 and spent[0].rev == 0
        raise AudioGenerationCancelled("stopped")

    async def cancel(handle):
        return audio_rungs.AudioCancelReceipt(outcome=cancel_outcome, vendor_stopped=vendor_stopped)

    paths = []

    def respond(request):
        paths.append(request.url.path)
        if request.url.path.endswith("/status"):
            return httpx.Response(200, json={"status": "COMPLETED"})
        if request.url.path.endswith("/result"):
            return httpx.Response(
                200,
                json={
                    "status": "COMPLETED",
                    "cost": {
                        "amount_usd": 0.042,
                        "basis": "billed",
                        "units": 31,
                        "unit": "chars",
                        "usage_record_id": "usage-cancel",
                    },
                    "audio": {
                        "url": "https://fixture.invalid/audio",
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "content_type": "audio/mpeg",
                    },
                },
            )
        return httpx.Response(200, content=data, headers={"content-type": "audio/mpeg"})

    monkeypatch.setattr(audio_tool, "run_audio_cascade", accepted_then_cancelled)
    monkeypatch.setattr(audio_rungs, "best_effort_cancel", cancel)
    original = settle.observe_audio_settlement
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:

        async def observe(**kwargs):
            return await original(**kwargs, client=client)

        monkeypatch.setattr(settle, "observe_audio_settlement", observe)
        result = await audio_tool.execute_generate_voiceover(
            "cancel-test",
            {"text": "hello"},
            None,
            None,
            _context(tmp_path, spent, spawn_background=spawned.append),
        )
        assert result.is_error
        if vendor_stopped:
            assert spawned == [] and paths == []
            assert spent[-1].status == "cancelled" and spent[-1].amount_micro == 0
        else:
            assert (
                len(spent) == 1 and spent[0].amount_micro == 50000
            ), "running cancel retains acceptance"
            assert len(spawned) == 1
            await spawned[0]
            assert spent[-1].amount_micro == 42000 and spent[-1].usage_record_id == "usage-cancel"
            assert spent[-1].units == 31
            assert paths == ["/v1/tools/audio/status", "/v1/tools/audio/result", "/audio"]
        assert [row.rev for row in spent] == [0, 1]
        assert len({row.record_id for row in spent}) == 1
        assert spent[-1].billing_basis == "billed"


# Batch envelopes validate before approval; individual provider rejection is a
# result, not a reason to discard an otherwise valid paid batch.
@pytest.mark.parametrize(
    "args",
    [
        {},
        {"prompt": "click"},
        {"duration_s": 1},
        {"sounds": []},
        {"sounds": [{"prompt": "click", "duration_s": 1}], "prompt": "other"},
        {"sounds": [{"prompt": "click", "duration_s": 1}], "duration_s": None},
        {"sounds": [{"prompt": "click", "duration_s": 1}], "filename": "inert-stem"},
        {"sounds": [{"prompt": "click", "duration_s": 1}] * 51},
    ],
)
def test_batch_envelope_refused_before_approval(args):
    assert audio_tool._refuse_sound_effect_args(args, frozenset())


def test_batch_names_are_filename_stems_with_a_real_extension():
    params = audio_tool.GenerateSoundEffectParams.model_validate(
        {"sounds": [{"prompt": " bad / sound ", "duration_s": 31}]}
    )
    assert params.sounds and params.sounds[0].duration_s == 31
    assert media_filename(" bad / sound ", None, kind="audio") == "bad-sound.mp3"
    assert media_filename("anything", None, suggested="../", kind="audio") == "anything.mp3"
    assert media_filename("anything", None, suggested=" named ", kind="audio") == "named.mp3"


def test_batch_approval_is_one_summed_estimate(monkeypatch):
    monkeypatch.setattr(audio_tool, "_preferred_route", lambda direction: AudioRoute.RADIENT)
    args = {"sounds": [{"prompt": "click", "duration_s": 0.5}] * 10}
    text = audio_tool._describe_sound_effect_approval(args, ".")
    assert "10 sound effects" in text
    assert "estimated total $0.0200" in text
    assert "billed figures are authoritative" in text


@pytest.mark.asyncio
async def test_batch_ten_named_attachments_cost_and_max_two(monkeypatch):
    from dataclasses import replace

    active = peak = 0
    updates = []

    async def cascade(**kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        spec = kwargs["spec"]
        assert spec.batch_submission
        await asyncio.sleep(0.001)
        active -= 1
        return _outcome(
            assets=(replace(_asset(0.5), data=b"ID3" + spec.text.encode()),),
            generation_id=spec.text,
            cost_usd=0.002,
        )

    monkeypatch.setattr(audio_tool, "run_audio_cascade", cascade)
    result = await audio_tool.execute_generate_sound_effect(
        "batch",
        {
            "sounds": [
                {"name": "duplicate" if n < 2 else None, "prompt": f"sound-{n}", "duration_s": 0.5}
                for n in range(10)
            ]
        },
        None,
        updates.append,
        None,
    )
    assert not result.is_error
    assert peak == 2
    blocks = [b for b in result.content if isinstance(b, AttachmentContent)]
    assert len(blocks) == 10
    assert [b.name for b in blocks][:3] == [
        "duplicate.mp3",
        "duplicate.mp3",
        "sound-2.mp3",
    ]
    details = _details(result)
    assert details["completed"] == 10 and details["failed"] == details["cancelled"] == 0
    assert details["cost_usd"] == pytest.approx(0.02)
    for n, row in enumerate(details["artifacts"]):
        assert row["name"] == blocks[n].name
        assert row["digest"] == blocks[n].attachment
        assert row["request_id"] == f"sound-{n}"
        assert row["cost"] == {
            "amount_usd": 0.002,
            "basis": "billed",
            "units": 1.0,
            "unit": "seconds",
        }
        assert row["error"] is None
        assert row["duration_s"] == 0.5 and row["format"] == "mp3"
    assert "1/10 duplicate.mp3: completed" in _text(result)
    assert any(
        "10/10 sound-9" in b.text for u in updates for b in u.content if isinstance(b, TextContent)
    )


@pytest.mark.asyncio
async def test_batch_partial_failure_uses_actual_http_rejection(monkeypatch):
    import hashlib
    import json

    import httpx

    from local_operator.audiogen import cascade, rungs

    monkeypatch.setattr(cascade.audio_availability, "radient_available", lambda cfg=None: True)
    monkeypatch.setattr(cascade.audio_availability, "elevenlabs_key", lambda cfg=None: None)
    monkeypatch.setattr(cascade.audio_availability, "openai_audio_key", lambda cfg=None: None)
    submissions = []
    audio = b"ID3mock-http-audio"

    def respond(request):
        if request.method == "POST":
            body = json.loads(request.content)
            submissions.append(body)
            if body["duration_seconds"] == 31:
                return httpx.Response(
                    400,
                    json={
                        "code": "audio_rejected",
                        "error": "duration_s must be between 0.5 and 30",
                    },
                )
            return httpx.Response(
                200,
                json={
                    "request_id": body["prompt"],
                    "status": "COMPLETED",
                    "model": "sfx",
                    "cost": {"amount_usd": 0.002, "basis": "billed", "units": 1, "unit": "seconds"},
                },
            )
        if request.url.path.endswith("/result"):
            return httpx.Response(
                200,
                json={
                    "status": "COMPLETED",
                    "audio": {
                        "url": "https://fixture.invalid/media.mp3",
                        "content_type": "audio/mpeg",
                        "duration_seconds": 0.5,
                        "sha256": hashlib.sha256(audio).hexdigest(),
                    },
                },
            )
        assert request.url.path == "/media.mp3"
        return httpx.Response(200, content=audio)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:

        async def route(route, **kwargs):
            assert route == AudioRoute.RADIENT
            return await rungs.run_radient(
                kwargs["spec"],
                credential="fixture-token",
                base_url="https://fixture.invalid/v1",
                handle=kwargs["handle"],
                emit=kwargs["emit"],
                pause=kwargs["pause"],
                client=client,
            )

        monkeypatch.setattr(cascade, "_run_route", route)
        result = await audio_tool.execute_generate_sound_effect(
            "partial",
            {
                "sounds": [
                    {"prompt": f"sound-{n}", "duration_s": 31 if n == 2 else 0.5} for n in range(4)
                ]
            },
            None,
            None,
            None,
        )
    assert not result.is_error
    assert len(submissions) == 4
    assert len([b for b in result.content if isinstance(b, AttachmentContent)]) == 3
    assert _details(result)["failed"] == 1
    rejected = _details(result)["artifacts"][2]
    assert rejected["error_type"] == "audio_rejected"
    assert rejected["http_status"] == 400
    assert "duration_s must be between 0.5 and 30" in rejected["error"]
    assert _details(result)["cost_usd"] == pytest.approx(0.006)
    assert _details(result)["cost"]["basis"] == "billed"
    assert _details(result)["cost_complete"] is False
    assert _details(result)["artifacts"][0]["units"] == 1


def test_hub_reason_error_appends_the_reason_and_keeps_typed_fields() -> None:
    """The refused-submit upgrade: reason in the sentence, classification intact."""
    from local_operator.clients._http import APIError

    exc = APIError(
        "This audio request could not be accepted.",
        status_code=400,
        code="audio_rejected",
        details={"reason": "duration_seconds must be between 0.5 and 30 seconds, got 31"},
        body='{"code": "audio_rejected"}',
    )
    wrapped = audio_rungs._hub_reason_error(exc)
    assert str(wrapped) == (
        "This audio request could not be accepted. "
        "(duration_seconds must be between 0.5 and 30 seconds, got 31)"
    )
    assert wrapped.status_code == 400
    assert wrapped.code == "audio_rejected"
    assert wrapped.details == exc.details
    assert wrapped.body == exc.body


def test_hub_reason_error_omits_without_a_reason_or_when_already_present() -> None:
    """Additive only: an older hub or an already-actionable message is unchanged."""
    from local_operator.clients._http import APIError

    plain = APIError("This audio request could not be accepted.", status_code=400)
    assert audio_rungs._hub_reason_error(plain) is plain
    empty = APIError(
        "This audio request could not be accepted.", status_code=400, details={"reason": "  "}
    )
    assert audio_rungs._hub_reason_error(empty) is empty
    already = APIError(
        "Refused: duration_seconds must be between 0.5 and 30 seconds, got 31",
        status_code=400,
        details={"reason": "duration_seconds must be between 0.5 and 30 seconds, got 31"},
    )
    assert audio_rungs._hub_reason_error(already) is already


@pytest.mark.asyncio
async def test_batch_rejected_item_surfaces_the_servers_details_reason(monkeypatch):
    """A 400 ``audio_rejected`` batch item shows ``details.reason`` verbatim.

    The live hub sends the generic sentence in ``error`` and the actionable
    parameter explanation in ``details.reason``; the batch row's ``error`` is
    composed from the attempt message, so the reason must ride that message.
    Scripted HTTP only — no provider call.
    """
    import hashlib
    import json

    import httpx

    from local_operator.audiogen import cascade, rungs

    monkeypatch.setattr(cascade.audio_availability, "radient_available", lambda cfg=None: True)
    monkeypatch.setattr(cascade.audio_availability, "elevenlabs_key", lambda cfg=None: None)
    monkeypatch.setattr(cascade.audio_availability, "openai_audio_key", lambda cfg=None: None)
    submitted = []
    audio = b"ID3mock-http-audio"
    reason = "duration_seconds must be between 0.5 and 30 seconds, got 31"

    def respond(request):
        if request.method == "POST":
            body = json.loads(request.content)
            submitted.append(body)
            if body["duration_seconds"] == 31:
                return httpx.Response(
                    400,
                    json={
                        "code": "audio_rejected",
                        "error": "This audio request could not be accepted.",
                        "details": {"reason": reason},
                    },
                )
            return httpx.Response(
                200,
                json={
                    "request_id": body["prompt"],
                    "status": "COMPLETED",
                    "model": "sfx",
                    "cost": {
                        "amount_usd": 0.002,
                        "basis": "billed",
                        "units": 1,
                        "unit": "seconds",
                    },
                },
            )
        if request.url.path.endswith("/result"):
            return httpx.Response(
                200,
                json={
                    "status": "COMPLETED",
                    "audio": {
                        "url": "https://fixture.invalid/media.mp3",
                        "content_type": "audio/mpeg",
                        "duration_seconds": 0.5,
                        "sha256": hashlib.sha256(audio).hexdigest(),
                    },
                },
            )
        assert request.url.path == "/media.mp3"
        return httpx.Response(200, content=audio)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:

        async def route(route, **kwargs):
            assert route == AudioRoute.RADIENT
            return await rungs.run_radient(
                kwargs["spec"],
                credential="test-credential",
                base_url="https://fixture.invalid/v1",
                handle=kwargs["handle"],
                emit=kwargs["emit"],
                pause=kwargs["pause"],
                client=client,
            )

        monkeypatch.setattr(cascade, "_run_route", route)
        result = await audio_tool.execute_generate_sound_effect(
            "partial",
            {
                "sounds": [
                    {"prompt": f"sound-{n}", "duration_s": 31 if n == 2 else 0.5} for n in range(4)
                ]
            },
            None,
            None,
            None,
        )
    assert not result.is_error
    assert len(submitted) == 4
    assert len([b for b in result.content if isinstance(b, AttachmentContent)]) == 3
    rejected = _details(result)["artifacts"][2]
    assert rejected["status"] == "failed"
    assert rejected["error_type"] == "audio_rejected"
    assert rejected["http_status"] == 400
    assert reason in rejected["error"]
    assert "could not be accepted" in rejected["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("refuse_all", [False, True])
async def test_batch_cache_refusal_is_isolated_and_all_refused_errors(monkeypatch, refuse_all):
    from local_operator.session import attachments

    original = attachments.cache_media
    registered = []

    def cache(*args, **kwargs):
        registered.append(kwargs["name"])
        if refuse_all or kwargs["name"] == "sound-0.mp3":
            return None
        return original(*args, **kwargs)

    async def cascade(**kwargs):
        return _outcome(cost_usd=0.002)

    monkeypatch.setattr(attachments, "cache_media", cache)
    monkeypatch.setattr(audio_tool, "run_audio_cascade", cascade)
    result = await audio_tool.execute_generate_sound_effect(
        "cache",
        {"sounds": [{"prompt": f"sound-{n}", "duration_s": 0.5} for n in range(4)]},
        None,
        None,
        None,
    )
    assert len(registered) == 4
    assert result.is_error is refuse_all
    assert _details(result)["completed"] == (0 if refuse_all else 3)
    # Billed output refused by storage still belongs in the total.
    assert _details(result)["cost_usd"] == pytest.approx(0.008)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_task", [False, True])
async def test_batch_stop_does_not_start_queue_and_retains_completed(monkeypatch, cancel_task):
    from local_operator.harness.types import AbortSignal

    signal = AbortSignal()
    started = []
    blocked = asyncio.Event()

    async def cascade(**kwargs):
        text = kwargs["spec"].text
        started.append(text)
        if text == "sound-0":
            return _outcome(cost_usd=0.002)
        blocked.set()
        if cancel_task:
            await asyncio.Event().wait()
        else:
            signal.abort()
            raise AudioGenerationCancelled()

    monkeypatch.setattr(audio_tool, "run_audio_cascade", cascade)
    task = asyncio.ensure_future(
        audio_tool.execute_generate_sound_effect(
            "cancel",
            {"sounds": [{"prompt": f"sound-{n}", "duration_s": 0.5} for n in range(10)]},
            signal,
            None,
            None,
        )
    )
    await blocked.wait()
    if cancel_task:
        task.cancel()
    result = await task
    assert not result.is_error
    assert _details(result)["completed"] == 1
    assert _details(result)["cancelled"] == 9
    assert len(started) <= 3
    rows = _details(result)["artifacts"]
    assert rows[-1]["started"] is False
    assert rows[-1]["error"] == "Never started; no provider call"
    assert "billed" in _text(result) or "stopped" in _text(result)


@pytest.mark.asyncio
async def test_batch_deadline_returns_partial_and_marks_unstarted(monkeypatch):
    calls = []

    async def cascade(**kwargs):
        text = kwargs["spec"].text
        calls.append(text)
        if text == "sound-0":
            return _outcome(cost_usd=0.002)
        await asyncio.Event().wait()

    monkeypatch.setattr(audio_tool, "SOUND_BATCH_TIMEOUT_S", 0.01)
    monkeypatch.setattr(audio_tool, "run_audio_cascade", cascade)
    result = await audio_tool.execute_generate_sound_effect(
        "slow",
        {"sounds": [{"prompt": f"sound-{n}", "duration_s": 0.5} for n in range(50)]},
        None,
        None,
        None,
    )
    assert not result.is_error
    assert len(calls) == 3  # one completed, then two in flight; no remaining submissions
    assert _details(result)["completed"] == 1
    assert _details(result)["cancelled"] == 49
    assert sum(row["started"] for row in _details(result)["artifacts"]) == 3
    assert "Batch time budget reached" in _text(result)
    assert "not a guaranteed vendor cancellation" in _text(result)


@pytest.mark.asyncio
async def test_batch_server_units_and_mixed_estimates_remain_honest(monkeypatch):
    async def cascade(**kwargs):
        spec = kwargs["spec"]
        if spec.text == "billed":
            kwargs["handle"].cost = {
                "amount_usd": 0.007,
                "basis": "billed",
                "units": 3.5,
                "unit": "seconds",
            }
            return _outcome(cost_usd=0.007)
        return _outcome(cost_usd=0.004, billing_basis="estimated", cost_source="rate_table")

    monkeypatch.setattr(audio_tool, "run_audio_cascade", cascade)
    result = await audio_tool.execute_generate_sound_effect(
        "costs",
        {"sounds": [{"prompt": name, "duration_s": 0.5} for name in ("billed", "estimated")]},
        None,
        None,
        None,
    )
    details = _details(result)
    assert details["cost_usd"] == pytest.approx(0.011)
    assert details["cost"]["basis"] == "estimated"
    assert details["cost_complete"] is True
    assert details["artifacts"][0]["units"] == 3.5  # server fact, not client rounding
    assert details["artifacts"][0]["cost"]["basis"] == "billed"
    assert details["artifacts"][1]["cost"]["basis"] == "estimated"


def test_batch_approval_keeps_known_subtotal_for_unpriceable_duration(monkeypatch):
    monkeypatch.setattr(audio_tool, "_preferred_route", lambda direction: AudioRoute.RADIENT)
    text = audio_tool._describe_sound_effect_approval(
        {
            "sounds": [
                {"prompt": "valid", "duration_s": 0.5},
                {"prompt": "invalid", "duration_s": 0},
            ]
        },
        ".",
    )
    assert "Generate 2 sound effects" in text
    assert "estimated known subtotal $0.0020" in text
    assert "1 item(s) unpriced" in text
    single = audio_tool._describe_sound_effect_approval(
        {
            "sounds": [
                {"prompt": "valid", "duration_s": 0.5},
            ]
        },
        ".",
    )
    assert "Generate 1 sound effect via" in single


def test_batch_names_and_limit_are_explicitly_bounded():
    base = {"prompt": "sound", "duration_s": 0.5}
    valid = audio_tool.GenerateSoundEffectParams.model_validate(
        {"sounds": [{**base, "name": "x" * 100}]}
    )
    assert valid.sounds and valid.sounds[0].name == "x" * 100
    refusal = audio_tool._refuse_sound_effect_args(
        {"sounds": [{**base, "name": "x" * 101}]}, frozenset()
    )
    assert refusal and "100" in refusal
    refusal = audio_tool._refuse_sound_effect_args({"sounds": [base] * 51}, frozenset())
    assert refusal and "50" in refusal
    # The stem cap is shared with every other lane: 60 chars at a word
    # boundary, a single overlong word hard-cut, never a trailing separator.
    assert media_filename("x" * 99 + " -tail", None, kind="audio") == "x" * 60 + ".mp3"
    capped = media_filename("word " * 40, None, kind="audio")
    assert len(capped.removesuffix(".mp3")) <= 60 and not capped.startswith("-")
    description = audio_tool.GenerateSoundEffectParams.model_json_schema()["description"]
    assert "prompt + duration_s" in description and "never both" in description


@pytest.mark.asyncio
async def test_batch_unknown_total_has_no_basis(monkeypatch):
    async def cascade(**kwargs):
        raise AudioGenerationUnavailable("No figure or asset reported")

    monkeypatch.setattr(audio_tool, "run_audio_cascade", cascade)
    result = await audio_tool.execute_generate_sound_effect(
        "unknown",
        {"sounds": [{"prompt": "sound", "duration_s": 0.5}]},
        None,
        None,
        None,
    )
    assert result.is_error
    assert _details(result)["cost"] == {"amount_usd": None, "basis": None}
    assert _details(result)["cost_complete"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_type,expected,expected_ext",
    [("audio/wav", "wav", "wav"), ("audio/unknown", None, "mp3")],
)
async def test_batch_identical_bytes_keep_names_ids_and_actual_format(
    monkeypatch, content_type, expected, expected_ext
):
    from dataclasses import replace

    async def cascade(**kwargs):
        return _outcome(
            generation_id=kwargs["spec"].text,
            assets=(replace(_asset(), format=None, content_type=content_type),),
        )

    monkeypatch.setattr(audio_tool, "run_audio_cascade", cascade)
    result = await audio_tool.execute_generate_sound_effect(
        "same-bytes",
        {
            "sounds": [
                {"prompt": name, "name": name, "duration_s": 0.5} for name in ("First", "Second")
            ]
        },
        None,
        None,
        None,
    )
    rows = _details(result)["artifacts"]
    blocks = [block for block in result.content if isinstance(block, AttachmentContent)]
    assert len(rows) == len(blocks) == 2
    assert [row["name"] for row in rows] == [
        f"first.{expected_ext}",
        f"second.{expected_ext}",
    ]
    assert [row["request_id"] for row in rows] == ["First", "Second"]
    assert rows[0]["digest"] == rows[1]["digest"]
    assert [row["format"] for row in rows] == [expected, expected]
    assert [block.name for block in blocks] == [
        f"first.{expected_ext}",
        f"second.{expected_ext}",
    ]
