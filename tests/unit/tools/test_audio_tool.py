"""The audio tools: frozen schema, gates, results, failure, cancel, ledger.

No provider calls anywhere: the cascade is faked at the tool module's
``run_audio_cascade`` seam, and the attachment store is redirected into
``tmp_path``. What is under test is the tool CONTRACT — the schema the model
sees, which sessions get which tool, the lane-D result shape, the failure and
cancellation receipts, and the ``audio`` channel row each outcome hands the
ledger.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr

from local_operator.artifacts import ArtifactKind, JobAttempt, JobOutcome
from local_operator.audiogen import AudioRoute
from local_operator.audiogen import availability as audio_availability
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
    assert set(properties) == {"prompt", "duration_s", "loop", "prompt_influence", "model"}
    assert set(schema["required"]) == {"prompt", "duration_s"}, "duration has no auto-duration"
    assert properties["prompt"]["maxLength"] == 600
    assert properties["duration_s"]["minimum"] == 0.5
    assert properties["duration_s"]["maximum"] == 30.0
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
        "call-3", {"prompt": "a clap", "duration_s": 4}, None, None, _context(tmp_path, spent)
    )
    assert not result.is_error
    assert "sound effect" in _text(result)
    assert spent[0].unit == "seconds" and spent[0].units == 4.0


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
    [None, "sha", "download", "store", "vendor_zero", "status_billed", "immediate_billed"],
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
                    **({"cost": terminal_cost} if failure == "status_billed" else {}),
                },
            )
        if request.url.path.endswith("/result"):
            return httpx.Response(
                200,
                json={
                    "status": "COMPLETED",
                    "cost": terminal_cost,
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
            403 if failure == "download" else 200,
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
    assert [row.rev for row in spent] == [0, 1]
    assert len({row.record_id for row in spent}) == 1
    assert all(row.request_id == "accepted-42" for row in spent)
    terminal = spent[-1]
    assert terminal.amount_micro == (0 if failure == "vendor_zero" else 42000)
    assert terminal.billing_basis == "billed" and terminal.cost_source == "server_reported"
    assert terminal.usage_record_id == "usage-known" and terminal.units == 30
    assert ChannelSpendRecord.from_details(terminal.to_details()) == terminal
    folded = ChannelSpend()
    for row in spent:
        folded.apply(row)
    assert folded.rows() == [terminal]
    assert folded.total_known_micro() == terminal.amount_micro
    if failure in {"sha", "download", "store"}:
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
