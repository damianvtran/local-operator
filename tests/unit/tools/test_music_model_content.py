"""What the MODEL reads from the music tools: tariff, quote, per-track cost and saved file.

Regression for installed-runtime QA finding Q1 (v0.101.48): content carried only
"Music" / "Music ready — job …" while price, cost and files lived in the
UI-only ``details``, so the agent could not report them. ``details`` must stay
byte-identical (the UI contract); these tests pin both halves.
"""

from __future__ import annotations

import asyncio
import copy

import pytest

from local_operator.harness.jobs import AsyncJobManager
from local_operator.harness.types import ToolContext
from local_operator.music import jobs as music_jobs
from local_operator.music.digest import job_text, money
from local_operator.tools import music_tool

# The live /tools/music/models payload, captured 2026-11-11 from the server.
LIVE_MODELS = {
    "models": [
        {
            "attribution": "Powered by ElevenLabs Music",
            "disabled_reason": "",
            "duration_ms": {"max": 300000, "min": 3000},
            "enabled": True,
            "formats": ["mp3"],
            "id": "radient/music",
            "operation_disabled_reasons": {
                "compose": "",
                "separate_stems": "unpriced_operation",
                "upload_reference": "unpriced_operation",
            },
            "operations": {"compose": True, "separate_stems": False, "upload_reference": False},
            "pricing": {
                "basis": "fixed_quote",
                "price_version": "music-fixed-requested-ms@2026-11-08",
                "unit": "millisecond",
                "unit_price_usd": 2.5e-06,
            },
            "provider": "radient",
        }
    ]
}


def _client(models):
    class Client:
        def __init__(self, **kwargs):
            pass

        async def request(self, action, **kwargs):
            assert action == "models"
            return copy.deepcopy(models)

        async def close(self):
            pass

    return Client


def _text(result) -> str:
    return "".join(getattr(block, "text", "") for block in result.content)


def test_money_keeps_sub_cent_quotes_and_two_decimals():
    assert [money(v) for v in (0.15, 0.0075, 0, 1.5)] == ["$0.15", "$0.0075", "$0.00", "$1.50"]


@pytest.mark.asyncio
async def test_models_content_reports_live_tariff_limits_operations_and_attribution(monkeypatch):
    monkeypatch.setattr(music_tool, "MusicClient", _client(LIVE_MODELS))
    result = await music_tool.execute_music_status("c", {"action": "models"}, None, None, None)
    text = _text(result)
    assert text == (
        "Music models:\n"
        "- radient/music: enabled; price $0.15/min per variation ($0.0025/s; fixed quote on "
        "requested duration, music-fixed-requested-ms@2026-11-08).\n"
        "  Duration 3 s–300 s; formats mp3.\n"
        "  Operations available: compose; unavailable: separate_stems (unpriced_operation), "
        "upload_reference (unpriced_operation).\n"
        "  Attribution: Powered by ElevenLabs Music"
    )
    assert result.details == LIVE_MODELS


@pytest.mark.asyncio
async def test_models_content_names_disabled_reason(monkeypatch):
    disabled = copy.deepcopy(LIVE_MODELS)
    disabled["models"][0].update(
        enabled=False,
        disabled_reason="commercial_authorization_required",
        operations={"compose": False},
        operation_disabled_reasons={"compose": "commercial_authorization_required"},
    )
    monkeypatch.setattr(music_tool, "MusicClient", _client(disabled))
    text = _text(await music_tool.execute_music_status("c", {"action": "models"}, None, None, None))
    assert "- radient/music: disabled (commercial_authorization_required); price $0.15/min" in text
    assert (
        "Operations available: none; unavailable: compose (commercial_authorization_required)."
        in text
    )


@pytest.mark.asyncio
async def test_start_content_quotes_duration_times_variations_and_how_to_follow(monkeypatch):
    blocked = asyncio.Event()
    manager = AsyncJobManager()

    async def journal(record):
        pass

    def parked_runner(context, record):
        async def run(watcher_id, signal, report):
            await blocked.wait()
            return ""

        return run

    monkeypatch.setattr(music_tool, "MusicClient", _client(LIVE_MODELS))
    monkeypatch.setattr(music_tool, "runner", parked_runner)
    context = ToolContext(jobs=manager, journal_music_job=journal, session_id="synthetic")
    try:
        result = await music_tool.execute_generate_music(
            "call", {"prompt": "Piano", "duration_ms": 30000, "variations": 2}, None, None, context
        )
        assert result.details is not None
        job = result.details["job_id"]
        assert _text(result) == (
            f"Music job {job} started (radient/music, 30 s × 2 variations). Estimated cost "
            "$0.15 at $0.15/min per variation (fixed quote); the server's billed cost is "
            f'authoritative. It runs in the background; use music_status(job_id="{job}") or '
            f'wait(job_id="{job}") for the tracks, costs and saved files.'
        )
        # The UI contract is unchanged.
        assert result.details == {
            "job_id": job,
            "stage": "queued",
            "material": "music",
            "model": "radient/music",
            "tracks": [],
            "cancel_support": "queued_only",
        }
        await asyncio.sleep(0)  # let the parked job start before teardown
    finally:
        blocked.set()
        await manager.dispose()


BILLED = {"v": 1, "amount_usd": 0.075, "basis": "billed", "units": 30000, "unit": "millisecond"}


def _two_track_record(**extra):
    return {
        "job": "894462259df4",
        "state": "completed",
        "attribution": "Powered by ElevenLabs Music",
        "tracks": [
            {
                "id": "one",
                "title": "Rain Piano",
                "status": "COMPLETED",
                "settled": True,
                "cost": BILLED,
                "artifacts_registered": True,
            },
            {
                "id": "two",
                "status": "COMPLETED",
                "settled": True,
                "error_type": "audio_failed",
                "cost": {**BILLED, "amount_usd": 0, "units": 0},
            },
        ],
        "artifacts": [
            {
                "track_id": "one",
                "name": "rain-piano-1.mp3",
                "digest": "sha256-synthetic",
                "duration_s": 30.0,
                # A signed URL must never reach the model even if a record had one.
                "url": "https://cdn.invalid/a?sig=secret",
            }
        ],
        **extra,
    }


@pytest.mark.asyncio
async def test_status_content_lists_tracks_files_costs_and_failed_reason(monkeypatch):
    record = _two_track_record()
    monkeypatch.setattr(music_tool, "read_records", lambda directory: {record["job"]: record})
    result = await music_tool.execute_music_status(
        "c", {"job_id": record["job"]}, None, None, ToolContext(jobs=AsyncJobManager())
    )
    assert _text(result) == (
        "Music job 894462259df4: completed (2 tracks, 1 saved).\n"
        '- Track 1 "Rain Piano": completed, saved as rain-piano-1.mp3 (attachment '
        "sha256-synthetic, 30 s); cost $0.075 billed.\n"
        "- Track 2: failed (audio_failed); cost $0.00 billed.\n"
        "Total billed: $0.075.\n"
        "Attribution: Powered by ElevenLabs Music"
    )
    assert result.details == _two_track_record()


def test_finished_after_cancel_says_charged_and_retained():
    text = job_text(_two_track_record(cancellation_requested=True))
    assert text.endswith(
        "Cancellation was requested, but the composition had already started: it finished "
        "and was charged, and its audio is retained.\nAttribution: Powered by ElevenLabs Music"
    )
    assert "Total billed: $0.075." in text


def test_pending_quote_is_estimated_not_billed_and_unknown_is_not_zero():
    text = job_text(
        {
            "job": "j",
            "state": "cancel_requested",
            "cancellation_requested": True,
            "tracks": [
                {
                    "id": "a",
                    "status": "IN_PROGRESS",
                    "settled": False,
                    "cost": {**BILLED, "basis": "estimated"},
                },
                {"id": "b", "status": "IN_QUEUE", "settled": False},
            ],
        }
    )
    assert "- Track 1: in progress; cost $0.075 estimated (quote, not yet billed)." in text
    assert "- Track 2: in queue; cost unknown." in text
    assert "Total billed: $0.00 so far (2 track(s) not yet settled)." in text
    assert "running generation may still complete and be charged" in text


@pytest.mark.asyncio
async def test_all_failed_job_failure_message_carries_the_digest():
    async def journal(record):
        pass

    class Client:
        def __init__(self, **kwargs):
            pass

        async def close(self):
            pass

    record = {
        "job": "failed-job",
        "state": "running",
        "tracks": [
            {
                "id": "paid",
                "settled": True,
                "status": "FAILED",
                "error_type": "provider_failed",
                "cost": {**BILLED, "amount_usd": 0},
            }
        ],
    }
    context = ToolContext(journal_music_job=journal)
    with pytest.raises(RuntimeError) as raised:
        await music_jobs.runner(context, record, client_factory=Client)(
            "w", None, lambda value: None
        )
    assert str(raised.value) == (
        "music_no_playable_results\n"
        "Music job failed-job: failed (1 track, 0 saved).\n"
        "- Track 1: failed (provider_failed); cost $0.00 billed.\n"
        "Total billed: $0.00."
    )
