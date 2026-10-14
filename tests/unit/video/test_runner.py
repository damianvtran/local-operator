"""The detached runner: records at every transition, cancel, completion text.

The walk itself is faked at ``video.jobs.run_video_cascade`` (one level below
what the tool registers), so these tests pin the RUNNER's contract: the
durable record sequence, the accept hook's ``provider_ref``, the two cancel
spellings, the completion receipt/text, and the wake's lifecycle.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.artifacts import MediaAsset
from local_operator.harness.types import AbortSignal
from local_operator.video import jobs as video_jobs


class _FakeScheduler:
    """The slice of ``WakeScheduler`` the wake handle reads and writes."""

    def __init__(self, rows: list[Any] | None = None) -> None:
        self.schedules = list(rows or [])
        self.updates: list[list[Any]] = []

    async def update(self, rows: list[Any]) -> None:
        self.updates.append(rows)
        self.schedules = rows


def _video_asset() -> MediaAsset:
    return MediaAsset(
        data=b"\x00\x00\x00\x18ftypmp42",
        content_type="video/mp4",
        source_url="https://example.com/v.mp4",
        duration_s=8.0,
    )


def _outcome() -> SimpleNamespace:
    return SimpleNamespace(
        route="fal",
        model="minimax/h3-max/image-to-video",
        assets=[_video_asset()],
        cost_usd=None,
        cost_source=None,
        billing_basis=None,
    )


@pytest.mark.asyncio
async def test_success_path_records_cancel_wake_and_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal: list[dict[str, Any]] = []
    outputs: list[str] = []
    progress: list[dict[str, Any]] = []

    async def fake_cascade(**kwargs: Any) -> Any:
        handle = kwargs["handle"]
        # The rung's accept moment: job exists, handle filled.
        handle.provider = "fal"
        handle.request_id = "req-1"
        handle.model = "minimax/h3-max/image-to-video"
        handle.status_url = "https://queue.fal.run/x/requests/req-1/status"
        handle.result_url = "https://queue.fal.run/x/requests/req-1"
        handle.cancel_url = "https://queue.fal.run/x/requests/req-1/cancel"
        await kwargs["on_accept"](handle)
        emit = kwargs["emit"]
        if emit is not None:
            emit("Generating video via FAL: queued — 5s", {"stage": "queued"})
        # The rung CLEARS the handle the instant the job is terminal (the
        # cancel semantics: nothing left to cancel) — so the completed row
        # must still carry the refs captured at ACCEPT time (V2: the QA drill
        # lost them on the cleared handle and the §1.5 refetch could then
        # never poll the provider).
        handle.clear()
        return _outcome()

    monkeypatch.setattr(video_jobs, "run_video_cascade", fake_cascade)
    monkeypatch.setattr(
        video_jobs, "_register_video_assets", lambda assets, model: (["d" * 32], [], [])
    )

    async def fake_journal(record: dict[str, Any]) -> None:
        journal.append(record)

    scheduler = _FakeScheduler(rows=[SimpleNamespace(id="video-check-j1")])
    runner = video_jobs.make_video_job_runner(
        mode="i2v",
        prompt="p",
        source_kind="image",
        source_url="data:image/png;base64,AAAA",
        sources=[{"kind": "image", "attachment": "a" * 32}],
        duration_s=8,
        model=None,
        config_dir=None,
        journal=fake_journal,
        wake_scheduler=scheduler,
        append_output=lambda job_id, text: outputs.append(text),
    )

    def collect(update: str | Mapping[str, Any]) -> None:
        if isinstance(update, Mapping):
            progress.append(dict(update))

    text = await runner("j1", AbortSignal(), collect)

    states = [record["state"] for record in journal]
    assert states == ["running", "completed"]
    running = journal[0]
    assert running["provider"] == "fal"
    assert running["provider_ref"]["request_id"] == "req-1"
    assert running["provider_ref"]["cancel_url"].endswith("/cancel")
    assert running["params"] == {"prompt": "p", "duration_s": 8}
    assert running["sources"] == [{"kind": "image", "attachment": "a" * 32}]
    completed = journal[1]
    assert completed["result"]["digest"] == "d" * 32
    assert completed["route_label"] == "FAL"
    assert (
        completed["provider_ref"]["request_id"] == "req-1"
    ), "the completed row must keep the refs the cleared handle no longer holds"

    # The wake row was cancelled on settle (exactly one update removing it).
    assert scheduler.updates == [[]]

    # Receipt on the job row + the §5.1 text.
    receipt = [entry for entry in progress if "video_receipt" in entry]
    assert receipt and receipt[0]["video_receipt"]["digest"] == "d" * 32
    assert text is not None, "a successful run returns its completion text"
    assert "Video ready — job j1" in text
    assert "model minimax/h3-max/image-to-video" in text
    assert "8.0s" in text
    assert "price not reported by provider" in text
    assert "digest dddd" in text
    assert outputs, "progress lines must land in the job's peek tail"


@pytest.mark.asyncio
async def test_signal_cancel_settles_cancelled_and_raises_carrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal: list[dict[str, Any]] = []
    cancelled: list[Any] = []
    cancels: list[Any] = []

    async def fake_cascade(**kwargs: Any) -> Any:
        handle = kwargs["handle"]
        handle.provider = "fal"
        handle.request_id = "req-2"
        from local_operator.video import VideoGenerationCancelled

        raise VideoGenerationCancelled("stop")

    async def fake_cancel(handle: Any) -> str:
        cancelled.append(handle.request_id)
        return "cancelled"

    monkeypatch.setattr(video_jobs, "run_video_cascade", fake_cascade)
    monkeypatch.setattr(video_jobs, "best_effort_cancel", fake_cancel)

    async def fake_journal(record: dict[str, Any]) -> None:
        journal.append(record)

    scheduler = _FakeScheduler(rows=[SimpleNamespace(id="video-check-j2")])
    runner = video_jobs.make_video_job_runner(
        mode="t2v",
        prompt="p",
        config_dir=None,
        journal=fake_journal,
        wake_scheduler=scheduler,
    )
    with pytest.raises(asyncio.CancelledError):
        await runner("j2", AbortSignal(), lambda d: cancels.append(d))

    assert cancelled == ["req-2"], "the bounded provider cancel must run"
    states = [record["state"] for record in journal]
    assert states == ["cancelled"]
    assert journal[0]["outcome_note"] == "Provider job cancelled."
    assert journal[0].get("error_type") is None, "a plain cancel is not a conflict"
    assert scheduler.updates == [[]]


@pytest.mark.asyncio
async def test_failure_settles_failed_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal: list[dict[str, Any]] = []

    from local_operator.video import VideoGenerationUnavailable

    async def fake_cascade(**kwargs: Any) -> Any:
        raise VideoGenerationUnavailable("no provider could run it")

    monkeypatch.setattr(video_jobs, "run_video_cascade", fake_cascade)

    async def fake_journal(record: dict[str, Any]) -> None:
        journal.append(record)

    scheduler = _FakeScheduler()
    runner = video_jobs.make_video_job_runner(
        mode="t2v",
        prompt="p",
        config_dir=None,
        journal=fake_journal,
        wake_scheduler=scheduler,
    )
    with pytest.raises(VideoGenerationUnavailable):
        await runner("j3", AbortSignal(), lambda d: None)
    assert [record["state"] for record in journal] == ["failed"]
    assert "no provider could run it" in journal[0]["outcome_note"]


def test_cancel_note_uses_the_frozen_wording_map() -> None:
    # Contract §1.6: the cancel-token phrases are NOT changed by video — the
    # note composes from the image lane's map, byte-for-byte (equality pinned
    # below), and never leaks a raw token.
    note = video_jobs._cancel_outcome_note("radient", "none")
    assert note == (
        "Provider job not cancelled — nothing was left to cancel. "
        "Provider cannot be stopped; the generation may complete upstream and "
        "bill; the result is not retrievable."
    )
    # A queued-only provider's ACKNOWLEDGED cancel still cannot stop a running
    # generation — the §4.3 clause rides the `cancelled` token too (round-1
    # design D1: FAL enforces aborts, Radient only queues).
    acked = video_jobs._cancel_outcome_note("radient", "cancelled")
    assert acked.startswith("Provider job cancelled.")
    assert "may complete upstream and bill" in acked
    # An abort-capable provider's acknowledged cancel is the plain phrase.
    assert video_jobs._cancel_outcome_note("fal", "cancelled") == "Provider job cancelled."
    # The conflict and gone branches carry the frozen phrases, never tokens.
    assert video_jobs._cancel_outcome_note("fal", "already_completed") == (
        "Provider job not cancelled — it had already completed; its result was discarded."
    )
    assert video_jobs._cancel_outcome_note("fal", "not_found") == (
        "Provider job not cancelled — the provider no longer knows the job."
    )
    assert video_jobs._cancel_outcome_note("fal", "timeout") == (
        "Provider job not cancelled — the cancel request timed out. "
        "The cancel request got no terminal answer; the generation may complete "
        "upstream and bill; the result is not retrievable."
    )
    # A token the map does not know is surfaced verbatim — never invented.
    assert video_jobs._cancel_outcome_note("fal", "weird") == "Provider job weird."


def test_cancel_copy_constants_match_the_image_lane() -> None:
    """The frozen map is byte-identical in both lanes (contract §1.6: the
    cancel-token phrases are NOT changed by video); the duplication is pinned
    here so the copies cannot drift."""
    from local_operator.tools import image_tool

    assert video_jobs._CANCEL_PHRASES == image_tool._CANCEL_PHRASES
    assert video_jobs._CANCEL_CONFLICT_SENTENCE == image_tool._CANCEL_CONFLICT_SENTENCE


@pytest.mark.asyncio
async def test_cancel_conflict_settles_the_frozen_already_finished_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancel that finds the generation already completed settles the frozen
    ``media_already_completed`` pair (round-1 design D1; contract §1.6/§4.1)."""
    journal: list[dict[str, Any]] = []

    async def fake_cascade(**kwargs: Any) -> Any:
        handle = kwargs["handle"]
        handle.provider = "fal"
        handle.request_id = "req-9"
        from local_operator.video import VideoGenerationCancelled

        raise VideoGenerationCancelled("stop")

    async def fake_cancel(handle: Any) -> str:
        return "already_completed"

    monkeypatch.setattr(video_jobs, "run_video_cascade", fake_cascade)
    monkeypatch.setattr(video_jobs, "best_effort_cancel", fake_cancel)

    async def fake_journal(record: dict[str, Any]) -> None:
        journal.append(record)

    runner = video_jobs.make_video_job_runner(
        mode="t2v",
        prompt="p",
        config_dir=None,
        journal=fake_journal,
        wake_scheduler=_FakeScheduler(rows=[SimpleNamespace(id="video-check-j9")]),
    )
    with pytest.raises(asyncio.CancelledError):
        await runner("j9", AbortSignal(), lambda d: None)

    record = journal[0]
    assert record["state"] == "cancelled"
    assert record["error_type"] == "media_already_completed"
    assert record["error"] == video_jobs._CANCEL_CONFLICT_SENTENCE
    assert record["outcome_note"] == (
        "Provider job not cancelled — it had already completed; its result was discarded."
    )


def _fake_wake(job_id: str) -> Any:
    return video_jobs.VideoWakeHandle(
        _FakeScheduler(rows=[SimpleNamespace(id=f"video-check-{job_id}")]), job_id
    )


@pytest.mark.asyncio
async def test_watcher_settles_provider_cancelled_as_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V5: a provider CANCELLED observation settles ``cancelled`` from the
    watcher exactly as the one-shot ``video_status`` path does — one
    observation, one word, wherever it was seen."""
    journal: list[dict[str, Any]] = []

    async def fake_resume(**_kwargs: Any) -> Any:
        from local_operator.clients._http import APIError

        raise APIError("provider says CANCELLED", status_code=200, code="cancelled")

    monkeypatch.setattr(video_jobs, "resume_video_observation", fake_resume)

    async def fake_journal(record: dict[str, Any]) -> None:
        journal.append(record)

    runner = video_jobs.make_video_watcher_runner(
        record={
            "job": "w1",
            "state": "running",
            "provider": "radient",
            "model": "m",
            "mode": "t2v",
            "provider_ref": {"request_id": "r1"},
        },
        config_dir=None,
        journal=fake_journal,
        wake=_fake_wake("w1"),
    )
    from local_operator.clients._http import APIError

    with pytest.raises(APIError):
        await runner("w1", AbortSignal(), lambda d: None)

    assert [record["state"] for record in journal] == ["cancelled"]
    assert journal[0]["outcome_note"] == "the provider reported the generation as CANCELLED"
    assert journal[0]["provider_ref"] == {"request_id": "r1"}


@pytest.mark.asyncio
async def test_watcher_settles_provider_gone_with_the_one_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D4: the provider-gone sentence has ONE home; the watcher's settle and
    every other 404 site render the identical bytes."""
    journal: list[dict[str, Any]] = []

    async def fake_resume(**_kwargs: Any) -> Any:
        from local_operator.clients._http import APIError

        raise APIError("gone", status_code=404, code="not_found")

    monkeypatch.setattr(video_jobs, "resume_video_observation", fake_resume)

    async def fake_journal(record: dict[str, Any]) -> None:
        journal.append(record)

    runner = video_jobs.make_video_watcher_runner(
        record={
            "job": "w2",
            "state": "running",
            "provider": "fal",
            "model": "m",
            "mode": "t2v",
        },
        config_dir=None,
        journal=fake_journal,
        wake=_fake_wake("w2"),
    )
    from local_operator.clients._http import APIError

    with pytest.raises(APIError):
        await runner("w2", AbortSignal(), lambda d: None)

    assert [record["state"] for record in journal] == ["failed"]
    assert journal[0]["outcome_note"] == video_jobs.PROVIDER_GONE_NOTE
    assert "(HTTP 404)" in journal[0]["outcome_note"]
