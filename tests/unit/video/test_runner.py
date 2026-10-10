"""The detached runner: records at every transition, cancel, completion text.

The walk itself is faked at ``video.jobs.run_video_cascade`` (one level below
what the tool registers), so these tests pin the RUNNER's contract: the
durable record sequence, the accept hook's ``provider_ref``, the two cancel
spellings, the completion receipt/text, and the wake's lifecycle.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.artifacts import MediaAsset
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
        return _outcome()

    monkeypatch.setattr(video_jobs, "run_video_cascade", fake_cascade)
    monkeypatch.setattr(
        video_jobs, "_register_video_assets", lambda assets, model: (["d" * 32], [])
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
    text = await runner("j1", SimpleNamespace(), lambda d: progress.append(dict(d)))

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

    # The wake row was cancelled on settle (exactly one update removing it).
    assert scheduler.updates == [[]]

    # Receipt on the job row + the §5.1 text.
    receipt = [entry for entry in progress if "video_receipt" in entry]
    assert receipt and receipt[0]["video_receipt"]["digest"] == "d" * 32
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
        return "ok"

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
        await runner("j2", SimpleNamespace(), lambda d: cancels.append(d))

    assert cancelled == ["req-2"], "the bounded provider cancel must run"
    states = [record["state"] for record in journal]
    assert states == ["cancelled"]
    assert "cancelled — provider cancel: ok" in journal[0]["outcome_note"]
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
        await runner("j3", SimpleNamespace(), lambda d: None)
    assert [record["state"] for record in journal] == ["failed"]
    assert "no provider could run it" in journal[0]["outcome_note"]


@pytest.mark.asyncio
async def test_cancel_note_honesty_clause_for_queued_only() -> None:
    note = video_jobs._cancel_outcome_note("radient", "none")
    assert "cannot be stopped" in note
    assert "may complete upstream and bill" in note
    assert "result is not retrievable" in note
    # The frozen sentence, verbatim second half (contract §4.3).
    assert note == (
        "cancelled — provider cancel: none. Provider cannot be stopped; the "
        "generation may complete upstream and bill; the result is not retrievable."
    )
    plain = video_jobs._cancel_outcome_note("fal", "ok")
    assert plain == "cancelled — provider cancel: ok."
