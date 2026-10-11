"""The video cascade: capability filter, skip pre-records, fail-forward walk.

Executor-level, with the rung dispatch faked at ``cascade._run_route`` (the
module-global read the executor deliberately makes at call time): what is
under test is the ORDER, the capability boundaries (Radient t2v-only until
its source mapping ships), the pre-recorded skips and the stop conditions —
not the wire shapes, which live against ``httpx.MockTransport`` in the
runner tests where they are faked at one level anyway.
"""

from __future__ import annotations

import pytest

from local_operator.artifacts import MediaAsset, RungAvailability
from local_operator.artifacts.rung import RungResult
from local_operator.video import (
    VideoGenerationCancelled,
    VideoGenerationUnavailable,
    VideoRoute,
)
from local_operator.video import cascade as video_cascade


def _resolution(*, radient: bool, fal: bool) -> video_cascade.VideoRouteResolution:
    rungs = (
        RungAvailability(
            route=str(VideoRoute.RADIENT),
            available=radient,
            reason="Signed in to Radient." if radient else "Not signed in to Radient.",
        ),
        RungAvailability(
            route=str(VideoRoute.FAL),
            available=fal,
            reason="A FAL key is stored." if fal else "No FAL key is stored.",
        ),
    )
    route = VideoRoute.RADIENT if radient else (VideoRoute.FAL if fal else VideoRoute.NONE)
    return video_cascade.VideoRouteResolution(route=route, reason="test", rungs=rungs)


def _asset(url: str = "https://example.com/v.mp4") -> MediaAsset:
    return MediaAsset(
        data=b"\x00\x00\x00\x18ftypmp42",
        content_type="video/mp4",
        source_url=url,
        duration_s=8.0,
    )


@pytest.mark.asyncio
async def test_capable_filter_and_message(monkeypatch: pytest.MonkeyPatch) -> None:
    both = _resolution(radient=True, fal=True)
    assert video_cascade.capable_video_routes(both, "t2v", None) == [
        VideoRoute.RADIENT,
        VideoRoute.FAL,
    ]
    # i2v needs a source kind: Radient declares none yet, so FAL only.
    assert video_cascade.capable_video_routes(both, "i2v", "image") == [VideoRoute.FAL]
    assert video_cascade.capable_video_routes(both, "v2v", "video") == [VideoRoute.FAL]
    only_radient = _resolution(radient=True, fal=False)
    assert video_cascade.capable_video_routes(only_radient, "i2v", "image") == []

    message = video_cascade.no_capable_message("i2v", "image")
    assert "FAL" in message and "sign in" in message.lower()
    # The Radient source refusal is the frozen sentence, one home.
    assert (
        video_cascade.RADIENT_VIDEO_SOURCE_SKIP_MESSAGE
        == video_cascade.VIDEO_RUNG_SPECS[VideoRoute.RADIENT].label
        + " cannot take image or video sources yet — its media route has no "
        "source mapping for video; use another provider"
    )


@pytest.mark.asyncio
async def test_no_capable_raises_named_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_resolution(*args, **kwargs):
        return _resolution(radient=True, fal=False)

    monkeypatch.setattr(video_cascade, "resolve_video_route", fake_resolution)

    with pytest.raises(VideoGenerationUnavailable) as excinfo:
        await video_cascade.run_video_cascade(prompt="p", mode="i2v", source_kind="image")
    assert "No available provider can generate a video from an image" in str(excinfo.value)
    assert excinfo.value.resolution is not None


@pytest.mark.asyncio
async def test_fail_over_pre_records_radients_source_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both rungs available, i2v: Radient is SKIPPED (never called); FAL runs."""
    calls: list[str] = []

    async def fake_resolution(*args, **kwargs):
        return _resolution(radient=True, fal=True)

    async def fake_run_route(route, **kwargs):
        calls.append(str(route))
        return RungResult(assets=[_asset()], model="minimax/h3-max/image-to-video")

    monkeypatch.setattr(video_cascade, "resolve_video_route", fake_resolution)
    monkeypatch.setattr(video_cascade, "_run_route", fake_run_route)

    outcome = await video_cascade.run_video_cascade(
        prompt="p",
        mode="i2v",
        source_kind="image",
        source_url="data:image/png;base64,AAAA",
    )
    assert calls == ["fal"], "the source request must never reach Radient in v1"
    assert str(outcome.route) == "fal"
    # Two recorded attempts: the pre-recorded skip in its cascade position,
    # then the rung that actually ran.
    assert [str(a.route) for a in outcome.attempts] == ["radient", "fal"]
    skipped = outcome.attempts[0]
    assert skipped.outcome == "skipped"
    assert skipped.reason_class == "unsupported"
    assert "cannot take image or video sources" in skipped.message
    assert outcome.attempts[1].outcome == "ok"


@pytest.mark.asyncio
async def test_t2v_runs_on_radient_first(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake_resolution(*args, **kwargs):
        return _resolution(radient=True, fal=True)

    async def fake_run_route(route, **kwargs):
        calls.append(str(route))
        return RungResult(assets=[_asset()], model="m")

    monkeypatch.setattr(video_cascade, "resolve_video_route", fake_resolution)
    monkeypatch.setattr(video_cascade, "_run_route", fake_run_route)

    outcome = await video_cascade.run_video_cascade(prompt="p", mode="t2v")
    assert calls == ["radient"]
    assert str(outcome.route) == "radient"


@pytest.mark.asyncio
async def test_cancellation_stops_without_failover(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake_resolution(*args, **kwargs):
        return _resolution(radient=True, fal=True)

    async def fake_run_route(route, **kwargs):
        calls.append(str(route))
        raise VideoGenerationCancelled("stop")

    monkeypatch.setattr(video_cascade, "resolve_video_route", fake_resolution)
    monkeypatch.setattr(video_cascade, "_run_route", fake_run_route)

    from local_operator.artifacts import JobCancelled

    with pytest.raises(JobCancelled):
        await video_cascade.run_video_cascade(prompt="p", mode="t2v")
    assert calls == ["radient"], "a stop is a stop — no rung after it may run"
