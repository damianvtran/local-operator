"""Video rung wire shapes against ``httpx.MockTransport``.

The result readers and the FAL queue happy path: what the harness sends and
how it folds responses back. The hub's video payload carries
``output.video{}`` (measured on prod, the hub RFC's C7) beside the neutral
``artifacts[]`` list; the FAL queue answers ``request_id`` + response-carried
URLs.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from local_operator.video import rungs as video_rungs


def test_video_asset_rows_reads_hub_output_video() -> None:
    rows = video_rungs._video_asset_rows(
        {
            "output": {
                "video": {
                    "url": "https://cdn.example/v.mp4",
                    "content_type": "video/mp4",
                    "duration": 5.2,
                }
            }
        }
    )
    assert rows == [
        {
            "url": "https://cdn.example/v.mp4",
            "content_type": "video/mp4",
            "duration": 5.2,
        }
    ]


def test_video_asset_rows_reads_neutral_artifacts() -> None:
    rows = video_rungs._video_asset_rows(
        {
            "artifacts": [
                {"kind": "video", "url": "https://cdn.example/v.mp4", "duration_s": 8.0},
                {"kind": "image", "url": "https://cdn.example/i.png"},
            ]
        }
    )
    assert len(rows) == 1
    assert rows[0]["url"] == "https://cdn.example/v.mp4"
    assert rows[0]["duration_s"] == 8.0


def test_video_asset_rows_tolerates_absent_shapes() -> None:
    assert video_rungs._video_asset_rows({}) == []
    assert video_rungs._video_asset_rows({"output": {"video": {}}}) == []


@pytest.mark.asyncio
async def test_poll_video_once_maps_fal_not_found_to_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_key(config_dir: Path | None) -> str:
        return "fk"

    monkeypatch.setattr(video_rungs, "_resolve_fal_key", fake_key)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "not found"})

    handle = video_rungs.CancelHandle()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await video_rungs.poll_video_once(
            record={
                "provider": "fal",
                "model": "fal-ai/wan/v2.7/edit-video",
                "provider_ref": {
                    "request_id": "req-404",
                    "status_url": "https://queue.fal.run/x/requests/req-404/status",
                    "result_url": "https://queue.fal.run/x/requests/req-404",
                },
            },
            handle=handle,
            client=client,
        )
    assert outcome.state == "gone"


@pytest.mark.asyncio
async def test_fal_submit_poll_download_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        path = request.url.path
        if request.method == "POST":
            body = json.loads(request.content.decode())
            assert body["sync_mode"] is False
            assert body["prompt"] == "p"
            assert body["duration"] == 5
            assert body["video_url"] == "https://example.com/src.mp4"
            return httpx.Response(
                200,
                json={
                    "request_id": "req-1",
                    "status_url": "https://queue.fal.run/app/requests/req-1/status",
                    "response_url": "https://queue.fal.run/app/requests/req-1",
                    "cancel_url": "https://queue.fal.run/app/requests/req-1/cancel",
                },
            )
        if path.endswith("/status"):
            return httpx.Response(200, json={"status": "COMPLETED"})
        if path.endswith("req-1"):
            return httpx.Response(
                200,
                json={
                    "video": {
                        "url": "https://cdn.fal.example/v.mp4",
                        "content_type": "video/mp4",
                        "duration": 5.0,
                    }
                },
            )
        if path.endswith("/v.mp4"):
            return httpx.Response(
                200,
                content=b"\x00\x00\x00\x18ftypmp42",
                headers={"content-type": "video/mp4"},
            )
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    accepted: list[video_rungs.CancelHandle] = []

    async def on_accept(handle: video_rungs.CancelHandle) -> None:
        accepted.append(handle)
        # The accept hook fires ONLY once the provider job exists.
        assert handle.request_id == "req-1"
        assert handle.cancel_url and handle.cancel_url.endswith("/cancel")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await video_rungs.run_fal_video(
            prompt="p",
            key="fk",
            mode="v2v",
            source_kind="video",
            source_url="https://example.com/src.mp4",
            duration_s=5,
            resolution=None,
            aspect_ratio=None,
            audio=None,
            seed=None,
            model=None,
            handle=video_rungs.CancelHandle(),
            emit=None,
            pause=None,
            on_accept=on_accept,
            client=client,
        )
    assert accepted, "the accept hook must fire at submit"
    assert result.model == "fal-ai/wan/v2.7/edit-video"
    assert result.generation_id == "req-1"
    assert len(result.assets) == 1
    assert result.assets[0].content_type == "video/mp4"
    assert result.assets[0].duration_s == 5.0
    assert result.assets[0].data.startswith(b"\x00\x00\x00\x18ftyp")
