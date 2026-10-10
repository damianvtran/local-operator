"""The OpenRouter rung's wire, against ``httpx.MockTransport`` — no network.

What is pinned: the Images API call (endpoint, bearer, body with the
documented aspect-ratio mapping, ``n`` only when > 1, ``seed`` only when
pinned), the ``data[].b64_json`` + ``media_type`` parse, the REPORTED
``usage.cost`` (the docs' own settlement shape), the recorded img2img skip,
and the failure classes (502 = the docs' all-or-nothing failure). NO live
probe ran this wave (operator decision) — tiered citations are in
``docs/design/image-providers.md``.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Callable

import httpx
import pytest

from local_operator.artifacts.rung import RungSkipped
from local_operator.clients._http import APIError
from local_operator.imagegen import rungs_openrouter as rung_mod
from local_operator.imagegen.errors import failure_reason_class

PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c626001000000ffff03000006000557bfabd4"
    "0000000049454e44ae426082"
)
PNG_B64 = base64.b64encode(PNG_1X1).decode("ascii")


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class _Recorder:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.bodies: list[Any] = []

    def handler(self, response: httpx.Response) -> Callable[[httpx.Request], httpx.Response]:
        def _handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            try:
                self.bodies.append(json.loads(request.content) if request.content else None)
            except ValueError:
                self.bodies.append(None)
            return response

        return _handle


def _ok_response(
    *,
    cost: float | None = 0.04,
    items: list[dict[str, Any]] | None = None,
) -> httpx.Response:
    payload: dict[str, Any] = {
        "created": 1748372400,
        "data": items if items is not None else [{"b64_json": PNG_B64, "media_type": "image/png"}],
    }
    if cost is not None:
        payload["usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": 4175,
            "total_tokens": 4175,
            "cost": cost,
        }
    return httpx.Response(200, json=payload)


async def _run(recorder: _Recorder, **kwargs: Any):
    defaults: dict[str, Any] = dict(
        prompt="a cat",
        key="ork-1",
        num_images=1,
        image_size="square_hd",
        source_url=None,
        seed=None,
        model=None,
        emit=None,
        pause=None,
    )
    defaults.update(kwargs)
    return await rung_mod.run_openrouter(**defaults)


@pytest.mark.asyncio
async def test_the_call_shape_and_the_reported_cost() -> None:
    recorder = _Recorder()
    http = _client(recorder.handler(_ok_response()))

    result = await _run(recorder, client=http)

    assert result.model == rung_mod.OPENROUTER_DEFAULT_IMAGE_MODEL
    assert result.assets[0].data == PNG_1X1
    assert result.assets[0].content_type == "image/png"
    # usage.cost is reported per request (the docs' settlement shape), so it
    # rides cost_usd under design D8.
    assert result.cost_usd == pytest.approx(0.04)
    assert result.cost_source == "reported"
    assert result.billing_basis == "billed"
    assert result.cost_provenance == "OpenRouter /images response usage.cost"

    request = recorder.requests[0]
    assert request.method == "POST"
    assert str(request.url) == rung_mod.OPENROUTER_IMAGES_URL
    assert request.headers["authorization"] == "Bearer ork-1"
    body = recorder.bodies[0]
    assert body["model"] == rung_mod.OPENROUTER_DEFAULT_IMAGE_MODEL
    assert body["prompt"] == "a cat"
    assert body["aspect_ratio"] == "1:1"
    # n=1 stays unset (the provider default); the single-image providers'
    # documented n>1 rejection is never tripped without a real request for more.
    assert "n" not in body
    assert "seed" not in body


@pytest.mark.asyncio
async def test_n_and_seed_are_sent_only_when_asked_for() -> None:
    recorder = _Recorder()
    http = _client(
        recorder.handler(
            _ok_response(
                items=[
                    {"b64_json": PNG_B64, "media_type": "image/png"},
                    {"b64_json": PNG_B64, "media_type": "image/webp"},
                ]
            )
        )
    )

    result = await _run(recorder, num_images=2, seed=7, image_size="landscape_16_9", client=http)

    body = recorder.bodies[0]
    assert body["n"] == 2
    assert body["seed"] == 7
    assert body["aspect_ratio"] == "16:9"
    assert len(result.assets) == 2
    assert result.assets[1].content_type == "image/webp"


@pytest.mark.asyncio
async def test_a_missing_usage_leaves_the_cost_unset() -> None:
    recorder = _Recorder()
    http = _client(recorder.handler(_ok_response(cost=None)))

    result = await _run(recorder, client=http)

    assert result.cost_usd is None
    assert result.cost_source is None
    assert result.billing_basis is None
    assert result.cost_provenance is None


@pytest.mark.asyncio
async def test_img2img_is_a_recorded_skip() -> None:
    http = _client(lambda request: httpx.Response(500))

    with pytest.raises(RungSkipped) as caught:
        await _run(_Recorder(), source_url="data:image/png;base64,AAAA", client=http)

    assert caught.value.reason_class == "unsupported"


@pytest.mark.asyncio
async def test_a_502_failure_maps_to_upstream_and_scrubs_the_bearer() -> None:
    recorder = _Recorder()
    response = httpx.Response(502, json={"error": {"message": "generation failed"}})
    http = _client(recorder.handler(response))

    with pytest.raises(APIError) as caught:
        await _run(recorder, client=http)

    assert caught.value.status_code == 502
    # The docs' all-or-nothing failure is an upstream failure to the walk's
    # classifier (5xx family), which is what a fail-forward decision reads.
    assert failure_reason_class(caught.value) == "upstream"
    assert "ork-1" not in (caught.value.body or "")


@pytest.mark.asyncio
async def test_a_response_without_image_data_is_invalid_response() -> None:
    recorder = _Recorder()
    http = _client(recorder.handler(httpx.Response(200, json={"data": []})))

    with pytest.raises(APIError) as caught:
        await _run(recorder, client=http)

    assert caught.value.code == "invalid_response"


@pytest.mark.asyncio
async def test_a_pre_aborted_signal_stops_before_the_request() -> None:
    # Reviewer round 1 Q6: a single-request rung has no poll loop, so the
    # zero-length pause is the abort check that must run before the spend.
    from local_operator.imagegen import cascade as image_cascade

    calls: list[float] = []

    async def pause(seconds: float) -> None:
        calls.append(seconds)
        raise image_cascade.ImageGenerationCancelled()

    recorder = _Recorder()
    http = _client(recorder.handler(_ok_response()))
    with pytest.raises(image_cascade.ImageGenerationCancelled):
        await _run(recorder, pause=pause, client=http)

    assert calls == [0.0]
    assert recorder.requests == []


def test_the_spec_declares_reported_cost_and_no_cancel() -> None:
    from local_operator.artifacts.rung import CancelSupport
    from local_operator.imagegen import ImageRoute, cascade

    spec = cascade.RUNG_SPECS[ImageRoute.OPENROUTER]
    assert spec.label == "OpenRouter"
    assert spec.cancel_support == CancelSupport.NONE
    assert spec.cost == "reported"
    assert spec.capabilities == frozenset({"t2i"})
