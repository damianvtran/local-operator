"""Rung wire behaviour, against ``httpx.MockTransport`` — no network.

What is pinned here is the contract the design froze: Radient's status/result/
cancel reads take ``request_id`` ONLY and its failures switch on ``error_type``
never prose; FAL drives its queue with the response-carried URLs (and derives
them from the app path when a response omits them); OpenAI decodes both
``b64_json`` and ``url`` items and has no provider-side cancel; and every
download is bounded.
"""

from __future__ import annotations

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from typing import Any, Callable

import httpx
import pytest
from pydantic import SecretStr

from local_operator.clients._http import APIError
from local_operator.imagegen import ImageRoute
from local_operator.imagegen import rungs as image_rungs
from local_operator.imagegen.errors import REASON_CLASSES

#: A 73-byte 1x1 PNG (real bytes, so nothing has to decode anything).
PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c626001000000ffff03000006000557bfabd4"
    "0000000049454e44ae426082"
)


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _no_pause(seconds: float) -> None:  # pragma: no cover - trivial
    return


class _Recorder:
    """Handlers append ``(request, body)`` here for post-hoc assertions."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.bodies: list[Any] = []

    def record(self, request: httpx.Request) -> Any:
        self.requests.append(request)
        try:
            body = json.loads(request.content) if request.content else None
        except ValueError:
            body = None
        self.bodies.append(body)
        return body

    def paths(self) -> list[str]:
        return [request.url.path for request in self.requests]


# ---------------------------------------------------------------------------
# Radient
# ---------------------------------------------------------------------------


def _radient_handler(
    recorder: _Recorder,
    *,
    statuses: list[dict[str, Any]] | None = None,
    capacity: dict[str, Any] | None = None,
    capacity_status: int = 200,
) -> Callable[[httpx.Request], httpx.Response]:
    status_script = list(statuses or [{"status": "IN_QUEUE"}, {"status": "COMPLETED"}])

    def handler(request: httpx.Request) -> httpx.Response:
        recorder.record(request)
        path = request.url.path
        if path.endswith("/tools/media/models"):
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"id": "other-model", "type": "video"},
                        {
                            "id": "flux/dev",
                            "type": "image",
                            "default": True,
                            "unit_price_usd": 0.05,
                        },
                    ]
                },
            )
        if path.endswith("/me/billing-sources/capacity"):
            if capacity_status != 200:
                return httpx.Response(capacity_status, json={})
            return httpx.Response(200, json=capacity or {"total_balance": 10.0})
        if path.endswith("/tools/media/generate"):
            return httpx.Response(
                200,
                json={"request_id": "r1", "status": "IN_QUEUE", "cost_usd": 0.08},
            )
        if path.endswith("/tools/media/status"):
            payload = status_script.pop(0) if status_script else {"status": "COMPLETED"}
            return httpx.Response(200, json=payload)
        if path.endswith("/tools/media/result"):
            return httpx.Response(
                200,
                json={
                    "images": [
                        {
                            "url": "https://img.test/one.png",
                            "width": 1024,
                            "height": 1024,
                            "content_type": "image/png",
                        }
                    ]
                },
            )
        if request.url.host == "img.test":
            return httpx.Response(200, content=PNG_1X1, headers={"content-type": "image/png"})
        return httpx.Response(404, json={"error": "unexpected"})

    return handler


@pytest.mark.asyncio
async def test_radient_settled_status_figure_replaces_the_quote_and_is_billed() -> None:
    # agent-server: submit answers the QUOTE (0.08 in the fake), the terminal
    # status carries the SETTLED figure (``settled: true``) that equals the
    # ledger - that one is a charge, so it wins and is labelled ``billed``.
    recorder = _Recorder()
    async with _client(
        _radient_handler(
            recorder,
            statuses=[{"status": "COMPLETED", "settled": True, "cost_usd": 0.05, "units": 1}],
        )
    ) as client:
        result = await _run_radient(client)

    assert result.cost_usd == 0.05
    assert result.cost_source == "reported"
    assert result.billing_basis == "billed"
    assert result.cost_provenance is not None and "settled" in result.cost_provenance


@pytest.mark.asyncio
async def test_radient_settled_zero_is_a_billed_zero_not_the_quote() -> None:
    recorder = _Recorder()
    async with _client(
        _radient_handler(
            recorder, statuses=[{"status": "COMPLETED", "settled": True, "cost_usd": 0.0}]
        )
    ) as client:
        result = await _run_radient(client)

    assert result.cost_usd == 0.0
    assert result.billing_basis == "billed"


@pytest.mark.asyncio
async def test_radient_unsettled_status_cost_keeps_the_quote_as_estimated() -> None:
    # A ``cost_usd`` on a status payload WITHOUT ``settled: true`` is not a
    # charge: the quote and the ``estimated`` label stand.
    recorder = _Recorder()
    async with _client(
        _radient_handler(
            recorder, statuses=[{"status": "COMPLETED", "settled": False, "cost_usd": 9.99}]
        )
    ) as client:
        result = await _run_radient(client)

    assert result.cost_usd == 0.08
    assert result.billing_basis == "estimated"


async def _run_radient(client: httpx.AsyncClient):
    return await image_rungs.run_radient(
        prompt="a cat",
        base_url="https://hub.test",
        credential="cred",
        num_images=1,
        image_size="square_hd",
        seed=None,
        strength=None,
        source_url=None,
        model=None,
        handle=image_rungs.CancelHandle(),
        emit=None,
        pause=_no_pause,
        client=client,
    )


@pytest.mark.asyncio
async def test_radient_happy_path_request_id_only_and_passthrough() -> None:
    recorder = _Recorder()
    handle = image_rungs.CancelHandle()
    progress: list[tuple[str, dict[str, Any]]] = []
    async with _client(
        _radient_handler(
            recorder,
            statuses=[
                {"status": "IN_QUEUE", "queue_position": 2},
                {
                    "status": "IN_PROGRESS",
                    "logs": [{"message": "step 1 of 4", "timestamp": 1700000000}],
                },
                {"status": "COMPLETED"},
            ],
        )
    ) as client:
        result = await image_rungs.run_radient(
            prompt="a cat",
            base_url="https://hub.test",
            credential="cred",
            num_images=2,
            image_size="square_hd",
            seed=42,
            strength=None,
            source_url=None,
            model=None,
            handle=handle,
            emit=lambda text, details: progress.append((text, details)),
            pause=_no_pause,
            client=client,
        )

    assert result.model == "flux/dev", "the default is read off the live list"
    assert result.generation_id == "r1"
    assert result.cost_usd == 0.08
    # The status never said ``settled``: the figure is still the submit-time
    # QUOTE, so it is labelled ``estimated`` (the amount/source are unchanged).
    assert result.cost_source == "reported"
    assert result.billing_basis == "estimated"
    assert result.cost_provenance is not None and "quote" in result.cost_provenance
    assert len(result.assets) == 1
    asset = result.assets[0]
    assert asset.data == PNG_1X1
    assert asset.content_type == "image/png"
    assert (asset.width, asset.height) == (1024, 1024)

    # request_id ONLY on status/result — no model param (freeze note).
    for request in recorder.requests:
        if request.url.path.endswith(("/tools/media/status", "/tools/media/result")):
            assert list(request.url.params.keys()) == ["request_id"]

    # The bearer rides EVERY hub call — models, capacity, generate, status and
    # result. The 2026-10-08 defect was exactly a call missing it, so this is
    # pinned per-path rather than on the generate call alone.
    hub_requests = [r for r in recorder.requests if r.url.host == "hub.test"]
    assert {r.url.path for r in hub_requests} == {
        "/tools/media/models",
        "/me/billing-sources/capacity",
        "/tools/media/generate",
        "/tools/media/status",
        "/tools/media/result",
    }
    for request in hub_requests:
        assert request.headers["authorization"] == "Bearer cred", request.url.path
    # ...and NOTHING else does: asset downloads go to the provider's own origin
    # and must never receive the account bearer.
    downloads = [r for r in recorder.requests if r.url.host == "img.test"]
    assert downloads, "sanity: the asset download happened through the same client"
    for request in downloads:
        assert "authorization" not in request.headers

    generate_body = next(
        body
        for request, body in zip(recorder.requests, recorder.bodies)
        if request.url.path.endswith("/tools/media/generate")
    )
    assert generate_body == {
        "model": "flux/dev",
        "prompt": "a cat",
        "num_images": 2,
        "image_size": "square_hd",
        "seed": 42,
    }, "flat passthrough, nothing invented"

    stages = [details["stage"] for _, details in progress]
    assert "queued" in stages and "in_progress" in stages
    assert (
        "downloading" not in stages
    ), "the download phase folds into in_progress (Q7 canonical vocabulary)"
    # The canonical field set (Q7 wire scope) rides EVERY update; a value no
    # provider supplied is an honest None, never a synthesized stand-in.
    canonical = {"stage", "queue_position", "progress_fraction", "log_lines", "error", "error_type"}
    assert all(canonical <= set(details) for _, details in progress)
    assert all(
        details["progress_fraction"] is None for _, details in progress
    ), "no provider reports a fraction; None, never synthesized"
    assert all(
        details["error"] is None and details["error_type"] is None for _, details in progress
    )
    queued_update = next(details for _, details in progress if details["stage"] == "queued")
    assert queued_update["queue_position"] == 2
    assert queued_update["log_lines"] is None, "no logs on that payload -> null, not []"
    running_update = next(details for _, details in progress if details["stage"] == "in_progress")
    assert running_update["log_lines"] == [
        {"message": "step 1 of 4", "timestamp": 1700000000}
    ], "the provider's logs list passes through verbatim"
    assert all(details["provider"] == "radient" for _, details in progress)
    # Terminal: nothing left for a cancel to do.
    assert handle.provider is None


def test_emit_progress_swallows_a_raising_emitter() -> None:
    """The rungs' guard (reviewer round-1 pin): progress never rides control flow.

    A raising emitter must never escape into its caller — the poll loop, the
    cascade's failure path, or a cancellation handler. ``None`` stays a no-op.
    """

    def raiser(text: str, details: dict[str, Any]) -> None:
        raise RuntimeError("emitter exploded")

    image_rungs.emit_progress(raiser, "line", stage="queued")  # must not raise
    image_rungs.emit_progress(None, "line", stage="queued")  # no-op


@pytest.mark.asyncio
async def test_radient_affordability_skips_the_rung() -> None:
    recorder = _Recorder()
    async with _client(_radient_handler(recorder, capacity={"total_balance": 0.01})) as client:
        with pytest.raises(image_rungs.RungSkipped) as caught:
            await image_rungs.run_radient(
                prompt="a cat",
                base_url="https://hub.test",
                credential="cred",
                num_images=1,
                image_size="square_hd",
                seed=None,
                strength=None,
                source_url=None,
                model=None,
                handle=image_rungs.CancelHandle(),
                emit=None,
                pause=_no_pause,
                client=client,
            )
    assert caught.value.reason_class == "insufficient_balance"
    assert not any(path.endswith("/tools/media/generate") for path in recorder.paths())


@pytest.mark.asyncio
async def test_radient_capacity_probe_failure_proceeds_optimistically() -> None:
    """A probe that cannot answer must not strand a working rung."""
    recorder = _Recorder()
    async with _client(_radient_handler(recorder, capacity_status=404)) as client:
        result = await image_rungs.run_radient(
            prompt="a cat",
            base_url="https://hub.test",
            credential="cred",
            num_images=1,
            image_size="square_hd",
            seed=None,
            strength=None,
            source_url=None,
            model=None,
            handle=image_rungs.CancelHandle(),
            emit=None,
            pause=_no_pause,
            client=client,
        )
    assert result.generation_id == "r1"
    assert any(path.endswith("/tools/media/generate") for path in recorder.paths())


@pytest.mark.asyncio
async def test_radient_affordability_reads_the_result_envelope() -> None:
    """The LIVE capacity shape nests ``total_balance`` under ``result``.

    Measured against production 2026-10-08: the hub answers
    ``{"msg": ..., "result": {"total_balance": ...}}``. A low balance inside
    the envelope must skip the rung exactly as a top-level one does — a
    silently unread balance would turn the affordability gate into a no-op.
    """
    recorder = _Recorder()
    envelope = {"msg": "Billing capacity retrieved", "result": {"total_balance": 0.01}}
    async with _client(_radient_handler(recorder, capacity=envelope)) as client:
        with pytest.raises(image_rungs.RungSkipped) as caught:
            await image_rungs.run_radient(
                prompt="a cat",
                base_url="https://hub.test",
                credential="cred",
                num_images=1,
                image_size="square_hd",
                seed=None,
                strength=None,
                source_url=None,
                model=None,
                handle=image_rungs.CancelHandle(),
                emit=None,
                pause=_no_pause,
                client=client,
            )
    assert caught.value.reason_class == "insufficient_balance"
    assert not any(path.endswith("/tools/media/generate") for path in recorder.paths())


def test_the_no_signal_backoff_is_bounded() -> None:
    """pause=None callers get 2 s -> 4 s -> 8 s(cap) — never an unbounded wait."""
    assert image_rungs._no_signal_poll_interval(0.0) == image_rungs.IMAGE_POLL_INTERVAL_S
    assert image_rungs._no_signal_poll_interval(30.0) == 4.0
    assert image_rungs._no_signal_poll_interval(60.0) == image_rungs.IMAGE_POLL_NO_SIGNAL_CAP_S
    assert image_rungs._no_signal_poll_interval(9_999.0) == image_rungs.IMAGE_POLL_NO_SIGNAL_CAP_S


@pytest.mark.asyncio
async def test_a_pauseless_poller_still_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    """The round-1 pace pin: with ``pause=None`` (the no-signal caller) the poll
    loop consults the bounded pacer instead of racing back-to-back reads."""
    waited: list[float] = []

    def pacer(elapsed_s: float) -> float:
        waited.append(elapsed_s)
        return 0.0  # keep the test instant; the interval values are pinned just above

    monkeypatch.setattr(image_rungs, "_no_signal_poll_interval", pacer)
    recorder = _Recorder()
    async with _client(
        _radient_handler(recorder, statuses=[{"status": "IN_QUEUE"}, {"status": "COMPLETED"}])
    ) as client:
        result = await image_rungs.run_radient(
            prompt="a cat",
            base_url="https://hub.test",
            credential="cred",
            num_images=1,
            image_size="square_hd",
            seed=None,
            strength=None,
            source_url=None,
            model=None,
            handle=image_rungs.CancelHandle(),
            emit=None,
            pause=None,
            client=client,
        )
    assert result.generation_id == "r1"
    assert waited, "the loop paced the poll instead of hot-polling"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_type", "expected_code"),
    [
        ("media_rejected", "media_rejected"),
        ("media_failed", "media_failed"),
        ("media_rate_limited", "media_rate_limited"),
        ("media_unavailable", "media_unavailable"),
        ("something_else", "upstream"),
    ],
)
async def test_radient_failed_switches_on_error_type_not_prose(
    error_type: str, expected_code: str
) -> None:
    recorder = _Recorder()
    failure = {"status": "FAILED", "error_type": error_type, "error": "the platform sentence"}
    async with _client(_radient_handler(recorder, statuses=[failure])) as client:
        with pytest.raises(APIError) as caught:
            await image_rungs.run_radient(
                prompt="a cat",
                base_url="https://hub.test",
                credential="cred",
                num_images=1,
                image_size="square_hd",
                seed=None,
                strength=None,
                source_url=None,
                model=None,
                handle=image_rungs.CancelHandle(),
                emit=None,
                pause=_no_pause,
                client=client,
            )
    assert caught.value.code == expected_code
    assert str(caught.value) == "the platform sentence"


# ---------------------------------------------------------------------------
# FAL
# ---------------------------------------------------------------------------


def _fal_handler(
    recorder: _Recorder,
    *,
    submit: dict[str, Any],
    derived: bool = False,
    statuses: list[dict[str, Any]] | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    status_script = list(statuses or [{"status": "COMPLETED"}])

    def handler(request: httpx.Request) -> httpx.Response:
        recorder.record(request)
        path = request.url.path
        if request.method == "POST" and path.startswith("/fal-ai/"):
            return httpx.Response(200, json=submit)
        if path.endswith("/status"):
            payload = status_script.pop(0) if status_script else {"status": "COMPLETED"}
            return httpx.Response(200, json=payload)
        if path.endswith(("/custom/req1", "/requests/r1", "/requests/req1")):
            return httpx.Response(
                200,
                json={
                    "images": [{"url": "https://falimg.test/a.png", "width": 1024, "height": 1024}]
                },
            )
        if request.url.host == "falimg.test":
            return httpx.Response(200, content=PNG_1X1, headers={"content-type": "image/png"})
        return httpx.Response(404, json={"error": "unexpected", "path": path})

    return handler


@pytest.mark.asyncio
async def test_fal_uses_response_carried_urls_and_key_header() -> None:
    recorder = _Recorder()
    handle = image_rungs.CancelHandle()
    submit = {
        "request_id": "req1",
        "status": "IN_QUEUE",
        "status_url": "https://queue.fal.test/custom/req1/status",
        "response_url": "https://queue.fal.test/custom/req1",
        "cancel_url": "https://queue.fal.test/custom/req1/cancel",
    }
    progress: list[tuple[str, dict[str, Any]]] = []
    async with _client(
        _fal_handler(
            recorder,
            submit=submit,
            statuses=[{"status": "IN_QUEUE", "queue_position": 1}, {"status": "COMPLETED"}],
        )
    ) as client:
        result = await image_rungs.run_fal(
            prompt="a fox",
            key="fk",
            num_images=1,
            image_size="landscape_4_3",
            seed=7,
            strength=None,
            source_url=None,
            model="fal-ai/flux/dev",
            handle=handle,
            emit=lambda text, details: progress.append((text, details)),
            pause=_no_pause,
            base_url="https://queue.fal.test",
            client=client,
        )
    assert result.model == "fal-ai/flux/dev"
    assert result.generation_id == "req1"
    assert (
        recorder.paths().count("/custom/req1/status") == 2
    ), "the carried status URL is used — both polls (queued, then completed)"
    assert "/custom/req1" in recorder.paths()
    submit_body = recorder.bodies[0]
    assert submit_body["sync_mode"] is False
    assert submit_body["image_size"] == "landscape_4_3"
    assert submit_body["seed"] == 7
    assert recorder.requests[0].headers["authorization"] == "Key fk"
    # The FAL branches emit the canonical set (reviewer round-1 pin): the poll
    # update carries queue_position, the download folds into in_progress —
    # both present every canonical key with honest nulls.
    stages = [details["stage"] for _, details in progress]
    assert stages == ["queued", "in_progress"]
    canonical = {"stage", "queue_position", "progress_fraction", "log_lines", "error", "error_type"}
    assert all(canonical <= set(details) for _, details in progress)
    assert progress[0][1]["queue_position"] == 1
    assert all(details["provider"] == "fal" for _, details in progress)
    assert all(
        details["error"] is None and details["error_type"] is None for _, details in progress
    )


@pytest.mark.asyncio
async def test_fal_derives_urls_from_the_app_path_when_the_response_omits_them() -> None:
    """The fallback reproduces the old client's derivation — from the app used.

    The legacy client hardcoded its app root while posting to ``model_path``;
    the derivation here reads the model actually posted, so a non-default app
    is polled at ITS OWN root (the latent bug the design records).
    """
    recorder = _Recorder()
    submit = {"request_id": "r1", "status": "IN_QUEUE"}
    async with _client(_fal_handler(recorder, submit=submit)) as client:
        await image_rungs.run_fal(
            prompt="a fox",
            key="fk",
            num_images=1,
            image_size="square_hd",
            seed=None,
            strength=None,
            source_url=None,
            model="fal-ai/flux/schnell",
            handle=image_rungs.CancelHandle(),
            emit=None,
            pause=_no_pause,
            base_url="https://queue.fal.test",
            client=client,
        )
    assert "/fal-ai/flux/requests/r1/status" in recorder.paths()
    assert "/fal-ai/flux/requests/r1" in recorder.paths()


@pytest.mark.asyncio
async def test_fal_img2img_rides_the_image_to_image_route_with_image_url() -> None:
    recorder = _Recorder()
    submit = {"request_id": "req1", "status": "IN_QUEUE"}
    async with _client(_fal_handler(recorder, submit=submit)) as client:
        await image_rungs.run_fal(
            prompt="make it rain",
            key="fk",
            num_images=1,
            image_size="square_hd",
            seed=None,
            strength=0.6,
            source_url="data:image/png;base64,AAAA",
            model="fal-ai/flux/dev",
            handle=image_rungs.CancelHandle(),
            emit=None,
            pause=_no_pause,
            base_url="https://queue.fal.test",
            client=client,
        )
    assert recorder.paths()[0].endswith("/fal-ai/flux/dev/image-to-image")
    body = recorder.bodies[0]
    assert body["image_url"] == "data:image/png;base64,AAAA"
    assert body["strength"] == 0.6
    assert "image_size" not in body, "the img2img route takes no image_size"


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def _openai_handler(
    recorder: _Recorder, response_body: dict[str, Any]
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        recorder.record(request)
        if request.url.path.endswith("/images/generations"):
            return httpx.Response(200, json=response_body)
        if request.url.host == "oai.test":
            return httpx.Response(200, content=PNG_1X1, headers={"content-type": "image/png"})
        return httpx.Response(404, json={"error": "unexpected"})

    return handler


def test_openai_size_mapping() -> None:
    assert image_rungs.openai_size("square_hd", "gpt-image-1") == "1024x1024"
    assert image_rungs.openai_size("portrait_4_3", "gpt-image-1") == "1024x1536"
    assert image_rungs.openai_size("landscape_16_9", "gpt-image-1") == "1536x1024"
    assert image_rungs.openai_size("portrait_4_3", "dall-e-3") == "1024x1792"
    assert image_rungs.openai_size("landscape_4_3", "dall-e-3") == "1792x1024"
    assert image_rungs.openai_size("square", "dall-e-3") == "1024x1024"


@pytest.mark.asyncio
async def test_openai_decodes_b64_items() -> None:
    recorder = _Recorder()
    body = {"data": [{"b64_json": base64.b64encode(PNG_1X1).decode("ascii")}]}
    async with _client(_openai_handler(recorder, body)) as client:
        result = await image_rungs.run_openai(
            prompt="a cat",
            key="sk",
            num_images=1,
            image_size="square_hd",
            source_url=None,
            model=None,
            emit=None,
            pause=_no_pause,
            base_url="https://oai.test/v1",
            client=client,
        )
    assert result.model == "gpt-image-1"
    assert result.assets[0].data == PNG_1X1
    assert result.assets[0].content_type == "image/png"
    sent = recorder.bodies[0]
    assert sent == {"model": "gpt-image-1", "prompt": "a cat", "n": 1, "size": "1024x1024"}


@pytest.mark.asyncio
async def test_openai_downloads_url_items() -> None:
    recorder = _Recorder()
    progress: list[tuple[str, dict[str, Any]]] = []
    body = {"data": [{"url": "https://oai.test/a.png"}]}
    async with _client(_openai_handler(recorder, body)) as client:
        result = await image_rungs.run_openai(
            prompt="a cat",
            key="sk",
            num_images=1,
            image_size="square_hd",
            source_url=None,
            model="dall-e-3",
            emit=lambda text, details: progress.append((text, details)),
            pause=_no_pause,
            base_url="https://oai.test/v1",
            client=client,
        )
    assert result.assets[0].source_url == "https://oai.test/a.png"
    assert result.assets[0].data == PNG_1X1
    # The OpenAI url branch emits the canonical set too (reviewer round-1 pin):
    # one download-phase update, stage folded to in_progress.
    assert len(progress) == 1
    assert progress[0][1]["stage"] == "in_progress"
    assert progress[0][1]["provider"] == "openai"
    canonical = {"stage", "queue_position", "progress_fraction", "log_lines", "error", "error_type"}
    assert canonical <= set(progress[0][1])


@pytest.mark.asyncio
async def test_openai_img2img_is_skipped_with_a_reason() -> None:
    recorder = _Recorder()
    async with _client(_openai_handler(recorder, {"data": []})) as client:
        with pytest.raises(image_rungs.RungSkipped) as caught:
            await image_rungs.run_openai(
                prompt="edit",
                key="sk",
                num_images=1,
                image_size="square_hd",
                source_url="data:image/png;base64,AAAA",
                model=None,
                emit=None,
                pause=_no_pause,
                base_url="https://oai.test/v1",
                client=client,
            )
    assert caught.value.reason_class == "unsupported"
    # And the token is INSIDE the closed vocabulary — a consumer switching on
    # ``REASON_CLASSES`` must never meet an out-of-set class (round-1 finding).
    assert caught.value.reason_class in REASON_CLASSES
    assert recorder.requests == [], "nothing is spent on a skipped rung"


# ---------------------------------------------------------------------------
# best_effort_cancel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_reports_none_without_a_handle() -> None:
    assert await image_rungs.best_effort_cancel(None) == "none"
    assert await image_rungs.best_effort_cancel(image_rungs.CancelHandle()) == "none"


@pytest.mark.asyncio
async def test_radient_cancel_sends_request_id_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        recorder.record(request)
        return httpx.Response(200, json={"status": "CANCELLED"})

    monkeypatch.setattr(image_rungs, "_client_scope", _scope_for(handler))
    handle = image_rungs.CancelHandle(
        provider=ImageRoute.RADIENT,
        request_id="r1",
        model="flux/dev",
        base_url="https://hub.test",
        credential=SecretStr("cred"),
    )
    assert await image_rungs.best_effort_cancel(handle) == "cancelled"
    request = recorder.requests[0]
    assert request.method == "POST"
    assert request.url.path == "/tools/media/cancel"
    assert request.headers["authorization"] == "Bearer cred", "the cancel carries the bearer"
    assert recorder.bodies[0] == {"request_id": "r1"}, "request_id ONLY, never model"


@pytest.mark.asyncio
async def test_radient_cancel_already_completed_and_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for response, expected in (
        (httpx.Response(200, json={"status": "ALREADY_COMPLETED"}), "already_completed"),
        (httpx.Response(404, json={"error": "not found"}), "not_found"),
    ):
        handler = _static_handler(response)
        monkeypatch.setattr(image_rungs, "_client_scope", _scope_for(handler))
        handle = image_rungs.CancelHandle(
            provider=ImageRoute.RADIENT,
            request_id="r1",
            base_url="https://hub.test",
            credential=SecretStr("cred"),
        )
        assert await image_rungs.best_effort_cancel(handle) == expected


@pytest.mark.asyncio
async def test_fal_cancel_put_and_settled_statuses(monkeypatch: pytest.MonkeyPatch) -> None:
    for response, expected in (
        (httpx.Response(202, json={"detail": "CANCELLATION_REQUESTED"}), "cancelled"),
        (httpx.Response(400, json={"detail": "ALREADY_COMPLETED"}), "already_completed"),
        (httpx.Response(404, json={"detail": "not found"}), "not_found"),
    ):
        recorder = _Recorder()

        def handler(request: httpx.Request, response: httpx.Response = response) -> httpx.Response:
            recorder.record(request)
            return response

        monkeypatch.setattr(image_rungs, "_client_scope", _scope_for(handler))
        handle = image_rungs.CancelHandle(
            provider=ImageRoute.FAL,
            request_id="r1",
            cancel_url="https://queue.fal.test/fal-ai/flux/requests/r1/cancel",
            credential=SecretStr("fk"),
        )
        assert await image_rungs.best_effort_cancel(handle) == expected
        assert recorder.requests[0].method == "PUT"
        assert recorder.requests[0].headers["authorization"] == "Key fk"


@pytest.mark.asyncio
async def test_cancel_transport_failure_is_failed_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(image_rungs, "_client_scope", _scope_for(handler))
    handle = image_rungs.CancelHandle(
        provider=ImageRoute.FAL,
        request_id="r1",
        cancel_url="https://queue.fal.test/x/cancel",
        credential=SecretStr("fk"),
    )
    assert await image_rungs.best_effort_cancel(handle) == "failed"


@pytest.mark.asyncio
async def test_cancel_abandons_silently_on_a_second_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Double-Esc: the shield lets the attempt run to its own bound; the outer
    await abandons — and never raises into (or masks) the cancellation path."""

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1.0)
        return httpx.Response(200, json={"status": "CANCELLED"})

    monkeypatch.setattr(image_rungs, "_client_scope", _scope_for(handler))
    handle = image_rungs.CancelHandle(
        provider=ImageRoute.RADIENT,
        request_id="r1",
        base_url="https://hub.test",
        credential=SecretStr("cred"),
    )
    task = asyncio.ensure_future(image_rungs.best_effort_cancel(handle))
    await asyncio.sleep(0.05)
    task.cancel()
    assert await task == "abandoned"
    # Let the shielded attempt finish so nothing is left pending at loop close.
    await asyncio.sleep(1.1)


# ---------------------------------------------------------------------------
# Helpers for the cancel tests (injected scope over a mock transport)
# ---------------------------------------------------------------------------


def _scope_for(handler):
    @asynccontextmanager
    async def scope(client=None):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as owned:
            yield owned

    return scope


def _static_handler(response: httpx.Response):
    def handler(request: httpx.Request) -> httpx.Response:
        return response

    return handler
