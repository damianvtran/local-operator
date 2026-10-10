"""The subscription rung's wire, against ``httpx.MockTransport`` — no network.

What is pinned here is the documented contract (no live probe ran in this
wave, by operator decision — see ``docs/design/image-providers.md`` for the
tiered citations): the Codex responses call shape (URL, headers, body flags),
the SSE parse that finds the ``image_generation_call`` result (and only a
completed one), the recorded skips (img2img; >1 image), the no-cancel absence,
and the failure classes that fail forward.
"""

from __future__ import annotations

import base64
import inspect
import json
from typing import Any, Callable

import httpx
import pytest

from local_operator.artifacts.rung import RungSkipped
from local_operator.clients._http import APIError
from local_operator.imagegen import ImageRoute
from local_operator.imagegen import rungs_openai_sub as rung_mod

#: A 73-byte 1x1 PNG (real bytes, so nothing has to decode anything).
PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c626001000000ffff03000006000557bfabd4"
    "0000000049454e44ae426082"
)
PNG_B64 = base64.b64encode(PNG_1X1).decode("ascii")


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _sse(*events: dict[str, Any]) -> bytes:
    """An SSE byte payload: one ``data:`` line (compact JSON) per event."""
    lines: list[str] = []
    for event in events:
        lines.append("event: " + str(event.get("type", "message")))
        lines.append("data: " + json.dumps(event))
        lines.append("")
    return ("\n".join(lines) + "\n").encode()


def _completed_event(result: str = PNG_B64) -> dict[str, Any]:
    return {
        "type": "response.completed",
        "response": {
            "id": "resp_1",
            "status": "completed",
            "output": [
                {
                    "type": "image_generation_call",
                    "id": "ig_1",
                    "status": "completed",
                    "result": result,
                }
            ],
        },
    }


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


@pytest.mark.asyncio
async def test_the_streamed_call_shape_and_the_decoded_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rung_mod, "_default_host_model", lambda: "host-model")
    recorder = _Recorder()
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=_sse(
            {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}},
            _completed_event(),
        ),
    )
    http = _client(recorder.handler(response))

    result = await rung_mod.run_openai_sub(
        prompt="a cat",
        access_token="tok-123",
        account_id="acc-9",
        num_images=1,
        image_size="square_hd",
        source_url=None,
        seed=7,
        model=None,
        emit=None,
        pause=None,
        client=http,
    )

    assert result.model == "host-model"
    assert result.cost_usd is None
    assert result.cost_source == "subscription"
    assert len(result.assets) == 1
    asset = result.assets[0]
    assert asset.data == PNG_1X1
    assert asset.content_type == "image/png"
    assert asset.source_url == ""

    request = recorder.requests[0]
    assert request.method == "POST"
    assert str(request.url) == rung_mod.CODEX_RESPONSES_URL
    assert request.headers["authorization"] == "Bearer tok-123"
    assert request.headers["chatgpt-account-id"] == "acc-9"
    assert request.headers["accept"] == "text/event-stream"
    assert request.headers["openai-beta"] == rung_mod.CODEX_BETA_HEADER
    body = recorder.bodies[0]
    assert body["model"] == "host-model"
    # Review round 1 F2: the top-level instructions follow the chat client's
    # same-backend shape (tier: open risk until the live probe).
    assert body["instructions"] == rung_mod.OPENAI_SUB_INSTRUCTIONS
    assert body["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "a cat"}],
        }
    ]
    assert body["tools"] == [{"type": "image_generation"}]
    assert body["tool_choice"] == {"type": "image_generation"}
    assert body["stream"] is True
    assert body["store"] is False
    # The wire cannot carry a seed: it is dropped, never smuggled.
    assert "seed" not in body


@pytest.mark.asyncio
async def test_the_account_header_is_omitted_without_one() -> None:
    recorder = _Recorder()
    response = httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=_sse(_completed_event())
    )
    http = _client(recorder.handler(response))

    await rung_mod.run_openai_sub(
        prompt="a cat",
        access_token="tok-123",
        account_id=None,
        num_images=1,
        image_size="square_hd",
        source_url=None,
        seed=None,
        model="m",
        emit=None,
        pause=None,
        client=http,
    )

    assert "chatgpt-account-id" not in recorder.requests[0].headers


@pytest.mark.asyncio
async def test_the_output_item_done_and_completed_events_deduplicate() -> None:
    recorder = _Recorder()
    item_event = {
        "type": "response.output_item.done",
        "item": {
            "type": "image_generation_call",
            "id": "ig_1",
            "status": "completed",
            "result": PNG_B64,
        },
    }
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=_sse(item_event, _completed_event()),
    )
    http = _client(recorder.handler(response))

    result = await rung_mod.run_openai_sub(
        prompt="a cat",
        access_token="tok",
        account_id=None,
        num_images=1,
        image_size="square_hd",
        source_url=None,
        seed=None,
        model="m",
        emit=None,
        pause=None,
        client=http,
    )

    assert len(result.assets) == 1, "the same render must not attach twice"


@pytest.mark.asyncio
async def test_progress_frames_are_canonical_and_labelled() -> None:
    recorder = _Recorder()
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=_sse(
            {"type": "response.created", "response": {"id": "r", "status": "in_progress"}},
            {"type": "response.in_progress", "response": {"id": "r", "status": "in_progress"}},
            _completed_event(),
        ),
    )
    http = _client(recorder.handler(response))
    updates: list[tuple[str, dict[str, Any]]] = []

    await rung_mod.run_openai_sub(
        prompt="a cat",
        access_token="tok",
        account_id=None,
        num_images=1,
        image_size="square_hd",
        source_url=None,
        seed=None,
        model="m",
        emit=lambda text, details: updates.append((text, details)),
        pause=None,
        client=http,
    )

    assert updates, "the stream should emit at least one in-progress frame"
    # The initial frame leads (review round 1 nit): a surface shows the rung
    # running from the first moment, before any SSE event lands.
    assert updates[0][0].endswith("running — 0s")
    canonical = {"stage", "queue_position", "progress_fraction", "log_lines", "error", "error_type"}
    for text, details in updates:
        assert details["provider"] == "openai-sub"
        assert details["tool_name"] == "generate_image"
        assert details["stage"] == "in_progress"
        assert canonical <= set(details)
        assert "ChatGPT plan" in text


@pytest.mark.asyncio
async def test_img2img_and_multi_image_are_recorded_skips() -> None:
    with pytest.raises(RungSkipped) as caught:
        await rung_mod.run_openai_sub(
            prompt="edit it",
            access_token="tok",
            account_id=None,
            num_images=1,
            image_size="square_hd",
            source_url="data:image/png;base64,AAAA",
            seed=None,
            model=None,
            emit=None,
            pause=None,
        )
    assert caught.value.reason_class == "unsupported"

    with pytest.raises(RungSkipped) as caught:
        await rung_mod.run_openai_sub(
            prompt="a cat",
            access_token="tok",
            account_id=None,
            num_images=2,
            image_size="square_hd",
            source_url=None,
            seed=None,
            model=None,
            emit=None,
            pause=None,
        )
    assert caught.value.reason_class == "unsupported"


@pytest.mark.asyncio
async def test_an_http_failure_maps_to_the_lane_error_shape() -> None:
    recorder = _Recorder()
    response = httpx.Response(401, json={"error": {"message": "bad grant"}})
    http = _client(recorder.handler(response))

    with pytest.raises(APIError) as caught:
        await rung_mod.run_openai_sub(
            prompt="a cat",
            access_token="tok",
            account_id=None,
            num_images=1,
            image_size="square_hd",
            source_url=None,
            seed=None,
            model="m",
            emit=None,
            pause=None,
            client=http,
        )
    assert caught.value.status_code == 401
    # The bearer is scrubbed from everything the error carries.
    assert "tok" not in (caught.value.body or "")


@pytest.mark.asyncio
async def test_a_failed_response_event_raises_upstream() -> None:
    recorder = _Recorder()
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=_sse(
            {
                "type": "response.failed",
                "response": {
                    "id": "r",
                    "status": "failed",
                    "error": {"message": "quota exhausted"},
                },
            }
        ),
    )
    http = _client(recorder.handler(response))

    with pytest.raises(APIError) as caught:
        await rung_mod.run_openai_sub(
            prompt="a cat",
            access_token="tok",
            account_id=None,
            num_images=1,
            image_size="square_hd",
            source_url=None,
            seed=None,
            model="m",
            emit=None,
            pause=None,
            client=http,
        )
    assert caught.value.code == "upstream"
    assert "quota exhausted" in str(caught.value)


@pytest.mark.asyncio
async def test_a_stream_without_an_image_is_invalid_response() -> None:
    recorder = _Recorder()
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=_sse({"type": "response.completed", "response": {"id": "r", "output": []}}),
    )
    http = _client(recorder.handler(response))

    with pytest.raises(APIError) as caught:
        await rung_mod.run_openai_sub(
            prompt="a cat",
            access_token="tok",
            account_id=None,
            num_images=1,
            image_size="square_hd",
            source_url=None,
            seed=None,
            model="m",
            emit=None,
            pause=None,
            client=http,
        )
    assert caught.value.code == "invalid_response"


def test_the_executor_takes_no_cancel_handle_documented_absence() -> None:
    """The subscription route has no provider-side cancel (RFC-aligned).

    The absence is the contract: the handle is never filled, so a
    best-effort cancel reports "none — nothing to cancel" exactly like the
    OpenAI-key precedent. A signature without ``handle`` is how this rung
    states that; if a future provider grows a cancel, it adds the parameter.
    """
    parameters = inspect.signature(rung_mod.run_openai_sub).parameters
    assert "handle" not in parameters


def test_the_route_spelling_matches_the_chat_client() -> None:
    """One URL, two spellings; a drift here would be a silent wire split."""
    from local_operator.providers import clients as chat_clients

    assert rung_mod.CODEX_RESPONSES_URL == chat_clients.CODEX_RESPONSES_URL
    assert rung_mod.CODEX_BETA_HEADER == chat_clients.CODEX_BETA_HEADER


def test_the_spec_declares_none_cancel_and_subscription_cost() -> None:
    from local_operator.artifacts.rung import CancelSupport
    from local_operator.imagegen import cascade

    spec = cascade.RUNG_SPECS[ImageRoute.OPENAI_SUB]
    assert spec.label == "ChatGPT plan"
    assert spec.cancel_support == CancelSupport.NONE
    assert spec.cost == "subscription"
    assert spec.capabilities == frozenset({"t2i"})
