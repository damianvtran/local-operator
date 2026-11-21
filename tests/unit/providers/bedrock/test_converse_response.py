"""Native JSON/stream contract edges, source Region and cancellation ownership."""

import asyncio

import httpx
import pytest
from botocore.credentials import Credentials

from local_operator.harness.types import StreamEndEvent, StreamTextDelta
from local_operator.providers.bedrock import auth
from local_operator.providers.bedrock.converse import (
    BedrockConverseClient,
    build_converse_body,
)
from local_operator.providers.bedrock.models import capabilities_for_spec

from .test_converse_body import NOVA, TOOL, request
from .test_converse_stream import KEY, events, wire


def test_arn_preserved_and_signed_for_connection_source_region(monkeypatch):
    arn = "arn:aws:bedrock:eu-west-1:123456789012:inference-profile/us." + NOVA
    req = request(arn)
    seen = []

    async def resolved(*args, **kwargs):
        return auth.ResolvedAuth(
            region="us-west-2",
            credentials=Credentials("synthetic-id", "synthetic-secret").get_frozen_credentials(),
        )

    monkeypatch.setattr(auth, "resolve", resolved)

    async def run():
        def handler(value):
            seen.append(value)
            return httpx.Response(200, content=wire(events()))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return [
                event async for event in BedrockConverseClient(http_client=http).stream(req, KEY)
            ]

    result = asyncio.run(run())
    assert isinstance(result[-1], StreamEndEvent)
    assert result[-1].stop_reason == "stop"
    assert seen[0].url.host == "bedrock-runtime.us-west-2.amazonaws.com"
    assert (
        b"arn%3Aaws%3Abedrock%3Aeu-west-1%3A123456789012%3Ainference-profile%2Fus."
        in seen[0].url.raw_path
    )
    assert "/us-west-2/bedrock/aws4_request" in seen[0].headers["authorization"]


def test_known_nonstreaming_metadata_uses_json_even_without_tools(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "synthetic-bearer")
    req = request("opaque-profile")
    req.model = req.model.model_copy(update={"bedrock_metadata": {"streaming": False}})
    paths = []

    async def run():
        def handler(value):
            paths.append(value.url.path)
            return httpx.Response(
                200,
                json={
                    "output": {"message": {"role": "assistant", "content": [{"text": "json"}]}},
                    "stopReason": "end_turn",
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return [
                event async for event in BedrockConverseClient(http_client=http).stream(req, KEY)
            ]

    result = asyncio.run(run())
    assert isinstance(result[1], StreamTextDelta)
    assert result[1].delta == "json"
    assert paths == ["/model/opaque-profile/converse"]


def test_converse_body_matches_installed_aws_contract_without_network():
    from botocore.session import Session
    from botocore.validate import validate_parameters

    req = request(tools=[TOOL])
    body = build_converse_body(req, capabilities_for_spec(req.model), "endpoint", "scope")
    shape = Session().get_service_model("bedrock-runtime").operation_model("Converse").input_shape
    validate_parameters({"modelId": req.model.model_id, **body}, shape)


def test_cancellation_closes_stream_and_preserves_injected_pool(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "synthetic-bearer")
    entered = asyncio.Event()
    closed = []

    class Blocking(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield wire(events()[:1])
            entered.set()
            await asyncio.Event().wait()

        async def aclose(self):
            closed.append(True)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=Blocking()))
        ) as http:

            async def consume():
                return [
                    event
                    async for event in BedrockConverseClient(http_client=http).stream(
                        request(), KEY
                    )
                ]

            task = asyncio.create_task(consume())
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not http.is_closed

    asyncio.run(run())
    assert closed == [True]


def test_image_rejection_uses_native_paths_and_only_retries_before_output(monkeypatch):
    import json

    from local_operator.harness.types import ImageContent, Message
    from local_operator.providers.image_rejection import forget_rejected_images

    from .test_converse_body import png

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "synthetic-bearer")
    forget_rejected_images()
    req = request()
    req.messages = [Message(role="user", content=[ImageContent(data=png())])]
    bodies = []

    async def run():
        def handler(value):
            bodies.append(json.loads(value.content))
            if len(bodies) == 1:
                return httpx.Response(
                    400,
                    headers={"x-amzn-ErrorType": "ValidationException"},
                    json={
                        "message": (
                            ".messages[0].content[0].image: "
                            "You have uploaded an unsupported image. "
                            "Please make sure your image is valid."
                        )
                    },
                )
            return httpx.Response(200, content=wire(events()))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return [
                event async for event in BedrockConverseClient(http_client=http).stream(req, KEY)
            ]

    try:
        result = asyncio.run(run())
        assert isinstance(result[-1], StreamEndEvent)
        assert result[-1].stop_reason == "stop"
        assert len(bodies) == 2 and "image" in bodies[0]["messages"][0]["content"][0]
        assert "image" not in bodies[1]["messages"][0]["content"][0]
        assert "text" in bodies[1]["messages"][0]["content"][0]
    finally:
        forget_rejected_images()
