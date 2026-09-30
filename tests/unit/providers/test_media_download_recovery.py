"""Media-download recovery: a provider 400 whose data inspection could not
download media the request carried is re-asked identically, bounded, and
legible.

The class under test is what the F2 arm hit: an HTTP 400 whose provider-side
inspection step could not download a media item the request carried, relayed
with the message ``Failed to download multimodal content`` -- the request was
never processed by the model. These tests drive the REAL
:func:`stream_with_failover` with scripted wires that replay the captured
message, so the recovery, its ceiling, and the terminal message are exercised
through the same walk every consumer (session loop, evaluation runner, errands)
uses.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import (
    ChatRequest,
    Content,
    ImageContent,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    TextContent,
)
from local_operator.providers.failover import (
    MAX_MEDIA_DOWNLOAD_RETRIES,
    ProviderError,
    is_input_refusal,
    is_media_download_failure,
    stream_with_failover,
)
from tests.unit.providers.test_failover import FakeAuth, ScriptedClient, _FnClient

pytestmark = pytest.mark.asyncio

#: The message the walk held in the field's record, verbatim apart from the
#: relay id: what ``_extract_error_message`` hands the walk when the provider's
#: data inspection cannot download media the request carried (captured from
#: runs/a1796-w3-task_009-20260930-001343, task_009 log line 16).
FIELD_MESSAGE = (
    'data: {"error":{"code":"invalid_parameter_error","param":null,'
    '"message":"Failed to download multimodal content",'
    '"type":"invalid_request_error"},'
    '"id":"chatcmpl-92b0db86-abb3-91a7-8810-bebad3940f29"}'
)

#: The same message as the frame rendered it, and as classification sees it
#: on the ``AgentEndEvent.error`` path.
_RECORDED = "invalid request (HTTP 400): " + FIELD_MESSAGE


@pytest.fixture(autouse=True)
def _fast_re_ask_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """The walk's flat settle delay is real time on the wire; shrink it here.

    Same discipline as the flap-retry tests in ``test_failover.py``, which pin
    ``MODEL_FLAP_RETRY_DELAY_MS`` to 1: the constant under test is the
    BEHAVIOUR, not the wall delay.
    """
    monkeypatch.setattr("local_operator.providers.failover.MEDIA_DOWNLOAD_RETRY_DELAY_MS", 1)


def _media_download() -> ProviderError:
    return ProviderError(400, FIELD_MESSAGE)


def _media_request() -> ChatRequest:
    """A request shaped like the field's: tool observations carrying frames."""
    messages: list[Message] = [Message.user("Task: fill the spreadsheet")]
    for index in range(3):
        content: list[Content] = [TextContent(text=f"Step: {index} observation")]
        content.append(ImageContent(data="aGVsbG8=", mime_type="image/png"))
        messages.append(
            Message(
                role="tool",
                content=content,
                tool_call_id=f"call_{index}",
                tool_name="apply_actions",
            )
        )
    return ChatRequest(
        model=ModelSpec(provider="openrouter", model_id="qwen/qwen3.8-max-0902"),
        messages=messages,
    )


def _scripted(
    failures: int, seen: list[ChatRequest], *, error: Any = None
) -> tuple[Any, dict[str, int], list[ChatRequest]]:
    """A client_for + bookkeeping: fail the first ``failures`` attempts."""

    calls = {"n": 0}
    raised = error or _media_download()

    async def client_for(spec: ModelSpec):
        def run(request: ChatRequest, api_key: str | None, oauth_access=None):
            calls["n"] += 1
            seen.append(request)

            async def gen():
                if calls["n"] <= failures:
                    raise raised
                yield StreamTextDelta(delta="continued")
                yield StreamEndEvent(stop_reason="stop")

            return gen()

        return _FnClient(run)

    return client_for, calls, seen


async def test_the_predicate_recognizes_the_field_message_and_only_the_class() -> None:
    assert _media_download().kind == "request"
    assert is_media_download_failure(_media_download())
    assert is_media_download_failure(_RECORDED)

    # The provider's own documented timeout variant is the SAME class, and its
    # "timed out" wording makes the classifier read it as kind == "timeout":
    # the recovery must not split on which wording a relay used.
    timeout_variant = ProviderError(
        400, "Download the media resource timed out during the data inspection process."
    )
    assert timeout_variant.kind == "timeout"
    assert is_media_download_failure(timeout_variant)
    assert is_media_download_failure(
        ProviderError(
            400, "Unable to download the media resource during the data inspection process."
        )
    )

    # A 5xx mentioning a download is the provider failing on its own side and
    # keeps the ordinary retry ladder.
    assert not is_media_download_failure(
        ProviderError(502, "download the media resource timed out", retryable=True)
    )
    # An unrelated 400 that merely mentions media keeps the terminal path:
    # the decode and size variants are their own documented classes, and a
    # corrupt payload IS deterministic in our bytes.
    assert not is_media_download_failure(ProviderError(400, "media_type field is required"))
    assert not is_media_download_failure(
        ProviderError(400, "Failed to decode the image during the data inspection")
    )
    # And this class is NOT the content screen's refusal: the degrade ladder's
    # premise ("the same bytes get the same answer") is exactly what this class
    # does not have, and the refusal's remedy (narrow the request) is not what
    # this class needs.
    assert not is_input_refusal(_media_download())


async def test_a_media_download_failure_is_re_asked_identically_and_recovers() -> None:
    seen: list[ChatRequest] = []
    client_for, calls, _ = _scripted(failures=1, seen=seen)

    events = [
        event
        async for event in stream_with_failover(
            _media_request(), FakeAuth({"openrouter": ["k"]}), None, client_for
        )
    ]

    assert calls["n"] == 2
    # The re-ask is IDENTICAL bytes: this class has no request change to make.
    assert seen[1].model_dump() == seen[0].model_dump()
    end = [event for event in events if isinstance(event, StreamEndEvent)][-1]
    assert end.stop_reason == "stop"
    assert end.provider_payload is None


async def test_the_re_ask_is_bounded_and_the_terminal_is_legible() -> None:
    """All attempts fail: two identical re-asks, then a legible terminal.

    The ceiling is ``MAX_MEDIA_DOWNLOAD_RETRIES`` re-asks (three wire requests
    total), every one byte-identical -- the bound is on ATTEMPTS because there
    is nothing about the request to narrow.
    """
    seen: list[ChatRequest] = []
    client_for, calls, _ = _scripted(failures=99, seen=seen)

    with pytest.raises(ProviderError) as caught:
        [
            event
            async for event in stream_with_failover(
                _media_request(), FakeAuth({"openrouter": ["k"]}), None, client_for
            )
        ]

    assert calls["n"] == 1 + MAX_MEDIA_DOWNLOAD_RETRIES
    assert all(request.model_dump() == seen[0].model_dump() for request in seen)
    error = caught.value
    assert error.kind == "request" and error.status == 400
    # The provider's words stay in FRONT (classifiers read the message)...
    assert "Failed to download multimodal content" in error.message
    # ...and the note adds the two facts the raw frame lacked: the class of
    # failure, the never-processed fact, and the spent budget.
    assert "the provider's data inspection could not download a media item" in error.message
    assert "the request was never processed by the model" in error.message
    assert "A bounded re-ask (2) also failed" in error.message


async def test_the_documented_timeout_wording_is_re_asked_too() -> None:
    """The family's OTHER documented wording must take the same recovery.

    "Download the media resource timed out during the data inspection
    process" carries "timed out", so the classifier reads it as kind
    ``timeout`` (see the predicate test) -- a kind the walk's ordinary ladder
    would treat as a hop-worthy 5xx-shaped failure at its worst. The walk must
    recover it exactly like the field's message, on the same credential.
    """
    timeout_message = "Download the media resource timed out during the data inspection process."
    seen: list[ChatRequest] = []
    client_for, calls, _ = _scripted(
        failures=2, seen=seen, error=ProviderError(400, timeout_message)
    )
    auth = FakeAuth({"openrouter": ["k"]})

    events = [
        event async for event in stream_with_failover(_media_request(), auth, None, client_for)
    ]

    assert calls["n"] == 3, "two bounded re-asks, then the recovered attempt"
    assert auth.rotations == []
    end = [event for event in events if isinstance(event, StreamEndEvent)][-1]
    assert end.stop_reason == "stop"


async def test_no_credential_rotation_is_spent_on_the_re_ask() -> None:
    """The re-ask stays on the same target AND the same credential.

    Nothing about the account can fix a media fetch on the provider's side, so
    rotation here would only spend siblings on a condition none of them caused.
    """
    seen: list[ChatRequest] = []
    client_for, calls, _ = _scripted(failures=99, seen=seen)
    auth = FakeAuth({"openrouter": ["k1", "k2"]})

    with pytest.raises(ProviderError):
        [event async for event in stream_with_failover(_media_request(), auth, None, client_for)]

    assert calls["n"] == 1 + MAX_MEDIA_DOWNLOAD_RETRIES
    assert auth.rotations == []


async def test_a_retry_disabled_call_is_not_re_asked_but_stays_legible() -> None:
    seen: list[ChatRequest] = []
    client_for, calls, _ = _scripted(failures=99, seen=seen)

    with pytest.raises(ProviderError) as caught:
        [
            event
            async for event in stream_with_failover(
                _media_request(),
                FakeAuth({"openrouter": ["k"]}),
                {"retry": {"enabled": False}},
                client_for,
            )
        ]

    assert calls["n"] == 1
    assert "the provider's data inspection could not download a media item" in caught.value.message
    # Policy-declined is its own fact, not "no re-ask existed" -- the message
    # must say the policy declined it, the same distinction the input-refusal
    # note draws.
    assert "retry policy declined" in caught.value.message


async def test_the_failure_is_terminal_for_the_walk_after_the_budget() -> None:
    """A media-download failure never hands its request to another provider.

    A hop would serve a different model's answer for a request the pinned route
    could not process -- the substitution red line -- so once the budget is
    spent the walk raises instead of walking the configured fallbacks.
    """
    tried: list[str] = []
    seen: list[ChatRequest] = []

    async def client_for(spec: ModelSpec):
        def run(request: ChatRequest, api_key: str | None, oauth_access=None):
            tried.append(spec.model_id)
            seen.append(request)
            # A ``ScriptedClient`` seeded with the exception raises from its
            # stream on first iteration; it is an async GENERATOR, which a
            # bare ``async def gen(): raise ...`` would not be (no yield in
            # the body, so it would hand ``_FnClient`` a coroutine it cannot
            # iterate).
            return ScriptedClient(_media_download()).stream(request, api_key, oauth_access)

        return _FnClient(run)

    settings = {
        "retry": {
            "baseDelayMs": 1,
            "fallbackChains": {"default": ["anthropic/claude-x", "google/gemini-x"]},
        }
    }
    with pytest.raises(ProviderError) as caught:
        [
            event
            async for event in stream_with_failover(
                _media_request(),
                FakeAuth({"openrouter": ["k"], "anthropic": ["k"], "google": ["k"]}),
                settings,
                client_for,
            )
        ]

    assert tried == ["qwen/qwen3.8-max-0902"] * (1 + MAX_MEDIA_DOWNLOAD_RETRIES)
    assert "A bounded re-ask (2) also failed" in caught.value.message


async def test_a_normal_call_is_untouched() -> None:
    """The happy path: one attempt, the request unmoved byte for byte, no note."""
    seen: list[ChatRequest] = []
    client_for, calls, _ = _scripted(failures=0, seen=seen)
    request = _media_request()
    before = request.model_dump()

    events = [
        event
        async for event in stream_with_failover(
            request, FakeAuth({"openrouter": ["k"]}), None, client_for
        )
    ]

    assert calls["n"] == 1
    assert seen[0].model_dump() == before
    end = [event for event in events if isinstance(event, StreamEndEvent)][-1]
    assert end.stop_reason == "stop"
    assert end.provider_payload is None
