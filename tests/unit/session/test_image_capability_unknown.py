"""A model nothing has described must not be treated as one that refuses images.

The defect these pin, reproduced on the real session runtime: a session on
``openrouter/deepseek/deepseek-v4.1-flash-1017`` (a vision model — OpenRouter's
own endpoint listing states ``input_modalities: ['text', 'image']``) stripped
every image and announced ``tui.notices.image_unsupported``. Its stored spec
was the aggregator PLACEHOLDER verbatim — 128k window, no prompt cache,
``supports_images=False`` — because the OpenRouter listing had not been cached
yet when the session started and models.dev never answers the image question
(``prices._row`` leaves it ``None`` by design). With every leg silent,
resolution fell to ``openrouter_default_model_info``, whose ``False`` was not
"unknown" but a positive denial ``build_model_spec`` is required to trust.

The fix is at the source: the "nothing is known" templates say ``None``. The
other half — a model whose listing DOES deny images still strips and still
announces — is pinned here too, so the fix cannot be "never strip".
"""

from __future__ import annotations

import asyncio

import pytest

from local_operator.harness.types import (
    ImageContent,
    Message,
    NoticeEvent,
    StreamEndEvent,
    StreamTextDelta,
    TextContent,
)
from local_operator.model import configure
from local_operator.model.configure import build_model_spec, info_from_discovered_model
from local_operator.model.discovery import DiscoveredModel
from local_operator.model.registry import (
    get_model_info,
    openrouter_default_model_info,
    radient_default_model_info,
    unknown_model_info,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

_VISION_ROUTE = "deepseek/deepseek-v4.1-flash-1017"


@pytest.fixture
def silent_legs(monkeypatch):
    """Every enrichment leg answers nothing — the episode's start-up state.

    Identity stubs rather than a blocked socket: the claim is about what the
    REGISTRY's fallback says when no source speaks, and a stub states that
    precondition exactly instead of depending on how a network failure is
    classified.
    """
    monkeypatch.setattr(configure, "_info_from_discovery", lambda _p, _m, info, **_k: info)
    monkeypatch.setattr(configure, "_from_price_catalogue", lambda _p, _m, info, **_k: info)
    monkeypatch.setattr(configure, "_from_aggregator_catalogue", lambda _p, _m, info, **_k: info)
    configure.invalidate_model_info_cache()
    yield
    configure.invalidate_model_info_cache()


@pytest.mark.parametrize(
    "template", [unknown_model_info, openrouter_default_model_info, radient_default_model_info]
)
def test_the_nothing_known_templates_do_not_deny_images(template) -> None:
    """``None`` is the registry's spelling of "no source stated it"."""
    assert template.supports_images is None


@pytest.mark.parametrize("provider", ["openrouter", "radient"])
def test_an_undescribed_aggregator_model_keeps_images(provider: str, silent_legs) -> None:
    """The reported failure, through the same resolver the session uses."""
    info = configure.resolve_model_info(provider, _VISION_ROUTE)
    assert info.supports_images is None, "no source spoke, so nothing may deny"

    spec = build_model_spec(provider, _VISION_ROUTE)
    assert spec.supports_images is True, "an unknown capability was collapsed into a denial"


def test_an_undescribed_direct_model_keeps_images(silent_legs) -> None:
    """The same collapse on a direct provider's unshipped id (the global sentinel)."""
    assert get_model_info("xai", "grok-never-shipped") is unknown_model_info
    assert build_model_spec("xai", "grok-never-shipped").supports_images is True


def test_a_listing_that_denies_images_still_wins() -> None:
    """A STATED ``False`` (a listing row whose modalities omit ``image``) is
    still a denial: the three-state contract is kept, not loosened to True."""
    row = DiscoveredModel(
        id="vendor/text-only", name="Text Only", context_window=64_000, supports_images=False
    )
    info = info_from_discovered_model(
        "openrouter", "vendor/text-only", row, openrouter_default_model_info
    )
    assert info.supports_images is False
    assert build_model_spec("openrouter", "vendor/text-only", info).supports_images is False


class _Stream:
    """Records requests and answers every turn with one short reply."""

    def __init__(self) -> None:
        self.requests = []

    def __call__(self, request, signal):
        self.requests.append(request)

        async def gen():
            yield StreamTextDelta(delta="ok")
            yield StreamEndEvent(stop_reason="stop")

        return gen()


async def _render_with_screenshot(tmp_path, spec):
    """Send one turn over a history carrying a screenshot; return images sent + notices."""
    stream = _Stream()
    session = Session(
        model=spec,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable", "env"],
    )
    events = []
    session.subscribe(events.append)
    await session.seed_history(
        [Message(role="user", content=[TextContent(text="screen"), ImageContent(data="Zm9v")])]
    )
    await session.prompt("what is on screen?")
    # The notice is emitted from a spawned task; give it one loop pass.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    sent = sum(
        isinstance(block, ImageContent)
        for message in stream.requests[-1].messages
        for block in message.content
    )
    notices = [
        event
        for event in events
        if isinstance(event, NoticeEvent) and event.code == "tui.notices.image_unsupported"
    ]
    await session.dispose()
    return sent, notices


@pytest.mark.asyncio
async def test_a_session_on_an_undescribed_model_sends_the_image(tmp_path, silent_legs) -> None:
    sent, notices = await _render_with_screenshot(
        tmp_path, build_model_spec("openrouter", _VISION_ROUTE)
    )
    assert sent == 1, "the screenshot was stripped from an undescribed model's request"
    assert notices == [], "the session announced an incapacity nothing stated"


@pytest.mark.asyncio
async def test_a_session_on_a_stated_text_only_model_still_strips(tmp_path) -> None:
    row = DiscoveredModel(id="vendor/text-only", name="Text Only", supports_images=False)
    info = info_from_discovered_model(
        "openrouter", "vendor/text-only", row, openrouter_default_model_info
    )
    sent, notices = await _render_with_screenshot(
        tmp_path, build_model_spec("openrouter", "vendor/text-only", info)
    )
    assert sent == 0, "a model whose listing denies images was sent one"
    assert len(notices) == 1, "the strip must still be announced"
