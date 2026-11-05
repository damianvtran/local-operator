"""Synthetic multimodal routing controls; no real image, user or session data.

The classifier still decides intent. These tests pin the metadata and prompt
contracts, not a keyword detector pretending to make that decision.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_operator import session_factory
from local_operator.classification.recommend import RecommendationRequest
from local_operator.classification.service import _cache_key
from local_operator.skills.index import render_block
from tests.unit.test_session_factory_classification import (
    _FakeClassifier,
    _FakeIndex,
    _hooks,
    _skill,
)


def transcript(*turns: tuple[str, int]):
    entries = [
        SimpleNamespace(
            type="message",
            payload={
                "role": "user",
                "content": [
                    {"text": text},
                    *[
                        {"attachment": "synthetic-reference", "mime_type": "image/png"}
                        for _ in range(count)
                    ],
                ],
            },
        )
        for text, count in turns
    ]
    return SimpleNamespace(entries=lambda: entries)


@pytest.mark.parametrize(
    "text,count",
    [
        ("Make a version with blue lighting instead", 1),
        ("Combine these into one scene", 2),
        ("Haz una versión con luz azul", 1),
        ("背景を夜にして", 1),
        ("Read the text in this screenshot", 1),
        ("Save this unchanged", 1),
        ("Use exact RGB replacement, no generative model", 1),
    ],
)
def test_multimodal_metadata_reaches_reasoning_without_claiming_intent(text, count):
    hooks = _hooks(_FakeIndex())
    hooks.image_routing_context = session_factory._image_routing_context(transcript((text, count)))
    request = session_factory._classification_request(hooks, text)
    assert request.user_message == text
    assert request.context is not None
    assert f"{count} attached reference image(s)" in request.context
    assert "synthetic-reference" not in request.context
    assert "do not imply generation" in request.context


def test_followup_context_is_reconstructed_after_resume_or_failover():
    history = transcript(("A softer portrait version", 2), ("Actually make it moonlit", 0))
    before = session_factory._image_routing_context(history)
    after = session_factory._image_routing_context(history)
    assert before == after
    assert after is not None
    assert "2 attached reference image(s): A softer portrait version" in after
    assert "0 attached reference image(s): Actually make it moonlit" in after


def test_nonimage_requests_keep_existing_routing():
    assert session_factory._image_routing_context(transcript(("Change a feature flag", 0))) is None
    skill = _skill("flag-targeting", "Change rollout targeting")
    assert "Read these selected skills immediately" in render_block([skill])
    assert "Read these selected skills immediately" not in render_block(
        [skill], conditional_skills=True
    )
    assert "ignore unrelated matches" in render_block([skill], conditional_skills=True)


def test_context_participates_in_cache_identity():
    edit = RecommendationRequest("Make it blue", "1 attached reference image: portrait", ())
    analysis = RecommendationRequest("Make it blue", "OCR bounding box annotation", ())
    assert _cache_key(edit, 12) != _cache_key(analysis, 12)
    assert _cache_key(edit, 12) == _cache_key(edit, 12)


@pytest.mark.asyncio
async def test_image_lane_never_mandates_an_unrelated_embedding_match():
    # Reproduce the selection side of the incident without any private source:
    # an embedding match is not a reason to require a domain-specific read.
    hooks = _hooks(
        _FakeIndex(picked=[_skill("flag-targeting", "Feature flags")]), classifier=_FakeClassifier()
    )
    hooks.image_routing_context = session_factory._image_routing_context(
        transcript(("Blue lighting instead", 1))
    )
    block = await session_factory._select_knowledge_block(
        hooks, "Blue lighting instead", task_id="fixture"
    )
    assert "Read these selected skills immediately" not in block
    assert "ignore unrelated matches" in block
    assert "tool://generate_image" not in block  # An empty classifier cannot force a tool.


@pytest.mark.asyncio
async def test_classifier_unavailable_still_preserves_image_lane():
    hooks = _hooks(_FakeIndex(picked=[_skill("flag-targeting", "Feature flags")]))
    hooks.image_routing_context = session_factory._image_routing_context(
        transcript(("Blue lighting instead", 1))
    )
    block = await session_factory._select_knowledge_block(
        hooks, "Blue lighting instead", task_id="fixture"
    )
    assert "ignore unrelated matches" in block
    assert "resource_recommendations" not in block
