"""Kimi K3.1 Pro's stated capabilities must reach the wire, not just the picker.

The listing read (``model.discovery``) and the spec build (``model.configure``)
are only half the chain: a ladder the body never carries is a control that does
nothing, and an image capability whose attachment the body drops is the same
omission with a friendlier status band. These tests build the real
``ChatRequest`` body for the model and read the keys back out of it.

Live counterpart, measured 2026-10-24 against
``https://api.kimi.com/coding/v1/chat/completions`` on ``k3.1-pro``: top-level
``reasoning_effort`` at ``low`` and ``max`` returned distinct reasoning-token
counts on every run, and an ``image_url`` data-URI turn answered HTTP 200 with
"Red". Nothing here sends; the provider evidence lives in the PR.
"""

from __future__ import annotations

from local_operator.harness.types import ChatRequest, ImageContent, Message, ModelSpec
from local_operator.model import discovery
from local_operator.model.configure import build_model_spec, info_from_discovered_model
from local_operator.model.registry import unknown_model_info
from local_operator.providers.clients import OpenAICompatClient

#: One verbatim-trimmed row off ``GET https://api.kimi.com/coding/v1/models``
#: (read 2026-10-24). The row is PARSED here rather than hand-built, so the whole
#: chain under test is the one a session walks: listing JSON → ``DiscoveredModel``
#: → ``ModelInfo`` → ``ModelSpec`` → request body.
_KIMI_LISTING_ENTRY: dict[str, object] = {
    "id": "k3.1-pro",
    "display_name": "K3.1 Pro",
    "type": "model",
    "context_length": 1_048_576,
    "supports_reasoning": True,
    "supports_image_in": True,
    "supports_video_in": True,
    "supports_dynamic_tools": True,
    "supports_thinking_type": "only",
    "think_efforts": {
        "support": True,
        "valid_efforts": ["low", "high", "max"],
        "default_effort": "high",
    },
    # The same ladder stated a second time, with the default nested under
    # ``effort``: the real row carries both shapes, and neither is carried as a
    # default (see ``discovery._row_from_openai_entry``).
    "reasoning": {
        "type": "only",
        "effort": {"valid": ["low", "high", "max"], "default": "high"},
    },
    "modalities": {"input": ["text", "image", "video"], "output": ["text"]},
}

_KIMI_BASE = "https://api.kimi.com/coding/v1"


def _kimi_spec() -> ModelSpec:
    """The spec a session started on ``kimi/k3.1-pro`` actually runs on.

    ``k3.1-pro`` has no registry twin — it is a DYNAMIC row — so the fallback
    is the same :data:`unknown_model_info` the catalogue path supplies for an
    unshipped id (``supports_images=False``, no ladder). That matters for the
    assertion below: with a permissive hand-built fallback the image assertion
    would pass even on the pre-fix tree, because ``build_model_spec`` defaults
    an unstated capability to ``True``. With the real fallback, a parse that
    reads nothing leaves ``supports_images`` at ``False`` — the defect.
    """
    row = discovery._row_from_openai_entry(_KIMI_LISTING_ENTRY, "kimi")
    assert row is not None
    info = info_from_discovered_model("kimi", "k3.1-pro", row, unknown_model_info)
    return build_model_spec("kimi", "k3.1-pro", info)


def _body(spec: ModelSpec) -> dict[str, object]:
    return OpenAICompatClient(_KIMI_BASE)._build_body(
        ChatRequest(
            model=spec,
            messages=[
                Message.user(
                    "What colour is this?",
                    [ImageContent(data="iVBORw0KGgo=", mime_type="image/png")],
                )
            ],
        )
    )


def test_a_listed_kimi_row_reaches_the_spec_with_its_ladder_and_images() -> None:
    """End of the chain the defect broke: the listing's statements must survive
    ``info_from_discovered_model`` and ``build_model_spec``. Before the fix this
    spec carried ``reasoning_efforts=()`` and ``supports_images=False``, which is
    why the composer had no effort control and the harness announced the model
    would not accept images."""
    spec = _kimi_spec()

    assert spec.reasoning_efforts == ("low", "high", "max")
    assert spec.supports_images is True
    assert spec.reasoning is True
    # Auto: the listing's ``default_effort`` is deliberately not seeded (see
    # ``discovery._row_from_openai_entry`` — the vendor doc says ``max``, the
    # listing says ``high``, and measured omission settles neither).
    assert spec.reasoning_effort is None
    assert spec.reasoning_default_effort is None


def test_the_chosen_level_reaches_the_chat_completions_body() -> None:
    """``/effort low`` and ``/effort max`` must be what the provider reads, and
    ``auto`` must omit the key entirely rather than send a default."""
    bodies = {
        level: _body(_kimi_spec().model_copy(update={"reasoning_effort": level}))
        for level in ("low", "max", None)
    }

    assert bodies["low"]["reasoning_effort"] == "low"
    assert bodies["max"]["reasoning_effort"] == "max"
    assert "reasoning_effort" not in bodies[None]


def test_the_image_part_survives_body_construction() -> None:
    """Body SHAPE only: ``_build_body`` carries the ``image_url`` part whether
    or not the spec says the model takes images — the strip is the session
    render gate at ``session.py`` (``if not self._model.supports_images``).

    Kept as a chain assertion for the wire shape the body builder emits
    (a real ``image_url`` data URI, not a bare text part); the capability GATE
    is pinned by ``_kimi_spec``'s spec assertion above, which uses the real
    ``unknown_model_info`` fallback and fails pre-fix.
    """
    body = _body(_kimi_spec())

    content = body["messages"][-1]["content"]  # type: ignore[index]
    images = [part for part in content if part.get("type") == "image_url"]
    assert len(images) == 1
    assert images[0]["image_url"]["url"].startswith("data:image/png;base64,")
