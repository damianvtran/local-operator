"""The ``generate_image`` tool: gate, args, approval text, result, cancel.

The result half runs the REAL ``cache_media`` against an isolated store (the
lane-D fixture pattern), because the contract being verified is exactly that
the tool and the attachment store compose — a monkeypatched ``cache_media``
would prove the shape of a call, not that bytes land where surfaces fetch
them.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from local_operator.harness.types import AttachmentContent, TextContent, ToolContext
from local_operator.imagegen import ImageAttempt, ImageOutcome, ImageRoute, MediaAsset
from local_operator.imagegen import cascade as image_cascade
from local_operator.imagegen import rungs as image_rungs
from local_operator.session.attachments import AttachmentStore
from local_operator.tools import image_tool

#: A 73-byte 1x1 PNG (real bytes: cache_media's sniffer reads the header).
PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c626001000000ffff03000006000557bfabd4"
    "0000000049454e44ae426082"
)


@pytest.fixture(autouse=True)
def _isolate_attachments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every registration lands in a tmp store, never the real config root."""
    store_root = tmp_path / "attachments"
    store_root.mkdir()
    monkeypatch.setattr("local_operator.session.attachments.attachments_dir", lambda: store_root)
    return store_root


def _outcome(
    assets: tuple[MediaAsset, ...] | None = None,
    *,
    seed: int | None = 7,
    cost: float | None = 0.08,
) -> ImageOutcome:
    if assets is None:
        assets = (
            MediaAsset(
                data=PNG_1X1,
                content_type="image/png",
                source_url="https://img.test/one.png",
                width=1,
                height=1,
            ),
        )
    return ImageOutcome(
        assets=assets,
        route=ImageRoute.RADIENT,
        attempts=(ImageAttempt(route=ImageRoute.RADIENT, outcome="ok"),),
        model="flux/dev",
        prompt="a cat",
        seed=seed,
        generation_id="r1",
        cost_usd=cost,
    )


def _patch_cascade(monkeypatch: pytest.MonkeyPatch, fn) -> None:
    monkeypatch.setattr(image_tool, "run_image_cascade", fn)


# ---------------------------------------------------------------------------
# The gate and the tool object
# ---------------------------------------------------------------------------


def test_the_builder_gates_on_reachability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_tool.image_availability, "image_provider_reachable", lambda *a: False)
    assert image_tool.build_generate_image_tool(ToolContext(cwd=".")) is None

    monkeypatch.setattr(image_tool.image_availability, "image_provider_reachable", lambda *a: True)
    tool = image_tool.build_generate_image_tool(ToolContext(cwd="."))
    assert tool is not None
    assert tool.name == "generate_image"
    assert tool.label == "Generate image"
    assert tool.approval_tier == "write"
    assert tool.interruptible is True
    # The wire description is pinned whole (design §2.3): 186 chars, and its
    # FIRST sentence is what the classification roster may quote.
    assert len(tool.description) == 186
    first = tool.description.split(". ", 1)[0] + "."
    assert len(first) == 117
    assert tool.describe_approval is not None


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


def test_params_forbid_extra_and_bound_their_ranges() -> None:
    model = image_tool.GenerateImageParams
    with pytest.raises(Exception):
        model.model_validate({"prompt": "x", "surprise": 1})
    with pytest.raises(Exception):
        model.model_validate({"prompt": "x", "num_images": 5})
    with pytest.raises(Exception):
        model.model_validate({"prompt": "x", "num_images": 0})
    with pytest.raises(Exception):
        model.model_validate({"prompt": "x", "strength": 1.5})
    with pytest.raises(Exception):
        model.model_validate({"prompt": "x", "image_size": "huge"})
    parsed = model.model_validate({"prompt": "x"})
    assert parsed.image_size == "square_hd"
    assert parsed.num_images == 1


# ---------------------------------------------------------------------------
# The approval sentence (D8: provider/quantity/size before the spend)
# ---------------------------------------------------------------------------


def test_approval_text_names_provider_quantity_and_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(image_tool, "_preferred_route_label", lambda: "Radient")
    text = image_tool._describe_generate_image_approval(
        {"prompt": "a cat", "num_images": 2, "image_size": "portrait_16_9"}, "."
    )
    assert text == (
        "Generate 2 images at portrait 16:9 via Radient — a paid provider call on your account."
    )

    single = image_tool._describe_generate_image_approval({"prompt": "a cat"}, ".")
    assert "Generate 1 image at square HD" in single

    # D1: the edit path carries the quantity too, and D2's map applies here.
    edited = image_tool._describe_generate_image_approval(
        {"prompt": "make it night", "source_image_path": "/tmp/in.png", "num_images": 2},
        ".",
    )
    assert edited == (
        "Edit 2 images (/tmp/in.png) at square HD via Radient — "
        "a paid provider call on your account."
    )

    edited_one = image_tool._describe_generate_image_approval(
        {"prompt": "make it night", "source_image_path": "/tmp/in.png"}, "."
    )
    assert edited_one.startswith("Edit 1 image (/tmp/in.png) at square HD")


def test_the_size_display_map_covers_every_accepted_token_and_falls_back_raw() -> None:
    """D2's map: every wire enum value has a prompt-facing spelling, and an
    unmapped token passes through raw rather than failing or blanking."""
    assert {token for token in image_tool.IMAGE_SIZE_VALUES} <= set(image_tool._SIZE_DISPLAY)
    assert image_tool._display_size("square_hd") == "square HD"
    assert image_tool._display_size("landscape_16_9") == "landscape 16:9"
    assert image_tool._display_size("weird_future_token") == "weird_future_token"


# ---------------------------------------------------------------------------
# Execution: result shape (the lane-D contract)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_registers_the_attachment_and_caption_first(
    monkeypatch: pytest.MonkeyPatch, _isolate_attachments: Path
) -> None:
    async def fake_cascade(**kwargs):
        return _outcome()

    _patch_cascade(monkeypatch, fake_cascade)
    result = await image_tool.execute_generate_image(
        "call-1", {"prompt": "a cat", "seed": 7, "image_size": "square_hd"}, None, None, None
    )

    assert result.is_error is False
    caption = result.content[0]
    assert isinstance(caption, TextContent)
    assert "Generated 1 image with Radient (flux/dev)" in caption.text
    assert "seed 7" in caption.text
    assert "Cost $0.08." in caption.text

    block = result.content[1]
    assert isinstance(block, AttachmentContent)
    assert block.kind == "image"
    assert len(block.attachment or "") == 32
    # The bytes are IN the store the surfaces fetch from.
    stored = AttachmentStore().get_bytes(block.attachment or "")
    assert stored is not None and stored[0] == PNG_1X1
    assert (_isolate_attachments / f"{block.attachment}.bin").exists()

    details = result.details or {}
    assert details["provider"] == "radient"
    assert details["model"] == "flux/dev"
    assert details["generation_id"] == "r1"
    assert details["cost_usd"] == 0.08
    assert details["attempts"] == [
        {"route": "radient", "outcome": "ok", "reason_class": "", "message": ""}
    ]


@pytest.mark.asyncio
async def test_partial_registration_failure_is_a_caption_note(
    monkeypatch: pytest.MonkeyPatch, _isolate_attachments: Path
) -> None:
    assets = (
        MediaAsset(data=PNG_1X1, content_type="image/png", source_url="https://img.test/1.png"),
        MediaAsset(data=b"", content_type="image/png", source_url="https://img.test/2.png"),
    )

    async def fake_cascade(**kwargs):
        return _outcome(assets)

    _patch_cascade(monkeypatch, fake_cascade)
    result = await image_tool.execute_generate_image(
        "call-1", {"prompt": "a cat"}, None, None, None
    )

    # An empty payload is one of cache_media's documented refusals (None on
    # empty bytes) — the tool must note it, not raise and not claim it.
    assert result.is_error is False
    caption = result.content[0]
    assert isinstance(caption, TextContent)
    assert "1 of 2 failed to attach" in caption.text
    assert "https://img.test/2.png" in caption.text
    assert len([b for b in result.content if isinstance(b, AttachmentContent)]) == 1


@pytest.mark.asyncio
async def test_every_registration_failing_is_an_error_naming_sources(
    monkeypatch: pytest.MonkeyPatch, _isolate_attachments: Path
) -> None:
    async def fake_cascade(**kwargs):
        return _outcome(
            (MediaAsset(data=b"", content_type="image/png", source_url="https://img.test/x.png"),)
        )

    _patch_cascade(monkeypatch, fake_cascade)
    result = await image_tool.execute_generate_image(
        "call-1", {"prompt": "a cat"}, None, None, None
    )

    assert result.is_error is True
    text = result.content[0].text  # type: ignore[union-attr]
    assert "none" in text and "https://img.test/x.png" in text


# ---------------------------------------------------------------------------
# Execution: failure and cancellation paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unavailable_carries_the_attempt_list(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_cascade(**kwargs):
        raise image_cascade.ImageGenerationUnavailable(
            "Image generation failed on every available provider:\n- Radient: boom",
            attempts=(
                ImageAttempt(
                    route=ImageRoute.RADIENT,
                    outcome="failed",
                    reason_class="upstream",
                    message="boom",
                ),
            ),
        )

    _patch_cascade(monkeypatch, fake_cascade)
    result = await image_tool.execute_generate_image(
        "call-1", {"prompt": "a cat"}, None, None, None
    )

    assert result.is_error is True
    assert "Radient: boom" in result.content[0].text  # type: ignore[union-attr]
    assert (result.details or {})["attempts"][0]["reason_class"] == "upstream"


@pytest.mark.asyncio
async def test_the_no_cancellation_race_returns_a_clean_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    async def fake_cascade(**kwargs):
        kwargs["handle"].provider = ImageRoute.RADIENT
        kwargs["handle"].request_id = "r1"
        kwargs["handle"].model = "flux/dev"
        raise image_cascade.ImageGenerationCancelled()

    async def fake_cancel(handle):
        calls.append(handle)
        return "cancelled"

    _patch_cascade(monkeypatch, fake_cascade)
    monkeypatch.setattr(image_rungs, "best_effort_cancel", fake_cancel)

    result = await image_tool.execute_generate_image(
        "call-1", {"prompt": "a cat"}, None, None, None
    )

    assert result.is_error is True
    assert calls, "the provider job must be best-effort cancelled"
    text = result.content[0].text  # type: ignore[union-attr]
    assert text == (
        "Generation cancelled before completion (provider job cancelled). "
        "Re-run to start a new generation."
    )
    assert (result.details or {})["cancel_handle"] == {
        "provider": "radient",
        "request_id": "r1",
        "model": "flux/dev",
    }


@pytest.mark.asyncio
async def test_task_cancellation_cancels_provider_side_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    async def fake_cascade(**kwargs):
        raise asyncio.CancelledError()

    async def fake_cancel(handle):
        calls.append(handle)
        return "none"

    _patch_cascade(monkeypatch, fake_cascade)
    monkeypatch.setattr(image_rungs, "best_effort_cancel", fake_cancel)

    with pytest.raises(asyncio.CancelledError):
        await image_tool.execute_generate_image("call-1", {"prompt": "a cat"}, None, None, None)
    assert calls, "an abort must attempt the provider-side cancel"


# ---------------------------------------------------------------------------
# Execution: local validation and img2img wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_strength_without_a_source_image_is_rejected() -> None:
    result = await image_tool.execute_generate_image(
        "call-1", {"prompt": "a cat", "strength": 0.5}, None, None, None
    )
    assert result.is_error is True
    text = result.content[0].text  # type: ignore[union-attr]
    assert "strength requires source_image_path" in text


@pytest.mark.asyncio
async def test_a_missing_source_image_is_an_argument_fault() -> None:
    result = await image_tool.execute_generate_image(
        "call-1", {"prompt": "edit this", "source_image_path": "/nope/never.png"}, None, None, None
    )
    assert result.is_error is True
    assert "not a readable image file" in result.content[0].text  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_a_real_source_image_travels_as_a_data_uri(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "in.png"
    source.write_bytes(PNG_1X1)
    seen: dict[str, object] = {}

    async def fake_cascade(**kwargs):
        seen.update(kwargs)
        return _outcome()

    _patch_cascade(monkeypatch, fake_cascade)
    result = await image_tool.execute_generate_image(
        "call-1",
        {"prompt": "make it night", "source_image_path": str(source), "strength": 0.4},
        None,
        None,
        ToolContext(cwd=str(tmp_path)),
    )

    assert result.is_error is False
    assert str(seen["source_url"]).startswith("data:image/png;base64,")
    assert seen["strength"] == 0.4
    assert (result.details or {})["source_image_path"] == str(source)
    assert (result.details or {})["strength"] == 0.4


@pytest.mark.asyncio
async def test_progress_updates_map_onto_agent_tool_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    updates: list[object] = []

    async def fake_cascade(**kwargs):
        emit = kwargs["emit"]
        assert emit is not None
        emit(
            "Generating via Radient (flux/dev): queued, #2 — 14s",
            {
                "tool_name": "generate_image",
                "stage": "queued",
                "provider": "radient",
                "model": "flux/dev",
                "elapsed_s": 14,
                "queue_position": 2,
                "num_images": 1,
            },
        )
        return _outcome()

    _patch_cascade(monkeypatch, fake_cascade)
    await image_tool.execute_generate_image(
        "call-1", {"prompt": "a cat"}, None, updates.append, None
    )

    assert len(updates) == 1
    update = updates[0]
    text_part = update.content[0]  # type: ignore[union-attr]
    assert text_part.text == "Generating via Radient (flux/dev): queued, #2 — 14s"
    assert update.details["stage"] == "queued"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_the_cancel_receipt_states_not_cancelled_reasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_cascade(**kwargs):
        kwargs["handle"].provider = ImageRoute.FAL
        kwargs["handle"].request_id = "r9"
        raise image_cascade.ImageGenerationCancelled()

    async def fake_cancel(handle):
        return "already_completed"

    _patch_cascade(monkeypatch, fake_cascade)
    monkeypatch.setattr(image_rungs, "best_effort_cancel", fake_cancel)

    result = await image_tool.execute_generate_image(
        "call-1", {"prompt": "a cat"}, None, None, None
    )
    text = result.content[0].text  # type: ignore[union-attr]
    assert "not cancelled — it had already completed" in text
