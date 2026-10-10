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

from local_operator.artifacts import BillingBasis, CostSource
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
    cost_source: CostSource | None = None,
    billing_basis: BillingBasis | None = None,
    cost_provenance: str | None = None,
    route: ImageRoute = ImageRoute.RADIENT,
    usage_record_id: str | None = None,
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
        route=route,
        attempts=(ImageAttempt(route=route, outcome="ok"),),
        model="flux/dev",
        prompt="a cat",
        seed=seed,
        generation_id="r1",
        cost_usd=cost,
        cost_source=cost_source,
        billing_basis=billing_basis,
        cost_provenance=cost_provenance,
        usage_record_id=usage_record_id,
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
    # The wire description is pinned whole (media wave-2, design D9): 160
    # chars, and its FIRST sentence is what the classification roster may
    # quote. The provider list left the wire for the guide; the pins moved
    # with the deliberate rewrite.
    assert len(tool.description) == 160
    first = tool.description.split(". ", 1)[0] + "."
    assert len(first) == 91
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
    monkeypatch.setattr(image_tool, "_preferred_route_label", lambda **_: "Radient")
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
async def test_details_carry_cost_source_billing_basis_and_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_cascade(**kwargs):
        return _outcome(
            cost=0.053,
            cost_source="subscription",
            billing_basis="subscription-api-equivalent",
            cost_provenance="API-equivalent, not billed: doc 2026-10-09",
        )

    _patch_cascade(monkeypatch, fake_cascade)
    result = await image_tool.execute_generate_image(
        "call-1", {"prompt": "a cat"}, None, None, None
    )

    details = result.details or {}
    assert details["cost_usd"] == 0.053
    assert details["cost_source"] == "subscription"
    assert details["billing_basis"] == "subscription-api-equivalent"
    assert details["cost_provenance"] == "API-equivalent, not billed: doc 2026-10-09"
    # Caption unchanged this round: amount only, no basis suffix.
    caption = result.content[0]
    assert isinstance(caption, TextContent)
    assert "Cost $0.053." in caption.text
    assert "equivalent" not in caption.text


@pytest.mark.asyncio
async def test_details_leave_basis_none_when_there_is_no_amount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_cascade(**kwargs):
        return _outcome(cost=None)

    _patch_cascade(monkeypatch, fake_cascade)
    result = await image_tool.execute_generate_image(
        "call-1", {"prompt": "a cat"}, None, None, None
    )

    details = result.details or {}
    assert details["billing_basis"] is None
    assert details["cost_source"] is None


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

    updates: list[object] = []
    result = await image_tool.execute_generate_image(
        "call-1", {"prompt": "a cat"}, None, updates.append, None
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
    assert (result.details or {})["stage"] == "cancelled"
    assert "error_type" not in (result.details or {}), "a plain cancel is not a failure"
    # The cancel-phase updates bracket the best-effort cancel; the second
    # mirrors the receipt exactly (one wording, no drift).
    assert [update.details["stage"] for update in updates] == [  # type: ignore[union-attr]
        "cancelling",
        "cancelled",
    ]
    assert updates[1].content[0].text == text  # type: ignore[union-attr]
    assert updates[1].details["provider"] == "radient"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_task_cancellation_cancels_provider_side_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    updates: list[object] = []

    async def fake_cascade(**kwargs):
        raise asyncio.CancelledError()

    async def fake_cancel(handle):
        calls.append(handle)
        return "none"

    _patch_cascade(monkeypatch, fake_cascade)
    monkeypatch.setattr(image_rungs, "best_effort_cancel", fake_cancel)

    with pytest.raises(asyncio.CancelledError):
        await image_tool.execute_generate_image(
            "call-1", {"prompt": "a cat"}, None, updates.append, None
        )
    assert calls, "an abort must attempt the provider-side cancel"
    # The cancel-phase updates emit from inside the CancelledError handler —
    # through the guarded emitter, so nothing can replace the cancellation.
    assert [update.details["stage"] for update in updates] == [  # type: ignore[union-attr]
        "cancelling",
        "cancelled",
    ]
    assert updates[0].details["provider"] is None  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Execution: local validation and img2img wiring
# ---------------------------------------------------------------------------


def test_the_emitter_swallows_a_raising_on_update() -> None:
    """The tool-level guard (reviewer round-1 pin): progress never rides control flow.

    Same contract as the rungs' own wrapper — a raising ``on_update`` is
    swallowed, because the terminal cancel-phase lines emit from inside
    cancellation handlers and must never replace the ``CancelledError``.
    """

    def raiser(update: object) -> None:
        raise RuntimeError("on_update exploded")

    emit = image_tool._progress_emitter(raiser)
    assert emit is not None
    emit("line", {"stage": "queued"})  # must not raise
    assert image_tool._progress_emitter(None) is None


@pytest.mark.asyncio
async def test_the_abort_receipt_survives_a_raising_emitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising on_update cannot break the abort-path cancellation (pin)."""

    async def fake_cascade(**kwargs):
        kwargs["handle"].provider = ImageRoute.RADIENT
        kwargs["handle"].request_id = "r1"
        raise image_cascade.ImageGenerationCancelled()

    async def fake_cancel(handle):
        return "none"

    _patch_cascade(monkeypatch, fake_cascade)
    monkeypatch.setattr(image_rungs, "best_effort_cancel", fake_cancel)

    def raiser(update: object) -> None:
        raise RuntimeError("on_update exploded")

    result = await image_tool.execute_generate_image(
        "call-1", {"prompt": "a cat"}, None, raiser, None
    )
    assert result.is_error is True
    assert (result.details or {})["stage"] == "cancelled"
    assert "error_type" not in (result.details or {})


@pytest.mark.asyncio
async def test_task_cancellation_survives_a_raising_emitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising on_update cannot replace the Esc-path CancelledError (pin)."""
    calls: list[object] = []

    async def fake_cascade(**kwargs):
        raise asyncio.CancelledError()

    async def fake_cancel(handle):
        calls.append(handle)
        return "none"

    _patch_cascade(monkeypatch, fake_cascade)
    monkeypatch.setattr(image_rungs, "best_effort_cancel", fake_cancel)

    def raiser(update: object) -> None:
        raise RuntimeError("on_update exploded")

    with pytest.raises(asyncio.CancelledError):
        await image_tool.execute_generate_image("call-1", {"prompt": "a cat"}, None, raiser, None)
    assert calls, "the cancel flow continued past the raising emitter"


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

    assert len(updates) == 2, "the synthetic update, then the tool's own completed"
    update, completed = updates
    text_part = update.content[0]  # type: ignore[union-attr]
    assert text_part.text == "Generating via Radient (flux/dev): queued, #2 — 14s"
    assert update.details["stage"] == "queued"  # type: ignore[union-attr]
    # The tool's terminal update closes the canonical lifecycle (Q7 wire scope)
    # and carries the canonical key set like every other update.
    assert completed.details["stage"] == "completed"  # type: ignore[union-attr]
    assert completed.content[0].text == (  # type: ignore[union-attr]
        "Generation complete — 1 image(s) via Radient (flux/dev)."
    )
    canonical = {"stage", "queue_position", "progress_fraction", "log_lines", "error", "error_type"}
    assert canonical <= set(completed.details)  # type: ignore[union-attr]
    assert completed.details["progress_fraction"] is None  # type: ignore[union-attr]


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
    details = result.details or {}
    # The cancel conflict is NOT a failure: the surfaces' frozen branch keys
    # on this pair (Q7), and the receipt sentence rides beside it verbatim.
    assert details["error_type"] == "media_already_completed"
    assert details["stage"] == "cancelled"
    assert details["error"] == (
        "The generation had already completed when the cancel arrived; " "its result was discarded."
    )
    assert "not cancelled — it had already completed" in text


# ---------------------------------------------------------------------------
# Source by reference (media wave-2 edit lane)
# ---------------------------------------------------------------------------


def test_approval_text_abbreviates_an_attachment_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(image_tool, "_preferred_route_label", lambda **_: "Radient")
    text = image_tool._describe_generate_image_approval(
        {"prompt": "make it night", "source_attachment": "ab12" + "0" * 28}, "."
    )
    assert text == (
        "Edit 1 image (attachment ab12…) at square HD via Radient — "
        "a paid provider call on your account."
    )


@pytest.mark.asyncio
async def test_a_source_attachment_resolves_through_the_session_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = AttachmentStore().put_bytes(PNG_1X1, "image/png")
    assert ref is not None
    seen: dict[str, object] = {}

    async def fake_cascade(**kwargs):
        seen.update(kwargs)
        return _outcome()

    _patch_cascade(monkeypatch, fake_cascade)
    result = await image_tool.execute_generate_image(
        "call-1",
        {"prompt": "make it night", "source_attachment": ref.digest, "strength": 0.4},
        None,
        None,
        None,
    )

    assert result.is_error is False
    assert str(seen["source_url"]).startswith("data:image/png;base64,")
    assert seen["strength"] == 0.4
    details = result.details or {}
    assert details["source_attachment"] == ref.digest
    assert details["strength"] == 0.4
    assert "source_image_path" not in details


@pytest.mark.asyncio
async def test_a_bad_attachment_digest_is_refused_before_touching_the_store() -> None:
    result = await image_tool.execute_generate_image(
        "call-1", {"prompt": "x", "source_attachment": "../etc/passwd"}, None, None, None
    )
    assert result.is_error is True
    assert "32-character hex digest" in result.content[0].text  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_an_unknown_attachment_digest_is_an_argument_fault() -> None:
    result = await image_tool.execute_generate_image(
        "call-1", {"prompt": "x", "source_attachment": "0" * 32}, None, None, None
    )
    assert result.is_error is True
    assert "was not found" in result.content[0].text  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_both_sources_at_once_are_refused() -> None:
    result = await image_tool.execute_generate_image(
        "call-1",
        {"prompt": "x", "source_image_path": "/a.png", "source_attachment": "a" * 32},
        None,
        None,
        None,
    )
    assert result.is_error is True
    text = result.content[0].text  # type: ignore[union-attr]
    assert "exactly one of source_image_path or source_attachment" in text


@pytest.mark.asyncio
async def test_an_ignored_strength_is_recorded_not_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = AttachmentStore().put_bytes(PNG_1X1, "image/png")
    assert ref is not None

    async def fake_cascade(**kwargs):
        return _outcome(route=ImageRoute.GOOGLE)

    _patch_cascade(monkeypatch, fake_cascade)
    result = await image_tool.execute_generate_image(
        "call-1",
        {"prompt": "make it night", "source_attachment": ref.digest, "strength": 0.4},
        None,
        None,
        None,
    )

    details = result.details or {}
    assert details["strength_ignored"] is True
    assert "Strength 0.4 was ignored" in result.content[0].text  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_a_strength_taking_provider_records_no_ignored_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "in.png"
    source.write_bytes(PNG_1X1)

    async def fake_cascade(**kwargs):
        return _outcome(route=ImageRoute.FAL)

    _patch_cascade(monkeypatch, fake_cascade)
    result = await image_tool.execute_generate_image(
        "call-1",
        {"prompt": "x", "source_image_path": str(source), "strength": 0.4},
        None,
        None,
        ToolContext(cwd=str(tmp_path)),
    )

    details = result.details or {}
    assert "strength_ignored" not in details
    assert "was ignored" not in result.content[0].text  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_a_usage_record_id_reaches_details_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_cascade(**kwargs):
        return _outcome(usage_record_id="ur-9")

    _patch_cascade(monkeypatch, fake_cascade)
    result = await image_tool.execute_generate_image("call-1", {"prompt": "x"}, None, None, None)
    assert (result.details or {})["usage_record_id"] == "ur-9"

    async def fake_plain(**kwargs):
        return _outcome()

    _patch_cascade(monkeypatch, fake_plain)
    plain = await image_tool.execute_generate_image("call-1", {"prompt": "x"}, None, None, None)
    assert "usage_record_id" not in (plain.details or {})


def test_the_approval_label_skips_edit_incapable_rungs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The same filter the cascade applies: an edit prompt must never name a
    # provider that cannot run it (Radient is available here, and skipped).
    monkeypatch.setattr(image_tool.image_availability, "radient_available", lambda *a: True)
    monkeypatch.setattr(image_tool.image_availability, "fal_key", lambda *a: "fk")

    assert image_tool._preferred_route_label() == "Radient"
    assert image_tool._preferred_route_label(editing=True) == "FAL"
