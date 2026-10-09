"""The ``generate_image`` agent tool — one createIf-gated image-generation tool.

**Why ONE tool** (design D1): the footprint ladder says a new core surface is
the last rung, and a second tool (img2img) would duplicate the whole schema.
``source_image_path`` present ⇒ image-to-image; absent ⇒ text-to-image. The
legacy tree shipped two tools; their params collapse cleanly into one model,
and the modern surface precedent is one params model per capability.

**Why this tool exists at all** (rather than a skill + bash): generation needs
structured params (size enum, count cap, seed), a live progress row and a
result that lands in the session's attachment store — none of which a shell
command can carry. It is rung 3: the builder returns ``None`` unless a
provider the harness can reach exists (D9), so sessions with no image
credential pay zero schema.

**Why write tier** (design D8): a generation spends real money on a third-party
account ($0.01–$0.50+/image class) and its product is billable. Default mode
is "ask", so the approval prompt is the ONE place cost is shown before spend —
which is what :func:`_describe_generate_image_approval` is for. The tier
records intent, not protection: ``auto``/``yolo`` modes and hosts without a
gate skip the prompt — a reviewable knob, not an oversight.

**Why PUBLISHED and not deferred** (design D3 + the manager's constraint): the
deferral machinery stays untouched in this PR. Measured on #2066, deferring a
brand-new capability whose description carries the "when to use it" text
regressed adoption; the candidate purpose phrase for a FUTURE data-driven
deferral is recorded here so the decision can be revisited with evidence:
``"generate an image via configured providers"`` (42 chars).

**Why nothing registers with ``group_reaper``** (design D7): that module reaps
detached PROCESS GROUPS spawned by ``bash``, keyed on owner liveness. This
tool spawns no process and no orphanable group — its work is in-process HTTP
cancelled through the standard loom machinery (``asyncio.CancelledError`` at
the socket, then a bounded best-effort provider cancel). Registering a thread
or a "generation" there would be a second, parallel reap concept with nothing
to kill on owner death; a SIGKILLed runtime leaves a provider-side job that
completes uncollected, identical to any client disconnect.

**Result shape** (lane D's frozen contract): ``ToolResult(content=[caption,
*AttachmentContent])`` — the caption FIRST and self-sufficient (provider
dispatch skips artifact blocks), bytes registered exactly once through
``session.attachments.cache_media``, provenance in ``details``. No
``ImageContent`` co-attach in v1 (the model rarely needs the pixels, and
co-attaching would ride base64 into the next request); a ``cache_media``
refusal degrades to a caption note, never a raise.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    AgentToolUpdate,
    InvalidToolArgumentsError,
    TextContent,
    ToolContext,
    ToolResult,
)
from local_operator.imagegen import availability as image_availability
from local_operator.imagegen import rungs as image_rungs
from local_operator.imagegen.cascade import (
    RUNG_LABELS,
    ImageGenerationCancelled,
    ImageGenerationUnavailable,
    run_image_cascade,
)
from local_operator.media import sniff_image_file
from local_operator.tools.builtin import _guard

logger = logging.getLogger(__name__)

#: Wire description (media wave-2, design D9/§2.3), 160 chars, first sentence
#: 91 — the classification roster reads ONLY the first sentence
#: (``_tool_candidate_description`` bounds it at 160), so the capability
#: statement must fit there and the pointers to the deeper docs follow it.
#: The provider enumeration moved OUT of the wire (four new provider rungs
#: made the parenthetical wrong, and the requirement forbids a list here):
#: providers live in ``guide://image-generation``, which costs no schema.
_DESCRIPTION = (
    "Generate or edit an image via the configured provider; the result is attached for the user. "
    "Docs: `tool://generate_image`; playbook: `guide://image-generation`."
)

#: The six FAL-shaped size values (design §2.2/§9): accepted by Radient's
#: passthrough verbatim, mapped onto OpenAI's size strings in the rung. Spelled
#: in the Literal below as well — pydantic needs the annotation literal for the
#: schema — and pinned by a unit test against this tuple so the two cannot
#: drift.
IMAGE_SIZE_VALUES = (
    "square_hd",
    "square",
    "portrait_4_3",
    "portrait_16_9",
    "landscape_4_3",
    "landscape_16_9",
)


class GenerateImageParams(BaseModel):
    """Arguments for the ``generate_image`` tool."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(description="Text description; with source_image_path, the edit to apply.")
    source_image_path: str | None = Field(
        default=None,
        description="Local image to edit (image-to-image); uploaded as a data URI.",
    )
    strength: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Edit strength 0..1; only with source_image_path.",
    )
    image_size: Literal[
        "square_hd",
        "square",
        "portrait_4_3",
        "portrait_16_9",
        "landscape_4_3",
        "landscape_16_9",
    ] = Field(default="square_hd", description="Output size; mapped per provider.")
    num_images: int = Field(default=1, ge=1, le=4, description="How many images to generate (1-4).")
    seed: int | None = Field(default=None, description="Seed for reproducibility where supported.")
    model: str | None = Field(
        default=None, description="Provider model id; its default when omitted."
    )


def build_generate_image_tool(context: ToolContext) -> AgentTool | None:
    """createIf builder: exists only where an image provider is reachable (D9).

    The gate is SYNC and socket-free (``imagegen.availability``: sqlite rows,
    the encrypted store, the environment) so no session-build path opens a
    network connection, and every probe failure reads as "not available".
    Sessions built before a login land the tool at the NEXT session — the gate
    is a snapshot by design, not a per-turn scan.
    """
    del context  # gating is on the machine's credentials, not session state
    from local_operator.paths import config_dir

    if not image_availability.image_provider_reachable(config_dir()):
        return None
    return AgentTool(
        name="generate_image",
        label="Generate image",
        description=_DESCRIPTION,
        parameters=GenerateImageParams.model_json_schema(),
        # write tier: the generation spends real money on a third-party
        # account and its product is billable (see the module docstring).
        approval_tier="write",
        # Generations are independent; two calls can run beside each other
        # without sharing anything but the credential reads.
        concurrency="shared",
        # A 30-60 s generation should yield to a redirect the way `bash` does
        # (design §4.2): steering cancels the wait, the CancelledError handler
        # best-effort cancels the provider job, and the loop pairs its own
        # synthetic receipt.
        interruptible=True,
        describe_approval=_describe_generate_image_approval,
        execute=execute_generate_image,
    )


def _preferred_route_label() -> str:
    """The cascade's first available route, as a human label for the prompt.

    A read of the same sync probes the gate uses; "the configured provider"
    when nothing answers (the tool would not have been built, but a describer
    must never fail a call — it is read by the approval renderer).
    """
    from local_operator.paths import config_dir

    cfg = config_dir()
    try:
        if image_availability.radient_available(cfg):
            return "Radient"
        if image_availability.fal_key(cfg):
            return "FAL"
        if image_availability.openai_images_key(cfg):
            return "OpenAI"
        if image_availability.openai_subscription_grant(cfg):
            return "ChatGPT plan"
        if image_availability.google_key(cfg):
            return "Google"
        if image_availability.xai_available(cfg):
            return "xAI"
    except Exception:  # noqa: BLE001 - a describer must never fail a call
        logger.debug("image availability read failed for approval text", exc_info=True)
    return "the configured provider"


#: The consent copy's humanized size tokens (design round 1, D2): the wire enum
#: is FAL-shaped ("square_hd"), which reads as jargon in the ONE prompt a
#: person answers under time pressure. Unmapped values fall back to the raw
#: token — a describer must never fail, or blank, a value it cannot prettify.
_SIZE_DISPLAY = {
    "square_hd": "square HD",
    "square": "square",
    "portrait_4_3": "portrait 4:3",
    "portrait_16_9": "portrait 16:9",
    "landscape_4_3": "landscape 4:3",
    "landscape_16_9": "landscape 16:9",
}


def _display_size(token: str) -> str:
    """The prompt-facing spelling of a size token; raw fallback (see above)."""
    return _SIZE_DISPLAY.get(token, token)


def _describe_generate_image_approval(args: dict[str, Any], cwd: str) -> str:
    """What the approval prompt says: provider, quantity, size, and the spend.

    Written for the person answering the prompt under time pressure: the
    decision-relevant facts are the QUANTITY (each image bills), the SIZE and
    that money leaves their account — not the prompt text (already on the
    card) nor the raw JSON the fallback would dump. Quantity is part of BOTH
    shapes (design round 1, D1): the edit path spends per image exactly as the
    generate path does, and a consent that omits the count hides the
    multiplier the person is approving.
    """
    del cwd  # the spend is account-level; no path decides it
    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return "[unresolvable] generate_image without a prompt"
    num = args.get("num_images")
    num = num if isinstance(num, int) and not isinstance(num, bool) and num > 0 else 1
    size = args.get("image_size")
    size = size if isinstance(size, str) and size else "square_hd"
    source = args.get("source_image_path")
    provider = _preferred_route_label()
    what = (
        f"Edit {num} image{'s' if num != 1 else ''} ({source})"
        if isinstance(source, str) and source
        else f"Generate {num} image{'s' if num != 1 else ''}"
    )
    return (
        f"{what} at {_display_size(size)} via {provider} — " "a paid provider call on your account."
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


#: The receipt phrase per best-effort-cancel token (design §4.3's
#: "provider job <cancelled|not cancelled, reason>"). A token this map does
#: not know is surfaced verbatim — a receipt must never invent an outcome.
_CANCEL_PHRASES = {
    "cancelled": "cancelled",
    "already_completed": "not cancelled — it had already completed; its result was discarded",
    "not_found": "not cancelled — the provider no longer knows the job",
    "none": "not cancelled — nothing was left to cancel",
    "timeout": "not cancelled — the cancel request timed out",
    "failed": "not cancelled — the cancel request failed",
    "abandoned": "not cancelled — a second stop abandoned the cancel attempt",
}


#: The ``error`` sentence beside ``error_type: media_already_completed`` —
#: the surfaces' frozen cancel-conflict branch (their words for it are
#: "already finished", never an error's ink). A platform sentence; the
#: provider's prose never rides here.
_CANCEL_CONFLICT_SENTENCE = (
    "The generation had already completed when the cancel arrived; its result was discarded."
)


def _progress_emitter(
    on_update: Callable[[AgentToolUpdate], None] | None,
) -> image_rungs.ProgressFn | None:
    """Adapt the cascade's ``emit(text, details)`` onto ``AgentToolUpdate``.

    The emitter never raises (same contract as the rungs' own wrapper): a
    progress line is presentation, and — because the terminal ``cancelling``/
    ``[redacted]`` lines emit from inside cancellation handlers — an
    emitter failure must never replace the ``CancelledError`` being handled.
    Updates are LIVE-ONLY by design (design §4.4): the transcript receives the
    final result, and no surface should expect historical progress rows.
    """
    if on_update is None:
        return None

    def emit(text: str, details: dict[str, Any]) -> None:
        try:
            on_update(AgentToolUpdate(content=[TextContent(text=text)], details=dict(details)))
        except Exception:  # noqa: BLE001 - progress is presentation, never control flow
            logger.debug("image progress emitter raised; continuing", exc_info=True)

    return emit


def _emit_cancel_stage(
    progress: image_rungs.ProgressFn | None,
    handle: image_rungs.CancelHandle,
    stage: str,
    text: str,
) -> None:
    """A cancel-phase update (``cancelling`` → ``[redacted]``).

    ``provider``/``model`` come off the handle: a cancel can land before any
    submit (both absent — honest nulls) or after one (both set).
    """
    if progress is None:
        return
    progress(
        text,
        image_rungs.progress_details(
            stage=stage,
            provider=str(handle.provider) if handle.provider is not None else None,
            model=handle.model,
        ),
    )


def _cancel_handle_details(handle: image_rungs.CancelHandle) -> dict[str, Any] | None:
    """``{provider, request_id, model?}`` — what a best-effort cancel used."""
    if handle.provider is None or handle.request_id is None:
        return None
    return {
        "provider": str(handle.provider),
        "request_id": handle.request_id,
        "model": handle.model,
    }


def _attempt_list(attempts: tuple[Any, ...]) -> list[dict[str, Any]]:
    """Attempt records as JSON-safe dicts (the lane-D ``details`` contract)."""
    return [
        {
            "route": str(attempt.route),
            "outcome": attempt.outcome,
            "reason_class": attempt.reason_class,
            "message": attempt.message,
        }
        for attempt in attempts
    ]


def _cancel_receipt(receipt: str) -> str:
    phrase = _CANCEL_PHRASES.get(receipt, receipt)
    return (
        f"Generation cancelled before completion (provider job {phrase}). "
        "Re-run to start a new generation."
    )


def _unavailable_text(exc: ImageGenerationUnavailable) -> str:
    text = str(exc).strip()
    if text:
        return text
    if exc.resolution is not None and exc.resolution.reason:
        return exc.resolution.reason
    return "Image generation is unavailable."


def _load_source_image(raw_path: str, cwd: str) -> str:
    """The ``data:`` URI for a local image, or an argument-shape refusal.

    The file is read and encoded HERE (design §2.2): the provider receives a
    data URI and the transcript carries only the path. A path that is not a
    readable image is a MODEL fault — ``InvalidToolArgumentsError`` — because
    the model can correct it, and the marker keeps it out of the
    execution-error bucket.
    """
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(cwd) / path
    info = sniff_image_file(str(path))
    if info is None:
        raise InvalidToolArgumentsError(
            f"source_image_path is not a readable image file: {raw_path}"
        )
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise InvalidToolArgumentsError(
            f"source_image_path could not be read ({exc.__class__.__name__}): {raw_path}"
        ) from exc
    return f"data:{info.mime_type};base64,{base64.b64encode(data).decode('ascii')}"


def _artifact_name(model: str, content_type: str, index: int) -> str:
    """A short human handle ("flux-dev-01.png"); NEVER a filesystem path."""
    slug = "".join(ch if ch.isalnum() else "-" for ch in (model or "image"))
    slug = "-".join(part for part in slug.split("-") if part)[:40] or "image"
    ext = mimetypes.guess_extension((content_type or "").split(";")[0].strip()) or ".png"
    return f"{slug}-{index:02d}{ext}"


def _dimensions_text(assets: tuple[Any, ...], fallback: str) -> str:
    """Distinct ``WxH`` pairs from the assets, else the requested size token."""
    seen: list[str] = []
    for asset in assets:
        if asset.width and asset.height:
            pair = f"{asset.width}x{asset.height}"
            if pair not in seen:
                seen.append(pair)
    return ", ".join(seen) if seen else fallback


#: Shown as the reason when a generation completed but NOTHING registered.
#: It must state the likely cause, because the model's next step (re-run vs
#: save the URLs by hand) depends on which half failed.
_REGISTER_FAILED_NOTE = "could not be registered in the session's attachment store"


@_guard("generate_image")
async def execute_generate_image(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Validate → resolve → cascade → the lane-D attachment result.

    Cancellation has two spellings and they are not the same path (design
    §4.2): the loop's task cancellation (Esc/steer) propagates untouched after
    a bounded best-effort provider cancel, while the no-cancellation race —
    the abort signal observed between polls — returns a clean receipt.
    """
    try:
        params = GenerateImageParams.model_validate(args)
    except ValidationError as exc:
        raise InvalidToolArgumentsError(str(exc)) from exc
    if params.strength is not None and params.source_image_path is None:
        raise InvalidToolArgumentsError("strength requires source_image_path.")

    cwd = (context.cwd if context is not None else None) or "."
    source_url: str | None = None
    if params.source_image_path:
        source_url = _load_source_image(params.source_image_path, cwd)

    from local_operator.paths import config_dir

    handle = image_rungs.CancelHandle()
    progress = _progress_emitter(on_update)
    try:
        outcome = await run_image_cascade(
            prompt=params.prompt,
            config_dir=config_dir(),
            source_url=source_url,
            strength=params.strength,
            image_size=params.image_size,
            num_images=params.num_images,
            seed=params.seed,
            model=params.model,
            signal=signal,
            handle=handle,
            emit=progress,
        )
    except ImageGenerationCancelled:
        _emit_cancel_stage(progress, handle, "cancelling", "Cancelling the generation…")
        receipt = await image_rungs.best_effort_cancel(handle)
        _emit_cancel_stage(progress, handle, "cancelled", _cancel_receipt(receipt))
        details: dict[str, Any] = {
            "cancel_handle": _cancel_handle_details(handle),
            "stage": "cancelled",
        }
        if receipt == "already_completed":
            # The cancel conflict is NOT a failure: the surfaces' frozen
            # branch renders it as "already finished" off this pair (Q7).
            details["error"] = _CANCEL_CONFLICT_SENTENCE
            details["error_type"] = "media_already_completed"
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="generate_image",
            is_error=True,
            content=[TextContent(text=_cancel_receipt(receipt))],
            details=details,
        )
    except asyncio.CancelledError:
        # The loop's cancellation. The best-effort cancel is bounded (5 s),
        # never raises and never masks the cancellation — a second stop
        # abandons it (see ``rungs.best_effort_cancel``). The terminal
        # updates go through the guarded emitter for the same reason.
        _emit_cancel_stage(progress, handle, "cancelling", "Cancelling the generation…")
        receipt = await image_rungs.best_effort_cancel(handle)
        _emit_cancel_stage(progress, handle, "cancelled", _cancel_receipt(receipt))
        raise
    except ImageGenerationUnavailable as exc:
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="generate_image",
            is_error=True,
            content=[TextContent(text=_unavailable_text(exc))],
            details={
                "attempts": _attempt_list(exc.attempts),
                "error": _unavailable_text(exc),
                # The last attempt's classification is the walk's final word;
                # ``attempts`` beside it carries the full story, so the two
                # cannot contradict.
                "error_type": exc.attempts[-1].reason_class if exc.attempts else None,
            },
        )

    if progress is not None:
        progress(
            f"Generation complete — {len(outcome.assets)} image(s) via "
            f"{RUNG_LABELS.get(outcome.route, str(outcome.route))} ({outcome.model}).",
            image_rungs.progress_details(
                stage="completed",
                provider=str(outcome.route),
                model=outcome.model,
                num_images=len(outcome.assets),
            ),
        )

    return _generated_result(tool_call_id, params, outcome, handle)


def _generated_result(
    tool_call_id: str,
    params: GenerateImageParams,
    outcome: Any,
    handle: image_rungs.CancelHandle,
) -> ToolResult:
    """Register every asset and shape the lane-D result (caption FIRST).

    Registration goes through ``cache_media`` exactly once per asset; a
    refusal is a caption note, never a raise — and if EVERY asset refuses,
    the result is an error naming the sources, because a caption claiming an
    attachment that does not exist is worse than an honest failure.
    """
    from local_operator.session.attachments import cache_media

    blocks: list[Any] = []
    failed: list[str] = []
    digests: list[str] = []
    for index, asset in enumerate(outcome.assets, start=1):
        block = cache_media(
            asset.data,
            asset.content_type or "application/octet-stream",
            kind="image",
            name=_artifact_name(outcome.model, asset.content_type, index),
            source_url=asset.source_url or None,
            width=asset.width,
            height=asset.height,
            duration_s=asset.duration_s,
        )
        if block is None:
            failed.append(asset.source_url or "(no source url)")
            continue
        blocks.append(block)
        if block.attachment:
            digests.append(block.attachment)

    details: dict[str, Any] = {
        "provider": str(outcome.route),
        "model": outcome.model,
        "prompt": outcome.prompt,
        "seed": outcome.seed,
        "generation_id": outcome.generation_id,
        "cancel_handle": _cancel_handle_details(handle),
        "cost_usd": outcome.cost_usd,
        "attempts": _attempt_list(outcome.attempts),
    }
    if params.source_image_path:
        details["source_image_path"] = params.source_image_path
        if params.strength is not None:
            details["strength"] = params.strength

    label = RUNG_LABELS.get(outcome.route, str(outcome.route))
    if not blocks:
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="generate_image",
            is_error=True,
            content=[
                TextContent(
                    text=(
                        f"Generated {len(outcome.assets)} image(s) with {label} "
                        f"({outcome.model}) but none {_REGISTER_FAILED_NOTE} "
                        f"(sources: {', '.join(failed)})."
                    )
                )
            ],
            details=details,
        )

    noun = "image" if len(blocks) == 1 else "images"
    seed_text = f", seed {outcome.seed}" if outcome.seed is not None else ""
    cost_text = f" Cost ${outcome.cost_usd:g}." if outcome.cost_usd is not None else ""
    caption = (
        f"Generated {len(blocks)} {noun} with {label} ({outcome.model}), "
        f"{_dimensions_text(outcome.assets, params.image_size)}{seed_text} — "
        f"attached to the session (digest {', '.join(digests)}).{cost_text}"
    )
    if failed:
        caption += (
            f" {len(failed)} of {len(outcome.assets)} failed to attach "
            f"(source: {', '.join(failed)})."
        )
    return ToolResult(
        tool_call_id=tool_call_id,
        tool_name="generate_image",
        content=[TextContent(text=caption), *blocks],
        details=details,
    )
