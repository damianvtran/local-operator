"""The ``generate_video`` tool and its ``video_status`` companion (media wave-2).

**Why two tools.** ``generate_video`` starts a detached generation and returns
a receipt in milliseconds — it registers an :class:`AsyncJob` of type
``video`` and NEVER awaits the walk (the operator's requirement: long
generations run in the conversation while the agent moves on, and the
completion wakes it). ``video_status`` answers "what happened to job X" at any
later point, cold or live: the submission path, the review turn and the cold
restart all read the same durable record. The two are ONE subsystem; the
status tool exists in this module because nothing else knows the record.

**The gate** (``imagegen``'s createIf precedent verbatim): the tool is
advertised only when a video credential exists AND a job manager is present —
``video_provider_reachable`` (sync, socket-free: a persisted Radient login or
a FAL key) and ``context.jobs``. A session that cannot run the tool carries no
schemas for it (design D5.2), and the deep docs are the two pointer docs:
``guide://video-generation`` and ``tool://generate_video``.

**Approval** (write tier, like ``generate_image``): the describer names the
route the resolver will try, the duration and a pre-spend estimate; the
figure is documentation-grade arithmetic (design D8) and the settle-time
receipt carries what the provider actually reported — one figure computed
once, echoed into the row, the details and the ledger.

**Sources** (frozen §2.2): ``t2v`` with no source; ``i2v`` with exactly one
image (``source_image_path`` or ``source_image_attachment`` — loaded to a
data URI through the IMAGE lane's own loaders, so "a readable image" cannot
mean two things); ``v2v`` with exactly one video, an https URL passed through
``source_video_path`` in v1 (local uploads and attachment-sourced videos are
refused with a named reason: the wire takes URLs, and a silent different-shape
call is the one outcome the contract forbids).

**Completion** (§5.1–§5.3): the runner registers each asset into the
session's attachment store (``kind="video"``), renders one result line
(model · duration · price · digest) and stamps the review receipt onto the
job row; the manager's existing settle path delivers it — a normal
``job_result`` row whose details carry the digest, so the review turn and the
ledger agree by construction.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any, Callable, Literal, cast
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    AgentToolUpdate,
    InvalidToolArgumentsError,
    ToolContext,
    ToolResult,
)
from local_operator.tools.builtin import _error, _guard, _text
from local_operator.video import VideoRoute
from local_operator.video.availability import video_provider_reachable
from local_operator.video.cascade import (
    VIDEO_RUNG_LABELS,
    VIDEO_RUNG_SPECS,
    capable_video_routes,
    no_capable_message,
    resolve_video_route,
)
from local_operator.video.jobs import (
    VideoWakeHandle,
    finish_video_result,
    make_video_job_runner,
    make_video_watcher_runner,
    video_job_label,
)
from local_operator.video.models import (
    FAL_DEFAULT_MODELS,
    estimate_unit_price_usd,
    family_for_model,
)
from local_operator.video.records import (
    build_record,
    read_video_records,
    record_is_terminal,
)
from local_operator.video.rungs import PROVIDER_GONE_NOTE, poll_video_once

logger = logging.getLogger(__name__)

#: Wire description (the ``generate_image`` shape: classification reads ONLY
#: the first sentence, 160 chars; deeper docs live behind the pointers). The
#: first sentence is the contract §2.3 quoted string verbatim.
_DESCRIPTION = (
    "Generate a video from text, an image, or a video clip; runs in the background and "
    "reports back. Docs: `tool://generate_video`; playbook: `guide://video-generation`."
)

#: The status tool's wire description (same shape and bounds).
_STATUS_DESCRIPTION = (
    "Report a video generation job's status (running, completed, failed or cancelled) and "
    "register its result once. Docs: `guide://video-generation`."
)

#: Cold-follow estimates for Radient, per second, by requested resolution —
#: CONSERVATIVE CEILINGS, not quotes: the hub's model is picked at call time
#: and the approval copy is written socket-free, so the figure must not
#: understate what a veo-class row can bill (the post-promo rates the memos
#: record: veo3.1 no-audio is the top row today at 0.20/s 720p and 0.40/s 4k;
#: every other current row is at or below its resolution tier). D8 posture:
#: laballed as an estimate, never as the charge.
_RADIENT_UNIT_CEILINGS: dict[str, float] = {
    "480p": 0.07,
    "720p": 0.20,
    "1080p": 0.28,
    "4k": 0.40,
}

#: Refused source forms, one sentence each so the model gets the reason (not
#: just "no"): local video uploads have no wire in v1, and the attachment
#: store has no path to a provider-hosted video URL.
_VIDEO_UPLOAD_REFUSAL = (
    "source_video_path must be an https URL for video-to-video in v1 — local "
    "video uploads are not supported yet; host the file and pass its URL."
)
_VIDEO_ATTACHMENT_REFUSAL = (
    "source_video_attachment is not supported yet — video-to-video takes a "
    "provider-reachable https URL via source_video_path in v1."
)

#: The transport cap for a source image (manager note, 2026-10-10: the hub's
#: source transport is capped at 16 MiB). Checked HERE so both rungs and the
#: hub agree, and the refusal names the cap rather than letting a provider
#: reject it opaquely.
_SOURCE_IMAGE_MAX_BYTES = 16 * 1024 * 1024


class GenerateVideoParams(BaseModel):
    """``generate_video``'s frozen parameter surface (video contract §2.2)."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, description="What the video should show.")
    model: str | None = Field(
        default=None,
        description=("Provider model id; omit to let the provider pick its default."),
    )
    duration_s: int | None = Field(
        default=None,
        gt=0,
        description="Requested duration in seconds; omit for the provider default.",
    )
    resolution: Literal["480p", "720p", "1080p", "4k"] | None = Field(
        default=None, description="Target resolution tier; omit for the provider default."
    )
    aspect_ratio: Literal["16:9", "9:16", "1:1"] | None = Field(
        default=None, description="Aspect ratio; omit for the provider default."
    )
    audio: bool | None = Field(
        default=None,
        description="Request a soundtrack where the model supports one; omit for the default.",
    )
    seed: int | None = Field(default=None, description="Deterministic seed where supported.")
    source_image_path: str | None = Field(
        default=None,
        description="Local image for image-to-video (i2v) — one source exactly.",
    )
    source_image_attachment: str | None = Field(
        default=None,
        description="Attachment digest for image-to-video (i2v) — one source exactly.",
    )
    source_video_path: str | None = Field(
        default=None,
        description=(
            "An https URL of the source video for video-to-video (v2v) — one source exactly."
        ),
    )
    source_video_attachment: str | None = Field(
        default=None,
        description="Reserved: attachment-sourced v2v is not supported in v1.",
    )


def _resolve_sources(
    params: GenerateVideoParams, cwd: str
) -> tuple[str, str | None, str | None, list[Mapping[str, Any]]]:
    """Validate the source fields; derive ``(mode, source_kind, wire, record_rows)``.

    Exactly one source field may be set (the frozen one-source v1 rule). The
    wire value is what the rungs send: a ``data:`` URI for images (loaded
    through the IMAGE lane's own loaders, so "a readable image" cannot mean
    two things), the URL itself for videos. The record rows are the compact
    forms the durable record carries — a path or a digest, NEVER the image
    bytes (a record is a status line, not a payload).
    """
    from local_operator.tools.image_tool import (
        _load_source_attachment,
        _load_source_image,
    )

    given = [
        name
        for name, value in (
            ("source_video_path", params.source_video_path),
            ("source_video_attachment", params.source_video_attachment),
            ("source_image_path", params.source_image_path),
            ("source_image_attachment", params.source_image_attachment),
        )
        if value
    ]
    if len(given) > 1:
        raise InvalidToolArgumentsError("Pass exactly one source: " + " or ".join(given) + ".")
    if params.source_video_attachment:
        raise InvalidToolArgumentsError(_VIDEO_ATTACHMENT_REFUSAL)
    if params.source_video_path:
        url = params.source_video_path.strip()
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise InvalidToolArgumentsError(_VIDEO_UPLOAD_REFUSAL)
        return "v2v", "video", url, [{"kind": "video", "url": url}]
    if params.source_image_path:
        data_uri = _load_source_image(params.source_image_path, cwd)
        _check_source_image_cap(data_uri)
        return (
            "i2v",
            "image",
            data_uri,
            [{"kind": "image", "path": params.source_image_path}],
        )
    if params.source_image_attachment:
        data_uri = _load_source_attachment(params.source_image_attachment)
        _check_source_image_cap(data_uri)
        return (
            "i2v",
            "image",
            data_uri,
            [{"kind": "image", "attachment": params.source_image_attachment}],
        )
    return "t2v", None, None, []


def _check_source_image_cap(data_uri: str) -> None:
    """Refuse a source image over the transport cap (named, not opaque)."""
    # base64 inflates by 4/3; the payload's size is what the cap speaks to.
    payload_bytes = len(data_uri) * 3 // 4
    if payload_bytes > _SOURCE_IMAGE_MAX_BYTES:
        raise InvalidToolArgumentsError(
            "The source image is larger than the 16 MiB source cap "
            f"(~{payload_bytes / (1024 * 1024):.1f} MiB); use a smaller image."
        )


def _record_params(params: GenerateVideoParams) -> dict[str, Any]:
    """The compact parameter set a durable record carries (provenance, no bytes)."""
    rows: dict[str, Any] = {"prompt": params.prompt}
    for key, value in (
        ("duration_s", params.duration_s),
        ("resolution", params.resolution),
        ("aspect_ratio", params.aspect_ratio),
        ("audio", params.audio),
        ("seed", params.seed),
    ):
        if value is not None:
            rows[key] = value
    return rows


def _estimate_usd(
    route: str | None, mode: str, duration_s: int | None, resolution: str | None
) -> float | None:
    """Pre-spend estimate for the approval copy and the submit receipt.

    FAL quotes its default family's post-promo rate (the model a default run
    actually meets); Radient quotes the conservative ceiling for the tier
    (its model is chosen at call time and this figure is written
    socket-free). ``duration_s=None`` quotes nothing — the provider's own
    default duration is unknown here, and inventing one would make the figure
    wrong in a way only the bill would reveal.
    """
    if duration_s is None or route is None:
        return None
    if route == str(VideoRoute.FAL):
        family = family_for_model(FAL_DEFAULT_MODELS.get(mode, ""))
        unit = estimate_unit_price_usd(family, resolution)
    else:
        unit = _RADIENT_UNIT_CEILINGS.get(resolution or "720p")
    if unit is None:
        return None
    return duration_s * unit


def _preferred_video_route(mode: str) -> str | None:
    """The cascade's first available route for ``mode``, sync and socket-free.

    Reads the same probes the gate reads and filters by the capability table
    (a v2v approval must not name a provider that cannot edit video); ``None``
    when nothing answers — the tool would not have been built, but a
    describer must never fail a call.
    """
    from local_operator.imagegen import availability as image_availability
    from local_operator.paths import config_dir

    try:
        cfg = config_dir()
        if mode in VIDEO_RUNG_SPECS[
            VideoRoute.RADIENT
        ].capabilities and image_availability.radient_available(cfg):
            return str(VideoRoute.RADIENT)
        if mode in VIDEO_RUNG_SPECS[VideoRoute.FAL].capabilities and image_availability.fal_key(
            cfg
        ):
            return str(VideoRoute.FAL)
    except Exception:  # noqa: BLE001 — a describer never fails a call
        logger.debug("video route label probe failed", exc_info=True)
    return None


def _describe_generate_video_approval(args: dict[str, Any], cwd: str) -> str:
    """What the approval prompt says: route, duration, and the pre-spend estimate.

    Written for the person answering under time pressure: the decision-
    relevant facts are the DURATION (it multiplies the per-second rate), the
    route and that money leaves their account. The prompt text is already on
    the card; the raw JSON fallback would dump what this sentence states.

    The route is named for the MODE the source fields imply, so a v2v request
    is never approved against a provider that cannot edit video.
    """
    del cwd  # the spend is account-level; no path decides it
    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return "[unresolvable] generate_video without a prompt"
    if args.get("source_video_path") or args.get("source_video_attachment"):
        mode = "v2v"
        what = "Generate a video from a source video"
    elif args.get("source_image_path") or args.get("source_image_attachment"):
        mode = "i2v"
        what = "Generate a video from an image"
    else:
        mode = "t2v"
        what = "Generate a video"
    duration = args.get("duration_s")
    duration = (
        duration
        if isinstance(duration, int) and not isinstance(duration, bool) and duration > 0
        else None
    )
    resolution = args.get("resolution")
    resolution = resolution if isinstance(resolution, str) and resolution else None
    if duration is not None:
        what = what.replace("Generate a video", f"Generate an {duration}s video")
    route = _preferred_video_route(mode)
    provider = (
        VIDEO_RUNG_LABELS.get(route, "the configured provider")
        if route
        else "the configured provider"
    )
    estimated = _estimate_usd(route, mode, duration, resolution) if route else None
    spend = "a paid provider call on your account"
    if estimated is not None:
        at = f" at {resolution}" if resolution is not None else ""
        spend = f"~${estimated:.2f}{at} estimated — a paid provider call on your account"
    return f"{what} via {provider} — {spend}."


# ---------------------------------------------------------------------------
# generate_video: the detached submission
# ---------------------------------------------------------------------------


@_guard("generate_video")
async def execute_generate_video(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Validate → resolve → register → receipt; the walk runs DETACHED.

    The call returns in milliseconds (the operator's requirement: the agent
    moves on). Everything after registration belongs to the runner: the
    provider submit, the durable records, the completion and its delivery.
    """
    del signal, on_update  # the job is not interruptible in-turn; the JOB owns abort
    try:
        params = GenerateVideoParams.model_validate(args)
    except ValidationError as exc:
        raise InvalidToolArgumentsError(str(exc)) from exc
    cwd = (context.cwd if context is not None else None) or "."
    mode, source_kind, source_url, source_rows = _resolve_sources(params, cwd)
    jobs = context.jobs if context is not None else None
    if jobs is None:
        return _error(tool_call_id, "generate_video", "no background job manager is available")

    from local_operator.paths import config_dir

    cfg = config_dir()
    route_resolution = await resolve_video_route(cfg)
    capable = capable_video_routes(route_resolution, mode, source_kind)
    if not capable:
        # The refusal names the capable providers (the remedy), and nothing
        # is registered: a job no rung can run would fail minutes later with
        # the same sentence, after a pointless wake cycle.
        return _error(tool_call_id, "generate_video", no_capable_message(mode, source_kind))
    first_route = capable[0]
    route_label = VIDEO_RUNG_LABELS.get(str(first_route), str(first_route))
    label = video_job_label(mode=mode, duration_s=params.duration_s, route_label=route_label)
    journal = getattr(context, "journal_video_job", None)
    scheduler = getattr(context, "wake_scheduler", None)
    runner = make_video_job_runner(
        mode=mode,
        prompt=params.prompt,
        source_kind=source_kind,
        source_url=source_url,
        sources=source_rows,
        duration_s=params.duration_s,
        resolution=params.resolution,
        aspect_ratio=params.aspect_ratio,
        audio=params.audio,
        seed=params.seed,
        model=params.model,
        config_dir=cfg,
        journal=journal,
        wake_scheduler=scheduler,
        append_output=getattr(jobs, "append_output", None),
    )
    # ``register`` is deliberately outside ``JobManagerProtocol`` (its
    # docstring: spawning must not ride the tool-facing surface), so the cast
    # is the same sanctioned spelling bash's background path uses — the object
    # beneath every production context is the real ``AsyncJobManager``.
    try:
        job_id = cast(Any, jobs).register(type="video", label=label, run=runner)
    except RuntimeError as exc:
        return _error(tool_call_id, "generate_video", f"could not start the video job: {exc}")
    wake = VideoWakeHandle(scheduler, job_id)
    # Arm the check-wake BEFORE the submit record: a crash between the two
    # still leaves a wake that asks about this job, and the ensure path is
    # idempotent, so the belt is never doubled.
    await wake.ensure()
    record_written = False
    if journal is not None:
        try:
            await journal(
                build_record(
                    job_id=job_id,
                    state="registered",
                    mode=mode,
                    provider=str(first_route),
                    route_label=route_label,
                    model=params.model,
                    params=_record_params(params),
                    sources=source_rows or None,
                )
            )
            record_written = True
        except Exception:  # noqa: BLE001 — the generation is already running
            logger.warning("video submit record write failed", exc_info=True)
    estimated = _estimate_usd(str(first_route), mode, params.duration_s, params.resolution)
    duration_text = (
        f"{params.duration_s} s" if params.duration_s is not None else "the default duration"
    )
    est_text = f", ~${estimated:.2f} est." if estimated is not None else ""
    text = (
        f"Video generation started — job `{job_id}` (mode {mode}, {duration_text}"
        f"{est_text}, {route_label}). You'll get a completion notice when it settles; "
        f"cancel with `jobs op='cancel', job_id='{job_id}'`."
    )
    details: dict[str, Any] = {
        "job_id": job_id,
        "stage": "queued",
        "provider": str(first_route),
        "route_label": route_label,
        "mode": mode,
        "model": params.model,
        "duration_s": params.duration_s,
        "estimated_cost_usd": estimated,
        "cost_basis": "estimate" if estimated is not None else None,
        "cancel_support": str(VIDEO_RUNG_SPECS[first_route].cancel_support),
        "cancel_hint": f"jobs op='cancel', job_id='{job_id}'",
        "record": record_written,
    }
    return _text(tool_call_id, "generate_video", text, details=details)


def build_generate_video_tool(context: ToolContext) -> AgentTool | None:
    """createIf builder: exists only where a video job can run (D9's shape).

    The gate is SYNC and socket-free (a persisted Radient login or a FAL key)
    plus a job manager: sessions built before a login land the tool at the
    NEXT session — the gate is a snapshot by design, not a per-turn scan.
    """
    if context.jobs is None:
        return None
    from local_operator.paths import config_dir

    if not video_provider_reachable(config_dir()):
        return None
    return AgentTool(
        name="generate_video",
        label="Generate video",
        description=_DESCRIPTION,
        parameters=GenerateVideoParams.model_json_schema(),
        # write tier: the generation spends real money on a third-party
        # account and its product is billable.
        approval_tier="write",
        # Generations are independent; two calls can run beside each other
        # without sharing anything but the credential reads.
        concurrency="shared",
        # The call is DETACHED: it returns in milliseconds, so there is no
        # in-turn wait to interrupt. Stopping the conversation does NOT
        # cancel the provider job; the job is cancelled through
        # `jobs op='cancel'` (the tool doc and the guide say so in as many
        # words).
        interruptible=False,
        describe_approval=_describe_generate_video_approval,
        execute=execute_generate_video,
    )


# ---------------------------------------------------------------------------
# video_status: the cold-safe read side
# ---------------------------------------------------------------------------


class VideoStatusParams(BaseModel):
    """Arguments for the ``video_status`` tool."""

    model_config = ConfigDict(extra="forbid")

    job_id: str | None = Field(
        default=None,
        description="Video job id; the newest pending job is reported when omitted.",
    )


def _receipt_from_record(record: dict[str, Any], job_id: str) -> dict[str, Any]:
    """The completion details block, rebuilt from the durable record."""
    raw_result = record.get("result")
    result: Mapping[str, Any] = raw_result if isinstance(raw_result, Mapping) else {}
    return {
        "job_id": job_id,
        "state": str(record.get("state") or ""),
        "provider": record.get("provider"),
        "model": result.get("model") or record.get("model"),
        "digest": result.get("digest"),
        "digests": result.get("digests"),
        "artifacts": result.get("artifacts"),
        "content_type": result.get("content_type"),
        "duration_s": result.get("duration_s"),
        "cost_usd": result.get("cost_usd"),
        "cost_source": result.get("cost_source"),
        "billing_basis": result.get("billing_basis"),
    }


def _terminal_text(record: dict[str, Any], job_id: str) -> str:
    state = str(record.get("state") or "")
    raw_result = record.get("result")
    result: Mapping[str, Any] = raw_result if isinstance(raw_result, Mapping) else {}
    note = record.get("outcome_note")
    if state == "completed":
        pieces = [f"Video job {job_id} completed"]
        if result.get("model"):
            pieces.append(f"model {result['model']}")
        if result.get("duration_s") is not None:
            pieces.append(f"{float(result['duration_s']):.1f}s")
        if result.get("cost_usd") is not None:
            pieces.append(f"${float(result['cost_usd']):g}")
        if result.get("digest"):
            pieces.append(f"attached: digest {result['digest']}")
        else:
            pieces.append("the result is not registered in the attachment store")
        return " — ".join(pieces) + "."
    if state == "cancelled":
        if record.get("error_type") == "media_already_completed" and record.get("error"):
            # The frozen conflict branch (contract §1.6/§4.1): the cancel
            # arrived after the generation finished, and the surfaces render
            # it as "already finished" — the record's own sentence, never an
            # invented error's ink.
            return f"Video job {job_id} was cancelled. {record['error']}"
        return f"Video job {job_id} was cancelled." + (f" {note}" if note else "")
    if state == "failed":
        return f"Video job {job_id} failed" + (f": {note}" if note else ".")
    return f"Video job {job_id} is {state or 'in an unknown state'}."


@_guard("video_status")
async def execute_video_status(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Answer "what happened to job X" from the durable record, cold or live.

    Four shapes (contract §1.5): live rows answer from the job manager; a
    pending record with no watcher gets one bounded provider poll and either
    an answer or a resumed observation (a watcher JOB, so the runtime stays
    resident and delivery rides the ordinary settle path); a terminal record
    answers with its receipt, fetching and registering once when the record
    is terminal but unregistered; an unknown id gets an honest absence.
    """
    del signal, on_update
    try:
        params = VideoStatusParams.model_validate(args)
    except ValidationError as exc:
        raise InvalidToolArgumentsError(str(exc)) from exc
    session_dir = getattr(context, "session_dir", None) if context is not None else None
    if not session_dir:
        return _error(
            tool_call_id,
            "video_status",
            "this session has no transcript directory to read records from",
        )
    records = await asyncio.to_thread(read_video_records, session_dir)
    jobs = context.jobs if context is not None else None
    job_id = params.job_id
    if job_id is not None and job_id not in records:
        return _text(
            tool_call_id,
            "video_status",
            f"No video job {job_id} is recorded for this session.",
            details={"job_id": job_id, "state": None},
        )
    if job_id is None:
        pending = {jid: rec for jid, rec in records.items() if not record_is_terminal(rec)}
        pool = pending or records
        if not pool:
            # A session that never ran a video job (QA round 3): the SAME
            # honest absence a named-but-unknown id gets — `max()` on an
            # empty pool would instead raise, and that traceback would
            # surface to the model as the tool's answer.
            return _text(
                tool_call_id,
                "video_status",
                "No video job is recorded for this session.",
                details={"job_id": None, "state": None},
            )
        job_id = max(pool, key=lambda jid: str(pool[jid].get("at") or ""))
    record = records[job_id]
    live = jobs.get(job_id) if jobs is not None else None
    if record_is_terminal(record):
        return await _answer_terminal(tool_call_id, job_id, record, context=context)
    if live is not None and live.status in ("running", "queued"):
        return _answer_live(tool_call_id, job_id, record, live)
    return await _cold_follow(tool_call_id, job_id, record, context, jobs=jobs)


def _answer_live(tool_call_id: str, job_id: str, record: dict[str, Any], live: Any) -> ToolResult:
    """A still-running job WITH a live runner: the row is the freshest source."""
    import time as _time

    raw_details = live.latest_details
    details_in: Mapping[str, Any] = raw_details if isinstance(raw_details, Mapping) else {}
    raw_video = details_in.get("video")
    video: Mapping[str, Any] = raw_video if isinstance(raw_video, Mapping) else {}
    stage = video.get("stage") or "in_progress"
    provider = record.get("provider") or video.get("provider")
    model = record.get("model") or video.get("model")
    try:
        elapsed = int(_time.time() - float(live.start_time))
    except (TypeError, ValueError):
        elapsed = None
    pieces = [f"Video job {job_id} is still running"]
    if provider:
        pieces.append(f"on {VIDEO_RUNG_LABELS.get(str(provider), provider)}")
    if elapsed is not None:
        pieces.append(f"{elapsed}s elapsed")
    queue = video.get("queue_position")
    if queue:
        pieces.append(f"queue #{queue}")
    parts = [
        " · ".join(pieces) + ".",
        f"stage {stage}; cancel with `jobs op='cancel', job_id='{job_id}'`.",
    ]
    return _text(
        tool_call_id,
        "video_status",
        " ".join(parts),
        details={
            "job_id": job_id,
            "state": "running",
            "stage": stage,
            "provider": provider,
            "model": model,
            "elapsed_s": elapsed,
            "queue_position": queue,
            "log_tail": (live.output_tail or "")[-600:],
        },
    )


async def _spawn_watcher(
    context: ToolContext | None, jobs: Any, job_id: str, record: dict[str, Any]
) -> str | None:
    """Register the cold-follow watcher; ``None`` when capacity refuses.

    The watcher's job id is then journaled on a record revision (same state,
    same refs): a SECOND ``video_status`` on the same record sees it and
    answers instead of registering a twin. Each watcher settles through the
    ordinary delivery path, so N registrations would deliver N completion
    rows for one generation.
    """
    from local_operator.paths import config_dir

    scheduler = getattr(context, "wake_scheduler", None)
    raw_params = record.get("params")
    params: Mapping[str, Any] = raw_params if isinstance(raw_params, Mapping) else {}
    duration = params.get("duration_s")
    provider = str(record.get("provider") or "")
    watcher = make_video_watcher_runner(
        record=record,
        config_dir=config_dir(),
        journal=getattr(context, "journal_video_job", None),
        wake=VideoWakeHandle(scheduler, job_id),
        append_output=getattr(jobs, "append_output", None),
    )
    label = video_job_label(
        mode=str(record.get("mode") or ""),
        duration_s=duration if isinstance(duration, int) else None,
        route_label=VIDEO_RUNG_LABELS.get(provider, provider or "provider"),
    )
    try:
        watcher_id = jobs.register(type="video", label=label, run=watcher)
    except RuntimeError:
        return None
    await _journal_watcher_stamp(context, job_id, record, watcher_id)
    return watcher_id


async def _journal_watcher_stamp(
    context: ToolContext | None, job_id: str, record: dict[str, Any], watcher_id: str
) -> None:
    """Append the record revision that carries the watcher's id (best-effort)."""
    journal = getattr(context, "journal_video_job", None) if context is not None else None
    if journal is None:
        return
    params = record.get("params") if isinstance(record.get("params"), Mapping) else None
    sources = record.get("sources") if isinstance(record.get("sources"), list) else None
    provider = str(record.get("provider") or "")
    try:
        await journal(
            build_record(
                job_id=job_id,
                state=str(record.get("state") or "running"),
                provider=provider or None,
                route_label=VIDEO_RUNG_LABELS.get(provider, None),
                model=str(record.get("model") or "") or None,
                mode=str(record.get("mode") or "") or None,
                params=params,
                sources=sources,
                provider_ref=(
                    record.get("provider_ref")
                    if isinstance(record.get("provider_ref"), Mapping)
                    else None
                ),
                watcher=watcher_id,
            )
        )
    except Exception:  # noqa: BLE001 — the watcher is already running
        logger.warning("video watcher stamp write failed", exc_info=True)


async def _cold_follow(
    tool_call_id: str,
    job_id: str,
    record: dict[str, Any],
    context: ToolContext | None,
    *,
    jobs: Any,
) -> ToolResult:
    """One bounded poll of a pending record's provider job; then answer or watch."""
    from local_operator.artifacts.rung import CancelHandle
    from local_operator.paths import config_dir

    handle = CancelHandle()
    try:
        outcome = await poll_video_once(record=record, handle=handle, config_dir=config_dir())
    except Exception as exc:  # noqa: BLE001 — an answer, never a raise
        return _text(
            tool_call_id,
            "video_status",
            f"Video job {job_id} is pending but the provider could not be reached "
            f"({exc.__class__.__name__}: {exc}); the check-wake remains armed.",
            details={"job_id": job_id, "state": "running", "provider": record.get("provider")},
        )
    if outcome.state == "completed" and outcome.result is not None:
        receipt, text = finish_video_result(
            record_job_id=job_id,
            route=str(record.get("provider") or ""),
            model=outcome.result.model or str(record.get("model") or ""),
            assets=outcome.result.assets,
            cost_usd=outcome.result.cost_usd,
            cost_source=outcome.result.cost_source,
            billing_basis=outcome.result.billing_basis,
        )
        await _write_settle(context, job_id, record, state="completed", result=receipt)
        await VideoWakeHandle(getattr(context, "wake_scheduler", None), job_id).cancel()
        return _text(tool_call_id, "video_status", text, details=receipt)
    if outcome.state in ("cancelled", "failed", "gone"):
        note = {
            "cancelled": "the provider reported the generation as CANCELLED",
            "failed": outcome.detail or "the provider reported the generation as FAILED",
            "gone": PROVIDER_GONE_NOTE,
        }[outcome.state]
        state = "cancelled" if outcome.state == "cancelled" else "failed"
        await _write_settle(context, job_id, record, state=state, outcome_note=note)
        await VideoWakeHandle(getattr(context, "wake_scheduler", None), job_id).cancel()
        return _text(
            tool_call_id,
            "video_status",
            f"Video job {job_id} {'was cancelled' if state == 'cancelled' else 'failed'}: {note}.",
            details={"job_id": job_id, "state": state, "error": note},
        )
    # Observation identity (V1): a live watcher already watching this record
    # owns the polls — a second registration would settle a SECOND completion
    # row for the same generation. The watcher's id is journaled on the record
    # by ``_spawn_watcher``; a terminal/absent row here means the watcher is
    # gone (crash, cancel, settle), so falling through to a fresh spawn is the
    # honest read.
    known_watcher = record.get("watcher") if isinstance(record.get("watcher"), str) else None
    if known_watcher and jobs is not None:
        watcher = jobs.get(known_watcher)
        if watcher is not None and watcher.status in ("running", "queued"):
            return _text(
                tool_call_id,
                "video_status",
                f"Video job {job_id} is still running on "
                f"{VIDEO_RUNG_LABELS.get(str(record.get('provider')), 'its provider')}; "
                f"observation is already resumed as job {known_watcher} — the completion "
                f"will be delivered. To cancel the provider generation, cancel that "
                f"job: `jobs op='cancel', job_id='{known_watcher}'`.",
                details={
                    "job_id": job_id,
                    "state": "running",
                    "provider": record.get("provider"),
                    "watcher_job_id": known_watcher,
                },
            )
    # Still running: a watcher job keeps the runtime resident and hands the
    # completion to the ordinary settle/delivery path.
    watcher_id = await _spawn_watcher(context, jobs, job_id, record) if jobs is not None else None
    wake = VideoWakeHandle(getattr(context, "wake_scheduler", None), job_id)
    if watcher_id is not None:
        await wake.cancel()
        text = (
            f"Video job {job_id} is still running on "
            f"{VIDEO_RUNG_LABELS.get(str(record.get('provider')), 'its provider')}; "
            f"observation resumed as job {watcher_id} — the completion will be "
            f"delivered. To cancel the provider generation, cancel that job: "
            f"`jobs op='cancel', job_id='{watcher_id}'`."
        )
        return _text(
            tool_call_id,
            "video_status",
            text,
            details={
                "job_id": job_id,
                "state": "running",
                "provider": record.get("provider"),
                "watcher_job_id": watcher_id,
            },
        )
    await wake.ensure()
    return _text(
        tool_call_id,
        "video_status",
        f"Video job {job_id} is still running; the check-wake remains armed.",
        details={"job_id": job_id, "state": "running", "provider": record.get("provider")},
    )


async def _write_settle(
    context: ToolContext | None,
    job_id: str,
    record: dict[str, Any],
    *,
    state: str,
    result: dict[str, Any] | None = None,
    outcome_note: str | None = None,
) -> None:
    """Append the settling record from the status tool (best-effort)."""
    journal = getattr(context, "journal_video_job", None) if context is not None else None
    if journal is None:
        return
    params = record.get("params") if isinstance(record.get("params"), Mapping) else None
    sources = record.get("sources") if isinstance(record.get("sources"), list) else None
    provider_ref = (
        record.get("provider_ref") if isinstance(record.get("provider_ref"), Mapping) else None
    )
    try:
        await journal(
            build_record(
                job_id=job_id,
                state=state,
                provider=str(record.get("provider") or "") or None,
                route_label=VIDEO_RUNG_LABELS.get(str(record.get("provider") or ""), None),
                model=str(record.get("model") or "") or None,
                mode=str(record.get("mode") or "") or None,
                params=params,
                sources=sources,
                provider_ref=provider_ref,
                result=result,
                outcome_note=outcome_note,
            )
        )
    except Exception:  # noqa: BLE001 — a record write never fails a status read
        logger.warning("video status record write failed", exc_info=True)


async def _answer_terminal(
    tool_call_id: str,
    job_id: str,
    record: dict[str, Any],
    *,
    context: ToolContext | None,
) -> ToolResult:
    """A terminal record answers from itself; a completed-but-UNREGISTERED one
    gets ONE bounded refetch first (contract §1.5: fetch+register once,
    idempotent) — the provider may still serve a result whose registration
    failed, and a digest is worth one more read before the answer is final."""
    state = str(record.get("state") or "")
    raw_result = record.get("result")
    result: Mapping[str, Any] = raw_result if isinstance(raw_result, Mapping) else {}
    if state == "completed" and not result.get("digest"):
        from local_operator.artifacts.rung import CancelHandle
        from local_operator.paths import config_dir

        handle = CancelHandle()
        try:
            outcome = await poll_video_once(record=record, handle=handle, config_dir=config_dir())
        except Exception:  # noqa: BLE001 — the record's own answer stands
            outcome = None
        if outcome is not None and outcome.state == "completed" and outcome.result is not None:
            receipt, text = finish_video_result(
                record_job_id=job_id,
                route=str(record.get("provider") or ""),
                model=outcome.result.model or str(record.get("model") or ""),
                assets=outcome.result.assets,
                cost_usd=outcome.result.cost_usd,
                cost_source=outcome.result.cost_source,
                billing_basis=outcome.result.billing_basis,
            )
            await _write_settle(context, job_id, record, state="completed", result=receipt)
            return _text(tool_call_id, "video_status", text, details=receipt)
    return _text(
        tool_call_id,
        "video_status",
        _terminal_text(record, job_id),
        details=_receipt_from_record(record, job_id),
    )


def build_video_status_tool(context: ToolContext) -> AgentTool | None:
    """createIf builder: exists wherever a video job can exist.

    Rides the SAME gate as ``generate_video`` (a credential plus the job
    manager): reading a record needs only the transcript, but a session with
    no way to START a video has no video job to report on, and the pair's
    roster signal must stay honest — the tool appearing means the capability
    is there.
    """
    if context.jobs is None:
        return None
    from local_operator.paths import config_dir

    if not video_provider_reachable(config_dir()):
        return None
    return AgentTool(
        name="video_status",
        label="Video status",
        description=_STATUS_DESCRIPTION,
        parameters=VideoStatusParams.model_json_schema(),
        # Read tier: the call reads records and may register a settled result,
        # but spends nothing at a provider and asks consent for nothing.
        approval_tier="read",
        concurrency="shared",
        interruptible=False,
        execute=execute_video_status,
    )
