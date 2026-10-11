"""httpx-async video rung executors, and the cold-follow provider poll.

**D6/D7 inherited verbatim from the image lane** (see
``local_operator.imagegen.rungs``'s module docstring for the full argument):
native asyncio, no threads, no subprocesses — cancellation lands at a
suspension point and closes the socket, and nothing registers with
``group_reaper`` because nothing here spawns an orphanable process group.

**Wire shapes.** Two rungs:

- **Radient (the hub)** — the SAME ``/tools/media/*`` routes the image lane
  drives, with a bearer: models list → affordability → generate → status poll
  → result → bounded download. The hub forwards the body verbatim to FAL
  except ``model``/``provider``, so the request keys come from the family
  table (``video.models``) — the frozen per-alias request contract. A video
  result carries ``output.video{}`` (measured on prod: the hub RFC's C7), and
  the reader also folds the neutral ``artifacts[]`` list the moment a row is
  upgraded — absence of each is handled tolerantly.
- **FAL (BYO key)** — the queue API driven natively, exactly like the image
  lane's ``run_fal``: submit ``sync_mode: false``, poll the response-carried
  ``status_url`` (fallback derived from the app path), read the
  response-carried ``response_url``.

The hub helpers this rung shares (``_request_json``, ``_radient_failure``, the
media-route constants, ``best_effort_cancel``) are imported from
``imagegen.rungs`` ON PURPOSE: they are the hub client, the same routes serve
image and video, and a second copy of the poll/failure mapping is exactly how
the two lanes would drift. The image lane owns those helpers; this module
adds only what is video-specific (family mapping, the video reader, the
larger download cap, the slower poll cadence).

**Cold-follow**: :func:`resume_video_observation` polls a job that outlived
its process, using only the durable record's ``provider_ref`` — this is what
``video_status`` and the watcher job call after a restart/rollover.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from pydantic import SecretStr

from local_operator.artifacts.progress import ProgressFn, emit_progress
from local_operator.artifacts.progress import (
    progress_details as _generic_progress_details,
)
from local_operator.artifacts.rung import CancelHandle, RungResult, RungSkipped
from local_operator.artifacts.walk import PauseFn
from local_operator.clients._http import APIError
from local_operator.imagegen import MediaAsset
from local_operator.imagegen import availability as image_availability
from local_operator.imagegen.media import download_asset
from local_operator.imagegen.rungs import (
    HTTP_TIMEOUT_POLL_S,
    HTTP_TIMEOUT_RESULT_S,
    HTTP_TIMEOUT_SUBMIT_S,
    RADIENT_MEDIA_GENERATE_PATH,
    RADIENT_MEDIA_MODELS_PATH,
    RADIENT_MEDIA_RESULT_PATH,
    RADIENT_MEDIA_STATUS_PATH,
    RADIENT_MODELS_TIMEOUT_S,
    _as_float,
    _bearer_headers,
    _client_scope,
    _fal_fallback_urls,
    _num,
    _radient_affordable,
    _radient_failure,
    _radient_unit_price,
    _request_json,
)
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.radient_credentials import resolve_radient_credential
from local_operator.video import VideoRoute
from local_operator.video.models import (
    FAL_DEFAULT_MODELS,
    VideoFamily,
    VideoParamRejected,
    family_for_model,
    map_family_params,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CancelHandle",
    "PROVIDER_GONE_NOTE",
    "RADIENT_VIDEO_SOURCE_SKIP_MESSAGE",
    "VIDEO_FAL_QUEUE_BASE_URL",
    "VideoPollOutcome",
    "poll_video_once",
    "resume_video_observation",
    "run_fal_video",
    "run_radient_video",
]

#: The provider-gone sentence — ONE home for all three sites that report a
#: 404 (the one-shot poll's ``gone`` note, the watcher's settle note, the
#: observation raise), so a user-visible sentence exists once (D4: the lane
#: shipped "(404)", "(HTTP 404)" and a third variant of the same fact).
PROVIDER_GONE_NOTE = (
    "the provider no longer knows this job (HTTP 404) — it was cancelled or "
    "reaped upstream; the result is not retrievable"
)

#: Radient's source refusal for video requests — ONE home, shared by the
#: rung's own defence-in-depth skip and the cascade's capability pre-record
#: (``cascade._route_skip_message``) so the two sentences cannot drift. WHY a
#: skip and not an attempt: the hub's media route forwards every body key
#: verbatim to FAL, and the server-side source mapping for video (the
#: ``video_url``/``source_url`` words — manager note 2026-10-10) ships with the
#: i2i lane's enabling PR; until it lands, a source request through the hub is
#: at best a rejection and at worst a silently BILLED text-to-video. The
#: enabling PR flips ``VIDEO_RUNG_SPECS[RADIENT]``'s capabilities and
#: ``source_kinds`` beside wiring the field constants in ``video.models``.
RADIENT_VIDEO_SOURCE_SKIP_MESSAGE = (
    "Radient cannot take image or video sources yet — its media route has no "
    "source mapping for video; use another provider"
)

#: FAL's queue host — the same constant the image lane uses, re-read here by
#: importing nothing new: the host is shared, declared once per module so a
#: staging override lands beside the code that spends it.
VIDEO_FAL_QUEUE_BASE_URL = "https://queue.fal.run"

#: One poll tick while the provider works. Slower than the image lane's 2 s on
#: purpose: video jobs run minutes-to-tens-of-minutes, so a 2 s cadence buys
#: nothing a 5 s one does not and multiplies the free reads by 2.5×.
VIDEO_POLL_INTERVAL_S = 5.0
#: Once a job is clearly inference-bound the poll relaxes again.
VIDEO_POLL_SLOW_AFTER_S = 60.0
VIDEO_POLL_SLOW_INTERVAL_S = 10.0
#: The no-signal (headless/library) back-off ceiling, as in the image lane.
VIDEO_POLL_NO_SIGNAL_CAP_S = 20.0

#: Per-asset byte ceiling for VIDEO downloads — the image lane's 32 MiB cap
#: parameterized (contract §3.3/G4): a 30 s 1080p clip plausibly runs tens of
#: MiB, so the video rung passes this figure; ``media.download_asset`` keeps
#: its own default for every image caller. A larger payload refuses rather
#: than truncates, exactly as there.
VIDEO_ARTIFACT_MAX_BYTES = 128 * 1024 * 1024

#: How long a WATCHER may observe a provider job before giving up
#: (``resume_video_observation``'s bound). Equal to the walk's overall budget:
#: a cold-followed job is the same generation, watched later — the bound is
#: about not observing forever, not about re-racing the original submission.
VIDEO_OBSERVATION_TIMEOUT_S = 1800.0
#: Read half sized for a large file over a slow link (128 MiB at ~0.5 MB/s);
#: connect half unchanged from the image lane's rationale.
VIDEO_DOWNLOAD_TIMEOUT = httpx.Timeout(300.0, connect=10.0)


def _video_poll_interval(elapsed_s: float) -> float:
    """5 s while queue-bound, relaxing to 10 s once the job is inference-bound.

    Slower than the image lane's 2 s on purpose: video jobs run minutes, and
    a 2 s tick over a 10-minute poll is 300 reads for a queue position nobody
    watches that closely.
    """
    if elapsed_s >= VIDEO_POLL_SLOW_AFTER_S:
        return VIDEO_POLL_SLOW_INTERVAL_S
    return VIDEO_POLL_INTERVAL_S


async def _notify_accept(
    on_accept: "Callable[[CancelHandle], Awaitable[None]] | None", handle: CancelHandle
) -> None:
    """Fire the runner's accept hook; a failing hook must not kill the job.

    The hook writes the durable ``running`` record. If that write fails the
    generation is still live and cancellable in this process — losing the
    record degrades restart-durability, it does not justify abandoning a
    provider job that is already billing.
    """
    if on_accept is None:
        return
    try:
        await on_accept(handle)
    except Exception:  # noqa: BLE001 — best-effort hook
        logger.warning("video accept hook failed; continuing", exc_info=True)


def _no_signal_video_poll_interval(elapsed_s: float) -> float:
    """5 s → 10 s → 20 s(cap) for callers nothing is watching."""
    return min(
        VIDEO_POLL_INTERVAL_S * (2 ** int(elapsed_s // VIDEO_POLL_SLOW_AFTER_S)),
        VIDEO_POLL_NO_SIGNAL_CAP_S,
    )


async def _pace_poll(pause: "PauseFn | None", elapsed_s: float) -> None:
    """Wait between poll reads — abortably when a signal exists, paced when not."""
    if pause is not None:
        await pause(_video_poll_interval(elapsed_s))
    else:
        await asyncio.sleep(_no_signal_video_poll_interval(elapsed_s))


def progress_details(
    *,
    stage: str | None,
    provider: str | None = None,
    model: str | None = None,
    elapsed_s: int | None = None,
    num_images: int | None = None,
    queue_position: int | None = None,
    log_lines: list[dict[str, Any]] | None = None,
    error: str | None = None,
    error_type: str | None = None,
) -> dict[str, Any]:
    """This lane's binding of the canonical payload builder (frozen stages).

    ``tool_name`` pins to ``generate_video`` here, so the surfaces' detection
    sets see the emitting tool; the stage vocabulary is the FROZEN one —
    queued / in_progress / completed / cancelled / cancelling / None — and
    this lane invents no words.
    """
    return _generic_progress_details(
        tool="generate_video",
        stage=stage,
        provider=provider,
        model=model,
        elapsed_s=elapsed_s,
        num_images=num_images,
        queue_position=queue_position,
        log_lines=log_lines,
        error=error,
        error_type=error_type,
    )


def _video_asset_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Video asset rows from a result payload, across the sibling shapes.

    Tolerant multi-shape read, in the image lane's spirit but for VIDEO's
    measured shapes: the hub serves ``output.video{}`` today and the neutral
    ``artifacts[]`` list is the additive upgrade, while FAL's queue result
    carries a top-level ``video{}`` object. The first shape that yields a
    dict with a URL wins; a payload with none raises at the call site.
    """
    rows: list[dict[str, Any]] = []

    def _collect(candidate: Any) -> None:
        if isinstance(candidate, dict):
            rows.append(candidate)
        elif isinstance(candidate, list):
            rows.extend(row for row in candidate if isinstance(row, dict))

    artifacts = payload.get("artifacts")
    if isinstance(artifacts, list):
        for row in artifacts:
            if isinstance(row, dict):
                kind = str(row.get("kind") or "").strip().lower()
                if kind in ("", "video"):
                    rows.append(row)
        if rows:
            return [row for row in rows if _row_bytes_url(row)]
    _collect(payload.get("video"))
    _collect(payload.get("videos"))
    output = payload.get("output")
    if isinstance(output, dict):
        _collect(output.get("video"))
        _collect(output.get("videos"))
        if not rows and output.get("url"):
            rows.append(output)
    if not rows:
        _collect(payload.get("data"))
    # A row with no URL is not an asset: it cannot be downloaded or
    # registered, and a payload that yields none reads as "no result" at the
    # call site (invalid_response) rather than as a registration failure with
    # an invented "(no source url)" note.
    return [row for row in rows if _row_bytes_url(row)]


def _row_bytes_url(row: Mapping[str, Any]) -> str | None:
    url = row.get("url") or row.get("video_url")
    return url if isinstance(url, str) and url else None


async def _download_video_rows(
    http: httpx.AsyncClient,
    rows: list[dict[str, Any]],
    *,
    provider: str,
    label: str,
    emit: ProgressFn | None,
    pause: PauseFn | None,
    started: float,
) -> list[MediaAsset]:
    """Download every video asset row under the video cap.

    ``pause(0.0)`` before each byte-moving request: the one abort point
    between the provider completing and the bytes arriving (the image lane's
    rule, kept for the same race).
    """
    assets: list[MediaAsset] = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        url = _row_bytes_url(row)
        if not url:
            raise APIError(
                f"{label} returned a video asset without a URL.",
                status_code=None,
                code="invalid_response",
            )
        if pause is not None:
            await pause(0.0)
        elapsed = int(time.monotonic() - started)
        emit_progress(
            emit,
            f"Generating via {label}: downloading {index}/{total} — {elapsed}s",
            **progress_details(
                stage="in_progress",
                provider=provider,
                elapsed_s=elapsed,
                num_images=total,
            ),
        )
        assets.append(
            await download_asset(
                url,
                client=http,
                fallback_content_type=row.get("content_type") or "video/mp4",
                width=_num(row.get("width")),
                height=_num(row.get("height")),
                duration_s=_as_float(row.get("duration_s") or row.get("duration")),
                max_bytes=VIDEO_ARTIFACT_MAX_BYTES,
                timeout=VIDEO_DOWNLOAD_TIMEOUT,
            )
        )
    if not assets:
        raise APIError(
            f"{label} completed the job but returned no video URLs.",
            status_code=None,
            code="invalid_response",
        )
    return assets


# ---------------------------------------------------------------------------
# Radient — the hub's media tools (video)
# ---------------------------------------------------------------------------


def _pick_radient_video_model(payload: Mapping[str, Any], requested: str | None) -> str:
    """The video model id to run, read off the live list; never a pinned default.

    Same contract as the image lane's picker, with the kind filter turned to
    ``video``: a requested id wins verbatim (a bad id fails forward), else the
    first video row the hub marks ``default: true``, else the first video row.
    A list with no usable video row raises ``invalid_response`` so the rung
    fails forward with a readable reason instead of submitting a nonsense id —
    the pre-video hub (rows exist but only for images) lands exactly here.
    """
    if requested:
        return requested
    rows = payload.get("models")
    if not isinstance(rows, list):
        nested = payload.get("data")
        rows = nested if isinstance(nested, list) else []
    candidates: list[tuple[bool, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("type") or "").strip().lower() != "video":
            continue
        model_id = row.get("id") or row.get("model") or row.get("name")
        if not isinstance(model_id, str) or not model_id:
            continue
        candidates.append((bool(row.get("default")), model_id))
    for is_default, model_id in candidates:
        if is_default:
            return model_id
    if candidates:
        return candidates[0][1]
    raise APIError(
        "Radient returned no usable video model in its media list.",
        status_code=None,
        code="invalid_response",
    )


def _family_params_or_skip(
    model_id: str,
    *,
    mode: str,
    prompt: str,
    duration_s: int | None,
    resolution: str | None,
    aspect_ratio: str | None,
    audio: bool | None,
    seed: int | None,
) -> dict[str, Any]:
    """The family's wire params, or a ``RungSkipped`` carrying the refusal.

    Both rungs share the mapping (the hub forwards the same body to FAL); the
    refusal class is ``unsupported`` — a value this provider cannot represent
    is a SKIP that fails forward, never a silent substitution.
    """
    family = family_for_model(model_id)
    try:
        return map_family_params(
            family,
            prompt=prompt,
            duration_s=duration_s,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            audio=audio,
            seed=seed,
        )
    except VideoParamRejected as exc:
        raise RungSkipped(str(exc), reason_class="unsupported") from exc


async def run_radient_video(
    *,
    prompt: str,
    base_url: str,
    credential: str,
    mode: str,
    source_kind: str | None,
    source_url: str | None,
    duration_s: int | None,
    resolution: str | None,
    aspect_ratio: str | None,
    audio: bool | None,
    seed: int | None,
    model: str | None,
    handle: CancelHandle,
    emit: ProgressFn | None,
    pause: PauseFn | None,
    on_accept: Callable[[CancelHandle], Awaitable[None]] | None = None,
    client: httpx.AsyncClient | None = None,
) -> RungResult:
    """Rung 1: the Radient hub's media tools, driven for a video model.

    Source requests REFUSE before any network call (defence in depth — the
    cascade's capability filter pre-records the skip without calling): the
    hub's media route has no source mapping for video yet, and a forwarded
    ``source_url`` would be a silent, BILLED text-to-video. The enabling PR
    flips ``VIDEO_RUNG_SPECS[RADIENT].sources``/``source_kinds`` beside wiring
    the hub's frozen source words (``source_url`` / ``video_url``).

    ``on_accept`` fires once, right after the provider job exists and the
    handle is filled: the detached runner's hook for writing the durable
    ``running`` record (its ``provider_ref`` comes from the handle), so a
    process death mid-poll still leaves a cold-followable job. Best-effort —
    a failing hook is logged, never allowed to kill a live generation.
    """
    if source_kind is not None and source_url is not None:
        raise RungSkipped(RADIENT_VIDEO_SOURCE_SKIP_MESSAGE, reason_class="unsupported")
    base = base_url.rstrip("/")
    async with _client_scope(client) as http:
        models_payload = await _request_json(
            http,
            "GET",
            f"{base}{RADIENT_MEDIA_MODELS_PATH}",
            label="Radient",
            timeout_s=RADIENT_MODELS_TIMEOUT_S,
            secrets=(credential,),
            headers=_bearer_headers(credential),
        )
        model_id = _pick_radient_video_model(models_payload, model)
        body = _family_params_or_skip(
            model_id,
            mode=mode,
            prompt=prompt,
            duration_s=duration_s,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            audio=audio,
            seed=seed,
        )

        # Affordability (the image lane's probe, unit = seconds here): a
        # balance that cannot fund `unit_price × duration_s` skips; a probe
        # that cannot answer proceeds optimistically and a 402 still fails
        # forward at generate.
        unit_price = _radient_unit_price(models_payload, model_id)
        units = duration_s
        affordable: bool | None = None
        if units is not None:
            affordable = await _radient_affordable(
                http, base, credential, unit_price=unit_price, num_images=units, pause=pause
            )
        if affordable is False and unit_price is not None and units is not None:
            raise RungSkipped(
                f"Radient balance cannot fund {units} s × ${unit_price:.4f} per second.",
                reason_class="insufficient_balance",
            )

        generate = await _request_json(
            http,
            "POST",
            f"{base}{RADIENT_MEDIA_GENERATE_PATH}",
            label="Radient",
            timeout_s=HTTP_TIMEOUT_SUBMIT_S,
            secrets=(credential,),
            headers=_bearer_headers(credential),
            json={"model": model_id, **body},
        )
        request_id = generate.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise APIError(
                "Radient accepted the request but returned no request_id.",
                status_code=None,
                code="invalid_response",
            )
        handle.provider = str(VideoRoute.RADIENT)
        handle.request_id = request_id
        handle.model = model_id
        handle.base_url = base
        handle.credential = SecretStr(credential)
        await _notify_accept(on_accept, handle)
        # Submit-time QUOTE (no charge exists yet); the settled figure arrives
        # on the terminal status (`settled: true`) — same contract as images.
        cost_usd = _as_float(generate.get("cost_usd"))
        cost_basis: str | None = "estimated" if cost_usd is not None else None
        cost_provenance = (
            "Radient POST /tools/media/generate cost_usd (submit-time quote, not a settled charge)"
            if cost_usd is not None
            else None
        )
        usage_record_id: str | None = None

        started = time.monotonic()
        while True:
            await _pace_poll(pause, time.monotonic() - started)
            status_payload = await _request_json(
                http,
                "GET",
                f"{base}{RADIENT_MEDIA_STATUS_PATH}",
                label="Radient",
                timeout_s=HTTP_TIMEOUT_POLL_S,
                secrets=(credential,),
                headers=_bearer_headers(credential),
                params={"request_id": request_id},
            )
            status = str(status_payload.get("status") or "").strip().upper()
            elapsed = int(time.monotonic() - started)
            failure = _radient_failure(status_payload, status=status)
            if failure is not None:
                raise failure
            if status == "COMPLETED":
                settled_cost = _as_float(status_payload.get("cost_usd"))
                if status_payload.get("settled") is True and settled_cost is not None:
                    cost_usd = settled_cost
                    cost_basis = "billed"
                    cost_provenance = (
                        "Radient GET /tools/media/status cost_usd (settled: true; "
                        "equals the usage record and ledger)"
                    )
                settled_usage = status_payload.get("usage_record_id")
                if isinstance(settled_usage, str) and settled_usage:
                    usage_record_id = settled_usage
                break
            if status == "CANCELLED":
                raise APIError(
                    "Radient reported the generation as CANCELLED.",
                    status_code=None,
                    code="cancelled",
                )
            queue_position = _num(status_payload.get("queue_position"))
            queued = status == "IN_QUEUE"
            stage = "queued" if queued else "in_progress"
            queue_note = f", #{queue_position}" if queued and queue_position else ""
            logs = status_payload.get("logs")
            log_note = ""
            if isinstance(logs, list) and logs:
                last = logs[-1]
                if isinstance(last, dict) and isinstance(last.get("message"), str):
                    log_note = " — " + " ".join(str(last["message"]).split())[:120]
            emit_progress(
                emit,
                f"Generating video via Radient ({model_id}): "
                f"{'queued' if queued else 'running'}{queue_note} — {elapsed}s{log_note}",
                **progress_details(
                    stage=stage,
                    provider=str(VideoRoute.RADIENT),
                    model=model_id,
                    elapsed_s=elapsed,
                    num_images=1,
                    queue_position=queue_position,
                    log_lines=logs if isinstance(logs, list) else None,
                ),
            )

        result_payload = await _request_json(
            http,
            "GET",
            f"{base}{RADIENT_MEDIA_RESULT_PATH}",
            label="Radient",
            timeout_s=HTTP_TIMEOUT_RESULT_S,
            secrets=(credential,),
            headers=_bearer_headers(credential),
            params={"request_id": request_id},
        )
        # Terminal: nothing left for a cancel to do (cleared BEFORE the
        # downloads so a cancellation mid-download reads "none").
        handle.clear()
        failure = _radient_failure(result_payload)
        if failure is not None:
            raise failure
        assets = await _download_video_rows(
            http,
            _video_asset_rows(result_payload),
            provider=str(VideoRoute.RADIENT),
            label="Radient",
            emit=emit,
            pause=pause,
            started=started,
        )
        return RungResult(
            assets=assets,
            model=model_id,
            generation_id=request_id,
            cost_usd=cost_usd,
            cost_source="reported" if cost_usd is not None else None,
            billing_basis=cost_basis,
            cost_provenance=cost_provenance,
            usage_record_id=usage_record_id,
        )


# ---------------------------------------------------------------------------
# FAL — the queue API, driven natively (BYO key)
# ---------------------------------------------------------------------------

#: The FAL submit response's URL fields, with the derived fallback (the image
#: lane's derivation: the response-carried URLs are authoritative; the
#: fallback covers a response that omits them).


def _fal_video_source_field(family: VideoFamily, source_kind: str | None) -> str | None:
    """The wire field a source rides for this family, or ``None`` if it cannot.

    ``i2v`` sources go to the family's ``image`` field (``image_url`` for
    veo3.1/h3-max/gemini; ``start_image_url`` for wan-3.0 — grounded in the
    captures); ``v2v`` sources to its ``video`` field.
    """
    if source_kind is None:
        return None
    return family.source_fields.get(source_kind)


async def run_fal_video(
    *,
    prompt: str,
    key: str,
    mode: str,
    source_kind: str | None,
    source_url: str | None,
    duration_s: int | None,
    resolution: str | None,
    aspect_ratio: str | None,
    audio: bool | None,
    seed: int | None,
    model: str | None,
    handle: CancelHandle,
    emit: ProgressFn | None,
    pause: PauseFn | None,
    on_accept: Callable[[CancelHandle], Awaitable[None]] | None = None,
    base_url: str = VIDEO_FAL_QUEUE_BASE_URL,
    client: httpx.AsyncClient | None = None,
) -> RungResult:
    """Rung 2: the user's own FAL key against the queue API.

    ``sync_mode: false`` and the response-carried URLs, the image lane's
    transport exactly. The MODEL defaults per mode (:data:`FAL_DEFAULT_MODELS`)
    when the caller pins none; a pinned family that cannot serve the mode (or
    the source kind) is a capability SKIP, never a different-shape call.

    v2v sources are https URLs in v1 (the phase manager's scope ruling): the
    tool refuses anything else, and this rung refuses again here so a direct
    caller cannot smuggle a local path into ``video_url``.
    """
    base = base_url.rstrip("/")
    model_path = (model or FAL_DEFAULT_MODELS.get(mode) or "").strip().strip("/")
    if not model_path:
        raise RungSkipped(
            f"FAL has no default video model for mode {mode}; pin one with `model`.",
            reason_class="unsupported",
        )
    family = family_for_model(model_path)
    if mode not in family.modes:
        raise RungSkipped(
            f"FAL model {model_path} does not serve {mode} requests.",
            reason_class="unsupported",
        )
    source_field: str | None = None
    if source_kind is not None and source_url is not None:
        if source_kind == "video" and not source_url.startswith("https://"):
            raise RungSkipped(
                "FAL video-to-video requires an https URL source in v1 "
                "(local uploads are not supported yet).",
                reason_class="unsupported",
            )
        source_field = _fal_video_source_field(family, source_kind)
        if source_field is None:
            raise RungSkipped(
                f"FAL model {model_path} documents no {source_kind} source field.",
                reason_class="unsupported",
            )
    body = _family_params_or_skip(
        model_path,
        mode=mode,
        prompt=prompt,
        duration_s=duration_s,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        audio=audio,
        seed=seed,
    )
    if source_field is not None and source_url is not None:
        body[source_field] = source_url
    headers = {"Authorization": f"Key {key}", "Content-Type": "application/json"}
    async with _client_scope(client) as http:
        submit = await _request_json(
            http,
            "POST",
            f"{base}/{model_path}",
            label="FAL",
            timeout_s=HTTP_TIMEOUT_SUBMIT_S,
            secrets=(key,),
            headers=headers,
            json={"sync_mode": False, **body},
        )
        request_id = submit.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise APIError(
                "FAL accepted the request but returned no request_id.",
                status_code=None,
                code="invalid_response",
            )
        fallback_status, fallback_response, fallback_cancel = _fal_fallback_urls(
            base, model_path, request_id
        )

        def _url(field: str, fallback: str) -> str:
            value = submit.get(field)
            return value if isinstance(value, str) and value else fallback

        status_url = _url("status_url", fallback_status)
        response_url = _url("response_url", fallback_response)
        handle.provider = str(VideoRoute.FAL)
        handle.request_id = request_id
        handle.model = model_path
        handle.cancel_url = _url("cancel_url", fallback_cancel)
        handle.status_url = status_url
        handle.result_url = response_url
        handle.credential = SecretStr(key)
        await _notify_accept(on_accept, handle)

        started = time.monotonic()
        while True:
            await _pace_poll(pause, time.monotonic() - started)
            status_payload = await _request_json(
                http,
                "GET",
                status_url,
                label="FAL",
                timeout_s=HTTP_TIMEOUT_POLL_S,
                secrets=(key,),
                headers=headers,
            )
            status = str(status_payload.get("status") or "").strip().upper()
            elapsed = int(time.monotonic() - started)
            if status == "COMPLETED":
                break
            if status == "CANCELLED":
                raise APIError(
                    "FAL reported the generation as CANCELLED.",
                    status_code=None,
                    code="cancelled",
                )
            if status in ("FAILED", "ERROR"):
                message = status_payload.get("error")
                raise APIError(
                    (
                        message
                        if isinstance(message, str) and message
                        else "FAL reported the generation as FAILED."
                    ),
                    status_code=None,
                    code="upstream",
                )
            queue_position = _num(status_payload.get("queue_position"))
            queued = status == "IN_QUEUE"
            stage = "queued" if queued else "in_progress"
            queue_note = f", #{queue_position}" if queued and queue_position else ""
            logs = status_payload.get("logs")
            emit_progress(
                emit,
                f"Generating video via FAL ({model_path}): "
                f"{'queued' if queued else 'running'}{queue_note} — {elapsed}s",
                **progress_details(
                    stage=stage,
                    provider=str(VideoRoute.FAL),
                    model=model_path,
                    elapsed_s=elapsed,
                    num_images=1,
                    queue_position=queue_position,
                    log_lines=logs if isinstance(logs, list) else None,
                ),
            )

        result_payload = await _request_json(
            http,
            "GET",
            response_url,
            label="FAL",
            timeout_s=HTTP_TIMEOUT_RESULT_S,
            secrets=(key,),
            headers=headers,
        )
        # Terminal: nothing for a cancel to do (see the Radient rung's note).
        handle.clear()
        assets = await _download_video_rows(
            http,
            _video_asset_rows(result_payload),
            provider=str(VideoRoute.FAL),
            label="FAL",
            emit=emit,
            pause=pause,
            started=started,
        )
        return RungResult(assets=assets, model=model_path, generation_id=request_id)


# ---------------------------------------------------------------------------
# Cold-follow: poll a provider job from its durable record
# ---------------------------------------------------------------------------


async def _resolve_fal_key(config_dir: Path | None) -> str:
    key = image_availability.fal_key(config_dir)
    if not key:
        raise APIError("No FAL key is available.", status_code=None, code="unauthorized")
    return key


@dataclass(frozen=True)
class VideoPollOutcome:
    """One provider observation: the settled states, or that it still runs.

    ``state`` is one of ``running``/``completed``/``cancelled``/``failed``/
    ``gone``; ``result`` carries the downloaded assets only for a completed
    observation; ``detail`` is the provider's sentence for a failed
    classification (callers wrap it into their own sentence — raw provider
    prose never reaches a user).
    """

    state: str
    result: RungResult | None = None
    detail: str = ""


async def poll_video_once(
    *,
    record: Mapping[str, Any],
    handle: CancelHandle,
    config_dir: Path | None = None,
    base_url: str | None = None,
    store: AuthStore | None = None,
    emit: ProgressFn | None = None,
    pause: PauseFn | None = None,
    client: httpx.AsyncClient | None = None,
) -> VideoPollOutcome:
    """ONE status read of a durable record's provider job (the cold-follow unit).

    ``resume_video_observation`` loops this; ``video_status`` calls it once
    for a bounded answer and hands a still-running job to the watcher.
    Credentials are resolved HERE, at call time (the standing rung rule); a
    revoked credential raises ``unauthorized`` rather than hanging. A 404
    answers ``gone`` (the provider no longer knows the job — honest, never an
    invented outcome); a failed pair answers ``failed`` with the sentence; a
    terminal CANCELLED answers ``cancelled``. A COMPLETED job fetches its
    result and downloads the assets, so the returned ``RungResult`` is
    finished bytes, not a URL.
    """
    provider = str(record.get("provider") or "")
    provider_ref = record.get("provider_ref")
    if not isinstance(provider_ref, Mapping):
        raise APIError(
            "The record carries no provider reference to poll.",
            status_code=None,
            code="invalid_response",
        )
    request_id = str(provider_ref.get("request_id") or "")
    model_id = str(record.get("model") or "")
    if not request_id:
        raise APIError(
            "The record carries no provider request id to poll.",
            status_code=None,
            code="invalid_response",
        )
    handle.provider = provider
    handle.request_id = request_id
    handle.model = model_id or None
    started = time.monotonic()

    if provider == str(VideoRoute.RADIENT):
        from local_operator.env import resolve_radient_api_base_url

        resolved_base = resolve_radient_api_base_url(
            base_url or str(provider_ref.get("base_url") or "")
        )
        credential = await resolve_radient_credential(config_dir, resolved_base, store=store)
        bearer = credential.get_secret_value()
        if not bearer:
            raise APIError(
                "No Radient credential is available.", status_code=None, code="unauthorized"
            )
        handle.base_url = resolved_base
        handle.credential = SecretStr(bearer)
        async with _client_scope(client) as http:
            try:
                status_payload = await _request_json(
                    http,
                    "GET",
                    f"{resolved_base}{RADIENT_MEDIA_STATUS_PATH}",
                    label="Radient",
                    timeout_s=HTTP_TIMEOUT_POLL_S,
                    secrets=(bearer,),
                    headers=_bearer_headers(bearer),
                    params={"request_id": request_id},
                )
            except APIError as exc:
                if exc.status_code == 404:
                    return VideoPollOutcome(state="gone", detail="HTTP 404 from Radient")
                raise
            status = str(status_payload.get("status") or "").strip().upper()
            failure = _radient_failure(status_payload, status=status)
            if failure is not None:
                return VideoPollOutcome(state="failed", detail=str(failure))
            if status == "CANCELLED":
                return VideoPollOutcome(state="cancelled", detail="Radient reported CANCELLED")
            if status != "COMPLETED":
                queue_position = _num(status_payload.get("queue_position"))
                queued = status == "IN_QUEUE"
                logs = status_payload.get("logs")
                emit_progress(
                    emit,
                    f"Video job still running on Radient ({model_id or 'video'}): "
                    f"{status.lower() or 'running'} — {int(time.monotonic() - started)}s",
                    **progress_details(
                        stage="queued" if queued else "in_progress",
                        provider=str(VideoRoute.RADIENT),
                        model=model_id or None,
                        elapsed_s=int(time.monotonic() - started),
                        num_images=1,
                        queue_position=queue_position,
                        log_lines=logs if isinstance(logs, list) else None,
                    ),
                )
                return VideoPollOutcome(state="running")
            # COMPLETED: settle the figure and fetch the result payload.
            cost_usd = _as_float(status_payload.get("cost_usd"))
            settled = status_payload.get("settled") is True
            cost_source = "reported" if cost_usd is not None else None
            cost_basis = ("billed" if settled else "estimated") if cost_usd is not None else None
            cost_provenance = (
                (
                    "Radient GET /tools/media/status cost_usd "
                    + (
                        "(settled: true; equals the usage record and ledger)"
                        if settled
                        else "(terminal figure not flagged settled)"
                    )
                )
                if cost_usd is not None
                else None
            )
            usage_record_id: str | None = None
            settled_usage = status_payload.get("usage_record_id")
            if isinstance(settled_usage, str) and settled_usage:
                usage_record_id = settled_usage
            result_payload = await _request_json(
                http,
                "GET",
                f"{resolved_base}{RADIENT_MEDIA_RESULT_PATH}",
                label="Radient",
                timeout_s=HTTP_TIMEOUT_RESULT_S,
                secrets=(bearer,),
                headers=_bearer_headers(bearer),
                params={"request_id": request_id},
            )
            handle.clear()
            failure = _radient_failure(result_payload)
            if failure is not None:
                return VideoPollOutcome(state="failed", detail=str(failure))
            assets = await _download_video_rows(
                http,
                _video_asset_rows(result_payload),
                provider=str(VideoRoute.RADIENT),
                label="Radient",
                emit=emit,
                pause=pause,
                started=started,
            )
            return VideoPollOutcome(
                state="completed",
                result=RungResult(
                    assets=assets,
                    model=model_id,
                    generation_id=request_id,
                    cost_usd=cost_usd,
                    cost_source=cost_source,
                    billing_basis=cost_basis,
                    cost_provenance=cost_provenance,
                    usage_record_id=usage_record_id,
                ),
            )

    if provider == str(VideoRoute.FAL):
        key = await _resolve_fal_key(config_dir)
        status_url = str(provider_ref.get("status_url") or "")
        response_url = str(provider_ref.get("result_url") or "")
        if not status_url or not response_url:
            # Prefer the stored absolute URLs; derive from the model path when
            # a record predates them (same derivation as the submit path).
            derived_status, derived_response, derived_cancel = _fal_fallback_urls(
                VIDEO_FAL_QUEUE_BASE_URL, model_id, request_id
            )
            status_url = status_url or derived_status
            response_url = response_url or derived_response
            if not handle.cancel_url:
                handle.cancel_url = str(provider_ref.get("cancel_url") or derived_cancel)
        handle.cancel_url = str(provider_ref.get("cancel_url") or handle.cancel_url or "")
        handle.status_url = status_url
        handle.result_url = response_url
        handle.credential = SecretStr(key)
        headers = {"Authorization": f"Key {key}", "Content-Type": "application/json"}
        async with _client_scope(client) as http:
            try:
                status_payload = await _request_json(
                    http,
                    "GET",
                    status_url,
                    label="FAL",
                    timeout_s=HTTP_TIMEOUT_POLL_S,
                    secrets=(key,),
                    headers=headers,
                )
            except APIError as exc:
                if exc.status_code == 404:
                    return VideoPollOutcome(state="gone", detail="HTTP 404 from FAL")
                raise
            status = str(status_payload.get("status") or "").strip().upper()
            if status == "CANCELLED":
                return VideoPollOutcome(state="cancelled", detail="FAL reported CANCELLED")
            if status in ("FAILED", "ERROR"):
                message = status_payload.get("error")
                return VideoPollOutcome(
                    state="failed",
                    detail=(
                        message
                        if isinstance(message, str) and message
                        else "FAL reported the generation as FAILED."
                    ),
                )
            if status != "COMPLETED":
                queue_position = _num(status_payload.get("queue_position"))
                emit_progress(
                    emit,
                    f"Video job still running on FAL ({model_id or 'video'}): "
                    f"{status.lower() or 'running'} — {int(time.monotonic() - started)}s",
                    **progress_details(
                        stage="queued" if status == "IN_QUEUE" else "in_progress",
                        provider=str(VideoRoute.FAL),
                        model=model_id or None,
                        elapsed_s=int(time.monotonic() - started),
                        num_images=1,
                        queue_position=queue_position,
                    ),
                )
                return VideoPollOutcome(state="running")
            result_payload = await _request_json(
                http,
                "GET",
                response_url,
                label="FAL",
                timeout_s=HTTP_TIMEOUT_RESULT_S,
                secrets=(key,),
                headers=headers,
            )
            handle.clear()
            assets = await _download_video_rows(
                http,
                _video_asset_rows(result_payload),
                provider=str(VideoRoute.FAL),
                label="FAL",
                emit=emit,
                pause=pause,
                started=started,
            )
            return VideoPollOutcome(
                state="completed",
                result=RungResult(assets=assets, model=model_id, generation_id=request_id),
            )

    raise APIError(
        f"The record names an unsupported provider ({provider!r}).",
        status_code=None,
        code="invalid_response",
    )


async def resume_video_observation(
    *,
    record: Mapping[str, Any],
    handle: CancelHandle,
    config_dir: Path | None = None,
    base_url: str | None = None,
    store: AuthStore | None = None,
    pause: PauseFn | None = None,
    emit: ProgressFn | None = None,
    client: httpx.AsyncClient | None = None,
) -> RungResult:
    """Poll a durable record's provider job TO TERMINAL; return its result.

    The cold-follow watcher path (contract §1.4/§4.3): a successor process
    drives this with ONLY the record, looping :func:`poll_video_once` with the
    abortable pace. Raises the same failure classes the rungs raise
    (``APIError`` with a frozen code, ``TimeoutError`` past the observation
    bound) so callers settle the record from one classifier. ``pause`` is the
    caller's abort-aware wait (the watcher job's own signal); the ``handle``
    is caller-owned so cancelling the WATCHER can best-effort-cancel the
    provider job it is watching.
    """
    started = time.monotonic()
    while True:
        outcome = await poll_video_once(
            record=record,
            handle=handle,
            config_dir=config_dir,
            base_url=base_url,
            store=store,
            emit=emit,
            pause=pause,
            client=client,
        )
        if outcome.state == "completed":
            if outcome.result is None:  # pragma: no cover - impossible by construction
                raise APIError(
                    "The provider completed without a readable result.",
                    status_code=None,
                    code="invalid_response",
                )
            return outcome.result
        if outcome.state == "cancelled":
            raise APIError(
                outcome.detail or "The provider reported the generation as CANCELLED.",
                status_code=None,
                code="cancelled",
            )
        if outcome.state == "failed":
            raise APIError(
                outcome.detail or "The provider reported the generation as FAILED.",
                status_code=None,
                code="upstream",
            )
        if outcome.state == "gone":
            raise APIError(
                PROVIDER_GONE_NOTE,
                status_code=404,
                code="not_found",
            )
        elapsed = time.monotonic() - started
        if elapsed >= VIDEO_OBSERVATION_TIMEOUT_S:
            raise TimeoutError(
                "the provider job did not settle within "
                f"{int(VIDEO_OBSERVATION_TIMEOUT_S)}s of observation"
            )
        await _pace_poll(pause, elapsed)
