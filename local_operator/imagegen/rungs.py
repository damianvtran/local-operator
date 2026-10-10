"""httpx-async rung executors for the image cascade, and the provider cancel.

**D6 (frozen design): native asyncio, no threads, no subprocesses.** Every long
wait here is an ``await`` on a socket or on ``asyncio.sleep`` — so asyncio
cancellation is delivered at a suspension point and cancelling mid-``GET``
closes the socket, which is what actually releases the provider connection
(``harness/loop.py``'s own doctrine). ``asyncio.to_thread`` is deliberately
NOT used: a thread cannot be cancelled, which would leave a generation running
to completion after Esc.

**D7 (frozen design): nothing registers with ``group_reaper``, and there is no
second reap concept.** That module tracks detached PROCESS GROUPS spawned by
``bash`` and keys on owner liveness; this tool spawns no process and no
orphanable group — its work is in-process HTTP bounded by the per-call
timeouts below and cancelled via asyncio. A SIGKILLed runtime leaves a
provider-side job that simply completes uncollected; that is identical to any
client disconnect, accepted, and documented here rather than papered over with
a reaper registration that would have nothing to kill.

**Wire shapes.** Radient's media rung is the lane-verified v0: models list,
generate, status, result, cancel, all under ``/v1/tools/media/*`` with a
bearer. FAL rides its queue API natively (``sync_mode: false``) and we use the
response-carried ``status_url``/``response_url``/``cancel_url`` — with
fallback derivation from the app path, which reproduces the OLD client's
hardcoded ``[redacted]…`` exactly (verified: ``clients/fal.py:170,198``
hardcode ``fal-ai/flux`` while posting to ``model_path`` — a latent bug for
any other model; the response-carried URLs are the fix, and the fallback
derives from the model actually used). OpenAI is a single synchronous request
with no provider-side cancel at all.

**Every bound and interval below is a named constant with its reason** (design
§9): the numbers are tuning guesses anchored to observed model latencies, and
naming them is what makes retuning a one-line change instead of an archaeology
project.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import time
from collections.abc import AsyncIterator
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
from local_operator.imagegen import ImageRoute, MediaAsset
from local_operator.imagegen.errors import api_error_from_httpx_response
from local_operator.imagegen.media import download_asset

__all__ = [
    "CancelHandle",
    "ProgressFn",
    "PauseFn",
    "RungResult",
    "RungSkipped",
    "best_effort_cancel",
    "emit_progress",
    "progress_details",
    "run_google",
    "run_openai_sub",
    "run_openrouter",
    "run_xai",
]

logger = logging.getLogger(__name__)

#: One poll tick while the provider works. Cheap free reads; 2 s keeps the
#: tool's live progress rows honest without hammering the queue.
IMAGE_POLL_INTERVAL_S = 2.0
#: After this long a generation is expected to be model-inference-bound, not
#: queue-bound, so the poll relaxes (design §9: ~model latencies observed).
IMAGE_POLL_SLOW_AFTER_S = 30.0
IMAGE_POLL_SLOW_INTERVAL_S = 4.0

#: Per-HTTP-call bounds. Submit carries the largest body; polls and result
#: reads are small metadata payloads; downloads are the one place a big body
#: legitimately moves.
HTTP_TIMEOUT_SUBMIT_S = 30.0
HTTP_TIMEOUT_POLL_S = 10.0
HTTP_TIMEOUT_RESULT_S = 15.0
#: The two free reads, kept short — a probe that cannot answer quickly must
#: not delay a call that could proceed optimistically (see the affordability
#: rule below).
RADIENT_MODELS_TIMEOUT_S = 5.0
RADIENT_CAPACITY_TIMEOUT_S = 5.0
#: OpenAI's images API is a single synchronous request (no queue to poll) and
#: gpt-image-1 class models can take tens of seconds; this is its whole bound.
OPENAI_IMAGE_TIMEOUT_S = 120.0

#: The best-effort provider cancel's TOTAL budget (connect 2 s / read 3 s):
#: Esc stays snappy, and a second Esc can abandon the cleanup (the function
#: below states the mechanic).
CANCEL_TIMEOUT_TOTAL_S = 5.0

#: FAL's default app path, mirroring the legacy client's constant. The only
#: place a FAL model id is written down, and it is a DEFAULT, not a pin — the
#: ``model`` parameter overrides it and no list of ids is cached anywhere.
FAL_DEFAULT_MODEL = "fal-ai/flux/dev"

#: How the cascade lets a rung wait abortably: ``await pause(seconds)`` returns
#: early (raising ``ImageGenerationCancelled``) when the user's abort signal
#: fires, so the rare no-cancellation race becomes a clean receipt instead of
#: a turn that waits out the provider. ``None`` in tests and library use.
# ``PauseFn`` itself now lives in :mod:`local_operator.artifacts.walk`
# (imported above); the protocol is unchanged.


def _poll_interval(elapsed_s: float) -> float:
    """2 s while queue-bound, relaxing to 4 s once the job is inference-bound."""
    if elapsed_s >= IMAGE_POLL_SLOW_AFTER_S:
        return IMAGE_POLL_SLOW_INTERVAL_S
    return IMAGE_POLL_INTERVAL_S


#: The no-signal poll ceiling (round-1 finding): a caller that can observe
#: nothing — no progress consumer, no abort signal, so ``pause is None`` —
#: still WAITS between status reads, backing off by doubling from the base
#: interval to this cap. Without it the loop hot-polls (the pace only ever
#: happened inside ``pause``), hammering a provider for no one's benefit;
#: bounded so the wait can never approach the rung budget, which owns the end.
IMAGE_POLL_NO_SIGNAL_CAP_S = 8.0


def _no_signal_poll_interval(elapsed_s: float) -> float:
    """2 s -> 4 s -> 8 s(cap) for callers nothing is watching."""
    return min(
        IMAGE_POLL_INTERVAL_S * (2 ** int(elapsed_s // IMAGE_POLL_SLOW_AFTER_S)),
        IMAGE_POLL_NO_SIGNAL_CAP_S,
    )


async def _pace_poll(pause: "PauseFn | None", elapsed_s: float) -> None:
    """Wait between poll reads — abortably when a signal exists, paced when not.

    The wait is where cancellation lands, so a signal always takes the pause
    path; without one the bounded back-off keeps a library/headless poller
    from hammering the provider with back-to-back reads.
    """
    if pause is not None:
        await pause(_poll_interval(elapsed_s))
    else:
        await asyncio.sleep(_no_signal_poll_interval(elapsed_s))


def _asset_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Asset rows from a result payload, across the sibling key spellings.

    Tolerant multi-key read on purpose: the v0 lane contract is "asset URLs
    plus optional ``content_type``/``width``/``height``/``duration``" without a
    frozen envelope, and a strict single-key read would turn a lane-side
    rename into a total outage. The first key that yields dict rows wins;
    "output" may nest its items under ``images``/``assets``.
    """
    for key in ("images", "assets", "output", "results", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            rows = [row for row in value if isinstance(row, dict)]
            if rows:
                return rows
        if isinstance(value, dict):
            for sub in ("images", "assets"):
                nested = value.get(sub)
                if isinstance(nested, list):
                    rows = [row for row in nested if isinstance(row, dict)]
                    if rows:
                        return rows
    return []


async def _download_rows(
    http: httpx.AsyncClient,
    rows: list[dict[str, Any]],
    *,
    provider: ImageRoute,
    label: str,
    emit: ProgressFn | None,
    pause: PauseFn | None,
    started: float,
) -> list[MediaAsset]:
    """Download every asset row (bounded per ``media.download_asset``).

    ``pause(0.0)`` before each byte-moving request: that is the one abort
    point available between the provider completing and the bytes arriving,
    and it keeps the no-cancellation race honest even mid-download.
    """
    assets: list[MediaAsset] = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        url = row.get("url")
        if not isinstance(url, str) or not url:
            raise APIError(
                f"{label} returned an asset without a URL.",
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
                provider=str(provider),
                elapsed_s=elapsed,
                num_images=total,
            ),
        )
        assets.append(
            await download_asset(
                url,
                client=http,
                fallback_content_type=row.get("content_type"),
                width=_num(row.get("width")),
                height=_num(row.get("height")),
                duration_s=_as_float(row.get("duration_s") or row.get("duration")),
            )
        )
    if not assets:
        raise APIError(
            f"{label} completed the job but returned no asset URLs.",
            status_code=None,
            code="invalid_response",
        )
    return assets


#: OpenAI's default image model (the ``gpt-image-1`` class). Same rule: a
#: default, never a pin.
OPENAI_DEFAULT_IMAGE_MODEL = "gpt-image-1"
OPENAI_IMAGE_BASE_URL = "https://api.openai.com/v1"

# ``RungSkipped``, ``CancelHandle`` and ``RungResult`` — with ``ProgressFn``/
# ``PauseFn`` beside them — moved to :mod:`local_operator.artifacts` in media
# wave-2 (the kind-neutral rung seam) and are imported above under these exact
# names, so the lane's pinned imports (tests, the tool) keep resolving here.

# ---------------------------------------------------------------------------
# Shared HTTP plumbing
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _client_scope(client: httpx.AsyncClient | None) -> AsyncIterator[httpx.AsyncClient]:
    """The caller's injected client, or one this rung owns and closes.

    The injected-client seam is how tests attach ``httpx.MockTransport``
    without a network (the ``stt/clients.py`` convention); production call
    sites leave it ``None`` and pay one client per rung.
    """
    if client is not None:
        yield client
        return
    async with httpx.AsyncClient() as owned:
        yield owned


async def _read_json(response: httpx.Response, *, label: str) -> dict[str, Any]:
    """A 2xx response's JSON object, or an ``invalid_response`` failure."""
    try:
        payload = response.json()
    except ValueError as exc:
        raise APIError(
            f"{label} returned a response that is not JSON (HTTP {response.status_code}).",
            status_code=response.status_code,
            code="invalid_response",
        ) from exc
    if not isinstance(payload, dict):
        raise APIError(
            f"{label} returned a {type(payload).__name__} where a JSON object was expected.",
            status_code=response.status_code,
            code="invalid_response",
        )
    return payload


async def _request_json(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    label: str,
    timeout_s: float,
    secrets: tuple[str, ...] = (),
    **kwargs: Any,
) -> dict[str, Any]:
    """One bounded JSON request, with this package's error shape on failure.

    Transport failures and timeouts are stamped with explicit ``code``s so the
    attempt records can say "timeout" vs "network" without text-matching; a
    >=400 response goes through the shared httpx mapper (which keeps the
    upstream's own sentence, scrubbed).
    """
    try:
        response = await client.request(method, url, timeout=httpx.Timeout(timeout_s), **kwargs)
    except httpx.TimeoutException as exc:
        raise APIError(
            f"{label} timed out after {timeout_s:.0f}s.", status_code=None, code="timeout"
        ) from exc
    except httpx.HTTPError as exc:
        # A transport failure never reached the upstream: no status, no body.
        raise APIError(f"{label} request failed: {exc}", status_code=None, code="network") from exc
    if response.status_code >= 400:
        raise api_error_from_httpx_response(
            response, fallback_message=f"{label} refused the request", secrets=secrets
        )
    return await _read_json(response, label=label)


def _as_float(value: Any) -> float | None:
    """A number from a payload field, or ``None`` — never raises, never bools."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _num(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ``emit_progress`` moved to :mod:`local_operator.artifacts.progress` (imported
# above) and is re-exported here under its exact name: the lane's own tests
# call it from this module, and it is the one guarded spelling for "progress is
# presentation, never control flow".


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
    """The canonical payload every ``generate_image`` update carries.

    This lane's binding of the generic builder: the frozen ``tool_name`` slot
    is pinned to THIS tool here, so every call site in the lane keeps today's
    exact signature (the pinned call sites in tests, the rungs' update lines,
    the tool's terminal stages). Field contract, constraints and the stage
    vocabulary: :func:`local_operator.artifacts.progress.progress_details`.
    """
    return _generic_progress_details(
        tool="generate_image",
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


# ---------------------------------------------------------------------------
# Radient — the hub's media tools
# ---------------------------------------------------------------------------
#
# v0 payload notes (lane-verified): generate/status/result/cancel are flat
# passthroughs except ``model``/``provider``, so the params model maps
# straight onto the wire; the model list is read LIVE per call and no model id
# is pinned anywhere in this tree (the default is whichever image model the
# hub currently marks ``default: true``); ``cost_usd`` comes back on generate
# and is surfaced, not budgeted against.

#: Where the hub's media routes hang. Paths, not hosts: the host is the
#: resolved Radient base URL, so a staging hub needs no code change.
RADIENT_MEDIA_MODELS_PATH = "/tools/media/models"
RADIENT_MEDIA_GENERATE_PATH = "/tools/media/generate"
RADIENT_MEDIA_STATUS_PATH = "/tools/media/status"
RADIENT_MEDIA_RESULT_PATH = "/tools/media/result"
RADIENT_MEDIA_CANCEL_PATH = "/tools/media/cancel"
RADIENT_CAPACITY_PATH = "/me/billing-sources/capacity"

#: The hub's structured failure vocabulary (``error_type`` on a FAILED status).
#: The cascade switches on THESE TOKENS and the HTTP status, never on the
#: ``error`` prose: the sentence is the platform's and FAL's free text is never
#: surfaced through it (manager freeze note, 2026-10-08). Carried verbatim
#: into attempt records so a consumer can group failures without parsing.
RADIENT_ERROR_TYPES = frozenset(
    {"media_rejected", "media_failed", "media_rate_limited", "media_unavailable"}
)


def _bearer_headers(credential: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {credential}", "Content-Type": "application/json"}


def _pick_radient_model(payload: dict[str, Any], requested: str | None) -> str:
    """The model id to run, read off the live list; never a pinned default.

    ``requested`` wins verbatim — the caller may pin any id the hub serves,
    and a bad id fails at generate and fails FORWARD (rung failure), which is
    the honest outcome for a free-text field. With none requested, the first
    image model the hub marks ``default: true`` wins; failing that, the first
    image model listed. A list with no usable image row raises
    ``invalid_response`` so the rung fails forward with a readable reason
    instead of submitting a nonsense model id.
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
        if str(row.get("type") or "image").strip().lower() not in ("", "image"):
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
        "Radient returned no usable image model in its media list.",
        status_code=None,
        code="invalid_response",
    )


def _radient_unit_price(payload: dict[str, Any], model_id: str) -> float | None:
    """The listed unit price for ``model_id``, when the row carries one.

    Tolerant on purpose (``unit_price_usd`` is the design's spelling; the two
    sibling spellings are accepted so a lane-side rename degrades to "cannot
    price" — the optimistic path) and NEVER raises: a price the probe cannot
    read must not strand a working rung.
    """
    rows = payload.get("models")
    if not isinstance(rows, list):
        nested = payload.get("data")
        rows = nested if isinstance(nested, list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_name = row.get("id") or row.get("model") or row.get("name")
        if model_name != model_id:
            continue
        for key in ("unit_price_usd", "price_usd", "price"):
            price = _as_float(row.get(key))
            if price is not None:
                return price
    return None


async def _radient_affordable(
    client: httpx.AsyncClient,
    base: str,
    credential: str,
    *,
    unit_price: float | None,
    num_images: int,
    pause: PauseFn | None,
) -> bool | None:
    """Whether the account can fund this request; ``None`` = probe could not answer.

    **A probe that cannot answer must not strand a working rung** (frozen
    design §3.3): a network failure, an older backend without the capacity
    route (404) or an unreadable payload all read as "proceed optimistically"
    — a 402 at generate still fails forward. An answered probe compares
    ``total_balance`` against ``unit_price_usd × num_images``; no listed
    price means no comparison (None).
    """
    if unit_price is None:
        return None
    if pause is not None:
        await pause(0.0)
    try:
        response = await client.get(
            f"{base.rstrip('/')}{RADIENT_CAPACITY_PATH}",
            headers=_bearer_headers(credential),
            timeout=httpx.Timeout(RADIENT_CAPACITY_TIMEOUT_S),
        )
        if response.status_code != 200:
            return None
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    balance = _as_float(payload.get("total_balance"))
    if balance is None:
        # Some deployments wrap the row. Measured live 2026-10-08: the
        # production hub answers {"msg": ..., "result": {"total_balance": ...}}
        # -- the result envelope. "data" is its sibling shape; the read stays
        # tolerant on purpose (a shape this probe cannot parse must fall back
        # to the optimistic path, never strand a working rung).
        for key in ("result", "data"):
            row = payload.get(key)
            if isinstance(row, dict):
                balance = _as_float(row.get("total_balance"))
                if balance is not None:
                    break
    if balance is None:
        return None
    return balance >= unit_price * num_images


async def run_radient(
    *,
    prompt: str,
    base_url: str,
    credential: str,
    num_images: int,
    image_size: str,
    seed: int | None,
    strength: float | None,
    source_url: str | None,
    model: str | None,
    handle: CancelHandle,
    emit: ProgressFn | None,
    pause: PauseFn | None,
    client: httpx.AsyncClient | None = None,
) -> RungResult:
    """Rung 1: the Radient hub's media tools. See the module docstring.

    ``credential`` is the bearer the cascade resolved (``SecretStr`` unwrapped
    at the boundary); ``source_url`` is the img2img data URI when the caller
    supplied one. The walk: live model list → affordability (optimistic on any
    probe failure) → submit → poll → result → bounded downloads.
    """
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
        model_id = _pick_radient_model(models_payload, model)
        unit_price = _radient_unit_price(models_payload, model_id)

        affordable = await _radient_affordable(
            http, base, credential, unit_price=unit_price, num_images=num_images, pause=pause
        )
        if affordable is False:
            assert unit_price is not None  # pragma: no cover - implied by the probe
            raise RungSkipped(
                f"Radient balance cannot fund {num_images} × ${unit_price:.4f} per image.",
                reason_class="insufficient_balance",
            )

        body: dict[str, Any] = {
            "model": model_id,
            "prompt": prompt,
            "num_images": num_images,
            "image_size": image_size,
        }
        if seed is not None:
            body["seed"] = seed
        if strength is not None:
            body["strength"] = strength
        if source_url is not None:
            body["source_url"] = source_url

        generate = await _request_json(
            http,
            "POST",
            f"{base}{RADIENT_MEDIA_GENERATE_PATH}",
            label="Radient",
            timeout_s=HTTP_TIMEOUT_SUBMIT_S,
            secrets=(credential,),
            headers=_bearer_headers(credential),
            json=body,
        )
        request_id = generate.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise APIError(
                "Radient accepted the request but returned no request_id.",
                status_code=None,
                code="invalid_response",
            )
        handle.provider = ImageRoute.RADIENT
        handle.request_id = request_id
        handle.model = model_id
        handle.base_url = base
        handle.credential = SecretStr(credential)
        cost_usd = _as_float(generate.get("cost_usd"))

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
                # request_id ONLY: no model param, and an unknown/foreign id
                # answers an identical 404 (non-disclosure; manager freeze
                # note, 2026-10-08).
                params={"request_id": request_id},
            )
            status = str(status_payload.get("status") or "").strip().upper()
            elapsed = int(time.monotonic() - started)
            if status == "COMPLETED":
                break
            if status == "CANCELLED":
                raise APIError(
                    "Radient reported the generation as CANCELLED.",
                    status_code=None,
                    code="cancelled",
                )
            if status in ("FAILED", "ERROR"):
                # Switch on error_type, never on prose. Anything outside the
                # frozen vocabulary classifies as an upstream failure.
                error_type = status_payload.get("error_type")
                code = (
                    error_type
                    if isinstance(error_type, str) and error_type in RADIENT_ERROR_TYPES
                    else "upstream"
                )
                message = status_payload.get("error")
                raise APIError(
                    (
                        message
                        if isinstance(message, str) and message
                        else "Radient reported the generation as FAILED."
                    ),
                    status_code=None,
                    code=code,
                )
            queue_position = _num(status_payload.get("queue_position"))
            queued = status == "IN_QUEUE"
            # Wire value vs display word (Q7 split): ``stage`` carries the
            # canonical ``in_progress``; the line keeps the friendly "running".
            stage = "queued" if queued else "in_progress"
            queue_note = f", #{queue_position}" if queued and queue_position else ""
            log_note = ""
            logs = status_payload.get("logs")
            log_lines = logs if isinstance(logs, list) else None
            if isinstance(logs, list) and logs:
                last = logs[-1]
                if isinstance(last, dict) and isinstance(last.get("message"), str):
                    log_note = " — " + " ".join(str(last["message"]).split())[:120]
            emit_progress(
                emit,
                f"Generating via Radient ({model_id}): "
                f"{'queued' if queued else 'running'}{queue_note} — {elapsed}s{log_note}",
                **progress_details(
                    stage=stage,
                    provider=str(ImageRoute.RADIENT),
                    model=model_id,
                    elapsed_s=elapsed,
                    num_images=num_images,
                    queue_position=queue_position,
                    log_lines=log_lines,
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
            # request_id only — same freeze note as the status read above.
            params={"request_id": request_id},
        )
        # Terminal: nothing left for a cancel to do. Cleared BEFORE the
        # downloads so a cancellation mid-download reads "none" rather than
        # firing an ALREADY_COMPLETED round-trip.
        handle.clear()
        assets = await _download_rows(
            http,
            _asset_rows(result_payload),
            provider=ImageRoute.RADIENT,
            label="Radient",
            emit=emit,
            pause=pause,
            started=started,
        )
        return RungResult(
            assets=assets, model=model_id, generation_id=request_id, cost_usd=cost_usd
        )


# ---------------------------------------------------------------------------
# FAL — the queue API, driven natively (the fal_client SDK is deliberately not
# added: the queue flow is four HTTP calls and a poll)
# ---------------------------------------------------------------------------

#: FAL's queue host, mirroring the legacy client's default.
FAL_QUEUE_BASE_URL = "https://queue.fal.run"


def _fal_fallback_urls(base: str, model_path: str, request_id: str) -> tuple[str, str, str]:
    """(status, response, cancel) URLs derived from the app path.

    The FALLBACK only — the submit response normally carries all three. The
    derivation is ``{base}/{app_root}/requests/{id}[/status|/cancel]`` where
    ``app_root = model_path.rsplit("/", 1)[0]``, which reproduces the old
    client's hardcoded value for its default app path while FIXING it for
    every other path: ``clients/fal.py`` posted to the caller's ``model_path``
    but polled a hardcoded app root, so any non-default model was polled at
    the wrong app (a latent bug, verified 2026-10-08; the response-carried
    URLs are the real fix and this is the degraded path behind them).
    """
    app_root = model_path.rsplit("/", 1)[0] if "/" in model_path else model_path
    root = f"{base}/{app_root}/requests/{request_id.strip('/')}"
    return f"{root}/status", root, f"{root}/cancel"


async def run_fal(
    *,
    prompt: str,
    key: str,
    num_images: int,
    image_size: str,
    seed: int | None,
    strength: float | None,
    source_url: str | None,
    model: str | None,
    handle: CancelHandle,
    emit: ProgressFn | None,
    pause: PauseFn | None,
    base_url: str = FAL_QUEUE_BASE_URL,
    client: httpx.AsyncClient | None = None,
) -> RungResult:
    """Rung 2: the user's own FAL key against the queue API.

    ``sync_mode: false`` because we drive the queue natively — FAL's own
    recommended async flow — and the response-carried URLs are used verbatim
    (see :func:`_fal_fallback_urls` for the degraded path).

    IMG2IMG rides FAL's dedicated ``/image-to-image`` app route with the field
    spelled ``image_url`` (not ``source_url``): that conversion is the legacy
    client's own, kept because it is the wire FAL documents for flux-class
    apps — the old code appended ``GenerationType.IMAGE_TO_IMAGE.value`` for
    exactly this reason.
    """
    base = base_url.rstrip("/")
    model_path = (model or FAL_DEFAULT_MODEL).strip().strip("/")
    headers = {"Authorization": f"Key {key}", "Content-Type": "application/json"}
    async with _client_scope(client) as http:
        body: dict[str, Any] = {"prompt": prompt, "num_images": num_images, "sync_mode": False}
        if source_url is not None:
            if not model_path.endswith("/image-to-image"):
                model_path = f"{model_path}/image-to-image"
            body["image_url"] = source_url
            if strength is not None:
                body["strength"] = strength
        else:
            body["image_size"] = image_size
        if seed is not None:
            body["seed"] = seed

        submit = await _request_json(
            http,
            "POST",
            f"{base}/{model_path}",
            label="FAL",
            timeout_s=HTTP_TIMEOUT_SUBMIT_S,
            secrets=(key,),
            headers=headers,
            json=body,
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
        handle.provider = ImageRoute.FAL
        handle.request_id = request_id
        handle.model = model_path
        handle.cancel_url = _url("cancel_url", fallback_cancel)
        handle.credential = SecretStr(key)

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
                f"Generating via FAL ({model_path}): "
                f"{'queued' if queued else 'running'}{queue_note} — {elapsed}s",
                **progress_details(
                    stage=stage,
                    provider=str(ImageRoute.FAL),
                    model=model_path,
                    elapsed_s=elapsed,
                    num_images=num_images,
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
        assets = await _download_rows(
            http,
            _asset_rows(result_payload),
            provider=ImageRoute.FAL,
            label="FAL",
            emit=emit,
            pause=pause,
            started=started,
        )
        return RungResult(assets=assets, model=model_path, generation_id=request_id)


# ---------------------------------------------------------------------------
# OpenAI — one synchronous request; NO provider-side cancel exists
# ---------------------------------------------------------------------------

OPENAI_IMAGES_PATH = "/images/generations"


def openai_size(image_size: str, model_id: str) -> str:
    """Map the FAL-shaped size enum onto OpenAI's size strings (design §3.3).

    Portrait/landscape square-HD-shaped values share one mapping per family
    and every ``square*`` value is the square size; the dall-e class takes the
    1792-family, the gpt-image class the 1536 one. Unknown values fall to the
    square default rather than raising: a size the caller invented is not
    worth refusing before the provider has had its say.
    """
    portrait = image_size.startswith("portrait")
    landscape = image_size.startswith("landscape")
    if model_id.startswith("dall-e"):
        if portrait:
            return "1024x1792"
        if landscape:
            return "1792x1024"
        return "1024x1024"
    if portrait:
        return "1024x1536"
    if landscape:
        return "1536x1024"
    return "1024x1024"


async def run_openai(
    *,
    prompt: str,
    key: str,
    num_images: int,
    image_size: str,
    source_url: str | None,
    model: str | None,
    emit: ProgressFn | None,
    pause: PauseFn | None,
    base_url: str = OPENAI_IMAGE_BASE_URL,
    client: httpx.AsyncClient | None = None,
) -> RungResult:
    """Rung 3: the user's own OpenAI API key.

    A single synchronous request bounded by :data:`OPENAI_IMAGE_TIMEOUT_S` —
    there is no queue to poll and **no provider-side cancel exists**: an abort
    discards our wait, but the server may still complete the generation and
    bill for it. That asymmetry is documented (here, the tool doc and the
    guide) rather than papered over; the cancel handle is never set for this
    rung, so a best-effort cancel reports "none — nothing to cancel".

    IMG2IMG is SKIPPED with a recorded reason (v1 scope): the design routes
    edits through Radient's ``source_url`` passthrough and FAL's app route,
    and there is no image-edit path in this rung.

    ``seed``/``strength`` are not part of OpenAI's images wire and are dropped
    silently — the design's rule is "passed only to providers that support
    it", and a model that never received a field cannot honour it.
    """
    if source_url is not None:
        raise RungSkipped(
            "OpenAI has no image-to-image route in this rung.", reason_class="unsupported"
        )
    model_id = (model or OPENAI_DEFAULT_IMAGE_MODEL).strip()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": model_id,
        "prompt": prompt,
        "n": num_images,
        "size": openai_size(image_size, model_id),
    }
    async with _client_scope(client) as http:
        started = time.monotonic()
        payload = await _request_json(
            http,
            "POST",
            f"{base_url.rstrip('/')}{OPENAI_IMAGES_PATH}",
            label="OpenAI images",
            timeout_s=OPENAI_IMAGE_TIMEOUT_S,
            secrets=(key,),
            headers=headers,
            json=body,
        )
        items = payload.get("data")
        if not isinstance(items, list) or not items:
            raise APIError(
                "OpenAI returned no image data.", status_code=None, code="invalid_response"
            )
        assets: list[MediaAsset] = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            b64 = item.get("b64_json")
            if isinstance(b64, str) and b64:
                try:
                    data = base64.b64decode(b64, validate=False)
                except (ValueError, TypeError) as exc:
                    raise APIError(
                        "OpenAI returned image data that is not valid base64.",
                        status_code=None,
                        code="invalid_response",
                    ) from exc
                # The images API serves PNG bytes for b64 items and gives no
                # content type or dimensions; cache_media sniffs dims, and PNG
                # is the documented container for both response forms.
                assets.append(MediaAsset(data=data, content_type="image/png", source_url=""))
                continue
            url = item.get("url")
            if isinstance(url, str) and url:
                if pause is not None:
                    await pause(0.0)
                elapsed = int(time.monotonic() - started)
                emit_progress(
                    emit,
                    f"Generating via OpenAI ({model_id}): downloading {index}/{len(items)} — "
                    f"{elapsed}s",
                    **progress_details(
                        stage="in_progress",
                        provider=str(ImageRoute.OPENAI),
                        model=model_id,
                        elapsed_s=elapsed,
                        num_images=len(items),
                    ),
                )
                assets.append(await download_asset(url, client=http))
        if not assets:
            raise APIError(
                "OpenAI returned no usable image entries.",
                status_code=None,
                code="invalid_response",
            )
        return RungResult(assets=assets, model=model_id)


# ---------------------------------------------------------------------------
# Best-effort provider cancel — the Esc path
# ---------------------------------------------------------------------------
#
# The bounded cleanup behind the tool's ``except asyncio.CancelledError`` and
# its no-cancellation-race receipt. It is deliberately NOT a new cancellation
# mechanism (the loop owns cancellation; this only tells the PROVIDER to stop
# a job that would otherwise run and bill to completion), and deliberately NOT
# a reaper registration (D7 — nothing here spawns a process or an orphanable
# group; see the module docstring).


def _parse_cancel_body(response: httpx.Response) -> str:
    """The cancel response's status token, from JSON or raw text; upper-cased.

    Tolerant on purpose — the settled vocabulary is a small closed set
    (``CANCELLED`` / ``ALREADY_COMPLETED``) and providers spell it in a body
    or a JSON field depending on the route; a body neither reads nor parses
    just fails the marker test and the HTTP status decides.
    """
    try:
        payload = response.json()
        if isinstance(payload, dict):
            token = payload.get("status") or payload.get("state") or payload.get("result")
            if isinstance(token, str):
                return token.upper()
    except ValueError:
        pass
    return (response.text or "").upper()


async def best_effort_cancel(handle: CancelHandle | None) -> str:
    """Ask the provider to cancel the in-flight job; NEVER raises, never blocks long.

    Returns one of: ``cancelled`` (the job is (becoming) cancelled),
    ``already_completed`` (nothing left to do), ``not_found`` (the id is gone —
    settled for our purposes), ``none`` (no provider job to cancel — including
    the OpenAI rung, which has no cancel route at all), ``timeout``,
    ``failed``, or ``abandoned`` (a second cancellation arrived; see below).

    The 5 s budget (:data:`CANCEL_TIMEOUT_TOTAL_S`, connect 2 s / read 3 s)
    keeps Esc snappy. The attempt runs under ``asyncio.shield`` so a SECOND
    cancellation (double-Esc) does not tear the inbound cancel request down
    mid-flight: the shield lets it run to its own bound while the outer await
    abandons silently. Abandoning never masks the cancellation already
    delivered — this coroutine is called from inside that cancellation's
    handler, which re-raises afterwards regardless.
    """
    if handle is None or handle.provider is None or handle.request_id is None:
        return "none"
    provider = handle.provider
    credential = handle.credential

    async def _attempt() -> str:
        timeout = httpx.Timeout(3.0, connect=2.0)
        try:
            async with _client_scope(None) as http:
                if provider == ImageRoute.FAL:
                    if not handle.cancel_url:
                        return "none"
                    headers = (
                        {"Authorization": f"Key {credential.get_secret_value()}"}
                        if credential is not None
                        else {}
                    )
                    response = await http.request(
                        "PUT", handle.cancel_url, headers=headers, timeout=timeout
                    )
                elif provider == ImageRoute.RADIENT:
                    if not handle.base_url:
                        return "none"
                    headers = (
                        _bearer_headers(credential.get_secret_value())
                        if credential is not None
                        else {"Content-Type": "application/json"}
                    )
                    response = await http.post(
                        f"{handle.base_url.rstrip('/')}{RADIENT_MEDIA_CANCEL_PATH}",
                        headers=headers,
                        # request_id ONLY (freeze note, 2026-10-08): a foreign
                        # or unknown id answers an identical 404 — and all
                        # three settled outcomes count as done.
                        json={"request_id": handle.request_id},
                        timeout=timeout,
                    )
                else:
                    return "none"
        except httpx.HTTPError:
            return "failed"
        if response.status_code == 404:
            # Identical for unknown and foreign ids (non-disclosure); for a
            # receipt this is settled: nothing is left to cancel.
            return "not_found"
        token = _parse_cancel_body(response)
        if "ALREADY_COMPLETED" in token:
            return "already_completed"
        if 200 <= response.status_code < 300:
            # FAL answers 202 CANCELLATION_REQUESTED; Radient answers 2xx with
            # a CANCELLED (or ALREADY_COMPLETED, handled above) settlement.
            # Any other 2xx is the cancel request ACCEPTED, which is the
            # honest outcome for a best-effort receipt.
            return "cancelled"
        if "CANCELLED" in token:
            # Some deployments report the settled state on a non-2xx code.
            return "cancelled"
        return "failed"

    async def _guarded() -> str:
        try:
            return await asyncio.wait_for(_attempt(), timeout=CANCEL_TIMEOUT_TOTAL_S)
        except TimeoutError:
            return "timeout"
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a cleanup must never raise
            logger.debug("provider cancel failed", exc_info=True)
            return "failed"

    try:
        return await asyncio.shield(_guarded())
    except asyncio.CancelledError:
        return "abandoned"
    except BaseException:  # noqa: BLE001 - incl. GeneratorExit; never raise from cleanup
        return "failed"


# ---------------------------------------------------------------------------
# Re-exported executors (media wave-2 breadth rungs)
# ---------------------------------------------------------------------------
#
# The add-a-rung checklist puts each new executor in ``rungs_<provider>.py``
# and re-exports it HERE so ``imagegen.rungs`` names every executor the wave
# ships. The imports stay at the bottom on purpose: each provider module
# reaches this module's shared plumbing at CALL time (one lazy import per
# function), which keeps every import order working — this module importing
# the provider module first, or the provider module imported first.
from local_operator.imagegen.rungs_google import run_google  # noqa: E402
from local_operator.imagegen.rungs_openai_sub import run_openai_sub  # noqa: E402
from local_operator.imagegen.rungs_openrouter import run_openrouter  # noqa: E402
from local_operator.imagegen.rungs_xai import run_xai  # noqa: E402
