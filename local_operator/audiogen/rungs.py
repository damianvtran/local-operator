"""The audio rungs: Radient's audio tools, and the two BYOK direct calls.

**The frozen server contract** (Radient Agent Server lane, session
6c9c8f09af70 — the wire these rungs consume):

- ``POST /tools/audio/voiceover`` and ``POST /tools/audio/sound-effects``
  submit (base URL carries ``/v1``; paths hang off it), answering
  ``{request_id, status, model, provider, billing, usage, cost}`` with
  ``status`` in ``IN_QUEUE | IN_PROGRESS | COMPLETED | CANCELLED``.
- ``GET /tools/audio/status``, ``GET /tools/audio/result`` (both
  ``?request_id=``), ``POST /tools/audio/cancel``, ``GET /tools/audio/models``.
- The result's ``audio`` block is ``{url, expires_at, content_type, format,
  sample_rate_hz, channels, bitrate_kbps, duration_seconds, size_bytes,
  sha256, provider_model_id}`` — a signed URL with a 24 h life, and the
  ``sha256`` every download is VERIFIED against before any bytes are
  attached (fails closed: an unverified payload is never an artifact).
- **Failure shape:** ``status: "COMPLETED"`` + ``error_type: "audio_failed"``
  + a settled zero cost. The walk branches on the ERROR PAIR, never the status
  word alone: a "COMPLETED" with an ``error_type`` is an error.
- **No progress fraction, terminal words only.** ``queue_position`` is
  absent-or-a-number; nothing is synthesized into a bar; the canonical
  ``progress_fraction`` stays ``None``.
- **Cancellation is honest and best-effort**: once a vendor is IN_PROGRESS
  there is no cancel API behind the server, so the receipt says what the
  server said — ``cancellation.vendor_stopped: false`` is a real,
  expected answer and nothing promises a guaranteed stop.
- **The server route never accepts a provider key**, and neither do these
  bodies: the Radient rung authenticates with the account bearer, the BYOK
  rungs carry the user's own key in a vendor header only.

**BYOK (``elevenlabs`` / ``openai``)** is harness-direct with the user's local
key: ElevenLabs ``/v1/text-to-speech/{voice_id}`` and ``/v1/sound-generation``;
OpenAI ``/v1/audio/speech`` (voiceover only — a sound effect can never select
this rung; the cascade's capability filter records the skip). Voiceover
mapping reuses the vendored voicing map's executor
(``local_operator.tts.adapters``) so the BYOK legs translate ``tone`` /
``expressiveness`` / ``pace`` / ``language`` / ``accent`` / ``instructions``
exactly as the TTS daemon's own BYO legs do — one contract, two callers. The
attempt's ``notes`` travel into the result's voicing record so nothing is
dropped silently.

**Cost rules** (frozen): only ``cost.amount_usd`` is authoritative;
``basis: "quote"`` → ``estimated``, ``"billed"`` → ``billed``; a BYOK call has
no figure from the vendor at all (``cost: null``), so the harness ESTIMATES
from :mod:`local_operator.audiogen.pricing` and labels it ``estimated`` at
``rate_table`` source — never written as a charge.

**No threads, no subprocesses** (the image lane's D7 note applies unchanged):
work is in-process httpx cancelled through the standard loom machinery, so
nothing registers with ``group_reaper``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import SecretStr

from local_operator.artifacts.progress import ProgressFn, emit_progress
from local_operator.artifacts.progress import (
    progress_details as _generic_progress_details,
)
from local_operator.artifacts.rung import CancelHandle, RungResult
from local_operator.artifacts.walk import PauseFn
from local_operator.audiogen import AudioRoute
from local_operator.audiogen.models import (
    ELEVENLABS_SOUND_EFFECT_MODEL,
    ELEVENLABS_VOICEOVER_MODEL,
    OPENAI_VOICEOVER_MODEL,
    AudioAsset,
    AudioJobSpec,
    sound_effect_request_body,
    voiceover_request_body,
)
from local_operator.audiogen.pricing import sound_effect_estimate, voiceover_estimate
from local_operator.clients._http import APIError
from local_operator.imagegen.media import download_asset
from local_operator.stt.errors import api_error_from_httpx_response

__all__ = [
    "AudioCancelReceipt",
    "CancelReceipt",
    "best_effort_cancel",
    "cancel_receipt_text",
    "progress_details",
    "run_elevenlabs",
    "run_openai",
    "run_radient",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Radient — the hub's audio tools
# ---------------------------------------------------------------------------

RADIENT_AUDIO_VOICEOVER_PATH = "/tools/audio/voiceover"
RADIENT_AUDIO_SOUND_EFFECTS_PATH = "/tools/audio/sound-effects"
RADIENT_AUDIO_STATUS_PATH = "/tools/audio/status"
RADIENT_AUDIO_RESULT_PATH = "/tools/audio/result"
RADIENT_AUDIO_CANCEL_PATH = "/tools/audio/cancel"

#: The terminal vocabulary the contract defines. A status outside it is an
#: unreadable payload (``invalid_response``) rather than an infinite poll loop:
#: a newer server status must be a reviewed amendment here, not a hang.
RADIENT_TERMINAL_STATUSES = frozenset({"COMPLETED", "CANCELLED", "FAILED"})
RADIENT_OPEN_STATUSES = frozenset({"IN_QUEUE", "IN_PROGRESS"})

#: Per-request bounds. Audio jobs finish in seconds; the generous halves are
#: for a cold vendor, and the rung's own budget (cascade) owns the end.
HTTP_TIMEOUT_SUBMIT_S = 60.0
HTTP_TIMEOUT_POLL_S = 30.0
HTTP_TIMEOUT_RESULT_S = 30.0
HTTP_TIMEOUT_BYOK_S = 180.0

#: Poll pacing — audio is seconds-scale, so the base interval is short and the
#: slow tier keeps a long job from hammering; the no-signal ceiling bounds a
#: headless poller that can observe nothing (the image lane's round-1 finding,
#: same contract).
AUDIO_POLL_INTERVAL_S = 1.5
AUDIO_POLL_SLOW_AFTER_S = 10.0
AUDIO_POLL_SLOW_INTERVAL_S = 3.0
AUDIO_POLL_NO_SIGNAL_CAP_S = 6.0

#: The best-effort cancel's TOTAL budget (connect 2 s / read 3 s): Esc stays
#: snappy and a second stop can abandon the cleanup (see ``best_effort_cancel``).
CANCEL_TIMEOUT_TOTAL_S = 5.0

#: The BYOK hosts. Overridable per instance for tests; the OpenAI default
#: carries the ``/v1`` the API reference uses, the ElevenLabs default does not
#: (its paths are ``/v1/...`` themselves) — the TTS lane's own convention.
ELEVENLABS_AUDIO_BASE_URL = "https://api.elevenlabs.io"
OPENAI_AUDIO_BASE_URL = "https://api.openai.com/v1"

#: The audio container both BYOK vendors serve on these paths (the frozen
#: contract's MP3-only rule; no WAV).
BYOK_MEDIA_TYPE = "audio/mpeg"

#: ElevenLabs' ``output_format`` for a voiceover — the vendored map's fixed
#: value (``mp3_44100_128``), restated here because the URL pairing happens in
#: THIS module and the map's constant lives behind the adapters' params.
ELEVENLABS_OUTPUT_FORMAT = "mp3_44100_128"


def progress_details(
    *,
    tool: str,
    stage: str | None,
    provider: str | None = None,
    model: str | None = None,
    elapsed_s: int | None = None,
    queue_position: int | None = None,
    log_lines: list[dict[str, Any]] | None = None,
    error: str | None = None,
    error_type: str | None = None,
) -> dict[str, Any]:
    """The canonical live-progress payload for an audio-tool update.

    The audio lane's binding of the generic builder
    (:func:`local_operator.artifacts.progress.progress_details`): one key-set
    authority (the generic builder), with the audio lane's values —
    ``progress_fraction`` stays ``None`` (the contract has no fraction) and
    ``num_images`` stays ``None`` (the key exists by contract; an audio
    generation has no image count, and the payload never synthesizes a value
    nobody supplied).
    """
    return _generic_progress_details(
        tool=tool,
        stage=stage,
        provider=provider,
        model=model,
        elapsed_s=elapsed_s,
        num_images=None,
        queue_position=queue_position,
        log_lines=log_lines,
        error=error,
        error_type=error_type,
    )


# ---------------------------------------------------------------------------
# Shared plumbing (mirrors ``imagegen.rungs``; same injected-client seam)
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _client_scope(client: httpx.AsyncClient | None) -> AsyncIterator[httpx.AsyncClient]:
    """The caller's injected client, or one this rung owns and closes.

    The injected-client seam is how tests attach ``httpx.MockTransport``
    without a network; production call sites leave it ``None`` and pay one
    client per rung.
    """
    if client is not None:
        yield client
        return
    async with httpx.AsyncClient() as owned:
        yield owned


def _as_float(value: Any) -> float | None:
    """A number from a payload field, or ``None`` — never raises, never bools."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


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


#: Codes and statuses a bounded retry may cover (the image lane's set): the
#: transport-level codes where the request never landed, and the retryable
#: server class. Anything else is the upstream's ANSWER and must reach the
#: walk unchanged.
_TRANSIENT_API_CODES = frozenset({"timeout", "network"})
_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def _is_transient_api_error(exc: APIError) -> bool:
    if isinstance(exc.code, str) and exc.code in _TRANSIENT_API_CODES:
        return True
    return exc.status_code in _TRANSIENT_STATUS_CODES


async def _request_json(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    label: str,
    timeout_s: float,
    secrets: tuple[str, ...] = (),
    transient_retries: int = 0,
    **kwargs: Any,
) -> dict[str, Any]:
    """One bounded JSON request with this package's error shape on failure.

    ``transient_retries`` is how many EXTRA attempts an IDEMPOTENT read may
    make: a poll blip must not orphan a billed job. Submits are never retried
    (a blind re-POST can double-queue and double-bill; failover is their
    remedy) — only the free GETs opt in.
    """
    attempt = 0
    while True:
        try:
            try:
                response = await client.request(
                    method, url, timeout=httpx.Timeout(timeout_s), **kwargs
                )
            except httpx.TimeoutException as exc:
                raise APIError(
                    f"{label} timed out after {timeout_s:.0f}s.", status_code=None, code="timeout"
                ) from exc
            except httpx.HTTPError as exc:
                raise APIError(
                    f"{label} request failed: {exc}", status_code=None, code="network"
                ) from exc
            if response.status_code >= 400:
                raise api_error_from_httpx_response(
                    response, fallback_message=f"{label} refused the request", secrets=secrets
                )
            return await _read_json(response, label=label)
        except APIError as exc:
            if attempt >= transient_retries or not _is_transient_api_error(exc):
                raise
            attempt += 1
            await asyncio.sleep(0.75 * attempt)


def _poll_interval(elapsed_s: float) -> float:
    """The signal path's interval: 1.5 s, relaxing to 3 s once vendor-bound."""
    if elapsed_s >= AUDIO_POLL_SLOW_AFTER_S:
        return AUDIO_POLL_SLOW_INTERVAL_S
    return AUDIO_POLL_INTERVAL_S


def _no_signal_poll_interval(elapsed_s: float) -> float:
    """1.5 s → 3 s → 6 s(cap) for callers nothing is watching."""
    return min(
        AUDIO_POLL_INTERVAL_S * (2 ** int(elapsed_s // AUDIO_POLL_SLOW_AFTER_S)),
        AUDIO_POLL_NO_SIGNAL_CAP_S,
    )


async def _pace_poll(pause: PauseFn | None, elapsed_s: float) -> None:
    """Wait between poll reads — abortably when a signal exists, paced when not."""
    if pause is not None:
        await pause(_poll_interval(elapsed_s))
    else:
        await asyncio.sleep(_no_signal_poll_interval(elapsed_s))


def _bearer_headers(credential: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {credential}", "Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# Radient failure/cost mapping
# ---------------------------------------------------------------------------


def _audio_failure(payload: dict[str, Any]) -> APIError | None:
    """The error PAIR's failure half: an ``error_type`` on a terminal payload.

    The frozen shape is ``status: "COMPLETED"`` carrying ``error_type`` — a
    *successful* status word with a failure inside — so every terminal read
    checks the pair FIRST and never trusts the status alone. The type rides
    verbatim as the error ``code`` (``REASON_CLASSES`` lists it), so attempt
    records group without parsing prose.
    """
    error_type = _as_str(payload.get("error_type"))
    if error_type is None:
        return None
    message = _as_str(payload.get("error")) or f"Radient reported {error_type} for this request."
    return APIError(message, status_code=None, code=error_type)


def _cost_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """The ``cost`` block's ledger-facing fields, or ``None``.

    Only ``amount_usd`` is authoritative (frozen rule); ``basis`` maps
    ``quote → estimated`` and ``billed → billed``; an absent basis beside an
    amount is conservatively ``estimated`` (a figure not yet labelled a charge
    must never read as one).
    """
    cost = payload.get("cost")
    if not isinstance(cost, dict):
        return None
    amount = _as_float(cost.get("amount_usd"))
    if amount is None:
        return None
    basis_label = (_as_str(cost.get("basis")) or "").lower()
    if basis_label == "billed":
        basis = "billed"
    else:
        basis = "estimated"
    return {
        "amount_usd": amount,
        "basis": basis,
        "raw_basis": basis_label,
        "usage_record_id": _as_str(cost.get("usage_record_id")),
        "price_version": _as_str(cost.get("price_version")),
        "unit": _as_str(cost.get("unit")),
        "units": _as_float(cost.get("units")),
        "tool_type": _as_str(cost.get("tool_type")),
    }


def _cost_provenance(path: str, cost: dict[str, Any]) -> str:
    return (
        f"Radient {path} cost.amount_usd (basis: {cost.get('raw_basis') or 'unstated'}, "
        f"price_version: {cost.get('price_version') or 'unstated'})"
    )


def _raise_if_terminal_error(payload: dict[str, Any], *, status: str) -> None:
    """Raise for every non-COMPLETED terminal; no-op while a job is open.

    ONE classifier for the submit, status and result reads, so the error-pair
    branch cannot drift between them: an ``error_type`` wins over the status
    word (the frozen failure shape rides ``COMPLETED`` — the pair, never the
    status alone), ``CANCELLED`` and ``FAILED`` map to their closed reason
    classes, and a terminal word this contract does not define is an
    unreadable payload rather than an infinite poll.
    """
    failure = _audio_failure(payload)
    if failure is not None:
        raise failure
    if status == "CANCELLED":
        raise APIError(
            "Radient reported the generation as CANCELLED.",
            status_code=None,
            code="cancelled",
        )
    if status == "FAILED":
        raise APIError(
            _as_str(payload.get("error")) or "Radient reported the generation as FAILED.",
            status_code=None,
            code="audio_failed",
        )
    if status not in RADIENT_OPEN_STATUSES and status != "COMPLETED":
        raise APIError(
            f"Radient returned an unknown status {status!r}.",
            status_code=None,
            code="invalid_response",
        )


# ---------------------------------------------------------------------------
# The Radient rung
# ---------------------------------------------------------------------------


async def run_radient(
    spec: AudioJobSpec,
    *,
    base_url: str,
    credential: str,
    handle: CancelHandle,
    emit: ProgressFn | None,
    pause: PauseFn | None,
    client: httpx.AsyncClient | None = None,
) -> RungResult:
    """Rung 1: the Radient hub's audio tools. See the module docstring.

    The walk: submit → poll (terminal words only) → result → bounded download
    → sha256 verification. ``credential`` is the bearer the cascade resolved
    (``SecretStr`` unwrapped at the boundary).
    """
    base = base_url.rstrip("/")
    submit_path = (
        RADIENT_AUDIO_VOICEOVER_PATH
        if spec.direction == "voiceover"
        else RADIENT_AUDIO_SOUND_EFFECTS_PATH
    )
    body = (
        voiceover_request_body(spec)
        if spec.direction == "voiceover"
        else sound_effect_request_body(spec)
    )
    async with _client_scope(client) as http:
        submit = await _request_json(
            http,
            "POST",
            f"{base}{submit_path}",
            label="Radient",
            timeout_s=HTTP_TIMEOUT_SUBMIT_S,
            secrets=(credential,),
            headers=_bearer_headers(credential),
            json=body,
        )
        request_id = _as_str(submit.get("request_id"))
        if request_id is None:
            raise APIError(
                "Radient accepted the request but returned no request_id.",
                status_code=None,
                code="invalid_response",
            )
        model_id = _as_str(submit.get("model")) or spec.model or ""
        handle.provider = AudioRoute.RADIENT
        handle.request_id = request_id
        handle.model = model_id or None
        handle.base_url = base
        handle.credential = SecretStr(credential)
        cost = _cost_from_payload(submit)
        cost_path = submit_path
        usage_record_id: str | None = (cost or {}).get("usage_record_id")
        voicing = submit.get("voicing") if isinstance(submit.get("voicing"), dict) else None

        started = time.monotonic()

        def _emit_stage(stage: str, status_word: str, payload: dict[str, Any]) -> None:
            elapsed = int(time.monotonic() - started)
            queue_position = _as_int(payload.get("queue_position"))
            queue_note = f", #{queue_position}" if stage == "queued" and queue_position else ""
            emit_progress(
                emit,
                f"Generating via Radient ({model_id or 'default model'}): "
                f"{status_word}{queue_note} — {elapsed}s",
                **progress_details(
                    tool=spec.tool,
                    stage=stage,
                    provider=str(AudioRoute.RADIENT),
                    model=model_id or None,
                    elapsed_s=elapsed,
                    queue_position=queue_position,
                ),
            )

        status = (submit.get("status") or "").upper()
        last_payload: dict[str, Any] = submit
        # The submit answer can already be terminal (a failure pair, or a job
        # the server answers straight from cache); every terminal read goes
        # through the same classifier as the polls below so the branches
        # cannot drift between submit, status and result.
        _raise_if_terminal_error(last_payload, status=status)
        while status in RADIENT_OPEN_STATUSES:
            _emit_stage(
                "queued" if status == "IN_QUEUE" else "in_progress",
                "queued" if status == "IN_QUEUE" else "running",
                last_payload,
            )
            await _pace_poll(pause, time.monotonic() - started)
            status_payload = await _request_json(
                http,
                "GET",
                f"{base}{RADIENT_AUDIO_STATUS_PATH}",
                label="Radient",
                timeout_s=HTTP_TIMEOUT_POLL_S,
                secrets=(credential,),
                # An idempotent read: one bounded retry is safe.
                transient_retries=1,
                headers=_bearer_headers(credential),
                params={"request_id": request_id},
            )
            last_payload = status_payload
            status = (status_payload.get("status") or "").upper()
            model_id = _as_str(status_payload.get("model")) or model_id
            _raise_if_terminal_error(status_payload, status=status)
            settled = _cost_from_payload(status_payload)
            if settled is not None:
                cost = settled
                cost_path = RADIENT_AUDIO_STATUS_PATH
            settled_usage = (settled or {}).get("usage_record_id") or _as_str(
                status_payload.get("usage_record_id")
            )
            if settled_usage:
                usage_record_id = settled_usage
            voicing = (
                status_payload.get("voicing")
                if isinstance(status_payload.get("voicing"), dict)
                else voicing
            )

        # Terminal: nothing left for a cancel to do. Cleared BEFORE the result
        # read and the downloads so a stop mid-download reads "none" rather
        # than firing an ALREADY_COMPLETED round-trip (the image lane's rule).
        handle.clear()

        result_payload = await _request_json(
            http,
            "GET",
            f"{base}{RADIENT_AUDIO_RESULT_PATH}",
            label="Radient",
            timeout_s=HTTP_TIMEOUT_RESULT_S,
            secrets=(credential,),
            # An idempotent read: one bounded retry is safe.
            transient_retries=1,
            headers=_bearer_headers(credential),
            params={"request_id": request_id},
        )
        failure = _audio_failure(result_payload)
        if failure is not None:
            raise failure
        settled = _cost_from_payload(result_payload)
        if settled is not None:
            cost = settled
            cost_path = RADIENT_AUDIO_RESULT_PATH
        settled_usage = (settled or {}).get("usage_record_id") or _as_str(
            result_payload.get("usage_record_id")
        )
        if settled_usage:
            usage_record_id = settled_usage
        result_voicing = result_payload.get("voicing")
        if isinstance(result_voicing, dict):
            voicing = result_voicing

        audio = result_payload.get("audio")
        if not isinstance(audio, dict):
            raise APIError(
                "Radient's result carried no audio block.",
                status_code=None,
                code="invalid_response",
            )
        url = _as_str(audio.get("url"))
        if url is None:
            raise APIError(
                "Radient's audio block carried no url.",
                status_code=None,
                code="invalid_response",
            )
        asset = await download_asset(
            url,
            client=http,
            fallback_content_type=_as_str(audio.get("content_type")),
        )
        expected_sha = _as_str(audio.get("sha256"))
        if expected_sha:
            digest = hashlib.sha256(asset.data).hexdigest()
            if digest.lower() != expected_sha.lower():
                # Fail closed: bytes that fail their own digest are never
                # attached. The rung fails and the cascade fails forward per
                # its contract; the message says exactly what happened.
                raise APIError(
                    "The downloaded audio failed its sha256 verification and was discarded.",
                    status_code=None,
                    code="invalid_response",
                )
        return RungResult(
            assets=[
                AudioAsset(
                    data=asset.data,
                    content_type=asset.content_type,
                    source_url=url,
                    duration_s=_as_float(audio.get("duration_seconds")),
                    format=_as_str(audio.get("format")),
                    sample_rate_hz=_as_int(audio.get("sample_rate_hz")),
                    channels=_as_int(audio.get("channels")),
                    bitrate_kbps=_as_int(audio.get("bitrate_kbps")),
                    provider_model_id=_as_str(audio.get("provider_model_id")),
                    expires_at=str(audio.get("expires_at")) if audio.get("expires_at") else None,
                    sha256=expected_sha,
                    voicing_notes=voicing,
                )
            ],
            model=model_id or "radient",
            generation_id=request_id,
            cost_usd=(cost or {}).get("amount_usd"),
            cost_source="reported" if cost is not None else None,
            billing_basis=(cost or {}).get("basis"),
            cost_provenance=_cost_provenance(cost_path, cost) if cost is not None else None,
            usage_record_id=usage_record_id,
        )


# ---------------------------------------------------------------------------
# BYOK — ElevenLabs
# ---------------------------------------------------------------------------


def _byok_voicing_notes(attempt: Any) -> dict[str, Any]:
    """The map attempt's notes as a JSON-safe voicing record.

    ``applied`` lists the fields honoured exactly; ``degraded`` carries the
    ``field:action[=detail]`` tokens (emulated / clamped / ignored / nearest /
    pinned) — the vendored map's own grammar, richer than the server's three
    words but self-describing, and the reason nothing is dropped silently.
    """
    return {
        "applied": list(attempt.applied_fields()),
        "degraded": list(attempt.degraded_tokens()),
    }


def _byok_voiceover_attempt(provider: str, model: str, spec: AudioJobSpec) -> Any:
    """The vendored map's mapping for one BYOK voiceover. See module docstring.

    An alias voice (``female``/``male``) rides the descriptor's gender; a
    vendor id rides the legacy pin; an absent voice leaves gender unstated and
    the map's documented nearest rule resolves it.
    """
    from local_operator.tts import adapters as tts_adapters
    from local_operator.tts.descriptor import VoiceDescriptor

    voice = (spec.voice or "").strip()
    is_alias = tts_adapters.is_voice_alias(voice)
    descriptor = VoiceDescriptor(
        gender=(voice.lower() if is_alias else ""),
        tone=(spec.tone or ""),
        expressiveness=(spec.expressiveness or ""),
        pace=spec.pace,
        language=(spec.language or ""),
        accent=(spec.accent or ""),
        instructions=spec.instructions,
    )
    # The descriptor's empty strings read as "absent" (the map's own nearest
    # rules), which is what an unset tool field means; a vendor-id voice is a
    # PIN (legacy), exactly as the TTS lane treats it.
    legacy = tts_adapters.Legacy(voice=("" if is_alias or not voice else voice))
    return tts_adapters.resolve(provider, model, descriptor, legacy)


def _elevenlabs_fixed() -> dict[str, Any]:
    """The vendored map's ``fixed.elevenlabs`` block — the one source for the
    platform constants (similarity_boost, use_speaker_boost, output_format)
    the TTS daemon's own BYO leg reads from the same file.
    """
    from local_operator.tts import adapters as tts_adapters

    return dict(tts_adapters.load_map().fixed.get("elevenlabs") or {})


async def run_elevenlabs(
    spec: AudioJobSpec,
    *,
    api_key: str,
    handle: CancelHandle,
    emit: ProgressFn | None,
    pause: PauseFn | None,
    client: httpx.AsyncClient | None = None,
    base_url: str | None = None,
) -> RungResult:
    """Rung 2 (BYOK): the user's own ElevenLabs key, direct.

    Voiceover: ``POST /v1/text-to-speech/{voice_id}`` with the map-resolved
    params. Sound effect: ``POST /v1/sound-generation``. The vendor reports no
    per-call figure on either path, so the cost is the central rate table's
    ESTIMATE (labelled ``estimated``/``rate_table``) — never a charge.
    """
    base = (base_url or ELEVENLABS_AUDIO_BASE_URL).rstrip("/")
    model = spec.model or (
        ELEVENLABS_VOICEOVER_MODEL
        if spec.direction == "voiceover"
        else ELEVENLABS_SOUND_EFFECT_MODEL
    )
    headers = {"xi-api-key": api_key, "Accept": BYOK_MEDIA_TYPE}
    voicing: dict[str, Any] | None = None

    async with _client_scope(client) as http:
        if spec.direction == "voiceover":
            attempt = _byok_voiceover_attempt("elevenlabs", model, spec)
            params = attempt.params
            voicing = _byok_voicing_notes(attempt)
            fixed = _elevenlabs_fixed()
            voice_settings: dict[str, Any] = {
                "similarity_boost": float(fixed.get("similarity_boost", 0.75)),
                "use_speaker_boost": bool(fixed.get("use_speaker_boost", True)),
            }
            if params.stability is not None:
                voice_settings["stability"] = params.stability
            if params.speed is not None:
                voice_settings["speed"] = params.speed
            payload: dict[str, Any] = {
                "text": spec.text,
                "model_id": params.model_id or model,
            }
            if params.language_code:
                payload["language_code"] = params.language_code
            payload["voice_settings"] = voice_settings
            url = f"{base}/v1/text-to-speech/{params.voice_id}"
            request_params: dict[str, str] = {
                "output_format": str(fixed.get("output_format") or ELEVENLABS_OUTPUT_FORMAT)
            }
            model = params.model_id or model
        else:
            if spec.duration_s is None:  # pragma: no cover - the tool validates first
                raise APIError(
                    "Sound effects require a duration_s.", status_code=None, code="unsupported"
                )
            payload = {
                "text": spec.text,
                "model_id": model,
                "duration_seconds": float(spec.duration_s),
            }
            if spec.prompt_influence is not None:
                payload["prompt_influence"] = float(spec.prompt_influence)
            if spec.loop:
                payload["loop"] = True
            url = f"{base}/v1/sound-generation"
            request_params = {}

        emit_progress(
            emit,
            f"Generating via ElevenLabs ({model})…",
            **progress_details(
                tool=spec.tool,
                stage="in_progress",
                provider=str(AudioRoute.ELEVENLABS),
                model=model,
            ),
        )
        data, content_type = await _post_audio(
            http,
            url,
            headers=headers,
            payload=payload,
            params=request_params,
            model_label="ElevenLabs",
            api_key=api_key,
        )

    estimate = (
        voiceover_estimate("elevenlabs", model, len(spec.text))
        if spec.direction == "voiceover"
        else sound_effect_estimate("elevenlabs", model, float(spec.duration_s or 0))
    )
    return RungResult(
        assets=[
            AudioAsset(
                data=data,
                content_type=content_type,
                source_url="",
                # The vendor returns the REQUESTED output format and states no
                # facts beyond it; duration is never derived client-side
                # (frozen rule), so it stays None on a BYOK leg.
                format="mp3" if content_type == BYOK_MEDIA_TYPE else None,
                sample_rate_hz=44100 if spec.direction == "voiceover" else None,
                bitrate_kbps=128 if spec.direction == "voiceover" else None,
                voicing_notes=voicing,
            )
        ],
        model=model,
        cost_usd=estimate.amount_usd if estimate else None,
        cost_source="rate_table" if estimate else None,
        billing_basis="estimated" if estimate else None,
        cost_provenance=estimate.provenance if estimate else None,
    )


# ---------------------------------------------------------------------------
# BYOK — OpenAI (voiceover only)
# ---------------------------------------------------------------------------


async def run_openai(
    spec: AudioJobSpec,
    *,
    api_key: str,
    handle: CancelHandle,
    emit: ProgressFn | None,
    pause: PauseFn | None,
    client: httpx.AsyncClient | None = None,
    base_url: str | None = None,
) -> RungResult:
    """Rung 3 (BYOK): the user's own OpenAI key, direct — voiceover only.

    A sound-effect request never reaches this function: the cascade's rung
    filter records the ``unsupported`` skip in its cascade position and never
    dispatches. A direct caller that arrives anyway is refused honestly.
    """
    if spec.direction != "voiceover":
        raise APIError(
            "OpenAI has no sound-effect endpoint; this rung serves voiceovers only.",
            status_code=None,
            code="unsupported",
        )
    base = (base_url or OPENAI_AUDIO_BASE_URL).rstrip("/")
    model = spec.model or OPENAI_VOICEOVER_MODEL
    attempt = _byok_voiceover_attempt("openai", model, spec)
    params = attempt.params
    model = params.model_id or model
    payload: dict[str, Any] = {
        "model": model,
        "input": spec.text,
        "voice": params.voice_id,
        "response_format": "mp3",
    }
    if params.speed is not None:
        payload["speed"] = params.speed
    if params.instructions:
        payload["instructions"] = params.instructions

    async with _client_scope(client) as http:
        emit_progress(
            emit,
            f"Generating via OpenAI ({model})…",
            **progress_details(
                tool=spec.tool,
                stage="in_progress",
                provider=str(AudioRoute.OPENAI),
                model=model,
            ),
        )
        data, content_type = await _post_audio(
            http,
            f"{base}/audio/speech",
            headers={"Authorization": f"Bearer {api_key}", "Accept": BYOK_MEDIA_TYPE},
            payload=payload,
            params={},
            model_label="OpenAI",
            api_key=api_key,
        )

    estimate = voiceover_estimate("openai", model, len(spec.text))
    return RungResult(
        assets=[
            AudioAsset(
                data=data,
                content_type=content_type,
                source_url="",
                format="mp3",
                voicing_notes=_byok_voicing_notes(attempt),
            )
        ],
        model=model,
        cost_usd=estimate.amount_usd if estimate else None,
        cost_source="rate_table" if estimate else None,
        billing_basis="estimated" if estimate else None,
        cost_provenance=estimate.provenance if estimate else None,
    )


async def _post_audio(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    params: dict[str, str],
    model_label: str,
    api_key: str,
) -> tuple[bytes, str]:
    """One JSON POST answered with audio bytes, with the shared error shape.

    ``timeout=HTTP_TIMEOUT_BYOK_S`` bounds the whole call; the vendor's
    content-type is read off the header (falling back to the frozen contract's
    MP3 type), and every failure is an ``APIError`` the walk classifies.
    """
    try:
        response = await client.post(
            url,
            headers={**headers, "Content-Type": "application/json"},
            json=payload,
            params=params,
            timeout=httpx.Timeout(HTTP_TIMEOUT_BYOK_S),
        )
    except httpx.TimeoutException as exc:
        raise APIError(
            f"{model_label} timed out after {HTTP_TIMEOUT_BYOK_S:.0f}s.",
            status_code=None,
            code="timeout",
        ) from exc
    except httpx.HTTPError as exc:
        raise APIError(
            f"{model_label} request failed: {exc}", status_code=None, code="network"
        ) from exc
    if response.status_code >= 400:
        raise api_error_from_httpx_response(
            response, fallback_message=f"{model_label} refused the request", secrets=(api_key,)
        )
    content_type = (response.headers.get("content-type") or "").split(";")[0].strip()
    if not content_type or content_type == "application/octet-stream":
        content_type = BYOK_MEDIA_TYPE
    return response.content, content_type


# ---------------------------------------------------------------------------
# Honest cancellation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioCancelReceipt:
    """What a best-effort cancel observed — the raw facts, before any prose.

    ``vendor_stopped`` is the server's own word (``cancellation.vendor_stopped``
    from the frozen contract) and is ``None`` when the server did not say.
    ``outcome`` is the closed token vocabulary below; ``detail`` carries the
    server's sentence when it offered one (never invented).
    """

    outcome: str
    vendor_stopped: bool | None = None
    detail: str = ""


#: The closed cancel-outcome tokens (mirrors the image lane's set).
_CANCEL_TOKENS = frozenset(
    {
        "cancelled",
        "already_completed",
        "not_found",
        "none",
        "timeout",
        "failed",
        "abandoned",
    }
)

#: The receipt phrase per token, for the platform's own cases. A token this
#: map does not know is surfaced verbatim — a receipt never invents an outcome.
_CANCEL_PHRASES = {
    "cancelled": "the cancel request was accepted",
    "already_completed": "it had already completed; its result was discarded",
    "not_found": "the server no longer knows the request",
    "none": "nothing was left to cancel",
    "timeout": "the cancel request timed out",
    "failed": "the cancel request failed",
    "abandoned": "a second stop abandoned the cancel attempt",
}

#: The byok sentence: there is no server-side job to stop, and the vendor may
#: still bill for work already delivered. Never promises a guaranteed stop.
BYOK_CANCEL_NOTE = (
    "this ran directly against your provider, which has no cancel API — the harness "
    "stopped waiting, but your provider may still bill for work already delivered"
)

#: The server-said sentences. ``vendor_stopped: true`` means the vendor was
#: stopped before completing, so nothing further is charged for this request;
#: ``false`` is the contract's honest expectation once a vendor is running.
VENDOR_STOPPED_TRUE = (
    "the server reports the vendor was stopped before completing — nothing further "
    "will be generated, and this request is not charged"
)
VENDOR_STOPPED_FALSE = (
    "the server reports the vendor could NOT be stopped — the work may still complete "
    "upstream and may still be charged (cancellation is best-effort; no guaranteed stop)"
)
VENDOR_STOPPED_UNKNOWN = (
    "the server did not report whether the vendor stopped — a request already running "
    "may still complete and may still be charged (cancellation is best-effort)"
)


def cancel_receipt_text(receipt: AudioCancelReceipt, *, byok: bool = False) -> str:
    """The one receipt sentence for a cancel, from what actually happened.

    ``byok`` marks a cancel of a direct provider call: the harness stopped
    waiting, the vendor cannot be cancelled, and the user may still be billed —
    stated plainly rather than promised away. A Radient receipt says what the
    SERVER said (``vendor_stopped``), including the expected
    ``false`` — never a claim of a guaranteed stop.
    """
    if byok:
        return f"Cancelled — {BYOK_CANCEL_NOTE}."
    phrase = _CANCEL_PHRASES.get(receipt.outcome, receipt.outcome)
    if receipt.outcome == "cancelled":
        if receipt.vendor_stopped is True:
            return f"Cancelled: {VENDOR_STOPPED_TRUE}."
        if receipt.vendor_stopped is False:
            return f"Cancelled: {VENDOR_STOPPED_FALSE}."
        return f"Cancelled: {VENDOR_STOPPED_UNKNOWN}."
    return f"Not cancelled — {phrase}."


def _parse_cancel_body(response: httpx.Response) -> tuple[str, bool | None]:
    """``(token, vendor_stopped)`` from a cancel response, tolerantly.

    The settled vocabulary is a small closed set spelled in a JSON field or a
    body depending on the deployment; a body neither reads nor parses just
    fails the marker test and the HTTP status decides. ``vendor_stopped`` is
    read only when the payload genuinely carries a bool.
    """
    vendor_stopped: bool | None = None
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        cancellation = payload.get("cancellation")
        if isinstance(cancellation, dict) and isinstance(cancellation.get("vendor_stopped"), bool):
            vendor_stopped = cancellation["vendor_stopped"]
        token = payload.get("status") or payload.get("state") or payload.get("result")
        if isinstance(token, str):
            return token.upper(), vendor_stopped
    return (response.text or "").upper(), vendor_stopped


async def best_effort_cancel(handle: CancelHandle | None) -> AudioCancelReceipt:
    """Ask the server to cancel the in-flight Radient job; NEVER raises.

    Returns an :class:`AudioCancelReceipt` whose ``outcome`` is one of
    ``cancelled`` / ``already_completed`` / ``not_found`` / ``none`` /
    ``timeout`` / ``failed`` / ``abandoned`` (the same closed set the image
    lane's receipt uses), plus what the server said about the vendor. A BYOK
    call has no server-side job, so its handle stays empty and this reads
    ``none`` — the TOOL renders the byok sentence in that case rather than
    pretending something was cancelled. The attempt runs under
    ``asyncio.shield`` so a SECOND cancellation does not tear the inbound
    request down mid-flight; the 5 s bound keeps Esc snappy.
    """
    if handle is None or handle.provider is None or handle.request_id is None:
        return AudioCancelReceipt(outcome="none")
    provider = handle.provider
    credential = handle.credential

    async def _attempt() -> AudioCancelReceipt:
        timeout = httpx.Timeout(3.0, connect=2.0)
        try:
            async with _client_scope(None) as http:
                if provider == AudioRoute.RADIENT:
                    if not handle.base_url:
                        return AudioCancelReceipt(outcome="none")
                    headers = (
                        _bearer_headers(credential.get_secret_value())
                        if credential is not None
                        else {"Content-Type": "application/json"}
                    )
                    response = await http.post(
                        f"{handle.base_url.rstrip('/')}{RADIENT_AUDIO_CANCEL_PATH}",
                        headers=headers,
                        json={"request_id": handle.request_id},
                        timeout=timeout,
                    )
                else:
                    return AudioCancelReceipt(outcome="none")
        except httpx.HTTPError:
            return AudioCancelReceipt(outcome="failed")
        if response.status_code == 404:
            return AudioCancelReceipt(outcome="not_found")
        token, vendor_stopped = _parse_cancel_body(response)
        if "ALREADY_COMPLETED" in token:
            return AudioCancelReceipt(outcome="already_completed", vendor_stopped=vendor_stopped)
        if 200 <= response.status_code < 300:
            return AudioCancelReceipt(outcome="cancelled", vendor_stopped=vendor_stopped)
        if "CANCELLED" in token:
            return AudioCancelReceipt(outcome="cancelled", vendor_stopped=vendor_stopped)
        return AudioCancelReceipt(outcome="failed")

    async def _guarded() -> AudioCancelReceipt:
        try:
            return await asyncio.wait_for(_attempt(), timeout=CANCEL_TIMEOUT_TOTAL_S)
        except TimeoutError:
            return AudioCancelReceipt(outcome="timeout")
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a cleanup must never raise
            logger.debug("provider cancel failed", exc_info=True)
            return AudioCancelReceipt(outcome="failed")

    try:
        return await asyncio.shield(_guarded())
    except asyncio.CancelledError:
        return AudioCancelReceipt(outcome="abandoned")
    except BaseException:  # noqa: BLE001 - incl. GeneratorExit; never raise from cleanup
        return AudioCancelReceipt(outcome="failed")


#: Kept for symmetry with the image lane's naming; audio has exactly one
#: receipt type.
CancelReceipt = AudioCancelReceipt
