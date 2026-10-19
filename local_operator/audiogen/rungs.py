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
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

import httpx
from pydantic import SecretStr

from local_operator.artifacts.progress import ProgressFn, emit_progress
from local_operator.artifacts.progress import (
    progress_details as _generic_progress_details,
)
from local_operator.artifacts.rung import CancelHandle as SharedCancelHandle
from local_operator.artifacts.rung import RungResult
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
from local_operator.audiogen.peaks import compute_duration
from local_operator.audiogen.pricing import sound_effect_estimate, voiceover_estimate
from local_operator.clients._http import APIError
from local_operator.imagegen.media import download_asset
from local_operator.stt.errors import api_error_from_httpx_response

__all__ = [
    "AudioCancelReceipt",
    "SettlementFacts",
    "CancelReceipt",
    "best_effort_cancel",
    "cancel_receipt_text",
    "progress_details",
    "run_elevenlabs",
    "run_openai",
    "run_radient",
]


@dataclass
class CancelHandle(SharedCancelHandle):
    """Audio keeps financial identity after completion, independently of bytes.

    A terminal job cannot be cancelled, but clearing its request id would lose
    the accepted charge when the CDN or local artifact store subsequently fails.
    The observer is a synchronous ledger enqueue, never a provider request.
    """

    terminal: bool = False
    cost: dict[str, Any] | None = None
    cost_rev: int = 0
    observe_cost: Callable[[dict[str, Any], bool], None] | None = None

    def clear(self) -> None:
        self.terminal = True


class AdmittedAudioError(APIError):
    """An accepted job failed; trying another vendor risks a second charge."""


def _observe_cost(handle: SharedCancelHandle, payload: dict[str, Any], *, terminal: bool) -> None:
    cost = _cost_from_payload(payload)
    if isinstance(handle, CancelHandle):
        if cost is not None:
            handle.cost = cost
        if terminal:
            handle.terminal = True
        if handle.observe_cost is not None:
            handle.observe_cost(payload, terminal)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Radient — the hub's audio tools
# ---------------------------------------------------------------------------

RADIENT_AUDIO_VOICEOVER_PATH = "/tools/audio/voiceover"
RADIENT_AUDIO_SOUND_EFFECTS_PATH = "/tools/audio/sound-effects"
RADIENT_AUDIO_STATUS_PATH = "/tools/audio/status"
RADIENT_AUDIO_RESULT_PATH = "/tools/audio/result"
RADIENT_AUDIO_CANCEL_PATH = "/tools/audio/cancel"

#: How many times the SUBMIT may be attempted when the server answers the
#: admission bound (HTTP 429, ``audio_rate_limited``). The contract calls this
#: refusal retryable-on-purpose: it happens before a job row exists, so nothing
#: is charged, and a burst of retries is what the bound itself is protecting
#: against — hence the small cap rather than an unbounded loop.
SUBMIT_MAX_ATTEMPTS = 4
#: The backoff before each retry (2, 4, 8; a fourth refusal is surfaced).
#: Capped at 30 s by the contract; the values never reach it.
SUBMIT_BACKOFF_S = (2.0, 4.0, 8.0, 16.0)
#: The server's fixed sentence for the admission bound (there is no
#: ``Retry-After`` header to read — the contract says so explicitly).
AUDIO_RATE_LIMITED_SENTENCE = "Audio generation is busy right now. Wait a moment and try again."

#: Injected sleeper for the submit backoff. Module-level so a unit test can
#: bind a recorder and never pay the wall clock — the same seam the rung's
#: ``pause`` uses for its polls.
_submit_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep

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


def _is_admission_refusal(exc: APIError) -> bool:
    """Whether this submit failure is the admission bound (retryable) — nothing else.

    Two spellings count: the HTTP status (429) and the contract's code
    (``audio_rate_limited``) in either the parsed ``code`` or the details map.
    A 429 from an INTERMEDIARY would not be safe to retry blindly, so the code
    is what decides when it is present; the bare status is the fallback for a
    deployment whose body is not JSON.
    """
    if exc.code == "audio_rate_limited":
        return True
    if exc.details.get("error_type") == "audio_rate_limited":
        return True
    return exc.status_code == 429


def _rate_limited_error(exc: APIError) -> APIError:
    """The surfaced failure for an exhausted admission retry.

    ``reason_class`` (the ``code``) is ``transient``: a busy process is a
    retry-later condition, never a balance or credential signal. The wire's
    ``error_type`` rides the details so a surface can say which refusal it was,
    and the message is the server's own sentence when it sent one.
    """
    message = str(exc).strip() or AUDIO_RATE_LIMITED_SENTENCE
    return APIError(
        message,
        status_code=exc.status_code or 429,
        code="transient",
        details={"error_type": "audio_rate_limited"},
    )


def _hub_reason_error(exc: APIError) -> APIError:
    """``exc`` with the hub's additive ``details.reason`` appended, else ``exc``.

    A refused audio submit carries the server's generic sentence in
    ``error`` and the parameter-level explanation in ``details.reason``
    (e.g. "duration_seconds must be between 0.5 and 30 seconds, got 31").
    Every surface that shows the failure — the single tool result and each
    batch row, whose ``error`` is composed from the attempt messages — shows
    ``str(exc)``, so appending the reason HERE is what makes the actionable
    text reach them without teaching every downstream consumer about the
    details map. The typed fields (status, code, details, body) are copied
    unchanged: this upgrades the sentence, never the classification.

    Optional by design: an older hub, a reasonless refusal, or a reason
    already present in the message returns the exception unchanged — the
    field is consumed additively only, never required.
    """
    reason = exc.details.get("reason")
    if isinstance(reason, str) and reason.strip() and reason not in str(exc):
        return APIError(
            f"{exc} ({reason.strip()})",
            status_code=exc.status_code,
            code=exc.code,
            details=exc.details,
            body=exc.body,
        )
    return exc


async def _submit_with_backoff(
    client: httpx.AsyncClient,
    url: str,
    *,
    body: dict[str, Any],
    credential: str,
    jitter: bool = False,
) -> dict[str, Any]:
    """POST the submit, retrying ONLY the admission-bound refusal.

    Why retry this one and nothing else: the contract states a refused submit
    leaves no job row and no quote, so a retry cannot double-queue or
    double-bill; every other submit error fails straight through (the comment
    on ``_request_json`` is the general rule, and this is its one sanctioned
    exception, spelled by the server). Backoff is bounded and the sleeper is
    injectable so tests never sleep.
    """
    last: APIError | None = None
    for attempt in range(SUBMIT_MAX_ATTEMPTS):
        try:
            return await _request_json(
                client,
                "POST",
                url,
                label="Radient",
                timeout_s=HTTP_TIMEOUT_SUBMIT_S,
                secrets=(credential,),
                headers=_bearer_headers(credential),
                json=body,
            )
        except APIError as exc:
            if not _is_admission_refusal(exc):
                # The hub's refusal sentence is generic; its per-parameter
                # reason rides ``details.reason``. Consume it here, at the
                # source, so the attempt message (and every surface built
                # from it) carries the actionable text (see the helper).
                raise _hub_reason_error(exc) from exc
            last = exc
            if attempt >= SUBMIT_MAX_ATTEMPTS - 1:
                break
            delay = SUBMIT_BACKOFF_S[min(attempt, len(SUBMIT_BACKOFF_S) - 1)]
            # Two batch workers may hit the same account cap simultaneously.
            # Desynchronize retries without exceeding the existing bounded ladder.
            await _submit_sleep(delay * random.uniform(0.75, 1.0) if jitter else delay)
    assert last is not None
    raise _rate_limited_error(last)


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
    handle: SharedCancelHandle,
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
        submit = await _submit_with_backoff(
            http,
            f"{base}{submit_path}",
            body=body,
            credential=credential,
            jitter=spec.batch_submission,
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
        _observe_cost(handle, submit, terminal=False)
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
        if status not in RADIENT_OPEN_STATUSES:
            _observe_cost(handle, last_payload, terminal=True)
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
            _observe_cost(handle, status_payload, terminal=status not in RADIENT_OPEN_STATUSES)
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

        # Retain financial identity through result/CDN/store failures. The
        # audio handle's terminal bit suppresses cancel writes, not bookkeeping.
        if isinstance(handle, CancelHandle):
            handle.terminal = True

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
        _observe_cost(handle, result_payload, terminal=True)
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

        try:
            audio_asset = await _download_audio_asset(
                result_payload.get("audio"), http=http, voicing_notes=voicing
            )
        except Exception as exc:
            if (cost or {}).get("basis") == "billed":
                raise AdmittedAudioError(
                    "The generation was billed but the artifact could not be stored: " + str(exc),
                    status_code=None,
                    code="invalid_response",
                ) from exc
            raise
        return RungResult(
            assets=[audio_asset],
            model=model_id or "radient",
            generation_id=request_id,
            cost_usd=(cost or {}).get("amount_usd"),
            cost_source="reported" if cost is not None else None,
            billing_basis=(cost or {}).get("basis"),
            cost_provenance=_cost_provenance(cost_path, cost) if cost is not None else None,
            usage_record_id=usage_record_id,
        )


async def _download_audio_asset(
    audio: Any, *, http: httpx.AsyncClient, voicing_notes: dict[str, Any] | None = None
) -> AudioAsset:
    """Download, verify and shape the result's ``audio`` block. Fails closed.

    One path for the rung's own result read and the post-cancel settlement
    fetch, so a late-settled artifact is verified and shaped exactly like a
    first-class one. Bytes that fail their own sha256 are NEVER returned: the
    caller fails the rung (or the settlement records a settlement without an
    artifact) rather than attach something the server cannot vouch for.
    """
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
    expected_sha = _as_str(audio.get("sha256"))
    if (
        expected_sha is None
        or len(expected_sha) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in expected_sha)
    ):
        raise APIError(
            "Radient's audio block carried no valid sha256; the audio was discarded.",
            status_code=None,
            code="invalid_response",
        )
    downloaded = await download_asset(
        url,
        client=http,
        fallback_content_type=_as_str(audio.get("content_type")),
    )
    digest = hashlib.sha256(downloaded.data).hexdigest()
    if digest.lower() != expected_sha.lower():
        raise APIError(
            "The downloaded audio failed its sha256 verification and was discarded.",
            status_code=None,
            code="invalid_response",
        )
    return AudioAsset(
        data=downloaded.data,
        content_type=downloaded.content_type,
        source_url=url,
        duration_s=_as_float(audio.get("duration_seconds")),
        format=_as_str(audio.get("format")),
        sample_rate_hz=_as_int(audio.get("sample_rate_hz")),
        channels=_as_int(audio.get("channels")),
        bitrate_kbps=_as_int(audio.get("bitrate_kbps")),
        provider_model_id=_as_str(audio.get("provider_model_id")),
        expires_at=str(audio.get("expires_at")) if audio.get("expires_at") else None,
        sha256=expected_sha,
        voicing_notes=voicing_notes,
    )


@dataclass(frozen=True)
class SettlementFacts:
    """What one status/result read says about a job that refused cancellation.

    ``terminal`` is the ONLY gate on recording money: the contract states that
    the STATUS read alone carries a late charge (``settled: true`` + billed
    cost), so the artifact below is a bonus, never a precondition. ``asset`` is
    ``None`` when the job has not finished, failed, or its bytes could not be
    vouched for; ``asset_error`` then says why, without inventing a charge.
    """

    status: str
    terminal: bool
    cost_usd: float | None = None
    billing_basis: str | None = None
    usage_record_id: str | None = None
    units: float | None = None
    unit: str | None = None
    error_type: str | None = None
    message: str = ""
    asset: AudioAsset | None = None
    asset_error: str = ""


async def fetch_radient_settlement(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    credential: str,
    request_id: str,
) -> SettlementFacts:
    """One bounded observation of a Radient audio job (status, then result).

    Read-only and idempotent throughout: this is what the post-cancel poll
    calls every few seconds, so it must never submit, cancel or mutate. A
    failure to fetch the artifact does NOT hide a settled charge — the status
    read's cost is returned even when the result or the download fails.
    """
    base = base_url.rstrip("/")
    status_payload = await _request_json(
        client,
        "GET",
        f"{base}{RADIENT_AUDIO_STATUS_PATH}",
        label="Radient",
        timeout_s=HTTP_TIMEOUT_POLL_S,
        secrets=(credential,),
        transient_retries=1,
        headers=_bearer_headers(credential),
        params={"request_id": request_id},
    )
    status = (status_payload.get("status") or "").upper()
    if status not in RADIENT_TERMINAL_STATUSES:
        return SettlementFacts(status=status, terminal=False)
    failure = _audio_failure(status_payload)
    cost = _cost_from_payload(status_payload)
    facts = SettlementFacts(
        status=status,
        terminal=True,
        cost_usd=(cost or {}).get("amount_usd"),
        billing_basis=(cost or {}).get("basis"),
        usage_record_id=_as_str(status_payload.get("usage_record_id"))
        or (cost or {}).get("usage_record_id"),
        units=(cost or {}).get("units"),
        unit=(cost or {}).get("unit"),
        error_type=failure.code if failure is not None else None,
        message=str(failure) if failure is not None else "",
    )
    if failure is not None:
        return facts
    try:
        result_payload = await _request_json(
            client,
            "GET",
            f"{base}{RADIENT_AUDIO_RESULT_PATH}",
            label="Radient",
            timeout_s=HTTP_TIMEOUT_RESULT_S,
            secrets=(credential,),
            transient_retries=1,
            headers=_bearer_headers(credential),
            params={"request_id": request_id},
        )
        result_cost = _cost_from_payload(result_payload)
        if result_cost is not None:
            # Money belongs to the terminal response, not to a successful CDN
            # read. Preserve it even if verification below rejects the bytes.
            facts = replace(
                facts,
                cost_usd=result_cost["amount_usd"],
                billing_basis=result_cost["basis"],
                usage_record_id=result_cost.get("usage_record_id") or facts.usage_record_id,
                units=result_cost.get("units"),
                unit=result_cost.get("unit"),
            )
        result_failure = _audio_failure(result_payload)
        if result_failure is not None:
            return replace(facts, error_type=result_failure.code, message=str(result_failure))
        voicing = (
            result_payload.get("voicing")
            if isinstance(result_payload.get("voicing"), dict)
            else None
        )
        asset = await _download_audio_asset(
            result_payload.get("audio"), http=client, voicing_notes=voicing
        )
    except APIError as exc:
        return replace(facts, asset_error=str(exc), error_type=facts.error_type or exc.code)
    return replace(
        facts,
        cost_usd=(
            (result_cost or {}).get("amount_usd") if result_cost is not None else facts.cost_usd
        ),
        billing_basis=(
            (result_cost or {}).get("basis") if result_cost is not None else facts.billing_basis
        ),
        usage_record_id=(result_cost or {}).get("usage_record_id") or facts.usage_record_id,
        asset=asset,
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
    handle: SharedCancelHandle,
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
                # THE DURATION COMES FROM THE BYTES (design: a length is a
                # property of the clip, not of the vendor's response shape).
                # ElevenLabs returns the requested FORMAT and states no facts
                # beyond it, so reading a duration field here would have left
                # this leg null forever - and a null duration is what makes a
                # sound record render with NO PLAYER: a generated clip the user
                # cannot play, with no error to explain it. Decoded with the
                # same guarded engine the waveform peaks use; undecodable bytes
                # keep the old None rather than failing the generation.
                duration_s=compute_duration(data),
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
    handle: SharedCancelHandle,
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
                # Derived from the bytes for the same reason as the ElevenLabs
                # leg above: OpenAI states no duration, and a record without one
                # cannot draw a player.
                duration_s=compute_duration(data),
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
    #: The server's structured reason (``cancellation.reason``), e.g.
    #: ``provider_has_no_cancel_api``. Read for the receipt's honesty: the
    #: "will be charged if it produces audio" sentence belongs to THAT reason,
    #: not to every unstopped cancel.
    reason: str = ""
    #: The cancel response's JSON object, VERBATIM, when the server sent one.
    #:
    #: The settled contract's pending-settlement facts live in that body —
    #: ``cost`` (a ``quote`` basis with its units and quote), ``cancellation``
    #: (vendor_stopped/reason/detail) and ``note`` — and a surface can only
    #: render "awaiting settlement" from fields a client actually receives.
    #: The tool result forwards these keys; nothing is invented here, and a
    #: body that states no ``settled``/``cost_usd`` is not turned into a figure.
    server_payload: dict[str, Any] | None = None


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

#: The server-said sentences (settled contract, security round 1). A cancel is
#: a GUARANTEED STOP only while the job is still queued: ``vendor_stopped:
#: true`` means the provider was never called, so the request is not charged.
#: Once a job is running the vendor has no cancel endpoint at all
#: (``reason: "provider_has_no_cancel_api"``) and the generation continues.
VENDOR_STOPPED_TRUE = (
    "the generation was stopped before the provider was called; you will not be charged"
)
VENDOR_STOPPED_FALSE = (
    "the provider has no cancellation endpoint; the generation continues and will be "
    "charged if it produces audio"
)
VENDOR_STOPPED_UNKNOWN = (
    "the server did not report whether the provider was called — a request already "
    "running may still complete and be charged if it produces audio"
)

#: The sentence for a running job whose settlement is being watched. Appended
#: only when the tool actually started the bounded poll — a receipt must not
#: promise a check that is not happening.
SETTLEMENT_WATCH_NOTE = "Checking for settlement."


def cancel_receipt_text(
    receipt: AudioCancelReceipt, *, byok: bool = False, settling: bool = False
) -> str:
    """The one receipt sentence for a cancel, from what actually happened.

    The settled contract draws one line that the prose must not blur: a queued
    job can be stopped for certain (nothing was charged), a RUNNING job cannot
    (the provider has no cancel endpoint, and the work is charged if it
    produces audio). ``byok`` marks a direct provider call, where the harness
    stops waiting but the user's provider may still bill. ``settling`` is true
    only when the bounded settle poll was actually started, so the receipt can
    say a check is happening without promising an outcome.
    """
    if byok:
        return f"Stop requested — {BYOK_CANCEL_NOTE}."
    phrase = _CANCEL_PHRASES.get(receipt.outcome, receipt.outcome)
    if receipt.outcome == "cancelled":
        if receipt.vendor_stopped is True:
            return f"Cancelled: {VENDOR_STOPPED_TRUE}."
        if receipt.vendor_stopped is False:
            text = f"Stop requested: {VENDOR_STOPPED_FALSE}."
            return f"{text} {SETTLEMENT_WATCH_NOTE}" if settling else text
        text = f"Stop requested: {VENDOR_STOPPED_UNKNOWN}."
        return f"{text} {SETTLEMENT_WATCH_NOTE}" if settling else text
    return f"Not cancelled — {phrase}."


def _parse_cancel_body(
    response: httpx.Response,
) -> tuple[str, bool | None, str, dict[str, Any] | None]:
    """``(token, vendor_stopped, reason, payload)`` from a cancel response, tolerantly.

    The settled vocabulary is a small closed set spelled in a JSON field or a
    body depending on the deployment; a body neither reads nor parses just
    fails the marker test and the HTTP status decides. ``vendor_stopped`` is
    read only when the payload genuinely carries a bool; ``reason`` is the
    server's own word (``provider_has_no_cancel_api`` today) and is surfaced
    for the receipt rather than parsed into behaviour. ``payload`` is the
    parsed object itself (``None`` when the body was not a JSON object) so the
    tool can forward the contract's cost/cancellation/note facts instead of
    dropping them here.
    """
    vendor_stopped: bool | None = None
    reason = ""
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        cancellation = payload.get("cancellation")
        if isinstance(cancellation, dict):
            if isinstance(cancellation.get("vendor_stopped"), bool):
                vendor_stopped = cancellation["vendor_stopped"]
            raw_reason = cancellation.get("reason")
            if isinstance(raw_reason, str):
                reason = raw_reason.strip()
        token = (
            payload.get("status")
            or payload.get("state")
            or payload.get("result")
            # A 409 body carries the CODE, not a status word (settled
            # contract): ``{"code": "audio_already_completed"}``.
            or payload.get("code")
        )
        if isinstance(token, str):
            return token.upper(), vendor_stopped, reason, payload
        return (response.text or "").upper(), vendor_stopped, reason, payload
    return (response.text or "").upper(), vendor_stopped, reason, None


async def best_effort_cancel(handle: SharedCancelHandle | None) -> AudioCancelReceipt:
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
    if isinstance(handle, CancelHandle) and handle.terminal:
        return AudioCancelReceipt(outcome="already_completed")
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
        token, vendor_stopped, reason, payload = _parse_cancel_body(response)
        # The settled contract spells a completed/failed job's refusal as 409
        # ``audio_already_completed``, not a 200 status word — and a status
        # word spelled ALREADY_COMPLETED is still honoured (older deployments).
        if response.status_code == 409 or "ALREADY_COMPLETED" in token:
            return AudioCancelReceipt(
                outcome="already_completed",
                vendor_stopped=vendor_stopped,
                reason=reason,
                server_payload=payload,
            )
        if 200 <= response.status_code < 300:
            return AudioCancelReceipt(
                outcome="cancelled",
                vendor_stopped=vendor_stopped,
                reason=reason,
                server_payload=payload,
            )
        if "CANCELLED" in token:
            return AudioCancelReceipt(
                outcome="cancelled",
                vendor_stopped=vendor_stopped,
                reason=reason,
                server_payload=payload,
            )
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
