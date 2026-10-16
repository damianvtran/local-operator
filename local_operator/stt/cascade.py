"""The cascade resolver and executor.

**Resolver** (:func:`resolve_audio_path`) — the frozen decision order, first
match wins (manager decisions, 2026-09-28):

1. ``provider_stt_radient`` — Radient credentials resolve non-empty, through
   the same :func:`resolve_radient_credential` the daemon's legacy route uses,
   so the resolver and the executor cannot disagree about what "signed in"
   means.
2. ``provider_stt_elevenlabs`` — a stored ElevenLabs key exists.
3. ``provider_stt_openai`` — a stored OpenAI API key exists: an ``api_key`` row in
   the ``openai-key`` namespace (``/login openai-key``) or the legacy
   ``OPENAI_API_KEY`` provider-class store row. A ChatGPT OAuth login does NOT
   count (its token is not valid at the audio endpoint).

"Stored" means PERSISTED ROWS ONLY for all three rungs (cascade-lane sign-off,
2026-10-01): rows in the encrypted store, and never a runtime/config override,
the process environment or the fallback resolver. See :func:`_probe_key` and
``AuthStore.has_persisted_credential``. Rungs 1-2 bind the probe only; the
executor's call-time key fetch keeps the full cascade. Rung 3's call-time key
follows its probe onto the same credential class (:func:`_openai_stt_key`).
4. ``provider_stt_superwhisper`` — reserved; ALWAYS unavailable, and no code
   path returns it (unit-pinned).
5. ``model_audio_sidecar`` — the selected model accepts audio input.
6. ``none`` — no path; the reason enumerates what is missing.

The credential probes are ``read_only=True`` everywhere: a probe must not
rotate account stickiness (or decide routing) for what is only a question.

**Availability caveat, deliberately loud:** rungs 1-3 answer "a credential
exists", NOT "the call will succeed". A refused key, an empty balance or a
model the account cannot reach all surface at call time — that is what the
executor's fall-forward is for.

**Executor** (:func:`transcribe_audio`) — walks the available STT rungs
(1→3) in order over one read of the audio, records an :class:`~local_operator.stt.SttAttempt`
per rung, and falls forward on every failure. When every available rung fails
it raises :class:`SttUnavailable` carrying the resolution, the attempts, and
the classified failure the route should report. When NO rung was available it
raises the same exception with no attempts — the route maps that case to the
structured 409 the surface reads to offer the audio door.

**Loopback rung (opt-in)** — a caller that passes ``local_stt_url`` (today only
the desktop voice routes, from ``voice.local_stt_url``) gets one more rung,
``local_stt_loopback``, appended LAST: an OpenAI-compatible
``/v1/audio/transcriptions`` on this machine's loopback (session-voice.md R8).
It is the key-free path, so it stands behind every credentialed rung exactly as
``local_tts_system`` stands behind the TTS ones, and a caller that passes
nothing resolves exactly as before. Availability is "configured and the host is
loopback"; whether the server is UP is a call-time fact, like every other rung.

**Token executor** (:func:`transcribe_backend`) — the same run for ONE NAMED
rung over in-memory bytes, with no walk and no fall-forward: the mobile
bridge's dispatch (``clients/stt.py``), where the surface already decided —
and advertised — which token runs. Settled against that bridge at the PR
convergence round; see the function for its failure contract.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Optional

from local_operator.clients._http import APIError
from local_operator.clients.radient import RadientClient
from local_operator.env import resolve_radient_api_base_url
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.radient_credentials import (
    has_persisted_radient_credential,
    resolve_radient_credential,
)
from local_operator.stt import (
    AudioPath,
    AudioPathResolution,
    RungAvailability,
    SttAttempt,
    SttOutcome,
)
from local_operator.stt import audio as stt_audio
from local_operator.stt.clients import (
    DEFAULT_TRANSCRIBE_TIMEOUT_S,
    ElevenLabsSttClient,
    OpenAiSttClient,
)
from local_operator.stt.errors import failure_status_and_detail

logger = logging.getLogger(__name__)

#: One STT rung's wire-call bound. The executor passes ``min`` of this and the
#: overall budget remaining, so a late rung never outlives the deadline.
STT_ATTEMPT_TIMEOUT_S = DEFAULT_TRANSCRIBE_TIMEOUT_S

#: The whole cascade's bound (three attempts, worst case). Rungs the deadline
#: can no longer fund are recorded as ``skipped`` rather than attempted.
STT_OVERALL_TIMEOUT_S = 120.0

#: The STT rungs, in cascade order. Deliberately NOT including SuperWhisper
#: (never emitted) or the model-audio rung (not an STT call).
STT_RUNG_PATHS = (
    AudioPath.PROVIDER_STT_RADIENT,
    AudioPath.PROVIDER_STT_ELEVENLABS,
    AudioPath.PROVIDER_STT_OPENAI,
)

#: The opt-in rung, walked after :data:`STT_RUNG_PATHS` when a caller supplies
#: a loopback URL (see the module docstring). Kept out of that tuple so the
#: token executor and every existing walk stay exactly as they were.
LOCAL_RUNG_PATH = AudioPath.LOCAL_STT_LOOPBACK

#: The model id sent to the loopback server. It is whatever that server loaded;
#: an OpenAI-compatible server requires the field, and mlx_audio's names the
#: model to load, so the default is the one measured for this rung (Parakeet
#: TDT 0.6B v3: exact transcript, 0.06 s warm, relay-voice R8).
LOCAL_STT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"

#: The loopback server takes no credential; the OpenAI client still sends a
#: bearer, so it sends this placeholder rather than any stored key. A real key
#: must never travel to a local process the operator merely pointed a URL at.
LOCAL_STT_PLACEHOLDER_KEY = "local"

#: Why the reserved rung reads unavailable. Kept as a constant because it is
#: surfaced verbatim and a test pins it.
SUPERWHISPER_REASON = (
    "SuperWhisper is reserved but unavailable: it has no non-interactive "
    "transcription interface."
)


class SttUnavailable(RuntimeError):
    """The cascade could not produce text.

    Carries everything each caller class needs:

    * ``resolution`` — the full rung report. ``attempts`` being empty means no
      STT rung was available at all, which is the route's 409 case; the surface
      then reads ``resolution.model_audio_capable`` to offer the audio door.
    * ``attempts`` — one entry per rung the walk actually spent (plus any the
      budget skipped).
    * ``error`` / ``status_code`` / ``detail`` — the chosen failure among the
      attempts, already classified by the shared mapper. ``None`` when no rung
      was attempted.
    """

    def __init__(
        self,
        message: str,
        *,
        resolution: AudioPathResolution,
        attempts: tuple[SttAttempt, ...] = (),
        error: Optional[APIError] = None,
        status_code: int | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.resolution = resolution
        self.attempts = attempts
        self.error = error
        self.status_code = status_code
        self.detail = detail


def _ensure_store(config_dir: Path | None, store: AuthStore | None) -> tuple[AuthStore, bool]:
    """The caller's store, or one this call owns (and must close).

    Mirrors ``resolve_radient_credential``'s ownership rule: a store created
    here is closed here. The db path mirrors it too (``config_dir/auth.db``).
    """
    if store is not None:
        return store, False
    db_path = (config_dir / "auth.db") if config_dir is not None else None
    return AuthStore(db_path, config_dir=config_dir), True


#: The credential namespace rung 3 reads: the ``openai-key`` login's own, holding
#: a platform API key. NOT ``openai`` -- that provider's only logins are ChatGPT
#: OAuth grants, and a ChatGPT token is not valid at ``/v1/audio/*``, so probing
#: it advertised a rung that could only 401 while the very key the speech login
#: stores never lit it (voicing S0 review round 1, MAJOR / S-1).
OPENAI_STT_NAMESPACE = "openai-key"

#: Rung 3 accepts API-key rows only, at probe AND call time. One constant so the
#: two cannot drift: an availability answer about one credential class and a
#: request sent with another is the defect this closes.
OPENAI_STT_KINDS = frozenset({"api_key"})


async def _openai_stt_key(store: AuthStore, session_id: str | None) -> str | None:
    """The API key rung 3 would send: ``openai-key`` rows, else the legacy store row.

    The legacy fallback is the provider-class STORE row ``lop credential update
    OPENAI_API_KEY`` writes (named by the registry's ``legacy_store_keys`` for
    ``openai-key``), kept so nobody who set the rung up that way loses it. It is an
    encrypted persisted secret, never the process environment. A ChatGPT OAuth row
    never answers (``OPENAI_STT_KINDS``).

    Never raises: ``None`` is "no key", which the probe reads as unavailable and
    the executor reports as the rung's own refusal.
    """
    return await store.get_persisted_api_key(
        OPENAI_STT_NAMESPACE, session_id, kinds=OPENAI_STT_KINDS
    )


async def _probe_key(store: AuthStore, provider: str, session_id: str | None) -> bool:
    """Whether ``provider`` is LOGGED IN, without letting a probe take the mic down.

    PERSISTED ROWS ONLY (cascade-lane sign-off, 2026-10-01): the question this
    answers is "advertise this rung?", and the 7-tier ``get_api_key`` it used to
    call answers a different one -- "what would a request authenticate with?" --
    so a runtime/config override or an exported ``OPENAI_API_KEY`` lit a rung the
    user never signed in to. :meth:`AuthStore.has_persisted_credential` reads
    stored rows (OAuth and ``api_key``) only, is ``read_only`` (the probe decides
    nothing, see module docstring) and never raises; the guard below is for a
    non-``AuthStore`` seam, and means the same thing: cannot tell -> unavailable.

    ONLY this probe changed for rungs 1-2: the executor's call-time fetch
    (``_run_radient_rung``, ``_run_elevenlabs_rung`` -> the full cascade) is
    untouched, so an operator's own export still runs a call that a signed-in rung
    would. RUNG 3 is the one deliberate exception (refinement of that split, cascade
    lane re-ack): its probe AND its call-time key both go through
    :func:`_openai_stt_key`, because a ChatGPT OAuth token is available to the
    full cascade but is not a credential the audio endpoint accepts.
    """
    try:
        return await store.has_persisted_credential(provider, session_id)
    except Exception:
        logger.warning("stt probe for %s failed; reporting the rung unavailable", provider)
        return False


def _model_capable(model: object | None) -> bool:
    """Whether the selected model accepts audio input.

    ``getattr`` on purpose: ``supports_audio_input`` lands on ``ModelSpec`` in
    phase 1 of this feature, and a resolver imported before that must read a
    missing attribute as "cannot take audio" (the safe direction), never crash
    a mic.
    """
    return model is not None and bool(getattr(model, "supports_audio_input", False))


def local_stt_rung(local_stt_url: str | None) -> RungAvailability:
    """The loopback rung's availability for ``local_stt_url`` (see module docstring).

    Loopback ONLY, checked here rather than trusted from the setting: the rung
    sends the operator's recorded voice to whatever the URL names, and "local"
    is the whole promise of a key-free rung. A LAN or remote host reads as
    unavailable with the reason, never as a quiet upload.
    """
    from urllib.parse import urlsplit

    from local_operator.server.utils.static_roots import is_loopback_host

    url = (local_stt_url or "").strip()
    if not url:
        return RungAvailability(
            LOCAL_RUNG_PATH, False, "No local transcription server is configured."
        )
    try:
        host = urlsplit(url).hostname
    except ValueError:
        host = None
    if not is_loopback_host(host):
        return RungAvailability(
            LOCAL_RUNG_PATH,
            False,
            "The local transcription server must be on this machine (localhost or 127.0.0.1).",
        )
    return RungAvailability(
        LOCAL_RUNG_PATH, True, "A local transcription server is configured on this machine."
    )


async def resolve_audio_path(
    *,
    config_dir: Path | None,
    base_url: str | None = None,
    model: object | None = None,
    session_id: str | None = None,
    store: AuthStore | None = None,
    local_stt_url: str | None = None,
) -> AudioPathResolution:
    """Decide which speech path a submission would take. See module docstring.

    ``local_stt_url`` opts in to the loopback rung, placed after the
    credentialed rungs and before the model-audio rung (it IS an STT call).
    ``None`` -- every caller but the voice routes -- leaves the rungs exactly as
    they were.
    """
    radient_base = resolve_radient_api_base_url(base_url)
    store, owned = _ensure_store(config_dir, store)
    try:
        try:
            # Persisted-only, like rungs 2-3 below (see ``_probe_key``): the
            # call-time ``resolve_radient_credential`` stays the executor's.
            radient_available = await has_persisted_radient_credential(
                config_dir, radient_base, store=store
            )
        except Exception:
            logger.warning("stt probe for radient failed; reporting the rung unavailable")
            radient_available = False
        elevenlabs_available = await _probe_key(store, "elevenlabs", session_id)
        try:
            openai_available = bool(await _openai_stt_key(store, session_id))
        except Exception:  # a non-AuthStore seam; the real store never raises
            logger.warning("stt probe for openai failed; reporting the rung unavailable")
            openai_available = False
    finally:
        if owned:
            store.close()

    model_capable = _model_capable(model)
    rungs = (
        RungAvailability(
            AudioPath.PROVIDER_STT_RADIENT,
            radient_available,
            "Signed in to Radient." if radient_available else "Not signed in to Radient.",
        ),
        RungAvailability(
            AudioPath.PROVIDER_STT_ELEVENLABS,
            elevenlabs_available,
            (
                "An ElevenLabs API key is stored."
                if elevenlabs_available
                else "No ElevenLabs API key is stored."
            ),
        ),
        RungAvailability(
            AudioPath.PROVIDER_STT_OPENAI,
            openai_available,
            "An OpenAI API key is stored." if openai_available else "No OpenAI API key is stored.",
        ),
        RungAvailability(AudioPath.PROVIDER_STT_SUPERWHISPER, False, SUPERWHISPER_REASON),
        *((local_stt_rung(local_stt_url),) if local_stt_url is not None else ()),
        RungAvailability(
            AudioPath.MODEL_AUDIO_SIDECAR,
            model_capable,
            (
                "The selected model accepts audio input."
                if model_capable
                else "The selected model does not accept audio input."
            ),
        ),
    )
    available = next((rung for rung in rungs if rung.available), None)
    if available is not None:
        return AudioPathResolution(
            path=available.path,
            reason=available.reason,
            rungs=rungs,
            model_audio_capable=model_capable,
        )
    if model is None:
        reason = (
            "No transcription provider is available and no model is selected " "for the audio path."
        )
    else:
        reason = (
            "No transcription provider is available and the selected model does "
            "not accept audio."
        )
    return AudioPathResolution(
        path=AudioPath.NONE, reason=reason, rungs=rungs, model_audio_capable=model_capable
    )


def _rung_label(path: AudioPath) -> str:
    return {
        AudioPath.PROVIDER_STT_RADIENT: "Radient",
        AudioPath.PROVIDER_STT_ELEVENLABS: "ElevenLabs",
        AudioPath.PROVIDER_STT_OPENAI: "OpenAI",
        AudioPath.LOCAL_STT_LOOPBACK: "The local transcription server",
    }.get(path, str(path))


async def _run_radient_rung(
    audio_path: Path,
    *,
    config_dir: Path | None,
    base_url: str,
    store: AuthStore,
    language: str | None,
    prompt: str | None,
    correlation_session_id: str | None = None,
) -> tuple[str, Any, str]:
    """Rung 1: the existing Radient route path, off the event loop.

    Wrapped in ``asyncio.to_thread`` because ``RadientClient`` is sync
    ``requests``; the legacy route calls it from an async handler directly,
    which is existing behaviour — new code does not block the loop. The
    executor's own ``wait_for`` bounds this call (a thread cannot be
    cancelled, but the cascade stops waiting on it either way).

    Returns ``(text, cost, provider)``: the hub's transcription envelope carries
    the versioned cost object inside ``result`` (PR-3), parsed by the ONE
    adapter that knows its field names, and its own ``provider`` field beside
    it — the emitting route reads the provider when the cost object does not
    state one (an amount-less older envelope still names the vendor; QA round
    1, Q3). ``None`` cost means the envelope carried none and the caller
    records the call with an UNSTATED amount, never a zero. The correlation id
    is minted per attempt, because one attempt is exactly one hub call: a
    cascade that fell forward must leave each charge joinable to the attempt
    that caused it. ``correlation_session_id`` is the CALLER's conversation id
    (attribution only — never credential scope, which is why it is a separate
    parameter from the walk's ``session_id``), sent as the client correlation
    header when grammatical so the hub's usage row can be joined back to the
    session (QA round 1, Q4).
    """
    from local_operator.clients.radient import new_correlation_id
    from local_operator.clients.radient_cost import cost_from_payload

    credential = await resolve_radient_credential(config_dir, base_url, store=store)
    client = RadientClient(api_key=credential, base_url=base_url)
    result = await asyncio.to_thread(
        client.create_transcription,
        file_path=str(audio_path),
        model=None,  # Radient's own configured default governs (legacy rule)
        prompt=prompt,
        response_format="json",
        temperature=0.0,
        language=language,
        provider=None,
        session_id=str(correlation_session_id or "") or None,
        client_request_id=new_correlation_id(),
    )
    cost = None
    try:
        cost = cost_from_payload(result.model_dump())
    except Exception:  # noqa: BLE001 — a cost parse must not fail a transcript
        cost = None
    provider = str(getattr(result, "provider", "") or "")
    return result.text or "", cost, provider


async def _run_elevenlabs_rung(
    audio: bytes,
    *,
    mime: str,
    store: AuthStore,
    session_id: str | None,
    language: str | None,
    prompt: str | None,
    timeout_s: float,
) -> tuple[str, Any, str]:
    """Rung 2: the user's own ElevenLabs key."""
    key = await store.get_api_key("elevenlabs", session_id, read_only=True)
    if not key:
        raise APIError("No ElevenLabs API key is stored.", status_code=None)
    client = ElevenLabsSttClient(key)
    result = await client.transcribe(
        audio, mime=mime, language=language, prompt=prompt, timeout_s=timeout_s
    )
    # No cost object on a BYO leg: the vendor is not the hub and reports no
    # charge in this family, and the envelope that would name it is the hub's
    # only — the route knows this leg from the path itself (PR-3).
    return result.text, None, ""


async def _run_openai_rung(
    audio: bytes,
    *,
    mime: str,
    store: AuthStore,
    session_id: str | None,
    language: str | None,
    prompt: str | None,
    timeout_s: float,
) -> tuple[str, Any, str]:
    """Rung 3: the user's own OpenAI API key (``openai-key`` rows, never ChatGPT OAuth)."""
    key = await _openai_stt_key(store, session_id)
    if not key:
        raise APIError("No OpenAI API key is stored.", status_code=None)
    client = OpenAiSttClient(key)
    result = await client.transcribe(
        audio, mime=mime, language=language, prompt=prompt, timeout_s=timeout_s
    )
    # See the ElevenLabs rung: no cost object for a BYO leg, and no provider
    # for the route to read — the path names it.
    return result.text, None, ""


async def _run_local_rung(
    audio: bytes,
    *,
    mime: str,
    local_stt_url: str,
    language: str | None,
    prompt: str | None,
    timeout_s: float,
) -> tuple[str, Any, str]:
    """The loopback rung: the OpenAI client pointed at the configured local server.

    ``base_url`` is normalised to the ``/v1`` root the client appends
    ``/audio/transcriptions`` to, so ``http://127.0.0.1:8000`` and
    ``http://127.0.0.1:8000/v1`` both work. The model is pinned (no fallback id):
    a local server that lacks it should say so, not be asked for ``whisper-1``.
    No cost and no provider: nothing was bought.
    """
    from local_operator.providers.local import normalize_base_url

    client = OpenAiSttClient(
        LOCAL_STT_PLACEHOLDER_KEY,
        base_url=normalize_base_url(local_stt_url),
        label="The local transcription server",
    )
    result = await client.transcribe(
        audio,
        mime=mime,
        model=LOCAL_STT_MODEL,
        language=language,
        prompt=prompt,
        timeout_s=timeout_s,
    )
    return result.text, None, ""


def _failed_attempt(path: AudioPath, exc: BaseException) -> SttAttempt:
    """One failed rung, with the shared mapper's sentence as the detail."""
    if isinstance(exc, APIError):
        _status, detail = failure_status_and_detail(exc, "upstream", rung=path)
    else:
        detail = str(exc) or exc.__class__.__name__
    return SttAttempt(path=path, outcome="failed", detail=detail)


def _choose_final_failure(
    failures: list[tuple[AudioPath, Optional[APIError]]],
) -> Optional[tuple[AudioPath, APIError]]:
    """Which rung's failure the all-failed report should carry.

    A payment refusal (402-class) wins over everything: it is the one failure
    whose remedy is rung-independent — the user tops up once and the whole
    cascade heals — so surfacing a per-rung key problem instead would send them
    to fix the wrong thing. With no payment refusal, the LAST failure is
    reported: the freshest rung is the one closest to "what happened when we
    tried". The rung travels with the error because the classification
    vocabulary is per-rung (Radient relay vs direct BYO).
    """
    for path, error in failures:
        if error is None:
            continue
        status, _detail = failure_status_and_detail(error, "upstream", rung=path)
        if status == 402:
            return path, error
    for path, error in reversed(failures):
        if error is not None:
            return path, error
    return None


async def transcribe_audio(
    audio_path: Path,
    *,
    config_dir: Path | None,
    base_url: str | None = None,
    session_id: str | None = None,
    store: AuthStore | None = None,
    model: object | None = None,
    language: str | None = None,
    prompt: str | None = None,
    correlation_session_id: str | None = None,
    local_stt_url: str | None = None,
) -> SttOutcome:
    """Run the STT cascade over one audio file. See module docstring.

    ``local_stt_url`` opts in to the loopback rung, walked last (see
    :func:`resolve_audio_path`).

    ``correlation_session_id`` (PR-3) is the CALLER's conversation id for the
    hub's client correlation header and the emitted spend record — deliberately
    NOT the ``session_id`` above: that one scopes credential lookup, a
    different question answered by the store (QA round 1, Q4).
    """
    radient_base = resolve_radient_api_base_url(base_url)
    store, owned = _ensure_store(config_dir, store)
    try:
        resolution = await resolve_audio_path(
            config_dir=config_dir,
            base_url=radient_base,
            model=model,
            session_id=session_id,
            store=store,
            local_stt_url=local_stt_url,
        )
        walkable = (*STT_RUNG_PATHS, LOCAL_RUNG_PATH)
        candidates = [
            rung.path for rung in resolution.rungs if rung.available and rung.path in walkable
        ]
        if not candidates:
            raise SttUnavailable(str(resolution.reason), resolution=resolution)

        deadline = time.monotonic() + STT_OVERALL_TIMEOUT_S
        attempts: list[SttAttempt] = []
        failures: list[tuple[AudioPath, Optional[APIError]]] = []
        audio: bytes | None = None
        mime = ""

        for path in candidates:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                attempts.append(
                    SttAttempt(
                        path=path,
                        outcome="skipped",
                        detail=(
                            "The overall transcription budget was spent before this "
                            "provider was reached."
                        ),
                    )
                )
                continue
            attempt_timeout = min(STT_ATTEMPT_TIMEOUT_S, remaining)
            try:
                if path == AudioPath.PROVIDER_STT_RADIENT:
                    attempt = _run_radient_rung(
                        audio_path,
                        config_dir=config_dir,
                        base_url=radient_base,
                        store=store,
                        language=language,
                        prompt=prompt,
                        correlation_session_id=correlation_session_id,
                    )
                else:
                    if audio is None:
                        audio = audio_path.read_bytes()
                        # CONTENT over extension (``media.py``'s own rule): the
                        # header is the stronger evidence of what the bytes
                        # are, and the suffix is only the fallback for a header
                        # the sniffer could not name — this upload is
                        # first-party, unlike an admission-gate block.
                        sniffed = stt_audio.sniff_audio(audio)
                        mime = (
                            sniffed.mime_type
                            if sniffed is not None
                            else stt_audio.mime_for_path(audio_path)
                        )
                    if path == LOCAL_RUNG_PATH:
                        attempt = _run_local_rung(
                            audio,
                            mime=mime,
                            local_stt_url=local_stt_url or "",
                            language=language,
                            prompt=prompt,
                            timeout_s=attempt_timeout,
                        )
                    elif path == AudioPath.PROVIDER_STT_ELEVENLABS:
                        attempt = _run_elevenlabs_rung(
                            audio,
                            mime=mime,
                            store=store,
                            session_id=session_id,
                            language=language,
                            prompt=prompt,
                            timeout_s=attempt_timeout,
                        )
                    else:
                        attempt = _run_openai_rung(
                            audio,
                            mime=mime,
                            store=store,
                            session_id=session_id,
                            language=language,
                            prompt=prompt,
                            timeout_s=attempt_timeout,
                        )
                # ONE bound for every rung, whatever the rung does with its own
                # timeouts: the deadline is the executor's promise, and a
                # client that outlived its own timeout must not be able to
                # extend the cascade.
                text, rung_cost, rung_provider = await asyncio.wait_for(
                    attempt, timeout=attempt_timeout
                )
            except asyncio.TimeoutError:
                # A timeout never reached an upstream status; it behaves like a
                # transport failure (502, text passed through).
                exc: BaseException = APIError(
                    f"{_rung_label(path)} did not respond within {attempt_timeout:.0f} s.",
                    status_code=None,
                )
                attempts.append(_failed_attempt(path, exc))
                failures.append((path, exc))
                continue
            except Exception as exc:
                attempts.append(_failed_attempt(path, exc))
                failures.append((path, exc if isinstance(exc, APIError) else None))
                continue
            attempts.append(SttAttempt(path=path, outcome="ok"))
            return SttOutcome(
                text=text,
                path=path,
                attempts=tuple(attempts),
                cost=rung_cost,
                provider=rung_provider,
            )

        chosen = _choose_final_failure(failures)
        status_code: int | None = None
        detail: str | None = None
        chosen_error: Optional[APIError] = None
        if chosen is not None:
            chosen_path, chosen_error = chosen
            status_code, detail = failure_status_and_detail(
                chosen_error, "upstream", rung=chosen_path
            )
        raise SttUnavailable(
            "The speech cascade could not produce a transcription.",
            resolution=resolution,
            attempts=tuple(attempts),
            error=chosen_error,
            status_code=status_code,
            detail=detail,
        )
    finally:
        if owned:
            store.close()


#: The rungs :func:`transcribe_backend` can run over IN-MEMORY bytes — the
#: mobile bridge's dispatch targets. The Radient leg is deliberately absent:
#: its caller (``clients/stt.py``) owns that adapter against ``RadientClient``
#: directly (the phone's Radient path needs no cascade machinery), and the
#: file-based :func:`transcribe_audio` remains the routes' whole-cascade entry.
_TOKEN_RUNNERS = {
    AudioPath.PROVIDER_STT_ELEVENLABS: _run_elevenlabs_rung,
    AudioPath.PROVIDER_STT_OPENAI: _run_openai_rung,
}


async def transcribe_backend(
    backend: str,
    audio: bytes,
    mime: str,
    *,
    config_dir: Path | None,
    session_id: str | None = None,
    store: AuthStore | None = None,
    language: str | None = None,
    prompt: str | None = None,
) -> SttOutcome:
    """Run ONE named rung over in-memory bytes — the mobile bridge's executor.

    THE SETTLED HALF OF THE #1734 SEAM (agents review convergence, B1): the
    phone dispatches the exact token its availability answer named
    (``clients/stt.py::transcribe_with_backend``), so this executor runs THAT
    rung and nothing else — a whole-cascade walk here would let a call land on
    a rung the surface never advertised. The file-based
    :func:`transcribe_audio` keeps owning fall-forward, budgets and the
    409-worthy no-rung report; this function serves callers that already hold
    the bytes and the decision.

    Availability is re-checked first through the same
    :func:`resolve_audio_path` the surface read, so a missing credential keeps
    the phone's typed "voice input isn't available" answer (this raises
    :class:`SttUnavailable`, which the bridge converts to its 503 class)
    instead of a bare provider 401 — while a credential deleted AFTER this
    check still surfaces the rung's own honest failure. An attempted rung's
    upstream failure propagates as its typed
    :class:`~local_operator.clients._http.APIError` (the phone's shared
    classifier maps 402/502 from it), and a hung rung is bounded by
    :data:`STT_ATTEMPT_TIMEOUT_S` exactly as the walk bounds it — as a
    transport failure, never a status.

    No ``model`` argument: the model-audio rung is not an STT call (see
    :data:`STT_RUNG_PATHS`), and the phone has no session model to offer.
    """
    try:
        path = AudioPath(backend)
    except ValueError:
        # Not a token this cascade knows. The NONE member keeps the walk's
        # lookups total (no rung matches it) while the sentence still quotes
        # the caller's own token.
        path = AudioPath.NONE
    runner = _TOKEN_RUNNERS.get(path)

    store, owned = _ensure_store(config_dir, store)
    try:
        resolution = await resolve_audio_path(
            config_dir=config_dir,
            model=None,
            session_id=session_id,
            store=store,
        )
        rung = next((r for r in resolution.rungs if r.path == path), None)
        if rung is None or runner is None or not rung.available:
            if rung is not None and not rung.available:
                reason = rung.reason
            else:
                reason = f"The speech-to-text cascade cannot run the {backend!r} path."
            raise SttUnavailable(reason, resolution=resolution)
        try:
            # The token runners return ``(text, cost, provider)`` like the walk's
            # rungs (PR-3); cost and provider are dropped here because this
            # executor's caller (the mobile bridge) records no spend channel yet
            # — the phone's route is not an emitter in this slice, and pretending
            # a cost it never records would be worse than not carrying it.
            text, _cost, _provider = await asyncio.wait_for(
                runner(
                    audio,
                    mime=mime,
                    store=store,
                    session_id=session_id,
                    language=language,
                    prompt=prompt,
                    timeout_s=STT_ATTEMPT_TIMEOUT_S,
                ),
                timeout=STT_ATTEMPT_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            # The walk's own convention: a timeout never reached an upstream
            # status, so it reads as a transport failure to the classifiers.
            raise APIError(
                f"{_rung_label(path)} did not respond within {STT_ATTEMPT_TIMEOUT_S:.0f} s.",
                status_code=None,
            ) from None
    finally:
        if owned:
            store.close()
    return SttOutcome(text=text, path=path, attempts=(SttAttempt(path=path, outcome="ok"),))
