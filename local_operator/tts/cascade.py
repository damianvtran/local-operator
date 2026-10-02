"""The TTS cascade resolver and executor.

**Resolver** (:func:`resolve_voice_path`) — the frozen decision order, first
match wins (design note §5):

1. ``provider_tts_radient`` — Radient credentials resolve, through the same
   :func:`has_persisted_radient_credential` the STT resolver uses, so the two
   cascades cannot disagree about what "signed in" means.
2. ``provider_tts_elevenlabs`` — a stored ElevenLabs key exists.
3. ``provider_tts_openai`` — a stored OpenAI API key exists: an ``api_key`` row
   in the ``openai-key`` namespace (``/login openai-key``) or the legacy
   ``OPENAI_API_KEY`` provider-class store row. A ChatGPT OAuth login does NOT
   count (its token is not valid at the audio endpoint).
4. ``none`` — no path; the reason enumerates what is missing.

"Stored" means PERSISTED ROWS ONLY, exactly as in the STT cascade: rows in the
encrypted store, and never a runtime/config override, the process environment
or the fallback resolver (``AuthStore.has_persisted_credential``). An exported
``OPENAI_API_KEY`` therefore never lights a rung — which is the mobile
contract's "never advertised from ambient state" rule, and the finding the S0
slice fixed for STT.

The credential probes are ``read_only=True`` everywhere: a probe must not
rotate account stickiness (or decide routing) for what is only a question.

**Availability caveat, deliberately loud:** these rungs answer "a credential
exists", NOT "the call will succeed". A refused key, an empty balance or a
model the account cannot reach all surface at call time — that is what the
executor's fall-forward is for.

**Executor** (:func:`synthesize_speech`) — walks the available TTS rungs in
order, records a :class:`~local_operator.tts.TtsAttempt` per rung, and falls
forward on every failure, including a Radient 402: the user's credits being out
is not the platform being broken, so spending the user's OWN key is the right
economics. When every available rung fails it raises :class:`TtsUnavailable`
carrying the resolution, the attempts and the classified failure the route
should report.

The descriptor is IMMUTABLE ACROSS THE WALK and each attempt maps it at
attempt time (the map is model-aware, and the leg that runs is not known until
it is tried). A BYO attempt therefore sends what its own adapter produced, and
a hub attempt sends the descriptor itself and lets the hub map.
"""

from __future__ import annotations

import asyncio
import logging
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
from local_operator.tts import (
    RungAvailability,
    TtsAttempt,
    TtsOutcome,
    VoicePath,
    VoicePathResolution,
    adapters,
)
from local_operator.tts import clients as tts_clients
from local_operator.tts.descriptor import VoiceDescriptor

logger = logging.getLogger(__name__)

__all__ = [
    "TTS_OVERALL_TIMEOUT_S",
    "TTS_RUNG_PATHS",
    "TtsUnavailable",
    "resolve_voice_path",
    "synthesize_speech",
]

#: Upper bound on one whole synthesis. The executor gives each attempt its own
#: slice of what is left, so a hung vendor cannot consume the whole budget.
TTS_OVERALL_TIMEOUT_S = 120.0

#: The TTS rungs, in cascade order. Radient first (the platform's own plane),
#: then the user's own keys — the order is written HERE and nowhere else.
TTS_RUNG_PATHS = (
    VoicePath.PROVIDER_TTS_RADIENT,
    VoicePath.PROVIDER_TTS_ELEVENLABS,
    VoicePath.PROVIDER_TTS_OPENAI,
)

#: The credential namespace rung 3 reads: the ``openai-key`` login's own,
#: holding a platform API key. NOT ``openai`` — that provider's only logins are
#: ChatGPT OAuth grants, and a ChatGPT token is not valid at the audio endpoint.
OPENAI_TTS_NAMESPACE = "openai-key"

#: Rung 3 accepts API-key rows only, at probe AND call time. One constant so
#: the two cannot drift: an availability answer about one credential class and
#: a request sent with another is the defect this closes.
OPENAI_TTS_KINDS = frozenset({"api_key"})


class TtsUnavailable(RuntimeError):
    """The cascade could not produce audio.

    Carries everything each caller class needs:

    * ``resolution`` — the full rung report. ``attempts`` being empty means no
      TTS rung was available at all, which is the route's "not signed in /
      nothing configured" case.
    * ``attempts`` — one entry per rung the walk actually spent.
    * ``error`` — the chosen failure among the attempts, or ``None`` when no
      rung was attempted. The route classifies it with its own refusal
      sentences, so a Radient 402 and an ElevenLabs 402 are described the same
      way they are described today (they are the same class of thing to the
      user: top up, or use another provider).
    """

    def __init__(
        self,
        message: str,
        *,
        resolution: VoicePathResolution,
        attempts: tuple[TtsAttempt, ...] = (),
        error: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.resolution = resolution
        self.attempts = attempts
        #: The chosen failure among the attempts, or ``None`` when no rung was
        #: attempted. An ``APIError`` carries a status the route classifies; any
        #: other exception is a defect and surfaces as a 500 with its message,
        #: never as a misleading sign-in refusal.
        self.error = error


def _ensure_store(config_dir: Path | None, store: AuthStore | None) -> tuple[AuthStore, bool]:
    """The caller's store, or one this call owns (and must close).

    Mirrors the STT resolver's ownership rule: a store created here is closed
    here, and the db path is the same (``config_dir/auth.db``).
    """
    if store is not None:
        return store, False
    db_path = (config_dir / "auth.db") if config_dir is not None else None
    return AuthStore(db_path, config_dir=config_dir), True


async def _openai_tts_key(store: AuthStore, session_id: str | None) -> str | None:
    """The API key rung 3 would send, or ``None``.

    The same credential class as the probe, asked the same way, so an
    availability answer and a sent request can never be about two different
    rows. Never raises: ``None`` is "no key", which the probe reads as
    unavailable and the executor reports as the rung's own refusal.
    """
    return await store.get_persisted_api_key(
        OPENAI_TTS_NAMESPACE, session_id, kinds=OPENAI_TTS_KINDS
    )


async def _probe_key(store: AuthStore, provider: str, session_id: str | None) -> bool:
    """Whether ``provider`` is LOGGED IN, without letting a probe take the voice down.

    PERSISTED ROWS ONLY, exactly as the STT cascade does it: the question is
    "advertise this rung?", and a runtime/config override or an exported
    ``ELEVENLABS_API_KEY`` answers a different one. The guard is for a
    non-``AuthStore`` seam and means the same thing: cannot tell -> unavailable.
    """
    try:
        return await store.has_persisted_credential(provider, session_id)
    except Exception:
        logger.warning("tts probe for %s failed; reporting the rung unavailable", provider)
        return False


async def resolve_voice_path(
    *,
    config_dir: Path | None,
    base_url: str | None = None,
    session_id: str | None = None,
    store: AuthStore | None = None,
) -> VoicePathResolution:
    """Decide which text-to-speech path a request would take. See the module docstring."""
    radient_base = resolve_radient_api_base_url(base_url)
    store, owned = _ensure_store(config_dir, store)
    try:
        try:
            radient_available = await has_persisted_radient_credential(
                config_dir, radient_base, store=store
            )
        except Exception:
            logger.warning("tts probe for radient failed; reporting the rung unavailable")
            radient_available = False
        elevenlabs_available = await _probe_key(store, "elevenlabs", session_id)
        try:
            openai_available = bool(await _openai_tts_key(store, session_id))
        except Exception:  # a non-AuthStore seam; the real store never raises
            logger.warning("tts probe for openai failed; reporting the rung unavailable")
            openai_available = False
    finally:
        if owned:
            store.close()

    rungs = (
        RungAvailability(
            VoicePath.PROVIDER_TTS_RADIENT,
            radient_available,
            "Signed in to Radient." if radient_available else "Not signed in to Radient.",
        ),
        RungAvailability(
            VoicePath.PROVIDER_TTS_ELEVENLABS,
            elevenlabs_available,
            (
                "An ElevenLabs API key is stored."
                if elevenlabs_available
                else "No ElevenLabs API key is stored."
            ),
        ),
        RungAvailability(
            VoicePath.PROVIDER_TTS_OPENAI,
            openai_available,
            "An OpenAI API key is stored." if openai_available else "No OpenAI API key is stored.",
        ),
    )
    available = next((rung for rung in rungs if rung.available), None)
    if available is not None:
        return VoicePathResolution(
            path=available.path,
            reason=available.reason,
            rungs=rungs,
            servable=True,
        )
    return VoicePathResolution(
        path=VoicePath.NONE,
        reason=(
            "No text-to-speech provider is available: sign in to Radient, or store an "
            "ElevenLabs or OpenAI API key."
        ),
        rungs=rungs,
        servable=False,
    )


def _rung_label(path: VoicePath) -> str:
    return {
        VoicePath.PROVIDER_TTS_RADIENT: "Radient",
        VoicePath.PROVIDER_TTS_ELEVENLABS: "ElevenLabs",
        VoicePath.PROVIDER_TTS_OPENAI: "OpenAI",
    }.get(path, str(path))


def _failed_attempt(path: VoicePath, exc: BaseException) -> TtsAttempt:
    """One failed attempt's record: the rung, and the upstream's own words."""
    return TtsAttempt(path=path, outcome="failed", detail=str(exc)[:400])


def _choose_final_failure(
    failures: list[tuple[VoicePath, BaseException]],
) -> Optional[BaseException]:
    """Which rung's failure the all-failed report should carry.

    A payment refusal (402) wins over everything: it is the one failure whose
    remedy is rung-independent — the user tops up once and the whole cascade
    heals — so surfacing a per-rung key problem instead would send them to fix
    the wrong thing. With no payment refusal, the LAST failure is reported: the
    freshest rung is the one closest to "what happened when we tried". The STT
    cascade chooses by the same rule (``stt.cascade._choose_final_failure``).
    """
    for _path, error in failures:
        if isinstance(error, APIError) and error.status_code == 402:
            return error
    for _path, error in reversed(failures):
        return error
    return None


async def _run_radient_rung(
    text: str,
    *,
    config_dir: Path | None,
    base_url: str,
    store: AuthStore,
    descriptor: Optional[VoiceDescriptor],
    legacy: adapters.Legacy,
    response_format: str,
    client: Optional[RadientClient] = None,
) -> tuple[bytes, str, list[tuple[str, str]]]:
    """Rung 1: the hub, off the event loop.

    Returns ``(audio, serving provider, relayed headers)``. The client is sync
    ``requests``; ``asyncio.to_thread`` keeps it off the loop.
    ``X-Radient-Speech-Provider`` is the hub's own receipt for which leg
    actually served, which is what the echoed-actual-path rule reports: a
    descriptor-bearing call names no leg, so the hub's header is the only place
    that fact exists — and the rest of the ``X-Radient-Speech-*`` family (the
    map version, what was applied, what degraded) is relayed verbatim for the
    same reason.
    """
    if client is None:
        credential = await resolve_radient_credential(config_dir, base_url, store=store)
        client = RadientClient(api_key=credential, base_url=base_url)
    payload = (
        descriptor.to_wire(gender=descriptor.resolved_gender()) if descriptor is not None else None
    )
    audio, headers = await asyncio.to_thread(
        client.create_speech_response,
        text,
        model=None,  # the hub owns model choice, including on the descriptor path
        voice=legacy.voice or None,
        instructions=legacy.instructions or None,
        response_format=response_format,
        speed=legacy.speed,
        provider=None,  # the hub's cascade and leg choice stay server-owned
        language_code=legacy.language_code or None,
        voice_descriptor=payload,
    )
    serving = headers.get("X-Radient-Speech-Provider") or headers.get("x-radient-speech-provider")
    relayed = [
        (name, value)
        for name, value in headers.items()
        if name.lower().startswith("x-radient-speech-")
    ]
    return audio, (serving or "radient"), relayed


async def _run_elevenlabs_rung(
    text: str,
    *,
    store: AuthStore,
    session_id: str | None,
    descriptor: VoiceDescriptor,
    legacy: adapters.Legacy,
    response_format: str,
    timeout_s: float,
    client: Optional[Any] = None,
) -> tuple[bytes, str, list[tuple[str, str]]]:
    """Rung 2: the user's own ElevenLabs key, mapped by the vendored map.

    The call-time key fetch keeps the FULL credential cascade (an operator's
    own export should run a request), unlike the probe that decided to
    advertise the rung — the same split the STT rungs use.
    """
    key = await store.get_api_key("elevenlabs", session_id, read_only=True)
    if not key:
        raise APIError("No ElevenLabs API key is stored.", status_code=None)
    attempt = adapters.resolve("elevenlabs", tts_clients.ELEVENLABS_TTS_MODEL, descriptor, legacy)
    elevenlabs = client or tts_clients.ElevenLabsTtsClient(key)
    result = await elevenlabs.synthesize(text, params=attempt.params, timeout_s=timeout_s)
    return result.audio, result.provider, _mapping_headers(attempt)


async def _run_openai_rung(
    text: str,
    *,
    store: AuthStore,
    session_id: str | None,
    descriptor: VoiceDescriptor,
    legacy: adapters.Legacy,
    response_format: str,
    timeout_s: float,
    client: Optional[Any] = None,
) -> tuple[bytes, str, list[tuple[str, str]]]:
    """Rung 3: the user's own OpenAI API key, mapped by the vendored map."""
    key = await _openai_tts_key(store, session_id)
    if not key:
        raise APIError("No OpenAI API key is stored.", status_code=None)
    attempt = adapters.resolve("openai", tts_clients.OPENAI_TTS_MODEL, descriptor, legacy)
    openai_client = client or tts_clients.OpenAiTtsClient(key)
    result = await openai_client.synthesize(
        text,
        params=attempt.params,
        response_format=response_format,
        timeout_s=timeout_s,
    )
    return result.audio, result.provider, _mapping_headers(attempt)


def _mapping_headers(attempt: adapters.Attempt) -> list[tuple[str, str]]:
    """The ``X-Radient-Speech-*`` family for a leg THIS daemon mapped.

    A hub-served call relays the hub's own headers instead (it is the mapper
    there). Two implementations of one mapping still report one shape, which is
    the point of the vendored map: a client cannot tell which side mapped by
    reading these names.

    ``Degraded`` is omitted when nothing degraded rather than sent empty — the
    hub's contract is the same, and an empty header reads as "something was
    degraded to nothing".
    """
    headers: list[tuple[str, str]] = [
        ("X-Radient-Speech-Map", attempt.map_version),
        ("X-Radient-Speech-Applied", ",".join(attempt.applied_fields())),
    ]
    degraded = adapters.format_degraded(attempt.degraded_tokens())
    if degraded:
        headers.append(("X-Radient-Speech-Degraded", degraded))
    return headers


async def synthesize_speech(
    text: str,
    *,
    config_dir: Path | None,
    base_url: str | None = None,
    session_id: str | None = None,
    store: AuthStore | None = None,
    descriptor: Optional[VoiceDescriptor] = None,
    legacy: adapters.Legacy | None = None,
    response_format: str = "mp3",
    resolution: VoicePathResolution | None = None,
    radient_client: Optional[RadientClient] = None,
    elevenlabs_client: Optional[Any] = None,
    openai_client: Optional[Any] = None,
    timeout_s: float = TTS_OVERALL_TIMEOUT_S,
) -> TtsOutcome:
    """Synthesize ``text`` down the cascade. See the module docstring.

    Available rungs are attempted in order and the walk falls forward on every
    failure. Rungs with no credential are never attempted at all — that is the
    resolver's answer, and attempting one would spend a round trip to learn
    what the store already said.
    """
    radient_base = resolve_radient_api_base_url(base_url)
    store, owned = _ensure_store(config_dir, store)
    legacy = legacy or adapters.Legacy()
    try:
        if resolution is None:
            resolution = await resolve_voice_path(
                config_dir=config_dir,
                base_url=radient_base,
                session_id=session_id,
                store=store,
            )

        attempts: list[TtsAttempt] = []
        # EVERY failure is kept, including a non-``APIError`` fault: a defect in
        # one leg must still fall forward (one bug cannot take a spoken message
        # down), but if nothing else can serve it, the route has to be able to
        # report the real fault. Swallowing it here is what would turn a hub
        # crash into a misleading "you are not signed in".
        failures: list[tuple[VoicePath, BaseException]] = []
        deadline = asyncio.get_running_loop().time() + timeout_s

        for path in TTS_RUNG_PATHS:
            rung = next((r for r in resolution.rungs if r.path == path), None)
            if rung is None or not rung.available:
                continue
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                attempts.append(TtsAttempt(path=path, outcome="skipped", detail="budget exhausted"))
                continue
            attempt_timeout = max(1.0, min(remaining, 60.0))
            try:
                if path == VoicePath.PROVIDER_TTS_RADIENT:
                    call = _run_radient_rung(
                        text,
                        config_dir=config_dir,
                        base_url=radient_base,
                        store=store,
                        descriptor=descriptor,
                        legacy=legacy,
                        response_format=response_format,
                        client=radient_client,
                    )
                elif descriptor is None:
                    # A BYO leg has nothing to map without a descriptor, and an
                    # absent descriptor means "behave as before" — which the
                    # daemon can only do through the hub. Skipping is honest:
                    # the rung is available, it just has no request to send.
                    attempts.append(
                        TtsAttempt(
                            path=path,
                            outcome="skipped",
                            detail="no descriptor to map for a direct provider call",
                        )
                    )
                    continue
                elif path == VoicePath.PROVIDER_TTS_ELEVENLABS:
                    call = _run_elevenlabs_rung(
                        text,
                        store=store,
                        session_id=session_id,
                        descriptor=descriptor,
                        legacy=legacy,
                        response_format=response_format,
                        timeout_s=attempt_timeout,
                        client=elevenlabs_client,
                    )
                else:
                    call = _run_openai_rung(
                        text,
                        store=store,
                        session_id=session_id,
                        descriptor=descriptor,
                        legacy=legacy,
                        response_format=response_format,
                        timeout_s=attempt_timeout,
                        client=openai_client,
                    )
                # ONE bound for every rung, whatever the rung does with its own
                # timeouts: the deadline is the executor's promise.
                audio, provider, mapped_headers = await asyncio.wait_for(
                    call, timeout=attempt_timeout
                )
            except asyncio.TimeoutError:
                exc = APIError(
                    f"{_rung_label(path)} did not respond within {attempt_timeout:.0f} s.",
                    status_code=None,
                )
                attempts.append(_failed_attempt(path, exc))
                failures.append((path, exc))
                continue
            except APIError as exc:
                attempts.append(_failed_attempt(path, exc))
                failures.append((path, exc))
                continue
            except Exception as exc:
                # A non-APIError fault is a defect, not an upstream refusal:
                # logged with its stack, and the walk still falls forward so one
                # leg's bug cannot take a spoken message down. It is KEPT as the
                # candidate failure for the all-failed report.
                logger.exception("tts rung %s failed unexpectedly", path)
                attempts.append(_failed_attempt(path, exc))
                failures.append((path, exc))
                continue
            attempts.append(TtsAttempt(path=path, outcome="ok"))
            return TtsOutcome(
                audio=audio,
                path=path,
                attempts=tuple(attempts),
                provider=provider,
                speech_headers=(
                    *mapped_headers,
                    # The daemon's own receipt, ALWAYS last and always present:
                    # it names the rung that executed, which on a hub-served
                    # call is not the same thing as the provider the hub's own
                    # header names for a non-descriptor request.
                    ("X-Radient-Speech-Path", f"provider_tts_{provider}"),
                ),
            )

        # Every available rung failed (or was skipped for lack of a descriptor).
        raise TtsUnavailable(
            "The text-to-speech cascade could not produce audio.",
            resolution=resolution,
            attempts=tuple(attempts),
            error=_choose_final_failure(failures),
        )
    finally:
        if owned:
            store.close()
