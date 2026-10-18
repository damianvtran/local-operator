"""The audio cascade resolver and executor.

**Resolver** (:func:`resolve_audio_route`) — the frozen order, first match
wins: Radient → ElevenLabs → OpenAI. Radient first because it is the
platform's own plane; the BYOK rungs follow, and the economics mirror the TTS
cascade (``tts/cascade.py``): the user's credits being out is not the platform
being broken, so a Radient refusal falls forward to the user's own key.

**Availability caveat, deliberately loud** (mirrors ``stt``/``tts``): a rung
answers "a credential exists", NOT "the call will succeed". A refused key, an
empty balance or a model the account cannot reach all surface at call time —
that is what the executor's fail-forward is for.

**Executor** (:func:`run_audio_cascade`) — re-resolves at call time, walks the
available rungs in order, records one attempt per rung the walk spent (plus
any the direction filter skipped), and fails FORWARD on every rung failure
except user cancellation (the kind's ``AudioGenerationCancelled`` /
``asyncio.CancelledError`` — a STOP, no failover) and local validation errors
(raised before dispatch by the tool). When every rung fails,
:class:`AudioGenerationUnavailable` carries the full resolution and the
attempt list; the tool renders it as the error note.

**The direction filter** is the capability gate the frozen contract implies:
``sound_effect`` runs on Radient and ElevenLabs only (OpenAI BYOK is voiceover
only), so an available-but-incapable rung is pre-recorded as a
``skipped``/``unsupported`` attempt in its cascade position and never called —
the same before-the-walk discipline the image edit lane applies to its
source-less rungs (``AUDIO_RUNG_DIRECTIONS`` is the one declaration).

The walk itself is generic
(:func:`local_operator.artifacts.walk.run_job_walk`); this module is the audio
binding: the kind, the candidates, the labels, the budgets and a dispatch
closure wired to THIS module's ``_run_route`` so the pinned monkeypatch seam
keeps intercepting.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from local_operator.artifacts import (
    ArtifactKind,
    JobAttempt,
    JobCancelled,
    JobSpec,
    JobUnavailable,
)
from local_operator.artifacts import walk as artifacts_walk
from local_operator.artifacts.rung import CancelHandle
from local_operator.audiogen import (
    AudioAttempt,
    AudioOutcome,
    AudioRoute,
    AudioRouteResolution,
    RungAvailability,
)
from local_operator.audiogen import availability as audio_availability
from local_operator.audiogen import rungs as audio_rungs
from local_operator.audiogen.models import AudioJobSpec
from local_operator.clients._http import APIError
from local_operator.env import resolve_radient_api_base_url
from local_operator.providers.radient_credentials import resolve_radient_credential

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.harness.types import AbortSignal

logger = logging.getLogger(__name__)

#: One rung's own budget: generous, because a voiceover can take tens of
#: seconds and the poll loop is cheap; the OVERALL budget below is the worst
#: case — one rung burns its budget and the next still starts.
AUDIO_RUNG_TIMEOUT_S = 180.0
#: The whole cascade's bound.
AUDIO_GENERATION_TIMEOUT_S = 240.0

#: The resolver's fixed order, first match wins (frozen design, 2026-10-17).
AUDIO_RUNG_ORDER: tuple[AudioRoute, ...] = (
    AudioRoute.RADIENT,
    AudioRoute.ELEVENLABS,
    AudioRoute.OPENAI,
)

#: Human labels for progress lines, attempt lists and approval text.
AUDIO_RUNG_LABELS: dict[str, str] = {
    AudioRoute.RADIENT: "Radient",
    AudioRoute.ELEVENLABS: "ElevenLabs",
    AudioRoute.OPENAI: "OpenAI",
}

#: Which directions each rung serves — the frozen contract's own boundaries
#: (OpenAI BYOK is voiceover only; there is no direct sound-effect path).
#: APPEND-ONLY in spirit: a rung's row changes only in the change that wires
#: the new path, which a test walks in both directions.
AUDIO_RUNG_DIRECTIONS: dict[str, frozenset[str]] = {
    AudioRoute.RADIENT: frozenset({"voiceover", "sound_effect"}),
    AudioRoute.ELEVENLABS: frozenset({"voiceover", "sound_effect"}),
    AudioRoute.OPENAI: frozenset({"voiceover"}),
}


class AudioGenerationCancelled(JobCancelled):
    """The in-flight job stopped because the USER asked it to (Esc / steer)."""


class AudioGenerationUnavailable(JobUnavailable):
    """The cascade could not produce an artifact.

    Carries the resolver's full report (``resolution``) and one attempt per
    rung the walk spent; ``attempts`` empty means no rung was available at
    all, which is the tool's "no provider configured" case.
    """


def resolve_audio_route(
    *,
    direction: str,
    config_dir: Path | None = None,
    base_url: str | None = None,
) -> AudioRouteResolution:
    """Decide which audio route a request would take. See the module docstring.

    Sync and socket-free, deliberately: both TOOLS' createIf gates and this
    resolver read the same probes, so the tool a session holds and the route a
    call takes cannot disagree about what exists. Call-time credential
    resolution (which may refresh an OAuth grant) happens in the dispatch
    closure, exactly as the image lane splits the two.
    """
    del base_url  # the probe is account-level; the run resolves the destination
    probes = (
        (AudioRoute.RADIENT, audio_availability.radient_available(config_dir)),
        (AudioRoute.ELEVENLABS, bool(audio_availability.elevenlabs_key(config_dir))),
        (AudioRoute.OPENAI, bool(audio_availability.openai_audio_key(config_dir))),
    )
    reasons = {
        AudioRoute.RADIENT: (
            "Signed in to Radient." if probes[0][1] else "Not signed in to Radient."
        ),
        AudioRoute.ELEVENLABS: (
            "An ElevenLabs API key is stored."
            if probes[1][1]
            else "No ElevenLabs API key is stored."
        ),
        AudioRoute.OPENAI: (
            "An OpenAI API key is stored." if probes[2][1] else "No OpenAI API key is stored."
        ),
    }
    rungs = tuple(
        RungAvailability(route=str(route), available=available, reason=reasons[route])
        for route, available in probes
    )
    for route, available in probes:
        if not available:
            continue
        if direction not in AUDIO_RUNG_DIRECTIONS[route]:
            continue
        return AudioRouteResolution(route=route, reason=reasons[route], rungs=rungs, servable=True)
    servable = any(
        available and direction in AUDIO_RUNG_DIRECTIONS[route] for route, available in probes
    )
    remedy = (
        "Sign in to Radient, or store an ElevenLabs or OpenAI API key."
        if direction == "voiceover"
        else "Sign in to Radient, or store an ElevenLabs API key (sound effects need one)."
    )
    return AudioRouteResolution(
        route=AudioRoute.NONE,
        reason=f"No audio provider is available for this request: {remedy}",
        rungs=rungs,
        servable=servable,
    )


def _direction_skip_message(route: str, direction: str) -> str:
    """The pre-recorded skip sentence for a rung that cannot serve ``direction``."""
    label = AUDIO_RUNG_LABELS.get(route, str(route))
    if direction == "sound_effect":
        return f"{label} has no sound-effect endpoint; it serves voiceovers only."
    return f"{label} cannot serve this request."


async def _run_route(
    route: AudioRoute,
    *,
    spec: AudioJobSpec,
    config_dir: Path | None,
    radient_base: str,
    handle: CancelHandle,
    emit: audio_rungs.ProgressFn | None,
    pause: artifacts_walk.PauseFn | None,
    client: httpx.AsyncClient | None,
) -> audio_rungs.RungResult:
    """One rung: resolve its credential at CALL time, then run it.

    The credential is resolved here rather than at resolve time because the
    two answer different questions: availability decides whether a tool
    EXISTS; this decides what a request authenticates with, with the FULL
    precedence (an exported key genuinely runs the call; a required OAuth
    refresh happens once, here). Each rung uses the same probe class it was
    advertised under, so a rung is never offered with one credential class and
    spent with a refused one.
    """
    if route == AudioRoute.RADIENT:
        credential = await resolve_radient_credential(config_dir, radient_base)
        secret = credential.get_secret_value()
        if not secret:
            raise APIError(
                "No Radient credential is available; sign in with `lop login radient`.",
                status_code=None,
            )
        return await audio_rungs.run_radient(
            spec,
            base_url=radient_base,
            credential=secret,
            handle=handle,
            emit=emit,
            pause=pause,
            client=client,
        )
    if route == AudioRoute.ELEVENLABS:
        key = audio_availability.elevenlabs_key(config_dir)
        if not key:
            raise APIError("No ElevenLabs API key is stored.", status_code=None)
        return await audio_rungs.run_elevenlabs(
            spec, api_key=key, handle=handle, emit=emit, pause=pause, client=client
        )
    if route == AudioRoute.OPENAI:
        key = audio_availability.openai_audio_key(config_dir)
        if not key:
            raise APIError("No OpenAI API key is stored.", status_code=None)
        return await audio_rungs.run_openai(
            spec, api_key=key, handle=handle, emit=emit, pause=pause, client=client
        )
    raise APIError(f"No audio rung is wired for route {route!r}.", status_code=None)


async def run_audio_cascade(
    *,
    spec: AudioJobSpec,
    config_dir: Path | None = None,
    base_url: str | None = None,
    signal: "AbortSignal | None" = None,
    handle: CancelHandle | None = None,
    emit: audio_rungs.ProgressFn | None = None,
    client: httpx.AsyncClient | None = None,
) -> AudioOutcome:
    """Run the audio cascade. See the module docstring for the failure contract."""
    resolution = resolve_audio_route(
        direction=spec.direction, config_dir=config_dir, base_url=base_url
    )
    available = [rung.route for rung in resolution.rungs if rung.available]
    pre_attempts: dict[str, JobAttempt] = {
        route: JobAttempt(
            route=route,
            outcome="skipped",
            reason_class="unsupported",
            message=_direction_skip_message(route, spec.direction),
        )
        for route in available
        if spec.direction not in AUDIO_RUNG_DIRECTIONS[route]
    }
    capable = [route for route in available if route not in pre_attempts]
    if not capable:
        raise AudioGenerationUnavailable(resolution.reason, resolution=resolution)
    radient_base = resolve_radient_api_base_url(base_url)
    handle = handle if handle is not None else CancelHandle()
    pause = artifacts_walk.make_pause(signal, AudioGenerationCancelled)

    def _dispatch(route_str: str) -> Awaitable[audio_rungs.RungResult]:
        """One rung, through the module-global ``_run_route`` AT CALL TIME.

        The module-global read is load-bearing: ``test_cascade`` monkeypatches
        ``cascade._run_route`` and every dispatch must be intercepted. The walk
        hands route STRINGS (its generic layer carries no enum); convert back
        to the ``AudioRoute`` member before calling the seam, because the
        lane's fakes and executors compare against the members.
        """
        return _run_route(
            AudioRoute(route_str),
            spec=spec,
            config_dir=config_dir,
            radient_base=radient_base,
            handle=handle,
            emit=emit,
            pause=pause,
            client=client,
        )

    def _on_exhausted(
        message: str, attempts: tuple[AudioAttempt, ...]
    ) -> AudioGenerationUnavailable:
        return AudioGenerationUnavailable(message, resolution=resolution, attempts=attempts)

    return await artifacts_walk.run_job_walk(
        kind=ArtifactKind.AUDIO,
        spec=JobSpec(
            kind=ArtifactKind.AUDIO,
            prompt=spec.text,
            count=1,
            seed=None,
            model=spec.model,
        ),
        # candidates is EVERY available rung in cascade order, not just the
        # capable ones: the walk records a pre-recorded route at its own
        # cascade position (``pre_attempts``) without dispatching it, which is
        # what keeps the attempt list in order (the image lane's rule).
        candidates=available,
        labels=AUDIO_RUNG_LABELS,
        call=_dispatch,
        handle=handle,
        emit=emit,
        pause=pause,
        tool=spec.tool,
        rung_timeout_s=AUDIO_RUNG_TIMEOUT_S,
        overall_timeout_s=AUDIO_GENERATION_TIMEOUT_S,
        on_exhausted=_on_exhausted,
        pre_attempts=pre_attempts,
        action="generation",
    )
