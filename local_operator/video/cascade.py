"""The video cascade resolver and executor.

**Resolver** (:func:`resolve_video_route`) — the frozen order (video contract
§3.5): Radient (the hub) → FAL (the user's own key) → honest reason, first
match wins. Radient requires a persisted login (the same probe the image
lane's resolver reads — "signed in" cannot mean two things in one process);
FAL requires a stored row or an exported ``FAL_API_KEY``.

**Executor** (:func:`run_video_cascade`) — the watch's walk binding: modes are
derived from the request's source (image → ``i2v``, video → ``v2v``, none →
``t2v``), capabilities and source kinds come from ``RUNG_SPECS``, and the
generic :func:`local_operator.artifacts.walk.run_job_walk` carries the
fail-forward contract: per-rung budget, ``JobCancelled``/``CancelledError`` as
a STOP (no failover — the operator asked), every other failure recorded and
the next rung tried. Budgets are the contract's first-cut video numbers
(§3.3): 900 s per rung, 1800 s overall — a 30 s single-take model lives well
inside the second, and a slow 8 s model inside the first.

**Capability filter** (the media wave-2 edit lane's grammar, generalized): a
request in a mode with a source kind is walkable only on rungs whose specs
declare the mode AND accept that source kind; every other AVAILABLE rung is
pre-recorded as a ``skipped``/``unsupported`` attempt in its cascade position
and never called, and zero capable rungs raise the capable-providers remedy
instead of ever running a different shape (a v2v request must never silently
become a fresh t2v). The hub rung declares ``sources=NONE`` until the i2i
lane's enabling PR wires its server-side source mapping (§3.5) — the same
gate the image lane's ``RADIENT_EDIT_SKIP_MESSAGE`` documents.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from local_operator.artifacts import ArtifactKind, JobAttempt, JobOutcome, JobSpec
from local_operator.artifacts import walk as artifacts_walk
from local_operator.artifacts.rung import CancelSupport, RungSpec, SourceSupport
from local_operator.clients._http import APIError
from local_operator.env import resolve_radient_api_base_url
from local_operator.imagegen import availability as image_availability
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.radient_credentials import (
    has_persisted_radient_credential,
    resolve_radient_credential,
)
from local_operator.video import (
    RungAvailability,
    VideoGenerationCancelled,
    VideoGenerationUnavailable,
    VideoRoute,
    VideoRouteResolution,
)
from local_operator.video import rungs as video_rungs
from local_operator.video.rungs import RADIENT_VIDEO_SOURCE_SKIP_MESSAGE

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.harness.types import AbortSignal

logger = logging.getLogger(__name__)

__all__ = [
    "RADIENT_VIDEO_SOURCE_SKIP_MESSAGE",
    "VIDEO_GENERATION_TIMEOUT_S",
    "VIDEO_RUNG_LABELS",
    "VIDEO_RUNG_ORDER",
    "VIDEO_RUNG_SPECS",
    "VIDEO_RUNG_TIMEOUT_S",
    "capable_video_routes",
    "no_capable_message",
    "resolve_video_route",
    "run_video_cascade",
]

#: Per-rung and overall budgets (contract §3.3, first-cut values).
VIDEO_RUNG_TIMEOUT_S = 900.0
VIDEO_GENERATION_TIMEOUT_S = 1800.0

#: Human labels for progress lines and approval text.
VIDEO_RUNG_LABELS: dict[str, str] = {
    VideoRoute.RADIENT: "Radient",
    VideoRoute.FAL: "FAL",
}

#: The resolver's fixed order, first match wins. Hub first, then BYO-key —
#: the image lane's tiering posture verbatim. APPEND-ONLY.
VIDEO_RUNG_ORDER: tuple[VideoRoute, ...] = (
    VideoRoute.RADIENT,
    VideoRoute.FAL,
)

#: Every rung's declaration — identity, capability, cancel support and cost
#: posture in ONE table. Values follow the contract's §3.4/§3.5 rows:
#:
#: - Radient (the hub): t2v only until the enabling PR wires source mapping
#:   (the image lane's binding rule: a rung declares a source capability only
#:   in the same change that wires its mapping). ``cancel_support`` is
#:   QUEUED_ONLY: the hub's own RFC records that FAL removes queued jobs but
#:   only *signals* in-progress ones (K2), and ``CANCELLED`` is acknowledged,
#:   not a promise (K1) — the honesty clause in the cancel receipt reads this.
#:   ``cost="reported"``: the hub returns a submit quote and a settled figure.
#: - FAL: t2v/i2v/v2v with one source (the v1 wire sends at most one), source
#:   kinds per the contract's §3.7 pin set (i2v takes an image, v2v a video).
#:   ``cancel_support="signal"``: the queue's cancel URL exists for every
#:   submission (the response carries it, with derivation as fallback) and the
#:   best-effort cancel PUTs it; mid-run honour is FAL's, and the receipt
#:   reports what came back rather than promising a stop.
VIDEO_RUNG_SPECS: dict[str, RungSpec] = {
    VideoRoute.RADIENT: RungSpec(
        route=VideoRoute.RADIENT,
        label="Radient",
        kinds=frozenset({"video"}),
        capabilities=frozenset({"t2v"}),
        cancel_support=CancelSupport.QUEUED_ONLY,
        cost="reported",
        sources=SourceSupport.NONE,
    ),
    VideoRoute.FAL: RungSpec(
        route=VideoRoute.FAL,
        label="FAL",
        kinds=frozenset({"video"}),
        capabilities=frozenset({"t2v", "i2v", "v2v"}),
        cancel_support=CancelSupport.SIGNAL,
        cost="rate_table",
        sources=SourceSupport.SINGLE,
        max_sources=1,
        source_kinds=frozenset({"image", "video"}),
    ),
}

#: The hub's source refusal — the ONE home of this sentence, shared by the
#: rung's own defence-in-depth skip and the cascade's pre-record, so the two
#: cannot drift. WHY a skip and not an attempt: the hub's media route forwards
#: every body key verbatim to FAL, and the neutral-source → vendor-field
#: mapping for video does not exist server-side yet — so a source today is
#: either a silent, BILLED t2v or a rejection, and this client cannot tell
#: which. The enabling PR (hub exposes capability + maps sources, wire word
#: ``video_url`` for video and ``source_url`` for images) flips
#: ``VIDEO_RUNG_SPECS[RADIENT].sources`` beside wiring those fields.
#: Per-mode human phrases for refusals (the capable-providers remedy and the
#: incapable-rung skip sentence). One table so the sentences cannot drift.
_MODE_PHRASES: dict[str, str] = {
    "t2v": "generate a video from text",
    "i2v": "generate a video from an image",
    "v2v": "edit a video",
}

#: Per-route availability reasons — the pair a user is shown.
_ROUTE_REASONS: dict[VideoRoute, tuple[str, str]] = {
    VideoRoute.RADIENT: ("Signed in to Radient.", "Not signed in to Radient."),
    VideoRoute.FAL: ("A FAL key is stored.", "No FAL key is stored."),
}


def _route_reason(route: VideoRoute, available: bool) -> str:
    yes, no = _ROUTE_REASONS.get(route, ("Available.", "Not available."))
    return yes if available else no


def _ensure_store(config_dir: Path | None, store: AuthStore | None) -> tuple[AuthStore, bool]:
    """The caller's store, or one this call owns (and must close).

    Mirrors the image lane including the db-path spelling: a caller whose
    ``config_dir`` is not the ambient root must not silently read the ambient
    ``auth.db``.
    """
    if store is not None:
        return store, False
    db_path = (config_dir / "auth.db") if config_dir is not None else None
    return AuthStore(db_path, config_dir=config_dir), True


async def _probe_route(
    route: VideoRoute,
    *,
    config_dir: Path | None,
    radient_base: str,
    store: AuthStore,
) -> bool:
    """Whether ``route``'s credential exists — the resolver's per-rung probe.

    Same shape and same failure contract as the image lane's: a probe that
    cannot answer reads as "not available" and never takes its caller down.
    """
    if route == VideoRoute.RADIENT:
        try:
            return await has_persisted_radient_credential(config_dir, radient_base, store=store)
        except Exception:  # noqa: BLE001 - a probe must not take its caller down
            logger.warning("video probe for radient failed; reporting the rung unavailable")
            return False
    if route == VideoRoute.FAL:
        return bool(image_availability.fal_key(config_dir))
    logger.warning("no availability probe for video route %s; reporting unavailable", route)
    return False


async def resolve_video_route(
    config_dir: Path | None = None,
    *,
    base_url: str | None = None,
    store: AuthStore | None = None,
) -> VideoRouteResolution:
    """Decide which route a video submission would take. See module docstring."""
    radient_base = resolve_radient_api_base_url(base_url)
    store, owned = _ensure_store(config_dir, store)
    try:
        rungs: list[RungAvailability] = []
        for route in VIDEO_RUNG_ORDER:
            ok = await _probe_route(
                route, config_dir=config_dir, radient_base=radient_base, store=store
            )
            rungs.append(RungAvailability(route, ok, _route_reason(route, ok)))
    finally:
        if owned:
            store.close()

    available = next((rung for rung in rungs if rung.available), None)
    if available is not None:
        return VideoRouteResolution(
            route=VideoRoute(available.route), reason=available.reason, rungs=tuple(rungs)
        )
    reason = (
        "No video provider is available: sign in to Radient (`/login radient`) "
        "or store a FAL key (`lop login fal`) or export FAL_API_KEY."
    )
    return VideoRouteResolution(route=VideoRoute.NONE, reason=reason, rungs=tuple(rungs))


async def _call_time_key(
    route: VideoRoute,
    *,
    config_dir: Path | None,
    radient_base: str,
    store: AuthStore,
) -> str:
    """The bearer/key a rung would send, resolved at call time; raise when absent.

    Call-time re-resolution is the design's rule (§3.4): a rung is advertised
    from a stored credential, and the executor still asks the full resolver —
    so a credential revoked since the gate was built surfaces as a rung
    failure that fails forward rather than a silent no-op.
    """
    if route == VideoRoute.RADIENT:
        credential = await resolve_radient_credential(config_dir, radient_base, store=store)
        bearer = credential.get_secret_value()
        if not bearer:
            raise APIError(
                "No Radient credential is available.", status_code=None, code="unauthorized"
            )
        return bearer
    key = image_availability.fal_key(config_dir)
    if not key:
        raise APIError("No FAL key is available.", status_code=None, code="unauthorized")
    return key


async def _run_route(
    route: VideoRoute,
    *,
    mode: str,
    prompt: str,
    config_dir: Path | None,
    radient_base: str,
    store: AuthStore,
    source_kind: str | None,
    source_url: str | None,
    duration_s: int | None,
    resolution: str | None,
    aspect_ratio: str | None,
    audio: bool | None,
    seed: int | None,
    model: str | None,
    handle: video_rungs.CancelHandle,
    emit: video_rungs.ProgressFn | None,
    pause: video_rungs.PauseFn | None,
    on_accept: Callable[[video_rungs.CancelHandle], Awaitable[None]] | None = None,
    client: httpx.AsyncClient | None = None,
) -> video_rungs.RungResult:
    """Dispatch one rung, resolving its credential at call time."""
    key = await _call_time_key(route, config_dir=config_dir, radient_base=radient_base, store=store)
    if route == VideoRoute.RADIENT:
        return await video_rungs.run_radient_video(
            prompt=prompt,
            mode=mode,
            base_url=radient_base,
            credential=key,
            source_kind=source_kind,
            source_url=source_url,
            duration_s=duration_s,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            audio=audio,
            seed=seed,
            model=model,
            handle=handle,
            emit=emit,
            pause=pause,
            on_accept=on_accept,
            client=client,
        )
    return await video_rungs.run_fal_video(
        prompt=prompt,
        mode=mode,
        key=key,
        source_kind=source_kind,
        source_url=source_url,
        duration_s=duration_s,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        audio=audio,
        seed=seed,
        model=model,
        handle=handle,
        emit=emit,
        pause=pause,
        on_accept=on_accept,
        client=client,
    )


def capable_video_routes(
    resolution: VideoRouteResolution, mode: str, source_kind: str | None
) -> list[VideoRoute]:
    """Available routes whose specs declare ``mode`` (and ``source_kind``).

    THE one capability predicate — the executor's candidate filter, the
    submit-time pre-check and the approval describer all read it, so a v2v
    request cannot be walkable in one place and refused in another.
    """
    return [
        VideoRoute(rung.route)
        for rung in resolution.rungs
        if rung.available
        and mode in VIDEO_RUNG_SPECS[rung.route].capabilities
        and (source_kind is None or source_kind in VIDEO_RUNG_SPECS[rung.route].source_kinds)
    ]


def no_capable_message(mode: str, source_kind: str | None) -> str:
    """The zero-capable refusal — the sentence IS the remedy list.

    Every capable route by label (capability + source kind), so a user with
    no capable credential learns exactly what to add; the caller never falls
    back to a different request shape (contract §2.1).
    """
    phrase = _MODE_PHRASES.get(mode, "generate a video")
    capable = ", ".join(
        VIDEO_RUNG_SPECS[route].label
        for route in VIDEO_RUNG_ORDER
        if mode in VIDEO_RUNG_SPECS[route].capabilities
        and (source_kind is None or source_kind in VIDEO_RUNG_SPECS[route].source_kinds)
    )
    return f"No available provider can {phrase} (capable: {capable} — sign in or add a key)"


def _route_skip_message(route: str, mode: str, source_kind: str | None) -> str:
    """The pre-recorded skip's sentence for an available rung that cannot serve.

    Radient's is the SPECIFIC sentence from beside its rung (the one a
    signed-in user will meet first); the rest are generic.
    """
    if route == VideoRoute.RADIENT and source_kind is not None:
        return RADIENT_VIDEO_SOURCE_SKIP_MESSAGE
    phrase = _MODE_PHRASES.get(mode, "generate a video")
    return f"{VIDEO_RUNG_SPECS[route].label} cannot {phrase}."


async def run_video_cascade(
    *,
    prompt: str,
    mode: str,
    source_kind: str | None = None,
    source_url: str | None = None,
    duration_s: int | None = None,
    resolution: str | None = None,
    aspect_ratio: str | None = None,
    audio: bool | None = None,
    seed: int | None = None,
    model: str | None = None,
    config_dir: Path | None = None,
    base_url: str | None = None,
    store: AuthStore | None = None,
    signal: "AbortSignal | None" = None,
    handle: video_rungs.CancelHandle | None = None,
    emit: video_rungs.ProgressFn | None = None,
    on_accept: Callable[[video_rungs.CancelHandle], Awaitable[None]] | None = None,
    client: httpx.AsyncClient | None = None,
) -> JobOutcome:
    """Run the video cascade. See the module docstring for the failure contract.

    ``mode`` is derived by the caller (``t2v``/``i2v``/``v2v``) and must agree
    with ``source_kind``; the walk itself is generic and this function is the
    video lane's binding of it — kind, candidates, labels, budgets, and a
    dispatch closure wired to THIS module's ``_run_route`` (a module-global
    read at call time, so a test monkeypatch intercepts like the image lane's).
    """
    route_resolution = await resolve_video_route(config_dir, base_url=base_url, store=store)
    available = [rung.route for rung in route_resolution.rungs if rung.available]
    capable = capable_video_routes(route_resolution, mode, source_kind)
    if not capable:
        raise VideoGenerationUnavailable(
            no_capable_message(mode, source_kind), resolution=route_resolution
        )
    pre_attempts: dict[str, JobAttempt] = {
        route: JobAttempt(
            route=route,
            outcome="skipped",
            reason_class="unsupported",
            message=_route_skip_message(route, mode, source_kind),
        )
        for route in available
        if route not in capable
    }

    radient_base = resolve_radient_api_base_url(base_url)
    store, owned = _ensure_store(config_dir, store)
    handle = handle if handle is not None else video_rungs.CancelHandle()
    pause = artifacts_walk.make_pause(signal, VideoGenerationCancelled)

    def _dispatch(route_str: str) -> Awaitable[video_rungs.RungResult]:
        """One rung, through the module-global ``_run_route`` AT CALL TIME."""
        return _run_route(
            VideoRoute(route_str),
            mode=mode,
            prompt=prompt,
            config_dir=config_dir,
            radient_base=radient_base,
            store=store,
            source_kind=source_kind,
            source_url=source_url,
            duration_s=duration_s,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            audio=audio,
            seed=seed,
            model=model,
            handle=handle,
            emit=emit,
            pause=pause,
            on_accept=on_accept,
            client=client,
        )

    def _on_exhausted(message: str, attempts: tuple[JobAttempt, ...]) -> VideoGenerationUnavailable:
        return VideoGenerationUnavailable(message, resolution=route_resolution, attempts=attempts)

    try:
        return await artifacts_walk.run_job_walk(
            kind=ArtifactKind.VIDEO,
            spec=JobSpec(
                kind=ArtifactKind.VIDEO,
                prompt=prompt,
                count=1,
                seed=seed,
                model=model,
            ),
            candidates=available,
            labels=VIDEO_RUNG_LABELS,
            call=_dispatch,
            handle=handle,
            emit=emit,
            pause=pause,
            tool="generate_video",
            rung_timeout_s=VIDEO_RUNG_TIMEOUT_S,
            overall_timeout_s=VIDEO_GENERATION_TIMEOUT_S,
            on_exhausted=_on_exhausted,
            pre_attempts=pre_attempts,
        )
    finally:
        if owned:
            store.close()
