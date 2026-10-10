"""The cascade resolver and executor.

**Resolver** (:func:`resolve_image_route`) — the frozen order, first match
wins (architect design, 2026-10-08): Radient → FAL → OpenAI → honest reason.

1. ``radient`` — a persisted Radient login exists
   (``has_persisted_radient_credential``, the same probe the executor's
   resolver reads, so the two cannot disagree about what "signed in" means).
2. ``fal`` — ``availability.fal_key`` resolves (login row → store row → env).
3. ``openai`` — ``availability.openai_images_key`` resolves (``api_key`` rows
   only — a ChatGPT OAuth grant is not valid at the images route — then the
   provider store row, then env).

**Availability caveat, deliberately loud** (mirrors ``stt/cascade``): a rung
answers "a credential exists", NOT "the call will succeed". A refused key, an
empty balance or an unreachable model surfaces at call time — that is what
the executor's fail-forward is for.

**Executor** (:func:`run_image_cascade`) — re-resolves at call time, walks the
available rungs in order, records one :class:`ImageAttempt` per rung the walk
spent (plus any the budget skipped), and fails FORWARD on every rung failure
EXCEPT:

* **user cancellation** — the abort signal firing between polls raises
  :class:`ImageGenerationCancelled` (the tool turns it into a receipt and its
  best-effort provider cancel), and task cancellation propagates untouched;
  neither fails over, because the user asked for the walk to stop.
* **local validation errors** — raised before dispatch by the tool.

When every rung fails, :class:`ImageGenerationUnavailable` carries the full
attempt list; the tool renders it as the error note.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from local_operator.artifacts import ArtifactKind, JobCancelled, JobSpec, JobUnavailable
from local_operator.artifacts import walk as artifacts_walk
from local_operator.artifacts.rung import CancelSupport, RungSpec
from local_operator.clients._http import APIError
from local_operator.env import resolve_radient_api_base_url
from local_operator.imagegen import (
    ImageAttempt,
    ImageOutcome,
    ImageRoute,
    ImageRouteResolution,
    RungAvailability,
)
from local_operator.imagegen import availability as image_availability
from local_operator.imagegen import rungs as image_rungs
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.radient_credentials import (
    has_persisted_radient_credential,
    resolve_radient_credential,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.harness.types import AbortSignal

logger = logging.getLogger(__name__)

#: One rung's own budget (design §9): generous, because image generation is
#: seconds-to-decades and the poll loop is cheap; the OVERALL budget below is
#: the worst case — one rung burns its budget and the next still starts.
IMAGE_RUNG_TIMEOUT_S = 240.0
#: The whole cascade's bound (design §9).
IMAGE_GENERATION_TIMEOUT_S = 300.0

#: Human labels for attempt lines (the tokens themselves travel in details).
#: ``str``-keyed on purpose: the members ARE their wire spellings, so string
#: routes (the generic walk's vocabulary) look up here without a conversion.
RUNG_LABELS: dict[str, str] = {
    ImageRoute.RADIENT: "Radient",
    ImageRoute.FAL: "FAL",
    ImageRoute.OPENAI: "OpenAI",
    ImageRoute.OPENAI_SUB: "ChatGPT plan",
    ImageRoute.GOOGLE: "Google",
    ImageRoute.XAI: "xAI",
    ImageRoute.OPENROUTER: "OpenRouter",
}

#: The resolver's fixed order, first match wins. APPEND-ONLY: the three routes
#: that shipped keep their exact positions (the v1 principle — an existing
#: user's path must not change), and each wave-2 breadth rung appends after
#: them in the order the manager signed off (openai-sub, google, xai,
#: openrouter). ``resolve_image_route`` iterates THIS constant; a new rung
#: goes through the add-a-rung checklist (design §5.3).
IMAGE_RUNG_ORDER: tuple[ImageRoute, ...] = (
    ImageRoute.RADIENT,
    ImageRoute.FAL,
    ImageRoute.OPENAI,
    ImageRoute.OPENAI_SUB,
    ImageRoute.GOOGLE,
    ImageRoute.XAI,
    ImageRoute.OPENROUTER,
)

#: Every rung's declaration — identity, capability, cancel support and cost
#: posture in ONE table (design D10), so an addition updates a spec entry
#: instead of rediscovering seven code sites. Values follow the design's §4
#: row and the manager's sign-off #2/#3:
#: - ``cancel_support`` is DECLARED, test-pinned, and read by no branch in v1;
#:   FAL's ``signal`` is per best available evidence (its cancel URL answers
#:   CANCELLATION_REQUESTED) with mid-run honour UNVERIFIED — declaration-only
#:   in v1, so nothing depends on it yet.
#: - ``cost`` labels where a future figure would come from: only ``reported``
#:   rungs can ever put a number in ``cost_usd`` (design D8); ``rate_table``
#:   entries are documentation with vendor provenance in the guide.
RUNG_SPECS: dict[str, RungSpec] = {
    ImageRoute.RADIENT: RungSpec(
        route=ImageRoute.RADIENT,
        label="Radient",
        kinds=frozenset({"image"}),
        capabilities=frozenset({"t2i", "i2i"}),
        cancel_support=CancelSupport.SIGNAL,
        cost="reported",
    ),
    ImageRoute.FAL: RungSpec(
        route=ImageRoute.FAL,
        label="FAL",
        kinds=frozenset({"image"}),
        capabilities=frozenset({"t2i", "i2i"}),
        cancel_support=CancelSupport.SIGNAL,
        cost="rate_table",
    ),
    ImageRoute.OPENAI: RungSpec(
        route=ImageRoute.OPENAI,
        label="OpenAI",
        kinds=frozenset({"image"}),
        capabilities=frozenset({"t2i"}),
        cancel_support=CancelSupport.NONE,
        cost="rate_table",
    ),
    ImageRoute.OPENAI_SUB: RungSpec(
        route=ImageRoute.OPENAI_SUB,
        label="ChatGPT plan",
        kinds=frozenset({"image"}),
        capabilities=frozenset({"t2i"}),
        cancel_support=CancelSupport.NONE,
        # Quota-funded: no cash figure exists, and none is ever synthesized
        # (design D8); the guide states the 3-5x quota burn.
        cost="subscription",
    ),
    ImageRoute.GOOGLE: RungSpec(
        route=ImageRoute.GOOGLE,
        label="Google",
        kinds=frozenset({"image"}),
        capabilities=frozenset({"t2i"}),
        cancel_support=CancelSupport.NONE,
        # Synchronous single request, no per-call figure; vendor rate table
        # (documentation only — see the guide + image-providers matrix).
        cost="rate_table",
    ),
    ImageRoute.XAI: RungSpec(
        route=ImageRoute.XAI,
        label="xAI",
        kinds=frozenset({"image"}),
        capabilities=frozenset({"t2i"}),
        cancel_support=CancelSupport.NONE,
        # REPORTED, not a rate table: the official OpenAPI schema's ``usage``
        # object (``oneOf [null, MediaUsage]``) REQUIRES
        # ``cost_in_usd_ticks`` whenever it is present, so a figure a provider
        # reported may ride ``cost_usd`` (design D8). This supersedes the
        # wave's initial ``rate_table`` pencil, taken before the schema was
        # read at implement time (2026-10-09).
        cost="reported",
    ),
    ImageRoute.OPENROUTER: RungSpec(
        route=ImageRoute.OPENROUTER,
        label="OpenRouter",
        kinds=frozenset({"image"}),
        capabilities=frozenset({"t2i"}),
        cancel_support=CancelSupport.NONE,
        # The docs' settlement shape carries ``usage.cost`` per request.
        cost="reported",
    ),
}


class ImageGenerationCancelled(JobCancelled):
    """The abort signal fired between polls; no task cancellation involved.

    The rare-race path (design §4.3): the tool converts it to a clean
    ``is_error=True`` receipt and performs the best-effort provider cancel.
    Deliberately NOT ``asyncio.CancelledError`` — that class is the loop's own
    cancellation machinery and must propagate untouched. A subclass of the
    generic :class:`~local_operator.artifacts.JobCancelled`: the walk treats
    every subclass as a STOP (no failover).
    """


class ImageGenerationUnavailable(JobUnavailable):
    """The cascade could not produce an image; carries the walk's record.

    Mirrors ``SttUnavailable``'s contract: ``attempts`` may be empty (no rung
    was available at all — the tool then names the setup remedies);
    ``resolution`` is the full rung report either way. Subclasses the generic
    :class:`~local_operator.artifacts.JobUnavailable` so a consumer can catch
    either layer.
    """

    def __init__(
        self,
        message: str,
        *,
        resolution: ImageRouteResolution | None = None,
        attempts: tuple[ImageAttempt, ...] = (),
    ) -> None:
        super().__init__(message)
        self.resolution = resolution
        self.attempts = attempts


def _ensure_store(config_dir: Path | None, store: AuthStore | None) -> tuple[AuthStore, bool]:
    """The caller's store, or one this call owns (and must close).

    Mirrors ``stt.cascade._ensure_store`` including the db-path spelling: a
    caller whose ``config_dir`` is not the ambient root must not silently
    read the ambient ``auth.db``.
    """
    if store is not None:
        return store, False
    db_path = (config_dir / "auth.db") if config_dir is not None else None
    return AuthStore(db_path, config_dir=config_dir), True


async def _probe_route(
    route: ImageRoute,
    *,
    config_dir: Path | None,
    radient_base: str,
    store: AuthStore,
) -> bool:
    """Whether ``route``'s credential exists — the resolver's per-rung probe.

    Radient's is the async persisted-credential check (shared with the
    executor's resolver, so the two cannot disagree about what "signed in"
    means); every other rung reads one of ``availability``'s sync, socket-free
    functions. A wave-2 rung's probe reads the credential class it SPENDS with
    — that rule is the reason the subscription rung probes the OAuth grant
    class, the deliberate inverse of the ``openai-key`` rule. A probe that
    cannot answer reads as "not available"; it must never take its caller
    down. An unhandled route is a programming error (order and probes are
    edited together) and degrades closed, not open.
    """
    if route == ImageRoute.RADIENT:
        try:
            return await has_persisted_radient_credential(config_dir, radient_base, store=store)
        except Exception:  # noqa: BLE001 - a probe must not take its caller down
            logger.warning("image probe for radient failed; reporting the rung unavailable")
            return False
    if route == ImageRoute.FAL:
        return bool(image_availability.fal_key(config_dir))
    if route == ImageRoute.OPENAI:
        return bool(image_availability.openai_images_key(config_dir))
    if route == ImageRoute.OPENAI_SUB:
        return image_availability.openai_subscription_grant(config_dir)
    if route == ImageRoute.GOOGLE:
        return bool(image_availability.google_key(config_dir))
    if route == ImageRoute.XAI:
        return image_availability.xai_available(config_dir)
    if route == ImageRoute.OPENROUTER:
        return bool(image_availability.openrouter_key(config_dir))
    logger.warning("no availability probe for image route %s; reporting unavailable", route)
    return False


#: Per-route availability reasons — the pair a user is shown (available /
#: not). The three v1 routes' strings are FROZEN (surfaces and tests quote
#: them); a new rung adds its pair here and appends its route to
#: IMAGE_RUNG_ORDER.
_ROUTE_REASONS: dict[ImageRoute, tuple[str, str]] = {
    ImageRoute.RADIENT: ("Signed in to Radient.", "Not signed in to Radient."),
    ImageRoute.FAL: ("A FAL key is stored.", "No FAL key is stored."),
    ImageRoute.OPENAI: ("An OpenAI API key is stored.", "No OpenAI API key is stored."),
    ImageRoute.OPENAI_SUB: (
        "A ChatGPT subscription sign-in is stored.",
        "No ChatGPT subscription sign-in is stored.",
    ),
    ImageRoute.GOOGLE: (
        "A Google AI Studio key is stored.",
        "No Google AI Studio key is stored.",
    ),
    ImageRoute.XAI: ("An xAI key or sign-in is stored.", "No xAI key or sign-in is stored."),
    ImageRoute.OPENROUTER: (
        "An OpenRouter key is stored.",
        "No OpenRouter key is stored.",
    ),
}


def _route_reason(route: ImageRoute, available: bool) -> str:
    yes, no = _ROUTE_REASONS.get(route, ("Available.", "Not available."))
    return yes if available else no


async def resolve_image_route(
    config_dir: Path | None = None,
    *,
    base_url: str | None = None,
    store: AuthStore | None = None,
) -> ImageRouteResolution:
    """Decide which route an image submission would take. See module docstring."""
    radient_base = resolve_radient_api_base_url(base_url)
    store, owned = _ensure_store(config_dir, store)
    try:
        rungs: list[RungAvailability] = []
        for route in IMAGE_RUNG_ORDER:
            ok = await _probe_route(
                route, config_dir=config_dir, radient_base=radient_base, store=store
            )
            rungs.append(RungAvailability(route, ok, _route_reason(route, ok)))
    finally:
        if owned:
            store.close()

    available = next((rung for rung in rungs if rung.available), None)
    if available is not None:
        return ImageRouteResolution(
            route=ImageRoute(available.route), reason=available.reason, rungs=tuple(rungs)
        )
    reason = (
        "No image provider is available: sign in to Radient (`/login radient`), "
        "store a FAL key (`lop login fal`) or export FAL_API_KEY, store an "
        "OpenAI API key (`lop login openai-key`) or export OPENAI_API_KEY, "
        "sign in to a ChatGPT plan (`lop login openai`), store a Google AI "
        "Studio key (`lop login google`) or export GOOGLE_AI_STUDIO_API_KEY, "
        "store an xAI key (`lop login xai`) or sign in to Grok "
        "(`lop login xai-oauth`), or store an OpenRouter key "
        "(`lop login openrouter`) or export OPENROUTER_API_KEY."
    )
    return ImageRouteResolution(route=ImageRoute.NONE, reason=reason, rungs=tuple(rungs))


def _make_pause(signal: "AbortSignal | None") -> image_rungs.PauseFn | None:
    """The abort-aware wait every rung's poll loop uses (this lane's binding).

    The generic helper lives in :mod:`local_operator.artifacts.walk`; this
    wrapper binds :class:`ImageGenerationCancelled` so the lane's pinned call
    sites (``test_cascade``) keep their exact spelling. ``None`` without a
    signal (library use, tests).
    """
    return artifacts_walk.make_pause(signal, ImageGenerationCancelled)


async def _call_time_key(
    route: ImageRoute,
    *,
    config_dir: Path | None,
    radient_base: str,
    store: AuthStore,
) -> str:
    """The bearer/key a rung would send, resolved at call time; raise when absent.

    Call-time re-resolution is the design's rule (§3.4): a rung is advertised
    from a stored credential, and the executor still asks the FULL resolver —
    so an operator's own export runs a call the gate would not light for
    Radient, and a credential revoked since the gate was built surfaces as a
    rung failure that fails forward rather than a silent no-op.
    """
    if route == ImageRoute.RADIENT:
        credential = await resolve_radient_credential(config_dir, radient_base, store=store)
        bearer = credential.get_secret_value()
        if not bearer:
            raise APIError(
                "No Radient credential is available.", status_code=None, code="unauthorized"
            )
        return bearer
    if route == ImageRoute.FAL:
        key = image_availability.fal_key(config_dir)
        if not key:
            raise APIError("No FAL key is available.", status_code=None, code="unauthorized")
        return key
    if route == ImageRoute.GOOGLE:
        key = await image_availability.google_call_key(store)
        if not key:
            raise APIError(
                "No Google AI Studio key is available.", status_code=None, code="unauthorized"
            )
        return key
    if route == ImageRoute.XAI:
        key = await image_availability.xai_call_bearer(store)
        if not key:
            raise APIError(
                "No xAI key or sign-in is available.", status_code=None, code="unauthorized"
            )
        return key
    if route == ImageRoute.OPENROUTER:
        key = await image_availability.openrouter_call_key(store)
        if not key:
            raise APIError("No OpenRouter key is available.", status_code=None, code="unauthorized")
        return key
    key = await image_availability.openai_call_key(store)
    if not key:
        raise APIError("No OpenAI API key is available.", status_code=None, code="unauthorized")
    return key


async def _run_route(
    route: ImageRoute,
    *,
    prompt: str,
    config_dir: Path | None,
    radient_base: str,
    store: AuthStore,
    source_url: str | None,
    strength: float | None,
    image_size: str,
    num_images: int,
    seed: int | None,
    model: str | None,
    handle: image_rungs.CancelHandle,
    emit: image_rungs.ProgressFn | None,
    pause: image_rungs.PauseFn | None,
    client: httpx.AsyncClient | None,
) -> image_rungs.RungResult:
    """Dispatch one rung, resolving its credential at call time."""
    if route == ImageRoute.OPENAI_SUB:
        # This rung's credential carries identity beyond the bearer (the
        # ``chatgpt-account-id`` the Codex backend wants), so it resolves its
        # own access record instead of going through ``_call_time_key``
        # (which returns a bare string). Same rule as every other rung: the
        # resolution happens HERE, at call time, and a missing or dead grant
        # surfaces as a rung failure that fails forward.
        #
        # The ``kind`` gate is load-bearing (reviewer round 1 F1 / QA Q1):
        # ``get_oauth_access`` is the chat path's CASCADE, and when an OAuth
        # row cannot mint a bearer it rotates to a sibling ``api_key`` row —
        # correct for chat, a leak for this rung, because that platform key
        # must never reach chatgpt.com (the whole point of the rung is that
        # the two credential classes stay delimited). Only a grant funds
        # this route; anything else is unauthorized and fails forward.
        access = await image_availability.openai_sub_access(store)
        if access is None or not access.access_token or access.kind != "oauth":
            raise APIError(
                "No ChatGPT subscription sign-in is available.",
                status_code=None,
                code="unauthorized",
            )
        return await image_rungs.run_openai_sub(
            prompt=prompt,
            access_token=access.access_token,
            account_id=access.org_id or access.account_id,
            num_images=num_images,
            image_size=image_size,
            source_url=source_url,
            seed=seed,
            model=model,
            emit=emit,
            pause=pause,
            client=client,
        )
    key = await _call_time_key(route, config_dir=config_dir, radient_base=radient_base, store=store)
    if route == ImageRoute.RADIENT:
        return await image_rungs.run_radient(
            prompt=prompt,
            base_url=radient_base,
            credential=key,
            num_images=num_images,
            image_size=image_size,
            seed=seed,
            strength=strength,
            source_url=source_url,
            model=model,
            handle=handle,
            emit=emit,
            pause=pause,
            client=client,
        )
    if route == ImageRoute.FAL:
        return await image_rungs.run_fal(
            prompt=prompt,
            key=key,
            num_images=num_images,
            image_size=image_size,
            seed=seed,
            strength=strength,
            source_url=source_url,
            model=model,
            handle=handle,
            emit=emit,
            pause=pause,
            client=client,
        )
    if route == ImageRoute.GOOGLE:
        return await image_rungs.run_google(
            prompt=prompt,
            key=key,
            num_images=num_images,
            image_size=image_size,
            source_url=source_url,
            seed=seed,
            model=model,
            emit=emit,
            pause=pause,
            client=client,
        )
    if route == ImageRoute.OPENROUTER:
        return await image_rungs.run_openrouter(
            prompt=prompt,
            key=key,
            num_images=num_images,
            image_size=image_size,
            source_url=source_url,
            seed=seed,
            model=model,
            emit=emit,
            pause=pause,
            client=client,
        )
    if route == ImageRoute.XAI:
        return await image_rungs.run_xai(
            prompt=prompt,
            key=key,
            num_images=num_images,
            image_size=image_size,
            source_url=source_url,
            seed=seed,
            model=model,
            emit=emit,
            pause=pause,
            client=client,
        )
    return await image_rungs.run_openai(
        prompt=prompt,
        key=key,
        num_images=num_images,
        image_size=image_size,
        source_url=source_url,
        model=model,
        emit=emit,
        pause=pause,
        client=client,
    )


# ``_attempt_message`` / ``_status_code_of`` / ``_all_failed_message`` /
# ``_emit_rung_failure`` moved to :mod:`local_operator.artifacts.walk` in media
# wave-2 (generic over kind and labels). This lane's byte-identical sentences
# and label lookups ride the walk's parameters: ``RUNG_LABELS`` below is the
# ``labels`` mapping it is given, and the all-failed header ("Image generation
# failed on every available provider:") is derived from the artifact kind.


async def run_image_cascade(
    *,
    prompt: str,
    config_dir: Path | None = None,
    base_url: str | None = None,
    store: AuthStore | None = None,
    source_url: str | None = None,
    strength: float | None = None,
    image_size: str = "square_hd",
    num_images: int = 1,
    seed: int | None = None,
    model: str | None = None,
    signal: "AbortSignal | None" = None,
    handle: image_rungs.CancelHandle | None = None,
    emit: image_rungs.ProgressFn | None = None,
    client: httpx.AsyncClient | None = None,
) -> ImageOutcome:
    """Run the image cascade. See module docstring for the failure contract.

    The walk itself is generic
    (:func:`local_operator.artifacts.walk.run_job_walk`); this function is the
    image lane's binding of it: the kind, the available candidates, the
    labels, the budgets, and a dispatch closure wired to THIS module's
    ``_run_route`` so the pinned monkeypatch seam keeps intercepting.
    """
    resolution = await resolve_image_route(config_dir, base_url=base_url, store=store)
    candidates = [rung.route for rung in resolution.rungs if rung.available]
    if not candidates:
        raise ImageGenerationUnavailable(resolution.reason, resolution=resolution)

    radient_base = resolve_radient_api_base_url(base_url)
    store, owned = _ensure_store(config_dir, store)
    handle = handle if handle is not None else image_rungs.CancelHandle()
    pause = _make_pause(signal)

    def _dispatch(route_str: str) -> Awaitable[image_rungs.RungResult]:
        """One rung, through the module-global ``_run_route`` AT CALL TIME.

        The module-global read is load-bearing: ``test_cascade`` monkeypatches
        ``cascade._run_route`` and every dispatch must be intercepted. The walk
        hands route STRINGS (its generic layer carries no enum); convert back
        to the ``ImageRoute`` member before calling the seam, because the
        lane's fakes and executors compare against the members.
        """
        return _run_route(
            ImageRoute(route_str),
            prompt=prompt,
            config_dir=config_dir,
            radient_base=radient_base,
            store=store,
            source_url=source_url,
            strength=strength,
            image_size=image_size,
            num_images=num_images,
            seed=seed,
            model=model,
            handle=handle,
            emit=emit,
            pause=pause,
            client=client,
        )

    def _on_exhausted(
        message: str, attempts: tuple[ImageAttempt, ...]
    ) -> ImageGenerationUnavailable:
        return ImageGenerationUnavailable(message, resolution=resolution, attempts=attempts)

    try:
        return await artifacts_walk.run_job_walk(
            kind=ArtifactKind.IMAGE,
            spec=JobSpec(
                kind=ArtifactKind.IMAGE,
                prompt=prompt,
                count=num_images,
                seed=seed,
                model=model,
            ),
            candidates=candidates,
            labels=RUNG_LABELS,
            call=_dispatch,
            handle=handle,
            emit=emit,
            pause=pause,
            tool="generate_image",
            rung_timeout_s=IMAGE_RUNG_TIMEOUT_S,
            overall_timeout_s=IMAGE_GENERATION_TIMEOUT_S,
            on_exhausted=_on_exhausted,
        )
    finally:
        if owned:
            store.close()
