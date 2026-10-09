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
        try:
            radient_ok = await has_persisted_radient_credential(
                config_dir, radient_base, store=store
            )
        except Exception:  # noqa: BLE001 - a probe must not take its caller down
            logger.warning("image probe for radient failed; reporting the rung unavailable")
            radient_ok = False
        fal_ok = bool(image_availability.fal_key(config_dir))
        openai_ok = bool(image_availability.openai_images_key(config_dir))
    finally:
        if owned:
            store.close()

    rungs = (
        RungAvailability(
            ImageRoute.RADIENT,
            radient_ok,
            "Signed in to Radient." if radient_ok else "Not signed in to Radient.",
        ),
        RungAvailability(
            ImageRoute.FAL,
            fal_ok,
            "A FAL key is stored." if fal_ok else "No FAL key is stored.",
        ),
        RungAvailability(
            ImageRoute.OPENAI,
            openai_ok,
            "An OpenAI API key is stored." if openai_ok else "No OpenAI API key is stored.",
        ),
    )
    available = next((rung for rung in rungs if rung.available), None)
    if available is not None:
        return ImageRouteResolution(
            route=ImageRoute(available.route), reason=available.reason, rungs=rungs
        )
    reason = (
        "No image provider is available: sign in to Radient (`/login radient`), "
        "store a FAL key (`lop login fal`) or export FAL_API_KEY, or store an "
        "OpenAI API key (`lop login openai-key`) or export OPENAI_API_KEY."
    )
    return ImageRouteResolution(route=ImageRoute.NONE, reason=reason, rungs=rungs)


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
