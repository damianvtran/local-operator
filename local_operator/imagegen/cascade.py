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

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from local_operator.clients._http import APIError
from local_operator.env import resolve_radient_api_base_url
from local_operator.imagegen import (
    ImageAttempt,
    ImageOutcome,
    ImageRoute,
    ImageRouteResolution,
    RungAvailability,
)
from local_operator.imagegen import rungs as image_rungs
from local_operator.imagegen import availability as image_availability
from local_operator.imagegen.errors import failure_reason_class
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
RUNG_LABELS = {
    ImageRoute.RADIENT: "Radient",
    ImageRoute.FAL: "FAL",
    ImageRoute.OPENAI: "OpenAI",
}


class ImageGenerationCancelled(Exception):
    """The abort signal fired between polls; no task cancellation involved.

    The rare-race path (design §4.3): the tool converts it to a clean
    ``is_error=True`` receipt and performs the best-effort provider cancel.
    Deliberately NOT ``asyncio.CancelledError`` — that class is the loop's own
    cancellation machinery and must propagate untouched.
    """


class ImageGenerationUnavailable(RuntimeError):
    """The cascade could not produce an image; carries the walk's record.

    Mirrors ``SttUnavailable``'s contract: ``attempts`` may be empty (no rung
    was available at all — the tool then names the setup remedies);
    ``resolution`` is the full rung report either way.
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
        return ImageRouteResolution(route=available.route, reason=available.reason, rungs=rungs)
    reason = (
        "No image provider is available: sign in to Radient (`/login radient`), "
        "store a FAL key (`lop login fal`) or export FAL_API_KEY, or store an "
        "OpenAI API key (`lop login openai-key`) or export OPENAI_API_KEY."
    )
    return ImageRouteResolution(route=ImageRoute.NONE, reason=reason, rungs=rungs)


def _make_pause(signal: "AbortSignal | None") -> image_rungs.PauseFn | None:
    """The abort-aware wait every rung's poll loop uses.

    ``None`` without a signal (library use, tests). With one: a zero-length
    wait is a plain abort check, and a real wait races ``signal.wait()``
    against the timeout — the signal winning raises
    :class:`ImageGenerationCancelled`, our own task's cancellation propagates
    untouched (that is the loop's path, not ours).
    """
    if signal is None:
        return None

    async def pause(seconds: float) -> None:
        if signal.aborted:
            raise ImageGenerationCancelled()
        try:
            await asyncio.wait_for(signal.wait(), timeout=max(0.0, seconds))
        except TimeoutError:
            return
        # The wait returned because the signal fired (not because the timeout
        # elapsed) — the user asked the walk to stop.
        raise ImageGenerationCancelled()

    return pause


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
    key = await _call_time_key(
        route, config_dir=config_dir, radient_base=radient_base, store=store
    )
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


def _attempt_message(exc: BaseException) -> str:
    text = str(exc).strip()
    return text or exc.__class__.__name__


def _status_code_of(exc: BaseException) -> int | None:
    return exc.status_code if isinstance(exc, APIError) else None


def _all_failed_message(attempts: list[ImageAttempt]) -> str:
    lines = ["Image generation failed on every available provider:"]
    for attempt in attempts:
        label = RUNG_LABELS.get(attempt.route, str(attempt.route))
        if attempt.outcome == "skipped":
            lines.append(f"- {label}: skipped — {attempt.message}")
        else:
            lines.append(f"- {label}: {attempt.message}")
    return "\n".join(lines)


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
    """Run the image cascade. See module docstring for the failure contract."""
    resolution = await resolve_image_route(config_dir, base_url=base_url, store=store)
    candidates = [rung.route for rung in resolution.rungs if rung.available]
    if not candidates:
        raise ImageGenerationUnavailable(resolution.reason, resolution=resolution)

    radient_base = resolve_radient_api_base_url(base_url)
    store, owned = _ensure_store(config_dir, store)
    handle = handle if handle is not None else image_rungs.CancelHandle()
    pause = _make_pause(signal)
    attempts: list[ImageAttempt] = []
    deadline = time.monotonic() + IMAGE_GENERATION_TIMEOUT_S
    try:
        for route in candidates:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                attempts.append(
                    ImageAttempt(
                        route=route,
                        outcome="skipped",
                        reason_class="timeout",
                        message=(
                            "The overall generation budget was spent before this "
                            "provider was reached."
                        ),
                    )
                )
                continue
            budget = min(IMAGE_RUNG_TIMEOUT_S, remaining)
            try:
                result = await asyncio.wait_for(
                    _run_route(
                        route,
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
                    ),
                    timeout=budget,
                )
            except (ImageGenerationCancelled, asyncio.CancelledError):
                # User cancellation is a STOP, not a failure: no failover.
                raise
            except TimeoutError:
                attempts.append(
                    ImageAttempt(
                        route=route,
                        outcome="failed",
                        reason_class="timeout",
                        message=f"{RUNG_LABELS.get(route, route)} exceeded its "
                        f"{int(budget)}s generation budget.",
                    )
                )
                continue
            except image_rungs.RungSkipped as exc:
                attempts.append(
                    ImageAttempt(
                        route=route,
                        outcome="skipped",
                        reason_class=exc.reason_class,
                        message=str(exc),
                    )
                )
                continue
            except Exception as exc:  # noqa: BLE001 - every rung failure fails forward
                attempts.append(
                    ImageAttempt(
                        route=route,
                        outcome="failed",
                        reason_class=failure_reason_class(exc),
                        message=_attempt_message(exc),
                        status_code=_status_code_of(exc),
                    )
                )
                continue
            attempts.append(ImageAttempt(route=route, outcome="ok"))
            return ImageOutcome(
                assets=tuple(result.assets),
                route=route,
                attempts=tuple(attempts),
                model=result.model,
                prompt=prompt,
                seed=seed,
                generation_id=result.generation_id,
                cost_usd=result.cost_usd,
            )
        raise ImageGenerationUnavailable(
            _all_failed_message(attempts), resolution=resolution, attempts=tuple(attempts)
        )
    finally:
        if owned:
            store.close()
