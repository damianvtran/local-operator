"""Video generation lane: the detached t2v/i2v/v2v job over the artifact walk.

The operator's requirement this lane implements (media wave-2, video phase):
video generation is ASYNC where the work happens in the conversation — the
tool returns a receipt in milliseconds, the job runs detached and pins the
runtime, the turn can end, and on settle the completion lands as a normal
transcript event (a ``job_result`` row) that can open a review turn. The
frozen contract lives in the video phase's architecture note
(``video-contract.md``, session 4dc01d842c67) and §7.2 names the build order.

**Composition** (all reused, no new lifecycle):

- ``generate_video`` registers the walk as an :class:`AsyncJob` of type
  ``video`` (see :mod:`local_operator.video.jobs`) — residency, ``peek``,
  ``cancel`` and settle-time delivery come from the existing job manager.
- A ``video_job_v1`` transcript record makes the job durable across restarts
  (see :mod:`local_operator.video.records`); a check-wake raises a cold
  session to cold-follow it via the ``video_status`` tool.
- Rungs: the Radient hub first, then the user's FAL key
  (:mod:`local_operator.video.cascade`); both speak the SAME frozen stage
  vocabulary the four surfaces already render — no new words.
- Cancellation rides the job manager's cancel into the rung's bounded
  best-effort provider cancel (:mod:`local_operator.video.rungs` reuses the
  image lane's executor); steer is cancel + resubmit, composed by the caller.

**Import-light on purpose** (the ``imagegen``/``stt`` precedent): THIS module
is stdlib-only — the tool builders and the session factory import these names
on paths that must not drag in HTTP clients, credential stores or pydantic.
Heavier companions are imported by their own consumers:

- :mod:`local_operator.video.records` — the durable ``video_job_v1`` record
  (stdlib-only; the boot path reads it).
- :mod:`local_operator.video.models` — the provider family table.
- :mod:`local_operator.video.cascade` — resolver + executor.
- :mod:`local_operator.video.rungs` — the httpx-async rung executors and the
  provider poll/cancel plumbing.
- :mod:`local_operator.video.jobs` — the detached runner, the check-wake
  ensure/cancel and the completion-text builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# Re-exported under their generic names (session-build paths import this
# module; the generic types are stdlib-only, so no heavy import rides in).
from local_operator.artifacts import (
    ArtifactKind,
    JobAttempt,
    JobCancelled,
    JobOutcome,
    JobUnavailable,
    MediaAsset,
    RungAvailability,
)

__all__ = [
    "ArtifactKind",
    "JobAttempt",
    "MediaAsset",
    "RungAvailability",
    "VideoAttempt",
    "VideoGenerationCancelled",
    "VideoGenerationUnavailable",
    "VideoOutcome",
    "VideoRoute",
    "VideoRouteResolution",
]


class VideoRoute(StrEnum):
    """The closed vocabulary of video-generation routes.

    Values are the wire spellings and travel in attempt records, the tool's
    ``details`` and the ``video_job_v1`` record — pinned here rather than
    re-derived per surface. Ordering lives in ``video.cascade``'s
    ``VIDEO_RUNG_ORDER``; the hub comes first (the image lane's tiering
    posture), FAL second.
    """

    RADIENT = "radient"
    FAL = "fal"
    #: No usable route. Appears in availability/refusal payloads only.
    NONE = "none"


#: ``VideoAttempt`` / ``VideoOutcome`` are ALIASES of the generic records (one
#: shape, no isinstance traps — the image lane's rule). ``VideoOutcome``
#: carries ``kind="video"`` from every walk-produced outcome and the asset's
#: ``duration_s`` rides :class:`MediaAsset`.
VideoAttempt = JobAttempt
VideoOutcome = JobOutcome


@dataclass(frozen=True)
class VideoRouteResolution:
    """The resolver's report: the chosen route, why, and every rung's state."""

    route: VideoRoute
    reason: str
    #: Fixed cascade order, every route present.
    rungs: tuple[RungAvailability, ...]


class VideoGenerationCancelled(JobCancelled):
    """The job's abort signal fired between polls; no task cancellation involved.

    Deliberately NOT ``asyncio.CancelledError`` — that class is the loop's own
    machinery. The detached runner converts the clean spelling into the
    manager's cancelled carrier only AFTER its bounded provider cancel and the
    record write (see ``video.jobs``), so the durable record always states
    what actually happened at the provider.
    """


class VideoGenerationUnavailable(JobUnavailable):
    """The walk could not produce a video; carries the walk's record.

    ``attempts`` may be empty (no rung was available at all — the tool then
    names the setup remedies); ``resolution`` is the full rung report either
    way. Subclasses the generic :class:`JobUnavailable` so a consumer can
    catch either layer.
    """

    def __init__(
        self,
        message: str,
        *,
        resolution: VideoRouteResolution | None = None,
        attempts: tuple[JobAttempt, ...] = (),
    ) -> None:
        super().__init__(message)
        self.resolution = resolution
        self.attempts = attempts
