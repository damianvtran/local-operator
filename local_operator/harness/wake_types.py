"""The wake DTOs, in a module that costs nothing but pydantic to import.

WHY THIS MODULE EXISTS, AND WHY IT IS SO SMALL. ``WakeSchedule`` and ``DueWake``
used to be defined in :mod:`local_operator.harness.wake`, beside the live
scheduler. That is the right home for *documentation* and the wrong one for
*cost*: ``local_operator.harness.types`` needs ``WakeSchedule`` to annotate
``WakeSchedulerProtocol``, so importing the shared TYPES module pulled the whole
wake machinery — 203.3 ms of cumulative import time on this host, 40.1 ms of it
pydantic constructing its model classes — onto every entry point that touches
``harness.types``. Measured on the CLI path: ``lop --version`` spent 0.402 s
inside ``build_cli_parser`` with ``harness/wake.py`` as the largest single term,
for a command that never schedules anything.

So the SOURCE OF TRUTH for the two models moves here and :mod:`wake` imports
*from* this module and re-exports, which keeps exactly one definition of each
type. ``wake.WakeSchedule is wake_types.WakeSchedule`` is asserted by a test,
because two classes with one name would silently make ``isinstance`` checks
false at whichever call site imported the other one.

WHAT MAY BE ADDED HERE, and the rule is the point of the module: nothing that
imports the scheduler, asyncio, or any of this package's heavier layers.
Anything added here must stay import-cheap — pydantic and the stdlib only —
or it re-introduces the cost this module exists to avoid.

``MIN_WAKE_INTERVAL_MS`` lives here rather than in :mod:`wake` for the same
reason: ``WakeSchedule.every_ms`` constrains its field with it, so a copy in the
scheduler would be a second definition that the model could not see.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

#: A wake starts a full turn; sub-minute starves the user. Constrained at the
#: FIELD level rather than validated in the scheduler: ``WakeScheduler.load``
#: adopts schedules straight from the transcript, and every_ms == 0 in a
#: hand-edited file used to raise ZeroDivisionError inside pump(), killing the
#: scheduler with an unobserved exception. Invalid rows are dropped with a
#: warning in load().
MIN_WAKE_INTERVAL_MS = 60_000

#: The cap on a wake's self-prompt, enforced by ``build_wake_schedule``.
#:
#: It lives HERE rather than in :mod:`wake` for a measured reason: ``cli.py``'s
#: parser prints it in the ``wake create`` help text, and the parser is built on
#: EVERY command. Importing it from the scheduler module therefore pulled
#: ``harness.wake`` — and the pydantic model construction beneath it — onto
#: ``lop --version``, which printed a version and built a wake option group it
#: never used. A profile of that command showed ``harness/wake.py:1(<module>)``
#: at **1.214 s cumulative** as the single largest term in ``build_cli_parser``.
#: The constant is data with no dependencies, so it belongs with the DTOs.
MAX_WAKE_MESSAGE_CHARS = 2_000

#: How many schedules one session may hold. Beside :data:`MAX_WAKE_MESSAGE_CHARS`
#: for the same reason: the parser's help text names both shared caps, and a
#: second copy of either number is how they drift.
MAX_WAKE_SCHEDULES = 16


class WakeSchedule(BaseModel):
    """One scheduled wake. ``id`` is a stable per-session handle (``w1``…)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    message: str  # the self-prompt delivered on fire
    next_due_at: int  # epoch ms
    every_ms: int | None = Field(default=None, ge=MIN_WAKE_INTERVAL_MS)
    until_at: int | None = None  # hard stop
    limit: int | None = None  # retire after N deliveries
    fired_count: int = 0
    created_at: int = 0
    #: The desktop request that ARMED this row, when one did — its origin, not its
    #: provenance in the bookkeeping sense. It exists so that "did this request's
    #: write land?" has an exact answer: every other field of a row is something a
    #: legitimate concurrent writer changes (the session's own persist advances
    #: ``next_due_at``/``fired_count`` when the wake fires, and re-times it on
    #: catch-up), so a question asked by comparing content can be answered wrongly
    #: by a writer that did nothing but let the wake run — review round 4, R9.
    #: Absent for rows the agent's ``wake`` tool or the CLI created, which have no
    #: request id to record; those keep the id-plus-message fallback that
    #: ``wakes/arm.py`` documents.
    request_id: str | None = None


class DueWake(BaseModel):
    """A wake that is due right now, handed to the ``deliver`` callback."""

    model_config = ConfigDict(extra="forbid")

    schedule: WakeSchedule
    occurrence: int  # 1-based = fired_count + 1 at fire time
    planned_total: int | None = None
    final: bool = False


__all__ = [
    "MAX_WAKE_MESSAGE_CHARS",
    "MAX_WAKE_SCHEDULES",
    "MIN_WAKE_INTERVAL_MS",
    "DueWake",
    "WakeSchedule",
]
