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

from typing import Literal

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
    #: When the row was created. For patience rows this is the EPISODE START and
    #: is preserved across a continuation's re-arms — the episode TTL
    #: (``proactive.patience.episode_ttl_ms``) is measured from it, and the
    #: fire/retire decision reads it back even when the row was retired and a
    #: fresh snapshot written.
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
    #: Whether this wake's turn completion NOTIFIES (docs/design/monitor-tool.md
    #: §14): the value rides the delivery's ``details`` and the session ORs it
    #: over a run's wake/monitor deliveries. Default False — a wake is quiet
    #: unless its arm asked to be told ("pass notify:true when the user asked to
    #: be told", the wake tool's guidance); an errored turn notifies regardless.
    #: Additive, with the usual skew rule: a build that predates the field drops
    #: a row carrying it (``extra="forbid"`` + ``load()``'s drop-with-warning),
    #: and a new build reads an old row as quiet — how those rows behaved.
    notify: bool = False

    # ---- patience fields (proactive-class agents; R29–R38) ------------------
    #
    # A patience wait is a HIDDEN internal timer attached to a message an agent
    # sent: it exists so the agent can notice "no reply arrived" and decide
    # whether to follow up, under hard bounds. It rides this same schedule
    # engine deliberately (one writer, one supervisor, one persist path — see
    # the design's §8.2.1): ``kind`` is what makes it a different MECHANISM on
    # shared substrate, and ``hidden`` is what keeps it out of every rendered
    # surface while its fire text still reaches the model's context.
    #
    # COST, STATED: ``extra="forbid"`` means an OLD build drops patience rows
    # on load. They are minutes-lived and the design accepted this; the PR that
    # added them states it too.

    #: ``scheduled`` is an ordinary wake (today's behaviour exactly, and what
    #: every pre-patience row reads as). ``patience`` rows are hidden,
    #: created only by the ``patience`` tool / ``send(patience=…)``, and
    #: filtered out of every human-facing listing and count.
    #:
    #: ``ask_timeout`` is the queued ask's deadline (design
    #: ``docs/design/ask-nonblocking.md`` §2.2, D10): also hidden, also filtered
    #: out of every human listing (the SAME ``is_internal_wake_row`` predicate),
    #: and created only by the ask queue. It rides this engine so a deadline
    #: survives a runtime that does not exist — the supervisor engages a runtime
    #: for the row, and that runtime's boot ``reconcile`` does the delivery, so
    #: the fire itself carries no payload (the ``WakeErrand`` rule). Adding a
    #: literal here is a LOAD change, not a persist-only change:
    #: ``extra="forbid"`` plus this Literal means an older build drops such a row
    #: on load, exactly as it does a patience row, and the queue then degrades to
    #: its in-runtime timer.
    kind: Literal["scheduled", "patience", "ask_timeout"] = "scheduled"
    #: Hidden deliveries emit no receipt event and are skipped by replay — the
    #: requirement is "no wake line, no card, no badge, no timer notification"
    #: while the TEXT stays in the model's context (``harness/render.py`` turns
    #: ``wake_prompt`` custom messages into user messages either way).
    hidden: bool = False
    #: One episode = a chain of unanswered waits (R31/R35). ``episode_id``
    #: equals the row's own id for patience rows; the FIRE carries it into the
    #: transcript so a continuation after the row retired can still be
    #: classified from the shared truth (restart-safe).
    episode_id: str = ""
    #: Which outbound message this wait belongs to: 1 = the message that
    #: started the episode; each re-arm increments it. ``max_attempts`` bounds
    #: it (R35).
    attempt: int = 0
    #: When THIS attempt was armed (epoch ms). The cancel-on-reply watermark
    #: compares it against the session's last real user/peer inbound: a fire
    #: whose ``armed_at`` predates that inbound is retired silently — the
    #: cross-runtime half of R34 that cannot rely on the in-memory cancel.
    armed_at: int = 0
    #: The outbound message reference this wait attaches to. Spellings:
    #: ``message:<id>`` (a conversation message; the default target flushes the
    #: turn's own output id at turn end) or ``peer:<name-or-id>`` (armed via
    #: ``send(patience=…)``; the fire note names that peer). Empty = unnamed.
    armed_after: str = ""
    #: Optional agent-supplied note, rendered into the fire text so the next
    #: turn knows what the wait was about (R38: the agent supplies judgement;
    #: the mechanism supplies defaults and bounds).
    note: str = ""


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
