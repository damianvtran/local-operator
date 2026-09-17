"""Response models for the machine-wide wake surface (``/v1/desktop/wakes``).

**Why a machine-wide surface at all.** Wakes are per-session (the transcript's
``wake_schedules`` entry is the truth) and the desktop already publishes a
session's own rows through ``CanonicalFrontendState.wakes``. What did not exist
was an answer to "which conversations on this machine have wakes, and when does
each next fire?" without opening one session at a time. That is what the
Schedules page is, so it reads the derived wake index (``wakes/store.py``)
through these models.

**Absolute epoch milliseconds on the wire, never a rendered clock.** Every
other timestamp on this wire is seconds and ``next_due_at`` is milliseconds —
named here rather than left to inference, because it has already been guessed
wrong once. The renderer formats locally, which is the whole reason the clocks
are not pre-rendered server-side: a second formatter on the same value is how
two surfaces come to disagree about one instant.

**The flags are the supervisor's own verdicts, not re-derivations.**
``dormant``/``ghost``/``stale`` come from the predicates in
``wakes/supervisor.py`` that decide whether a wake can fire at all. The CLI
learned this the hard way: a listing that derived freshness itself said "1
armed, 10m overdue" about a wake the supervisor had already given up on, which
is a painted frame contradicting the process.

``extra="allow"`` matches ``SessionRow``'s stance: an added field is additive
for an older client, and ``paused_at`` is already planned as one.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class WakeScheduleRow(BaseModel):
    """One schedule, as the index stores it plus the derived lateness fields."""

    model_config = ConfigDict(extra="allow")

    id: str
    message: str = ""
    next_due_at: int
    every_ms: int | None = None
    until_at: int | None = None
    limit: int | None = None
    fired_count: int = 0
    #: Seconds past ``next_due_at`` right now, 0.0 when it is still ahead.
    overdue_s: float = 0.0
    #: Past ``wakes.supervisor.STALE_AFTER_S``, the point at which the
    #: supervisor stops engaging and leaves the wake to the session's own
    #: catch-up. A row can be overdue without being stale.
    stale: bool = False
    #: Written by the session (the one writer of schedule state) as it passes
    #: each instant; absent rather than zeroed on an entry that predates them,
    #: because a defaulted stamp is a measurement the server never made.
    last_fired_at: int | None = None
    last_attempt_at: int | None = None


class WakeEntry(BaseModel):
    """One wake-carrying conversation, with its schedules beneath it.

    One entry per SESSION rather than per schedule: the object a user opens,
    can cancel a wake on, and just created is the conversation, and a session
    may legally hold up to ``MAX_WAKE_SCHEDULES`` rows.
    """

    model_config = ConfigDict(extra="allow")

    session_id: str
    #: ``resume.session_name``: the stored title, else the opening message, else
    #: a floor composed of the shortened id and the cwd's basename. Never
    #: empty — a nameless row is a row the user cannot identify.
    name: str
    cwd: str = ""
    #: ``resume.session_origin``: ``""`` for the user's own conversations,
    #: ``"subagent"``/``"fork"``/``"team"`` for what the harness made. For
    #: grouping and labelling only; nothing here is hidden by it.
    origin: str = ""
    #: The index's own write stamp, not the session's activity time.
    updated_at: int = 0
    #: A session the operator STOPPED: its wakes stay armed but do not fire
    #: until it is reopened (``runtime/control._mark_wakes_dormant``).
    dormant: bool = False
    #: The index has an entry and the session has no transcript on disk, so the
    #: supervisor will refuse to engage it. Distinct from ``dormant``: nothing
    #: is parked here, the session is gone.
    ghost: bool = False
    #: Soonest ``next_due_at`` across ``schedules``, or null when none is
    #: readable.
    next_due_at: int | None = None
    schedules: list[WakeScheduleRow] = Field(default_factory=list)


class SupervisorInfo(BaseModel):
    """Whether something will actually fire these wakes.

    The index alone cannot answer that: the supervisor is a separate process
    whose states — unsupported platform, plist written but launchd not
    addressable, loaded-but-exited — are invisible from the files. A listing
    that omitted this would invite a user to trust a schedule nothing is
    watching.
    """

    model_config = ConfigDict(extra="allow")

    supported: bool = False
    running: bool = False
    detail: str = ""
    #: Whether the probe could speak about THIS store at all. On a store
    #: outside the real home (every sandboxed run) launchd addresses the user's
    #: live domain instead, so ``supported``/``running`` say nothing about it —
    #: reporting them there would be a claim about someone else's supervisor,
    #: which is the class of lie this module exists to remove.
    verifiable: bool = True


class WakeListing(BaseModel):
    """``wakes.list``'s payload: every wake-carrying session on this machine."""

    model_config = ConfigDict(extra="allow")

    entries: list[WakeEntry] = Field(default_factory=list)
    generated_at: int = 0
    #: Entries before ``limit`` was applied, so a client can say "showing 50 of
    #: 61" rather than implying the store holds only what it was sent.
    total: int = 0
    truncated: bool = False
    supervisor: SupervisorInfo = Field(default_factory=SupervisorInfo)
    #: The index DIRECTORY could not be listed. Distinguishable from an empty
    #: store on purpose: "no scheduled tasks" over a store this process could
    #: not read is a claim it has not earned, and the honest answer is that
    #: the read failed.
    read_error: bool = False


class WakeWriteReceipt(BaseModel):
    """The outcome of arming, editing or cancelling one wake.

    Shared by all three writes because the facts a client needs are the same
    three: which row changed, when it is next due (null after a cancel), and
    whether the derived index agrees. ``created_session`` and ``receipt`` are
    only meaningful on a create, and are reported as null/false elsewhere
    rather than by growing a second model.
    """

    model_config = ConfigDict(extra="allow")

    session_id: str
    wake_id: str
    next_due_at: int | None = None
    #: True only for a create that made the session as part of the same
    #: request, so a client can open the conversation it just created.
    created_session: bool = False
    supervisor: SupervisorInfo = Field(default_factory=SupervisorInfo)
    #: ``"applied"`` for this request's own effect, ``"replayed"`` when an
    #: at-most-once receipt answered a retry with the first attempt's result.
    receipt: str = "applied"
    #: Whether the derived index reflects the change. False is a REPORT, not an
    #: error: the transcript won and the next open rebuilds the index.
    index_written: bool = False
