"""Response models for the machine-wide monitor surface (``/v1/desktop/monitors``).

**Why a machine-wide surface.** Monitors are per-session (the transcript's
``monitor_schedules`` entry is the truth) and the desktop already publishes a
session's own rows through ``CanonicalFrontendState.monitors``. What did not
exist was an answer to "which conversations on this machine have standing
watches, and when does each next check?" without opening one session at a
time. That is what the monitors page is, so it reads the derived monitor
index (``monitors/store.py``) through these models — the wake surface's shape
and, deliberately, not its every field.

**Absolute epoch milliseconds on the wire, never a rendered clock.** Every
other timestamp on this wire is seconds and ``next_due_at`` is milliseconds —
named here rather than left to inference, because it has already been guessed
wrong once (the wake module's note). The renderer formats locally; a second
formatter on the same value is how two surfaces come to disagree about one
instant.

**No ``supervisor`` block, deliberately.** The wake listing carries one
because a promise to fire cold needs something installed to keep it. Monitors
never engage a cold session (§10.4) — dormancy is the honest state — so a
supervisor-shaped field here would advertise a watcher that does not exist.

**The clock fields are best-effort, and say so.** The monitor index is only
rewritten on change events, not on every quiet tick (§10.2), so
``next_due_at``/``due_in_s`` are the last values a writer recorded — they can
be behind the live schedule by up to a tick, and the ``state`` word is what a
surface should lead with when the two seem to disagree.

``extra="allow"`` matches ``SessionRow``'s stance: an added field is additive
for an older client, and the counters file has more keys planned than this
listing advertises.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MonitorRow(BaseModel):
    """One standing watch, as the index stores it plus the derived clock fields."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str = ""
    #: The session tool re-run each tick, and the exact arguments it is given.
    tool: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    #: The check cadence in milliseconds; ``until_at`` is the stop time
    #: (``None`` = durable), also milliseconds.
    every_ms: int | None = None
    until_at: int | None = None
    notify: bool = False
    sort_lines: bool = False
    ignore: list[str] = Field(default_factory=list)
    cwd: str = ""
    created_at: int = 0
    #: Written by the session (the one writer of monitor state) on every check
    #: and change event — absent rather than zeroed on a row that predates
    #: them, because a defaulted stamp is a measurement the server never made.
    next_due_at: int | None = None
    last_check_at: int = 0
    checks: int = 0
    deliveries: int = 0
    consecutive_failures: int = 0
    disabled: bool = False
    disabled_reason: str = ""
    #: Seconds until ``next_due_at`` — NEGATIVE when the check is late, which
    #: is ordinary between change events (see the module docstring); ``None``
    #: when no due time is recorded.
    due_in_s: float | None = None
    #: Seconds since ``last_check_at``, or ``None`` when it has not run yet.
    last_check_age_s: float | None = None
    #: ``"dormant"`` (the session is stopped), ``"disabled"`` (the failure
    #: ladder gave up), ``"expired"`` (``until_at`` has passed), else
    #: ``"armed"`` — the CLI's precedence (``cli._monitor_state_word``:
    #: dormancy wins over disabled so a failure word cannot point a reader at
    #: the wrong remedy; disabled wins over the clock so a watch that does not
    #: tick never reads as merely late), with the rendered clock left to the
    #: client.
    state: str = "armed"
    #: When an unavailable episode began (epoch ms; 0 = none). An episode is a
    #: tool that is temporarily out of reach — an MCP server reconnecting — and
    #: it is deliberately NOT a failure: the watch keeps retrying without
    #: counting strikes.
    unavailable_since: int = 0
    #: The §D6 health line every monitor surface shares (``store.health_hint``),
    #: or ``None`` when there is nothing to say. Rendered, not re-derived: the
    #: CLI, the agent tool and the TUI band read the same sentence, so one
    #: monitor cannot read as healthy here and stalled there.
    health: str | None = None


class MonitorEntry(BaseModel):
    """One monitor-carrying conversation, with its watches beneath it.

    One entry per SESSION rather than per monitor: the object a user opens,
    the one a cancel targets, and the one the desktop just armed against.
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
    #: A session the operator STOPPED: its monitors stay armed but do not tick
    #: until it is reopened (``runtime/control._mark_monitors_dormant``).
    dormant: bool = False
    #: The index has an entry and the session has no transcript on disk, so its
    #: monitors can never tick in place. Distinct from ``dormant``: nothing is
    #: parked here, the session is gone.
    ghost: bool = False
    #: Soonest ``next_due_at`` across ``monitors``, or null when none is
    #: readable.
    next_due_at: int | None = None
    monitors: list[MonitorRow] = Field(default_factory=list)


class MonitorListing(BaseModel):
    """``monitors.list``'s payload: every monitor-carrying session on this machine."""

    model_config = ConfigDict(extra="allow")

    entries: list[MonitorEntry] = Field(default_factory=list)
    generated_at: int = 0
    #: Entries before ``limit`` was applied, so a client can say "showing 50 of
    #: 61" rather than implying the store holds only what it was sent.
    total: int = 0
    truncated: bool = False
    #: The index DIRECTORY could not be listed. Distinguishable from an empty
    #: store on purpose: "no standing watches" over a store this process could
    #: not read is a claim it has not earned, and the honest answer is that
    #: the read failed.
    read_error: bool = False


class MonitorWriteReceipt(BaseModel):
    """The outcome of arming or cancelling one monitor.

    Shared by both writes because the facts a client needs are the same:
    which row changed, when it is next due (null after a cancel), how many
    remain, and whether the derived index agrees. ``already_armed`` and
    ``reactivated`` are only meaningful on an arm; on a cancel they are false
    rather than absent, so a client reads one shape.

    There is no ``replayed`` state to report, unlike the wake receipt: monitor
    writes carry no request journal because they do not need one — a retried
    arm is answered by the dedupe identity (``already_armed``) and a retried
    cancel by an honest ``no monitor with id`` refusal about a watch that is
    already gone, so neither can make a second durable thing.
    """

    model_config = ConfigDict(extra="allow")

    session_id: str
    monitor_id: str
    name: str = ""
    next_due_at: int | None = None
    #: How many monitors the conversation holds after this write.
    remaining: int = 0
    #: True only for an arm whose spec was already in force: nothing was
    #: appended and ``monitor_id`` names the existing row.
    already_armed: bool = False
    #: True only for an arm that reset a DISABLED monitor's counters (§11.3's
    #: "re-arm to reactivate").
    reactivated: bool = False
    #: ``"applied"`` for this request's own effect (see the class docstring for
    #: why it is never ``"replayed"``).
    receipt: str = "applied"
    #: Whether the derived index reflects the change. False is a REPORT, not an
    #: error: the transcript won and the next open rebuilds the index.
    index_written: bool = False
