"""Response models for the machine-wide runtime roster (``/v1/desktop/runtimes``).

**Why a roster, when a session listing already exists.** ``GET
/v1/desktop/sessions`` answers "which conversations are there, and what is each
one doing?" — a question about SESSIONS, ranked by recency and paginated. This
answers a different one: "which runtime PROCESSES are live on this machine, where
is each one listening, and did it answer?" The difference is not academic: on
2026-09-17 this machine had 34 live session runtimes that no session listing could
name, because every one of them had no discovery record (their config roots had
been deleted). A client that can only enumerate sessions cannot find them, and a
client that can only dial a record cannot reach them.

**Every field names its source, because they disagree.** ``session_id`` and
``build_version`` come from a record when one exists and from the boot record
otherwise; ``port`` from a record or from the machine's socket table; ``age_s``
and ``cpu_s`` always from the process table. A field that could not be measured is
``None``/``""`` rather than a default, because a roster is read by a client that
will act on it: a defaulted port is a dial that cannot work, and a defaulted
heartbeat age is a runtime reported as fresh while nothing has beaten.

**``reachability`` is a measurement, not a state.** It is the result of one
bounded loopback connect made while the answer was composed, so ``unknown`` means
"not measured" (no port, or the budget ran out) and never "not reachable". A dead
pid is never dialled at all and reads ``gone``.

``extra="allow"`` matches the wake surface's stance: a field added later is
additive for a client that predates it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class RuntimeEntry(BaseModel):
    """One live (or recorded-but-dead) session runtime."""

    model_config = ConfigDict(extra="allow")

    pid: int
    #: The session this runtime hosts: from its discovery record, else its boot
    #: record, else the spawn environment. ``""`` for a runtime that published
    #: nothing and was spawned outside the harness's own contract.
    session_id: str = ""
    #: The loopback control port it listens on. ``None`` when no source could
    #: supply one — a record carries it, and a record-less runtime's port is read
    #: from the socket table, which a machine without ``lsof`` cannot offer.
    port: int | None = None
    build_version: str = ""
    install_root: str = ""
    #: An interface is looking at it right now: a live viewer lease names its
    #: session, or something holds an ESTABLISHED connection to its control port.
    attached: bool = False
    #: How many live viewer leases name its session.
    observers: int = 0
    #: Seconds since the runtime last beat. ``None`` when no record exists at all:
    #: a record-less runtime does not beat anywhere readable, and reporting 0.0
    #: would read as "fresh".
    heartbeat_age_s: float | None = None
    #: ``live`` (connected), ``unreachable`` (dialled, refused), ``gone`` (pid is
    #: not alive), ``unknown`` (no port, or the probe did not run inside the
    #: budget).
    reachability: str
    #: ``registry.classify``'s verdict (``live``/``wedged``/``stale``) for a
    #: recorded runtime, ``None`` for a record-less one.
    state: str | None = None
    has_record: bool = False
    #: The config root whose record supplied this row's facts, ``""`` when none did.
    #: ``has_record`` is the swept store's answer; this is the roster's, and they
    #: differ for a runtime belonging to a sibling store.
    record_root: str = ""
    has_boot_record: bool = False
    #: The runtime's own report that a turn is running, ``None`` when no record
    #: exists to ask.
    busy: bool | None = None
    #: A short phrase while it is finishing work in flight after a signal.
    leaving: str = ""
    config_root: str = ""
    parent_pid: int = 0
    age_s: float | None = None
    cpu_s: float | None = None
    #: Whether the residency sweep would end this runtime, and the token saying
    #: why not when it would not. Computed by the sweep's own verdict function, so
    #: the roster can never describe a runtime differently from the way the sweep
    #: treats it.
    reclaimable: bool = False
    reclaim_refusal: str = ""
    reclaim_detail: str = ""


class RuntimeRoster(BaseModel):
    """The machine's answer to "which runtimes exist, and can I reach them?"."""

    model_config = ConfigDict(extra="allow")

    runtimes: list[RuntimeEntry] = []
    count: int = 0
    #: The budget the composition was given, and whether the socket table could be
    #: read at all. Both are on the wire so a client can tell a machine with one
    #: runtime from a machine whose evidence was incomplete.
    budget_s: float = 0.0
    socket_table: bool = False
    #: The sources that contributed at least one row, named so an operator reading
    #: a suspicious roster can see which evidence was actually available.
    sources: list[str] = []
