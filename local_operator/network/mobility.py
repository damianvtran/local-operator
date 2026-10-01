"""Session mobility across the mesh: move, recall, and archive/delete on a peer.

WHAT THIS MODULE OWNS. One peer op carries the whole protocol — ``net_session_move``,
discriminated by ``phase`` (``status``/``prepare``/``ready``/``done``/``invite``) —
plus the local verbs ``session_move`` and ``session_lifecycle`` that ``lop sessions
move``, the TUI's ``/move --to`` and the desktop transfer route drive.

THE PROTOCOL IN ONE PARAGRAPH (§6.3). The DESTINATION pulls; the OWNER decides.
``D`` asks ``O`` to ``prepare``; ``O`` refuses if the session is busy, otherwise
retires its runtime, waits for the record to go, and journals the handoff. ``D``
then copies into ``<config>/network/staging/`` — outside ``sessions/``, so no
scanner can see a half-copied session — verifies every byte against the manifest
``O`` served, and writes ``ready.json``. ``O`` verifies the digest it served,
advances the journal to ``handing-off`` (**from there a rollback is impossible by
rule**), tombstones the id to ``D``, deletes the directory through
``cleanup.remove_session_dir``, clears the journal and answers ``committed``. Only
then does ``D`` promote, with one ``os.replace``. At every instant at least one
device holds a complete copy, and the promote decision is one monotone state the
owner answers.

INV-1, THE INVARIANT THIS EXISTS TO KEEP: never two runtimes for one session, and
never a second owner for an id that might still be alive on the device it left.
Two guards enforce it structurally rather than by care:

* ``session/runtime/launch.py`` refuses to engage an id this module's journal names
  as ``prepared``/``handing-off``. That is the ONE engage entry point every path
  uses, so a local viewer, a wake, ``lop exec`` and this very relay's
  ``net_session_engage`` are all behind it.
* a replica is never promoted under the original id (``sync.promote_replica``): an
  EC2 peer that spins down can come back, and a same-id promotion is then two
  writers on one transcript (build plan §7, unsafe item 4).

DIRECTIONS. One code path, two ends (§6.7). "Bring it home" (``--to local``) is
this device being the destination. "Shed it to the peer" (``--to peer``) is much
the same, except the requester is the owner, so ``O`` sends ``invite`` and ``D``
runs the same pull a recall does. The alternative — a push-shaped second path —
would double the copy code and the verification with it.

DEADLOCK DISCIPLINE (build plan §0 finding 4). A handler must never issue a
request over the link it is serving, and ``PeerLink.request`` enforces that. So
the source-side handlers here answer from local durable state only, and anything
that needs to ASK the peer runs either on the control thread (the local verbs) or
on a thread this module starts (``invite``). ``reconcile`` takes ``ask=None`` when
it is called from a handler for exactly this reason.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict, cast

if TYPE_CHECKING:
    from local_operator.network.relay import PeerLink, RelayServer

logger = logging.getLogger(__name__)

#: The owner-side deadline for one ``net_session_move`` request (seconds).
#:
#: ``prepare`` is the long phase: it retires the source runtime and waits for its
#: registry record to go (build plan §1.2). A busy session is REFUSED rather than
#: waited on (§1.3), so this bounds an IDLE runtime's exit plus the journal write,
#: not a turn. 90 s is the idle-exit path with room for a loaded host; past it the
#: requester is told the move did not finish and asks ``status`` before retrying.
MOVE_OP_DEADLINE_S = 90.0

#: The owner-side deadline for ``net_session_lifecycle`` (seconds). Archive is one
#: small JSON write; a delete is an in-use probe (which forks ``ps`` and ``lsof``
#: for the keep-alive guards) plus an ``rmtree``. 30 s is those two with room.
LIFECYCLE_OP_DEADLINE_S = 30.0

#: How often ``--wait N`` re-asks a busy owner (§6.4). Always a FRESH idle probe:
#: a session that stays busy for the whole wait produces a refusal carrying the
#: owner's own reason, and the operator decides again.
MOVE_WAIT_POLL_S = 5.0

#: The longest ``--wait`` the CLI accepts. The design caps the no-value form at the
#: session's idle horizon; 30 minutes is the cap it names, and it is also the
#: longest a person will sit at a shell for a move.
MOVE_MAX_WAIT_S = 1800.0

#: How long the requester waits for a ``--keep`` copy to land at the destination
#: before answering "still copying". Generous: this covers a full transcript on a
#: slow link, and nothing on the SOURCE has changed while it runs.
#:
#: IT IS ALSO A RECALL'S COPY BUDGET (``move_hold_s``): a recall copies the same
#: transcript over the same link, and the destination is simply the device that asked
#: for it — so both directions share one number rather than two that can drift.
KEEP_COPY_WAIT_S = 300.0

#: How long an offload waits for the destination to confirm the promote. The
#: commit itself is local and fast (the copy happened before it); this only covers
#: the peer's promote and its ``done`` frame.
OFFLOAD_CONFIRM_WAIT_S = 30.0

#: The round trip a caller adds around a move's own work before it stops waiting: the
#: relay's answer has to travel back over the control socket after the work is done,
#: and a caller that cut it too fine would report its own timeout for a move the relay
#: was about to answer (the whole of QA round 1, Q4a).
MOVE_CONTROL_SLACK_S = 10.0

#: The margin a CLIENT adds to a published bound before its own deadline. It has to
#: exceed the round trip that carries the answer back — the link has just been busy
#: for the whole bound — so a client may not simply EQUAL the bound it is given. 15 s
#: is the desktop's own margin, and the number it was measured with: it gave up at
#: ``wait_s + 15`` against a route answering at ``wait_s + 30``, so the timeout's
#: vaguer sentence always won (QA round 1, Q4a).
MOVE_CLIENT_MARGIN_S = 15.0


#: How long the destination waits to hand a refusal back to the inviter. A
#: NOTIFICATION rather than an act — the owner records it in memory and answers at
#: once — so it is bounded short instead of taking the move op's own copy-sized
#: deadline, which would hold a worker for a minute on an owner that had gone away.
REFUSAL_REPORT_TIMEOUT_S = 10.0


def move_bound_s(wait_s: float, *, keep: bool = False) -> float:
    """How long an OFFLOAD — a move this device INVITED a peer to pull — is held.

    ONE OF THE SHAPES, NOT THE FORMULA FOR ALL OF THEM. ``move_hold_s`` is the one
    place that names every shape the transfer route accepts and ``move_client_bound_s``
    is the published client bound derived from it; this function is the offload's term
    alone, and publishing it as if it were the whole contract is exactly what review
    round 1 caught (the ``keep`` copy's 300 s term and a recall's retire deadline were
    both missing from the advice the client lane was implementing).

    A MOVE IS BOUNDED, AND THE BOUND IS DERIVABLE (QA round 1, Q4a). The inviter holds
    a ``session_move`` request for at most

        ``wait_s + (KEEP_COPY_WAIT_S if keep else OFFLOAD_CONFIRM_WAIT_S)``

    seconds — 30 s at ``wait_s=0``, which is the default on BOTH routes that take it
    (``TransferSession.wait_s`` and the CLI's ``--wait``) — and a front end whose own
    deadline is SHORTER never sees that answer. Measured: the desktop gave up at
    ``wait_s + 15`` s, so the user was told "the move may have happened, check the
    other device" while this device was about to answer "nothing was deleted" — the
    timeout's vaguer sentence always won.

    The bound is what an UNANSWERED request costs, not what a move costs: the route
    returns as soon as it has a definite outcome — the commit, or the destination's
    refusal, which ``_source_refused`` propagates the moment it arrives.
    """
    return (KEEP_COPY_WAIT_S if keep else OFFLOAD_CONFIRM_WAIT_S) + max(0.0, float(wait_s or 0.0))


def move_hold_s(wait_s: float, *, keep: bool = False, to: str = "") -> float:
    """How long THIS DEVICE'S RELAY holds a move before it answers. THE SHAPES, ONCE.

    Every number a client is handed and every deadline this side waits on is taken
    from here, so a shape cannot be corrected in one place and left wrong in another —
    which is what happened on review round 1, when the offload's formula was published
    as if it covered the route.

    * **OFFLOAD** (``to`` names a peer): ``move_bound_s``. This device invited the peer
      to pull, so what it waits for is its OWN durable progress — and the ``keep`` term
      is the DESTINATION's copy, which is why the copy's budget (not this device's
      confirm window) is what bounds it.
    * **RECALL** (``to="local"``): the copy THIS device runs, i.e. ``wait_s`` of busy
      re-polls plus ``KEEP_COPY_WAIT_S``. It is NOT ``move_bound_s``, and nothing in
      that formula bounds it: the direction is reversed (``_recall`` →
      ``_destination_move`` makes this device the DESTINATION), so there is no invite
      and no settle window in it at all. The owner's retire-plus-record deadline
      (``MOVE_OP_DEADLINE_S``) is charged by ``move_client_bound_s``, which counts it
      for every shape.

    A ``--from-replica`` RECOVERY is neither and is not covered here: it promotes bytes
    already on THIS disk with no peer in the loop (``_recover_from_replica``), so it
    keeps the confirm-sized term its caller has always given it rather than borrowing a
    copy's.
    """
    if to == "local":
        # A COPY, NOT A CONFIRMATION: nothing is confirmed over the link, and the work
        # is transcript-sized, so the offload's 30 s window is the wrong term by two
        # orders of magnitude.
        return KEEP_COPY_WAIT_S + max(0.0, float(wait_s or 0.0))
    return move_bound_s(wait_s, keep=keep)


def move_client_bound_s(wait_s: float, *, keep: bool = False, to: str = "") -> float:
    """THE PUBLISHED BOUND: the deadline a CLIENT's own request must not be shorter than.

    Derived, never restated, and published with its terms so a front end can check its
    own arithmetic against this side's:

    ======================  =====================================================
    shape                   the bound (seconds)
    ======================  =====================================================
    offload (``to`` a peer) ``wait_s + 30 + MOVE_OP_DEADLINE_S + 10 + MOVE_CLIENT_MARGIN_S``
    ``keep`` copy           ``wait_s + 300 + MOVE_OP_DEADLINE_S + 10 + MOVE_CLIENT_MARGIN_S``
    recall (``to="local"``) ``wait_s + 300 + MOVE_OP_DEADLINE_S + 10 + MOVE_CLIENT_MARGIN_S``
    ======================  =====================================================

    Concretely, at the ``wait_s=0`` both routes default to: **145 s** for an offload,
    **415 s** for a ``keep`` copy (``keep`` costs its copy either way), and **415 s**
    for a recall. A front end that gives up sooner than these reports its own timeout
    for a move this side was about to answer, which is QA round 1's Q4a symptom — the
    desktop gave up at ``wait_s + 15`` against a route answering at ``wait_s + 30``, so
    the user read "the move may have happened, check the other device" instead of this
    device's own answer.

    THE TERMS, because which one a client is waiting on is the whole question:

    * ``move_hold_s`` — the relay's own held time for that shape (30 s for an offload's
      confirmation, 300 s for a copy, either way).
    * ``MOVE_OP_DEADLINE_S`` — the PEER's slow-op budget. It bounds the frame the
      inviter sent (an offload's prepare/commit) or the owner's retire-plus-record
      deadline (a recall's prepare), so it is added to EVERY shape rather than folded
      into one of them.
    * ``MOVE_CONTROL_SLACK_S`` — the control socket's answer travelling back.
    * ``MOVE_CLIENT_MARGIN_S`` — the client's own margin over the route's answer, which
      has to exceed the round trip on a link that has just been busy for the bound.

    A RECALL IS BOUNDED BY A BUDGET, NOT BY A PROMISE, and the difference is what a
    client must act on: the copy above ``move_hold_s`` is transcript-sized and nothing
    here caps it, so the route may answer 503 "unconfirmed" — the request WAS sent —
    with the copy still running past every number above. A client whose own deadline
    fires on a recall therefore knows NOTHING about the outcome: it must report it as
    unknown, never as a refusal, and never retry into a second move.
    """
    return (
        MOVE_OP_DEADLINE_S
        + move_hold_s(wait_s, keep=keep, to=to)
        + MOVE_CONTROL_SLACK_S
        + MOVE_CLIENT_MARGIN_S
    )


#: Age after which an abandoned staging directory is swept. `ready.json` marks one
#: awaiting a commit and exempts it (design §7's GC): the bytes of a verified copy
#: are the only thing standing between a crash and a lost conversation.
STAGING_MAX_AGE_S = 7 * 24 * 60 * 60.0

#: The audit vocabulary this slice emits (§6.3 step 19). One record per SEMANTIC
#: event, never per chunk. They are deliberately spelled here rather than declared
#: in ``network/audit.py``: that module's ``DETAIL_KEYS``/``EVENT_KINDS`` belong to
#: the credentials slice (build plan §5), and the writer accepts an event it does
#: not know — it writes the record and drops the un-whitelisted detail. So these
#: land today with their actor and outcome and no detail; the detail joins them when
#: audit.py's tables gain the entries.
AUDIT_PREPARE = "session.handoff.prepare"
AUDIT_READY = "session.handoff.ready"
AUDIT_HANDING_OFF = "session.handoff.handing_off"
AUDIT_COMMITTED = "session.handoff.committed"
AUDIT_ROLLED_BACK = "session.handoff.rolled_back"
AUDIT_DONE = "session.handoff.done"

# ---------------------------------------------------------------------------
# The ``session_move`` contract (frozen at P0; slices V and DB build against it)
# ---------------------------------------------------------------------------

#: The phases a move passes through, in order. MONOTONE: a result never reports
#: an earlier phase than one already reported for the same move.
#:
#: * ``prepared``    — the source retired its runtime and journalled the move;
#:                     nothing is copied yet and a rollback is still free.
#: * ``handing_off`` — the destination verified the copy and asked to commit; the
#:                     source is writing its tombstone. Not yet safe to open.
#: * ``committed``   — the DESTINATION OWNS THE SESSION. This is the phase a front
#:                     end acts on: open it on ``to_device``.
#: * ``done``        — the source confirmed its cleanup. Best-effort: a move that
#:                     stops at ``committed`` is complete from the user's side.
SessionMovePhase = Literal["prepared", "handing_off", "committed", "done"]

MOVE_RESULT_PHASES: tuple[SessionMovePhase, ...] = (
    "prepared",
    "handing_off",
    "committed",
    "done",
)

#: Phases after which the session is usable at its destination.
MOVE_OPENABLE_PHASES: frozenset[str] = frozenset({"committed", "done"})

#: ``mode``: ``move`` keeps the session id and retires the source; ``keep``
#: (``--keep``) mints a NEW id at the destination and leaves the source untouched.
SessionMoveMode = Literal["move", "keep"]

#: The refusal codes a front end may branch on. The sentence (``message``) is for
#: the person; a code outside this set is still a refusal, rendered by sentence.
MOVE_REFUSAL_CODES: frozenset[str] = frozenset(
    {
        "busy",  # the source has a turn in flight; message is its idle reason verbatim
        "viewed_elsewhere",  # another front end is attached to the source
        "unreachable",  # a device stopped answering; nothing changed
        "digest_mismatch",  # the copy did not verify; rolled back
        "revoked",  # a device lost its membership mid-move
        "not_authorised",  # a device lacks the ``move`` capability
        "not_owner",  # the source does not own the id (moved already?)
        "in_progress",  # another move of this id is under way
        "deadline_exceeded",  # the owner did not finish in time; ask status
        "not_implemented",  # this build does not move sessions yet
        # The three this slice adds. They are not in P0's list and are still
        # refusals a front end renders by sentence: a code outside the set is
        # explicitly allowed to exist (see the contract's own note).
        "already_local",  # --to local for a session that is already here
        "third_device",  # a move asked for by a device that is neither end
        "no_replica",  # --from-replica with nothing synced
        "unconfirmed",  # committed here, never confirmed at the destination (QA R1 Q1)
        # This slice's fourth, and it is a data-integrity refusal rather than a
        # protocol one: the source holds an entry the copy set does not carry, so
        # moving it would delete that entry with nothing to copy it from (B-M2).
        "unlisted_content",
        # PR #1756 round 1 (design finding D1), and it exists for the SENTENCE
        # discipline rather than the wire: split off ``unreachable`` so a consumer
        # can tell "a peer stopped answering" (its remedy is time) from "no row
        # at all" (its remedy is looking where the id was last staged, HERE).
        "no_holder",
    }
)


class MoveDevice(TypedDict):
    """One end of a move. ``name`` is the mesh name, ``""`` when unknown."""

    device_id: str
    name: str


class MovePhaseStamp(TypedDict):
    phase: SessionMovePhase
    at: float  # unix seconds, the reporting device's clock


class SessionMoveResult(TypedDict):
    """``lop sessions move --json`` on success (exit status 0).

    ``session_id`` is the id the user asked to move; ``new_session_id`` is the id
    to OPEN — equal to ``session_id`` for ``mode == "move"``, freshly minted for
    ``keep``. ``phase`` is the last phase reached and is always in
    :data:`MOVE_OPENABLE_PHASES` on this shape; ``phases`` is the history.

    ``engagement`` is the ONE optional member, and it is ``NotRequired`` rather than
    ``... | None`` on purpose: it is absent — not null — for every caller that did
    not ask for it, so the document every existing reader parses is unchanged
    (``test_the_session_move_contract_is_frozen`` pins the set, and this is the one
    addition made to it since the contract was frozen).
    """

    ok: Literal[True]
    session_id: str
    new_session_id: str
    mode: SessionMoveMode
    from_device: MoveDevice
    to_device: MoveDevice
    phase: SessionMovePhase
    phases: list[MovePhaseStamp]
    #: Present ONLY when the move was asked to engage on arrival
    #: (``--engage-on-arrival``); absent otherwise.
    engagement: NotRequired[MoveEngagement]


class MoveEngagement(TypedDict):
    """What the destination did about arriving live (``--engage-on-arrival``).

    A SEPARATE BLOCK ON A SUCCESSFUL RECEIPT, because the failure it can report is a
    different kind of failure from every refusal in :data:`MOVE_REFUSAL_CODES`:
    ``engaged: False`` means THE MOVE SUCCEEDED and the conversation was not started
    on this device. Collapsing the two would make a committed transfer look like a
    refused one, and a caller that retried on it would ask for something that has
    already happened.

    The fields are the engage op's own (``net_session_engage``), because this is
    that act: ``detail`` is the sentence the engage path gave when ``engaged`` is
    ``False``, and the state ("runtime joining") when it is ``True``.
    """

    engaged: bool
    detail: str
    #: The id engaged, which for a ``keep`` copy is the freshly minted one — the same
    #: id the receipt tells the caller to open.
    session_id: str


class SessionMoveRefusal(TypedDict):
    """``lop sessions move --json`` on refusal (exit status 1).

    ``phase_reached`` is ``None`` when nothing started. ``changed`` is the one
    field a front end must honour before retrying: ``False`` means nothing moved
    and a retry is safe; ``True`` means the move reached a phase it could not roll
    back from here, and the front end asks for status instead of retrying.
    """

    ok: Literal[False]
    code: str
    message: str
    session_id: str
    phase_reached: SessionMovePhase | None
    changed: bool


LifecycleAction = Literal["archive", "unarchive", "delete"]


# ---------------------------------------------------------------------------
# Progress: the in-memory history a refusal or a local verb reports from
# ---------------------------------------------------------------------------


class _Progress:
    """Per-relay move history, so a local verb can report the phases it saw.

    NOT durable state, deliberately: the truth about a move is the journal and the
    tombstone, both on disk, and a restarted relay rebuilds what it can from them
    (see :func:`_phases_from_disk`). This exists only because a phase list is a
    HISTORY — after the commit the journal is gone, and `committed` cannot be
    re-derived from a file that no longer says it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._phases: dict[str, list[MovePhaseStamp]] = {}
        self._events: dict[str, threading.Event] = {}
        self._refusals: dict[str, tuple[str, str, str]] = {}

    def note(self, session_id: str, phase: SessionMovePhase) -> None:
        with self._lock:
            history = self._phases.setdefault(session_id, [])
            if phase not in [item["phase"] for item in history]:
                history.append({"phase": phase, "at": time.time()})

    def phases(self, session_id: str) -> list[MovePhaseStamp]:
        with self._lock:
            return list(self._phases.get(session_id, []))

    @staticmethod
    def _event_key(session_id: str, marker: str) -> str:
        """One session's slot for ``marker``. The ONE spelling of the key format:
        ``forget`` clears by this same prefix, and two spellings would drift."""
        return f"{session_id}:{marker}"

    def wait_for(self, session_id: str, marker: str, timeout: float) -> bool:
        with self._lock:
            event = self._events.setdefault(self._event_key(session_id, marker), threading.Event())
        return event.wait(timeout)

    def signal(self, session_id: str, marker: str) -> None:
        with self._lock:
            event = self._events.setdefault(self._event_key(session_id, marker), threading.Event())
        event.set()

    def note_refusal(self, session_id: str, *, code: str, message: str, from_device: str) -> None:
        """Record the DESTINATION's refusal of a move THIS device invited.

        The offload's inviter watches its own durable progress and never asks the
        destination (§_offload), so a destination that refuses writes nothing here
        and the wait would run out its whole budget — measured on 2026-09-24 as 60 s
        against a refusal the destination produced in 3 ms (QA round 1, Q4a). The
        refusing device says so over the same link it was invited on, which is the
        only channel that carries the fact.

        ``from_device`` is kept so a refusal can only end the wait it belongs to: the
        inviter names the device it invited, and a member that was not the destination
        of this move cannot shorten it.
        """
        with self._lock:
            self._refusals[session_id] = (code, message, from_device)

    def refusal(self, session_id: str, *, from_device: str = "") -> tuple[str, str] | None:
        """The recorded refusal for ``session_id``, or ``None``.

        ``from_device`` filters by who refused: an empty string takes any device's
        answer (a caller with no invite to match), and a named device ignores a
        refusal recorded by anybody else.
        """
        with self._lock:
            found = self._refusals.get(session_id)
        if found is None or (from_device and found[2] != from_device):
            return None
        return found[0], found[1]

    def forget(self, session_id: str) -> None:
        """Drop EVERYTHING this relay holds in memory for ``session_id``.

        THE EVENTS MUST GO WITH THE PHASES AND REFUSALS, because their staleness is a
        LIE rather than a missing sentence: ``signal`` sets a level-triggered
        ``threading.Event`` that nothing else clears, and ``_await_own_progress``'s
        very first poll treats it as THIS move's completion. Before this clear
        existed, a session that had once moved OUT through a relay answered every
        LATER offload with an instant synthetic ``committed`` while nothing on the
        wire ran — and the busy refusal the protocol did produce was never consulted
        (measured live 2026-09-29, relay pid 20929: ~4 µs against the first move's
        8.7 s, no tombstone, no audit, nothing moved). A completion signal is about
        ONE move; clearing it here is what scopes it to one.
        """
        with self._lock:
            self._phases.pop(session_id, None)
            self._refusals.pop(session_id, None)
            prefix = self._event_key(session_id, "")
            for key in [item for item in self._events if item.startswith(prefix)]:
                del self._events[key]


_progress_lock = threading.Lock()
_progress: dict[int, _Progress] = {}


def progress_for(server: "RelayServer") -> _Progress:
    with _progress_lock:
        found = _progress.get(id(server))
        if found is None:
            found = _Progress()
            _progress[id(server)] = found
        return found


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _device_block(server: "RelayServer", device_id: str, name: str = "") -> MoveDevice:
    return {
        "device_id": device_id,
        "name": name or server._member_name(device_id) or device_id,  # noqa: SLF001
    }


def _own_block(server: "RelayServer") -> MoveDevice:
    return {"device_id": server.identity.device_id, "name": server._own_label()}  # noqa: SLF001


def new_lease_epoch() -> str:
    """A fresh handoff epoch. Its only job is to be unique per handoff attempt."""
    return "e_" + os.urandom(4).hex()


def _move_refusal(
    session_id: str,
    code: str,
    message: str,
    *,
    phase_reached: SessionMovePhase | None = None,
    changed: bool = False,
) -> SessionMoveRefusal:
    return {
        "ok": False,
        "code": code,
        "message": message,
        "session_id": session_id,
        "phase_reached": phase_reached,
        "changed": changed,
    }


#: ``created_at.json`` is the ONLY entry a bare remnant may hold: it is the birth
#: sidecar a ``Transcript`` construction publishes, so it is the one file a READER
#: that never writes leaves behind (see :func:`_bare_remnant`).
_REMNANT_ALLOWED_NAMES = frozenset({"created_at.json"})


def _bare_remnant(directory: Path) -> bool:
    """A directory holding NO conversation: a corpse, not a session.

    THE SHAPE, MEASURED 2026-09-29 (the recall strand): constructing a
    ``Transcript`` MATERIALISES the directory it is pointed at and publishes
    ``created_at.json`` (``session/transcript.py``'s constructor), so a READER
    that never writes leaves exactly this remnant. The live writer found that
    night was a viewer on the device that does NOT own the id, running its
    owner-lost verdict (``session/attached.py``'s cut-off journalling); its
    corpse sat at the recall's promote target and made the promote refuse.

    STRICT ON PURPOSE, and this predicate is the only license the promote has to
    clear one: an EMPTY directory qualifies, and otherwise every entry must be
    the birth sidecar or an atomic write's killed temp (``*.tmp``, ``tmp*``). A
    transcript, a stamp, a lease/pid claim — or ANY other entry — means somebody
    could own or be writing this directory, and callers must treat it exactly as
    they always have.
    """
    try:
        children = list(directory.iterdir())
    except OSError:
        # A file, a vanished directory, an unreadable one: never a remnant.
        return False
    for child in children:
        if child.is_dir():
            return False
        name = child.name
        if name in _REMNANT_ALLOWED_NAMES or name.endswith(".tmp") or name.startswith("tmp"):
            continue
        return False
    return True


def _owned_here(server: "RelayServer", session_id: str) -> bool:
    """Is ``session_id`` this device's own? The stamp, not the directory.

    A directory whose ``mesh.json`` names another device is the leftover a crash
    leaves in a handoff window (§6.5), and answering "mine" for it here is how a
    moved-away session gets a runtime on the device it left.

    AND A BARE REMNANT IS NOBODY'S (measured 2026-09-29): with no stamp at all
    the answer used to be "mine" for any directory, so the empty corpse a
    viewer's cut-off verdict left (see :func:`_bare_remnant`) made every recall
    of that id answer ``already_local`` — a dead end, because the only complete
    copy was the one staged for the very move being refused. A transcript or
    anything else a session can hold still answers "mine"; a corpse does not.
    """
    from local_operator.session.placement import read_stamp

    directory = Path(server.root) / "sessions" / session_id
    if not directory.is_dir():
        return False
    stamp = read_stamp(server.root, session_id)
    if stamp is not None:
        home = stamp.home_device
        return not home or home == server.identity.device_id
    return not _bare_remnant(directory)


def _tombstone(root: Path, session_id: str) -> dict[str, Any]:
    from local_operator.network.projection import read_tombstones

    return read_tombstones(root).get(session_id) or {}


def _asking_device(link: Any, frame: dict[str, Any]) -> str:
    """Which device a move frame is FROM: the link's identity first, then the frame's.

    The link's device id is the identity the mesh ADMITTED (``PeerLink.device_id``, set
    where the handshake resolved the peer), so it is preferred over ``to_device``, a
    field the asker writes itself: a frame that claims to be the device a record names
    must not be believed by the reader of that record. The frame is the fallback for a
    caller with no admitted link — the handler driven directly, which is how a unit
    test reaches it.
    """
    return str(getattr(link, "device_id", "") or "") or str(frame.get("to_device") or "")


def _tombstone_names_this_device(server: "RelayServer", tombstone: dict[str, Any]) -> bool:
    """Whether the owner's record says the id went to THIS device and no other.

    THE PROMOTION PROOF IS AN IDENTITY, NOT AN EXISTENCE (review round 2, MAJOR). A
    tombstone is written by the device that handed the id away and names the ONE device
    that may adopt it (``projection.write_tombstone``); its presence on its own proves
    only that the id left its owner. A receiver whose handoff was rolled back KEEPS its
    verified bytes — deliberately, ``_reconcile_destination``'s rollback branch says so,
    and ``ready.json`` exempts them from the staging sweep for as long as they exist —
    so the owner can hand the same id to a different device later, and the older copy is
    then the copy of a handoff that no longer describes it. Promoting it would mint a
    SECOND OWNER for one id, the invariant this slice exists to enforce, with the OLDER
    copy winning.

    A record whose ``device_id`` is empty names nobody and is therefore not this device:
    a record that carries no destination cannot prove that any device may adopt.
    """
    return str(tombstone.get("device_id") or "") == server.identity.device_id


def _adopt_elsewhere_refusal(
    server: "RelayServer", session_id: str, tombstone: dict[str, Any]
) -> SessionMoveRefusal:
    """The refusal for a staged copy whose owner's record names ANOTHER device.

    It names the CONDITION rather than only the outcome, because the user's remedy is
    not "retry": the record would have to name this device for the bytes here to be
    adoptable, and the verb that adopts them has to run on the device it does name —
    the same rule Q1c applied to the refusals that used to say "run this from the
    device that holds it" and name a device that holds nothing.

    ``changed`` stays False and the phase stays None on purpose: nothing on this device
    moved, and the staged copy is left exactly where it is (it is the only copy here).
    """
    device_id = str(tombstone.get("device_id") or "")
    if not device_id:
        # A RECORD WITH NO TAKER, which today's writers cannot produce
        # (``projection.write_tombstone`` always writes the destination, and
        # ``_complete_handoff`` takes it from the journal) and a hand-edited or
        # truncated file can: it cannot be adopted anywhere, so naming a device or a
        # verb here would invent both. The sentence says only what is true.
        return _move_refusal(
            session_id,
            "third_device",
            f"{session_id} was handed away and its record names no device, so it does not "
            "name this one: the copy staged here is not this device's to adopt and "
            "nothing was changed",
        )
    name = str(tombstone.get("device_name") or "") or server._member_name(device_id)  # noqa: SLF001
    who = name or device_id
    return _move_refusal(
        session_id,
        "third_device",
        f"{session_id} was handed to {who}, not to this device: its record names {who}, "
        "so the copy staged here is not this device's to adopt and nothing was changed; "
        f"`lop sessions move {session_id} --to local` on {who} adopts it there",
    )


def _audit(server: "RelayServer", event: str, session_id: str, peer: str, **detail: Any) -> None:
    """One handoff record. Never raises: an audit loss cannot undo a move."""
    try:
        from local_operator.network.audit import AuditEvent

        server.audit.record(
            AuditEvent(
                event=event,
                actor=server.identity.device_id,
                subject=session_id,
                outcome="ok",
                network_id=detail.pop("network_id", ""),
                detail={"peer": peer, **detail},
            )
        )
    except Exception:  # noqa: BLE001 — a record is not a gate
        logger.debug("mobility: could not record %s for %s", event, session_id, exc_info=True)


# ---------------------------------------------------------------------------
# Retiring the source runtime (§6.3 step 3)
# ---------------------------------------------------------------------------


class _RetireOutcome(TypedDict):
    #: ``cold`` (nothing was running), ``retired``, ``busy`` (the runtime kept
    #: itself) or ``viewed`` (another attach client is present).
    result: str
    #: The runtime's own words, verbatim, when it kept itself.
    sentence: str


def _retire_local_runtime(
    root: Path, session_id: str, *, deadline_s: float = MOVE_OP_DEADLINE_S
) -> _RetireOutcome:
    """Quiesce and retire this device's runtime for ``session_id``, if any.

    THREE OUTCOMES, and the middle one is the design's whole position on busy
    sessions (§6.4): the runtime re-asks its own idle predicate and either retires
    or answers ``kept: <reason>``, and a refusal CANNOT lose a turn that was in
    flight. So this never drains, never signals, and never waits on a turn.

    ``exclusive=True`` is what makes "a viewer is attached" a refusal rather than
    a race: the owner honours the fence only when no other attach client is
    registered, so a TUI or a phone holding the session blocks the move with the
    runtime's own sentence. Our own connection is the requester, so it is measured
    as nobody.
    """
    import asyncio

    from local_operator.mobile.attach_client import AttachClient, find_runtime_record

    record, pid = find_runtime_record(root, session_id)
    if record is None:
        # "NO RECORD" IS NOT "NOBODY IS WRITING". ``find_runtime_record`` returns
        # ``(None, pid)`` when a live process holds the transcript lease but has
        # published no discovery record: ``lop exec``, the headless REPL, the
        # server, or a runtime still booting. Reading that as ``cold`` let the
        # commit delete the directory out from under a live writer. The lease
        # claim is the authority here, read WITHOUT acquiring (see
        # ``session_lease.lease_holder``), and anything but a proven-absent
        # holder refuses — a runtime mid-boot is exactly when a delete does the
        # most damage, so "probably harmless" is not an answer this may give.
        refusal = _lease_refusal(root, session_id, pid)
        if refusal:
            return {"result": "viewed", "sentence": refusal}
        return {"result": "cold", "sentence": ""}
    if not getattr(record, "capabilities", None) or (
        _EXCLUSIVE_MOVE_CAPABILITY not in tuple(record.capabilities)
    ):
        # FAIL CLOSED: an owner too old to advertise the fence would ignore the
        # flag and retire under a live viewer, which is the one thing the fence
        # exists to prevent. A move is worth a /reload.
        return {
            "result": "viewed",
            "sentence": "this session's runtime is too old to be moved safely; reload it first",
        }

    async def _ask() -> str:
        client = AttachClient(
            lambda _projection: None,
            lambda _reason: None,
            locality="remote",
            on_operator_prompt=lambda copy: logger.debug("mesh move: %s", copy),
        )
        try:
            await client.connect(record, session_id)
            return await client.retire_now(exclusive=True)
        finally:
            client.close()

    try:
        answer = str(asyncio.run(_ask()))
    except Exception as exc:  # noqa: BLE001 — every failure here is a refusal, not a crash
        # An unreachable runtime is NOT evidence that it is idle, and the whole
        # point of the protocol is that O never deletes a directory something may
        # still be writing to. Refuse, with the transport's own words.
        return {
            "result": "busy",
            "sentence": (
                f"this session's runtime could not be asked to stop ({exc}), so it was not " "moved"
            ),
        }
    if answer == "retiring":
        if not _await_record_gone(root, session_id, deadline_s=deadline_s):
            return {
                "result": "busy",
                "sentence": (
                    "this session's runtime is still shutting down, so nothing was moved; "
                    "try again in a moment"
                ),
            }
        return {"result": "retired", "sentence": ""}
    sentence = answer[len("kept:") :].strip() if answer.startswith("kept:") else answer
    return {"result": "busy" if sentence else "retired", "sentence": _busy_sentence(sentence)}


def _wait_seconds(value: Any) -> float:
    """A peer's ``wait_s`` as a number this side can hold, or ``0.0``.

    EVERY OTHER FIELD IN THE INVITE is read with ``or <default>`` on a type the
    reader can absorb; this one arrived as ``float(frame.get("wait_s") or 0.0)``, so
    a malformed value (``"soon"``, a list) raised out of the handler and turned a
    move into an exception instead of a refusal (review round 1, NIT). ``0.0`` is
    the honest reading of a wait that cannot be understood — it is the frame's own
    default — and the clamp is the same cap the CLI applies, so a peer cannot ask
    this side to re-probe for a week.
    """
    if isinstance(value, bool):
        # ``True`` is not a duration, and ``float(True) == 1.0`` would read as one.
        return 0.0
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return 0.0
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        # NaN fails its own equality and infinities pass the clamp below only by
        # accident of comparison; neither is a duration anybody asked for.
        return 0.0
    return min(max(0.0, seconds), MOVE_MAX_WAIT_S)


def _handoff_age(entry: dict[str, Any]) -> str:
    """The ``started Ns ago`` clause for a journal entry, or an honest unknown.

    ``float(existing.get("at") or 0.0)`` rendered ~1.79e9 seconds for an entry with
    no ``at`` — a number about the epoch rather than about the handoff. Every writer
    on this head stamps the field, so this is the defensive branch (review round 1,
    NIT), but a sentence whose whole job is legibility may not invent a number: the
    clause says what it does not know.
    """
    raw = entry.get("at")
    # ``isinstance`` RATHER THAN A BARE ``float()`` IN A ``try``: pyright reads
    # ``entry.get(...)`` as ``Any | None`` and rejects the call, and the narrowed
    # form is also the honest one — a bool is an ``int``, and ``True`` is not a
    # timestamp (type-check, 2026-09-27: this was the branch's only pyright error).
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return "started at an unknown time"
    at = float(raw)
    if at <= 0:
        return "started at an unknown time"
    return f"started {int(max(0.0, time.time() - at))}s ago"


def _busy_sentence(answer: str) -> str:
    """The owner's own words for a refused retire, or a sentence when it gave a TOKEN.

    THIS IS WHAT REACHES A PERSON. ``_source_prepare`` returns this string as the
    refusal's ``message``, and the design's contract for the mesh surface is that
    every refusal names its cause (``MOVE_REFUSAL_CODES``: "the source has a turn in
    flight; message is its idle reason verbatim"). Verbatim is right for a reason
    that is a sentence, and it is what the runtime's ``kept: <why>`` answers
    normally carry.

    Measured on the two-device rig 2026-09-26: the runtime's ``retire_now`` answered
    the bare token ``busy`` for a session parked in a turn (the reason word
    ``may_refresh`` returns for its own notice), and ``lop sessions move`` therefore
    printed ``{"code": "busy", "message": "busy"}`` — a machine word in the one
    field a person reads, with no cause and no remedy, beside every sibling refusal
    that is a full sentence. A token is now rendered into the sentence it stands
    for; anything that already reads as prose passes through untouched, so this can
    never overwrite a runtime's own explanation.
    """
    reason = answer.strip()
    if not reason:
        return ""
    if " " in reason or reason[-1] in ".!?:;":
        return reason
    if reason in {"busy", "working"}:
        # THE REMEDY NAMES ITS SURFACE (the /move papercut, 2026-09-29). ``--wait``
        # belongs to the CLI verb; the front ends that echo this sentence through a
        # composer (the TUI's ``/move --to``, and every surface that renders the
        # refusal verbatim) refuse the flag, so offering it bare invited a SECOND
        # refusal — ``/move --to takes only --keep, not '--wait'`` — about a flag the
        # reader could not use where they read it. The shell clause is what makes the
        # offer runnable; ``tui/session_move.py`` refuses the flag itself by the same
        # route, so the two sentences cannot disagree about where the wait lives.
        #
        # REVISIT IF THE TUI EVER GAINS ``--wait``, and move all three together: that
        # parse would then accept the flag, this sentence's shell clause would be
        # wrong for that surface, and ``tests/unit/network/test_mobility_busy_wait.py``
        # pins these exact words (review round 1, NIT-1).
        return (
            "this session is working right now, so nothing was moved; try again when the "
            "turn finishes; from a shell, pass --wait <seconds> to re-check"
        )
    return f"this session is not idle ({reason}), so nothing was moved"


_EXCLUSIVE_MOVE_CAPABILITY = "exclusive-move-v1"


def _lease_refusal(root: Path, session_id: str, record_pid: int | None) -> str:
    """The sentence refusing a move of a session whose lease is held, or ``""``.

    Two readings, and EITHER one refuses: the pid ``find_runtime_record`` already
    reported, and the lease claim itself. The second is what closes the case the
    first cannot see — a claim whose writer has not (yet) written the compatibility
    mirror that discovery reads.
    """
    from local_operator.session_lease import lease_holder

    holder, verdict = lease_holder(Path(root) / "sessions" / session_id)
    if verdict == "none" and record_pid is None:
        return ""
    who = holder or record_pid
    if verdict == "uncertain" and who is None:
        return (
            "this session's transcript lease could not be read, so it was not moved; "
            "nothing was changed"
        )
    return (
        f"this session is open in another Local Operator process (pid {who}) that "
        "published no runtime record, so it was not moved; close that process "
        "(or let it finish) and try again"
    )


def _await_record_gone(root: Path, session_id: str, *, deadline_s: float) -> bool:
    """Wait for the runtime's discovery record to disappear (§6.3 step 4).

    A local viewer on O that dials a record which outlived its process gets a
    cryptic attach failure mid-handoff; waiting here is what makes the window
    silent. Bounded, and a timeout is reported as "still shutting down" rather
    than as success — the caller must not delete a directory whose writer has not
    provably left.
    """
    from local_operator.mobile.attach_client import find_runtime_record

    deadline = time.monotonic() + max(0.0, deadline_s)
    while time.monotonic() < deadline:
        record, _pid = find_runtime_record(root, session_id)
        if record is None:
            return True
        time.sleep(0.05)
    record, _pid = find_runtime_record(root, session_id)
    return record is None


# ---------------------------------------------------------------------------
# The transport: one place that turns a peer link into "ask and get a detail"
# ---------------------------------------------------------------------------


class Moved(Exception):
    """A peer answered a refusal. Carries the family's code and sentence."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class LinkTransport:
    """``ask(frame) -> detail`` over one peer link.

    A refusal raised by the peer arrives as an ``error`` frame carrying the SENTENCE
    only — ``wire.refusal_frame`` deliberately drops the code, so a peer cannot
    learn which of membership, epoch or capability failed. The expected refusals of
    this protocol therefore travel INSIDE the ack as ``{"result": "refused", "code": …}``,
    which is also what the design's own sequence shows (§6.3 step 3). Both shapes are
    normalised here, so no caller has to know which one it got.
    """

    def __init__(self, server: "RelayServer", link: "PeerLink", session_id: str) -> None:
        self.server = server
        self.link = link
        self.session_id = session_id

    def ask(self, frame: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        request = {"req": self.server._next_relay_req(), **frame}  # noqa: SLF001
        if timeout is None:
            timeout = self.server.slow_request_timeout(str(frame.get("op") or ""))
        reply = self.link.request(request, timeout=timeout)
        if reply is None:
            raise Moved("unreachable", _unreachable(None))
        if reply.get("op") == "error":
            raise Moved("refused", str(reply.get("message") or "the peer refused that"))
        if reply.get("op") != "ack":
            raise Moved(
                "unreachable",
                "the peer answered something this build does not understand, so nothing was "
                "changed",
            )
        detail = reply.get("detail")
        if not isinstance(detail, dict):
            raise Moved(
                "unreachable",
                "the peer answered with nothing this build can read, so nothing was changed",
            )
        if str(detail.get("result") or "") == "refused":
            raise Moved(
                str(detail.get("code") or "refused"),
                str(detail.get("message") or detail.get("detail") or "the move was refused"),
            )
        return detail


def _unreachable(peer: str | None) -> str:
    name = peer or "that device"
    return f"{name} is unreachable (no answer on its link); nothing was changed"


# ---------------------------------------------------------------------------
# The destination's half: pull, verify, ready, promote
# ---------------------------------------------------------------------------


def _staging_content_digest(server: "RelayServer", session_id: str, staging: Path) -> str:
    """The digest of the bytes a copy holds, as the adopting device sees them.

    ``sync.copy_content_digest`` over the staging directory and this install's
    shared attachment store: the SAME function the owner runs over its own session
    directory, so the two values are comparable and the owner can commit on their
    equality (review round 1, M-2).
    """
    from local_operator.network import sync as sync_mod

    # THE BLOB DIRECTORY, not the config root: ``_member_stamps`` reads
    # ``<dir>/<ref>.bin``, so passing the root silently dropped every attachment from the
    # comparison — a copy whose blobs were all zeroed would have matched. Both ends pass
    # the same directory now, which is what makes the comparison mean what it says.
    return sync_mod.copy_content_digest(staging, session_id, Path(server.root) / "attachments")


def _verify_staging(
    server: "RelayServer", session_id: str, staging: Path, ready: dict[str, Any]
) -> str:
    """Re-hash a staged copy against what ``ready.json`` recorded, before adopting.

    THE LAST CHECK BEFORE THE ONLY ``os.replace`` INTO ``sessions/``. The
    destination re-derives the digest of the bytes it is about to promote and
    compares it with the value it wrote when it verified the copy; a mismatch — a
    truncation between the two moments, a disk that reported the write and kept
    less, a staging directory somebody edited — refuses the promote and leaves
    ``sessions/`` untouched. Returning the sentence rather than raising keeps the
    callers' two refusal shapes (result/refusal documents) intact.

    A ``ready.json`` with no ``content_digest`` is a copy from a build that did not
    record one, and it is refused rather than trusted: the whole point of the field
    is that the adopting device has a value to check against.
    """
    expected = str(ready.get("content_digest") or "")
    if not expected:
        return (
            "the copy of that conversation was staged before this build recorded what it "
            "verified, so it was not adopted; asking again evaluates it from scratch"
        )
    if _staging_content_digest(server, session_id, staging) != expected:
        return (
            "the staged copy of that conversation no longer matches the bytes that were "
            "verified, so it was not adopted and nothing was deleted"
        )
    return ""


def _manifest_digest(plan_id: str, transcript_digest: str) -> str:
    """The plan-and-transcript digest, for the handoff journal's own record.

    NOT EVIDENCE, and deliberately no longer the value a commit decides on. It is
    computed from the OWNER's own manifest and (before this round) echoed back by
    the destination unchanged, so comparing it proved only that the destination
    could copy a string it was handed — a zeroed digest and a truncated copy both
    committed through it (review round 1, M-2). What the owner decides on is
    ``sync.copy_content_digest``: a digest each end derives from its OWN bytes.
    """
    import hashlib

    return "sha256:" + hashlib.sha256(f"{plan_id}|{transcript_digest}".encode()).hexdigest()


def _destination_move(
    server: "RelayServer",
    session_id: str,
    *,
    transport: LinkTransport,
    keep: bool,
    wait_s: float,
    owner_device: str,
    owner_name: str,
    asked: bool = False,
    adopt_under_id: str = "",
    engage_on_arrival: bool = False,
    unattended: bool = False,
) -> tuple[SessionMoveResult | None, SessionMoveRefusal | None, str]:
    """Pull ``session_id`` from its owner and adopt it here. THE destination path.

    Returns ``(result, refusal, new_id)``. ``asked`` distinguishes the offload
    (the owner invited us, so no status/prepare handshake is owed) from a recall.

    ``adopt_under_id`` is the id a ``keep`` copy must be adopted under, for the one caller
    that has to name it BEFORE this function runs: an invite acks the id the pull will
    use, so the inviter's receipt names a session that will exist (QA delta, Q-D2).
    Empty means "mint one here", which is what every uninvited caller does. It is NOT
    called ``target_id``: that name belongs to the id this session is adopted under
    (``session_id`` for a move, the mint for a copy), and one parameter holding both
    meanings is how the receipt and the directory drifted apart in the first place
    (Aida round 2, nit 4).

    The phases are reported as they happen, and the caller's contract only requires
    that a SUCCESS reports an openable phase — so a move that stopped at
    ``committed`` (the destination holds it, the source's cleanup confirmation is
    still in flight) is a success, which is exactly what §6.3 step 18 says about
    ``done`` being best-effort.

    ``engage_on_arrival`` is the ONE opt-in step this path has beyond the protocol
    (§6.3 step 17's ``--open``, spelled as a flag): the conversation is engaged here
    as soon as it has arrived, so it is live on this device without a first prompt.
    Default off, and absent from the receipt when off, so a caller that does not ask
    for it — every caller that existed before this parameter — gets the identical
    move it always got. It is NOT turn handoff and cannot become one: it runs AFTER
    the move has committed, it gives the runtime no work, and a session with a turn
    in flight is still refused by the owner before any of this can start (§6.4).
    """
    from local_operator import fork as fork_mod
    from local_operator.network import sync as sync_mod
    from local_operator.resume import ORIGIN_FORK, mark_session_origin
    from local_operator.session.placement import (
        HANDOFF_PHASE_HANDING_OFF,
        MeshStamp,
        SessionPlacement,
        clear_handoff_entry,
        write_handoff_entry,
        write_stamp_into,
    )

    progress = progress_for(server)
    me = server.identity.device_id
    phases: list[MovePhaseStamp] = []

    def note(phase: SessionMovePhase) -> None:
        if phase not in [item["phase"] for item in phases]:
            phases.append({"phase": phase, "at": time.time()})
        progress.note(session_id, phase)

    lease_epoch = ""
    manifest: dict[str, Any] = {}
    attempts = 0
    while True:
        try:
            status = transport.ask(
                {
                    "op": "net_session_move",
                    "phase": "status",
                    "session_id": session_id,
                    "to_device": me,
                }
            )
        except Moved as refusal:
            if refusal.code == "unreachable":
                return None, _move_refusal(session_id, "unreachable", refusal.message), ""
            return None, _move_refusal(session_id, refusal.code, refusal.message), ""
        if str(status.get("result") or "") == "tombstone":
            tombstone = status.get("tombstone") or {}
            if not _tombstone_names_this_device(server, tombstone):
                # THE RECORD NAMES ANOTHER DEVICE (review round 2, MAJOR), so there is
                # nothing here to finish: the collapse of "a record exists" into "the
                # record is mine" is how a stale staged copy becomes a second owner.
                # Checked on THIS side rather than trusted from the peer's answer,
                # because the promote below is the irreversible act.
                return None, _adopt_elsewhere_refusal(server, session_id, tombstone), ""
            # THE CRASH WINDOW THIS EXISTS FOR (§6.5 row 5): the owner already
            # committed and we may hold a verified staging directory. Ask for what
            # we have and finish; there is no `prepare` to make, the id is gone.
            return _finish_from_tombstone(server, session_id, owner_device=owner_device)
        if not status.get("owner"):
            return (
                None,
                _move_refusal(
                    session_id,
                    "not_owner",
                    f"{owner_name or owner_device or 'that device'} does not hold "
                    f"{session_id} any more, so nothing was moved",
                ),
                "",
            )
        pending = status.get("pending") or {}
        # WHETHER THE OWNER HAS IT ARCHIVED, asked for once here and carried to the
        # promote in ``ready.json`` (see that file's ``archived`` field). Default
        # False so a peer too old to send the field behaves exactly as before.
        archived = bool(status.get("archived"))
        if pending and str(pending.get("to_device") or "") != me:
            return (
                None,
                _move_refusal(
                    session_id,
                    "in_progress",
                    "another move of that conversation is already under way, so nothing was "
                    "changed here",
                ),
                "",
            )
        try:
            prepared = transport.ask(
                {
                    "op": "net_session_move",
                    "phase": "prepare",
                    "session_id": session_id,
                    "to_device": me,
                    "mode": "keep" if keep else "move",
                    "resume": bool(pending),
                    "have": {},
                }
            )
        except Moved as refusal:
            if refusal.code == "busy" and wait_s > 0:
                # §6.4: a FRESH idle probe every 5 s. Each retry can land, and none
                # of them can lose a turn — the runtime re-asks its own predicate.
                remaining = wait_s - attempts * MOVE_WAIT_POLL_S
                if remaining > 0:
                    attempts += 1
                    time.sleep(min(MOVE_WAIT_POLL_S, remaining))
                    continue
            code = refusal.code if refusal.code in MOVE_REFUSAL_CODES else "refused"
            return None, _move_refusal(session_id, code, refusal.message), ""
        lease_epoch = str(prepared.get("lease_epoch") or "")
        manifest_raw = prepared.get("manifest")
        manifest: dict[str, Any] = manifest_raw if isinstance(manifest_raw, dict) else {}
        note("prepared")
        break

    target_id = session_id
    if keep:
        # ONE MINT, AND IT IS THIS DEVICE'S (QA delta, Q-D2). A ``--keep`` copy is
        # created HERE, so the directory that appears is the one this id names — an
        # inviter that minted its own would publish a receipt naming a session no
        # device holds, and the follow-up move the user is told to run would answer
        # "no device in this network holds …". The invited path therefore hands its
        # own mint down (``_destination_invite``), because its ack has to carry the id
        # before this thread has run at all.
        target_id = adopt_under_id or fork_mod.new_session_id()
    staging = sync_mod.staging_dir(server.root, target_id)
    try:
        sync_mod.sync_from(
            server.root,
            session_id,
            ask=transport.ask,
            owner_device=owner_device,
            into=staging,
            # THE STORE ROOT, NOT THE BLOB DIRECTORY. ``_destination_for`` appends
            # the store-relative name (``attachments/<d>.bin``) to whatever it is given,
            # so handing it ``<config>/attachments`` put every blob at
            # ``<config>/attachments/attachments/<d>.bin`` — a path nothing reads,
            # which is why images in a moved conversation did not load on the device
            # they moved to (review round 1, M-1).
            attachments_root=Path(server.root),
            write_cursor=False,
            # A MOVE'S DESTINATION IS NOT A REPLICA HOLDER: see ``sync._plan``.
            purpose="move",
        )
    except sync_mod.SyncRefused as refusal:
        # A ROLLBACK, not a failure (§6.3 step 12): the source still holds an intact
        # directory and no writer, so the honest report is "nothing moved".
        _roll_back_destination(server, session_id, staging)
        return (
            None,
            _move_refusal(
                session_id,
                "digest_mismatch" if refusal.code == "digest_mismatch" else refusal.code,
                refusal.message if refusal.code == "digest_mismatch" else refusal.message,
            ),
            "",
        )

    # THE DESTINATION'S OWN CONTENT DIGEST, computed from the bytes it actually
    # holds over the same function the owner uses (``sync.copy_content_digest``),
    # and the value the owner commits on (review round 1, M-2). Computed HERE,
    # before the stamp and the origin marker land, and again before the promote
    # below: the two names the adopting device writes itself are skipped by the
    # function, so the value is stable across both moments.
    content_digest = _staging_content_digest(server, target_id, staging)
    # THE STAMP GOES INTO THE STAGING DIRECTORY, before the promote, so the
    # session, its ownership and its lineage become visible in ONE step and no
    # scanner ever sees a half-promoted session (design §6.3 step 16).
    # ``write_stamp_into`` rather than ``write_stamp``: the latter builds
    # ``sessions/<id>/mesh.json``, and creating that directory is both the
    # half-promoted state this ordering exists to prevent and the reason the
    # promote would then refuse (the target is in the way).
    write_stamp_into(
        staging,
        MeshStamp(
            session_id=target_id,
            # The network this move ran over, so a listing can file the row.
            network_id=transport.link.network_id,
            home_device=me,
            placement=SessionPlacement(
                mode="peer", network_id=transport.link.network_id, home_device=me, stamp_revision=1
            ),
            origin={
                "kind": ORIGIN_FORK if keep else "moved",
                "source_device": owner_device,
                "source_session_id": session_id,
                "moved_at": time.time(),
            },
            # CARRIED AUTO AUTHORITY (§6 defect 2): the flag rode the invite from the
            # device that held the session, and it is stamped HERE because the
            # destination's runtime construction reads the STAMP, not the invite. It
            # is a carry, never the authority: the construction re-checks this
            # device's ``unattended`` grant for ``owner_device`` (the row the origin
            # names), so a revoked grant takes effect at the next engage.
            unattended=unattended,
        ),
    )
    # ORIGIN_FORK FOR BOTH MODES, and this is not cosmetic: ``resume`` decides
    # whether an install's picker shows a directory as the USER's conversation from
    # ``origin.json`` alone (``resume.USER_ORIGINS`` is ``{fork}``, and
    # ``is_user_origin`` reads any other value as a delegated child). Writing
    # ``"moved"`` there would make a moved conversation read as somebody else's
    # subagent run. The mesh's own "this was moved" fact belongs on the stamp's
    # ``origin.kind``, which is a different axis with a different reader.
    mark_session_origin(staging, ORIGIN_FORK, parent=session_id, source_device=owner_device)
    if keep:
        # An off-by-one row boundary is then a DIVERGENCE POINT rather than a
        # corruption (§7.4), which is what makes copying a live session safe.
        (staging / fork_mod.FORK_BOUNDARY_NAME).write_text(
            json.dumps({"version": fork_mod.FORK_BOUNDARY_VERSION, "created_at": time.time()}),
            encoding="utf-8",
        )
    ready = {
        "version": 1,
        "lease_epoch": lease_epoch,
        # THE VALUE THE OWNER CHECKS. It is a digest of THIS device's bytes (not of
        # the owner's manifest, which used to be echoed back unchanged and proved
        # nothing), so the move commits only when the copy is complete.
        "content_digest": content_digest,
        "plan_id": str(manifest.get("plan_id") or ""),
        "mode": "keep" if keep else "move",
        "owner_device": owner_device,
        "source_session_id": session_id,
        # WHETHER THE CONVERSATION WAS HIDDEN WHERE IT CAME FROM, and why it rides
        # here rather than being asked for again at the promote: this file is the
        # ONE thing that survives a crash between the copy and the adopt (§6.5 row
        # 5 adopts from it with the owner possibly gone), so a bit carried only in
        # memory would be lost in exactly the case the file exists for. The archive
        # index is device-scoped (``session/archived.py``), which is why nothing
        # carried it before: a session archived on A arrived unarchived on B.
        # ``--keep`` DELIBERATELY DOES NOT INHERIT IT: a fork mints a NEW id, and a
        # copy the user just asked for must not arrive hidden from every listing.
        "archived": bool(archived) and not keep,
        "promoted": False,
        "at": time.time(),
    }
    (staging / "ready.json").write_text(json.dumps(ready, sort_keys=True), encoding="utf-8")
    if not keep:
        # From HERE the id is in transit on this device too, so an engage here
        # cannot create a fresh empty session under an id that is about to be
        # promoted onto it. ``prepared`` until the owner confirms; ``handing-off``
        # once this device's copy is verified and waiting on the commit.
        write_handoff_entry(
            server.root,
            session_id,
            {
                "role": "destination",
                "phase": HANDOFF_PHASE_HANDING_OFF,
                "from_device": owner_device,
                # THE SOURCE'S NAME TOO, because this entry's refusal sentence has to
                # name the device the conversation is coming FROM: with only
                # ``to_device`` (which is this device) an engage here was refused
                # with "being received from <this device's own id>" — a sentence
                # that names the wrong end and reads as nonsense to the person
                # reading it (review round 1, M-3/NIT 5).
                "from_name": owner_name,
                "to_device": me,
                "lease_epoch": lease_epoch,
                "mode": "move",
                "instance_id": str(server.instance_id),
                "at": time.time(),
            },
        )

    if not keep:
        try:
            transport.ask(
                {
                    "op": "net_session_move",
                    "phase": "ready",
                    "session_id": session_id,
                    "lease_epoch": lease_epoch,
                    # WHAT THIS DEVICE HOLDS, so the owner can compare it against the
                    # digest of its own bytes before it deletes anything.
                    "content_digest": content_digest,
                    "plan_id": str(manifest.get("plan_id") or ""),
                }
            )
        except Moved as refusal:
            clear_handoff_entry(server.root, session_id)
            _roll_back_destination(server, session_id, staging)
            return None, _move_refusal(session_id, refusal.code, refusal.message), ""
        # THE OWNER HAS COMMITTED. ``handing_off`` is reported here rather than at
        # the ``ready`` SEND, because until the answer arrives this device has no
        # evidence the owner agreed, and a phase that ran ahead of the protocol
        # would tell a front end to open a session nobody has handed over.
        note("handing_off")
    # LAST CHECK BEFORE THE ONLY ``os.replace`` INTO ``sessions/``: the staged bytes
    # are re-hashed against what this device recorded when it verified them, so a
    # copy damaged between the two moments is never presented as a session (M-2's
    # destination half, and the structural form of M-5's "never a partial copy").
    damaged = _verify_staging(server, target_id, staging, ready)
    if damaged:
        # THE STAGING DIRECTORY IS KEPT, deliberately: by now the owner may already
        # have deleted its copy, so sweeping it would destroy the only one left. The
        # id is released and the sentence names where the bytes are.
        clear_handoff_entry(server.root, session_id)
        return (
            None,
            _move_refusal(
                session_id,
                "digest_mismatch",
                f"{damaged} (the copy is still under {staging}, and this device's "
                "``lop sessions sync`` cannot replace it)",
                phase_reached="handing_off",
                changed=not keep,
            ),
            "",
        )
    try:
        promoted = _promote(server, staging, target_id)
    finally:
        # THE PROMOTE IS THE POINT OF NO RETURN. However it exits — including a raise
        # between the rename and the cleanup below — the id is on this device now, so an
        # entry still saying "being received" would refuse every engage until this relay
        # restarted (review round 2, the ``p1c`` probe). Clearing it here cannot lose a
        # move: the entry is only written for a deleting move, whose directory is either
        # present (settled) or absent (and then the helper returns without touching it).
        if not keep:
            settle_promoted_handoff(server.root, session_id)
    if not promoted:
        clear_handoff_entry(server.root, session_id)
        return (
            None,
            _move_refusal(
                session_id,
                "in_progress",
                f"{target_id} could not be adopted here because a session with that id "
                "already exists on this device; nothing was changed",
                phase_reached="handing_off",
                changed=True,
            ),
            "",
        )
    note("committed")
    if not keep:
        clear_handoff_entry(server.root, session_id)
    try:
        # SENT FOR BOTH MODES, and for ``--keep`` it is the ONLY completion signal
        # the source gets: nothing on that device changed, so it has nothing to
        # wait for except this frame. Best effort by contract (§6.3 step 18) — a
        # move that stops at ``committed`` is complete from the user's side.
        transport.ask(
            {
                "op": "net_session_move",
                "phase": "done",
                "session_id": session_id,
                "new_session_id": target_id,
            }
        )
    except Moved:
        logger.debug("mobility: %s did not acknowledge the handoff", session_id)
    note("done")
    _audit(server, AUDIT_DONE, session_id, owner_device, new_session_id=target_id)
    # §6.3 STEP 17's ``--open``, AS AN OPT-IN FLAG, AND LAST ON PURPOSE. The engage
    # runs after the protocol's own final frame rather than between the commit and
    # it: ``done`` is the SOURCE's completion signal, and a runtime spawn on this
    # device (1-3 s, and up to the engage deadline when it is cold) is not something
    # a peer's confirmation should have to wait behind. ``note("done")`` above means
    # the phases reported are the MOVE's phases either way — arriving live is not a
    # phase, and a failed engage must never make a finished move report a shorter
    # one.
    engagement = _arrival_engagement(server, target_id) if engage_on_arrival else None
    receipt: dict[str, Any] = {
        "ok": True,
        "session_id": session_id,
        "new_session_id": target_id,
        "mode": "keep" if keep else "move",
        "from_device": _device_block(server, owner_device, owner_name),
        "to_device": _own_block(server),
        "phase": "done",
        "phases": phases,
    }
    if engagement is not None:
        receipt["engagement"] = engagement
    return (
        cast(SessionMoveResult, receipt),
        None,
        target_id,
    )


def _arrival_engagement(server: "RelayServer", session_id: str) -> MoveEngagement:
    """Start (or join) a runtime for a conversation that has just arrived here.

    THE WHOLE OF ``engage_on_arrival``, and the only place the feature touches the
    runtime. The caller has already committed the move, so this is a SECOND,
    SEPARATE act on what is now an ordinary local session — and that is what makes
    the three properties below load-bearing rather than incidental:

    * **A failure here is not a failed move, and cannot become one.** The bytes are
      on this disk, the journal is settled, the source's copy is retired; nothing in
      this function promotes, journals, copies or unlinks anything. The sentence
      travels beside a successful receipt (``MoveEngagement``) so a caller can tell
      "the conversation is here" from "it is live here" — never a refusal that would
      send someone to retry a transfer that has already happened.
    * **It SUPERVISES NO TURN.** Engaging starts or joins a runtime and hands it no
      work, so this is not the turn handoff the design rejected (§6.4, and the note
      at step 17): a session with a turn in flight was refused by its owner before
      this path ran, and ``--wait`` is the remedy for that refusal, unchanged.
    * **It is the same engage every other caller makes, arbitration included.** The
      call goes to ``RelayServer.engage_session`` — the function
      ``net_session_engage`` answers with — and so through ``launch.engage_runtime``,
      the single point where the LEASE decides who runs. ``_promote`` leaves the
      lease unclaimed on purpose (§6.3 step 17), which makes this a first-engager
      racing any other first-engager, and LOSING THAT RACE IS A JOIN rather than a
      refusal: the engage loop probes the lease BEFORE it spawns, waits for the
      winner's record and delivers to it, so the contender that came second is joined
      onto the runtime that won and this function reports that layer's own state —
      never a victory, and never a defeat, of its own invention. (The refusal shape on
      this route is the OTHER event: no runtime reachable within the deadline, or every
      candidate the engage was allowed to start died. ``RelayServer.engage_session``
      owns both spellings.) There is deliberately no retry and no "did anything
      beat me" check here: both would be a second opinion about a decision the
      lease already made, and the second defeat the first one's honesty.

    The order inside ``_destination_move`` matters to this call: for a ``move`` (not
    ``--keep``) the handoff entry is cleared before this runs, because INV-1's guard
    refuses to engage an id the journal still names as in transit — the same
    ``prepared`` state this device has just finished leaving.

    TOTALLY DEFENSIVE ON PURPOSE, AND ONLY HERE. ``_engage_locally`` describes the
    failures a runtime start is known to produce (its own caught tuple) and this adds
    the ones it does not: at this point the transfer is COMMITTED and irreversible, so
    an exception nobody anticipated in an optional step must not be allowed to destroy
    the answer to a move that has already happened. Nothing is swallowed — the failure
    is logged with its traceback and reported in ``detail`` — and the catch is the
    outermost frame of the move, where there is no later reader to hand it to.
    """
    try:
        engagement = cast(MoveEngagement, server.engage_session(session_id))
    except Exception:  # noqa: BLE001 — see the docstring: reported, never swallowed
        logger.exception("mobility: engaging %s on arrival failed", session_id)
        return {
            "engaged": False,
            "detail": (
                f"could not start a runtime for session {session_id} on arrival, for a "
                "reason the engage path does not recognise; that device's own log has "
                "the cause (`lop network doctor`)"
            ),
            "session_id": session_id,
        }
    if not engagement.get("engaged"):
        # A REFUSAL IS AN ANSWER, NOT AN EXCEPTION, so the handler above never sees it —
        # and on the OFFLOAD route nothing else does either: the inviter's receipt is the
        # SOURCE's, because that device settles on its OWN durable progress and drops this
        # block when it arrives (``_destination_invite._run``), so a destination that could
        # not start the runtime left no trace anywhere without this line (review round 1,
        # R1-2; QA round 1, Q-1 — one asymmetry seen from its two sides). INFO rather than
        # DEBUG because this log IS the surface: the words the CLI gives an operator point
        # at "that device's own log" by name, and a DEBUG line is not what that promise
        # means at the default level.
        logger.info(
            "mobility: could not engage %s on arrival: %s",
            session_id,
            engagement.get("detail") or "no reason given",
        )
    return engagement


def _finish_from_tombstone(
    server: "RelayServer",
    session_id: str,
    *,
    owner_device: str,
) -> tuple[SessionMoveResult | None, SessionMoveRefusal | None, str]:
    """The owner has tombstoned the id to us: adopt the verified copy we hold.

    §6.5's last row, and the reason the tombstone exists: the owner answered
    ``committed`` and died, or its answer never reached us, and the ONE fact that
    settles it is that the id is now tombstoned to THIS device. Nothing else is
    asked of the owner — which is also why this works when the owner is gone.

    WHO ESTABLISHES "TO THIS DEVICE" IS THE CALLER, and every caller compares before it
    gets here (review round 2, MAJOR): ``_tombstone_names_this_device`` in
    ``_destination_move`` and ``_reconcile_destination``, where a peer's answer is the
    input, and ``_recall``'s own comparison of this device's local record. This function
    cannot make that comparison for itself — the record is the OWNER's file, so the only
    thing it could read here is the answer somebody else wrote.
    """
    from local_operator.network import sync as sync_mod
    from local_operator.session.placement import clear_handoff_entry

    progress = progress_for(server)
    staging = sync_mod.staging_dir(server.root, session_id)
    ready_path = staging / "ready.json"
    if not ready_path.is_file():
        return (
            None,
            _move_refusal(
                session_id,
                "no_replica",
                f"{session_id} was handed to this device, and this device no longer holds "
                "the copy it made; nothing was changed",
                phase_reached="handing_off",
                changed=True,
            ),
            "",
        )
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    # THE SAME RE-HASH AS THE LIVE PATH, and for the same reason: this is a promote
    # into ``sessions/`` driven by a file, and after a crash the file is the only
    # thing that says the bytes were ever verified (M-2). A ``ready.json`` whose
    # digest no longer describes the staged bytes refuses here, keeping the
    # staging directory so the bytes are not lost with the id.
    damaged = _verify_staging(server, session_id, staging, ready)
    if damaged:
        return (
            None,
            _move_refusal(
                session_id,
                "digest_mismatch",
                f"{damaged} (the copy is still under {staging})",
                phase_reached="handing_off",
                changed=True,
            ),
            "",
        )
    if not _promote(server, staging, session_id):
        return (
            None,
            _move_refusal(
                session_id,
                "in_progress",
                f"{session_id} already exists on this device; nothing was changed",
                phase_reached="handing_off",
                changed=True,
            ),
            "",
        )
    clear_handoff_entry(server.root, session_id)
    progress.note(session_id, "committed")
    _audit(server, AUDIT_DONE, session_id, owner_device, resumed=True)
    phases: list[MovePhaseStamp] = [
        {"phase": "prepared", "at": float(ready.get("at") or time.time())},
        {"phase": "handing_off", "at": float(ready.get("at") or time.time())},
        {"phase": "committed", "at": time.time()},
    ]
    return (
        {
            "ok": True,
            "session_id": session_id,
            "new_session_id": session_id,
            "mode": "move",
            "from_device": _device_block(server, owner_device),
            "to_device": _own_block(server),
            "phase": "committed",
            "phases": phases,
        },
        None,
        session_id,
    )


def _promote(server: "RelayServer", staging: Path, target_id: str) -> bool:
    """``os.replace(staging, sessions/<id>)`` — the ONE atomic promote (§6.3 step 16).

    One rename, on one filesystem (both are under the config root), so the session
    directory, its stamp and its lineage marker appear in a single step. A scanner
    that listed the store one syscall earlier sees nothing, not half a session.

    A TARGET IN THE WAY REFUSES — EXCEPT A BARE REMNANT. An id that already has a
    directory here is the one thing a promote must never write over, so a real
    session (or anything that could be one) returns False and the caller words
    the refusal. The single exception is the corpse ``_bare_remnant`` describes:
    a reader's leftover holding no conversation, whose strict-predicate match is
    the ONLY license this function has to clear it (measured 2026-09-29; the
    recall stranded on exactly that remnant).

    WHAT IS DELIBERATELY NOT HERE, because the design's step 17 says to do it and
    the code says otherwise: ``claim_session``. It writes ``.session.pid`` naming
    ``os.getpid()`` — which for a relay is a live process that holds no transcript
    lease at all. The successor runtime's ``acquire_session_lease`` reads that
    mirror and refuses while the holder looks alive (``session_lease.py:374-380``),
    and ``fork_session`` documents the same trap arriving through its own door
    ("the marker left behind names the parent as the fork's live owner"). So the
    lease stays unclaimed and the FIRST runtime to engage claims it, which is
    exactly what happens to a session that has just been resumed.
    """
    target = Path(server.root) / "sessions" / target_id
    if target.exists():
        # A BARE REMNANT IS NOT AN OWNER (measured 2026-09-29 — see
        # ``_bare_remnant`` for the shape). A corpse holds nothing a session
        # could lose: it is the leftover of a READER (a viewer's cut-off verdict
        # on the device that does not own the id), and the recall arriving with
        # the verified copy — which is then the ONLY complete copy anywhere — is
        # exactly the moment it must not be refused. Anything else in the way
        # refuses exactly as before; nothing is ever cleared without the strict
        # predicate in ``_bare_remnant`` agreeing, and the removal goes through
        # ``cleanup.remove_session_dir`` — the ONE rmtree of a session directory
        # in this codebase (``tests/unit/session/test_no_session_deletion.py``
        # fails the build on any other), whose guards refuse an unmarked store.
        # The store is marked first, the same authority ``_remove_handed_away``
        # exercises for the hand-back this device decided: a promote's target
        # store is one this device is adopting INTO, which is the same claim
        # ``session_factory`` makes when it marks a store it created.
        if not _bare_remnant(target):
            return False
        from local_operator.session.cleanup import (
            MESH_REMNANT_POLICY,
            mark_store,
            remove_session_dir,
        )

        mark_store(target.parent)
        try:
            cleared = remove_session_dir(
                target,
                config_dir=server.root,
                policy=MESH_REMNANT_POLICY,
                reason="mesh-recall: cleared a bare remnant so the verified copy could land",
                actor="mesh:promote",
            )
        except OSError as exc:
            # AN I/O FAILURE REFUSES, IT DOES NOT RAISE — the contract this call
            # site kept when it wrapped its own ``rmtree``, and ``apply_cleanup``
            # wraps the same call in the same shape for the same reason.
            # ``remove_session_dir`` does not catch its own rmtree, and an
            # exception escaping a relay's frame handler drops the control
            # connection, which the caller reads as ``relay_unavailable`` — "a
            # wedged relay" — about a relay that is serving (PR #1756 round 3,
            # MINOR). A promote must never write over what it could not clear,
            # so warn and leave the remnant where it is.
            logger.warning("mobility: a bare remnant at %s could not be cleared: %s", target, exc)
            return False
        if not cleared:
            # The guard refused (or the target moved under us): the remnant is
            # still in the way, and a promote must never write over what it
            # could not clear — refuse exactly as before.
            return False
        logger.info(
            "mobility: cleared a bare remnant (no session content) at %s so the "
            "verified copy of %s could land",
            target,
            target_id,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(str(staging), str(target))
    except OSError as exc:
        logger.warning("mobility: could not adopt %s on this device: %s", target_id, exc)
        return False
    # ``ready.json`` is the move's own boot marker, not session content: it exists
    # so a crash before the promote can be settled, and inside the session it would
    # be a file every future reader has to learn to ignore. IT IS READ FIRST, and
    # for a second reason: it is the only carrier of whether the conversation was
    # ARCHIVED where it came from (the ``archived`` field), and the archive index is
    # device-scoped — so the bit cannot be re-derived here, and a session archived on
    # one device arrived unarchived on the other until this read existed.
    ready_path = target / "ready.json"
    try:
        ready_doc = json.loads(ready_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Absent (a replica promote never wrote one) or unreadable: not-archived is
        # the same answer a store that never archived anything gives, and a promote
        # must never fail because bookkeeping could not be read.
        ready_doc = {}
    ready_path.unlink(missing_ok=True)
    # THE STORE MARKER, which is what makes this directory REMOVABLE later.
    # ``remove_session_dir`` refuses a store that carries no marker, and the marker
    # is written where a session is CREATED (``session_factory`` calls
    # ``cleanup.mark_store``); a destination that has never created one — a freshly
    # paired second machine, which is the case mobility exists for — therefore had
    # no marker, and the copy it adopted could never be deleted. Measured
    # 2026-09-26 on the two-device rig: device B adopted a session, then handed it
    # back on a recall; the source-side commit wrote its tombstone, called
    # ``remove_session_dir`` and logged ``cleanup="pending"`` (the refusal is
    # logged at WARNING), so B KEPT the directory — the same id on two devices,
    # both listing it as a local conversation, which is exactly INV-1. Marking the
    # store is the honest statement that this device now IS a session store: it
    # holds one.
    from local_operator.session.cleanup import mark_store

    mark_store(target.parent)
    # AND THE ID IS THIS DEVICE'S AGAIN, so any tombstone naming somewhere else is
    # now a lie about a session this device holds. Measured 2026-09-26 (session-
    # mobility audit, probe 8): A handed `bc371569005d` to B, recalled it, and A
    # held the directory with `mesh.json home_device: A` WHILE its tombstone index
    # still said the id had moved to B — the state the audit's wedged session was
    # found in, where three verb paths disagreed about where it lived (a local
    # `--resume` routed the user to the peer, `move --to peer` answered
    # "already lives on this device", and `--to local` answered "already_local").
    # A tombstone and a directory for one id are never both true, and the promote
    # is the instant the directory becomes true.
    from local_operator.network.projection import forget_tombstone

    forget_tombstone(target_id, config_dir=server.root)
    # AND THE ARCHIVE STATE COMES BACK WITH IT. The operator's requirement: a
    # session archived on the device it left must still be archived on the device it
    # arrives on — otherwise a conversation the user deliberately hid reappears in
    # the default listing of the machine they moved it to. The index is
    # device-scoped (``session/archived.py``), and the source's own entry is pruned
    # at read once its directory is gone, so this is the whole hand-off: one write
    # here, no coordination with the source afterwards.
    if bool(ready_doc.get("archived")):
        from local_operator.session.archived import set_archived

        set_archived(server.root, target_id, True)
    return True


def _roll_back_destination(server: "RelayServer", session_id: str, staging: Path) -> None:
    """Sweep a staging directory whose move was refused. Never touches ``sessions/``.

    A REFUSED move is DONE from this device's side: leaving the bytes would make a
    retry resume against a plan that no longer describes them, and the whole
    staging tree is disposable by construction — it exists only between ``prepare``
    and the promote.
    """
    import shutil

    from local_operator.session.placement import clear_handoff_entry

    clear_handoff_entry(server.root, session_id)
    try:
        shutil.rmtree(staging)
    except OSError:
        logger.debug("mobility: could not sweep %s", staging, exc_info=True)
    _audit(server, AUDIT_ROLLED_BACK, session_id, "")


# ---------------------------------------------------------------------------
# The source's half: status, prepare, ready, done, invite
# ---------------------------------------------------------------------------


def settle_promoted_handoff(root: Path, session_id: str) -> bool:
    """Clear a destination-side handoff entry whose session has ALREADY ARRIVED.

    THE ONE FACT THE INSTANCE RULE CANNOT SEE, and the reason the ``p1c`` probe still
    reproduced as written (review round 2, data-integrity items). An entry left by a
    relay that is still running is deliberately skipped by every recovery path — a live
    relay may be driving a handoff right now — but a destination entry whose session
    directory EXISTS with a transcript is not a handoff in progress: it is a handoff that
    finished. It can only be left in that state by a promote that landed and then failed
    (or died) before its cleanup, which is exactly the window between ``os.replace`` and
    ``clear_handoff_entry``. Before this, the conversation was present and usable on this
    device and every engage was refused with "This conversation is being received from
    device-a; it will be available when the move finishes" until the relay itself
    restarted.

    WHY THIS CANNOT ROLL A LIVE MOVE BACK: the promote happens only after the owner has
    answered ``committed``, and until it happens there is no session directory to find.
    The phase is required to be ``handing-off`` as well, so a ``prepared`` entry (the state
    a destination holds while its copy is still being made) is never touched.

    AND THE BYTES STILL STAGED FOR THIS ID SAY IT HAS NOT ARRIVED (review round 4). A
    session directory plus a transcript is NOT sufficient evidence: ``session_factory``'s
    opener calls ``recover_stale_handoff`` and then the handoff guard, so a recovery that
    settled the entry on that evidence alone disarmed the refusal it runs immediately
    before — a destination entry whose verified copy is still in ``network/staging``
    (the owner may already have deleted its own bytes, so that copy is the only one left)
    was cleared instead of refused. ``_promote`` is ONE ``os.replace``, so a promote that
    landed consumed this device's staging directory; a staging directory that is still
    here is a handoff that has not finished, whatever else is on disk.

    ``ready.json`` goes with it: the marker is the move's own bookkeeping, and a promote
    that died before deleting it would otherwise leave it inside a conversation the user
    opens (``_reconcile_destination`` removes it for the same reason).
    """
    from local_operator.session.placement import clear_handoff_entry, handoff_in_flight

    try:
        entry = handoff_in_flight(root, session_id)
    except Exception:  # noqa: BLE001 — an unreadable journal is the guard's refusal
        return False
    if not entry or str(entry.get("role") or "source") != "destination":
        return False
    if str(entry.get("phase") or "") != "handing-off":
        return False
    target = Path(root) / "sessions" / session_id
    if not (target / "transcript.jsonl").is_file():
        return False
    from local_operator.network import sync as sync_mod

    # STAGED BYTES OUTRANK A PRESENT SESSION DIRECTORY (see the docstring's last
    # paragraph). ``staging_dir``/``NETWORK_DIRNAME`` are the sync plane's own names, so
    # they are read from it rather than re-spelled here — a second spelling of the store
    # layout is how this check would drift away from the thing it is checking.
    if sync_mod.staging_dir(Path(root), session_id).exists():
        return False
    (target / "ready.json").unlink(missing_ok=True)
    clear_handoff_entry(root, session_id)
    logger.info(
        "mobility: %s had already arrived on this device; cleared the finished handoff entry",
        session_id,
    )
    return True


def _source_refused(
    server: "RelayServer", link: "PeerLink", frame: dict[str, Any]
) -> dict[str, Any]:
    """The DESTINATION's refusal of a move this device invited, reported back.

    WHY THIS PHASE EXISTS (QA round 1, Q4a). An offload's inviter watches its OWN
    durable progress and never asks the destination (``_offload``): the destination
    drives the copy, so the only thing the source can honestly report is what its own
    files say. That rule breaks for a REFUSAL, which writes nothing here — so the
    inviter sat out its whole budget and answered "the outcome is unconfirmed" for a
    move that never happened. Measured on 2026-09-24 with the desktop's own view
    holding the session: this device's refusal ("This session is open in another
    terminal or attached client. Disconnect that client, then move again.") was
    produced in 3 ms and reached the user 9 times out of 9 as a 60 s timeout reading
    "the request was sent, so the move may have happened".

    NOTHING IS TRUSTED BEYOND WHICH WAY THE WAIT ENDS. It changes no state and
    survives no further than the waiting call: the refusal is recorded in memory for
    the one front end that is waiting (:class:`_Progress.note_refusal`), and it can
    only SHORTEN a wait — the committed path is checked first on both sides, so a
    refusal that crosses a commit in flight is ignored rather than reported over it.
    The recording is keyed by the device that sent it, so a member that did not
    receive this move cannot end it.

    The sentence is carried VERBATIM. This side never saw the guard that fired, and a
    paraphrase here would be the second sentence table for one condition that §8.2
    exists to prevent.

    The reply says ``recorded`` rather than ``refused`` on purpose: ``refused`` is
    :meth:`LinkTransport.ask`'s sentinel for a REFUSED OP, so a handler answering with
    it would make the reporting device raise on its own notification.
    """
    session_id = str(frame["session_id"])
    progress_for(server).note_refusal(
        session_id,
        code=str(frame.get("code") or "refused"),
        message=str(frame.get("message") or "the other device refused the move"),
        from_device=str(getattr(link, "device_id", "") or ""),
    )
    return {"result": "recorded", "session_id": session_id}


def _source_status(
    server: "RelayServer", link: "PeerLink", frame: dict[str, Any]
) -> dict[str, Any]:
    """What the owner knows about an id the destination is asking about.

    Answers the tombstone for an id this device handed away, which is what makes
    §6.5's recovery table work: a destination that crashed after the commit asks
    here, and the answer — "not mine any more, and it is yours" — is the commit.

    AND IT IS THE COMMIT ONLY FOR THE DEVICE IT NAMES (review round 2, MAJOR). The
    record carries its taker (``tombstone["device_id"]``), so the same answer is not
    true for a device the handoff was not made to: one that still holds a verified
    copy of an earlier, rolled-back attempt would read "yours" as permission to promote
    it, and two devices would each hold a promoted copy of one id. The asker is the
    link's admitted identity (:func:`_asking_device`) — never a field the asker writes
    — and a mismatch is a refusal that says where the id went instead.

    THE REACHABILITY OF A MISMATCH IS NOT WHAT MAKES THIS WORTH DOING: a peer frame
    for an id this device no longer owns is already scoped by ``authorizer._move_scope``
    to the device the record names, so the ask is normally refused a layer earlier. The
    handler is gated anyway, because the two-layer agreement is exactly what a future
    change to one of them breaks, and this is the layer that produces the answer.
    """
    from local_operator.session.archived import archived_ids

    session_id = str(frame["session_id"])
    if _owned_here(server, session_id):
        from local_operator.mobile.attach_client import find_runtime_record
        from local_operator.session.placement import handoff_in_flight

        record, _pid = find_runtime_record(server.root, session_id)
        entry = None
        try:
            entry = handoff_in_flight(server.root, session_id)
        except Exception:  # noqa: BLE001 — the journal's own refusal text is elsewhere
            entry = None
        return {
            "result": "state",
            "owner": True,
            "busy": record is not None,
            # THE ARCHIVE BIT TRAVELS WITH THE STATE ANSWER. It is device-scoped
            # (``session/archived.py`` writes this device's own index), so the only
            # place a destination can learn it is from the device that has it — and
            # the operator named archiving as a thing that must hold on the remote
            # session (review round 1, MINOR 1).
            "archived": session_id in archived_ids(server.root),
            "pending": entry,
            "session_id": session_id,
        }
    tombstone = _tombstone(server.root, session_id)
    if tombstone:
        if str(tombstone.get("device_id") or "") == _asking_device(link, frame):
            return {
                "result": "tombstone",
                "owner": False,
                "tombstone": tombstone,
                "session_id": session_id,
            }
        # A RECORD FOR SOMEBODY ELSE IS NOT A REFUSAL TO ANSWER (review round 2,
        # MAJOR): it says where the id went, and the asker may be holding the only
        # bytes of an older attempt under the same id. The sentence is the module's
        # own for that state (``_tombstone_recovery_clause``) rather than a second
        # wording of it, and it names the device the record names.
        return {
            "result": "refused",
            "code": "not_owner",
            "message": (
                f"this device does not hold {session_id} any more"
                + _tombstone_recovery_clause(server, session_id)
            ),
            "session_id": session_id,
        }
    return {
        "result": "refused",
        "code": "not_owner",
        "message": f"this device does not hold {session_id}",
        "session_id": session_id,
    }


def _source_prepare(
    server: "RelayServer", link: "PeerLink", frame: dict[str, Any]
) -> dict[str, Any]:
    """Quiesce, journal, and hand the destination a manifest. NO mutation yet.

    Every validation is before the first mutation, and the busy refusal leaves both
    devices untouched (§6.3 step 2/3) — which is what makes ``--wait`` a re-probe
    rather than a resumed half-move.
    """
    from local_operator.network import sync as sync_mod
    from local_operator.session.placement import (
        HANDOFF_PHASE_PREPARED,
        handoff_in_flight,
        write_handoff_entry,
    )

    session_id = str(frame["session_id"])
    to_device = str(frame.get("to_device") or link.device_id)
    mode = str(frame.get("mode") or "move")
    keep = mode == "keep"
    if not _owned_here(server, session_id):
        return {
            "result": "refused",
            "code": "not_owner",
            "message": f"this device does not hold {session_id} any more",
            "session_id": session_id,
        }
    try:
        existing = handoff_in_flight(server.root, session_id)
    except Exception:  # noqa: BLE001 — an unreadable journal refuses elsewhere
        existing = None
    if existing and str(existing.get("phase") or "") in ("prepared", "handing-off"):
        # Reconcile a stale entry before refusing, then re-read: a journal entry
        # left by a crash is exactly what `--wait`'s retry must be able to clear.
        reconcile(server.root, server=server, only=session_id)
        try:
            existing = handoff_in_flight(server.root, session_id)
        except Exception:  # noqa: BLE001
            existing = None
        if existing:
            # A HALF-MOVED SESSION HAS TO BE LEGIBLE FROM THE REFUSAL ALONE. The entry
            # this device is holding may be a handoff that is genuinely under way, or
            # one whose destination died — and until it is settled, every route to
            # this conversation answers the same sentence. Measured on the two-device
            # rig 2026-09-26 (probe 9): the destination's relay was killed mid-copy,
            # A answered ``deadline_exceeded`` with "Nothing was deleted: this device
            # still holds the conversation", and SEVEN retries over ten minutes then
            # read "a handoff ... is already in progress on this device" with no way
            # to tell what was wrong or what to do; the same move landed the moment
            # A's relay restarted. The phase and the age are the two facts that make
            # the state legible (``prepared`` = no bytes were committed anywhere;
            # ``handing-off`` = monotone, the destination owns the outcome), so they
            # are NAMED here rather than left for the user to infer from a JSON
            # journal. The recovery the design's §6.5 table would apply to a stale
            # entry written by a DEAD relay is deliberately not applied here — this
            # entry's writer is alive — and that gap is recorded, not guessed at.
            phase = str(existing.get("phase") or "")
            age = _handoff_age(existing)
            # THE ESCAPE IS NAMED, because it is not derivable from the state. An
            # entry written by this device's LIVE relay is skipped by every reconcile
            # (the instance rule that stops a merely slow peer losing an in-flight
            # handoff), so no amount of retrying clears it and no other verb does
            # either: what clears it is making the writer instance stale, after which
            # the next attempt reconciles it at ``prepared`` and rolls it back
            # (measured 2026-09-26: seven retries over ten minutes read this sentence;
            # the same move landed the moment the source's relay restarted). Review
            # round 1's ruling is a user-facing abandon verb, and this clause is its
            # prerequisite: a person can act on the state they can read.
            return {
                "result": "refused",
                "code": "in_progress",
                "message": (
                    "a handoff of that conversation is already in progress on this "
                    f"device (phase: {phase or 'unknown'}, {age}), so nothing was "
                    "changed; if the other device has stopped answering, restart this "
                    "device's relay (`lop network restart`) and try the move again"
                ),
                "session_id": session_id,
            }
    if keep:
        # ``--keep`` MUTATES NOTHING here: no retirement, no journal, no tombstone.
        # The destination mints its own id and copies; the source keeps running, so
        # this is a read of the copy set with a lease still held (hence the copy's
        # torn-tail tolerance, §7.4).
        #
        # THE ``--keep`` WINDOW, STATED EXPLICITLY (review round 1 asked for this to
        # be documented or removed). There is no grace period and no fence in this
        # branch, and nothing that could be removed to make one: the copy is served
        # from a LIVE runtime, so a turn that lands while the copy is in flight may
        # be absent from the copy while present on the source. The window is
        # therefore the COPY'S OWN DURATION — ~1 s for a session of a few hundred KB
        # over loopback on this workstation, and bounded by the transfer rather than
        # by a timer. (``_require_current_plan`` refuses a plan the source has moved
        # past and ``sync_from`` re-plans up to ``SYNC_REPLAN_ATTEMPTS`` times, so a
        # source that keeps writing produces a refusal rather than a splice.) What
        # makes the result safe is that the copy is a FORK by construction: the
        # destination stamps ``origin: fork`` and writes ``fork-boundary.json``, so
        # an off-by-one row boundary is a divergence point, not a corruption (§7.4).
        manifest = sync_mod.build_manifest(
            server.root,
            session_id,
            have=frame.get("have") or {},
            attachments_dir=Path(server.root) / "attachments",
        )
        _audit(server, AUDIT_PREPARE, session_id, link.device_id, mode="keep", keep=True)
        return {
            "result": "prepared",
            "phase": "prepared",
            "lease_epoch": "",
            "mode": "keep",
            "manifest": manifest,
            "session_id": session_id,
        }
    # BEFORE ANY WORK AND BEFORE THE RETIRE: does this directory hold anything the
    # copy set cannot carry? Asking here costs a directory listing and refuses a
    # move that would delete unaccounted-for content without stopping the
    # conversation or copying a byte; the commit asks again, because a file can
    # appear in between (B-M2).
    try:
        sync_mod.assert_complete(Path(server.root) / "sessions" / session_id)
    except sync_mod.SyncRefused as incomplete:
        return {
            "result": "refused",
            "code": incomplete.code,
            "message": incomplete.message,
            "session_id": session_id,
        }
    # THE EXPENSIVE PASS RUNS BEFORE THE CONVERSATION IS STOPPED (review round 2, MAJOR
    # 2). Every member of the copy set is digested here — the one pass a move cannot avoid,
    # because the commit compares digests derived from these bytes — and only then is the
    # local runtime retired. Stopping first made a 1.9 GB scratchpad hold the session shut
    # for the whole digest, and then for the copy as well; now the stop covers the copy
    # plus a stat-level re-check, which is the only part that has to be quiescent.
    # THE MOVE'S TAIL RULE, ASKED FOR RATHER THAN ASSERTED. Three ``whole_transcript=True``
    # literals used to sit in this file, and NONE of them changed what the destination
    # receives — that is decided by the wire purpose in ``sync._plan`` (review round 3,
    # MINOR 2: reverting all three here passes every torn-tail cell; reverting the
    # predicate below fails them). All four sites now derive the answer from
    # ``sync.whole_transcript_for``, so "fixing the wrong one" is no longer possible: the
    # source's own prepare-vs-commit comparison and the plan a destination reads are keyed
    # on the same predicate.
    stamps = sync_mod.member_stamps(
        server.root,
        session_id,
        attachments_dir=Path(server.root) / "attachments",
        whole_transcript=sync_mod.whole_transcript_for(sync_mod.COPY_PURPOSE_MOVE),
    )
    outcome = _retire_local_runtime(server.root, session_id)
    if outcome["result"] != "retired" and outcome["result"] != "cold":
        return {
            "result": "refused",
            "code": "busy",
            "message": outcome["sentence"] or "this session is busy, so nothing was moved",
            "session_id": session_id,
        }
    if not sync_mod.stamps_valid(
        Path(server.root) / "sessions" / session_id,
        session_id,
        Path(server.root) / "attachments",
        stamps,
    ):
        # A TURN LANDED BETWEEN THE DIGEST AND THE RETIRE. Nothing has been journaled and
        # nothing has been deleted, so the honest answer is "ask again": the retry digests
        # the session as it now is. Re-digesting here instead would double the stopped
        # window for a race the caller can simply repeat, and ``--wait`` already re-probes.
        return {
            "result": "refused",
            "code": "busy",
            "message": (
                "the conversation changed while this device was preparing to hand it over, "
                "so nothing was changed; try again"
            ),
            "session_id": session_id,
        }
    lease_epoch = new_lease_epoch()
    manifest = sync_mod.build_manifest(
        server.root,
        session_id,
        have=frame.get("have") or {},
        attachments_dir=Path(server.root) / "attachments",
        whole_transcript=sync_mod.whole_transcript_for(sync_mod.COPY_PURPOSE_MOVE),
        stamps=stamps,
    )
    write_handoff_entry(
        server.root,
        session_id,
        {
            "role": "source",
            "phase": HANDOFF_PHASE_PREPARED,
            "to_device": to_device,
            "to_name": server._member_name(to_device),  # noqa: SLF001
            "lease_epoch": lease_epoch,
            "mode": "move",
            "requester": link.device_id,
            "plan_id": str(manifest.get("plan_id") or ""),
            # WHAT THE SOURCE WILL COMPARE THE DESTINATION'S COPY AGAINST, recorded
            # now and recomputed at the commit. ``sync.copy_content_digest`` derives
            # it from THIS device's own bytes, which is the property that makes the
            # comparison mean something: the destination reports a digest it
            # computed from ITS bytes over the same function, and the two are equal
            # only if the copy is complete (review round 1, M-2).
            "content_digest": sync_mod.copy_content_digest(
                Path(server.root) / "sessions" / session_id,
                session_id,
                Path(server.root) / "attachments",
            ),
            # WHICH RELAY WROTE THIS (see ``reconcile``): without it an entry is
            # indistinguishable from one a crashed process left, and the reconcile
            # would roll back a move that is still running.
            "instance_id": str(server.instance_id),
            "at": time.time(),
        },
    )
    progress_for(server).note(session_id, "prepared")
    _audit(server, AUDIT_PREPARE, session_id, link.device_id, mode="move")
    return {
        "result": "prepared",
        "phase": "prepared",
        "lease_epoch": lease_epoch,
        "mode": "move",
        "manifest": manifest,
        "session_id": session_id,
    }


def _remove_handed_away(root: Path, session_id: str, to_device: str) -> bool:
    """Remove a session this device has handed to another, by the ONE rm path.

    THE STORE MARKER IS WHY THIS IS A FUNCTION rather than three call sites. A
    destination that has never CREATED a session carries no
    ``.local-operator-store`` marker in its ``sessions/`` — ``session_factory``
    writes it at creation — and ``remove_session_dir`` refuses an unmarked store,
    correctly: that guard is what stops cleanup from walking into a directory that
    is not a session store. But a store holding a session ADOPTED BY A MOVE *is* a
    session store, and this device knows that better than any scanner: it adopted
    the directory, stamped it with ``mesh.json``, and on a later hand-back is the
    one handing it away. Measured 2026-09-26 on the two-device loopback rig:
    without this, the recall's commit wrote its tombstone, called
    ``remove_session_dir``, logged ``cleanup="pending"``, and LEFT THE DIRECTORY
    BEHIND — the same id then live on two devices, each listing it as a local
    conversation, which is exactly INV-1.

    ``mark_store`` is idempotent and best-effort, and it is called from here rather
    than from ``remove_session_dir`` for the reason that function's own note gives
    ("the policy must not be able to authorise its own target"): this is the
    product path that decided the session lives here, which is the same authority
    ``session_factory`` exercises when it marks a store it just created.
    """
    from local_operator.session.cleanup import (
        MESH_MOVE_POLICY,
        mark_store,
        remove_session_dir,
    )

    directory = Path(root) / "sessions" / session_id
    mark_store(directory.parent)
    return remove_session_dir(
        directory,
        config_dir=root,
        policy=MESH_MOVE_POLICY,
        reason=f"mesh-move: handed to {to_device}",
        actor=f"mesh:{to_device}",
    )


def _source_commit(
    server: "RelayServer", link: "PeerLink", frame: dict[str, Any]
) -> dict[str, Any]:
    """The owner decides, and only the owner (§6.3 step 12-15).

    Monotone: after the journal says ``handing-off`` there is no rollback, and a
    repeated ``ready`` is answered ``committed`` from the tombstone. That is what
    makes the destination's retry safe against a commit whose reply was lost.
    """
    from local_operator.network import sync as sync_mod
    from local_operator.session.placement import (
        HANDOFF_PHASE_HANDING_OFF,
        clear_handoff_entry,
        handoff_in_flight,
        write_handoff_entry,
    )

    session_id = str(frame["session_id"])
    to_device = str(link.device_id)
    tombstone = _tombstone(server.root, session_id)
    if tombstone and str(tombstone.get("device_id") or "") == to_device:
        # Already committed, and the destination is retrying because it never
        # heard. The tombstone IS the answer (§6.5).
        return {
            "result": "committed",
            "phase": "committed",
            "lease_epoch": str(tombstone.get("lease_epoch") or ""),
            "session_id": session_id,
        }
    try:
        entry = handoff_in_flight(server.root, session_id)
    except Exception:  # noqa: BLE001 — the journal's refusal text is the CLI's
        entry = None
    if not entry:
        return {
            "result": "refused",
            "code": "not_owner",
            "message": (
                f"this device has no handoff of {session_id} in progress, so nothing was "
                "handed over"
            ),
            "session_id": session_id,
        }
    if str(entry.get("to_device") or "") not in ("", to_device):
        return {
            "result": "refused",
            "code": "not_authorised",
            "message": "that handoff belongs to another device, so nothing was changed",
            "session_id": session_id,
        }
    # The SAME predicate the plan used, so a commit cannot disagree with the plan it is
    # completing about whether the transcript travelled whole.
    current = sync_mod.plan_id(
        server.root,
        session_id,
        attachments_dir=Path(server.root) / "attachments",
        whole_transcript=sync_mod.whole_transcript_for(sync_mod.COPY_PURPOSE_MOVE),
    )
    directory = Path(server.root) / "sessions" / session_id
    # WHAT THE DESTINATION SAYS IT HOLDS, and what this device's own bytes hash to
    # over the SAME function. Both are needed: ``plan_id`` is the SOURCE-state
    # digest (has the source moved on since it was prepared?), the content digest is
    # derived from a directory's bytes on each end and is therefore the only value
    # that can prove the copy is complete (review round 1, M-2).
    reported = str(frame.get("content_digest") or "")
    expected = sync_mod.copy_content_digest(
        directory, session_id, Path(server.root) / "attachments"
    )
    refusal_code = ""
    refusal_cause = ""
    refusal_message = ""
    if current != str(entry.get("plan_id") or ""):
        # A DIGEST MISMATCH IS A ROLLBACK, NOT A FAILURE (§6.3 step 12): this device
        # still holds an intact directory and no writer, so `prepared` is cleared
        # and the session simply stays here.
        refusal_code = "digest_mismatch"
        refusal_cause = "source_changed"
        refusal_message = (
            "the copy did not verify against what this device served: the conversation "
            "changed while it was being copied, so nothing was moved"
        )
    elif not reported or reported != expected:
        # M-2: THE OWNER COMPARES CONTENT IT CAN RE-DERIVE, NOT A STRING IT HANDED
        # OUT. The old check accepted any non-empty value, so a zeroed digest and a
        # copy truncated to 100 of 2,580 bytes both committed and deleted the source.
        refusal_code = "digest_mismatch"
        refusal_cause = "destination_content"
        refusal_message = (
            "the copy the destination holds does not match the bytes this device "
            "served, so nothing was moved and this device still holds the conversation"
        )
    else:
        # B-M1, AND IT IS THE POINT OF THE WHOLE COMMIT: ``prepare`` checked the
        # lease, and a process can take it in between — every way the product opens
        # a session (``lop -r``, ``lop exec``, the TUI's in-process open, a booting
        # runtime) acquires it directly, so the engage guard alone cannot stop them.
        # This is the LAST check before the only delete, and it fails closed: a
        # holder that cannot be PROVEN dead refuses the move and is named.
        from local_operator.mobile.attach_client import find_runtime_record

        _record, record_pid = find_runtime_record(server.root, session_id)
        busy = _lease_refusal(server.root, session_id, record_pid)
        if busy:
            refusal_code = "busy"
            refusal_cause = "lease_taken_after_prepare"
            refusal_message = busy
        else:
            # THE COPY SET HAS TO ACCOUNT FOR WHAT IS ABOUT TO BE DELETED (B-M2).
            # A name neither list covers is a file type the copy set has never been
            # taught, and "not copied, then deleted" is data loss with nothing to
            # recover it from — which is how ``scratchpad/`` and ``created_at.json``
            # were destroyed. Refusing is the only answer that cannot lose it.
            try:
                sync_mod.assert_complete(directory)
            except sync_mod.SyncRefused as incomplete:
                refusal_code = incomplete.code
                refusal_cause = "unlisted_content"
                refusal_message = incomplete.message
    if refusal_code:
        clear_handoff_entry(server.root, session_id)
        progress_for(server).note(session_id, "prepared")
        _audit(server, AUDIT_ROLLED_BACK, session_id, to_device, cause=refusal_cause)
        return {
            "result": "refused",
            "code": refusal_code,
            "message": refusal_message,
            "session_id": session_id,
        }
    entry = dict(entry)
    entry["phase"] = HANDOFF_PHASE_HANDING_OFF
    entry["at"] = time.time()
    write_handoff_entry(server.root, session_id, entry)
    progress_for(server).note(session_id, "handing_off")
    _audit(server, AUDIT_HANDING_OFF, session_id, to_device)

    from local_operator.network.projection import write_tombstone

    write_tombstone(
        session_id,
        device_id=to_device,
        device_name=server._member_name(to_device),  # noqa: SLF001
        network_id=link.network_id,
        config_dir=server.root,
    )
    directory = Path(server.root) / "sessions" / session_id
    removed = _remove_handed_away(Path(server.root), session_id, to_device)
    if removed:
        clear_handoff_entry(server.root, session_id)
    else:
        # THE TOMBSTONE IS WHAT GOVERNS THE ID, so the move IS complete and the
        # destination may promote. What is left is this device's own cleanup, and
        # the journal entry stays so `reconcile` retries it — never so a runtime
        # here could start (the launch guard reads the same entry).
        logger.warning(
            "mobility: %s was handed to %s but its directory could not be removed; "
            "reconcile will retry",
            session_id,
            to_device,
        )
        _audit(server, AUDIT_COMMITTED, session_id, to_device, cleanup="pending")
    progress_for(server).note(session_id, "committed")
    _audit(server, AUDIT_COMMITTED, session_id, to_device)
    return {
        "result": "committed",
        "phase": "committed",
        "lease_epoch": str(entry.get("lease_epoch") or ""),
        "session_id": session_id,
    }


def _source_done(server: "RelayServer", link: "PeerLink", frame: dict[str, Any]) -> dict[str, Any]:
    """The destination's best-effort confirmation. One event, no state change."""
    session_id = str(frame["session_id"])
    progress_for(server).note(session_id, "done")
    progress_for(server).signal(session_id, "done")
    _audit(server, AUDIT_DONE, session_id, link.device_id)
    return {"result": "done", "session_id": session_id}


def _destination_invite(
    server: "RelayServer", link: "PeerLink", frame: dict[str, Any]
) -> dict[str, Any]:
    """``O`` asks US to pull: the offload's first hop (build plan §1.1).

    Acked IMMEDIATELY and the pull runs on a thread this module starts. It cannot
    run here: this handler is serving the very link the pull must ask over, and
    ``PeerLink.request`` refuses that shape by design (it would wait on a reply
    only this thread could deliver). A new thread is not serving anything, so it
    may ask.
    """
    from local_operator import fork as fork_mod
    from local_operator.session.placement import (
        HANDOFF_PHASE_PREPARED,
        write_handoff_entry,
    )

    session_id = str(frame["session_id"])
    keep = bool(frame.get("keep"))
    # THE ``--wait`` THAT WAS DROPPED HERE. The destination's own flow re-probes a
    # busy owner every ``MOVE_WAIT_POLL_S`` for as long as it is given (the loop in
    # ``_destination_move``), and this handler passed ``wait_s=0.0`` unconditionally —
    # so an OFFLOAD never re-probed, whatever ``lop sessions move --to <peer> --wait
    # N`` the user typed, while the recall route (which passes ``wait_s`` through)
    # did. Measured on the two-device rig 2026-09-26: a session whose mock runtime
    # was parked in a 45 s ``sleep`` answered ``--wait 120`` with ``busy`` in 5.09 s,
    # the turn still running, against the guide's "re-checks ... every five seconds,
    # up to the design's thirty minutes". The term is the OWNER's (it is the side
    # that refuses), so it travels with the invite, bounded by the same cap the CLI
    # applies.
    wait_s = _wait_seconds(frame.get("wait_s"))
    # THE MOVE'S ONE OPT-IN, CARRIED LIKE ``wait_s`` AND FOR THE SAME REASON: the step
    # it configures runs HERE (this device is the destination), so a flag that stopped
    # at the inviter would be a promise nobody keeps. Read with ``bool(...or False)``
    # so a peer older than this field — which sends no key at all — arrives cold,
    # exactly as it did before the flag existed.
    engage_on_arrival = bool(frame.get("engage_on_arrival") or False)
    # THE CARRY OF THE SESSION'S AUTO AUTHORITY (§6 defect 2), read off the invite
    # like every other optional field on it: absent reads off, which is what an
    # older source sends and what every session that never asked for full-auto
    # carries. The DESTINATION's runtime re-checks the grant at engage, so this
    # bit only ever decides whether that check is asked at all.
    unattended = bool(frame.get("unattended") or False)
    owner_device = link.device_id
    owner_name = server._member_name(owner_device)  # noqa: SLF001
    if not keep and _owned_here(server, session_id):
        return {
            "result": "refused",
            "code": "not_owner",
            "message": f"{session_id} already lives on this device, so nothing was changed",
            "session_id": session_id,
        }
    new_id = fork_mod.new_session_id() if keep else ""
    if not keep:
        write_handoff_entry(
            server.root,
            session_id,
            {
                "role": "destination",
                "phase": HANDOFF_PHASE_PREPARED,
                "from_device": owner_device,
                "to_device": server.identity.device_id,
                "mode": "move",
                "instance_id": str(server.instance_id),
                "at": time.time(),
            },
        )

    def _run() -> None:
        transport = LinkTransport(server, link, session_id)
        try:
            result, refusal, _target = _destination_move(
                server,
                session_id,
                transport=transport,
                keep=keep,
                wait_s=wait_s,
                owner_device=owner_device,
                owner_name=owner_name,
                asked=True,
                # THE ID THE ACK ALREADY CARRIED, so the copy is adopted under the id
                # the inviter was told about rather than a second, unreachable one.
                adopt_under_id=new_id,
                engage_on_arrival=engage_on_arrival,
                unattended=unattended,
            )
        except Exception:  # noqa: BLE001 — the inviter polls durable state, not this
            logger.debug("mobility: invited pull of %s failed", session_id, exc_info=True)
            return
        # NOTHING IS SIGNALLED *LOCALLY*, and that is still right: the inviter watches
        # its OWN durable progress (the tombstone, or this device's `done` handler),
        # never this thread, and a notification from a worker the inviter cannot observe
        # would be a second, weaker source of truth for the same fact.
        #
        # A REFUSAL IS THE EXCEPTION, and it has to cross the link to be visible at all
        # (QA round 1, Q4a). A refusal writes nothing on the inviter's disk — that is
        # what makes it a refusal — so an inviter watching only its files sits out its
        # whole budget and then answers "the outcome is unconfirmed" for a move that
        # never started. Measured on 2026-09-24: this device refused in 3 ms (the
        # session was open in another client) and the user was told 60 s later that the
        # move might have happened. The owner is told here, over the same link it
        # invited on, which is the only channel that carries the fact.
        if result is None and refusal is not None:
            logger.info("mobility: invited pull of %s refused: %s", session_id, refusal["message"])
            try:
                transport.ask(
                    {
                        "op": "net_session_move",
                        "phase": "refused",
                        "session_id": session_id,
                        "code": str(refusal.get("code") or "refused"),
                        "message": str(refusal.get("message") or "this device refused the move"),
                    },
                    # A NOTIFICATION, NOT AN OP WITH A BUDGET: the owner answers from
                    # memory in microseconds, and the move op's own slow-op deadline
                    # (the copy's) would hold this worker for a minute if the owner
                    # were gone. A miss is harmless — the inviter still has its own
                    # deadline — so this is bounded short rather than generously.
                    timeout=REFUSAL_REPORT_TIMEOUT_S,
                )
            except Exception:  # noqa: BLE001 — an unreported refusal is the old behaviour
                logger.debug(
                    "mobility: could not report the refusal of %s to the owner",
                    session_id,
                    exc_info=True,
                )

    threading.Thread(target=_run, name=f"mesh-move-pull-{session_id}", daemon=True).start()
    return {
        "result": "accepted",
        "phase": "prepared",
        "mode": "keep" if keep else "move",
        "new_session_id": new_id if keep else "",
        "session_id": session_id,
    }


# ---------------------------------------------------------------------------
# Lifecycle on a peer: archive / restore / delete run on the OWNER
# ---------------------------------------------------------------------------


def _lifecycle_on_owner(
    server: "RelayServer", link: "PeerLink", frame: dict[str, Any]
) -> dict[str, Any]:
    """Archive/restore/delete a session THIS device owns, at a peer's request.

    ROUTED, NOT REPLICATED (design §8): the guards, the confirmation semantics, the
    retention interaction and the wake-index pruning stay in one place, on the
    device that owns the disk. A second ``rmtree`` of a session directory anywhere
    else in the tree is what ``tests/unit/session/test_no_session_deletion.py``
    exists to prevent.
    """
    from local_operator.session import archived
    from local_operator.session.cleanup import delete_session

    action = str(frame.get("action") or "")
    session_id = str(frame.get("session_id") or "")
    if not session_id:
        # SAME RULE AS THE MOVE OP: a frame that cannot be understood is a protocol
        # error (the family's MeshRefusal), while a refusal it CAN understand - a
        # delete the owner's guards refuse - travels as a ``refused`` document whose
        # code a front end branches on.
        raise _peer_error("bad_request", "a lifecycle request needs a conversation id")
    actor = f"mesh:{link.device_id}"
    if action in ("archive", "unarchive"):
        wanted = bool(frame.get("archived", action == "archive"))
        if not (Path(server.root) / "sessions" / session_id).is_dir():
            return {
                "refused": True,
                "code": "session_lifecycle_refused",
                "message": f"this device does not hold {session_id}",
            }
        changed, evicted = archived.archive_change(server.root, session_id, wanted)
        return {
            "ok": True,
            "action": "archive" if wanted else "unarchive",
            "session_id": session_id,
            "changed": changed,
            "evicted": list(evicted),
            "message": (
                f"{'Archived' if wanted else 'Restored'} {session_id} on this device."
                if changed
                else f"{session_id} was already {'archived' if wanted else 'restored'} here."
            ),
        }
    if action == "delete":
        # ``confirmed`` IS REQUIRED ON THE PEER HOP exactly as it is on the HTTP
        # route (design §8.1): a delete dispatched without it is a DRY RUN — the
        # owner runs its own rehearsal and deletes nothing.
        outcome = delete_session(
            server.root,
            session_id,
            actor=actor,
            dry_run=not bool(frame.get("confirmed")),
        )
        if not outcome.found:
            return {
                "refused": True,
                "code": "session_not_found",
                "message": f"this device does not hold {session_id}",
            }
        if outcome.refusal:
            return {
                "refused": True,
                "code": "session_delete_refused",
                "message": outcome.refusal,
                "session_id": session_id,
            }
        if not bool(frame.get("confirmed")):
            return {
                "ok": True,
                "action": "delete",
                "session_id": session_id,
                "deleted": False,
                "confirmed": False,
                "children": outcome.children,
                "message": outcome.rehearsal(),
            }
        return {
            "ok": True,
            "action": "delete",
            "session_id": session_id,
            "deleted": outcome.deleted,
            "confirmed": True,
            "children": outcome.children,
            "message": f"Deleted {outcome.label} on this device.",
        }
    return {
        "refused": True,
        "code": "session_lifecycle_refused",
        "message": f"this build does not know the action {action!r}",
    }


# ---------------------------------------------------------------------------
# Resolving the owner, and the peer plumbing the local verbs share
# ---------------------------------------------------------------------------


def resolve_remote_owner(server: "RelayServer", session_id: str) -> tuple[str, str]:
    """Which DEVICE holds ``session_id``, from the federated listing.

    Asked of this device's own relay, which is the only process holding links: a
    cold session on a peer is still a row (``local_session_rows``), which is what
    makes "recall the idle session I left on the build box" a thing this can do at
    all. Raises :class:`Moved` when nobody owns it.
    """
    rows = server.federated_rows().get("sessions") or []
    for row in rows:
        if str(row.get("session_id") or "") != session_id:
            continue
        if str(row.get("locality") or "") == "local":
            raise Moved("already_local", "")
        peer = row.get("peer") or {}
        device_id = str(peer.get("device_id") or "")
        if not device_id:
            continue
        if not peer.get("reachable", True):
            raise Moved(
                "unreachable",
                f"{peer.get('name') or device_id} is not answering right now"
                + (f" (last seen {int(peer.get('age_s') or 0)}s ago)" if peer.get("age_s") else "")
                + "; nothing was changed",
            )
        return device_id, str(peer.get("name") or "")
    # Its OWN code, distinct from the per-peer ``unreachable`` above (design
    # review round 1, D1): the two sentences have DIFFERENT remedies — a device
    # that stopped answering may recover on its own, while "no row at all" is
    # answered by looking at THIS device's own staging area — and the sync
    # clause that names a staged copy must compose onto the latter only, where
    # the named verb can actually complete.
    raise Moved(
        "no_holder",
        f"no device in this network holds {session_id}, so nothing was moved",
    )


def _tombstone_recovery_clause(server: "RelayServer", session_id: str) -> str:
    """What to add to a "not this device's to move" refusal when we handed it on.

    The unconditional sentence those refusals used to carry — "run this from the
    device that holds it" — names a device that can hold NOTHING: a destination that
    committed the handoff here and then died before its promote holds the only copy in
    ``network/staging/<id>`` and has no session to show for it. This device is the one
    that tombstoned the id, so it is the one that knows where the id went; the clause
    says that, and says what to run THERE (QA round 1, Q1).
    """
    tombstone = _tombstone(server.root, session_id)
    if not tombstone:
        return ""
    device_id = str(tombstone.get("device_id") or "")
    if not device_id:
        return ""
    name = str(tombstone.get("device_name") or "") or server._member_name(device_id)  # noqa: SLF001
    return (
        f". This device handed {session_id} to {name or device_id}; if it is not there "
        f"yet, its verified copy is in that device's `network/staging/{session_id}`, and "
        f"`lop sessions move {session_id} --to local` THERE adopts it"
    )


def staged_copy_clause(server: "RelayServer", session_id: str) -> str:
    """What to add to a "no device holds this id" refusal when a VERIFIED copy
    is sitting in this device's own staging area (§6.5's last row).

    THE FACT THE REFUSAL USED TO HIDE (measured 2026-09-29, the recall strand):
    after a recall whose promote was refused, the staged copy was the only
    complete copy in existence — and `lop sessions sync <id>` answered "no device
    in this network holds <id>, so nothing was moved", denying bytes this device
    was holding. The clause names the directory and the one verb that adopts it.

    Added ONLY when ``ready.json`` is present (a copy verified against the
    owner's cut), so it can never point at an empty or unverified staging
    directory. Public rather than private because the sync plane's refusal is
    the caller the operator's report is about.
    """
    from local_operator.network import sync as sync_mod

    staging = sync_mod.staging_dir(Path(server.root), session_id)
    if not (staging / "ready.json").is_file():
        return ""
    return (
        f". This device holds a verified copy of it in `network/staging/{session_id}`; "
        f"`lop sessions move {session_id} --to local` here adopts it"
    )


def adopt_staged_if_mine(
    server: "RelayServer", session_id: str
) -> SessionMoveResult | SessionMoveRefusal | None:
    """§6.5 row 5, asked for BY THE USER on the device that holds the staged copy.

    THE GAP THIS CLOSES, measured on the two-device rig 2026-09-27 (QA round 1, Q1):
    with the owner tombstoned and a verified ``network/staging/<id>/ready.json`` here,
    a relay restart adopted nothing and ``move <id> --to local`` answered "no device
    in this network holds <id>" — so the only copy of the conversation was on no
    device's listing and no verb could reach it. The destination's own journal entry
    is written by THIS device's live relay, and ``reconcile`` skips an entry its own
    live instance wrote (the guard that keeps a genuinely in-flight handoff safe from
    a reconcile racing it). That guard is right for a move nobody is driving; when the
    owner has already tombstoned the id there is no move in flight on either side, and
    a user asking this device for that id is the one moment the destination can act.

    The proof is unchanged and is not re-derived here: ``_reconcile_destination`` asks
    the owner named in ``ready.json`` for the id's fate and promotes ONLY on a
    tombstone that names this device.

    THAT CLAUSE IS ENFORCED, AND IT IS WHY THIS FUNCTION CAN RETURN A REFUSAL (review
    round 2, MAJOR): the record's own ``device_id`` is read on this side before anything
    is promoted, and a record naming another device comes back as
    ``action: not_mine`` — answered with the refusal its detail names rather than
    swallowed into the generic "no device in this network holds <id>" the caller would
    otherwise fall back to, which told a user holding the only verified copy of a
    conversation that no device had it.
    """
    from local_operator.network import sync as sync_mod

    staging = sync_mod.staging_dir(server.root, session_id)
    try:
        ready = json.loads((staging / "ready.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    owner = str(ready.get("owner_device") or "")
    if not owner:
        return None
    entry = {"role": "destination", "phase": "handing_off", "from_device": owner}
    try:
        report = _reconcile_destination(server.root, session_id, entry, server=server)
    except Moved:
        return None
    if str(report.get("action") or "") == "not_mine" and isinstance(report.get("refusal"), dict):
        # THE RECORD NAMED SOMEBODY ELSE, and this refusal travels OUT of here: the user
        # asked THIS device for the id, and the fact they need is that the copy here is
        # not this device's to adopt — logged only, it reached them as "no device in
        # this network holds <id>" (review round 2, MAJOR).
        return cast(SessionMoveRefusal, report["refusal"])
    if str(report.get("action") or "") != "promoted":
        # LOGGED, not swallowed: this refusal is the difference between "there is nothing
        # to adopt here" and "there is a verified copy here and this device would not
        # take it", and only the report says which (QA round 2, R1: the rig answered the
        # generic `unreachable` while the copy sat in staging).
        logger.info("mobility: staged copy of %s not adopted: %s", session_id, report)
        return None
    return cast(
        SessionMoveResult,
        {
            "ok": True,
            "session_id": session_id,
            "new_session_id": session_id,
            "mode": "move",
            "from_device": _device_block(server, owner),
            "to_device": _own_block(server),
            "phase": "committed",
            "phases": [
                {"phase": "prepared", "at": time.time()},
                {"phase": "handing_off", "at": time.time()},
                {"phase": "committed", "at": time.time()},
            ],
            "recovered": True,
        },
    )


def _link_for_move(server: "RelayServer", device_id: str, name: str) -> "PeerLink":
    link = server._ensure_link(device_id)  # noqa: SLF001 — the one dial seam
    if link is None:
        label = name or server._member_name(device_id)  # noqa: SLF001
        raise Moved("unreachable", _unreachable(label))
    return link


def _move_phases(server: "RelayServer", session_id: str) -> list[MovePhaseStamp]:
    """What this move went THROUGH: the durable stamps, plus what memory saw.

    A UNION, NOT A PREFERENCE (QA round 1 integration, Q-INT-3). The two sources are
    partial in different directions, and a receipt that took disk whenever disk had
    anything at all reported an offload as ``['committed']`` (0.75) — so a renderer
    drawing progress from this list showed every offload stuck at three quarters,
    while the model documents ``prepared`` 0.25 … ``done`` 1.0 and a recall
    publishes all four:

    * the JOURNAL and the TOMBSTONE survive a restart but forget what is behind a
      completed commit — an offload's journal entry is cleared once the source is
      retired, so all the disk can still prove is the tombstone, and the phases
      before it are gone;
    * ``_Progress`` recorded the transitions the journal cannot prove afterwards
      (see its own docstring: after the commit the journal is gone, and ``committed``
      cannot be re-derived from a file that no longer says it) but it dies with the
      process, so it is the weaker half after a restart rather than in general.

    ORDERED BY THE CONTRACT (``MOVE_RESULT_PHASES``), never by ``at``:
    :func:`_phases_from_disk` synthesises its timestamps when it READS, so a disk
    stamp can carry a later time than a memory stamp for an earlier phase, and
    sorting by time would put ``committed`` after ``done``. Where both have a phase,
    memory's stamp wins — its ``at`` is when the transition happened rather than when
    somebody looked.
    """
    order = {phase: index for index, phase in enumerate(MOVE_RESULT_PHASES)}
    stamps: dict[str, MovePhaseStamp] = {}
    for stamp in _phases_from_disk(server.root, session_id):
        stamps.setdefault(str(stamp.get("phase") or ""), stamp)
    for stamp in progress_for(server).phases(session_id):
        stamps[str(stamp.get("phase") or "")] = stamp
    stamps.pop("", None)
    return sorted(stamps.values(), key=lambda stamp: order.get(str(stamp["phase"]), len(order)))


def _phases_from_disk(root: Path, session_id: str) -> list[MovePhaseStamp]:
    """Rebuild what a restarted relay can still prove about a move, in order."""
    from local_operator.session.placement import handoff_in_flight

    phases: list[MovePhaseStamp] = []
    try:
        entry = handoff_in_flight(root, session_id)
    except Exception:  # noqa: BLE001 — an unreadable journal is the launch guard's refusal
        entry = None
    if entry is not None:
        at = float(entry.get("at") or time.time())
        phases.append({"phase": "prepared", "at": at})
        if str(entry.get("phase") or "") == "handing-off":
            phases.append({"phase": "handing_off", "at": at})
    if _tombstone(root, session_id):
        phases.append({"phase": "committed", "at": time.time()})
    return phases


# ---------------------------------------------------------------------------
# Reconcile (§6.5)
# ---------------------------------------------------------------------------


def reconcile(
    root: Path,
    *,
    server: "RelayServer | None" = None,
    only: str = "",
    own_instance: str = "",
    include_own: bool = False,
) -> list[dict[str, Any]]:
    """Apply §6.5's recovery table to the entries a PREVIOUS process left behind.

    THE INSTANCE RULE, and it is the difference between recovery and sabotage.
    Every entry records the ``instance_id`` of the relay that wrote it, and a
    reconcile SKIPS its own instance's entries. An entry in this file is not
    evidence of a crash: ``prepared`` is the normal state of a move whose copy is
    being made right now, and a reconcile that treated it as a leftover rolled back
    LIVE handoffs — it did, on every ``ready``, until this rule existed. An entry
    from ANOTHER instance is a different animal: that relay is gone (or is not this
    one), so the table applies to it exactly as the design says.

    ``server`` is optional because the DESTINATION branch needs a link to ask the
    owner what happened; with ``server=None`` — which is what a handler for a
    request from that same peer must pass, or it would be asking over the link it
    is serving — only the durable local facts are applied.
    """
    from local_operator.session.placement import read_handoff_journal

    owner_instance = own_instance or (str(server.instance_id) if server is not None else "")
    try:
        entries = read_handoff_journal(root)
    except Exception as exc:  # noqa: BLE001 — the journal's own reader names the file
        logger.warning("mobility: cannot read the handoff journal: %s", exc)
        return [{"session_id": "", "action": "unreadable", "detail": str(exc)}]
    report: list[dict[str, Any]] = []
    for session_id, entry in sorted(entries.items()):
        if only and session_id != only:
            continue
        wrote = str(entry.get("instance_id") or "")
        if not include_own and owner_instance and wrote == owner_instance:
            report.append(
                {"session_id": session_id, "action": "in_flight", "phase": entry.get("phase")}
            )
            continue
        if str(entry.get("role") or "source") == "source":
            report.append(_reconcile_source(root, session_id, entry, server=server))
        else:
            report.append(_reconcile_destination(root, session_id, entry, server=server))
    return report


def _reconcile_source(
    root: Path, session_id: str, entry: dict[str, Any], *, server: "RelayServer | None"
) -> dict[str, Any]:
    """The source's half of the table: roll back at ``prepared``, complete after."""
    from local_operator.session.placement import clear_handoff_entry

    to_device = str(entry.get("to_device") or "")
    directory = Path(root) / "sessions" / session_id
    tombstone = _tombstone(root, session_id)
    if str(entry.get("phase") or "") == "prepared" and not tombstone:
        # NOTHING WAS COMMITTED, so the session is still ours and simply cold. No
        # write to the directory at all: that asymmetry — rollback is free,
        # `handing-off` never rolls back — is what makes the protocol safe under
        # two generals (§6.5).
        clear_handoff_entry(root, session_id)
        return {"session_id": session_id, "action": "rolled_back", "phase": "prepared"}
    if not tombstone:
        # ``handing-off`` and no tombstone: the crash landed between the journal
        # write and the tombstone. COMPLETE it — the phase says the destination was
        # told to expect a commit, so the owner may not change its mind now.
        return _complete_handoff(root, session_id, entry)

    if directory.is_dir():
        removed = _remove_handed_away(root, session_id, to_device)
        if not removed:
            return {"session_id": session_id, "action": "cleanup_pending", "phase": "committed"}
        clear_handoff_entry(root, session_id)
        return {"session_id": session_id, "action": "cleaned", "phase": "committed"}
    clear_handoff_entry(root, session_id)
    return {"session_id": session_id, "action": "cleared", "phase": "committed"}


def _complete_handoff(root: Path, session_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Finish a commit the relay died in the middle of (§6.5 row 3)."""
    from local_operator.network.projection import write_tombstone
    from local_operator.session.placement import (
        HANDOFF_PHASE_HANDING_OFF,
        clear_handoff_entry,
    )

    to_device = str(entry.get("to_device") or "")
    entry = dict(entry)
    entry["phase"] = HANDOFF_PHASE_HANDING_OFF
    from local_operator.session.placement import write_handoff_entry

    write_handoff_entry(root, session_id, entry)
    write_tombstone(
        session_id,
        device_id=to_device,
        device_name=str(entry.get("to_name") or ""),
        network_id=str(entry.get("network_id") or ""),
        config_dir=root,
    )
    removed = _remove_handed_away(root, session_id, to_device)
    if removed:
        clear_handoff_entry(root, session_id)
    return {
        "session_id": session_id,
        "action": "completed" if removed else "cleanup_pending",
        "phase": "committed",
    }


def _reconcile_destination(
    root: Path, session_id: str, entry: dict[str, Any], *, server: "RelayServer | None"
) -> dict[str, Any]:
    """The destination's half: promote what the owner already committed.

    Requires a link to ask the owner, because the tombstone is the only durable
    proof that the handoff may be adopted — so with ``server=None`` (a handler
    serving the very link it would have to ask over) this reports the entry as
    still in flight rather than guessing. The staging directory is exempt from the
    GC while this entry exists: those verified bytes are the only copy of a
    conversation the owner may already have deleted.

    THREE ANSWERS, and the middle one is what makes the two-generals problem
    survivable: the owner's tombstone naming US means promote; the owner still
    holding the id means it rolled back, so this device drops the entry and keeps
    the bytes (a retry resumes them); no answer at all means wait, because both a
    promote and a rollback would be guesses about another device's disk.

    AND A FOURTH, WHICH IS A REFUSAL (review round 2, MAJOR): a record that names
    ANOTHER device is not this device's commit. It is the same answer as "no answer"
    in one respect — nothing here may be promoted — and the opposite in another: the
    fate is decided, so the report says so (``action: not_mine``) and carries the
    refusal the user's verb returns, rather than dropping the entry in silence and
    leaving the caller to answer "no device holds <id>" about bytes this device is
    holding (see ``_tombstone_names_this_device``).
    """
    from local_operator.network import sync as sync_mod
    from local_operator.session.placement import clear_handoff_entry

    staging = sync_mod.staging_dir(root, session_id)
    owner = str(entry.get("from_device") or "")
    if not (staging / "ready.json").is_file():
        # Nothing verified was ever written here, so a rollback is free — and the
        # source's own reconcile reaches the same conclusion from its side.
        #
        # THE ONE CASE WHERE STAGING IS ABSENT AND THE SESSION EXISTS ANYWAY (§6.5
        # row 5, and M-3's second half): a promote that reached ``os.replace`` and
        # died before ``_promote`` could delete its own boot marker leaves
        # ``ready.json`` INSIDE the session directory. That marker is not content
        # (``sync.EXCLUDED_ENTRIES`` says so) and this is the only moment anything
        # still knows it is ours, so it goes here rather than lingering in a
        # conversation the user will open.
        (Path(root) / "sessions" / session_id / "ready.json").unlink(missing_ok=True)
        clear_handoff_entry(root, session_id)
        return {"session_id": session_id, "action": "rolled_back", "phase": "prepared"}
    if server is None:
        return {"session_id": session_id, "action": "waiting", "phase": "handing_off"}
    link = server._ensure_link(owner)  # noqa: SLF001 — the one dial seam
    if link is None:
        return {"session_id": session_id, "action": "waiting", "phase": "handing_off"}
    transport = LinkTransport(server, link, session_id)
    try:
        status = transport.ask(
            {
                "op": "net_session_move",
                "phase": "status",
                "session_id": session_id,
                "to_device": server.identity.device_id,
            }
        )
    except Moved as refusal:
        if refusal.code in ("unreachable", "not_owner"):
            return {"session_id": session_id, "action": "waiting", "phase": "handing_off"}
        raise
    if str(status.get("result") or "") != "tombstone":
        # The owner still holds it, so the move was rolled back (or never
        # committed). The entry goes; the verified bytes STAY, so a retry resumes
        # instead of copying a transcript again.
        clear_handoff_entry(root, session_id)
        return {"session_id": session_id, "action": "rolled_back", "phase": "prepared"}
    tombstone = status.get("tombstone") or {}
    if not _tombstone_names_this_device(server, tombstone):
        # THE RECORD NAMES ANOTHER DEVICE (review round 2, MAJOR). This device holds
        # verified bytes of a handoff that no longer describes it — the owner has
        # handed the same id on — and promoting them would give one id two owners,
        # the older copy winning. Measured on this head 2026-09-27: with the record
        # naming a third device, `move <id> --to local` on the old receiver refuses and
        # leaves the copy in place; and a peer that answers the existence-only way —
        # what this handler did before the fix, and what any build without the check
        # still does — is refused HERE, by this branch (cell 2 of
        # tests/unit/network/test_mobility_promotion_identity.py promotes on the
        # pre-fix code and refuses here).
        clear_handoff_entry(root, session_id)
        return {
            "session_id": session_id,
            "action": "not_mine",
            "phase": "prepared",
            "refusal": _adopt_elsewhere_refusal(server, session_id, tombstone),
        }
    result, refusal, _target = _finish_from_tombstone(server, session_id, owner_device=owner)
    if result is None:
        return {
            "session_id": session_id,
            "action": "waiting",
            "phase": "handing_off",
            "detail": (refusal or {}).get("message", ""),
        }
    return {"session_id": session_id, "action": "promoted", "phase": "committed"}


def sweep_staging(root: Path, *, max_age_s: float = STAGING_MAX_AGE_S) -> list[str]:
    """Age-capped sweep of abandoned staging directories. NEVER touches ``sessions/``.

    ``ready.json`` exempts a directory whatever its age: it is a VERIFIED copy of a
    session the owner may already have deleted, so it is the last copy in existence
    and its age is not evidence that nobody wants it (design §7; §13 Q6 bounds the
    grace and asks the user after it, which this build reports rather than decides).
    """
    import shutil

    root_staging = Path(root) / "network" / "staging"
    swept: list[str] = []
    try:
        children = list(root_staging.iterdir())
    except OSError:
        return swept
    cutoff = time.time() - max_age_s
    for child in children:
        if not child.is_dir():
            continue
        if (child / "ready.json").is_file():
            continue
        try:
            if child.stat().st_mtime >= cutoff:
                continue
            shutil.rmtree(child)
        except OSError:
            logger.debug("mobility: could not sweep %s", child, exc_info=True)
            continue
        swept.append(child.name)
    return swept


# ---------------------------------------------------------------------------
# The relay's handlers
# ---------------------------------------------------------------------------


def _peer_error(code: str, message: str) -> Exception:
    """A MALFORMED-FRAME refusal, as the family's own exception.

    Deliberately NOT the ``result: refused`` document the expected refusals travel
    in: a frame this device cannot make sense of is a protocol error, and the two are
    kept apart because a front end BRANCHES on the refusal documents' codes while a
    refusal frame deliberately carries only a sentence (``wire.refusal_frame``).
    """
    from local_operator.network.types import MeshRefusal

    return MeshRefusal(code, message)


def make_handler(
    server: "RelayServer",
) -> Any:
    """``net_session_move`` and ``net_session_lifecycle`` for one relay.

    Registering the lifecycle op HERE is how the relay's own stale refusal retires
    without editing ``relay.py`` (its module docstring still lists
    ``net_session_lifecycle`` among the not-implemented names; the hook replaces
    the handler, which is what a peer can observe).
    """

    def _move(link: "PeerLink", frame: dict[str, Any]) -> dict[str, Any]:
        phase = str(frame.get("phase") or "")
        session_id = str(frame.get("session_id") or "")
        if not session_id:
            # A MALFORMED FRAME IS A REFUSAL, not a result: the family's own answer,
            # shaped by ``_run_handler``. The expected refusals of this protocol
            # (busy, not_owner, in_progress) travel as DATA inside the ack instead,
            # because a front end branches on their codes and the wire deliberately
            # drops a code from a refusal frame.
            raise _peer_error("bad_request", "a move needs a conversation id")
        if phase == "invite":
            return _destination_invite(server, link, frame)
        if phase == "refused":
            # BEFORE the reconcile below, on purpose: this frame ends a wait rather
            # than touching the journal, so reconciling for it would be recovery work
            # done for a move that was refused.
            return _source_refused(server, link, frame)
        # ONE TARGETED RECONCILE, before anything else: this is where a
        # destination's post-crash retry completes an interrupted commit on this
        # side, and where a stale `prepared` is cleared so the retry can proceed.
        # ``server=None`` on purpose — this handler IS serving the link the
        # reconcile would have to ask over.
        try:
            reconcile(
                server.root, server=None, only=session_id, own_instance=str(server.instance_id)
            )
        except Exception:  # noqa: BLE001 — recovery is best effort; the phase decides
            logger.debug("mobility: reconcile of %s failed", session_id, exc_info=True)
        if phase == "status":
            return _source_status(server, link, frame)
        if phase == "prepare":
            return _source_prepare(server, link, frame)
        if phase == "ready":
            return _source_commit(server, link, frame)
        if phase == "done":
            return _source_done(server, link, frame)
        raise _peer_error("bad_request", f"unknown move phase {phase!r}")

    def _lifecycle(link: "PeerLink", frame: dict[str, Any]) -> dict[str, Any]:
        return _lifecycle_on_owner(server, link, frame)

    return {"net_session_move": _move, "net_session_lifecycle": _lifecycle}


# ---------------------------------------------------------------------------
# The local verbs (control socket)
# ---------------------------------------------------------------------------


def local_move_handler(server: "RelayServer") -> Any:
    """The ``session_move`` local op: recall, offload, recover, reconcile."""

    def _handle(
        frame: dict[str, Any],
    ) -> SessionMoveResult | SessionMoveRefusal | dict[str, Any]:
        action = str(frame.get("action") or "")
        if action == "reconcile":
            swept = sweep_staging(server.root)
            return {
                "ok": True,
                "reconciled": reconcile(server.root, server=server),
                "swept": swept,
            }
        if action == "recover":
            return _recover_from_replica(server, str(frame.get("session_id") or ""))
        session_id = str(frame.get("session_id") or "")
        if not session_id:
            return _move_refusal(session_id, "not_owner", "no conversation was named")
        keep = bool(frame.get("keep"))
        wait_s = float(frame.get("wait_s") or 0.0)
        # THE FLAG IS READ AS ``bool``, so a client older than it sends nothing and an
        # older DESTINATION that never learned the key ignores it: absent reads as
        # False on both hops, which is the default-off contract stated at
        # ``SessionMoveResult``. A non-bool (a string from a hand-rolled client) is
        # not a direction to engage either — the same reading every other optional
        # field on this frame gets.
        engage_on_arrival = bool(frame.get("engage_on_arrival") or False)
        to = str(frame.get("to") or "local")
        try:
            reconcile(server.root, server=server, only=session_id)
        except Exception:  # noqa: BLE001 — recovery is best effort
            logger.debug("mobility: reconcile before a move failed", exc_info=True)
        if to == "local":
            return _recall(
                server, session_id, keep=keep, wait_s=wait_s, engage_on_arrival=engage_on_arrival
            )
        return _offload(
            server,
            session_id,
            to=to,
            keep=keep,
            wait_s=wait_s,
            engage_on_arrival=engage_on_arrival,
        )

    return _handle


def _recall(
    server: "RelayServer",
    session_id: str,
    *,
    keep: bool,
    wait_s: float,
    engage_on_arrival: bool = False,
) -> SessionMoveResult | SessionMoveRefusal:
    """``--to local``: this device is the destination and pulls.

    ``engage_on_arrival`` travels straight through to ``_destination_move`` because
    THIS device is both the requester and the destination on this route — the flag
    asks for an engage here, and nothing has to cross a link to carry it.
    """
    from local_operator.network import sync as sync_mod

    if _owned_here(server, session_id):
        return _move_refusal(
            session_id,
            "already_local",
            f"{session_id} is already on this device, so nothing was moved",
        )
    if not keep:
        tombstone = _tombstone(server.root, session_id)
        if tombstone and str(tombstone.get("device_id") or "") == server.identity.device_id:
            # THIS DEVICE ALREADY WON THE ID (§6.5 row 5). The owner's tombstone is
            # the commit, so there is nothing left to ask it for.
            return _finish_from_ready(
                server, session_id, owner_device=str(tombstone.get("device_id") or "")
            )
    try:
        owner_device, owner_name = resolve_remote_owner(server, session_id)
    except Moved as refusal:
        if refusal.code == "already_local":
            return _move_refusal(
                session_id,
                "already_local",
                f"{session_id} is already on this device, so nothing was moved",
            )
        # BEFORE ANY TALK OF REPLICAS OR OF GIVING UP: this device may hold the ONLY
        # copy — the destination of a handoff whose promote never ran (QA round 1, Q1).
        adopted = adopt_staged_if_mine(server, session_id)
        if adopted is not None:
            return adopted
        # NOTHING SYNCED AND THE OWNER IS GONE: offer the replica (§1.6's last row).
        replica = sync_mod.replica_summary(server.root, session_id)
        if replica:
            age = max(0.0, time.time() - float(replica.get("last_synced_at") or 0.0))
            return _move_refusal(
                session_id,
                "unreachable",
                f"{refusal.message} This device has a copy synced {_age_words(age)}. "
                "Recover it as a new session with `--from-replica`; it may be missing "
                "the last turns.",
                changed=False,
            )
        return _move_refusal(session_id, "unreachable", refusal.message)
    try:
        link = _link_for_move(server, owner_device, owner_name)
    except Moved as refusal:
        return _move_refusal(session_id, "unreachable", refusal.message)
    transport = LinkTransport(server, link, session_id)
    result, refusal, _target = _destination_move(
        server,
        session_id,
        transport=transport,
        keep=keep,
        wait_s=wait_s,
        owner_device=owner_device,
        owner_name=owner_name,
        engage_on_arrival=engage_on_arrival,
    )
    return result or refusal or _move_refusal(session_id, "unreachable", "the move did not finish")


def _finish_from_ready(
    server: "RelayServer", session_id: str, *, owner_device: str
) -> SessionMoveResult | SessionMoveRefusal:
    """A tombstone already names this device: adopt the verified copy (§6.5).

    No peer is asked and none needs to be: the owner's own durable record is the
    whole proof that the handoff may be adopted, which is exactly why this works
    when the owner has since gone away.
    """
    from local_operator.network import sync as sync_mod

    staging = sync_mod.staging_dir(server.root, session_id)
    if (staging / "ready.json").is_file():
        result, refusal, _target = _finish_from_tombstone(
            server, session_id, owner_device=owner_device
        )
        return (
            result
            or refusal
            or _move_refusal(session_id, "unreachable", "the handoff did not finish")
        )
    return _move_refusal(
        session_id,
        "no_replica",
        f"{session_id} was handed to this device, and the copy this device made is gone; "
        "nothing was changed",
        phase_reached="handing_off",
        changed=True,
    )


class _NullTransport:
    """A transport that refuses to ask anything.

    The tombstone path needs NO peer: the owner's own durable record is the whole
    proof that the handoff may be adopted. A class rather than ``None`` so an
    unexpected call fails loudly instead of silently returning nothing.
    """

    link: Any = None

    def ask(self, frame: dict[str, Any]) -> dict[str, Any]:
        raise Moved(
            "unreachable",
            f"this device needs no peer to finish that handoff ({frame.get('phase')!r} was "
            "never asked)",
        )


def _peer_may_not_take(
    link: "PeerLink", session_id: str, *, device_id: str, label: str
) -> SessionMoveRefusal | None:
    """A refusal for a peer this device does not admit to move, or ``None``.

    WHY THE INVITER LOOKS BEFORE IT INVITES (QA delta, Q-D3). Every frame a pull
    sends lands on THIS device and is authorised here against the invited peer's own
    member row (``authorizer.Authorizer.check``), so a peer holding only ``drive``
    has all of them refused here. Before this check that refusal was produced where
    nothing was waiting for it: the invite went out, the peer's first ask was refused
    at t+0.4 s, and the user was told 31.77 s later that the move "did not finish in
    time" with the advice to ask again — advice no retry can satisfy, because a
    capability is a decision and asking again gets the same answer. Measured on two
    real devices, the peer admitted ``drive``.

    ``link.role_capabilities()`` is the SAME row the authoriser decides on, resolved
    from the current member record rather than captured at admission (see
    ``PeerLink.role_capabilities``), so this cannot refuse a move the authoriser would
    have admitted. It is a pre-check and not a second guard: what it removes is the
    WAIT, never the decision, which stays in the authoriser.

    THE COMMAND IN THE REMEDY NAMES THE DEVICE **ID**, and that is not cosmetic (Aida
    round 2, finding 1): ``lop network member grant`` takes an id — ``cli._cmd_member_caps``
    passes ``args.device`` straight to ``set_member_capabilities``, which matches
    ``record.member(device_id)`` — while the name/tail resolver
    (``cli._resolve_device``) is wired into ``credential share`` and not into this verb.
    Measured through the real handler with a peer named ``laptop``: the name form
    answers rc=1 ``laptop is not an active member of …`` and grants nothing, the id form
    answers rc=0. So the advice prints the id and the sentence keeps the NAME as prose,
    which is the one distinction that makes the remedy runnable.
    """
    if "move" in link.role_capabilities():
        return None
    return _move_refusal(
        session_id,
        "not_authorised",
        f"{label} may not take sessions from this device (it does not hold the 'move' "
        f"capability here), so {session_id} was not offered to it. Grant it with "
        f"`lop network member grant <network> {device_id} move` if that was the intent.",
    )


def _offload(
    server: "RelayServer",
    session_id: str,
    *,
    to: str,
    keep: bool,
    wait_s: float,
    engage_on_arrival: bool = False,
) -> SessionMoveResult | SessionMoveRefusal:
    """``--to <peer>``: this device owns the session and asks the peer to pull.

    THE INVITE PHASE, and why it is not a push (§1.1): a push-shaped second path
    would double the copy code and its verification. ``O`` asks ``D`` to "pull this
    from me"; ``D`` acks and runs the SAME destination flow a recall runs. What
    this side then watches is its OWN durable progress — the journal it wrote and
    the tombstone it will write — because asking the destination would be a request
    over the link this device may be serving, and because the phases it must report
    are the ones it can prove.
    """
    from local_operator.session.placement import handoff_in_flight

    if not _owned_here(server, session_id):
        if (Path(server.root) / "sessions" / session_id).is_dir():
            return _move_refusal(
                session_id,
                "not_owner",
                f"{session_id} is not this device's to move; run this from the device that "
                "holds it" + _tombstone_recovery_clause(server, session_id),
            )
        return _move_refusal(
            session_id,
            "third_device",
            f"{session_id} does not live on this device, so it cannot be moved FROM here; "
            "run this from the device that holds it"
            + _tombstone_recovery_clause(server, session_id),
        )
    try:
        target_device, target_name = _resolve_typed_peer(server, to)
    except Moved as refusal:
        return _move_refusal(session_id, refusal.code or "unreachable", refusal.message)
    if target_device == server.identity.device_id:
        return _move_refusal(
            session_id,
            "already_local",
            f"{session_id} is already on this device, so nothing was moved",
        )
    try:
        link = _link_for_move(server, target_device, target_name)
    except Moved as refusal:
        return _move_refusal(session_id, "unreachable", refusal.message)
    progress = progress_for(server)
    progress.forget(session_id)
    # THE PRE-CHECK SITS BELOW THE BOOKKEEPING, not above it (Aida round 2, finding 3):
    # every other exit from this function has already cleared the session's in-memory
    # phases and refusal, and an early return that skipped that would be the one path
    # leaving a stale refusal on a session this device never asked about.
    blocked = _peer_may_not_take(
        link, session_id, device_id=target_device, label=target_name or target_device
    )
    if blocked is not None:
        return blocked
    transport = LinkTransport(server, link, session_id)
    # THE CARRY IS READ FROM THIS DEVICE'S OWN STAMP, and it has to be: the source
    # holds the session, so the source is the only device that can say whether it
    # was created or re-stamped with accepted full-auto authority. It travels with
    # the invite (below) because the step it configures — the destination's runtime
    # construction — runs there. A MISSING key reads as off at the destination: an
    # older source sends none and an older destination ignores it, both default-off.
    from local_operator.session.placement import read_stamp

    _stamp_here = read_stamp(server.root, session_id)
    _unattended = bool(_stamp_here is not None and _stamp_here.unattended)
    try:
        accepted = transport.ask(
            {
                "op": "net_session_move",
                "phase": "invite",
                "session_id": session_id,
                "keep": keep,
                # THE OWNER'S WAIT TRAVELS WITH THE INVITE: the re-probe happens
                # where the busy answer is produced (the destination's loop), so
                # a ``--wait`` that stops at this device is a promise nobody keeps.
                "wait_s": wait_s,
                # ... AND SO DOES THE ENGAGE, for exactly the same reason: the runtime
                # is started by the device that adopts the conversation, which is the
                # one this frame is addressed to. SENT ON EVERY INVITE, never only when
                # true: the key is present whatever its value, and a MISSING key reads as
                # off — the older-peer half of the contract, which is what
                # ``request_move`` sends the same field for on the way in.
                "engage_on_arrival": engage_on_arrival,
                "to_device": target_device,
                # THE SESSION'S AUTO AUTHORITY TRAVELS WITH THE INVITE, for the same
                # reason as the two fields above: the runtime it would be applied at
                # is constructed on the device this frame is addressed to. Sent on
                # every invite so the key's presence does not depend on a boolean.
                "unattended": _unattended,
            }
        )
    except Moved as refusal:
        return _move_refusal(session_id, refusal.code or "unreachable", refusal.message)
    new_id = str(accepted.get("new_session_id") or "")
    budget = move_bound_s(wait_s, keep=keep)
    outcome, refusal = _await_own_progress(server, session_id, budget=budget, invited=target_device)
    if outcome == "done":
        phases = _move_phases(server, session_id)
        # THE HISTORY IS RENDERED, NEVER DRESSED UP (the false-success fix, 2026-09-29).
        # A done outcome with nothing on disk or in memory is an INCONSISTENT state:
        # ``_source_done`` notes ``done`` in memory BEFORE it signals, so the record
        # cannot be empty unless the bookkeeping lost the move. What it must not do is
        # INVENT the missing steps — the old fallback read an EMPTY list as
        # ``prepared`` + ``committed`` and reported instant success for a move that
        # never ran (live 2026-09-29: two stamps 3.8 µs apart under a ``committed``).
        # ``done`` is the one phase the completion signal itself attests; the steps
        # before it are not fabricated.
        if not phases:
            phases = [{"phase": "done", "at": time.time()}]
        if str(phases[-1]["phase"]) not in MOVE_OPENABLE_PHASES:
            phases.append({"phase": "done", "at": time.time()})
        # ``cast`` because the document is assembled field by field and pyright
        # cannot narrow a literal to the contract's TypedDict: the shape is pinned by
        # `test_the_session_move_contract_is_frozen` and parsed by the TUI (slice V).
        return cast(
            SessionMoveResult,
            {
                "ok": True,
                "session_id": session_id,
                "new_session_id": new_id or session_id,
                "mode": "keep" if keep else "move",
                "from_device": _own_block(server),
                "to_device": _device_block(server, target_device, target_name),
                "phase": str(phases[-1]["phase"]),
                "phases": phases,
            },
        )
    phases = _move_phases(server, session_id)
    reached = phases[-1]["phase"] if phases else None
    try:
        entry = handoff_in_flight(server.root, session_id)
    except Exception:  # noqa: BLE001 — an unreadable journal refuses elsewhere
        entry = None
    del entry
    if outcome == "committed":
        # THE COMMIT HAPPENED HERE AND THE HANDOFF DID NOT FINISH (QA round 1, Q1).
        # This device retired its copy on the destination's ``ready`` — the copy was
        # verified, which is what the commit means — and the destination then never
        # confirmed the promote. What the user needs is not a success line but the
        # facts only this device has: the id left here, and the two states a missing
        # confirmation can be, with the one check that tells them apart. The answer is
        # a REFUSAL with ``changed: true`` on purpose: from here the move is unfinished,
        # and a front end that branched on ``ok`` would open a session that may not be
        # on the other device yet.
        #
        # BOTH STATES ARE NAMED, BECAUSE THIS SIDE CANNOT TELL THEM APART (QA round 2,
        # Q1). No ``done`` frame means EITHER that the destination promoted and the frame
        # was lost, OR that it died before promoting and its verified bytes are still in
        # staging. The sentence used to assert the second — "its verified copy is in that
        # device's ``network/staging/<id>``" — in both, which names a directory that is
        # EMPTY after a promote: measured 2026-09-27 on the two-device rig with the
        # promote rename frozen so the frame is lost, staging was 0 bytes, the
        # destination listed the conversation with its transcript, and the verb the
        # sentence named answered ``already_local``. So the reader is handed the check
        # that decides the state — the destination's own listing — and the verb is named
        # only INSIDE the branch it can act on: on the promoted side it is a no-op by the
        # time it can run (the destination adopts the copy on restart and answers
        # ``already_local``).
        return _move_refusal(
            session_id,
            "unconfirmed",
            f"{session_id} was handed to {target_name or target_device} and this device "
            "has retired its copy, but that device never confirmed it as a conversation; "
            f"check that device's listing first. If {session_id} is there, the move "
            "finished and only its confirmation was lost. If it is not there, its "
            f"verified copy is in that device's `network/staging/{session_id}`, and "
            f"`lop sessions move {session_id} --to local` THERE adopts it",
            phase_reached=reached,
            changed=True,
        )
    if refusal is not None:
        # THE DESTINATION REFUSED, so this is a DEFINITE outcome and not a deadline:
        # the sentence is the refusing device's own (this side never saw the guard
        # that fired, and a paraphrase would be a second sentence table for one
        # condition — §8.2), and the wait ended when it arrived rather than at the
        # budget (QA round 1, Q4a). ``changed`` is read from THIS device's own phases
        # by the same rule the deadline path below uses: the refusal is about whether
        # a retry is safe, and only the disk this device holds can answer that.
        code, message = refusal
        return _move_refusal(
            session_id,
            code,
            message,
            phase_reached=reached,
            changed=bool(reached and reached != "prepared"),
        )
    if keep:
        return _move_refusal(
            session_id,
            "deadline_exceeded",
            f"{target_name or target_device} is still copying {session_id}"
            + (f" as {new_id}" if new_id else "")
            + ". Nothing on this device changed, and asking again would make another copy.",
            phase_reached=reached,
            changed=True,
        )
    return _move_refusal(
        session_id,
        "deadline_exceeded",
        f"{target_name or target_device} did not finish taking {session_id} in time. "
        + (
            "This device has already committed the handoff, so ask again rather than "
            "retrying from scratch."
            if reached == "committed"
            else "Nothing was deleted: this device still holds the conversation."
        ),
        phase_reached=reached,
        changed=bool(reached and reached != "prepared"),
    )


def _await_own_progress(
    server: "RelayServer", session_id: str, *, budget: float, invited: str = ""
) -> tuple[str, tuple[str, str] | None]:
    """Wait for this device's own side of an invited move to reach ``done``.

    Watches the DURABLE facts (the journal and the tombstone) rather than asking
    anybody: the invite's whole design is that the destination drives, so the only
    thing the source can honestly report is what its own files say.

    THREE OUTCOMES, AND THE MIDDLE ONE IS WHY THIS RETURNS A WORD (QA round 1, Q1).
    ``"done"`` is the destination's frame that says it PROMOTED, which is the only
    answer that means the conversation is openable there. ``"committed"`` is this
    device having retired its copy and tombstoned the id to a peer that never
    confirmed: the bytes are in that peer's staging and the move is unfinished.
    ``"none"`` is nothing happened here. Answering the second as if it were the first
    is how a receipt came to claim a finished move for a conversation the destination
    never opened — measured on the two-device rig 2026-09-27: with the receiver
    SIGKILLed after its ``ready`` (so the owner commits) the answer was ``phase: done``
    for an id that was then on NO device's listing, with its only copy in staging.

    THE TOMBSTONE NO LONGER ENDS THE WAIT BY ITSELF — it starts the confirmation
    window the bound already carries: ``move_bound_s`` is ``wait_s +
    OFFLOAD_CONFIRM_WAIT_S``, so the last ``OFFLOAD_CONFIRM_WAIT_S`` of the budget IS
    the window for the destination's ``done``, and a wait that returned at the
    tombstone never spent it. A ``done`` frame that is merely LOST therefore costs the
    window rather than the budget, and the healthy path is unchanged: the frame
    arrives in milliseconds and ends the wait immediately.

    THE ONE THING DISK CANNOT SAY is that the destination REFUSED — a refusal writes
    nothing here, so a wait that only watched files sat out its whole budget and then
    reported an unknown outcome for a move that never started (QA round 1, Q4a:
    60 s spent on a refusal that arrived in 3 ms). That device reports it over the
    link (``_source_refused``), and the second half of this return value is that
    answer, noticed within one poll (0.2 s).

    ``invited`` names the device this move was invited to, so a refusal can only end
    the wait it belongs to. A refusal is consulted ONLY while nothing has committed
    here, because a handoff that committed has an answer a late refusal must not
    overwrite.
    """
    deadline = time.monotonic() + max(0.0, budget)
    progress = progress_for(server)
    committed_at: float | None = None
    while True:
        if progress.wait_for(session_id, "done", 0.05 if committed_at else 0.2):
            return "done", None
        now = time.monotonic()
        if committed_at is None:
            if _tombstone(server.root, session_id):
                committed_at = now
            else:
                refusal = progress.refusal(session_id, from_device=invited)
                if refusal is not None:
                    return "none", refusal
                if now >= deadline:
                    return "none", None
                continue
        # Committed, now or earlier: hold the confirmation window, then answer.
        if now >= min(deadline, committed_at + OFFLOAD_CONFIRM_WAIT_S):
            return "committed", None
        if now >= deadline:
            return "committed", None


def _recover_from_replica(
    server: "RelayServer", session_id: str
) -> SessionMoveResult | SessionMoveRefusal:
    """``--from-replica``: a new-id fork of the last synced copy (§1.6, §6.5)."""
    from local_operator.network import sync as sync_mod

    try:
        promoted = sync_mod.promote_replica(server.root, session_id)
    except sync_mod.SyncRefused as refusal:
        return _move_refusal(session_id, refusal.code, refusal.message)
    return cast(
        SessionMoveResult,
        {
            "ok": True,
            "session_id": session_id,
            "new_session_id": promoted["session_id"],
            "mode": "keep",
            "from_device": _device_block(server, str(promoted.get("owner_device") or "")),
            "to_device": _own_block(server),
            "phase": "committed",
            "phases": [
                {"phase": "prepared", "at": time.time()},
                {"phase": "committed", "at": time.time()},
            ],
            "recovered": True,
        },
    )


def _resolve_typed_peer(server: "RelayServer", target: str) -> tuple[str, str]:
    """A peer NAME or id as typed on the command line, resolved to an id.

    ``server._resolve_peer`` is the mesh's own resolver (it refuses an ambiguous
    name rather than guessing), and this maps its ``MeshRefusal`` into this
    module's refusal channel so the CLI has one error shape to render.
    """
    from local_operator.network.types import MeshRefusal

    try:
        device_id = server._resolve_peer(target)  # noqa: SLF001
    except MeshRefusal as refusal:
        raise Moved(refusal.code, refusal.sentence) from refusal
    return device_id, server._member_name(device_id)  # noqa: SLF001


def local_lifecycle_handler(server: "RelayServer") -> Any:
    """The ``session_lifecycle`` local op: archive/restore/delete on a peer."""

    def _handle(frame: dict[str, Any]) -> dict[str, Any]:
        action = str(frame.get("action") or "")
        session_id = str(frame.get("session_id") or "")
        peer = str(frame.get("peer") or "")
        confirmed = bool(frame.get("confirmed"))
        if action not in ("archive", "unarchive", "delete"):
            return {
                "ok": False,
                "code": "unknown_action",
                "message": f"this build does not know the action {action!r}",
                "session_id": session_id,
            }
        try:
            device_id, name = _resolve_typed_peer(server, peer)
        except Moved as refusal:
            return {
                "ok": False,
                "code": refusal.code,
                "message": refusal.message,
                "session_id": session_id,
            }
        try:
            link = _link_for_move(server, device_id, name)
        except Moved as refusal:
            return {
                "ok": False,
                "code": "unreachable",
                "message": refusal.message,
                "session_id": session_id,
            }
        request: dict[str, Any] = {
            "op": "net_session_lifecycle",
            "action": action,
            "session_id": session_id,
            "confirmed": confirmed,
        }
        if action == "archive":
            request["archived"] = True
        elif action == "unarchive":
            request["archived"] = False
        try:
            detail = LinkTransport(server, link, session_id).ask(request)
        except Moved as refusal:
            return {
                "ok": False,
                "code": refusal.code or "unreachable",
                "message": refusal.message,
                "session_id": session_id,
            }
        if detail.get("refused"):
            return {
                "ok": False,
                "code": str(detail.get("code") or "session_lifecycle_refused"),
                "message": str(detail.get("message") or "the owner refused that"),
                "session_id": session_id,
                **{
                    key: value
                    for key, value in detail.items()
                    if key not in ("refused", "code", "message")
                },
            }
        return dict(detail)

    return _handle


# ---------------------------------------------------------------------------
# The CLI's entry points
# ---------------------------------------------------------------------------


def request_move(
    session_id: str,
    *,
    to: str,
    keep: bool = False,
    wait_s: float = 0.0,
    root: Path | None = None,
    from_replica: bool = False,
    engage_on_arrival: bool = False,
) -> SessionMoveResult | SessionMoveRefusal:
    """Move (or with ``keep``, copy) ``session_id`` to the device named by ``to``.

    ``to`` is a peer name or device id, or ``"local"`` to recall a session to this
    device. ``wait_s`` re-polls a busy source every 5 s for up to that long
    (``--wait N``). Returns one of the two contract shapes above and never raises
    for a refusal.

    ``engage_on_arrival`` asks the DESTINATION to start the conversation's runtime as
    soon as the copy lands, so it is live there without a first prompt (the engage is
    a separate act on a move that has already committed; see
    ``_arrival_engagement``). Default off, and it changes nothing when off.

    RUN THROUGH THIS DEVICE'S RELAY, not by opening a peer link here: the relay
    owns the links and is the only process that speaks the mesh, and a second
    implementation of the protocol in a CLI would be a second thing to get wrong.
    A missing relay is a REFUSAL naming the remedy, never a silent no-op.
    """
    from local_operator.network import relay, store

    resolved = Path(root) if root is not None else None
    record = store.find_own_relay(resolved)
    if record is None:
        return _relay_refusal(session_id, "relay_unavailable", _relay_message())
    action = "recover" if from_replica else ("recall" if to == "local" else "offload")
    # THE CALLER'S OWN ENVELOPE IS THE RELAY'S BOUND PLUS ITS MARGIN, and both terms
    # come from ``move_hold_s`` so the two cannot drift: a caller that gives up FIRST
    # does not report a slow move, it reports ``relay_unavailable`` — "this device's
    # relay could not be asked" — while the relay is still working.
    #
    # THE TERM IS PER SHAPE, and the recall is why: a recall copies the transcript with
    # the DESTINATION running the copy, so it is bounded by ``KEEP_COPY_WAIT_S`` (300 s)
    # rather than the offload's 30 s confirmation window. Charging every shape the
    # offload's term made this caller give up at 130 s on a recall that was still
    # copying, and report ``relay_unavailable`` about work in flight.
    #
    # A RECOVERY (``from_replica``) IS NEITHER SHAPE and passes ``to=""``: it promotes
    # bytes already on this disk with no peer in the loop, so it keeps the confirm-sized
    # term it has always had rather than borrowing a copy's.
    hold = move_hold_s(wait_s, keep=bool(keep), to="" if from_replica else str(to))
    timeout = max(60.0, MOVE_OP_DEADLINE_S + hold + MOVE_CONTROL_SLACK_S)
    reply = relay.control_request(
        record,
        "session_move",
        timeout=timeout,
        action=action,
        session_id=session_id,
        to=str(to),
        keep=bool(keep),
        wait_s=float(wait_s or 0.0),
        # SENT ON EVERY REQUEST, never only when true: the destination reads a missing
        # key and a false one the same way (off), and a frame whose shape depends on a
        # boolean is one a future reader has to prove is not load-bearing.
        engage_on_arrival=bool(engage_on_arrival),
    )
    if reply is None:
        return _relay_refusal(session_id, "relay_unavailable", _relay_message())
    detail = reply.get("detail")
    if reply.get("op") != "ack" or not isinstance(detail, dict):
        return _relay_refusal(
            session_id,
            str(reply.get("code") or "relay_refused"),
            str(reply.get("message") or "this device's relay refused the move"),
        )
    if detail.get("ok"):
        return detail  # type: ignore[return-value]
    shaped: SessionMoveRefusal = {
        "ok": False,
        "code": str(detail.get("code") or "relay_refused"),
        "message": str(detail.get("message") or "the move was refused"),
        "session_id": session_id,
        "phase_reached": detail.get("phase_reached"),  # type: ignore[typeddict-item]
        "changed": bool(detail.get("changed", False)),
    }
    return shaped


def _relay_refusal(session_id: str, code: str, message: str) -> SessionMoveRefusal:
    return {
        "ok": False,
        "code": code,
        "message": message,
        "session_id": session_id,
        "phase_reached": None,
        "changed": False,
    }


def _relay_message() -> str:
    from local_operator.network.cli import _relay_unavailable_message

    return _relay_unavailable_message()


def lifecycle(
    session_id: str,
    *,
    action: LifecycleAction,
    peer: str,
    confirmed: bool = False,
    root: Path | None = None,
) -> dict[str, object]:
    """Archive, restore or delete a session that lives on ``peer``.

    Runs the OWNER's own implementation (``archived.archive_change`` /
    ``cleanup.delete_session``) over ``net_session_lifecycle``; a delete without
    ``confirmed`` is the owner's dry run. Returns ``{"ok", "code", "message", ...}``
    in the family's shape.
    """
    from local_operator.network import relay, store

    resolved = Path(root) if root is not None else None
    record = store.find_own_relay(resolved)
    if record is None:
        return {"ok": False, "code": "relay_unavailable", "message": _relay_message()}
    reply = relay.control_request(
        record,
        "session_lifecycle",
        timeout=LIFECYCLE_OP_DEADLINE_S * 3,
        action=action,
        session_id=session_id,
        peer=peer,
        confirmed=bool(confirmed),
    )
    if reply is None:
        return {"ok": False, "code": "relay_unavailable", "message": _relay_message()}
    detail = reply.get("detail")
    if reply.get("op") != "ack" or not isinstance(detail, dict):
        return {
            "ok": False,
            "code": str(reply.get("code") or "relay_refused"),
            "message": str(reply.get("message") or "this device's relay refused that"),
        }
    return dict(detail)


def _age_words(seconds: float) -> str:
    """A duration a person reads, for the one sentence that names an age."""
    if seconds < 90:
        return f"{int(seconds)}s ago"
    if seconds < 5400:
        return f"{int(seconds / 60)} min ago"
    if seconds < 172800:
        return f"{int(seconds / 3600)}h ago"
    return f"{int(seconds / 86400)}d ago"


def install(server: "RelayServer") -> None:
    """Register this slice's peer ops and local verbs on ``server``.

    ``net_session_move`` is SLOW — ``prepare`` retires a runtime and waits for its
    record to go, far past the 10 s inline budget. ``net_session_lifecycle`` is
    slow too: the owner's delete runs an in-use probe that forks ``ps`` and
    ``lsof`` before it removes anything.

    AND IT REGISTERS A START HOOK. Recovery existed and was correct and NOTHING
    CALLED IT: the crash tests passed only because they invoked ``reconcile`` by
    hand, so in the product a stale ``prepared`` entry left by a dead relay blocked
    the owner's own conversation until somebody happened to run another move, and a
    destination that died just after its rename stayed stuck (review round 1,
    M-3). A relay starting on a root a previous relay died in is the one moment
    guaranteed to happen, so the hook runs the table there: ``sweep_staging`` for
    abandoned copies, then ``reconcile`` scoped away from this instance's own
    in-flight entries.
    """
    server.register_ops(
        make_handler(server),
        local_handlers={
            "session_move": local_move_handler(server),
            "session_lifecycle": local_lifecycle_handler(server),
        },
        slow={
            "net_session_move": MOVE_OP_DEADLINE_S,
            "net_session_lifecycle": LIFECYCLE_OP_DEADLINE_S,
        },
        on_start={"mobility-recovery": lambda: recover_on_start(server)},
    )


def recover_on_start(server: "RelayServer") -> list[dict[str, Any]]:
    """``reconcile`` + ``sweep_staging`` for a relay that has just started.

    Runs on the relay's own start-hook thread (never inline in ``start()``): the
    destination branch asks the owner over a link, and a relay whose start-up
    waited for another machine would make every ``lop`` command's cost depend on
    that machine. Nothing here is on the critical path of a move — an entry it
    cannot settle stays in the journal and the next attempt picks it up.
    """
    try:
        swept = sweep_staging(server.root)
        report = reconcile(server.root, server=server)
    except Exception:  # noqa: BLE001 — housekeeping is never worth a dead relay
        logger.debug("mobility: recovery at relay start failed", exc_info=True)
        return []
    if swept or any(str(row.get("action")) not in ("in_flight",) for row in report):
        logger.info("mobility: recovery at relay start: swept=%s %s", swept, report)
    return report
