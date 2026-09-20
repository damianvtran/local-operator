"""Wire and discovery types for the session runtime.

These moved here from :mod:`local_operator.mobile.types` because they were
never about the phone. A record, a heartbeat, a client-kind and an attach cap
describe **one session made reachable over a loopback control socket** — the
phone daemon is one client of that, an attach terminal is another, and wakes
and background automations are the next. Keeping them under ``mobile/`` made
every non-phone consumer import a package named for a front end it does not
use, and made the phone look like the owner of a mechanism it merely borrows.

Stdlib-only and import-light by contract: the runtime publishes its record on the
CLI startup path, so anything imported here is paid by every ``lop``
invocation including ``--version``. ``tests/unit/test_import_graph.py`` pins
that. In particular this module must never reach asyncio, pydantic, or
:mod:`local_operator.session.session`.

``local_operator.mobile.types`` re-exports every name below, so the phone
stack's imports keep working unchanged (§8.3 of the design).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Mapping, Protocol

# The one spelling of the session-directory name, owned by the cleanup policy's
# vocabulary (``retention``) and imported here rather than re-spelled: the
# DURABLE STOP MARKER has to be found from two sides that never meet
# (``control.stop_session`` writes it from a record plus a config root;
# ``attention._classify_orphaned_run`` reads it from the transcript directory),
# and a second copy of the literal is what lets those two drift. Stdlib-only on
# the other side, so it costs this import-light module nothing.
from local_operator.session.retention import SESSIONS_DIRNAME

if TYPE_CHECKING:
    from pathlib import Path

#: Bumped on any breaking change to control frames or web payloads. The
#: runtime and daemon always ship together; the phone UI learns the
#: version in its bootstrap payload and can warn on a stale cached bundle.
#:
#: v2 (attach + reaping) added the ``watch``/``unwatch`` ops, the auth
#: frame's optional ``client`` field, and multi-connection runtimes. The
#: bump is load-bearing for ATTACH specifically: an old (v1) runtime
#: treats any authenticated dial as THE daemon and evicts the real one, so
#: an attach client must refuse to dial a record whose ``protocol`` is < 2
#: rather than silently breaking the owner's phone bridge. The record's
#: version field is the only pre-dial gate — the socket itself speaks the
#: same frame shapes either side of the bump.
#:
#: v4 (full-TUI attach) is ADDITIVE: an attach client's auth frame may carry
#: ``"events": true`` to subscribe to the owner's raw ``AgentEvent`` relay,
#: and the ``recall_steer`` op lets a follower unsend a queued steer. A v3 attach
#: client that omits the flag gets exactly the v3 behaviour (projection
#: frames only), and daemon connections never see the new frames, so the
#: phone path is byte-identical across the bump.
# v5 adds an attach-only canonical frontend state channel. Phone/daemon
# connections still receive projection frames only; the new capability is
# negotiated explicitly by full TUI clients.
#
# It lives HERE rather than in mobile/types.py only because SessionRecord
# defaults ``protocol`` to it: splitting them would make the mobile re-export
# shim import this module and this module import the shim. The move is not a
# bump — the value is unchanged and no frame shape moved with it.
PROTOCOL_VERSION = 5

# Additive attach metadata: a desktop proxy is not itself a person watching.
# Clients must negotiate this before connecting or an old owner would count
# their background socket as a terminal and suppress its fallback notification.
DESKTOP_WATCH_CAPABILITY = "desktop-watch-v1"
DESKTOP_WATCH_LEASE_S = 45.0

#: Additive attach capability: this owner can retire its runtime for a
#: desktop MOVE under an exclusivity fence (``retire_now`` with
#: ``exclusive: true``).
#:
#: WHY A CAPABILITY AND NOT JUST A NEW FIELD. A move changes the directory a
#: successor runtime spawns in, and a successor is engaged by EVERY facade that
#: was attached when the retire landed — each from its OWN ``_cwd``. Two facades
#: with different directories therefore request contradictory successors, and a
#: sibling that does not read the rewritten desktop marker can win that race with
#: the OLD path. Propagating the new target to arbitrary siblings is a broad
#: cross-viewer protocol this release deliberately does not ship, so the bounded
#: answer is to refuse a move while another ACTUAL attach is registered.
#:
#: An old owner ignores the unknown ``exclusive`` field and would retire anyway,
#: so the desktop must never send it without first seeing this string in the
#: owner's record: the capability is what makes the refusal fail-CLOSED on old
#: owners instead of silently unsynchronised. Advertised only by an owner whose
#: handle carries the safe retirement latch (``begin_retire``), because the
#: fence promises a re-check at that latch and a reduced handle cannot honour it.
EXCLUSIVE_MOVE_CAPABILITY = "exclusive-move-v1"

#: Additive attach capability: this owner accepts ``event_mute``/``event_unmute``
#: ops, which stop and resume DELTA-GRADE frames on an attach connection that
#: already subscribed to the raw event relay (``"events": true``).
#:
#: The parking sidebar mints viewer connections it does not paint: every delta
#: those viewers receive is materialised (socket read, JSON decode, event
#: deserialization) and then discarded app-side, so the cheapest correct thing
#: is for the owner not to send them at all while parked. Muting is strictly
#: narrower than the app-side discard it replaces (the same delta-grade types;
#: state-bearing events keep flowing), it is per CONNECTION, and it is
#: reversible: the viewer unmutes on reveal and rebuilds from history plus the
#: canonical live seed, exactly as the app-side drop already requires.
#:
#: Negotiated by capability string rather than a ``PROTOCOL_VERSION`` bump for
#: the reason ``recall_steer`` documents: purely additive op, no frame shape
#: changes, and an owner that does not know the op simply never sees it (the
#: client gates the send on this string being present in the record).
EVENT_MUTE_CAPABILITY = "event-mute-v1"

#: Event types a MUTED attach connection stops receiving: the wire half of
#: ``EVENT_MUTE_CAPABILITY``, and deliberately THE SAME SET the parked
#: ``EventController`` discards app-side (``tui/events.py`` assigns its
#: ``_PARKED_DROP_TYPES`` from this constant, so the two cannot drift).
#:
#: MEMBERSHIP RULE, all three clauses required: the event carries a fragment of
#: something in flight, it leaves the viewport REBUILDABLE after a parked gap —
#: either folded into the owner's ``live_events`` seed, which
#: ``restore_live_projection`` replays (``message_update``), or, for the two
#: types that seed does NOT fold, self-replacing: the first frame after unmute
#: carries the whole accumulated state again (``tool_execution_update``
#: re-sends its full output; ``subagent_progress`` describes the child's
#: current step) — AND it is emitted UNTHROTTLED, once per token or chunk. The
#: third clause is what keeps the set finite, and it is why ``tool_call_compose``
#: is DELIBERATELY EXCLUDED despite satisfying the first two: it carries partial
#: argument bytes of an in-flight call, but the harness already rate-limits it
#: to one per ``COMPOSE_NOTICE_INTERVAL_S`` (0.2 s), so it is not volume traffic
#: and muting it would buy nothing while costing a compose preview on reveal.
#: Do not "complete" this set by adding it.
#:
#: Also deliberately absent: ``message_start``/``message_end`` (row identity and
#: the settled row the dedupe and card pairing key on), every turn/agent
#: boundary, tool start/end, compaction, retry, model change, and every
#: delivery notice — those change state a parked source is still expected to
#: have right. These three ARE the volume: at 12 streaming sessions they were
#: ~229 events/s of the traffic measured on the reporting machine.
EVENT_MUTE_DROP_TYPES = frozenset(
    {
        "message_update",  # one per assistant token
        "tool_execution_update",  # one per streamed tool-output chunk
        "subagent_progress",  # one per child progress beat
    }
)

#: Which side of the owner relationship a control connection speaks for.
#: ``daemon`` (the default when the auth frame omits ``client``) may rebind
#: the owner's conversation; ``attach`` is a follower terminal that may
#: watch and steer but never rebind. Absent-means-daemon keeps an OLD
#: daemon dialing a NEW runtime on the same class it always had.
ClientKind = Literal["daemon", "attach"]

#: Whether the human driving a control connection sits at THIS machine.
#:
#: Some operations are only meaningful where the user is: an OAuth grant opens
#: a browser tab and writes a credential into this machine's ``auth.db``, so
#: running one for a phone would pop a tab nobody is looking at and store a
#: grant the phone's owner cannot use.
#:
#: This cannot be inferred from inside the runtime, which is exactly the bug
#: that produced it: ``/mcp reauth`` refused every routed invocation on the
#: theory that a routed command comes from elsewhere, when the control socket
#: binds ``127.0.0.1`` only and every client is therefore local. So locality is
#: DECLARED by the client in its auth frame and defaults to ``local``: today
#: every dialer reaches the runtime over loopback and is local by construction.
#: A relay that forwards a remote device's commands is the case that must
#: declare ``remote``, and until one exists this stays a one-value union in
#: practice while giving that relay a seam that does not require re-deciding
#: the question at each call site.
ClientLocality = Literal["local", "remote"]

#: How many concurrent attach (viewer terminal) connections one runtime
#: accepts before evicting the least-recently-seen one. Connection close is
#: detected anyway (the reader loop drops the registry entry); the cap is
#: defense against leaked-but-open sockets — half-open TCP with no FIN —
#: which liveness detection cannot see.
ATTACH_MAX_CLIENTS = 4

#: ``SlashResult`` ``data.type`` values that carry an ACTION for the invoking
#: terminal: the owner attached a team or an agent profile, and the invoker is
#: expected to submit ``data["request"]`` as a user turn of its own.
#:
#: This exists because that expectation used to be implicit, and an older
#: viewer that did not hold it dropped the request in total silence: the
#: runtime attached the team, returned "sending to <team>. <manager> is
#: coordinating.", and the pre-#624 renderer printed the line and had no
#: consumer for ``data.request``. No user row, no turn, no error.
#:
#: So a client that renders these DECLARES them in its auth frame's
#: ``slash_consumers``; an undeclared (older) client means the RUNTIME admits
#: the request itself. Absent-means-old, exactly like ``ClientKind`` above.
#: The list is the single source of truth for both sides of that seam, and
#: ``tests/unit/tui/test_noop_consumers.py`` fails CI if a producer emits a
#: ``request``-carrying receipt whose type is missing here.
# Goal submissions follow the same ownership rule: old/mobile clients leave
# admission to the owner; current terminals consume the receipt exactly once.
SLASH_ACTION_RECEIPTS: tuple[str, ...] = ("team_attached", "agent_attached", "goal_set")


def runtime_must_complete(receipt_type: Any, consumers: Any) -> bool:
    """Whether the OWNER must submit this receipt's request itself.

    The one rule, in one place: a client completes a receipt only if it
    DECLARED that type, so ``type not in declared`` — never ``declared is
    None``. Both ``None`` (a client built before the field) and ``[]``
    (declared, consumes nothing) mean undeclared and therefore admit here.

    Extracted because the predicate has two hosts by contract — a session is
    owned either by a detached runtime or by a TUI, and both must answer
    identically. Hand-duplicating it left the two copies free to drift, where
    a drift toward ``declared is None`` silently double-submits on one host
    only (review round 1, NIT-2). Living beside ``SLASH_ACTION_RECEIPTS``
    keeps the rule next to the list it is applied to; this module is
    import-light by contract and this adds no imports.
    """
    if receipt_type not in SLASH_ACTION_RECEIPTS:
        return False
    return receipt_type not in (consumers or ())


# ---------------------------------------------------------------------------
# Discovery record
# ---------------------------------------------------------------------------

#: Directory (under the config root) holding one record per live session.
#:
#: The name is now misleading and is kept anyway, deliberately. It is not a
#: mobile-only directory — every live session publishes here, and `lop
#: sessions`, `lop send`, attach and the daemon all read it. Renaming it would
#: buy nothing but a tidier string and would cost a mixed-version split-brain:
#: during any upgrade window an old binary writes and scans ``run/mobile``
#: while a new one uses the new name, so neither can see the other's sessions
#: — `lop sessions` goes half-blind, `lop send` cannot resolve a peer, and the
#: daemon drops live sessions off the phone. There is no migration that avoids
#: it, because the two binaries genuinely coexist in running processes. The
#: literal is a wire constant; treat it as one.
RUN_DIRNAME = "run/mobile"

#: Directory (under the config root) holding one record per live ``serve``
#: daemon — the rendezvous record that says WHICH install is listening WHERE.
#:
#: A second namespace rather than a second record shape in ``run/mobile``, and
#: the reason is what a record MEANS to the code that reads it. Every reader of
#: ``run/mobile`` — ``lop sessions``, the phone daemon, ``find_runtime_record``
#: — treats each file there as a SESSION, and ``SessionRecord.kind`` is a
#: ``Literal`` those readers pass through unvalidated, so a daemon record
#: dropped in beside them would surface as a phantom session with an empty
#: ``session_id`` and no error anywhere. The dirname is also a WIRE CONSTANT
#: (see above) whose whole point is that one upgrade window has two binaries
#: scanning it; widening what a file there may contain is the one change that
#: constant cannot absorb. A daemon is also not a session: it outlives every
#: session it hosts, holds no control socket a person attaches to, and is
#: found by a different question ("which install is serving?").
#:
#: Keyed ``<pid>.json`` like the session records, because it is read the same
#: way and for the same reason: a pid is a process's uniqueness token, and
#: ``kill -9`` leaves exactly one file behind for the next scan to reap.
SERVE_RUN_DIRNAME = "run/serve"

#: Directory (under the config root) holding the BOOT RECORDS of processes that
#: spawn session runtimes — today the runtime itself, later the supervised
#: session host (design-session-survival §4).
#:
#: A THIRD namespace for the same reason ``run/serve`` is a second one: what a
#: reader of a directory assumes about every file in it decides whether a new
#: kind of file may live there, and ``run/mobile`` is read by everything that
#: lists SESSIONS. A boot record is not a session: it says "this pid existed, on
#: this build, under this parent" for a process that may already be gone, which
#: is the one fact a successor cannot reconstruct from a corpse. Dropped beside
#: session records it would surface as a phantom session with an empty
#: ``session_id``; dropped in ``run/serve`` it would be read as a daemon by
#: every discovery reader that globs that directory.
#:
#: A boot record is NOT a liveness signal and nothing may treat it as one: a
#: record whose process died without exiting cleanly is precisely the evidence
#: the namespace exists to preserve (see ``registry.REAPED_DIRNAME`` for the
#: same rule applied to a session record).
HOST_RUN_DIRNAME = "run/host"

#: The ``-m`` target of a session runtime process — THE SPAWN CONTRACT, and the
#: one thing an external census may match a runtime on.
#:
#: ONE HOME, because the two halves of this contract live in different modules
#: and nothing used to tie them together: three spawners write this string into
#: an argv (``session/runtime/launch.py``, ``mobile/daemon.py``) and the
#: residency sweep matches it (``session/runtime/reclaim.py``, which reads it
#: from here). A drift between them is SILENT and one-directional — the sweep's
#: census matches nothing, every root reads as having no runtimes, and the
#: feature goes inert with no failing test anywhere. It sits here rather than in
#: ``reclaim`` because ``launch`` must not import the sweep to write an argv:
#: ``reclaim`` pulls in ``registry`` and ``viewers``, and this module is the
#: runtime's shared vocabulary with no local imports of its own.
#:
#: Matched as a WHOLE ARGV WORD after a ``-m``, never as a substring, by the
#: census that consumes it: a person running ``grep
#: local_operator.session.runtime.process`` would otherwise be listed as a
#: runtime, which is the single misidentification that could make a sweep signal
#: a stranger.
RUNTIME_MODULE = "local_operator.session.runtime.process"


# SESSIONS_DIRNAME (imported above) is the name of the directory holding one
# directory per conversation, and session_dir() names the join once for the
# stop marker's writer and reader, so neither re-derives the layout.


def session_dir(root: "Path", session_id: str) -> "Path":
    """The conversation directory ``root/sessions/<session_id>``.

    A function rather than a bare join so the writer side of the stop marker
    names the same directory the transcript (and therefore the classifier)
    lives in, without either module re-deriving the layout.
    """
    return root / SESSIONS_DIRNAME / session_id


#: How often a runtime rewrites its record's ``heartbeat_at``. The daemon
#: treats a record as wedged after ``HEARTBEAT_TIMEOUT_S``. A ``serve`` daemon
#: beats at the same interval — one freshness budget for both record kinds, so
#: a reader needs a single rule for "is this alive".
#:
#: THE BEAT IS AUTHORED BY THE RUNTIME'S OWN EVENT LOOP, and that is a limit on
#: what freshness can prove rather than a detail. For ``kind=daemon``/``exec``
#: the runtime shares the workload's loop, so a long turn or a starved
#: scheduler stalls this write while the process is demonstrably working —
#: measured at 105.8 s and 205.8 s gaps against the 45 s timeout below on
#: sessions whose CPU time was advancing. Treat a stale beat as "the owner has
#: not reported and is not answering", never as death and never as proof the
#: workload stopped: see ``registry.classify``, which owns that rule, and the
#: per-surface wording that follows it.
HEARTBEAT_INTERVAL_S = 15.0
HEARTBEAT_TIMEOUT_S = 45.0

#: How long a session runtime with WORK IN FLIGHT may defer its own disposal
#: after a termination signal, before it disposes anyway (the drain in
#: :func:`~local_operator.session.runtime.process._drain_for_signal`).
#:
#: WHY A RUNTIME DEFERS AT ALL: SIGTERM is catchable, so a runtime that receives
#: one can look at what it is doing. Disposing under a running turn is not a
#: shutdown, it is data loss — the turn is aborted mid-tool and the transcript
#: is left with a cut-off — and on 2026-09-14 one broadcast sweep SIGTERM'd 21
#: runtimes within 6 ms and cut 32 turns off that way. The graceful paths were
#: always work-aware (``may_refresh``); the signal path was the gap.
#:
#: WHY THE DEFERRAL IS BOUNDED, and never an unbounded wait: a wedged or runaway
#: runtime must stay killable, and "that process ignores SIGTERM" is a worse
#: failure than losing one turn. On expiry the runtime disposes exactly as it
#: did before this constant existed. SIGKILL, power loss and a crash are outside
#: its reach — nothing catchable happens there — and the durable outcome already
#: reports those honestly.
#:
#: WHY IT LIVES HERE, in a module neither side owns: the runtime child
#: (:mod:`~local_operator.session.runtime.process`) obeys it and the kill ladder
#: (:mod:`~local_operator.session.runtime.control`) must outlast it, and those
#: two modules may not import each other — the ladder runs on the CLI startup
#: path and the child is a ``python -m`` entry point. Same reason
#: ``HEARTBEAT_TIMEOUT_S`` is published here: one number both ends must agree
#: on. ``control.SIGTERM_GRACE_S`` is derived from it rather than typed
#: alongside it, and ``tests/unit/session/runtime/test_signal_drain.py`` pins
#: that the ladder's escalation cannot land inside this window.
SIGNAL_DRAIN_S = 120.0

#: How long a BUILD drain may hold with NO MOVEMENT in the work it is holding
#: for, before it stops waiting and leaves through the signal drain's own
#: bounded exit (``process._drain_for``).
#:
#: WHY THE DRAIN NEEDS ONE AT ALL. ``_drain_for``'s promise is the honest one —
#: no new work is admitted and the work in flight FINISHES — and it deliberately
#: draws no clock of its own, because a bound on time would cut a turn merely for
#: being long. That promise is only as good as the work's willingness to finish,
#: and ``is_busy()`` counts two things that can hold for hours: a gate parked on a
#: user's answer (30 s to 24 h, by design) and a lane parked behind a child
#: process. Nothing in the drain bounds STALENESS of work, and the state it leaves
#: behind is not a slow exit but a permanent one — measured on the reporting host:
#: a runtime latched a stale-build drain at 20:32:29 and two hours later still
#: refused every prompt, reporting ``state=wedged`` with three subagent lanes
#: stalled behind a bash child that had been running 23 minutes, RSS falling
#: 134 MB -> 33 MB at 0.1% CPU. A successor has nothing to take while the
#: predecessor holds the transcript lease, so the session is uncontrollable for
#: exactly as long as the drain holds.
#:
#: WHAT THIS BOUND IS NOT. It is not a clock on the drain, and it does not
#: resurrect the rejected "bound the drain by time": the clock it feeds is reset by
#: every observable sign of movement (``process._work_motion``: the turn's durable
#: footprint, the subagent roster generation, the job rows — status, live output
#: offset and activity — and the spool), so a turn that is stepping, a lane that is
#: reporting, a job that is PRINTING or settling, or a peer message reaching the
#: successor all push it out again however long the hold runs.
#:
#: IT BOUNDS REPORTS OF MOVEMENT, NOT WORK, and that distinction is the whole
#: honesty of the bound (agent review round 1, R1). What the clock sees is what
#: reaches this process: a foreground tool call commits nothing until its result
#: lands, so a long `bash`/build/test step that prints the whole time still looks
#: still — only a BACKGROUNDED job mirrors its output into its row as it arrives.
#: The runtime therefore cannot tell a silently-running step from a hung one, which
#: is why the phrase it publishes says "no movement in 15 min" rather than
#: "stalled", and why the caller's docstring states the residual rather than
#: implying a diagnosis. Only a hold that has gone silent at EVERY one of those
#: layers for the whole bound is cut.
#:
#: CALIBRATION, and the basis is stated because there is no distribution of
#: silent-step lengths to size it from. It must sit well beyond a legitimately slow
#: SILENT step, because a tool call that runs for minutes without emitting anything
#: is normal work rather than a stall: the repo's own boot-bound suite slices
#: measure 171-313 s wall at real fleet depth, and the headless agent tasks on this
#: host (``logs/exec-jobs.jsonl``, the eight rows with both ends) ran 4.5, 5.0, 7.5,
#: 8.3 and 9.2 min with a tail of 44.6. It must also sit well under a human's
#: patience for "my session is locked", because the alternative to firing is what
#: the incident measured: unbounded, and it ended in a wedge nobody could clear.
#: 15 min clears the longest silent step evidenced here, tolerates a silent model
#: stream of the same order, and hands a stuck session over a quarter of an hour
#: after its work stopped rather than never. Force-cutting something that was
#: merely slow is the residual risk, accepted deliberately: it costs the turn in
#: flight, while not firing costs the whole session.
#:
#: HERE, beside ``SIGNAL_DRAIN_S``, for that constant's own reason: the phrase that
#: NAMES this bound is published on the record two front ends read
#: (:data:`LEAVING_FOR_BUILD_OVERDUE`), and a number rendered in one module from a
#: constant in another is one rename away from describing a wait nobody waits.
BUILD_DRAIN_PROGRESS_S = 15 * 60.0


def bound_text(seconds: float) -> str:
    """One bound, as a person reads it: ``2 min``, ``2.5 min``, ``30s``.

    ONE FORMATTER, BECAUSE TWO NEARBY BOUNDS IN TWO UNITS READ AS TWO DIFFERENT
    KINDS OF NUMBER. The receiver's drain bound and the sender's grace are
    120 s and 150 s — genuinely different waits, deliberately 30 s apart — and
    they used to print as ``(up to 2 min)`` and ``waiting up to 150s``, which
    invites the reader to compare a rounded figure against an exact one and
    wonder whether they are the same bound (design round 2, D4, PR #1141).

    Minutes, because both bounds are minutes-scale and both lines are prose
    about a wait a person is sitting in front of; ``:g`` trims the trailing
    zero so the common case stays ``2 min`` rather than ``2.0 min``, while a
    sub-minute bound falls back to whole seconds rather than printing
    ``0.5 min``. Every call site derives its number from the constant, never
    types it: a bound that moves must move in the words too, or the receipt
    lies about the wait it is describing.

    HERE, beside the numbers, for the same reason those are: the runtime's
    record phrase and the ladder's receipts are two front ends' prose about
    ONE constant, and the formatter is how they agree on how to say it.
    """
    if seconds < 60.0:
        return f"{seconds:.0f}s"
    return f"{seconds / 60.0:g} min"


#: What a runtime publishes on its record (``SessionRecord.leaving``) the moment
#: a termination signal arrives and it starts draining, and what it answers a
#: rotation with while that drain runs.
#:
#: WHY A PHRASE AND NOT A BOOL. The field exists to close an INVISIBILITY, and
#: its reader is an operator looking at a fleet, not a parser: ``lop sessions``
#: has no room for a legend, so a ``True`` would be a fact nobody could read —
#: the same reason ``pending`` publishes the word ``approval``. It has to say
#: both halves of what is happening (something signalled it; it is finishing a
#: turn rather than ignoring the signal), because either half alone is wrong:
#: "signalled" reads as wedged, "busy" is indistinguishable from the ordinary
#: spinner the same record already publishes.
#:
#: HERE, in the module the child and the ladder both already import, for the
#: reason ``SIGNAL_DRAIN_S`` is: the runtime writes it and two front ends read
#: it, and a copy of the sentence in each would be three places to drift.
#:
#: WHY THE BOUND IS IN THE SENTENCE. The row that carries this is the one the
#: operator reads most (``lop sessions``' trailing column), and on its own the
#: sentence promises a boundary the 120 s bound can take away: a turn longer
#: than ``SIGNAL_DRAIN_S`` does not reach its boundary, it is cut, and after
#: that the row simply disappears — the honest cause is only visible by
#: reopening the session. ``lop refresh``'s receipt already carried
#: ``(up to 2 min)``; every other surface that shows the drain now says it too,
#: because they all read this one phrase (UX round 2, U9). The bound is
#: rendered from the constant beside it (``bound_text``), never typed, so it
#: cannot drift from the wait it describes.
LEAVING_ON_SIGNAL = f"signalled; leaving when its turn ends (up to {bound_text(SIGNAL_DRAIN_S)})"

#: What a runtime publishes when the departure was forced by THE BUILD ON DISK
#: rather than by a signal: it has committed to leaving and is finishing the
#: turn in flight. The same field, the same readers, a different reason — and
#: the reason has to be the trigger's own words, because "signalled" would be
#: false here and "leaving" alone would not say why.
#:
#: NO BOUND IS NAMED HERE, and that is the one substantive difference from
#: ``LEAVING_ON_SIGNAL``: the build drain waits for this runtime's work and
#: nothing else — the signal path is cut by ``SIGNAL_DRAIN_S`` and says so — and a
#: bound in this sentence would promise a wait nothing imposes on an ordinary
#: handover. It stays the phrase for the ordinary case even after the backstop
#: below exists, because the backstop does not change what the drain promises: it
#: only says how long that promise may go unhonoured before the runtime gives up
#: on it and publishes the OVERDUE phrase instead.
LEAVING_FOR_BUILD = "leaving for the build on disk when its turn ends"

#: What a runtime publishes when its build drain has held with NO MOVEMENT
#: REPORTED for the whole of ``BUILD_DRAIN_PROGRESS_S`` and has therefore stopped
#: waiting (``process._leave_overdue``). The third phrase beside the two above,
#: and the only one that reports a bound already SPENT rather than a wait still
#: running.
#:
#: WHY IT EXISTS SEPARATELY, AND WHY EVERY FRONT END KEYS ON IT.
#: ``LEAVING_FOR_BUILD`` promises that the turn in flight finishes, and for the
#: whole window that promise is the truth — so a runtime that CUTS the turn while
#: still wearing that phrase publishes the opposite of what it is doing, and the
#: one sentence the operator gets at the moment their work is abandoned would be
#: the reassurance people act on. The phrase is the primary carrier of WHICH
#: trigger a departure belongs to (:func:`drain_phrase_for_frame`), so a third
#: sentence is a third trigger to every reader — and a trigger every reader has
#: to be TAUGHT. The four consumers are ``tui.app._DRAIN_NOTICES`` (the notice a
#: viewer paints), ``tui.widgets.info_panel._LEAVING_SHORT`` (the narrow-frame
#: form), ``session.errors._TRIGGER_FOR_LEAVING`` (the refusal's head) and
#: ``cli.LEAVING_COLUMN_WIDTH`` (the fleet column's width);
#: ``tests/unit/session/runtime/test_leaving_vocabulary.py`` pins all four
#: against :data:`PUBLISHED_LEAVING_PHRASES`, so the next phrase cannot be added
#: without them. A phrase with no entry paints the FALLBACK, which for this
#: trigger is a sentence about in-flight work finishing at the instant the
#: runtime gave up on it (design round 1, D1/D2/D3; QA round 1, Q-1).
#:
#: WHAT IT SAYS, AND WHAT IT DELIBERATELY DOES NOT. "no movement (15 min)" is
#: the OBSERVATION the clock made — nothing was reported at any of the layers
#: ``process._work_motion`` reads, over the last 15 min — and not a diagnosis: the
#: runtime cannot tell a step running silently (a foreground tool call commits
#: nothing until its result lands, and only a backgrounded job mirrors its output
#: as it prints) from a step that is hung. That is exactly why the sentence must
#: not say "stalled", which would assert the half the runtime cannot establish
#: (agent review round 1, R1). The bound is rendered from the constant beside it,
#: never typed, and the parenthetical states it the way ``LEAVING_ON_SIGNAL``'s
#: does: the clause in front is what happened, the bracket is how long.
#:
#: WIDTH IS A CONSTRAINT, NOT A PREFERENCE. It is held exactly AT
#: ``cli.LEAVING_COLUMN_WIDTH`` — measured 51, which is the same cell count the
#: signal phrase takes and therefore does not widen the fleet column — so the
#: clause is never the thing ``lop sessions`` silently cuts, and inside the
#: ``/info`` card-67 budget of 53, which is where the wide rung draws. The
#: explicit "reported" half lives on the prose surfaces
#: (``tui.app.OVERDUE_DRAIN_NOTICE`` and the refusal), which are not width-bound.
LEAVING_FOR_BUILD_OVERDUE = (
    f"leaving for the build on disk; no movement ({bound_text(BUILD_DRAIN_PROGRESS_S)})"
)

#: Every phrase a runtime publishes on ``SessionRecord.leaving``, in the one
#: place a test can enumerate them.
#:
#: IT EXISTS SO THE CONSUMERS CANNOT DRIFT. Four front-end tables turn a phrase
#: into words a person reads, and each is keyed BY the phrase (see the note above
#: ``LEAVING_FOR_BUILD_OVERDUE``); a phrase missing from one of them is not a
#: missing entry but a WRONG SENTENCE, because every one of those readers falls
#: back to a sentence about a trigger it does not have. The pin is
#: ``tests/unit/session/runtime/test_leaving_vocabulary.py``, which walks this
#: tuple: adding a phrase here without teaching all four consumers fails there
#: rather than on an operator's screen.
PUBLISHED_LEAVING_PHRASES: tuple[str, ...] = (
    LEAVING_ON_SIGNAL,
    LEAVING_FOR_BUILD,
    LEAVING_FOR_BUILD_OVERDUE,
)

#: The CAUSE token the SIGNAL drain commits with: ``process._drain_for_signal``
#: passes it to ``begin_drain``, ``_drain_for`` re-passes it to the exit rung that
#: finally disposes the runtime, and
#: ``serving.ServingSessionHandle._retiring_refusal`` reads it back to name the
#: departure a refusal is about.
#:
#: IT IS THE ONLY DEPARTURE THAT ACCESSOR NAMES, deliberately. The build arm's
#: own cause token is ``runtime-retired``, which ``/move`` latches too — the same
#: retirement leaves a session whose directory changed, where no build is owed —
#: so naming that cause a build would tell a moved session its loaded build is
#: gone from disk. Which BUILD drain a refusal is about is read off the phrase
#: the drain frame published instead (:func:`drain_phrase_for_frame`), which a
#: build drain carries and a move does not.
#:
#: A CONSTANT RATHER THAN A LITERAL, because the reader is 150 lines away in
#: another module and the failure mode of a rename is silent: the refusal would
#: go on describing the build handover for a signalled runtime, which is exactly
#: the falsehood agent review round 4 (MAJOR-2) filed. It is also a
#: ``incidents.CUT_OFF_CAUSES`` key — the same token classifies the turn this
#: drain could not save — so the spelling is already load-bearing beyond this
#: pair.
SIGNAL_DRAIN_CAUSE = "runtime-shutdown"

#: The CAUSE token the BOUNDED build handover cuts with: ``process._leave_overdue``
#: passes it to ``_clean_exit``, whose journal row is what a successor reads to
#: learn how the turn it finds open ended, and ``incidents.CUT_OFF_CAUSES``
#: renders it as the sentence every surface repeats ("Stopped with an error — …").
#:
#: WHY IT IS NOT ``runtime-retired``. The build drain's own token is true of this
#: exit — the runtime IS leaving so the next engage runs the build on disk — but it
#: is the SAME token an ordinary build handover leaves and the same sentence the
#: fleet already sees for one, so a handover that had to be forced was
#: indistinguishable, in every durable record, from one that waited its turn out
#: (QA round 1, Q-2: `lop sessions --json` returns an empty list because the
#: process leaves ~97 ms after the escalation, so the journal row and the
#: successor's incident are the only places left to look). The bound has to be
#: legible there or nowhere.
#:
#: IT IS NOT A ``DELIBERATE_CUT_OFF_CAUSE``: the user did not ask for this, and
#: the taxonomy's deliberate set exists precisely to keep an involuntary cut from
#: being narrated as a stop.
BUILD_DRAIN_OVERDUE_CAUSE = "runtime-overdue"

#: The ``reason`` a drain frame is announced with, as the producers write it, and
#: neither literal is the one you would guess: ``process._commit_to_leaving``
#: announces its ``label`` as the frame's ``reason`` (``announce(label, …)``),
#: while the longer ``reason`` it also takes (``"leaving after SIGTERM"``,
#: ``"retiring for <newer>"``) is the LOG line and the ``_Drain``'s own label —
#: it is not on the wire at all. So:
#:
#: * ``shutdown-drain`` — ``process._SIGNAL_DRAIN_REASON``; the SIGNAL drain.
#:   Measured across every build of this branch from the work-aware SIGTERM rung
#:   through the fix that added the phrase key: twelve commit ranges, all
#:   announcing ``draining=True`` and passing this label, none of them sending a
#:   ``leaving`` key. That is the population the old fallback mislabelled.
#: * ``stale-build`` — the build handover, from BOTH paths that raise one
#:   (``process._begin_drain`` for the draining one, ``process._refresh_for``
#:   for the idle one, whose frame is not draining and never reaches a reader).
#:   This is also the label a RELEASED build announces, which is why it must keep
#:   the build sentence; ``retiring …`` is the idle rotate op's wording
#:   (``server.announce_retiring`` via ``_retire_if_pristine``, likewise not
#:   draining) and is accepted for a trigger nobody has measured yet, because it
#:   too says a successor is coming.
_SIGNAL_REASON_LABELS = ("shutdown-drain",)
_BUILD_REASON_LABELS = ("stale-build", "retiring")


def leaving_phrase_for_frame(reason: str, to: str = "") -> str:
    """Which trigger a ``retiring`` frame's OWN WORDS establish, or ``""``.

    FOR THE FRAMES THAT CARRY NO ``leaving`` PHRASE, and only for them. The
    phrase is the primary carrier and a runtime that writes it says exactly
    which trigger committed the drain; this answers the same question for the
    runtimes that do not, which is one population with two members and neither
    of them served correctly by the app's old fallback:

    * a runtime older than this branch — a RELEASED build, whose only draining
      announce is the stale-build handover. Its build sentence is true, and this
      returns :data:`LEAVING_FOR_BUILD` for it.
    * a build of THIS BRANCH from the work-aware SIGTERM rung through the commit
      that added the key: those announce ``draining=True`` on BOTH triggers and
      send no phrase, so an app that treated an absent phrase as "the build
      handover" told a signalled runtime it was switching builds — both clauses
      false, and the record saying the opposite at the same moment (design round
      4, D9; agent review round 4, MAJOR-1). Their ``reason`` separates the two
      arms, so this returns the phrase that matches the trigger that committed.

    Anything this cannot place returns ``""``, which the app paints with its
    neutral sentence: a trigger nobody has established must not inherit another
    trigger's copy. Two readers, both fed by :func:`drain_phrase_for_frame`:
    ``AttachedSession._on_retiring_frame`` paints from the phrase, and
    ``AttachClient`` remembers it for the refusals a runtime too old to name its
    own departure hands back (agent review round 5, MINOR-1).
    """
    words = (reason or "").strip()
    if words.startswith(_SIGNAL_REASON_LABELS):
        return LEAVING_ON_SIGNAL
    if to or words.startswith(_BUILD_REASON_LABELS):
        # ``to`` is the second, independent corroboration: a build handover
        # normally names the successor it is leaving for, and the signal path
        # passes none (``_drain_for_signal`` has no successor to name).
        return LEAVING_FOR_BUILD
    return ""


def drain_phrase_for_frame(frame: Mapping[str, Any]) -> str:
    """This departure's own words, for a reader that has to speak about it.

    THE ONE ANSWER TO WHICH TRIGGER COMMITTED A DRAIN, called from both ends that
    read a ``retiring`` frame: the host painting its notice
    (``AttachedSession._on_retiring_frame``) and the attach client, which must
    remember which departure refused a message it is about to hand back as a
    typed refusal (``AttachClient._raise_for_reply_error`` — a runtime built
    before the refusal's own ``error_trigger`` field cannot say it there). Two
    copies of the precedence below would be two answers to one question.

    THE PHRASE IS THE PRIMARY CARRIER: a runtime that writes ``leaving`` said
    exactly which trigger committed the drain, and only the runtimes that do not
    send it are read off their ``reason``/``to`` (:func:`leaving_phrase_for_frame`).
    ``""`` is an answer rather than a failure — a frame that named no trigger at
    all — and never another trigger's default.
    """
    return str(frame.get("leaving") or "") or leaving_phrase_for_frame(
        str(frame.get("reason") or ""), str(frame.get("to") or "")
    )


class DiscoveryRecord(Protocol):
    """The members the shared publication path actually touches.

    :mod:`local_operator.session.runtime.registry` is the ONE implementation of
    a staged write at 0600 under a 0700 directory, and both record kinds use
    it: a session record and a serve record differ in their FIELDS, not in how
    they are written, read back, or classified. The shared code reads exactly
    three members — the ``pid`` that keys the file and decides liveness, the
    ``heartbeat_at`` that decides wedged-ness, and ``to_json`` for the payload
    — so that, and no more, is the contract.

    Structural rather than inherited, deliberately. The two record types live
    in different namespaces (``SessionRecord`` here, ``ServeRecord`` in
    ``local_operator.server.registry``) and the server module must be able to
    define its own record without this one importing it — a base class would
    make this startup-path module reach into the daemon's, in one direction or
    the other. A record type that answers these three members is publishable
    and scannable without the registry knowing anything about it.

    The deserializer is NOT part of this contract: it is a callable the caller
    passes (:func:`local_operator.session.runtime.registry.scan`), because a
    protocol describes an instance while parsing happens before one exists.
    """

    pid: int
    heartbeat_at: float

    def to_json(self) -> dict[str, Any]: ...


#: Subagent roster statuses that count as a RUNNING trajectory — one agent loop
#: that can independently make model calls right now.
#:
#: Lives here, beside the record fields it defines the meaning of, because two
#: modules must agree on it and a divergence is invisible: the runtime publishes
#: ``subagents_running`` from this set (``serving.ServingSessionHandle``) and
#: ``/info`` tallies this session's own tree from it (``info.collect``). If they
#: drifted, the fleet total and the tree drawn directly beneath it on the same
#: card would disagree, which is the one error that section must never make.
#:
#: ``queued`` is excluded and counted separately: a delegated child waiting for
#: a capacity slot is not spending anything. This module stays stdlib-only, so
#: the constant costs nothing to import on the CLI startup path.
RUNNING_SUBAGENT_STATUSES = frozenset({"running", "starting", "pausing"})


@dataclass
class SessionRecord:
    """The discovery record one ``lop`` process publishes for one session.

    Lives at ``~/.local-operator/run/mobile/<pid>.json`` — keyed by pid
    because a process hosts exactly one interactive session at a time, so the
    pid is the natural uniqueness token and ``kill -9`` leaves exactly one
    stale file to reap.

    ``control_key`` is the whole authorization story of the control socket:
    the record is mode 0600 under a 0700 directory, so anything that can read
    the key is already the owning account. The daemon never transmits it
    further — the phone never learns it.
    """

    pid: int
    kind: Literal["tui", "exec", "daemon"]
    session_id: str
    conversation_name: str
    cwd: str
    model_label: str
    control_port: int
    control_key: str
    protocol: int = PROTOCOL_VERSION
    started_at: float = field(default_factory=time.time)
    heartbeat_at: float = field(default_factory=time.time)
    capabilities: list[str] = field(default_factory=list)

    # -- live state ---------------------------------------------------------
    # Purely ADDITIVE, and PROTOCOL_VERSION deliberately does NOT move for
    # them. Nothing is required to read these: an older reader drops unknown
    # keys in ``from_json`` and behaves exactly as it did, and a newer reader
    # sees the dataclass defaults for a record an older runtime wrote. Bumping
    # the protocol would instead make every older peer refuse a record it can
    # in fact use — the compatibility cost of a field nobody has to read is
    # zero, and the version is the one thing that would make it non-zero.

    #: A turn is running right now. The picker's liveness marker, and the
    #: difference between a session that is working and one merely resident.
    busy: bool = False
    #: This session has run at least one REAL turn (a user prompt, a wake
    #: delivery, a resume catch-up — anything through ``_run_turn_pipeline``).
    #: ``False`` marks the window after ``/new`` when the record is already
    #: published but the owner is still composing their first prompt, so a
    #: peer broadcast or an exact-address wake/steer must not drive a turn
    #: into it. One-way per session identity: once a real turn has run it
    #: stays True. Two paths other than a turn set it — a TUI ``/new``
    #: rebind re-seeds it for the NEW identity (see
    #: ``RuntimeServer.reset_record_started``), and a boot that RESUMED a
    #: history-bearing conversation seeds True at record construction (see
    #: ``RuntimeServer.__init__``) so the idle session is peer-visible
    #: before any turn runs in the new process. Every heartbeat/republish
    #: carries it forward so it is never reset by a later write.
    #: Deserialization overrides the default for pre-field records —
    #: see ``from_json``.
    started: bool = False
    #: No front end is attached. A working session with nobody watching is
    #: exactly what this release makes possible, so it is worth naming.
    detached: bool = False
    #: This session is WAITING FOR A PERSON: ``"approval"``, ``"ask"``, or
    #: None. A parked gate holds the runtime resident for up to a day, so the
    #: cost has to be findable — this field is what puts it in `lop sessions`
    #: and sorts it first in the picker.
    pending: str | None = None
    #: This runtime HAS COMMITTED TO LEAVING and is finishing work in flight
    #: first: a short phrase (``LEAVING_ON_SIGNAL``) while that drain runs,
    #: ``""`` when it is going nowhere. Set by the signal drain
    #: (``process._drain_for_signal``) and never cleared, because a drain always
    #: ends in an exit.
    #:
    #: WHY IT IS ON THE RECORD. The drain is bounded by ``SIGNAL_DRAIN_S``, so a
    #: signalled-but-working runtime stays alive — and, before this field, stayed
    #: ORDINARY — for up to two minutes. Every surface an operator reads
    #: (``lop sessions``, a picker, a peer's ``lop stop``) saw an unremarkable
    #: ``live`` row throughout, so the honest reading of that window was
    #: impossible and the natural remedy was destructive: a plain stop against a
    #: draining session cuts the very turn the drain exists to save (U1/U2, PR
    #: #1141). ``busy`` cannot carry it — that is the picker's spinner bit and is
    #: ``None`` of the fact that a signal has already been received and acted on.
    leaving: str = ""

    # -- build stamp --------------------------------------------------------
    # Same additive contract as the live-state block above, and for the same
    # reason: PROTOCOL_VERSION deliberately does not move for a field nobody
    # is required to read.
    #
    # The record IS the version channel between a viewer and a runtime. An
    # attach client reads it before dialing (``find_runtime_record``) and holds
    # it at bind, so one comparison there is complete — a runtime's build
    # cannot change while the process lives.

    #: What build this runtime is running (``update.installed_version()``).
    #: ``""`` means a runtime older than this field, which by construction is
    #: older than any terminal that can read it. A viewer compares this with
    #: its own build to NAME skew rather than fail silently under it.
    version: str = ""
    #: The git ref of that install when ``lop-update`` recorded one; ``""``
    #: for PyPI/pipx/editable installs. Needed because same-version rebuilds
    #: are this host's common drift — see ``update.BuildStamp``.
    source_ref: str = ""
    #: The install ROOT this runtime imports from (``update.process_install_root``)
    #: — under the generation layout, the one generation this process belongs
    #: to. It is here for pruning: a generation named by a live record is never
    #: deleted, and the record is the only place a running runtime's tree is
    #: written down (``lop install prune`` reads exactly this). ``""`` for a
    #: runtime older than the field, which pruning reads as "no objection" and
    #: therefore keeps MORE trees rather than fewer.
    install_root: str = ""

    # -- agent trajectories -------------------------------------------------
    # Same additive contract as the live-state and build-stamp blocks above,
    # and for the same reason: PROTOCOL_VERSION deliberately does NOT move.
    # It gates SOCKET FRAME compatibility and is read as a pre-dial CAPABILITY
    # ASSERTION by peers already running — ``attach_client`` refuses below 2,
    # ``session_factory`` and the TUI's takeover path below 4, ``attached``'s
    # canonical attach below 5. Those readers take a HIGHER number as a promise
    # that every frame through v5 is understood. Two JSON integers that touch
    # no frame do not make that promise different, so bumping would spend the
    # one number that carries it on a field nobody has to read, and would leave
    # nothing to distinguish a build that genuinely changed the frames. A
    # future reader that must REQUIRE these fields negotiates through
    # ``capabilities``, which is the seam for exactly that (see
    # ``FRONTEND_CAPABILITY``, gated alongside the version rather than by it).
    #
    # ``None`` (not 0) is the "this build does not report" signal, and the
    # distinction is load-bearing: a runtime predating these fields has not
    # told us it has no subagents, and a reader that collapsed the two would
    # publish a confident total that is silently missing terms. See
    # ``SessionsInfo.subagents_unreported``.

    #: Subagent trajectories spending tokens right now: roster entries whose
    #: status is running/starting/pausing. ``SubagentComms.nodes()`` returns
    #: the COMPLETE roster including nested descendants (nested launches land
    #: in the root session's single records map tagged with their true
    #: parent), so this is a flat count over one read and must never be summed
    #: with a recursive child walk.
    subagents_running: int | None = None
    #: Delegated but still waiting for a capacity slot. Kept separate from
    #: ``subagents_running`` because a queued child is not spending anything,
    #: and folding it in would inflate "what is running right now".
    subagents_queued: int | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(data: dict[str, Any]) -> "SessionRecord":
        # Tolerate unknown keys (a NEWER binary's record read by an older
        # daemon mid-upgrade): forward-compat here is what lets a restart
        # rolling-upgrade the daemon without the phone losing sessions.
        #
        # ``started`` is the one field whose ABSENT key carries meaning: this
        # binary always serializes it (``to_json`` is ``asdict``), so a record
        # without the key was written by a PRE-field binary — and a pre-field
        # session had no composer gate at all, so to it "started" can only
        # mean True. Reading absent-as-True restores old-peer behaviour
        # exactly: broadcasts still reach a working old runtime (it keeps
        # heartbeating its key-less record until it restarts, which for
        # daemon/cmux sessions is days, not an upgrade window), and an exact
        # send dials it rather than spooling into an inbox only a boot-time
        # drain ever reads. The honest cost is the mirror image: an OLD
        # binary's fresh ``/new`` composer stays wakeable — but that is
        # precisely the old binary's own behaviour, unfixable from here until
        # it restarts, so True is the only default that does not penalise the
        # working old sessions for a bug they do not have. The dataclass
        # default stays ``False`` because a CONSTRUCTED record is a fresh
        # this-binary session (the composer window the field exists for);
        # only a deserialized key-less record is assumed pre-field.
        known = {f for f in SessionRecord.__dataclass_fields__}
        fields = {k: v for k, v in data.items() if k in known}
        if "started" not in data:
            fields["started"] = True
        return SessionRecord(**fields)
