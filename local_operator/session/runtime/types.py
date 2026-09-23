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

#: Additive attach capability: this owner accepts the ``operator_challenge`` op
#: and will admit an authority-increasing frame that carries a valid
#: ``operator_sig`` (with ``operator_key_id``, and ``operator_cert`` for a
#: device) against the operator anchor it has pinned.
#:
#: WHY A CAPABILITY AND NOT A PROTOCOL BUMP (revision 2, §2.3). The whole change
#: is ADDITIVE: a new ordinary op that grants nothing, and three optional fields
#: on frames that already exist. An older owner answers the new op with its
#: generic unknown-op error frame, which the client reads as "this runtime
#: predates the feature" and handles by... not being able to loosen, which is
#: exactly what that runtime could do before. Bumping ``PROTOCOL_VERSION`` would
#: instead refuse the CONNECTION, breaking ordinary control (a phone could not
#: even read a session) for a capability it can live without — the OPPOSITE of
#: what this revision is for.
#:
#: Advertised by every runtime that can verify a signature, which is every
#: runtime of this build: verification needs only the anchor's public half, so an
#: owner with no anchor installed is still a correct answer to "can you check
#: one" (it checks and refuses). See the record's capability list for why it is
#: not conditioned on the anchor's presence.
OPERATOR_SIGNATURE_CAPABILITY = "operator-signature-v1"

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
#: ``reasoning_delta`` is the one member that satisfies clauses one and three
#: while being UNREBUILDABLE, and it is admitted on an explicit exception that is
#: narrower than the rule reads: reasoning is display-only and never durable
#: (``harness/types.ReasoningDeltaEvent``), so there is nothing downstream that a
#: gap can leave WRONG — a revealed viewer simply sees the thinking from the
#: reveal onward. Everything else in this module's contract is state a parked
#: source must still have right; this is not state. It is admitted because it is
#: emitted once per reasoning token, which makes it the largest single frame
#: family a long-thinking turn produces: leaving it out is what would let a
#: parked viewer's queue grow without bound, the exact cost this set exists to
#: avoid.
#:
#: Also deliberately absent: ``message_start``/``message_end`` (row identity and
#: the settled row the dedupe and card pairing key on), every turn/agent
#: boundary, tool start/end, compaction, retry, model change, and every
#: delivery notice — those change state a parked source is still expected to
#: have right. These four ARE the volume: at 12 streaming sessions they were
#: ~229 events/s of the traffic measured on the reporting machine, before
#: ``reasoning_delta`` joined them.
EVENT_MUTE_DROP_TYPES = frozenset(
    {
        "message_update",  # one per assistant token
        "reasoning_delta",  # one per reasoning token; see the exception above
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
#: stream of the same order, and reports a stuck session a quarter of an hour after
#: its work stopped rather than never.
#:
#: WHAT THE BOUND DOES NOW, since the obvious reading of the paragraph above is the
#: one that was removed: it does NOT cut the turn. Force-cutting something that was
#: merely slow was the accepted residual risk — it cost the turn in flight — and the
#: operator's rule for every build move is that a runtime is replaced when its turn
#: is COMPLETE, never on a heuristic of inactivity. So the bound now ABANDONS the
#: handover: the runtime keeps the build it loaded, the messages the drain queued
#: come back in, and the failure is published under ``UPDATE_FAILED_CAUSE``
#: (``process._abandon_move``). What the bound still buys is unchanged and is what
#: its calibration was for: a session whose work has gone silent stops being a
#: handover nobody can complete, and becomes a reported condition on a runtime that
#: is still serving.
#:
#: HERE, beside ``SIGNAL_DRAIN_S``, for that constant's own reason: the phrase that
#: NAMES this bound is published on the record two front ends read
#: (:data:`LEAVING_FOR_BUILD_OVERDUE`), and a number rendered in one module from a
#: constant in another is one rename away from describing a wait nobody waits.
BUILD_DRAIN_PROGRESS_S = 15 * 60.0

#: How long a BUILD drain may hold the handover at all — the DWELL bound — before
#: the runtime stops waiting and gives the handover up (``process._drain_for``).
#:
#: WHY A SECOND BOUND, WHEN THE ONE ABOVE EXISTS. :data:`BUILD_DRAIN_PROGRESS_S`
#: bounds STALENESS, and the clock that feeds it is reset by every observable sign
#: that the work advanced (``process._work_motion``). That is exactly right for a
#: hold that has gone quiet, and it is blind by construction to the shape that
#: wedged this host: a hold whose work KEEPS MOVING. A subagent lane that steps,
#: a job that keeps printing, or a parent whose transcript keeps gaining rows resets
#: the staleness clock forever, so the latched drain never expires — and while it is
#: latched nothing else in the process ends it: a draining runtime REFUSES admissions,
#: it holds the transcript lease (``session/runtime/launch.py`` forbids a successor
#: while a live pid holds it), so the session is unwritable and unhandover-able for
#: as long as the lane keeps stepping. Measured on the reporting host: a runtime
#: latched a stale-build drain and held it for EIGHT HOURS while its subagents kept
#: stepping, and the staleness bound only fired once the lanes finally stopped. The
#: dwell bounds the HOLD itself, which is the only thing left to bound once movement
#: is no longer a signal of "this will finish soon".
#:
#: WHY IT IS GENEROUS, AND WHY IT DOES NOT CUT ANYTHING. 30 min, which is D7's own
#: proposed ceiling for this state (``BUILD_DRAIN_MAX_S``, default 1800 s, in
#: ``docs/design-ownerless-session-attach.md``) adopted for a different arm. The
#: length is not free to choose from below: the bound above is 15 min and the dwell
#: must sit strictly ABOVE it, or the staleness arm could never fire and this one
#: would become the only clock — abandoning holds that were merely silent at the
#: 15-minute mark, which reports the weaker observation for the sharper state. It
#: reads no work at all, so the residual it must tolerate is the opposite of the
#: staleness arm's: a handover whose work is genuinely progressing and merely long.
#: The measured long silent steps (up to 44.6 min in ``logs/exec-jobs.jsonl``) are
#: NOT this arm's problem — a silent step is the arm above's, and it fires at 15 min
#: — while a handover that is still REPORTING after half an hour is one an operator
#: should be told about rather than made to wait out.
#:
#: WHAT FIRING COSTS, which is what makes a bound this short defensible where a
#: force-cut was not. Firing in a COMPLETE state still happens: the ticks below keep
#: asking (``process._reaper`` keeps the drain object and its commitment), so the
#: departure lands at the first idle instant and the newer build still gets the
#: handover. Firing while work continues releases the latch, so the session takes
#: work again, and publishes the failure under :data:`UPDATE_FAILED_CAUSE` so the
#: state is reportable instead of silent. It does NOT exit the process and does NOT
#: cut the turn in flight: a runtime is replaced when its turn is COMPLETE, never on
#: a heuristic of inactivity (the operator's rule for every build move), and it is
#: the reason the arm this bound drives abandons the handover rather than taking the
#: signal drain's bounded exit.
#:
#: HERE, beside the two bounds it is measured against, for their own reason: an
#: operator comparing what a runtime promises against what a bound can take away has
#: to read all three in one place, and the failure this bound publishes carries its
#: number onto the incident row from this constant rather than from a copy.
BUILD_DRAIN_DWELL_S = 30 * 60.0


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

#: What a runtime USED TO publish when its build drain had held with NO MOVEMENT
#: REPORTED for the whole of ``BUILD_DRAIN_PROGRESS_S``. NO ARM SENDS IT ANY MORE
#: (``process._abandon_move`` abandons the handover instead of leaving, and keeps
#: :data:`LEAVING_FOR_BUILD` on the record), and it stays in this module and in all
#: four consumer tables for the reason a published vocabulary always outlives its
#: publisher: the records that carry it are on disk, and this build still has to
#: RENDER them. So it remains a member of :data:`PUBLISHED_LEAVING_PHRASES` — which
#: is also what keeps ``cli.LEAVING_COLUMN_WIDTH`` sized for a row a reader may
#: still meet — and what changed is only that nothing advertises it as a state this
#: runtime can reach.
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
#: This phrase is the widest of the three, which is why the historical entry above
#: keeps the column where it is rather than letting a re-derivation shrink it.
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
#: THE PIN IS BY PHRASE, AND ONE STATE IS NOT A PHRASE — the abandoned handover, whose
#: record keeps :data:`LEAVING_FOR_BUILD` and is told apart by the SECOND half of the
#: pair (``SessionRecord.update_failed``). No phrase is added for it, so nothing here
#: changes; ``tui.app.DRAIN_NOTICE_ABANDONED`` is keyed on the pair beside the phrase
#: tables rather than inside them, and its reader is ``tui.app.drain_notice_for``.
PUBLISHED_LEAVING_PHRASES: tuple[str, ...] = (
    LEAVING_ON_SIGNAL,
    LEAVING_FOR_BUILD,
    LEAVING_FOR_BUILD_OVERDUE,
)

#: The three phases of an UPDATE WINDOW, as stable tokens.
#:
#: WHAT A WINDOW IS. An IDLE runtime that decides to move to the build on disk
#: (``process._refresh_for``) publishes one of these, holds them for the ~1 s the
#: handover takes, and QUEUES admissions for its successor instead of refusing
#: them. The incident this answers (2026-09-19): a fleet on a superseded build
#: refusing the operator's own messages with "This session is leaving; it will not
#: start a new turn. Your message is back in the composer — send it again once the
#: session is running again", recoverable only with ``/stop`` + ``/resume``. The
#: runtime was IDLE, so the refusal protected nothing: the successor would have run
#: the message had anyone held it.
#:
#: WHY TOKENS AND NOT SENTENCES, which is the one structural difference from
#: :data:`PUBLISHED_LEAVING_PHRASES`. Every updating sentence carries a BUILD PAIR
#: ("0.59.9 → 0.59.11@ead71b6"), so a consumer table keyed by the full sentence —
#: the shape the leaving vocabulary uses and pins — could not be written at all:
#: the key would be different for every pair a host ever runs. So the tokens are
#: the enumerable, test-keyed vocabulary, and the two renderers below are the only
#: places a sentence is composed from them (``update_phrase`` for prose surfaces,
#: ``update_short`` for the fleet cell), which is what keeps the copy from drifting
#: now that it cannot be keyed.
UPDATING = "updating"
UPDATING_DONE = "updated"
UPDATE_FAILED = "update-failed"

#: Every phase of an update window, in the one place a test can enumerate them.
#: The pin is ``tests/unit/session/runtime/test_updating_vocabulary.py``, which
#: walks this tuple against every consumer table: a phase with no entry there
#: paints that surface's FALLBACK, which is a sentence about a different phase —
#: for the failed phase, the reassurance that the update is still coming.
PUBLISHED_UPDATE_PHASES: tuple[str, ...] = (UPDATING, UPDATING_DONE, UPDATE_FAILED)

#: The CAUSE token a window that ran out of its bound is recorded under: the
#: runtime stays on the build it is running, releases the admission lock, and
#: journals this token so the failure is reportable rather than silent.
#:
#: SAME CLASS AS :data:`BUILD_DRAIN_OVERDUE_CAUSE`, and for its reason: an
#: ``incidents.CUT_OFF_CAUSES`` key, so every surface that repeats a cause can
#: render it as a sentence. It is deliberately NOT ``runtime-retired``, which is
#: what a handover that SUCCEEDED records — a failed update narrated as an
#: ordinary retirement is the failure mode QA round 1 (Q-2) measured for the
#: overdue bound.
#:
#: IT IS NOT A ``DELIBERATE_CUT_OFF_CAUSE`` either: nobody asked for the update to
#: fail, and the deliberate set exists to keep an involuntary event from being
#: narrated as a user's own stop.
UPDATE_FAILED_CAUSE = "runtime-update-failed"

#: What the pair-less rendering says: a runtime whose boot stamp is unreadable can
#: still move to the build on disk (``buildwatch.proves_a_move`` is the guard on the
#: ACT, not on the reading), and a window that said nothing about which build would
#: leave the operator unable to tell a rebuild from a version bump.
#:
#: IT IS ALSO THE PUBLISHED FALLBACK, so a window is never opened with an empty
#: pair: ``""`` is simultaneously the record's "no window" sentinel and the
#: admission gate, so a window that stored it would be invisible AND would queue
#: nothing while the sender got a receipt (agent review round 1, NIT 2).
#: ``process._refresh_for`` substitutes this when ``buildwatch.update_pair_text``
#: cannot name the pair, and ``begin_update`` refuses an empty one outright.
UPDATE_UNNAMED_PAIR = "the build on disk"


def update_phase(
    updating: str = "", updated: str = "", update_failed: str = ""
) -> "tuple[str, str]":
    """``(phase, pair)`` for a record's three update fields: the ONE reading of them.

    WHY A READER AND NOT THREE FIELDS ON EVERY SURFACE. The record keeps the three
    facts apart because they are written at different times by different processes
    (a window opens, a window fails, a successor boots having applied one), and each
    has to survive the wire on its own. Every READER wants the same single answer —
    which phase is this session in, and about which build — so the precedence lives
    here rather than being re-derived (differently) by ``lop sessions``, the info
    panel and the phone.

    PRECEDENCE, and each step is a fact about time: an OPEN window is the live state
    and outranks both terminal facts; a FAILED one is the newest terminal fact about
    the last attempt (and the runtime that owns it is still serving, which is what a
    reader must act on); ``updated`` is the oldest — it is true of this process's
    whole life, so it loses to anything newer. An EMPTY pair is not a phase: all
    three fields empty returns ``("", "")``, which is the ordinary idle row.
    """
    if updating:
        return UPDATING, updating
    if update_failed:
        return UPDATE_FAILED, update_failed
    if updated:
        return UPDATING_DONE, updated
    return "", ""


def update_phrase(phase: str, pair: str = "") -> str:
    """The sentence for one phase of an update window, from the ONE vocabulary.

    The prose surfaces (the TUI notice, the info panel's note) render through
    this and never compose their own copy — see :data:`UPDATING` for why the
    sentence cannot be a table key.

    WHAT EACH PHASE PROMISES, because the three are not degrees of one thing:
    ``UPDATING`` says the message is HELD and arrives when the successor is up
    (the promise the incident's refusal broke); ``UPDATING_DONE`` says the move
    happened and the successor is running it; ``UPDATE_FAILED`` says the runtime
    is still here, on the OLD build, and the update needs reporting. A reader
    who cannot tell the last one from the first would keep waiting for a handover
    that has already been abandoned.
    """
    where = pair or UPDATE_UNNAMED_PAIR
    if phase == UPDATING:
        return (
            f"updating to {where} — messages are queued and will be sent when the new "
            f"build is up"
        )
    if phase == UPDATING_DONE:
        return f"updated to {where} — the new build is running this session"
    if phase == UPDATE_FAILED:
        return (
            f"the update to {where} did not finish — this session is still running the "
            f"build it loaded, and your messages are running on it"
        )
    return ""


def update_short(phase: str, pair: str = "") -> str:
    """The fleet cell for one phase: one row of ``lop sessions``, not a sentence.

    The pair is the wide part, so the compact form names the NEW build only — the
    question a rotation script asks of a row is "which of these is still on the
    old build, and what is it moving to" (the same question
    ``server._refresh_if_idle`` answers with ``retiring to <label>``). "" for a
    phase this build cannot place, which renders as no cell at all rather than as
    another phase's words.

    A PAIR THIS BUILD CANNOT PARSE NAMES NO BUILD. The phase alone is the cell then,
    for the reason the failed phase gives below and because the alternative —
    ``"updating → the build on disk"`` — was 28 cells into a 26-cell column, i.e. a
    silent cut of the one string whose job is to say the destination is unknown
    (design review round 1, D3).
    """
    to = pair.split("→", 1)[1].strip() if "→" in pair else ""
    if phase == UPDATING:
        return f"updating → {to}" if to else "updating"
    if phase == UPDATING_DONE:
        return f"updated → {to}" if to else "updated"
    if phase == UPDATE_FAILED:
        # PHASE ONLY, and the missing pair is the point rather than an omission: the
        # move this names did NOT happen, so the build pair belongs to the failure
        # notice ("could not move to X"), never to a fleet cell that would read as
        # "went to X". Which build the row is actually on is the neighbouring version
        # column's job, and it is already there.
        return "update failed"
    return ""


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


#: The largest value a subagent count on a record may carry and still be read as
#: a measurement.
#:
#: Lives here for the same reason the set above does: THREE readers refuse a value
#: past this ceiling — ``info.collect`` when it tallies a fleet, ``resume._counted``
#: when it asks whether a sidebar row is delegating, and the desktop listing's own
#: response model when it serializes a row — and they must refuse at the SAME
#: number, because the failure they guard is a display one: a 31-digit figure
#: renders 81 cells wide and overflows every frame, including the abbreviated rung
#: that exists to serve it. It is not that such a count is wrong; it is that nothing
#: past this could be real, so refusing it is the honest reading, and the readers
#: agreeing is what keeps one surface from printing a number another drops.
#:
#: DELIBERATELY FAR ABOVE ANYTHING THIS CODEBASE CAN PRODUCE —
#: ``DEFAULT_MAX_RUNNING_JOBS`` is 15 and the count is a ``len()`` over a bounded
#: roster — so it can only reject a foreign or damaged record, never a real fleet.
#: Six digits still fit the narrow rung.
#:
#: ``SessionRecord.from_json`` does no type validation, so this is a bound on what
#: a RECORD may say, not on what the runtime publishes (its own writer counts real
#: children). This module stays stdlib-only, so neither the constant nor the reader
#: below costs anything to import on the CLI startup path.
MAX_REPORTED_SUBAGENT_COUNT = 999_999


def reported_subagent_count(value: Any) -> int | None:
    """A published subagent count, or ``None`` when the record did not report one.

    THE ONE RULE, read by every consumer of these two fields: ``info.collect``'s
    fleet tally, ``resume._counted`` (the sidebar's predicate), and the desktop
    listing's response model at the wire edge. It lives beside the fields it
    validates rather than in the first module that needed it, because the three
    disagreeing about which values are believable is how one surface prints a
    figure another drops.

    ``SessionRecord.from_json`` filters keys and calls the constructor — it does
    no type validation — so every field on a record is whatever the writer put in
    the file. That is fine for the strings and bools read elsewhere, which only
    ever get formatted, but these two are the first record fields its readers do
    ARITHMETIC on, and arithmetic is where a foreign value stops being cosmetic:

    * a ``str`` or ``list`` raises ``TypeError`` inside a roll-up. ``info``'s
      ``_safe`` guards whole SECTIONS, so one bad record cost the entire sessions
      block — no table, no runtimes row, and no lower-bound caveat — on a screen
      whose whole purpose is describing a host that is already broken. Before
      these fields existed there was no arithmetic there and the same record
      listed normally, so that was a regression rather than a new limitation.
    * a merely-numeric wrong value does not raise at all, which is worse: a float
      printed ``4.5 total — 1 sessions + 3.5 subagents`` and a negative printed
      ``-1 subagents``, both as measured fact.
    * at the desktop listing's wire edge the same choice is between a degradation
      and an outage, because a validation error on ONE row fails the WHOLE
      response: a damaged record would take out the conversation list rather than
      lose a count from it.

    Anything that is not a non-negative ``int`` at or below
    :data:`MAX_REPORTED_SUBAGENT_COUNT` is therefore treated as NOT REPORTED
    rather than sanitised into a number: an unusable value is not a measurement,
    and calling it ``None`` is what each reader's own contract already says to do
    with a missing term. ``bool`` is excluded explicitly — it is an ``int``
    subclass, so ``True`` would otherwise count as one subagent.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    if value > MAX_REPORTED_SUBAGENT_COUNT:
        return None
    return value


@dataclass
class SessionRecord:
    """The discovery record one ``lop`` process publishes for one session.

    Lives at ``~/.local-operator/run/mobile/<pid>.json`` — keyed by pid
    because a process hosts exactly one interactive session at a time, so the
    pid is the natural uniqueness token and ``kill -9`` leaves exactly one
    stale file to reap.

    ``control_key`` is the whole authorization story of the control socket for
    ORDINARY operations: the record is mode 0600 under a 0700 directory, so
    anything that can read the key is already the owning account. The daemon
    never transmits it further — the phone never learns it.

    It is deliberately NOT the whole story for the operations that INCREASE
    authority (issue #1310). ``/approvals auto`` and an approved card remove the
    gate that constrains the caller, and a model-authored tool call runs as this
    same uid — so it can read this very file. Those two classes therefore also
    demand the per-session operator capability, which is held only in the memory
    of the process that started the session (``harness/approval.py``). Nothing
    about it belongs in this record: a field here is readable under the same uid
    and would reinstate the defect. See ``docs/design/approval-authority.md``.
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
    #: WHEN this session's last viewer left (``time.time()``), or ``None`` if no
    #: viewer has ever been attached. Cleared back to ``None`` when one attaches.
    #:
    #: WHICH MOMENT, not merely that. The residency policy keeps a runtime a
    #: viewer has left warm for ``runtime.keep_alive_seconds``, and bounds how
    #: many such runtimes one machine holds by evicting the least recently
    #: detached (``process._keep_alive_victim``). That bound needs an ORDER, and
    #: this is the only field that carries one: ``heartbeat_at`` cannot, because
    #: a detached idle runtime keeps beating, and ``detached`` is a boolean.
    #: ``None`` is load-bearing rather than a missing value — it is what tells
    #: the keep-alive that this runtime is one nobody has looked at, which is the
    #: population the ordinary 3 s drain was written for.
    #:
    #: ADDITIVE AND KEYLESS ON AN OLDER READER, exactly like the block above:
    #: ``from_json`` drops unknown keys, so a mixed-version fleet reads and
    #: writes records with and without this field interchangeably, and
    #: ``PROTOCOL_VERSION`` deliberately does not move for it. Nothing is
    #: required to read it: a runtime without it keeps today's residency.
    detached_at: float | None = None
    #: At least one attach CLIENT is connected — the reaper's own term 3
    #: (``RuntimeServer.attach_clients``), which is NOT the same fact as
    #: ``detached`` directly above.
    #:
    #: WHY BOTH EXIST, because two fields that look alike invite exactly one
    #: mistake (review round 1, F1). ``detached`` is VISIBILITY: it is true while
    #: a multiplexing TUI has switched to another session and left this one's
    #: terminal attached but not on screen (``viewer_watch displaying=False``),
    #: and it is what a picker paints a row from. This one is ATTACHMENT, which
    #: is what forbids an exit. A caller asking "may this runtime go?" needs
    #: this one; a caller painting "nobody is watching" needs the other. The
    #: keep-alive cap charged itself on ``detached`` until this field existed, so
    #: a switched-away TUI's runtime — which can never enter a drain while its
    #: viewer holds it — sat in a cap slot that could never be given back.
    #:
    #: ADDITIVE AND KEYLESS ON AN OLDER READER, like ``detached_at`` above:
    #: ``from_json`` drops unknown keys, an older runtime's record defaults to
    #: False, and ``PROTOCOL_VERSION`` deliberately does not move for it.
    watching: bool = False
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
    #: An UPDATE WINDOW is open: this IDLE runtime has committed to moving to the
    #: build on disk, holds the admission lock while it announces and exits, and is
    #: QUEUEING admissions for the successor (``inbox.jsonl``, drained at boot)
    #: rather than refusing them. The value is the build PAIR it is moving to —
    #: ``"0.59.9 → 0.59.11@ead71b6"`` — and ``""`` means no window is open.
    #:
    #: WHY A PAIR AND NOT A SENTENCE, which is what :data:`UPDATING` explains at
    #: length: the sentence carries a pair, so the field has to carry the data and
    #: the surfaces compose their own copy (``types.update_phrase`` for prose,
    #: ``types.update_short`` for the fleet cell). A field that held the sentence
    #: could not be enumerated for the vocabulary pin and could not be compared by
    #: a reader that wants "which build is it moving to".
    #:
    #: CLEARED WHEN THE WINDOW CLOSES, unlike :attr:`leaving`, and the asymmetry is
    #: the point: a leave always ends in an exit (the field is a one-way door), while
    #: a window can ABORT — the bound expired, the runtime kept the build it loaded,
    #: and it is serving again. A window that stayed published after that would tell
    #: every front end the session is mid-move when it is not.
    updating: str = ""
    #: The one-shot terminal facts of a window, each carrying the pair it is about.
    #:
    #: ``updated`` is written by the SUCCESSOR's record — the fact that this
    #: process exists because an update applied, which is the only place it can be
    #: known (the predecessor is gone by the time the spool is drained). Set once at
    #: boot from the handover marker the outgoing runtime left, and never cleared:
    #: it is true for the whole life of this process, and a surface that showed it
    #: for a moment and then dropped it would be racing the reader it exists for.
    #:
    #: ``update_failed`` is written by a runtime that STAYED — its window ran out
    #: of bound — and names the pair it failed to move to. Together the two make the
    #: outcome of the move legible on every surface that reads a record, including
    #: the ones that never saw the window itself (a fleet listing taken afterwards,
    #: the phone's projection, the desktop feed).
    updated: str = ""
    update_failed: str = ""

    # -- what the last beat measured ---------------------------------------
    # Same additive contract as the blocks above, and PROTOCOL_VERSION again
    # deliberately does not move: nothing is required to read these, and a
    # reader that ignores them loses a distinction rather than a fact.
    #
    # WHY THEY EXIST. ``heartbeat_age_s`` says only HOW LONG the owner has been
    # quiet, and that one number cannot tell three different situations apart:
    # a runtime wedged in its own work, a runtime the host stopped scheduling,
    # and a runtime that is simply gone. All three read as ``wedged`` at 45 s,
    # and on 2026-09-20 five sessions sat in that single ambiguous word for
    # 1.5-7.2 h before being reaped by hand. These two fields are in-process
    # readings taken by the owner itself, so they cost no fork and remain true
    # of the PROCESS rather than of the machine's average: a large lag with CPU
    # that advanced says the runtime burned its own core (the measured shape —
    # ~0.9 core against a stale beat), while a large lag with CPU that did NOT
    # advance says it was descheduled (the host). ``None`` means this build does
    # not report, never zero — a pre-field runtime has not told us either way.
    #
    #: Seconds since the PREVIOUS beat, measured by the beating loop. The gap is
    #: what a stalled loop leaves behind: a healthy runtime owns up to
    #: ``HEARTBEAT_INTERVAL_S`` (15 s) here, and this host has produced 105.8 s
    #: and 205.8 s on sessions whose CPU was advancing. Note that a turn-boundary
    #: republish rewrites ``heartbeat_at`` without re-measuring, so this can
    #: exceed the age the record appears to have by up to one interval.
    beat_lag_s: float | None = None
    #: How much CPU time this PROCESS spent since the previous beat — all
    #: threads, ``time.process_time()``, read in-process. Paired with the lag
    #: above it separates "starved by its own work" (this advanced) from
    #: "starved by the host" (this did not), which no other field can.
    cpu_since_beat_s: float | None = None

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
