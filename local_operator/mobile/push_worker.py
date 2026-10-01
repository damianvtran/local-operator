"""The push worker: the mobile daemon's emit loop (push/ack-sync S5 of ADR 0006).

The slice plan's S5 row is the specification, and ADR 0006's §2.1 (the chain and
its cursors), §2.3 (who decides that a push may be raised) and §3.4 (the two
idempotency keys) are the rules this module implements. It lives in the mobile
daemon because that is the always-on process that owns ``attention.db`` and
serves the phone — NOT the tunnel service, whose 10 s control-plane poll exists
for the relay's reachability, so coupling push to it would mean "stop the tunnel,
stop the pushes" (§2.1).

**Why a worker at all, and why it is not a banner loop.** ``unseen`` is a LEVEL
(ADR §Context 1), so a worker with no position either re-pushes every historical
completion after a restart or silently drops what arrived while it was down. The
answer is two durable cursors with a baseline at enablement, plus the
acknowledgement map as a third, in-memory delta — the same shape the desktop feed
already uses (ADR §2.1). Everything below is one of those three inputs, one of
the §2.3 gates, or the queue that carries an emit to the cloud.

Three decisions are load-bearing and each one has a trap it avoids:

- **Detection is structural, never a count.** A tick in which a publish and an
  ack land together leaves any derived count *equal*, so a count rule would emit
  nothing for the case the attention emit exists for (ADR §2.1, round 2 M1).
  ``published_since``/``superseded_since``/``acknowledgement_map`` are three
  separate reads, so "which conversation moved" is answered by the read that saw
  it rather than inferred from a total.

- **A cursor advances when an item RESOLVES, and an emit-bearing item resolves
  only on the cloud's accept.** A presence-deferred item (§2.3) is still ahead of
  its cursor, so a daemon restart re-considers it instead of losing it. An item
  that needs no emit at all — a row the operator asked not to be told about
  (``notify`` false), a completion already read on another surface, a computer
  with no live device to deliver to — resolves without an emit, because the
  alternative is a cursor pinned forever on a row no push will ever describe.
  What the ADR forbids is advancing on *consideration* of a would-be emit, and
  this module is careful to keep those two apart.

- **The cursor is the durable position; the queue is derived from it.** The
  pending backlog IS the region behind the cursor (ADR §2.1), so nothing about an
  in-flight emit needs its own storage: a refusal is retried on the wire with the
  same key, and a restart re-derives the item from the cursor and re-emits it
  with a key the cloud's ``Idempotency-Key`` turns into a duplicate ``202``
  rather than a duplicate push. The two exceptions, stated where they matter,
  are the attention emit's SEQUENCE key and the digest's WINDOW ID (§3.4), which
  are the keys that cannot be re-derived from a record — see
  :meth:`PushWorker._mint_emit` and :meth:`PushWorker._digest`.

**The transport is a seam, and it is injected.** No route emits these shapes yet
(S7 is the cloud lane), so the worker is built against a callable: the tests pass
a recorder, a stub control plane answers ``202``/``5xx`` on demand, and the real
cloud call — the outbound POST to ``/v1/tunnels/{tunnel_id}/push/events``
authenticated as the connector's credential — binds here when it exists. The
worker owns no URL and no header spelling beyond the idempotency key it hands
over: route, auth and retry transport are the transport's business, exactly as
:class:`local_operator.mobile.push_credentials.CredentialTransport` argues for the
report block.

**What this module deliberately does NOT do.** It does not touch ``deliveries``:
the phone is not a rung of the local banner ladder and ``claim_delivery`` is
untouched, because a claim is per conversation and per machine, so a push that
claimed would silence the desktop banner on this machine and a desktop claim
would silence the phone (ADR §2.3, decision 1). It NEVER acknowledges anything:
the attention emit is a consequence of a read that already moved, so
:meth:`PushWorker.note_ack` takes the acting device's id and writes no receipt —
the route acks, and the worker only asks the OTHER devices to re-read (S6). That
nudge emits nothing itself: the pass that follows reads the store and does the
emitting, so §3.1's "the nudge consumes the change" is kept in OUTCOME — at most
one emit per change, with the entry's detector state advanced by the pass —
rather than in the entry's literal spelling of an immediate emit that the next
tick then has to suppress; there is no immediate emit here to duplicate. And it
does not decide WHO the cloud wakes: the ``Idempotency-Key`` and the payloads
come from ``push_payload`` (S3's freeze), the ``exclude`` list is carried
verbatim, and the per-device skip belongs to the emit-side gate in the cloud,
fed by the S4c report block.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from local_operator.mobile import push_devices as mobile_push_devices
from local_operator.mobile import push_handles
from local_operator.mobile.push_payload import (
    attention_emit_key,
    attention_payload,
    completion_emit_key,
    completion_payload,
    digest_emit_key,
    digest_payload,
    emit_body,
)
from local_operator.session.attention import AttentionStore
from local_operator.session.runtime.presence import PRESENCE_TTL_S, desktop_presence

logger = logging.getLogger(__name__)

#: The worker's durable position, directly under ``config_dir()`` beside the
#: daemon's other owner-private state (``mobile-seen.json``,
#: ``mobile-push-devices.json``, ``push-handle.key``).
PUSH_WORKER_STATE_NAME = "mobile-push-worker.json"

#: The coalescing ceiling, and it is the SAME number the desktop feed and the
#: TUI's background announcer hold (``server/utils/desktop_feed.BURST_LIMIT``,
#: "shared with the TUI's per-tick cap and asserted equal by a test"). It is
#: spelled here as a literal rather than imported for the reason the TUI spells
#: it too: the mobile daemon must not grow an import edge into the desktop feed's
#: module for one integer, and ``tests/unit/mobile/test_push_worker.py`` asserts
#: the two constants are equal, which is the guard an import would only move.
BURST_LIMIT = 3

#: The bound on a presence deferral (§2.3, decision 2). A suppressed push is
#: DEFERRED, never dropped silently: the worker re-checks while the completion is
#: still ``unseen`` and still inside this window, and emits when the window
#: expires even if the user is still looking at the desktop app. The window is
#: what keeps "do not interrupt a user who is already looking at the work" from
#: becoming "never tell the phone".
DEFERRAL_WINDOW_S = 5 * 60

#: How often a deferred item re-reads the presence. It is the presence TTL and
#: not an independent number on purpose: a desktop lease stops being believed one
#: TTL after its last beat, so re-checking sooner than that would only re-read a
#: value that cannot have changed, and re-checking later would hold a push past
#: the moment the gate it is waiting on has already opened.
PRESENCE_RECHECK_S = PRESENCE_TTL_S

#: The wire retry (§2.1, round 2 m1(b)): three attempts over ~2 minutes with the
#: SAME idempotency key, then one drop and one log line. The interval is derived
#: rather than spelled twice, so the two bounds cannot drift apart.
EMIT_RETRY_ATTEMPTS = 3
EMIT_RETRY_WINDOW_S = 2 * 60
EMIT_RETRY_INTERVAL_S = EMIT_RETRY_WINDOW_S / (EMIT_RETRY_ATTEMPTS - 1)

#: The three kinds of emit this worker composes. A DIGEST is the coalesced
#: catch-up of more than ``BURST_LIMIT`` eligible rows (§2.1); an ATTENTION emit
#: is a badge correction raised because a read moved somewhere on this machine.
EVENT_COMPLETION = "completion"
EVENT_DIGEST = "digest"
EVENT_ATTENTION = "attention"

#: Each kind's value IS its ``type`` on the wire (a test asserts the three
#: against ``push_payload``), and the three are deliberately not one form with
#: flags: a completion names a record, an attention emit corrects the badge
#: SILENTLY, and a digest is the visible coalesced catch-up.


def state_path(config_dir: Path) -> Path:
    """The cursor file's path under one config root.

    Public because the daemon's refusal log names it: "which file" is the first
    question a reader of that line asks — the same rule the device registry and
    the handle key state for their own stores.
    """
    return config_dir / PUSH_WORKER_STATE_NAME


@dataclass
class WorkerCursors:
    """The worker's durable position, and the only state a restart inherits.

    ``publication_cursor`` and ``supersede_cursor`` are the two cursors §2.1
    names: a heal moves neither ``MAX(sequence)`` nor ``SUM(acknowledged)``, so
    the publication cursor can never see it and the supersede log is a read of
    its own. ``attention_sequence`` is §3.4's machine-minted monotone counter for
    attention emits — persisted here rather than derived, because a key that
    could be re-used for a different emit would let the cloud's dedupe swallow a
    real push. ``enabled_at`` records the instant the baseline was taken, which
    is what makes "nothing from before enabling is ever pushed" checkable rather
    than implied. ``digest_window`` is the pending coalescing window's identity
    (see its own comment), and it is the one field here that is not a position.
    """

    publication_cursor: int = 0
    supersede_cursor: int = 0
    attention_sequence: int = 0
    enabled_at: float = 0.0
    #: The pending coalescing window: ``{emit_id, publications, supersedes}``.
    #:
    #: DURABLE, and that is a contract requirement rather than tidiness (§3.4 at
    #: ``5da35710``): a digest is the one VISIBLE emit, so a restart mid-window
    #: that re-minted its ``emit_id`` would put a second banner in front of the
    #: user — the cloud dedupes on the key alone, and it may have delivered the
    #: first one and lost the ``202``. THE CAVEAT THIS FIELD CLOSES IS THE
    #: WINDOW'S, NOT THE ATTENTION EMIT'S (round 3, R13): the attention sequence
    #: is persisted with the cursor, so a restart RESUMES it rather than
    #: re-minting. What the cursor only DERIVES is this window, which is exactly
    #: why the id has to be stored for the one visible type.
    digest_window: dict[str, Any] | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "publication_cursor": self.publication_cursor,
            "supersede_cursor": self.supersede_cursor,
            "attention_sequence": self.attention_sequence,
            "enabled_at": self.enabled_at,
            "digest_window": self.digest_window,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> WorkerCursors:
        """Read a stored position, refusing anything that is not one.

        A value that is not an ``int`` is refused rather than coerced: the
        cursors are compared against SQLite's own integers, and a coerced
        ``None`` — "the file was half-written" — would silently read as 0, i.e.
        as "push everything this machine has ever recorded". The caller
        re-baselines on the refusal instead, which is the safe direction.
        """
        fields: dict[str, Any] = {}
        for name in ("publication_cursor", "supersede_cursor", "attention_sequence"):
            value = data.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"push worker state has no usable {name!r}")
            fields[name] = value
        enabled_at = data.get("enabled_at")
        if not isinstance(enabled_at, (int, float)) or isinstance(enabled_at, bool):
            raise ValueError("push worker state has no usable 'enabled_at'")
        fields["enabled_at"] = float(enabled_at)
        fields["digest_window"] = _usable_digest_window(data.get("digest_window"))
        return cls(**fields)


def _usable_digest_window(value: object) -> dict[str, Any] | None:
    """The persisted coalescing window, or ``None`` when it is not usable.

    DROPPED rather than refused, unlike every other field here: refusing the file
    re-baselines the cursors, which re-pushes this machine's whole backlog to
    save one window's identity. Losing the identity costs at most one duplicate
    banner for a window that was mid-flight when the file was written — and the
    positions are what make it usable, so a window without them is not a window
    at all.
    """
    if not isinstance(value, Mapping):
        return None
    emit_id = value.get("emit_id")
    publications = value.get("publications")
    supersedes = value.get("supersedes")
    if not isinstance(emit_id, str) or not emit_id:
        return None
    if not isinstance(publications, list) or not isinstance(supersedes, list):
        return None
    if not all(isinstance(seq, int) and not isinstance(seq, bool) for seq in publications):
        return None
    if not all(isinstance(seq, int) and not isinstance(seq, bool) for seq in supersedes):
        return None
    return {"emit_id": emit_id, "publications": publications, "supersedes": supersedes}


@dataclass(frozen=True)
class EmitAccepted:
    """The cloud took the emit (§3.1's ``202 {emit_id, accepted_at}``)."""

    emit_id: str
    accepted_at: int


@dataclass(frozen=True)
class EmitRefused:
    """The cloud did not take it: a 5xx, a timeout, or any non-``202``.

    Deliberately one shape for every refusal. §2.1's retry policy is a wire
    retry of the SAME key, so the worker does not act on the status — a 500 and
    a timeout cost the same attempt — but the status is carried because it is
    the whole of what one log line can usefully say after the third one.
    """

    status: int


class PushEmitTransport(Protocol):
    """Where one emit goes, and what the cloud answered.

    ``body`` is the §3.2 emit body (the payload, plus the report block when the
    caller has its coalescer wired) and ``idempotency_key`` is §3.4's key for
    this exact emit. A retry re-invokes this with the same key and the same body,
    which is what makes the cloud's ``Idempotency-Key`` collapse it into a
    duplicate ``202`` rather than a duplicate push.

    THE TRANSPORT MUST BOUND ITS OWN CALL (review round 1, m3). §2.1's retry
    rule counts "a cloud refusal or timeout" as one failure mode, and this
    Protocol is synchronous: a call that blocks forever blocks the worker's
    thread, which is the daemon's 2 s scan's thread. The bound is the
    transport's because only the transport knows its own client (this repo's
    outbound Radient calls carry ``timeout=30``; ``daemon.py`` also cuts the
    scan loose with ``PUSH_TICK_TIMEOUT_S``, and the worker admits one pass at a
    time so a late pass can never meet a second one).
    """

    def __call__(
        self, body: Mapping[str, Any], *, idempotency_key: str
    ) -> EmitAccepted | EmitRefused: ...


@dataclass(frozen=True)
class EmitRecord:
    """One emit this worker attempted, for the caller to log or assert on.

    ``accepted`` is the cloud's verdict; ``status`` is its HTTP status (0 when a
    transport could not be reached at all, which the real transport reports the
    same way a refusal because §2.1's policy is identical for both).
    """

    kind: str
    idempotency_key: str
    accepted: bool
    status: int


@dataclass
class _PendingEmit:
    """One emit that has not been ACCEPTED yet — the region behind the cursor.

    ``body`` is composed on the first attempt and kept, because ``emit_id`` is
    the machine's identity for one emit (§3.2): re-composing on a retry would
    hand the cloud a different id for the same event and defeat its dedupe. The
    key is fixed when the item is created for the same reason.

    ``deferred_since`` is set only for a presence-deferred item, and it is what
    makes the deferral a delay rather than a drop: the item is still pending, so
    it is still ahead of its cursor, and the window bounds how long that can
    last.
    """

    key: str
    kind: str
    publications: tuple[int, ...] = ()
    supersedes: tuple[int, ...] = ()
    body: dict[str, Any] | None = None
    attempts: int = 0
    first_attempt_at: float | None = None
    last_attempt_at: float | None = None
    deferred_since: float | None = None
    #: The machine's identity for this emit (§3.2), minted when the item is
    #: created — at WINDOW-CLOSE for a digest, because its key is derived from it.
    emit_id: str = ""
    #: The conversations this emit names, so a deferral can be re-evaluated
    #: against the store (an ack terminates it early — §2.3).
    conversations: tuple[str, ...] = ()
    #: The devices that must NOT be woken by this emit — §3.2's ``exclude``, and
    #: the device that just acknowledged when the nudge path named one. Empty
    #: means "exclude nobody", which is what a TICK-detected change sends because
    #: it never knows who acked. It rides the ATTENTION emit only: §3.2 permits
    #: the field on a digest and never requires it there, and a digest's trigger
    #: is a burst rather than an ack.
    exclude: tuple[str, ...] = ()

    @property
    def sequences(self) -> tuple[int, ...]:
        """Every position this emit covers, across both cursors."""
        return self.publications + self.supersedes


class PushWorker:
    """One tick per scan: read the deltas, apply the gates, emit the backlog.

    Constructed by whoever arms push on this machine. Every collaborator that
    touches the outside world is injected — the attention store, the presence
    read, the device registry, the report block, the unread count and the clock
    — so the whole state machine is exercisable without a cloud, a desktop app
    or a device, which is exactly the hedge the plan takes for S5 ("tested
    against a stub control plane", ``docs/push-plan.md`` §"Order, and why" 3).
    """

    def __init__(
        self,
        *,
        store: AttentionStore,
        transport: PushEmitTransport,
        config_dir: Path,
        computer: str,
        clock: Callable[[], float] = time.time,
        presence: Callable[[], bool] | None = None,
        live_devices: Callable[[], Sequence[str]] | None = None,
        report_block: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
        unread_count: Callable[[], int] | None = None,
    ) -> None:
        """``computer`` is the opaque per-account handle §3.2 carries.

        Injected for the reason ``CredentialReport`` injects it: it is the
        account's name for this machine rather than the machine's, so nothing
        here can mint it. ``presence``/``live_devices``/``unread_count`` default
        to the machine-local reads §2.3 names, each behind a parameter so a test
        can put the gate in the state under test instead of manufacturing a
        desktop lease or a registered phone.

        ``live_devices`` answers §2.3's gate 4 with the live devices' IDS rather
        than a yes/no, because S6 has a second question for the same read: which
        devices a correction would actually reach once §3.2's ``exclude`` has
        been subtracted (`[]` is "no device this machine may deliver to").

        ``report_block`` is the S4c coalescer's ``piggyback(CARRIER_EMIT)``. It
        is OPTIONAL and the body falls back to the bare §3.2 payload rather than
        an empty block: an empty block is a claim about every device ("nothing
        has changed") that a worker with no coalescer has not made and must not
        invent.
        """
        self.store = store
        self.transport = transport
        self.config_dir = config_dir
        self.computer = computer
        self._clock = clock
        self._presence = presence if presence is not None else _desktop_attended
        self._live_devices = live_devices if live_devices is not None else self._live_device_ids
        self._report_block = report_block
        self._unread_count = unread_count if unread_count is not None else self._store_count
        self._cursors = WorkerCursors()
        self._loaded = False
        #: What the last successful write put on disk, so an unchanged position
        #: costs no rename (see :meth:`_save`).
        self._written: dict[str, Any] | None = None
        #: The single-claimant guard over one pass (see :meth:`tick`).
        self._pass = threading.Lock()
        #: The highest positions READ this run, so the cursor can move to them
        #: once nothing is pending above them (see :meth:`_advance`).
        self._seen_publication = 0
        self._seen_supersede = 0
        #: Items whose emit is not accepted yet — the pending backlog.
        self._pending: dict[str, _PendingEmit] = {}
        #: Sequences already resolved that sit ABOVE the cursor, because a
        #: still-pending item below them holds the cursor back. Without this the
        #: next tick would re-emit them: the reads are "everything newer than the
        #: cursor", not "everything not yet done".
        self._resolved_publications: set[int] = set()
        self._resolved_supersedes: set[int] = set()
        #: The previous tick's ``{conversation: acknowledged}``. In-memory on
        #: purpose (see :meth:`_load`).
        self._acks: dict[str, int] = {}
        #: The ``(device, conversation, acknowledged)`` hints a ``/seen`` nudge has
        #: left since the last pass. The ONE piece of state an off-tick thread may
        #: write (the route's), so it carries its own lock and is held for a
        #: set-add only — it must never be able to block a pass, and a pass must
        #: never block on it while holding a store read.
        self._nudge_lock = threading.Lock()
        self._nudge_exclude: set[tuple[str, str, int]] = set()

    # -- the tick ------------------------------------------------------------

    def tick(self, *, now: float | None = None) -> list[EmitRecord]:
        """One pass: refresh the position, apply the gates, drain what may go.

        ``now`` is injectable because every bound here is a duration, and a test
        that sleeps five minutes to prove a five-minute window is a test nobody
        runs. Returns the emits attempted in this pass (accepted or not), which
        is what the daemon logs and what a test asserts on.

        ONE PASS AT A TIME. The daemon drives this from a worker thread and cuts
        the scan loose if the pass outlives its bound, so a still-running pass
        must not meet a second one: the queue, the cursors and the minted key
        space are single-claimant state, and a second pass would re-attempt items
        the first is holding. A tick that cannot claim the pass does nothing and
        says so at debug level; the next scan interval retries.
        """
        if not self._pass.acquire(blocking=False):
            logger.debug("push worker tick skipped: a pass is still running")
            return []
        try:
            stamp = self._clock() if now is None else now
            self._load(stamp)
            attended = self._presence()
            self._collect(stamp, attended)
            records = self._drain(stamp, attended)
            self._advance(stamp)
            self._save()
            return records
        finally:
            self._pass.release()

    def _load(self, now: float) -> None:
        """Read the durable position, or BASELINE it on first use.

        Write-lazy, like ``push_handles``' key: the file appears the first time
        the worker actually runs, never at construction, so a daemon that never
        gets a transport leaves no state behind.

        **Three outcomes, and each is a different decision:**

        - **no file** — this is ENABLEMENT. The baseline is taken now and
          nothing before it is ever pushed: ``publication_cursor`` is the store's
          own ``MAX(sequence)``, ``supersede_cursor`` is the newest retained
          supersede, and the acknowledgement map is snapshotted. This is the one
          moment at which "everything the store already holds" is deliberately
          not pushed, and it is why the cursors exist at all.
        - **a readable file** — a restart. The cursors come back as they were, so
          the backlog behind them is re-considered and nothing already accepted
          is re-emitted. The ACKNOWLEDGEMENT SNAPSHOT is re-baselined from the
          store, deliberately: it is not one of the two durable cursors, and a
          restart that replayed every receipt the machine ever wrote would fire
          one badge correction per historical ack. An ack that landed while the
          daemon was down is not lost — it is exactly what the app's next read
          already sees, and the badge is not carried on a push at all (§1.5).
        - **an unreadable file** — one bounded log line and the baseline is
          taken again. Refusing to run would stop every push on this machine for
          a corrupt handful of integers, which is the silent stop §4 forbids; the
          cost of re-baselining is bounded and named in the line itself.
        """
        if self._loaded:
            return
        path = state_path(self.config_dir)
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._baseline(now)
        except OSError as exc:
            logger.warning("push worker re-baselined at %s: %s", path, exc)
            self._baseline(now)
        else:
            try:
                self._cursors = WorkerCursors.from_json(json.loads(raw))
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "push worker re-baselined at %s: the stored position is unusable (%s)",
                    path,
                    exc,
                )
                self._baseline(now)
            else:
                self._acks = self.store.acknowledgement_map()
                # Record what is already on disk, so the write-skip in
                # :meth:`_save` is armed from the first tick: without this the
                # first tick after every restart rewrites an identical file.
                self._written = self._cursors.as_json()
        self._loaded = True

    def _baseline(self, now: float) -> None:
        """Take the position that makes "nothing from before enabling" true."""
        revision = self.store.revision()
        # ``superseded_since(0)`` is a bounded read by construction: the store
        # prunes ``supersede_log`` to its newest 256 rows, and the newest row is
        # the only thing the baseline needs.
        superseded = self.store.superseded_since(0)
        self._cursors = WorkerCursors(
            publication_cursor=int(revision[0]),
            supersede_cursor=int(superseded[-1]["sequence"]) if superseded else 0,
            attention_sequence=0,
            enabled_at=now,
        )
        self._acks = self.store.acknowledgement_map()
        self._save()

    # -- the /seen nudge -----------------------------------------------------

    def note_ack(self, *, device_id: str, conversation: str, acknowledged: int) -> None:
        """A ``/seen`` landed: this device acted, on this conversation (ADR §3.1).

        A HINT, not an emit: the caller has already written the receipt, and the
        correction is a consequence of that read moving rather than a second ack
        path — so nothing here acknowledges, writes a receipt or touches a
        ``deliveries`` row, and the next pass decides from the store whether
        there is anything to correct. What the hint buys is §3.2's ``exclude``:
        without it a tick-detected change cannot name the acting device ("absent
        means exclude nobody"), so the device the user just read on would be
        woken to re-read state it already has.

        BOTH HALVES ARE CARRIED, and they are what make the exclusion exact
        rather than approximate. A hint is a claim about ONE change, and the pass
        applies it only when that change is one it is actually carrying: the
        conversation must have moved in the read being diffed, and the receipt's
        own ``acknowledged`` value must be the one that read returned. Without
        them a hint that outlived its change would be subtracted from whatever
        change came next — on another conversation (review round 1, m1) or on a
        NEWER receipt of the same one (review round 2, m2), where the acting
        devices would both be excluded and, with no device left, the correction
        dropped outright.

        NOT CONSUMED HERE, and that is the point of the split. The pass takes the
        hints BEFORE it reads the acknowledgement map, so a hint it takes always
        describes a receipt written before that read — the receipt precedes the
        nudge, because the route acks first — and therefore a change in the map
        that pass diffs. The one case that order alone cannot cover (a receipt an
        earlier pass already consumed while its nudge was still in flight) is
        covered by the two match terms above: the hint matches neither a later
        pass's movement of that conversation nor a newer receipt of it, so it is
        spent rather than applied, the acting device is corrected like every other
        device, and a correction is never dropped for lack of a recipient — one
        benign extra silent wake, the window §3.1 states, and the badge is right
        on its next read either way.

        ``device_id``, ``conversation`` and ``acknowledged`` are the worker's own
        vocabulary: the conversation is the store key (``session/<id>``), not a
        handle, and ``acknowledged`` is the value the receipt moved it to
        (``state()["revision"][1]`` — the same watermark
        ``acknowledgement_map()`` returns), so a later receipt on that
        conversation is a DIFFERENT value and cannot match this hint. A desk ack
        sends no hint at all — the TUI and the desktop write straight into the
        store and the tick detects them — which is the same thing as a device
        list that excludes nobody.
        """
        if not device_id or not conversation:
            return
        with self._nudge_lock:
            self._nudge_exclude.add((device_id, conversation, int(acknowledged)))

    def _take_nudged_devices(self) -> set[tuple[str, str, int]]:
        """The ``(device, conversation, acknowledged)`` hints left since the last pass.

        ONE CALL PER PASS, taken BEFORE the acknowledgement-map read and cleared
        whether or not anything moved: a hint describes the change the pass is
        about to read, and one held over would describe a pass that has already
        happened.
        """
        with self._nudge_lock:
            taken = set(self._nudge_exclude)
            self._nudge_exclude.clear()
        return taken

    # -- the three deltas ----------------------------------------------------

    def _collect(self, now: float, attended: bool) -> None:
        """Refresh the pending set from the three reads, and gate it.

        Positions that need no emit are RESOLVED here rather than returned: the
        cursor's job is to be the durable position, and this method is the only
        one that knows which of the positions it just read will never carry an
        emit.
        """
        # Gate 4 is a property of the COMPUTER, not of the row (§2.3), so it is
        # read once per pass: a computer with no device it may deliver to has
        # nothing to emit for, and every candidate below resolves rather than
        # waits (see :meth:`_live_device_ids` for why it is a skip, not a
        # deferral). The IDS are read rather than a yes/no because the
        # attention emit below subtracts §3.2's ``exclude`` from them.
        devices = self._live_devices()
        deliverable = bool(devices)

        # Positions a PENDING emit already speaks for. A digest covers a whole
        # batch, and while it is in the queue the rows it covers must not be
        # re-derived as fresh candidates: re-deriving them mints a second,
        # different key for one batch, which is exactly the duplicate §3.4's
        # ``Idempotency-Key`` exists to prevent (review round 1, B1). The rows are
        # still re-EVALUATED below — the ack path depends on that — only the
        # re-creation is skipped.
        covered = {sequence for item in self._pending.values() for sequence in item.sequences}

        # 1. The supersede cursor first, because a heal is the one change the
        #    publication cursor cannot see (ADR §2.1): a correction rewrites the
        #    row in place and moves neither ``MAX(sequence)`` nor
        #    ``SUM(acknowledged)``.
        entries = self.store.superseded_since(self._cursors.supersede_cursor)
        if entries:
            newest = int(entries[-1]["sequence"])
            # THE STORE PRUNES ITS OWN HEAL LOG (``supersede_log`` keeps its
            # newest 256 rows), so a downtime longer than that loses the identity
            # of the oldest heals while ``revision()`` still reports that heals
            # happened. The condition is computable from this read alone: the
            # rows are contiguous seqs, so anything past the cursor that is not
            # in the list was pruned. The response is the ADR's own — re-baseline
            # to the newest, say ONE line naming what could not be carried, and
            # do not sweep: a lost heal identity costs one push correction, and
            # the badge is read from the machine and is right on the next read.
            lost = newest - self._cursors.supersede_cursor - len(entries)
            if lost > 0:
                logger.warning(
                    "push worker re-baselined the supersede cursor to %d: %d heal(s) "
                    "were pruned before it could carry them",
                    newest,
                    lost,
                )
                self._cursors.supersede_cursor = newest
                self._seen_supersede = max(self._seen_supersede, newest)
                entries = []
        for entry in entries:
            sequence = int(entry["sequence"])
            self._seen_supersede = max(self._seen_supersede, sequence)
            if sequence in self._resolved_supersedes:
                continue
            conversation = str(entry["conversation"])
            if not deliverable or not self._is_candidate(conversation):
                self._discard_positions(supersedes=(sequence,))
                continue
            if sequence in covered:
                continue
            self._add_completion(conversation, supersedes=(sequence,), now=now, attended=attended)

        # 2. The publication cursor: new completions.
        for row in self.store.published_since(self._cursors.publication_cursor):
            sequence = int(row["sequence"])
            self._seen_publication = max(self._seen_publication, sequence)
            if sequence in self._resolved_publications:
                continue
            conversation = str(row["conversation"])
            if not deliverable or not self._is_candidate(conversation):
                self._discard_positions(publications=(sequence,))
                continue
            if sequence in covered:
                continue
            self._add_completion(conversation, publications=(sequence,), now=now, attended=attended)

        # 3. The acknowledgement map, diffed against the last tick. An ack is a
        #    durable change to the same watermark ``unseen`` is computed from, and
        #    the phone has to be told to re-read: that is the attention emit.
        #    ``exclude`` comes from the nudges the ``/seen`` route left since this
        #    pass's predecessor — a tick-detected change (the TUI's or the
        #    desktop's ack) never knows who acknowledged, and sends none (§3.2).
        #
        #    THE HINTS ARE TAKEN BEFORE THE MAP IS READ, and that order is what
        #    makes a hint mean something. A route acks its receipt and only then
        #    nudges, so a hint taken here describes a receipt written before this
        #    read, and its change is therefore in the map being diffed. Taking it
        #    after the read would let a receipt that landed inside the gap sit in
        #    ``exclude`` while its conversation sits outside ``acked``, which is
        #    how one change's actor gets skipped on another change's correction
        #    (review round 1, m1).
        hints = self._take_nudged_devices()
        current = self.store.acknowledgement_map()
        acked = sorted(
            conversation
            for conversation, value in current.items()
            if self._acks.get(conversation) != value
        )
        self._acks = current
        # ... and each hint is applied ONLY to the change it names, because the
        # order above cannot cover the case where a hint outlives its own change:
        # a pass that read the receipt and consumed it while the nudge was still
        # in flight. Two terms, and both are needed. The conversation must be one
        # this read moved, or a spent hint would be subtracted from somebody
        # else's change (review round 1, m1); and the value must be the one this
        # read returned, or a hint survives a NEWER receipt on the same
        # conversation, excludes a device that did not cause that later read, and
        # — with every live device excluded — drops the correction outright
        # (review round 2, m2).
        moved = set(acked)
        exclude = {
            device
            for device, conversation, acknowledged in hints
            if conversation in moved and current.get(conversation) == acknowledged
        }
        # §3.2's exclusion, applied where it means something: the recipients are
        # the live devices MINUS the ones that just acted, so a correction
        # addressed at nobody is not sent at all — "an ack with no other device
        # emits nothing", because the only device that could render it already
        # has the state it would be asked to re-read. The change is still
        # consumed either way: ``self._acks`` has moved above, which is what
        # keeps the same ack from being emitted again on the next pass (the
        # detector state §3.1 says the nudge consumes).
        #
        # NOT GATED BY PRESENCE, deliberately (QA round 1, O-1). §2.3's presence
        # gate is a DEFERRAL for a completion banner, and it cannot apply to an
        # emit whose trigger IS the ack; this form carries no ``alert``, so a
        # desktop window being attended is no reason to leave every other device
        # stale. The ADR's own Q5/Q6 read it the same way — gate 1 (``unseen``)
        # cannot suppress a correction that exists because the read moved.
        #
        # ONE EMIT PER PASS, not one per changed conversation (QA round 1, O-2):
        # the form names no conversation and carries no record, so N frames would
        # be N identical silent wakes telling each device to do the same one
        # thing — re-read the machine — which is what §2.1's "at most one emit per
        # change" asks for. A shape that needs per-conversation addressing would
        # be a different emit type, not this one.
        recipients = [device for device in devices if device not in exclude]
        if acked and recipients:
            key = self._mint_emit()
            self._pending.setdefault(
                key,
                _PendingEmit(
                    key=key,
                    kind=EVENT_ATTENTION,
                    emit_id=uuid.uuid4().hex,
                    exclude=tuple(sorted(exclude)),
                ),
            )

    def _is_candidate(self, conversation: str) -> bool:
        """Whether a moved conversation needs the phone told about it.

        Gates 1 and 2 of §2.3, in order: ``unseen`` still true for the
        completion, and ``notify`` — the flag the store COMPUTES per run and the
        worker must never derive. Gate 3 (presence) is not asked here because its
        answer is a deferral rather than a decision, and gate 4 (a live device)
        is a property of the computer rather than of the row.
        """
        state = self.store.state(conversation)
        return bool(state.get("unseen")) and bool(state.get("notify"))

    def _add_completion(
        self,
        conversation: str,
        *,
        publications: tuple[int, ...] = (),
        supersedes: tuple[int, ...] = (),
        now: float,
        attended: bool,
    ) -> None:
        """Queue one completion emit, or defer it behind the presence gate.

        The key is composed HERE and not at send time, because §3.4's recipe is
        the record's CONTENT (``sha256(completion_token ‖ anchor_id ‖ kind)``):
        composing it early is what makes a heal mint a DIFFERENT key, so the
        correction is a new delivery that idempotency cannot swallow. A deferred
        item is moved to the back of the queue's attention by carrying
        ``deferred_since`` — the emit must be composed at the moment it is
        actually sent, because §3.2's ``count`` is "the machine's unread count at
        composition time".
        """
        state = self.store.state(conversation)
        token = state.get("completion_token")
        anchor = state.get("anchor_id")
        kind = state.get("kind")
        if not isinstance(token, str) or not isinstance(anchor, str) or not isinstance(kind, str):
            # A conversation whose state holds no completion cannot be described
            # on the wire, and there is nothing to retry: it is resolved so the
            # cursor moves rather than being pinned on a row the store cannot
            # name.
            self._resolved_publications |= set(publications)
            self._resolved_supersedes |= set(supersedes)
            return
        key = completion_emit_key(token, anchor, kind)
        existing = self._pending.get(key)
        if existing is None:
            existing = _PendingEmit(
                key=key,
                kind=EVENT_COMPLETION,
                conversations=(conversation,),
                # The machine's identity for this emit (§3.2), minted with the
                # item and reused by every retry: it is what the cloud's delivery
                # record is keyed on, so it must not move between attempts of one
                # emit.
                emit_id=uuid.uuid4().hex,
            )
            self._pending[key] = existing
        existing.publications += tuple(
            sequence for sequence in publications if sequence not in existing.publications
        )
        existing.supersedes += tuple(
            sequence for sequence in supersedes if sequence not in existing.supersedes
        )
        if attended and existing.deferred_since is None:
            existing.deferred_since = now

    # -- the emit ------------------------------------------------------------

    def _drain(self, now: float, attended: bool) -> list[EmitRecord]:
        """Send what may go now, oldest first, and close out what cannot.

        Order matters at the edges: the completions go before the badge
        correction, so a tick that carries both tells the phone about the work
        before it tells it to re-read.
        """
        records: list[EmitRecord] = []
        for _key, item in self._emissions(now, attended):
            # An earlier emit in this same pass may have resolved this item (the
            # ack path drops a whole batch), so membership is re-checked here.
            if item.key in self._pending:
                records.append(self._attempt(item, now))
        return records

    def _emissions(self, now: float, attended: bool) -> list[tuple[str, _PendingEmit]]:
        """The queue in the order it must leave, applying the burst rule.

        More than ``BURST_LIMIT`` eligible completions in ONE pass become a
        single digest rather than four banners (ADR §2.1's "catch-up is bounded
        and coalesced"; the same ceiling the desktop feed holds). The digest
        replaces the batch rather than following the first ``BURST_LIMIT`` of it,
        which is what "emits one digest push naming the count" asks for, and it
        is the difference between a phone that buzzes three times on a reconnect
        and one that buzzes once.

        A completion still inside its presence deferral is NOT eligible and does
        not count toward the burst: it is not a banner that was coalesced, it is
        a banner whose gate has not opened.

        EVERY kind goes through :meth:`_due`, the digest included. A digest is a
        first-class retryable item with its own key (§2.1's retry rule is one
        rule, not one per shape), and an attention correction that ignored the
        interval would spend the frozen budget roughly twenty times faster on the
        one emit whose key cannot be re-derived from the record (review round 1,
        m1).
        """
        ready_completions = [
            (key, item)
            for key, item in self._pending.items()
            if item.kind == EVENT_COMPLETION and self._due(item, now, attended)
        ]
        others = [
            (key, item)
            for key, item in self._pending.items()
            if item.kind in (EVENT_ATTENTION, EVENT_DIGEST) and self._due(item, now, attended)
        ]

        # Only items that have NEVER been on the wire may be folded into a
        # digest. An item already attempted carries a key the cloud may have
        # seen, and §2.1's retry policy is a retry with the SAME key — folding it
        # into a fresh digest would hand the cloud a second, different key for
        # one event, which is exactly the duplicate the ``Idempotency-Key``
        # exists to prevent.
        fresh = [(key, item) for key, item in ready_completions if item.first_attempt_at is None]
        if len(fresh) > BURST_LIMIT:
            digest = self._digest(fresh)
            if digest is not None:
                key, item = digest
                folded = {old_key for old_key, _member in fresh}
                for old_key in folded:
                    self._pending.pop(old_key, None)
                self._pending[key] = item
                remaining = [(key, item) for key, item in ready_completions if key not in folded]
                return [*remaining, (key, item), *others]
        ordered = sorted(
            ready_completions,
            key=lambda pair: min(pair[1].sequences) if pair[1].sequences else 0,
        )
        return [*ordered, *others]

    def _digest(
        self,
        members: list[tuple[str, _PendingEmit]],
    ) -> tuple[str, _PendingEmit] | None:
        """Fold a batch of eligible completions into ONE digest emit.

        THE THIRD TYPE, AND IT IS VISIBLE ON PURPOSE (ADR §2.1's burst rule, as
        the lane ruled it on 2026-10-01). §3.2 fixes the attention form's
        envelope as a silent, best-effort wake, so a coalesced catch-up riding it
        would arrive with no banner, no body and no count a user could read — the
        opposite of "one digest push naming the count". ``type: "digest"`` is
        the signal, and the ALERT object the machine composes (a house state plus
        the count) is the text the user reads.

        THE ID IS MINTED HERE, WHEN THE WINDOW CLOSES, and it is PERSISTED in the
        state file — a restart that re-folds this same window must wear the SAME
        key (§3.4 at ``5da35710``). That is what makes an attempt the
        same emit: a key re-derived per attempt would hand one batch a new
        identity on every pass, so a cloud that delivered but lost its ``202``
        could not dedupe, and the cursor would pin behind a batch that keeps
        wearing new names (review round 1, B1). The window's identity is the SET
        OF POSITIONS it speaks for; a batch that has grown is a different window
        and mints its own id, so a genuine second burst is never deduped away.

        A batch with no sequence behind it (only possible if two members share a
        key, which §3.4's content recipe makes a collision) is refused rather
        than emitted: an emit that would not move a cursor is an emit that would
        be re-sent forever.
        """
        publications = tuple(sequence for _key, item in members for sequence in item.publications)
        supersedes = tuple(sequence for _key, item in members for sequence in item.supersedes)
        if not publications and not supersedes:
            return None
        emit_id = self._window_emit_id(publications, supersedes)
        key = digest_emit_key(emit_id, self.computer)
        # The batch is formed from members that are all ready to go NOW, so the
        # digest inherits no deferral of its own: every member's gate has already
        # opened, and re-deferring the set would hold a push past the bound the
        # window exists to enforce.
        conversations = tuple(
            conversation for _key, item in members for conversation in item.conversations
        )
        return key, _PendingEmit(
            key=key,
            kind=EVENT_DIGEST,
            publications=publications,
            supersedes=supersedes,
            conversations=conversations,
            emit_id=emit_id,
        )

    def _window_emit_id(self, publications: tuple[int, ...], supersedes: tuple[int, ...]) -> str:
        """The emit id for this window: the persisted one when it IS this window.

        SAMENESS IS THE POSITION SET, exactly (§3.4). A restart mid-window
        re-reads the same rows behind the same cursor, so the persisted
        ``emit_id`` is reused and the retry wears the key the cloud may already
        have seen. A batch that has grown, or that is a later burst, is a
        DIFFERENT window and mints its own id — reusing the id there would let
        the cloud dedupe a delivery whose update the user never saw, which is the
        opposite failure.

        Persisted at mint time rather than at the end of the pass, because the
        wire attempt happens between the two: a process that dies after the cloud
        took the emit and before the pass closed would otherwise come back with
        no memory of which identity it used.
        """
        pending = self._cursors.digest_window
        if isinstance(pending, dict) and (
            pending.get("publications") == list(publications)
            and pending.get("supersedes") == list(supersedes)
        ):
            return str(pending["emit_id"])
        emit_id = uuid.uuid4().hex
        self._cursors.digest_window = {
            "emit_id": emit_id,
            "publications": list(publications),
            "supersedes": list(supersedes),
        }
        self._save()
        return emit_id

    def _due(self, item: _PendingEmit, now: float, attended: bool) -> bool:
        """Whether this item may be attempted NOW.

        Three cases, and they are the whole of the scheduling policy:

        - a completion still inside its deferral is due when the presence is
          gone, or when the window has expired — the hard bound that keeps a
          focused window from silencing the phone for good (§2.3). The OTHER way
          a deferral ends, an ack, never reaches here: a completion the user has
          read is no longer a candidate at all, so ``_collect`` has already
          discarded the item (see :meth:`_discard_positions`) and nothing is
          emitted for it;
        - an item that has never been attempted is due as soon as it exists;
        - anything already attempted retries on the wire interval, with the same
          key. An item that has exhausted its attempts is resolved by
          :meth:`_attempt`, so it never reaches this test again.
        """
        if item.deferred_since is not None:
            return not attended or now - item.deferred_since >= DEFERRAL_WINDOW_S
        if item.last_attempt_at is None:
            return True
        return now - item.last_attempt_at >= EMIT_RETRY_INTERVAL_S

    def _attempt(self, item: _PendingEmit, now: float) -> EmitRecord:
        """One wire attempt, and the bookkeeping the cursor depends on.

        A refusal does NOT move a cursor (§2.1: it advances only on the cloud's
        accept), which is what makes a refused push survive a restart. The item
        is retried with the same key until the attempts or the window are spent,
        and then it is dropped with ONE log line so the cursor can never be
        blocked forever by one undeliverable item.
        """
        if item.body is None:
            body = self._compose(item)
            if body is None:
                self._resolve(item)
                return EmitRecord(
                    kind=item.kind, idempotency_key=item.key, accepted=False, status=0
                )
            item.body = body
        if item.first_attempt_at is None:
            item.first_attempt_at = now
        item.attempts += 1
        item.last_attempt_at = now
        verdict = self.transport(dict(item.body), idempotency_key=item.key)
        if isinstance(verdict, EmitAccepted):
            self._resolve(item)
            return EmitRecord(kind=item.kind, idempotency_key=item.key, accepted=True, status=202)
        if item.attempts >= EMIT_RETRY_ATTEMPTS or (
            item.first_attempt_at is not None and now - item.first_attempt_at > EMIT_RETRY_WINDOW_S
        ):
            logger.warning(
                "push worker dropped a %s emit after %d attempts (last status %s): %s",
                item.kind,
                item.attempts,
                verdict.status,
                item.key,
            )
            self._resolve(item)
        return EmitRecord(
            kind=item.kind,
            idempotency_key=item.key,
            accepted=False,
            status=verdict.status,
        )

    def _compose(self, item: _PendingEmit) -> dict[str, Any] | None:
        """Build the §3.2 emit body for one item, or ``None`` if it cannot be.

        Composed at the first attempt rather than when the item was queued, so
        ``count`` is the machine's unread count at the moment of composition
        (§3.2's field table) — a deferred item may sit for five minutes, and a
        count taken when it was queued would be a number the phone reads as
        current.
        """
        count = self._count()
        emit_id = item.emit_id or uuid.uuid4().hex
        payload: dict[str, Any]
        if item.kind == EVENT_COMPLETION:
            conversation_identity = item.conversations[0] if item.conversations else ""
            session_id = _session_id_of(conversation_identity)
            handle = self._handle(session_id)
            state = self.store.state(conversation_identity)
            token = state.get("completion_token")
            anchor = state.get("anchor_id")
            kind = state.get("kind")
            if handle is None or not isinstance(token, str) or not isinstance(kind, str):
                return None
            if completion_emit_key(token, str(anchor), kind) != item.key:
                # The record moved on since the key was minted: this item's
                # content no longer exists, so its emit is stale by
                # construction. Resolved, not re-keyed — the move that changed
                # the content has its own cursor read and its own emit.
                return None
            payload = completion_payload(
                computer=self.computer,
                conversation=handle,
                completion_token=token,
                kind=kind,
                emit_id=emit_id,
                count=count,
            )
        elif item.kind == EVENT_DIGEST:
            # The coalesced catch-up: a set, so it names no conversation and
            # carries no record field — and its own TYPE, which is what makes it
            # a visible alert rather than the attention form's silent wake. The
            # alert's leading phrase is the SET's honest state, so the members'
            # kinds are read here (the batch is small by construction, and this
            # runs once per emit rather than once per pass).
            kinds: list[str] = []
            for conversation in item.conversations:
                member_kind = self.store.state(conversation).get("kind")
                if isinstance(member_kind, str):
                    kinds.append(member_kind)
            payload = digest_payload(
                computer=self.computer, count=count, emit_id=emit_id, kinds=kinds
            )
        else:
            # The badge correction: it stands for no record at all and carries a
            # count and nothing the app could deep-link to.
            payload = attention_payload(
                computer=self.computer,
                count=count,
                emit_id=emit_id,
                # ``None``, not ``[]``: §3.2 reads "absent means exclude nobody",
                # and an empty list would be a second spelling of it. A tick-
                # detected change has nothing here at all.
                exclude=list(item.exclude) if item.exclude else None,
            )
        if self._report_block is None:
            return payload
        return emit_body(payload, self._report_block())

    def _handle(self, session_id: str | None) -> str | None:
        """The conversation handle for one session id, or ``None``.

        Minted through ``push_handles`` so there is one spelling of the recipe —
        the push's ``conversation`` is the same handle the aggregate read serves
        and the deep link resolves — and a key this build cannot use leaves the
        item unresolvable rather than emitting a conversation the phone could
        never open. One bounded line and no ``exc_info``, the log hygiene the
        handle module states for its refusal.
        """
        if session_id is None:
            return None
        try:
            handles = push_handles.conversation_handles(self.config_dir, [session_id])
        except (push_handles.PushHandleKeyCorrupt, OSError) as exc:
            logger.warning(
                "push worker has no conversation handle (%s): %s",
                push_handles.key_path(self.config_dir),
                exc,
            )
            return None
        return handles[0] if handles else None

    # -- positions -----------------------------------------------------------

    def _resolve(self, item: _PendingEmit) -> None:
        """Close an item: it is accepted, or it is decided not to be sent.

        A digest takes its window's identity with it — §3.4 at ``5da35710`` clears
        it on the same three exits the completion path uses
        (the cloud's ``202``, the third failure, the drop-with-log), and this
        path adds its own: the gate that made the item moot. Leaving the id
        behind would let the NEXT burst wear a key the cloud has already seen,
        and its dedupe would swallow a real banner.
        """
        self._pending.pop(item.key, None)
        self._resolved_publications |= set(item.publications)
        self._resolved_supersedes |= set(item.supersedes)
        if item.kind == EVENT_DIGEST:
            self._cursors.digest_window = None

    def _advance(self, now: float) -> None:
        """Move each cursor over everything resolved behind its lowest blocker.

        The cursor is a POSITION, not a count of emits, so it can only move to
        the largest position such that everything at or below it is resolved.
        Nothing may be skipped over: an item still pending at position *n* holds
        the cursor at *n-1*, which is exactly what makes a deferred or refused
        push survive a restart (§2.1).
        """
        publications = [
            sequence for item in self._pending.values() for sequence in item.publications
        ]
        supersedes = [sequence for item in self._pending.values() for sequence in item.supersedes]
        # With nothing pending, the cursor moves to the newest position this pass
        # actually read — otherwise it would never move at all, the resolved set
        # above it would grow with the machine's history, and every tick would
        # re-scan the same rows.
        if publications:
            self._cursors.publication_cursor = min(publications) - 1
        else:
            self._cursors.publication_cursor = max(
                self._cursors.publication_cursor, self._seen_publication
            )
        if supersedes:
            self._cursors.supersede_cursor = min(supersedes) - 1
        else:
            self._cursors.supersede_cursor = max(
                self._cursors.supersede_cursor, self._seen_supersede
            )
        # Resolved-but-above-the-cursor bookkeeping is dropped below the cursor:
        # it exists only to stop a re-emit while a lower item is pending, so it
        # is bounded by the backlog rather than by the machine's history.
        self._resolved_publications = {
            sequence
            for sequence in self._resolved_publications
            if sequence > self._cursors.publication_cursor
        }
        self._resolved_supersedes = {
            sequence
            for sequence in self._resolved_supersedes
            if sequence > self._cursors.supersede_cursor
        }

    def _discard_positions(
        self,
        *,
        publications: tuple[int, ...] = (),
        supersedes: tuple[int, ...] = (),
    ) -> None:
        """Resolve positions that no longer need an emit, and the items over them.

        Two callers, one rule. A position read for the first time and found
        ineligible (``notify`` off, already read, no device to deliver to) never
        had an item, and resolving the position is the whole of it. A position
        whose PENDING item has just stopped being eligible — an ack landed while
        the item sat in its presence deferral — must take the item with it, or
        the deferral would keep a push alive for something the user has already
        read. That is ADR §2.3's "the deferral terminates early on an ack", and
        it is a DROP rather than an emit: pushing a completion the user has read
        is the noise the gate exists to prevent, so it is not logged as a loss.
        """
        self._resolved_publications.update(publications)
        self._resolved_supersedes.update(supersedes)
        for _key, item in list(self._pending.items()):
            if set(publications) & set(item.publications) or set(supersedes) & set(item.supersedes):
                self._resolve(item)

    def _mint_emit(self) -> str:
        """Mint the idempotency key the ATTENTION emit wears (ADR §3.4).

        The other two kinds are keyed elsewhere, each from what makes a retry the
        same emit: a completion on the record's CONTENT (``push_payload``), and a
        digest on the id minted when its coalescing window closed
        (:meth:`_digest`). The attention emit has neither — there is no record and
        no set — so its key is the machine's monotone sequence, PERSISTED as it
        is minted. That is the safer of the two possible orders: reserving the
        number before the attempt means two different emits can never wear one
        key, which is the property the cloud's dedupe rests on. The cost is
        stated rather than implied — an attention emit refused and then abandoned
        by a restart is not re-emitted under its old key, and it does not need to
        be: the badge is read from the machine and is right on the app's next
        read §1.5.
        """
        self._cursors.attention_sequence += 1
        self._save()
        return attention_emit_key(self._cursors.attention_sequence)

    # -- the machine-local gates --------------------------------------------

    def _live_device_ids(self) -> list[str]:
        """Gate 4 of §2.3: which devices this machine may deliver to.

        The ADR's own words are "the worker skips such a device entirely — no
        emit is attempted for it", and a computer whose every device is marked or
        expired therefore has nothing to emit FOR. This is a skip and not a
        deferral: a row held back for a device that does not exist yet would
        pin the cursor until one is registered, and "nothing from before
        enabling is ever pushed" is the rule the design already chose.

        The IDS rather than a count, because §3.2's ``exclude`` is subtracted
        from exactly this list: an excluded device is a device, so "are there
        devices?" is the wrong question once a nudge has named one — the right
        one is "is there one LEFT", and only the identities can answer it.

        An unreadable registry is treated as "no device is live" rather than
        raised: this runs on the ~2 s scan, and a per-tick traceback is the log
        noise the device registry warns about — the register and list routes
        already answer the operator loudly.
        """
        try:
            facts = mobile_push_devices.credential_facts(self.config_dir)
        except (mobile_push_devices.PushRegistryCorrupt, OSError):
            return []
        return [
            str(fact["device_id"])
            for fact in facts
            if fact.get("state") == mobile_push_devices.STATE_LIVE
            and bool(fact.get("credential_live"))
        ]

    def _count(self) -> int:
        """§3.2's ``count``: the machine's unread count at composition time."""
        return max(0, int(self._unread_count()))

    def _store_count(self) -> int:
        """The default unread count: a census over this machine's own store.

        INTERIM, and named as such: §1.4's badge population is the LISTING's rows
        in the listing's snapshot, and the daemon already has that implementation
        (``SessionTable.unread_block``, S1) — a caller that has it passes
        ``unread_count`` and the push's number matches the badge exactly. The
        census here is the same question asked of the store without the listing's
        origin filter, so it can only over-count (a subagent-only or scheduled
        conversation), never under-count, and it is what keeps this module
        runnable with no daemon at all. The read is O(completions) and happens
        only when an emit is composed.
        """
        acks = self.store.acknowledgement_map()
        newest: dict[str, int] = {}
        for row in self.store.published_since(0):
            conversation = str(row["conversation"])
            if not _is_conversation_identity(conversation):
                continue
            newest[conversation] = int(row["sequence"])
        return sum(
            1 for conversation, sequence in newest.items() if sequence > acks.get(conversation, 0)
        )

    # -- the durable position ------------------------------------------------

    # -- the queue -----------------------------------------------------------

    def pending(self) -> int:
        """How many emits are queued right now.

        The backlog behind the cursor IS the queue (ADR §2.1), so this number is
        the ADR's own boundedness claim made readable: a caller — or a test —
        asserts it is flat rather than reaching into the private map.
        """
        return len(self._pending)

    def pass_held(self) -> bool:
        """Whether a pass is running right now (the single-claimant guard's state).

        Read from INSIDE a pass, where the interesting fact is that the guard is
        held: a cell that only asserts a nested pass emitted nothing cannot tell
        the guard apart from any other reason it would have had nothing to do
        (review round 2, R2-1 — the re-entrancy cell was green with the guard
        removed, because the in-flight item was not due).
        """
        return self._pass.locked()

    def _save(self) -> None:
        """Atomic 0600 write of the cursors: same discipline as the registry.

        The whole file is written every time, and it is tiny, so there is no
        partial-write case to reason about; the replace plus the pre-replace
        chmod is what keeps a reader from seeing a half-written position or a
        briefly world-readable one.

        SKIPPED WHEN NOTHING MOVED. The tick runs every ``SCAN_INTERVAL_S``
        (~2 s) and most ticks change no position, so an unconditional write would
        be a rename per tick for no durability at all.
        """
        data = self._cursors.as_json()
        if data == self._written:
            return
        path = state_path(self.config_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{PUSH_WORKER_STATE_NAME}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self._cursors.as_json(), handle, separators=(",", ":"))
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, path)
            self._written = data
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise


def _desktop_attended() -> bool:
    """§2.3 gate 3: is a desktop window focused, visible and not minimised?

    The UNCACHED read, deliberately. ``desktop_presence``'s docstring reserves
    that for a caller whose decision is terminal, and this gate's decision is
    terminal in the direction that matters: a two-second-stale "the desktop is
    attended" defers a push the user has already walked away from, and the only
    thing that revisits it is the 5-minute window.

    The TUI viewer record is NOT consulted, and that is the ADR's own correction
    (review M3): "a TUI is running on this machine" says nothing about whether a
    phone in a pocket should be told.
    """
    return desktop_presence(cached=False).attended


def _is_conversation_identity(conversation: str) -> bool:
    """Whether a store key names a conversation the app can be shown.

    The identity is ``<namespace>/<id>`` (``attention.conversation_identity``),
    and only the ``session`` namespace is a conversation a phone lists:
    ``agent/<id>`` is the durable namespace the ADR's §1.2 population excludes.
    """
    return conversation.startswith("session/")


def _session_id_of(conversation: str) -> str | None:
    """The directory name behind a conversation identity, or ``None``.

    The mint takes a session id (``push_handles.conversation_handles`` builds
    ``session/<id>`` from it), so the inverse is a parsing job and not a
    filesystem one — which is what lets a push be composed for a conversation
    whose directory has since been archived.
    """
    namespace, _, session_id = conversation.partition("/")
    if namespace != "session" or not session_id:
        return None
    return session_id
