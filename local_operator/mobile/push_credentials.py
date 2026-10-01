"""Per-device credential facts, the lapse rule, and the COALESCED report.

Push/ack-sync S4c; ADR 0006 §4 rule 2 (the report, its carrier and the
derivation) and §3.2's report block.

**Why this module exists at all.** §4 rule 2 makes delivery depend on a
per-device flag — a registered device receives nothing while its credential is
dead — and the flag is the RELAY's to compute, because the relay is the only
component that sees the ``lop_mobile`` cookie. The flag has to reach a cloud the
machine does not otherwise talk to on a schedule, and the obvious implementation
(the one the ADR rejects by name) is to report it on every request: that is
O(requests) traffic from a phone polling every two seconds, and it is a
read-history stream by construction. So the fact is recomputed on every
authenticated request (``push_devices.note_credential``) and SENT coalesced:

- **change-triggered** — a device is reported when its facts differ from the ones
  last reported, never on a timer that repeats an unchanged value;
- **piggybacked** — a report rides the next ``register`` or ``emit`` for this
  computer, because both already talk to the cloud and a third call would be one
  more thing to fail;
- **throttled per device** — at most one report per device per
  :data:`MIN_REPORT_INTERVAL_S`. The throttle is a bound on when the CLOUD hears,
  never on when the machine acts: a rotation writes its markers immediately and
  the emitter skips a marked device on its very next tick, so a change can wait
  up to that window for its report (`MIN_REPORT_INTERVAL_S`, or
  :data:`HEARTBEAT_INTERVAL_S` if nothing else is due) and is never lost —
  :meth:`CredentialReport.note_rotation` states the same bound at the code
  (round 1, QA Q-F2);
- **heartbeated** — when neither carrier has fired for
  :data:`HEARTBEAT_INTERVAL_S`, a call of its own carries the whole block, which
  is what catches the device whose credential lapsed while its app was shut: no
  request exists to observe it, and the lapse is derived from the stored expiry
  the heartbeat re-reads;
- **batched per computer** — one report names every device that has something to
  say, and a rotation lapses all of them at once, which is exactly the case
  §4 rule 2 requires to be ONE credential-change event rather than one per
  device.

**The lapse rule lives here** (:meth:`CredentialFact.live_at`), and only here.
It is the ADR's round 7 Q-F15 derivation: a credential is live while the cookie
the device last presented has not died, and the instant is the COOKIE's — not a
timer, not the last-request stamp, and not a value the cloud can reason about. A
marker a rotation wrote outranks the arithmetic, because a rotation kills every
cookie while each of their nominal expiries is still days away.

...and it is asked THROUGH ``push_devices.credential_live_at`` rather than
restated (the delegation is in :meth:`CredentialFact.live_at`). One rule, one
home: the Settings list and the report both answer "live?" by calling it, which
is what review round 1's AR-1 restored after this branch briefly had the list
render the stored flag while the report derived.

**The wire is behind one seam, deliberately, and it is BOUND — not provisional.**
Every machine→cloud name this module needs is spelled once, in the constants
below, and those constants are the ADR §3.2 freeze at
``damianvtran/local-operator-mobile`` **``b03aeb1``** (§3.2's three literals and
its field table). The state machine, the bounds and the batching are testable
without a network because the transport is an injected callable: the tests pass a
recorder, and part 2 passes the cloud call. What is NOT in this module, by scope,
is S5's emission — the cursor, the queue, the retry — and the cloud-side pause;
these constants are what those bind to.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from local_operator.mobile import push_devices as mobile_push_devices

#: At most one report per DEVICE per window (ADR §4 rule 2's per-device ceiling).
MIN_REPORT_INTERVAL_S = 5 * 60

#: Nothing may stay unreported longer than this, change or no change — the
#: heartbeat's ceiling, and the bound on how long a device whose credential
#: lapsed while its app was shut can keep looking deliverable to the cloud.
HEARTBEAT_INTERVAL_S = 15 * 60

#: The carriers a report may ride. ``register`` and ``emit`` are the two the ADR
#: names (both already cross the machine→cloud boundary); ``heartbeat`` is the
#: call of its own that exists only when neither of them has happened.
CARRIER_REGISTER = "register"
CARRIER_EMIT = "emit"
CARRIER_HEARTBEAT = "heartbeat"

# The machine→cloud wire, frozen once. ADR §3.2's three literals each spell a
# route; §3.1 and the cloud ops note both freeze the emit's spelling and say in
# as many words that S3 must not freeze two, so these are copied from the freeze
# rather than paraphrased. The pin is `damianvtran/local-operator-mobile`
# `b03aeb1` and `tests/unit/mobile/test_push_credentials.py` asserts every string
# below against it, because a wire name is the one thing a later edit can change
# without anything here breaking.
#
# An in-flight sibling (S3, the interface-fixture branch) spells three of these
# as ``EMIT_ROUTE``/``IDEMPOTENCY_HEADER``/``PAYLOAD_VERSION`` for the same
# values; whichever of the two branches lands second should import from the
# other rather than keep a second copy, and the names are greppable against each
# other for exactly that reason.

#: THE ONE emit route (§3.2 literal 2, and the same route §3.1's completion and
#: attention emits already use). ``{tunnel_id}`` is substituted by the transport,
#: which is the only component that knows the tunnel.
REPORT_EMIT_ROUTE = "/v1/tunnels/{tunnel_id}/push/events"

#: The idempotency header that route carries (§3.1: ``Idempotency-Key: <emit key,
#: §3.4>``). One key per emit, and a retry of the same emit reuses it — the cloud
#: dedupes that into a duplicate ``202``, never a duplicate push.
REPORT_IDEMPOTENCY_HEADER = "Idempotency-Key"

#: The heartbeat's own call (§3.2 literal 3): the block and nothing else, and the
#: only carrier that can report a lapse to an app that is closed.
REPORT_HEARTBEAT_ROUTE = "/v1/push/credentials"

#: The registration forward (§3.2 literal 1). The block rides a route that
#: already exists rather than one of its own; named here because it is the third
#: carrier this block can travel on (the forward itself is S7's).
REPORT_REGISTER_ROUTE = "/v1/push/register"

#: The EMIT body's payload version (§3.2 literal 2 and its field table). It
#: belongs to the emit body, not to the block — the heartbeat and the register
#: forward carry the block WITHOUT it (§3.2 literals 1 and 3) — so the transport
#: that composes the emit owns it. Spelled here so every machine→cloud name lives
#: in one place rather than two.
REPORT_VERSION_FIELD = "v"
REPORT_VERSION = 1

#: The block's two keys, and the four fields of one row IN THE FROZEN ORDER —
#: §3.2's literals write ``device_id``, ``credential_live``,
#: ``credential_expires_at``, ``last_authenticated_at``, and the field table
#: argues each one. ``REPORT_DEVICES_FIELD`` keeps its name and value: the
#: in-flight S3 branch imports it from here.
REPORT_COMPUTER_FIELD = "computer"
REPORT_DEVICES_FIELD = "devices"
REPORT_DEVICE_FIELDS = (
    "device_id",
    "credential_live",
    "credential_expires_at",
    "last_authenticated_at",
)

#: The one credential-change event kind this part of the slice produces.
EVENT_ROTATION = "rotation"


@dataclass(frozen=True)
class CredentialFact:
    """One device's credential fact, and the MAPPING the one rule reads.

    ``credential_live`` is the value the relay computed when it last saw this
    device's cookie; :meth:`live_at` is that same answer asked at a LATER instant,
    which is the question the heartbeat asks about a phone that has not been heard
    from in a week. ``expired_at`` rides along because a marker a rotation wrote
    outranks any arithmetic — see the module docstring.

    Frozen because it is a fact about an instant, not a mutable handle: the
    coalescer compares two of these for equality to decide whether anything
    changed, so a fact mutated in place after it was recorded would silently
    erase its own change.
    """

    device_id: str
    state: str
    credential_live: bool
    credential_expires_at: int | None = None
    last_authenticated_at: int | None = None
    expired_at: int | None = None

    @classmethod
    def from_facts(cls, facts: Mapping[str, Any]) -> CredentialFact:
        """Build one fact from ``push_devices.credential_facts``' reduced row.

        The reduced row and not the stored record: the emit side has no business
        with a device's ``device_key``, and the one field this class does not read
        off it (``credential_live``) is already derived by the registry.
        """
        expires = facts.get("credential_expires_at")
        authenticated = facts.get("last_authenticated_at")
        expired = facts.get("expired_at")
        return cls(
            device_id=str(facts["device_id"]),
            state=str(facts["state"]),
            credential_live=bool(facts["credential_live"]),
            credential_expires_at=expires if isinstance(expires, int) else None,
            last_authenticated_at=authenticated if isinstance(authenticated, int) else None,
            expired_at=expired if isinstance(expired, int) else None,
        )

    def live_at(self, now: float) -> bool:
        """THE lapse rule at ``now`` — delegated, so it is one rule and not two.

        ``push_devices.credential_live_at`` owns the order (a standing marker
        outranks the arithmetic, then the expiry, then the stored flag); this
        class only asks it a later question, because the fact it holds was
        computed when the device was last seen and the heartbeat asks about now.
        """
        record: dict[str, Any] = {"credential_live": self.credential_live}
        if self.credential_expires_at is not None:
            record["credential_expires_at"] = self.credential_expires_at
        if self.expired_at is not None:
            record["expired_at"] = self.expired_at
        return mobile_push_devices.credential_live_at(record, now)

    def wire(self, now: float) -> dict[str, Any]:
        """One row of the report block, in the ADR §3.2 shape.

        ``credential_live`` is DERIVED here rather than echoed, so the value the
        cloud enforces against is this instant's answer and not the one the last
        request happened to compute. The two optional fields are omitted when the
        row does not carry them, which is the repo's absence rule and the same
        rule ``GET /api/push/devices`` already follows — a row with no credential
        fact at all is not the case this loop produces (a fact exists for a device
        the registry holds), but a row an earlier build wrote has no expiry, and
        inventing one would be a number the cloud could reason about that the
        machine never observed.
        """
        row: dict[str, Any] = {
            "device_id": self.device_id,
            "credential_live": self.live_at(now),
            "credential_expires_at": self.credential_expires_at,
            "last_authenticated_at": self.last_authenticated_at,
        }
        # Emitted THROUGH the frozen tuple, so the row this sends cannot drift from
        # the row §3.2 freezes even if the four names above are reordered: the
        # constant decides the order and the membership, and the dict above only
        # binds each name to its value. What the row does not hold is dropped
        # rather than sent as a null.
        return {name: row[name] for name in REPORT_DEVICE_FIELDS if row[name] is not None}


@dataclass
class CredentialEvent:
    """ONE credential-change event (ADR §4 rule 2).

    Mutable on purpose: the ADR's "one credential-change event per rotation" is a
    claim about how many of these a rotation produces, so the fields a caller
    fills in (which rotation, at what instant, naming which devices) have to be
    settable at the site that knows them. It is deliberately NOT the wire shape —
    §3.2's block carries devices and nothing else, and the event is the machine's
    own record of why a block went out.
    """

    kind: str
    at: int
    devices: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class _PendingEvent:
    """A recorded event, plus the devices it names that no report has carried yet.

    Private, and the reason it exists rather than the coalescer clearing all
    events on any send (review round 1, AR-4): a rotation names EVERY device,
    and the per-device throttle can split them across blocks, so "was this event
    carried?" is not a question about one block. It is carried once every device
    it names has ridden a report SINCE it was recorded — across however many
    blocks that takes — and until then it survives every block that does not
    finish the set. A device the registry no longer names is dropped from
    ``outstanding`` when the facts are re-read (it cannot ride anything again), so
    an event cannot be kept alive by a row that no longer exists.
    """

    event: CredentialEvent
    outstanding: set[str]


class CredentialTransport(Protocol):
    """Where a coalesced report goes.

    One block, one carrier name, no return value: the ADR makes the cloud's
    answer an acknowledgement the machine does not act on (a device may be
    offline for hours, and the badge is right on the app's next read), so the
    deferred binding needs no response shape to be designed here.
    """

    def __call__(self, block: dict[str, Any], *, carrier: str) -> None: ...


class CredentialReport:
    """The coalescing state machine. One per computer, held in the relay.

    ``clock`` is injectable because every bound in this class is a duration, and
    a test that sleeps for five minutes to prove a five-minute ceiling is a test
    nobody runs.
    """

    def __init__(
        self,
        transport: Callable[..., None],
        *,
        loader: Callable[[float], Sequence[CredentialFact]],
        computer: str,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """``loader`` reads the registry's current facts at a given instant.

        Injected rather than imported so the machine's clock and the store are
        both the test's, and so this class never reads ``push_devices``' file
        itself: it is a state machine over facts, and the module function that
        wires it to the registry is :func:`registry_loader`.

        ``computer`` is the envelope's opaque per-account handle (ADR §3.2) —
        injected for the same reason and one more: it is the one wire value this
        module cannot mint for itself, because it is the account's name for this
        machine rather than the machine's. It rides every block, not just the
        heartbeat's, because every §3.2 carrier carries it (AR-3).
        """
        self._transport = transport
        self._loader = loader
        self.computer = computer
        self._clock = clock
        self._facts: dict[str, CredentialFact] = {}
        self._reported: dict[str, CredentialFact] = {}
        self._reported_at: dict[str, float] = {}
        self._last_send_at: float | None = None
        self._pending: list[_PendingEvent] = []

    # -- observation ----------------------------------------------------------

    def refresh(self, *, now: float | None = None) -> tuple[str, ...]:
        """Re-read the registry and return the ids whose facts CHANGED.

        The hook every authenticated request lands on: the recompute itself is
        ``push_devices.note_credential``'s, and this is the coalescer learning
        what it produced. It sends nothing and writes nothing — which is the
        whole point of the split, because the ADR puts "recompute" on the request
        rate and "send" on this class's bounds.

        The facts are re-read from the store rather than taken from the caller,
        and that is deliberate: the heartbeat's case is a device whose credential
        lapsed with no request to observe it, and only a fresh read sees that.
        """
        stamp = self._clock() if now is None else now
        return self.observe(self._loader(stamp))

    def observe(self, facts: Iterable[CredentialFact]) -> tuple[str, ...]:
        """Record the computer's facts as a recompute just produced them.

        Returns the ids whose recorded fact MOVED — the change signal the
        change-triggered rule is built on, and the thing a caller can assert
        without reaching into the store to compare before and after.

        It REPLACES the recorded set rather than merging into it, and that is a
        correctness rule rather than a tidiness one: the caller hands over the
        whole registry, so a device the registry no longer names has left this
        computer — dropped as a dead token or by the cloud's idle drop — and a
        merge would keep offering it to the cloud for as long as the process
        lived. The throttle bookkeeping is pruned with it, so a machine that has
        seen many devices over its life holds one entry per device it still has.
        """
        incoming = {fact.device_id: fact for fact in facts}
        moved = tuple(
            device_id for device_id, fact in incoming.items() if self._facts.get(device_id) != fact
        )
        self._facts = incoming
        self._reported = {
            device_id: fact for device_id, fact in self._reported.items() if device_id in incoming
        }
        self._reported_at = {
            device_id: at for device_id, at in self._reported_at.items() if device_id in incoming
        }
        # A device the registry no longer names cannot ride a report again, so it
        # stops holding its event open. Without this a rotation naming a device
        # that was later dropped would leave its event pending for the life of the
        # process — the same "the loader is authoritative" rule the replacement
        # above applies to the facts.
        for item in self._pending:
            item.outstanding &= set(incoming)
        self._pending = [item for item in self._pending if item.outstanding]
        return moved

    def note_rotation(
        self, devices: Sequence[str], *, now: float | None = None
    ) -> CredentialEvent | None:
        """A relay-password rotation lapsed these devices: exactly ONE event.

        The bound is the ADR's ("one credential-change event per rotation", §4
        rule 2) and it is why this takes the whole set rather than being called
        once per device: a rotation kills every cookie at once, and a device-per-
        call API is how a caller ends up reporting N events for one action.
        ``devices`` is the CALLER's list — ``push_devices.rotate_credentials``'
        return value, which is the set the rotation actually wrote — and this
        method does not second-guess it; what the re-read below refreshes is the
        FACTS the following report will carry, not the event's roster (review
        round 1, AR-7: an earlier revision of this sentence claimed the re-read
        was what put the devices in the event).

        **The event is never lost, and the bound on its LATENCY is stated rather
        than implied** (round 1, QA Q-F2). Suppression is immediate: a rotation
        writes ``expired_at`` on every row, so ``credential_facts`` — the emitting
        side's only input — answers not-live for all of them at once, and the
        emitter (S5) skips them from its very next tick. What waits is the REPORT:
        the event rides the next block that is due, so it reaches the cloud within
        :data:`MIN_REPORT_INTERVAL_S` if some device is reportable, and within
        :data:`HEARTBEAT_INTERVAL_S` at the outside. That is ADR §2.2's coalescing
        rule ("the cloud sees O(devices) state, never O(requests) traffic") doing
        its job, not a dropped signal: the machine has already stopped delivering,
        and the report tells the cloud why.

        A rotation of a computer with no devices returns ``None`` and records
        nothing: there is no credential whose change the cloud could act on, and
        an event naming nobody would be a report about nothing.
        """
        stamp = int(self._clock() if now is None else now)
        self.refresh(now=stamp)
        if not devices:
            return None
        event = CredentialEvent(
            kind=EVENT_ROTATION,
            at=stamp,
            devices=tuple(devices),
        )
        self._pending.append(_PendingEvent(event=event, outstanding=set(event.devices)))
        return event

    # -- reporting -----------------------------------------------------------

    def pending_events(self) -> tuple[CredentialEvent, ...]:
        """The events no report has yet carried IN FULL (never on the wire).

        An event retires once every device it names has ridden a report since it
        was recorded — which for the ordinary rotation is the one block that
        carries them all, and for a rotation split by the per-device throttle is
        the block that completes the set (``_PendingEvent``).
        """
        return tuple(item.event for item in self._pending)

    def changed(self) -> tuple[str, ...]:
        """Device ids whose facts differ from the ones last reported."""
        return tuple(
            device_id
            for device_id, fact in self._facts.items()
            if self._reported.get(device_id) != fact
        )

    def due(self, *, now: float | None = None) -> bool:
        """True when at least one device may be reported under the throttle."""
        return bool(self._reportable(now))

    def heartbeat_due(self, *, now: float | None = None) -> bool:
        """True when nothing has been sent for the heartbeat window.

        An unknown last-send (a fresh process) counts as DUE. Silence is the
        failure mode this bound exists to close, so a machine that has just
        started reports rather than assuming the cloud already knows — the cost is
        one call per process start, and the alternative is a daemon whose first
        report is fifteen minutes late on the very launch that changed something.
        """
        stamp = self._clock() if now is None else now
        if self._last_send_at is None:
            return True
        return stamp - self._last_send_at >= HEARTBEAT_INTERVAL_S

    def piggyback(self, carrier: str, *, now: float | None = None) -> dict[str, Any] | None:
        """Take the register/emit opportunity: report iff the throttle allows.

        Returns the block that was sent, or ``None`` when nothing was due — which
        is the common case, and the reason a caller can put this on the launch
        path: a register that follows a quiet minute sends nothing at all.
        """
        stamp = self._clock() if now is None else now
        self.refresh(now=stamp)
        return self._flush(self._reportable(stamp), carrier, stamp)

    def heartbeat(self, *, now: float | None = None) -> dict[str, Any] | None:
        """The heartbeat call: the whole block, when nothing else has gone out.

        Forced past the per-device throttle rather than filtered by it: this call
        exists precisely for the device that was unchanged since it was last
        reported and whose credential has since lapsed, so a throttle that held it
        back would leave the cloud believing a dead credential is live until the
        next request that device never makes. The throttle bounds ``piggyback``,
        where a chatty app is the pressure; this is one call per fifteen minutes
        by construction.
        """
        stamp = self._clock() if now is None else now
        if not self.heartbeat_due(now=stamp):
            return None
        self.refresh(now=stamp)
        return self._flush(tuple(self._facts), CARRIER_HEARTBEAT, stamp)

    # -- internals -----------------------------------------------------------

    def _reportable(self, now: float | None) -> tuple[str, ...]:
        stamp = self._clock() if now is None else now
        allow = stamp - MIN_REPORT_INTERVAL_S
        return tuple(
            device_id
            for device_id in self.changed()
            if self._reported_at.get(device_id, allow - 1) <= allow
        )

    def _flush(self, device_ids: Sequence[str], carrier: str, now: float) -> dict[str, Any] | None:
        if not device_ids:
            return None
        carried = set(device_ids)
        block = {
            REPORT_COMPUTER_FIELD: self.computer,
            REPORT_DEVICES_FIELD: [self._facts[device_id].wire(now) for device_id in device_ids],
        }
        # The transport runs BEFORE any bookkeeping, and it is the only call here
        # that can raise: a send that did not go out must not look like one that
        # did, so a raise leaves every counter and every pending event exactly as
        # it found them (AR-4).
        self._transport(block, carrier=carrier)
        for device_id in device_ids:
            self._reported[device_id] = self._facts[device_id]
            self._reported_at[device_id] = now
        self._last_send_at = now
        # Only the events every device of which this block has now carried are
        # retired — see ``_PendingEvent``: a rotation names every device and the
        # per-device throttle can split them across blocks, so a block retires an
        # event only when it completes the set. Clearing everything here would let
        # some OTHER device's change in the same window drop the one
        # credential-change event the ADR requires, silently (AR-4).
        for item in self._pending:
            item.outstanding -= carried
        self._pending = [item for item in self._pending if item.outstanding]
        return block


def registry_loader(config_dir: Path) -> Callable[[float], Sequence[CredentialFact]]:
    """The report's loader over one computer's registry — READ-ONLY by construction.

    It goes through ``push_devices.credential_facts``, which cannot write, so the
    emit side has no writer to reach for; the returned callable is safe to hand to
    anything because calling it twice at the same instant is idempotent and
    calling it never is a no-op.
    """

    def load(now: float) -> Sequence[CredentialFact]:
        return [
            CredentialFact.from_facts(row)
            for row in mobile_push_devices.credential_facts(config_dir, now=now)
        ]

    return load


def block_devices(block: dict[str, Any]) -> list[dict[str, Any]]:
    """The device rows of a report block — for a caller that reads one back.

    Exists so a test (and part 2's transport) reads the rows through the same
    name the block was written with, instead of a literal ``"devices"`` that
    would survive a rename of :data:`REPORT_DEVICES_FIELD` by silently finding
    nothing.
    """
    rows = block.get(REPORT_DEVICES_FIELD)
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]
