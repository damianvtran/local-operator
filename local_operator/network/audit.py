"""The audit log: append-only JSONL, one record per SEMANTIC event.

ONE RECORD PER SEMANTIC EVENT — never per frame, never per token delta (A7). A
membership change, a pairing, a handshake refusal, a rotation, a panic: those are
the facts worth reconstructing after an incident. A busy session moves thousands
of frames a minute and a healthy link produces zero interesting facts about each
one, so a per-frame log would make disk cost proportional to *traffic* rather
than to *event rate*, which is the prohibition this module exists to satisfy.

WHAT IS NEVER WRITTEN, in any encoding: the network secret, a link key, a device
private key, a wrapped key, a session ``control_key``, a token or a token prefix,
transcript content, prompt text, and any absolute path containing a username. Two
enforcement points, because one is a promise and two are a property:

* the per-event detail WHITELIST below — an unknown key is DROPPED, never raised
  on, because a lost audit record is worse than a dropped key; and
* ``_forbidden_key``, which drops the never-list even when a future whitelist
  entry names one by mistake.

ONE WRITER PER INSTALL — AND ONE LOCK PER WRITER. The relay process owns this
file; no other process opens it for append, and that is why the CLI's ``lop
network log`` only ever READS (tail/export). Within that process, however, the
writer is shared by EVERY relay thread — the accept loop, each handshake thread,
the control loop and the heartbeat all ``record`` or ``flush`` — and "no other
process" says nothing about those. Two of them used to read the same buffer and
each write it out, so the log carried the SAME record twice with an identical
``seq`` and ``ts``: measured, the cap-drop e2e test failed 3 of 30 isolated runs,
and it read as two cap refusals where the relay had recorded one. ``_write_lock``
makes the append and the drain one critical section, and the payload is taken off
the buffer BEFORE the file is opened, so a second writer in that window finds
nothing left to publish.

BATCHING, AND THE TWO WRITES THAT CANNOT WAIT. Records go into a byte buffer and
are flushed when it reaches ``BUFFER_BYTES`` or when a second has passed since the
last flush. A clock cut-off for an unflushed tail is deliberately NOT implemented
with a background thread: a flusher thread left running by a test or a CLI process
is a resource leak that outlives the command, and the relay already ticks (its
heartbeat calls ``flush``). Events whose loss to a power cut would be
unrecoverable — a panic, a removal, a rotation — are written through immediately.

RETENTION IS BY SIZE AND BY AGE with a fixed number of compressed generations, so
the steady-state footprint is bounded by construction rather than by tuning. The
numbers are ``mesh-incident-response.md``'s, because R18 requires them measured
and that document owns the measurement; they live here as module-level defaults
beside their reader.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_operator.network.store import audit_path, network_root

SCHEMA = "lop.mesh.audit.v1"

#: Bytes per generation (of the live file; a rotated generation is compressed, so
#: the on-disk footprint is smaller than this number times the generation count).
AUDIT_MAX_BYTES = 8 * 1024 * 1024
#: Compressed generations retained after the live file.
AUDIT_GENERATIONS = 5
#: Age cap. On a quiet install the age binds; on a busy one the size does, which is
#: why both exist rather than one.
AUDIT_MAX_AGE_DAYS = 90.0
#: The write batch, in bytes — a constant with a comment rather than a setting,
#: because a tunable here would only ever be turned the wrong way.
AUDIT_BUFFER_BYTES = 16_384
#: How long a record may sit in the buffer before the next call flushes it. A
#: bounded *opportunistic* flush rather than a timer thread (see the module note).
AUDIT_TICK_S = 1.0
#: The whole detail map, after serialisation. Longer is truncated to fit and gains
#: ``"truncated": true``.
MAX_DETAIL_BYTES = 2048
MAX_DETAIL_VALUE_CHARS = 200

#: The closed taxonomy. A drift guard test asserts every emitted event is in it,
#: so "an event nobody decided the shape of" cannot ship silently.
EVENT_KINDS: frozenset[str] = frozenset(
    {
        # transport scope (§7.6 of mesh-transport-identity.md)
        "pairing_refused",
        "invite_minted",
        "member_admitted",
        "member_removed",
        "member_left",
        # `lop network member grant/revoke`: what a peer may do on THIS device
        # changed. A membership-class event: an incident review asking "how could
        # that device move a session here" is answered by exactly this record.
        "member_capabilities_changed",
        # Learning a membership change from a peer is a MEMBERSHIP event: it is how
        # a member admitted after this device joined becomes visible here at all
        # (Q-R2-1), so an incident review that could not see it would read a
        # newcomer as having always been a member.
        "membership_learned",
        "device_rotated",
        "epoch_rotated",
        "epoch_conflict",
        "handshake_refused",
        "authorisation_refused",
        "link_opened",
        "link_closed",
        "link_idle",
        "link_replaced",
        "duplicate_identity",
        "self_link",
        # The inviter's human step (§5.3): the pairing is parked until a person on
        # the INVITING device confirms the code. That wait is a semantic event an
        # incident review needs to see — a pairing that waited and was never
        # answered is exactly the shape of an attempt nobody noticed.
        "pairing_awaiting_confirmation",
        "pairing_confirmed",
        "panic_raised",
        "panic_received",
        "trust_changed",
        "reconcile_granted",
        "reconcile_refused",
        # incident scope (mesh-incident-response.md §4.3), emitted by this writer
        "disconnect_initiated",
        "audit_rotated",
        "audit_pruned",
        # The credential broker (mesh-credentials.md; build plan §2.3). Five events,
        # because a lent bearer is a DELEGATION and an incident review has to be able
        # to reconstruct who could spend what, on whose account, from where.
        #
        # `act` is the owning device (the broker), `sub` is the borrowing device that
        # received the delegation; both are device ids, which is the RFC 8693
        # vocabulary mesh-credentials.md §1.4 names.
        "credential.grant",
        # A refusal: which device asked, for which key, and the machine code. The
        # person-facing sentence is rendered where the operator is, never stored here.
        "credential.grant_refused",
        # The owner refreshed because a peer's report asked it to. At most one per
        # credential per five minutes, and worth having: a burst is a borrower
        # hammering a failing login.
        "credential.refresh",
        # A borrower's attribution of a failure it observed with a borrowed bearer.
        # Deliberately NOT a verdict about the login — see `owner.py`'s report arm for
        # why routing this into `rotate_sibling` would log the operator out everywhere.
        "credential.report",
        # A membership-class event: who may borrow what changed here. "How could that
        # device spend my OpenAI account" is answered by exactly this record.
        "credential.placement",
    }
)

#: Closed machine enum for ``cause`` (mesh-incident-response.md §4.2). Prose lives
#: in the renderer, so a cause can be counted without parsing an English sentence.
CAUSES: frozenset[str] = frozenset(
    {
        "",
        "owner_offline",
        "not_a_holder",
        "revoked",
        "epoch_stale",
        "not_a_member",
        "not_authorised",
        "sas_mismatch",
        "wrong_device",
        "tick_expired",
        "capability_denied",
        "auth_failed",
        "replay",
        "untrusted",
        "reconcile_rate_limited",
        "protocol_mismatch",
        "duplicate_identity",
        # The pre-auth bound refused a connection: named as its own cause because the
        # operator's remedy differs from every other refusal here (nothing is wrong
        # with that peer; this device is saturated), and because a cause that is not
        # in this enum is written as "internal" — which would have reported a
        # deliberately-bounded relay as a relay with an internal fault.
        "handshake_cap",
        "policy",
        "timeout",
        "internal",
        # A human said no at the inviter's confirmation prompt, distinct from
        # `sas_mismatch` (the codes disagreed) and from `timeout` (nobody answered).
        "declined",
        "unanswered",
    }
)

#: Per-event ``detail`` whitelists. A key not listed for the event is dropped by
#: the writer, which is what makes "never key material" checkable rather than
#: aspirational.
DETAIL_KEYS: dict[str, frozenset[str]] = {
    "pairing_refused": frozenset({"cause", "subject"}),
    "pairing_awaiting_confirmation": frozenset({"subject", "role", "seconds_left"}),
    "pairing_confirmed": frozenset({"subject", "role", "answered_by"}),
    "invite_minted": frozenset({"role", "expires_at", "bound_device"}),
    "member_admitted": frozenset({"role", "member_kind", "epoch"}),
    # A device learns about a member it did not know about, from a peer that does.
    # This is a MEMBERSHIP change and it is how a newcomer becomes visible to the
    # devices that were already in the network (Q-R2-1), so it is recorded with the
    # rows it added and the table it produced.
    "membership_learned": frozenset({"source", "added", "members", "members_digest"}),
    "member_removed": frozenset({"initiated_by", "rekeyed", "epoch_after"}),
    "member_capabilities_changed": frozenset({"added", "removed", "capabilities", "initiated_by"}),
    "member_left": frozenset({"epoch"}),
    "device_rotated": frozenset({"old_device", "new_device"}),
    "epoch_rotated": frozenset({"epoch_before", "epoch_after", "rotation_id", "removed"}),
    "epoch_conflict": frozenset({"epoch", "rotation_id", "winner"}),
    "handshake_refused": frozenset({"cause", "their_epoch", "their_device", "mode"}),
    "authorisation_refused": frozenset({"op", "capability", "phase", "link_epoch"}),
    "link_opened": frozenset({"role", "epoch", "phase"}),
    "link_closed": frozenset({"cause", "frames_in", "frames_out"}),
    "link_idle": frozenset({"last_seen_at", "missed_beats"}),
    "link_replaced": frozenset({"instance_id", "age_s"}),
    "duplicate_identity": frozenset({"instance_id", "duplicate_count"}),
    "self_link": frozenset({"instance_id"}),
    "panic_raised": frozenset({"epoch_before", "epoch_after", "reachable_peers"}),
    "panic_received": frozenset({"from_device", "epoch_before", "epoch_after", "reason"}),
    "trust_changed": frozenset({"from", "to", "reason"}),
    "reconcile_granted": frozenset({"epoch_from", "epoch_to", "grants_used"}),
    "reconcile_refused": frozenset({"cause", "grants_used"}),
    "disconnect_initiated": frozenset({"epoch", "reachable_peers"}),
    "audit_rotated": frozenset({"generation", "bytes", "records"}),
    "audit_pruned": frozenset({"generation", "age_days"}),
    # -- the credential broker (mesh-credentials.md §2.3, DOC2 §4.3) -----------
    #
    # NOTE THE KEY NAME: `credential_key`, never `key`. `key` is in
    # FORBIDDEN_DETAIL_KEYS below because in this log it means KEY MATERIAL, and the
    # credential's NAME (`openai`) is not material — so the placement document may use
    # `key` for it while the audit log may not. Renaming here rather than relaxing
    # there is the whole reason both lists exist.
    #
    # `act`/`sub` are the delegation markers: act = the broker (owning) device,
    # sub = the device the grant was lent to. Both are device ids.
    "credential.grant": frozenset(
        {
            "credential_key",
            "act",
            "sub",
            "grant_id",
            "credential_kind",
            "scope",
            "refreshed",
            "latency_ms",
        }
    ),
    "credential.grant_refused": frozenset({"credential_key", "act", "sub", "code", "capability"}),
    "credential.refresh": frozenset({"credential_key", "act", "sub", "cause"}),
    "credential.report": frozenset({"credential_key", "act", "sub", "failure"}),
    "credential.placement": frozenset({"credential_key", "act", "sub", "owner_device", "holders"}),
}

#: Detail keys that are dropped on sight, whatever the whitelist says. The second
#: enforcement point: a future whitelist entry that names one of these is a
#: mistake, and a mistake must not be able to write key material into the log.
FORBIDDEN_DETAIL_KEYS: frozenset[str] = frozenset(
    {
        "secret",
        "secrets",
        "material",
        "private_key",
        "public_key",
        "device_key",
        "epoch_key",
        "invite_key",
        "link_key",
        "key",
        "mac",
        "sig",
        "signature",
        "token",
        "refresh_token",
        "access_token",
        "control_key",
        "transcript",
        "prompt",
        "text",
        "body",
        "password",
    }
)

#: Events whose loss to a power cut would be unrecoverable, so they are written
#: through instead of batched: an incident action, a membership change, a rotation.
#: Events flushed through immediately, because they are the ones an incident is
#: reconstructed FROM. Everything else is batched: a record that describes routine
#: liveness is not worth a syscall per frame, and losing the last second of
#: ``link_closed`` records to a crash costs nothing an investigator needs.
#:
#: The rule for adding one here: would its absence, in the last second before a
#: crash, hide an ATTACK or a MEMBERSHIP change? Refusals qualify — an attacker
#: probing a port wants exactly those records to be the ones that never landed.
DURABLE_EVENTS: frozenset[str] = frozenset(
    {
        "panic_raised",
        "panic_received",
        "disconnect_initiated",
        "trust_changed",
        "member_removed",
        "member_admitted",
        # A widened authority lost to a power cut would leave a device able to
        # move or delete sessions here with no record of who allowed it.
        "member_capabilities_changed",
        # Learning a membership change is the same class as being told one: an
        # operator investigating who could reach this device needs the moment a
        # newcomer became visible here, and a lost record would leave the mesh
        # looking as though that device had always been a member.
        "membership_learned",
        "epoch_rotated",
        "epoch_rejected",
        "handshake_refused",
        "authorisation_refused",
        "pairing_refused",
        # The inviter's human step (§5.3): the pairing is parked until a person on
        # the INVITING device confirms the code. That wait is a semantic event an
        # incident review needs to see — a pairing that waited and was never
        # answered is exactly the shape of an attempt nobody noticed.
        "pairing_awaiting_confirmation",
        "pairing_confirmed",
        "duplicate_identity",
        "self_link",
        "device_rotated",
        "audit_rotated",
        "audit_pruned",
    }
)

OUTCOMES = ("ok", "refused", "failed", "partial")
ACTOR_KINDS = ("human", "agent", "relay", "unknown")


def _resolve(name: str, explicit: Any, default: Any, cast: Any, root: Path | None = None) -> Any:
    """One bound: the caller's value, else the config store's, else the default.

    An unreadable or nonsensical configured value falls back to the default rather
    than raising. The audit log is the one component that must still work when
    everything else is broken, and refusing to open it because somebody typed a
    string into a number would lose exactly the records an incident needs.
    """
    if explicit is not None:
        return cast(explicit)
    from local_operator.network import store

    raw = store.read_config(("network", "audit", name), default, root)
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return default


_ROTATION_MARKER = ".gz"


@dataclass
class AuditEvent:
    """One semantic event. The transport's field list, plus the incident doc's
    convenience fields, because both documents describe one file."""

    event: str
    #: The acting device id; ``"self"`` for a local human action on this device.
    actor: str = "self"
    outcome: str = "ok"
    network_id: str = ""
    epoch: int | None = None
    subject: str = ""
    session_id: str = ""
    cause: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    network_name: str = ""
    actor_name: str = ""
    #: A HINT, not an attestation: from ``LOP_ACTOR`` when a harness sets it, else
    #: ``unknown``. Never presented as proof of who typed a command.
    actor_kind: str = "unknown"
    ts: float = field(default_factory=time.time)


class AuditLog:
    """Append-only, batched, bounded. The only writer of ``audit.jsonl``."""

    BUFFER_BYTES = AUDIT_BUFFER_BYTES
    TICK_S = AUDIT_TICK_S
    DURABLE_EVENTS = DURABLE_EVENTS

    @classmethod
    def from_config(cls, root: Path | None = None, *, enabled: bool = True) -> AuditLog:
        """The log this install's configuration asks for.

        The relay and the CLI both build their writer through here, so "which file
        did the writer open, against which cap" has exactly one answer.
        """
        return cls(root, enabled=enabled)

    def __init__(
        self,
        root: Path | None = None,
        *,
        max_bytes: int | None = None,
        generations: int | None = None,
        max_age_days: float | None = None,
        enabled: bool = True,
    ) -> None:
        """``None`` on a bound means "ask the config store, then the default".

        The three numbers follow the repo's configuration rule (AGENTS.md,
        "Adding a configuration key"): ``network.audit.max_bytes``,
        ``.generations`` and ``.max_age_days`` are registered in
        ``settings_io.SETTINGS``, the module constants below are the defaults the
        registry mirrors, and :meth:`from_config` is the one caller that reads
        them. A caller that passes an explicit value (a test, a probe) still wins.
        """
        self._root = root
        self._path = audit_path(root)
        self._max_bytes = _resolve("max_bytes", max_bytes, AUDIT_MAX_BYTES, int, root)
        self._generations = _resolve("generations", generations, AUDIT_GENERATIONS, int, root)
        self._max_age_s = (
            _resolve("max_age_days", max_age_days, AUDIT_MAX_AGE_DAYS, float, root) * 24 * 60 * 60.0
        )
        self._enabled = enabled
        #: Serialises append + drain. ``record`` and ``flush`` are called from every
        #: thread the relay runs (see the module docstring), so this is the one lock
        #: that keeps a record from being published twice; re-entrant because
        #: ``record`` flushes a DURABLE event from inside its own critical section.
        self._write_lock = threading.RLock()
        self._buffer: list[str] = []
        self._buffered_bytes = 0
        self._seq = _last_sequence(self._path)
        self._last_flush = time.monotonic()
        #: Set when a write fails. A failed AUDIT write must not stop the relay; it
        #: is reported by `lop network status` instead, so the operator learns the
        #: trail has a hole rather than discovering it during an incident.
        self.degraded = False
        self.degraded_reason = ""
        self.records_written = 0

    @property
    def max_bytes(self) -> int:
        """The per-generation cap this writer actually resolved."""
        return self._max_bytes

    @property
    def generations(self) -> int:
        """How many rotated generations are kept."""
        return self._generations

    @property
    def max_age_days(self) -> float:
        """The age bound, in days, as this writer resolved it."""
        return self._max_age_s / (24 * 60 * 60.0)

    # -- writing ------------------------------------------------------------

    def record(self, event: AuditEvent) -> None:
        """Append one event. Never raises: a lost record is worse than a dropped key.

        The failure mode is a ``degraded`` flag plus a line on stderr, the same
        shape §10.6 gives the writer: an audit write that fails is reported, not
        fatal, because a relay that dies when its log is full is a relay an
        attacker can kill with a full disk.
        """
        if not self._enabled:
            return
        # The render, the append and the flush are ONE critical section. ``_render``
        # stamps ``seq``, so two threads rendering at once could stamp the same number
        # onto two different records — the same corruption as a duplicated line, from
        # the other end (the module docstring has the measured account).
        with self._write_lock:
            line = self._render(event)
            self._buffer.append(line)
            self._buffered_bytes += len(line)
            self.records_written += 1
            if event.event in self.DURABLE_EVENTS:
                self.flush()
                return
            elapsed = time.monotonic() - self._last_flush
            if self._buffered_bytes >= self.BUFFER_BYTES or elapsed >= self.TICK_S:
                self.flush()

    def flush(self) -> None:
        """Write the buffer to disk, rotating first if the file is at its cap.

        THE BUFFER IS DRAINED BEFORE THE FILE IS OPENED, and both halves run under
        ``_write_lock``. Draining after the write (which is where this started) leaves
        a window in which a second thread — the heartbeat's own tick, or a reader's
        ``tail()``, both of which flush — reads the same buffer and publishes it
        again: one event, two lines, identical ``seq``. The audit trail is what an
        incident is reconstructed from, and "this happened twice" is a different
        story from "this happened once", so a caller that could not write still finds
        the records gone from the buffer rather than waiting to be written twice.
        """
        with self._write_lock:
            if not self._buffer:
                self._last_flush = time.monotonic()
                return
            payload = "".join(self._buffer)
            self._buffer.clear()
            self._buffered_bytes = 0
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                os.chmod(self._path.parent, 0o700)
                self._rotate_if_needed()
                if not self._path.exists():
                    # Created 0600 at birth: a mode applied after the first write is a
                    # window during which the trail is world-readable.
                    descriptor = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                    os.close(descriptor)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(payload)
                self._last_flush = time.monotonic()
                # Checked AGAIN after the write: a single flush can carry many records
                # (that is the point of batching), so checking only beforehand would
                # let one large batch overshoot the cap by however much it held.
                self._rotate_if_needed()
            except OSError as exc:
                self.degraded = True
                self.degraded_reason = str(exc)
                print(
                    f"warning: the network audit log could not be written: {exc}",
                    file=sys.stderr,
                )

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> AuditLog:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- reading ------------------------------------------------------------

    def tail(
        self,
        limit: int = 50,
        *,
        network_id: str | None = None,
        since: float | None = None,
    ) -> list[dict[str, Any]]:
        """The most recent records, oldest-first, filtered by network and time.

        FLUSHES FIRST: the writer may be holding the last second in its buffer, and a
        reader in the same process (``lop network log``, or a CLI mutation that audits
        and then reports) asking "what just happened" must see what this process
        already knows. Reads only the live file: the compressed history is for
        ``--export``.
        """
        self.flush()
        if not self._path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(record, dict):
                    continue
                if network_id and record.get("network_id") not in ("", network_id):
                    continue
                if since is not None and float(record.get("ts") or 0.0) < since:
                    continue
                out.append(record)
        return out[-limit:]

    # -- rendering ----------------------------------------------------------

    def _render(self, event: AuditEvent) -> str:
        self._seq += 1
        record: dict[str, Any] = {
            "schema": SCHEMA,
            "seq": self._seq,
            "ts": round(event.ts, 3),
            "ts_iso": _iso(event.ts),
            "event": event.event,
            "actor": event.actor or "self",
            "outcome": event.outcome,
            "cause": event.cause if event.cause in CAUSES else "internal",
        }
        if event.network_id:
            record["network_id"] = event.network_id
        if event.network_name:
            record["network_name"] = event.network_name
        if event.epoch is not None:
            record["epoch"] = event.epoch
        if event.subject:
            record["subject"] = event.subject
        if event.session_id:
            record["session_id"] = event.session_id
        if event.actor_name:
            record["actor_name"] = event.actor_name
        record["actor_kind"] = event.actor_kind if event.actor_kind in ACTOR_KINDS else "unknown"
        detail = _sanitize_detail(event.event, event.detail)
        if detail:
            record["detail"] = detail
        return json.dumps(record, separators=(",", ":"), ensure_ascii=False, sort_keys=True) + "\n"

    # -- retention ----------------------------------------------------------

    def _rotate_if_needed(self) -> None:
        if not self._path.exists():
            return
        size = self._path.stat().st_size
        age = time.time() - self._path.stat().st_mtime
        if size < self._max_bytes and age < self._max_age_s:
            return
        self._rotate(size)

    def _rotate(self, size: int) -> None:
        """Close the live file into ``audit.jsonl.<n>.gz`` and open a fresh one.

        Gzipped rather than kept raw: 8 MiB of JSONL is ~3:1 compressible, and the
        generation count times the cap is the number a person has to hold in their
        head when they ask "how much disk is this log using".
        """
        generation = self._next_generation()
        target = self._path.with_name(f"{self._path.name}.{generation}{_ROTATION_MARKER}")
        try:
            with self._path.open("rb") as source, gzip.open(target, "wb") as sink:
                while chunk := source.read(65536):
                    sink.write(chunk)
            os.chmod(target, 0o600)
            self._path.unlink()
        except OSError as exc:
            self.degraded = True
            self.degraded_reason = str(exc)
            return
        self._write_rotation_record(generation, size)
        self._prune_generations()

    def _next_generation(self) -> int:
        generations = _generations_on_disk(self._path)
        return (max(generations) + 1) if generations else 1

    def _write_rotation_record(self, generation: int, size: int) -> None:
        """The log records its own truncation — the one thing an attacker with
        disk access would most like to hide."""
        record = AuditEvent(
            event="audit_rotated",
            actor="self",
            actor_kind="relay",
            detail={"generation": generation, "bytes": size, "records": self.records_written},
        )
        line = self._render(record)
        try:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError as exc:
            self.degraded = True
            self.degraded_reason = str(exc)

    def _prune_generations(self) -> None:
        """Oldest first, by count and by age; each deletion is itself logged."""
        now = time.time()
        generations = sorted(_generations_on_disk(self._path))
        for index, generation in enumerate(generations):
            path = self._path.with_name(f"{self._path.name}.{generation}{_ROTATION_MARKER}")
            age_days = (now - path.stat().st_mtime) / 86400.0
            over_count = index < len(generations) - self._generations
            over_age = age_days > AUDIT_MAX_AGE_DAYS
            if not (over_count or over_age):
                continue
            try:
                path.unlink()
            except OSError:
                continue
            self._append_prune_record(generation, age_days)

    def _append_prune_record(self, generation: int, age_days: float) -> None:
        record = AuditEvent(
            event="audit_pruned",
            actor="self",
            actor_kind="relay",
            detail={"generation": generation, "age_days": round(age_days, 2)},
        )
        try:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(self._render(record))
        except OSError as exc:
            self.degraded = True
            self.degraded_reason = str(exc)


# ---------------------------------------------------------------------------
# Sanitising
# ---------------------------------------------------------------------------


def _sanitize_detail(event: str, detail: dict[str, Any]) -> dict[str, Any]:
    """Drop what is not whitelisted, then fit the map into its byte cap.

    Never raises on an unknown key: a lost audit record is worse than a dropped
    key, and the whitelist exists to bound what a record can say rather than to
    make an emitter's typo fatal at runtime.
    """
    allowed = DETAIL_KEYS.get(event, frozenset())
    out: dict[str, Any] = {}
    for key, value in detail.items():
        if not isinstance(key, str) or key in FORBIDDEN_DETAIL_KEYS:
            continue
        if key not in allowed:
            continue
        out[key] = _clean_value(value)
    if len(json.dumps(out, default=str)) <= MAX_DETAIL_BYTES:
        return out
    # Truncate by dropping keys from the end, keeping the flag, so a reader can
    # tell "this record is short" from "this record was cut".
    trimmed: dict[str, Any] = {}
    for key, value in out.items():
        candidate = dict(trimmed)
        candidate[key] = value
        candidate["truncated"] = True
        if len(json.dumps(candidate, default=str)) > MAX_DETAIL_BYTES:
            break
        trimmed[key] = value
    trimmed["truncated"] = True
    return trimmed


def _clean_value(value: Any) -> Any:
    """Bound a value's size, and refuse a control character in a string field.

    A delimiter-scanning reader must not have to define what happens when a payload
    contains the delimiter — the same rule the flat credential store states for its
    own fields.
    """
    if isinstance(value, str):
        text = "".join(ch for ch in value if ch >= " " or ch == "\t")
        return text[:MAX_DETAIL_VALUE_CHARS]
    if isinstance(value, bool) or isinstance(value, (int, float)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_clean_value(item) for item in list(value)[:16]]
    return str(value)[:MAX_DETAIL_VALUE_CHARS]


def _iso(ts: float) -> str:
    """UTC ISO-8601 with millisecond precision, derived from ``ts``.

    For humans and ``jq`` only: it is never parsed back, because a value that is
    both written and read is a second representation to keep honest.
    """
    seconds = int(ts)
    millis = int(round((ts - seconds) * 1000))
    if millis == 1000:
        seconds += 1
        millis = 0
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(seconds)) + f".{millis:03d}Z"


def _last_sequence(path: Path) -> int:
    """Resume the monotonic counter from the tail of an existing log.

    From the TAIL rather than from a state file: one fewer artifact to lose, and
    the only property that matters is that a new record's number is greater than
    every number already on disk.
    """
    if not path.exists():
        return 0
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 65536))
            tail = handle.read().decode("utf-8", errors="ignore")
    except OSError:
        return 0
    for line in reversed(tail.splitlines()):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and isinstance(record.get("seq"), int):
            return int(record["seq"])
    return 0


def _generations_on_disk(path: Path) -> list[int]:
    prefix = f"{path.name}."
    found: list[int] = []
    for candidate in path.parent.glob(f"{path.name}.*{_ROTATION_MARKER}"):
        stem = candidate.name[len(prefix) : -len(_ROTATION_MARKER)]
        if stem.isdigit():
            found.append(int(stem))
    return found


def events_logged(log: AuditLog, *, since: float | None = None) -> list[str]:
    """The event NAMES in the log — what a test asserts on, so a test never has to
    pin the whole record shape to ask "was this recorded"."""
    return [str(record.get("event")) for record in log.tail(limit=10_000, since=since)]


def write_event(
    root: Path | None = None,
    *,
    event: str,
    **fields: Any,
) -> None:
    """Convenience for a caller with no long-lived log: one event, flushed.

    Used by ``lop network`` commands that mutate state outside the relay process
    (they are not the log's writer, but a local human action is still a semantic
    event, and an unrecorded `member rm` is exactly the gap an incident review
    cannot afford). Flushed immediately: this process may exit in a millisecond.
    """
    log = AuditLog(root)
    log.record(AuditEvent(event=event, **fields))
    log.close()


def root_dir(root: Path | None = None) -> Path:
    """The network root — re-exported so callers of this module need one import."""
    return network_root(root)
