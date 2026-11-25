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

MANY WRITERS, ONE NUMBERING — A FILE LOCK ACROSS PROCESSES. The relay process is
the file's main writer, but it is NOT the only one: approval transitions
(``approvals._audit``), ``write_event`` callers in the CLI and a second ``AuditLog``
in the same process each open their own writer and append. Each used to seed its
``seq`` once, at open, and append with no cross-process exclusion, so two writers
stamped the SAME number on different rows — and the P3 mentioned-file admission
(:meth:`AuditLog.record_mentioned_file_required`) refuses whenever its number is not
unique on disk, so a duplicate turned an approval into refused reads until the
relay's counter passed the other writer's. Every append now runs under an exclusive
lock on the sibling ``audit.jsonl.lock`` (:data:`AUDIT_LOCK_SUFFIX`): re-read the
tail's last ``seq``, renumber this writer's still-buffered rows to follow it,
rotate if due, append, and (durable) fsync, all as one critical section. Details
and the timeout contract: :meth:`AuditLog._acquire_file_lock`.

AND ONE THREAD LOCK PER WRITER. Within a process the writer is shared by EVERY relay thread — the accept loop, each handshake thread,
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

WHAT A READER OF THE FILE SEES FOLLOWS FROM THAT, and it is the thing an independent
verifier gets wrong. While the relay runs, ``audit.jsonl`` lags the log's own memory
by up to one second of batching, or up to one heartbeat (the relay's own interval,
15 s) when the machine is otherwise idle: a row recorded a moment ago is
appended to the buffer and is not in the file yet, so reading the file is reading the
last PUBLISHED state rather than the last event. Once the relay stops, nothing
drains the tail until the next record or a ``close()``. A reader in ANOTHER process
inherits exactly this lag — ``lop network log`` included, because it reads the same
file and cannot drain a buffer it does not own — so a verification that must see the
tail should trigger one more event, or stop the relay, before it counts rows.

TWO THINGS CUT THAT LAG DOWN, and the second is what makes it OBSERVABLE. A stream's
own rows publish at the moment the stream's state changes
(:data:`STREAM_LIFECYCLE_EVENTS`) rather than at the next tick, because "did the
release happen, and why" is the question an incident reader is asking about a pipe
that just ended, and 15 s of a row that exists being indistinguishable from a row
that does not is long enough to conclude a stream was never closed. And a writer
answers :meth:`AuditLog.publication_of` for any sequence number, so a reader that
finds nothing in the file can tell "recorded, not published yet" from "no such row"
instead of inferring one from the other. AN ANSWER NOBODY RENDERS IS NOT AN ANSWER,
so the same fact reaches the three surfaces a person actually looks at —
``lop network status``'s block, the TUI's ``/network`` panel and the agent's own
digest (``relay.audit_status_words``, one renderer for all three) — including the
case where the relay is running but not answering, where the surfaces say so in a
sentence rather than going quiet, because that is the state in which the reader is
most likely to be asking.

RETENTION IS BY SIZE AND BY AGE with a fixed number of compressed generations, so
the steady-state footprint is bounded by construction rather than by tuning. The
numbers are ``mesh-incident-response.md``'s, because R18 requires them measured
and that document owns the measurement; they live here as module-level defaults
beside their reader.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import stat
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_operator.network.store import audit_path

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
#: The cross-process append lock lives beside the log as ``audit.jsonl.lock``. A
#: sibling rather than a lock on ``audit.jsonl`` itself because rotation RENAMES the
#: live file away (``_rotate`` archives then unlinks it): a lock held on that inode
#: would no longer exclude a writer that opens the fresh file by name. The lock file
#: is never unlinked — unlinking a contended lock splits waiters across two inodes
#: (the hazard ``wakes/lock.py`` documents) — and ``_generations_on_disk`` globs only
#: ``audit.jsonl.<n>.gz``, so it is never mistaken for a generation.
AUDIT_LOCK_SUFFIX = ".lock"
#: How long one writer waits for another PROCESS's writer before giving up.
#: MEASURED on the reference Linux node (2026-11-25, base a9ecde898, scratch
#: script): the longest hold is a rotation of a full 8 MiB generation (gzip +
#: two fsyncs), n=5, median 30 ms, max 32 ms; a durable fsynced append is n=200,
#: median 2.8 ms, p95 3.1 ms, max 115 ms; the tail re-read is n=2000, median
#: 0.05 ms, max 0.44 ms. 5 s is ~40x the worst observed hold, which leaves room for
#: a slow disk's fsync under load, and short enough that a wedged peer cannot park
#: a relay thread (``record`` runs on the accept/handshake/heartbeat threads) for
#: longer than one heartbeat-scale pause. What a timeout MEANS differs per path —
#: see :meth:`AuditLog._acquire_file_lock` — but no path ever writes unlocked.
AUDIT_LOCK_WAIT_S = 5.0
#: Backoff between non-blocking attempts: start short (the common hold is ~3 ms),
#: cap low so a released lock is noticed within one fsync's time.
_AUDIT_LOCK_RETRY_S = 0.001
_AUDIT_LOCK_RETRY_MAX_S = 0.02
#: The whole detail map, after serialisation. Longer is truncated to fit and gains
#: ``"truncated": true``.
MAX_DETAIL_BYTES = 2048
MAX_DETAIL_VALUE_CHARS = 200

#: The closed taxonomy. A drift guard test asserts every emitted event is in it,
#: so "an event nobody decided the shape of" cannot ship silently.
EVENT_KINDS: frozenset[str] = frozenset(
    {
        # A move's working set (move-working-set.md §8.3): the owner served its
        # manifest, and a destination probed remotes at a peer's request.
        "session.workset.preflight",
        "session.workset.probe",
        # The owner accepted a working-set consent: which surface, which scope
        # digest, whether the files travel (mobility.AUDIT_WORKSET_CONSENT).
        "session.workset.consent",
        # A moved session's working set (move-working-set.md §8.3; workset_move).
        "session.workset.serve",
        "session.workset.restore",
        "session.workset.departed",
        # transport scope (§7.6 of mesh-transport-identity.md)
        "pairing_refused",
        "invite_minted",
        "member_admitted",
        "member_removed",
        "member_left",
        # THIS device learned, from a member that proved it on a live handshake, that
        # it was removed while it was offline (G9-B, ``relay._apply_verified_removal``).
        "membership_ended",
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
        # A device renaming ITSELF (identity set-name). Cosmetic, never authority —
        # which is why it is registered here and NOT added to DURABLE_EVENTS: a lost
        # rename costs a label, while the durable list is the events whose loss would
        # hide who could do what.
        "device_renamed",
        "epoch_rotated",
        "epoch_conflict",
        "handshake_refused",
        # A dialled endpoint answered with a POSITIVE wrong listener proof (wrong
        # device, bad signature, bad MAC) and was removed from the member row it
        # was dialled from (mesh-listener-auth §6, G4 heal-on-negative-proof).
        "endpoint_demoted",
        # A CONNECTION THAT DIED BETWEEN ITS HELLO AND ITS WELCOME (F4, drill
        # 2026-10-04): not a refusal — nothing decided anything — and the row that
        # answers "the relay saw it" when a join stops before any pairing record
        # exists. Bounded to connections that got past their hello, so port scans
        # leave nothing; ``cause`` carries the local socket class and
        # ``detail.stage`` the statement boundary it died at.
        "handshake_stopped",
        "authorisation_refused",
        "link_opened",
        "session_stream_opened",
        "session_stream_closed",
        "session_file_read_admitted",
        "mentioned_file_read_admitted",
        "link_closed",
        "link_idle",
        # Host clock-gap diagnostics and coalesced scheduler failures: technical
        # facts only, never raw exception messages or another credential path.
        "host_resumed",
        "redial_failed",
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
        # THE FAN-OUT'S OWN RECORD (mesh-incident-response.md §1.5). Three names, and
        # the third is the summary the operator's `lop network log` is read for:
        # "which peers got it" is a question about the PEERS, and `broadcast_to` — a
        # count of links written to — could not answer it (QA round 1, Q-R1-2).
        "panic_delivered",
        "panic_undelivered",
        "panic_broadcast_result",
        "trust_changed",
        "reconcile_granted",
        "reconcile_refused",
        # incident scope (mesh-incident-response.md §4.3), emitted by this writer
        "disconnect_initiated",
        "audit_rotated",
        "audit_pruned",
        # The credential broker (mesh-credentials.md; build plan §2.3). The events a
        # lent bearer's life produces, because that life is a DELEGATION and an
        # incident review has to be able to reconstruct who could spend what, on
        # whose account, from where.
        #
        # `act` is the owning device (the broker), `sub` is the borrowing device that
        # received the delegation; both are device ids, which is the RFC 8693
        # vocabulary mesh-credentials.md §1.4 names.
        "credential.grant",
        # Exact finite recovery, never future inheritance or credential material.
        # The refresh delegation's six states (§3.7f): each is a state CHANGE a
        # reader reconstructs a delegation from, and none carries material.
        "credential.refresh_delegated",
        "credential.refresh_taken",
        "credential.refresh_returned",
        "credential.refresh_refused",
        "credential.refresh_revoked",
        "credential.refresh_family_moved",
        # The automatic renewal lend (mesh-credential-renewal §B.7), owner side.
        # Ids and counts only: the material crosses inside link records, never here.
        "credential.renewal_lent",
        "credential.renewal_claim",
        "credential.renewal_returned",
        "credential.renewed_by_peer",
        "credential.renewal_conflict",
        "credential.renewal_stale",
        # The owner ended a renewal lend (revoke, removal, panic, leave: #3955 Q1).
        "credential.renewal_ended",
        # ``lop logout`` of a lent login: the provider-side revoke's outcome (§D D3).
        "credential.provider_revoked",
        "credential.finite_requested",
        "credential.finite_applied",
        # Operator-authored future-login permission (credentials/inheritance.py): the card
        # filed, and the permission recorded. Ids and the provider key only.
        "credential.inherit_requested",
        "credential.inherit_applied",
        # A credential card signed, refused before signing, or applied FROM THE DESKTOP
        # (credentials/desktop_approval.py). The signature row names its ASSURANCE --
        # `presence` (an OS prompt proved a person) or `file_only` (the operator's 0600 key
        # signed; no person is proven) -- so an incident review can tell the two apart.
        "credential.desktop_signed",
        "credential.desktop_sign_refused",
        "credential.desktop_applied",
        "credential.desktop_apply_refused",
        # A refusal: which device asked, for which key, and the machine code. The
        # person-facing sentence is rendered where the operator is, never stored here.
        "credential.grant_refused",
        # The BORROWER's receipts for the offline hold (credentials/client.py
        # ``_held_or``, operator-approved 2026-11-13): a grant served from this
        # device's memory while its owner was unreachable, coalesced to one record
        # per grant per minute with a `serves` count, and the hold's end. Written on
        # the borrowing device (the owner is asleep: it cannot record them), ids and
        # counts only, never the token.
        "credential.held_served",
        "credential.held_lapsed",
        # A SUBAGENT session first using its ancestor session's held GitHub grant
        # while the owner was away (credentials/client.py ``_bridge_ancestor_hold``,
        # 2026-11-21): names the child as the consumer and the ancestor it came from,
        # so the serve is never silently attributed to the parent. Ids only.
        "credential.held_bridged",
        # The BORROWER's own record that a dial did not get a grant (credentials/client.py
        # ``_note_borrow_failed``): the 2026-11-17 owner_busy burst was visible only in
        # the owner's log. Coalesced per (owner, key) per window, codes only.
        "credential.borrow_failed",
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
        # The mint-revoke half of the credential broker (github adapter): one row per
        # revoke batch the owner's lender performed — the scheduled window-end DELETE,
        # the operator's immediate revoke, and the retry after a failed call.
        "credential.revoke",
        # A member removal pruned every placement row the removed device held
        # (mesh-mcp-credentials.md §4): key NAMES only, plus whether the prune
        # itself failed (a removal must proceed regardless, so the failure has to
        # be visible somewhere).
        "credential.revoke_all",
        # THE SYNC ENGINE (mesh-consent-provisioning.md §5): a copy served,
        # refused or applied is the same delegation life the grant rows record,
        # one lifecycle further on. `gen` names the per-key generation the row
        # is about, so a reader can line a holder's ack up against a serve.
        "credential.copy",
        "credential.copy_refused",
        "credential.copy_applied",
        # A recipient's lifecycle ack (credential lifecycle L1): the owner recorded
        # that `sub` acknowledged the named material generation / login-chain
        # revision. Counters and keys only, never material.
        "credential.lifecycle_acked",
        # The owner's sender: a recipient APPLIED (or already held) the named
        # counters. `sub` is the recipient. Counters and the key only.
        "credential.lifecycle_announced",
        # The consent reconciler (credentials/reconcile.py): a holder the owner DERIVED
        # from a signed approval, and a candidate it refused (with the machine code).
        # Keys, device ids and a code only — never material, never an approval body.
        "credential.permission_projected",
        "credential.permission_refused",
        # A derived holder taken back because its approval is gone, and a push the
        # recipient could not take (with the machine reason). Keys/ids/codes only.
        "credential.permission_withdrawn",
        "credential.permission_undelivered",
        # The github narrowing's delivery receipts (the adapter's §3.4 bound):
        # the owner's designated repositories reached a borrowing device and
        # were written to its config — or were refused, with the machine reason.
        # `act` is the borrowing device, `sub` the owner they came from.
        "credential.narrowing_applied",
        "credential.narrowing_refused",
        # THE COPY'S ENDINGS (S4, review round 1 M1): a wipe the owner performed
        # or confirmed (`copy_wiped`, with the row count a reader uses to see a
        # batch from a single key), and the operator's selection mark on a store
        # secret (`marked` — the standing policy word that decides whether the
        # next device copies it at all). Both were emitted unregistered, which
        # the sanitizer answers by dropping the WHOLE detail silently; the guard
        # in test_audit.py now derives emitter literals from source so a third
        # one cannot ship the same way.
        "credential.copy_wiped",
        "credential.marked",
        # THE APPLY ROWS the taxonomy drifted on (M1's guard, deriving emitters
        # from source, found these three the same way it found the two above):
        # all three were emitted unregistered, so the sanitizer dropped their
        # whole detail — which definitions applied and which conflicted, and why
        # a background warm failed — since they landed. Registered so the
        # guard's answer and the taxonomy agree; the detail keys below are the
        # ones the emitters pass, read from them, not invented.
        "definitions_applied",
        "mcp_defs_applied",
        # The mesh skill sync's receiver (``skillsync.py``): one skill per apply,
        # named under the outcome it got — never its content.
        "skills_applied",
        "session.create.warm_failed",
        # THE REMOTE ONBOARDING APPROVAL (remote-onboarding §2.4). The lifecycle
        # events for ONE durable record: the operator's single gesture that lets an
        # agent install this product and the operator anchor on a remote device (or
        # bootstrap the operator's own anchor locally — the same events, `kind`
        # tells them apart). Membership-CLASS: `onboard_connected` is the moment a
        # device could start appearing in the mesh, and `onboard_approved` is the
        # authority decision that led there.
        "onboard_requested",
        "onboard_approved",
        "onboard_denied",
        "onboard_expired",
        "onboard_connected",
        "onboard_failed",
        # A run that stopped reporting was superseded by a retry: the receipt that
        # lands says ``step=superseded``, and this row is how the record's history
        # shows the gap between an approval that ran and the run that replaced
        # it (drill finding, 2026-10-04).
        "onboard_superseded",
        # The REQUESTER's own settle for an un-actioned request (drill finding,
        # same day): no operator gesture rides it, and an incident reader must be
        # able to tell it apart from a deny (the operator said no) and from an
        # expiry (the window lapsed on its own).
        "onboard_withdrawn",
        # A decision-scoped run (a phone pairing) whose runner stopped reporting
        # INSIDE its window settles ``failed`` with a ``step=stale`` receipt before
        # the fold may expire it (pairing design v2 §2.1). A row of its own because
        # an incident reader must tell "the run died after the operator approved"
        # apart from `onboard_failed` (a step reported failure) and from
        # `onboard_expired` (no decision inside the window). Terminal and routine,
        # so it stays batched like its two neighbours.
        "onboard_run_stale",
        # THE PHONE PAIRING APPROVAL (S4a-pair-op, ``operator/pair_op.py``). One row per
        # semantic step of ONE ``operator_device_pair`` record, and the reason the family
        # exists at all: a pairing is the one approval that issues a STANDING signer, so an
        # incident review asks of it in particular "which phone, under which key, and was a
        # PERSON proven". `desktop_signed` names the assurance (``presence`` -- an OS prompt
        # per signature -- or ``file_only``, a 0600 file signed with no person proven), which
        # is the disclosure the operator accepted (a-2a8b); `desktop_ack` names the typed code
        # check and how many strikes it has taken (a wrong code costs one, and the third
        # closes the card as a deny); `desktop_applied` is the certificate landing in the
        # device store, and `desktop_apply_refused` is why it did not. `desktop_confirm`
        # is the CONFIRM-mode decision ("Codes match — Approve") on an ATTESTED card:
        # the Radient edge, not a typed code, proved the filer, so the row names that
        # edge (`via`) in place of a strike count.
        "device_pair.desktop_ack",
        "device_pair.desktop_confirm",
        "device_pair.desktop_signed",
        "device_pair.desktop_applied",
        "device_pair.desktop_apply_refused",
        # MESH ROLLING UPDATES, member side (mesh-update-propagation.md §6). One
        # row per semantic change of one update request: the request was admitted
        # by the authoriser, a rule refused it, or the install began/ended. A
        # per-poll row is exactly what this set must NOT become — the split
        # between these five names is what keeps a busy member's re-asks
        # countable without writing one row per ask.
        "update_requested",
        "update_refused",
        "update_started",
        "update_completed",
        "update_failed",
        # S3a of the rolling updates: the ORIGIN's side of a rollout. Three
        # kinds, one per semantic change — the record opening (S3b's hook
        # writes it through the record API), a member's state changing, and
        # the pass closing (complete, lapsed or cancelled). A per-poll row is
        # exactly what these must NOT become; a member handled by the cadence
        # produces no rows between its changes (note §6).
        "update_rollout_started",
        "update_member_state",
        "update_rollout_done",
        # S3b's suppression row: an update moved but the switch (``off`` or
        # ``--no-roll``) stopped the roll before any record opened. Registered
        # here with the other three because ``_sanitize_detail`` whitelists by
        # ``DETAIL_KEYS`` — an unregistered name renders with NO detail,
        # silently.
        "update_rollout_suppressed",
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
        # A memory handler refused a batch itself (`bad_memory`): a move whose
        # source tombstone would erase an unseen local revision, or an invalid
        # payload. Registered so the refusal reads as itself rather than as
        # "internal" — that substitution is what made the round-2 silent stall
        # invisible in the audit file too.
        "bad_memory",
        "timeout",
        "internal",
        # A member answered ``busy`` and the pass parked it (S3a): a deferral is
        # its own cause because its remedy is time, not repair — the same reason
        # ``declined`` below is not ``not_authorised``.
        "busy",
        # A human said no at the inviter's confirmation prompt, distinct from
        # `sas_mismatch` (the codes disagreed) and from `timeout` (nobody answered).
        "declined",
        "unanswered",
        # WHY A FORWARDED STREAM CLOSED (Q-XH-2's close record). Four countable
        # classes, not one per caller: a viewer-ended stream, one the peer relay
        # ended, one whose owning runtime went away, and one the link could not
        # carry. The FINER word — which of the viewer shapes it was, whether the
        # owner's socket hit EOF or a write error — rides `detail.cause`, which is
        # whitelisted for these events and is where an incident reader looks for
        # the specific story. Deliberately four and not nine: this is the COUNTING
        # surface ("a cause can be counted without parsing an English sentence"),
        # and a lifecycle's worth of hyphenated words inside a snake_case enum of
        # refusal reasons would be a second vocabulary in the same field.
        "viewer_left",
        "peer_closed",
        "owner_gone",
        "peer_unreachable",
        # The LOCAL class of a connection that died in a handshake or a pairing
        # (F4, drill 2026-10-04): a sealed record that failed against this side's
        # own keys, and everything else the OS said went wrong. Countable here so
        # an incident reader can tell a crypto failure apart from a peer that
        # merely departed, without parsing a sentence.
        "link_crypto",
        "io",
    }
)

#: Per-event ``detail`` whitelists. A key not listed for the event is dropped by
#: the writer, which is what makes "never key material" checkable rather than
#: aspirational.
DETAIL_KEYS: dict[str, frozenset[str]] = {
    # ``kind``/``stage`` ride the SOCKET-CLASS rows only (F4, drill 2026-10-04):
    # ``kind`` is the codec's local class (auth/sequence/parse/limit) and
    # ``stage`` the statement boundary the read was at. Both are LOCAL facts,
    # never sent to any peer.
    "pairing_refused": frozenset({"cause", "subject", "kind", "stage"}),
    # The relay's own row for a connection that got past its hello and then died
    # before the welcome: stage + mode + addr say WHERE, and ``kind`` distinguishes
    # the codec's classes for the one case that has them.
    "handshake_stopped": frozenset({"stage", "mode", "their_addr", "their_device", "kind"}),
    "pairing_awaiting_confirmation": frozenset(
        {
            "subject",
            "role",
            "seconds_left",
            "offer_sent",
            "offer_digest",
            "offer_skip",
            "offer_enumeration",
        }
    ),
    "pairing_confirmed": frozenset({"subject", "role", "answered_by", "shares"}),
    "invite_minted": frozenset({"role", "expires_at", "bound_device"}),
    "member_admitted": frozenset({"role", "member_kind", "epoch", "grants"}),
    # A device learns about a member it did not know about, from a peer that does.
    # This is a MEMBERSHIP change and it is how a newcomer becomes visible to the
    # devices that were already in the network (Q-R2-1), so it is recorded with the
    # rows it added and the table it produced.
    "membership_learned": frozenset({"source", "added", "members", "members_digest"}),
    "member_removed": frozenset({"initiated_by", "rekeyed", "epoch_after"}),
    "member_capabilities_changed": frozenset({"added", "removed", "capabilities", "initiated_by"}),
    "member_left": frozenset({"epoch"}),
    # ``recorded`` appears only as ``False``: the drop ran but the self-tombstone
    # could not be saved (relay ``_apply_verified_removal``, #4394 review R2-2).
    "membership_ended": frozenset({"notice_epoch", "credentials_dropped", "recorded"}),
    "device_rotated": frozenset({"old_device", "new_device"}),
    "device_renamed": frozenset({"previous", "name", "networks", "failed"}),
    "epoch_rotated": frozenset({"epoch_before", "epoch_after", "rotation_id", "removed"}),
    "epoch_conflict": frozenset({"epoch", "rotation_id", "winner"}),
    "endpoint_demoted": frozenset({"endpoint", "code"}),
    "handshake_refused": frozenset(
        {
            "cause",
            "their_epoch",
            "their_device",
            "mode",
            # WHERE IT CAME FROM, AND WHY IT IS NOT NAMED (Q-R1-4). `their_device` is
            # written only when the peer PROVED that id; `declared_device` is what it
            # claimed, `expected_device` is who this device dialled, and
            # `unidentified` is the reason the actor is unknown — because a bare
            # `actor: unknown` beside an ephemeral socket port is unreadable.
            "their_addr",
            "declared_device",
            "expected_device",
            "unidentified",
            # How many identical refusals the relay folded into this row since the
            # last one it wrote (``network.redial.RefusalCoalescer``). Absent = 0.
            "repeats",
        }
    ),
    # Handler refusals share this event with chokepoint denials. Keep their
    # original typed code even when the closed cause vocabulary says "policy";
    # the reason is scrubbed before recording and bounded by _clean_value.
    "authorisation_refused": frozenset(
        {"op", "capability", "phase", "link_epoch", "refusal_code", "reason"}
    ),
    "link_opened": frozenset({"role", "epoch", "phase"}),
    # A VIEWER'S PIPE, BOTH HALVES. `session_stream_opened` is emitted by whichever
    # relay opened the stream and by the one that accepted it, so a leak is visible
    # as a count that does not balance (the cross-host round read "5 opened, 0
    # closed" on the owner, which is how the unreleased dial was found — Q-XH-2).
    # The close's `cause` is what tells the four ways a viewer goes away apart.
    "session_stream_opened": frozenset({"stream", "peer", "role"}),
    "session_stream_closed": frozenset({"stream", "peer", "role", "cause"}),
    # Owner-observed admission, NOT transfer completion. No file contents or
    # caller-asserted identity belong here; actor is the authenticated peer.
    "session_file_read_admitted": frozenset(
        {"reference", "admitted_bytes", "offset", "decision", "phase"}
    ),
    "mentioned_file_read_admitted": frozenset(
        {"schema", "operation", "binding", "read_binding", "source_stamp", "nonce"}
    ),
    "link_closed": frozenset({"cause", "frames_in", "frames_out"}),
    "link_idle": frozenset({"last_seen_at", "missed_beats"}),
    "host_resumed": frozenset({"gap_s"}),
    "redial_failed": frozenset({"device", "code", "attempt", "next_retry_at", "repeats"}),
    "link_replaced": frozenset({"instance_id", "age_s"}),
    "duplicate_identity": frozenset({"instance_id", "duplicate_count"}),
    "self_link": frozenset({"instance_id"}),
    "panic_raised": frozenset({"epoch_before", "epoch_after", "reachable_peers"}),
    "panic_received": frozenset(
        {"from_device", "epoch_before", "epoch_after", "reason", "rotation"}
    ),
    # One row per peer, and the OUTCOME is the point: `acked` means the peer applied
    # it, `unacked` means it still holds the old epoch, `refused` means it answered
    # and said no. The `reason` is the peer's own sentence, never re-derived here.
    "panic_delivered": frozenset({"outcome", "reason"}),
    "panic_undelivered": frozenset({"outcome", "reason"}),
    "panic_broadcast_result": frozenset({"sent", "acked", "unacked", "failed"}),
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
            # Which OWNER row was lent (an int id, never an email), whether the
            # borrower's affinity anchor chose it, and the selection policy the
            # lend ran under: the per-turn record that remote load-balancing
            # matches a local session's (first pick, stickiness, failover).
            "credential_id",
            "anchored",
            "policy",
            # Owner-local display metadata (a label may intentionally be email),
            # not another identity or candidate list. Reason is a closed scalar
            # basis:ordinal within the winning eligible tier, never global best.
            "account_label",
            "ranked_reason",
        }
    ),
    # -- the refresh delegation (mesh-refresh-delegation.md §3.7f) ------------
    #
    # Six records, one per state change the design names, and every allowlist is
    # NON-MATERIAL BY CONSTRUCTION: ids, a key NAME, an epoch and a cause. The
    # material never appears — not because an emitter is careful, but because the
    # writer drops any key outside an event's allowlist AND drops FORBIDDEN_DETAIL_KEYS
    # even if a future allowlist names one (T11 asserts both halves).
    #
    # `act` is the device that ACTED and `sub` the other device, matching
    # `credential.grant`'s existing shape: a delegation is always about two devices,
    # and a reader reconstructing one from the log needs to know which end each
    # record is speaking from.
    "credential.refresh_delegated": frozenset(
        {"credential_key", "act", "sub", "grant_id", "epoch", "cause", "approval_id"}
    ),
    "credential.refresh_taken": frozenset({"credential_key", "act", "sub", "grant_id", "epoch"}),
    "credential.renewal_lent": frozenset({"credential_key", "act", "sub", "epoch"}),
    "credential.renewal_claim": frozenset({"credential_key", "act", "sub", "outcome"}),
    "credential.renewal_returned": frozenset(
        {"credential_key", "act", "sub", "epoch", "renewals", "last_at"}
    ),
    "credential.renewed_by_peer": frozenset({"credential_key", "act", "sub", "epoch", "at"}),
    "credential.renewal_conflict": frozenset({"credential_key", "act", "sub", "epoch", "code"}),
    "credential.renewal_stale": frozenset({"credential_key", "act", "sub", "epoch", "code"}),
    "credential.renewal_ended": frozenset({"credential_key", "act", "sub", "cause"}),
    "credential.provider_revoked": frozenset({"credential_key", "provider", "outcome"}),
    "credential.refresh_returned": frozenset({"credential_key", "act", "sub", "grant_id", "epoch"}),
    "credential.refresh_refused": frozenset({"credential_key", "act", "sub", "grant_id", "cause"}),
    "credential.refresh_revoked": frozenset({"credential_key", "act", "sub", "grant_id", "epoch"}),
    "credential.refresh_family_moved": frozenset(
        {"credential_key", "act", "sub", "grant_id", "epoch", "cause"}
    ),
    "credential.finite_requested": frozenset(
        {"credential_key", "act", "sub", "scope", "approval_id"}
    ),
    "credential.finite_applied": frozenset(
        {"credential_key", "act", "sub", "scope", "approval_id"}
    ),
    "credential.inherit_requested": frozenset({"credential_key", "act", "sub", "approval_id"}),
    "credential.inherit_applied": frozenset({"credential_key", "act", "sub", "approval_id"}),
    "credential.desktop_signed": frozenset(
        {"surface", "assurance", "operation", "key_id", "disclosure_digest"}
    ),
    "credential.desktop_sign_refused": frozenset(
        {"surface", "operation", "code", "authority_level"}
    ),
    "credential.desktop_applied": frozenset({"surface", "operation"}),
    "credential.desktop_apply_refused": frozenset({"surface", "operation", "code"}),
    # `reason` is the refusal's arm marker (`holder_excluded`, `pool_depleted`) when it
    # has one: a fixed vocabulary word, never an id or label (#3066 follow-up Q7).
    "credential.grant_refused": frozenset(
        {"credential_key", "act", "sub", "code", "capability", "reason"}
    ),
    # `act` is the OWNER whose login was served, `sub` this borrowing device (the
    # credential.grant vocabulary). `hold_until` is the hold end (epoch ms) and
    # `bound` which limit set it (token|hold|cap|grant); `serves` counts the
    # offline serves this record covers, `pending` those not yet in a served row.
    "credential.held_served": frozenset(
        {"credential_key", "act", "sub", "grant_id", "hold_until", "bound", "serves"}
    ),
    "credential.held_lapsed": frozenset(
        {"credential_key", "act", "sub", "grant_id", "serves", "pending", "bound"}
    ),
    # `consumer` is the session that used the grant, `bridged_from` the ancestor
    # session whose held grant it was (both opaque session ids), `grant_id` the
    # ORIGINAL grant, `hold_until` that grant's (never extended) hold end.
    "credential.held_bridged": frozenset(
        {"credential_key", "act", "sub", "grant_id", "consumer", "bridged_from", "hold_until"}
    ),
    # `code`/`reason` are the owner's machine answer (never its sentence), `held`
    # whether this device's held grant bridged it, `retry_after_ms` the owner's
    # stated wait (0 = none), `repeats` the rows folded since the last one written.
    "credential.borrow_failed": frozenset(
        {
            "credential_key",
            "act",
            "sub",
            "provider",
            "code",
            "reason",
            "held",
            "retry_after_ms",
            "repeats",
        }
    ),
    "credential.refresh": frozenset({"credential_key", "act", "sub", "cause"}),
    "credential.report": frozenset({"credential_key", "act", "sub", "failure"}),
    "credential.placement": frozenset(
        {"credential_key", "act", "sub", "owner_device", "holders", "skipped"}
    ),
    # The mint-revoke contract's other half (github adapter, F4): one row per
    # revoke batch — the scheduled window-end DELETE, the operator's immediate
    # revoke, and the retry after a failed call. `cause` tells the three apart,
    # `revoked`/`deferred` are counts (tokens, never material), and the act/sub
    # delegation markers ride here exactly as they do on `credential.grant`.
    "credential.revoke": frozenset(
        {"credential_key", "act", "sub", "cause", "revoked", "deferred"}
    ),
    # The member-removal prune: `keys` is the list of placement keys stripped of
    # the removed device (names, never values), and `failed`/`reason` record a
    # prune that could not run without blocking the removal itself.
    "credential.revoke_all": frozenset({"keys", "failed", "reason"}),
    # The copy path's rows (mesh-consent-provisioning.md §5): what the owner
    # SERVED (`credential.copy`), why it refused to serve (`copy_refused`, with
    # the machine reason — the digest pin's mismatch is the one that matters),
    # and what a member APPLIED (`copy_applied`). Same act/sub delegation
    # markers as `credential.grant`; `gen` is the generation transferred.
    "credential.copy": frozenset({"credential_key", "act", "sub", "gen"}),
    "credential.copy_refused": frozenset({"credential_key", "act", "sub", "reason"}),
    "credential.copy_applied": frozenset({"credential_key", "act", "sub", "gen"}),
    # The lifecycle ack's two integers are the generation and login-chain revision
    # the recipient acknowledged (owner.py's ack arm) — counters, never material.
    "credential.lifecycle_acked": frozenset(
        {"credential_key", "material_generation", "login_chain_revision"}
    ),
    "credential.lifecycle_announced": frozenset(
        {"credential_key", "material_generation", "login_chain_revision"}
    ),
    "credential.permission_projected": frozenset({"credential_key", "act", "sub"}),
    "credential.permission_refused": frozenset({"credential_key", "act", "sub", "code"}),
    "credential.permission_withdrawn": frozenset({"credential_key", "act", "sub", "code"}),
    "credential.permission_undelivered": frozenset({"credential_key", "act", "sub", "code"}),
    # The narrowing's delivery (the github adapter's §3.4 bound): `repositories`
    # is the LIST APPLIED (a config value, never material); a refused delivery
    # carries the machine reason instead.
    # `digest`/`generation` name WHICH owner designation landed (authoritative
    # deliveries only), so out-of-order rows can be read in owner order.
    "credential.narrowing_applied": frozenset(
        {"credential_key", "act", "sub", "repositories", "digest", "generation"}
    ),
    "credential.narrowing_refused": frozenset({"credential_key", "act", "sub", "reason"}),
    # `rows` is the deleted-row count on the member's own store — never the rows
    # themselves — so an incident review can tell a confirmed wipe from a notice
    # that matched nothing (`rows: 0`).
    "credential.copy_wiped": frozenset({"credential_key", "act", "sub", "rows"}),
    "credential.marked": frozenset({"credential_key", "mark"}),
    "definitions_applied": frozenset({"installed", "updated", "conflicts", "refused"}),
    # `withdrawn`: mirrors removed because their origin turned the server off
    # (mcpdefs' explicit-withdrawal exception) — names only, like the rest.
    "mcp_defs_applied": frozenset({"installed", "updated", "withdrawn", "conflicts", "refused"}),
    "skills_applied": frozenset({"installed", "updated", "conflicts", "refused"}),
    # `detail` here is the warm failure's own sentence (a string), not a nested map.
    "session.create.warm_failed": frozenset({"peer", "detail"}),
    # -- the remote onboarding approval -------------------------------------------
    #
    # `kind` is the record's own enum (device_onboard / local_authority);
    # `key_id`/`spki_fp` are PUBLIC fingerprints of the anchor key the decision
    # authorised (the anchor itself is public data) — never material, and neither
    # spelling collides with FORBIDDEN_DETAIL_KEYS. `step`/`run_id` describe the
    # receipt a runner transition carried, so an incident reader can tie the
    # audit line to the record's own fold.
    "onboard_requested": frozenset({"kind", "request_id", "surface", "device_id", "host"}),
    "onboard_approved": frozenset({"kind", "key_id", "spki_fp"}),
    "onboard_denied": frozenset({"kind"}),
    "onboard_expired": frozenset({"kind"}),
    "onboard_connected": frozenset({"kind", "run_id"}),
    "onboard_failed": frozenset({"kind", "step", "run_id"}),
    "onboard_superseded": frozenset({"kind", "run_id"}),
    # The requester's own settle: `surface`/`session_id` name the filer that
    # withdrew (the store only ever accepts the filer), so the row is
    # self-explaining without reading the record.
    "onboard_withdrawn": frozenset({"kind", "surface", "session_id"}),
    # The stale-run settle carries only the record's `kind`: the run id and the
    # ``stale`` step live on the record's receipt, which is the fold's own copy.
    "onboard_run_stale": frozenset({"kind"}),
    # The phone pairing approval (``operator/pair_op.py``). `device_id` is DERIVED from the
    # phone's public key (public data, the same string the certificate and the revocation
    # list carry). `strikes`/`ok` are the typed-code check's own outcome, `assurance` the
    # level the signature actually reached, `operation` the record's ``what`` key, and
    # `code` a refusal token -- never a sentence, never key material, and never the six
    # digits themselves (they are compared in constant time and are never stored).
    "device_pair.desktop_ack": frozenset({"surface", "ok", "strikes", "device_id"}),
    # `via` names the attestation that stood in for the typed code
    # (``radient:<edge version>``) -- a fixed token, never the account or a token.
    "device_pair.desktop_confirm": frozenset({"surface", "device_id", "via"}),
    "device_pair.desktop_signed": frozenset(
        {"surface", "operation", "device_id", "key_id", "assurance"}
    ),
    "device_pair.desktop_applied": frozenset({"surface", "device_id", "run_id", "lifetime_days"}),
    "device_pair.desktop_apply_refused": frozenset({"surface", "device_id", "code"}),
    # Mesh rolling updates, member side. `rollout` is the origin's record id when
    # one exists (empty for the single-peer verb — S1 carries it through so the
    # S3 driver's rows need no schema change). `target` is the VERSION asked for,
    # `code` the refusal token this module's own register names, `method` the
    # install shape on the done/failed pair, and `from` the version the member
    # was on when the install began. No file paths, no installer output beyond
    # the bounded tail that rides the REPLY (a sentence for the origin), never
    # the log.
    "update_requested": frozenset({"rollout", "target", "capability"}),
    "update_refused": frozenset({"rollout", "target", "code"}),
    "update_started": frozenset({"rollout", "target", "method"}),
    # ``serve``/``serve_stale``/``serve_remedy`` (2026-10-28): whether this
    # device's `lop serve` came along with the install — the state token, the
    # still-stale daemons as short ``pid@version`` strings, the alive-but-not-
    # serving ones as ``pid@verdict`` (#3093 QA Q1), and the command that moves
    # them. Flat, because this log's values are bounded scalars and lists
    # (``_clean_value`` stringifies a dict). Built by ``meshupdate.serve_audit``.
    "update_completed": frozenset(
        {
            "rollout",
            "target",
            "method",
            "from",
            "version",
            "serve",
            "serve_stale",
            "serve_stuck",
            "serve_remedy",
        }
    ),
    "update_failed": frozenset({"rollout", "target", "method"}),
    # The origin's side (S3a): the opening row carries WHO decided (a closed
    # token, never a sentence), under which policy and front end; the member
    # row carries the state it changed to plus the member's own words when
    # they exist; the closing row carries the counts in one compact string
    # (the log's values are bounded short strings, and the pass's leftovers
    # are what a reader needs).
    "update_rollout_started": frozenset(
        {"rollout", "authority", "policy", "via", "target", "from", "members"}
    ),
    # ``left_busy`` (#3028, QA round 2 Q8): the ids a generation-layout member
    # deliberately left on its old build — the receipt's sentence is capped, so
    # the names it carries can be lost. A list of short ids, bounded by count at
    # the emitter (``rollouts.LEFT_BUSY_CAP``) and here by ``_clean_value``.
    "update_member_state": frozenset(
        {
            "rollout",
            "state",
            "code",
            "reason",
            "method",
            "version",
            "left_busy",
            "serve",
            "serve_stale",
            "serve_stuck",
            "serve_remedy",
        }
    ),
    "update_rollout_done": frozenset({"rollout", "counts"}),
    # The suppression row (S3b): which front end moved (``via``), what the key
    # said (``policy``) and the per-run override when it was the flag that
    # stopped the roll (``flag``, empty when the policy did).
    "update_rollout_suppressed": frozenset({"via", "policy", "flag"}),
    # A move's working set (move-working-set.md §8.3). Counts, host names and
    # fixed result codes only — never a path, a URL path, userinfo or git output.
    "session.workset.preflight": frozenset({"peer", "worktrees", "planned", "bytes", "excluded"}),
    "session.workset.probe": frozenset({"peer", "hosts", "results", "refused"}),
    # ``surface`` is a closed token (mobility.CONSENT_SURFACES, or "unattributed"),
    # CLAIMED by the requesting surface and not verified: attribution, never proof;
    # ``digest`` the scope digest the gate matched (a hash of the manifest, never a
    # path); ``carry_decision`` "carry"/"stay"; ``lease_epoch`` binds the row to the
    # move it gated.
    "session.workset.consent": frozenset(
        {"peer", "surface", "consent", "digest", "carry_decision", "lease_epoch"}
    ),
    # A moved session's working set (design move-working-set.md §8.3). Bounded
    # metadata only — ids, sizes, one home-relative path, reason codes — never
    # bytes or file contents. ``serve`` is the required row written before the
    # first byte of each artifact (and before each resumed serve) leaves.
    # ``kind``: ``span`` (bytes served from ``offset``) or ``prefix`` (the sha256 of the
    # first ``offset`` bytes, answered for a resume; security review #4245 S4).
    "session.workset.serve": frozenset(
        {"peer", "to_device", "lease_epoch", "artifact", "bytes", "offset", "path", "kind"}
    ),
    "session.workset.restore": frozenset({"peer", "state", "worktrees"}),
    "session.workset.departed": frozenset({"peer", "marked"}),
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
        # A content admission lost to a crash leaves disclosed session bytes
        # unattributable. Each independently authorized fetch is an admission,
        # not transport telemetry; ordinary frames remain unlogged/batched.
        "session_file_read_admitted",
        "mentioned_file_read_admitted",
        # A moved session's working-set bytes leave this device only after this row
        # is on disk (``workset_move._audit_workset_serve``, design §8.3): an egress
        # an incident cannot reconstruct is the loss this table exists for.
        "session.workset.serve",
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
        # Losing membership on a removed device's own word-of-a-member: as durable as
        # the removal it records, or an incident review cannot see why bearers ended.
        "membership_ended",
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
        # The onboarding approval's DECISIONS and its arrival: an authority
        # decision lost to a power cut would leave an installation with no record
        # of who allowed it (the `member_capabilities_changed` argument).
        # `onboard_requested`/`onboard_expired`/`onboard_failed` stay batched —
        # routine lifecycle that a crash's last second need not preserve.
        "onboard_approved",
        "onboard_denied",
        "onboard_connected",
        # A desktop credential signature is the same authority decision as
        # `onboard_approved`, plus the assurance it was made at.
        "credential.desktop_signed",
        # The mesh update request and its by-rule refusals are AUTHORITY
        # decisions ("may this peer move my install"): a widened standing grant
        # whose request landed with no record would leave a device explaining an
        # install nobody can attribute. The begin/end rows stay batched — a crash
        # in their last second narrates nothing an incident needs.
        "update_requested",
        "update_refused",
        # The record's OPENING row is an authority decision too (S3a/S3b): it
        # names that the primary decided to roll under the standing instruction
        # (or an operator command), and losing it would leave a fleet of moved
        # members explaining an update nobody can attribute — the exact
        # argument `update_requested` is here for. The member-state and closing
        # rows stay batched: their history is reconstructable from the record
        # and the members' own durable rows.
        "update_rollout_started",
    }
)

#: Events that are PUBLISHED at their own moment rather than at the next tick: the two
#: halves of a viewer's pipe.
#:
#: WHY THESE AND NOT OTHERS. A stream's open and close answer the one question an
#: incident reader asks about a pipe that just ended — did the release happen, and why
#: — and until it is published, a row that EXISTS is indistinguishable in the file from
#: a row that does not. Measured on the pre-fix head over three runs, the owner's close
#: row took 14.74 / 14.72 / 14.73 s to reach ``audit.jsonl``: one full 15 s heartbeat,
#: because the close lands just after a flush and nothing else on an otherwise idle
#: relay drains the buffer. The bound the cell used to carry was 20 s, which is 1.24x
#: the product's own publication latency — a cell that cannot tell a working relay from
#: one whose publication had doubled (the lane's own test-robustness round, where this
#: cell's bound was cut to 5 s: the round that found it, not an agent-review round).
#:
#: PUBLISHED, NOT SYNCED, AND NOT PER ROW. This is a ``flush`` — ONE append-open and one
#: ``.write()`` carrying the whole batch — and never an ``fsync``: nothing here claims to
#: survive a power cut, which is the whole of what :data:`DURABLE_EVENTS` is for. The
#: count is of CALLS and not of syscalls (agent review round 3, NIT 1: a batch past the
#: text layer's 8 KiB buffer decomposes into more than one ``write(2)`` while
#: ``write_calls`` moves by one, so nothing here is a syscall count, and the rate claim
#: does not need one). The cost is bounded by the STREAM
#: COUNT and by nothing else: a stream opens and closes when a viewer attaches and
#: leaves, both human-scale events, so an idle relay writes nothing at all (no timer
#: fires when nothing happened) and a busy one pays at most two extra writes per stream
#: rather than one per row. That is the operator's own bound for this subsystem —
#: "proper logs and forensics with sane retention but not too much disk IO" — and it is
#: why neither of the two rejected shapes is here: a per-row write-through would put the
#: disk cost back on traffic, and a timer would write on an idle machine.
STREAM_LIFECYCLE_EVENTS: frozenset[str] = frozenset(
    {"session_stream_opened", "session_stream_closed"}
)

OUTCOMES = ("ok", "refused", "failed", "partial")
#: Who an event's actor IS, as far as this log can say. A CLOSED set, because an
#: unlisted value is rewritten to "unknown" on the way in — the field is for reading,
#: and a free-form one would make every consumer guess.
#:
#: ``device`` IS HERE BECAUSE TEN CALL SITES ACROSS SEVEN MODULES PASS IT, and the
#: list is enumerated rather than counted because a numeral hid an omission once
#: already: the five credential modules' emitters (``credentials/`` github,
#: lifecycle, owner, reconcile, sync), the relay's ``credential.placement`` record,
#: the CLI's ``credential.placement`` audit, ``device_renamed``, ``device_rotated``
#: and the PEER-APPLIED ``epoch_rotated`` (``relay.py``, the emitter whose
#: ``actor=link.device_id``). Before this value existed, every one of those rows
#: silently rendered ``actor_kind: "unknown"`` — the audit recorded that SOMETHING
#: happened and lost the fact that a DEVICE did it. The alternative fix (stop passing
#: it, let those rows say ``human`` or ``unknown``) would erase a distinction every
#: emitter clearly intends, so the set was widened instead. Do not "clean up" the
#: emitters back to three values: "device" is a peer acting under its own key, which
#: is neither a person at this keyboard nor an agent process here.
#:
#: NOT every device-shaped row carries it, and the rule matters more than a list:
#: this value answers WHO ACTED, not who is NAMED. It is passed where the audited
#: action is a device acting under its own key (the credential emitters, a rename, a
#: rotation — the list above), and withheld where the actor id names a device but the
#: initiator was the operator at this keyboard (the LOCAL ``epoch_rotated`` refusal
#: after ``member rm``; ``member_removed``, ``invite_minted``, ``trust_changed``) or
#: where the event audits THIS device's own act described from a peer's row
#: (``membership_learned``, ``member_admitted``, ``panic_received``, and the link,
#: pairing, reconcile, session-stream, authorisation and definitions families). Those
#: keep the default deliberately; reclassifying them is a per-event decision and not
#: a sweep, which is why the review's three examples are examples and not the tail.
ACTOR_KINDS = ("human", "agent", "relay", "device", "unknown")


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
    #: ``unknown``. Never presented as proof of who typed a command. The closed set is
    #: :data:`ACTOR_KINDS`; a value outside it is stored as ``unknown``.
    actor_kind: str = "unknown"
    ts: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class MentionedFileAuditIdentity:
    """Closed P3 identity; JSON strings freeze nested context before rendering.

    Binding contains only public identity/authority metadata and key digests,
    never credential material or owner paths. The separate schema deliberately
    does not inherit legacy detail truncation: losing any field denies admission.
    ``establish`` covers the entire stamped source (size is stamp[4]); its
    ReadBinding retains the first eventual response's exact requested span.
    """

    operation: str
    binding_json: str
    read_binding_json: str
    source_stamp: tuple[int, ...]
    nonce: str

    def detail(self) -> dict[str, Any]:
        if type(self.source_stamp) is not tuple:
            raise ValueError("source stamp must be immutable")
        detail = {
            "schema": "lop.mentioned-file-read.v1",
            "operation": self.operation,
            "binding": json.loads(self.binding_json),
            "read_binding": json.loads(self.read_binding_json),
            "source_stamp": list(self.source_stamp),
            "nonce": self.nonce,
        }
        if not _mentioned_file_detail_valid(detail):
            raise ValueError("invalid mentioned-file audit identity")
        return detail


@dataclass(frozen=True, slots=True)
class MentionedFileAdmissionReceipt:
    """Exact persisted P3 row, not a capability to emit content.

    The inode and row bytes are captured only after the sole rotating writer's
    durability/readback boundary. B2 still owns revocation and socket release.
    """

    sequence: int
    device: int
    inode: int
    record: bytes
    generation: int


class AuditLog:
    """Append-only, batched, bounded. The only writer of ``audit.jsonl``."""

    BUFFER_BYTES = AUDIT_BUFFER_BYTES
    TICK_S = AUDIT_TICK_S
    DURABLE_EVENTS = DURABLE_EVENTS
    #: Class-level like ``DURABLE_EVENTS``, because these two tables are the writer's
    #: POLICY and :meth:`record` reads them off the instance — so a caller (or a test)
    #: can state a narrower one without a second writer.
    STREAM_LIFECYCLE_EVENTS = STREAM_LIFECYCLE_EVENTS

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
        #: The other half of ``_seq``: the highest ``seq`` this writer has actually
        #: written to the file. The two together are what let a reader tell an
        #: unflushed row from one that does not exist — see :meth:`publication_of` —
        #: and the gap between them on a quiet relay used to be a whole heartbeat with
        #: nothing saying so. Seeded from the file, because rows already on disk are
        #: published whatever this process has done.
        self._published_seq = self._seq
        self._last_flush = time.monotonic()
        #: Two MEASURED counters, because §4.8's bounds are asserted against them
        #: rather than estimated: ``write_calls`` counts the flush calls that actually
        #: opened the file and wrote (one append-open plus one ``.write()`` carrying the
        #: batch — CALLS, not ``write(2)`` syscalls, which a batch over the text layer's
        #: buffer can split; see :data:`STREAM_LIFECYCLE_EVENTS`). Its whole point is
        #: that those calls are bounded by the 1 Hz tick plus the buffer filling plus
        #: durable events, NOT by the frame rate, and ``syncs``
        #: counts completed live-file durability cycles for ``DURABLE_EVENTS``
        #: and rotation. Each cycle now also persists directory names; archive
        #: publication has its own file/directory fsync before live retirement.
        #: ``scripts/mesh_audit_probe.py`` reads both; a
        #: bound with no instrument behind it is a claim nobody can check.
        self.write_calls = 0
        self.syncs = 0
        #: Set when a write fails. A failed AUDIT write must not stop the relay; it
        #: is reported by `lop network status` instead, so the operator learns the
        #: trail has a hole rather than discovering it during an incident.
        self.degraded = False
        self.degraded_reason = ""
        self.records_written = 0
        # Only record_required owns this receipt, under the writer lock. Counters
        # alone cannot certify a row: rotation may move it to a different inode
        # and render later bookkeeping before the caller receives its answer.
        self._required_seq: int | None = None
        self._required_line: str | None = None
        self._required_path: Path | None = None
        self._required_identity: tuple[int, int] | None = None
        self._required_durable = False
        self._required_file_detail: dict[str, Any] | None = None
        self._required_file_phase = "fetch"
        self._pending_directory_syncs: set[Path] = set()
        #: The cross-process lock (``audit.jsonl.lock``), held by THIS writer while
        #: depth > 0. Re-entrant by depth because ``record`` -> ``flush`` ->
        #: ``_rotate`` -> ``sync`` nest, and a second ``flock`` on a fresh descriptor
        #: from the same instance would wait on itself. Touched only under
        #: ``_write_lock``.
        self._file_lock_fd: int | None = None
        self._file_lock_depth = 0
        #: Acquisitions that gave up at :data:`AUDIT_LOCK_WAIT_S`. Deliberately NOT
        #: ``degraded``: nothing was lost (rows stay buffered and the next flush
        #: retries), and ``degraded`` is sticky - it would refuse every later content
        #: admission over one slow peer.
        self.lock_timeouts = 0

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

    @property
    def published_through(self) -> int:
        """The highest ``seq`` THIS WRITER has put in the file; 0 when it has none.

        THE BOUNDARY, NOT A COUNT, AND IT IS ABOUT THIS WRITER. Every record numbered
        at or below it has been written by this log, and the rows above it are either
        still in its buffer or were never recorded by it — :meth:`publication_of` is
        the one-answer form of that. The counter is seeded from ``audit.jsonl`` when the
        writer opens it, so it starts at whatever the file already held; a row some
        OTHER process appends afterwards is not this writer's to report, which is a
        distinction that only matters because the file has more than one writer in
        practice (see :meth:`publication_of`).

        A FAILED WRITE IS THE ONE CASE THIS OVERSTATES IT. ``flush`` drains the buffer
        before it opens the file, so a batch lost to an ``OSError`` is a hole this
        counter cannot see; ``degraded`` is the flag that says the trail is incomplete,
        and it is set on exactly that path.
        """
        with self._write_lock:
            return self._published_seq

    @property
    def recorded_through(self) -> int:
        """The highest ``seq`` this writer has RECORDED, published or still buffered."""
        with self._write_lock:
            return self._seq

    def publication_of(self, seq: int) -> str:
        """Where the record numbered ``seq`` is: ``file``, ``buffered`` or ``unknown``.

        THE MIDDLE ANSWER IS THE ONE THIS EXISTS FOR, and it is the difference between
        a reader and a guess. ``file`` means PUBLISHED BY THIS WRITER — written to the
        live file, or to a generation that retention still holds — and it is a
        HIGH-WATER MARK rather than a claim of presence: everything at or below
        ``_published_seq`` went through this log's own ``flush``, but a row a rotation
        or the retention cap has since PRUNED keeps this answer for good, so ``file``
        must not be read as "a record with this number is on disk somewhere" (agent
        review round 3, MINOR 1: measured with seq 1 absent from every generation and
        ``publication_of(1)`` still ``file``). ``lop network log --export`` is what
        reads the compressed generations back; ``§4.5`` of the incident playbook is the
        retention rule. ``buffered`` means this writer has RECORDED it and has not
        published it, so a reader that reads the file and finds
        nothing is seeing the LAG rather than an absence — which is the state an
        operator reading ``audit.jsonl`` during an incident cannot otherwise
        distinguish from a row that never existed, and it is why stream lifecycle rows
        are published at the state change instead (see
        :data:`STREAM_LIFECYCLE_EVENTS`). ``unknown`` means this writer never recorded
        that number: there is no such row, and no amount of waiting will produce one.

        In-process, because it is this writer's own memory that answers the middle
        case: a reader in ANOTHER process holds its own ``AuditLog`` and must take the
        numbers from the one that owns the file — ``lop network status`` carries them
        (``relay``'s ``audit_published_through`` / ``audit_recorded_through``) for
        exactly that reader, and the human surfaces print them in the reader's own
        words (``relay.audit_status_words``). THE ANSWER IS ABOUT THIS WRITER'S OWN
        COUNTER. Other writers append to the same file (approvals, CLI
        ``write_event``), and every append runs under the cross-process lock after a
        re-read of the tail (:meth:`flush`), so a number on disk names ONE row. The
        consequence for this answer: a ``buffered`` number is PROVISIONAL - if another
        writer appended meanwhile, :meth:`flush` renumbers this writer's buffered rows
        to follow the file, so ``recorded_through`` can jump forward at a flush and a
        ``buffered`` row may publish under a larger number. (Before that lock the
        middle answer could name the WRONG ROW: agent review round 3, MINOR 2 measured
        ``publication_of(1)`` answering ``buffered`` while the file held a different
        row numbered 1.) Callers that need a number that cannot move - the content
        admissions - number under the lock at write time (:meth:`record_required`).
        """
        with self._write_lock:
            if seq <= self._published_seq:
                return "file"
            if seq <= self._seq:
                return "buffered"
            return "unknown"

    # -- writing ------------------------------------------------------------

    def admits_required(self) -> bool:
        """Would :meth:`record_required` admit a row right now (enabled, not degraded)?

        WHY A PUBLIC PREDICATE: a caller whose ONE durable row already covers a stream
        (the working-set serve's contiguous continuations, #4245 security review S-1)
        must still stop the moment the trail it relies on is disabled or has a hole,
        without writing another row per chunk or reading this writer's privates.
        ``record_required`` uses this same predicate, so the two cannot drift. A
        ``True`` here is advisory, not an admission: only ``record_required`` holds
        the write lock and proves a row durable.
        """
        return bool(self._enabled) and not self.degraded

    def record_required(self, event: AuditEvent, *, expected_file_phase: str = "fetch") -> bool:
        """Commit a durable admission before its caller releases protected bytes.

        Ordinary logging must not kill a relay on disk failure. Content egress,
        however, must fail closed: disabled, previously degraded, failed append
        and failed fsync are all refusals. Hold the writer's existing lock over
        both the write and its acknowledgement so concurrent rows cannot stand
        in for this one. This reuses the sole rotating writer, not another log.
        """
        if event.event not in self.DURABLE_EVENTS:
            raise ValueError("a required audit record must be a durable event")
        if event.event == "mentioned_file_read_admitted":
            raise ValueError("P3 requires record_mentioned_file_required and exact disk readback")
        with self._write_lock:
            if not self.admits_required():
                return False
            expected_detail = None
            if event.event == "session_file_read_admitted":
                # A missing hash phase must not masquerade as a legacy fetch.
                # Snapshot BEFORE sanitizer/render hooks can mutate the event;
                # strict primitive types also reject True == 1 and 1.0 == 1.
                if not _file_read_detail_valid(event.detail, expected_file_phase):
                    return False
                expected_detail = event.detail.copy()
                sanitized = _sanitize_detail(event.event, event.detail)
                if not _file_read_detail_matches(sanitized, expected_detail, expected_file_phase):
                    return False
                if not _file_read_detail_matches(
                    event.detail, expected_detail, expected_file_phase
                ):
                    return False
            # NUMBERED UNDER THE CROSS-PROCESS LOCK, held until the ack below: the
            # acknowledgement compares against this number, so it must be the one that
            # reaches disk. ``_sequence_under_file_lock`` drains buffered rows and takes
            # the file's own tail as the base; no other writer can append before this
            # row does, so ``flush`` will not renumber it. A lock that cannot be taken
            # (timeout, no primitive) is a REFUSAL - nothing is rendered or written.
            if not self._sequence_under_file_lock():
                return False
            requested_seq = self._seq + 1
            self._required_seq = requested_seq
            self._required_line = None
            self._required_path = None
            self._required_identity = None
            self._required_durable = False
            self._required_file_detail = expected_detail
            self._required_file_phase = expected_file_phase
            try:
                self.record(event)
                return (
                    not self.degraded
                    and self._published_seq >= requested_seq
                    and self._required_path is not None
                    and self._required_durable
                    and (
                        expected_detail is None
                        or _file_read_line_matches(
                            self._required_line,
                            expected_detail,
                            expected_file_phase,
                            requested_seq,
                        )
                    )
                )
            finally:
                self._required_seq = None
                self._required_line = None
                self._required_path = None
                self._required_identity = None
                self._required_durable = False
                self._required_file_detail = None
                self._required_file_phase = "fetch"
                self._exit_file_lock()

    def record_mentioned_file_required(
        self, identity: MentionedFileAuditIdentity
    ) -> MentionedFileAdmissionReceipt | None:
        """Persist and READ BACK the exact P3 admission, under the sole writer lock.

        Legacy S1 acknowledgements intentionally retain their historical contract.
        An in-memory rendered line or a sequence counter cannot prove what reached
        disk. Rotation may relocate the row: acknowledge that archive's inode,
        never the new live file merely because its sequence is higher.
        """
        if type(identity) is not MentionedFileAuditIdentity:
            return None
        try:
            detail = identity.detail()
        except (ValueError, TypeError, KeyError):
            return None
        with self._write_lock:
            if not self.admits_required():
                return None
            # NUMBERED UNDER THE CROSS-PROCESS LOCK, held until the ack below: the
            # acknowledgement compares against this number, so it must be the one that
            # reaches disk. ``_sequence_under_file_lock`` drains buffered rows and takes
            # the file's own tail as the base; no other writer can append before this
            # row does, so ``flush`` will not renumber it. A lock that cannot be taken
            # (timeout, no primitive) is a REFUSAL - nothing is rendered or written.
            if not self._sequence_under_file_lock():
                return None
            seq = self._seq + 1
            self._required_seq = seq
            self._required_line = None
            self._required_path = None
            self._required_identity = None
            self._required_durable = False
            event = AuditEvent(event="mentioned_file_read_admitted", detail=detail)
            # Freeze independently: a render/sanitizer must not redefine the
            # identity we intended to admit by mutating its input map in place.
            expected_detail = json.loads(json.dumps(detail))
            try:
                self.record(event)
                if (
                    self.degraded
                    or not self._required_durable
                    or self._required_path is None
                    or self._required_identity is None
                    or self._required_line is None
                ):
                    return None
                row = json.loads(self._required_line)
                if (
                    row.get("schema") != SCHEMA
                    or type(row.get("seq")) is not int
                    or row["seq"] != seq
                    or row.get("event") != event.event
                    or row.get("detail") != expected_detail
                    or not _mentioned_file_detail_valid(row.get("detail"))
                ):
                    return None
                expected = self._required_line.encode("utf-8")
                path = self._required_path
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(fd, "rb") as handle:
                    info = os.fstat(handle.fileno())
                    if (
                        not stat.S_ISREG(info.st_mode)
                        or info.st_nlink != 1
                        or (info.st_dev, info.st_ino) != self._required_identity
                    ):
                        return None
                    if path.name.endswith(_ROTATION_MARKER):
                        with gzip.GzipFile(fileobj=handle, mode="rb") as archive:
                            exact = _p3_persisted_row_matches(archive, expected, seq)
                    else:
                        exact = _p3_persisted_row_matches(handle, expected, seq)
                    after = os.fstat(handle.fileno())
                    named = path.stat(follow_symlinks=False)
                    if (named.st_dev, named.st_ino) != self._required_identity:
                        return None
                    if not exact or (
                        info.st_dev,
                        info.st_ino,
                        info.st_size,
                        info.st_mtime_ns,
                        info.st_ctime_ns,
                    ) != (
                        after.st_dev,
                        after.st_ino,
                        after.st_size,
                        after.st_mtime_ns,
                        after.st_ctime_ns,
                    ):
                        return None
                generation = (
                    int(path.name.rsplit(".", 2)[-2]) if path.name.endswith(_ROTATION_MARKER) else 0
                )
                return MentionedFileAdmissionReceipt(
                    seq, info.st_dev, info.st_ino, expected, generation
                )
            except (OSError, ValueError, TypeError, EOFError):
                return None
            finally:
                self._required_seq = None
                self._required_line = None
                self._required_path = None
                self._required_identity = None
                self._required_durable = False
                self._exit_file_lock()

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
            before_seq = self._seq
            required_file = self._required_file_detail is not None
            try:
                line = self._render(event)
            except (TypeError, ValueError):
                if not required_file:
                    raise
                self._seq = before_seq
                return
            if required_file and (
                not _file_read_line_matches(
                    line,
                    self._required_file_detail,
                    self._required_file_phase,
                    self._required_seq,
                )
                or not _file_read_detail_matches(
                    event.detail, self._required_file_detail, self._required_file_phase
                )
            ):
                # _render reserves a sequence, but no bytes/counters have been
                # appended yet. Under this SAME lock we can discard that candidate
                # without writing an apparent fetch row or leaving a sequence gap.
                self._seq = before_seq
                return
            if self._seq == self._required_seq:
                self._required_line = line
            self._buffer.append(line)
            self._buffered_bytes += len(line)
            self.records_written += 1
            if event.event in self.DURABLE_EVENTS:
                # A DURABLE EVENT IS WRITTEN THROUGH **AND FSYNCED** (§4.7): the
                # record whose loss to a power cut is unrecoverable must be on the
                # platter before the act it describes can be reported as done. The
                # reasoning the design gives is the one this code had drifted from —
                # it flushed (a buffered write into the page cache) and never synced, so
                # "the latch is written and fsynced before the broadcast" was true of
                # the buffer and false of the disk. Only these events pay it: a
                # per-record fsync on ordinary traffic is exactly the disk I/O the
                # operator asked to be bounded.
                # Append AND fsync under one hold of the cross-process lock (re-entrant
                # when ``record_required`` already holds it). If it cannot be taken
                # the row stays BUFFERED - reported on stderr, retried by the next
                # flush - because ``record`` never raises and must never write unlocked.
                if self._enter_file_lock():
                    try:
                        self.flush()
                        self.sync()
                    finally:
                        self._exit_file_lock()
                return
            if event.event in self.STREAM_LIFECYCLE_EVENTS:
                # A STREAM'S OWN ROWS PUBLISH WHEN THE STREAM'S STATE CHANGES, which is
                # why this branch is here and not at a call site: both halves of a pane's
                # pipe — the owner's and the opener's, the open and the close — carry the
                # whole of what a reader wants from them the moment they happen, and any
                # one emitter that forgot would put the row back on the heartbeat with a
                # 15 s window in which it is indistinguishable from a row that does not
                # exist (:data:`STREAM_LIFECYCLE_EVENTS` has the measurement and the cost
                # argument). PUBLISHED, NOT SYNCED, and the flush also resets the tick, so
                # the fall-through below would find nothing left to do.
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

        UNDER THE CROSS-PROCESS LOCK TOO (``audit.jsonl.lock``): another process's
        writer may have appended since this one last looked, so the tail is re-read
        and this writer's buffered rows are renumbered to follow it before anything
        is written; rotation (which renames the file and writes ``audit_rotated``)
        runs inside the same hold. If the lock cannot be taken the rows stay in the
        buffer for the next flush: publishing them unlocked IS the duplicate-``seq``
        bug, and dropping them would be a hole.
        """
        with self._write_lock:
            if not self._buffer:
                self._last_flush = time.monotonic()
                return
            if not self._enter_file_lock():
                return
            try:
                self._flush_locked()
            finally:
                self._exit_file_lock()

    def _flush_locked(self) -> None:
        self._renumber_buffer(self._disk_sequence() + 1)
        payload = "".join(self._buffer)
        contains_required = (
            self._required_line is not None and self._required_line in self._buffer
        )
        self._buffer.clear()
        self._buffered_bytes = 0
        try:
            # A newly created directory also needs its own parent synced;
            # syncing only audit.jsonl's parent can lose the whole directory.
            parent = self._path.parent
            while not parent.exists():
                self._pending_directory_syncs.add(parent.parent)
                parent = parent.parent
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
                written = os.fstat(handle.fileno())
            if contains_required:
                self._required_path = self._path
                self._required_identity = (written.st_dev, written.st_ino)
                self._required_durable = False
            self.write_calls += 1
            # EVERYTHING RENDERED SO FAR IS NOW IN THE FILE. Each rendered record
            # either joined this batch (and has just been written) or was a rotation
            # record, which ``_write_rotation_record`` writes to the file directly,
            # so ``_seq`` is the honest upper bound of what the file holds. This is
            # the boundary :meth:`publication_of` answers on.
            self._published_seq = self._seq
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

    def sync(self) -> None:
        """``fsync`` the live file — durable events and rotation, nothing else.

        Called after :meth:`flush`, so the buffer is already on the descriptor; a
        failure is reported the same way a failed write is (``degraded`` plus a line
        on stderr) rather than raised, for the reason :meth:`record` gives: a log that
        cannot be synced must not kill the relay that is trying to record an incident.
        """
        with self._write_lock:
            try:
                descriptor = os.open(self._path, os.O_WRONLY | os.O_APPEND)
                try:
                    synced = os.fstat(descriptor)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                self._sync_directories()
                self.syncs += 1
                if self._required_path == self._path and self._required_identity == (
                    synced.st_dev,
                    synced.st_ino,
                ):
                    self._required_durable = True
            except OSError as exc:
                self.degraded = True
                self.degraded_reason = str(exc)
                print(
                    f"warning: the network audit log could not be synced: {exc}",
                    file=sys.stderr,
                )

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

    # -- the cross-process lock ---------------------------------------------

    def _enter_file_lock(self) -> bool:
        """Take (or re-enter) the exclusive lock on ``audit.jsonl.lock``; ``False`` = refused.

        WHY A FILE LOCK: ``_write_lock`` only orders THIS instance's threads; the
        approvals path opens a fresh ``AuditLog`` per event, the CLI's ``write_event``
        does the same, and the relay holds a long-lived one, so numbering needs an
        exclusion every process honours. The primitive is ``wakes.lock``'s
        (``fcntl.flock`` LOCK_EX|LOCK_NB on POSIX, ``msvcrt.locking`` LK_NBLCK on one
        byte on Windows) so the platform branches live in one place, retried to a
        deadline as ``credentials.placement._cross_process_lock`` does: never a parked
        blocking ``flock`` (on macOS a thread inside ``flock()`` blocks a sibling's
        ``close()`` of that fd).

        WHAT ``False`` MEANS, per caller - never "write anyway":
        * ``flush`` / ``record``: rows stay buffered, a warning goes to stderr,
          ``lock_timeouts`` counts it, the next flush retries. ``record`` keeps its
          never-raises contract; ``degraded`` is NOT set for a timeout (nothing lost).
        * ``record_required`` / ``record_mentioned_file_required``: a refusal
          (``False`` / ``None``) before anything is rendered - content egress fails
          closed.
        * No lock primitive importable (neither ``fcntl`` nor ``msvcrt``): refused
          LOUDLY and ``degraded`` is set, because on such a host no row can ever be
          written safely and the operator must learn the trail has stopped.
        """
        if self._file_lock_depth:
            self._file_lock_depth += 1
            return True
        from local_operator.procstate import O_BINARY
        from local_operator.wakes import lock as wake_lock

        lock_path = self._path.with_name(self._path.name + AUDIT_LOCK_SUFFIX)
        try:
            # The lock lives in the log's directory, so create it the way ``flush``
            # does (0700, unsynced ancestors remembered for the next durable sync).
            parent = self._path.parent
            while not parent.exists():
                self._pending_directory_syncs.add(parent.parent)
                parent = parent.parent
            self._path.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(self._path.parent, 0o700)
            flags = os.O_CREAT | os.O_RDWR | O_BINARY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            return self._lock_refused(f"the audit lock could not be opened: {exc}")
        reason = ""
        try:
            deadline = time.monotonic() + AUDIT_LOCK_WAIT_S
            pause = _AUDIT_LOCK_RETRY_S
            while not wake_lock._try_lock(fd):
                if time.monotonic() >= deadline:
                    self.lock_timeouts += 1
                    reason = (
                        f"another writer held {lock_path.name} for over "
                        f"{AUDIT_LOCK_WAIT_S:g} s; the records stay buffered"
                    )
                    break
                time.sleep(pause)
                pause = min(pause * 2, _AUDIT_LOCK_RETRY_MAX_S)
        except ImportError as exc:
            os.close(fd)
            self.degraded = True
            self.degraded_reason = f"no file-lock primitive on this platform: {exc}"
            return self._lock_refused(self.degraded_reason + "; nothing was written")
        except OSError as exc:
            reason = f"the audit lock could not be taken: {exc}"
        if reason:
            os.close(fd)
            return self._lock_refused(reason)
        self._file_lock_fd = fd
        self._file_lock_depth = 1
        return True

    def _exit_file_lock(self) -> None:
        self._file_lock_depth -= 1
        if self._file_lock_depth:
            return
        fd, self._file_lock_fd = self._file_lock_fd, None
        if fd is None:
            return
        from local_operator.wakes import lock as wake_lock

        try:
            wake_lock._unlock(fd)
        except (ImportError, OSError):
            pass
        finally:
            os.close(fd)

    @staticmethod
    def _lock_refused(reason: str) -> bool:
        print(f"warning: the network audit log was not written: {reason}", file=sys.stderr)
        return False

    def _disk_sequence(self) -> int:
        """The highest ``seq`` on disk; call ONLY under the file lock.

        Floored at ``_published_seq`` so a vanished live file (or an unreadable tail,
        which ``_last_sequence`` answers 0) can never restart numbering under rows
        this writer already published. MEASURED cheap: 0.05 ms median on an 8 MiB
        file (it reads only the last 64 KiB) - see :data:`AUDIT_LOCK_WAIT_S`.
        """
        return max(_last_sequence(self._path), self._published_seq)

    def _sequence_under_file_lock(self) -> bool:
        """Enter the lock and base ``_seq`` on the file (the durable paths).

        On ``True`` the caller OWNS one lock level and must ``_exit_file_lock``.
        Buffered rows are RENUMBERED here (to follow the file) rather than flushed
        as a batch of their own: the durable row then joins them in ONE append,
        exactly as before this lock existed, so rotation still sees the same batch
        cross the cap (``test_archive_failure_preserves_prior_copy_and_refuses_content``
        pins that). With the lock held until the ack, no other writer appends in
        between, so the flush's own renumber is a no-op and the number the caller
        compares against is the one that reaches disk.
        """
        if not self._enter_file_lock():
            return False
        self._renumber_buffer(self._disk_sequence() + 1)
        return True

    def _renumber_buffer(self, start: int) -> None:
        """Restamp buffered rows ``start, start+1, ...``; only ``seq`` changes.

        Re-serialised with ``_render``'s exact ``json.dumps`` arguments, so a row's
        bytes differ from a fresh render only in the number (pinned by
        ``test_audit_cross_process``); a row already at its number keeps its string.
        If a durable caller's row would MOVE, its receipt is dropped so it refuses
        rather than acknowledging a number that is not on disk - the durable paths
        number under the lock, so this is a guard, not a path they take.
        """
        renumbered: list[str] = []
        for offset, line in enumerate(self._buffer):
            seq = start + offset
            row = json.loads(line)
            if row.get("seq") != seq:
                row["seq"] = seq
                fresh = (
                    json.dumps(row, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
                if line == self._required_line:
                    self._required_line = None
                line = fresh
            renumbered.append(line)
        self._buffer[:] = renumbered
        self._buffered_bytes = sum(len(line) for line in renumbered)
        self._seq = start - 1 + len(renumbered)

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

    def _sync_directories(self) -> None:
        """Make published names discoverable, not just their inode bytes durable.

        Durable events/rotation pay this cost; ordinary batched telemetry does
        not. Keep failed ancestors pending so a later sync cannot hide a hole.
        """
        # Windows has no ``os.O_DIRECTORY`` and cannot open a directory for
        # ``fsync`` at all; NTFS makes a new name durable through the file's own
        # flush, which the caller has already done. Referencing the attribute
        # unguarded raised AttributeError there (observed on a windows-latest
        # runner), which escapes sync()'s OSError handling and loses the row's
        # durability ack. Clear the pending set too: nothing will ever be
        # retryable on such a platform, so keeping entries would only grow it.
        # POSIX keeps the exact open/fsync/close sequence below.
        directory_flag = getattr(os, "O_DIRECTORY", None)
        if directory_flag is None:
            self._pending_directory_syncs.clear()
            return
        directories = self._pending_directory_syncs | {self._path.parent}
        for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            descriptor = os.open(path, os.O_RDONLY | directory_flag)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        self._pending_directory_syncs.clear()

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
            # gzip.close writes its trailer. Sync the COMPLETE archive and its
            # published name before retiring the live copy; syncing the fresh
            # audit_rotated inode afterwards says nothing about these bytes.
            descriptor = os.open(target, os.O_RDONLY)
            try:
                archived = os.fstat(descriptor)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._sync_directories()
            if self._required_path == self._path:
                self._required_path = target
                self._required_identity = (archived.st_dev, archived.st_ino)
                self._required_durable = True
            self._path.unlink()
        except OSError as exc:
            self.degraded = True
            self.degraded_reason = str(exc)
            return
        self._write_rotation_record(generation, size)
        # THE ROTATION RECORD IS SYNCED TOO (§4.7): "fsync is called for DURABLE_EVENTS
        # and on rotation". A rotation that only reached the page cache, after the
        # generation it describes was moved and the live file replaced, is the one
        # edit an attacker with root would most like to make disappear.
        self.sync()
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
            # The active caller must receive its receipt before retention can
            # retire that exact generation (including a zero-generation policy).
            if path == self._required_path:
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


def _p3_persisted_row_matches(lines: Any, expected: bytes, seq: int) -> bool:
    """One exact persisted row at this sequence, not a coexistence test.

    A duplicated/substituted row with the same sequence cannot stand beside the
    intended one and still produce an acknowledgement. Stream generations rather
    than loading the rotating log into memory.
    """
    found = 0
    for line in lines:
        row = json.loads(line)
        if type(row) is not dict:
            return False
        if row.get("seq") == seq:
            if type(row["seq"]) is not int or line != expected:
                return False
            found += 1
    return found == 1


def _mentioned_file_detail_valid(detail: Any) -> bool:
    """Validate the separate closed P3 schema without coercion or truncation."""
    if type(detail) is not dict or set(detail) != {
        "schema",
        "operation",
        "binding",
        "read_binding",
        "source_stamp",
        "nonce",
    }:
        return False
    binding = detail["binding"]
    read = detail["read_binding"]
    binding_keys = {
        "subject_issuer",
        "subject_id",
        "owner_key_digest",
        "requester_key_digest",
        "native_account_digest",
        "anchor_digest",
        "owner_admission_digest",
        "requester_admission_digest",
        "association",
        "authority_epoch",
        "generation",
        "journal_revision",
        "deny_generation",
        "evidence_digest",
        "relay_instance",
    }
    read_keys = {
        "owner_key_digest",
        "network_id",
        "network_epoch",
        "link_id",
        "transcript_hash",
        "request_id",
        "session_id",
        "source_row_id",
        "provenance_id",
        "proof_digest",
        "file_version",
        "phase",
        "offset",
        "length",
        "transfer_nonce",
    }
    if (
        detail["schema"] != "lop.mentioned-file-read.v1"
        or detail["operation"] not in {"establish", "stat", "fetch", "verify", "resume"}
        or type(binding) is not dict
        or set(binding) != binding_keys
        or type(read) is not dict
        or set(read) != read_keys
        or type(detail["source_stamp"]) is not list
        or len(detail["source_stamp"]) != 7
        or any(type(v) is not int for v in detail["source_stamp"])
        # POSIX timestamps may precede the epoch. Preserve the exact signed
        # ctime/mtime rather than inventing a new file-age admission policy.
        or any(detail["source_stamp"][i] < 0 for i in (0, 1, 4, 6))
        or detail["source_stamp"][5] != 1
        or not stat.S_ISREG(detail["source_stamp"][6])
        or type(detail["nonce"]) is not str
        or re.fullmatch(r"[a-f0-9]{32}", detail["nonce"]) is None
    ):
        return False
    integer_keys = {
        "generation",
        "journal_revision",
        "deny_generation",
        "network_epoch",
        "offset",
        "length",
    }
    for key, value in {**binding, **read}.items():
        if key in integer_keys:
            if type(value) is not int or value < 0:
                return False
        elif type(value) is not str or not value or any(ord(c) < 32 for c in value):
            return False
        if key.endswith("_digest") and (
            type(value) is not str or not re.fullmatch(r"sha256:[a-f0-9]{64}", value)
        ):
            return False
    return (
        binding["generation"] > 0
        and read["network_epoch"] > 0
        and binding["owner_key_digest"] == read["owner_key_digest"]
        and read["phase"] in {"stat", "fetch", "verify", "resume"}
        and (detail["operation"] == "establish" or detail["operation"] == read["phase"])
        and re.fullmatch(r"[a-f0-9]{64}", read["transcript_hash"]) is not None
        and re.fullmatch(r"[a-f0-9]{32}", read["transfer_nonce"]) is not None
        and (
            read["file_version"] == "establish:" + detail["nonce"]
            if detail["operation"] == "establish"
            else re.fullmatch(r"sealed:sha256:[a-f0-9]{64}", read["file_version"]) is not None
        )
    )


def _file_read_detail_valid(detail: Any, phase: str) -> bool:
    """Closed planned-read variants, not Python's coercive numeric equality.

    A phaseless row retains historical fetch meaning. Hash callers must supply
    their trusted operation outside the detail, so losing phase cannot downgrade
    an admission into an apparently valid fetch receipt.
    """
    if type(phase) is not str or phase not in {"fetch", "stat", "verify"}:
        return False
    keys = {"reference", "admitted_bytes", "offset", "decision"}
    if phase != "fetch":
        keys.add("phase")
    return (
        type(detail) is dict
        and set(detail) == keys
        and type(detail["reference"]) is str
        and bool(detail["reference"])
        and type(detail["decision"]) is str
        and detail["decision"] == "allow"
        and type(detail["offset"]) is int
        and detail["offset"] >= 0
        and type(detail["admitted_bytes"]) is int
        and detail["admitted_bytes"] >= 0
        and (phase == "fetch" or (type(detail["phase"]) is str and detail["phase"] == phase))
    )


def _file_read_detail_matches(detail: Any, expected: dict[str, Any] | None, phase: str) -> bool:
    return expected is not None and _file_read_detail_valid(detail, phase) and detail == expected


def _file_read_line_matches(
    line: str | None,
    expected: dict[str, Any] | None,
    phase: str,
    seq: int | None,
) -> bool:
    try:
        row = json.loads(line or "{}")
    except (ValueError, TypeError):
        return False
    return (
        type(row) is dict
        and row.get("event") == "session_file_read_admitted"
        and type(row.get("seq")) is int
        and row["seq"] == seq
        and expected is not None
        and _file_read_detail_matches(row.get("detail"), expected, phase)
    )


def file_read_reference(reference: str, offset: int, admitted: int, *, phase: str = "fetch") -> str:
    """Bound an owner-canonical reference without reading the referenced content.

    Test the actual sanitizer, whose ASCII-escaped JSON budget differs from its
    string-length cap. The typed fallback hashes only the relative reference:
    it is neither a file-content digest nor authority to resolve an absolute path.
    """
    detail = {
        "reference": reference,
        "admitted_bytes": admitted,
        "offset": offset,
        "decision": "allow",
    }
    if phase != "fetch":
        detail["phase"] = phase
    if not _file_read_detail_valid(detail, phase):
        raise ValueError("invalid planned file-read detail")
    # Budget the FINAL variant: a reference fitting fetch can overflow once the
    # hash phase is present. Fallback hashes ONLY the canonical reference string.
    expected = detail.copy()
    if not any(ch < " " for ch in reference) and _file_read_detail_matches(
        _sanitize_detail("session_file_read_admitted", detail), expected, phase
    ):
        return reference
    fallback = "reference:sha256:" + hashlib.sha256(reference.encode("utf-8")).hexdigest()
    expected["reference"] = fallback
    if not _file_read_detail_matches(
        _sanitize_detail("session_file_read_admitted", expected.copy()), expected, phase
    ):
        raise ValueError("unrepresentable planned file-read detail")
    return fallback


def _sanitize_detail(event: str, detail: dict[str, Any]) -> dict[str, Any]:
    """Drop what is not whitelisted, then fit the map into its byte cap.

    Never raises on an unknown key: a lost audit record is worse than a dropped
    key, and the whitelist exists to bound what a record can say rather than to
    make an emitter's typo fatal at runtime.
    """
    if event == "mentioned_file_read_admitted":
        # P3's closed structured identity must survive byte-for-byte. Legacy
        # telemetry caps/sanitization cannot silently drop an authority field.
        return json.loads(json.dumps(detail)) if _mentioned_file_detail_valid(detail) else {}
    if event == "session_file_read_admitted":
        # This event's variants are closed: never drop an extra/invalid phase
        # into a valid-looking legacy fetch row. Required callers reject {}.
        phase = detail["phase"] if "phase" in detail else "fetch"
        if not _file_read_detail_valid(detail, phase):
            return {}
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
    # The MAX in the tail, not the last line: a flush that rotates writes the
    # ``audit_rotated`` row (numbered after its batch) BEFORE the batch, so the fresh
    # file's last line is not its highest number. With writers resyncing from this
    # value under the lock, "last line" would hand that number out a second time.
    highest = 0
    for line in tail.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and type(record.get("seq")) is int:
            highest = max(highest, record["seq"])
    return highest


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
