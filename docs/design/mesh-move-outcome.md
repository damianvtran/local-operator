# A move reports only what it verified

Status: implementing (2026-10-21). Companion to `mesh-session-mobility.md` §6.3,
whose protocol this document does not change: it changes **when a phase may be
called completion**, what a source device must do before it authorises the
destination to adopt, and how every surface words the three possible outcomes.

## Why this exists

Measured from the operator's screen: a conversation titled *Refresh local
operator docs for release* was "moved" from `cloud-node-1` to the local device.
The surface entered its moving state, the request ended, and the conversation was
still on the remote: `On cloud-node-1`, working directory `/home/ec2-user`, an
unchanged composer. `cloud-node-1` had not answered the last two mesh probes.

A move that says it started and then changes nothing is worse than a refusal,
because the user stops watching. The failure is not only cosmetic: two protocol
holes can produce it honestly.

1. **Acceptance is not completion.** A queued move (`--queue`, and the runtime's
   `queue_move` intent) answers `ok: true` with phase `queued`. Several surfaces
   rendered that as "Moved". Nothing has moved; the copy has not started.
2. **The commit can report success while the source copy survives.**
   `_source_commit` writes the tombstone, tries the delete, and answers
   `committed` even when the delete failed (`cleanup="pending"`). The destination
   then promotes: two directories for one id, both listed as local, which is the
   exact state INV-1 exists to prevent. The code's own history records this
   happening on a real two-device run before `mark_store` existed.

## The contract

A move ends in exactly one of three states, and the state is built from facts the
reporting device has verified — never from an acknowledgement alone.

| State | What must be true | What the caller may do |
|---|---|---|
| **completed** | The destination holds a promoted, verified copy, **and** the source no longer holds a directory for the id. | Open the conversation at the destination. |
| **failed** | Nothing durable about the placement changed: source present, target absent. A reason is named. | Keep working where it is; a retry is safe. |
| **in progress** | The intent is durable and a live driver owns it (a queued move), **or** a committed handoff has not yet landed (the destination holds staging and will adopt it). A reason names the step. | Wait; the move is actually progressing. |

Anything else — notably "we started and cannot say" — is a **failed** answer with
`changed: true` when the device cannot prove the placement did not change. The
`unconfirmed` refusal keeps that meaning and remains a refusal, never a success.

### Clause 1: exactly one copy

Before this change, the source wrote the tombstone (the destination's proof to
adopt), then deleted its directory, then answered. A delete failure answered
`committed` anyway, so the destination adopted a second copy.

New order in `_source_commit`, after every existing validation (plan digest,
destination content digest, lease, completeness):

1. **Atomically take the directory out of the store**: rename
   `sessions/<id>` → `<config>/network/handed-away/<id>.<stamp>` under the same
   config root. This is the commit point: from the rename on, no scanner, list,
   picker or resume on the source can see a live copy. A refusal or an I/O error
   here happens **before** the tombstone and before the destination is told
   anything: the journal is cleared, the transaction rolls back, and the source
   still holds the whole conversation.
2. Write the tombstone naming the destination.
3. Clear the journal and answer `committed`.
4. Delete the quarantined tree with the existing bounded removal; a failure
   there is cleanup only — the tree is outside `sessions/`, it is swept later,
   and no reader can mistake it for a live conversation.

Crash windows:

* die after 1 before 2: the journal says `handing-off`, `sessions/<id>` is gone,
  no tombstone. `reconcile`/`_complete_handoff` writes the tombstone for the
  recorded taker; the destination then adopts from its verified staging.
* die after 2: the tombstone is the destination's proof; recovery is unchanged.
* die before 1: nothing happened, roll back as today.

`_remove_handed_away` treats an absent directory as **gone** — the earlier
behaviour answered "cleanup pending" forever when the directory was already
removed, which left the journal open and the source's engage guard latched.

### Clause 2: an unreachable peer fails, loudly

`lop sessions move … --to local` for a session last seen on a peer that is not
answering, and an offload to a peer that does not answer, both **refuse**:
`unreachable`, `changed: false`, and a sentence that names the device and says
nothing was changed. The refusal does not come from a timeout dressed as
progress: the requester reports it as soon as it knows.

The TUI additionally **checks the target before it leaves** the session it is in,
and it does **not** leave for a target KNOWN to be unreachable — the reported
experience, where the leave happened first and the unreachable notice arrived over
a conversation the user had already been put back on. That check is a PRE-CHECK,
not an acceptance: acceptance belongs to the CLI's own first step (the move's
`prepared` answer), and a peer that answers the pre-check and then refuses the
move — `busy`, `viewed_elsewhere`, `relay_unavailable` — is handled by the failure
path, which reopens the conversation where it still lives and says so. What this
clause rules out is leaving for a target the device already knows cannot take it;
it does not claim the TUI never leaves before acceptance.

### Clause 3: interruption is survivable

The staged copy is the only intermediate state, it lives outside `sessions/`, and
it carries `ready.json`. Either end may die at any point and `reconcile` finishes
in one direction: the source's journal/tombstone decides, and the destination
adopts only a copy it can verify against the owner's record. A recovered source
that finds its directory already quarantined writes the tombstone rather than
re-deleting.

### Clause 4: state travels

The copy set already carries the transcript, wakes, monitors, todos, subagent
state and project links (sync `COPY_SET_NAMES`); `carry.rebuild_indexes` rebuilds
the derived indexes at the destination. The one piece that was host-specific and
did not change was the **working directory**. At promote, the destination:

* keeps the recorded cwd when that path exists **on this host**;
* otherwise maps the recorded path's trailing components under the user's home
  when that exists (e.g. `/home/ec2-user/project` → `~/project`);
* otherwise falls back to the user's home.

The rewrite lands in `desktop.json` before `carry.rebuild_indexes` reads it, so
the wake and monitor indexes carry the rewritten path. The **transcript is never
edited** — it is the conversation, verified byte-for-byte before the promote —
and the move's receipt carries `carry.cwd = {from, to}` so the user is told.

### Clause 5: a running session completes its step, pauses, copies, resumes

That is the queued move's own sequence (`move_queue`: `queued → finishing →
paused → copying → resumed`), unchanged. What this document adds is that a
`queued` record is **in progress**, never reported as a completed move, and that
reading the record starts its driver if the thread is gone, so the state keeps
moving rather than merely existing.

## Surfaces

* `SessionMoveResult` gains `state: "completed" | "in_progress"`. A refusal
  (`ok: false`) is the failed state. `phase` keeps its existing vocabulary.
* `lop sessions move` prints "Moved …" only for `completed`; a queued receipt
  prints "Queued … nothing has moved yet" with the cancel verb.
* The TUI's initial line says the request is being made, not that the move has
  started, and only `completed` reopens the session at its new home. Its pre-check
  keeps it from LEAVING for a target known to be unreachable; a target that
  answers and then refuses is handled by the move's refusal path, which reopens
  the conversation where it lives (acceptance itself is the CLI's first step).
* `TransferReceipt` (desktop/phone) gains `state`; `source_retired` is true only
  for a completed move, so no surface can say "the copy here is gone" about a
  queued one.

## Evidence required before merge

* Real two-relay run, both directions: exactly one holder each time, transcript
  byte-identical at the destination, source directory gone, and a source
  tombstone naming the destination.
* A move whose source removal is refused: the operation fails with a reason,
  the tombstone is absent, and both devices still hold exactly what they did
  before.
* A recall from a peer that has stopped answering: refusal names the peer, the
  source still holds it, the destination holds nothing, and the TUI does not
  leave the session it is in.
* A cwd recorded on the source that does not exist at the destination: the
  destination's marker/wake index carry the mapped local path, the transcript
  is unchanged, and the receipt names both paths.
* A queued receipt: `state=in_progress`, CLI and TUI word it as not moved.
