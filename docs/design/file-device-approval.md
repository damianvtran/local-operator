# Dormant file-device approval (P1)

The selected trust contract is **“Use local device approval”** for the eventual
**“Allow files the transcript names”** feature. The owner explicitly associates
an authenticated requester's **full mesh key** with the owner's exact,
canonical-verified account. It is not a server registry, an attestation that the
requester remains logged in, or a reuse of credential lending/enrollment consent.

## Deliberately dormant

`network/file_device_approval.py` supplies a file-specific operation through the
existing `ApprovalOp`/typed approval spine. Its new signing domain is
`lop-file-device-approval-v1\0`; no existing kind's decision domain changes.
There is **no new public mint, CLI verb, agent tool, relay wire operation or UI**.
The kind is hidden from generic Mesh listings/badges and refuses generic CLI and
desktop approvers. The typed backend cannot bypass its own provenance, presence,
account, request/disclosure digest and apply checks.

No existing file/attachment/PAD caller consults this state. This PR does not
list, mark readable, cache, precache, index, download or serve outside-PAD files.
P2 live-mention issuance, strict per-read audit and P3 serving remain separate
prerequisites. `authorized_context` returns control-plane metadata, **not bytes
or a file-serving capability**. A future consumer still needs its own session,
source, no-follow/inode, sensitivity, version and audit-before-bytes boundaries.
Future mention capture includes owner-live tool/file-mention records, not only
human-text paths. It must not backfill from scans or transcript imports.

## Exact commitments

All security facts live inside the existing immutable `what` commitment:

- Canonical hub and exact account ID, never email/JWT labels or `account_ref`.
- Full owner/requester Ed25519 keys and algorithms, with derived IDs validated.
- Network/current epoch and both members' stable admission identities. Labels,
  `last_seen` and `previous_ids` are not authority.
- One explicit native OAuth row ID, created-at and login incarnation. No
  organization cascade, borrowed row, peer bearer or refresh-policy change.
- Installed presence anchor key ID, point fingerprint and statement digest.
- File authority epoch, expected/current CAS generation, nonce, request ID,
  request window, authenticated listener link/transcript identity and the
  cryptographically distinct `approve` versus `revoke` action.

The canonical `/me` transport/parser is shared with native login exchange:
redirects off, HTTP 200, nonempty trimmed exact `identity.account_id == account.id`.
Exact-row OAuth freshness is reused. Native metadata, active membership/full
keys and the installed anchor are checked again after network/prompt work and
before activation. Routine token refresh does not change login incarnation or
invent an authority generation; a true relogin does invalidate the old context.
The selected native-store handle is compared with the current profile-local
read-only database, so an unlinked old SQLite handle is not an offline positive.

## Ceremony and coverage limits

Filing, signing, write-side approval and apply require the installed,
root-pinned **presence-backed** anchor and matching backend/profile. Staged,
file-only, unpinned and missing anchors refuse. There is no quiet file-key or
historical-admin-install fallback. File-only owners have an explicit coverage
gap until a separately reviewed, evidenced per-action administrator ceremony
exists; this dormant backend is not eventual-feature completion.

No file-domain or approval-record mutation lock spans canonical HTTP or a human
prompt. Existing prompt admission remains the typed spine's own policy. The
provider serializes record then file-domain mutations with existing locks.

## File-domain durability, witness and retention

The private `<config>/network/file-device-approvals.json` contains a current
slot per owner/network/requester association, not a mesh-wide binding map. It
stores exact intent, generation, active/revoked status, signed activation and
anchor evidence. No raw file path/content, bearer or private key belongs in it.
Missing/corrupt state denies; approval history never reconstructs positive
authority. Minimal signed evidence survives card-history pruning.

A signed current slot alone is authentic, **not rollback-fresh**. The initial
prototype really reauthorized a restored active-g1 snapshot after revoking g2;
the preserved regression drove the separately approved correction. A distinct
file-domain journal and initialization sentinel now bind the authority epoch.
The journal publishes a pending fence durably **before** the current projection,
then a completed witness only after exact projection readback. A retained denial
generation dominates restored older current bytes. Indeterminate completion
attempts compensate with a higher-revision pending fence; uncertainty never
means "guaranteed no mutation." Missing, malformed, redirected or mismatched
witness state fails closed. Selective **current-projection** restore denies
with intact authoritative negative witnesses. Exact signed retries reconcile
the same generation, rather than minting a replacement request or epoch.

This mirrors the existing finite journal's revision/capacity discipline, **not**
its credential authority or namespace. Journal reads/publications are bounded
at 256 KiB, with at most 64 association fences and 64 candidate rows. Safe
expired/consumed candidates may be pruned; live authority/denial fences are not
evicted to make room. Capacity exhaustion refuses. Candidate preparation is
not a transaction-pending fence and does not suspend an existing association.

Only this security state opts into the new strict writer: existing staging,
rename and locks are reused, created directory entries and file/parent are
fsynced, and exact private-file readback is required. Legacy writer definitions
and defaults remain unchanged. Unsupported directory/no-follow durability
refuses rather than silently weakening persistence. Record-then-domain lock
order and CAS prevent two workers from applying one expected generation twice.

**Trusted-owner filesystem boundary:** authoritative negative journal/sentinel
integrity is trusted. Coherently rewinding **all surviving knowledge of a later
transition** is unsupported; it does **not** require rewriting every physical
file. For example, after a revoke-g2 pending fence is published but its projection
write fails, current g1 can remain untouched. Restoring only the old journal can
remove the sole surviving knowledge of that attempted g2 transition. Ordinary
owner-private files are not a protected monotonic service, hardware counter or
privileged anti-rollback store. The guarantee is peer/API replay resistance and
selective current restore denial with intact witnesses, not hostile-owner/OS
freshness. No peer/API/import writer that lowers the journal was identified by
the P1 review. No new root service, seal, key rotation or restart-reapproval
policy is added by P2; its metadata does not strengthen this boundary.

## Relay and desktop process separation

There is no process-local pending-candidate map. The relay captures only its
actual authenticated incoming listener result and publishes a durable candidate
shelf tied to the exact request/digest, native row, installed anchor, relay
instance and link. The desktop independently opens the selected local AuthStore
and validates the existing authenticated **local** `net_status` seam before
sign/apply. It consumes the existing `op: ack` reply, not a new protocol.

`net_status` appends only `role`, `alive` and `closing` to each existing link row.
Role is handshake direction, not member role; liveness/closing use existing
transport state. Old keys, values and relative order remain intact, including
old/unknown link objects (whose new fields are null). Missing/old/malformed
status denies only this new consumer, never status itself or its old rows.
No public key/account inventory or new control operation is exposed.

An unapplied candidate is invalid after relay-instance/link replacement. A
completed association is not tied to a Python PID or link lifetime: an ordinary
new incoming same-key link remains eligible with unchanged membership, epoch,
native account/incarnation, installed anchor and current journal. Relevant
context changes require approval, not continuity or automatic enrollment.

## Evidence scope and explicit non-proofs

Unit cells use real SQLite login completion, envelope/signature verification,
strict writes and failure injection. Installed presence and canonical-account
HTTP boundaries are synthetic, not physical native proof.

The separate-process fit uses two actual loopback relays/authenticated member
handshake inside relay process R and a distinct desktop process D. It exercises
actual authenticated local status, D restart with usable pending state, R
restart rejecting pending but retaining completed authority, two D workers with
one CAS winner, durable revocation and selective-current restore denial.
Canonical HTTP and installed-presence signing remain explicitly synthetic in
**both** processes. Existing typed desktop functions are exercised directly;
this is not an HTTP-router, macOS/native-hardware, installed-app or transport
artifact validation. The original failing prototype and first real-fit failure
are retained; that first real fit exposed the new consumer's incorrect `ok`
reply assumption, corrected to the actual existing `op: ack` contract.

Independent QA still owns the isolated typed approve/apply HTTP matrix with
actual statuses/state/frames. No UI, per-read audit, outside-PAD access,
physical-native proof, full-suite, wheel/RSS/zstd or new global-runtime claim
may be inferred from this dormant control-plane implementation.
