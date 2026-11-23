# Trusted account-subject → mesh-device-key binding

**Status: proposal only; independently reviewed design required before any push or implementation.**

Source baseline: `219a0d40c9c83e43a2c5fa1a638e2dff71b35330` in
`local-operator`. Every source citation below is to that exact commit, using
`local-operator:<path>:<line-range>`. Inspection was Git-object/source-only:
no runtime, imports, tests, builds, native ceremonies, real accounts or keys.
This is not native/security acceptance evidence.

Request: **“P1 = a trusted account-subject→device-key binding.”** The priority is
simple automatic UX with security that adds no friction. The allocator reports
that 7c leaves this generic primitive unowned and #4179G1 proves a listener's
device key, not which keys belong to one account. Those are allocation inputs,
not claims verified against those other branches here.

## 1. Problem and scope

Possession of a mesh key answers *which device authenticated this connection*.
It does not answer *which canonical account an owner has assigned that key to*.
Network membership, `view`, nickname, shared network epoch material, and phone
pairing must not be promoted into that second assertion.

Keep three independent propositions:

1. **Device authentication:** this live listener accepted the holder of full
   requester Ed25519 key K under this network/membership incarnation.
2. **Account assignment:** a trusted issuer explicitly assigned K to subject
   `(canonical account issuer, opaque account ID)`, at the stated assurance and
   purpose. With approved local binding this means **operator-endorsed
   assignment**, not continuous requester login at the remote account server.
3. **Resource permission:** this owner may disclose this particular session's
   file version now. Account assignment alone grants no file or other access.

The existing dormant **file-domain P1** already combines an explicit local
assignment with a particular file consent domain. It is **not** the unowned
generic mesh account-binding primitive. P2 is owner-live provenance, also not
account assignment. Neither new naming nor a generic adapter may widen either.
The prior choices **“Allow files the transcript names”** and **“Use local device
approval”** remain authoritative; this proposal does not replace them with
cloud registration or automatic mesh grants.

Non-goals: broad authentication overhaul, credential lending, enrollment
changes, requester OAuth sharing, outside-PAD activation, fixing native/file-only
coverage, new UI, inferred historical provenance, or changing PAD/attachment
permissions. Existing held paths and other owners' work remain untouched.

## 2. Source truth: existing mechanisms and missing pieces

| Source at the baseline | What it actually establishes | What it does not establish |
| --- | --- | --- |
| `local-operator:local_operator/network/handshake.py:720-745,812-829` | Listener checks the authenticated member's stored key and exports peer full key, SAS and transcript hash. | An account subject. |
| `local-operator:local_operator/network/handshake.py:557-587` | At this baseline the dialer's `peer_public_key` is deliberately empty. | A proven individual listener key at the dialer. A shared-network MAC is not an individual-account certificate. #4179G1 is a separately reported dependency, not present proof here. |
| `local-operator:local_operator/network/file_device_approval.py:77-155,242-296` | Strict full Ed25519 key/derived ID; actual incoming `PeerLink`; active non-suspect owner/requester; exact network epoch and stable admission fields; both need `view`. | Membership or view alone as account authority; an arbitrary link-shaped dict is refused. |
| `local-operator:local_operator/network/file_device_approval.py:158-231` | Installed root-pinned presence-backed matching anchor/profile and exact native OAuth row/incarnation, compared to the current profile database. | File-only/staged authority, borrowed credentials, or a stale database handle as positive evidence. |
| `local-operator:local_operator/providers/oauth/radient.py:252-306` | Owner bearer authenticated to fixed canonical `/me`, no redirects, HTTP 200, exact nonempty trimmed `identity.account_id == account.id`. | A server-signed portable device certificate; a requester login. Token-response identity labels are explicitly non-authority at `:309-323`. |
| `local-operator:local_operator/network/file_device_approval.py:50-52,117-155,1186-1237` | File-specific signed domain and live authorized-context check, including fresh owner `/me`, current association, anchor, membership and negative journal fence. Returns association/epoch/generation/digest metadata. | Generic account membership, file bytes, or an existing P3 serving capability. |
| `local-operator:local_operator/operator/verify.py:42-67,195-228,285-326`; `local-operator:local_operator/operator/sign.py:216-245` | Operator action/approval/device domains are separate. Existing phone/device certificate signs a P-256 SPKI, device ID, label and time window. | An Ed25519 mesh-key → canonical account assertion. Its payload has no account subject, mesh network, member incarnation or file purpose. |
| `local-operator:local_operator/network/enrollment.py:22-33,251-265` | Explicit network-enrollment-only domain and commitment. | Authority to certify account assignment; the remedy says account logins are not shared. |
| `local-operator:local_operator/network/readiness.py:395-426,3426-3442` | Readiness can truthfully report installed authority with signing elsewhere; the local host may be verify-only. | That any member has been account-bound, that the signer is present here, or that it can sign a new domain. |
| `local-operator:local_operator/network/authorizer.py:152-158,322-374`; `local-operator:local_operator/network/files.py:180-219,238-268,302-340,469-492` | Phase/epoch/capability/session ownership plus current PAD/reference/sensitivity/no-follow/hardlink confinement. | Outside-PAD authority or strict per-read durable disclosure audit. |
| `local-operator:local_operator/session/file_provenance.py:83-137,739-828` | Future-live point proof, exact recorded row spans/digests, owner incarnation and current descriptor/root identity; no viewer-supplied path accepted. | Read permission or a full content digest: `:83-84` explicitly says capture does not read target bytes. |

The second handshake observation was checked one step beyond the empty result:
`read_welcome()` calls `_peer_key_from_welcome()`, whose implementation returns
`""`. Do not claim the dialer gap is fixed by this document. P3 below uses the
**owner's incoming authenticated listener-side peer identity**.

Existing design contracts are also baseline evidence:
`local-operator:docs/design/file-device-approval.md:3-26,28-59` (selected local
trust and dormancy), `local-operator:docs/design/file-provenance.md:159-181`
(native/file-only coverage and P3 holds), and
`local-operator:docs/design/mesh-file-transfer.md:258-263` (view authority for
PAD pulls only, not a new outside-PAD grant).

## 3. Issuer options and trade-offs

### A. Existing operator-authority key signs account-bound member mesh keys

**Recommended trust direction, with a new explicit delegation, not an existing
capability.** Reuse the installed operator P-256 trust anchor and presence-backed
signer; assign each *authenticated full Ed25519 member key* to the owner's
canonical-verified subject in a new typed statement. The installed anchor is
trusted only by owners where it is already securely pinned; peer readiness,
TOFU, enrollment, a phone certificate or a received certificate cannot install
or replace that anchor.

Proposed new signing domain: `lop-mesh-account-binding-v1\0`. It is distinct
from mesh auth, phone certificates, enrollment, generic approvals and file
approval. A signature in any old domain is never retried in this domain. A
new operator-local verifier must require an explicit account-assignment
statement, current canonical owner account context, exact anchor identity and
matching owner-local issuer delegation. Being authorized to sign `approve` does
not itself authorize this new statement. If a phone transports a decision, its
existing operator-device certificate only authenticates the decision channel;
it does not become the mesh/account issuer.

Pros: preserves the selected local trust boundary, no requester cloud login,
no new registry dependency, automatic repeated use after one genuine approval.
Cons: owner-local trust does not automatically propagate to another owner;
new issuance/delegation and revocation semantics need independent review; the
current native coverage gaps remain. If an owner trusts a different anchor,
require its own endorsement or an explicitly authorized delegation, not an
unproven cross-anchor chain. The threat boundary still trusts the owner OS and
negative-state store; root/full owner compromise is not repaired by signatures.

**Smallest file integration:** do not mint a second generic certificate solely
to repeat an already valid file-domain assignment. A proposed typed verifier
may project the existing `authorized_context()` into a **purpose-limited local
binding result**, after all its live checks, only for
`future_live_mentioned_file_access`. This is a verifier interface, **not** generic
backfill or a new issuer. It must recover the full committed evidence, not just
trust the four returned metadata fields. No other consumer may accept that
projection. Conversely, a new generic certificate never substitutes for the
file-specific approval. This avoids a duplicate journal and a second approval
ceremony for the selected file feature.

### B. Radient account-signed device certificate

**Proposed external API, not an available endpoint proven here.** A service
would authenticate the requester account and require fresh full-key
proof-of-possession, then issue a domain-separated certificate binding issuer +
opaque account ID to the exact Ed25519 key, serial/generation, purposes,
validity, and revocation/status authority. Owners would need independently
pinned issuer keys and an explicit decision to trust this issuer for this
purpose. `/me` JSON over TLS is not that certificate. The inspected native OAuth
path at `local-operator:local_operator/providers/oauth/radient.py:252-306`
provides canonical account verification only; this proposal has no source or
live evidence of a device-certificate issuance/status endpoint.

Pros: potentially automatic across already authenticated devices, central
account/device removal. Cons: new server contract and status availability,
issuer-compromise blast radius, login/recovery friction on headless devices,
and a materially different trust policy. Short certificate expiry alone cannot
provide live revocation. Even with such a certificate, outside-PAD local file
approval remains required under the approved trust. Do not choose B by default
or silently migrate existing users to it.

### C. Pairing-time human SAS attestation

Existing pairing has SAS machinery
(`local-operator:local_operator/network/handshake.py:828,926-933,1057-1065`),
but comparing SAS authenticates that pairing, **not an account subject**.
Proposed enhancement: during an already required owner-local pairing ceremony,
a presence-authorized owner explicitly attests that the transcript-bound full
key is assigned to its freshly canonical-verified subject. Sign and persist the
same typed account-assignment statement as A, with ceremony evidence bound to
both full keys, network, member admissions, link/transcript, request nonce,
subject and purpose. A peer's own statement “I compared SAS” is never authority.

Pair-phase links cannot call the existing member-only file backend. Complete
member admission first, capture a current authenticated incoming member link and
its stable admission facts, then explicitly bind the earlier SAS ceremony to
that same full key in the new assignment. If that correlation cannot be proved,
refuse or perform a fresh explicit member-key approval; do not bypass `_mesh` or
reuse a pair-phase grant. This sequencing is proposed work, not current support
(`local-operator:local_operator/network/authorizer.py:167-183`;
`local-operator:local_operator/network/file_device_approval.py:248-284`).

Pros: no second key-comparison step during new pairing, visible MITM defense.
Cons: human comparison/incorrect approval remains a risk; unattended enrollment
has no human SAS proof; historical SAS success records without the account
statement cannot be retrofitted. It is an issuance ceremony for A, not an
independent cryptographic account issuer. Never invent a new SAS size or retry
policy without evidence; reuse the existing pairing mechanism.

## 4. Proposed primitive: inputs, assurance and exact result

New component proposal: `network/account_binding.py`, a small typed verifier
contract, **not** a new authentication stack or generic permission map. This
path does not exist at the baseline and is not implemented here. Its initial
file-purpose evidence adapter should fix/reuse the existing file approval spine;
other consumers require separately scoped issuance and review.

The new A/C statement, if authorized, commits canonical encoded fields:

- `schema=1`, exact domain, assertion kind `operator_endorsed_assignment`;
- `subject={issuer: canonicalHub, opaque_id: canonicalAccountID}`; byte-exact
  issuer/ID, never email, nickname, organization cascade or `account_ref`;
- binding issuer kind, installed anchor key ID/SPKI fingerprint/statement
  digest, explicit issuer-delegation ID/generation, owner full mesh key;
- requester algorithm `ed25519` and full canonical 32-byte key; derived device
  ID checked against it, not used in place of it;
- owner-local audience, network ID/current epoch, stable owner and requester
  admission identities (`added_at`, `added_by`, `added_via` as currently checked);
- closed allowed-purpose set; initially file-purpose only where authorized;
- binding ID, authority epoch, expected/current CAS generation, serial/nonce,
  action `assign` versus `revoke`, issuance request digest, current link and
  transcript evidence, issuance and expiry times;
- exact owner native credential row/created-at/login incarnation and presence
  anchor evidence for local issuance. No access/refresh token in the statement.

Unsupported versions, algorithms, assurance, purpose, issuer or extra security
fields fail closed; no weak-mode fallback. Canonical subject namespace and
binding-issuer identity are separate: the operator endorses an assignment to a
Radient subject but does not pretend to be Radient's account-login service.

Proposed owner-only call (names and types are **new**, not existing APIs):

```text
binding = await account_binding.verify_live(
    incoming_link=owner_PeerLink,
    owner_account=ExactNativeAccountSelection(credential_id),
    evidence=OwnerEvidenceReference(binding_id_or_file_association),
    required_subject=OwnerCanonicalSubject,       # recomputed by owner, not peer
    required_purpose="future_live_mentioned_file_access",
    read_binding=ReadBinding(
        owner_key, network_id, network_epoch, link_id, transcript_hash,
        request_id, session_id, source_row_id, provenance_id, proof_digest,
        file_version, phase, offset, length, transfer_nonce),
)
```

`verify_live` derives the authoritative root, peer key and current network facts
from the **actual owner listener link**, not any viewer-supplied peer/account
field. Evidence references are owner-local selectors, never bearer authority.
For the existing file adapter it calls
`await file_device_approval.authorized_context(link, auth, credential_id)`
(`local-operator:local_operator/network/file_device_approval.py:1186-1237`),
then reads the same verified committed slot/fence under the existing lock order
to construct the typed result. This requires a reviewed internal extension:
today's four-field dict is not enough and there is no public raw-slot getter.
It must not re-parse unverified JSON or weaken the original checks.
`required_subject` is an equality constraint, never an input authority: the
verifier itself obtains the fresh canonical result and exact native row context.
A caller-supplied or previously cached subject cannot make verification pass.

Success returns an immutable, owner-process-only `VerifiedAccountDeviceBinding`:
exact subject + issuer/assurance/purpose + full owner/requester keys + network
and stable member admissions + native row/incarnation + anchor/delegation
identity + binding ID/authority epoch/generation/evidence digest + current
revocation-state revision + link/transcript identity + **the entire ReadBinding**.
Failure is a typed refusal, never `None` interpreted as old behavior. It is not
a wire token, bytes grant, cached session authorization or a transferable
capability. A result is usable only by this one read's final fence; the next
phase/chunk/resume must verify again.

## 5. Exact P3 boundary before any outside-PAD bytes

**Proposed integration only. Outside-PAD stays disabled.** A new mentioned-file
route must be distinct from the current PAD/digest route so an unknown/old peer
cannot downgrade it into raw-path access. Names such as `row_id`, `proof_digest`
and the following guard are proposed protocol/integration fields.

1. Keep the owner authorizer's phase/epoch/`view` and `_session_scope` checks;
   require an owned session ID in exactly the frame the handler reads. No nested
   session-ID substitution. A binding cannot grant membership or sensitivity
   exceptions. Reject legacy raw path, extra selectors and unsupported protocol.
2. Resolve only `(owned session, source row ID, owner provenance ID/digest)`.
   Call existing `file_provenance.lookup_proof(root, session_id, provenance_id)`
   and require its source row ID/digests to match. The proof ID is **not** the
   transcript row ID. The viewer never chooses an owner filesystem path.
   Unknown, old, imported, edited, pending or uncommitted proof means refusal;
   no history scan/backfill. P2 freezes metadata, not a content hash: P3 must
   establish and seal a content-version commitment under the strict audit below,
   before any response can disclose that version. Descriptor identity, size and
   mtime alone do not commit immutable bytes.
3. Run the `verify_live(...)` call above with this exact resource/read context.
   Require exact equality to the owner's freshly canonical-verified subject,
   including issuer, and the full authenticated requester key; ensure purpose
   and the **separate file approval** are active. New generic evidence alone
   never passes the latter check. Existing file-only evidence can satisfy both
   assignment and consent only within its original file-purpose boundary.
4. Reopen metadata via S1 no-follow/nonblocking regular-only/single-link rules;
   recheck current sensitivity/denylist, relocated config-dir exclusion, parent
   root identity, file identity/version and owner provenance. Do not use a
   requester path, follow symlinks, weaken config protections or treat proof
   validity as current file availability. Revalidate after every await. Metadata
   equality is a necessary source check, not proof that prepared content matches
   the committed version; even same-size/same-mtime changes must be caught before
   that response's irreversible release.
5. **New strict audit/read guard:** durably record the exact subject, full-key
   binding digest, generation/revocation revision, link/request/session/row/proof,
   file descriptor identity/metadata stamp and requested span before content reads,
   hashes, response metadata or bytes. On append/fsync/readback failure deny;
   existing best-effort audit is not sufficient. Completion/failure can be a
   second event; never report success by rewriting a failed event. No secret or
   raw owner path in remote refusals. This guard is separate future work, not
   provided by P2 or `authorized_context` today. For initial version establishment,
   the pre-content audit records the exact source metadata, ReadBinding and
   establishment operation; no content digest exists yet. Only after that audit
   succeeds may the owner read/hash content, establish a consistent version and
   durably seal its commitment plus establishment receipt. That receipt binds the
   establishment nonce and sealed version to the first response's exact ReadBinding
   before release; an establishment marker is never an accepted content version.
   Subsequent responses
   bind both the sealed commitment and their exact ReadBinding into their own
   pre-content audit. A failure to establish/seal the version releases nothing;
   it cannot be treated as an unversioned success or as evidence of the bytes at
   the earlier mention time.
6. **Normative owner content invariant:** each exact prepared response's content
   MUST match its sealed committed content version BEFORE irreversible release,
   including every fetch span, reconnect and resume response. An immutable,
   owner-controlled snapshot established and committed under the audit in step 5
   is one valid implementation; owner verification of a span/chunk commitment
   over the **exact prepared bytes** against the sealed version is another.
   A freshly computed hash of uncommitted current bytes is not such verification.
   Descriptor/size/mtime checks and a later requester whole-file digest alone are
   insufficient. Once verified, the prepared response must remain immutable
   through emission; checking a buffer and then re-reading the mutable source is
   invalid. These mechanisms add no authority and exempt no request from current
   provenance, source/sensitivity/config-dir, account, membership or revocation
   checks. No new snapshot/chunk size, retention or retry cap is chosen here.
   After audit/content verification and immediately before committing the response,
   recheck account
   row/incarnation, account subject and local account-disabled state, both memberships/keys, network epoch,
   anchor/delegation, binding and file-approval generations, all negative fences,
   file version, and equality of the exact ReadBinding. Abort on any difference.
   Final owner-side emission must serialize with revocation through a new shared
   read/revoke fence. Do not hold a cross-process file lock over `/me`, presence
   prompts or socket waits. Every mutator affecting these predicates must join
   the fence or publish an invalidation revision that the guarded emission
   atomically observes; current separate stores do **not** prove this property.
7. Only that guarded path may release a frame. Repeat for `stat`, each `fetch`,
   verification/hash/resume, and every live-link request. The same checks apply
   to owner-held/not-yet-released content, owner staging, reconnect/resume,
   refetch and every fresh owner disclosure, including owner-side cache hits.
   A viewer's **already delivered, verified local copy** instead remains subject
   to viewer-local authorization and retention; reading it needs no new owner or
   mesh call. This is not copy recall. Neither a cache key nor local possession
   authorizes any **new remote bytes**. Requester whole-file digest verification
   remains integrity defense, not owner pre-release version proof or authority.

Exact future usage contract:

```text
proof = owner_provenance.require_exact(session_id, row_id, proof_id, proof_digest)
binding = await account_binding.verify_live(..., read_binding=exact_read)
with mentioned_files.authorized_read(proof, binding, exact_read) as read:
    # NEW guard audits BEFORE any content read/hash, seals the initial version,
    # or binds a subsequent read to that sealed commitment and exact ReadBinding.
    response = read.prepare_verified_response()  # exact immutable prepared bytes
    read.commit_response_to_link(response)  # final content/revoke fence; no raw dict
```

This must be implemented at the owner link/response-release seam, not merely as
one boolean before calling current `serve_file()`. Today `serve_file()` returns
a dict containing bytes (`local-operator:local_operator/network/files.py:440-454`)
and its installed handler does not carry the proposed verified context
(`:469-492`). Its size/mtime stamp admits content changes with the same stamp,
then fetch returns current bytes before the requester's whole-file verification
(`local-operator:local_operator/network/files.py:429-454`). P2 explicitly captures
no target bytes (`local-operator:local_operator/session/file_provenance.py:83-89`).
Those are concrete gaps the new owner-side content invariant must close, not
existing support. Current viewer cache lookup is local/stat-based
(`local-operator:local_operator/remote_files/cache.py:128-157`), and desktop code
returns a local hit before any peer fetch
(`local-operator:local_operator/server/utils/desktop_sessions.py:7842-7850,7919-7926`).
That existing local-copy path is not a grant to request fresh owner bytes and
must not be turned into a mandatory mesh round trip by this proposal.

### IMPLEMENTATION-STATUS: dormant owner-wide coordinator prerequisite

`network/file_release_coordinator.py` now supplies default-OFF, Linux-only durable
negative coordination mechanics, with no runtime consumer, switch, or installer.
OS-account home plus canonical existing anchor-path spelling chooses one domain,
independent of HOME/XDG/profile and anchor key content. This is location, **not
account proof or new authority**. An explicit never-used initializer creates its
private witness, stable lock and state; reads never initialize, repair or clear
missing/corrupt/pending state. Distinct overlapping transactions survive process
exit, with no PID/time/profile/key-rotation reset or new retention policy.

A successful begin acknowledges pending only after file/parent fsync and exact
readback, before any affected effect/prompt. External publication and independent
outcome verification stay outside the innermost coordinator lock. Finish may
remove only its exact transaction after that verification, never on a failed or
unknown outcome or from a generic finally. Rename is the visibility linearization
point, not the acknowledgement: an uncertain call latches its handle negative.
The closed outcome names are `finished`, `denied`, and `indeterminate`;
indeterminate is never success or a retry token, and its original call/handle
cannot be upgraded by a later read. Begin binds an exact external operation key
and boundary/profile key. Completion durably binds those keys, domain
incarnation, exact transaction and the independently verified outcome SHA256 to
its new revision. A fresh read-only inspection can recognize only that **exact**
committed fact; another effect/profile/transaction/fingerprint or absent match
refuses. It never re-executes an effect or changes state. A verified completion
already published before a final sync/readback/ack failure may be found committed
after restart; this is not recovery or permission, and the pre-begin receipt
remains invalid. An uncommitted failed begin permits no effect,
not an impossible promise to durably poison unchanged state. Partial/missing or
ambiguous state refuses. Owner-private integrity/coherent all-state rollback,
not hardware anti-rollback, remains the stated trust boundary.

**All four integration limits remain:** no actual PeerLink queue/handshake/close
handoff proof; no exhaustive real-mutator participation/source closure; no
production concurrent reader-close ownership proof; no macOS/Windows proof.
Raw storage/process/TCP tests are not the parked actual-PeerLink proof. A guarded
body permits at most one nonblocking emission, never source reads, external locks,
audit fsync, prompts, presence checks, socket readiness waits or sendall. No
production emitter uses this boundary, so no exhaustive fence is claimed.

The owner-wide coordination root is **managed control state, never eligible
outside-PAD content**. Future activation must integrate the existing central
managed-root/config exclusion (reuse S1 protection, not a parallel denylist),
besides complete authority/audit verification and participation. This dormant
prerequisite adds no serving path and cannot create production state by default.
Current strict source eligibility and C/outside-PAD serving remain unchanged/OFF.

### Dormant B1 implementation boundary

`network/mentioned_file_read.py` supplies only the owner-process audit → snapshot
→ seal → immutable preparation prerequisite. It has **no production caller,
handler, route, tool/CLI registration, wire projection or send method**. Existing
S1 PAD/digest serving and Files' outside-session denial are unchanged. B2's
actual cross-process mutator/socket-release fence and C's Files/preview/download
integration remain prerequisites; outside-PAD serving is still OFF.

The sole rotating `AuditLog` has a separate closed `mentioned_file_read_admitted`
schema and typed receipt, acknowledged only after append, file/directory fsync
and exact persisted-row readback on the actual live/archive inode. Legacy
`session_file_read_admitted` detail and acknowledgement semantics remain intact.
Each establishment has an owner-generated nonce and a distinct typed marker,
never a usable sealed version. A private session-owned snapshot is streamed and
read back against its chunk manifest before sealing. Each new establishment
gets a new directory and commitment even when its content digest repeats.
The durable exact seal binds its initial receipt, source/proof, nonce and first
final ReadBinding. Each subsequent preparation, including stat/verify/resume,
refreshes P1/P2 and commits its own exact pre-content admission; intersecting
chunks are verified before slicing the very bytes retained in the response.

This adds no authority or history/backfill path. Current source/parent identity,
ctime, sensitivity and relocated-config exclusions still govern every operation;
a snapshot never preserves a deleted/replaced/reclassified source's authority.
It reuses S1 acceptance rules and `FILE_CHUNK_BYTES`, with no new cap/TTL or eager
sweeper. Pending/torn state has no recovery-to-positive API and remains under
normal owning-session retention. Synthetic fixtures do not establish native
presence, real transport safety or security/QA approval.

## 6. Threats, replay and live revocation

| Case | Required refusal/behavior |
| --- | --- |
| Stolen member node with its private mesh key | It still authenticates; cryptography cannot detect theft alone. Owner revokes binding/file consent or removes/suspects member. Next guarded read on **already open links** denies. Before notice it has precisely the old authorized privileges, not a guarantee of physical possession. |
| Rogue `view` member, plausible nickname or same email | No trusted exact subject/full-key assignment: refuse. It may retain approved PAD rights; this proposal never turns view into outside-PAD rights. |
| MITM without member key | Existing handshake/key check plus account evidence bound to authenticated full key and owner-local issuer pin; SAS option binds the ceremony. A rogue member with shared epoch material must not impersonate another individual key. Dialer identity remains a separately required dependency. |
| Replayed assign/certificate/old approval after revoke | Current authority epoch and CAS generation plus durable dominant negative fence win. Reject old statement, pending transition, restored current-slot file or unknown journal; no re-initialize-to-active recovery. |
| Key rotation/re-add, copied device ID, previous-ID alias | Full key and stable member admissions must match. Existing rotation statements prove key continuity only (`local-operator:local_operator/network/identity.py:470-506`), not renewed account assignment. No silent transfer; explicitly authorize successor key. |
| Cross-network, account, issuer, purpose, owner or protocol replay | Exact audience/subject/key/epoch/purpose and domain mismatch denies. No retry with old schema, raw path, generic approval, enrollment or phone certificate. |
| Request/chunk replay on same live link | Exact request ID, transfer nonce, phase/span and ReadBinding; duplicate uses reverify and re-audit, never reuse a success. A transport retry cannot revive an old version or generation. |
| Owner logout, native row replacement/relogin, native account disabled or `/me` failure | Per-read owner freshness/canonical identity and row/incarnation checks deny; refresh alone is not relogin. No fallback to JWT labels or stored account metadata. |
| Anchor removed/replaced/unpinned, issuer delegation revoked, state unreadable | Deny on live links and future links; no readiness-based replacement or file-only fallback. |
| Audit unavailable, source changes, config/sensitive path, symlink/hardlink, revoked view | No outside-PAD bytes, even with a valid binding and old cached proof. |

Local binding revocation is owner-local: a connected requester does not need to
acknowledge it. Revoke publishes a durable dominating denial before success and
invalidates read guards/transfers; drop queued **not-yet-emitted** unauthorized
frames and staging/resume authority. The revoke acknowledgement must await that
local cancellation/fence. Already emitted or downloaded bytes cannot be recalled;
there is no promise to erase a recipient's prior copy. Closing a link is useful
but not the security boundary. Re-check on every read even if the link survives.

The existing file journal already makes pending/revoked state dominant before
projection (`local-operator:local_operator/network/file_device_approval.py:623-640,1118-1165`).
Reuse this for the file adapter, not a parallel weaker revocation database.
A future generic issuer needs equivalent durable authority-epoch/generation
and negative-state semantics. Recovery must not clear denies or mint a new
active generation after a partial failure. Coherent rollback of *all* trusted
owner-private transition witnesses remains outside the documented filesystem
trust boundary (`local-operator:local_operator/network/file_device_approval.py:10-19`).

A remote service cannot be made synchronously revocable everywhere by a signed
certificate alone. Option B needs authoritative live status for every read,
fail-closed on unavailable/stale status, and a specified observation boundary;
a remote revocation concurrent with its final status response still has a race.
A push/lease protocol would be additional reviewed work, not an implied claim.
For recommended A, requester cloud logout does **not** revoke local assignment:
this is exactly the approved operator-endorsement semantics. Owner account
verification is point-in-time, not continuous remote-server login monitoring.
At the current source boundary account revocation means native-row disable/
replacement or canonical bearer/`/me` refusal; the parser does not inspect an
additional remote account-disabled flag (`local-operator:local_operator/network/file_device_approval.py:196-220`;
`local-operator:local_operator/providers/oauth/radient.py:286-300`). Do not claim
instant remote account-status invalidation if the service still returns the
accepted canonical response. Stronger remote subject-status guarantees need a
separately proved server contract, not a fabricated local check.

## 7. Time and clock assumptions

Issuance ceremonies bind nonce, request window and one action. Existing file
approval limits the request window via `DEFAULT_EXPIRY_S`
(`local-operator:local_operator/network/file_device_approval.py:145-153`). Its
standing completed association is not silently converted into an expiring phone
certificate or new lifetime. A new generic certificate must explicitly include
`issued_at/not_before/not_after`; no lifetime number is selected here without
its operational evidence. Unknown/untrusted time, now before issuance/not-before,
or now after expiry denies; no unmeasured skew grace. Existing device-certificate
verification similarly rejects both future and expired certificates
(`local-operator:local_operator/operator/verify.py:322-325`).

Trusted local clock and trusted surviving negative-state store are assumptions.
Monotonic time can bound one running request but cannot authenticate wall time
across restart or repair filesystem rollback. A clock rollback detected by an
owner-local high-water observation invalidates the operation; undetected full
clock/state rollback is not solved here. Nonces, network epochs, generations
and negative fences provide replay isolation independently of expiry; none is
replaced by a wall-clock TTL. Existing local file-adapter evidence keeps its
original validity rules, not an invented expiry/backfill policy.

## 8. Migration and honest friction

| Existing evidence | No new prompt justified? | Outcome |
| --- | --- | --- |
| Current genuine completed file-domain P1 assignment, exact full key/subject/admissions/native incarnation/anchor, signature and negative fences all verify now | **Yes, for that file purpose only.** | Proposed purpose-limited projection can reuse the existing proof. Still no bytes until all P3/strict-audit/native gates are implemented and independently validated. |
| Already paired member, view role, nickname, same OS user/home, matching email, old SAS success or old enrollment approval | **No.** | Remain paired with existing PAD rights; outside-PAD refuses pending one explicit owner-local account/key assignment/file approval. No prompt on ordinary PAD use. |
| Existing operator phone P-256 certificate | **No.** | May authenticate the approved signing channel, not assign an unrelated Ed25519 mesh key/account. |
| Old file record with changed account, key, native incarnation, anchor, admission or unknown journal | **No.** | Refuse; obtain fresh exact evidence, never infer repair from legacy filesystem/account metadata. |
| Hypothetical valid account-signed device certificate | Only if that issuer/purpose was already explicitly trusted **and** file consent exists | Not today's migration route; do not claim an endpoint or installed delegation. |

For new pairing, the proposed assignment can be disclosed with the already
required pairing/local approval, avoiding a separate account-login or SAS step.
That is a future flow design, not permission to silently expand today's pairing
consent. For already-paired devices without authenticated assignment evidence,
the honest minimum is **one explicit owner-local approval per exact requester
key/account/owner association**; mere membership cannot make that zero. If a
future bulk disclosure lists several keys, each must be explicit and authenticated.
One biometric gesture for multiple signatures is **not proven** by source and
must not be promised. Repeated file reads require no user gesture after genuine
standing consent, but owner `/me` freshness/network availability remains a
fail-closed latency/availability cost. No TTL cache is added to hide that cost.

Linux file-only/staged-owner **NOFIT/H7** runtime gaps remain external holds,
not solved here. Baseline file P1 denies those anchors; P2's design explicitly
carries the held native integration and write-unconfirmed UX work
(`local-operator:docs/design/file-provenance.md:159-170`). A verify-only readiness
row and “sign from your device” wording are not evidence that the particular
new/file ceremony works end to end. No automatic phone fallback is assumed.

## 9. Recommendation, one owner decision, rollout checks

**Recommend A's local operator-endorsement trust, C only as an optional issuance
ceremony during already-required pairing. Fix/reuse the existing file spine
rather than adding a second file-account consent store.** Define the small typed
primitive and purpose-limited adapter first; new generic issuance is a separate
explicit delegation, not a reinterpretation of old grants. Do not implement B
or a registry as the default. Do nothing to outside-PAD serving until its own
held dependencies and independent reviews are clean.

**One decision for allocation 79 to raise, only for new generic issuance:**
Authorize the installed presence-backed operator authority to issue the new
`lop-mesh-account-binding-v1\0` statements as **owner-local assignments to its
canonical subject**, with no automatic cross-owner/cloud trust, or require
Radient-issued account certificates instead? Recommendation: authorize the
local, purpose-scoped delegation. This does not reopen or delay the already
selected file-domain local-binding trust; it decides whether a reusable new
issuer may exist beyond that existing file evidence. Until resolved, leave new
generic minting/delegation absent and design review pending. No human question
or implementation is initiated by this document.

Implementation questions to settle by evidence, not additional product choices:
which existing locked accessors can expose verified slot facts without widening
API authority; where all owner response releases and relevant mutations can
share a cross-process fence; strict audit durability/negative controls; and
native/remote-signing coverage. Source shows the constraints above, not their
resolution. Proposed source scope is the small account-binding interface,
file-approval verified-context extension, and later separate P3 owner-link guard;
not a broad authorizer or native-auth rewrite.

Design proof performed here: traced canonical `/me` through exact row/incarnation
rechecks; traced listener key into file intent and standing-context membership
rechecks; inspected actual phone certificate payload/algorithm before rejecting
its reuse; followed readiness's verify-only branch; traced current P3-adjacent
source lookup/byte-return seam and P2 metadata-only capture. These checks prove
the architectural distinction and missing seams, **not executable acceptance**.

Before a later implementation can ship, independent review/QA must exercise:

- positive existing file evidence reused without a second prompt, only under
  its exact purpose; view-only/unbound member denied;
- different full key with same ID/label, different issuer/opaque subject,
  rotated/re-added member, new epoch, owner relogin and replaced anchor denied;
- old signed assignment after revoke, selectively restored positive slot,
  pending/unknown journal and replay across links/requests/versions denied;
- revoke/account/member/anchor changes between canonical check, audit, content
  read and response release, on a live multi-chunk link and resume/cache paths;
- content-version mismatch on any exact prepared fetch span, reconnect or resume
  response, including same-size/same-mtime substitution: **zero content released
  for that response**. Previously emitted bytes cannot be recalled. Exercise the
  race between verification and emission; mutation of the verified buffer or a
  mutable-source re-read must not bypass the sealed commitment;
- initial version establishment/sealing failure and pre-content audit failure
  release no content; subsequent responses are bound to the sealed version and
  exact ReadBinding, with current source/provenance and authorization still checked;
- already delivered verified viewer-local copy remains readable under local
  authorization/retention without a new mesh call; cache key/local possession
  cannot authorize refetch or any new owner disclosure;
- audit append/fsync/readback failure, source substitution/symlink/hardlink,
  sensitive/config paths, bogus row/digest, imported/old/no proof denied before
  any outside-PAD disclosure;
- future/expired certificate, clock anomaly, network failure and unsupported
  version fail closed with no raw-path or old-domain fallback;
- native/file-only refusal retained until its own real-path evidence exists;
  source-only and synthetic checks never labeled native proof.

Watch during rollout: inadvertent generic reuse of file-purpose evidence,
readiness misread as authority, prompts multiplying across owner/device pairs,
per-read `/me` latency/refusal rate, unguarded cache/chunk paths, negative-state
recovery, and revoke acknowledgements preceding queued-response cancellation.
Every success/refusal receipt must identify source revision and actual evidence
surface. No runtime rollout, push, PR or self-approval is part of this proposal.
