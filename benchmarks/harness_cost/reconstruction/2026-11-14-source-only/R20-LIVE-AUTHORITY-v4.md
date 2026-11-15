# R20 remediation — bounded live authority v4, SOURCE ONLY

Scope:9d6884a06673959713494378d6013294049c8081..candidate. R20 is the ONE
accepted MAJOR from terminal round20 (reviewer429f3e8e5cb2). No other scope or
operator question is reopened. Both operator answers remain bound. No native
attempt, real GO, registered03 claim, Docker/engine/network socket, app/helper,
model/account/paid call, install, build or settings access/write occurs here.
Local anonymous IPC in inert Python children is explicitly parent-authorized.
Last native attempt remains UNCONSUMED, pending independent closure AND NEW GO.

## Actual RED, not an assertion-only diagnosis

At9d6884a the actual supervisor Python source prefix, stopping BEFORE pins and
engine/socket statements, accepted four real direct child executions: original,
two new pipes rebuilt from the consumed/retired claim, and a fabricated claim
with no issuer ever registered. All exited0; effects were empty. RED receipts
are retained in the new R20 results, not inserted into old v3 evidence.

## Trust boundary (explicit, not counterfeit secure boot)

Trusted: approved source execution in the actual controller and child, review/
desk-supplied source/head/GO digest anchors, private registered storage and kernel
PID birth. Untrusted: saved claim/GO/marker values by themselves, new pipes or
socketpairs reconstructed from them, old copied frames, arbitrary verified
booleans and stale PID numbers. Durable records are evidence/anti-replay, NEVER
bearer authority or proof of a live issuer. A self-chosen hash/secret in a fake
claim cannot enroll a permit in the actual controller's _ACTIVE registry.

This is NOT a sandbox against arbitrary malicious same-UID Python/root replacing
approved code, reading live process memory/inherited capabilities, implementing
a counterfeit authentication server and rewriting every record. Unix PID/UID
alone cannot attest approved source; do not claim that impossible boundary.
Actual tests use the real source server/registry, not a fake success responder.

## Minimal coherent protocol — no daemon, listener, dependencies or threads

1. Permanent parent attempt claim is still O_EXCL,0600, fsynced and never released.
   It now carries a PUBLIC claim_id plus stable kernel controller birth. It does
   NOT contain the new32-byte secret, its hex form or a reconstructible bearer.
   The secret exists only in the actual registered permit's live memory and its
   inherited anonymous socket capability. No token/key/credential store is read.
2. issue_resolver_handoff consumes issuance once, opens an anonymous socketpair,
   nonblockingly prefills one bounded initial frame, and retains the parent end.
   Child FD is the only inherited locator; no environment allow flag. Initial
   partial/empty frames and pipe/regular FDs fail without waiting. Message frames
   are <=2048bytes and the complete initial frame must fit one nonblocking send.
3. Child validates current source/GO, claim/public bindings, actual parent PID
   AND kernel birth, then sends a fresh random challenge + its own kernel birth,
   authenticated by the ephemeral secret. Parent serves synchronously in the
   existing begin_resolver path. It must find the exact live _ACTIVE permit,
   issued endpoint, unretired controller and expected real direct child birth;
   no data file or caller-supplied secret creates that registry entry.
4. Under a nonblocking exclusive flock on the parent claim, parent validates live
   authority and atomically creates resolver-consumed.json O_EXCL,0600, fsync file
   AND directory. Record binds original claim hash, authenticated child birth,
   request digest, GO and source. Partial/existing markers are terminal. Parent
   sends an authenticated grant, receives authenticated ACK and rechecks live/
   unretired authority + exact consumed record under the same claim lock before
   authenticated final commit. A second redemption never receives a grant.
5. Child checks commit and its durable redemption owner before returning the
   actual preflight. It retains the live channel and rechecks peer closure,
   retired marker, claim, consumed owner, parent/child birth and GO immediately
   before host WORK effects: socket/engine stage, info, image inspect, container
   dispatch. These guards deliberately NEVER gate known-owned cleanup.
6. retire_permit revokes _ACTIVE, clears memory secret and closes the peer FIRST,
   then creates a0600/fsynced retired marker under claim flock. Even journal/lock
   failure cannot keep a live grant: child sees peer closure. Parent failure is
   still reported, never hidden. Main's finally still retires after cleanup.

The parent exchange reuses the existing monotonic WORK deadline (840s maximum),
GO expiry and75min slot; it does NOT allocate a new timeout/lease or retry lane.
Each select is nonblocking/bounded by the existing0.5s monitor cadence and work
remaining. Parent polls source/GO/live registry plus disk, swap, attributed RSS/
ownership during waits; channel closure/errors abort. Test-only shorter deadlines
exercise stalls and cannot raise the production840s ceiling. No background
server/thread/interactive permission or service lifetime exists.

Concurrency is linearized at durable O_EXCL consumed creation under flock; final
commit follows a fresh live/revocation check. Revocation before a work boundary
rejects; no claim is made that a userspace check can retract an already-started
syscall. Existing parent process-group/container cleanup and independent inner
container deadline remain authoritative for work already dispatched.

## Retained invariants / exact next review surface

Configured capacity remains<=8.6GB/14 CPUs; attributed RSS<=4.3GB. Container1CPU/
2GiB,840/900s bounds,120s readiness, disk/fleet/swap/slot, causal launch/identity,
foreign-work and exact-owned cleanup guards are NOT relaxed. Paid/general holds,
R18 meter/test and R16 broker/session/identity bytes remain frozen. Only the actual
begin_resolver handoff, child work fences, authority module/tests and prospective
source metadata change. No real last-attempt claim is touched.

New generation-overlay-v4.json selects v4; immutable standalone baseline, old
meter and v2/v3 overlays/evidence remain unchanged. runtime_qualified=false and
execution_authorized=false source receipts are NOT native grants. Historical
verifier retains its existing failure, never 'ignore one drift'. Normal acyclic
manifest rule remains: external approved overlay SHA; derived raw-byte manifest.

## Actual-child GREEN requirements

New test_resolver_authority_r20.py runs VERBATIM child source through the exact
preflight boundary, real direct subprocesses and actual parent registry/server.
No getppid mock; only private fixture pwd home and explicit faults. Original once,
post-consume/retire rebuilt pipes, old real frame, never-issued fabricated claim,
self-chosen secret/no active issuer, wrong/stale births/GO, partial/empty/regular
FD, bounded deadline, concurrent started children(one winner), child exception
and retirement before effects are exercised. No engine/socket/runtime statements
are executed. Tests reap exact child process groups, never broad process kills.

Command receipts, RED and GREEN outcomes, unchanged guard proofs and final source
validation are in R20-LIVE-AUTHORITY-v4-RESULTS.txt. Return candidate for R20 delta
closure; this document is NOT a fresh desk GO and never authorizes an attempt.
