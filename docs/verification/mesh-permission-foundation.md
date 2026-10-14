# Owner-local permission foundation: source verification and activation hold

Authority: approved `docs/design/mesh-credential-lifecycle.md` (#2347), source base
`07420555a662389f7b03734f73a72899021b1af3`. Design approval is **not runnable live
consent**. This slice adds owner-local permission/executed-receipt persistence and
sticky denial enforcement; it does **not complete automatic sign-in inheritance**.

## Contract implemented

- `credentials/<network-id>/consent.json`: approved schema1, owner-local only;
  strict IDs/field allowlists, private atomic writes, directory fsync, bounded
  cross-process lock, revision CAS. Never peer-merged/exported or placed in SyncState.
- Restriction-only entries do not fabricate positive approval/inheritance. Revoke
  persists denial before projection; a failed projection write retains authority.
  CLI revoke/share and finite onboarding projection respect that negative overlay.
- Positive typed `what.credential_inheritance` uses an actual software-signed source
  and admitted current run in tests; current owner/member/public-key fingerprint,
  exact provider/MCP subset, native origin and required scope constraints must match.
  No approval from mesh membership, role, legacy shares or a bare PR/approval ID.
- Provision prepares disabled metadata, appends a distinct EXECUTED permission
  receipt, then re-admits and enables under approval-record -> consent lock/CAS.
  Cancellation/expiry/source changes fail closed; interrupted state is reported
  prepared/pending, not successful. No network refresh occurs under consent lock.
- Runtime authority read verifies immutable intent/signature/digest, execution,
  current identity, scope/revision and denial, not the initial onboarding lease.
  Completed runs/window expiry do not invent a permission TTL. Executed permission
  proof survives ordinary terminal approval pruning. Replay cannot clear denial;
  only a fresh signed exact selector/revision/withdrawal-reference action can.
- Owner grant rejects provider-wide denial before native resolve; owner-selected
  account context is checked before mint. Caller account_ref is never identity.
  Current denial is rechecked after asynchronous resolution; coalescing separates
  authenticated recipient/model/holder context. Existing native refresh ownership
  remains unchanged. Copy enforcement precedes local value extraction.
- Legacy finite grant/copy behavior remains where no applicable explicit deny exists.
  Positive future policy alone adds **no holder, capability, grant or copy**.

## Executed discriminating proof (not just tests)

A standalone scratch driver used real AuthStore, current owner broker, placement,
software operator approval/run/receipt store, and immutable base owner source. It
used only isolated synthetic roots/material; no provider endpoint/native signer.
Actual keyless output:

```text
actual admitted software-signed run:
permission_recorded revision3; static_storage_activation_pending;
deny_aware_owner_generation_required; enabled_readback=true;
placement_byte_unchanged=true
Projection fault OBSERVED; sticky denial persisted before fault
old074 SAME request after deny + stale holder: grant
new source SAME synthetic session: error/not_a_holder
material_response_present=false; stale_holder_remains=true
All synthetic broker loops/stores closed
```

This reproduces the old-reader bypass **against source base074**, not the actual
installed 0.70.12 live broker. It proves why even a new CLI is insufficient for
activation. No credential/header value appears in evidence.

## Targeted source matrix

23 isolated cases exercise real software-signature/admitted-run and owner-broker
requests, plus actual filesystem effects: success/executed readback/no projection;
requested/denied/expired/connected/stale run/bad signature/tampered digest refusals;
wrong owner/network/recipient/fingerprint; unknown required context/copied origin;
provider denial zero resolve; account denial zero mint; copy denial zero extraction;
second-write crash/stale holder; regrant/replay/CAS; withdrawal during native resolve;
duplicate JSON; fresh exact supersession without auto-grant; completion/window expiry
and 40-day sweep preserving durable executed permission, then explicit revoke.

Commands run serially through `scripts/run_bounded.py` with explicit worktree
PYTHONPATH, isolated HOME/config/basetemp and existing interpreter; imported
consent/owner/sync/approval APIs were asserted to come from this exact worktree.

```sh
$PY scripts/run_bounded.py --timeout 120 -- $PY -m pytest \
  tests/unit/network/test_credentials_consent.py -q -n0 --basetemp "$ISO/tests"
# 23 passed in 1.08s (before final type-only narrowing; rerun frozen head evidence in PR)
$PY scripts/run_bounded.py --timeout 120 -- $PY -m pytest \
  tests/unit/network/test_credentials_placement.py \
  tests/unit/network/test_credentials_sync.py -q -n0 \
  -k 'owner_cannot_revoke or regrant_widens or replayed_row or copy_is_withheld or replacing_a_copy or older_served_generation'
# 6 passed,100 deselected,2 pre-existing Pydantic warnings,1.09s
$PY scripts/run_bounded.py --timeout 120 -- pyright --pythonpath "$PY" <changed Python files>
# 0 errors,0 warnings,0 informations;3.0s
```

Scoped Black26.1/isort9.0.2 were executed from pre-existing read-only shared UV
archives with their cached dependencies, not installed/copied into a new tree.
Scoped flake8 is the reused interpreter's installed tool. CI remains authoritative
for full automated coverage/formatter-version parity; no full local suite or new
venv/dependency tree is created. Branch pyproject remains base version0.70.21.

## Round-1 unified remediation — candidate publication and forge ambiguity

Independent R1-1/Q1-1 and R1-2/Q1-2 were reproduced here at immutable PR head8c:
actual256→257 denial refusal **changed authority bytes and made load unreadable**;
actual GitHub account-denial request returned scope_unknown **after mint/register**.
Each producer used the real consent writer/broker entrypoint with only external
forge boundaries synthetic; no live material or upstream mint.

The corrected writer validates the entire separately copied candidate, all existing
schema/cardinality/context constraints and exact UTF-8 writer serialization bound
BEFORE any staging/replacement. Failed candidates preserve current bytes,
readability, denial/revision and caller in-memory snapshot; post-write durability
and strict readback remain required. No denial pruning is used to fit capacity.

GitHub presently supplies no genuine read-only native account identity before its
source ladder. Existing account-specific denies therefore refuse scope_unknown
BEFORE resolving a source or reading App/PAT/gh private material, minting or
registering. No app/config/name/caller account_ref is invented as identity. Current
denial is rechecked at asynchronous boundaries and before grant emission; no file
lock spans network/subprocess work. A mint/extraction already in flight at a later
withdrawal cannot be retracted by this preflight: final denial prevents emission,
and existing registration/revocation cleanup remains. **No blanket zero-in-flight
mint/extraction or upstream bearer retraction claim.**

Actual exact after-control output:

```text
257th refusal: consent_invalid; bytes_changed=false; unreadable=false
GitHub account ambiguity: scope_unknown; external_boundary_calls=[];
material_response_present=false
```

Eight additional regression controls cover denial/recipient capacities, serialized
bytes, malformed/deep context and unchanged snapshots/tombstones; App/PAT/gh source
preflight zero boundaries, ignoring forged caller account_ref. Bounded existing
real encrypted-TCP/control fixture retained: authenticated grant with1 synthetic
loopback refresh; forged recipient identity_mismatch with no added refresh;
actual CLI revoke then SAME session not_a_holder with0 added refreshes. Legacy
forge mint/audit and missing-source ladder tests pass. These are isolated source
fixtures, not live owner adoption, external provider or native UI acceptance.

## Round-2 integration — safe opaque IDs and explicit finite recovery

The existing network/identity records preserve opaque identifiers, not exclusively
new mint spellings. The shared `store.safe_component` rejects empty/dot/dotdot,
separators/backslashes/drives and C0/C1 controls, and bounds exact UTF-8 bytes;
consent uses it without normalization, storage relocation or weakened exact
network/owner/member/public-key binding. The UNMODIFIED `n_copy_receipts` suite
went from 10 failures/27 passes to **37 passes**. The reason audit's exact two
new internal receipt/proof `detail` reads are declared, not renamed/ignored:
**2 passes** after the reproduced 1 failure/1 pass.

A bare re-share after withdrawal now returns actionable `supersede_required`.
Use `credential share KEY --with PEER --scope SCOPE --request-approval`, then
normal `network approvals approve AP_ID`, then the same exact share arguments
with `--approval AP_ID`. Request is NOT authority. The normal signed
`device_onboard` envelope contains exclusive `what.credential_finite_share` and
the real anchor trio, NO fictional SSH host or setup target. Its summary precedes
normal signing (stderr for JSON); generic run admission refuses onboarding.
Owned provider/secret/MCP/forge keys use the same finite adapter, not the native
future-inheritance gate. Existing successful share JSON/prose/copy ceilings remain.

Signed apply binds current owner/network/active member/current public fingerprint,
exact owned key/scope and ALL matching deny selector/revision/reference tuples,
plus policy CAS and the current unexpired signed run. Source-record → consent →
placement locks have no await/network/SSH. Existing crypto receipt/terminal writers
are factored into internal locked helpers; public entrypoints retain their locks.
Finite projection readback and EXECUTED/terminal proof precede the final deny
removal. Final expiry/member checks run before authority publish. No future
positive permission is created. Publication failures restore original denies and
EXACT original projection bytes; failed deny compensation persists/readbacks a
withheld holder instead. A simultaneous inability to write either durable store
cannot be claimed as validated recovery; filesystem repair remains required.

Overlapping enabled future policy refuses `finite_future_scope_conflict` rather
than implicitly reviving inheritance. Exact signed provider IDs/MCP URLs prove
non-overlap; broad selectors whose exclusion cannot be proved also refuse. No key
namespace/account guess, disabling all permissions, new schema/exemption or new
reconciler was added. Exact selector-exclusion recovery is unsupported/pending a
separate design. Copy says **No automatic future-sign-in inheritance enabled by
this operation**: it does not revoke normal authorized refresh/rotation of the
same existing key/subject. No future reconciler exists in this slice; old busy
actors remain an adoption limitation.

Final bounded, isolated, serial source file results (existing Python 3.12.13;
512MiB process-group ceiling, each `run_bounded.py --timeout 120`; n0/addopts empty):
- finite public parser/dispatch + signed context/replay/fault/future guard: **27 passed**;
- unchanged copy receipts: **37 passed**; exact reason audit: **2 passed**;
- retained consent/R1 majors: **31 passed**;
- full existing wipe: **14 passed**; owned secret copies: **6 passed**;
- full existing encrypted real-link/static-key receipt: **10 passed**;
- normal approval receipt/run/terminal neighbors: **10 passed,24 deselected**;
- audit taxonomy/material filters: **4 passed,14 deselected**;
- unchanged CLI parser-demand neighbors: **4 passed,80 deselected** (caught and
  corrected the share-only optional flag read for revoke before CI; no assertion edit);
- scoped black/isort/flake8 pass; explicit existing-interpreter pyright **0 errors**.

Own actual source CLI proof also emitted `supersede_required` → `requested` →
`approved` → normal `sharing` JSON; holder=true, matching denies=0,
future_enabled=false, terminal=connected with executed finite steps. Replay
returned finite `approval_already_connected`, not an onboarding claim. Presign
summary was on stderr while JSON stdout remained exactly one object. Software
operator fixture ONLY; actual native operator prompt, live policy, installed old
actor, full remote runtime and activation are NOT validated. Independent CLI
UX/design/QA rounds still required; no merge/install/version changes.

## Round-3 durability fence — owner-local PENDING journal

QA Q3-1 reproduced HERE at cc5 with their OWN unmodified producer (1.1s): a
consent-only compensation failure left a withheld projection, but the bare public
share then EXITed 0 restoring the holder, the source record unchanged, and a
fresh Owner in the SAME session returned `grant` with material. A missing holder
alone is NOT a durable negative.

Fix: an owner-LOCAL, keyless `finite_journal.json` (plus an init sentinel)
records a PENDING fence BEFORE any mutable policy/projection write, persisted
with fsync + readback. The central `consent.check_borrow` consults it FIRST — ahead
of the legacy absent-consent fast path — so every owner borrow/mint/copy path and
the placement bulk granters refuse with `finite_recovery_pending` while a fence is
live. Actual serving holds consent→journal across its final check; positive
producers already holding placement read only the atomic snapshot (no reverse
lock). The journal is cleared LAST, under the held lock, only after the exact
current admitted signed run, EXECUTED+terminal receipts, policy/holder/capability
readbacks and a current member/key/public-fingerprint recheck. Completion writes
the existing LOCAL `set_member_capabilities` (revoke legitimately drops the last-key
capability) with `store.save` + readback, NOT the relay-first CLI variant — no
socket/native I/O under locks. Uncertainty at ANY boundary leaves an INDEPENDENT
negative fence.

Recovery is PUBLIC and signed: a retained fence requires a NEW exact finite
request that binds the fence id+revision, the retained ORIGINAL withdrawals, and
CURRENT policy/member/public-fingerprint/key/scope; apply clears ONLY that target
fence and preserves unrelated denies and future-policy state. Bare share, replay,
generic run and Owner-broker copy all refuse while the fence is live; a fresh
signed apply succeeds and same-session grant resumes. The journal is bounded
(262144 B, ≤64 records, ≤256 deny tuples, ≤200 key/≤256 ref, strict dup/schema/
symlink/enum/MAXREV), never evicts/normalizes, and refuses on a missing/corrupt/
unconfirmed initialized store. Losing BOTH durable stores (all-IO outage) is
explicitly NOT claimed recoverable: status stays unknown and borrows refuse.

Exact delta evidence at this head (existing Python 3.12.13, serial env-i HOME/
config/PYTHONPATH, `run_bounded.py --timeout 120`, 512 MiB, n0/addopts empty):
- `test_finite_journal.py`: **29 passed** (source fault→bare/replay/generic/Owner
  copy replay refused, fresh signed public recovery succeeds; crash boundaries;
  initialized missing/marker/corrupt/duplicate/symlink; bounds/schema; disclosure
  before signer; uncertain final publish retains fence; finite expired/denied/
  stale-scope/FP/revision remedies).
- `test_finite_share.py` **27**, `test_credentials_consent.py` **31**,
  `test_reason_surfaces.py` **2**, unchanged `test_copy_receipts.py` **37**,
  full `test_wipe_notice.py` **14**, full `test_secret_copies.py` **6**, full
  `test_credentials_real_link.py` **10**, unchanged parser-demand **4/80 deselected**.
- scoped black/isort/flake8 pass; explicit existing-interpreter pyright **0 errors**.
Own after-control keeps the QA injector untouched, corrects only the expected
invariant, then drives public request/show/software-approve/apply: retained
fence id/rev + ORIGINAL deny in the new signed intent, clears only the target
fence, same-session GRANT resumes (1.3s). `finite_journal.py` is a new module;
bounded/whole-store loss and all-IO outage remain explicitly unproven-recoverable.

### Round-3 design D1 / CLI-UX U1 / U2 (source CLI disclosure and remedies)

The normal human `approvals show` **and** the pre-sign `approvals approve` stderr
now print the leading key/peer/scope sentence followed by the bounded signed
policy revision and EACH matched withdrawal's selector/revision/reference (no
credential values, tokens, private keys or account-kind guesses). `--json` keeps
stdout exactly one object; the disclosure rides stderr. Own actual source capture
(request → `approvals show` → approve) shows both denies disclosed before the
signer hook runs. The summary now reads "clear only the withdrawals listed below"
so the sentence matches what is shown.

Finite-only failures keep their security codes but carry an actionable finite
remedy (regenerate the exact `credential share … --request-approval`, approve,
then apply with `--approval`), never onboarding/setup/SSH text: the `_apply`
wrapper relabels `approval_denied`/`approval_expired`/`approval_already_connected`
and the mismatch/CAS codes, leaving the normal onboarding surfaces unchanged. A
journal is refused as its own durable fence, not an onboarding error.

## Explicit deferrals / activation holds

- **Deferred — next reviewed lifecycle slice:** positive inventory reconciliation,
  monotonic lifecycle/login-chain checkpoints, negotiated strict metadata validator,
  capability advertisement, login/tick/startup/reconnect catch-up, provider/MCP hot
  revalidation. No new protocol or automatic placement is advertised here.
- **Deferred — disclosure producer:** existing PairDecision is finite; user-visible
  pairing/onboarding future opt-in capture would broaden into UI/wire/real-flow
  gates. Missing typed opt-in means no future consent. Internal typed ingress alone
  is not a completed pairing flow.
- **Deferred — foundational separate approval/PR:** reader-first all-provider
  sealing/storage writer/migration. Existing static sync can write unsealed auth.db;
  no new automatically inherited static holder is projected. 0600 is not encryption.
  Permission response explicitly says static_storage_activation_pending.
- **Live activation BLOCKED:** actual owner broker AND copy-serving processes and
  relevant writers must be proven deny-aware, with scope-specific Aida/e915 leases.
  Runtime pointer/new CLI version or a consent file is not reader-adoption proof.
  Old owner processes ignore the new file and can serve stale placement after a
  projection failure. No live config/approval/policy/material/node/installed writer
  was touched. No install/restart/version bump/release/paid call is authorized here.
- Foreign trust, local-only overrides, borrowed/copied re-export, refresh-secret
  copying and raw material metadata remain excluded. Native wire/performance/real
  pairing/transfer/live UI latency/coalescing matrices are later gates; no optimal
  performance, full native acceptance or uniform at-rest encryption claim.
