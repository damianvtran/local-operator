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
