# DRAFT — campaign reconstruction-20261114-source-v1

Not frozen, not qualified, not a continuation. The original seven trial outcomes,
ledger and receipts are lost. Historical source is not paid evidence.

Authorization for this phase: source downloads and static checks only. No Docker,
builds, installs, services, inference, credentials or paid calls. Current spend: $0
(no paid dispatch authorized). Original lane cap $100; operator programme cap $800;
any future paid campaign must account for prior actual spend before dispatch.

The dirty surviving checkout is untouched. This isolated sparse worktree starts
at 2078250d8e990ee0840b4d045964ed8a68871d08, not latest main, deliberately:
this is historical reconstruction, not a product PR. No PR or merge requested.

Public source receipts and restored payloads are retained here in git indefinitely
until the operator explicitly approves archival/deletion. A HOLD label alone is
not deletion protection. No private payloads or credentials belong in this tree.
If private inputs become necessary, coordinate an exact durable path with parent
before collecting them. No raw private storage was allocated in this phase.

Qualification gate: recover exact per-file bindings; inventory every unavailable
byte or lock; review source-only proposal; separately authorize dependency
resolution/builds; then test service health, metering and egress negative controls,
harness identity and oracle correctness before any separately authorized paid run.
All new evidence uses this campaign ID, never historical run IDs.

## One proposal for Aida

Proceed only as a **new, source-reconstructed campaign**, after resolving the
listed gaps; do not attempt to resume the deleted run. Keep this draft unfrozen
until the new environment passes requalification. The source-only work is useful
even if the full benchmark is deferred: 380/407 bound payload files are exact,
19/28 tasks have every bound file restored, and all 37 retained source/closure
bindings match. `TASK-MATRIX.md` and `task-matrix.json` give every task and gap.

### Requalification, gated order

1. Review this proposal and authorize the exact next phase; coordinate durable
   raw-input/build/output roots with the parent and disk owner before collection.
   Finish missing source/data/license review; resolve the single mismatching
   sanitize-git-repo file or declare a prospective task amendment (never silently
   substitute). Reconstruct missing image identities from independently verified
   public manifests, not mutable tags. Do not infer DEV payload hashes.
2. Resolve all UNPINNED rows in BUILD-RECIPES.md; independently review the new
   dependency locks. Select exact host/Linux interpreters, OS/toolchain and build
   tools. A different lock is a new environment, not restored historical bytes.
3. Separately authorized build-only lane: one shared dependency/toolchain tree,
   compiled-source provenance and full per-file/runtime manifests. No model keys
   or external model traffic. Setup failure stops the lane, not paid fallback.
4. Loopback-only service and synthetic wire tests: unknown/wrong-run identity,
   malformed request, wrong model, missing pricing, exhausted budget, disconnect
   and stream settlement; gate deny for an open non-forwarded listener and blocked
   external host; wrong source/binary/CA negative controls. Capture actual
   requests/responses, ledger side effects and cleanup PIDs. Test every harness
   against a local fixture endpoint without paid inference.
5. One oracle sweep for the proposed 30-task selection (28 held-out + 2 DEV),
   hidden-fixture contamination classification, before/after isolation receipts,
   quiescence and grading, egress and all seven runtime/wire attestation rows.
   No sampling shortcut based on old7 outcomes. No claim of recovered outcomes.
6. Review and freeze NEW prereg/environment/evidence hashes. Only after actual
   old spend reconciliation and explicit paid authorization may paid trials begin.
   Retain historical ordering and 28 x 7 = 196 design only if requalification and
   model/price availability support it; otherwise preregister a visible amendment.

### Time and disk envelope (planning assumptions, not measurements)

Static task TOMLs give 28 held-out agent timeout sums of 45,750 seconds and
verifier sums of 45,360 seconds: a 7-arm all-timeout ceiling is
7 x (45,750 + 45,360) / 3,600 = **177.16 hours**, excluding builds, pulls,
service startup, DEV qualification and retries. This is NOT expected runtime.
A 28-task verifier-only oracle pass can consume up to **12.6 hours** of verifier
budgets alone. DEV task durations are missing and must be added before scheduling.

For initial scheduling reserve **20–32 hours** for requalification, explicitly
assuming 4–12 hours build/lock work, 2–4 hours synthetic services/wire/negative
controls and up to 12.6 hours held-out verifier budget, rounded upward; DEV
oracles/solution setup and human review can extend it. No measured success or
runtime distribution survives, so this is not a performance forecast. Native
attempt budget remains 3 failed/cancelled attempts or 6 hours per costly lane,
then escalate; do not spend the full envelope retrying a broken setup.

Surviving subset metadata records **1,753,275,201 compressed bytes** for 21
images, max **190,685,882 bytes**; this is metadata, not a current registry check.
Assuming each of nine missing entries still meets the historical 1,000,000,000
byte selection cap, compressed total <=10,753,275,201 bytes. Assuming 4x expanded
image storage +10 GB shared toolchains/build cache +10 GB logs/output, reserve
**about 75 GB incremental** (10.754 + 43.014 + 10 + 10 = 73.768 decimal GB,
rounded upward). This is a planning envelope, not a measured or hard disk bound;
layer sharing can reduce it and expansion/model artifacts can exceed it. Verify
actual free space with the disk owner and preflight each build/pull. Do not count
APFS logical du sums as physical usage; no automatic cleanup of other sessions.

### Hard spend rule

**No spend now.** Future lane hard cap is at most **$100**, trial cap **$5**,
operator programme total at most **$800**, including known prior and concurrent
campaign spend. Dispatch cap = min($100, $800 - reconciled programme spend -
other outstanding reservations). Unknown prior spend or unavailable meter means
**$0 dispatch**, not an assumed fresh $800 allowance. These are inherited budget
constraints, not derived new experimental thresholds. Fresh prices/model identity
must be checked only in the separately authorized qualification phase.

### Durable receipts and lifetime

Everything in this directory is committed public source or public metadata.
`source-receipt-*.jsonl` and `metadata-receipt-*.jsonl` are exclusive-created
observation manifests: append a new file for another collection; never rewrite
an earlier receipt. `SHA256SUMS` binds retained files; git binds the collection.
`derived-subset.json`, matrix and partial locks are generated derivations, not
append-only original observations. The initial future evidence contract is
`evidence.schema.json`; future receipts require campaign/source/artifact hashes,
privacy outcome, origin and a previous-receipt hash chain. No trial ledger or
historical trial records were created. No private path was needed; parent ACKed
that public receipts stay committed here and will coordinate any future raw root.
No HOLD marker is asserted to override the disk owner's deletion policy.
