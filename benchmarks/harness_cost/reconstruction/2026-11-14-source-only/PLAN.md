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

## Bounded source-lock follow-on (after approved source-only baseline)

`ENVIRONMENT-AMENDMENT.md` proposes E1–E4 prospectively; none is accepted/frozen.
OMP/DSH/OpenCode exact source locks/manifests are retained with provenance limits;
LoP's runtime Python closure remains unavailable. Published OpenCode source/listing
clarifies the candidate's Bun build, not the lost Go-described historical binary.
All 26 non-source acquisition records remain on conservative licensing HOLD in
`binary-eligibility.json`. These source findings need delta review; they do not
request build admission or change LOCAL-only/one-image/memory/disk requirements.

## One proposal for Aida

Proceed only as a **new, LOCAL-only, source-reconstructed campaign**, after
resolving the listed gaps; do not attempt to resume the deleted run. Remote build
or execution is not proposed or authorized. Retain **ONE benchmark task image at
a time**, run its seven arms serially, and release that campaign-owned image under
explicit disk-owner lifecycle authorization before admitting the next. Never
prune shared images or another session's data. No image collection starts now. Keep this draft unfrozen
until the new environment passes requalification. The source-only work is useful
even if the full benchmark is deferred: 381/407 bound payload files are exact,
20/28 tasks have every bound file restored locally, and all 37 retained source/
closure bindings match. Publication retains 380 payload files: the additional
exact sanitize verifier is LOCAL-ONLY because GitHub push protection flags its
public synthetic Hugging Face canary. It remains at the same worktree path with
its expected hash in local-only-inputs.json; no bypass or byte substitution.
Fresh checkouts must reacquire that public local-only input before full local
verification. The campaign remains unqualified. `TASK-MATRIX.md` and `task-matrix.json` give every task and gap.

### Requalification, gated order

1. Review this proposal and authorize the exact next phase; coordinate durable
   raw-input/build/output roots with the parent and disk owner before collection.
   Finish missing data/license review. The sanitize-git-repo verifier is now
   recovered exactly from an immutable upstream2 ancestor, without substitution.
   Nine freshly observed public registry rows now include all seven previously
   missing image digests, explicitly NOT recovered historical receipts. Acquire
   only by frozen content digest after review; never trust a tag to stay fixed.
   Do not infer DEV payload hashes.
2. Resolve all UNPINNED rows in BUILD-RECIPES.md; independently review the new
   dependency locks. Select exact host/Linux interpreters, OS/toolchain and build
   tools. A different lock is a new environment, not restored historical bytes.
   Local build admission additionally requires disk desk session 00e6a1cd8f62 to
   confirm >=30 GiB free AND a fresh memory GO. This is a necessary admission
   floor, not sufficient proof of the working-set estimate or unknown overhead.
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
   The historical driver is already task-major then harness-major (driver.py
   loops at lines 1375/1399): one-image serial retention does NOT justify changing
   task order or arm order. Any later scheduling change needs a visible prospective
   amendment before model outcomes; do not quietly reorder to fit resources.

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

**Primary required policy: LOCAL, serial, ONE benchmark task image retained.**
The 21 surviving size rows plus nine new public registry observations sum to
**2,421,466,837 compressed bytes**; largest single tag reports **202,928,151 bytes**.
These are metadata observations, not a measured local peak or expansion bound.
Under an explicit 4x expansion assumption, one image costs 0.202928151 +
0.811712604 decimal GB; add assumed 10 GB shared toolchain/build cache and 10 GB
logs/output: **21.014640755 GB**, rounded upward to a **25 GB planning baseline**.
The largest task storage allowance is 10,240 MB; treating that as an additional
10.24 decimal GB yields **35.24 GB including an active writable task allowance**.
This is NOT a hard bound. Docker VM allocation/overhead, intermediate build layers,
actual expansion, optional model weights and host overhead remain unknown; the
>=30 GiB admission floor alone does not prove those fit. `planning-metadata.json`
records the arithmetic and sources. Obtain disk-desk admission and memory GO
before any heavy work, and preflight actual free space for each image transition.

The largest-RAM task (`torch-pipeline-parallelism/task.toml`) requests 1 CPU,
8,192 MB RAM, 10,240 MB storage and no GPU. Across restored task configurations,
`overfull-hbox` requests 2 CPUs; host/services/VM memory is additional and unknown.
Do not call the task's RAM allowance a measured process-tree peak or host budget.

The old **75 GB** figure was a loose **all-images-retained** planning envelope
using nine unknown images each assumed <=1 GB, 4x expansion and 20 GB tooling/logs.
It is retained only as historical planning context: it is NOT the required serial
working set and NOT authorization to retain all images. Do not count APFS logical
`du` sums as physical usage or automatically clean another session's storage.

### Objective eligibility and amendment rule — freeze before model outcomes

Before any model outcome is observed, freeze the task/arm matrix, source/data
hashes, licensing/admissibility decision, exact version and platform availability,
configuration comparability, resource fit, oracle validity, contamination policy,
egress and metering criteria. Apply those criteria symmetrically to every arm.
A task or artifact is blocked only by a recorded objective failed prerequisite,
not by expected difficulty, observed success, relative performance or cost outcome.

Retain all 28 held-out tasks and the declared ordering by default. If a missing
artifact/licence cannot be cleared, propose a version/task amendment with the
specific failed criterion, comparable replacement rationale, all affected arms
and changed estimand; independently review it BEFORE freeze/model dispatch.
Missing OpenCode historical runtime identity is a prerequisite gap, not grounds
to silently substitute another implementation with the same version string.
If a new problem appears after outcomes, preserve every outcome and blocked row;
stop the affected work and preregister a new campaign/amendment prospectively.
Never retroactively exclude poor outcomes or optimize the eligible sample against
model results. No outcome-conditioned exclusions or amendments are permitted.

### Hard spend rule

**No spend now.** Future lane hard cap is at most **$100**, trial cap **$5**,
operator programme total at most **$800**, including known prior and concurrent
campaign spend. Dispatch cap = min($100, $800 - reconciled programme spend -
other outstanding reservations). Unknown prior spend or unavailable meter means
**$0 dispatch**, not an assumed fresh $800 allowance. These are inherited budget
constraints, not derived new experimental thresholds. Fresh prices/model identity
must be checked only in the separately authorized qualification phase.

### Durable receipts and lifetime

Published content in this directory is public source or public metadata.
One exact upstream verifier is retained LOCAL-ONLY under a precise .gitignore
entry after remote push protection rejected its public synthetic canary; see
local-only-inputs.json and PUBLICATION-BLOCK.txt. It is not private operator data,
but it is NOT included in git or the published SHA256SUMS. Its exact hash and
immutable acquisition URL remain in source receipts/local-only-inputs.json.
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
