# Prospective environment amendment — DRAFT, not accepted or frozen

Baseline: 229694dad8b53c05ae884508c5247e2a95e9be79 (source-only review clean).
This bounded follow-on investigates static provenance and dependency manifests;
it does not authorize an installer, binary/media download, runtime execution,
Docker, service, model call, credential access or disk/memory admission request.

The deleted historical environment cannot be asserted equivalent from package
version strings. Any new source/artifact/dependency selection must be visibly
amended across the affected arms before model outcomes and independently reviewed.
Original seven outcomes, actual spend receipts and installed-file attestations
remain lost. No old run ID or outcome is reconstructed.

Eligibility stays symmetric: exact source/data bytes, licensing and acquisition
rights, compatible configuration, locked dependencies and platform, oracle and
egress/metering qualification are prerequisites for every affected arm. A failed
prerequisite blocks qualification; it is not an outcome-conditioned exclusion.
LOCAL-only, one benchmark task image retained, unchanged task/arm order, disk-desk
>=30 GiB plus fresh memory GO remain necessary future build gates, not proof that
unknown Docker VM/build/weights overhead fits. No heavy work starts in this phase.

## Findings from this bounded pass

`source-closure-findings.json` binds the exact public observations and retained
manifests; its `source_receipt` names the new append-only metadata receipt.

### OpenCode: candidate build identity clarified, historical identity NOT recovered

The published @opencode/cli 2.0.27 package declares a platform-binary wrapper;
its package.json and postinstall.mjs bytes both match the public package listing's
SHA256 integrity. Postinstall copies the selected optional platform binary to the
command path. The Linux-x64 listing declares a 210,367,968-byte binary and a file
integrity value; no binary bytes were acquired, hashed locally or executed.
The wrapper's .exe command suffix alone does not identify its runtime language.

Independently, immutable source 1f4d07a4a519185955ae8ab2ea3607805509e386 declares
version 2.0.27 and its retained CLI build script explicitly uses Bun.build compile
plus Bun release asset fetching, not Go. Its exact source bun.lock is retained.
No binary backend/magic was inspected: this establishes the pinned source's build
mechanism, not an attestation that the published platform binary came from it.
This resolves the CURRENT candidate's declared build mechanism, not the identity
of the deleted original binary described as Go by the old adapter comment. No
surviving installed-file manifest or binary is available to settle that history;
no amount of new tag metadata proves historical equivalence.

### Source dependency locks recovered, not complete build recipes

* OMP 18.11.2: immutable source 470b7efb67673960d37e3e64584669e6e9caad1c;
  exact bun.lock, root and CLI manifests, and both declared patches retained.
  Public npm provenance names that commit; statement subject SHA512 agrees with
  the previously pinned npm tarball SRI. Sigstore/provenance signatures were NOT
  cryptographically verified in this phase. Root manager declaration is a range
  (bun@>=1.4), not an executable pin; native/optional build inputs remain open.
* DSH 0.2.0-rc.2: tag dsh-v0.2.0-rc.2 resolves to source
  639ed015397290b3745d163aafe02ffee4aa3f84. Exact pnpm-lock.yaml, workspace,
  root/CLI manifests and seven declared patches retained. Every patch's raw
  SHA256 occurs in the lock's patchedDependencies. Source CLI version matches
  published version; publish attestation was unavailable. Source manager names
  pnpm@11.7.0, but executable hashes/Node/native hooks remain unpinned. Workspace
  links and vendor overrides require the SAME source checkout, not an npm-only
  install with a guessed resolution.
* LoP v0.94.22: exact source tree has no root Python runtime lock. Retained
  flake.lock pins development Nix inputs; flake.nix provides Python/pip/virtualenv,
  NOT a resolved pip runtime/extras closure. Extension or OSWorld locks are
  unrelated and cannot fill this gap. Source-only inspection cannot reconstruct
  the missing historical Python installation; no resolver/build was run.

These recover source-resolution inputs for OMP/DSH/OpenCode. They do NOT close
all-arm build admission: package managers/interpreters, OS/native tools, generated
or downloaded build assets, target extras and installed-file provenance still
need explicit locks and subsequent separately authorized qualification.

### Binary/media acquisition eligibility

`binary-eligibility.json` evaluates each of the 26 existing public outer-blob
records: **zero cleared, zero acquired**. This is a conservative HOLD, not a legal
conclusion that distribution is forbidden or that bytes are irretrievable.
The repository-root Apache-2.0 grant was not promoted into a file-specific media,
weights, encrypted-archive or embedded-repository licence. Video README explicitly
attributes a YouTube source; DOOM resources, MNIST-like image/model provenance,
Git-bundle contents and encrypted archive contents require independent review.
Public URLs, Git blob IDs and expected SHA256 permit future acquisition identity
checks but do not prove rights or matching bytes without collection.

## Objective prospective amendment proposal (before outcomes)

E1 — **new environment, never historical restoration**: retain the declared seven
versions and historical task/arm ordering, but freeze newly reviewed acquisition
artifacts, source commits and complete per-platform locks for EVERY arm. LoP and
Hermes keep their declared original source revisions. OMP/DSH adopt the newly
retained same-version source-lock candidates only after publish/source review.
Claude/Codex retain exact versioned vendor distribution pins; their new installed
manifests must be generated just as for the other arms. No arm inherits lost
runtime/S4/toolwire/oracle evidence, and no arm gets a weaker identity gate.

E2 — **explicit OpenCode identity choice**: propose the versioned public Bun-built
@opencode/cli(-linux-x64) 2.0.27 candidate at the retained source/SRI as the NEW
campaign arm, rather than claiming it is the original Go-described binary.
Independent review and campaign-owner acceptance must approve that changed
identity/estimand BEFORE freeze or model outcomes. Until accepted and compatible
in real-path qualification, that arm and the full seven-arm campaign stay blocked.
A different release or implementation is not implicitly authorized.

E3 — **complete locks, not just analysis requirements**: LoP needs a prospectively
resolved runtime/extras/build-backend lock; all arms need exact shared host/Linux
base/interpreter/manager/native build pins, with any per-arm differences recorded.
Resolution/install/build require a separate authorization and disk/memory admission;
this source-only proposal does not grant either. Freeze all dependency/asset bytes
and test egress/metering/source identity negative controls before inference.

E4 — **task rights before selection changes**: retain the 28-task estimand while
seeking objective clearance for all 26 inputs. If a specific input cannot be cleared
or obtained, stop and submit an independent, explicit sample/estimand amendment
applying symmetrically across arms BEFORE model outcomes. No task exclusions or
replacement sample were enacted here; no performance-conditioned selection.

Technical disposition after clean source-only review of a90e62405f94 (parent
instruction, recorded before any model outcomes): E1/E3 new-environment policy
ACCEPTED; E2 public same-version OpenCode candidate ACCEPTED CONDITIONALLY on
future identity/CLI/wire qualification; E4 RETAINS all 28 tasks with NO sample
amendment and NO rights assumption. This is NOT input freeze or runtime approval.
Historical environment byte recovery is not possible with the missing installation/
attestation artifacts. The exact next resolver-only proposal and unfulfilled disk/
memory/daemon/artifact gates are in RESOLVER-ADMISSION.md and RESOLVER-COMMAND.md.
No Docker/resolver/install/build admission or execution is implied by this policy.
