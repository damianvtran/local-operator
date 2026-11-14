# Agent review remediation — round 1

Reviewed base: 4c8b2269d761ee5819fd00275166d784b7e0219a.
Reviewer session: b9579b98ed06 (findings relayed by parent).
Scope: one coordinated source-only remediation commit after initial review completed.
Commit identity: this file ships with the unified remediation; resolve its SHA
from git history. Parent handoff names the exact SHA for R1/R2/R3 together.
No PR, merge or independent sign-off is claimed by this note.

| ID | Disposition | Change and evidence |
|---|---|---|
| R1 MAJOR | fixed in this commit | PLAN/build inputs now explicitly LOCAL-only, one task image retained, historical task/arm order preserved; primary serial planning baseline 25 GB, 35.24 decimal GB including task writable allowance under stated assumptions. Prior 75 GB all-images envelope is historical context only. Disk desk >=30 GiB + memory GO required; VM/build/weights/host overhead unknown. planning-metadata.json derives metadata/arithmetic; not a measured peak. |
| R2 MINOR | fixed in this commit | verify_static.py derives and compares each complete per-task row, counts, gaps and immutable source identities against receipts, not only totals. R2-RED.txt records the original verifier accepting a tampered complete_exact/gaps claim; test_reconstruction.py checks rejection after the fix. |
| R3 MINOR | fixed in this commit | PLAN freezes objective symmetric eligibility/amendment criteria before model outcomes and explicitly forbids outcome-conditioned task/arm exclusions or retrospective amendments. No observed cherry-pick was alleged or inferred. |

## Coordinated pin-gap work

* Exact sanitize verifier recovered from immutable upstream2 ancestor, not a
  local transform. Public FAKE_* canaries are classified only at the exact
  prereg-bound path+digest, limited to expected AWS/GitHub/Hugging Face families;
  tampered hash, another path or new family fails. Regression test covers each.
  GitHub push protection blocked publication of its synthetic Hugging Face value.
  No bypass: raw exact bytes remain LOCAL-ONLY at the unchanged worktree path,
  excluded from the amended source commit. See PUBLICATION-BLOCK.txt and
  local-only-inputs.json. Local exact recovery=381; published payloads=380.
* All seven missing image digest candidates now have NEW public tag-metadata
  observations; two historical digest/size controls match. These are prospective
  acquisition identities, not recovered old runtime receipts. No images pulled.
* OpenCode 2.0.27 correct scoped npm wrapper/Linux package SRI and immutable source
  candidate are recorded. Historical runtime identity discrepancy (adapter says
  Go, inspected current versioned source has no Go files) remains an explicit
  review/compatibility gate; proposed identity amendment is not approved/frozen.
* 26 non-source blobs have immutable public acquisition URLs, Git blob IDs, sizes
  and historical expected SHA256. Bytes were not fetched or SHA256-verified;
  root Apache-2.0 does not imply independent third-party/media rights clearance.
* Source and metadata observation manifests were appended as new files; earlier
  receipts and STATIC-CHECKS.txt remain unchanged. Incremental metadata inventory
  now aggregates named observations instead of discarding earlier package pins.

## Evidence boundary

R2-RED.txt is redacted only to replace the operator's home prefix with `~`.
REMEDIATION-CHECKS.txt and REMEDIATION-CHECKS-PUBLICATION.txt contain bounded
stdlib tests plus actual static-verifier
and incomplete-campaign refusal command results for this remediation. No recovered
payload code was executed, no service/browser was started, and no Docker, install,
build, inference, credential access or paid benchmark operation occurred.
The runtime remains unqualified; original seven outcomes/ledger remain lost.
