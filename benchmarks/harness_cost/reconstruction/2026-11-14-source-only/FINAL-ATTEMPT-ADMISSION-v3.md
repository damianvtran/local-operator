# Final03 prospective native admission — SOURCE ONLY, NOT a desk GO

Base: `e4c934242565f32bc29216880a031f3508662e80`, parent-relayed R19 CLEAN.
R18 meter/test bytes remain FROZEN. This new cap/admission/CRLF delta needs its
own independent review. No native app/VM/helper/pull, account/model/paid call,
settings write, dependency install or build was performed. No GO, real admission
or registered final03 claim is produced by this source task. Unit GO/claims exist
ONLY in private temporary fixtures with pwd identity patched; native effects are
replaced by an inert driver. Lost ledgers/outcomes remain lost.

## Two explicit decisions, one narrowly bounded exception

Parent verified these actual operator answers:

- `a-d2d2`: **Authorize one final bounded native attempt**.
- Origin `00e6a1cd8f62`, `a-9509`, relay `a-1ca7`:
  **Accept the 8.2 GB / 14 CPU setting**.

These are source-bound in `final_attempt_admission.DECISIONS`, not flags such as
`verified:true`. They authorize ONLY one final resolver attempt, namespace
`linux-amd64-03`, and do not reuse a previous desk lease. Paid/container/legacy
campaign execution and general builds remain HOLD through the unchanged
fail-closed generic execution gate. Source verification flags cannot execute.
Failure/cancel/exception after consumption means STOP; no automatic fix/retry,
new namespace04, claim deletion or in-memory capability restoration.

## Configured capacity is NOT actual usage

The configured `docker info` cap is now **8,600,000,000 bytes /14 CPUs**. Exact
integer values only: no booleans, strings, floats, zero or over-cap values. This
admits the accepted8.2GB/14CPU setting and bounds tolerance at8.6GB. A first
successful docker-info result over either configured bound aborts before any
pull/container. Existing first-API/readiness semantics remain.

Actual attributed VM helper **RSS CAP stays4,300,000,000 bytes**, including exact
boundary acceptance and over-bound abort. RSS is not guest/whole-host footprint.
The old `CAP` constant and all RSS guards remain unchanged; new configured
constants are separate. Container **1 CPU/2GiB**,128pids,840s inner/900s outer,
120s readiness,840s whole-work and60s cleanup budgets,10s RSS samples,
fleet<=26000MB, start free>=30GiB, free-stop25GiB, raw allocated growth<=5GiB,
swap<=12,000,000,000bytes, one-image/exact-owned cleanup and foreign-work fences
are unchanged. No Desktop settings read/write/bypass has been added.

## External trust anchors and actual final-attempt command

The source overlay SHA, reviewed source HEAD and admission SHA must come from
the independent review/desk's trusted communication—not be self-computed and
called approval. Hashes prove bytes, not the identity of a person who typed them;
this uses the same explicit external-hash trust model as the reviewed source gate.
Owned local files/JSON booleans alone do not grant permission.

**PROPOSED ONLY, do NOT run before independent review AND a NEW desk GO:**

```sh
python3 -B "$KIT/reconstruction/2026-11-14-source-only/lifecycle_proposal.py" \
  --execute \
  --source-generation "$KIT/reconstruction/2026-11-14-source-only/generation-overlay-v3.json" \
  --source-generation-sha256 "$REVIEW_APPROVED_OVERLAY_SHA256" \
  --reviewed-source-head "$REVIEW_APPROVED_COMMIT" \
  --admission ~/benchmark-runs/reconstruction-20261114-source-v1/admissions/linux-amd64-03.json \
  --admission-sha256 "$DESK_APPROVED_ADMISSION_SHA256"
```

The protected0600 admission must bind ALL of the following (not a live example):

- All original admission fences remain: exact a-7072 quote and truthful prestart
  skipped acknowledgement (never a verified-before-start field), exclusive
  Desktop lease, explicit absolute Docker.raw path and unchanged direct-backend
  UI-effects acknowledgement. These known schema prerequisites are checked
  BEFORE consumption, not left to predictably fail after burning the claim.
- Exact03 attempt, scope resolver-only, fresh GO and exact `DECISIONS`/`LIMITS`
  dictionaries from the source module. Only pinned
  `python@sha256:9bbb8720ae0a24a6ca8dd678bfdf57818fe70caf54c52a53ec01d8db43405056`,
  platform linux/amd64. Optional pull remains default NO; YES needs fresh explicit
  exact-digest full-scope signoff, never another image or general pull.
- `full_scope_review`: kind full-scope, clean, independent reviewer, reviewed
  source_commit, source_generation_sha256 and actual current six native-file
  SHA256 values. The review/desk head must equal `--reviewed-source-head` and both
  overlay hashes must equal the ACTUALLY validated current source overlay.
- `desk_go`: fresh opaque receipt_id, the same source_commit and overlay hash,
  UTC issued_utc/start_by_utc/expires_utc. Issue/start-by may not span more than
  the75min lease; stale, future-issued, expired or incoherent records reject.
- Existing slot_start_utc/slot_end_utc/valid_until_utc, with expiry exactly binding
  the slot end; at least960s remain at initial/start checks, slot end never later
  than claim-time+75min. Thus actual native start has no longer than75min, often
  less. Fleet must be an integer within26000 and fresh_memory_go explicitly true.
- Existing prestart-validation SKIPPED acknowledgement, ownership/UI-effect
  acknowledgement, artifact/source/cap/resource checks remain mandatory in
  NativeOps preflight. The new decisions do not fabricate prestart verification.

## Permanent claim and unforgeable-in-process monotone capability

After source/GO/decision/head/hash/time validation and BEFORE NativeOps is
constructed, claim exactly:
`~/benchmark-runs/reconstruction-20261114-source-v1/claims/linux-amd64-03.claim.json`.
The registered root is anchored to the native account via pwd, not Docker CLI
HOME. Claim directory0700, claim0600, owner/regular-file/no-symlink checks,
`O_CREAT|O_EXCL|O_NOFOLLOW`, file fsync and directory fsync enforce exclusive
persistent consumption. A partial claim is terminal. No recovery path deletes,
rewrites or releases it. Simultaneous starts get exactly one winner; a restarted
interpreter cannot reconstruct an active capability from a claim/verified JSON.

The controller receives a module-registry-bound permit, not a serialized allow
flag. Controller construction, native start and resolver handoff each consume
monotone sub-capabilities. Same-object replay is rejected too. Every preflight
and first launch rechecks exact GO/claim/source bytes and current times. Main
retires in-memory authority on every completion/exception while leaving the
persistent consumed claim untouched. A setup failure after claim also burns it.

The resolver gets ONE inherited pipe via pass_fds; the environment carries only
the numeric FD locator. The frame is <=min(512,PC_PIPE_BUF)bytes, fully written and parent-write-end
closed BEFORE Popen. The child performs one NONBLOCKING read and closes it; an
unfilled/partial forged FIFO cannot stall beyond the work budget. Child requires a FIFO, direct claimed parent
PID, exact persistent nonce/claim/source/head/GO hashes, current GO and registered
admission path. A regular file, allow boolean, replayed/empty pipe, changed GO,
wrong parent or another attempt rejects. Standalone direct supervisor execution
cannot mint a second claim or bypass source/permission checks. Anonymous Docker
HOME/config isolation is unchanged; no credential helper/token is involved.

## Source generation v3 / R19 minor

Use `source_generation.py` with explicit `generation-overlay-v3.json` and the
externally approved SHA. The immutable738-file standalone baseline and exact old
meter bytes stay unchanged. Old v2 overlay/docs/results and historical bindings
are RETAINED, not re-labeled current. v3 binds the narrowly declared admission,
metadata/test/wiring deltas and all unchanged rest; frozen meter/test and
owned-session/Darwin-identity/broker modules remain exact. General paid/legacy
entrypoint bytes are unchanged; their generic execution function still refuses.
The source-only receipt still says runtime_qualified=false and
execution_authorized=false: it is never the final-attempt permission itself.

R19 minor is fixed: derived SHA256SUMS is checked as raw bytes against
`expected.encode('utf-8')`, not `read_text()` newline normalization. Actual CRLF
tampering now rejects. Source SHA graph remains acyclic; final approved overlay
SHA is supplied out-of-band, not embedded inside its own hashed document.

## Targeted proof and deliberate limits

`test_final_attempt_admission.py`: real private temp claims/GO bytes, actual source
validator, exact integer/configured/RSS boundaries, actual lifecycle stubs proving
configured/RSS failures occur before pull/container, both decision bindings,
wrong scope/namespace/head/digest, expiry/stale GO, limits coercion and reviewed
file mismatch. Tests cover one claim only, restart, cancellation, partial claim,
concurrent starts, mutated GO, forged booleans, pipe one-use/parent binding and
actual lifecycle CLI with an INERT replacement effect factory. No test launches
Desktop or a real resolver. Updated `test_source_generation.py` covers the new
missing-permit path and CRLF tamper; old cap-dependent lifecycle expectations are
updated, not weakened to bless higher RSS.

Results and guard byte/AST proofs are in `FINAL-ATTEMPT-ADMISSION-v3-RESULTS.txt`.
This is not native qualification or a new desk lease. Return for independent
review, then only the parent/desk can supply fresh pinned GO identity and resource
admission. No financial17/provider/Radient/MCP/account/campaign authorization is
inferred. Last-attempt failure/cancel means STOP, with truthful cleanup evidence.
