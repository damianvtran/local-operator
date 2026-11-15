# Explicit standalone generation gate v2 — SOURCE ONLY, execution HOLD

Base meter review: round18 `fc9ea6c37d75`, CLEAN/TERMINAL at
`74e9b4b1fb0d22a8f919e909488a21a3082d5c83` (parent relay). Its meter and
`test_meter_privacy.py` bytes are FROZEN. This new generation-gate/wiring delta
requires its OWN independent review. No old run/ledger/outcome was recovered.

## Trust model and hash-cycle boundary

`generation-baseline-v1.json` binds ALL738 tracked public-kit files from immutable
base `9dee883473441bc1db300abb975cd11297c6ef01`, with size/SHA256/Git-blob identity.
Its SHA256 is hard-pinned in `source_generation.py`:
`a27405d15656ba385fed1c8b3bfb4752a7658a301483a950c9cb737fe3ad412b`.
`historical-meter-pre-privacy.py.txt` is the EXACT old Git-object-derived public
source, not a reimplementation; SHA256
`e5d5d626e36e89dc7544eac3f826623fbd264e285dc35f2431fdf35540e7cc5a`.
The validator independently checks both its SHA256 and original Git blob SHA1.
No live `.git`, Git CLI, source server, registry, credentials or build is needed.

The immutable baseline and explicit named delta/addition sets prevent turning
an unrelated source edit into a supposedly valid new generation. The current
meter AND test SHA256 are additionally hard-pinned to round18's exact bytes.
All five wiring-file SHA256 values are also hard-pinned: resealing a modified
lifecycle/runner file into a new overlay digest cannot authorize extra behavior.
The only existing-file edits allowed here are that meter delta and minimal gate
wiring in container driver, legacy runner, lifecycle, extracted resolver recipe
and build recipe. Ownership/identity module files remain unchanged. Historical
bindings, rate cards, old receipts and frozen preregistration remain unchanged.

`generation-overlay-v2.json` binds the current hash/size of EVERY public-kit file,
including the validator itself, all tests, immutable baseline and historical
source, except TWO manifest-control files:

1. The overlay itself: its independently reviewed SHA256 is supplied EXPLICITLY
   to the real command. It is never inferred from a neighboring receipt.
2. Reconstruction `SHA256SUMS`: its entire canonical byte representation is
   separately recomputed and compared; it includes the overlay and all other
   reconstruction files, excluding itself. This is a normal acyclic manifest
   construction, NOT an allowance to ignore one historical mismatch.

The final review-supplied digest is intentionally outside this document/overlay
hash cycle. Computing a digest yourself establishes byte identity, not that an
independent reviewer approved it. Unknown versions, wrong base, duplicate JSON
keys, missing/unknown files, symlinks, unsupported deltas and false runtime claims
reject. One pre-existing local-only immutable public canary is allowed solely
at its exact payload path/hash; its bytes are NOT copied, redistributed or used
to exempt arbitrary files. Standalone public kits omit that private fixture.
VCS/bytecode cache metadata is not source; this is NOT dependency/runtime proof.

## Exact INERT verification commands — no runtime effects

Set `KIT` to the absolute public-kit directory (`benchmarks/harness_cost` in the
worktree) and obtain `APPROVED_GENERATION_SHA256` from independent review. There
is NO default-old generation or automatic trust in a `verified:true` receipt.

```sh
OVERLAY="$KIT/reconstruction/2026-11-14-source-only/generation-overlay-v2.json"
python3 -B "$KIT/source_generation.py" \
  --source-generation "$OVERLAY" \
  --source-generation-sha256 "$APPROVED_GENERATION_SHA256"
python3 -B "$KIT/experimental/coding/driver.py" --verify-source \
  --source-generation "$OVERLAY" \
  --source-generation-sha256 "$APPROVED_GENERATION_SHA256"
python3 -B "$KIT/experimental/runner_v2.py" --verify-source \
  --source-generation "$OVERLAY" \
  --source-generation-sha256 "$APPROVED_GENERATION_SHA256"
python3 -B "$KIT/reconstruction/2026-11-14-source-only/lifecycle_proposal.py" \
  --verify-source --source-generation "$OVERLAY" \
  --source-generation-sha256 "$APPROVED_GENERATION_SHA256"
```

Expected success: source_verified true, runtime_qualified false,
execution_authorized false, git_required false, exact generation/overlay digest.
Missing selection/digest, wrong version/base/hash or any tamper exits nonzero.
The default inert lifecycle proposal is NOT admission. Mixing verify-only and
execute is refused. These commands never read registered private admissions,
contact APIs, inspect/start Docker, create a VM or build/install dependencies.

## Actual command/preflight wiring

- Container `driver.preflight` verifies the real source bytes before legacy
  checks, credentials, meter health, task artifacts or dispatch. Its explicit
  `--verify-source` branch uses the same validator without calling those checks.
- Legacy runner verifies before configuration writes, home/profile augmentation,
  attestation or execution. Configuration-only is not an old-generation bypass.
- Lifecycle CLI verifies before NativeOps construction/admission reads; NativeOps
  preflight revalidates rather than trusting a retained boolean. Independent
  review must bind the exact source-generation SHA256, replacing moving Git HEAD
  checks for standalone use. Existing six lifecycle-file checks remain.
- The ACTUAL fenced resolver supervisor revalidates `BENCH_SOURCE_GENERATION`
  and `BENCH_SOURCE_GENERATION_SHA256` before socket checks, Docker commands or
  output creation. The lifecycle passes both explicitly alongside its isolated
  Docker CLI HOME; no source identity is derived from HOME. Resolver provenance
  uses `generation:<approved SHA256>`, not an unavailable Git command.
- Build recipes require the gate before any install/build/service instructions.
  Those instructions remain UNEXECUTED and blocked; this gate does not turn a
  prose command, direct `docker` invocation or maliciously replaced Python into
  authorized work. Existing published-kit release manifests remain historical;
  no packaging/release claim is made by this source-overlay proof.

## Budget HOLD cannot be waived by source proof

Parent reports first native start05:47 and6h wall-clock budget exhausted by12:30,
with2 failed attempts. No remaining-attempt availability is claimed. Candidate
execution entrypoints fail closed after source verification with the explicit
budget HOLD BEFORE native/account/build effects. BOTH operator decisions are
still required: extension-versus-stop ask `a-d2d2`, resource/cap ask `a-9509`
(relay `a-1ca7`), plus independent new manifest review and NEW desk GO. No answer
is presumed. The candidate does not implement a future operator-decision waiver;
removing/replacing the hold needs a separately reviewed admission change after
actual answers. Source-only work may continue; no heavy/native evidence may run.

## Targeted actual-path/tamper proof

`experimental/test_source_generation.py` runs the REAL validator and the four
real inert CLI entrypoints. It supplies altered filesystem reads instead of
mutating shared reviewer-frozen meter/test files; no source tree/venv is cloned.
New meter, new tests, historical source, baseline, unrelated owned runtime,
restricted edge source and validator tamper each reject. Resealing unrelated
runtime/edge edits into a newly hashed overlay STILL rejects immutable-baseline
constraints. Missing/unknown files, derived-manifest tamper, duplicate keys,
unknown version/generation, wrong base/record/digest and receipt-only selection
reject. Real driver preflight and actual extracted resolver stage reject missing
selection before their downstream paths. A fully hash-valid lifecycle execute
request is refused by the exhausted-budget gate before admission/NativeOps.
No native fixture dispatch, network or helper is used.

The captured targeted result is `GENERATION-VALIDATION-v2-RESULTS.txt`. Historical
`verify_static.py` remains a historical comparison and still FAILS truthfully;
wiring adds explicitly declared source drift, not retroactive green history.
The v1 prospective receipt's pending-wiring statement remains historical. This
v2 validator resolves source-selection wiring only, NOT future freeze/runtime,
financial17/provider/Radient/MCP privacy qualification or any admission authority.
