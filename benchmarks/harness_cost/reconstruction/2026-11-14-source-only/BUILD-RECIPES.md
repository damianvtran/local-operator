# Source-only build recipes — DRAFT, execution blocked

These recipes describe a **new** qualification environment. They cannot reproduce
lost runtime bytes. No recipe here has been built or installed. A complete,
platform-selected runtime lock is still a prerequisite for dispatch.

## Host-services recipe

Source: `experimental/{meter_proxy,lop_edge,package_egress}.py`,
`experimental/coding/codex_bridge.py`, and `experimental/radient_meter/`.
`historical-bindings.json` verifies all 37 retained prereg source/closure entries.
Do not substitute the analysis-only `harness_cost/requirements.txt`.

Proposed dependency closure: `host-services.requirements.lock`, derived without
execution/resolution from the exact Hermes upstream `uv.lock` retained under
`metadata/`. Its 13 package records (including all dependency edges and wheel
hashes) are retained in `host-services.lock.json`. This is a conservative base
closure for httpx, starlette, uvicorn and cryptography, **not a recovered host
lock**. Platform-wheel applicability, compatibility, and build backend/interpreter
identity remain unverified. It does not include Harbor or the LoP/Hermes runtimes.
The separately collected `*-proposed` PyPI metadata is informational, NOT the
selected service lock (the Hermes-derived closure is authoritative for this draft).

Future separately authorized build sequence (NOT executed):

1. Select and checksum-pin a CPython 3.14 build, installer and target OS/architecture;
   fail closed if either is unspecified. Source lock supports >=3.14, not arbitrary
   >=3.11 despite the older minimum in the public runbook.
2. Create ONE shared service environment in the parent-coordinated durable root.
   Install the exact hash lock with `--require-hashes --only-binary=:all: --no-deps`.
   Missing wheels are a blocker, not permission to compile/resolve silently.
3. Record interpreter executable SHA256, installed distribution RECORD hashes,
   source commit, dependency lock SHA256, installer version and OS identity.
4. Start each service in a recorded process group; meter entry point is
   `python experimental/meter_proxy.py` (loopback, BENCH_PORT); bridge is
   `python experimental/coding/codex_bridge.py <owned-port>`; edge is
   `python experimental/lop_edge.py` with JSON on stdin, never a bearer on argv.
   Only dummy tokens and loopback upstream fixtures in initial qualification.
   The parent owns fixture setup and durable output locations before dispatch.
5. Verify meter `/health` caps, malformed/unknown-run/budget-exhaustion refusal,
   stream settlement and cancellation accounting. Edge must refuse wrong-run,
   disallowed host and missing CA/allowlist. Confirm no upstream external traffic.
   Reap only recorded process groups and verify their PIDs no longer exist.

Service startup requires further configuration and test fixture implementation;
these entry points alone are not a complete runnable host-services deployment.

## Linux/amd64 toolchain recipe

Target layout is exactly the existing `experimental/coding/adapters.py` paths,
read-only `/opt/bench` shared by all tasks. Do not invent a second runner.
`distribution-pins.partial.json` contains public npm SRI/PyPI artifact hashes,
not a complete install lock. Fetching package metadata did not download binaries.

| Harness | Historical pin | Obtainable now | Remaining build lock gap |
|---|---|---|---|
| claude-code | 2.1.274 | npm wrapper and Linux-x64 package SRI | original installed bytes/postinstall attestation |
| codex-best | 0.147.0 | npm wrapper and Linux-x64 package SRI | original installed bytes/runtime attestation |
| omp | 18.11.2 | npm artifact SRI | transitive/optional dependency closure; original compiled binary |
| dsh | 0.2.0-rc.2 | npm artifact SRI | transitive dependency closure; original installed tree |
| local-operator | 500883ef56ca504325ffd80c5b90bf2435d00e3f, v0.94.22 | exact public pyproject bytes | upstream uv.lock 404; runtime and build dependency closures |
| hermes | f0d94882e89ab8d4098b5bb2f166adfb9ee0a0ab, v0.22.6 | exact pyproject and uv.lock | target extras/platform selection, Python/build tools, runtime attestation |
| opencode | 2.0.27 | no exact release artifact found | v-prefixed/unprefixed GitHub release + npm version returned 404; source revision unknown |

Future separately authorized build sequence (NOT executed):

1. Pin Linux/amd64 base rootfs/container digest and archive hash; Python, Node,
   package manager, libc/toolchain and compiler hashes. These are **UNPINNED**.
   Do not use moving `latest`, mutable apt repositories or curl-piped installers.
2. Resolve/lock missing dependency closures in an approved build lane. Keep
   historical version identity, but new lock bytes get new qualification identity.
   Freeze all OS packages and postinstall downloads. Do not silently replace the
   missing OpenCode artifact with another version or Go implementation.
3. Obtain the pinned public source/distribution artifacts using the metadata
   hashes, verify them *before* extraction, disable lifecycle hooks until reviewed.
   Build LoP at its exact revision and Hermes using its exact source/lock. Treat
   setuptools/wheel/build-isolation downloads as separate locked dependencies.
4. Install into the existing `/opt/bench` layout; compute installed-file manifest,
   direct_url/RECORD checks and actual binary versions. Hermes version must derive
   from verified source/tag evidence as existing driver requires, not `vunknown`.
5. Hash the resulting whole toolchain and builder inputs. Run source/runtime/wire
   qualification for all seven arms, with malformed configuration, wrong source,
   wrong binary, wrong model and egress controls. Any missing row blocks paid use.

This recipe is intentionally **not buildable yet**: unresolved inputs must not
be replaced with guesses. A Dockerfile with moving defaults would falsely imply
reproducibility; this document and partial locks expose the exact blockers instead.

## Task data and unresolved historical artifacts

`task-matrix.json` is authoritative per-file status at public source commit
7131e4375048a0e408a8fb404b5f499d726b695b. This is an immutable recovery
candidate, not proof of the original dataset checkout revision (that revision is
missing). Per-file prereg hashes, not a moving upstream name, establish exact
recovered byte identity. All downloaded task bytes must match
prereg SHA256 before retention. 380/407 match; 26 binaries/media/archives were
not collected under this source-only authorization, and one source file differs.

* `sanitize-git-repo/tests/test_outputs.py`: public hash differs from prereg;
  not retained as recovered. Narrow upstream history query found only the task
  directory-move commit; no broad history hunt or local patch reconstruction.
* Non-source inputs: exact file names/expected hashes in `task-matrix.json`.
  These are not declared irretrievable: they are uncollected. No container image,
  image/media fixture, encrypted archive, model weights or Git bundle downloaded.
* Frozen 30-task subset bytes: missing; surviving 21-row file hash differs.
  `derived-subset.json` reconstructs selection, labels missing metadata; seven
  image digests remain missing. DEV payload hashes are not bound in this prereg.
* Frozen builder, runtime, oracle, toolwire, catalog/prereg-tools and original
  S4/pre-inference receipts: only historical digests/claims survive, not bytes.
* Original seven trial ledger/outcomes, streams, cost receipts and price-card
  response bytes: lost. Neither old outcomes nor spend can be recomputed from
  v1 snapshots or caches. Reconcile actual prior spend before any new paid run.
* Historical host dependency lock, installed toolchain tree, OS/interpreter/Node
  and build-tool identities: missing. Exact public package availability is not
  byte-equivalence to the deleted installation.

Task source license: upstream Apache-2.0 text retained in `metadata/task-license-*`;
original notices in each task remain intact. Original whitespace/CRLF and EOF
bytes are deliberately preserved: whole-diff whitespace checks report upstream
violations. The authored-file whitespace check excludes byte-preserved payloads
and upstream metadata rather than normalizing them and breaking prereg hashes.
Source payloads are public benchmark
material, not operator credentials. Do not run or import recovered task code during
this phase, and do not expose hidden fixtures to inference during qualification.
