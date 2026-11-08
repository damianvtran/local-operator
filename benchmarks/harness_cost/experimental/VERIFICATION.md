# Experimental port verification — 2026-11-03

## Actually performed on the assembled public copy

No model API calls, native harness runs, Docker operations, dependency installs,
global runtime updates, commits or PR actions were performed by this port lane.
During round-2 remediation, one delegated worker made a public documentation
search about Darwin zombie-group EPERM, contrary to the requested no-network
constraint. This was a workflow deviation, not model/provider qualification;
no benchmark payload download, paid provider request or installation occurred.
Only the experimental folder was edited by this lane and its delegated workers.

From this directory with an existing dependency-bearing Python:

```sh
PYTHONDONTWRITEBYTECODE=1 "$BENCH_PYTHON" -m unittest -v \
  test_cache_namespace test_cache_forwarder test_adapters test_runner_v2 test_bridge_sse
(cd coding && python3 -B -m unittest -v test_guard test_driver_safety)
# Retained core tests, read-only check from the parent directory:
(cd .. && python3 -B -m unittest -q test_kit)
python3 -B verify_cache_evidence.py
python3 -B freeze_inputs.py
```

Round-2 remediation results: **42 tests passed** in the first command (0.600s),
**16 passed** in the coding suite (3.479s), and **16 retained core tests passed**
(1.033s): **74 passed total**. The prior 62-test baseline is retained, with 12
new regression methods. Core files were not edited. Cache forwarder/meter tests use in-memory ASGI/mock
upstreams, no listening sockets or provider calls. Runner tests exercise the
actual configuration and filesystem-copy path with synthetic skill text, not
native CLI/model behavior. Coding tests include stdlib budget/freeze/disk/CLI-refusal controls and a real
owned Python process-group lifecycle matrix (live/exited leaders, ordinary/TERM-
resistant children, exact-group cleanup in finally, unowned-group refusal).
They do not launch Harbor or native benchmark runtimes. All **28 Python modules** compiled in
memory and have module docstrings. CLI help was exercised without dispatch.

Minimal evidence verification reproduced both A/A/B/A read-token arrays. An
in-memory changed post-transformation hash was rejected after a review finding
prompted explicit request/after and namespace-lineage consistency checks. This
checks supplied metadata relationships, not withheld body hashes/provider truth.

The optional `test_cache_bridge` was attempted and **could not import LiteLLM**
(`ModuleNotFoundError`); it did not execute its transformation test. No dependency
was installed to turn that into a green result. Public translator/native Codex
integration remains unverified. The separate bridge mock tests above do pass.

The task-fetch worker exercised offline public-definition planning from a clean
pinned local OSWorld checkout, full-pin/output guards, unsafe path/URL rejection,
tamper refusal, environment precedence and no-network planning. Planned assets
had no downloaded bytes/hashes, and evaluator preflight correctly rejected them.
Selected metric import also stopped at missing `PyPDF2`; the existing task
interpreter additionally lacked `rapidfuzz`/`formulas`. No fetched fixtures,
planning outputs or task payloads are included, and no oracle pass is inferred.

The runner was copied **only after** the parent relayed its owner's final handoff,
and SHA256 `3f36b8aba2454bd496b42726790863dd8ec2cdfa2d35342781d32ef4992b9e22`
matched the original bytes. The public portability delta was then retested.
Original-source tests/review are not passed off as public-port live validation.

## Read-only review and privacy/attribution checks

A separate reviewer examined stable cache/protocol/evidence files, then the
adapted runner's portability/guard sites and neighboring code. No blocker/major
was reported. The one minor hash-consistency issue was remediated and checked.
The runner review was bounded static/offline review, not a byte-proven full
original-to-port review or a paid/native QA round. Coordinator review remains
separate and owns any future PR review/commit gate.

Scoped scans checked personal home paths, private/company MCP names, token-shaped
literal credentials, copied runtime logs/bodies/archives and missing module
docstrings. No prohibited payloads were found. Code catalogs retain Codex's
prompt bytes and its Apache-2.0 license/NOTICE; the OSWorld license is included.
`SOURCE-AUDIT.json` records source and destination digests. Markdown local links
and public URL targets were inspected statically; **remote links were not fetched**
and current remote reachability is not claimed. References to future external
freeze/attestation records are explicitly requirements, not bundled evidence.

## Round-2 findings and bounded remediation

Review source: https://github.com/damianvtran/local-operator/pull/3345#issuecomment-6599583021
at reviewed head `7c9ffa97ca21d5123cfb28fbcd8bb5475b57f83d`.

- **R2-F1 reproduced first:** the two new direct billing regressions failed:
  missing Messages usage incorrectly yielded `cost_complete=true`, and a billed
  HTTP 500 was dropped (`0.2 != 0.5`). After remediation, the actual direct outputs
  for USD 0.20 plus an unknown attempt are `measured_model_cost_usd=0.2`,
  `model_cost_usd=null`, `total_cost_usd=null`, `unknown_billing_attempts=1`,
  `cost_complete=false`. USD 0.20 plus a billed HTTP 500 of USD 0.30 yields
  `total_cost_usd=0.5`, `non_200_message_attempts=1`, `cost_complete=true`.
  Discovery/count-token rows remain distinct from recorded Messages attempts.
- **R2-F2 reproduced first:** both actual ASGI routes leaked search events under
  CRLF/CR (eight failing subcases; LF controls passed). Both now use one incremental
  `sse_stream.py` helper; three new ASGI regressions pass with whole/byte-at-a-time
  streams, LF/CRLF/CR/mixed framing, multiline JSON, Unicode splits, search event
  suppression/rewrites and byte-preserved usage/errors/DONE/EOF tails. No sockets
  or model calls are involved (ASGITransport plus upstream MockTransport).
- **R2-F3 reproduced first:** a real owned TERM-resistant child survived its
  leader and old cleanup. Cleanup now requires locally registered fresh-session
  ownership and sweeps that exact group even after leader exit, TERM then bounded
  wait then KILL; repeat/unowned/unsafe cleanup cannot signal other groups. Tests
  establish no live fixture descendant remains; only direct children can be
  waited/reaped by this process, while orphan zombies belong to their adopter.
- **R2-F4 reproduced first:** matching legacy JSON admitted `omp` with only
  `not-a-runtime.txt` and no captures. `preflight`, `run_trial` and `lane_services`
  now unconditionally raise `HARDBLOCK PAID CODING` before artifact/auth reads or
  dispatch effects. Real CLI refusal tests cover all seven selections even with
  self-asserted JSON. Actual executable/capture admission remains unimplemented;
  hashes/booleans cannot lift the hold. Paid LOP classifier hold remains too.

All four fixes are one bounded experimental-only remediation batch. The next
coordinator review should scope to billing, shared framing, owned lifecycle and
hard-hold deltas plus their direct regression surfaces—not unchanged analysis.
No reviewed fix or green test is claimed to complete pending native experiments.

## Follow-up: actual ASGI lifecycle compatibility

A development integration probe found that Starlette 1.10.0 rejects the legacy
`on_shutdown` constructor argument. The coordinator reproduced the public
coding guard's startup failure with the installed optional dependencies:
`TypeError: Starlette.__init__() got an unexpected keyword argument 'on_shutdown'`.

The guard now uses an async lifespan context manager. Two real in-process
`TestClient` tests construct and start the actual app, exercise HTTP 200 health,
HTTP 403 unpriced-route rejection and HTTP 400 invalid-input rejection, then
verify that shutdown releases the ledger lock and a second app can acquire it.
No upstream request, socket, credential or model charge is involved. The same
command runs the prior reservation and owned-process controls: **18 coding
tests pass** (16 prior plus 2 lifecycle tests). Combined with the unchanged 42
experimental and 16 core tests, the local total is **76 passing tests**.

This is source compatibility/lifecycle evidence, not a coding pilot or permission
to lift either paid-dispatch hold. Verified optional dependency versions were
Starlette **1.10.0** and httpx **0.28.1**; no new environment was installed.

## CI lint/type and optional dependency follow-up

The first hosted whole-tree checks exposed formatting/type errors in the newly
added standalone scripts and absent optional scientific imports in CI's core
`[dev]` environment. Source was formatted and typed rather than excluded or
suppressed. The optional Harbor/LiteLLM boundaries validate required attributes,
callability and real runtime bases; their new structural tests do **not** prove
installed plugin signatures or native integration. Missing DSH entry metadata
now fails closed before writing a configuration. Model-facing strings, cache
counter arrays, paid holds and historical numerical outputs remain unchanged.

The benchmark offline suite is now **84 tests**: 16 core, 46 experimental and
22 coding tests. The coordinator also ran **213 CI workflow/scoping contract
tests** with this checkout's own Python environment. A foreign editable test
environment was rejected by the repository guard before collection, and was not
used as passing evidence.

CI installs this kit's pinned optional analysis requirements only in its
analysis/type-check job; no application runtime dependency was added. The
`python-dateutil` pin is **2.9.0.post0**, compatible with core botocore's `<3`
constraint. The coordinator assembled one worktree-owned Python 3.14 environment
from the shared offline uv cache with Matplotlib 3.10.7, NumPy 2.3.4 and SciPy
1.16.3; `uv pip check` confirms all 113 installed packages are compatible.
An earlier Python 3.12 offline-only attempt lacked the exact cached Matplotlib
wheel and did not install it. Hosted Linux Python 3.12 validation remains a
separate CI result, not inferred from this local environment.

The coordinator executed the exact new workflow shell step locally in the
compatible pinned Python 3.14 environment: publication verification passed,
16 core + 46 experimental + 22 coding tests passed, and historical numerical
regeneration passed. Targeted whole-kit Pyright reported **0 errors and 0
warnings**; flake8 was clean. The first isolated attempt exposed that fixture
tests need an explicit scratch directory; the workflow allowlist now supplies
one, and the ordinary-terminal instructions document the same setup.

The workflow's offline step clears environment credentials, uses an isolated
HOME/config and temporary output directory, checks publication integrity, runs
the offline suite and regenerates analysis without model/harness/Docker/login
operations. Checkout credentials are not persisted in this read-only job.
Real LiteLLM transformation and Harbor native integration remain unverified;
no optional integration failure was converted to a passing skip.

## Hosted Linux profile-copy follow-up

Hosted Python 3.12 installed the compatible analysis packages and passed the
whole-tree type check, then the new offline step exposed a genuine platform
precondition: its filesystem rejects `cp --reflink=always` with `Operation not
supported`. The tiny public-profile preparation fixture consequently failed.
It was not skipped or moved to a privileged filesystem.

CoW remains preferred. Linux now has an explicit, bounded fallback only for
small dependency-free profile data: at most 8 MiB and 4,096 entries. Same-device
checks remain; symlinks, special files, dependency/runtime/cache directories,
oversized trees and pre-existing destinations are refused. Descriptor-anchored
traversal, actual read/write limits and inode-owned partial cleanup prevent an
unchecked full-home copy. Large/runtime profiles still need CoW.

The original failure was reproduced before editing. Actual local APFS cloning
and the real prepare/config path with a simulated unsupported Linux clone both
pass. Fourteen new safety/preparation regressions bring the offline benchmark
suite to **98 tests** (16 core, 60 experimental, 22 coding). All other runner
sections, prompts and paid holds are AST-identical. Actual Linux fallback
execution remains a separate hosted-CI result, not inferred from the local
simulated failure test.

## Bounded-copy review remediation

The focused copy review found two real cases the 98-test suite had missed:
`os.fwalk` ignored permission errors and omitted a source subtree, and replacing
the destination with a symlink could redirect pathname-based writes/mode changes.
Both were independently reproduced before the unified fix.

Both source traversals now raise on errors. Every destination content, directory,
metadata and cleanup operation uses owned descriptors/no-follow semantics;
success also validates the destination binding. Native CoW uses an isolated
stdlib `fchdir`/`exec` carrier rather than a threaded-host `preexec_fn` or a
replaceable destination pathname. Cleanup identifies the owned inode, not a
replacement symlink. Real disposable-I/O controls verify permission-error aborts
in both passes, no outside file creation or mode change during replacement, and
owned-partial cleanup. A native-CoW carrier replacement control also passes.

The local offline suite is now **101 tests** (16 core, 63 experimental, 22 coding).
All other runner sections are AST-identical. These are source-safety/portability
checks, not model calls or native-harness qualification. Actual hosted Linux
execution remains a separate CI result.

## Remaining gates — not ready

- Frozen/amended preregistration and feasible source selection; original 60-task
  OSWorld quota infeasible, 83 metadata candidates not all oracle-feasible.
- Download/hash public assets and independently qualify actual oracle/checker
  behavior. Private 21/30 Terminal-Bench and six Linux runtime reports are not
  re-created by this distribution. Hermes remains incomplete; no paid pilots.
- Native seven-harness public-loadout exposure, Linux reflink/runtime behavior,
  translator streaming/deferred tool wire, provider cache/usage/rate bounds.
- Hosted classifier billing, conservative reservation and cache attribution.
  Paid LOP remains explicitly blocked, not silently run without Radient.
- Independent protocol/runtime/design review as applicable, complete freeze and
  actual side-effect/negative-control evidence before any paid collection.

Draft SHA256 agreement is an input-integrity check only. It cannot authorize a
paid run, make a missing gate pass, or establish quality parity/performance wins.
