# Experimental port verification — 2026-11-03

## Actually performed on the assembled public copy

No model API calls, native harness runs, Docker operations, network downloads,
dependency installs, global runtime updates, commits or PR actions were performed.
Only the experimental folder was edited by this lane and its delegated workers.

From this directory with an existing dependency-bearing Python:

```sh
PYTHONDONTWRITEBYTECODE=1 "$BENCH_PYTHON" -m unittest -v \
  test_cache_namespace test_cache_forwarder test_adapters test_runner_v2
(cd coding && python3 -B -m unittest -v test_guard)
python3 -B verify_cache_evidence.py
python3 -B freeze_inputs.py
```

Results: **35 tests passed** in the first command, **11 passed** in the coding
suite: **46 passed total**. Cache forwarder/meter tests use in-memory ASGI/mock
upstreams, no listening sockets or provider calls. Runner tests exercise the
actual configuration and filesystem-copy path with synthetic skill text, not
native CLI/model behavior. Coding tests are stdlib budget/freeze/disk/CLI-refusal
controls, not Harbor/native integration. All **25 Python modules** compiled in
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
