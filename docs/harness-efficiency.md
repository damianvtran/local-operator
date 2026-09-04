# Harness efficiency: changes and evidence

This release addresses the September 2026 harness audit, originally measured
against `5cbea141`. The final measurements below compare committed baseline
`765c619b` with candidate `1a7bd26b`, after integrating concurrent upstream work.
Both checkouts own an editable virtual environment; probes verify imported
source paths. Local timings are observations on macOS, not promises about
provider latency or fleet completion time. Earlier artifacts remain retained.

## Measured improvements

| Workload | Baseline | Changed | Scope |
| --- | ---: | ---: | --- |
| Short foreground bash, median of 12 | 275.62 ms | 36.96 ms | Real subprocess, cancellation enabled |
| Capture 32 MiB stdout, peak Python allocations | 140.05 MiB | 28.14 MiB | Same head and final diagnostic retained |
| Capture 1 MiB stdout | 4.04 MiB | 4.09 MiB | Small-output memory effectively unchanged |
| Fetch 12 distinct local pages | 12 TCP connections | 1 | All response bodies verified |
| Durable 48-message paired flush, median of 7 | 3.52 ms | 1.21 ms | fsync count 48 → 1; loop-thread fsync 48 → 0 |
| 100 prompt builds over 5,000 rows, median thread CPU of 7 | 72.63 ms | 0.254 ms | Indexed state and unchanged-block reuse |
| Matching Anthropic wire prefix after a goal change | 11.46% | 99.984% | Structural cache eligibility, not a cache hit rate |
| Prescribed three-read pipeline, model requests | 4 | 2 | Scripted plan; both execute three reads and return 6 |
| Same pipeline, retained tool-result bytes | 150,078 | 55 | Intermediate data stays in Python |

Tool/network JSON: [before](benchmarks/harness-efficiency/final-tool-before.json),
[after](benchmarks/harness-efficiency/final-tool-after.json).
Reproduce using `scripts/bench_tool_io.py --repo <checkout>` with each checkout's
own interpreter. Final session JSON: [before](benchmarks/harness-efficiency/final-session-before.json),
[after](benchmarks/harness-efficiency/final-session-after.json). Reproduction
commands and persistence contracts are in
[the session validation report](harness-session-efficiency.md).

Composition JSON: [before](benchmarks/harness-efficiency/final-composition-before.json),
[after](benchmarks/harness-efficiency/final-composition-after.json). Reproduce with
`scripts/bench_tool_composition.py --repo <checkout> --output <json>`, again using
that checkout's interpreter. This compares explicitly selected valid plans,
not model sampling: it proves the shorter execution path works and preserves
the answer, not that a live model always discovers or chooses it.

## Actual provider cache reuse

One three-request synthetic sequence per arm used native Anthropic Sonnet 4.6,
low effort, an output cap of 1,024 tokens, and roughly 43,000 input tokens.
Each sequence seeded a conversation, repeated a request with an unchanged goal,
then changed the goal. All six requests completed with five output tokens each.

| Phase | Baseline cache reuse | Candidate cache reuse |
| --- | ---: | ---: |
| Unchanged-goal control | 99.9631% | 99.9632% |
| Changed goal | 18.2914% | 99.8575% |

After the goal change, new cache writes fell from **35,434 tokens to 59**.
The denominator is Anthropic's disjoint input + cache-read + cache-write buckets,
which matches its reported context count in every call. Credentials stayed
stable within each arm. The arms used different credential scopes, UUIDs and
temporary directories, so the unchanged control is the within-arm comparison;
this is not a claim of identical cross-arm wire bytes. The native home catalogue
cache was shared and outside the measured provider calls.

This verifies the goal-change caching mechanism, not general task latency,
quality, or a fleet-wide cache rate. Standing instruction changes deliberately
start a new persisted prefix, since current repository and user instructions
take priority over reuse.

Data: [baseline](benchmarks/harness-efficiency/cache-live-before.json),
[candidate](benchmarks/harness-efficiency/cache-live-after.json). Reproduce with
`scripts/bench_live_cache.py --repo <checkout> --arm <label> --output <json>`
using that checkout's interpreter from outside the repository. This command
makes three live requests using the native login and therefore incurs usage.

## Accepted live tasks

Twenty fixed-source native OpenAI sessions passed independent artifact checks:
five repeats of two tasks on each arm. Total elapsed time was 381.17 seconds
before and 379.78 seconds after; model requests were 52 and 51. This does not
establish a general task-time improvement. Repair's mean decreased 2.4%, while
aggregation's increased 1.9%. OpenAI cache results were mixed and are reported
without a universal cache-rate claim.

Continuation input-estimate median absolute error fell from 455 to 29 tokens
for repair and from 568 to 28 for aggregation. The full paired measurements,
including slower runs and cache-state confounds, are retained in
[the live results](benchmarks/harness-efficiency/live-results.md) and
[sanitized run data](benchmarks/harness-efficiency/live-final.json).
The [method](benchmarks/harness-efficiency/live-method.md) documents acceptance
checks, source verification, isolation and limits of the sample.

## Audit disposition

| # | Finding | Implemented change or evidence-based decision |
| --- | --- | --- |
| 1 | Shared parent/child provider state | Fork conversation handles; share only refcounted transports. Child initialization failures roll back ownership. |
| 2 | OpenAI reasoning lost between tools | Retain ordered native output and encrypted reasoning with provenance checks; normalize on incompatible history/provider changes. |
| 3 | Google signatures and result images lost | Replay native signed parts, pair actual call IDs with results, and include image bytes. |
| 4 | Stale context estimate | Reconcile provider input counts with the counted boundary plus new material; invalidate after prefix/history/model changes. Oversized reconciled counts remain admission failures. |
| 5 | Changing state rewrites historical prefix | Persist initial system blocks; append durable typed state updates at the conversation tail before building that request. |
| 6 | MCP discovery steps and schema churn | Search operation/object terms across services, return bounded full schemas, and register only selected tools for deferred execution. |
| 7 | Shell polling delay | Wait for process exit, unfinished readers or cancellation with FIRST_COMPLETED. |
| 8 | Blocking durable transcript I/O | Ordered off-loop batch commits, with cancellation settlement and one fsync per closed batch. |
| 9 | Unbounded subprocess capture | Bounded head/tail collection, streaming redaction, honest incomplete spill metadata and off-loop completion formatting. |
| 10 | Repeated prompt/history scans | Indexed durable entry IDs/latest user/compaction state and unchanged prompt reuse. Context preparation is off-loop. |
| 11 | Slow presentation subscribers | Production TUI/SSE/headless handlers already enqueue cheaply. Added opt-in bounded ordered presentation subscriptions for async SDK consumers; no existing production speedup claimed. |
| 12 | Invisible eval eviction | Explicit reset receipt before executing stale-namespace code, generation metadata and idle retention that accounts for busy kernels. |
| 13 | Coarse tool serialization/unbounded fan-out | Independent canonical-path/inode writes overlap; conflicts and legacy read/write barriers stay ordered. At most eight workers refill immediately as calls finish. This is not a global cross-session quota. |
| 14 | Stale or missing task knowledge | Refresh selected knowledge at new user boundaries; children inherit repository guidance and bounded selected knowledge. |
| 15 | Unnecessary post-compaction turns | Continue only for unfinished/interrupted output; a completed answer remains completed. |
| 16 | Generate then truncate summaries | Set summary-specific output budget and effort before generation; preserve explicit short cache policy on standalone helpers. |
| 17 | Repeated web setup/duplicate reads | Session-owned HTTP pooling and cancellation-aware in-flight coalescing; retain per-hop DNS/SSRF checks, origin/SNI identity and credential/settings isolation. |
| 18 | Identical error loops | One recovery notice after three identical all-error batches; stop after six. Successful polling, changed errors and new steering reset the guard. |
| 19 | Misleading metrics | Deduplicate benchmark end events; price provider cache conventions; retain real wire schemas in prefix probes. Record missing-usage failures and request timing/identity/purpose. Unknown spend remains unknown. |
| 20 | Excess model round trips and permanent schemas | Eval's request-owned `tool(name, **args)` composes discovered calls and projects intermediate data locally. It reuses schema validation, approvals, role resolution, events and cancellation. |
| 21 | Aggressive effort/compaction/TTL defaults | Retain current defaults: no controlled quality evidence justifies changing them. Summary budgets and repaired measurement are separate concrete improvements. |

## Execution and measurement boundaries

Discovered tools do not inflate the advertised schema array. `read` of
`mcp://?search=operation+object&limit=4` returns bounded reference schemas;
`eval` can call the resulting tools, inspect `is_error`, and print only the
needed projection. Nested calls execute sequentially inside eval's exclusive
slot. Recursive eval, background cells and Python background threads cannot
use the bridge. Only the foreground execution thread may read its protocol.
Oversized
responses preserve the completed operation's outcome in a truncated receipt;
they never turn a successful mutation into a retryable failure.

Web pooling does not add a sequential search-result TTL, negative enrichment
cache or shared cookie state. Fetch refresh still bypasses the response cache.
Tool output remains bounded; omitted bytes are explicitly marked rather than
advertised as recoverable full output.

Analytics latency is observed end-to-end stream duration, including routing,
retries and consumer backpressure. TTFT is the first text/tool delta. Preparation
time covers request preparation. Rows identify logical requests, owning and
parent sessions, helper purpose, outcome and whether usage was reported. These
are not provider-internal compute times or separate per-retry attempt records.
Historical unknown timing fields remain unknown. Actual accepted task results
must accompany any future tuning of effort, compaction or cache TTL.

## Validation strategy

Regression tests assert overlap/order, thread identity, bounded retained bytes,
durability before publication, request ownership and exact provider wire data.
Timing ceilings are not calibrated from the development machine. The assembled
loop tests use a real eval subprocess, exercise discovered tools and denied or
invalid calls, and check paired execution events. Unit tests cannot establish
model task quality or actual provider cache hit rates; live trials are reported
separately with their sample size and acceptance criteria.
