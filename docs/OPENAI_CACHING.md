# OpenAI cache affinity and efficiency audit

This change adds **`session-id` and `thread-id` affinity headers to the built-in
ChatGPT OAuth Responses route**. The existing request body, account scope,
`store:false`, full-history replay, reasoning policy and error handling remain
unchanged. It also corrects and instruments the existing OAuth cache benchmark.
It does **not** implement the newer public API caching protocol or Codex Lite.

## Sources and boundaries

Audit baseline: Local Operator `e0c27ea94e918b831bc2429ebfb7804a447a502a`.
Codex source: [`459a79eb85400af759e9220c7bafb4429ae07516`](https://github.com/openai/codex/commit/459a79eb85400af759e9220c7bafb4429ae07516).
The relevant immutable files are:

- [`codex-api/src/requests/headers.rs`](https://github.com/openai/codex/blob/459a79eb85400af759e9220c7bafb4429ae07516/codex-rs/codex-api/src/requests/headers.rs#L5-L13): canonical session/thread pair, not model-gated.
- [`codex-api/src/endpoint/responses.rs`](https://github.com/openai/codex/blob/459a79eb85400af759e9220c7bafb4429ae07516/codex-rs/codex-api/src/endpoint/responses.rs#L130-L134): attaching the pair to requests.
- [`core/src/client.rs`](https://github.com/openai/codex/blob/459a79eb85400af759e9220c7bafb4429ae07516/codex-rs/core/src/client.rs#L906-L1031): reasoning context, Lite developer input and request construction.
- [`codex-api/src/common.rs`](https://github.com/openai/codex/blob/459a79eb85400af759e9220c7bafb4429ae07516/codex-rs/codex-api/src/common.rs): request serialization and field omission.

Public documentation was checked September 4–5, 2026:
[prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md),
[reasoning](https://developers.openai.com/api/docs/guides/reasoning.md),
[tool search](https://developers.openai.com/api/docs/guides/tools-tool-search.md),
[WebSockets](https://developers.openai.com/api/docs/guides/websocket-mode.md),
[conversation state](https://developers.openai.com/api/docs/guides/conversation-state.md),
[GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra.md).
These pages are mutable; the Codex source revision above is not.

Public API docs describe GPT-5.6+ explicit breakpoints, implicit end-only caching,
`prompt_cache_options`, a 1,024-visible-token minimum, at least 30-minute lifetime,
1.25× writes and 0.1× reads. **Those docs are not proof that the ChatGPT OAuth
endpoint accepts the same fields.** Live OAuth probes rejected options and
breakpoints, including breakpoints under the Lite header. No public API key was
available for this audit, so public endpoint compatibility remains unvalidated.

## Why this identity and this scope

`SessionStreamFn` already forwards `ChatRequest.prompt_cache_key` from the
persisted cache lineage. A fork intentionally inherits its parent's lineage.
A deterministic UUIDv5 over `local-operator:codex-cache:<lineage>` in the standard
URL namespace supplies both headers. It is stable across retries, restored
sessions, fresh clients and model changes; it does not transmit a raw path or
session label in the headers. UUIDv5 is identity encoding, not encryption or an
authorization mechanism.

This is an **affinity group**, not the unique identity of every fork. That choice
preserves existing fork cache grouping without adding another persisted identity
or plumbing a new session field through the request stack. It also correlates
forks in upstream telemetry. It does not establish ownership of responses,
substitute for the account header, or retrieve stored conversation state.
Native replay guards still bind provider/model/endpoint/account/content.

Headers are omitted without a usable lineage and outside the built-in OpenAI
client + ChatGPT OAuth + exact Codex Responses destination. Public API keys,
custom base URLs and other providers do not receive them. There is no fallback
UUID generated per request and no application-wide constant group. The canonical
header mechanism is model-independent; measured performance claims below are
limited to **`openai/gpt-5.6-sol` and `openai/gpt-6-astra`**.

If a future implementation adds `previous_response_id`, stored conversations or
stateful WebSocket lanes, revisit this identity choice: real conversation and
connection identities must not be confused with shared cache affinity. This
change neither promises nor requires an absence of backend telemetry.

## Bounded live evidence

All inference requests used the real lop request builder and streaming parser,
existing unexpired OAuth credentials loaded read-only without refresh, synthetic
input only, and isolated HOME **and** LOCAL_OPERATOR_CONFIG_DIR. No private
conversations were sent. Exact requested and returned model IDs matched on every
successful request. Failed attempts were retained; none were retried.

The exploratory variants changed fields or headers in a scratch transport
observer, never the installed runtime. Each cohort used independent leading
synthetic content and a stable key. Bodies were identical **within repeated
cohorts**, not across baseline/candidate cohorts: independent namespaces create
small tokenizer-length differences. Cache keys influence routing, not isolation.
The final shipped benchmark isolates the initial tool-schema prefix too.

### Preserve the negative controls

| Batch | POSTs | Actual result |
|---|---:|---|
| Legacy body / public options / explicit marker, both models | 6 | Two completed; options400 and marker400 on each model |
| Legacy, ~6.4k input, changing suffix / append-only / actual tool loop | 18 | All completed, all cached0 / writes0 |
| Lite, same scenarios and scale | 18 | All completed, all cached0 / writes0; ~80 extra input tokens per call |
| Legacy and Lite, ~28k input, three identical requests then append | 16 | All completed, all cached0 / writes0 despite five-second gaps |
| Explicit developer marker with Lite header, both models | 2 | Both400: `prompt_cache_breakpoint is not supported on this model` |

These first 60 POSTs produced 54 completed streams: 684,861 input, 327 output
(including 19 reasoning), zero cached and zero write tokens. Raw SSE usage and
attribution also reported zero; this was not a parser-loss diagnosis. Lite
acceptance demonstrated compatibility of the tested shape, **not an efficiency
benefit**. Early small runs did not enforce moderate cadence; the larger controls
did. No 30-minute expiration experiment was performed.

### Isolate the header effect

An independent four-POST QA control found warm hits with four source-backed
headers. It also omitted the tools supplied by the earlier coder controls, so it
was not sufficient causal evidence on its own. The follow-up used **18 matched,
interleaved POSTs** with the same no-tools shape and five-second gaps:

| Requested / returned model | No extra headers: cached per call | Session + thread only | Four-header bundle |
|---|---|---|---|
| `gpt-5.6-sol` | `[0, 0, 0]`, input6363 each | `[0, 6144, 6144]`, input6360 each | `[0, 0, 6144]`, input6362 each |
| `gpt-6-astra` | `[0, 0, 0]`, input6360 each | `[0, 6144, 6144]`, input6359 each | `[0, 0, 6144]`, input6361 each |

All completed with `OK`. The canonical pair achieved about **96.6% warm-only**
and **64.4% cold-inclusive** cached input versus zero without it. The bundle's
`x-client-request-id` and `x-codex-routing-hint` supplied no incremental benefit,
so they are not added. Cache writes remained zero.

### Validate normal tool behavior

A further **12 matched, interleaved POSTs** used `tool_choice:auto`, genuine
returned tool-call IDs, an executed harmless `synthetic_lookup` returning `blue`,
actual assistant/native payload replay, and a new user turn. All completed without
errors; no shell or filesystem tool was executed.

| Model | Baseline total input / cached | Pair total input / cached | Pair cold-inclusive hit rate | Public-list input-equivalent units: baseline → pair |
|---|---:|---:|---:|---:|
| `gpt-5.6-sol` | 19328 / 0 | 19340 / 12544 | 64.86% | 19328 → 8050.4 |
| `gpt-6-astra` | 19337 / 0 | 19367 / 12544 | 64.77% | 19337 → 8077.4 |

Both pair cohorts reported cached `[0, 6272, 6272]`, around 97% on the warm
continuations. The input-equivalent reduction was approximately 58% across these
three-call cohorts, including the cold first call. It is **not an OAuth bill
reduction claim**, a universal cache-hit guarantee, or a latency SLA. Latency was
recorded but noisy; no general speedup percentage is asserted. The exploratory
header values were stable UUID4 strings; independent post-implementation QA must
exercise the actual deterministic UUIDv5 helper before release.

Through these experiments plus the independent four-call control: **94 POSTs**,
88 completed and six expected validation rejections; 902,174 reported input,
533 output (19 reasoning), 74,240 cached, zero writes. One additional authenticated
catalogue GET confirmed model availability. Historical usage aggregates were
reviewed separately, but heterogeneous historical workloads are not an A/B control.
Per-attempt JSONL, commands and independent QA results belong on the PR rather
than publishing raw provider payloads in this repository.

## Candidate inventory and disposition

“Deferred” means not implemented here, not permission to create a tracker issue.

| Candidate | Baseline/source evidence | Disposition |
|---|---|---|
| Stable body cache key | `clients.py` Responses builder | Already present; keys are routing hints, not machine pins |
| Session/fork lineage | `configure.py` `_cache_lineage_id` | Already present; reused for header affinity |
| Canonical session/thread headers | Pinned Codex headers + matched probes above | **Implemented**, transport-only |
| Frozen initial system prefix and append-only state updates | `session.py` `_prepare_system_blocks` | Already present; authoritative instruction changes deliberately invalidate cache |
| Native opaque history replay | `providers/replay.py` scope/fingerprint guards | Already present; preserve exact items and permissions |
| Ordered native output and encrypted reasoning include | `clients.py` Responses stream/builders | Already present; unconditional include without explicit effort remains a separate conditional follow-up |
| Stable builtin tool ordering | `tools/registry.py` and Responses serializer | Already present; no new sorting convention |
| Reused HTTP connections | `OpenAICompatClient` owned AsyncClient | Already present; no second pool |
| Raw read/write parsing and displayed cache rate | `clients.py`, `analytics/model.py` | Already correct; no analytics UI change |
| Disjoint cache pricing buckets | `configure.py` `cost_for_usage`; Sol writes already $5/M | Already correct; do not infer writes from misses or guess subscription billing |
| Benchmark denominator and fidelity | Old `bench_openai_oauth_cache.py` counted cached twice and replayed canned replies | **Fixed**: cached/input, actual native replies, safe bounded execution, raw usage and explicit failures |
| Public modern breakpoints / implicit+explicit / TTL | Current public caching docs | Deferred: no API-key live coverage; current legacy public retention compatibility gap remains |
| GPT6 public Responses family policy | Baseline `_OPENAI_RESPONSES_API` matches GPT5, not GPT6 | Deferred public-path follow-up; OAuth already routes Responses independently |
| Codex Lite developer AdditionalTools/instructions | Pinned client.rs906–975 | Accepted experimentally, **not shipped**: no demonstrated benefit and reasoning semantics differ |
| All-turns native reasoning accounting | `providers/context.py` counts only after latest user | Required if Lite/all_turns is adopted later; do not change legacy current-turn accounting here |
| Discovering Lite capability flags | Pinned Codex model catalogue | Deferred with any future Lite rollout; avoid model-name guessing |
| Dynamic additional_tools / deferred tool search | Public tool-search guide; current full inventory refresh | Deferred structural feature: durable placement, removals, revocation, names, resume/fork and dispatch must be designed together |
| Keep tool schemas with restricted tool_choice/allowed_tools | Public caching tools guidance | Conditional follow-up; never keep forbidden tools callable for cache statistics |
| GPT6 configuration_update effort changes | Public reasoning guide | Deferred: requires persisted update placement, compaction compatibility and user-effort correctness |
| WebSocket delta transport / previous_response_id | Pinned Codex client.rs1368–1449 and public WS guide | Deferred transport project: concurrency, cancellation, reconnect, auth rotation and full-context recovery |
| HTTP stored responses/conversations | Public conversation-state guide | Deferred persistence change; fewer upload bytes do not exempt input tokens from billing |
| WS preconnect and generate:false prewarm | Pinned Codex client.rs1440–1449 | Deferred with WS; meter speculative warmup separately |
| Extra routing/request headers and server turn-state echo | Pinned Codex transport; four-header probe | Not added: no demonstrated incremental benefit; server turn-state requires separate ownership |
| Zstd request compression | Pinned responses endpoint174–177 | Deferred until an upload bottleneck is measured; affects bytes, not cache tokens |
| Cross-model-family reasoning replay | Public Sol/Terra/Luna docs vs exact-model replay guard | Deferred; no widening across accounts/endpoints or Astra by name similarity |
| Compaction and tool-result pruning | Existing context management | Preserve correctness-first behavior; lower total input can beat a higher cache rate |
| Globally force24h or add Anthropic cache_control | Different public/Codex/compatible contracts | Rejected as a blanket change |
| Universal or per-request-random cache key | Routing locality and load limits | Rejected as a default; preserve lineage grouping |
| Artificial prompt padding | Public minimum length and write economics | Rejected for ordinary agent turns; synthetic benchmark size controls are not product padding |
| Lower effort/output quality or remove validation to win a benchmark | Changes new work and task quality | Rejected as a cache optimization; pin effort within comparisons |
| Local answer memoization | Not provider KV caching | Rejected; would change freshness and tool side effects |

## Re-running the benchmark safely

Use the existing script on the checkout under test. Live calls are opt-in, targets
are the two audited exact model IDs, each scenario is capped at 1–8 POSTs, and
failed/incomplete/unknown-usage responses stop the run with a nonzero exit status.
No automatic auth refresh, provider retries or alternative model substitution is
performed. An active turn has a 90-second wall deadline. Codex omits output caps,
so the requested terse answer is not a provider-enforced token ceiling.

```sh
.venv/bin/python scripts/bench_openai_oauth_cache.py --dry-run
.venv/bin/python scripts/bench_openai_oauth_cache.py --live \
  --model gpt-6-astra --scenario tool_heavy --turns 3 \
  --output /tmp/openai-cache-run.jsonl
```

`--auth-db` chooses a read-only OAuth source; otherwise the script resolves the
existing config path before isolating itself. The token must already be unexpired.
Both HOME and config are redirected temporarily and restored, and existing output
files are never overwritten. The real tool inventory is measured, but only the
empty-argument synthetic lookup can execute. All other model tool requests fail
the benchmark. Default tool choice is production's `auto`, not an artificial
`none` restriction.

`--seed` defaults to a fresh value; reuse it only for deliberate warm-cache controls.
`--prefix-rows` adds bounded synthetic reference content when testing prompt size.
`repeat_then_append` repeats the first request up to three times before appending
an actual reply and another user turn. Long-session and tool scenarios preserve
actual native assistant payloads. The raw JSONL includes adapter/script hashes,
source revision, request-body hash/shape, affinity headers, model/auth/endpoint,
raw and normalized usage, text TTFT when emitted, total latency, actual synthetic
outputs and errors. It does not expose auth/account headers or encrypted history.

For OpenAI, cached and written tokens are **inside** input. Report weighted rates,
not an average of per-call percentages:

```
hit_rate = sum(cached_tokens) / sum(input_tokens)
public_list_input_equivalent = (input - cached - writes) + 0.1*cached + 1.25*writes
```

The equivalent units use documented modern public list-rate multipliers; they
are neither dollars nor the OAuth subscription's measured charge. Keep raw
writes even when they are zero, include cold writes in comparisons, and report
warm-only rates separately. A high cache percentage with a larger prompt is not
necessarily cheaper. Regression fixtures and green tests are not substitutes for
independent live QA on the installed/worktree runtime and real tool loops.
