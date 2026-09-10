# xAI cache affinity audit

This change adds **the `x-grok-conv-id` routing header to xAI requests** on the
shared OpenAI-compatible client. The existing request body, credential
handling, `prompt_cache_key` body field on Responses (unchanged, and not sent
on chat/completions either before or after), reasoning replay policy and error
handling are untouched. It also ships `scripts/bench_xai_cache_rate.py`, the
live A/B benchmark that produced the evidence below.

## The problem, measured correctly

xAI's cache is automatic but **per-server**: without a routing key the load
balancer can send consecutive turns of one conversation to different servers,
where each cold server bills the full prompt. The comparable providers on this
machine all route with an explicit affinity mechanism, so their rates bound
what "no avoidable cold misses" looks like.

Measured from the local analytics ledger (`~/.local-operator/analytics.db`),
14 days to 2026-09-10, `ok=1` calls, using the harness formula
`cache_read_tokens / context_tokens`
(`local_operator/analytics/model.py::UsageAggregate.cache_hit_rate` — for
OpenAI-shaped wires `context_tokens` equals `input_tokens`, which already
INCLUDES cached tokens):

| provider | prompt-side tokens | cached | hit rate |
| ---------- | -----------------: | -----: | -------: |
| zai        |      1,897,961,916 | 1,855,083,840 | 97.7% |
| kimi       |      1,397,048,491 | 1,354,048,566 | 97.4% |
| anthropic  |    (denominator includes its separately-reported read/write buckets) | | 97.0% |
| **xai**    |      1,483,992,010 | 1,395,680,128 | **94.0%** |

Per-session (grok models, sessions ≥ 100k context tokens, n=68): aggregate
94.0%; 10 short sessions sit below 80% but carry only 1.5% of tokens.

> **Denominator trap, recorded so it is not rediscovered:** an earlier
> diagnosis divided `cache_read / (input + cache_read + cache_write)`. On
> Anthropic's wire that is correct (its `input_tokens` EXCLUDES cached
> tokens); on every OpenAI-shaped wire `input_tokens` ALREADY includes them,
> so the formula double-counts the denominator and reports ~half the true rate
> (it produced a spurious "48.5%"). `cache_hit_rate` exists precisely because
> the two wire conventions differ; use it.

So the real gap xai carries against its affinity-routed peers is **~3–4
points of prompt-side tokens**, not a factor of two. The header aims at that
gap. Nobody should expect a doubling from it.

## Why this header, and why derived from the cache lineage

- xAI documents `x-grok-conv-id` (Chat Completions header) and
  `prompt_cache_key` (Responses body field) as the way to route a conversation
  to the same server and maximize cache hits:
  https://docs.x.ai/developers/advanced-api-usage/prompt-caching —
  see its `maximizing-cache-hits` and `multi-turn` subsections. The same pages
  name omitting previous `reasoning_content` as the top cause of misses on
  reasoning models (see "Reasoning replay" below — already correct here, now
  pinned by a test).
- omp (upstream `can1357/oh-my-pi`) ships
  `promptCacheSessionHeader: "x-grok-conv-id"` gated on grok models or a host
  matching `xai` (`packages/catalog/src/compat/openai.ts`, `resolve.ts`), with
  the cache key falling back to the session id.
- xAI's own open-source grok-build sends
  `.header("x-grok-conv-id", self.conv_id)` on every request
  (`crates/codegen/xai-grok-sampler/src/client.rs`).

The value is `str(uuid.uuid5(uuid.NAMESPACE_URL,
"local-operator:grok-cache:<prompt_cache_key>"))` — the same identity
discipline as the Codex `session-id`/`thread-id` pair
(see [OPENAI_CACHING.md](OPENAI_CACHING.md)):

- **Opaque**: never a filesystem or session label on the wire.
- **Stable**: across retries, resumes, model changes and fresh client
  instances — and identical across the `xai` ↔ `xai-oauth` credential
  flavours, so failover mid-session keeps the conversation on its warm server
  (pinned by `test_failover_between_credential_flavours_keeps_conv_id`).
- **A group, not stored history**: forks intentionally inherit the cache
  lineage and still replay full input; the header does not retrieve state.

## Scope and gating

Header is attached only when `ChatRequest.prompt_cache_key` is non-empty
(isolated calls keep today's credential-agnostic routing — no key, no header),
and only when the request actually goes to xAI:

- provider is `xai` or `xai-oauth` (both share `https://api.x.ai/v1`), or
- the request's actual URL host is `api.x.ai` — which covers custom
  OpenAI-compatible entries pointed at xAI, mirroring omp's host matching.

Both wire paths apply it: `stream()`'s chat/completions route (what xai uses
today — the client factory forces `openai_api="chat_completions"` for every
non-openai provider) and `_stream_responses` (reachable by direct
construction; the Responses `prompt_cache_key` body field is unchanged).
`tests/unit/providers/test_xai_affinity.py` pins presence, stability, both
absent-cases, the host gate, failover equality, no-raw-key, and both wire
paths end to end over a mock transport.

## Live evidence (scripts/bench_xai_cache_rate.py)

Method: same adapter, same bodies, 7-turn small conversations (grok-4.6,
~3–4.3k prompt tokens/turn, reasoning replay on). The baseline arm strips
exactly one header in a transport wrapper after proving the client added it;
the affinity arm sends it. Arms use disjoint synthetic prefixes AND disjoint
cache lineages, and the per-seed namespace PADS the first ~512-token block —
see the isolation lesson below for why that is load-bearing. Full per-turn
JSONL is quoted verbatim in the PR description.

Run 3 (isolated; `--seed xai-iso3`, cold start confirmed by turn 0 caching
zero):

```text
arm          turns    prompt    cached   hit rate
baseline         7     21506     13696     63.7%
affinity         7     26289     13824     52.6%

baseline per turn: 512/2993, 640/3019, 512/3045, 2944/3071, 2944/3099, 3072/3126, 3072/3153
affinity per turn:    0/2993, 512/3436, 3328/3664, 3584/3854,  512/3923, 2944/4092, 2944/4327
```

What this run actually shows:

- **Affinity warms faster**: 91% cached by turn 2, 93% by turn 3; the
  headerless arm needed until turn 3 to reach 96% — consistent with pinned
  routing finding the warm server immediately.
- **One collapse event**: the affinity arm fell from 3584 to 512 cached tokens
  on turn 4 and never fully recovered (68–72% vs baseline's 97%). Whether
  that is per-server eviction (pinning means one entry, and one entry can be
  lost) or noise cannot be resolved at this sample size.
- **Aggregates are noise at this scale**: cached tokens move in 512-token
  blocks — 13% of a whole prompt — so single-block effects swing the
  aggregate by tens of points. A 3-point effect (the size of the ledger gap)
  is UNMEASURABLE with 7 small turns. This bench proves wire behaviour, not
  rate deltas.

Both arms also verified on the live wire: every affinity turn carried the
same `x-grok-conv-id` (UUIDv5, stable), every baseline turn proved the header
was added then stripped, no request was rejected, and every turn persisted
`reasoning_content` into `native_replay` (`replay_items=1`); the affinity
arm's prompt visibly grows by its replayed reasoning turn over turn, so the
replay reached the wire, not just the ledger.

> **Isolation lesson (why the namespace pads the first block):** runs 1–2
> used a one-line namespace followed by synthetic records that were
> IDENTICAL across invocations. xAI's cache matches content prefixes at
> block granularity, so unrelated invocations hit each other's records: a
> fresh conversation's FIRST turn cached 1152/1235 tokens it could only have
> inherited from an earlier run. Any future cache bench against xAI must make
> the first block seed-unique or its arms share cache.

The durable rate evidence remains the ledger: watch xai's aggregate
`cache_read/context_tokens` in the weeks after this ships (94.0% baseline,
68 sessions ≥ 100k tokens, 14-day window) against the 97.0–97.7%
affinity-routed peers. If the gap does not close, the collapse-event
hypothesis (pinning to a single evictable entry) is the first thing to
re-examine.

## Reasoning replay (verified, not changed)

xAI's docs name omitted previous `reasoning_content` as the top documented
cause of cache misses on reasoning models. The shared client already captures
`delta.reasoning_content` on the chat stream, persists it via
`native_payload` (endpoint/protocol/credential-scoped), and replays it through
`_replay_chat_message` on the next turn of the same conversation. This change
adds `test_grok_chat_reasoning_round_trip_survives_for_replay` to pin that
round-trip on the grok wire, and the live bench confirmed it on real grok-4.6
turns (`native_replay_items=1` every turn, reasoning visibly growing the
replayed prefix).

## What this deliberately does not do

- No Responses-side changes: `prompt_cache_key` body behaviour on the public
  Responses route is untouched, and xai cannot reach it through the client
  factory anyway (non-openai providers are forced to chat/completions).
- No new persisted identity, no config flag, no transport session reuse: the
  conv id is derived, never stored.
- No change to isolated calls (no cache lineage ⇒ no header, byte-identical
  requests to today's).
