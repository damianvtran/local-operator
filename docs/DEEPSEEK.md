# Direct DeepSeek

The `deepseek` provider uses the native OpenAI-compatible Chat Completions
endpoint. This document records the contract first checked on 2026-09-12, with
the official-harness comparison re-audited on 2026-10-17; it is not a claim that
every optional feature of DeepSeek's official harness is present.
OpenRouter's DeepSeek routes retain OpenRouter's own routing, cache-affinity and
effort behavior.

## Inventory and metadata authority

The authenticated [`GET /models`](https://api-docs.deepseek.com/api/list-models)
response returned exactly `deepseek-flash` and `deepseek-v4-pro`. Each row had
only `id`, `object` and `owned_by`: **the endpoint did not advertise vision,
tools, limits, prices or reasoning capabilities**.

A successful native listing, including an empty one, owns the selectable set in
both the model picker and `/v1/models` HTTP catalogue. The existing catalogue
cache avoids redundant fetches and provides stale results when offline. When
there is no usable listing, documented static rows remain an honest fallback;
legacy manual selectors are not removed from saved sessions.

Present, valid fields in the recognized compatibility schema override static
metadata through discovery, disk cache, `ModelInfo` and `ModelSpec`. This
includes explicit `false`, zero prices and an empty reasoning effort allowlist.
Omitted fields are enriched from the native documented rows, not from another
provider's route. Unknown future ids do not acquire an aggregator's capabilities
merely because the native listing is sparse.

[Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing) documents:

| Native selector | Current implementation | Context | Maximum output | Images |
| --- | --- | ---: | ---: | --- |
| `deepseek-flash` | V4.1 Flash | 1,000,000 | 393,216 | Yes |
| `deepseek-v4-pro` | V4 Pro 0813 | 1,000,000 | 393,216 | No |

The documented floating aliases `deepseek-v4-flash` and
`deepseek-v4-flash-vision-exp` now serve V4.1 Flash. Old pinned `0731` metadata is
not upgraded speculatively. A live listing hides these legacy selectors unless
the endpoint itself lists them.

Prices are **conservative peak estimates**, not exact time-of-use bills:

| Per million tokens | Flash | Pro |
| --- | ---: | ---: |
| Uncached input | $0.30 | $1.32 |
| Cached input | $0.006 | $0.044 |
| Output | $1.20 | $3.96 |

Off-peak prices are half. Peak is Monday–Friday 01:00–04:00 and 06:00–10:00 UTC.
The local usage model does not apply a time-of-use tariff. The provider's bill
remains authoritative; these estimates must not be presented as charged cost.

## Thinking, images and stable caching

The native [thinking contract](https://api-docs.deepseek.com/guides/thinking_mode)
offers `none`, `low`, `high`, `max`, with `high` as the default. `none` sends
`thinking.type=disabled` and omits `reasoning_effort`; other levels enable
thinking and send their native effort. This is provider-scoped, not a regex
applied to every DeepSeek model on every host.

The [completion schema](https://api-docs.deepseek.com/api/create-chat-completion)
distinguishes maximum output from request budget. Unspecified budgets use 8,192
in non-thinking mode, 65,536 in thinking mode, and 131,072 at max effort. Explicit
small requests stay small; requests cannot exceed the advertised maximum.
Sampling parameters that are ignored or constrained by native thinking mode
remain omitted, leaving DeepSeek's defaults intact.

With tools, **all previous native reasoning must be replayed**, including
reasoning-only assistant turns. Existing replay provenance checks (provider,
model, endpoint, account and message integrity) still apply. Display text or
another account's retained payload cannot bypass them. Raw tool argument bytes
remain intact.

The [vision guide](https://api-docs.deepseek.com/guides/vision) places image input
in user messages. Tool results therefore remain strings; images from a complete
consecutive tool-result group are projected into a following synthetic user
message. Parallel assistant/tool pairing is preserved, and appending a new turn
does not rewrite earlier request messages.

[Context caching](https://api-docs.deepseek.com/guides/kv_cache) is automatic.
DeepSeek persists complete prefix units at request boundaries, detected common
prefixes and fixed token intervals. Matching just part of a previously cached
unit does not guarantee a hit. The direct route sends neither ephemeral
`cache_control` markers nor `prompt_cache_key`; moving markers must not rewrite
old messages. Native hit/miss counters remain the evidence of actual reuse.
Cache persistence takes seconds and is best-effort, not a 100% hit guarantee.

## Official harness comparison

Audit reference: [`deepseek-ai/deepseek-harness` at
`477b4f420553e8a52c2fbccc464d7561b239c443`](https://github.com/deepseek-ai/deepseek-harness/tree/477b4f420553e8a52c2fbccc464d7561b239c443)
(dsh `0.1.7-rc.2`, MIT). The previous audit pinned
`c291e7961a515f6d7af9304e7fd1d257929aef26`; rows are re-checked against the
newer reference, and two of its rows were retired with the Chat adapter it
described.

| Surface | Official reference | Local Operator |
| --- | --- | --- |
| Discovery | Configured static `listModels` | Native live inventory, cached with documented fallback |
| Wire | Anthropic-compatible Messages at `/anthropic/v1/messages` (`PUBLIC_BASE_URL` in `llm-deepseek/src/config.ts`) | Native OpenAI-compatible `/v1/chat/completions`; parity measured, see below |
| Reasoning replay | `serialize.ts` replays every assistant turn's reasoning as a `thinking` block, carrying the provider `signature` when one was recorded | Replays native `reasoning_content` on every assistant turn — the Chat wire's spelling of the same obligation — and substitutes a visible `[thinking not recorded]` where a turn's reasoning was not captured, because this wire's validator rejects an absent field |
| Stream completion | `translate.ts` completes only on `message_stop` with every block closed and a stop reason, and an unterminated stream is `STREAM_CLOSED`; `sse.ts` rejects invalid SSE JSON as `MALFORMED_RESPONSE` | Native route requires `[DONE]` plus a recognized finish; malformed and truncated streams use retryable provider errors |
| Images | `images.ts` accepts them in user messages and tool results; assistant image output is unsupported | Accepted in user messages; images in a tool-result group are projected into a following synthetic user message |
| Resource exhaustion | Explicit failure | `insufficient_system_resource` becomes retryable provider failure, never success |
| Cache reuse | Preserves the serialized conversation prefix; rewrites the leading prompt only on routes without `systemPromptUpdate: 'in-history'` | No direct explicit cache controls; append-only projected messages; measured native hit counters |
| Cache-stable prompt changes | Appends a changed prompt as an in-history `system` message for `deepseek-flash`, so the prefix survives | Persists the initial four system blocks and journals later changes as host-state records at the history tail; a genuine head change starts a new prefix epoch |
| Auxiliary calls | Forces reasoning effort `off` for `purpose: 'session-title'` | Errands clamp to the lowest real rung of the model's ladder, which is `none` — thinking disabled — for this model |
| Oversized tool output | `compaction-tool-result-pruner`: once compaction pressure qualifies, an over-budget text result becomes a bounded head, a "middle pruned" marker and a bounded tail, with the full original kept in the session log | `compaction/pruning.py` prunes by supersede key and byte range — a different mechanism toward the same end |
| Contentless terminal turn | `translate.ts`: a terminal `stop` with zero content blocks becomes retryable `EMPTY_RESPONSE` | Different trigger: an empty `length` truncation (no text, no calls) is retried from its own budget, then steps the effort ceiling down (`harness/loop.py`); an empty `stop` is not what that code recovers |
| Session telemetry | `dsh_session_log` and `dsh_plugin_packages` body extensions plus `x-deepseek-harness-*` headers | Not adopted: a session sends its whole transcript, system prompts and tool output on its first request and a suffix on each later one, and `dsh_plugin_packages` ships the active package inventory every time |

Optional Files API uploads, Responses migration, a server-generated `user_id`
policy and the official tokenizer are not introduced here. They are not needed
to fix the native discovery, metadata, replay and cache-continuity defects, and
this change makes no universal “same or better” claim about those features.

## Measured parity against the official harness

Measured 2026-10-17 against dsh `0.1.7-rc.2`, whose own DeepSeek models run on
the Messages wire. The question was whether moving to that wire, or to anything
else in that harness, buys a measurable improvement for this provider.

`scripts/bench_deepseek_wire_parity.py` runs one verifiable tool task (read a
spec, compute totals from a CSV, write `result.json`) over each wire with the
same three tool schemas, the same executor and the same workspace. Three
independent 8-run datasets per arm; each cell lists those datasets in order:

| | Chat | Messages |
| --- | ---: | ---: |
| correct | 24/24 | 24/24 |
| tool calls | 24, 25, 24 | 25, 24, 26 |
| schema-invalid calls | 0, 0, 0 | 0, 0, 0 |
| turns (median) | 3, 3, 3 | 3, 3, 3 |
| prompt tokens (median) | 2075, 2086, 2080 | 2070, 2070, 2074 |
| cached share | 74.3%, 73.9%, 73.5% | 68.5%, 68.0%, 67.6% |
| wall time (median) | 3.19 s, 3.35 s, 3.41 s | 3.48 s, 3.14 s, 3.36 s |

No correctness, difficulty or speed difference survives replication: turn
medians are 3 on both wires in every dataset, prompt-token medians agree within
0.75%, and wall time changes direction between datasets (Chat faster in one,
Messages in the next, a tie in the third). At this sample size that is noise,
not an effect.

One difference *is* repeatable across all three, and it favours the wire already
in use: the Chat shape's four separate `system` messages keep a larger share of
the prompt cached across runs than one joined `system` string (about 74% vs 68%,
stable across every dataset). That is roughly 120 more uncached tokens per run on
the Messages wire — small in absolute terms, consistent in direction, and not a
reason to migrate.

`malformed` is only observable on the Chat wire: the Messages API returns
`input` already parsed (`translate.ts`), so a zero there is not evidence of zero
attempts. Schema-invalid calls are counted on both.

The Messages endpoint does accept what dsh relies on: `thinking`,
`output_config.effort`, `cache_control` markers (accepted, and inert — the cache
is automatic), a mid-conversation `system` message that becomes the effective
prompt, and the `mid-conversation-tool-changes-2026-07-01` beta (advertised; its
block schema is not publicly documented and was not reproduced here).

**The two wires disagree on multiple system messages, which makes a migration a
correctness hazard rather than a swap.** On Chat, four leading `system` messages
concatenate: an instruction in block 0 survived three later neutral blocks. On
Messages, a later `system` message is the *effective* prompt, so the same
four-block shape would silently drop blocks 0-2. Any Messages route must join
the blocks into one `system` string first.

**Cache headroom is already spent.** Over the 14 days to 2026-10-17, `deepseek`
`purpose='turn'` calls read 81,619,038,707 cached of 82,864,969,705 prompt
tokens — **98.5%** across 433,360 calls, and the same 98.5% over the 3,787
sessions that made 5 or more calls. The source is
`~/.local-operator/analytics.db` and the formula is the one the app itself
reports, `cache_read_tokens / context_tokens` restricted to `ok=1`
(`local_operator/analytics/model.py::UsageAggregate.cache_hit_rate`); on this
OpenAI-shaped wire `context_tokens` already includes the cached tokens. The
residual 1.5% is the hard ceiling on any cache work, and it includes unavoidable
first calls and native cache expiry. For
contrast, the lowest provider in the same ledger is `meta-muse` at 91.1%; the
mechanism dsh uses to keep a prefix warm is already present here and is visibly
working on this provider.

**Tool-call validity has little room either.** The ledger records a `fault` at
the source for every tool call, so model faults (`invalid_arguments`,
`unknown_tool`, `duplicate_id`) are separable from execution, denial and abort
classes. Over the same window, attributing calls by the session's dominant
provider gives 0.56% model faults over 561,080 model-origin calls; counting
every session that used DeepSeek at all gives 0.71% over 688,876. Both bounds
leave under one percent, and neither moves the conclusion; the exact rate
depends on which attribution the reader considers right, so both are stated
rather than one being presented as fact.

Reproduce:

```sh
.venv/bin/python scripts/bench_deepseek_wire_parity.py --live --repeat 8
```

### What this does not establish

The benchmark isolates request shape. It does not exercise the telemetry
extensions (deliberately not adopted), dsh's in-history prompt replacement, or
reasoning replay under a long tool-using history beyond what the task above
produces. On the two metrics the ledger can measure at this provider's scale —
cache reuse and tool-call validity — a harness change has at most 1.5% of
prompt tokens and 0.7% of tool calls to win. What remains unexplained is
model-side tuning against that harness, which none of these measurements can
attribute to anything a caller can send.
