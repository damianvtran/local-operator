# Direct DeepSeek

The `deepseek` provider uses the native OpenAI-compatible Chat Completions
endpoint. This document records the contract checked on 2026-09-12; it is not
a claim that every optional feature of DeepSeek's official harness is present.
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
`c291e7961a515f6d7af9304e7fd1d257929aef26`](https://github.com/deepseek-ai/deepseek-harness/tree/c291e7961a515f6d7af9304e7fd1d257929aef26).

| Surface | Official reference | Local Operator |
| --- | --- | --- |
| Discovery | Configured static `listModels` | Native live inventory, cached with documented fallback |
| Reasoning replay | Retains previous `reasoning_content` | Retains all provenance-valid native reasoning; raw arguments preserved |
| Tool screenshots | `serialize.ts`: user-only image projection after tool groups | Same role constraint and stable grouping |
| Stream completion | `sse.ts`: requires `[DONE]`; `translate.ts`: malformed JSON fails | Native route requires `[DONE]` plus recognized finish; malformed/truncated streams use retryable provider errors |
| Resource exhaustion | Explicit failure | `insufficient_system_resource` becomes retryable provider failure, never success |
| Cache reuse | Preserves serialized conversation prefix | No direct explicit cache controls; append-only projected messages; measured native hit counters |

Optional Files API uploads, Responses migration, a server-generated `user_id`
policy and the official tokenizer are not introduced here. They are not needed
to fix the native discovery, metadata, replay and cache-continuity defects, and
this change makes no universal “same or better” claim about those features.
