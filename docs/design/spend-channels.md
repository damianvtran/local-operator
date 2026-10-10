# Channel spend: non-inference spend in the session cost story (wire contract v1)

Status: PR-1 of the cost-channels project (design: the project's
`design-cost-channels.md`, held in the manager session's scratchpad; this file
is the in-repo note and the contract the UI PR types against).

## What this is

Session money used to be inference only. `SessionSpend`
(`local_operator/session/spend.py`) counts model turns; web search kept a
second in-memory ledger folded in at paint time by the TUI alone; images, TTS
and STT recorded nothing. Surfaces disagreed, and a resumed conversation could
not answer "what did the images cost".

PR-1 adds **one channel-record ledger beside `SessionSpend`**:

- `local_operator/session/channel_spend.py` — the frozen record, the fold and
  the ONE combiner.
- Transcript rows: `custom_type="session_channel_spend.v1"`, one row per
  record event (`details` = `to_details()`), plus a one-time
  `{"kind": "start", "version": 1, "ts_ms": …}` marker. Bookkeeping rows:
  `preserve_mtime=True`, never LLM context, not collapsible.
- `analytics.db`: `channel_calls` (raw, `record_id` UNIQUE) + `channel_daily` /
  `channel_monthly` rollups (rev-upgrade deltas in the same transaction) +
  the `spend_all` view (`calls` as `channel='inference'` UNION ALL
  `channel_calls`).
- Image and web-search/read emission through a `ToolContext.record_channel_spend`
  callback; `Session` folds, journals, enqueues analytics and republishes.
- The published object below, and a TUI band + `/session` section that read
  ONLY it.

## The rules that are load-bearing

1. **`amount_micro: int | None`; `None` means unknown, NEVER 0.** A call we
   could not size and a call that was free are different facts.
2. **`record_id` is the idempotency key.** `<channel>:<provider correlation id
   or uuid4>`; backfill uses `legacy:<entry id>`. Replaying a journal, re-running
   a backfill or copying rows into a fork cannot double count.
3. **`rev` supersedes.** A higher `rev` with the same `record_id` replaces the
   earlier record (quote → settled); the analytics rollups receive a DELTA.
4. **Subscription money is never mixed with cash.** A plan-funded call carries
   `billing_basis="subscription_api_equivalent"`: the published API price of
   what the plan funded. It gets its own `by_basis` bucket and never joins
   `billed`.
5. **No fabricated zeros.** A session whose journal has no `start` marker has
   `tracked: false` and an inference-only total; every surface must say
   "channels not tracked" rather than imply $0 of channel spend.
6. **One arithmetic site.** `channel_spend.combine()` is the only place that
   sums; every surface reads the published object.

## Wire object: `FrontendSessionState.spend_channels`

Additive on `session_spend`-era state (`extra="allow"` keeps old readers
tolerant). Absent/`null` = "this build does not publish it"; a UI then renders
its legacy inference-only view. Gate new UI on `features.cost_channels >= 1`.

```json
{
  "version": 1,
  "tracked": true,
  "total_micro": 1016000,
  "knowledge": "partial",
  "by_basis": {
    "billed": 53000,
    "subscription_api_equivalent": 53000,
    "estimated": 10000,
    "not_tracked_calls": 2
  },
  "rows": [
    {
      "channel": "inference",
      "provider": "anthropic",
      "model": "claude-sonnet-5-5",
      "label": "anthropic/claude-sonnet-5-5",
      "units": 12.0,
      "unit": "calls",
      "amount_micro": 900000,
      "knowledge": "exact",
      "basis": [
        "not_tracked"
      ],
      "price_versions": []
    },
    {
      "channel": "image",
      "provider": "openai-sub",
      "model": "gpt-image-2",
      "label": "",
      "units": 1.0,
      "unit": "images",
      "amount_micro": 53000,
      "knowledge": "exact",
      "basis": [
        "subscription_api_equivalent"
      ],
      "price_versions": [
        "OpenAI image-generation pricing (gpt-image-2, 1024x1024 medium, 2026-10-09)"
      ]
    },
    {
      "channel": "image",
      "provider": "radient",
      "model": "gpt-image-2",
      "label": "",
      "units": 3.0,
      "unit": "images",
      "amount_micro": 53000,
      "knowledge": "partial",
      "basis": [
        "billed",
        "not_tracked"
      ],
      "price_versions": [
        "Radient GET /tools/media/status cost_usd"
      ]
    },
    {
      "channel": "read",
      "provider": "deepseek:read",
      "model": "",
      "label": "",
      "units": 1.0,
      "unit": "reads",
      "amount_micro": 2000,
      "knowledge": "exact",
      "basis": [
        "estimated"
      ],
      "price_versions": [
        "client-search-table-2026-09"
      ]
    },
    {
      "channel": "search",
      "provider": "tavily",
      "model": "",
      "label": "",
      "units": 1.0,
      "unit": "searches",
      "amount_micro": 8000,
      "knowledge": "exact",
      "basis": [
        "estimated"
      ],
      "price_versions": [
        "client-search-table-2026-09"
      ]
    }
  ],
  "children": {
    "total_micro": 0,
    "knowledge": "exact"
  }
}
```

### Field semantics

| Field | Meaning |
|---|---|
| `version` | Wire version (1). An unknown version renders nothing. |
| `tracked` | False = no channel `start` marker in the journal (pre-feature session). Inference-only total; say so. |
| `total_micro` | Grand total: session inference + channel records + children, integer micro-USD. |
| `knowledge` | `unknown` \| `partial` \| `floor` \| `exact` — the same four values as `cost_knowledge`; see the matrix. |
| `by_basis` | `billed`/`subscription_api_equivalent`/`estimated` are micro-USD sums; `not_tracked_calls` is a COUNT of records with no trackable money basis (inference contributes one until PR-3 adds its basis columns). |
| `rows` | Aggregated rows: inference by provider/model (`label` = the identity bucket) then one row per (channel, provider, model, unit). `amount_micro: null` = nothing in that group was sized. `basis` lists the bases present; `price_versions` lists the labels. |
| `children` | The subagent/forked-children contribution already inside `total_micro` (PR-1: the children's inference ledger; relayed child channel records join in the follow-up). |

### Knowledge matrix

- `unknown` — nothing stateable: no money anywhere **and** no sized record/call
  (this includes a session with no activity at all, matching
  `SessionSpend.knowledge()`).
- `partial` — some money is unstateable: a channel record with `amount_micro:
  null` that is not a `failed`-unbilled job (unreported `ok`, unsettled
  `cancelled`), unpriced model calls, a child ledger with unknowns, or channel
  rows on a session with `tracked: false` (recovered history).
- `floor` — rows were positively reported lost.
- `exact` — everything that spent money has a stated figure.
- Precedence: `unknown` > `partial` > `floor` > `exact` (unpriced noise is
  permanent; a floor is a retained bound).

A `failed` record with `amount_micro: null` is the documented presumption that
an unreported failure was not charged: it does not degrade knowledge. A
`cancelled` record that never settled DOES, because a cancelled job may still
be charged.

## Emission points (PR-1)

- **image** — `tools/image_tool.py`, one record per generation from the wave-2
  `cost_usd`/`cost_source`/`billing_basis` details (any of them absent ⇒
  `amount=null`/labelled). Radient's generate-time figure is a QUOTE
  (`estimated`); the settled figure is `billed`, and the fold/analytics
  rev-upgrade carries quote→settled when the rung exposes both.
- **search / read** — `web_search/tool.py` and `web_search/read_tool.py`, from
  the same estimate the search ledger already computes: `estimated`,
  `catalogue`, `price_version="client-search-table-2026-09"` (marked
  MIGRATE to catalogue).
- **TTS / STT** — the tolerant Radient `cost`-object adapter exists
  (`clients/radient_cost.py`, marked PENDING CONTRACT); route-level wiring and
  the owner hand-off are the PR-3 slice, per design §4.2/§6.
- **backfill** — a pre-feature journal recovers `search_cost`/`read_cost` and
  image rows that carry a figure; rows without a figure are NOT invented.

## Out of scope here

- Inference `billing_basis`/`cost_source` columns (PR-3): inference rows carry
  `basis: ["not_tracked"]` until then.
- Analytics HTTP `channels` sections, mobile relay (PR-2).
- TTS/STT route wiring (PR-3).
