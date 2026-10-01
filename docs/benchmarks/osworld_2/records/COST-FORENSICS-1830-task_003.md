# COST FORENSICS — arm 1830, task_003 r1 ($14.32) vs its own arm's normals

Records-only analysis (no re-runs). Source: each run's own `evidence/*/events.jsonl`
usage records (`input_tokens`, `output_tokens`, `cache_read_tokens`,
`cache_write_tokens`, `usd_cost`). Manager's question: is $14.32 the model genuinely
thrashing on a hard task, or is the harness re-billing a prefix it could have cached?

## Answer

**Not the model thrashing.** Output tokens are trivial (129,862 across 86 calls). The
spend is **cache_write: 5,022,937 tokens** — the harness re-sent a large prefix to the
provider as *uncached* input on 40 of 86 calls, and cache-write is the premium rate.

## Per-run totals (same arm, same route, same budgets)

| run | calls | input | output | cache_read | cache_write | read% | usd |
|---|---|---|---|---|---|---|---|
| **003 r1 (subject)** | 86 | 10,825,613 | 129,862 | 5,802,160 | **5,022,937** | 53.6% | **$14.3239** |
| 001 r1 (normal) | 144 | 15,017,864 | 118,430 | 14,615,483 | 144,449 | 97.3% | $1.6746* |
| 004 r1 (normal) | 51 | 4,610,430 | 69,501 | 4,494,753 | 115,371 | 97.5% | $1.4702 |

*001 priced on 62/144 calls (provider-reported only on those); its cache_write ratio is
the comparable figure, not the dollar total.

## Inside 003 — the two populations of call

| group | calls | cache_write | cache_read | median preceding gap |
|---|---|---|---|---|
| clean calls | 46/86 | 106,650 | 4,722,408 | 35 s |
| **re-writes** (uncached prefix >15k) | **40/86** | **4,916,287** | 1,079,752 | 104 s |

- First re-write at call **#27**; the calls before it are clean.
- On a re-write call, `cache_read` collapses to a fixed ~27,031-token base (system +
  tools) while `cache_write` balloons (73k → 175k and growing with the conversation):
  the prefix beyond the base is re-sent uncached every time.
- On a clean call, `cache_read ≈ input − ~1.3k` (the normal incremental append).

## What it is NOT

- **Not #1805/the newest-frame fix.** Arm **1796-c1-004** — built *without* #1805 — shows
  the identical signature: 111 calls, read 58.0%, **cache_write 6,023,682**, 37 re-writes
  (first at #71), **$17.1957**. So this class pre-exists arm 1830's newest fix and is not
  introduced by it.
- **Not cache TTL expiry.** Re-write gaps are 33–180 s (median 104 s), all far below any
  5-minute cache TTL; clean calls show gaps up to 287 s. No wall-time threshold separates
  them. (The `cache_write_5m_tokens` / `cache_write_1h_tokens` fields are zero on this
  route, so no tier signal is recoverable from the records.)
- **Not image count.** 003 (broken) carries 125 `image`-type events; 001 (clean) carries
  **271**. Raw frame churn does not predict the class.
- **Not retries or fallbacks.** 003: `model_change=0` (no provider fallback),
  `provider_turn_start=87` for 86 calls → ~1:1, no retries. (001 had the one fallback —
  `provider failure: timeout HTTP 400 — falling back to deepseek/deepseek-flash`.)

## What it correlates with

Re-write calls follow **longer** preceding gaps (median 104 s vs 35 s) with overlapping
ranges — i.e. the turns that drive the guest (action + observation) are the ones that
re-bill the prefix, while short reasoning/text turns hit the cache. That is consistent
with a new observation frame entering the prompt mid-conversation and invalidating
everything after its insertion point — but the correlation is not a proof, and the
opposite case exists in-arm (004 r1: 45 action batches, **0** re-writes).

## Verdict for the tranche

003's $14.32 is **harness-attributable re-billing of ~4.9M prefix tokens**, not model
thrash — and it is **pre-existing, not caused by any of the seven fixes**. The
generalizable-fix candidate (append/anchoring the observation frame so the cached prefix
survives an action turn) is real and would benefit every long computer-use episode, but
**fixing it is out of this tranche's scope** by instruction; this record is the evidence
base for a dedicated PR.

Nothing was changed mid-tranche. Arm 1830 remains frozen.
