# Model catalogue freshness and provider-neutral pricing — design

Status: proposal for one coder PR. Author: architect (lopdev team), 2026-09-01.
Scope was extended twice mid-design by the manager; the final scope is:

1. A model released today is offered today by `/model` and priced today by
   the status band, while repeated queries stay at disk speed.
2. Direct-provider prices stop depending on OpenRouter: a user signed in only
   to Anthropic gets correct price analytics for `claude-fable-5-1`.
3. `DiscoveredModel` learns `cache_write_price`.

Everything below cites the working tree at `origin/main` as of today.

> **Addendum (2026-09-02, follow-up PR to #527).** Two amendments after the
> design shipped. (a) §3.3/§3.5: leg 2 is a **ranked chain of two independent
> keyless sources**, models.dev first and OpenRouter's public listing second,
> rather than models.dev alone — the operator ruled that a single community
> source is a single point of failure. (b) §3.6: the picker's rows go through
> the **same chain** as the status band, over one read of each document. The
> "delete `_AGGREGATOR_NAMESPACE`" instruction in §3.3 and §7 is superseded:
> the map lives on as `prices.OPENROUTER_NAMESPACE`, behind models.dev.

---

## 1. The problem as found

### 1.1 One 24h TTL, one synchronous miss path, no user-driven refresh

`catalogue.cached_listing` (`local_operator/model/catalogue.py:316-376`) has
exactly three behaviours: age < `DEFAULT_TTL_S` (24h, `:64`) → serve from disk;
otherwise → **synchronous** fetch under a cross-process lease, rewrite, serve;
fetch failure → serve the stale document. There is no "serve stale AND refresh"
state, so the only way to get a newer document is to wait for the TTL or delete
the file.

Every consumer goes through `discovery.available_models` (`discovery.py:1141`)
→ `_available_models` (`:1190`) → `cached_listing` (`:1268`), and every
consumer uses the default TTL:

| Caller | Path | Budget | TTL passed |
|---|---|---|---|
| `configure._info_from_discovery` (`configure.py:918`) | session boot and, via `refresh_model_info_background`, the TUI cost poll off-loop | `DEFAULT_TIMEOUT_S`=10s blocking / `_REFRESH_TIMEOUT_S`=2s non-blocking (`:1284`) | default |
| `configure._from_aggregator_catalogue` (`:1140`) | same resolution, leg 2 | `_AGGREGATOR_TIMEOUT_S`=3s within `_remaining_budget` (`:1139`) | default |
| `controller.live_catalogue` (`controller.py:1049`) | `/model` picker worker, `asyncio.to_thread` per provider (`:1090`) | none beyond discovery's 10s per provider | **`ttl_s` is accepted (`:1050`) but the only caller, `tui/app.py:13655`, never passes it** |

So `/model` paints the static registry, runs `_refresh_catalogue`
(`app.py:13650`) and gets a 24h-cached answer. There is no `/model refresh`,
no `r` key, no CLI command (`grep -n "models" local_operator/cli.py` finds
nothing relevant). Today's incident followed exactly: `anthropic.listing.json`
was 22h old with 10 models; `/model` did not show `claude-fable-5-1`.

### 1.2 Prices for direct providers come from OpenRouter, and the same TTL

Anthropic's `/v1/models` carries no prices (verified: keys are `id`,
`display_name`, `created_at`, `max_input_tokens`, `max_tokens`,
`capabilities`, `type`); OpenAI's and Google's listings carry none either. The
only price source for a direct-provider id the registry has not been taught is
`_from_aggregator_catalogue` (`configure.py:1088-1184`): it reads the keyless
OpenRouter listing (`_AGGREGATOR_CATALOGUE = "openrouter"`, `:975`) and looks
the id up under a per-provider namespace (`_AGGREGATOR_NAMESPACE`, `:996`).
Today `openrouter.listing.json` was 6h old and predated OpenRouter's
`anthropic/claude-fable-5.1` row, so `resolve_model_info("anthropic",
"claude-fable-5-1")` returned `0.0/0.0` until the doc was force-refreshed.

That leg is also structurally fragile: one third party's public listing,
keyless *today*, with its own id spellings (`_aggregator_spellings`,
`:1056`), gates every direct provider's cost display. The operator has ruled
it out as the price source for direct providers.

### 1.3 The write price is guessed

`DiscoveredModel` (`discovery.py:149-181`) has `cache_read_price` and no write
price. `_row_from_openai_entry` (`:396-438`) reads `pricing.input_cache_read`
and ignores `input_cache_write`, which OpenRouter does publish (fable-5.1:
`input_cache_write` 1.25e-5/token = $12.50/MTok). Both consumers then
substitute the input price (`configure.py:958-959` and `:1170-1171`, whose
comment admits the 20% understatement). The legacy `_info_from_listing`
(`configure.py:610-621`) already reads `input_cache_write` correctly — the
discovery path regressed it.

### 1.4 The in-process memo is bucketed, not aged

`resolve_model_info` memoises per `(provider, model_id, time // DEFAULT_TTL_S)`
(`configure.py:1364`). A document refreshed on disk mid-bucket is not seen by
this process until 00:00 UTC. That is documented and acceptable for *drift*
(`:1340-1350`), but it means a *new id* must be refetched inside the memoised
body on its first resolution, or the session runs unpriced all day. This is why
the "miss" trigger (§3.4) has to live inside `_resolve_model_info_cached`, not
beside it.

---

## 2. Options evaluated

Costs are stated per session start (SS), boot/paint latency (L), staleness
bound (S), offline (O), thundering herd (H), code footprint (F).

| # | Option | SS network | L | S | O | H | F | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | Shorter TTL only (1h) | 1 fetch/provider/hour, **synchronous on the boot path** | up to 10s per provider once an hour; the `_REFRESH_TIMEOUT_S` paint leg would hit its 2s cap hourly | 1h | fine (stale on failure) | lease covers | tiny | **Reject.** Moves the pain onto the path that must be cheap. |
| 2 | Stale-while-revalidate (soft/hard TTL) | 0 on the calling path; 1 background fetch/provider/soft-TTL | disk speed whenever a document exists | soft TTL for the *next* call; this process's memo bucket still pins (§1.4) | unchanged: hard TTL keeps stale-beats-absent | lease + per-process in-flight set | moderate (catalogue.py only) | **Adopt** for boot/paint paths. |
| 3 | Explicit refresh: picker passes a short TTL | 0 at boot; 1 fetch/provider/15min *only when the user opens `/model`* | none on boot; picker already paints static rows first and fetches off-loop (`app.py:13636-13648`, `controller.py:1090`) | 15 min for the picker | picker shows "no live list" | lease covers | **one argument** at `app.py:13655` | **Adopt.** Fixes the reported incident by itself. |
| 4 | Event-driven: refetch on a requested-id miss | 1 fetch per *new id* per document, rate-limited by document age | bounded by the caller's existing budget (`_remaining_budget`) | same day for the id the user actually asked for | fails → registry template, as today | lease covers; a young document is believed, so a typo cannot loop | small (discovery + configure) | **Adopt.** The only option that beats the memo bucket for a brand-new id. |
| 5 | Per-provider TTLs | fewer fetches for first-party | none | worse for first-party (the ones that matter) | — | — | small but a second knob | **Reject.** With SWR a background fetch per provider per hour is already cheap; distinguishing classes buys nothing measurable and every class needs "today". |

Recommendation: **2 + 3 + 4 together**, one mechanism each, no per-provider
TTLs, no new user command in this PR.

---

## 3. Recommended design

### 3.1 `catalogue.py`: stale-while-revalidate with soft/hard TTL

Add alongside `DEFAULT_TTL_S` (keep its name and value — it is the memo bucket
in `configure.py:1364,1424` and the hard TTL):

```python
#: Age past which a served document is ALSO refreshed in the background. An
#: hour bounds "released today, offered today" for the next call without
#: putting a request on the calling path; below it a listing endpoint would be
#: asked repeatedly for information that changes on the order of days.
SOFT_TTL_S = 60 * 60
#: A document younger than this that lacks a requested id is believed: the id
#: is genuinely unknown, not newly released. Bounds the miss-triggered refetch
#: (see discovery.available_models(want_id=...)) so a typo cannot refetch on
#: every resolution.
MISS_REFETCH_MIN_AGE_S = 10 * 60
#: Per-process floor between background attempts on one key, so an offline
#: machine does not spawn a thread every time a stale document is served.
REVALIDATE_BACKOFF_S = 5 * 60
```

Introduce a result type and a richer reader; keep `cached_listing` as a thin
wrapper so its 20+ existing tests and both callers keep working unchanged:

```python
@dataclasses.dataclass(frozen=True)
class Listing:
    payload: dict[str, Any] | None
    age_s: float            # inf when absent
    fetched: bool           # a live fetch produced this payload on THIS call
    refreshing: bool        # a background revalidation was scheduled by this call

def read_listing(key, fetch, *, soft_ttl_s=SOFT_TTL_S, ttl_s=DEFAULT_TTL_S,
                 cache_dir=None) -> Listing: ...

def cached_listing(key, fetch, *, ttl_s=DEFAULT_TTL_S, cache_dir=None):
    return read_listing(key, fetch, ttl_s=ttl_s, soft_ttl_s=ttl_s, cache_dir=cache_dir).payload
```

State machine in `read_listing` (replaces `catalogue.py:343-376`):

1. `age < soft_ttl_s` → serve, no I/O beyond the read.
2. `soft_ttl_s <= age < ttl_s` → serve **immediately**, then
   `_schedule_revalidate(key, fetch, cache_dir)`.
3. `age >= ttl_s` or missing → today's synchronous lease path, unchanged
   (`:349-376`). The `fetched` flag replaces the `nonlocal fetched` trick in
   `discovery.py:1229-1246`, which can then be deleted.

`_schedule_revalidate`:

- Per-process dedupe: module-level `_revalidating: set[str]` and
  `_last_attempt: dict[str, float]` under one `threading.Lock`; return
  without a thread if the key is in flight or attempted within
  `REVALIDATE_BACKOFF_S`.
- Thread body: `lease = _ListingFetchLease(path)`; if `acquire()` is False,
  **return** (a peer is fetching; unlike the sync path there is nothing to wait
  for). Else `fetch()` → `_write_cache` → `release()`, any exception logged at
  debug, `finally` clears the in-flight entry. `daemon=True`, named
  `catalogue-revalidate:<key>`. A daemon thread killed by process exit leaves
  a lease that lapses in `_LISTING_LEASE_S` (60s, `:221`) — the existing
  crash contract, no new state.
- *Amended in review round 1 (R1-2, R1-3, R1-5).* The thread runs a separate
  `revalidate` thunk (defaulting to `fetch`) that carries the provider's full
  `DEFAULT_TIMEOUT_S`, not the caller's on-path budget: a background refresh
  that inherited a repaint's 2 s ceiling failed on every slower link, backed
  off, and left the document to the 24 h sync path. A daemon kill mid-`json.dump`
  *does* strand `<key>.listing.json.<random>.tmp` (§9 risk 1 confirmed), so
  `read_listing` also sweeps such files older than five minutes
  (`purge_stranded_temp_files`; age-gated so a peer's in-flight write is never
  taken). And a sync reader that loses the lease to *this process's own*
  thread (call A scheduled it, call B was declared expired by `refetch_if` a
  moment later) joins that thread for at most `_LISTING_LEASE_WAIT_S` instead
  of polling the document's mtime — same ceiling, but it returns the instant the
  refresh lands *or fails*, where the poll spun the whole window on a failure.
- Plain `threading.Thread`, not the asyncio loop: `catalogue.py` is
  synchronous and is entered from the CLI, the server, `asyncio.to_thread`
  workers and `run_in_executor` threads (`configure.py:1478`). The existing
  precedent for "fire and forget off the calling path" is
  `refresh_model_info_background` (`configure.py:1431`); this is the same
  shape one layer down.

`fetch` thunks capture the caller's `client` and credentials
(`discovery.py:1233-1250`); running one in a thread is safe because
`fetch_models` opens and closes its own `httpx.Client` when none is passed
(`discovery.py:862`). The tests' `_StubClient` is passed explicitly and is
single-threaded; tests that exercise the background path must join the thread
(§5).

**Conditional GET support (in scope, small).** Add
`peek_listing(key, cache_dir=None) -> Listing` (read-only, no fetch, no
sweep). A fetch thunk that wants `If-None-Match` reads the previous payload's
`etag` through it and, on 304, returns the previous payload unchanged;
`_write_cache` re-stamps `fetched_at`, which is exactly "validated just now".
No change to `read_listing`'s signature. Used by the price catalogue (§3.5);
verified live: models.dev answers `304` with 0 bytes to a matching
`If-None-Match` (ETag `"2afcb862…"`).

### 3.2 `discovery.py`: statuses, `want_id`, `cache_write_price`

**Statuses.** Extend `ListingStatus` (`:145`) to
`"ok" | "cached" | "stale" | "static" | "unauthenticated" | "empty"`:

- `"ok"` — fetched live on this call (unchanged).
- `"cached"` — served from disk, no fetch attempted or needed (age < TTL).
- `"stale"` (new) — a fetch was attempted and failed; the document on disk is
  what you got. Today both collapse into `"cached"` (`:1308-1309`), which is
  why the picker footer cannot tell "fresh enough" from "offline".
  `tui/app.py:19960` already tests for a status `"unavailable"` that no code
  emits — that dead branch becomes the `"stale"` branch.

`_available_models` computes them from `Listing.fetched` and from whether the
thunk raised (`_ListingUnavailable` is still how the thunk reports failure,
`:186`). The `fetched` nonlocal (`:1229`) goes away.

**`want_id` (option 4).** Add `want_id: str | None = None` to
`available_models`/`_available_models`. After mapping rows (`:1269`), if
`want_id` is set, no row's id matches it exactly *or* after the same
normalisation `_info_from_discovery` applies (`configure._normalised_id`,
`:868` — move the helper to discovery or import it lazily; it is pure), the
listing was **not** fetched on this call, and `listing.age_s >=
MISS_REFETCH_MIN_AGE_S`: call `read_listing(..., ttl_s=0, soft_ttl_s=0)` once
(sync, under the lease, within the caller's `timeout` — which is the caller's
remaining budget, see §3.3) and re-map. Status becomes `"ok"` or `"stale"`
accordingly. One retry, never a loop: the refetched document is 0s old, so a
second miss for the same id in the same process is believed for ten minutes.

*Amended in review round 1 (R1-1).* "Matches" must be wider than the lookup's
exact/normalised test. Anthropic's `/v1/models` lists dated snapshots only
(`claude-sonnet-4-5-20250929`) while the API accepts the undated alias, so the
common configured id `claude-sonnet-4-5` was a miss against a document that
listed its snapshot — one blocking listing fetch on **every** process start
older than the miss floor, indefinitely, for a document the refetch could not
improve. The trigger asks "could a refetch plausibly list this id?", so a row
counts as present when any of its `ids.id_spellings` (as given, date-stripped,
dotted) coincides with any of the wanted id's after normalisation, in either
direction. A genuinely new family id (`claude-fable-5-1` against a
`claude-sonnet-4-5-*` document) still misses and still refetches.

**`cache_write_price`.** Add `cache_write_price: float = 0.0` to
`DiscoveredModel` after `cache_read_price` (`:180`), and thread it through:

- `_row_from_openai_entry` (`:415-437`): `_per_million(pricing.get("input_cache_write"))`.
  Ignore `input_cache_write_1h`.
- `_from_static` (`:945`): `_positive_float(info.cache_writes_price)`.
- `_merge_one` (`:1006`): live-or-static, same rule as `cache_read_price`.
- `_rows_from_payload` (`:1129`): `_positive_float(entry.get("cache_write_price"))`.
- `LISTING_CAPTURE_VERSIONS` (`:118`): `{"anthropic": 2, "openrouter": 2,
  "radient": 2}` with a docstring line per bump. The stamp is per provider id
  (`listing_capture_version`, `:125`), so only the two OpenAI-compat
  aggregators that quote a write price pay the one-time refetch. A version-1
  document parses to rows with `cache_write_price=0`, which is harmless, but
  without the bump the fix would be invisible for a day on every install — the
  same argument as `:110-117`.

**xAI first-party prices (best-effort, optional in this PR).** xAI documents
`GET /v1/language-models` returning `prompt_text_token_price`,
`completion_text_token_price`, `cached_prompt_text_token_price` in USD cents
per 100M tokens (÷ 10 000 → $/MTok). The manager's OAuth credential gets 403
on it, so it is API-key only. Design: a provider-keyed override table
`_PROVIDER_TRANSPORTS = {"xai": _fetch_xai}` consulted before
`_WIRE_TRANSPORTS` in `_build_transports` (`:793-808`); `_fetch_xai` tries the
priced endpoint and on any non-200 falls back to `_fetch_openai_compat`
unchanged. Bump `LISTING_CAPTURE_VERSIONS["xai"]`. **Defer to a follow-up PR
unless the coder has an xAI API key to prove the conversion against a real
response** — models.dev covers 7/12 xAI models today, and a guessed unit
factor on a price field is worse than no field.

### 3.3 `configure.py`: resolution legs and budgets

`_resolve_model_info_cached` (`:1241-1311`) becomes:

```
info = _registry_fallback(...)                                   # unchanged
if _listing_can_correct(info):
    info = _info_from_discovery(..., want_id=model_id, timeout=…)  # leg 1: provider listing
if _needs_enrichment(info):
    info = _from_price_catalogue(canonical, model_id, info, timeout=_remaining_budget(started))  # leg 2: models.dev (NEW)
if _needs_enrichment(info) and canonical in AGGREGATOR_PROVIDERS:
    info = _from_aggregator_catalogue(...)                        # leg 3: only for aggregator ids
```

Precedence stays "fill the holes" (`_from_aggregator_catalogue` only writes
where `not (info.input_price or info.output_price)`, `:1159`; same for the
window `:1172` and `max_tokens` `:1182`). A registry price is a first-hand
transcription and is not overridden by the catalogue in this PR; a
`prices_from_catalogue` flag mirroring `limits_from_listing`
(`configure.py:677-701`) would let the catalogue correct it and is deferred.

**Leg 2 is a ranked chain, not one source (amended 2026-09-02).** Inside
`_from_price_catalogue`, `prices.price_catalogue_row` runs:

```
models.dev projection  → PRICED row?  yes → answer (limits ride along)
                       → no  → OpenRouter public listing, `<namespace>/<spelling>`, priced rows only
                                 → hit → answer (money from OpenRouter; any native limits
                                          models.dev's cost-less stub carried are kept)
                                 → miss → None → the registry row the caller already holds
```

Why two: one community-maintained JSON is one point of failure. A day-0 gap
(the row not merged yet), a shape drift, or the host being down would unprice
*every* direct provider at once — the same class of outage that motivated
leaving OpenRouter, only with the roles swapped. OpenRouter's listing is
keyless and already on disk for the picker's `openrouter` rows, so keeping it
as the secondary costs users nothing and gives a second, independently
maintained opinion. Rules, all enforced in `price_row` and its tests:

- OpenRouter never overrides a price models.dev quoted, and neither overrides
  a price the provider's own listing quoted (`_fill_from_row` is holes-only).
- OpenRouter stays authoritative for `openrouter/*` ids; those never enter the
  chain's secondary step (no namespace entry), and legs 1/3 own them.
- Both documents sit under the same SWR / lease / `want_id` machinery
  (`prices._models_dev_providers`, `prices.openrouter_rows` →
  `available_models("openrouter", want_id="<ns>/<id>")`).
- The OpenRouter step fits `_remaining_budget`: it receives what the models.dev
  step left of the leg's 3 s, and is **not reached at all** when models.dev
  answered — no second parse, no second request on the paint path.
- `alibaba-token-plan` has no namespace on purpose: models.dev's 0/0 for a
  credit-billed plan is the intended answer, and borrowing `qwen/`'s USD rate
  would print a cost the user is not paying.
- When both sources price an id and disagree by more than 5 %
  (`PRICE_DISAGREEMENT_RATIO`), the fact is logged at debug and nothing acts
  on it.

`_info_from_discovery` (`:886`): pass `want_id=model_name` through to
`available_models`; take `cache_writes_price` from `row.cache_write_price`
when `> 0`, else keep the input-price fallback at `:958-959` (now only for
listings that genuinely quote no write price).

`_from_aggregator_catalogue` (`:1088`): gate on `provider in
AGGREGATOR_PROVIDERS` (it is the `openrouter/*` and `radient/*` ids' own
listing, so leg 1 has normally already priced them; this leg survives only for
the case where leg 1 was unavailable). Delete `_AGGREGATOR_NAMESPACE` (`:996`)
and `_aggregator_spellings` (`:1056`) — their only purpose was direct-provider
lookup. Their spelling rules (`_DOTTED_VERSION_RE`, `:1017`) move to the
price-catalogue lookup, which needs them for the same reason (`gpt-5.4` vs
`gpt-5-4`). Existing tests that assert direct providers price via OpenRouter
(`tests/unit/model/test_configure.py`, the `_from_aggregator_catalogue`
cases) are re-pointed at `_from_price_catalogue` with the same fixtures
reshaped to the models.dev projection.

Budgets are unchanged in kind: leg 2 caps at
`_PRICE_CATALOGUE_TIMEOUT_S = _AGGREGATOR_TIMEOUT_S = 3.0` inside
`_remaining_budget(started)` — the same "one ceiling for the whole
resolution" rule documented at `:1028-1050`. Leg 3 shares that remainder. The
paint path is unaffected: `tui/costs.py:51-61` only ever reads
`_paint_memo` on-loop and runs the full resolver via
`refresh_model_info_background` in an executor thread.

The `want_id` miss refetch on leg 1 runs inside the same `timeout` the leg
already has (10s blocking at boot when the registry cannot describe the model
— the user is watching a spinner for exactly this; 2s when only correcting a
complete row, in which case a miss is not possible because the registry row
*is* the answer).

### 3.4 Freshness triggers, summarised

| Trigger | Where | Mechanism | Bound |
|---|---|---|---|
| Any read of a document older than 1h | `read_listing` | background thread, lease, backoff | 0 on the calling path; ≤1 fetch/key/hour/process |
| `/model` opened | `app.py:13655` → `live_catalogue(ttl_s=PICKER_TTL_S)` | existing sync fetch off-loop, lease | ≤1 fetch/provider/15 min, only when the user asks |
| Requested id absent from a cached document ≥10 min old | `available_models(want_id=…)` | one sync refetch within the caller's remaining budget, lease | ≤1 refetch per key per 10 min, regardless of how many ids miss |
| Capture version bump | `_rows_from_payload` (`:1080`) | existing invalidate+re-enter (`:1270-1289`) | once per upgrade |

`PICKER_TTL_S = 15 * 60` lives in `providers/controller.py` beside
`live_catalogue`: "the user is asking now; fifteen minutes is short enough
that a release announced during a working session shows on the next open, and
long enough that scrolling in and out of the picker does not re-list nine
providers."

### 3.5 Provider-neutral price catalogue: models.dev

New module `local_operator/model/prices.py` (the manager asked for the document
to go through the same machinery; a separate module keeps the 1,300-line
`discovery.py` from growing a fourth transport that is not a provider).

**Source.** `https://models.dev/api.json`: public, keyless, 4.44 MB, weak
ETag, `cache-control: max-age=0, must-revalidate`, honours `If-None-Match`
with a 304. Provider-keyed; each model carries `cost {input, output,
cache_read, cache_write}` in $/MTok, `limit {context, output}`, `name`,
`attachment`, `reasoning`, `tool_call`, `release_date`. Verified today:
`anthropic.models["claude-fable-5-1"]` → cost 10/50/0.25/12.5, limit
1,000,000/128,000, `release_date` 2026-09-01. Coverage for our providers
(priced/total): anthropic 14/14, openai 43/47, xai 7/12, deepseek 3/3, zai
16/16, google 33/38, mistral 32/34, moonshotai 10/10, alibaba 55/55,
alibaba-token-plan 26/26, openrouter 349/353. **radient is absent** — its own
listing quotes prices, so leg 1 covers it.

**Document.** Key `models-dev.listing` → `~/.local-operator/cache/models-dev.listing.json`
(ends in `.listing`, so `_LEGACY_DOCUMENT_GLOB` cannot match it,
`catalogue.py:77`). Store a **projection**, not the 4.4 MB body:

```json
{"capture": 1, "etag": "\"2afcb…\"",
 "providers": {"anthropic": {"claude-fable-5-1": {"name": …, "cost": {...}, "limit": {...}, "attachment": true, "release_date": "2026-09-01"}}, …}}
```

Measured: the projection over the 13 provider keys we map is **141 KB** — ~30x
smaller than the source and a quarter of `openrouter.listing.json`, so the
25 ms JSON parse the memo exists to avoid (`configure.py:1338`) stays in that
range. Only providers in `_PRICE_CATALOGUE_KEYS` are kept.

**Key map** (`_PRICE_CATALOGUE_KEYS: dict[str, tuple[str, ...]]`, first match
wins; the local id is the *canonical* provider after
`credential_provider_id`, so `xai-oauth`/`openai-device` need no entry):

```
anthropic → ("anthropic",)      openai → ("openai",)         google → ("google",)
xai → ("xai",)                  deepseek → ("deepseek",)     mistral → ("mistral",)
kimi → ("moonshotai", "kimi-for-coding")    zai → ("zai", "zai-coding-plan")
alibaba → ("alibaba",)          alibaba-token-plan → ("alibaba-token-plan", "alibaba")
openrouter → ("openrouter",)
```

**Lookup.** `price_catalogue_row(provider, model_id, *, timeout) ->
DiscoveredModel | None`: `read_listing("models-dev.listing", fetch,
...)`, then for each mapped key try the id exactly, then
`_normalised_id`, then the dotted/dashed spellings that `_aggregator_spellings`
produces today. Returns a `DiscoveredModel` (so `cache_write_price`, window and
`max_tokens` ride the same struct the other legs use) with
`supports_prompt_cache = cache_read > 0`, `supports_images` from
`attachment`... **no**: `supports_images` is deliberately not taken from a
second-hand source (`configure.py:1116-1118` states the three-valued
contract); leave it `None`.

**Fetch thunk.** `httpx.get(URL, headers={"If-None-Match": prev_etag} if
prev_etag else {}, timeout=timeout)`; 304 → return the previous payload
(read via `peek_listing`); 200 → project and stamp the new ETag; anything
else → raise (`read_listing` turns that into stale-beats-absent).
`want_id` miss semantics apply exactly as in §3.2: a direct-provider id
absent from a document ≥10 min old triggers one refetch — which for models.dev
is usually a 0-byte 304 unless the row really did just land.

**Second source.** models.dev is the primary, not the only, keyless price
source — see the chain in §3.3. `prices.OPENROUTER_NAMESPACE` maps each direct
provider to its OpenRouter vendor namespace (`anthropic`, `openai`, `google`,
`deepseek`, `mistral→mistralai`, `xai→x-ai`, `alibaba→qwen`, `kimi→moonshotai`,
`zai→z-ai`); `openrouter_lookup` tries `<ns>/<spelling>` for every
`id_spellings` candidate against PRICED rows only. The pure `price_row(provider,
model_id, *, models_dev, openrouter)` helper takes both documents pre-read so
the resolver (one id, may fetch) and the picker's row builder (hundreds of ids,
disk only) share one ranking and cannot drift.

**Cold start.** The very first resolution on a machine with no document pays
one 4.4 MB download inside the 3s leg budget. On a slow link that times out
and the session runs unpriced until the next resolution — so on a cold miss
that times out, `read_listing` additionally schedules a background fetch with
the full `DEFAULT_TIMEOUT_S` (the one case where a failed sync fetch is
followed by a background retry: a timeout on a large body is evidence the
budget was too small, not that the network is down). Every later start is a
141 KB read and, hourly, a 304.

### 3.6 Picker: what the user sees

`_populate_model_picker` (`app.py:13622`) is unchanged: static rows paint on
the keystroke with the footer clause `checking providers…`.
`_refresh_catalogue` (`:13650`) calls `live_catalogue(ttl_s=PICKER_TTL_S)`.

**Row prices come from the same chain (amended 2026-09-02).** #527 left the
picker pricing rows from `merge_models(registry, listing)` only, so a
direct-provider model absent from the shipped registry painted a blank price
(`_price`'s `-1` sentinel) while the status band priced it at `$10/50` the
moment it was selected. `live_catalogue` now runs `controller._enrich_prices`
after the listings: one `models_dev_providers()` read (disk only, any age, no
fetch), the `openrouter` provider's own rows from the same call as the
secondary, `price_row` per row whose listing quoted no money. Only the money
and the limits the listing left at zero are filled; aggregator rows are never
enriched; providers the chain does not map keep `_price`'s unknown-≠-free
rule. Measured: ~1–3 ms over 588 rows. `static_catalogue` (the first frame)
stays registry-only — it reads no document.
`_catalogue_status` (`:19946`) becomes:

- `"cached"` → **silent**. With a 15-minute picker TTL "cached" means "listed
  within the last quarter hour", which is not something the user needs to
  read; the existing docstring's "a footer that always says something is a
  footer nobody reads" applies.
- `"stale"` → `stale list: anthropic, xai`; when every provider that produced
  a listing is stale (offline), `stale list: all providers`. The label is
  budgeted for the 39 cells left beside the access note at 100 columns and
  parallels `no live list:` (design review round 1, D1/D2).
- `"empty"` → `no live list: …` (unchanged).

The widget API (`ModelPicker.set_rows`, `model_picker.py:312`) is untouched;
`ModelPicker` now keeps its status row (blank) once one has painted in an
open, so `checking providers…` clearing on settle no longer shrinks the card
by a row and moves the transcript above it (design review round 1, D3). This footer wording is a user-visible
string change and needs the designer round with before/after stills of the
open picker in the `stale` state (use `FakeProviderController.live_catalogue`
in `tests/unit/tui/test_app_pilot.py:3672` to force statuses, then
`app.save_screenshot`).

**Deferred:** a `/model refresh` argument (or `r` in the picker, mirroring
`/usage`'s `force_refresh`, `app.py:15089-15123`) that calls
`live_catalogue(ttl_s=0)`. Cheap, but it is a new interaction and would add a
UX round to this PR; the 15-minute TTL already covers the incident.

---

## 4. Concurrency story

- **Cross-process**: `_ListingFetchLease` (`catalogue.py:229`) unchanged and
  reused by all three fetch paths. Background revalidators that lose the lease
  simply exit; sync fetchers that lose it keep today's brief wait
  (`await_peer_briefly`, `:283`).
- **In-process**: one in-flight set per key for background work; the
  miss-refetch and the picker's sync fetch both go through the lease, so two
  threads in one process (the picker worker and a `refresh_model_info_background`
  executor) resolving the same provider at once produce one request.
- **Write safety**: `_write_cache` (`:148`) is already mkstemp+rename; the
  background writer adds nothing new.
- **Memo**: `_resolve_model_info_cached`'s bucket is unchanged. The new-id
  case is handled *inside* the memoised body by `want_id`, so the memo caches
  the fresh answer. `invalidate_model_info_cache` (`:1314`) is untouched.

---

## 5. Test plan

Conventions: `tests/unit/model/test_catalogue.py` uses `tmp_path` cache dirs
and a `calls` list in the thunk; `test_discovery.py` uses `_StubClient` /
`_Response` and `cache_dir=tmp_path`; AGENTS.md "Timing" forbids clock waits
— join threads or wait on the document's existence, never `sleep`.

`test_catalogue.py`

- `test_a_document_between_soft_and_hard_ttl_is_served_and_revalidated_in_the_background`
  — plant `fetched_at = now - 2h`; `read_listing` returns the old payload
  with `refreshing=True` and zero sync calls; join `catalogue._revalidation_threads()`
  (expose a test hook that returns live threads) and assert the file now holds
  the new payload.
- `test_a_document_under_the_soft_ttl_schedules_nothing`.
- `test_a_document_past_the_hard_ttl_still_fetches_synchronously` (guards the
  offline/cold contract; extends `test_an_expired_cache_refetches`).
- `test_a_background_revalidation_that_loses_the_lease_exits_without_fetching`
  — plant an unexpired `.fetching` file.
- `test_a_failed_background_revalidation_keeps_the_stale_document_and_backs_off`
  — thunk raises; second `read_listing` within `REVALIDATE_BACKOFF_S` spawns
  no thread (assert via the hook, not timing).
- `test_two_reads_in_one_process_spawn_one_revalidation`.
- `test_cached_listing_is_unchanged_for_existing_callers` — existing tests
  already prove this; add one asserting `cached_listing` never schedules a
  thread (its soft TTL equals its hard TTL).
- `test_peek_listing_reads_without_sweeping_or_fetching`.
- `test_a_background_thread_runs_off_the_calling_thread` — structural
  `threading.get_ident()` spy, per AGENTS.md.

`test_discovery.py`

- `test_available_models_reports_stale_when_a_refetch_fails` (split from
  `test_available_models_reports_cached_when_a_stale_refetch_fails`, `:1236`).
- `test_a_missing_want_id_refetches_a_document_old_enough_to_be_wrong` — 10
  models cached 22h ago, stub answers 19 including the id; status `"ok"`, one
  call, row present.
- `test_a_missing_want_id_is_believed_when_the_document_is_young` — cached
  1 min ago, zero calls.
- `test_a_want_id_that_is_present_makes_no_request`.
- `test_openrouter_cache_write_price_is_read_and_scaled` (extend
  `test_openai_compat_parses_a_captured_openrouter_payload`, `:154`, with an
  `input_cache_write` in the fixture).
- `test_merge_prefers_a_live_cache_write_price` / `..._keeps_the_static_one_when_live_is_zero`.
- `test_a_cache_write_price_survives_the_disk_round_trip` (`_rows_from_payload`).
- `test_an_openrouter_document_from_capture_one_is_refetched_once` (mirrors `:1372`).

`tests/unit/model/test_prices.py` (new)

- Projection: a captured models.dev fixture (trim to two providers) projects to
  only mapped providers and the five fields; size assertion is structural (no
  `description` key survives), not numeric.
- `test_a_matching_etag_returns_the_previous_payload_and_restamps_it` — stub
  304.
- Lookup: exact id, normalised id, dotted/dashed spelling, `kimi →
  moonshotai` then `kimi-for-coding`, unknown provider → `None`.
- `test_supports_images_is_never_taken_from_the_price_catalogue`.

`tests/unit/model/test_configure.py`

- `test_a_direct_provider_is_priced_from_the_neutral_catalogue_not_openrouter`
  — `available_models` stub for anthropic returns the row unpriced; models.dev
  stub prices it; OpenRouter stub asserts **zero** calls.
- `test_the_write_price_reaches_model_info` — `cache_writes_price == 12.5`,
  not `10.0`.
- `test_the_input_price_fallback_survives_a_listing_with_no_write_price`.
- `test_leg_two_shares_the_resolution_budget` — extend the existing
  `_remaining_budget` test to the new leg.
- `test_openrouter_ids_still_fall_back_to_their_own_listing`.
- Re-point the direct-provider `_from_aggregator_catalogue` tests as in §3.3.

`tests/unit/tui`

- `test_the_picker_asks_for_a_fifteen_minute_listing` — the pilot's
  `FakeProviderController.live_catalogue(self, *, ttl_s=None)` records
  `ttl_s`; assert `== PICKER_TTL_S`.
- `test_model_picker.py`: footer renders `stale list: …` for `stale` and
  nothing for `cached`; every-provider-stale collapses to `all providers`.

Real-execution evidence for the PR (per the operator's standing rule): with
`anthropic.listing.json` and `models-dev.listing.json` deleted, boot `lop`
on `anthropic/claude-fable-5-1`, show the status band pricing at 10/50, show
both documents written, `stat` their ages; then age `anthropic.listing.json`
to 2h by editing `fetched_at`, open `/model`, capture the picker before and
after the worker lands, and show the document's `fetched_at` advanced with
the `.fetching` lease gone.

---

## 6. Constants, in one place

| Name | Module | Value | Why |
|---|---|---|---|
| `DEFAULT_TTL_S` | catalogue | 24h (unchanged) | hard TTL; memo bucket; sync fetch beyond it keeps offline semantics |
| `SOFT_TTL_S` | catalogue | 1h | staleness bound for the next call at zero on-path cost |
| `MISS_REFETCH_MIN_AGE_S` | catalogue | 10 min | a young document that lacks an id is right; bounds typo refetches |
| `REVALIDATE_BACKOFF_S` | catalogue | 5 min | one background attempt per key per five minutes while offline |
| `PICKER_TTL_S` | controller | 15 min | the user is asking; sync fetch is already off-loop behind painted rows |
| `_PRICE_CATALOGUE_TIMEOUT_S` | configure | 3s (= `_AGGREGATOR_TIMEOUT_S`) | same leg-2 budget rule; reachable from the executor thread of the 1 Hz poll |
| `LISTING_CAPTURE_VERSIONS` | discovery | `anthropic 2, openrouter 2, radient 2` (+`xai 2` if the transport lands) | readers now need `cache_write_price` |

---

## 7. Files to touch

- `local_operator/model/catalogue.py` — `Listing`, `read_listing`,
  `peek_listing`, `_schedule_revalidate`, constants; `cached_listing` becomes
  a wrapper. Module docstring §"WHY IT IS CACHED" gains the SWR state.
- `local_operator/model/discovery.py` — `ListingStatus` + `"stale"`;
  `want_id`; `cache_write_price` in five places; capture bumps; drop the
  `fetched` nonlocal. Optional `_fetch_xai`.
- `local_operator/model/prices.py` — new (§3.5).
- `local_operator/model/configure.py` — leg order, `_from_price_catalogue`,
  `want_id` pass-through, write-price plumbing, gate/prune
  `_from_aggregator_catalogue`, delete `_AGGREGATOR_NAMESPACE` and
  `_aggregator_spellings` (move spelling helpers to `prices.py`).
  *Amended:* the namespace map returns as `prices.OPENROUTER_NAMESPACE`, the
  secondary step of leg 2's chain (§3.3).
- `local_operator/providers/controller.py` — `PICKER_TTL_S`.
- `local_operator/tui/app.py` — `:13655` pass the TTL; `_catalogue_status`
  wording.
- Tests as in §5. `pyproject.toml` patch bump: this is a fix plus a
  self-contained enrichment change, not a new surface.

Rough size: ~350 lines of source, ~500 of tests. One coder, one PR.

---

## 8. Deferred (recorded here, not as issues)

- `/model refresh` / `r` key — needs a UX round; not required for the incident.
- xAI first-party price transport — needs a real API key to validate units.
- `prices_from_catalogue` registry flag so models.dev can *correct* a
  transcribed registry price, not only fill a hole.
- Parallelising `live_catalogue`'s per-provider fetches with `gather` — turns
  the picker's worst case from a sum of provider timeouts into a max; small,
  independent, and reviewable on its own.
- LiteLLM's `model_prices_and_context_window.json` as a third neutral source
  — per-token floats under litellm-specific ids; worse fit than the OpenRouter
  secondary the chain already carries.

---

## 9. Risks to watch at rollout

1. **Background threads in short-lived CLI processes.** A `lop --model …`
   that exits in under a second may kill the revalidator mid-write; mkstemp+rename
   means the document is never half-written, and the lease lapses in 60s. Watch
   for stranded `*.tmp` files in `~/.local-operator/cache`
   (`_write_cache`'s `finally` should prevent them; a daemon kill bypasses
   `finally`). Review round 1 confirmed the daemon-kill path is reachable;
   `purge_stranded_temp_files` now sweeps `*.listing.json.*.tmp` older than
   five minutes on every `read_listing`.
2. **models.dev availability / shape drift.** It is a GitHub-maintained JSON;
   a key rename lands as "no prices" (holes stay holes), never as wrong prices.
   `capture` stamp on the projection lets us force a refetch when the reader
   changes. Rate: one 304 per machine per hour — negligible.
3. **The 4.4 MB cold fetch inside a 3 s budget.** Mitigated by the cold-miss
   background retry (§3.5). Confirm with evidence that a cold boot on a
   throttled link ends with the document present on the *second* resolution.
4. **Statuses**: `"stale"` is a new `Literal` member; any exhaustive match on
   `ListingStatus` (pyright will find them) must learn it.
5. **Removing the OpenRouter namespace lookup** changes prices for any
   direct-provider id models.dev lacks but OpenRouter had (openai 4, xai 5,
   google 5, mistral 2 today). Those rows fall back to the registry template —
   the honest "cost unavailable", which is the pre-aggregator behaviour and the
   operator's stated preference over a second coupling.
