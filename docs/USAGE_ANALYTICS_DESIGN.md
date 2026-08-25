# Design: Calendar time-series analytics (daily/monthly rollups)

Status: IMPLEMENTED (this document describes the shipped extension).
Audience: the reviewer/designer/ux-reviewer rounds and future maintainers.

> **History note.** An earlier draft of this file proposed a *new* parallel
> `UsageStore` / `AnalyticsPanel` / `/analytics` command / recording hook. By
> the time it was implemented the harness already had a full analytics
> subsystem (PRs #247/#254/#259): an append-only per-call ledger
> (`local_operator/analytics/store.py` `AnalyticsStore`), a non-blocking
> background recorder (`recorder.py`), a live `/analytics` command, and the
> `AnalyticsScreen` report. Building a second store/command/hook would have
> duplicated that and re-introduced the very double-count the old draft warned
> about. This design therefore **extends** the existing subsystem instead. The
> stale proposal is superseded by what follows.

---

## 1. What was missing

The existing subsystem aggregates the raw `calls` ledger with `GROUP BY
provider`/`session` and a flat 90-day retention. It could not answer the
*historical* question — "what did I spend each day / each month, per model,
over time" — because:

- the ledger is pruned at 90 days, so a year-long daily view is impossible from
  it alone;
- it is grouped by provider/session, never by calendar day/month;
- the `/analytics` screen showed totals and breakdown tables, no time series.

### 1.1 Cost semantics and current billing-mode limit

Provider-reported account charges take precedence over catalogue arithmetic. In
particular, OpenRouter's final streaming `usage.cost` is persisted on the call,
summed across tool-loop calls, and reused by analytics and the TUI instead of
reconstructing a routed request from a flat model price. When no receipt exists,
the displayed dollar amount remains a token-price estimate.

The estimate cannot yet be labelled reliably as metered spend versus an API-rate
equivalent for every call. Credential selection happens inside the failover
iterator, which currently stamps only the serving `provider` and `model_id` onto
`Usage`; it does not expose the selected credential's identity or kind. Inferring
billing from the provider id would be wrong because one id can rotate between API
keys and OAuth credentials, and provider variants include token-plan and local
endpoints with different economics. Billing provenance therefore needs a new
per-attempt field stamped beside the serving identity after credential selection,
then preserved through transcript, subagent, analytics, and TUI aggregation.
Until that contract exists, the implementation deliberately does not guess a
billing mode from the session's selected provider.

## 2. The extension

Two **rollup tables** maintained by the ledger's own write path, plus a
time-series **bar chart** on the existing screen. No new store, no new command,
no new recording hook.

### 2.1 Rollup tables (`store.py`)

`usage_daily(day, model, tokens…, cost_micro, cost_known, calls, updated_at_ms,
PK(day, model))` and identical `usage_monthly(month, model, …, PK(month,
model))`. `day` is the **local** `YYYY-MM-DD` the call's `ts_ms` falls on,
`month` the local `YYYY-MM` (single-machine tool → wall-clock day; a turn
spanning midnight records under its end day). Both TEXT so they sort lexically.
Added via `CREATE TABLE IF NOT EXISTS` in `_SCHEMA`, so an old DB gains them on
first open — the same idempotent-open discipline the cost columns use.

### 2.2 Population — inside `record_batch`, one transaction, no double-count

The **single existing** ledger write path also upserts each snapshot into
`usage_daily` and `usage_monthly`, in the **same transaction** as the `calls`
insert, via `INSERT … ON CONFLICT(key, model) DO UPDATE SET x = x +
excluded.x`. This is why there is **no double-count risk**: the rollups are fed
by the one write every call already makes, not by a separate app-level hook
that could observe the same spend twice. Each snapshot is priced **once** (on
the recorder's background thread, never the event loop — a cold
`resolve_model_info` blocks) and that one figure feeds the ledger row *and* both
rollup rows, so the three tables can never disagree on a call's cost. The
accumulate upsert is lossless under the many-parallel-`lop` reality (WAL +
`busy_timeout` serialise the physical write; `x = x + excluded.x` makes the
logical merge order-independent).

`model` is the finest identity a `CallSnapshot` carries — `provider/model_id` —
because cost depends entirely on it and subagents/`/model` switches routinely
change it; a per-model row is trivially summed back to a per-day total in SQL.

### 2.3 Retention (`prune()`)

Three independent windows, none affecting the others:
- raw `calls`: unchanged 90-day `ts_ms` prune (its rowcount is still the return
  value);
- `usage_daily`: keep the most recent **365 distinct days** (not rows — each day
  holds one row per model);
- `usage_monthly`: keep the most recent **120 distinct months** (a 10-year
  safety cap on an effectively-unbounded table).

The rollup prunes key on the stored day/month strings (newest-N-that-exist),
not a wall-clock cutoff, so an idle week does not drop a still-recent bucket.
Best-effort and guarded so a missing table degrades rather than raises.

**Forward-fill, not backfill (review C1).** On the release that ships this, the
rollup tables are created empty and filled only by calls recorded from that
point forward; the up-to-90 days of existing `calls` history is deliberately
**not** rolled up. Re-bucketing stored `ts_ms` would require a `strftime` that
exactly reproduces the local bucketing `_local_day_month` does, and any
UTC/local mismatch there would silently misattribute a day's spend — the one
error this store must never make — while the ledger prune bounds a backfill to
90 days anyway. So the historical view starts near-empty at ship and fills in
over the following days/weeks. This is intentional and user-visible.

### 2.4 Read API (`store.py`)

`daily_series(days=30, *, by_model=False)`, `monthly_series(months=12, *,
by_model=False)`, and `series_totals(*, daily_days=30)` — returning frozen
`UsagePeriod` dataclasses (`model.py`). `by_model=False` sums across models in
SQL (`GROUP BY key`); `by_model=True` returns one row per `(bucket, model)`.
Oldest-first so the chart renders newest at the bottom (transcript reading
order). `UsagePeriod` mirrors `UsageAggregate`'s cost interface
(`cost_usd`/`cost_is_known`/`cost_is_partial`, plus `cost_is_floor` for the bar
renderer) so the view reuses the screen's money vocabulary. Every method never
raises; a degraded/empty store returns `[]` / a zeroed period.

### 2.5 View (`analytics_panel.py`)

`build_report` gains optional `daily`/`monthly`/`window_totals`/`metric` params
and renders two titled horizontal bar charts (a daily "**N days with usage**"
chart and a "**Monthly**" chart) between the headline Totals and the
input-attribution section, reusing `proportion_bar` / `_section_header` and
column-measured alignment. Each bar's fill is that bucket's value / window max.

**Metric + toggle self-description (reviews U1/U2/U4/D5).** A `t` key on
`AnalyticsScreen` toggles the charted metric between **cost** (default — the
historical view's purpose) and **tokens**. The active metric is carried in the
**pinned title** (`Usage analytics … bars: cost`/`bars: tokens`), so `t` gives
on-screen feedback even when the charts are scrolled off; both chart section
metas also self-describe (`cost · t → tokens`), so a reader parked on the
Monthly chart still knows what its bars mean. The daily meta additionally leads
with the window's grand total from `series_totals` (review C2 — the total and
the bars describe the same 30-day window).

**Floor mark (reviews D1/D2).** The `≥` mark prefixes a cost cell only when the
bucket is a **genuine lower bound** — cost mode AND `cost_is_floor` AND
`cost_is_known` (mixed priced + unpriced calls). A **fully-unpriced** bucket has
no dollar figure to bound, so it renders a clean `$—` with **no `≥`** (a
`≥ $—` would be "≥ unknown", a contradiction — the common case for a
local-model-only run in the default cost view). Because `≥` is the single
lower-bound signal in the chart, `format_cost`'s redundant trailing `+` is
stripped from a marked cell (`≥ $0.700`, not `≥ $0.700+`).

**Labels (reviews D3/D4).** The daily title counts DAYS WITH USAGE
(`daily_series` skips idle days), so it says exactly that — `3 days with usage`,
singularized to `1 day with usage` — rather than "Last N days", which read as a
calendar window and misstated sparse usage.

The screen reads the series from the store on open (via the existing
`_open_analytics_worker`, off the event loop) and re-renders in place on `t` —
no second store read, since the toggle only changes which number the same data
plots. The toggle and its footer hint appear only when non-empty series are
loaded (no dead control on an empty store).

## 3. Testing

`tests/unit/analytics/test_store.py`: rollup upsert accumulation, local
day/month/year bucketing via injected timestamps, 365-day daily prune + 120-
month cap, by-model split vs aggregate, window selection, unpriced→floor,
lossless concurrent accumulate (two stores, same file), degradation, and
`series_totals`. `tests/unit/tui/test_analytics_panel.py`: chart labels/bars,
default-cost and tokens metrics, the `≥` floor mark, empty and no-series
states, and the `t` toggle driven in the real `OperatorApp`. Visual validation
per AGENTS.md: before/after SVGs from the real app (cost, tokens, empty),
rendered and inspected.
