# Design: Tokens per second — a model-agnostic generation-rate metric

Status: PROPOSED (design only; no code in this document has been written).
Audience: the implementer, the reviewer round, and the benchmark/QA round.

> **Scope note.** This document specifies one new measurement taken inside one
> existing wrapper, three new integer columns, and the surfaces that read them.
> It deliberately does NOT add a provider hook, a new recording path, a new
> store, or a new aggregation vocabulary. Where it declines to do something the
> brief sketched, it says so and gives the reason with the number behind it.

---

## 1. What was missing

`lop` accounts for tokens, dollars, cache, context and latency, but it has never
answered the question an operator asks while watching a long generation: **how
fast is this model actually producing output, and which model is fastest for
what I am doing.** A grep of the tree finds no `tok/s`, no
`tokens_per_second`, and no rate formatter on any surface.

Worse, the two columns that look like they should answer it do not:

- `duration_ms` is observed **stream-consumption** wall time, which includes
  TTFT, provider queueing, retry gaps and consumer backpressure. Dividing
  tokens by it measures a request, not a generation.
- `ttft_ms` is recorded only on `text_delta`/`tool_call_delta`
  (`model/configure.py:5710-5715`), so for a reasoning model it misses the
  entire thinking phase. Subtracting it from `duration_ms` therefore does not
  produce a "generation window" — §9 has the measured proof that the result is
  garbage, not merely imprecise.

The metric has to come from a window that is **measured**, not derived.

---

## 2. The metric, and its provenance

The operator's other harness, DeepSeek Harness, shows `315 tok/s` in its status
strip and per-message speeds from the `@deepseek-ai/dsh-session-stats` package.
Its fold rules, quoted from `lib/types/projection.js`:

```js
// projection.js:112-118
if (firstToken !== null) {
    next.ttftMs += Math.max(0, firstToken - open.startTime);
    if (!skipUsage) {
        const outputTokens = usageOutputTokens(event.data.usage);
        if (outputTokens !== null) {
            next.decodeMs += Math.max(0, event.time - firstToken);
            next.decodeTokens += outputTokens;
        }
    }
}
```

Three properties of that code are the design target, and all three are adopted:

1. **The window starts at the first output delta**, and first-token time
   survives an in-step `llm/retry` (the projection tracks `firstTokenTime` on
   the open step).
2. **The rate is `SUM(decodeTokens) / SUM(decodeMs)`** — a sum-weighted rate
   over a *population*, never a mean of per-call rates. The display is
   `stats.decodeTokens / (stats.decodeMs / 1000)`, formatted `{value} tok/s`.
3. **Numerator and denominator cover the same calls.** `decodeTokens` sums only
   over the steps that were decode-timed, so a call with no first token and a
   call with no usage report are excluded from *both* halves. The package also
   exposes `ttftSteps` — the coverage count — so a reader can tell how many
   steps contributed.

### 2.1 Where this design deviates, and why

**The window END is the last output delta, not the stream end.** DSH closes its
window at its own `assistant/message` step-end event. `lop`'s equivalent
boundary is the end of `_record_stream`'s loop, i.e. exactly `duration_ms`,
which is the column §9 shows cannot be used to close a generation window (it
carries the `finish` frame, the usage frame and any consumer backpressure). The
last output delta is the only end we can *measure inside the generation*
without relying on subtraction, so it is the one we take.

The trade is stated rather than hidden: this makes our window **narrower** than
DSH's, so our rate is a slight **over**-estimate relative to DSH's on the same
call — by the gap between the final delta and the step-end frame. That gap is
not decodable output, so the narrower window is the more defensible of the two;
but a reader comparing the two harnesses' numbers should know they are not
bit-identical definitions. This is recorded in the docstring, not just here.

---

## 3. The seam: one wrapper, every model

Measurement happens in `SessionStreamFn._record_stream`
(`local_operator/model/configure.py:5656-5809`), and nowhere else.

**Why that is general, not a coincidence.** `__call__` routes EVERY provider
call through this one wrapper before it reaches the caller:

- `model/configure.py:5531` — the isolated/errand path (compaction summary,
  auto-naming, `request.isolated`).
- `model/configure.py:5628` — the main path, whose inner iterator is
  `stream_with_failover(...)`.

`stream_with_failover` is itself the single funnel for every provider client, so
the wrapper sits *outside* credential rotation, model fallback, `request.replayable`
buffering and per-provider wire differences. It sees the `ChatRequest` and the
final `Usage` for turns, tool loops, compaction, naming, `aside`, and every
subagent at every depth, with no per-call wiring anywhere else. A provider
added tomorrow is measured the moment it streams, because it must go through
this iterator to reach a caller. **There is no provider-specific code in this
change**, and no provider client is touched.

### 3.1 Hot-path budget

Constraint: **at most ONE `time.monotonic()` per output delta, no allocation,
no lock, no I/O.**

**CORRECTED AFTER MEASUREMENT (benchmark round 1).** An earlier revision of this
section claimed the change *saves* work by collapsing the loop's two
`getattr(event, "type", "")` lookups into one. That claim is **false**, and the
benchmark falsified it: both of the old guards are `x is None and getattr(...)`,
so once `first_token_at` and `first_reasoning_at` are set they **short-circuit to
zero lookups per event** in the steady state, while the decode window must
classify *every* event to know which are output deltas. The change therefore
ADDS one event dispatch (one pydantic field read plus one `frozenset` membership
test) per event, on top of the clock read per output delta.

Measured, interleaved A/B on this host at load average 44-61, min of 600 samples
per point (`scripts/bench_tps_overhead.py`):

| N deltas | base ns/event | head ns/event | paired delta (3 rounds) |
|---|---|---|---|
| 50 | 3 567.5 | 3 662.5 | +169.2 / +80.8 / +246.7 |
| 500 | 3 289.8 | 3 515.7 | +354.7 / +101.8 / +225.8 |
| 5 000 | 3 782.1 | 3 998.1 | +366.2 / −68.8 / +445.9 |

best-of-600 slope, 500 → 5 000: **base 3 836.8, head 4 051.8, delta
+214.9 ns/event** — against a bare `time.monotonic()` of 37-43 ns (best) /
68-73 ns (median) on the same host at the same moment, i.e. 2.4-5.7× one clock
read, not the ~1× the earlier revision implied.

What the measurement DID confirm, structurally and exactly (counted, not timed):
`head` reads the clock **exactly once per output delta** (1.0006/drain at
N = 5 000; base 0.0008), and `head` performs exactly one type dispatch per event
(N/drain; base 3 per drain, i.e. only while its guards are still unset).

The honest statement of the cost is therefore: **one clock read plus one event
dispatch per output delta, about 0.2 µs/event in total** — ~1 ms on a 5 000-delta
turn, against a generation measured in seconds. That is not a significant
overhead for a turn, which is what the operator asked for; it is simply not the
*saving* the earlier revision claimed, and the code comment beside the loop says
the same thing as this table rather than the old claim.

Everything else in the budget holds: no allocation, no lock, no `await`, no I/O
in the delta branch (structural: the branch is four scalar locals and two
comparisons).

```python
event_type = getattr(event, "type", "")   # ONE dispatch per event (see above)
if event_type == "reasoning_delta":
    now = time.monotonic() if first_reasoning_at is None else None
    if now is not None:
        first_reasoning_at = now
    output_deltas += 1
    last_output_at = now or time.monotonic()
    if first_output_at is None:
        first_output_at = last_output_at
elif event_type in _OUTPUT_DELTA_TYPES:          # module-level frozenset
    if first_token_at is None:
        first_token_at = time.monotonic()        # unchanged: TTFT
    output_deltas += 1
    last_output_at = time.monotonic()            # ONE clock read
    if first_output_at is None:
        first_output_at = last_output_at
```

**Byte-for-byte transparency is preserved.** Nothing is added to, removed from,
or reordered in the yielded stream: the loop still ends in `yield event`, and
the four new locals are scalars that never reach the consumer. This is the
property `tests/unit/analytics/test_stream_recording.py::test_forwards_events_unchanged`
already pins; the implementer must extend that test, not relax it.

### 3.2 What the `finally:` block passes on

The eligibility decision is made **once, off the hot path**, in `_record_usage`,
because `output_tokens` is only known from `final_usage`:

```python
decode_us = (
    max(0, int((last_output_at - first_output_at) * 1_000_000))
    if first_output_at is not None and last_output_at is not None
    else 0
)
```

and then, in `_record_usage`:

```python
eligible = decode_us > 0 and output_deltas >= 2 and int(usage.output_tokens) > 0
decode_us     = decode_us if eligible else 0
decode_tokens = int(usage.output_tokens) if eligible else 0
decode_calls  = 1 if eligible else 0
```

`_record_usage` is already documented as "Off the hot path; never raises"
(`model/configure.py:5811`), and the new block lives inside its existing
`try/except Exception` — so **constraint 8 holds by construction**: a failure in
any new analytics code cannot raise into the stream loop. The three new
`CallSnapshot` fields default to `0`, so every existing constructor call and
every existing test keeps working unchanged.

### 3.3 Why `decode_us` is INTEGER microseconds

`decode_us` is an integer count of microseconds, not a REAL count of
milliseconds, and the choice is load-bearing rather than stylistic:

- `_aggregate_from_row` (`store.py:3480-3510`) coerces **every** measure with
  `int(values[idx] or 0)`. A REAL would be truncated on entry and then summed
  as an integer, so the value read back would not be the value written.
- `session_report` (`store.py:3076-3084`) documents that its three breakdowns
  come from ONE grouped scan merged by **integer addition over the finest
  grouping**, and that "Integer addition is what makes the merge exact". A REAL
  measure would make `SUM` order-dependent and the merge merely close, not
  exact — and `tests/unit/analytics/test_session_report_equivalence.py` pins the
  whole report key-for-key against an independently computed oracle, so
  "close" would fail it or, worse, be papered over by widening a tolerance.
- The rollup upserts are `x = x + excluded.x` (`store.py:583-598`). Integer
  accumulation is lossless and order-independent under the many-parallel-`lop`
  reality the store is built for; REAL accumulation is neither.

Microseconds rather than milliseconds because a fast model's whole window can be
under a millisecond and a millisecond-grained integer would round it to 0 or 1,
which is exactly the resolution a tok/s figure needs.

---

## 4. Schema and migration

### 4.1 `calls` — three columns via the supported registry

Appended to `_MIGRATION_COLUMNS` (`store.py:457`), which `_migrate`
(`store.py:1324`) turns into an idempotent `ALTER TABLE ADD COLUMN` per entry on
open — the documented, supported way to add a column to `calls`:

```python
("decode_us",     "INTEGER NOT NULL DEFAULT 0"),
("decode_tokens", "INTEGER NOT NULL DEFAULT 0"),
("decode_calls",  "INTEGER NOT NULL DEFAULT 0"),
```

**Why three and not one.** A rate needs a numerator and a denominator over the
*same* calls, plus an honest coverage count:

- `decode_us` — the summed window, the denominator.
- `decode_tokens` — `output_tokens` summed over **only the calls that
  contributed a window**. It cannot be derived from `SUM(output_tokens)`: a
  one-delta call has output tokens and no window, so using the all-calls total
  as the numerator would divide one population by another and inflate the rate.
  DSH avoids this the same way (`decodeTokens` sums only decode-timed steps).
- `decode_calls` — 1 per contributing call. This is the coverage count of §7.

`0` is the "no window" value, **not** `-1`. The `-1` sentinel is the convention
for the REAL *timing* columns (`duration_ms`, `ttft_ms`, `first_reasoning_ms`),
which are never summed and are read with conditional aggregates into
`TimingSummary`. These three are *measures*: they are summed by plain `SUM()`
and folded by integer addition, so a `-1` would be summed into the total and
corrupt it. `decode_us = 0` with `decode_calls = 0` means "unknown", exactly as
`cost_micro = 0` with `cost_known_calls = 0` means "unpriced" rather than
"free" (`model.py:431-436`). The count is what disambiguates; the sum alone
never is.

Also appended, in this order and at the end of `_CALL_COLUMNS` and of
`CallSnapshot`:

- `CallSnapshot.decode_us: int = 0`, `.decode_tokens: int = 0`,
  `.decode_calls: int = 0` — appended last because both are positional
  contracts (`store.py:419`, `_row_values` at `store.py:833`). The dataclass's
  own comment already states the rule: fields are chained rather than inserted
  "so the positional writer in `store.py` keeps its existing field order".

### 4.2 `session_daily` — the ONE rollup table, and its ALTER path

`session_daily` is the day-grain per-session rollup that `aggregate()` reads
instead of scanning the ledger (`store.py:~100`, measured 187 ms against the
ledger's 4,868 ms). It must carry the decode measures, because the headline
Totals, the By-provider table and the By-session table all come from
`aggregate()` and must keep telling **one** answer on both of its paths.

**`usage_daily` and `usage_monthly` are deliberately NOT changed.** They are
keyed `(day, model)` and `(month, model)`, they feed only the daily/monthly
*charts*, and no rate surface reads them (§10 has the per-model plan; it does
not go through them). Leaving them alone removes an entire class of risk — see
§4.3 — and is the smallest change that satisfies the constraints.

Three measures are appended to `_SESSION_DAILY_MEASURE_COLUMNS`
(`store.py:629`) **after** the component sums, and this one edit propagates
correctly to four places by construction, which is why the tuple exists:

- `_SESSION_DAILY_INSERT_COLUMNS` (`store.py:635`) — the upsert's columns.
- `_SESSION_DAILY_UPSERT_SQL` (`store.py:660`) — accumulates them with
  `x = x + excluded.x`.
- `_SESSION_DAILY_REDERIVE_SQL` (`store.py:769`) — re-derives them as
  `SUM(<col>)` from `calls`. **This is the property that matters**: a re-derived
  day (the rebucket path) and an accumulated day produce the same value, so the
  zone-change repair cannot make the rate disagree with itself.
- The shape guard's `set(_SESSION_DAILY_INSERT_COLUMNS) <= present` check.

Two hand-written projections must be updated in lockstep because they are
positional contracts with `_aggregate_from_row`:

- `_session_daily_rows` (`store.py:1049-1113`) — its `measures[...]` writes are
  literal indices `0..9` plus `10 + offset` for components, so the three new
  sums are written at `10 + len(COMPONENT_KEYS) + 0..2`.
- `_SESSION_DAILY_READ_SUMS` (`store.py:774`) — append `SUM(decode_us)`,
  `SUM(decode_tokens)`, `SUM(decode_calls)`.

**The migration itself.** A new `_SESSION_DAILY_MIGRATION_COLUMNS` registry, the
same `(name, definition)` shape as `_MIGRATION_COLUMNS`, plus a
`_migrate_session_daily(conn)` step called from `_migrate` **before** the
existing shape check (`store.py:1385-1401`). It reads
`PRAGMA table_info(session_daily)`, skips the table entirely when it does not
exist (a fresh DB creates it WITH the columns via `_SCHEMA`), and runs an
idempotent `ALTER TABLE session_daily ADD COLUMN ... NOT NULL DEFAULT 0` for
each missing entry. A failure is logged and swallowed, exactly as the `calls`
ALTERs are — "a failed add leaves the older shape", and the shape guard is what
turns that into a safe fallback.

**The guard has to grow, and the code says so.** The comment at
`store.py:1399-1404` states the superset check's one uncovered case: a column
present in `session_daily` but absent from `_SESSION_DAILY_INSERT_COLUMNS`,
`NOT NULL` and with no default, passes the check and then makes the upsert
raise — "Unreachable while rollup tables have no ALTER path (nothing can add a
column to one), and if one ever needs an ALTER this check has to grow the
`notnull`/`dflt_value` columns of `PRAGMA table_info` with it."

This change is the one that needs an ALTER, so it is the one that must grow the
check. Required: after the ALTERs, read `PRAGMA table_info(session_daily)` and
require that every column **not** in `_SESSION_DAILY_INSERT_COLUMNS` is either
nullable or carries a default. Only then may `_has_session_daily` be true. The
ALTERs we add ourselves are `NOT NULL DEFAULT 0` and so pass; a future release's
column without a default correctly reads as "no rollup", which is the fail-closed
answer.

### 4.3 The failure this avoids, stated plainly

`record_batch` writes `calls`, `usage_daily`, `usage_monthly` and (guarded)
`session_daily` in **one transaction** (`store.py:1560-1575`). If a rollup
upsert names a column the table does not have, SQLite raises, the `except
Exception` at `store.py:1590` rolls the whole attempt back and returns `0` —
**the ledger row is dropped with it**, silently, for the life of that binary on
that ledger. `session_daily` is protected by its shape guard; the calendar
rollups are protected only by the fact that nothing has ever added a column to
them. Not touching `usage_daily`/`usage_monthly` in this change keeps that
protection intact, and the section above is what gives `session_daily` the ALTER
path its guard was written in anticipation of.

### 4.4 Migration cost on the real ledger

Measured on the operator's ledger: `calls` = 1,948,473 rows / 658 MB (plus a
~528 MB WAL), while the rollups are tiny — `session_daily` 16,806 rows,
`usage_daily` 189, `usage_monthly` 51, `session_names` 13,856,
`tool_calls` 1,646,883.

`ALTER TABLE ... ADD COLUMN` with a **constant** default is a schema-only change
in SQLite: it rewrites the `sqlite_master` row, not the table. So the expected
cost of all three `calls` ALTERs plus the three `session_daily` ALTERs is
**O(1) in row count** — metadata work on the order of the `calls` ALTERs
`_migrate` already performs on every open. It does not touch the 658 MB table
and does not touch the 16,806 rollup rows either.

This expectation is stated **so the benchmark can falsify it**. The precedent
worth contrasting is `idx_calls_parent` (`store.py:488-520`), which cost a
one-time **~677 ms** stall and ~4.9 MB of growth on a 475k-row ledger — but that
is an *index build*, which reads and writes every row, and it is not the
operation this change performs. A benchmark that measures ~677 ms here is
measuring the wrong thing; the check is that the first open after migration
costs no more than the baseline `_migrate` (which already runs ~20
`PRAGMA table_info` reads and up to 14 `calls` ALTERs) plus noise.

**Forward-fill, not backfill — said out loud, as this repo says it for every
schema addition.** The three new `calls` columns read `0` on every row recorded
before this release, and `session_daily` gains no historical decode sums. Every
call in the existing 90-day ledger therefore has `decode_us = 0`,
`decode_tokens = 0`, `decode_calls = 0` — which is **unknown**, not a measured
zero, and §7 is the rule that keeps a report from rendering it as `0 tok/s`.
Historical decode rates are not recoverable: the window was never observed, and
§9 shows the one derivation that might have stood in for it is broken. The
wall rate in §8 covers existing history instead, and that is the deliberate
answer to "I want this on my ledger today".

---

## 5. Aggregation rules, and why weighted

`decode_tps = SUM(decode_tokens) / (SUM(decode_us) / 1e6)` over the scope's
eligible calls, and `None` when `SUM(decode_calls) == 0`.

Two rules make it correct:

**Weighted, never a mean of per-call rates.** DSH does `SUM/SUM` and so do we.
A mean of per-call rates weights a 20-token reply and a 20,000-token generation
equally, so a scope's headline would be dominated by its shortest calls. The
sum-weighted rate answers the question the operator is actually asking — "over
this window, how many output tokens per second of decoding did this model
produce" — and it is the only form that composes: the rate over a window equals
the rate computed from that window's summed columns, so a per-model row folds
into a per-provider row folds into the headline with no re-weighting step and no
second definition.

**Integer-exact throughout.** Every input to the rate is an additive integer
(`decode_us`, `decode_tokens`, `decode_calls`), so the finest-grouping merge in
`session_report` (`store.py:3076-3084`) and the `x = x + excluded.x` rollup
upserts are exact, and the division happens once at the very end on two summed
integers. There is no intermediate float anywhere in the storage or the merge.

### 5.1 Read plumbing (both paths, one answer)

- `UsageAggregate` gains `decode_us`, `decode_tokens`, `decode_calls` (default
  `0`) and the properties `decode_tps -> float | None` and
  `decode_coverage -> float | None` (= `decode_calls / calls`, `None` when
  `calls == 0`). `sum_aggregates` (`model.py:530`) adds the three counters, the
  same "SUM THE COUNTERS, NEVER THE BOOLEANS" discipline `cost_known_calls`
  already uses.
- `_aggregate_from_row` (`store.py:3480`) — the positional contract. The three
  measures are read at `10 + len(COMPONENT_KEYS) + 0..2`, i.e. **appended after
  the component sums**, so indices `0..9` and the component block keep their
  current meaning and no existing reader shifts.
- `_ledger_aggregate`'s `base_cols`/`component_sum` (`store.py:2480-2500`) — the
  three sums appended after `component_sum`. Because `decode_*` are in
  `_MIGRATION_COLUMNS` they are in `_OPTIONAL_COLUMN_NAMES`, so they take the
  same "substitute a constant 0 when this DB lacks the column" treatment
  `component_sum` and `cost_cols` already implement. A `rate_cols` expression
  beside `cost_cols` is the shape: absent columns read as unknown, the query
  does not fail, and the report shows `—` rather than dying on a missing column.
- `session_report`'s `sums` tuple (`store.py:2952`) — append
  `SUM(col("decode_us"))`, `SUM(col("decode_tokens"))`, `SUM(col("decode_calls"))`
  using the existing `col()` helper (`store.py:2947`) so a pre-column ledger substitutes `0`.
  Because these ride the SAME grouped scan that already feeds `by_model`,
  `by_purpose` and `by_model_outcome`, **`SessionReport.by_model` inherits the
  new fields for free** — one indexed scan, no new read, and the fold stays
  integer-exact.
- `_assemble_aggregate` needs no change: it feeds `_aggregate_from_row`, which
  is where the new positions are read. That is deliberate — the store's
  "TWO PATHS, ONE ANSWER" property is preserved because there is still exactly
  one row-to-result translation.

---

## 6. The two rates, and why they are different types

### 6.1 `decode_tps` — the primary, DSH-equivalent metric

`SUM(decode_tokens) / SUM(decode_us)`, over calls with a measured window. This
is the number the feature exists for. It is **forward-fill only**: only calls
recorded after this release have a window.

### 6.2 `wall_tps` — the honest secondary, available on all existing history

`SUM(output_tokens) / SUM(duration_ms)` over calls where `duration_ms > 0` and
`output_tokens > 0` — both **already-stored** columns, so it works on the
operator's entire existing 90-day ledger with no migration and no new capture.

It is real, useful and sane, and it is measured rather than argued. On the
operator's ledger, 30-day window, **1,527,889 eligible calls**:
p1 = 6, p10 = 31, p50 = 92, p90 = 187, p99 = 249 tok/s (mean 104). Per-model
for calls with `output > 500`: `deepseek/deepseek-flash` 164 tok/s (n = 460,267),
`anthropic/claude-opus-5` 77 (n = 80,322), `deepseek-v4.1-flash` 110,
`radient/auto` 200, `kimi-k3` 51.

**It is a WALL rate and must never be labelled decode speed.** It includes TTFT,
provider queueing and any consumer backpressure. A model with a 4-second
first-token wait and a fast decode shows a low `wall_tps` and a high
`decode_tps`, and that difference is the diagnosis, not a defect.

**The two rates live on different types, and that is the mechanism that keeps
them apart.** `wall_tps` is a property of `ModelRateRow` only; `UsageAggregate`
carries no wall field at all. There is therefore no object on which a surface
could pick the wrong one by reading a similarly-named attribute, and no scope in
which a wall figure can silently inherit a decode label. The field names differ
(`decode_tokens`/`decode_us` vs `wall_tokens`/`wall_us`) at every layer, in
Python and on the wire.

Why not put `wall_us` on `UsageAggregate` too, so the existing tables could show
it? Because that would require accumulating it, which would require a column,
which would make it forward-fill — destroying the one property that makes it
worth having (§8's rejected-alternative list records this). Its cost is instead
borne by an explicit ledger-only read, which is honest about what it is.

---

## 7. Coverage, and the unknown-never-zero rule

**A rate is never presented as if it covered every call.** Every aggregate
carries `decode_calls`; every rate is `None` — rendered `—`, never `0` — when
`decode_calls == 0`; and every section that shows a rate states its coverage
when coverage is partial.

This is the repo's existing doctrine applied to a new number, not a new
invention:

- `TimingSummary` (`model.py:650`) counts its samples and reports
  `mean_ms = None` rather than a fabricated zero.
- `cost_is_partial` / `format_cost` render `$12.30+` rather than presenting a
  partial sum as complete (`analytics_panel.py:144-182`).
- `cache_hit_rate` returns `None` and `format_percent(None)` prints `—`
  (`analytics_panel.py:128-140`).
- `usage_reported` distinguishes "the provider told us nothing" from "the
  provider told us zero".

The renderings:

- **Section legend, not per row.** Per-row coverage would cost a column on
  every table for a number that is uniform across the table. The panel already
  has this mechanism for cost: `scope_needs_cost_legend`
  (`analytics_panel.py:209`) and `_needs_cost_legend` (`:220`), with the shared
  `COST_LEGEND` text drawn once at the body's foot (`:1495-1499`) — the same
  "draw it only when a mark actually appears" rule. The rate equivalent is
  `scope_needs_rate_legend(scope)` and `_needs_rate_legend(...)` over a
  `RATE_LEGEND` constant → a single dim line under the section when
  `decode_calls < calls`, e.g.
  `decode rate over 1,240 of 1,530 calls · wall rate covers all`.
- **The empty state is explicit.** `decode_calls == 0` over a whole table
  renders `—` in every rate cell and one line saying why: a pre-release ledger
  has no windows, so the rate is *unknown*, and the section says so rather than
  drawing a row of zeros.
- **The 2-delta guard is part of this, not just a correctness filter.** §9
  shows a whole population of calls whose implied window is meaningless; they
  are excluded from both halves of the rate AND counted as excluded, so the
  coverage number is the honest report of how much of the ledger the rate
  actually speaks for.

---

## 8. Rejected: `duration_ms - ttft_ms` as the decode window

Recorded with the measurement, because it is the obvious shortcut and it is
wrong.

On the same 30-day window of the operator's ledger, the implied rate
`output_tokens / ((duration_ms - ttft_ms) / 1000)` gives p50 = **534 tok/s**,
p90 = **2,716**, p99 = **41,587**, max = **10,211,055**, with **10.2% of calls
above 2,000 tok/s**. Two independent causes, both measured:

**(a) `ttft_ms` misses the reasoning phase.** It is recorded only on
`text_delta`/`tool_call_delta` (`model/configure.py:5710-5715`), so for a
reasoning model it measures the wait to the first *visible* token, not the first
token produced. Provenance over 1,527,783 calls: `only_text` = 1,240,887,
`both` = 286,644, `only_reasoning` = 252. The `first_reasoning_ms` column exists
in the same wrapper precisely because these are two different waits — and it
currently has **no reader anywhere in the tree.** This change is its first
consumer, which is a small incidental payoff worth naming: the decode window's
start is the first delta of *any* output type, so a reasoning-only opening is
timed correctly for the first time.

**(b) Single/late-blob responses.** The "window" collapses to the trailing
chunk. For calls with `output > 500` whose implied rate exceeds 2,000 tok/s —
**18.1% of 644,267 decodable calls** — `window / duration` has p50 = 0.041 and
p90 = 0.074, and **62.8% are below 0.05**: the whole answer lands in the last
few percent of the call. Overall `window / duration` p50 = 0.292.

**Consequence for this design: the window MUST be measured explicitly at the
seam, and the ≥2-output-delta guard is load-bearing rather than decorative.**
Cause (b) is exactly the population the guard excludes: a one-delta call has
`window ≈ 0`, so it would produce an absurd rate if admitted, and a mean over
such calls is how a headline reaches four figures. Those calls are **excluded
and counted**, never silently averaged in — which is why `decode_calls` exists
as a separate measure rather than being inferred from `calls`.

The wall rate (§6.2) is **not** vulnerable to (b): a long call that emits its
answer in one blob genuinely did take that long, so `output/duration` is merely
a low wall rate, which is the truth about that call.

---

## 9. Surfaces

### 9.1 TUI `/analytics`

Free wins first, because `aggregate()` already carries the new fields:

- **Totals** — one headline `decode_tps` line plus the coverage legend.
- **By provider** (`_group_section`, `analytics_panel.py:2122`) — a new
  `tok/s` column immediately after `tokens`, reading `agg.decode_tps`.
- **By session** (`_session_section`, `analytics_panel.py:1937`) — the same
  column, same position, so the two tables still read as one table.
- **NEW "By model" section** — from `model_rates()` (§9.4), showing both rates.

Column mechanics, because the module documents a repeat-offender defect here
(`analytics_panel.py:1553-1603`: "no column width is assumed from a literal
where the data can size it"):
`_row_overhead` and `_group_section`/`_session_section` must be changed
**together**. Add `_tps_col(groups) -> int` = `max(3, max(len(format_tps(agg.decode_tps)) ...))`
and add `3 + _tps_col(groups) + len(" tok/s")` to `_row_overhead`, mirroring the
existing `tokens`/`calls` terms. `ReportLayout` gains `tps_col: int = 0` so a
single-row hover repaint recomposes against the table's column rather than its
own.

Rendering: `f"{format_tps(agg.decode_tps):>{tps_col}} tok/s"`. `format_tps` is
total over its domain, returns `"—"` for `None` (1 cell) and a bounded string
otherwise, so the pad cannot be overrun by data — the same argument
`format_percent`'s `4` rests on.

### 9.2 TUI `/session`

`SessionReport.by_model` already exists and inherits `decode_tps` at no extra
cost (§5.1), so the by-model table in `session_panel.py` gains the column from
the read it already performs. `_group_rows` (`session_panel.py:447`) and
`_measure_columns` (`session_panel.py:986`) must both learn the new cell, since
that screen measures one shared column set across all its proportional sections.

The wall rate for a session comes from `model_rates(session_id=...)` — the same
method, scoped — so `/session` has one vocabulary with `/analytics` rather than
a second spelling.

`SessionReport.timings` already carries `duration_ms` (`TimingSummary`), so the
screen can state the session's wall-vs-decode gap without a new field.

### 9.3 Status band — ASSESSED AND RECOMMENDED FOR DEFERRAL

**A live per-session rate cannot reach the band cheaply and correctly.**
Recommend deferring it, with the reason on the record.

The band is push-based: `StatusLine.update(**segments)`
(`status_line.py:1449`) with a `_DROP_LADDER` of ~12 rungs
(`status_line.py:334`) and one width model. Adding a segment costs a rung, a
render term and a layout test — that part is ordinary. The blocker is the
**data**, and it is structural, not a matter of effort:

1. **The TUI process never sees the deltas.** A grep for `text_delta` across
   `local_operator/tui/` returns nothing; the transcript reaches the UI through
   frontend-state/trajectory relays, not through the provider stream.
2. **The only usage the band reads is per-call and retrospective.**
   `FrontendState.last_usage` is stamped when a call's usage arrives
   (`frontend_state.py:6216`, `:6237`, `:6542`, `:6563`), and `Usage` carries no
   start time and no window — a single `at_ms` stamp only
   (`harness/types.py:509`). During a stream, `last_usage` describes the
   *previous completed* call, so a "live" figure built from it would be stale by
   one call and would not move while the user watches.
3. **Both routes to a real live rate are expensive.** Either new per-frame
   fields cross the attach boundary, or the TUI process measures deltas itself.
   The frame budget is documented as *elastic and already over-subscribed*: the
   worst-case frame measured **1,048,408 of 1,048,576 bytes under the pre-goal
   shape — the slack is NOT 168 bytes** (that figure is the goal-less frame; the
   judged-goal record costs 95 B and the arms absorb it differently, so the live
   number is **under ~110 B**, re-derived by capturing the guard's member arm at
   the head under review rather than quoted forward — two independent measurements
   disagreed with 168 B and with each other, which is why this says a bound and a
   method rather than a number) — and `JOB_TEXT_FRAME_BUDGET_CHARS` was cut
   120,000 → 119,872 → 119,360 to *pay* for two per-frame fields, with the
   comment stating "the next per-frame field is paid for out of here too"
   (`frontend_state.py:205-236`). Teaching the TUI to observe deltas is a far
   larger change than this feature, touching the relay path the frame budget
   exists to protect.
4. **A per-paint DB read is ruled out by the band's own contract**: it is
   push-based, deliberately never reaches into the session, and must not do I/O
   on a repaint.

**Recommended disposition:** defer. When it is built, the honest cheap version
is a **per-completed-call** figure — "the last call decoded at N tok/s" — which
needs only the call's own `decode_us`/`decode_tokens` carried on the existing
usage relay, is not a live rate, and must be labelled as the last call rather
than as the session. That is a strictly smaller change than a live counter and
it is the one to price first; it still needs a frame-budget decision, which is
why it is a follow-up rather than part of this change.

**That frame-budget decision, priced — and the first pricing was WRONG (2026-09-26).**
The first version of this paragraph put the fields on `FrontendUsage` and claimed
they cost the frame "~46 B, once, because `FrontendUsage` appears exactly once".
Review round 1 measured the claim and refuted it: `FrontendUsage` reaches the wire
up to **201 times per frame**, so the shape is exactly the one that broke attach
before.

`FrontendState.last_usage` is one slot, but `FrontendState.usage_components` is a
**`list[FrontendUsage]`** (`frontend_state.py:2679`) capped at
`USAGE_COMPONENT_CAP = 200` (`:156`) and serialised in full (`:3248-3250`, `:5787`);
the calibrated fixture itself fills it with 5,000 receipts, which the wire caps at
200. Measured through `sync_wire_payload` with both keys set on every
`FrontendUsage`, using the guard's own `_receipt` shape: 1 receipt 2,061 → 2,157 B
(+96); 200 receipts 70,318 → 79,966 B (**+9,648**); 5,000 receipts (wire-caps to
200) 70,518 → 80,166 B (**+9,648**). That is 48 B x 201 objects against 168 B of
slack — **57x over** on the pre-goal slack and still ~40x on the ~110 B the live
frame actually leaves, taking the calibrated worst case from 1,048,408 to about
**1,058,056 B, some 9.5 KB past the 1 MiB line**. The payload carries 201
`"decode_us"` occurrences.

Two details worth keeping, because they are where the wrong pricing came from:
the ROSTER is not the multiplier (`jobs[i].usage` and `descendant_usage[j]` are
declared `Usage` with the default extra policy, so a field declared on the
subclass is dropped from those slots — verified on a 3-row frame: ten
usage-shaped dicts, the keys only under `$.snapshot.last_usage`), and the guard's
200-row fixture is exactly where the multiplier lands, so an implementation would
go red rather than ship a broken attach. The harm of the wrong pricing was a
follow-up that could not be built, not a broken frame.

**A second route that does NOT fit, measured the same way (2026-09-26, later the
same day).** The obvious hook — declare the pair on the base `Usage` and have the
model seam stamp the object it already relays — was implemented and then
measured, and it costs the frame **once per roster row**. Two reasons compound:
the field is in the SCHEMA of every usage on the wire, so nothing drops it from
`jobs[i].usage` the way a `FrontendUsage`-only field was dropped; and the relay
assigns the stamped object to the assistant message AFTER the stream loop
(`loop.py:2641` → `:2835` → `subagent.py:1459` → `:1537`), so the window lands on
the very object `_accumulate_usage` copies into each job row. Measured through
`sync_wire_payload`: 1 row +39 B (2 occurrences), 20 rows +780 B (21), **200 rows
+7,800 B (201 occurrences)** — ~46x over the 168 B of slack, i.e. the recorded
attach failure at `frontend_state.py:132-140`. The omission rule cannot help,
because an ELIGIBLE call's pair is set by definition: it drops only the unset
pair, which is the one that was never a cost.

The lesson is worth more than the route: `test_attach_frame_size.py` stayed GREEN
through that implementation (123 passed) because its 200 rows are built from an
UNSTAMPED `Usage` — the guard measures the without-column and never the
with-column. So the follow-up's first test is not "add the fields and see if the
guard notices"; it is an occurrence assertion **with stamped rows**, added to that
file, which is the only thing that makes any of the numbers above a guard rather
than a measurement someone has to remember.

**A third shape, and the first whose frame cost is ZERO (2026-07-26, measured at the
mechanism level rather than through a frame).** Both falsified routes shared one
assumption — that the numbers must live in a PYDANTIC FIELD to travel from the seam
to the frame, which is what put them in the serialised schema and so on every row
that shares that schema. They do not have to. A ``Usage`` accepts an UNDECLARED
private attribute (``usage._decode_window = (us, tokens)``) and pydantic keeps it
out of ``model_dump`` entirely: measured on this tree, the assignment succeeds and
a dump of the same object carries **no key for it at all**. So the seam can hand the
window forward through the relay for **0 wire bytes**, and the only public field
pair in the design is the one the status band reads.

Put together, the shape is: the seam stamps a PRIVATE pair early, where the usage
event is handled rather than in the stream's ``finally`` (the ordering finding —
``on_usage`` fires while the seam is suspended at ``yield``); the session copies
that private pair onto ``FrontendUsage`` — the wire subclass — when it builds the
frame's usage; and the pair is declared ONLY there, because ``jobs[i].usage`` and
``descendant_usage[j]`` are declared ``Usage`` and therefore drop a
``FrontendUsage``-only field (review round 1 verified that drop directly). Net
frame cost: two integers on ``last_usage``, one occurrence, ~48 B against the 168 B
of slack — and **nothing at all** on the 200 roster rows, which is the multiplier
that made the first two routes cost 9,648 B and 7,800 B.

Two spelling traps, both measured (review round 2), because the natural first
attempt at each fails:

* the wrap cannot be ``FrontendUsage.model_validate(usage)`` — that raises
  ``ValidationError``. ``last_usage`` is built by ``_usage_wire``
  (``frontend_state.py:2804-2806``), which DUMPS the live ``Usage``, so the pair has
  to be injected into the dumped dict after reading the private attribute off the
  object;
* and it must be materialised on the ``last_usage`` VALUE alone, never on an object
  also fed to ``usage_components``. That list is ``list[FrontendUsage]`` — 200
  ``FrontendUsage`` slots, NOT ``Usage``-typed ones — so a pair set on a receipt
  WOULD serialise; the receipts are safe only because they are dumped from the live
  ``Usage``. ``accrue_usage`` builds ``last_usage`` and ``usage_components`` in one
  call (``:6216-6247``), which is what makes this a one-line mistake.

Measured on the guard's own roster shape through the real ``sync_wire_payload``
(review round 2): 200 rows plain with a plain last_usage is 186,657 B and **0**
occurrences; putting the private pair on ALL 201 objects is still 186,657 B —
**+0 B exactly** — and declaring the pair on the ``last_usage`` value alone is
186,696 B, **1 occurrence**, +39 B at a two-digit window and 48 B at the 7-digit
worst case. Stamping EARLY also closes the ordering finding the reverted attempt
left open: with the pair set before ``on_usage`` fires, the consumer's dump and
``accrue_usage``'s injection can both read it, so the band's source is populated on
the live path rather than only on refresh.

That is still a hypothesis with a measured mechanism rather than a measured frame in
the SHAPE the band would ship, and the difference is exactly what has been wrong
three times in this section. So the first
thing the next attempt does is still the SAME test: an occurrence assertion with
stamped rows in ``tests/unit/session/test_attach_frame_size.py`` — a file that
builds its 200 rows from an unstamped ``Usage`` and therefore stayed green through
every wrong implementation above. Only after that assertion exists and counts
one is this shape worth building on.

**The re-priced form, which fits.** Put the two scalars on
**`FrontendSessionState`** rather than on the per-receipt type: one occurrence per
frame, ~48 B against 168 B of slack. That is also the HONEST shape rather than a
workaround — the band shows the last completed call only, so a per-receipt field
would be 200 copies of a fact 199 of which nothing reads. Two alternatives were
priced and are worse: a separate receipt type (a new wire type for a figure the
band reads once), and re-budgeting ~9.6 KB out of the roster's text budgets, which
is a much larger decision than this feature.

What remains open is the labelling rather than the cost: a bare `N tok/s` beside a
live context reading is the misreading §9.3 exists to prevent, so the segment has
to say it is the last completed call, and that wording is a design-round question.
The follow-up is therefore: two scalars on `FrontendSessionState`, the seam
stamping them (`_record_usage` has both numbers in hand), one `StatusLine` segment
plus its drop-ladder rung — a figure that is NOT re-derivable from the transcript
belongs near `cost`, which the ladder's own comment calls "the one figure that is
not a live reading of this turn" — and two tests: the frame delta pinned as an
assertion in `test_attach_frame_size.py` (the file that would have caught the
first pricing), and the band's rendering plus shed order.

### 9.4 Desktop endpoint

**One new op, and no change to the cost of the existing one.**

`analytics.get` is unchanged in shape and in cost. Its `aggregate` payload gains
`decode_us`, `decode_tokens`, `decode_calls` for free, as ordinary new fields on
`dataclasses.asdict(UsageAggregate)` — so the UI's existing By-provider and
By-session tables get a rate column with **no new request and no added
latency**. That is the single most important property for the benchmark.

New op for the per-model table, because it must NOT ride `analytics.get`:

```
GET /v1/desktop/analytics/models?since_ms&until_ms&session_id&days
→ CRUDResponse[{ "data": { "rows": ..., "scope": "ledger",
                           "since_ms": ..., "until_ms": ... } }]
```

Registered beside the existing route in
`server/routes/desktop_catalogues.py:253`, same `AnalyticsStore` construction,
same `asyncio.to_thread(read_report)` off-loop discipline, same 422 on
`since_ms > until_ms`. The rows are a plain list of dataclass dumps, so no
tuple-keyed `asdict` problem arises (unlike `by_model`/`by_purpose_outcome`,
which the session-report route has to rebuild as arrays —
`desktop_catalogues.py:432-470`).

**Why a separate op rather than `by_model` on `analytics.get`:** the per-model
rows come from the raw ledger (§9.5), which is a scan of 1.95 M rows. Putting it
on `analytics.get` would add that scan's cost to every analytics panel load,
including the ones that never scroll to the table. As its own op it is fetched
lazily, once per window change, and its cost is isolated and separately
benchmarkable.

### 9.5 Where the window per-model rows come from — and the consistency argument

**They come from the raw ledger, in one grouped scan, through a new
`AnalyticsStore.model_rates(...)`.**

```sql
SELECT provider, model_id, COUNT(*), SUM(output_tokens),
       SUM(decode_us), SUM(decode_tokens), SUM(decode_calls),
       SUM(CASE WHEN duration_ms > 0 AND output_tokens > 0
                THEN CAST(ROUND(duration_ms * 1000) AS INTEGER) ELSE 0 END),
       SUM(CASE WHEN duration_ms > 0 AND output_tokens > 0 THEN output_tokens ELSE 0 END),
       SUM(CASE WHEN duration_ms > 0 AND output_tokens > 0 THEN 1 ELSE 0 END)
FROM calls [WHERE ts_ms >= ? AND ts_ms < ? [AND session_id = ?]]
GROUP BY provider, model_id
ORDER BY SUM(output_tokens) DESC, provider, model_id
LIMIT ?
```

with the `col()`-style substitution of a constant `0` for any absent optional
column, so a pre-migration ledger answers rather than failing. It never raises:
an unopenable store returns `[]`.

**Why not `usage_daily`.** It IS keyed `(day, model)` and would be nearly free
(189 rows). It is rejected for two reasons. First, it is **forward-fill**: it
covers only days since the rollup shipped, so the table would be empty or
partial beside a headline Totals figure that *is* backfilled by the
`session_daily` sweep — the exact inconsistency the manager flagged, and the
worst possible first impression on the operator's own ledger. Second, it would
add no information the ledger scan does not already have, while making the
feature's usefulness depend on how long the rollup has been running.

**Why not `session_daily`.** It is keyed `(day, session_id, provider)` and its
schema comment states the rule outright (`store.py:324-327`): `model_id` is
deliberately not in the key, "a key dimension no read consumes cannot be removed
later without a table rebuild". Adding model to that key would multiply the
16,806 rows by the model count and invalidate the existing rollup wholesale, for
one new table. Rejected.

**Why not widen `aggregate()`'s per-provider `GROUP BY` to `(provider,
model_id)`.** This was the most attractive alternative and it is rejected on
cost and correctness grounds. The store already measures what a second GROUP BY
column costs: `_PARENT_EDGE_SQL`'s comment records "610 ms (vs 290) for widening
the GROUP BY to two columns, which forces a temp B-tree for identical output"
on a 475k-row ledger. More importantly, on the **rollup** path there is no model
dimension to widen — so the per-model rows would come from a different source
than the headline, which is precisely what the store refuses to do
(`_assemble_aggregate`'s WINDOW RULE: "any other rule makes the per-session
column stop summing to the headline total, which is the invariant this whole
re-partition exists to protect").

**The consistency argument, stated on the screen.** Because the per-model rows
cannot partition the headline without a model dimension in the rollup, the
section does not claim to. It is a **separately-labelled table with its own
meta line** naming its source and its window —
`By model · last 30 days · from the ledger` — and it is the *only* rate table
whose scope is the ledger rather than the rollup. Concretely:

- It **does** cover the same window the headline covers, because both are
  windowed by `ts_ms` over the same bounds. So the biggest model's share of
  output tokens is comparable to the headline's output-token figure, and a
  reader can check one against the other.
- Its per-model `decode_calls` / `wall_calls` coverage counts are shown, so a
  reader sees how much of each row's calls actually carry a decode window
  (all of them, forward-fill permitting, for wall).
- The **daily chart** on the same screen still reads `usage_daily`, and the
  **Totals** still read `aggregate()`. This change does not touch either, so
  neither can be moved by it; the new table is additive and is labelled as
  coming from a different source.

**Cost, and how it is bounded.** One grouped scan of the ledger. `idx_calls_ts`
serves the window predicate; `GROUP BY provider, model_id` forces a temp B-tree,
and no covering index is proposed (a non-covering index range scan is exactly
what made the ledger path slow — the 4,868 ms figure — so an index here would
cost a large one-time build and buy little). Expect seconds, not milliseconds,
on the operator's 1.95 M-row ledger; the benchmark must measure it on a copy and
the number belongs in this document when it exists. Mitigations, all already in
the design: a separate op, a worker thread, lazy fetch, and a result cached for
the screen's lifetime.

---

## 10. Non-goals

- **No live/streaming rate in the status band.** Deferred, with the reason in
  §9.3.
- **No visible-text (`generation_tokens`) decode rate.** The primary rate is
  `output_tokens / decode_us`, which includes reasoning — the DSH-equivalent and
  the honest "how fast does this model generate" number, since `reasoning_tokens`
  is a documented SUBSET of `output_tokens` (`harness/types.py:472-478`). The
  visible-text variant (`output − reasoning`) would need a **fourth** correlated
  measure (`decode_reasoning_tokens`) so that numerator and denominator stay on
  one population; that is real schema surface for a secondary reading, so it is
  a named follow-up rather than part of this change.
- **No per-day or per-month rate chart.** `usage_daily`/`usage_monthly` are
  untouched (§4.2); a decode rate over calendar buckets is a follow-up that
  would ride the same three columns once those tables accumulate them.
- **No `ttft_ms` or `first_reasoning_ms` changes.** Both keep their names,
  meanings and sentinels so historical comparisons survive. `first_reasoning_ms`
  merely gains its first reader (§8).
- **No provider-specific anything.** No provider client, adapter or registry
  entry is touched.
- **No backfill.** §4.4.
- **No UI-repo change in this PR.** The mirror in `local-operator-ui` is a
  separate repo and a separate change (§12 lists it as a hand-off, not a task).

---

## 11. Rejected alternatives

| Alternative | Rejected because |
|---|---|
| Derive the window as `duration_ms - ttft_ms` | Measured broken: p90 = 2,716 tok/s, p99 = 41,587, max = 10.2 M, 10.2% of calls above 2,000. Two causes, both measured (§8). |
| Close the window at the stream end (DSH's exact boundary) | The stream end *is* `duration_ms`, which carries the finish/usage frames and consumer backpressure. Taking the last output delta measures inside the generation; the deviation is documented in §2.1 rather than glossed. |
| Store `decode_us` as REAL milliseconds | `_aggregate_from_row` coerces with `int()`, the finest-grouping merge is exact only over additive integers, and the rollup upsert is `x = x + excluded.x`. A REAL measure is truncated on read and order-dependent on merge (§3.3). |
| Store `-1` as the "no window" sentinel, matching the timing columns | The timing columns are never summed; these are. A `-1` would be summed into the total. `0` + a coverage count is the `cost_micro`/`cost_known_calls` precedent (§4.1). |
| Use `SUM(output_tokens)` as the rate's numerator | Divides one population (all calls) by another (decode-timed calls) and inflates the rate. `decode_tokens` is a separate measure for exactly this reason; DSH does the same (§2, §4.1). |
| Report the mean of per-call rates | Not composable and dominated by the shortest calls. DSH sums; so do we (§5). |
| Add `wall_us` to `UsageAggregate` so the existing tables can show a wall rate | Requires accumulating it, which requires a column, which makes it forward-fill — destroying the only property that makes it worth having (works on the existing ledger today). Wall stays on `ModelRateRow` (§6.2). |
| Put `by_model` on `analytics.get` | Adds a 1.95 M-row ledger scan to every analytics load, including loads that never open the table. Separate op, lazily fetched (§9.4). |
| Serve the window per-model rows from `usage_daily` | Forward-fill only: on the operator's ledger it would render empty or partial beside a backfilled headline. Rejected as the first impression on the very ledger this is for (§9.5). |
| Add `model_id` to `session_daily`'s key | Multiplies 16,806 rows by the model count and invalidates the existing rollup, for one new table. The schema comment already forbids it (§9.5). |
| Widen `aggregate()`'s `GROUP BY provider` to `(provider, model_id)` | Measured +320 ms on 475k rows for the temp B-tree, and it still cannot serve the rollup path, which has no model dimension — so per-model rows would come from a different source than the headline (§9.5). |
| Add the decode measures to `usage_daily`/`usage_monthly` too | Nothing reads them (§4.2), and touching them would require an ALTER path plus a new shape guard on tables that currently have neither — new risk for no read (§4.3). |
| A new analytics store / recorder hook / per-provider instrumentation | The existing wrapper already sees every provider call for every model; a second path is a second thing to keep in sync and a new double-count risk (§3). |
| A live rate in the status band | Cannot be done cheaply and correctly: the TUI never sees deltas, `last_usage` is per-call and carries no window, the attach frame has 168 bytes of headroom at its worst case, and the band must not do I/O on a repaint (§9.3). A per-completed-call figure IS affordable, but only on `FrontendSessionState` — the same fields on `FrontendUsage` cost 48 B x 201 objects and miss the frame by ~9.5 KB (§9.3, re-priced after review round 1 refuted the first estimate). |

---

## 12. Testing, benchmark and visual validation

### 12.1 Unit tests

`tests/unit/analytics/test_stream_recording.py` (extend the existing seam tests;
`_drain` already drives `_record_stream` with a synthetic async iterator, so
**no network call is needed**):

- transparency is unchanged — `test_forwards_events_unchanged` still passes
  with the new locals present;
- a `text_delta` × N stream yields `decode_us > 0`, `decode_calls == 1`,
  `decode_tokens == output_tokens`;
- a **reasoning-only** stream (no text delta) still measures a window, and
  `ttft_ms == -1` while `decode_calls == 1` — the §8(a) case, and the first
  test of `first_reasoning_ms`'s consumer;
- a **single-delta** call yields `decode_calls == 0` and `decode_us == 0` — the
  §8(b) guard, asserted as an exclusion rather than a rounding;
- a call with `output_tokens == 0` yields `decode_calls == 0`;
- a stream that raises mid-way still records, with `ok == 0` and whatever
  window was measured, and does not re-raise from the analytics block;
- a failure injected into the new analytics code does not reach the caller.

`tests/unit/analytics/test_model.py`: `decode_tps` is `None` at zero coverage;
it is sum-weighted (a fixture whose per-call rates differ, asserting the value
is `SUM/SUM` and **not** the mean); `sum_aggregates` adds the three counters;
`decode_coverage` arithmetic.

`tests/unit/analytics/test_store.py`: the `session_daily` ALTER path on a
**pre-column database** (build one with the old schema, open it, assert the
columns appear and `_has_session_daily` is true); the same database with the
ALTER made to fail reads as no-rollup and still records to `calls`; the grown
shape guard rejects a table carrying an extra `NOT NULL` column with no default;
`model_rates` grouping, `wall_*` predicate, `col()` substitution on a
pre-column ledger, ordering, `LIMIT`, and the never-raises contract;
`decode_tps` identical between `aggregate()`'s rollup and ledger paths
(`dataclasses.asdict` equality — the existing two-paths-one-answer discipline);
the rebucket path producing the same decode sums as the accumulated path.

`tests/unit/analytics/test_session_report_equivalence.py` (**the frozen oracle**
— see Risk 4): its hand-written `sums` tuple and its independently computed
report must both gain the three measures, or the whole-report comparison fails.
This is an expected, required edit, not a workaround.

`tests/unit/analytics/test_session_daily_rollup.py`: the insert-plan guard, the
in-sync gate, and `_last_aggregate_refusal` still naming the right reason.

`tests/unit/server/` (route tests beside the existing `desktop_catalogues`
ones): the new op's envelope, param forwarding, the 422, `session_id` scoping,
and an empty store returning `{"rows": []}` rather than an error.

`tests/unit/tui/test_analytics_panel.py` and the `session_panel` tests:
`format_tps` totality (`None` → `—`); the `tok/s` column present in both tables
and absent when `decode_calls == 0`; `_row_overhead` and the painted rows
agreeing at the widths `_WIDE_TABLE_MIN` straddles (the D8/D11 class of defect —
assert the row width against the content box, not just the presence of a
string); the rate legend appearing only when coverage is partial; the By-model
section's empty state.

### 12.2 Benchmark — what to time, and that it needs no network

**The function to time is `SessionStreamFn._record_stream`.** It can be driven
with no network call: `tests/unit/analytics/test_stream_recording.py::_drain`
already constructs it via `object.__new__(SessionStreamFn)` and feeds a
synthetic async iterator, and the recorder can be pointed at an in-memory or
temporary store. The benchmark harness should follow that shape exactly.

Two arms, same interpreter, same synthetic stream, recorded in one session:

1. **Hot path.** Time a drain of a synthetic stream of N output deltas
   (N ≈ 50, 500, 5,000; mixed `text_delta`/`reasoning_delta`/`tool_call_delta`)
   before and after. The claim to falsify is "at most one `time.monotonic()` per
   delta and no allocation": report the per-delta delta in nanoseconds against
   the measured cost of a bare `time.monotonic()` on the same host. A regression
   larger than that bound means the loop allocated or took an extra clock read.
2. **Migration.** Time the first open of a **copy** of the operator's real
   ledger through the migration (before/after), and the second open. The claim
   is O(1) in row count (§4.4): the first open should not exceed the second plus
   noise, and neither should resemble the ~677 ms index-build precedent. Record
   the `ALTER` wall time separately so "metadata-only" is demonstrated rather
   than asserted.
3. **Read cost, separately.** Time `model_rates()` on the same copy, and
   `analytics.get` before and after — the latter's claim is that it is
   **unchanged**, because the new op carries all the new read cost (§9.4).

Per the repo's rule, a timing assertion is a bet on machine load: the benchmark
records wall times as measurements with the load average beside them, and the
**tests** assert structure (`last_aggregate_source`, table shapes, absence of
extra clock reads via a counted monkeypatch) rather than durations.

### 12.3 Visual validation

Per the repo's rule, look at a rendered frame. Before/after SVGs from the real
`OperatorApp` via `run_test` + `save_screenshot`, for: `/analytics` populated
with rates, `/analytics` on a **pre-release ledger** (every rate cell `—` plus
the explanation — the state the operator's own history produces on day one),
`/analytics` where coverage is partial (legend present), the By-model section
empty, and `/session` with by-model rates. Capture at a width where
`_WIDE_TABLE_MIN` sheds the cache column, because that is where the new column
competes for space. Back the stills with the geometry: the row width against the
content box at 104 / 114 / 120 cells — the widths the module documents as the
ones that silently clipped `% cache`.

---

## 13. Risks, and what holds each one

1. **Rollup migration on the 658 MB ledger.** The dangerous failure is not
   slowness, it is `record_batch` dropping the whole batch — ledger row included
   — when a rollup upsert names a missing column (§4.3). *Mitigation:* only
   `session_daily` changes; it gains an explicit ALTER path; its existing shape
   guard stays the fail-closed backstop; `usage_daily`/`usage_monthly` are not
   touched at all, so the two unguarded rollups keep their current safety.
   *Residual:* a failed ALTER silently degrades every read to the 4.9 s ledger
   path. *Acceptable because* it is the store's documented posture (slow, never
   wrong), it is named by `last_aggregate_refusal`, and the benchmark measures
   the ALTER on a real copy.
2. **`_aggregate_from_row`'s positional contract.** Four producers must append
   the three sums in the same order: `_ledger_aggregate`, `_session_daily_aggregate`,
   `session_report`, and the test oracle. *Mitigation:* append **after** the
   component block so indices `0..9` and `10+i` are untouched, and assert
   `asdict` equality between paths so a mis-ordered producer fails loudly rather
   than shifting a number into `cost_micro`.
3. **`session_daily`'s in-sync gate and rebucket machinery.** The gate compares
   the rollup's `MAX(ts_ms)` and row count against the ledger; the rebucket
   re-derives days from `calls`. *Mitigation:* because the new columns ride
   `_SESSION_DAILY_MEASURE_COLUMNS`, `_SESSION_DAILY_REDERIVE_SQL` derives them
   as `SUM(<col>)` automatically, so a re-derived day and an accumulated day are
   equal by construction — which is the property the rebucket exists to
   preserve. A test asserts exactly that equality for the new sums.
4. **`test_session_report_equivalence.py`'s frozen oracle.** It pins a
   hand-written positional projection (its own `sums`/`fields` tuples) and
   compares the whole report key-for-key via `dataclasses.asdict`. *Mitigation:*
   the oracle gains the three measures in the same positions in the same PR.
   This is a required, mechanical edit — and it is exactly the guard that makes
   Risk 2 safe, because a positional slip fails it.
5. **The desktop contract mirror in the other repo.** `local-operator-ui`
   mirrors these types **by hand** and does not zod-validate responses (request
   args are `.strict()`; responses are not). A new required field on
   `DesktopUsageAggregate` is additive and cannot break an older client, but the
   client will not *see* it until its own mirror gains the fields.
   *Mitigation:* the backend change is strictly additive; the UI mirror and the
   `docs/design/panel-views.md` §5.3/§6.1 update are a separate, sequenced
   change in that repo. Note for the courier: that repo's checked-out working
   tree on this machine is **stale** (v0.25.11) relative to its `origin/main`
   (a7df70995, v0.30.29) — the contract must be read from the ref, not the
   working tree.
6. **Buffer/replayable calls.** `request.replayable` errands (compaction
   summaries, auto-naming) are BUFFERED rather than forwarded
   (`providers/failover.py:2883-2904`). *Assessment:* the wrapper sees the same
   event sequence either way — buffering changes when the consumer receives the
   events, not what the iterator yields — so a window measured here is still a
   real first-to-last-delta window. It does mean the "wall" semantics of such a
   call include the buffering, which is already true of `duration_ms` and is one
   more reason the wall rate is not labelled decode speed.
7. **Non-streaming providers / one-shot responses.** A provider or route that
   returns the whole answer in one frame yields one output delta →
   `decode_calls == 0`. *Mitigation:* the §8(b) guard excludes them and the
   coverage count reports the exclusion. These calls still appear in their
   model's `wall_tps`, which is the honest thing to say about them.
8. **Aborted and failed calls.** The `finally:` block records on every exit
   path, so a partial stream is recorded with whatever window it produced and
   `ok = 0`. *Mitigation:* the eligibility predicate does not test `ok` — an
   aborted call's measured generation is a real measurement, and excluding it
   would bias every rate upward by dropping exactly the slow calls. A test pins
   this, because "exclude failures" is the tempting wrong answer.
9. **`ttft_ms` and `first_reasoning_ms` are semantically untouched but now
   adjacent.** Collapsing the two `getattr` lookups into one risks changing when
   those two are set. *Mitigation:* the dispatch is written so each is set on
   exactly its own event type with its own `is None` guard, and the existing
   TTFT tests plus a new reasoning-only test pin both.
10. **The rate is forward-fill, so the feature looks empty on day one.**
    *Mitigation:* this is stated out loud (§4.4), the UI renders `—` with an
    explanation rather than `0 tok/s` (§7), and the wall rate covers the whole
    existing ledger immediately (§6.2) so the operator's first look at the table
    is useful rather than blank.
11. **`_row_overhead` / paint divergence (the D8/D11 repeat offender).** A new
    column added to one of the two functions and not the other silently clips
    the last column off the widest rows. *Mitigation:* `_tps_col` is shared by
    both, the new term is added to both in the same edit, and the test asserts
    painted row width against the content box at the boundary widths.
12. **A rate of `0` is reachable and meaningful.** A call can legitimately
    decode at under 1 tok/s. *Mitigation:* `decode_tps` returns `None` only for
    **no coverage**; a measured slow rate is rendered as its real value, and the
    formatter keeps one decimal below 10 so a genuine `0.4 tok/s` is not
    rounded to `0`. This is the inverse of the usual unknown-vs-zero trap and
    needs its own test.
