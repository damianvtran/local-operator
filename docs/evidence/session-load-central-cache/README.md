# Session loads: one page at a time, one row at a time, and one session at a time

Evidence for `perf/session-load-central-cache`, whose subject is the load path the
operator's complaint is about: *"The backend could not complete this request"*,
and one conversation's open making the other conversations stop loading.

Three things caused that, and they are three separate costs:

| # | cost | where | fixed by |
|---|---|---|---|
| 1 | a one-row metadata question answered by parsing the whole journal | `locate()`'s checkpoint fallback, `_persisted_children`, and four more | `read_latest_custom*` (backward, bounded) |
| 2 | the same page decoded twice per open, and again on every re-request | the open's two `snapshot()`s, the 1 Hz child read, the renderer's reconcile walk | the process-wide page cache + single-flight |
| 3 | **the pool-wide lock held across both of those** | `DesktopSessions.session` | the handout reservation + lookup single-flight |

## Read this first: an earlier reconnaissance of this work measured the wrong tree

An earlier pass over this work (and the design brief it produced) took its
measurements with `sys.path` pointed at `~/local-operator`, a checkout 20+
releases behind `main` and 20+ commits before the backward page reader landed
(PR #1109, `d437dfdc6`, merged 2026-09-14). So it reported `read_transcript_page`
at **1565-1940 ms** and concluded that a whole-journal *page scan* was the load
path's cost.

That is not the before-state of anything shipping, and **the backward page reader
is not re-claimed here**: `read_transcript_page` reads backward on `main` and its
forward-oracle differential test is committed (`tests/unit/session/test_transcript.py`).
The numbers below are the corrected ones, re-measured on `origin/main`
(`bf67bf699`) and on this branch, on the same store, minutes apart.

The corrected picture, and the reason the change is shaped the way it is: the
page reads are ALREADY cheap (1-4 ms on the 261 MB conversation). The remaining
whole-journal parse is `Transcript(...)` CONSTRUCTION, and it reached the load
path through the metadata reads of (1). That parse runs with the pool-wide lock
in hand, which is what turned a slow open into a server-wide stall (3).

## 1. The before/after table, on the operator's real store

`scripts/bench_session_page.py`, `--samples 3 --rounds 2`, alternating the two
trees session by session with the order flipped each round. Host load average
during the run: **179-260** at the start of each worker (this machine runs at
150-270, so the milliseconds are weather); 176.87 / 273.00 / 312.10 at the end.
The `rows decoded` column is a COUNT and is load-independent — it is the column
to read, and the milliseconds are beside it only because a reader wants them.

```sh
.venv/bin/python scripts/bench_session_page.py \
    --source-root ~/workspace/repos/lo-session-load-base \
    --output /tmp/page-bench/ab.json
```

| session | size MB | operation | before ms | after ms | before rows decoded | after rows decoded |
|---|---|---|---|---|---|---|
| bda7b76d34e0 | 261.0 | tail_page | 1.3 | 1.2 | 101 | 101 |
| bda7b76d34e0 | 261.0 | through_id_page | 0.9 | 1.0 | 101 | 101 |
| bda7b76d34e0 | 261.0 | before_id_page | 1.7 | 1.5 | 201 | 201 |
| bda7b76d34e0 | 261.0 | replay_suffix | 6.3 | 9.4 | 833 | 833 |
| bda7b76d34e0 | 261.0 | transcript_construct | 2286.7 | 1842.9 | 21287 | 21287 |
| bda7b76d34e0 | 261.0 | metadata_row_via_transcript | 2059.5 | 2313.8 | 21287 | 21287 |
| bda7b76d34e0 | 261.0 | **metadata_row_scan** (new) | absent | **12.2** | absent | **975** |
| bda7b76d34e0 | 261.0 | cached_page_miss | absent | 4.6 | absent | 101 |
| bda7b76d34e0 | 261.0 | cached_page_hit | absent | **0.0** | absent | **0** |
| 2f95e374dd22 | 107.6 | tail_page | 84.1 | 133.7 | 101 | 101 |
| 2f95e374dd22 | 107.6 | transcript_construct | 248.2 | 295.4 | 1656 | 1656 |
| 2f95e374dd22 | 107.6 | metadata_row_via_transcript | 224.6 | 233.5 | 1656 | 1656 |
| 2f95e374dd22 | 107.6 | **metadata_row_scan** (new) | absent | **10.3** | absent | **9** |
| 2f95e374dd22 | 107.6 | cached_page_miss | absent | 92.0 | absent | 101 |
| 2f95e374dd22 | 107.6 | cached_page_hit | absent | 97.0 | absent | 101 |
| 9f8e5b652ac7 | 95.9 | tail_page | 2.0 | 1.7 | 101 | 101 |
| 9f8e5b652ac7 | 95.9 | transcript_construct | 580.7 | 540.6 | 14599 | 14599 |
| 9f8e5b652ac7 | 95.9 | metadata_row_via_transcript | 546.3 | 564.3 | 14599 | 14599 |
| 9f8e5b652ac7 | 95.9 | **metadata_row_scan** (new) | absent | **11.8** | absent | **1177** |
| 9f8e5b652ac7 | 95.9 | cached_page_miss | absent | 4.8 | absent | 101 |
| 9f8e5b652ac7 | 95.9 | cached_page_hit | absent | **0.1** | absent | **0** |
| 4140ee201ce1 | 86.9 | tail_page | 4.5 | 1.9 | 101 | 101 |
| 4140ee201ce1 | 86.9 | transcript_construct | 565.0 | 782.5 | 12121 | 12121 |
| 4140ee201ce1 | 86.9 | metadata_row_via_transcript | 570.2 | 607.1 | 12121 | 12121 |
| 4140ee201ce1 | 86.9 | **metadata_row_scan** (new) | absent | **5.2** | absent | **470** |
| 4140ee201ce1 | 86.9 | cached_page_miss | absent | 9.0 | absent | 101 |
| 4140ee201ce1 | 86.9 | cached_page_hit | absent | **0.0** | absent | **0** |

What to take from it, in the order the numbers matter:

1. **The page reads are the same in both trees** — 1-4 ms, and 84-134 ms on the
   session whose rows are enormous. They are not the change, and the table says
   so rather than implying otherwise.
2. **`metadata_row_scan` is 5-12 ms where the same row costs 225-2314 ms** through
   `Transcript(...)`, and it decodes 9-1177 rows where the construction decodes
   1656-21287. That is the load path's remaining parse, removed from the six call
   sites in §1 of the design brief.
3. **`transcript_construct` is unchanged (and noisy in both directions)**: the
   runtime's own cold engage and any caller that needs the whole replay still pay
   it. That cost is (D) in the brief, deliberately deferred — see *What remains*.
4. **A cache hit decodes ZERO rows.** The hit is a different kind of thing from a
   faster read, and the structural column is what shows it.
5. **`cached_page_hit` on `2f95e374dd22` is a MISS, by design.** That session's
   100-row page accounts **84 MB**, more than the whole budget, so it is refused
   and re-read (the worker's tally reports `oversize: 4`). Holding it would cost
   more than re-reading it; a page too large to admit is a miss, never an error.

The cache tally each worker prints is cumulative for that worker's run, so the
per-session figures read as "hits/misses/oversize/entries at this point in the
run". For the four ordinary sessions above the shape is `hits 3 / misses 1 /
oversize 0 / entries 1` against a page of 0.5-1.5 MB accounted.

**On the budget.** The design brief proposed 4 MiB, sized from *serialized* page
sizes — but the same section specifies `retained_bytes`, which charges
`sys.getsizeof` per object and measures 1.7-2.7x a page's serialized size. At
4 MiB accounted the cache admitted nothing for any ordinary conversation: this
harness's own `oversize` tally caught it on a 9 MB journal whose 100-row page
accounts 4.6 MiB, i.e. the "bounded cache" was a bound on an empty one. The
shipped constant is 24 MiB (`session/page_cache.py`), and a test pins the
relationship rather than the number.

## 2. The unit-level results

The differential — the reader must answer exactly what the resident object
answered — runs twice: as a committed test matrix, and against the operator's
real journals.

```sh
# 7 journal shapes (plain, dirty, torn, no-newline, huge row, one row, empty) x
# every cursor and limit shape, against the forward implementation kept as an oracle
.venv/bin/python -m pytest tests/unit/session/test_transcript.py -q

# The same comparison on the real store: every custom type in six journals,
# both projections, values AND exceptions.
.venv/bin/python docs/evidence/session-load-central-cache/real_store_differential.py
```

```
bda7b76d34e0: 11 custom types compared, 0 mismatches so far
2f95e374dd22: 9 custom types compared, 0 mismatches so far
9f8e5b652ac7: 11 custom types compared, 0 mismatches so far
4140ee201ce1: 11 custom types compared, 0 mismatches so far
29435655756c: 7 custom types compared, 0 mismatches so far
03f18d75b736: 10 custom types compared, 0 mismatches so far

comparisons=130 mismatches=0
```

Structural tests, each driven in the direction that FAILS before being trusted
(the mutation is named, and was run):

| test | mutation that turns it red |
|---|---|
| `test_a_metadata_read_decodes_only_the_rows_above_the_match` | the early return removed → the decode count becomes the journal (4000 rows for a 3-row tail) |
| `test_a_metadata_read_near_the_head_costs_the_journal_and_is_still_correct` | pins the honest worst case instead: 501 decodes for a row at byte zero, never worse than today |
| `test_the_one_row_read_matches_the_resident_transcript` | the `ENTRY_CUSTOM` filter dropped → a message row carrying a `custom_type` key answers instead |
| `test_the_newest_match_wins_...` | the first (oldest) match returned instead → 7 red |
| `test_a_hit_reads_nothing_and_returns_the_same_page` | `size`+`inode` dropped from the key, or `put` not storing → 3 red |
| `test_a_same_size_replacement_serves_the_new_content` | `inode` dropped from the key → the old rows are served |
| `test_two_concurrent_loads_issue_one_read` | single-flight removed → 2 reads |
| `test_a_follower_after_an_append_does_not_share_the_stale_read` | the post-read re-stat dropped → a read that spanned a write is published |
| `test_the_byte_budget_admits_a_page_the_size_of_a_real_conversation` | the budget back at 4 MiB → the ordinary page is refused |
| `test_retained_bytes_sees_dataclass_payloads` | the `dataclass` branch dropped → the account collapses to the page object |
| `test_a_parked_lookup_does_not_delay_another_sessions_open` | the pool lock held across the lookup → hangs, then fails on the backstop |
| `test_a_parked_lookup_does_not_delay_an_already_warm_session` | same mutation |
| `test_two_cold_callers_share_one_lookup_and_one_bridge` | the single-flight removed → two flights, two journal reads |
| `test_a_bridge_awaiting_its_first_acquire_is_not_evicted` | the handout dropped from `_evictable` → the handed-out bridge is evicted |
| `test_a_failed_open_leaves_no_reservation_behind` | the handout never released → the reservation leaks |

Gate results for the branch (`bf67bf699` + 6 commits):

```
.venv/bin/python -m flake8 .                              → rc=0 (clean)
uvx --from black==26.1.0 black --check .                  → 1450 files unchanged
uvx isort==5.13.2 --check .                               → rc=0
.venv/bin/python -m pyright --pythonpath .venv/bin/python . → 0 errors, 0 warnings
env -u NO_COLOR TERM=xterm-256color .venv/bin/python -m pytest tests/e2e -m e2e -n0 -q
                                                          → 168 passed, 7 skipped in 18:55
.venv/bin/python -m pytest tests/unit -q                  → 22828 passed, 22 skipped, 5 failed
```

**The five unit failures are pre-existing load flakes, and that is a measurement
rather than an assertion.** Two full-suite runs on this branch failed DIFFERENT
sets (`wakes/test_supervisor` is the only overlap), and every one of them passes
in isolation on both trees:

```
# this branch, the five named tests in one process, three runs
   5 passed · 1 failed (test_apply_frontend_state_preserves_auto_effort_label) · 5 passed
# origin/main, the SAME five, the same way
   5 passed · 5 passed · 1 failed
# origin/main under ~6 CPU burners (load average 340), same five
   5 passed · 5 passed · 1 failed (test_a_reselect_after_navigating_away_gets_a_full_budget
                — "the precondition left no residue to clear", the same assertion
                that run made on this branch)
```

Both trees flake; the branch is not the variable. Six tests failed between the two
full-suite runs on this branch: `tests/unit/wakes/test_supervisor.py::test_a_record_is_dropped_when_its_occurrence_is_no_longer_owed`
in BOTH, and in one run each
`tests/unit/tui/test_ledger_lane_refit.py::test_opening_the_sidebar_behind_a_scroll_refits_every_row`,
`tests/unit/tui/test_ask_picker.py::test_the_reveal_never_makes_another_rows_prose_unreachable`,
`tests/unit/tui/test_cut_off_notices.py::test_the_poller_paints_a_cut_off_in_the_danger_tier`,
`tests/unit/tui/test_sidebar_connect_retry.py::test_a_reselect_after_navigating_away_gets_a_full_budget`
and `tests/unit/tui/test_effort.py::test_apply_frontend_state_preserves_auto_effort_label`
— all TUI pilot or wake-timer tests, and all green when run alone on either tree.

`tests/unit/session`, `tests/unit/server`, `tests/unit/mobile` and the two TUI
test files this branch edits are green in every run.

## 3. The live reproduction, in isolation

The operator's live backend, captured before this change (load average 244):

```
GET /v1/desktop/sessions/2f95e374dd22  (snapshot)        200 0.244637
GET /v1/desktop/sessions/4140ee201ce1  (snapshot)        200 0.762162
GET /v1/desktop/sessions/9f8e5b652ac7  (snapshot)        503 25.373243
GET /v1/desktop/sessions/bda7b76d34e0  (snapshot)        200 32.115061
GET /v1/desktop/sessions/9f8e5b652ac7/history?limit=100  503 25.457624
GET /v1/desktop/sessions/bda7b76d34e0/history?limit=100  200 1.769011
GET /health                                              200 0.005543
```

The 503 body is `{"detail":"Session owner is unavailable. Reconnect and reconcile
before retrying."}`, and a 642-byte conversation's snapshot answered in **25 ms**
alone but **829 ms** while the 261 MB session's snapshot was in flight — **33x**,
for a session whose own read is trivial. `/health` answered in 5.5 ms throughout,
so the server was alive and serialised rather than broken.

That is reproduced here against a **copy** of the 261 MB session (never the
operator's store), with the server on a redirected `HOME`/config and the desktop
bearer set by the script:

```sh
# one per tree, on its own port
.venv/bin/python docs/evidence/session-load-central-cache/live_repro.py \
    --tree ~/workspace/repos/lo-session-load-base --side before --port 11823
.venv/bin/python docs/evidence/session-load-central-cache/live_repro.py \
    --tree ~/workspace/repos/lo-session-load      --side after  --port 11822
```

Result (`--side before` on `origin/main`, `--side after` on the branch, one run
each, host load average 290-393 during both):

| measurement | before (`origin/main`) | after (branch) |
|---|---|---|
| `/health` | 4.7 ms | 6.5 ms |
| small session's snapshot, warm, alone | 20.2 ms (median of 3) | 5.9 ms (median of 3) |
| **large session, COLD snapshot** | 200 in **3990 ms** | 200 in **2215 ms** |
| **small requests completed INSIDE that window** | **1** — and it took 3990.9 ms | **22** — median 69.3 ms, worst 322.8 ms |
| large session, `/history?limit=100` | 200 in 1564 ms | 200 in 2282 ms |
| **small requests completed inside THAT window** | **2** — median 789.3 ms, worst 1570.0 ms | **21** — median 100.8 ms, worst 267.8 ms |

Three readings of that:

1. **The stall is reproduced and then gone.** Before, one request fits in the
   open's window and it lasts exactly as long as the open (3990.9 ms against
   3990.0 ms — the request was not slow, it was WAITING). After, the same window
   serves 22 requests, each roughly its own cost.
2. **The same signature on the page read**, which is the other half of an open:
   2 requests before, 21 after. The page read's own duration is unchanged (1564
   vs 2282 ms here is the run's load, not the change — the harness's table above
   is the honest comparison for that number), and the reasoning for that is the
   point: the page read was never the cost; being serialised behind it was.
3. **`/health` is fast in both**, which is what makes the live 503s readable as a
   queue rather than a dead server — the same control the live capture has.

What did NOT change, and is deliberately not claimed: `Transcript(...)`
construction is still 1.8-2.3 s on this journal, so a cold open is still seconds
long. It no longer stops the other conversations.

## 4. What remains, stated rather than implied

- **`Transcript(...)` construction is unchanged** (1.8-2.3 s on the 261 MB
  journal). It is the runtime's own cold engage, and making `_entries` lazy is
  (D) in the design brief: deferred, because `_write_entries`' rebuild branch
  would replace a 261 MB journal with the new batch alone if `_entries` were an
  unmaterialised empty list, and because a `threading.Lock`-guarded load called
  from a loop thread is the shape of #401.
- **A `<` 84 MB page cannot be cached** and is re-read; the tally reports it.
- **`harness/comms.py`'s peek parse** renders every row to number the steps —
  the same defect shape one layer away, out of this change and unclaimed here.

## Artifacts in this directory

| file | what it is |
|---|---|
| `real_store_differential.py` | the one-row reader against the resident `Transcript`, over six real journals; non-zero exit on a mismatch |
| `live_repro.py` | the isolated two-session server reproduction of the stall, with `--tree`/`--side` so both sides run the same code path |

`scripts/bench_session_page.py` is the reusable harness (in `scripts/` because it
is worth running again, with `--source-root` for the A/B).
