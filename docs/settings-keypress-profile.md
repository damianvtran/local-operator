# `/settings` arrow-key profiling report

**Verdict: the operator's report reproduces, and the dominant cost is NOT where
the task brief expected it.** `_row_text`'s per-row `Style`/`semantic_color`
work is real but secondary (≈14% of the widget's own time, ≈2.5% of a press).
The dominant cost is that **every arrow press re-rasterises the entire settings
list — all 96 lines — in order to repaint a 14-line viewport in which 2 lines
changed.** That cost is linear in row count, so a Hotkeys section makes it
proportionally worse: at 218 rows a press already costs **2.1× more CPU**.

Measured, not inferred. Everything below is reproducible with the harnesses in
`scripts/` (paths at the end).

---

## 1. Environment

| | |
|---|---|
| Machine | Apple M3 Max, 14 cores, macOS 26.6.2 (idle, on AC) |
| Worktree | `~/workspace/repos/lo-hotkeys`, branch `feat/hotkeys` @ `298cadb3b` |
| Interpreter | `.venv/bin/python` 3.12.13 — verified resolving `lo-hotkeys/local_operator/__init__.py` |
| Isolation | `scripts/probe_isolation` (fresh `HOME` + `LOCAL_OPERATOR_CONFIG_DIR`), every `CMUX_*` var deleted before app import |
| Terminal | headless Textual `run_test`, size 100×30 → settings body viewport = **14 lines** |
| Page shape | **96 rows** (79 selectable) from the current registry |
| Clock | `time.thread_time()` (per-thread CPU) per AGENTS.md; wall recorded alongside |

**No application source was modified.** All counterfactuals are monkeypatches
applied in throwaway processes. `git status` shows only new untracked scripts.

---

## 2. Two measurement bugs I hit and fixed — read this before trusting any number

These are recorded because both produced *plausible wrong answers*, and a
re-run of these harnesses must not reintroduce them.

**(a) Clamped presses are silent no-ops.** `/settings` is AGENTS.md's
documented wrap-vs-clamp exception: `action_move` clamps at the ends. There are
only 79 selectable rows, so a probe sending 100 `down` presses gets 79 real
moves and 21 presses where the cursor does not move and **`_list_text` is
byte-identical before and after** (verified directly). My first
elision-counterfactual sampled mostly clamped presses and reported ~0% saving
where the corrected run reports 12–16%. Fixed by `scripts/settings_press_driver.py`,
which bounces direction before either end; every sampled press now genuinely
moves the cursor and genuinely changes two lines.

**(b) cProfile's overhead is not a measurement.** Profiling and timing in the
same pass inflated the 96-row baseline from 2.49 ms to 6.93 ms, which made the
scaling table read as if 96 rows cost *more* than 218. cProfile is now a
separate pass and its milliseconds are labelled as non-comparable.

I also discarded a frame-equality check that compared variants at different
cursor positions and therefore reported every variant as "DIFFERS".

---

## 3. Per-press CPU, broken down by phase

`action_move` → `_settle_row` → `_land` → `_repaint` → four paints. Timers are
**exclusive** (a nested phase's time is subtracted from its parent), so the
column sums to the total. 100 bounced presses, 96 rows.

| phase | CPU/press (ms) | calls/press | share |
|---|---:|---:|---:|
| `_row_text` | 1.489 | 96.0 | 50.5% |
| `_paint_list` (own: `Text` assembly + `Static.update`) | 0.933 | 1.0 | 31.7% |
| `_paint_chrome` | 0.304 | 1.0 | 10.3% |
| `_paint_pane` | 0.076 | 1.0 | 2.6% |
| `_build_rows` | 0.048 | 1.0 | 1.6% |
| `_paint_detail` | 0.041 | 1.0 | 1.4% |
| `action_move` (own) | 0.020 | 1.0 | 0.7% |
| `_scroll_to_selection` | 0.017 | 1.0 | 0.6% |
| `_repaint` (own) | 0.014 | 1.0 | 0.5% |
| `_apply_height_ladder` | 0.006 | 1.0 | 0.2% |
| `_settle_row` / `_settle_expansion` / `_preview_choice` | 0.001 | 1.0 each | 0.05% |
| **total** | **2.95** | | |

Helper calls inside that press:

| helper | calls/press | CPU/press (ms) | share of press |
|---|---:|---:|---:|
| `theme_mod.semantic_color` | **552.7** | 0.396 | 13.6% |
| ↳ of which `theme_spec` | 552.7 | 0.121 | 4.2% |
| `settings_io.read_setting` | 165.0 | 0.091 | 3.1% |
| `settings_io.is_default` | 79.8 | 0.081 | 2.8% |
| `Static.update` | 9.1 | 0.821 | 28.1% |
| `settings_io.read_chains` | 1.0 | 0.002 | 0.1% |

### But `action_move` is only a fraction of the real keypress

| rows | `action_move` CPU | full press CPU | full press WALL | wall p90 |
|---:|---:|---:|---:|---:|
| 96 | 2.16 ms | **16.21 ms** | 86.7 ms | 129.9 ms |
| 138 | 3.05 ms | 19.12 ms | 85.2 ms | 124.8 ms |
| 218 | 4.63 ms | 24.28 ms | 89.2 ms | 147.5 ms |

**The widget's own work is ~13% of the CPU a press costs.** The other ~87% is
Textual's dispatch, refresh and compositing — which `_repaint` *causes*. This
is the single most important number in the report: optimising inside
`_row_text` alone is optimising 13% of the problem.

*(Wall figures under `run_test` include the pilot awaiting quiescence and are
an upper bound, not the felt latency in a real terminal. The CPU column is the
load-bearing one.)*

---

## 4. Row-count scaling — what a Hotkeys section costs

Registry synthetically inflated by cloning non-cascade settings (fresh keys,
same config paths; the probe never writes).

| rows | `action_move` CPU/press | µs per row | × baseline | full press CPU |
|---:|---:|---:|---:|---:|
| 96 (today) | 3.07 ms | 31.99 | 1.00× | 16.2 ms |
| 138 | 4.28 ms | 31.04 | 1.39× | 19.1 ms |
| 218 | 6.45 ms | 29.57 | 2.10× | 24.3 ms |

**Cleanly linear at ~30 µs per row per keypress**, with no cliff. The forecast
is therefore arithmetic: a Hotkeys section adding *R* rows adds ≈ 0.03·R ms to
every arrow press. A 60-row Hotkeys section ⇒ ≈ +1.8 ms of widget time and
≈ +5 ms of full-press CPU, on top of a page the operator *already* calls
laggy. **Fix the scaling before adding the section.**

---

## 5. The actual waste, named and quantified

### 5.1 The dominant one: full-height re-rasterisation (~48–51% of a press)

`_paint_list` pins the list `Static` to `height = len(self._rows)` so the
scroll container has something to scroll, then calls `update()`. Textual's
`Widget._render_content` (`textual/widget.py:4242`) rasterises **`self.size`,
i.e. all N lines**, caches them, and `Static.update` invalidates that cache
wholesale.

Measured exactly, via `Visual.to_strips` interception:

```
rows=96  listStatic.height=96  viewport=14
  Visual.to_strips calls on LIST /press: 1.00
  LINES rasterised into strips /press  : 96.0     <- the full height, every press

rows=218 listStatic.height=218 viewport=14
  LINES rasterised into strips /press  : 218.0
```

So the page rasterises 96 lines to show 14, of which **2 changed**. The
invalidation multiplies the cost of the identical 14-line draw by **179×** at
96 rows and **337×** at 218 rows (same viewport, with vs without a preceding
`update()`: 0.020 ms → 3.535 ms, and 0.024 ms → 7.975 ms).

Isolated cost of that one operation, against the page's real content:

| height (lines) | `to_strips` CPU | µs/line |
|---:|---:|---:|
| 2 (the lines that changed) | 0.363 ms | 181.7 |
| 14 (the viewport) | 0.602 ms | 43.0 |
| **96 (as shipped)** | **2.473 ms** | 25.8 |
| 218 (inflated) | 5.973 ms | 27.4 |

Rasterising only the viewport would save **1.87 ms/press (76%)** of this
operation at 96 rows and **5.01 ms (84%)** at 218.

Interleaved A/B/C confirms the share of the whole press:

| variant | 96 rows | 218 rows |
|---|---:|---:|
| A: as shipped | 17.17 ms | 24.68 ms |
| B: no-op `Static.update`s elided (correct frame) | 14.84 ms | 23.67 ms |
| C: **+ list update elided** (floor; deliberately wrong frame) | 9.00 ms | 12.15 ms |
| **A − C = list-invalidation budget** | **8.17 ms (47.6%)** | **12.52 ms (50.7%)** |

### 5.2 `_repaint` repaints things that did not change (100% redundant)

Over 59 consecutive real transitions:

| producer | identical to previous press | verdict |
|---|---|---|
| `_build_rows` | **59/59 (100%)** | re-derives the whole row list every press; never structurally different on a cursor move |
| `_paint_chrome` (title + rule) | **59/59 (100%)** | title is the config path; rule is `"─"×width` |
| `_paint_pane` | **59/59 (100%)** | providers/teams/agents, resolved once at open |
| `_paint_list` | **2 of 96 lines changed** (median; max 2, min 2) | 97.9% of composed rows are re-composed identically |
| `_paint_detail` | 0/59 identical | legitimately changes — it describes the cursor row |

`_build_rows` is cheap in itself (0.048 ms) but it is what forces `_paint_list`
to re-walk every row; and `_paint_chrome` at 0.304 ms is 10% of the widget's
time spent producing a byte-identical title and rule.

### 5.3 `_row_text`'s per-row theme + Style work (~14% of widget time)

Confirmed as described in the brief: five `semantic_color` lookups and five
`Style` constructions per row per paint, executed **before** the row's kind is
examined — so a header row pays for the `accent`/`faint` styles it never uses.
That is 5×96 = 480 of the 553 `semantic_color` calls per press.

`semantic_color` is not cached: it calls `theme_spec` → `_registry()` → dict
lookup on every call. Counterfactuals (medians over 60 bounced presses):

| variant | ms/press | saved |
|---|---:|---:|
| baseline | 2.141 | — |
| A: `semantic_color` memoised | 2.113 | 1.3% |
| B: A + `Style` objects interned | 1.858 | **13.2%** |
| C: whole row `Text` memoised (per-row cache ceiling) | 1.456 | **32.0%** |

Note **A alone is nearly worthless (1.3%)** — the lookup is already fast. The
cost is constructing 480 `Style` objects, not resolving the colours. This
matters for ranking: "cache `semantic_color`" is the intuitive fix and it is
the wrong one.

### 5.4 Ranked candidate fixes, measured end-to-end

Interleaved, frame-correctness verified against the control at a fixed cursor
position (all reported IDENTICAL):

| variant | 96 rows | saved | 218 rows | saved |
|---|---:|---:|---:|---:|
| A: as shipped | 16.89 ms | — | 22.67 ms | — |
| B: elide no-op `Static.update` (pane/title/rule) | 14.14 ms | 16.3% | 20.68 ms | 8.8% |
| D: B + `_row_text` memo | 13.87 ms | 17.9% | 18.55 ms | 18.2% |
| E: D + per-line strip cache | 13.33 ms | 21.1% | 18.36 ms | 19.0% |

**Variant E under-reports its own ceiling** and should not be read as the limit:
it is implemented as a *post-pass* that still pays the full rasterisation before
discarding it, so it demonstrates reuse (94/96 = 97.9% of line strips reusable;
216/218 = 99.1% at scale) without banking the saving. The true ceiling for that
change is §5.1's table: **1.87–5.01 ms/press**.

---

## 6. Recommended optimisations, ranked by measured payoff

**R1 — Stop rasterising the whole list every press. (~48–51% of a press;
1.87 ms at 96 rows, 5.01 ms at 218; this is the one that fixes the scaling.)**
Two viable shapes:
 (a) keep the pinned-height `Static` but let unchanged lines survive an
 `update()` (line-keyed strip cache), or
 (b) stop handing Textual one N-line widget — render only the visible window
 and let the scroll container map offsets.
*Risk: high — this is the load-bearing one.* (b) touches `_index_at` (click→row
mapping), `_scroll_to_selection`, `_attached_bottom`, the scrollbar's virtual
size, and the `_list.styles.height = len(rows)` contract that makes scrolling
work at all. (a) is more local but needs a correct invalidation key: theme
change (`theme_mod._theme_epoch` already exists for exactly this), width
change, config write, expansion open/close, hover. **Not safe to do blind — it
needs the guards in §7 landed first.** Prefer (a).

**R2 — Skip `Static.update` when content is unchanged. (12–16% at 96 rows,
~9% at 218; measured frame-identical.)** Pane, title and rule produce
byte-identical `Text` on 59/59 transitions. Compare against the last content
handed to that widget and return early.
*Risk: low-to-moderate, and it is a real risk, not a nit.* Correctness depends
entirely on the comparison key. Rich `Text` equality covers plain text + spans
+ style, which is what rasterises — but a change in the widget's **own styling**
(a `$lo-*` variable moving under a theme switch) does not change the `Text` and
would leave a stale frame. Invalidate on theme epoch as well as content, and
keep the key structural (`plain`, `spans`, `style`), never `id()`.
*Note it degrades with scale (16.3% → 8.8%), so it is not a substitute for R1.*

**R3 — Hoist the `Style` prelude out of the per-row loop. (13.2% of widget
time; ≈2% of a press.)** Build the five `Style` objects **once per
`_paint_list`** and pass them down, rather than 5×N times. Do *not* bother
memoising `semantic_color` alone — measured at 1.3%.
*Risk: low.* Styles are immutable and already per-paint; the only invalidation
is a theme change, and a per-paint lifetime makes that automatic. This is the
safest item on the list and the natural place to start.

**R4 — Don't rebuild rows on a pure cursor move. (1.6% of widget time; small,
but it unlocks R1(a).)** `_build_rows` was structurally identical on 59/59
transitions. Cache the row list, invalidate on the things that actually change
structure: `_expanded`, `_editing`, `_chain`, `_suggest_index`, a config write,
`set_context`.
*Risk: moderate — this is where a stale-list bug would hurt.* `_repaint`
currently relies on the rebuild to re-anchor `_selected` after a row vanishes,
and `_cascade_rows` calls `settings_io.read_chains(self._manager)` on every
build, so the cache must be dropped on any write. The payoff is not the 0.048 ms;
it is that a stable row list is the precondition for caching row text safely.

**R5 — Don't repaint chrome on a cursor move. (10.3% of widget time; ≈2% of a
press.)** `_paint_chrome` recomputes the title, re-multiplies the rule string
and calls `_paint_hints()` every press. Only the hints can change on a move
(they depend on the cursor row), so split it: hints per move, title/rule on
resize or config-path change.
*Risk: low,* provided the resize path still repaints the rule (its width
depends on `self.size.width`).

Doing **R3 + R5 + R2** is a low-risk package worth ~20% with no architectural
change. **R1 is the one that decides whether the Hotkeys section is affordable**,
and it should be done before the section lands, not after.

---

## 7. Regression guard proposal — structural, not timed

Per AGENTS.md ("Prefer a structural invariant to a numeric one"; no portable
numeric bound survives the CI/laptop core-speed spread), these count operations
per keypress. They cannot flake on machine load.

Proposed test, in `tests/unit/tui/test_settings_view.py`, driving the real
`OperatorApp` and using a bounced (never clamped) press:

```python
# rows == 96, body viewport == 14 at 100x30
assert semantic_color_calls / presses <= 40      # G1: not 5 per row per paint
assert list_strip_lines / presses <= 3 * viewport # G2: not the full widget height
assert row_text_calls / presses <= 2 * viewport   # G3: not one call per row
assert build_rows_calls / presses == 0            # G4: no rebuild on a cursor move
```

**Proven to fail on the current tree** (10 bounced presses, 96 rows) — i.e.
they detect the waste rather than passing vacuously:

| guard | measured now | bound | result |
|---|---:|---:|---|
| G1 `semantic_color` ≤ 40/press | **554.1** | 40 | FAIL |
| G2 list strip lines ≤ 42/press | **96.0** | 42 | FAIL |
| G3 `_row_text` calls ≤ 28/press | **96.0** | 28 | FAIL |
| G4 `_build_rows` == 0/press | **1.0** | 0 | FAIL |

Each bound is expressed **relative to the viewport, not to the row count**,
which is what makes them Hotkeys-proof: adding 60 rows must not change any of
these numbers, and if it does, the guard fires. G2 is the important one — it is
the direct assertion that a cursor move does not re-rasterise the whole list.

Adopt them as each optimisation lands (G1/G3 with R3, G4 with R4, G2 with R1),
rather than all at once against unfixed code.

---

## 8. Harnesses (all in `scripts/`, re-runnable)

Prefix every invocation with `env -u NO_COLOR TERM=xterm-256color .venv/bin/python`.

| script | what it answers |
|---|---|
| `settings_press_driver.py` | the bounce driver — **import this in any new probe**; a clamped press invalidates results |
| `settings_keypress_profile.py` | per-phase exclusive CPU, helper call counts, row-count scaling (`--rows 120,200`, `--cprofile`, `--json`) |
| `settings_keypress_waste.py` | redundancy of produced output; Style-prelude counterfactuals |
| `settings_keypress_endtoend.py` | `action_move` vs full press round trip, CPU and wall |
| `settings_keypress_render.py` | lines Textual rasterises per press vs viewport |
| `settings_keypress_widgets.py` | per-widget rasterisation; the `update()` invalidation multiplier |
| `settings_strips_scaling.py` | `Visual.to_strips` cost vs height — the R1 ceiling |
| `settings_keypress_fixmodel.py` | interleaved A/B/D/E ranking with frame-correctness checks |
| `settings_keypress_ceiling.py` | interleaved A/B/C list-invalidation budget |
| `settings_keypress_dirty.py` | dirty-widget count vs compositor path |
| `settings_keypress_fullpress.py` | cProfile of the whole press round trip |

Gates on all of them: `flake8`, `black --check` (26.1.0), `isort --check`
(5.13.2) clean.

---

## 9. What I could not measure — stated plainly

1. **Real-terminal latency.** Everything is headless `run_test`. Textual
   coalesces refreshes on a timer in a real terminal and the pilot awaits
   quiescence, so the WALL figures (~87 ms/press) are an upper bound and should
   not be quoted as the felt lag. CPU figures and all *ratios* transfer; the
   absolute wall number does not. Confirming the felt improvement needs a real
   terminal after a fix lands.
2. **No CI calibration.** Per AGENTS.md, ceilings must come from CI logs across
   several runs. Every number here is one M3 Max. This is why §7 proposes
   **counting** guards and no timing bound at all.
3. **The operator's own config.** Measured against an isolated empty config, so
   the cascade section renders its empty state. A populated cascade adds rows
   and would make the page slower, never faster — the 96-row figure is a
   floor.
4. **Variant E is not a validated implementation**, only evidence that 97.9%
   of line strips are reusable. §5.1's `to_strips` table is the honest ceiling.
5. **Terminal width.** All at 100×30. `_row_text` and `_paint_list` both key off
   `_list_width()`, so a wider terminal costs more per line; not swept.
6. **Hover.** `_hovered` participates in row rendering; mouse-move repaints were
   not profiled, only keyboard movement, which is what was reported.
