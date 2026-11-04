# es · tui.analytics · round 3 (w6a3) — delta audit + class sweep

**Verdict: passed** for `tui.analytics` (0 blocker, 0 major). One **major** found by the class sweep
in another namespace: `tui.small.goal_panel.done_suffix` (bundle `../../tui.small/w6a3/`, status `changes_requested`).

Auditor: i18n-auditor (lopdev), session 13ac587c68a7, model **anthropic/claude-sonnet-5-5**, no fallback or
model switch this round. Independent of the author. Target verified:
`es/tui.analytics.json` sha256 `0287f77ba86d6b7e1ea1e29107f7b3344932a338d5a3e70e4440c2b5397358b7`
(modified vs HEAD `4912975a4b5dd0c2c6011dbfcb20188c50e44084`, so the hash, not HEAD, binds the bytes).
`git diff` confirms exactly two values changed. `scripts/i18n/check.py --catalogues-only`: 0 problems.

## 1. The two labels

| key | before | after | cells | column |
|---|---|---|---|---|
| `totals.cache_hit` | `Tasa de aciertos de caché` | `Aciertos de caché` | 25 → 17 | 22 (21 usable) |
| `totals.decode` | `Tasa de decodificación` | `Decodificación` | 22 → 14 | 22 |

**Correct.** The column is code-fixed (`analytics_panel.py` `kv()`: `f"  {name:<22}"`, value at cell 24). In the
rendered frame both rows now keep ≥ 1 blank cell and start the value at cell 24; the English control has no fused row.
Meaning: the row reads `Aciertos de caché  86%  del contexto servido desde la caché` and
`Decodificación  42 tok/s  velocidad de decodificación medida`. The % and tok/s carry the unit and the note names the
measure, so a Spanish developer reads both as measurements; `Tasa` was redundant. Nothing better needed (nit R3-3 only).
Note the rate footnote `rate.legend` ("tasa de decodificación en N de M llamadas") still says *tasa* under the tables — consistent.

**Negative control:** restoring the round-2 labels trips the static column check (25 and 22 cells in a 22-cell column)
and the rendered frame fuses (`Tasa de aciertos de caché86%`, `Tasa de decodificación42 tok/s`); see audit.log §3–4.

## 2. Class sweep (code-fixed columns, 8 namespaces)

Method: AST taint scan of `tui/*.py` and `tui/widgets/*.py` for catalogue strings reaching `ljust/rjust/center/add_column/kv/marked/_pad_cells`
or format-spec pads; CSS fixed widths; module constants named `*CELL*/*COL*/*WIDTH*`; each traced to its key and
measured with `rich.cells.cell_len`. Discovery is a pinned set in `checks.py` (it fails if the sink set changes).
Only two direct sinks exist: `kv()` (analytics, checked) and `info_panel.marked()` (a width ladder). Full table: audit.log §6 (verbatim).

| key / group | site, column | en | es | verdict |
|---|---|---|---|---|
| 8 Totals labels (`row_billed … decode`) | analytics `kv()` 22 | ≤ 16 | ≤ 18 | OK |
| `search.row_total` / `row_session` | `_SEARCH_NAME_COL` 22 | 11/12 | 11/11 | OK |
| `Est. cost` (`cost_label`, 18 en) | costs.py literal, **not catalogued** | 18 | — | n/a, English-only |
| 21 library rail labels | CSS `width:22` → 14–20 cells | ≤ 13 | ≤ 16 | OK; `category_spreadsheet` 16 → `Hojas de cálc…` (truncated by design, en 12 fits) |
| `projects.render.no_estimate_or_dates`, `no_progress`, `no_progress_recorded` | board card `BOARD_COLUMN_WIDTH` 32 (31 usable) | 20/11/20 | 24/11/23 | OK |
| `projects.render.reported_ago + live_count` | same card, rendered | 24 | 29 (fresh, 2 live) | OK |
| same, stale 3d, 1 live | same card, rendered | `…· 1 li…` | `…· 1 …` | truncated with ellipsis in **both** locales (code tolerates; es loses the `activas` word 1 cell sooner) — nit |
| **`small.goal_panel.done_suffix`** | `_GOAL_TAG_CELLS = 8` | 8 | **9** | **OVER — MAJOR**, see below |
| `app.help.{paste,copy,copy_more,skill,model_saved,cmd_paste}_note` | /help gutter 20, ceiling 74 | 59–74 | 64–74 | OK (`copy_note_more` es = 74, zero spare) |
| `app.help.{title,title_more,notify,notify_more}_note` | same | 110–151 | 128–178 | wraps in both locales (parity), already wrapped in en |
| `analytics.totals.{cache_read,cache_write,cache_hit,decode}_note` | kv note room 29 at 80x24 (body 66) | 23/26/28/21 | 30/33/35/34 | es cropped, en fits — **minor R3-2**; value intact |
| `analytics.totals.fresh_note` | same | 30 | 42 | cropped in both |
| `analytics.table.*`, `usage.*`, `hub_links.policy.*`, `import.*`, `small.session_sidebar.*`, `small.todo_panel.more_below*`, `small.info_panel.*` | data-sized columns or first-fit ladders | — | — | OK (tolerated by construction); hub_links rendered at 40/60: no overrun, policy wraps cleanly |

### The OVER (outside this namespace)
`tui.small.goal_panel.done_suffix` `  — hecho` is 9 cells; code reserves 8. Rendered in the real `GoalPanel` (100x40, `d`):
`▌ Migrar … con e…  —` / `hecho` — the done tag splits across two lines and the card grows a row; English is one line.
Recommended fix: raise `_GOAL_TAG_CELLS` to 9 (code). Recorded as S1 in `i18n/audits/evidence/es/tui.small/w6a3/`.

### Not frame-rendered (stated, not claimed)
Library detail/preview pane, settings value column (`default_marker` 18 vs en 11, `read_only_label`), picker footers
(`session_picker.no_match_filter` es 38 cells vs en 30 at `PICKER_MIN_WIDTH` 30 — how that line fits was not checked), `ask_queue` compact headline, `model_picker`. These were read, not measured in a frame.

## 3. Findings

| id | sev | key | status |
|---|---|---|---|
| R3-1 | — | new labels correct and fit | closed |
| R3-2 | minor | `totals.*_note` crop at 80x24 (es only on 4 rows) | open; optional ≤ 29-cell wordings in record.json |
| R3-3 | nit | `Decodificación` has no `rate`; preference only | open |

No label/value fusion and no clipped figure remains in `tui.analytics`.

## 4. Provenance
`checks.py` exited 0 (captured verbatim in `audit.log`); probes are retained as `probe_*.py.txt`. The first checks run
exited 1 on my own discovery guard (`_model_rate_section` pad takes a WIDTH derived from `cell_len` of catalogue
suffixes, not a catalogue string); I added it to the pinned set with that reason and re-ran. The first `probe_goal_frame`
attempt printed `ERR` because machine load average was ~120 and its 40 s wait expired; the re-run is in the log.
