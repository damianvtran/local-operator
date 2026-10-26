# Independent French audit — `tui.notifications` (audit `ntt1`)

Date: 2026-10-25. Auditor: `translation-auditor`, session `429b867caa2e`.
**Actual model performing this audit: `radient/auto`** — the runtime emitted a
provider-quota fallback chain during this task
(`anthropic/claude-opus-5-5` → `anthropic/claude-sonnet-5-5` →
`openai/gpt-6.1-sol` → `radient/auto`). The requested primary did not serve the
audit; neither intermediate did.

Independent of the translator (`coder`, per delegation provenance, model
`Opus5.5`). This auditor authored none of the English, French, desktop or wire
copy audited here, and **edited no catalogue, code, generated asset, git object
or ledger row**. The worktree `~/workspace/repos/lo-notifications-l1` was read
only.

## Verdict

**passed / clean-with-findings.** 72 English/French key pairs and 72 context
entries read (every key in the namespace). **0 blocker, 0 major, 3 minor,
3 nit.** No placeholder, plural, literal-token, NFC or typography defect.

| Cluster (keys) | Verdict |
|---|---|
| list rows, chips, time column (12) | clean — `row_now` `<1m` = 3 cells of a 5-cell column |
| header, filters, group labels (16) | clean except FX-NTT-6 nit (`header_partial`) |
| detail pane (11) | clean except FX-NTT-2 minor (`updates_since` fit at 4-digit counts) |
| footer hints (13) | clean — `cocher` / `lu` / `non lu` verified against the 8-cell budget |
| receipts, refusals, pointer (9) | clean except FX-NTT-5 nit (`arg_read` scope) |
| empty states, banner, sections (11) | clean |

## Inputs, pinned and verified

| Input | sha256 | Result |
|---|---|---|
| en `tui.notifications.json` @ `d25392f3553c38ce1740c98e53c9c66a700e9e64` | `b415bc70…e70b68` | matches delegation pin |
| en `tui.notifications.context.json` @ same | `0a9ee260…d85907` | matches delegation pin |
| fr target (untracked working-tree file) | `87866da0d850589f04c12d28599bfc9235c0facf9aa5cc37b39ac20cbea29905` | matches delegation pin |

The target was copied to this scratch directory **as the first action**, before
the coder's branch switch in that worktree, and the copy — not the worktree —
is what was audited. Re-checked at the end of the audit: the worktree file still
hashs to `87866da0…a29905`, and the worktree HEAD moved to
`b33846f2feba0e33acc9292b8dd16947de404be5` with unrelated modified files
(`mobile/daemon.py`, `notifications/delivery.py`, `notifications/store.py`);
that movement touched neither the pinned source nor the target bytes.

Guidance read at the pin: `i18n/glossary.md` (`ef07b446…fd6fdcd`) and
`i18n/style-guides/fr.md` (`b916f3d1…d52ec39`).

Cross-surface inputs (terminology parity): desktop
`fr/ui.notifications.json` `c9176eb1…6ebfc6e50c` (identical hash to the copy the
sibling `wire.slash.notifications_page` audit read), its `en` and `context`
sidecars; core `fr/wire.slash.json` (`361f316d…0282dab9`) and
`fr/wire.errors.notifications.json` (`a8b19b6c…9b672122`), both unchanged
between the pin and HEAD.

## What actually ran

`audit.py` (retained as `audit.py.txt`) imported the repo's own modules from the
pinned tree — `local_operator.i18n.runtime`, `catalogue`/`scripts.i18n.check`
(`_arguments_problem`, `_check_plural_categories`, `check_catalogues`),
`rich.cells.cell_len` — with `PYTHONDONTWRITEBYTECODE=1`, writing only inside
this scratch directory. `audit.log` is that run's real stdout, including one
earlier failing run: the first attempt aborted on a rig bug of mine
(`PIN_SHA` keyed `ctx`, read as `context`), and one negative control did not
fail closed because my own `replace(…, 2)` left a third `{time}` standing. Both
were repaired and the whole probe re-run; nothing was rewritten as a clean
success it was not.

Measured, not assumed:

- **144 ICU parses** (72 en + 72 fr); argument names, kinds and order identical
  to English on every key; no `number → plural` upgrade attempted.
- **Plural categories**: every literal selector is a category of fr
  (`one`/`many`/`other`; `few` rejected) and every plural has `other`.
- **`check_catalogues()` on an isolated root** holding only this namespace
  (en + fr + sidecar): 0 problems — key parity, argument contract, plural
  categories and sidecar shape all clean.
- **CLDR renders** for all 7 plural keys × 0/1/2/5/21/1000/10⁶/1 500 000: `0`
  renders `one` ("0 mise à jour ignorée", "0 non lue") per the fr rule; `many`
  fires only at exact 10⁶ multiples and correctly inserts *de* ("1 000 000 **de**
  mises à jour") while 1 500 000 correctly takes `other`.
- **Cells** for all 53 budgeted keys at realistic worst-case argument values,
  English measured the same way for comparison.
- **Typography** checked in the bytes: U+202F before `:` `?` everywhere prose
  requires it (`notifications : {error}`, `Sélection : {count}`,
  `« {source} » ?`), U+202F inside every guillemet pair, U+2019 apostrophes and
  no straight `'`, NFC holds, no manual bidi controls (fr is LTR).
- **Literal tokens** byte-exact: `/notifications`, `/unread`, `/unread read`,
  `read`, `y/n`, key letters `a`/`f`, and the `x` in `normale+` untouched.
- **Register**: `vous` throughout the 72 values — no `tu`/`ton`/`ta`/`tes`, no
  `Veuillez`; sentence case with lowercase only on compact fragments (filters,
  chips, hints), per the fr guide.
- **Nine negative controls, all fail closed**: renamed argument, dropped
  `{time}` placeholder, `few` category, deleted `many` category, non-NFC value,
  U+202E bidi, lost `/unread read` literal, overflow past `maxCells`, plain
  space before a colon.

## The fit question the manager asked: `updates_since` at 4-digit counts

**Decision: MINOR, not major.** Reasoning, with the measurements it rests on:

1. **Threshold.** fr first exceeds its 30-cell budget at count 1 000 → 31 cells;
   it stays 31 through 9 999 (the U+202F grouping keeps 4-digit spans at 5
   cells), 33 at 100 000, 38 at 10⁶. At 999 and below it fits (29). English
   never exceeds 30 anywhere up to 1 500 000 (29 at 10⁶). So no 1-, 2- or
   3-digit count overflows, and the worst realistic-ish overshoot is **one
   cell**.
2. **Realism.** The value is `store.repeat_count`, incremented once per keyed
   `notify` on the same item. In the pinned store, `HISTORY_CAP = 20` caps only
   the history rows and `NEW_ITEMS_PER_HOUR = 30` caps only *new* items, so the
   counter itself is unbounded in code — but reaching 1 000 requires 1 000
   updates to one item. A long agent loop on a single key reaches tens, not
   thousands.
3. **Consequence.** This string is one segment of the detail's meta line
   (`" · ".join(meta)` — source, age, `updates_since`, seen) appended to a
   `Text(no_wrap=False)` in the pinned `notifications_render.py`. The pane
   **wraps**; nothing is cropped, no column overruns, no control loses its
   label. The only hard cell budget in this namespace is the 5-cell row time
   column (`TIME_CELLS = 5`), and `row_now` respects it at 3 cells.
4. **Budgets here are nominal.** English itself exceeds `header_counts`' 40-cell
   budget at 10⁶ (43 cells), and fr's header breach first appears at the same
   count (44 cells) — so the header is not an fr regression, and a one-cell
   overshoot at a count no surface reaches is not a translation defect.

The fr `updates_since` value is, additionally, **byte-identical to the already
audited desktop string** `ui.notifications.detail.meta.updates_since`
(`c9176eb1…`), so changing it would create a cross-surface divergence to fix a
one-cell wrap. What would change this verdict: a breach at 3-digit counts, a
non-wrapping or cropped container for the meta line, or evidence that
`repeat_count` reaches 1 000 in operator use. Optional headroom if the setter
wants it: `{count, plural, one {# mise à jour} many {# de mises à jour} other
{# mises à jour}} ({time})` measures 28 cells worst case over 1…100 000
(`9 999 mises à jour (14:05)`) at the cost of dropping *depuis*.

## Findings

### FX-NTT-1 — MINOR — `tui.notifications.kind_updates`, `tui.notifications.update_kind`

- Current: `infos` / `info`
- Drift: the same English concept is `Mises à jour` / `Mise à jour` in the audited
  desktop copy (`ui.notifications.filter.kind_updates`, `detail.kind.update`),
  and `mise à jour` / `mises à jour` in **five** values of this very catalogue
  (`confirm_dismiss_session`, `dismissed`, `empty_none_body`, `updates_since`,
  and the `Mises à jour précédentes` section header).
- Verified claim: the translator's "`mises à jour` does not fit 8 cells" is
  true — it renders **12 cells**, and `mise à jour` **11**, against `maxCells`
  8. No full-form rendering is possible in either slot.
- Proposed (optional): `majs` (4 cells) / `maj` (3) keep the term root; rejected
  as the default because `maj` is informal register and the glossary carries no
  entry endorsing it.
- Disposition: **accept as declared**, non-blocking; drift recorded with both
  spellings.

### FX-NTT-2 — MINOR — `tui.notifications.updates_since`

- Current: `{count, plural, one {# mise à jour depuis {time}} many {# de mises à jour depuis {time}} other {# mises à jour depuis {time}}}`
- Renders 31 cells at counts 1 000–9 999 against `maxCells` 30; 999 and below
  fit. Detail, reasoning and the optional shorter form: see the section above.
- Disposition: **accept**; no change required to ship.

### FX-NTT-3 — MINOR — `tui.notifications.seen_live`

- Current: `vue aussitôt` (12/12 cells)
- Drift: desktop `ui.notifications.seen.live` = `Vue en direct`, which measures
  **13 cells** and therefore cannot be used here; the desktop's exact term is
  excluded by the budget, and `vue aussitôt` preserves the sense (seen as it
  arrived) rather than the desktop's wording.
- Disposition: **accept**; drift recorded (`vue aussitôt` vs `Vue en direct`).

### FX-NTT-4 — NIT — `tui.notifications.seen_in_chat`

- Current: `vue en conv.` (12/12) vs desktop `Vue dans la conversation` (25
  cells). `conv.` is the standard French abbreviation and the sense is intact.
- Disposition: accept.

### FX-NTT-5 — NIT — `tui.notifications.arg_read`

- Current: `Marquer toutes les notifications comme lues` (43/44)
- English: `Mark every unread notification read` — the English restricts the
  scope to unread; the French says all notifications. No behavioural
  difference (already-read rows are no-ops), and the wording matches the
  desktop's own inclusive formula `action.mark_all_read` = `Tout marquer comme
  lu`. If literal parity is wanted, `Marquer les notifications non lues comme
  lu` measures **43 cells** and fits.
- Disposition: accept; option recorded, not required.

### FX-NTT-6 — NIT — `tui.notifications.header_partial`

- Current: `{count} visibles` → renders `100 visibles` (12/12). English
  `first {count}` carries the "first" nuance, which is dropped.
- Measured alternatives: `100 premières` = **13 cells** (over budget);
  `100 premiers` = 12 but masculine, disagreeing with *notifications*
  (feminine). The current text is the best available inside 12 cells, so this
  is a budget limit, not a translation error.
- Disposition: accept, no change possible without a budget change.

## Translator's declared choices, verified one by one

| Declared choice | Verified |
|---|---|
| `row_now` = `<1m` in a 5-cell column | 3 cells ≤ 5; matches the column's other compact ages (`4m`, `3h`, `2d`) |
| `hint_select` = `cocher` | 6 ≤ 8; `sélectionner` measures 12 and does not fit |
| `hint_read`/`hint_unread` = `lu` / `non lu` | 2 and 6 ≤ 8; mirrors the English adjective pair and the desktop's `comme lu` / `comme non lu` |
| `seen_live` = `vue aussitôt`, `seen_in_chat` = `vue en conv.` (12 cells) | both exactly 12/12; spelled-out forms 13 and 25 — abbreviation forced, recorded as FX-NTT-3/4 |
| `kind_updates`/`update_kind` = `infos`/`info` (8 cells) | `mises à jour` 12, `mise à jour` 11 — full form impossible; recorded as FX-NTT-1 |
| `header_counts` waiting = `en attente` | renders `1 en attente`; same elliptical form as the audited desktop `rail.aria_with_updates` / `nav.unread` family |
| Typography: `’`, `« »`, U+202F before `:` `?` and inside guillemets | confirmed in all 72 values byte-wise; no straight apostrophes; no U+202F inside `/notifications`, `/unread`, `read`, `y/n` |
| `updates_since` reaches 31/30 at 9 999; header and `updates_since` overflow at 10⁶, English also overflows the header | confirmed exactly: 31 at 9 999; 44 (fr) and 43 (en) at 10⁶; `updates_since` 38 at 10⁶ |

## Clean areas worth naming

Action distinctness is intact everywhere it matters: dismiss (`ignorer`),
mark read (`marquer comme lu` / `lu`), undo (`annuler`), select (`cocher`),
answer (`répondre`), open (`ouvrir`) are six separate actions with six
distinct labels — no two controls collapse into the same French word. The
`ignorer` / `annuler` pair matches the desktop's already-audited mapping.
`pointer_unread` (`Les tours terminés sont désormais dans /unread.`) matches the
core `wire.slash.notifications` vocabulary (`Tours terminés non lus`), and the
page name stays **Notifications** — never "Boîte de réception". `asks_stay`,
`stale_skipped`, `dismissed` and `updates_since` all supply the `many` branch
with the obligatory *de* and keep `#` as the number. `empty_none_title`
(`Pas encore de notification`) diverges from the desktop's `Aucune notification
pour l’instant` only because the latter measures 34 cells against a 30-cell
budget — the shorter form is correct here and is not a finding.

## Limitations

- LLM linguistic judgement, not a native-speaker attestation.
- Static copy/ICU audit: no rendered frame, TUI screenshot, runtime-service or
  release-eligibility claim. `maxCells` comparisons are `rich.cells.cell_len`
  numbers on real renders, not screenshots.
- No ledger edit; this audit gates nothing by itself — the manager owns the
  ledger row.
