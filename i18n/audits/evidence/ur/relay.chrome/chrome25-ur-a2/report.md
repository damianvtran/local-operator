# Terminal delta audit — `ur/relay.chrome.json` (`chrome25-ur-a2`)

**Verdict: TERMINAL PASS** — 0 blocker, 0 major. The 8-key delta is verified against the
before-snapshot the remediation record names, the remaining 75 keys are byte-identical to the bytes
`chrome25-ur-a1` audited, and no new out-of-scope issue was introduced. Four non-semantic residuals
(two width signals, one budget deferral, one cross-namespace naming decision) go to the rendered pass.

| Item | Value |
|---|---|
| Locale / namespace | `ur` / `relay.chrome` (83 keys) |
| Audit id | `chrome25-ur-a2` (terminal delta of `chrome25-ur-a1`) |
| Repository | `~/local-operator/.worktrees/relay-chrome-i18n`, HEAD `535cea359825411741069cbffcee42e2927df9d3` |
| Auditor | agent `translation-auditor` (team `lopdev`), session `439789cb67e3` |
| Model (runtime header) | `radient/auto` — the header's model string only; the provider/model behind the auto route is not exposed, so no underlying model id is inferred or claimed |
| Fallback | none observed in this session (no fallback or incident notice) |
| Independence | same auditor as a1, now verifying the remediator's bytes; the remediation record was read for its before-snapshot only, and every hash below was recomputed from the bytes, not copied from prose |

## 1. Inputs and hashes (all recomputed here)

| Artefact | SHA-256 | Status |
|---|---|---|
| `en/relay.chrome.json` | `f30049d1ec7bce1d0586d8d5d380368e64fb0ac841d91b9b04c1625d17688aac` | unchanged vs R1 |
| `en/relay.chrome.context.json` | `1c727d1c6c8ddff7c33cbd3abb1ed716e89ab84c864cdb9c319b5df2595fb69a` | unchanged vs R1 |
| before-snapshot (`candidate-before.json`, remediation record) | `6059005113786f1d74fc848a9f254fadd5fba259be11aecfcbf59ebe2cd24d28` | equals the R1-audited target |
| **current target** `ur/relay.chrome.json` | **`a90b9bea54c17e3b74527bb7c503363df348d1d625b105431ff8bbed56f39fff`** | matches the remediation record's after-hash |

83 keys, identical key order to English, UTF-8, no BOM, trailing newline. The remediation record's
after-hash and this recomputation agree, so the ledger can bind to `a90b9bea…` rather than to prose.

## 2. Scope and R1 carry-forward

- **Delta = exactly 8 keys**, the declared list byte-for-byte: `drawer.status`, `link.since_seconds`,
  `link.since_minutes`, `list.starting`, `row.saving`, `appearance.group_by_tag`, `session.resuming`,
  `session.degraded`.
- **75 keys unchanged** — byte-identical to the R1-audited snapshot, whose hash equals
  `chrome25-ur-a1`'s target (`605900…`). R1 coverage therefore transfers without re-reading, and the
  18 direct safety/ICU neighbours were re-checked to be byte-identical (§5).
- No tracked file was modified by the remediation: `git diff` is empty; the only untracked paths are
  the seven locale candidates.

## 3. Changed-key review

| Key | Before | After | Disposition |
|---|---|---|---|
| `drawer.status` (`retrying`) | `{state, select, online {منسلک · یہی فون} retrying {آف لائن · دوبارہ کوشش} other {منسلک ہو رہا ہے…}}` | `… retrying {آف لائن · دوبارہ کوشش ہو رہی ہے} …` | **F3 fixed** — 31/32 cells; branch names `online`/`retrying`/`other` unchanged |
| `link.since_seconds` | `… · # سیکنڈ پہلے` | `… · سنک # سیکنڈ پہلے` | **F8 + MAJ-01 fixed** — 43/44 at a three-digit count (42 at two); `#` in both branches |
| `link.since_minutes` | `… · # منٹ پہلے` | `… · سنک # منٹ پہلے` | **F8 + MAJ-02 fixed** — 41/44; `#` in both branches |
| `list.starting` | `شروع…` | `شروع ہو رہا…` | **F2 fixed** — 12/12; progressive matches every other busy label |
| `row.saving` | `محفوظ…` | `محفوظ ہو رہا…` | **F1 fixed** — 13/12, deliberate +1 cell (residual 1) |
| `appearance.group_by_tag` | `گفتگوئیں ٹیگ سے گروپ کریں` | `گفتگوئیں ٹیگ کے حساب سے گروپ کریں` | **F5 fixed** — 33/36; Urdu instrumental |
| `session.resuming` | `کھل رہا ہے…` | `جاری ہو رہا…` | **F7 fixed** — 12/12; same verb as the `جاری رکھیں` button |
| `session.degraded` | `جواب نہیں دے رہا — پیغام روکے ہیں` | `جواب نہیں دے رہا — پیغامات رکے ہیں` | **F4 + MIN-09 fixed** — 34/36; plural subject + intransitive `رکے`, no delivery claim |

Language check of the new text (native reading, not just mechanics):

- `آف لائن · دوبارہ کوشش ہو رہی ہے` — a state, not a retry control; correct progressive.
- `سنک # سیکنڈ پہلے` / `سنک # منٹ پہلے` — the number now has an explicit referent ("synced # ago")
  and the placeholder renders inside the phrase (`سنک 12 سیکنڈ پہلے`). `سنک` is the corpus's own
  settled rendering for *sync* (`wire.settings.network_sync_debounce_s.label` = `سنک خاموش مدت (s)`).
  Better than my own `آخری اپ ڈیٹ` suggestion: no invented mid-string ellipsis and two cells cheaper.
- `شروع ہو رہا…`, `محفوظ ہو رہا…`, `جاری ہو رہا…` — one busy-state pattern across the namespace.
- `گفتگوئیں ٹیگ کے حساب سے گروپ کریں` — idiomatic instrumental; `گروپ کریں` retained.
- `پیغامات رکے ہیں` — grammatical agreement, intransitive "held/waiting", no promise of delivery.

## 4. Every finding, as dispositioned

| ID | Sev. | Key | Disposition | Verdict of this audit |
|---|---|---|---|---|
| F1 | minor | `row.saving` | fixed | **accepted** — wording correct; 13/12 is a width signal for the rendered pass, not a semantic defect |
| F2 | minor | `list.starting` | fixed | **accepted** — 12/12 |
| F3 | minor | `drawer.status` | fixed | **accepted** — branch integrity re-validated |
| F4 | minor | `session.degraded` | fixed (`پیغامات رکے ہیں`) | **accepted** — the remediator's repair is better than my `پیغام قطار میں ہے`: the English does not assert a queue, and the plural + intransitive fixes the actual defect |
| F5 | minor | `appearance.group_by_tag` | fixed | **accepted** |
| F6 | minor | `drawer.library` | rejected | **accepted with reasoning** — the drift is in `wire.settings` prose (different shipped, already-audited namespace); the glossary is silent and the guide allows settled loanwords. Recorded as a corpus/glossary decision; no churn requested inside this wave |
| F7 | nit | `session.resuming` | fixed | **accepted** |
| F8 | nit | `link.since_seconds`, `link.since_minutes` | fixed via `سنک` | **accepted** — resolves the missing referent without the punctuation I had proposed |
| F9 | nit | `list.filter.needs_you` | deferred (rendered fit) | **accepted** — width-only; wording unchanged and natural |
| MAJ-01 / MAJ-02 (safety stream) | major | `link.since_seconds`, `link.since_minutes` | fixed | **verified fixed on head** — data-freshness referent present, both branches keep `#` |
| MIN-09 (safety stream) | minor | `session.degraded` | fixed | **verified fixed on head** |
| NIT-01 (safety stream) | nit | `list.mark_all` | rejected | **accepted** — the grammatical 23-cell form exceeds the 20-cell header-action budget; the retained form is 20/20 and my R1 audit did not raise it |

Thirteen finding IDs (twelve rows — `MAJ-01`/`MAJ-02` share one row). No finding was accepted
without reading the current bytes, and no rejected finding was re-opened as a cosmetic request.

## 5. Safety and neighbour re-check on head

Byte-identical to the audited snapshot (carry-forward): `session.tray.approve`,
`session.tray.approve_label`, `session.resume`, `session.ended`, `session.ended_hint`,
`session.subtitle.not_answering`, `session.subtitle.waiting`, `session.tray.questions`,
`session.tray.tasks`, `session.tray.agents`, `session.tray.agents_failed`, `list.marking`,
`list.connecting`, `list.filter.needs_you`, `row.pin`, `row.unpin`, `appearance.on`,
`appearance.off`.

Semantics, re-read on head:

- **Approve** `منظور کریں`; its accessible name still **starts with the same word** and keeps the em
  dash (`منظور کریں — ایک درخواست آپ کی منتظر ہے`).
- **Resume** `جاری رکھیں`; the busy state is now `جاری ہو رہا…` — same verb, so the pair reads as one
  action.
- **Ended** `سیشن ختم ہو گیا · تاریخچہ محفوظ`; **ended hint** still says a send or the quoted
  `جاری رکھیں` reopens it in the **home** folder (not the project folder).
- **Degraded** now `پیغامات رکے ہیں` — held/waiting, no claim that anything was lost or guaranteed.

## 6. Checks actually executed (output in `audit.log`)

- **Structure, all 83 keys:** argument names + kinds, **select branch names**, and plural branch
  shape `(selector, empty, renders_value)` compared to English → **0 mismatches**. This closes, for
  this file, the two gaps the R1 audits named in `scripts/i18n/check.py` (renamed select branch,
  dropped plural category).
- **Renders** (`messages.render(locale="ur")`): status `online`/`retrying`/`other`/unknown-token →
  `منسلک · یہی فون` / `آف لائن · دوبارہ کوشش ہو رہی ہے` / `منسلک ہو رہا ہے…` (unknown falls back to
  `other`); `سنک 0/1/2/12/100 سیکنڈ پہلے` and `… منٹ پہلے`; the five changed plain strings; the four
  safety strings.
- **Width** (ur style-guide proxy, worst-case params): status 31/32, since_seconds 43/44,
  since_minutes 41/44, starting 12/12, **row.saving 13/12**, group_by_tag 33/36, resuming 12/12,
  degraded 34/36, needs_you 17/16 (deferred), mark_all 20/20.
- **Repository check:** `python scripts/i18n/check.py` → exit 0,
  `catalogues consistent` (182 files scanned; 34 pre-existing advisory ratchet lines).
- **Script/bidi hygiene:** same Urdu inventory as R1 (U+06A9/U+06CC/U+06C1/U+06D2 present, Arabic
  `ي/ك/ه/ة` absent), no bidi controls, no native digits, no danda, no `!`; `·`×12, `—`×2, `…`×9,
  `“”`×4, `،`×1 unchanged.
- **Working tree:** `git diff` empty; only the seven untracked locale candidates.

## 7. Residuals for the rendered pass (non-gating)

1. `row.saving` — **13 vs 12 cells**; accepted deliberately (honest "saving…" over a misleading
   "saved"); confirm fit in the action sheet and the 72 px swipe tile.
2. `list.filter.needs_you` — 16 cells at one digit, **17 vs 16 at two digits**; wording unchanged,
   layout only; the chip row scrolls per the sidecar.
3. `list.mark_all` — 20/20 telegraphese; the fully grammatical 23-cell form needs a budget raise
   (sidecar decision, not a copy change).
4. `drawer.library` `لائبریری` vs untranslated `Library` in `wire.settings` prose — corpus/glossary
   decision for the owner.

## 8. Limitations

- No rendered frame/device evidence in this pass; the two width residuals are signals awaiting the
  rendered fit, not semantic findings.
- The web `IntlMessageFormat` path was not executed (package absent; no scratch installs). The web
  runtime maps `ur`→RTL and delegates plural selection to `Intl.PluralRules` (`one`/`other`,
  `0`→`other`), the same categories validated against the repository table.
- The independent **safety stream's own delta sign-off** is a separate artefact and is **not claimed
  here**; this report verifies those keys' bytes and semantics within the primary audit's scope.

## 9. Verdict

| Scope | Verdict |
|---|---|
| `drawer.status` | PASS (F3 fixed) |
| `link.since_seconds`, `link.since_minutes` | PASS (F8, MAJ-01/02 fixed) |
| `list.starting` | PASS (F2 fixed) |
| `row.saving` | PASS (F1 fixed; width residual) |
| `appearance.group_by_tag` | PASS (F5 fixed) |
| `session.resuming` | PASS (F7 fixed) |
| `session.degraded` | PASS (F4, MIN-09 fixed) |
| 75 unchanged keys | PASS by hash-bound carry-forward from `chrome25-ur-a1` |
| **`relay.chrome` (ur, head `a90b9bea…`)** | **TERMINAL PASS — 0 blocker, 0 major, no new findings** |
