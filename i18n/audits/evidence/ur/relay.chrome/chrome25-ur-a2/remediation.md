# Urdu (`ur`) — `relay.chrome` remediation, round 1 (unified)

Locale: **`ur`** (RTL). Namespace `relay.chrome`, 83 keys.
Worktree `~/local-operator/.worktrees/relay-chrome-i18n`, commit `535cea359825411741069cbffcee42e2927df9d3`.
Inputs: first locale audit `chrome25-ur-a1` (0 blocker / 0 major / 6 minor / 3 nit) and the second
independent safety audit `chrome25-safety-b1` (ur: 2 major / 1 minor / 1 nit).

Actor, as the session header shows it: role **translator** (team `lopdev`), session `d409548ee35d`,
model string **`radient/auto`** — the provider/model behind the auto route is not exposed in-session;
no fallback model was used. English and context bytes were not touched.

## Hashes

| Artefact | SHA-256 |
|---|---|
| `en/relay.chrome.json` (frozen, unchanged) | `f30049d1ec7bce1d0586d8d5d380368e64fb0ac841d91b9b04c1625d17688aac` |
| `en/relay.chrome.context.json` (frozen, unchanged) | `1c727d1c6c8ddff7c33cbd3abb1ed716e89ab84c864cdb9c319b5df2595fb69a` |
| `ur/relay.chrome.json` — **before** (audited candidate) | `6059005113786f1d74fc848a9f254fadd5fba259be11aecfcbf59ebe2cd24d28` |
| `ur/relay.chrome.json` — **after** (this round) | `a90b9bea54c17e3b74527bb7c503363df348d1d625b105431ff8bbed56f39fff` |

Snapshot of the audited candidate is kept at `chrome25-ur-remediation/candidate-before.json`
(same before-hash), so the terminal auditor can diff exactly this delta.

## Changes — 8 keys (75 unchanged)

| Key | Before | After | Cells before → after |
|---|---|---|---|
| `drawer.status` | `… retrying {آف لائن · دوبارہ کوشش} …` | `… retrying {آف لائن · دوبارہ کوشش ہو رہی ہے} …` | 21 → 31 (max 32) |
| `link.since_seconds` | `دوبارہ منسلک ہو رہا ہے · # سیکنڈ پہلے` | `دوبارہ منسلک ہو رہا ہے · سنک # سیکنڈ پہلے` | 37 → 42 (max 44) |
| `link.since_minutes` | `دوبارہ منسلک ہو رہا ہے · # منٹ پہلے` | `دوبارہ منسلک ہو رہا ہے · سنک # منٹ پہلے` | 35 → 40 (max 44) |
| `list.starting` | `شروع…` | `شروع ہو رہا…` | 5 → 12 (max 12) |
| `row.saving` | `محفوظ…` | `محفوظ ہو رہا…` | 6 → 13 (max 12) |
| `appearance.group_by_tag` | `گفتگوئیں ٹیگ سے گروپ کریں` | `گفتگوئیں ٹیگ کے حساب سے گروپ کریں` | 25 → 33 (max 36) |
| `session.resuming` | `کھل رہا ہے…` | `جاری ہو رہا…` | 11 → 12 (max 12) |
| `session.degraded` | `جواب نہیں دے رہا — پیغام روکے ہیں` | `جواب نہیں دے رہا — پیغامات رکے ہیں` | 32 → 34 (max 36) |

Widths are `rich.cells.cell_len` with `{count, number}` rendered as a two-digit count (worst case).

## Disposition of every finding

| ID | Audit | Sev. | Key(s) | Disposition | Why |
|---|---|---|---|---|---|
| F1 | a1 | minor | `row.saving` | **fixed** | bare `محفوظ` is the completed participle (“saved”) while the pin is unconfirmed; now the locale's own busy form `محفوظ ہو رہا…`. 13/12 cells — tense honesty kept over abbreviation, flagged below. |
| F2 | a1 | minor | `list.starting` | **fixed** | `شروع ہو رہا…`, matching every other busy label; 12/12. |
| F3 | a1 | minor | `drawer.status` (retrying) | **fixed** | bare noun read like a *Retry* control; now `دوبارہ کوشش ہو رہی ہے` (31/32). Branch names `online`/`retrying`/`other` unchanged. |
| F4 | a1 | minor | `session.degraded` | **fixed** | agreement and transitivity repaired with `پیغامات رکے ہیں`. F4's primary proposal `پیغام قطار میں ہے` **not** adopted: “queued” asserts a state the English does not state (wave brief forbids inventing a queued/delivered guarantee), and it keeps the singular `پیغام`. |
| F5 | a1 | minor | `appearance.group_by_tag` | **fixed** | `ٹیگ کے حساب سے گروپ کریں` — Urdu instrumental, `کے حساب سے` attested in `wire.settings`; 33/36. |
| F6 | a1 | minor | `drawer.library` | **rejected** | the drift is in `wire.settings` prose (shipped, already-audited, outside this wave); glossary §1 does not make *Library* keep-English and the guide allows settled loanwords. `لائبریری` retained; recorded as a corpus/glossary decision for the owner. |
| F7 | a1 | nit | `session.resuming` | **fixed** | `جاری ہو رہا…` — same verb as the `جاری رکھیں` button; 12/12. |
| F8 | a1 | nit | `link.since_seconds`, `link.since_minutes` | **fixed** | resolved by the safety round's form (`سنک # …`). F8's own `… · آخری اپ ڈیٹ` proposal not used: it puts a mid-string ellipsis into a string English does not punctuate that way, and `سنک` is attested in `wire.settings` (“سنک خاموش مدت (s)”). |
| F9 | a1 | nit | `list.filter.needs_you` | **deferred** (rendered fit) | wording retained — natural and matches `session.subtitle.waiting`; 16 cells at one digit (= budget), 17 at two. No copy change; the chip row scrolls. |
| MAJ-01 | b1 | major | `link.since_seconds` | **fixed** | `سنک` restores explicit data-freshness: the number now means “synced # seconds ago”, not outage duration. 42/44; `#` kept in both branches. |
| MAJ-02 | b1 | major | `link.since_minutes` | **fixed** | same fix; 40/44. |
| MIN-09 | b1 | minor | `session.degraded` | **fixed** | plural subject + intransitive `رکے` (34/36); says messages are held/waiting, makes no delivery claim. |
| NIT-01 | b1 | nit | `list.mark_all` | **rejected** | the grammatical form `سب کو پڑھا نشان زد کریں` measures **23 cells against the 20-cell header-action budget** — 3 cells of clipping risk on a header action; the retained `سب پڑھا نشان زد کریں` is 20/20 and the object marker is optional with `سب` in Urdu imperatives. Both audits call it a nit, “not a semantic error”. |

No finding was rejected for convenience; F6 and NIT-01 carry measured or scope evidence above.

## Validation actually run (after the edits)

- **Key parity:** 83/83, key order identical to English; no missing or unknown keys.
- **Structural ICU** (walks the repository parser AST, closing the two checker gaps the audits named — renamed `select` branch, dropped plural category): `structural mismatches (args, kinds, skeletons, branch selectors): none`.
- **Plurals:** `ur` `one`/`other` on all four plural messages; every branch renders `#`.
- **Renders** (`local_operator.i18n.messages.render(locale="ur")`): `سنک 1 سیکنڈ پہلے`, `سنک 12 سیکنڈ پہلے`, `سنک 2 منٹ پہلے`, `آف لائن · دوبارہ کوشش ہو رہی ہے`, `منسلک · یہی فون`, other-state `منسلک ہو رہا ہے…`, `ایجنٹس 2/3 · 1 ناکام`, `1 سوال` / `3 سوالات`.
- **Repository check:** `check.py` exit 0 —
  `check.py: ok — 182 file(s) scanned, 2253 literal(s) at or below the baseline, catalogues consistent; 34 advisory finding(s) (report-only until activated)`;
  `--catalogues-only`: `check.py: catalogue checks: 0 problem(s)`. The 34 advisories are the pre-existing
  hardcoded-literal ceilings in non-catalogue Python files (they were identical before this round).
- **Script hygiene:** no bidi control characters inserted; `·` U+00B7 and `…` U+2026 preserved; Urdu
  punctuation only; English and context hashes unchanged; no other file touched (no commit).

## Residual issues for the terminal auditor / rendered pass

1. `row.saving` is **13 cells vs a 12-cell budget** — accepted deliberately (a misleading “saved” is worse
   than one cell); needs a rendered fit check in the action sheet and the 72 px swipe tile.
2. `list.filter.needs_you` reaches **17 vs 16** at two-digit counts — unchanged wording, layout check only.
3. `list.mark_all` stays telegraphese at 20/20; the fully grammatical 23-cell form needs a budget raise
   (sidecar change, not a translation change).
4. `drawer.library` (`لائبریری`) vs untranslated `Library` in `wire.settings` prose — corpus/glossary
   decision outside this wave.
