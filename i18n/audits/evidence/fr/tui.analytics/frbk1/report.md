# French audit — `tui.analytics` (frbk1)

- **Locale / namespace**: `fr` / `tui.analytics`
- **Audit id**: `frbk1` · **date**: 2026-10-29
- **Auditor**: translation-auditor (lopdev team), session `d5db18882cb9`, model `radient/auto`
- **Independence**: not the translator; read-only; this session edited no catalogue, ledger, allowance, generated file or code.
- **Model source**: Actual served model is the harness's auto route (radient/auto). This session received two provider-failure fallbacks during the audit: anthropic/claude-sonnet-5-5 -> openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto (reason: quota). No pinned model is claimed.
- **Audited revision**: `9495f8a6c3bf2b5438ef5c33d7274cf519435dda`; the French catalogue is an untracked working-tree file there, so the target bytes are bound by sha256, not by a commit.

## Inputs

- `en/tui.analytics.json` — sha256 `aa5221066aa84eedf0f924628d51bc8f08ccf5f26b5075638b9d40062f7b617e` · 98 keys
- `en/tui.analytics.context.json` — sha256 `04356facdf1802a34a4be9ab6d3255cc9e92826d2b4e9ae917c43f687a9325eb` · 98 entries
- `fr/tui.analytics.json` — sha256 `355bbd9d7f431fc0f088c0c5bbd452b7d1102cabe06411b401ec76e031e05f9c` · 98 keys

## Method

I read every English and French value side by side, with the sidecar role/tone/note where one exists, and judged accuracy (including what the English only implies from its role), idiom, register, glossary consistency, placeholder/ICU integrity, plural categories, typography and width.

Mechanical work was re-derived rather than trusted: key parity and order, `message_arguments()` / `plural_selectors()` argument tuples (the repo parser, not a regex), French plural categories against `local_operator/i18n/data/plural_rules.json`, whitespace signatures (leading/trailing and every run of 2+ spaces), and codepoint-level typography (U+2019 apostrophes, U+202F before `: ; ? !`, guillemets, NBSP). `scripts/i18n/check.py --catalogues-only` was run read-only; its actual output is in `audit.log`.

## Mechanical result

- Key parity: 98/98; order identical to English; no missing and no extra key.
- Placeholder/ICU argument tuples: **no diff** on any key (name and kind).
- Plurals: fr categories one/many/other; no `few`; every plural has `other` and a rendering of its own count. `many` is added wherever English has a plural (14 keys).
- Typography: apostrophes are U+2019 (no ASCII `'`), spacing before `: ; ? !` is U+202F (no ASCII space outside code spans), guillemets are « … » with U+202F inside.
- Whitespace/layout signature: **no diff** on any key (indents, separators, trailing spaces preserved).

## Findings

| ID | Severity | Key | Current | Proposed | Why |
|---|---|---|---|---|---|
| FR-AN-1 | major | `tui.analytics.hint.expand.{compact,full,full_metric,keys,keys_bare}` | ↑↓ ligne · enter déplier · e tout (33c) \| esc retour · ↑↓ ligne · enter déplier · e tout (46c) \| esc retour · t coût/tokens · ↑↓ ligne · enter déplier · e tout (62c) \| ↑↓ ligne · enter · e tout (25c) \| enter · e tout (14c) | Raise the fr budget to the measured width (33/46/62/25/14), or shorten as: enter déplier → enter ouvrir (−1 cell each tier), and drop the verb after esc in the two long tiers ("esc · ↑↓ ligne · enter ouvrir · e tout" = 38 ≤ 42; "esc · t coût/tokens · ↑↓ ligne · enter ouvrir · e tout" = 54 ≤ 57). | The sidecar pins 31/42/57/25/12 cells for the footer hint tiers, and every French tier is at or over: 33/46/62/25/14. The two long tiers can be brought inside their budgets by shortening (see proposed), but compact (31) and keys_bare (12) cannot fit three key segments in French at all — EN itself is 13 cells against keys_bare's 12 — so those two need a budget decision, not a rewording. |
| FR-AN-2 | minor | `tui.analytics.search.summary_free_paid / summary_free_paid_short` | {free} gratuite(s) sur {priced} facturable(s) · {money} payés | {free} gratuite(s) sur {priced} facturable(s) · {money} de frais | "gratuite(s)"/"facturable(s)" are feminine (agreeing with the counted operations) while "payés" is masculine plural in the same clause, and the sibling key summary_paid uses feminine "payée(s)"; the three rungs cannot all agree with one referent. |
| FR-AN-3 | nit | `tui.analytics.search.note_read_unpriced / note_search_unpriced` | … · non tarifée | … · non tarifées | The sibling keys note_*_partial and unpriced_clause use the plural "non tarifées" for the same tally; the bare-state variant switches to singular in the same row family. |

## Values identical to English

5 value(s) are byte-identical to English. Each was checked individually; all are either correct French (the word has the same spelling) or a token the guide keeps verbatim — none is an untranslated leftover.

| Key | Value | Verdict |
|---|---|---|
| `tui.analytics.chart.metric_clause` | `{active} · t → {other}` | protocol template: `{active}`/`{other}` are rendered labels and `t` is the toggle key; no prose to translate |
| `tui.analytics.search.note_join` | `{note} · {read_note}` | pure assembly template joining two already-rendered strings |
| `tui.analytics.table.cache` | `{value} cache` | `cache` is on the fr guide's kept-English list; the corpus uses it throughout |
| `tui.analytics.table.tok_s` | `{value} tok/s` | `tok/s` is a unit token with no established French rendering |
| `tui.analytics.table.tokens` | `{value} tokens` | `tokens` is kept English corpus-wide (fr guide kept-English list) |

## Cleared on purpose (not findings)

- All 51 plural occurrences across these eight catalogues (in 47 keys) carry the fr categories one/many/other, never few; every many branch that names a noun after the number uses "de"/"d'".
- Placeholder name and ICU kind tuples match English on all 98 keys (parser, not regex).
- Typographic spaces: 73 U+202F before ; : ? ! across the eight catalogues, zero ASCII-space violations outside code spans; apostrophes are U+2019 throughout (92 occurrences).

## Verdict

**failed** — 98 keys read; accuracy and idiom are strong, but four of the five footer-hint tiers exceed their declared maxCells budget and the fifth sits exactly on it.

Findings: blocker 0 · major 1 · minor 1 · nit 1. Keys read: 98.

`current_report_verdict` uses the repository's gate wording: `clean` when no blocker and no major remain, `findings` otherwise; minor and nit items are listed either way.
