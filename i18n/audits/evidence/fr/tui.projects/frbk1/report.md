# French audit — `tui.projects` (frbk1)

- **Locale / namespace**: `fr` / `tui.projects`
- **Audit id**: `frbk1` · **date**: 2026-10-29
- **Auditor**: translation-auditor (lopdev team), session `d5db18882cb9`, model `radient/auto`
- **Independence**: not the translator; read-only; this session edited no catalogue, ledger, allowance, generated file or code.
- **Model source**: Actual served model is the harness's auto route (radient/auto). This session received two provider-failure fallbacks during the audit: anthropic/claude-sonnet-5-5 -> openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto (reason: quota). No pinned model is claimed.
- **Audited revision**: `9495f8a6c3bf2b5438ef5c33d7274cf519435dda`; the French catalogue is an untracked working-tree file there, so the target bytes are bound by sha256, not by a commit.

## Inputs

- `en/tui.projects.json` — sha256 `aeb473eb7de44143822962b44b6be239d63f930710e754ce5205781ca4b4ae91` · 97 keys
- `en/tui.projects.context.json` — sha256 `fa39ad8f21674a42e5218698d11396a08d47e2f82297467ad17bb1ab292bb1fe` · 97 entries
- `fr/tui.projects.json` — sha256 `8df04688ab0d3aedbf9f06d532829adf45478fce5a9b796b96b3ec39fea55f83` · 97 keys

## Method

I read every English and French value side by side, with the sidecar role/tone/note where one exists, and judged accuracy (including what the English only implies from its role), idiom, register, glossary consistency, placeholder/ICU integrity, plural categories, typography and width.

Mechanical work was re-derived rather than trusted: key parity and order, `message_arguments()` / `plural_selectors()` argument tuples (the repo parser, not a regex), French plural categories against `local_operator/i18n/data/plural_rules.json`, whitespace signatures (leading/trailing and every run of 2+ spaces), and codepoint-level typography (U+2019 apostrophes, U+202F before `: ; ? !`, guillemets, NBSP). `scripts/i18n/check.py --catalogues-only` was run read-only; its actual output is in `audit.log`.

## Mechanical result

- Key parity: 97/97; order identical to English; no missing and no extra key.
- Placeholder/ICU argument tuples: **no diff** on any key (name and kind).
- Plurals: fr categories one/many/other; no `few`; every plural has `other` and a rendering of its own count. `many` is added wherever English has a plural (1 key).
- Typography: apostrophes are U+2019 (no ASCII `'`), spacing before `: ; ? !` is U+202F (no ASCII space outside code spans), guillemets are « … » with U+202F inside.
- Whitespace/layout signature: **no diff** on any key (indents, separators, trailing spaces preserved).

## Findings

| ID | Severity | Key | Current | Proposed | Why |
|---|---|---|---|---|---|
| FR-PR-1 | major | `tui.projects.send.wake_unconfirmed` | réveil non confirmé — ne renvoyez pas ; il a peut-être abouti (61c) | réveil non confirmé — ne pas renvoyer, il a pu aboutir (54c) | The sidecar states this notice fits a 56-cell narrow box and may truncate; at 61 cells the clause that carries the actual warning ("il a peut-être abouti") is what gets cut. |
| FR-PR-2 | major | `tui.projects.send.delivery_unconfirmed` | livraison non confirmée — consultez d’abord sa transcription (60c) | livraison non confirmée — vérifiez sa transcription (51c) | Same 56-cell notice box; 60 cells truncates the verb that tells the user what to do instead of resending. |
| FR-PR-3 | minor | `tui.projects.send.no_match` | aucune cible correspondante — retour arrière pour élargir (57c) | aucune correspondance — retour arrière pour élargir (51c) | The sidecar pins this to a single row in the same picker whose sibling notices are sized to a 56-cell narrow box; at 57 cells the last word (the recovery hint) is the one that goes. |
| FR-PR-4 | minor | `tui.projects.send.legend` | tapez pour filtrer · ↑↓ déplacer · ↵ sélectionner · esc fermer (62c) | filtrer · ↑↓ déplacer · ↵ choisir · esc fermer (46c) | Fixed chrome row with no wrap allowance, +15 cells over English; the shorter form keeps every control and its verb. |
| FR-PR-5 | minor | `tui.projects.start.legend` | tapez pour filtrer · ↑↓ déplacer · ↵ sélectionner · esc fermer (62c) | filtrer · ↑↓ déplacer · ↵ choisir · esc fermer (46c) | The same legend string on the start card; identical overflow, identical fix so the two surfaces stay consistent. |
| FR-PR-6 | nit | `tui.projects.binding_zoom_out` | Temps plus grossier | Temps plus large | Pair with "Temps plus fin"; "grossier" reads as coarse/crude in French and is the only occurrence of the word in the catalogue. |

## Values identical to English

5 value(s) are byte-identical to English. Each was checked individually; all are either correct French (the word has the same spelling) or a token the guide keeps verbatim — none is an untranslated leftover.

| Key | Value | Verdict |
|---|---|---|
| `tui.projects.send.session` | `session {short}` | correct French — `session` is the French noun; `{short}` is an opaque id |
| `tui.projects.active_view` | ` · {view}` | title suffix skeleton (`· {view}`) whose value is already localized |
| `tui.projects.binding_message` | `Message` | correct French — `Message` is the same word |
| `tui.projects.form.date_placeholder` | `YYYY-MM-DD` | accepted date format token; must stay byte-identical |
| `tui.projects.form.estimate_placeholder` | `points` | correct French — `points` is the unit name in both languages |

## Cleared on purpose (not findings)

- Every placeholder, separator, leading space and digit stays put; the title suffixes keep their leading " · " and clipped-row budgets.
- "wake unconfirmed" correctly renders réveil (glossary) and "transcript" renders transcription; state_word and handle stay opaque.
- status/notice sentences that the sidecar says may wrap (notice_read_only) are allowed to exceed English length and were not flagged.

## Verdict

**failed** — 97 keys read; copy is idiomatic, but two 56-cell caution notices overflow their box and two no-wrap legend rows grow 15 cells over English.

Findings: blocker 0 · major 2 · minor 3 · nit 1. Keys read: 97.

`current_report_verdict` uses the repository's gate wording: `clean` when no blocker and no major remain, `findings` otherwise; minor and nit items are listed either way.
