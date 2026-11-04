# Independent audit — French `wire.slash.library`

| | |
|---|---|
| locale | `fr` |
| namespace | `wire.slash.library` — 2 keys, all read |
| audit id | `frbk1` |
| audit date | 2026-10-29 |
| audited revision | `9495f8a6c3bf2b5438ef5c33d7274cf519435dda` (branch `feat/i18n-core-fr-backfill`; the target catalogue is an untracked working-tree file, so its bytes are bound by sha256) |
| English source | `en/wire.slash.library.json` — sha256 `0f8615f2ed9ffaf3aeb0288a4ff4b8fc8becd46017a931db9d4b3d9ec32fce46` |
| context sidecar | none exists for this namespace |
| French target | `fr/wire.slash.library.json` — sha256 `fcfeec0d1705a9a20cdeda8fad39daf423f5105a67ae2f7b1c440c5b650107e9` |
| auditor | `translation-auditor` (lopdev team), session `500e883ce1b1`, not the translator; edited nothing in the repository |
| model | `radient/auto` — provider fallbacks: anthropic/claude-sonnet-5-5 (primary) -> openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto, each move caused by a provider quota failure; radient/auto served every command in this audit |
| **verdict** | **findings** — 0 Blocker, 0 Major, 0 Minor, 1 Nit (no Blocker and no Major, so the namespace is not blocked; the findings below are open and reported, not accepted) |

## What these strings are

The `/library` slash-command help lines: browsing the global asset library and driving the Librarian.

## Method

Read `i18n/glossary.md`, `i18n/style-guides/fr.md` and `docs/I18N.md`, then read the
English and the French side by side key by key, in English key order. Every mechanical
claim below was re-derived here, not taken from a translator's summary: ICU argument
tuples and plural selectors through the repository's own parser
(`local_operator.i18n.runtime.message_arguments` / `plural_selectors`), typography by
codepoint, identifier preservation by token, and width through `rich.cells.cell_len`
against any `maxCells` the sidecar declares. Terminology was checked against the 64
tracked `fr` catalogues (2,262 keys) as the corpus control, and every candidate finding
was tested against that corpus before being reported — several first-pass suspicions
were dropped because the corpus already settles them (recorded at the end).

## Findings

### F7 — Nit — `wire.slash.library.help`

- **Current**: Parcourir la Bibliothèque globale d’éléments
- **Proposed**: Parcourir la Bibliothèque globale
- **Reason**: The shipped fr corpus names this surface 'la Bibliothèque globale' (wire.settings.librarian.description) and 'Bibliothèque' alone as the drawer label (relay.chrome.drawer.library); this help string invents a third form, 'Bibliothèque globale d’éléments'. The capital B and the 'Bibliothèque'/'Bibliothécaire' naming are correct and match the corpus — only the appended qualifier is a new variant, and dropping it also shortens the line for the /help column.
- **Disposition**: open — reported to the requesting session; not applied by the
  auditor, who edits nothing.

## Mechanical results

| check | result |
|---|---|
| key parity | en 2 / fr 2; missing 0, extra 0 |
| key order | identical to English |
| ICU arguments | 0 keys carry arguments, 0 tuple mismatches |
| plurals | 0 keys carry a plural selector |
| typography | 0 prose anomalies; 0 hits confined to quoted/code spans |
| placeholder/identifier loss | 0 key(s) |
| mask marker in a French value | 0 |
| `check.py --catalogues-only` | `check.py: catalogue checks: 0 problem(s)` (exit 0) |

## Terminology and corpus control

'Library' and 'Librarian' are named surfaces and are capitalised in the shipped corpus (`wire.settings.librarian.description` = 'Le Bibliothécaire garde la Bibliothèque globale organisée', `relay.chrome.drawer.library` = 'Bibliothèque'), so the capitals here are conformant. Width note: both lines sit inside the `/help` column budget — 44 and 51 cells against the guide's ~55. Recorded, not a finding: two shipped fr strings are the corpus outliers on this very point — `wire.errors.desktop.librarian_disabled` ('Le bibliothécaire est désactivé') and `wire.errors.desktop.library_asset_not_found` ('dans la bibliothèque') lowercase a surface their own English capitalises; the audited strings follow English and the majority, so those outliers are outside this namespace.

## Verdict

`wire.slash.library` — **findings** at 2 keys read. no Blocker and no Major, so the namespace is not blocked; the findings below are open and reported, not accepted.

Limitations: this is an LLM audit, not a native-speaker review, and no native review is
claimed. No rendered capture was taken — these are JSON wire payloads and settings copy
with no surface pinning them to a width, so fit is judged from measured cells plus any
declared `maxCells`. The corpus control is the tracked `fr` set only; the other 24
untracked backfill catalogues are other auditors' scope and were not used as precedent.
