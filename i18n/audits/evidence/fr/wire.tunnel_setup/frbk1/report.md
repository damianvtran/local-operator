# Independent audit — French `wire.tunnel_setup`

| | |
|---|---|
| locale | `fr` |
| namespace | `wire.tunnel_setup` — 5 keys, all read |
| audit id | `frbk1` |
| audit date | 2026-10-29 |
| audited revision | `9495f8a6c3bf2b5438ef5c33d7274cf519435dda` (branch `feat/i18n-core-fr-backfill`; the target catalogue is an untracked working-tree file, so its bytes are bound by sha256) |
| English source | `en/wire.tunnel_setup.json` — sha256 `3854be7dce54dd3aeb9564921ed6a50f0560b32c7ac40955d81be93acd7e2ac1` |
| context sidecar | `en/wire.tunnel_setup.context.json` — sha256 `0ff6f387f29ff89b62aa96c73aaffdb47f18ec775588043b63ea56ecd25c4457` |
| French target | `fr/wire.tunnel_setup.json` — sha256 `b79e45fca2a5bee0e23c22395c82affd57f8ce40014ff84647b97d913c8bdfb5` |
| auditor | `translation-auditor` (lopdev team), session `500e883ce1b1`, not the translator; edited nothing in the repository |
| model | `radient/auto` — provider fallbacks: anthropic/claude-sonnet-5-5 (primary) -> openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto, each move caused by a provider quota failure; radient/auto served every command in this audit |
| **verdict** | **findings** — 0 Blocker, 0 Major, 0 Minor, 1 Nit (no Blocker and no Major, so the namespace is not blocked; the findings below are open and reported, not accepted) |

## What these strings are

Desktop phone-access setup: refusals and progress. Each string has to keep the caller's next step and must not promise a rollback the code does not perform.

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

### F8 — Nit — `wire.tunnel_setup.corrupt_intent`

- **Current**: La demande de configuration du téléphone enregistrée n’est pas valide. Laissez create.json inchangé et demandez à votre agent de récupérer la demande d’origine avant de réessayer. Aucune modification de configuration n’a été apportée.
- **Proposed**: La demande de configuration enregistrée pour l’accès téléphone n’est pas valide. Laissez create.json inchangé et demandez à votre agent de récupérer la demande d’origine avant de réessayer. Aucune modification de configuration n’a été apportée.
- **Reason**: Four of the five keys in this namespace name the feature 'l’accès téléphone' ('La configuration de l’accès téléphone'); this one alone says 'la configuration du téléphone', so the same feature is named two ways inside one refusal set. English itself abbreviates here ('phone setup request' versus 'Phone access setup'), so the French is faithfully mirroring a source inconsistency, but one name across the namespace reads better.
- **Disposition**: open — reported to the requesting session; not applied by the
  auditor, who edits nothing.

## Mechanical results

| check | result |
|---|---|
| key parity | en 5 / fr 5; missing 0, extra 0 |
| key order | identical to English |
| ICU arguments | 0 keys carry arguments, 0 tuple mismatches |
| plurals | 0 keys carry a plural selector |
| typography | 0 prose anomalies; 0 hits confined to quoted/code spans |
| placeholder/identifier loss | 0 key(s) |
| mask marker in a French value | 0 |
| `check.py --catalogues-only` | `check.py: catalogue checks: 0 problem(s)` (exit 0) |

## Terminology and corpus control

'setup' renders as **configuration** throughout, matching the shipped tunnel vocabulary; `create.json` is preserved byte-identically. The refusal strings keep their imperative remedy and their warning against assuming a rollback ('ne partez pas du principe qu’un retour arrière a eu lieu'), which is the polarity the English carries.

## Verdict

`wire.tunnel_setup` — **findings** at 5 keys read. no Blocker and no Major, so the namespace is not blocked; the findings below are open and reported, not accepted.

Limitations: this is an LLM audit, not a native-speaker review, and no native review is
claimed. No rendered capture was taken — these are JSON wire payloads and settings copy
with no surface pinning them to a width, so fit is judged from measured cells plus any
declared `maxCells`. The corpus control is the tracked `fr` set only; the other 24
untracked backfill catalogues are other auditors' scope and were not used as precedent.
