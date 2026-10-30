# Independent audit — French `wire.errors.agents`

| | |
|---|---|
| locale | `fr` |
| namespace | `wire.errors.agents` — 44 keys, all read |
| audit id | `frbk1` |
| audit date | 2026-10-29 |
| audited revision | `9495f8a6c3bf2b5438ef5c33d7274cf519435dda` (branch `feat/i18n-core-fr-backfill`; the target catalogue is an untracked working-tree file, so its bytes are bound by sha256) |
| English source | `en/wire.errors.agents.json` — sha256 `7800f04d031a9a97222f383dea74878acadb12d9d9382b16ba638990b77627c7` |
| context sidecar | none exists for this namespace |
| French target | `fr/wire.errors.agents.json` — sha256 `087ec1d64303c4dfaa3a8669117fb4086c3b45c42420a7f5087a5a945e1992b7` |
| auditor | `translation-auditor` (lopdev team), session `500e883ce1b1`, not the translator; edited nothing in the repository |
| model | `radient/auto` — provider fallbacks: anthropic/claude-sonnet-5-5 (primary) -> openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto, each move caused by a provider quota failure; radient/auto served every command in this audit |
| **verdict** | **clean** — 0 Blocker, 0 Major, 0 Minor, 0 Nit (no finding at any severity) |

## What these strings are

HTTP and tool error details for agent, team, schedule, execution variable and hub-listing operations, raised by the local API and rendered by the desktop app and the CLI. Developer-facing refusals.

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

None. No Blocker, Major, Minor or Nit in `wire.errors.agents`.

## Mechanical results

| check | result |
|---|---|
| key parity | en 44 / fr 44; missing 0, extra 0 |
| key order | identical to English |
| ICU arguments | 32 keys carry arguments, 0 tuple mismatches |
| plurals | 0 keys carry a plural selector |
| typography | 0 prose anomalies; 8 hits confined to quoted/code spans |
| placeholder/identifier loss | 0 key(s) |
| mask marker in a French value | 0 |
| `check.py --catalogues-only` | `check.py: catalogue checks: 0 problem(s)` (exit 0) |

## Terminology and corpus control

'tenant' renders as **locataire**, which is the shipped rendering (`cli.teams_help.push_org` = 'Locataire d’organisation…'), so 'Vérifiez le locataire' is conformant rather than a French false friend. 'listing' renders as **annonce**, also shipped (`cli.agents_help.push_hub_id`). 'schedule' is **planification** per the glossary. 'skill' is **compétence**. The `''…''` in `system_prompt_required` and `variable_not_found` is the wire layer's own quoting carried byte-identically from English, not a French typography defect.

## Verdict

`wire.errors.agents` — **clean** at 44 keys read. Nothing to remediate.

Limitations: this is an LLM audit, not a native-speaker review, and no native review is
claimed. No rendered capture was taken — these are JSON wire payloads and settings copy
with no surface pinning them to a width, so fit is judged from measured cells plus any
declared `maxCells`. The corpus control is the tracked `fr` set only; the other 24
untracked backfill catalogues are other auditors' scope and were not used as precedent.
