# Independent audit — French `wire.errors.media`

| | |
|---|---|
| locale | `fr` |
| namespace | `wire.errors.media` — 20 keys, all read |
| audit id | `frbk1` |
| audit date | 2026-10-29 |
| audited revision | `9495f8a6c3bf2b5438ef5c33d7274cf519435dda` (branch `feat/i18n-core-fr-backfill`; the target catalogue is an untracked working-tree file, so its bytes are bound by sha256) |
| English source | `en/wire.errors.media.json` — sha256 `1c109f63b8d3ce00657a47836199ce8ff924bb8e4bd999fda9f1d64f760a823b` |
| context sidecar | none exists for this namespace |
| French target | `fr/wire.errors.media.json` — sha256 `222d3cf6667d23a9b8de6e44eaddaf9fd7e5f155eb53365bc43e79d6fd921010` |
| auditor | `translation-auditor` (lopdev team), session `500e883ce1b1`, not the translator; edited nothing in the repository |
| model | `radient/auto` — provider fallbacks: anthropic/claude-sonnet-5-5 (primary) -> openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto, each move caused by a provider quota failure; radient/auto served every command in this audit |
| **verdict** | **findings** — 0 Blocker, 0 Major, 0 Minor, 1 Nit (no Blocker and no Major, so the namespace is not blocked; the findings below are open and reported, not accepted) |

## What these strings are

Speech and transcription error details, including upstream failures, credit and key refusals, and the sign-in prompt that enables speaking aloud.

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

### F4 — Nit — `wire.errors.media.speech_upstream_no_response`

- **Current**: La génération de parole a échoué en amont : aucune réponse de l’amont : {error}
- **Proposed**: La génération de parole a échoué en amont : aucune réponse du service amont : {error}
- **Reason**: The sentence already says 'a échoué en amont', then repeats the same adverb as a noun: 'aucune réponse de l’amont'. The English 'No response from the upstream' is itself clumsy, but French has a clean noun here ('le service amont') and using it removes the doubled 'amont' without changing what is reported.
- **Disposition**: open — reported to the requesting session; not applied by the
  auditor, who edits nothing.

## Mechanical results

| check | result |
|---|---|
| key parity | en 20 / fr 20; missing 0, extra 0 |
| key order | identical to English |
| ICU arguments | 11 keys carry arguments, 0 tuple mismatches |
| plurals | 0 keys carry a plural selector |
| typography | 0 prose anomalies; 0 hits confined to quoted/code spans |
| placeholder/identifier loss | 0 key(s) |
| mask marker in a French value | 0 |
| `check.py --catalogues-only` | `check.py: catalogue checks: 0 problem(s)` (exit 0) |

## Terminology and corpus control

'speech' renders as **parole**, matching the shipped `wire.settings.speech_voice_pace.help` ('Rythme de parole'); 'fournisseur', 'clé API' and 'crédits' match shipped usage. `Radient Console` stays English as a surface name.

## Verdict

`wire.errors.media` — **findings** at 20 keys read. no Blocker and no Major, so the namespace is not blocked; the findings below are open and reported, not accepted.

Limitations: this is an LLM audit, not a native-speaker review, and no native review is
claimed. No rendered capture was taken — these are JSON wire payloads and settings copy
with no surface pinning them to a width, so fit is judged from measured cells plus any
declared `maxCells`. The corpus control is the tracked `fr` set only; the other 24
untracked backfill catalogues are other auditors' scope and were not used as precedent.
