# Independent audit — French `wire.exec`

| | |
|---|---|
| locale | `fr` |
| namespace | `wire.exec` — 17 keys, all read |
| audit id | `frbk1` |
| audit date | 2026-10-29 |
| audited revision | `9495f8a6c3bf2b5438ef5c33d7274cf519435dda` (branch `feat/i18n-core-fr-backfill`; the target catalogue is an untracked working-tree file, so its bytes are bound by sha256) |
| English source | `en/wire.exec.json` — sha256 `041ca802dd4e1329cf1f1b2984311c628f314223f5941b35a7007f5b3d198c94` |
| context sidecar | none exists for this namespace |
| French target | `fr/wire.exec.json` — sha256 `d097a33c1b83a499c732a012c6256de412fa13136b4c57fee09049fc1e9c9fa4` |
| auditor | `translation-auditor` (lopdev team), session `500e883ce1b1`, not the translator; edited nothing in the repository |
| model | `radient/auto` — provider fallbacks: anthropic/claude-sonnet-5-5 (primary) -> openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto, each move caused by a provider quota failure; radient/auto served every command in this audit |
| **verdict** | **findings** — 0 Blocker, 0 Major, 1 Minor, 0 Nit (no Blocker and no Major, so the namespace is not blocked; the findings below are open and reported, not accepted) |

## What these strings are

`lop exec` CLI text: help, epilog, warnings and pointers. Mostly developer output on a terminal, so flags, examples and stream routing carry weight.

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

### F3 — Minor — `wire.exec.turn_did_not_complete`

- **Current**: le tour n’est pas allé au bout
- **Proposed**: le tour ne s’est pas terminé
- **Reason**: 'n’est pas allé au bout' is colloquial where the shipped fr corpus renders a turn that ran to its end with 'terminé' — 'les tours terminés', 'Tour condensé', 'plutôt que terminé ou perdu' — and the fr style guide asks for terse, neutral statuses. The noun 'tour' itself is correct and matches the corpus (37 occurrences); only the verb phrase drifts in register.
- **Disposition**: open — reported to the requesting session; not applied by the
  auditor, who edits nothing.

## Mechanical results

| check | result |
|---|---|
| key parity | en 17 / fr 17; missing 0, extra 0 |
| key order | identical to English |
| ICU arguments | 8 keys carry arguments, 0 tuple mismatches |
| plurals | 0 keys carry a plural selector |
| typography | 0 prose anomalies; 14 hits confined to quoted/code spans |
| placeholder/identifier loss | 0 key(s) |
| mask marker in a French value | 0 |
| `check.py --catalogues-only` | `check.py: catalogue checks: 0 problem(s)` (exit 0) |

## Terminology and corpus control

'turn' renders as **tour**, the shipped term (37 occurrences, e.g. 'les tours terminés'). CLI flags, the `NAME[,NAME]` metavar, the shell examples in `exec_epilog`, `stdout`/`stderr`, `SESSION_ID`, `JOB_ID` and `tool_approval_mode: auto` are all byte-identical to English, which is required. All fourteen English sentences of `exec_epilog` are present in the French.

## Verdict

`wire.exec` — **findings** at 17 keys read. no Blocker and no Major, so the namespace is not blocked; the findings below are open and reported, not accepted.

Limitations: this is an LLM audit, not a native-speaker review, and no native review is
claimed. No rendered capture was taken — these are JSON wire payloads and settings copy
with no surface pinning them to a width, so fit is judged from measured cells plus any
declared `maxCells`. The corpus control is the tracked `fr` set only; the other 24
untracked backfill catalogues are other auditors' scope and were not used as precedent.
