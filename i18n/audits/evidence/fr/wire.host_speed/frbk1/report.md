# Independent audit — French `wire.host_speed`

| | |
|---|---|
| locale | `fr` |
| namespace | `wire.host_speed` — 2 keys, all read |
| audit id | `frbk1` |
| audit date | 2026-10-29 |
| audited revision | `9495f8a6c3bf2b5438ef5c33d7274cf519435dda` (branch `feat/i18n-core-fr-backfill`; the target catalogue is an untracked working-tree file, so its bytes are bound by sha256) |
| English source | `en/wire.host_speed.json` — sha256 `e193473d58c0de1779bcefab06d0e9b1734ee493aea30ea827e4499510282b8c` |
| context sidecar | `en/wire.host_speed.context.json` — sha256 `d5316700be4f79de5280b95f8ae96e48dfa66bef378dec38074fd3311969098d` |
| French target | `fr/wire.host_speed.json` — sha256 `14fe20f545483363af505d78f7ed61f209eada9b8467cd0dfae6a4ade5aad863` |
| auditor | `translation-auditor` (lopdev team), session `500e883ce1b1`, not the translator; edited nothing in the repository |
| model | `radient/auto` — provider fallbacks: anthropic/claude-sonnet-5-5 (primary) -> openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto, each move caused by a provider quota failure; radient/auto served every command in this audit |
| **verdict** | **findings** — 0 Blocker, 0 Major, 0 Minor, 1 Nit (no Blocker and no Major, so the namespace is not blocked; the findings below are open and reported, not accepted) |

## What these strings are

The experimental host-speed routing row: its 29-cell label and the help that explains the floor, the exclusions and the cold cache.

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

### F5 — Nit — `wire.host_speed.retire_below_tps.help`

- **Current**: Expérimental, désactivé par défaut : les preuves viennent d’un seul modèle/exécution. Un seuil positif cesse d’épingler un hôte après trois longues réponses sous ce débit de tokens de sortie par seconde, temps de consommation locale exclu. 0 efface les exclusions de débit à la prochaine requête ; les exclusions dues aux ratés de cache subsistent. Changer d’hôte peut repartir d’un cache de prompt froid.
- **Proposed**: Expérimental, désactivé par défaut : les preuves viennent d’un seul modèle/exécution. Un seuil positif cesse d’épingler un hôte après trois longues réponses sous ce débit de tokens de sortie par seconde, temps de consommation locale exclu. 0 efface les exclusions de débit à la prochaine requête ; les exclusions dues aux défauts de cache subsistent. Changer d’hôte peut repartir d’un cache de prompt froid.
- **Reason**: French computing usage settled on 'défaut de cache' for a cache miss, with 'succès de cache' as its opposite; 'raté de cache' is a coinage. The shipped fr corpus carries no precedent either way (0 hits for 'raté de cache', 'défaut de cache' or 'échec de cache'), so this is a first-use term worth settling now, not a drift against existing copy.
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

'tok/s' stays as written. 'hôte' for host matches `wire.settings.providers_openrouter_provider_affinity.help`, and 'épingler' matches its 'Réutilise l’hôte qui a servi le tour précédent'. The 28-cell French label sits inside its 29-cell cap.

## Verdict

`wire.host_speed` — **findings** at 2 keys read. no Blocker and no Major, so the namespace is not blocked; the findings below are open and reported, not accepted.

Limitations: this is an LLM audit, not a native-speaker review, and no native review is
claimed. No rendered capture was taken — these are JSON wire payloads and settings copy
with no surface pinning them to a width, so fit is judged from measured cells plus any
declared `maxCells`. The corpus control is the tracked `fr` set only; the other 24
untracked backfill catalogues are other auditors' scope and were not used as precedent.
