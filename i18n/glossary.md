# Glossary — one term, one decision, for every locale

Translation invariant per term: the SAME rendered word in every locale unless a
row says otherwise. Translators and auditors cite this file; the ledger's
audit step is where a violation is caught. Terms are added by the wave that
first needs them (wave 1: es/fr) — this skeleton records only the SHARED
decisions that never wait for a wave.

## Product names (never translated)

| Term | Decision |
|---|---|
| Local Operator | keep as-is |
| `lop`, `lo` | keep as-is (commands) |
| `omp` / oh-my-pi | keep as-is (external product) |
| tool names (`bash`, `read`, `edit`, …) | keep as-is — they are the model's vocabulary |
| `--flags`, config keys, file paths, identifiers | keep as-is |

## Common nouns (translated)

Default rule (§6): translate common nouns (`session`, `wake`, `agent`, `trace`,
`project`, `subagent`), keep product-surface names. A wave may record a
per-locale exception here with its reason.

| Term | Notes |
|---|---|
| session | translate |
| wake | translate (it is not a product name) |
| agent / subagent | translate; `agent` as a ROLE name in `task(agent=…)` stays |
| trace | translate |
| tool | translate; specific tool names stay English |

## Formatting decisions

| Topic | Decision |
|---|---|
| digits | Latin digits for `hi`/`ur` in v1 (§2.8); per-locale style guide may revisit |
| formality | per locale, in `style-guides/<locale>.md` (tu/vous, Sie/du, 您/你) |
| RTL | `ur` ships best-effort in the TUI (FSI/PDI isolation); full mirroring web/Electron/RN |
| punctuation | follow the locale's own conventions inside a translated sentence; never translate inside code spans |

## How to change this file

One row per decision, with enough context that a translator who never saw the
discussion lands on the same answer. A change here is a repo-artifact change:
it does not need regenerating (it is not generated), but the translation
ledger's freshness rule treats it as style-guide scope — a wave re-audits the
terms it cites when either changes.
