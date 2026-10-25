# Independent es audit — wire.settings `compaction_aged_clear_*` delta — ac25r2

- Rounds: R1 on `62813bbd7a` (translator commit), R2 on `43da538c4d` (remediation commit; this record). Scope: the 10 new `compaction.aged_clear_*` keys of PR #2922, plus a byte check that nothing else in the namespace changed.
- Translator: i18n-translator, model anthropic/claude-opus-5-5 (per the delegating manager; translator session id not provided to this auditor).
- Auditor: i18n-auditor (lopdev team), session `d5aa5aa4c387`, model anthropic/claude-opus-5-5 (this session's model header; no fallback observed). Independent of the translator at agent/session level; SAME model — disclosed. This session wrote no catalogue value; all fixes were applied by the translator.
- Inputs read: English source + context sidecar, `i18n/glossary.md`, `i18n/style-guides/es.md`; meaning derived from `local_operator/compaction/thresholds.py`, `compaction/pruning.py` (`clear_aged_tool_results`, `aged_clear_notice`), `settings_io.py` rows, and `session/session.py` (`_clear_aged_results` → `_journal_prunes`). No MT/translation service used.

- Prior wire.settings rows for this locale stay in the ledger as history; the 818 keys outside this delta are byte-identical to `eaae289edf` (`non_aged_changed_since_base=0`), so prior acceptance carries.

## Round 1 findings and disposition

| ID | Sev | key (`wire.settings.compaction_aged_clear_…`) | R1 text | problem | proposal | disposition |
|---|---|---|---|---|---|---|
| A-es-1 | MAJOR | `enabled.label` | Borrar resultados antiguos de herramientas | 42 cells > 29-cell label budget; rendered clipped 'Borrar resultados antiguos d…' | Borrar resultados antiguos | fixed verbatim in 43da538c4d |
| A-es-2 | NIT | `batch_tokens.help / enabled.help` | …, así la caché… / Sustituye por lotes … por una nota…, así las ejecuciones… | comma splices; double 'por' | …, para que la caché de prompts se reescriba pocas veces. / Sustituye, por lotes, … de modo que las ejecuciones largas dejen de reenviarlos. | fixed verbatim in 43da538c4d |

Round-1 cross-locale totals: 0 BLOCKER, 5 MAJOR, 3 MINOR, 4 NIT. MAJORs were label overflow past the settings page's 29-cell label column (`settings_view._LABEL_BUDGET` = 34−2−2−1), measured via `truncate_cells`.

## Round 2 (executed; see audit.log, mechanical.json)

- Applied values equal the R1 proposals byte-for-byte: `fixes_match_proposals=True`; no change outside them (`changes_outside_proposals=0`).
- Key parity 828/828, missing 0, extra 0, order identical; NFC, control/bidi/zero-width 0; ICU arguments equal en (none); runtime render ok for all 10.
- The en context sidecar now carries rows for all 10 keys (labels `maxCells` 29). Context notes checked against code: model-call age, estimated-token floor, batch as a threshold, and "nothing deleted from the transcript" (aged clears journal via `append_prune`, which records the stub; the transcript rows are folded, not removed) — accurate.

| key | cells | maxCells | survives label truncation |
|---|---|---|---|
| `after_calls.help` | 97 | — | yes |
| `after_calls.label` | 22 | 29 | yes |
| `batch_tokens.help` | 127 | — | yes |
| `batch_tokens.label` | 24 | 29 | yes |
| `enabled.choice.false.description` | 30 | — | yes |
| `enabled.choice.true.description` | 26 | — | yes |
| `enabled.help` | 163 | — | yes |
| `enabled.label` | 26 | 29 | yes |
| `min_tokens.help` | 61 | — | yes |
| `min_tokens.label` | 27 | 29 | yes |

Verdict: PASSED — 0 BLOCKER, 0 MAJOR unresolved; no new findings in round 2. String-level audit only: no native terminal or rendered-frame pass is claimed.
