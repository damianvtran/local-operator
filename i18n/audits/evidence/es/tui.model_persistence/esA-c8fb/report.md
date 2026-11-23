# Independent Spanish A audit — tui.model_persistence

Verdict: **passed**, 4/4 keys assessed; no BLOCKER, MAJOR, MINOR or NIT findings. This is independent audit A only, not B, admission, integration, native UI or release approval.

Auditor: i18n-auditor, session c8fbb0cb0ff6, actual model openai/gpt-6.1-sol. Primary anthropic/claude-opus-5-5 failed quota; the runtime's explicit model-switch message is the provenance source. All substantive reading, native assessment and validation occurred on the fallback. Translator e069889d36f8 is a different session; its canonical author record esa-e069/record.json explicitly records openai/gpt-6.1-sol with quota fallback, not the primary model. No MT/API or translation service used; no copy edited.

## Native assessment of every key

| Key | Current Spanish | Assessment |
|---|---|---|
| persist | `/model default guarda este modelo para nuevas sesiones` | Natural concise declarative instruction: the command is the subject of `guarda`. Explicit `nuevas sesiones` correctly limits persistence to future new sessions, not a switch now. `este modelo` clarifies the source's `this` without adding a new action. Exact command preserved. |
| settings | `/settings también cambia el modelo de inicio predeterminado` | Natural neutral Spanish. `también` preserves `too`; `cambia` correctly describes editing the starting default, not changing the current session. Works as the sole settings route in the hosting dead end: no model command is inserted. `modelo de inicio predeterminado` is clear, descriptive product copy, accepted on linguistic merit rather than treating the author's proposal as binding glossary policy. |
| saved | `cambia al modelo de inicio predeterminado con /model saved` | Natural tú imperative, appropriate for instructional prose (not a button/action label, where the guide requires infinitive). `cambia al` denotes switching to the saved default now; does not imply saving the current model. The shared default term remains consistent with settings. |
| with_model | `modelo: {label} — {routes}` | Correct localized human noun, lowercase fragment and Spanish colon/dash spacing. Both simple string arguments occur exactly once, in source order. Already-localized clauses and opaque identity pass through unchanged. |

No destructive/permission reversal, omitted negation, untranslated human prose, register drift or glossary violation found. Model and session match `modelo` / `sesión`; command arguments are deliberately opaque English. Accents, NFC, punctuation and spacing are appropriate. RTL-specific checks are inapplicable to Spanish; native Urdu/Hindi and global UI gates remain open.

## Executed validation

`script.py.txt` is the exact reproduction script; `audit.log` is actual stdout/stderr from its successful exit-0 execution, not narrative notes. Primary repository `.venv/bin/python`, worktree `PYTHONPATH`; module and checker paths verified in the log. `env -i` with scratch HOME/config removed inherited CMUX/LOP variables. No install, test suite, full checker, native process, browser or build ran.

Official `scripts.i18n.check.check_catalogues()` was invoked with only namespace/locales discovery scoped to this namespace and en/es using mocks; all catalogue, context and ICU checks themselves stayed official and unchanged. Result `[]`. Negative missing-key fixture was rejected with `missing key 'tui.model_persistence.saved'`; negative missing-argument fixture was rejected with `en has 2 argument(s), translation has 1`. Runtime missing label/routes raised MessageFormatError for en and es. All four keys matched, JSON loaded, argument names/types/order matched; zero plural/select constructs in either language, so no CLDR arms exist or were fabricated. Exact commands, no markup/backticks, NFC and unchanged source/target hashes were checked.

48 plain-Rich wrapping cases: en/es × four semantic states × short/long opaque IDs × 80/100/120 cells. Every row fit the requested plain width; no last row was a lone command. Spanish ordinary short label wraps to 3/2/2 rows versus English 2/2/2; Spanish long labels require 6/5/4 rows versus English 5/4/4. Spanish settings-only short rendering is 73 cells versus English 49. This is expansion risk, not a fabricated maxCells violation: context declares no cap. Long unbreakable IDs put the human prefix on its own row in both languages, a same-code control rather than a Spanish-only defect. Rendered text preserves identity bytes once and each command count. Synthetic state composition is not proof of live branching. Local pseudo expansion protected all command tokens and ICU arguments and rendered successfully.

Plain Rich diagnostics do **not** certify styled NoticeBlock, pinned working-tail placement, pristine/non-empty transcript-family budgets, all-width command cohesion, bidi/shaping, or rendered native UI. The manager's final centralized/base-aware checker and independent B audit remain required. No shared ledger/history was written. `receipt.json` is append-ready; its dry-run result is in `ledger-dry-run.log` when present. This report is audit_notes_not_execution_log; only logs prove command execution.

## Finding ledger

Empty: unresolved BLOCKER/MAJOR count 0. No suggested copy changes; no forced nit loop or deferred finding.
