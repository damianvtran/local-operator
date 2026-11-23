# Spanish author receipt — translated, not audited

Namespace: `tui.model_persistence`; four keys, source order unchanged.
Author: i18n-translator, session `e069889d36f8`.
Actual author/validation model: `openai/gpt-6.1-sol`.
Primary model: `anthropic/claude-opus-5-5`; fallback was explicitly announced by the runtime's model-switch message, reason `provider failure: quota`. The primary handled only the initial guide read before fallback; Spanish authorship and validation occurred after the switch. No translation API, SDK, MT service, website, borrowed translation or additional agent was used.

## Native choices and glossary conformance

- `model` → `modelo`, `session` → `sesión` per shared glossary. Product commands and opaque identities remain unchanged.
- Lowercase instructional fragments match the English tone. `cambia` addresses tú per the Spanish style guide; action-label infinitives do not apply to this instruction sentence.
- `guarda este modelo para nuevas sesiones` explicitly distinguishes saving the current selection for future sessions from switching this session.
- `modelo de inicio predeterminado` describes the saved boot-time model, consistently in the settings and saved-switch clauses. `cambia al … con /model saved` is an instruction to switch, never to save.
- `/settings también cambia …` retains the source's additive “too” and is still a complete usable clause when it is the sole dead-end route. No `/model` route is introduced in absent/unknown-hosting state.
- `modelo: {label} — {routes}` translates the human prefix only. Both arguments occur once and are rendered without retranslation; punctuation is preserved.
- Glossary proposal: boot default → `modelo de inicio predeterminado` when it means the model for new sessions. No repository glossary edits were made.
- No unresolved source ambiguity requiring a product decision. `/model default` and `/model saved` have opposite save/switch roles as specified in the sidecar and task; these were not inferred from their command names.

## Actual validation, not an independent audit

`validator.py.txt` is the exact scoped validation program; `audit.log` is actual combined subprocess output, including the command, environment isolation, module path, frozen input hashes and exit status. These notes are `audit_notes_not_execution_log` and do not substitute for the log.

Official APIs exercised: catalogue loader, context loader/schema checker, ICU parser/argument matcher/plural-category checker and ICU renderer. All four keys match in order, placeholders are exact, plural/select arms are zero, literal command bytes are exact, and no manual bidi controls are present. A missing-argument negative control is rejected. The primary checkout's `.venv` interpreter loads worktree source via PYTHONPATH; HOME/config are isolated under session scratch and CMUX/LOP variables stripped. No Git command ran under redirected HOME.

48 plain-Rich wrapping cases: en/es × four semantic route sets × short/long opaque IDs × 80/100/120 columns. Every case preserves opaque label/route bytes and has no lone final-row command. The exact geometry is in the log. Spanish ordinary/short takes 3 rows at 80 versus English's 2; Spanish ordinary/long takes 6 versus English's 5 at 80 and 5 versus 4 at 100. This is recorded expansion, not overflow or an invented fixed cap. Long opaque IDs fold under the same code in both languages. The actual native NoticeBlock and pristine/non-empty transcript-family budgets were NOT exercised or certified; root integration owns those checks, pseudo/RTL and all-seven-locale validation. No native/release clearance is claimed.

Only the new Spanish catalogue and this own canonical author bundle were written. Nothing was staged, committed or pushed; no shared ledger/history/generated keys/source/other locale was changed. Root must freeze/commit the target before independent auditors access it through the approved reader. Status is `translated`, never `passed`.
