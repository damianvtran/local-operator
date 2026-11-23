# Independent Urdu audit B — tui.model_persistence

Date: 2026-11-22. Worktree head: b1c5ca8be245dcc3f9cae4463d9a5553b3e0c01f.
Auditor: i18n-auditor B, session 41b08626ce5f, actual model openai/gpt-6.1-sol (quota fallback from anthropic/claude-opus-5-5; explicit host switch). Independent of translator 3c1e111e2598, auditor A caf2c0ebbb90 and source author a97. A conclusions were not read. Translator identity was read solely from t-3c1e/record.json: openai/gpt-6.1-sol, explicitly recorded quota fallback.

## Verdict
PASSED for these four catalogue keys only. No BLOCKER, MAJOR, MINOR or NIT findings. No target/source edits. This is independent LLM linguistic reasoning, not a human native-speaker certification. No MT/API/service was used.

## Every-key assessment (source order)
1. `tui.model_persistence.persist`: `/model default اس ماڈل کو نئے سیشنز کے لیے محفوظ کرتا ہے`. Correctly says the command saves this model for new sessions, not that it immediately switches the current session. `محفوظ کرتا ہے` is a natural declarative command explanation; `سیشنز` is a familiar software plural, not Hindi or Arabic substitution. Command bytes preserved.
2. `tui.model_persistence.settings`: `/settings سے نئے سیشن کا ابتدائی ماڈل بھی بدلا جا سکتا ہے`. Natural polite impersonal explanation: the starting model for a new session can also be changed through settings. `بھی` preserves “too”; the clause remains usable on its own in the settings-only dead end and does not invent a refused /model escape route. `ابتدائی ماڈل` conveys starting model without misleading OS-boot terminology.
3. `tui.model_persistence.saved`: `اس سیشن میں محفوظ ابتدائی ماڈل پر آنے کے لیے /model saved استعمال کریں`. Explicit `اس سیشن میں` distinguishes this-session switching from future-session saving. `پر آنے` is natural for moving to a model in this context; `استعمال کریں` follows respectful register. Saved starting-model meaning is retained; no destructive/permission ambiguity.
4. `tui.model_persistence.with_model`: `ماڈل: {label} — {routes}`. Human model prefix translated with glossary `ماڈل`; both simple parameters occur once and in source order. Opaque provider/model identity and already-localized complete route clauses remain untouched, with source colon/dash structure.

## Mechanical and execution evidence
`audit.log` is actual stdout/stderr and exit code from `script.py.txt`, not reconstructed notes. Exit 0. Primary checkout .venv interpreter with explicit worktree PYTHONPATH; imported catalogue/runtime module paths proved worktree-local. Isolated HOME/config and CMUX_*/LOP_* stripped; no git ran under redirected HOME. Official `check_catalogues()` restricted in memory to en/ur and this namespace: [] problems. Missing-key and renamed-argument controls each produced the expected official failure. No global/base checker, generated outputs, installs, app startup or tests were run.

Verified four ordered keys, valid catalogue/context shapes, exact command/placeholder occurrences, simple argument types, zero plural/select nodes (so CLDR one/other arms are not applicable), NFC, no stored manual bidi controls, unchanged markup/code/flag tokens (none present). Synthetic human-text expansion retained commands/arguments; this is a local pseudo diagnostic, not a shipped pseudo-locale claim.

Plain Rich rendering covered en/ur, ordinary/missing-model/settings-only compositions, 12- and 81-character opaque labels, and widths 80/100/120: 36 frames. Outputs and cell lengths are logged. No maxCells/maxChars hint exists for these wrapping notices; no footer cap was invented. At width 100 the long-label command `/model default` splits across lines in BOTH English and Urdu; at width 80 the 81-character identity folds in BOTH. These matched-control diagnostics are not a new Urdu wording defect or an all-width command guarantee. Settings-only clauses remained complete; no final row consisting solely of /settings was observed.

## RTL and retained integration gates
Stored logical order is sound, Latin commands/IDs remain byte-exact, no manual controls are inserted, and Urdu punctuation/word order are appropriate. Plain Rich does not provide native bidi/shaping/font proof, styled NoticeBlock geometry, real transcript width-family proof, or live-app presentation-boundary proof. Outer/embedded isolation behavior and native rendering remain integration responsibilities. No global UI, release, eligibility, or native-Urdu-rendering approval is granted.

## Provenance and immutability
Before/after SHA-256 of en/context/ur match the task pins. Prior bundles and central audit history were not modified. `receipt.json` is append-ready under the current ledger contract but is NOT applied; manager owns the central append. This report is `audit_notes_not_execution_log`. Initial discovery-only path misses (guide suffix, guessed messages paths, non-existent pseudo module) occurred before validation; they are not hidden validator failures. The sole executed validation run passed; no failed validation log was overwritten.
