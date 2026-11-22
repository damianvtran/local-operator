# Russian independent A — tui.model_persistence

Date: 2026-11-22. Audited revision: b1c5ca8be245dcc3f9cae4463d9a5553b3e0c01f.
Auditor: i18n-auditor / session 2986f83bd020 / actual openai/gpt-6.1-sol. Requested anthropic/claude-opus-5-5; host switched before substantive auditing because provider quota failed. Independent of translator e6cc991f4cd7 and English author a97c7f76d5ee. Translator's actual openai/gpt-6.1-sol fallback provenance comes from ru-e6cc/record.json, not inferred from auditor configuration.

## Verdict

**passed** — 4/4 keys read and assessed; zero unresolved BLOCKER/MAJOR, zero findings. This is independent linguistic audit A only, not completed B, admission, global checker or UI/native qualification. No translation/MT/API service used; no catalogue edits.

## Native-language assessment, complete ordered key set

1. `tui.model_persistence.persist`: `/model default сохраняет эту модель по умолчанию для новых сессий`. Accurate present-tense instructional description: saves THIS model as the default for NEW sessions. `для новых сессий` explicitly prevents interpreting it as switching the current session. `модель по умолчанию` and the inflection `сессий` are standard Russian software terminology. Command unchanged. No change suggested.
2. `tui.model_persistence.settings`: `/settings тоже меняет модель по умолчанию`. Natural concise Russian; `тоже` accurately corresponds to “too”, and `меняет` describes editing/changing the startup default, not a session switch. In the sole-route hosting-deadend state it remains a complete usable instruction without inventing a `/model` route. The additive wording is specified in the English/context contract, not a Russian-specific omission. No change suggested.
3. `tui.model_persistence.saved`: `перейти в этой сессии на сохранённую модель по умолчанию — /model saved`. Idiomatic infinitive action instruction, expressly scoped to THIS session. `сохранённую` preserves saved-default semantics, not a save operation. Correct case agreement, ё, and spaced em dash. Command unchanged. No change suggested.
4. `tui.model_persistence.with_model`: `модель: {label} — {routes}`. Human prefix translated using glossary `модель`; both simple string arguments exactly once. Colon and spaced em dash are correct Russian punctuation. Already-localized routes and opaque model identity are inserted without retranslating. No change suggested.

Register is impersonal/infinitive and concise throughout, consistent with ru style. No informal address, politeness padding, title case, quote misuse or unsafe polarity. No markup/backtick/code/flag content exists beyond the protected commands and placeholder-provided opaque IDs. No ICU plural/select exists: zero branches verified, so Russian CLDR category coverage is not applicable to these messages.

## Actual execution

See `script.py.txt` and raw `audit.log`, which are execution evidence; this report is `audit_notes_not_execution_log`.

Executed primary checkout interpreter with PYTHONPATH pointing at the frozen worktree, isolated HOME/config, env -i (CMUX count 0), TERM=xterm-256color. All four imported module paths asserted inside the worktree. Official `scripts.i18n.check.check_catalogues()` was called with only namespace/locale enumeration narrowed to tui.model_persistence and en/ru; rules unchanged. Actual result `[]`.

Checks passed: strict repository catalogue loading, exact ordered 4-key/context parity, official context metadata, ICU parse/argument parity, label/routes simple type and once-only use, zero plural/select, NFC, protected command byte identity, no manual bidi controls, source/context/target hashes equal before/after. Official missing-key and missing-argument counterfactuals both returned diagnostic errors. Runtime missing label/routes raised MessageFormatError in both English and Russian; unknown-key production resolver returned the missing code with source_locale=None in both. Human-only pseudo expansion preserved command/placeholder tokens and rendered opaque arguments unchanged.

Plain Rich renders covered English first as same-code control, then Russian: ordinary, missing-model setup, hosting-deadend; short 3-character and long 49-character opaque labels; widths 80/100/120 (36 wrapping cases total). Route-only/no-label composition was also inspected. RU row counts: ordinary short 3/2/2, long 4/3/3; missing-model short 2/2/2, long 3/2/2; hosting-deadend short 1/1/1, long 2/2/1. All measured lines fit selected widths; no RU command was split and no final row consisted solely of a command. Long-label Russian ordinary notices can leave `умолчанию` on a 9-cell final row at widths 80 and 120: wrapping risk recorded, not fabricated fixed-cap overflow. English control itself split `/model saved` at width 100 with the short label, establishing that normal-space command folding is a shared renderer baseline, not proof of a Russian translation defect.

No maxCells/maxChars exists in this context; the footer's 43-cell cap is explicitly inapplicable. These plain Rich diagnostics are not styled NoticeBlock, pristine/non-empty transcript-family geometry, rendered screenshots, native TUI or Urdu/Devanagari qualification. Manager-owned global/base-aware check remains outstanding after all A/B/admission rows. B remains outstanding.

## Current finding ledger

Empty. No BLOCKER, MAJOR, MINOR or NIT. No deferred copy fixes.

## Retention

This is a NEW A bundle; no prior author/failed/admission/A/B artifact was changed. Append-ready receipt is provided; the official ledger append is dry-run only here, with actual output retained separately. Manager owns append/commit; this bundle is not claimed merged or already on main.
