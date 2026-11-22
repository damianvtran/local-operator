# Independent Russian B audit — tui.model_persistence

Date: 2026-11-22. Auditor: i18n-auditor, session bcbb66e8cd81, actual model openai/gpt-6.1-sol. Host disclosed quota-triggered fallback from anthropic/claude-opus-5-5 before audit execution. Independent from translator e6cc991f4cd7, A auditor 2986f83bd020, and English source author a97c7f76d5ee. No A conclusions were read. Native-fluent linguistic reasoning only; no MT/API/service.

Scope: all four ordered keys at b1c5ca8be245dcc3f9cae4463d9a5553b3e0c01f. Read Russian style guide and glossary, then English and full context sidecar via approved public-copy reader, then Russian. Budget class: bounded fix/audit lane, eight-minute cap.

## Verdict: passed

Zero unresolved BLOCKER/MAJOR; zero findings at any severity. This is independent linguistic/catalogue approval, not styled consumer qualification, native terminal certification, UI completion, release approval, or eligibility admission.

## Per-key independent assessment (audit_notes_not_execution_log)

1. `tui.model_persistence.persist`: `/model default сохраняет эту модель по умолчанию для новых сессий`. Natural declarative command-subject construction. «сохраняет … по умолчанию» clearly assigns the chosen model as the future default; «для новых сессий» explicitly excludes changing the current session. «эту модель» resolves the English “this” without inventing behavior. «сессий» is correct genitive plural. Exact command preserved once. No correction warranted.
2. `tui.model_persistence.settings`: `/settings тоже меняет модель по умолчанию`. Idiomatic, concise Russian; «тоже» preserves source “too” and reads as a parallel settings capability. In the sole-route state this remains a complete usable instruction, not a dependency on an unavailable /model route: it names only /settings. The established «модель по умолчанию» denotes the boot preference, not a claimed current-session switch. Adding “для новых сессий” might be more explicit in isolation but is not required to correct an ambiguity against this source/context. No omission of required behavior or glossary violation.
3. `tui.model_persistence.saved`: `перейти в этой сессии на сохранённую модель по умолчанию — /model saved`. Infinitive instruction matches Russian UI register. «в этой сессии» explicitly scopes switching to THIS session; «сохранённую» correctly describes the previously saved default and does not say the command saves the current model. «перейти … на … модель» is natural technical product wording. Proper feminine accusative agreement, ё spelling and spaced em dash. Literal command preserved once.
4. `tui.model_persistence.with_model`: `модель: {label} — {routes}`. Human prefix translated with glossary term «модель»; label and routes remain simple ICU arguments in source order, exactly once. Natural sentence-case envelope, source punctuation/spacing retained. Opaque IDs and prelocalized routes were each rendered exactly once without retranslating or parsing.

Across the group: professional impersonal/infinitive register, no inappropriate familiar address, no padding, no added permission/destructive implications, no spurious English human prefix. No plural/select constructions in either source or target; Russian CLDR branch coverage therefore inapplicable, not silently skipped. No fixed maxCells/maxChars in the sidecar; do not inherit the legacy footer's 43-cell budget. Cyrillic text is NFC and cell counts are not inferred from bytes.

## Execution evidence

`audit.log` is actual stdout/stderr from the retained `script.py.txt`, exit 0. The primary checkout interpreter was used with PYTHONPATH pointing at the manager-owned worktree; all three imported module paths were asserted under that worktree. HOME/config isolated under session scratch, environment cleared (including CMUX variables). No Git ran under redirected HOME.

Official `scripts.i18n.check.check_catalogues()` executed with only locale/namespace enumeration narrowed to en/ru and tui.model_persistence; all validation rules unchanged. Result `[]`; injected missing-key control returned the expected missing saved key; renamed-label control returned argument mismatch; restored control returned `[]`. No fixture/catalogue writes. Independent assertions covered JSON/order/key count, args/types/exact-once, NFC, markup/backticks and literal commands. All 4 keys passed; plural/select count zero.

Plain Rich evidence: 27 cases = en/ru/pseudo × (short+long ID envelopes × 80/100/120 columns, plus settings-only × 80/100/120). Raw line frames are retained in audit.log. Actual runtime render_message preserves opaque IDs and composed routes exactly once. Pseudo expansion is explicitly synthetic doubled human letters, not an official pseudo CLI; command tokens and ICU placeholders protected.

The English same-code control split `/model saved` at 100 columns with the short ID; pseudo split `/model default` at 80 with the long ID. Russian commands did not split in these cases; Russian envelopes occupied 2–4 lines. A leading separator appears in some wrapped rows. These observations do not establish a Russian translation defect or a guarantee: real styled NoticeBlock integration must still check pristine/non-empty transcript budgets, padding, short/long IDs, command readability and lone-final-command risk. Plain Rich snapshots are NOT styled/native evidence. No new cap invented; no new native proof claimed.

All three frozen input SHA-256 values matched before and after execution. No source/target, history, baseline, allowance, eligibility, generated output, HEAD or other audit bundle changed. This bundle is new, uncommitted evidence for the manager to retain/commit; not claimed on main. Ledger apply deliberately left to root's serial operation; receipt.json is append-ready, no ledger mutation or dry-run lock performed here.
