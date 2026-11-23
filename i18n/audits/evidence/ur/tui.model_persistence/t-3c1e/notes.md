# Urdu author receipt — translated, not audited

Author: i18n-translator, session `3c1e111e2598`, actual model `openai/gpt-6.1-sol`.
The host's explicit model-switch message reports fallback from `anthropic/claude-opus-5-5` because of provider quota failure. Translation and validation were performed after that switch. No translation API, SDK, service or website was used.

## Native wording decisions

- `persist`: `/model default` is the subject of a factual sentence: it saves **this model for new sessions**. `محفوظ کرتا ہے` describes saving, not switching the active session. `نئے سیشنز` is the ordinary software-register plural of the style guide's settled loanword سیشن.
- `settings`: `نئے سیشن کا ابتدائی ماڈل` explains boot default as the initial model of a new session, avoiding an opaque English “boot” loanword. `بھی` retains “too”; the sentence remains a complete usable instruction alone. No other route was added.
- `saved`: `اس سیشن میں` explicitly locates the switch in this session. `محفوظ ابتدائی ماڈل پر آنے` means going to the saved starting model, not saving the present choice. The respectful imperative `استعمال کریں` follows the guide.
- `with_model`: human prefix `ماڈل` follows the glossary; both simple arguments remain exactly once. The already-localized routes argument and opaque identity are interpolated without modification or reparsing.

Glossary proposals (not edits): “boot default” → `نئے سیشن کا ابتدائی ماڈل` when introducing it; `محفوظ ابتدائی ماڈل` where the saved starting choice is already understood. Neither is an English-copy exception.

## Inputs and reading

Read `guide://i18n`, AGENTS.md localization/environment sections, docs/I18N.md, RFC §§2.2/6, Urdu style guide and shared glossary. Source and context were read through `read_public_copy`. An attempted sibling `ur/tui.model_picker.json` read was refused because it is not committed at HEAD; no bypass was used. The committed `ur/cli.model_help.json` supplied consistency context (`ماڈل`, `سیشن`, `بدلیں`). No existing target namespace file was overwritten.

## Mechanical evidence and boundaries

`audit.log` is actual captured execution, not reconstructed notes. `validator.py.txt` is the exact executed scoped reproduction. The primary checkout `.venv/bin/python` loaded the worktree's official catalogue/context/ICU APIs and scoped checker `arguments_match`, as logged. `env -i` removed inherited CMUX and other session settings; HOME/config were isolated beneath this session's scratchpad. Git commands retained normal HOME.

Four ordered keys, all protected commands, exactly-once simple arguments, opaque short/long identities, zero plural/select arms, NFC, absence of manual bidi controls and frozen English/context hashes passed. Human Latin residue is rejected after excluding the exact approved commands and argument identifiers.

Widths 80/100/120 are diagnostics only. Rich's 80-column long-ID result begins with a 6-cell prefix-only line: the tested opaque ID alone is 81 characters, exceeding that width, so its folding is not a translation length cap failure. The log records the actual row sizes, not a native rendering promise. No command-only final row was observed in these diagnostic samples. This plain-text wrapping is not the styled NoticeBlock or both transcript families.

RTL risks remain: Latin command word order, the colon after ماڈل, the em dash, middle-dot seams and long opaque model IDs must be checked at the integration paint boundary. No stored bidi controls were added, per the style guide. No native font/shaping, terminal bidi, browser, native programme, whole-UI, eligibility or release clearance is claimed.

Independent approved-reader A/B audits must wait for the manager's private root commit. This author is not either auditor. Shared history/ledger/generated assets and other locales were not edited; no stage, commit or push was performed.
