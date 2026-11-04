# Independent French delta audit — core #3349

**Passed**, terminal linguistic/mechanical delta verdict. All 2 revised keys audited. BLOCKER 0; MAJOR 0; MINOR 0; NIT 0. No findings.

## Provenance
Auditor: i18n-auditor, session 05c51f9b1aba, actual anthropic/claude-sonnet-5-5 from current runtime header. No fallback observed in this resumed delta audit. Earlier audit provenance is neither reused nor overwritten. Translator session a58a48732351 / openai/gpt-6.1-sol is delegation-supplied; separate from auditor. No translation service, self-translation, or translator/auditor opinions consulted.

## Per-key reasoning (audit_notes_not_execution_log)
- `wire.aida_cadence.interval_hours.label`: “Intervalle des points (h)” is idiomatic compact French settings copy, with standard hour abbreviation h and established wire.settings “point” terminology. Unit explicit, register neutral.
- `wire.aida_cadence.interval_hours.help`: “0 : point quotidien à l’heure locale. 1–168 : heures entières écoulées (4 : toutes les 4 h). Un point dû le reste après modification.” accurately distinguishes daily local-time mode 0 from inclusive whole elapsed-hour values 1–168. Parenthetical 4 is an example, not a universal default. “Un point dû le reste après modification” explicitly preserves an owed check-in; it adds no promise of delivery during pause. Nominal clauses are appropriate compact help. U+202F before colons and between 4 and h, U+2019 apostrophe, en dash and Latin digits follow French typography. No omissions, polarity issue or glossary violation.

## Executed mechanical checks / fit
Actual repo validator `~/local-operator/.venv/bin/python scripts/i18n/check.py --catalogues-only` returned 0 problems, exit 0. Direct runtime checks confirmed exact two-key EN/context/FR parity, valid JSON, NFC and no ICU arguments. No plural/select, markup, backtick, command or flag present. Input hashes independently matched all three delegated hashes.

Measured Rich cell widths, independently rechecked against Unicode character lengths: label 25 cells, help 133 cells. Both satisfy delegation fit criteria <=29 and <=137. Neither actual source sidecar defines maxCells/maxChars; external fit targets are not claimed as source metadata. Previous French widths were 36/206; English is 25/137. Qualitative pseudo-long-text assessment: revised static copy is no longer longer than English, with one-cell Latin glyphs, precomposed accents and no combining sequence hazards. This is not rendered fit proof; no synthetic pseudo-render, screenshot, wrapping geometry or pixels produced. RTL inapplicable.

The first bundle-assembly attempt had a speculative width assertion 24/129, which failed on actual measured 25/133 before report/record creation. Corrected by actual remeasurement; original validator/probe log retained with labelled recovery note. No output or failed execution is presented as a pass.

## Limits / retention
No full suite, native build, install, repository edits, ledger write, push or PR operation. Render QA remains separate. This new bundle does not overwrite the earlier receipt. Retain byte-identically under `i18n/audits/evidence/fr/wire.aida_cadence/frca-delta-05c51/`. Currently scratch only, not tracked/main.
