# Independent Spanish B audit — tui.model_persistence

Verdict: **passed**. All four keys independently assessed; no BLOCKER, MAJOR, MINOR or NIT findings. Audit A verdict/report was not read. Author record and author mechanical script were read for provenance/API discovery, not accepted as audit evidence; the retained script and execution log are this auditor’s independent run.

## Per-key linguistic assessment (audit notes, not execution log)

- `persist`: `/model default guarda este modelo para nuevas sesiones`. “Guarda” correctly describes saving the current model for future new sessions, not switching the current session. “Este modelo” removes the English “this” ambiguity naturally. The command is the subject, so indicative “guarda” is correct, not an action-label infinitive.
- `settings`: `/settings también cambia el modelo de inicio predeterminado`. Correct complete clause; “también” naturally links the settings alternative in normal composition. Alone it still presents a usable route, without inventing a `/model` route in absent/unknown hosting. “Cambia” accurately means changing the persisted starting model here, not a promise of switching this session. “Modelo de inicio predeterminado” is understandable neutral Spanish for the configured model used at session start. Independently accepted from the source context; the author’s glossary proposal was not treated as authoritative.
- `saved`: `cambia al modelo de inicio predeterminado con /model saved`. Natural tú imperative instructs a switch to the saved startup model. “Al modelo” makes this a switch, not saving the current choice; matches the contrasting `persist` clause. Not a button/action label, so the style guide’s label-infinitive rule does not apply.
- `with_model`: `modelo: {label} — {routes}`. Human prefix translated; opaque ID and already-localized routes remain exact single string arguments in source order. Lowercase sentence fragment and spaced em dash are appropriate.

Terminology is consistent: modelo, sesión/sesiones and predeterminado; protected commands stay byte-identical. No destructive/permission/polarity mismatch. Neutral tú register; accents, NFC, spacing and punctuation appropriate. There is no plural/select in any key; Spanish one/many/other categories are therefore not instantiated or missing.

## Executed mechanical/render scope

See `audit.log` and retained `script.py.txt`. Scoped official catalogue loader, context-sidecar checker, argument checker, plural checker and ICU renderer on en/es only. Exact ordered four-key parity; zero plural/select; placeholder names/types/order/occurrence; protected commands; NFC; absence of markup/backtick spans and bidi controls. Missing-key negative is the exact key-set assertion (not a claim of an official whole-catalogue validator); missing-argument negatives exercise the official parity checker and runtime exception.

96 plain-Rich wrapping cases: en/es × ordinary/missing-model/absent-hosting/unknown-hosting × short/long opaque labels × 80/100/120 columns × original/human-only pseudo expansion. English same-code controls included. All line-cell widths stay within the selected wrapping widths, no lone final command in any case, IDs/routes preserved once, settings-only states contain no model command. Long opaque IDs can split under Rich folding; IDs remain unchanged before wrapping, and no native all-width guarantee is inferred. No source-declared fixed maxCells/maxChars exists; no cap was invented.

Input en/context/es SHA-256 values matched frozen task values before and after execution. Real execution exit 0; no initial failed execution log exists. The earlier approved-reader attempt used nonexistent `.meta.json` and was refused as uncommitted; the actual committed `.context.json` was then read successfully. No source/target changes.

## Boundaries

Plain Rich text wrapping is **not** styled NoticeBlock/native geometry, pristine/non-empty transcript-family validation, UI/consumer integration, shipped eligibility, global checker or release clearance. Those remain the manager’s integration lane. No translation/MT/API/service used. No target, ledger/history, eligibility, baseline or generated file edited; only this new B bundle written. Bundle remains uncommitted for manager collection.

Actual model: openai/gpt-6.1-sol, temporary fallback from anthropic/claude-opus-5-5 after runtime provider quota failure. Actual session: 2e43efb61fa1. Independent from translator e069889d36f8, auditor A c8fb and source author a97; no fallback concealment.
