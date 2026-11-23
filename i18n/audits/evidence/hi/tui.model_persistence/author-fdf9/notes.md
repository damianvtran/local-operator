# Hindi authorship — translated, not audited

Four strings authored directly by i18n-translator session `fdf96049990f` using actual model `openai/gpt-6.1-sol`. The host reported a quota/provider-failure fallback from `anthropic/claude-opus-5-5` before the Hindi wording was authored. No translation API, SDK, website or service was used. This is author reasoning and mechanical validation, not an independent linguistic audit or self-approval.

## Native wording decisions

- `persist`: “इस मॉडल को नए सत्रों के डिफ़ॉल्ट के रूप में सहेजें” explicitly saves the current choice for future new sessions. It does not suggest switching this session. The extra “के रूप में” is idiomatic and prevents an unclear object/default relationship; this wrapping notice has no 43-cell footer cap.
- `settings`: “से भी नए सत्रों का डिफ़ॉल्ट बदलें” retains the alternative-route “too”, explains the boot default in everyday Hindi, and is a complete usable instruction by itself when hosting is absent/unknown. No unavailable `/model` route is added.
- `saved`: “इस सत्र में सहेजे गए डिफ़ॉल्ट पर जाएँ” makes the THIS-session switch explicit and distinct from saving. “पर जाएँ” naturally describes switching to the saved choice; `/model saved` remains an exact command.
- `with_model`: human prefix “मॉडल” follows the shared glossary; `{label}` and `{routes}` remain simple arguments exactly once. The opaque model ID and already-localized clauses are neither translated nor recursively interpolated.
- `सत्र` follows the glossary and Hindi guide. Polite imperatives `सहेजें`, `बदलें`, `जाएँ` follow the guide. English command bytes, punctuation seams and ASCII spaces inside commands are unchanged. No manual bidi controls were inserted. All four messages have zero plural/select arms; no Hindi branches were invented.
- Glossary proposal: default → `डिफ़ॉल्ट` (the saved startup choice); boot default → `नए सत्रों का डिफ़ॉल्ट` in this notice. These terms were absent from the shared glossary, so this proposal records the choice without editing it. Save → `सहेजें` follows the guide's imperative example, not the distinct “keep temporary chat” glossary term.

## Actual validation and limitations

The final `audit.log` records an executed, isolated scoped validator using the primary checkout's `.venv/bin/python`, `PYTHONPATH` pointing at this worktree, scratch HOME/config, and `env -i` (no inherited CMUX variables). Module paths prove worktree catalogue/runtime code was loaded. The script imports the official checker and calls only its scoped argument/context/plural APIs, plus official catalogue load and ICU render APIs. No global/base checker, test suite, install, generator, browser, server, native app or release command ran.

Final exit code: 0. Four ordered keys, full parity, protected command counts, simple arguments exactly once, zero plural/select arms, absent-hosting settings-only composition, opaque strings, missing-label and missing-routes negative controls all passed. English/context and target hashes were identical before/after execution.

`initial-attempt.log` preserves a failed first validator invocation: an incorrect keyword call to `arguments_match` raised TypeError. The invocation was corrected to the inspected signature; the successful rerun is a distinct execution, not a rewritten failure log.

Rich wrapping diagnostics at 80/100/120 columns use short and long IDs and three route sets. They are NOT styled NoticeBlock geometry, English-control comparisons, native Devanagari shaping/font proof, or pristine/non-empty transcript integration evidence. A long opaque ID can wrap, and the 100-column short-label setup diagnostic leaves a final three-cell word; no all-width presentation guarantee is claimed. There is no fixed maxCells/maxChars in this source context. Root-owned native Hindi validation and independent A/B reading of all four committed target values remain required. Status is `translated` only; no audit, admission, shipped eligibility, full UI or release clearance.
