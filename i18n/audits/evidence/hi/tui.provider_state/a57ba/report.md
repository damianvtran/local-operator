# Hindi A — tui.provider_state — round 1

Verdict: **passed**, 5/5 keys reviewed independently; 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT. No copy rewrite requested. This is A only, not the required independent B pass.

## Native-fluent LLM judgment (audit notes, not execution log)

Read Hindi style guide and shared glossary, English/context, then Hindi. No MT service/API, other auditor conclusion, or self-translation used. Translator identity is sourced from hi22a/record.json, session 520071768802; auditor session 57ba373043b9 is distinct. Same reported model family does not make these the same actor.

| Key | Current Hindi | Assessment |
|---|---|---|
| logged_in | लॉग इन है | Natural compact completed/current login status; है avoids an imperative. Does not invent a health claim. Stored-credential distinction is supplied by adjacent alternative states, as in English. |
| env_key | परिवेश चर से कुंजी | Reads as a key **from an environment variable**, not a literal variable identifier or environment name. परिवेश चर is standard technical Hindi; से makes provenance explicit. कुंजी preserves source “key”; the glossary's क्रेडेंशियल need not replace this different noun. Slightly technical but appropriate for provider settings. |
| needs_login | लॉग इन ज़रूरी | Natural terse requirement status, not an instruction to click a button or assertion that login succeeded. ज़ uses the nukta correctly. |
| local_ready | सर्वर कॉन्फ़िगर है | Natural developer-product Hindi indicating configured state, not successful authentication or tested reachability. Locality is inherited from the local-provider context, matching the English rather than adding words. कॉन्फ़िगर retains फ़; सर्वर follows the guide. |
| local_unconfigured | सर्वर कॉन्फ़िगर करें | Polite imperative करें correctly contrasts with completed-state है above. Does not claim the configuration already exists. |

All are compact labels without sentence-final danda: no full prose sentence needs punctuation here. No product IDs, flags, paths, digits or script-preservation exception is violated. No inappropriate तुम/तू or added कृपया. No destructive/permission error text exists in this namespace. RTL audit is not applicable to Hindi.

## Actual execution

`audit.log` records the actual primary-venv command, worktree module paths, isolated HOME/config, official scoped `check_catalogues` result, discriminating missing-key/added-argument negatives, and 10 English/Hindi render controls; exit 0. Official catalogue discovery was scoped in-memory to en/hi and this namespace; no global checker or full suite was run. Both negative mutations were in-memory only. All five ordered keys have zero arguments, zero plural/select branches; Hindi CLDR categories one/other are logged, no plural obligations arise. JSON loading, English context schema/key parity, NFC, zero protected tokens and bidi/ZWJ/ZWNJ checks passed. Plain ICU rendering and Rich text rendering preserve all bytes. Pseudo expansion is explicitly synthetic, not an official pseudo-locale or native display test.

Hindi Rich cell_len diagnostics: 7 / 12 / 9 / 12 / 13. English: 9 / 7 / 11 / 17 / 16. Context declares no fixed maxCells/maxChars. These values are not native glyph clearance, and the 17-cell English example is not a new limit. Fit/ellipsis needs real styled geometry, so no invented overflow finding is raised.

## Limits and provenance

Host switched configured anthropic/claude-opus-5-5 to openai/gpt-6.1-sol due to quota before judgment/execution. Per-response served identity is not independently available; this record relies on that host notice, not the configured Claude label. Initial discovery commands had nonzero exits from guessed nonexistent module paths, and one public-copy read used the wrong path family; these are discovery errors, not validator runs. The first and only audit harness execution passed; no failed harness output was discarded.

Native terminal/font shaping, whole UI/screenshots, future resolver/producer wiring (producer still English per task), actual settings geometry, second B audit, all-seven-locale pair rooting/admission, release eligibility and release are **unverified/out of scope**. No source/target/context/history/generated/allowance/cap edits, staging, commit or push. Bundle is created in the canonical tracked-tree root but is untracked until manager commits it; it is not on main.
