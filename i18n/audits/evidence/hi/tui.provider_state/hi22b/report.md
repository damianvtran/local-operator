# Independent Hindi B audit — tui.provider_state

Verdict: **passed**, all 5 keys assessed; no BLOCKER, MAJOR, MINOR or NIT findings.
This is independent lexical B only, not audit A, native-font evidence, live-UI proof, programme completion or release approval. No translation/MT/API service was used. No translator text was edited; no A conclusions were read.

## Native-fluent LLM judgment (audit notes, not execution log)

1. `tui.provider_state.logged_in` — **लॉग इन है**. Natural terse developer-UI completed authentication state. The copula है makes this a current established state, not an instruction. In this provider context it appropriately denotes an existing stored login, without inventing health verification.
2. `tui.provider_state.env_key` — **परिवेश चर से कुंजी**. Correctly identifies the human credential source: a key from an environment variable. परिवेश चर is standard technical Hindi; से expresses origin. This is not an actual variable name and need not remain Latin. कुंजी faithfully renders key; replacing it with the glossary's general credentials noun is not required here.
3. `tui.provider_state.needs_login` — **लॉग इन ज़रूरी**. Idiomatic compact requirement status, not a completed login or a login button. The omitted copula is normal label ellipsis, and ज़रूरी preserves the nukta spelling and requirement polarity.
4. `tui.provider_state.local_ready` — **सर्वर कॉन्फ़िगर है**. Natural developer-register configured state. It claims configuration, not that a server is running, healthy, reachable, or logged in. सर्वर follows the locale glossary/style; कॉन्फ़िगर is an appropriate everyday software loan, with फ़ preserved. Locality belongs to the provider context, just as in English; no need to add a redundant local adjective.
5. `tui.provider_state.local_unconfigured` — **सर्वर कॉन्फ़िगर करें**. Clear polite imperative, contrasting correctly with है in the completed-state label. No invented login error or completed configuration claim. करें follows the guide's आप-register imperative.

All five have appropriate short-label punctuation (no prose full stop/danda needed), no gratuitous honorifics, and consistent technical register. No product names, flags, literal identifiers, backtick code or markup occur in these values; none were lost. Latin/Devanagari digit policy has no applicable digits. No RTL audit is applicable to Hindi.

## Mechanical execution

`audit.log` is actual stdout from `script.py.txt`, run with the primary checkout venv, this worktree on PYTHONPATH, isolated HOME/config and env -i (no CMUX). Exit code 0. The official check_catalogues API was narrowed only by catalogue locale/namespace enumeration to en/hi and tui.provider_state. It ran catalogue, context, argument and plural checks: zero problems. In-memory missing-key and added-argument controls were both rejected. Every key has zero arguments, zero plural/select nodes, matching order and valid NFC JSON; all five ICU renders and plain/Rich controls passed. English controls passed. Synthetic pseudo wrapping/expansion passed low-level ICU/plain Rich checks; this is not a shipped pseudo locale or whole-app pseudo test.

Rich cell_len diagnostics are 7/12/9/12/13, not native glyph geometry. The context declares no maxCells/maxChars cap; no shortening mandate is inferred. Conjunct/font/terminal truncation remain unverified, rather than being asserted acceptable from these numbers. Current producer remains English: this audit does not claim integration is fixed.

## Provenance and scope

Auditor i18n-auditor, session 7d0c5bb3f103. Host model-switch notice reports openai/gpt-6.1-sol after anthropic/claude-opus-5-5 quota failure. No fresh served-response telemetry was collected; model is host-reported, not independently telemetry-certified. Translator identity is sourced from committed hi22a/provenance.json: session 520071768802, openai/gpt-6.1-sol. Different session/agent instance; same model does not imply same translator. This B audit did not consult A (c3ac7ba4ef12).

Read-only discovery had three failed path probes (nonexistent icu.py, render.py, scripts/i18n/pseudo.py); these were interface discovery, not failed validation. Correct runtime.py/check.py interfaces were then read and exercised. They do not constitute native or execution proof. No validation failure was suppressed or replaced.

History is not applied. directledger-receipt.json is the direct append receipt; ledger-dry-run.log records the optional CLI dry-run only. Only this new evidence directory was written; no staging/commit/push, translations, contexts, eligibility or generated files were changed.
