# Hindi provider-state author notes — NOT an independent audit

Status: translated only. All five keys authored by native-level LLM reasoning in this session; no MT/API/service used. Hindi style guide and glossary were read before authoring. English/context read through read_public_copy at the assigned committed head. Existing hi tui.setup_cues voice was read through the same approved reader.

- logged_in → लॉग इन है: terse completed/current authentication status, not a login button or imperative; the stored-credential meaning comes from the source context, not a live-session assertion.
- env_key → परिवेश चर से कुंजी: explicitly identifies the environment variable as the key source. It is a human label, not a literal variable ID or environment name. Prefer semantic clarity over the shorter but ambiguous “परिवेश की कुंजी”.
- needs_login → लॉग इन ज़रूरी: required-authentication status, deliberately not the imperative “लॉग इन करें”.
- local_ready → सर्वर कॉन्फ़िगर है: configuration status only, no reachability, server-health, or successful-login promise. Locality is supplied by the local provider row, just as in English.
- local_unconfigured → सर्वर कॉन्फ़िगर करें: polite imperative, contrasting the completed status above. सर्वर follows the style guide; कॉन्फ़िगर is a familiar technical Hindi loan rather than an unnatural over-Sanskritized replacement. No human label stays in Latin English.

## Glossary proposals (not glossary edits)
- log in / logged in: लॉग इन करें / लॉग इन है; needs login: लॉग इन ज़रूरी.
- environment variable: परिवेश चर; key (credential): कुंजी; environment-sourced key: परिवेश चर से कुंजी.
- configured / configure server: सर्वर कॉन्फ़िगर है / सर्वर कॉन्फ़िगर करें.

## Mechanical evidence and limits
validation.log is actual execution output; these notes are not an execution log. validator.py.txt reuses the existing es22a scoped recipe (code only, no result reuse) and adds protected-token absence and Rich/plain equivalence assertions. It executes the official scripts.i18n.check.check_catalogues with only namespace/locale enumeration narrowed to en/hi and tui.provider_state; no eligibility patch. Actual missing-key and added-argument negative controls are rejected. Exactly five ordered keys, zero ICU arguments/plural/select structures, NFC, no bidi controls or protected tokens, official render_record source locale hi, and English control outputs are verified.

Rich cell_len values 7/12/9/12/13 are combining-character diagnostics, NOT native font/shaping/style proof. Pane 31 with provider-ID budget 17 is illustrative, not a universal maxCells constraint. Pseudo is a synthetic wrapper/expansion diagnostic, not a shipped pseudo locale. No app startup, live provider, browser, native screenshot, independent self-audit, eligibility, fixed-UI or release claim. Current UI producer remains English. Independent A/B audits and parent integration remain pending. No ledger/history/generated/source files edited; nothing staged/committed/pushed.
