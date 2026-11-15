# Independent translation audit — ur / mobile.setup — a4

- PR: #3958 (branch fix/tunnel-remedy-no-commands), head 139bca9037fda9e936e77dd9f145c99ce1188315; scope `git diff 983511c54..139bca903 -- local_operator/i18n/` (3 new keys, 1 removed key, 0 existing keys changed).
- Translator: coder subagent of host Local Operator session 25fe8dfbf610 (values committed in 139bca903 on fix/tunnel-remedy-no-commands; English source by the same coder); model not disclosed to the auditor (recorded null).
- Auditor: translation-auditor (lopdev team), independent Local Operator session b3a764eb5573; not the translator/coder (25fe8dfbf610); edited no catalogue value; model anthropic/claude-opus-5-5 (session system prompt; no provider fallback observed during the audit).
- Date: 2026-11-14. Result: **PASSED** — 0 blocker / 0 major; 0 minor/nit finding(s), none gating (deferred to the PR thread).

## Method
Read side by side with the en source and context sidecar; compared terminology with the namespace's disabled, billing_lapsed, login_tui, *_plain keys and #3945's tunnel.status/action_* copy. Verified: no placeholders in source or target; NFC; no bidi controls; typographic apostrophes; fr U+202F before ; : ! ?; zh-CN no CJK/Latin space; the *_resume keys carry no automatic-reconnection wording and no command/flag/URL; operation_failed_tui keeps `/login radient` verbatim and no `lop login`; mobile.setup.mobile_bundle_tui absent from the catalogue; key parity with en; runtime `ms.<key>().envelope(locale=…)` equals the catalogue value. No maxCells on these keys (prose result line / remedy card that wraps).

## Locale notes
Clean. آپ register; RTL with embedded Latin runs 'Radient' and '/login radient' and no manual bidi controls (same shape as login_tui); Urdu punctuation ۔ ، ؛ as in disabled/billing_lapsed; feminine agreement 'فون رسائی … رہے گی' correct.

## Findings
- None.

## Keys read (3)
| key | en | ur |
|---|---|---|
| `mobile.setup.operation_failed_tui` | Phone access could not be checked: Radient could not be reached or your sign-in needs renewing. Check your internet connection or sign in with /login radient, then try again. | فون رسائی کی جانچ نہیں ہو سکی: Radient سے رابطہ نہیں ہو سکا یا آپ کو دوبارہ سائن اِن کرنا ہوگا۔ اپنا انٹرنیٹ کنکشن چیک کریں یا /login radient سے سائن اِن کریں، پھر دوبارہ کوشش کریں۔ |
| `mobile.setup.disabled_resume` | Phone access for this computer is turned off in the Radient console, so it stays off. Turn it back on in the console, then turn phone access back on here. | اس کمپیوٹر کی فون رسائی Radient کنسول میں بند ہے، اس لیے یہ بند ہی رہے گی۔ اسے کنسول میں دوبارہ آن کریں، پھر یہاں فون رسائی دوبارہ آن کریں۔ |
| `mobile.setup.billing_lapsed_resume` | Your Radient balance has run out, so phone access stays turned off. Setup never purchases credit; add credit, then turn phone access back on here. | آپ کا Radient بیلنس ختم ہو گیا ہے، اس لیے فون رسائی بند ہی رہے گی۔ سیٹ اپ کبھی کریڈٹ نہیں خریدتا؛ کریڈٹ شامل کریں، پھر یہاں فون رسائی دوبارہ آن کریں۔ |
