# Independent translation audit — hi / mobile.setup — a4

- PR: #3958 (branch fix/tunnel-remedy-no-commands), head 139bca9037fda9e936e77dd9f145c99ce1188315; scope `git diff 983511c54..139bca903 -- local_operator/i18n/` (3 new keys, 1 removed key, 0 existing keys changed).
- Translator: coder subagent of host Local Operator session 25fe8dfbf610 (values committed in 139bca903 on fix/tunnel-remedy-no-commands; English source by the same coder); model not disclosed to the auditor (recorded null).
- Auditor: translation-auditor (lopdev team), independent Local Operator session b3a764eb5573; not the translator/coder (25fe8dfbf610); edited no catalogue value; model anthropic/claude-opus-5-5 (session system prompt; no provider fallback observed during the audit).
- Date: 2026-11-14. Result: **PASSED** — 0 blocker / 0 major; 1 minor/nit finding(s), none gating (deferred to the PR thread).

## Method
Read side by side with the en source and context sidecar; compared terminology with the namespace's disabled, billing_lapsed, login_tui, *_plain keys and #3945's tunnel.status/action_* copy. Verified: no placeholders in source or target; NFC; no bidi controls; typographic apostrophes; fr U+202F before ; : ! ?; zh-CN no CJK/Latin space; the *_resume keys carry no automatic-reconnection wording and no command/flag/URL; operation_failed_tui keeps `/login radient` verbatim and no `lop login`; mobile.setup.mobile_bundle_tui absent from the catalogue; key parity with en; runtime `ms.<key>().envelope(locale=…)` equals the catalogue value. No maxCells on these keys (prose result line / remedy card that wraps).

## Locale notes
Accurate, आप register; 'फ़ोन एक्सेस', 'Radient कंसोल', 'क्रेडिट जोड़ें', 'बंद ही रहेगा' consistent with disabled/action_add_credit. One minor idiom finding (HI4-m1).

## Findings
- HI4-m1 (minor) `mobile.setup.operation_failed_tui` — current: … या आपका साइन-इन फिर से करना होगा। — proposed: … या आपको फिर से साइन इन करना होगा। — 'आपका साइन-इन … करना होगा' makes the noun 'sign-in' the object of करना and reads translated; the namespace's idiom is the verb 'साइन इन करें' with the person as subject.

## Keys read (3)
| key | en | hi |
|---|---|---|
| `mobile.setup.operation_failed_tui` | Phone access could not be checked: Radient could not be reached or your sign-in needs renewing. Check your internet connection or sign in with /login radient, then try again. | फ़ोन एक्सेस की जाँच नहीं हो सकी: Radient तक पहुँच नहीं हो पाई या आपका साइन-इन फिर से करना होगा। अपना इंटरनेट कनेक्शन जाँचें या /login radient से साइन इन करें, फिर दोबारा कोशिश करें। |
| `mobile.setup.disabled_resume` | Phone access for this computer is turned off in the Radient console, so it stays off. Turn it back on in the console, then turn phone access back on here. | इस कंप्यूटर का फ़ोन एक्सेस Radient कंसोल में बंद है, इसलिए यह बंद ही रहेगा। इसे कंसोल में फिर से चालू करें, फिर यहाँ फ़ोन एक्सेस दोबारा चालू करें। |
| `mobile.setup.billing_lapsed_resume` | Your Radient balance has run out, so phone access stays turned off. Setup never purchases credit; add credit, then turn phone access back on here. | आपका Radient बैलेंस ख़त्म हो गया है, इसलिए फ़ोन एक्सेस बंद ही रहेगा। सेटअप कभी क्रेडिट नहीं खरीदता; क्रेडिट जोड़ें, फिर यहाँ फ़ोन एक्सेस दोबारा चालू करें। |
