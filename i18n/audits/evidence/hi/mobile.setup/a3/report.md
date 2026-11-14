# Independent translation audit — hi / mobile.setup — a3

- PR: #3958 (branch fix/tunnel-remedy-no-commands, stacked on #3945), head 01113e5aa048884a507d9bb4868815f909f63a48; scope `git diff 374037856..01113e5aa -- local_operator/i18n/` (12 new `*_plain` keys; 0 existing keys changed).
- Translator: coder subagent of host Local Operator session 25fe8dfbf610 (values committed in 01113e5aa on fix/tunnel-remedy-no-commands; English source by the same coder); model not disclosed to the auditor (recorded null).
- Auditor: translation-auditor (lopdev team), independent Local Operator session 9054cd229d3f; not the translator/coder of host session 25fe8dfbf610; edited no catalogue value; model anthropic/claude-opus-5-5 (session system prompt; no provider fallback during the audit).
- Date: 2026-11-14. Result: **PASSED** — 0 blocker / 0 major; 1 minor/nit finding(s), none gating.

## Method
Read side by side with en source + context sidecar (12 new keys, 0 changed); glossary agent/approvals/service and the namespace's phone-access, phone-address, sign-in, account and setup terms match #3945's shipped copy; register per style guide; no placeholders in source or target; no command, flag, slash or code-span text (setup._SHELL_COMMAND + slash/flag/backtick scan); NFC; no bidi controls; no maxCells on these keys — the TUI's 50-cell park card renders tunnel.status card_* keys, not these; the *_plain keys reach the TUI only through setup.text(surface='tui') in a wrapping system notice and the desktop's remedy.text. Runtime: setup.plain_remedy({'step': s}) equals the catalogue value for all 12 steps.

## Locale notes
आप register; danda; 'स्वामी' matches owner_missing/owner_mismatch; 'स्वीकृति' per glossary.

## Findings
- HI3-01 minor (account_plain): 'Radient खाते और फ़ोन पते की…' — 'Radient खाते' reads as the compound 'Radient account' (cf. mobile.setup.account 'Radient खाता:'), so Radient as the subject that could not confirm is easily lost. Proposed: 'Radient आपके खाते और फ़ोन पते की पुष्टि नहीं कर सका। अपना इंटरनेट कनेक्शन जाँचें और दोबारा कोशिश करें; कुछ भी नहीं बदला गया।' Deferred to the PR thread; meaning (nothing confirmed, nothing changed) survives.

## Keys read (12)
| key | en | hi |
|---|---|---|
| `mobile.setup.stopped_plain` | Phone access is turned off on this computer. Turn it back on whenever you want to reach this computer from your phone. | इस कंप्यूटर पर फ़ोन एक्सेस बंद है। जब भी आप अपने फ़ोन से इस कंप्यूटर तक पहुँचना चाहें, इसे फिर से चालू करें। |
| `mobile.setup.owner_missing_plain` | The Radient account that set up phone access is no longer signed in on this computer. Sign in with that account to bring phone access back. | जिस Radient खाते से फ़ोन एक्सेस सेट किया गया था, वह अब इस कंप्यूटर पर साइन इन नहीं है। फ़ोन एक्सेस वापस पाने के लिए उसी खाते से साइन इन करें। |
| `mobile.setup.login_plain` | Sign in to Radient with the account that owns phone access, then try again. | फ़ोन एक्सेस के स्वामी खाते से Radient में साइन इन करें, फिर दोबारा कोशिश करें। |
| `mobile.setup.choose_account_plain` | Choose the Radient account phone access should use. Setup never moves an existing phone address to a different account. | चुनें कि फ़ोन एक्सेस किस Radient खाते का उपयोग करे। सेटअप किसी मौजूदा फ़ोन पते को कभी दूसरे खाते में नहीं ले जाता। |
| `mobile.setup.account_plain` | Radient could not confirm the account and phone address. Check your internet connection and try again; nothing was changed. | Radient खाते और फ़ोन पते की पुष्टि नहीं कर सका। अपना इंटरनेट कनेक्शन जाँचें और दोबारा कोशिश करें; कुछ भी नहीं बदला गया। |
| `mobile.setup.cloudflared_plain` | Phone access needs a helper app called cloudflared, which is not installed on this computer. Ask your agent to install it (it asks for your approval first), then try again. | फ़ोन एक्सेस को cloudflared नाम के एक सहायक ऐप की ज़रूरत है, जो इस कंप्यूटर पर इंस्टॉल नहीं है। अपने एजेंट से इसे इंस्टॉल करवाएँ (वह पहले आपकी स्वीकृति माँगेगा), फिर दोबारा कोशिश करें। |
| `mobile.setup.mobile_bundle_plain` | The phone interface is not prepared on this computer yet. Ask your agent to prepare it (it asks for your approval first), then try again. Nothing was changed. | इस कंप्यूटर पर फ़ोन इंटरफ़ेस अभी तैयार नहीं है। अपने एजेंट से इसे तैयार करवाएँ (वह पहले आपकी स्वीकृति माँगेगा), फिर दोबारा कोशिश करें। कुछ भी नहीं बदला गया। |
| `mobile.setup.supervisor_plain` | This computer cannot keep phone access running in the background yet. Ask your agent to set up a background service for it, then try again. Nothing was changed. | यह कंप्यूटर अभी फ़ोन एक्सेस को बैकग्राउंड में चालू नहीं रख सकता। अपने एजेंट से इसके लिए बैकग्राउंड सेवा सेट करवाएँ, फिर दोबारा कोशिश करें। कुछ भी नहीं बदला गया। |
| `mobile.setup.apply_plain` | Everything is ready. Turn on phone access to finish. | सब कुछ तैयार है। पूरा करने के लिए फ़ोन एक्सेस चालू करें। |
| `mobile.setup.verification_plain` | The connection is not confirmed yet. Check again in a moment; completed steps stay saved. | कनेक्शन की अभी पुष्टि नहीं हुई है। थोड़ी देर में फिर से जाँचें; पूरे हो चुके चरण सहेजे रहते हैं। |
| `mobile.setup.failed_plain` | Phone access setup did not finish. Try again; completed steps stay saved. | फ़ोन एक्सेस सेटअप पूरा नहीं हुआ। दोबारा कोशिश करें; पूरे हो चुके चरण सहेजे रहते हैं। |
| `mobile.setup.complete_plain` | Open the address above on your phone and sign in with the same Radient account. | ऊपर दिया पता अपने फ़ोन पर खोलें और उसी Radient खाते से साइन इन करें। |
