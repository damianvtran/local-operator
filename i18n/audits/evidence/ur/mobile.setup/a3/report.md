# Independent translation audit — ur / mobile.setup — a3

- PR: #3958 (branch fix/tunnel-remedy-no-commands, stacked on #3945), head 01113e5aa048884a507d9bb4868815f909f63a48; scope `git diff 374037856..01113e5aa -- local_operator/i18n/` (12 new `*_plain` keys; 0 existing keys changed).
- Translator: coder subagent of host Local Operator session 25fe8dfbf610 (values committed in 01113e5aa on fix/tunnel-remedy-no-commands; English source by the same coder); model not disclosed to the auditor (recorded null).
- Auditor: translation-auditor (lopdev team), independent Local Operator session 9054cd229d3f; not the translator/coder of host session 25fe8dfbf610; edited no catalogue value; model anthropic/claude-opus-5-5 (session system prompt; no provider fallback during the audit).
- Date: 2026-11-14. Result: **PASSED** — 0 blocker / 0 major; 2 minor/nit finding(s), none gating.

## Method
Read side by side with en source + context sidecar (12 new keys, 0 changed); glossary agent/approvals/service and the namespace's phone-access, phone-address, sign-in, account and setup terms match #3945's shipped copy; register per style guide; no placeholders in source or target; no command, flag, slash or code-span text (setup._SHELL_COMMAND + slash/flag/backtick scan); NFC; no bidi controls; no maxCells on these keys — the TUI's 50-cell park card renders tunnel.status card_* keys, not these; the *_plain keys reach the TUI only through setup.text(surface='tui') in a wrapping system notice and the desktop's remedy.text. Runtime: setup.plain_remedy({'step': s}) equals the catalogue value for all 12 steps.

## Locale notes
آپ register; Urdu ۔ ، ؛ only, no Hindi danda or Latin punctuation; parentheses in logical order; Latin runs (Radient, cloudflared) untouched; account_plain opens with a Latin run like 10 already-passed keys in this namespace (render layer FSI-isolates; web/Electron dir=rtl).

## Findings
- UR3-01 minor (account_plain): 'Radient اکاؤنٹ اور فون ایڈریس…' — 'Radient اکاؤنٹ' reads as the compound 'Radient account' (cf. mobile.setup.account 'Radient اکاؤنٹ:'), losing Radient as subject. Proposed: 'Radient آپ کے اکاؤنٹ اور فون ایڈریس کی تصدیق نہیں کر سکا۔ اپنا انٹرنیٹ کنکشن چیک کریں اور دوبارہ کوشش کریں؛ کچھ بھی تبدیل نہیں کیا گیا۔'
- UR3-02 nit (stopped_plain): 'اس کمپیوٹر تک پہنچنا چاہیں' — the namespace's login_offer says 'اس کمپیوٹر تک رسائی حاصل کریں'; optional: 'جب بھی آپ اپنے فون سے اس کمپیوٹر تک رسائی حاصل کرنا چاہیں، اسے دوبارہ آن کریں۔'

## Keys read (12)
| key | en | ur |
|---|---|---|
| `mobile.setup.stopped_plain` | Phone access is turned off on this computer. Turn it back on whenever you want to reach this computer from your phone. | اس کمپیوٹر پر فون رسائی بند ہے۔ جب بھی آپ اپنے فون سے اس کمپیوٹر تک پہنچنا چاہیں، اسے دوبارہ آن کریں۔ |
| `mobile.setup.owner_missing_plain` | The Radient account that set up phone access is no longer signed in on this computer. Sign in with that account to bring phone access back. | جس Radient اکاؤنٹ سے فون رسائی سیٹ اپ کی گئی تھی، وہ اب اس کمپیوٹر پر سائن اِن نہیں ہے۔ فون رسائی واپس لانے کے لیے اسی اکاؤنٹ سے سائن اِن کریں۔ |
| `mobile.setup.login_plain` | Sign in to Radient with the account that owns phone access, then try again. | فون رسائی کے مالک اکاؤنٹ سے Radient میں سائن اِن کریں، پھر دوبارہ کوشش کریں۔ |
| `mobile.setup.choose_account_plain` | Choose the Radient account phone access should use. Setup never moves an existing phone address to a different account. | منتخب کریں کہ فون رسائی کون سا Radient اکاؤنٹ استعمال کرے۔ سیٹ اپ کسی موجودہ فون ایڈریس کو کبھی دوسرے اکاؤنٹ میں منتقل نہیں کرتا۔ |
| `mobile.setup.account_plain` | Radient could not confirm the account and phone address. Check your internet connection and try again; nothing was changed. | Radient اکاؤنٹ اور فون ایڈریس کی تصدیق نہیں کر سکا۔ اپنا انٹرنیٹ کنکشن چیک کریں اور دوبارہ کوشش کریں؛ کچھ بھی تبدیل نہیں کیا گیا۔ |
| `mobile.setup.cloudflared_plain` | Phone access needs a helper app called cloudflared, which is not installed on this computer. Ask your agent to install it (it asks for your approval first), then try again. | فون رسائی کو cloudflared نامی ایک معاون ایپ درکار ہے، جو اس کمپیوٹر پر انسٹال نہیں ہے۔ اپنے ایجنٹ سے اسے انسٹال کروائیں (وہ پہلے آپ کی منظوری مانگے گا)، پھر دوبارہ کوشش کریں۔ |
| `mobile.setup.mobile_bundle_plain` | The phone interface is not prepared on this computer yet. Ask your agent to prepare it (it asks for your approval first), then try again. Nothing was changed. | اس کمپیوٹر پر فون انٹرفیس ابھی تیار نہیں ہے۔ اپنے ایجنٹ سے اسے تیار کروائیں (وہ پہلے آپ کی منظوری مانگے گا)، پھر دوبارہ کوشش کریں۔ کچھ بھی تبدیل نہیں کیا گیا۔ |
| `mobile.setup.supervisor_plain` | This computer cannot keep phone access running in the background yet. Ask your agent to set up a background service for it, then try again. Nothing was changed. | یہ کمپیوٹر ابھی فون رسائی کو بیک گراؤنڈ میں چالو نہیں رکھ سکتا۔ اپنے ایجنٹ سے اس کے لیے بیک گراؤنڈ سروس سیٹ اپ کروائیں، پھر دوبارہ کوشش کریں۔ کچھ بھی تبدیل نہیں کیا گیا۔ |
| `mobile.setup.apply_plain` | Everything is ready. Turn on phone access to finish. | سب کچھ تیار ہے۔ مکمل کرنے کے لیے فون رسائی آن کریں۔ |
| `mobile.setup.verification_plain` | The connection is not confirmed yet. Check again in a moment; completed steps stay saved. | کنکشن کی ابھی تصدیق نہیں ہوئی۔ تھوڑی دیر میں دوبارہ چیک کریں؛ مکمل شدہ مراحل محفوظ رہتے ہیں۔ |
| `mobile.setup.failed_plain` | Phone access setup did not finish. Try again; completed steps stay saved. | فون رسائی کا سیٹ اپ مکمل نہیں ہوا۔ دوبارہ کوشش کریں؛ مکمل شدہ مراحل محفوظ رہتے ہیں۔ |
| `mobile.setup.complete_plain` | Open the address above on your phone and sign in with the same Radient account. | اوپر دیا گیا ایڈریس اپنے فون پر کھولیں اور اسی Radient اکاؤنٹ سے سائن اِن کریں۔ |
