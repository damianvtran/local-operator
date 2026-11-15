# Independent translation audit — hi / mobile.setup / a1

Commit: 1fce7ecb1f5fb587e7b91ef2f9c7fe658521e0d0 (branch fix/tunnel-remedies-auto-up). Date: 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent, session 4aaeb39ddb2f; neither the i18n-translator (7af08d89b259) nor the English-drafting coder (b0300e2c2c7f); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 4aaeb39ddb2f: `Model: anthropic/claude-opus-5-5`; no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session 7af08d89b259 (commit 1fce7ecb1f5f); English source drafted by coder session b0300e2c2c7f; model anthropic/claude-opus-5-5 (as stated by the delegating brief; not independently verified); 2026-11-14.
Prior audit: i18n/audits/evidence/hi/mobile.setup/r1/record.json.

**Verdict: passed** — 0 unresolved blocker/major; 1 minor/nit finding(s).

Scope: the 7 keys new in 1fce7ecb1f5f judged in full (meaning, safety, placeholders, glossary, tone, length); the receipt covers the whole namespace (86 keys) at the new English source hash. Previously audited keys were re-read only for terminology consistency, not re-litigated.

Read: docs/I18N.md L260-308, i18n/glossary.md (no tunnel/billing/credit row — corpus usage governs), i18n/style-guides/hi.md, en + context sidecar, every new value in full, corpus term counts across all hi catalogues.

## Keys judged

| key | EN | hi | result |
|---|---|---|---|
| `disabled` | This tunnel is turned off in the Radient console. Turn it back on at https://console.radienthq.com/dashboard/tunnels; it reconnects by itself, and setup will not override that choice. | यह टनल Radient कंसोल में बंद है। इसे https://console.radienthq.com/dashboard/tunnels पर फिर से चालू करें; यह अपने-आप फिर से जुड़ जाता है, और सेटअप आपकी इस पसंद को नहीं बदलेगा। | OK |
| `billing_lapsed` | This tunnel's billing lapsed because the Radient balance ran out. Add credit at https://console.radienthq.com/dashboard/tunnels; setup never purchases credit, and the tunnel reconnects by itself once the balance is restored. | Radient बैलेंस ख़त्म होने से इस टनल की बिलिंग रुक गई है। https://console.radienthq.com/dashboard/tunnels पर क्रेडिट जोड़ें; सेटअप कभी क्रेडिट नहीं खरीदता, और बैलेंस बहाल होते ही टनल अपने-आप फिर से जुड़ जाता है। | OK |
| `reconnect` | Radient has paused this tunnel, but its billing is in order. Reconnect to bring phone access back; nothing is charged. | Radient ने इस टनल को रोक दिया है, लेकिन इसकी बिलिंग ठीक है। फ़ोन एक्सेस वापस लाने के लिए फिर से जोड़ें; कोई शुल्क नहीं लगता। | OK |
| `network` | This computer could not reach Radient (its network may have changed or dropped). Phone access reconnects by itself once the connection is back; nothing needs doing. | यह कंप्यूटर Radient तक नहीं पहुँच सका (शायद इसका नेटवर्क बदल गया है या टूट गया है)। कनेक्शन लौटते ही फ़ोन एक्सेस अपने-आप फिर से जुड़ जाता है; आपको कुछ नहीं करना है। | HI-04 |
| `radient_unavailable` | Radient answered with a temporary server error. Nothing is wrong with this computer or its sign-in; phone access reconnects by itself once Radient recovers. | Radient ने एक अस्थायी सर्वर त्रुटि लौटाई। इस कंप्यूटर या इसके साइन-इन में कोई समस्या नहीं है; Radient के ठीक होते ही फ़ोन एक्सेस अपने-आप फिर से जुड़ जाता है। | OK |
| `reconnecting` | Phone access is reconnecting in the background after your sign-in. Nothing is charged; this clears by itself in a minute or two. | आपके साइन-इन के बाद फ़ोन एक्सेस बैकग्राउंड में फिर से जुड़ रहा है। कोई शुल्क नहीं लगता; यह एक-दो मिनट में अपने-आप ठीक हो जाएगा। | OK |
| `login_offer` | Reach this computer from the phone app: run /mobile enable. Any monthly price is shown to you before billing starts. | फ़ोन ऐप से इस कंप्यूटर तक पहुँचें: /mobile enable चलाएँ। बिलिंग शुरू होने से पहले कोई भी मासिक मूल्य आपको दिखाया जाता है। | OK |

## Judgement

Polite आप register and polite imperatives (खोलें, जोड़ें, चलाएँ) throughout; danda । full stops; nukta kept (फ़ोन, ख़त्म). Terms consistent with audited corpus: फ़ोन एक्सेस (26 prior uses), टनल, बिलिंग, क्रेडिट, कंसोल, मासिक मूल्य, सेटअप, बैकग्राउंड, साइन-इन/साइन इन (noun/verb split as in corpus), 'reconnect' as फिर से जोड़ें (= audited mobile.setup.repoint). Safety meanings verified per key: no terminal command in any tunnel.status detail or mobile.setup value (only `/mobile enable` verbatim in login_offer); «कोई शुल्क नहीं लगता» (reconnect, reconnecting); «सेटअप कभी क्रेडिट नहीं खरीदता» (billing_lapsed, identical to audited mobile.setup.balance); «बिलिंग शुरू होने से पहले कोई भी मासिक मूल्य आपको दिखाया जाता है» (login_offer); «कभी बिलिंग सक्रिय नहीं करता और न ही क्रेडिट खर्च करता» (auto_setup.help); «सेटअप आपकी इस पसंद को नहीं बदलेगा» (disabled). network/radient_unavailable ask nothing of the reader («आपको कुछ नहीं करना है»; «कोई समस्या नहीं है»). {url} and the console URL byte-identical.

## Findings

| ID | severity | key | finding | suggested fix |
|---|---|---|---|---|
| HI-04 | nit | `mobile.setup.network` | «नेटवर्क ... टूट गया है» — 'टूटना' collocates with कनेक्शन, slightly odd with नेटवर्क. | «...नेटवर्क बदल गया है या बंद हो गया है». |

## Translator open questions (this locale)

- billing-lapsed label: acceptable as «बिलिंग रुक गई»; optional cause clarifier (HI-01).
- toggle label infinitive: grammatical, but inconsistent with the corpus polite-imperative toggle labels; recommend the imperative (HI-02).

## Mechanical

check.py --catalogues-only: 0 problems; key parity, placeholders, URLs, literal tokens, bidi controls, punctuation script and no-command checks OK (audit.log).

## Not verified

No native-speaker review and no rendered terminal/desktop/phone frame. The Reconnect action label the copy points at lives outside this catalogue and was not compared. Judgement by reasoning only; no MT service used.

This bundle is written in the shared worktree, uncommitted; generate.py was not run (coordinator regenerates).
