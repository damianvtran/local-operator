# Independent translation audit — hi / tunnel.status / a3

Commit: 1fce7ecb1f5fb587e7b91ef2f9c7fe658521e0d0 (branch fix/tunnel-remedies-auto-up). Date: 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent, session 4aaeb39ddb2f; neither the i18n-translator (7af08d89b259) nor the English-drafting coder (b0300e2c2c7f); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 4aaeb39ddb2f: `Model: anthropic/claude-opus-5-5`; no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session 7af08d89b259 (commit 1fce7ecb1f5f); English source drafted by coder session b0300e2c2c7f; model anthropic/claude-opus-5-5 (as stated by the delegating brief; not independently verified); 2026-11-14.
Prior audit: i18n/audits/evidence/hi/tunnel.status/a2/record.json.

**Verdict: passed** — 0 unresolved blocker/major; 1 minor/nit finding(s).

Scope: the 8 keys new in 1fce7ecb1f5f judged in full (meaning, safety, placeholders, glossary, tone, length); the receipt covers the whole namespace (21 keys) at the new English source hash. Previously audited keys were re-read only for terminology consistency, not re-litigated.

Read: docs/I18N.md L260-308, i18n/glossary.md (no tunnel/billing/credit row — corpus usage governs), i18n/style-guides/hi.md, en + context sidecar, every new value in full, corpus term counts across all hi catalogues.

## Keys judged

| key | EN | hi | result |
|---|---|---|---|
| `tunnel_inactive.label` | tunnel not active | टनल सक्रिय नहीं | OK |
| `tunnel_inactive.detail` | Radient has paused this tunnel, but its billing is in order. Reconnect phone access in Local Operator on the computer running it; it also keeps retrying by itself. If it stays paused, review the tunnel at {url}. | Radient ने इस टनल को रोक दिया है, लेकिन इसकी बिलिंग ठीक है। जिस कंप्यूटर पर यह चल रहा है, उस पर Local Operator में फ़ोन एक्सेस फिर से जोड़ें; यह अपने-आप भी दोबारा कोशिश करता रहता है। अगर यह रुका ही रहे, तो {url} पर टनल की जाँच करें। | OK |
| `tunnel_disabled.label` | tunnel turned off in the console | कंसोल में टनल बंद है | OK |
| `tunnel_disabled.detail` | This tunnel is turned off in the Radient console, so remote access stays off until it is turned back on there: {url}. It reconnects by itself once it is on. | यह टनल Radient कंसोल में बंद है, इसलिए जब तक इसे वहाँ फिर से चालू नहीं किया जाता, रिमोट एक्सेस बंद रहेगा: {url}। चालू होते ही यह अपने-आप फिर से जुड़ जाता है। | OK |
| `billing_inactive.label` | billing not activated | बिलिंग सक्रिय नहीं | OK |
| `billing_inactive.detail` | Billing for this tunnel has not been activated, so Radient will not connect it. Open phone access setup in Local Operator on the computer running it and accept the monthly price. | इस टनल की बिलिंग सक्रिय नहीं की गई है, इसलिए Radient इसे कनेक्ट नहीं करेगा। जिस कंप्यूटर पर यह चल रहा है, उस पर Local Operator में फ़ोन एक्सेस सेटअप खोलें और मासिक मूल्य स्वीकार करें। | OK |
| `billing_lapsed.label` | billing lapsed | बिलिंग रुक गई | HI-01 |
| `billing_lapsed.detail` | This tunnel's billing lapsed because the Radient balance ran out, so Radient will not connect it. Add credit at {url}; it reconnects by itself once the balance is restored. | Radient बैलेंस ख़त्म होने से इस टनल की बिलिंग रुक गई है, इसलिए Radient इसे कनेक्ट नहीं करेगा। {url} पर क्रेडिट जोड़ें; बैलेंस बहाल होते ही यह अपने-आप फिर से जुड़ जाता है। | OK |

## Judgement

Polite आप register and polite imperatives (खोलें, जोड़ें, चलाएँ) throughout; danda । full stops; nukta kept (फ़ोन, ख़त्म). Terms consistent with audited corpus: फ़ोन एक्सेस (26 prior uses), टनल, बिलिंग, क्रेडिट, कंसोल, मासिक मूल्य, सेटअप, बैकग्राउंड, साइन-इन/साइन इन (noun/verb split as in corpus), 'reconnect' as फिर से जोड़ें (= audited mobile.setup.repoint). Safety meanings verified per key: no terminal command in any tunnel.status detail or mobile.setup value (only `/mobile enable` verbatim in login_offer); «कोई शुल्क नहीं लगता» (reconnect, reconnecting); «सेटअप कभी क्रेडिट नहीं खरीदता» (billing_lapsed, identical to audited mobile.setup.balance); «बिलिंग शुरू होने से पहले कोई भी मासिक मूल्य आपको दिखाया जाता है» (login_offer); «कभी बिलिंग सक्रिय नहीं करता और न ही क्रेडिट खर्च करता» (auto_setup.help); «सेटअप आपकी इस पसंद को नहीं बदलेगा» (disabled). network/radient_unavailable ask nothing of the reader («आपको कुछ नहीं करना है»; «कोई समस्या नहीं है»). {url} and the console URL byte-identical.

## Findings

| ID | severity | key | finding | suggested fix |
|---|---|---|---|---|
| HI-01 | minor | `tunnel.status.billing_lapsed.label` | «बिलिंग रुक गई» reads 'billing has stopped' — can be taken as benign (no longer being charged) rather than a lapse needing credit, and shares the root रुक with tunnel_inactive's «रोक दिया» (paused). Meaning not wrong; the detail sentence states the cause. | Optional: «बिलिंग रुकी (बैलेंस ख़त्म)» or «बैलेंस ख़त्म, बिलिंग रुकी». |

## Translator open questions (this locale)

- billing-lapsed label: acceptable as «बिलिंग रुक गई»; optional cause clarifier (HI-01).
- toggle label infinitive: grammatical, but inconsistent with the corpus polite-imperative toggle labels; recommend the imperative (HI-02).

## Mechanical

check.py --catalogues-only: 0 problems; key parity, placeholders, URLs, literal tokens, bidi controls, punctuation script and no-command checks OK (audit.log).

## Not verified

No native-speaker review and no rendered terminal/desktop/phone frame. The Reconnect action label the copy points at lives outside this catalogue and was not compared. Judgement by reasoning only; no MT service used.

This bundle is written in the shared worktree, uncommitted; generate.py was not run (coordinator regenerates).
