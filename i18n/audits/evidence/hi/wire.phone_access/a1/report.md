# Independent translation audit — hi / wire.phone_access / a1

Commit: 1fce7ecb1f5fb587e7b91ef2f9c7fe658521e0d0 (branch fix/tunnel-remedies-auto-up). Date: 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent, session 4aaeb39ddb2f; neither the i18n-translator (7af08d89b259) nor the English-drafting coder (b0300e2c2c7f); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 4aaeb39ddb2f: `Model: anthropic/claude-opus-5-5`; no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session 7af08d89b259 (commit 1fce7ecb1f5f); English source drafted by coder session b0300e2c2c7f; model anthropic/claude-opus-5-5 (as stated by the delegating brief; not independently verified); 2026-11-14.
Prior audit: none (new namespace).

**Verdict: passed** — 0 unresolved blocker/major; 3 minor/nit finding(s).

Scope: the 4 keys new in 1fce7ecb1f5f judged in full (meaning, safety, placeholders, glossary, tone, length); the receipt covers the whole namespace (4 keys) at the new English source hash. New namespace: all keys are new.

Read: docs/I18N.md L260-308, i18n/glossary.md (no tunnel/billing/credit row — corpus usage governs), i18n/style-guides/hi.md, en + context sidecar, every new value in full, corpus term counts across all hi catalogues.

## Keys judged

| key | EN | hi | result |
|---|---|---|---|
| `title` | Phone access | फ़ोन एक्सेस | OK |
| `description` | How this computer keeps its Radient tunnel for the phone app working. Changes apply to the next sign-in. | यह कंप्यूटर फ़ोन ऐप के लिए अपने Radient टनल को कैसे चालू रखता है। बदलाव अगले साइन-इन पर लागू होते हैं। | HI-05 |
| `auto_setup.label` | Reconnect phone access after signing in | साइन-इन के बाद फ़ोन एक्सेस फिर से जोड़ना | HI-02 |
| `auto_setup.help` | When you sign in to Radient on a computer that already has phone access, reconnect or repair its tunnel in the background. It never activates billing or spends credit; any price or missing credit is shown to you instead. | जब आप ऐसे कंप्यूटर पर Radient में साइन इन करते हैं जिस पर पहले से फ़ोन एक्सेस है, तो उसका टनल बैकग्राउंड में फिर से जोड़ा या ठीक किया जाता है। यह कभी बिलिंग सक्रिय नहीं करता और न ही क्रेडिट खर्च करता है; इसके बजाय कोई भी मूल्य या क्रेडिट की कमी आपको बताई जाती है। | HI-03 |

## Judgement

Polite आप register and polite imperatives (खोलें, जोड़ें, चलाएँ) throughout; danda । full stops; nukta kept (फ़ोन, ख़त्म). Terms consistent with audited corpus: फ़ोन एक्सेस (26 prior uses), टनल, बिलिंग, क्रेडिट, कंसोल, मासिक मूल्य, सेटअप, बैकग्राउंड, साइन-इन/साइन इन (noun/verb split as in corpus), 'reconnect' as फिर से जोड़ें (= audited mobile.setup.repoint). Safety meanings verified per key: no terminal command in any tunnel.status detail or mobile.setup value (only `/mobile enable` verbatim in login_offer); «कोई शुल्क नहीं लगता» (reconnect, reconnecting); «सेटअप कभी क्रेडिट नहीं खरीदता» (billing_lapsed, identical to audited mobile.setup.balance); «बिलिंग शुरू होने से पहले कोई भी मासिक मूल्य आपको दिखाया जाता है» (login_offer); «कभी बिलिंग सक्रिय नहीं करता और न ही क्रेडिट खर्च करता» (auto_setup.help); «सेटअप आपकी इस पसंद को नहीं बदलेगा» (disabled). network/radient_unavailable ask nothing of the reader («आपको कुछ नहीं करना है»; «कोई समस्या नहीं है»). {url} and the console URL byte-identical.

## Findings

| ID | severity | key | finding | suggested fix |
|---|---|---|---|---|
| HI-02 | minor | `wire.phone_access.auto_setup.label` | Infinitive «…फिर से जोड़ना» is grammatical but breaks the corpus toggle-label convention: wire.settings boolean labels use the polite imperative («बातचीत अपने-आप सहेजें», «/resume के बाद भी काम जारी रखें»). Answers the translator's open question. | «साइन-इन के बाद फ़ोन एक्सेस फिर से जोड़ें». |
| HI-03 | nit | `wire.phone_access.auto_setup.help` | «यह कभी बिलिंग सक्रिय नहीं करता…» follows a passive sentence whose subject is the tunnel, so «यह» can be read as the tunnel; the safety meaning (never activates billing / never spends credit) survives either way. | «यह प्रक्रिया कभी बिलिंग सक्रिय नहीं करती और न ही क्रेडिट ख़र्च करती है…» |
| HI-05 | nit | `wire.phone_access.description` | Stand-alone indirect question «यह कंप्यूटर … कैसे चालू रखता है।» mirrors the English fragment; acceptable as a section description. | Optional: «फ़ोन ऐप के लिए यह कंप्यूटर अपना Radient टनल कैसे चालू रखे।» |

## Translator open questions (this locale)

- billing-lapsed label: acceptable as «बिलिंग रुक गई»; optional cause clarifier (HI-01).
- toggle label infinitive: grammatical, but inconsistent with the corpus polite-imperative toggle labels; recommend the imperative (HI-02).

## Mechanical

check.py --catalogues-only: 0 problems; key parity, placeholders, URLs, literal tokens, bidi controls, punctuation script and no-command checks OK (audit.log).

## Not verified

No native-speaker review and no rendered terminal/desktop/phone frame. The Reconnect action label the copy points at lives outside this catalogue and was not compared. Judgement by reasoning only; no MT service used.

This bundle is written in the shared worktree, uncommitted; generate.py was not run (coordinator regenerates).
