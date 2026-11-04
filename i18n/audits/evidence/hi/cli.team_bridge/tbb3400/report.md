# hi cli.team_bridge audit tbb3400 (PR #3400)

Scope: the two added keys `notify.budget_exhausted_title` and `notify.budget_exhausted_body` at PR head `5456b8db1d017c421c7632324ab2065e64e8b85e`, read against the en source, `cli.team_bridge.context.json` notes (budget used up so new agent messages wait; nothing lost, screened automatically as the budget refills; people and messages addressed to the agent screened first; no implication of rejection, deletion or notification), the hi style guide, `i18n/glossary.md`, and the namespace's existing renderings. The other 55 keys are byte-unchanged since the PR #3198 passed row and were not re-audited.

## Translator values (as committed at head)

- title: एजेंट संदेशों की जाँच अभी रुकी है
- body: आपके एजेंट ने इस घंटे का जाँच बजट पूरा कर लिया है, इसलिए नए एजेंट संदेश प्रतीक्षा में हैं। बजट फिर से भरते ही उनकी जाँच अपने-आप हो जाएगी। लोगों के भेजे संदेश और आपके एजेंट को संबोधित संदेश पहले जाँचे जाते हैं।

## Findings and dispositions (all fixed by this auditor in the audit commit)

- **H1 MAJOR** (title+body) — current: `जाँच`. In this namespace जाँच is the inbox/readiness 'check' (status.last_check, setup.failed); screening is स्क्रीनिंग (status.screening). Using जाँच makes the notice read as 'inbox check stopped'. Fix: स्क्रीनिंग in title and body.
- **H2 MINOR** (body) — current: `बजट पूरा कर लिया / भरते ही`. 'पूरा कर लिया' reads as 'met/completed the budget'; 'भरते ही' implies all at once rather than as it refills. Fix: 'स्क्रीनिंग बजट पूरा इस्तेमाल कर लिया'; 'बजट फिर से भरने पर'.

## Shipped values after remediation

- title: एजेंट संदेशों की स्क्रीनिंग अभी रुकी है
- body: आपके एजेंट ने इस घंटे का स्क्रीनिंग बजट पूरा इस्तेमाल कर लिया है, इसलिए नए एजेंट संदेश प्रतीक्षा में हैं। बजट फिर से भरने पर उनकी स्क्रीनिंग अपने-आप हो जाएगी। लोगों के भेजे संदेशों और आपके एजेंट को संबोधित संदेशों की स्क्रीनिंग पहले होती है।

Accuracy: all three context facts survive; no rejection, deletion or notification implied. Unresolved blocker/major: 0.

## Mechanical

Key parity 57/57; no placeholders, ICU or markup in either key (none in source); NFC; no Cf/bidi control characters; no code tokens. No `maxChars`/`maxCells` hint exists for these keys; both are inbox-notification prose, not width-bound terminal cells.

## Negative control

At head 5456b8db (rows absent) the generator's `_shipped_locales()` returned `['en']`: the stale `cli.team_bridge` hash would have dropped all seven locales. After these rows it returns all eight (see audit.log).

## Limitations

Independent LLM review, not native-speaker certification; no MT or translation service used. Scalar text only: no rendered inbox, terminal, phone or RTL paint geometry was exercised. The translator's model is not named in PR #3400 and is recorded as unrecorded, not inferred.
